from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate
from typing import Dict, List
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
import copy
from evaluate_dataset import RougeCallback, format_prompt
import time
from transformers import TrainerCallback




def formatting_func(example: Dict[str, str]) -> str:
    return (
        format_prompt(example["document"]) +
        f"{example['summary']}"
    )

def fine_tune_with_lora(model: AutoModelForCausalLM, dataset: Dataset, tokenizer: AutoTokenizer, finetune_name: str, rank: int, epoch: int, max_length: int) -> str:
    print(formatting_func(dataset["train"][0]))


    sft_config = SFTConfig(
        output_dir=finetune_name,
        num_train_epochs=epoch,             # Liczba epok treningu
        per_device_train_batch_size=16,  # Rozmiar wsadu per GPU
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        #optim="adafactor",             # Optymalizator AdaFactor
        optim="adamw_torch_fused",      # Efektywna wersja optymalizatora AdamW
        learning_rate=2e-4,             # Stopa uczenia
        max_grad_norm=0.3,              # Ograniczenie gradientu
        warmup_ratio=0.03,              # warm-up
        lr_scheduler_type="constant", #"cosine",   # Stała stopa uczenia po początkowym okresie warm-up
        logging_steps=100,               # Co ile kroków logować wartość metryk
        eval_strategy="steps",          # Ewaluacja po wykonaniu określonej liczby kroków
        eval_steps=100,                  # Częstotliwość ewaluacji
        bf16=True,                     # Precyzja bfloat16 wymaga architektury Ampere
        ddp_find_unused_parameters=True,  # DDP dla multi-GPU
        gradient_checkpointing=True, 
        save_strategy="steps",
        save_steps=1000,               
        load_best_model_at_end=True,
        report_to="wandb",

        max_length=max_length,
    )

    peft_config = LoraConfig(
        r=rank,                    # Rank dimension - typically between 4-32
        lora_alpha=rank,           # LoRA scaling factor
        lora_dropout=0.05,      # Dropout probability for LoRA layers
        bias="none",
        #target_modules=["in_proj", "x_proj", "dt_proj", "embeddings"],
        #target_modules=["q_a_proj", "q_b_proj", "kv_proj", "o_a_proj", "o_b_proj"],
        task_type="CAUSAL_LM",  # Task type for model architecture
        ensure_weight_tying=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_func,
        processing_class=tokenizer,
    )
    trainer.add_callback(
        EpochTimeCallback()
    )

    trainer.add_callback(
        RougeCallback(
            tokenizer,
            dataset["validation"]
        )
    )

    peft_model = get_peft_model(copy.deepcopy(model), peft_config)
    peft_model.print_trainable_parameters()

    trainer.train()
    return trainer.state.best_model_checkpoint
    


class EpochTimeCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        end_time = time.time()
        epoch_time = end_time - self.start_time

        print(f"\nEpoch {int(state.epoch)} time: {epoch_time:.2f} seconds")

        if args.report_to == "wandb":
            import wandb
            wandb.log(
                {"time/epoch_seconds": epoch_time},
                step=state.global_step
            )