from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate
from typing import Dict, List
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
import copy

def format_prompt(document: str) -> str:
    return (
        "### Instruction:\n"
        "Summarize the following text.\n\n"
        f"### Input:\n{document}\n\n"
        "### Response:\n"
    )

def formatting_func(example: Dict[str, str]) -> str:
    return (
        format_prompt(example["document"]) +
        f"{example['summary']}"
    )

def fine_tune_with_lora(model: AutoModelForCausalLM, dataset: Dataset, tokenizer: AutoTokenizer, finetune_name: str):
    print(formatting_func(dataset["train"][0]))


    sft_config = SFTConfig(
        output_dir=finetune_name,
        num_train_epochs=1,             # Liczba epok treningu
        per_device_train_batch_size= 16,  # Rozmiar wsadu per GPU
        per_device_eval_batch_size = 16,
        gradient_accumulation_steps = 2,
        #optim="adafactor",             # Optymalizator AdaFactor
        optim="adamw_torch_fused",      # Efektywna wersja optymalizatora AdamW
        learning_rate=2e-4,             # Stopa uczenia
        max_grad_norm=0.3,              # Ograniczenie gradientu
        warmup_ratio=0.03,              # warm-up
        lr_scheduler_type="constant",   # Stała stopa uczenia po początkowym okresie warm-up
        logging_steps=100,               # Co ile kroków logować wartość metryk
        eval_strategy="steps",          # Ewaluacja po wykonaniu określonej liczby kroków
        eval_steps=100,                  # Częstotliwość ewaluacji
        bf16=True,                     # Precyzja bfloat16 wymaga architektury Ampere
        ddp_find_unused_parameters=True,  # DDP dla multi-GPU
        gradient_checkpointing=True, 
        dataset_num_proc=2,
        save_strategy="steps",
        save_steps=1000,               
        load_best_model_at_end=True,
    )

    peft_config = LoraConfig(
        r=8,                    # Rank dimension - typically between 4-32
        lora_alpha=8,           # LoRA scaling factor
        lora_dropout=0.05,      # Dropout probability for LoRA layers
        bias="none",
        target_modules=["in_proj", "x_proj", "dt_proj"],  # Nazwy modułów, które mają być dostrojone za pomocą LoRA
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

    peft_model = get_peft_model(copy.deepcopy(model), peft_config)
    peft_model.print_trainable_parameters()

    trainer.train()
    
    trainer.save_model(f"./{finetune_name}")
    return trainer.state.best_model_checkpoint
    