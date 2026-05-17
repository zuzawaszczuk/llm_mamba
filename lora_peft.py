from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate
from typing import Dict, List
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
import copy

def formatting_func(example):
    return example["document"]


def fine_tune_with_lora(model: AutoModelForCausalLM, dataset: Dataset, tokenizer: AutoTokenizer, finetune_name: str):
    sft_config = SFTConfig(
        output_dir=finetune_name,
        num_train_epochs=3,             # Liczba epok treningu
        per_device_train_batch_size= 8,  # Rozmiar wsadu per GPU
        per_device_eval_batch_size = 8,
        gradient_accumulation_steps = 2,
        #optim="adafactor",             # Optymalizator AdaFactor
        optim="adamw_torch_fused",      # Efektywna wersja optymalizatora AdamW
        learning_rate=2e-4,             # Stopa uczenia
        max_grad_norm=0.3,              # Ograniczenie gradientu
        warmup_ratio=0.03,              # warm-up
        lr_scheduler_type="constant",   # Stała stopa uczenia po początkowym okresie warm-up
        logging_steps=10,               # Co ile kroków logować wartość metryk
        eval_strategy="steps",          # Ewaluacja po wykonaniu określonej liczby kroków
        eval_steps=100,                  # Częstotliwość ewaluacji
        bf16=True,                     # Precyzja bfloat16 wymaga architektury Ampere
        ddp_find_unused_parameters=False,  # DDP dla multi-GPU
        gradient_checkpointing=True, 
        dataset_num_proc=4,
    )

    peft_config = LoraConfig(
        r=8,                    # Rank dimension - typically between 4-32
        lora_alpha=8,           # LoRA scaling factor
        lora_dropout=0.05,      # Dropout probability for LoRA layers
        bias="none",
        target_modules=["in_proj",], #"x_proj", "dt_proj"],  # Nazwy modułów, które mają być dostrojone za pomocą LoRA
        modules_to_save=["lm_head", "embeddings"],
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