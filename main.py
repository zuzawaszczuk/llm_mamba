from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_dataset import score_model
import torch
from lora_peft import fine_tune_with_lora
import torch.distributed as dist
from peft import PeftModel
import wandb
from dotenv import load_dotenv
import os
import argparse


def ddp_setup(finetune_name: str):
    dist.init_process_group(backend="nccl")

    is_main_process = dist.get_rank() == 0
    if not is_main_process:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        wandb.login()
        wandb.init(
            project="mamba-finetuning",
            name=finetune_name
        )


def parse():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA")
    parser.add_argument("--model", type=str)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--ddp_setup", type=bool)
    parser.add_argument("--max_length", type=int, default=10000)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse()
    finetune_name = f"{args.model}-r{args.rank}-e{args.epoch}" 
    
    if args.ddp_setup:
        print("Setting up DDP...")
        ddp_setup(finetune_name)

    dataset = load_dataset("EdinburghNLP/xsum")
    dataset["train"] = dataset["train"].select(range(1000))
    dataset["train"] = dataset["train"].select(range(1000))
    dataset["train"] = dataset["train"].select(range(1000))
    print(dataset.keys())

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=None)
    model.config.use_cache = False
    print(model)
    print(f"Using device: {model.device}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    
    rouge_score = score_model(model, tokenizer, dataset["validation"], 32)
    print("ROUGE Score:", rouge_score)

    best_ckpt = fine_tune_with_lora(model, dataset, tokenizer, finetune_name, args.rank, args.epoch, args.max_length)

    print(f"Best checkpoint path: {best_ckpt}")
    peft_model = PeftModel.from_pretrained(model, best_ckpt)
    
    rouge_score = score_model(peft_model, tokenizer, dataset["validation"], 32)
    print("ROUGE Score after fine-tuning:", rouge_score)


if __name__ == "__main__":
    main()