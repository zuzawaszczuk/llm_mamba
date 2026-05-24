from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_dataset import score_model
import torch
from lora_peft import fine_tune_with_lora
import torch.distributed as dist
from peft import PeftModel
import wandb
from dotenv import load_dotenv


def main():
    load_dotenv()
    wandb.login()
    wandb.init(project="mamba-finetuning", name="mamba-130m-lora-finetune")
    dist.init_process_group(backend="nccl")
    dataset = load_dataset("EdinburghNLP/xsum")
    #dataset['validation'] = dataset['validation'].shuffle(42).select(range(100))
    #dataset['test'] = dataset['test'].shuffle(42).select(range(1000))

    print(dataset.keys())
    model_name = "state-spaces/mamba-130m-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    print(model)
    print(f"Using device: {model.device}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    
    rouge_score = score_model(model, tokenizer, dataset["validation"], 16)
    print("ROUGE Score:", rouge_score)

    finetune_name = "Lora-FT" 
    best_ckpt = fine_tune_with_lora(model, dataset, tokenizer, finetune_name)

    print(f"Best checkpoint path: {best_ckpt}")
    peft_model = PeftModel.from_pretrained(model, best_ckpt)
    
    rouge_score = score_model(peft_model, tokenizer, dataset["validation"], 16)
    print("ROUGE Score after fine-tuning:", rouge_score)


if __name__ == "__main__":
    main()