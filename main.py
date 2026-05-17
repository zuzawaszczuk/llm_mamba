from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_dataset import score_model
import torch
from lora_peft import fine_tune_with_lora
import torch.distributed as dist


def main():
    dist.init_process_group(backend="gloo|nccl")
    dataset = load_dataset("EdinburghNLP/xsum")
    print(dataset.keys())
    model_name = "state-spaces/mamba-130m-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    print(model)
    print(f"Using device: {model.device}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    
    #rouge_score = score_model(model, tokenizer, dataset["validation"], 16)
    #print("ROUGE Score:", rouge_score)

    finetune_name = "Lora-FT"      # Nazwa pod jaką zostanie zapisanny model
    fine_tune_with_lora(model, dataset, tokenizer, finetune_name)

    model = AutoModelForCausalLM.from_pretrained(f"./{finetune_name}", dtype=torch.bfloat16, device_map="auto")
    rouge_score = score_model(model, tokenizer, dataset["validation"], 16)


if __name__ == "__main__":
    main()