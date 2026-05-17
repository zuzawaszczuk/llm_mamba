from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate
from typing import Dict, List
from tqdm import tqdm


def score_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, batch_size: int = 8) -> Dict[str, float]:
    rouge = evaluate.load("rouge")
    generated_summary = []
    
    documents = dataset["document"]
    
    # Process in batches
    for i in tqdm(range(0, len(documents), batch_size)):
        batch_prompts = documents[i:i + batch_size]
        batch_predictions = generate_answers_batch(model, tokenizer, batch_prompts)
        generated_summary.extend(batch_predictions)

    return rouge.compute(predictions=generated_summary, references=list(dataset["summary"]))


def generate_answers_batch(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: List[str]) -> List[str]:
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]

    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    return predictions
