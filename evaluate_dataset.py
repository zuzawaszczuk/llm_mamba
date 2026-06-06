from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import evaluate
from typing import Dict, List
from tqdm import tqdm
from transformers import TrainerCallback
import wandb


def format_prompt(document: str) -> str:
    return (
        "### Instruction:\n"
        "Summarize the following text.\n\n"
        f"### Input:\n{document}\n\n"
        "### Response:\n"
    )


def score_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, batch_size: int = 8) -> Dict[str, float]:
    rouge = evaluate.load("rouge")
    generated_summary = []
    
    documents = dataset["document"]
    
    # Process in batches
    for i in tqdm(range(0, len(documents), batch_size)):
        prompts = documents[i:i + batch_size]

        batch_prompts = [format_prompt(doc) for doc in prompts]
        batch_predictions = generate_answers_batch(model, tokenizer, batch_prompts)
        generated_summary.extend(batch_predictions)

    return rouge.compute(predictions=generated_summary, references=list(dataset["summary"]))


def generate_answers_batch(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompts: List[str]) -> List[str]:
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40
        )
   
    decoded = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    predictions = [
        text.split("### Response:\n")[-1].strip()
        for text in decoded
    ]

    return predictions


class RougeCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]

        scores = score_model(
            model=model,
            tokenizer=self.tokenizer,
            dataset=self.eval_dataset,
            batch_size=64,
        )

        print(f"\nEpoch {state.epoch}")
        print(scores)

        if "wandb" in args.report_to:
            wandb.log(
                {f"eval/{k}": v for k, v in scores.items()},
                step=state.global_step,
            )