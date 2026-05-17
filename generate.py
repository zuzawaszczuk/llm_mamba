from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    dataset = load_dataset("EdinburghNLP/xsum")
    model_name = "state-spaces/mamba-130m-hf"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for example in dataset["train"].select(range(5)):
        prompt = example["document"]
        answer = generate_answer(model, tokenizer, prompt)
        print(f"Ground Truth: {example['summary']}")
        print(f"Answer: {answer}")
        print("-" * 50)


def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=30
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]

    return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    main()



