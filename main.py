from datasets import load_dataset
from transformers import AutoModel


def main():
    print("Hello from llm-mamba!")
    dataset = load_dataset("EdinburghNLP/xsum")
    print(dataset)
    print(dataset.keys())
    model_name = "state-spaces/mamba-130m-hf"
    model = AutoModel.from_pretrained(model_name, dtype="auto", trust_remote_code=True)
    print(model)
    print(f"Parameters: {model.num_parameters()}")


if __name__ == "__main__":
    main()
