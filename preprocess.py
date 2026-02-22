import os
import yaml
import json
from datasets import load_dataset
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_function(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": example["input"].strip()
            },
            {
                "role": "assistant",
                "content": example["label"].strip()
            }
        ]
    }


def main():
    config = load_config()
    
    input_file = config["data"]["train_path"]
    
    print(f"\n Arquivo de entrada: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")
    
    dataset = load_dataset(
        "json",
        data_files=input_file,
        field="dataset",
        split="train"
    )
    
    #pré-processamento
    processed_dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        desc="Preprocessando"
    )
    
    print(f"Total de exemplos: {len(processed_dataset)}")
        
    return processed_dataset


if __name__ == "__main__":
    main()
