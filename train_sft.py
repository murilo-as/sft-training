import os
import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

#pegar as configurações do arquivo yaml
def load_config(path="config/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()

#modelo, dados e saída
model_id = cfg["model"]["id"]
data_path = cfg["data"]["train_path"]
output_dir = cfg["output"]["dir"]

print(f"  - Modelo: {model_id}")
print(f"  - Dataset: {data_path}")
print(f"  - Output: {output_dir}")

#configs quantização   
bnb_config = BitsAndBytesConfig(
    load_in_4bit=cfg["quantization"]["load_in_4bit"],
    bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
    bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=torch.bfloat16 if cfg["quantization"]["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16,
)

#tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=cfg["model"]["device_map"],
)
model.config.use_cache = False

print(f"\n Carregando dataset: {data_path}")
dataset = load_dataset("json", data_files=data_path, split="train")
print(f"Dataset carregado: {len(dataset)} exemplos")

#convertendo para formato de mensagens
def convert_to_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["completion"]}
        ]
    }

dataset = dataset.map(convert_to_messages, remove_columns=dataset.column_names)

#formatando os dados para o modelo
def formatting_func(example):
    if tokenizer.chat_template is None:
        text = f"User: {example['messages'][0]['content']}\n\nAssistant: {example['messages'][1]['content']}"
    else:
        text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False, #retornar token (true) ou string (false)
            add_generation_prompt=False
        )
    return text

#configs lora
peft_config = LoraConfig(
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["lora_alpha"],
    lora_dropout=cfg["lora"]["lora_dropout"],
    bias=cfg["lora"]["bias"],
    task_type=cfg["lora"]["task_type"],
    target_modules=cfg["lora"]["target_modules"],
)

#configs treinamento
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
    gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
    num_train_epochs=cfg["training"]["num_train_epochs"],
    learning_rate=cfg["training"]["learning_rate"],
    fp16=cfg["training"]["fp16"],
    bf16=cfg["training"]["bf16"],
    logging_steps=cfg["training"]["logging_steps"],
    save_steps=cfg["training"]["save_steps"],
    save_total_limit=cfg["training"]["save_total_limit"],
    report_to=cfg["training"]["report_to"],
    optim=cfg["training"]["optimizer"],
)

print("\n Iniciando treinamento")

#treinamento
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
    peft_config=peft_config,
    formatting_func=formatting_func,
)
trainer.train()

print(f"\nSalvando modelo em {output_dir}")
trainer.save_model(output_dir)
print("Treinamento finalizado")