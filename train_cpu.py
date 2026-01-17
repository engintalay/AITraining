import sys
import torch
import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser(description="Fine-tune a model on CPU.")
parser.add_argument("data_file", type=str, nargs="?", default="data.json", help="Path to the data file.")
args = parser.parse_args()

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.float32,
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print(f"Loading data from: {args.data_file}")
dataset = load_dataset("json", data_files=args.data_file)

def format_prompt(example):
    text = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{example['instruction']}</s>\n<|assistant|>\n{example['response']}</s>"
    return text

training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=False,
    bf16=False,
    logging_steps=1,
    dataloader_num_workers=0,
    output_dir="./out_cpu",
    optim="adamw_torch",
    max_grad_norm=0.5,
    save_steps=50,
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    formatting_func=format_prompt,
    processing_class=tokenizer,
    args=training_args
)

trainer.train()
trainer.save_model()
