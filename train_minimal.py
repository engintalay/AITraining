import sys
import torch
import argparse
import os

# Minimal GPU usage for RDNA3
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ["HIP_LAUNCH_BLOCKING"] = "1"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser(description="Fine-tune a model.")
parser.add_argument("data_file", type=str, nargs="?", default="data.json", help="Path to the data file.")
args = parser.parse_args()

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load on CPU first
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    dtype=torch.float32,
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=4,  # Minimal rank
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # Only 2 modules
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Move only LoRA parameters to GPU
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.to("cuda:0")

print(f"Loading data from: {args.data_file}")
dataset = load_dataset("json", data_files=args.data_file)

def format_prompt(example):
    text = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{example['instruction']}</s>\n<|assistant|>\n{example['response']}</s>"
    return text

training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=1e-4,
    fp16=False,
    bf16=False,
    logging_steps=5,
    dataloader_num_workers=0,
    output_dir="./out_minimal",
    optim="adamw_torch",
    max_grad_norm=1.0,
    save_steps=100,
    dataloader_pin_memory=False,
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
