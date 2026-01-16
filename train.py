import sys
import torch
import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser(description="Fine-tune a model.")
parser.add_argument("data_file", type=str, nargs="?", default="data.json", help="Path to the data file.")
parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device.")
parser.add_argument("--grad_acc", type=int, default=8, help="Gradient accumulation steps.")
parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 mixed precision.")
args = parser.parse_args()

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Fix padding token issue if present, though not explicitly asked, it is good practice
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
    torch_dtype=torch.float16,
    attn_implementation="eager"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Force-cast all parameters to float16 to avoid BFloat16 AMP error
for param in model.parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)

print("Verifying model dtypes:")
dtypes = {}
for name, param in model.named_parameters():
    dtype = param.dtype
    if dtype not in dtypes:
        dtypes[dtype] = 0
    dtypes[dtype] += 1
print(f"Parameter dtypes: {dtypes}")


print(f"Loading data from: {args.data_file}")
dataset = load_dataset("json", data_files=args.data_file)

def format_prompt(example):
    # TinyLlama Chat Template
    # <|system|> ... </s> <|user|> ... </s> <|assistant|> ... </s>
    text = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{example['instruction']}</s>\n<|assistant|>\n{example['response']}</s>"
    return text

training_args = SFTConfig(
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.grad_acc,
    num_train_epochs=30,
    learning_rate=1e-4, # Lowered LR
    fp16=not args.no_fp16,
    bf16=False,
    logging_steps=1,
    dataloader_num_workers=4,
    output_dir="./out",
    optim="adamw_torch",
    gradient_checkpointing=True,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    formatting_func=format_prompt,
    processing_class=tokenizer,
    args=training_args
)


trainer.train(resume_from_checkpoint=args.resume)
trainer.save_model()
