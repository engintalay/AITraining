import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Fix padding token issue if present, though not explicitly asked, it is good practice
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

data_file = "data.json"
if len(sys.argv) > 1:
    data_file = sys.argv[1]

print(f"Loading data from: {data_file}")
dataset = load_dataset("json", data_files=data_file)

def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""

training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    fp16=True,
    bf16=False,
    logging_steps=1,
    output_dir="./out",
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
