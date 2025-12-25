import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load PEFT config
peft_model_id = "./out"
config = PeftConfig.from_pretrained(peft_model_id)

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    load_in_4bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, peft_model_id)

print("Creating pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

import json

# Define the instruction
instruction = "Linux'ta bir dizini içindekilerle birlikte nasıl kopyalarım?"

# Load ground truth
with open("data.json", "r") as f:
    data = json.load(f)
    expected_response = next((item["response"] for item in data if item["instruction"] == instruction), "Not found")

print("Running inference...")
result = pipe(f"### Instruction:\n{instruction}\n\n### Response:\n", max_new_tokens=1256)
generated_text = result[0]['generated_text']

print("\n" + "="*50)
print(f"Instruction: {instruction}")
print("-" * 20)
print(f"Expected Response (from data.json): {expected_response}")
print("-" * 20)
print(f"Model Response:\n{generated_text}")
print("="*50)

