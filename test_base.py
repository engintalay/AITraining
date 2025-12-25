import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Use correct quantization config to avoid warnings
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define the instruction
instruction = "Linux'ta bir dizini içindekilerle birlikte nasıl kopyalarım?"

# Load ground truth
with open("data.json", "r") as f:
    data = json.load(f)
    expected_response = next((item["response"] for item in data if item["instruction"] == instruction), "Not found")

print("Running inference with BASE model...")
result = pipe(f"### Instruction:\n{instruction}\n\n### Response:\n", max_new_tokens=256)
generated_text = result[0]['generated_text']

print("\n" + "="*50)
print(f"Instruction: {instruction}")
print("-" * 20)
print(f"Expected Response (from data.json): {expected_response}")
print("-" * 20)
print(f"BASE Model Response:\n{generated_text}")
print("="*50)
