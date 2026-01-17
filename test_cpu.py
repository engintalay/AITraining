import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def generate_response(model, tokenizer, prompt, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    if "<|assistant|>" in output:
        return output.split("<|assistant|>")[1].replace("</s>", "").strip()
    return output.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", help="Path to test data JSON")
    parser.add_argument("--model_path", default="./out", help="Path to trained model")
    args = parser.parse_args()

    # Load test data
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Loading model from {args.model_path}...")
    
    # Load config and base model
    config = PeftConfig.from_pretrained(args.model_path)
    base_model_name = config.base_model_name_or_path
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model on CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cpu",
        dtype=torch.float32,
        trust_remote_code=True
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, args.model_path)
    
    print("Running inference on CPU...")
    
    for i, item in enumerate(test_data[:3]):  # Test first 3 items
        instruction = item['instruction']
        expected = item.get('expected_logic', 'No expected response provided')
        
        prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n"
        
        print(f"\n--- Test {i+1} ---")
        print(f"Question: {instruction}")
        print(f"Expected: {expected}")
        
        response = generate_response(model, tokenizer, prompt)
        print(f"Generated: {response}")
        print("-" * 50)

if __name__ == "__main__":
    main()
