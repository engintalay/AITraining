import sys
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def run_inference(model, tokenizer, instruction):
    # Match the training template: <|system|>...<|user|>...<|assistant|>
    prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{instruction}</s>\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    if "<|assistant|>" in output:
        return output.split("<|assistant|>")[1].replace("</s>", "").strip()
    return output.strip()

def phase_inference(args, mode):
    peft_model_id = "./out"
    config = PeftConfig.from_pretrained(peft_model_id)
    base_model_name = config.base_model_name_or_path

    print(f"Loading data from {args.test_file}...")
    with open(args.test_file, 'r') as f:
        data = json.load(f)

    print(f"Loading Tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Base Model...")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        return_dict=True,
        device_map="auto",
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    if mode == "lora":
        print("Loading LoRA Adapter...")
        model = PeftModel.from_pretrained(model, peft_model_id)
        # model.to(device)

    results = []
    print(f"Running Inference ({mode})...")
    for item in data:
        resp = run_inference(model, tokenizer, item['instruction'])
        results.append(resp)

    output_file = f"{mode}_results.tmp"
    with open(output_file, 'w') as f:
        json.dump(results, f)
    print(f"Saved results to {output_file}")

def phase_compare(args):
    print("Loading test data...")
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)

    try:
        with open("base_results.tmp", 'r') as f:
            base_results = json.load(f)
        with open("lora_results.tmp", 'r') as f:
            lora_results = json.load(f)
    except FileNotFoundError:
        print("Error: Result files not found. Run inference phases first.")
        sys.exit(1)

    output_text = []
    output_text.append("#"*80)
    output_text.append(" COMPARATIVE RESULTS ")
    output_text.append("#"*80)

    for i, item in enumerate(test_data):
        output_text.append(f"\nTEST CASE {i+1}:")
        output_text.append(f"Instruction: {item['instruction']}")
        output_text.append(f"Expected Logic: {item.get('expected_logic', 'N/A')}")
        output_text.append("-" * 40)
        output_text.append(f"BASE MODEL: {base_results[i]}")
        output_text.append("-" * 40)
        output_text.append(f"LORA MODEL: {lora_results[i]}")
        output_text.append("-" * 40)
        output_text.append(f"Note: {item.get('note', '')}")
        output_text.append("="*80)
    
    final_output = "\n".join(output_text)
    print(final_output)
    
    with open("comparison_results.txt", "w") as f:
        f.write(final_output)
    print(f"\nResults saved to comparison_results.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", help="Path to test data JSON")
    parser.add_argument("--mode", choices=["base", "lora", "compare"], required=True)
    args = parser.parse_args()

    if args.mode == "compare":
        phase_compare(args)
    else:
        phase_inference(args, args.mode)

if __name__ == "__main__":
    main()
