
import torch
import os

# Explicitly set device before anything else
os.environ["HIP_VISIBLE_DEVICES"] = "0"

print("Initializing minimal debug script...")
try:
    if not torch.cuda.is_available():
        print("❌ CUDA/ROCm not available.")
        exit(1)
    
    device = torch.device("cuda:0")
    print(f"✅ Device: {torch.cuda.get_device_name(0)}")

    # Create a small dummy model to test kernels
    print("Creating dummy model on GPU...")
    model = torch.nn.Linear(1024, 1024).to(device, dtype=torch.float16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("Running dummy training loop (Pure PyTorch)...")
    for i in range(50):
        optimizer.zero_grad()
        data = torch.randn(16, 1024, device=device, dtype=torch.float16)
        output = model(data)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Step {i}: Loss {loss.item()}")
            
    print("✅ Dummy training loop passed.")

    # Now try loading the actual model (just layer structure, random weights is fine or pretrained)
    # We use random to avoid download for quick debug if possible, but user has the model cached.
    # Let's try loading the Real TinyLlama to test RoPE/Attention kernels.
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print("Loading TinyLlama config...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config = AutoConfig.from_pretrained(model_id)
    # Force eager attention
    config._attn_implementation = "eager"

    print("Initializing TinyLlama with random weights (to test kernels)...")
    # We use random weights to save time, the kernels are the same.
    real_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16).to(device)
    
    print("Running Real Model forward/backward (10 steps)...")
    real_optimizer = torch.optim.AdamW(real_model.parameters(), lr=1e-4)
    
    input_ids = torch.randint(0, 32000, (2, 128), device=device)
    
    for i in range(10):
        real_optimizer.zero_grad()
        outputs = real_model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        real_optimizer.step()
        print(f"Real Model Step {i}: Loss {loss.item()}")

    print("🎉 SUCCESS! The hardware and basic kernels are working.")

except Exception as e:
    print(f"\n❌ CRASHED: {e}")
    import traceback
    traceback.print_exc()
