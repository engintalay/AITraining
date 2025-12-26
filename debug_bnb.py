import torch
import bitsandbytes as bnb
import os

print(f"HSA_OVERRIDE: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")

if not torch.cuda.is_available():
    print("CUDA not available")
    exit(1)

device = torch.device("cuda:0")
print(f"Testing bitsandbytes 4-bit on {torch.cuda.get_device_name(0)}")

try:
    a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16) # Input as bf16
    # Test Linear4bit
    layer = bnb.nn.Linear4bit(1024, 1024, compute_dtype=torch.bfloat16).to(device)
    y = layer(a)
    print("BNB Linear4bit successful")
except Exception as e:
    print(f"BNB Error: {e}")
