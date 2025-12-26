import torch
import os

print(f"HSA_OVERRIDE extracted: {os.environ.get('HSA_OVERRIDE_GFX_VERSION')}")
print(f"Torch version: {torch.__version__}")
print(f"HIP version: {torch.version.hip}")

if not torch.cuda.is_available():
    print("CUDA not available")
    exit(1)

print(f"Device: {torch.cuda.get_device_name(0)}")

try:
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    y = torch.matmul(x, x)
    print("Matmul successful")
except Exception as e:
    print(f"Error: {e}")
