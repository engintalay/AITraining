
import sys
import torch
import importlib.util

def check_package(name):
    if name in sys.modules:
        print(f"✅ {name} is already imported")
        return True
    
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"❌ {name} NOT FOUND")
        return False
    
    try:
        module = __import__(name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {name} imported successfully (Version: {version})")
        return True
    except ImportError as e:
        print(f"❌ {name} failed to import: {e}")
        return False
    except Exception as e:
        print(f"❌ {name} crashed during import: {e}")
        return False

print(f"Python Version: {sys.version}")
print("-" * 40)

# 1. Basic Imports
packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "trl"
]

all_passed = True
for pkg in packages:
    if not check_package(pkg):
        all_passed = False

print("-" * 40)

# 2. PyTorch GPU Check
if torch.cuda.is_available():
    print(f"✅ CUDA/ROCm is available.")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   Device Name : {torch.cuda.get_device_name(0)}")
    try:
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("✅ Simple Tensor Operation on GPU successful")
    except Exception as e:
        print(f"❌ Tensor Operation Failed: {e}")
        all_passed = False
else:
    print("❌ CUDA/ROCm is NOT available (Torch is running on CPU)")
    all_passed = False

print("-" * 40)

# Bitsandbytes specific check (often tricky on ROCm)
if "bitsandbytes" in sys.modules:
    try:
        import bitsandbytes as bnb
        print(f"✅ Bitsandbytes imported. path: {bnb.__file__}")
    except Exception as e:
         print(f"⚠️ Bitsandbytes import issue: {e}")

if all_passed:
    print("\n🎉 All critical checks PASSED! Environment seems ready.")
else:
    print("\n⚠️ Some checks FAILED. Please review above.")
