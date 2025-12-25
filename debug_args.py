
import trl
import inspect
from trl import SFTTrainer, SFTConfig

print(f"TRL version: {trl.__version__}")

print("\n--- SFTConfig arguments ---")
try:
    print(inspect.signature(SFTConfig.__init__))
except Exception as e:
    print(e)

print("\n--- SFTTrainer arguments ---")
try:
    print(inspect.signature(SFTTrainer.__init__))
except Exception as e:
    print(e)
