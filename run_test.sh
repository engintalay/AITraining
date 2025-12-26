#!/bin/bash
# Override for AMD 780M (RDNA3) compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0
source finetune/bin/activate

echo ">>> Phase 1: Base Model Inference"
python test.py Zogoria.test.json --mode base
if [ $? -ne 0 ]; then echo "Phase 1 failed"; exit 1; fi

echo ">>> Phase 2: LoRA Model Inference"
python test.py Zogoria.test.json --mode lora
if [ $? -ne 0 ]; then echo "Phase 2 failed"; exit 1; fi

echo ">>> Phase 3: Comparison"
python test.py Zogoria.test.json --mode compare
