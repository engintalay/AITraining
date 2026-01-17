#!/bin/bash
# Override for AMD 780M (RDNA3) compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ACCELERATE_MIXED_PRECISION=no
export HSA_ENABLE_SDMA=0
export HIP_VISIBLE_DEVICES=0
export HIPBLASLT_ENABLE=0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512
source ../finetune/bin/activate

MODEL_PATH=${1:-"./out"}

python check_env.py
echo ">>> Phase 1: Base Model Inference"
python test.py Zogoria.test.json --mode base --model_path "$MODEL_PATH"
if [ $? -ne 0 ]; then echo "Phase 1 failed"; exit 1; fi

echo ">>> Phase 2: LoRA Model Inference"
python test.py Zogoria.test.json --mode lora --model_path "$MODEL_PATH"
if [ $? -ne 0 ]; then echo "Phase 2 failed"; exit 1; fi

echo ">>> Phase 3: Comparison"
python test.py Zogoria.test.json --mode compare --model_path "$MODEL_PATH"
