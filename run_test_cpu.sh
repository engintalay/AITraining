#!/bin/bash
# CPU-only testing to avoid GPU hangs
source ../finetune/bin/activate

MODEL_PATH=${1:-"./out"}

python check_env.py
echo ">>> Testing with CPU-only inference"
python test_cpu.py Zogoria.test.json --model_path "$MODEL_PATH"
