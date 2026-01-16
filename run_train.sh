#!/bin/bash
# Auto-detect AMD GPU and apply overrides
if [ -e "/dev/kfd" ]; then
    echo "AMD GPU detected. Applying ROCm compatibility overrides."
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export ACCELERATE_MIXED_PRECISION=no
fi
source ../finetune/bin/activate
python check_env.py
# Handle arguments
if [ "$1" == "reset" ]; then
    echo "Resetting training..."
    rm -rf out
    python train.py Zogoria.json "${@:2}"
elif [ "$1" == "resume" ]; then
    echo "Resuming training..."
    python train.py Zogoria.json --resume "${@:2}"
else
    echo "Starting training (default)..."
    python train.py Zogoria.json "$@"
fi
