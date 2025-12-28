#!/bin/bash
# Override for AMD 780M (RDNA3) compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ACCELERATE_MIXED_PRECISION=no
source finetune/bin/activate

# Handle arguments
if [ "$1" == "reset" ]; then
    echo "Resetting training..."
    rm -rf out
    python train.py Zogoria.json
elif [ "$1" == "resume" ]; then
    echo "Resuming training..."
    python train.py Zogoria.json --resume
else
    echo "Starting training (default)..."
    python train.py Zogoria.json
fi
