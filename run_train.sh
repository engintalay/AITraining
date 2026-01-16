#!/bin/bash
# Auto-detect AMD GPU and apply overrides
if [ -e "/dev/kfd" ]; then
    echo "AMD GPU detected. Applying ROCm compatibility overrides."
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export ACCELERATE_MIXED_PRECISION=no
    export HSA_ENABLE_SDMA=0
    export HIP_VISIBLE_DEVICES=0
    export HIPBLASLT_ENABLE=0
    export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256
    export ROCM_FORCE_SYNC_COPY=1
    export GPU_MAX_HW_QUEUES=1
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
