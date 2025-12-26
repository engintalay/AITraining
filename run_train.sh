#!/bin/bash
# Override for AMD 780M (RDNA3) compatibility
export HSA_OVERRIDE_GFX_VERSION=11.0.0
source finetune/bin/activate
python train.py Zogoria.json
