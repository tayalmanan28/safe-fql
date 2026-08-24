#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/tayalmanan/miniconda3/envs/SafeFQL/bin/python

for env_id in 21 22 23 24 25 26 27 28; do
    echo "=== Training env_id=$env_id ==="
    $PYTHON launcher/examples/train_offline.py \
        --env_id $env_id \
        --config configs/train_config.py:normalized_safefql \
        2>&1 | tee logs/bullet_env_${env_id}.log
done
