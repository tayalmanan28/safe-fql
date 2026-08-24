#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/tayalmanan/miniconda3/envs/SafeFQL/bin/python

# Coarse alpha sweep for HalfCheetah (env 17)
# Starting with a broad range to understand the scale
alphas=(0.01 0.1 1.0 10 100 1000 10000)

for alpha in "${alphas[@]}"; do
    echo "=== HalfCheetah (env 17), alpha=$alpha ==="
    $PYTHON launcher/examples/train_offline.py \
        --env_id 17 \
        --config configs/train_config.py:normalized_safefql \
        --alpha $alpha \
        --experiment_name "normsafefql_alpha${alpha}" \
        2>&1 | tee logs/mujoco_halfcheetah_alpha${alpha}.log
done

echo "=== HalfCheetah alpha sweep complete ==="
