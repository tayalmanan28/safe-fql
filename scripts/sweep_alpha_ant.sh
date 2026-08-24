#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/tayalmanan/miniconda3/envs/SafeFQL/bin/python

# Coarse alpha sweep for Ant (env 16)
# Based on HalfCheetah results, focusing on 0.01-100 range
alphas=(0.01 0.1 1.0 10 100)

for alpha in "${alphas[@]}"; do
    echo "=== Ant (env 16), alpha=$alpha ==="
    $PYTHON launcher/examples/train_offline.py \
        --env_id 16 \
        --config configs/train_config.py:normalized_safefql \
        --alpha $alpha \
        --experiment_name "normsafefql_alpha${alpha}" \
        2>&1 | tee logs/mujoco_ant_alpha${alpha}.log
done

echo "=== Ant alpha sweep complete ==="
