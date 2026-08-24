#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/tayalmanan/miniconda3/envs/SafeFQL/bin/python

# Ultra-low safety_weight for poor performers
for env_id in 22 23 24 25 26 28; do
    for sw in 0.01 0.05; do
        echo "=== Env $env_id, safety_weight=$sw ==="
        $PYTHON launcher/examples/train_offline.py \
            --env_id $env_id \
            --config configs/train_config.py:normalized_safefql \
            --safety_weight $sw \
            --experiment_name "normsafefql_aggressive_sw${sw}" \
            2>&1 | tee logs/aggressive_env${env_id}_sw${sw}.log
    done
done

# For DroneCircle and CarRun, also try without safety constraints entirely
for env_id in 24 25; do
    echo "=== Env $env_id, safety_weight=0.001 (near-zero) ==="
    $PYTHON launcher/examples/train_offline.py \
        --env_id $env_id \
        --config configs/train_config.py:normalized_safefql \
        --safety_weight 0.001 \
        --experiment_name "normsafefql_nearly_unsafe" \
        2>&1 | tee logs/aggressive_env${env_id}_sw0.001.log
done
