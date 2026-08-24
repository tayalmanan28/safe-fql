#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/tayalmanan/miniconda3/envs/SafeFQL/bin/python

# Complete remaining conservative envs
for env_id in 24 27 28; do
    for sw in 0.5; do
        echo "=== Env $env_id, safety_weight=$sw ==="
        $PYTHON launcher/examples/train_offline.py \
            --env_id $env_id \
            --config configs/train_config.py:normalized_safefql \
            --safety_weight $sw \
            --experiment_name "normsafefql_sw${sw}" \
            2>&1 | tee logs/sweep_env${env_id}_sw${sw}.log
    done
done

# Env 28 also needs sw=0.1
echo "=== Env 28, safety_weight=0.1 ==="
$PYTHON launcher/examples/train_offline.py \
    --env_id 28 \
    --config configs/train_config.py:normalized_safefql \
    --safety_weight 0.1 \
    --experiment_name "normsafefql_sw0.1" \
    2>&1 | tee logs/sweep_env28_sw0.1.log

# Env 25 (CarRun): increase safety_weight for safety
for sw in 10 50 100; do
    echo "=== Env 25, safety_weight=$sw ==="
    $PYTHON launcher/examples/train_offline.py \
        --env_id 25 \
        --config configs/train_config.py:normalized_safefql \
        --safety_weight $sw \
        --experiment_name "normsafefql_sw${sw}" \
        2>&1 | tee logs/sweep_env25_sw${sw}.log
done
