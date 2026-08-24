#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/tayalmanan/miniconda3/envs/SafeFQL/bin/python

# Final training with optimal safety_weight per environment
# Env 21: CarCircle - sw=0.1
echo "=== Env 21: CarCircle (safety_weight=0.1) ==="
$PYTHON launcher/examples/train_offline.py \
    --env_id 21 \
    --config configs/train_config.py:normalized_safefql \
    --safety_weight 0.1 \
    --experiment_name "normsafefql_final" \
    2>&1 | tee logs/final_env21.log

# Env 22: AntRun - sw=0.5
echo "=== Env 22: AntRun (safety_weight=0.5) ==="
$PYTHON launcher/examples/train_offline.py \
    --env_id 22 \
    --config configs/train_config.py:normalized_safefql \
    --safety_weight 0.5 \
    --experiment_name "normsafefql_final" \
    2>&1 | tee logs/final_env22.log

# Env 23: DroneRun - sw=0.1
echo "=== Env 23: DroneRun (safety_weight=0.1) ==="
$PYTHON launcher/examples/train_offline.py \
    --env_id 23 \
    --config configs/train_config.py:normalized_safefql \
    --safety_weight 0.1 \
    --experiment_name "normsafefql_final" \
    2>&1 | tee logs/final_env23.log

# Env 24: DroneCircle - sw=0.1
echo "=== Env 24: DroneCircle (safety_weight=0.1) ==="
$PYTHON launcher/examples/train_offline.py \
    --env_id 24 \
    --config configs/train_config.py:normalized_safefql \
    --safety_weight 0.1 \
    --experiment_name "normsafefql_final" \
    2>&1 | tee logs/final_env24.log

# Env 25: CarRun - sw=100 (FIXED for safety)
echo "=== Env 25: CarRun (safety_weight=100) ==="
$PYTHON launcher/examples/train_offline.py \
    --env_id 25 \
    --config configs/train_config.py:normalized_safefql \
    --safety_weight 100 \
    --experiment_name "normsafefql_final" \
    2>&1 | tee logs/final_env25.log

# Env 26: AntCircle - sw=1.0 (baseline)
echo "=== Env 26: AntCircle (safety_weight=1.0) ==="
$PYTHON launcher/examples/train_offline.py \
    --env_id 26 \
    --config configs/train_config.py:normalized_safefql \
    --safety_weight 1.0 \
    --experiment_name "normsafefql_final" \
    2>&1 | tee logs/final_env26.log

# Env 27: BallCircle - sw=1.0 (baseline)
echo "=== Env 27: BallCircle (safety_weight=1.0) ==="
$PYTHON launcher/examples/train_offline.py \
    --env_id 27 \
    --config configs/train_config.py:normalized_safefql \
    --safety_weight 1.0 \
    --experiment_name "normsafefql_final" \
    2>&1 | tee logs/final_env27.log

# Env 28: BallRun - sw=0.5
echo "=== Env 28: BallRun (safety_weight=0.5) ==="
$PYTHON launcher/examples/train_offline.py \
    --env_id 28 \
    --config configs/train_config.py:normalized_safefql \
    --safety_weight 0.5 \
    --experiment_name "normsafefql_final" \
    2>&1 | tee logs/final_env28.log

echo "=== All final training complete ==="
