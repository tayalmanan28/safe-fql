#!/bin/bash
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/tayalmanan/miniconda3/envs/SafeFQL/bin/python

# Alpha sweep for Swimmer (env 19)
echo "=== Swimmer Alpha Sweep ==="
for alpha in 0.01 0.1 1.0 10 100; do
    echo "=== Swimmer (env 19), alpha=$alpha ==="
    $PYTHON launcher/examples/train_offline.py \
        --env_id 19 \
        --config configs/train_config.py:normalized_safefql \
        --alpha $alpha \
        --experiment_name "normsafefql_alpha${alpha}" \
        2>&1 | tee logs/mujoco_swimmer_alpha${alpha}.log
done

# Alpha sweep for Walker2d (env 20)
echo "=== Walker2d Alpha Sweep ==="
for alpha in 0.01 0.1 1.0 10 100; do
    echo "=== Walker2d (env 20), alpha=$alpha ==="
    $PYTHON launcher/examples/train_offline.py \
        --env_id 20 \
        --config configs/train_config.py:normalized_safefql \
        --alpha $alpha \
        --experiment_name "normsafefql_alpha${alpha}" \
        2>&1 | tee logs/mujoco_walker2d_alpha${alpha}.log
done

# Alpha sweep for Hopper (env 18)
echo "=== Hopper Alpha Sweep ==="
for alpha in 0.01 0.1 1.0 10 100; do
    echo "=== Hopper (env 18), alpha=$alpha ==="
    $PYTHON launcher/examples/train_offline.py \
        --env_id 18 \
        --config configs/train_config.py:normalized_safefql \
        --alpha $alpha \
        --experiment_name "normsafefql_alpha${alpha}" \
        2>&1 | tee logs/mujoco_hopper_alpha${alpha}.log
done

echo "=== All MuJoCo alpha sweeps complete ==="
