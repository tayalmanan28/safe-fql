#!/bin/bash
# =============================================================================
# Training time benchmark for SafeFQL_Base, SafeFQL_Base+IFQL, and Ours (SafeFQL / SafeFQL)
# on the BoatRobot environment (env_id=30).
#
# Trains each framework once and records wall-clock training time.
# Evaluation is effectively disabled (eval_interval > max_steps for all
# steps except step 0, which runs a single fast eval episode).
#
# Usage (run from the repo root):
#   chmod +x scripts/benchmark_training_time.sh
#   conda activate SafeFQL_Base
#   bash scripts/benchmark_training_time.sh
# =============================================================================

set -e  # exit on error

ENV_ID=30
LOG_DIR="training_time_results"
mkdir -p "$LOG_DIR"

echo "============================================="
echo " Training Time Benchmark — BoatRobot (env_id=$ENV_ID)"
echo "============================================="
echo ""

# ---- SafeFQL_Base ----
echo ">>> [1/3] Training SafeFQL_Base ..."
START_SafeFQL_Base=$(date +%s.%N)
python launcher/examples/train_offline.py \
    --config=configs/train_config.py:safefql_base \
    --env_id=$ENV_ID \
    --experiment_name="time_bench_safefql_base" \
    2>&1 | tee "$LOG_DIR/safefql_base_train.log"
END_SafeFQL_Base=$(date +%s.%N)
TIME_SafeFQL_Base=$(echo "$END_SafeFQL_Base - $START_SafeFQL_Base" | bc)
echo ">>> SafeFQL_Base training time: ${TIME_SafeFQL_Base}s"
echo ""

# ---- SafeFQL_Base+IFQL ----
echo ">>> [2/3] Training SafeFQL_Base+IFQL ..."
START_IFQL=$(date +%s.%N)
python launcher/examples/train_offline.py \
    --config=configs/train_config.py:safeifql \
    --env_id=$ENV_ID \
    --experiment_name="time_bench_safefql_base_ifql" \
    2>&1 | tee "$LOG_DIR/safefql_base_ifql_train.log"
END_IFQL=$(date +%s.%N)
TIME_IFQL=$(echo "$END_IFQL - $START_IFQL" | bc)
echo ">>> SafeFQL_Base+IFQL training time: ${TIME_IFQL}s"
echo ""

# ---- Ours (SafeFQL / SafeFQL) ----
echo ">>> [3/3] Training Ours (SafeFQL) ..."
START_OURS=$(date +%s.%N)
python launcher/examples/train_offline.py \
    --config=configs/train_config.py:safefql \
    --env_id=$ENV_ID \
    --experiment_name="time_bench_safefql" \
    2>&1 | tee "$LOG_DIR/safefql_train.log"
END_OURS=$(date +%s.%N)
TIME_OURS=$(echo "$END_OURS - $START_OURS" | bc)
echo ">>> Ours (SafeFQL) training time: ${TIME_OURS}s"
echo ""

# ---- Summary ----
echo "============================================="
echo " TRAINING TIME SUMMARY"
echo "============================================="
printf "%-20s %12s %12s\n" "Framework" "Time (s)" "Time (min)"
echo "---------------------------------------------"
printf "%-20s %12.2f %12.2f\n" "SafeFQL_Base"        "$TIME_SafeFQL_Base" "$(echo "$TIME_SafeFQL_Base / 60" | bc -l | xargs printf '%.2f')"
printf "%-20s %12.2f %12.2f\n" "SafeFQL_Base+IFQL"   "$TIME_IFQL"  "$(echo "$TIME_IFQL / 60"  | bc -l | xargs printf '%.2f')"
printf "%-20s %12.2f %12.2f\n" "Ours (SafeFQL)" "$TIME_OURS" "$(echo "$TIME_OURS / 60" | bc -l | xargs printf '%.2f')"
echo "============================================="

# Save summary to file
SUMMARY_FILE="$LOG_DIR/training_time_summary.txt"
{
    echo "Training Time Benchmark — BoatRobot (env_id=$ENV_ID)"
    echo "Date: $(date)"
    echo ""
    printf "%-20s %12s %12s\n" "Framework" "Time (s)" "Time (min)"
    echo "---------------------------------------------"
    printf "%-20s %12.2f %12.2f\n" "SafeFQL_Base"        "$TIME_SafeFQL_Base" "$(echo "$TIME_SafeFQL_Base / 60" | bc -l | xargs printf '%.2f')"
    printf "%-20s %12.2f %12.2f\n" "SafeFQL_Base+IFQL"   "$TIME_IFQL"  "$(echo "$TIME_IFQL / 60"  | bc -l | xargs printf '%.2f')"
    printf "%-20s %12.2f %12.2f\n" "Ours (SafeFQL)" "$TIME_OURS" "$(echo "$TIME_OURS / 60" | bc -l | xargs printf '%.2f')"
} > "$SUMMARY_FILE"

echo ""
echo "Summary saved to $SUMMARY_FILE"
echo "Full logs saved to $LOG_DIR/"
