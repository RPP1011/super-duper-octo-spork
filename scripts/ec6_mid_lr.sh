#!/bin/bash
# EC6: Mid LR, 1 epoch
# Start from P2 best
# LR=2e-5 (between EC3's 3e-5 which oscillates, and EC5's 1e-5 which is too slow)
# 1 epoch (3 epochs proven unhelpful in EC4)
set -euo pipefail

OUT_DIR="generated/impala_ec6"
REGISTRY="generated/ability_embedding_registry.json"
LOG="/tmp/impala_ec6.log"

COMMON_FLAGS=(
    --embedding-registry "$REGISTRY"
    --external-cls-dim 128
    --threads 64 --sims-per-thread 64
    --episodes-per-scenario 5
    --gpu --temperature 1.0 --batch-size 1024
    --train-epochs 1
    --value-coeff 0.5 --entropy-coeff 0.01
    --reward-scale 1.0 --kl-coeff 0.0 --max-train-steps 0
    --eval-every 5
)

mkdir -p "$OUT_DIR"

run_phase() {
    local phase=$1
    local scenarios=$2
    local iters=$3
    local ckpt=$4
    local lr=$5

    echo "=== Phase $phase: $scenarios ($iters iters, lr=$lr, 1 epoch) ===" | tee -a "$LOG"

    PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/impala_learner.py \
        --scenarios "$scenarios" \
        --checkpoint "$ckpt" \
        --output-dir "$OUT_DIR/phase${phase}" \
        --iters "$iters" \
        --eval-scenarios "$scenarios" \
        --lr "$lr" \
        "${COMMON_FLAGS[@]}" \
        2>&1 | tee -a "$LOG"

    echo "  Phase $phase complete" | tee -a "$LOG"
}

echo "Starting EC6 mid-LR experiment $(date)" | tee "$LOG"

# Phase 3: lr=2e-5, 1 epoch
run_phase 3 "dataset/scenarios/curriculum/phase3" 20 "generated/impala_scratch/phase2/best.pt" 2e-5

# Phase 4: lr=2e-5, 1 epoch
run_phase 4 "dataset/scenarios/curriculum/phase4" 20 "$OUT_DIR/phase3/best.pt" 2e-5

echo "EC6 complete $(date)" | tee -a "$LOG"
