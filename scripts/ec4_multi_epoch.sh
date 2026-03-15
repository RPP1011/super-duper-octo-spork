#!/bin/bash
# EC4: Multi-epoch experiment
# Start from P2 best (47.3% eval, healthy KL)
# Change: 3 epochs per iter instead of 1, lower LR for P3/P4
# Hypothesis: multiple smaller gradient passes = smoother policy updates = less KL oscillation
set -euo pipefail

OUT_DIR="generated/impala_ec4"
REGISTRY="generated/ability_embedding_registry.json"
LOG="/tmp/impala_ec4.log"

COMMON_FLAGS=(
    --embedding-registry "$REGISTRY"
    --external-cls-dim 128
    --threads 64 --sims-per-thread 64
    --episodes-per-scenario 5
    --gpu --temperature 1.0 --batch-size 1024
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
    local epochs=$6

    echo "=== Phase $phase: $scenarios ($iters iters, lr=$lr, epochs=$epochs) ===" | tee -a "$LOG"

    PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/impala_learner.py \
        --scenarios "$scenarios" \
        --checkpoint "$ckpt" \
        --output-dir "$OUT_DIR/phase${phase}" \
        --iters "$iters" \
        --eval-scenarios "$scenarios" \
        --lr "$lr" \
        --train-epochs "$epochs" \
        "${COMMON_FLAGS[@]}" \
        2>&1 | tee -a "$LOG"

    echo "  Phase $phase complete" | tee -a "$LOG"
}

echo "Starting EC4 multi-epoch experiment $(date)" | tee "$LOG"

# Phase 3: 3 epochs, lr=3e-5 (same LR, more passes)
run_phase 3 "dataset/scenarios/curriculum/phase3" 20 "generated/impala_scratch/phase2/best.pt" 3e-5 3

# Phase 4: 3 epochs, lr=1e-5 (lower LR + more passes for hardest phase)
run_phase 4 "dataset/scenarios/curriculum/phase4" 20 "$OUT_DIR/phase3/best.pt" 1e-5 3

echo "EC4 complete $(date)" | tee -a "$LOG"
