#!/bin/bash
# EC7: KL penalty + advantage clipping
# Best LR (3e-5) but with KL penalty to prevent oscillation
# kl-coeff=1.0 (moderate, default is 2.0 but we've been at 0.0)
# advantage-clip=3.0 (clip extreme returns)
# 1 epoch (3 proven useless)
set -euo pipefail

OUT_DIR="generated/impala_ec7"
REGISTRY="generated/ability_embedding_registry.json"
LOG="/tmp/impala_ec7.log"

COMMON_FLAGS=(
    --embedding-registry "$REGISTRY"
    --external-cls-dim 128
    --threads 64 --sims-per-thread 64
    --episodes-per-scenario 5
    --gpu --temperature 1.0 --batch-size 1024
    --train-epochs 1
    --value-coeff 0.5 --entropy-coeff 0.01
    --reward-scale 1.0
    --kl-coeff 1.0
    --advantage-clip 3.0
    --max-train-steps 0
    --eval-every 5
)

mkdir -p "$OUT_DIR"

run_phase() {
    local phase=$1
    local scenarios=$2
    local iters=$3
    local ckpt=$4
    local lr=$5

    echo "=== Phase $phase: $scenarios ($iters iters, lr=$lr, kl=1.0, adv-clip=3.0) ===" | tee -a "$LOG"

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

echo "Starting EC7 KL-penalty experiment $(date)" | tee "$LOG"

# Phase 3: lr=3e-5, kl=1.0, advantage-clip=3.0
run_phase 3 "dataset/scenarios/curriculum/phase3" 20 "generated/impala_scratch/phase2/best.pt" 3e-5

# Phase 4: same settings
run_phase 4 "dataset/scenarios/curriculum/phase4" 20 "$OUT_DIR/phase3/best.pt" 3e-5

echo "EC7 complete $(date)" | tee -a "$LOG"
