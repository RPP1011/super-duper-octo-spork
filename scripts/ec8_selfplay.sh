#!/bin/bash
# EC8: Symmetric self-play via GPU
# Both teams use the same GPU inference server (same weights, truly symmetric)
# Start from EC7 best (42.5% vs default AI)
# KL penalty + high entropy to prevent collapse
set -euo pipefail

OUT_DIR="generated/impala_ec8"
REGISTRY="generated/ability_embedding_registry.json"
LOG="/tmp/impala_ec8.log"

COMMON_FLAGS=(
    --embedding-registry "$REGISTRY"
    --external-cls-dim 128
    --threads 64 --sims-per-thread 64
    --episodes-per-scenario 5
    --gpu --temperature 1.0 --batch-size 1024
    --train-epochs 1
    --value-coeff 0.5 --entropy-coeff 0.15
    --reward-scale 5.0
    --kl-coeff 1.0
    --advantage-clip 3.0
    --max-train-steps 0
    --eval-every 5
    --self-play-gpu
    --swap-sides
)

mkdir -p "$OUT_DIR"

echo "Starting EC8 symmetric self-play $(date)" | tee "$LOG"
echo "  Both teams on GPU, same weights" | tee -a "$LOG"

echo "=== Self-play P3: 272 scenarios, lr=3e-5, kl=1.0, symmetric GPU ===" | tee -a "$LOG"

PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/impala_learner.py \
    --scenarios "dataset/scenarios/curriculum/phase3" \
    --checkpoint "generated/impala_ec7/phase3/best.pt" \
    --output-dir "$OUT_DIR/sp2" \
    --iters 30 \
    --eval-scenarios "dataset/scenarios/curriculum/phase3" \
    --lr 3e-5 \
    "${COMMON_FLAGS[@]}" \
    2>&1 | tee -a "$LOG"

echo "EC8 SP2 complete $(date)" | tee -a "$LOG"
