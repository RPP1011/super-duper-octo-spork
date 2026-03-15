#!/bin/bash
# GRU temporal context + mixed pool training
# Based on "Multi-agent cooperation through in-context co-player inference"
# Key: train against diverse opponents to force in-context adaptation
#
# Phase 1: Train vs default AI on curriculum P3 (learn basic competence + temporal patterns)
# Phase 2: Self-play on mirrors (once > 40% vs default)
set -euo pipefail

OUT_DIR="generated/gru_mixed"
REGISTRY="generated/ability_embedding_registry.json"
LOG="/tmp/gru_mixed.log"

mkdir -p "$OUT_DIR"

echo "Starting GRU mixed-pool training $(date)" | tee "$LOG"

# Phase 1: vs default AI with temporal context
echo "=== Phase 1: vs default AI, P3 curriculum, h_dim=64 ===" | tee -a "$LOG"

PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/impala_learner.py \
    --scenarios "dataset/scenarios/curriculum/phase3" \
    --checkpoint "generated/random_init_v4_gru.pt" \
    --output-dir "$OUT_DIR/phase1" \
    --embedding-registry "$REGISTRY" \
    --external-cls-dim 128 \
    --h-dim 64 \
    --threads 64 --sims-per-thread 64 \
    --episodes-per-scenario 5 \
    --gpu --temperature 1.0 --batch-size 1024 \
    --train-epochs 1 \
    --value-coeff 0.5 --entropy-coeff 0.01 \
    --reward-scale 1.0 \
    --kl-coeff 1.0 \
    --max-train-steps 0 \
    --eval-every 5 \
    --lr 3e-5 \
    --iters 30 \
    --eval-scenarios "dataset/scenarios/curriculum/phase3" \
    2>&1 | tee -a "$LOG"

echo "Phase 1 complete $(date)" | tee -a "$LOG"
