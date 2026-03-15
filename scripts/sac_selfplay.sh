#!/bin/bash
# SAC-Discrete self-play training
# Start from EC7 best (42.5% vs default AI)
# Both teams on GPU, swap sides for balanced matchups
# SAC auto-tunes entropy — no more manual coeff tuning
set -euo pipefail

OUT_DIR="generated/sac_sp1"
REGISTRY="generated/ability_embedding_registry.json"
LOG="/tmp/sac_sp1.log"

mkdir -p "$OUT_DIR"

echo "Starting SAC self-play $(date)" | tee "$LOG"

PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/sac_learner.py \
    --scenarios "dataset/scenarios/mirror" \
    --checkpoint "generated/random_init_v4.pt" \
    --output-dir "$OUT_DIR" \
    --embedding-registry "$REGISTRY" \
    --external-cls-dim 128 \
    --threads 64 --sims-per-thread 64 \
    --episodes-per-scenario 2 \
    --gpu --temperature 1.5 \
    --self-play-gpu \
    --iters 50 \
    --lr 1e-4 \
    --alpha-lr 1e-3 \
    --batch-size 256 \
    --buffer-size 100000 \
    --updates-per-iter 256 \
    --tau 0.001 \
    --init-alpha 0.2 \
    --gamma 0.99 \
    --reward-scale 1.0 \
    --min-buffer-size 5000 \
    --eval-scenarios "dataset/scenarios/curriculum/phase3" \
    --eval-every 5 \
    2>&1 | tee -a "$LOG"

echo "SAC SP1 complete $(date)" | tee -a "$LOG"
