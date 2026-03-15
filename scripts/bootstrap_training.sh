#!/bin/bash
# Bootstrap the full training pipeline from raw data.
# 1. Pretrain ability transformer (MLM, d=128)
# 2. Export embedding registry
# 3. Create random-init actor-critic checkpoint
# 4. Launch IMPALA curriculum
set -euo pipefail

echo "=== Step 1: Pretrain ability transformer (MLM) ==="
PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/pretrain.py \
    dataset/abilities_tokenized.npz \
    -o generated/ability_transformer_pretrained.pt \
    --d-model 128 --n-heads 4 --n-layers 4 --d-ff 256 \
    --max-steps 50000 --eval-every 1000 --batch-size 512 \
    --behavioral-data dataset/ability_profiles.npz \
    --behavioral-weight 1.0

echo "=== Step 2: Export embedding registry ==="
PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/export_embedding_registry.py \
    generated/ability_transformer_pretrained.pt \
    --ability-data dataset/ability_profiles.npz \
    --behavioral-data dataset/ability_profiles.npz \
    --d-model 128 --n-heads 4 --n-layers 4 --d-ff 256 --max-seq-len 256 \
    -o generated/ability_embedding_registry.json

echo "=== Step 3: Create random-init V4 checkpoint ==="
uv run --with numpy --with torch python3 -c "
import sys; sys.path.insert(0, 'training')
import torch
from model import AbilityActorCriticV4
from tokenizer import AbilityTokenizer
tok = AbilityTokenizer()
model = AbilityActorCriticV4(
    vocab_size=tok.vocab_size,
    entity_encoder_layers=4,
    external_cls_dim=128,
    d_model=32, d_ff=64, n_layers=4, n_heads=4,
)
torch.save({'model_state_dict': model.state_dict()}, 'generated/actor_critic_v4_random_init.pt')
n = sum(p.numel() for p in model.parameters())
print(f'Saved random-init V4 checkpoint: {n:,} params')
"

echo "=== Step 4: Launch IMPALA curriculum ==="
exec bash scripts/impala_from_scratch.sh
