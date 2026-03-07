#!/usr/bin/env python3
"""Export pretrained entity encoder weights to JSON for Rust inference.

Extracts the encoder portion (proj, type_emb, self-attention layers, norms)
from the EntityEncoderPretraining model and writes JSON matching the Rust
EntityEncoderJson schema.

Usage:
    uv run --with numpy --with torch training/export_entity_encoder.py \
        generated/entity_encoder_pretrained.pt \
        -o generated/entity_encoder_weights.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Import the pretraining model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pretrain_entity import EntityEncoderPretraining


def export_linear(sd: dict, prefix: str) -> dict:
    w = sd[f"{prefix}.weight"].cpu().numpy().tolist()
    b = sd[f"{prefix}.bias"].cpu().numpy().tolist()
    return {"w": w, "b": b}


def export_in_proj(sd: dict, prefix: str) -> dict:
    w = sd[f"{prefix}.in_proj_weight"].cpu().numpy().tolist()
    b = sd[f"{prefix}.in_proj_bias"].cpu().numpy().tolist()
    return {"w": w, "b": b}


def export_layer_norm(sd: dict, prefix: str) -> dict:
    gamma = sd[f"{prefix}.weight"].cpu().numpy().tolist()
    beta = sd[f"{prefix}.bias"].cpu().numpy().tolist()
    return {"gamma": gamma, "beta": beta}


def main():
    p = argparse.ArgumentParser(description="Export entity encoder weights to JSON")
    p.add_argument("checkpoint", help="PyTorch checkpoint (.pt)")
    p.add_argument("-o", "--output", default="generated/entity_encoder_weights.json")
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    args = p.parse_args()

    model = EntityEncoderPretraining(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    sd = model.state_dict()

    export = {
        "proj": export_linear(sd, "proj"),
        "type_emb": sd["type_emb.weight"].cpu().numpy().tolist(),
        "input_norm": export_layer_norm(sd, "input_norm"),
        "self_attn_layers": [],
        "out_norm": export_layer_norm(sd, "out_norm"),
    }

    for i in range(args.n_layers):
        prefix = f"encoder.layers.{i}"
        layer = {
            "self_attn_in_proj": export_in_proj(sd, f"{prefix}.self_attn"),
            "self_attn_out_proj": export_linear(sd, f"{prefix}.self_attn.out_proj"),
            "ff_linear1": export_linear(sd, f"{prefix}.linear1"),
            "ff_linear2": export_linear(sd, f"{prefix}.linear2"),
            "norm1": export_layer_norm(sd, f"{prefix}.norm1"),
            "norm2": export_layer_norm(sd, f"{prefix}.norm2"),
        }
        export["self_attn_layers"].append(layer)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(export, f)

    size_kb = out.stat().st_size / 1024
    n_encoder_params = sum(
        p.numel() for name, p in model.named_parameters()
        if not name.startswith(("win_head", "hp_head"))
    )
    print(f"Exported {n_encoder_params:,} encoder params to {out} ({size_kb:.0f} KB)")
    print(f"(Dropped win_head and hp_head — only encoder portion exported)")


if __name__ == "__main__":
    main()
