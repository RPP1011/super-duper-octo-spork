#!/usr/bin/env python3
"""Extract attention weights from AbilityActorCriticV4 for visualization.

Uses PyTorch forward hooks to capture attention weights without modifying model code.
"""
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "training"))

import torch
from model import AbilityActorCriticV4


def load_model(weights_path: str, device: str = "cpu"):
    """Load a V4 model from checkpoint."""
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)

    # Detect model config from checkpoint
    state = ckpt if isinstance(ckpt, dict) and "model_state_dict" not in ckpt else ckpt.get("model_state_dict", ckpt)

    # Detect vocab size from token embedding shape
    vocab_size = 252
    for key in state:
        if "token_emb.weight" in key:
            vocab_size = state[key].shape[0]
            break

    model = AbilityActorCriticV4(
        vocab_size=vocab_size,
        d_model=32,
        d_ff=64,
        n_layers=4,
        n_heads=4,
        entity_encoder_layers=4,
        external_cls_dim=128,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


class AttentionCapture:
    """Captures attention weights via forward hooks."""

    def __init__(self, model: AbilityActorCriticV4):
        self.model = model
        self.hooks = []
        self.cross_attn_weights = []  # per cross-attn block
        self.entity_self_attn_weights = []  # per encoder layer
        self._install_hooks()

    def _install_hooks(self):
        # Hook cross-attention blocks
        for name, module in self.model.named_modules():
            if hasattr(module, 'cross_attn') and hasattr(module.cross_attn, 'forward'):
                # This is a CrossAttentionBlock — hook its internal nn.MultiheadAttention
                self.hooks.append(
                    module.cross_attn.register_forward_hook(self._cross_attn_hook)
                )

        # Hook entity encoder self-attention layers
        if hasattr(self.model, 'entity_encoder') and hasattr(self.model.entity_encoder, 'encoder'):
            for layer in self.model.entity_encoder.encoder.layers:
                if hasattr(layer, 'self_attn'):
                    self.hooks.append(
                        layer.self_attn.register_forward_hook(self._self_attn_hook)
                    )

    def _cross_attn_hook(self, module, input, output):
        """Capture cross-attention weights from nn.MultiheadAttention."""
        # output is (attn_output, attn_weights) when need_weights=True
        # But by default need_weights=True, average_attn_weights=True
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            # attn_weights shape: (batch, tgt_len, src_len) or (batch, n_heads, tgt_len, src_len)
            self.cross_attn_weights.append(output[1].detach().cpu())

    def _self_attn_hook(self, module, input, output):
        """Capture self-attention weights from encoder layers."""
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.entity_self_attn_weights.append(output[1].detach().cpu())

    def clear(self):
        self.cross_attn_weights = []
        self.entity_self_attn_weights = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def extract_attention_for_episode(model, episode, embedding_registry=None, device="cpu"):
    """Run model forward on each step and capture attention weights.

    Returns list of per-step attention data.
    """
    capture = AttentionCapture(model)
    results = []

    # Load embedding registry for CLS embeddings
    cls_cache = {}
    if embedding_registry:
        with open(embedding_registry) as f:
            registry = json.load(f)
        for entry in registry.get("abilities", registry.get("entries", [])):
            name = entry.get("name", entry.get("ability_name", ""))
            emb = entry.get("embedding", entry.get("cls_embedding", []))
            if name and emb:
                cls_cache[name] = torch.tensor(emb, dtype=torch.float32)

    for step in episode["steps"]:
        capture.clear()

        # Build input tensors from step data
        entities = torch.tensor([step["entities"]], dtype=torch.float32, device=device)
        entity_types = torch.tensor([step["entity_types"]], dtype=torch.long, device=device)
        raw_threats = step.get("threats", [])
        feat_dim = entities.shape[2]
        if raw_threats and len(raw_threats) > 0 and len(raw_threats[0]) > 0:
            threats = torch.tensor([raw_threats], dtype=torch.float32, device=device)
            threat_mask = torch.zeros(1, threats.shape[1], dtype=torch.bool, device=device)
        else:
            # No threats — create 1 dummy threat, masked out
            threats = torch.zeros(1, 1, feat_dim, device=device)
            threat_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

        entity_mask = torch.zeros(1, entities.shape[1], dtype=torch.bool, device=device)

        # Build ability CLS embeddings
        unit_id = str(step["unit_id"])
        ability_names = episode.get("unit_ability_names", {}).get(unit_id, [])
        ability_cls_list = []
        for name in ability_names:
            if name in cls_cache:
                ability_cls_list.append(cls_cache[name].unsqueeze(0).to(device))
            else:
                ability_cls_list.append(None)

        # Pad to at least 1
        if not ability_cls_list:
            ability_cls_list = [None]

        # Positions
        positions = step.get("positions")
        pos_tensor = None
        pos_mask = None
        if positions and len(positions) > 0:
            pos_tensor = torch.tensor([positions], dtype=torch.float32, device=device)
            pos_mask = torch.zeros(1, len(positions), dtype=torch.bool, device=device)

        with torch.no_grad():
            try:
                output, value = model(
                    entities, entity_types, threats,
                    entity_mask, threat_mask,
                    ability_cls_list,
                    pos_tensor, pos_mask,
                )
            except Exception as e:
                results.append({"error": str(e)})
                continue

        # Collect attention data
        step_attn = {
            "tick": step["tick"],
            "unit_id": step["unit_id"],
            "cross_attention": [],
            "pointer_logits": {},
        }

        # Cross-attention weights (one per ability that had CLS embedding)
        for i, w in enumerate(capture.cross_attn_weights):
            # w shape: (1, tgt_len, src_len) — averaged over heads
            # tgt_len=1 (CLS query), src_len=n_entities
            attn = w.squeeze().numpy()
            if attn.ndim == 1:
                step_attn["cross_attention"].append({
                    "ability": ability_names[i] if i < len(ability_names) else f"ability_{i}",
                    "weights": attn.tolist(),
                })
            elif attn.ndim == 2:
                # Multi-head: (n_heads, src_len) or (tgt, src)
                step_attn["cross_attention"].append({
                    "ability": ability_names[i] if i < len(ability_names) else f"ability_{i}",
                    "weights": attn.mean(axis=0).tolist() if attn.shape[0] <= 8 else attn[0].tolist(),
                })

        # Pointer logits from output
        if "attack_ptr_logits" in output:
            logits = output["attack_ptr_logits"].squeeze().cpu().numpy()
            probs = _softmax(logits)
            step_attn["pointer_logits"]["attack"] = probs.tolist()

        if "move_logits" in output:
            logits = output["move_logits"].squeeze().cpu().numpy()
            probs = _softmax(logits)
            step_attn["pointer_logits"]["move"] = probs.tolist()

        if "combat_logits" in output:
            logits = output["combat_logits"].squeeze().cpu().numpy()
            probs = _softmax(logits)
            step_attn["pointer_logits"]["combat"] = probs.tolist()

        # Value prediction
        step_attn["value"] = value.item()

        results.append(step_attn)

    capture.remove_hooks()
    return results


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--episode-file", required=True)
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--registry", default=str(ROOT / "generated/ability_embedding_registry.json"))
    p.add_argument("--output", default="-")
    args = p.parse_args()

    model = load_model(args.weights)

    with open(args.episode_file) as f:
        for i, line in enumerate(f):
            if i == args.episode_idx:
                episode = json.loads(line)
                break
        else:
            print(f"Episode {args.episode_idx} not found", file=sys.stderr)
            sys.exit(1)

    results = extract_attention_for_episode(model, episode, args.registry)

    output = json.dumps(results, indent=2)
    if args.output == "-":
        print(output)
    else:
        Path(args.output).write_text(output)
        print(f"Wrote {len(results)} steps to {args.output}", file=sys.stderr)
