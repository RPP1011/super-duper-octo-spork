#!/usr/bin/env python3
"""Phase 2: Fine-tune ability transformer for urgency + target prediction.

Loads a pre-trained transformer checkpoint and fine-tunes it on oracle-labeled
ability evaluation data. Uses grokking-informed settings (AdamW λ=1.0, no
dropout, oracle agreement as stopping criterion).

The model processes ability DSL tokens via the transformer encoder, concatenates
the [CLS] embedding with game state features, and predicts urgency + target.

Dataset format (JSONL):
    {
        "ability_dsl": "ability Fireball { ... }",
        "game_state": [0.5, 0.3, ...],   // category-specific features
        "urgency": 0.72,
        "target_idx": 1,                 // -1 if no target (aoe/self)
        "category": "damage_unit"
    }

Usage:
    # 1. Generate dataset with DSL text (new xtask subcommand needed):
    cargo xtask scenario oracle ability-dataset-dsl scenarios/ \
        -o generated/ability_eval_dsl.jsonl

    # 2. Fine-tune:
    uv run --with numpy --with torch training/finetune_decision.py \
        generated/ability_eval_dsl.jsonl \
        --pretrained generated/ability_transformer_pretrained.pt \
        -o generated/ability_transformer_decision.pt \
        --max-steps 300000 --patience 30000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import AbilityTransformerDecision, AbilityTransformer
from tokenizer import AbilityTokenizer
from grokfast import GrokfastEMA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Categories that have target selection (3-target output)
UNIT_TARGET_CATEGORIES = {"damage_unit", "cc_unit", "heal_unit"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AbilityDecisionDataset:
    """Loads oracle-labeled ability evaluation data with DSL text."""

    def __init__(self, path: Path, tokenizer: AbilityTokenizer):
        self.tokenizer = tokenizer
        self.samples: list[dict] = []

        with open(path) as f:
            for line in f:
                sample = json.loads(line)
                if "ability_dsl" not in sample or "urgency" not in sample:
                    continue
                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples from {path}")

        # Compute game state dimension from first sample
        self.game_state_dim = len(self.samples[0].get("game_state", []))
        print(f"Game state dim: {self.game_state_dim}")

        # Category distribution
        cats = {}
        for s in self.samples:
            c = s.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        for c, n in sorted(cats.items()):
            print(f"  {c}: {n}")

    def __len__(self) -> int:
        return len(self.samples)

    def split(self, val_frac: float = 0.15) -> tuple["AbilityDecisionDataset", "AbilityDecisionDataset"]:
        n_val = max(1, int(len(self.samples) * val_frac))
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        val_ds = AbilityDecisionDataset.__new__(AbilityDecisionDataset)
        val_ds.tokenizer = self.tokenizer
        val_ds.game_state_dim = self.game_state_dim
        val_ds.samples = [self.samples[i] for i in indices[:n_val]]

        train_ds = AbilityDecisionDataset.__new__(AbilityDecisionDataset)
        train_ds.tokenizer = self.tokenizer
        train_ds.game_state_dim = self.game_state_dim
        train_ds.samples = [self.samples[i] for i in indices[n_val:]]

        print(f"Split: {len(train_ds)} train, {len(val_ds)} val")
        return train_ds, val_ds

    def make_batch(self, indices: list[int]) -> dict[str, torch.Tensor]:
        """Create a batch from sample indices."""
        batch_samples = [self.samples[i] for i in indices]

        # Tokenize DSL texts
        texts = [s["ability_dsl"] for s in batch_samples]
        ids_list, mask_list = self.tokenizer.batch_encode(texts, add_cls=True)

        # Game state features
        game_states = [s.get("game_state", [0.0] * self.game_state_dim) for s in batch_samples]

        # Labels
        urgencies = [s["urgency"] for s in batch_samples]
        target_idxs = [s.get("target_idx", -1) for s in batch_samples]
        has_target = [1.0 if s.get("category", "") in UNIT_TARGET_CATEGORIES and s.get("target_idx", -1) >= 0
                      else 0.0 for s in batch_samples]

        return {
            "input_ids": torch.tensor(ids_list, dtype=torch.long, device=DEVICE),
            "attention_mask": torch.tensor(mask_list, dtype=torch.float, device=DEVICE),
            "game_state": torch.tensor(game_states, dtype=torch.float, device=DEVICE),
            "urgency": torch.tensor(urgencies, dtype=torch.float, device=DEVICE),
            "target_idx": torch.tensor([max(0, t) for t in target_idxs], dtype=torch.long, device=DEVICE),
            "has_target": torch.tensor(has_target, dtype=torch.float, device=DEVICE),
        }

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = random.choices(range(len(self.samples)), k=batch_size)
        return self.make_batch(indices)

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        """Iterate over all samples in batches."""
        indices = list(range(len(self.samples)))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) > 0:
                yield self.make_batch(batch_indices)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_loss(
    pred_urgency: torch.Tensor,
    target_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined urgency + target loss."""
    pred_u = pred_urgency.squeeze(-1)
    true_u = batch["urgency"]
    has_t = batch["has_target"]

    # Urgency: MSE loss
    urgency_loss = F.mse_loss(pred_u, true_u)

    # Target: cross-entropy (only for unit-targeted samples)
    target_loss = torch.tensor(0.0, device=pred_u.device)
    if has_t.sum() > 0:
        mask = has_t.bool()
        target_loss = F.cross_entropy(
            target_logits[mask],
            batch["target_idx"][mask],
        )

    total = urgency_loss + 0.5 * target_loss

    return total, {
        "total": total.item(),
        "urgency": urgency_loss.item(),
        "target": target_loss.item(),
    }


def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AbilityTokenizer(max_length=args.max_seq_len)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load dataset
    dataset = AbilityDecisionDataset(Path(args.dataset), tokenizer)
    train_ds, val_ds = dataset.split(val_frac=args.val_frac)

    # Model
    model = AbilityTransformerDecision(
        vocab_size=tokenizer.vocab_size,
        game_state_dim=dataset.game_state_dim,
        n_targets=3,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        pad_id=tokenizer.pad_id,
        cls_id=tokenizer.cls_id,
    ).to(DEVICE)

    # Load pre-trained transformer weights
    if args.pretrained:
        pretrained_path = Path(args.pretrained)
        if pretrained_path.exists():
            print(f"Loading pre-trained weights from {pretrained_path}")
            pretrained_state = torch.load(pretrained_path, map_location=DEVICE, weights_only=True)

            # Map MLM model keys to decision model keys
            mapped = {}
            for k, v in pretrained_state.items():
                if k.startswith("transformer."):
                    new_k = k  # AbilityTransformerDecision has .transformer
                    mapped[new_k] = v
                elif k.startswith("mlm_head."):
                    continue  # Skip MLM head
                else:
                    mapped[k] = v

            missing, unexpected = model.load_state_dict(mapped, strict=False)
            print(f"  Loaded {len(mapped)} params, missing {len(missing)}, unexpected {len(unexpected)}")
            if missing:
                print(f"  Missing (expected — new heads): {missing[:5]}...")
        else:
            print(f"Warning: pretrained checkpoint {pretrained_path} not found, training from scratch")

    # Load pre-trained entity encoder weights (from outcome prediction)
    if args.entity_encoder and hasattr(model, "entity_encoder"):
        ee_path = Path(args.entity_encoder)
        if ee_path.exists():
            print(f"Loading pre-trained entity encoder from {ee_path}")
            ee_state = torch.load(ee_path, map_location=DEVICE, weights_only=True)

            # Map pretrain_entity.py keys → model.entity_encoder keys
            mapped_ee = {}
            for k, v in ee_state.items():
                if k.startswith(("win_head.", "hp_head.")):
                    continue  # Skip prediction heads
                mapped_ee[f"entity_encoder.{k}"] = v

            missing, unexpected = model.load_state_dict(mapped_ee, strict=False)
            loaded = len(mapped_ee)
            print(f"  Loaded {loaded} entity encoder params, missing {len(missing)}, unexpected {len(unexpected)}")
        else:
            print(f"Warning: entity encoder {ee_path} not found")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer — grokking plan §2.1
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps,
    )

    # Batch size — grokking plan §2.4
    batch_size = min(args.batch_size, len(train_ds) // 2) if len(train_ds) > 4 else len(train_ds)
    print(f"Batch size: {batch_size}")

    # Metrics log
    log_path = Path(args.output).with_suffix(".csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "step", "train_loss", "urgency_loss", "target_loss",
        "val_urgency_mae", "val_target_acc", "oracle_agreement",
        "weight_norm", "grad_norm", "max_eigenvalue", "lr", "elapsed_s",
    ])

    # Import oracle agreement metric
    from eval.oracle_agreement import compute_oracle_agreement

    # Grokfast EMA gradient filter
    gf = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    # Training loop — run full budget, checkpoint on best urgency MAE.
    # Grokking: expect flat plateau then sudden phase transition.
    # Do NOT early stop — the transition may happen very late.
    best_metric = float("inf")  # lower is better
    start_time = time.time()
    model.train()

    print(f"\nStarting fine-tuning: max_steps={args.max_steps}")
    print(f"Weight decay={args.weight_decay}, lr={args.lr}")
    print(f"Grokfast EMA: alpha={args.grokfast_alpha}, lamb={args.grokfast_lamb}")
    print(f"Device: {DEVICE}\n")

    for step in range(1, args.max_steps + 1):
        batch = train_ds.sample_batch(batch_size)
        pred_urgency, target_logits = model(
            batch["input_ids"], batch["attention_mask"], batch["game_state"],
        )
        loss, loss_parts = compute_loss(pred_urgency, target_logits, batch)

        optimizer.zero_grad()
        loss.backward()
        gf.step()  # Grokfast: amplify slow gradient components
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step <= args.warmup_steps:
            warmup_scheduler.step()

        # Evaluation
        if step % args.eval_every == 0:
            model.eval()

            # Oracle agreement on validation set
            val_batches = list(val_ds.iter_batches(batch_size, shuffle=False))
            metrics = compute_oracle_agreement(model, val_batches, DEVICE)

            weight_norm = sum(
                p.data.norm().item() ** 2 for p in model.parameters()
            ) ** 0.5

            # Spectral monitoring (anti-grokking detection)
            max_eig = 0.0
            for p in model.parameters():
                if p.ndim == 2 and p.shape[0] >= 4 and p.shape[1] >= 4:
                    try:
                        s = torch.linalg.svdvals(p.data)
                        max_eig = max(max_eig, s[0].item())
                    except Exception:
                        pass

            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]

            urgency_mae = metrics["urgency_mae"]
            urgency_corr = metrics.get("urgency_corr", 0.0)

            log_writer.writerow([
                step, f"{loss_parts['total']:.6f}",
                f"{loss_parts['urgency']:.6f}", f"{loss_parts['target']:.6f}",
                f"{urgency_mae:.4f}", f"{metrics['target_acc']:.4f}",
                f"{metrics['oracle_agreement']:.4f}",
                f"{weight_norm:.4f}", f"{grad_norm:.4f}", f"{max_eig:.4f}", f"{lr:.6f}",
                f"{elapsed:.1f}",
            ])
            log_file.flush()

            # Checkpoint on best urgency MAE (no early stopping)
            marker = ""
            if urgency_mae < best_metric:
                best_metric = urgency_mae
                torch.save(model.state_dict(), args.output)
                marker = " *"

            print(
                f"step {step:>7d} | "
                f"loss {loss_parts['total']:.4f} (u={loss_parts['urgency']:.4f} t={loss_parts['target']:.4f}) | "
                f"val_mae {urgency_mae:.4f} | "
                f"corr {urgency_corr:.4f} | "
                f"w_norm {weight_norm:.1f}"
                f"{marker}"
            )

            model.train()

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step} steps in {elapsed:.0f}s")
    print(f"Best urgency MAE: {best_metric:.4f}")
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {log_path}")


def main():
    p = argparse.ArgumentParser(description="Phase 2: Ability transformer fine-tuning")
    p.add_argument("dataset", help="JSONL dataset with ability_dsl + game_state + urgency")
    p.add_argument("--pretrained", help="Pre-trained checkpoint (Phase 1 output)")
    p.add_argument("-o", "--output", default="generated/ability_transformer_decision.pt")

    # Grokking settings
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--val-frac", type=float, default=0.15)

    # Architecture — 4 layers per Murty et al. (structural grokking)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--entity-encoder", help="Pre-trained entity encoder checkpoint (.pt)")

    # Grokfast (Lee et al., 2405.20233)
    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
