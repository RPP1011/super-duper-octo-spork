#!/usr/bin/env python3
"""Pre-train entity encoder on fight outcome prediction.

Learns rich entity representations by predicting who wins from game state
snapshots. Uses self-attention over entity tokens so the model can learn
inter-entity relationships (threat assessment, terrain interactions, focus
fire potential).

Architecture:
    Entity features (7×30) → Linear proj → + type embeddings
    → Self-attention over entities → [GAME] pooled embedding
    → Outcome head (win probability + HP remaining)

Dataset format (JSONL):
    {
        "game_state": [0.5, 0.3, ...],   // 210 floats (7 entities × 30 features)
        "hero_wins": 1.0,
        "hero_hp_remaining": 0.45,
        "fight_progress": 0.3
    }

Usage:
    uv run --with numpy --with torch training/pretrain_entity.py \
        generated/outcome_dataset.jsonl \
        -o generated/entity_encoder_pretrained.pt \
        --max-steps 300000
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from grokfast import GrokfastEMA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENTITY_DIM = 30
NUM_ENTITIES = 7
NUM_TYPES = 3  # self=0, enemy=1, ally=2
TYPE_IDS = [0, 1, 1, 1, 2, 2, 2]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EntityEncoderPretraining(nn.Module):
    """Entity encoder with self-attention + outcome prediction head.

    The encoder portion (proj, type_emb, self-attention, norm) can be
    extracted after pre-training and frozen for cross-attention in Phase 2.
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model

        # Entity projection + type embeddings
        self.proj = nn.Linear(ENTITY_DIM, d_model)
        self.type_emb = nn.Embedding(NUM_TYPES, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Self-attention over entities
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        # Outcome prediction heads
        self.win_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.hp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Reduced init scale (3x smaller) per Kumar et al. (2310.06110)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.007)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_entities(
        self, game_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode game state into entity tokens.

        Parameters
        ----------
        game_state : (batch, 210)

        Returns
        -------
        entity_tokens : (batch, 7, d_model)
        entity_mask : (batch, 7) — True where entity doesn't exist
        """
        B = game_state.shape[0]
        device = game_state.device

        entities = game_state.view(B, NUM_ENTITIES, ENTITY_DIM)

        tokens = self.proj(entities)
        type_ids = torch.tensor(TYPE_IDS, device=device, dtype=torch.long)
        tokens = tokens + self.type_emb(type_ids).unsqueeze(0)
        tokens = self.input_norm(tokens)

        # Entity mask: exists feature is index 29
        exists = entities[:, :, 29]
        entity_mask = exists < 0.5  # True = ignore

        tokens = self.encoder(tokens, src_key_padding_mask=entity_mask)
        tokens = self.out_norm(tokens)

        return tokens, entity_mask

    def forward(self, game_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict fight outcome from game state.

        Returns
        -------
        win_logit : (batch, 1)
        hp_pred : (batch, 1) — predicted hero HP remaining
        """
        tokens, mask = self.encode_entities(game_state)

        # Pool: mean over existing entities (mask=True means ignore)
        exist_mask = (~mask).float().unsqueeze(-1)  # (B, 7, 1)
        pooled = (tokens * exist_mask).sum(dim=1) / exist_mask.sum(dim=1).clamp(min=1)

        return self.win_head(pooled), self.hp_head(pooled)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OutcomeDataset:
    def __init__(self, path: Path):
        self.samples: list[dict] = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {path}")

        wins = sum(1 for s in self.samples if s["hero_wins"] > 0.5)
        losses = len(self.samples) - wins
        print(f"  {wins} win snapshots, {losses} loss snapshots ({wins/(wins+losses)*100:.1f}% win)")

        gs_dim = len(self.samples[0]["game_state"])
        print(f"  game_state dim: {gs_dim}")

    def __len__(self) -> int:
        return len(self.samples)

    def split(self, val_frac: float = 0.15) -> tuple["OutcomeDataset", "OutcomeDataset"]:
        n_val = max(1, int(len(self.samples) * val_frac))
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        val_ds = OutcomeDataset.__new__(OutcomeDataset)
        val_ds.samples = [self.samples[i] for i in indices[:n_val]]

        train_ds = OutcomeDataset.__new__(OutcomeDataset)
        train_ds.samples = [self.samples[i] for i in indices[n_val:]]

        print(f"Split: {len(train_ds)} train, {len(val_ds)} val")
        return train_ds, val_ds

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = random.choices(range(len(self.samples)), k=batch_size)
        batch = [self.samples[i] for i in indices]
        return {
            "game_state": torch.tensor([s["game_state"] for s in batch], dtype=torch.float, device=DEVICE),
            "hero_wins": torch.tensor([s["hero_wins"] for s in batch], dtype=torch.float, device=DEVICE),
            "hero_hp": torch.tensor([s["hero_hp_remaining"] for s in batch], dtype=torch.float, device=DEVICE),
            "progress": torch.tensor([s["fight_progress"] for s in batch], dtype=torch.float, device=DEVICE),
        }

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        indices = list(range(len(self.samples)))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            if batch_idx:
                batch = [self.samples[j] for j in batch_idx]
                yield {
                    "game_state": torch.tensor([s["game_state"] for s in batch], dtype=torch.float, device=DEVICE),
                    "hero_wins": torch.tensor([s["hero_wins"] for s in batch], dtype=torch.float, device=DEVICE),
                    "hero_hp": torch.tensor([s["hero_hp_remaining"] for s in batch], dtype=torch.float, device=DEVICE),
                    "progress": torch.tensor([s["fight_progress"] for s in batch], dtype=torch.float, device=DEVICE),
                }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = OutcomeDataset(Path(args.dataset))
    train_ds, val_ds = dataset.split(val_frac=args.val_frac)

    model = EntityEncoderPretraining(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10,
    )

    batch_size = min(args.batch_size, len(train_ds) // 2) if len(train_ds) > 4 else len(train_ds)

    # Grokfast EMA gradient filter
    gf = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    log_path = Path(args.output).with_suffix(".csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "step", "train_loss", "win_loss", "hp_loss",
        "val_win_acc", "val_hp_mae", "weight_norm", "grad_norm", "max_eigenvalue", "lr", "elapsed_s",
    ])

    best_metric = 0.0  # win accuracy, higher is better
    start_time = time.time()
    model.train()

    print(f"\nStarting entity encoder pre-training: max_steps={args.max_steps}")
    print(f"Weight decay={args.weight_decay}, lr={args.lr}")
    print(f"Grokfast EMA: alpha={args.grokfast_alpha}, lamb={args.grokfast_lamb}")
    print(f"Device: {DEVICE}\n")

    for step in range(1, args.max_steps + 1):
        batch = train_ds.sample_batch(batch_size)
        win_logit, hp_pred = model(batch["game_state"])

        win_loss = F.binary_cross_entropy_with_logits(
            win_logit.squeeze(-1), batch["hero_wins"]
        )
        hp_loss = F.mse_loss(hp_pred.squeeze(-1), batch["hero_hp"])
        loss = win_loss + hp_loss

        optimizer.zero_grad()
        loss.backward()
        gf.step()  # Grokfast: amplify slow gradient components
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step <= 10:
            warmup.step()

        if step % args.eval_every == 0:
            model.eval()

            # Evaluate on val set
            total, correct, hp_errors = 0, 0, []
            for vb in val_ds.iter_batches(batch_size, shuffle=False):
                with torch.no_grad():
                    vw, vh = model(vb["game_state"])
                pred_win = (vw.squeeze(-1) > 0).float()
                correct += (pred_win == vb["hero_wins"]).sum().item()
                hp_errors.append((vh.squeeze(-1) - vb["hero_hp"]).abs())
                total += len(vb["hero_wins"])

            val_win_acc = correct / max(total, 1)
            val_hp_mae = torch.cat(hp_errors).mean().item()
            weight_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5

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

            log_writer.writerow([
                step, f"{loss.item():.6f}", f"{win_loss.item():.6f}", f"{hp_loss.item():.6f}",
                f"{val_win_acc:.4f}", f"{val_hp_mae:.4f}",
                f"{weight_norm:.4f}", f"{grad_norm:.4f}", f"{max_eig:.4f}", f"{lr:.6f}",
                f"{elapsed:.1f}",
            ])
            log_file.flush()

            marker = ""
            if val_win_acc > best_metric:
                best_metric = val_win_acc
                torch.save(model.state_dict(), args.output)
                marker = " *"

            print(
                f"step {step:>7d} | "
                f"loss {loss.item():.4f} (win={win_loss.item():.4f} hp={hp_loss.item():.4f}) | "
                f"val_acc {val_win_acc:.4f} | hp_mae {val_hp_mae:.4f} | "
                f"w_norm {weight_norm:.1f} | eig {max_eig:.2f}"
                f"{marker}"
            )

            model.train()

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step} steps in {elapsed:.0f}s")
    print(f"Best val win accuracy: {best_metric:.4f}")
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {log_path}")


def main():
    p = argparse.ArgumentParser(description="Pre-train entity encoder on outcome prediction")
    p.add_argument("dataset", help="JSONL dataset with game_state + hero_wins")
    p.add_argument("-o", "--output", default="generated/entity_encoder_pretrained.pt")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=300_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.15)

    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)

    # Grokfast (Lee et al., 2405.20233)
    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
