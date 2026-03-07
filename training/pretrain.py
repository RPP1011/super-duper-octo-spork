#!/usr/bin/env python3
"""Phase 1: Masked token pre-training for the ability transformer.

Trains the transformer to predict randomly masked tokens in ability DSL
sequences.  Uses grokking-informed settings:
  - AdamW with weight_decay=1.0, betas=(0.9, 0.98)
  - No dropout (regularization via weight decay only)
  - Extended training with accuracy-based stopping, not loss-based
  - Minibatch stochasticity as implicit regularization

Usage:
    uv run --with numpy --with torch training/pretrain.py \
        generated/ability_dataset/ \
        -o generated/ability_transformer_pretrained.pt \
        --max-steps 500000

    # With diagnostics:
    uv run --with numpy --with torch --with scikit-learn --with matplotlib \
        training/pretrain.py generated/ability_dataset/ \
        -o generated/ability_transformer_pretrained.pt \
        --diagnostics-dir diagnostics/pretrain/
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
import torch.nn.functional as F

# Add training/ to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from model import AbilityTransformerMLM
from tokenizer import AbilityTokenizer, MASK, PAD
from grokfast import GrokfastEMA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def augment_ability_text(text: str) -> str:
    """Data augmentation: randomly reorder property lines in ability DSL.

    Property lines (target, range, cooldown, cast, hint, cost, charges,
    recharge, recast, recast_window) are order-independent in the DSL.
    Shuffling them creates semantically equivalent training examples that
    help the model learn position-invariant property understanding.
    """
    lines = text.split("\n")
    header_end = -1
    prop_lines = []
    other_lines = []

    # Find the opening brace, then collect property lines vs effect lines
    in_props = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "{" in stripped and not in_props:
            in_props = True
            other_lines.append(line)
            continue

        if in_props and stripped and not stripped.startswith(("deliver", "damage", "heal",
                "shield", "stun", "slow", "knockback", "dash", "buff", "debuff",
                "root", "silence", "fear", "taunt", "pull", "swap", "reflect",
                "lifesteal", "stealth", "blink", "summon", "execute", "blind",
                "resurrect", "when", "}")):
            # Likely a property line (target:, cooldown:, hint:, etc.)
            if ":" in stripped or stripped.startswith(("charges", "recast", "unstoppable", "toggle")):
                prop_lines.append(line)
                if header_end < 0:
                    header_end = i
                continue

        other_lines.append(line)

    if len(prop_lines) > 1:
        random.shuffle(prop_lines)
        # Re-insert property lines after the header
        result = []
        inserted = False
        for line in other_lines:
            result.append(line)
            if "{" in line and not inserted:
                result.extend(prop_lines)
                inserted = True
        return "\n".join(result)

    return text


class AbilityMLMDataset:
    """Loads .ability files, tokenizes, and applies random masking."""

    def __init__(
        self,
        ability_dir: Path,
        tokenizer: AbilityTokenizer,
        mask_prob: float = 0.15,
        holdout_hashes: set[str] | None = None,
        augment: bool = True,
    ):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.augment = augment

        # Load all .ability files
        files = sorted(ability_dir.glob("*.ability"))
        self.texts: list[str] = []
        for f in files:
            text = f.read_text().strip()
            if not text:
                continue
            # Skip holdout abilities if filter provided
            if holdout_hashes and _hash_structure(text) in holdout_hashes:
                continue
            self.texts.append(text)

        print(f"Loaded {len(self.texts)} abilities from {ability_dir}")

        # Pre-tokenize all abilities
        self.encoded: list[list[int]] = []
        for text in self.texts:
            ids = tokenizer.encode(text, add_cls=True)
            if len(ids) > 3:  # skip trivially short
                self.encoded.append(ids)

        print(f"Tokenized {len(self.encoded)} abilities (skipped {len(self.texts) - len(self.encoded)} too short)")

    def __len__(self) -> int:
        return len(self.encoded)

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch with random masking applied."""
        indices = random.choices(range(len(self.encoded)), k=batch_size)
        if self.augment:
            # Re-tokenize with augmented text for ~50% of samples
            seqs = []
            for i in indices:
                if random.random() < 0.5:
                    augmented = augment_ability_text(self.texts[i])
                    seqs.append(self.tokenizer.encode(augmented, add_cls=True))
                else:
                    seqs.append(self.encoded[i])
            return self._make_batch(seqs)
        return self._make_batch([self.encoded[i] for i in indices])

    def _make_batch(self, sequences: list[list[int]]) -> dict[str, torch.Tensor]:
        """Create masked batch from pre-tokenized sequences."""
        max_len = min(max(len(s) for s in sequences), self.tokenizer.max_length)

        input_ids = []
        labels = []
        attention_masks = []

        pad_id = self.tokenizer.pad_id
        mask_id = self.tokenizer.mask_id
        vocab_size = self.tokenizer.vocab_size

        for seq in sequences:
            seq = seq[:max_len]
            orig = list(seq)
            masked = list(seq)
            label = [-100] * len(seq)  # -100 = ignore in CE loss

            for i in range(1, len(seq)):  # skip [CLS] at position 0
                if random.random() < self.mask_prob:
                    label[i] = orig[i]
                    r = random.random()
                    if r < 0.8:
                        masked[i] = mask_id
                    elif r < 0.9:
                        masked[i] = random.randint(0, vocab_size - 1)
                    # else: keep original (10%)

            # Pad
            pad_len = max_len - len(seq)
            masked += [pad_id] * pad_len
            label += [-100] * pad_len
            attn = [1] * len(seq) + [0] * pad_len

            input_ids.append(masked)
            labels.append(label)
            attention_masks.append(attn)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=DEVICE),
            "labels": torch.tensor(labels, dtype=torch.long, device=DEVICE),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.float, device=DEVICE),
        }

    def split(self, val_frac: float = 0.15) -> tuple["AbilityMLMDataset", "AbilityMLMDataset"]:
        """Split into train/val datasets. Returns (train, val)."""
        n_val = max(1, int(len(self.encoded) * val_frac))
        indices = list(range(len(self.encoded)))
        random.shuffle(indices)

        val_ds = AbilityMLMDataset.__new__(AbilityMLMDataset)
        val_ds.tokenizer = self.tokenizer
        val_ds.mask_prob = self.mask_prob
        val_ds.augment = False  # No augmentation on val set
        val_ds.texts = [self.texts[i] for i in indices[:n_val]]
        val_ds.encoded = [self.encoded[i] for i in indices[:n_val]]

        train_ds = AbilityMLMDataset.__new__(AbilityMLMDataset)
        train_ds.tokenizer = self.tokenizer
        train_ds.mask_prob = self.mask_prob
        train_ds.augment = self.augment
        train_ds.texts = [self.texts[i] for i in indices[n_val:]]
        train_ds.encoded = [self.encoded[i] for i in indices[n_val:]]

        print(f"Split: {len(train_ds)} train, {len(val_ds)} val")
        return train_ds, val_ds


def _hash_structure(text: str) -> str:
    """Simple hash for holdout exclusion."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: AbilityTransformerMLM,
    dataset: AbilityMLMDataset,
    batch_size: int = 256,
    n_batches: int = 10,
) -> dict[str, float]:
    """Evaluate masked token accuracy and loss on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for _ in range(n_batches):
        batch = dataset.sample_batch(min(batch_size, len(dataset)))
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=-100,
        )
        total_loss += loss.item()

        # Accuracy on masked positions only
        mask_positions = batch["labels"] != -100
        if mask_positions.any():
            preds = logits.argmax(dim=-1)
            correct = (preds == batch["labels"]) & mask_positions
            total_correct += correct.sum().item()
            total_masked += mask_positions.sum().item()

    model.train()

    acc = total_correct / total_masked if total_masked > 0 else 0.0
    return {
        "val_loss": total_loss / n_batches,
        "masked_token_acc": acc,
        "n_masked": total_masked,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def maybe_visualize(
    model: AbilityTransformerMLM,
    dataset: AbilityMLMDataset,
    step: int,
    output_dir: Path,
):
    """Generate t-SNE visualization of [CLS] embeddings if dependencies available."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    embeddings = []
    hints = []

    tok = dataset.tokenizer
    # Use the hint token that follows "hint :" in each ability
    for text in dataset.texts[:200]:  # cap for speed
        ids = tok.encode(text, add_cls=True)
        ids_t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        cls = model.transformer.cls_embedding(ids_t)
        embeddings.append(cls[0].detach().cpu().numpy())

        # Extract hint from text
        hint = "unknown"
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("hint:"):
                hint = line.split(":", 1)[1].strip().split()[0].split(",")[0]
                break
        hints.append(hint)

    model.train()

    if len(embeddings) < 10:
        return

    emb_arr = np.array(embeddings)
    perp = min(30, len(emb_arr) - 1)
    coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(emb_arr)

    label_set = sorted(set(hints))
    colors = {l: i for i, l in enumerate(label_set)}
    c = [colors[h] for h in hints]

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=c, cmap="tab10", s=15, alpha=0.7)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=plt.cm.tab10(colors[l] / max(len(label_set) - 1, 1)),
               markersize=8, label=l) for l in label_set]
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.set_title(f"[CLS] embeddings step {step} (color=hint)")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"cls_hint_{step:07d}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AbilityTokenizer(max_length=args.max_seq_len)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load holdout hashes if provided
    holdout = None
    if args.holdout_hashes and Path(args.holdout_hashes).exists():
        holdout = set(Path(args.holdout_hashes).read_text().strip().split("\n"))
        print(f"Loaded {len(holdout)} holdout hashes")

    # Load dataset
    dataset = AbilityMLMDataset(
        Path(args.ability_dir), tokenizer,
        mask_prob=args.mask_prob, holdout_hashes=holdout,
        augment=not args.no_augment,
    )
    train_ds, val_ds = dataset.split(val_frac=args.val_frac)

    # Model
    model = AbilityTransformerMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        pad_id=tokenizer.pad_id,
        cls_id=tokenizer.cls_id,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer — grokking plan §2.1
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    # Linear warmup — grokking plan §2.1
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps,
    )

    # Batch size — grokking plan §2.4
    batch_size = min(args.batch_size, len(train_ds) // 2) if len(train_ds) > 4 else len(train_ds)
    print(f"Batch size: {batch_size}")

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"Resuming from {resume_path}")
            state = torch.load(resume_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state)
            # Infer step from CSV log
            csv_path = Path(args.output).with_suffix(".csv")
            if csv_path.exists():
                with open(csv_path) as cf:
                    for line in cf:
                        pass
                    try:
                        start_step = int(line.split(",")[0])
                    except (ValueError, UnboundLocalError):
                        pass
            print(f"  Resuming from step {start_step}")

    # Grokfast EMA gradient filter (Lee et al., 2405.20233)
    gf = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)
    print(f"Grokfast EMA: alpha={args.grokfast_alpha}, lamb={args.grokfast_lamb}")

    # Metrics logging
    log_path = Path(args.output).with_suffix(".csv")
    if start_step > 0:
        log_file = open(log_path, "a", newline="")
    else:
        log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    if start_step == 0:
        log_writer.writerow([
            "step", "train_loss", "val_loss", "masked_token_acc",
            "weight_norm", "grad_norm", "lr", "max_eigenvalue", "elapsed_s",
        ])

    # Training — monitor for anti-grokking via spectral diagnostics
    best_acc = 0.0
    start_time = time.time()
    model.train()

    print(f"\nStarting pre-training: max_steps={args.max_steps}")
    print(f"Weight decay={args.weight_decay}, lr={args.lr}")
    print(f"Grokfast + spectral monitoring (anti-grokking detection)")
    print(f"Device: {DEVICE}\n")

    for step in range(start_step + 1, args.max_steps + 1):
        batch = train_ds.sample_batch(batch_size)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch["labels"].view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()

        # Grokfast: amplify slow gradient components before optimizer step
        gf.step()

        # Track gradient norm (diagnostic — grokking plan §4.4)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if step <= args.warmup_steps:
            warmup_scheduler.step()

        # Evaluation
        if step % args.eval_every == 0:
            metrics = evaluate(model, val_ds, batch_size=batch_size)

            # Weight norm (diagnostic — grokking plan §4.4)
            weight_norm = sum(
                p.data.norm().item() ** 2 for p in model.parameters()
            ) ** 0.5

            # Spectral monitoring: track max eigenvalue across weight matrices
            # (anti-grokking detection — Prakash & Martin, 2602.02859)
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
                step, f"{loss.item():.6f}", f"{metrics['val_loss']:.6f}",
                f"{metrics['masked_token_acc']:.4f}",
                f"{weight_norm:.4f}", f"{grad_norm:.4f}", f"{lr:.6f}",
                f"{max_eig:.4f}", f"{elapsed:.1f}",
            ])
            log_file.flush()

            acc = metrics["masked_token_acc"]
            marker = ""
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), args.output)
                marker = " *"

            print(
                f"step {step:>7d} | "
                f"train_loss {loss.item():.4f} | "
                f"val_loss {metrics['val_loss']:.4f} | "
                f"acc {acc:.4f} | "
                f"w_norm {weight_norm:.1f} | "
                f"eig {max_eig:.2f} | "
                f"lr {lr:.6f}"
                f"{marker}"
            )

        # Diagnostics
        if args.diagnostics_dir and step % args.diag_every == 0:
            maybe_visualize(model, val_ds, step, Path(args.diagnostics_dir))

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step} steps in {elapsed:.0f}s")
    print(f"Best masked token accuracy: {best_acc:.4f}")
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {log_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Phase 1: Ability transformer pre-training (MLM)")
    p.add_argument("ability_dir", help="Directory of .ability files")
    p.add_argument("-o", "--output", default="generated/ability_transformer_pretrained.pt")

    # Grokking plan settings
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0, help="High weight decay per grokking plan")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--val-frac", type=float, default=0.15)

    # Architecture — 4 layers per Murty et al. (structural grokking)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=256)

    # Grokfast (Lee et al., 2405.20233) — EMA gradient filter
    p.add_argument("--grokfast-alpha", type=float, default=0.98, help="EMA decay for gradient filter")
    p.add_argument("--grokfast-lamb", type=float, default=2.0, help="Amplification of slow gradient components")

    # Resume
    p.add_argument("--resume", help="Resume from checkpoint (.pt)")
    p.add_argument("--no-augment", action="store_true", help="Disable data augmentation")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-hashes", help="Path to holdout_hashes.txt for exclusion")
    p.add_argument("--diagnostics-dir", help="Directory for t-SNE plots")
    p.add_argument("--diag-every", type=int, default=10_000)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
