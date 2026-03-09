#!/usr/bin/env python3
"""Plot next-state prediction training curves from CSV log.

Usage:
    uv run --with matplotlib --with numpy scripts/plot_nextstate.py generated/entity_encoder_nextstate.csv
"""
import sys
import csv
import numpy as np

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "generated/entity_encoder_nextstate.csv"

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data yet.")
        return

    steps = [int(r["step"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_mae = [float(r["val_mae_all"]) for r in rows]
    base_mae = [float(r["baseline_mae_all"]) for r in rows]

    groups = ["hp", "pos", "cd", "state", "exists"]
    group_mae = {g: [float(r[f"val_mae_{g}"]) for r in rows] for g in groups}
    group_base = {g: [float(r[f"base_mae_{g}"]) for r in rows] for g in groups}

    # Compute improvement % per group
    group_improv = {}
    for g in groups:
        group_improv[g] = [
            100.0 * (b - v) / b if b > 0 else 0.0
            for v, b in zip(group_mae[g], group_base[g])
        ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Training loss
    ax = axes[0, 0]
    ax.plot(steps, train_loss, "b-", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (beta-NLL)")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # 2. Overall MAE vs baseline
    ax = axes[0, 1]
    ax.plot(steps, val_mae, "b-", linewidth=1.2, label="Model")
    ax.plot(steps, base_mae, "r--", linewidth=1.0, label="Baseline (no change)")
    ax.set_xlabel("Step")
    ax.set_ylabel("MAE")
    ax.set_title("Overall Val MAE vs Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Per-group MAE
    ax = axes[1, 0]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    for g, c in zip(groups, colors):
        ax.plot(steps, group_mae[g], color=c, linewidth=0.8, label=g)
        ax.plot(steps, group_base[g], color=c, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("MAE")
    ax.set_title("Per-Group Val MAE (solid=model, dashed=baseline)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Per-group improvement %
    ax = axes[1, 1]
    for g, c in zip(groups, colors):
        ax.plot(steps, group_improv[g], color=c, linewidth=1.0, label=g)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Step")
    ax.set_ylabel("Improvement over baseline (%)")
    ax.set_title("Per-Group Improvement %")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Next-State Prediction (Decomposed)", fontsize=14)
    plt.tight_layout()

    out_path = path.replace(".csv", ".png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
