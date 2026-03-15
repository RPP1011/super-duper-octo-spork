#!/usr/bin/env python3
"""Data quality probes for V5 Stage 0a training data.

Validates entity dims, threat dims, aggregate features, label distributions,
and feature-label correlations before training.

Usage:
    uv run --with numpy python training/probe_v5_data.py generated/v5_stage0a_random.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: probe_v5_data.py <episodes.jsonl> [episodes2.jsonl ...]")
        sys.exit(1)

    episodes = []
    for path in sys.argv[1:]:
        with open(path) as f:
            for line in f:
                episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes from {len(sys.argv)-1} file(s)")

    # --- Episode-level stats ---
    outcomes = Counter(ep["outcome"] for ep in episodes)
    rewards = [ep["reward"] for ep in episodes]
    print(f"\nOutcome distribution: {dict(outcomes)}")
    print(f"Reward: mean={np.mean(rewards):.3f} std={np.std(rewards):.3f} min={np.min(rewards):.3f} max={np.max(rewards):.3f}")

    if outcomes.get("Victory", 0) == 0 or outcomes.get("Defeat", 0) == 0:
        print("WARNING: Only one outcome class — model cannot learn to discriminate")

    # --- Step-level stats ---
    all_steps = [s for ep in episodes for s in ep["steps"]]
    print(f"\nTotal steps: {len(all_steps)}")

    # Entity dimension check
    ent_dims = set()
    thr_dims = set()
    n_with_agg = 0
    n_with_threats = 0
    agg_values = []

    for s in all_steps:
        if s.get("entities"):
            for e in s["entities"]:
                ent_dims.add(len(e))
        if s.get("threats"):
            for t in s["threats"]:
                thr_dims.add(len(t))
            if len(s["threats"]) > 0:
                n_with_threats += 1
        if s.get("aggregate_features"):
            n_with_agg += 1
            agg_values.append(s["aggregate_features"])

    print(f"\nEntity feature dims: {ent_dims}")
    print(f"Threat feature dims: {thr_dims}")
    assert 34 in ent_dims or len(ent_dims) == 0, f"Expected 34-dim entities, got {ent_dims}"
    assert 10 in thr_dims or len(thr_dims) == 0, f"Expected 10-dim threats, got {thr_dims}"

    print(f"Steps with threats: {n_with_threats}/{len(all_steps)} ({100*n_with_threats/max(len(all_steps),1):.1f}%)")
    print(f"Steps with aggregate: {n_with_agg}/{len(all_steps)} ({100*n_with_agg/max(len(all_steps),1):.1f}%)")

    # --- Spatial feature probe ---
    spatial_nonzero = 0
    for s in all_steps:
        if s.get("entities"):
            for e in s["entities"]:
                if len(e) >= 34 and any(e[i] != 0 for i in range(30, 34)):
                    spatial_nonzero += 1
                    break
    print(f"Steps with nonzero spatial features: {spatial_nonzero}/{len(all_steps)}")
    if spatial_nonzero == 0:
        print("NOTE: Spatial features are all zero — extract_game_state_v2_spatial() not called yet")

    # --- Threat kind distribution ---
    kind_counts = Counter()
    for s in all_steps:
        if s.get("threats"):
            for t in s["threats"]:
                if len(t) >= 10:
                    kind = t[8]
                    if kind == 0.25: kind_counts["zone"] += 1
                    elif kind == 0.5: kind_counts["obstacle"] += 1
                    elif kind == 0.75: kind_counts["cast"] += 1
                    elif kind == 1.0: kind_counts["projectile"] += 1
                    else: kind_counts[f"unknown({kind})"] += 1
    print(f"Threat kind distribution: {dict(kind_counts)}")

    # --- Feature correlation with outcome ---
    if len(all_steps) > 100:
        print("\n--- Feature-label correlation probe ---")
        # Sample entity features for self (index 0)
        self_feats = []
        labels = []
        ep_map = {}  # step -> episode outcome
        step_idx = 0
        for ep in episodes:
            for s in ep["steps"]:
                ep_map[step_idx] = 1.0 if ep["outcome"] == "Victory" else 0.0
                step_idx += 1

        for i, s in enumerate(all_steps):
            if s.get("entities") and len(s["entities"]) > 0:
                self_feats.append(s["entities"][0][:30])  # base features only
                labels.append(ep_map.get(i, 0.0))

        if len(self_feats) > 50:
            X = np.array(self_feats)
            y = np.array(labels)

            # Spearman rank correlation per feature
            from scipy.stats import spearmanr
            feature_names = [
                "hp%", "shield%", "resource%", "armor", "mr",
                "pos_x", "pos_y", "dist", "cover", "elevation",
                "hostile_zones", "friendly_zones",
                "auto_dps", "atk_range", "atk_cd",
                "abil_dmg", "abil_range", "abil_cd",
                "heal_amt", "heal_range", "heal_cd",
                "ctrl_range", "ctrl_dur", "ctrl_cd",
                "is_casting", "cast_prog", "cc_remaining", "move_speed",
                "total_dmg_done", "exists",
            ]
            correlations = []
            for j in range(min(X.shape[1], 30)):
                if X[:, j].std() > 1e-8:
                    rho, _ = spearmanr(X[:, j], y)
                    correlations.append((feature_names[j] if j < len(feature_names) else f"f{j}", rho))

            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            print("Top 10 correlated features with outcome:")
            for name, rho in correlations[:10]:
                print(f"  {name:20s}: rho={rho:+.3f}")

            max_corr = max(abs(c[1]) for c in correlations) if correlations else 0
            if max_corr > 0.95:
                print(f"WARNING: Feature with |rho|={max_corr:.3f} — possible label leak")
            if max_corr < 0.05:
                print("WARNING: No feature correlates with outcome — features may carry no signal")

    # --- Duplicate check ---
    gs_hashes = Counter()
    for s in all_steps:
        h = hash(tuple(s["game_state"][:30]))  # hash first 30 features
        gs_hashes[h] += 1
    n_dupes = sum(c - 1 for c in gs_hashes.values() if c > 1)
    print(f"\nDuplicate game states: {n_dupes}/{len(all_steps)} ({100*n_dupes/max(len(all_steps),1):.1f}%)")

    print("\n--- Probe complete ---")


if __name__ == "__main__":
    main()
