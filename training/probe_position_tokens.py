#!/usr/bin/env python3
"""Probe position token features from generated episodes.

Analyzes the 8-dim position tokens extracted by game_state.rs:
  0: dx from self / 20
  1: dy from self / 20
  2: path distance / 30
  3: elevation / 5
  4: chokepoint_score / 3
  5: wall_proximity / 5
  6: n_hostile_zones / 3
  7: n_friendly_zones / 3

Usage:
  uv run --with numpy training/probe_position_tokens.py \
      --episodes generated/some_episodes.jsonl

  # Or generate fresh from scenarios:
  uv run --with numpy training/probe_position_tokens.py \
      --scenarios dataset/scenarios/ --threads 8
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "dx/20", "dy/20", "path_dist/30", "elevation/5",
    "chokepoint/3", "wall_prox/5", "hostile_zones/3", "friendly_zones/3",
]

FEATURE_SCALES = [20.0, 20.0, 30.0, 5.0, 3.0, 5.0, 3.0, 3.0]


def load_episodes(path: str) -> list:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def generate_episodes(scenarios_dir: str, threads: int = 8,
                      episodes_per_scenario: int = 1,
                      max_scenarios: int = 50) -> list:
    """Generate episodes with position token data using transformer-rl."""
    outdir = tempfile.mkdtemp(prefix="probe_pos_")
    outfile = Path(outdir) / "episodes.jsonl"

    cmd = [
        "cargo", "run", "--release", "--bin", "xtask", "--",
        "scenario", "oracle", "transformer-rl", "generate",
        "--scenarios", scenarios_dir,
        "--weights", "generated/ability_transformer_weights_v2.json",
        "--embedding-registry", "generated/ability_embedding_registry.json",
        "--policy", "transformer",
        "--episodes-per-scenario", str(episodes_per_scenario),
        "--threads", str(threads),
        "--output", str(outfile),
        "--v3",  # needed for position tokens (GridNav)
    ]

    print(f"Generating episodes from {scenarios_dir}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Generation failed:\n{result.stderr[:500]}")
        sys.exit(1)

    return load_episodes(str(outfile))


def extract_position_data(episodes: list) -> dict:
    """Extract position token arrays from episode steps."""
    all_tokens = []       # (N, 8) — all position tokens
    tokens_per_step = []  # how many position tokens per step
    step_count = 0
    steps_with_positions = 0

    for ep in episodes:
        for step in ep.get("steps", []):
            step_count += 1
            positions = step.get("positions", [])
            n_pos = len(positions)
            tokens_per_step.append(n_pos)
            if n_pos > 0:
                steps_with_positions += 1
            for pos in positions:
                if len(pos) == 8:
                    all_tokens.append(pos)

    return {
        "tokens": np.array(all_tokens) if all_tokens else np.zeros((0, 8)),
        "tokens_per_step": np.array(tokens_per_step),
        "step_count": step_count,
        "steps_with_positions": steps_with_positions,
    }


def analyze_features(data: dict):
    """Print statistics about position token features."""
    tokens = data["tokens"]
    n = len(tokens)

    print("=" * 70)
    print("Position Token Feature Probe")
    print("=" * 70)
    print(f"Total steps:            {data['step_count']}")
    print(f"Steps with positions:   {data['steps_with_positions']} "
          f"({data['steps_with_positions']/max(data['step_count'],1)*100:.1f}%)")
    print(f"Total position tokens:  {n}")

    if n == 0:
        print("\nNo position tokens found! Position tokens require GridNav (--v3 flag).")
        return

    tps = data["tokens_per_step"]
    tps_nonzero = tps[tps > 0]
    print(f"Tokens per step:        mean={tps.mean():.1f}, "
          f"median={np.median(tps):.0f}, "
          f"min={tps.min()}, max={tps.max()}")
    if len(tps_nonzero) > 0:
        print(f"  (when present):       mean={tps_nonzero.mean():.1f}, "
              f"median={np.median(tps_nonzero):.0f}")

    print()
    print("-" * 70)
    print(f"{'Feature':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} "
          f"{'P5':>8} {'P95':>8} {'Raw Mean':>10}")
    print("-" * 70)

    for i, (name, scale) in enumerate(zip(FEATURE_NAMES, FEATURE_SCALES)):
        col = tokens[:, i]
        raw_mean = col.mean() * scale
        print(f"{name:<20} {col.mean():>8.4f} {col.std():>8.4f} "
              f"{col.min():>8.4f} {col.max():>8.4f} "
              f"{np.percentile(col, 5):>8.4f} {np.percentile(col, 95):>8.4f} "
              f"{raw_mean:>10.2f}")

    # Direction analysis
    print()
    print("-" * 70)
    print("Direction Distribution (dx, dy)")
    print("-" * 70)

    dx = tokens[:, 0] * 20.0
    dy = tokens[:, 1] * 20.0
    angles = np.arctan2(dy, dx) * 180 / np.pi
    dists = np.sqrt(dx**2 + dy**2)

    # Bin into 8 compass directions
    bins = {"N": 0, "NE": 0, "E": 0, "SE": 0,
            "S": 0, "SW": 0, "W": 0, "NW": 0}
    for a in angles:
        if -22.5 <= a < 22.5:
            bins["E"] += 1
        elif 22.5 <= a < 67.5:
            bins["NE"] += 1
        elif 67.5 <= a < 112.5:
            bins["N"] += 1
        elif 112.5 <= a < 157.5:
            bins["NW"] += 1
        elif a >= 157.5 or a < -157.5:
            bins["W"] += 1
        elif -157.5 <= a < -112.5:
            bins["SW"] += 1
        elif -112.5 <= a < -67.5:
            bins["S"] += 1
        elif -67.5 <= a < -22.5:
            bins["SE"] += 1

    for direction, count in bins.items():
        bar = "#" * int(count / n * 80)
        print(f"  {direction:>3}: {count:>5} ({count/n*100:5.1f}%) {bar}")

    print(f"\n  Distance: mean={dists.mean():.2f}, "
          f"min={dists.min():.2f}, max={dists.max():.2f}")

    # Path distance vs euclidean analysis
    print()
    print("-" * 70)
    print("Path Distance Analysis")
    print("-" * 70)

    path_dist = tokens[:, 2] * 30.0
    euclidean = dists
    mask = euclidean > 0.1
    if mask.sum() > 0:
        ratio = path_dist[mask] / euclidean[mask]
        print(f"  Path/Euclidean ratio: mean={ratio.mean():.3f}, "
              f"std={ratio.std():.3f}")
        blocked = (ratio > 1.1).sum()
        print(f"  Line-of-sight blocked: {blocked}/{mask.sum()} "
              f"({blocked/mask.sum()*100:.1f}%)")

    # Terrain analysis
    print()
    print("-" * 70)
    print("Terrain Features")
    print("-" * 70)

    elevation = tokens[:, 3] * 5.0
    chokepoint = tokens[:, 4] * 3.0
    wall_prox = tokens[:, 5] * 5.0
    hostile = tokens[:, 6] * 3.0
    friendly = tokens[:, 7] * 3.0

    print(f"  Elevation:       mean={elevation.mean():.2f}, "
          f"nonzero={np.count_nonzero(elevation)}/{n} "
          f"({np.count_nonzero(elevation)/n*100:.1f}%)")
    print(f"  Chokepoint:      mean={chokepoint.mean():.2f}, "
          f"nonzero={np.count_nonzero(chokepoint)}/{n} "
          f"({np.count_nonzero(chokepoint)/n*100:.1f}%)")
    print(f"  Wall proximity:  mean={wall_prox.mean():.2f}, "
          f"P5={np.percentile(wall_prox, 5):.2f}, "
          f"P50={np.percentile(wall_prox, 50):.2f}")
    print(f"  Hostile zones:   nonzero={np.count_nonzero(hostile)}/{n} "
          f"({np.count_nonzero(hostile)/n*100:.1f}%)")
    print(f"  Friendly zones:  nonzero={np.count_nonzero(friendly)}/{n} "
          f"({np.count_nonzero(friendly)/n*100:.1f}%)")

    # Correlation matrix
    print()
    print("-" * 70)
    print("Feature Correlations (|r| > 0.3)")
    print("-" * 70)

    corr = np.corrcoef(tokens.T)
    for i in range(8):
        for j in range(i + 1, 8):
            r = corr[i, j]
            if abs(r) > 0.3:
                print(f"  {FEATURE_NAMES[i]:>20} × {FEATURE_NAMES[j]:<20} r={r:+.3f}")

    # Feature variance analysis — which features carry signal?
    print()
    print("-" * 70)
    print("Signal Analysis (variance)")
    print("-" * 70)
    variances = tokens.var(axis=0)
    total_var = variances.sum()
    for i, (name, v) in enumerate(zip(FEATURE_NAMES, variances)):
        bar = "#" * int(v / total_var * 60)
        print(f"  {name:<20} var={v:.5f} ({v/total_var*100:5.1f}%) {bar}")


def main():
    parser = argparse.ArgumentParser(description="Probe position token features")
    parser.add_argument("--episodes", type=str,
                        help="Path to episodes JSONL file")
    parser.add_argument("--scenarios", type=str,
                        help="Generate episodes from scenarios directory")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--episodes-per-scenario", type=int, default=1)
    args = parser.parse_args()

    if args.episodes:
        print(f"Loading episodes from {args.episodes}...")
        episodes = load_episodes(args.episodes)
    elif args.scenarios:
        episodes = generate_episodes(args.scenarios, args.threads,
                                     args.episodes_per_scenario)
    else:
        parser.error("Provide --episodes or --scenarios")

    print(f"Loaded {len(episodes)} episodes")

    data = extract_position_data(episodes)
    analyze_features(data)


if __name__ == "__main__":
    main()
