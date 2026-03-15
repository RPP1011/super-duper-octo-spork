#!/usr/bin/env python3
"""Curriculum training orchestrator.

Manages multi-phase training with per-phase reward shaping, hyperparameters,
and automatic progression based on eval win rate thresholds.

Usage:
    uv run --with numpy --with torch training/curriculum.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Phase configuration
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    name: str
    scenarios: str                    # scenario directory
    max_iters: int = 30              # max iterations before forced advancement
    advance_win_rate: float = 0.40   # eval win% threshold to advance
    min_iters: int = 5               # minimum iters before checking advancement

    # Hyperparameters
    lr: float = 3e-5
    temperature: float = 1.0
    entropy_coeff: float = 0.01
    kl_coeff: float = 1.0
    reward_scale: float = 1.0
    train_epochs: int = 1

    # Reward shaping (Python-side, applied after Rust step_reward)
    win_bonus: float = 1.0           # bonus added to final step on Victory
    loss_penalty: float = -1.0       # penalty added to final step on Defeat
    timeout_penalty: float = -0.5    # penalty for Timeout
    damage_reward_scale: float = 1.0 # multiplier on HP-differential component
    engagement_bonus: float = 0.0    # bonus per step where damage was dealt
    combo_bonus: float = 0.0         # bonus for damage on CC'd targets (future)

    # Self-play
    self_play_gpu: bool = False
    swap_sides: bool = False
    enemy_weights: str | None = None
    enemy_registry: str | None = None

    # Episode generation
    episodes_per_scenario: int = 5
    eval_every: int = 5

    description: str = ""


def build_curriculum() -> list[PhaseConfig]:
    """Define the full training curriculum."""
    return [
        PhaseConfig(
            name="P1_basics",
            scenarios="dataset/scenarios/curriculum/phase1",
            description="Tier 1 vs Tier 1. Learn movement, targeting, basic ability usage.",
            max_iters=20,
            advance_win_rate=0.35,
            min_iters=5,
            lr=3e-5,
            entropy_coeff=0.01,
            kl_coeff=0.5,
            reward_scale=1.0,
            # Heavy outcome bonus to learn "winning is good" quickly
            win_bonus=2.0,
            loss_penalty=-1.0,
            timeout_penalty=-0.5,
            damage_reward_scale=1.0,
            engagement_bonus=0.0,
        ),
        PhaseConfig(
            name="P2_abilities",
            scenarios="dataset/scenarios/curriculum/phase2",
            description="Add Tier 2 (CC, heals, buffs). Learn ability timing.",
            max_iters=25,
            advance_win_rate=0.35,
            min_iters=5,
            lr=3e-5,
            entropy_coeff=0.01,
            kl_coeff=1.0,
            reward_scale=1.0,
            # Reward engagement to encourage ability usage
            win_bonus=1.5,
            loss_penalty=-1.0,
            timeout_penalty=-0.5,
            damage_reward_scale=1.0,
            engagement_bonus=0.005,  # small bonus per step where any damage dealt
        ),
        PhaseConfig(
            name="P3_tactics",
            scenarios="dataset/scenarios/curriculum/phase3",
            description="Cross-tier asymmetric matchups. Learn tactical adaptation.",
            max_iters=30,
            advance_win_rate=0.40,
            min_iters=10,
            lr=2e-5,  # slightly lower LR for stability
            entropy_coeff=0.01,
            kl_coeff=1.0,
            reward_scale=1.0,
            win_bonus=1.0,
            loss_penalty=-1.0,
            timeout_penalty=-0.3,
            damage_reward_scale=1.5,  # emphasize damage differential
            engagement_bonus=0.003,
        ),
        PhaseConfig(
            name="P4_diversity",
            scenarios="dataset/scenarios/curriculum/phase4",
            description="Full roster including Tier 5/6. Maximum diversity.",
            max_iters=40,
            advance_win_rate=0.45,  # higher bar for final phase
            min_iters=10,
            lr=1e-5,  # conservative LR for large scenario set
            entropy_coeff=0.01,
            kl_coeff=1.0,
            reward_scale=1.0,
            win_bonus=1.0,
            loss_penalty=-1.0,
            timeout_penalty=-0.3,
            damage_reward_scale=1.5,
            engagement_bonus=0.002,
        ),
    ]


# ---------------------------------------------------------------------------
# Reward transformation (Python-side post-processing)
# ---------------------------------------------------------------------------

def transform_rewards(
    episodes_path: str,
    output_path: str,
    cfg: PhaseConfig,
) -> dict:
    """Read episodes JSONL, apply reward shaping, write back.

    Returns stats dict.
    """
    wins = losses = timeouts = 0
    total_steps = 0

    with open(episodes_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue

            outcome = ep.get("outcome", "")
            steps = ep.get("steps", [])
            n = len(steps)

            if outcome == "Victory":
                wins += 1
            elif outcome == "Defeat":
                losses += 1
            else:
                timeouts += 1

            for i, step in enumerate(steps):
                r = step.get("step_reward", 0.0)

                # Scale damage component
                r *= cfg.damage_reward_scale

                # Engagement bonus: reward steps where damage was dealt
                if cfg.engagement_bonus > 0 and r > 0:
                    r += cfg.engagement_bonus

                # Outcome bonus on final step
                if i == n - 1:
                    if outcome == "Victory":
                        r += cfg.win_bonus
                    elif outcome == "Defeat":
                        r += cfg.loss_penalty
                    else:
                        r += cfg.timeout_penalty

                step["step_reward"] = r

            total_steps += n
            fout.write(json.dumps(ep) + "\n")

    total = wins + losses + timeouts
    return {
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "total": total,
        "win_rate": wins / max(total, 1),
        "total_steps": total_steps,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_phase(
    phase: PhaseConfig,
    checkpoint: str,
    output_dir: str,
    common_args: dict,
) -> tuple[str, float]:
    """Run a single curriculum phase. Returns (best_checkpoint_path, best_eval_win_rate)."""
    from impala_learner import (
        generate_episodes,
        start_gpu_server,
        reload_gpu_server,
        PreTensorizedData,
        flatten_trajectories,
        train_on_trajectories,
        compute_policy_values,
        export_weights,
    )
    from model import AbilityActorCriticV4
    from tokenizer import AbilityTokenizer
    from grokfast import GrokfastEMA
    import torch
    import numpy as np

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SHM_NAME = "impala_inf"

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Phase: {phase.name} — {phase.description}")
    print(f"  Scenarios: {phase.scenarios}")
    print(f"  LR: {phase.lr}, entropy: {phase.entropy_coeff}, kl: {phase.kl_coeff}")
    print(f"  Reward: win_bonus={phase.win_bonus}, engagement={phase.engagement_bonus}")
    print(f"  Advance at: {phase.advance_win_rate:.0%} eval win rate")
    print(f"{'='*70}\n")

    tok = AbilityTokenizer()
    h_dim = common_args.get("h_dim", 64)
    model = AbilityActorCriticV4(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=common_args.get("entity_encoder_layers", 4),
        external_cls_dim=common_args.get("external_cls_dim", 128),
        h_dim=h_dim,
        d_model=common_args.get("d_model", 32),
        d_ff=common_args.get("d_ff", 64),
        n_layers=common_args.get("n_layers", 4),
        n_heads=common_args.get("n_heads", 4),
    ).to(DEVICE)

    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=phase.lr, weight_decay=1e-4,
    )
    grokfast = GrokfastEMA(model, alpha=0.98, lamb=2.0)

    # Start GPU server
    gpu_proc = start_gpu_server(
        checkpoint, shm_name=SHM_NAME,
        d_model=common_args.get("d_model", 32),
        d_ff=common_args.get("d_ff", 64),
        n_layers=common_args.get("n_layers", 4),
        n_heads=common_args.get("n_heads", 4),
        entity_encoder_layers=common_args.get("entity_encoder_layers", 4),
        external_cls_dim=common_args.get("external_cls_dim", 128),
        temperature=phase.temperature,
        h_dim=h_dim,
    )
    gpu_shm = f"/dev/shm/{SHM_NAME}"

    best_win_rate = -1.0
    best_ckpt = checkpoint
    episodes_path = str(Path(output_dir) / "episodes.jsonl")
    shaped_path = str(Path(output_dir) / "episodes_shaped.jsonl")
    csv_path = str(Path(output_dir) / "training.csv")

    # CSV header
    with open(csv_path, "w") as f:
        f.write("iter,gen_time,n_episodes,win_rate,n_steps,policy_loss,value_loss,entropy,mean_reward,elapsed_s\n")

    t_start = time.time()

    try:
        for iteration in range(1, phase.max_iters + 1):
            # 1. Generate episodes
            gen_time, gen_output = generate_episodes(
                scenario_dirs=phase.scenarios,
                output_path=episodes_path,
                episodes_per_scenario=phase.episodes_per_scenario,
                threads=common_args.get("threads", 64),
                temperature=phase.temperature,
                step_interval=common_args.get("step_interval", 3),
                embedding_registry=common_args.get("embedding_registry"),
                self_play_gpu=phase.self_play_gpu,
                swap_sides=phase.swap_sides,
                enemy_weights=phase.enemy_weights,
                enemy_registry=phase.enemy_registry,
                gpu_shm=gpu_shm,
                sims_per_thread=common_args.get("sims_per_thread", 64),
            )

            # 2. Apply reward shaping
            stats = transform_rewards(episodes_path, shaped_path, phase)
            print(f"  Iter {iteration}: {stats['total']} eps, {stats['total_steps']} steps, "
                  f"win={stats['win_rate']:.1%}, gen={gen_time:.1f}s")

            # 3. Tensorize and train
            from impala_learner import extract_trajectories, load_episodes
            episodes = load_episodes(Path(shaped_path))
            trajectories = extract_trajectories(episodes)
            all_steps, step_traj_map = flatten_trajectories(trajectories)

            if all_steps:
                ptd = PreTensorizedData(
                    all_steps,
                    common_args.get("embedding_registry"),
                    common_args.get("external_cls_dim", 128),
                )

                metrics = train_on_trajectories(
                    model, optimizer, grokfast,
                    trajectories, all_steps, step_traj_map, ptd,
                    gamma=0.99,
                    step_interval=common_args.get("step_interval", 3),
                    batch_size=common_args.get("batch_size", 1024),
                    value_coeff=0.5,
                    entropy_coeff=phase.entropy_coeff,
                    clip_rho=1.0,
                    clip_c=1.0,
                    max_grad_norm=1.0,
                    reward_scale=phase.reward_scale,
                    kl_coeff=phase.kl_coeff,
                )

                elapsed = time.time() - t_start
                print(f"    Epoch 1/1 ({metrics.get('grad_steps', 0)} steps): "
                      f"pg={metrics['policy_loss']:.4f} vl={metrics['value_loss']:.4f} "
                      f"ent={metrics['entropy']:.3f} kl={metrics['kl_div']:.3f} "
                      f"rew={metrics['mean_reward']:.4f}")

                # CSV logging
                with open(csv_path, "a") as f:
                    f.write(f"{iteration},{gen_time:.1f},{stats['total']},"
                            f"{stats['win_rate']:.3f},{stats['total_steps']},"
                            f"{metrics['policy_loss']:.4f},{metrics['value_loss']:.4f},"
                            f"{metrics['entropy']:.3f},{metrics['mean_reward']:.4f},{elapsed:.0f}\n")

            # 4. Save checkpoint and reload GPU
            ckpt_path = str(Path(output_dir) / "current.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "iteration": iteration,
            }, ckpt_path)
            reload_gpu_server(gpu_shm, ckpt_path)

            # Track best
            if stats["win_rate"] > best_win_rate:
                best_win_rate = stats["win_rate"]
                best_ckpt = str(Path(output_dir) / "best.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "iteration": iteration,
                }, best_ckpt)
                print(f"    New best: {best_win_rate:.1%}")

            # 5. Eval periodically
            if iteration % phase.eval_every == 0:
                eval_path = str(Path(output_dir) / "eval_episodes.jsonl")
                eval_time, _ = generate_episodes(
                    scenario_dirs=phase.scenarios,
                    output_path=eval_path,
                    episodes_per_scenario=1,
                    threads=common_args.get("threads", 64),
                    temperature=0.0,  # greedy eval
                    step_interval=common_args.get("step_interval", 3),
                    embedding_registry=common_args.get("embedding_registry"),
                    gpu_shm=gpu_shm,
                    sims_per_thread=common_args.get("sims_per_thread", 64),
                )
                eval_stats = {"wins": 0, "losses": 0, "timeouts": 0}
                with open(eval_path) as f:
                    for line in f:
                        try:
                            ep = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        o = ep.get("outcome", "")
                        if o == "Victory": eval_stats["wins"] += 1
                        elif o == "Defeat": eval_stats["losses"] += 1
                        else: eval_stats["timeouts"] += 1
                eval_total = sum(eval_stats.values())
                eval_wr = eval_stats["wins"] / max(eval_total, 1)
                print(f"    EVAL: {eval_stats['wins']}W/{eval_stats['losses']}L/"
                      f"{eval_stats['timeouts']}T = {eval_wr:.1%}")

                # Check advancement
                if iteration >= phase.min_iters and eval_wr >= phase.advance_win_rate:
                    print(f"    ✓ Advancing! Eval {eval_wr:.1%} >= threshold {phase.advance_win_rate:.1%}")
                    break

            print(f"    Iter time: {time.time() - t_start - (elapsed if 'elapsed' in dir() else 0):.0f}s "
                  f"(total: {time.time() - t_start:.0f}s)")

    finally:
        gpu_proc.terminate()
        gpu_proc.wait()

    return best_ckpt, best_win_rate


def main():
    p = argparse.ArgumentParser(description="Curriculum training orchestrator")
    p.add_argument("--checkpoint", required=True, help="Initial checkpoint (.pt)")
    p.add_argument("--output-dir", default="generated/curriculum_gru")
    p.add_argument("--embedding-registry", default="generated/ability_embedding_registry.json")
    p.add_argument("--external-cls-dim", type=int, default=128)
    p.add_argument("--h-dim", type=int, default=64)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--sims-per-thread", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--step-interval", type=int, default=3)
    p.add_argument("--start-phase", type=int, default=1, help="Phase to start from (1-4)")
    p.add_argument("--phases", type=str, default="1,2,3,4",
                   help="Comma-separated phase numbers to run")
    args = p.parse_args()

    common_args = {k: v for k, v in vars(args).items()}
    curriculum = build_curriculum()

    phases_to_run = [int(x) for x in args.phases.split(",")]
    checkpoint = args.checkpoint
    log_path = Path(args.output_dir) / "curriculum.log"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Curriculum training: {len(phases_to_run)} phases")
    print(f"Starting checkpoint: {checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"GRU h_dim: {args.h_dim}")

    results = []
    for phase_num in phases_to_run:
        phase = curriculum[phase_num - 1]
        phase_dir = str(Path(args.output_dir) / phase.name)

        best_ckpt, best_wr = run_phase(
            phase, checkpoint, phase_dir, common_args,
        )

        results.append({"phase": phase.name, "best_wr": best_wr, "checkpoint": best_ckpt})
        print(f"\n{'='*70}")
        print(f"Phase {phase.name} complete: best eval = {best_wr:.1%}")
        print(f"Best checkpoint: {best_ckpt}")
        print(f"{'='*70}\n")

        # Use best checkpoint as starting point for next phase
        checkpoint = best_ckpt

    # Summary
    print("\n" + "="*70)
    print("CURRICULUM COMPLETE")
    print("="*70)
    for r in results:
        print(f"  {r['phase']}: {r['best_wr']:.1%} — {r['checkpoint']}")


if __name__ == "__main__":
    # Add training dir to path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
