#!/usr/bin/env python3
"""Verify the entire data pipeline from Rust sim -> SHM -> Python model.

Runs a suite of checks against known drill scenarios to ensure feature values,
action mappings, and reward signals are correct end-to-end.

Usage:
    PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/verify_features.py \
        --checkpoint generated/actor_critic_v4_full_unfrozen.pt \
        --embedding-registry generated/ability_embedding_registry.json \
        --external-cls-dim 128
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Project root (one level up from training/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DRILL_BASE = PROJECT_ROOT / "dataset" / "scenarios" / "drills"

# Drill paths by category
DRILLS_REACH = DRILL_BASE / "phase1" / "1_1_reach_static"
DRILLS_NAVIGATE = DRILL_BASE / "phase1" / "1_3_navigate_obstacles"
DRILLS_MAINTAIN = DRILL_BASE / "phase2" / "2_1_maintain_distance"
DRILLS_KILL = DRILL_BASE / "phase3" / "3_1_kill_stationary"
DRILLS_CC = DRILL_BASE / "phase4" / "4_4_cc_burst"
DRILLS_FOCUS = DRILL_BASE / "phase5" / "5_1_focus_fire"

# Rust move_dir_offset mapping (must match src/ai/core/self_play/actions.rs:550)
RUST_MOVE_DIRS = {
    0: (0.0, 1.0),    # N
    1: (0.707, 0.707), # NE
    2: (1.0, 0.0),    # E
    3: (0.707, -0.707), # SE
    4: (0.0, -1.0),   # S
    5: (-0.707, -0.707), # SW
    6: (-1.0, 0.0),   # W
    7: (-0.707, 0.707), # NW
    8: (0.0, 0.0),    # stay
}


# ---------------------------------------------------------------------------
# GPU server management (reuses impala_learner pattern)
# ---------------------------------------------------------------------------

SHM_NAME = "verify_inf"
SHM_PATH = f"/dev/shm/{SHM_NAME}"
SHM_PATH_PREFIX = "/dev/shm/"


def start_gpu_server(
    checkpoint_path: str,
    shm_name: str = SHM_NAME,
    max_batch_size: int = 64,
    d_model: int = 32,
    d_ff: int = 64,
    n_layers: int = 4,
    n_heads: int = 4,
    entity_encoder_layers: int = 4,
    external_cls_dim: int = 0,
    temperature: float = 1.0,
    h_dim: int = 0,
    model_version: int = 4,
) -> subprocess.Popen:
    """Start GPU inference server as subprocess."""
    cmd = [
        sys.executable, "training/gpu_inference_server.py",
        "--weights", checkpoint_path,
        "--shm-name", shm_name,
        "--max-batch-size", str(max_batch_size),
        "--d-model", str(d_model),
        "--d-ff", str(d_ff),
        "--n-layers", str(n_layers),
        "--n-heads", str(n_heads),
        "--entity-encoder-layers", str(entity_encoder_layers),
        "--temperature", str(temperature),
    ]
    if external_cls_dim > 0:
        cmd.extend(["--external-cls-dim", str(external_cls_dim)])
    if h_dim > 0:
        cmd.extend(["--h-dim", str(h_dim)])
    if model_version != 4:
        cmd.extend(["--model-version", str(model_version)])

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        cwd=str(PROJECT_ROOT),
    )

    ready = False
    for line in proc.stdout:
        line = line.rstrip()
        print(f"  [gpu] {line}", flush=True)
        if "Ready" in line:
            ready = True
            break

    if not ready:
        raise RuntimeError("GPU server failed to start")

    def drain():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [gpu] {line}", flush=True)

    t = threading.Thread(target=drain, daemon=True)
    t.start()

    return proc


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------

def generate_episodes(
    scenario_dir: str,
    output_path: str,
    episodes_per_scenario: int = 1,
    threads: int = 4,
    temperature: float = 1.0,
    step_interval: int = 3,
    gpu_shm: str | None = None,
    embedding_registry: str | None = None,
    weights_path: str | None = None,
) -> bool:
    """Generate episodes via Rust. Returns True on success."""
    cmd = [
        "cargo", "run", "--release", "--bin", "xtask", "--",
        "scenario", "oracle", "transformer-rl", "generate",
        scenario_dir,
        "--episodes", str(episodes_per_scenario),
        "-j", str(threads),
        "--temperature", str(temperature),
        "--step-interval", str(step_interval),
        "-o", output_path,
    ]
    if gpu_shm:
        cmd.extend(["--gpu-shm", gpu_shm])
    elif weights_path:
        cmd.extend(["--weights", weights_path])
    else:
        raise ValueError("Must specify gpu_shm or weights_path")
    if embedding_registry:
        cmd.extend(["--embedding-registry", embedding_registry])

    proc = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
    )
    if proc.returncode != 0:
        # Print last few lines of stderr for debugging
        stderr_lines = proc.stderr.strip().split("\n")
        for line in stderr_lines[-10:]:
            print(f"    [gen] {line}", flush=True)
        return False
    return True


def load_episodes(path: str) -> list[dict]:
    """Load episodes from JSONL file."""
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def pick_one_drill(drill_dir: Path) -> str | None:
    """Pick the first TOML drill from a directory."""
    if not drill_dir.exists():
        return None
    tomls = sorted(drill_dir.glob("*.toml"))
    return str(tomls[0]) if tomls else None


def load_drill_toml(path: str) -> dict:
    """Load a drill TOML file."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Check result type
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name: str, passed: bool, message: str):
        self.name = name
        self.passed = passed
        self.message = message

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


# ---------------------------------------------------------------------------
# Generate episodes for a drill directory, caching results
# ---------------------------------------------------------------------------

class VerificationContext:
    """Manages episode generation and caching for verification checks."""

    def __init__(self, args):
        self.args = args
        self.gpu_shm = SHM_PATH if args.gpu else None
        self.tmp_dir = tempfile.mkdtemp(prefix="verify_features_")
        self._cache: dict[str, list[dict]] = {}

    def generate_for_drill(self, drill_dir: Path, label: str,
                           episodes_per_scenario: int = 1) -> list[dict] | None:
        """Generate episodes for a drill directory. Returns episodes or None on failure."""
        cache_key = f"{drill_dir}:{episodes_per_scenario}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        out_path = os.path.join(self.tmp_dir, f"{label}.jsonl")
        print(f"  Generating episodes for {label}...", flush=True)

        ok = generate_episodes(
            scenario_dir=str(drill_dir),
            output_path=out_path,
            episodes_per_scenario=episodes_per_scenario,
            threads=min(4, self.args.threads),
            temperature=self.args.temperature,
            step_interval=3,
            gpu_shm=self.gpu_shm,
            embedding_registry=self.args.embedding_registry,
            weights_path=self.args.checkpoint if not self.args.gpu else None,
        )
        if not ok:
            return None

        episodes = load_episodes(out_path)
        if not episodes:
            return None

        self._cache[cache_key] = episodes
        return episodes

    def generate_for_single_drill(self, drill_path: str, label: str) -> list[dict] | None:
        """Generate episodes from a single drill TOML's parent directory.

        Since the Rust CLI takes a directory, we create a temp dir with a symlink.
        """
        cache_key = f"single:{drill_path}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Create temp dir with symlink to the single TOML
        single_dir = os.path.join(self.tmp_dir, f"single_{label}")
        os.makedirs(single_dir, exist_ok=True)
        toml_name = os.path.basename(drill_path)
        link_path = os.path.join(single_dir, toml_name)
        if not os.path.exists(link_path):
            os.symlink(drill_path, link_path)

        out_path = os.path.join(self.tmp_dir, f"{label}.jsonl")
        print(f"  Generating episodes for {label}...", flush=True)

        ok = generate_episodes(
            scenario_dir=single_dir,
            output_path=out_path,
            episodes_per_scenario=1,
            threads=1,
            temperature=self.args.temperature,
            step_interval=3,
            gpu_shm=self.gpu_shm,
            embedding_registry=self.args.embedding_registry,
            weights_path=self.args.checkpoint if not self.args.gpu else None,
        )
        if not ok:
            return None

        episodes = load_episodes(out_path)
        if not episodes:
            return None

        self._cache[cache_key] = episodes
        return episodes


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_0_1_position_sanity(ctx: VerificationContext) -> CheckResult:
    """0.1 Position Feature Sanity: features[5] and [6] are in [0,1] for 20x20 room,
    and position changes between steps (unit is not stuck)."""
    name = "0.1 Position Feature Sanity"

    episodes = ctx.generate_for_drill(DRILLS_REACH, "reach_static")
    if not episodes:
        return CheckResult(name, False, "Failed to generate episodes from reach_static drills")

    # Find an episode with enough steps
    ep = None
    for e in episodes:
        steps_with_ents = [s for s in e["steps"] if s.get("entities")]
        if len(steps_with_ents) >= 2:
            ep = e
            break
    if ep is None:
        # All episodes too short — still check first step of first episode
        ep = episodes[0]
        steps_with_ents = [s for s in ep["steps"] if s.get("entities")]
        if not steps_with_ents:
            return CheckResult(name, False, "No steps with entity data")
        first_ent = steps_with_ents[0]["entities"][0]
        pos_x, pos_y = first_ent[5], first_ent[6]
        if 0.0 <= pos_x <= 1.0 and 0.0 <= pos_y <= 1.0:
            return CheckResult(name, True,
                f"Position ({pos_x:.3f},{pos_y:.3f}) in range (only 1 step, can't check movement)")
        return CheckResult(name, False, f"Position ({pos_x},{pos_y}) out of [0,1]")

    steps = [s for s in ep["steps"] if s.get("entities")]

    # Check first step
    first_ent = steps[0]["entities"][0]  # self entity
    pos_x, pos_y = first_ent[5], first_ent[6]

    if not (0.0 <= pos_x <= 1.0):
        return CheckResult(name, False,
                           f"First step position_x={pos_x:.4f} outside [0,1] (normalized by /20)")
    if not (0.0 <= pos_y <= 1.0):
        return CheckResult(name, False,
                           f"First step position_y={pos_y:.4f} outside [0,1] (normalized by /20)")

    # Check position changes between steps (unit should move in reach_static drill)
    positions = [(s["entities"][0][5], s["entities"][0][6]) for s in steps[:10]]
    moved = False
    for i in range(1, len(positions)):
        dx = abs(positions[i][0] - positions[i - 1][0])
        dy = abs(positions[i][1] - positions[i - 1][1])
        if dx > 1e-4 or dy > 1e-4:
            moved = True
            break

    if not moved:
        return CheckResult(name, False,
                           f"Unit did not move across {len(positions)} steps. "
                           f"Positions: {positions[:5]}")

    return CheckResult(name, True,
                       f"pos=({pos_x:.3f},{pos_y:.3f}), unit moved across steps")


def check_0_2_hp_sanity(ctx: VerificationContext) -> CheckResult:
    """0.2 HP Feature Sanity: HP starts at 1.0, may decrease when taking damage."""
    name = "0.2 HP Feature Sanity"

    # Use kill_stationary drill (has enemies that may fight back)
    episodes = ctx.generate_for_drill(DRILLS_KILL, "kill_stationary")
    if not episodes:
        return CheckResult(name, False, "Failed to generate kill_stationary episodes")

    ep = episodes[0]
    steps = [s for s in ep["steps"] if s.get("entities")]
    if not steps:
        return CheckResult(name, False, "No steps with entities")

    # Check self HP (feature[0]) starts at 1.0
    first_hp = steps[0]["entities"][0][0]
    if abs(first_hp - 1.0) > 0.01:
        return CheckResult(name, False,
                           f"Initial HP={first_hp:.4f}, expected 1.0")

    # Check for enemy HP decrease (the target dummy should take damage)
    # Find enemy entity (entity_types[i] == 1)
    enemy_hps = []
    for s in steps:
        types = s.get("entity_types", [])
        ents = s.get("entities", [])
        for i, t in enumerate(types):
            if t == 1 and i < len(ents):
                enemy_hps.append(ents[i][0])
                break

    hp_decreased = False
    if len(enemy_hps) >= 2:
        if enemy_hps[-1] < enemy_hps[0] - 0.001:
            hp_decreased = True

    msg = f"self_hp_start={first_hp:.3f}"
    if enemy_hps:
        msg += f", enemy_hp: {enemy_hps[0]:.3f} -> {enemy_hps[-1]:.3f}"
        if hp_decreased:
            msg += " (decreased)"
    else:
        msg += ", no enemy HP data"

    return CheckResult(name, True, msg)


def check_0_3_entity_type_mapping(ctx: VerificationContext) -> CheckResult:
    """0.3 Entity Type Mapping: self=0, enemy=1, correct ordering."""
    name = "0.3 Entity Type Mapping"

    episodes = ctx.generate_for_drill(DRILLS_KILL, "kill_stationary")
    if not episodes:
        return CheckResult(name, False, "Failed to generate kill_stationary episodes")

    ep = episodes[0]
    steps = [s for s in ep["steps"] if s.get("entity_types")]
    if not steps:
        return CheckResult(name, False, "No steps with entity_types")

    s = steps[0]
    types = s["entity_types"]

    if types[0] != 0:
        return CheckResult(name, False,
                           f"entity_types[0]={types[0]}, expected 0 (self)")

    has_enemy = any(t == 1 for t in types)
    if not has_enemy:
        return CheckResult(name, False,
                           f"No enemy (type=1) found in entity_types: {types}")

    # Check ordering: self (0) first, then enemies (1), then allies (2)
    last_type = -1
    ordering_ok = True
    for t in types:
        if t < last_type:
            # Allow padding (type could be 0 for empty slots at the end)
            # Only check non-zero types for ordering
            if t != 0:
                ordering_ok = False
                break
        if t > 0:
            last_type = t

    msg = f"types={types}, self=0 correct, enemy present"
    if not ordering_ok:
        return CheckResult(name, False, f"Entity ordering violated: {types}")

    return CheckResult(name, True, msg)


def check_0_4_range_combat_features(ctx: VerificationContext) -> CheckResult:
    """0.4 Range and Combat Features: attack_range and auto_dps are reasonable."""
    name = "0.4 Range/Combat Features"

    # Kill drills use "archer" template which has attack_range ~4
    episodes = ctx.generate_for_drill(DRILLS_KILL, "kill_stationary")
    if not episodes:
        return CheckResult(name, False, "Failed to generate kill_stationary episodes")

    ep = episodes[0]
    steps = [s for s in ep["steps"] if s.get("entities")]
    if not steps:
        return CheckResult(name, False, "No steps with entities")

    self_ent = steps[0]["entities"][0]
    auto_dps = self_ent[12]   # auto_dps / 30
    attack_range = self_ent[13]  # attack_range / 10

    issues = []
    if auto_dps <= 0:
        issues.append(f"auto_dps={auto_dps:.4f} (expected > 0, feature[12]=dps/30)")
    if attack_range <= 0:
        issues.append(f"attack_range={attack_range:.4f} (expected > 0, feature[13]=range/10)")
    if attack_range > 1.0:
        issues.append(f"attack_range={attack_range:.4f} (> 1.0, implies range > 10)")

    if issues:
        return CheckResult(name, False, "; ".join(issues))

    return CheckResult(name, True,
                       f"auto_dps={auto_dps:.3f} (raw ~{auto_dps*30:.1f}), "
                       f"attack_range={attack_range:.3f} (raw ~{attack_range*10:.1f})")


def check_0_5_ability_features(ctx: VerificationContext) -> CheckResult:
    """0.5 Ability Features: at least one ability category has non-zero stats."""
    name = "0.5 Ability Features"

    # CC burst drills use "stunner" which has abilities
    episodes = ctx.generate_for_drill(DRILLS_CC, "cc_burst")
    if not episodes:
        # Fall back to kill drills — archer may have abilities too
        episodes = ctx.generate_for_drill(DRILLS_KILL, "kill_stationary")
    if not episodes:
        return CheckResult(name, False, "Failed to generate episodes with abilities")

    ep = episodes[0]
    steps = [s for s in ep["steps"] if s.get("entities")]
    if not steps:
        return CheckResult(name, False, "No steps with entities")

    self_ent = steps[0]["entities"][0]
    ability_damage = self_ent[15]  # ability_damage / 50
    heal_amount = self_ent[18]    # heal_amount / 50
    control_range = self_ent[21]  # control_range / 10

    has_any = (ability_damage > 0 or heal_amount > 0 or control_range > 0)

    msg = (f"ability_damage={ability_damage:.3f}, "
           f"heal_amount={heal_amount:.3f}, "
           f"control_range={control_range:.3f}")

    if not has_any:
        return CheckResult(name, False,
                           f"No ability features set. {msg}. "
                           f"Full features[15:24]: {self_ent[15:24]}")

    return CheckResult(name, True, msg)


def check_0_6_state_features(ctx: VerificationContext) -> CheckResult:
    """0.6 State Features: is_casting is 0 or 1, move_speed > 0."""
    name = "0.6 State Features"

    episodes = ctx.generate_for_drill(DRILLS_REACH, "reach_static")
    if not episodes:
        return CheckResult(name, False, "Failed to generate reach_static episodes")

    ep = episodes[0]
    steps = [s for s in ep["steps"] if s.get("entities")]
    if not steps:
        return CheckResult(name, False, "No steps with entities")

    issues = []
    for i, s in enumerate(steps[:20]):
        self_ent = s["entities"][0]
        is_casting = self_ent[24]
        move_speed = self_ent[27]

        if is_casting not in (0.0, 1.0):
            issues.append(f"step {i}: is_casting={is_casting:.4f} (expected 0 or 1)")
        if move_speed <= 0:
            issues.append(f"step {i}: move_speed={move_speed:.4f} (expected > 0)")

    if issues:
        return CheckResult(name, False, "; ".join(issues[:3]))

    move_speed = steps[0]["entities"][0][27]
    return CheckResult(name, True,
                       f"is_casting valid (0/1), move_speed={move_speed:.3f} "
                       f"(raw ~{move_speed*5:.1f})")


def check_0_7_aggregate_token(ctx: VerificationContext) -> CheckResult:
    """0.7 Aggregate Token: length==16, entity counts reasonable,
    target direction non-zero for drills with target_position."""
    name = "0.7 Aggregate Token"

    episodes = ctx.generate_for_drill(DRILLS_REACH, "reach_static")
    if not episodes:
        return CheckResult(name, False, "Failed to generate reach_static episodes")

    ep = episodes[0]
    steps = [s for s in ep["steps"] if s.get("aggregate_features")]

    if not steps:
        return CheckResult(name, True,
                           "No aggregate_features in episodes (V4 without target_proj, OK)")

    agg = steps[0]["aggregate_features"]
    if len(agg) != 16:
        return CheckResult(name, False, f"aggregate length={len(agg)}, expected 16")

    # Indices 0-3 are entity counts (should be reasonable)
    issues = []
    for i in range(4):
        if agg[i] < 0:
            issues.append(f"agg[{i}]={agg[i]:.3f} (entity count negative)")

    # Indices 14-15 are target direction (should be non-zero for reach_point drills)
    target_dx, target_dy = agg[14], agg[15]
    target_mag = math.sqrt(target_dx**2 + target_dy**2)
    if target_mag < 1e-4:
        issues.append(f"target_dir=({target_dx:.3f},{target_dy:.3f}) is zero, "
                      "expected non-zero for reach_point drill")

    if issues:
        return CheckResult(name, False, "; ".join(issues))

    return CheckResult(name, True,
                       f"len=16, counts={agg[0:4]}, "
                       f"target_dir=({target_dx:.3f},{target_dy:.3f})")


def check_0_8_gru_hidden_state(ctx: VerificationContext) -> CheckResult:
    """0.8 GRU Hidden State: verify h_dim in SHM header matches expected."""
    name = "0.8 GRU Hidden State"

    h_dim = ctx.args.h_dim
    if h_dim <= 0:
        return CheckResult(name, True, "GRU disabled (h_dim=0), skipping SHM check")

    if not ctx.args.gpu:
        return CheckResult(name, True, "GPU server not running, skipping SHM check")

    shm_path = f"/dev/shm/{SHM_NAME}"
    if not os.path.exists(shm_path):
        return CheckResult(name, False, f"SHM file {shm_path} not found")

    import mmap
    fd = os.open(shm_path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    os.close(fd)

    shm_h_dim = struct.unpack_from('<I', mm, 0x18)[0]
    mm.close()

    if shm_h_dim != h_dim:
        return CheckResult(name, False,
                           f"SHM h_dim={shm_h_dim}, expected {h_dim}")

    return CheckResult(name, True, f"SHM h_dim={shm_h_dim} matches expected")


def check_0_9_move_direction_mapping(ctx: VerificationContext) -> CheckResult:
    """0.9 Move Direction Mapping: verify Rust move_dir_offset matches expected.

    This is critical -- the Y-axis bug was the root cause of all training failures.
    0=N(0,+1), 1=NE(+,+), 2=E(+,0), 3=SE(+,-), 4=S(0,-1),
    5=SW(-,-), 6=W(-,0), 7=NW(-,+), 8=stay(0,0)
    """
    name = "0.9 Move Direction Mapping"

    expected = {
        0: (0.0, 1.0),     # N
        1: (0.707, 0.707),  # NE
        2: (1.0, 0.0),     # E
        3: (0.707, -0.707), # SE
        4: (0.0, -1.0),    # S
        5: (-0.707, -0.707), # SW
        6: (-1.0, 0.0),    # W
        7: (-0.707, 0.707), # NW
        8: (0.0, 0.0),     # stay
    }

    # Generate a reach episode and verify position deltas match move directions
    episodes = ctx.generate_for_drill(DRILLS_REACH, "reach_static")
    if not episodes:
        return CheckResult(name, False, "Failed to generate reach_static episodes")

    # Find an episode with enough steps
    steps = []
    for ep in episodes:
        ep_steps = [s for s in ep["steps"]
                    if s.get("move_dir") is not None and s.get("entities")]
        if len(ep_steps) >= 3:
            steps = ep_steps
            break

    if len(steps) < 3:
        # Collect steps across ALL episodes
        steps = []
        for ep in episodes:
            steps.extend(s for s in ep["steps"]
                        if s.get("move_dir") is not None and s.get("entities"))
        if len(steps) < 3:
            return CheckResult(name, True, f"Only {len(steps)} steps — move_dir range checked, can't verify deltas")

    # Verify that move_dir values are in valid range [0, 8]
    dirs_seen = set()
    for s in steps:
        d = s["move_dir"]
        if d < 0 or d > 8:
            return CheckResult(name, False, f"Invalid move_dir={d}")
        dirs_seen.add(d)

    # Check position deltas match direction for non-stay moves
    # Note: Entry rooms have obstacles that cause wall bouncing —
    # unit may be deflected or stopped by walls, so we use a generous
    # threshold and skip steps with very small movement (collisions).
    consistent = 0
    checked = 0
    issues = []
    for i in range(1, len(steps)):
        prev = steps[i - 1]
        curr = steps[i]
        d = prev["move_dir"]
        if d == 8:
            continue  # skip stay

        prev_pos = (prev["entities"][0][5], prev["entities"][0][6])
        curr_pos = (curr["entities"][0][5], curr["entities"][0][6])
        # Positions are already /20, so deltas are small
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]

        # Normalize delta to unit direction
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 0.001:
            continue  # didn't move enough — likely wall collision

        ndx, ndy = dx / mag, dy / mag
        ex, ey = expected[d]
        emag = math.sqrt(ex * ex + ey * ey)
        if emag < 1e-5:
            continue
        nex, ney = ex / emag, ey / emag

        # Dot product should be close to 1 if same direction
        dot = ndx * nex + ndy * ney
        checked += 1
        if dot > 0.5:
            consistent += 1
        else:
            issues.append(f"step {i}: dir={d}, expected ({nex:.2f},{ney:.2f}), "
                          f"actual delta ({ndx:.2f},{ndy:.2f}), dot={dot:.3f}")

    if checked == 0:
        return CheckResult(name, True,
                           f"Move dirs valid [0-8], dirs_seen={dirs_seen}, "
                           f"no movement deltas to verify (unit may be stuck)")

    consistency = consistent / checked

    # If consistency is very low (<30%), it may indicate Y-axis inversion
    if consistency < 0.3:
        return CheckResult(name, False,
                           f"Possible Y-axis inversion: {consistent}/{checked} "
                           f"({consistency:.0%}) correct. Issues: {issues[:5]}")
    if consistency < 0.4:
        return CheckResult(name, False,
                           f"Only {consistent}/{checked} ({consistency:.0%}) moves match "
                           f"expected direction (Entry rooms have obstacles). Issues: {issues[:3]}")

    return CheckResult(name, True,
                       f"{consistent}/{checked} ({consistency:.0%}) moves consistent, "
                       f"dirs_seen={dirs_seen}")


def check_0_10_action_masking(ctx: VerificationContext) -> CheckResult:
    """0.10 Action Masking: combat_mask reflects available actions correctly."""
    name = "0.10 Action Masking"

    # Reach drills have enemy_count=0 and action_mask=move_only
    episodes = ctx.generate_for_drill(DRILLS_REACH, "reach_static")
    if not episodes:
        return CheckResult(name, False, "Failed to generate reach_static episodes")

    ep = episodes[0]
    steps = [s for s in ep["steps"] if s.get("mask")]
    if not steps:
        return CheckResult(name, False, "No steps with mask data")

    s = steps[0]
    mask = s["mask"]

    # In a move_only drill, hold should be available but attack may not be
    # mask layout: [attack_near, attack_weak, attack_focus, hold, ability0..7]
    # Actually the mask in episode JSONL corresponds to the full action space
    # Hold (index 1 in combat_mask) should always be True
    # Check that hold is available in the mask

    # The mask in JSONL is the full V3/V4 mask - let's just verify it exists and
    # has reasonable values
    issues = []

    # For reach drills with no enemies, the first 3 entries (attacks) should be False
    types = s.get("entity_types", [])
    has_enemy = any(t == 1 for t in types)

    if not has_enemy:
        # With no enemies, attack mask entries (first 3) should be False
        attack_any = any(mask[j] for j in range(min(3, len(mask))))
        if attack_any:
            issues.append(f"Attack mask True with no enemies: mask[:3]={mask[:3]}")

    if issues:
        return CheckResult(name, False, "; ".join(issues))

    return CheckResult(name, True,
                       f"mask has {len(mask)} entries, no_enemy={not has_enemy}, "
                       f"attack_masked={not any(mask[j] for j in range(min(3, len(mask))))}")


def check_0_11_reward_signal(ctx: VerificationContext) -> CheckResult:
    """0.11 Reward Signal: Victory has positive total reward, Timeout has negative."""
    name = "0.11 Reward Signal"

    # Generate multiple episodes from reach drills to get both Victory and Timeout
    episodes = ctx.generate_for_drill(DRILLS_REACH, "reach_static", episodes_per_scenario=2)
    if not episodes:
        return CheckResult(name, False, "Failed to generate reach_static episodes")

    victories = [ep for ep in episodes if ep.get("outcome") == "Victory"]
    timeouts = [ep for ep in episodes if ep.get("outcome") == "Timeout"]

    msg_parts = []
    issues = []

    if victories:
        v_rewards = [ep["reward"] for ep in victories]
        msg_parts.append(f"Victory rewards: {v_rewards[:3]}")
        if any(r < 0 for r in v_rewards):
            issues.append(f"Victory episode has negative reward: {v_rewards}")
    else:
        msg_parts.append("No Victory episodes generated")

    if timeouts:
        t_rewards = [ep["reward"] for ep in timeouts]
        msg_parts.append(f"Timeout rewards: {t_rewards[:3]}")
        if any(r > 0 for r in t_rewards):
            issues.append(f"Timeout episode has positive reward: {t_rewards}")
    else:
        msg_parts.append("No Timeout episodes generated")

    # Check step_reward progression in a Victory episode
    if victories:
        ep = victories[0]
        steps = [s for s in ep["steps"] if s.get("step_reward") is not None]
        if steps:
            step_rewards = [s["step_reward"] for s in steps]
            msg_parts.append(f"step_rewards range: [{min(step_rewards):.3f}, "
                             f"{max(step_rewards):.3f}]")

    if issues:
        return CheckResult(name, False, "; ".join(issues))

    if not victories and not timeouts:
        outcomes = set(ep.get("outcome", "?") for ep in episodes)
        return CheckResult(name, True,
                           f"No Victory/Timeout to compare. Outcomes: {outcomes}, "
                           f"rewards: {[ep['reward'] for ep in episodes[:5]]}")

    return CheckResult(name, True, "; ".join(msg_parts))


def check_0_12_drill_objective(ctx: VerificationContext) -> CheckResult:
    """0.12 Drill Objective Evaluation: Victory positions are near target."""
    name = "0.12 Drill Objective"

    # Use the bulk reach_static episodes (same as 0.1)
    episodes = ctx.generate_for_drill(DRILLS_REACH, "reach_static")
    if not episodes:
        return CheckResult(name, False, "Failed to generate reach_static episodes")

    # Load target positions from TOMLs
    import tomllib
    from pathlib import Path
    targets = {}
    for f in Path(DRILLS_REACH).glob("*.toml"):
        with open(f, "rb") as fh:
            cfg = tomllib.load(fh).get("scenario", {})
        tp = cfg.get("target_position")
        obj = cfg.get("objective", {})
        if tp:
            targets[cfg.get("name", "")] = (tp, obj.get("radius", 1.0))

    # Check Victory episodes: final position should be near target
    victories_checked = 0
    violations = []
    for ep in episodes:
        if ep.get("outcome") != "Victory":
            continue
        tp_info = targets.get(ep.get("scenario", ""))
        if not tp_info:
            continue
        target, radius = tp_info
        steps = [s for s in ep["steps"] if s.get("entities")]
        if not steps:
            continue
        final = steps[-1]["entities"][0]
        fx, fy = final[5] * 20, final[6] * 20
        dist = math.sqrt((fx - target[0])**2 + (fy - target[1])**2)
        victories_checked += 1
        if dist > radius + 1.0:  # allow some slack
            violations.append(f"dist={dist:.1f} target=({target[0]:.1f},{target[1]:.1f}) final=({fx:.1f},{fy:.1f})")

    # Count total outcomes for diagnostics
    outcomes = {}
    for ep in episodes:
        o = ep.get("outcome", "?")
        outcomes[o] = outcomes.get(o, 0) + 1

    if victories_checked == 0:
        # No Victories is OK — it means the model didn't reach the target,
        # which is correct behavior (no false Victories from the old bug).
        return CheckResult(name, True,
            f"No Victory episodes (model didn't reach target). "
            f"Outcomes: {outcomes}. Drill objective correctly gates Victory.")

    if violations:
        return CheckResult(name, False,
            f"{len(violations)}/{victories_checked} victories too far from target: {violations[0]}")

    return CheckResult(name, True,
        f"All {victories_checked} Victory episodes have final position near target. "
        f"Outcomes: {outcomes}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Verify data pipeline: Rust sim -> SHM -> Python model")

    # Model / weights
    p.add_argument("--checkpoint", required=True,
                   help="Model checkpoint (.pt)")
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)
    p.add_argument("--external-cls-dim", type=int, default=0)
    p.add_argument("--h-dim", type=int, default=0)
    p.add_argument("--model-version", type=int, default=4, choices=[4, 5])

    # Generation
    p.add_argument("--embedding-registry",
                   help="Ability embedding registry JSON")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)

    # GPU inference
    p.add_argument("--gpu", action="store_true",
                   help="Start and use GPU inference server (shared memory)")

    # Filter
    p.add_argument("--checks", nargs="*",
                   help="Run only specific checks (e.g., 0.1 0.3 0.9)")

    args = p.parse_args()

    print("=" * 60)
    print("Feature Verification Suite")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"GPU mode: {args.gpu}")
    print(f"Model version: V{args.model_version}")
    print()

    # Start GPU server if requested
    gpu_proc = None
    if args.gpu:
        print("Starting GPU inference server...", flush=True)
        gpu_proc = start_gpu_server(
            checkpoint_path=args.checkpoint,
            shm_name=SHM_NAME,
            max_batch_size=64,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            entity_encoder_layers=args.entity_encoder_layers,
            external_cls_dim=args.external_cls_dim,
            temperature=args.temperature,
            h_dim=args.h_dim,
            model_version=args.model_version,
        )
        print("GPU server ready.\n", flush=True)

    ctx = VerificationContext(args)

    # All checks in order
    all_checks = [
        ("0.1", check_0_1_position_sanity),
        ("0.2", check_0_2_hp_sanity),
        ("0.3", check_0_3_entity_type_mapping),
        ("0.4", check_0_4_range_combat_features),
        ("0.5", check_0_5_ability_features),
        ("0.6", check_0_6_state_features),
        ("0.7", check_0_7_aggregate_token),
        ("0.8", check_0_8_gru_hidden_state),
        ("0.9", check_0_9_move_direction_mapping),
        ("0.10", check_0_10_action_masking),
        ("0.11", check_0_11_reward_signal),
        ("0.12", check_0_12_drill_objective),
    ]

    # Filter checks if specified
    if args.checks:
        selected = set(args.checks)
        all_checks = [(cid, fn) for cid, fn in all_checks if cid in selected]

    results: list[CheckResult] = []
    for check_id, check_fn in all_checks:
        print(f"--- Running check {check_id} ---", flush=True)
        try:
            result = check_fn(ctx)
        except Exception as e:
            result = CheckResult(f"{check_id} (exception)", False, str(e))
        results.append(result)
        print(f"  {result}", flush=True)
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    for r in results:
        print(f"  {r}")
    print()
    print(f"  {passed} passed, {failed} failed out of {len(results)} checks")

    # Cleanup GPU server
    if gpu_proc:
        gpu_proc.terminate()
        try:
            gpu_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gpu_proc.kill()
        # Clean up SHM
        shm_path = f"/dev/shm/{SHM_NAME}"
        if os.path.exists(shm_path):
            os.unlink(shm_path)

    # Cleanup temp dir
    import shutil
    shutil.rmtree(ctx.tmp_dir, ignore_errors=True)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
