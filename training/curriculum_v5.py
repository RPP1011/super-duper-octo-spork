#!/usr/bin/env python3
"""V5 Skill Curriculum Orchestrator.

Drill-based training with:
- Per-drill reward shaping (see docs/plans/v5_migration/reward_functions.md)
- 100/100 pass gate before advancing
- Action head gating (move_only → move_attack → all)
- Regression testing on previously passed drills
- Phase 0 feature verification
- Action flicker penalty (Phase 3+)
- Phase transition warmup (halved time penalty for first 5 iters of new action head)

Usage:
    uv run --with numpy --with torch --with tomli_w training/curriculum_v5.py \
        --checkpoint generated/random_init_v5.pt \
        --output-dir generated/curriculum_v5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Drill definitions
# ---------------------------------------------------------------------------

@dataclass
class DrillConfig:
    """Configuration for a single drill type."""
    name: str
    phase: int
    scenario_dir: str           # path to drill TOML directory
    action_mask: str            # "move_only", "move_attack", "all"
    pass_threshold: int = 100   # pass N out of 100 trials
    max_iters: int = 30
    min_iters: int = 3

    # Training hyperparams
    lr: float = 5e-4
    temperature: float = 1.0
    episodes_per_scenario: int = 1  # 1 per drill variant

    # Reward shaping params (from reward_functions.md v3.1)
    time_penalty: float = -0.002
    win_bonus: float = 0.3
    reward_scale: float = 1.0

    # Prerequisites (drill names that must be passed first)
    prerequisites: list[str] = field(default_factory=list)

    # Which modules to train (None = all). List of param name prefixes.
    # E.g., ["move_head", "temporal_gru"] freezes everything else.
    trainable_modules: list[str] | None = None
    train_epochs: int = 1  # epochs per iteration
    # trainer field removed — all drills use SAC

    description: str = ""


def build_curriculum() -> list[DrillConfig]:
    """Define all drills in curriculum order."""
    drills_root = "dataset/scenarios/drills"
    return [
        # ── Phase 1: Movement (move_only) ──
        DrillConfig(
            name="1.1_reach_static",
            phase=1,
            scenario_dir=f"{drills_root}/phase1/1_1_reach_static",
            action_mask="move_only",
            max_iters=50,
            lr=1e-4,
            episodes_per_scenario=10,
            time_penalty=-0.002,
            win_bonus=0.3,
            reward_scale=1.0,
            train_epochs=1,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            description="Reach a static target point",
        ),
        DrillConfig(
            name="1.2_reach_moving",
            phase=1,
            scenario_dir=f"{drills_root}/phase1/1_2_reach_moving",
            action_mask="move_only",
            max_iters=50,
            lr=1e-4,
            episodes_per_scenario=5,
            time_penalty=-0.002,
            win_bonus=0.4,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            prerequisites=["1.1_reach_static"],
            description="Reach a moving target point",
        ),
        DrillConfig(
            name="1.3_navigate_obstacles",
            phase=1,
            scenario_dir=f"{drills_root}/phase1/1_3_navigate_obstacles",
            action_mask="move_only",
            max_iters=50,
            lr=3e-5,
            episodes_per_scenario=5,
            time_penalty=-0.002,
            win_bonus=0.3,
            reward_scale=1.0,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            prerequisites=["1.1_reach_static"],
            description="Navigate around obstacles to target",
        ),
        DrillConfig(
            name="1.4_navigate_time_pressure",
            phase=1,
            scenario_dir=f"{drills_root}/phase1/1_4_navigate_time_pressure",
            action_mask="move_only",
            max_iters=50,
            lr=1e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            prerequisites=["1.3_navigate_obstacles"],
            description="Navigate to target under expanding damage zone",
        ),
        DrillConfig(
            name="1.5_navigate_moving_obstacles",
            phase=1,
            scenario_dir=f"{drills_root}/phase1/1_5_navigate_moving_obstacles",
            action_mask="move_only",
            max_iters=50,
            lr=1e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            prerequisites=["1.3_navigate_obstacles"],
            description="Navigate through moving obstacles",
        ),
        DrillConfig(
            name="1.6_react_dynamic_terrain",
            phase=1,
            scenario_dir=f"{drills_root}/phase1/1_6_react_dynamic_terrain",
            action_mask="move_only",
            max_iters=50,
            lr=1e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            prerequisites=["1.3_navigate_obstacles"],
            description="React to dynamic terrain changes and reroute",
        ),

        # ── Phase 2: Spatial Awareness (move_only, enemies present) ──
        DrillConfig(
            name="2.1_maintain_distance",
            phase=2,
            scenario_dir=f"{drills_root}/phase2/2_1_maintain_distance",
            action_mask="move_only",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.001,  # halved for survival drill
            win_bonus=0.3,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            prerequisites=["1.1_reach_static"],
            description="Kite a melee enemy without taking damage",
        ),
        DrillConfig(
            name="2.2_dodge_zones",
            phase=2,
            scenario_dir=f"{drills_root}/phase2/2_2_dodge_zones",
            action_mask="move_only",
            max_iters=25,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            prerequisites=["1.3_navigate_obstacles"],
            description="Navigate through danger zones without taking damage",
        ),
        DrillConfig(
            name="2.3_dodge_telegraphed",
            phase=2,
            scenario_dir=f"{drills_root}/phase2/2_3_dodge_telegraphed",
            action_mask="move_only",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.001,
            win_bonus=0.3,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            prerequisites=["2.2_dodge_zones"],
            description="Dodge telegraphed AoE abilities",
        ),
        DrillConfig(
            name="2.4_kite_melee",
            phase=2,
            scenario_dir=f"{drills_root}/phase2/2_4_kite_melee",
            action_mask="move_only",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.001,  # halved
            win_bonus=0.2,
            trainable_modules=["entity_encoder", "latent_interface", "move_head"],
            prerequisites=["2.1_maintain_distance"],
            description="Kite melee enemy at max attack range, kill via auto-attacks",
        ),

        # ── Phase 3: Target Selection (move_attack) ──
        DrillConfig(
            name="3.1_kill_stationary",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_1_kill_stationary",
            action_mask="move_attack",
            max_iters=15,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["1.1_reach_static"],
            description="Attack and kill a stationary dummy",
        ),
        DrillConfig(
            name="3.2_kill_moving",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_2_kill_moving",
            action_mask="move_attack",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["3.1_kill_stationary", "1.2_reach_moving"],
            description="Chase and kill a moving target",
        ),
        DrillConfig(
            name="3.3_prioritize_low_hp",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_3_prioritize_low_hp",
            action_mask="move_attack",
            max_iters=20,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["3.1_kill_stationary"],
            description="Kill the low-HP enemy first",
        ),
        DrillConfig(
            name="3.4_prioritize_high_threat",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_4_prioritize_high_threat",
            action_mask="move_attack",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["3.3_prioritize_low_hp"],
            description="Kill the high-DPS enemy first to protect ally",
        ),
        DrillConfig(
            name="3.5_kill_healer",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_5_kill_healer",
            action_mask="move_attack",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["3.4_prioritize_high_threat"],
            description="Kill healer first despite DPS being closer",
        ),
        DrillConfig(
            name="3.6_protect_ally",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_6_protect_ally",
            action_mask="move_attack",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["3.4_prioritize_high_threat"],
            description="Attack the enemy threatening your ally",
        ),
        DrillConfig(
            name="3.7_react_threat_change",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_7_react_threat_change",
            action_mask="move_attack",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["3.6_protect_ally"],
            description="Switch target when enemy starts casting",
        ),
        DrillConfig(
            name="3.8_multi_threat",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_8_multi_threat",
            action_mask="move_attack",
            max_iters=40,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["3.5_kill_healer", "3.6_protect_ally", "3.7_react_threat_change"],
            description="Assess and prioritize multiple threats in correct order",
        ),
        DrillConfig(
            name="3.9_horde_combat",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_9_horde_combat",
            action_mask="move_attack",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["3.3_prioritize_low_hp", "2.4_kite_melee"],
            description="Fight multiple weak enemies without getting surrounded",
        ),
        DrillConfig(
            name="3.10_use_terrain",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_10_use_terrain",
            action_mask="move_attack",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.2,
            prerequisites=["1.3_navigate_obstacles", "3.1_kill_stationary"],
            description="Use chokepoints and terrain for tactical advantage",
        ),
        DrillConfig(
            name="3.11_elevation_advantage",
            phase=3,
            scenario_dir=f"{drills_root}/phase3/3_11_elevation_advantage",
            action_mask="move_attack",
            max_iters=30,
            lr=3e-4,
            time_penalty=-0.002,
            win_bonus=0.2,
            prerequisites=["3.10_use_terrain"],
            description="Hold elevation advantage over enemies",
        ),

        # ── Phase 4: Ability Usage (all) ──
        DrillConfig(
            name="4.1_heal_low_ally",
            phase=4,
            scenario_dir=f"{drills_root}/phase4/4_1_heal_low_ally",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["3.1_kill_stationary"],
            description="Use heal ability on low-HP ally",
        ),
        DrillConfig(
            name="4.2_cc_enemy",
            phase=4,
            scenario_dir=f"{drills_root}/phase4/4_2_cc_enemy",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.15,
            prerequisites=["3.6_protect_ally"],
            description="Use CC ability on enemy threatening ally",
        ),
        DrillConfig(
            name="4.3_interrupt_cast",
            phase=4,
            scenario_dir=f"{drills_root}/phase4/4_3_interrupt_cast",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.4,
            prerequisites=["4.2_cc_enemy"],
            description="Interrupt enemy cast with CC ability",
        ),
        DrillConfig(
            name="4.3b_selective_interrupt",
            phase=4,
            scenario_dir=f"{drills_root}/phase4/4_3b_selective_interrupt",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["4.3_interrupt_cast"],
            description="Save interrupt for dangerous cast, ignore weak cast",
        ),
        DrillConfig(
            name="4.4_cc_burst",
            phase=4,
            scenario_dir=f"{drills_root}/phase4/4_4_cc_burst",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.2,
            prerequisites=["4.2_cc_enemy", "3.1_kill_stationary"],
            description="Stun then burst during CC window",
        ),
        DrillConfig(
            name="4.5_knockback_wall",
            phase=4,
            scenario_dir=f"{drills_root}/phase4/4_5_knockback_wall",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["4.2_cc_enemy", "1.3_navigate_obstacles"],
            description="Knockback enemy into a wall",
        ),
        DrillConfig(
            name="4.6_aoe_positioning",
            phase=4,
            scenario_dir=f"{drills_root}/phase4/4_6_aoe_positioning",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.15,
            prerequisites=["1.3_navigate_obstacles", "3.1_kill_stationary"],
            description="Position for optimal AoE coverage",
        ),
        DrillConfig(
            name="4.7_cooldown_management",
            phase=4,
            scenario_dir=f"{drills_root}/phase4/4_7_cooldown_management",
            action_mask="all",
            max_iters=40,
            lr=2e-4,
            time_penalty=-0.001,  # halved — endurance drill
            win_bonus=0.2,
            prerequisites=["4.4_cc_burst"],
            description="Manage ability cooldowns across multiple waves",
        ),
        DrillConfig(
            name="4.8_respect_enemy_cds",
            phase=4,
            scenario_dir=f"{drills_root}/phase4/4_8_respect_enemy_cds",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.001,  # halved — endurance drill
            win_bonus=0.3,
            prerequisites=["4.4_cc_burst", "2.1_maintain_distance"],
            description="Engage when enemy CCs are on cooldown, disengage when ready",
        ),

        # ── Phase 5: Team Coordination (all) ──
        DrillConfig(
            name="5.1_focus_fire",
            phase=5,
            scenario_dir=f"{drills_root}/phase5/5_1_focus_fire",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.2,
            prerequisites=["3.3_prioritize_low_hp", "3.4_prioritize_high_threat"],
            description="Both allies attack the same target",
        ),
        DrillConfig(
            name="5.2_chain_cc",
            phase=5,
            scenario_dir=f"{drills_root}/phase5/5_2_chain_cc",
            action_mask="all",
            max_iters=40,
            lr=2e-4,
            time_penalty=-0.001,  # halved — endurance drill
            win_bonus=0.15,
            prerequisites=["4.2_cc_enemy", "4.4_cc_burst"],
            description="Chain CC abilities without overlap",
        ),
        DrillConfig(
            name="5.3_peel_for_carry",
            phase=5,
            scenario_dir=f"{drills_root}/phase5/5_3_peel_for_carry",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.2,
            prerequisites=["3.6_protect_ally", "4.2_cc_enemy"],
            description="Tank peels for DPS ally",
        ),
        DrillConfig(
            name="5.4_dive_coordination",
            phase=5,
            scenario_dir=f"{drills_root}/phase5/5_4_dive_coordination",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.2,
            prerequisites=["5.1_focus_fire", "3.5_kill_healer"],
            description="Coordinate dive onto enemy healer",
        ),
        DrillConfig(
            name="5.5_engage_disengage",
            phase=5,
            scenario_dir=f"{drills_root}/phase5/5_5_engage_disengage",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.001,  # halved — endurance drill
            win_bonus=0.3,
            prerequisites=["5.1_focus_fire", "4.8_respect_enemy_cds"],
            description="Engage when safe, disengage when enemy CCs ready",
        ),
        DrillConfig(
            name="5.6_horde_defense",
            phase=5,
            scenario_dir=f"{drills_root}/phase5/5_6_horde_defense",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.2,
            prerequisites=["3.9_horde_combat", "3.10_use_terrain", "5.1_focus_fire"],
            description="Tank and DPS coordinate defense against horde",
        ),
        DrillConfig(
            name="5.7_dynamic_terrain_coord",
            phase=5,
            scenario_dir=f"{drills_root}/phase5/5_7_dynamic_terrain_coord",
            action_mask="all",
            max_iters=30,
            lr=2e-4,
            time_penalty=-0.002,
            win_bonus=0.3,
            prerequisites=["1.6_react_dynamic_terrain", "5.4_dive_coordination"],
            description="Use dynamic terrain to isolate and kill enemy healer",
        ),
    ]


# ---------------------------------------------------------------------------
# Reward shaping (Python-side)
# ---------------------------------------------------------------------------

def _load_drill_targets(scenario_dir: str) -> dict[str, list[float]]:
    """Load target_position from all TOMLs in a drill directory."""
    import tomllib
    targets = {}
    d = Path(scenario_dir)
    if d.is_dir():
        for f in d.glob("*.toml"):
            with open(f, "rb") as fh:
                cfg = tomllib.load(fh).get("scenario", {})
            name = cfg.get("name", "")
            tp = cfg.get("target_position")
            if name and tp:
                targets[name] = tp
    return targets


def _gauss(x: float, mu: float, sigma: float) -> float:
    """Gaussian reward kernel."""
    return math.exp(-((x - mu) ** 2) / (2 * sigma * sigma))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _entity_distance(ent_a: list[float], ent_b: list[float]) -> float:
    """Euclidean distance between two entities using pos features [5],[6]."""
    ax, ay = ent_a[5] * 20.0, ent_a[6] * 20.0
    bx, by = ent_b[5] * 20.0, ent_b[6] * 20.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _self_pos(entities: list[list[float]]) -> tuple[float, float]:
    """Extract self position from entity features."""
    if entities:
        return entities[0][5] * 20.0, entities[0][6] * 20.0
    return 0.0, 0.0


def _dist_to_point(entities: list[list[float]], target: list[float]) -> float:
    """Distance from self entity to target point."""
    if not entities or not target:
        return 999.0
    sx, sy = _self_pos(entities)
    dx = sx - target[0]
    dy = sy - target[1]
    return (dx * dx + dy * dy) ** 0.5


def _get_enemies(entities: list[list[float]], entity_types: list[int] | None) -> list[int]:
    """Return indices of enemy entities (type==1)."""
    if entity_types:
        return [i for i, t in enumerate(entity_types) if t == 1]
    # Fallback: enemies are slots 1-3 typically
    return [i for i in range(1, min(4, len(entities)))]


def _get_allies(entities: list[list[float]], entity_types: list[int] | None) -> list[int]:
    """Return indices of ally entities (type==2)."""
    if entity_types:
        return [i for i, t in enumerate(entity_types) if t == 2]
    return []


def _nearest_enemy_dist(entities: list[list[float]], entity_types: list[int] | None) -> float:
    """Distance to nearest enemy."""
    enemies = _get_enemies(entities, entity_types)
    if not entities or not enemies:
        return 999.0
    return min(_entity_distance(entities[0], entities[i]) for i in enemies if i < len(entities))


# ---------------------------------------------------------------------------
# Per-drill reward functions
# ---------------------------------------------------------------------------

def _reward_distance_progress(step, prev_step, drill, ep_state):
    """Generic distance-to-target progress shaping (Phase 1 drills)."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    target_pos = ep_state.get("target_pos")

    if target_pos and entities:
        curr_dist = _dist_to_point(entities, target_pos)
        start_dist = ep_state.get("start_dist", curr_dist)
        if ep_state.get("step_idx", 0) == 0:
            ep_state["start_dist"] = curr_dist
            start_dist = curr_dist

        prev_dist = ep_state.get("prev_dist", curr_dist)
        if start_dist > 0.1:
            # Raw distance closed this step (not normalized by start_dist)
            # Typical step closes ~0.1 units, so reward ≈ 0.01/step
            dist_closed = max(0, prev_dist - curr_dist)
            r += dist_closed * 0.1  # ~0.01/step when moving toward target
        ep_state["prev_dist"] = curr_dist

    return r


def _reward_fn_1_1(step, prev_step, drill, ep_state):
    """1.1 Reach Static: strong distance progress."""
    return _reward_distance_progress(step, prev_step, drill, ep_state)


def _reward_fn_1_2(step, prev_step, drill, ep_state):
    """1.2 Reach Moving Target: closing rate relative to prev_dist."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    target_pos = ep_state.get("target_pos")

    if target_pos and entities:
        curr_dist = _dist_to_point(entities, target_pos)
        prev_dist = ep_state.get("prev_dist", curr_dist)
        if prev_dist > 0.1:
            closing = _clamp01((prev_dist - curr_dist) / prev_dist) * 0.003
            r += closing
        ep_state["prev_dist"] = curr_dist

    return r


def _reward_fn_1_3(step, prev_step, drill, ep_state):
    """1.3 Navigate Obstacles: distance progress + wall bump penalty."""
    r = _reward_distance_progress(step, prev_step, drill, ep_state)
    # Wall bump: if position didn't change but agent tried to move, penalize
    if prev_step:
        prev_ents = prev_step.get("entities", [])
        curr_ents = step.get("entities", [])
        if prev_ents and curr_ents:
            px, py = _self_pos(prev_ents)
            cx, cy = _self_pos(curr_ents)
            if abs(px - cx) < 0.01 and abs(py - cy) < 0.01:
                action = prev_step.get("action_type", -1)
                if action != 8:  # not hold
                    r -= 0.05  # wall bump
    return r


def _reward_fn_1_4(step, prev_step, drill, ep_state):
    """1.4 Navigate Under Time Pressure: progress + zone damage penalty."""
    r = _reward_distance_progress(step, prev_step, drill, ep_state)
    # Zone damage: check if step_reward from engine is negative (took damage)
    raw = step.get("step_reward", 0.0)
    if raw < -0.01:
        r -= 0.02  # in damage zone
    return r


def _reward_fn_1_5(step, prev_step, drill, ep_state):
    """1.5 Navigate Moving Obstacles: same as 1.3."""
    return _reward_fn_1_3(step, prev_step, drill, ep_state)


def _reward_fn_1_6(step, prev_step, drill, ep_state):
    """1.6 React to Dynamic Terrain: same as 1.3 with wall-stuck penalty."""
    return _reward_fn_1_3(step, prev_step, drill, ep_state)


def _reward_fn_2_1(step, prev_step, drill, ep_state):
    """2.1 Maintain Distance: Gaussian at ideal kite range + corner penalty."""
    r = drill.time_penalty  # -0.001
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")

    if entities and len(entities) > 1:
        enemy_dist = _nearest_enemy_dist(entities, entity_types)
        r += _gauss(enemy_dist, 3.0, 2.0) * 0.004

        # Penalize distance from room center (discourages running to walls)
        # Positions are /20, center of 100x100 room = 50/20 = 2.5
        sx, sy = _self_pos(entities)
        center_dist = ((sx - 2.5)**2 + (sy - 2.5)**2)**0.5
        r -= 0.001 * min(1.0, center_dist / 2.0)  # ramps from 0 at center to -0.001 at edge

    # Hit penalty (proportional)
    raw = step.get("step_reward", 0.0)
    if raw < -0.01:
        r -= 0.05 * abs(raw)

    return r


def _reward_fn_2_2(step, prev_step, drill, ep_state):
    """2.2 Dodge Zones: progress + zone penalty."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    target_pos = ep_state.get("target_pos")

    if target_pos and entities:
        curr_dist = _dist_to_point(entities, target_pos)
        start_dist = ep_state.get("start_dist", curr_dist)
        if ep_state.get("step_idx", 0) == 0:
            ep_state["start_dist"] = curr_dist
            start_dist = curr_dist
        prev_dist = ep_state.get("prev_dist", curr_dist)
        if start_dist > 0.1:
            progress = _clamp01((prev_dist - curr_dist) / start_dist) * 0.004
            r += progress
        ep_state["prev_dist"] = curr_dist

    # Zone penalty from engine reward
    raw = step.get("step_reward", 0.0)
    if raw < -0.01:
        r -= 0.02

    return r


def _reward_fn_2_3(step, prev_step, drill, ep_state):
    """2.3 Dodge Telegraphed: survive + dodge shaping during telegraph."""
    r = drill.time_penalty  # -0.001
    r += 0.001  # survive bonus per tick

    # Dodge shaping: if enemy is casting (feature[24]=is_casting), reward distance
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    if entities and len(entities) > 1:
        enemies = _get_enemies(entities, entity_types)
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 24:
                is_casting = entities[ei][24] if len(entities[ei]) > 24 else 0.0
                if is_casting > 0.5:
                    dist = _entity_distance(entities[0], entities[ei])
                    r += 0.003 * _clamp01(dist / 5.0)

    # Hit penalty
    raw = step.get("step_reward", 0.0)
    if raw < -0.01:
        r -= 0.06

    return r


def _reward_fn_2_4(step, prev_step, drill, ep_state):
    """2.4 Kite Melee: Gaussian at attack range + hit bonuses."""
    r = drill.time_penalty  # -0.001
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")

    if entities and len(entities) > 1:
        enemy_dist = _nearest_enemy_dist(entities, entity_types)
        # Attack range from features: feature[13] = attack_range / 10
        atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
        r += _gauss(enemy_dist, atk_range - 0.5, 2.0) * 0.003

        # Too far penalty
        if enemy_dist > atk_range:
            r -= 0.002 * (enemy_dist - atk_range) / max(atk_range, 1.0)

    # Auto-attack hit bonus (positive raw step reward = dealing damage)
    raw = step.get("step_reward", 0.0)
    if raw > 0.01:
        r += 0.02

    # Hit taken penalty
    if raw < -0.01:
        r -= 0.02 * abs(raw)

    return r


def _reward_fn_3_1(step, prev_step, drill, ep_state):
    """3.1 Kill Stationary: attack reward + in-range + holding penalty."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    combat_type = step.get("combat_type", -1)  # 0=attack, 1=hold

    if entities and len(entities) > 1:
        enemy_dist = _nearest_enemy_dist(entities, entity_types)
        atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
        in_range = enemy_dist <= atk_range + 0.5

        if combat_type == 0:  # attacking
            r += 0.003
        if in_range:
            r += 0.001
            if combat_type == 1 or combat_type == -1:  # holding while in range
                r -= 0.002

    return r


def _reward_fn_3_2(step, prev_step, drill, ep_state):
    """3.2 Kill Moving Target: range tracking + closing + attacking."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    combat_type = step.get("combat_type", -1)

    if entities and len(entities) > 1:
        enemy_dist = _nearest_enemy_dist(entities, entity_types)
        atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
        in_range = enemy_dist <= atk_range + 0.5

        if in_range:
            r += 0.002
        else:
            # Closing reward when out of range
            prev_dist = ep_state.get("prev_enemy_dist", enemy_dist)
            if prev_dist > 0.1:
                closing = _clamp01((prev_dist - enemy_dist) / prev_dist) * 0.002
                r += closing

        if combat_type == 0:  # attacking / dealing damage
            r += 0.002

        ep_state["prev_enemy_dist"] = enemy_dist

    return r


def _reward_fn_3_3(step, prev_step, drill, ep_state):
    """3.3 Prioritize Low HP: reward attacking low HP target, penalize wrong."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    target_idx = step.get("target_idx", -1)

    enemies = _get_enemies(entities, entity_types)
    if len(enemies) >= 2 and entities:
        # Find lowest HP enemy
        enemy_hps = []
        for ei in enemies:
            if ei < len(entities):
                hp = entities[ei][0]  # feature[0] = hp_pct
                enemy_hps.append((ei, hp))
        enemy_hps.sort(key=lambda x: x[1])

        if enemy_hps:
            low_hp_idx = enemy_hps[0][0]
            combat_type = step.get("combat_type", -1)
            if combat_type == 0:  # attacking
                if target_idx == low_hp_idx:
                    r += 0.005  # correct target
                elif enemy_hps[0][1] > 0.01:  # low HP enemy still alive
                    r -= 0.003  # wrong target

    return r


def _reward_fn_3_4(step, prev_step, drill, ep_state):
    """3.4 Prioritize High Threat: attack high-DPS enemy first."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    target_idx = step.get("target_idx", -1)
    combat_type = step.get("combat_type", -1)

    enemies = _get_enemies(entities, entity_types)
    allies = _get_allies(entities, entity_types)

    if len(enemies) >= 2 and entities:
        # Find highest DPS enemy (feature[12] = dps / 30)
        enemy_dps = []
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 12:
                dps = entities[ei][12]
                enemy_dps.append((ei, dps))
        enemy_dps.sort(key=lambda x: -x[1])  # highest first

        if enemy_dps:
            high_threat_idx = enemy_dps[0][0]
            if combat_type == 0:
                if target_idx == high_threat_idx:
                    r += 0.004
                elif entities[high_threat_idx][0] > 0.01:  # still alive
                    r -= 0.002

    # Closing reward when out of range of high-threat target
    if enemy_dps and entities:
        high_threat_idx = enemy_dps[0][0]
        if high_threat_idx < len(entities) and entities[high_threat_idx][0] > 0.01:
            dist = _entity_distance(entities[0], entities[high_threat_idx])
            atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
            if dist > atk_range:
                prev_dist = ep_state.get("prev_threat_dist", dist)
                if prev_dist > 0.1:
                    closing_rate = (prev_dist - dist) / prev_dist
                    r += 0.002 * _clamp01(closing_rate)
                ep_state["prev_threat_dist"] = dist

    # Ally alive bonus
    for ai in allies:
        if ai < len(entities) and entities[ai][0] > 0.01:
            r += 0.001
            break

    return r


def _reward_fn_3_5(step, prev_step, drill, ep_state):
    """3.5 Kill the Healer: attack healer first."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    target_idx = step.get("target_idx", -1)
    combat_type = step.get("combat_type", -1)

    enemies = _get_enemies(entities, entity_types)
    if len(enemies) >= 2 and entities:
        # Healer: enemy with highest heal_amount (feature[18] = heal_amount / 50)
        healer_idx = -1
        best_heal = 0.0
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 18:
                heal = entities[ei][18]
                if heal > best_heal:
                    best_heal = heal
                    healer_idx = ei

        if healer_idx >= 0 and combat_type == 0:
            if target_idx == healer_idx:
                r += 0.004  # attacking healer
            else:
                r += 0.001  # attacking DPS (some reward)

        # Closing toward healer when out of range
        if healer_idx >= 0:
            dist = _entity_distance(entities[0], entities[healer_idx])
            atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
            if dist > atk_range:
                prev_dist = ep_state.get("prev_healer_dist", dist)
                if prev_dist > dist:
                    r += 0.002
                ep_state["prev_healer_dist"] = dist

    return r


def _reward_fn_3_6(step, prev_step, drill, ep_state):
    """3.6 Protect Ally: attack the enemy threatening ally."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    target_idx = step.get("target_idx", -1)
    combat_type = step.get("combat_type", -1)

    enemies = _get_enemies(entities, entity_types)
    allies = _get_allies(entities, entity_types)

    if enemies and allies and entities:
        # Find enemy closest to ally (threatening)
        threat_idx = -1
        min_dist = 999.0
        for ai in allies:
            if ai < len(entities) and entities[ai][0] > 0.01:
                for ei in enemies:
                    if ei < len(entities) and entities[ei][0] > 0.01:
                        d = _entity_distance(entities[ai], entities[ei])
                        if d < min_dist:
                            min_dist = d
                            threat_idx = ei

        if threat_idx >= 0 and combat_type == 0:
            if target_idx == threat_idx:
                r += 0.004
            elif entities[threat_idx][0] > 0.01:
                r -= 0.002

        # Ally alive bonus
        for ai in allies:
            if ai < len(entities) and entities[ai][0] > 0.01:
                r += 0.001
                break

    return r


def _reward_fn_3_7(step, prev_step, drill, ep_state):
    """3.7 React to Threat Change: switch target when enemy starts casting."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    target_idx = step.get("target_idx", -1)
    combat_type = step.get("combat_type", -1)

    enemies = _get_enemies(entities, entity_types)
    if entities:
        # Find casting enemy
        caster_idx = -1
        cast_progress = 0.0
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 25:
                is_casting = entities[ei][24] if len(entities[ei]) > 24 else 0.0
                if is_casting > 0.5:
                    caster_idx = ei
                    cast_progress = entities[ei][25] if len(entities[ei]) > 25 else 0.0
                    break

        if caster_idx >= 0:
            # Post-cast-start: reward attacking caster with urgency scaling
            if combat_type == 0 and target_idx == caster_idx:
                r += 0.005 * (1.0 + cast_progress)
            elif combat_type != 0 or target_idx != caster_idx:
                r -= 0.003 * (1.0 + cast_progress)
        else:
            # Pre-cast: generic closing reward
            if combat_type == 0:
                r += 0.001

    return r


def _reward_fn_3_8(step, prev_step, drill, ep_state):
    """3.8 Multi-Threat Assessment: kill priority order B(healer) > A > C."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    target_idx = step.get("target_idx", -1)
    combat_type = step.get("combat_type", -1)

    enemies = _get_enemies(entities, entity_types)
    allies = _get_allies(entities, entity_types)

    if len(enemies) >= 2 and entities:
        # Healer is priority (highest heal stat)
        healer_idx = -1
        best_heal = 0.0
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 18:
                heal = entities[ei][18]
                if heal > best_heal:
                    best_heal = heal
                    healer_idx = ei

        alive_enemies = [ei for ei in enemies if ei < len(entities) and entities[ei][0] > 0.01]

        if combat_type == 0:
            if healer_idx >= 0 and healer_idx in alive_enemies and target_idx == healer_idx:
                r += 0.004
            elif healer_idx >= 0 and healer_idx not in alive_enemies:
                # Healer dead, reward attacking anyone
                r += 0.002
            else:
                r -= 0.002

        # Closing toward healer
        if healer_idx >= 0 and healer_idx in alive_enemies:
            dist = _entity_distance(entities[0], entities[healer_idx])
            atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
            if dist > atk_range:
                r += 0.002

    # Ally alive
    for ai in allies:
        if ai < len(entities) and entities[ai][0] > 0.01:
            r += 0.001
            break

    return r


def _reward_fn_3_9(step, prev_step, drill, ep_state):
    """3.9 Horde Combat: avoid surround + kite at range."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")

    if entities:
        enemies = _get_enemies(entities, entity_types)
        # Count enemies within 2 units
        close_enemies = 0
        nearest_dist = 999.0
        for ei in enemies:
            if ei < len(entities) and entities[ei][0] > 0.01:
                d = _entity_distance(entities[0], entities[ei])
                if d < 2.0:
                    close_enemies += 1
                nearest_dist = min(nearest_dist, d)

        # Surround penalty
        r -= 0.003 * max(0, close_enemies - 1)

        # Kite reward
        atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
        if nearest_dist < 999.0:
            r += _gauss(nearest_dist, atk_range - 0.5, 2.0) * 0.002

    # Kill bonus from raw reward
    raw = step.get("step_reward", 0.0)
    if raw > 0.05:
        r += 0.05

    # On-hit penalty (taking damage)
    if raw < -0.01:
        r -= 0.01 * abs(raw)

    # Death penalty on defeat
    outcome = step.get("outcome")
    if outcome == "defeat":
        r -= 0.3

    return r


def _reward_fn_3_10(step, prev_step, drill, ep_state):
    """3.10 Use Terrain: chokepoint + funnel rewards."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    combat_type = step.get("combat_type", -1)

    if entities:
        enemies = _get_enemies(entities, entity_types)
        # Proxy: reward being near walls while enemies are in range
        sx, sy = _self_pos(entities)
        near_walls = sum([sx < 2.0, sx > 18.0, sy < 2.0, sy > 18.0,
                          abs(sx - 10.0) < 1.0, abs(sy - 10.0) < 1.0])

        alive_enemies = [ei for ei in enemies if ei < len(entities) and entities[ei][0] > 0.01]
        in_range_enemies = 0
        for ei in alive_enemies:
            d = _entity_distance(entities[0], entities[ei])
            atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
            if d <= atk_range + 0.5:
                in_range_enemies += 1

        if near_walls > 0 and in_range_enemies > 0:
            r += 0.003

        # Funnel: only 1 enemy in melee while more alive
        melee_enemies = sum(1 for ei in alive_enemies
                           if ei < len(entities) and _entity_distance(entities[0], entities[ei]) < 2.0)
        if melee_enemies <= 1 and len(alive_enemies) > 1:
            r += 0.002

    raw = step.get("step_reward", 0.0)
    if raw > 0.05:
        r += 0.06

    return r


def _reward_fn_3_11(step, prev_step, drill, ep_state):
    """3.11 Elevation Advantage: reward being higher than enemy."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")

    if entities and len(entities) > 1:
        enemies = _get_enemies(entities, entity_types)
        # Elevation feature: feature[7] = elevation / 10 (if available)
        self_elev = entities[0][7] * 10.0 if len(entities[0]) > 7 else 0.0
        for ei in enemies:
            if ei < len(entities) and entities[ei][0] > 0.01:
                enemy_elev = entities[ei][7] * 10.0 if len(entities[ei]) > 7 else 0.0
                d = _entity_distance(entities[0], entities[ei])
                atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
                if self_elev > enemy_elev and d <= atk_range + 0.5:
                    r += 0.003
                    break

        # Descend penalty
        if prev_step:
            prev_ents = prev_step.get("entities", [])
            if prev_ents:
                prev_elev = prev_ents[0][7] * 10.0 if len(prev_ents[0]) > 7 else 0.0
                if self_elev < prev_elev - 0.1:
                    r -= 0.01

    raw = step.get("step_reward", 0.0)
    if raw > 0.05:
        r += 0.1

    return r


def _reward_fn_4_1(step, prev_step, drill, ep_state):
    """4.1 Heal Low Ally: reward healing, penalize not healing low ally."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    allies = _get_allies(entities, entity_types)

    if allies and entities:
        for ai in allies:
            if ai < len(entities):
                ally_hp = entities[ai][0]
                # Check if ally HP increased vs prev step
                if prev_step:
                    prev_ents = prev_step.get("entities", [])
                    if prev_ents and ai < len(prev_ents):
                        prev_hp = prev_ents[ai][0]
                        hp_gain = max(0, ally_hp - prev_hp)
                        r += 0.05 * hp_gain

                # Penalty for not healing when ally low
                if ally_hp < 0.5:
                    # Check if heal is ready (feature[20] = heal_cd_remaining_pct)
                    heal_cd = entities[0][20] if len(entities[0]) > 20 else 1.0
                    if heal_cd < 0.01:  # heal ready
                        r -= 0.005

    return r


def _reward_fn_4_2(step, prev_step, drill, ep_state):
    """4.2 CC on Enemy: reward stunning threatening enemy."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)
    allies = _get_allies(entities, entity_types)

    if enemies and allies and entities:
        for ei in enemies:
            if ei < len(entities) and entities[ei][0] > 0.01:
                for ai in allies:
                    if ai < len(entities) and entities[ai][0] > 0.01:
                        enemy_to_ally = _entity_distance(entities[ei], entities[ai])
                        # Threat rising
                        r += 0.002 * _clamp01(1.0 - enemy_to_ally / 6.0)

                        # Stun idle penalty (only when enemy close to ally)
                        if enemy_to_ally < 3.0:
                            cc_cd = entities[0][22] if len(entities[0]) > 22 else 1.0
                            if cc_cd < 0.01:  # CC ready
                                r -= 0.003
                        break
                break

    # Stun bonus from raw reward
    raw = step.get("step_reward", 0.0)
    if raw > 0.1:
        r += 0.15

    # Ally death
    if allies:
        any_alive = False
        for ai in allies:
            if ai < len(entities) and entities[ai][0] > 0.01:
                any_alive = True
                break
        if not any_alive:
            r -= 0.3

    return r


def _reward_fn_4_3(step, prev_step, drill, ep_state):
    """4.3 Interrupt Cast: urgency penalty for inaction during cast."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)

    if entities:
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 25:
                is_casting = entities[ei][24] if len(entities[ei]) > 24 else 0.0
                cast_progress = entities[ei][25] if len(entities[ei]) > 25 else 0.0
                if is_casting > 0.5:
                    # Inaction penalty grows with cast progress
                    combat_type = step.get("combat_type", -1)
                    target_idx = step.get("target_idx", -1)
                    cc_cd = entities[0][22] if len(entities[0]) > 22 else 1.0
                    if cc_cd < 0.01 and (combat_type != 0 or target_idx != ei):
                        r -= 0.005 * cast_progress
                    break

    # Interrupt bonus
    raw = step.get("step_reward", 0.0)
    if raw > 0.1:
        r += 0.4 * (1.0 - ep_state.get("cast_progress_at_interrupt", 0.5))

    # Cast completed penalty: defeat/timeout with enemy alive = failed to interrupt
    outcome = step.get("outcome")
    if outcome in ("defeat", "timeout"):
        enemies = _get_enemies(entities, entity_types)
        any_enemy_alive = any(
            ei < len(entities) and entities[ei][0] > 0.01 for ei in enemies
        )
        if any_enemy_alive:
            r -= 0.5

    return r


def _reward_fn_4_3b(step, prev_step, drill, ep_state):
    """4.3b Selective Interrupt: save stun for dangerous cast."""
    r = drill.time_penalty
    # This drill requires detailed cast tracking from step data
    # Use generic shaping with interrupt bonus
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)

    if entities:
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 25:
                is_casting = entities[ei][24] if len(entities[ei]) > 24 else 0.0
                cast_progress = entities[ei][25] if len(entities[ei]) > 25 else 0.0
                if is_casting > 0.5:
                    # Only penalize inaction on dangerous casts (high cast progress = dangerous)
                    # The scenario should set up weak cast first, then strong cast
                    cc_cd = entities[0][22] if len(entities[0]) > 22 else 1.0
                    if cc_cd < 0.01:
                        r -= 0.005 * cast_progress
                    break

    raw = step.get("step_reward", 0.0)
    if raw > 0.1:
        r += 0.3
    if raw < -0.1:
        r -= 0.2

    # Dangerous cast completed penalty: defeat/timeout with enemy alive
    outcome = step.get("outcome")
    if outcome in ("defeat", "timeout"):
        enemies_alive = any(
            ei < len(entities) and entities[ei][0] > 0.01
            for ei in _get_enemies(entities, entity_types)
        )
        if enemies_alive:
            r -= 0.5

    return r


def _reward_fn_4_4(step, prev_step, drill, ep_state):
    """4.4 CC Then Burst: extra reward for damage during CC."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)

    if entities:
        target_cc = False
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 26:
                cc_remaining = entities[ei][26] if len(entities[ei]) > 26 else 0.0
                if cc_remaining > 0.01:
                    target_cc = True
                    break

        raw = step.get("step_reward", 0.0)
        if raw > 0.01:  # dealing damage
            if target_cc:
                r += 0.008
            else:
                r += 0.003

    # Stun bonus
    raw = step.get("step_reward", 0.0)
    if raw > 0.15:
        r += 0.05

    # On-kill bonus (raw > 0.1 indicates kill event)
    if raw > 0.1:
        r += 0.2

    # Burst during CC bonus: ability used while enemy has CC remaining
    combat_type = step.get("combat_type", -1)
    if combat_type >= 2 and target_cc:  # combat_type >= 2 = ability usage
        r += 0.15

    return r


def _reward_fn_4_5(step, prev_step, drill, ep_state):
    """4.5 Knockback Into Wall: wall alignment + proximity shaping."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)

    if entities and enemies:
        for ei in enemies:
            if ei < len(entities) and entities[ei][0] > 0.01:
                dist = _entity_distance(entities[0], entities[ei])
                # In range bonus
                kb_range = 3.0  # typical knockback range
                if dist <= kb_range:
                    r += 0.001
                    # On-knockback base bonus: ability used near enemy
                    combat_type = step.get("combat_type", -1)
                    if combat_type >= 2:  # ability usage
                        r += 0.1

                # Wall proximity fallback: reward being near enemy when walls nearby
                ex, ey = entities[ei][5] * 20.0, entities[ei][6] * 20.0
                walls_near = sum([ex < 2.0, ex > 18.0, ey < 2.0, ey > 18.0])
                if walls_near > 0:
                    r += 0.002 * walls_near * _clamp01(1.0 - dist / kb_range)
                break

    raw = step.get("step_reward", 0.0)
    if raw > 0.2:
        r += 0.3  # wall collision bonus

    return r


def _reward_fn_4_6(step, prev_step, drill, ep_state):
    """4.6 AoE Positioning: cluster proximity."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)

    if entities and len(enemies) >= 2:
        # Compute enemy cluster center
        alive = [(entities[ei][5] * 20.0, entities[ei][6] * 20.0)
                 for ei in enemies if ei < len(entities) and entities[ei][0] > 0.01]
        if alive:
            cx = sum(p[0] for p in alive) / len(alive)
            cy = sum(p[1] for p in alive) / len(alive)
            sx, sy = _self_pos(entities)
            dist_to_cluster = ((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5
            ability_range = 5.0  # typical AoE range
            r += 0.004 * _gauss(dist_to_cluster, ability_range * 0.5, 2.0)

    raw = step.get("step_reward", 0.0)
    if raw > 0.1:
        # Scale per-hit by number of enemies hit (proxy: count alive enemies in AoE range)
        enemies_in_range = 0
        if entities and len(enemies) >= 2:
            ability_range = 5.0
            sx, sy = _self_pos(entities)
            for ei in enemies:
                if ei < len(entities) and entities[ei][0] > 0.01:
                    ex, ey = entities[ei][5] * 20.0, entities[ei][6] * 20.0
                    d = ((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5
                    if d <= ability_range:
                        enemies_in_range += 1
        r += 0.05 * max(1, enemies_in_range)

    return r


def _reward_fn_4_7(step, prev_step, drill, ep_state):
    """4.7 Cooldown Management: damage + save abilities for strong enemy."""
    r = drill.time_penalty  # -0.001

    raw = step.get("step_reward", 0.0)
    if raw > 0.01:
        r += 0.002  # damage dealt

    # Kill bonuses and ability usage handled by outcome
    if raw > 0.1:
        r += 0.05

    return r


def _reward_fn_4_8(step, prev_step, drill, ep_state):
    """4.8 Respect Enemy Cooldowns: engage when enemy CCs on cooldown."""
    r = drill.time_penalty  # -0.001
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)
    combat_type = step.get("combat_type", -1)

    if entities and enemies:
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 22 and entities[ei][0] > 0.01:
                # Enemy CC cooldown: feature[22] = control_duration or cc_cd
                # Using cc_remaining feature[26] as proxy
                enemy_cc_cd = entities[ei][26] if len(entities[ei]) > 26 else 0.0
                dist = _entity_distance(entities[0], entities[ei])
                atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
                in_range = dist <= atk_range + 0.5

                if in_range:
                    if enemy_cc_cd > 0.3:  # enemy CC on cooldown — safe
                        if combat_type == 0:
                            r += 0.004
                    elif enemy_cc_cd < 0.1:  # enemy CC ready — dangerous
                        r -= 0.005
                break

    raw = step.get("step_reward", 0.0)
    if raw > 0.01:
        r += 0.002  # damage dealt

    # Stunned penalty
    if entities and len(entities[0]) > 26:
        self_cc = entities[0][26] if len(entities[0]) > 26 else 0.0
        if self_cc > 0.01:
            r -= 0.05

    return r


def _reward_fn_5_1(step, prev_step, drill, ep_state):
    """5.1 Focus Fire: same target bonus, different target penalty."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    target_idx = step.get("target_idx", -1)
    combat_type = step.get("combat_type", -1)
    allies = _get_allies(entities, entity_types)

    # Heuristic proxy: attack lowest HP enemy (bootstrapping)
    pass_rate = ep_state.get("pass_rate", 0.0)
    hw = _clamp01(1.0 - pass_rate / 0.6)  # heuristic weight

    if combat_type == 0 and entities:
        enemies = _get_enemies(entities, entity_types)
        # Heuristic: attack lowest HP
        if enemies:
            enemy_hps = [(ei, entities[ei][0]) for ei in enemies
                        if ei < len(entities) and entities[ei][0] > 0.01]
            if enemy_hps:
                lowest = min(enemy_hps, key=lambda x: x[1])[0]
                if target_idx == lowest:
                    heuristic_r = 0.004
                else:
                    heuristic_r = -0.002
            else:
                heuristic_r = 0.0
        else:
            heuristic_r = 0.0

        # True coordination: same target as ally
        # We can't directly observe ally target from step data, so use proxy
        coord_r = 0.004 if combat_type == 0 else -0.002
        r += hw * heuristic_r + (1.0 - hw) * coord_r

    raw = step.get("step_reward", 0.0)
    if raw > 0.01:
        r += 0.001  # damage dealt

    return r


def _reward_fn_5_2(step, prev_step, drill, ep_state):
    """5.2 Chain CC: damage during CC + stun bonuses."""
    r = drill.time_penalty  # -0.001
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)

    if entities:
        target_cc = False
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 26:
                cc_remaining = entities[ei][26] if len(entities[ei]) > 26 else 0.0
                if cc_remaining > 0.01:
                    target_cc = True
                    break

        raw = step.get("step_reward", 0.0)
        if raw > 0.01 and target_cc:
            r += 0.004  # damage during CC

    # Stun bonus
    raw = step.get("step_reward", 0.0)
    if raw > 0.15:
        r += 0.08

    # On-kill bonus
    if raw > 0.1:
        r += 0.15

    return r


def _reward_fn_5_3(step, prev_step, drill, ep_state):
    """5.3 Peel for Carry: body block + CC near ally."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)
    allies = _get_allies(entities, entity_types)

    if entities and enemies and allies:
        for ei in enemies:
            if ei < len(entities) and entities[ei][0] > 0.01:
                for ai in allies:
                    if ai < len(entities) and entities[ai][0] > 0.01:
                        self_to_enemy = _entity_distance(entities[0], entities[ei])
                        ally_to_enemy = _entity_distance(entities[ai], entities[ei])
                        # Body block: self closer to enemy than ally
                        if self_to_enemy < ally_to_enemy:
                            r += 0.003
                        break
                break

        # Ally alive
        for ai in allies:
            if ai < len(entities) and entities[ai][0] > 0.01:
                r += 0.001
                break

    # Absorb damage bonus
    raw = step.get("step_reward", 0.0)
    if raw < -0.01:
        r += 0.002  # absorbing damage for ally

    # On-ally-death penalty
    outcome = step.get("outcome")
    if outcome == "defeat":
        r -= 0.3

    return r


def _reward_fn_5_4(step, prev_step, drill, ep_state):
    """5.4 Dive Coordination: attack healer together."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    combat_type = step.get("combat_type", -1)
    target_idx = step.get("target_idx", -1)
    enemies = _get_enemies(entities, entity_types)

    if entities and enemies:
        # Find healer
        healer_idx = -1
        best_heal = 0.0
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 18:
                heal = entities[ei][18]
                if heal > best_heal:
                    best_heal = heal
                    healer_idx = ei

        if healer_idx >= 0 and combat_type == 0:
            if target_idx == healer_idx:
                r += 0.004
            else:
                r += 0.001  # attacking DPS

        if healer_idx >= 0:
            dist = _entity_distance(entities[0], entities[healer_idx])
            atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
            if dist > atk_range:
                prev_dist = ep_state.get("prev_healer_dist", dist)
                if prev_dist > dist:
                    r += 0.002
                ep_state["prev_healer_dist"] = dist

    # On-ally-death penalty
    allies = _get_allies(entities, entity_types)
    if prev_step and allies:
        prev_ents = prev_step.get("entities", [])
        for ai in allies:
            if ai < len(entities) and ai < len(prev_ents):
                # Ally exists field dropped (was alive, now dead)
                if prev_ents[ai][0] > 0.01 and entities[ai][0] <= 0.01:
                    r -= 0.1

    return r


def _reward_fn_5_5(step, prev_step, drill, ep_state):
    """5.5 Engage/Disengage: attack when safe, retreat when enemy CCs ready."""
    r = drill.time_penalty  # -0.001
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    enemies = _get_enemies(entities, entity_types)
    combat_type = step.get("combat_type", -1)

    if entities and enemies:
        for ei in enemies:
            if ei < len(entities) and entities[ei][0] > 0.01 and len(entities[ei]) > 26:
                dist = _entity_distance(entities[0], entities[ei])
                atk_range = entities[0][13] * 10.0 if len(entities[0]) > 13 else 4.0
                in_range = dist <= atk_range + 0.5
                enemy_cc_cd = entities[ei][26] if len(entities[ei]) > 26 else 0.0

                if in_range:
                    if enemy_cc_cd > 0.3:  # safe
                        r += 0.003
                    elif enemy_cc_cd < 0.1:  # dangerous
                        r -= 0.005
                break

    raw = step.get("step_reward", 0.0)
    if raw > 0.01:
        r += 0.002

    # On-stunned penalty: self has CC remaining
    if entities and len(entities[0]) > 26:
        self_cc = entities[0][26]
        if self_cc > 0.0:
            r -= 0.03

    return r


def _reward_fn_5_6(step, prev_step, drill, ep_state):
    """5.6 Horde Defense: tank at choke, DPS behind."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    allies = _get_allies(entities, entity_types)

    # Generic: ally alive bonus + damage
    if allies:
        for ai in allies:
            if ai < len(entities) and entities[ai][0] > 0.01:
                r += 0.001
                break

    raw = step.get("step_reward", 0.0)
    if raw > 0.01:
        r += 0.003  # damage dealt
    if raw > 0.1:
        r += 0.03  # kill bonus

    return r


def _reward_fn_5_7(step, prev_step, drill, ep_state):
    """5.7 Dynamic Terrain Coord: isolate healer with terrain."""
    r = drill.time_penalty
    entities = step.get("entities", [])
    entity_types = step.get("entity_types")
    combat_type = step.get("combat_type", -1)
    target_idx = step.get("target_idx", -1)
    enemies = _get_enemies(entities, entity_types)
    allies = _get_allies(entities, entity_types)

    if entities and enemies:
        # Find healer
        healer_idx = -1
        best_heal = 0.0
        for ei in enemies:
            if ei < len(entities) and len(entities[ei]) > 18:
                heal = entities[ei][18]
                if heal > best_heal:
                    best_heal = heal
                    healer_idx = ei

        if healer_idx >= 0 and combat_type == 0 and target_idx == healer_idx:
            r += 0.004

    if allies:
        for ai in allies:
            if ai < len(entities) and entities[ai][0] > 0.01:
                r += 0.001
                break

    return r


# ---------------------------------------------------------------------------
# Reward function dispatch table
# ---------------------------------------------------------------------------

REWARD_FNS: dict[str, callable] = {
    "1.1_reach_static": _reward_fn_1_1,
    "1.2_reach_moving": _reward_fn_1_2,
    "1.3_navigate_obstacles": _reward_fn_1_3,
    "1.4_navigate_time_pressure": _reward_fn_1_4,
    "1.5_navigate_moving_obstacles": _reward_fn_1_5,
    "1.6_react_dynamic_terrain": _reward_fn_1_6,
    "2.1_maintain_distance": _reward_fn_2_1,
    "2.2_dodge_zones": _reward_fn_2_2,
    "2.3_dodge_telegraphed": _reward_fn_2_3,
    "2.4_kite_melee": _reward_fn_2_4,
    "3.1_kill_stationary": _reward_fn_3_1,
    "3.2_kill_moving": _reward_fn_3_2,
    "3.3_prioritize_low_hp": _reward_fn_3_3,
    "3.4_prioritize_high_threat": _reward_fn_3_4,
    "3.5_kill_healer": _reward_fn_3_5,
    "3.6_protect_ally": _reward_fn_3_6,
    "3.7_react_threat_change": _reward_fn_3_7,
    "3.8_multi_threat": _reward_fn_3_8,
    "3.9_horde_combat": _reward_fn_3_9,
    "3.10_use_terrain": _reward_fn_3_10,
    "3.11_elevation_advantage": _reward_fn_3_11,
    "4.1_heal_low_ally": _reward_fn_4_1,
    "4.2_cc_enemy": _reward_fn_4_2,
    "4.3_interrupt_cast": _reward_fn_4_3,
    "4.3b_selective_interrupt": _reward_fn_4_3b,
    "4.4_cc_burst": _reward_fn_4_4,
    "4.5_knockback_wall": _reward_fn_4_5,
    "4.6_aoe_positioning": _reward_fn_4_6,
    "4.7_cooldown_management": _reward_fn_4_7,
    "4.8_respect_enemy_cds": _reward_fn_4_8,
    "5.1_focus_fire": _reward_fn_5_1,
    "5.2_chain_cc": _reward_fn_5_2,
    "5.3_peel_for_carry": _reward_fn_5_3,
    "5.4_dive_coordination": _reward_fn_5_4,
    "5.5_engage_disengage": _reward_fn_5_5,
    "5.6_horde_defense": _reward_fn_5_6,
    "5.7_dynamic_terrain_coord": _reward_fn_5_7,
}


# ---------------------------------------------------------------------------
# Action flicker detection
# ---------------------------------------------------------------------------

def _detect_action_flicker(step, prev_step) -> bool:
    """Return True if action flicker penalty should apply.

    Conditions (all must be true):
    - action_type changed
    - No ability was used on prev tick
    - Target didn't change
    - No new cast started on any enemy this tick
    - No ally HP dropped below 30% this tick
    """
    if prev_step is None:
        return False

    action = step.get("action_type", -1)
    prev_action = prev_step.get("action_type", -1)
    if action == prev_action:
        return False

    # Ability used on prev tick exempts
    prev_ability = prev_step.get("ability_used", False)
    if prev_ability:
        return False

    # Target changed exempts
    target = step.get("target_idx", -1)
    prev_target = prev_step.get("target_idx", -1)
    if target != prev_target:
        return False

    # New enemy cast started exempts
    entities = step.get("entities", [])
    prev_entities = prev_step.get("entities", [])
    for i in range(1, min(len(entities), len(prev_entities))):
        if len(entities[i]) > 24 and len(prev_entities[i]) > 24:
            if entities[i][24] > 0.5 and prev_entities[i][24] < 0.5:
                return False  # new cast started

    # Ally HP dropped below 30% exempts
    entity_types = step.get("entity_types")
    if entity_types:
        for i, t in enumerate(entity_types):
            if t == 2 and i < len(entities) and i < len(prev_entities):
                if entities[i][0] < 0.3 and prev_entities[i][0] >= 0.3:
                    return False

    return True


# ---------------------------------------------------------------------------
# Phase transition helpers
# ---------------------------------------------------------------------------

# Map from action_mask to the phase where that action head first activates
_ACTION_HEAD_PHASE = {
    "move_only": 1,
    "move_attack": 3,
    "all": 4,
}


def _get_time_penalty_multiplier(drill: DrillConfig, iteration: int) -> float:
    """Phase transition warmup: halved time penalty for first 5 iters of new head."""
    head_phase = _ACTION_HEAD_PHASE.get(drill.action_mask, 1)
    if drill.phase == head_phase:
        # This drill is in the phase where a new action head activates
        if iteration <= 5:
            return 0.5
        elif iteration <= 8:
            return 0.75
    return 1.0


# ---------------------------------------------------------------------------
# Main reward transform
# ---------------------------------------------------------------------------

def transform_drill_rewards(episodes_path: str, output_path: str,
                            drill: DrillConfig, iteration: int = 1,
                            current_pass_rate: float = 0.0) -> dict:
    """Apply per-drill reward shaping to episodes. Returns stats."""
    wins = losses = timeouts = 0
    total_steps = 0
    total_reward = 0.0
    passes = 0

    # Preload target positions for distance shaping
    drill_target_positions = _load_drill_targets(drill.scenario_dir)

    # Get drill-specific reward function (fallback to generic distance)
    reward_fn = REWARD_FNS.get(drill.name)

    # Phase transition warmup
    time_penalty_mult = _get_time_penalty_multiplier(drill, iteration)

    # Action flicker applies Phase 3+
    apply_flicker = drill.phase >= 3

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
                passes += 1
            elif outcome == "Defeat":
                losses += 1
            else:
                timeouts += 1

            # Target position from scenario TOML
            target_pos = None
            scenario_name = ep.get("scenario", "")
            if drill_target_positions and scenario_name in drill_target_positions:
                target_pos = drill_target_positions[scenario_name]

            # Per-episode state for reward functions
            ep_state = {
                "target_pos": target_pos,
                "pass_rate": current_pass_rate,
                "step_idx": 0,
                "first_new_action_used": False,
            }

            for i, step in enumerate(steps):
                ep_state["step_idx"] = i
                prev_step = steps[i - 1] if i > 0 else None

                if reward_fn:
                    # Use drill-specific reward function
                    r = reward_fn(step, prev_step, drill, ep_state)
                    # Apply time penalty warmup multiplier
                    # The reward_fn already includes drill.time_penalty, so we
                    # adjust by the difference
                    if time_penalty_mult != 1.0:
                        r += drill.time_penalty * (1.0 - time_penalty_mult)
                else:
                    # Fallback: generic distance + time shaping
                    r = step.get("step_reward", 0.0) * drill.reward_scale
                    r += drill.time_penalty * time_penalty_mult

                    entities = step.get("entities", [])
                    if target_pos and entities and len(entities) > 0:
                        curr_dist = _dist_to_point(entities, target_pos)
                        if i == 0:
                            ep_state["start_dist"] = curr_dist
                            ep_state["prev_dist"] = curr_dist
                        prev_dist = ep_state.get("prev_dist", curr_dist)
                        start_dist = ep_state.get("start_dist", curr_dist)
                        if start_dist > 0.1:
                            progress = (prev_dist - curr_dist) / start_dist
                            r += progress * 1.0
                        ep_state["prev_dist"] = curr_dist

                # Action flicker penalty (Phase 3+)
                if apply_flicker and prev_step and _detect_action_flicker(step, prev_step):
                    r -= 0.01

                # Phase transition one-shot exploration bonus
                head_phase = _ACTION_HEAD_PHASE.get(drill.action_mask, 1)
                if drill.phase == head_phase and not ep_state["first_new_action_used"]:
                    action_type = step.get("action_type", -1)
                    # For move_attack (phase 3), combat actions are type 0-1
                    # For all (phase 4), ability actions are type 2+
                    if drill.action_mask == "move_attack" and action_type in (0, 1):
                        r += 0.01
                        ep_state["first_new_action_used"] = True
                    elif drill.action_mask == "all" and action_type is not None and action_type >= 2:
                        r += 0.01
                        ep_state["first_new_action_used"] = True

                # Outcome bonus on final step
                if i == n - 1:
                    if outcome == "Victory":
                        ticks = ep.get("ticks", n * 3)
                        max_ticks = ep.get("max_ticks", drill.max_iters * 200)
                        speed_bonus = max(0, 1.0 - ticks / max(max_ticks, 1))
                        r += drill.win_bonus * (0.5 + 0.5 * speed_bonus)
                    elif outcome == "Defeat":
                        r -= 0.3
                    else:  # Timeout
                        r -= 0.15

                step["step_reward"] = r

            total_steps += n
            # Track mean shaped step reward (what SAC actually trains on)
            total_reward += sum(s.get("step_reward", 0.0) for s in steps)
            fout.write(json.dumps(ep) + "\n")

    total = wins + losses + timeouts
    return {
        "wins": wins, "losses": losses, "timeouts": timeouts,
        "total": total, "passes": passes,
        "win_rate": wins / max(total, 1),
        "pass_rate": passes / max(total, 1),
        "total_steps": total_steps,
        "mean_reward": total_reward / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Drill evaluation (100/100 pass gate)
# ---------------------------------------------------------------------------

def evaluate_drill(drill: DrillConfig, checkpoint: str, common_args: dict,
                   n_trials: int = 100) -> tuple[int, int]:
    """Run drill scenarios with greedy policy, count passes.

    Returns (passes, total).
    """
    from impala_learner import generate_episodes

    eval_path = "/tmp/drill_eval.jsonl"
    gen_time, _ = generate_episodes(
        scenario_dirs=drill.scenario_dir,
        weights_path=checkpoint,  # unused when gpu_shm is set
        output_path=eval_path,
        episodes_per_scenario=1,
        threads=common_args.get("threads", 64),
        temperature=0.0,  # greedy
        step_interval=common_args.get("step_interval", 3),
        embedding_registry=common_args.get("embedding_registry"),
        gpu_shm=common_args.get("gpu_shm"),
        sims_per_thread=common_args.get("sims_per_thread", 64),
    )

    passes = 0
    total = 0
    with open(eval_path) as f:
        for line in f:
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if ep.get("outcome") == "Victory":
                passes += 1
            if total >= n_trials:
                break

    return passes, total


# ---------------------------------------------------------------------------
# Regression testing with retraining
# ---------------------------------------------------------------------------

def regression_test(passed_drills: list[DrillConfig], checkpoint: str,
                    common_args: dict, threshold: int = 95) -> list[str]:
    """Re-run all passed drills, return list of regressed drill names."""
    regressed = []
    for drill in passed_drills:
        passes, total = evaluate_drill(drill, checkpoint, common_args, n_trials=100)
        if passes < threshold:
            regressed.append(drill.name)
            print(f"  REGRESSION: {drill.name} -- {passes}/{total} (threshold: {threshold})")
    return regressed


def retrain_regressed_drill(drill: DrillConfig, model, sac_trainer,
                            checkpoint: str, common_args: dict, args,
                            embedding_registry) -> str:
    """Retrain a regressed drill until it passes 100/100 again.

    Returns path to best checkpoint.
    """
    from impala_learner import generate_episodes

    drill_dir = os.path.join(args.output_dir, drill.name, "regression_fix")
    os.makedirs(drill_dir, exist_ok=True)
    episodes_path = os.path.join(drill_dir, "episodes.jsonl")
    shaped_path = os.path.join(drill_dir, "episodes_shaped.jsonl")
    gpu_shm = common_args.get("gpu_shm")

    print(f"    Retraining {drill.name} to pass 100/100...")

    for retrain_iter in range(1, 21):
        gen_time, _ = generate_episodes(
            scenario_dirs=drill.scenario_dir,
            weights_path=checkpoint,
            output_path=episodes_path,
            episodes_per_scenario=drill.episodes_per_scenario,
            threads=args.threads,
            temperature=drill.temperature,
            step_interval=args.step_interval,
            embedding_registry=args.embedding_registry,
            gpu_shm=gpu_shm,
            sims_per_thread=args.sims_per_thread,
        )

        stats = transform_drill_rewards(episodes_path, shaped_path, drill,
                                        iteration=retrain_iter)
        print(f"      Retrain iter {retrain_iter}: pass={stats['pass_rate']:.0%}")

        sac_trainer.add_episodes(shaped_path, embedding_registry, args.external_cls_dim)
        sac_trainer.train_n_steps(500)

        import torch
        ckpt_path = os.path.join(drill_dir, "current.pt")
        torch.save({"model_state_dict": model.state_dict(),
                    "iteration": retrain_iter, "drill": drill.name}, ckpt_path)
        sac_trainer.sync_actor_to_gpu(ckpt_path)

        passes, total = evaluate_drill(drill, ckpt_path, common_args)
        if passes >= 100:
            print(f"      {drill.name} recovered: {passes}/{total}")
            return ckpt_path

    print(f"      WARNING: {drill.name} did not recover after 20 retrain iters")
    return checkpoint


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="V5 Skill Curriculum")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="generated/curriculum_v5")
    p.add_argument("--embedding-registry", default="generated/ability_embedding_registry.json")
    p.add_argument("--external-cls-dim", type=int, default=128)
    p.add_argument("--h-dim", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--entity-encoder-layers", type=int, default=4)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--sims-per-thread", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--step-interval", type=int, default=3)
    p.add_argument("--model-version", type=int, default=5, choices=[4, 5])
    p.add_argument("--start-drill", type=str, default=None,
                   help="Skip to this drill (must have passed all prerequisites)")
    p.add_argument("--skip-regression", action="store_true")
    args = p.parse_args()

    common_args = vars(args)
    curriculum = build_curriculum()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("V5 SKILL CURRICULUM")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: V{args.model_version}, d={args.d_model}")
    print(f"Output: {args.output_dir}")
    print(f"Drills: {len(curriculum)}")
    print()

    # Import training utilities
    from impala_learner import generate_episodes, start_gpu_server
    from sac_learner import SACTrainer
    if args.model_version == 5:
        from model import AbilityActorCriticV5 as ModelClass
    else:
        from model import AbilityActorCriticV4 as ModelClass
    from tokenizer import AbilityTokenizer
    import torch
    import numpy as np

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SHM_NAME = "impala_inf"

    # Start GPU server
    gpu_proc = start_gpu_server(
        args.checkpoint, shm_name=SHM_NAME,
        d_model=args.d_model, d_ff=args.d_ff,
        n_layers=args.n_layers, n_heads=args.n_heads,
        entity_encoder_layers=args.entity_encoder_layers,
        external_cls_dim=args.external_cls_dim,
        temperature=1.0,
        model_version=args.model_version,
        h_dim=args.h_dim,
    )
    gpu_shm = f"/dev/shm/{SHM_NAME}"
    common_args["gpu_shm"] = gpu_shm

    # Load embedding registry
    embedding_registry = None
    if args.embedding_registry and os.path.exists(args.embedding_registry):
        with open(args.embedding_registry) as f:
            embedding_registry = json.load(f)
        print(f"Loaded embedding registry: {len(embedding_registry.get('embeddings', {}))} abilities")

    # Build model
    tok = AbilityTokenizer()
    model_kwargs = dict(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=args.entity_encoder_layers,
        external_cls_dim=args.external_cls_dim,
        h_dim=args.h_dim,
        d_model=args.d_model, d_ff=args.d_ff,
        n_layers=args.n_layers, n_heads=args.n_heads,
    )
    model = ModelClass(**model_kwargs).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=False)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}")

    passed_drills: list[DrillConfig] = []
    checkpoint = args.checkpoint

    # Skip to start_drill if specified — auto-pass drills with existing best.pt
    start_idx = 0
    if args.start_drill:
        for i, d in enumerate(curriculum):
            if d.name == args.start_drill:
                start_idx = i
                break
            # Auto-pass drills before start_drill that have a best checkpoint
            best_path = os.path.join(args.output_dir, d.name, "best.pt")
            if os.path.exists(best_path):
                passed_drills.append(d)
                checkpoint = best_path
                print(f"  Auto-passed {d.name} (best.pt exists)")

    try:
        for drill_idx in range(start_idx, len(curriculum)):
            drill = curriculum[drill_idx]

            # Check prerequisites
            passed_names = {d.name for d in passed_drills}
            unmet = [p for p in drill.prerequisites if p not in passed_names]
            if unmet:
                print(f"\nSkipping {drill.name} -- unmet prerequisites: {unmet}")
                continue

            drill_dir = os.path.join(args.output_dir, drill.name)
            os.makedirs(drill_dir, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"DRILL: {drill.name} (Phase {drill.phase})")
            print(f"  {drill.description}")
            print(f"  Action mask: {drill.action_mask}")
            print(f"  Pass gate: {drill.pass_threshold}/100")
            print(f"  LR: {drill.lr}")
            print(f"  Training: SAC-Discrete")
            print(f"  Reward fn: {'specific' if drill.name in REWARD_FNS else 'generic'}")
            print(f"{'='*60}")

            # Freeze/unfreeze based on trainable_modules
            if drill.trainable_modules:
                for name, param in model.named_parameters():
                    param.requires_grad = any(name.startswith(m) for m in drill.trainable_modules)
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in model.parameters())
                print(f"  Trainable: {trainable:,} / {total:,} params")
            else:
                for param in model.parameters():
                    param.requires_grad = True

            # Create SAC trainer for this drill
            sac = SACTrainer(
                actor=model,
                d_model=args.d_model,
                lr_actor=drill.lr,
                lr_critic=drill.lr * 3,
                gamma=0.99, tau=0.005,
                buffer_size=200000,
                batch_size=args.batch_size,
                device=DEVICE,
            )
            print(f"  SAC: alpha={sac.log_alpha.exp().item():.3f}, "
                  f"target_H={sac.target_entropy:.2f}")

            episodes_path = os.path.join(drill_dir, "episodes.jsonl")
            shaped_path = os.path.join(drill_dir, "episodes_shaped.jsonl")
            best_pass_rate = 0.0
            metrics = {}

            drill_start_time = time.time()

            for iteration in range(1, drill.max_iters + 1):
                iter_start = time.time()
                # 1. Generate episodes
                gen_time, _ = generate_episodes(
                    scenario_dirs=drill.scenario_dir,
                    weights_path=checkpoint,
                    output_path=episodes_path,
                    episodes_per_scenario=drill.episodes_per_scenario,
                    threads=args.threads,
                    temperature=drill.temperature,
                    step_interval=args.step_interval,
                    embedding_registry=args.embedding_registry,
                    gpu_shm=gpu_shm,
                    sims_per_thread=args.sims_per_thread,
                )

                # 2. Shape rewards
                stats = transform_drill_rewards(
                    episodes_path, shaped_path, drill,
                    iteration=iteration,
                    current_pass_rate=best_pass_rate,
                )
                print(f"  Iter {iteration}: {stats['total']} eps, "
                      f"pass={stats['pass_rate']:.0%}, gen={gen_time:.1f}s")

                # 3. Train with SAC
                train_start = time.time()
                sac.add_episodes(shaped_path, embedding_registry, args.external_cls_dim)
                n_train_steps = min(2000, max(500, stats["total_steps"] // 20))
                metrics = sac.train_n_steps(n_train_steps)
                buf_sz = sac.buffer.size if sac.buffer else 0
                print(f"    SAC ({n_train_steps} steps, buf={buf_sz}): "
                      f"critic={metrics.get('critic_loss',0):.4f} "
                      f"actor={metrics.get('actor_loss',0):.4f} "
                      f"alpha={metrics.get('alpha',0):.3f} "
                      f"ent={metrics.get('entropy',0):.3f} "
                      f"Q={metrics.get('q_mean',0):.3f}")
                train_time = time.time() - train_start

                # 3b. Log to CSV for dashboard
                csv_path = os.path.join(drill_dir, "training.csv")
                csv_exists = os.path.exists(csv_path)
                total_elapsed = time.time() - drill_start_time
                with open(csv_path, "a") as csv_f:
                    if not csv_exists:
                        csv_f.write("iter,gen_time,train_time,n_episodes,win_rate,n_steps,"
                                    "policy_loss,value_loss,entropy,mean_reward,"
                                    "alpha,q_mean,elapsed_s\n")
                    pl = metrics.get('actor_loss', metrics.get('policy_loss', 0))
                    vl = metrics.get('critic_loss', metrics.get('value_loss', 0))
                    csv_f.write(f"{iteration},{gen_time:.1f},{train_time:.1f},{stats['total']},"
                                f"{stats['win_rate']:.3f},{stats['total_steps']},"
                                f"{pl:.4f},{vl:.4f},"
                                f"{metrics.get('entropy',0):.3f},"
                                f"{stats.get('mean_reward',0):.4f},"
                                f"{metrics.get('alpha',0):.4f},"
                                f"{metrics.get('q_mean',0):.4f},{total_elapsed:.0f}\n")

                # 4. Save + reload
                ckpt_path = os.path.join(drill_dir, "current.pt")
                torch.save({"model_state_dict": model.state_dict(),
                            "iteration": iteration, "drill": drill.name}, ckpt_path)
                sac.sync_actor_to_gpu(ckpt_path)

                if stats["pass_rate"] > best_pass_rate:
                    best_pass_rate = stats["pass_rate"]
                    best_path = os.path.join(drill_dir, "best.pt")
                    torch.save({"model_state_dict": model.state_dict(),
                                "iteration": iteration, "drill": drill.name}, best_path)
                    print(f"    New best: {best_pass_rate:.0%}")

                # 5. Check pass gate (after min_iters)
                if iteration >= drill.min_iters and stats["passes"] >= drill.pass_threshold:
                    # Confirm with greedy eval
                    passes, total = evaluate_drill(drill, ckpt_path, common_args)
                    print(f"    EVAL: {passes}/{total}")
                    if passes >= drill.pass_threshold:
                        print(f"    PASSED {drill.name}: {passes}/{total}")
                        # Log greedy eval pass to CSV so dashboard shows it
                        with open(csv_path, "a") as csv_f:
                            csv_f.write(f"{iteration}.1,0,0,{total},"
                                        f"{passes/max(total,1):.3f},0,"
                                        f"0,0,0,0,0,0,{time.time()-drill_start_time:.0f}\n")
                        passed_drills.append(drill)
                        checkpoint = best_path

                        # Regression test
                        if not args.skip_regression and len(passed_drills) > 1:
                            print("    Running regression tests...")
                            regressed = regression_test(
                                passed_drills[:-1], checkpoint, common_args)
                            if regressed:
                                print(f"    REGRESSION detected: {regressed}")
                                print("    Retraining regressed drills...")
                                # Find the drill configs for regressed drills
                                for reg_name in regressed:
                                    reg_drill = next(
                                        (d for d in passed_drills if d.name == reg_name),
                                        None)
                                    if reg_drill:
                                        checkpoint = retrain_regressed_drill(
                                            reg_drill, model, sac,
                                            checkpoint, common_args, args,
                                            embedding_registry,
                                        )
                                # Re-verify all after retraining
                                still_regressed = regression_test(
                                    passed_drills[:-1], checkpoint, common_args)
                                if still_regressed:
                                    print(f"    WARNING: Still regressed after retraining: {still_regressed}")
                                else:
                                    print(f"    All regressions fixed ({len(regressed)} drills retrained)")
                            else:
                                print(f"    No regressions ({len(passed_drills)-1} drills)")
                        break

            else:
                # Hit max_iters without passing
                print(f"    FAILED {drill.name}: max_iters reached, "
                      f"best pass rate = {best_pass_rate:.0%}")
                print("    Stopping — drill must pass before advancing.")
                break

    finally:
        gpu_proc.terminate()
        gpu_proc.wait()

    # Summary
    print("\n" + "=" * 70)
    print("CURRICULUM SUMMARY")
    print("=" * 70)
    for d in passed_drills:
        print(f"  PASSED: {d.name}")
    print(f"\nPassed: {len(passed_drills)}/{len(curriculum)}")


if __name__ == "__main__":
    main()
