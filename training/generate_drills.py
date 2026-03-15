#!/usr/bin/env python3
"""Generate drill scenario TOML files for the skill curriculum.

Each drill gets 100+ randomized variants with different seeds, positions, and layouts.
Output: dataset/scenarios/drills/phase{N}/{drill_name}/drill_{seed}.toml
"""
import os
import random
import tomli_w
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dataset" / "scenarios" / "drills"


def write_drill(phase: int, drill_name: str, seed: int, cfg: dict):
    """Write a single drill TOML file."""
    d = OUT / f"phase{phase}" / drill_name
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"drill_{seed}.toml"
    with open(path, "wb") as f:
        tomli_w.dump({"scenario": cfg}, f)


def rand_pos(rng, room_w=20, room_h=20, margin=2):
    return [round(rng.uniform(margin, room_w - margin), 1),
            round(rng.uniform(margin, room_h - margin), 1)]


def rand_pos_far(rng, other, min_dist=8, room_w=20, room_h=20, margin=2):
    """Random position at least min_dist from other."""
    for _ in range(100):
        p = rand_pos(rng, room_w, room_h, margin)
        dx, dy = p[0] - other[0], p[1] - other[1]
        if (dx*dx + dy*dy)**0.5 >= min_dist:
            return p
    return rand_pos(rng, room_w, room_h, margin)


CENTER = [10.0, 10.0]


# ─── Phase 1: Movement ───

def gen_1_1_reach_static(n=120):
    """Reach a static target point. Hero spawns at room center."""
    rng = random.Random(42)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        hero_pos = CENTER[:]
        target = rand_pos_far(rng, hero_pos, min_dist=5)
        write_drill(1, "1_1_reach_static", seed, {
            "name": f"drill_1_1_reach_static_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 200,
            "room_type": "Entry",
            "hero_templates": ["scout"],
            "hero_positions": [hero_pos],
            "enemy_hero_templates": [],
            "drill_type": "reach_point",
            "target_position": target,
            "action_mask": "move_only",
            "objective": {
                "objective_type": "reach_position",
                "position": target,
                "radius": 1.0,
            },
        })
    print(f"  1.1 reach_static: {n} drills")


def gen_1_2_reach_moving(n=120):
    """Reach a moving target. Enemy flees on random waypoints at 50% hero speed."""
    rng = random.Random(142)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        hero_pos = CENTER[:]
        enemy_pos = rand_pos_far(rng, hero_pos, min_dist=6)
        write_drill(1, "1_2_reach_moving", seed, {
            "name": f"drill_1_2_reach_moving_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 400,
            "room_type": "Entry",
            "hero_templates": ["scout"],
            "hero_positions": [hero_pos],
            "enemy_hero_templates": [],
            "drill_type": "reach_moving",
            "action_mask": "move_only",
            "enemy_units": [{
                "behavior": "fleeing_target",
                "template": "scout",
                "hp_override": 9999,
                "dps_override": 0.0,
                "move_speed_override": 2.5,
                "position": enemy_pos,
            }],
            "objective": {
                "objective_type": "reach_entity",
                "radius": 1.5,
            },
        })
    print(f"  1.2 reach_moving: {n} drills")


def gen_1_3_navigate_obstacles(n=120):
    """Navigate around obstacles to reach target."""
    rng = random.Random(43)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        hero_pos = CENTER[:]
        target = rand_pos_far(rng, hero_pos, min_dist=12)
        write_drill(1, "1_3_navigate_obstacles", seed, {
            "name": f"drill_1_3_navigate_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 400,
            "room_type": "Pivot",  # 16×16, has obstacles
            "hero_templates": ["scout"],
            "hero_positions": [hero_pos],
            "enemy_hero_templates": [],
            "drill_type": "navigate",
            "target_position": target,
            "action_mask": "move_only",
            "objective": {
                "objective_type": "reach_position",
                "position": target,
                "radius": 1.0,
            },
        })
    print(f"  1.3 navigate_obstacles: {n} drills")


def gen_1_4_navigate_time_pressure(n=120):
    """Navigate obstacles under tighter time pressure (T=300)."""
    rng = random.Random(143)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        hero_pos = CENTER[:]
        target = rand_pos_far(rng, hero_pos, min_dist=12)
        write_drill(1, "1_4_navigate_time_pressure", seed, {
            "name": f"drill_1_4_time_pressure_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 300,
            "room_type": "Entry",
            "hero_templates": ["scout"],
            "hero_positions": [hero_pos],
            "enemy_hero_templates": [],
            "drill_type": "navigate",
            "target_position": target,
            "action_mask": "move_only",
            "objective": {
                "objective_type": "reach_position",
                "position": target,
                "radius": 1.0,
            },
        })
    print(f"  1.4 navigate_time_pressure: {n} drills")


def gen_1_5_navigate_moving_obstacles(n=120):
    """Navigate in larger room with more obstacles (Setpiece) under tight timeout."""
    rng = random.Random(144)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        hero_pos = CENTER[:]
        target = rand_pos_far(rng, hero_pos, min_dist=10)
        write_drill(1, "1_5_navigate_moving_obstacles", seed, {
            "name": f"drill_1_5_moving_obs_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 500,
            "room_type": "Setpiece",
            "hero_templates": ["scout"],
            "hero_positions": [hero_pos],
            "enemy_hero_templates": [],
            "drill_type": "navigate",
            "target_position": target,
            "action_mask": "move_only",
            "objective": {
                "objective_type": "reach_position",
                "position": target,
                "radius": 1.0,
            },
        })
    print(f"  1.5 navigate_moving_obstacles: {n} drills")


def gen_1_6_react_dynamic_terrain(n=120):
    """Navigate varied room types to reach target."""
    rng = random.Random(145)
    room_types = ["Entry", "Pivot", "Setpiece", "Pressure"]
    for i in range(n):
        seed = rng.randint(10000, 99999)
        hero_pos = CENTER[:]
        room = rng.choice(room_types)
        target = rand_pos_far(rng, hero_pos, min_dist=8)
        write_drill(1, "1_6_react_dynamic_terrain", seed, {
            "name": f"drill_1_6_dynamic_terrain_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 500,
            "room_type": room,
            "hero_templates": ["scout"],
            "hero_positions": [hero_pos],
            "enemy_hero_templates": [],
            "drill_type": "navigate",
            "target_position": target,
            "action_mask": "move_only",
            "objective": {
                "objective_type": "reach_position",
                "position": target,
                "radius": 1.0,
            },
        })
    print(f"  1.6 react_dynamic_terrain: {n} drills")


# ─── Phase 2: Spatial Awareness ───

def gen_2_1_maintain_distance(n=100):
    """Maintain distance from chasing melee enemy."""
    rng = random.Random(44)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(2, "2_1_maintain_distance", seed, {
            "name": f"drill_2_1_kite_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 500,
            "room_type": "Setpiece",
            "hero_templates": ["scout"],
            "enemy_hero_templates": [],
            "drill_type": "survive",
            "action_mask": "move_only",
            "enemy_units": [{
                "behavior": "melee_chaser",
                "template": "brute",
                "hp_override": 9999,
                "dps_override": 30.0,
                "range_override": 1.5,
                "move_speed_override": 2.5,  # ~50% of scout speed
            }],
            "objective": {
                "objective_type": "survive",
                "duration": 500,
                "max_damage_taken": 0.0,
            },
        })
    print(f"  2.1 maintain_distance: {n} drills")


def gen_2_2_dodge_zones(n=100):
    """Navigate through danger zones to reach goal."""
    rng = random.Random(45)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        # Place 4-6 zones between spawn and goal
        n_zones = rng.randint(4, 6)
        zones = []
        for _ in range(n_zones):
            zones.append({
                "hazard_type": "damage_zone",
                "position": [round(rng.uniform(5, 15), 1), round(rng.uniform(5, 15), 1)],
                "radius": round(rng.uniform(2.0, 3.0), 1),
                "damage_per_tick": 5.0,
                "team": "neutral",
            })
        write_drill(2, "2_2_dodge_zones", seed, {
            "name": f"drill_2_2_dodge_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 300,
            "room_type": "Entry",
            "hero_templates": ["scout"],
            "enemy_hero_templates": [],
            "drill_type": "dodge_zones",
            "target_position": [18.0, 18.0],
            "action_mask": "move_only",
            "hazards": zones,
            "objective": {
                "objective_type": "reach_position",
                "position": [18.0, 18.0],
                "radius": 1.0,
            },
        })
    print(f"  2.2 dodge_zones: {n} drills")


def gen_2_3_dodge_telegraphed(n=100):
    """Dodge AoE abilities from an enemy caster."""
    rng = random.Random(245)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(2, "2_3_dodge_telegraphed", seed, {
            "name": f"drill_2_3_dodge_aoe_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 500,
            "room_type": "Entry",
            "hero_templates": ["scout"],
            "enemy_hero_templates": [],
            "drill_type": "dodge_abilities",
            "action_mask": "move_only",
            "enemy_units": [{
                "behavior": "aoe_caster",
                "template": "mage",
                "hp_override": 9999,
                "dps_override": 0.0,
            }],
            "objective": {
                "objective_type": "survive",
                "duration": 500,
                "max_damage_taken": 0.0,
            },
        })
    print(f"  2.3 dodge_telegraphed: {n} drills")


def gen_2_4_kite_melee(n=100):
    """Kite a melee enemy using ranged auto-attacks."""
    rng = random.Random(46)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(2, "2_4_kite_melee", seed, {
            "name": f"drill_2_4_kite_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 600,
            "room_type": "Entry",
            "hero_templates": ["archer"],  # ranged, range 4
            "enemy_hero_templates": [],
            "drill_type": "kite",
            "action_mask": "move_only",  # auto-attacks fire automatically
            "enemy_units": [{
                "behavior": "melee_chaser",
                "template": "brute",
                "hp_override": 500,  # ~10 auto-attacks to kill
                "dps_override": 30.0,
                "range_override": 1.5,
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  2.4 kite_melee: {n} drills")


# ─── Phase 3: Target Selection ───

def gen_3_1_kill_stationary(n=100):
    """Kill a stationary, non-attacking dummy."""
    rng = random.Random(47)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_1_kill_stationary", seed, {
            "name": f"drill_3_1_kill_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 200,
            "room_type": "Entry",
            "hero_templates": ["archer"],
            "enemy_hero_templates": [],
            "drill_type": "kill_target",
            "action_mask": "move_attack",
            "enemy_units": [{
                "behavior": "stationary_dummy",
                "template": "brute",
                "hp_override": 500,
                "dps_override": 0.0,
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  3.1 kill_stationary: {n} drills")


def gen_3_2_kill_moving(n=100):
    """Kill a fleeing target."""
    rng = random.Random(347)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_2_kill_moving", seed, {
            "name": f"drill_3_2_kill_moving_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 400,
            "room_type": "Entry",
            "hero_templates": ["archer"],
            "enemy_hero_templates": [],
            "drill_type": "kill_target",
            "action_mask": "move_attack",
            "enemy_units": [{
                "behavior": "fleeing_target",
                "template": "scout",
                "hp_override": 400,
                "dps_override": 0.0,
                "move_speed_override": 2.5,
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  3.2 kill_moving: {n} drills")


def gen_3_3_prioritize_low_hp(n=100):
    """Kill the low-HP enemy first."""
    rng = random.Random(48)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_3_prioritize_low_hp", seed, {
            "name": f"drill_3_3_prio_hp_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 300,
            "room_type": "Entry",
            "hero_templates": ["archer"],
            "enemy_hero_templates": [],
            "drill_type": "prioritize_target",
            "action_mask": "move_attack",
            "enemy_units": [
                {
                    "behavior": "stationary_dummy",
                    "template": "brute",
                    "hp_override": 1000,
                    "dps_override": 0.0,
                    "tag": "full_hp",
                },
                {
                    "behavior": "stationary_dummy",
                    "template": "brute",
                    "hp_override": 100,
                    "dps_override": 0.0,
                    "tag": "low_hp",
                },
            ],
            "objective": {
                "objective_type": "kill_target",
                "target_tag": "low_hp",
            },
        })
    print(f"  3.3 prioritize_low_hp: {n} drills")


def gen_3_4_prioritize_high_threat(n=100):
    """Prioritize the high-threat (high DPS) enemy. Hero + ally vs 2 enemies."""
    rng = random.Random(348)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_4_prioritize_high_threat", seed, {
            "name": f"drill_3_4_prio_threat_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 400,
            "room_type": "Entry",
            "hero_templates": ["archer", "duelist"],
            "enemy_hero_templates": [],
            "drill_type": "prioritize_target",
            "action_mask": "move_attack",
            "enemy_units": [
                {
                    "behavior": "stationary_dummy",
                    "template": "brute",
                    "hp_override": 600,
                    "dps_override": 5.0,
                    "tag": "low_threat",
                },
                {
                    "behavior": "melee_chaser",
                    "template": "berserker",
                    "hp_override": 600,
                    "dps_override": 40.0,
                    "tag": "high_threat",
                },
            ],
            "objective": {
                "objective_type": "kill_target",
                "target_tag": "high_threat",
            },
        })
    print(f"  3.4 prioritize_high_threat: {n} drills")


def gen_3_5_kill_healer(n=100):
    """Kill the healer first (healer keeps healing the DPS)."""
    rng = random.Random(49)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_5_kill_healer", seed, {
            "name": f"drill_3_5_healer_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 600,
            "room_type": "Entry",
            "hero_templates": ["duelist"],  # melee DPS
            "enemy_hero_templates": [],
            "drill_type": "kill_healer",
            "action_mask": "move_attack",
            "enemy_units": [
                {
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 800,
                    "dps_override": 20.0,
                    "tag": "dps",
                },
                {
                    "behavior": "healer_bot",
                    "template": "cleric",
                    "hp_override": 500,
                    "dps_override": 5.0,
                    "tag": "healer",
                },
            ],
            "objective": {
                "objective_type": "kill_target",
                "target_tag": "healer",
            },
        })
    print(f"  3.5 kill_healer: {n} drills")


def gen_3_6_protect_ally(n=100):
    """Protect ally by killing the enemy attacking them first."""
    rng = random.Random(349)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_6_protect_ally", seed, {
            "name": f"drill_3_6_protect_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 400,
            "room_type": "Entry",
            "hero_templates": ["duelist", "cleric"],
            "enemy_hero_templates": [],
            "drill_type": "protect_ally",
            "action_mask": "move_attack",
            "enemy_units": [
                {
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 600,
                    "dps_override": 30.0,
                    "tag": "ally_attacker",
                    "target_preference": "ally",
                },
                {
                    "behavior": "stationary_dummy",
                    "template": "brute",
                    "hp_override": 400,
                    "dps_override": 0.0,
                    "tag": "idle",
                },
            ],
            "objective": {
                "objective_type": "survive",
            },
        })
    print(f"  3.6 protect_ally: {n} drills")


def gen_3_7_react_threat_change(n=100):
    """Enemy starts passive then becomes aggressive at tick 100."""
    rng = random.Random(350)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_7_react_threat_change", seed, {
            "name": f"drill_3_7_threat_change_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 300,
            "room_type": "Entry",
            "hero_templates": ["duelist"],
            "enemy_hero_templates": [],
            "drill_type": "react_threat",
            "action_mask": "move_attack",
            "enemy_units": [
                {
                    "behavior": "stationary_dummy",
                    "template": "brute",
                    "hp_override": 400,
                    "dps_override": 10.0,
                    "tag": "passive",
                },
                {
                    "behavior": "melee_chaser",
                    "template": "berserker",
                    "hp_override": 500,
                    "dps_override": 35.0,
                    "tag": "activates_tick_100",
                    "activation_tick": 100,
                },
            ],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  3.7 react_threat_change: {n} drills")


def gen_3_8_multi_threat(n=100):
    """Multi-threat assessment: healer + DPS + tank. Kill order matters."""
    rng = random.Random(351)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_8_multi_threat", seed, {
            "name": f"drill_3_8_multi_threat_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 800,
            "room_type": "Entry",
            "hero_templates": ["archer", "duelist"],
            "enemy_hero_templates": [],
            "drill_type": "multi_threat",
            "action_mask": "move_attack",
            "enemy_units": [
                {
                    "behavior": "healer_bot",
                    "template": "cleric",
                    "hp_override": 400,
                    "dps_override": 5.0,
                    "tag": "healer",
                },
                {
                    "behavior": "melee_chaser",
                    "template": "berserker",
                    "hp_override": 600,
                    "dps_override": 30.0,
                    "tag": "dps",
                },
                {
                    "behavior": "melee_chaser",
                    "template": "knight",
                    "hp_override": 1200,
                    "dps_override": 10.0,
                    "tag": "tank",
                },
            ],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  3.8 multi_threat: {n} drills")


def gen_3_9_horde_combat(n=100):
    """Kill 5+ weak enemies."""
    rng = random.Random(352)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        n_enemies = rng.randint(5, 7)
        enemies = []
        for j in range(n_enemies):
            enemies.append({
                "behavior": "melee_chaser",
                "template": "scout",
                "hp_override": 150,
                "dps_override": 8.0,
                "tag": f"minion_{j}",
            })
        write_drill(3, "3_9_horde_combat", seed, {
            "name": f"drill_3_9_horde_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 600,
            "room_type": "Entry",
            "hero_templates": ["berserker"],
            "enemy_hero_templates": [],
            "drill_type": "horde",
            "action_mask": "move_attack",
            "enemy_units": enemies,
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  3.9 horde_combat: {n} drills")


def gen_3_10_use_terrain(n=100):
    """Use terrain chokepoints (Pivot room) against melee enemies."""
    rng = random.Random(353)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_10_use_terrain", seed, {
            "name": f"drill_3_10_terrain_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 500,
            "room_type": "Pivot",
            "hero_templates": ["archer"],
            "enemy_hero_templates": [],
            "drill_type": "use_terrain",
            "action_mask": "move_attack",
            "enemy_units": [
                {
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 500,
                    "dps_override": 20.0,
                },
                {
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 500,
                    "dps_override": 20.0,
                },
                {
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 500,
                    "dps_override": 20.0,
                },
            ],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  3.10 use_terrain: {n} drills")


def gen_3_11_elevation_advantage(n=100):
    """Use elevation advantage (Setpiece room) against enemies below."""
    rng = random.Random(354)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(3, "3_11_elevation_advantage", seed, {
            "name": f"drill_3_11_elevation_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 400,
            "room_type": "Setpiece",
            "hero_templates": ["archer"],
            "enemy_hero_templates": [],
            "drill_type": "elevation",
            "action_mask": "move_attack",
            "enemy_units": [
                {
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 500,
                    "dps_override": 20.0,
                },
                {
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 500,
                    "dps_override": 20.0,
                },
            ],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  3.11 elevation_advantage: {n} drills")


# ─── Phase 4: Ability Usage ───

def gen_4_1_heal_low_ally(n=100):
    """Use heal on low-HP ally. Hero=cleric, ally at 30% HP."""
    rng = random.Random(401)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(4, "4_1_heal_low_ally", seed, {
            "name": f"drill_4_1_heal_ally_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 100,
            "room_type": "Entry",
            "hero_templates": ["cleric", "warrior"],
            "enemy_hero_templates": [],
            "drill_type": "heal_ally",
            "action_mask": "all",
            "ally_hp_percent": 0.3,
            "objective": {
                "objective_type": "heal_ally",
                "target_hp_percent": 0.9,
            },
        })
    print(f"  4.1 heal_low_ally: {n} drills")


def gen_4_2_cc_enemy(n=100):
    """Use CC on enemy chasing ally."""
    rng = random.Random(402)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(4, "4_2_cc_enemy", seed, {
            "name": f"drill_4_2_cc_enemy_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 200,
            "room_type": "Entry",
            "hero_templates": ["stunner", "archer"],
            "enemy_hero_templates": [],
            "drill_type": "cc_target",
            "action_mask": "all",
            "enemy_units": [{
                "behavior": "melee_chaser",
                "template": "brute",
                "hp_override": 500,
                "dps_override": 25.0,
                "target_preference": "ally",
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  4.2 cc_enemy: {n} drills")


def gen_4_3_interrupt_cast(n=100):
    """Interrupt enemy cast with stun."""
    rng = random.Random(403)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(4, "4_3_interrupt_cast", seed, {
            "name": f"drill_4_3_interrupt_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 120,
            "room_type": "Entry",
            "hero_templates": ["stunner"],
            "enemy_hero_templates": [],
            "drill_type": "interrupt",
            "action_mask": "all",
            "enemy_units": [{
                "behavior": "aoe_caster",
                "template": "mage",
                "hp_override": 400,
                "dps_override": 50.0,
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  4.3 interrupt_cast: {n} drills")


def gen_4_3b_selective_interrupt(n=100):
    """Two enemies cast in sequence; only interrupt the strong one."""
    rng = random.Random(4030)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(4, "4_3b_selective_interrupt", seed, {
            "name": f"drill_4_3b_selective_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 300,
            "room_type": "Entry",
            "hero_templates": ["stunner"],
            "enemy_hero_templates": [],
            "drill_type": "selective_interrupt",
            "action_mask": "all",
            "enemy_units": [
                {
                    "behavior": "aoe_caster",
                    "template": "elementalist",
                    "hp_override": 400,
                    "dps_override": 10.0,
                    "tag": "weak_caster",
                },
                {
                    "behavior": "aoe_caster",
                    "template": "mage",
                    "hp_override": 500,
                    "dps_override": 60.0,
                    "tag": "strong_caster",
                },
            ],
            "objective": {
                "objective_type": "survive",
            },
        })
    print(f"  4.3b selective_interrupt: {n} drills")


def gen_4_4_cc_then_burst(n=100):
    """Stun enemy, then burst during CC window."""
    rng = random.Random(50)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(4, "4_4_cc_burst", seed, {
            "name": f"drill_4_4_cc_burst_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 400,
            "room_type": "Entry",
            "hero_templates": ["stunner"],  # has stun + damage ability
            "enemy_hero_templates": [],
            "drill_type": "cc_burst",
            "action_mask": "all",
            "enemy_units": [{
                "behavior": "melee_chaser",
                "template": "brute",
                "hp_override": 800,
                "dps_override": 10.0,
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  4.4 cc_then_burst: {n} drills")


def gen_4_5_knockback_wall(n=100):
    """Knockback enemy into wall."""
    rng = random.Random(405)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(4, "4_5_knockback_wall", seed, {
            "name": f"drill_4_5_knockback_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 300,
            "room_type": "Entry",
            "hero_templates": ["monk"],  # has knockback
            "enemy_hero_templates": [],
            "drill_type": "knockback",
            "action_mask": "all",
            "enemy_units": [{
                "behavior": "melee_chaser",
                "template": "brute",
                "hp_override": 600,
                "dps_override": 20.0,
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  4.5 knockback_wall: {n} drills")


def gen_4_6_aoe_positioning(n=100):
    """Use AoE on 3 clustered enemies."""
    rng = random.Random(406)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(4, "4_6_aoe_positioning", seed, {
            "name": f"drill_4_6_aoe_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 300,
            "room_type": "Entry",
            "hero_templates": ["elementalist"],  # has AoE
            "enemy_hero_templates": [],
            "drill_type": "aoe_position",
            "action_mask": "all",
            "enemy_units": [
                {
                    "behavior": "stationary_dummy",
                    "template": "brute",
                    "hp_override": 300,
                    "dps_override": 5.0,
                    "tag": "cluster_0",
                },
                {
                    "behavior": "stationary_dummy",
                    "template": "brute",
                    "hp_override": 300,
                    "dps_override": 5.0,
                    "tag": "cluster_1",
                },
                {
                    "behavior": "stationary_dummy",
                    "template": "brute",
                    "hp_override": 300,
                    "dps_override": 5.0,
                    "tag": "cluster_2",
                },
            ],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  4.6 aoe_positioning: {n} drills")


def gen_4_7_cooldown_management(n=100):
    """Manage cooldowns across waves of enemies."""
    rng = random.Random(407)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        # 2-3 waves of enemies, spawning at intervals
        n_waves = rng.randint(2, 3)
        enemies = []
        for wave in range(n_waves):
            for j in range(2):
                enemies.append({
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 400,
                    "dps_override": 15.0,
                    "tag": f"wave_{wave}_{j}",
                    "activation_tick": wave * 400,
                })
        write_drill(4, "4_7_cooldown_management", seed, {
            "name": f"drill_4_7_cooldowns_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 1500,
            "room_type": "Entry",
            "hero_templates": ["mage"],
            "enemy_hero_templates": [],
            "drill_type": "cooldown_management",
            "action_mask": "all",
            "enemy_units": enemies,
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  4.7 cooldown_management: {n} drills")


def gen_4_8_respect_enemy_cooldowns(n=100):
    """Play around enemy CC ability cooldowns."""
    rng = random.Random(408)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(4, "4_8_respect_enemy_cooldowns", seed, {
            "name": f"drill_4_8_enemy_cd_{seed}",
            "seed": seed,
            "hero_count": 1,
            "enemy_count": 0,
            "max_ticks": 800,
            "room_type": "Entry",
            "hero_templates": ["duelist"],
            "enemy_hero_templates": [],
            "drill_type": "respect_cooldowns",
            "action_mask": "all",
            "enemy_units": [{
                "behavior": "cc_gatekeeper",
                "template": "stunner",
                "hp_override": 700,
                "dps_override": 15.0,
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  4.8 respect_enemy_cooldowns: {n} drills")


# ─── Phase 5: Team Coordination ───

def gen_5_1_focus_fire(n=100):
    """Both allies must attack the same target."""
    rng = random.Random(51)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(5, "5_1_focus_fire", seed, {
            "name": f"drill_5_1_focus_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 400,
            "room_type": "Entry",
            "hero_templates": ["archer", "duelist"],
            "enemy_hero_templates": [],
            "drill_type": "focus_fire",
            "action_mask": "all",
            "enemy_units": [
                {
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 600,
                    "dps_override": 15.0,
                },
                {
                    "behavior": "melee_chaser",
                    "template": "brute",
                    "hp_override": 600,
                    "dps_override": 15.0,
                },
            ],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  5.1 focus_fire: {n} drills")


def gen_5_2_chain_cc(n=100):
    """Chain CC: 2 heroes with stuns vs 1 strong enemy."""
    rng = random.Random(502)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(5, "5_2_chain_cc", seed, {
            "name": f"drill_5_2_chain_cc_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 800,
            "room_type": "Entry",
            "hero_templates": ["stunner", "monk"],
            "enemy_hero_templates": [],
            "drill_type": "chain_cc",
            "action_mask": "all",
            "enemy_units": [{
                "behavior": "melee_chaser",
                "template": "berserker",
                "hp_override": 1500,
                "dps_override": 40.0,
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  5.2 chain_cc: {n} drills")


def gen_5_3_peel_for_carry(n=100):
    """Tank peels for DPS carry. Enemy attacks DPS."""
    rng = random.Random(503)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(5, "5_3_peel_for_carry", seed, {
            "name": f"drill_5_3_peel_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 500,
            "room_type": "Entry",
            "hero_templates": ["knight", "archer"],
            "enemy_hero_templates": [],
            "drill_type": "peel",
            "action_mask": "all",
            "enemy_units": [{
                "behavior": "melee_chaser",
                "template": "berserker",
                "hp_override": 800,
                "dps_override": 35.0,
                "target_preference": "ally",  # targets the DPS
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  5.3 peel_for_carry: {n} drills")


def gen_5_4_dive_coordination(n=100):
    """2 heroes dive healer behind DPS."""
    rng = random.Random(504)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(5, "5_4_dive_coordination", seed, {
            "name": f"drill_5_4_dive_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 600,
            "room_type": "Entry",
            "hero_templates": ["duelist", "assassin"],
            "enemy_hero_templates": [],
            "drill_type": "dive",
            "action_mask": "all",
            "enemy_units": [
                {
                    "behavior": "healer_bot",
                    "template": "cleric",
                    "hp_override": 500,
                    "dps_override": 5.0,
                    "tag": "healer",
                },
                {
                    "behavior": "melee_chaser",
                    "template": "berserker",
                    "hp_override": 700,
                    "dps_override": 25.0,
                    "tag": "dps",
                },
            ],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  5.4 dive_coordination: {n} drills")


def gen_5_5_engage_disengage(n=100):
    """Engage/disengage vs enemy with CC. 2 heroes vs 1 CC enemy."""
    rng = random.Random(505)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(5, "5_5_engage_disengage", seed, {
            "name": f"drill_5_5_engage_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 1000,
            "room_type": "Entry",
            "hero_templates": ["duelist", "archer"],
            "enemy_hero_templates": [],
            "drill_type": "engage_disengage",
            "action_mask": "all",
            "enemy_units": [{
                "behavior": "cc_gatekeeper",
                "template": "stunner",
                "hp_override": 1000,
                "dps_override": 25.0,
            }],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  5.5 engage_disengage: {n} drills")


def gen_5_6_horde_defense(n=100):
    """Tank + DPS defend against 6 weak enemies in Pivot room."""
    rng = random.Random(506)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        enemies = []
        for j in range(6):
            enemies.append({
                "behavior": "melee_chaser",
                "template": "scout",
                "hp_override": 200,
                "dps_override": 10.0,
                "tag": f"minion_{j}",
            })
        write_drill(5, "5_6_horde_defense", seed, {
            "name": f"drill_5_6_horde_def_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 800,
            "room_type": "Pivot",
            "hero_templates": ["knight", "elementalist"],
            "enemy_hero_templates": [],
            "drill_type": "horde_defense",
            "action_mask": "all",
            "enemy_units": enemies,
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  5.6 horde_defense: {n} drills")


def gen_5_7_dynamic_terrain_coord(n=100):
    """Hero with wall ability + DPS vs healer + DPS."""
    rng = random.Random(507)
    for i in range(n):
        seed = rng.randint(10000, 99999)
        write_drill(5, "5_7_dynamic_terrain_coord", seed, {
            "name": f"drill_5_7_terrain_coord_{seed}",
            "seed": seed,
            "hero_count": 2,
            "enemy_count": 0,
            "max_ticks": 600,
            "room_type": "Entry",
            "hero_templates": ["elementalist", "duelist"],
            "enemy_hero_templates": [],
            "drill_type": "terrain_coord",
            "action_mask": "all",
            "enemy_units": [
                {
                    "behavior": "healer_bot",
                    "template": "cleric",
                    "hp_override": 500,
                    "dps_override": 5.0,
                    "tag": "healer",
                },
                {
                    "behavior": "melee_chaser",
                    "template": "berserker",
                    "hp_override": 600,
                    "dps_override": 30.0,
                    "tag": "dps",
                },
            ],
            "objective": {
                "objective_type": "kill_all",
            },
        })
    print(f"  5.7 dynamic_terrain_coord: {n} drills")


def main():
    print("Generating drill scenarios...")
    print()

    print("Phase 1: Movement")
    gen_1_1_reach_static()
    gen_1_2_reach_moving()
    gen_1_3_navigate_obstacles()
    gen_1_4_navigate_time_pressure()
    gen_1_5_navigate_moving_obstacles()
    gen_1_6_react_dynamic_terrain()
    print()

    print("Phase 2: Spatial Awareness")
    gen_2_1_maintain_distance()
    gen_2_2_dodge_zones()
    gen_2_3_dodge_telegraphed()
    gen_2_4_kite_melee()
    print()

    print("Phase 3: Target Selection")
    gen_3_1_kill_stationary()
    gen_3_2_kill_moving()
    gen_3_3_prioritize_low_hp()
    gen_3_4_prioritize_high_threat()
    gen_3_5_kill_healer()
    gen_3_6_protect_ally()
    gen_3_7_react_threat_change()
    gen_3_8_multi_threat()
    gen_3_9_horde_combat()
    gen_3_10_use_terrain()
    gen_3_11_elevation_advantage()
    print()

    print("Phase 4: Ability Usage")
    gen_4_1_heal_low_ally()
    gen_4_2_cc_enemy()
    gen_4_3_interrupt_cast()
    gen_4_3b_selective_interrupt()
    gen_4_4_cc_then_burst()
    gen_4_5_knockback_wall()
    gen_4_6_aoe_positioning()
    gen_4_7_cooldown_management()
    gen_4_8_respect_enemy_cooldowns()
    print()

    print("Phase 5: Team Coordination")
    gen_5_1_focus_fire()
    gen_5_2_chain_cc()
    gen_5_3_peel_for_carry()
    gen_5_4_dive_coordination()
    gen_5_5_engage_disengage()
    gen_5_6_horde_defense()
    gen_5_7_dynamic_terrain_coord()
    print()

    # Count totals
    total = 0
    for phase_dir in sorted(OUT.iterdir()):
        if phase_dir.is_dir():
            n = sum(1 for _ in phase_dir.rglob("*.toml"))
            total += n
            print(f"  {phase_dir.name}: {n} drills")
    print(f"\nTotal: {total} drill scenarios in {OUT}")


if __name__ == "__main__":
    main()
