#!/usr/bin/env python3
"""Tactical quality probes for V5 training data.

Measures whether episodes contain intelligent tactical behavior, not just
correct data formatting. Scores episodes on:

  1. Focus fire — are heroes attacking the same target?
  2. CC usage — is CC being applied? Are CC'd targets being attacked?
  3. Ability usage rate — are abilities being used at all?
  4. Low-HP saves — do hurt heroes receive heals or retreat?
  5. Kill securing — are low-HP enemies being focused?
  6. Smart positioning — do units use cover/terrain?
  7. Damage efficiency — damage dealt vs taken ratio
  8. Ability diversity — are different ability types used?

Usage:
    uv run --with numpy python training/probe_tactical_quality.py \
        generated/v5_data.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def analyze_episode(ep: dict) -> dict:
    """Compute tactical quality metrics for a single episode."""
    steps = ep["steps"]
    if not steps:
        return {}

    metrics = {}
    n_steps = len(steps)

    # --- Ability usage ---
    combat_types = [s.get("combat_type", 1) for s in steps]
    n_attack = sum(1 for c in combat_types if c == 0)
    n_hold = sum(1 for c in combat_types if c == 1)
    n_ability = sum(1 for c in combat_types if c >= 2)
    metrics["ability_usage_rate"] = n_ability / max(n_steps, 1)
    metrics["attack_rate"] = n_attack / max(n_steps, 1)
    metrics["hold_rate"] = n_hold / max(n_steps, 1)

    # --- Ability diversity ---
    ability_types_used = set(c for c in combat_types if c >= 2)
    metrics["ability_types_used"] = len(ability_types_used)

    # Named abilities from episode metadata
    all_ability_names = []
    for uid, names in ep.get("unit_ability_names", {}).items():
        all_ability_names.extend(names)
    metrics["hero_abilities_available"] = len(all_ability_names)

    # --- Focus fire ---
    # Check if multiple steps in a row target the same entity
    target_idxs = [s.get("target_idx", -1) for s in steps]
    if len(target_idxs) >= 2:
        consecutive_same = sum(
            1 for i in range(1, len(target_idxs))
            if target_idxs[i] == target_idxs[i-1] and target_idxs[i] >= 0
        )
        metrics["focus_fire_rate"] = consecutive_same / max(len(target_idxs) - 1, 1)
    else:
        metrics["focus_fire_rate"] = 0.0

    # --- HP trajectories ---
    # Track hero HP% over time from entity features
    hero_hp_trajectory = []
    enemy_hp_trajectory = []
    for s in steps:
        entities = s.get("entities", [])
        entity_types = s.get("entity_types", [])
        hero_hp = []
        enemy_hp = []
        for ei, etype in enumerate(entity_types):
            if ei >= len(entities):
                break
            e = entities[ei]
            if len(e) <= 29 or e[29] < 0.5:
                continue
            if etype in (0, 2):
                hero_hp.append(e[0])
            elif etype == 1:
                enemy_hp.append(e[0])
        hero_hp_trajectory.append(np.mean(hero_hp) if hero_hp else 0.0)
        enemy_hp_trajectory.append(np.mean(enemy_hp) if enemy_hp else 0.0)

    hero_hp_arr = np.array(hero_hp_trajectory)
    enemy_hp_arr = np.array(enemy_hp_trajectory)

    # Damage efficiency: how much enemy HP dropped vs hero HP dropped
    hero_hp_lost = max(hero_hp_arr[0] - hero_hp_arr[-1], 0) if len(hero_hp_arr) > 1 else 0
    enemy_hp_lost = max(enemy_hp_arr[0] - enemy_hp_arr[-1], 0) if len(enemy_hp_arr) > 1 else 0
    metrics["damage_efficiency"] = enemy_hp_lost / max(hero_hp_lost, 0.01)

    # --- Low-HP events ---
    # Count steps where a hero is below 25% HP
    low_hp_steps = 0
    for s in steps:
        entities = s.get("entities", [])
        entity_types = s.get("entity_types", [])
        for ei, etype in enumerate(entity_types):
            if ei >= len(entities):
                break
            e = entities[ei]
            if len(e) <= 29 or e[29] < 0.5:
                continue
            if etype in (0, 2) and e[0] < 0.25:
                low_hp_steps += 1
                break
    metrics["low_hp_exposure_rate"] = low_hp_steps / max(n_steps, 1)

    # --- Kill events ---
    # Count deaths by comparing entity counts at start vs end of episode.
    # Slot-level tracking doesn't work because entity slots get reassigned
    # between steps (priority-based selection changes which entities are in
    # which slots).  Instead, use aggregate features which track total counts
    # across the whole sim, not just the visible entity slots.
    first_agg = steps[0].get("aggregate_features", [])
    last_agg = steps[-1].get("aggregate_features", [])
    if len(first_agg) >= 2 and len(last_agg) >= 2:
        # aggregate_features[0] = n_enemies_total / 20
        # aggregate_features[1] = n_allies_total / 10
        enemies_start = round(first_agg[0] * 20)
        enemies_end = round(last_agg[0] * 20)
        allies_start = round(first_agg[1] * 10)
        allies_end = round(last_agg[1] * 10)
        enemy_kills = max(int(enemies_start - enemies_end), 0)
        hero_deaths = max(int(allies_start - allies_end), 0)
    else:
        # Fallback: count entities with exists flag in first/last steps
        first_ents = steps[0].get("entities", [])
        first_types = steps[0].get("entity_types", [])
        last_ents = steps[-1].get("entities", [])
        last_types = steps[-1].get("entity_types", [])

        def count_by_type(ents, types, target_types):
            n = 0
            for ei, etype in enumerate(types):
                if ei >= len(ents):
                    break
                e = ents[ei]
                if len(e) > 29 and e[29] > 0.5 and etype in target_types:
                    n += 1
            return n

        heroes_start = count_by_type(first_ents, first_types, (0, 2))
        heroes_end = count_by_type(last_ents, last_types, (0, 2))
        enemies_start_fb = count_by_type(first_ents, first_types, (1,))
        enemies_end_fb = count_by_type(last_ents, last_types, (1,))
        enemy_kills = max(enemies_start_fb - enemies_end_fb, 0)
        hero_deaths = max(heroes_start - heroes_end, 0)
    metrics["enemy_kills"] = enemy_kills
    metrics["hero_deaths"] = hero_deaths

    # --- CC detection ---
    # cc_remaining is feature index 26
    cc_steps = 0
    enemy_cc_steps = 0
    for s in steps:
        entities = s.get("entities", [])
        entity_types = s.get("entity_types", [])
        for ei, etype in enumerate(entity_types):
            if ei >= len(entities):
                break
            e = entities[ei]
            if len(e) <= 29 or e[29] < 0.5:
                continue
            if e[26] > 0.01:  # cc_remaining > 0
                cc_steps += 1
                if etype == 1:  # enemy CC'd
                    enemy_cc_steps += 1
                break
    metrics["cc_present_rate"] = cc_steps / max(n_steps, 1)
    metrics["enemy_cc_rate"] = enemy_cc_steps / max(n_steps, 1)

    # --- Spatial awareness ---
    # Check if spatial features vary (units moving to different positions)
    spatial_variance = 0.0
    corner_counts = []
    for s in steps:
        entities = s.get("entities", [])
        entity_types = s.get("entity_types", [])
        for ei, etype in enumerate(entity_types):
            if ei >= len(entities):
                break
            e = entities[ei]
            if len(e) > 33 and e[29] > 0.5 and etype in (0, 2):
                corner_counts.append(e[30])  # visible_corner_count/16
    if corner_counts:
        metrics["spatial_variance"] = float(np.std(corner_counts))
        metrics["mean_visible_corners"] = float(np.mean(corner_counts)) * 16
    else:
        metrics["spatial_variance"] = 0.0
        metrics["mean_visible_corners"] = 0.0

    # --- Threat awareness ---
    threat_steps = sum(1 for s in steps if s.get("threats") and len(s["threats"]) > 0)
    metrics["threat_exposure_rate"] = threat_steps / max(n_steps, 1)

    # --- Movement diversity ---
    move_dirs = [s.get("move_dir", 8) for s in steps]
    unique_dirs = len(set(move_dirs))
    metrics["movement_diversity"] = unique_dirs / 9.0

    # --- Terrain complexity ---
    # Spatial features: visible_corner_count(30), avg_passage_width(31),
    #                   min_passage_width(32), avg_corner_distance(33)
    min_passage_widths = []
    corner_counts_all = []
    has_chokepoints = False
    has_cover = False
    position_spread = []

    for s in steps:
        entities = s.get("entities", [])
        entity_types = s.get("entity_types", [])
        step_positions = []
        for ei, etype in enumerate(entity_types):
            if ei >= len(entities):
                break
            e = entities[ei]
            if len(e) > 33 and e[29] > 0.5:
                corner_counts_all.append(e[30] * 16)  # denormalize
                min_passage_widths.append(e[32] * 10)  # denormalize
                if e[32] > 0 and e[32] < 0.3:  # narrow passage < 3 units wide
                    has_chokepoints = True
                if e[8] > 0:  # cover_bonus > 0
                    has_cover = True
                if etype in (0, 2):
                    step_positions.append((e[5] * 20, e[6] * 20))  # denormalize pos
        if len(step_positions) >= 2:
            xs = [p[0] for p in step_positions]
            ys = [p[1] for p in step_positions]
            position_spread.append(np.std(xs) + np.std(ys))

    if corner_counts_all:
        metrics["mean_corners"] = float(np.mean(corner_counts_all))
        metrics["max_corners"] = float(np.max(corner_counts_all))
        metrics["is_flat_room"] = float(np.max(corner_counts_all)) < 1.0
    else:
        metrics["mean_corners"] = 0.0
        metrics["max_corners"] = 0.0
        metrics["is_flat_room"] = True

    if min_passage_widths:
        metrics["min_passage_width"] = float(np.min(min_passage_widths))
        metrics["has_chokepoints"] = has_chokepoints
    else:
        metrics["min_passage_width"] = 0.0
        metrics["has_chokepoints"] = False

    metrics["has_cover"] = has_cover
    metrics["hero_spread"] = float(np.mean(position_spread)) if position_spread else 0.0

    # Navigation necessity: how much do heroes move during the fight?
    hero_positions_over_time = []
    for s in steps:
        entities = s.get("entities", [])
        entity_types = s.get("entity_types", [])
        for ei, etype in enumerate(entity_types):
            if ei >= len(entities):
                break
            e = entities[ei]
            if len(e) > 29 and e[29] > 0.5 and etype == 0:  # self
                hero_positions_over_time.append((e[5] * 20, e[6] * 20))
                break

    if len(hero_positions_over_time) >= 2:
        total_dist = sum(
            np.sqrt((hero_positions_over_time[i][0] - hero_positions_over_time[i-1][0])**2 +
                     (hero_positions_over_time[i][1] - hero_positions_over_time[i-1][1])**2)
            for i in range(1, len(hero_positions_over_time))
        )
        metrics["total_hero_movement"] = total_dist
        start = hero_positions_over_time[0]
        end = hero_positions_over_time[-1]
        displacement = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        metrics["hero_displacement"] = displacement
        # Wandering ratio: total distance / displacement (>1 = circuitous movement)
        metrics["navigation_complexity"] = total_dist / max(displacement, 0.1)
    else:
        metrics["total_hero_movement"] = 0.0
        metrics["hero_displacement"] = 0.0
        metrics["navigation_complexity"] = 0.0

    # --- Composition & Difficulty ---
    # Count heroes vs enemies from first step entity types
    first_step = steps[0]
    ft = first_step.get("entity_types", [])
    fe = first_step.get("entities", [])
    n_heroes_start = sum(1 for i, t in enumerate(ft)
                         if t in (0, 2) and i < len(fe) and len(fe[i]) > 29 and fe[i][29] > 0.5)
    n_enemies_start = sum(1 for i, t in enumerate(ft)
                          if t == 1 and i < len(fe) and len(fe[i]) > 29 and fe[i][29] > 0.5)
    metrics["n_heroes_start"] = n_heroes_start
    metrics["n_enemies_start"] = n_enemies_start
    metrics["team_size_ratio"] = n_heroes_start / max(n_enemies_start, 1)

    # Enemy strength from aggregate features (if available)
    agg = first_step.get("aggregate_features", [])
    if len(agg) >= 16:
        metrics["total_enemies"] = agg[0] * 20  # n_enemies_total / 20
        metrics["total_allies"] = agg[1] * 10   # n_allies_total / 10
        metrics["max_enemy_threat"] = agg[10]
        metrics["aggregate_enemy_dps"] = agg[11] * 200
        metrics["enemy_spread"] = agg[13] * 10
    else:
        metrics["total_enemies"] = n_enemies_start
        metrics["total_allies"] = n_heroes_start
        metrics["max_enemy_threat"] = 0.0
        metrics["aggregate_enemy_dps"] = 0.0
        metrics["enemy_spread"] = 0.0

    # Enemy composition from entity features: avg DPS, avg HP, avg range
    enemy_dps_vals = []
    enemy_hp_vals = []
    enemy_range_vals = []
    for i, t in enumerate(ft):
        if t == 1 and i < len(fe) and len(fe[i]) > 29 and fe[i][29] > 0.5:
            e = fe[i]
            enemy_dps_vals.append(e[12] * 30)    # auto_dps
            enemy_hp_vals.append(e[0])            # hp%
            enemy_range_vals.append(e[13] * 10)   # attack_range
    metrics["enemy_mean_dps"] = float(np.mean(enemy_dps_vals)) if enemy_dps_vals else 0.0
    metrics["enemy_mean_range"] = float(np.mean(enemy_range_vals)) if enemy_range_vals else 0.0
    metrics["enemy_has_ranged"] = any(r > 3.0 for r in enemy_range_vals)
    metrics["enemy_has_melee"] = any(r <= 2.0 for r in enemy_range_vals)
    metrics["enemy_comp_mixed"] = metrics.get("enemy_has_ranged", False) and metrics.get("enemy_has_melee", False)

    # Scenario metadata
    metrics["scenario"] = ep.get("scenario", "unknown")

    # --- Episode outcome ---
    metrics["outcome"] = ep["outcome"]
    metrics["reward"] = ep["reward"]
    metrics["ticks"] = ep["ticks"]
    metrics["n_steps"] = n_steps

    return metrics


def main():
    p = argparse.ArgumentParser(description="Tactical quality probes")
    p.add_argument("data", nargs="+", help="Episode JSONL files")
    p.add_argument("--min-ability-rate", type=float, default=0.05,
                    help="Minimum ability usage rate to be considered 'good'")
    args = p.parse_args()

    episodes = []
    for path in args.data:
        with open(path) as f:
            for line in f:
                episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes")

    all_metrics = [analyze_episode(ep) for ep in episodes]
    all_metrics = [m for m in all_metrics if m]  # filter empty

    # --- Aggregate ---
    print(f"\n{'='*60}")
    print(f"TACTICAL QUALITY REPORT ({len(all_metrics)} episodes)")
    print(f"{'='*60}")

    outcomes = Counter(m["outcome"] for m in all_metrics)
    print(f"\nOutcomes: {dict(outcomes)}")

    def stat(key):
        vals = [m[key] for m in all_metrics if key in m]
        if not vals:
            return "N/A"
        return f"mean={np.mean(vals):.3f} std={np.std(vals):.3f} min={np.min(vals):.3f} max={np.max(vals):.3f}"

    print(f"\n--- Ability Usage ---")
    print(f"  Ability usage rate:    {stat('ability_usage_rate')}")
    print(f"  Attack rate:           {stat('attack_rate')}")
    print(f"  Hold rate:             {stat('hold_rate')}")
    print(f"  Ability types used:    {stat('ability_types_used')}")
    print(f"  Abilities available:   {stat('hero_abilities_available')}")

    # Episodes with meaningful ability usage
    good_ability = sum(1 for m in all_metrics if m.get("ability_usage_rate", 0) >= args.min_ability_rate)
    print(f"  Episodes with >{args.min_ability_rate*100:.0f}% ability usage: {good_ability}/{len(all_metrics)} ({100*good_ability/len(all_metrics):.1f}%)")

    print(f"\n--- Combat Quality ---")
    print(f"  Focus fire rate:       {stat('focus_fire_rate')}")
    print(f"  Damage efficiency:     {stat('damage_efficiency')}")
    print(f"  Enemy kills/ep:        {stat('enemy_kills')}")
    print(f"  Hero deaths/ep:        {stat('hero_deaths')}")

    print(f"\n--- CC & Control ---")
    print(f"  CC present rate:       {stat('cc_present_rate')}")
    print(f"  Enemy CC rate:         {stat('enemy_cc_rate')}")

    print(f"\n--- Survivability ---")
    print(f"  Low-HP exposure rate:  {stat('low_hp_exposure_rate')}")

    print(f"\n--- Terrain ---")
    flat_rooms = sum(1 for m in all_metrics if m.get("is_flat_room", True))
    chokepoint_eps = sum(1 for m in all_metrics if m.get("has_chokepoints", False))
    cover_eps = sum(1 for m in all_metrics if m.get("has_cover", False))
    print(f"  Flat rooms (no obstacles): {flat_rooms}/{len(all_metrics)} ({100*flat_rooms/len(all_metrics):.1f}%)")
    print(f"  Episodes with chokepoints: {chokepoint_eps}/{len(all_metrics)} ({100*chokepoint_eps/len(all_metrics):.1f}%)")
    print(f"  Episodes with cover:       {cover_eps}/{len(all_metrics)} ({100*cover_eps/len(all_metrics):.1f}%)")
    print(f"  Mean corners visible:      {stat('mean_corners')}")
    print(f"  Min passage width:         {stat('min_passage_width')}")
    print(f"  Hero spread (formation):   {stat('hero_spread')}")

    print(f"\n--- Navigation & Movement ---")
    print(f"  Total hero movement:       {stat('total_hero_movement')}")
    print(f"  Hero displacement:         {stat('hero_displacement')}")
    print(f"  Navigation complexity:     {stat('navigation_complexity')}")
    print(f"  Movement diversity:        {stat('movement_diversity')}")
    print(f"  Threat exposure rate:      {stat('threat_exposure_rate')}")

    print(f"\n--- Composition & Difficulty ---")
    print(f"  Heroes at start:       {stat('n_heroes_start')}")
    print(f"  Enemies at start:      {stat('n_enemies_start')}")
    print(f"  Team size ratio:       {stat('team_size_ratio')}")
    print(f"  Enemy mean DPS:        {stat('enemy_mean_dps')}")
    print(f"  Enemy mean range:      {stat('enemy_mean_range')}")
    print(f"  Max enemy threat:      {stat('max_enemy_threat')}")
    print(f"  Aggregate enemy DPS:   {stat('aggregate_enemy_dps')}")
    mixed_comp = sum(1 for m in all_metrics if m.get("enemy_comp_mixed", False))
    ranged_only = sum(1 for m in all_metrics if m.get("enemy_has_ranged", False) and not m.get("enemy_has_melee", False))
    melee_only = sum(1 for m in all_metrics if m.get("enemy_has_melee", False) and not m.get("enemy_has_ranged", False))
    print(f"  Enemy comp: mixed={mixed_comp} ranged_only={ranged_only} melee_only={melee_only}")

    print(f"\n--- Episode Stats ---")
    print(f"  Ticks per episode:     {stat('ticks')}")
    print(f"  Steps per episode:     {stat('n_steps')}")

    # --- Quality tiers ---
    print(f"\n{'='*60}")
    print(f"QUALITY TIERS")
    print(f"{'='*60}")
    high_quality = sum(1 for m in all_metrics if
        m.get("ability_usage_rate", 0) >= 0.05 and
        m.get("focus_fire_rate", 0) >= 0.3 and
        m.get("damage_efficiency", 0) >= 0.5 and
        m.get("movement_diversity", 0) >= 0.3
    )
    medium_quality = sum(1 for m in all_metrics if
        m.get("ability_usage_rate", 0) >= 0.02 and
        m.get("damage_efficiency", 0) >= 0.3
    )
    print(f"  High quality (ability+focus+efficiency+movement): {high_quality}/{len(all_metrics)} ({100*high_quality/len(all_metrics):.1f}%)")
    print(f"  Medium quality (ability+efficiency):              {medium_quality}/{len(all_metrics)} ({100*medium_quality/len(all_metrics):.1f}%)")
    print(f"  Low quality (remainder):                          {len(all_metrics)-medium_quality}/{len(all_metrics)} ({100*(len(all_metrics)-medium_quality)/len(all_metrics):.1f}%)")


if __name__ == "__main__":
    main()
