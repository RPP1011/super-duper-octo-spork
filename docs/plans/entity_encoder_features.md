# Entity Encoder Feature Reference

## Overview

The entity encoder processes structured game state as a set of typed tokens. Each token is projected to d_model=32 and processed by a 4-layer transformer encoder with self-attention across all token types.

**Token budget per sample:**

| Token Type | Max Slots | Feature Dim | Type ID | Source |
|------------|-----------|-------------|---------|--------|
| Entity     | 7         | 32          | 0/1/2   | Game state extraction |
| Threat     | 8         | 8           | 3       | Game state extraction |
| Position   | 8         | 8           | 4       | Game state extraction |
| Ability    | 63 (9×7)  | 32          | 5       | Frozen ability transformer |
| **Total**  | **86**    |             |         |        |

---

## Entity Tokens (7 slots × 32 features)

Slot order: `[self, enemy0, enemy1, enemy2, ally0, ally1, ally2]`
- Enemies sorted by distance (nearest first)
- Allies sorted by HP% (lowest first)
- Type IDs: 0=self, 1=enemy, 2=ally

### Raw Features (30 dimensions)

#### Vitals (indices 0–4)

| Index | Feature | Normalization | Dynamic? | Predicted? |
|-------|---------|---------------|----------|------------|
| 0 | hp_pct | [0, 1] | Yes | Yes (hp group) |
| 1 | shield_pct | shield_hp / max_hp | Yes | Yes (hp group) |
| 2 | resource_pct | [0, 1] | Yes | Yes (hp group) |
| 3 | armor | / 200 | No | No |
| 4 | magic_resist | / 200 | No | No |

#### Position / Terrain (indices 5–11)

| Index | Feature | Normalization | Dynamic? | Predicted? |
|-------|---------|---------------|----------|------------|
| 5 | position_x | / 20 (absolute) | Yes | Yes (pos group) |
| 6 | position_y | / 20 (absolute) | Yes | Yes (pos group) |
| 7 | distance_from_caster | / 10 (0 for self) | Yes | No |
| 8 | cover_bonus | [0, 1] | Yes* | No |
| 9 | elevation | / 5 | Yes* | No |
| 10 | hostile_zones_nearby | / 3 | Yes | No |
| 11 | friendly_zones_nearby | / 3 | Yes | No |

*Derived from position — changes when unit moves.

#### Combat Stats (indices 12–14)

| Index | Feature | Normalization | Dynamic? | Predicted? |
|-------|---------|---------------|----------|------------|
| 12 | auto_dps | / 30 | No | No |
| 13 | attack_range | / 10 | No | No |
| 14 | attack_cd_remaining_pct | [0, 1] | Yes | No |

#### Ability Readiness (indices 15–17)

| Index | Feature | Normalization | Dynamic? | Predicted? |
|-------|---------|---------------|----------|------------|
| 15 | ability_damage | / 50 | No | No |
| 16 | ability_range | / 10 | No | No |
| 17 | ability_cd_remaining_pct | [0, 1] | Yes | No (weight=0) |

#### Healing (indices 18–20)

| Index | Feature | Normalization | Dynamic? | Predicted? |
|-------|---------|---------------|----------|------------|
| 18 | heal_amount | / 50 | No | No |
| 19 | heal_range | / 10 | No | No |
| 20 | heal_cd_remaining_pct | [0, 1] | Yes | No (weight=0) |

#### CC Capability (indices 21–23)

| Index | Feature | Normalization | Dynamic? | Predicted? |
|-------|---------|---------------|----------|------------|
| 21 | control_range | / 10 | No | No |
| 22 | control_duration | / 2000 | No | No |
| 23 | control_cd_remaining_pct | [0, 1] | Yes | No (weight=0) |

#### Current State (indices 24–27)

| Index | Feature | Normalization | Dynamic? | Predicted? |
|-------|---------|---------------|----------|------------|
| 24 | is_casting | 0/1 | Yes | Yes (state group, weight=0.1) |
| 25 | cast_progress | [0, 1] | Yes | Yes (state group, weight=0.1) |
| 26 | cc_remaining | / 2000 | Yes | Yes (state group, weight=0.1) |
| 27 | move_speed | / 5 | No | Yes (state group, weight=0.1) |

#### Cumulative / Meta (indices 28–29)

| Index | Feature | Normalization | Dynamic? | Predicted? |
|-------|---------|---------------|----------|------------|
| 28 | total_damage_done | / 1000 | Yes | No |
| 29 | exists | 0/1 (alive) | Yes | Yes (exists group) |

### Augmented Features (indices 30–31, computed at training time)

| Index | Feature | Description |
|-------|---------|-------------|
| 30 | dx_from_self | entity_x - self_x (relative position) |
| 31 | dy_from_self | entity_y - self_y (relative position) |

These are computed in the Python data pipeline from raw features, not stored in the npz dataset. Self entity always has dx=0, dy=0.

---

## Threat Tokens (8 slots × 8 features)

Incoming dangers relative to the unit being encoded: in-flight projectiles, hostile zones, enemy casts targeting this unit or nearby ground.

Sorted by urgency (time_to_impact ascending). Type ID: 3.

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | dx | / 10 (relative to unit) |
| 1 | dy | / 10 (relative to unit) |
| 2 | distance | / 10 (to impact point) |
| 3 | radius | / 5 (AoE radius, 0 for single-target) |
| 4 | time_to_impact | / 2000 (ms until damage lands) |
| 5 | damage_ratio | incoming_damage / unit_hp (>1.0 = lethal) |
| 6 | has_cc | 0/1 (includes stun/root/silence) |
| 7 | exists | 0/1 (slot occupied) |

---

## Position Tokens (8 slots × 8 features)

Points of interest in the environment: cover positions, chokepoints, elevated ground.

Type ID: 4.

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | dx | relative to unit |
| 1 | dy | relative to unit |
| 2 | pathfinding_distance | actual path distance |
| 3 | elevation | / 5 |
| 4 | chokepoint_proximity | [0, 1] |
| 5 | wall_proximity | [0, 1] |
| 6 | hostile_zones | count nearby |
| 7 | friendly_zones | count nearby |

---

## Prediction Target Groups

The model predicts changes (residual deltas in symlog space) for dynamic feature groups only. Each group has its own prediction head and loss weight.

| Group | Indices | N Features | Loss Weight | Notes |
|-------|---------|------------|-------------|-------|
| hp | 0, 1, 2 | 3 | 1.0 | Core vitals |
| pos | 5, 6 | 2 | 1.0 | Absolute position |
| cd | 17, 20, 23 | 3 | 0.0 | Disabled — phase transitions unpredictable |
| state | 24, 25, 26, 27 | 4 | 0.1 | Low weight — noisy binary signals |
| exists | 29 | 1 | 1.0 | Death prediction |

**Total predicted:** 13 features (10 actively weighted)

---

## Conditioning

| Input | Shape | Description |
|-------|-------|-------------|
| delta_normalized | scalar [0, 1] | Prediction horizon in ticks, normalized by max_delta |

Appended to each entity token before the prediction heads. Tells the model how far ahead to predict.

---

## Architecture Summary

```
Input:  entity(7×32) + threat(8×8) + position(8×8)
        ↓ per-type linear projection → d_model=32
        ↓ + type embedding (5 types)
        ↓ LayerNorm
        ↓ TransformerEncoder (4 layers, 4 heads, d_ff=64, pre-norm, GELU)
        ↓ LayerNorm
        ↓ extract entity tokens (first 7)
        ↓ concat delta_normalized → (B, 7, 33)
        ↓ per-group PredictionHead → (mean_delta, log_var)
        ↓ mean = symlog(current) + mean_delta
Output: predicted future features in symlog space
```

**Parameters:** ~48,700 total (36,100 encoder + 12,600 heads)

---

## Unpredicted Dynamic Features

These features change during combat but are not currently predicted:

| Index | Feature | Why Not Predicted |
|-------|---------|-------------------|
| 7 | distance_from_caster | Derivable from pos group |
| 8 | cover_bonus | Derivable from position |
| 9 | elevation | Derivable from position |
| 10 | hostile_zones_nearby | Changes with position + game events |
| 11 | friendly_zones_nearby | Changes with position + game events |
| 14 | attack_cd_remaining_pct | Phase transition (like ability CDs) |
| 28 | total_damage_done | Monotonically increasing, low signal |
