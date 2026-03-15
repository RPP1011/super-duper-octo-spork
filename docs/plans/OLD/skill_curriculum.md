# Skill Curriculum: Training Drills

## Goal

Train a neural combat AI that produces **tactically interesting behavior** — not just
stat-optimal play, but coordinated team actions, terrain utilization, combo execution,
and adaptive opponent reading. The kind of plays that create "hype moments" in competitive
games: a perfectly timed CC chain, a clutch peel, a knockback into a wall, a team diving
through a chokepoint to isolate a healer.

Previous attempts at end-to-end RL (IMPALA, SAC) failed because the model tried to learn
everything at once from a sparse win/loss signal. This curriculum decomposes the problem
into isolated, verifiable skills — each with its own reward function, pass criteria, and
runtime checks — then composes them into full combat.

## Architecture

```
Model: AbilityActorCriticV4 (134K params)
├── Entity Encoder (self-attention, 7 slots × 30 features)
├── GRU Temporal Context (h=64, carries state across ticks)
├── Cross-Attention (ability embeddings → entity tokens)
├── Move Head (9-way directional) ← enabled Phase 1
├── Combat Head (attack/hold + pointer) ← enabled Phase 3
├── Ability Head (ability slots via combat mask) ← enabled Phase 4
└── Value Head (state → scalar, training only)

Inference: GPU SHM at ~60K inf/sec (4096 parallel Rust sims)
Training: IMPALA V-trace with batched trajectory GRU unrolling
Enemies: Behavior DSL scripts (reusable for game NPCs)
```

**Action head gating**: Only the move head is active in early phases. Combat and ability
actions are masked to `hold` in Rust — the model literally cannot attack until Phase 3.
This forces it to learn spatial reasoning before combat decision-making.

**Drill opponents**: Defined via a behavior DSL (`.behavior` files) that scripts enemy
decision-making. These use real abilities from hero templates — the DSL controls when
and on whom to use them. The same behaviors serve as reusable NPC AI in the game.

## Reward Design Principles

All drill rewards follow these rules to ensure stable training and composability into Phase 6.

**Detailed reward functions for every drill are in [reward_functions.md](reward_functions.md).**

### 1. Normalized scale
Every drill's expected episode return falls in **[-0.5, +0.8]** for a competent policy.
Total reward budget per drill is bounded. Raw per-tick signals are scaled so that
the sum over a typical episode length lands in this range.

Formula: `per_tick_reward ≈ target_total / expected_episode_length`

Example: drill with 300-tick expected episode and target total of +0.5 → per-tick ≈ +0.0017.
One-time bonuses are expressed as multiples of the per-tick signal, not as raw large numbers.

### 2. Continuous shaping toward the goal
Every drill has a **continuous, per-tick shaping signal** that provides gradient toward
the desired behavior. No drill relies solely on sparse outcome bonuses.
- Movement: continuous distance reduction signal
- Positioning: continuous proximity to ideal position
- Targeting: per-tick reward for attacking the correct target
- Timing: partial credit for near-correct timing (not just exact thresholds)

### 3. Reward matches verification
If runtime verification checks for a behavior, the reward must incentivize it.
No "verify but don't reward" mismatches. Conversely, no reward for behaviors
we don't verify (prevents reward hacking on unmonitored dimensions).

### 4. Time penalty everywhere
Every drill includes a small per-tick time penalty (`-0.002`) so faster completion
is always preferred. This prevents "correct but slow" policies that fail in Phase 6.

### 5. No exploitable structure
- Distance penalties use `max(0, distance - attack_range)` not raw distance (no melee incentive)
- Position rewards use Gaussians centered on ideal position, not step functions
- Corner/wall camping is penalized where kiting is tested
- Action flickering penalty: `-0.01` if `action_type[t] != action_type[t-1]` and game state
  didn't meaningfully change (same entities in range, no new cast, no HP change)

### 6. Credit assignment for coordination
Phase 5 drills use the single-model-multiple-units architecture (one forward pass per unit,
but same model weights). Each unit receives its own reward based on:
- **Individual component**: reward for that unit's specific role behavior
- **Shared component**: small fraction of team outcome (e.g., +0.1 × team_win)
- **Coordination bonus**: reward for matching ally's target, timing CC after ally's CC, etc.
  These are observable from the entity features (ally target, ally cc_remaining).

This is CTDE-compatible: each unit can learn from its own observation what the "coordinated"
reward signal means, because it sees ally state in entity features.

## Advancement

- Each phase: **100/100 randomized trials** must pass before advancing
- Each trial has **runtime verification** checks (not just pass/fail — checks HOW it passed)
- **Regression testing**: after advancing, re-run all prior phases at 95/100 threshold
- **Feature verification**: Phase 0 validates the entire data pipeline before any training

## Phase Summary

| Phase | Skills | Action Heads | Drills | Key Capability |
|-------|--------|-------------|--------|----------------|
| 0 | — | — | 14 checks | Feature pipeline verification |
| 1 | Movement | Move only | 6 drills | Navigate to targets, avoid obstacles, react to dynamic terrain |
| 2 | Spatial | Move only | 4 drills | Kite enemies, dodge zones, dodge telegraphed abilities |
| 3 | Targeting | Move + Attack | 11 drills | Kill priority, threat assessment, healer focus, horde combat, terrain use |
| 4 | Abilities | Move + Attack + Abilities | 9 drills | Heal, CC, interrupt, selective interrupt, combos, knockback, AoE, cooldown mgmt |
| 5 | Coordination | All | 7 drills | Focus fire, CC chain, peel, dive, engage timing, terrain coordination |
| 6 | Full Combat | All | 4 stages | Attrition, asymmetric, full diversity, self-play |

---

---

## Phase 0: Feature Verification (no training — diagnostic only)

Before any training, verify that the data pipeline is correct end-to-end.
Each check spawns a known scenario, runs one tick, and asserts feature values match expected.

### 0.1 Position Feature Sanity
- Spawn unit at known position (10.0, 10.0). Read entity features.
- **Assert**: `feature[5] == 10.0 / 20.0 == 0.5` (pos_x / 20)
- **Assert**: `feature[6] == 10.0 / 20.0 == 0.5` (pos_y / 20)
- Move unit one step north. Assert position changes by expected delta.
- Assert position features are different between two entities at different positions.

### 0.2 HP Feature Sanity
- Spawn unit at full HP. Assert `feature[0] == 1.0` (hp_pct).
- Apply known damage (50% HP). Assert `feature[0] ≈ 0.5`.
- Kill the unit. Assert `feature[0] == 0.0` or entity slot removed.

### 0.3 Entity Type Mapping
- Spawn 1 hero, 1 enemy, 1 ally.
- **Assert**: `entity_types[self_slot] == 0`
- **Assert**: `entity_types[enemy_slot] == 1`
- **Assert**: `entity_types[ally_slot] == 2`
- **Assert**: entity slots are ordered: self first, then enemies, then allies (per game_state.rs comment)

### 0.4 Range and Combat Features
- Spawn unit with known attack_range (e.g., 4.0).
- **Assert**: `feature[13] == 4.0 / 10.0 == 0.4` (attack_range / 10)
- Spawn unit with known DPS. Assert `feature[12]` matches `dps / 30`.
- Set attack on cooldown. Assert `feature[14] > 0` (attack_cd_remaining_pct).

### 0.5 Ability Features
- Spawn unit with known ability (e.g., 50 damage, range 5, 300 tick cooldown).
- **Assert**: `feature[15] == 50.0 / 50.0 == 1.0` (ability_damage / 50)
- **Assert**: `feature[16] == 5.0 / 10.0 == 0.5` (ability_range / 10)
- Use the ability. Assert `feature[17] > 0` (ability_cd_remaining_pct).

### 0.6 Healing Features
- Spawn unit with heal ability (heal 30 HP, range 6).
- **Assert**: `feature[18] == 30.0 / 50.0 == 0.6` (heal_amount / 50)
- **Assert**: `feature[19] == 6.0 / 10.0 == 0.6` (heal_range / 10)

### 0.7 CC Features
- Spawn unit with stun (range 4, duration 1500ms).
- **Assert**: `feature[21] == 4.0 / 10.0 == 0.4` (control_range / 10)
- **Assert**: `feature[22] == 1500.0 / 2000.0 == 0.75` (control_duration / 2000)
- Apply CC to enemy. Assert enemy's `feature[26] > 0` (cc_remaining / 2000).

### 0.8 State Features
- Spawn unit, start casting an ability.
- **Assert**: `feature[24] == 1.0` (is_casting)
- **Assert**: `feature[25]` increases over time (cast_progress)
- Unit finishes cast. Assert `feature[24] == 0.0`.

### 0.9 Position Token Verification
- Spawn unit in room with known wall at position (5, 5).
- Read position tokens. Assert at least one position token has `wall_proximity > 0`.
- Spawn in open room. Assert all position tokens have `wall_proximity == 0`.
- Assert position token directions are correct (token pointing north has dy < 0).

### 0.10 Threat Token Verification
- Spawn enemy projectile heading toward unit.
- Read threat tokens. Assert at least one threat token is populated.
- Assert threat position is near expected impact point.
- No projectiles active → assert all threat tokens are masked.

### 0.11 GRU Hidden State Round-Trip
- Submit inference request with known hidden_state_in = [1.0, 0.0, ...].
- Read response's hidden_state_out.
- **Assert**: hidden_state_out differs from hidden_state_in (GRU processed it).
- **Assert**: hidden_state_out has same dimension (h_dim).
- Submit second request with hidden_state_in = hidden_state_out from first.
- **Assert**: second hidden_state_out differs from first (state evolves).

### 0.12 Action Masking Verification
- Spawn unit with no enemies in range. Assert `combat_mask[0] == false` (attack disabled).
- Spawn unit with enemy in range. Assert `combat_mask[0] == true`.
- Spawn unit with ability on cooldown. Assert `combat_mask[2+ability_idx] == false`.
- Spawn unit with ability ready. Assert `combat_mask[2+ability_idx] == true`.

### 0.13 Reward Signal Verification
- Run a known scenario where hero kills enemy. Assert `step_reward > 0` on damage-dealing ticks.
- Run a scenario where hero takes damage. Assert `step_reward < 0` on those ticks.
- Assert time penalty is present: `step_reward` on idle ticks should be negative (≈ -0.01).
- Assert outcome: Victory episode has positive total reward, Defeat has negative.

### 0.14 Cross-Attention Embedding Verification
- Spawn unit with known ability "Fireball" (exists in registry).
- Assert ability CLS embedding is non-zero and matches registry entry.
- Spawn unit with unknown ability. Assert CLS embedding is None/zero (graceful fallback).

### Implementation
Phase 0 runs as a test suite (`training/verify_features.py`) that:
1. Creates known sim states via Rust (`xtask scenario oracle verify-features`)
2. Runs one inference cycle through SHM
3. Reads back features from the episode data
4. Asserts all values match expected

All Phase 0 checks must pass before Phase 1 starts. If any check fails, training
is blocked and the specific feature pipeline bug is reported with expected vs actual values.

---

## Phase 1: Movement Fundamentals (move head only)

### 1.1 Reach Static Point
- **Prerequisites**: None (first drill)
- **Setup**: Empty 20×20 room, unit spawns at random position, target marker at random position. Min distance between spawn and target: 8 units.
- **Reward**: `(start_distance - current_distance) / start_distance` per step
- **Pass**: Unit center within 1.0 units of target
- **Timeout**: 200 ticks
- **What it teaches**: Basic directional movement, mapping 9-way move head to spatial goals
- **Runtime verification**:
  - Assert unit position changes between ticks (not stuck)
  - Assert final distance < 1.0 (not just timing out near target)
  - Assert path length < 3× euclidean distance (not walking in circles)
  - Log: ticks_to_reach, path_length, euclidean_distance, path_efficiency_ratio

### 1.2 Reach Moving Target
- **Prerequisites**: 1.1 passed
- **Setup**: Empty 20×20 room, target moves at 50% of unit's move speed on random waypoints, changing direction every 50-100 ticks
- **Reward**: `(prev_distance - current_distance)` per step (reward for closing distance)
- **Pass**: Touch target (within 1.0 units) at any point
- **Timeout**: 400 ticks
- **What it teaches**: Pursuit/interception, predicting target trajectory
- **Runtime verification**:
  - Assert unit moves toward target on >60% of ticks (not wandering randomly)
  - Assert interception, not just chasing (path should cut corners)
  - Assert not exploiting target corner-trapping (target must be in open space when caught)
  - Log: ticks_to_catch, avg_closing_speed, target_direction_changes_before_catch

### 1.3 Navigate Around Obstacles
- **Prerequisites**: 1.1 passed
- **Setup**: 20×20 room with 3-5 random wall segments (each 3-8 units long). Target on opposite side from spawn. Guaranteed pathable (validated via flood fill).
- **Reward**: `(prev_nav_distance - current_nav_distance)` per step (nav grid distance, not euclidean)
- **Pass**: Reach target within 1.0 units
- **Timeout**: 400 ticks
- **What it teaches**: Reading position tokens (wall_proximity, blocked_neighbors), pathfinding
- **Runtime verification**:
  - Assert unit never walks into a wall (position never in blocked cell)
  - Assert unit navigates around at least one obstacle (path deviates from straight line)
  - Assert path length < 2× optimal nav grid path
  - Validate room layout: flood fill confirms spawn→target reachable before running drill
  - Log: path_length, optimal_path_length, walls_adjacent_count, backtrack_count

### 1.4 Navigate Under Time Pressure
- **Prerequisites**: 1.3 passed
- **Setup**: Same as 1.3 but after tick 100, a damage zone appears at spawn and expands outward at 0.1 units/tick. Unit takes 5% max HP per tick inside the zone.
- **Reward**: Distance reward + `-0.5` per tick inside damage zone
- **Pass**: Reach target before dying
- **Timeout**: 300 ticks
- **What it teaches**: Urgency, balancing speed vs safe pathing
- **Runtime verification**:
  - Assert unit reaches target faster than in 1.3 on average (time pressure works)
  - Assert total damage taken < 30% max HP (didn't just tank through the zone)
  - Assert unit moved away from expanding zone (not toward it)
  - Log: ticks_to_reach, damage_taken_pct, ticks_in_danger_zone, time_vs_1_3_baseline

### 1.5 Navigate Moving Obstacles
- **Prerequisites**: 1.3 passed
- **Setup**: 20×20 room. 2-3 obstacles that move back and forth on fixed paths (patrol between two points, speed 2.0 units/sec). Target on opposite side. Obstacles block movement (treated as dynamic walls).
- **Reward**: `(prev_nav_distance - current_nav_distance)` per step, `-3.0` for colliding with moving obstacle
- **Pass**: Reach target without colliding with any moving obstacle
- **Timeout**: 500 ticks
- **What it teaches**: Temporal reasoning about obstacle positions — must time movement through gaps. GRU tracks obstacle patrol patterns.
- **Runtime verification**:
  - Assert unit never occupied same cell as a moving obstacle
  - Assert unit waited at least once for an obstacle to pass (didn't just rush through)
  - Assert at least 2 obstacles were active and moving during traversal
  - Validate: obstacles actually moved (positions changed between ticks)
  - Log: ticks_to_reach, wait_ticks, obstacle_near_misses, path_length

### 1.6 React to Dynamic Terrain (Engineer's Wall)
- **Prerequisites**: 1.3 passed
- **Setup**: 20×20 room. Unit is navigating to target. At tick 80, a wall segment spawns blocking the current shortest path (simulating an engineer ability). A longer alternate path exists.
- **Reward**: Distance reward. `-1.0` per tick spent walking into the new wall (stuck).
- **Pass**: Reach target after wall spawns, using alternate route
- **Timeout**: 500 ticks
- **What it teaches**: Reacting to terrain changes mid-navigation. GRU must detect "my path is now blocked" and re-route.
- **Runtime verification**:
  - Assert unit was moving toward target before wall spawn
  - Assert unit changed direction within 20 ticks of wall appearing (detected the change)
  - Assert unit did not walk into wall after it spawned (0 collision ticks)
  - Assert unit found alternate path (reached target despite wall)
  - Validate: wall actually spawned at tick 80 and blocked the previous shortest path
  - Log: ticks_to_reroute, collision_ticks, pre_wall_path_dir, post_wall_path_dir, total_ticks

---

## Phase 2: Spatial Awareness (move head only, enemies present)

### 2.1 Maintain Distance from Enemy
- **Prerequisites**: 1.1 passed
- **Setup**: Single melee enemy (attack range 1.5) that chases at 80% unit speed. No combat enabled. 15×15 room.
- **Reward**: `+0.1` per tick alive, `-10.0` on taking damage
- **Pass**: Survive 500 ticks without taking any damage
- **Timeout**: 500 ticks
- **What it teaches**: Kiting fundamentals, reading enemy position + move speed
- **Runtime verification**:
  - Assert unit maintains avg distance > 2.0 from enemy (actually kiting, not just running to corner)
  - Assert unit uses >4 distinct move directions (not just running in one direction)
  - Assert enemy is within 8.0 units at all times (not just running to far corner — room is bounded)
  - Assert unit HP = 100% at end
  - Log: avg_distance, min_distance, direction_entropy, quadrants_visited

### 2.2 Dodge Danger Zones
- **Prerequisites**: 1.3 passed
- **Setup**: 20×20 room, 4-6 circular danger zones (radius 2.0-3.0) placed between spawn and goal. Goal on opposite side.
- **Reward**: `-1.0` per tick inside a zone, `+5.0` for reaching goal
- **Pass**: Reach goal with zero ticks spent inside any danger zone
- **Timeout**: 300 ticks
- **What it teaches**: Reading n_hostile_zones_nearby, spatial avoidance
- **Runtime verification**:
  - Assert unit position never inside any zone center ± radius
  - Assert unit doesn't just wait at spawn (must reach goal)
  - Assert path navigates between zones (not around the entire perimeter)
  - Validate zone layout: confirm gap exists between zones (pathable without entering any)
  - Log: ticks_to_goal, min_zone_clearance, path_length, zones_skirted_count

### 2.3 Dodge Telegraphed Abilities
- **Prerequisites**: 2.2 passed
- **Setup**: 15×15 room. Enemy casts ground-targeted AoE every 60 ticks (warning zone visible for 30 ticks before damage). AoE radius 3.0, targeted at unit's position at cast start.
- **Reward**: `-5.0` per hit, `+0.1` per tick survived
- **Pass**: Survive 500 ticks (at least 8 casts) with zero hits taken
- **Timeout**: 500 ticks
- **What it teaches**: Reading zone positions + timing, reactive movement
- **Runtime verification**:
  - Assert unit moves >1.5 units between cast-start and cast-land for each dodge
  - Assert unit doesn't just stay in one corner (enemy re-targets each cast)
  - Assert at least 8 casts happen (unit didn't somehow prevent casting)
  - Assert unit HP = 100% at end
  - Log: casts_dodged, avg_dodge_distance, avg_reaction_ticks, closest_call_distance

### 2.4 Kite Melee Enemy
- **Prerequisites**: 2.1 passed
- **Setup**: 20×20 room. Melee enemy (range 1.5, 100% unit speed). Unit has ranged auto-attack (range 4.0, fires automatically when enemy in range and attack off cooldown). Enemy HP = 10× auto-attack damage (takes ~10 hits to kill).
- **Reward**: `+0.5` per auto-attack landed, `-2.0` per hit taken
- **Pass**: Enemy dies, unit HP > 50%
- **Timeout**: 600 ticks
- **What it teaches**: Attack-move pattern (auto-attacks fire on their own, unit just needs to position at max range)
- **Runtime verification**:
  - Assert unit dealt damage (auto-attacks fired)
  - Assert unit took < 50% HP in damage (actually kited, not face-tanking)
  - Assert avg distance to enemy between 3.0 and 5.0 during combat (at attack range, not running away completely)
  - Assert unit moved in >2 distinct directions (orbiting, not just backpedaling)
  - Log: hits_dealt, hits_taken, avg_combat_distance, kill_tick, hp_remaining_pct

---

## Phase 3: Target Selection (enable combat head — attack + hold only, no abilities)

### 3.1 Kill Stationary Target
- **Prerequisites**: 1.1 passed
- **Setup**: Single dummy enemy (0 DPS, doesn't move, 500 HP). Unit has attack range 4.0.
- **Reward**: `damage_dealt_this_tick * 0.1`
- **Pass**: Target reaches 0 HP
- **Timeout**: 200 ticks
- **What it teaches**: Basic attack action, target pointer selecting the enemy
- **Runtime verification**:
  - Assert unit selected attack action (not hold) on >80% of ticks in range
  - Assert target_idx pointed at the enemy (entity_type=1)
  - Assert damage dealt > 0 by tick 50 (not standing around)
  - Log: total_damage, ticks_attacking, ticks_holding, kill_tick

### 3.2 Kill Moving Target
- **Prerequisites**: 3.1 passed, 1.2 passed
- **Setup**: Enemy moves away at 50% speed, 300 HP, doesn't attack back.
- **Reward**: `damage_dealt * 0.1 - distance_to_enemy * 0.01`
- **Pass**: Target dies
- **Timeout**: 400 ticks
- **What it teaches**: Combining movement toward target with attack actions
- **Runtime verification**:
  - Assert unit alternates between moving toward and attacking (not just one or the other)
  - Assert time-to-kill < timeout * 0.7 (efficient chase)
  - Assert unit spent >60% of ticks within attack range
  - Log: kill_tick, pct_ticks_in_range, chase_distance_traveled

### 3.3 Prioritize Low HP
- **Prerequisites**: 3.1 passed
- **Setup**: 2 enemies, both passive (0 DPS, don't move). Enemy A: 100% HP (1000 HP). Enemy B: 10% HP (100 HP). Both within attack range.
- **Reward**: `+10.0` if enemy B dies first, `-2.0` if enemy A dies first
- **Pass**: Enemy B (low HP) dies before enemy A
- **Timeout**: 300 ticks
- **What it teaches**: Reading entity HP, selecting optimal target
- **Runtime verification**:
  - Assert first kill is enemy B (the low HP one)
  - Assert >70% of attacks targeted enemy B before it died
  - Assert unit didn't waste attacks on enemy A while B was alive
  - Log: first_kill_id, pct_attacks_on_B, ticks_to_first_kill, damage_wasted_on_A

### 3.4 Prioritize High Threat (DPS)
- **Prerequisites**: 3.3 passed, 2.1 passed
- **Setup**: Ally (800 HP, no attack) being attacked by 2 enemies:
  - Enemy A: Melee, 10 DPS, close to ally
  - Enemy B: Ranged caster, 30 DPS, far from unit (range 6)
  Both have 500 HP.
- **Reward**: `ally_hp_remaining * 0.01 + damage_to_B * 0.2`
- **Pass**: Enemy B (high DPS) dies while ally is still alive
- **Timeout**: 400 ticks
- **What it teaches**: Reading auto_dps + attack_range to assess threat. Ignoring proximity bias.
- **Runtime verification**:
  - Assert enemy B dies before enemy A
  - Assert ally HP > 0 when B dies
  - Assert unit moved toward B (may need to close distance to ranged enemy)
  - Assert unit didn't just attack the nearest enemy (A is closer)
  - Log: first_kill_id, ally_hp_at_first_kill, ally_hp_at_end, ticks_moving_toward_B

### 3.5 Kill the Healer
- **Prerequisites**: 3.4 passed
- **Setup**: 2 enemies:
  - Enemy DPS: 20 DPS, 800 HP, attacking your unit
  - Enemy Healer: 5 DPS, 500 HP, heals DPS for 15 HP/sec
  Healer positioned behind DPS.
- **Reward**: `damage_to_healer * 0.3 + damage_to_dps * 0.05 - damage_taken * 0.1`
- **Pass**: Healer dies (in any order, but killing DPS first should be nearly impossible due to healing)
- **Timeout**: 600 ticks
- **What it teaches**: Reading heal_amount on entities, understanding healing negates damage
- **Runtime verification**:
  - Assert healer dies before DPS (or within 100 ticks after — healing should make DPS-first impossible)
  - Assert unit moved past/around DPS to reach healer
  - Assert unit recognized healer (>50% attacks on healer while both alive)
  - Validate: confirm healer actually healed DPS at least 3 times (heal AI working)
  - Log: healer_kill_tick, dps_kill_tick, total_healing_done_by_healer, pct_attacks_on_healer

### 3.6 Protect Ally (Defensive Targeting)
- **Prerequisites**: 3.4 passed
- **Setup**: Your ally (500 HP, DPS) being attacked by enemy A (20 DPS). Enemy B (10 DPS) is idle, not attacking anyone.
- **Reward**: `ally_survival_ticks * 0.02 + damage_to_A * 0.2 + damage_to_B * 0.05`
- **Pass**: Enemy A dies while ally is alive
- **Timeout**: 400 ticks
- **What it teaches**: Defensive target priority — kill the active threat to your ally
- **Runtime verification**:
  - Assert enemy A dies first (the one threatening ally)
  - Assert ally HP > 0 when A dies
  - Assert >60% of attacks targeted enemy A
  - Assert unit didn't ignore the fight to attack idle enemy B
  - Log: first_kill_id, ally_hp_at_A_death, pct_attacks_on_A, ally_damage_taken

### 3.7 React to Threat Change
- **Prerequisites**: 3.6 passed
- **Setup**: 2 enemies, both passive for first 100 ticks. At tick 100, enemy A starts casting a high-damage ability (visible via is_casting=1, cast_progress increases over 60 ticks, would deal 80% of unit's max HP).
- **Reward**: Before tick 100: small distance reward toward enemies. After tick 100: `+20.0` for killing A before cast completes, `-15.0` if cast completes.
- **Pass**: Enemy A dies or loses cast before cast_progress reaches 1.0
- **Timeout**: 300 ticks
- **What it teaches**: Reading is_casting + cast_progress, reactive target switching. GRU detects the state transition.
- **Runtime verification**:
  - Assert unit changed target to A within 30 ticks of cast starting (reaction time)
  - Assert A's cast was interrupted (died or CC'd — cast_progress didn't reach 1.0)
  - Assert unit was not already attacking A before tick 100 (wasn't pre-committed)
  - Validate: confirm cast actually started (is_casting was set on A at tick 100)
  - Log: reaction_ticks_to_switch, cast_progress_at_interrupt, pre_cast_target, post_cast_target

### 3.8 Multi-Threat Assessment
- **Prerequisites**: 3.5 passed, 3.6 passed, 3.7 passed
- **Setup**: Ally (600 HP, moderate DPS). 3 enemies:
  - Enemy A: High DPS (25), low HP (300), attacking your ally
  - Enemy B: Healer (0 DPS), full HP (500), healing A for 10 HP/sec
  - Enemy C: Tank (5 DPS), full HP (1500), positioned between you and B
- **Reward**: Shaped: `kill_B * 15.0 + kill_A * 8.0 + kill_C * 3.0 + ally_survival * 0.02 - damage_taken * 0.05`
- **Pass**: All enemies die, ally survives
- **Timeout**: 800 ticks
- **What it teaches**: Combining multiple threat signals — navigate past tank, prioritize healer (stops sustain), then kill DPS (saves ally), tank last
- **Runtime verification**:
  - Assert kill order: B first (healer), then A (DPS threatening ally), then C (tank) — with some tolerance
  - Assert ally HP > 0 at end
  - Assert unit pathed around or through C to reach B (didn't just attack C the whole time)
  - Assert healer B actually healed A at least twice (scenario working correctly)
  - Validate all 3 enemies had correct stats at spawn
  - Log: kill_order, ally_hp_at_end, healer_total_healing, ticks_spent_on_each_target, path_to_healer_length

### 3.9 Horde Combat (Many Weak Enemies)
- **Prerequisites**: 3.3 passed (target priority), 2.4 passed (kiting)
- **Setup**: 6 weak melee enemies (100 HP each, 5 DPS each, range 1.5) converging on unit from multiple directions. Unit has ranged attack (range 4, 20 DPS). Room 20×20.
- **Reward**: `+2.0` per kill, `-0.5` per hit taken, `+0.1` per tick alive
- **Pass**: Kill all 6 enemies, survive with >30% HP
- **Timeout**: 600 ticks
- **What it teaches**: Kiting multiple enemies, prioritizing the closest threat, not getting surrounded. Must keep moving while attacking — if cornered, overwhelmed by combined DPS.
- **Runtime verification**:
  - Assert unit was never surrounded (no tick where 3+ enemies within range 2)
  - Assert unit kited (avg distance to nearest enemy > 2.0 during combat)
  - Assert kills happened at a reasonable rate (not clumped — picked off one by one)
  - Assert unit moved in multiple directions (not cornered)
  - Assert HP > 30% at end
  - Log: kills_in_order, avg_nearest_enemy_dist, max_enemies_in_range_2, hp_remaining, direction_entropy

### 3.10 Use Terrain for Advantage
- **Prerequisites**: 1.3 passed (navigation), 3.1 passed (attacking)
- **Setup**: Room with a narrow chokepoint (corridor between walls). 3 melee enemies on the other side. Unit has ranged attack (range 4). If unit holds the chokepoint, enemies can only approach one at a time.
- **Reward**: `+3.0` per kill, `-1.0` per hit taken, `+0.2` per tick in chokepoint position
- **Pass**: Kill all 3 enemies, survive with >50% HP
- **Timeout**: 500 ticks
- **What it teaches**: Using terrain to funnel enemies. Instead of fighting 3v1 in open ground (lose), hold a chokepoint so only 1 can attack at a time (win).
- **Runtime verification**:
  - Assert unit positioned in or near chokepoint for >40% of combat ticks
  - Assert enemies were funneled (max 1 enemy attacking simultaneously for >60% of ticks)
  - Assert unit took less damage than in an open-field version of the same fight (terrain advantage)
  - Assert unit didn't retreat through the chokepoint (held position)
  - Validate: chokepoint exists in room layout (validated via nav grid analysis)
  - Log: ticks_at_chokepoint, max_simultaneous_attackers, damage_taken, kills_at_chokepoint

### 3.11 Elevation Advantage
- **Prerequisites**: 3.10 passed
- **Setup**: Room with elevated platform (ramp access). Unit starts on platform (elevation 2.0). 2 enemies below (elevation 0). Elevation gives cover_bonus and vision advantage. Enemies must climb ramp to reach unit.
- **Reward**: `+2.0` per kill, `+0.1` per tick on elevated position, `-2.0` for leaving platform unnecessarily
- **Pass**: Kill both enemies without leaving the platform
- **Timeout**: 400 ticks
- **What it teaches**: Reading elevation feature, understanding that high ground is advantageous. Hold position rather than chasing.
- **Runtime verification**:
  - Assert unit stayed on elevated platform for >80% of combat ticks
  - Assert unit did not descend to enemy level
  - Assert enemies had to climb ramp (pathing worked correctly)
  - Assert cover_bonus was active while on platform
  - Log: ticks_on_platform, ticks_off_platform, elevation_at_kill_ticks, cover_bonus_avg

---

## Phase 4: Ability Usage (enable ability actions)

### 4.1 Use Heal on Low Ally
- **Prerequisites**: 3.1 passed (basic targeting)
- **Setup**: Ally at 20% HP (100/500), no enemies. Unit has single-target heal ability (heals 200 HP, range 6, cooldown 300 ticks). Ally within heal range.
- **Reward**: `+10.0` for healing ally (ally HP increases), `-0.1` per tick ally stays below 50%
- **Pass**: Ally HP rises above 60%
- **Timeout**: 100 ticks
- **What it teaches**: Ability targeting on allies, reading ally HP, choosing heal over attack
- **Runtime verification**:
  - Assert heal ability was used (combat_type matched heal ability index)
  - Assert heal targeted the ally (target_idx pointed at ally entity)
  - Assert heal was used within first 20 ticks (didn't delay unnecessarily)
  - Assert unit didn't use attack action (no enemies to attack)
  - Log: tick_heal_used, ally_hp_before, ally_hp_after, ability_used_id

### 4.2 Use CC on Enemy
- **Prerequisites**: 3.6 passed (defensive targeting)
- **Setup**: Enemy (high DPS) charging toward ally (low HP). Unit has stun ability (1.5s duration, range 5, cooldown 600 ticks). Enemy starts 6 units from ally.
- **Reward**: `+8.0` for stunning enemy, `+0.5` per tick enemy is stunned, `+5.0` if ally survives to timeout
- **Pass**: Enemy gets stunned at least once, ally survives
- **Timeout**: 200 ticks
- **What it teaches**: CC ability usage, defensive ability timing
- **Runtime verification**:
  - Assert stun ability was used
  - Assert stun targeted the enemy (not ally or self)
  - Assert enemy cc_remaining > 0 after stun lands
  - Assert ally HP > 0 at end
  - Assert stun was used before enemy reached ally (proactive, not reactive after ally takes damage)
  - Log: tick_stun_used, enemy_distance_to_ally_at_stun, ally_hp_at_end, enemy_cc_duration

### 4.3 Interrupt Enemy Cast
- **Prerequisites**: 4.2 passed
- **Setup**: Enemy starts channeling a powerful ability at tick 0 (is_casting=1, 90-tick cast time, would deal 90% of ally's max HP). Unit has stun ability (range 5). Enemy is 4 units away.
- **Reward**: `+15.0` for interrupting cast (CC applied while is_casting=1), `-15.0` if cast completes
- **Pass**: Enemy's cast is interrupted (cast_progress never reaches 1.0)
- **Timeout**: 120 ticks
- **What it teaches**: Reactive ability usage, reading is_casting, urgency
- **Runtime verification**:
  - Assert stun was used while enemy is_casting == 1
  - Assert enemy cast_progress < 1.0 at stun application
  - Assert stun was used within 60 ticks (didn't wait too long)
  - Validate: enemy was actually casting (is_casting was 1 at tick 0)
  - Log: tick_stun_used, cast_progress_at_interrupt, enemy_was_casting

### 4.3b Selective Interrupt
- **Prerequisites**: 4.3 passed
- **Setup**: Enemy casts two abilities in sequence. Cast A at tick 0: weak DoT (20 damage, 40-tick cast). Cast B at tick 80: big nuke (90% max HP, 60-tick cast). Unit has ONE stun with long cooldown (won't recharge between casts).
- **Reward**: See reward_functions.md. Per-tick urgency shaping during cast B, penalty for wasting stun on cast A.
- **Pass**: Cast B is interrupted, cast A is allowed to complete
- **Timeout**: 200 ticks
- **What it teaches**: Cast discrimination — not every cast is worth interrupting. Save CC for dangerous abilities.
- **Runtime verification**:
  - Assert stun was NOT used during cast A (patience)
  - Assert stun WAS used during cast B (correct target)
  - Assert unit survived (cast B didn't complete)
  - Assert stun was still on cooldown when cast B started (wasn't available for both)
  - Validate: both casts actually happened with correct damage values
  - Log: stun_used_on_which_cast, cast_A_completed, cast_B_interrupted, hp_remaining

### 4.4 CC Then Burst
- **Prerequisites**: 4.2 passed, 3.1 passed
- **Setup**: Single enemy (800 HP, 10 DPS). Unit has stun (1.5s duration, 5 range) + high-damage burst ability (200 damage, 3 range, 600 tick CD). Enemy fights back.
- **Reward**: `+3.0` per damage dealt while target cc_remaining > 0, `+1.0` per normal damage, `+10.0` for kill
- **Pass**: Enemy dies
- **Timeout**: 400 ticks
- **What it teaches**: Ability sequencing — stun first, then burst during CC window
- **Runtime verification**:
  - Assert stun was used before burst ability
  - Assert burst was used while enemy cc_remaining > 0 (during stun window)
  - Assert time between stun and burst < 30 ticks (immediate follow-up, not delayed)
  - Assert total burst damage landed during CC window > 0
  - Log: stun_tick, burst_tick, gap_ticks, damage_during_cc, damage_outside_cc, kill_tick

### 4.5 Knockback Into Wall
- **Prerequisites**: 4.2 passed, 1.3 passed (obstacle awareness)
- **Setup**: 15×15 room with walls. Enemy positioned 2-3 units from a wall. Unit has knockback ability (pushes 4 units in direction away from caster). Bonus stun (1s) if enemy hits wall.
- **Reward**: `+5.0` for knockback, `+15.0` bonus if enemy hits wall (cc_remaining increases from wall collision)
- **Pass**: Enemy receives knockback + wall collision stun
- **Timeout**: 300 ticks
- **What it teaches**: Positional ability usage — must position between enemy and wall, then knockback
- **Runtime verification**:
  - Assert knockback ability was used
  - Assert enemy moved away from unit after knockback
  - Assert enemy cc_remaining > stun_duration (wall collision added extra stun)
  - Assert unit positioned itself on the correct side of enemy before knockback (enemy between unit and wall)
  - Validate: wall existed within knockback range behind enemy
  - Log: unit_pos_at_knockback, enemy_pos_before, enemy_pos_after, wall_distance, cc_duration_total

### 4.6 AoE Positioning
- **Prerequisites**: 1.3 passed, 3.1 passed
- **Setup**: 3 enemies clustered within 4 units of each other, all passive. Unit has AoE ability (circle radius 3.0, range 6).
- **Reward**: `damage * enemies_hit_count` (multiplicative bonus for hitting multiple)
- **Pass**: All 3 enemies hit by a single AoE cast
- **Timeout**: 300 ticks
- **What it teaches**: Positioning to maximize AoE value
- **Runtime verification**:
  - Assert AoE ability was used
  - Assert 3 enemies took damage on the same tick
  - Assert unit was within ability range (6 units) when cast
  - Assert unit moved to optimal position before casting (center of enemy cluster)
  - Log: enemies_hit, unit_distance_to_cluster_center, tick_ability_used, damage_per_enemy

### 4.7 Cooldown Management
- **Prerequisites**: 4.4 passed
- **Setup**: Enemy waves: 2 weak enemies (200 HP) spawn, then 2 more after first wave dies, then 1 strong enemy (1000 HP). Unit has powerful ability (400 damage, 1000 tick CD). Total HP pool: 400 + 400 + 1000 = 1800.
- **Reward**: `+0.5` per weak kill, `+5.0` for strong kill, `-10.0` for dying, `+3.0` for using ability on strong enemy
- **Pass**: All 3 waves cleared, unit survives
- **Timeout**: 1500 ticks
- **What it teaches**: Saving powerful abilities for high-value targets. Auto-attacking weak enemies, ability for boss.
- **Runtime verification**:
  - Assert powerful ability was used on strong enemy (wave 3), not on weak enemies (wave 1-2)
  - Assert weak enemies were killed with auto-attacks
  - Assert powerful ability was off cooldown when strong enemy spawned (wasn't wasted earlier)
  - Assert unit survived all 3 waves
  - Log: ability_used_on_wave, strong_enemy_kill_tick, ability_damage_on_weak, ability_damage_on_strong

### 4.8 Respect Enemy Cooldowns
- **Prerequisites**: 4.4 passed, 2.1 passed (kiting)
- **Setup**: Enemy has powerful stun (2s duration, range 4, 500 tick CD). When stun is ready (control_cd_remaining_pct=0), enemy will use it immediately if unit is in range. Enemy also has auto-attack (range 2, 15 DPS). Unit has range 4 attack.
- **Reward**: `+1.0` per tick attacking while enemy CC is on cooldown, `-5.0` per tick stunned, `+10.0` for kill
- **Pass**: Kill enemy while being stunned at most once
- **Timeout**: 800 ticks
- **What it teaches**: GRU temporal tracking — observe enemy CC usage, learn cooldown timing, engage only during safe windows
- **Runtime verification**:
  - Assert unit was stunned ≤ 1 time (respected cooldown after first stun)
  - Assert unit disengaged (moved out of range 4) while enemy CC was ready
  - Assert unit re-engaged when enemy CC went on cooldown
  - Assert there were at least 2 engage/disengage cycles (not just face-tanking)
  - Validate: enemy actually used CC ability at least once
  - Log: times_stunned, engage_disengage_cycles, avg_engage_duration, avg_disengage_duration, ticks_at_safe_range

---

## Phase 5: Team Coordination (multi-unit control)

### 5.1 Focus Fire
- **Prerequisites**: 3.3 passed (targeting), 3.4 passed (threat assessment)
- **Setup**: 2 allied units (both melee DPS), 2 enemies (both 600 HP, 15 DPS). All in attack range.
- **Reward**: `+5.0` per tick both allies attacking same target, `+10.0` per kill, `-5.0` per ally death
- **Pass**: Both enemies die, both allies survive
- **Timeout**: 400 ticks
- **What it teaches**: Coordinated target selection — both units converge on same target
- **Runtime verification**:
  - Assert both allies attacked the same enemy for >60% of ticks (focus fire)
  - Assert first enemy died in < 200 ticks (faster than splitting damage)
  - Assert both allies survived
  - Assert allies didn't split targets (each attacking a different enemy)
  - Log: pct_ticks_same_target, first_kill_tick, second_kill_tick, ally_hp_at_end

### 5.2 Chain CC
- **Prerequisites**: 4.2 passed (CC usage), 4.4 passed (sequencing)
- **Setup**: 2 allied units, each with a stun (1.5s duration, 600 tick CD). Single tough enemy (2000 HP, 20 DPS). Enemy attacks nearest ally.
- **Reward**: `+5.0` per stun applied, `+15.0` if second stun lands within 100ms of first expiring (CC chain, no gap), `-5.0` if stuns overlap (wasted CC)
- **Pass**: Enemy CC'd for total > 4 seconds via chaining (no overlap, no >200ms gap)
- **Timeout**: 800 ticks
- **What it teaches**: Temporal coordination — unit 2 watches cc_remaining from unit 1's stun, times its stun to land as the first wears off. Core GRU skill.
- **Runtime verification**:
  - Assert 2 stuns were applied by different units
  - Assert gap between stun1 end and stun2 start < 200ms (chained, not random)
  - Assert stuns didn't overlap by more than 100ms (not wasted)
  - Assert enemy was CC'd continuously for >3 seconds
  - Validate: both ally units had stun ability, both were used
  - Log: stun1_start, stun1_end, stun2_start, stun2_end, gap_ms, overlap_ms, total_cc_duration

### 5.3 Peel for Carry
- **Prerequisites**: 3.6 passed (protect ally), 4.2 passed (CC usage)
- **Setup**: Ally DPS (high damage 30 DPS, low HP 400) + your unit (tank, 15 DPS, 1200 HP, has taunt/stun). Enemy assassin (25 DPS) diving the DPS ally.
- **Reward**: `ally_DPS_survival * 1.0 + ally_DPS_damage_dealt * 0.05 + CC_on_enemy_near_ally * 3.0`
- **Pass**: Ally DPS survives and enemy dies
- **Timeout**: 500 ticks
- **What it teaches**: Protective play — CC the diver, body-block, keep enemy off ally
- **Runtime verification**:
  - Assert your unit applied CC to enemy while enemy was within 3 units of ally (peeling)
  - Assert ally DPS survived
  - Assert ally DPS dealt majority of lethal damage (your unit tanked/CC'd, ally DPS killed)
  - Assert your unit was between enemy and ally for >30% of ticks (body blocking)
  - Log: cc_applications_near_ally, ally_hp_at_end, ally_damage_dealt, unit_damage_taken, pct_ticks_between_enemy_ally

### 5.4 Dive Coordination
- **Prerequisites**: 5.1 passed (focus fire), 3.5 passed (kill healer)
- **Setup**: 2 allied units vs enemy healer (behind, 400 HP) + enemy DPS (front, 800 HP). Healer heals DPS for 15 HP/sec.
- **Reward**: `+15.0` for healer kill, `+5.0` for DPS kill, `ally_survival * 0.5`
- **Pass**: Healer dies, at least 1 ally survives
- **Timeout**: 600 ticks
- **What it teaches**: Coordinated aggression — both units bypass/ignore DPS to reach and kill healer
- **Runtime verification**:
  - Assert healer dies first (or within 100 ticks of DPS)
  - Assert both allies moved toward healer (not just attacking DPS)
  - Assert at least 1 ally survived
  - Assert healer received damage from both allies (coordinated dive, not just one unit)
  - Validate: healer was actually healing (healed DPS at least twice)
  - Log: healer_kill_tick, dps_kill_tick, allies_surviving, healer_damage_from_each_ally

### 5.5 Engage/Disengage Timing
- **Prerequisites**: 5.1 passed, 4.8 passed (cooldown respect)
- **Setup**: 2v2. Both enemies have powerful CC abilities (stun, 2s, 600 tick CD). If you engage while both CCs are ready, both allies get stunned and die. If you engage when at least one enemy CC is on cooldown, you win.
  - Enemies use CC immediately when a target is in range and CC is ready.
  - Enemies auto-attack at range 2.
- **Reward**: `+20.0` for winning, `HP_remaining * 0.5`, `-10.0` for losing
- **Pass**: Win the 2v2
- **Timeout**: 1000 ticks
- **What it teaches**: Team-level cooldown tracking. GRU must observe: enemy A used CC at tick X, enemy B used CC at tick Y. Engage when both are on cooldown. Disengage if one comes back up.
- **Runtime verification**:
  - Assert team did not engage while both enemy CCs were ready (first 50 ticks should be waiting/poking)
  - Assert team engaged after at least one enemy CC was observed on cooldown
  - Assert team won the fight
  - Assert total stun time received by allies < 3 seconds (avoided most CC)
  - Validate: both enemies used their CC at least once
  - Log: engage_tick, enemy_A_cc_state_at_engage, enemy_B_cc_state_at_engage, total_ally_stun_time, win

---

## Phase 6: Full Combat (all systems, real scenarios)

### 5.6 Horde Defense (Coordinated Terrain Use)
- **Prerequisites**: 3.9 passed (horde combat), 3.10 passed (terrain), 5.1 passed (focus fire)
- **Setup**: 2 allied units (1 ranged DPS, 1 tank with CC). Room with chokepoint. 8 weak melee enemies approach in waves (4 + 4) from one side. Allies start on the defended side.
- **Reward**: `+2.0` per kill, `ally_survival * 1.0`, `+0.3` per tick tank holds chokepoint, `-3.0` per ally death
- **Pass**: All 8 enemies die, both allies survive
- **Timeout**: 800 ticks
- **What it teaches**: Tank holds chokepoint, DPS shoots from behind. Coordinated terrain control under sustained pressure.
- **Runtime verification**:
  - Assert tank was in chokepoint for >50% of combat ticks
  - Assert DPS stayed behind tank (further from enemies) for >70% of ticks
  - Assert max simultaneous enemies attacking DPS < 2 (tank absorbing)
  - Assert both allies survived
  - Assert enemies were funneled through chokepoint (not flanking around)
  - Log: tank_chokepoint_pct, dps_behind_tank_pct, max_enemies_on_dps, wave1_clear_tick, wave2_clear_tick

### 5.7 Dynamic Terrain Coordination
- **Prerequisites**: 1.6 passed (dynamic terrain), 5.4 passed (dive coordination)
- **Setup**: 2 allied units. One ally is an "engineer" type with a wall-spawning ability (creates temporary wall segment for 200 ticks). 2 enemies: DPS + healer (healer behind DPS). Engineer can wall off the healer from the DPS, isolating the healer.
- **Reward**: `+15.0` if healer dies while walled off from DPS, `+5.0` for any kill, ally survival
- **Pass**: Healer dies, both allies survive
- **Timeout**: 600 ticks
- **What it teaches**: Using ability-created terrain to split enemy formation. Engineer walls between healer and DPS, then team kills isolated healer. Requires coordination: wall timing + dive timing.
- **Runtime verification**:
  - Assert wall ability was used
  - Assert wall separated healer from DPS (no nav path between them while wall active)
  - Assert healer was attacked while isolated (during wall duration)
  - Assert healer died during or shortly after wall duration
  - Validate: wall actually blocked pathing (nav grid updated)
  - Log: wall_tick, healer_isolated_ticks, healer_kill_tick, dps_damage_during_wall, wall_effectiveness

---

## Phase 6: Full Combat (all systems, real scenarios)

### 6.1 Attrition Scenarios
- **Prerequisites**: All Phase 5 drills passed
- **Setup**: Standard curriculum phase 1-2 scenarios (Tier 1 vs Tier 1, Tier 2 vs Tier 2)
- **Reward**: Standard HP differential + time penalty + outcome bonus
- **Pass**: >50% eval win rate on 200 trials
- **What it teaches**: Applying all learned skills in unstructured combat
- **Runtime verification**:
  - Assert win rate is statistically significant (binomial test p < 0.01)
  - Assert avg game length < 1500 ticks (not stalling)
  - Assert avg damage dealt > 0 per tick in range (actively fighting)
  - Regression check: re-run Phase 5 drills, assert >95/100 on each
  - Log: win_rate, avg_game_length, avg_damage_per_tick, ability_usage_rate

### 6.2 Asymmetric Tactics
- **Prerequisites**: 6.1 passed
- **Setup**: Curriculum phase 3 cross-tier matchups
- **Reward**: Standard + engagement bonus
- **Pass**: >40% eval win rate
- **What it teaches**: Adaptation to different enemy compositions
- **Runtime verification**:
  - All 6.1 verification checks
  - Assert per-matchup win rate variance is not too extreme (not just winning easy ones)
  - Assert ability usage rate > 50% (using abilities, not just auto-attacking)
  - Regression check: 6.1 scenarios still >45% win rate
  - Log: per_tier_matchup_win_rates, ability_usage_by_type

### 6.3 Full Diversity
- **Prerequisites**: 6.2 passed
- **Setup**: Curriculum phase 4 (all tiers)
- **Reward**: Standard
- **Pass**: >35% eval win rate
- **What it teaches**: Maximum generalization across all unit types and abilities
- **Runtime verification**:
  - All 6.2 verification checks
  - Assert model handles unseen ability combinations (held-out test set)
  - Regression check: 6.1 >40%, 6.2 >35%
  - Log: win_rate_by_phase, generalization_gap

### 6.4 Self-Play (mixed pool)
- **Prerequisites**: 6.3 passed
- **Setup**: Mixed pool per the paper:
  - 50% vs frozen snapshot of current best policy
  - 25% vs default AI (diverse personality weights)
  - 25% vs previous checkpoints (league-style)
  Swap sides on all matchups. Asymmetric compositions.
- **Reward**: Standard + temporal engagement bonus
- **Pass**: No fixed threshold — convergence monitoring:
  - Eval win rate vs default AI remains > 40%
  - Eval win rate vs frozen self remains > 45%
  - No phase 1-5 drill regression below 90/100
- **What it teaches**: In-context opponent adaptation, diverse strategy repertoire
- **Runtime verification**:
  - Assert opponent diversity: played against >3 distinct opponent types this iteration
  - Assert in-context adaptation: win rate in second half of episode > first half (GRU learning)
  - Assert no strategy collapse: action entropy remains above threshold
  - Periodic regression: all Phase 1-5 drills pass at 95/100
  - Log: win_rate_per_opponent_type, first_half_vs_second_half_wr, action_entropy, drill_regression_results

---

## Implementation Notes

### Action Head Gating
- **Phases 1-2**: Only move head active. Rust emits `combat_type=1 (hold)` regardless of model output. No combat gradients.
- **Phase 3**: Enable combat head (attack + hold + pointer). `combat_mask = [attack, hold, false, false, ...]`. No ability slots.
- **Phase 4+**: Enable abilities per unit's kit via combat_mask.

### Regression Testing
After each phase advancement, re-run all previously passed drills. If any phase regresses below 95/100:
1. Log which specific drills failed and why (runtime verification tells us)
2. Freeze current phase, retrain on regressed drills until 100/100 restored
3. Resume curriculum

### Drill Randomization
Each drill randomizes:
- Spawn positions (within constraints)
- Room layout (obstacles)
- Enemy positions
- Target positions
- Ability cooldown offsets (±10% to prevent memorized timing)

Seed is recorded per trial for reproducibility. Failed trials can be replayed with the same seed.

### Metrics Dashboard Integration
Each drill reports structured metrics to the training dashboard (tools/dashboard/):
- Pass/fail per trial with seed
- Per-drill verification check results (which checks passed/failed)
- Aggregate statistics (pass rate trend over training iterations)
- Comparison to previous checkpoints (regression detection)
