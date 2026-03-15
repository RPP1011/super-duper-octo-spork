# Drill Reward Functions (v3)

Revised addressing: scale normalization, continuous shaping, reward-verification
alignment, time pressure, exploit prevention, value head signal separation,
action flicker edge cases, premature-cast incentives, urgency gradient direction,
Gaussian tuning fragility, coordination credit bootstrapping, and phase transitions.

**Design target**: competent policy returns ≈ **[+0.1, +0.8]**, random/failing ≈ **[-0.5, -0.2]**.
Minimum separation between competent and random: **0.3** for value head signal.

---

## Notation

```
d(a, b)         = euclidean distance between a and b
d_nav(a, b)     = nav grid pathfinding distance
d_atk(u)        = unit u's attack range
clamp01(x)      = max(0, min(1, x))
gauss(x, μ, σ)  = exp(-(x - μ)² / (2σ²))
```

---

## Global Modifiers

### Time penalty
- **Survival drills** (2.1, 2.3, 2.4): `-0.001/tick` (halved — point is to survive full duration)
- **Endurance drills** (4.7, 4.8, 5.2, 5.5): `-0.001/tick` (halved — long multi-phase encounters)
- **All other drills**: `-0.002/tick`

### Action flicker penalty (Phase 3+)
`-0.01` if ALL of the following are true:
- `action_type[t] != action_type[t-1]`
- No ability was used on tick `t-1` (ability sequences exempt)
- Target didn't change (same entity in target slot)
- No new cast started on any enemy this tick
- No ally HP dropped below 30% this tick

This exempts legitimate rapid sequences (stun → burst) and reactive switches (enemy starts casting).

### Phase transition warmup
When a new action head activates (Phase 3: combat, Phase 4: abilities):
- **One-shot exploration bonus**: `+0.01` on the **first tick** of each episode where the new
  action type is used. No ongoing bonus — prevents rewarding mindless spam.
- **First 5 training iterations**: time penalty halved
- **Iterations 6-8**: time penalty at 75%
- **Iteration 9+**: full time penalty

### Gaussian tuning
All `gauss(x, μ, σ)` rewards use **wide σ** (σ ≥ 2.0) initially.

**Continuous annealing** (no discontinuity):
```
σ(pass_rate) = σ_wide - (σ_wide - σ_target) * clamp01((pass_rate - 0.3) / 0.6)
```
- At 0-30% pass rate: σ = σ_wide (full width, easy to get reward)
- At 30-90%: smoothly tightens
- At 90%+: σ = σ_target (precise)

Log actual distance distributions during training. If mean successful distance
differs from μ by >1.0, adjust μ to match observed optimal.

### Coordination bootstrapping (Phase 5)
Phase 5 drills blend heuristic proxy rewards with true coordination rewards.
Anneal is **self-pacing**, tied to pass rate, not iteration count:

```
heuristic_weight = clamp01(1.0 - pass_rate / 0.6)
reward = heuristic_weight * heuristic_reward + (1.0 - heuristic_weight) * coordination_reward
```

- At 0% pass rate: 100% heuristic (e.g., "attack lowest HP enemy")
- At 60% pass rate: 0% heuristic, 100% true coordination reward
- Transition is smooth and self-pacing — fast learners transition faster

Heuristic proxies per drill:
- 5.1 Focus Fire: "attack lowest HP enemy" (converges to focus fire naturally)
- 5.2 Chain CC: "stun any CC'd enemy" (approximate chain by stunning stunned targets)
- 5.3 Peel: "attack enemy nearest to your ally" (approximate body-blocking)
- 5.5 Engage: "attack when enemy HP is lower than yours" (approximate safe engagement)

---

## Phase 1: Movement

### 1.1 Reach Static Point (T=200)
```
per_tick:
  progress  = clamp01((prev_dist - curr_dist) / start_dist) * 0.005
  time      = -0.002
on_pass:
  bonus     = +0.3 * (1.0 - ticks_taken / timeout)   # faster = more bonus
total ≈ +0.3 competent, -0.2 random
```

### 1.2 Reach Moving Target (T=400)
```
per_tick:
  closing   = clamp01((prev_dist - curr_dist) / prev_dist) * 0.003
  time      = -0.002
on_pass:
  bonus     = +0.4 * (1.0 - ticks_taken / timeout)
total ≈ +0.25 competent, -0.3 random (harder to catch)
```

### 1.3 Navigate Around Obstacles (T=400)
```
per_tick:
  progress  = clamp01((prev_nav_dist - curr_nav_dist) / start_nav_dist) * 0.005
  wall_bump = -0.05 per tick in blocked cell
  time      = -0.002
on_pass:
  bonus     = +0.3 * (1.0 - ticks_taken / timeout)
  efficiency= +0.1 * clamp01(2.0 - path_length / optimal_path_length)
total ≈ +0.3 competent, -0.3 random
```

### 1.4 Navigate Under Time Pressure (T=300)
```
per_tick:
  progress  = (same as 1.3)
  zone_dmg  = -0.02 per tick inside expanding damage zone
  time      = -0.002
on_pass:
  bonus     = +0.3 * (1.0 - ticks_taken / timeout)
on_death:
  penalty   = -0.5
```

### 1.5 Navigate Moving Obstacles (T=500)
```
per_tick:
  progress  = (same as 1.3)
  time      = -0.002
on_collision:
  penalty   = -0.3 per collision with moving obstacle
on_pass:
  bonus     = +0.3 + 0.1 * (wait_ticks > 0)
```

### 1.6 React to Dynamic Terrain (T=500)
```
per_tick:
  progress  = (same as 1.3)
  wall_stuck= -0.03 per tick walking into the new wall
  time      = -0.002
post_wall_spawn (tick > 80):
  reroute   = +0.1 if unit changed heading within 20 ticks of wall appearing
on_pass:
  bonus     = +0.3 * (1.0 - ticks_taken / timeout)
```

---

## Phase 2: Spatial Awareness

### 2.1 Maintain Distance from Enemy (T=500)
```
per_tick:
  # Gaussian centered on ideal kite distance (μ starts at 3.0, adjusted empirically)
  dist_reward = gauss(enemy_distance, 3.0, 2.0) * 0.004
  # Corner penalty: within 2.0 of room edge on 2+ sides
  corner_pen  = -0.005 if corner_detected
  # HALVED time penalty for survival drill
  time        = -0.001
on_hit:
  penalty     = -0.05 * (hp_lost / max_hp)  # proportional, not flat
on_survive_full:
  # Scaling survival bonus
  bonus       = +0.3 if zero damage, +0.15 if < 10% damage
total ≈ +0.35 competent (kites well, no damage, +0.3 bonus), -0.15 random (gets hit a few times)
separation: 0.50 ✓
```
**Fixes**: halved time penalty (survival drill), proportional hit penalty, wider Gaussian σ=2.0.

### 2.2 Dodge Danger Zones (T=300)
```
per_tick:
  progress    = clamp01((prev_dist_to_goal - curr_dist_to_goal) / start_dist) * 0.004
  zone_pen    = -0.02 per tick inside any danger zone
  zone_margin = +0.001 * clamp01(min_clearance / zone_radius) (reward safe margin)
  time        = -0.002
on_pass:
  bonus       = +0.3 if zero zone ticks, +0.15 otherwise
```

### 2.3 Dodge Telegraphed Abilities (T=500)
```
per_tick:
  survive     = +0.001
  # HALVED time penalty for survival drill
  time        = -0.001
  # During telegraph window:
  dodge_shaping = +0.003 * clamp01(d(self, impact_center) / aoe_radius)
on_hit:
  penalty     = -0.06 per AoE hit
on_survive_clean:
  bonus       = +0.3 if zero hits
total ≈ +0.25 competent, -0.15 random
separation: 0.40 ✓
```

### 2.4 Kite Melee Enemy (T=600)
```
per_tick:
  range_reward = gauss(enemy_distance, d_atk(self) - 0.5, 2.0) * 0.003
  too_far_pen  = -0.002 * max(0, enemy_distance - d_atk(self)) / d_atk(self)
  # HALVED time penalty for survival-ish drill
  time         = -0.001
on_auto_attack:
  hit_bonus    = +0.02
on_hit_taken:
  penalty      = -0.02 * (hp_lost / max_hp)
on_kill:
  bonus        = +0.2
on_death:
  penalty      = -0.3
total ≈ +0.25 competent (kills enemy, few hits taken), -0.3 random (dies or timeout)
separation: 0.55 ✓
```

---

## Phase 3: Target Selection

### 3.1 Kill Stationary Target (T=200)
```
per_tick:
  attacking   = +0.003 per tick attacking the enemy
  in_range    = +0.001 per tick within attack range
  not_acting  = -0.002 per tick in range but holding
  time        = -0.002
on_kill:
  bonus       = +0.3 * (1.0 - ticks_taken / timeout)
```

### 3.2 Kill Moving Target (T=400)
```
per_tick:
  range_ok    = +0.002 per tick where max(0, enemy_dist - d_atk(self)) == 0
  closing     = +0.002 * clamp01((prev_dist - curr_dist) / prev_dist) when out of range ONLY
  attacking   = +0.002 per tick dealing damage
  time        = -0.002
on_kill:
  bonus       = +0.3 * (1.0 - ticks_taken / timeout)
```

### 3.3 Prioritize Low HP (T=300)
```
per_tick:
  correct     = +0.005 per tick attacking enemy B (low HP)
  wrong       = -0.003 per tick attacking enemy A while B alive
  time        = -0.002
on_B_first:
  bonus       = +0.3
on_A_first:
  penalty     = -0.2
```

### 3.4 Prioritize High Threat (T=400)
```
per_tick:
  correct     = +0.004 per tick attacking enemy B (high DPS)
  wrong       = -0.002 per tick attacking enemy A while B alive
  ally_alive  = +0.001 per tick ally alive
  closing_B   = +0.002 * closing_rate when out of range of B
  time        = -0.002
on_B_killed_ally_alive:
  bonus       = +0.3
on_ally_death:
  penalty     = -0.3
```

### 3.5 Kill the Healer (T=600)
```
per_tick:
  atk_healer  = +0.004 per tick attacking healer while both alive
  atk_dps     = +0.001 per tick attacking DPS
  close_healer= +0.002 when moving toward healer and out of range
  time        = -0.002
on_healer_kill:
  bonus       = +0.3
on_dps_first:
  penalty     = -0.1
```

### 3.6 Protect Ally (T=400)
```
per_tick:
  correct     = +0.004 per tick attacking enemy A (threatening ally)
  wrong       = -0.002 per tick attacking idle enemy B while A alive
  ally_alive  = +0.001 per tick
  time        = -0.002
on_A_killed_ally_alive:
  bonus       = +0.3
on_ally_death:
  penalty     = -0.3
```

### 3.7 React to Threat Change (T=300)
```
pre_tick_100:
  closing     = +0.001 per tick closing to enemies
  time        = -0.002
post_tick_100 (cast starts):
  # FIXED: urgency INCREASES reward for correct action, not decreases
  correct     = +0.005 * (1.0 + cast_progress) per tick attacking casting enemy
  # So at progress=0: +0.005, at progress=0.9: +0.0095 (more reward for attacking late)
  wrong       = -0.003 * (1.0 + cast_progress) per tick NOT attacking caster
  # Inaction penalty GROWS with progress (urgency)
  time        = -0.002
on_interrupt:
  bonus       = +0.3 * (1.0 - cast_progress_at_interrupt)
on_cast_completes:
  penalty     = -0.4
```
**Fix**: urgency multiplier now increases reward for correct action and increases penalty for inaction, not the other way around.

### 3.8 Multi-Threat Assessment (T=800)
```
per_tick:
  atk_B       = +0.004 while B(healer) alive
  atk_A       = +0.002 while B dead and A alive
  atk_C       = +0.001 while A dead
  atk_wrong   = -0.002 (e.g., attacking C while B alive)
  close_B     = +0.002 when moving toward B out of range
  ally_alive  = +0.001
  time        = -0.002
on_correct_order:
  bonus       = +0.3
on_ally_death:
  penalty     = -0.2
total ≈ +0.15 competent, -0.35 random
separation: 0.50 ✓ (improved from v2's -0.20 competent)
```

### 3.9 Horde Combat (T=600)
```
per_tick:
  surround    = -0.003 * max(0, enemies_in_2_units - 1)
  kite_reward = gauss(nearest_enemy_dist, d_atk(self) - 0.5, 2.0) * 0.002
  time        = -0.002
on_kill:
  bonus       = +0.05 per kill
on_hit:
  penalty     = -0.01 * hp_lost_frac
on_death:
  penalty     = -0.3
```

### 3.10 Use Terrain for Advantage (T=500)
```
per_tick:
  chokepoint  = +0.003 if blocked_neighbors >= 2 and enemy_in_range
  funnel      = +0.002 if enemies_in_melee <= 1 and enemies_alive > 1
  time        = -0.002
on_kill:
  bonus       = +0.06 per kill
on_clear:
  bonus       = +0.2 if HP > 50%
```

### 3.11 Elevation Advantage (T=400)
```
per_tick:
  elevation   = +0.003 if self.elevation > nearest_enemy.elevation and enemy_in_range
  descend_pen = -0.01 if elevation decreased
  time        = -0.002
on_kill:
  bonus       = +0.1 per kill
on_clear:
  bonus       = +0.2 if never left platform
```

---

## Phase 4: Ability Usage

### 4.1 Use Heal on Low Ally (T=100)
```
per_tick:
  ally_hp_up  = +0.05 * max(0, ally_hp_pct[t] - ally_hp_pct[t-1])
  not_healing = -0.005 per tick heal ready and ally < 50%
  time        = -0.002
on_pass:
  bonus       = +0.3
```

### 4.2 Use CC on Enemy (T=200)
```
per_tick:
  # FIXED: only penalize holding stun when enemy is actually threatening ally
  stun_idle   = -0.003 per tick stun ready AND enemy_dist_to_ally < 3.0
  # No penalty when enemy far from ally (might be better to wait)
  threat_rise = +0.002 * clamp01(1.0 - enemy_dist_to_ally / 6.0) (threat growing)
  time        = -0.002
on_stun:
  bonus       = +0.15
  timing      = +0.1 * clamp01(1.0 - enemy_dist_to_ally / stun_range) (early = better)
on_ally_death:
  penalty     = -0.3
```
**Fix**: stun_idle penalty only when enemy is close to ally (< 3.0), not at any distance. Prevents learning "always stun immediately."

### 4.3a Interrupt Enemy Cast (T=120)
```
per_tick:
  # Urgency penalty for INACTION, not for correct action
  cast_inaction = -0.005 * cast_progress per tick stun ready and enemy casting and NOT attacking caster
  # If attacking caster or stunned it: no urgency penalty
  time          = -0.002
on_interrupt:
  bonus         = +0.4 * (1.0 - cast_progress_at_interrupt)
on_cast_completes:
  penalty       = -0.5
```
Teaches the mechanical skill of interrupting. See 4.3b for cast discrimination.

### 4.3b Selective Interrupt (T=300)
**Prerequisites**: 4.3a passed
```
Setup: Enemy casts two abilities in sequence:
  - Cast A: weak DoT (20 damage, 40-tick cast) starts at tick 0
  - Cast B: big nuke (90% max HP damage, 60-tick cast) starts at tick 80
  Unit has ONE stun (long cooldown, won't be ready for both).

per_tick:
  # Reward patience: don't stun the weak cast
  stun_on_weak     = -0.2 (used stun during cast A — wasted it)
  # Reward waiting for the dangerous cast
  stun_on_strong   = +0.3 (used stun during cast B — correct)
  cast_B_inaction  = -0.005 * cast_progress_B when stun ready and B casting and not stunning
  time             = -0.002
on_interrupt_B:
  bonus            = +0.3 * (1.0 - cast_progress_B)
on_B_completes:
  penalty          = -0.5
on_interrupt_A:
  penalty          = -0.15 (wasted stun on low-value target)

Runtime verification:
  - Assert stun was NOT used on cast A
  - Assert stun WAS used on cast B
  - Assert unit survived (B didn't land)
  - Validate: both casts happened, cast B was genuinely dangerous
  Log: stun_used_on_which_cast, cast_A_damage, cast_B_interrupted, hp_remaining
```
Teaches cast discrimination: don't burn your interrupt on trivial casts. This is the bridge
from 4.3a's "always interrupt" to Phase 6's selective ability usage.

### 4.4 CC Then Burst (T=400)
```
per_tick:
  dmg_during_cc = +0.008 per damage while target cc_remaining > 0
  dmg_normal    = +0.003 per damage otherwise
  time          = -0.002
on_stun:
  bonus         = +0.05
on_burst_during_cc:
  # Burst ability used within 20 ticks of stun — tight window bonus
  bonus         = +0.15
on_kill:
  bonus         = +0.2
```

### 4.5 Knockback Into Wall (T=300)
```
per_tick:
  # PRIMARY: wall alignment shaping (when ray-cast hits a wall)
  wall_alignment    = +0.004 * clamp01(1.0 - d_to_ideal_kb_pos / 5.0)

  # FALLBACK: when no wall is in knockback direction (ray-cast returns no hit),
  # reward proximity to enemy when walls are nearby the enemy
  wall_proximity_fb = +0.002 * walls_near_enemy * clamp01(1.0 - d(self, enemy) / kb_range)
  #   walls_near_enemy = count of walls within knockback_range of enemy position
  #   This gives gradient toward "get close to the enemy when walls are nearby"
  #   even before alignment is achieved

  in_range          = +0.001 per tick within knockback range
  time              = -0.002
on_knockback:
  base              = +0.1
on_wall_collision:
  bonus             = +0.3

d_to_ideal_kb_pos: ray-cast from self through enemy, find nearest wall intersection.
If no wall found: d_to_ideal_kb_pos = inf, wall_alignment = 0, fallback kicks in.
walls_near_enemy: number of blocked cells within knockback_range of enemy position.
```

### 4.6 AoE Positioning (T=300)
```
per_tick:
  cluster_prox  = +0.004 * gauss(d_to_cluster_center, ability_range * 0.5, 2.0)
  time          = -0.002
on_aoe_cast:
  per_hit       = +0.05 * enemies_hit
  perfect       = +0.15 if all 3 hit
```

### 4.7 Cooldown Management (T=1500)
```
per_tick:
  damage        = +0.002 per damage dealt
  time          = -0.001 (HALVED — endurance drill, ~1000 active ticks)
on_kill:
  weak          = +0.05 per weak kill (4 kills = +0.20)
  strong        = +0.15 for strong kill
on_ability_on_strong:
  bonus         = +0.15
on_ability_on_weak:
  penalty       = -0.05
on_death:
  penalty       = -0.3
on_clear:
  bonus         = +0.2
total ≈ damage(+1.0) + kills(+0.35) + ability_bonus(+0.15) + clear(+0.2) - time(-1.0) = +0.70 competent
random ≈ dies wave 2: damage(+0.3) + kills(+0.10) - time(-0.6) - death(-0.3) = -0.50
separation: 1.20 ✓
```

### 4.8 Respect Enemy Cooldowns (T=800)
```
per_tick:
  safe_engage   = +0.004 per tick attacking while enemy_cc_cd > 0.3
  unsafe_engage = -0.005 per tick in range while enemy_cc_cd < 0.1
  damage        = +0.002 per damage dealt
  time          = -0.001 (halved — long survival drill)
on_stunned:
  penalty       = -0.05
on_kill:
  bonus         = +0.3
total ≈ +0.25 competent, -0.2 random
separation: 0.45 ✓
```

---

## Phase 5: Team Coordination

**Per-unit rewards. Shared outcome: +0.05 per unit on win.**

### Coordination bootstrapping (all Phase 5 drills)
For the first 5 training iterations of each Phase 5 drill:
- Replace coordination-dependent rewards (e.g., `same_target`) with heuristic proxies
  that don't depend on the other unit's action
- Example: instead of "both attack same target," reward "attack lowest HP enemy"
  (a heuristic that naturally converges to focus fire)
- Anneal from 100% heuristic to 100% true coordination reward over iterations 1-10

### 5.1 Focus Fire (T=400)
```
per_tick (per unit):
  # Bootstrap (iters 1-5): attack lowest HP enemy
  # Anneal (iters 6-10): blend with true same_target
  # Final (iter 11+): true coordination reward
  same_target   = +0.004 if my_target == ally_target and both attacking
  diff_target   = -0.002 if different targets
  damage        = +0.001 per damage dealt
  time          = -0.002
on_kill:
  bonus         = +0.1 shared
on_ally_death:
  penalty       = -0.15 shared
```

### 5.2 Chain CC (T=800)
```
per_tick (per unit):
  dmg_during_cc = +0.004 per damage while target cc > 0 (INCREASED from v2)
  time          = -0.001 (halved — long coordination drill)
on_stun:
  bonus         = +0.08 (INCREASED from v2)
  # Continuous chain quality — partial credit for near-chains
  if ally_stun_ended_within_500ms:
    chain       = +0.25 * max(0, 1.0 - gap_ms / 300.0)
  overlap_pen   = -0.03 * clamp01(overlap_ms / 500.0)
on_kill:
  bonus         = +0.15
total ≈ +0.20 competent (chain lands, enemy dies), -0.25 random
separation: 0.45 ✓ (fixed from v2's -0.35 competent)
```
**Fixes**: increased per-tick and stun bonuses, halved time penalty, added kill bonus.

### 5.3 Peel for Carry (T=500)
```
per_tick (tank):
  body_block    = +0.003 if closer to enemy than ally_DPS
  cc_near_ally  = +0.01 per CC on enemy within 3 units of ally
  absorb        = +0.002 per damage taken
  time          = -0.002
per_tick (shared):
  ally_alive    = +0.001
on_kill:
  bonus         = +0.2 if ally survived
on_ally_death:
  penalty       = -0.3
```

### 5.4 Dive Coordination (T=600)
```
per_tick (per unit):
  atk_healer    = +0.004 attacking healer
  close_healer  = +0.002 closing to healer
  atk_dps       = +0.001 attacking DPS
  time          = -0.002
on_healer_kill:
  bonus         = +0.2 shared
on_ally_death:
  penalty       = -0.1
```

### 5.5 Engage/Disengage Timing (T=1000)
```
per_tick (per unit):
  safe_engage   = +0.003 in combat range when enemy CCs on cooldown
  unsafe_engage = -0.005 in combat range when enemy CCs ready
  damage        = +0.002 per damage
  time          = -0.001 (halved — long tactical drill)
on_stunned:
  penalty       = -0.03
on_win:
  bonus         = +0.3
on_lose:
  penalty       = -0.3
```

### 5.6 Horde Defense (T=800)
```
per_tick (tank):
  chokepoint    = +0.003 at choke with enemies approaching
  body_block    = +0.002 between enemies and DPS
per_tick (DPS):
  behind_tank   = +0.002 further from enemies than tank
  damage        = +0.003 per damage
per_tick (shared):
  ally_alive    = +0.001
  time          = -0.002
on_kill:
  bonus         = +0.03 per kill
on_clear:
  bonus         = +0.2 if both survive
```

### 5.7 Dynamic Terrain Coordination (T=600)
```
per_tick (engineer):
  wall_isolates = +0.02 per tick wall up AND healer pathing blocked
per_tick (DPS):
  atk_healer    = +0.004 attacking isolated healer
per_tick (shared):
  ally_alive    = +0.001
  time          = -0.002
on_healer_kill_walled:
  bonus         = +0.3
on_healer_kill_no_wall:
  bonus         = +0.1
```

---

## Reward Budget Verification (v3.1)

| Drill | Ticks | Per-tick sum | Bonuses | Time pen | **Comp.** | **Random** | **Sep.** |
|-------|-------|-------------|---------|----------|-----------|------------|----------|
| 1.1   | ~100  | +0.25       | +0.25   | -0.20    | **+0.30** | -0.20      | 0.50 ✓   |
| 2.1   | 500   | +1.00       | +0.30   | -0.50    | **+0.35** | -0.15      | 0.50 ✓   |
| 2.4   | ~300  | +0.45       | +0.30   | -0.30    | **+0.25** | -0.30      | 0.55 ✓   |
| 3.3   | ~100  | +0.25       | +0.30   | -0.20    | **+0.35** | -0.15      | 0.50 ✓   |
| 3.7   | ~150  | +0.40       | +0.30   | -0.30    | **+0.40** | -0.30      | 0.70 ✓   |
| 3.8   | ~500  | +0.60       | +0.30   | -0.70    | **+0.20** | -0.35      | 0.55 ✓   |
| 4.3b  | ~200  | +0.15       | +0.30   | -0.40    | **+0.25** | -0.35      | 0.60 ✓   |
| 4.5   | ~200  | +0.45       | +0.40   | -0.40    | **+0.45** | -0.20      | 0.65 ✓   |
| 4.7   | ~1000 | +1.00       | +0.70   | -1.00    | **+0.70** | -0.50      | 1.20 ✓   |
| 4.8   | ~500  | +0.60       | +0.30   | -0.50    | **+0.40** | -0.20      | 0.60 ✓   |
| 5.2   | ~400  | +0.40       | +0.33   | -0.40    | **+0.20** | -0.25      | 0.45 ✓   |
| 5.5   | ~600  | +0.50       | +0.30   | -0.60    | **+0.20** | -0.30      | 0.50 ✓   |

All drills ≥ 0.45 separation. Key fixes in v3.1:
- 4.7: -0.53 → **+0.70** (halved time, increased kill/clear bonuses)
- 4.5: +0.40 → **+0.45** (added wall_proximity fallback shaping)
- Added 4.3b (selective interrupt) to budget
- Gaussian σ now anneals continuously (no discontinuity at 50/100)
- Coordination bootstrapping tied to pass_rate, not iteration count
- Phase transition exploration bonus is one-shot per episode

