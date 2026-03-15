# Implementation Plan: Behavior DSL + Skill Curriculum + Sliding Window

Combined phased plan for shipping the full training pipeline.

## Overview

Four parallel workstreams, sequenced by dependency:

1. **Scenario TOML extensions** — hazards, objectives, scripted events, enemy behavior references
2. **Behavior DSL** — parser, interpreter, ability eval integration
3. **Sliding window encoder** — replace GRU with frame-stacked temporal attention
4. **Skill curriculum** — drill scenarios, reward shaping, orchestrator

```
Week 1:         [Scenario TOML]──────┐    [Sliding Window]
                [Behavior DSL]────┐  │
                                  │  │
Week 2:         [Drill behaviors] │  │    [Training loop update]
                                  ▼  ▼
Week 3:         [Phase 0: Feature verification]
                [Phase 1: Movement drills]
                [Phase 2: Spatial awareness drills]
                                  │
Week 4:         [Phase 3-5: Combat + coordination drills]
                [Phase 6: Full combat + self-play]
```

---

## Stream 1: Scenario TOML Extensions

### 1.1 Hazard definitions
Add to `ScenarioCfg` in `src/scenario/types.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HazardDef {
    pub hazard_type: String,       // "damage_zone", "slow_zone", "heal_zone"
    pub position: [f32; 2],
    pub radius: f32,
    pub damage_per_tick: f32,      // or heal_per_tick for heal zones
    pub team: String,              // "neutral", "hero", "enemy"
    pub start_tick: u64,           // when it appears (0 = immediately)
    pub duration: Option<u64>,     // None = permanent
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveDef {
    pub objective_type: String,    // "reach_position", "survive", "kill_target", "protect_ally"
    pub position: Option<[f32; 2]>,
    pub radius: Option<f32>,
    pub duration: Option<u64>,
    pub target_tag: Option<String>,
    pub max_damage_taken: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptedEventDef {
    pub trigger: String,           // "every N", "at_tick N", "when hp_below 0.5"
    pub action: String,            // "spawn_zone", "spawn_enemy", "set_objective"
    pub params: toml::Value,       // flexible params per action type
}

// Add to ScenarioCfg:
pub hazards: Vec<HazardDef>,
pub objectives: Vec<ObjectiveDef>,
pub events: Vec<ScriptedEventDef>,
```

### 1.2 Enemy unit behavior reference
Add to scenario TOML per-unit enemy config:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnemyUnitDef {
    pub template: Option<String>,  // hero template name
    pub behavior: Option<String>,  // behavior DSL file name (assets/behaviors/*.behavior)
    pub tag: Option<String>,       // named tag for behavior targeting
    pub position: Option<[f32; 2]>,
    pub hp_override: Option<i32>,
    pub dps_override: Option<f32>,
}

// Add to ScenarioCfg:
pub enemy_units: Vec<EnemyUnitDef>,  // replaces or augments enemy_hero_templates
```

### 1.3 Drill pass/fail evaluation
Add `DrillResult` to scenario runner:

```rust
pub struct DrillResult {
    pub passed: bool,
    pub metrics: HashMap<String, f64>,  // per-drill metrics for verification
    pub verification_failures: Vec<String>,
}
```

**Files changed:**
- `src/scenario/types.rs` — add structs, serde derive
- `src/scenario/runner.rs` — spawn hazards as ActiveZones, evaluate objectives
- `src/scenario/simulation.rs` — tick hazards, check objectives per tick

### 1.4 Hazard tick logic
In `src/ai/core/step.rs` or equivalent, after normal sim step:
- Iterate `scenario.hazards`, apply damage/slow/heal to units inside each zone
- Check scripted events trigger conditions, execute actions
- Check objectives, report pass/fail

**Estimated effort:** 2 days

---

## Stream 2: Behavior DSL

### 2.1 Grammar + Parser
New module: `src/ai/behavior/`

```
src/ai/behavior/
  mod.rs          — public API: load_behavior, evaluate_behavior
  parser.rs       — tokenizer + recursive descent parser
  types.rs        — AST types (BehaviorTree, Rule, Condition, Action, Target)
  interpreter.rs  — tick evaluation: BehaviorTree + SimState → IntentAction
  bridge.rs       — ability eval integration (use_best_ability, use_ability_type)
```

Parser input: `.behavior` files in `assets/behaviors/`
Parser output: `BehaviorTree` (vec of `Rule { priority: bool, condition: Option<Condition>, action: Action }`)

### 2.2 Interpreter
`fn evaluate(tree: &BehaviorTree, sim: &SimState, unit_id: u32, tick: u64) -> IntentAction`

Evaluation loop:
1. For each rule in priority order:
   a. Evaluate condition against sim state
   b. If condition met (or no condition), convert action to IntentAction
   c. Return first match
2. If no rule matches, return Hold

Action → IntentAction conversion:
- `chase(target)` → `IntentAction::MoveToward(resolve_target(target))`
- `attack(target)` → `IntentAction::Attack(resolve_target(target))`
- `cast ability0 on target` → `IntentAction::UseAbility(0, resolve_target(target))`
- `use_best_ability` → call `evaluate_abilities()` from ability_eval, return top result
- `maintain_distance(target, range)` → compute direction to maintain range, return MoveDir
- `flee(target)` → `IntentAction::MoveAway(resolve_target(target))`
- `hold` → `IntentAction::Hold`

Target resolution:
- `nearest_enemy` → find closest enemy unit by distance
- `lowest_hp_ally` → find ally with min hp_pct
- `highest_dps_enemy` → find enemy with max auto_dps
- `casting_enemy` → find enemy with is_casting == true
- `enemy_attacking target` → find enemy whose current target == resolved target
- `tagged "name"` → find unit with matching tag from scenario TOML

### 2.3 Ability eval bridge
`use_best_ability` calls existing `evaluate_abilities()`:
```rust
fn bridge_ability_eval(sim: &SimState, unit: &UnitState) -> Option<IntentAction> {
    let results = evaluate_abilities(sim, unit);
    results.iter()
        .filter(|r| r.urgency > 0.4)
        .max_by(|a, b| a.urgency.partial_cmp(&b.urgency).unwrap())
        .map(|r| IntentAction::UseAbility(r.ability_index, r.target_id))
}
```

`use_ability_type damage` filters results by category before picking.
`best_ability_urgency` returns the max urgency score as a float for conditions.

### 2.4 Integration with squad AI
In `src/ai/squad/intents.rs`, after generating default intents:
- If unit has a behavior assigned, evaluate behavior tree instead of default logic
- Behavior-generated IntentAction replaces the default intent for that unit

**Files changed:**
- New: `src/ai/behavior/` module (5 files, ~1500 lines total)
- Modified: `src/ai/squad/intents.rs` — check for behavior override
- Modified: `src/scenario/runner.rs` — load .behavior files, assign to units

**Estimated effort:** 3-4 days

---

## Stream 3: Sliding Window Encoder

Replace the GRU hidden state with a sliding window of K recent game states
processed by temporal attention. This is conceptually simpler (no hidden state
to propagate through SHM) and more expressive (attends over full recent history).

### 3.1 Architecture change

Remove `TemporalGRU`. Add `TemporalWindow`:

```python
class TemporalWindow(nn.Module):
    """Sliding window temporal encoder.

    Maintains a buffer of K recent pooled representations.
    Processes them with 1D self-attention to produce a
    temporally-enriched output.
    """
    def __init__(self, d_model: int, window_size: int = 8, n_heads: int = 4):
        super().__init__()
        self.window_size = window_size
        self.pos_emb = nn.Embedding(window_size, d_model)
        self.attn = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=0.0, norm_first=True, batch_first=True,
        )

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            window: (B, K, d_model) — K recent pooled states, newest last
        Returns:
            (B, d_model) — temporally enriched representation (last position output)
        """
        K = window.shape[1]
        pos = self.pos_emb(torch.arange(K, device=window.device))
        x = window + pos.unsqueeze(0)
        x = self.attn(x)
        return x[:, -1]  # take the most recent position's output
```

### 3.2 SHM protocol change

Instead of sending/receiving hidden state vectors, send the window buffer:

**Request**: append `K * d_model * 4` bytes = `8 * 32 * 4 = 1024` bytes of window history
**Response**: just the action (16 bytes, no hidden state to return)

This is simpler than the GRU approach — no hidden state output to propagate. The Rust
side maintains a circular buffer of recent pooled states per unit and sends the whole
window each tick.

But wait — the Rust side doesn't have access to `pooled` (that's computed in the Python model).
Two options:

**Option A**: Rust sends K recent raw game states (K × entity features). Python encodes all K,
then runs temporal attention. SHM request grows by K × entity_dim, which is large.

**Option B**: Python server maintains per-unit window buffers internally. Rust sends a unit_id
alongside each request. Server looks up the unit's recent pooled states, appends the new one,
runs temporal attention. Response is just actions (no state to return).

**Option C (recommended)**: Rust sends K recent game state snapshots in the request.
Python encodes them all in one batched call (entity encoder is already batched),
then runs temporal attention. The SHM per-sample size grows by `(K-1) * entity_features_size`
since the current frame is already sent. For K=8, that's 7 extra frames.

Revised sizes for Option C:
- Current sample: ~7144 bytes (with GRU hidden)
- Window sample: ~7144 - 256 (no hidden) + 7 * (entity + types + mask) ≈ 7144 + 7 * 2500 ≈ 24,644 bytes
- At 1024 batch: 24.6MB request region (vs current 7.1MB)
- GPU bandwidth at PCIe 4.0: ~25GB/s → 24.6MB takes ~1ms. Negligible vs inference time.

Actually this is too large. Let's reconsider.

**Option D (recommended)**: Rust maintains a per-unit circular buffer of the **raw entity
feature snapshots** (7 × 30 = 210 floats per frame). Sends the full window in the request.
But instead of sending ALL features for K frames, send only a **compressed summary** of
each historical frame:

Per historical frame, send only:
- 7 entity positions (7 × 2 = 14 floats)
- 7 entity HP values (7 floats)
- 7 entity casting states (7 floats)
- 7 entity CC remaining (7 floats)
Total per historical frame: 35 floats = 140 bytes

For K=8, history = 7 × 140 = 980 bytes extra. Total sample ≈ 7144 - 256 + 980 ≈ 7868 bytes.
Only 10% larger than current with GRU. This is the sweet spot.

**Option E (simplest, recommended)**: Keep the GRU but switch training from step-by-step
to the batched approach we already built. The GRU is already working in the SHM protocol.
Training with batched trajectory processing (encoding all steps, then GRU loop over time
dimension with batch parallelism) is already ~100x faster than step-by-step.

The GRU approach:
- ✅ Already implemented in SHM protocol (both Rust and Python)
- ✅ Already compiles and runs
- ✅ Batched training already works
- ✅ Minimal SHM overhead (+512 bytes vs +980 or more)
- ✅ cuDNN GRU for training is fast
- ❌ Slightly less interpretable than sliding window
- ❌ Information bottleneck (64-dim hidden vs K × d_model)

**Decision: Keep GRU for now.** The sliding window is architecturally nicer but the GRU
is already working end-to-end. We can swap later if the GRU bottlenecks temporal learning.
The batched training approach makes GRU training fast enough.

### 3.3 Training loop (already done)
The batched trajectory processing in `impala_learner.py` already:
- Batch-encodes all steps (no GRU, fast)
- Groups trajectories into mini-batches
- Runs GRU sequentially over time dimension with batch parallelism
- Truncated BPTT every 32 steps

No further changes needed for this stream.

**Estimated effort:** 0 days (already done, keeping GRU)

---

## Stream 4: Skill Curriculum

### 4.1 Drill scenario generation
Script: `training/generate_drills.py`

Generates drill scenario TOMLs for each phase:
- Phase 1: movement drills (target positions, obstacle layouts)
- Phase 2: spatial awareness (enemy behaviors, danger zones)
- Phase 3: target selection (enemy configurations, ally setups)
- Phase 4: ability usage (specific ability kits, timing scenarios)
- Phase 5: team coordination (multi-unit setups)

Each drill type generates 100+ randomized variants (different seeds, positions, layouts).

### 4.2 Drill behavior files
Create `assets/behaviors/` directory with drill-specific behaviors:

```
assets/behaviors/
  # Training dummies
  stationary_dummy.behavior
  fleeing_target.behavior
  slow_patrol.behavior

  # Phase 2 enemies
  melee_chaser.behavior
  aoe_caster_periodic.behavior

  # Phase 3 enemies
  healer_bot.behavior
  tank_guard.behavior
  aggressive_dps.behavior
  cc_gatekeeper.behavior

  # Phase 5 enemy pairs
  dive_pair_tank.behavior
  dive_pair_dps.behavior

  # Reusable game NPCs
  default_fighter.behavior
  default_healer.behavior
  default_tank.behavior
  boss_flame_lord.behavior
```

### 4.3 Feature verification suite
Script: `training/verify_features.py`

Runs Phase 0 checks (all 14 diagnostic tests from curriculum doc):
- Spawns known sim states
- Reads entity features
- Asserts values match expected
- Reports pass/fail with expected vs actual

Must pass before any training starts.

### 4.4 Curriculum orchestrator
Script: `training/curriculum.py` (already started, needs update)

Updated to:
- Run Phase 0 verification before training
- Generate drill scenarios per phase
- Train with per-phase reward shaping
- Evaluate 100/100 pass gate with runtime verification
- Regression test previously passed phases
- Gate action heads per phase (move only → attack → abilities)
- Log structured metrics to dashboard

### 4.5 Runtime verification framework
Module: `training/drill_verification.py`

Per-drill verification functions:
```python
def verify_1_1_reach_static(episode, drill_config) -> VerificationResult:
    """Check that unit actually navigated to target, didn't circle, etc."""
    steps = episode["steps"]
    final_dist = distance(steps[-1].position, drill_config.target)
    path_length = sum(distance(s[i].pos, s[i+1].pos) for i in range(len(s)-1))
    euclidean = distance(steps[0].position, drill_config.target)

    checks = {
        "reached_target": final_dist < 1.0,
        "path_efficient": path_length < 3 * euclidean,
        "not_stuck": len(set(s.position for s in steps)) > 5,
    }
    return VerificationResult(
        passed=all(checks.values()),
        checks=checks,
        metrics={"final_dist": final_dist, "path_ratio": path_length / euclidean},
    )
```

**Estimated effort:** 4-5 days

---

## Execution Order

### Day 1-2: Foundation
- [ ] Scenario TOML extensions (hazards, objectives, enemy_units)
- [ ] Behavior DSL types + parser (tokenizer, AST)
- [ ] Phase 0 feature verification script

### Day 3-4: Behavior System
- [ ] Behavior DSL interpreter
- [ ] Ability eval bridge
- [ ] Squad AI integration (behavior override per unit)
- [ ] Write drill behavior files (stationary_dummy through cc_gatekeeper)

### Day 5: Drill Generation
- [ ] Drill scenario generator (all phases)
- [ ] Runtime verification framework
- [ ] Test: spawn each drill type, run with random policy, verify metrics collected

### Day 6-7: Training Pipeline
- [ ] Curriculum orchestrator (updated with drill support)
- [ ] Per-phase reward shaping
- [ ] Action head gating (move-only through Phase 2)
- [ ] Regression testing framework
- [ ] Dashboard integration for drill metrics

### Day 8+: Training
- [ ] Run Phase 0 verification → fix any feature bugs found
- [ ] Train Phase 1 (movement fundamentals)
- [ ] Train Phase 2 (spatial awareness)
- [ ] Train Phase 3 (target selection)
- [ ] Train Phase 4-5 (abilities + coordination)
- [ ] Train Phase 6 (full combat + self-play)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Behavior DSL too complex to implement | Start with subset: chase, flee, attack, hold, cast, use_best_ability. Add composites later. |
| Drill scenarios don't isolate skills cleanly | Phase 0 verification catches feature bugs. Manual playtesting of each drill before training. |
| GRU training too slow despite batching | Already ~60s/iter with batched approach. Acceptable for curriculum (fewer scenarios per phase). |
| Model forgets earlier phases during later training | Regression testing after each phase. Mixed replay from earlier phases if regression detected. |
| Ability eval integration doesn't work with behavior DSL | Test bridge independently: spawn known scenario, call eval, verify result matches expected ability. |
