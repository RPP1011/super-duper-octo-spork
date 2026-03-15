# AbilityActorCriticV5 Migration Plan

## Context

Current model (V4): 134K params, d=32, 4 heads, h_dim=64. Training time ~30min.
Target model (V5): ~1.5M params, d=128, 8 heads, h_dim=256. Estimated training time ~2-3hr.

The GPU (RTX 4090) is massively underutilized at ~5-10% during episode generation.
The bottleneck is SHM round-trip and Rust sim tick rate, not GPU compute.
The model can be 10-20x larger before GPU becomes the bottleneck.

Key motivation: at d=32 with 4 heads, each head has only 8 dimensions — too narrow
for attention to represent meaningful tactical relationships. d=128 / 8 heads = 16d/head
is where transformer attention becomes genuinely expressive.

---

## Scene Representation: Information Capacity as a Design Knob

### The Problem

Entity/threat slot truncation is a **data-level** bottleneck, not a model-level one.
Architectural improvements (d=128, latent interface, CfC) only improve how the model
reasons about what it sees — they cannot recover information discarded before the model
runs. For a 12-enemy chokepoint push with 7 entity slots, the model sees 3 enemies and
has no signal that 9 more exist. "Mass push" and "3 isolated scouts" look identical.

This is distinct from capacity limits inside the model (which are intentional — see below).

### Design Philosophy: Intentional Capacity Limits

The slot limit is not purely a bug. A model that always has complete information never
learns to operate under uncertainty. The goal is:

- **Identified individual slots** for tactically significant units (full detail)
- **Aggregate summary** for the unrepresented crowd (meta-awareness without full info)
- **Importance-based selection** in Rust to decide which units earn individual slots

This means the model knows "12 enemies exist, centroid is northeast, aggregate threat
is high" without knowing each individual's details. It learns to condition on that
uncertainty. This is more realistic and more robust than full-information policies,
and it enables a natural scaling axis: allocate more powerful units more individual
compute by giving them guaranteed slots.

### Aggregate Summary Token

Add 1 aggregate token per tick summarizing all entities that did not earn individual slots.
Projects to 128d via `agg_proj(N_agg_features → 128)`, appended to the token sequence.
The latent interface naturally routes attention to it when the crowd signal matters.

**Proposed feature vector (~16 features):**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `n_enemies_total` | Total enemies in scene (including truncated) /20 |
| 1 | `n_allies_total` | Total allies in scene (including truncated) /10 |
| 2 | `n_enemies_truncated` | How many enemies didn't get a slot /15 |
| 3 | `n_allies_truncated` | How many allies didn't get a slot /8 |
| 4 | `enemy_centroid_x` | Mean X of all enemies /20 |
| 5 | `enemy_centroid_y` | Mean Y of all enemies /20 |
| 6 | `ally_centroid_x` | Mean X of all allies /20 |
| 7 | `ally_centroid_y` | Mean Y of all allies /20 |
| 8 | `mean_enemy_hp_pct` | Mean HP fraction of all enemies |
| 9 | `min_enemy_hp_pct` | Lowest HP enemy (focus fire signal) |
| 10 | `max_enemy_threat` | Highest individual threat score /1.0 |
| 11 | `aggregate_enemy_dps` | Sum of auto DPS of truncated enemies /200 |
| 12 | `n_projectiles_total` | Total projectiles in flight (including truncated threats) /10 |
| 13 | `enemy_spread` | Std dev of enemy positions (tight=push, wide=spread) /10 |
| 14 | `dominant_enemy_type` | One-hot or ordinal: melee/ranged/caster/mixed |
| 15 | `aggregate_cc_threat` | Sum of CC durations of truncated enemies /5000 |

**Type ID:** `5` (aggregate) — new type embedding added to encoder.

**SHM addition:** 16 × f32 = 64 bytes appended to request. Small.

**Token sequence with aggregate:** N = 20 tokens (7 entity + 4 threat + 8 position + 1 aggregate).

### Importance-Based Slot Selection (Rust Side)

The current selection heuristic (presumably nearest/highest threat) is replaced with a
priority scoring system. Units are scored and the top-K earn individual entity slots.
The rest contribute only to the aggregate token.

**Priority score components:**

```rust
fn unit_priority_score(unit: &Unit, context: &BattleContext) -> f32 {
    let mut score = 0.0;

    // Guaranteed slots — always earn individual representation
    if unit.is_boss || unit.tier >= UnitTier::Elite {
        return f32::MAX;  // always gets a slot
    }

    // Threat to self
    score += unit.auto_dps / 30.0;
    score += unit.ability_damage / 50.0;
    score += if unit.has_ready_cc() { 0.3 } else { 0.0 };

    // Proximity (closer = higher priority, but not the only signal)
    let dist = unit.distance_from_caster;
    score += (1.0 - (dist / 20.0).min(1.0)) * 0.4;

    // Low HP enemies are high priority (focus fire opportunity)
    if unit.is_enemy && unit.hp_pct < 0.25 {
        score += 0.5;
    }

    // Allies in danger are high priority
    if unit.is_ally && unit.hp_pct < 0.3 {
        score += 0.4;
    }

    // Active casts are high priority (dodge/interrupt signal)
    if unit.is_casting {
        score += 0.6;
    }

    score
}
```

**Slot allocation order:**
1. Self (always slot 0)
2. Guaranteed units (boss/elite tier) — consume slots in score order
3. Remaining slots filled by score descending
4. All remaining units → aggregate token only

**Rust implementation location:** `src/ai/core/ability_eval/game_state.rs` — modify
entity serialization to score and sort before slot assignment, compute aggregate stats
over the truncated remainder.

### Future Direction: Per-Unit Compute Allocation

The natural extension of importance-based selection is importance-based compute
allocation inside the model. A boss enemy's entity token could pass through more
latent interface blocks before the Write step — effectively giving it more
representational depth. This requires a variable-depth forward pass and is nontrivial
to implement cleanly in PyTorch (dynamic computation graph per token), but is the
correct long-term direction. The priority scoring system in Rust is the prerequisite
— once units have explicit importance scores, routing them to different compute paths
is architecturally natural.

### SHM Changes for Aggregate Token

```
aggregate_features:  16 × f32 = 64 bytes  (new, appended to entity block)
sample_size:         +64 bytes
```

New token count in header or derived from existing counts — aggregate is always
present (1 token always), so no mask needed. Zero-fill if scene has ≤ slot count
entities (nothing was truncated).

### Files to Modify

| File | Change |
|------|--------|
| `src/ai/core/ability_eval/game_state.rs` | Priority scoring, slot assignment, aggregate computation |
| `training/gpu_inference_server.py` | Parse aggregate_features from SHM |
| `training/model.py` | `agg_proj(16→128)`, type_id=5 embedding, N=20 tokens |
| `src/ai/core/ability_transformer/gpu_client.rs` | Write aggregate block to SHM |

---

## Target Architecture (V5)

```
Input:
  7 entity slots × 30 features      (importance-selected, see Scene Representation)
  4 threat tokens × 8 features
  8 position tokens × 8 features
  1 aggregate token × 16 features   ← new: crowd summary for truncated entities
  8 ability CLS embeddings × 128d   ← no longer projected down

                    ┌──────────────────────────────────┐
                    │       Entity Encoder V5           │
                    │   d=128, 8 heads, 4 layers        │
                    │   pre-norm, same structure as V4  │
                    │                                   │
  Entities ────────►│   entity_proj(30→128)             │
  Threats ─────────►│   threat_proj(8→128)              │
  Positions ───────►│   position_proj(8→128)            │
  Aggregate ───────►│   agg_proj(16→128)                │
                    │   + type embeddings (6 types)     │
                    │   → TransformerEncoder(4 layers)  │
                    └────────────┬─────────────────────┘
                                 │
                         tokens (B, 20, 128)
                                 │
                    ┌────────────▼─────────────────────┐
                    │     Latent Interface (ELIT-style) │
                    │                                   │
                    │   K=12 learned latent tokens      │
                    │   Read: latents attend to tokens  │
                    │   2× latent transformer blocks    │
                    │   Write: tokens updated by latents│
                    │                                   │
                    │   → pool latents → (B, 128)       │
                    └────────────┬─────────────────────┘
                                 │
                         pooled (B, 128)
                                 │
                    ┌────────────▼─────────────────────┐
                    │       CfC Temporal Cell           │
                    │   input=128, hidden=256           │
                    │   replaces GRUCell                │
                    │   → proj(256→128)                 │
                    └────────────┬─────────────────────┘
                                 │
                         pooled_enriched (B, 128)
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
    Move Head              Combat Pointer            Value Head
    d=128 → 9              d=128, pointer             d=128 → 1
                           over 128d entity tokens
```

**Ability CLS embeddings:** 128d matches d_model — `external_cls_proj` (Linear 128→32)
is removed entirely. Ability embeddings feed directly into cross-attention. This removes
a lossy bottleneck that was discarding 75% of the ability representation capacity.

---

## Parameter Estimates

| Component | V4 | V5 (estimated) |
|-----------|-----|----------------|
| Entity Encoder | 46,848 | ~750,000 |
| Aggregate Proj | — | ~2,000 |
| Latent Interface (K=12) | — | ~200,000 |
| CfC Cell | — | ~350,000 |
| GRU | 20,896 | removed |
| External CLS Proj | 4,224 | removed |
| Cross-Attention (ability) | 6,560 | ~50,000 |
| Move Head | 1,097 | ~17,000 |
| Combat Pointer Head | 22,122 | ~200,000 |
| Value Head | 1,089 | ~17,000 |
| **Total** | **134,548** | **~1,586,000** |

---

## Phase Plan

### Phase 1 — Scale d: 32 → 128, remove CLS bottleneck, add aggregate token
**Changes:**
- All projections: 30→128, 8→128
- All transformer blocks: d=32→128, 4 heads→8 heads
- Remove `external_cls_proj: Linear(128→32)` — ability CLS feeds directly at 128d
- Add `agg_proj: Linear(16→128)` — aggregate token projection
- Add type_id=5 embedding for aggregate token
- GRU input/output updated: GRUCell(128→64), proj Linear(64→128)
- All heads: updated to d=128
- Rust: priority-based slot selection + aggregate computation (see Scene Representation)
- SHM: +64 bytes for aggregate_features block

**Training notes:**
- Train from scratch — no weight transfer possible from V4
- Lower LR: if V4 used 1e-3, target ~5e-4 (LR ∝ 1/√d_model)
- Watch combat head early: ability CLS now feeds raw 128d instead of projected 32d.
  Expect slightly noisier pointer head for first few thousand steps during calibration.
- Profile training step time at d=128 — may be able to increase batch size further

**Validation:**
- Confirm policy quality clearly exceeds V4 within equivalent wall-clock time
- Confirm attention weights are non-uniform (heads are specializing)

---

### Phase 2 — Add Latent Interface

**Changes (Python only, no SHM/Rust changes):**

Insert between entity encoder output and pool→CfC step.

```python
class LatentInterface(nn.Module):
    def __init__(self, d_model=128, n_latents=12, n_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros(n_latents, d_model))
        nn.init.normal_(self.latents, std=0.02)

        self.read = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.write = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.latent_block_1 = TransformerBlock(d_model, n_heads)
        self.latent_block_2 = TransformerBlock(d_model, n_heads)

        self.read_norm = nn.LayerNorm(d_model)
        self.write_norm = nn.LayerNorm(d_model)

        # Zero-init write output proj → identity at init, trains from there
        nn.init.zeros_(self.write.out_proj.weight)
        nn.init.zeros_(self.write.out_proj.bias)

    def forward(self, entity_tokens):  # (B, 20, 128)
        B = entity_tokens.shape[0]
        L = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, 12, 128)

        # Read: latents attend to entity tokens
        L, _ = self.read(L, entity_tokens, entity_tokens)
        L = self.read_norm(L)
        L = self.latent_block_1(L)
        L = self.latent_block_2(L)

        # Write: entity tokens updated by latents
        updated, _ = self.write(entity_tokens, L, L)
        entity_tokens = self.write_norm(entity_tokens + updated)

        # Pool latents for temporal cell input (not entity tokens)
        pooled = L.mean(dim=1)  # (B, 128)
        return entity_tokens, pooled
```

**Why pool latents (not entity tokens):** The latent tokens are a coordinated tactical
summary — they've already compressed the important cross-entity information. Entity tokens
still go to the pointer heads for targeting (they need per-entity resolution).

**Why K=12:** 20 input tokens → 12 latents is a moderate compression ratio. At d=128
the latent self-attention is cheap and more slots allow more specialized tactical
representations to emerge. ELIT found diminishing returns at high K — 12 is a reasonable
starting point to tune from.

**Validation:**
- Inspect Read attention weights: should be non-uniform and interpretable
  (e.g. one latent attending heavily to low-HP allies, another to nearby enemies)
- If Read attention is flat/uniform after several thousand steps, increase K or
  add a diversity regularization term

---

### Phase 3 — Replace GRU with CfC, scale h_dim: 64 → 256

**Changes:**
- Python: swap GRUCell for CfCCell
- SHM: h_dim 64→256, hidden_state_in/out 256→1024 bytes
- Rust: update h_dim in header parsing, update SHM buffer sizes

```python
class CfCCell(nn.Module):
    def __init__(self, input_size=128, hidden_size=256):
        super().__init__()
        total = input_size + hidden_size
        self.f_gate = nn.Linear(total, hidden_size)
        self.h_gate = nn.Linear(total, hidden_size)
        self.t_a    = nn.Linear(total, hidden_size)
        self.t_b    = nn.Linear(total, hidden_size)
        self.proj   = nn.Linear(hidden_size, input_size)

        # Init: start near full memory retention, let it learn to forget
        nn.init.constant_(self.f_gate.bias, 1.0)
        nn.init.constant_(self.t_b.bias, 1.0)

    def forward(self, x, h, delta_t=1.0):
        combined = torch.cat([x, h], dim=-1)
        f = torch.sigmoid(self.f_gate(combined))
        candidate = torch.tanh(self.h_gate(combined))
        t = torch.sigmoid(self.t_a(combined)) * delta_t + self.t_b(combined)
        h_new = torch.tanh(f * h + (1 - f) * candidate * t)
        return self.proj(h_new), h_new
```

**SHM layout changes:**
```
hidden_state_in:  256 × f32 = 1024 bytes  (was 64 × f32 = 256 bytes)
hidden_state_out: 256 × f32 = 1024 bytes  (was 64 × f32 = 256 bytes)
sample_size:      increases by 1536 bytes
h_dim header:     256 (was 64)
```

**Rust changes:**
- Update h_dim constant / header read
- Update SHM buffer allocation for hidden_state_in/out
- `transformer_rl.rs`: per-unit hidden state buffer grows from 256→1024 bytes

**delta_t:** Keep at 1.0 for now. The input-dependent time constant is the main
immediate benefit — true delta_t scheduling is Phase 4.

---

### Phase 4 — delta_t via SHM (optional, post-validation)

Add 1 × f32 to request for ticks-since-last-meaningful-event.

**Meaningful event definition (Rust side):**
- Damage taken or dealt above threshold
- Ability fired
- CC applied or received
- Unit death in scene

**SHM change:** +4 bytes to request. Append after `hidden_state_in` or use a
dedicated field in the header region.

**Payoff:** A unit that has been kiting for 20 ticks sends delta_t=20. The CfC
cell naturally decays short-term tactical memory, reducing the signal from stale
positional history. Tactically relevant for: tracking cooldown cycles, remembering
ability timing, recognizing re-engagement vs. sustained engagement.

---

## Risk / Rollback Summary

| Phase | Rust changes | SHM changes | Rollback path |
|-------|-------------|-------------|---------------|
| 1 — Scale d=128 + aggregate token | Slot scoring, aggregate serialization | +64 bytes aggregate block | Revert model.py + game_state.rs, retrain |
| 2 — Latent Interface | None | None | Zero-init write = identity passthrough |
| 3 — CfC + h_dim=256 | Buffer sizes, h_dim header | +1536 bytes | Swap CfCCell → GRUCell, revert h_dim |
| 4 — delta_t | +4 bytes request, event tracking | +4 bytes | Hardcode delta_t=1.0 |

---

## Open Questions / Decisions Deferred

- **K (latent count):** Starting at 12, tune based on Read attention diversity
- **n_latent_blocks:** Starting at 2, could increase if GPU headroom confirms
- **Pointer head LR:** May need separate lower LR or delayed unfreeze if it collapses
  early due to raw 128d CLS input (vs. previously projected 32d)
- **Batch size at d=128:** Profile training step time — may be able to increase
  beyond 1024 for better gradient estimates given the underutilized GPU
- **Whether to do Phases 1+2 together or sequentially:** Phase 1 alone is a big
  jump; adding the latent interface simultaneously is low risk since write is
  zero-initialized, but harder to attribute any regression

---

## Key Files to Modify

| File | Phase | Change |
|------|-------|--------|
| `src/ai/core/ability_eval/game_state.rs` | 1 | Priority scoring, slot assignment, aggregate token serialization |
| `training/model.py` | 1 | d=128, 8 heads, remove CLS proj, add agg_proj + type_id=5 |
| `training/model.py` | 2 | Add LatentInterface class, wire into V5 |
| `training/model.py` | 3 | Add CfCCell, replace TemporalGRU |
| `training/gpu_inference_server.py` | 1 | Parse aggregate_features from SHM |
| `training/gpu_inference_server.py` | 3 | h_dim=256, buffer sizes |
| `src/ai/core/ability_transformer/gpu_client.rs` | 1 | Write aggregate block to SHM |
| `src/ai/core/ability_transformer/gpu_client.rs` | 3 | h_dim, SHM buffer sizes |
| `src/bin/xtask/oracle_cmd/transformer_rl.rs` | 3 | Per-unit hidden state buffer |
| `training/export_actor_critic_v4.py` | 1 | Clone → export_actor_critic_v5.py |
| `src/ai/core/ability_transformer/weights.rs` | 1 | New weight layout for V5 |
