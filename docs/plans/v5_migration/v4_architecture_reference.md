# Model Architecture: AbilityActorCriticV4

**134,548 parameters** (113K base + 21K GRU temporal context)

```
Input:
  7 entity slots × 30 features    (vitals, position, combat, abilities, CC, state)
  4 threat tokens × 8 features    (incoming projectiles, enemy casts)
  8 position tokens × 8 features  (nearby cover spots, elevation, chokepoints)
  8 ability CLS embeddings × 128d (pretrained behavioral embeddings from registry)

                    ┌──────────────────────────────────┐
                    │       Entity Encoder V3           │
                    │   self-attention, 4 layers        │
                    │   d=32, 4 heads, pre-norm         │
                    │                                   │
  Entities ────────►│   Entities → entity_proj(30→32)   │
  Threats ─────────►│   Threats  → threat_proj(8→32)    │
  Positions ───────►│   Positions→ position_proj(8→32)  │
                    │   + type embeddings (5 types)     │
                    │   → TransformerEncoder(4 layers)  │
                    └────────────┬──────────────────────┘
                                 │
                         tokens (B, N, 32)
                         pooled = mean(tokens)
                                 │
                    ┌────────────▼──────────────────────┐
                    │       Temporal GRU (h=64)          │
                    │   GRUCell(32→64) + Linear(64→32)  │
                    │   carries hidden state across ticks│
                    └────────────┬──────────────────────┘
                                 │
                         pooled_enriched (B, 32)
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
  ┌──────────────┐   ┌───────────────────────┐   ┌──────────────┐
  │  Move Head   │   │  Combat Pointer Head  │   │  Value Head  │
  │  Linear→GELU │   │                       │   │  Linear→GELU │
  │  →Linear(9)  │   │  combat_type: MLP→10  │   │  →Linear(1)  │
  │              │   │  attack_ptr: Q·Kᵀ/√d  │   │  (train only)│
  │  9-way dir:  │   │  ability_ptrs: Q·Kᵀ/√d│   │              │
  │  N,NE,E,SE,  │   │  (per ability via     │   │  V(s) scalar │
  │  S,SW,W,NW,  │   │   cross-attention)    │   │              │
  │  stay        │   │                       │   │              │
  └──────────────┘   └───────────────────────┘   └──────────────┘

Cross-Attention (runs per ability):
  Ability CLS (128d) → external_cls_proj(128→32) → Query
  Entity tokens (N×32) → Key/Value
  → MultiheadAttention(4 heads) → ability_cross_emb (32d)
  → Used as Query in pointer head for ability targeting
```

---

## Entity Features (30 per entity, 7 slots)

Entity order: `[self, enemy0, enemy1, enemy2, ally0, ally1, ally2]`
Type IDs: `0=self, 1=enemy, 2=ally`

| Index | Feature | Normalization | Description |
|-------|---------|--------------|-------------|
| **Vitals** | | | |
| 0 | `hp_pct` | [0, 1] | Current HP / max HP |
| 1 | `shield_pct` | [0, 1] | Shield HP / max HP |
| 2 | `resource_pct` | [0, 1] | Resource / max resource |
| 3 | `armor` | /200 | Physical damage reduction |
| 4 | `magic_resist` | /200 | Magic damage reduction |
| **Position / Terrain** | | | |
| 5 | `position_x` | /20 | Absolute X position |
| 6 | `position_y` | /20 | Absolute Y position |
| 7 | `distance_from_caster` | /10 | Distance to acting unit (0 for self) |
| 8 | `cover_bonus` | [0, 1] | Terrain cover at this position |
| 9 | `elevation` | /5 | Height of terrain |
| 10 | `n_hostile_zones` | /3 | Nearby enemy damage zones |
| 11 | `n_friendly_zones` | /3 | Nearby ally zones |
| **Combat Stats** | | | |
| 12 | `auto_dps` | /30 | Auto-attack damage per second |
| 13 | `attack_range` | /10 | Auto-attack range in world units |
| 14 | `attack_cd_pct` | [0, 1] | Attack cooldown remaining (0=ready) |
| **Strongest Ability** | | | |
| 15 | `ability_damage` | /50 | Highest-damage ability's damage |
| 16 | `ability_range` | /10 | That ability's range |
| 17 | `ability_cd_pct` | [0, 1] | That ability's cooldown (0=ready) |
| **Healing** | | | |
| 18 | `heal_amount` | /50 | Strongest heal ability's heal amount |
| 19 | `heal_range` | /10 | That ability's range |
| 20 | `heal_cd_pct` | [0, 1] | That ability's cooldown |
| **Crowd Control** | | | |
| 21 | `control_range` | /10 | CC ability's range |
| 22 | `control_duration` | /2000 | CC duration in ms |
| 23 | `control_cd_pct` | [0, 1] | CC ability's cooldown |
| **Current State** | | | |
| 24 | `is_casting` | 0/1 | Currently channeling an ability |
| 25 | `cast_progress` | [0, 1] | How far through current cast |
| 26 | `cc_remaining` | /2000 | Time until CC wears off |
| 27 | `move_speed` | /5 | Movement speed |
| **Cumulative** | | | |
| 28 | `total_damage_done` | /1000 | Cumulative damage this fight |
| 29 | `exists` | 0/1 | 1.0 if entity slot occupied, 0.0 if padding |

**Source**: `src/ai/core/ability_eval/game_state.rs:15-54`

---

## Threat Tokens (8 features, up to 4 tokens)

Incoming threats: projectiles, enemy casts, danger zones targeting this unit.

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `impact_x` | Relative X of impact point |
| 1 | `impact_y` | Relative Y of impact point |
| 2 | `radius` | AoE radius (0 for single-target) |
| 3 | `time_to_impact` | Ticks until damage lands |
| 4 | `damage_estimate` | Expected damage |
| 5 | `is_projectile` | 1 if projectile, 0 if cast/zone |
| 6 | `source_distance` | Distance to source unit |
| 7 | `can_dodge` | 1 if movable before impact |

Type ID: `3` (threat)

**Source**: `src/ai/core/ability_eval/game_state.rs:342-470`

---

## Position Tokens (8 features, up to 8 tokens)

Candidate positions sampled in 8 directions at 2-3 distances around the unit.
Represent nearby cover spots, elevated positions, chokepoints, retreat paths.

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `dx` | Relative X from self (/20) |
| 1 | `dy` | Relative Y from self (/20) |
| 2 | `path_distance` | Nav grid path distance (/30) |
| 3 | `elevation` | Terrain height (/5) |
| 4 | `blocked_neighbors` | Number of blocked adjacent cells (/3). ≥2 = chokepoint |
| 5 | `wall_proximity` | Distance to nearest wall (/5) |
| 6 | `hostile_zones` | Enemy zones near this position (/3) |
| 7 | `friendly_zones` | Ally zones near this position (/3) |

Type ID: `4` (position)

**Source**: `src/ai/core/ability_eval/game_state.rs:242-340`

---

## Ability CLS Embeddings (128d, up to 8 per unit)

Pretrained behavioral embeddings from the ability transformer (2.8M params, frozen).
Each ability's DSL text is tokenized and run through a 4-layer transformer encoder.
The [CLS] token output is the 128d embedding stored in the registry.

- **Registry**: `generated/ability_embedding_registry.json` (943 abilities × 128d)
- **Lookup**: Rust matches ability names at episode start, sends via SHM
- **Projection**: `external_cls_proj: Linear(128→32)` maps to model's d_model

**Source**: `training/export_embedding_registry.py`, `training/pretrain.py`

---

## GRU Temporal Context

Hidden state (64d) persists across ticks within an episode.
Enables in-context opponent inference: tracking enemy behavior patterns,
cooldown cycles, ally ability timing.

| Component | Shape | Description |
|-----------|-------|-------------|
| `GRUCell` | input=32, hidden=64 | Core recurrent cell |
| `proj` | Linear(64→32) | Project back to d_model for heads |
| `h_prev` | (B, 64) | Previous tick's hidden state (zeros on first tick) |
| `h_new` | (B, 64) | Output hidden state (propagated to next tick) |

**SHM protocol**: hidden_in (256 bytes) appended to request, hidden_out (256 bytes) appended to response.

**Training**: Batched trajectory unrolling — encode all steps, run GRU sequentially over time dimension with batch parallelism across trajectories. Truncated BPTT every 32 steps.

**Source**: `training/model.py:1339` (TemporalGRU), `training/gpu_inference_server.py:86` (SHM)

---

## Action Space

Dual-head output, combined by the sim each tick:

### Move Head
9-way categorical: N, NE, E, SE, S, SW, W, NW, Stay

```python
move_logits = MLP(pooled) → (B, 9)
move_dir = sample(softmax(move_logits / temperature))
```

### Combat Head
Two-stage: action type selection + pointer target selection

**Stage 1: Combat Type** (10-way categorical)
```
combat_logits = MLP(pooled) → (B, 10)
combat_type = sample(softmax(masked_logits / temperature))
```
Types: `0=attack, 1=hold, 2-9=ability_0..7`
Masked by `combat_mask[10]`: abilities only available when off cooldown and in range.

**Stage 2: Target Pointer** (scaled dot-product over entities)
```
keys = pointer_key(entity_tokens)        → (B, N, 32)

# For attack (type=0):
  query = attack_query(pooled)           → (B, 1, 32)
  logits = (query @ keys.T) * 1/√32     → (B, N)
  masked to enemies only (type_id == 1)

# For ability i (type=2+i):
  query = ability_queries[i](cross_attn_emb[i])  → (B, 1, 32)
  logits = (query @ keys.T) * 1/√32              → (B, N)
  masked to valid targets for this ability
```

**Source**: `training/model.py:1238` (CombatPointerHead)

---

## SHM Inference Protocol

Per-sample request layout (7144 bytes with h_dim=64):
```
[0..8]      counts: 4 × u16 (n_entities, n_threats, n_positions, padding)
[8..2408]   entity_features: 20 × 30 × f32
[2408..2488] entity_types: 20 × i32
[2488..2508] entity_mask: 20 × u8 (padded to 4)
[2508..2636] threat_features: 4 × 8 × f32
[2636..2640] threat_mask: 4 × u8
[2640..2768] position_features: 4 × 8 × f32
[2768..2772] position_mask: 4 × u8
[2772..2784] combat_mask: 10 × u8 + 2 padding
[2784..2792] ability_has: 8 × u8
[2792..6888] ability_cls: 8 × 128 × f32
[6888..7144] hidden_state_in: 64 × f32
```

Per-sample response layout (272 bytes with h_dim=64):
```
[0]         move_dir: u8
[1]         combat_type: u8
[2..4]      target_idx: u16 LE
[4..8]      lp_move: f32 LE
[8..12]     lp_combat: f32 LE
[12..16]    lp_pointer: f32 LE
[16..272]   hidden_state_out: 64 × f32
```

Header (512 bytes):
```
[0x00] magic: 0x47505549
[0x04] version: 1
[0x08] cls_dim: u32 (128)
[0x0C] max_batch_size: u32 (1024)
[0x10] sample_size: u32 (7144)
[0x14] response_sample_size: u32 (272)
[0x18] h_dim: u32 (64)
[0x40] flag: u32 (0=idle, 1=request_ready, 2=response_ready)
[0x44] batch_size: u32
[0x80] reload_path: 256 bytes
```

**Throughput**: ~60K inferences/sec on single GPU (RTX-class)

**Source**: `training/gpu_inference_server.py`, `src/ai/core/ability_transformer/gpu_client.rs`

---

## Parameter Breakdown

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Entity Encoder V3 | 46,848 | 4-layer transformer, d=32, entity/threat/position projections, type embeddings |
| Ability Transformer | 29,952 | 4-layer encoder for ability DSL tokens (frozen during RL, used for CLS) |
| Cross-Attention | 6,560 | MHA(4 heads) + FF + norms for ability→entity attention |
| External CLS Proj | 4,224 | Linear(128→32) + bias for ability embedding projection |
| Move Head | 1,097 | Linear(32→32)→GELU→Linear(32→9) |
| Combat Pointer Head | 22,122 | combat_type MLP + attack_query + 8 ability_queries + pointer_key |
| Value Head | 1,089 | Linear(32→32)→GELU→Linear(32→1) |
| Temporal GRU | 20,896 | GRUCell(32→64) + Linear(64→32) |
| Ability Transformer (frozen) | 1,760 | Token + positional embeddings (not counted in trainable) |
| **Total** | **134,548** | **98,516 trainable** (transformer frozen) |

---

## Key Files

| File | What |
|------|------|
| `training/model.py:1313` | `AbilityActorCriticV4` — full model definition |
| `training/model.py:1339` | `TemporalGRU` — GRU temporal context |
| `training/model.py:889` | `EntityEncoderV3` — self-attention over entities/threats/positions |
| `training/model.py:512` | `CrossAttentionBlock` — ability CLS → entity tokens |
| `training/model.py:1238` | `CombatPointerHead` — combat type + pointer targeting |
| `src/ai/core/ability_eval/game_state.rs:15` | Entity feature layout (30 features) |
| `src/ai/core/ability_eval/game_state.rs:242` | Position token extraction |
| `src/ai/core/ability_eval/game_state.rs:342` | Threat token extraction |
| `training/gpu_inference_server.py` | GPU SHM protocol (Python side) |
| `src/ai/core/ability_transformer/gpu_client.rs` | GPU SHM protocol (Rust side) |
| `src/bin/xtask/oracle_cmd/transformer_rl.rs` | Episode generation + per-unit hidden state |
| `training/impala_learner.py` | IMPALA V-trace training loop |
| `training/sac_learner.py` | SAC-Discrete training loop |
| `training/curriculum.py` | Curriculum training orchestrator |
| `src/ai/core/ability_eval.rs` | Hand-tuned ability eval (9-category urgency) |
| `training/export_actor_critic_v4.py` | Weight export (Python .pt → JSON for Rust) |
| `src/ai/core/ability_transformer/weights.rs` | Rust-side frozen inference from JSON |
