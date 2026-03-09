# Ability System

Full pipeline from TOML definition through runtime execution to AI representations.

## Table of Contents

1. [Ability Definition (TOML)](#ability-definition-toml)
2. [DSL Text Representation](#dsl-text-representation)
3. [Effect System](#effect-system)
4. [Runtime Slots](#runtime-slots)
5. [80-dim Property Vector](#80-dim-property-vector)
6. [Autoencoder (80 to 32)](#autoencoder-80-to-32)
7. [Ability Transformer](#ability-transformer)
8. [AI Pipeline Integration](#ai-pipeline-integration)
9. [Example Heroes](#example-heroes)

---

## Ability Definition (TOML)

Heroes are defined in `assets/hero_templates/*.toml`. Each file contains stats, attack, abilities (active), and passives.

### TOML Structure

```toml
[hero]
name = "Mage"

[stats]
hp = 70
move_speed = 2.5

[attack]
damage = 8
range = 4.0

[[abilities]]
name = "Fireball"
targeting = "target_enemy"       # see AbilityTargeting enum
range = 5.0
cooldown_ms = 5000
cast_time_ms = 500
ai_hint = "damage"              # damage | heal | crowd_control | defense | utility

# Direct effects (instant delivery)
[[abilities.effects]]
type = "damage"
amount = 55
[abilities.effects.tags]
FIRE = 60.0

# OR delivery-based effects
[abilities.delivery]
method = "projectile"           # projectile | zone | channel | chain | tether | trap
speed = 8.0
[[abilities.delivery.on_hit]]
type = "damage"
amount = 55

[[passives]]
name = "ArcaneShield"
cooldown_ms = 30000
[passives.trigger]
type = "on_hp_below"
percent = 50.0
[[passives.effects]]
type = "shield"
amount = 40
duration_ms = 4000
```

### AbilityDef Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Ability identifier |
| `targeting` | Enum | TargetEnemy, TargetAlly, SelfCast, SelfAoe, GroundTarget, Direction, Vector, Global |
| `range` | f32 | Cast range in world units |
| `cooldown_ms` | u32 | Cooldown between uses |
| `cast_time_ms` | u32 | Wind-up time before effects fire |
| `ai_hint` | String | Category hint for AI: damage, heal, crowd_control, defense, utility |
| `effects` | Vec | Direct effects (instant delivery) |
| `delivery` | Option | Delivery mechanism with sub-effects (on_hit, on_arrival) |
| `resource_cost` | i32 | Resource consumed on cast |
| `max_charges` | u32 | Ammo system (0 = normal cooldown) |
| `is_toggle` | bool | On/off toggle ability |
| `recast_count` | u32 | Number of recasts before cooldown |
| `unstoppable` | bool | CC-immune during cast |
| `morph_into` | Option | Ability replacement on cast |
| `zone_tag` | Option | Element tag for zone-reaction combos |

Source: `src/ai/effects/defs.rs`

---

## DSL Text Representation

Abilities can also be defined in `.ability` files using a compact brace-based DSL. The DSL is bidirectional: TOML abilities can be emitted as DSL text (`emit.rs`) and DSL text can be parsed back to `AbilityDef` (`parser.rs`).

### DSL Example

```
ability Fireball {
    target: enemy, range: 5.0
    cooldown: 5s, cast: 300ms
    hint: damage

    deliver projectile { speed: 8.0, width: 0.3 } {
        on_hit { damage 55 [FIRE: 60] }
        on_arrival { damage 15 in circle(2.0) }
    }
}
```

### DSL Grammar Summary

| Construct | Syntax |
|-----------|--------|
| Header | `ability NAME { ... }` |
| Targeting | `target: enemy/ally/self/self_aoe/ground/direction/vector/global` |
| Timing | `cooldown: 5s, cast: 300ms` |
| Effect | `damage 55`, `heal 30`, `stun 2s`, `shield 40 for 4s` |
| Area | `in circle(3.0)`, `in cone(4.0, 60.0)`, `in line(5.0, 1.0)` |
| Condition | `when target_hp_below(25%)`, `when target_is_stunned` |
| Tags | `[FIRE: 60, PHYSICAL: 40]` |
| Delivery | `deliver projectile { speed: 8.0 } { on_hit { ... } }` |
| DoT/HoT | `damage 10/tick for 4s`, `heal 5/tick for 3s` |
| Scaling | `+ 15% caster_max_hp` |

### Pipeline Role

DSL text is the input to the **ability transformer**. The emitter (`src/ai/effects/dsl/emit.rs`) converts any `AbilityDef` into DSL text, which is then tokenized into integer sequences for the transformer.

Source: `src/ai/effects/dsl/` (mod.rs, parser.rs, ast.rs, lower.rs, emit.rs)

---

## Effect System

The effect system follows a five-dimensional composition model:

**WHAT** (Effect) x **WHERE** (Area) x **HOW** (Delivery) x **WHEN** (Condition/Trigger) + **Tags**

### Effects (52 types)

| Category | Effects |
|----------|---------|
| Damage | Damage, SelfDamage, Execute, DamageModify, DeathMark, Detonate |
| Healing | Heal, Shield, Resurrect, OverhealShield, AbsorbToHeal, ShieldSteal |
| Hard CC | Stun, Root, Silence, Suppress, Fear, Charm, Polymorph, Banish, Confuse |
| Soft CC | Slow, Knockback, Pull, Grounded, Blind, Taunt |
| Mobility | Dash (with blink variant), Swap, Stealth, Attach |
| Buffs | Buff, Debuff, Lifesteal, Reflect, OnHitBuff, Immunity |
| Utility | Dispel, Summon, CommandSummons, Duel, Leash, Link, Redirect, Rewind |
| Stacks | ApplyStacks, StatusClone, StatusTransfer, CooldownModify |
| Terrain | Obstacle, ProjectileBlock |
| Meta | EvolveAbility |

### Areas (7 shapes)

| Shape | Parameters |
|-------|-----------|
| SingleTarget | (default) |
| SelfOnly | Targets caster |
| Circle | radius |
| Cone | radius, angle_deg |
| Line | length, width |
| Ring | inner_radius, outer_radius |
| Spread | radius, max_targets |

### Deliveries (7 methods)

| Method | Parameters | Sub-effects |
|--------|-----------|-------------|
| Instant | (default) | - |
| Projectile | speed, pierce, width | on_hit, on_arrival |
| Channel | duration_ms, tick_interval_ms | - |
| Zone | duration_ms, tick_interval_ms | - |
| Tether | max_range, tick_interval_ms | on_complete |
| Trap | duration_ms, trigger_radius, arm_time_ms | - |
| Chain | bounces, bounce_range, falloff | on_hit |

### ConditionalEffect Wrapper

Every effect is wrapped in `ConditionalEffect` which adds:

| Field | Purpose |
|-------|---------|
| `condition` | When to apply (e.g., `TargetHpBelow(25%)`, `TargetIsStunned`) |
| `area` | Where to apply (circle, cone, line, etc.) |
| `tags` | Power tags for resistance checks |
| `stacking` | Refresh, Extend, Strongest, or Stack |
| `chance` | Probability (0.0 = always) |
| `else_effects` | Fallback effects when condition is false |

### Conditions (27 types)

Target state checks (`TargetHpBelow`, `TargetIsStunned`, `TargetHasTag`, etc.), caster state checks (`CasterHpBelow`, `CasterResourceAbove`, etc.), spatial checks (`TargetDistanceBelow/Above`), count checks (`AllyCountBelow`, `TargetDebuffCount`), and compound logic (`And`, `Or`, `Not`).

### Triggers (for passives, 17 types)

`OnDamageDealt`, `OnDamageTaken`, `OnKill`, `OnDeath`, `OnHpBelow`, `Periodic`, `OnAllyDamaged`, `OnAbilityUsed`, `OnShieldBroken`, `OnHealReceived`, `OnAutoAttack`, `OnStackReached`, etc.

Source: `src/ai/effects/effect_enum.rs`, `src/ai/effects/types.rs`

---

## Runtime Slots

At runtime, abilities are stored as `AbilitySlot` which wraps the definition with mutable state.

```
AbilitySlot
  def: AbilityDef           -- the static definition
  cooldown_remaining_ms: u32 -- ticks down each frame
  base_def: Option<Box>     -- original def when morphed
  morph_remaining_ms: u32   -- morph revert timer
  charges: u32              -- current ammo count
  charge_recharge_remaining_ms: u32
  toggled_on: bool          -- toggle state
  recasts_remaining: u32    -- remaining recasts this window
  recast_window_remaining_ms: u32
```

`is_ready()` returns true when:
- Charge-based: `charges > 0`
- Toggle: always ready (on/off)
- Mid-recast: `recasts_remaining > 0 && recast_window_remaining_ms > 0`
- Normal: `cooldown_remaining_ms == 0`

Passives use `PassiveSlot` with `periodic_elapsed_ms` for periodic triggers.

Source: `src/ai/effects/defs.rs`

---

## 80-dim Property Vector

The property extractor converts any `AbilityDef` into a fixed 80-float feature vector. This is the input to the autoencoder, not used directly by the policy.

### Feature Layout

| Indices | Dim | Category | Features |
|---------|-----|----------|----------|
| 0-7 | 8 | **Targeting** | One-hot: TargetEnemy, TargetAlly, SelfCast, SelfAoe, GroundTarget, Direction, Vector, Global |
| 8-13 | 6 | **Core scalars** | range/10, cooldown/20000, cast_time/2000, resource_cost/30, (2 reserved for runtime) |
| 14-20 | 7 | **Delivery** | One-hot: Instant, Projectile, Channel, Zone, Tether, Trap, Chain |
| 21-26 | 6 | **Delivery props** | speed/20, pierce, duration/5000, tick_interval/2000, bounces/5, trigger_radius/5 |
| 27-31 | 5 | **Mechanic flags** | is_toggle, unstoppable, has_recast, has_charges, has_morph |
| 32-37 | 6 | **AI hint** | One-hot: damage, heal, crowd_control, defense, utility, other |
| 38-40 | 3 | **Damage type** | has_physical, has_magic, has_true |
| 41-44 | 4 | **Damage** | total_instant/150, dot_dps/50, has_execute, damage_modify_factor |
| 45-47 | 3 | **Healing** | total_instant_heal/150, hot_hps/50, total_shield/100 |
| 48-54 | 7 | **Hard CC** | stun/3000, root/3000, silence/3000, suppress/3000, fear/3000, charm/3000, polymorph/3000 |
| 55-58 | 4 | **Soft CC** | slow_factor, slow/3000, knockback_dist/5, pull_dist/5 |
| 59-62 | 4 | **Other CC** | taunt/3000, banish/3000, confuse/3000, grounded/3000 |
| 63-65 | 3 | **Mobility** | has_dash, is_blink, has_stealth |
| 66-69 | 4 | **Buffs/debuffs** | buff_factor, buff_dur/5000, debuff_factor, debuff_dur/5000 |
| 70-74 | 5 | **Area** | max_radius/6, has_cone, has_line, has_spread, is_aoe |
| 75-79 | 5 | **Special** | has_summon, has_obstacle, has_reflect, has_lifesteal, num_effects/8 |

All values are normalized to approximately [0, 1]. The extraction walks all effects including delivery sub-effects (on_hit, on_arrival, on_complete) via `collect_all_effects()`.

Source: `src/ai/core/ability_encoding/properties.rs`, `src/ai/core/ability_encoding/effects.rs`

---

## Autoencoder (80 to 32) — Legacy

> **Superseded by the Ability Transformer below.** The autoencoder was the first approach to ability encoding, mapping hand-crafted 80-dim property vectors to 32-dim embeddings. It is no longer used in the primary AI pipeline — the ability transformer operates on raw DSL tokens and produces richer, learned representations without manual feature engineering.

A frozen two-layer MLP autoencoder compresses 80-dim properties to 32-dim L2-normalized embeddings.

### Architecture

```
Encoder: 80 → 64 (ReLU) → 32 (L2-norm)
Decoder: 32 → 64 (ReLU) → 80 (linear)
```

| Component | Parameters |
|-----------|-----------|
| Encoder | 80x64 + 64 + 64x32 + 32 = 7,264 |
| Decoder | 32x64 + 64 + 64x80 + 80 = 7,312 |
| **Total** | **14,576** |

### Training

- Script: `scripts/train_ability_encoder.py`
- Data: 856 abilities x 80-dim properties (`generated/ability_encoder_data.json`)
- Loss: supervised contrastive (9 ability categories) + reconstruction MSE
- Results: 99.9% kNN@5 accuracy, 0.035 MSE
- Weights: `generated/ability_encoder.json`

Source: `src/ai/core/ability_encoding/autoencoder.rs`

---

## Ability Transformer (Primary)

The ability transformer is the **primary ability encoding model**. It processes raw DSL token sequences end-to-end, replacing the manual 80-dim feature engineering with learned representations. This is the model used as a frozen subnetwork in the actor-critic and entity encoder pipelines.

### Tokenizer

252-token vocabulary with semantic number bucketing. Ability names map to `[NAME]`, string literals to `[STR]`.

| Token range | Count | Category |
|-------------|-------|----------|
| 0-7 | 8 | Special: [PAD], [CLS], [MASK], [SEP], [UNK], [NAME], [STR], [TAG] |
| 8-17 | 10 | Punctuation: `{ } ( ) [ ] : , + %` |
| 18-226 | 209 | Keywords (all DSL keywords, sorted) |
| 227-251 | 25 | Number buckets (0, 1, 2, ..., 10, 15, 20, 25, 30, ..., 100, 200, 300, 500, 1000) |

Source: `src/ai/core/ability_transformer/tokenizer.rs`, `training/tokenizer.py`

### AbilityTransformer (Encoder)

| Hyperparameter | Value |
|----------------|-------|
| vocab_size | 252 |
| d_model | 32 (production) / 64 (pretrain) |
| n_heads | 4 |
| n_layers | 4 |
| d_ff | 64 (production) / 128 (pretrain) |
| max_seq_len | 256 |
| dropout | 0.0 (grokking setup) |
| activation | GELU |
| norm | pre-norm (LayerNorm) |
| init std | 0.007 (3x reduced, per Kumar et al.) |
| [CLS] init | zero |

### Grokking Training Setup

Designed for delayed generalization (grokking):
- AdamW with lambda=1.0, beta2=0.98
- No dropout, no early stopping
- Grokfast EMA gradient filter (alpha=0.98, lamb=2.0) from `training/grokfast.py`
- Data augmentation: random property line reordering in DSL
- Structural depth: 4 layers (Murty et al.)
- Spectral monitoring for anti-grokking detection

### Pre-training (Phase 1)

Masked Language Modeling (MLM) on 75K ability DSL sequences.

```
AbilityTransformerMLM
  transformer: AbilityTransformer
  mlm_head: Linear(d) → GELU → LayerNorm → Linear(vocab_size)
```

Auxiliary `HintClassificationHead` predicts ability category from [CLS] during pretraining.

Source: `training/pretrain.py`

### Entity Encoder

Encodes the game state (7 entity slots x 30 features = 210 floats) into d_model-dim tokens.

```
EntityEncoder
  proj: Linear(30, d_model)
  type_emb: Embedding(3, d_model)     -- 0=self, 1=enemy, 2=ally
  input_norm: LayerNorm(d_model)
  encoder: TransformerEncoder(n_layers, n_heads)
  out_norm: LayerNorm(d_model)
```

Input ordering: [self, enemy0, enemy1, enemy2, ally0, ally1, ally2]. Padding detected via `exists` feature (index 29).

### Cross-Attention Block

Bridges ability [CLS] embedding with entity tokens.

```
CrossAttentionBlock
  Query:  norm_q([CLS])                      -- from ability transformer
  Key/Value: norm_kv(entity_tokens)           -- from entity encoder
  cross_attn: MultiheadAttention(d_model, 4 heads)
  ff: Linear(d, 2d) → GELU → Linear(2d, d)  -- residual feed-forward
```

Output: context-aware ability representation used by the decision/policy head.

### Fine-tuning (Phase 2)

```
AbilityTransformerDecision
  transformer → [CLS] embedding
  entity_encoder → entity tokens
  cross_attn: [CLS] x entities → context-aware embedding
  decision_head:
    urgency: Linear(d,d) → GELU → Linear(d,1) → Sigmoid    -- [0,1]
    target:  Linear(d,d) → GELU → Linear(d,3)               -- 3-class
```

Best results: 91% oracle agreement, 94% target accuracy (d_model=32, 96,740 params).

Source: `training/model.py`, `training/finetune_decision.py`

---

## AI Pipeline Integration

The ability system feeds into multiple AI components:

### Pipeline Overview

```
TOML/DSL → AbilityDef → AbilitySlot (runtime)
                ↓
          emit DSL text (emit.rs)
                ↓
          tokenize (252-token vocab)
                ↓
     Ability Transformer (frozen)
          ↓ [CLS] embedding (32-dim, cached at fight start)
          ↓
     CrossAttention × EntityEncoder → context-aware ability repr
          ↓
     DecisionHead / ActorCritic / EntityEncoder (next-state)
```

The ability transformer is the **frozen subnetwork** — pre-trained via MLM, fine-tuned in Phase 2, then weights are locked. Only entity encoder + cross-attention + heads run per tick. [CLS] embeddings are computed once per ability at fight start and cached.

### Ability Eval (Interrupt Layer)

Nine per-category micro-models evaluate abilities each tick. Fires as an interrupt in `src/ai/squad/intents.rs` when urgency > 0.4. Categories derived from `AbilityCategory::from_ability_full()` using ai_hint, targeting, effects, and delivery.

Source: `src/ai/core/ability_eval/`

### Actor-Critic (RL Policy)

The `AbilityActorCritic` model uses abilities as part of its 14-action space:

| Action Index | Meaning |
|-------------|---------|
| 0-2 | attack_nearest, attack_weakest, attack_focus |
| 3-10 | use ability 0-7 |
| 11-12 | move_toward, move_away |
| 13 | hold |

Per-ability logits are produced by cross-attending each ability's [CLS] embedding with entity tokens, then projecting to a scalar. Abilities are pre-tokenized and [CLS] embeddings are cached at fight start; only the entity encoder + cross-attention + heads run per tick.

### V3 Pointer Architecture (In Progress)

Replaces flat action space with `(action_type, target_pointer)`:
- 11 action types: attack(0), move(1), hold(2), ability_0..7(3..10)
- Target selected via scaled dot-product attention over entity/threat/position tokens
- `EntityEncoderV3`: adds position tokens (type=4) with 8-dim spatial features
- `PointerHead`: per-action-type query projections with masked attention

Source: `training/model.py`, `src/ai/core/ability_transformer/weights.rs`

---

## Example Heroes

### Mage (ranged DPS, 8 abilities)

```toml
[stats]
hp = 70, move_speed = 2.5

[attack]
damage = 8, range = 4.0
```

| Ability | Targeting | Range | CD | Hint | Key Effects |
|---------|-----------|-------|----|------|-------------|
| Fireball | target_enemy | 5.0 | 5s | damage | Projectile: 55 dmg on_hit + 15 AoE r=2.0 on_arrival |
| FrostNova | self_aoe | 0.0 | 10s | crowd_control | 20 dmg + 2s stun in circle(3.0) |
| ArcaneMissiles | target_enemy | 5.0 | 4s | damage | Chain 3 bounces: 35 dmg/hit, 0.8 falloff |
| Blizzard | ground_target | 6.0 | 12s | damage | Zone 4s: 15 dmg + 30% slow per tick in circle(3.0) |
| Meteor | ground_target | 6.0 | 18s | damage | 80 dmg in circle(2.5) |
| Blink | self_cast | 0.0 | 10s | utility | Dash 3.0 away + 15 AoE dmg at origin |
| Polymorph | target_enemy | 5.0 | 15s | crowd_control | 3s polymorph |
| ManaShield | self_cast | 0.0 | 14s | defense | 50 shield for 5s |

### Assassin (melee burst, 8 abilities)

```toml
[stats]
hp = 70, move_speed = 4.0

[attack]
damage = 25, range = 1.2
```

| Ability | Targeting | Range | CD | Hint | Key Effects |
|---------|-----------|-------|----|------|-------------|
| ShadowStrike | target_enemy | 5.0 | 5s | damage | Dash to target + 55 dmg |
| Eviscerate | target_enemy | 1.3 | 6s | damage | 70 dmg + 35 bonus when target_hp_below(25%) |
| SmokeBomb | self_aoe | 0.0 | 12s | crowd_control | 20 dmg + 50% slow 2.5s in circle(2.5) |
| PoisonBlade | target_enemy | 1.3 | 6s | damage | 15 dmg + 10/tick DoT for 4s |
| FanOfKnives | self_aoe | 0.0 | 8s | damage | 30 dmg in circle(3.0) |
| Garrote | target_enemy | 1.3 | 10s | crowd_control | 2s stun + 8/tick DoT for 2s |
| ShadowStep | target_enemy | 5.0 | 4s | utility | Dash to target + 20 dmg |
| CripplingPoison | target_enemy | 1.3 | 8s | utility | 50% slow 3s + debuff damage_output 85% 3s |
