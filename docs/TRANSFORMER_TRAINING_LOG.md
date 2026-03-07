# Ability Transformer Training Log

History of approaches, experiments, and results for replacing the manual
feature-engineering ability evaluation system with a learned transformer
operating on raw DSL token sequences and rich game state.

---

## Problem Statement

The existing ability evaluation system uses 9 hand-crafted per-category
micro-models (~20 features each) to score ability urgency and target
selection. This works (92.9% win rate on attrition scenarios) but:

- Every new ability effect requires manual feature engineering
- Category boundaries are arbitrary (is a damage+heal ability "damage" or "heal"?)
- No compositional understanding (can't reason about ability *structure*)
- Doesn't generalize to novel ability combinations

The goal: a single transformer that reads raw ability DSL text, observes
the game state via entity tokens, and predicts urgency + target — replacing
all 9 micro-models with one architecture.

---

## Architecture Evolution

### v1: Concatenation (abandoned)

Concatenated [CLS] embedding from ability tokens with a flat game state
vector, then fed through MLP decision heads.

**Problem:** t-SNE showed complete embedding collapse — all ability
categories mapped to near-identical embeddings. The model learned to
ignore ability structure and rely entirely on game state features.

### v2: Cross-Attention (current)

Ability [CLS] (query) cross-attends to game state entity tokens (key/value).
This separates ability understanding from game state understanding:

```
ability DSL tokens → Transformer Encoder → [CLS] embedding
                                              ↓ (query)
game state → Entity Encoder → entity tokens → Cross-Attention → decision head
                                (key/value)
```

Three independently pretrained components:
1. **Ability transformer** — learns DSL grammar structure via masked token prediction
2. **Entity encoder** — learns game state representation via fight outcome prediction
3. **Cross-attention + decision head** — bridges the two, trained on oracle labels

### Entity Features (30 per entity, 7 slots = 210 total)

Evolved from 10 basic features to 30 rich features per entity:

| Group | Features | Indices |
|-------|----------|---------|
| Vitals | hp_pct, shield_pct, resource_pct, armor/200, mr/200 | 0–4 |
| Position/terrain | pos_x/20, pos_y/20, dist_from_caster/10, cover, elevation/5, hostile_zones/3, friendly_zones/3 | 5–11 |
| Combat stats | auto_dps/30, attack_range/10, attack_cd_pct | 12–14 |
| Ability readiness | ability_damage/50, ability_range/10, ability_cd_pct | 15–17 |
| Healing | heal_amount/50, heal_range/10, heal_cd_pct | 18–20 |
| CC capability | control_range/10, control_duration/2000, control_cd_pct | 21–23 |
| Current state | is_casting, cast_progress, cc_remaining/2000, move_speed/5 | 24–27 |
| Cumulative | total_damage_done/1000, exists | 28–29 |

Entity order: [self, enemy0, enemy1, enemy2, ally0, ally1, ally2].
Enemies sorted by distance (nearest first), allies by HP% (lowest first).
Implemented in `src/ai/core/ability_eval/game_state.rs`.

---

## Training Approaches

### Phase 1a: Ability Transformer MLM Pre-training

**Task:** Masked token prediction on 75,000 generated .ability files.
Learns compositional grammar structure — the model must understand that
`damage 55 [FIRE: 60]` and `heal 38 when target_is_stunned` are
structurally different.

**Script:** `training/pretrain.py`

#### v1 Run (d=32, 2 layers)
- 148K/500K steps completed before process died
- Peak: 82% masked token accuracy
- Flat plateau from ~40K onward — no grokking transition observed
- Settings: AdamW λ=1.0, β₂=0.98, std=0.02 init, no Grokfast

#### v2 Run (d=32, 4 layers, with improvements)
- All five literature-informed improvements applied (see below)
- 76.8% accuracy at 2K steps (v1 took ~40K to reach 82%)
- Training in progress

### Phase 1b: Entity Encoder Pre-training

**Task:** Predict fight outcome (who wins + hero HP remaining) from
game state snapshots. Forces the encoder to learn threat assessment,
positioning value, team composition strength, and temporal dynamics.

**Script:** `training/pretrain_entity.py`
**Dataset:** 5,750 samples from 28 attrition scenarios (11 wins, 17 losses)
via `xtask scenario oracle outcome-dataset`.

The encoder uses self-attention over 7 entity tokens so entities can
learn to attend to each other (e.g., "this enemy threatens me because
it has high DPS and is close").

#### v1 Run (d=32, 2 layers)
- 300K steps, completed in 2,369s (~40 min)
- Best val win accuracy: 94.2%
- HP MAE: ~0.06
- Weight norm stabilized at ~7.0
- No dramatic grokking transition — accuracy oscillated 88–94%
- Settings: AdamW λ=1.0, β₂=0.98, std=0.02 init

#### v2 Run (d=32, 2 layers, with improvements)
- **96.4% win accuracy at 6.5K steps** (v1 needed 300K for 94.2%)
- **97.9% at 12K steps**, HP MAE 0.037
- Grokfast dramatically accelerated convergence
- Training in progress

### Phase 2: Cross-Attention Fine-tuning (planned)

**Task:** Load frozen pretrained ability transformer and entity encoder,
train cross-attention + decision head on oracle-labeled ability evaluation
data (urgency + target prediction).

**Script:** `training/finetune_decision.py`

Not yet run — waiting for Phase 1a/1b v2 to complete. The script supports:
- `--pretrained` to load MLM checkpoint
- `--entity-encoder` to load outcome prediction checkpoint
- Grokfast + spectral monitoring

### Phase 3: Actor-Critic with Sim-in-the-Loop (proposed)

**Insight:** The pretrained entity encoder is already a fight outcome
predictor — it can serve as a **value function** (critic) in an
actor-critic RL setup. The ability transformer + cross-attention
becomes the policy (actor).

**Credit assignment via TD learning:**
1. Before ability use: V(s) = entity_encoder.predict_win(game_state)
2. Execute ability, sim advances
3. After: V(s') = entity_encoder.predict_win(game_state')
4. Advantage = V(s') - V(s) + (bonus if terminal win)

This eliminates the oracle rollout bottleneck entirely. Every ability
decision gets immediate dense reward signal from the value function.

**Status:** Design phase. Depends on Phase 1b producing a high-quality
value function and Phase 2 bootstrapping a reasonable initial policy.

---

## Literature-Informed Improvements (v2)

Applied to all training scripts based on review of 30+ post-Power-et-al.
grokking papers. See `training/grokfast.py` for the gradient filter.

### 1. Grokfast EMA Gradient Filter

**Source:** Lee et al. (2405.20233, 2024)
**Change:** After `loss.backward()`, apply EMA filter to amplify slow-varying
gradient components (generalization signal) over fast-varying components
(memorization signal).
**Parameters:** α=0.98, λ=2.0
**Impact:** >50× speedup reported in paper. Entity encoder v2 confirmed:
96.4% at 6.5K steps vs 94.2% at 300K steps (v1).

### 2. Data Augmentation (property reordering)

**Source:** Park et al. (2405.16658, 2025)
**Change:** For 50% of training batches, randomly reorder property lines
(target, range, cooldown, cast, hint, cost, charges, etc.) in ability DSL
text before tokenization. Properties are order-independent in the grammar.
**Rationale:** Exploits structural symmetry to increase effective dataset
diversity without changing semantics.

### 3. Smaller Initialization Scale

**Source:** Kumar et al. (2310.06110, ICLR 2024)
**Change:** Reduced init std from 0.02 (PyTorch default) to 0.007 (3× smaller).
**Rationale:** Grokking delay scales with distance from the "Goldilocks zone"
of weight norms. Smaller init starts closer to the feature-learning regime.

### 4. Increased Model Depth (4 layers)

**Source:** Murty et al. (2305.18741, ACL 2023)
**Change:** Ability transformer from 2 → 4 encoder layers.
**Rationale:** "Structural grokking" on grammar tasks shows an inverted U-shape:
~6 layers optimal, 1–2 layers may lack capacity for hierarchical structure.
Entity encoder kept at 2 layers (simpler task, larger relative dataset).

### 5. Spectral Monitoring (anti-grokking detection)

**Source:** Prakash & Martin (2602.02859, 2026)
**Change:** Track max singular value of weight matrices at each eval step.
Log as `max_eigenvalue` in CSV.
**Rationale:** Anti-grokking (late-stage generalization collapse) is caused
by "Correlation Traps" — anomalously large eigenvalues. Standard metrics
(weight norm, loss) don't detect it. Sudden eigenvalue spikes signal
impending collapse.

---

## Key Files

### Training Scripts
| File | Purpose |
|------|---------|
| `training/pretrain.py` | Phase 1a: MLM pre-training for ability transformer |
| `training/pretrain_entity.py` | Phase 1b: Outcome prediction pre-training for entity encoder |
| `training/finetune_decision.py` | Phase 2: Cross-attention fine-tuning on oracle labels |
| `training/export_weights.py` | Export full model (transformer + entity encoder + cross-attn) to JSON |
| `training/export_entity_encoder.py` | Export entity encoder only to JSON |
| `training/model.py` | PyTorch model definitions (AbilityTransformer, EntityEncoder, CrossAttention) |
| `training/tokenizer.py` | Ability DSL tokenizer (252-token closed vocabulary) |
| `training/grokfast.py` | Grokfast EMA gradient filter |

### Rust Inference
| File | Purpose |
|------|---------|
| `src/ai/core/ability_transformer/weights.rs` | Frozen transformer inference (self-attn, cross-attn, entity encoder with self-attn layers) |
| `src/ai/core/ability_eval/game_state.rs` | 210-dim game state extraction + outcome dataset generation |

### Dataset Generation (xtask CLI)
| Command | Output |
|---------|--------|
| `xtask scenario oracle ability-dataset` | Ability eval samples (urgency + target labels) |
| `xtask scenario oracle outcome-dataset` | Fight outcome samples for entity encoder |
| `xtask scenario oracle ability-encoder-export` | Ability properties for embedding autoencoder |

### Generated Artifacts
| File | Description |
|------|-------------|
| `generated/ability_dataset/` | 75,000 .ability files for MLM pre-training |
| `generated/outcome_dataset.jsonl` | 5,750 fight outcome samples (28 scenarios) |
| `generated/entity_encoder_pretrained.pt` | v1 entity encoder (94.2% win acc) |
| `generated/entity_encoder_weights.json` | v1 exported for Rust (18,304 encoder params, 410 KB) |
| `generated/ability_transformer_pretrained_d32.pt` | v1 ability transformer (82% token acc, 148K steps) |

---

## Model Specifications

### Ability Transformer (v2)
- Vocabulary: 252 tokens (closed, context-free)
- d_model: 32, n_heads: 4, n_layers: 4, d_ff: 64
- Positional embeddings: learned
- Activation: GELU, pre-norm
- No dropout (regularization via weight decay only)

### Entity Encoder
- Input: 210 floats (7 entities × 30 features)
- d_model: 32, n_heads: 4, n_layers: 2
- Self-attention over entity tokens
- Type embeddings: 3 types (self=0, enemy=1, ally=2)
- Output: 7 entity tokens of dim 32

### Cross-Attention Block
- Query: [CLS] from ability transformer (1 × 32)
- Key/Value: entity tokens from entity encoder (7 × 32)
- 4-head pre-norm attention with FF residual (d_ff: 64)
- Output: 32-dim embedding → decision head

### Decision Head
- Urgency: Linear(32→32) → GELU → Linear(32→1) → Sigmoid
- Target: Linear(32→32) → GELU → Linear(32→3) (logits over 3 enemy slots)

### Optimizer (all phases)
- AdamW, lr=1e-3, β₁=0.9, β₂=0.98, weight_decay=1.0
- Linear warmup (10 steps)
- Grokfast EMA: α=0.98, λ=2.0
- Gradient clipping: max_norm=1.0

---

## Baseline Comparison

| System | Architecture | Win Rate (28 scenarios) |
|--------|-------------|------------------------|
| Default AI | Hand-coded heuristics | 21% |
| Ability eval v2 (micro-models) | 9 × MLP per category | 39% |
| Combined (eval + student) | Micro-models + 5-class MLP | 64% |
| Combined + OOR fix | Same + out-of-range movement | **92.9%** |
| Transformer (target) | Single cross-attention model | TBD |

The transformer system aims to match or exceed the 92.9% combined system
while being fully general — no per-category feature engineering required.

---

## Open Questions

1. **Is oracle supervision necessary?** The entity encoder as value function
   + sim-in-the-loop RL could bypass oracle rollouts entirely (Phase 3).

2. **Optimal entity encoder dataset size.** Current 5,750 samples from
   28 scenarios may limit accuracy. Generated scenarios (~3,300) could
   provide 100K+ samples.

3. **Grokking phase transition.** MLM pre-training v1 plateaued at 82%.
   Will v2 (Grokfast + 4 layers + augmentation) break through?

4. **Anti-grokking risk.** Spectral monitoring is in place but we haven't
   observed a late-stage collapse yet. Unclear if our regime triggers it.

5. **Transfer to novel abilities.** The ultimate test: does the transformer
   correctly evaluate abilities it has never seen, including new effect types?
