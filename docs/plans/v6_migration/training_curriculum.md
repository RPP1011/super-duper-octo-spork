# V6 Training Curriculum

## Motivation

The V6 model is ~12x larger than V4 (1.7M vs 134K params) with multiple new
components (spatial cross-attention, latent interface, CfC temporal cell, aggregate
token, wider entity/threat features). End-to-end RL from random initialization
will not converge in reasonable time — the critic is random, advantages are noise,
and policy gradients are meaningless.

The curriculum pretrains **representations and the value function** bottom-up,
then introduces RL on top of a stable foundation. The key principle: pretrain the
critic first, because a calibrated value function stabilizes everything else.

---

## Stage 0a — Encoder + Value Head (Fight Outcome Prediction)

**Goal:** Train the entity encoder, all projection layers, type embeddings, and a
value head to assess fight state quality from single-tick snapshots.

**Architecture trained:**
- `entity_proj(34→128)`, `threat_proj(10→128)`, `position_proj(8→128)`, `agg_proj(16→128)`
- Type embeddings (6 types)
- TransformerEncoder (4 layers, 8 heads, d=128)
- Value head (Linear 128→1) — two copies: attrition head + survival head

**Architecture NOT trained (frozen / absent):**
- Spatial cross-attention (not yet added, Phase 2)
- Latent interface (not yet added, Phase 2)
- CfC temporal cell (not yet added, Phase 3)
- Move head, combat pointer head (untouched until RL)

### Data Generation

Run the sim with a **policy mixture** across a **difficulty grid**.

**Policy mixture** (critical for state space coverage):
- V4 trained policy (~40% of episodes)
- Random actions (~30% of episodes)
- Scripted heuristics (~30% of episodes):
  - "Always move toward nearest enemy"
  - "Always kite / maintain max range"
  - "Focus lowest HP target"
  - "Random movement + smart targeting"

Using only V4 policy data produces a narrow state distribution — the model
only learns to evaluate states that V4 actually visits. Random and scripted
policies explore broadly, exposing the model to losing positions, unusual
formations, and states V4 would never reach.

**Difficulty grid:**

| Axis | Values |
|------|--------|
| Enemy count | 1, 3, 5, 8, 12 |
| Enemy tier | trash, normal, elite, mixed, boss |
| Room type | open, corridor, arena, multi-choke |
| Ally count | 1, 3, 5 |

~300 grid cells × ~1000 episodes per cell = **300K episodes**.
At ~200 ticks average, sampling 5 random ticks each = **~1.5M training samples**.

The difficulty grid is the key to compositional reasoning about fight difficulty.
"5 trash mobs in a corridor" and "5 elites in an open arena" have the same enemy
count but very different difficulty — the encoder must learn why.

### Target Signals

Binary win/loss is low information per sample. Two continuous targets, predicted
jointly via multi-task heads:

| Target | Type | Description |
|--------|------|-------------|
| **Attrition ratio** | Continuous [0, 1] | `allies_alive_at_end / allies_at_start`, normalized against enemy attrition. Distinguishes "won with full team" from "won with 1 HP on last unit." |
| **Per-unit survival ticks** | Continuous [0, 1] | Ticks until the acting unit dies, normalized by episode length. Captures "about to die" vs "safe for now" even in eventual wins. |

The attrition head trains global situation assessment. The survival head provides
gradient to individual entity token representations — "this unit's features predict
it's about to die" trains the encoder to represent per-unit danger.

### Training

- Loss: `0.7 × MSE(attrition_ratio) + 0.3 × MSE(survival_ticks)`
- Optimizer: Adam, cosine LR schedule, base LR ~5e-4
- Pure supervised — no RL overhead, no SHM, no Rust in the loop
- Convergence: ~30-60 min on the 4090 at d=128

---

## Stage 0b — Spatial Cross-Attention Pretraining (Optional, Parallel)

**Goal:** Give the spatial cross-attention layer a head start on learning what
corners mean tactically. Gated on Phase 2 of the migration plan — skip if
Phase 1 spatial summary features show no improvement.

**Decision:** Skip initially. Rely on zero-init (cross-attention starts as
identity passthrough) and let spatial reasoning emerge from RL signal. Add
this stage only if Phase 2 spatial metrics (corridor damage, choke control)
are slow to improve.

If needed, two standalone supervised tasks using a small model
(`corner_proj` + 1-layer cross-attention + linear head):

| Task | Input | Target | What it teaches |
|------|-------|--------|-----------------|
| **Cover effectiveness** | Unit position + visible corner tokens | Damage taken in next 20 ticks | Concave corner nearby + enemies opposite = low damage |
| **Choke value** | Corner tokens for a position | Enemies passing through cell in next 50 ticks | Identify high-traffic chokepoints from geometry |

Transplant trained `corner_proj` + cross-attention weights into V6's
`SpatialCrossAttention` module. Training time: ~15-20 min.

---

## Stage 0c — Temporal Pretraining (Sequential, After 0a)

**Goal:** Teach the CfC temporal cell to track fight momentum from sequences
of encoder outputs.

**Architecture trained:**
- CfCCell (input=128, hidden=256) + projection
- Value head (fine-tuned from 0a checkpoint)

**Frozen:** Entire encoder from Stage 0a.

### Method

Feed contiguous sequences of 10-20 ticks through:
`frozen_encoder(tick_t) → CfC(encoding_t, h_{t-1}) → value_head(h_t)`

Predict fight outcome (attrition ratio) from the **final** CfC hidden state.
The temporal cell learns what sequential patterns matter: is enemy HP declining
steadily (being focused) or stable (being ignored)? Is the unit kiting
successfully or getting cornered?

### Data

Reuse episodes from Stage 0a. Instead of sampling single random ticks, sample
**contiguous 10-20 tick windows**. Same labels (attrition ratio at episode end).

### Training

- Freeze encoder — only CfC + value head receive gradients
- ~30 min on the 4090
- Validate: compare value prediction accuracy between single-tick (0a) and
  sequence-based (0c). If 0c is not meaningfully better, the temporal cell
  isn't learning useful dynamics and may need longer windows or more data.

---

## Stage 0d — Latent Interface Warmup (Quick, After 0c)

**Goal:** Teach the latent interface to compress encoder output into a useful
bottleneck representation without disrupting the pretrained encoder.

**Architecture trained:**
- LatentInterface (K=12 latent tokens, Read/Write attention, 2 latent blocks)
- Value head (fine-tuned)

**Frozen:** Encoder, CfC cell.

### Method

The latent interface was zero-initialized (identity passthrough) during 0a-0c.
Now unfreeze it and run fight outcome prediction for a few thousand steps.
The latent interface learns to compress the 22 encoder output tokens into 12
latent tokens that pool into an effective value prediction — without changing
the representations it compresses.

### Training

- ~10-15 min. The latent interface is small and has a clear optimization target.
- If value prediction accuracy degrades when the latent interface is introduced,
  the compression is lossy in a bad way — reduce K or increase n_latent_blocks.

---

## Stage 0e — Combat Pointer BC Warmup (Optional, Short)

**Goal:** Give the combat pointer head a warm start on target selection. This is
the one place where behavioral cloning from the existing AI is justified —
target selection heuristics ("focus low HP," "prioritize casters") are largely
geometry-independent, and the correct target is hard for RL to discover from
scratch because reward signal is sparse.

**Freeze everything except the combat pointer head.** The encoder is pretrained
on fight outcomes; BC gradients must not corrupt those representations.

### Why BC is limited to target selection only

The existing AI (V4 / GOAP) has no spatial awareness. If the move head is
trained via BC, it learns "move the way a spatially-unaware teacher moves."
The spatial summary features and cross-attention receive zero gradient because
the teacher's movement actions are conditionally independent of spatial
information. The model learns to treat spatial tokens as noise — then RL must
un-learn this anti-spatial prior before discovering cover-seeking, choke-holding,
or geometry-aware formation behavior. This is worse than starting the move head
from random initialization, which at least has no prior against spatial behavior.

Target selection is less affected because "attack the low-HP enemy" and "focus
the caster" are valid regardless of room geometry.

### Method

- Source: GOAP's target priority decisions from recorded episodes
- Labels: which entity slot the GOAP system selected as combat target
- Loss: cross-entropy over entity slots
- Schedule: **short** — a few thousand steps, well before convergence. The goal
  is "in the right neighborhood" so RL can refine, not "perfectly imitating GOAP."
- Duration: ~5-10 min

---

## Stage 1 — RL Fine-Tuning (Graduated Unfreezing)

Assemble the full V6 model with all pretrained components. The encoder understands
state, the CfC tracks temporal dynamics, the value head is a calibrated critic,
the latent interface compresses effectively, and the combat pointer head has a
warm start on targeting. Only the move head is fully untrained.

### Unfreezing schedule

| Step | Trainable | Frozen | What it achieves |
|------|-----------|--------|------------------|
| **1a** | Move head, combat pointer head, value head | Encoder, CfC, latent interface, spatial cross-attn | Action heads learn on top of rich frozen representations. Few trainable params, stable gradients from pretrained critic. |
| **1b** | + Latent interface | Encoder, CfC, spatial cross-attn | Bottleneck representation co-adapts with action heads. ~200K trainable params. |
| **1c** | + Spatial cross-attention | Encoder, CfC | Spatial reasoning enters the policy loop. Monitor spatial metrics. |
| **1d** | + Encoder (last layer first) | CfC, encoder layers 1-3 | Gradual encoder unfreezing, standard transfer learning. Monitor value head for catastrophic forgetting — if value predictions spike, refreeze and lower LR. |
| **1e** | Full model | — | Full fine-tuning at reduced LR (~1e-4 or lower). |

### Monitoring

| Signal | Action if triggered |
|--------|-------------------|
| Value loss spikes after unfreezing a layer | Refreeze that layer, lower LR by 2x, try again |
| Move head loss plateaus at 1a | Encoder representations may not capture what movement needs — check if position tokens are informative |
| Spatial metrics don't improve at 1c | Spatial cross-attention may need Stage 0b pretraining after all |
| Combat pointer head regresses at 1b | Latent interface compression is losing per-entity resolution — increase K or check Write attention |
| Win rate exceeds V4 before step 1d | Encoder pretraining was sufficient — consider skipping graduated unfreezing and going straight to 1e at low LR |

---

## Pre-RL Validation: Value-Based Action Ranking

Before committing to RL, validate the pretrained encoder + value head with a
zero-training-required test: run the model with a **scripted search policy**
that uses the value head as a ranking function.

For each candidate action (9 move directions × N targets), evaluate V(s')
using the pretrained value head and pick the highest-value action. This is
1-step greedy planning with a learned value function — no policy network
needed.

| Result | Interpretation |
|--------|---------------|
| Decent win rate (>60% on medium difficulty) | Encoder and value head are good. Action heads just need to learn to shortcut the search. RL should converge fast. |
| Poor win rate despite good value prediction accuracy | Value function is well-calibrated but doesn't capture action-relevant distinctions (two actions lead to similar value). Need finer-grained value targets or action-conditional value prediction. |
| Poor win rate and poor value accuracy | Encoder needs more pretraining. Don't start RL yet. |

---

## Wall-Clock Budget Estimate

| Stage | Duration | Cumulative |
|-------|----------|------------|
| Data generation (300K episodes) | 2-4 hr (CPU) | 2-4 hr |
| 0a — Encoder + value head | 30-60 min | 3-5 hr |
| 0b — Spatial cross-attn (if needed) | 15-20 min | — |
| 0c — Temporal pretraining | ~30 min | 3.5-5.5 hr |
| 0d — Latent interface warmup | 10-15 min | 4-6 hr |
| 0e — Combat pointer BC (if used) | 5-10 min | 4-6 hr |
| Pre-RL validation | ~15 min | 4-6 hr |
| 1a-1e — RL fine-tuning | 2-4 hr | 6-10 hr |

Total: **~6-10 hours** from data generation to trained V6 policy, vs. potentially
days of unstable RL from random initialization.

Data generation can overlap with other work (CPU-bound, sim only). Stages 0a-0e
are fast supervised passes on the 4090 with no SHM/Rust overhead.

---

## Files

| File | Stage | Change |
|------|-------|--------|
| `training/pretrain_encoder.py` | 0a | New: supervised fight outcome prediction training loop |
| `training/pretrain_temporal.py` | 0c | New: sequential CfC pretraining on contiguous tick windows |
| `training/pretrain_latent.py` | 0d | New: latent interface warmup loop (or fold into pretrain_encoder.py) |
| `training/pretrain_pointer_bc.py` | 0e | New: combat pointer head BC from GOAP target decisions |
| `training/generate_pretrain_data.py` | 0a | New: episode generation across difficulty grid with policy mixture |
| `training/model.py` | 0a-1e | Multi-task heads (attrition + survival), freeze/unfreeze API |
| `training/validate_value.py` | Pre-RL | New: value-based action ranking evaluation |
| `src/bin/xtask/oracle_cmd/` | 0a | Episode generation with configurable policy + difficulty |
