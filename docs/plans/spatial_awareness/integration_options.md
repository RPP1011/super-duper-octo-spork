# Spatial Awareness: Model Integration Options

Four approaches for feeding geometry corner tokens into the model, ordered from
least to most expressive. These are not mutually exclusive — combining option 3
with option 1 or 4 may be the strongest approach.

## Option 1: Additional entity tokens

Concatenate spatial tokens with unit tokens in the entity encoder's self-attention.
The model cross-attends to corners the same way it attends to other units.

```
Input: [unit_0, ..., unit_7, corner_0, ..., corner_N]
         ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^
         entity tokens (30d)  spatial tokens (11d, padded to 30d)
```

**Pros:**
- No architectural changes to the entity encoder
- Corners participate in self-attention — model learns unit-corner relationships
- Action head still only selects over unit entities (spatial tokens are context-only)

**Cons:**
- Attention is O(n^2): going from 7 to 23 tokens is ~10x more computation
- Zero-padding 11d to 30d wastes attention capacity
- Spatial tokens compete for attention with unit tokens

**Experiment:** Train entity encoder with 7 unit + 8 corner tokens on fight outcome
prediction (existing pretraining task). Compare held-out accuracy to baseline.

## Option 2: Separate spatial encoder

A small transformer over just the spatial tokens, producing a pooled spatial
embedding that conditions the entity encoder.

```
Spatial tokens → small transformer → [CLS] pooled embedding (32d)
Entity tokens + spatial_embedding → entity encoder → decisions
```

The spatial embedding can be added to entity tokens (additive conditioning,
like positional encoding) or concatenated as an extra feature.

**Pros:**
- Keeps entity attention matrix small
- Spatial encoder can be pretrained independently (room classification task)
- Clean separation of spatial vs entity reasoning

**Cons:**
- Two transformers to train
- Pooling loses per-corner detail (model can't attend to specific corners)

**Experiment:** Pretrain a 2-layer spatial encoder on room-type classification
(predict Entry/Corridor/Arena from corner tokens). Then freeze and use the
pooled embedding as entity encoder conditioning. Compare to no spatial info.

## Option 3: Per-unit feature augmentation

Aggregate visible corners into a fixed-size feature vector per unit using
`VisibilitySummary`. Append to the existing 30-dim entity features.

```
Per-unit features (30d) + [visible_corner_count, avg_passage_width,
                           min_passage_width, avg_corner_distance] (4d)
= 34-dim entity features
```

**Pros:**
- Simplest to implement — just widen the entity feature dim
- No architectural changes
- No extra attention cost

**Cons:**
- Loses per-corner spatial detail (can't distinguish "narrow choke to my left"
  from "narrow choke to my right")
- Can't learn to target specific corners (e.g., for barricade placement)

**Experiment:** Add 4 summary features to entity encoder, retrain on fight
outcome prediction. A/B test on attrition win rate vs baseline.

## Option 4: Cross-attention keys (recommended)

Entity tokens are queries, spatial tokens are keys/values in a dedicated
cross-attention layer.

```
Entity tokens (Q) × Corner tokens (K,V) → spatially-aware entity embeddings
```

Each unit attends to the corners it can see. The attention weights reveal
which corners matter for each unit's decision — interpretable.

**Pros:**
- Most architecturally clean separation
- Each unit has its own spatial attention pattern
- Variable number of corners handled naturally by attention
- Attention weights are interpretable (which corners does the engineer attend to?)

**Cons:**
- New cross-attention layer to add and train
- Need to handle variable corner count (masking for units that see fewer corners)
- Slight training complexity increase

**Experiment:** Add a CrossAttentionBlock(d_model=32, n_heads=4) between
entity self-attention and the decision head. Entity tokens attend to their
visible corner tokens (masked by the per-unit visibility bitset). Train
end-to-end on fight outcome + urgency prediction.

## Recommended experiment order

1. **Option 3** first — cheapest to implement, establishes whether spatial info
   helps at all. If win rate doesn't improve, the other options won't either.
2. **Option 4** if option 3 shows improvement — captures per-corner detail that
   summary features lose. This is the target architecture.
3. **Option 1** as a quick A/B against option 4 — tests whether shared
   self-attention works as well as dedicated cross-attention.
4. **Option 2** only if the spatial encoder needs to be pretrained separately
   (e.g., for room-adaptive behavior without retraining the full model).
