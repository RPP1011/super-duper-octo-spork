# AbilityActorCriticV5 Architecture

## Target: ~1.5M params, d=128, 8 heads, h_dim=256

```
Input:
  7 entity slots × 30 features      (importance-selected)
  4 threat tokens × 8 features
  8 position tokens × 8 features
  1 aggregate token × 16 features   ← crowd summary for truncated entities
  8 ability CLS embeddings × 128d   ← direct feed, no projection bottleneck

                    ┌──────────────────────────────────┐
                    │       Entity Encoder V5           │
                    │   d=128, 8 heads, 4 layers        │
                    │   pre-norm, QK-norm, RoPE         │
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
                    │   Tail dropping during training   │
                    │                                   │
                    │   → pool latents → (B, 128)       │
                    └────────────┬─────────────────────┘
                                 │
                         pooled (B, 128)
                                 │
                    ┌────────────▼─────────────────────┐
                    │       CfC Temporal Cell           │
                    │   input=128, hidden=256           │
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

## Key Design Decisions

### No grouping for Read/Write
ELIT groups 1024 image tokens into 16 groups for efficiency. We have 20 tokens total —
full cross-attention between 12 latents and 20 tokens is O(20×12×128), trivially cheap.
No grouping needed.

### Tail dropping during training
Per ELIT Section 3.4: randomly sample J̃ ∈ [J_min, K] per training step, keep only first
J̃ latents. Creates importance ordering where earlier latents capture more critical info.

For game AI: latent 0 = overall tactical situation, latent 6 = specific combo window,
latent 11 = minor positioning refinement. During inference, can drop tail latents for
faster evaluation on less critical ticks.

### Ability CLS direct feed
V4 projected 128d → 32d, losing 75% of ability representation. V5 feeds 128d directly
since d_model=128. The cross-attention sees the full behavioral embedding.

### Aggregate token
Summarizes all entities that didn't earn individual slots. See Scene Representation
section for the full 16-feature layout and importance-based slot selection.

## Parameter Breakdown

| Component | V4 (d=32) | V5 (d=128) |
|-----------|-----------|------------|
| Entity Encoder | 46,848 | ~750,000 |
| Aggregate Proj | — | ~2,000 |
| Latent Interface (K=12) | — | ~200,000 |
| CfC Cell (h=256) | — | ~350,000 |
| GRU (h=64) | 20,896 | removed |
| External CLS Proj | 4,224 | removed |
| Cross-Attention | 6,560 | ~50,000 |
| Move Head | 1,097 | ~17,000 |
| Combat Pointer Head | 22,122 | ~200,000 |
| Value Head | 1,089 | ~17,000 |
| **Total** | **134,548** | **~1,586,000** |

## Implementation Phases

### Phase 1+2 (combined): Scale d=128 + Latent Interface
- All projections: 30→128, 8→128, 16→128
- Transformer: d=128, 8 heads, 4 layers, QK-norm
- Remove external_cls_proj (128d feeds direct)
- Add LatentInterface(K=12, d=128, 8 heads)
- Zero-init Write output → identity at init
- Tail dropping: sample J̃ ∈ [4, 12] per training step
- Add aggregate token + agg_proj(16→128) + type_id=5
- GRU updated: GRUCell(128→64), proj(64→128)
- Rust: importance-based slot selection + aggregate computation
- SHM: +64 bytes aggregate, wider entity projections

### Phase 3 (later): Replace GRU with CfC, h_dim=256
Deferred until Phase 1+2 validated.

### Phase 4 (later): delta_t via SHM
Deferred until CfC validated.
