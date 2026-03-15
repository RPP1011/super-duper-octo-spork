# V6 Data Validation Probes

Before training any stage, run quick probes to verify the data provides signal.
Most past failures were data issues, not approach issues.

## General Probes (run on any new dataset)

### 1. Label distribution check
- Histogram of all target labels / regression targets
- Flag: >80% single class, near-zero variance, bimodal when expecting smooth
- Flag: NaN/inf in any feature or label

### 2. Feature-label correlation scan
- Per-feature Spearman rank correlation with each target
- Flag: all correlations < 0.05 (features carry no signal)
- Flag: single feature correlation > 0.95 (label is trivially derivable, likely leak)

### 3. Memorization baseline
- Train a tiny model (single linear layer) on 100 samples for 1000 steps
- Must overfit to near-zero loss — if it can't, labels are noisy or features are disconnected
- Then check held-out: if linear model gets >70% of the way to full model, the task is too easy or features are already sufficient without the architecture

### 4. Duplicate / near-duplicate check
- Hash feature vectors, count exact duplicates
- If >10% duplicates with different labels, the labeling process is noisy
- Check train/val overlap — leaks here inflate val metrics silently

### 5. Temporal sanity (for sequential data)
- Plot feature values across consecutive ticks for a single episode
- Should see smooth trajectories (HP declining, positions shifting)
- Jumps or resets mid-episode indicate episode boundary bugs or tick sampling errors

## Stage-Specific Probes

### Stage 0a: Fight Outcome Prediction
- Verify attrition_ratio has reasonable spread (not all 1.0 or all 0.0)
- Check that tick sampling covers early/mid/late fight (not biased to first 5 ticks)
- Verify policy mixture actually produces diverse outcomes (random policy should lose more)
- Spot-check: a fight at tick 1 with full HP on both sides should have ~0.5 attrition prediction

### Stage 0c: Temporal Pretraining
- Verify contiguous windows are truly contiguous (no gaps in tick indices)
- Check that final-tick value prediction differs from single-tick (if identical, temporal signal is absent)

### Stage 0e: Combat Pointer BC
- Verify target indices map to actual enemy entities (not padding slots or allies)
- Check that GOAP target decisions have variety (not always targeting slot 0)
- Verify entity slot ordering matches between recording time and training time

## SHM Data Probes (before GPU inference)

### Threat token validation
- Log a few ticks of raw threat token data from SHM
- Verify relative features (dx, dy, distance) are actually relative to acting unit
- Verify 10-feature format matches what gpu_inference_server.py expects
- Check that threat_mask correctly marks padding vs real threats

### Spatial summary validation
- For a known corridor room, verify visible_corner_count > 0 and min_passage_width < 0.5
- For a known open arena, verify visible_corner_count is low and avg_passage_width is high
- Zero spatial features should only occur for units with no visible corners (rare)

### Aggregate token validation
- In a 3v3 fight, aggregate should show n_truncated = 0 (nothing was dropped)
- In a 12v4 fight, aggregate should show n_enemies_truncated > 0
- Centroids should be in valid room coordinate range
