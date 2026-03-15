# Grokking-Informed Training Plan: Ability Transformer

## Overview

Apply findings from Power et al. (2022) "Grokking: Generalization Beyond Overfitting
on Small Algorithmic Datasets" to the ability transformer training pipeline. The paper
studies small-dataset transformer training on structured algebraic tasks — a regime
that closely matches the ability transformer's setup (~270 real abilities, ~160-token
vocabulary, fixed compositional grammar). Left unaddressed, the default training
configuration will produce a model that memorizes training examples rather than
learning the grammar's compositional structure.

This plan covers three areas: training pipeline changes (optimizer, schedule, data
generation), transformer architecture adjustments, and evaluation/diagnostic tooling.

---

## 1. Problem Statement

### 1.1 Why Default Training Fails Here

The grokking paper's central observation: on small structured datasets, a transformer
achieves near-zero training loss (memorization) far earlier than it achieves
generalization. Validation loss rises — sometimes for 10,000–100,000 steps — before
a second descent to low validation loss. Standard early stopping terminates training
during this rise, returning a memorized model.

For the ability transformer this matters because:

- **Real ability dataset is tiny.** ~270 hero ability → oracle label pairs can be
  memorized by a 50K-parameter model in under 1,000 steps. The model doesn't need
  to learn *why* `DELIVER + CHAIN + BOUNCES_3` implies high urgency when enemies
  cluster — it can just remember the answer.
- **Runtime generalization is the actual goal.** Phase 3 (grammar-constrained
  generation) and future new heroes require the model to compose novel token
  sequences it has never seen. A memorized model fails here entirely.
- **The grammar is algebraic.** Ability token sequences obey the same compositional
  rules regardless of which specific ability they describe. This structure is learnable
  but only discoverable after the memorization phase ends.

### 1.2 What "Grokking" Means for This System

Grokking is the *goal state*, not the failure mode. A grokked model has internalized
the grammar's production rules — it understands that `SELF_AOE + DAMAGE + CIRCLE`
is structurally a burst AoE regardless of the specific numeric tokens. The failure
mode is stopping training before grokking occurs.

---

## 2. Training Pipeline Changes

### 2.1 Optimizer Configuration

Replace default Adam with AdamW, `weight_decay=1.0`. This is the paper's highest-impact
single change — it more than halves the data required for generalization in ablation
experiments. The β₂=0.98 value (vs. the default 0.999) is also from the paper's
Appendix A.1.2 and is non-obvious but supported by their ablations.

Apply this optimizer across all three training phases.

```python
# In pretrain_ability_transformer.py, finetune_decision.py, finetune_generation.py
import torch

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.98),    # β₂=0.98 — not the default 0.999
    weight_decay=1.0      # aggressively high; tune down if training destabilizes
)

# Linear warmup over first 10 steps (paper Appendix A.1.2)
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=10
)
```

**Tuning guidance**: if training loss fails to decrease in the first 500 steps, reduce
`weight_decay` to 0.1 and retry. The paper found that `weight_decay=1.0` works at
`lr=1e-3`; if `lr` is significantly lower than that, weight decay may need to scale
down proportionally.

### 2.2 Extended Training Schedules

Stop using loss convergence as the termination criterion. The paper shows validation
loss can rise for tens of thousands of steps before its second descent. Use oracle
*agreement accuracy* as the stopping criterion with a long patience window.

| Phase | Max Steps | Patience (on accuracy) | Checkpoint Metric |
|---|---|---|---|
| Phase 1 (pre-train) | 500,000 | 50,000 | masked token accuracy |
| Phase 2 (fine-tune) | 300,000 | 30,000 | oracle agreement % |
| Phase 3 (generation) | 500,000 | 50,000 | grammar-structure perplexity |

```python
# finetune_decision.py — extended training loop
best_val_acc = 0.0
patience = 0
MAX_STEPS = 300_000
MAX_PATIENCE = 30_000

for step in range(MAX_STEPS):
    loss = train_step(model, optimizer, warmup, batch)

    if step % 500 == 0:
        val_loss, oracle_agreement = evaluate(model, val_loader)

        # Log both — val_loss will rise; that is expected and not a stop signal
        log(step=step, train_loss=loss, val_loss=val_loss,
            oracle_agreement=oracle_agreement)

        if oracle_agreement > best_val_acc:
            best_val_acc = oracle_agreement
            patience = 0
            torch.save(model.state_dict(), f"checkpoints/best_agreement_{step}.pt")
        else:
            patience += 1

        if patience >= MAX_PATIENCE:
            print(f"Stopped at step {step}, best oracle agreement: {best_val_acc:.4f}")
            break
```

### 2.3 Synthetic Data Volume

Phase 1 pre-training uses grammar-generated synthetic abilities. The current plan
targets 5K–10K. This is likely too low: with 270 real abilities, the model can reach
near-zero training loss quickly and spend most of training in the memorization regime.
More synthetic data makes memorization harder and compresses the number of steps
required to grok.

Target 50K–100K grammar-generated abilities for Phase 1. Quality requirements are
low — syntactic validity is sufficient; the abilities don't need to be balanced or
playable. The generative grammar (§6 of the DSL plan) already guarantees syntactic
validity, so this is a matter of running more samples.

```python
# In generate.py — extend generation run
from dsl.generate import AbilityGenerator
from dsl.constraints import BalanceConstraints

generator = AbilityGenerator(seed=42)
abilities = []

# Increase from 10K to 75K
for i in range(75_000):
    # Use loose constraints — syntactic validity only, not balance
    ability = generator.sample(
        enforce_balance=False,    # skip power budget checks
        grammar_valid_only=True   # only requirement: parses cleanly
    )
    abilities.append(ability)

# Save to training/data/synthetic_abilities.jsonl
save_jsonl(abilities, "training/data/synthetic_abilities.jsonl")
print(f"Generated {len(abilities)} synthetic abilities")
```

**Coverage check**: after generation, verify that all major production rule branches
are represented — at minimum, each delivery type (projectile, chain, zone, channel,
tether, trap), each targeting type, each area shape, and at least 10 examples of
each condition type. Gaps in coverage are gaps in what the model can learn.

### 2.4 Minibatch Size

Use `batch_size = min(512, len(train_dataset) // 2)`. The paper found that
minibatch stochasticity is itself a regularization mechanism — full-batch training
delays generalization. For Phase 2 with ~270 real examples, this means batches of
~135, which forces repeated passes over the data with different gradient noise.

---

## 3. Transformer Architecture Changes

### 3.1 No Changes to Core Architecture

The 2-layer, 4-head, d=64 transformer from §7.5 of the DSL plan is appropriate. The
grokking paper's results were obtained on a comparable architecture (2-layer decoder
transformer, d=128, 4 heads). Changing architecture is not motivated by these findings.

### 3.2 [CLS] Initialization

Initialize the [CLS] token embedding to zero rather than random. The paper's
supplementary analysis suggests that learned pooling tokens benefit from a
well-conditioned starting point, particularly when weight decay is high (large
weight decay penalizes the [CLS] embedding's magnitude directly).

```rust
// In transformer.rs — CLS token init
fn new(vocab_size: usize, d_model: usize) -> Self {
    let mut model = TransformerEncoder { ... };

    // Zero-init CLS embedding (index 0 in vocab)
    model.token_embedding.weight[CLS_TOKEN_ID]
        .fill_(0.0);

    model
}
```

### 3.3 No Dropout

Counterintuitively, the grokking paper finds that dropout does not help and can
interfere with the weight-decay-driven generalization mechanism. If the current
transformer implementation includes dropout layers, disable them and rely on
weight decay alone for regularization.

```rust
// In transformer.rs — disable dropout
let config = TransformerConfig {
    n_layers: 2,
    n_heads: 4,
    d_model: 64,
    d_ff: 128,
    dropout: 0.0,   // disabled — use weight decay only
    ..Default::default()
};
```

---

## 4. Evaluation and Diagnostics

### 4.1 [CLS] Embedding Visualization

The paper (Section 3.4) uses t-SNE on learned embeddings to verify that the model
has transitioned from memorization to structural understanding. For the ability
transformer, the [CLS] pooled embedding is the diagnostic target.

A grokked model will produce [CLS] embeddings that cluster by semantically meaningful
groupings: `HINT_DAMAGE` abilities cluster away from `HINT_CC` and `HINT_HEAL`; zone
delivery abilities cluster separately from projectile abilities; conditional abilities
cluster by condition type.

```python
# training/diagnostics/visualize_embeddings.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_cls_embeddings(model, ability_dataset, step, color_by="hint"):
    """
    Extract [CLS] embeddings and visualize via t-SNE.
    Call every 10,000 steps. Structured clustering = grokking in progress.
    """
    embeddings = []
    labels = []

    model.eval()
    with torch.no_grad():
        for ability in ability_dataset:
            tokens = tokenize(ability)
            # Shape: [seq_len, d_model] — take position 0 ([CLS])
            hidden = model.encode(tokens)
            cls_emb = hidden[0].cpu().numpy()
            embeddings.append(cls_emb)

            if color_by == "hint":
                labels.append(ability.hint)
            elif color_by == "delivery":
                labels.append(ability.delivery_type or "none")
            elif color_by == "targeting":
                labels.append(ability.targeting)

    embeddings = np.array(embeddings)
    coords = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1),
                  random_state=42).fit_transform(embeddings)

    label_set = sorted(set(labels))
    colors = {l: i for i, l in enumerate(label_set)}
    c = [colors[l] for l in labels]

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    scatter = axes.scatter(coords[:, 0], coords[:, 1], c=c,
                           cmap='tab10', s=20, alpha=0.7)
    plt.colorbar(scatter, ax=axes, ticks=range(len(label_set)),
                 label=color_by)
    axes.set_title(f"[CLS] embeddings at step {step} (color={color_by})")
    plt.savefig(f"diagnostics/cls_embeddings_{color_by}_{step:07d}.png",
                dpi=150, bbox_inches='tight')
    plt.close()

# Call from training loop
if step % 10_000 == 0:
    visualize_cls_embeddings(model, val_abilities, step, color_by="hint")
    visualize_cls_embeddings(model, val_abilities, step, color_by="delivery")
```

**Interpreting the output**: early in training, embeddings will be scattered
randomly. Around the grokking transition, same-hint and same-delivery-type abilities
will begin clustering. If clusters never form, weight decay is too low or training
is under-run.

### 4.2 Oracle Agreement Tracking

Track oracle agreement as a percentage of validation examples where the model's
top-ranked target matches the oracle's top-ranked target. This is the primary
generalization metric — use it for early stopping, not BCE/CE loss.

```python
# training/eval/oracle_agreement.py

def compute_oracle_agreement(model, oracle, val_scenarios):
    """
    For each scenario in val_scenarios, compare model's top target
    to oracle's top target. Returns agreement percentage.
    """
    agreements = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for scenario in val_scenarios:
            ability_tokens = tokenize(scenario.ability)
            game_state = scenario.game_state_features

            model_urgency, model_target_scores = model(ability_tokens, game_state)
            oracle_urgency, oracle_target_scores = oracle(scenario)

            model_top = model_target_scores.argmax().item()
            oracle_top = oracle_target_scores.argmax().item()

            if model_top == oracle_top:
                agreements += 1
            total += 1

    return agreements / total if total > 0 else 0.0
```

### 4.3 Phase 3 Generalization Validation Set

For the grammar-constrained generation phase, oracle agreement isn't sufficient —
the model needs to generalize to ability *structures* it has never seen during
fine-tuning. Create a held-out generalization set before any training begins.

The held-out set should contain grammar-valid ability structures that are explicitly
excluded from both the real and synthetic training corpora. The measure is the
model's token-level perplexity on these held-out sequences — lower perplexity means
the model has internalized the grammar's rules, not just its training examples.

```python
# training/data/create_holdout.py

def create_structure_holdout(generator, n=500):
    """
    Generate abilities using grammar rule combinations that are deliberately
    withheld from training. For example:
      - tether delivery + chain area (uncommon combo)
      - recast 2 + zone delivery (uncommon combo)
      - passive with on_stack_reached trigger + condition
    """
    HOLDOUT_COMBINATIONS = [
        {"delivery": "tether", "area": "ring"},
        {"delivery": "channel", "condition": "or"},
        {"trigger": "on_stack_reached", "condition": "not"},
        {"recast": 2, "delivery": "zone"},
        # ... extend as needed
    ]

    holdout = []
    for combo in HOLDOUT_COMBINATIONS:
        samples = generator.sample_with_constraints(combo, n=n // len(HOLDOUT_COMBINATIONS))
        holdout.extend(samples)

    save_jsonl(holdout, "training/data/holdout_structures.jsonl")
    print(f"Created {len(holdout)} holdout ability structures")
    # IMPORTANT: add holdout structures to training exclusion filter
    return [ability.structure_hash() for ability in holdout]
```

### 4.4 Training Metrics Dashboard

All training scripts should log the following metrics to a common format (CSV or
wandb) to enable cross-phase comparison:

| Metric | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| `train_loss` | ✓ | ✓ | ✓ |
| `val_loss` | ✓ | ✓ | ✓ |
| `masked_token_acc` | ✓ | — | — |
| `oracle_agreement` | — | ✓ | — |
| `holdout_perplexity` | — | — | ✓ |
| `weight_norm` | ✓ | ✓ | ✓ |
| `gradient_norm` | ✓ | ✓ | ✓ |

`weight_norm` is particularly useful: a decreasing weight norm during the memorization
phase (weight decay taking effect) followed by a stabilization or slight increase
is a reliable early signal that the grokking transition is occurring.

---

## 5. New Files

```
training/
    pretrain_ability_transformer.py     — updated with AdamW, extended schedule
    finetune_decision.py                — updated with oracle agreement stopping
    finetune_generation.py              — updated with holdout perplexity stopping
    export_weights.py                   — unchanged

    data/
        generate_synthetic.py           — updated to produce 75K abilities
        create_holdout.py               — NEW: holdout structure set creation
        synthetic_abilities.jsonl       — generated (not committed)
        holdout_structures.jsonl        — generated (not committed)
        holdout_hashes.txt              — committed: exclusion filter for training

    eval/
        oracle_agreement.py             — NEW: oracle agreement metric
        holdout_perplexity.py           — NEW: generalization perplexity metric

    diagnostics/
        visualize_embeddings.py         — NEW: t-SNE on [CLS] embeddings
        plot_training_curves.py         — NEW: val_loss vs oracle_agreement overlay
```

---

## 6. Implementation Steps

| Step | What | Details |
|---|---|---|
| 1 | **Create holdout set** | Run `create_holdout.py` before any training. Commit `holdout_hashes.txt` as the exclusion filter. Cannot be done after training starts. |
| 2 | **Generate synthetic data** | Update `generate_synthetic.py` to produce 75K abilities with `enforce_balance=False`. Verify grammar coverage across all delivery, targeting, area, and condition types. |
| 3 | **Update optimizer** | Replace Adam with AdamW `(lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0)` in all three training scripts. Add 10-step linear warmup. |
| 4 | **Disable dropout** | Set `dropout=0.0` in `TransformerConfig`. Add [CLS] zero-init in `transformer.rs`. |
| 5 | **Update training loops** | Replace loss-based early stopping with accuracy/perplexity-based stopping. Set max steps and patience per §2.2 table. |
| 6 | **Add oracle agreement metric** | Implement `eval/oracle_agreement.py`. Wire into Phase 2 evaluation loop at every 500-step checkpoint. |
| 7 | **Add holdout perplexity metric** | Implement `eval/holdout_perplexity.py`. Wire into Phase 3 evaluation loop. |
| 8 | **Add diagnostics** | Implement `diagnostics/visualize_embeddings.py`. Call every 10K steps in Phase 1 and Phase 2 training loops. |
| 9 | **Baseline run** | Run Phase 1 with default settings (Adam, weight_decay=0.01) and record metrics. This is the before-state for comparison. |
| 10 | **Updated Phase 1 run** | Run Phase 1 with all changes. Compare masked token accuracy curve and [CLS] embedding visualization against baseline. |
| 11 | **Updated Phase 2 run** | Run Phase 2 fine-tuning. Monitor oracle agreement — expect it to plateau or dip before rising. Do not stop during plateau. |
| 12 | **Updated Phase 3 run** | Run Phase 3 generation fine-tuning. Monitor holdout perplexity as primary stopping criterion. |
| 13 | **A/B comparison** | Run existing MLP pipeline and new transformer pipeline against the same oracle validation set. Report oracle agreement delta. |

---

## 7. Key Design Decisions

1. **Weight decay is the primary lever.** The paper's ablation is unambiguous: AdamW
   with λ=1.0 outperforms every other regularization strategy tested, including
   dropout, gradient noise, and weight noise. This is the change most likely to move
   the needle.

2. **Oracle agreement is the only meaningful stopping criterion for Phase 2.**
   BCE loss and CE loss are necessary for gradient signal but are poor proxies for
   the actual goal. A model can achieve low training loss by memorizing. Oracle
   agreement on a held-out set cannot be gamed by memorization.

3. **Synthetic data serves Phase 1 only.** Grammar-generated abilities are used for
   the masked token pre-training phase, where syntactic coverage matters more than
   semantic quality. Phase 2 and Phase 3 use only real abilities + oracle labels,
   because the generative grammar cannot produce oracle ground truth.

4. **No dropout.** Dropout and aggressive weight decay are partially redundant as
   regularizers and the paper finds they don't compound well. Choose one. Weight
   decay is chosen because it has stronger theoretical grounding for driving the
   model toward flat minima (which the paper connects to generalization via a
   Spearman correlation of −0.80 between sharpness and val accuracy).

5. **Holdout set must be created first.** If holdout structures bleed into the
   synthetic training corpus, Phase 3 generalization metrics are meaningless.
   `holdout_hashes.txt` acts as the exclusion filter and is the only artifact that
   must be committed before any training run begins.
