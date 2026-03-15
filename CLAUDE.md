# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test                     # All tests
cargo test ai::core::tests     # Tests in a specific module
cargo test shield_absorbs      # Single test by name substring
cargo test -- --nocapture      # Show println output
cargo test -- --test-threads=1 # Serial execution (for determinism tests)
```

### CLI (xtask)

```bash
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml
cargo run --bin xtask -- scenario bench scenarios/
cargo run --bin xtask -- scenario oracle eval scenarios/
cargo run --bin xtask -- scenario oracle dataset scenarios/ --output generated/oracle.jsonl
cargo run --bin xtask -- scenario oracle transformer-play PATH --weights W
cargo run --bin xtask -- scenario oracle transformer-rl generate scenarios/
```

### Python Training

Use `uv run --with numpy --with torch` for Python training scripts (no virtualenv needed):
```bash
uv run --with numpy --with torch python training/train_rl.py ...
uv run --with numpy --with torch python training/pretrain_entity.py ...
```

## Architecture

### Three-Layer Simulation

```
Campaign (turn-based overworld) → Mission (multi-room dungeons) → Combat (100ms fixed-tick deterministic sim)
```

The combat layer is the core of the AI/ML work. Everything runs through `step(state, intents, dt_ms) → (state, events)` in `src/ai/core/simulation.rs`.

### Key Module Map

- **`src/ai/core/`** — Simulation engine: `SimState`, `UnitState`, `step()`, damage calc, effect application
- **`src/ai/effects/`** — Data-driven ability system with DSL parser (`.ability` files). Five composable dimensions: Effect (what), Area (where), Delivery (how), Trigger (when), Tags (power levels)
- **`src/ai/effects/dsl/`** — winnow-based parser for ability DSL: `parser.rs` → `lower.rs` (AST→AbilityDef)
- **`src/ai/core/ability_eval/`** — Neural ability evaluator (urgency interrupt layer, fires when urgency > 0.4)
- **`src/ai/core/ability_transformer/`** — Grokking-based transformer for ability decisions, with cross-attention over entity tokens
- **`src/ai/core/self_play/`** — RL policy learning (REINFORCE + PPO, pointer action space)
- **`src/ai/squad/`** — Squad-level AI: personality profiles, formation modes, intent generation
- **`src/ai/pathing/`** — Grid navigation, pathfinding, cover
- **`src/scenario/`** — Scenario config (TOML), runner, coverage-driven generation
- **`src/bin/xtask/`** — CLI task runner (scenarios, oracle, map gen)
- **`src/bin/sim_bridge/`** — Headless sim for external agents via NDJSON protocol
- **`training/`** — Python model training: `model.py` (architectures), `train_rl.py`/`train_rl_v3.py` (RL), `pretrain_entity.py`, `finetune_decision.py`

### Determinism Contract

All simulation randomness flows through `SimState.rng_state` via `next_rand_u32()`. Never use `thread_rng()` or any external RNG in simulation code. Unit processing order is shuffled per tick to prevent first-mover bias. Tests in `src/ai/core/tests/determinism.rs` verify reproducibility.

### Effect System

Effects are plain data structs dispatched via pattern matching (no closures). The pipeline:
1. `.ability` DSL file → parser (`winnow`) → AST
2. AST → `lower.rs` → `AbilityDef` (with `Effect`, `Area`, `Delivery`, conditions)
3. At runtime: `apply_effect.rs` / `apply_effect_ext.rs` dispatch effects onto `SimState`

### AI Decision Pipeline

Intent generation flows through layers, each can override:
1. **Squad AI** (`squad/intents.rs`): team-wide personality-driven behavior
2. **Ability Evaluator** (optional): neural urgency interrupt for ability usage
3. **Transformer** (optional): cross-attention decision head over entity + ability tokens
4. **Control AI** (optional): hard CC timing coordination / GOAP overrides

### Workspace

Two crates: root (`bevy_game`) and `crates/ability_operator` (behavioral embeddings for abilities).

### Hero Templates

Defined in `assets/hero_templates/` as hybrid TOML (stats) + `.ability` DSL (abilities). The `.ability` files are the source of truth for ability definitions. 27 base heroes + 172 LoL hero imports in `assets/lol_heroes/`.

### Test Helpers

In `src/ai/core/tests/mod.rs`: `hero_unit(id, team, pos)`, `make_state(units, seed)` for creating deterministic test fixtures. Tests assert on `SimEvent` logs and unit state after `step()`.
