# Cast State Enrichment

## Problem

`CastState` currently stores minimal info:
```rust
pub struct CastState {
    pub target_id: u32,
    pub target_pos: Option<SimVec2>,
    pub remaining_ms: u32,
    pub kind: CastKind,  // Attack, Ability, Heal, Control, HeroAbility(usize)
}
```

The spatial system generates threat tokens for ground-targeted casts, but can
only guess at the area radius (hardcoded 2.0) because the actual ability
definition isn't preserved in the cast state.

## Proposed addition

Add fields to `CastState` that capture the ability's spatial and tactical
properties at cast start time:

```rust
pub struct CastState {
    pub target_id: u32,
    pub target_pos: Option<SimVec2>,
    pub remaining_ms: u32,
    pub kind: CastKind,
    // --- New fields ---
    /// Area of effect (from AbilityDef). None for single-target casts.
    pub area: Option<Area>,
    /// Ability index in the caster's ability list (for HeroAbility only).
    /// Allows downstream systems to look up the full AbilityDef if needed.
    pub ability_index: Option<usize>,
    /// Simplified effect category for AI consumption.
    pub effect_hint: CastEffectHint,
}

/// What this cast will do when it completes. Derived from AbilityDef effects
/// at cast start time.
pub enum CastEffectHint {
    Damage,
    Heal,
    CrowdControl,
    Obstacle,
    Buff,
    Summon,
    Mixed,
    Unknown,
}
```

## Where to populate

In `src/ai/core/hero/resolution.rs` (and the legacy cast paths in
`tick_systems.rs`), the `CastState` is constructed when a unit begins casting.
At that point, the `AbilityDef` is in scope — extract the area and classify
the effects before storing the cast state.

For `HeroAbility(index)`, the ability def is `unit.abilities[index].def`.
Scan `def.effects` for damage/heal/control/obstacle effects to derive
`CastEffectHint`.

## Impact on spatial tokens

With enriched cast state, `extract_threat_tokens` can:
- Use the actual `area` radius instead of guessing 2.0
- Include `effect_hint` in the threat token (damage vs heal vs obstacle)
- Distinguish "dodge this AoE" from "stand in this healing circle"

The threat token feature vector would gain 1-2 features:
- `effect_hint` encoded as a float (damage=0, heal=0.25, cc=0.5, obstacle=0.75, buff=1.0)
- `area_shape` (circle=0, cone=0.5, line=1.0) if needed

## Migration

- `CastState` gains 3 new fields, all with sensible defaults (area=None,
  ability_index=None, effect_hint=Unknown)
- Existing code that constructs CastState needs updating at ~5 call sites
- Serde: new fields are `#[serde(default)]` for backward compat with saved states
- No behavioral change — this is purely additive information
