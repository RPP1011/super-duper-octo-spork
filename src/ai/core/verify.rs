use std::collections::HashSet;

use crate::ai::effects::StatusKind;

use super::types::SimState;

/// A specific invariant violation detected during a simulation tick.
#[derive(Debug, Clone, PartialEq)]
pub enum Violation {
    /// Unit HP exceeds its max_hp.
    HpExceedsMax { unit_id: u32, hp: i32, max_hp: i32 },
    /// Alive unit has non-positive HP (should have been marked dead).
    AliveWithNonPositiveHp { unit_id: u32, hp: i32 },
    /// Dead unit (hp <= 0) still has an active cast.
    DeadUnitCasting { unit_id: u32 },
    /// Dead unit still has an active channel.
    DeadUnitChanneling { unit_id: u32 },
    /// Shield HP is negative.
    NegativeShield { unit_id: u32, shield_hp: i32 },
    /// Resource exceeds max_resource.
    ResourceExceedsMax { unit_id: u32, resource: i32, max_resource: i32 },
    /// Position contains NaN or infinity.
    InvalidPosition { unit_id: u32, x: f32, y: f32 },
    /// Duplicate unit IDs detected.
    DuplicateUnitId { unit_id: u32 },
    /// Zone references a source unit that doesn't exist.
    ZoneOrphanedSource { zone_id: u32, source_id: u32 },
    /// Tether references a unit that doesn't exist.
    TetherOrphanedUnit { source_id: u32, target_id: u32 },
    /// Move speed is negative.
    NegativeMoveSpeed { unit_id: u32, speed: f32 },
    /// Cooldown remaining exceeds the base cooldown value.
    CooldownExceedsBase { unit_id: u32, field: &'static str, remaining: u32, base: u32 },
    /// Negative resource value.
    NegativeResource { unit_id: u32, resource: i32 },

    // --- New runtime verification checks ---

    /// Unit is simultaneously casting and channeling.
    CastingAndChanneling { unit_id: u32 },
    /// Dead unit still has active crowd-control timer.
    DeadUnitControlled { unit_id: u32, control_remaining_ms: u32 },
    /// Ability slot charges exceed max_charges.
    AbilityChargesExceedMax { unit_id: u32, ability_index: usize, charges: u32, max_charges: u32 },
    /// Ability has an active morph timer but no saved base definition.
    MorphWithoutBaseDef { unit_id: u32, ability_index: usize },
    /// Recast window is active but no recasts remain.
    RecastWindowWithoutRecasts { unit_id: u32, ability_index: usize },
    /// Directed summon without an owner.
    DirectedWithoutOwner { unit_id: u32 },
    /// Status effect references a unit that doesn't exist.
    StatusEffectOrphanedRef { unit_id: u32, kind: &'static str, referenced_id: u32 },
    /// Status effect references a dead unit (partner/protector/host is dead).
    StatusEffectDeadRef { unit_id: u32, kind: &'static str, referenced_id: u32 },
    /// Cover bonus is outside the valid range [0.0, 1.0].
    CoverBonusOutOfRange { unit_id: u32, cover_bonus: i32 },
    /// Zone remaining duration is zero but zone still exists.
    ZoneExpired { zone_id: u32 },
    /// Tether remaining duration is zero but tether still exists.
    TetherExpired { source_id: u32, target_id: u32 },

    // --- Projectile invariants ---

    /// Projectile source unit doesn't exist.
    ProjectileOrphanedSource { projectile_index: usize, source_id: u32 },
    /// Projectile position contains NaN or infinity.
    ProjectileInvalidPosition { projectile_index: usize, x: f32, y: f32 },
    /// Projectile speed is zero or negative.
    ProjectileInvalidSpeed { projectile_index: usize, speed: f32 },
    /// Projectile has traveled past its max distance.
    ProjectileOvershot { projectile_index: usize, distance_traveled_x100: i32, max_distance_x100: i32 },

    // --- Status effect value-range checks ---

    /// Status effect numeric parameter is outside expected bounds.
    StatusEffectValueOutOfRange { unit_id: u32, kind: &'static str, value_x1000: i32 },
    /// Stacks count exceeds max_stacks.
    StatusEffectStacksExceedMax { unit_id: u32, count: u32, max_stacks: u32 },
    /// Summon owner doesn't exist in the unit list.
    SummonOwnerMissing { unit_id: u32, owner_id: u32 },
    /// Negative armor or magic resist value.
    NegativeResist { unit_id: u32, field: &'static str, value_x100: i32 },
}

/// Result of running verification checks on a simulation state.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub tick: u64,
    pub violations: Vec<Violation>,
}

impl VerificationReport {
    pub fn is_ok(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
}

/// Run all invariant checks against the current simulation state.
/// Call this after each `step()` to detect violations early.
pub fn verify_tick(state: &SimState) -> VerificationReport {
    let mut violations = Vec::new();
    let unit_ids: HashSet<u32> = state.units.iter().map(|u| u.id).collect();

    // Build alive-unit set for status-effect reference checks
    let alive_ids: HashSet<u32> = state.units.iter()
        .filter(|u| u.hp > 0)
        .map(|u| u.id)
        .collect();

    // Check for duplicate unit IDs
    if unit_ids.len() != state.units.len() {
        let mut seen = HashSet::new();
        for u in &state.units {
            if !seen.insert(u.id) {
                violations.push(Violation::DuplicateUnitId { unit_id: u.id });
            }
        }
    }

    for unit in &state.units {
        // HP bounds
        if unit.hp > unit.max_hp {
            violations.push(Violation::HpExceedsMax {
                unit_id: unit.id,
                hp: unit.hp,
                max_hp: unit.max_hp,
            });
        }

        let alive = unit.hp > 0;

        // Dead unit still casting
        if !alive && unit.casting.is_some() {
            violations.push(Violation::DeadUnitCasting { unit_id: unit.id });
        }

        // Dead unit still channeling
        if !alive && unit.channeling.is_some() {
            violations.push(Violation::DeadUnitChanneling { unit_id: unit.id });
        }

        // Dead unit still under crowd control
        if !alive && unit.control_remaining_ms > 0 {
            violations.push(Violation::DeadUnitControlled {
                unit_id: unit.id,
                control_remaining_ms: unit.control_remaining_ms,
            });
        }

        // Casting and channeling at the same time
        if unit.casting.is_some() && unit.channeling.is_some() {
            violations.push(Violation::CastingAndChanneling { unit_id: unit.id });
        }

        // Negative shield
        if unit.shield_hp < 0 {
            violations.push(Violation::NegativeShield {
                unit_id: unit.id,
                shield_hp: unit.shield_hp,
            });
        }

        // Resource bounds
        if unit.max_resource > 0 && unit.resource > unit.max_resource {
            violations.push(Violation::ResourceExceedsMax {
                unit_id: unit.id,
                resource: unit.resource,
                max_resource: unit.max_resource,
            });
        }
        if unit.resource < 0 {
            violations.push(Violation::NegativeResource {
                unit_id: unit.id,
                resource: unit.resource,
            });
        }

        // Position validity (NaN / Inf)
        if !unit.position.x.is_finite() || !unit.position.y.is_finite() {
            violations.push(Violation::InvalidPosition {
                unit_id: unit.id,
                x: unit.position.x,
                y: unit.position.y,
            });
        }

        // Negative move speed
        if unit.move_speed_per_sec < 0.0 {
            violations.push(Violation::NegativeMoveSpeed {
                unit_id: unit.id,
                speed: unit.move_speed_per_sec,
            });
        }

        // Cover bonus bounds
        if unit.cover_bonus < 0.0 || unit.cover_bonus > 1.0 {
            violations.push(Violation::CoverBonusOutOfRange {
                unit_id: unit.id,
                cover_bonus: (unit.cover_bonus * 1000.0) as i32,
            });
        }

        // Directed summon must have an owner
        if unit.directed && unit.owner_id.is_none() {
            violations.push(Violation::DirectedWithoutOwner { unit_id: unit.id });
        }

        // Summon owner must exist
        if let Some(owner_id) = unit.owner_id {
            if !unit_ids.contains(&owner_id) {
                violations.push(Violation::SummonOwnerMissing {
                    unit_id: unit.id,
                    owner_id,
                });
            }
        }

        // Negative armor / magic resist
        if unit.armor < 0.0 {
            violations.push(Violation::NegativeResist {
                unit_id: unit.id,
                field: "armor",
                value_x100: (unit.armor * 100.0) as i32,
            });
        }
        if unit.magic_resist < 0.0 {
            violations.push(Violation::NegativeResist {
                unit_id: unit.id,
                field: "magic_resist",
                value_x100: (unit.magic_resist * 100.0) as i32,
            });
        }

        // Cooldown sanity: remaining should not exceed base
        if unit.attack_cooldown_ms > 0
            && unit.cooldown_remaining_ms > unit.attack_cooldown_ms
        {
            violations.push(Violation::CooldownExceedsBase {
                unit_id: unit.id,
                field: "attack",
                remaining: unit.cooldown_remaining_ms,
                base: unit.attack_cooldown_ms,
            });
        }
        if unit.ability_cooldown_ms > 0
            && unit.ability_cooldown_remaining_ms > unit.ability_cooldown_ms
        {
            violations.push(Violation::CooldownExceedsBase {
                unit_id: unit.id,
                field: "ability",
                remaining: unit.ability_cooldown_remaining_ms,
                base: unit.ability_cooldown_ms,
            });
        }
        if unit.heal_cooldown_ms > 0
            && unit.heal_cooldown_remaining_ms > unit.heal_cooldown_ms
        {
            violations.push(Violation::CooldownExceedsBase {
                unit_id: unit.id,
                field: "heal",
                remaining: unit.heal_cooldown_remaining_ms,
                base: unit.heal_cooldown_ms,
            });
        }
        if unit.control_cooldown_ms > 0
            && unit.control_cooldown_remaining_ms > unit.control_cooldown_ms
        {
            violations.push(Violation::CooldownExceedsBase {
                unit_id: unit.id,
                field: "control",
                remaining: unit.control_cooldown_remaining_ms,
                base: unit.control_cooldown_ms,
            });
        }

        // --- Hero ability slot invariants ---
        for (i, slot) in unit.abilities.iter().enumerate() {
            // Charges cannot exceed max_charges
            if slot.def.max_charges > 0 && slot.charges > slot.def.max_charges {
                violations.push(Violation::AbilityChargesExceedMax {
                    unit_id: unit.id,
                    ability_index: i,
                    charges: slot.charges,
                    max_charges: slot.def.max_charges,
                });
            }

            // Active morph must have a saved base definition
            if slot.morph_remaining_ms > 0 && slot.base_def.is_none() {
                violations.push(Violation::MorphWithoutBaseDef {
                    unit_id: unit.id,
                    ability_index: i,
                });
            }

            // Recast window active implies recasts remaining
            if slot.recast_window_remaining_ms > 0 && slot.recasts_remaining == 0 {
                violations.push(Violation::RecastWindowWithoutRecasts {
                    unit_id: unit.id,
                    ability_index: i,
                });
            }
        }

        // --- Status effect reference validation ---
        for effect in &unit.status_effects {
            match &effect.kind {
                StatusKind::Duel { partner_id } => {
                    if !unit_ids.contains(partner_id) {
                        violations.push(Violation::StatusEffectOrphanedRef {
                            unit_id: unit.id,
                            kind: "Duel",
                            referenced_id: *partner_id,
                        });
                    }
                }
                StatusKind::Taunt { taunter_id } => {
                    if !unit_ids.contains(taunter_id) {
                        violations.push(Violation::StatusEffectOrphanedRef {
                            unit_id: unit.id,
                            kind: "Taunt",
                            referenced_id: *taunter_id,
                        });
                    } else if !alive_ids.contains(taunter_id) {
                        violations.push(Violation::StatusEffectDeadRef {
                            unit_id: unit.id,
                            kind: "Taunt",
                            referenced_id: *taunter_id,
                        });
                    }
                }
                StatusKind::Link { partner_id, .. } => {
                    if !unit_ids.contains(partner_id) {
                        violations.push(Violation::StatusEffectOrphanedRef {
                            unit_id: unit.id,
                            kind: "Link",
                            referenced_id: *partner_id,
                        });
                    } else if !alive_ids.contains(partner_id) {
                        violations.push(Violation::StatusEffectDeadRef {
                            unit_id: unit.id,
                            kind: "Link",
                            referenced_id: *partner_id,
                        });
                    }
                }
                StatusKind::Redirect { protector_id, .. } => {
                    if !unit_ids.contains(protector_id) {
                        violations.push(Violation::StatusEffectOrphanedRef {
                            unit_id: unit.id,
                            kind: "Redirect",
                            referenced_id: *protector_id,
                        });
                    } else if !alive_ids.contains(protector_id) {
                        violations.push(Violation::StatusEffectDeadRef {
                            unit_id: unit.id,
                            kind: "Redirect",
                            referenced_id: *protector_id,
                        });
                    }
                }
                StatusKind::Attached { host_id } => {
                    if !unit_ids.contains(host_id) {
                        violations.push(Violation::StatusEffectOrphanedRef {
                            unit_id: unit.id,
                            kind: "Attached",
                            referenced_id: *host_id,
                        });
                    } else if !alive_ids.contains(host_id) {
                        violations.push(Violation::StatusEffectDeadRef {
                            unit_id: unit.id,
                            kind: "Attached",
                            referenced_id: *host_id,
                        });
                    }
                }
                StatusKind::Charm { .. } => {
                    if !unit_ids.contains(&effect.source_id) {
                        violations.push(Violation::StatusEffectOrphanedRef {
                            unit_id: unit.id,
                            kind: "Charm",
                            referenced_id: effect.source_id,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // --- Projectile invariants ---
    for (i, proj) in state.projectiles.iter().enumerate() {
        if !unit_ids.contains(&proj.source_id) {
            violations.push(Violation::ProjectileOrphanedSource {
                projectile_index: i,
                source_id: proj.source_id,
            });
        }
        if !proj.position.x.is_finite() || !proj.position.y.is_finite() {
            violations.push(Violation::ProjectileInvalidPosition {
                projectile_index: i,
                x: proj.position.x,
                y: proj.position.y,
            });
        }
        if proj.speed <= 0.0 {
            violations.push(Violation::ProjectileInvalidSpeed {
                projectile_index: i,
                speed: proj.speed,
            });
        }
        if proj.max_travel_distance > 0.0
            && proj.distance_traveled > proj.max_travel_distance + 1.0
        {
            violations.push(Violation::ProjectileOvershot {
                projectile_index: i,
                distance_traveled_x100: (proj.distance_traveled * 100.0) as i32,
                max_distance_x100: (proj.max_travel_distance * 100.0) as i32,
            });
        }
    }

    // --- Status effect value-range checks ---
    for unit in &state.units {
        for effect in &unit.status_effects {
            match &effect.kind {
                StatusKind::Slow { factor } => {
                    if *factor < 0.0 || *factor > 1.0 {
                        violations.push(Violation::StatusEffectValueOutOfRange {
                            unit_id: unit.id,
                            kind: "Slow",
                            value_x1000: (*factor * 1000.0) as i32,
                        });
                    }
                }
                StatusKind::Blind { miss_chance } => {
                    if *miss_chance < 0.0 || *miss_chance > 1.0 {
                        violations.push(Violation::StatusEffectValueOutOfRange {
                            unit_id: unit.id,
                            kind: "Blind",
                            value_x1000: (*miss_chance * 1000.0) as i32,
                        });
                    }
                }
                StatusKind::Stacks { count, max_stacks, .. } => {
                    if *max_stacks > 0 && *count > *max_stacks {
                        violations.push(Violation::StatusEffectStacksExceedMax {
                            unit_id: unit.id,
                            count: *count,
                            max_stacks: *max_stacks,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // Zone source validation
    for zone in &state.zones {
        if !unit_ids.contains(&zone.source_id) {
            violations.push(Violation::ZoneOrphanedSource {
                zone_id: zone.id,
                source_id: zone.source_id,
            });
        }
        // Expired zones should have been removed
        if zone.remaining_ms == 0 && !zone.trigger_on_enter {
            violations.push(Violation::ZoneExpired { zone_id: zone.id });
        }
    }

    // Tether endpoint validation
    for tether in &state.tethers {
        if !unit_ids.contains(&tether.source_id) || !unit_ids.contains(&tether.target_id) {
            violations.push(Violation::TetherOrphanedUnit {
                source_id: tether.source_id,
                target_id: tether.target_id,
            });
        }
        // Expired tethers should have been removed
        if tether.remaining_ms == 0 {
            violations.push(Violation::TetherExpired {
                source_id: tether.source_id,
                target_id: tether.target_id,
            });
        }
    }

    VerificationReport {
        tick: state.tick,
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::simulation::sample_duel_state;
    use crate::ai::core::types::*;
    use crate::ai::effects::{AbilitySlot, AbilityDef, ActiveStatusEffect, Stacking};

    fn make_ability_def() -> AbilityDef {
        AbilityDef::default()
    }

    #[test]
    fn clean_state_passes_verification() {
        let state = sample_duel_state(42);
        let report = verify_tick(&state);
        assert!(report.is_ok(), "Expected no violations, got: {:?}", report.violations);
    }

    #[test]
    fn detects_hp_exceeds_max() {
        let mut state = sample_duel_state(42);
        state.units[0].hp = 150; // max_hp is 100
        let report = verify_tick(&state);
        assert!(!report.is_ok());
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::HpExceedsMax { unit_id: 1, hp: 150, max_hp: 100 }
        )));
    }

    #[test]
    fn detects_dead_unit_casting() {
        let mut state = sample_duel_state(42);
        state.units[0].hp = 0;
        state.units[0].casting = Some(CastState {
            target_id: 2,
            target_pos: None,
            remaining_ms: 100,
            kind: CastKind::Attack,
        });
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::DeadUnitCasting { unit_id: 1 }
        )));
    }

    #[test]
    fn detects_negative_shield() {
        let mut state = sample_duel_state(42);
        state.units[0].shield_hp = -5;
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::NegativeShield { unit_id: 1, shield_hp: -5 }
        )));
    }

    #[test]
    fn detects_invalid_position() {
        let mut state = sample_duel_state(42);
        state.units[0].position = sim_vec2(f32::NAN, 0.0);
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::InvalidPosition { unit_id: 1, .. }
        )));
    }

    #[test]
    fn detects_duplicate_unit_ids() {
        let mut state = sample_duel_state(42);
        state.units[1].id = 1; // same as units[0]
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::DuplicateUnitId { unit_id: 1 }
        )));
    }

    #[test]
    fn detects_resource_exceeds_max() {
        let mut state = sample_duel_state(42);
        state.units[0].max_resource = 100;
        state.units[0].resource = 120;
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::ResourceExceedsMax { unit_id: 1, resource: 120, max_resource: 100 }
        )));
    }

    #[test]
    fn detects_negative_resource() {
        let mut state = sample_duel_state(42);
        state.units[0].resource = -10;
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::NegativeResource { unit_id: 1, resource: -10 }
        )));
    }

    #[test]
    fn full_simulation_passes_verification() {
        use crate::ai::core::simulation::{step, sample_duel_script};
        let mut state = sample_duel_state(42);
        let script = sample_duel_script(200);
        for tick in 0..200 {
            let intents = &script[tick];
            let (new_state, _events) = step(state, intents, FIXED_TICK_MS);
            state = new_state;
            let report = verify_tick(&state);
            assert!(
                report.is_ok(),
                "Violation at tick {}: {:?}",
                report.tick,
                report.violations
            );
        }
    }

    #[test]
    fn detects_cooldown_exceeds_base() {
        let mut state = sample_duel_state(42);
        state.units[0].cooldown_remaining_ms = 5000; // base is 700
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::CooldownExceedsBase { unit_id: 1, field: "attack", .. }
        )));
    }

    // --- New runtime verification tests ---

    #[test]
    fn detects_casting_and_channeling() {
        let mut state = sample_duel_state(42);
        state.units[0].casting = Some(CastState {
            target_id: 2,
            target_pos: None,
            remaining_ms: 100,
            kind: CastKind::Attack,
        });
        state.units[0].channeling = Some(ChannelState {
            ability_index: 0,
            target_id: 2,
            target_pos: None,
            remaining_ms: 500,
            tick_interval_ms: 100,
            tick_elapsed_ms: 0,
        });
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::CastingAndChanneling { unit_id: 1 }
        )));
    }

    #[test]
    fn detects_dead_unit_controlled() {
        let mut state = sample_duel_state(42);
        state.units[0].hp = 0;
        state.units[0].control_remaining_ms = 500;
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::DeadUnitControlled { unit_id: 1, control_remaining_ms: 500 }
        )));
    }

    #[test]
    fn detects_ability_charges_exceed_max() {
        let mut state = sample_duel_state(42);
        let mut def = make_ability_def();
        def.max_charges = 3;
        let mut slot = AbilitySlot::new(def);
        slot.charges = 5;
        state.units[0].abilities.push(slot);
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::AbilityChargesExceedMax { unit_id: 1, ability_index: 0, charges: 5, max_charges: 3 }
        )));
    }

    #[test]
    fn detects_morph_without_base_def() {
        let mut state = sample_duel_state(42);
        let def = make_ability_def();
        let mut slot = AbilitySlot::new(def);
        slot.morph_remaining_ms = 1000;
        // base_def is None by default
        state.units[0].abilities.push(slot);
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::MorphWithoutBaseDef { unit_id: 1, ability_index: 0 }
        )));
    }

    #[test]
    fn detects_recast_window_without_recasts() {
        let mut state = sample_duel_state(42);
        let def = make_ability_def();
        let mut slot = AbilitySlot::new(def);
        slot.recast_window_remaining_ms = 500;
        slot.recasts_remaining = 0;
        state.units[0].abilities.push(slot);
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::RecastWindowWithoutRecasts { unit_id: 1, ability_index: 0 }
        )));
    }

    #[test]
    fn detects_directed_without_owner() {
        let mut state = sample_duel_state(42);
        state.units[0].directed = true;
        state.units[0].owner_id = None;
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::DirectedWithoutOwner { unit_id: 1 }
        )));
    }

    #[test]
    fn detects_status_effect_orphaned_link() {
        let mut state = sample_duel_state(42);
        state.units[0].status_effects.push(ActiveStatusEffect {
            kind: StatusKind::Link { partner_id: 999, share_percent: 0.5 },
            source_id: 2,
            remaining_ms: 1000,
            tags: Default::default(),
            stacking: Stacking::default(),
        });
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::StatusEffectOrphanedRef { unit_id: 1, kind: "Link", referenced_id: 999 }
        )));
    }

    #[test]
    fn detects_status_effect_dead_redirect_protector() {
        let mut state = sample_duel_state(42);
        state.units[1].hp = 0; // kill protector
        state.units[0].status_effects.push(ActiveStatusEffect {
            kind: StatusKind::Redirect { protector_id: 2, charges: 3 },
            source_id: 2,
            remaining_ms: 1000,
            tags: Default::default(),
            stacking: Stacking::default(),
        });
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::StatusEffectDeadRef { unit_id: 1, kind: "Redirect", referenced_id: 2 }
        )));
    }

    #[test]
    fn detects_cover_bonus_out_of_range() {
        let mut state = sample_duel_state(42);
        state.units[0].cover_bonus = 1.5;
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::CoverBonusOutOfRange { unit_id: 1, .. }
        )));
    }

    #[test]
    fn valid_ability_slots_pass() {
        let mut state = sample_duel_state(42);
        let mut def = make_ability_def();
        def.max_charges = 3;
        let mut slot = AbilitySlot::new(def);
        slot.charges = 2;
        state.units[0].abilities.push(slot);
        let report = verify_tick(&state);
        assert!(report.is_ok(), "Expected no violations, got: {:?}", report.violations);
    }

    #[test]
    fn detects_projectile_orphaned_source() {
        use crate::ai::effects::Projectile;
        let mut state = sample_duel_state(42);
        state.projectiles.push(Projectile {
            source_id: 999,
            target_id: 2,
            position: sim_vec2(1.0, 1.0),
            direction: sim_vec2(1.0, 0.0),
            speed: 10.0,
            pierce: false,
            width: 0.5,
            on_hit: Vec::new(),
            on_arrival: Vec::new(),
            already_hit: Vec::new(),
            target_position: sim_vec2(5.0, 0.0),
            max_travel_distance: 0.0,
            distance_traveled: 0.0,
        });
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::ProjectileOrphanedSource { projectile_index: 0, source_id: 999 }
        )));
    }

    #[test]
    fn detects_projectile_invalid_position() {
        use crate::ai::effects::Projectile;
        let mut state = sample_duel_state(42);
        state.projectiles.push(Projectile {
            source_id: 1,
            target_id: 2,
            position: sim_vec2(f32::NAN, 0.0),
            direction: sim_vec2(1.0, 0.0),
            speed: 10.0,
            pierce: false,
            width: 0.5,
            on_hit: Vec::new(),
            on_arrival: Vec::new(),
            already_hit: Vec::new(),
            target_position: sim_vec2(5.0, 0.0),
            max_travel_distance: 0.0,
            distance_traveled: 0.0,
        });
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::ProjectileInvalidPosition { projectile_index: 0, .. }
        )));
    }

    #[test]
    fn detects_projectile_invalid_speed() {
        use crate::ai::effects::Projectile;
        let mut state = sample_duel_state(42);
        state.projectiles.push(Projectile {
            source_id: 1,
            target_id: 2,
            position: sim_vec2(1.0, 1.0),
            direction: sim_vec2(1.0, 0.0),
            speed: 0.0,
            pierce: false,
            width: 0.5,
            on_hit: Vec::new(),
            on_arrival: Vec::new(),
            already_hit: Vec::new(),
            target_position: sim_vec2(5.0, 0.0),
            max_travel_distance: 0.0,
            distance_traveled: 0.0,
        });
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::ProjectileInvalidSpeed { projectile_index: 0, .. }
        )));
    }

    #[test]
    fn detects_slow_factor_out_of_range() {
        let mut state = sample_duel_state(42);
        state.units[0].status_effects.push(ActiveStatusEffect {
            kind: StatusKind::Slow { factor: 1.5 },
            source_id: 2,
            remaining_ms: 1000,
            tags: Default::default(),
            stacking: Stacking::default(),
        });
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::StatusEffectValueOutOfRange { unit_id: 1, kind: "Slow", .. }
        )));
    }

    #[test]
    fn detects_stacks_exceed_max() {
        let mut state = sample_duel_state(42);
        state.units[0].status_effects.push(ActiveStatusEffect {
            kind: StatusKind::Stacks { name: "Bleed".into(), count: 10, max_stacks: 5 },
            source_id: 2,
            remaining_ms: 1000,
            tags: Default::default(),
            stacking: Stacking::default(),
        });
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::StatusEffectStacksExceedMax { unit_id: 1, count: 10, max_stacks: 5 }
        )));
    }

    #[test]
    fn detects_summon_owner_missing() {
        let mut state = sample_duel_state(42);
        state.units[0].owner_id = Some(999);
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::SummonOwnerMissing { unit_id: 1, owner_id: 999 }
        )));
    }

    #[test]
    fn detects_negative_armor() {
        let mut state = sample_duel_state(42);
        state.units[0].armor = -10.0;
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::NegativeResist { unit_id: 1, field: "armor", .. }
        )));
    }

    #[test]
    fn detects_negative_magic_resist() {
        let mut state = sample_duel_state(42);
        state.units[0].magic_resist = -5.0;
        let report = verify_tick(&state);
        assert!(report.violations.iter().any(|v| matches!(v,
            Violation::NegativeResist { unit_id: 1, field: "magic_resist", .. }
        )));
    }
}
