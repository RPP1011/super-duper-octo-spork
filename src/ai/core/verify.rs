use std::collections::HashSet;

use super::types::{SimState, Team};

/// A specific invariant violation detected during a simulation tick.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    }

    // Zone source validation
    for zone in &state.zones {
        if !unit_ids.contains(&zone.source_id) {
            violations.push(Violation::ZoneOrphanedSource {
                zone_id: zone.id,
                source_id: zone.source_id,
            });
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
}
