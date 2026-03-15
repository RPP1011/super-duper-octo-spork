//! Target resolution and action conversion for the behavior interpreter.
//!
//! Split from `interpreter.rs` to keep files under 500 lines.

use crate::ai::core::{
    distance, is_alive, position_at_range, move_away,
    IntentAction, SimState, SimVec2, UnitState,
};
use crate::ai::effects::AbilityTarget;

use super::types::*;

// ---------------------------------------------------------------------------
// Target resolution
// ---------------------------------------------------------------------------

pub(super) fn resolve_target(target: &Target, state: &SimState, unit_idx: usize) -> Option<u32> {
    let unit = &state.units[unit_idx];
    match target {
        Target::Self_ => Some(unit.id),
        Target::NearestEnemy => {
            nearest_by(state, unit, |u| u.team != unit.team)
        }
        Target::NearestAlly => {
            nearest_by(state, unit, |u| u.team == unit.team && u.id != unit.id)
        }
        Target::LowestHpEnemy => {
            min_by_key(state, unit, |u| u.team != unit.team, |u| u.hp)
        }
        Target::LowestHpAlly => {
            min_by_key(state, unit, |u| u.team == unit.team && u.id != unit.id, |u| u.hp)
        }
        Target::HighestDpsEnemy => {
            max_by_key(state, |u| u.team != unit.team, |u| u.total_damage_done)
        }
        Target::HighestThreatEnemy => {
            // threat = dps * (1 / distance), approximate with damage_done / distance
            let pos = unit.position;
            state
                .units
                .iter()
                .filter(|u| is_alive(u) && u.team != unit.team)
                .max_by(|a, b| {
                    let da = distance(pos, a.position).max(0.1);
                    let db = distance(pos, b.position).max(0.1);
                    let ta = a.total_damage_done as f32 / da;
                    let tb = b.total_damage_done as f32 / db;
                    ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|u| u.id)
        }
        Target::CastingEnemy => {
            state
                .units
                .iter()
                .find(|u| is_alive(u) && u.team != unit.team && u.casting.is_some())
                .map(|u| u.id)
        }
        Target::EnemyAttacking(inner) => {
            let inner_id = resolve_target(inner, state, unit_idx)?;
            state
                .units
                .iter()
                .find(|u| {
                    is_alive(u)
                        && u.team != unit.team
                        && u.casting
                            .map_or(false, |c| c.target_id == inner_id)
                })
                .map(|u| u.id)
        }
        Target::Tagged(_tag) => {
            // Tags are not stored in UnitState currently — would need scenario integration.
            None
        }
        Target::UnitId(id) => {
            if state.units.iter().any(|u| u.id == *id && is_alive(u)) {
                Some(*id)
            } else {
                None
            }
        }
    }
}

fn nearest_by(state: &SimState, unit: &UnitState, pred: impl Fn(&UnitState) -> bool) -> Option<u32> {
    let pos = unit.position;
    state
        .units
        .iter()
        .filter(|u| is_alive(u) && pred(u))
        .min_by(|a, b| {
            distance(pos, a.position)
                .partial_cmp(&distance(pos, b.position))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|u| u.id)
}

fn min_by_key(
    state: &SimState,
    unit: &UnitState,
    pred: impl Fn(&UnitState) -> bool,
    key: impl Fn(&UnitState) -> i32,
) -> Option<u32> {
    state
        .units
        .iter()
        .filter(|u| is_alive(u) && pred(u))
        .min_by_key(|u| key(u))
        .map(|u| u.id)
}

fn max_by_key(
    state: &SimState,
    pred: impl Fn(&UnitState) -> bool,
    key: impl Fn(&UnitState) -> i32,
) -> Option<u32> {
    state
        .units
        .iter()
        .filter(|u| is_alive(u) && pred(u))
        .max_by_key(|u| key(u))
        .map(|u| u.id)
}

pub(super) fn find_unit(state: &SimState, id: u32) -> Option<&UnitState> {
    state.units.iter().find(|u| u.id == id && is_alive(u))
}

// ---------------------------------------------------------------------------
// Action conversion
// ---------------------------------------------------------------------------

pub(super) fn convert_action(
    action: &Action,
    state: &SimState,
    unit_idx: usize,
    _tick: u64,
) -> Option<IntentAction> {
    let unit = &state.units[unit_idx];
    match action {
        Action::Hold => Some(IntentAction::Hold),

        Action::Chase(target) => {
            let target_id = resolve_target(target, state, unit_idx)?;
            let target_unit = find_unit(state, target_id)?;
            Some(IntentAction::MoveTo {
                position: target_unit.position,
            })
        }

        Action::Flee(target) => {
            let target_id = resolve_target(target, state, unit_idx)?;
            let target_unit = find_unit(state, target_id)?;
            let away = move_away(unit.position, target_unit.position, 100.0);
            Some(IntentAction::MoveTo { position: away })
        }

        Action::MoveTo(pos) => {
            let position = resolve_position(pos, state, unit_idx)?;
            Some(IntentAction::MoveTo { position })
        }

        Action::Attack(target) => {
            let target_id = resolve_target(target, state, unit_idx)?;
            Some(IntentAction::Attack { target_id })
        }

        Action::Focus(target) => {
            let target_id = resolve_target(target, state, unit_idx)?;
            let target_unit = find_unit(state, target_id)?;
            let dist = distance(unit.position, target_unit.position);
            if dist <= unit.attack_range {
                Some(IntentAction::Attack { target_id })
            } else {
                Some(IntentAction::MoveTo {
                    position: target_unit.position,
                })
            }
        }

        Action::MaintainDistance(target, range) => {
            let target_id = resolve_target(target, state, unit_idx)?;
            let target_unit = find_unit(state, target_id)?;
            let dist = distance(unit.position, target_unit.position);
            if dist < range * 0.9 {
                // Too close — move away
                let away_pos = position_at_range(unit.position, target_unit.position, *range);
                Some(IntentAction::MoveTo { position: away_pos })
            } else if dist > range * 1.1 {
                // Too far — move toward
                let toward_pos = position_at_range(unit.position, target_unit.position, *range);
                Some(IntentAction::MoveTo { position: toward_pos })
            } else {
                Some(IntentAction::Hold)
            }
        }

        Action::CastAbility(slot, target) => {
            let ability_target = resolve_ability_target(target, state, unit_idx)?;
            Some(IntentAction::UseAbility {
                ability_index: *slot,
                target: ability_target,
            })
        }

        Action::CastIfReady(slot, target) => {
            // Only fire if ability is off cooldown
            let ready = unit
                .abilities
                .get(*slot)
                .map_or(false, |a| a.cooldown_remaining_ms == 0);
            if !ready {
                return None;
            }
            let ability_target = resolve_ability_target(target, state, unit_idx)?;
            Some(IntentAction::UseAbility {
                ability_index: *slot,
                target: ability_target,
            })
        }

        Action::UseBestAbility | Action::UseBestAbilityOn(_) => {
            // Stub — would need ability eval integration.
            None
        }

        Action::UseAbilityType(_cat, _target) => {
            // Stub — would need ability eval integration.
            None
        }

        Action::Run(_name) => {
            // Would need a behavior registry to look up sub-behaviors.
            None
        }
    }
}

fn resolve_ability_target(
    target: &Target,
    state: &SimState,
    unit_idx: usize,
) -> Option<AbilityTarget> {
    match target {
        Target::Self_ => Some(AbilityTarget::Unit(state.units[unit_idx].id)),
        _ => {
            let id = resolve_target(target, state, unit_idx)?;
            Some(AbilityTarget::Unit(id))
        }
    }
}

fn resolve_position(pos: &Position, state: &SimState, unit_idx: usize) -> Option<SimVec2> {
    match pos {
        Position::Entity(target) => {
            let id = resolve_target(target, state, unit_idx)?;
            find_unit(state, id).map(|u| u.position)
        }
        Position::Fixed(x, y) => Some(SimVec2 { x: *x, y: *y }),
        Position::Random => {
            // Deterministic random would need rng — return center for now
            Some(SimVec2 { x: 10.0, y: 10.0 })
        }
        Position::TargetPosition => {
            // Would need drill objective marker integration
            Some(SimVec2 { x: 10.0, y: 10.0 })
        }
    }
}
