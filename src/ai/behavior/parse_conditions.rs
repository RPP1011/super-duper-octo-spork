//! Condition, value, and state check parsing for the behavior DSL.
//!
//! Split from `parser.rs` to keep files under 500 lines.

use super::parser::{parse_target, parse_ability_slot};
use super::types::*;

// ---------------------------------------------------------------------------
// Condition parsing
// ---------------------------------------------------------------------------

/// Parse a condition from tokens. Handles `and`/`or` as infix operators with
/// left-to-right associativity (no precedence — `and` and `or` bind equally).
pub(super) fn parse_condition(tokens: &[String]) -> Result<Condition, String> {
    // Split on `and` / `or` at the top level
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i].as_str();
        if tok == "and" || tok == "or" {
            let left = parse_condition_atom(&tokens[..i])?;
            let right = parse_condition(&tokens[i + 1..])?;
            return if tok == "and" {
                Ok(Condition::And(Box::new(left), Box::new(right)))
            } else {
                Ok(Condition::Or(Box::new(left), Box::new(right)))
            };
        }
        // Skip over multi-token constructs to avoid splitting inside them
        if is_target_keyword(tok) {
            i += target_token_count(&tokens[i..]);
            // After target, might be `.field` then op then value — skip those too
            if i < tokens.len() && tokens[i].starts_with('.') {
                i += 1; // .field
                if i < tokens.len() && is_comp_op(&tokens[i]) {
                    i += 1; // op
                    if i < tokens.len() {
                        i += 1; // value
                    }
                }
            }
            continue;
        }
        i += 1;
    }
    // No and/or found — parse as atom
    parse_condition_atom(tokens)
}

fn parse_condition_atom(tokens: &[String]) -> Result<Condition, String> {
    if tokens.is_empty() {
        return Err("empty condition".into());
    }

    // Handle `not`
    if tokens[0] == "not" {
        let inner = parse_condition_atom(&tokens[1..])?;
        return Ok(Condition::Not(Box::new(inner)));
    }

    // Try state checks first (single or two-token)
    if let Some(sc) = try_parse_state_check(tokens) {
        return Ok(Condition::StateCheck(sc));
    }

    // Try comparison: value op value
    if let Some(cond) = try_parse_comparison(tokens)? {
        return Ok(cond);
    }

    Err(format!("cannot parse condition: {}", tokens.join(" ")))
}

fn try_parse_state_check(tokens: &[String]) -> Option<StateCheck> {
    match tokens[0].as_str() {
        "heal_ready" => Some(StateCheck::HealReady),
        "cc_ready" | "stun_ready" => Some(StateCheck::CcReady),
        "aoe_ready" => Some(StateCheck::AoeReady),
        "is_casting" => Some(StateCheck::IsCasting),
        "is_cc_d" => Some(StateCheck::IsCcd),
        "can_attack" => Some(StateCheck::CanAttack),
        "in_danger_zone" => Some(StateCheck::InDangerZone),
        "ally_in_danger" => Some(StateCheck::AllyInDanger),
        "ability_ready" => {
            if tokens.len() >= 2 {
                parse_ability_slot(&tokens[1]).ok().map(StateCheck::AbilityReady)
            } else {
                None
            }
        }
        "every" => {
            if tokens.len() >= 2 {
                tokens[1].parse::<u64>().ok().map(StateCheck::Every)
            } else {
                None
            }
        }
        "near_wall" => {
            if tokens.len() >= 2 {
                tokens[1].parse::<f32>().ok().map(StateCheck::NearWall)
            } else {
                None
            }
        }
        "target_is_casting" => {
            if tokens.len() >= 2 {
                parse_target(&tokens[1..]).ok().map(|t| StateCheck::TargetIsCasting(t))
            } else {
                None
            }
        }
        "target_is_cc_d" => {
            if tokens.len() >= 2 {
                parse_target(&tokens[1..]).ok().map(|t| StateCheck::TargetIsCcd(t))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Try to parse `target.field op value` or `value op value` comparisons.
fn try_parse_comparison(tokens: &[String]) -> Result<Option<Condition>, String> {
    // Scan for a comparison operator
    for i in 0..tokens.len() {
        if is_comp_op(&tokens[i]) {
            let lhs = parse_value(&tokens[..i])?;
            let op = parse_comp_op(&tokens[i]);
            let rhs = parse_value(&tokens[i + 1..])?;
            return Ok(Some(Condition::Compare(lhs, op, rhs)));
        }
    }
    Ok(None)
}

fn is_comp_op(tok: &str) -> bool {
    matches!(tok, "<" | ">" | "<=" | ">=" | "==" | "!=")
}

fn parse_comp_op(tok: &str) -> CompOp {
    match tok {
        "<" => CompOp::Lt,
        ">" => CompOp::Gt,
        "<=" => CompOp::Lte,
        ">=" => CompOp::Gte,
        "==" => CompOp::Eq,
        "!=" => CompOp::Neq,
        _ => unreachable!(),
    }
}

pub(super) fn parse_value(tokens: &[String]) -> Result<Value, String> {
    if tokens.is_empty() {
        return Err("expected value".into());
    }

    // Single-token values
    if tokens.len() == 1 {
        let tok = &tokens[0];
        // Try number
        if let Ok(n) = tok.parse::<f32>() {
            return Ok(Value::Number(n));
        }
        return match tok.as_str() {
            "self.hp" => Ok(Value::SelfHp),
            "self.hp_pct" => Ok(Value::SelfHpPct),
            "enemy_count" => Ok(Value::EnemyCount),
            "ally_count" => Ok(Value::AllyCount),
            "tick" => Ok(Value::Tick),
            "best_ability_urgency" => Ok(Value::BestAbilityUrgency),
            _ => {
                // Check for target.field pattern (dot in single token)
                if let Some(dot_pos) = tok.find('.') {
                    let target_str = &tok[..dot_pos];
                    let field = &tok[dot_pos + 1..];
                    let target = parse_target(&[target_str.to_string()])?;
                    return match field {
                        "hp" => Ok(Value::TargetHp(target)),
                        "hp_pct" => Ok(Value::TargetHpPct(target)),
                        "distance" => Ok(Value::TargetDistance(target)),
                        "dps" => Ok(Value::TargetDps(target)),
                        "cc_remaining" => Ok(Value::TargetCcRemaining(target)),
                        "cast_progress" => Ok(Value::TargetCastProgress(target)),
                        other => Err(format!("unknown field '.{other}' on target")),
                    };
                }
                Err(format!("unknown value '{tok}'"))
            }
        };
    }

    // Multi-token values
    match tokens[0].as_str() {
        "enemy_count_in_range" => {
            if tokens.len() < 2 {
                return Err("enemy_count_in_range requires a range".into());
            }
            let r: f32 = tokens[1].parse().map_err(|_| "invalid range number")?;
            Ok(Value::EnemyCountInRange(r))
        }
        "ally_count_below_hp" => {
            if tokens.len() < 2 {
                return Err("ally_count_below_hp requires a threshold".into());
            }
            let t: f32 = tokens[1].parse().map_err(|_| "invalid hp threshold")?;
            Ok(Value::AllyCountBelowHp(t))
        }
        "ability" => {
            // ability ability0 .cooldown_pct
            if tokens.len() >= 2 {
                if let Ok(slot) = parse_ability_slot(&tokens[1]) {
                    return Ok(Value::AbilityCooldownPct(slot));
                }
            }
            Err("invalid ability value".into())
        }
        _ => {
            // Multi-token target.field: e.g. `tagged "foo" .hp_pct`
            // or `nearest_enemy .distance`
            // Find the token with a leading dot
            for i in 1..tokens.len() {
                if tokens[i].starts_with('.') {
                    let target = parse_target(&tokens[..i])?;
                    let field = &tokens[i][1..]; // strip leading dot
                    return match field {
                        "hp" => Ok(Value::TargetHp(target)),
                        "hp_pct" => Ok(Value::TargetHpPct(target)),
                        "distance" => Ok(Value::TargetDistance(target)),
                        "dps" => Ok(Value::TargetDps(target)),
                        "cc_remaining" => Ok(Value::TargetCcRemaining(target)),
                        "cast_progress" => Ok(Value::TargetCastProgress(target)),
                        other => Err(format!("unknown field '.{other}' on target")),
                    };
                }
            }
            Err(format!("cannot parse value: {}", tokens.join(" ")))
        }
    }
}

pub(super) fn is_target_keyword(tok: &str) -> bool {
    matches!(
        tok,
        "self"
            | "nearest_enemy"
            | "nearest_ally"
            | "lowest_hp_enemy"
            | "lowest_hp_ally"
            | "highest_dps_enemy"
            | "highest_threat_enemy"
            | "casting_enemy"
            | "enemy_attacking"
            | "tagged"
            | "unit"
    )
}

/// How many tokens a target starting at tokens[0] consumes.
pub(super) fn target_token_count(tokens: &[String]) -> usize {
    match tokens[0].as_str() {
        "tagged" => 2,                                                // tagged "name"
        "unit" => 2,                                                  // unit 42
        "enemy_attacking" => 1 + target_token_count(&tokens[1..]),    // recursive
        _ => 1,                                                       // single-word targets
    }
}
