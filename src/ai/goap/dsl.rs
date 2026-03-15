//! Parser for `.goap` and `.behavior` files.
//!
//! `.goap` files are parsed natively. `.behavior` files are desugared into
//! equivalent `GoapDef`s — each rule becomes a single-step GOAP action with
//! cost derived from priority level, and a single catch-all goal.

use crate::ai::behavior::parser::parse_target;
use crate::ai::behavior::types::{
    Action, BehaviorTree, Condition, CompOp as BehaviorCompOp, RulePriority, StateCheck, Target,
    Value,
};

use super::action::{GoapAction, IntentTemplate};
use super::goal::{CompOp, Goal, InsistenceFn, Precondition};
use super::world_state::{self, prop_index, PropValue};

/// Role hint for threat/formation systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleHint {
    MeleeEngage,
    RangedDamage,
    Support,
    Controller,
    Skirmisher,
}

/// A complete GOAP definition parsed from a `.goap` or `.behavior` file.
#[derive(Debug, Clone)]
pub struct GoapDef {
    pub name: String,
    pub role_hint: Option<RoleHint>,
    pub goals: Vec<Goal>,
    pub actions: Vec<GoapAction>,
}

// ===========================================================================
// Unified entry point
// ===========================================================================

/// Parse either a `.goap` or `.behavior` file. Auto-detects format from the
/// first non-blank keyword.
pub fn parse_goap_or_behavior(input: &str) -> Result<GoapDef, String> {
    let first_keyword = input
        .lines()
        .map(|l| l.trim())
        .find(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("//"))
        .and_then(|l| l.split_whitespace().next())
        .unwrap_or("");

    match first_keyword {
        "goap" => parse_goap(input),
        "behavior" => {
            let tree = crate::ai::behavior::parser::parse_behavior(input)?;
            Ok(behavior_to_goap(&tree))
        }
        other => Err(format!(
            "Expected 'goap' or 'behavior' block, got '{}'",
            other
        )),
    }
}

/// Load a `.goap` or `.behavior` file from disk.
pub fn load_goap_file(path: &str) -> Result<GoapDef, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
    parse_goap_or_behavior(&content)
}

// ===========================================================================
// Behavior → GOAP desugaring
// ===========================================================================

/// Convert a parsed `BehaviorTree` into an equivalent `GoapDef`.
///
/// The mapping:
/// - One catch-all goal ("act") that is never satisfied (desire enemy_count == 0
///   with fixed insistence 1.0) — forces the planner to always pick an action.
/// - Each rule becomes a GOAP action:
///   - Priority → cost 0.1, Default → cost 1.0, Fallback → cost 10.0
///   - Rule condition → GOAP preconditions (lowered to world-state props)
///   - Rule action → IntentTemplate
///   - Duration = 1 tick (behavior trees re-evaluate every tick)
pub fn behavior_to_goap(tree: &BehaviorTree) -> GoapDef {
    let goal = Goal {
        name: "act".to_string(),
        desired: vec![(
            world_state::ENEMY_COUNT,
            Precondition {
                op: CompOp::Eq,
                value: PropValue::Float(0.0),
            },
        )],
        insistence: InsistenceFn::Fixed(1.0),
    };

    let mut actions = Vec::with_capacity(tree.rules.len());

    for (i, rule) in tree.rules.iter().enumerate() {
        let base_cost = match rule.priority {
            RulePriority::Priority => 0.1,
            RulePriority::Default => 1.0,
            RulePriority::Fallback => 10.0,
        };
        // Tie-break by declaration order
        let cost = base_cost + i as f32 * 0.001;

        let preconditions = match &rule.condition {
            Some(cond) => lower_condition(cond),
            None => vec![],
        };

        let intent = lower_action(&rule.action);

        // All behavior actions effect "enemy_count = 0" so the planner
        // considers them as resolving the catch-all goal.
        let effects = vec![(world_state::ENEMY_COUNT, PropValue::Float(0.0))];

        let name = format!("rule_{}", i);

        actions.push(GoapAction {
            name,
            cost,
            preconditions,
            effects,
            intent,
            duration_ticks: 1, // behavior trees re-evaluate every tick
        });
    }

    GoapDef {
        name: tree.name.clone(),
        role_hint: None,
        goals: vec![goal],
        actions,
    }
}

/// Lower a behavior `Action` to an `IntentTemplate`.
fn lower_action(action: &Action) -> IntentTemplate {
    match action {
        Action::Attack(t) | Action::Focus(t) => IntentTemplate::AttackTarget(t.clone()),
        Action::Chase(t) => IntentTemplate::ChaseTarget(t.clone()),
        Action::Flee(t) => IntentTemplate::FleeTarget(t.clone()),
        Action::MaintainDistance(t, range) => IntentTemplate::MaintainDistance(t.clone(), *range),
        Action::CastIfReady(slot, t) | Action::CastAbility(slot, t) => {
            IntentTemplate::CastIfReady(*slot, t.clone())
        }
        Action::Hold => IntentTemplate::Hold,
        // Actions that don't have a direct GOAP equivalent degrade to Hold.
        // The GOAP planner will skip them (Hold has no useful effect).
        Action::UseBestAbility
        | Action::UseBestAbilityOn(_)
        | Action::UseAbilityType(_, _)
        | Action::MoveTo(_)
        | Action::Run(_) => IntentTemplate::Hold,
    }
}

/// Lower a behavior `Condition` tree into flat GOAP preconditions.
///
/// This is best-effort: complex conditions (Or, Not, nested And) that can't
/// map cleanly to independent world-state property checks are dropped (the
/// action becomes unconditional for those parts, which is conservative — the
/// intent resolver will still fail gracefully if the target doesn't exist).
fn lower_condition(cond: &Condition) -> Vec<(usize, Precondition)> {
    let mut out = Vec::new();
    collect_preconditions(cond, &mut out);
    out
}

fn collect_preconditions(cond: &Condition, out: &mut Vec<(usize, Precondition)>) {
    match cond {
        Condition::And(a, b) => {
            collect_preconditions(a, out);
            collect_preconditions(b, out);
        }
        // Or / Not can't be represented as flat conjunctive preconditions — skip
        Condition::Or(_, _) | Condition::Not(_) => {}
        Condition::Compare(lhs, op, rhs) => {
            if let Some(pre) = lower_compare(lhs, *op, rhs) {
                out.push(pre);
            }
        }
        Condition::StateCheck(sc) => {
            if let Some(pre) = lower_state_check(sc) {
                out.push(pre);
            }
        }
    }
}

/// Try to lower a comparison `value op value` into a `(prop_idx, Precondition)`.
fn lower_compare(
    lhs: &Value,
    op: BehaviorCompOp,
    rhs: &Value,
) -> Option<(usize, Precondition)> {
    let goap_op = convert_comp_op(op);

    // Pattern: prop op number (e.g. `nearest_enemy.distance < 1.5`)
    if let Value::Number(n) = rhs {
        if let Some(prop) = value_to_prop_index(lhs) {
            return Some((
                prop,
                Precondition {
                    op: goap_op,
                    value: PropValue::Float(*n),
                },
            ));
        }
    }

    // Pattern: number op prop (reversed)
    if let Value::Number(n) = lhs {
        if let Some(prop) = value_to_prop_index(rhs) {
            let flipped = flip_op(goap_op);
            return Some((
                prop,
                Precondition {
                    op: flipped,
                    value: PropValue::Float(*n),
                },
            ));
        }
    }

    None // Can't lower — drop silently
}

/// Map a behavior `Value` to a GOAP world state property index.
fn value_to_prop_index(val: &Value) -> Option<usize> {
    match val {
        Value::SelfHpPct => Some(world_state::SELF_HP_PCT),
        Value::SelfHp => Some(world_state::SELF_HP_PCT), // approximate
        Value::EnemyCount => Some(world_state::ENEMY_COUNT),
        Value::AllyCount => Some(world_state::ALLY_COUNT),
        Value::TargetDistance(Target::NearestEnemy) => Some(world_state::NEAREST_ENEMY_DISTANCE),
        Value::TargetHpPct(Target::NearestEnemy) => Some(world_state::NEAREST_ENEMY_HP_PCT),
        Value::TargetHpPct(Target::LowestHpAlly) => Some(world_state::LOWEST_ALLY_HP_PCT),
        Value::TargetDistance(_) => Some(world_state::TARGET_DISTANCE),
        Value::TargetHpPct(_) => Some(world_state::TARGET_HP_PCT),
        _ => None,
    }
}

/// Lower a `StateCheck` into a GOAP precondition.
fn lower_state_check(sc: &StateCheck) -> Option<(usize, Precondition)> {
    let bool_true = Precondition {
        op: CompOp::Eq,
        value: PropValue::Bool(true),
    };
    let _bool_false = Precondition {
        op: CompOp::Eq,
        value: PropValue::Bool(false),
    };

    match sc {
        StateCheck::IsCasting => Some((world_state::SELF_IS_CASTING, bool_true)),
        StateCheck::IsCcd => Some((world_state::SELF_IS_CCD, bool_true)),
        StateCheck::AbilityReady(slot) => {
            let prop = world_state::ABILITY_0_READY + slot;
            if prop <= world_state::ABILITY_7_READY {
                Some((prop, bool_true))
            } else {
                None
            }
        }
        StateCheck::CanAttack => Some((world_state::IN_ATTACK_RANGE, bool_true)),
        StateCheck::InDangerZone => Some((world_state::IN_DANGER_ZONE, bool_true)),
        StateCheck::AllyInDanger => Some((
            world_state::LOWEST_ALLY_HP_PCT,
            Precondition {
                op: CompOp::Lt,
                value: PropValue::Float(0.3),
            },
        )),
        StateCheck::HealReady => Some((world_state::HAS_HEAL_TARGET, bool_true)),
        StateCheck::TargetIsCasting(Target::NearestEnemy) | StateCheck::TargetIsCasting(_) => {
            Some((world_state::ENEMY_IS_CASTING, bool_true))
        }
        // Can't map these to world state props
        StateCheck::CcReady
        | StateCheck::AoeReady
        | StateCheck::TargetIsCcd(_)
        | StateCheck::NearWall(_)
        | StateCheck::Every(_) => None,
    }
}

fn convert_comp_op(op: BehaviorCompOp) -> CompOp {
    match op {
        BehaviorCompOp::Lt => CompOp::Lt,
        BehaviorCompOp::Gt => CompOp::Gt,
        BehaviorCompOp::Lte => CompOp::Lte,
        BehaviorCompOp::Gte => CompOp::Gte,
        BehaviorCompOp::Eq => CompOp::Eq,
        BehaviorCompOp::Neq => CompOp::Neq,
    }
}

fn flip_op(op: CompOp) -> CompOp {
    match op {
        CompOp::Lt => CompOp::Gt,
        CompOp::Gt => CompOp::Lt,
        CompOp::Lte => CompOp::Gte,
        CompOp::Gte => CompOp::Lte,
        CompOp::Eq => CompOp::Eq,
        CompOp::Neq => CompOp::Neq,
    }
}

// ===========================================================================
// Native GOAP parser
// ===========================================================================

/// Parse a `.goap` file from text.
pub fn parse_goap(input: &str) -> Result<GoapDef, String> {
    let lines: Vec<&str> = input.lines().collect();
    let mut idx = 0;

    skip_blank(&lines, &mut idx);

    let header_line = lines.get(idx).ok_or("Expected 'goap' block")?;
    let tokens = tokenize(header_line);
    if tokens.first().map(|s| s.as_str()) != Some("goap") {
        return Err(format!("Expected 'goap', got: {:?}", tokens.first()));
    }
    let name = unquote(tokens.get(1).ok_or("Expected goap name")?);
    expect_token(&tokens, 2, "{")?;
    idx += 1;

    let mut role_hint = None;
    let mut goals = Vec::new();
    let mut actions = Vec::new();

    while idx < lines.len() {
        skip_blank(&lines, &mut idx);
        if idx >= lines.len() {
            break;
        }

        let line = lines[idx].trim();
        if line == "}" {
            break;
        }

        let tokens = tokenize(line);
        if tokens.is_empty() {
            idx += 1;
            continue;
        }

        match tokens[0].as_str() {
            "role_hint" => {
                role_hint = Some(parse_role_hint(
                    tokens.get(1).ok_or("Expected role hint value")?,
                )?);
                idx += 1;
            }
            "goal" => {
                let (goal, new_idx) = parse_goal(&lines, idx)?;
                goals.push(goal);
                idx = new_idx;
            }
            "action" => {
                let (action, new_idx) = parse_action_block(&lines, idx)?;
                actions.push(action);
                idx = new_idx;
            }
            other => {
                return Err(format!("Unexpected token in goap block: '{}'", other));
            }
        }
    }

    Ok(GoapDef {
        name,
        role_hint,
        goals,
        actions,
    })
}

fn parse_goal(lines: &[&str], start: usize) -> Result<(Goal, usize), String> {
    let header = tokenize(lines[start]);
    let name = unquote(header.get(1).ok_or("Expected goal name")?);
    expect_token(&header, 2, "{")?;

    let mut idx = start + 1;
    let mut desired = Vec::new();
    let mut insistence = InsistenceFn::Fixed(0.5);

    while idx < lines.len() {
        let line = lines[idx].trim();
        if line.is_empty() || line.starts_with('#') {
            idx += 1;
            continue;
        }
        if line == "}" {
            idx += 1;
            break;
        }

        let tokens = tokenize(line);
        match tokens[0].as_str() {
            "desire" => {
                let (prop, pre) = parse_precondition(&tokens[1..])?;
                desired.push((prop, pre));
            }
            "insistence" => {
                insistence = parse_insistence(&tokens[1..])?;
            }
            other => return Err(format!("Unknown goal keyword: '{}'", other)),
        }
        idx += 1;
    }

    Ok((Goal { name, desired, insistence }, idx))
}

fn parse_action_block(lines: &[&str], start: usize) -> Result<(GoapAction, usize), String> {
    let header = tokenize(lines[start]);
    let name = unquote(header.get(1).ok_or("Expected action name")?);
    expect_token(&header, 2, "{")?;

    let mut idx = start + 1;
    let mut cost = 1.0_f32;
    let mut preconditions = Vec::new();
    let mut effects = Vec::new();
    let mut intent = IntentTemplate::Hold;
    let mut duration = 1u32;

    while idx < lines.len() {
        let line = lines[idx].trim();
        if line.is_empty() || line.starts_with('#') {
            idx += 1;
            continue;
        }
        if line == "}" {
            idx += 1;
            break;
        }

        let tokens = tokenize(line);
        match tokens[0].as_str() {
            "cost" => {
                cost = tokens[1].parse().map_err(|_| "Invalid cost value")?;
            }
            "precondition" => {
                let (prop, pre) = parse_precondition(&tokens[1..])?;
                preconditions.push((prop, pre));
            }
            "effect" => {
                let (prop, val) = parse_effect(&tokens[1..])?;
                effects.push((prop, val));
            }
            "intent" => {
                intent = parse_intent(&tokens[1..])?;
            }
            "duration" => {
                duration = tokens[1].parse().map_err(|_| "Invalid duration")?;
            }
            other => return Err(format!("Unknown action keyword: '{}'", other)),
        }
        idx += 1;
    }

    Ok((
        GoapAction {
            name,
            cost,
            preconditions,
            effects,
            intent,
            duration_ticks: duration,
        },
        idx,
    ))
}

fn parse_precondition(tokens: &[String]) -> Result<(usize, Precondition), String> {
    if tokens.len() < 3 {
        return Err(format!("Precondition needs 3 tokens, got: {:?}", tokens));
    }
    let prop = prop_index(&tokens[0])
        .ok_or_else(|| format!("Unknown property: '{}'", tokens[0]))?;
    let op = parse_comp_op(&tokens[1])?;
    let value = parse_prop_value(&tokens[2])?;
    Ok((prop, Precondition { op, value }))
}

fn parse_effect(tokens: &[String]) -> Result<(usize, PropValue), String> {
    if tokens.len() < 3 {
        return Err(format!("Effect needs 3 tokens, got: {:?}", tokens));
    }
    let prop = prop_index(&tokens[0])
        .ok_or_else(|| format!("Unknown property: '{}'", tokens[0]))?;
    let value = parse_prop_value(&tokens[2])?;
    Ok((prop, value))
}

fn parse_insistence(tokens: &[String]) -> Result<InsistenceFn, String> {
    match tokens[0].as_str() {
        "fixed" => {
            let val: f32 = tokens[1]
                .parse()
                .map_err(|_| "Invalid fixed insistence")?;
            Ok(InsistenceFn::Fixed(val))
        }
        "linear" => {
            let prop = prop_index(&tokens[1])
                .ok_or_else(|| format!("Unknown property: '{}'", tokens[1]))?;
            let mut scale = 1.0_f32;
            let mut offset = 0.0_f32;
            let mut i = 2;
            while i < tokens.len() {
                match tokens[i].as_str() {
                    "scale" => {
                        scale = tokens[i + 1].parse().map_err(|_| "Invalid scale")?;
                        i += 2;
                    }
                    "offset" => {
                        offset = tokens[i + 1].parse().map_err(|_| "Invalid offset")?;
                        i += 2;
                    }
                    _ => return Err(format!("Unknown insistence modifier: '{}'", tokens[i])),
                }
            }
            Ok(InsistenceFn::Linear {
                prop,
                scale,
                offset,
            })
        }
        "threshold" => {
            let prop = prop_index(&tokens[1])
                .ok_or_else(|| format!("Unknown property: '{}'", tokens[1]))?;
            let op = parse_comp_op(&tokens[2])?;
            let threshold: f32 = tokens[3].parse().map_err(|_| "Invalid threshold")?;
            if tokens.get(4).map(|s| s.as_str()) != Some("value") {
                return Err("Expected 'value' keyword in threshold insistence".to_string());
            }
            let value: f32 = tokens[5].parse().map_err(|_| "Invalid threshold value")?;
            Ok(InsistenceFn::Threshold {
                prop,
                op,
                threshold,
                value,
            })
        }
        other => Err(format!("Unknown insistence type: '{}'", other)),
    }
}

fn parse_intent(tokens: &[String]) -> Result<IntentTemplate, String> {
    match tokens[0].as_str() {
        "attack" => {
            let target = parse_target(&tokens[1..])?;
            Ok(IntentTemplate::AttackTarget(target))
        }
        "chase" => {
            let target = parse_target(&tokens[1..])?;
            Ok(IntentTemplate::ChaseTarget(target))
        }
        "flee" => {
            let target = parse_target(&tokens[1..])?;
            Ok(IntentTemplate::FleeTarget(target))
        }
        "maintain_distance" => {
            let target = parse_target(&tokens[1..tokens.len() - 1])?;
            let range: f32 = tokens
                .last()
                .ok_or("Missing range")?
                .parse()
                .map_err(|_| "Invalid range")?;
            Ok(IntentTemplate::MaintainDistance(target, range))
        }
        "cast_if_ready" => {
            let slot = parse_ability_slot(&tokens[1])?;
            let target_start = if tokens.get(2).map(|s| s.as_str()) == Some("on") {
                3
            } else {
                2
            };
            let target = parse_target(&tokens[target_start..])?;
            Ok(IntentTemplate::CastIfReady(slot, target))
        }
        "hold" => Ok(IntentTemplate::Hold),
        other => Err(format!("Unknown intent: '{}'", other)),
    }
}

fn parse_ability_slot(s: &str) -> Result<usize, String> {
    if let Some(rest) = s.strip_prefix("ability") {
        rest.parse()
            .map_err(|_| format!("Invalid ability slot: '{}'", s))
    } else {
        Err(format!("Expected 'abilityN', got '{}'", s))
    }
}

fn parse_comp_op(s: &str) -> Result<CompOp, String> {
    match s {
        "==" => Ok(CompOp::Eq),
        "!=" => Ok(CompOp::Neq),
        "<" => Ok(CompOp::Lt),
        ">" => Ok(CompOp::Gt),
        "<=" => Ok(CompOp::Lte),
        ">=" => Ok(CompOp::Gte),
        _ => Err(format!("Unknown comparison operator: '{}'", s)),
    }
}

fn parse_prop_value(s: &str) -> Result<PropValue, String> {
    match s {
        "true" => Ok(PropValue::Bool(true)),
        "false" => Ok(PropValue::Bool(false)),
        _ => {
            let f: f32 = s
                .parse()
                .map_err(|_| format!("Invalid value: '{}'", s))?;
            Ok(PropValue::Float(f))
        }
    }
}

fn parse_role_hint(s: &str) -> Result<RoleHint, String> {
    match s {
        "melee_engage" => Ok(RoleHint::MeleeEngage),
        "ranged_damage" => Ok(RoleHint::RangedDamage),
        "support" => Ok(RoleHint::Support),
        "controller" => Ok(RoleHint::Controller),
        "skirmisher" => Ok(RoleHint::Skirmisher),
        _ => Err(format!("Unknown role hint: '{}'", s)),
    }
}

// ===========================================================================
// Shared utilities
// ===========================================================================

fn tokenize(line: &str) -> Vec<String> {
    let line = line.trim();
    let line = if let Some(idx) = line.find('#') {
        &line[..idx]
    } else {
        line
    };
    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();
    let mut current = String::new();

    while let Some(&ch) = chars.peek() {
        if ch == '"' {
            chars.next();
            let mut quoted = String::new();
            while let Some(&c) = chars.peek() {
                if c == '"' {
                    chars.next();
                    break;
                }
                quoted.push(c);
                chars.next();
            }
            tokens.push(quoted);
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            chars.next();
        } else {
            current.push(ch);
            chars.next();
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn unquote(s: &str) -> String {
    s.trim_matches('"').to_string()
}

fn expect_token(tokens: &[String], idx: usize, expected: &str) -> Result<(), String> {
    match tokens.get(idx) {
        Some(t) if t == expected => Ok(()),
        Some(t) => Err(format!("Expected '{}', got '{}'", expected, t)),
        None => Err(format!("Expected '{}', got end of line", expected)),
    }
}

fn skip_blank(lines: &[&str], idx: &mut usize) {
    while *idx < lines.len() {
        let line = lines[*idx].trim();
        if line.is_empty() || line.starts_with('#') {
            *idx += 1;
        } else {
            break;
        }
    }
}
