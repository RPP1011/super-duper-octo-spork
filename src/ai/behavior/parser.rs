use super::parse_conditions::parse_condition;
use super::types::*;

/// Parse a `.behavior` file into a `BehaviorTree`.
///
/// Grammar (subset):
/// ```text
/// behavior "name" {
///   priority action when condition
///   default action when condition
///   default action
///   fallback action
/// }
/// ```
pub fn parse_behavior(input: &str) -> Result<BehaviorTree, String> {
    let input = strip_comments(input);
    let (name, body) = extract_block(&input)?;
    let mut rules = Vec::new();
    for (line_no, raw_line) in body.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let rule = parse_rule(line)
            .map_err(|e| format!("line {}: {}", line_no + 1, e))?;
        rules.push(rule);
    }
    if rules.is_empty() {
        return Err("behavior has no rules".into());
    }
    Ok(BehaviorTree { name, rules })
}

/// Remove `//` line comments and `/* */` block comments.
fn strip_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.lines() {
        if let Some(idx) = line.find("//") {
            out.push_str(&line[..idx]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

/// Extract the behavior name and the inner block body.
fn extract_block(input: &str) -> Result<(String, String), String> {
    // Find `behavior "name" {`
    let trimmed = input.trim();
    let rest = trimmed
        .strip_prefix("behavior")
        .ok_or("expected 'behavior' keyword")?
        .trim_start();

    // Parse quoted name
    let name_end;
    let name;
    if rest.starts_with('"') {
        let closing = rest[1..]
            .find('"')
            .ok_or("unterminated string for behavior name")?;
        name = rest[1..1 + closing].to_string();
        name_end = 1 + closing + 1; // skip closing quote
    } else {
        return Err("expected quoted behavior name".into());
    }

    let rest = rest[name_end..].trim_start();
    let rest = rest
        .strip_prefix('{')
        .ok_or("expected '{' after behavior name")?;

    // Find matching closing brace (handle nesting if ever needed)
    let mut depth = 1u32;
    let mut end_idx = None;
    for (i, ch) in rest.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end_idx = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end_idx.ok_or("unterminated behavior block — missing closing '}'")?;
    Ok((name, rest[..end].to_string()))
}

// ---------------------------------------------------------------------------
// Rule parsing
// ---------------------------------------------------------------------------

fn parse_rule(line: &str) -> Result<Rule, String> {
    let tokens = tokenize(line);
    if tokens.is_empty() {
        return Err("empty rule".into());
    }

    let (priority, rest) = match tokens[0].as_str() {
        "priority" => (RulePriority::Priority, &tokens[1..]),
        "default" => (RulePriority::Default, &tokens[1..]),
        "fallback" => (RulePriority::Fallback, &tokens[1..]),
        other => return Err(format!("expected priority/default/fallback, got '{other}'")),
    };

    // Split on `when` to separate action tokens from condition tokens
    let when_pos = rest.iter().position(|t| t == "when");
    let (action_tokens, cond_tokens) = match when_pos {
        Some(pos) => (&rest[..pos], Some(&rest[pos + 1..])),
        None => (rest, None),
    };

    if action_tokens.is_empty() {
        return Err("missing action".into());
    }

    let action = parse_action(action_tokens)?;
    let condition = match cond_tokens {
        Some(toks) if !toks.is_empty() => {
            let cond = parse_condition(toks)?;
            Some(cond)
        }
        _ => {
            if priority == RulePriority::Fallback {
                None // fallback never has a condition
            } else {
                None
            }
        }
    };

    Ok(Rule {
        priority,
        condition,
        action,
    })
}

// ---------------------------------------------------------------------------
// Tokenizer — splits on whitespace but keeps quoted strings together
// ---------------------------------------------------------------------------

fn tokenize(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == '"' {
            chars.next(); // skip opening quote
            let mut s = String::new();
            while let Some(&c) = chars.peek() {
                if c == '"' {
                    chars.next();
                    break;
                }
                s.push(c);
                chars.next();
            }
            tokens.push(format!("\"{s}\""));
            continue;
        }
        let mut tok = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_whitespace() {
                break;
            }
            tok.push(c);
            chars.next();
        }
        tokens.push(tok);
    }
    tokens
}

// ---------------------------------------------------------------------------
// Action parsing
// ---------------------------------------------------------------------------

fn parse_action(tokens: &[String]) -> Result<Action, String> {
    if tokens.is_empty() {
        return Err("empty action".into());
    }
    match tokens[0].as_str() {
        "hold" => Ok(Action::Hold),
        "chase" => {
            let target = parse_target(&tokens[1..])?;
            Ok(Action::Chase(target))
        }
        "flee" => {
            let target = parse_target(&tokens[1..])?;
            Ok(Action::Flee(target))
        }
        "attack" => {
            let target = parse_target(&tokens[1..])?;
            Ok(Action::Attack(target))
        }
        "focus" => {
            let target = parse_target(&tokens[1..])?;
            Ok(Action::Focus(target))
        }
        "maintain_distance" => {
            if tokens.len() < 3 {
                return Err("maintain_distance requires target and range".into());
            }
            let range: f32 = tokens
                .last()
                .unwrap()
                .parse()
                .map_err(|_| "maintain_distance: last token must be a number (range)".to_string())?;
            let target = parse_target(&tokens[1..tokens.len() - 1])?;
            Ok(Action::MaintainDistance(target, range))
        }
        "cast" => {
            if tokens.len() < 4 {
                return Err("cast requires: cast abilityN on target".into());
            }
            let slot = parse_ability_slot(&tokens[1])?;
            if tokens[2] != "on" {
                return Err(format!("expected 'on' after ability slot, got '{}'", tokens[2]));
            }
            let target = parse_target(&tokens[3..])?;
            Ok(Action::CastAbility(slot, target))
        }
        "cast_if_ready" => {
            if tokens.len() < 4 {
                return Err("cast_if_ready requires: cast_if_ready abilityN on target".into());
            }
            let slot = parse_ability_slot(&tokens[1])?;
            if tokens[2] != "on" {
                return Err(format!("expected 'on' after ability slot, got '{}'", tokens[2]));
            }
            let target = parse_target(&tokens[3..])?;
            Ok(Action::CastIfReady(slot, target))
        }
        "use_best_ability" => {
            if tokens.len() >= 3 && tokens[1] == "on" {
                let target = parse_target(&tokens[2..])?;
                Ok(Action::UseBestAbilityOn(target))
            } else {
                Ok(Action::UseBestAbility)
            }
        }
        "use_ability_type" => {
            if tokens.len() < 2 {
                return Err("use_ability_type requires a category".into());
            }
            let cat = parse_ability_category(&tokens[1])?;
            if tokens.len() >= 4 && tokens[2] == "on" {
                let target = parse_target(&tokens[3..])?;
                Ok(Action::UseAbilityType(cat, Some(target)))
            } else {
                Ok(Action::UseAbilityType(cat, None))
            }
        }
        "run" => {
            if tokens.len() < 2 {
                return Err("run requires a behavior name".into());
            }
            let name = unquote(&tokens[1]);
            Ok(Action::Run(name))
        }
        "move_to" => {
            let pos = parse_position(&tokens[1..])?;
            Ok(Action::MoveTo(pos))
        }
        other => Err(format!("unknown action '{other}'")),
    }
}

// ---------------------------------------------------------------------------
// Target parsing
// ---------------------------------------------------------------------------

pub fn parse_target(tokens: &[String]) -> Result<Target, String> {
    if tokens.is_empty() {
        return Err("expected target".into());
    }
    match tokens[0].as_str() {
        "self" => Ok(Target::Self_),
        "nearest_enemy" => Ok(Target::NearestEnemy),
        "nearest_ally" => Ok(Target::NearestAlly),
        "lowest_hp_enemy" => Ok(Target::LowestHpEnemy),
        "lowest_hp_ally" => Ok(Target::LowestHpAlly),
        "highest_dps_enemy" => Ok(Target::HighestDpsEnemy),
        "highest_threat_enemy" => Ok(Target::HighestThreatEnemy),
        "casting_enemy" => Ok(Target::CastingEnemy),
        "enemy_attacking" => {
            let inner = parse_target(&tokens[1..])?;
            Ok(Target::EnemyAttacking(Box::new(inner)))
        }
        "tagged" => {
            if tokens.len() < 2 {
                return Err("tagged requires a name".into());
            }
            let name = unquote(&tokens[1]);
            Ok(Target::Tagged(name))
        }
        "unit" => {
            if tokens.len() < 2 {
                return Err("unit requires an ID".into());
            }
            let id: u32 = tokens[1]
                .parse()
                .map_err(|_| format!("invalid unit ID '{}'", tokens[1]))?;
            Ok(Target::UnitId(id))
        }
        other => Err(format!("unknown target '{other}'")),
    }
}

fn parse_position(tokens: &[String]) -> Result<Position, String> {
    if tokens.is_empty() {
        return Err("expected position".into());
    }
    match tokens[0].as_str() {
        "random_position" | "random" => Ok(Position::Random),
        "target_position" => Ok(Position::TargetPosition),
        "position" => {
            if tokens.len() < 3 {
                return Err("position requires two numbers".into());
            }
            let x: f32 = tokens[1].parse().map_err(|_| "invalid x coordinate")?;
            let y: f32 = tokens[2].parse().map_err(|_| "invalid y coordinate")?;
            Ok(Position::Fixed(x, y))
        }
        _ => {
            // Try parsing as entity target
            let target = parse_target(tokens)?;
            Ok(Position::Entity(target))
        }
    }
}

pub(super) fn parse_ability_slot(tok: &str) -> Result<usize, String> {
    tok.strip_prefix("ability")
        .and_then(|n| n.parse::<usize>().ok())
        .ok_or_else(|| format!("expected abilityN (e.g. ability0), got '{tok}'"))
}

fn parse_ability_category(tok: &str) -> Result<AbilityCategory, String> {
    match tok {
        "damage" => Ok(AbilityCategory::Damage),
        "heal" => Ok(AbilityCategory::Heal),
        "cc" => Ok(AbilityCategory::Cc),
        "buff" => Ok(AbilityCategory::Buff),
        "aoe" => Ok(AbilityCategory::Aoe),
        other => Err(format!("unknown ability category '{other}'")),
    }
}

fn unquote(s: &str) -> String {
    s.trim_matches('"').to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_stationary_dummy() {
        let input = r#"behavior "stationary_dummy" {
            fallback hold
        }"#;
        let tree = parse_behavior(input).unwrap();
        assert_eq!(tree.name, "stationary_dummy");
        assert_eq!(tree.rules.len(), 1);
        assert!(matches!(tree.rules[0].action, Action::Hold));
        assert!(tree.rules[0].condition.is_none());
    }

    #[test]
    fn parse_melee_chaser() {
        let input = r#"behavior "melee_chaser" {
            priority attack nearest_enemy when nearest_enemy.distance < 1.5
            default chase nearest_enemy
            fallback hold
        }"#;
        let tree = parse_behavior(input).unwrap();
        assert_eq!(tree.name, "melee_chaser");
        assert_eq!(tree.rules.len(), 3);
        assert!(matches!(tree.rules[0].action, Action::Attack(Target::NearestEnemy)));
        assert!(tree.rules[0].condition.is_some());
        assert!(matches!(tree.rules[1].action, Action::Chase(Target::NearestEnemy)));
        assert!(matches!(tree.rules[2].action, Action::Hold));
    }

    #[test]
    fn parse_healer_bot() {
        let input = r#"behavior "healer_bot" {
            priority cast_if_ready ability0 on lowest_hp_ally when lowest_hp_ally.hp_pct < 0.5 and ability_ready ability0
            default attack nearest_enemy when can_attack
            default chase nearest_enemy
            fallback hold
        }"#;
        let tree = parse_behavior(input).unwrap();
        assert_eq!(tree.name, "healer_bot");
        assert_eq!(tree.rules.len(), 4);
        assert!(matches!(tree.rules[0].action, Action::CastIfReady(0, Target::LowestHpAlly)));
        assert!(matches!(tree.rules[0].condition, Some(Condition::And(_, _))));
    }

    #[test]
    fn parse_aoe_caster() {
        let input = r#"behavior "aoe_caster" {
            priority cast_if_ready ability0 on nearest_enemy when ability_ready ability0
            default maintain_distance nearest_enemy 8
            fallback hold
        }"#;
        let tree = parse_behavior(input).unwrap();
        assert_eq!(tree.name, "aoe_caster");
        assert_eq!(tree.rules.len(), 3);
        assert!(matches!(tree.rules[1].action, Action::MaintainDistance(Target::NearestEnemy, r) if (r - 8.0).abs() < 0.01));
    }

    #[test]
    fn parse_error_no_brace() {
        let input = r#"behavior "broken""#;
        assert!(parse_behavior(input).is_err());
    }

    #[test]
    fn parse_error_unknown_action() {
        let input = r#"behavior "bad" {
            default explode
        }"#;
        assert!(parse_behavior(input).is_err());
    }
}
