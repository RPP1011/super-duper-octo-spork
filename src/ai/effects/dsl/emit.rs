//! Emit AbilityDef as DSL text (reverse of parsing).
//!
//! Used to generate DSL text for transformer training data.

use crate::ai::effects::defs::{AbilityDef, AbilityTargeting};
use crate::ai::effects::effect_enum::Effect;
use crate::ai::effects::types::*;

/// Convert an AbilityDef to its DSL text representation.
pub fn emit_ability_dsl(def: &AbilityDef) -> String {
    let mut lines = Vec::new();
    // Use underscore-joined name to avoid multi-word UNK issues in tokenizer
    let safe_name = def.name.replace(' ', "_");
    lines.push(format!("ability {safe_name} {{"));

    // Header line: target + range
    let target = match def.targeting {
        AbilityTargeting::TargetEnemy => "enemy",
        AbilityTargeting::TargetAlly => "ally",
        AbilityTargeting::SelfCast => "self",
        AbilityTargeting::SelfAoe => "self_aoe",
        AbilityTargeting::GroundTarget => "ground",
        AbilityTargeting::Direction => "direction",
        AbilityTargeting::Vector => "vector",
        AbilityTargeting::Global => "global",
    };
    if def.range > 0.0 {
        lines.push(format!("    target: {target}, range: {}", fmt_f32(def.range)));
    } else {
        lines.push(format!("    target: {target}"));
    }

    // Cooldown + cast
    let mut timing = Vec::new();
    if def.cooldown_ms > 0 {
        timing.push(format!("cooldown: {}", fmt_duration(def.cooldown_ms)));
    }
    if def.cast_time_ms > 0 {
        timing.push(format!("cast: {}", fmt_duration(def.cast_time_ms)));
    }
    if !timing.is_empty() {
        lines.push(format!("    {}", timing.join(", ")));
    }

    // Hint
    if !def.ai_hint.is_empty() {
        lines.push(format!("    hint: {}", def.ai_hint));
    }

    // Cost
    if def.resource_cost > 0 {
        lines.push(format!("    cost: {}", def.resource_cost));
    }

    // Charges
    if def.max_charges > 0 {
        lines.push(format!("    charges: {}", def.max_charges));
        if def.charge_recharge_ms > 0 {
            lines.push(format!("    recharge: {}", fmt_duration(def.charge_recharge_ms)));
        }
    }

    // Toggle
    if def.is_toggle {
        lines.push("    toggle".to_string());
    }

    // Recast
    if def.recast_count > 0 {
        lines.push(format!("    recast: {}", def.recast_count));
        if def.recast_window_ms > 0 {
            lines.push(format!("    recast_window: {}", fmt_duration(def.recast_window_ms)));
        }
    }

    // Unstoppable
    if def.unstoppable {
        lines.push("    unstoppable".to_string());
    }

    // Zone tag
    if let Some(ref tag) = def.zone_tag {
        lines.push(format!("    zone_tag: {tag}"));
    }

    // Blank line before effects
    if !def.effects.is_empty() || def.delivery.is_some() {
        lines.push(String::new());
    }

    // Delivery
    if let Some(ref delivery) = def.delivery {
        emit_delivery(delivery, &mut lines, 1);
    }

    // Top-level effects
    for eff in &def.effects {
        emit_conditional_effect(eff, &mut lines, 1);
    }

    lines.push("}".to_string());
    lines.join("\n")
}

fn emit_delivery(delivery: &Delivery, lines: &mut Vec<String>, indent: usize) {
    let pad = "    ".repeat(indent);
    match delivery {
        Delivery::Instant => {}
        Delivery::Projectile { speed, pierce, width, on_hit, on_arrival } => {
            let mut params = vec![format!("speed: {}", fmt_f32(*speed))];
            if *pierce { params.push("pierce".to_string()); }
            if *width > 0.0 { params.push(format!("width: {}", fmt_f32(*width))); }
            lines.push(format!("{pad}deliver projectile {{ {} }} {{", params.join(", ")));
            emit_delivery_hooks(on_hit, on_arrival, lines, indent + 1);
            lines.push(format!("{pad}}}"));
        }
        Delivery::Channel { duration_ms, tick_interval_ms } => {
            lines.push(format!("{pad}deliver channel {{ duration: {}, tick: {} }} {{",
                fmt_duration(*duration_ms), fmt_duration(*tick_interval_ms)));
            // Channel uses on_hit for tick effects
            lines.push(format!("{pad}}}"));
        }
        Delivery::Zone { duration_ms, tick_interval_ms } => {
            lines.push(format!("{pad}deliver zone {{ duration: {}, tick: {} }} {{",
                fmt_duration(*duration_ms), fmt_duration(*tick_interval_ms)));
            lines.push(format!("{pad}}}"));
        }
        Delivery::Tether { max_range, tick_interval_ms, on_complete } => {
            let mut params = vec![format!("range: {}", fmt_f32(*max_range))];
            if *tick_interval_ms > 0 {
                params.push(format!("tick: {}", fmt_duration(*tick_interval_ms)));
            }
            lines.push(format!("{pad}deliver tether {{ {} }} {{", params.join(", ")));
            if !on_complete.is_empty() {
                let inner_pad = "    ".repeat(indent + 1);
                lines.push(format!("{inner_pad}on_complete {{"));
                for eff in on_complete {
                    emit_conditional_effect(eff, lines, indent + 2);
                }
                lines.push(format!("{inner_pad}}}"));
            }
            lines.push(format!("{pad}}}"));
        }
        Delivery::Trap { duration_ms, trigger_radius, arm_time_ms } => {
            let mut params = vec![
                format!("duration: {}", fmt_duration(*duration_ms)),
                format!("radius: {}", fmt_f32(*trigger_radius)),
            ];
            if *arm_time_ms > 0 {
                params.push(format!("arm_time: {}", fmt_duration(*arm_time_ms)));
            }
            lines.push(format!("{pad}deliver trap {{ {} }}", params.join(", ")));
        }
        Delivery::Chain { bounces, bounce_range, falloff, on_hit } => {
            let mut params = vec![
                format!("bounces: {bounces}"),
                format!("range: {}", fmt_f32(*bounce_range)),
            ];
            if *falloff > 0.0 { params.push(format!("falloff: {}", fmt_f32(*falloff))); }
            lines.push(format!("{pad}deliver chain {{ {} }} {{", params.join(", ")));
            if !on_hit.is_empty() {
                let inner_pad = "    ".repeat(indent + 1);
                lines.push(format!("{inner_pad}on_hit {{"));
                for eff in on_hit {
                    emit_conditional_effect(eff, lines, indent + 2);
                }
                lines.push(format!("{inner_pad}}}"));
            }
            lines.push(format!("{pad}}}"));
        }
    }
}

fn emit_delivery_hooks(
    on_hit: &[ConditionalEffect],
    on_arrival: &[ConditionalEffect],
    lines: &mut Vec<String>,
    indent: usize,
) {
    let pad = "    ".repeat(indent);
    if !on_hit.is_empty() {
        lines.push(format!("{pad}on_hit {{"));
        for eff in on_hit {
            emit_conditional_effect(eff, lines, indent + 1);
        }
        lines.push(format!("{pad}}}"));
    }
    if !on_arrival.is_empty() {
        lines.push(format!("{pad}on_arrival {{"));
        for eff in on_arrival {
            emit_conditional_effect(eff, lines, indent + 1);
        }
        lines.push(format!("{pad}}}"));
    }
}

fn emit_conditional_effect(ce: &ConditionalEffect, lines: &mut Vec<String>, indent: usize) {
    let pad = "    ".repeat(indent);
    let mut parts = vec![emit_effect(&ce.effect)];

    // Area
    if let Some(ref area) = ce.area {
        parts.push(emit_area(area));
    }

    // Tags
    if !ce.tags.is_empty() {
        parts.push(emit_tags(&ce.tags));
    }

    // Condition
    if let Some(ref cond) = ce.condition {
        if !matches!(cond, Condition::Always) {
            parts.push(format!("when {}", emit_condition(cond)));
        }
    }

    // Scaling (from effect)
    let scaling = emit_scaling(&ce.effect);
    if !scaling.is_empty() {
        parts.push(scaling);
    }

    lines.push(format!("{pad}{}", parts.join(" ")));
}

fn emit_effect(effect: &Effect) -> String {
    match effect {
        Effect::Damage { amount, amount_per_tick, duration_ms, damage_type, .. } => {
            if *amount_per_tick > 0 && *duration_ms > 0 {
                let dt = match damage_type {
                    DamageType::Magic => " magic",
                    DamageType::True => " true",
                    _ => "",
                };
                format!("damage {amount_per_tick}/tick{dt} for {}", fmt_duration(*duration_ms))
            } else {
                let dt = match damage_type {
                    DamageType::Magic => " magic",
                    DamageType::True => " true",
                    _ => "",
                };
                format!("damage {amount}{dt}")
            }
        }
        Effect::Heal { amount, amount_per_tick, duration_ms, .. } => {
            if *amount_per_tick > 0 && *duration_ms > 0 {
                format!("heal {amount_per_tick}/tick for {}", fmt_duration(*duration_ms))
            } else {
                format!("heal {amount}")
            }
        }
        Effect::Shield { amount, duration_ms } => {
            format!("shield {amount} for {}", fmt_duration(*duration_ms))
        }
        Effect::Stun { duration_ms } => format!("stun {}", fmt_duration(*duration_ms)),
        Effect::Slow { factor, duration_ms } => {
            format!("slow {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::Root { duration_ms } => format!("root {}", fmt_duration(*duration_ms)),
        Effect::Silence { duration_ms } => format!("silence {}", fmt_duration(*duration_ms)),
        Effect::Fear { duration_ms } => format!("fear {}", fmt_duration(*duration_ms)),
        Effect::Taunt { duration_ms } => format!("taunt {}", fmt_duration(*duration_ms)),
        Effect::Knockback { distance } => format!("knockback {}", fmt_f32(*distance)),
        Effect::Pull { distance } => format!("pull {}", fmt_f32(*distance)),
        Effect::Dash { distance, to_target, is_blink, .. } => {
            let mut s = if *is_blink {
                format!("blink {}", fmt_f32(*distance))
            } else {
                format!("dash {}", fmt_f32(*distance))
            };
            if *to_target { s.push_str(" to_target"); }
            s
        }
        Effect::Buff { stat, factor, duration_ms } => {
            format!("buff {stat} {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::Debuff { stat, factor, duration_ms } => {
            format!("debuff {stat} {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::Duel { duration_ms } => format!("duel {}", fmt_duration(*duration_ms)),
        Effect::Summon { template, count, hp_percent, clone, directed, .. } => {
            let mut s = format!("summon \"{template}\"");
            if *count > 1 { s.push_str(&format!(" x{count}")); }
            if *clone { s.push_str(" clone"); }
            if *directed { s.push_str(" directed"); }
            if (*hp_percent - 100.0).abs() > 0.1 {
                s.push_str(&format!(" hp:{}%", *hp_percent as i32));
            }
            s
        }
        Effect::CommandSummons { speed } => format!("command_summons speed:{}", fmt_f32(*speed)),
        Effect::Dispel { target_tags } => {
            if target_tags.is_empty() {
                "dispel".to_string()
            } else {
                // Wrap tags as string literals so tokenizer maps them to [STR]
                let tags: Vec<String> = target_tags.iter()
                    .map(|t| format!("\"{t}\""))
                    .collect();
                format!("dispel {}", tags.join(" "))
            }
        }
        Effect::Reflect { percent, duration_ms } => {
            format!("reflect {} for {}", fmt_f32(*percent), fmt_duration(*duration_ms))
        }
        Effect::Lifesteal { percent, duration_ms } => {
            format!("lifesteal {} for {}", fmt_f32(*percent), fmt_duration(*duration_ms))
        }
        Effect::DamageModify { factor, duration_ms } => {
            format!("damage_modify {} for {}", fmt_f32(*factor), fmt_duration(*duration_ms))
        }
        Effect::SelfDamage { amount } => format!("self_damage {amount}"),
        Effect::Execute { hp_threshold_percent } => {
            format!("execute {}%", *hp_threshold_percent as i32)
        }
        Effect::Blind { miss_chance, duration_ms } => {
            format!("blind {} for {}", fmt_f32(*miss_chance), fmt_duration(*duration_ms))
        }
        Effect::OnHitBuff { duration_ms, .. } => {
            format!("on_hit_buff for {}", fmt_duration(*duration_ms))
        }
        Effect::Resurrect { hp_percent } => format!("resurrect {}%", *hp_percent as i32),
        Effect::OverhealShield { duration_ms, .. } => {
            format!("overheal_shield for {}", fmt_duration(*duration_ms))
        }
        Effect::AbsorbToHeal { shield_amount, duration_ms, .. } => {
            format!("absorb_to_heal {shield_amount} for {}", fmt_duration(*duration_ms))
        }
        Effect::ShieldSteal { amount } => format!("shield_steal {amount}"),
        Effect::StatusClone { max_count } => format!("status_clone {max_count}"),
        Effect::Immunity { immune_to, duration_ms } => {
            format!("immunity {} for {}", immune_to.join(","), fmt_duration(*duration_ms))
        }
        Effect::Detonate { damage_multiplier } => {
            format!("detonate {}", fmt_f32(*damage_multiplier))
        }
        Effect::StatusTransfer { steal_buffs } => {
            if *steal_buffs { "status_transfer steal".to_string() }
            else { "status_transfer".to_string() }
        }
        Effect::DeathMark { duration_ms, damage_percent } => {
            format!("death_mark {} for {}", fmt_f32(*damage_percent), fmt_duration(*duration_ms))
        }
        Effect::Polymorph { duration_ms } => format!("polymorph {}", fmt_duration(*duration_ms)),
        Effect::Banish { duration_ms } => format!("banish {}", fmt_duration(*duration_ms)),
        Effect::Confuse { duration_ms } => format!("confuse {}", fmt_duration(*duration_ms)),
        Effect::Charm { duration_ms } => format!("charm {}", fmt_duration(*duration_ms)),
        Effect::Stealth { duration_ms, break_on_damage, break_on_ability } => {
            let mut s = format!("stealth for {}", fmt_duration(*duration_ms));
            if *break_on_damage { s.push_str(" break_on_damage"); }
            if *break_on_ability { s.push_str(" break_on_ability"); }
            s
        }
        Effect::Leash { max_range, duration_ms } => {
            format!("leash {} for {}", fmt_f32(*max_range), fmt_duration(*duration_ms))
        }
        Effect::Link { duration_ms, share_percent } => {
            format!("link {}% for {}", *share_percent as i32, fmt_duration(*duration_ms))
        }
        Effect::Redirect { duration_ms, charges } => {
            format!("redirect {charges} for {}", fmt_duration(*duration_ms))
        }
        Effect::Rewind { lookback_ms } => format!("rewind {}", fmt_duration(*lookback_ms)),
        Effect::CooldownModify { amount_ms, ability_name } => {
            if let Some(ref name) = ability_name {
                format!("cooldown_modify {amount_ms}ms \"{name}\"")
            } else {
                format!("cooldown_modify {amount_ms}ms")
            }
        }
        Effect::ApplyStacks { name, count, max_stacks, duration_ms } => {
            let mut s = format!("apply_stacks \"{name}\" {count}/{max_stacks}");
            if *duration_ms > 0 {
                s.push_str(&format!(" for {}", fmt_duration(*duration_ms)));
            }
            s
        }
        Effect::Obstacle { width, height } => {
            format!("obstacle {} {}", fmt_f32(*width), fmt_f32(*height))
        }
        Effect::Suppress { duration_ms } => format!("suppress {}", fmt_duration(*duration_ms)),
        Effect::Grounded { duration_ms } => format!("grounded {}", fmt_duration(*duration_ms)),
        Effect::ProjectileBlock { duration_ms } => {
            format!("projectile_block {}", fmt_duration(*duration_ms))
        }
        Effect::Attach { duration_ms } => format!("attach for {}", fmt_duration(*duration_ms)),
        Effect::EvolveAbility { ability_index } => format!("evolve {ability_index}"),
        Effect::Swap => "swap".to_string(),
    }
}

fn emit_area(area: &Area) -> String {
    match area {
        Area::SingleTarget | Area::SelfOnly => String::new(),
        Area::Circle { radius } => format!("in circle({})", fmt_f32(*radius)),
        Area::Cone { radius, angle_deg } => {
            format!("in cone({}, {})", fmt_f32(*radius), fmt_f32(*angle_deg))
        }
        Area::Line { length, width } => {
            format!("in line({}, {})", fmt_f32(*length), fmt_f32(*width))
        }
        Area::Ring { inner_radius, outer_radius } => {
            format!("in ring({}, {})", fmt_f32(*inner_radius), fmt_f32(*outer_radius))
        }
        Area::Spread { radius, max_targets } => {
            if *max_targets > 0 {
                format!("in spread({}, {})", fmt_f32(*radius), max_targets)
            } else {
                format!("in spread({})", fmt_f32(*radius))
            }
        }
    }
}

fn emit_tags(tags: &Tags) -> String {
    if tags.is_empty() { return String::new(); }
    let mut pairs: Vec<_> = tags.iter().collect();
    pairs.sort_by_key(|(k, _)| k.to_string());
    let inner: Vec<String> = pairs.iter()
        .map(|(k, v)| format!("{k}: {}", **v as i32))
        .collect();
    format!("[{}]", inner.join(", "))
}

fn emit_condition(cond: &Condition) -> String {
    match cond {
        Condition::Always => String::new(),
        Condition::TargetHpBelow { percent } => format!("target_hp_below({}%)", *percent as i32),
        Condition::TargetHpAbove { percent } => format!("target_hp_above({}%)", *percent as i32),
        Condition::CasterHpBelow { percent } => format!("caster_hp_below({}%)", *percent as i32),
        Condition::CasterHpAbove { percent } => format!("caster_hp_above({}%)", *percent as i32),
        Condition::TargetIsStunned => "target_is_stunned".to_string(),
        Condition::TargetIsSlowed => "target_is_slowed".to_string(),
        Condition::TargetIsRooted => "target_is_rooted".to_string(),
        Condition::TargetIsSilenced => "target_is_silenced".to_string(),
        Condition::TargetIsFeared => "target_is_feared".to_string(),
        Condition::TargetIsTaunted => "target_is_taunted".to_string(),
        Condition::TargetIsBanished => "target_is_banished".to_string(),
        Condition::TargetIsStealthed => "target_is_stealthed".to_string(),
        Condition::TargetIsCharmed => "target_is_charmed".to_string(),
        Condition::TargetIsPolymorphed => "target_is_polymorphed".to_string(),
        Condition::HitCountAbove { count } => format!("hit_count_above({count})"),
        Condition::TargetHasTag { tag } => format!("target_has_tag(\"{tag}\")"),
        Condition::CasterHasStatus { status } => format!("caster_has_status(\"{status}\")"),
        Condition::TargetHasStatus { status } => format!("target_has_status(\"{status}\")"),
        Condition::TargetDebuffCount { min_count } => format!("target_debuff_count({min_count})"),
        Condition::CasterBuffCount { min_count } => format!("caster_buff_count({min_count})"),
        Condition::AllyCountBelow { count } => format!("ally_count_below({count})"),
        Condition::EnemyCountBelow { count } => format!("enemy_count_below({count})"),
        Condition::TargetStackCount { name, min_count } => {
            format!("target_stack_count(\"{name}\", {min_count})")
        }
        Condition::TargetDistanceBelow { range } => {
            format!("target_distance_below({})", fmt_f32(*range))
        }
        Condition::TargetDistanceAbove { range } => {
            format!("target_distance_above({})", fmt_f32(*range))
        }
        Condition::CasterResourceBelow { percent } => {
            format!("caster_resource_below({}%)", *percent as i32)
        }
        Condition::CasterResourceAbove { percent } => {
            format!("caster_resource_above({}%)", *percent as i32)
        }
        Condition::And { conditions } => {
            let inner: Vec<String> = conditions.iter().map(emit_condition).collect();
            format!("and({})", inner.join(", "))
        }
        Condition::Or { conditions } => {
            let inner: Vec<String> = conditions.iter().map(emit_condition).collect();
            format!("or({})", inner.join(", "))
        }
        Condition::Not { condition } => {
            format!("not({})", emit_condition(condition))
        }
    }
}

fn emit_scaling(effect: &Effect) -> String {
    let bonus = match effect {
        Effect::Damage { bonus, scaling_stat, scaling_percent, .. } => {
            if let Some(ref stat) = scaling_stat {
                if *scaling_percent > 0.0 {
                    return format!("+ {}% {stat}", *scaling_percent as i32);
                }
            }
            bonus
        }
        Effect::Heal { bonus, scaling_stat, scaling_percent, .. } => {
            if let Some(ref stat) = scaling_stat {
                if *scaling_percent > 0.0 {
                    return format!("+ {}% {stat}", *scaling_percent as i32);
                }
            }
            bonus
        }
        _ => return String::new(),
    };
    if bonus.is_empty() { return String::new(); }
    let terms: Vec<String> = bonus.iter().map(|t| {
        let stat_name = match &t.stat {
            StatRef::TargetMaxHp => "target_max_hp",
            StatRef::TargetCurrentHp => "target_current_hp",
            StatRef::TargetMissingHp => "target_missing_hp",
            StatRef::CasterMaxHp => "caster_max_hp",
            StatRef::CasterCurrentHp => "caster_current_hp",
            StatRef::CasterMissingHp => "caster_missing_hp",
            StatRef::CasterAttackDamage => "caster_attack_damage",
            StatRef::TargetStacks { name } => return format!("{}% target_stacks({name})", t.percent as i32),
            StatRef::CasterStacks { name } => return format!("{}% caster_stacks({name})", t.percent as i32),
        };
        format!("{}% {stat_name}", t.percent as i32)
    }).collect();
    format!("+ {}", terms.join(" + "))
}

fn fmt_duration(ms: u32) -> String {
    if ms == 0 {
        "0ms".to_string()
    } else if ms % 1000 == 0 {
        format!("{}s", ms / 1000)
    } else {
        format!("{ms}ms")
    }
}

fn fmt_f32(v: f32) -> String {
    if v == v.floor() && v.abs() < 10000.0 {
        format!("{:.1}", v)
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_simple_ability() {
        let def = AbilityDef {
            name: "Fireball".to_string(),
            targeting: AbilityTargeting::TargetEnemy,
            range: 5.0,
            cooldown_ms: 5000,
            cast_time_ms: 300,
            ai_hint: "damage".to_string(),
            effects: vec![ConditionalEffect {
                effect: Effect::Damage {
                    amount: 55,
                    amount_per_tick: 0,
                    duration_ms: 0,
                    tick_interval_ms: 0,
                    scaling_stat: None,
                    scaling_percent: 0.0,
                    damage_type: DamageType::Physical,
                    bonus: vec![],
                },
                condition: None,
                area: None,
                tags: [("FIRE".to_string(), 60.0)].into_iter().collect(),
                stacking: Stacking::default(),
                chance: 0.0,
                else_effects: vec![],
            }],
            ..Default::default()
        };

        let dsl = emit_ability_dsl(&def);
        assert!(dsl.contains("ability Fireball {"));
        assert!(dsl.contains("target: enemy, range: 5.0"));
        assert!(dsl.contains("cooldown: 5s, cast: 300ms"));
        assert!(dsl.contains("hint: damage"));
        assert!(dsl.contains("damage 55"));
        assert!(dsl.contains("[FIRE: 60]"));
    }

    #[test]
    fn test_roundtrip_hero_abilities() {
        // Parse real hero ability files, emit DSL, parse again
        let dir = std::path::Path::new("assets/hero_templates");
        let mut total = 0;
        let mut roundtripped = 0;
        for entry in std::fs::read_dir(dir).expect("read_dir failed") {
            let path = entry.unwrap().path();
            if path.extension().map_or(true, |e| e != "ability") { continue; }
            let source = std::fs::read_to_string(&path).unwrap();
            let (abilities, _passives) = match crate::ai::effects::dsl::parse_abilities(&source) {
                Ok(r) => r,
                Err(_) => continue,
            };
            for def in &abilities {
                total += 1;
                let emitted = emit_ability_dsl(def);
                match crate::ai::effects::dsl::parse_abilities(&emitted) {
                    Ok((reparsed, _)) => {
                        assert_eq!(reparsed.len(), 1, "Expected 1 ability from re-parse of:\n{emitted}");
                        let r = &reparsed[0];
                        assert_eq!(r.name, def.name);
                        assert_eq!(r.ai_hint, def.ai_hint);
                        assert_eq!(r.cooldown_ms, def.cooldown_ms);
                        assert_eq!(r.cast_time_ms, def.cast_time_ms);
                        roundtripped += 1;
                    }
                    Err(e) => {
                        panic!("Failed to re-parse emitted DSL for {}:\n{emitted}\nError: {e}", def.name);
                    }
                }
            }
        }
        println!("Round-tripped {roundtripped}/{total} abilities from hero templates");
        assert!(total > 0, "No hero ability files found");
    }

    #[test]
    fn test_emit_duration_formatting() {
        assert_eq!(fmt_duration(1000), "1s");
        assert_eq!(fmt_duration(500), "500ms");
        assert_eq!(fmt_duration(2500), "2500ms");
        assert_eq!(fmt_duration(0), "0ms");
    }
}
