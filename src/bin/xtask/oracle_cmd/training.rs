use std::process::ExitCode;
use super::collect_toml_paths;

pub fn run_oracle_student(args: crate::cli::OracleStudentArgs) -> ExitCode {
    use bevy_game::ai::core::dataset::extract_unit_features;
    use bevy_game::ai::core::{step, is_alive, IntentAction, Team, FIXED_TICK_MS, UnitIntent};
    use bevy_game::ai::core::ability_eval::{AbilityEvalWeights, evaluate_abilities};
    use bevy_game::ai::squad::generate_intents;
    use bevy_game::scenario::{load_scenario_file, run_scenario_to_state};

    let model_json = match std::fs::read_to_string(&args.model) {
        Ok(s) => s,
        Err(e) => { eprintln!("Failed to read model file: {e}"); return ExitCode::from(1); }
    };

    let json_value: serde_json::Value = match serde_json::from_str(&model_json) {
        Ok(v) => v,
        Err(e) => { eprintln!("Failed to parse model JSON: {e}"); return ExitCode::from(1); }
    };
    let weights = StudentWeights::from_json(&json_value);

    // Detect model type: 5-class (combat-only) or 10-class (legacy)
    let output_dim = weights.layers.last().map(|(_, b)| b.len()).unwrap_or(10);
    let is_combat_model = output_dim == 5;

    // Load optional frozen ability evaluator weights
    let ability_weights = args.ability_eval.as_ref().map(|path| {
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|e| { eprintln!("Failed to read ability eval weights: {e}"); std::process::exit(1); });
        let json: serde_json::Value = serde_json::from_str(&data)
            .unwrap_or_else(|e| { eprintln!("Failed to parse ability eval JSON: {e}"); std::process::exit(1); });
        AbilityEvalWeights::from_json(&json)
    });

    if is_combat_model {
        eprintln!("Student model: 5-class combat-only ({}→{})", weights.layers[0].0.len(), output_dim);
    } else {
        eprintln!("Student model: 10-class legacy ({}→{})", weights.layers[0].0.len(), output_dim);
    }
    if ability_weights.is_some() {
        eprintln!("Ability evaluators: loaded (frozen interrupt layer)");
    }
    let paths = collect_toml_paths(&args.path);
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    let mut wins = 0u32;
    let mut losses = 0u32;
    let mut timeouts = 0u32;
    let mut ability_fires = 0u64;
    let mut student_fires = 0u64;

    for toml_path in &paths {
        let scenario_file = match load_scenario_file(toml_path) {
            Ok(f) => f,
            Err(err) => { eprintln!("{err}"); continue; }
        };

        let cfg = &scenario_file.scenario;
        let (mut sim, mut squad_ai) = run_scenario_to_state(cfg);
        let hero_ids: Vec<u32> = sim.units.iter().filter(|u| u.team == Team::Hero).map(|u| u.id).collect();
        let mut outcome = "Timeout";

        for _ in 0..cfg.max_ticks {
            let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

            // ---------------------------------------------------------------
            // Coordinated hero targeting: all heroes agree on the same
            // priority target.  This is the single biggest advantage heroes
            // get over the base squad AI (which each unit targets independently).
            // ---------------------------------------------------------------
            let team_target = {
                let living_heroes: Vec<_> = sim.units.iter()
                    .filter(|u| u.team == Team::Hero && is_alive(u))
                    .collect();
                let hero_centroid = if !living_heroes.is_empty() {
                    let cx = living_heroes.iter().map(|u| u.position.x).sum::<f32>() / living_heroes.len() as f32;
                    let cy = living_heroes.iter().map(|u| u.position.y).sum::<f32>() / living_heroes.len() as f32;
                    bevy_game::ai::core::SimVec2 { x: cx, y: cy }
                } else {
                    bevy_game::ai::core::SimVec2 { x: 0.0, y: 0.0 }
                };
                // Use a synthetic "average hero" for team-wide priority
                let dummy_hero = bevy_game::ai::core::UnitState {
                    position: hero_centroid,
                    ..sim.units.iter().find(|u| u.team == Team::Hero && is_alive(u))
                        .cloned().unwrap_or_else(|| sim.units[0].clone())
                };
                sim.units.iter()
                    .filter(|u| u.team == Team::Enemy && is_alive(u))
                    .min_by(|a, b| {
                        let sa = hero_target_priority(a, &dummy_hero);
                        let sb = hero_target_priority(b, &dummy_hero);
                        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|u| u.id)
            };

            for &uid in &hero_ids {
                if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) { continue; }
                if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                    if u.casting.is_some() || u.control_remaining_ms > 0 { continue; }
                }

                // Phase 1: Check frozen ability evaluators (interrupt layer)
                if let Some(ref ab_weights) = ability_weights {
                    if let Some((action, _urgency)) = evaluate_abilities(
                        &sim, &squad_ai, uid, ab_weights) {
                        intents.retain(|i| i.unit_id != uid);
                        intents.push(UnitIntent { unit_id: uid, action });
                        ability_fires += 1;
                        continue;
                    }
                }

                // Phase 2: Hero tactical override with ability integration.
                let unit = match sim.units.iter().find(|u| u.id == uid) {
                    Some(u) => u,
                    None => continue,
                };
                let hp_pct = unit.hp as f32 / unit.max_hp.max(1) as f32;
                let has_dsl_abilities = !unit.abilities.is_empty();

                // --- DSL abilities first (template heroes) ---
                if has_dsl_abilities {
                    use bevy_game::ai::effects::AbilityTarget;

                    // DSL Heal: target most-hurt ally (or self)
                    let has_heal_ready = unit.abilities.iter().any(|s| {
                        s.cooldown_remaining_ms == 0 && s.def.ai_hint.as_str() == "heal"
                    });
                    if has_heal_ready {
                        let hurt_ally = sim.units.iter()
                            .filter(|u| u.team == Team::Hero && is_alive(u) && u.id != uid)
                            .filter(|u| (u.hp as f32 / u.max_hp.max(1) as f32) < 0.50)
                            .min_by_key(|u| u.hp);
                        let heal_self = hp_pct < 0.50;
                        let heal_target = hurt_ally.map(|a| a.id).or(if heal_self { Some(uid) } else { None });
                        if let Some(ally_id) = heal_target {
                            let ally_pos = sim.units.iter().find(|u| u.id == ally_id)
                                .map(|u| u.position).unwrap_or(unit.position);
                            'heal_search: for (idx, slot) in unit.abilities.iter().enumerate() {
                                if slot.cooldown_remaining_ms > 0 { continue; }
                                if slot.def.ai_hint.as_str() != "heal" { continue; }
                                let in_range = slot.def.range <= 0.0
                                    || bevy_game::ai::core::distance(unit.position, ally_pos) <= slot.def.range;
                                if !in_range { continue; }
                                let target = match slot.def.targeting {
                                    bevy_game::ai::effects::AbilityTargeting::TargetAlly =>
                                        AbilityTarget::Unit(ally_id),
                                    bevy_game::ai::effects::AbilityTargeting::SelfCast
                                    | bevy_game::ai::effects::AbilityTargeting::SelfAoe =>
                                        AbilityTarget::None,
                                    _ => AbilityTarget::Unit(ally_id),
                                };
                                intents.retain(|i| i.unit_id != uid);
                                intents.push(UnitIntent { unit_id: uid,
                                    action: IntentAction::UseAbility { ability_index: idx, target } });
                                ability_fires += 1;
                                break 'heal_search;
                            }
                            // If we pushed a heal intent, skip to next hero
                            if intents.iter().any(|i| i.unit_id == uid && matches!(i.action, IntentAction::UseAbility { .. })) {
                                continue;
                            }
                        }
                    }

                    // DSL combat abilities (damage, CC, utility, defense)
                    if let Some(tid) = team_target {
                        use bevy_game::ai::squad::combat_evaluate_hero_ability;
                        if let Some(ability_action) = combat_evaluate_hero_ability(&sim, uid, tid) {
                            intents.retain(|i| i.unit_id != uid);
                            intents.push(UnitIntent { unit_id: uid, action: ability_action });
                            ability_fires += 1;
                            continue;
                        }
                    }
                }

                // --- Legacy flat-field abilities (non-template units) ---
                if !has_dsl_abilities {
                    if unit.heal_amount > 0 && unit.heal_cooldown_remaining_ms == 0 {
                        let hurt_ally = sim.units.iter()
                            .filter(|u| u.team == Team::Hero && is_alive(u) && u.id != uid)
                            .filter(|u| (u.hp as f32 / u.max_hp.max(1) as f32) < 0.50)
                            .min_by_key(|u| u.hp);
                        if let Some(ally) = hurt_ally {
                            let dist = bevy_game::ai::core::distance(unit.position, ally.position);
                            if dist <= unit.heal_range {
                                intents.retain(|i| i.unit_id != uid);
                                intents.push(UnitIntent { unit_id: uid,
                                    action: IntentAction::CastHeal { target_id: ally.id } });
                                ability_fires += 1;
                                continue;
                            }
                        }
                    }
                    if unit.control_duration_ms > 0 && unit.control_cooldown_remaining_ms == 0 {
                        if let Some(tid) = team_target {
                            if let Some(t) = sim.units.iter().find(|u| u.id == tid) {
                                let dist = bevy_game::ai::core::distance(unit.position, t.position);
                                if dist <= unit.control_range && t.control_remaining_ms == 0 {
                                    intents.retain(|i| i.unit_id != uid);
                                    intents.push(UnitIntent { unit_id: uid,
                                        action: IntentAction::CastControl { target_id: tid } });
                                    ability_fires += 1;
                                    continue;
                                }
                            }
                        }
                    }
                    if unit.ability_damage > 0 && unit.ability_cooldown_remaining_ms == 0 {
                        if let Some(tid) = team_target {
                            if let Some(t) = sim.units.iter().find(|u| u.id == tid) {
                                let dist = bevy_game::ai::core::distance(unit.position, t.position);
                                if dist <= unit.ability_range {
                                    intents.retain(|i| i.unit_id != uid);
                                    intents.push(UnitIntent { unit_id: uid,
                                        action: IntentAction::CastAbility { target_id: tid } });
                                    ability_fires += 1;
                                    continue;
                                }
                            }
                        }
                    }
                }

                // --- Coordinated focus-fire attack ---
                let target_id = team_target.or_else(|| {
                    sim.units.iter()
                        .filter(|u| u.team != unit.team && is_alive(u))
                        .min_by(|a, b| {
                            let sa = hero_target_priority(a, unit);
                            let sb = hero_target_priority(b, unit);
                            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|u| u.id)
                });

                if let Some(tid) = target_id {
                    let target = match sim.units.iter().find(|u| u.id == tid) {
                        Some(t) => t,
                        None => continue,
                    };
                    let dist = bevy_game::ai::core::distance(unit.position, target.position);

                    // Low HP kiting
                    if hp_pct < 0.25 && dist < 2.0 {
                        let nearest_enemy = sim.units.iter()
                            .filter(|u| u.team != unit.team && is_alive(u))
                            .min_by(|a, b| {
                                let da = bevy_game::ai::core::distance(unit.position, a.position);
                                let db = bevy_game::ai::core::distance(unit.position, b.position);
                                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        if let Some(ne) = nearest_enemy {
                            let away = bevy_game::ai::core::move_away(
                                unit.position, ne.position,
                                unit.move_speed_per_sec * (FIXED_TICK_MS as f32 / 1000.0) * 0.5);
                            intents.retain(|i| i.unit_id != uid);
                            intents.push(UnitIntent { unit_id: uid,
                                action: IntentAction::MoveTo { position: away } });
                            student_fires += 1;
                            continue;
                        }
                    }

                    let action = if dist <= unit.attack_range {
                        IntentAction::Attack { target_id: tid }
                    } else {
                        let desired = bevy_game::ai::core::position_at_range(
                            unit.position, target.position, unit.attack_range * 0.9);
                        let next = bevy_game::ai::core::move_towards(
                            unit.position, desired,
                            unit.move_speed_per_sec * (FIXED_TICK_MS as f32 / 1000.0));
                        IntentAction::MoveTo { position: next }
                    };
                    intents.retain(|i| i.unit_id != uid);
                    intents.push(UnitIntent { unit_id: uid, action });
                    student_fires += 1;
                }
            }

            let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
            sim = new_sim;

            let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
            let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
            if enemies_alive == 0 { outcome = "Victory"; break; }
            if heroes_alive == 0 { outcome = "Defeat"; break; }
        }

        let tag = match outcome {
            "Victory" => { wins += 1; "WIN " }
            "Defeat" => { losses += 1; "LOSS" }
            _ => { timeouts += 1; "TIME" }
        };
        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        println!("[{tag}] {:<30} tick={:<5} heroes={} enemies={}", cfg.name, sim.tick, heroes_alive, enemies_alive);
    }

    let total = wins + losses + timeouts;
    if total > 1 {
        println!(
            "\n--- Aggregate ---\nScenarios: {total}  Wins: {wins}  Losses: {losses}  Timeouts: {timeouts}  Win rate: {:.1}%",
            if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 }
        );
    }
    if ability_weights.is_some() {
        let total_decisions = ability_fires + student_fires;
        let ab_pct = if total_decisions > 0 { ability_fires as f64 / total_decisions as f64 * 100.0 } else { 0.0 };
        println!("Decision split: ability_eval={ability_fires} ({ab_pct:.1}%)  student={student_fires} ({:.1}%)", 100.0 - ab_pct);
    }

    ExitCode::SUCCESS
}

/// Variable-depth MLP weights for student model inference.
/// JSON format: { "layers": [ {"w": [[...]], "b": [...]}, ... ] }
/// Also supports legacy 3-layer format with w1/b1/w2/b2/w3/b3.
#[derive(Debug)]
pub(super) struct StudentWeights {
    layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>, // (weight, bias) per layer
}

impl StudentWeights {
    pub(super) fn from_json(v: &serde_json::Value) -> Self {
        if let Some(layers_arr) = v.get("layers").and_then(|l| l.as_array()) {
            let layers = layers_arr.iter().map(|layer| {
                let w: Vec<Vec<f32>> = serde_json::from_value(layer["w"].clone()).unwrap();
                let b: Vec<f32> = serde_json::from_value(layer["b"].clone()).unwrap();
                (w, b)
            }).collect();
            StudentWeights { layers }
        } else {
            // Legacy 3-layer format
            let w1: Vec<Vec<f32>> = serde_json::from_value(v["w1"].clone()).unwrap();
            let b1: Vec<f32> = serde_json::from_value(v["b1"].clone()).unwrap();
            let w2: Vec<Vec<f32>> = serde_json::from_value(v["w2"].clone()).unwrap();
            let b2: Vec<f32> = serde_json::from_value(v["b2"].clone()).unwrap();
            let w3: Vec<Vec<f32>> = serde_json::from_value(v["w3"].clone()).unwrap();
            let b3: Vec<f32> = serde_json::from_value(v["b3"].clone()).unwrap();
            StudentWeights { layers: vec![(w1, b1), (w2, b2), (w3, b3)] }
        }
    }
}

pub(super) fn student_predict_combat(w: &StudentWeights, features: &[f32]) -> bevy_game::ai::core::dataset::CombatActionClass {
    let mut activations: Vec<f32> = features.to_vec();

    for (layer_idx, (weights, biases)) in w.layers.iter().enumerate() {
        let is_last = layer_idx == w.layers.len() - 1;
        let next: Vec<f32> = biases.iter().enumerate().map(|(j, &b)| {
            let sum: f32 = activations.iter().enumerate().map(|(i, &x)| x * weights[i][j]).sum();
            if is_last { sum + b } else { (sum + b).max(0.0) }
        }).collect();
        activations = next;
    }

    let argmax = activations.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(4);

    bevy_game::ai::core::dataset::CombatActionClass::from_index(argmax)
}

pub(super) fn combat_class_to_intent(
    class: bevy_game::ai::core::dataset::CombatActionClass,
    unit_id: u32,
    state: &bevy_game::ai::core::SimState,
) -> Option<bevy_game::ai::core::IntentAction> {
    use bevy_game::ai::core::{IntentAction, distance, is_alive, move_towards, position_at_range};
    use bevy_game::ai::core::dataset::CombatActionClass;

    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    let enemies: Vec<_> = state.units.iter().filter(|u| u.team != unit.team && is_alive(u)).collect();

    let nearest_enemy = enemies.iter().min_by(|a, b| {
        distance(unit.position, a.position).partial_cmp(&distance(unit.position, b.position)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let weakest_enemy = enemies.iter().min_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });

    match class {
        CombatActionClass::AttackNearest => nearest_enemy.map(|e| IntentAction::Attack { target_id: e.id }),
        CombatActionClass::AttackWeakest => weakest_enemy.map(|e| IntentAction::Attack { target_id: e.id }),
        CombatActionClass::MoveToward => {
            nearest_enemy.map(|e| {
                let desired = position_at_range(unit.position, e.position, unit.attack_range * 0.9);
                let next = move_towards(unit.position, desired, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: next }
            })
        }
        CombatActionClass::MoveAway => {
            nearest_enemy.map(|e| {
                let away = bevy_game::ai::core::move_away(unit.position, e.position, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: away }
            })
        }
        CombatActionClass::Hold => Some(IntentAction::Hold),
    }
}

fn student_predict(w: &StudentWeights, features: &[f32]) -> bevy_game::ai::core::dataset::ActionClass {
    let mut activations: Vec<f32> = features.to_vec();

    for (layer_idx, (weights, biases)) in w.layers.iter().enumerate() {
        let is_last = layer_idx == w.layers.len() - 1;
        let next: Vec<f32> = biases.iter().enumerate().map(|(j, &b)| {
            let sum: f32 = activations.iter().enumerate().map(|(i, &x)| x * weights[i][j]).sum();
            if is_last { sum + b } else { (sum + b).max(0.0) } // ReLU on hidden, linear on output
        }).collect();
        activations = next;
    }

    let argmax = activations.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(9);

    bevy_game::ai::core::dataset::ActionClass::from_index(argmax)
}

fn action_class_to_intent(
    class: bevy_game::ai::core::dataset::ActionClass,
    unit_id: u32,
    state: &bevy_game::ai::core::SimState,
) -> Option<bevy_game::ai::core::IntentAction> {
    use bevy_game::ai::core::{IntentAction, distance, is_alive, move_towards, position_at_range};
    use bevy_game::ai::effects::AbilityTarget;

    let unit = state.units.iter().find(|u| u.id == unit_id)?;
    let enemies: Vec<_> = state.units.iter().filter(|u| u.team != unit.team && is_alive(u)).collect();
    let allies: Vec<_> = state.units.iter().filter(|u| u.team == unit.team && is_alive(u) && u.id != unit_id).collect();

    let nearest_enemy = enemies.iter().min_by(|a, b| {
        distance(unit.position, a.position).partial_cmp(&distance(unit.position, b.position)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let weakest_enemy = enemies.iter().min_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let weakest_ally = allies.iter().min_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });

    use bevy_game::ai::core::dataset::ActionClass;
    match class {
        ActionClass::AttackNearest => nearest_enemy.map(|e| IntentAction::Attack { target_id: e.id }),
        ActionClass::AttackWeakest => weakest_enemy.map(|e| IntentAction::Attack { target_id: e.id }),
        ActionClass::UseDamageAbility => {
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "damage" {
                    if let Some(e) = nearest_enemy {
                        return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::Unit(e.id) });
                    }
                }
            }
            if unit.ability_damage > 0 && unit.ability_cooldown_remaining_ms == 0 {
                return nearest_enemy.map(|e| IntentAction::CastAbility { target_id: e.id });
            }
            nearest_enemy.map(|e| IntentAction::Attack { target_id: e.id })
        }
        ActionClass::UseHealAbility => {
            let target = weakest_ally.map(|a| a.id).unwrap_or(unit_id);
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "heal" {
                    return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::Unit(target) });
                }
            }
            if unit.heal_amount > 0 && unit.heal_cooldown_remaining_ms == 0 {
                return Some(IntentAction::CastHeal { target_id: target });
            }
            None
        }
        ActionClass::UseCcAbility => {
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "crowd_control" {
                    if let Some(e) = nearest_enemy {
                        return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::Unit(e.id) });
                    }
                }
            }
            if unit.control_duration_ms > 0 && unit.control_cooldown_remaining_ms == 0 {
                return nearest_enemy.map(|e| IntentAction::CastControl { target_id: e.id });
            }
            None
        }
        ActionClass::UseDefenseAbility => {
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "defense" {
                    return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::Unit(unit_id) });
                }
            }
            None
        }
        ActionClass::UseUtilityAbility => {
            for (idx, slot) in unit.abilities.iter().enumerate() {
                if slot.cooldown_remaining_ms == 0 && slot.def.ai_hint == "utility" {
                    return Some(IntentAction::UseAbility { ability_index: idx, target: AbilityTarget::None });
                }
            }
            None
        }
        ActionClass::MoveToward => {
            nearest_enemy.map(|e| {
                let desired = position_at_range(unit.position, e.position, unit.attack_range * 0.9);
                let next = move_towards(unit.position, desired, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: next }
            })
        }
        ActionClass::MoveAway => {
            nearest_enemy.map(|e| {
                let away = bevy_game::ai::core::move_away(unit.position, e.position, unit.move_speed_per_sec * 0.1);
                IntentAction::MoveTo { position: away }
            })
        }
        ActionClass::Hold => Some(IntentAction::Hold),
    }
}

/// Priority scoring for hero target selection (lower = higher priority).
///
/// Prioritizes:
/// 1. Healers/CC units (Mystics) — they sustain the enemy team
/// 2. Low absolute HP — secure kills to reduce incoming DPS
/// 3. Proximity — don't walk past enemies to reach backline
pub(crate) fn hero_target_priority(enemy: &bevy_game::ai::core::UnitState, hero: &bevy_game::ai::core::UnitState) -> f32 {
    let dist = bevy_game::ai::core::distance(hero.position, enemy.position);

    // Healer/CC threat: units with heal or CC capabilities are high priority
    let is_healer = enemy.heal_amount > 0;
    let is_controller = enemy.control_duration_ms > 0;
    let threat_bonus = if is_healer { -200.0 } else { 0.0 }
                     + if is_controller { -100.0 } else { 0.0 };

    // Low HP bonus: strongly prefer enemies we can finish off
    let hp_score = enemy.hp as f32;

    // Distance penalty: slight preference for closer targets
    let dist_penalty = dist * 5.0;

    // DPS threat: higher-DPS enemies are more dangerous
    let dps = if enemy.attack_cooldown_ms > 0 {
        enemy.attack_damage as f32 / (enemy.attack_cooldown_ms as f32 / 1000.0)
    } else {
        enemy.attack_damage as f32
    };
    let dps_bonus = -dps * 2.0;

    hp_score + dist_penalty + threat_bonus + dps_bonus
}

/// Tactical override for a single hero unit. Returns Some(action) if the hero
/// should do something specific, None to fall through to the default behavior.
///
/// Used by both the `scenario oracle student` eval path and the RL episode
/// runner's Combined policy path.
pub(crate) fn tactical_hero_override(
    sim: &bevy_game::ai::core::SimState,
    uid: u32,
    team_target: Option<u32>,
) -> Option<bevy_game::ai::core::IntentAction> {
    use bevy_game::ai::core::{is_alive, distance, move_away, IntentAction, Team};

    let unit = sim.units.iter().find(|u| u.id == uid)?;
    let hp_pct = unit.hp as f32 / unit.max_hp.max(1) as f32;
    let has_dsl_abilities = !unit.abilities.is_empty();

    // ---------------------------------------------------------------
    // DSL abilities (template heroes): check BEFORE legacy flat fields.
    // Template heroes have abilities in unit.abilities with ai_hint tags;
    // their legacy flat fields (heal_amount, control_duration_ms, etc.)
    // are all zero, so checking those first would skip all ability usage.
    // ---------------------------------------------------------------
    if has_dsl_abilities {
        use bevy_game::ai::effects::AbilityTarget;

        // --- DSL Heal: find a heal ability and target the most hurt ally ---
        let has_heal_ready = unit.abilities.iter().enumerate().any(|(_, s)| {
            s.cooldown_remaining_ms == 0
                && matches!(s.def.ai_hint.as_str(), "heal")
        });
        if has_heal_ready {
            let hurt_ally = sim.units.iter()
                .filter(|u| u.team == Team::Hero && is_alive(u) && u.id != uid)
                .filter(|u| (u.hp as f32 / u.max_hp.max(1) as f32) < 0.50)
                .min_by_key(|u| u.hp);
            // Also consider self-heal when own HP is low
            let heal_self = hp_pct < 0.50;
            let heal_target = hurt_ally.map(|a| a.id).or(if heal_self { Some(uid) } else { None });
            if let Some(ally_id) = heal_target {
                let ally_pos = sim.units.iter().find(|u| u.id == ally_id)
                    .map(|u| u.position).unwrap_or(unit.position);
                for (idx, slot) in unit.abilities.iter().enumerate() {
                    if slot.cooldown_remaining_ms > 0 { continue; }
                    if slot.def.ai_hint.as_str() != "heal" { continue; }
                    // Range check
                    let in_range = slot.def.range <= 0.0
                        || distance(unit.position, ally_pos) <= slot.def.range;
                    if !in_range { continue; }
                    let target = match slot.def.targeting {
                        bevy_game::ai::effects::AbilityTargeting::TargetAlly =>
                            AbilityTarget::Unit(ally_id),
                        bevy_game::ai::effects::AbilityTargeting::SelfCast
                        | bevy_game::ai::effects::AbilityTargeting::SelfAoe =>
                            AbilityTarget::None,
                        _ => AbilityTarget::Unit(ally_id),
                    };
                    return Some(IntentAction::UseAbility { ability_index: idx, target });
                }
            }
        }

        // --- DSL combat abilities (damage, CC, utility, defense, etc.) ---
        // Use the squad AI's scoring function which handles all ai_hint types,
        // threat-reduction scoring, conditional combos, AoE, etc.
        if let Some(tid) = team_target {
            use bevy_game::ai::squad::combat_evaluate_hero_ability;
            if let Some(action) = combat_evaluate_hero_ability(sim, uid, tid) {
                return Some(action);
            }
        }
    }

    // ---------------------------------------------------------------
    // Legacy flat-field abilities (non-template units only).
    // These are kept as fallback for units that don't use the DSL system.
    // ---------------------------------------------------------------
    if !has_dsl_abilities {
        // Heal allies below 50%
        if unit.heal_amount > 0 && unit.heal_cooldown_remaining_ms == 0 {
            let hurt_ally = sim.units.iter()
                .filter(|u| u.team == Team::Hero && is_alive(u) && u.id != uid)
                .filter(|u| (u.hp as f32 / u.max_hp.max(1) as f32) < 0.50)
                .min_by_key(|u| u.hp);
            if let Some(ally) = hurt_ally {
                if distance(unit.position, ally.position) <= unit.heal_range {
                    return Some(IntentAction::CastHeal { target_id: ally.id });
                }
            }
        }

        // CC the focus target
        if unit.control_duration_ms > 0 && unit.control_cooldown_remaining_ms == 0 {
            if let Some(tid) = team_target {
                if let Some(t) = sim.units.iter().find(|u| u.id == tid) {
                    if distance(unit.position, t.position) <= unit.control_range && t.control_remaining_ms == 0 {
                        return Some(IntentAction::CastControl { target_id: tid });
                    }
                }
            }
        }

        // Ability damage on focus target
        if unit.ability_damage > 0 && unit.ability_cooldown_remaining_ms == 0 {
            if let Some(tid) = team_target {
                if let Some(t) = sim.units.iter().find(|u| u.id == tid) {
                    if distance(unit.position, t.position) <= unit.ability_range {
                        return Some(IntentAction::CastAbility { target_id: tid });
                    }
                }
            }
        }
    }

    // Focus-fire attack on team target (or nearest enemy)
    let target_id = team_target.or_else(|| {
        sim.units.iter()
            .filter(|u| u.team != unit.team && is_alive(u))
            .min_by(|a, b| {
                hero_target_priority(a, unit)
                    .partial_cmp(&hero_target_priority(b, unit))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|u| u.id)
    });

    if let Some(tid) = target_id {
        if let Some(target) = sim.units.iter().find(|u| u.id == tid) {
            let dist = distance(unit.position, target.position);

            // Low HP kiting
            if hp_pct < 0.25 && dist < 2.0 {
                let nearest_enemy = sim.units.iter()
                    .filter(|u| u.team != unit.team && is_alive(u))
                    .min_by(|a, b| {
                        distance(unit.position, a.position)
                            .partial_cmp(&distance(unit.position, b.position))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                if let Some(ne) = nearest_enemy {
                    let away = move_away(unit.position, ne.position, unit.move_speed_per_sec * 0.1);
                    return Some(IntentAction::MoveTo { position: away });
                }
            }

            return Some(IntentAction::Attack { target_id: tid });
        }
    }

    None
}

/// Compute the coordinated team target for all heroes.
pub(crate) fn compute_team_target(sim: &bevy_game::ai::core::SimState) -> Option<u32> {
    use bevy_game::ai::core::{is_alive, Team, SimVec2};

    let living_heroes: Vec<_> = sim.units.iter()
        .filter(|u| u.team == Team::Hero && is_alive(u))
        .collect();
    if living_heroes.is_empty() {
        return None;
    }

    let cx = living_heroes.iter().map(|u| u.position.x).sum::<f32>() / living_heroes.len() as f32;
    let cy = living_heroes.iter().map(|u| u.position.y).sum::<f32>() / living_heroes.len() as f32;
    let dummy_hero = bevy_game::ai::core::UnitState {
        position: SimVec2 { x: cx, y: cy },
        ..living_heroes[0].clone()
    };

    sim.units.iter()
        .filter(|u| u.team == Team::Enemy && is_alive(u))
        .min_by(|a, b| {
            hero_target_priority(a, &dummy_hero)
                .partial_cmp(&hero_target_priority(b, &dummy_hero))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|u| u.id)
}

