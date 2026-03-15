//! Behavioral ability profiling — controlled sim experiments for embedding generation.
//!
//! For each ability, spawns minimal sims (1 caster + N targets) under varied conditions,
//! casts the ability, and records clean before/after outcome deltas.
//! Outputs `dataset/ability_profiles.npz` for training behavioral embeddings.

use std::process::ExitCode;
use std::sync::Mutex;

use bevy_game::ai::core::{
    sim_vec2, step, Team, UnitIntent, IntentAction, FIXED_TICK_MS,
};
use bevy_game::ai::effects::{
    AbilityDef, AbilityTarget, AbilityTargeting,
};
use rayon::prelude::*;

use super::ability_profile_io::*;

/// Determine the correct AbilityTarget based on the ability's targeting type.
fn make_target(targeting: &AbilityTargeting, first_target_id: u32, first_target_pos: bevy_game::ai::core::SimVec2) -> AbilityTarget {
    match targeting {
        AbilityTargeting::TargetEnemy => AbilityTarget::Unit(first_target_id),
        AbilityTargeting::TargetAlly => AbilityTarget::Unit(first_target_id),
        AbilityTargeting::SelfCast => AbilityTarget::None,
        AbilityTargeting::SelfAoe => AbilityTarget::Position(first_target_pos),
        AbilityTargeting::GroundTarget | AbilityTargeting::Direction | AbilityTargeting::Vector => {
            AbilityTarget::Position(first_target_pos)
        }
        AbilityTargeting::Global => AbilityTarget::None,
    }
}

/// Run a single trial: set up sim, cast ability, tick until resolved, record outcomes.
fn run_trial(ability: &AbilityDef, cond: &TrialCondition) -> TrialOutcome {
    let target_hp = (cond.target_hp_pct * MAX_HP as f32).max(1.0) as i32;
    let is_ally = matches!(ability.targeting, AbilityTargeting::TargetAlly);
    let target_team = if is_ally { Team::Hero } else { Team::Enemy };

    let mut units = vec![caster_unit(ability.clone())];
    units[0].attack_cooldown_ms = 999999;

    let first_target_id = 1u32;
    for i in 0..cond.n_targets {
        let angle = (i as f32) * std::f32::consts::TAU / (cond.n_targets as f32);
        let spread = 0.5;
        let x = cond.distance + angle.cos() * spread * (i as f32).min(1.0);
        let y = angle.sin() * spread * (i as f32).min(1.0);
        units.push(target_unit(
            first_target_id + i as u32,
            target_team,
            sim_vec2(x, y),
            target_hp,
            cond.armor,
        ));
    }

    let pre_hp: Vec<i32> = units[1..].iter().map(|u| u.hp).collect();
    let pre_shield: Vec<i32> = units[1..].iter().map(|u| u.shield_hp).collect();
    let pre_pos: Vec<(f32, f32)> = units[1..].iter().map(|u| (u.position.x, u.position.y)).collect();
    let caster_pre_hp = units[0].hp;
    let caster_pre_shield = units[0].shield_hp;
    let caster_pre_pos = (units[0].position.x, units[0].position.y);

    let first_target_pos = units[1].position;
    let mut sim = make_sim(units);

    let mut peak_outcomes: Vec<TargetOutcome> = (0..cond.n_targets).map(|_| TargetOutcome::default()).collect();
    let mut caster_outcome = TargetOutcome::default();

    let target = make_target(&ability.targeting, first_target_id, first_target_pos);
    let intents = vec![UnitIntent {
        unit_id: CASTER_ID,
        action: IntentAction::UseAbility { ability_index: 0, target },
    }];

    let (new_sim, _events) = step(sim, &intents, FIXED_TICK_MS);
    sim = new_sim;

    let max_resolve_ticks = (ability.cast_time_ms / FIXED_TICK_MS).max(1) + 100;
    for _ in 0..max_resolve_ticks {
        for i in 0..cond.n_targets {
            let tid = first_target_id + i as u32;
            if let Some(unit) = sim.units.iter().find(|u| u.id == tid) {
                scan_status_effects(unit, &mut peak_outcomes[i]);
            }
        }
        if let Some(caster) = sim.units.iter().find(|u| u.id == CASTER_ID) {
            scan_status_effects(caster, &mut caster_outcome);
        }
        let (new_sim, _events) = step(sim, &[], FIXED_TICK_MS);
        sim = new_sim;
    }

    // Final sample
    for i in 0..cond.n_targets {
        let tid = first_target_id + i as u32;
        if let Some(unit) = sim.units.iter().find(|u| u.id == tid) {
            scan_status_effects(unit, &mut peak_outcomes[i]);
        }
    }
    if let Some(caster) = sim.units.iter().find(|u| u.id == CASTER_ID) {
        scan_status_effects(caster, &mut caster_outcome);
    }

    // Record post-state and compute deltas
    let mut per_target = Vec::new();
    let mut total_damage = 0.0f32;
    let mut total_heal = 0.0f32;
    let mut n_hit = 0u32;
    let mut n_killed = 0u32;

    for i in 0..cond.n_targets {
        let target_id = first_target_id + i as u32;
        let mut outcome = std::mem::take(&mut peak_outcomes[i]);

        if let Some(unit) = sim.units.iter().find(|u| u.id == target_id) {
            outcome.delta_hp = unit.hp as f32 - pre_hp[i] as f32;
            outcome.delta_shield = unit.shield_hp as f32 - pre_shield[i] as f32;
            outcome.delta_x = unit.position.x - pre_pos[i].0;
            outcome.delta_y = unit.position.y - pre_pos[i].1;
            outcome.killed = unit.hp <= 0;

            let has_effect = outcome.delta_hp.abs() > 0.01
                || outcome.delta_shield.abs() > 0.01
                || outcome.delta_x.abs() > 0.01
                || outcome.delta_y.abs() > 0.01
                || outcome.has_any_status();
            if has_effect { n_hit += 1; }
            if outcome.delta_hp < 0.0 { total_damage += -outcome.delta_hp; }
            if outcome.delta_hp > 0.0 { total_heal += outcome.delta_hp; }
            if outcome.killed { n_killed += 1; }
        } else {
            outcome.delta_hp = -(pre_hp[i] as f32);
            outcome.killed = true;
            total_damage += -outcome.delta_hp;
            n_hit += 1;
            n_killed += 1;
        }
        per_target.push(outcome);
    }

    // Fill caster deltas
    if let Some(caster) = sim.units.iter().find(|u| u.id == CASTER_ID) {
        caster_outcome.delta_hp = caster.hp as f32 - caster_pre_hp as f32;
        caster_outcome.delta_shield = caster.shield_hp as f32 - caster_pre_shield as f32;
        caster_outcome.delta_x = caster.position.x - caster_pre_pos.0;
        caster_outcome.delta_y = caster.position.y - caster_pre_pos.1;
    }

    TrialOutcome { total_damage, total_heal, n_targets_hit: n_hit, n_targets_killed: n_killed, per_target, caster: caster_outcome }
}

pub fn run_ability_profile(args: crate::cli::AbilityProfileArgs) -> ExitCode {
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let abilities = load_all_abilities();
    if abilities.is_empty() {
        eprintln!("No abilities found!");
        return ExitCode::from(1);
    }

    // Build the full condition grid
    let mut conditions = Vec::new();
    for &hp_pct in HP_PCTS {
        for &distance in DISTANCES {
            for &n_targets in TARGET_COUNTS {
                for &armor in ARMORS {
                    conditions.push(TrialCondition { target_hp_pct: hp_pct, distance, n_targets, armor });
                }
            }
        }
    }
    let n_conditions = conditions.len();
    eprintln!("Condition grid: {} combinations ({}x{}x{}x{} = hp x dist x targets x armor)",
        n_conditions, HP_PCTS.len(), DISTANCES.len(), TARGET_COUNTS.len(), ARMORS.len());

    let n_trials_per_ability = n_conditions.min(args.samples_per_ability);

    eprintln!("Profiling {} abilities x {} trials = {} total trials",
        abilities.len(), n_trials_per_ability, abilities.len() * n_trials_per_ability);

    let all_samples: Mutex<Vec<ProfileSample>> = Mutex::new(Vec::new());
    let done = std::sync::atomic::AtomicUsize::new(0);
    let n_abilities = abilities.len();

    abilities.par_iter().enumerate().for_each(|(abl_idx, (name, def, _dsl_text))| {
        let mut local_samples = Vec::new();

        for cond in conditions.iter().take(n_trials_per_ability) {
            let outcome = run_trial(def, cond);
            local_samples.push(ProfileSample {
                ability_idx: abl_idx as u32,
                ability_name: name.clone(),
                condition: condition_to_vec(cond),
                outcome: outcome_to_vec(&outcome),
            });
        }

        all_samples.lock().unwrap().extend(local_samples);

        let completed = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if completed % 20 == 0 || completed == n_abilities {
            eprintln!("  [{completed}/{n_abilities}] {name}");
        }
    });

    let samples = all_samples.into_inner().unwrap();
    let n = samples.len();
    eprintln!("\nTotal: {n} profile samples from {n_abilities} abilities");

    if n == 0 {
        eprintln!("No samples generated.");
        return ExitCode::from(1);
    }

    // Print summary stats
    let agg_offset = MAX_TARGETS * PER_TARGET_DIM + PER_TARGET_DIM;
    let mut damage_count = 0u32;
    let mut heal_count = 0u32;
    let mut cc_count = 0u32;
    let mut status_count = 0u32;
    let mut displacement_count = 0u32;
    let mut kill_count = 0u32;
    let mut zero_effect_count = 0u32;

    for s in &samples {
        let total_dmg = s.outcome[agg_offset];
        let total_heal = s.outcome[agg_offset + 1];
        let n_killed = s.outcome[agg_offset + 3];

        let mut any_cc = false;
        let mut any_status = false;
        let mut any_disp = false;
        for i in 0..MAX_TARGETS {
            let base = i * PER_TARGET_DIM;
            if s.outcome[base + 2].abs() > 0.01 || s.outcome[base + 3].abs() > 0.01 {
                any_disp = true;
            }
            if s.outcome[base + 5] > 0.0 || s.outcome[base + 8] > 0.0
                || s.outcome[base + 10] > 0.0 || s.outcome[base + 11] > 0.0
                || s.outcome[base + 13] > 0.0 || s.outcome[base + 14] > 0.0
                || s.outcome[base + 16] > 0.0 {
                any_cc = true;
            }
            if (5..PER_TARGET_DIM).any(|j| s.outcome[base + j] > 0.0) {
                any_status = true;
            }
        }

        if total_dmg > 0.0 { damage_count += 1; }
        if total_heal > 0.0 { heal_count += 1; }
        if any_cc { cc_count += 1; }
        if any_status { status_count += 1; }
        if any_disp { displacement_count += 1; }
        if n_killed > 0.0 { kill_count += 1; }
        let caster_base = MAX_TARGETS * PER_TARGET_DIM;
        let caster_has_effect = (0..PER_TARGET_DIM).any(|j| s.outcome[caster_base + j].abs() > 0.01);

        if total_dmg == 0.0 && total_heal == 0.0 && !any_status && !any_disp && !caster_has_effect {
            zero_effect_count += 1;
        }
    }

    eprintln!("\nOutcome distribution:");
    eprintln!("  Damage:       {damage_count:>6} ({:.1}%)", damage_count as f32 / n as f32 * 100.0);
    eprintln!("  Heal:         {heal_count:>6} ({:.1}%)", heal_count as f32 / n as f32 * 100.0);
    eprintln!("  Hard CC:      {cc_count:>6} ({:.1}%)", cc_count as f32 / n as f32 * 100.0);
    eprintln!("  Any status:   {status_count:>6} ({:.1}%)", status_count as f32 / n as f32 * 100.0);
    eprintln!("  Displacement: {displacement_count:>6} ({:.1}%)", displacement_count as f32 / n as f32 * 100.0);
    eprintln!("  Kill:         {kill_count:>6} ({:.1}%)", kill_count as f32 / n as f32 * 100.0);
    eprintln!("  Zero-effect:  {zero_effect_count:>6} ({:.1}%)", zero_effect_count as f32 / n as f32 * 100.0);

    write_profile_npz(&args.output, &samples, &abilities);
    eprintln!("\nWritten to: {}", args.output.display());

    ExitCode::SUCCESS
}
