//! Main generation function and core strategy implementations (1-4), helpers.

use super::super::types::{ScenarioAssert, ScenarioCfg, ScenarioFile};
use super::coverage::CoverageTracker;
use super::metadata::{
    heroes_by_role, Role, ALL_HEROES, ROOM_TYPES, DedupSet, Lcg,
};
use super::strategies_advanced;
use super::GenConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Estimated blocked percentage by room type, used for terrain-aware
/// difficulty scaling.  Small, obstacle-dense rooms get fewer enemies
/// so units aren't stacked on top of each other.
fn room_blocked_estimate(room: &str) -> f32 {
    match room {
        "Pressure" => 0.20,
        "Pivot" => 0.15,
        "Setpiece" => 0.12,
        "Climax" => 0.12,
        "Entry" => 0.10,
        "Recovery" => 0.08,
        _ => 0.0, // Open / unknown
    }
}

pub(super) fn build_scenario(
    name: String,
    heroes: Vec<String>,
    enemy_count: usize,
    difficulty: u32,
    hp_mult: f32,
    room_type: &str,
    seed: u64,
    max_ticks: u64,
) -> ScenarioFile {
    // Terrain-aware difficulty scaling: reduce enemy count for high-blocked rooms
    let blocked_pct = room_blocked_estimate(room_type);
    let scaled_enemy_count = (enemy_count as f32 * (1.0 - blocked_pct * 0.3)).round() as usize;
    let final_enemy_count = scaled_enemy_count.max(1);

    ScenarioFile {
        scenario: ScenarioCfg {
            name,
            seed,
            hero_count: heroes.len(),
            enemy_count: final_enemy_count,
            difficulty,
            max_ticks,
            room_type: room_type.to_string(),
            hero_templates: heroes,
            enemy_hero_templates: Vec::new(),
            hp_multiplier: hp_mult,
            manifest_path: None,
            drill_type: None,
            target_position: None,
            enemy_units: Vec::new(),
            hazards: Vec::new(),
            objective: None,
            action_mask: None,
        },
        assert: Some(ScenarioAssert {
            outcome: Some("Any".to_string()),
            max_ticks_to_win: None,
            min_heroes_alive: None,
            max_heroes_dead: None,
        }),
    }
}

pub(super) fn build_hvh_scenario(
    name: String,
    heroes: Vec<String>,
    enemy_heroes: Vec<String>,
    hp_mult: f32,
    room_type: &str,
    seed: u64,
    max_ticks: u64,
) -> ScenarioFile {
    ScenarioFile {
        scenario: ScenarioCfg {
            name,
            seed,
            hero_count: heroes.len(),
            enemy_count: enemy_heroes.len(),
            difficulty: 1,
            max_ticks,
            room_type: room_type.to_string(),
            hero_templates: heroes,
            enemy_hero_templates: enemy_heroes,
            hp_multiplier: hp_mult,
            manifest_path: None,
            drill_type: None,
            target_position: None,
            enemy_units: Vec::new(),
            hazards: Vec::new(),
            objective: None,
            action_mask: None,
        },
        assert: Some(ScenarioAssert {
            outcome: Some("Any".to_string()),
            max_ticks_to_win: None,
            min_heroes_alive: None,
            max_heroes_dead: None,
        }),
    }
}

pub(super) fn to_toml(file: &ScenarioFile) -> String {
    let cfg = &file.scenario;
    let heroes_str = cfg.hero_templates.iter()
        .map(|h| format!("\"{}\"", h))
        .collect::<Vec<_>>()
        .join(", ");

    let mut s = format!(
        "[scenario]\n\
         name = \"{}\"\n\
         seed = {}\n\
         hero_count = {}\n\
         enemy_count = {}\n\
         difficulty = {}\n\
         max_ticks = {}\n\
         room_type = \"{}\"\n\
         hero_templates = [{}]\n",
        cfg.name, cfg.seed, cfg.hero_count, cfg.enemy_count,
        cfg.difficulty, cfg.max_ticks, cfg.room_type, heroes_str,
    );

    if !cfg.enemy_hero_templates.is_empty() {
        let enemy_str = cfg.enemy_hero_templates.iter()
            .map(|h| format!("\"{}\"", h))
            .collect::<Vec<_>>()
            .join(", ");
        s.push_str(&format!("enemy_hero_templates = [{}]\n", enemy_str));
    }

    s.push_str(&format!("hp_multiplier = {:.1}\n", cfg.hp_multiplier));

    if let Some(ref assert) = file.assert {
        s.push_str("\n[assert]\n");
        if let Some(ref outcome) = assert.outcome {
            s.push_str(&format!("outcome = \"{}\"\n", outcome));
        }
    }
    s
}

/// Room-type affinity: which roles benefit from room geometry.
fn room_role_affinity(room: &str) -> &'static [Role] {
    match room {
        "Pressure" => &[Role::RangedDps, Role::Tank, Role::Healer],
        "Climax" => &[Role::Tank, Role::Healer, Role::MeleeDps],
        "Pivot" => &[Role::MeleeDps, Role::RangedDps, Role::Hybrid],
        "Setpiece" => &[Role::RangedDps, Role::Hybrid, Role::Tank],
        "Recovery" => &[Role::Healer, Role::Hybrid, Role::Tank],
        _ => &[Role::Tank, Role::Healer, Role::MeleeDps, Role::RangedDps],
    }
}

// ---------------------------------------------------------------------------
// Emit helper — dedup + optional seed variants
// ---------------------------------------------------------------------------

pub(super) fn emit(
    scenario: ScenarioFile,
    seed_variants: u32,
    out: &mut Vec<ScenarioFile>,
    dedup: &mut DedupSet,
    coverage: &mut CoverageTracker,
) {
    if !dedup.insert(&scenario.scenario) {
        return;
    }
    coverage.record(&scenario.scenario);
    out.push(scenario.clone());

    // Extra seed variants — same composition, different RNG seed
    for v in 1..seed_variants {
        let mut variant = scenario.clone();
        variant.scenario.seed += v as u64 * 10_000;
        variant.scenario.name = format!("{}_s{}", variant.scenario.name, v);
        coverage.record(&variant.scenario);
        out.push(variant);
    }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

pub fn generate(config: &GenConfig) -> Vec<ScenarioFile> {
    let mut rng = Lcg::new(config.seed);
    let mut out: Vec<ScenarioFile> = Vec::new();
    let mut dedup = DedupSet::new();
    let mut coverage = CoverageTracker::default();
    let mut seed_counter: u64 = 1000;
    let mut next_seed = || { seed_counter += 1; seed_counter };

    let hero_names: Vec<&str> = ALL_HEROES.iter().map(|h| h.name).collect();
    let sv = config.seed_variants;

    // -----------------------------------------------------------------------
    // 1. Synergy pairs
    // -----------------------------------------------------------------------
    if config.include_synergy_pairs {
        for i in 0..hero_names.len() {
            for j in (i + 1)..hero_names.len() {
                let (a, b) = (hero_names[i], hero_names[j]);

                let room = *rng.choose(ROOM_TYPES);
                let diff = (rng.next_usize(2) + 1) as u32;
                let seed = next_seed();
                let name = format!("pair_{a}_{b}");
                let s = build_scenario(
                    name, vec![a.to_string(), b.to_string()],
                    3, diff, 1.0, room, seed, 8_000,
                );
                emit(s, sv, &mut out, &mut dedup, &mut coverage);

                let filler = loop {
                    let h = rng.choose(&hero_names);
                    if *h != a && *h != b { break *h; }
                };
                let room2 = *rng.choose(ROOM_TYPES);
                let seed2 = next_seed();
                let name2 = format!("trio_{a}_{b}_{filler}");
                let s2 = build_scenario(
                    name2, vec![a.to_string(), b.to_string(), filler.to_string()],
                    4, diff, 1.0, room2, seed2, 8_000,
                );
                emit(s2, sv, &mut out, &mut dedup, &mut coverage);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 2. Stress archetypes x all rooms
    // -----------------------------------------------------------------------
    if config.include_stress_archetypes {
        let archetypes: &[(&str, &[Role], usize, u32)] = &[
            ("all_tank",       &[Role::Tank, Role::Tank, Role::Tank, Role::Tank], 6, 2),
            ("all_healer",     &[Role::Healer, Role::Healer, Role::Healer, Role::Healer], 5, 2),
            ("all_melee",      &[Role::MeleeDps, Role::MeleeDps, Role::MeleeDps, Role::MeleeDps, Role::MeleeDps], 6, 2),
            ("all_ranged",     &[Role::RangedDps, Role::RangedDps, Role::RangedDps, Role::RangedDps], 7, 2),
            ("all_hybrid",     &[Role::Hybrid, Role::Hybrid, Role::Hybrid, Role::Hybrid], 5, 2),
            ("no_tank",        &[Role::Healer, Role::MeleeDps, Role::RangedDps, Role::Hybrid], 5, 3),
            ("no_healer",      &[Role::Tank, Role::MeleeDps, Role::RangedDps, Role::MeleeDps], 5, 2),
            ("glass_cannon",   &[Role::RangedDps, Role::RangedDps, Role::MeleeDps, Role::MeleeDps], 3, 3),
            ("double_front",   &[Role::Tank, Role::Tank, Role::Healer, Role::Healer], 7, 2),
            ("solo_tank",      &[Role::Tank], 4, 1),
            ("solo_healer",    &[Role::Healer], 3, 1),
            ("duo_heal_swarm", &[Role::Healer, Role::Healer], 5, 1),
            ("full_8",         &[Role::Tank, Role::Tank, Role::Healer, Role::Healer,
                                 Role::MeleeDps, Role::MeleeDps, Role::RangedDps, Role::RangedDps], 8, 3),
        ];

        for (tag, roles, ec, diff) in archetypes {
            for room in ROOM_TYPES {
                let party: Vec<String> = roles.iter().map(|r| {
                    let pool = heroes_by_role(*r);
                    rng.choose(&pool).to_string()
                }).collect();
                let hp = if *ec > party.len() + 2 { 2.0 } else { 1.0 };
                let seed = next_seed();
                let name = format!("stress_{tag}_{}", room.to_lowercase());
                let s = build_scenario(name, party, *ec, *diff, hp, room, seed, 10_000);
                emit(s, sv, &mut out, &mut dedup, &mut coverage);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. Difficulty ladders
    // -----------------------------------------------------------------------
    if config.include_difficulty_ladders {
        let ladder_parties: &[&[&str]] = &[
            &["warrior", "cleric", "mage", "rogue"],
            &["paladin", "druid", "ranger", "assassin"],
            &["knight", "bard", "pyromancer", "berserker"],
            &["templar", "shaman", "engineer", "monk"],
            &["warden", "alchemist", "elementalist", "samurai"],
            &["warrior", "warrior", "cleric", "druid", "mage", "ranger"],
            &["blood_mage", "necromancer", "warlock", "witch_doctor"],
            &["shadow_dancer", "monk", "cryomancer", "arcanist"],
        ];

        for party_tmpl in ladder_parties {
            let party: Vec<String> = party_tmpl.iter().map(|s| s.to_string()).collect();
            let ec = party.len() + 1;
            for diff in 1..=5u32 {
                for hp in &[1.0f32, 3.0] {
                    let room = *rng.choose(ROOM_TYPES);
                    let seed = next_seed();
                    let name = format!("ladder_{}_d{diff}_hp{hp:.0}x", party_tmpl[0]);
                    let s = build_scenario(name, party.clone(), ec, diff, *hp, room, seed, 10_000);
                    emit(s, sv, &mut out, &mut dedup, &mut coverage);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 4. Room-aware compositions
    // -----------------------------------------------------------------------
    if config.include_room_aware {
        for room in ROOM_TYPES {
            let preferred = room_role_affinity(room);

            for _ in 0..3 {
                let party_size = rng.next_usize(2) + 3;
                let mut party = Vec::new();
                for _ in 0..party_size {
                    let role = *rng.choose(preferred);
                    let pool = heroes_by_role(role);
                    party.push(rng.choose(&pool).to_string());
                }
                let ec = party.len() + rng.next_usize(3);
                let diff = (rng.next_usize(3) + 1) as u32;
                let seed = next_seed();
                let name = format!("room_fit_{}_{}", room.to_lowercase(), out.len());
                let s = build_scenario(name, party, ec, diff, 1.0, room, seed, 10_000);
                emit(s, sv, &mut out, &mut dedup, &mut coverage);
            }

            let all_roles = [Role::Tank, Role::Healer, Role::MeleeDps, Role::RangedDps, Role::Hybrid];
            let mismatch: Vec<Role> = all_roles.iter()
                .filter(|r| !preferred.contains(r))
                .copied()
                .collect();
            if !mismatch.is_empty() {
                let mut party = Vec::new();
                for _ in 0..4 {
                    let role = *rng.choose(&mismatch);
                    let pool = heroes_by_role(role);
                    party.push(rng.choose(&pool).to_string());
                }
                let seed = next_seed();
                let name = format!("room_mis_{}", room.to_lowercase());
                let s = build_scenario(name, party, 5, 2, 1.0, room, seed, 10_000);
                emit(s, sv, &mut out, &mut dedup, &mut coverage);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 5-7. Delegated to strategies_advanced
    // -----------------------------------------------------------------------
    if config.include_size_spectrum {
        strategies_advanced::generate_size_spectrum(
            config, &mut rng, &mut next_seed, &hero_names,
            &mut out, &mut dedup, &mut coverage,
        );
    }

    if config.include_hero_vs_hero {
        strategies_advanced::generate_hero_vs_hero(
            config, &mut rng, &mut next_seed,
            &mut out, &mut dedup, &mut coverage,
        );
    }

    strategies_advanced::generate_random_fill(
        config, &mut rng, &mut next_seed, &hero_names,
        &mut out, &mut dedup, &mut coverage,
    );

    if config.verbose {
        coverage.print_summary();

        let pair_count = out.iter().filter(|s| {
            let n = &s.scenario.name;
            n.starts_with("pair_") || n.starts_with("trio_")
        }).count();
        let stress_count = out.iter().filter(|s| s.scenario.name.starts_with("stress_")).count();
        let ladder_count = out.iter().filter(|s| s.scenario.name.starts_with("ladder_")).count();
        let room_count = out.iter().filter(|s| s.scenario.name.starts_with("room_")).count();
        let size_count = out.iter().filter(|s| s.scenario.name.starts_with("size_")).count();
        let hvh_count = out.iter().filter(|s| s.scenario.name.starts_with("hvh_")).count();
        let rand_count = out.iter().filter(|s| s.scenario.name.starts_with("rand_")).count();

        println!("\nStrategy breakdown:");
        println!("  synergy pairs:  {pair_count}");
        println!("  stress:         {stress_count}");
        println!("  ladders:        {ladder_count}");
        println!("  room-aware:     {room_count}");
        println!("  size spectrum:  {size_count}");
        println!("  hero vs hero:   {hvh_count}");
        println!("  random fill:    {rand_count}");
        println!("  TOTAL:          {}", out.len());
    }

    out
}

// Re-export write_scenarios from the advanced module
pub use strategies_advanced::write_scenarios;
