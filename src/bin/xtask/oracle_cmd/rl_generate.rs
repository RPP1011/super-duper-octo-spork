//! Episode generation CLI for transformer RL.

use std::io::Write;
use std::process::ExitCode;

use rayon::prelude::*;

use super::collect_toml_paths;
use super::transformer_rl::{Policy, RlEpisode, MAX_ABILITIES};
use super::rl_policies::run_single_episode;
use super::rl_gpu::run_gpu_multiplexed;

pub(crate) fn run_generate(args: crate::cli::TransformerRlGenerateArgs) -> ExitCode {
    use bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer;
    use bevy_game::scenario::load_scenario_file;

    // Load ability eval weights for Combined policy (also used to configure squad AI)
    let ability_eval_weights = if let Some(ref path) = args.ability_eval {
        let json_str = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => { eprintln!("Failed to read ability eval weights: {e}"); return ExitCode::from(1); }
        };
        let json_val: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => { eprintln!("Failed to parse ability eval JSON: {e}"); return ExitCode::from(1); }
        };
        Some(std::sync::Arc::new(
            bevy_game::ai::core::ability_eval::AbilityEvalWeights::from_json(&json_val)
        ))
    } else {
        None
    };

    // Load combat student model for Combined policy
    let student_weights = if let Some(ref path) = args.student_model {
        let json_str = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => { eprintln!("Failed to read student model: {e}"); return ExitCode::from(1); }
        };
        let json_val: serde_json::Value = match serde_json::from_str(&json_str) {
            Ok(v) => v,
            Err(e) => { eprintln!("Failed to parse student model JSON: {e}"); return ExitCode::from(1); }
        };
        Some(std::sync::Arc::new(super::training::StudentWeights::from_json(&json_val)))
    } else {
        None
    };

    let policy = if args.random_policy {
        Policy::Random
    } else if let Some(ref shm_path) = args.gpu_shm {
        use bevy_game::ai::core::ability_transformer::gpu_client::GpuInferenceClient;
        match GpuInferenceClient::new(shm_path, 1024, 1) {
            Ok(client) => Policy::GpuServer(client),
            Err(e) => { eprintln!("Failed to open GPU SHM at {shm_path}: {e}"); return ExitCode::from(1); }
        }
    } else if args.policy == "combined" {
        Policy::Combined
    } else {
        let weights_path = match &args.weights {
            Some(p) => p,
            None => { eprintln!("--weights is required for transformer policy"); return ExitCode::from(1); }
        };
        match Policy::load(weights_path) {
            Ok(p) => p,
            Err(e) => { eprintln!("Failed to load weights: {e}"); return ExitCode::from(1); }
        }
    };
    let policy_type = match &policy {
        Policy::ActorCritic(_) => "actor-critic",
        Policy::ActorCriticV2(_) => "actor-critic-v2",
        Policy::ActorCriticV3(_) => "actor-critic-v3 (pointer)",
        Policy::ActorCriticV4(_) => "actor-critic-v4 (dual-head)",
        Policy::ActorCriticV5(_) => "actor-critic-v5 (d=128, aggregate)",
        Policy::GpuServer(_) => "gpu-server (dual-head)",
        Policy::Legacy(_) => "legacy (bootstrap)",
        Policy::Combined => "combined (ability-eval + squad AI)",
        Policy::Random => "random",
    };
    let is_v3 = matches!(&policy, Policy::ActorCriticV3(_) | Policy::ActorCriticV4(_) | Policy::ActorCriticV5(_) | Policy::GpuServer(_));

    let tokenizer = AbilityTokenizer::new();

    let paths: Vec<_> = args.path.iter().flat_map(|p| collect_toml_paths(p)).collect();
    if paths.is_empty() {
        eprintln!("No *.toml files found.");
        return ExitCode::from(1);
    }

    eprintln!("Generating RL episodes: {} scenarios x {} episodes, temp={:.2}, policy={}",
        paths.len(), args.episodes, args.temperature, policy_type);

    let scenarios: Vec<_> = paths.iter().filter_map(|p| {
        match load_scenario_file(p) {
            Ok(f) => Some(f),
            Err(e) => { eprintln!("{e}"); None }
        }
    }).collect();

    let threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();

    // Load embedding registry if provided
    let registry = if let Some(ref reg_path) = args.embedding_registry {
        match bevy_game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded embedding registry: {} abilities, hash={}",
                    r.len(), r.model_hash);
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    // Load enemy policy for self-play
    let enemy_policy: Option<Policy> = if let Some(ref ew_path) = args.enemy_weights {
        match Policy::load(ew_path) {
            Ok(p) => {
                eprintln!("Loaded enemy policy from {}", ew_path.display());
                Some(p)
            }
            Err(e) => { eprintln!("Failed to load enemy policy: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };
    let enemy_registry = if let Some(ref reg_path) = args.enemy_registry {
        match bevy_game::ai::core::ability_transformer::EmbeddingRegistry::from_file(
            reg_path.to_str().unwrap_or(""),
        ) {
            Ok(r) => {
                eprintln!("Loaded enemy embedding registry: {} abilities", r.len());
                Some(r)
            }
            Err(e) => { eprintln!("Failed to load enemy registry: {e}"); return ExitCode::from(1); }
        }
    } else {
        None
    };

    let policy_ref = &policy;
    let tokenizer_ref = &tokenizer;
    let ability_eval_ref = &ability_eval_weights;
    let student_ref = &student_weights;
    let registry_ref = registry.as_ref();
    let enemy_policy_ref = enemy_policy.as_ref();
    let enemy_registry_ref = enemy_registry.as_ref();
    let step_interval = args.step_interval;
    let temperature = args.temperature;
    let max_ticks_override = args.max_ticks;

    // --swap-sides: duplicate scenarios with hero/enemy templates swapped
    let mut scenarios = scenarios;
    if args.swap_sides {
        let n = scenarios.len();
        let mut swapped = Vec::with_capacity(n);
        for sf in &scenarios {
            let mut cfg = sf.scenario.clone();
            std::mem::swap(&mut cfg.hero_templates, &mut cfg.enemy_hero_templates);
            cfg.name = format!("{}_swapped", cfg.name);
            // Also swap counts if they differ
            std::mem::swap(&mut cfg.hero_count, &mut cfg.enemy_count);
            swapped.push(bevy_game::scenario::ScenarioFile { scenario: cfg, assert: None });
        }
        scenarios.extend(swapped);
        eprintln!("Swap-sides: {} original + {} swapped = {} total scenarios",
            n, n, scenarios.len());
    }

    let episode_tasks: Vec<(usize, usize)> = scenarios.iter().enumerate()
        .flat_map(|(si, _)| (0..args.episodes as usize).map(move |ei| (si, ei)))
        .collect();

    // Use multiplexed GPU path when sims_per_thread > 1 and policy is GpuServer
    let episodes: Vec<RlEpisode> = if args.sims_per_thread > 1 {
        if let Policy::GpuServer(ref gpu) = policy {
            eprintln!("GPU multiplexed: {} threads x {} sims/thread", threads, args.sims_per_thread);
            run_gpu_multiplexed(
                gpu, &scenarios, &episode_tasks,
                threads, args.sims_per_thread,
                tokenizer_ref, temperature, step_interval,
                max_ticks_override, registry_ref,
                args.self_play_gpu,
            )
        } else {
            eprintln!("Warning: --sims-per-thread > 1 only works with --gpu-shm, falling back to par_iter");
            pool.install(|| {
                episode_tasks.par_iter().map(|&(si, ei)| {
                    run_single_episode(
                        &scenarios[si], si, ei, max_ticks_override, is_v3,
                        policy_ref, tokenizer_ref, temperature, step_interval,
                        student_ref, ability_eval_ref, registry_ref,
                        enemy_policy_ref, enemy_registry_ref,
                    )
                }).collect()
            })
        }
    } else {
        pool.install(|| {
            episode_tasks.par_iter().map(|&(si, ei)| {
                run_single_episode(
                    &scenarios[si], si, ei, max_ticks_override, is_v3,
                    policy_ref, tokenizer_ref, temperature, step_interval,
                    student_ref, ability_eval_ref, registry_ref,
                    enemy_policy_ref, enemy_registry_ref,
                )
            }).collect()
        })
    };

    let wins = episodes.iter().filter(|e| e.outcome == "Victory").count();
    let losses = episodes.iter().filter(|e| e.outcome == "Defeat").count();
    let timeouts = episodes.iter().filter(|e| e.outcome == "Timeout").count();
    let total_steps: usize = episodes.iter().map(|e| e.steps.len()).sum();
    let mean_reward: f32 = episodes.iter().map(|e| e.reward).sum::<f32>() / episodes.len().max(1) as f32;

    eprintln!("Episodes: {}  Wins: {}  Losses: {}  Timeouts: {}  Win rate: {:.1}%",
        episodes.len(), wins, losses, timeouts,
        wins as f64 / episodes.len().max(1) as f64 * 100.0);
    eprintln!("Total steps: {}  Mean reward: {:.3}", total_steps, mean_reward);

    let file = std::fs::File::create(&args.output).unwrap();
    let mut writer = std::io::BufWriter::new(file);
    for ep in &episodes {
        let line = serde_json::to_string(ep).unwrap();
        writeln!(writer, "{}", line).unwrap();
    }
    writer.flush().unwrap();
    eprintln!("Wrote {} episodes to {}", episodes.len(), args.output.display());

    ExitCode::SUCCESS
}
