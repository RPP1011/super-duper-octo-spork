//! `xtask diagnose` — Run model inference with diagnostic capture and generate
//! an HTML report with attention heatmaps, embedding similarity, and action
//! distributions.

use std::path::PathBuf;
use std::process::ExitCode;

use bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer;
use bevy_game::ai::core::ability_transformer::ActorCriticWeightsV3;

pub fn run_diagnose(args: DiagnoseArgs) -> ExitCode {
    // Load model weights
    let weights_str = match std::fs::read_to_string(&args.weights) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read weights file {:?}: {e}", args.weights);
            return ExitCode::FAILURE;
        }
    };

    let model = match ActorCriticWeightsV3::from_json(&weights_str) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to parse weights: {e}");
            return ExitCode::FAILURE;
        }
    };

    let tokenizer = AbilityTokenizer::new();

    // Parse ability DSL texts
    let ability_texts: Vec<String> = if !args.ability.is_empty() {
        args.ability.clone()
    } else {
        // Default test ability
        vec![
            "ability Fireball { target: enemy range: 5.0 cooldown: 5s cast: 300ms hint: damage deliver projectile { speed: 8.0 } { on_hit { damage 55 } } }".to_string()
        ]
    };

    // Tokenize abilities
    let ability_token_ids: Vec<Vec<u32>> = ability_texts
        .iter()
        .map(|text| tokenizer.encode_with_cls(text))
        .collect();

    let ability_refs: Vec<Option<&[u32]>> = {
        let mut refs = Vec::new();
        for (i, ids) in ability_token_ids.iter().enumerate().take(8) {
            refs.push(Some(ids.as_slice()));
            eprintln!("Ability {i}: {} tokens — {:?}", ids.len(),
                &ability_texts.get(i).map(|s| if s.len() > 60 { &s[..60] } else { s }).unwrap_or(&""));
        }
        while refs.len() < 8 {
            refs.push(None);
        }
        refs
    };

    // Build synthetic game state entities (or load from scenario)
    let (entities, entity_types, threats, positions, entity_labels) = if let Some(ref scenario_path) = args.scenario_state {
        // Load from JSON file
        match load_scenario_state(scenario_path) {
            Ok(state) => state,
            Err(e) => {
                eprintln!("Failed to load scenario state: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        build_default_state()
    };

    let ent_refs: Vec<&[f32]> = entities.iter().map(|e| e.as_slice()).collect();
    let threat_refs: Vec<&[f32]> = threats.iter().map(|t| t.as_slice()).collect();
    let pos_refs: Vec<&[f32]> = positions.iter().map(|p| p.as_slice()).collect();

    eprintln!("Running diagnostic inference...");
    eprintln!("  Entities: {} (types: {:?})", entities.len(), entity_types);
    eprintln!("  Threats: {}, Positions: {}", threats.len(), positions.len());

    let capture = model.diagnose(
        &ent_refs,
        &entity_types,
        &threat_refs,
        &pos_refs,
        &ability_refs,
        entity_labels,
    );

    // Get token labels from vocabulary
    let vocab = bevy_game::ai::core::ability_transformer::tokenizer::VOCAB;
    let token_labels: Vec<&str> = vocab.to_vec();

    let html = capture.to_html(&token_labels);

    let out = &args.output;
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(out, &html) {
        Ok(()) => {
            eprintln!("Diagnostic report written to {}", out.display());
            eprintln!("  Size: {} KB", html.len() / 1024);
            eprintln!("  Transformer attention layers: {}", capture.transformer_attention.len());
            eprintln!("  Entity attention layers: {}", capture.entity_attention.len());
            eprintln!("  Cross-attention captures: {}", capture.cross_attention.len());
            eprintln!("  Value: {:.4}", capture.value);

            // Print action type probabilities
            let max_logit = capture.action_type_logits.iter()
                .cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = capture.action_type_logits.iter()
                .map(|&v| (v - max_logit).exp())
                .collect();
            let sum: f32 = exps.iter().sum();
            let action_names = ["Attack", "Move", "Hold",
                "Ab0", "Ab1", "Ab2", "Ab3", "Ab4", "Ab5", "Ab6", "Ab7"];
            eprintln!("  Action probabilities:");
            for (i, &e) in exps.iter().enumerate() {
                let prob = e / sum;
                if prob > 0.01 {
                    let name = action_names.get(i).unwrap_or(&"?");
                    eprintln!("    {name}: {:.1}%", prob * 100.0);
                }
            }

            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Failed to write report: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Build a default synthetic game state for testing.
fn build_default_state() -> (Vec<Vec<f32>>, Vec<usize>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<String>) {
    let mut labels = Vec::new();

    // Self entity
    let mut self_ent = vec![0.0f32; 30];
    self_ent[0] = 1.0;   // hp_pct
    self_ent[1] = 100.0;  // hp
    self_ent[2] = 100.0;  // max_hp
    self_ent[12] = 0.5;  // auto_dps
    self_ent[29] = 1.0;  // exists
    labels.push("Self".to_string());

    // Enemy 0: half health
    let mut enemy0 = vec![0.0f32; 30];
    enemy0[0] = 0.5;
    enemy0[1] = 50.0;
    enemy0[2] = 100.0;
    enemy0[3] = 3.0;    // rel_x
    enemy0[4] = 1.0;    // rel_y
    enemy0[5] = 3.2;    // distance
    enemy0[12] = 0.8;
    enemy0[29] = 1.0;
    labels.push("Enemy0 (50%)".to_string());

    // Enemy 1: full health, farther
    let mut enemy1 = vec![0.0f32; 30];
    enemy1[0] = 1.0;
    enemy1[1] = 120.0;
    enemy1[2] = 120.0;
    enemy1[3] = 6.0;
    enemy1[4] = -2.0;
    enemy1[5] = 6.3;
    enemy1[12] = 0.4;
    enemy1[29] = 1.0;
    labels.push("Enemy1 (100%)".to_string());

    // Ally 0: low health
    let mut ally0 = vec![0.0f32; 30];
    ally0[0] = 0.3;
    ally0[1] = 30.0;
    ally0[2] = 100.0;
    ally0[3] = -1.0;
    ally0[4] = 0.5;
    ally0[5] = 1.1;
    ally0[29] = 1.0;
    labels.push("Ally0 (30%)".to_string());

    let entities = vec![self_ent, enemy0, enemy1, ally0];
    let entity_types = vec![0, 1, 1, 2]; // self, enemy, enemy, ally

    // One threat zone
    let mut threat0 = vec![0.0f32; 8];
    threat0[0] = 2.0;   // x
    threat0[1] = 0.0;   // y
    threat0[2] = 3.0;   // radius
    threat0[3] = 1.0;   // damage_pct
    threat0[7] = 1.0;   // exists
    labels.push("Threat0".to_string());

    let threats = vec![threat0];

    // One cover position
    let mut pos0 = vec![0.0f32; 8];
    pos0[0] = -3.0;  // x
    pos0[1] = 2.0;   // y
    pos0[2] = 1.0;   // cover_value
    labels.push("Cover0".to_string());

    let positions = vec![pos0];

    (entities, entity_types, threats, positions, labels)
}

/// Load game state from a JSON file.
fn load_scenario_state(
    path: &PathBuf,
) -> Result<(Vec<Vec<f32>>, Vec<usize>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<String>), String> {
    let json_str = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {e}", path.display()))?;

    #[derive(serde::Deserialize)]
    struct ScenarioState {
        entities: Vec<Vec<f32>>,
        entity_types: Vec<usize>,
        #[serde(default)]
        threats: Vec<Vec<f32>>,
        #[serde(default)]
        positions: Vec<Vec<f32>>,
        #[serde(default)]
        entity_labels: Vec<String>,
    }

    let state: ScenarioState = serde_json::from_str(&json_str)
        .map_err(|e| format!("Failed to parse scenario state: {e}"))?;

    let mut labels = state.entity_labels;
    if labels.is_empty() {
        for (i, &t) in state.entity_types.iter().enumerate() {
            labels.push(match t {
                0 => format!("Self{i}"),
                1 => format!("Enemy{i}"),
                2 => format!("Ally{i}"),
                _ => format!("Ent{i}"),
            });
        }
        for i in 0..state.threats.len() {
            labels.push(format!("Threat{i}"));
        }
        for i in 0..state.positions.len() {
            labels.push(format!("Pos{i}"));
        }
    }

    Ok((state.entities, state.entity_types, state.threats, state.positions, labels))
}

// ---------------------------------------------------------------------------
// CLI args (defined here to keep cli/mod.rs simple)
// ---------------------------------------------------------------------------

use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Run model diagnostic inference and generate HTML report")]
pub struct DiagnoseArgs {
    /// Path to V3 actor-critic weights JSON.
    #[arg(long, default_value = "generated/actor_critic_v3_weights.json")]
    pub weights: PathBuf,

    /// Ability DSL text(s) to analyze. Can specify multiple.
    #[arg(long = "ability")]
    pub ability: Vec<String>,

    /// Path to scenario state JSON (entities, types, threats, positions).
    /// If not provided, uses a synthetic default state.
    #[arg(long = "scenario-state")]
    pub scenario_state: Option<PathBuf>,

    /// Output HTML file path.
    #[arg(long, short, default_value = "generated/diagnostics/report.html")]
    pub output: PathBuf,
}
