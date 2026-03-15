use crate::ai::core::{SimState, Team};
use super::game_state_nextstate::extract_game_state_with_ids;

// ---------------------------------------------------------------------------
// Ability operator dataset
// ---------------------------------------------------------------------------

/// A single training sample for the ability latent operator.
#[derive(Debug, Clone)]
pub struct OperatorSample {
    /// 23-dim entity features for 7 slots (flattened: 7*23=161).
    pub entity_features: Vec<f32>,
    /// Type ID per entity slot (7 values).
    pub entity_types: Vec<i32>,
    /// Mask per entity slot (1=padding, 0=real).
    pub entity_mask: Vec<i32>,
    /// 8-dim threat features for 8 slots (flattened).
    pub threat_features: Vec<f32>,
    pub threat_mask: Vec<i32>,
    /// 8-dim position features for 8 slots (flattened).
    pub position_features: Vec<f32>,
    pub position_mask: Vec<i32>,
    /// 34-dim ability slot tokens (flattened).
    pub ability_slot_features: Vec<f32>,
    pub ability_slot_types: Vec<i32>,
    pub ability_slot_mask: Vec<i32>,
    pub n_ability_slots: usize,
    /// Frozen CLS embedding for the cast ability (dimension matches transformer d_model).
    pub ability_cls: Vec<f32>,
    /// Caster entity slot index (0-6).
    pub caster_slot: i32,
    /// duration_norm = window_ms / MAX_WINDOW_MS.
    pub duration_norm: f32,
    /// 80-dim ability properties for loss masking.
    pub ability_props: Vec<f32>,
    /// Target deltas (flattened).
    pub target_hp: Vec<f32>,     // 7*3
    pub target_cc: Vec<f32>,     // 7
    pub target_cc_stun: Vec<f32>,// 7
    pub target_pos: Vec<f32>,    // 7*2
    pub target_exists: Vec<f32>, // 7
    /// Scenario index for train/val splitting.
    pub scenario_id: u32,
}

const OPERATOR_MAX_ENTITIES: usize = 7;
const OPERATOR_ENTITY_DIM: usize = 23;
const OPERATOR_MAX_WINDOW_MS: f32 = 6000.0;

/// Extract 23-dim entity features (dropping collapsed ability scalars from 30-dim).
fn extract_23dim_features(
    entities_30: &[Vec<f32>],
    entity_types_raw: &[u8],
) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let mut features = Vec::with_capacity(OPERATOR_MAX_ENTITIES * OPERATOR_ENTITY_DIM);
    let mut types = Vec::with_capacity(OPERATOR_MAX_ENTITIES);
    let mut mask = Vec::with_capacity(OPERATOR_MAX_ENTITIES);

    let self_x = entities_30.first().map(|e| e[5]).unwrap_or(0.0);
    let self_y = entities_30.first().map(|e| e[6]).unwrap_or(0.0);

    for (i, ent) in entities_30.iter().enumerate() {
        if i >= OPERATOR_MAX_ENTITIES {
            break;
        }
        // [0..15] vitals+position+terrain+combat → keep
        // [15..24] ability/heal/CC scalars → drop
        // [24..28] state → keep
        // [28..30] cumulative → keep
        // Add dx_from_self, dy_from_self
        features.extend_from_slice(&ent[0..15]);
        features.extend_from_slice(&ent[24..28]);
        features.extend_from_slice(&ent[28..30]);
        if i == 0 {
            features.push(0.0);
            features.push(0.0);
        } else {
            features.push(ent[5] - self_x);
            features.push(ent[6] - self_y);
        }
        types.push(entity_types_raw[i] as i32);
        mask.push(0);
    }

    // Pad to OPERATOR_MAX_ENTITIES
    let n = entities_30.len().min(OPERATOR_MAX_ENTITIES);
    for _ in n..OPERATOR_MAX_ENTITIES {
        features.extend(std::iter::repeat(0.0).take(OPERATOR_ENTITY_DIM));
        types.push(0);
        mask.push(1);
    }

    (features, types, mask)
}

/// Extract ability slot tokens for all entities in a snapshot.
fn extract_ability_slot_tokens(
    state: &SimState,
    unit_ids: &[u32],
    entity_types: &[u8],
    transformer: Option<&crate::ai::core::ability_transformer::AbilityTransformerWeights>,
    tokenizer: &crate::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
) -> (Vec<f32>, Vec<i32>, Vec<i32>, usize) {
    use crate::ai::effects::dsl::emit::emit_ability_dsl;

    let mut features = Vec::new();
    let mut types = Vec::new();
    let mut mask = Vec::new();
    let mut count = 0;

    for (ent_idx, &uid) in unit_ids.iter().enumerate() {
        if ent_idx >= OPERATOR_MAX_ENTITIES {
            break;
        }
        let unit = match state.units.iter().find(|u| u.id == uid) {
            Some(u) => u,
            None => continue,
        };

        let type_id = match entity_types.get(ent_idx) {
            Some(&0) => 5i32,
            Some(&1) => 6,
            Some(&2) => 7,
            _ => 5,
        };

        for slot in &unit.abilities {
            let is_ready = if slot.cooldown_remaining_ms == 0 { 1.0f32 } else { 0.0 };
            let cd_frac = if slot.def.cooldown_ms > 0 {
                slot.cooldown_remaining_ms as f32 / slot.def.cooldown_ms as f32
            } else {
                0.0
            };

            let cls = if let Some(tw) = transformer {
                let dsl = emit_ability_dsl(&slot.def);
                let tokens = tokenizer.encode_with_cls(&dsl);
                tw.encode_cls(&tokens)
            } else {
                vec![0.0f32; 128]
            };

            features.extend_from_slice(&cls);
            features.push(is_ready);
            features.push(cd_frac);
            types.push(type_id);
            mask.push(0);
            count += 1;
        }
    }

    (features, types, mask, count)
}

/// Compute ability effect window in milliseconds.
fn compute_ability_window(def: &crate::ai::effects::AbilityDef) -> f32 {
    use crate::ai::effects::Delivery;

    let mut max_duration = def.cast_time_ms as f32;

    if let Some(ref delivery) = def.delivery {
        let d = match delivery {
            Delivery::Channel { duration_ms, .. } => *duration_ms as f32,
            Delivery::Zone { duration_ms, .. } => *duration_ms as f32,
            Delivery::Tether { .. } => 3000.0,
            Delivery::Trap { duration_ms, arm_time_ms, .. } => {
                (*duration_ms + *arm_time_ms) as f32
            }
            Delivery::Projectile { speed, .. } => {
                if *speed > 0.0 { 10.0 / speed * 1000.0 } else { 0.0 }
            }
            _ => 0.0,
        };
        max_duration = max_duration.max(d);
    }

    // Also check cooldown as a proxy for effect duration (abilities often
    // have effects that last roughly their cooldown)
    max_duration = max_duration.max(def.cooldown_ms as f32 * 0.5);

    (max_duration * 1.2).min(OPERATOR_MAX_WINDOW_MS).max(500.0)
}

/// Generate operator training dataset by replaying scenarios and capturing
/// ability cast events with before/after state snapshots.
pub fn generate_operator_dataset_streaming(
    initial_sim: SimState,
    initial_squad_ai: crate::ai::squad::SquadAiState,
    grid_nav: Option<crate::ai::pathing::GridNav>,
    max_ticks: u64,
    scenario_id: u32,
    transformer: Option<&crate::ai::core::ability_transformer::AbilityTransformerWeights>,
    tokenizer: &crate::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    mut emit: impl FnMut(OperatorSample),
) -> usize {
    use crate::ai::core::{step, FIXED_TICK_MS};
    use crate::ai::core::events::SimEvent;
    use crate::ai::squad::generate_intents;
    use crate::ai::core::ability_encoding::extract_ability_properties;

    let mut sim = initial_sim;
    if let Some(nav) = grid_nav {
        sim.grid_nav = Some(nav);
    }
    let mut squad_ai = initial_squad_ai;
    let mut count = 0usize;

    // Pending cast: state snapshot before, waiting for target_tick
    struct Pending {
        caster_id: u32,
        ability_idx: usize,
        target_tick: u64,
        entity_features: Vec<f32>,
        entity_types: Vec<i32>,
        entity_mask: Vec<i32>,
        threat_features: Vec<f32>,
        threat_mask: Vec<i32>,
        position_features: Vec<f32>,
        position_mask: Vec<i32>,
        ability_slot_features: Vec<f32>,
        ability_slot_types: Vec<i32>,
        ability_slot_mask: Vec<i32>,
        n_ability_slots: usize,
        caster_slot: i32,
        cls: Vec<f32>,
        props: Vec<f32>,
        duration_norm: f32,
        unit_ids: Vec<u32>,
    }

    let mut pending: Vec<Pending> = Vec::new();

    for tick in 0..max_ticks {
        let intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        // Capture hero ability cast events from this tick
        for event in &events {
            if let SimEvent::AbilityUsed { unit_id, ability_index, .. } = event {
                let unit = match sim.units.iter().find(|u| u.id == *unit_id) {
                    Some(u) => u,
                    None => continue,
                };

                let ability_idx = *ability_index;
                if ability_idx >= unit.abilities.len() {
                    continue;
                }

                let def = &unit.abilities[ability_idx].def;
                let window_ms = compute_ability_window(def);
                let window_ticks = (window_ms / FIXED_TICK_MS as f32).ceil() as u64;
                let target_tick = tick + window_ticks;

                if target_tick >= max_ticks {
                    continue;
                }

                // Snapshot state before from caster's perspective
                let (gs, unit_ids) = extract_game_state_with_ids(&sim, unit);
                let (ent_features, ent_types, ent_mask) =
                    extract_23dim_features(&gs.entities, &gs.entity_types);

                // Also emit a "just resolved" sample to capture immediate effect
                // (after cast time + 1 tick for effect application)
                let cast_ticks = (def.cast_time_ms as f32 / FIXED_TICK_MS as f32).ceil() as u64;
                let instant_tick = tick + cast_ticks + 1;
                let has_instant_target = instant_tick < max_ticks && instant_tick < target_tick;

                // Pad threats
                let mut threat_features = Vec::new();
                let mut threat_mask = Vec::new();
                for (i, thr) in gs.threats.iter().enumerate() {
                    if i >= 8 { break; }
                    threat_features.extend_from_slice(thr);
                    threat_mask.push(0i32);
                }
                for _ in gs.threats.len().min(8)..8 {
                    threat_features.extend(std::iter::repeat(0.0f32).take(8));
                    threat_mask.push(1);
                }

                // Pad positions
                let mut position_features = Vec::new();
                let mut position_mask = Vec::new();
                for (i, pos) in gs.positions.iter().enumerate() {
                    if i >= 8 { break; }
                    position_features.extend_from_slice(pos);
                    position_mask.push(0i32);
                }
                for _ in gs.positions.len().min(8)..8 {
                    position_features.extend(std::iter::repeat(0.0f32).take(8));
                    position_mask.push(1);
                }

                // Ability slot tokens
                let (abl_feat, abl_types, abl_mask, n_abl) =
                    extract_ability_slot_tokens(&sim, &unit_ids, &gs.entity_types, transformer, tokenizer);

                // Caster slot
                let caster_slot = unit_ids.iter()
                    .position(|&id| id == *unit_id)
                    .unwrap_or(0) as i32;

                // CLS embedding for the cast ability
                use crate::ai::effects::dsl::emit::emit_ability_dsl;
                let cls = if let Some(tw) = transformer {
                    let dsl = emit_ability_dsl(def);
                    let tokens = tokenizer.encode_with_cls(&dsl);
                    tw.encode_cls(&tokens)
                } else {
                    vec![0.0f32; 128]
                };

                let props = extract_ability_properties(def).to_vec();

                // Emit instant-effect sample (delta=1 tick) to capture
                // immediate damage/heal before ongoing combat obscures it
                if has_instant_target {
                    pending.push(Pending {
                        caster_id: *unit_id,
                        ability_idx,
                        target_tick: instant_tick,
                        entity_features: ent_features.clone(),
                        entity_types: ent_types.clone(),
                        entity_mask: ent_mask.clone(),
                        threat_features: threat_features.clone(),
                        threat_mask: threat_mask.clone(),
                        position_features: position_features.clone(),
                        position_mask: position_mask.clone(),
                        ability_slot_features: abl_feat.clone(),
                        ability_slot_types: abl_types.clone(),
                        ability_slot_mask: abl_mask.clone(),
                        n_ability_slots: n_abl,
                        caster_slot,
                        cls: cls.clone(),
                        props: props.clone(),
                        duration_norm: (cast_ticks + 1) as f32 * FIXED_TICK_MS as f32 / OPERATOR_MAX_WINDOW_MS,
                        unit_ids: unit_ids.clone(),
                    });
                }

                // Windowed sample (full ability window)
                pending.push(Pending {
                    caster_id: *unit_id,
                    ability_idx,
                    target_tick,
                    entity_features: ent_features,
                    entity_types: ent_types,
                    entity_mask: ent_mask,
                    threat_features,
                    threat_mask,
                    position_features,
                    position_mask,
                    ability_slot_features: abl_feat,
                    ability_slot_types: abl_types,
                    ability_slot_mask: abl_mask,
                    n_ability_slots: n_abl,
                    caster_slot,
                    cls,
                    props,
                    duration_norm: window_ms / OPERATOR_MAX_WINDOW_MS,
                    unit_ids: unit_ids.clone(),
                });
            }
        }

        // Resolve pending casts that reached their target tick
        let mut i = 0;
        while i < pending.len() {
            if tick >= pending[i].target_tick {
                let pc = pending.swap_remove(i);

                // Find caster unit (may have died)
                let caster = match sim.units.iter().find(|u| u.id == pc.caster_id) {
                    Some(u) => u,
                    None => continue,
                };

                // Snapshot state after
                let (gs_after, unit_ids_after) = extract_game_state_with_ids(&sim, caster);
                let (after_features, _, _) =
                    extract_23dim_features(&gs_after.entities, &gs_after.entity_types);

                // Compute target deltas aligned by unit_id
                let mut target_hp = vec![0.0f32; OPERATOR_MAX_ENTITIES * 3];
                let mut target_cc = vec![0.0f32; OPERATOR_MAX_ENTITIES];
                let mut target_cc_stun = vec![0.0f32; OPERATOR_MAX_ENTITIES];
                let mut target_pos = vec![0.0f32; OPERATOR_MAX_ENTITIES * 2];
                let mut target_exists = vec![1.0f32; OPERATOR_MAX_ENTITIES];

                for (before_slot, &before_uid) in pc.unit_ids.iter().enumerate() {
                    if before_slot >= OPERATOR_MAX_ENTITIES {
                        break;
                    }
                    let after_slot = unit_ids_after.iter().position(|&id| id == before_uid);

                    if let Some(after_idx) = after_slot {
                        if after_idx >= OPERATOR_MAX_ENTITIES { continue; }
                        let b = before_slot * OPERATOR_ENTITY_DIM;
                        let a = after_idx * OPERATOR_ENTITY_DIM;

                        // HP delta: indices 0-2
                        for j in 0..3 {
                            target_hp[before_slot * 3 + j] =
                                after_features[a + j] - pc.entity_features[b + j];
                        }

                        // CC remaining: index 17 (pos 15+2 in 23-dim: state group at [15..19])
                        let cc_idx = 17;
                        target_cc[before_slot] = after_features[a + cc_idx];
                        target_cc_stun[before_slot] =
                            if after_features[a + cc_idx] > 0.01 { 1.0 } else { 0.0 };

                        // Position delta: indices 5-6
                        for j in 0..2 {
                            target_pos[before_slot * 2 + j] =
                                after_features[a + 5 + j] - pc.entity_features[b + 5 + j];
                        }

                        // Exists: index 20
                        target_exists[before_slot] = after_features[a + 20];
                    } else {
                        // Unit died
                        target_exists[before_slot] = 0.0;
                    }
                }

                emit(OperatorSample {
                    entity_features: pc.entity_features,
                    entity_types: pc.entity_types,
                    entity_mask: pc.entity_mask,
                    threat_features: pc.threat_features,
                    threat_mask: pc.threat_mask,
                    position_features: pc.position_features,
                    position_mask: pc.position_mask,
                    ability_slot_features: pc.ability_slot_features,
                    ability_slot_types: pc.ability_slot_types,
                    ability_slot_mask: pc.ability_slot_mask,
                    n_ability_slots: pc.n_ability_slots,
                    ability_cls: pc.cls,
                    caster_slot: pc.caster_slot,
                    duration_norm: pc.duration_norm,
                    ability_props: pc.props,
                    target_hp,
                    target_cc,
                    target_cc_stun,
                    target_pos,
                    target_exists,
                    scenario_id,
                });
                count += 1;
            } else {
                i += 1;
            }
        }

        // Check termination
        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 || heroes_alive == 0 {
            break;
        }
    }

    count
}
