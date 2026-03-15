//! V4 dual-head actor-critic weights and inference.

use serde::Deserialize;
use super::weights::{
    gelu,
    FlatLinear, FlatLayerNorm, TransformerLayer, TransformerScratch,
    FlatCrossAttention, CrossAttnJson,
    LinearWeights, LayerNormWeights, TransformerLayerJson,
    AC_MAX_ABILITIES,
    HeadJson, ActorCriticArchJson,
};
use super::weights_encoder::{FlatEntityEncoderV3, EntityEncoderV3Json};
use super::weights_actor_critic_v3::EntityStateV3;

// ---------------------------------------------------------------------------
// Combat pointer head
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CombatHeadJson {
    combat_type_head: HeadJson,
    pointer_key: LinearWeights,
    attack_query: LinearWeights,
    ability_queries: Vec<LinearWeights>,
}

#[derive(Debug, Clone)]
struct FlatCombatHead {
    combat_type_l1: FlatLinear,
    combat_type_l2: FlatLinear,
    pointer_key: FlatLinear,
    attack_query: FlatLinear,
    ability_queries: Vec<FlatLinear>,
    d_model: usize,
    scale: f32,
}

impl FlatCombatHead {
    fn from_json(j: &CombatHeadJson, d_model: usize) -> Self {
        Self {
            combat_type_l1: FlatLinear::from_json(&j.combat_type_head.linear1),
            combat_type_l2: FlatLinear::from_json(&j.combat_type_head.linear2),
            pointer_key: FlatLinear::from_json(&j.pointer_key),
            attack_query: FlatLinear::from_json(&j.attack_query),
            ability_queries: j.ability_queries.iter().map(FlatLinear::from_json).collect(),
            d_model,
            scale: (d_model as f32).sqrt().recip(),
        }
    }

    fn forward(
        &self,
        pooled: &[f32],
        tokens: &[f32],
        mask: &[bool],
        n_total: usize,
        type_ids: &[usize],
        ability_cross_embs: &[Option<Vec<f32>>],
    ) -> (Vec<f32>, Vec<f32>, Vec<Option<Vec<f32>>>) {
        let d = self.d_model;

        // Combat type logits
        let mut h = vec![0.0f32; d];
        self.combat_type_l1.forward(pooled, &mut h);
        for v in h.iter_mut() { *v = gelu(*v); }
        let n_types = self.combat_type_l2.out_dim;
        let mut combat_logits = vec![0.0f32; n_types];
        self.combat_type_l2.forward(&h, &mut combat_logits);

        // Compute keys for all tokens
        let mut keys = vec![0.0f32; n_total * d];
        for i in 0..n_total {
            self.pointer_key.forward(
                &tokens[i * d..(i + 1) * d],
                &mut keys[i * d..(i + 1) * d],
            );
        }

        // Attack pointer: query from pooled, masked to enemies only (type_id == 1)
        let mut atk_q = vec![0.0f32; d];
        self.attack_query.forward(pooled, &mut atk_q);
        let mut attack_ptr = vec![f32::NEG_INFINITY; n_total];
        for i in 0..n_total {
            if mask[i] && type_ids[i] == 1 {
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot += atk_q[j] * keys[i * d + j];
                }
                attack_ptr[i] = dot * self.scale;
            }
        }

        // Ability pointers
        let mut ability_ptrs: Vec<Option<Vec<f32>>> = Vec::with_capacity(AC_MAX_ABILITIES);
        for (ab_idx, cross_emb_opt) in ability_cross_embs.iter().enumerate().take(AC_MAX_ABILITIES) {
            if let Some(cross_emb) = cross_emb_opt {
                if ab_idx < self.ability_queries.len() {
                    let mut ab_q = vec![0.0f32; d];
                    self.ability_queries[ab_idx].forward(cross_emb, &mut ab_q);
                    let mut ab_ptr = vec![f32::NEG_INFINITY; n_total];
                    for i in 0..n_total {
                        if mask[i] {
                            let mut dot = 0.0f32;
                            for j in 0..d {
                                dot += ab_q[j] * keys[i * d + j];
                            }
                            ab_ptr[i] = dot * self.scale;
                        }
                    }
                    ability_ptrs.push(Some(ab_ptr));
                } else {
                    ability_ptrs.push(None);
                }
            } else {
                ability_ptrs.push(None);
            }
        }

        (combat_logits, attack_ptr, ability_ptrs)
    }
}

// ---------------------------------------------------------------------------
// Move head
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FlatMoveHead {
    l1: FlatLinear,
    l2: FlatLinear,
}

impl FlatMoveHead {
    fn from_json(j: &HeadJson) -> Self {
        Self {
            l1: FlatLinear::from_json(&j.linear1),
            l2: FlatLinear::from_json(&j.linear2),
        }
    }

    fn forward(&self, pooled: &[f32]) -> Vec<f32> {
        let d = self.l1.out_dim;
        let mut h = vec![0.0f32; d];
        self.l1.forward(pooled, &mut h);
        for v in h.iter_mut() { *v = gelu(*v); }
        let n_out = self.l2.out_dim;
        let mut out = vec![0.0f32; n_out];
        self.l2.forward(&h, &mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// V4 dual-head output
// ---------------------------------------------------------------------------

/// V4 output: separate move direction + combat action.
#[derive(Debug)]
pub struct DualHeadOutput {
    /// Movement direction logits (9 elements: 8 cardinal + stay).
    pub move_logits: Vec<f32>,
    /// Combat type logits (10 elements: attack(0), hold(1), ability_0..7(2..9)).
    pub combat_logits: Vec<f32>,
    /// Attack pointer logits over all tokens.
    pub attack_ptr: Vec<f32>,
    /// Per-ability pointer logits (None if ability slot empty).
    pub ability_ptrs: Vec<Option<Vec<f32>>>,
}

// ---------------------------------------------------------------------------
// V4 actor-critic JSON schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ActorCriticV4FileJson {
    #[allow(dead_code)]
    format: Option<String>,
    architecture: ActorCriticArchJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder_v3: EntityEncoderV3Json,
    cross_attn: CrossAttnJson,
    move_head: HeadJson,
    combat_head: CombatHeadJson,
    external_cls_proj: Option<LinearWeights>,
}

/// V4 actor-critic weights with dual-head action space.
#[derive(Debug, Clone)]
pub struct ActorCriticWeightsV4 {
    d_model: usize,
    max_seq_len: usize,
    pad_id: usize,
    vocab_size: usize,

    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    out_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,

    entity_encoder: FlatEntityEncoderV3,
    cross_attn: FlatCrossAttention,
    move_head: FlatMoveHead,
    combat_head: FlatCombatHead,
    external_cls_proj: Option<FlatLinear>,
}

impl ActorCriticWeightsV4 {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: ActorCriticV4FileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        let token_emb: Vec<f32> = file.token_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();
        let pos_emb: Vec<f32> = file.position_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err("token_emb size mismatch".to_string());
        }

        let layers: Vec<TransformerLayer> = file.layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, arch.d_model, arch.n_heads))
            .collect();

        let entity_encoder = FlatEntityEncoderV3::from_json(
            &file.entity_encoder_v3, arch.d_model, arch.n_heads,
        );
        let cross_attn = FlatCrossAttention::from_json(
            &file.cross_attn, arch.d_model, arch.n_heads,
        );
        let move_head = FlatMoveHead::from_json(&file.move_head);
        let combat_head = FlatCombatHead::from_json(&file.combat_head, arch.d_model);

        Ok(Self {
            d_model: arch.d_model,
            max_seq_len: arch.max_seq_len,
            pad_id: arch.pad_id,
            vocab_size: arch.vocab_size,
            token_emb,
            pos_emb,
            out_norm: FlatLayerNorm::from_json(&file.output_norm),
            layers,
            entity_encoder,
            cross_attn,
            move_head,
            combat_head,
            external_cls_proj: file.external_cls_proj.as_ref().map(FlatLinear::from_json),
        })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read V4 weights: {e}"))?;
        Self::from_json(&data)
    }

    pub fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        if let Some(proj) = &self.external_cls_proj {
            let mut out = vec![0.0f32; self.d_model];
            proj.forward(cls, &mut out);
            out
        } else {
            cls.to_vec()
        }
    }

    pub fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.d_model;
        let seq_len = token_ids.len().min(self.max_seq_len);

        let mut seq = vec![0.0f32; seq_len * d];
        let mut mask = vec![false; seq_len];

        for (t, &tid) in token_ids.iter().take(seq_len).enumerate() {
            let id = (tid as usize).min(self.vocab_size - 1);
            mask[t] = id != self.pad_id;
            for i in 0..d {
                seq[t * d + i] = self.token_emb[id * d + i] + self.pos_emb[t * d + i];
            }
        }

        let mut scratch = TransformerScratch::default();
        for layer in &self.layers {
            layer.forward(&mut seq, seq_len, &mask, &mut scratch);
        }

        let mut cls = vec![0.0f32; d];
        cls.copy_from_slice(&seq[0..d]);
        self.out_norm.forward(&mut cls);
        cls
    }

    pub fn encode_entities_v3(
        &self,
        entities: &[&[f32]],
        entity_types: &[usize],
        threats: &[&[f32]],
        positions: &[&[f32]],
    ) -> EntityStateV3 {
        let d = self.d_model;
        let (tokens, mask, n_total, type_ids) = self.entity_encoder.forward(
            entities, entity_types, threats, positions,
        );

        let mut pooled = vec![0.0f32; d];
        let mut count = 0.0f32;
        for i in 0..n_total {
            if mask[i] {
                count += 1.0;
                for j in 0..d {
                    pooled[j] += tokens[i * d + j];
                }
            }
        }
        if count > 0.0 {
            for v in &mut pooled { *v /= count; }
        }

        EntityStateV3 { tokens, mask, pooled, type_ids, n_total }
    }

    /// Compute dual-head action logits: move direction + combat pointer.
    pub fn dual_head_logits(
        &self,
        entity_state: &EntityStateV3,
        ability_cls: &[Option<&[f32]>],
    ) -> DualHeadOutput {
        let n_total = entity_state.n_total;

        // Cross-attend each ability CLS to entity tokens
        let mut ability_cross_embs: Vec<Option<Vec<f32>>> = Vec::with_capacity(AC_MAX_ABILITIES);
        for cls_opt in ability_cls.iter().take(AC_MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb = self.cross_attn.forward(
                    cls, &entity_state.tokens, &entity_state.mask, n_total,
                );
                ability_cross_embs.push(Some(cross_emb));
            } else {
                ability_cross_embs.push(None);
            }
        }

        // Move direction logits (9-way)
        let move_logits = self.move_head.forward(&entity_state.pooled);

        // Combat pointer logits
        let (combat_logits, attack_ptr, ability_ptrs) = self.combat_head.forward(
            &entity_state.pooled,
            &entity_state.tokens,
            &entity_state.mask,
            n_total,
            &entity_state.type_ids,
            &ability_cross_embs,
        );

        DualHeadOutput { move_logits, combat_logits, attack_ptr, ability_ptrs }
    }
}
