//! Actor-critic V1 and V2 weights and inference.

use serde::Deserialize;
use super::weights::{
    FlatLinear, FlatLayerNorm, TransformerLayer, TransformerScratch,
    FlatCrossAttention, CrossAttnJson,
    LinearWeights, LayerNormWeights, TransformerLayerJson,
    NUM_ENTITIES, NUM_BASE_ACTIONS, AC_MAX_ABILITIES, AC_NUM_ACTIONS,
    HeadJson, ActorCriticArchJson,
};
use super::weights_encoder::{
    FlatEntityEncoder, EntityEncoderJson,
    FlatEntityEncoderV2, EntityEncoderV2Json,
};

/// Pre-computed entity state for a single tick.
#[derive(Debug, Clone)]
pub struct EntityState {
    pub tokens: Vec<f32>,
    pub mask: Vec<bool>,
    pub pooled: Vec<f32>,
}

// ---------------------------------------------------------------------------
// V1 actor-critic JSON schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ActorCriticFileJson {
    #[allow(dead_code)]
    format: Option<String>,
    architecture: ActorCriticArchJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder: EntityEncoderJson,
    cross_attn: CrossAttnJson,
    base_head: HeadJson,
    ability_proj: HeadJson,
    /// Optional projection for external CLS embeddings (e.g. 128->32 behavioral).
    external_cls_proj: Option<LinearWeights>,
}

/// Frozen actor-critic weights for 14-action policy inference.
#[derive(Debug, Clone)]
pub struct ActorCriticWeights {
    d_model: usize,
    max_seq_len: usize,
    pad_id: usize,
    vocab_size: usize,

    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    out_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,

    entity_encoder: FlatEntityEncoder,
    cross_attn: FlatCrossAttention,

    // Base head: pooled entity state -> 6 logits
    base_l1: FlatLinear,
    base_l2: FlatLinear,
    // Ability projection: cross-attended CLS -> 1 logit
    ability_l1: FlatLinear,
    ability_l2: FlatLinear,
    // Optional projection for external CLS (e.g. 128-dim behavioral -> d_model)
    external_cls_proj: Option<FlatLinear>,
}

impl ActorCriticWeights {
    /// Load from JSON string exported by `training/export_actor_critic.py`.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: ActorCriticFileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        let token_emb: Vec<f32> = file.token_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();
        let pos_emb: Vec<f32> = file.position_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err(format!("token_emb size mismatch"));
        }

        let layers: Vec<TransformerLayer> = file.layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, arch.d_model, arch.n_heads))
            .collect();

        let entity_encoder = FlatEntityEncoder::from_json(
            &file.entity_encoder, arch.d_model, arch.n_heads,
        );
        let cross_attn = FlatCrossAttention::from_json(
            &file.cross_attn, arch.d_model, arch.n_heads,
        );

        if arch.num_base_actions != NUM_BASE_ACTIONS {
            return Err(format!(
                "num_base_actions mismatch: {} vs {}",
                arch.num_base_actions, NUM_BASE_ACTIONS
            ));
        }

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
            base_l1: FlatLinear::from_json(&file.base_head.linear1),
            base_l2: FlatLinear::from_json(&file.base_head.linear2),
            ability_l1: FlatLinear::from_json(&file.ability_proj.linear1),
            ability_l2: FlatLinear::from_json(&file.ability_proj.linear2),
            external_cls_proj: file.external_cls_proj.as_ref().map(FlatLinear::from_json),
        })
    }

    /// Project external CLS embedding (e.g. 128-dim behavioral) to d_model.
    /// If no projection layer exists, returns the input unchanged.
    pub fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        if let Some(proj) = &self.external_cls_proj {
            let mut out = vec![0.0f32; self.d_model];
            proj.forward(cls, &mut out);
            // GELU activation not needed for single linear projection
            out
        } else {
            cls.to_vec()
        }
    }

    /// Encode ability tokens -> [CLS] embedding. Cache at fight start.
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

    /// Encode game state -> entity tokens + mask + mean-pooled state.
    pub fn encode_entities(&self, game_state: &[f32]) -> EntityState {
        let d = self.d_model;
        let (tokens, mask) = self.entity_encoder.forward(game_state);

        // Mean pool over existing entities
        let mut pooled = vec![0.0f32; d];
        let mut count = 0.0f32;
        for ent in 0..NUM_ENTITIES {
            if mask[ent] {
                count += 1.0;
                for i in 0..d {
                    pooled[i] += tokens[ent * d + i];
                }
            }
        }
        if count > 0.0 {
            for v in &mut pooled { *v /= count; }
        }

        EntityState { tokens, mask, pooled }
    }

    /// Compute 14-action logits.
    ///
    /// * `entity_state` -- precomputed entity encoding for this tick
    /// * `ability_cls` -- list of cached [CLS] embeddings per ability slot (None if slot empty)
    pub fn action_logits(
        &self,
        entity_state: &EntityState,
        ability_cls: &[Option<&[f32]>],
    ) -> [f32; AC_NUM_ACTIONS] {
        let d = self.d_model;
        let mut logits = [f32::NEG_INFINITY; AC_NUM_ACTIONS];

        // Base action logits from pooled entity state
        let mut base_hidden = vec![0.0f32; self.base_l1.out_dim];
        self.base_l1.forward_gelu(&entity_state.pooled, &mut base_hidden);
        let mut base_out = vec![0.0f32; NUM_BASE_ACTIONS];
        self.base_l2.forward(&base_hidden, &mut base_out);

        // Map base actions: [attack_near, attack_weak, attack_focus, move_toward, move_away, hold]
        // -> action indices [0, 1, 2, 11, 12, 13]
        logits[0] = base_out[0];   // attack nearest
        logits[1] = base_out[1];   // attack weakest
        logits[2] = base_out[2];   // attack focus
        logits[11] = base_out[3];  // move toward
        logits[12] = base_out[4];  // move away
        logits[13] = base_out[5];  // hold

        // Ability logits: cross-attend each ability [CLS] with entity tokens
        let mut ability_hidden = vec![0.0f32; self.ability_l1.out_dim];
        let mut ability_out = vec![0.0f32; 1];

        for (idx, cls_opt) in ability_cls.iter().enumerate().take(AC_MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb = self.cross_attn.forward(
                    cls, &entity_state.tokens, &entity_state.mask, NUM_ENTITIES,
                );
                self.ability_l1.forward_gelu(&cross_emb, &mut ability_hidden);
                self.ability_l2.forward(&ability_hidden, &mut ability_out);
                logits[3 + idx] = ability_out[0];
            }
        }

        logits
    }
}

// ---------------------------------------------------------------------------
// V2 actor-critic inference
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ActorCriticV2FileJson {
    #[allow(dead_code)]
    format: Option<String>,
    architecture: ActorCriticArchJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder_v2: EntityEncoderV2Json,
    cross_attn: CrossAttnJson,
    base_head: HeadJson,
    ability_proj: HeadJson,
}

/// V2 actor-critic weights with variable-length entity encoder.
#[derive(Debug, Clone)]
pub struct ActorCriticWeightsV2 {
    d_model: usize,
    max_seq_len: usize,
    pad_id: usize,
    vocab_size: usize,

    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    out_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,

    entity_encoder: FlatEntityEncoderV2,
    cross_attn: FlatCrossAttention,

    base_l1: FlatLinear,
    base_l2: FlatLinear,
    ability_l1: FlatLinear,
    ability_l2: FlatLinear,
}

impl ActorCriticWeightsV2 {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: ActorCriticV2FileJson =
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

        let entity_encoder = FlatEntityEncoderV2::from_json(
            &file.entity_encoder_v2, arch.d_model, arch.n_heads,
        );
        let cross_attn = FlatCrossAttention::from_json(
            &file.cross_attn, arch.d_model, arch.n_heads,
        );

        if arch.num_base_actions != NUM_BASE_ACTIONS {
            return Err(format!(
                "num_base_actions mismatch: {} vs {}",
                arch.num_base_actions, NUM_BASE_ACTIONS
            ));
        }

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
            base_l1: FlatLinear::from_json(&file.base_head.linear1),
            base_l2: FlatLinear::from_json(&file.base_head.linear2),
            ability_l1: FlatLinear::from_json(&file.ability_proj.linear1),
            ability_l2: FlatLinear::from_json(&file.ability_proj.linear2),
        })
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

    /// Encode v2 game state -> entity state with variable-length tokens.
    pub fn encode_entities_v2(
        &self,
        entities: &[&[f32]],
        entity_types: &[usize],
        threats: &[&[f32]],
    ) -> EntityState {
        let d = self.d_model;
        let (tokens, mask, n_total) = self.entity_encoder.forward(
            entities, entity_types, threats,
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

        EntityState { tokens, mask, pooled }
    }

    /// Compute 14-action logits using v2 entity state.
    pub fn action_logits(
        &self,
        entity_state: &EntityState,
        ability_cls: &[Option<&[f32]>],
    ) -> [f32; AC_NUM_ACTIONS] {
        let _d = self.d_model;
        let n_entities = entity_state.mask.len();
        let mut logits = [f32::NEG_INFINITY; AC_NUM_ACTIONS];

        let mut base_hidden = vec![0.0f32; self.base_l1.out_dim];
        self.base_l1.forward_gelu(&entity_state.pooled, &mut base_hidden);
        let mut base_out = vec![0.0f32; NUM_BASE_ACTIONS];
        self.base_l2.forward(&base_hidden, &mut base_out);

        logits[0] = base_out[0];
        logits[1] = base_out[1];
        logits[2] = base_out[2];
        logits[11] = base_out[3];
        logits[12] = base_out[4];
        logits[13] = base_out[5];

        let mut ability_hidden = vec![0.0f32; self.ability_l1.out_dim];
        let mut ability_out = vec![0.0f32; 1];

        for (idx, cls_opt) in ability_cls.iter().enumerate().take(AC_MAX_ABILITIES) {
            if let Some(cls) = cls_opt {
                let cross_emb = self.cross_attn.forward(
                    cls, &entity_state.tokens, &entity_state.mask, n_entities,
                );
                self.ability_l1.forward_gelu(&cross_emb, &mut ability_hidden);
                self.ability_l2.forward(&ability_hidden, &mut ability_out);
                logits[3 + idx] = ability_out[0];
            }
        }

        logits
    }
}
