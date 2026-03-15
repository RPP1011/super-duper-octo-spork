//! Entity encoder weights and inference (V1, V2, V3).

use serde::Deserialize;
use super::weights::{
    FlatLinear, FlatLayerNorm, TransformerLayer, TransformerScratch,
    LinearWeights, LayerNormWeights, TransformerLayerJson,
    ENTITY_DIM, NUM_ENTITIES, NUM_ENTITY_TYPES, ENTITY_TYPE_IDS,
};

// ---------------------------------------------------------------------------
// V1 entity encoder
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(super) struct EntityEncoderJson {
    pub proj: LinearWeights,
    pub type_emb: Vec<Vec<f32>>,  // [3 x d_model]
    pub input_norm: LayerNormWeights,
    /// Self-attention layers over entities (from pre-training).
    /// Empty if no pre-trained encoder loaded.
    #[serde(default)]
    pub self_attn_layers: Vec<TransformerLayerJson>,
    pub out_norm: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub(super) struct FlatEntityEncoder {
    pub proj: FlatLinear,          // (ENTITY_DIM -> d_model)
    pub type_emb: Vec<f32>,        // [NUM_ENTITY_TYPES x d_model]
    pub input_norm: FlatLayerNorm,
    pub self_attn_layers: Vec<TransformerLayer>,
    pub out_norm: FlatLayerNorm,
    pub d_model: usize,
}

impl FlatEntityEncoder {
    pub fn from_json(ej: &EntityEncoderJson, d_model: usize, n_heads: usize) -> Self {
        let type_emb: Vec<f32> = ej.type_emb.iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let self_attn_layers: Vec<TransformerLayer> = ej.self_attn_layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, d_model, n_heads))
            .collect();
        Self {
            proj: FlatLinear::from_json(&ej.proj),
            type_emb,
            input_norm: FlatLayerNorm::from_json(&ej.input_norm),
            self_attn_layers,
            out_norm: FlatLayerNorm::from_json(&ej.out_norm),
            d_model,
        }
    }

    /// Encode game_state (210 floats) into entity tokens [7 x d_model].
    /// Returns (entity_tokens, entity_mask) where mask[i] = true means attend.
    pub fn forward(&self, game_state: &[f32]) -> (Vec<f32>, Vec<bool>) {
        let d = self.d_model;
        let mut tokens = vec![0.0f32; NUM_ENTITIES * d];
        let mut mask = vec![false; NUM_ENTITIES];

        for ent in 0..NUM_ENTITIES {
            let entity_feats = &game_state[ent * ENTITY_DIM..(ent + 1) * ENTITY_DIM];

            // exists = feature index 29
            let exists = entity_feats[29] > 0.5;
            mask[ent] = exists;

            // Project entity features to d_model
            self.proj.forward(entity_feats, &mut tokens[ent * d..(ent + 1) * d]);

            // Add type embedding
            let type_id = ENTITY_TYPE_IDS[ent];
            for i in 0..d {
                tokens[ent * d + i] += self.type_emb[type_id * d + i];
            }

            // Input layer norm per entity
            self.input_norm.forward(&mut tokens[ent * d..(ent + 1) * d]);
        }

        // Self-attention over entities (from pre-training)
        let mut scratch = TransformerScratch::default();
        for layer in &self.self_attn_layers {
            layer.forward(&mut tokens, NUM_ENTITIES, &mask, &mut scratch);
        }

        // Output norm per entity
        for ent in 0..NUM_ENTITIES {
            self.out_norm.forward(&mut tokens[ent * d..(ent + 1) * d]);
        }

        (tokens, mask)
    }
}

// ---------------------------------------------------------------------------
// V2 entity encoder: variable-length entities + threats
// ---------------------------------------------------------------------------

pub(super) const THREAT_DIM: usize = 8;
pub(super) const NUM_ENTITY_TYPES_V2: usize = 4; // self=0, enemy=1, ally=2, threat=3

#[derive(Debug, Deserialize)]
pub(super) struct EntityEncoderV2Json {
    pub entity_proj: LinearWeights,
    pub threat_proj: LinearWeights,
    pub type_emb: Vec<Vec<f32>>,  // [4 x d_model]
    pub input_norm: LayerNormWeights,
    #[serde(default)]
    pub self_attn_layers: Vec<TransformerLayerJson>,
    pub out_norm: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub(super) struct FlatEntityEncoderV2 {
    pub entity_proj: FlatLinear,   // (30 -> d_model)
    pub threat_proj: FlatLinear,   // (8 -> d_model)
    pub type_emb: Vec<f32>,        // [4 x d_model]
    pub input_norm: FlatLayerNorm,
    pub self_attn_layers: Vec<TransformerLayer>,
    pub out_norm: FlatLayerNorm,
    pub d_model: usize,
}

impl FlatEntityEncoderV2 {
    pub fn from_json(ej: &EntityEncoderV2Json, d_model: usize, n_heads: usize) -> Self {
        let type_emb: Vec<f32> = ej.type_emb.iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let self_attn_layers: Vec<TransformerLayer> = ej.self_attn_layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, d_model, n_heads))
            .collect();
        Self {
            entity_proj: FlatLinear::from_json(&ej.entity_proj),
            threat_proj: FlatLinear::from_json(&ej.threat_proj),
            type_emb,
            input_norm: FlatLayerNorm::from_json(&ej.input_norm),
            self_attn_layers,
            out_norm: FlatLayerNorm::from_json(&ej.out_norm),
            d_model,
        }
    }

    /// Encode variable-length entities + threats into tokens.
    /// Returns (tokens [n_total x d_model], mask [n_total], n_total).
    pub fn forward(
        &self,
        entities: &[&[f32]],     // each 30-dim
        entity_types: &[usize],  // 0/1/2 per entity
        threats: &[&[f32]],      // each 8-dim
    ) -> (Vec<f32>, Vec<bool>, usize) {
        let d = self.d_model;
        let n_ents = entities.len();
        let n_threats = threats.len();
        let n_total = n_ents + n_threats;

        let mut tokens = vec![0.0f32; n_total * d];
        let mut mask = vec![true; n_total]; // true = attend

        // Project entities
        for (i, (feats, &type_id)) in entities.iter().zip(entity_types).enumerate() {
            let exists = feats.len() >= 30 && feats[29] > 0.5;
            mask[i] = exists;

            self.entity_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);

            let tid = type_id.min(NUM_ENTITY_TYPES_V2 - 1);
            for j in 0..d {
                tokens[i * d + j] += self.type_emb[tid * d + j];
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Project threats (type_id = 3)
        for (ti, feats) in threats.iter().enumerate() {
            let i = n_ents + ti;
            let exists = feats.len() >= 8 && feats[7] > 0.5;
            mask[i] = exists;

            self.threat_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);

            for j in 0..d {
                tokens[i * d + j] += self.type_emb[3 * d + j]; // threat type
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Self-attention with reused scratch buffer
        let mut scratch = TransformerScratch::default();
        for layer in &self.self_attn_layers {
            layer.forward(&mut tokens, n_total, &mask, &mut scratch);
        }

        // Output norm
        for i in 0..n_total {
            self.out_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        (tokens, mask, n_total)
    }
}

// ---------------------------------------------------------------------------
// V3 entity encoder: variable-length entities + threats + positions
// ---------------------------------------------------------------------------

pub(super) const POSITION_DIM: usize = 8;
pub(super) const NUM_ENTITY_TYPES_V3: usize = 5; // self=0, enemy=1, ally=2, threat=3, position=4

#[derive(Debug, Deserialize)]
pub(super) struct EntityEncoderV3Json {
    pub entity_proj: LinearWeights,
    pub threat_proj: LinearWeights,
    pub position_proj: LinearWeights,
    pub type_emb: Vec<Vec<f32>>,  // [5 x d_model]
    pub input_norm: LayerNormWeights,
    #[serde(default)]
    pub self_attn_layers: Vec<TransformerLayerJson>,
    pub out_norm: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub(super) struct FlatEntityEncoderV3 {
    pub entity_proj: FlatLinear,     // (30 -> d_model)
    pub threat_proj: FlatLinear,     // (8 -> d_model)
    pub position_proj: FlatLinear,   // (8 -> d_model)
    pub type_emb: Vec<f32>,          // [5 x d_model]
    pub input_norm: FlatLayerNorm,
    pub self_attn_layers: Vec<TransformerLayer>,
    pub out_norm: FlatLayerNorm,
    pub d_model: usize,
}

impl FlatEntityEncoderV3 {
    pub fn from_json(ej: &EntityEncoderV3Json, d_model: usize, n_heads: usize) -> Self {
        let type_emb: Vec<f32> = ej.type_emb.iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let self_attn_layers: Vec<TransformerLayer> = ej.self_attn_layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, d_model, n_heads))
            .collect();
        Self {
            entity_proj: FlatLinear::from_json(&ej.entity_proj),
            threat_proj: FlatLinear::from_json(&ej.threat_proj),
            position_proj: FlatLinear::from_json(&ej.position_proj),
            type_emb,
            input_norm: FlatLayerNorm::from_json(&ej.input_norm),
            self_attn_layers,
            out_norm: FlatLayerNorm::from_json(&ej.out_norm),
            d_model,
        }
    }

    /// Encode variable-length entities + threats + positions into tokens.
    /// Returns (tokens [n_total x d_model], mask [n_total], n_total, entity_type_ids [n_total]).
    pub fn forward(
        &self,
        entities: &[&[f32]],     // each 30-dim
        entity_types: &[usize],  // 0/1/2 per entity
        threats: &[&[f32]],      // each 8-dim
        positions: &[&[f32]],    // each 8-dim
    ) -> (Vec<f32>, Vec<bool>, usize, Vec<usize>) {
        let d = self.d_model;
        let n_ents = entities.len();
        let n_threats = threats.len();
        let n_positions = positions.len();
        let n_total = n_ents + n_threats + n_positions;

        let mut tokens = vec![0.0f32; n_total * d];
        let mut mask = vec![true; n_total]; // true = attend
        let mut full_type_ids = vec![0usize; n_total];

        // Project entities
        for (i, (feats, &type_id)) in entities.iter().zip(entity_types).enumerate() {
            let exists = feats.len() >= 30 && feats[29] > 0.5;
            mask[i] = exists;
            full_type_ids[i] = type_id;

            self.entity_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
            let tid = type_id.min(NUM_ENTITY_TYPES_V3 - 1);
            for j in 0..d {
                tokens[i * d + j] += self.type_emb[tid * d + j];
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Project threats (type_id = 3)
        for (ti, feats) in threats.iter().enumerate() {
            let i = n_ents + ti;
            let exists = feats.len() >= THREAT_DIM && feats[7] > 0.5;
            mask[i] = exists;
            full_type_ids[i] = 3;

            self.threat_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
            for j in 0..d {
                tokens[i * d + j] += self.type_emb[3 * d + j];
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Project positions (type_id = 4)
        for (pi, feats) in positions.iter().enumerate() {
            let i = n_ents + n_threats + pi;
            mask[i] = true; // positions always exist (no padding in this direction)
            full_type_ids[i] = 4;

            self.position_proj.forward(feats, &mut tokens[i * d..(i + 1) * d]);
            for j in 0..d {
                tokens[i * d + j] += self.type_emb[4 * d + j];
            }
            self.input_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        // Self-attention
        let mut scratch = TransformerScratch::default();
        for layer in &self.self_attn_layers {
            layer.forward(&mut tokens, n_total, &mask, &mut scratch);
        }

        // Output norm
        for i in 0..n_total {
            self.out_norm.forward(&mut tokens[i * d..(i + 1) * d]);
        }

        (tokens, mask, n_total, full_type_ids)
    }
}
