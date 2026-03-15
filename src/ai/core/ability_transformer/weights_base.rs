//! Base transformer weights for ability evaluation (AbilityTransformerWeights).

use serde::Deserialize;
use super::weights::{
    sigmoid,
    FlatLinear, FlatLayerNorm, TransformerLayer, TransformerScratch,
    FlatCrossAttention, CrossAttnJson,
    LinearWeights, LayerNormWeights, TransformerLayerJson,
    NUM_ENTITIES,
    DecisionHeadJson,
};
use super::weights_encoder::{FlatEntityEncoder, EntityEncoderJson};

// ---------------------------------------------------------------------------
// JSON schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ArchitectureJson {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    d_ff: usize,
    max_seq_len: usize,
    #[serde(default)]
    game_state_dim: usize,
    #[serde(default)]
    n_targets: usize,
    pad_id: usize,
    cls_id: usize,
}

#[derive(Debug, Deserialize)]
struct TransformerFileJson {
    architecture: ArchitectureJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder: Option<EntityEncoderJson>,
    cross_attn: Option<CrossAttnJson>,
    decision_head: Option<DecisionHeadJson>,
}

// ---------------------------------------------------------------------------
// Top-level inference struct
// ---------------------------------------------------------------------------

/// Frozen ability transformer for inference.
#[derive(Debug, Clone)]
pub struct AbilityTransformerWeights {
    d_model: usize,
    n_heads: usize,
    max_seq_len: usize,
    pad_id: usize,
    _cls_id: usize,
    n_targets: usize,

    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    vocab_size: usize,

    out_norm: FlatLayerNorm,
    layers: Vec<TransformerLayer>,

    entity_encoder: Option<FlatEntityEncoder>,
    cross_attn: Option<FlatCrossAttention>,

    urgency_l1: Option<FlatLinear>,
    urgency_l2: Option<FlatLinear>,
    target_l1: Option<FlatLinear>,
    target_l2: Option<FlatLinear>,
}

/// Output of transformer inference.
#[derive(Debug, Clone)]
pub struct TransformerOutput {
    pub urgency: f32,
    pub target_logits: Vec<f32>,
}

/// Pre-computed entity encoding, reusable across ability evaluations.
#[derive(Debug, Clone)]
pub struct EncodedEntities {
    pub tokens: Vec<f32>,
    pub mask: Vec<bool>,
}

impl AbilityTransformerWeights {
    /// Load from JSON string exported by `training/export_weights.py`.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: TransformerFileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        let token_emb: Vec<f32> = file.token_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();
        let pos_emb: Vec<f32> = file.position_embedding.iter()
            .flat_map(|row| row.iter().copied()).collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err(format!(
                "token_emb size mismatch: {} vs {}x{}",
                token_emb.len(), arch.vocab_size, arch.d_model
            ));
        }

        let layers: Vec<TransformerLayer> = file.layers.iter()
            .map(|lj| TransformerLayer::from_json(lj, arch.d_model, arch.n_heads))
            .collect();

        let entity_encoder = file.entity_encoder.as_ref()
            .map(|ej| FlatEntityEncoder::from_json(ej, arch.d_model, arch.n_heads));
        let cross_attn_block = file.cross_attn.as_ref()
            .map(|cj| FlatCrossAttention::from_json(cj, arch.d_model, arch.n_heads));

        Ok(Self {
            d_model: arch.d_model,
            n_heads: arch.n_heads,
            max_seq_len: arch.max_seq_len,
            pad_id: arch.pad_id,
            _cls_id: arch.cls_id,
            n_targets: arch.n_targets,
            token_emb, pos_emb,
            vocab_size: arch.vocab_size,
            out_norm: FlatLayerNorm::from_json(&file.output_norm),
            layers,
            entity_encoder,
            cross_attn: cross_attn_block,
            urgency_l1: file.decision_head.as_ref().map(|dh| FlatLinear::from_json(&dh.urgency.linear1)),
            urgency_l2: file.decision_head.as_ref().map(|dh| FlatLinear::from_json(&dh.urgency.linear2)),
            target_l1: file.decision_head.as_ref().map(|dh| FlatLinear::from_json(&dh.target.linear1)),
            target_l2: file.decision_head.as_ref().map(|dh| FlatLinear::from_json(&dh.target.linear2)),
        })
    }

    /// Encode ability tokens into a [CLS] embedding.
    pub fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.d_model;
        let seq_len = token_ids.len().min(self.max_seq_len);
        let mut seq = vec![0.0f32; seq_len * d];
        let mut mask = vec![false; seq_len];
        for (t, &tid) in token_ids.iter().take(seq_len).enumerate() {
            let id = (tid as usize).min(self.vocab_size - 1);
            mask[t] = id != self.pad_id;
            for i in 0..d { seq[t * d + i] = self.token_emb[id * d + i] + self.pos_emb[t * d + i]; }
        }
        let mut scratch = TransformerScratch::default();
        for layer in &self.layers { layer.forward(&mut seq, seq_len, &mask, &mut scratch); }
        let mut cls = vec![0.0f32; d];
        cls.copy_from_slice(&seq[0..d]);
        self.out_norm.forward(&mut cls);
        cls
    }

    /// Predict from a cached [CLS] embedding and precomputed entity tokens.
    pub fn predict_from_cls(&self, cls: &[f32], entities: Option<&EncodedEntities>) -> TransformerOutput {
        let head_input = if let (Some(ca), Some(ent)) = (&self.cross_attn, entities) {
            ca.forward(cls, &ent.tokens, &ent.mask, NUM_ENTITIES)
        } else { cls.to_vec() };

        let ul1 = self.urgency_l1.as_ref().expect("predict_from_cls requires Phase 2 weights");
        let ul2 = self.urgency_l2.as_ref().unwrap();
        let tl1 = self.target_l1.as_ref().unwrap();
        let tl2 = self.target_l2.as_ref().unwrap();

        let mut u_hidden = vec![0.0f32; ul1.out_dim];
        ul1.forward_gelu(&head_input, &mut u_hidden);
        let mut u_out = vec![0.0f32; 1];
        ul2.forward(&u_hidden, &mut u_out);

        let mut t_hidden = vec![0.0f32; tl1.out_dim];
        tl1.forward_gelu(&head_input, &mut t_hidden);
        let mut target_logits = vec![0.0f32; self.n_targets];
        tl2.forward(&t_hidden, &mut target_logits);

        TransformerOutput { urgency: sigmoid(u_out[0]), target_logits }
    }

    /// Run full forward inference (convenience, no caching).
    pub fn predict(&self, token_ids: &[u32], game_state: Option<&[f32]>) -> TransformerOutput {
        let cls = self.encode_cls(token_ids);
        let entities = game_state.and_then(|gs| self.encode_entities(gs));
        self.predict_from_cls(&cls, entities.as_ref())
    }

    /// Convenience: get the argmax target index.
    pub fn predict_target(&self, token_ids: &[u32], game_state: Option<&[f32]>) -> (f32, usize) {
        let out = self.predict(token_ids, game_state);
        let best = out.target_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i).unwrap_or(0);
        (out.urgency, best)
    }

    /// Pre-computed entity encoding for a single unit's game state.
    pub fn encode_entities(&self, game_state: &[f32]) -> Option<EncodedEntities> {
        let enc = self.entity_encoder.as_ref()?;
        let (tokens, mask) = enc.forward(game_state);
        Some(EncodedEntities { tokens, mask })
    }

    /// Evaluate a single ability using precomputed entity tokens.
    pub fn predict_with_entities(&self, token_ids: &[u32], entities: Option<&EncodedEntities>) -> TransformerOutput {
        let cls = self.encode_cls(token_ids);
        self.predict_from_cls(&cls, entities)
    }

    /// Evaluate all abilities for a single unit at once.
    pub fn predict_batch(&self, abilities: &[&[u32]], game_state: Option<&[f32]>) -> Vec<TransformerOutput> {
        let entities = game_state.and_then(|gs| self.encode_entities(gs));
        abilities.iter().map(|tokens| self.predict_with_entities(tokens, entities.as_ref())).collect()
    }

    /// Evaluate all abilities from cached [CLS] embeddings.
    pub fn predict_batch_cached(&self, cached_cls: &[&[f32]], game_state: Option<&[f32]>) -> Vec<TransformerOutput> {
        let entities = game_state.and_then(|gs| self.encode_entities(gs));
        cached_cls.iter().map(|cls| self.predict_from_cls(cls, entities.as_ref())).collect()
    }
}

// ---------------------------------------------------------------------------
// Embedding Registry
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct EmbeddingRegistryJson {
    model_hash: String,
    d_model: usize,
    n_abilities: usize,
    embeddings: std::collections::HashMap<String, Vec<f32>>,
    #[serde(default)]
    outcome_mean: Vec<f32>,
    #[serde(default)]
    outcome_std: Vec<f32>,
}

/// Pre-computed CLS embeddings for known abilities.
#[derive(Debug, Clone)]
pub struct EmbeddingRegistry {
    pub model_hash: String,
    pub d_model: usize,
    embeddings: std::collections::HashMap<String, Vec<f32>>,
}

impl EmbeddingRegistry {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: EmbeddingRegistryJson =
            serde_json::from_str(json_str).map_err(|e| format!("Registry parse error: {e}"))?;
        assert_eq!(file.embeddings.len(), file.n_abilities, "n_abilities mismatch");
        Ok(Self { model_hash: file.model_hash, d_model: file.d_model, embeddings: file.embeddings })
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("Failed to read registry: {e}"))?;
        Self::from_json(&data)
    }

    pub fn get(&self, ability_name: &str) -> Option<&[f32]> {
        self.embeddings.get(ability_name).map(|v| v.as_slice())
    }

    pub fn len(&self) -> usize { self.embeddings.len() }
}
