//! V3 pointer-based actor-critic weights and inference.

use serde::Deserialize;
use super::weights::{
    dot_product,
    FlatLinear, FlatLayerNorm, TransformerLayer, TransformerScratch,
    FlatCrossAttention, CrossAttnJson,
    LinearWeights, LayerNormWeights, TransformerLayerJson,
    AC_MAX_ABILITIES, NUM_ACTION_TYPES,
    HeadJson, ActorCriticArchJson,
};
use super::weights_encoder::{FlatEntityEncoderV3, EntityEncoderV3Json};

// ---------------------------------------------------------------------------
// Pointer head
// ---------------------------------------------------------------------------

/// Pointer output from V3 action head.
#[derive(Debug)]
pub struct PointerOutput {
    /// Action type logits (11 elements).
    pub type_logits: Vec<f32>,
    /// Attack pointer logits over all tokens.
    pub attack_ptr: Vec<f32>,
    /// Move pointer logits over all tokens.
    pub move_ptr: Vec<f32>,
    /// Per-ability pointer logits (None if ability slot empty).
    pub ability_ptrs: Vec<Option<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct PointerHeadJson {
    action_type_head: HeadJson,
    pointer_key: LinearWeights,
    attack_query: LinearWeights,
    move_query: LinearWeights,
    ability_queries: Vec<LinearWeights>,
}

#[derive(Debug, Clone)]
struct FlatPointerHead {
    action_type_l1: FlatLinear,
    action_type_l2: FlatLinear,
    pointer_key: FlatLinear,
    attack_query: FlatLinear,
    move_query: FlatLinear,
    ability_queries: Vec<FlatLinear>,
    d_model: usize,
    scale: f32,
}

impl FlatPointerHead {
    fn from_json(pj: &PointerHeadJson, d_model: usize) -> Self {
        Self {
            action_type_l1: FlatLinear::from_json(&pj.action_type_head.linear1),
            action_type_l2: FlatLinear::from_json(&pj.action_type_head.linear2),
            pointer_key: FlatLinear::from_json(&pj.pointer_key),
            attack_query: FlatLinear::from_json(&pj.attack_query),
            move_query: FlatLinear::from_json(&pj.move_query),
            ability_queries: pj.ability_queries.iter()
                .map(|q| FlatLinear::from_json(q))
                .collect(),
            d_model,
            scale: (d_model as f32).powf(-0.5),
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
    ) -> PointerOutput {
        let d = self.d_model;

        // Action type logits
        let mut type_hidden = vec![0.0f32; self.action_type_l1.out_dim];
        self.action_type_l1.forward_gelu(pooled, &mut type_hidden);
        let mut type_logits = vec![0.0f32; NUM_ACTION_TYPES];
        self.action_type_l2.forward(&type_hidden, &mut type_logits);

        // Compute keys for all tokens
        let mut keys = vec![0.0f32; n_total * d];
        for i in 0..n_total {
            self.pointer_key.forward(
                &tokens[i * d..(i + 1) * d],
                &mut keys[i * d..(i + 1) * d],
            );
        }

        // Attack pointer: Q*K^T, masked to enemies only
        let mut atk_query = vec![0.0f32; d];
        self.attack_query.forward(pooled, &mut atk_query);
        let mut attack_ptr = vec![f32::NEG_INFINITY; n_total];
        for i in 0..n_total {
            if mask[i] && type_ids[i] == 1 {
                attack_ptr[i] = dot_product(&atk_query, &keys[i * d..(i + 1) * d]) * self.scale;
            }
        }

        // Move pointer: Q*K^T, masked to non-self tokens
        let mut mv_query = vec![0.0f32; d];
        self.move_query.forward(pooled, &mut mv_query);
        let mut move_ptr = vec![f32::NEG_INFINITY; n_total];
        for i in 0..n_total {
            if mask[i] && type_ids[i] != 0 {
                move_ptr[i] = dot_product(&mv_query, &keys[i * d..(i + 1) * d]) * self.scale;
            }
        }

        // Ability pointers
        let mut ability_ptrs = Vec::with_capacity(AC_MAX_ABILITIES);
        for (idx, cross_emb_opt) in ability_cross_embs.iter().enumerate().take(AC_MAX_ABILITIES) {
            if let Some(cross_emb) = cross_emb_opt {
                if idx < self.ability_queries.len() {
                    let mut ab_query = vec![0.0f32; d];
                    self.ability_queries[idx].forward(cross_emb, &mut ab_query);
                    let mut ab_ptr = vec![f32::NEG_INFINITY; n_total];
                    for i in 0..n_total {
                        if mask[i] {
                            ab_ptr[i] = dot_product(&ab_query, &keys[i * d..(i + 1) * d]) * self.scale;
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

        PointerOutput { type_logits, attack_ptr, move_ptr, ability_ptrs }
    }
}

// ---------------------------------------------------------------------------
// V3 actor-critic JSON schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ActorCriticV3FileJson {
    #[allow(dead_code)]
    format: Option<String>,
    architecture: ActorCriticArchJson,
    token_embedding: Vec<Vec<f32>>,
    position_embedding: Vec<Vec<f32>>,
    output_norm: LayerNormWeights,
    layers: Vec<TransformerLayerJson>,
    entity_encoder_v3: EntityEncoderV3Json,
    cross_attn: CrossAttnJson,
    pointer_head: PointerHeadJson,
    /// Optional projection for external CLS embeddings (e.g. 128->32 behavioral).
    external_cls_proj: Option<LinearWeights>,
}

/// V3 entity state: tokens + mask + pooled + type IDs.
#[derive(Debug)]
pub struct EntityStateV3 {
    pub tokens: Vec<f32>,
    pub mask: Vec<bool>,
    pub pooled: Vec<f32>,
    pub type_ids: Vec<usize>,
    pub n_total: usize,
}

/// V3 actor-critic weights with pointer-based action space.
#[derive(Debug, Clone)]
pub struct ActorCriticWeightsV3 {
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
    pointer_head: FlatPointerHead,
    // Optional projection for external CLS (e.g. 128-dim behavioral -> d_model)
    external_cls_proj: Option<FlatLinear>,
}

impl ActorCriticWeightsV3 {
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: ActorCriticV3FileJson =
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
        let pointer_head = FlatPointerHead::from_json(
            &file.pointer_head, arch.d_model,
        );

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
            pointer_head,
            external_cls_proj: file.external_cls_proj.as_ref().map(FlatLinear::from_json),
        })
    }

    /// Project external CLS embedding (e.g. 128-dim behavioral) to d_model.
    pub fn project_external_cls(&self, cls: &[f32]) -> Vec<f32> {
        if let Some(proj) = &self.external_cls_proj {
            let mut out = vec![0.0f32; self.d_model];
            proj.forward(cls, &mut out);
            out
        } else {
            cls.to_vec()
        }
    }

    /// Encode ability token IDs to [CLS] embedding.
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

    /// Encode v3 game state -> entity state with variable-length tokens + positions.
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

    /// Compute pointer action logits using v3 entity state.
    pub fn pointer_logits(
        &self,
        entity_state: &EntityStateV3,
        ability_cls: &[Option<&[f32]>],
    ) -> PointerOutput {
        let n_total = entity_state.n_total;

        // Cross-attend each ability CLS to entity tokens
        // NOTE: CLS is already projected to d_model at cache time (project_external_cls
        // is called in transformer_rl.rs during CLS caching), so no projection here.
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

        self.pointer_head.forward(
            &entity_state.pooled,
            &entity_state.tokens,
            &entity_state.mask,
            n_total,
            &entity_state.type_ids,
            &ability_cross_embs,
        )
    }
}
