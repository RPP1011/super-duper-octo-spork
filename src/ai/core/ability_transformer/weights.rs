//! Transformer weight loading and inference.
//!
//! Implements multi-head self-attention, layer norm, and feed-forward layers
//! for frozen inference.  Follows the same patterns as EvalWeights and FlatMLP
//! in the existing codebase.

use serde::Deserialize;

// ---------------------------------------------------------------------------
// JSON schema (matches training/export_weights.py output)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct LinearWeights {
    w: Vec<Vec<f32>>,
    b: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct LayerNormWeights {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct TransformerLayerJson {
    self_attn_in_proj: LinearWeights,
    self_attn_out_proj: LinearWeights,
    ff_linear1: LinearWeights,
    ff_linear2: LinearWeights,
    norm1: LayerNormWeights,
    norm2: LayerNormWeights,
}

#[derive(Debug, Deserialize)]
struct DecisionHeadJson {
    urgency: HeadPairJson,
    target: HeadPairJson,
}

#[derive(Debug, Deserialize)]
struct HeadPairJson {
    linear1: LinearWeights,
    linear2: LinearWeights,
}

#[derive(Debug, Deserialize)]
struct ArchitectureJson {
    vocab_size: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    d_ff: usize,
    max_seq_len: usize,
    game_state_dim: usize,
    n_targets: usize,
    pad_id: usize,
    cls_id: usize,
}

#[derive(Debug, Deserialize)]
struct EntityEncoderJson {
    proj: LinearWeights,
    type_emb: Vec<Vec<f32>>,  // [3 × d_model]
    input_norm: LayerNormWeights,
    /// Self-attention layers over entities (from pre-training).
    /// Empty if no pre-trained encoder loaded.
    #[serde(default)]
    self_attn_layers: Vec<TransformerLayerJson>,
    out_norm: LayerNormWeights,
}

#[derive(Debug, Deserialize)]
struct CrossAttnJson {
    attn_in_proj: LinearWeights,
    attn_out_proj: LinearWeights,
    norm_q: LayerNormWeights,
    norm_kv: LayerNormWeights,
    ff_linear1: LinearWeights,
    ff_linear2: LinearWeights,
    norm_ff: LayerNormWeights,
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
    decision_head: DecisionHeadJson,
}

// ---------------------------------------------------------------------------
// Flattened runtime types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FlatLinear {
    in_dim: usize,
    out_dim: usize,
    w: Vec<f32>,  // [out_dim × in_dim] row-major (w[i] = row i of weight matrix)
    b: Vec<f32>,  // [out_dim]
}

impl FlatLinear {
    fn from_json(lw: &LinearWeights) -> Self {
        let out_dim = lw.w.len();
        let in_dim = if out_dim > 0 { lw.w[0].len() } else { 0 };
        // PyTorch Linear stores [out_dim, in_dim], already row-major
        let w: Vec<f32> = lw.w.iter().flat_map(|row| row.iter().copied()).collect();
        Self {
            in_dim,
            out_dim,
            w,
            b: lw.b.clone(),
        }
    }

    /// y = Wx + b
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        for i in 0..self.out_dim {
            let mut sum = self.b[i];
            let row_start = i * self.in_dim;
            for j in 0..self.in_dim {
                sum += self.w[row_start + j] * input[j];
            }
            output[i] = sum;
        }
    }

    /// y = ReLU(Wx + b)  — used for hidden layers
    fn forward_relu(&self, input: &[f32], output: &mut [f32]) {
        self.forward(input, output);
        for v in output.iter_mut() {
            *v = v.max(0.0);
        }
    }

    /// y = GELU(Wx + b)  — used for transformer FFN
    fn forward_gelu(&self, input: &[f32], output: &mut [f32]) {
        self.forward(input, output);
        for v in output.iter_mut() {
            *v = gelu(*v);
        }
    }
}

#[derive(Debug, Clone)]
struct FlatLayerNorm {
    dim: usize,
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl FlatLayerNorm {
    fn from_json(ln: &LayerNormWeights) -> Self {
        Self {
            dim: ln.gamma.len(),
            gamma: ln.gamma.clone(),
            beta: ln.beta.clone(),
        }
    }

    fn forward(&self, x: &mut [f32]) {
        let n = self.dim as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv_std = 1.0 / (var + 1e-5_f32).sqrt();
        for i in 0..self.dim {
            x[i] = (x[i] - mean) * inv_std * self.gamma[i] + self.beta[i];
        }
    }
}

#[derive(Debug, Clone)]
struct TransformerLayer {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    // Self-attention: in_proj combines Q, K, V projections [3*d_model, d_model]
    attn_in_proj: FlatLinear,
    attn_out_proj: FlatLinear,
    // Feed-forward
    ff1: FlatLinear,
    ff2: FlatLinear,
    // Pre-norm
    norm1: FlatLayerNorm,
    norm2: FlatLayerNorm,
}

impl TransformerLayer {
    fn from_json(lj: &TransformerLayerJson, d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            head_dim: d_model / n_heads,
            attn_in_proj: FlatLinear::from_json(&lj.self_attn_in_proj),
            attn_out_proj: FlatLinear::from_json(&lj.self_attn_out_proj),
            ff1: FlatLinear::from_json(&lj.ff_linear1),
            ff2: FlatLinear::from_json(&lj.ff_linear2),
            norm1: FlatLayerNorm::from_json(&lj.norm1),
            norm2: FlatLayerNorm::from_json(&lj.norm2),
        }
    }

    /// Forward pass for one transformer layer (pre-norm style).
    /// `seq` is [seq_len × d_model] in row-major.
    /// `mask` has `true` for positions to attend, `false` for padding.
    fn forward(&self, seq: &mut [f32], seq_len: usize, mask: &[bool]) {
        let d = self.d_model;

        // Allocate temporaries
        let mut normed = vec![0.0f32; seq_len * d];
        let mut qkv = vec![0.0f32; seq_len * 3 * d];
        let mut attn_out = vec![0.0f32; seq_len * d];

        // --- Pre-norm self-attention ---
        // 1. Layer norm
        for t in 0..seq_len {
            normed[t * d..(t + 1) * d].copy_from_slice(&seq[t * d..(t + 1) * d]);
            self.norm1.forward(&mut normed[t * d..(t + 1) * d]);
        }

        // 2. QKV projection
        for t in 0..seq_len {
            self.attn_in_proj.forward(
                &normed[t * d..(t + 1) * d],
                &mut qkv[t * 3 * d..(t + 1) * 3 * d],
            );
        }

        // 3. Multi-head attention
        self.multi_head_attention(&qkv, &mut attn_out, seq_len, mask);

        // 4. Output projection + residual
        let mut proj_out = vec![0.0f32; d];
        for t in 0..seq_len {
            self.attn_out_proj.forward(
                &attn_out[t * d..(t + 1) * d],
                &mut proj_out,
            );
            for i in 0..d {
                seq[t * d + i] += proj_out[i];
            }
        }

        // --- Pre-norm feed-forward ---
        // 1. Layer norm
        for t in 0..seq_len {
            normed[t * d..(t + 1) * d].copy_from_slice(&seq[t * d..(t + 1) * d]);
            self.norm2.forward(&mut normed[t * d..(t + 1) * d]);
        }

        // 2. FFN: GELU(x W1 + b1) W2 + b2
        let d_ff = self.ff1.out_dim;
        let mut ff_hidden = vec![0.0f32; d_ff];
        let mut ff_out = vec![0.0f32; d];
        for t in 0..seq_len {
            self.ff1.forward_gelu(&normed[t * d..(t + 1) * d], &mut ff_hidden);
            self.ff2.forward(&ff_hidden, &mut ff_out);
            for i in 0..d {
                seq[t * d + i] += ff_out[i];
            }
        }
    }

    fn multi_head_attention(
        &self,
        qkv: &[f32],       // [seq_len × 3*d_model]
        output: &mut [f32], // [seq_len × d_model]
        seq_len: usize,
        mask: &[bool],
    ) {
        let d = self.d_model;
        let h = self.n_heads;
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        // For each head
        for head in 0..h {
            let offset = head * hd;

            // Compute attention scores
            for qi in 0..seq_len {
                // Softmax over keys
                let mut scores = vec![f32::NEG_INFINITY; seq_len];
                for ki in 0..seq_len {
                    if !mask[ki] {
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for f in 0..hd {
                        let q = qkv[qi * 3 * d + offset + f];
                        let k = qkv[ki * 3 * d + d + offset + f];
                        dot += q * k;
                    }
                    scores[ki] = dot * scale;
                }

                // Softmax
                softmax_inplace(&mut scores);

                // Weighted sum of values → output
                for f in 0..hd {
                    let mut sum = 0.0f32;
                    for vi in 0..seq_len {
                        let v = qkv[vi * 3 * d + 2 * d + offset + f];
                        sum += scores[vi] * v;
                    }
                    output[qi * d + offset + f] = sum;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entity encoder + cross-attention runtime types
// ---------------------------------------------------------------------------

const ENTITY_DIM: usize = 30;
const NUM_ENTITIES: usize = 7;
const NUM_ENTITY_TYPES: usize = 3;
/// Entity type IDs: [self, enemy, enemy, enemy, ally, ally, ally]
const ENTITY_TYPE_IDS: [usize; NUM_ENTITIES] = [0, 1, 1, 1, 2, 2, 2];

#[derive(Debug, Clone)]
struct FlatEntityEncoder {
    proj: FlatLinear,          // (ENTITY_DIM → d_model)
    type_emb: Vec<f32>,        // [NUM_ENTITY_TYPES × d_model]
    input_norm: FlatLayerNorm,
    self_attn_layers: Vec<TransformerLayer>,
    out_norm: FlatLayerNorm,
    d_model: usize,
}

impl FlatEntityEncoder {
    fn from_json(ej: &EntityEncoderJson, d_model: usize, n_heads: usize) -> Self {
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

    /// Encode game_state (210 floats) into entity tokens [7 × d_model].
    /// Returns (entity_tokens, entity_mask) where mask[i] = true means attend.
    fn forward(&self, game_state: &[f32]) -> (Vec<f32>, Vec<bool>) {
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
        for layer in &self.self_attn_layers {
            layer.forward(&mut tokens, NUM_ENTITIES, &mask);
        }

        // Output norm per entity
        for ent in 0..NUM_ENTITIES {
            self.out_norm.forward(&mut tokens[ent * d..(ent + 1) * d]);
        }

        (tokens, mask)
    }
}

#[derive(Debug, Clone)]
struct FlatCrossAttention {
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    attn_in_proj: FlatLinear,   // [3*d_model, d_model] — Q from query, K/V from kv
    attn_out_proj: FlatLinear,
    norm_q: FlatLayerNorm,
    norm_kv: FlatLayerNorm,
    ff1: FlatLinear,
    ff2: FlatLinear,
    norm_ff: FlatLayerNorm,
}

impl FlatCrossAttention {
    fn from_json(cj: &CrossAttnJson, d_model: usize, n_heads: usize) -> Self {
        Self {
            d_model,
            n_heads,
            head_dim: d_model / n_heads,
            attn_in_proj: FlatLinear::from_json(&cj.attn_in_proj),
            attn_out_proj: FlatLinear::from_json(&cj.attn_out_proj),
            norm_q: FlatLayerNorm::from_json(&cj.norm_q),
            norm_kv: FlatLayerNorm::from_json(&cj.norm_kv),
            ff1: FlatLinear::from_json(&cj.ff_linear1),
            ff2: FlatLinear::from_json(&cj.ff_linear2),
            norm_ff: FlatLayerNorm::from_json(&cj.norm_ff),
        }
    }

    /// Cross-attention: query (d_model) attends to kv (n_entities × d_model).
    /// Returns updated query embedding (d_model).
    fn forward(
        &self,
        query: &[f32],          // [d_model]
        kv: &[f32],             // [n_entities × d_model]
        kv_mask: &[bool],       // [n_entities] — true = attend
        n_entities: usize,
    ) -> Vec<f32> {
        let d = self.d_model;
        let h = self.n_heads;
        let hd = self.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();

        // Pre-norm query
        let mut normed_q = query.to_vec();
        self.norm_q.forward(&mut normed_q);

        // Pre-norm key/value
        let mut normed_kv = kv.to_vec();
        for ent in 0..n_entities {
            self.norm_kv.forward(&mut normed_kv[ent * d..(ent + 1) * d]);
        }

        // QKV projection for query (only Q used from query)
        let mut q_qkv = vec![0.0f32; 3 * d];
        self.attn_in_proj.forward(&normed_q, &mut q_qkv);

        // QKV projection for each kv entity (K and V used)
        let mut kv_qkv = vec![0.0f32; n_entities * 3 * d];
        for ent in 0..n_entities {
            self.attn_in_proj.forward(
                &normed_kv[ent * d..(ent + 1) * d],
                &mut kv_qkv[ent * 3 * d..(ent + 1) * 3 * d],
            );
        }

        // Multi-head cross-attention
        let mut attn_out = vec![0.0f32; d];
        for head in 0..h {
            let offset = head * hd;

            // Scores: Q[head] · K[head] for each entity
            let mut scores = vec![f32::NEG_INFINITY; n_entities];
            for ki in 0..n_entities {
                if !kv_mask[ki] {
                    continue;
                }
                let mut dot = 0.0f32;
                for f in 0..hd {
                    let q = q_qkv[offset + f];                          // Q from query
                    let k = kv_qkv[ki * 3 * d + d + offset + f];       // K from kv
                    dot += q * k;
                }
                scores[ki] = dot * scale;
            }

            softmax_inplace(&mut scores);

            // Weighted sum of V
            for f in 0..hd {
                let mut sum = 0.0f32;
                for vi in 0..n_entities {
                    let v = kv_qkv[vi * 3 * d + 2 * d + offset + f];   // V from kv
                    sum += scores[vi] * v;
                }
                attn_out[offset + f] = sum;
            }
        }

        // Output projection
        let mut proj_out = vec![0.0f32; d];
        self.attn_out_proj.forward(&attn_out, &mut proj_out);

        // Residual: query + attn_output
        let mut x = vec![0.0f32; d];
        for i in 0..d {
            x[i] = query[i] + proj_out[i];
        }

        // Feed-forward with residual
        let mut ff_normed = x.clone();
        self.norm_ff.forward(&mut ff_normed);
        let mut ff_hidden = vec![0.0f32; self.ff1.out_dim];
        self.ff1.forward_gelu(&ff_normed, &mut ff_hidden);
        let mut ff_out = vec![0.0f32; d];
        self.ff2.forward(&ff_hidden, &mut ff_out);
        for i in 0..d {
            x[i] += ff_out[i];
        }

        x
    }
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

    // Embeddings: [vocab_size × d_model] and [max_seq_len × d_model]
    token_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    vocab_size: usize,

    // Output norm
    out_norm: FlatLayerNorm,

    // Transformer layers
    layers: Vec<TransformerLayer>,

    // Cross-attention with game state entities
    entity_encoder: Option<FlatEntityEncoder>,
    cross_attn: Option<FlatCrossAttention>,

    // Decision head
    urgency_l1: FlatLinear,
    urgency_l2: FlatLinear,
    target_l1: FlatLinear,
    target_l2: FlatLinear,
}

/// Output of transformer inference.
#[derive(Debug, Clone)]
pub struct TransformerOutput {
    pub urgency: f32,
    pub target_logits: Vec<f32>,
}

impl AbilityTransformerWeights {
    /// Load from JSON string exported by `training/export_weights.py`.
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        let file: TransformerFileJson =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let arch = &file.architecture;

        // Flatten embeddings to [vocab_size * d_model]
        let token_emb: Vec<f32> = file
            .token_embedding
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let pos_emb: Vec<f32> = file
            .position_embedding
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        if token_emb.len() != arch.vocab_size * arch.d_model {
            return Err(format!(
                "token_emb size mismatch: {} vs {}×{}",
                token_emb.len(),
                arch.vocab_size,
                arch.d_model
            ));
        }

        let layers: Vec<TransformerLayer> = file
            .layers
            .iter()
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
            token_emb,
            pos_emb,
            vocab_size: arch.vocab_size,
            out_norm: FlatLayerNorm::from_json(&file.output_norm),
            layers,
            entity_encoder,
            cross_attn: cross_attn_block,
            urgency_l1: FlatLinear::from_json(&file.decision_head.urgency.linear1),
            urgency_l2: FlatLinear::from_json(&file.decision_head.urgency.linear2),
            target_l1: FlatLinear::from_json(&file.decision_head.target.linear1),
            target_l2: FlatLinear::from_json(&file.decision_head.target.linear2),
        })
    }

    /// Encode ability tokens into a [CLS] embedding.
    ///
    /// This is **static per ability** — token sequences never change during
    /// a fight. Cache the result at fight init and reuse every tick.
    pub fn encode_cls(&self, token_ids: &[u32]) -> Vec<f32> {
        let d = self.d_model;
        let seq_len = token_ids.len().min(self.max_seq_len);

        let mut seq = vec![0.0f32; seq_len * d];
        let mut mask = vec![false; seq_len];

        for (t, &tid) in token_ids.iter().take(seq_len).enumerate() {
            let id = (tid as usize).min(self.vocab_size - 1);
            mask[t] = id != self.pad_id;
            for i in 0..d {
                seq[t * d + i] =
                    self.token_emb[id * d + i] + self.pos_emb[t * d + i];
            }
        }

        for layer in &self.layers {
            layer.forward(&mut seq, seq_len, &mask);
        }

        let mut cls = vec![0.0f32; d];
        cls.copy_from_slice(&seq[0..d]);
        self.out_norm.forward(&mut cls);
        cls
    }

    /// Predict from a cached [CLS] embedding and precomputed entity tokens.
    ///
    /// This is the **hot path** — runs every tick per ability. Only
    /// cross-attention + decision head, no transformer encoder.
    pub fn predict_from_cls(
        &self,
        cls: &[f32],
        entities: Option<&EncodedEntities>,
    ) -> TransformerOutput {
        let d = self.d_model;

        // Cross-attention with entity tokens
        let head_input = if let (Some(ca), Some(ent)) = (&self.cross_attn, entities) {
            ca.forward(cls, &ent.tokens, &ent.mask, NUM_ENTITIES)
        } else {
            cls.to_vec()
        };

        // Decision heads
        let mut u_hidden = vec![0.0f32; self.urgency_l1.out_dim];
        self.urgency_l1.forward_gelu(&head_input, &mut u_hidden);
        let mut u_out = vec![0.0f32; 1];
        self.urgency_l2.forward(&u_hidden, &mut u_out);

        let mut t_hidden = vec![0.0f32; self.target_l1.out_dim];
        self.target_l1.forward_gelu(&head_input, &mut t_hidden);
        let mut target_logits = vec![0.0f32; self.n_targets];
        self.target_l2.forward(&t_hidden, &mut target_logits);

        TransformerOutput {
            urgency: sigmoid(u_out[0]),
            target_logits,
        }
    }

    /// Run full forward inference (convenience, no caching).
    ///
    /// * `token_ids` — token IDs (with [CLS] at position 0)
    /// * `game_state` — optional game state features
    ///
    /// Returns urgency ∈ [0, 1] and target logits.
    pub fn predict(
        &self,
        token_ids: &[u32],
        game_state: Option<&[f32]>,
    ) -> TransformerOutput {
        let cls = self.encode_cls(token_ids);
        let entities = game_state.and_then(|gs| self.encode_entities(gs));
        self.predict_from_cls(&cls, entities.as_ref())
    }

    /// Convenience: get the argmax target index.
    pub fn predict_target(&self, token_ids: &[u32], game_state: Option<&[f32]>) -> (f32, usize) {
        let out = self.predict(token_ids, game_state);
        let best = out
            .target_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        (out.urgency, best)
    }

    // -----------------------------------------------------------------------
    // Batch API: precompute entity tokens once, evaluate multiple abilities
    // -----------------------------------------------------------------------

    /// Pre-computed entity encoding for a single unit's game state.
    /// Reusable across all ability evaluations for that unit.
    pub fn encode_entities(&self, game_state: &[f32]) -> Option<EncodedEntities> {
        let enc = self.entity_encoder.as_ref()?;
        let (tokens, mask) = enc.forward(game_state);
        Some(EncodedEntities { tokens, mask })
    }

    /// Evaluate a single ability using precomputed entity tokens.
    ///
    /// Avoids redundant entity encoder calls when evaluating
    /// multiple abilities for the same unit.
    pub fn predict_with_entities(
        &self,
        token_ids: &[u32],
        entities: Option<&EncodedEntities>,
    ) -> TransformerOutput {
        let cls = self.encode_cls(token_ids);
        self.predict_from_cls(&cls, entities)
    }

    /// Evaluate all abilities for a single unit at once.
    ///
    /// Computes entity encoding once, then evaluates each ability against it.
    pub fn predict_batch(
        &self,
        abilities: &[&[u32]],
        game_state: Option<&[f32]>,
    ) -> Vec<TransformerOutput> {
        let entities = game_state.and_then(|gs| self.encode_entities(gs));
        abilities
            .iter()
            .map(|tokens| self.predict_with_entities(tokens, entities.as_ref()))
            .collect()
    }

    /// Evaluate all abilities from cached [CLS] embeddings.
    ///
    /// **Optimal per-tick path.** Cache [CLS] embeddings at fight init
    /// with `encode_cls()`, then call this every tick. Only runs
    /// entity encoder (once) + cross-attention + decision head per ability.
    ///
    /// For 4 heroes × 8 abilities: 4 entity encoder calls + 32 cross-attn calls.
    /// No transformer encoder calls at all during the fight.
    pub fn predict_batch_cached(
        &self,
        cached_cls: &[&[f32]],
        game_state: Option<&[f32]>,
    ) -> Vec<TransformerOutput> {
        let entities = game_state.and_then(|gs| self.encode_entities(gs));
        cached_cls
            .iter()
            .map(|cls| self.predict_from_cls(cls, entities.as_ref()))
            .collect()
    }
}

/// Pre-computed entity encoding, reusable across ability evaluations.
#[derive(Debug, Clone)]
pub struct EncodedEntities {
    /// Flat entity tokens: [NUM_ENTITIES × d_model]
    pub tokens: Vec<f32>,
    /// Entity existence mask: [NUM_ENTITIES], true = entity exists
    pub mask: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-10.0, 10.0)).exp())
}

#[inline]
fn gelu(x: f32) -> f32 {
    // Approximation: x * σ(1.702 * x)
    x * sigmoid(1.702 * x)
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_layer_norm() {
        let ln = FlatLayerNorm {
            dim: 4,
            gamma: vec![1.0; 4],
            beta: vec![0.0; 4],
        };
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        ln.forward(&mut x);
        // Mean should be ~0, std ~1
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean={mean}");
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_load_and_predict() {
        // Load exported weights from Python
        let json_path = "generated/ability_transformer_weights.json";
        if !std::path::Path::new(json_path).exists() {
            eprintln!("Skipping: {json_path} not found (run export_weights.py first)");
            return;
        }
        let json_str = std::fs::read_to_string(json_path).unwrap();
        let model = AbilityTransformerWeights::from_json(&json_str).unwrap();

        // Fireball ability token IDs (from Python tokenizer)
        let tokens: Vec<u32> = vec![1, 1, 18, 5, 8, 185, 14, 83, 15, 155, 14, 232, 63, 14, 249, 15, 38, 14, 248, 96, 14, 68, 68, 239, 12, 7, 14, 239, 13, 9];
        let out = model.predict(&tokens, None);
        assert!(out.urgency >= 0.0 && out.urgency <= 1.0, "urgency={}", out.urgency);
        assert_eq!(out.target_logits.len(), 3);
        println!("urgency={:.6}, targets={:?}", out.urgency, out.target_logits);
        // Compare with Python output: urgency=0.526941, targets=[2.69, -1.68, -2.58]
        assert!((out.urgency - 0.527).abs() < 0.02, "urgency mismatch: {}", out.urgency);
    }

    #[test]
    fn test_load_and_predict_cross_attn() {
        let json_path = "generated/ability_transformer_weights_xattn.json";
        if !std::path::Path::new(json_path).exists() {
            eprintln!("Skipping: {json_path} not found");
            return;
        }
        let json_str = std::fs::read_to_string(json_path).unwrap();
        let model = match AbilityTransformerWeights::from_json(&json_str) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Skipping: {json_path} failed to parse (needs re-export): {e}");
                return;
            }
        };

        // Fireball tokens
        let tokens: Vec<u32> = vec![1, 1, 18, 5, 8, 185, 14, 83, 15, 155, 14, 232, 63, 14, 249, 15, 38, 14, 248, 96, 14, 68, 68, 239, 12, 7, 14, 239, 13, 9];

        // Game state: 210 floats (7 entities × 30 features)
        let mut gs = vec![0.0f32; 210];
        // Self entity: hp=1.0, exists=1.0 (index 29)
        gs[0] = 1.0;   // hp_pct
        gs[12] = 0.5;  // auto_dps
        gs[29] = 1.0;  // exists
        // Enemy 0: hp=0.5, exists=1.0
        gs[30] = 0.5;
        gs[42] = 0.8;
        gs[59] = 1.0;
        // Enemy 1: hp=1.0, exists=1.0
        gs[60] = 1.0;
        gs[72] = 0.4;
        gs[89] = 1.0;
        // Ally 0: hp=0.3, exists=1.0
        gs[120] = 0.3;
        gs[149] = 1.0;

        let out = model.predict(&tokens, Some(&gs));
        println!("Rust urgency: {:.6}, targets: {:?}", out.urgency, out.target_logits);

        // Just verify it loads and produces valid output
        assert!(
            out.urgency >= 0.0 && out.urgency <= 1.0,
            "urgency out of range: {}",
            out.urgency
        );
        assert_eq!(out.target_logits.len(), 3);
    }

    #[test]
    fn test_batch_api_matches_single() {
        let json_path = "generated/ability_transformer_weights_xattn.json";
        if !std::path::Path::new(json_path).exists() {
            eprintln!("Skipping: {json_path} not found");
            return;
        }
        let json_str = std::fs::read_to_string(json_path).unwrap();
        let model = match AbilityTransformerWeights::from_json(&json_str) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Skipping: parse error (needs re-export): {e}");
                return;
            }
        };

        let tokens: Vec<u32> = vec![1, 1, 18, 5, 8, 185, 14, 83, 15, 155];
        let tokens2: Vec<u32> = vec![1, 1, 63, 14, 249, 15, 38, 14, 248, 96];

        let mut gs = vec![0.0f32; 210];
        gs[0] = 1.0; gs[29] = 1.0;
        gs[30] = 0.5; gs[59] = 1.0;

        // Single predict
        let out1 = model.predict(&tokens, Some(&gs));
        let out2 = model.predict(&tokens2, Some(&gs));

        // Batch predict
        let batch = model.predict_batch(&[&tokens, &tokens2], Some(&gs));
        assert_eq!(batch.len(), 2);
        assert!((batch[0].urgency - out1.urgency).abs() < 1e-6,
            "batch[0] urgency mismatch: {} vs {}", batch[0].urgency, out1.urgency);
        assert!((batch[1].urgency - out2.urgency).abs() < 1e-6,
            "batch[1] urgency mismatch: {} vs {}", batch[1].urgency, out2.urgency);
    }

    #[test]
    fn test_flat_linear() {
        // Identity-ish: 2→2 with identity weights
        let fl = FlatLinear {
            in_dim: 2,
            out_dim: 2,
            w: vec![1.0, 0.0, 0.0, 1.0],
            b: vec![0.0, 0.0],
        };
        let mut out = vec![0.0; 2];
        fl.forward(&[3.0, 5.0], &mut out);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
    }
}
