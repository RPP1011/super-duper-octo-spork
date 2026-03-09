//! Serializable attention recording format for 3D replay.
//!
//! Captures sequences of attention weight snapshots from inference or training,
//! serialized to compact binary files (`.attn`) that can be loaded into the
//! Bevy 3D attention replay viewer.
//!
//! # Format
//!
//! An `AttentionRecording` contains:
//! - Model architecture metadata (d_model, n_heads, n_layers)
//! - A sequence of `AttentionFrame`s, each capturing per-layer, per-head
//!   attention weights plus optional cross-attention and decision outputs
//! - Token/entity labels for visualization
//!
//! # Usage
//!
//! ```ignore
//! let mut recorder = AttentionRecorder::new(64, 4, 2);
//! recorder.set_token_labels(&["[CLS]", "DAMAGE", "FIRE", "..."]);
//!
//! // During training/inference:
//! recorder.push_frame(AttentionFrame { ... });
//!
//! // Save to disk:
//! recorder.save("run_001.attn")?;
//!
//! // Load for replay:
//! let recording = AttentionRecording::load("run_001.attn")?;
//! ```

use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single layer's attention weights for all heads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAttention {
    /// Layer index in the encoder stack.
    pub layer: usize,
    /// Flattened attention weights: `[n_heads * query_len * key_len]`.
    /// Stored flat for compact serialization; reshape as `[n_heads][q][k]`.
    pub weights_flat: Vec<f32>,
    /// Query sequence length.
    pub query_len: usize,
    /// Key sequence length.
    pub key_len: usize,
}

impl LayerAttention {
    /// Create from nested `[n_heads][query_len][key_len]` attention weights.
    pub fn from_nested(layer: usize, weights: &[Vec<Vec<f32>>]) -> Self {
        let n_heads = weights.len();
        let query_len = weights.first().map_or(0, |h| h.len());
        let key_len = weights
            .first()
            .and_then(|h| h.first())
            .map_or(0, |r| r.len());
        let mut flat = Vec::with_capacity(n_heads * query_len * key_len);
        for head in weights {
            for row in head {
                flat.extend_from_slice(row);
            }
        }
        Self {
            layer,
            weights_flat: flat,
            query_len,
            key_len,
        }
    }

    /// Get attention weight for a specific head, query, key position.
    pub fn get(&self, n_heads: usize, head: usize, q: usize, k: usize) -> f32 {
        debug_assert!(head < n_heads && q < self.query_len && k < self.key_len);
        let idx = head * self.query_len * self.key_len + q * self.key_len + k;
        self.weights_flat[idx]
    }

    /// Iterate over all weights for a given head as `[query_len][key_len]`.
    pub fn head_matrix(&self, n_heads: usize, head: usize) -> &[f32] {
        let stride = self.query_len * self.key_len;
        debug_assert!(head < n_heads);
        let start = head * stride;
        &self.weights_flat[start..start + stride]
    }
}

/// Cross-attention weights from a single ability CLS → entity tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAttentionFrame {
    /// Ability slot index (0-7).
    pub ability_slot: usize,
    /// Flattened weights: `[n_heads * key_len]`.
    /// Query is always the single CLS token.
    pub weights_flat: Vec<f32>,
    /// Number of entity/threat/position tokens attended to.
    pub key_len: usize,
}

/// Decision outputs at a single frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionSnapshot {
    /// Action type logits `[n_action_types]`.
    pub action_type_logits: Vec<f32>,
    /// Attack pointer logits `[n_tokens]`.
    pub attack_pointer: Vec<f32>,
    /// Move pointer logits `[n_tokens]`.
    pub move_pointer: Vec<f32>,
    /// Per-ability pointer logits (None if ability not present).
    pub ability_pointers: Vec<Option<Vec<f32>>>,
    /// Value head estimate.
    pub value: f32,
}

/// A single snapshot of attention state, captured during one forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFrame {
    /// Training step or game tick number.
    pub step: u64,
    /// Optional timestamp in seconds since recording start.
    pub timestamp_secs: f64,

    /// Ability transformer self-attention per layer.
    pub transformer_attention: Vec<LayerAttention>,
    /// Entity encoder self-attention per layer.
    pub entity_attention: Vec<LayerAttention>,
    /// Cross-attention per active ability slot.
    pub cross_attention: Vec<CrossAttentionFrame>,

    /// Decision outputs (if available).
    pub decision: Option<DecisionSnapshot>,

    /// CLS embedding `[d_model]` (for embedding drift tracking).
    pub cls_embedding: Option<Vec<f32>>,
    /// Pooled entity state `[d_model]`.
    pub pooled_state: Option<Vec<f32>>,
}

/// Complete attention recording with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionRecording {
    /// Model dimensionality.
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of entity encoder layers.
    pub n_entity_layers: usize,

    /// Token labels for the ability transformer sequence.
    pub token_labels: Vec<String>,
    /// Entity/threat/position labels.
    pub entity_labels: Vec<String>,
    /// Entity type IDs (0=self, 1=enemy, 2=ally, 3=threat, 4=position).
    pub entity_type_ids: Vec<usize>,

    /// Recorded frames, ordered by step.
    pub frames: Vec<AttentionFrame>,

    /// Optional description / run name.
    pub description: String,
}

impl AttentionRecording {
    /// Save recording to a binary file.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let data = serde_json::to_vec(self).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("serialize error: {e}"))
        })?;
        // Write with a magic header for identification
        let mut out = Vec::with_capacity(8 + data.len());
        out.extend_from_slice(b"ATTN\x01\x00\x00\x00"); // magic + version
        out.extend_from_slice(&data);
        std::fs::write(path, &out)
    }

    /// Load recording from a binary file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 8 || &bytes[0..4] != b"ATTN" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a valid .attn file (bad magic)",
            ));
        }
        let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported .attn version: {version}"),
            ));
        }
        serde_json::from_slice(&bytes[8..]).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("deserialize error: {e}"))
        })
    }

    /// Total number of frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Recorder — builder for recording during training/inference
// ---------------------------------------------------------------------------

/// Incrementally builds an `AttentionRecording` during training or inference.
pub struct AttentionRecorder {
    recording: AttentionRecording,
    start_time: std::time::Instant,
}

impl AttentionRecorder {
    pub fn new(d_model: usize, n_heads: usize, n_layers: usize, n_entity_layers: usize) -> Self {
        Self {
            recording: AttentionRecording {
                d_model,
                n_heads,
                n_layers,
                n_entity_layers,
                token_labels: Vec::new(),
                entity_labels: Vec::new(),
                entity_type_ids: Vec::new(),
                frames: Vec::new(),
                description: String::new(),
            },
            start_time: std::time::Instant::now(),
        }
    }

    pub fn set_description(&mut self, desc: impl Into<String>) {
        self.recording.description = desc.into();
    }

    pub fn set_token_labels(&mut self, labels: &[&str]) {
        self.recording.token_labels = labels.iter().map(|s| s.to_string()).collect();
    }

    pub fn set_entity_labels(&mut self, labels: &[String], type_ids: &[usize]) {
        self.recording.entity_labels = labels.to_vec();
        self.recording.entity_type_ids = type_ids.to_vec();
    }

    /// Record a frame with automatic timestamp.
    pub fn push_frame(&mut self, mut frame: AttentionFrame) {
        frame.timestamp_secs = self.start_time.elapsed().as_secs_f64();
        self.recording.frames.push(frame);
    }

    /// Finalize and return the recording.
    pub fn finish(self) -> AttentionRecording {
        self.recording
    }

    /// Save current recording to disk (can continue recording after).
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.recording.save(path)
    }
}

// ---------------------------------------------------------------------------
// Conversion from DiagnosticCapture
// ---------------------------------------------------------------------------

impl super::diagnostics::DiagnosticCapture {
    /// Convert a single diagnostic capture into an `AttentionFrame`.
    pub fn to_attention_frame(&self, step: u64) -> AttentionFrame {
        let transformer_attention = self
            .transformer_attention
            .iter()
            .map(|ac| LayerAttention::from_nested(ac.layer, &ac.weights))
            .collect();

        let entity_attention = self
            .entity_attention
            .iter()
            .map(|ac| LayerAttention::from_nested(ac.layer, &ac.weights))
            .collect();

        let cross_attention = self
            .cross_attention
            .iter()
            .map(|ca| {
                let key_len = ca.weights.first().map_or(0, |h| h.len());
                let mut flat = Vec::new();
                for head in &ca.weights {
                    flat.extend_from_slice(head);
                }
                CrossAttentionFrame {
                    ability_slot: ca.ability_slot,
                    weights_flat: flat,
                    key_len,
                }
            })
            .collect();

        let decision = Some(DecisionSnapshot {
            action_type_logits: self.action_type_logits.clone(),
            attack_pointer: self.attack_pointer.clone(),
            move_pointer: self.move_pointer.clone(),
            ability_pointers: self.ability_pointers.clone(),
            value: self.value,
        });

        AttentionFrame {
            step,
            timestamp_secs: 0.0,
            transformer_attention,
            entity_attention,
            cross_attention,
            decision,
            cls_embedding: Some(self.cls_embedding.clone()),
            pooled_state: Some(self.pooled_state.clone()),
        }
    }
}
