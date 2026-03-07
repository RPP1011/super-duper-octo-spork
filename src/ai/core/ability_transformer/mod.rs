//! Frozen transformer inference for ability evaluation.
//!
//! Loads JSON weights exported from the Python training pipeline
//! (`training/export_weights.py`) and runs forward inference.
//!
//! Architecture: 2-layer, 4-head, d_model=64 transformer encoder with
//! [CLS] pooling → decision head (urgency + target).

mod weights;

pub use weights::AbilityTransformerWeights;
