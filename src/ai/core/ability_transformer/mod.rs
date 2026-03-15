//! Frozen transformer inference for ability evaluation.
//!
//! Loads JSON weights exported from the Python training pipeline
//! (`training/export_weights.py`) and runs forward inference.
//!
//! Architecture: 2-layer, 4-head, d_model=64 transformer encoder with
//! [CLS] pooling -> decision head (urgency + target).

mod weights;
mod weights_math;
mod weights_base;
mod weights_encoder;
mod weights_actor_critic;
mod weights_actor_critic_v3;
mod weights_actor_critic_v4;
mod weights_actor_critic_v5;
mod tokenizer_vocab;
pub mod tokenizer;
pub mod gpu_client;

pub use weights::AC_NUM_ACTIONS;
pub use weights_base::{AbilityTransformerWeights, EmbeddingRegistry};
pub use weights_actor_critic::{ActorCriticWeights, ActorCriticWeightsV2, EntityState};
pub use weights_actor_critic_v3::{ActorCriticWeightsV3, EntityStateV3, PointerOutput};
pub use weights_actor_critic_v4::{ActorCriticWeightsV4, DualHeadOutput};
pub use weights_actor_critic_v5::ActorCriticWeightsV5;
