//! Cross-model traits and helpers (Model-as-Code shared infrastructure).

pub mod decoder_unified;
pub mod llm;
pub mod paged_pool;

pub use llm::{DecoderOnlyLLM, LlmRuntimeConfig};
