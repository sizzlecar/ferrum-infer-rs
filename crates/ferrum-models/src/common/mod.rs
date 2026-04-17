//! Cross-model traits and helpers (Model-as-Code shared infrastructure).

pub mod llm;

pub use llm::{DecoderOnlyLLM, LlmRuntimeConfig};
