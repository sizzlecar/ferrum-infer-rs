//! Model-as-Code implementations.
//!
//! Each module defines one model family (Qwen3, Llama, ...) as explicit
//! Rust code: structs for weights + `forward` methods using the `Backend`
//! trait and `Linear` trait directly. This replaces the earlier
//! "generic ModelRunner + TransformerConfig" approach, which could not
//! express MoE / MLA / multimodal / quantization cleanly.

pub mod qwen3;

pub use qwen3::{Qwen3Config, Qwen3Model};
