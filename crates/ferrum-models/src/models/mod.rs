//! Model-as-Code implementations.
//!
//! Each module defines one model family as explicit Rust code: structs for
//! weights + `forward` methods using the `Backend` trait and `Linear` trait
//! directly. This replaces the earlier "generic ModelRunner +
//! TransformerConfig" approach, which could not express MoE / MLA /
//! multimodal / quantization cleanly.
//!
//! Current coverage:
//!   - `llama_family` — Llama / Llama-2 / Llama-3 / Qwen2 / Qwen2.5 / Qwen3
//!                      (standard GQA + SwiGLU + RoPE, optional QK-norm).
//!
//! Planned (Phase D):
//!   - `mistral`     — sliding-window attention variant.
//!   - `deepseek_v3` — MLA compressed KV + MoE expert routing.
//!   - `qwen_vl`     — ViT backbone + LLM (multimodal).

pub mod llama_family;

pub use llama_family::{LlamaFamilyConfig, LlamaFamilyModel};
