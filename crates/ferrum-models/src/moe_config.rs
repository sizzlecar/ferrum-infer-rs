//! Mixture-of-Experts (MoE) configuration types.
//!
//! Phase 2A: data types only. The runtime (router + expert dispatch) lands
//! in Phase 2B/2C/2D. This file's job is to give the project a single
//! unambiguous representation of MoE hyperparameters so subsequent PRs
//! can wire it into the model code, the loader, and the benchmark suite
//! without each making up their own shape.
//!
//! ## Design choice: composition, not inheritance
//!
//! [`Qwen3MoeConfig`] **wraps** [`LlamaFamilyConfig`] rather than adding
//! `Option<MoeConfig>` fields to it. Reasons:
//!
//! 1. Every existing dense call site (Qwen3 / Llama / Mistral / TinyLlama
//!    via `*_from_def`) keeps working unchanged — no `..Default::default()`
//!    breakage, no "MoE field always present even for dense" awkwardness.
//! 2. The MoE forward path is structurally different (per-token router,
//!    per-token expert subset, weighted sum) — it'll live in a separate
//!    `Qwen3MoeModel<B>` rather than branching inside `LlamaFamilyModel`.
//!    Sharing a config type would force the two models to coevolve.
//! 3. `Qwen3MoeConfig::base` reuses every field that genuinely is the
//!    same (hidden_size, num_layers, attention dims, RoPE, vocab) so we
//!    aren't duplicating dense fields.
//!
//! Trade-off: callers that just want "either dense or MoE config" will
//! need an `enum`. We'll add that wrapper if/when it earns its keep.

use crate::models::llama_family::LlamaFamilyConfig;

/// Configuration for Qwen3-MoE family models (Qwen3-30B-A3B and friends).
///
/// All MoE-specific hyperparameters live here; dense fields are inherited
/// via [`Qwen3MoeConfig::base`]. The `base.intermediate_size` is set to
/// [`Self::expert_intermediate_size`] for compatibility — Qwen3-MoE has
/// no shared dense FFN, every layer is MoE.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3MoeConfig {
    /// Shared transformer hyperparameters (attention, RoPE, vocab, …).
    pub base: LlamaFamilyConfig,

    /// Total number of experts per MoE layer. Qwen3-30B-A3B = 128.
    pub num_experts: usize,

    /// Top-K experts activated per token. Qwen3-30B-A3B = 8.
    pub num_experts_per_tok: usize,

    /// Per-expert FFN inner size. Distinct from `base.intermediate_size`
    /// (which is the *shared* FFN size for dense layers; for MoE we
    /// duplicate this value into base for downstream sanity but the real
    /// number lives here). Qwen3-30B-A3B per-expert = 768.
    pub expert_intermediate_size: usize,

    /// Whether the router output is normalised across the top-K experts
    /// (softmax over selected logits) before the weighted combine.
    /// Qwen3-MoE: true. Mixtral: also true. Older variants: false.
    pub norm_topk_prob: bool,
}

impl Qwen3MoeConfig {
    /// Construct from an already-built `LlamaFamilyConfig` plus the MoE
    /// hyperparameters. Mostly useful for tests and synthetic scenarios;
    /// real models go through [`Self::from_gguf`] in `gguf_config`.
    pub fn from_base(
        base: LlamaFamilyConfig,
        num_experts: usize,
        num_experts_per_tok: usize,
        expert_intermediate_size: usize,
        norm_topk_prob: bool,
    ) -> Self {
        Self {
            base,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            norm_topk_prob,
        }
    }

    /// True iff the router is configured to activate `< num_experts` experts
    /// per token. (Sanity guard — a config with `top_k == num_experts` is
    /// equivalent to a dense layer with extra dispatch cost.)
    pub fn is_truly_sparse(&self) -> bool {
        self.num_experts_per_tok > 0 && self.num_experts_per_tok < self.num_experts
    }
}
