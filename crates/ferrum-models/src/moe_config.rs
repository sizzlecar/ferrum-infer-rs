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

    /// Build from a parsed config.json (`ModelDefinition`). Same role as
    /// `LlamaFamilyConfig::qwen3_from_def` but pulls the MoE-specific
    /// fields (`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`,
    /// `norm_topk_prob`) from `def.extra_params`.
    pub fn from_def(def: &crate::definition::ModelDefinition) -> ferrum_types::Result<Self> {
        // Underlying Llama-family base — Qwen3-MoE shares the dense
        // attention path (QK-norm on, rope_theta from checkpoint).
        let mut base = crate::models::llama_family::LlamaFamilyConfig::qwen3_from_def(def);
        // For MoE the per-layer FFN size is the *expert* size, not the
        // dense intermediate. Stash the expert size in base.intermediate_size
        // so any code reading base sees the right value (Qwen3MoeModel
        // primarily reads `cfg.expert_intermediate_size` directly).
        let extra = &def.extra_params;
        let num_experts = extra
            .get("num_experts")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                ferrum_types::FerrumError::model("qwen3_moe config.json missing num_experts")
            })? as usize;
        let num_experts_per_tok = extra
            .get("num_experts_per_tok")
            .and_then(|v| v.as_u64())
            .unwrap_or(8) as usize;
        let expert_intermediate_size = extra
            .get("moe_intermediate_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                ferrum_types::FerrumError::model(
                    "qwen3_moe config.json missing moe_intermediate_size",
                )
            })? as usize;
        let norm_topk_prob = extra
            .get("norm_topk_prob")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        base.intermediate_size = expert_intermediate_size;
        Ok(Self {
            base,
            num_experts,
            num_experts_per_tok,
            expert_intermediate_size,
            norm_topk_prob,
        })
    }
}
