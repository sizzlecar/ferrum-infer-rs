//! `Qwen3MoeLayer` — bundles the three pieces a single MoE layer needs
//! (router linear, expert weight stack, top-K configuration) into a
//! struct with one ergonomic `forward()` method. Drop-in replacement for
//! a dense MLP layer in the wider transformer body.
//!
//! Phase 2 ships a CPU-only forward via [`moe_forward_cpu`]. Generic
//! `Backend<B>` support is deferred (see `dispatch.rs` for why).

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::Backend;
use ferrum_kernels::Linear;
use ferrum_quantization::gguf::{GgufFile, GgufLinear};
use ferrum_types::{FerrumError, Result};

use crate::moe::dispatch::{moe_forward_cpu, ExpertStack};
use crate::moe::router::route;
use crate::moe_config::Qwen3MoeConfig;

/// Per-layer Qwen3-MoE state: router + expert weights + config knobs.
///
/// Construct with [`Self::load_from_gguf`] (or its convenience wrappers).
/// Call [`Self::forward_cpu`] to run one MoE layer's forward pass.
pub struct Qwen3MoeLayer<B: Backend> {
    /// Gating linear: `[hidden] → [num_experts]` per token.
    pub router: Box<dyn Linear<B>>,
    /// Per-expert MLP weights.
    pub experts: ExpertStack<B>,
    /// Number of experts to activate per token.
    pub top_k: usize,
    /// Whether to renormalise the K selected probs to sum to 1.
    pub norm_topk_prob: bool,
    /// Hidden size (= `router.in_features()`, kept here to avoid pointer
    /// chasing in tight loops).
    pub hidden_size: usize,
    /// Per-expert FFN inner size (= `experts.gate_up[e].out_features() / 2`).
    pub expert_intermediate: usize,
    /// Total expert count (= `experts.num_experts()`).
    pub num_experts: usize,
}

impl<B: Backend> Qwen3MoeLayer<B> {
    /// Load both router and expert weights for layer `layer_idx` from a
    /// GGUF file. Convenience wrapper around the lower-level GGUF reader
    /// + `ExpertStack::load_from_gguf` + manual router construction.
    pub fn load_from_gguf(gguf: &GgufFile, layer_idx: usize, cfg: &Qwen3MoeConfig) -> Result<Self> {
        // Router lives at `blk.{i}.ffn_gate_inp.weight` — 2-D, fits the
        // standard Linear path. Build via GgufLinear directly to avoid
        // pulling a full WeightLoader into the dependency surface here.
        let router_name = format!("blk.{layer_idx}.ffn_gate_inp.weight");
        if !gguf.has_tensor(&router_name) {
            return Err(FerrumError::model(format!(
                "Qwen3MoeLayer: router tensor '{router_name}' not in GGUF"
            )));
        }
        let router_qt = gguf
            .read_tensor(&router_name, &candle_core::Device::Cpu)
            .map_err(|e| FerrumError::model(format!("read router: {e}")))?;
        let router = GgufLinear::<B>::from_qtensor(&router_qt)
            .map_err(|e| FerrumError::model(format!("router from_qtensor: {e}")))?;
        let router: Box<dyn Linear<B>> = Box::new(router);

        // Expert weight stack — three 3-D tensors.
        let experts = ExpertStack::<B>::load_from_gguf(
            gguf,
            layer_idx,
            cfg.num_experts,
            cfg.base.hidden_size,
            cfg.expert_intermediate_size,
        )?;

        // Sanity: dimensions should agree with the config.
        if router.in_features() != cfg.base.hidden_size {
            return Err(FerrumError::model(format!(
                "router in_features {} != hidden_size {}",
                router.in_features(),
                cfg.base.hidden_size
            )));
        }
        if router.out_features() != cfg.num_experts {
            return Err(FerrumError::model(format!(
                "router out_features {} != num_experts {}",
                router.out_features(),
                cfg.num_experts
            )));
        }

        Ok(Self {
            router,
            experts,
            top_k: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            hidden_size: cfg.base.hidden_size,
            expert_intermediate: cfg.expert_intermediate_size,
            num_experts: cfg.num_experts,
        })
    }
}

impl Qwen3MoeLayer<CpuBackend> {
    /// Run one MoE layer's forward pass on CPU.
    ///
    /// `x`: `[batch, hidden_size]` — typically the hidden state after the
    /// post-attention RMSNorm in the surrounding transformer block.
    /// `out`: same shape as `x`. Resized + zero-initialised.
    ///
    /// Internally:
    ///   1. Run `router.forward(x)` → router_logits `[batch, num_experts]`
    ///   2. Call `route(...)` to pick top-K and weights per token.
    ///   3. Call `moe_forward_cpu(...)` for the per-expert MLP loop.
    pub fn forward_cpu(&self, x: &[f32], batch: usize, out: &mut Vec<f32>) -> Result<()> {
        if x.len() != batch * self.hidden_size {
            return Err(FerrumError::model(format!(
                "Qwen3MoeLayer::forward_cpu: x len {} != batch*hidden = {}*{} = {}",
                x.len(),
                batch,
                self.hidden_size,
                batch * self.hidden_size
            )));
        }

        // Step 1: router logits
        let mut router_logits: Vec<f32> = vec![0.0; batch * self.num_experts];
        let mut ctx = <CpuBackend as Backend>::new_context();
        let x_buf: Vec<f32> = x.to_vec();
        self.router
            .forward(&mut ctx, &x_buf, &mut router_logits, batch);

        // Step 2: top-K + softmax + (optional) renorm
        let router_out = route(
            &router_logits,
            batch,
            self.num_experts,
            self.top_k,
            self.norm_topk_prob,
        );

        // Step 3: per-token, per-expert MLP dispatch and weighted combine
        moe_forward_cpu(
            x,
            batch,
            self.hidden_size,
            self.expert_intermediate,
            self.top_k,
            &router_out,
            &self.experts,
            out,
        )
    }
}
