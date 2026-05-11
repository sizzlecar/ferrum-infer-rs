//! `MarlinExpertStack<CudaBackend>` impl on top of the existing
//! `GptqStoreCuda` opaque store.
//!
//! Phase C step 2: pure facade — every method delegates to the
//! corresponding `Backend::moe_gemm_phase_*` / `make_stacked_expert_linear`
//! / `marlin_zero_stacked_workspace` method that's still on the
//! `Backend` trait. Step 3 migrates `ExpertStack` to hold a
//! `Box<dyn MarlinExpertStack<B>>` and drops the direct `B::moe_gemm_*`
//! calls. Step 4 inlines the kernel bodies into this impl and deletes
//! the Backend methods.
//!
//! Holding `Arc<GptqStoreCuda>` keeps the existing shared-ownership
//! semantics (gate_up + per-expert column-slice views all share the
//! same repacked Marlin tile).

#![cfg(feature = "cuda")]

use crate::backend::cuda::{CudaBackend, GptqStoreCuda};
use crate::marlin_expert_stack::MarlinExpertStack;
use crate::Linear;
use ferrum_types::Result;
use std::sync::Arc;

pub struct CudaMarlinExpertStack {
    pub store: Arc<GptqStoreCuda>,
    pub num_experts: usize,
    pub n_per_expert: usize,
    pub k: usize,
}

impl CudaMarlinExpertStack {
    pub fn new(store: Arc<GptqStoreCuda>, num_experts: usize, n_per_expert: usize, k: usize) -> Self {
        Self {
            store,
            num_experts,
            n_per_expert,
            k,
        }
    }
}

impl MarlinExpertStack<CudaBackend> for CudaMarlinExpertStack {
    fn n_per_expert(&self) -> usize {
        self.n_per_expert
    }
    fn k(&self) -> usize {
        self.k
    }
    fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn zero_workspace(&self, ctx: &mut <CudaBackend as crate::backend::Backend>::Context) -> Result<()> {
        <CudaBackend as crate::backend::BackendQuantMarlin>::marlin_zero_stacked_workspace(
            ctx, &self.store,
        )
    }

    fn gemm_phase_batched(
        &self,
        ctx: &mut <CudaBackend as crate::backend::Backend>::Context,
        input: &<CudaBackend as crate::backend::Backend>::Buffer,
        dispatches: &[(usize, usize, usize, usize)],
        output: &mut <CudaBackend as crate::backend::Backend>::Buffer,
        k: usize,
    ) -> Result<()> {
        #[cfg(feature = "marlin")]
        {
            <CudaBackend as crate::backend::BackendQuantMarlin>::moe_gemm_phase_batched(
                ctx,
                input,
                &self.store,
                dispatches,
                self.n_per_expert,
                output,
                k,
            )
        }
        #[cfg(not(feature = "marlin"))]
        {
            let _ = (ctx, input, dispatches, output, k);
            Err(ferrum_types::FerrumError::unsupported(
                "moe_gemm_phase_batched: cargo feature `marlin` disabled",
            ))
        }
    }

    fn gemm_phase_vllm(
        &self,
        ctx: &mut <CudaBackend as crate::backend::Backend>::Context,
        input: &<CudaBackend as crate::backend::Backend>::Buffer,
        sorted_token_ids: &<CudaBackend as crate::backend::Backend>::Buffer,
        expert_ids: &<CudaBackend as crate::backend::Backend>::Buffer,
        num_tokens_past_padded: &<CudaBackend as crate::backend::Backend>::Buffer,
        output: &mut <CudaBackend as crate::backend::Backend>::Buffer,
        prob_m: usize,
        moe_block_size: usize,
        top_k: usize,
    ) -> Result<()> {
        <CudaBackend as crate::backend::BackendQuantMarlin>::moe_gemm_phase_vllm(
            ctx,
            input,
            &self.store,
            sorted_token_ids,
            expert_ids,
            num_tokens_past_padded,
            output,
            prob_m,
            self.n_per_expert,
            self.k,
            moe_block_size,
            top_k,
        )
    }

    fn make_expert_linear(
        self: Arc<Self>,
        expert_offset: usize,
        expert_n: usize,
        bias_host: Option<&[f32]>,
    ) -> Result<Box<dyn Linear<CudaBackend> + Send + Sync>> {
        <CudaBackend as crate::backend::BackendQuantMarlin>::make_stacked_expert_linear(
            self.store.clone(),
            expert_offset,
            expert_n,
            self.k,
            bias_host,
        )
    }
}
