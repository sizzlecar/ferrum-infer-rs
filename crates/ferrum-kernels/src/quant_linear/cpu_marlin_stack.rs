//! `MarlinExpertStack<CpuBackend>` impl on top of CPU's dequant-on-load
//! GptqStore. Facade — delegates to the existing
//! `BackendQuantMarlin::moe_gemm_phase_*` (default trait impl that loops
//! calling `gemm_gptq_with_offset_strided` on CPU) and
//! `make_stacked_expert_linear` methods.
//!
//! CPU has no real Marlin tiles; the impl exists so the bucketed MoE
//! path's parity test (`tests/moe_bucketed_parity_test.rs`) still
//! compiles after the Phase C trait-object migration. Phase C step 4
//! inlines the kernel calls here and deletes the trait methods.

use crate::backend::cpu::CpuBackend;
use crate::marlin_expert_stack::MarlinExpertStack;
use crate::Linear;
use ferrum_types::Result;
use std::sync::Arc;

pub struct CpuMarlinExpertStack {
    pub store: Arc<<CpuBackend as crate::backend::Backend>::GptqStore>,
    pub num_experts: usize,
    pub n_per_expert: usize,
    pub k: usize,
}

impl CpuMarlinExpertStack {
    pub fn new(
        store: Arc<<CpuBackend as crate::backend::Backend>::GptqStore>,
        num_experts: usize,
        n_per_expert: usize,
        k: usize,
    ) -> Self {
        Self {
            store,
            num_experts,
            n_per_expert,
            k,
        }
    }
}

impl MarlinExpertStack<CpuBackend> for CpuMarlinExpertStack {
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

    fn zero_workspace(&self, ctx: &mut <CpuBackend as crate::backend::Backend>::Context) -> Result<()> {
        <CpuBackend as crate::backend::BackendQuantMarlin>::marlin_zero_stacked_workspace(
            ctx, &self.store,
        )
    }

    fn gemm_phase_batched(
        &self,
        ctx: &mut <CpuBackend as crate::backend::Backend>::Context,
        input: &<CpuBackend as crate::backend::Backend>::Buffer,
        dispatches: &[(usize, usize, usize, usize)],
        output: &mut <CpuBackend as crate::backend::Backend>::Buffer,
        k: usize,
    ) -> Result<()> {
        <CpuBackend as crate::backend::BackendQuantMarlin>::moe_gemm_phase_batched(
            ctx,
            input,
            &self.store,
            dispatches,
            self.n_per_expert,
            output,
            k,
        )
    }

    fn make_expert_linear(
        self: Arc<Self>,
        expert_offset: usize,
        expert_n: usize,
        bias_host: Option<&[f32]>,
    ) -> Result<Box<dyn Linear<CpuBackend> + Send + Sync>> {
        <CpuBackend as crate::backend::BackendQuantMarlin>::make_stacked_expert_linear(
            self.store.clone(),
            expert_offset,
            expert_n,
            self.k,
            bias_host,
        )
    }
}
