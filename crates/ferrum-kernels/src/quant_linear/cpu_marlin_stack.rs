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

    fn zero_workspace(&self, _ctx: &mut <CpuBackend as crate::backend::Backend>::Context) -> Result<()> {
        // CPU GptqStore (dequant-on-load) has no per-expert workspace
        // mutex slots — Marlin-specific GPU artefact. No-op.
        Ok(())
    }

    fn gemm_phase_batched(
        &self,
        ctx: &mut <CpuBackend as crate::backend::Backend>::Context,
        input: &<CpuBackend as crate::backend::Backend>::Buffer,
        dispatches: &[(usize, usize, usize, usize)],
        output: &mut <CpuBackend as crate::backend::Backend>::Buffer,
        k: usize,
    ) -> Result<()> {
        // Phase C step 4c: inlined from the default
        // BackendQuantMarlin::moe_gemm_phase_batched impl — serial loop
        // calling gemm_gptq_with_offset_strided per active expert.
        for (expert_idx, in_row_offset, out_row_offset, m) in dispatches {
            <CpuBackend as crate::backend::BackendQuantMarlin>::gemm_gptq_with_offset_strided(
                ctx,
                input,
                *in_row_offset,
                &self.store,
                expert_idx * self.n_per_expert,
                self.n_per_expert,
                output,
                *out_row_offset,
                *m,
                k,
            )?;
        }
        Ok(())
    }

    fn make_expert_linear(
        self: Arc<Self>,
        expert_offset: usize,
        expert_n: usize,
        bias_host: Option<&[f32]>,
    ) -> Result<Box<dyn Linear<CpuBackend> + Send + Sync>> {
        // Inlined from BackendQuantMarlin::make_stacked_expert_linear
        // (Phase C step 4b).
        if expert_offset + expert_n > self.store.n {
            return Err(ferrum_types::FerrumError::model(format!(
                "make_expert_linear OOB: offset {expert_offset} + n {expert_n} > stacked_n {}",
                self.store.n
            )));
        }
        if self.k != self.store.k {
            return Err(ferrum_types::FerrumError::model(format!(
                "make_expert_linear k mismatch: arg {} vs store.k {}",
                self.k, self.store.k
            )));
        }
        let row_start = expert_offset * self.k;
        let row_end = (expert_offset + expert_n) * self.k;
        let slice = self.store.weight_f32[row_start..row_end].to_vec();
        Ok(Box::new(crate::quant_linear::cpu_dequant::CpuGptqLinear {
            weight_f32: slice,
            bias: bias_host.map(|b| b.to_vec()),
            in_features: self.k,
            out_features: expert_n,
        }))
    }
}
