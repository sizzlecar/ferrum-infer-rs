//! `StackedExpertGgufLinear<MetalBackend>` impl — wraps a Metal
//! `Q4KExperts` / `Q6KExperts` `MetalQuantStore` and dispatches via the
//! existing `dispatch_gemv_moe_id*` / `dispatch_gemm_moe_id*` /
//! `dispatch_gemv_moe_id_gate_up_silu*` free functions in
//! `crate::backend::metal`.
//!
//! Phase 3e/4: replaces the `BackendQuantGguf::gemv_quant_moe_id*` /
//! `gemm_quant_moe_id*` trait methods + the `Backend::QuantStore`
//! associated type. Model code (ExpertStack<B>) now holds
//! `Box<dyn StackedExpertGgufLinear<B>>` instead of `B::QuantStore`,
//! so adding a new k-quant flavour doesn't touch the Backend trait.

use std::any::Any;

use crate::backend::metal::{
    dispatch_gemv_moe_id, dispatch_gemv_moe_id_offset, st, MetalBackend, MetalQuantStore,
};
use crate::stacked_expert::StackedExpertGgufLinear;
use ferrum_types::{FerrumError, Result};

/// Metal MoE-stacked GGUF Linear. Internally holds a `MetalQuantStore`
/// that MUST be `Q4KExperts` or `Q6KExperts` (constructor enforces).
/// Plain enum wrapper keeps the existing dispatcher signatures usable.
pub struct MetalStackedExpertGgufLinear {
    pub store: MetalQuantStore,
}

impl MetalStackedExpertGgufLinear {
    /// Wrap a `MetalQuantStore` that's been built by `load_q4k_experts`
    /// or `load_q6k_experts`. Returns Err if any other variant is passed.
    pub fn new(store: MetalQuantStore) -> Result<Self> {
        match &store {
            MetalQuantStore::Q4KExperts { .. } | MetalQuantStore::Q6KExperts { .. } => {
                Ok(Self { store })
            }
            _ => Err(FerrumError::model(
                "MetalStackedExpertGgufLinear requires Q4KExperts or Q6KExperts variant"
                    .to_string(),
            )),
        }
    }

    /// Borrow the inner `MetalQuantStore` — exposed for fused methods that
    /// need to dispatch on both gate and up stores together.
    pub fn store(&self) -> &MetalQuantStore {
        &self.store
    }

    fn dims(&self) -> (usize, usize, usize) {
        match &self.store {
            MetalQuantStore::Q4KExperts {
                num_experts,
                n_rows,
                n_cols,
                ..
            }
            | MetalQuantStore::Q6KExperts {
                num_experts,
                n_rows,
                n_cols,
                ..
            } => (*num_experts, *n_rows, *n_cols),
            _ => unreachable!("constructor guarantees experts variant"),
        }
    }

    /// Downcast helper for cross-impl fused methods.
    fn from_dyn<'a>(
        other: &'a dyn StackedExpertGgufLinear<MetalBackend>,
        ctx: &'static str,
    ) -> Result<&'a Self> {
        other.as_any().downcast_ref::<Self>().ok_or_else(|| {
            FerrumError::model(format!(
                "{ctx}: expected MetalStackedExpertGgufLinear (got a different StackedExpertGgufLinear<MetalBackend> impl)"
            ))
        })
    }
}

impl StackedExpertGgufLinear<MetalBackend> for MetalStackedExpertGgufLinear {
    fn num_experts(&self) -> usize {
        self.dims().0
    }
    fn n_rows(&self) -> usize {
        self.dims().1
    }
    fn n_cols(&self) -> usize {
        self.dims().2
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn gemv_moe_id(
        &self,
        ctx: &mut <MetalBackend as crate::backend::Backend>::Context,
        a: &<MetalBackend as crate::backend::Backend>::Buffer,
        ids: &<MetalBackend as crate::backend::Backend>::Buffer,
        out: &mut <MetalBackend as crate::backend::Backend>::Buffer,
        n_selected: usize,
        src1_stride: usize,
    ) -> Result<()> {
        let a_buf = a.expect_f32("gemv_moe_id a");
        let ids_buf = &ids.raw;
        let out_buf = out.expect_f32_mut("gemv_moe_id out");
        let enc = ctx.compute_encoder();
        dispatch_gemv_moe_id(
            enc,
            a_buf,
            &self.store,
            ids_buf,
            out_buf,
            n_selected,
            src1_stride,
        )
    }

    fn gemv_moe_id_offset(
        &self,
        ctx: &mut <MetalBackend as crate::backend::Backend>::Context,
        a: &<MetalBackend as crate::backend::Backend>::Buffer,
        a_offset: usize,
        ids: &<MetalBackend as crate::backend::Backend>::Buffer,
        ids_offset: usize,
        out: &mut <MetalBackend as crate::backend::Backend>::Buffer,
        n_selected: usize,
        src1_stride: usize,
    ) -> Result<()> {
        let a_buf = a.expect_f32("gemv_moe_id_offset a");
        let ids_buf = &ids.raw;
        let out_buf = out.expect_f32_mut("gemv_moe_id_offset out");
        let enc = ctx.compute_encoder();
        let a_byte_offset = (a_offset * std::mem::size_of::<f32>()) as u64;
        let ids_byte_offset = (ids_offset * std::mem::size_of::<i32>()) as u64;
        dispatch_gemv_moe_id_offset(
            enc,
            a_buf,
            a_byte_offset,
            &self.store,
            ids_buf,
            ids_byte_offset,
            out_buf,
            n_selected,
            src1_stride,
        )
    }

    fn gemv_moe_id_gate_up_silu(
        &self,
        ctx: &mut <MetalBackend as crate::backend::Backend>::Context,
        a: &<MetalBackend as crate::backend::Backend>::Buffer,
        other_up: &dyn StackedExpertGgufLinear<MetalBackend>,
        ids: &<MetalBackend as crate::backend::Backend>::Buffer,
        silu_out: &mut <MetalBackend as crate::backend::Backend>::Buffer,
        n_selected: usize,
    ) -> Result<()> {
        let up = Self::from_dyn(other_up, "gemv_moe_id_gate_up_silu")?;
        let (gate_blocks, gate_byte_offset, gate_n_rows, gate_n_cols) = match &self.store {
            MetalQuantStore::Q4KExperts {
                blocks,
                byte_offset,
                n_rows,
                n_cols,
                ..
            } => (blocks, *byte_offset, *n_rows, *n_cols),
            _ => {
                return Err(FerrumError::model(
                    "gemv_moe_id_gate_up_silu: gate weight must be Q4KExperts".to_string(),
                ));
            }
        };
        let (up_blocks, up_byte_offset, up_n_rows, up_n_cols) = match &up.store {
            MetalQuantStore::Q4KExperts {
                blocks,
                byte_offset,
                n_rows,
                n_cols,
                ..
            } => (blocks, *byte_offset, *n_rows, *n_cols),
            _ => {
                return Err(FerrumError::model(
                    "gemv_moe_id_gate_up_silu: up weight must be Q4KExperts".to_string(),
                ));
            }
        };
        if gate_n_rows != up_n_rows || gate_n_cols != up_n_cols {
            return Err(FerrumError::model(format!(
                "gemv_moe_id_gate_up_silu: gate/up shape mismatch — \
                 gate=({gate_n_rows}, {gate_n_cols}) up=({up_n_rows}, {up_n_cols})"
            )));
        }

        let a_buf = a.expect_f32("gemv_moe_id_gate_up_silu a");
        let ids_buf = &ids.raw;
        let out_buf = silu_out.expect_f32_mut("gemv_moe_id_gate_up_silu silu_out");
        let enc = ctx.compute_encoder();
        crate::q4_k_moe_id_gate_up_silu::dispatch_gemv_q4k_moe_id_gate_up_silu_on_encoder(
            &st().pipes.device,
            enc,
            a_buf,
            gate_blocks,
            gate_byte_offset,
            up_blocks,
            up_byte_offset,
            ids_buf,
            out_buf,
            gate_n_rows,
            gate_n_cols,
            n_selected,
        );
        Ok(())
    }

    fn gemv_moe_id_batched(
        &self,
        ctx: &mut <MetalBackend as crate::backend::Backend>::Context,
        a: &<MetalBackend as crate::backend::Backend>::Buffer,
        ids: &<MetalBackend as crate::backend::Backend>::Buffer,
        out: &mut <MetalBackend as crate::backend::Backend>::Buffer,
        m: usize,
        top_k: usize,
        src1_outer_stride: usize,
        src1_inner_stride: usize,
    ) -> Result<()> {
        let a_buf = a.expect_f32("gemv_moe_id_batched a");
        let ids_buf = &ids.raw;
        let out_buf = out.expect_f32_mut("gemv_moe_id_batched out");
        let enc = ctx.compute_encoder();
        match &self.store {
            MetalQuantStore::Q4KExperts {
                blocks,
                byte_offset,
                n_rows,
                n_cols,
                ..
            } => {
                crate::q4_k_moe_id_gemv_batched::dispatch_gemv_q4k_moe_id_batched_on_encoder(
                    &st().pipes.device,
                    enc,
                    a_buf,
                    blocks,
                    *byte_offset,
                    ids_buf,
                    out_buf,
                    *n_rows,
                    *n_cols,
                    m,
                    top_k,
                    src1_outer_stride,
                    src1_inner_stride,
                );
                Ok(())
            }
            MetalQuantStore::Q6KExperts {
                blocks,
                byte_offset,
                n_rows,
                n_cols,
                ..
            } => {
                crate::q6_k_moe_id_gemv_batched::dispatch_gemv_q6k_moe_id_batched_on_encoder(
                    &st().pipes.device,
                    enc,
                    a_buf,
                    blocks,
                    *byte_offset,
                    ids_buf,
                    out_buf,
                    *n_rows,
                    *n_cols,
                    m,
                    top_k,
                    src1_outer_stride,
                    src1_inner_stride,
                );
                Ok(())
            }
            _ => unreachable!("constructor guarantees experts variant"),
        }
    }

    fn gemv_moe_id_gate_up_silu_batched(
        &self,
        ctx: &mut <MetalBackend as crate::backend::Backend>::Context,
        a: &<MetalBackend as crate::backend::Backend>::Buffer,
        other_up: &dyn StackedExpertGgufLinear<MetalBackend>,
        ids: &<MetalBackend as crate::backend::Backend>::Buffer,
        silu_out: &mut <MetalBackend as crate::backend::Backend>::Buffer,
        m: usize,
        top_k: usize,
        src1_outer_stride: usize,
        src1_inner_stride: usize,
    ) -> Result<()> {
        let up = Self::from_dyn(other_up, "gemv_moe_id_gate_up_silu_batched")?;
        let (gate_blocks, gate_byte_offset, gate_n_rows, gate_n_cols) = match &self.store {
            MetalQuantStore::Q4KExperts {
                blocks,
                byte_offset,
                n_rows,
                n_cols,
                ..
            } => (blocks, *byte_offset, *n_rows, *n_cols),
            _ => {
                return Err(FerrumError::model(
                    "gemv_moe_id_gate_up_silu_batched: gate must be Q4KExperts".to_string(),
                ));
            }
        };
        let (up_blocks, up_byte_offset, up_n_rows, up_n_cols) = match &up.store {
            MetalQuantStore::Q4KExperts {
                blocks,
                byte_offset,
                n_rows,
                n_cols,
                ..
            } => (blocks, *byte_offset, *n_rows, *n_cols),
            _ => {
                return Err(FerrumError::model(
                    "gemv_moe_id_gate_up_silu_batched: up must be Q4KExperts".to_string(),
                ));
            }
        };
        if gate_n_rows != up_n_rows || gate_n_cols != up_n_cols {
            return Err(FerrumError::model(format!(
                "gemv_moe_id_gate_up_silu_batched: gate/up shape mismatch — \
                 gate=({gate_n_rows}, {gate_n_cols}) up=({up_n_rows}, {up_n_cols})"
            )));
        }
        let a_buf = a.expect_f32("gemv_moe_id_gate_up_silu_batched a");
        let ids_buf = &ids.raw;
        let out_buf = silu_out.expect_f32_mut("gemv_moe_id_gate_up_silu_batched silu_out");
        let enc = ctx.compute_encoder();
        crate::q4_k_moe_id_gate_up_silu_batched::dispatch_gemv_q4k_moe_id_gate_up_silu_batched_on_encoder(
            &st().pipes.device,
            enc,
            a_buf,
            gate_blocks,
            gate_byte_offset,
            up_blocks,
            up_byte_offset,
            ids_buf,
            out_buf,
            gate_n_rows,
            gate_n_cols,
            m,
            top_k,
            src1_outer_stride,
            src1_inner_stride,
        );
        Ok(())
    }

    fn gemm_moe_id(
        &self,
        ctx: &mut <MetalBackend as crate::backend::Backend>::Context,
        a: &<MetalBackend as crate::backend::Backend>::Buffer,
        ids: &<MetalBackend as crate::backend::Backend>::Buffer,
        tpe: &<MetalBackend as crate::backend::Backend>::Buffer,
        out: &mut <MetalBackend as crate::backend::Backend>::Buffer,
        ne11: usize,
        top_k: usize,
        max_per_expert: usize,
        batch: usize,
    ) -> Result<()> {
        let a_buf = a.expect_f32("gemm_moe_id a");
        let ids_buf = &ids.raw;
        let tpe_buf = &tpe.raw;
        let out_buf = out.expect_f32_mut("gemm_moe_id out");
        let enc = ctx.compute_encoder();
        match &self.store {
            MetalQuantStore::Q4KExperts {
                blocks,
                byte_offset,
                num_experts,
                n_rows,
                n_cols,
            } => {
                crate::q4_k_moe_id_gemm::dispatch_gemm_q4k_moe_id_on_encoder(
                    &st().pipes.device,
                    enc,
                    blocks,
                    *byte_offset,
                    a_buf,
                    ids_buf,
                    tpe_buf,
                    out_buf,
                    *num_experts,
                    *n_rows,
                    *n_cols,
                    ne11,
                    top_k,
                    max_per_expert,
                    batch,
                );
                Ok(())
            }
            MetalQuantStore::Q6KExperts {
                blocks,
                byte_offset,
                num_experts,
                n_rows,
                n_cols,
            } => {
                crate::q6_k_moe_id_gemm::dispatch_gemm_q6k_moe_id_on_encoder(
                    &st().pipes.device,
                    enc,
                    blocks,
                    *byte_offset,
                    a_buf,
                    ids_buf,
                    tpe_buf,
                    out_buf,
                    *num_experts,
                    *n_rows,
                    *n_cols,
                    ne11,
                    top_k,
                    max_per_expert,
                    batch,
                );
                Ok(())
            }
            _ => unreachable!("constructor guarantees experts variant"),
        }
    }

    fn gemm_moe_id_indirect(
        &self,
        ctx: &mut <MetalBackend as crate::backend::Backend>::Context,
        src1: &<MetalBackend as crate::backend::Backend>::Buffer,
        ids: &<MetalBackend as crate::backend::Backend>::Buffer,
        tpe: &<MetalBackend as crate::backend::Backend>::Buffer,
        out: &mut <MetalBackend as crate::backend::Backend>::Buffer,
        args_buf: &<MetalBackend as crate::backend::Backend>::Buffer,
        ne11: usize,
        top_k: usize,
        max_per_expert: usize,
        batch: usize,
    ) -> Result<()> {
        let a_buf = src1.expect_f32("gemm_moe_id_indirect a");
        let ids_buf = &ids.raw;
        let tpe_buf = &tpe.raw;
        let out_buf = out.expect_f32_mut("gemm_moe_id_indirect out");
        let args = &args_buf.raw;
        let enc = ctx.compute_encoder();
        match &self.store {
            MetalQuantStore::Q4KExperts {
                blocks,
                byte_offset,
                num_experts,
                n_rows,
                n_cols,
            } => {
                crate::q4_k_moe_id_gemm::dispatch_gemm_q4k_moe_id_indirect_on_encoder(
                    &st().pipes.device,
                    enc,
                    blocks,
                    *byte_offset,
                    a_buf,
                    ids_buf,
                    tpe_buf,
                    out_buf,
                    args,
                    *num_experts,
                    *n_rows,
                    *n_cols,
                    ne11,
                    top_k,
                    max_per_expert,
                    batch,
                );
                Ok(())
            }
            MetalQuantStore::Q6KExperts {
                blocks,
                byte_offset,
                num_experts,
                n_rows,
                n_cols,
            } => {
                crate::q6_k_moe_id_gemm::dispatch_gemm_q6k_moe_id_indirect_on_encoder(
                    &st().pipes.device,
                    enc,
                    blocks,
                    *byte_offset,
                    a_buf,
                    ids_buf,
                    tpe_buf,
                    out_buf,
                    args,
                    *num_experts,
                    *n_rows,
                    *n_cols,
                    ne11,
                    top_k,
                    max_per_expert,
                    batch,
                );
                Ok(())
            }
            _ => unreachable!("constructor guarantees experts variant"),
        }
    }
}
