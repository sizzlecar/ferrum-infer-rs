//! GPTQ linear projection — thin factory wrapper.
//!
//! Phase 3e/2: the actual kernel dispatch lives inside the boxed
//! `Linear<B>` returned by `B::load_gptq` (`CudaMarlinLinear` on
//! CUDA, `CpuGptqLinear` on CPU). This module just re-exposes the
//! historical constructor names so callers don't have to switch.

use ferrum_kernels::backend::{Backend, BackendQuantMarlin};
use ferrum_kernels::Linear;
use ferrum_kernels::LinearMetadata;
use ferrum_types::Result;
use std::sync::Arc;

/// GPTQ-format Linear projection, polymorphic over backend.
///
/// Holds a boxed backend-specific `Linear<B>` produced by `B::load_gptq`.
/// `forward()` delegates straight through.
pub struct GptqLinear<B: Backend + BackendQuantMarlin> {
    inner: Box<dyn Linear<B> + Send + Sync>,
    metadata: LinearMetadata,
}

impl<B: Backend + BackendQuantMarlin> GptqLinear<B> {
    /// Build from raw host-side GPTQ tensors. The Backend repacks into
    /// its preferred format once (Marlin tiles on CUDA, dequant on CPU)
    /// and returns a boxed Linear; inference uses the boxed forward.
    ///
    /// `qweight`: `[k/8, n]` i32 (packed int4)
    /// `scales`:  `[k/group_size, n]` f32 (converted from f16 by caller)
    /// `qzeros`:  `[k/group_size, n/8]` i32
    /// `g_idx`:   `[k]` i32 — optional, only used for desc_act=true
    /// `bias`:    `[n]` f32 — optional fused bias (Qwen2.5 attention)
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw(
        qweight: &[i32],
        scales: &[f32],
        qzeros: &[i32],
        g_idx: Option<&[i32]>,
        bias: Option<&[f32]>,
        bits: u32,
        group_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        Self::from_raw_with_metadata(
            qweight,
            scales,
            qzeros,
            g_idx,
            bias,
            bits,
            group_size,
            in_features,
            out_features,
            LinearMetadata::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_with_metadata(
        qweight: &[i32],
        scales: &[f32],
        qzeros: &[i32],
        g_idx: Option<&[i32]>,
        bias: Option<&[f32]>,
        bits: u32,
        group_size: usize,
        in_features: usize,
        out_features: usize,
        metadata: LinearMetadata,
    ) -> Result<Self> {
        let inner = B::load_gptq(
            qweight,
            scales,
            qzeros,
            g_idx,
            bias,
            bits,
            group_size,
            in_features,
            out_features,
        )?;
        Ok(Self { inner, metadata })
    }
}

impl<B: Backend + BackendQuantMarlin> Linear<B> for GptqLinear<B> {
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    fn metadata(&self) -> LinearMetadata {
        if self.metadata.is_empty() {
            self.inner.metadata()
        } else {
            self.metadata
        }
    }

    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize) {
        self.inner.forward(ctx, input, out, m);
    }
}

/// View into a single column-slice of a shared stacked GPTQ store.
///
/// Phase 3e/2: backed by a `Box<dyn Linear<B>>` produced by
/// `B::make_stacked_expert_linear` (CUDA: `CudaMarlinStackedExpertLinear`;
/// CPU: `CpuGptqLinear` over a sliced row range). The store itself is
/// `Arc<B::GptqStore>` so cloning a view is cheap; dropping all views
/// drops the underlying store.
pub struct StackedExpertLinear<B: Backend + BackendQuantMarlin> {
    inner: Box<dyn Linear<B> + Send + Sync>,
    /// Kept for in_features() reporting.
    k: usize,
    /// Kept for out_features() reporting.
    expert_n: usize,
}

impl<B: Backend + BackendQuantMarlin> StackedExpertLinear<B> {
    /// Phase C step 4b: takes the trait-object MarlinExpertStack
    /// directly (was `Arc<B::GptqStore>` + `B::make_stacked_expert_linear`).
    pub fn new(
        stack: Arc<dyn ferrum_kernels::MarlinExpertStack<B>>,
        expert_offset: usize,
        expert_n: usize,
    ) -> Result<Self> {
        let k = stack.k();
        let inner = stack.make_expert_linear(expert_offset, expert_n, None)?;
        Ok(Self { inner, k, expert_n })
    }

    pub fn new_with_bias(
        stack: Arc<dyn ferrum_kernels::MarlinExpertStack<B>>,
        expert_offset: usize,
        expert_n: usize,
        bias: &[f32],
    ) -> Result<Self> {
        let k = stack.k();
        let inner = stack.make_expert_linear(expert_offset, expert_n, Some(bias))?;
        Ok(Self { inner, k, expert_n })
    }
}

impl<B: Backend + BackendQuantMarlin> Linear<B> for StackedExpertLinear<B> {
    fn in_features(&self) -> usize {
        self.k
    }

    fn out_features(&self) -> usize {
        self.expert_n
    }

    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize) {
        self.inner.forward(ctx, input, out, m);
    }
}
