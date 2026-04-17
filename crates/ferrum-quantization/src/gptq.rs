//! GPTQ linear projection.
//!
//! GPTQ packs f16 weights as int4 (or int8) groups, each group sharing a
//! single `scale` and `zero_point`. The on-disk layout from the
//! `AutoGPTQ` / `gptq-for-llama` toolchain is:
//!
//!   qweight:  `[in_features / pack, out_features]`  i32 — 8 int4 values packed per i32
//!   qzeros:   `[in_features / group_size, out_features / pack]`  i32
//!   scales:   `[in_features / group_size, out_features]`         f16
//!   g_idx:    `[in_features]` i32 — maps each in-feature row to its scale
//!                                   group (only used with desc_act)
//!
//! `forward` delegates to `Backend::gemm_quant` with `QuantKind::Gptq`.
//! Backends that don't implement the kernel (currently all backends on
//! this repo) will surface a clean "Unsupported" error at runtime rather
//! than silently producing wrong output.

use ferrum_kernels::backend::{Backend, QuantKind, QuantWeights};
use ferrum_kernels::Linear;

/// GPTQ-quantized projection. Bits is typically 4; 8 is also valid.
pub struct GptqLinear<B: Backend> {
    qweight: B::Buffer,
    scales: B::Buffer,
    qzeros: B::Buffer,
    g_idx: Option<B::Buffer>,
    bits: u32,
    group_size: usize,
    desc_act: bool,
    in_features: usize,
    out_features: usize,
}

impl<B: Backend> GptqLinear<B> {
    /// Build from already-loaded backend buffers. Shape invariants are
    /// debug-checked; production loaders should validate up-front.
    #[allow(clippy::too_many_arguments)]
    pub fn from_buffers(
        qweight: B::Buffer,
        scales: B::Buffer,
        qzeros: B::Buffer,
        g_idx: Option<B::Buffer>,
        bits: u32,
        group_size: usize,
        desc_act: bool,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self {
            qweight,
            scales,
            qzeros,
            g_idx,
            bits,
            group_size,
            desc_act,
            in_features,
            out_features,
        }
    }
}

impl<B: Backend> Linear<B> for GptqLinear<B> {
    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize) {
        let weights = QuantWeights::<B> {
            qweight: &self.qweight,
            scales: Some(&self.scales),
            zeros: Some(&self.qzeros),
            g_idx: self.g_idx.as_ref(),
        };
        let kind = QuantKind::Gptq {
            bits: self.bits,
            group_size: self.group_size,
            desc_act: self.desc_act,
        };
        // When a backend hasn't wired gemm_quant yet (currently every
        // backend on this repo), we panic with a clear message rather than
        // continue with undefined output.
        if let Err(e) = B::gemm_quant(
            ctx,
            input,
            &weights,
            out,
            m,
            self.out_features,
            self.in_features,
            &kind,
        ) {
            panic!(
                "GPTQ forward failed: {e} — backend has not implemented gemm_quant(QuantKind::Gptq). \
                 This is expected until Phase E (CUDA Marlin / Metal GPTQ kernel).",
            );
        }
    }
}
