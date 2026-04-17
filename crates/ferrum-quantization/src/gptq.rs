//! GPTQ linear projection.
//!
//! GPTQ packs f16 weights as int4 groups, each group sharing a scale +
//! zero_point. On-disk layout from AutoGPTQ / gptq-for-llama:
//!
//!   qweight:  `[in_features / 8, out_features]`  i32 — 8 int4s per int32
//!   qzeros:   `[in_features / group_size, out_features / 8]`  i32
//!   scales:   `[in_features / group_size, out_features]`      f16
//!   g_idx:    `[in_features]` i32 — per-row scale-group map (desc_act only)
//!
//! `GptqLinear<B>` stores a backend-specific `B::GptqStore` produced by
//! `Backend::load_gptq`. The store holds whatever format the backend
//! needs (CUDA: Marlin-repacked tiles; CPU: dequantized f32 weights;
//! Metal: unsupported).

use ferrum_kernels::backend::Backend;
use ferrum_kernels::Linear;
use ferrum_types::Result;

pub struct GptqLinear<B: Backend> {
    store: B::GptqStore,
    in_features: usize,
    out_features: usize,
}

impl<B: Backend> GptqLinear<B> {
    /// Build from raw host-side GPTQ tensors. The Backend repacks into
    /// its preferred format once; inference uses the repacked store.
    ///
    /// `qweight`: `[k/8, n]` i32 (packed int4)
    /// `scales`:  `[k/group_size, n]` f32 (converted from f16 by caller)
    /// `qzeros`:  `[k/group_size, n/8]` i32
    /// `g_idx`:   `[k]` i32 — optional, only used for desc_act=true
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw(
        qweight: &[i32],
        scales: &[f32],
        qzeros: &[i32],
        g_idx: Option<&[i32]>,
        bits: u32,
        group_size: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Self> {
        let store = B::load_gptq(
            qweight,
            scales,
            qzeros,
            g_idx,
            bits,
            group_size,
            in_features,
            out_features,
        )?;
        Ok(Self {
            store,
            in_features,
            out_features,
        })
    }

    /// Construct directly from a pre-built backend store (e.g. tests).
    pub fn from_store(store: B::GptqStore, in_features: usize, out_features: usize) -> Self {
        Self {
            store,
            in_features,
            out_features,
        }
    }

    pub fn store(&self) -> &B::GptqStore {
        &self.store
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
        B::gemm_gptq(ctx, input, &self.store, out, m)
            .unwrap_or_else(|e| panic!("GPTQ forward failed: {e}"));
    }
}
