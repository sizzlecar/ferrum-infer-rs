//! `QuantLinear<B>` — keeps Q4_K_M (or future k-quant) weights quantised
//! in backend memory and dequants on-demand per `forward` call.
//!
//! Contrast with `GgufLinear<B>` which **eagerly** dequants Q4_K_M to
//! fp32/fp16 at load time. That eager path inflates an 8B model from
//! ~5 GB on disk to 16-32 GB in RAM — fine for safetensors-fp16 sources
//! but wasteful for GGUF Q4_K_M and a non-starter for 30B-A3B on a
//! 32 GB Mac.
//!
//! The Q4 → fp16 conversion happens inside `Backend::gemm_q4_k`, into a
//! transient buffer that's freed after the matmul. Memory footprint is
//! the on-disk Q4 size + a per-call transient ~= one weight matrix's
//! worth of fp16.
//!
//! Phase 1D scope: direct (un-fused) Q4_K_M projections only —
//! `o_proj`, `down_proj`, `lm_head`, `embed_tokens`, etc. Fused
//! projections (`qkv_proj`, `gate_up_proj`) keep falling through to
//! `GgufLinear`'s eager-dequant path; the loader's split-fusion logic
//! already concatenates the dequanted parts into one dense weight.

use ferrum_kernels::backend::{Backend, GgufQuantType};
use ferrum_kernels::Linear;
use ferrum_types::Result;

/// Linear projection backed by a GGUF k-quant weight kept quantised in
/// backend memory.
///
/// `forward` calls into `Backend::gemm_quant`, which dequants the
/// weight into a transient fp16 buffer (Metal) or pre-dequanted fp32
/// weights (CPU) and then runs the matmul. See `B::QuantStore` per
/// backend for the storage format details.
///
/// Future k-quant flavours (Q5_K, Q6_K, Q8_0) plug in via the
/// [`GgufQuantType`] discriminator passed to the constructor — no new
/// `QuantLinear` type required.
pub struct QuantLinear<B: Backend> {
    store: B::QuantStore,
    in_features: usize,
    out_features: usize,
}

impl<B: Backend> QuantLinear<B> {
    /// Build from raw GGUF block bytes.
    ///
    /// `kind`: which k-quant flavour the bytes encode (Q4_K, Q5_K, …).
    /// `bytes`: the on-disk payload, sized by the kind's block layout.
    pub fn from_gguf_bytes(
        kind: GgufQuantType,
        bytes: &[u8],
        out_features: usize,
        in_features: usize,
    ) -> Result<Self> {
        let store = B::load_quant(kind, bytes, out_features, in_features)?;
        Ok(Self {
            store,
            in_features,
            out_features,
        })
    }

    /// Build a fused projection from multiple `(kind, bytes, rows)`
    /// parts that share `in_features`. Each part stays in its own
    /// QuantStore (no byte-concat); forward dispatches one matvec per
    /// part. Used for Qwen3 `qkv_proj` when q+k are Q4_K and v is Q6_K
    /// — the homogeneous fused-Q4 fast path would have to fall back
    /// to eager-fp32, blowing 100 MB per layer.
    pub fn from_gguf_fused(
        parts: &[(GgufQuantType, &[u8], usize)],
        in_features: usize,
    ) -> Result<Self> {
        let store = B::load_quant_fused(parts, in_features)?;
        let out_features = parts.iter().map(|(_, _, n)| *n).sum();
        Ok(Self {
            store,
            in_features,
            out_features,
        })
    }

    /// For tests / advanced callers that have already constructed a
    /// `B::QuantStore` (e.g. through the Backend's own ingestion path).
    pub fn from_store(store: B::QuantStore, out_features: usize, in_features: usize) -> Self {
        Self {
            store,
            in_features,
            out_features,
        }
    }

    pub fn store(&self) -> &B::QuantStore {
        &self.store
    }
}

impl<B: Backend> Linear<B> for QuantLinear<B> {
    fn in_features(&self) -> usize {
        self.in_features
    }

    fn out_features(&self) -> usize {
        self.out_features
    }

    fn forward(&self, ctx: &mut B::Context, input: &B::Buffer, out: &mut B::Buffer, m: usize) {
        // Trait-level dispatch — the kernel choice (Metal compute kernel vs
        // CPU dequant+gemm, and which k-quant flavour) is encapsulated in
        // the backend's `QuantStore` enum.
        B::gemm_quant(ctx, input, &self.store, out, m).expect("gemm_quant");
    }
}
