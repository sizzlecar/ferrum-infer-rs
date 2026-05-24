//! `marlin_matmul` (GPTQ INT4) op-diff harness — **PARTIAL: planning stub**.
//!
//! The full op needs:
//!   - A: fp16 input `[m, k]`
//!   - B: packed INT4 weight in Marlin tile layout `[k / pack_factor, n]`
//!   - scales: fp16 `[k / group_size, n]`
//!   - zeros: int32 optional, `[k / group_size, n / pack_factor]`
//!   - g_idx: int32 optional, `[k]` for desc_act
//!
//! Setup needs a Marlin packer that converts a reference fp32 weight
//! matrix into the specific tile layout (`pack_factor=8`, `tile_size=16`,
//! interleaved nibbles). The packer lives in `ferrum-quantization` /
//! `ferrum-kernels/quantization/gptq_marlin/` but isn't exposed as a
//! testkit-callable helper.
//!
//! Reference impl: CPU backend's `gemm_quant` for `QuantKind::Gptq`
//! dequantizes the packed B back to fp32 then runs a regular sgemm.
//! That's what we'd compare CUDA's hand-tuned Marlin kernel against.
//!
//! Punted to follow-up: needs `marlin_pack_fixture(fp32 weight) ->
//! QuantWeights<B>` helper that all backends agree on. Without it
//! the test would be testing the PACKER not the matmul.

#![allow(dead_code)]

pub struct MarlinMatmulOp {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub group_size: usize,
}

// impl OpUnderTest for MarlinMatmulOp — pending marlin_pack_fixture helper.
