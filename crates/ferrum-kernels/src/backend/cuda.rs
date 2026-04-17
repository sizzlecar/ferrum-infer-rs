//! CUDA backend — skeleton for Phase E.
//!
//! Buffer type: `CudaSlice<f16>` — all operations share a single CUDA
//! stream. Kernel PTX modules already exist under `crates/ferrum-kernels/src/`
//! (they were kept from the pre-architecture-v2 CUDA path). This file is
//! the Backend-trait adapter; each `todo!()` below points at the kernel
//! file that already implements the math and at the PTX module to load.
//!
//! ## cudarc 0.19 API notes
//!
//! - `CudaContext::new(ordinal) -> Result<Arc<CudaContext>>` replaces the
//!   pre-0.17 `CudaDevice::new` name.
//! - Allocation and memcpy live on `CudaStream` in 0.19 (not on the
//!   context). Use `stream.alloc(len)`, `stream.clone_htod(&host)`,
//!   `stream.memcpy_dtoh(src, &mut host)`, etc.
//! - Synchronization: `stream.synchronize()`. The context has its own
//!   `synchronize()` that covers all streams in the context.
//!
//! ## Phase E wiring recipe for each op
//!
//! 1. In `crate::ptx` the PTX bytes are compiled at build time by
//!    `ferrum-kernels/build.rs`. Module names match file basenames:
//!        ptx::RMS_NORM               ← src/rms_norm.cu
//!        ptx::ROPE                   ← src/rope.cu
//!        ptx::FUSED_ADD_RMS_NORM     ← src/fused_add_rms_norm.cu
//!        ptx::FUSED_SILU_MUL         ← src/fused_silu_mul.cu
//!        ptx::DECODE_ATTENTION       ← src/decode_attention.cu (paged + legacy)
//!        ptx::RESIDUAL_ADD           ← src/residual_add.cu
//!
//! 2. For each Backend method, the pattern is (ported from the existing
//!    kernel wrapper files):
//!
//!        let module = ctx.ctx.load_module(ptx::RMS_NORM)?;
//!        let func   = module.load_function("rms_norm_f16")?;
//!        let cfg    = cudarc::driver::LaunchConfig {
//!            grid_dim:        (tokens as u32, 1, 1),
//!            block_dim:       (dim.min(1024) as u32, 1, 1),
//!            shared_mem_bytes: 0,
//!        };
//!        let mut b = ctx.stream.launch_builder(&func);
//!        b.arg(x); b.arg(w); b.arg(out);
//!        b.arg(&(dim as i32)); b.arg(&eps);
//!        unsafe { b.launch(cfg) }?;
//!
//! 3. `ferrum_kernels::{rms_norm,rope,fused_add_rms_norm,...}` are the
//!    reference implementations that went through candle's Tensor API.
//!    Their argument order and launch config are battle-tested — copy
//!    them, but use CudaSlice directly instead of extracting from
//!    candle's Storage.
//!
//! 4. GEMM uses cuBLAS hgemm via `cudarc::cublas::CudaBlas`. See
//!    `crate::cublas` for the existing hgemm wrapper (36 lines).
//!
//! 5. Parity verify on GPU with `scripts/phase-e-verify.sh`.

#![allow(unused_variables, dead_code)]

use super::{AttnConfig, Backend, QuantKind, QuantWeights, ReduceOp};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use ferrum_types::{FerrumError, Result};
use half::f16;
use std::sync::Arc;

/// CUDA backend state: context + stream.
///
/// Created once per inference session. All kernel launches go through
/// the same stream for implicit ordering. NCCL multi-GPU support
/// (for Tensor Parallel) will add rank/world_size here in Phase E.
pub struct CudaState {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
}

pub struct CudaBackend;

impl Backend for CudaBackend {
    type Buffer = CudaSlice<f16>;
    type Context = CudaState;

    // ── Lifecycle (mechanically correct, can be trusted on GPU) ──────────

    fn new_context() -> Self::Context {
        // Ordinal comes from env `FERRUM_CUDA_DEVICE` or defaults to 0.
        // This is the only place device selection happens.
        let ordinal = std::env::var("FERRUM_CUDA_DEVICE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let ctx = CudaContext::new(ordinal).unwrap_or_else(|e| {
            panic!("CudaBackend::new_context: failed to init context {ordinal}: {e}")
        });
        // Use a dedicated stream so we don't contend with the default
        // stream that candle uses during prefill / weight-loading.
        let stream = ctx
            .new_stream()
            .unwrap_or_else(|e| panic!("CudaBackend::new_context: failed to create stream: {e}"));
        // Populate thread-local so context-free ops can route allocations.
        GLOBAL_STREAM.with(|slot| *slot.borrow_mut() = Some(stream.clone()));
        Self::Context { ctx, stream }
    }

    fn sync(ctx: &mut Self::Context) {
        // Wait on this backend's stream. Other streams (candle prefill,
        // weight init) are unaffected — callers that need a full device
        // barrier must call `ctx.ctx.synchronize()` instead.
        ctx.stream.synchronize().expect("cuda stream sync");
    }

    fn alloc(len: usize) -> Self::Buffer {
        // Transient device allocation. Signature inherited from Backend
        // trait (no ctx arg) — route through thread-local stream.
        GLOBAL_STREAM.with(|slot| {
            let stream = slot
                .borrow()
                .as_ref()
                .cloned()
                .expect("cuda alloc: GLOBAL_STREAM empty; call new_context first");
            unsafe { stream.alloc::<f16>(len) }.expect("cuda alloc")
        })
    }

    fn from_slice(data: &[f32]) -> Self::Buffer {
        // f32 → f16 on host, then htod copy via the thread-local stream.
        GLOBAL_STREAM.with(|slot| {
            let stream = slot
                .borrow()
                .as_ref()
                .cloned()
                .expect("cuda from_slice: GLOBAL_STREAM empty");
            let f16_host: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
            stream.clone_htod(&f16_host).expect("htod copy")
        })
    }

    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        // dtoh + f16 → f32. Forces a stream sync so the caller sees
        // completed data.
        GLOBAL_STREAM.with(|slot| {
            let stream = slot
                .borrow()
                .as_ref()
                .cloned()
                .expect("cuda to_vec: GLOBAL_STREAM empty");
            let mut host = vec![f16::ZERO; len];
            stream.memcpy_dtoh(buf, &mut host).expect("dtoh copy");
            stream.synchronize().expect("dtoh sync");
            host.into_iter().map(|x| x.to_f32()).collect()
        })
    }

    // ── Math primitives (PHASE E: wire to PTX) ───────────────────────────

    /// TODO(Phase E): launch `rms_norm_f16` from `ptx::RMS_NORM`.
    /// See `ferrum_kernels::rms_norm::rms_norm` for the candle-based reference —
    /// grid = tokens, block = min(dim, 1024), args = (x, w, out, dim_i32, eps).
    fn rms_norm(
        _ctx: &mut Self::Context,
        _x: &Self::Buffer,
        _w: &Self::Buffer,
        _eps: f32,
        _out: &mut Self::Buffer,
        _tokens: usize,
        _dim: usize,
    ) {
        todo!("rms_norm — see src/rms_norm.rs reference (candle-based)");
    }

    /// TODO(Phase E): launch `fused_add_rms_norm_f16` from `ptx::FUSED_ADD_RMS_NORM`.
    /// See `ferrum_kernels::fused_add_rms_norm` for reference impl.
    /// Note: this op is in-place on residual — `residual` arg is both in and out.
    fn fused_add_rms_norm(
        _ctx: &mut Self::Context,
        _residual: &mut Self::Buffer,
        _x: &Self::Buffer,
        _w: &Self::Buffer,
        _eps: f32,
        _out: &mut Self::Buffer,
        _tokens: usize,
        _dim: usize,
    ) {
        todo!("fused_add_rms_norm — see src/fused_add_rms_norm.rs reference");
    }

    /// TODO(Phase E): cuBLAS hgemm via `cudarc::cublas::CudaBlas::hgemm`.
    /// See `crate::cublas` (36 lines) — that's already a working wrapper;
    /// can be called directly. Row-major `A @ B^T` maps to column-major
    /// `B @ A^T` — double-check the transpose flags.
    fn gemm(
        _ctx: &mut Self::Context,
        _a: &Self::Buffer,
        _b: &Self::Buffer,
        _out: &mut Self::Buffer,
        _m: usize,
        _n: usize,
        _k: usize,
    ) {
        todo!("gemm — wire cudarc::cublas::CudaBlas::hgemm via crate::cublas");
    }

    /// TODO(Phase E): launch flash-decode or flash-attn full based on q_len.
    /// See `ferrum_kernels::cuda_decode` — has production-grade
    /// `flash_decode_attn_f16` + `flash_decode_reduce_f16` implementations
    /// and a `decode_attention_f16` single-block variant. cfg.kv_seq_stride
    /// > 0 means paged / stride-aware; propagate that through.
    fn flash_attention(
        _ctx: &mut Self::Context,
        _q: &Self::Buffer,
        _k: &Self::Buffer,
        _v: &Self::Buffer,
        _out: &mut Self::Buffer,
        _batch: usize,
        _q_len: usize,
        _kv_len: usize,
        _pos_offset: usize,
        _cfg: &AttnConfig,
    ) {
        todo!("flash_attention — reuse kernels from src/cuda_decode.rs");
    }

    /// TODO(Phase E): `stream.memcpy_dtod` on a view into src/dst using
    /// `CudaView::slice` at the appropriate offsets.
    fn copy_slice(
        _ctx: &mut Self::Context,
        _src: &Self::Buffer,
        _src_offset: usize,
        _dst: &mut Self::Buffer,
        _dst_offset: usize,
        _len: usize,
    ) {
        todo!("copy_slice — slice views + stream.memcpy_dtod");
    }

    /// TODO(Phase E): upload ids to device (small buffer) + launch
    /// `embedding_lookup_f16` kernel. Reference: existing code uses
    /// candle's Tensor::index_select which already does the right thing
    /// — extract just the CUDA launch bit.
    fn embedding_lookup(
        _ctx: &mut Self::Context,
        _table: &Self::Buffer,
        _ids: &[u32],
        _out: &mut Self::Buffer,
        _dim: usize,
    ) {
        todo!("embedding_lookup — simple launch, no reference kernel yet");
    }

    /// TODO(Phase E): simple element-wise kernel; 3 writes per row.
    /// Launch shape: grid = (tokens * (q_dim + 2*kv_dim) / block), block = 256.
    fn split_qkv(
        _ctx: &mut Self::Context,
        _qkv: &Self::Buffer,
        _q: &mut Self::Buffer,
        _k: &mut Self::Buffer,
        _v: &mut Self::Buffer,
        _tokens: usize,
        _q_dim: usize,
        _kv_dim: usize,
    ) {
        todo!("split_qkv — write a tiny .cu kernel or do 3 memcpy_dtod ops");
    }

    /// TODO(Phase E): `ptx::FUSED_SILU_MUL` has `fused_silu_mul_f16`; it
    /// expects already-split gate / up. For the split variant, either
    /// (a) write a new kernel, or (b) split+call existing. See
    /// `ferrum_kernels::fused_silu_mul` for reference.
    fn fused_silu_mul_split(
        _ctx: &mut Self::Context,
        _gate_up: &Self::Buffer,
        _out: &mut Self::Buffer,
        _tokens: usize,
        _im: usize,
    ) {
        todo!("fused_silu_mul_split — reuse src/fused_silu_mul.rs");
    }

    /// TODO(Phase E): `ptx::ROPE` has `rope_f16` (single token) and
    /// `batched_rope_f16` (multi-token). QK-norm path needs a fused
    /// `qk_norm_rope_transpose_f16`. The Metal version splits modes via
    /// `apply_norm` param (0/1/2); same logic should port.
    fn qk_norm_rope(
        _ctx: &mut Self::Context,
        _input: &Self::Buffer,
        _norm_w: &Self::Buffer,
        _cos: &Self::Buffer,
        _sin: &Self::Buffer,
        _output: &mut Self::Buffer,
        _tokens: usize,
        _heads: usize,
        _head_dim: usize,
        _pos_offset: usize,
        _eps: f32,
        _mode: i32,
    ) {
        todo!("qk_norm_rope — new fused kernel needed, port from Metal norm_rope.metal");
    }

    /// TODO(Phase E): simple strided copy into pre-allocated cache.
    /// Grid = `(new_tokens * nkv * hd / block)`, block = 256, one write per element.
    fn kv_cache_append_head_major(
        _ctx: &mut Self::Context,
        _cache_k: &mut Self::Buffer,
        _cache_v: &mut Self::Buffer,
        _cache_len: usize,
        _cache_capacity: usize,
        _new_k_head_major: &Self::Buffer,
        _new_v_head_major: &Self::Buffer,
        _new_tokens: usize,
        _nkv: usize,
        _hd: usize,
    ) {
        todo!("kv_cache_append_head_major — tiny append kernel, port Metal kv_cache_append shader");
    }

    /// TODO(Phase E): transpose `[heads, tokens, dim]` → `[tokens, heads, dim]`.
    /// Port the Metal `transpose_out_f32` shader line-by-line.
    fn transpose_head_to_token(
        _ctx: &mut Self::Context,
        _src: &Self::Buffer,
        _dst: &mut Self::Buffer,
        _tokens: usize,
        _heads: usize,
        _dim: usize,
    ) {
        todo!("transpose_head_to_token — tiny kernel, port Metal shader");
    }

    /// TODO(Phase E): reuse `ptx::RESIDUAL_ADD::residual_add_f16`.
    /// Launch: grid = (len/block, 1, 1), block = 256, args = (a, b, out, len).
    fn add_inplace(
        _ctx: &mut Self::Context,
        _residual: &mut Self::Buffer,
        _x: &Self::Buffer,
        _len: usize,
    ) {
        todo!("add_inplace — reuse ptx::RESIDUAL_ADD, in/out = same buffer");
    }

    /// TODO(Phase E): broadcast bias add. Simple kernel:
    /// `data[r * cols + c] += bias[c]`. Grid = rows, block = min(cols, 1024).
    fn add_bias(
        _ctx: &mut Self::Context,
        _data: &mut Self::Buffer,
        _bias: &Self::Buffer,
        _rows: usize,
        _cols: usize,
    ) {
        todo!("add_bias — new kernel, trivial");
    }

    /// TODO(Phase E): full LayerNorm (mean + variance + affine).
    /// Port the Metal `layer_norm_f32` shader (two-pass simd_sum). Used by
    /// Bert/Clip/Whisper encoders only; LLM path uses rms_norm.
    fn layer_norm(
        _ctx: &mut Self::Context,
        _x: &Self::Buffer,
        _gamma: &Self::Buffer,
        _beta: &Self::Buffer,
        _eps: f32,
        _out: &mut Self::Buffer,
        _tokens: usize,
        _dim: usize,
    ) {
        todo!("layer_norm — new .cu kernel, port Metal layer_norm_f32");
    }

    /// TODO(Phase E): element-wise GELU (erf-based). Trivial kernel.
    fn gelu(
        _ctx: &mut Self::Context,
        _x: &Self::Buffer,
        _out: &mut Self::Buffer,
        _len: usize,
    ) {
        todo!("gelu — element-wise, use erff() or polynomial approx");
    }

    // ── Quantized GEMM (Phase E) ─────────────────────────────────────────

    /// TODO(Phase E-GPTQ): wire `ferrum_kernels::marlin` (existing 316 LOC
    /// Marlin kernel binding). Production-grade, already achieved 112 tok/s
    /// on RTX PRO 6000. Takes cudarc CudaSlice<i32> (qweight, qzeros) +
    /// CudaSlice<f16> (scales) — types need to match up cleanly with
    /// `QuantWeights<'_, Self>` shape.
    ///
    /// AWQ and GGUF stay Unsupported until their kernels are wired (far
    /// later — GPTQ covers the main production need).
    fn gemm_quant(
        _ctx: &mut Self::Context,
        _a: &Self::Buffer,
        _weights: &QuantWeights<'_, Self>,
        _out: &mut Self::Buffer,
        _m: usize,
        _n: usize,
        _k: usize,
        kind: &QuantKind,
    ) -> Result<()> {
        Err(FerrumError::unsupported(format!(
            "CudaBackend::gemm_quant({kind:?}) not yet wired — see \
             crate::marlin for GPTQ kernel; AWQ/GGUF land later"
        )))
    }

    // ── TP collectives (Phase E with NCCL) ───────────────────────────────

    fn world_size(_ctx: &Self::Context) -> usize {
        // TODO(Phase E-TP): read from NcclComm group if present, else 1.
        // See `crate::nccl_comm` for existing NCCL wrapper.
        std::env::var("FERRUM_TP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
    }

    fn rank(_ctx: &Self::Context) -> usize {
        std::env::var("FERRUM_RANK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
    }

    fn all_reduce(
        _ctx: &mut Self::Context,
        _buf: &mut Self::Buffer,
        _len: usize,
        _op: ReduceOp,
    ) {
        // TODO(Phase E-TP): NcclComm::all_reduce_sum with op mapping.
        // World size 1 path is already no-op via default trait impl;
        // this override activates when FERRUM_TP > 1.
    }

    fn all_gather(
        _ctx: &mut Self::Context,
        _local: &Self::Buffer,
        _global: &mut Self::Buffer,
        _local_len: usize,
    ) {
        // TODO(Phase E-TP): NcclComm::all_gather.
    }

    fn broadcast(
        _ctx: &mut Self::Context,
        _buf: &mut Self::Buffer,
        _len: usize,
        _src_rank: usize,
    ) {
        // TODO(Phase E-TP): NcclComm::broadcast.
    }
}

// ── GLOBAL_STREAM: thread-local stream handle for context-free ops ──────
//
// `alloc` / `from_slice` / `to_vec` don't get a `&ctx` argument (trait
// signature is inherited from CpuBackend where they're truly context-free).
// CUDA needs a stream though — cudarc 0.19 hangs the alloc/memcpy APIs
// off `CudaStream`, not the context. `new_context` populates the
// thread-local so subsequent context-free calls can route through it.

use std::cell::RefCell;

thread_local! {
    static GLOBAL_STREAM: RefCell<Option<Arc<CudaStream>>> = const { RefCell::new(None) };
}

/// Install a stream as the current thread's default for context-free
/// allocations. Called automatically by `new_context`; override manually
/// if a worker thread issues allocations without having created its own
/// context.
pub fn install_thread_stream(stream: Arc<CudaStream>) {
    GLOBAL_STREAM.with(|slot| {
        *slot.borrow_mut() = Some(stream);
    });
}
