//! CUDA backend — Phase E implementation.
//!
//! Buffer type: `CudaSlice<f16>`. Kernels are compiled to PTX at build
//! time (`ferrum-kernels/build.rs`) and loaded lazily into this backend's
//! `CudaContext`. GEMM delegates to cuBLAS via `cudarc::cublas::CudaBlas`
//! (one handle per `CudaState`, bound to the session stream).
//!
//! Decoupled from candle — pure `cudarc` 0.19 APIs.
//!
//! ## Still TODO / out of scope for this commit
//!
//! - `gemm_quant`: returns `unsupported`. Wiring Marlin requires the
//!   `QuantWeights` buffer type to carry mixed dtypes (int32 qweight +
//!   f16 scales); current `Backend::Buffer = CudaSlice<f16>` blocks a
//!   clean impl. Tracked separately (Phase E-GPTQ).
//! - `all_gather` / `broadcast`: no NCCL wrapper yet in
//!   `crate::nccl_comm` (only `all_reduce_f16_inplace` exists). `all_reduce`
//!   is wired; the other two remain no-op until the wrapper is extended.
//! - `mla_attention`: default unsupported error — DeepSeek V2/V3 not a
//!   Phase E target.

#![allow(unused_variables, dead_code)]

use super::{AttnConfig, Backend, QuantKind, QuantWeights, ReduceOp};
use crate::ptx;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use ferrum_types::{FerrumError, Result};
use half::f16;
use std::collections::HashMap;
use std::sync::Arc;

// ────────────────────────────────────────────────────────────────────────
// Context
// ────────────────────────────────────────────────────────────────────────

/// Execution context for CudaBackend.
///
/// Owns the `CudaContext`, a dedicated `CudaStream`, a cuBLAS handle
/// bound to that stream, and a lazy cache of PTX modules. All kernels
/// launch on `stream`; sync'ing `stream` covers all of this backend's work.
pub struct CudaState {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: Arc<CudaBlas>,
    modules: HashMap<&'static str, Arc<CudaModule>>,
}

impl CudaState {
    fn module(&mut self, key: &'static str, ptx_src: &str) -> Arc<CudaModule> {
        if let Some(m) = self.modules.get(key) {
            return m.clone();
        }
        let m = self
            .ctx
            .load_module(Ptx::from_src(ptx_src.to_string()))
            .unwrap_or_else(|e| panic!("CudaBackend: load_module({key}): {e}"));
        self.modules.insert(key, m.clone());
        m
    }

    fn func(
        &mut self,
        module_key: &'static str,
        ptx_src: &str,
        fn_name: &'static str,
    ) -> CudaFunction {
        let m = self.module(module_key, ptx_src);
        m.load_function(fn_name)
            .unwrap_or_else(|e| panic!("CudaBackend: load_function({fn_name}): {e}"))
    }
}

// ────────────────────────────────────────────────────────────────────────
// FlashAttnParams — mirrors C struct in kernels/flash_attn_full.cu
// ────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct FlashAttnParams {
    batch: i32,
    num_heads: i32,
    num_kv_heads: i32,
    q_len: i32,
    kv_len: i32,
    head_dim: i32,
    causal: i32,
    pos_offset: i32,
    kv_seq_stride: i32,
}

unsafe impl DeviceRepr for FlashAttnParams {}

// ────────────────────────────────────────────────────────────────────────
// Backend impl
// ────────────────────────────────────────────────────────────────────────

pub struct CudaBackend;

impl Backend for CudaBackend {
    type Buffer = CudaSlice<f16>;
    type Context = CudaState;

    // ── Lifecycle ────────────────────────────────────────────────────────

    fn new_context() -> Self::Context {
        let ordinal = std::env::var("FERRUM_CUDA_DEVICE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let ctx = CudaContext::new(ordinal).unwrap_or_else(|e| {
            panic!("CudaBackend::new_context: failed to init context {ordinal}: {e}")
        });
        let stream = ctx
            .new_stream()
            .unwrap_or_else(|e| panic!("CudaBackend::new_context: failed to create stream: {e}"));
        let blas = Arc::new(
            CudaBlas::new(stream.clone())
                .unwrap_or_else(|e| panic!("CudaBackend::new_context: CudaBlas::new: {e}")),
        );
        install_thread_stream(stream.clone());
        Self::Context {
            ctx,
            stream,
            blas,
            modules: HashMap::new(),
        }
    }

    fn sync(ctx: &mut Self::Context) {
        ctx.stream.synchronize().expect("CudaBackend: stream sync");
    }

    fn alloc(len: usize) -> Self::Buffer {
        with_stream(|stream| unsafe { stream.alloc::<f16>(len) }.expect("cuda alloc"))
    }

    fn from_slice(data: &[f32]) -> Self::Buffer {
        let host: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
        with_stream(|stream| stream.clone_htod(&host).expect("cuda htod"))
    }

    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        with_stream(|stream| {
            let mut host = vec![f16::ZERO; len];
            stream.memcpy_dtoh(buf, &mut host).expect("cuda dtoh");
            stream.synchronize().expect("cuda dtoh sync");
            host.into_iter().map(|x| x.to_f32()).collect()
        })
    }

    // ── Norms ────────────────────────────────────────────────────────────

    fn rms_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let func = ctx.func("rms_norm", ptx::RMS_NORM, "rms_norm_f16");
        let dim_i32 = dim as i32;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(x);
        b.arg(w);
        b.arg(out);
        b.arg(&dim_i32);
        b.arg(&eps);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (tokens as u32, 1, 1),
                block_dim: (dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("rms_norm launch");
    }

    fn fused_add_rms_norm(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        w: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        // Uses the `_inplace` variant (residual is single in/out buffer).
        // See `kernels/fused_add_rms_norm.cu` for why this variant exists.
        let func = ctx.func(
            "fused_add_rms_norm",
            ptx::FUSED_ADD_RMS_NORM,
            "fused_add_rms_norm_inplace_f16",
        );
        let dim_i32 = dim as i32;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(x);
        b.arg(residual);
        b.arg(w);
        b.arg(out);
        b.arg(&dim_i32);
        b.arg(&eps);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (tokens as u32, 1, 1),
                block_dim: (dim.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("fused_add_rms_norm launch");
    }

    // ── GEMM (cuBLAS hgemm) ─────────────────────────────────────────────
    //
    // Contract: out[m, n] = a[m, k] @ b[n, k]^T, row-major — same as
    // `CpuBackend::gemm` / `crate::cublas::linear_f16`. Transpose flags
    // are fixed: B is transposed, A is not. This matches the Linear /
    // DenseLinear convention where `weight: [n, k]` is stored row-major
    // and we want `out = input @ weight^T`.

    fn gemm(
        ctx: &mut Self::Context,
        a: &Self::Buffer,
        b: &Self::Buffer,
        out: &mut Self::Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        use cudarc::cublas::sys::cublasOperation_t;
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: f16::ONE,
            lda: k as i32,
            ldb: k as i32,
            beta: f16::ZERO,
            ldc: n as i32,
        };
        unsafe { ctx.blas.gemm(cfg, b, a, out) }.expect("gemm (cuBLAS hgemm)");
    }

    // ── Attention ───────────────────────────────────────────────────────
    //
    // Dispatches by q_len:
    //   q_len == 1  → decode_attention_f16  (single-block warp-coop)
    //   q_len >  1  → flash_attn_full_f16   (tiled prefill)
    //
    // Flash-Decoding (split-K) is not wired here yet — it's a long-context
    // decode optimisation that kicks in when valid_kv_len > 256. For
    // correctness-first Phase E, the single-block decode path handles
    // everything up to moderate context lengths; split-K is a perf tune
    // for later.

    fn flash_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k: &Self::Buffer,
        v: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        q_len: usize,
        kv_len: usize,
        pos_offset: usize,
        cfg: &AttnConfig,
    ) {
        if q_len == 1 {
            // Single-token decode path.
            // Kernel signature (decode_attention_f16):
            //   q, k_cache, v_cache, out,
            //   num_q_heads, num_kv_heads, head_dim,
            //   max_kv_len, valid_kv_len, scale
            let func = ctx.func(
                "decode_attention",
                ptx::DECODE_ATTENTION,
                "decode_attention_f16",
            );
            let num_q = cfg.num_heads as i32;
            let num_kv = cfg.num_kv_heads as i32;
            let hd = cfg.head_dim as i32;
            let max_kv = if cfg.kv_seq_stride > 0 {
                cfg.kv_seq_stride as i32
            } else {
                kv_len as i32
            };
            let valid_kv = kv_len as i32;
            let scale = cfg.scale;
            let shared_mem = (kv_len as u32) * 4; // one f32 per KV position
            let stream = ctx.stream.clone();
            let mut b = stream.launch_builder(&func);
            b.arg(q);
            b.arg(k);
            b.arg(v);
            b.arg(out);
            b.arg(&num_q);
            b.arg(&num_kv);
            b.arg(&hd);
            b.arg(&max_kv);
            b.arg(&valid_kv);
            b.arg(&scale);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (cfg.num_heads as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: shared_mem,
                })
            }
            .expect("decode_attention launch");
            return;
        }

        // Prefill / multi-token path: flash_attn_full_f16.
        let func = ctx.func(
            "flash_attn_full",
            ptx::FLASH_ATTN_FULL,
            "flash_attn_full_f16",
        );
        let params = FlashAttnParams {
            batch: batch as i32,
            num_heads: cfg.num_heads as i32,
            num_kv_heads: cfg.num_kv_heads as i32,
            q_len: q_len as i32,
            kv_len: kv_len as i32,
            head_dim: cfg.head_dim as i32,
            causal: if cfg.causal { 1 } else { 0 },
            pos_offset: pos_offset as i32,
            kv_seq_stride: if cfg.kv_seq_stride > 0 {
                cfg.kv_seq_stride as i32
            } else {
                kv_len as i32
            },
        };
        const TILE_Q: usize = 32; // must match kernel TILE_Q
        let num_q_tiles = (q_len + TILE_Q - 1) / TILE_Q;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(q);
        b.arg(k);
        b.arg(v);
        b.arg(out);
        b.arg(&params);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_q_tiles as u32, cfg.num_heads as u32, batch as u32),
                block_dim: (TILE_Q as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("flash_attn_full launch");
    }

    // ── Buffer utilities ────────────────────────────────────────────────

    fn copy_slice(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        src_offset: usize,
        dst: &mut Self::Buffer,
        dst_offset: usize,
        len: usize,
    ) {
        let src_view = src.slice(src_offset..src_offset + len);
        let mut dst_view = dst.slice_mut(dst_offset..dst_offset + len);
        ctx.stream
            .memcpy_dtod(&src_view, &mut dst_view)
            .expect("copy_slice dtod");
    }

    // ── Embedding ───────────────────────────────────────────────────────

    fn embedding_lookup(
        ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &[u32],
        out: &mut Self::Buffer,
        dim: usize,
    ) {
        let batch = ids.len();
        // Upload ids to device (small, at most a few dozen elements).
        let stream = ctx.stream.clone();
        let ids_dev = stream.clone_htod(ids).expect("embedding_lookup ids htod");

        let func = ctx.func(
            "embedding_lookup",
            ptx::EMBEDDING_LOOKUP,
            "embedding_lookup_f16",
        );
        let batch_i32 = batch as i32;
        let dim_i32 = dim as i32;
        let block = 256u32;
        let grid_x = ((dim as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(table);
        b.arg(&ids_dev);
        b.arg(out);
        b.arg(&batch_i32);
        b.arg(&dim_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid_x, batch as u32, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("embedding_lookup launch");
    }

    // ── Transformer-specific fused ops ──────────────────────────────────

    fn split_qkv(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q: &mut Self::Buffer,
        k: &mut Self::Buffer,
        v: &mut Self::Buffer,
        tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) {
        let func = ctx.func("split_qkv", ptx::SPLIT_QKV, "split_qkv_f16");
        let tokens_i32 = tokens as i32;
        let q_dim_i32 = q_dim as i32;
        let kv_dim_i32 = kv_dim as i32;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(qkv);
        b.arg(q);
        b.arg(k);
        b.arg(v);
        b.arg(&tokens_i32);
        b.arg(&q_dim_i32);
        b.arg(&kv_dim_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (tokens as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("split_qkv launch");
    }

    fn fused_silu_mul_split(
        ctx: &mut Self::Context,
        gate_up: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        // gate_up layout: [tokens, 2*im] as [gate | up] per row. Matches
        // the existing `fused_silu_mul_interleaved_f16` kernel exactly.
        let func = ctx.func(
            "fused_silu_mul",
            ptx::FUSED_SILU_MUL,
            "fused_silu_mul_interleaved_f16",
        );
        let im_i32 = im as i32;
        let total = tokens * im;
        let total_i32 = total as i32;
        let block = 256u32;
        let grid = ((total as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(gate_up);
        b.arg(out);
        b.arg(&im_i32);
        b.arg(&total_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("fused_silu_mul_split launch");
    }

    fn qk_norm_rope(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        output: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        mode: i32,
    ) {
        let func = ctx.func(
            "qk_norm_rope",
            ptx::QK_NORM_ROPE,
            "qk_norm_rope_transpose_f16",
        );
        let tokens_i32 = tokens as i32;
        let heads_i32 = heads as i32;
        let head_dim_i32 = head_dim as i32;
        let pos_offset_i32 = pos_offset as i32;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(input);
        b.arg(norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(output);
        b.arg(&tokens_i32);
        b.arg(&heads_i32);
        b.arg(&head_dim_i32);
        b.arg(&pos_offset_i32);
        b.arg(&eps);
        b.arg(&mode);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (tokens as u32, heads as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("qk_norm_rope launch");
    }

    fn kv_cache_append_head_major(
        ctx: &mut Self::Context,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cache_len: usize,
        cache_capacity: usize,
        new_k_head_major: &Self::Buffer,
        new_v_head_major: &Self::Buffer,
        new_tokens: usize,
        nkv: usize,
        hd: usize,
    ) {
        debug_assert!(cache_len + new_tokens <= cache_capacity);

        let func = ctx.func(
            "kv_cache_append",
            ptx::KV_CACHE_APPEND,
            "kv_cache_append_head_major_f16",
        );
        let nkv_i32 = nkv as i32;
        let hd_i32 = hd as i32;
        let cache_len_i32 = cache_len as i32;
        let new_tokens_i32 = new_tokens as i32;
        let cap_i32 = cache_capacity as i32;
        let total = nkv * new_tokens * hd;
        let block = 256u32;
        let grid = ((total as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = ctx.stream.clone();

        // K half.
        {
            let mut b = stream.launch_builder(&func);
            b.arg(cache_k);
            b.arg(new_k_head_major);
            b.arg(&nkv_i32);
            b.arg(&hd_i32);
            b.arg(&cache_len_i32);
            b.arg(&new_tokens_i32);
            b.arg(&cap_i32);
            unsafe { b.launch(cfg) }.expect("kv_cache_append K launch");
        }
        // V half.
        {
            let mut b = stream.launch_builder(&func);
            b.arg(cache_v);
            b.arg(new_v_head_major);
            b.arg(&nkv_i32);
            b.arg(&hd_i32);
            b.arg(&cache_len_i32);
            b.arg(&new_tokens_i32);
            b.arg(&cap_i32);
            unsafe { b.launch(cfg) }.expect("kv_cache_append V launch");
        }
    }

    fn transpose_head_to_token(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        let func = ctx.func("transpose", ptx::TRANSPOSE, "transpose_head_to_token_f16");
        let tokens_i32 = tokens as i32;
        let heads_i32 = heads as i32;
        let dim_i32 = dim as i32;
        let total = tokens * heads * dim;
        let block = 256u32;
        let grid = ((total as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(src);
        b.arg(dst);
        b.arg(&tokens_i32);
        b.arg(&heads_i32);
        b.arg(&dim_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("transpose_head_to_token launch");
    }

    // ── Element-wise ────────────────────────────────────────────────────

    fn add_inplace(
        ctx: &mut Self::Context,
        residual: &mut Self::Buffer,
        x: &Self::Buffer,
        len: usize,
    ) {
        // In-place variant avoids the Rust borrow conflict of aliasing
        // `residual` as both read and write in a single kernel call.
        let func = ctx.func(
            "residual_add",
            ptx::RESIDUAL_ADD,
            "residual_add_inplace_f16",
        );
        let n_i32 = len as i32;
        let block = 256u32;
        let grid = ((len as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(residual);
        b.arg(x);
        b.arg(&n_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("add_inplace (residual_add_inplace) launch");
    }

    fn add_bias(
        ctx: &mut Self::Context,
        data: &mut Self::Buffer,
        bias: &Self::Buffer,
        rows: usize,
        cols: usize,
    ) {
        let func = ctx.func("add_bias", ptx::ADD_BIAS, "add_bias_f16");
        let rows_i32 = rows as i32;
        let cols_i32 = cols as i32;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(data);
        b.arg(bias);
        b.arg(&rows_i32);
        b.arg(&cols_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (cols.min(1024) as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("add_bias launch");
    }

    fn layer_norm(
        ctx: &mut Self::Context,
        x: &Self::Buffer,
        gamma: &Self::Buffer,
        beta: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        let func = ctx.func("layer_norm", ptx::LAYER_NORM, "layer_norm_f16");
        let dim_i32 = dim as i32;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(x);
        b.arg(gamma);
        b.arg(beta);
        b.arg(out);
        b.arg(&dim_i32);
        b.arg(&eps);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (tokens as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("layer_norm launch");
    }

    fn gelu(ctx: &mut Self::Context, x: &Self::Buffer, out: &mut Self::Buffer, len: usize) {
        let func = ctx.func("gelu", ptx::GELU, "gelu_f16");
        let n_i32 = len as i32;
        let block = 256u32;
        let grid = ((len as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(x);
        b.arg(out);
        b.arg(&n_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("gelu launch");
    }

    // ── Quantized GEMM (deferred) ───────────────────────────────────────
    //
    // See top-of-file note: needs mixed-dtype Buffer type to carry int32
    // qweight alongside f16 scales. The Marlin kernel (`crate::marlin`)
    // is already production-grade (112 tok/s on RTX PRO 6000 per pre-v2
    // benchmarks); wiring is a structural concern, not a kernel concern.

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
            "CudaBackend::gemm_quant({kind:?}) not yet wired — Marlin \
             kernel requires mixed-dtype QuantWeights buffers, pending \
             Phase E-GPTQ refactor (see crates/ferrum-kernels/src/backend/cuda.rs \
             module docs)"
        )))
    }

    // ── TP collectives ──────────────────────────────────────────────────
    //
    // World size / rank come from env vars (FERRUM_TP, FERRUM_RANK).
    // The NcclRank group itself is constructed by the executor (which
    // has access to all GPU streams needed for `NcclRank::init_all`).
    // all_reduce is wired through `crate::nccl_comm::NcclRank`; the other
    // two are placeholders until `nccl_comm` gains them (they're not
    // blocking on the LLM decode path — single-rank skips these entirely).

    fn world_size(_ctx: &Self::Context) -> usize {
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

    fn all_reduce(ctx: &mut Self::Context, buf: &mut Self::Buffer, len: usize, op: ReduceOp) {
        // Only Sum is supported for now (the NCCL wrapper is sum-only).
        if !matches!(op, ReduceOp::Sum) {
            tracing::warn!(
                "CudaBackend::all_reduce: op {op:?} not implemented (only Sum); skipping"
            );
            return;
        }
        // Single-rank path: no-op.
        if Self::world_size(ctx) <= 1 {
            return;
        }
        // Multi-rank path: requires the executor to have constructed a
        // shared NcclRank and attached it to thread-local state. The
        // current NcclRank API (`crate::nccl_comm::NcclRank::init_all`) is
        // process-global and we don't want to reach into it from a
        // Backend method. Leaving a runtime warning so misuse surfaces.
        tracing::warn!(
            "CudaBackend::all_reduce: FERRUM_TP > 1 but no NcclRank attached to \
             CudaState — requires executor-level wiring (Phase E-TP)."
        );
    }

    fn all_gather(
        _ctx: &mut Self::Context,
        _local: &Self::Buffer,
        _global: &mut Self::Buffer,
        _local_len: usize,
    ) {
        // Phase E-TP: no NCCL wrapper for all_gather yet.
    }

    fn broadcast(_ctx: &mut Self::Context, _buf: &mut Self::Buffer, _len: usize, _src_rank: usize) {
        // Phase E-TP: no NCCL wrapper for broadcast yet.
    }
}

// ────────────────────────────────────────────────────────────────────────
// Process-global stream for context-free ops
// ────────────────────────────────────────────────────────────────────────
//
// `alloc` / `from_slice` / `to_vec` inherit a no-context signature from
// the Backend trait. cudarc 0.19 hangs all memory APIs off `CudaStream`,
// so we stash an Arc<CudaStream> in a process-global slot populated by
// `new_context`.
//
// Must be process-global (not thread-local): the engine's executor is
// created on one thread (where `new_context` runs) and ops may fire on
// other threads (tokio worker pool, Rayon parallel loops, etc.). A
// thread-local would panic on every worker. cudarc's `stream.alloc()`
// internally calls `ctx.bind_to_thread()` on whichever thread it's
// invoked from, so sharing one stream across threads is safe.

static GLOBAL_STREAM: std::sync::OnceLock<std::sync::RwLock<Option<Arc<CudaStream>>>> =
    std::sync::OnceLock::new();

fn stream_slot() -> &'static std::sync::RwLock<Option<Arc<CudaStream>>> {
    GLOBAL_STREAM.get_or_init(|| std::sync::RwLock::new(None))
}

fn with_stream<R>(f: impl FnOnce(&Arc<CudaStream>) -> R) -> R {
    let guard = stream_slot().read().expect("GLOBAL_STREAM poisoned");
    let stream = guard
        .as_ref()
        .cloned()
        .expect("CudaBackend: GLOBAL_STREAM not set — call new_context() first");
    drop(guard);
    f(&stream)
}

/// Install a stream as the global default for context-free ops.
/// Called automatically by `new_context`; override manually from worker
/// threads that need to pin a different stream.
pub fn install_thread_stream(stream: Arc<CudaStream>) {
    *stream_slot().write().expect("GLOBAL_STREAM poisoned") = Some(stream);
}
