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
use cudarc::cublas::CudaBlas;
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
    /// Pre-allocated cuBLAS workspace (32 MB) — held so cuBLAS doesn't
    /// malloc per-GEMM. The old CudaDecodeRunner had this and it's one
    /// of the cheap perf wins for decode-heavy workloads.
    _blas_workspace: CudaSlice<u8>,
    modules: HashMap<&'static str, Arc<CudaModule>>,

    // ── Device-side dynamic decode state (for CUDA graph capture) ──
    // Kernels that depend on per-step values (token id, position,
    // valid-kv-len) have `_dyn` variants reading these device slots.
    // When `use_dev_state = true`, the `_dyn` kernels are launched; the
    // captured graph sees stable pointer args and the scalar values get
    // updated via memcpy_htod_async before each graph replay.
    pub token_buf: CudaSlice<u32>,
    pub pos_buf: CudaSlice<i32>, // pos_offset == cache_len for decode
    pub kv_buf: CudaSlice<i32>,  // valid_kv_len == pos + 1
    pub use_dev_state: bool,

    /// True between begin_graph_capture and end_graph_capture.
    pub capture_in_flight: bool,
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
        // Reuse the process-global stream populated by `default_stream()`.
        // Model constructors call `B::from_slice` thousands of times to
        // upload weights BEFORE the engine ever calls `new_context()`, so
        // `default_stream()` has already lazily spun up a stream. Reusing
        // it here keeps allocations + ops on the SAME stream — no
        // cross-stream synchronization needed.
        let stream = default_stream();
        let ctx = stream.context().clone();
        let blas = Arc::new(
            CudaBlas::new(stream.clone())
                .unwrap_or_else(|e| panic!("CudaBackend::new_context: CudaBlas::new: {e}")),
        );
        // Pre-allocate a 32 MB cuBLAS workspace and pin it to the handle.
        // Without this, cuBLAS internally mallocs a workspace per call on
        // some algorithms (visible as ~1-2ms overhead per GEMM on Blackwell).
        const BLAS_WS_BYTES: usize = 32 * 1024 * 1024;
        let blas_workspace = unsafe { stream.alloc::<u8>(BLAS_WS_BYTES) }
            .expect("CudaBackend::new_context: cuBLAS workspace alloc");
        unsafe {
            use cudarc::cublas::sys;
            use cudarc::driver::DevicePtr;
            let (ws_ptr, _guard) = blas_workspace.device_ptr(&stream);
            let status = sys::cublasSetWorkspace_v2(
                *blas.handle(),
                ws_ptr as *mut _,
                BLAS_WS_BYTES,
            );
            if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                panic!("cublasSetWorkspace_v2 failed: {:?}", status);
            }
        }
        // Per-step dynamic state (for graph replay).
        let token_buf = unsafe { stream.alloc::<u32>(1) }
            .expect("CudaBackend::new_context: token_buf alloc");
        let pos_buf = unsafe { stream.alloc::<i32>(1) }
            .expect("CudaBackend::new_context: pos_buf alloc");
        let kv_buf = unsafe { stream.alloc::<i32>(1) }
            .expect("CudaBackend::new_context: kv_buf alloc");

        Self::Context {
            ctx,
            stream,
            blas,
            _blas_workspace: blas_workspace,
            modules: HashMap::new(),
            token_buf,
            pos_buf,
            kv_buf,
            use_dev_state: false,
            capture_in_flight: false,
        }
    }

    fn set_decode_state(ctx: &mut Self::Context, token: u32, step: u32) {
        let valid_kv = (step as i32) + 1;
        let step_i = step as i32;
        ctx.stream
            .memcpy_htod(&[token], &mut ctx.token_buf)
            .expect("token_buf memcpy");
        ctx.stream
            .memcpy_htod(&[step_i], &mut ctx.pos_buf)
            .expect("pos_buf memcpy");
        ctx.stream
            .memcpy_htod(&[valid_kv], &mut ctx.kv_buf)
            .expect("kv_buf memcpy");
    }

    fn set_dev_state_mode(ctx: &mut Self::Context, enable: bool) {
        ctx.use_dev_state = enable;
    }

    fn begin_graph_capture(ctx: &mut Self::Context) -> Result<()> {
        use cudarc::driver::sys::CUstreamCaptureMode;
        // Disable cudarc's per-slice event tracking during capture —
        // without this, CUDA_ERROR_STREAM_CAPTURE_ISOLATION fires because
        // weight buffers carry event handles from pre-capture htod calls.
        // Matches what the old CudaDecodeRunner does around graph capture.
        unsafe {
            ctx.ctx.disable_event_tracking();
        }
        ctx.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| FerrumError::unsupported(format!("begin_capture: {e}")))?;
        ctx.capture_in_flight = true;
        Ok(())
    }

    fn end_graph_capture(ctx: &mut Self::Context) -> Result<()> {
        use cudarc::driver::sys::CUgraphInstantiate_flags;
        if !ctx.capture_in_flight {
            return Err(FerrumError::unsupported("end_capture without begin"));
        }
        ctx.capture_in_flight = false;
        let graph_opt = ctx
            .stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| {
                // Re-enable event tracking even on error to avoid permanent disable.
                unsafe { ctx.ctx.enable_event_tracking() };
                FerrumError::unsupported(format!("end_capture: {e}"))
            })?;
        // Re-enable event tracking for subsequent non-captured work.
        unsafe {
            ctx.ctx.enable_event_tracking();
        }
        match graph_opt {
            Some(g) => {
                install_decode_graph(g);
                Ok(())
            }
            None => Err(FerrumError::unsupported(
                "end_capture returned None (empty graph?)",
            )),
        }
    }

    fn replay_last_graph(_ctx: &mut Self::Context) -> Result<bool> {
        with_decode_graph(|g_opt| {
            if let Some(g) = g_opt {
                g.launch()
                    .map_err(|e| FerrumError::unsupported(format!("graph.launch: {e}")))?;
                Ok(true)
            } else {
                Ok(false)
            }
        })
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
        use cudarc::cublas::result::gemm_ex;
        use cudarc::cublas::sys::{
            cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t,
        };
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let (a_ptr, _rec_a) = b.device_ptr(&ctx.stream); // cuBLAS arg "A" = weight = our `b`
        let (b_ptr, _rec_b) = a.device_ptr(&ctx.stream); // cuBLAS arg "B" = input = our `a`
        let (c_ptr, _rec_c) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            gemm_ex(
                *ctx.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                (&alpha) as *const f32 as *const _,
                a_ptr as *const _,
                cudaDataType_t::CUDA_R_16F,
                k as i32,
                b_ptr as *const _,
                cudaDataType_t::CUDA_R_16F,
                k as i32,
                (&beta) as *const f32 as *const _,
                c_ptr as *mut _,
                cudaDataType_t::CUDA_R_16F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .expect("gemm (cublasGemmEx, compute=32F_FAST_16F, algo=TENSOR_OP)");
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
        // Dispatch by q_len:
        //   q_len == 1  → `decode_attention_head_major_f16` — single-block
        //                 warp-cooperative for head-major cache (fast decode).
        //   q_len >  1  → `flash_attn_full_f16` — tiled prefill (TILE_Q=32).
        //
        // Both kernels read the cache as HEAD-MAJOR `[nkv, capacity, hd]`
        // matching `kv_cache_append_head_major_f16`'s write layout.
        if q_len == 1 {
            let use_dyn = ctx.use_dev_state;
            let func_name = if use_dyn {
                "decode_attention_head_major_f16_dyn"
            } else {
                "decode_attention_head_major_f16"
            };
            let func = ctx.func("decode_attention_hm", ptx::DECODE_ATTENTION_HM, func_name);
            let num_q = cfg.num_heads as i32;
            let num_kv = cfg.num_kv_heads as i32;
            let hd = cfg.head_dim as i32;
            let capacity = if cfg.kv_seq_stride > 0 {
                cfg.kv_seq_stride as i32
            } else {
                kv_len as i32
            };
            let valid_kv_scalar = kv_len as i32;
            let scale = cfg.scale;
            let shared_mem = (kv_len as u32) * 4;
            let stream = ctx.stream.clone();
            let mut bld = stream.launch_builder(&func);
            bld.arg(q);
            bld.arg(k);
            bld.arg(v);
            bld.arg(out);
            bld.arg(&num_q);
            bld.arg(&num_kv);
            bld.arg(&hd);
            bld.arg(&capacity);
            if use_dyn {
                bld.arg(&ctx.kv_buf);
            } else {
                bld.arg(&valid_kv_scalar);
            }
            bld.arg(&scale);
            unsafe {
                bld.launch(LaunchConfig {
                    grid_dim: (cfg.num_heads as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: shared_mem,
                })
            }
            .expect("decode_attention_head_major launch");
            return;
        }
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
        let dim_i32 = dim as i32;
        let block = 256u32;
        let grid_x = ((dim as u32) + block - 1) / block;

        if ctx.use_dev_state {
            // Graph-friendly: read token id from device state buffer.
            // Limited to batch=1 (decode path). Prefill uses the scalar path.
            debug_assert!(ids.len() == 1, "dev_state embedding requires batch=1");
            let func = ctx.func(
                "embedding_lookup",
                ptx::EMBEDDING_LOOKUP,
                "embedding_lookup_f16_dyn",
            );
            let stream = ctx.stream.clone();
            let mut b = stream.launch_builder(&func);
            b.arg(table);
            b.arg(&ctx.token_buf);
            b.arg(out);
            b.arg(&dim_i32);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (block, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .expect("embedding_lookup_dyn launch");
            return;
        }

        let batch = ids.len();
        let stream = ctx.stream.clone();
        let ids_dev = stream.clone_htod(ids).expect("embedding_lookup ids htod");

        let func = ctx.func(
            "embedding_lookup",
            ptx::EMBEDDING_LOOKUP,
            "embedding_lookup_f16",
        );
        let batch_i32 = batch as i32;
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
        let use_dyn = ctx.use_dev_state && tokens == 1;
        let fn_name = if use_dyn {
            "qk_norm_rope_transpose_f16_dyn"
        } else {
            "qk_norm_rope_transpose_f16"
        };
        let func = ctx.func("qk_norm_rope", ptx::QK_NORM_ROPE, fn_name);
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
        if use_dyn {
            b.arg(&ctx.pos_buf);
        } else {
            b.arg(&pos_offset_i32);
        }
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

        let use_dyn = ctx.use_dev_state && new_tokens == 1;
        let fn_name = if use_dyn {
            "kv_cache_append_head_major_f16_dyn"
        } else {
            "kv_cache_append_head_major_f16"
        };
        let func = ctx.func("kv_cache_append", ptx::KV_CACHE_APPEND, fn_name);
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
            if use_dyn {
                b.arg(&ctx.pos_buf);
            } else {
                b.arg(&cache_len_i32);
            }
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
            if use_dyn {
                b.arg(&ctx.pos_buf);
            } else {
                b.arg(&cache_len_i32);
            }
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

/// Return the global stream, lazily creating it if neither `new_context`
/// nor `install_thread_stream` has populated it yet.
///
/// This is needed because `LlamaFamilyModel::new()` (and other model
/// constructors) call `B::from_slice` on thousands of weights before
/// any engine code creates a `Context`. We can't easily force ordering
/// through trait signatures, so the first `from_slice` lazily spins up
/// a default context (ordinal from `FERRUM_CUDA_DEVICE` env, else 0)
/// and a dedicated stream. Subsequent `new_context()` calls reuse this
/// same stream — no divergence.
fn default_stream() -> Arc<CudaStream> {
    if let Some(s) = stream_slot()
        .read()
        .expect("GLOBAL_STREAM poisoned")
        .as_ref()
    {
        return s.clone();
    }
    let mut w = stream_slot().write().expect("GLOBAL_STREAM poisoned");
    if w.is_none() {
        let ordinal = std::env::var("FERRUM_CUDA_DEVICE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let ctx = CudaContext::new(ordinal).unwrap_or_else(|e| {
            panic!("CudaBackend: failed to init default context {ordinal}: {e}")
        });
        let stream = ctx
            .new_stream()
            .unwrap_or_else(|e| panic!("CudaBackend: failed to create default stream: {e}"));
        *w = Some(stream);
    }
    w.as_ref().cloned().expect("just inserted")
}

fn with_stream<R>(f: impl FnOnce(&Arc<CudaStream>) -> R) -> R {
    let stream = default_stream();
    f(&stream)
}

/// Install a stream as the global default for context-free ops.
/// `new_context` calls this to make its freshly-created stream the
/// process default — subsequent `alloc`/`from_slice`/`to_vec` calls
/// route through it. If `default_stream` already lazily created a
/// stream, this replaces it.
pub fn install_thread_stream(stream: Arc<CudaStream>) {
    *stream_slot().write().expect("GLOBAL_STREAM poisoned") = Some(stream);
}

// ────────────────────────────────────────────────────────────────────────
// Process-global decode graph slot
// ────────────────────────────────────────────────────────────────────────
//
// Stored here (not on `CudaState`) because:
//   - Backend::Context isn't Send+Sync for all backends (Metal holds a
//     raw CommandBufferRef) — the model struct gets Send issues if ctx
//     is stored on it.
//   - Only CUDA uses graph capture, so global-per-process is fine.
//   - Kernel arg pointers captured in the graph reference model-owned
//     scratch buffers; the model outlives any graph, so no dangling refs.
//
// `CudaGraph` isn't automatically `Send+Sync` in cudarc's public API —
// we wrap in our own marker struct with `unsafe impl`. The stream itself
// is single-threaded per model (engine serialises via Mutex), so graph
// launch from the same thread is safe.

struct GraphSlot(cudarc::driver::CudaGraph);
// SAFETY: graph launch is serialised through the model's stream, which
// is accessed from one thread at a time (engine Mutex-wraps the model).
unsafe impl Send for GraphSlot {}
unsafe impl Sync for GraphSlot {}

static DECODE_GRAPH: std::sync::OnceLock<std::sync::RwLock<Option<GraphSlot>>> =
    std::sync::OnceLock::new();

fn graph_slot() -> &'static std::sync::RwLock<Option<GraphSlot>> {
    DECODE_GRAPH.get_or_init(|| std::sync::RwLock::new(None))
}

fn install_decode_graph(g: cudarc::driver::CudaGraph) {
    *graph_slot().write().expect("DECODE_GRAPH poisoned") = Some(GraphSlot(g));
}

fn with_decode_graph<R>(
    f: impl FnOnce(Option<&cudarc::driver::CudaGraph>) -> Result<R>,
) -> Result<R> {
    let guard = graph_slot().read().expect("DECODE_GRAPH poisoned");
    f(guard.as_ref().map(|s| &s.0))
}

/// Evict the cached graph — call this when model weights/scratch pointers
/// change (e.g. scratch resize, model reload). Next decode will re-capture.
pub fn invalidate_decode_graph() {
    *graph_slot().write().expect("DECODE_GRAPH poisoned") = None;
}
