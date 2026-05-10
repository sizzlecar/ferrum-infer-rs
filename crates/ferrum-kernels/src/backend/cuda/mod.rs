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

#![allow(unused_variables, dead_code, unused_imports, unused_mut)]

// Submodules. Per-supertrait files live under `cuda/`; the main module
// keeps `impl Backend for CudaBackend` (the core trait) plus the
// supertraits whose impls haven't been extracted yet.
//
//   Phase 1 (#8): INT8 KV (`BackendInt8KvOps` + helpers + `KvCacheQuant`
//     constructor) → `cuda/int8_kv.rs`.
//   Phase 2: `BackendCollective` → `cuda/collective.rs`. `BackendGraph` +
//     `GraphSlotRaw` + `DECODE_GRAPHS` helpers → `cuda/graph.rs`.
//   Phase 3: `BackendQuantMarlin` + `BackendQuantGguf` (incl.
//     `GptqStoreCuda`, `marlin_gemm_with_perm`, `launch_vllm_marlin`,
//     `MarlinGatherScratch`, `moe_gemm_phase_fused_impl`) → `cuda/quant.rs`.
//   Future phases: BackendPagedKv, BackendMoeFused.
pub mod collective;
pub mod graph;
pub mod int8_kv;
pub mod quant;
pub use int8_kv::{OptionalCudaInt8, OptionalCudaScalesF16};
// Preserve historical `crate::backend::cuda::*` paths used by external
// callers (`quant_linear::cuda_marlin`, parity tests).
pub use quant::{marlin_gemm_with_perm, GptqStoreCuda};
#[cfg(feature = "marlin")]
pub use quant::pregrow_marlin_gather_scratch;

use super::{
    AttnConfig, Backend, BackendCollective, BackendGraph, BackendMoeFused, BackendPagedKv,
    BackendQuantGguf, BackendQuantMarlin, QuantKind, QuantWeights, ReduceOp,
};
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
    /// Shared cuBLAS handle (process-global, initialised once). Graph
    /// capture records pointers to the cuBLAS workspace; a per-ctx
    /// workspace would dangle after ctx drop → CUDA_ERROR_INVALID_VALUE
    /// at next sync. Share the handle so the workspace outlives captures.
    pub blas: Arc<CudaBlas>,
    modules: HashMap<&'static str, Arc<CudaModule>>,
    pub use_dev_state: bool,
    /// True between begin_graph_capture and end_graph_capture.
    pub capture_in_flight: bool,
    /// Stable scratch buffers for batched-decode kernels that take per-item
    /// device-pointer arrays (flash_attn_batched, kv_cache_append_batched).
    /// Per-call `alloc_zeros::<T>(m)` was 3 allocs × 32 layers × 3 ops
    /// = ~96 allocs/token. Caching here saves the allocator overhead AND
    /// keeps the buffer addresses stable across calls — required so a
    /// future CUDA-graph capture can replay over them.
    /// Allocated lazily to `BATCHED_SCRATCH_CAP` (covers max_seqs ≤ 64).
    batched_scratch_u64_k: Option<CudaSlice<u64>>,
    batched_scratch_u64_v: Option<CudaSlice<u64>>,
    batched_scratch_u64_cache: Option<CudaSlice<u64>>,
    batched_scratch_i32_kv_lens: Option<CudaSlice<i32>>,
    batched_scratch_i32_cache_lens: Option<CudaSlice<i32>>,
    /// Stable HOST-side staging buffers for the per-item u64/i32 arrays
    /// fed into the device-side scratch via `stream.memcpy_htod`. The
    /// memcpy is async and gets recorded in any active CUDA-graph
    /// capture, which captures the HOST POINTER. If we used a local
    /// Vec, that Vec drops at function return → captured graph holds a
    /// dangling host pointer → replay reads garbage → MISALIGNED at
    /// later kernel. Owning these on the long-lived CudaState keeps
    /// the host pointers stable, and clearing+re-filling between calls
    /// updates the contents the captured memcpy reads on each replay.
    /// Fixed-size arrays (not Vec) avoid the realloc-invalidates-ptr trap.
    batched_host_k_ptrs: Box<[u64; HOST_STAGING_TOTAL]>,
    batched_host_v_ptrs: Box<[u64; HOST_STAGING_TOTAL]>,
    batched_host_cache_ptrs: Box<[u64; HOST_STAGING_TOTAL]>,
    batched_host_kv_lens: Box<[i32; HOST_STAGING_TOTAL]>,
    batched_host_cache_lens: Box<[i32; HOST_STAGING_TOTAL]>,
    /// Stream pool for parallel MoE expert dispatch. At c=32 with 128
    /// active experts, ~256 sequential Marlin GEMMs/layer hit launch +
    /// SM-allocation serialization. Round-robin across N streams lets
    /// multiple Marlin kernels overlap; only valid for small-m where
    /// each kernel uses a fraction of available SMs. Lazy-init on first
    /// MoE call.
    moe_streams: Option<Vec<Arc<CudaStream>>>,
    /// Persistent cuEvents for `moe_gemm_phase_batched` cross-stream
    /// sync. `moe_entry_event` is recorded on default → waited on each
    /// pool stream; `moe_exit_events[i]` is recorded on pool stream i →
    /// waited on default. Per-call reuse (record/wait only) replaces
    /// the per-call create/destroy pair: at c=32 / 48 layers / 2 phases
    /// that's ~960 driver calls saved per token. Lazy-init alongside
    /// `moe_streams`.
    ///
    /// Stored as raw `CUevent` (= `*mut CUevent_st`); the pointer is
    /// owned by CUDA's driver, not Rust's allocator, so we just hold
    /// the handle and call `cuEventDestroy_v2` in `Drop`.
    moe_entry_event: Option<usize>,
    moe_exit_events: Option<Vec<usize>>,
    /// GPU-side route output scratch. Device buffers sized to
    /// `MAX_ROUTE_PAIRS` (= 32 batch × 8 top_k = 256 by default — covers
    /// every Qwen3-MoE config we ship). Lazy-init on first
    /// `try_gpu_route_topk_into_host` call. Kept as f16 storage since
    /// `Buffer = CudaSlice<f16>`; the kernel writes raw int / float
    /// bytes via reinterpret-cast.
    moe_route_ids: Option<CudaSlice<f16>>,
    moe_route_weights: Option<CudaSlice<f16>>,
    /// Capacity hint — buffers grow if a larger (batch × top_k) shows
    /// up. Reset on grow.
    moe_route_capacity: usize,
    /// Cached scratch for paged_decode_attention's prefill path: holds
    /// the token-major attn output before transpose-back to head-major.
    /// Lazy-grow on first use. Caching prevents per-call alloc churn
    /// that triggered CUDA_ERROR_ILLEGAL_ADDRESS via stream-ordered free.
    paged_attn_out_tm: Option<CudaSlice<f16>>,
    paged_attn_out_tm_capacity: usize,
    /// fp32 reduce scratch for vLLM marlin_moe_wna16. Sized at the upper
    /// bound vLLM uses internally: `sms * 4 * moe_block_size * max_thread_n`
    /// (= 128 * 4 * 16 * 256 = 2M fp32 = 8MB on a 4090). Lazy-alloc; once
    /// up the buffer is reused across all layers and forwards. Without
    /// this scratch we fall back to atomic_add which is 1.3-1.5× slower.
    #[cfg(feature = "vllm-moe-marlin")]
    vllm_moe_c_tmp_f32: Option<CudaSlice<f32>>,
}

const BATCHED_SCRATCH_CAP: usize = 64;
/// Number of distinct call sites per token-step that may be captured
/// inside one CUDA graph: `cache_ptrs` is shared by K-append AND V-append
/// calls (so 2 × MAX_LAYERS_FOR_GRAPH); `k_ptrs`/`v_ptrs` are used once
/// per layer (so 1 × MAX_LAYERS_FOR_GRAPH each). Sizing every host
/// staging array to `2 × MAX_LAYERS_FOR_GRAPH` simplifies indexing —
/// we waste a few KB to keep call sites uniform. Each captured
/// `stream.memcpy_htod` reads from a non-overlapping host slice — the
/// graph records the host pointer, so two memcpys sharing the same
/// region read each other's latest write on replay (verified bug:
/// `cudarc_graph_shared_host_array_multi_memcpy`).
const MAX_GRAPH_SLOTS: usize = 2 * super::MAX_LAYERS_FOR_GRAPH;
/// Total size of the per-call host staging arrays for graph capture.
const HOST_STAGING_TOTAL: usize = MAX_GRAPH_SLOTS * BATCHED_SCRATCH_CAP;

impl CudaState {
    /// Lazy-init the MoE stream pool on first access. Pool size is
    /// 4 by default; override via FERRUM_MOE_STREAMS env (1 disables
    /// multi-stream dispatch).
    pub fn moe_stream_pool(&mut self) -> &[Arc<CudaStream>] {
        if self.moe_streams.is_none() {
            let n = std::env::var("FERRUM_MOE_STREAMS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4)
                .max(1);
            let mut pool = Vec::with_capacity(n);
            for _ in 0..n {
                let s = self
                    .ctx
                    .new_stream()
                    .expect("CudaState::moe_stream_pool: new_stream failed");
                pool.push(s);
            }
            tracing::info!("MoE stream pool initialized: {} streams", n);
            self.moe_streams = Some(pool);
        }
        self.moe_streams.as_ref().unwrap()
    }

    /// Lazy-alloc the fp32 reduce scratch buffer used by
    /// `moe_gemm_phase_vllm`. Sized at vLLM's static upper bound:
    /// `sms * 4 * moe_block_size * max_thread_n` (= 128 * 4 * 16 * 256 =
    /// 2M fp32 = 8MB on RTX 4090). Once allocated it's reused for every
    /// vLLM moe call — phase 1, phase 3, every layer, every forward.
    #[cfg(feature = "vllm-moe-marlin")]
    pub fn vllm_moe_c_tmp(&mut self) -> &mut CudaSlice<f32> {
        if self.vllm_moe_c_tmp_f32.is_none() {
            // Static upper bound: covers any (size_n, total_padded) pair
            // we'd ever feed under our shapes (gate_up_dim ≤ 16k,
            // total_padded ≤ num_experts * block_size = 128 * 16 = 2048).
            // 2M f32 = 8 MB.
            const C_TMP_SIZE_F32: usize = 2 * 1024 * 1024;
            let buf = self
                .stream
                .alloc_zeros::<f32>(C_TMP_SIZE_F32)
                .expect("alloc_zeros vllm_moe_c_tmp_f32");
            self.vllm_moe_c_tmp_f32 = Some(buf);
            tracing::info!(
                "vLLM moe c_tmp scratch allocated: {} fp32 ({:.1} MB)",
                C_TMP_SIZE_F32,
                (C_TMP_SIZE_F32 * 4) as f32 / 1e6
            );
        }
        self.vllm_moe_c_tmp_f32.as_mut().unwrap()
    }

    /// Lazy-init the persistent cuEvents used by
    /// `moe_gemm_phase_batched` for cross-stream sync. The (entry,
    /// exits) tuple is stable across calls — events are only ever
    /// recorded / waited on, never destroyed (until `Drop`).
    pub fn moe_sync_events(
        &mut self,
    ) -> (
        cudarc::driver::sys::CUevent,
        Vec<cudarc::driver::sys::CUevent>,
    ) {
        use cudarc::driver::sys as cu;
        if self.moe_entry_event.is_none() {
            let n = self.moe_stream_pool().len();
            let mut entry: cu::CUevent = std::ptr::null_mut();
            unsafe {
                // CU_EVENT_DISABLE_TIMING (= 2) — fastest event create,
                // no GPU timestamp tracking. Required for sync only.
                cu::cuEventCreate(&mut entry, 2);
            }
            let mut exits: Vec<usize> = Vec::with_capacity(n);
            for _ in 0..n {
                let mut e: cu::CUevent = std::ptr::null_mut();
                unsafe {
                    cu::cuEventCreate(&mut e, 2);
                }
                exits.push(e as usize);
            }
            self.moe_entry_event = Some(entry as usize);
            self.moe_exit_events = Some(exits);
            tracing::info!("MoE sync events initialized: 1 entry + {} exits", n);
        }
        let entry = self.moe_entry_event.unwrap() as cu::CUevent;
        let exits: Vec<cu::CUevent> = self
            .moe_exit_events
            .as_ref()
            .unwrap()
            .iter()
            .map(|&p| p as cu::CUevent)
            .collect();
        (entry, exits)
    }

    fn module(&mut self, key: &'static str, ptx_src: &str) -> Arc<CudaModule> {
        if let Some(m) = self.modules.get(key) {
            return m.clone();
        }
        // Route through process-global cache — keeps Arc<CudaModule>
        // alive forever so captured CUfunction handles never go stale
        // even after this CudaState drops.
        let m = ensure_module(&self.ctx, key, ptx_src);
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
    type GptqStore = GptqStoreCuda;
    type QuantStore = (); // not yet wired on CUDA — load_quant / gemm_quant return unsupported

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
        // Process-global blas handle + workspace. Critical for graph capture:
        // the captured kernel args include the workspace pointer, which must
        // outlive the ctx that owned it at capture time.
        let blas = ensure_blas_handle(&stream);
        // Ensure process-global decode state buffers exist.
        ensure_decode_state_bufs(&stream);
        // Process-global batched-scratch device + host arrays. SAME
        // graph-capture lifetime requirement as cuBLAS workspace above:
        // captured stream.memcpy_htod records the host array address;
        // captured kernel arg holds the device scratch address; both
        // must outlive every CudaState that triggers a capture+replay.
        ensure_batched_scratch(&stream);

        // Disable cudarc's per-slice event tracking globally. We run everything
        // on one stream → CUDA stream semantics handle ordering natively.
        // Critical for graph capture: without this, the post-capture `to_vec`
        // dtoh sync hits cuStreamWaitEvent on events that were recorded during
        // pre-capture weight htods and are stale after replay.
        unsafe {
            ctx.disable_event_tracking();
        }

        Self::Context {
            ctx,
            stream,
            blas,
            modules: HashMap::new(),
            use_dev_state: false,
            capture_in_flight: false,
            batched_scratch_u64_k: None,
            batched_scratch_u64_v: None,
            batched_scratch_u64_cache: None,
            batched_scratch_i32_kv_lens: None,
            batched_scratch_i32_cache_lens: None,
            batched_host_k_ptrs: Box::new([0u64; HOST_STAGING_TOTAL]),
            batched_host_v_ptrs: Box::new([0u64; HOST_STAGING_TOTAL]),
            batched_host_cache_ptrs: Box::new([0u64; HOST_STAGING_TOTAL]),
            batched_host_kv_lens: Box::new([0i32; HOST_STAGING_TOTAL]),
            batched_host_cache_lens: Box::new([0i32; HOST_STAGING_TOTAL]),
            moe_streams: None,
            moe_entry_event: None,
            moe_exit_events: None,
            moe_route_ids: None,
            moe_route_weights: None,
            moe_route_capacity: 0,
            paged_attn_out_tm: None,
            paged_attn_out_tm_capacity: 0,
            #[cfg(feature = "vllm-moe-marlin")]
            vllm_moe_c_tmp_f32: None,
        }
    }

    fn alloc_u32(n: usize) -> Self::Buffer {
        // Buffer storage is f16 (2 bytes per element). For n u32 slots
        // we need n*4 bytes = 2*n f16 elements. The trait default
        // collapses f32→f16 one-for-one which under-allocates (buffer
        // is HALF the bytes the kernel expects), causing writes into
        // un-mapped pool memory and CUDA_ERROR_MISALIGNED_ADDRESS at
        // sync. Override to allocate the right byte count.
        let n = n.max(1);
        with_stream(|stream| {
            stream
                .alloc_zeros::<f16>(2 * n)
                .expect("CudaBackend::alloc_u32: alloc_zeros<f16>")
        })
    }

    fn write_u32(ctx: &mut Self::Context, dst: &mut Self::Buffer, data: &[u32]) {
        // Synchronous host→device write of int32 values. Used by callers
        // to populate device-side scratch buffers (positions, kv_lens)
        // BEFORE a CUDA-graph replay so the buffer's contents are
        // current when the replay re-runs the kernels that read them.
        //
        // - Synchronous (`cuMemcpyHtoD_v2`, not `..Async`) so the local
        //   host Vec stays alive across the copy. Async memcpy on a
        //   captured stream would record a stale host pointer.
        // - Explicit `bind_to_thread` because tokio shifts decode_batch
        //   calls across worker threads. Without it, calls landing on a
        //   thread that hadn't bound the context fail with
        //   `CUDA_ERROR_INVALID_CONTEXT` — observed after graph capture
        //   activated and the next call ran on a fresh worker.
        if data.is_empty() {
            return;
        }
        if let Err(e) = ctx.ctx.bind_to_thread() {
            eprintln!("write_u32 bind_to_thread failed: {e}");
            return;
        }
        let stream = ctx.stream.clone();
        let host_i32: Vec<i32> = data.iter().map(|&x| x as i32).collect();
        use cudarc::driver::DevicePtrMut;
        let (dst_ptr, _g) = dst.device_ptr_mut(&stream);
        unsafe {
            let st = cudarc::driver::sys::cuMemcpyHtoD_v2(
                dst_ptr,
                host_i32.as_ptr() as *const std::ffi::c_void,
                host_i32.len() * std::mem::size_of::<i32>(),
            );
            if st != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                eprintln!("write_u32 cuMemcpyHtoD failed: {st:?}");
            }
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
            // cudarc asserts host.len() >= buf.len() — but we may want a
            // PARTIAL read (len < buf capacity), e.g. reading only 4 rows
            // out of a batch_logits buffer sized for max_batch. Slice the
            // device buffer so its reported length matches `len`.
            let view = buf.slice(0..len);
            stream.memcpy_dtoh(&view, &mut host).expect("cuda dtoh");
            stream.synchronize().expect("cuda dtoh sync");
            host.into_iter().map(|x| x.to_f32()).collect()
        })
    }

    /// Greedy fast path: argmax on device, return n_rows u32 indices.
    /// Replaces ~n_rows × vocab × 2 bytes DTOH + host argmax with a
    /// single launch + n_rows × 4 bytes DTOH.
    fn argmax_rows_to_u32(buf: &Self::Buffer, n_rows: usize, vocab: usize) -> Vec<u32> {
        if n_rows == 0 {
            return Vec::new();
        }
        with_stream(|stream| {
            let ctx = stream.context();
            let module = ensure_module(ctx, "argmax_rows", ptx::ARGMAX_ROWS);
            let func = module
                .load_function("argmax_rows_f16")
                .expect("argmax_rows_f16 load_function");

            let mut dev_out: cudarc::driver::CudaSlice<i32> = stream
                .alloc_zeros::<i32>(n_rows)
                .expect("argmax dev_out alloc");

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (n_rows as u32, 1, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            let view = buf.slice(0..(n_rows * vocab));
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&view)
                    .arg(&mut dev_out)
                    .arg(&(n_rows as i32))
                    .arg(&(vocab as i32))
                    .launch(cfg)
                    .expect("argmax launch");
            }
            stream.synchronize().expect("argmax sync");
            let host_i32 = stream.clone_dtoh(&dev_out).expect("argmax dtoh");
            host_i32.into_iter().map(|x| x as u32).collect()
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

        // cuBLAS is set to CUBLAS_POINTER_MODE_DEVICE (see ensure_blas_handle)
        // so alpha/beta are read from device memory. Using the process-global
        // alpha_f32/beta_f32 slices keeps pointers stable for graph capture.
        let (a_ptr, _rec_a) = b.device_ptr(&ctx.stream); // cuBLAS arg "A" = weight = our `b`
        let (b_ptr, _rec_b) = a.device_ptr(&ctx.stream); // cuBLAS arg "B" = input = our `a`
        let (c_ptr, _rec_c) = out.device_ptr_mut(&ctx.stream);
        let blas_guard = blas_slot().read().expect("BLAS poisoned");
        let slot = blas_guard.as_ref().expect("BLAS not init");
        let (alpha_ptr, _ga) = slot.alpha_f32.device_ptr(&ctx.stream);
        let (beta_ptr, _gb) = slot.beta_f32.device_ptr(&ctx.stream);

        unsafe {
            gemm_ex(
                *ctx.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                alpha_ptr as *const _,
                a_ptr as *const _,
                cudaDataType_t::CUDA_R_16F,
                k as i32,
                b_ptr as *const _,
                cudaDataType_t::CUDA_R_16F,
                k as i32,
                beta_ptr as *const _,
                c_ptr as *mut _,
                cudaDataType_t::CUDA_R_16F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .expect("gemm (cublasGemmEx, compute=32F_FAST_16F, algo=TENSOR_OP)");
        // blas_guard dropped at scope end — `_ga`/`_gb` borrow from it.
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
            // Opt the kernel into Blackwell's full per-SM dynamic shared
            // memory (up to 228 KB). The default cap is 48 KB which is
            // smaller than the `capacity * 4` bytes we bake into the
            // captured graph for long max_seq_len models (Qwen3 = 160 KB).
            // Without this, graph launch fails with CUDA_ERROR_INVALID_VALUE.
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
            // Shared-memory sizing (graph-safe):
            // - Kernel writes `s_scores[0..valid_kv_len]` per step. valid_kv_len
            //   grows over time; captured graph has a fixed shared_mem_bytes.
            // - Must bake in a size that covers max expected kv_len during
            //   decode. Kernel also uses ~8 bytes of static shared mem
            //   (s_block_max / s_block_sum), so we can't claim the full 48 KB
            //   default limit. 32 KB = 8192 positions is a safe default with
            //   no cudaFuncSetAttribute opt-in needed; longer sequences can
            //   raise via FERRUM_CUDA_MAX_KV env (bumps dynamic shared beyond
            //   48 KB via CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES).
            const DECODE_MAX_KV_POS_DEFAULT: usize = 8192; // 32 KB
            let env_cap = std::env::var("FERRUM_CUDA_MAX_KV")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(DECODE_MAX_KV_POS_DEFAULT);
            let max_kv_pos = capacity.min(env_cap as i32) as u32;
            let shared_mem = max_kv_pos * 4;
            // If user bumped the cap beyond 48 KB default, opt into the
            // higher limit on Blackwell (up to 228 KB).
            if shared_mem > 48 * 1024 {
                let _ = func.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem as i32,
                );
            }
            let stream = ctx.stream.clone();
            // Hold read-guard on global state bufs for the builder's lifetime.
            let dec_guard = if use_dyn {
                Some(decode_state_slot().read().expect("DECODE_STATE poisoned"))
            } else {
                None
            };
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
                let bufs = dec_guard.as_ref().unwrap().as_ref().unwrap();
                bld.arg(&bufs.kv);
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
            drop(dec_guard);
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
        // Must match `#define TILE_Q 16` in kernels/flash_attn_full.cu.
        // Was 32 — produced grid with too few blocks for q_len > 16, so
        // the last q-tile never launched and its attention output stayed
        // uninitialized. Observed as garbage first token ("emas") on any
        // prefill longer than 16 tokens (multi-turn chat, long prompts).
        const TILE_Q: usize = 16;
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
            let dec_guard = decode_state_slot().read().expect("DECODE_STATE poisoned");
            let bufs = dec_guard.as_ref().expect("DecodeStateBufs");
            let mut b = stream.launch_builder(&func);
            b.arg(table);
            b.arg(&bufs.token);
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
            drop(dec_guard);
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

    fn kv_cache_append_batched_per_cache(
        ctx: &mut Self::Context,
        caches: &[&Self::Buffer],
        new_data: &Self::Buffer,
        cache_lens: &Self::Buffer,
        capacity: usize,
        m: usize,
        nkv: usize,
        hd: usize,
        slot: usize,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;
        if m == 0 {
            return Ok(());
        }
        if caches.len() != m {
            return Err(FerrumError::model(
                "kv_cache_append_batched_per_cache: caches length != m",
            ));
        }

        let stream = ctx.stream.clone();
        if m > BATCHED_SCRATCH_CAP {
            return Err(FerrumError::model(format!(
                "kv_cache_append_batched_per_cache: m={m} exceeds BATCHED_SCRATCH_CAP={BATCHED_SCRATCH_CAP}",
            )));
        }
        if slot >= MAX_GRAPH_SLOTS {
            return Err(FerrumError::model(format!(
                "kv_cache_append_batched_per_cache: slot={slot} exceeds MAX_GRAPH_SLOTS={MAX_GRAPH_SLOTS}",
            )));
        }
        let host_start = slot * BATCHED_SCRATCH_CAP;
        let func = ctx.func(
            "kv_cache_append_batched",
            ptx::KV_CACHE_APPEND,
            "kv_cache_append_batched_per_cache_f16",
        );
        // Per-slot region of the PROCESS-GLOBAL host_cache_ptrs +
        // scratch_u64_cache. Each call site uses a distinct slot.
        // Captured graph records host pointer (per-slot region of the
        // global Box) + device pointer (per-slot view of the global
        // scratch); both outlive any CudaState. Replay across decode
        // calls re-reads CURRENT host content from the same address,
        // launches kernel reading CURRENT device content from the same
        // address. This is what makes FERRUM_BATCHED_GRAPH=1 safe.
        let m_i32 = m as i32;
        let nkv_i32 = nkv as i32;
        let hd_i32 = hd as i32;
        let capacity_i32 = capacity as i32;
        let per_item = nkv * hd;
        let block_dim = 256u32;
        let grid_x = (per_item as u32 + block_dim - 1) / block_dim;
        with_batched_scratch_mut(|slot_g| {
            for i in 0..m {
                let (cp, _) = caches[i].device_ptr(&stream);
                slot_g.host_cache_ptrs[host_start + i] = cp;
            }
            // Async memcpy host_slice → device per-slot region. Recorded
            // into the captured graph when capture is in flight; both
            // endpoints are stable global addresses.
            {
                let host_slice: &[u64] = &slot_g.host_cache_ptrs[host_start..host_start + m];
                let mut view = slot_g
                    .scratch_u64_cache
                    .slice_mut(host_start..host_start + m);
                stream
                    .memcpy_htod(host_slice, &mut view)
                    .map_err(|e| FerrumError::model(format!("memcpy cache_ptrs: {e}")))?;
            }
            let cache_ptrs_view = slot_g.scratch_u64_cache.slice(host_start..host_start + m);
            let cache_lens_dev = cache_lens;
            let mut b = stream.launch_builder(&func);
            b.arg(&cache_ptrs_view);
            b.arg(new_data);
            b.arg(cache_lens_dev);
            b.arg(&m_i32);
            b.arg(&nkv_i32);
            b.arg(&hd_i32);
            b.arg(&capacity_i32);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (grid_x, m as u32, 1),
                    block_dim: (block_dim, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| FerrumError::model(format!("kv_cache_append_batched: {e}")))?;
            Ok::<(), FerrumError>(())
        })?;
        Ok(())
    }

    fn flash_attention_batched_per_cache(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_caches: &[&Self::Buffer],
        v_caches: &[&Self::Buffer],
        kv_lens: &Self::Buffer,
        out: &mut Self::Buffer,
        nq: usize,
        nkv: usize,
        hd: usize,
        scale: f32,
        max_valid_kv: usize,
        capacity: usize,
        slot: usize,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;
        let m = k_caches.len();
        if m == 0 {
            return Ok(());
        }
        if v_caches.len() != m {
            return Err(FerrumError::model(
                "flash_attention_batched_per_cache: k/v length mismatch",
            ));
        }

        let stream = ctx.stream.clone();
        if m > BATCHED_SCRATCH_CAP {
            return Err(FerrumError::model(format!(
                "flash_attention_batched_per_cache: m={m} exceeds BATCHED_SCRATCH_CAP={BATCHED_SCRATCH_CAP}",
            )));
        }
        if slot >= MAX_GRAPH_SLOTS {
            return Err(FerrumError::model(format!(
                "flash_attention_batched_per_cache: slot={slot} exceeds MAX_GRAPH_SLOTS={MAX_GRAPH_SLOTS}",
            )));
        }
        let host_start = slot * BATCHED_SCRATCH_CAP;
        let func = ctx.func(
            "batched_decode_attn",
            ptx::BATCHED_DECODE_ATTENTION,
            "batched_decode_attention_f16",
        );
        let nq_i32 = nq as i32;
        let nkv_i32 = nkv as i32;
        let hd_i32 = hd as i32;
        let capacity_i32 = capacity as i32;
        // Shared mem must cover post-append max kv_len. Caller passes
        // `max_valid_kv` already accounting for the +1; sizing also
        // bounded by capacity to mirror the per-item kernel's pattern.
        let shared_bytes = (max_valid_kv.min(capacity).max(1) as u32) * 4;
        with_batched_scratch_mut(|slot_g| {
            for i in 0..m {
                let (kp, _) = k_caches[i].device_ptr(&stream);
                let (vp, _) = v_caches[i].device_ptr(&stream);
                slot_g.host_k_ptrs[host_start + i] = kp;
                slot_g.host_v_ptrs[host_start + i] = vp;
            }
            // Two captured memcpys, each into its own per-slot region of
            // process-global device scratch. Both host arrays + both
            // device scratches live in BATCHED_SCRATCH (process-global,
            // outlives every CudaState) — replay across decode calls is
            // safe because no pointer dangles.
            {
                let k_host_slice: &[u64] = &slot_g.host_k_ptrs[host_start..host_start + m];
                let mut view = slot_g.scratch_u64_k.slice_mut(host_start..host_start + m);
                stream
                    .memcpy_htod(k_host_slice, &mut view)
                    .map_err(|e| FerrumError::model(format!("memcpy k_ptrs: {e}")))?;
            }
            {
                let v_host_slice: &[u64] = &slot_g.host_v_ptrs[host_start..host_start + m];
                let mut view = slot_g.scratch_u64_v.slice_mut(host_start..host_start + m);
                stream
                    .memcpy_htod(v_host_slice, &mut view)
                    .map_err(|e| FerrumError::model(format!("memcpy v_ptrs: {e}")))?;
            }
            let k_ptrs_view = slot_g.scratch_u64_k.slice(host_start..host_start + m);
            let v_ptrs_view = slot_g.scratch_u64_v.slice(host_start..host_start + m);
            let kv_lens_dev = kv_lens;
            let mut b = stream.launch_builder(&func);
            b.arg(q);
            b.arg(&k_ptrs_view);
            b.arg(&v_ptrs_view);
            b.arg(out);
            b.arg(kv_lens_dev);
            b.arg(&nq_i32);
            b.arg(&nkv_i32);
            b.arg(&hd_i32);
            b.arg(&capacity_i32);
            b.arg(&scale);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (nq as u32, m as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: shared_bytes,
                })
            }
            .map_err(|e| FerrumError::model(format!("flash_attn_batched: {e}")))?;
            Ok::<(), FerrumError>(())
        })?;
        Ok(())
    }

    fn qk_norm_rope_batched_per_item(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        output: &mut Self::Buffer,
        positions: &Self::Buffer,
        m: usize,
        heads: usize,
        head_dim: usize,
        eps: f32,
        mode: i32,
    ) -> Result<()> {
        let func = ctx.func(
            "qk_norm_rope_batched",
            ptx::QK_NORM_ROPE,
            "qk_norm_rope_batched_decode_f16",
        );
        let m_i32 = m as i32;
        let heads_i32 = heads as i32;
        let head_dim_i32 = head_dim as i32;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(input);
        b.arg(norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(output);
        b.arg(&m_i32);
        b.arg(&heads_i32);
        b.arg(&head_dim_i32);
        b.arg(positions);
        b.arg(&eps);
        b.arg(&mode);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (m as u32, heads as u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| FerrumError::model(format!("qk_norm_rope_batched_per_item: {e}")))?;
        Ok(())
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
        let dec_guard = if use_dyn {
            Some(decode_state_slot().read().expect("DECODE_STATE poisoned"))
        } else {
            None
        };
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
            let bufs = dec_guard.as_ref().unwrap().as_ref().unwrap();
            b.arg(&bufs.pos);
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
        drop(dec_guard);
    }

    /// Split QKV + qk-norm + RoPE into FP16 head-major scratch buffers.
    /// Implemented as a chain over the existing primitives: `split_qkv` →
    /// 3× `qk_norm_rope` (Q/K with their respective norms; V with mode=0).
    /// Used by the INT8 KV path's `KvLayer<KvInt8>::paged_write` to
    /// materialize FP16 K/V before quantizing into the INT8 paged pool.
    /// FP16 paths use the fused `split_qkv_norm_rope_into_paged_cache`
    /// directly and never hit this method.
    fn split_qkv_norm_rope(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        k_out: &mut Self::Buffer,
        v_out: &mut Self::Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
    ) -> Result<()> {
        // Lazy scratch — split_qkv writes into per-token-major buffers.
        // We allocate just-in-time; the caller's `q_out/k_out/v_out` are
        // head-major after the chain.
        let q_dim = q_heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let q_buf_size = tokens * q_dim;
        let kv_buf_size = tokens * kv_dim;
        let mut q_buf = <Self as Backend>::alloc(q_buf_size);
        let mut k_buf = <Self as Backend>::alloc(kv_buf_size);
        let mut v_buf = <Self as Backend>::alloc(kv_buf_size);
        Self::split_qkv(ctx, qkv, &mut q_buf, &mut k_buf, &mut v_buf, tokens, q_dim, kv_dim);
        Self::qk_norm_rope(
            ctx, &q_buf, q_norm_w, cos, sin, q_out,
            tokens, q_heads, head_dim, pos_offset, eps, qk_mode,
        );
        Self::qk_norm_rope(
            ctx, &k_buf, k_norm_w, cos, sin, k_out,
            tokens, kv_heads, head_dim, pos_offset, eps, qk_mode,
        );
        // V: no norm + RoPE-only (qk_mode=0); pass q_norm_w as a dummy
        // (kernel ignores it when mode=0).
        Self::qk_norm_rope(
            ctx, &v_buf, q_norm_w, cos, sin, v_out,
            tokens, kv_heads, head_dim, pos_offset, eps, 0,
        );
        Ok(())
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
        let dec_guard = if use_dyn {
            Some(decode_state_slot().read().expect("DECODE_STATE poisoned"))
        } else {
            None
        };

        // K half.
        {
            let mut b = stream.launch_builder(&func);
            b.arg(cache_k);
            b.arg(new_k_head_major);
            b.arg(&nkv_i32);
            b.arg(&hd_i32);
            if use_dyn {
                let bufs = dec_guard.as_ref().unwrap().as_ref().unwrap();
                b.arg(&bufs.pos);
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
                let bufs = dec_guard.as_ref().unwrap().as_ref().unwrap();
                b.arg(&bufs.pos);
            } else {
                b.arg(&cache_len_i32);
            }
            b.arg(&new_tokens_i32);
            b.arg(&cap_i32);
            unsafe { b.launch(cfg) }.expect("kv_cache_append V launch");
        }
        drop(dec_guard);
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

    /// Inverse of `transpose_head_to_token`. Used by the CUDA paged
    /// attention wrapper to convert paged_varlen_attention's token-major
    /// output back to the head-major buffer Qwen3MoeModel expects.
    fn transpose_token_to_head(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
    ) {
        let func = ctx.func("transpose", ptx::TRANSPOSE, "transpose_token_to_head_f16");
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
        .expect("transpose_token_to_head launch");
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

    fn fused_silu_mul_split_strided(
        ctx: &mut Self::Context,
        gate_up: &Self::Buffer,
        in_row_offset: usize,
        out: &mut Self::Buffer,
        out_row_offset: usize,
        tokens: usize,
        intermediate: usize,
    ) {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        // Same kernel as `fused_silu_mul_split`, but feed it adjusted
        // device pointers so it operates on a row-range slice.
        let func = ctx.func(
            "fused_silu_mul",
            ptx::FUSED_SILU_MUL,
            "fused_silu_mul_interleaved_f16",
        );
        let im_i32 = intermediate as i32;
        let total = tokens * intermediate;
        let total_i32 = total as i32;
        let block = 256u32;
        let grid = ((total as u32) + block - 1) / block;

        let stream = ctx.stream.clone();
        let in_byte_off = in_row_offset * 2 * intermediate * std::mem::size_of::<half::f16>();
        let out_byte_off = out_row_offset * intermediate * std::mem::size_of::<half::f16>();

        let (gu_base, _g) = gate_up.device_ptr(&stream);
        let (out_base, _g2) = out.device_ptr_mut(&stream);
        let gu_ptr = gu_base + in_byte_off as u64;
        let out_ptr = out_base + out_byte_off as u64;

        let mut b = stream.launch_builder(&func);
        b.arg(&gu_ptr);
        b.arg(&out_ptr);
        b.arg(&im_i32);
        b.arg(&total_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("fused_silu_mul_split_strided launch");
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

    // ── GPTQ INT4 dispatch (Marlin default; Triton-rs alt via env) ──────
    //
    // gemm_quant moved to `impl BackendQuantGguf for CudaBackend {}`
    // (empty — CUDA inherits the unsupported default; INT4 goes through
    // gemm_gptq + GptqStore).

    fn zero_buffer(ctx: &mut Self::Context, buf: &mut Self::Buffer, len: usize) -> Result<()> {
        use cudarc::driver::DevicePtr;
        let stream = ctx.stream.clone();
        let (ptr, _g) = buf.device_ptr(&stream);
        unsafe {
            cudarc::driver::sys::cuMemsetD16Async(
                ptr as cudarc::driver::sys::CUdeviceptr,
                0,
                len,
                stream.cu_stream(),
            )
        }
        .result()
        .map_err(|e| FerrumError::model(format!("cuMemsetD16Async: {e}")))?;
        Ok(())
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
pub(super) fn default_stream() -> Arc<CudaStream> {
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
        // Disable cudarc event tracking BEFORE any buffer is allocated on
        // this context. Previously we only disabled in `new_context`, which
        // runs after model construction — meaning every weight buffer had a
        // dependency event recorded from its htod. During graph capture the
        // captured launches picked up those event dependencies, and on
        // replay cuGraphLaunch dereferenced a stale event pointer → SIGSEGV
        // inside libcuda.so. Flipping this here means all weight loads go
        // in cleanly, and the graph captured later sees no stray event
        // dependencies. Bare C++ reproducers (scripts/graph_repro{,_v2}.cu)
        // work because they never enable event tracking in the first place.
        unsafe {
            ctx.disable_event_tracking();
        }
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
// Process-global decode state buffers (token_id, pos, kv_len)
// ────────────────────────────────────────────────────────────────────────
//
// Must be global (not per-ctx): captured graph holds pointers to these
// buffers. If ctx is recreated per decode step (which it is), per-ctx
// bufs would be freed between capture and replay → dangling pointer →
// CUDA_ERROR_INVALID_VALUE on next sync.

pub struct DecodeStateBufs {
    pub token: CudaSlice<u32>,
    pub pos: CudaSlice<i32>,
    pub kv: CudaSlice<i32>,
}
unsafe impl Send for DecodeStateBufs {}
unsafe impl Sync for DecodeStateBufs {}

static DECODE_STATE: std::sync::OnceLock<std::sync::RwLock<Option<DecodeStateBufs>>> =
    std::sync::OnceLock::new();

pub(super) fn decode_state_slot() -> &'static std::sync::RwLock<Option<DecodeStateBufs>> {
    DECODE_STATE.get_or_init(|| std::sync::RwLock::new(None))
}

fn ensure_decode_state_bufs(stream: &Arc<CudaStream>) {
    let guard = decode_state_slot().read().expect("DECODE_STATE poisoned");
    if guard.is_some() {
        return;
    }
    drop(guard);
    let mut w = decode_state_slot().write().expect("DECODE_STATE poisoned");
    if w.is_none() {
        let token = unsafe { stream.alloc::<u32>(1) }.expect("token_buf alloc");
        let pos = unsafe { stream.alloc::<i32>(1) }.expect("pos_buf alloc");
        let kv = unsafe { stream.alloc::<i32>(1) }.expect("kv_buf alloc");
        *w = Some(DecodeStateBufs { token, pos, kv });
    }
}

fn with_decode_state<R>(f: impl FnOnce(&DecodeStateBufs) -> R) -> R {
    let guard = decode_state_slot().read().expect("DECODE_STATE poisoned");
    f(guard.as_ref().expect("DecodeStateBufs not initialised"))
}

// ────────────────────────────────────────────────────────────────────────
// Process-global cuBLAS handle + 32MB workspace
// ────────────────────────────────────────────────────────────────────────
//
// Must be process-global (not per-ctx): graph capture records the workspace
// device pointer as a kernel arg. Per-ctx workspace would be freed when ctx
// drops → dangling pointer on replay → CUDA_ERROR_INVALID_VALUE.

struct BlasSlot {
    blas: Arc<CudaBlas>,
    _workspace: CudaSlice<u8>,
    // Device-resident alpha/beta for f16/f32 GEMMs. cuBLAS captures the
    // scalar-copy of alpha/beta into the graph; host pointers would
    // dangle on replay (stack-local). Device pointers persist.
    pub alpha_f32: CudaSlice<f32>, // [1.0]
    pub beta_f32: CudaSlice<f32>,  // [0.0]
}
unsafe impl Send for BlasSlot {}
unsafe impl Sync for BlasSlot {}

static BLAS_HANDLE: std::sync::OnceLock<std::sync::RwLock<Option<BlasSlot>>> =
    std::sync::OnceLock::new();

fn blas_slot() -> &'static std::sync::RwLock<Option<BlasSlot>> {
    BLAS_HANDLE.get_or_init(|| std::sync::RwLock::new(None))
}

fn ensure_blas_handle(stream: &Arc<CudaStream>) -> Arc<CudaBlas> {
    if let Some(slot) = blas_slot().read().expect("BLAS poisoned").as_ref() {
        return slot.blas.clone();
    }
    let mut w = blas_slot().write().expect("BLAS poisoned");
    if w.is_none() {
        const WS_BYTES: usize = 32 * 1024 * 1024;
        let blas = Arc::new(CudaBlas::new(stream.clone()).expect("CudaBlas::new"));
        let workspace = unsafe { stream.alloc::<u8>(WS_BYTES) }.expect("blas ws alloc");
        let alpha_f32 = stream.clone_htod(&[1.0f32]).expect("alpha htod");
        let beta_f32 = stream.clone_htod(&[0.0f32]).expect("beta htod");
        unsafe {
            use cudarc::cublas::sys;
            use cudarc::driver::DevicePtr;
            let (ws_ptr, _g) = workspace.device_ptr(stream);
            let st = sys::cublasSetWorkspace_v2(*blas.handle(), ws_ptr as *mut _, WS_BYTES);
            assert_eq!(
                st,
                sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                "set workspace"
            );
            // Switch to device-pointer mode so alpha/beta pass cleanly through
            // graph capture. cuBLAS is HOST mode by default; in HOST mode it
            // internally memcpies the scalar from host, and that memcpy lands
            // in the captured graph with a stack-local pointer → UB at replay.
            let st = sys::cublasSetPointerMode_v2(
                *blas.handle(),
                sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
            );
            assert_eq!(
                st,
                sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                "set pointer mode"
            );
        }
        *w = Some(BlasSlot {
            blas,
            _workspace: workspace,
            alpha_f32,
            beta_f32,
        });
    }
    w.as_ref().unwrap().blas.clone()
}

/// Access the process-global alpha/beta device scalars for cuBLAS.
fn with_blas_scalars<R>(f: impl FnOnce(&CudaSlice<f32>, &CudaSlice<f32>) -> R) -> R {
    let g = blas_slot().read().expect("BLAS poisoned");
    let s = g.as_ref().expect("BLAS not init");
    f(&s.alpha_f32, &s.beta_f32)
}

// ────────────────────────────────────────────────────────────────────────
// Process-global batched-scratch device + host arrays
// ────────────────────────────────────────────────────────────────────────
//
// Used by `kv_cache_append_batched_per_cache` and
// `flash_attention_batched_per_cache`. CRITICAL for FERRUM_BATCHED_GRAPH=1:
//
//   * Captured graph contains `cuMemcpy(host_array, device_scratch, ...)`
//     and kernel launches reading `device_scratch[slot..slot+m]`.
//   * Captured memcpy reads HOST POINTER at REPLAY time
//     (verified: cudarc_graph_shared_host_array_multi_memcpy).
//   * Captured kernel arg holds a fixed device pointer.
//
// If either array lives in per-call CudaState, drop()-ing the state
// between decode calls invalidates the pointer the captured graph
// holds → 2nd replay's cuGraphLaunch SIGSEGVs inside libcuda.so.
// Per-call new_context() in this file is exactly that pattern.
// Process-global slot keeps both arrays alive forever.

struct BatchedScratchSlot {
    /// Device staging for K cache pointers (flash_attn read).
    pub scratch_u64_k: CudaSlice<u64>,
    /// Device staging for V cache pointers (flash_attn read).
    pub scratch_u64_v: CudaSlice<u64>,
    /// Device staging for K-or-V cache pointers (kv_cache_append read).
    pub scratch_u64_cache: CudaSlice<u64>,
    /// Host staging — captured memcpy reads these heap addresses on replay.
    /// Box keeps stable heap address; static slot keeps Box alive forever.
    pub host_k_ptrs: Box<[u64; HOST_STAGING_TOTAL]>,
    pub host_v_ptrs: Box<[u64; HOST_STAGING_TOTAL]>,
    pub host_cache_ptrs: Box<[u64; HOST_STAGING_TOTAL]>,
}
unsafe impl Send for BatchedScratchSlot {}
unsafe impl Sync for BatchedScratchSlot {}

static BATCHED_SCRATCH: std::sync::OnceLock<std::sync::RwLock<Option<BatchedScratchSlot>>> =
    std::sync::OnceLock::new();

fn batched_scratch_slot() -> &'static std::sync::RwLock<Option<BatchedScratchSlot>> {
    BATCHED_SCRATCH.get_or_init(|| std::sync::RwLock::new(None))
}

fn ensure_batched_scratch(stream: &Arc<CudaStream>) {
    {
        let g = batched_scratch_slot()
            .read()
            .expect("BATCHED_SCRATCH poisoned");
        if g.is_some() {
            return;
        }
    }
    let mut w = batched_scratch_slot()
        .write()
        .expect("BATCHED_SCRATCH poisoned");
    if w.is_none() {
        let scratch_u64_k = unsafe { stream.alloc::<u64>(HOST_STAGING_TOTAL) }
            .expect("batched scratch_u64_k alloc");
        let scratch_u64_v = unsafe { stream.alloc::<u64>(HOST_STAGING_TOTAL) }
            .expect("batched scratch_u64_v alloc");
        let scratch_u64_cache = unsafe { stream.alloc::<u64>(HOST_STAGING_TOTAL) }
            .expect("batched scratch_u64_cache alloc");
        *w = Some(BatchedScratchSlot {
            scratch_u64_k,
            scratch_u64_v,
            scratch_u64_cache,
            host_k_ptrs: Box::new([0u64; HOST_STAGING_TOTAL]),
            host_v_ptrs: Box::new([0u64; HOST_STAGING_TOTAL]),
            host_cache_ptrs: Box::new([0u64; HOST_STAGING_TOTAL]),
        });
    }
}

/// Access the process-global batched scratch for the duration of one
/// captured-or-eager kernel call. Holds the slot's RwLock write guard
/// for that duration — no other batched op runs concurrently (single
/// stream, single decode_batch_internal at a time per iteration_lock).
fn with_batched_scratch_mut<R>(f: impl FnOnce(&mut BatchedScratchSlot) -> R) -> R {
    let mut g = batched_scratch_slot()
        .write()
        .expect("BATCHED_SCRATCH poisoned");
    f(g.as_mut().expect("BatchedScratchSlot not initialised"))
}

// ────────────────────────────────────────────────────────────────────────
// Process-global PTX module cache
// ────────────────────────────────────────────────────────────────────────
//
// Same graph-capture lifetime requirement as cuBLAS workspace + batched
// scratch above: cudarc records a CUfunction handle into the captured
// graph's kernel node. CUfunction handles are owned by their CUmodule;
// when the last Arc<CudaModule> drops, cudarc unloads the module and the
// CUfunction goes invalid → captured graph's kernel node references a
// stale handle → 2nd cuGraphLaunch SIGSEGVs inside libcuda.so.
//
// Per-CudaState `modules: HashMap<...>` was the third pointer-lifetime
// bug in this file (after BLAS workspace and batched scratch). Routing
// loads through a process-global cache keeps every loaded CudaModule
// alive for the rest of the process — captured graphs always find a
// valid CUfunction at replay time.
//
// CudaState still keeps its local HashMap as a hot-path cache so that
// per-kernel launches don't lock the global Mutex.

static MODULES: std::sync::OnceLock<std::sync::Mutex<HashMap<&'static str, Arc<CudaModule>>>> =
    std::sync::OnceLock::new();

fn modules_cache() -> &'static std::sync::Mutex<HashMap<&'static str, Arc<CudaModule>>> {
    MODULES.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}


// ────────────────────────────────────────────────────────────────────────
// Paged-varlen split-K scratch + dispatch
// ────────────────────────────────────────────────────────────────────────

/// Process-global scratch for split-K phase1 outputs. Three buffers
/// (partial_out f32, partial_m f32, partial_l f32) sized to the largest
/// shape ever requested. Same lazy-grow pattern as Marlin gather scratch.
struct SplitKScratch {
    partial_out: CudaSlice<f32>, // [M_total * num_q_heads * num_splits * head_dim]
    partial_m: CudaSlice<f32>,   // [M_total * num_q_heads * num_splits]
    partial_l: CudaSlice<f32>,   // [M_total * num_q_heads * num_splits]
    out_capacity: usize,
    ml_capacity: usize,
}
unsafe impl Send for SplitKScratch {}
unsafe impl Sync for SplitKScratch {}

static SPLIT_K_SCRATCH: std::sync::OnceLock<std::sync::RwLock<Option<SplitKScratch>>> =
    std::sync::OnceLock::new();

fn split_k_scratch_slot() -> &'static std::sync::RwLock<Option<SplitKScratch>> {
    SPLIT_K_SCRATCH.get_or_init(|| std::sync::RwLock::new(None))
}

fn with_split_k_scratch<R>(
    stream: &Arc<CudaStream>,
    out_required: usize,
    ml_required: usize,
    body: impl FnOnce(&mut CudaSlice<f32>, &mut CudaSlice<f32>, &mut CudaSlice<f32>) -> R,
) -> R {
    let slot = split_k_scratch_slot();
    {
        let g = slot.read().expect("SPLIT_K_SCRATCH poisoned");
        if let Some(ref s) = *g {
            if s.out_capacity >= out_required && s.ml_capacity >= ml_required {
                drop(g);
                let mut w = slot.write().expect("SPLIT_K_SCRATCH poisoned");
                let s = w.as_mut().expect("just observed Some");
                return body(&mut s.partial_out, &mut s.partial_m, &mut s.partial_l);
            }
        }
    }
    let mut w = slot.write().expect("SPLIT_K_SCRATCH poisoned");
    let need_new = match &*w {
        Some(s) => s.out_capacity < out_required || s.ml_capacity < ml_required,
        None => true,
    };
    if need_new {
        let partial_out = unsafe { stream.alloc::<f32>(out_required) }
            .expect("SPLIT_K_SCRATCH partial_out alloc");
        let partial_m =
            unsafe { stream.alloc::<f32>(ml_required) }.expect("SPLIT_K_SCRATCH partial_m alloc");
        let partial_l =
            unsafe { stream.alloc::<f32>(ml_required) }.expect("SPLIT_K_SCRATCH partial_l alloc");
        *w = Some(SplitKScratch {
            partial_out,
            partial_m,
            partial_l,
            out_capacity: out_required,
            ml_capacity: ml_required,
        });
    }
    let s = w.as_mut().expect("just allocated");
    body(&mut s.partial_out, &mut s.partial_m, &mut s.partial_l)
}

#[allow(clippy::too_many_arguments)]
fn paged_varlen_split_k_dispatch(
    ctx: &mut CudaState,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    cu_seqlens_q: &CudaSlice<f16>,
    pos_offsets: &CudaSlice<f16>,
    block_tables: &CudaSlice<f16>,
    num_seqs: usize,
    total_q_tokens: usize,
    max_kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    // Pick num_splits based on kv_len. Microbench peak points:
    //   kv ≤ 384  → N=2 (or skip if c≥16)
    //   kv ≤ 1024 → N=4
    //   kv ≤ 2048 → N=8
    //   else      → N=16
    let num_splits: usize = match max_kv_len {
        kv if kv <= 384 => 2,
        kv if kv <= 1024 => 4,
        kv if kv <= 2048 => 8,
        _ => 16,
    };

    let chunk = (max_kv_len + num_splits - 1) / num_splits;
    let out_required = total_q_tokens * num_heads * num_splits * head_dim;
    let ml_required = total_q_tokens * num_heads * num_splits;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let stream = ctx.stream.clone();

    let phase1 = ctx.func(
        "paged_varlen_split_k_phase1",
        ptx::PAGED_VARLEN_ATTENTION,
        "paged_varlen_attn_split_k_phase1_f16",
    );
    let reduce = ctx.func(
        "paged_varlen_split_k_reduce",
        ptx::PAGED_VARLEN_ATTENTION,
        "paged_varlen_split_k_reduce_f16",
    );

    with_split_k_scratch(
        &stream,
        out_required,
        ml_required,
        |partial_out, partial_m, partial_l| {
            let qv = q.slice(..);
            let kp = k_pool.slice(..);
            let vp = v_pool.slice(..);
            let csq = cu_seqlens_q.slice(..);
            let po = pos_offsets.slice(..);
            let bt = block_tables.slice(..);
            let pout = partial_out.slice(..);
            let pm = partial_m.slice(..);
            let pl = partial_l.slice(..);
            let ns = num_seqs as i32;
            let nqi = num_heads as i32;
            let nkvi = num_kv_heads as i32;
            let hdi = head_dim as i32;
            let mbps = max_blocks_per_seq as i32;
            let bsi = block_size as i32;
            let nsp = num_splits as i32;

            // Phase 1: (num_heads, M_total, num_splits)
            let mut b1 = stream.launch_builder(&phase1);
            b1.arg(&qv);
            b1.arg(&kp);
            b1.arg(&vp);
            b1.arg(&csq);
            b1.arg(&po);
            b1.arg(&bt);
            b1.arg(&pout);
            b1.arg(&pm);
            b1.arg(&pl);
            b1.arg(&ns);
            b1.arg(&nqi);
            b1.arg(&nkvi);
            b1.arg(&hdi);
            b1.arg(&mbps);
            b1.arg(&bsi);
            b1.arg(&scale);
            b1.arg(&nsp);
            let shmem1 = (chunk.max(1) as u32) * 4;
            unsafe {
                b1.launch(LaunchConfig {
                    grid_dim: (num_heads as u32, total_q_tokens as u32, num_splits as u32),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: shmem1,
                })
            }
            .map_err(|e| FerrumError::model(format!("paged_varlen_split_k_phase1: {e}")))?;

            // Phase 2: (num_heads, M_total)
            let pout2 = partial_out.slice(..);
            let pm2 = partial_m.slice(..);
            let pl2 = partial_l.slice(..);
            let mut b2 = stream.launch_builder(&reduce);
            b2.arg(&pout2);
            b2.arg(&pm2);
            b2.arg(&pl2);
            b2.arg(out);
            b2.arg(&nqi);
            b2.arg(&hdi);
            b2.arg(&nsp);
            unsafe {
                b2.launch(LaunchConfig {
                    grid_dim: (num_heads as u32, total_q_tokens as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| FerrumError::model(format!("paged_varlen_split_k_reduce: {e}")))?;

            Ok::<(), FerrumError>(())
        },
    )
}

// ────────────────────────────────────────────────────────────────────────
// Stage 13a — Batched paged-decode flash (split-K)
//
// Same idea as the varlen split-K path but for q_len=1 batched decode
// (gridDim.y = num_seqs). Phase 1 splits each seq's kv across `num_splits`
// chunks; phase 2 reduces partials per (seq, head).
//
// FERRUM_PAGED_FLASH=1 selects this over the single-pass kernel. Default
// OFF.
// ────────────────────────────────────────────────────────────────────────
#[allow(clippy::too_many_arguments)]
fn paged_batched_flash_dispatch(
    ctx: &mut CudaState,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    block_tables: &CudaSlice<f16>,
    valid_kv_lens: &CudaSlice<f16>,
    num_seqs: usize,
    max_kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    // Pick num_splits. Heuristic combines two effects:
    //   1. SM occupancy: a (num_q_heads, num_seqs, splits) grid wants
    //      total blocks ≳ 2 × SM count for full pipelining. When the
    //      base grid (num_seqs × num_heads) already saturates SMs,
    //      splits ≥ 2 just add launch + reduce overhead.
    //   2. kv_len: longer kv → more inherent serial work in step 3 →
    //      split-K helps even at moderate occupancy.
    //
    // Bench (RTX 4090, M3, 32 q_heads):
    //   c=1  splits=8 → +22% (grid was 1/4 wave, splits saturate)
    //   c=16 splits=2 → -3.7% (grid already 4 waves)
    //   c=32 splits=2 → +5% (grid 8 waves, kv split still helps)
    // Override via FERRUM_PAGED_FLASH_SPLITS for tuning.
    let force_splits = std::env::var("FERRUM_PAGED_FLASH_SPLITS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    // 128 SMs on Ada/Hopper-class; conservative under-estimate.
    const SM_TARGET: usize = 128;
    let base_grid = num_seqs * num_heads;
    let saturated = base_grid >= 2 * SM_TARGET;
    let num_splits: usize = force_splits.unwrap_or_else(|| {
        if saturated {
            // Only split when kv is so long that the V loop dominates.
            match max_kv_len {
                kv if kv <= 768 => 1,
                kv if kv <= 2048 => 2,
                _ => 4,
            }
        } else {
            // Low concurrency: aggressive splits to fill SMs.
            let needed = (SM_TARGET + base_grid - 1) / base_grid;
            let by_kv = match max_kv_len {
                kv if kv <= 256 => 4,
                kv if kv <= 1024 => 8,
                _ => 16,
            };
            needed.max(1).min(by_kv).min(16)
        }
    });
    if num_splits <= 1 {
        // Caller's main path is the single-pass kernel.
        // FERRUM_PAGED_FLASH=1 still routes here, so do the single-pass
        // launch inline (avoids env-flag round-trip).
        return paged_batched_decode_single_pass(
            ctx,
            q,
            k_pool,
            v_pool,
            out,
            block_tables,
            valid_kv_lens,
            num_seqs,
            max_kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
        );
    }

    let chunk = (max_kv_len + num_splits - 1) / num_splits;
    let total_qh = num_seqs * num_heads;
    let out_required = total_qh * num_splits * head_dim;
    let ml_required = total_qh * num_splits;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let stream = ctx.stream.clone();

    let phase1 = ctx.func(
        "paged_batched_flash_attn",
        ptx::PAGED_DECODE_ATTENTION,
        "paged_batched_flash_decode_attn_f16",
    );
    let phase2 = ctx.func(
        "paged_batched_flash_reduce",
        ptx::PAGED_DECODE_ATTENTION,
        "paged_batched_flash_decode_reduce_f16",
    );

    with_split_k_scratch(
        &stream,
        out_required,
        ml_required,
        |partial_out, partial_m, partial_l| {
            let qv = q.slice(..);
            let kp = k_pool.slice(..);
            let vp = v_pool.slice(..);
            let bt = block_tables.slice(..);
            let kvl = valid_kv_lens.slice(..);
            let pout = partial_out.slice(..);
            let pm = partial_m.slice(..);
            let pl = partial_l.slice(..);
            let nqi = num_heads as i32;
            let nkvi = num_kv_heads as i32;
            let hdi = head_dim as i32;
            let mbps = max_blocks_per_seq as i32;
            let bsi = block_size as i32;
            let nsp = num_splits as i32;

            let mut b1 = stream.launch_builder(&phase1);
            b1.arg(&qv);
            b1.arg(&kp);
            b1.arg(&vp);
            b1.arg(&bt);
            b1.arg(&kvl);
            b1.arg(&pout);
            b1.arg(&pm);
            b1.arg(&pl);
            b1.arg(&nqi);
            b1.arg(&nkvi);
            b1.arg(&hdi);
            b1.arg(&mbps);
            b1.arg(&bsi);
            b1.arg(&scale);
            b1.arg(&nsp);
            // Match graph-capture sizing rationale used elsewhere — size
            // shared to FERRUM_KV_CAPACITY ceiling, not current chunk.
            let safe_kv: usize = std::env::var("FERRUM_KV_CAPACITY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(512);
            let safe_chunk = (safe_kv + num_splits - 1) / num_splits;
            let shmem1 = (safe_chunk.max(chunk).max(1) as u32) * 4;
            unsafe {
                b1.launch(LaunchConfig {
                    grid_dim: (num_heads as u32, num_seqs as u32, num_splits as u32),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: shmem1,
                })
            }
            .map_err(|e| FerrumError::model(format!("paged_batched_flash phase1: {e}")))?;

            let pout2 = partial_out.slice(..);
            let pm2 = partial_m.slice(..);
            let pl2 = partial_l.slice(..);
            let mut b2 = stream.launch_builder(&phase2);
            b2.arg(&pout2);
            b2.arg(&pm2);
            b2.arg(&pl2);
            b2.arg(out);
            b2.arg(&nqi);
            b2.arg(&hdi);
            b2.arg(&nsp);
            unsafe {
                b2.launch(LaunchConfig {
                    grid_dim: (num_heads as u32, num_seqs as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| FerrumError::model(format!("paged_batched_flash phase2: {e}")))?;

            Ok::<(), FerrumError>(())
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn paged_batched_decode_single_pass(
    ctx: &mut CudaState,
    q: &CudaSlice<f16>,
    k_pool: &CudaSlice<f16>,
    v_pool: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    block_tables: &CudaSlice<f16>,
    valid_kv_lens: &CudaSlice<f16>,
    num_seqs: usize,
    max_kv_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    let func = ctx.func(
        "paged_batched_decode_attn",
        ptx::PAGED_DECODE_ATTENTION,
        "paged_batched_decode_attn_f16",
    );
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let stream = ctx.stream.clone();
    let qv = q.slice(..);
    let kp = k_pool.slice(..);
    let vp = v_pool.slice(..);
    let bt = block_tables.slice(..);
    let kvl = valid_kv_lens.slice(..);
    let nqi = num_heads as i32;
    let nkvi = num_kv_heads as i32;
    let hdi = head_dim as i32;
    let mbps = max_blocks_per_seq as i32;
    let bsi = block_size as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(&qv);
    b.arg(&kp);
    b.arg(&vp);
    b.arg(&bt);
    b.arg(&kvl);
    b.arg(out);
    b.arg(&nqi);
    b.arg(&nkvi);
    b.arg(&hdi);
    b.arg(&mbps);
    b.arg(&bsi);
    b.arg(&scale);
    let safe_kv_max: usize = std::env::var("FERRUM_KV_CAPACITY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512);
    let shared_kv = safe_kv_max.max(max_kv_len).max(1);
    let shared_bytes = (shared_kv as u32) * 4;
    unsafe {
        b.launch(LaunchConfig {
            grid_dim: (num_heads as u32, num_seqs as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared_bytes,
        })
    }
    .map(|_| ())
    .map_err(|e| FerrumError::model(format!("paged_batched_decode_attn: {e}")))
}

pub(super) fn ensure_module(
    ctx: &Arc<CudaContext>,
    key: &'static str,
    ptx_src: &str,
) -> Arc<CudaModule> {
    {
        let g = modules_cache().lock().expect("MODULES poisoned");
        if let Some(m) = g.get(key) {
            return m.clone();
        }
    }
    let mut g = modules_cache().lock().expect("MODULES poisoned");
    if let Some(m) = g.get(key) {
        return m.clone();
    }
    let m = ctx
        .load_module(Ptx::from_src(ptx_src.to_string()))
        .unwrap_or_else(|e| panic!("CudaBackend: load_module({key}): {e}"));
    g.insert(key, m.clone());
    m
}


impl BackendPagedKv for CudaBackend {
    /// Default ON for CUDA. Mixed-batch unified_forward path requires
    /// paged_pools; without it the engine's run_unified_iter falls back
    /// to serial prefill that stalls in-flight decoders (~50% of bench
    /// wall time at c=16). Override via FERRUM_METAL_PAGED_KV=0 if a
    /// caller specifically wants legacy contig KV.
    fn supports_paged_kv() -> bool {
        true
    }
    fn supports_varlen_qkv() -> bool {
        true
    }
    fn populate_batched_pointers(
        ctx: &mut Self::Context,
        k_caches: &[&Self::Buffer],
        v_caches: &[&Self::Buffer],
        num_layers: usize,
        m: usize,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;
        if num_layers == 0 || m == 0 {
            return Ok(());
        }
        if num_layers > super::MAX_LAYERS_FOR_GRAPH {
            return Err(FerrumError::model(format!(
                "populate_batched_pointers: num_layers={num_layers} > MAX_LAYERS_FOR_GRAPH={}",
                super::MAX_LAYERS_FOR_GRAPH
            )));
        }
        if m > BATCHED_SCRATCH_CAP {
            return Err(FerrumError::model(format!(
                "populate_batched_pointers: m={m} > BATCHED_SCRATCH_CAP={BATCHED_SCRATCH_CAP}",
            )));
        }
        if k_caches.len() != num_layers * m || v_caches.len() != num_layers * m {
            return Err(FerrumError::model(
                "populate_batched_pointers: k/v_caches length != num_layers * m",
            ));
        }

        let stream = ctx.stream.clone();
        // Lazy-alloc all three device scratch buffers to HOST_STAGING_TOTAL
        // u64 elements. Done outside any captured stream — sync allocs only.
        if ctx.batched_scratch_u64_cache.is_none() {
            ctx.batched_scratch_u64_cache = Some(
                stream
                    .alloc_zeros::<u64>(HOST_STAGING_TOTAL)
                    .map_err(|e| FerrumError::model(format!("alloc cache_ptrs: {e}")))?,
            );
        }
        if ctx.batched_scratch_u64_k.is_none() {
            ctx.batched_scratch_u64_k = Some(
                stream
                    .alloc_zeros::<u64>(HOST_STAGING_TOTAL)
                    .map_err(|e| FerrumError::model(format!("alloc k_ptrs: {e}")))?,
            );
        }
        if ctx.batched_scratch_u64_v.is_none() {
            ctx.batched_scratch_u64_v = Some(
                stream
                    .alloc_zeros::<u64>(HOST_STAGING_TOTAL)
                    .map_err(|e| FerrumError::model(format!("alloc v_ptrs: {e}")))?,
            );
        }
        // Fill host arrays at every slot we'll launch from. Layout:
        //   K-append (kv_cache_append): slot = li → host_cache_ptrs[li * CAP ..]
        //   V-append (kv_cache_append): slot = li + MAX_LAYERS_FOR_GRAPH
        //   flash_attn:                 slot = li → host_k_ptrs / host_v_ptrs
        for li in 0..num_layers {
            let k_off = li * BATCHED_SCRATCH_CAP;
            let v_off = (li + super::MAX_LAYERS_FOR_GRAPH) * BATCHED_SCRATCH_CAP;
            for i in 0..m {
                let (kp, _) = k_caches[li * m + i].device_ptr(&stream);
                let (vp, _) = v_caches[li * m + i].device_ptr(&stream);
                ctx.batched_host_cache_ptrs[k_off + i] = kp;
                ctx.batched_host_cache_ptrs[v_off + i] = vp;
                ctx.batched_host_k_ptrs[k_off + i] = kp;
                ctx.batched_host_v_ptrs[k_off + i] = vp;
            }
        }
        // Bind context for sync memcpys (tokio thread-shift safe).
        ctx.ctx
            .bind_to_thread()
            .map_err(|e| FerrumError::unsupported(format!("populate bind_to_thread: {e}")))?;

        // Sync memcpy each entire host array to its device scratch in one shot.
        // cuMemcpyHtoD_v2 is on the legacy default (null) stream → NOT
        // captured by stream capture, so the captured graph contains
        // only kernel launches. Device scratch is fresh before every
        // call, including pure-replay (which doesn't re-enter Rust).
        let total_bytes = HOST_STAGING_TOTAL * std::mem::size_of::<u64>();
        unsafe {
            use cudarc::driver::{sys, DevicePtrMut};
            let scratch_cache = ctx.batched_scratch_u64_cache.as_mut().unwrap();
            let (dst, _g) = scratch_cache.device_ptr_mut(&stream);
            let st = sys::cuMemcpyHtoD_v2(
                dst,
                ctx.batched_host_cache_ptrs.as_ptr() as *const std::ffi::c_void,
                total_bytes,
            );
            if st != sys::CUresult::CUDA_SUCCESS {
                return Err(FerrumError::unsupported(format!(
                    "populate cache_ptrs sync memcpy: {st:?}"
                )));
            }
            let scratch_k = ctx.batched_scratch_u64_k.as_mut().unwrap();
            let (dst, _g) = scratch_k.device_ptr_mut(&stream);
            let st = sys::cuMemcpyHtoD_v2(
                dst,
                ctx.batched_host_k_ptrs.as_ptr() as *const std::ffi::c_void,
                total_bytes,
            );
            if st != sys::CUresult::CUDA_SUCCESS {
                return Err(FerrumError::unsupported(format!(
                    "populate k_ptrs sync memcpy: {st:?}"
                )));
            }
            let scratch_v = ctx.batched_scratch_u64_v.as_mut().unwrap();
            let (dst, _g) = scratch_v.device_ptr_mut(&stream);
            let st = sys::cuMemcpyHtoD_v2(
                dst,
                ctx.batched_host_v_ptrs.as_ptr() as *const std::ffi::c_void,
                total_bytes,
            );
            if st != sys::CUresult::CUDA_SUCCESS {
                return Err(FerrumError::unsupported(format!(
                    "populate v_ptrs sync memcpy: {st:?}"
                )));
            }
        }
        Ok(())
    }
    fn paged_varlen_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        pos_offsets: &Self::Buffer,
        block_tables: &Self::Buffer,
        num_seqs: usize,
        total_q_tokens: usize,
        max_kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if num_seqs == 0 || total_q_tokens == 0 {
            return Ok(());
        }

        // Auto-tune: split-K helps under-occupied grids — low concurrency
        // OR long context. Microbench (scripts/microbench_split_k.cu)
        // shows c=1/kv=4096 9× speedup, c=4/kv=384 +103%, c=16/kv=384
        // marginal/-2%. Heuristic gates split-K to regions where it wins.
        // FERRUM_SPLIT_K_ATTN=1 forces on, FERRUM_SPLIT_K_ATTN=0 forces off.
        let split_k_force = std::env::var("FERRUM_SPLIT_K_ATTN").ok();
        let use_split_k = match split_k_force.as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => num_seqs <= 4 || max_kv_len >= 768,
        };

        if use_split_k {
            return paged_varlen_split_k_dispatch(
                ctx,
                q,
                k_pool,
                v_pool,
                out,
                cu_seqlens_q,
                pos_offsets,
                block_tables,
                num_seqs,
                total_q_tokens,
                max_kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_blocks_per_seq,
            );
        }

        let func = ctx.func(
            "paged_varlen_attn",
            ptx::PAGED_VARLEN_ATTENTION,
            "paged_varlen_attn_f16",
        );
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let stream = ctx.stream.clone();
        // CudaBackend::Buffer is monomorphic CudaSlice<f16>; i32 data
        // (cu_seqlens_q / pos_offsets / block_tables) is stored in
        // f16-typed buffers via `from_slice_i32` + matching alloc, the
        // kernel reads them as `int*`. Same pattern as kv_lens in
        // `flash_attention_batched_per_cache`.
        let qv = q.slice(..);
        let kp = k_pool.slice(..);
        let vp = v_pool.slice(..);
        let csq = cu_seqlens_q.slice(..);
        let po = pos_offsets.slice(..);
        let bt = block_tables.slice(..);
        let ns = num_seqs as i32;
        let nqi = num_heads as i32;
        let nkvi = num_kv_heads as i32;
        let hdi = head_dim as i32;
        let mbps = max_blocks_per_seq as i32;
        let bsi = block_size as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(&qv);
        b.arg(&kp);
        b.arg(&vp);
        b.arg(&csq);
        b.arg(&po);
        b.arg(&bt);
        b.arg(out);
        b.arg(&ns);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&mbps);
        b.arg(&bsi);
        b.arg(&scale);
        // CUDA graph capture freezes `shared_mem_bytes` at capture time;
        // graph keys at the engine level are (m_total, num_seqs) — they
        // do NOT distinguish kv_len buckets. So a graph captured at
        // kv_len=300 (shared=300*4) replays unchanged at kv_len=600 →
        // kernel writes scores[300..600] OOB into shared.
        // compute-sanitizer caught it:
        //   "Invalid __shared__ write of size 4 bytes at paged_varlen_attn_f16
        //    Address 0x84c is out of bounds (in captured graph replay)".
        //
        // Allocate the worst-case kv slot length for ANY future decode
        // iter that may replay this graph. FERRUM_KV_CAPACITY caps it
        // (default 512, bench sets 2048). 8 KB shared = 2048 floats
        // — well within Ada's 96 KB/SM and Hopper's 228 KB/SM budgets.
        // For models with longer effective contexts the cap raises with
        // capacity at the cost of one alloc, never per-launch.
        let safe_kv_max: usize = std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let shared_kv = safe_kv_max.max(max_kv_len).max(1);
        let shared_bytes = (shared_kv as u32) * 4;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_heads as u32, total_q_tokens as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| FerrumError::model(format!("paged_varlen_attn: {e}")))
    }
    /// Paged attention dispatcher (CUDA-only, replaces missing native
    /// `paged_decode_attention`). Routes:
    ///   - q_len==1 (decode for any num_seqs): paged_batched_decode_attention.
    ///     The layouts coincide for q_len==1 — a [num_seqs, heads, dim]
    ///     buffer is identical to [heads, num_seqs, dim] when seen as a
    ///     single seq's contribution, so no transpose is needed.
    ///   - q_len>1 (prefill, single-seq only): paged_varlen_attention.
    ///     Caller's q is `[heads, q_len, dim]` (head-major) but varlen
    ///     reads `[q_len, heads, dim]` (token-major), so we transpose
    ///     in/out around the call. Cold path (prefill is rare per token).
    #[allow(clippy::too_many_arguments)]
    fn paged_decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        block_tables: &Self::Buffer,
        context_lens: &Self::Buffer,
        num_seqs: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
        q_len: usize,
    ) -> Result<()> {
        let max_kv_len = block_size * max_num_blocks_per_seq;

        if q_len == 1 {
            return Self::paged_batched_decode_attention(
                ctx,
                q,
                k_pool,
                v_pool,
                out,
                block_tables,
                context_lens,
                num_seqs,
                max_kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_num_blocks_per_seq,
            );
        }

        // q_len > 1: prefill. Only single-seq is exercised — the only
        // caller (Qwen3MoeModel::forward_layer) always passes num_seqs=1.
        if num_seqs != 1 {
            return Err(FerrumError::model(format!(
                "paged_decode_attention(CUDA): q_len={q_len} num_seqs={num_seqs} \
                 not supported (caller must split prefill into per-seq calls)"
            )));
        }

        // Build cu_seqlens_q = [0, q_len] and pos_offsets = [final_kv_len - q_len].
        // alloc_u32 + write_u32 (NOT from_slice_i32 — that default goes
        // through f32→f16 and zeroes the bit pattern).
        // Need final_kv_len from context_lens[0] — D2H 4 bytes (cold path).
        let cl_host: Vec<u32> = {
            let stream = ctx.stream.clone();
            let view = unsafe {
                context_lens
                    .transmute::<u32>(1)
                    .ok_or_else(|| FerrumError::model("context_lens transmute failed"))?
            };
            let mut h = vec![0u32; 1];
            stream
                .memcpy_dtoh(&view, h.as_mut_slice())
                .map_err(|e| FerrumError::model(format!("dtoh context_lens: {e}")))?;
            stream
                .synchronize()
                .map_err(|e| FerrumError::model(format!("dtoh sync: {e}")))?;
            h
        };
        let final_kv_len = cl_host[0] as usize;
        if final_kv_len < q_len {
            return Err(FerrumError::model(format!(
                "paged_decode_attention(CUDA): final_kv_len={final_kv_len} < q_len={q_len}"
            )));
        }
        let pos_offset = (final_kv_len - q_len) as u32;
        let mut cu_seqlens_q_buf = <Self as Backend>::alloc_u32(2);
        <Self as Backend>::write_u32(ctx, &mut cu_seqlens_q_buf, &[0u32, q_len as u32]);
        let mut pos_offsets_buf = <Self as Backend>::alloc_u32(1);
        <Self as Backend>::write_u32(ctx, &mut pos_offsets_buf, &[pos_offset]);

        // The caller's q buffer (despite being named `q_head_major` in
        // Qwen3MoeModel) is ALREADY token-major in paged mode: the
        // paged-write kernel split_qkv_norm_rope_into_paged_cache_f16
        // writes `q_out[tok, head, hd]` (see kernel comment at
        // kernels/split_qkv_norm_rope_into_paged_cache.cu:102). So
        // paged_varlen_attention can read q directly. No Q transpose.
        //
        // Output, however, is written by paged_varlen as
        // `[M_total, num_q_heads, head_dim]` token-major (kernel:16),
        // while Qwen3MoeModel's downstream code does
        // `transpose_head_to_token(attn_head_major_out → attn_flat)`,
        // expecting head-major. We transpose token→head into `out`.
        let q_n = q_len * num_heads * head_dim;

        // Lazy-grow the cached token-major output scratch. Stable
        // address across calls — required to avoid stream-ordered
        // free / kernel-still-running races at higher concurrency.
        if ctx.paged_attn_out_tm_capacity < q_n {
            let stream = ctx.stream.clone();
            let n_grown = q_n.next_power_of_two().max(q_n);
            ctx.paged_attn_out_tm = Some(
                stream
                    .alloc_zeros::<f16>(n_grown)
                    .map_err(|e| FerrumError::model(format!("alloc paged_attn_out_tm: {e}")))?,
            );
            ctx.paged_attn_out_tm_capacity = n_grown;
        }

        // SAFETY: paged_varlen_attention only touches ctx.modules and
        // ctx.stream (disjoint from paged_attn_out_tm). Same for
        // transpose_token_to_head. We take a raw pointer to the cached
        // buffer so we can pass it as a normal &mut/& while ctx is also
        // borrowed by the kernel-call methods.
        let out_tm_ptr: *mut CudaSlice<f16> =
            ctx.paged_attn_out_tm
                .as_mut()
                .expect("paged_attn_out_tm allocated") as *mut _;
        unsafe {
            Self::paged_varlen_attention(
                ctx,
                q,
                k_pool,
                v_pool,
                &mut *out_tm_ptr,
                &cu_seqlens_q_buf,
                &pos_offsets_buf,
                block_tables,
                1,
                q_len,
                final_kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_num_blocks_per_seq,
            )?;

            // Restore head-major layout: [q_len, heads, hd] → [heads, q_len, hd]
            // → caller's `out` buffer.
            <Self as Backend>::transpose_token_to_head(
                ctx,
                &*out_tm_ptr,
                out,
                q_len,
                num_heads,
                head_dim,
            );
        }

        Ok(())
    }
    fn paged_batched_decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        block_tables: &Self::Buffer,
        valid_kv_lens: &Self::Buffer,
        num_seqs: usize,
        max_kv_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if num_seqs == 0 {
            return Ok(());
        }

        // Stage 13a: split-K path for batched paged decode. Default ON;
        // smart heuristic auto-tunes splits based on (num_seqs × num_heads)
        // and kv_len, so it falls back to single-pass when grid already
        // saturates SMs at low kv. Bench M3 c=1/8/16/32 across +21% / +3% /
        // +3.5% / +10.8% over the single-pass kernel — every concurrency
        // wins. Set FERRUM_PAGED_FLASH=0 to opt out.
        if std::env::var("FERRUM_PAGED_FLASH").map_or(true, |v| v != "0") {
            return paged_batched_flash_dispatch(
                ctx,
                q,
                k_pool,
                v_pool,
                out,
                block_tables,
                valid_kv_lens,
                num_seqs,
                max_kv_len,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_blocks_per_seq,
            );
        }

        let func = ctx.func(
            "paged_batched_decode_attn",
            ptx::PAGED_DECODE_ATTENTION,
            "paged_batched_decode_attn_f16",
        );
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
        let stream = ctx.stream.clone();
        let qv = q.slice(..);
        let kp = k_pool.slice(..);
        let vp = v_pool.slice(..);
        let bt = block_tables.slice(..);
        let kvl = valid_kv_lens.slice(..);
        let nqi = num_heads as i32;
        let nkvi = num_kv_heads as i32;
        let hdi = head_dim as i32;
        let mbps = max_blocks_per_seq as i32;
        let bsi = block_size as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(&qv);
        b.arg(&kp);
        b.arg(&vp);
        b.arg(&bt);
        b.arg(&kvl);
        b.arg(out);
        b.arg(&nqi);
        b.arg(&nkvi);
        b.arg(&hdi);
        b.arg(&mbps);
        b.arg(&bsi);
        b.arg(&scale);
        // Same shared-mem sizing rationale as paged_varlen_attention
        // (graph capture freezes shared_mem_bytes; size to
        // FERRUM_KV_CAPACITY ceiling).
        let safe_kv_max: usize = std::env::var("FERRUM_KV_CAPACITY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(512);
        let shared_kv = safe_kv_max.max(max_kv_len).max(1);
        let shared_bytes = (shared_kv as u32) * 4;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (num_heads as u32, num_seqs as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared_bytes,
            })
        }
        .map(|_| ())
        .map_err(|e| FerrumError::model(format!("paged_batched_decode_attn: {e}")))
    }
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        qkv_byte_offset: u64,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        q_out_byte_offset: u64,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        block_table: &Self::Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
        cache_len: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        let func = ctx.func(
            "split_qkv_norm_rope_into_paged_cache",
            ptx::SPLIT_QKV_NORM_ROPE_INTO_PAGED_CACHE,
            "split_qkv_norm_rope_into_paged_cache_f16",
        );
        let stream = ctx.stream.clone();
        let qkv_byte_offset_u64 = qkv_byte_offset;
        let q_out_byte_offset_u64 = q_out_byte_offset;
        let tokens_i32 = tokens as i32;
        let q_heads_i32 = q_heads as i32;
        let kv_heads_i32 = kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let pos_offset_i32 = pos_offset as i32;
        let cache_len_i32 = cache_len as i32;
        let block_size_i32 = block_size as i32;
        let max_blocks_per_seq_i32 = max_blocks_per_seq as i32;
        let qk_mode_i32 = qk_mode;
        let mut b = stream.launch_builder(&func);
        b.arg(qkv);
        b.arg(&qkv_byte_offset_u64);
        b.arg(q_norm_w);
        b.arg(k_norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(q_out);
        b.arg(&q_out_byte_offset_u64);
        b.arg(cache_k);
        b.arg(cache_v);
        b.arg(block_table);
        b.arg(&tokens_i32);
        b.arg(&q_heads_i32);
        b.arg(&kv_heads_i32);
        b.arg(&head_dim_i32);
        b.arg(&pos_offset_i32);
        b.arg(&eps);
        b.arg(&qk_mode_i32);
        b.arg(&cache_len_i32);
        b.arg(&block_size_i32);
        b.arg(&max_blocks_per_seq_i32);
        let total_heads = (q_heads + 2 * kv_heads) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (tokens as u32, total_heads, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| FerrumError::model(format!("split_qkv_norm_rope_into_paged_cache: {e}")))
    }
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache_varlen(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        cu_seqlens_q: &Self::Buffer,
        pos_offsets: &Self::Buffer,
        block_tables: &Self::Buffer,
        num_seqs: usize,
        m_total: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f32,
        qk_mode: i32,
        block_size: usize,
        max_blocks_per_seq: usize,
    ) -> Result<()> {
        if m_total == 0 || num_seqs == 0 {
            return Ok(());
        }
        let func = ctx.func(
            "split_qkv_norm_rope_into_paged_cache_varlen",
            ptx::SPLIT_QKV_NORM_ROPE_INTO_PAGED_CACHE,
            "split_qkv_norm_rope_into_paged_cache_varlen_f16",
        );
        let stream = ctx.stream.clone();
        let num_seqs_i32 = num_seqs as i32;
        let m_total_i32 = m_total as i32;
        let q_heads_i32 = q_heads as i32;
        let kv_heads_i32 = kv_heads as i32;
        let head_dim_i32 = head_dim as i32;
        let qk_mode_i32 = qk_mode;
        let block_size_i32 = block_size as i32;
        let max_blocks_per_seq_i32 = max_blocks_per_seq as i32;
        let mut b = stream.launch_builder(&func);
        b.arg(qkv);
        b.arg(q_norm_w);
        b.arg(k_norm_w);
        b.arg(cos);
        b.arg(sin);
        b.arg(q_out);
        b.arg(cache_k);
        b.arg(cache_v);
        b.arg(cu_seqlens_q);
        b.arg(pos_offsets);
        b.arg(block_tables);
        b.arg(&num_seqs_i32);
        b.arg(&m_total_i32);
        b.arg(&q_heads_i32);
        b.arg(&kv_heads_i32);
        b.arg(&head_dim_i32);
        b.arg(&eps);
        b.arg(&qk_mode_i32);
        b.arg(&block_size_i32);
        b.arg(&max_blocks_per_seq_i32);
        let total_heads = (q_heads + 2 * kv_heads) as u32;
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (m_total as u32, total_heads, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| {
            FerrumError::model(format!("split_qkv_norm_rope_into_paged_cache_varlen: {e}"))
        })
    }
}

impl BackendMoeFused for CudaBackend {
    fn route_topk_softmax(
        ctx: &mut Self::Context,
        logits: &Self::Buffer,
        out_ids: &mut Self::Buffer,
        out_weights: &mut Self::Buffer,
        batch: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<()> {
        // Block: one warp (32 threads), one block per row.
        // Shared mem: num_experts × 4 bytes (per-row probability vector
        // in fp32). At Qwen3-MoE num_experts=128 this is 512 bytes /
        // block — far below the 48 KB / SM limit. Larger MoE configs
        // (DeepSeek 256 experts) still only use 1 KB.
        let func = ctx.func(
            "moe_router_topk_softmax",
            ptx::MOE_ROUTER,
            "moe_router_topk_softmax_f16",
        );
        let batch_i32 = batch as i32;
        let n_exp_i32 = num_experts as i32;
        let top_k_i32 = top_k as i32;
        let norm_i32 = if norm_topk_prob { 1i32 } else { 0i32 };
        let smem_bytes = (num_experts as u32) * 4;

        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(logits);
        b.arg(out_ids);
        b.arg(out_weights);
        b.arg(&batch_i32);
        b.arg(&n_exp_i32);
        b.arg(&top_k_i32);
        b.arg(&norm_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (batch as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: smem_bytes,
            })
        }
        .map_err(|e| FerrumError::model(format!("moe_router launch: {e}")))?;
        Ok(())
    }
    fn try_gpu_route_topk_into_host(
        ctx: &mut Self::Context,
        logits_dev: &Self::Buffer,
        out_ids_host: &mut Vec<u32>,
        out_weights_host: &mut Vec<f32>,
        batch: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<()> {
        let total_pairs = batch * top_k;

        // Lazy-init the scratch device buffers. Sized to total_pairs;
        // grow if a larger shape shows up. i32 storage = 4*total_pairs
        // bytes = 2*total_pairs f16 elements; f32 storage same (4 bytes
        // per element).
        if ctx.moe_route_capacity < total_pairs {
            let stream = ctx.stream.clone();
            // 2 × total_pairs because each i32 / f32 element needs 4
            // bytes = 2 f16 slots in the underlying CudaSlice<f16>.
            let nf16 = 2 * total_pairs;
            ctx.moe_route_ids = Some(
                stream
                    .alloc_zeros::<f16>(nf16)
                    .map_err(|e| FerrumError::model(format!("alloc moe_route_ids: {e}")))?,
            );
            ctx.moe_route_weights = Some(
                stream
                    .alloc_zeros::<f16>(nf16)
                    .map_err(|e| FerrumError::model(format!("alloc moe_route_weights: {e}")))?,
            );
            ctx.moe_route_capacity = total_pairs;
        }

        // 1. Launch the kernel into the cached scratch. Scoped so the
        // launch_builder (which moves the &mut buffer references) drops
        // before we re-borrow them immutably for the D2H phase.
        let func = ctx.func(
            "moe_router_topk_softmax",
            ptx::MOE_ROUTER,
            "moe_router_topk_softmax_f16",
        );
        let batch_i32 = batch as i32;
        let n_exp_i32 = num_experts as i32;
        let top_k_i32 = top_k as i32;
        let norm_i32 = if norm_topk_prob { 1i32 } else { 0i32 };
        let smem_bytes = (num_experts as u32) * 4;

        let stream = ctx.stream.clone();
        {
            let ids_dev = ctx
                .moe_route_ids
                .as_mut()
                .expect("moe_route_ids should be allocated");
            let weights_dev = ctx
                .moe_route_weights
                .as_mut()
                .expect("moe_route_weights should be allocated");

            let mut b = stream.launch_builder(&func);
            b.arg(logits_dev);
            b.arg(ids_dev);
            b.arg(weights_dev);
            b.arg(&batch_i32);
            b.arg(&n_exp_i32);
            b.arg(&top_k_i32);
            b.arg(&norm_i32);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (batch as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: smem_bytes,
                })
            }
            .map_err(|e| FerrumError::model(format!("moe_router launch: {e}")))?;
        }

        // 2. D2H ids (i32) and weights (f32) into the host destinations.
        out_ids_host.clear();
        out_ids_host.resize(total_pairs, 0u32);
        out_weights_host.clear();
        out_weights_host.resize(total_pairs, 0.0f32);

        let ids_dev = ctx
            .moe_route_ids
            .as_ref()
            .expect("moe_route_ids should be allocated");
        let weights_dev = ctx
            .moe_route_weights
            .as_ref()
            .expect("moe_route_weights should be allocated");

        // Reinterpret the f16-typed scratch as i32 / f32 views. transmute
        // verifies byte-fit (returns None if undersized).
        let ids_view = unsafe {
            ids_dev
                .transmute::<i32>(total_pairs)
                .ok_or_else(|| FerrumError::model("ids transmute size mismatch"))?
        };
        let weights_view = unsafe {
            weights_dev
                .transmute::<f32>(total_pairs)
                .ok_or_else(|| FerrumError::model("weights transmute size mismatch"))?
        };

        // out_ids_host is Vec<u32>; reinterpret as &mut [i32] for the
        // memcpy. Same byte pattern.
        let out_ids_i32: &mut [i32] = unsafe {
            std::slice::from_raw_parts_mut(out_ids_host.as_mut_ptr() as *mut i32, total_pairs)
        };
        stream
            .memcpy_dtoh(&ids_view, out_ids_i32)
            .map_err(|e| FerrumError::model(format!("dtoh route ids: {e}")))?;
        stream
            .memcpy_dtoh(&weights_view, out_weights_host.as_mut_slice())
            .map_err(|e| FerrumError::model(format!("dtoh route weights: {e}")))?;
        // Synchronize so the host can read the results immediately.
        stream
            .synchronize()
            .map_err(|e| FerrumError::model(format!("dtoh sync: {e}")))?;

        Ok(())
    }
    fn moe_align_block_size(
        ctx: &mut Self::Context,
        expert_ids_per_pair: &Self::Buffer,
        sorted_token_ids: &mut Self::Buffer,
        block_ids: &mut Self::Buffer,
        total_tokens_post_pad: &mut Self::Buffer,
        batch_x_topk: usize,
        num_experts: usize,
        block_size: usize,
        sorted_max_size: usize,
    ) -> Result<()> {
        if num_experts > 256 {
            return Err(FerrumError::model(format!(
                "moe_align_block_size: num_experts={num_experts} exceeds compile-time MAX_NUM_EXPERTS=256"
            )));
        }
        let func = ctx.func(
            "moe_align_block_size",
            ptx::MOE_ALIGN_BLOCK_SIZE,
            "moe_align_block_size_f32",
        );
        let n = batch_x_topk as i32;
        let ne = num_experts as i32;
        let bs = block_size as i32;
        let smax = sorted_max_size as i32;
        let stream = ctx.stream.clone();
        // Single block — algorithm uses shared mem for counts + offsets,
        // sized to MAX_NUM_EXPERTS=256. Use 256 threads to cover the
        // ≤256-experts and ≤1024-pair Qwen3-MoE configs cleanly.
        let mut b = stream.launch_builder(&func);
        b.arg(expert_ids_per_pair);
        b.arg(sorted_token_ids);
        b.arg(block_ids);
        b.arg(total_tokens_post_pad);
        b.arg(&n);
        b.arg(&ne);
        b.arg(&bs);
        b.arg(&smax);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| FerrumError::model(format!("moe_align_block_size launch: {e}")))?;
        Ok(())
    }
    fn moe_combine(
        ctx: &mut Self::Context,
        packed_down: &Self::Buffer,
        pairs_by_token: &[i32],
        pair_weights: &[f32],
        out: &mut Self::Buffer,
        batch: usize,
        hidden: usize,
        top_k: usize,
        _total_pairs: usize,
    ) {
        debug_assert_eq!(pairs_by_token.len(), batch * top_k);
        debug_assert_eq!(pair_weights.len(), batch * top_k);

        let stream = ctx.stream.clone();
        let pairs_dev = stream
            .clone_htod(pairs_by_token)
            .expect("moe_combine pairs htod");
        let weights_dev = stream
            .clone_htod(pair_weights)
            .expect("moe_combine weights htod");

        let func = ctx.func("moe_combine", ptx::MOE_COMBINE, "moe_combine_f16");
        let batch_i32 = batch as i32;
        let hidden_i32 = hidden as i32;
        let top_k_i32 = top_k as i32;

        let block = 256u32;
        let grid_x = ((hidden as u32) + block - 1) / block;

        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(packed_down);
        b.arg(&pairs_dev);
        b.arg(&weights_dev);
        b.arg(out);
        b.arg(&batch_i32);
        b.arg(&hidden_i32);
        b.arg(&top_k_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid_x, batch as u32, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("moe_combine launch");
    }
    #[cfg(feature = "vllm-moe-marlin")]
    fn upload_moe_routing(
        ctx: &mut Self::Context,
        sorted_token_ids: &[i32],
        expert_ids: &[i32],
        num_tokens_past_padded: &[i32],
    ) -> Result<crate::backend::traits::MoeRouting<Self>> {
        use cudarc::driver::sys::CUdeviceptr;
        use cudarc::driver::CudaSlice;
        use cudarc::driver::DevicePtr;

        // Per-call alloc + leak. Persistent buffers (one-alloc + reuse)
        // were attempted as Stage 13c-1 but hit a CUDA driver edge case
        // — leaving the bench-validated path in place for now.
        // TODO(stage-13c-redo): retry with stream sync between forwards.
        let stream = ctx.stream.clone();
        let st: CudaSlice<i32> = stream
            .clone_htod(sorted_token_ids)
            .map_err(|e| FerrumError::model(format!("htod sorted_token_ids: {e}")))?;
        let eid: CudaSlice<i32> = stream
            .clone_htod(expert_ids)
            .map_err(|e| FerrumError::model(format!("htod expert_ids: {e}")))?;
        let npp: CudaSlice<i32> = stream
            .clone_htod(num_tokens_past_padded)
            .map_err(|e| FerrumError::model(format!("htod num_tokens_past_padded: {e}")))?;
        let (st_ptr, eid_ptr, npp_ptr) = {
            let (st_ptr, _g0) = st.device_ptr(&stream);
            let (eid_ptr, _g1) = eid.device_ptr(&stream);
            let (npp_ptr, _g2) = npp.device_ptr(&stream);
            (st_ptr, eid_ptr, npp_ptr)
        };
        let st_f16: CudaSlice<f16> = unsafe { stream.upgrade_device_ptr(st_ptr as CUdeviceptr, 0) };
        let eid_f16: CudaSlice<f16> =
            unsafe { stream.upgrade_device_ptr(eid_ptr as CUdeviceptr, 0) };
        let npp_f16: CudaSlice<f16> =
            unsafe { stream.upgrade_device_ptr(npp_ptr as CUdeviceptr, 0) };
        // Forget the i32 owning slices so the memory survives — the f16
        // views above point at the same allocation. Acceptable leak: a
        // few KB per layer × 48 layers/forward × forwards/sec = ~MB/s
        // sustained, freed only at process exit. Not ideal but works.
        // TODO(stage-13c-redo): re-attempt persistent buffers once we
        // understand why memcpy_htod / cuMemcpyHtoDAsync rejects the
        // 2nd call's params under our driver version.
        std::mem::forget(st);
        std::mem::forget(eid);
        std::mem::forget(npp);

        Ok(crate::backend::traits::MoeRouting {
            sorted_token_ids: st_f16,
            expert_ids: eid_f16,
            num_tokens_past_padded: npp_f16,
        })
    }
}
// CUDA: existing KV cache path is FP16.
impl crate::backend::BackendKvDtype<crate::backend::KvFp16> for CudaBackend {
    type KvBuffer = <Self as crate::backend::Backend>::Buffer;
    type KvScales = ();
}
