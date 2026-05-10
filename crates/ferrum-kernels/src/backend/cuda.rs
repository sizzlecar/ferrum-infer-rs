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
// GPTQ store dispatch — Marlin (default) vs Triton-rs (FERRUM_TRITON_INT4=1)
// ────────────────────────────────────────────────────────────────────────

/// CUDA-side GPTQ weight, pre-loaded for one of two fused INT4×FP16 paths.
///
/// Selected once per weight at load time (`load_gptq`) based on the
/// `FERRUM_TRITON_INT4` env var:
///   - unset / `0`: build `Marlin` (the existing default — Marlin's tile
///     repack runs on the CPU; inference uses the hand-tuned CUTLASS
///     kernel from IST-DASLab/marlin).
///   - `1`: build `Triton` (port of vLLM's `triton_w4a16_gemm_kernel`,
///     adapted to ferrum's on-disk GPTQ layout — no host-side repack).
///
/// The two stores cannot coexist for the same layer (load-time gating),
/// so a process-wide A/B comparison runs as two separate `bench`
/// invocations. This avoids doubling VRAM and keeps the dispatch
/// branchless on the hot path.
#[cfg(feature = "triton-kernels")]
pub enum GptqStoreCuda {
    Marlin(crate::marlin::MarlinWeight),
    Triton(crate::triton_w4a16::TritonGptqWeight),
}

/// When triton-kernels is disabled at compile time the GPTQ store is just
/// a transparent re-alias of `MarlinWeight`. Keeping this typed alias
/// (rather than `pub use`) means the trait impl block in `backend::cuda`
/// can stay shape-identical between the two cfgs.
#[cfg(not(feature = "triton-kernels"))]
pub type GptqStoreCuda = crate::marlin::MarlinWeight;

/// Read `FERRUM_TRITON_INT4` once. Returns true iff `=1`. Anything else
/// (unset, `0`, empty, garbage) → false.
fn use_triton_int4() -> bool {
    std::env::var("FERRUM_TRITON_INT4").map_or(false, |v| v == "1")
}

/// Read `FERRUM_VLLM_MOE` once. Returns true iff `=1`. Selects the
/// vendored vLLM marlin_moe_wna16 path for stacked-MoE GPTQ INT4
/// weights (load + dispatch pair must be enabled together).
#[cfg(feature = "vllm-moe-marlin")]
pub(crate) fn use_vllm_moe() -> bool {
    std::env::var("FERRUM_VLLM_MOE").map_or(false, |v| v == "1")
}

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

impl BackendCollective for CudaBackend {
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

impl BackendGraph for CudaBackend {
    fn set_decode_state(ctx: &mut Self::Context, token: u32, step: u32) {
        let valid_kv = (step as i32) + 1;
        let step_i = step as i32;
        let stream = ctx.stream.clone();
        let mut w = decode_state_slot().write().expect("DECODE_STATE poisoned");
        let bufs = w.as_mut().expect("DecodeStateBufs not initialised");
        stream
            .memcpy_htod(&[token], &mut bufs.token)
            .expect("token_buf memcpy");
        stream
            .memcpy_htod(&[step_i], &mut bufs.pos)
            .expect("pos_buf memcpy");
        stream
            .memcpy_htod(&[valid_kv], &mut bufs.kv)
            .expect("kv_buf memcpy");
    }

    fn set_dev_state_mode(ctx: &mut Self::Context, enable: bool) {
        ctx.use_dev_state = enable;
    }

    fn begin_graph_capture(ctx: &mut Self::Context) -> Result<()> {
        use cudarc::driver::sys::CUstreamCaptureMode;
        // Event tracking already disabled globally in default_stream; begin
        // capture directly in relaxed mode. Bare-Rust cudarc reproducer
        // confirms this configuration works on Blackwell + CUDA 13
        // (`cudarc_graph_no_event_tracking` test). The full ferrum bench
        // path still SIGSEGVs though — remaining delta is likely one of
        // PTX module load timing, cuBLAS workspace interaction, or a
        // specific kernel's use of constant memory that doesn't survive
        // capture. See `docs/phase-e-cuda-status.md` graph section.
        ctx.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| FerrumError::unsupported(format!("begin_capture: {e}")))?;
        ctx.capture_in_flight = true;
        Ok(())
    }

    fn end_graph_capture(ctx: &mut Self::Context, key: u64) -> Result<()> {
        use cudarc::driver::sys;
        if !ctx.capture_in_flight {
            return Err(FerrumError::unsupported("end_capture without begin"));
        }
        ctx.capture_in_flight = false;

        // Bypass cudarc's end_capture — it does cuStreamEndCapture +
        // cuGraphInstantiateWithFlags in one call, and one of those corrupts
        // the context on Blackwell. Call them separately so we can see which.
        ctx.ctx
            .bind_to_thread()
            .map_err(|e| FerrumError::unsupported(format!("bind pre-end: {e}")))?;

        let cu_stream = ctx.stream.cu_stream();
        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        let st1 = unsafe { sys::cuStreamEndCapture(cu_stream, &mut cu_graph) };
        if st1 != sys::CUresult::CUDA_SUCCESS {
            return Err(FerrumError::unsupported(format!(
                "cuStreamEndCapture failed: {st1:?}"
            )));
        }
        if cu_graph.is_null() {
            return Err(FerrumError::unsupported(
                "cuStreamEndCapture returned null graph",
            ));
        }

        // flags=0: no AUTO_FREE_ON_LAUNCH. The captured graph contains
        // only kernel launches (memcpys are sync via cuMemcpyHtoD_v2
        // outside capture, see populate_batched_pointers), so
        // AUTO_FREE has nothing to free. With AUTO_FREE on, replays
        // worked for ~14 iters then SIGSEGV — likely device-side
        // launch resources getting freed mid-launch sequence.
        let flags = 0u64;
        let mut cu_graph_exec: sys::CUgraphExec = std::ptr::null_mut();
        let st2 = unsafe { sys::cuGraphInstantiateWithFlags(&mut cu_graph_exec, cu_graph, flags) };
        if st2 != sys::CUresult::CUDA_SUCCESS {
            unsafe {
                sys::cuGraphDestroy(cu_graph);
            }
            return Err(FerrumError::unsupported(format!(
                "cuGraphInstantiate failed: {st2:?}"
            )));
        }

        // Upload graph to GPU before first launch. Without this, the first
        // cuGraphLaunch does lazy JIT + resource upload while the stream
        // still has pending ops — libcuda dereferences not-yet-uploaded
        // graph state and SIGSEGVs on Blackwell + CUDA 13.
        let st3 = unsafe { sys::cuGraphUpload(cu_graph_exec, cu_stream) };
        if st3 != sys::CUresult::CUDA_SUCCESS {
            unsafe {
                sys::cuGraphExecDestroy(cu_graph_exec);
                sys::cuGraphDestroy(cu_graph);
            }
            return Err(FerrumError::unsupported(format!(
                "cuGraphUpload failed: {st3:?}"
            )));
        }

        // Install into the multi-slot cache keyed by `key`. Replaces any
        // existing graph for the same key; the old GraphSlotRaw drops
        // (cuCtxSync + cuGraphExecDestroy + cuGraphDestroy in its Drop
        // impl) before the new one takes its place.
        install_decode_graph_raw(key, cu_graph, cu_graph_exec, ctx.stream.clone());
        Ok(())
    }

    fn reset_graph(_ctx: &mut Self::Context, key: u64) {
        invalidate_decode_graph(key);
    }

    fn reset_all_graphs(_ctx: &mut Self::Context) {
        invalidate_all_decode_graphs();
    }

    fn replay_graph(ctx: &mut Self::Context, key: u64) -> Result<bool> {
        use cudarc::driver::sys;
        let cu_stream = ctx.stream.cu_stream();
        ctx.ctx
            .bind_to_thread()
            .map_err(|e| FerrumError::unsupported(format!("bind pre-replay: {e}")))?;
        with_decode_graph(key, |g_opt| {
            if let Some(g) = g_opt {
                let prof = std::env::var("FERRUM_GRAPH_PROF").is_ok();
                let t_pre = if prof {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                // Re-upload before each launch. Without it, c=4 throughput
                // drops 257→178 tok/s (post-Phase-8 measurement). The
                // graph instantiate-then-upload-once design didn't pan out
                // empirically; keep the per-replay upload until we
                // understand why removing it slows things down.
                let skip_upload =
                    std::env::var("FERRUM_GRAPH_SKIP_UPLOAD").map_or(false, |v| v == "1");
                if !skip_upload {
                    let st_up = unsafe { sys::cuGraphUpload(g.cu_graph_exec, cu_stream) };
                    if st_up != sys::CUresult::CUDA_SUCCESS {
                        return Err(FerrumError::unsupported(format!(
                            "cuGraphUpload: {st_up:?}"
                        )));
                    }
                }
                let t_after_upload = if prof {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let st = unsafe { sys::cuGraphLaunch(g.cu_graph_exec, cu_stream) };
                if st != sys::CUresult::CUDA_SUCCESS {
                    return Err(FerrumError::unsupported(format!("cuGraphLaunch: {st:?}")));
                }
                let t_after_launch = if prof {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let skip_sync = std::env::var("FERRUM_GRAPH_SKIP_SYNC").map_or(false, |v| v == "1");
                if !skip_sync {
                    let st_sync = unsafe { sys::cuStreamSynchronize(cu_stream) };
                    if st_sync != sys::CUresult::CUDA_SUCCESS {
                        return Err(FerrumError::unsupported(format!(
                            "post-launch sync: {st_sync:?}"
                        )));
                    }
                }
                if let (Some(t0), Some(t1), Some(t2)) = (t_pre, t_after_upload, t_after_launch) {
                    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n.is_multiple_of(64) {
                        let upload = t1.duration_since(t0).as_micros();
                        let launch = t2.duration_since(t1).as_micros();
                        let sync = t2.elapsed().as_micros();
                        eprintln!(
                            "[graph-prof] call#{n} upload={upload}us launch={launch}us sync={sync}us total={}us",
                            t0.elapsed().as_micros()
                        );
                    }
                }
                Ok(true)
            } else {
                Ok(false)
            }
        })
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

/// Raw graph slot holding cuGraph + cuGraphExec pointers directly, bypassing
/// cudarc's CudaGraph wrapper. The wrapper's `end_capture` does
/// cuStreamEndCapture + cuGraphInstantiateWithFlags in one non-overridable
/// call, and one of those corrupts the context on Blackwell; bypassing lets
/// us split the FFI calls and choose which instantiate variant to use.
struct GraphSlotRaw {
    cu_graph: cudarc::driver::sys::CUgraph,
    cu_graph_exec: cudarc::driver::sys::CUgraphExec,
    // Keep the stream Arc alive so its underlying cu_stream stays valid.
    _stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

impl Drop for GraphSlotRaw {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        unsafe {
            // Sync device before destroying graph resources to ensure no
            // kernel launches from this graph are still in flight.
            sys::cuCtxSynchronize();
            if !self.cu_graph_exec.is_null() {
                sys::cuGraphExecDestroy(self.cu_graph_exec);
            }
            if !self.cu_graph.is_null() {
                sys::cuGraphDestroy(self.cu_graph);
            }
            // Sync again after destroy so any cleanup completes.
            sys::cuCtxSynchronize();
        }
    }
}

// SAFETY: graph launch is serialised through the model's stream, which
// is accessed from one thread at a time (engine Mutex-wraps the model).
unsafe impl Send for GraphSlotRaw {}
unsafe impl Sync for GraphSlotRaw {}

// Multi-slot graph cache, keyed by an opaque `u64`. Caller chooses the
// key — the model uses `m_padded` (or 0 for single-item) so that
// different batch shapes get their own captured graph instead of
// thrashing a single slot at every shape change.
//
// Native CUDA microbench (graph_upload_bench.cu, 320 launches × 500 iters,
// alternating two graph sizes) confirmed multi-slot replay is stable
// at ~0.26ms/iter with no degradation vs single slot.
static DECODE_GRAPHS: std::sync::OnceLock<std::sync::RwLock<HashMap<u64, GraphSlotRaw>>> =
    std::sync::OnceLock::new();

fn graph_slots() -> &'static std::sync::RwLock<HashMap<u64, GraphSlotRaw>> {
    DECODE_GRAPHS.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
}

fn install_decode_graph_raw(
    key: u64,
    cu_graph: cudarc::driver::sys::CUgraph,
    cu_graph_exec: cudarc::driver::sys::CUgraphExec,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
) {
    let mut g = graph_slots().write().expect("DECODE_GRAPHS poisoned");
    g.insert(
        key,
        GraphSlotRaw {
            cu_graph,
            cu_graph_exec,
            _stream: stream,
        },
    );
}

fn with_decode_graph<R>(key: u64, f: impl FnOnce(Option<&GraphSlotRaw>) -> Result<R>) -> Result<R> {
    let guard = graph_slots().read().expect("DECODE_GRAPHS poisoned");
    f(guard.get(&key))
}

/// Evict ONE cached graph — call when its kernel-arg pointers (KV cache,
/// scratch buffers) might be invalidated.
pub fn invalidate_decode_graph(key: u64) {
    graph_slots()
        .write()
        .expect("DECODE_GRAPHS poisoned")
        .remove(&key);
}

/// Evict ALL cached graphs — used by hard reset (model reload, scratch
/// realloc) when every captured pointer might be stale.
pub fn invalidate_all_decode_graphs() {
    graph_slots()
        .write()
        .expect("DECODE_GRAPHS poisoned")
        .clear();
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

fn decode_state_slot() -> &'static std::sync::RwLock<Option<DecodeStateBufs>> {
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
// Perm-aware Marlin GEMM (desc_act=true GPTQ act-order)
// ────────────────────────────────────────────────────────────────────────
//
// When MarlinWeight.perm is Some, the qweight rows have been permuted at
// load time by `argsort(g_idx)`. For the standard Marlin kernel to compute
// the un-permuted GEMM result, we gather input columns by the same perm
// before the call: input_perm[m, j] = input[m, perm[j]]. After:
//   y[n] = Σⱼ input_perm[m, j] · dequant(qweight_sorted[j, n])
//        = Σⱼ input[m, perm[j]] · dequant(qweight[perm[j], n])  (let k = perm[j])
//        = Σ_k input[m, k] · dequant(qweight[k, n])             ← un-permuted GEMM
//
// Same approach as vLLM's gptq_marlin runtime gather. Adds one f16 column
// gather kernel per Marlin call (~10us on H≤14336 / m≤32) — net cost
// dominated by the Marlin GEMM that follows, not the gather.

#[cfg(feature = "marlin")]
/// Process-global scratch buffer for the perm-aware Marlin's `a_gathered`
/// staging slot. Without this, every `marlin_gemm_with_perm` call did
/// `stream.alloc::<f16>(m * k)` and dropped at scope end — cudarc's
/// allocator pool grew unboundedly (≥ 32 MB / iter × 200 iters → ~6 GB
/// VRAM after one c=16 bench rep, OOM on rep 2). Now we keep one slot
/// per process, grown on demand to the largest `m * k` ever requested.
struct MarlinGatherScratch {
    buf: CudaSlice<f16>,
    capacity: usize, // in f16 elements
}
unsafe impl Send for MarlinGatherScratch {}
unsafe impl Sync for MarlinGatherScratch {}

static MARLIN_GATHER_SCRATCH: std::sync::OnceLock<std::sync::RwLock<Option<MarlinGatherScratch>>> =
    std::sync::OnceLock::new();

fn marlin_gather_scratch_slot() -> &'static std::sync::RwLock<Option<MarlinGatherScratch>> {
    MARLIN_GATHER_SCRATCH.get_or_init(|| std::sync::RwLock::new(None))
}

/// Run `body` with a mut ref to a Marlin-gather scratch buffer of at
/// least `required` f16 elements. Reuses the existing slot if it fits;
/// reallocates (replacing the old one) if it needs to grow.
/// Pre-grow the marlin_gather_scratch slot to at least `required`
/// elements. Idempotent. Used by callers that are about to enter a
/// CUDA-graph capture region — `with_marlin_gather_scratch`'s
/// in-place grow does `stream.alloc::<f16>(required)`, and CUDA
/// graph capture rejects runtime allocs inside the captured region
/// with `CUDA_ERROR_INVALID_VALUE`. Pre-warming OUTSIDE the capture
/// keeps the alloc eager, the capture-region kernel launches just
/// re-use the existing slot's pointer.
#[cfg(feature = "marlin")]
pub fn pregrow_marlin_gather_scratch(stream: &Arc<CudaStream>, required: usize) {
    let slot = marlin_gather_scratch_slot();
    {
        let g = slot.read().expect("MARLIN_GATHER_SCRATCH poisoned");
        if let Some(ref s) = *g {
            if s.capacity >= required {
                return;
            }
        }
    }
    let mut w = slot.write().expect("MARLIN_GATHER_SCRATCH poisoned");
    let need_new = match &*w {
        Some(s) => s.capacity < required,
        None => true,
    };
    if need_new {
        let buf = unsafe { stream.alloc::<f16>(required) }
            .expect("MARLIN_GATHER_SCRATCH pregrow alloc failed");
        *w = Some(MarlinGatherScratch {
            buf,
            capacity: required,
        });
    }
}

fn with_marlin_gather_scratch<R>(
    stream: &Arc<CudaStream>,
    required: usize,
    body: impl FnOnce(&mut CudaSlice<f16>) -> R,
) -> R {
    let slot = marlin_gather_scratch_slot();
    {
        let g = slot.read().expect("MARLIN_GATHER_SCRATCH poisoned");
        if let Some(ref s) = *g {
            if s.capacity >= required {
                drop(g);
                let mut w = slot.write().expect("MARLIN_GATHER_SCRATCH poisoned");
                let s = w.as_mut().expect("just observed Some");
                return body(&mut s.buf);
            }
        }
    }
    // Need to (re)allocate.
    let mut w = slot.write().expect("MARLIN_GATHER_SCRATCH poisoned");
    let need_new = match &*w {
        Some(s) => s.capacity < required,
        None => true,
    };
    if need_new {
        let buf =
            unsafe { stream.alloc::<f16>(required) }.expect("MARLIN_GATHER_SCRATCH alloc failed");
        *w = Some(MarlinGatherScratch {
            buf,
            capacity: required,
        });
    }
    let s = w.as_mut().expect("just allocated");
    body(&mut s.buf)
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

// ────────────────────────────────────────────────────────────────────────
// Stage 12 — Fused MoE Marlin: bucket dispatches by max-m, fire one
// `marlin_gemm_moe` per bucket. ONE launch per bucket replaces N round-
// robin calls of the multi-stream path.
//
// Buckets are by ceil(m / 16) ∈ {1, 2, 3, 4}. Caller's dispatch list
// (built by moe_forward_bucketed) gives (expert_idx, in_row, out_row, m)
// per active expert. Inactive experts (m=0) are not in the list.
//
// Per-call uploads: one i32 array of size num_experts_used (active
// experts in this dispatch list) for active_expert_ids, and one
// indexed-by-active-position array for tokens + a_row_offsets. Total
// staging is ~3 × num_active × 4 bytes ≈ 1 KB at c=32 — htod cost is
// O(1) per phase, dwarfed by the launch savings.
//
// Workspace: caller already calls `marlin_zero_stacked_workspace` per
// phase; the fused kernel reuses the same per-expert workspace ranges.
// ────────────────────────────────────────────────────────────────────────
#[cfg(feature = "marlin")]
fn moe_gemm_phase_fused_impl(
    ctx: &mut CudaState,
    input: &CudaSlice<f16>,
    weight: &crate::marlin::MarlinWeight,
    dispatches: &[(usize, usize, usize, usize)],
    n_per_expert: usize,
    output: &mut CudaSlice<f16>,
    _k: usize,
) -> Result<()> {
    if dispatches.is_empty() {
        return Ok(());
    }
    let num_active = dispatches.len();

    // Bucket by ceil(m / 16). All 4 buckets share the same per-active
    // tokens + a_row_offsets layout (indexed by active position, NOT
    // global expert id) — the kernel's `tokens_per_expert[e]` and
    // `A_row_offsets[e]` use the bucket-local index `e_local`. Caller's
    // `active_expert_ids[e_local]` maps that to the global expert id.
    //
    // The kernel reads:
    //   int e_local = blockIdx.y;
    //   int e_global = active_expert_ids[e_local];
    //   int row_start = A_row_offsets[e_global];
    //   int m_e       = tokens_per_expert[e_global];
    // So `tokens_per_expert` and `A_row_offsets` MUST be sized to the
    // global expert id space and indexed by the global id.
    //
    // We don't know num_experts_global directly from the dispatch list,
    // but max(expert_idx) + 1 is a safe lower bound. Real production
    // usage at c=32 has num_experts_global ≤ 256 (Qwen3-MoE: 128).
    let max_global_e = dispatches.iter().map(|d| d.0).max().unwrap();
    let num_experts_global = max_global_e + 1;

    // Build the per-global-expert index arrays. Both default to 0; only
    // active experts are populated. Inactive experts won't be referenced
    // by active_expert_ids[] so their tokens=0 / row=0 values are dead.
    let mut tokens_global = vec![0i32; num_experts_global];
    let mut row_offsets_global = vec![0i32; num_experts_global];
    let mut bucket_active_ids: [Vec<i32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for &(e_idx, in_row, _out_row, m_e) in dispatches {
        debug_assert!(m_e > 0);
        debug_assert!(m_e <= 64);
        tokens_global[e_idx] = m_e as i32;
        row_offsets_global[e_idx] = in_row as i32;
        let bucket = ((m_e + 15) / 16).clamp(1, 4) - 1;
        bucket_active_ids[bucket].push(e_idx as i32);
    }

    // Upload the global per-expert arrays once.
    let stream = ctx.stream.clone();
    let row_off_dev = stream
        .clone_htod(&row_offsets_global)
        .map_err(|e| FerrumError::model(format!("htod row_offsets: {e}")))?;
    let tok_dev = stream
        .clone_htod(&tokens_global)
        .map_err(|e| FerrumError::model(format!("htod tokens: {e}")))?;

    // Fire one launch per non-empty bucket.
    for (b, ids) in bucket_active_ids.iter().enumerate() {
        if ids.is_empty() {
            continue;
        }
        let prob_m_bucket = ((b + 1) * 16) as i32;
        let active_dev = stream
            .clone_htod(ids)
            .map_err(|e| FerrumError::model(format!("htod active_ids[b={b}]: {e}")))?;
        crate::marlin::marlin_gemm_moe(
            &stream,
            input,
            weight,
            output,
            &row_off_dev,
            &tok_dev,
            Some(&active_dev),
            ids.len() as i32,
            prob_m_bucket,
            n_per_expert as i32,
            num_experts_global as i32,
        )
        .map_err(|e| FerrumError::model(format!("marlin_gemm_moe (bucket={b}): {e}")))?;
    }

    let _ = num_active;
    Ok(())
}

/// Marlin GEMM dispatcher that handles act-order permutation if present.
///
/// Made `pub` in Phase 3e/1 so the new `CudaMarlinLinear` impl can call
/// it from outside the trait method body. Same dispatch logic the
/// `BackendQuantMarlin::gemm_gptq` impl used to wrap.
pub fn marlin_gemm_with_perm(
    ctx: &mut CudaState,
    a: &CudaSlice<f16>,
    weight: &crate::marlin::MarlinWeight,
    out: &mut CudaSlice<f16>,
    m: usize,
) -> Result<()> {
    let use_vllm = std::env::var("FERRUM_VLLM_MARLIN").map_or(false, |v| v == "1");

    if let Some(perm) = weight.perm.as_ref() {
        let k = weight.k;
        let stream = ctx.stream.clone();
        let func = ctx.func("gather_columns", ptx::GATHER_COLUMNS, "gather_columns_f16");
        let m_i32 = m as i32;
        let k_i32 = k as i32;
        let block_x: u32 = 512;
        let grid_y: u32 = ((k as u32) + block_x - 1) / block_x;
        // Borrow the gather-scratch slot for the entire (gather +
        // marlin_gemm) sequence so the GPU work using `a_gathered`
        // completes (or at least gets stream-ordered) before another
        // marlin_gemm_with_perm caller can grab the same buffer.
        // RwLock write-guard is held across the launch+marlin_gemm
        // calls; both queue async on the same `stream`, and the next
        // caller's Marlin work also serialises on this stream — so
        // even a weaker single-slot pool is correct as long as we
        // don't leave the function before the kernel queues complete.
        with_marlin_gather_scratch(&stream, m * k, |a_gathered| -> Result<()> {
            let mut b = stream.launch_builder(&func);
            b.arg(a);
            b.arg(perm);
            b.arg(&mut *a_gathered);
            b.arg(&m_i32);
            b.arg(&k_i32);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (m as u32, grid_y, 1),
                    block_dim: (block_x, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| FerrumError::model(format!("gather_columns launch: {e}")))?;
            if use_vllm {
                return launch_vllm_marlin(&ctx.stream, a_gathered, weight, out, m);
            }
            crate::marlin::marlin_gemm(&ctx.stream, a_gathered, weight, out, m as i32)
                .map_err(|e| FerrumError::model(format!("marlin_gemm (perm): {e}")))
        })
    } else {
        if use_vllm {
            return launch_vllm_marlin(&ctx.stream, a, weight, out, m);
        }
        crate::marlin::marlin_gemm(&ctx.stream, a, weight, out, m as i32)
            .map_err(|e| FerrumError::model(format!("marlin_gemm: {e}")))
    }
}

#[cfg(feature = "vllm-marlin")]
pub fn launch_vllm_marlin(
    stream: &Arc<cudarc::driver::CudaStream>,
    a: &CudaSlice<f16>,
    weight: &crate::marlin::MarlinWeight,
    out: &mut CudaSlice<f16>,
    m: usize,
) -> Result<()> {
    use cudarc::driver::DevicePtr;
    use std::sync::atomic::{AtomicU64, Ordering};
    static VLLM_MARLIN_CALLS: AtomicU64 = AtomicU64::new(0);
    let n = VLLM_MARLIN_CALLS.fetch_add(1, Ordering::Relaxed);
    if n == 0 || n.is_multiple_of(1024) {
        eprintln!(
            "[vllm-marlin] launch #{n} m={m} n={} k={} group_size={}",
            weight.n, weight.k, weight.group_size,
        );
    }
    // Zero workspace (it's used as mutex locks in vLLM's marlin too).
    {
        let (ws_ptr, _g) = weight.workspace.device_ptr(stream);
        let raw_stream = stream.cu_stream();
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(ws_ptr, 0, weight.workspace.len(), raw_stream);
        }
    }
    let (a_ptr, _g_a) = a.device_ptr(stream);
    let (b_ptr, _g_b) = weight.qweight.device_ptr(stream);
    let (c_ptr, _g_c) = out.device_ptr(stream);
    let (s_ptr, _g_s) = weight.scales.device_ptr(stream);
    let (ws_ptr, _g_w) = weight.workspace.device_ptr(stream);
    let raw_stream = stream.cu_stream();
    let n = weight.n as i32;
    let k = weight.k as i32;
    let group_size = weight.group_size;
    let num_groups = if group_size > 0 { k / group_size } else { 1 };
    // RTX 4090 = 128 SMs. TODO: query CudaDevice attribute.
    let sms = std::env::var("FERRUM_VLLM_MARLIN_SMS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(128);
    // vLLM perf knobs — try toggling via env to see if either helps.
    let use_atomic_add = std::env::var("FERRUM_VLLM_ATOMIC_ADD").map_or(false, |v| v == "1");
    let use_fp32_reduce = std::env::var("FERRUM_VLLM_FP32_REDUCE").map_or(false, |v| v == "1");

    unsafe {
        crate::vllm_marlin::launch_marlin_mm_f16_u4b8(
            a_ptr as *const _,
            b_ptr as *const _,
            c_ptr as *mut _,
            std::ptr::null_mut(), // C_tmp (use_fp32_reduce=false)
            std::ptr::null_mut(), // a_s   (FP16 act, no per-token scale)
            s_ptr as *mut _,      // b_s
            std::ptr::null_mut(), // g_idx (we already gathered A by perm)
            std::ptr::null_mut(), // perm  (ditto)
            std::ptr::null_mut(), // a_tmp
            m as i32,
            n,
            k,
            k, // lda = K (row-major FP16 A)
            ws_ptr as *mut _,
            false, // has_act_order — pre-applied via perm-gather
            true,  // is_k_full
            num_groups,
            group_size,
            0, // dev
            raw_stream as cudarc::driver::sys::CUstream,
            sms,
            use_atomic_add,
            use_fp32_reduce,
        );
    }
    Ok(())
}

#[cfg(not(feature = "vllm-marlin"))]
fn launch_vllm_marlin(
    _stream: &Arc<cudarc::driver::CudaStream>,
    _a: &CudaSlice<f16>,
    _weight: &crate::marlin::MarlinWeight,
    _out: &mut CudaSlice<f16>,
    _m: usize,
) -> Result<()> {
    Err(FerrumError::model(
        "FERRUM_VLLM_MARLIN=1 set but binary not built with --features vllm-marlin",
    ))
}

fn ensure_module(ctx: &Arc<CudaContext>, key: &'static str, ptx_src: &str) -> Arc<CudaModule> {
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

impl BackendQuantMarlin for CudaBackend {
    fn pregrow_marlin_gather_scratch(ctx: &mut Self::Context, required: usize) {
        #[cfg(feature = "marlin")]
        {
            let stream = ctx.stream.clone();
            pregrow_marlin_gather_scratch(&stream, required);
        }
        #[cfg(not(feature = "marlin"))]
        {
            let _ = (ctx, required);
        }
    }
    #[cfg(feature = "marlin")]
    fn gemm_gptq_with_offset_strided(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        in_row_offset: usize,
        weight: &Self::GptqStore,
        expert_offset: usize,
        expert_n: usize,
        output: &mut Self::Buffer,
        out_row_offset: usize,
        m: usize,
        _k: usize,
    ) -> Result<()> {
        #[cfg(feature = "triton-kernels")]
        let mw = match weight {
            GptqStoreCuda::Marlin(mw) => mw,
            GptqStoreCuda::Triton(_) => {
                return Err(FerrumError::unsupported(
                    "gemm_gptq_with_offset_strided: Triton w4a16 store has no \
                     stride-aware variant; load MoE via Marlin (default)",
                ));
            }
        };
        #[cfg(not(feature = "triton-kernels"))]
        let mw: &crate::marlin::MarlinWeight = weight;
        let stream = ctx.stream.clone();
        crate::marlin::marlin_gemm_with_offset_strided(
            &stream,
            input,
            in_row_offset as i32,
            mw,
            output,
            out_row_offset as i32,
            m as i32,
            expert_offset as i32,
            expert_n as i32,
        )
        .map_err(|e| FerrumError::model(format!("marlin offset_strided gemm: {e}")))
    }
    fn load_gptq(
        qweight: &[i32],
        scales: &[f32],
        qzeros: &[i32],
        g_idx: Option<&[i32]>,
        bias_host: Option<&[f32]>,
        bits: u32,
        group_size: usize,
        k: usize,
        n: usize,
    ) -> Result<Box<dyn crate::Linear<Self> + Send + Sync>> {
        if bits != 4 {
            return Err(FerrumError::unsupported(format!(
                "CUDA GPTQ: only bits=4 supported (got {bits})"
            )));
        }
        let _ = qzeros; // qzeros baked into Marlin scales path; unused for Marlin store

        // Path B: triton-rs w4a16 GPTQ kernel — operates on the on-disk
        // GPTQ tensor layout directly. Just upload the three tensors and
        // tag the store as Triton. No CPU-side repack.
        #[cfg(feature = "triton-kernels")]
        if use_triton_int4() {
            let stream = default_stream();
            let scales_f16: Vec<f16> = scales.iter().map(|&x| f16::from_f32(x)).collect();
            let qweight_dev = stream
                .clone_htod(qweight)
                .map_err(|e| FerrumError::model(format!("triton qweight htod: {e}")))?;
            let scales_dev = stream
                .clone_htod(&scales_f16)
                .map_err(|e| FerrumError::model(format!("triton scales htod: {e}")))?;
            let qzeros_dev = stream
                .clone_htod(qzeros)
                .map_err(|e| FerrumError::model(format!("triton qzeros htod: {e}")))?;
            tracing::info!("GPTQ load (triton-rs w4a16): K={k}, N={n}, gs={group_size}");
            let store = GptqStoreCuda::Triton(crate::triton_w4a16::TritonGptqWeight {
                qweight: qweight_dev,
                scales: scales_dev,
                qzeros: qzeros_dev,
                k,
                n,
                group_size: group_size as i32,
            });
            let bias = bias_host.map(<Self as crate::backend::Backend>::from_slice);
            return Ok(Box::new(
                crate::quant_linear::cuda_marlin::CudaMarlinLinear {
                    store,
                    bias,
                    in_features: k,
                    out_features: n,
                },
            ));
        }

        // Detect desc_act=true (act-order GPTQ): g_idx is non-trivial.
        // AutoGPTQ writes g_idx[k] = k/group_size for desc_act=false; any
        // deviation = act-order. Build perm = argsort(g_idx) and permute
        // qweight rows so that group_idx = i / group_size after the perm
        // (matches Marlin's sequential group access). Standard scales/zeros
        // layout works because they're already indexed by group, not by k.
        let (qweight_for_repack, perm_dev_opt): (Vec<i32>, Option<CudaSlice<i32>>) =
            if let Some(gx) = g_idx {
                let is_desc_act = gx
                    .iter()
                    .enumerate()
                    .any(|(i, &g)| g != (i as i32) / group_size as i32);
                if is_desc_act {
                    // perm[i] = disk row whose g_idx is the i-th smallest.
                    let mut perm: Vec<usize> = (0..k).collect();
                    perm.sort_by_key(|&i| gx[i]);
                    let permuted_qweight =
                        crate::marlin::permute_gptq_qweight_rows(qweight, &perm, k, n);
                    let perm_i32: Vec<i32> = perm.iter().map(|&p| p as i32).collect();
                    let stream = default_stream();
                    let perm_dev = stream
                        .clone_htod(&perm_i32)
                        .map_err(|e| FerrumError::model(format!("perm htod: {e}")))?;
                    tracing::info!(
                        "GPTQ load (Marlin + desc_act perm-aware): K={k} N={n} gs={group_size}"
                    );
                    (permuted_qweight, Some(perm_dev))
                } else {
                    (qweight.to_vec(), None)
                }
            } else {
                (qweight.to_vec(), None)
            };

        // Path A (default): Marlin. Repack on CPU, then upload. Matches
        // IST-DASLab/marlin Layer.pack().
        let marlin_qweight_i32 = crate::marlin::repack_gptq_to_marlin(&qweight_for_repack, k, n);
        // Scales arrive as f32 but Marlin expects f16. Convert + permute.
        let scales_f16: Vec<f16> = scales.iter().map(|&x| f16::from_f32(x)).collect();
        let marlin_scales_f16 =
            crate::marlin::repack_scales_to_marlin(&scales_f16, k, n, group_size);

        // Upload on the global stream (same one inference uses).
        let stream = default_stream();
        let qweight_dev = stream
            .clone_htod(&marlin_qweight_i32)
            .map_err(|e| FerrumError::model(format!("qweight htod: {e}")))?;
        let scales_dev = stream
            .clone_htod(&marlin_scales_f16)
            .map_err(|e| FerrumError::model(format!("scales htod: {e}")))?;

        // Workspace: Marlin uses int32 atomic locks, [N/128 * max_par] zeroed.
        // max_par hardcoded to 16 per kernel launch; allocate + zero here.
        let max_par = 16usize;
        let ws_len = (n / 128).max(1) * max_par;
        let workspace_dev = stream
            .alloc_zeros::<i32>(ws_len)
            .map_err(|e| FerrumError::model(format!("ws alloc: {e}")))?;

        let marlin_weight = crate::marlin::MarlinWeight {
            qweight: qweight_dev,
            scales: scales_dev,
            workspace: workspace_dev,
            k,
            n,
            group_size: group_size as i32,
            perm: perm_dev_opt,
        };

        // Wrap in the Marlin variant (or pass through, depending on cfg)
        // and box as a Linear<Self>. Phase 3e/2: kernel dispatch lives
        // inside CudaMarlinLinear::forward, not the trait method.
        #[cfg(feature = "triton-kernels")]
        let store = GptqStoreCuda::Marlin(marlin_weight);
        #[cfg(not(feature = "triton-kernels"))]
        let store: GptqStoreCuda = marlin_weight;

        let bias = bias_host.map(<Self as crate::backend::Backend>::from_slice);
        Ok(Box::new(
            crate::quant_linear::cuda_marlin::CudaMarlinLinear {
                store,
                bias,
                in_features: k,
                out_features: n,
            },
        ))
    }
    fn load_gptq_stacked(
        qweights: &[&[i32]],
        scales: &[&[f32]],
        qzeros: &[&[i32]],
        g_idx: Option<&[i32]>,
        bits: u32,
        group_size: usize,
        k: usize,
        n_per_expert: usize,
    ) -> Result<Self::GptqStore> {
        if bits != 4 {
            return Err(FerrumError::unsupported(format!(
                "CUDA GPTQ stacked: only bits=4 supported (got {bits})"
            )));
        }
        let num_experts = qweights.len();
        if num_experts == 0 {
            return Err(FerrumError::model("load_gptq_stacked: 0 experts"));
        }
        if scales.len() != num_experts || qzeros.len() != num_experts {
            return Err(FerrumError::model(format!(
                "load_gptq_stacked length mismatch: qw={} sc={} qz={}",
                num_experts,
                scales.len(),
                qzeros.len()
            )));
        }
        let _ = qzeros; // Marlin doesn't read qzeros (sym=true)

        // vLLM marlin_moe_wna16 path: stacked weight in vLLM Marlin tile
        // format (NOT IST-DASLab). Run gptq_marlin_repack per expert,
        // permute scales with the same _scale_perm IST-DASLab uses.
        // Opt-in via FERRUM_VLLM_MOE=1 — paired with the dispatch-side
        // switch in moe_forward_bucketed.
        #[cfg(feature = "vllm-moe-marlin")]
        if use_vllm_moe() {
            let stream = default_stream();
            let store = crate::vllm_marlin::load_stacked_gptq_vllm_marlin(
                &stream,
                qweights,
                scales,
                bits,
                group_size,
                k,
                n_per_expert,
            )
            .map_err(|e| FerrumError::model(format!("load_stacked_gptq_vllm_marlin: {e}")))?;
            tracing::info!(
                "GPTQ stacked load (vLLM marlin path): {num_experts} experts × N={n_per_expert} × K={k} (gs={group_size})",
            );
            return Ok(store);
        }

        // Triton path: would need a stacked variant — not implemented.
        // Fall through to Marlin.
        #[cfg(feature = "triton-kernels")]
        if use_triton_int4() {
            return Err(FerrumError::unsupported(
                "load_gptq_stacked: Triton w4a16 path not implemented; \
                 unset FERRUM_TRITON_INT4 to use Marlin",
            ));
        }

        // desc_act perm-aware path: sample expert 0's g_idx (all experts
        // share K-axis quantization, so g_idx is identical across experts).
        let (perm_dev_opt, perm_for_repack): (Option<CudaSlice<i32>>, Option<Vec<usize>>) =
            if let Some(gx) = g_idx {
                let is_desc_act = gx
                    .iter()
                    .enumerate()
                    .any(|(i, &g)| g != (i as i32) / group_size as i32);
                if is_desc_act {
                    let mut perm: Vec<usize> = (0..k).collect();
                    perm.sort_by_key(|&i| gx[i]);
                    let perm_i32: Vec<i32> = perm.iter().map(|&p| p as i32).collect();
                    let stream = default_stream();
                    let perm_dev = stream
                        .clone_htod(&perm_i32)
                        .map_err(|e| FerrumError::model(format!("perm htod: {e}")))?;
                    (Some(perm_dev), Some(perm))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };

        // Per-expert repack in parallel via rayon. Each produces its
        // own packed qweight + permuted scales tile. Concat in expert
        // order — each expert's bytes are CONTIGUOUS in the output
        // buffer, so an offset GEMM with prob_n=n_per_expert just
        // walks the right slice.
        use rayon::prelude::*;
        let qw_per_expert_i32 = (n_per_expert * k) / 8;
        let sc_per_expert_f16 = (k / group_size) * n_per_expert;

        let mut packed_qw: Vec<i32> = vec![0i32; num_experts * qw_per_expert_i32];
        let mut packed_sc: Vec<f16> = vec![f16::ZERO; num_experts * sc_per_expert_f16];

        // Parallelize across experts: each writes to a disjoint output
        // slice. Per-expert repack runs the inner-rayon-parallel
        // 4-pass kernel — at num_experts > num_cores, outer parallelism
        // wins; at smaller num_experts, inner parallelism wins. Rayon
        // nests fine.
        packed_qw
            .par_chunks_mut(qw_per_expert_i32)
            .zip(packed_sc.par_chunks_mut(sc_per_expert_f16))
            .enumerate()
            .for_each(|(e, (qw_out, sc_out))| {
                let qw_in: Vec<i32> = if let Some(perm) = &perm_for_repack {
                    crate::marlin::permute_gptq_qweight_rows(qweights[e], perm, k, n_per_expert)
                } else {
                    qweights[e].to_vec()
                };
                let qw_packed = crate::marlin::repack_gptq_to_marlin(&qw_in, k, n_per_expert);
                qw_out.copy_from_slice(&qw_packed);

                let sc_f16: Vec<f16> = scales[e].iter().map(|&x| f16::from_f32(x)).collect();
                let sc_packed =
                    crate::marlin::repack_scales_to_marlin(&sc_f16, k, n_per_expert, group_size);
                sc_out.copy_from_slice(&sc_packed);
            });

        // Single htod upload of the concatenated packed buffer.
        let stream = default_stream();
        let qweight_dev = stream
            .clone_htod(&packed_qw)
            .map_err(|e| FerrumError::model(format!("stacked qweight htod: {e}")))?;
        let scales_dev = stream
            .clone_htod(&packed_sc)
            .map_err(|e| FerrumError::model(format!("stacked scales htod: {e}")))?;

        // Workspace: per-expert range concatenated. Each expert owns
        // (n_per_expert/128) × max_par i32 mutex slots starting at
        // its expert_idx * that_size.
        let max_par = 16usize;
        let ws_per_expert = (n_per_expert / 128).max(1) * max_par;
        let ws_len = num_experts * ws_per_expert;
        let workspace_dev = stream
            .alloc_zeros::<i32>(ws_len)
            .map_err(|e| FerrumError::model(format!("stacked ws alloc: {e}")))?;

        // Total stacked N for downstream offset arithmetic. We store
        // total_n in MarlinWeight.n so byte_per_col-style helpers see
        // the full width; the offset GEMM uses per-expert stride
        // (= n_per_expert) regardless of weight.n.
        let total_n = num_experts * n_per_expert;
        let marlin_weight = crate::marlin::MarlinWeight {
            qweight: qweight_dev,
            scales: scales_dev,
            workspace: workspace_dev,
            k,
            n: total_n,
            group_size: group_size as i32,
            perm: perm_dev_opt,
        };

        tracing::info!(
            "GPTQ stacked load: {} experts × N={n_per_expert} × K={k} (gs={group_size})",
            num_experts
        );

        #[cfg(feature = "triton-kernels")]
        {
            Ok(GptqStoreCuda::Marlin(marlin_weight))
        }
        #[cfg(not(feature = "triton-kernels"))]
        {
            Ok(marlin_weight)
        }
    }
    fn make_stacked_expert_linear(
        store: std::sync::Arc<Self::GptqStore>,
        expert_offset: usize,
        expert_n: usize,
        k: usize,
        bias_host: Option<&[f32]>,
    ) -> Result<Box<dyn crate::Linear<Self> + Send + Sync>> {
        let bias = bias_host.map(<Self as crate::backend::Backend>::from_slice);
        Ok(Box::new(
            crate::quant_linear::cuda_marlin::CudaMarlinStackedExpertLinear {
                store,
                expert_offset,
                expert_n,
                k,
                bias,
            },
        ))
    }
    #[cfg(feature = "marlin")]
    fn moe_gemm_phase_batched(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        weight: &Self::GptqStore,
        dispatches: &[(usize, usize, usize, usize)],
        n_per_expert: usize,
        output: &mut Self::Buffer,
        k: usize,
    ) -> Result<()> {
        #[cfg(feature = "triton-kernels")]
        let mw = match weight {
            GptqStoreCuda::Marlin(mw) => mw,
            GptqStoreCuda::Triton(_) => {
                return Err(FerrumError::unsupported(
                    "moe_gemm_phase_batched: Triton w4a16 not supported",
                ));
            }
        };
        #[cfg(not(feature = "triton-kernels"))]
        let mw: &crate::marlin::MarlinWeight = weight;

        // ── Stage 12.1: fused MoE Marlin path (default ON) ──────────────
        // Dispatches all experts in this phase as a small number of
        // bucketed `marlin_gemm_moe` launches (one per thread_m_blocks
        // bucket ∈ {1, 2, 3, 4}) instead of N round-robin calls. At c=32
        // with ~100 active experts/layer this is 1-4 launches/layer
        // instead of 100. Bench: +25% c=32, +35% c=16, +48% c=8 over
        // the multi-stream per-expert path. Set FERRUM_MOE_FUSED=0 to
        // opt out (fall back to multi-stream pool).
        if std::env::var("FERRUM_MOE_FUSED").map_or(true, |v| v != "0") {
            return moe_gemm_phase_fused_impl(ctx, input, mw, dispatches, n_per_expert, output, k);
        }

        // n_streams=1: serial dispatch on the DEFAULT context stream
        // (avoids the cross-stream sync overhead and matches the
        // pre-multi-stream path bit-for-bit).
        let n_streams = std::env::var("FERRUM_MOE_STREAMS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        if n_streams == 1 {
            let default_stream = ctx.stream.clone();
            for (expert_idx, in_row_offset, out_row_offset, m) in dispatches {
                crate::marlin::marlin_gemm_with_offset_strided(
                    &default_stream,
                    input,
                    *in_row_offset as i32,
                    mw,
                    output,
                    *out_row_offset as i32,
                    *m as i32,
                    (expert_idx * n_per_expert) as i32,
                    n_per_expert as i32,
                )
                .map_err(|e| FerrumError::model(format!("marlin offset_strided: {e}")))?;
            }
            let _ = k;
            return Ok(());
        }

        // n_streams ≥ 2: round-robin across the pool, then ALL streams
        // wait for the default's prior work (cuStreamWaitEvent on an
        // event recorded into default), and the default waits for ALL
        // pool streams before returning. Without this cross-stream
        // sync, silu_mul on default may run before pool GEMMs commit
        // → races → junk output.
        //
        // The 1 + N events are persistent on `CudaState` — re-recording
        // an event silently overwrites the prior recording, so per-call
        // create+destroy is replaced with record+wait only. At c=32 /
        // 48 layers / 2 phases that's ~960 driver calls saved per token.
        let (entry_event, exit_events) = ctx.moe_sync_events();
        let pool: Vec<Arc<CudaStream>> = ctx.moe_stream_pool().to_vec();
        let default_stream = ctx.stream.clone();
        use cudarc::driver::sys as cu;
        // Default → pool: record on default, each pool waits.
        unsafe {
            cu::cuEventRecord(entry_event, default_stream.cu_stream());
        }
        for stream in &pool {
            unsafe {
                cu::cuStreamWaitEvent(stream.cu_stream(), entry_event, 0);
            }
        }

        for (i, (expert_idx, in_row_offset, out_row_offset, m)) in dispatches.iter().enumerate() {
            let stream = &pool[i % n_streams];
            crate::marlin::marlin_gemm_with_offset_strided(
                stream,
                input,
                *in_row_offset as i32,
                mw,
                output,
                *out_row_offset as i32,
                *m as i32,
                (expert_idx * n_per_expert) as i32,
                n_per_expert as i32,
            )
            .map_err(|e| FerrumError::model(format!("marlin offset_strided: {e}")))?;
        }

        // pool → default: each pool stream records its exit event;
        // default waits for all of them. After return, default's next
        // op (silu) will see all GEMM outputs committed.
        let _ = k;
        debug_assert_eq!(
            exit_events.len(),
            pool.len(),
            "moe_sync_events exit count != pool size"
        );
        for (i, stream) in pool.iter().enumerate() {
            unsafe {
                cu::cuEventRecord(exit_events[i], stream.cu_stream());
            }
        }
        for ev in &exit_events {
            unsafe {
                cu::cuStreamWaitEvent(default_stream.cu_stream(), *ev, 0);
            }
        }
        Ok(())
    }
    #[cfg(feature = "vllm-moe-marlin")]
    fn moe_gemm_phase_vllm(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        weight: &Self::GptqStore,
        sorted_token_ids: &Self::Buffer,
        expert_ids: &Self::Buffer,
        num_tokens_past_padded: &Self::Buffer,
        output: &mut Self::Buffer,
        prob_m: usize,
        n_per_expert: usize,
        k: usize,
        moe_block_size: usize,
        top_k: usize,
    ) -> Result<()> {
        #[cfg(feature = "triton-kernels")]
        let mw = match weight {
            GptqStoreCuda::Marlin(mw) => mw,
            GptqStoreCuda::Triton(_) => {
                return Err(FerrumError::unsupported(
                    "moe_gemm_phase_vllm: Triton store unsupported",
                ));
            }
        };
        #[cfg(not(feature = "triton-kernels"))]
        let mw: &crate::marlin::MarlinWeight = weight;

        // sorted_token_ids / expert_ids / num_tokens_past_padded are i32
        // device buffers handed in as Self::Buffer (= CudaSlice<f16>).
        // The marlin_gemm_moe_vllm wrapper uses CudaSlice<i32>, so we
        // reinterpret each view via upgrade_device_ptr. The view length
        // is irrelevant — the kernel reads via raw device pointers — so
        // we hand it `0` to make the leakage cost explicit.
        use cudarc::driver::sys::CUdeviceptr;
        use cudarc::driver::CudaSlice;
        use cudarc::driver::DevicePtr;

        // Stream is Arc<CudaStream>, so cloning it doesn't borrow ctx and
        // we can subsequently take &mut for the c_tmp helper.
        let stream = ctx.stream.clone();
        // Resolve c_tmp scratch (lazy-allocates 8 MB on first call). The
        // wrapper takes &mut so it can forward the device pointer; the
        // kernel uses c_tmp as a global fp32 reduce scratch
        // (use_fp32_reduce=1 path, ~1.3-1.5× faster than atomic_add).
        let c_tmp_mut: &mut CudaSlice<f32> = ctx.vllm_moe_c_tmp();

        let (st_ptr, _g0) = sorted_token_ids.device_ptr(&stream);
        let (eid_ptr, _g1) = expert_ids.device_ptr(&stream);
        let (npp_ptr, _g2) = num_tokens_past_padded.device_ptr(&stream);
        let st_view: CudaSlice<i32> =
            unsafe { stream.upgrade_device_ptr(st_ptr as CUdeviceptr, 0) };
        let eid_view: CudaSlice<i32> =
            unsafe { stream.upgrade_device_ptr(eid_ptr as CUdeviceptr, 0) };
        let npp_view: CudaSlice<i32> =
            unsafe { stream.upgrade_device_ptr(npp_ptr as CUdeviceptr, 0) };

        let r = crate::marlin::marlin_gemm_moe_vllm(
            &stream,
            input,
            mw,
            output,
            Some(c_tmp_mut), // fp32_reduce path (atomic_add fallback if None)
            &st_view,
            &eid_view,
            &npp_view,
            None, // topk_weights
            moe_block_size as i32,
            top_k as i32,
            false, // mul_topk_weights
            false, // is_ep
            prob_m as i32,
            n_per_expert as i32,
            k as i32,
        );
        // Views borrow the original device allocations; forgetting prevents
        // double-free at scope end.
        std::mem::forget(st_view);
        std::mem::forget(eid_view);
        std::mem::forget(npp_view);
        r.map_err(|e| FerrumError::model(format!("marlin_gemm_moe_vllm: {e}")))
    }
    #[cfg(feature = "marlin")]
    fn marlin_zero_stacked_workspace(
        ctx: &mut Self::Context,
        weight: &Self::GptqStore,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;
        #[cfg(feature = "triton-kernels")]
        let mw = match weight {
            GptqStoreCuda::Marlin(mw) => mw,
            GptqStoreCuda::Triton(_) => {
                return Err(FerrumError::unsupported(
                    "marlin_zero_stacked_workspace: not applicable to Triton store",
                ));
            }
        };
        #[cfg(not(feature = "triton-kernels"))]
        let mw: &crate::marlin::MarlinWeight = weight;
        let stream = ctx.stream.clone();
        let raw_stream = stream.cu_stream();
        let (ws_ptr, _g) = mw.workspace.device_ptr(&stream);
        let ws_len = mw.workspace.len();
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(ws_ptr, 0, ws_len, raw_stream);
        }
        Ok(())
    }
}

// CUDA does not ship GGUF k-quant kernels; inherit unsupported defaults.
impl BackendQuantGguf for CudaBackend {}

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

// CUDA: INT8 KV cache (vLLM-style scale-per-token symmetric quantization).
// Kernel-side dispatch is exposed via [`crate::int8_kv::launch_int8_paged_decode_attention`]
// and [`crate::int8_kv::launch_int8_kv_cache_append`]. See
// `tests/int8_kv_parity.rs` for a host-reference parity check
// (cos sim ≈ 0.99999 vs FP32 ref). With the associated types declared
// here, `KvCache<CudaBackend, KvInt8>` carries `CudaSlice<i8>` for K/V
// and `CudaSlice<f16>` for scales — distinct types from the FP16 path.
//
// Note: `KvScales = OptionalCudaScalesF16` rather than a bare
// `CudaSlice<f16>` so the `Default` bound on `KvScales` can be
// satisfied without holding a CUDA stream at struct-default time.
impl crate::backend::BackendKvDtype<crate::backend::KvInt8> for CudaBackend {
    type KvBuffer = OptionalCudaInt8;
    type KvScales = OptionalCudaScalesF16;
}

/// Lazily-allocated INT8 KV buffer. `Default` produces an empty
/// placeholder; the real allocation happens via the `init` method
/// once a CUDA stream is in scope.
#[derive(Default)]
pub struct OptionalCudaInt8(pub Option<cudarc::driver::CudaSlice<i8>>);

impl OptionalCudaInt8 {
    /// Allocate `len` zeroed `int8_t` elements on the default CUDA stream.
    pub fn alloc(len: usize) -> Self {
        let stream = default_stream();
        let buf = stream
            .alloc_zeros::<i8>(len)
            .expect("alloc int8 KV buffer");
        Self(Some(buf))
    }

    pub fn buffer(&self) -> &cudarc::driver::CudaSlice<i8> {
        self.0.as_ref().expect("OptionalCudaInt8 not allocated")
    }

    pub fn buffer_mut(&mut self) -> &mut cudarc::driver::CudaSlice<i8> {
        self.0.as_mut().expect("OptionalCudaInt8 not allocated")
    }
}

/// Lazily-allocated INT8 scales buffer (FP16 storage on CUDA).
#[derive(Default)]
pub struct OptionalCudaScalesF16(pub Option<cudarc::driver::CudaSlice<half::f16>>);

impl OptionalCudaScalesF16 {
    /// Allocate `len` zeroed FP16 scales on the default CUDA stream.
    pub fn alloc(len: usize) -> Self {
        let stream = default_stream();
        let buf = stream
            .alloc_zeros::<half::f16>(len)
            .expect("alloc int8 KV scales");
        Self(Some(buf))
    }

    pub fn buffer(&self) -> &cudarc::driver::CudaSlice<half::f16> {
        self.0.as_ref().expect("OptionalCudaScalesF16 not allocated")
    }

    pub fn buffer_mut(&mut self) -> &mut cudarc::driver::CudaSlice<half::f16> {
        self.0.as_mut().expect("OptionalCudaScalesF16 not allocated")
    }
}

// Implement INT8 KV launchers as Backend trait methods so the model
// layer can dispatch via `B::int8_kv_append_paged(...)` /
// `B::int8_paged_decode_attention(...)` without reaching into
// cudarc primitives directly.
impl crate::backend::BackendInt8KvOps for CudaBackend {
    fn alloc_paged_int8_layer(
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> crate::backend::KvCacheQuant<Self, crate::backend::KvInt8> {
        crate::backend::KvCacheQuant::<CudaBackend, crate::backend::KvInt8>::new_paged_cuda(
            max_blocks_per_seq,
            block_size,
            num_kv_heads,
            head_dim,
        )
    }

    fn int8_kv_append_paged(
        ctx: &mut Self::Context,
        k_in: &Self::Buffer,
        v_in: &Self::Buffer,
        layer_k: &mut <Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvBuffer,
        layer_v: &mut <Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvBuffer,
        layer_k_scales: &mut <Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvScales,
        layer_v_scales: &mut <Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvScales,
        block_table: &Self::Buffer,
        cache_len_before: usize,
        tokens: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        if tokens == 0 {
            return Ok(());
        }
        // Read block_table host-side to compute the per-token slot mapping
        // expected by `launch_int8_kv_cache_append`. block_table is stored
        // as `CudaSlice<f16>` (Self::Buffer) but actually holds u32 indices
        // — alloc_u32 over-allocates exactly so the byte layout matches.
        let last_block = (cache_len_before + tokens - 1) / block_size;
        let n_blocks_to_read = last_block + 1;
        let stream = ctx.stream.clone();
        let bt_u32_view = unsafe {
            block_table
                .transmute::<u32>(n_blocks_to_read)
                .ok_or_else(|| FerrumError::model("block_table transmute<u32> failed"))?
        };
        let mut block_indices = vec![0u32; n_blocks_to_read];
        stream
            .memcpy_dtoh(&bt_u32_view, &mut block_indices)
            .map_err(|e| FerrumError::model(format!("dtoh block_table: {e}")))?;
        stream
            .synchronize()
            .map_err(|e| FerrumError::model(format!("sync after block_table dtoh: {e}")))?;

        // Compute flat slot indices: physical_block * block_size + slot.
        let mut slot_mapping_host = vec![0i32; tokens];
        for t in 0..tokens {
            let global_pos = cache_len_before + t;
            let block_logical = global_pos / block_size;
            let slot_in_block = global_pos % block_size;
            let block_physical = block_indices[block_logical] as usize;
            slot_mapping_host[t] = (block_physical * block_size + slot_in_block) as i32;
        }
        let slot_mapping = stream
            .memcpy_stod(&slot_mapping_host)
            .map_err(|e| FerrumError::model(format!("htod slot_mapping: {e}")))?;

        // Lazily alloc INT8 buffers + scales on first call (the constructor
        // populates them already, but defensive in case callers clear).
        if layer_k.0.is_none() {
            return Err(FerrumError::model(
                "int8_kv_append_paged: layer_k not allocated",
            ));
        }
        if layer_v.0.is_none() || layer_k_scales.0.is_none() || layer_v_scales.0.is_none() {
            return Err(FerrumError::model(
                "int8_kv_append_paged: layer_v / scales not allocated",
            ));
        }

        crate::int8_kv::launch_int8_kv_cache_append(
            &ctx.ctx,
            k_in,
            v_in,
            layer_k.buffer_mut(),
            layer_v.buffer_mut(),
            layer_k_scales.buffer_mut(),
            layer_v_scales.buffer_mut(),
            &slot_mapping,
            tokens,
            num_kv_heads,
            head_dim,
        )
        .map_err(|e| FerrumError::model(format!("launch_int8_kv_cache_append: {e}")))?;
        Ok(())
    }

    fn int8_paged_decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        layer_k: &<Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvBuffer,
        layer_v: &<Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvBuffer,
        layer_k_scales: &<Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvScales,
        layer_v_scales: &<Self as crate::backend::BackendKvDtype<crate::backend::KvInt8>>::KvScales,
        block_table: &Self::Buffer,
        output: &mut Self::Buffer,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        valid_kv_len: usize,
        block_size: usize,
        scale: f32,
    ) -> Result<()> {
        // block_table is stored as f16 but holds i32 (alloc_u32 doubles
        // bytes). Reinterpret to i32 view of length max_blocks_per_seq.
        let n_blocks = valid_kv_len.div_ceil(block_size).max(1);
        let bt_i32_view = unsafe {
            block_table
                .transmute::<i32>(n_blocks)
                .ok_or_else(|| FerrumError::model("block_table transmute<i32> failed"))?
        };
        crate::int8_kv::launch_int8_paged_decode_attention(
            &ctx.ctx,
            q,
            layer_k.buffer(),
            layer_v.buffer(),
            layer_k_scales.buffer(),
            layer_v_scales.buffer(),
            &bt_i32_view,
            output,
            num_q_heads,
            num_kv_heads,
            head_dim,
            valid_kv_len,
            block_size,
            scale,
        )
        .map_err(|e| FerrumError::model(format!("launch_int8_paged_decode_attention: {e}")))?;
        Ok(())
    }
}

// Convenience constructor for paged INT8 KV caches on CUDA.
impl crate::backend::KvCacheQuant<CudaBackend, crate::backend::KvInt8> {
    /// Allocate a paged INT8 KV cache for one sequence.
    ///
    /// - `max_blocks_per_seq` × `block_size` = capacity in tokens
    /// - K/V pool size: `max_blocks_per_seq * block_size * num_kv_heads * head_dim` int8 elems
    /// - scales pool size: `max_blocks_per_seq * block_size * num_kv_heads` FP16 elems
    /// - `block_table` is allocated as u32[max_blocks_per_seq] via `B::alloc_u32`
    /// - `context_lens` is allocated as u32[1] (single seq for now)
    pub fn new_paged_cuda(
        max_blocks_per_seq: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        use crate::backend::Backend;
        let pool_tokens = max_blocks_per_seq * block_size;
        let elem_count = pool_tokens * num_kv_heads * head_dim;
        let scale_count = pool_tokens * num_kv_heads;
        let block_table = <CudaBackend as Backend>::alloc_u32(max_blocks_per_seq);
        let mut context_lens = <CudaBackend as Backend>::alloc_u32(1);
        let mut bt_ctx = <CudaBackend as Backend>::new_context();
        <CudaBackend as Backend>::write_u32(&mut bt_ctx, &mut context_lens, &[0u32]);
        <CudaBackend as Backend>::sync(&mut bt_ctx);

        // Re-cast typed u32 buffer to the trait's Buffer (FP16) — same
        // pattern the FP16 paged path uses for block_table/context_lens
        // (they are u32 device tensors written through alloc_u32).
        let bt_buf = block_table;
        let cl_buf = context_lens;

        crate::backend::KvCacheQuant {
            k: OptionalCudaInt8::alloc(elem_count),
            v: OptionalCudaInt8::alloc(elem_count),
            k_scales: OptionalCudaScalesF16::alloc(scale_count),
            v_scales: OptionalCudaScalesF16::alloc(scale_count),
            len: 0,
            capacity: pool_tokens,
            num_kv_heads,
            head_dim,
            block_size,
            block_table: Some(bt_buf),
            context_lens: Some(cl_buf),
            paged_block_indices: Vec::new(),
            _kv_dtype: std::marker::PhantomData,
        }
    }
}
