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
//   Phase 4: `BackendPagedKv` (incl. `SplitKScratch` +
//     `paged_varlen_split_k_dispatch` + `paged_batched_flash_dispatch` +
//     `paged_batched_decode_single_pass`) → `cuda/paged.rs`.
//   Phase 5 (final): `BackendMoeFused` (`route_topk_softmax`,
//     `try_gpu_route_topk_into_host`, `moe_align_block_size`,
//     `moe_combine`) → `cuda/moe.rs`.
//
// After Phase 5, `cuda/mod.rs` only carries `impl Backend for
// CudaBackend` (the core trait) + `CudaState` struct + global
// stream/decode-state slots + `KvFp16` BackendKvDtype impl.
pub mod collective;
pub mod fa2_ffi;
#[cfg(feature = "fa2-source")]
pub mod fa2_source;
pub mod gated_delta_rule;
pub mod graph;
pub mod int8_kv;
pub mod linear_attention;
pub mod moe;
pub mod paged;
pub mod quant;

// Audit #9: CUDA-only kernels moved from crate-root to backend/cuda/.
// Re-exported via `pub use backend::cuda::{...}` (or the more specific
// `pub use backend::cuda::foo::Foo`) in `crate::lib` so the historical
// `ferrum_kernels::foo::*` public paths + internal `crate::foo::*`
// references keep working unchanged.
pub mod cublas;
pub mod cuda_decode;
pub mod cuda_graph;
pub mod decode_attention;
pub mod decode_buffers;
pub mod fused_add_rms_norm;
pub mod fused_silu_mul;
pub mod gpu_paged_kv;
pub mod marlin;
pub mod nccl_comm;
pub mod residual_add;
pub mod rms_norm;
pub mod rope;
pub mod tp_decode;
pub mod weight_store;

// Triton kernels (only when the `triton-kernels` feature is also on —
// `cuda` alone doesn't enable them).
#[cfg(feature = "triton-kernels")]
pub mod triton_add_bias;
#[cfg(feature = "triton-kernels")]
pub mod triton_fused_add_rms_norm;
#[cfg(feature = "triton-kernels")]
pub mod triton_fused_moe;
#[cfg(feature = "triton-kernels")]
pub mod triton_fused_silu_mul;
#[cfg(feature = "triton-kernels")]
pub mod triton_gelu;
#[cfg(feature = "triton-kernels")]
pub mod triton_layer_norm;
#[cfg(feature = "triton-kernels")]
pub mod triton_meta;
#[cfg(feature = "triton-kernels")]
pub mod triton_ptx;
#[cfg(feature = "triton-kernels")]
pub mod triton_residual_add;
#[cfg(feature = "triton-kernels")]
pub mod triton_residual_add_inplace;
#[cfg(feature = "triton-kernels")]
pub mod triton_rms_norm;
#[cfg(feature = "triton-kernels")]
pub mod triton_softmax;
#[cfg(feature = "triton-kernels")]
pub mod triton_w4a16;

// vLLM gptq_marlin port (opt-in feature, depends on `cuda`).
#[cfg(feature = "vllm-marlin")]
pub mod vllm_marlin;
// vLLM paged_attention_v2 port (opt-in, depends on `cuda`). Wraps the
// extern "C" launcher in `kernels/vllm_attn/launcher.cu`.
#[cfg(feature = "vllm-paged-attn-v2")]
pub mod vllm_paged_attn;
// Re-export so submodules (paged, etc.) can reach the constant via
// `super::MAX_LAYERS_FOR_GRAPH` like the original mod.rs code did.
pub(super) use super::MAX_LAYERS_FOR_GRAPH;
pub use int8_kv::{OptionalCudaInt8, OptionalCudaScalesF16};
// Preserve historical `crate::backend::cuda::*` paths used by external
// callers (`quant_linear::cuda_marlin`, parity tests).
#[cfg(feature = "marlin")]
pub use quant::pregrow_marlin_gather_scratch;
pub use quant::{marlin_gemm_with_perm, GptqStoreCuda};

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaBackendRuntimeEnv {
    moe_streams: usize,
    cuda_max_kv: Option<usize>,
    cuda_device: usize,
}

impl CudaBackendRuntimeEnv {
    fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        let mut moe_streams = None;
        let mut cuda_max_kv = None;
        let mut cuda_device = None;

        for (key, value) in vars {
            let value = value.into();
            match key.as_ref() {
                "FERRUM_MOE_STREAMS" => moe_streams = value.parse::<usize>().ok(),
                "FERRUM_CUDA_MAX_KV" => cuda_max_kv = value.parse::<usize>().ok(),
                "FERRUM_CUDA_DEVICE" => cuda_device = value.parse::<usize>().ok(),
                _ => {}
            }
        }

        Self {
            moe_streams: moe_streams.unwrap_or(4).max(1),
            cuda_max_kv,
            cuda_device: cuda_device.unwrap_or(0),
        }
    }
}

fn cuda_backend_runtime_env() -> &'static CudaBackendRuntimeEnv {
    static CONFIG: std::sync::OnceLock<CudaBackendRuntimeEnv> = std::sync::OnceLock::new();
    CONFIG.get_or_init(CudaBackendRuntimeEnv::from_env)
}

thread_local! {
    static CUDA_DEVICE_SCOPE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

struct CudaDeviceScopeGuard {
    previous: Option<usize>,
}

impl CudaDeviceScopeGuard {
    fn enter(ordinal: usize) -> Self {
        let previous = CUDA_DEVICE_SCOPE.with(|scope| {
            let previous = scope.get();
            scope.set(Some(ordinal));
            previous
        });
        Self { previous }
    }
}

impl Drop for CudaDeviceScopeGuard {
    fn drop(&mut self) {
        CUDA_DEVICE_SCOPE.with(|scope| scope.set(self.previous));
    }
}

pub(super) fn current_device_ordinal() -> usize {
    CUDA_DEVICE_SCOPE
        .with(|scope| scope.get())
        .unwrap_or(cuda_backend_runtime_env().cuda_device)
}

fn with_cuda_device_ordinal<R>(device_ordinal: Option<usize>, body: impl FnOnce() -> R) -> R {
    if let Some(ordinal) = device_ordinal {
        let _guard = CudaDeviceScopeGuard::enter(ordinal);
        body()
    } else {
        body()
    }
}

// ────────────────────────────────────────────────────────────────────────
// Context
// ────────────────────────────────────────────────────────────────────────

/// Execution context for CudaBackend.
///
/// Owns the `CudaContext`, a dedicated `CudaStream`, a cuBLAS handle
/// bound to that stream, and a lazy cache of PTX modules. All kernels
/// launch on `stream`; sync'ing `stream` covers all of this backend's work.
pub struct CudaState {
    pub ordinal: usize,
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
    paged_attn_out_tm: Option<crate::backend::CudaBuf>,
    paged_attn_out_tm_capacity: usize,
}

// Process-global fp32 reduce scratch for vLLM marlin_moe_wna16.
//
// Sized at the upper bound vLLM uses internally:
// `sms * 4 * moe_block_size * max_thread_n`, with the vLLM special-case
// doubling for `moe_block_size=8`. That is 4M fp32 = 16MB on a 4090.
//
// MUST be process-global (not per-CudaState) for CUDA Graph capture.
// `new_context()` builds a fresh CudaState every `decode_batch_internal`
// call; if c_tmp lived on the state it would be dropped + reallocated
// per call, but the captured graph holds the c_tmp pointer from
// capture time → next replay reads a freed/reassigned address →
// `cuGraphLaunch: CUDA_ERROR_INVALID_VALUE` on every pre-capture
// replay (the post-capture replay just happens to still see the
// original ctx's allocation). Mirrors the pattern already used by
// `MARLIN_GATHER_SCRATCH`, cuBLAS workspace, and `BATCHED_SCRATCH_*`.
#[cfg(feature = "vllm-moe-marlin")]
static VLLM_MOE_C_TMP: std::sync::OnceLock<std::sync::RwLock<HashMap<usize, CudaSlice<f32>>>> =
    std::sync::OnceLock::new();

// Greedy-argmax output scratch — see `Backend::argmax_rows_f16`.
// Process-global like VLLM_MOE_C_TMP so the GPU address baked into
// captured kernel args stays valid for the engine's life. Allocated
// to MAX_BATCH capacity on first use; the caller passes m ≤ capacity
// and reads only the first m entries.
static ARGMAX_OUT: std::sync::OnceLock<std::sync::RwLock<HashMap<usize, CudaSlice<i32>>>> =
    std::sync::OnceLock::new();

fn argmax_out_slots() -> &'static std::sync::RwLock<HashMap<usize, CudaSlice<i32>>> {
    ARGMAX_OUT.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
}

fn with_argmax_out<R>(
    stream: &Arc<CudaStream>,
    ordinal: usize,
    m: usize,
    body: impl FnOnce(&mut CudaSlice<i32>) -> R,
) -> R {
    let slots = argmax_out_slots();
    // Fast path: existing buffer large enough.
    {
        let g = slots.read().expect("ARGMAX_OUT poisoned");
        if let Some(buf) = g.get(&ordinal) {
            if buf.len() >= m {
                drop(g);
                let mut w = slots.write().expect("ARGMAX_OUT poisoned");
                return body(w.get_mut(&ordinal).expect("just observed Some"));
            }
        }
    }
    // Slow path: allocate / grow. Round up so growth amortises.
    let capacity = m.max(64).next_power_of_two();
    let mut w = slots.write().expect("ARGMAX_OUT poisoned");
    let need_alloc = w.get(&ordinal).map(|b| b.len() < m).unwrap_or(true);
    if need_alloc {
        let new = unsafe { stream.alloc::<i32>(capacity) }.expect("argmax_out alloc");
        w.insert(ordinal, new);
    }
    body(w.get_mut(&ordinal).expect("alloc above"))
}

#[cfg(feature = "vllm-moe-marlin")]
fn vllm_moe_c_tmp_slots() -> &'static std::sync::RwLock<HashMap<usize, CudaSlice<f32>>> {
    VLLM_MOE_C_TMP.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
}

/// Run `body` with a `&mut CudaSlice<f32>` pointing at the process-
/// global vLLM MoE fp32 reduce scratch. Allocates on first call only;
/// every subsequent caller sees the SAME GPU address — graph-capture
/// safe (the address baked into a captured kernel arg stays valid
/// for the lifetime of the process).
#[cfg(feature = "vllm-moe-marlin")]
pub fn with_vllm_moe_c_tmp<R>(
    stream: &Arc<CudaStream>,
    ordinal: usize,
    body: impl FnOnce(&mut CudaSlice<f32>) -> R,
) -> R {
    let slots = vllm_moe_c_tmp_slots();
    // Fast path: scratch already up. Take the write lock briefly to
    // hand a `&mut` to the body. There is at most one engine forward
    // call serialized on the model's iteration_lock at any time, so
    // contention here is zero in practice.
    {
        let g = slots.read().expect("VLLM_MOE_C_TMP poisoned");
        if g.contains_key(&ordinal) {
            drop(g);
            let mut w = slots.write().expect("VLLM_MOE_C_TMP poisoned");
            let s = w.get_mut(&ordinal).expect("just observed Some");
            return body(s);
        }
    }
    // First-time alloc.
    let mut w = slots.write().expect("VLLM_MOE_C_TMP poisoned");
    if !w.contains_key(&ordinal) {
        const C_TMP_SIZE_F32: usize = 4 * 1024 * 1024;
        let buf = stream
            .alloc_zeros::<f32>(C_TMP_SIZE_F32)
            .expect("alloc_zeros vllm_moe_c_tmp_f32 (per-device)");
        tracing::info!(
            "vLLM moe c_tmp scratch allocated (device {ordinal}): {} fp32 ({:.1} MB)",
            C_TMP_SIZE_F32,
            (C_TMP_SIZE_F32 * 4) as f32 / 1e6
        );
        w.insert(ordinal, buf);
    }
    body(w.get_mut(&ordinal).unwrap())
}

pub(super) const BATCHED_SCRATCH_CAP: usize = 64;
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
pub(super) const HOST_STAGING_TOTAL: usize = MAX_GRAPH_SLOTS * BATCHED_SCRATCH_CAP;

impl CudaState {
    /// Lazy-init the MoE stream pool on first access. Pool size is
    /// 4 by default; override via FERRUM_MOE_STREAMS env (1 disables
    /// multi-stream dispatch).
    pub fn moe_stream_pool(&mut self) -> &[Arc<CudaStream>] {
        if self.moe_streams.is_none() {
            let n = cuda_backend_runtime_env().moe_streams;
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

    // `vllm_moe_c_tmp` moved out of CudaState — see VLLM_MOE_C_TMP
    // process-global below + `with_vllm_moe_c_tmp` helper. Was on
    // per-state lazy-alloc; that caused INVALID_VALUE on every
    // graph replay since each new_context() reseats the buffer.

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
        let m = ensure_module(self.ordinal, &self.ctx, key, ptx_src);
        self.modules.insert(key, m.clone());
        m
    }

    pub(crate) fn func(
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

thread_local! {
    static CUDA_ALLOC_LABELS: std::cell::RefCell<Vec<&'static str>> =
        std::cell::RefCell::new(Vec::new());
}

#[must_use]
pub struct CudaAllocLabelGuard;

pub fn push_alloc_label(label: &'static str) -> CudaAllocLabelGuard {
    CUDA_ALLOC_LABELS.with(|labels| labels.borrow_mut().push(label));
    CudaAllocLabelGuard
}

impl Drop for CudaAllocLabelGuard {
    fn drop(&mut self) {
        CUDA_ALLOC_LABELS.with(|labels| {
            labels.borrow_mut().pop();
        });
    }
}

fn current_cuda_alloc_label() -> String {
    CUDA_ALLOC_LABELS.with(|labels| {
        let labels = labels.borrow();
        if labels.is_empty() {
            "label=<none>".to_string()
        } else {
            format!("label={}", labels.join(">"))
        }
    })
}

fn cuda_alloc_failed(
    op: &str,
    dtype: crate::backend::Dtype,
    n: usize,
    elem_bytes: usize,
    err: impl std::fmt::Debug,
) -> ! {
    let bytes = n.saturating_mul(elem_bytes);
    let mem_info = cudarc::driver::result::mem_get_info()
        .map(|(free, total)| format!("free={free} total={total}"))
        .unwrap_or_else(|mem_err| format!("mem_get_info_failed={mem_err:?}"));
    let alloc_label = current_cuda_alloc_label();
    let backtrace = std::backtrace::Backtrace::force_capture();
    panic!(
        "{op} failed: dtype={dtype:?} elements={n} bytes={bytes} {mem_info} {alloc_label}: {err:?}\n{backtrace}"
    );
}

impl Backend for CudaBackend {
    // Phase B-2: typed-buffer migration. `CudaBuf` is an enum over
    // `CudaSlice<{f16,f32,u32,i32,i8}>` — Phase B-1 added the wrapper,
    // this PR switches `Self::Buffer` to use it. Existing
    // `CudaSlice<f16>` ops migrate via `.as_f16()` / `.as_f16_mut()`
    // accessors on the wrapper. Integer storage (block tables,
    // expert ids, ...) gets a proper typed dtype tag instead of the
    // old i32-bit-cast-through-f16 type tunnel that under-allocated
    // by half (`alloc_u32` default was wrong on CUDA).
    type Buffer = crate::backend::CudaBuf;
    type Context = CudaState;
    type Timer = crate::backend::timer::CudaTimer;
    fn make_timer() -> Self::Timer {
        crate::backend::timer::CudaTimer::new()
    }
    // type GptqStore: removed in Phase C step 4e. GptqStoreCuda is
    // now a private (crate-internal) detail of CudaMarlinExpertStack.

    // ── Lifecycle ────────────────────────────────────────────────────────

    fn new_context() -> Self::Context {
        // Reuse the process-global stream populated by `default_stream()`.
        // Model constructors call `B::from_slice` thousands of times to
        // upload weights BEFORE the engine ever calls `new_context()`, so
        // `default_stream()` has already lazily spun up a stream. Reusing
        // it here keeps allocations + ops on the SAME stream — no
        // cross-stream synchronization needed.
        let ordinal = current_device_ordinal();
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
            ordinal,
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
        }
    }

    fn with_device_ordinal<R>(device_ordinal: Option<usize>, body: impl FnOnce() -> R) -> R {
        with_cuda_device_ordinal(device_ordinal, body)
    }

    fn supports_device_ordinal_scope() -> bool {
        true
    }

    /// Phase D step 2+3 unified typed allocator. Replaces alloc_u32 and
    /// the per-dtype family (alloc_typed_i32 / etc. were never needed).
    fn alloc_typed(dtype: crate::backend::Dtype, n: usize) -> Self::Buffer {
        use crate::backend::{CudaBuf, Dtype};
        let n = n.max(1);
        with_stream(|stream| match dtype {
            Dtype::F32 => match stream.alloc_zeros::<f32>(n) {
                Ok(buf) => CudaBuf::from_f32(buf),
                Err(err) => cuda_alloc_failed(
                    "CudaBackend::alloc_typed alloc_zeros",
                    dtype,
                    n,
                    std::mem::size_of::<f32>(),
                    err,
                ),
            },
            Dtype::F16 => match stream.alloc_zeros::<f16>(n) {
                Ok(buf) => CudaBuf::from_f16(buf),
                Err(err) => cuda_alloc_failed(
                    "CudaBackend::alloc_typed alloc_zeros",
                    dtype,
                    n,
                    std::mem::size_of::<f16>(),
                    err,
                ),
            },
            Dtype::U32 => match stream.alloc_zeros::<u32>(n) {
                Ok(buf) => CudaBuf::from_u32(buf),
                Err(err) => cuda_alloc_failed(
                    "CudaBackend::alloc_typed alloc_zeros",
                    dtype,
                    n,
                    std::mem::size_of::<u32>(),
                    err,
                ),
            },
            Dtype::I32 => match stream.alloc_zeros::<i32>(n) {
                Ok(buf) => CudaBuf::from_i32(buf),
                Err(err) => cuda_alloc_failed(
                    "CudaBackend::alloc_typed alloc_zeros",
                    dtype,
                    n,
                    std::mem::size_of::<i32>(),
                    err,
                ),
            },
            Dtype::I8 => match stream.alloc_zeros::<i8>(n) {
                Ok(buf) => CudaBuf::from_i8(buf),
                Err(err) => cuda_alloc_failed(
                    "CudaBackend::alloc_typed alloc_zeros",
                    dtype,
                    n,
                    std::mem::size_of::<i8>(),
                    err,
                ),
            },
        })
    }

    /// Phase D step 2+3 unified typed uploader. Dispatches on
    /// `T::DTYPE` to select the right CudaBuf variant. Replaces
    /// `from_slice_i32` + ad-hoc `from_u32` helpers.
    fn from_slice_typed<T: crate::backend::HostDtype>(data: &[T]) -> Self::Buffer {
        use crate::backend::{CudaBuf, Dtype};
        with_stream(|stream| match T::DTYPE {
            Dtype::F32 => {
                // SAFETY: T::DTYPE = F32 implies T = f32 (HostDtype is
                // sealed by trait coherence on concrete primitives).
                let host: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
                CudaBuf::from_f32(stream.clone_htod(host).expect("cuda htod f32"))
            }
            Dtype::F16 => {
                let host: &[f16] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f16, data.len()) };
                CudaBuf::from_f16(stream.clone_htod(host).expect("cuda htod f16"))
            }
            Dtype::U32 => {
                let host: &[u32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) };
                CudaBuf::from_u32(stream.clone_htod(host).expect("cuda htod u32"))
            }
            Dtype::I32 => {
                let host: &[i32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, data.len()) };
                CudaBuf::from_i32(stream.clone_htod(host).expect("cuda htod i32"))
            }
            Dtype::I8 => {
                let host: &[i8] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i8, data.len()) };
                CudaBuf::from_i8(stream.clone_htod(host).expect("cuda htod i8"))
            }
        })
    }

    /// Phase D step 2+3 unified typed in-place write. Buffer dtype
    /// must match `T::DTYPE` (panic otherwise via `.as_<T>_mut()`).
    /// Replaces write_u32 / write_i32_into / write_f32_into.
    fn write_typed<T: crate::backend::HostDtype>(
        ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        data: &[T],
    ) {
        use crate::backend::Dtype;
        if data.is_empty() {
            return;
        }
        let stream = ctx.stream.clone();
        // memcpy_htod is enqueued on ctx.stream — stream-ordered against
        // subsequent kernel launches on the same stream. No explicit
        // synchronize: (1) cudarc's stream_synced_slice handles host-Vec
        // lifetime so we can drop `data` immediately, and (2) explicit
        // sync inside a CUDA-graph capture region raises
        // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED — must not call it here
        // when callers are inside `B::begin_graph_capture()`. The earlier
        // sync was belt-and-suspenders for the kv_cache_append parity
        // test; that test's actual fix was switching off the legacy
        // NULL-stream cuMemcpyHtoD_v2, not the sync.
        match T::DTYPE {
            Dtype::U32 => {
                let host: &[u32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len()) };
                let d = dst.as_u32_mut();
                stream.memcpy_htod(host, d).expect("cuda write_typed u32");
            }
            Dtype::I32 => {
                let host: &[i32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, data.len()) };
                let d = dst.as_i32_mut();
                stream.memcpy_htod(host, d).expect("cuda write_typed i32");
            }
            Dtype::F32 => {
                let host: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };
                let d = dst.as_f32_mut();
                stream.memcpy_htod(host, d).expect("cuda write_typed f32");
            }
            Dtype::F16 => {
                let host: &[f16] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f16, data.len()) };
                let d = dst.as_f16_mut();
                stream.memcpy_htod(host, d).expect("cuda write_typed f16");
            }
            Dtype::I8 => {
                let host: &[i8] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i8, data.len()) };
                let d = dst.as_i8_mut();
                stream.memcpy_htod(host, d).expect("cuda write_typed i8");
            }
        }
    }

    fn sync(ctx: &mut Self::Context) {
        ctx.stream.synchronize().expect("CudaBackend: stream sync");
    }

    fn graph_capture_in_flight(ctx: &Self::Context) -> bool {
        ctx.capture_in_flight
    }

    fn alloc(len: usize) -> Self::Buffer {
        with_stream(|stream| {
            let len = len.max(1);
            match unsafe { stream.alloc::<f16>(len) } {
                Ok(buf) => crate::backend::CudaBuf::from_f16(buf),
                Err(err) => cuda_alloc_failed(
                    "CudaBackend::alloc",
                    crate::backend::Dtype::F16,
                    len,
                    std::mem::size_of::<f16>(),
                    err,
                ),
            }
        })
    }

    fn from_slice(data: &[f32]) -> Self::Buffer {
        let host: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
        with_stream(|stream| {
            crate::backend::CudaBuf::from_f16(stream.clone_htod(&host).expect("cuda htod"))
        })
    }

    fn write_f32_to_activation(ctx: &mut Self::Context, dst: &mut Self::Buffer, data: &[f32]) {
        if data.is_empty() {
            return;
        }
        match dst.dtype() {
            crate::backend::Dtype::F16 => {
                let host: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
                let mut dst_view = dst.as_f16_mut().slice_mut(0..data.len());
                ctx.stream
                    .memcpy_htod(&host, &mut dst_view)
                    .expect("cuda write_f32_to_activation f16");
            }
            crate::backend::Dtype::F32 => {
                let mut dst_view = dst.as_f32_mut().slice_mut(0..data.len());
                ctx.stream
                    .memcpy_htod(data, &mut dst_view)
                    .expect("cuda write_f32_to_activation f32");
            }
            other => panic!(
                "CudaBackend::write_f32_to_activation unsupported dtype {}",
                other.name()
            ),
        }
    }

    fn supports_device_f32_residual_shadow() -> bool {
        true
    }

    fn activation_to_f32_shadow(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        dst_f32: &mut Self::Buffer,
        len: usize,
    ) {
        if len == 0 {
            return;
        }
        assert_eq!(
            dst_f32.dtype(),
            crate::backend::Dtype::F32,
            "CudaBackend::activation_to_f32_shadow dst must be F32, got {}",
            dst_f32.dtype().name()
        );
        match src.dtype() {
            crate::backend::Dtype::F16 => {
                let func = ctx.func(
                    "sandwich_norm",
                    ptx::SANDWICH_NORM,
                    "activation_to_f32_shadow_f16",
                );
                let n_i32 = len as i32;
                let block = 256u32;
                let grid = ((len as u32) + block - 1) / block;
                let stream = ctx.stream.clone();
                let mut b = stream.launch_builder(&func);
                b.arg(src);
                b.arg(dst_f32);
                b.arg(&n_i32);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .expect("activation_to_f32_shadow launch");
            }
            crate::backend::Dtype::F32 => {
                Self::copy_slice(ctx, src, 0, dst_f32, 0, len);
            }
            other => panic!(
                "CudaBackend::activation_to_f32_shadow unsupported src dtype {}",
                other.name()
            ),
        }
    }

    fn rms_norm_activation_to_f32(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        eps: f32,
        out_f32: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        match (input.dtype(), weight.dtype(), out_f32.dtype()) {
            (crate::backend::Dtype::F16, crate::backend::Dtype::F16, crate::backend::Dtype::F32) => {
                let func = ctx.func("sandwich_norm", ptx::SANDWICH_NORM, "rms_norm_f16_to_f32");
                let dim_i32 = dim as i32;
                let stream = ctx.stream.clone();
                let mut b = stream.launch_builder(&func);
                b.arg(input);
                b.arg(weight);
                b.arg(out_f32);
                b.arg(&dim_i32);
                b.arg(&eps);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (tokens as u32, 1, 1),
                        block_dim: (dim.min(1024) as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .expect("rms_norm_activation_to_f32 launch");
            }
            (crate::backend::Dtype::F32, crate::backend::Dtype::F32, crate::backend::Dtype::F32) => {
                Self::rms_norm(ctx, input, weight, eps, out_f32, tokens, dim);
            }
            (input_dtype, weight_dtype, out_dtype) => panic!(
                "CudaBackend::rms_norm_activation_to_f32 unsupported dtypes input={} weight={} out={}",
                input_dtype.name(),
                weight_dtype.name(),
                out_dtype.name()
            ),
        }
    }

    fn rms_norm_activation_add_to_f32(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        weight: &Self::Buffer,
        eps: f32,
        residual_f32: &mut Self::Buffer,
        scratch_f32: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        match (
            input.dtype(),
            weight.dtype(),
            residual_f32.dtype(),
            scratch_f32.dtype(),
        ) {
            (
                crate::backend::Dtype::F16,
                crate::backend::Dtype::F16,
                crate::backend::Dtype::F32,
                crate::backend::Dtype::F32,
            ) => {
                let func = ctx.func(
                    "sandwich_norm",
                    ptx::SANDWICH_NORM,
                    "rms_norm_f16_add_to_f32",
                );
                let dim_i32 = dim as i32;
                let stream = ctx.stream.clone();
                let mut b = stream.launch_builder(&func);
                b.arg(input);
                b.arg(weight);
                b.arg(residual_f32);
                b.arg(&dim_i32);
                b.arg(&eps);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (tokens as u32, 1, 1),
                        block_dim: (dim.min(1024) as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .expect("rms_norm_activation_add_to_f32 launch");
            }
            _ => {
                Self::rms_norm_activation_to_f32(ctx, input, weight, eps, scratch_f32, tokens, dim);
                Self::add_inplace(ctx, residual_f32, scratch_f32, tokens * dim);
            }
        }
    }

    fn rms_norm_f32_to_activation(
        ctx: &mut Self::Context,
        input_f32: &Self::Buffer,
        weight: &Self::Buffer,
        eps: f32,
        out: &mut Self::Buffer,
        tokens: usize,
        dim: usize,
    ) {
        match (input_f32.dtype(), weight.dtype(), out.dtype()) {
            (crate::backend::Dtype::F32, crate::backend::Dtype::F16, crate::backend::Dtype::F16) => {
                let func = ctx.func("sandwich_norm", ptx::SANDWICH_NORM, "rms_norm_f32_to_f16");
                let dim_i32 = dim as i32;
                let stream = ctx.stream.clone();
                let mut b = stream.launch_builder(&func);
                b.arg(input_f32);
                b.arg(weight);
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
                .expect("rms_norm_f32_to_activation launch");
            }
            (crate::backend::Dtype::F32, crate::backend::Dtype::F32, crate::backend::Dtype::F32) => {
                Self::rms_norm(ctx, input_f32, weight, eps, out, tokens, dim);
            }
            (input_dtype, weight_dtype, out_dtype) => panic!(
                "CudaBackend::rms_norm_f32_to_activation unsupported dtypes input={} weight={} out={}",
                input_dtype.name(),
                weight_dtype.name(),
                out_dtype.name()
            ),
        }
    }

    fn to_vec(buf: &Self::Buffer, len: usize) -> Vec<f32> {
        with_stream(|stream| {
            // cudarc asserts host.len() >= buf.len() — but we may want a
            // PARTIAL read (len < buf capacity), e.g. reading only 4 rows
            // out of a batch_logits buffer sized for max_batch. Slice the
            // device buffer so its reported length matches `len`.
            match buf.dtype() {
                crate::backend::Dtype::F16 => {
                    let mut host = vec![f16::ZERO; len];
                    let view = buf.as_f16().slice(0..len);
                    stream.memcpy_dtoh(&view, &mut host).expect("cuda dtoh f16");
                    stream.synchronize().expect("cuda dtoh sync");
                    host.into_iter().map(|x| x.to_f32()).collect()
                }
                crate::backend::Dtype::F32 => {
                    let mut host = vec![0.0f32; len];
                    let view = buf.as_f32().slice(0..len);
                    stream.memcpy_dtoh(&view, &mut host).expect("cuda dtoh f32");
                    stream.synchronize().expect("cuda dtoh sync");
                    host
                }
                other => panic!(
                    "CudaBackend::to_vec unsupported dtype {} (expected F16 or F32)",
                    other.name()
                ),
            }
        })
    }

    fn argmax_rows_f16(
        ctx: &mut Self::Context,
        logits: &Self::Buffer,
        m: usize,
        n: usize,
    ) -> Result<Vec<u32>> {
        // Greedy fast path: one kernel + tiny D2H replaces the
        // m × n × 2 bytes logit download + host-side argmax scan.
        // At c=32, vocab=152064: 19.5 MB + 4.8 ms CPU → 128 B + ~0.3 ms GPU.
        let func = ctx.func("argmax_rows", ptx::ARGMAX_ROWS, "argmax_rows_f16");
        let stream = ctx.stream.clone();
        // Output buffer: process-global, grown lazily. Reuses the same
        // device allocation across iters (avoids ~30-50 µs / iter for
        // `stream.alloc_zeros`). Mirrors the MARLIN_GATHER_SCRATCH /
        // VLLM_MOE_C_TMP pattern; the slot is owned process-wide so the
        // GPU address it hands out is stable for the engine's life
        // (graph-capture safe — see vllm_moe_c_tmp's doc for rationale).
        let host = with_argmax_out(&stream, ctx.ordinal, m, |out_dev| -> Result<Vec<i32>> {
            let n_i32 = n as i32;
            let mut b = stream.launch_builder(&func);
            b.arg(logits);
            b.arg(&n_i32);
            b.arg(&mut *out_dev);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (m as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| FerrumError::internal(format!("argmax_rows launch: {e}")))?;
            let mut host = vec![0i32; m];
            // Use a sliced view so cudarc's host.len() == src.len() guard
            // accepts (out_dev may be capacity > m).
            let view = out_dev.slice(0..m);
            stream
                .memcpy_dtoh(&view, &mut host)
                .map_err(|e| FerrumError::internal(format!("argmax_rows dtoh: {e}")))?;
            stream
                .synchronize()
                .map_err(|e| FerrumError::internal(format!("argmax_rows sync: {e}")))?;
            Ok(host)
        })?;
        Ok(host.into_iter().map(|x| x as u32).collect())
    }

    fn argmax_rows_f16_masked(
        ctx: &mut Self::Context,
        logits: &Self::Buffer,
        valid_token_mask: &Self::Buffer,
        mask_len: usize,
        m: usize,
        n: usize,
    ) -> Result<Vec<u32>> {
        let func = ctx.func("argmax_rows", ptx::ARGMAX_ROWS, "argmax_rows_f16_masked");
        let stream = ctx.stream.clone();
        let host = with_argmax_out(&stream, ctx.ordinal, m, |out_dev| -> Result<Vec<i32>> {
            let n_i32 = n as i32;
            let mask_len_i32 = mask_len as i32;
            let mut b = stream.launch_builder(&func);
            b.arg(logits);
            b.arg(&n_i32);
            b.arg(valid_token_mask);
            b.arg(&mask_len_i32);
            b.arg(&mut *out_dev);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (m as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| FerrumError::internal(format!("argmax_rows_masked launch: {e}")))?;
            let mut host = vec![0i32; m];
            let view = out_dev.slice(0..m);
            stream
                .memcpy_dtoh(&view, &mut host)
                .map_err(|e| FerrumError::internal(format!("argmax_rows_masked dtoh: {e}")))?;
            stream
                .synchronize()
                .map_err(|e| FerrumError::internal(format!("argmax_rows_masked sync: {e}")))?;
            Ok(host)
        })?;
        Ok(host.into_iter().map(|x| x as u32).collect())
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
        let x_dtype = x.dtype();
        assert_eq!(
            x_dtype,
            w.dtype(),
            "CudaBackend::rms_norm dtype mismatch: x={} w={}",
            x_dtype.name(),
            w.dtype().name()
        );
        assert_eq!(
            x_dtype,
            out.dtype(),
            "CudaBackend::rms_norm dtype mismatch: x={} out={}",
            x_dtype.name(),
            out.dtype().name()
        );
        let fn_name = match x_dtype {
            crate::backend::Dtype::F16 => "rms_norm_f16",
            crate::backend::Dtype::F32 => "rms_norm_f32",
            other => panic!("CudaBackend::rms_norm unsupported dtype {}", other.name()),
        };
        let func = ctx.func("rms_norm", ptx::RMS_NORM, fn_name);
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
        let (a_ptr, _rec_a) = b.as_f16().device_ptr(&ctx.stream); // cuBLAS arg "A" = weight = our `b`
        let (b_ptr, _rec_b) = a.as_f16().device_ptr(&ctx.stream); // cuBLAS arg "B" = input = our `a`
        let (c_ptr, _rec_c) = out.as_f16_mut().device_ptr_mut(&ctx.stream);
        with_blas_scalars(ctx.ordinal, |alpha_f32, beta_f32| {
            let (alpha_ptr, _ga) = alpha_f32.device_ptr(&ctx.stream);
            let (beta_ptr, _gb) = beta_f32.device_ptr(&ctx.stream);

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
        });
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
            let sliding_window = cfg.sliding_window as i32;
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
            let env_cap = cuda_backend_runtime_env()
                .cuda_max_kv
                .unwrap_or(DECODE_MAX_KV_POS_DEFAULT);
            let max_kv_pos = capacity.min(env_cap as i32) as u32;
            let active_kv_pos = if cfg.sliding_window > 0 {
                max_kv_pos.min(cfg.sliding_window as u32)
            } else {
                max_kv_pos
            };
            let shared_mem = active_kv_pos * 4;
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
                Some(
                    decode_state_slot_for_ordinal(ctx.ordinal)
                        .read()
                        .expect("DECODE_STATE poisoned"),
                )
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
            bld.arg(&sliding_window);
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

    #[allow(clippy::too_many_arguments)]
    fn recurrent_gated_delta_rule_f32(
        ctx: &mut Self::Context,
        query: &Self::Buffer,
        key: &Self::Buffer,
        value: &Self::Buffer,
        g: &Self::Buffer,
        beta: &Self::Buffer,
        initial_state: &Self::Buffer,
        out: &mut Self::Buffer,
        final_state: &mut Self::Buffer,
        tokens: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        use_qk_l2norm: bool,
        scale: f32,
    ) -> Result<()> {
        gated_delta_rule::recurrent_gated_delta_rule_f32(
            ctx,
            query,
            key,
            value,
            g,
            beta,
            initial_state,
            out,
            final_state,
            tokens,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            use_qk_l2norm,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_prepare_f32(
        ctx: &mut Self::Context,
        mixed_qkv_raw: &Self::Buffer,
        conv_weight: &Self::Buffer,
        a_raw: &Self::Buffer,
        b_raw: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        query: &mut Self::Buffer,
        key: &mut Self::Buffer,
        value: &mut Self::Buffer,
        g: &mut Self::Buffer,
        beta: &mut Self::Buffer,
        tokens: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        apply_qk_l2norm: bool,
    ) -> Result<()> {
        linear_attention::linear_attention_prepare_f32(
            ctx,
            mixed_qkv_raw,
            conv_weight,
            a_raw,
            b_raw,
            a_log,
            dt_bias,
            query,
            key,
            value,
            g,
            beta,
            tokens,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
            apply_qk_l2norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gated_rms_norm_f32(
        ctx: &mut Self::Context,
        core: &Self::Buffer,
        z: &Self::Buffer,
        weight: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        linear_attention::gated_rms_norm_f32(ctx, core, z, weight, out, tokens, heads, dim, eps)
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
        match (src.dtype(), dst.dtype()) {
            (crate::backend::Dtype::F16, crate::backend::Dtype::F16) => {
                let src_view = src.as_f16().slice(src_offset..src_offset + len);
                let mut dst_view = dst.as_f16_mut().slice_mut(dst_offset..dst_offset + len);
                ctx.stream
                    .memcpy_dtod(&src_view, &mut dst_view)
                    .expect("copy_slice f16 dtod");
            }
            (crate::backend::Dtype::F32, crate::backend::Dtype::F32) => {
                let src_view = src.as_f32().slice(src_offset..src_offset + len);
                let mut dst_view = dst.as_f32_mut().slice_mut(dst_offset..dst_offset + len);
                ctx.stream
                    .memcpy_dtod(&src_view, &mut dst_view)
                    .expect("copy_slice f32 dtod");
            }
            (src_dtype, dst_dtype) => panic!(
                "CudaBackend::copy_slice unsupported dtypes src={} dst={}",
                src_dtype.name(),
                dst_dtype.name()
            ),
        }
    }

    // ── Embedding ───────────────────────────────────────────────────────

    fn embedding_lookup_dev(
        ctx: &mut Self::Context,
        table: &Self::Buffer,
        ids: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        dim: usize,
    ) {
        // Device-buffer variant — no clone_htod, so the kernel launch
        // captures cleanly under CUDA Graph. `ids` is treated as the
        // I32 variant of CudaBuf (the kernel reads `const int*`).
        let dim_i32 = dim as i32;
        let batch_i32 = batch as i32;
        let block = 256u32;
        let grid_x = ((dim as u32) + block - 1) / block;
        let func = ctx.func(
            "embedding_lookup",
            ptx::EMBEDDING_LOOKUP,
            "embedding_lookup_f16",
        );
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(table);
        b.arg(ids);
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
        .expect("embedding_lookup_dev launch");
    }

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
            let dec_guard = decode_state_slot_for_ordinal(ctx.ordinal)
                .read()
                .expect("DECODE_STATE poisoned");
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

    fn scale_inplace(ctx: &mut Self::Context, buf: &mut Self::Buffer, scale: f32, len: usize) {
        // The trait's host-roundtrip default would rebuild the buffer via
        // from_slice (F32) and silently flip the CUDA lane's f16 dtype —
        // every downstream kernel then misreads the residual. Keep it
        // on-device and typed.
        let func = ctx.func("fused_silu_mul", ptx::FUSED_SILU_MUL, "scale_inplace_f16");
        let n_i32 = len as i32;
        let block = 256u32;
        let grid = ((len as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(buf);
        b.arg(&scale);
        b.arg(&n_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("scale_inplace launch");
    }

    fn fused_gelu_tanh_mul_split(
        ctx: &mut Self::Context,
        gate_up: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        im: usize,
    ) {
        // GeGLU (Gemma family): same interleaved [tokens, 2*im] layout as
        // the SiLU variant, gelu_pytorch_tanh activation.
        let func = ctx.func(
            "fused_silu_mul",
            ptx::FUSED_SILU_MUL,
            "fused_gelu_tanh_mul_interleaved_f16",
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
        .expect("fused_gelu_tanh_mul_split launch");
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
        with_batched_scratch_mut(ctx.ordinal, |slot_g| {
            for i in 0..m {
                let (cp, _) = caches[i].as_f16().device_ptr(&stream);
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
        sliding_window: usize,
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
        let sliding_window_i32 = sliding_window as i32;
        // Shared mem must cover post-append max kv_len. Caller passes
        // `max_valid_kv` already accounting for the +1; sizing also
        // bounded by capacity to mirror the per-item kernel's pattern.
        let active_kv = if sliding_window > 0 {
            max_valid_kv.min(sliding_window)
        } else {
            max_valid_kv
        };
        let shared_bytes = (active_kv.min(capacity).max(1) as u32) * 4;
        with_batched_scratch_mut(ctx.ordinal, |slot_g| {
            for i in 0..m {
                let (kp, _) = k_caches[i].as_f16().device_ptr(&stream);
                let (vp, _) = v_caches[i].as_f16().device_ptr(&stream);
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
            b.arg(&sliding_window_i32);
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
            Some(
                decode_state_slot_for_ordinal(ctx.ordinal)
                    .read()
                    .expect("DECODE_STATE poisoned"),
            )
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
        Self::split_qkv(
            ctx, qkv, &mut q_buf, &mut k_buf, &mut v_buf, tokens, q_dim, kv_dim,
        );
        Self::qk_norm_rope(
            ctx, &q_buf, q_norm_w, cos, sin, q_out, tokens, q_heads, head_dim, pos_offset, eps,
            qk_mode,
        );
        Self::qk_norm_rope(
            ctx, &k_buf, k_norm_w, cos, sin, k_out, tokens, kv_heads, head_dim, pos_offset, eps,
            qk_mode,
        );
        // V: no norm + RoPE-only (qk_mode=0); pass q_norm_w as a dummy
        // (kernel ignores it when mode=0).
        Self::qk_norm_rope(
            ctx, &v_buf, q_norm_w, cos, sin, v_out, tokens, kv_heads, head_dim, pos_offset, eps, 0,
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
            Some(
                decode_state_slot_for_ordinal(ctx.ordinal)
                    .read()
                    .expect("DECODE_STATE poisoned"),
            )
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
        let residual_dtype = residual.dtype();
        let x_dtype = x.dtype();
        assert_eq!(
            residual_dtype,
            x_dtype,
            "CudaBackend::add_inplace dtype mismatch: residual={} x={}",
            residual_dtype.name(),
            x_dtype.name()
        );
        let fn_name = match residual_dtype {
            crate::backend::Dtype::F16 => "residual_add_inplace_f16",
            crate::backend::Dtype::F32 => "residual_add_inplace_f32",
            other => panic!(
                "CudaBackend::add_inplace unsupported dtype {}",
                other.name()
            ),
        };
        let func = ctx.func("residual_add", ptx::RESIDUAL_ADD, fn_name);
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

    fn scaled_add_inplace(
        ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        src: &Self::Buffer,
        scale: f32,
        len: usize,
    ) {
        if len == 0 {
            return;
        }
        let dst_dtype = dst.dtype();
        let src_dtype = src.dtype();
        assert_eq!(
            dst_dtype,
            src_dtype,
            "CudaBackend::scaled_add_inplace dtype mismatch: dst={} src={}",
            dst_dtype.name(),
            src_dtype.name()
        );
        assert!(
            len <= dst.len() && len <= src.len(),
            "CudaBackend::scaled_add_inplace len={len} exceeds dst_len={} src_len={}",
            dst.len(),
            src.len()
        );
        let fn_name = match dst_dtype {
            crate::backend::Dtype::F16 => "scaled_add_inplace_f16",
            crate::backend::Dtype::F32 => "scaled_add_inplace_f32",
            other => panic!(
                "CudaBackend::scaled_add_inplace unsupported dtype {}",
                other.name()
            ),
        };
        let func = ctx.func("scaled_add_inplace", ptx::SCALED_ADD_INPLACE, fn_name);
        let n_i32 = len as i32;
        let block = 256u32;
        let grid = ((len as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(dst);
        b.arg(src);
        b.arg(&scale);
        b.arg(&n_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("scaled_add_inplace launch");
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

        let (gu_base, _g) = gate_up.as_f16().device_ptr(&stream);
        let (out_base, _g2) = out.as_f16_mut().device_ptr_mut(&stream);
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
        let (ptr, _g) = buf.as_f16().device_ptr(&stream);
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

static GLOBAL_STREAMS: std::sync::OnceLock<std::sync::RwLock<HashMap<usize, Arc<CudaStream>>>> =
    std::sync::OnceLock::new();

fn stream_slots() -> &'static std::sync::RwLock<HashMap<usize, Arc<CudaStream>>> {
    GLOBAL_STREAMS.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
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
    let ordinal = current_device_ordinal();
    if let Some(s) = stream_slots()
        .read()
        .expect("GLOBAL_STREAMS poisoned")
        .get(&ordinal)
    {
        return s.clone();
    }
    let mut w = stream_slots().write().expect("GLOBAL_STREAMS poisoned");
    if !w.contains_key(&ordinal) {
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
        w.insert(ordinal, stream);
    }
    w.get(&ordinal).cloned().expect("just inserted")
}

fn with_stream<R>(f: impl FnOnce(&Arc<CudaStream>) -> R) -> R {
    let stream = default_stream();
    f(&stream)
}

/// Install a stream as the ordinal-local default for context-free ops.
/// Subsequent `alloc`/`from_slice`/`to_vec` calls made under the same
/// device scope route through it. If `default_stream` already lazily
/// created a stream for that ordinal, this replaces it.
pub fn install_thread_stream(stream: Arc<CudaStream>) {
    stream_slots()
        .write()
        .expect("GLOBAL_STREAMS poisoned")
        .insert(current_device_ordinal(), stream);
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

static DECODE_STATES: std::sync::OnceLock<
    std::sync::RwLock<HashMap<usize, &'static std::sync::RwLock<Option<DecodeStateBufs>>>>,
> = std::sync::OnceLock::new();

fn decode_state_slots(
) -> &'static std::sync::RwLock<HashMap<usize, &'static std::sync::RwLock<Option<DecodeStateBufs>>>>
{
    DECODE_STATES.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
}

pub(super) fn decode_state_slot_for_ordinal(
    ordinal: usize,
) -> &'static std::sync::RwLock<Option<DecodeStateBufs>> {
    {
        let g = decode_state_slots().read().expect("DECODE_STATES poisoned");
        if let Some(slot) = g.get(&ordinal) {
            return *slot;
        }
    }
    let mut w = decode_state_slots()
        .write()
        .expect("DECODE_STATES poisoned");
    *w.entry(ordinal)
        .or_insert_with(|| Box::leak(Box::new(std::sync::RwLock::new(None))))
}

fn ensure_decode_state_bufs(stream: &Arc<CudaStream>) {
    let slot = decode_state_slot_for_ordinal(current_device_ordinal());
    let guard = slot.read().expect("DECODE_STATE poisoned");
    if guard.is_some() {
        return;
    }
    drop(guard);
    let mut w = slot.write().expect("DECODE_STATE poisoned");
    if w.is_none() {
        let token = unsafe { stream.alloc::<u32>(1) }.expect("token_buf alloc");
        let pos = unsafe { stream.alloc::<i32>(1) }.expect("pos_buf alloc");
        let kv = unsafe { stream.alloc::<i32>(1) }.expect("kv_buf alloc");
        *w = Some(DecodeStateBufs { token, pos, kv });
    }
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

static BLAS_HANDLES: std::sync::OnceLock<std::sync::RwLock<HashMap<usize, BlasSlot>>> =
    std::sync::OnceLock::new();

fn blas_slots() -> &'static std::sync::RwLock<HashMap<usize, BlasSlot>> {
    BLAS_HANDLES.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
}

fn ensure_blas_handle(stream: &Arc<CudaStream>) -> Arc<CudaBlas> {
    let ordinal = current_device_ordinal();
    if let Some(slot) = blas_slots().read().expect("BLAS poisoned").get(&ordinal) {
        return slot.blas.clone();
    }
    let mut w = blas_slots().write().expect("BLAS poisoned");
    if !w.contains_key(&ordinal) {
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
        w.insert(
            ordinal,
            BlasSlot {
                blas,
                _workspace: workspace,
                alpha_f32,
                beta_f32,
            },
        );
    }
    w.get(&ordinal).unwrap().blas.clone()
}

/// Access the process-global alpha/beta device scalars for cuBLAS.
fn with_blas_scalars<R>(
    ordinal: usize,
    f: impl FnOnce(&CudaSlice<f32>, &CudaSlice<f32>) -> R,
) -> R {
    let g = blas_slots().read().expect("BLAS poisoned");
    let s = g.get(&ordinal).expect("BLAS not init");
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

static BATCHED_SCRATCH: std::sync::OnceLock<
    std::sync::RwLock<HashMap<usize, &'static std::sync::RwLock<Option<BatchedScratchSlot>>>>,
> = std::sync::OnceLock::new();

fn batched_scratch_slots() -> &'static std::sync::RwLock<
    HashMap<usize, &'static std::sync::RwLock<Option<BatchedScratchSlot>>>,
> {
    BATCHED_SCRATCH.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
}

fn batched_scratch_slot_for_ordinal(
    ordinal: usize,
) -> &'static std::sync::RwLock<Option<BatchedScratchSlot>> {
    {
        let g = batched_scratch_slots()
            .read()
            .expect("BATCHED_SCRATCH poisoned");
        if let Some(slot) = g.get(&ordinal) {
            return *slot;
        }
    }
    let mut w = batched_scratch_slots()
        .write()
        .expect("BATCHED_SCRATCH poisoned");
    *w.entry(ordinal)
        .or_insert_with(|| Box::leak(Box::new(std::sync::RwLock::new(None))))
}

fn ensure_batched_scratch(stream: &Arc<CudaStream>) {
    let slot = batched_scratch_slot_for_ordinal(current_device_ordinal());
    {
        let g = slot.read().expect("BATCHED_SCRATCH poisoned");
        if g.is_some() {
            return;
        }
    }
    let mut w = slot.write().expect("BATCHED_SCRATCH poisoned");
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
fn with_batched_scratch_mut<R>(ordinal: usize, f: impl FnOnce(&mut BatchedScratchSlot) -> R) -> R {
    let mut g = batched_scratch_slot_for_ordinal(ordinal)
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

static MODULES: std::sync::OnceLock<
    std::sync::Mutex<HashMap<(usize, &'static str), Arc<CudaModule>>>,
> = std::sync::OnceLock::new();

fn modules_cache() -> &'static std::sync::Mutex<HashMap<(usize, &'static str), Arc<CudaModule>>> {
    MODULES.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

pub(super) fn ensure_module(
    ordinal: usize,
    ctx: &Arc<CudaContext>,
    key: &'static str,
    ptx_src: &str,
) -> Arc<CudaModule> {
    let cache_key = (ordinal, key);
    {
        let g = modules_cache().lock().expect("MODULES poisoned");
        if let Some(m) = g.get(&cache_key) {
            return m.clone();
        }
    }
    let mut g = modules_cache().lock().expect("MODULES poisoned");
    if let Some(m) = g.get(&cache_key) {
        return m.clone();
    }
    let m = ctx
        .load_module(Ptx::from_src(ptx_src.to_string()))
        .unwrap_or_else(|e| panic!("CudaBackend: load_module({key}): {e}"));
    g.insert(cache_key, m.clone());
    m
}

// CUDA: existing KV cache path is FP16.
impl crate::backend::BackendKvDtype<crate::backend::KvFp16> for CudaBackend {
    type KvBuffer = <Self as crate::backend::Backend>::Buffer;
    type KvScales = ();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_runtime_env_parses_values() {
        let env = CudaBackendRuntimeEnv::from_env_vars([
            ("FERRUM_MOE_STREAMS", "8"),
            ("FERRUM_CUDA_MAX_KV", "16384"),
            ("FERRUM_CUDA_DEVICE", "2"),
        ]);

        assert_eq!(env.moe_streams, 8);
        assert_eq!(env.cuda_max_kv, Some(16384));
        assert_eq!(env.cuda_device, 2);
    }

    #[test]
    fn cuda_backend_runtime_env_defaults_invalid_values() {
        let env = CudaBackendRuntimeEnv::from_env_vars([
            ("FERRUM_MOE_STREAMS", "0"),
            ("FERRUM_CUDA_MAX_KV", "invalid"),
            ("FERRUM_CUDA_DEVICE", "invalid"),
        ]);

        assert_eq!(env.moe_streams, 1);
        assert_eq!(env.cuda_max_kv, None);
        assert_eq!(env.cuda_device, 0);
    }

    #[test]
    fn cuda_device_scope_nests_and_restores() {
        let default = current_device_ordinal();

        with_cuda_device_ordinal(Some(1), || {
            assert_eq!(current_device_ordinal(), 1);
            with_cuda_device_ordinal(Some(2), || {
                assert_eq!(current_device_ordinal(), 2);
            });
            assert_eq!(current_device_ordinal(), 1);
        });

        assert_eq!(current_device_ordinal(), default);
    }
}
