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
//     `GraphSlotRaw` + instance graph cache helpers → `cuda/graph.rs`.
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
// CudaBackend` (the core trait) + original `CudaState` resource owner
// + `KvFp16` BackendKvDtype impl.
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
mod qkv;
pub mod quant;

// Audit #9: CUDA-only kernels moved from crate-root to backend/cuda/.
// Re-exported via `pub use backend::cuda::{...}` (or the more specific
// `pub use backend::cuda::foo::Foo`) in `crate::lib` so the historical
// `ferrum_kernels::foo::*` public paths + internal `crate::foo::*`
// references keep working unchanged.
pub mod cublas;
#[cfg(feature = "candle-cuda-compat")]
pub mod cuda_decode;
#[cfg(feature = "candle-cuda-compat")]
pub mod cuda_graph;
#[cfg(feature = "candle-cuda-compat")]
pub mod decode_attention;
pub mod decode_buffers;
#[cfg(feature = "candle-cuda-compat")]
pub mod fused_add_rms_norm;
#[cfg(feature = "candle-cuda-compat")]
pub mod fused_silu_mul;
pub mod gpu_paged_kv;
pub mod marlin;
#[cfg(not(target_os = "windows"))]
pub mod nccl_comm;
#[cfg(feature = "candle-cuda-compat")]
pub mod residual_add;
#[cfg(feature = "candle-cuda-compat")]
pub mod rms_norm;
#[cfg(feature = "candle-cuda-compat")]
pub mod rope;
#[cfg(all(feature = "candle-cuda-compat", not(target_os = "windows")))]
pub mod tp_decode;
pub mod vnext_ops;
mod vnext_replay;
pub mod vnext_runtime;
mod vnext_tool_correlation;
#[cfg(feature = "candle-cuda-compat")]
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
// extern "C" launcher supplied by the paged-attention native artifact.
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

// ────────────────────────────────────────────────────────────────────────
// Context
// ────────────────────────────────────────────────────────────────────────

/// Execution context for CudaBackend.
///
/// Owns the `CudaContext`, a dedicated `CudaStream`, a cuBLAS handle
/// bound to that stream, and a lazy cache of PTX modules. All kernels
/// launch on the instance's stream(s); completion observes the main stream
/// and any auxiliary MoE streams before releasing execution leases.
pub struct CudaState {
    configuration: ferrum_types::BackendContextConfig,
    fa2_shim: Option<fa2_ffi::Fa2Shim>,
    // Programs and in-flight leases retire before their captured resource closure.
    graph_cache: HashMap<u64, Arc<graph::GraphSlotRaw>>,
    graph_leases: graph::GraphLeases,
    // Replaced scratch remains owned until no cached or in-flight graph can
    // reference it and actual completion has been observed. Growth is cold.
    captured_storage: Vec<Box<dyn Send + Sync>>,
    pub ordinal: usize,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: Arc<CudaBlas>,
    blas_resources: BlasSlot,
    modules: HashMap<&'static str, Arc<CudaModule>>,
    decode_state: DecodeStateBufs,
    batched_scratch: BatchedScratchSlot,
    argmax_out: Option<CudaSlice<i32>>,
    qkv_scratch: qkv::QkvScratch,
    #[cfg(feature = "vllm-moe-marlin")]
    vllm_moe_c_tmp: Option<CudaSlice<f32>>,
    #[cfg(feature = "marlin")]
    marlin_gather_scratch: Option<quant::MarlinGatherScratch>,
    split_k_scratch: Option<paged::SplitKScratch>,
    #[cfg(feature = "vllm-paged-attn-v2")]
    paged_attention_scratch: Option<vllm_paged_attn::PagedAttnScratch>,
    pub use_dev_state: bool,
    pub capture_in_flight: bool,
    completion_observed: bool,
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

fn with_argmax_out<R>(
    stream: &Arc<CudaStream>,
    slot: &mut Option<CudaSlice<i32>>,
    capture_in_flight: bool,
    captured_storage: &mut Vec<Box<dyn Send + Sync>>,
    m: usize,
    body: impl FnOnce(&mut CudaSlice<i32>) -> R,
) -> R {
    if slot.as_ref().is_none_or(|buf| buf.len() < m) {
        assert!(
            !capture_in_flight,
            "argmax scratch must be grown before graph capture"
        );
        let capacity = m.max(64).next_power_of_two();
        let next = unsafe { stream.alloc::<i32>(capacity) }.expect("argmax_out alloc");
        if let Some(previous) = slot.replace(next) {
            captured_storage.push(Box::new(previous));
        }
    }
    body(slot.as_mut().expect("argmax scratch initialized"))
}

#[cfg(feature = "vllm-moe-marlin")]
pub fn with_vllm_moe_c_tmp<R>(
    stream: &Arc<CudaStream>,
    slot: &mut Option<CudaSlice<f32>>,
    capture_in_flight: bool,
    body: impl FnOnce(&mut CudaSlice<f32>) -> R,
) -> R {
    if slot.is_none() {
        assert!(
            !capture_in_flight,
            "MoE scratch must be initialized before graph capture"
        );
        *slot = Some(
            stream
                .alloc_zeros::<f32>(4 * 1024 * 1024)
                .expect("alloc vllm_moe_c_tmp"),
        );
    }
    body(slot.as_mut().expect("MoE scratch initialized"))
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
    /// 4 by default; the original context's `moe_streams` policy fixes
    /// its size (1 selects serial expert dispatch).
    pub fn moe_stream_pool(&mut self) -> &[Arc<CudaStream>] {
        if self.moe_streams.is_none() {
            let n = self.configuration.moe.moe_streams;
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
        let m = self
            .ctx
            .load_module(Ptx::from_src(ptx_src.to_owned()))
            .unwrap_or_else(|error| panic!("CudaBackend: load_module({key}): {error}"));
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

    fn supports_qk_norm_rope_batched_per_item() -> bool {
        true
    }

    fn supports_kv_cache_append_batched_per_cache() -> bool {
        true
    }

    fn supports_flash_attention_batched_per_cache() -> bool {
        true
    }

    // type GptqStore: removed in Phase C step 4e. GptqStoreCuda is
    // now a private (crate-internal) detail of CudaMarlinExpertStack.

    // ── Lifecycle ────────────────────────────────────────────────────────

    fn new_context(config: &ferrum_types::BackendContextConfig) -> Result<Self::Context> {
        config.validate()?;
        if config.collective.world_size != 1 {
            return Err(FerrumError::unsupported(
                "retained CUDA context has no multi-rank collective owner",
            ));
        }
        #[cfg(not(feature = "triton-kernels"))]
        if config.cuda.triton_int4 {
            return Err(FerrumError::unsupported(
                "triton_int4 requires the triton-kernels feature",
            ));
        }
        #[cfg(not(feature = "vllm-marlin"))]
        if config.cuda.vllm_marlin {
            return Err(FerrumError::unsupported(
                "vllm_marlin requires the vllm-marlin feature",
            ));
        }
        #[cfg(not(feature = "vllm-moe-marlin"))]
        if config.moe.vllm_moe {
            return Err(FerrumError::unsupported(
                "vllm_moe requires the vllm-moe-marlin feature",
            ));
        }
        #[cfg(not(feature = "fa2-source"))]
        if config.cuda.fa2_source {
            return Err(FerrumError::unsupported(
                "fa2_source requires the fa2-source feature",
            ));
        }
        let fa2_shim = if config.cuda.fa2_source {
            None
        } else {
            config
                .cuda
                .fa2_direct_ffi_shim
                .as_deref()
                .map(fa2_ffi::Fa2Shim::load)
                .transpose()?
        };
        let ordinal = config.device_ordinal;
        let stream = fresh_stream(ordinal)?;
        let ctx = stream.context().clone();
        let blas_resources = BlasSlot::new(&stream);
        let blas = blas_resources.blas.clone();
        let decode_state = DecodeStateBufs::new(&stream);
        let batched_scratch = BatchedScratchSlot::new(&stream);

        Ok(Self::Context {
            configuration: config.clone(),
            fa2_shim,
            ordinal,
            ctx,
            stream,
            blas,
            graph_cache: HashMap::new(),
            graph_leases: graph::GraphLeases::default(),
            captured_storage: Vec::new(),
            blas_resources,
            modules: HashMap::new(),
            decode_state,
            batched_scratch,
            argmax_out: None,
            qkv_scratch: qkv::QkvScratch::default(),
            #[cfg(feature = "vllm-moe-marlin")]
            vllm_moe_c_tmp: None,
            #[cfg(feature = "marlin")]
            marlin_gather_scratch: None,
            split_k_scratch: None,
            #[cfg(feature = "vllm-paged-attn-v2")]
            paged_attention_scratch: None,
            use_dev_state: false,
            capture_in_flight: false,
            completion_observed: false,
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
        })
    }

    fn context_config(ctx: &Self::Context) -> &ferrum_types::BackendContextConfig {
        &ctx.configuration
    }

    fn begin_execution(ctx: &mut Self::Context) -> Result<()> {
        if !ctx.completion_observed {
            Self::observe_completion(ctx).into_result()?;
        } else {
            // A previous observation may have proven device completion while
            // graph cleanup failed. Retry that original cleanup before reuse.
            ctx.graph_leases.retire_completed(&ctx.graph_cache)?;
        }
        ctx.completion_observed = false;
        ctx.use_dev_state = false;
        Ok(())
    }

    fn retire_context(ctx: &mut Self::Context) -> Result<()> {
        Self::reset_all_graphs(ctx);
        // Final context retirement also observes ordinary eager work when
        // there was no captured program; the storage-growth hook need not.
        Self::observe_completion(ctx).into_result()
    }

    fn retain_buffer_until_completion(ctx: &mut Self::Context, buffer: Self::Buffer) {
        ctx.captured_storage.push(Box::new(buffer));
    }

    fn from_weight_bytes_in_context(
        ctx: &mut Self::Context,
        raw: &[u8],
        src_dtype: crate::backend::SrcDtype,
    ) -> Self::Buffer {
        Self::from_slice_in_context(ctx, &src_dtype.to_f32_vec(raw))
    }

    fn supports_device_ordinal() -> bool {
        true
    }

    /// Phase D step 2+3 unified typed allocator. Replaces alloc_u32 and
    /// the per-dtype family (alloc_typed_i32 / etc. were never needed).
    fn alloc_typed(dtype: crate::backend::Dtype, n: usize) -> Self::Buffer {
        use crate::backend::{CudaBuf, Dtype};
        let n = n.max(1);
        with_standalone_allocation(|stream| match dtype {
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

    fn alloc_typed_in_context(
        ctx: &mut Self::Context,
        dtype: crate::backend::Dtype,
        n: usize,
    ) -> Self::Buffer {
        use crate::backend::{CudaBuf, Dtype};
        let n = n.max(1);
        with_instance_allocation(ctx, |stream| match dtype {
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
        with_standalone_allocation(|stream| match T::DTYPE {
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

    fn from_slice_typed_in_context<T: crate::backend::HostDtype>(
        ctx: &mut Self::Context,
        data: &[T],
    ) -> Self::Buffer {
        use crate::backend::{CudaBuf, Dtype};
        with_instance_allocation(ctx, |stream| match T::DTYPE {
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
        Self::observe_completion(ctx)
            .into_result()
            .expect("CudaBackend: instance synchronization");
    }

    fn try_sync(ctx: &mut Self::Context) -> ferrum_types::Result<()> {
        Self::observe_completion(ctx).into_result()
    }

    fn observe_completion(ctx: &mut Self::Context) -> super::BackendCompletion {
        use super::BackendCompletion;
        if ctx.capture_in_flight {
            return BackendCompletion::Unresolved(ferrum_types::FerrumError::backend(
                "CUDA completion is unresolved while graph capture is active",
            ));
        }
        // An encode error can bypass the usual pool -> main-stream event
        // joins. Observe the actual pool as well as the main stream before
        // permitting any context-owned scratch or host staging to retire.
        let mut failure = None;
        for stream in ctx
            .moe_streams
            .iter()
            .flatten()
            .chain(std::iter::once(&ctx.stream))
        {
            if let Err(error) = stream.synchronize() {
                failure.get_or_insert_with(|| {
                    ferrum_types::FerrumError::backend(format!(
                        "CUDA execution stream synchronization failed: {error}"
                    ))
                });
            }
        }
        match failure {
            Some(error) => BackendCompletion::Unresolved(error),
            None => {
                ctx.completion_observed = true;
                if let Err(error) = ctx.graph_leases.retire_completed(&ctx.graph_cache) {
                    // The device is terminal. This is a separate cleanup
                    // failure, with every unfinished handle still retained.
                    return BackendCompletion::Failed(error);
                }
                if ctx.graph_cache.is_empty() && ctx.graph_leases.is_empty() {
                    ctx.captured_storage.clear();
                }
                BackendCompletion::Succeeded
            }
        }
    }

    fn graph_capture_in_flight(ctx: &Self::Context) -> bool {
        ctx.capture_in_flight
    }

    fn alloc(len: usize) -> Self::Buffer {
        with_standalone_allocation(|stream| {
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

    fn alloc_in_context(ctx: &mut Self::Context, len: usize) -> Self::Buffer {
        with_instance_allocation(ctx, |stream| {
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
        with_standalone_allocation(|stream| {
            crate::backend::CudaBuf::from_f16(stream.clone_htod(&host).expect("cuda htod"))
        })
    }

    fn from_slice_in_context(ctx: &mut Self::Context, data: &[f32]) -> Self::Buffer {
        let host: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
        with_instance_allocation(ctx, |stream| {
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

    fn f32_to_activation(
        ctx: &mut Self::Context,
        input_f32: &Self::Buffer,
        out: &mut Self::Buffer,
        len: usize,
    ) {
        match (input_f32.dtype(), out.dtype()) {
            (crate::backend::Dtype::F32, crate::backend::Dtype::F16) => {
                let func = ctx.func("sandwich_norm", ptx::SANDWICH_NORM, "f32_to_activation_f16");
                let n = len as i32;
                let block = 256u32;
                let grid = ((len as u32) + block - 1) / block;
                let stream = ctx.stream.clone();
                let mut b = stream.launch_builder(&func);
                b.arg(input_f32);
                b.arg(out);
                b.arg(&n);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .expect("f32_to_activation_f16 launch");
            }
            (crate::backend::Dtype::F32, crate::backend::Dtype::F32) => {
                Self::copy_slice(ctx, input_f32, 0, out, 0, len);
            }
            (src, dst) => panic!(
                "CudaBackend::f32_to_activation unsupported dtypes input={} out={}",
                src.name(),
                dst.name()
            ),
        }
    }

    fn supports_device_f32_residual_shadow() -> bool {
        true
    }

    fn supports_qwen35_indexed_recurrent_state() -> bool {
        true
    }

    fn qwen35_indexed_recurrent_state_dtype() -> crate::backend::Dtype {
        crate::backend::Dtype::F16
    }

    fn supports_qwen35_packed_gdn_decode_prepare() -> bool {
        true
    }

    fn supports_qwen35_packed_gdn_prefill_prepare() -> bool {
        true
    }

    fn supports_qwen35_packed_gdn_recurrent_decode() -> bool {
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

    fn activation_add_to_f32_shadow(
        ctx: &mut Self::Context,
        src: &Self::Buffer,
        residual_f32: &mut Self::Buffer,
        scratch_f32: &mut Self::Buffer,
        len: usize,
    ) {
        if len == 0 {
            return;
        }
        assert_eq!(
            residual_f32.dtype(),
            crate::backend::Dtype::F32,
            "CudaBackend::activation_add_to_f32_shadow residual must be F32, got {}",
            residual_f32.dtype().name()
        );
        match (src.dtype(), residual_f32.dtype()) {
            (crate::backend::Dtype::F16, crate::backend::Dtype::F32) => {
                let func = ctx.func(
                    "sandwich_norm",
                    ptx::SANDWICH_NORM,
                    "activation_add_to_f32_shadow_f16",
                );
                let n_i32 = len as i32;
                let block = 256u32;
                let grid = ((len as u32) + block - 1) / block;
                let stream = ctx.stream.clone();
                let mut b = stream.launch_builder(&func);
                b.arg(src);
                b.arg(residual_f32);
                b.arg(&n_i32);
                unsafe {
                    b.launch(LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (block, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .expect("activation_add_to_f32_shadow launch");
            }
            (crate::backend::Dtype::F32, crate::backend::Dtype::F32) => {
                Self::add_inplace(ctx, residual_f32, src, len);
            }
            _ => {
                Self::activation_to_f32_shadow(ctx, src, scratch_f32, len);
                Self::add_inplace(ctx, residual_f32, scratch_f32, len);
            }
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
        // External callers must finish any use on another context first.
        // This compatibility readback uses the actual allocation stream.
        let stream = match buf.dtype() {
            crate::backend::Dtype::F16 => buf.as_f16().stream(),
            crate::backend::Dtype::F32 => buf.as_f32().stream(),
            other => panic!("CUDA readback unsupported dtype {}", other.name()),
        };
        (|stream: &Arc<CudaStream>| {
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
        })(stream)
    }

    fn to_vec_in_context(ctx: &mut Self::Context, buf: &Self::Buffer, len: usize) -> Vec<f32> {
        with_instance_readback(ctx, |stream| {
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
        // Reuse this context's output allocation across invocations.
        // Captured predecessors survive growth until graph retirement.
        let host = with_argmax_out(
            &stream,
            &mut ctx.argmax_out,
            ctx.capture_in_flight,
            &mut ctx.captured_storage,
            m,
            |out_dev| -> Result<Vec<i32>> {
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
            },
        )?;
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
        let host = with_argmax_out(
            &stream,
            &mut ctx.argmax_out,
            ctx.capture_in_flight,
            &mut ctx.captured_storage,
            m,
            |out_dev| -> Result<Vec<i32>> {
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
            },
        )?;
        Ok(host.into_iter().map(|x| x as u32).collect())
    }

    fn argmax_rows_f16_sparse_repetition_penalty(
        ctx: &mut Self::Context,
        logits: &mut Self::Buffer,
        valid_token_mask: Option<(&Self::Buffer, usize)>,
        row_offsets: &Self::Buffer,
        token_ids: &Self::Buffer,
        repetition_penalties: &Self::Buffer,
        total_token_ids: usize,
        m: usize,
        n: usize,
    ) -> Result<Vec<u32>> {
        if total_token_ids > 0 {
            let func = ctx.func(
                "argmax_rows",
                ptx::ARGMAX_ROWS,
                "apply_repetition_penalties_sparse_f16",
            );
            let n_i32 = n as i32;
            let total_i32 = total_token_ids as i32;
            let mut b = ctx.stream.launch_builder(&func);
            b.arg(&mut *logits);
            b.arg(&n_i32);
            b.arg(row_offsets);
            b.arg(token_ids);
            b.arg(repetition_penalties);
            b.arg(&total_i32);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (m as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| {
                FerrumError::internal(format!("apply_repetition_penalties_sparse_f16 launch: {e}"))
            })?;
        }

        match valid_token_mask {
            Some((mask, mask_len)) => {
                Self::argmax_rows_f16_masked(ctx, logits, mask, mask_len, m, n)
            }
            None => Self::argmax_rows_f16(ctx, logits, m, n),
        }
    }

    fn supports_argmax_rows_f16_sparse_repetition_penalty() -> bool {
        true
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

        // The instance BLAS handle uses CUBLAS_POINTER_MODE_DEVICE;
        // its original scalar allocations remain stable through capture.
        let (a_ptr, _rec_a) = b.as_f16().device_ptr(&ctx.stream); // cuBLAS arg "A" = weight = our `b`
        let (b_ptr, _rec_b) = a.as_f16().device_ptr(&ctx.stream); // cuBLAS arg "B" = input = our `a`
        let (c_ptr, _rec_c) = out.as_f16_mut().device_ptr_mut(&ctx.stream);
        with_blas_scalars(&ctx.blas_resources, |alpha_f32, beta_f32| {
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
            let env_cap = ctx
                .configuration
                .cuda
                .max_kv
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
            // Borrow the original context's decode state for the builder.
            let dec_guard = use_dyn.then_some(&ctx.decode_state);
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
                let bufs = dec_guard.as_ref().unwrap();
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
            return;
        }
        let func = ctx.func(
            "flash_attn_full",
            ptx::FLASH_ATTN_FULL,
            "flash_attn_full_f16",
        );
        if cfg.head_dim > 256 {
            panic!(
                "flash_attn_full_f16 supports head_dim <= 256, got {}",
                cfg.head_dim
            );
        }
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
    fn recurrent_gated_delta_rule_batch_f32(
        ctx: &mut Self::Context,
        query: &Self::Buffer,
        key: &Self::Buffer,
        value: &Self::Buffer,
        g: &Self::Buffer,
        beta: &Self::Buffer,
        initial_states: &Self::Buffer,
        out: &mut Self::Buffer,
        final_states: &mut Self::Buffer,
        batch: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        use_qk_l2norm: bool,
        scale: f32,
    ) -> Result<()> {
        gated_delta_rule::recurrent_gated_delta_rule_batch_f32(
            ctx,
            query,
            key,
            value,
            g,
            beta,
            initial_states,
            out,
            final_states,
            batch,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            use_qk_l2norm,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn recurrent_gated_delta_rule_batch_indexed_f32(
        ctx: &mut Self::Context,
        query: &Self::Buffer,
        key: &Self::Buffer,
        value: &Self::Buffer,
        g: &Self::Buffer,
        beta: &Self::Buffer,
        state_slots: &mut Self::Buffer,
        slot_indices: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        max_slots: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        use_qk_l2norm: bool,
        scale: f32,
    ) -> Result<()> {
        gated_delta_rule::recurrent_gated_delta_rule_batch_indexed_f32(
            ctx,
            query,
            key,
            value,
            g,
            beta,
            state_slots,
            slot_indices,
            out,
            batch,
            max_slots,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            use_qk_l2norm,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn recurrent_gated_delta_rule_batch_indexed_packed_f32(
        ctx: &mut Self::Context,
        mixed_qkv: &Self::Buffer,
        ba_raw: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        state_slots: &mut Self::Buffer,
        slot_indices: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        max_slots: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        scale: f32,
    ) -> Result<()> {
        gated_delta_rule::recurrent_gated_delta_rule_batch_indexed_packed_f32(
            ctx,
            mixed_qkv,
            ba_raw,
            a_log,
            dt_bias,
            state_slots,
            slot_indices,
            out,
            batch,
            max_slots,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            scale,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn recurrent_gated_delta_rule_varlen_f32(
        ctx: &mut Self::Context,
        query: &Self::Buffer,
        key: &Self::Buffer,
        value: &Self::Buffer,
        g: &Self::Buffer,
        beta: &Self::Buffer,
        initial_states: &Self::Buffer,
        cu_seqlens: &Self::Buffer,
        out: &mut Self::Buffer,
        final_states: &mut Self::Buffer,
        batch: usize,
        total_tokens: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        use_qk_l2norm: bool,
        scale: f32,
    ) -> Result<()> {
        gated_delta_rule::recurrent_gated_delta_rule_varlen_f32(
            ctx,
            query,
            key,
            value,
            g,
            beta,
            initial_states,
            cu_seqlens,
            out,
            final_states,
            batch,
            total_tokens,
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
    fn linear_attention_prepare_varlen_f32(
        ctx: &mut Self::Context,
        mixed_qkv_raw: &Self::Buffer,
        conv_weight: &Self::Buffer,
        initial_conv_states: &Self::Buffer,
        a_raw: &Self::Buffer,
        b_raw: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        cu_seqlens: &Self::Buffer,
        token_seq_indices: &Self::Buffer,
        query: &mut Self::Buffer,
        key: &mut Self::Buffer,
        value: &mut Self::Buffer,
        g: &mut Self::Buffer,
        beta: &mut Self::Buffer,
        final_conv_states: &mut Self::Buffer,
        batch: usize,
        total_tokens: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        apply_qk_l2norm: bool,
    ) -> Result<()> {
        linear_attention::linear_attention_prepare_varlen_f32(
            ctx,
            mixed_qkv_raw,
            conv_weight,
            initial_conv_states,
            a_raw,
            b_raw,
            a_log,
            dt_bias,
            cu_seqlens,
            token_seq_indices,
            query,
            key,
            value,
            g,
            beta,
            final_conv_states,
            batch,
            total_tokens,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
            apply_qk_l2norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_prepare_varlen_packed_qkvz_ba_f32(
        ctx: &mut Self::Context,
        mixed_qkvz_raw: &Self::Buffer,
        ba_raw: &Self::Buffer,
        conv_weight: &Self::Buffer,
        initial_conv_states: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        cu_seqlens: &Self::Buffer,
        token_seq_indices: &Self::Buffer,
        query: &mut Self::Buffer,
        key: &mut Self::Buffer,
        value: &mut Self::Buffer,
        z: &mut Self::Buffer,
        g: &mut Self::Buffer,
        beta: &mut Self::Buffer,
        final_conv_states: &mut Self::Buffer,
        batch: usize,
        total_tokens: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        apply_qk_l2norm: bool,
    ) -> Result<()> {
        linear_attention::linear_attention_prepare_varlen_packed_qkvz_ba_f32(
            ctx,
            mixed_qkvz_raw,
            ba_raw,
            conv_weight,
            initial_conv_states,
            a_log,
            dt_bias,
            cu_seqlens,
            token_seq_indices,
            query,
            key,
            value,
            z,
            g,
            beta,
            final_conv_states,
            batch,
            total_tokens,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
            apply_qk_l2norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_decode_prepare_f32(
        ctx: &mut Self::Context,
        mixed_qkv_raw: &Self::Buffer,
        conv_weight: &Self::Buffer,
        conv_state: &Self::Buffer,
        a_raw: &Self::Buffer,
        b_raw: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        query: &mut Self::Buffer,
        key: &mut Self::Buffer,
        value: &mut Self::Buffer,
        g: &mut Self::Buffer,
        beta: &mut Self::Buffer,
        next_conv_state: &mut Self::Buffer,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        apply_qk_l2norm: bool,
    ) -> Result<()> {
        linear_attention::linear_attention_decode_prepare_f32(
            ctx,
            mixed_qkv_raw,
            conv_weight,
            conv_state,
            a_raw,
            b_raw,
            a_log,
            dt_bias,
            query,
            key,
            value,
            g,
            beta,
            next_conv_state,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
            apply_qk_l2norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_decode_prepare_batch_f32(
        ctx: &mut Self::Context,
        mixed_qkv_raw: &Self::Buffer,
        conv_weight: &Self::Buffer,
        conv_states: &Self::Buffer,
        a_raw: &Self::Buffer,
        b_raw: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        query: &mut Self::Buffer,
        key: &mut Self::Buffer,
        value: &mut Self::Buffer,
        g: &mut Self::Buffer,
        beta: &mut Self::Buffer,
        next_conv_states: &mut Self::Buffer,
        batch: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        apply_qk_l2norm: bool,
    ) -> Result<()> {
        linear_attention::linear_attention_decode_prepare_batch_f32(
            ctx,
            mixed_qkv_raw,
            conv_weight,
            conv_states,
            a_raw,
            b_raw,
            a_log,
            dt_bias,
            query,
            key,
            value,
            g,
            beta,
            next_conv_states,
            batch,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
            apply_qk_l2norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_decode_prepare_batch_indexed_f32(
        ctx: &mut Self::Context,
        mixed_qkv_raw: &Self::Buffer,
        conv_weight: &Self::Buffer,
        conv_state_slots: &mut Self::Buffer,
        slot_indices: &Self::Buffer,
        a_raw: &Self::Buffer,
        b_raw: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        query: &mut Self::Buffer,
        key: &mut Self::Buffer,
        value: &mut Self::Buffer,
        g: &mut Self::Buffer,
        beta: &mut Self::Buffer,
        batch: usize,
        max_slots: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        apply_qk_l2norm: bool,
    ) -> Result<()> {
        linear_attention::linear_attention_decode_prepare_batch_indexed_f32(
            ctx,
            mixed_qkv_raw,
            conv_weight,
            conv_state_slots,
            slot_indices,
            a_raw,
            b_raw,
            a_log,
            dt_bias,
            query,
            key,
            value,
            g,
            beta,
            batch,
            max_slots,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
            apply_qk_l2norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32(
        ctx: &mut Self::Context,
        mixed_qkvz_raw: &Self::Buffer,
        ba_raw: &Self::Buffer,
        conv_weight: &Self::Buffer,
        conv_state_slots: &mut Self::Buffer,
        slot_indices: &Self::Buffer,
        a_log: &Self::Buffer,
        dt_bias: &Self::Buffer,
        query: &mut Self::Buffer,
        key: &mut Self::Buffer,
        value: &mut Self::Buffer,
        z: &mut Self::Buffer,
        g: &mut Self::Buffer,
        beta: &mut Self::Buffer,
        batch: usize,
        max_slots: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
        apply_qk_l2norm: bool,
    ) -> Result<()> {
        linear_attention::linear_attention_decode_prepare_batch_indexed_packed_qkvz_ba_f32(
            ctx,
            mixed_qkvz_raw,
            ba_raw,
            conv_weight,
            conv_state_slots,
            slot_indices,
            a_log,
            dt_bias,
            query,
            key,
            value,
            z,
            g,
            beta,
            batch,
            max_slots,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
            apply_qk_l2norm,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f32(
        ctx: &mut Self::Context,
        mixed_qkvz_raw: &Self::Buffer,
        conv_weight: &Self::Buffer,
        conv_state_slots: &mut Self::Buffer,
        slot_indices: &Self::Buffer,
        mixed_qkv: &mut Self::Buffer,
        z: &mut Self::Buffer,
        batch: usize,
        max_slots: usize,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_kernel: usize,
    ) -> Result<()> {
        linear_attention::linear_attention_decode_prepare_batch_indexed_packed_qkvz_to_mixed_f32(
            ctx,
            mixed_qkvz_raw,
            conv_weight,
            conv_state_slots,
            slot_indices,
            mixed_qkv,
            z,
            batch,
            max_slots,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_kernel,
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
        if len == 0 {
            return;
        }
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
            (crate::backend::Dtype::F16, crate::backend::Dtype::F32) => {
                let func = ctx.func("sandwich_norm", ptx::SANDWICH_NORM, "cast_f16_to_f32_slice");
                let src_offset_i32 = src_offset as i32;
                let dst_offset_i32 = dst_offset as i32;
                let n_i32 = len as i32;
                let block = 256u32;
                let grid = ((len as u32) + block - 1) / block;
                let stream = ctx.stream.clone();
                let mut builder = stream.launch_builder(&func);
                builder.arg(src.as_f16());
                builder.arg(dst.as_f32_mut());
                builder.arg(&src_offset_i32);
                builder.arg(&dst_offset_i32);
                builder.arg(&n_i32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .expect("copy_slice f16 to f32 cast");
                }
            }
            (crate::backend::Dtype::F32, crate::backend::Dtype::F16) => {
                let func = ctx.func("sandwich_norm", ptx::SANDWICH_NORM, "cast_f32_to_f16_slice");
                let src_offset_i32 = src_offset as i32;
                let dst_offset_i32 = dst_offset as i32;
                let n_i32 = len as i32;
                let block = 256u32;
                let grid = ((len as u32) + block - 1) / block;
                let stream = ctx.stream.clone();
                let mut builder = stream.launch_builder(&func);
                builder.arg(src.as_f32());
                builder.arg(dst.as_f16_mut());
                builder.arg(&src_offset_i32);
                builder.arg(&dst_offset_i32);
                builder.arg(&n_i32);
                unsafe {
                    builder
                        .launch(LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (block, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .expect("copy_slice f32 to f16 cast");
                }
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
            let bufs = &ctx.decode_state;
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
        qkv::launch_split_qkv(&ctx.stream, &func, qkv, q, k, v, tokens, q_dim, kv_dim);
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
        // Original context's stable host/device pointer-table regions.
        // Each captured call site has a distinct slot; the context and
        // its tables remain owned until actual graph retirement.
        let m_i32 = m as i32;
        let nkv_i32 = nkv as i32;
        let hd_i32 = hd as i32;
        let capacity_i32 = capacity as i32;
        let per_item = nkv * hd;
        let block_dim = 256u32;
        let grid_x = (per_item as u32 + block_dim - 1) / block_dim;
        with_batched_scratch_mut(&mut ctx.batched_scratch, |slot_g| {
            for i in 0..m {
                let (cp, _) = caches[i].as_f16().device_ptr(&stream);
                slot_g.host_cache_ptrs[host_start + i] = cp;
            }
            // Async memcpy host_slice → device per-slot region. Recorded
            // into the captured graph when capture is in flight; both
            // endpoints are stable addresses owned by this context.
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
        with_batched_scratch_mut(&mut ctx.batched_scratch, |slot_g| {
            for i in 0..m {
                let (kp, _) = k_caches[i].as_f16().device_ptr(&stream);
                let (vp, _) = v_caches[i].as_f16().device_ptr(&stream);
                slot_g.host_k_ptrs[host_start + i] = kp;
                slot_g.host_v_ptrs[host_start + i] = vp;
            }
            // Two captured memcpys, each into its own per-slot region of
            // instance device scratch. The persistent original context
            // owns both host arrays and device arrays through retirement.
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
        qkv::launch_qk_norm_rope(
            &ctx.stream,
            &func,
            use_dyn.then_some(&ctx.decode_state.pos),
            input,
            norm_w,
            cos,
            sin,
            output,
            tokens,
            heads,
            head_dim,
            pos_offset,
            eps,
            mode,
        );
    }

    fn qk_norm_rope_partial(
        ctx: &mut Self::Context,
        input: &Self::Buffer,
        norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        output: &mut Self::Buffer,
        tokens: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        input_stride: usize,
        input_offset: usize,
        input_head_stride: usize,
        pos_offset: usize,
        eps: f32,
        mode: i32,
    ) -> Result<()> {
        if rope_dim == head_dim
            && input_stride == heads * head_dim
            && input_offset == 0
            && input_head_stride == head_dim
            && mode != 3
        {
            Self::qk_norm_rope(
                ctx, input, norm_w, cos, sin, output, tokens, heads, head_dim, pos_offset, eps,
                mode,
            );
            return Ok(());
        }

        if tokens == 0 || heads == 0 || head_dim == 0 || rope_dim == 0 {
            return Err(FerrumError::model(format!(
                "qk_norm_rope_partial shape must be positive, got tokens={tokens} heads={heads} head_dim={head_dim} rope_dim={rope_dim}"
            )));
        }
        if rope_dim > head_dim || rope_dim % 2 != 0 {
            return Err(FerrumError::model(format!(
                "qk_norm_rope_partial rope_dim {rope_dim} must be even and <= head_dim {head_dim}"
            )));
        }
        if input_head_stride == 0 {
            return Err(FerrumError::model(
                "qk_norm_rope_partial input_head_stride must be positive",
            ));
        }
        let required_width = input_offset + (heads - 1) * input_head_stride + head_dim;
        if input_stride < required_width {
            return Err(FerrumError::model(format!(
                "qk_norm_rope_partial input_stride {input_stride} is too small for offset {input_offset}, heads {heads}, head_dim {head_dim}, input_head_stride {input_head_stride}"
            )));
        }

        let func = ctx.func(
            "qk_norm_rope_partial",
            ptx::QK_NORM_ROPE,
            "qk_norm_rope_partial_transpose_f16",
        );
        let tokens_i32 = tokens as i32;
        let heads_i32 = heads as i32;
        let head_dim_i32 = head_dim as i32;
        let rope_dim_i32 = rope_dim as i32;
        let input_stride_i32 = input_stride as i32;
        let input_offset_i32 = input_offset as i32;
        let input_head_stride_i32 = input_head_stride as i32;
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
        b.arg(&rope_dim_i32);
        b.arg(&input_stride_i32);
        b.arg(&input_offset_i32);
        b.arg(&input_head_stride_i32);
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
        .map_err(|e| FerrumError::model(format!("qk_norm_rope_partial: {e}")))?;
        Ok(())
    }

    fn qwen35_apply_attention_gate(
        ctx: &mut Self::Context,
        context: &mut Self::Buffer,
        query_raw: &Self::Buffer,
        tokens: usize,
        q_total: usize,
        q_proj_total: usize,
        head_dim: usize,
    ) -> Result<()> {
        if head_dim == 0 || q_total % head_dim != 0 {
            return Err(FerrumError::model(format!(
                "qwen35_apply_attention_gate q_total {q_total} must be divisible by head_dim {head_dim}"
            )));
        }
        let heads = q_total / head_dim;
        if q_proj_total < heads * 2 * head_dim {
            return Err(FerrumError::model(format!(
                "qwen35_apply_attention_gate q_proj_total {q_proj_total} must include per-head query and gate slices for q_total {q_total}, head_dim {head_dim}"
            )));
        }
        if tokens == 0 || q_total == 0 {
            return Ok(());
        }

        let func = ctx.func(
            "qk_norm_rope_gate",
            ptx::QK_NORM_ROPE,
            "qwen35_apply_attention_gate_f16",
        );
        let tokens_i32 = tokens as i32;
        let q_total_i32 = q_total as i32;
        let q_proj_total_i32 = q_proj_total as i32;
        let head_dim_i32 = head_dim as i32;
        let total = tokens * q_total;
        let block = 256u32;
        let grid = ((total as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(context);
        b.arg(query_raw);
        b.arg(&tokens_i32);
        b.arg(&q_total_i32);
        b.arg(&q_proj_total_i32);
        b.arg(&head_dim_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| FerrumError::model(format!("qwen35_apply_attention_gate: {e}")))?;
        Ok(())
    }

    fn qwen35_apply_token_gate(
        ctx: &mut Self::Context,
        values: &mut Self::Buffer,
        gate: &Self::Buffer,
        tokens: usize,
        hidden_size: usize,
    ) -> Result<()> {
        if tokens == 0 || hidden_size == 0 {
            return Ok(());
        }

        let func = ctx.func(
            "qk_norm_rope_gate",
            ptx::QK_NORM_ROPE,
            "qwen35_apply_token_gate_f16",
        );
        let tokens_i32 = tokens as i32;
        let hidden_i32 = hidden_size as i32;
        let total = tokens * hidden_size;
        let block = 256u32;
        let grid = ((total as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(values);
        b.arg(gate);
        b.arg(&tokens_i32);
        b.arg(&hidden_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| FerrumError::model(format!("qwen35_apply_token_gate: {e}")))?;
        Ok(())
    }

    fn qwen35_apply_token_gate_and_add_inplace(
        ctx: &mut Self::Context,
        dst: &mut Self::Buffer,
        values: &mut Self::Buffer,
        gate: &Self::Buffer,
        tokens: usize,
        hidden_size: usize,
    ) -> Result<()> {
        if tokens == 0 || hidden_size == 0 {
            return Ok(());
        }
        let expected = tokens * hidden_size;
        if dst.len() < expected || values.len() < expected || gate.len() < tokens {
            return Err(FerrumError::model(format!(
                "qwen35_apply_token_gate_and_add_inplace buffer too small: dst={} values={} gate={} expected={} gate_expected={}",
                dst.len(),
                values.len(),
                gate.len(),
                expected,
                tokens
            )));
        }
        if dst.dtype() != values.dtype() || dst.dtype() != gate.dtype() {
            return Err(FerrumError::model(format!(
                "qwen35_apply_token_gate_and_add_inplace dtype mismatch: dst={} values={} gate={}",
                dst.dtype().name(),
                values.dtype().name(),
                gate.dtype().name()
            )));
        }

        let func_name = match dst.dtype() {
            crate::backend::Dtype::F16 => "qwen35_apply_token_gate_and_add_inplace_f16",
            crate::backend::Dtype::F32 => "qwen35_apply_token_gate_and_add_inplace_f32",
            dtype => {
                return Err(FerrumError::model(format!(
                    "qwen35_apply_token_gate_and_add_inplace unsupported dtype {}",
                    dtype.name()
                )))
            }
        };
        let func = ctx.func("qk_norm_rope_gate", ptx::QK_NORM_ROPE, func_name);
        let tokens_i32 = tokens as i32;
        let hidden_i32 = hidden_size as i32;
        let block = 256u32;
        let grid = ((expected as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(dst);
        b.arg(values);
        b.arg(gate);
        b.arg(&tokens_i32);
        b.arg(&hidden_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| FerrumError::model(format!("qwen35_apply_token_gate_and_add_inplace: {e}")))?;
        Ok(())
    }

    fn qwen35_interleave_gate_up(
        ctx: &mut Self::Context,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        out: &mut Self::Buffer,
        tokens: usize,
        intermediate: usize,
    ) -> Result<()> {
        if tokens == 0 || intermediate == 0 {
            return Ok(());
        }
        let expected = tokens * intermediate;
        if gate.len() < expected || up.len() < expected || out.len() < 2 * expected {
            return Err(FerrumError::model(format!(
                "qwen35_interleave_gate_up buffer too small: gate={} up={} out={} expected={} out_expected={}",
                gate.len(),
                up.len(),
                out.len(),
                expected,
                2 * expected
            )));
        }
        if gate.dtype() != up.dtype() || gate.dtype() != out.dtype() {
            return Err(FerrumError::model(format!(
                "qwen35_interleave_gate_up dtype mismatch: gate={} up={} out={}",
                gate.dtype().name(),
                up.dtype().name(),
                out.dtype().name()
            )));
        }

        let func_name = match gate.dtype() {
            crate::backend::Dtype::F16 => "qwen35_interleave_gate_up_f16",
            crate::backend::Dtype::F32 => "qwen35_interleave_gate_up_f32",
            dtype => {
                return Err(FerrumError::model(format!(
                    "qwen35_interleave_gate_up unsupported dtype {}",
                    dtype.name()
                )))
            }
        };
        let func = ctx.func("qk_norm_rope_gate", ptx::QK_NORM_ROPE, func_name);
        let tokens_i32 = tokens as i32;
        let intermediate_i32 = intermediate as i32;
        let block = 256u32;
        let grid = ((expected as u32) + block - 1) / block;
        let stream = ctx.stream.clone();
        let mut b = stream.launch_builder(&func);
        b.arg(gate);
        b.arg(up);
        b.arg(out);
        b.arg(&tokens_i32);
        b.arg(&intermediate_i32);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| FerrumError::model(format!("qwen35_interleave_gate_up: {e}")))?;
        Ok(())
    }

    /// Split QKV + qk-norm + RoPE into FP16 head-major scratch buffers.
    /// Implemented as a chain over the existing primitives: `split_qkv` →
    /// 3× `qk_norm_rope` (Q/K with their respective norms; V with mode=0).
    /// Used by contiguous FP16 KV and the INT8 paged-write path. Token-major
    /// temporaries belong to the original context, so eager warmup prepares
    /// stable storage which capture and replay reuse without allocation.
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
        qkv::split_qkv_norm_rope(
            ctx, qkv, q_norm_w, k_norm_w, cos, sin, q_out, k_out, v_out, tokens, q_heads, kv_heads,
            head_dim, pos_offset, eps, qk_mode,
        )
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
        let dec_guard = use_dyn.then_some(&ctx.decode_state);

        // K half.
        {
            let mut b = stream.launch_builder(&func);
            b.arg(cache_k);
            b.arg(new_k_head_major);
            b.arg(&nkv_i32);
            b.arg(&hd_i32);
            if use_dyn {
                let bufs = dec_guard.as_ref().unwrap();
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
                let bufs = dec_guard.as_ref().unwrap();
                b.arg(&bufs.pos);
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

    /// Inverse of `transpose_head_to_token`. Used by the CUDA paged
    /// attention wrapper to convert paged_varlen_attention's token-major
    /// output back to the wrapper's head-major buffer.
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
// Explicit instance allocation and external context-free constructors
// ────────────────────────────────────────────────────────────────────────
//
// Context-free compatibility helpers create their own allocation stream.
// Retained model production uses the explicit instance context APIs.
fn with_instance_allocation<R>(ctx: &mut CudaState, body: impl FnOnce(&Arc<CudaStream>) -> R) -> R {
    assert!(
        !ctx.capture_in_flight,
        "CUDA allocation must be prepared before graph capture"
    );
    ctx.completion_observed = false;
    body(&ctx.stream)
}

fn with_instance_readback<R>(ctx: &mut CudaState, body: impl FnOnce(&Arc<CudaStream>) -> R) -> R {
    assert!(
        !ctx.capture_in_flight,
        "CUDA readback cannot complete an active capture"
    );
    body(&ctx.stream)
}

fn fresh_stream(ordinal: usize) -> Result<Arc<CudaStream>> {
    let context = CudaContext::new(ordinal)
        .map_err(|error| FerrumError::backend(format!("CudaBackend context {ordinal}: {error}")))?;
    unsafe {
        context.disable_event_tracking();
    }
    context
        .new_stream()
        .map_err(|error| FerrumError::backend(format!("CudaBackend new instance stream: {error}")))
}

fn standalone_allocation_stream() -> Arc<CudaStream> {
    fresh_stream(ferrum_types::BackendContextConfig::default().device_ordinal)
        .expect("standalone default CUDA stream")
}

fn with_standalone_allocation<R>(f: impl FnOnce(&Arc<CudaStream>) -> R) -> R {
    let stream = standalone_allocation_stream();
    let value = f(&stream);
    // No destination stream is supplied by the compatibility constructor.
    // Complete initialization before returning; explicit APIs avoid this.
    stream
        .synchronize()
        .expect("context-free CUDA allocation initialization");
    value
}

pub struct DecodeStateBufs {
    pub token: CudaSlice<u32>,
    pub pos: CudaSlice<i32>,
    pub kv: CudaSlice<i32>,
}
impl DecodeStateBufs {
    fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            token: unsafe { stream.alloc::<u32>(1) }.expect("token state alloc"),
            pos: unsafe { stream.alloc::<i32>(1) }.expect("position state alloc"),
            kv: unsafe { stream.alloc::<i32>(1) }.expect("KV state alloc"),
        }
    }
}

struct BlasSlot {
    blas: Arc<CudaBlas>,
    _workspace: CudaSlice<u8>,
    alpha_f32: CudaSlice<f32>,
    beta_f32: CudaSlice<f32>,
}
impl BlasSlot {
    fn new(stream: &Arc<CudaStream>) -> Self {
        const WS_BYTES: usize = 32 * 1024 * 1024;
        let blas = Arc::new(CudaBlas::new(stream.clone()).expect("CudaBlas::new"));
        let workspace = unsafe { stream.alloc::<u8>(WS_BYTES) }.expect("blas workspace alloc");
        let alpha_f32 = stream.clone_htod(&[1.0f32]).expect("alpha upload");
        let beta_f32 = stream.clone_htod(&[0.0f32]).expect("beta upload");
        unsafe {
            use cudarc::cublas::sys;
            use cudarc::driver::DevicePtr;
            let (workspace_ptr, _record) = workspace.device_ptr(stream);
            assert_eq!(
                sys::cublasSetWorkspace_v2(*blas.handle(), workspace_ptr as *mut _, WS_BYTES),
                sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                "set instance BLAS workspace"
            );
            assert_eq!(
                sys::cublasSetPointerMode_v2(
                    *blas.handle(),
                    sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE
                ),
                sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
                "set instance BLAS scalar mode"
            );
        }
        Self {
            blas,
            _workspace: workspace,
            alpha_f32,
            beta_f32,
        }
    }
}

fn with_blas_scalars<R>(
    slot: &BlasSlot,
    f: impl FnOnce(&CudaSlice<f32>, &CudaSlice<f32>) -> R,
) -> R {
    f(&slot.alpha_f32, &slot.beta_f32)
}

struct BatchedScratchSlot {
    scratch_u64_k: CudaSlice<u64>,
    scratch_u64_v: CudaSlice<u64>,
    scratch_u64_cache: CudaSlice<u64>,
    host_k_ptrs: Box<[u64; HOST_STAGING_TOTAL]>,
    host_v_ptrs: Box<[u64; HOST_STAGING_TOTAL]>,
    host_cache_ptrs: Box<[u64; HOST_STAGING_TOTAL]>,
}
impl BatchedScratchSlot {
    fn new(stream: &Arc<CudaStream>) -> Self {
        Self {
            scratch_u64_k: unsafe { stream.alloc::<u64>(HOST_STAGING_TOTAL) }
                .expect("batched K pointers"),
            scratch_u64_v: unsafe { stream.alloc::<u64>(HOST_STAGING_TOTAL) }
                .expect("batched V pointers"),
            scratch_u64_cache: unsafe { stream.alloc::<u64>(HOST_STAGING_TOTAL) }
                .expect("batched cache pointers"),
            host_k_ptrs: Box::new([0; HOST_STAGING_TOTAL]),
            host_v_ptrs: Box::new([0; HOST_STAGING_TOTAL]),
            host_cache_ptrs: Box::new([0; HOST_STAGING_TOTAL]),
        }
    }
}

fn with_batched_scratch_mut<R>(
    slot: &mut BatchedScratchSlot,
    body: impl FnOnce(&mut BatchedScratchSlot) -> R,
) -> R {
    body(slot)
}

impl Drop for CudaState {
    fn drop(&mut self) {
        if let Err(error) = CudaBackend::retire_context(self) {
            eprintln!(
                "fatal: dropping CUDA instance after unsuccessful checked retirement: {error}"
            );
            std::process::abort();
        }
        // Graphs are now destroyed while modules, workspace, staging, and
        // buffers are still owned. Auxiliary events retire only after all
        // their streams have been observed by retire_context.
        unsafe {
            if let Some(event) = self.moe_entry_event.take() {
                cudarc::driver::sys::cuEventDestroy_v2(event as _);
            }
            if let Some(events) = self.moe_exit_events.take() {
                for event in events {
                    cudarc::driver::sys::cuEventDestroy_v2(event as _);
                }
            }
        }
    }
}

// CUDA: existing KV cache path is FP16.
impl crate::backend::BackendKvDtype<crate::backend::KvFp16> for CudaBackend {
    type KvBuffer = <Self as crate::backend::Backend>::Buffer;
    type KvScales = ();
}

#[cfg(test)]
mod submission_tests;
