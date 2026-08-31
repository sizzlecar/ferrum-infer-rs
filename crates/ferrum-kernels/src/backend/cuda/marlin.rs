//! Marlin INT4xFP16 fused GEMM kernel (IST Austria).
//!
//! Near-ideal 3.9x speedup over FP16 cuBLAS for INT4 quantized weights.
//! Weights must be in Marlin packed format (different from GPTQ).
//!
//! Constraints: K % 128 == 0, N % 256 == 0, SM >= 8.0 (Ampere+).

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::backend::native_status::StagedNativeStatus;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaMarlinRuntimeConfig {
    profile: bool,
    skip_ws_zero: bool,
    trace_shapes: bool,
    trace_shapes_max: u64,
}

impl CudaMarlinRuntimeConfig {
    fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut config = Self {
            profile: false,
            skip_ws_zero: false,
            trace_shapes: false,
            trace_shapes_max: 256,
        };
        for (name, value) in vars {
            match name.as_ref() {
                "FERRUM_MARLIN_PROFILE" => config.profile = value.as_ref() == "1",
                "FERRUM_MARLIN_SKIP_WS_ZERO" => config.skip_ws_zero = value.as_ref() == "1",
                "FERRUM_MARLIN_TRACE_SHAPES" => config.trace_shapes = value.as_ref() == "1",
                "FERRUM_MARLIN_TRACE_SHAPES_MAX" => {
                    if let Ok(max) = value.as_ref().parse::<u64>() {
                        config.trace_shapes_max = max;
                    }
                }
                _ => {}
            }
        }
        config
    }
}

fn cuda_marlin_runtime_config() -> &'static CudaMarlinRuntimeConfig {
    static CONFIG: OnceLock<CudaMarlinRuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(CudaMarlinRuntimeConfig::from_env)
}

/// Cached `FERRUM_MARLIN_SKIP_WS_ZERO=1` flag. Read once on first
/// access, cheap for hot paths (called per Marlin GEMM dispatch).
fn skip_ws_zero() -> bool {
    cuda_marlin_runtime_config().skip_ws_zero
}

fn should_zero_workspace(config: &CudaMarlinRuntimeConfig) -> bool {
    !config.skip_ws_zero
}

/// Profile-only nested dense Marlin counters. They are intentionally not part
/// of normal model timings because callers already time the full projection.
pub static MARLIN_WS_ZERO_TIME_US: AtomicU64 = AtomicU64::new(0);
pub static MARLIN_WS_ZERO_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MARLIN_GATHER_TIME_US: AtomicU64 = AtomicU64::new(0);
pub static MARLIN_GATHER_CALLS: AtomicU64 = AtomicU64::new(0);
pub static MARLIN_KERNEL_TIME_US: AtomicU64 = AtomicU64::new(0);
pub static MARLIN_KERNEL_CALLS: AtomicU64 = AtomicU64::new(0);
static MARLIN_TRACE_SHAPE_CALLS: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarlinProfileBucketStats {
    pub ws_zero_us: u64,
    pub ws_zero_calls: u64,
    pub gather_us: u64,
    pub gather_calls: u64,
    pub kernel_us: u64,
    pub kernel_calls: u64,
}

impl MarlinProfileBucketStats {
    pub const ZERO: Self = Self {
        ws_zero_us: 0,
        ws_zero_calls: 0,
        gather_us: 0,
        gather_calls: 0,
        kernel_us: 0,
        kernel_calls: 0,
    };

    fn record_ws_zero(&mut self, us: u64) {
        self.ws_zero_us += us;
        self.ws_zero_calls += 1;
    }

    fn record_gather(&mut self, us: u64) {
        self.gather_us += us;
        self.gather_calls += 1;
    }

    fn record_kernel(&mut self, us: u64) {
        self.kernel_us += us;
        self.kernel_calls += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarlinProfileByProjection {
    pub qkv: MarlinProfileBucketStats,
    pub o_proj: MarlinProfileBucketStats,
    pub gate_up: MarlinProfileBucketStats,
    pub down: MarlinProfileBucketStats,
    pub lm_head: MarlinProfileBucketStats,
    pub other: MarlinProfileBucketStats,
}

impl MarlinProfileByProjection {
    pub const ZERO: Self = Self {
        qkv: MarlinProfileBucketStats::ZERO,
        o_proj: MarlinProfileBucketStats::ZERO,
        gate_up: MarlinProfileBucketStats::ZERO,
        down: MarlinProfileBucketStats::ZERO,
        lm_head: MarlinProfileBucketStats::ZERO,
        other: MarlinProfileBucketStats::ZERO,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarlinProfileBucket {
    Qkv,
    OProj,
    GateUp,
    Down,
    LmHead,
    Other,
}

static MARLIN_PROFILE_BY_PROJECTION: Mutex<MarlinProfileByProjection> =
    Mutex::new(MarlinProfileByProjection::ZERO);

struct CudaMarlinEventTimer {
    start: cudarc::driver::sys::CUevent,
    end: cudarc::driver::sys::CUevent,
}

impl CudaMarlinEventTimer {
    fn start(raw_stream: cudarc::driver::sys::CUstream) -> Option<Self> {
        use cudarc::driver::sys as cu;
        let mut start: cu::CUevent = std::ptr::null_mut();
        let mut end: cu::CUevent = std::ptr::null_mut();
        unsafe {
            let _ = cu::cuEventCreate(&mut start, 0);
            let _ = cu::cuEventCreate(&mut end, 0);
        }
        if start.is_null() || end.is_null() {
            unsafe {
                if !start.is_null() {
                    let _ = cu::cuEventDestroy_v2(start);
                }
                if !end.is_null() {
                    let _ = cu::cuEventDestroy_v2(end);
                }
            }
            return None;
        }
        let timer = Self { start, end };
        timer.record_start(raw_stream);
        Some(timer)
    }

    fn record_start(&self, raw_stream: cudarc::driver::sys::CUstream) {
        unsafe {
            let _ = cudarc::driver::sys::cuEventRecord(self.start, raw_stream);
        }
    }

    fn finish_us(&self, raw_stream: cudarc::driver::sys::CUstream) -> u64 {
        unsafe {
            let _ = cudarc::driver::sys::cuEventRecord(self.end, raw_stream);
            let _ = cudarc::driver::sys::cuEventSynchronize(self.end);
        }
        (unsafe { cudarc::driver::result::event::elapsed(self.start, self.end) }
            .ok()
            .unwrap_or(0.0) as f64
            * 1000.0) as u64
    }
}

impl Drop for CudaMarlinEventTimer {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::cuEventDestroy_v2(self.start);
            let _ = cudarc::driver::sys::cuEventDestroy_v2(self.end);
        }
    }
}

fn marlin_profile_bucket_from_label(label: &str) -> MarlinProfileBucket {
    if label.contains("qkv_proj") {
        MarlinProfileBucket::Qkv
    } else if label.contains("o_proj") {
        MarlinProfileBucket::OProj
    } else if label.contains("gate_up_proj") {
        MarlinProfileBucket::GateUp
    } else if label.contains("down_proj") {
        MarlinProfileBucket::Down
    } else if label.contains("lm_head") {
        MarlinProfileBucket::LmHead
    } else {
        MarlinProfileBucket::Other
    }
}

fn current_marlin_profile_bucket() -> MarlinProfileBucket {
    marlin_profile_bucket_from_label(&super::current_cuda_alloc_label())
}

fn marlin_profile_bucket_mut(
    stats: &mut MarlinProfileByProjection,
    bucket: MarlinProfileBucket,
) -> &mut MarlinProfileBucketStats {
    match bucket {
        MarlinProfileBucket::Qkv => &mut stats.qkv,
        MarlinProfileBucket::OProj => &mut stats.o_proj,
        MarlinProfileBucket::GateUp => &mut stats.gate_up,
        MarlinProfileBucket::Down => &mut stats.down,
        MarlinProfileBucket::LmHead => &mut stats.lm_head,
        MarlinProfileBucket::Other => &mut stats.other,
    }
}

fn with_marlin_profile_bucket_stats(
    bucket: MarlinProfileBucket,
    f: impl FnOnce(&mut MarlinProfileBucketStats),
) {
    let mut stats = MARLIN_PROFILE_BY_PROJECTION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(marlin_profile_bucket_mut(&mut stats, bucket));
}

fn record_marlin_ws_zero(bucket: MarlinProfileBucket, us: u64) {
    with_marlin_profile_bucket_stats(bucket, |stats| stats.record_ws_zero(us));
}

fn record_marlin_gather(bucket: MarlinProfileBucket, us: u64) {
    with_marlin_profile_bucket_stats(bucket, |stats| stats.record_gather(us));
}

fn record_marlin_kernel(bucket: MarlinProfileBucket, us: u64) {
    with_marlin_profile_bucket_stats(bucket, |stats| stats.record_kernel(us));
}

pub fn record_marlin_gather_for_current_label(us: u64) {
    MARLIN_GATHER_TIME_US.fetch_add(us, Ordering::Relaxed);
    MARLIN_GATHER_CALLS.fetch_add(1, Ordering::Relaxed);
    record_marlin_gather(current_marlin_profile_bucket(), us);
}

pub fn drain_marlin_profile_by_projection() -> MarlinProfileByProjection {
    let mut stats = MARLIN_PROFILE_BY_PROJECTION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let snapshot = *stats;
    *stats = MarlinProfileByProjection::ZERO;
    snapshot
}

pub fn profile_marlin() -> bool {
    cuda_marlin_runtime_config().profile
}

fn trace_marlin_shapes() -> bool {
    cuda_marlin_runtime_config().trace_shapes
}

fn marlin_shape_trace_max() -> u64 {
    cuda_marlin_runtime_config().trace_shapes_max
}

// FFI declaration for the Marlin CUDA kernel.
// Only linked when the "marlin" feature is enabled (requires nvcc + SM >= 8.0).
#[cfg(feature = "marlin")]
extern "C" {
    fn marlin_cuda(
        A: *const std::ffi::c_void,
        B: *const std::ffi::c_void,
        C: *mut std::ffi::c_void,
        s: *const std::ffi::c_void,
        prob_m: i32,
        prob_n: i32,
        prob_k: i32,
        workspace: *mut std::ffi::c_void,
        groupsize: i32,
        dev: i32,
        stream: cudarc::driver::sys::CUstream,
        thread_k: i32,
        thread_n: i32,
        sms: i32,
        max_par: i32,
        // -1 ⇒ same as prob_n. For offset GEMM into a stacked B/s
        // buffer, pass total_n so b_gl_stride and s_gl_stride see the
        // full N width while iteration covers only the expert subset.
        prob_n_full: i32,
    ) -> i32;

    // Stage 11: fused MoE Marlin. ONE launch processes all experts in a
    // bucket. Caller pre-buckets experts by their thread_m_blocks need
    // (prob_m here = 16 * thread_m_blocks). gridDim.y = expert_count.
    fn marlin_cuda_moe(
        A: *const std::ffi::c_void,
        B: *const std::ffi::c_void,
        C: *mut std::ffi::c_void,
        s: *const std::ffi::c_void,
        prob_m: i32,
        prob_n: i32,
        prob_k: i32,
        workspace: *mut std::ffi::c_void,
        a_row_offsets: *const i32, // device [E_global] cumulative row offsets in A
        tokens_per_expert: *const i32, // device [E_global]
        active_expert_ids: *const i32, // device [expert_count] (or null for identity)
        expert_count: i32,
        b_int4_per_expert: i32,
        s_int4_per_expert: i32,
        locks_i32_per_expert: i32,
        groupsize: i32,
        dev: i32,
        stream: cudarc::driver::sys::CUstream,
        thread_k: i32,
        thread_n: i32,
        sms: i32,
        prob_n_full: i32,
    ) -> i32;
}

// vLLM marlin_moe_wna16 port (Stage 14). Supplied by the versioned native
// operator artifact set. Single fused
// (sorted_token_ids, expert_ids) launch — eliminates the m=16 padding
// waste of our Stage 12.1 path. Linked statically only when the
// `vllm-moe-marlin` feature is built in.
#[cfg(feature = "vllm-moe-marlin")]
extern "C" {
    fn ferrum_vllm_marlin_moe_set_profile_config(
        path: *const std::ffi::c_char,
        commit_sha: *const std::ffi::c_char,
        env_hash: *const std::ffi::c_char,
        model: *const std::ffi::c_char,
        concurrency: i32,
        runtime_flags_json: *const std::ffi::c_char,
    );

    fn ferrum_vllm_marlin_moe_clear_profile_config();

    fn ferrum_vllm_marlin_moe_f16(
        a: *const std::ffi::c_void,        // [size_m, size_k] fp16
        b: *const std::ffi::c_void,        // [num_experts, k/16, n*pack/16] i32 marlin-packed
        c: *mut std::ffi::c_void,          // [size_m * top_k, size_n] fp16
        c_tmp: *mut std::ffi::c_void,      // fp32 scratch (or null)
        b_scales: *const std::ffi::c_void, // [num_experts, num_groups, size_n] fp16
        b_zeros: *const std::ffi::c_void,  // [num_experts, num_groups, size_n/8] i32 or null
        workspace: *mut std::ffi::c_void,  // [N/128 * sms * 4] i32
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_past_padded: *const i32,
        topk_weights: *const f32, // (or null when mul_topk_weights=0)
        moe_block_size: i32,      // 8 / 16 / 32 / 48 / 64
        top_k: i32,
        mul_topk_weights: i32, // 0 or 1
        is_ep: i32,            // 0 or 1
        prob_m: i32,
        prob_n: i32,
        prob_k: i32,
        group_size: i32, // 128 typically
        has_zp: i32,     // 0 symmetric kU4B8, 1 asymmetric kU4 + b_zeros
        dev: i32,
        stream: cudarc::driver::sys::CUstream,
        use_atomic_add: i32,
        use_fp32_reduce: i32,
    ) -> i32;

    fn ferrum_vllm_marlin_moe_fp8_f16(
        a: *const std::ffi::c_void,        // [size_m, size_k] fp16
        b: *const std::ffi::c_void,        // [num_experts, ...] E4M3 Marlin-packed
        c: *mut std::ffi::c_void,          // [size_m * top_k, size_n] fp16
        c_tmp: *mut std::ffi::c_void,      // fp32 scratch (or null)
        b_scales: *const std::ffi::c_void, // [num_experts, 1, size_n] fp16
        b_zeros: *const std::ffi::c_void,  // always null for E4M3
        workspace: *mut std::ffi::c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_past_padded: *const i32,
        topk_weights: *const f32,
        moe_block_size: i32,
        top_k: i32,
        mul_topk_weights: i32,
        is_ep: i32,
        prob_m: i32,
        prob_n: i32,
        prob_k: i32,
        group_size: i32, // -1 for channelwise or 128 for block-grouped E4M3
        has_zp: i32,     // always 0 for E4M3
        dev: i32,
        stream: cudarc::driver::sys::CUstream,
        use_atomic_add: i32,
        use_fp32_reduce: i32,
    ) -> i32;

    fn ferrum_vllm_marlin_moe_mxfp4_bf16(
        a: *const std::ffi::c_void,        // [size_m, size_k] bf16
        b: *const std::ffi::c_void,        // [num_experts, ...] E2M1 Marlin-packed nibbles
        c: *mut std::ffi::c_void,          // [size_m * top_k, size_n] bf16
        c_tmp: *mut std::ffi::c_void,      // fp32 scratch (or null)
        b_bias: *const std::ffi::c_void,   // [num_experts, size_n] bf16
        b_scales: *const std::ffi::c_void, // [num_experts, size_k / 32, size_n] E8M0 bytes
        workspace: *mut std::ffi::c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        num_tokens_past_padded: *const i32,
        topk_weights: *const f32,
        moe_block_size: i32,
        top_k: i32,
        mul_topk_weights: i32,
        is_ep: i32,
        prob_m: i32,
        prob_n: i32,
        prob_k: i32,
        group_size: i32, // exactly 32 for E2M1 + E8M0 MXFP4
        dev: i32,
        stream: cudarc::driver::sys::CUstream,
        use_atomic_add: i32,
        use_fp32_reduce: i32,
    ) -> i32;
}

#[cfg(feature = "vllm-moe-marlin")]
pub fn configure_vllm_moe_profile_sink(
    config: &ferrum_bench_core::ProfileSinkConfig,
) -> std::io::Result<()> {
    use std::ffi::CString;

    let Some(path) = &config.jsonl_path else {
        unsafe { ferrum_vllm_marlin_moe_clear_profile_config() };
        return Ok(());
    };

    let path = CString::new(path.as_os_str().to_string_lossy().into_owned()).map_err(|err| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("profile path contains NUL byte: {err}"),
        )
    })?;
    let commit_sha = CString::new(
        config
            .metadata
            .commit_sha
            .as_deref()
            .unwrap_or_default()
            .to_string(),
    )
    .map_err(profile_cstring_error("profile commit_sha"))?;
    let env_hash = CString::new(config.metadata.env_hash.clone())
        .map_err(profile_cstring_error("env_hash"))?;
    let model =
        CString::new(config.metadata.model.clone()).map_err(profile_cstring_error("model"))?;
    let runtime_flags_json =
        serde_json::to_string(&config.metadata.runtime_flags).unwrap_or_else(|_| "{}".to_string());
    let runtime_flags_json =
        CString::new(runtime_flags_json).map_err(profile_cstring_error("runtime_flags_json"))?;

    unsafe {
        ferrum_vllm_marlin_moe_set_profile_config(
            path.as_ptr(),
            commit_sha.as_ptr(),
            env_hash.as_ptr(),
            model.as_ptr(),
            config.metadata.concurrency.min(i32::MAX as u32) as i32,
            runtime_flags_json.as_ptr(),
        );
    }
    Ok(())
}

#[cfg(feature = "vllm-moe-marlin")]
fn profile_cstring_error(field: &'static str) -> impl FnOnce(std::ffi::NulError) -> std::io::Error {
    move |err| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{field} contains NUL byte: {err}"),
        )
    }
}

/// Check if Marlin kernel is available at compile time.
pub fn is_available() -> bool {
    cfg!(feature = "marlin")
}

/// Marlin-format quantized weight for one linear layer.
pub struct MarlinWeight {
    /// Repacked INT4 weights in Marlin tile format: varies by K, N
    pub qweight: CudaSlice<i32>,
    /// Per-group FP16 scales (permuted for Marlin access pattern)
    pub scales: CudaSlice<half::f16>,
    /// Optional per-group GPTQ zero-points for vLLM Marlin-MoE asymmetric
    /// INT4. Stored packed as actual zero-point codes, not AutoGPTQ's
    /// on-disk `qzeros = zero - 1`.
    pub qzeros: Option<CudaSlice<i32>>,
    /// Workspace for Marlin kernel: [N/128 * max_par] int32, zeroed
    pub workspace: CudaSlice<i32>,
    pub k: usize,
    pub n: usize,
    pub group_size: i32,
    /// True when `qweight` is in vLLM Marlin-MoE tile layout. Such stacks
    /// must be dispatched through `marlin_gemm_moe_vllm`, not bucketed
    /// IST-DASLab offset GEMMs.
    pub vllm_moe: bool,
    /// Activation gather permutation for desc_act=true (act-order) GPTQ.
    /// `perm[i]` = original column index that should appear at position i
    /// after gather. Computed at load time as `argsort(g_idx_disk)`.
    /// `qweight` rows have already been permuted by this; runtime gathers
    /// input columns by the same perm so the standard Marlin kernel
    /// produces the un-permuted GEMM result. None for desc_act=false.
    pub perm: Option<CudaSlice<i32>>,
}

/// Run Marlin INT4xFP16 fused GEMM.
///
/// Computes: C[m, n] = A[m, k] @ dequant(B[k, n])
/// where B is in Marlin packed INT4 format.
///
/// Only available when compiled with `--features marlin`.
#[cfg(feature = "marlin")]
pub fn marlin_gemm(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<half::f16>,
    weight: &MarlinWeight,
    output: &mut CudaSlice<half::f16>,
    m: i32,
) -> candle_core::Result<()> {
    let n = weight.n as i32;
    let k = weight.k as i32;

    // Layers with n % 256 != 0 (e.g. a quantized 2048x128 MoE router or a
    // 151936-wide lm_head from repacked repos) only have valid kernel
    // instantiations for the small-m 128x128 tile (THREAD_M_BLOCKS == 1,
    // i.e. m <= 16); the m > 16 auto-config picks 64x256 tiles and the
    // kernel rejects the shape. Split such GEMMs into m <= 16 chunks —
    // a handful of extra ~5us launches on at most two tiny/tall layers.
    if n % 256 != 0 && m > 16 {
        let mut row = 0usize;
        while row < m as usize {
            let chunk = (m as usize - row).min(16);
            let a_view = input.slice(row * weight.k..(row + chunk) * weight.k);
            let mut c_view = output.slice_mut(row * weight.n..(row + chunk) * weight.n);
            marlin_gemm_chunk(stream, &a_view, weight, &mut c_view, chunk as i32)?;
            row += chunk;
        }
        return Ok(());
    }
    marlin_gemm_chunk(
        stream,
        &input.slice(..),
        weight,
        &mut output.slice_mut(..),
        m,
    )
}

fn marlin_gemm_chunk(
    stream: &Arc<CudaStream>,
    input: &cudarc::driver::CudaView<'_, half::f16>,
    weight: &MarlinWeight,
    output: &mut cudarc::driver::CudaViewMut<'_, half::f16>,
    m: i32,
) -> candle_core::Result<()> {
    let n = weight.n as i32;
    let k = weight.k as i32;

    let raw_stream = stream.cu_stream();
    let profile = profile_marlin();
    let profile_bucket = profile.then(current_marlin_profile_bucket);

    // Zero workspace on the runner's stream — Marlin uses it as mutex locks.
    // All operations (memset + kernel) on same stream → naturally ordered.
    if should_zero_workspace(cuda_marlin_runtime_config()) {
        let timer = profile
            .then(|| CudaMarlinEventTimer::start(raw_stream))
            .flatten();
        let (ws_ptr, _guard) = weight.workspace.device_ptr(stream);
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(ws_ptr, 0, weight.workspace.len(), raw_stream);
        }
        if let Some(timer) = timer {
            let elapsed_us = timer.finish_us(raw_stream);
            MARLIN_WS_ZERO_TIME_US.fetch_add(elapsed_us, Ordering::Relaxed);
            MARLIN_WS_ZERO_CALLS.fetch_add(1, Ordering::Relaxed);
            if let Some(bucket) = profile_bucket {
                record_marlin_ws_zero(bucket, elapsed_us);
            }
        }
    }

    // Get raw device pointers
    let (a_ptr, _a_guard) = input.device_ptr(stream);
    let (b_ptr, _b_guard) = weight.qweight.device_ptr(stream);
    let (c_ptr, _c_guard) = output.device_ptr(stream);
    let (s_ptr, _s_guard) = weight.scales.device_ptr(stream);
    let (ws_ptr, _ws_guard) = weight.workspace.device_ptr(stream);

    if trace_marlin_shapes() {
        let call = MARLIN_TRACE_SHAPE_CALLS.fetch_add(1, Ordering::Relaxed);
        if call < marlin_shape_trace_max() {
            let label = super::current_cuda_alloc_label();
            let bucket = marlin_profile_bucket_from_label(&label);
            eprintln!(
                "[marlin-shape-trace] call={} label={} bucket={:?} m={} n={} k={} gs={} qweight_len={} scales_len={} workspace_len={} a=0x{:x} b=0x{:x} c=0x{:x} s=0x{:x} ws=0x{:x}",
                call,
                label,
                bucket,
                m,
                n,
                k,
                weight.group_size,
                weight.qweight.len(),
                weight.scales.len(),
                weight.workspace.len(),
                a_ptr,
                b_ptr,
                c_ptr,
                s_ptr,
                ws_ptr,
            );
        }
    }

    let timer = profile
        .then(|| CudaMarlinEventTimer::start(raw_stream))
        .flatten();
    let ret = unsafe {
        marlin_cuda(
            a_ptr as *const _,
            b_ptr as *const _,
            c_ptr as *mut _,
            s_ptr as *const _,
            m,
            n,
            k,
            ws_ptr as *mut _,
            weight.group_size,
            0, // dev
            raw_stream,
            -1, // auto thread_k
            -1, // auto thread_n
            -1, // auto sms
            16, // max_par
            -1, // prob_n_full = prob_n (non-stacked)
        )
    };
    if let Some(timer) = timer {
        let elapsed_us = timer.finish_us(raw_stream);
        MARLIN_KERNEL_TIME_US.fetch_add(elapsed_us, Ordering::Relaxed);
        MARLIN_KERNEL_CALLS.fetch_add(1, Ordering::Relaxed);
        if let Some(bucket) = profile_bucket {
            record_marlin_kernel(bucket, elapsed_us);
        }
    }

    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "marlin_cuda failed: ret={ret} (m={m}, n={n}, k={k}, gs={})",
            weight.group_size
        )));
    }

    // No per-call sync needed — all operations (memset + kernel) are on the
    // runner's stream. decode_step syncs once at the end before returning logits.
    Ok(())
}

/// Stub when Marlin feature is not enabled.
#[cfg(not(feature = "marlin"))]
pub fn marlin_gemm(
    _stream: &Arc<CudaStream>,
    _input: &CudaSlice<half::f16>,
    _weight: &MarlinWeight,
    _output: &mut CudaSlice<half::f16>,
    _m: i32,
) -> candle_core::Result<()> {
    Err(candle_core::Error::Msg(
        "Marlin kernel not available (compile with --features marlin)".into(),
    ))
}

/// Marlin GEMM on a column-slice of a stacked weight (used for MoE
/// expert dispatch). The stacked `weight` holds num_experts × n_per_expert
/// columns concatenated along N; this call processes columns
/// `[expert_offset .. expert_offset + expert_n)` only.
///
/// `expert_offset` and `expert_n` MUST be multiples of Marlin's `tile_n`
/// (typically 64). The repack laid out the whole N contiguously so a
/// pointer offset lands on a tile boundary.
///
/// Workspace: shares the parent stacked workspace; we offset its pointer
/// by `expert_offset / 128` ints so each expert uses its own mutex slot
/// range.
#[cfg(feature = "marlin")]
pub fn marlin_gemm_with_offset(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<half::f16>,
    weight: &MarlinWeight,
    output: &mut CudaSlice<half::f16>,
    m: i32,
    expert_offset: i32,
    expert_n: i32,
) -> candle_core::Result<()> {
    use cudarc::driver::DevicePtr;
    let n = expert_n;
    let k = weight.k as i32;
    if expert_offset < 0 || expert_n <= 0 || expert_offset + expert_n > weight.n as i32 {
        return Err(candle_core::Error::Msg(format!(
            "marlin offset out of range: offset={expert_offset} n={expert_n} stacked_n={}",
            weight.n
        )));
    }
    let raw_stream = stream.cu_stream();

    // PER-EXPERT CONTIGUOUS LAYOUT (built by load_gptq_stacked):
    // Each expert's packed bytes are CONTIGUOUS in the buffer.
    // Buffer = [exp0_marlin_tile | exp1_marlin_tile | ...].
    // expert_idx is implicit: expert_offset / expert_n.
    //
    // qweight: per-expert tile = (n_per_expert * k / 8) i32. Offset
    //          by expert_idx × that_size i32.
    // scales:  per-expert tile = (k/group_size * n_per_expert) f16.
    //          Offset by expert_idx × that_size f16.
    // workspace: per-expert range = (n_per_expert/128) * MAX_PAR i32.
    //          Offset by expert_idx × that_size i32.
    //
    // Marlin sees a regular N=expert_n tile per call. prob_n =
    // prob_n_full = expert_n (no stride decoupling needed).
    let expert_idx = (expert_offset / expert_n) as usize;
    let n_per = expert_n as usize;
    let k_us = k as usize;

    const MAX_PAR: usize = 16;
    let ws_per_expert = (n_per / 128).max(1) * MAX_PAR;
    let ws_offset_bytes = expert_idx * ws_per_expert * std::mem::size_of::<i32>();
    if should_zero_workspace(cuda_marlin_runtime_config()) {
        let (ws_ptr, _g) = weight.workspace.device_ptr(stream);
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(
                ws_ptr + ws_offset_bytes as u64,
                0,
                ws_per_expert,
                raw_stream,
            );
        }
    }

    let qw_per_expert_i32 = (n_per * k_us) / 8;
    let qw_offset_bytes = expert_idx * qw_per_expert_i32 * std::mem::size_of::<i32>();

    let num_groups = k_us / weight.group_size as usize;
    let sc_per_expert_f16 = num_groups * n_per;
    let scales_offset_bytes = expert_idx * sc_per_expert_f16 * std::mem::size_of::<half::f16>();

    let (a_ptr, _a_guard) = input.device_ptr(stream);
    let (b_ptr_full, _b_guard) = weight.qweight.device_ptr(stream);
    let (c_ptr, _c_guard) = output.device_ptr(stream);
    let (s_ptr_full, _s_guard) = weight.scales.device_ptr(stream);
    let (ws_ptr_full, _ws_guard) = weight.workspace.device_ptr(stream);
    let b_ptr = b_ptr_full + qw_offset_bytes as u64;
    let s_ptr = s_ptr_full + scales_offset_bytes as u64;
    let ws_ptr = ws_ptr_full + ws_offset_bytes as u64;

    let ret = unsafe {
        marlin_cuda(
            a_ptr as *const _,
            b_ptr as *const _,
            c_ptr as *mut _,
            s_ptr as *const _,
            m,
            n,
            k,
            ws_ptr as *mut _,
            weight.group_size,
            0,
            raw_stream,
            -1,
            -1,
            -1,
            16,
            // Per-expert contiguous: stride == iteration width.
            -1,
        )
    };
    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "marlin_cuda (offset) failed ret={ret} m={m} n={n} k={k} offset={expert_offset}"
        )));
    }
    Ok(())
}

#[cfg(not(feature = "marlin"))]
pub fn marlin_gemm_with_offset(
    _stream: &Arc<CudaStream>,
    _input: &CudaSlice<half::f16>,
    _weight: &MarlinWeight,
    _output: &mut CudaSlice<half::f16>,
    _m: i32,
    _expert_offset: i32,
    _expert_n: i32,
) -> candle_core::Result<()> {
    Err(candle_core::Error::Msg(
        "Marlin kernel not available (compile with --features marlin)".into(),
    ))
}

/// Same as [`marlin_gemm_with_offset`] but also strides the input and
/// output buffers by row offsets. Used by the bucketed MoE dispatcher
/// to run a single expert's column-slice GEMM against a sub-range of
/// the packed input/output buffer without needing a buffer-view type.
///
/// `in_row_offset` rows of `K` f16 elements at the start of `input`
/// are skipped; `out_row_offset` rows of `expert_n` f16 elements at
/// the start of `output` are skipped.
#[cfg(feature = "marlin")]
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm_with_offset_strided(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<half::f16>,
    in_row_offset: i32,
    weight: &MarlinWeight,
    output: &mut CudaSlice<half::f16>,
    out_row_offset: i32,
    m: i32,
    expert_offset: i32,
    expert_n: i32,
) -> candle_core::Result<()> {
    use cudarc::driver::DevicePtr;
    let n = expert_n;
    let k = weight.k as i32;
    if expert_offset < 0 || expert_n <= 0 || expert_offset + expert_n > weight.n as i32 {
        return Err(candle_core::Error::Msg(format!(
            "marlin offset out of range: offset={expert_offset} n={expert_n} stacked_n={}",
            weight.n
        )));
    }
    let raw_stream = stream.cu_stream();

    // Per-expert contiguous layout, same offset arithmetic as
    // marlin_gemm_with_offset.
    let expert_idx = (expert_offset / expert_n) as usize;
    let n_per = expert_n as usize;
    let k_us = k as usize;

    const MAX_PAR: usize = 16;
    let ws_per_expert = (n_per / 128).max(1) * MAX_PAR;
    let ws_offset_bytes = expert_idx * ws_per_expert * std::mem::size_of::<i32>();
    // Skip per-call workspace zeroing if env says so. Caller is then
    // responsible for bulk-zeroing the workspace before the batch
    // (saves N-1 cuMemsetD32Async launches per phase). Cached on
    // first access — std::env::var is too slow for the hot path.
    if !skip_ws_zero() {
        let (ws_ptr, _g) = weight.workspace.device_ptr(stream);
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(
                ws_ptr + ws_offset_bytes as u64,
                0,
                ws_per_expert,
                raw_stream,
            );
        }
    }

    let qw_per_expert_i32 = (n_per * k_us) / 8;
    let qw_offset_bytes = expert_idx * qw_per_expert_i32 * std::mem::size_of::<i32>();

    let num_groups = k_us / weight.group_size as usize;
    let sc_per_expert_f16 = num_groups * n_per;
    let scales_offset_bytes = expert_idx * sc_per_expert_f16 * std::mem::size_of::<half::f16>();

    let in_offset_bytes = in_row_offset as usize * (k as usize) * std::mem::size_of::<half::f16>();
    let out_offset_bytes =
        out_row_offset as usize * (n as usize) * std::mem::size_of::<half::f16>();

    let (a_ptr, _a_guard) = input.device_ptr(stream);
    let (b_ptr_full, _b_guard) = weight.qweight.device_ptr(stream);
    let (c_ptr, _c_guard) = output.device_ptr(stream);
    let (s_ptr_full, _s_guard) = weight.scales.device_ptr(stream);
    let (ws_ptr_full, _ws_guard) = weight.workspace.device_ptr(stream);
    let a_ptr_off = a_ptr + in_offset_bytes as u64;
    let b_ptr = b_ptr_full + qw_offset_bytes as u64;
    let c_ptr_off = c_ptr + out_offset_bytes as u64;
    let s_ptr = s_ptr_full + scales_offset_bytes as u64;
    let ws_ptr = ws_ptr_full + ws_offset_bytes as u64;

    let ret = unsafe {
        marlin_cuda(
            a_ptr_off as *const _,
            b_ptr as *const _,
            c_ptr_off as *mut _,
            s_ptr as *const _,
            m,
            n,
            k,
            ws_ptr as *mut _,
            weight.group_size,
            0,
            raw_stream,
            -1,
            -1,
            -1,
            16,
            // Per-expert contiguous: stride == iteration.
            -1,
        )
    };
    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "marlin_cuda (offset_strided) failed ret={ret} m={m} n={n} k={k} \
             expert_offset={expert_offset} in_row_offset={in_row_offset} \
             out_row_offset={out_row_offset}"
        )));
    }
    Ok(())
}

#[cfg(not(feature = "marlin"))]
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm_with_offset_strided(
    _stream: &Arc<CudaStream>,
    _input: &CudaSlice<half::f16>,
    _in_row_offset: i32,
    _weight: &MarlinWeight,
    _output: &mut CudaSlice<half::f16>,
    _out_row_offset: i32,
    _m: i32,
    _expert_offset: i32,
    _expert_n: i32,
) -> candle_core::Result<()> {
    Err(candle_core::Error::Msg(
        "Marlin kernel not available (compile with --features marlin)".into(),
    ))
}

/// Stage 11 — fused MoE Marlin: ONE launch processes all experts in
/// `active_expert_ids` (len = `expert_count`) by reading `expert_id =
/// active_expert_ids[blockIdx.y]`, applying pointer offsets to the
/// stacked B / s / workspace, and reading per-expert (m, A_row_offset)
/// from the per-layer `tokens_per_expert` / `a_row_offsets` arrays.
///
/// `prob_m` is the bucket-wide max-m: every active expert MUST have
/// `tokens_per_expert[e] ≤ prob_m`, and `prob_m` MUST be a multiple of
/// 16. The kernel selects `thread_m_blocks = prob_m / 16` (1..=4); for
/// experts with fewer tokens the kernel pads with zeros.
///
/// Caller is responsible for:
///   - bucketing active experts by max-m (prob_m ∈ {16, 32, 48, 64})
///   - pre-zeroing the bucketed workspace slots (or relying on
///     `marlin_zero_stacked_workspace` having been called this iter)
///   - ensuring all active experts share the same `prob_n`, `prob_k`,
///     `group_size` (true for MoE — every expert in a layer has the
///     same shape)
#[cfg(feature = "marlin")]
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm_moe(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<half::f16>,
    weight: &MarlinWeight,
    output: &mut CudaSlice<half::f16>,
    a_row_offsets: &CudaSlice<i32>,
    tokens_per_expert: &CudaSlice<i32>,
    active_expert_ids: Option<&CudaSlice<i32>>,
    expert_count: i32,
    prob_m: i32,
    n_per_expert: i32,
    num_experts_global: i32,
) -> candle_core::Result<()> {
    use cudarc::driver::DevicePtr;
    if expert_count <= 0 {
        return Ok(());
    }
    if prob_m <= 0 || prob_m > 64 || prob_m % 16 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "marlin_gemm_moe: prob_m must be in {{16, 32, 48, 64}}, got {prob_m}"
        )));
    }
    let n = n_per_expert;
    let k = weight.k as i32;
    let n_per = n as usize;
    let k_us = k as usize;
    if n_per == 0 || (weight.n as i32) < num_experts_global * n {
        return Err(candle_core::Error::Msg(format!(
            "marlin_gemm_moe: stacked weight N={} too small for E_global={num_experts_global} × n_per={n}",
            weight.n
        )));
    }

    // Stacked-tile strides (int4 elems = 16 bytes each).
    // qweight per expert = (n_per * k) / 8 i32 = (n_per * k) / 32 int4
    // scales per expert  = (k/group_size * n_per) f16 = (...)/8 int4
    // workspace per expert = (n_per/128) * MAX_PAR i32
    const MAX_PAR: usize = 16;
    let b_int4_per_expert = ((n_per * k_us) / 32) as i32;
    let groups = k_us / weight.group_size as usize;
    let s_int4_per_expert = ((groups * n_per) / 8) as i32;
    let locks_i32_per_expert = (((n_per / 128).max(1)) * MAX_PAR) as i32;

    let raw_stream = stream.cu_stream();
    let (a_ptr, _ag) = input.device_ptr(stream);
    let (b_ptr, _bg) = weight.qweight.device_ptr(stream);
    let (c_ptr, _cg) = output.device_ptr(stream);
    let (s_ptr, _sg) = weight.scales.device_ptr(stream);
    let (ws_ptr, _wg) = weight.workspace.device_ptr(stream);
    let (off_ptr, _og) = a_row_offsets.device_ptr(stream);
    let (tok_ptr, _tg) = tokens_per_expert.device_ptr(stream);
    let act_ptr_opt = active_expert_ids.map(|s| s.device_ptr(stream));
    let act_raw: u64 = match &act_ptr_opt {
        Some((p, _)) => *p,
        None => 0,
    };

    let ret = unsafe {
        marlin_cuda_moe(
            a_ptr as *const _,
            b_ptr as *const _,
            c_ptr as *mut _,
            s_ptr as *const _,
            prob_m,
            n,
            k,
            ws_ptr as *mut _,
            off_ptr as *const _,
            tok_ptr as *const _,
            act_raw as *const _,
            expert_count,
            b_int4_per_expert,
            s_int4_per_expert,
            locks_i32_per_expert,
            weight.group_size,
            0, // dev
            raw_stream,
            -1,
            -1,
            -1,
            n, // prob_n_full = prob_n (per-expert contiguous stacking)
        )
    };

    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "marlin_cuda_moe failed: ret={ret} (prob_m={prob_m}, n={n}, k={k}, \
             experts={expert_count}, gs={})",
            weight.group_size
        )));
    }
    Ok(())
}

#[cfg(not(feature = "marlin"))]
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm_moe(
    _stream: &Arc<CudaStream>,
    _input: &CudaSlice<half::f16>,
    _weight: &MarlinWeight,
    _output: &mut CudaSlice<half::f16>,
    _a_row_offsets: &CudaSlice<i32>,
    _tokens_per_expert: &CudaSlice<i32>,
    _active_expert_ids: Option<&CudaSlice<i32>>,
    _expert_count: i32,
    _prob_m: i32,
    _n_per_expert: i32,
    _num_experts_global: i32,
) -> candle_core::Result<()> {
    Err(candle_core::Error::Msg(
        "Marlin kernel not available (compile with --features marlin)".into(),
    ))
}

// ===================== Stage 14: vLLM marlin_moe_wna16 port =====================

fn marlin_moe_ffi_status(ret: i32) -> (&'static str, u32) {
    let status =
        StagedNativeStatus::decode(ret).expect("Marlin-MoE status decoder requires a failure");
    let stage = match status.stage() {
        1 => "sm-count",
        2 => "max-shared-memory",
        3 => "act-order-launch",
        4 => "blocks-per-sm",
        5 => "function-attribute",
        6 => "kernel-launch",
        _ => "unknown",
    };
    (stage, u32::from(status.native_status()))
}

/// Raw, allocation-agnostic arguments for the vLLM Marlin-MoE launch.
///
/// The owning caller must retain every allocation until work enqueued on
/// `stream` has completed. Optional pointers deliberately retain their
/// corresponding mode flags so this boundary can reject inconsistent FFI
/// states before the native C++ implementation reaches `TORCH_CHECK`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MarlinMoeF16WeightType {
    U4B8,
    E4M3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MarlinMoeRawLaunchArgs {
    pub(crate) weight_type: MarlinMoeF16WeightType,
    pub(crate) a: cudarc::driver::sys::CUdeviceptr,
    pub(crate) b: cudarc::driver::sys::CUdeviceptr,
    pub(crate) c: cudarc::driver::sys::CUdeviceptr,
    pub(crate) c_tmp: Option<cudarc::driver::sys::CUdeviceptr>,
    pub(crate) scales: cudarc::driver::sys::CUdeviceptr,
    pub(crate) zero_points: Option<cudarc::driver::sys::CUdeviceptr>,
    pub(crate) workspace: cudarc::driver::sys::CUdeviceptr,
    pub(crate) sorted_token_ids: cudarc::driver::sys::CUdeviceptr,
    pub(crate) expert_ids: cudarc::driver::sys::CUdeviceptr,
    pub(crate) num_tokens_past_padded: cudarc::driver::sys::CUdeviceptr,
    pub(crate) topk_weights: Option<cudarc::driver::sys::CUdeviceptr>,
    pub(crate) moe_block_size: i32,
    pub(crate) top_k: i32,
    pub(crate) mul_topk_weights: bool,
    pub(crate) is_ep: bool,
    pub(crate) prob_m: i32,
    pub(crate) prob_n: i32,
    pub(crate) prob_k: i32,
    pub(crate) group_size: i32,
    pub(crate) has_zero_points: bool,
    pub(crate) device_ordinal: i32,
    pub(crate) use_atomic_add: bool,
    pub(crate) use_fp32_reduce: bool,
}

impl MarlinMoeRawLaunchArgs {
    fn validate(&self) -> candle_core::Result<()> {
        validate_marlin_moe_pointer("a", self.a, 16)?;
        validate_marlin_moe_pointer("b", self.b, 16)?;
        validate_marlin_moe_pointer("c", self.c, 16)?;
        validate_marlin_moe_pointer("scales", self.scales, 16)?;
        validate_marlin_moe_pointer("workspace", self.workspace, 4)?;
        validate_marlin_moe_pointer("sorted_token_ids", self.sorted_token_ids, 4)?;
        validate_marlin_moe_pointer("expert_ids", self.expert_ids, 4)?;
        validate_marlin_moe_pointer("num_tokens_past_padded", self.num_tokens_past_padded, 4)?;
        if let Some(pointer) = self.c_tmp {
            validate_marlin_moe_pointer("c_tmp", pointer, 16)?;
        }
        if let Some(pointer) = self.zero_points {
            validate_marlin_moe_pointer("zero_points", pointer, 16)?;
        }
        if let Some(pointer) = self.topk_weights {
            validate_marlin_moe_pointer("topk_weights", pointer, 4)?;
        }

        if self.prob_m <= 0 || self.prob_n <= 0 || self.prob_k <= 0 {
            return Err(invalid_marlin_moe_args(format!(
                "prob_m, prob_n, and prob_k must be positive, got [{}, {}, {}]",
                self.prob_m, self.prob_n, self.prob_k
            )));
        }
        if !matches!(self.moe_block_size, 8 | 16 | 32 | 48 | 64) {
            return Err(invalid_marlin_moe_args(format!(
                "unsupported moe_block_size {}; expected one of 8, 16, 32, 48, 64",
                self.moe_block_size
            )));
        }
        if self.top_k <= 0 {
            return Err(invalid_marlin_moe_args(format!(
                "top_k must be positive, got {}",
                self.top_k
            )));
        }
        if self.prob_m.checked_mul(self.top_k).is_none() {
            return Err(invalid_marlin_moe_args(
                "prob_m * top_k overflows the kernel's i32 output-row domain",
            ));
        }
        if self.prob_n % 64 != 0 {
            return Err(invalid_marlin_moe_args(format!(
                "prob_n {} must be divisible by the Marlin minimum thread width 64",
                self.prob_n
            )));
        }
        if self.prob_k % 64 != 0 {
            return Err(invalid_marlin_moe_args(format!(
                "prob_k {} must be divisible by the Marlin minimum thread width 64",
                self.prob_k
            )));
        }
        match self.weight_type {
            MarlinMoeF16WeightType::U4B8 => {
                if self.group_size != -1 {
                    if self.group_size <= 0 || self.group_size % 16 != 0 {
                        return Err(invalid_marlin_moe_args(format!(
                            "group_size must be -1 or a positive multiple of 16, got {}",
                            self.group_size
                        )));
                    }
                    if self.prob_k % self.group_size != 0 {
                        return Err(invalid_marlin_moe_args(format!(
                            "prob_k {} must be divisible by group_size {}",
                            self.prob_k, self.group_size
                        )));
                    }
                }
            }
            MarlinMoeF16WeightType::E4M3 => {
                if !matches!(self.group_size, -1 | 128) {
                    return Err(invalid_marlin_moe_args(format!(
                        "E4M3 group_size must be -1 or 128, got {}",
                        self.group_size
                    )));
                }
                if self.group_size == 128 && self.prob_k % 128 != 0 {
                    return Err(invalid_marlin_moe_args(format!(
                        "E4M3 prob_k {} must be divisible by group_size 128",
                        self.prob_k
                    )));
                }
            }
        }
        if self.device_ordinal < 0 {
            return Err(invalid_marlin_moe_args(format!(
                "device_ordinal must be non-negative, got {}",
                self.device_ordinal
            )));
        }
        if self.has_zero_points != self.zero_points.is_some() {
            return Err(invalid_marlin_moe_args(
                "has_zero_points must exactly match the zero_points pointer",
            ));
        }
        if self.weight_type == MarlinMoeF16WeightType::E4M3
            && (self.has_zero_points || self.zero_points.is_some())
        {
            return Err(invalid_marlin_moe_args("E4M3 forbids zero points"));
        }
        if self.mul_topk_weights && self.topk_weights.is_none() {
            return Err(invalid_marlin_moe_args(
                "mul_topk_weights requires a non-null topk_weights pointer",
            ));
        }
        if self.use_atomic_add == self.use_fp32_reduce {
            return Err(invalid_marlin_moe_args(
                "exactly one of use_atomic_add and use_fp32_reduce must be enabled",
            ));
        }
        if self.use_fp32_reduce != self.c_tmp.is_some() {
            return Err(invalid_marlin_moe_args(
                "use_fp32_reduce must exactly match the c_tmp pointer",
            ));
        }
        Ok(())
    }
}

/// Native MXFP4 encoding accepted by the BF16 Marlin-MoE entrypoint.
///
/// The packed weight contains two E2M1 values per byte and uses one E8M0
/// scale byte for every 32 values along K. Keeping this separate from
/// `MarlinMoeF16WeightType` prevents the existing FP16 U4/E4M3 ABI from
/// accidentally selecting the BF16-only entrypoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MarlinMoeMxfp4WeightType {
    E2M1E8M0,
}

/// Raw, allocation-agnostic arguments for BF16 x MXFP4 Marlin-MoE.
///
/// `a`, `c`, and `bias` must point to BF16 data. `b` is Marlin-packed E2M1
/// nibble data and `scales` is the corresponding group-32 E8M0 byte data.
/// The owning provider must retain every allocation until work enqueued on
/// `stream` has completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MarlinMoeMxfp4Bf16LaunchArgs {
    pub(crate) weight_type: MarlinMoeMxfp4WeightType,
    pub(crate) expert_count: i32,
    pub(crate) a: cudarc::driver::sys::CUdeviceptr,
    pub(crate) b: cudarc::driver::sys::CUdeviceptr,
    pub(crate) c: cudarc::driver::sys::CUdeviceptr,
    pub(crate) c_tmp: Option<cudarc::driver::sys::CUdeviceptr>,
    pub(crate) bias: cudarc::driver::sys::CUdeviceptr,
    pub(crate) scales: cudarc::driver::sys::CUdeviceptr,
    pub(crate) workspace: cudarc::driver::sys::CUdeviceptr,
    pub(crate) sorted_token_ids: cudarc::driver::sys::CUdeviceptr,
    pub(crate) expert_ids: cudarc::driver::sys::CUdeviceptr,
    pub(crate) num_tokens_past_padded: cudarc::driver::sys::CUdeviceptr,
    pub(crate) topk_weights: Option<cudarc::driver::sys::CUdeviceptr>,
    pub(crate) moe_block_size: i32,
    pub(crate) top_k: i32,
    pub(crate) mul_topk_weights: bool,
    pub(crate) is_ep: bool,
    pub(crate) prob_m: i32,
    pub(crate) prob_n: i32,
    pub(crate) prob_k: i32,
    pub(crate) group_size: i32,
    pub(crate) device_ordinal: i32,
    pub(crate) use_atomic_add: bool,
    pub(crate) use_fp32_reduce: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MarlinMoeMxfp4Bf16RequiredBytes {
    a: u64,
    b: u64,
    c: u64,
    c_tmp: Option<u64>,
    bias: u64,
    scales: u64,
    topk_weights: Option<u64>,
}

impl MarlinMoeMxfp4Bf16LaunchArgs {
    fn validate(&self) -> candle_core::Result<MarlinMoeMxfp4Bf16RequiredBytes> {
        validate_marlin_moe_pointer("a", self.a, 16)?;
        validate_marlin_moe_pointer("b", self.b, 16)?;
        validate_marlin_moe_pointer("c", self.c, 16)?;
        validate_marlin_moe_pointer("bias", self.bias, 16)?;
        validate_marlin_moe_pointer("scales", self.scales, 16)?;
        validate_marlin_moe_pointer("workspace", self.workspace, 4)?;
        validate_marlin_moe_pointer("sorted_token_ids", self.sorted_token_ids, 4)?;
        validate_marlin_moe_pointer("expert_ids", self.expert_ids, 4)?;
        validate_marlin_moe_pointer("num_tokens_past_padded", self.num_tokens_past_padded, 4)?;
        if let Some(pointer) = self.c_tmp {
            validate_marlin_moe_pointer("c_tmp", pointer, 16)?;
        }
        if let Some(pointer) = self.topk_weights {
            validate_marlin_moe_pointer("topk_weights", pointer, 4)?;
        }

        if self.expert_count <= 0 {
            return Err(invalid_marlin_moe_args(format!(
                "MXFP4 expert_count must be positive, got {}",
                self.expert_count
            )));
        }
        if self.prob_m <= 0 || self.prob_n <= 0 || self.prob_k <= 0 {
            return Err(invalid_marlin_moe_args(format!(
                "MXFP4 BF16 prob_m, prob_n, and prob_k must be positive, got [{}, {}, {}]",
                self.prob_m, self.prob_n, self.prob_k
            )));
        }
        if !matches!(self.moe_block_size, 8 | 16 | 32 | 48 | 64) {
            return Err(invalid_marlin_moe_args(format!(
                "unsupported moe_block_size {}; expected one of 8, 16, 32, 48, 64",
                self.moe_block_size
            )));
        }
        if self.top_k <= 0 || self.top_k > self.expert_count {
            return Err(invalid_marlin_moe_args(format!(
                "MXFP4 top_k must be in 1..=expert_count, got top_k={} expert_count={}",
                self.top_k, self.expert_count
            )));
        }
        let output_rows = self.prob_m.checked_mul(self.top_k).ok_or_else(|| {
            invalid_marlin_moe_args(
                "MXFP4 prob_m * top_k overflows the kernel's i32 output-row domain",
            )
        })?;
        if self.prob_n % 64 != 0 {
            return Err(invalid_marlin_moe_args(format!(
                "MXFP4 prob_n {} must be divisible by the Marlin minimum thread width 64",
                self.prob_n
            )));
        }
        let expected_group_size = match self.weight_type {
            MarlinMoeMxfp4WeightType::E2M1E8M0 => 32,
        };
        if self.group_size != expected_group_size {
            return Err(invalid_marlin_moe_args(format!(
                "MXFP4 E2M1/E8M0 group_size must be exactly {expected_group_size}, got {}",
                self.group_size,
            )));
        }
        if self.prob_k % expected_group_size != 0 {
            return Err(invalid_marlin_moe_args(format!(
                "MXFP4 prob_k {} must be divisible by group_size {expected_group_size}",
                self.prob_k,
            )));
        }
        if self.device_ordinal < 0 {
            return Err(invalid_marlin_moe_args(format!(
                "device_ordinal must be non-negative, got {}",
                self.device_ordinal
            )));
        }
        if self.mul_topk_weights && self.topk_weights.is_none() {
            return Err(invalid_marlin_moe_args(
                "mul_topk_weights requires a non-null topk_weights pointer",
            ));
        }
        if self.use_atomic_add == self.use_fp32_reduce {
            return Err(invalid_marlin_moe_args(
                "exactly one of use_atomic_add and use_fp32_reduce must be enabled",
            ));
        }
        if self.use_fp32_reduce != self.c_tmp.is_some() {
            return Err(invalid_marlin_moe_args(
                "use_fp32_reduce must exactly match the c_tmp pointer",
            ));
        }

        let experts = self.expert_count as u64;
        let m = self.prob_m as u64;
        let n = self.prob_n as u64;
        let k = self.prob_k as u64;
        let output_rows = output_rows as u64;
        let required = MarlinMoeMxfp4Bf16RequiredBytes {
            a: checked_marlin_moe_bytes("MXFP4 BF16 input [M,K]", &[m, k, 2])?,
            b: checked_marlin_moe_bytes(
                "MXFP4 E2M1 packed weight [E,N,K/2]",
                &[experts, n, k / 2],
            )?,
            c: checked_marlin_moe_bytes("MXFP4 BF16 output [M*top_k,N]", &[output_rows, n, 2])?,
            c_tmp: self
                .c_tmp
                .map(|_| {
                    checked_marlin_moe_bytes(
                        "MXFP4 FP32 reduction scratch [M*top_k,N]",
                        &[output_rows, n, 4],
                    )
                })
                .transpose()?,
            bias: checked_marlin_moe_bytes("MXFP4 BF16 bias [E,N]", &[experts, n, 2])?,
            scales: checked_marlin_moe_bytes(
                "MXFP4 E8M0 scales [E,K/32,N]",
                &[experts, k / 32, n],
            )?,
            topk_weights: self
                .topk_weights
                .map(|_| {
                    checked_marlin_moe_bytes("MXFP4 top-k weights [M,top_k]", &[output_rows, 4])
                })
                .transpose()?,
        };

        validate_marlin_moe_span("a", self.a, required.a)?;
        validate_marlin_moe_span("b", self.b, required.b)?;
        validate_marlin_moe_span("c", self.c, required.c)?;
        validate_marlin_moe_span("bias", self.bias, required.bias)?;
        validate_marlin_moe_span("scales", self.scales, required.scales)?;
        if let (Some(pointer), Some(bytes)) = (self.c_tmp, required.c_tmp) {
            validate_marlin_moe_span("c_tmp", pointer, bytes)?;
        }
        if let (Some(pointer), Some(bytes)) = (self.topk_weights, required.topk_weights) {
            validate_marlin_moe_span("topk_weights", pointer, bytes)?;
        }
        Ok(required)
    }
}

fn checked_marlin_moe_bytes(label: &str, factors: &[u64]) -> candle_core::Result<u64> {
    factors.iter().try_fold(1_u64, |bytes, factor| {
        bytes.checked_mul(*factor).ok_or_else(|| {
            invalid_marlin_moe_args(format!("{label} byte size overflows the u64 device domain"))
        })
    })
}

fn validate_marlin_moe_span(
    name: &str,
    pointer: cudarc::driver::sys::CUdeviceptr,
    bytes: u64,
) -> candle_core::Result<()> {
    debug_assert!(bytes > 0);
    if pointer.checked_add(bytes - 1).is_none() {
        return Err(invalid_marlin_moe_args(format!(
            "{name} pointer range overflows the u64 device domain"
        )));
    }
    Ok(())
}

fn validate_marlin_moe_pointer(
    name: &str,
    pointer: cudarc::driver::sys::CUdeviceptr,
    alignment: u64,
) -> candle_core::Result<()> {
    if pointer == 0 {
        return Err(invalid_marlin_moe_args(format!(
            "{name} pointer must be non-null"
        )));
    }
    if pointer % alignment != 0 {
        return Err(invalid_marlin_moe_args(format!(
            "{name} pointer 0x{pointer:x} must be aligned to {alignment} bytes"
        )));
    }
    Ok(())
}

fn invalid_marlin_moe_args(reason: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("invalid vLLM Marlin-MoE launch: {reason}"))
}

#[cfg(feature = "vllm-moe-marlin")]
pub(crate) fn launch_marlin_moe_vllm_raw(
    stream: &CudaStream,
    args: MarlinMoeRawLaunchArgs,
) -> candle_core::Result<()> {
    args.validate()?;
    let (entrypoint, ret) = unsafe {
        match args.weight_type {
            MarlinMoeF16WeightType::U4B8 => (
                "ferrum_vllm_marlin_moe_f16",
                ferrum_vllm_marlin_moe_f16(
                    args.a as *const _,
                    args.b as *const _,
                    args.c as *mut _,
                    args.c_tmp.unwrap_or_default() as *mut _,
                    args.scales as *const _,
                    args.zero_points.unwrap_or_default() as *const _,
                    args.workspace as *mut _,
                    args.sorted_token_ids as *const i32,
                    args.expert_ids as *const i32,
                    args.num_tokens_past_padded as *const i32,
                    args.topk_weights.unwrap_or_default() as *const f32,
                    args.moe_block_size,
                    args.top_k,
                    i32::from(args.mul_topk_weights),
                    i32::from(args.is_ep),
                    args.prob_m,
                    args.prob_n,
                    args.prob_k,
                    args.group_size,
                    i32::from(args.has_zero_points),
                    args.device_ordinal,
                    stream.cu_stream(),
                    i32::from(args.use_atomic_add),
                    i32::from(args.use_fp32_reduce),
                ),
            ),
            MarlinMoeF16WeightType::E4M3 => (
                "ferrum_vllm_marlin_moe_fp8_f16",
                ferrum_vllm_marlin_moe_fp8_f16(
                    args.a as *const _,
                    args.b as *const _,
                    args.c as *mut _,
                    args.c_tmp.unwrap_or_default() as *mut _,
                    args.scales as *const _,
                    std::ptr::null(),
                    args.workspace as *mut _,
                    args.sorted_token_ids as *const i32,
                    args.expert_ids as *const i32,
                    args.num_tokens_past_padded as *const i32,
                    args.topk_weights.unwrap_or_default() as *const f32,
                    args.moe_block_size,
                    args.top_k,
                    i32::from(args.mul_topk_weights),
                    i32::from(args.is_ep),
                    args.prob_m,
                    args.prob_n,
                    args.prob_k,
                    args.group_size,
                    0,
                    args.device_ordinal,
                    stream.cu_stream(),
                    i32::from(args.use_atomic_add),
                    i32::from(args.use_fp32_reduce),
                ),
            ),
        }
    };
    if ret != 0 {
        let (stage, cuda_status) = marlin_moe_ffi_status(ret);
        return Err(candle_core::Error::Msg(format!(
            "{entrypoint} failed at {stage}: \
             cuda_status={cuda_status}, ret={ret} (m={}, n={}, k={})",
            args.prob_m, args.prob_n, args.prob_k
        )));
    }
    Ok(())
}

/// Validate and enqueue the BF16 x MXFP4 Marlin-MoE native entrypoint.
#[cfg(feature = "vllm-moe-marlin")]
pub(crate) fn launch_marlin_moe_mxfp4_bf16(
    stream: &CudaStream,
    args: MarlinMoeMxfp4Bf16LaunchArgs,
) -> candle_core::Result<()> {
    let _required = args.validate()?;
    let ret = unsafe {
        ferrum_vllm_marlin_moe_mxfp4_bf16(
            args.a as *const _,
            args.b as *const _,
            args.c as *mut _,
            args.c_tmp.unwrap_or_default() as *mut _,
            args.bias as *const _,
            args.scales as *const _,
            args.workspace as *mut _,
            args.sorted_token_ids as *const i32,
            args.expert_ids as *const i32,
            args.num_tokens_past_padded as *const i32,
            args.topk_weights.unwrap_or_default() as *const f32,
            args.moe_block_size,
            args.top_k,
            i32::from(args.mul_topk_weights),
            i32::from(args.is_ep),
            args.prob_m,
            args.prob_n,
            args.prob_k,
            args.group_size,
            args.device_ordinal,
            stream.cu_stream(),
            i32::from(args.use_atomic_add),
            i32::from(args.use_fp32_reduce),
        )
    };
    if ret != 0 {
        let (stage, cuda_status) = marlin_moe_ffi_status(ret);
        return Err(candle_core::Error::Msg(format!(
            "ferrum_vllm_marlin_moe_mxfp4_bf16 failed at {stage}: \
             cuda_status={cuda_status}, ret={ret} (m={}, n={}, k={}, experts={})",
            args.prob_m, args.prob_n, args.prob_k, args.expert_count
        )));
    }
    Ok(())
}

#[cfg(not(feature = "vllm-moe-marlin"))]
pub(crate) fn launch_marlin_moe_mxfp4_bf16(
    _stream: &CudaStream,
    _args: MarlinMoeMxfp4Bf16LaunchArgs,
) -> candle_core::Result<()> {
    Err(candle_core::Error::Msg(
        "vLLM MXFP4 BF16 Marlin-MoE not built — compile with --features vllm-moe-marlin".into(),
    ))
}

#[cfg(not(feature = "vllm-moe-marlin"))]
pub(crate) fn launch_marlin_moe_vllm_raw(
    _stream: &CudaStream,
    _args: MarlinMoeRawLaunchArgs,
) -> candle_core::Result<()> {
    Err(candle_core::Error::Msg(
        "vLLM marlin_moe_wna16 not built — compile with --features vllm-moe-marlin".into(),
    ))
}

/// Stage 14 - fused MoE Marlin via the vLLM marlin_moe_wna16 native artifact
/// kernel. Replaces our Stage 12.1 bucketed `marlin_gemm_moe` with a
/// single launch that processes ALL `(token, expert)` pairs of a layer
/// in one go using vLLM's `(sorted_token_ids, expert_ids)` indirection.
///
/// vLLM's design eliminates the m=16 padding waste of our Stage 12.1
/// path: each output tile reads its expert id from the per-tile
/// `expert_ids[block_idx]` array, gathers its 16 input rows via
/// `sorted_token_ids[block_idx*moe_block_size .. ]`, and accumulates
/// directly. Inactive (sentinel) rows are masked out without compute.
///
/// Caller must:
/// - Run `B::moe_align_block_size` first to build sorted_token_ids,
///   expert_ids, num_tokens_past_padded.
/// - Allocate output `c[size_m * top_k, size_n]` fp16.
/// - Provide a stacked Marlin-packed weight tile (the same one our
///   per-expert `marlin_gemm_with_offset` consumes).
/// - Pre-zero the workspace (or rely on `marlin_zero_stacked_workspace`).
///
/// `prob_m = size_m` (number of original input tokens), `prob_n` =
/// per-expert n, `prob_k` = k. Inputs are flat across all experts; the
/// kernel routes per-tile via expert_ids.
///
/// Only available with `--features vllm-moe-marlin`.
#[cfg(feature = "vllm-moe-marlin")]
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm_moe_vllm(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<half::f16>,
    weight: &MarlinWeight,
    output: &mut CudaSlice<half::f16>,
    c_tmp: Option<&mut CudaSlice<f32>>,
    sorted_token_ids: &CudaSlice<i32>,
    expert_ids: &CudaSlice<i32>,
    num_tokens_past_padded: &CudaSlice<i32>,
    topk_weights: Option<&CudaSlice<f32>>,
    moe_block_size: i32,
    top_k: i32,
    mul_topk_weights: bool,
    is_ep: bool,
    prob_m: i32,
    prob_n: i32,
    prob_k: i32,
) -> candle_core::Result<()> {
    use cudarc::driver::DevicePtr;
    let raw_stream = stream.cu_stream();
    let profile = profile_marlin();
    let profile_bucket = profile.then(current_marlin_profile_bucket);

    let (a_ptr, _ag) = input.device_ptr(stream);
    let (b_ptr, _bg) = weight.qweight.device_ptr(stream);
    let (c_ptr, _cg) = output.device_ptr(stream);
    let (s_ptr, _sg) = weight.scales.device_ptr(stream);
    let z_ptr = match weight.qzeros.as_ref() {
        Some(z) => Some(z.device_ptr(stream).0),
        None => None,
    };
    let (ws_ptr, _wg) = weight.workspace.device_ptr(stream);
    let (st_ptr, _stg) = sorted_token_ids.device_ptr(stream);
    let (eid_ptr, _eidg) = expert_ids.device_ptr(stream);
    let (npp_ptr, _nppg) = num_tokens_past_padded.device_ptr(stream);

    let c_tmp_ptr = match c_tmp.as_ref() {
        Some(c) => Some(c.device_ptr(stream).0),
        None => None,
    };
    let topk_w_ptr = match topk_weights {
        Some(w) => Some(w.device_ptr(stream).0),
        None => None,
    };

    let timer = profile
        .then(|| CudaMarlinEventTimer::start(raw_stream))
        .flatten();
    let result = launch_marlin_moe_vllm_raw(
        stream,
        MarlinMoeRawLaunchArgs {
            weight_type: MarlinMoeF16WeightType::U4B8,
            a: a_ptr,
            b: b_ptr,
            c: c_ptr,
            c_tmp: c_tmp_ptr,
            scales: s_ptr,
            zero_points: z_ptr,
            workspace: ws_ptr,
            sorted_token_ids: st_ptr,
            expert_ids: eid_ptr,
            num_tokens_past_padded: npp_ptr,
            topk_weights: topk_w_ptr,
            moe_block_size,
            top_k,
            mul_topk_weights,
            is_ep,
            prob_m,
            prob_n,
            prob_k,
            group_size: weight.group_size,
            has_zero_points: weight.qzeros.is_some(),
            device_ordinal: 0,
            use_atomic_add: c_tmp_ptr.is_none(),
            use_fp32_reduce: c_tmp_ptr.is_some(),
        },
    );
    if let Some(timer) = timer {
        let elapsed_us = timer.finish_us(raw_stream);
        MARLIN_KERNEL_TIME_US.fetch_add(elapsed_us, Ordering::Relaxed);
        MARLIN_KERNEL_CALLS.fetch_add(1, Ordering::Relaxed);
        if let Some(bucket) = profile_bucket {
            record_marlin_kernel(bucket, elapsed_us);
        }
    }
    result
}

#[cfg(not(feature = "vllm-moe-marlin"))]
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm_moe_vllm(
    _stream: &Arc<CudaStream>,
    _input: &CudaSlice<half::f16>,
    _weight: &MarlinWeight,
    _output: &mut CudaSlice<half::f16>,
    _c_tmp: Option<&mut CudaSlice<f32>>,
    _sorted_token_ids: &CudaSlice<i32>,
    _expert_ids: &CudaSlice<i32>,
    _num_tokens_past_padded: &CudaSlice<i32>,
    _topk_weights: Option<&CudaSlice<f32>>,
    _moe_block_size: i32,
    _top_k: i32,
    _mul_topk_weights: bool,
    _is_ep: bool,
    _prob_m: i32,
    _prob_n: i32,
    _prob_k: i32,
) -> candle_core::Result<()> {
    Err(candle_core::Error::Msg(
        "vLLM marlin_moe_wna16 not built — compile with --features vllm-moe-marlin".into(),
    ))
}

pub use crate::marlin_repack::{
    permute_gptq_qweight_rows, repack_gptq_to_marlin, repack_scales_to_marlin,
};

#[cfg(test)]
mod tests {
    #[cfg(feature = "vllm-moe-marlin")]
    use super::{launch_marlin_moe_mxfp4_bf16, launch_marlin_moe_vllm_raw};
    use super::{
        marlin_moe_ffi_status, marlin_profile_bucket_from_label, should_zero_workspace,
        CudaMarlinRuntimeConfig, MarlinMoeF16WeightType, MarlinMoeMxfp4Bf16LaunchArgs,
        MarlinMoeMxfp4WeightType, MarlinMoeRawLaunchArgs, MarlinProfileBucket,
        MarlinProfileBucketStats,
    };
    #[cfg(feature = "vllm-moe-marlin")]
    use crate::marlin_repack::{prepare_block_fp8_weight_for_fp8_marlin, repack_gptq_to_marlin};
    #[cfg(feature = "vllm-moe-marlin")]
    use crate::mxfp4_marlin_materializer::{
        prepare_mxfp4_expert_scales_for_marlin, transpose_mxfp4_expert_blocks_to_gptq_words,
    };
    #[cfg(feature = "vllm-moe-marlin")]
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
    #[cfg(feature = "vllm-moe-marlin")]
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
    #[cfg(feature = "vllm-moe-marlin")]
    use half::{bf16, f16};
    #[cfg(feature = "vllm-moe-marlin")]
    use sha2::{Digest, Sha256};

    fn valid_marlin_moe_raw_args() -> MarlinMoeRawLaunchArgs {
        MarlinMoeRawLaunchArgs {
            weight_type: MarlinMoeF16WeightType::U4B8,
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            c_tmp: None,
            scales: 0x4000,
            zero_points: None,
            workspace: 0x5000,
            sorted_token_ids: 0x6000,
            expert_ids: 0x7000,
            num_tokens_past_padded: 0x8000,
            topk_weights: None,
            moe_block_size: 16,
            top_k: 8,
            mul_topk_weights: false,
            is_ep: false,
            prob_m: 4,
            prob_n: 1024,
            prob_k: 2048,
            group_size: 128,
            has_zero_points: false,
            device_ordinal: 0,
            use_atomic_add: true,
            use_fp32_reduce: false,
        }
    }

    fn valid_marlin_moe_mxfp4_bf16_args() -> MarlinMoeMxfp4Bf16LaunchArgs {
        MarlinMoeMxfp4Bf16LaunchArgs {
            weight_type: MarlinMoeMxfp4WeightType::E2M1E8M0,
            expert_count: 4,
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            c_tmp: None,
            bias: 0x4000,
            scales: 0x5000,
            workspace: 0x6000,
            sorted_token_ids: 0x7000,
            expert_ids: 0x8000,
            num_tokens_past_padded: 0x9000,
            topk_weights: None,
            moe_block_size: 16,
            top_k: 2,
            mul_topk_weights: false,
            is_ep: false,
            prob_m: 4,
            prob_n: 512,
            prob_k: 256,
            group_size: 32,
            device_ordinal: 0,
            use_atomic_add: true,
            use_fp32_reduce: false,
        }
    }

    fn assert_invalid_marlin_moe_mxfp4_bf16_args(
        args: MarlinMoeMxfp4Bf16LaunchArgs,
        expected: &str,
    ) {
        let error = args.validate().expect_err("launch arguments must fail");
        assert!(
            error.to_string().contains(expected),
            "expected error containing {expected:?}, got {error}"
        );
    }

    #[test]
    fn marlin_moe_mxfp4_bf16_args_compute_required_buffer_shapes() {
        let required = valid_marlin_moe_mxfp4_bf16_args().validate().unwrap();
        assert_eq!(required.a, 2_048);
        assert_eq!(required.b, 262_144);
        assert_eq!(required.c, 8_192);
        assert_eq!(required.c_tmp, None);
        assert_eq!(required.bias, 4_096);
        assert_eq!(required.scales, 16_384);
        assert_eq!(required.topk_weights, None);

        let mut fp32_reduce = valid_marlin_moe_mxfp4_bf16_args();
        fp32_reduce.c_tmp = Some(0xa000);
        fp32_reduce.topk_weights = Some(0xb000);
        fp32_reduce.mul_topk_weights = true;
        fp32_reduce.use_atomic_add = false;
        fp32_reduce.use_fp32_reduce = true;
        let required = fp32_reduce.validate().unwrap();
        assert_eq!(required.c_tmp, Some(16_384));
        assert_eq!(required.topk_weights, Some(32));
    }

    #[test]
    fn marlin_moe_mxfp4_bf16_args_reject_invalid_pointers() {
        let mut args = valid_marlin_moe_mxfp4_bf16_args();
        args.bias = 0;
        assert_invalid_marlin_moe_mxfp4_bf16_args(args, "bias pointer must be non-null");

        let mut args = valid_marlin_moe_mxfp4_bf16_args();
        args.scales += 2;
        assert_invalid_marlin_moe_mxfp4_bf16_args(args, "scales pointer");

        let mut args = valid_marlin_moe_mxfp4_bf16_args();
        args.b = u64::MAX - 15;
        assert_invalid_marlin_moe_mxfp4_bf16_args(args, "b pointer range overflows");
    }

    #[test]
    fn marlin_moe_mxfp4_bf16_args_reject_invalid_shapes_and_overflow() {
        let mut args = valid_marlin_moe_mxfp4_bf16_args();
        args.group_size = 128;
        assert_invalid_marlin_moe_mxfp4_bf16_args(args, "group_size must be exactly 32");

        let mut args = valid_marlin_moe_mxfp4_bf16_args();
        args.prob_k = 240;
        assert_invalid_marlin_moe_mxfp4_bf16_args(args, "must be divisible by group_size 32");

        let mut args = valid_marlin_moe_mxfp4_bf16_args();
        args.prob_n = 96;
        assert_invalid_marlin_moe_mxfp4_bf16_args(args, "must be divisible");

        let mut args = valid_marlin_moe_mxfp4_bf16_args();
        args.top_k = 5;
        assert_invalid_marlin_moe_mxfp4_bf16_args(args, "1..=expert_count");

        let mut args = valid_marlin_moe_mxfp4_bf16_args();
        args.prob_m = i32::MAX;
        assert_invalid_marlin_moe_mxfp4_bf16_args(args, "prob_m * top_k overflows");

        let mut args = valid_marlin_moe_mxfp4_bf16_args();
        args.expert_count = i32::MAX;
        args.top_k = 1;
        args.prob_m = 1;
        args.prob_n = i32::MAX - i32::MAX.rem_euclid(64);
        args.prob_k = i32::MAX - i32::MAX.rem_euclid(32);
        assert_invalid_marlin_moe_mxfp4_bf16_args(args, "packed weight");
    }

    #[cfg(feature = "vllm-moe-marlin")]
    #[test]
    fn marlin_moe_mxfp4_bf16_extern_abi_matches_locked_export() {
        type ExpectedAbi = unsafe extern "C" fn(
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *mut std::ffi::c_void,
            *mut std::ffi::c_void,
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *mut std::ffi::c_void,
            *const i32,
            *const i32,
            *const i32,
            *const f32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            cudarc::driver::sys::CUstream,
            i32,
            i32,
        ) -> i32;

        let _: ExpectedAbi = super::ferrum_vllm_marlin_moe_mxfp4_bf16;
    }

    #[test]
    fn marlin_moe_ffi_status_preserves_failure_stage_and_cuda_status() {
        assert_eq!(marlin_moe_ffi_status((1 << 16) | 10), ("sm-count", 10));
        assert_eq!(
            marlin_moe_ffi_status((5 << 16) | 9),
            ("function-attribute", 9)
        );
        assert_eq!(
            marlin_moe_ffi_status((6 << 16) | 701),
            ("kernel-launch", 701)
        );
        assert_eq!(marlin_moe_ffi_status(17), ("unknown", 17));
    }

    fn assert_invalid_marlin_moe_args(args: MarlinMoeRawLaunchArgs, expected: &str) {
        let error = args.validate().expect_err("launch arguments must fail");
        assert!(
            error.to_string().contains(expected),
            "expected error containing {expected:?}, got {error}"
        );
    }

    #[test]
    fn marlin_moe_raw_args_accept_supported_modes() {
        valid_marlin_moe_raw_args().validate().unwrap();

        let mut fp32_reduce = valid_marlin_moe_raw_args();
        fp32_reduce.c_tmp = Some(0x9000);
        fp32_reduce.zero_points = Some(0xa000);
        fp32_reduce.topk_weights = Some(0xb000);
        fp32_reduce.has_zero_points = true;
        fp32_reduce.mul_topk_weights = true;
        fp32_reduce.use_atomic_add = false;
        fp32_reduce.use_fp32_reduce = true;
        fp32_reduce.validate().unwrap();

        let mut per_channel = valid_marlin_moe_raw_args();
        per_channel.group_size = -1;
        per_channel.validate().unwrap();

        let mut fp8 = valid_marlin_moe_raw_args();
        fp8.weight_type = MarlinMoeF16WeightType::E4M3;
        fp8.group_size = -1;
        fp8.validate().unwrap();

        let mut fp8_group128 = valid_marlin_moe_raw_args();
        fp8_group128.weight_type = MarlinMoeF16WeightType::E4M3;
        fp8_group128.group_size = 128;
        fp8_group128.validate().unwrap();
    }

    #[test]
    fn marlin_moe_raw_args_reject_invalid_pointers() {
        let mut args = valid_marlin_moe_raw_args();
        args.a = 0;
        assert_invalid_marlin_moe_args(args, "a pointer must be non-null");

        let mut args = valid_marlin_moe_raw_args();
        args.scales += 2;
        assert_invalid_marlin_moe_args(args, "scales pointer");

        let mut args = valid_marlin_moe_raw_args();
        args.topk_weights = Some(0xb002);
        assert_invalid_marlin_moe_args(args, "topk_weights pointer");
    }

    #[test]
    fn marlin_moe_raw_args_reject_invalid_shapes_and_config() {
        let mut args = valid_marlin_moe_raw_args();
        args.prob_m = 0;
        assert_invalid_marlin_moe_args(args, "must be positive");

        let mut args = valid_marlin_moe_raw_args();
        args.moe_block_size = 24;
        assert_invalid_marlin_moe_args(args, "unsupported moe_block_size");

        let mut args = valid_marlin_moe_raw_args();
        args.top_k = 0;
        assert_invalid_marlin_moe_args(args, "top_k must be positive");

        let mut args = valid_marlin_moe_raw_args();
        args.prob_m = i32::MAX;
        assert_invalid_marlin_moe_args(args, "prob_m * top_k overflows");

        let mut args = valid_marlin_moe_raw_args();
        args.prob_n = 96;
        assert_invalid_marlin_moe_args(args, "prob_n 96 must be divisible");

        let mut args = valid_marlin_moe_raw_args();
        args.prob_k = 96;
        args.group_size = -1;
        assert_invalid_marlin_moe_args(args, "prob_k 96 must be divisible");

        let mut args = valid_marlin_moe_raw_args();
        args.group_size = 0;
        assert_invalid_marlin_moe_args(args, "group_size must be -1");

        let mut args = valid_marlin_moe_raw_args();
        args.group_size = 96;
        assert_invalid_marlin_moe_args(args, "must be divisible by group_size");

        let mut args = valid_marlin_moe_raw_args();
        args.device_ordinal = -1;
        assert_invalid_marlin_moe_args(args, "device_ordinal must be non-negative");
    }

    #[test]
    fn marlin_moe_raw_args_reject_inconsistent_optional_modes() {
        let mut args = valid_marlin_moe_raw_args();
        args.has_zero_points = true;
        assert_invalid_marlin_moe_args(args, "has_zero_points must exactly match");

        let mut args = valid_marlin_moe_raw_args();
        args.mul_topk_weights = true;
        assert_invalid_marlin_moe_args(args, "requires a non-null topk_weights");

        let mut args = valid_marlin_moe_raw_args();
        args.use_fp32_reduce = true;
        assert_invalid_marlin_moe_args(args, "exactly one");

        let mut args = valid_marlin_moe_raw_args();
        args.use_atomic_add = false;
        assert_invalid_marlin_moe_args(args, "exactly one");

        let mut args = valid_marlin_moe_raw_args();
        args.c_tmp = Some(0x9000);
        assert_invalid_marlin_moe_args(args, "use_fp32_reduce must exactly match");

        for group_size in [0, 16, 64, 256] {
            let mut args = valid_marlin_moe_raw_args();
            args.weight_type = MarlinMoeF16WeightType::E4M3;
            args.group_size = group_size;
            assert_invalid_marlin_moe_args(args, "E4M3 group_size must be -1 or 128");
        }

        let mut args = valid_marlin_moe_raw_args();
        args.weight_type = MarlinMoeF16WeightType::E4M3;
        args.prob_k = 192;
        assert_invalid_marlin_moe_args(args, "must be divisible by group_size 128");

        let mut args = valid_marlin_moe_raw_args();
        args.weight_type = MarlinMoeF16WeightType::E4M3;
        args.group_size = -1;
        args.zero_points = Some(0xa000);
        args.has_zero_points = true;
        assert_invalid_marlin_moe_args(args, "E4M3 forbids zero points");
    }

    #[test]
    #[ignore = "requires an sm89 CUDA host and the GPT-OSS MXFP4 Marlin-MoE native artifact"]
    #[cfg(feature = "vllm-moe-marlin")]
    fn gpt_oss_mxfp4_marlin_moe_bf16_ffi_matches_source_reference_for_four_cases() {
        const MOE_BLOCK_SIZE: usize = 16;
        const LCG_MULTIPLIER: u64 = 0x5851_f42d_4c95_7f2d;
        const LCG_INCREMENT: u64 = 0x1405_7b7e_f767_814f;
        const ROOT_SEED: u64 = 0x4750_544f_5353_4d58;
        const SHAPE_SEED_XOR: u64 = 0x9e37_79b9_7f4a_7c15;
        const WEIGHT_SEED_XOR: u64 = 0x5745_4947_4854_5f31;
        const SCALE_SEED_XOR: u64 = 0x5343_414c_455f_5f31;
        const ACTIVATION_SEED_XOR: u64 = 0x4143_5449_5641_5445;
        const QUALITY_VECTOR_DEFINITION: &str = concat!(
            "gpt-oss-mxfp4-marlin-moe-v1;",
            "cases=gate-up-512x256-b1,b4|down-256x512-b1,b4;",
            "root_seed=0x4750544f53534d58;",
            "lcg=0x5851f42d4c957f2d+0x14057b7ef767814f;",
            "stream_xors=0x9e3779b97f4a7c15,0x5745494748545f31,",
            "0x5343414c455f5f31,0x4143544956415445;",
            "activation=bf16[-1,1];source=e2m1-low-nibble-first/e8m0-group32;",
            "decode=e2m1-table[0,0.5,1,1.5,2,3,4,6]*2^(e8m0-127);",
            "bias=bf16;accumulator=f32;output=bf16;",
            "reference=source-decoded-matmul-plus-bias;",
            "relative_l2_max=0.05;nan=0;inf=0",
        );
        const QUALITY_VECTOR_DIGEST: &str =
            "7b8d4908cbee9c68aa4ff4c47c5e883bf76788250cfbb334603dbbd746218b21";
        const CASES: [(&str, usize, usize, usize, usize); 4] = [
            ("gate-up-512x256-batch-1", 0, 512, 256, 1),
            ("gate-up-512x256-batch-4", 0, 512, 256, 4),
            ("down-256x512-batch-1", 1, 256, 512, 1),
            ("down-256x512-batch-4", 1, 256, 512, 4),
        ];

        fn next(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            *state
        }

        fn stream_seed(shape_index: usize, stream_xor: u64) -> u64 {
            let mut state = ROOT_SEED
                ^ (u64::try_from(shape_index + 1)
                    .expect("shape index fits u64")
                    .wrapping_mul(SHAPE_SEED_XOR))
                ^ stream_xor;
            next(&mut state)
        }

        fn decode_e2m1(bits: u8) -> f32 {
            const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            let magnitude = MAGNITUDES[usize::from(bits & 0x07)];
            if bits & 0x08 == 0 {
                magnitude
            } else {
                -magnitude
            }
        }

        let context = CudaContext::new(0).expect("CUDA context");
        let stream = context.default_stream();
        assert_eq!(
            format!("{:x}", Sha256::digest(QUALITY_VECTOR_DEFINITION.as_bytes())),
            QUALITY_VECTOR_DIGEST,
            "GPT-OSS MXFP4 quality-vector definition changed without a digest update"
        );
        let sms = usize::try_from(
            context
                .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .expect("query CUDA SM count"),
        )
        .expect("CUDA SM count must be positive");

        for (case_id, shape_index, n, k, batch) in CASES {
            let mut activation_state = stream_seed(shape_index, ACTIVATION_SEED_XOR);
            let input = (0..batch * k)
                .map(|_| {
                    let signed = i32::try_from(next(&mut activation_state) % 129)
                        .expect("activation residue fits i32")
                        - 64;
                    bf16::from_f32(signed as f32 / 64.0)
                })
                .collect::<Vec<_>>();

            let mut weight_state = stream_seed(shape_index, WEIGHT_SEED_XOR);
            let source_blocks = (0..n * k / 2)
                .map(|_| {
                    let even =
                        u8::try_from(next(&mut weight_state) & 0x0f).expect("E2M1 nibble fits u8");
                    let odd =
                        u8::try_from(next(&mut weight_state) & 0x0f).expect("E2M1 nibble fits u8");
                    even | (odd << 4)
                })
                .collect::<Vec<_>>();

            let groups_per_row = k / 32;
            let mut scale_state = stream_seed(shape_index, SCALE_SEED_XOR);
            let source_scales = (0..n * groups_per_row)
                .map(|_| {
                    125_u8
                        + u8::try_from(next(&mut scale_state) % 5).expect("E8M0 scale tier fits u8")
                })
                .collect::<Vec<_>>();
            let bias = (0..n)
                .map(|output| bf16::from_f32((output as i32 % 7 - 3) as f32 / 32.0))
                .collect::<Vec<_>>();

            let mut reference = vec![0.0_f32; batch * n];
            for row in 0..batch {
                for output in 0..n {
                    let mut sum = bias[output].to_f32();
                    for input_feature in 0..k {
                        let packed = source_blocks[output * (k / 2) + input_feature / 2];
                        let nibble = if input_feature.is_multiple_of(2) {
                            packed & 0x0f
                        } else {
                            packed >> 4
                        };
                        let exponent =
                            i32::from(source_scales[output * groups_per_row + input_feature / 32])
                                - 127;
                        let decoded_weight = decode_e2m1(nibble) * 2.0_f32.powi(exponent);
                        sum += input[row * k + input_feature].to_f32() * decoded_weight;
                    }
                    reference[row * n + output] = sum;
                }
            }

            let gptq_words = transpose_mxfp4_expert_blocks_to_gptq_words(&source_blocks, n, k)
                .expect("transpose source MXFP4 nibbles");
            let packed_weight = repack_gptq_to_marlin(&gptq_words, k, n);
            let packed_scales = prepare_mxfp4_expert_scales_for_marlin(&source_scales, n, k)
                .expect("prepare source E8M0 scales");

            let input_device: CudaSlice<bf16> = stream.clone_htod(&input).expect("upload input");
            let weight_device: CudaSlice<i32> = stream
                .clone_htod(&packed_weight)
                .expect("upload packed MXFP4 weight");
            let scales_device: CudaSlice<u8> = stream
                .clone_htod(&packed_scales)
                .expect("upload packed E8M0 scales");
            let bias_device: CudaSlice<bf16> = stream.clone_htod(&bias).expect("upload bias");
            let mut output_device: CudaSlice<bf16> = stream
                .alloc_zeros(batch * n)
                .expect("allocate MXFP4 output");
            let mut reduce_device: CudaSlice<f32> = stream
                .alloc_zeros(batch * n)
                .expect("allocate MXFP4 reduction scratch");

            let mut sorted_token_ids = vec![i32::try_from(batch).unwrap(); MOE_BLOCK_SIZE];
            for (token, sorted) in sorted_token_ids.iter_mut().take(batch).enumerate() {
                *sorted = i32::try_from(token).expect("token index fits i32");
            }
            let sorted_token_ids_device: CudaSlice<i32> = stream
                .clone_htod(&sorted_token_ids)
                .expect("upload padded sorted token ids");
            let expert_ids_device: CudaSlice<i32> =
                stream.clone_htod(&[0]).expect("upload expert block id");
            let num_tokens_past_padded_device: CudaSlice<i32> = stream
                .clone_htod(&[i32::try_from(MOE_BLOCK_SIZE).unwrap()])
                .expect("upload padded token count");
            let workspace: CudaSlice<i32> = stream
                .alloc_zeros(n.div_ceil(128) * sms * 4)
                .expect("allocate Marlin-MoE workspace");

            {
                let (input_pointer, _input_guard) = input_device.device_ptr(&stream);
                let (weight_pointer, _weight_guard) = weight_device.device_ptr(&stream);
                let (output_pointer, _output_guard) = output_device.device_ptr_mut(&stream);
                let (reduce_pointer, _reduce_guard) = reduce_device.device_ptr_mut(&stream);
                let (bias_pointer, _bias_guard) = bias_device.device_ptr(&stream);
                let (scales_pointer, _scales_guard) = scales_device.device_ptr(&stream);
                let (workspace_pointer, _workspace_guard) = workspace.device_ptr(&stream);
                let (sorted_pointer, _sorted_guard) = sorted_token_ids_device.device_ptr(&stream);
                let (expert_pointer, _expert_guard) = expert_ids_device.device_ptr(&stream);
                let (padded_pointer, _padded_guard) =
                    num_tokens_past_padded_device.device_ptr(&stream);

                launch_marlin_moe_mxfp4_bf16(
                    &stream,
                    MarlinMoeMxfp4Bf16LaunchArgs {
                        weight_type: MarlinMoeMxfp4WeightType::E2M1E8M0,
                        expert_count: 1,
                        a: input_pointer,
                        b: weight_pointer,
                        c: output_pointer,
                        c_tmp: Some(reduce_pointer),
                        bias: bias_pointer,
                        scales: scales_pointer,
                        workspace: workspace_pointer,
                        sorted_token_ids: sorted_pointer,
                        expert_ids: expert_pointer,
                        num_tokens_past_padded: padded_pointer,
                        topk_weights: None,
                        moe_block_size: i32::try_from(MOE_BLOCK_SIZE).unwrap(),
                        top_k: 1,
                        mul_topk_weights: false,
                        is_ep: false,
                        prob_m: i32::try_from(batch).unwrap(),
                        prob_n: i32::try_from(n).unwrap(),
                        prob_k: i32::try_from(k).unwrap(),
                        group_size: 32,
                        device_ordinal: 0,
                        use_atomic_add: false,
                        use_fp32_reduce: true,
                    },
                )
                .expect("launch GPT-OSS MXFP4 Marlin-MoE");
                stream
                    .synchronize()
                    .expect("synchronize GPT-OSS MXFP4 Marlin-MoE");
            }

            let actual = stream
                .clone_dtoh(&output_device)
                .expect("download GPT-OSS MXFP4 output");
            let mut reference_squared = 0.0_f64;
            let mut error_squared = 0.0_f64;
            let mut nan_count = 0_usize;
            let mut infinity_count = 0_usize;
            for (actual, expected) in actual.iter().zip(reference.iter().copied()) {
                let actual = actual.to_f32();
                reference_squared += f64::from(expected) * f64::from(expected);
                if actual.is_nan() {
                    nan_count += 1;
                } else if actual.is_infinite() {
                    infinity_count += 1;
                } else {
                    let error = f64::from(actual - expected);
                    error_squared += error * error;
                }
            }
            let relative_l2 = if nan_count == 0 && infinity_count == 0 {
                error_squared.sqrt() / reference_squared.sqrt().max(1.0e-6)
            } else {
                f64::INFINITY
            };

            eprintln!(
                "GPT_OSS_MXFP4_MARLIN_MOE_FFI_FIXTURE name={case_id} \
                 quality_vector_digest={QUALITY_VECTOR_DIGEST} \
                 rel_err={relative_l2:.8} nan_count={nan_count} \
                 infinity_count={infinity_count}"
            );
            assert_eq!(nan_count, 0, "{case_id} emitted NaN");
            assert_eq!(infinity_count, 0, "{case_id} emitted Inf");
            assert!(
                relative_l2 <= 0.05,
                "{case_id} rel_err={relative_l2:.8} exceeds 0.05"
            );
        }
    }

    #[test]
    #[ignore = "requires an sm89 CUDA host and the FP8 Marlin-MoE native artifact"]
    #[cfg(feature = "vllm-moe-marlin")]
    fn qwen36_a3_fp8_marlin_moe_ffi_matches_cpu_reference_for_four_cases() {
        const BLOCK_SHAPE: [usize; 2] = [128, 128];
        const MOE_BLOCK_SIZE: usize = 16;
        const LCG_MULTIPLIER: u64 = 0x5851_f42d_4c95_7f2d;
        const LCG_INCREMENT: u64 = 0x1405_7b7e_f767_814f;
        const ROOT_SEED: u64 = 0x5147_454e_3338_4650;
        const SHAPE_SEED_XOR: u64 = 0x9e37_79b9_7f4a_7c15;
        const WEIGHT_SEED_XOR: u64 = 0x5745_4947_4854_5f31;
        const SCALE_SEED_XOR: u64 = 0x5343_414c_455f_5f31;
        const ACTIVATION_SEED_XOR: u64 = 0x4143_5449_5641_5445;
        const CASES: [(&str, usize, usize, usize, usize); 4] = [
            ("weight-256x128-batch-1", 0, 256, 128, 1),
            ("weight-256x128-batch-4", 0, 256, 128, 4),
            ("weight-256x256-batch-1", 1, 256, 256, 1),
            ("weight-256x256-batch-4", 1, 256, 256, 4),
        ];

        fn next(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(LCG_MULTIPLIER)
                .wrapping_add(LCG_INCREMENT);
            *state
        }

        fn stream_seed(shape_index: usize, stream_xor: u64) -> u64 {
            let mut state = ROOT_SEED
                ^ (u64::try_from(shape_index + 1)
                    .expect("shape index fits u64")
                    .wrapping_mul(SHAPE_SEED_XOR))
                ^ stream_xor;
            next(&mut state)
        }

        let context = CudaContext::new(0).expect("CUDA context");
        let stream = context.default_stream();
        let sms = usize::try_from(
            context
                .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .expect("query CUDA SM count"),
        )
        .expect("CUDA SM count must be positive");

        for (case_id, shape_index, n, k, batch) in CASES {
            let mut activation_state = stream_seed(shape_index, ACTIVATION_SEED_XOR);
            let input = (0..batch * k)
                .map(|_| {
                    let signed = i32::try_from(next(&mut activation_state) % 129)
                        .expect("activation residue fits i32")
                        - 64;
                    f16::from_f32(signed as f32 / 64.0)
                })
                .collect::<Vec<_>>();

            let scale_rows = n.div_ceil(BLOCK_SHAPE[0]);
            let scale_columns = k.div_ceil(BLOCK_SHAPE[1]);
            let mut scale_state = stream_seed(shape_index, SCALE_SEED_XOR);
            let inverse_scales = (0..scale_rows * scale_columns)
                .map(|_| {
                    bf16::from_bits(
                        0x3b80
                            + 0x20
                                * u16::try_from(next(&mut scale_state) % 5)
                                    .expect("scale residue fits u16"),
                    )
                })
                .collect::<Vec<_>>();
            let inverse_scale_bytes = inverse_scales
                .iter()
                .flat_map(|scale| scale.to_le_bytes())
                .collect::<Vec<_>>();

            let mut weight_state = stream_seed(shape_index, WEIGHT_SEED_XOR);
            let source_values = (0..n * k)
                .map(|index| {
                    let word = next(&mut weight_state);
                    if word & 0x0f == 0 {
                        0
                    } else {
                        let output_channel = index / k;
                        let exponent_tier =
                            u8::try_from((output_channel % 8) / 2).expect("exponent tier fits u8");
                        let magnitude = 0x20 + exponent_tier * 8 + ((word >> 9) & 0x07) as u8;
                        let sign = ((word >> 31) & 0x80) as u8;
                        magnitude | sign
                    }
                })
                .collect::<Vec<_>>();

            let mut reference = vec![0.0_f32; batch * n];
            for row in 0..batch {
                for output in 0..n {
                    let mut sum = 0.0_f32;
                    for input_feature in 0..k {
                        let source =
                            float8::F8E4M3::from_bits(source_values[output * k + input_feature])
                                .to_f32();
                        let scale_index = (output / BLOCK_SHAPE[0]) * scale_columns
                            + input_feature / BLOCK_SHAPE[1];
                        let decoded_weight = source * inverse_scales[scale_index].to_f32();
                        sum += input[row * k + input_feature].to_f32() * decoded_weight;
                    }
                    reference[row * n + output] = sum;
                }
            }

            let prepared = prepare_block_fp8_weight_for_fp8_marlin(
                &source_values,
                &inverse_scale_bytes,
                n,
                k,
                BLOCK_SHAPE,
            )
            .expect("prepare block-FP8 Marlin-MoE weight");
            let (packed_weight, packed_scales) = prepared.into_parts();

            let input_device: CudaSlice<f16> = stream.clone_htod(&input).expect("upload input");
            let weight_device: CudaSlice<u8> = stream
                .clone_htod(&packed_weight)
                .expect("upload packed FP8 weight");
            let scales_device: CudaSlice<f16> = stream
                .clone_htod(&packed_scales)
                .expect("upload packed FP8 scales");
            let mut output_device: CudaSlice<f16> = stream
                .alloc_zeros(batch * n)
                .expect("allocate Marlin-MoE output");

            // One active expert owns one 16-row block. Actual token ids occupy
            // the prefix and every remaining row uses the vLLM sentinel
            // `prob_m * top_k` so the kernel skips padded rows.
            let mut sorted_token_ids = vec![i32::try_from(batch).unwrap(); MOE_BLOCK_SIZE];
            for (token, sorted) in sorted_token_ids.iter_mut().take(batch).enumerate() {
                *sorted = i32::try_from(token).expect("token index fits i32");
            }
            let sorted_token_ids_device: CudaSlice<i32> = stream
                .clone_htod(&sorted_token_ids)
                .expect("upload padded sorted token ids");
            let expert_ids_device: CudaSlice<i32> =
                stream.clone_htod(&[0]).expect("upload expert block id");
            let num_tokens_past_padded_device: CudaSlice<i32> = stream
                .clone_htod(&[i32::try_from(MOE_BLOCK_SIZE).unwrap()])
                .expect("upload padded token count");
            let workspace: CudaSlice<i32> = stream
                .alloc_zeros(n.div_ceil(128) * sms * 4)
                .expect("allocate Marlin-MoE workspace");

            {
                let (input_pointer, _input_guard) = input_device.device_ptr(&stream);
                let (weight_pointer, _weight_guard) = weight_device.device_ptr(&stream);
                let (output_pointer, _output_guard) = output_device.device_ptr_mut(&stream);
                let (scales_pointer, _scales_guard) = scales_device.device_ptr(&stream);
                let (workspace_pointer, _workspace_guard) = workspace.device_ptr(&stream);
                let (sorted_pointer, _sorted_guard) = sorted_token_ids_device.device_ptr(&stream);
                let (expert_pointer, _expert_guard) = expert_ids_device.device_ptr(&stream);
                let (padded_pointer, _padded_guard) =
                    num_tokens_past_padded_device.device_ptr(&stream);

                launch_marlin_moe_vllm_raw(
                    &stream,
                    MarlinMoeRawLaunchArgs {
                        weight_type: MarlinMoeF16WeightType::E4M3,
                        a: input_pointer,
                        b: weight_pointer,
                        c: output_pointer,
                        c_tmp: None,
                        scales: scales_pointer,
                        zero_points: None,
                        workspace: workspace_pointer,
                        sorted_token_ids: sorted_pointer,
                        expert_ids: expert_pointer,
                        num_tokens_past_padded: padded_pointer,
                        topk_weights: None,
                        moe_block_size: i32::try_from(MOE_BLOCK_SIZE).unwrap(),
                        top_k: 1,
                        mul_topk_weights: false,
                        is_ep: false,
                        prob_m: i32::try_from(batch).unwrap(),
                        prob_n: i32::try_from(n).unwrap(),
                        prob_k: i32::try_from(k).unwrap(),
                        group_size: -1,
                        has_zero_points: false,
                        device_ordinal: 0,
                        use_atomic_add: true,
                        use_fp32_reduce: false,
                    },
                )
                .expect("launch ferrum_vllm_marlin_moe_fp8_f16");
                stream.synchronize().expect("synchronize FP8 Marlin-MoE");
            }

            let actual = stream
                .clone_dtoh(&output_device)
                .expect("download FP8 Marlin-MoE output");
            let mut reference_squared = 0.0_f64;
            let mut error_squared = 0.0_f64;
            let mut nan_count = 0_usize;
            let mut infinity_count = 0_usize;
            for (actual, expected) in actual.iter().zip(reference.iter().copied()) {
                let actual = actual.to_f32();
                reference_squared += f64::from(expected) * f64::from(expected);
                if actual.is_nan() {
                    nan_count += 1;
                } else if actual.is_infinite() {
                    infinity_count += 1;
                } else {
                    let error = f64::from(actual - expected);
                    error_squared += error * error;
                }
            }
            let relative_l2 = if nan_count == 0 && infinity_count == 0 {
                error_squared.sqrt() / reference_squared.sqrt().max(1.0e-6)
            } else {
                f64::INFINITY
            };

            eprintln!(
                "QWEN36_A3_FP8_MARLIN_MOE_FFI_FIXTURE name={case_id} \
                 rel_err={relative_l2:.8} nan_count={nan_count} \
                 infinity_count={infinity_count}"
            );
            assert_eq!(nan_count, 0, "{case_id} emitted NaN");
            assert_eq!(infinity_count, 0, "{case_id} emitted Inf");
            assert!(
                relative_l2 <= 0.05,
                "{case_id} rel_err={relative_l2:.8} exceeds 0.05"
            );
        }
    }

    #[test]
    fn cuda_marlin_runtime_config_parses_skip_ws_zero() {
        let config = CudaMarlinRuntimeConfig::from_env_vars([
            ("FERRUM_MARLIN_PROFILE", "1"),
            ("FERRUM_MARLIN_SKIP_WS_ZERO", "1"),
            ("FERRUM_MARLIN_TRACE_SHAPES", "1"),
            ("FERRUM_MARLIN_TRACE_SHAPES_MAX", "17"),
        ]);
        assert!(config.profile);
        assert!(config.skip_ws_zero);
        assert!(config.trace_shapes);
        assert_eq!(config.trace_shapes_max, 17);
    }

    #[test]
    fn cuda_marlin_runtime_config_defaults_to_zero_workspace() {
        let config = CudaMarlinRuntimeConfig::from_env_vars([
            ("FERRUM_MARLIN_PROFILE", "true"),
            ("FERRUM_MARLIN_SKIP_WS_ZERO", "true"),
            ("FERRUM_MARLIN_TRACE_SHAPES", "true"),
            ("FERRUM_MARLIN_TRACE_SHAPES_MAX", "not-a-number"),
        ]);
        assert!(!config.profile);
        assert!(!config.skip_ws_zero);
        assert!(!config.trace_shapes);
        assert_eq!(config.trace_shapes_max, 256);
    }

    #[test]
    fn marlin_workspace_zeroing_follows_runtime_config() {
        let default_config = CudaMarlinRuntimeConfig::from_env_vars(Vec::<(&str, &str)>::new());
        assert!(should_zero_workspace(&default_config));

        let skip_config =
            CudaMarlinRuntimeConfig::from_env_vars([("FERRUM_MARLIN_SKIP_WS_ZERO", "1")]);
        assert!(!should_zero_workspace(&skip_config));
    }

    #[test]
    fn marlin_profile_bucket_labels_match_projection_names() {
        assert_eq!(
            marlin_profile_bucket_from_label("label=llama.batched_layer.qkv_proj"),
            MarlinProfileBucket::Qkv
        );
        assert_eq!(
            marlin_profile_bucket_from_label("label=llama.forward_layer.o_proj"),
            MarlinProfileBucket::OProj
        );
        assert_eq!(
            marlin_profile_bucket_from_label("label=llama.forward_layer.gate_up_proj"),
            MarlinProfileBucket::GateUp
        );
        assert_eq!(
            marlin_profile_bucket_from_label("label=llama.forward_layer.down_proj"),
            MarlinProfileBucket::Down
        );
        assert_eq!(
            marlin_profile_bucket_from_label("label=llama.batched.lm_head"),
            MarlinProfileBucket::LmHead
        );
        assert_eq!(
            marlin_profile_bucket_from_label("label=<none>"),
            MarlinProfileBucket::Other
        );
    }

    #[test]
    fn marlin_profile_bucket_stats_record_all_profile_phases() {
        let mut stats = MarlinProfileBucketStats::ZERO;

        stats.record_ws_zero(3);
        stats.record_gather(5);
        stats.record_kernel(7);

        assert_eq!(stats.ws_zero_us, 3);
        assert_eq!(stats.ws_zero_calls, 1);
        assert_eq!(stats.gather_us, 5);
        assert_eq!(stats.gather_calls, 1);
        assert_eq!(stats.kernel_us, 7);
        assert_eq!(stats.kernel_calls, 1);
    }
}
