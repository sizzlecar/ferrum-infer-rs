//! Marlin INT4xFP16 fused GEMM kernel (IST Austria).
//!
//! Near-ideal 3.9x speedup over FP16 cuBLAS for INT4 quantized weights.
//! Weights must be in Marlin packed format (different from GPTQ).
//!
//! Constraints: K % 128 == 0, N % 256 == 0, SM >= 8.0 (Ampere+).

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

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

// vLLM marlin_moe_wna16 port (Stage 14). Vendored kernel under
// `crates/ferrum-kernels/kernels/vllm_marlin_moe/`. Single fused
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
        if profile {
            stream
                .synchronize()
                .map_err(|e| candle_core::Error::Msg(format!("marlin ws profile pre-sync: {e}")))?;
        }
        let t0 = profile.then(std::time::Instant::now);
        let (ws_ptr, _guard) = weight.workspace.device_ptr(stream);
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(ws_ptr, 0, weight.workspace.len(), raw_stream);
        }
        if let Some(t0) = t0 {
            stream.synchronize().map_err(|e| {
                candle_core::Error::Msg(format!("marlin ws profile post-sync: {e}"))
            })?;
            let elapsed_us = t0.elapsed().as_micros() as u64;
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

    if profile {
        stream
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("marlin kernel profile pre-sync: {e}")))?;
    }
    let t0 = profile.then(std::time::Instant::now);
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
    if let Some(t0) = t0 {
        stream.synchronize().map_err(|e| {
            candle_core::Error::Msg(format!("marlin kernel profile post-sync: {e}"))
        })?;
        let elapsed_us = t0.elapsed().as_micros() as u64;
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

/// Stage 14 — fused MoE Marlin via the vendored vLLM marlin_moe_wna16
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

    let (a_ptr, _ag) = input.device_ptr(stream);
    let (b_ptr, _bg) = weight.qweight.device_ptr(stream);
    let (c_ptr, _cg) = output.device_ptr(stream);
    let (s_ptr, _sg) = weight.scales.device_ptr(stream);
    let z_ptr = match weight.qzeros.as_ref() {
        Some(z) => z.device_ptr(stream).0 as *const std::ffi::c_void,
        None => std::ptr::null(),
    };
    let (ws_ptr, _wg) = weight.workspace.device_ptr(stream);
    let (st_ptr, _stg) = sorted_token_ids.device_ptr(stream);
    let (eid_ptr, _eidg) = expert_ids.device_ptr(stream);
    let (npp_ptr, _nppg) = num_tokens_past_padded.device_ptr(stream);

    let c_tmp_ptr = match c_tmp.as_ref() {
        Some(c) => c.device_ptr(stream).0 as *mut std::ffi::c_void,
        None => std::ptr::null_mut(),
    };
    let topk_w_ptr = match topk_weights {
        Some(w) => w.device_ptr(stream).0 as *const f32,
        None => std::ptr::null(),
    };

    let ret = unsafe {
        ferrum_vllm_marlin_moe_f16(
            a_ptr as *const _,
            b_ptr as *const _,
            c_ptr as *mut _,
            c_tmp_ptr,
            s_ptr as *const _,
            z_ptr,
            ws_ptr as *mut _,
            st_ptr as *const _,
            eid_ptr as *const _,
            npp_ptr as *const _,
            topk_w_ptr,
            moe_block_size,
            top_k,
            if mul_topk_weights { 1 } else { 0 },
            if is_ep { 1 } else { 0 },
            prob_m,
            prob_n,
            prob_k,
            weight.group_size,
            if weight.qzeros.is_some() { 1 } else { 0 },
            0, // dev
            raw_stream,
            // Atomic-add path when c_tmp is null (fp32-reduce needs the
            // scratch buffer; passing it but having c_tmp=null makes the
            // kernel deref a null pointer → NaN / OOB).
            if c_tmp_ptr.is_null() { 1 } else { 0 }, // use_atomic_add
            if c_tmp_ptr.is_null() { 0 } else { 1 }, // use_fp32_reduce
        )
    };
    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "ferrum_vllm_marlin_moe_f16 failed: ret={ret} (m={prob_m}, n={prob_n}, k={prob_k})"
        )));
    }
    Ok(())
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

// ===================== Weight Repacking (GPTQ → Marlin) =====================

/// Permute GPTQ INT4 qweight rows by `perm` (one-time, CPU work at load).
/// Input qweight is [K/8, N] packed (8 INT4s per i32 along K). Output is
/// the same shape, with rows reordered: out[i, n] = in[perm[i], n] (after
/// unpacking + repacking). Used to put rows in g_idx-sorted order so the
/// downstream Marlin repack sees a layout where group_idx = i / group_size.
pub fn permute_gptq_qweight_rows(
    qweight_gptq: &[i32], // [K/8, N] packed
    perm: &[usize],       // [K]
    k: usize,
    n: usize,
) -> Vec<i32> {
    debug_assert_eq!(perm.len(), k);
    debug_assert_eq!(qweight_gptq.len(), (k / 8) * n);

    // Step 1: Unpack [K/8, N] → [K, N] (one INT4 per byte slot, low 4 bits).
    let mut kn = vec![0u8; k * n];
    let packed_rows = k / 8;
    for pr in 0..packed_rows {
        for col in 0..n {
            let packed = qweight_gptq[pr * n + col] as u32;
            for i in 0..8 {
                kn[(pr * 8 + i) * n + col] = ((packed >> (i * 4)) & 0xF) as u8;
            }
        }
    }

    // Step 2: Permute rows. out[i, n] = in[perm[i], n].
    let mut sorted = vec![0u8; k * n];
    for i in 0..k {
        let src_row = perm[i];
        for col in 0..n {
            sorted[i * n + col] = kn[src_row * n + col];
        }
    }

    // Step 3: Repack [K, N] → [K/8, N] (8 INT4s per i32 along K).
    let mut packed = vec![0i32; (k / 8) * n];
    for pr in 0..packed_rows {
        for col in 0..n {
            let mut word = 0u32;
            for i in 0..8 {
                word |= (sorted[(pr * 8 + i) * n + col] as u32) << (i * 4);
            }
            packed[pr * n + col] = word as i32;
        }
    }
    packed
}

/// Repack GPTQ INT4 weights to Marlin format on CPU.
///
/// GPTQ format: qweight [K/8, N] int32 (in_features packed, out_features columns)
/// Marlin format: [N/16, K*16/8] int32, tiled and permuted for tensor core access
///
/// Key: Marlin operates on [N, K] layout (out_features first, like PyTorch Linear.weight).
/// GPTQ stores [K, N]. Must transpose before tiling.
///
/// Reference: IST-DASLab/marlin __init__.py Layer.pack()
pub fn repack_gptq_to_marlin(
    qweight_gptq: &[i32], // [K/8, N]
    k: usize,
    n: usize,
) -> Vec<i32> {
    use rayon::prelude::*;

    // Step 1: Unpack GPTQ [K/8, N] → individual INT4 values [K, N].
    // Parallelize over packed_rows. Each packed row produces 8 output rows
    // of `n` u8s — disjoint output slices, fully independent.
    let _packed_rows = k / 8;
    let mut kn = vec![0u8; k * n];
    kn.par_chunks_mut(8 * n)
        .zip(qweight_gptq.par_chunks(n))
        .for_each(|(kn_block, qw_row)| {
            // qw_row is one packed row [n] i32; kn_block is 8 unpacked rows [8 * n] u8.
            for col in 0..n {
                let packed = qw_row[col];
                for i in 0..8 {
                    kn_block[i * n + col] = ((packed >> (i * 4)) & 0xF) as u8;
                }
            }
        });

    // Step 2: Tile [K, N] → [K/16, N/16, 16, 16].
    // tiled[tk * (n * tile) + tn * (tile*tile) + ik * tile + in_]
    //                         = kn[(tk*tile + ik) * n + (tn*tile + in_)]
    // Parallelize over tk (each tk owns a disjoint output range
    // tiled[tk * (n * tile) .. (tk+1) * (n * tile)]).
    let tile = 16;
    let _kt = k / tile;
    let nt = n / tile;
    let mut tiled = vec![0u8; k * n];
    tiled
        .par_chunks_mut(n * tile)
        .enumerate()
        .for_each(|(tk, tile_block)| {
            for tn in 0..nt {
                for ik in 0..tile {
                    for in_ in 0..tile {
                        let src = (tk * tile + ik) * n + (tn * tile + in_);
                        let dst = tn * (tile * tile) + ik * tile + in_;
                        tile_block[dst] = kn[src];
                    }
                }
            }
        });
    // Drop kn early — its memory can be reused for permuted/result.
    drop(kn);

    // Step 3: Apply _perm in blocks of 1024. Each block reads 1024 contiguous
    // u8s from `tiled` and writes 1024 to `permuted` via the perm table.
    // Blocks are disjoint in both src and dst → trivially parallel.
    let perm = build_marlin_perm();
    let total = k * n;
    let mut permuted = vec![0u8; total];
    permuted
        .par_chunks_mut(1024)
        .zip(tiled.par_chunks(1024))
        .for_each(|(out_blk, in_blk)| {
            for (dst, &src) in perm.iter().enumerate() {
                out_blk[dst] = in_blk[src];
            }
        });
    drop(tiled);

    // Step 4: Pack 8 INT4 → i32, output shape [N/16, K*2].
    // Each output i32 reads 8 contiguous u8s — independent.
    let packed_len = total / 8;
    let mut result = vec![0i32; packed_len];
    result
        .par_iter_mut()
        .zip(permuted.par_chunks_exact(8))
        .for_each(|(out, chunk)| {
            let mut word = 0u32;
            for (j, &b) in chunk.iter().enumerate() {
                word |= (b as u32) << (j * 4);
            }
            *out = word as i32;
        });

    result
}

/// Permute scales from GPTQ layout to Marlin access pattern.
///
/// GPTQ: [num_groups, N] row-major (groups along K, columns are out_features)
/// Marlin: [num_groups, N] but reshuffled to match the kernel's tile access.
///
/// Reference: IST-DASLab/marlin __init__.py _scale_perm / _scale_perm_single
pub fn repack_scales_to_marlin(
    scales_gptq: &[half::f16], // [num_groups, N]
    k: usize,
    n: usize,
    group_size: usize,
) -> Vec<half::f16> {
    let num_groups = k / group_size;

    // Build permutation table matching Marlin's scale access pattern
    let scale_perm: Vec<usize> = if num_groups > 1 {
        // Grouped quantization (group_size=128, group_blocks=8)
        // _scale_perm = [i + 8*j for i in range(8) for j in range(8)]
        (0..8)
            .flat_map(|i| (0..8).map(move |j| i + 8 * j))
            .collect()
    } else {
        // Per-channel (group_size=-1, group_blocks=-1)
        // _scale_perm_single = [2*i+j for i in range(4) for j in [0,1,8,9,16,17,24,25]]
        (0..4)
            .flat_map(|i| [0, 1, 8, 9, 16, 17, 24, 25].map(move |j| 2 * i + j))
            .collect()
    };

    // Flatten scales, apply permutation in blocks
    let total = num_groups * n;
    let perm_len = scale_perm.len();
    let mut result = vec![half::f16::ZERO; total];

    // Reshape scales as flat array, permute in blocks of perm_len
    for blk in 0..(total / perm_len) {
        let base = blk * perm_len;
        for (dst, &src) in scale_perm.iter().enumerate() {
            result[base + dst] = scales_gptq[base + src];
        }
    }
    // Remainder (if total not divisible by perm_len)
    let rem_start = (total / perm_len) * perm_len;
    for i in rem_start..total {
        result[i] = scales_gptq[i];
    }
    result
}

/// Build the 1024-element Marlin weight permutation array.
///
/// This encodes the m16n8k16 tensor core mma fragment layout.
/// Each 1024-element block of the tiled weight [N/16, K*16] is
/// permuted to match how the Marlin kernel loads data into
/// tensor core fragments via shared memory.
///
/// Reference: IST-DASLab/marlin __init__.py _perm construction
fn build_marlin_perm() -> Vec<usize> {
    let mut perm = Vec::with_capacity(1024);

    for i in 0..32 {
        let col = i / 4;
        let mut perm1 = Vec::with_capacity(8);

        for _block in 0..2 {
            for &row_off in &[0, 1, 8, 9] {
                let row = 2 * (i % 4) + row_off / 8 * 8 + row_off % 8;
                // Actually, the original Python is:
                // for row in [2*(i%4), 2*(i%4)+1, 2*(i%4+4), 2*(i%4+4)+1]:
                //     perm1.append(16*row + col + 8*block)
                let _ = row; // ignore, use direct construction below
            }
        }

        // Direct from Python: for block in [0,1]: for row in [...]: perm1.append(...)
        perm1.clear();
        for block in 0..2 {
            for &row in &[
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1.push(16 * row + col + 8 * block);
            }
        }

        for j in 0..4 {
            for &p in &perm1 {
                perm.push(p + 256 * j);
            }
        }
    }

    assert_eq!(perm.len(), 1024);

    // KEY: apply interleave [0,2,4,6,1,3,5,7] within each group of 8
    let interleave = [0usize, 2, 4, 6, 1, 3, 5, 7];
    let mut perm_interleaved = vec![0usize; 1024];
    for g in 0..128 {
        for i in 0..8 {
            perm_interleaved[g * 8 + i] = perm[g * 8 + interleave[i]];
        }
    }

    perm_interleaved
}

#[cfg(test)]
mod tests {
    use super::{
        marlin_profile_bucket_from_label, should_zero_workspace, CudaMarlinRuntimeConfig,
        MarlinProfileBucket, MarlinProfileBucketStats,
    };

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
