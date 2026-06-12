//! `BackendQuantMarlin` + `BackendQuantGguf` for `CudaBackend` —
//! GPTQ INT4 GEMM (Marlin / vLLM-marlin / Triton-rs w4a16) and the
//! GGUF k-quant placeholder.
//!
//! Extracted from `cuda/mod.rs` (#8 Phase 3). Owns:
//!
//! - `GptqStoreCuda` enum + cfg-gated alias — chooses Marlin (default)
//!   vs Triton-rs w4a16 store at runtime via `FERRUM_TRITON_INT4=1`.
//! - `use_triton_int4()` / `use_vllm_moe()` env-var dispatch toggles.
//! - `MarlinGatherScratch` + `marlin_gather_scratch_slots` +
//!   `with_marlin_gather_scratch` + `pregrow_marlin_gather_scratch_helper`
//!   — per-device staging buffers for desc_act perm-aware Marlin
//!   (avoids per-call cudarc alloc that grew unboundedly).
//! - `moe_gemm_phase_fused_impl` (Stage 12 fused MoE Marlin one-launch-per-bucket).
//! - `marlin_gemm_with_perm` — perm-aware dispatcher used by both
//!   `BackendQuantMarlin` and `quant_linear::cuda_marlin::CudaMarlinLinear`.
//! - `launch_vllm_marlin` (vendored vLLM marlin path) + cfg(not) shim.
//! - `impl BackendQuantMarlin for CudaBackend` — the trait body itself.
//! - `impl BackendQuantGguf for CudaBackend {}` — empty (CUDA has no
//!   k-quant kernels yet, inherits trait defaults).
//!
//! `mod.rs` re-exports `GptqStoreCuda` + `marlin_gemm_with_perm` to
//! preserve the historical `crate::backend::cuda::*` paths used by
//! `quant_linear::cuda_marlin` and parity tests.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use ferrum_types::{FerrumError, Result};
use half::f16;

use super::{current_device_ordinal, default_stream, CudaBackend, CudaState};
use crate::backend::{Backend, BackendQuantGguf, BackendQuantMarlin};
use crate::ptx;

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
    cuda_quant_runtime_config().triton_int4
}

/// Read `FERRUM_VLLM_MOE` once. Returns true iff `=1`. Selects the
/// vendored vLLM marlin_moe_wna16 path for stacked-MoE GPTQ INT4
/// weights (load + dispatch pair must be enabled together).
#[cfg(feature = "vllm-moe-marlin")]
pub(crate) fn use_vllm_moe() -> bool {
    cuda_quant_runtime_config().vllm_moe
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CudaQuantRuntimeConfig {
    triton_int4: bool,
    vllm_moe: bool,
    vllm_marlin: bool,
    vllm_marlin_sms: i32,
    vllm_atomic_add: bool,
    vllm_fp32_reduce: bool,
    moe_fused: bool,
    moe_streams: usize,
}

impl CudaQuantRuntimeConfig {
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
            triton_int4: false,
            vllm_moe: false,
            vllm_marlin: false,
            vllm_marlin_sms: 128,
            vllm_atomic_add: false,
            vllm_fp32_reduce: false,
            moe_fused: true,
            moe_streams: 4,
        };

        for (name, value) in vars {
            let value = value.as_ref();
            match name.as_ref() {
                "FERRUM_TRITON_INT4" => config.triton_int4 = value == "1",
                "FERRUM_VLLM_MOE" => config.vllm_moe = value == "1",
                "FERRUM_VLLM_MARLIN" => config.vllm_marlin = value == "1",
                "FERRUM_VLLM_MARLIN_SMS" => {
                    if let Ok(sms) = value.parse::<i32>() {
                        config.vllm_marlin_sms = sms;
                    }
                }
                "FERRUM_VLLM_ATOMIC_ADD" => config.vllm_atomic_add = value == "1",
                "FERRUM_VLLM_FP32_REDUCE" => config.vllm_fp32_reduce = value == "1",
                "FERRUM_MOE_FUSED" => config.moe_fused = value != "0",
                "FERRUM_MOE_STREAMS" => {
                    if let Ok(streams) = value.parse::<usize>() {
                        config.moe_streams = streams.max(1);
                    }
                }
                _ => {}
            }
        }

        config
    }
}

fn cuda_quant_runtime_config() -> &'static CudaQuantRuntimeConfig {
    static CONFIG: OnceLock<CudaQuantRuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(CudaQuantRuntimeConfig::from_env)
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

static MARLIN_GATHER_SCRATCH: std::sync::OnceLock<
    std::sync::RwLock<HashMap<usize, MarlinGatherScratch>>,
> = std::sync::OnceLock::new();

fn marlin_gather_scratch_slots() -> &'static std::sync::RwLock<HashMap<usize, MarlinGatherScratch>>
{
    MARLIN_GATHER_SCRATCH.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
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
    pregrow_marlin_gather_scratch_for_ordinal(current_device_ordinal(), stream, required);
}

fn pregrow_marlin_gather_scratch_for_ordinal(
    ordinal: usize,
    stream: &Arc<CudaStream>,
    required: usize,
) {
    let slots = marlin_gather_scratch_slots();
    {
        let g = slots.read().expect("MARLIN_GATHER_SCRATCH poisoned");
        if let Some(s) = g.get(&ordinal) {
            if s.capacity >= required {
                return;
            }
        }
    }
    let mut w = slots.write().expect("MARLIN_GATHER_SCRATCH poisoned");
    let need_new = match w.get(&ordinal) {
        Some(s) => s.capacity < required,
        None => true,
    };
    if need_new {
        let buf = unsafe { stream.alloc::<f16>(required) }
            .expect("MARLIN_GATHER_SCRATCH pregrow alloc failed");
        w.insert(
            ordinal,
            MarlinGatherScratch {
                buf,
                capacity: required,
            },
        );
    }
}

fn with_marlin_gather_scratch<R>(
    stream: &Arc<CudaStream>,
    ordinal: usize,
    required: usize,
    body: impl FnOnce(&mut CudaSlice<f16>) -> R,
) -> R {
    let slots = marlin_gather_scratch_slots();
    {
        let g = slots.read().expect("MARLIN_GATHER_SCRATCH poisoned");
        if let Some(s) = g.get(&ordinal) {
            if s.capacity >= required {
                drop(g);
                let mut w = slots.write().expect("MARLIN_GATHER_SCRATCH poisoned");
                let s = w.get_mut(&ordinal).expect("just observed Some");
                return body(&mut s.buf);
            }
        }
    }
    // Need to (re)allocate.
    let mut w = slots.write().expect("MARLIN_GATHER_SCRATCH poisoned");
    let need_new = match w.get(&ordinal) {
        Some(s) => s.capacity < required,
        None => true,
    };
    if need_new {
        let buf =
            unsafe { stream.alloc::<f16>(required) }.expect("MARLIN_GATHER_SCRATCH alloc failed");
        w.insert(
            ordinal,
            MarlinGatherScratch {
                buf,
                capacity: required,
            },
        );
    }
    let s = w.get_mut(&ordinal).expect("just allocated");
    body(&mut s.buf)
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
    let use_vllm = cuda_quant_runtime_config().vllm_marlin;

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
        with_marlin_gather_scratch(&stream, ctx.ordinal, m * k, |a_gathered| -> Result<()> {
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
    let runtime_config = cuda_quant_runtime_config();
    let sms = runtime_config.vllm_marlin_sms;
    // vLLM perf knobs — try toggling via env to see if either helps.
    let use_atomic_add = runtime_config.vllm_atomic_add;
    let use_fp32_reduce = runtime_config.vllm_fp32_reduce;

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
impl BackendQuantMarlin for CudaBackend {
    fn pregrow_marlin_gather_scratch(ctx: &mut Self::Context, required: usize) {
        #[cfg(feature = "marlin")]
        {
            let stream = ctx.stream.clone();
            pregrow_marlin_gather_scratch_for_ordinal(ctx.ordinal, &stream, required);
        }
        #[cfg(not(feature = "marlin"))]
        {
            let _ = (ctx, required);
        }
    }
    // Phase C step 4e: CUDA gemm_gptq_with_offset_strided dead code
    // removed. The remaining caller (moe_gemm_phase_batched_impl
    // serial/multi-stream path below) calls marlin_gemm_with_offset_strided
    // directly.
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

        // Marlin's smallest instantiated tile is 128x128 (m <= 16; larger
        // m chunks down to it in marlin_gemm), so n and k must both be
        // multiples of 128. Refuse anything else loudly — silently
        // garbled layers are worse than a load error.
        if n % 128 != 0 || k % 128 != 0 {
            return Err(FerrumError::unsupported(format!(
                "CUDA GPTQ: layer shape K={k} N={n} violates Marlin tile \
                 constraints (both must be multiples of 128)"
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
    ) -> Result<std::sync::Arc<dyn crate::MarlinExpertStack<Self>>> {
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
            let mw = crate::vllm_marlin::load_stacked_gptq_vllm_marlin(
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
            // Phase C step 4e: wrap in trait-object MarlinExpertStack.
            #[cfg(feature = "triton-kernels")]
            let store: GptqStoreCuda = GptqStoreCuda::Marlin(mw);
            #[cfg(not(feature = "triton-kernels"))]
            let store: GptqStoreCuda = mw;
            return Ok(std::sync::Arc::new(
                crate::quant_linear::cuda_marlin_stack::CudaMarlinExpertStack::new(
                    std::sync::Arc::new(store),
                    num_experts,
                    n_per_expert,
                    k,
                ),
            ));
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
        let store: GptqStoreCuda = GptqStoreCuda::Marlin(marlin_weight);
        #[cfg(not(feature = "triton-kernels"))]
        let store: GptqStoreCuda = marlin_weight;
        Ok(std::sync::Arc::new(
            crate::quant_linear::cuda_marlin_stack::CudaMarlinExpertStack::new(
                std::sync::Arc::new(store),
                num_experts,
                n_per_expert,
                k,
            ),
        ))
    }
    // Phase C step 4b: make_stacked_expert_linear inlined into
    // CudaMarlinExpertStack::make_expert_linear.
    // Phase C step 4e: make_marlin_expert_stack subsumed by load_gptq_stacked
    // (now returns the trait object directly).
    // Phase C step 4a: marlin_zero_stacked_workspace inlined into
    // CudaMarlinExpertStack::zero_workspace (quant_linear/cuda_marlin_stack.rs).
    // Phase C step 4c/4d: moe_gemm_phase_batched / moe_gemm_phase_vllm
    // moved to free functions below + inlined into the trait-object impl.
}

/// Free-function moved out of `BackendQuantMarlin::moe_gemm_phase_batched`
/// (Phase C step 4c). Bucketed/multi-stream per-expert Marlin GEMM
/// dispatch. Called by `CudaMarlinExpertStack::gemm_phase_batched`.
#[cfg(feature = "marlin")]
pub(crate) fn moe_gemm_phase_batched_impl(
    ctx: &mut CudaState,
    input: &<CudaBackend as crate::backend::Backend>::Buffer,
    weight: &GptqStoreCuda,
    dispatches: &[(usize, usize, usize, usize)],
    n_per_expert: usize,
    output: &mut <CudaBackend as crate::backend::Backend>::Buffer,
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
    let runtime_config = cuda_quant_runtime_config();
    if runtime_config.moe_fused {
        return moe_gemm_phase_fused_impl(
            ctx,
            input.as_f16(),
            mw,
            dispatches,
            n_per_expert,
            output.as_f16_mut(),
            k,
        );
    }

    // n_streams=1: serial dispatch on the DEFAULT context stream
    // (avoids the cross-stream sync overhead and matches the
    // pre-multi-stream path bit-for-bit).
    let n_streams = runtime_config.moe_streams;
    if n_streams == 1 {
        let default_stream = ctx.stream.clone();
        for (expert_idx, in_row_offset, out_row_offset, m) in dispatches {
            crate::marlin::marlin_gemm_with_offset_strided(
                &default_stream,
                input.as_f16(),
                *in_row_offset as i32,
                mw,
                output.as_f16_mut(),
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
            input.as_f16(),
            *in_row_offset as i32,
            mw,
            output.as_f16_mut(),
            *out_row_offset as i32,
            *m as i32,
            (expert_idx * n_per_expert) as i32,
            n_per_expert as i32,
        )
        .map_err(|e| FerrumError::model(format!("marlin offset_strided: {e}")))?;
    }

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

/// Free-function moved out of `BackendQuantMarlin::moe_gemm_phase_vllm`
/// (Phase C step 4d). Fused vLLM `marlin_moe_wna16` dispatch.
/// Called by `CudaMarlinExpertStack::gemm_phase_vllm`.
#[cfg(feature = "vllm-moe-marlin")]
pub(crate) fn moe_gemm_phase_vllm_impl(
    ctx: &mut CudaState,
    input: &<CudaBackend as crate::backend::Backend>::Buffer,
    weight: &GptqStoreCuda,
    sorted_token_ids: &<CudaBackend as crate::backend::Backend>::Buffer,
    expert_ids: &<CudaBackend as crate::backend::Backend>::Buffer,
    num_tokens_past_padded: &<CudaBackend as crate::backend::Backend>::Buffer,
    output: &mut <CudaBackend as crate::backend::Backend>::Buffer,
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

    let stream = ctx.stream.clone();

    // Phase D step 2+3: upload_moe_routing now returns CudaBuf::I32
    // directly — no more upgrade_device_ptr leak. Pass the typed
    // CudaSlice<i32> refs straight to the kernel wrapper.
    let st_ref = sorted_token_ids.as_i32();
    let eid_ref = expert_ids.as_i32();
    let npp_ref = num_tokens_past_padded.as_i32();

    // c_tmp moved process-global (was per-ctx lazy-alloc). Captured
    // graph kernel args stay valid across `new_context()` calls now.
    crate::backend::cuda::with_vllm_moe_c_tmp(&stream, ctx.ordinal, |c_tmp_mut| {
        crate::marlin::marlin_gemm_moe_vllm(
            &stream,
            input.as_f16(),
            mw,
            output.as_f16_mut(),
            Some(c_tmp_mut),
            st_ref,
            eid_ref,
            npp_ref,
            None,
            moe_block_size as i32,
            top_k as i32,
            false,
            false,
            prob_m as i32,
            n_per_expert as i32,
            k as i32,
        )
        .map_err(|e| FerrumError::model(format!("marlin_gemm_moe_vllm: {e}")))
    })
}
// CUDA does not ship GGUF k-quant kernels; inherit unsupported defaults.
impl BackendQuantGguf for CudaBackend {}

#[cfg(test)]
mod tests {
    use super::CudaQuantRuntimeConfig;

    #[test]
    fn cuda_quant_runtime_config_parses_marlin_and_moe_knobs() {
        let config = CudaQuantRuntimeConfig::from_env_vars([
            ("FERRUM_TRITON_INT4", "1"),
            ("FERRUM_VLLM_MOE", "1"),
            ("FERRUM_VLLM_MARLIN", "1"),
            ("FERRUM_VLLM_MARLIN_SMS", "132"),
            ("FERRUM_VLLM_ATOMIC_ADD", "1"),
            ("FERRUM_VLLM_FP32_REDUCE", "1"),
            ("FERRUM_MOE_FUSED", "0"),
            ("FERRUM_MOE_STREAMS", "0"),
        ]);

        assert!(config.triton_int4);
        assert!(config.vllm_moe);
        assert!(config.vllm_marlin);
        assert_eq!(config.vllm_marlin_sms, 132);
        assert!(config.vllm_atomic_add);
        assert!(config.vllm_fp32_reduce);
        assert!(!config.moe_fused);
        assert_eq!(config.moe_streams, 1);
    }

    #[test]
    fn cuda_quant_runtime_config_keeps_existing_defaults() {
        let config = CudaQuantRuntimeConfig::from_env_vars([
            ("FERRUM_TRITON_INT4", "true"),
            ("FERRUM_VLLM_MARLIN_SMS", "bad"),
            ("FERRUM_MOE_STREAMS", "bad"),
        ]);

        assert!(!config.triton_int4);
        assert!(!config.vllm_moe);
        assert!(!config.vllm_marlin);
        assert_eq!(config.vllm_marlin_sms, 128);
        assert!(!config.vllm_atomic_add);
        assert!(!config.vllm_fp32_reduce);
        assert!(config.moe_fused);
        assert_eq!(config.moe_streams, 4);
    }
}
