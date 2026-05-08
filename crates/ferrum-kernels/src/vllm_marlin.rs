//! Rust FFI binding for the vLLM gptq_marlin port.
//!
//! See `crates/ferrum-kernels/vllm_marlin/` for the C++ source. The
//! `extern "C"` entry point at the bottom of `vllm_marlin/marlin.cu`
//! (`ferrum_marlin_mm_f16_u4b8`) is the only symbol we expose for now —
//! it forwards to `marlin::marlin_mm()` with a/b/c/s types fixed to
//! kFloat16 / kU4B8 / kFloat16 / kFloat16. That covers the FP16 +
//! GPTQ-INT4 path used by Llama-3.x INT4 (our M2 bench).
//!
//! Compile time: nvcc compiling `marlin.cu` + `gptq_marlin_repack.cu` +
//! `sm80_kernel_float16_u4b8_float16.cu` is ~10-20 min on a fresh build
//! (heavy template instantiation). Subsequent rebuilds are incremental.

use cudarc::driver::sys::CUstream;
use std::os::raw::{c_int, c_void};

extern "C" {
    /// GPTQ → vLLM-Marlin tile-format repack. Same total bytes as input
    /// (size_k × size_n / pack_factor uint32), just a permutation. Single
    /// expert per call; caller loops for stacked MoE.
    ///
    /// Returns 0 on success, non-zero on shape/config error.
    ///
    /// Output stride (in u32 elements): per expert = `(size_k / 16) *
    /// (size_n * 16 / pack_factor) = size_k * size_n / pack_factor` —
    /// same as input. So a stacked weight is `num_experts * (size_k *
    /// size_n / pack_factor)` u32, expert e at offset `e * stride`.
    ///
    /// `has_perm = 0` for our path (sym=true GPTQ, no act-order).
    /// Pass `perm = std::ptr::null()` when has_perm=0.
    pub fn ferrum_vllm_gptq_marlin_repack(
        qweight_in: *const c_void,
        perm_in: *const c_void,
        qweight_out: *mut c_void,
        size_k: c_int,
        size_n: c_int,
        num_bits: c_int,
        has_perm: c_int,
        dev: c_int,
        stream: CUstream,
    ) -> c_int;

    /// Forwards to `marlin::marlin_mm` with fixed FP16+kU4B8+FP16+FP16
    /// dtype combo. See `vllm_marlin/marlin.cu` end-of-file for the
    /// wrapper. Caller ensures all device pointers are valid + on
    /// `dev` + the workspace buffer is at least `sms` ints.
    pub fn ferrum_marlin_mm_f16_u4b8(
        // Buffers (device pointers, FP16 / INT4-packed / etc.)
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        c_tmp: *mut c_void,
        a_s: *mut c_void,
        b_s: *mut c_void,
        g_idx: *mut c_void,
        perm: *mut c_void,
        a_tmp: *mut c_void,
        // Shape
        prob_m: c_int,
        prob_n: c_int,
        prob_k: c_int,
        lda: c_int,
        // Workspace
        workspace: *mut c_void,
        // Flags
        has_act_order: bool,
        is_k_full: bool,
        num_groups: c_int,
        group_size: c_int,
        // Device + stream
        dev: c_int,
        stream: CUstream,
        // Tile init hints (-1 = let the kernel choose)
        thread_k_init: c_int,
        thread_n_init: c_int,
        sms: c_int,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
    );
}

/// Safe-ish wrapper over `ferrum_marlin_mm_f16_u4b8`. Caller still has
/// to guarantee the device pointers point at valid CUDA memory and live
/// for the duration of the call.
///
/// # Safety
/// - `a`, `b`, `c`, `c_tmp`, `a_s`, `b_s`, `g_idx`, `perm`, `a_tmp`,
///   `workspace` must be valid device pointers on device `dev`.
/// - `stream` must be a valid CUstream associated with device `dev`.
/// - Caller must respect Marlin shape constraints (size_n divisible by
///   min_thread_n, size_k divisible by tile_k_size, etc.). The kernel
///   abort()s otherwise.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_marlin_mm_f16_u4b8(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    c_tmp: *mut c_void,
    a_s: *mut c_void,
    b_s: *mut c_void,
    g_idx: *mut c_void,
    perm: *mut c_void,
    a_tmp: *mut c_void,
    prob_m: i32,
    prob_n: i32,
    prob_k: i32,
    lda: i32,
    workspace: *mut c_void,
    has_act_order: bool,
    is_k_full: bool,
    num_groups: i32,
    group_size: i32,
    dev: i32,
    stream: CUstream,
    sms: i32,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
) {
    ferrum_marlin_mm_f16_u4b8(
        a,
        b,
        c,
        c_tmp,
        a_s,
        b_s,
        g_idx,
        perm,
        a_tmp,
        prob_m,
        prob_n,
        prob_k,
        lda,
        workspace,
        has_act_order,
        is_k_full,
        num_groups,
        group_size,
        dev,
        stream,
        -1, // thread_k_init: let kernel choose
        -1, // thread_n_init: let kernel choose
        sms,
        use_atomic_add,
        use_fp32_reduce,
    );
}

/// Build a stacked `MarlinWeight` whose `qweight` is in **vLLM marlin
/// tile format** (NOT IST-DASLab Marlin format). For each expert we
/// repack the raw GPTQ qweight via `ferrum_vllm_gptq_marlin_repack`
/// and concatenate into one stacked buffer. Scales are concatenated
/// as-is (raw GPTQ); vLLM's marlin kernel reads them without a
/// permute step in the sym/no-act-order path.
///
/// Caller-side per-expert input:
///   qweights[e]: `[K/8, N]` i32 (GPTQ on-disk, sym=true)
///   scales[e]:   `[K/G, N]` f32 (NativeSafetensorsLoader format)
pub fn load_stacked_gptq_vllm_marlin(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    qweights: &[&[i32]],
    scales_f32: &[&[f32]],
    bits: u32,
    group_size: usize,
    k: usize,
    n_per_expert: usize,
) -> candle_core::Result<crate::marlin::MarlinWeight> {
    if bits != 4 {
        return Err(candle_core::Error::Msg(format!(
            "vLLM stacked Marlin: bits={bits} unsupported (only 4)"
        )));
    }
    let num_experts = qweights.len();
    if num_experts == 0 || scales_f32.len() != num_experts {
        return Err(candle_core::Error::Msg(format!(
            "vLLM stacked Marlin: shape mismatch qw={} sc={}",
            num_experts,
            scales_f32.len()
        )));
    }
    let qw_per = (k / 8) * n_per_expert;
    let groups = k / group_size;
    let sc_per = groups * n_per_expert;

    let total_qw = num_experts * qw_per;
    let total_sc = num_experts * sc_per;
    let mut qw_out: cudarc::driver::CudaSlice<i32> = stream
        .alloc_zeros::<i32>(total_qw)
        .map_err(|err| candle_core::Error::Msg(format!("alloc stacked qw: {err}")))?;

    use cudarc::driver::DevicePtr;
    let raw_stream = stream.cu_stream();
    for e in 0..num_experts {
        if qweights[e].len() != qw_per {
            return Err(candle_core::Error::Msg(format!(
                "vLLM stacked Marlin: qweight[{e}].len()={} expected {qw_per}",
                qweights[e].len()
            )));
        }
        let qw_in_dev: cudarc::driver::CudaSlice<i32> = stream
            .clone_htod(qweights[e])
            .map_err(|err| candle_core::Error::Msg(format!("htod qw[{e}]: {err}")))?;

        let (out_base_ptr, _g) = qw_out.device_ptr(stream);
        let out_offset_bytes = (e * qw_per * std::mem::size_of::<i32>()) as u64;
        let (in_ptr, _ig) = qw_in_dev.device_ptr(stream);
        let ret = unsafe {
            ferrum_vllm_gptq_marlin_repack(
                in_ptr as *const _,
                std::ptr::null(),
                (out_base_ptr + out_offset_bytes) as *mut _,
                k as i32,
                n_per_expert as i32,
                bits as i32,
                0, // has_perm
                0, // dev
                raw_stream,
            )
        };
        if ret != 0 {
            return Err(candle_core::Error::Msg(format!(
                "repack expert {e} failed ret={ret}"
            )));
        }
    }

    let mut sc_flat_f16: Vec<half::f16> = Vec::with_capacity(total_sc);
    for e in 0..num_experts {
        if scales_f32[e].len() != sc_per {
            return Err(candle_core::Error::Msg(format!(
                "vLLM stacked Marlin: scales[{e}].len()={} expected {sc_per}",
                scales_f32[e].len()
            )));
        }
        // Per-expert: convert to f16 then apply IST-DASLab Marlin scale
        // permutation. The vLLM marlin_template.h kernel reads scales
        // through a fragment-pattern shared-memory load (s_sh_rd) — same
        // as IST-DASLab — so the on-disk row-major scales need the same
        // host-side permute before the GEMM lines them up correctly with
        // the dequant-loop output channel.
        let sc_e_f16: Vec<half::f16> = scales_f32[e]
            .iter()
            .map(|&x| half::f16::from_f32(x))
            .collect();
        let sc_e_perm =
            crate::marlin::repack_scales_to_marlin(&sc_e_f16, k, n_per_expert, group_size);
        sc_flat_f16.extend(sc_e_perm);
    }
    let sc_dev: cudarc::driver::CudaSlice<half::f16> = stream
        .clone_htod(sc_flat_f16.as_slice())
        .map_err(|err| candle_core::Error::Msg(format!("htod stacked scales: {err}")))?;

    // Workspace: stacked across experts. IST-DASLab uses ceil(N/min_thread_n=64) ×
    // max_par lock slots. We mirror that and multiply by num_experts so
    // marlin_zero_stacked_workspace can clear per-expert tiles.
    let ws_per_expert = (n_per_expert / 64).max(1) * 16;
    let ws_total = num_experts * ws_per_expert;
    let workspace: cudarc::driver::CudaSlice<i32> = stream
        .alloc_zeros::<i32>(ws_total)
        .map_err(|err| candle_core::Error::Msg(format!("alloc workspace: {err}")))?;

    stream
        .synchronize()
        .map_err(|err| candle_core::Error::Msg(format!("sync after repack: {err}")))?;

    Ok(crate::marlin::MarlinWeight {
        qweight: qw_out,
        scales: sc_dev,
        workspace,
        k,
        n: n_per_expert * num_experts, // stacked N (per-expert tiles concatenated)
        group_size: group_size as i32,
        perm: None,
    })
}

/// Safe wrapper for the GPTQ → vLLM-Marlin repack. Allocates an output
/// buffer the same size as the input (in u32 elements) and runs the
/// repack kernel on `stream`.
///
/// `qweight_in_dev` MUST be a `[size_k / 8, size_n]` GPTQ-on-disk i32
/// buffer (sym=true, no act-order). Caller is responsible for stream
/// sync if they need to use the output before the kernel finishes.
pub fn vllm_gptq_marlin_repack(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    qweight_in_dev: &cudarc::driver::CudaSlice<i32>,
    qweight_out_dev: &mut cudarc::driver::CudaSlice<i32>,
    size_k: i32,
    size_n: i32,
) -> candle_core::Result<()> {
    use cudarc::driver::DevicePtr;
    let raw_stream = stream.cu_stream();
    let (in_ptr, _ig) = qweight_in_dev.device_ptr(stream);
    let (out_ptr, _og) = qweight_out_dev.device_ptr(stream);
    let ret = unsafe {
        ferrum_vllm_gptq_marlin_repack(
            in_ptr as *const _,
            std::ptr::null(),
            out_ptr as *mut _,
            size_k,
            size_n,
            4, // num_bits — INT4 GPTQ
            0, // has_perm — sym=true
            0, // dev
            raw_stream,
        )
    };
    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "vllm gptq_marlin_repack failed: ret={ret} (size_k={size_k}, size_n={size_n})"
        )));
    }
    Ok(())
}
