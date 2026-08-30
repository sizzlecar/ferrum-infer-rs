//! Rust FFI binding for the versioned vLLM Marlin native operator artifact.
//!
//! Marlin compiles one CUDA specialization per supported scalar combination,
//! but all Rust callers share one versioned C launch ABI. Rust exposes a typed
//! FP16-activation weight kind rather than one FFI symbol per quantization
//! precision.
//!
//! Compile time: nvcc compiling `marlin.cu` + `gptq_marlin_repack.cu` +
//! `sm80_kernel_float16_u4b8_float16.cu` is ~10-20 min on a fresh build
//! (heavy template instantiation). Subsequent rebuilds are incremental.

use cudarc::driver::{sys::CUstream, CudaStream};
use std::os::raw::{c_int, c_void};

const FERRUM_MARLIN_ABI_VERSION: u32 = 1;
const FERRUM_MARLIN_SCALAR_F16: i32 = 1;
const FERRUM_MARLIN_SCALAR_U4: i32 = 4;
const FERRUM_MARLIN_SCALAR_U4B8: i32 = 5;
const FERRUM_MARLIN_SCALAR_FE4M3FN: i32 = 8;

const FERRUM_MARLIN_HAS_ACT_ORDER: u32 = 1 << 1;
const FERRUM_MARLIN_IS_K_FULL: u32 = 1 << 2;
const FERRUM_MARLIN_HAS_ZERO_POINTS: u32 = 1 << 3;
const FERRUM_MARLIN_USE_ATOMIC_ADD: u32 = 1 << 4;
const FERRUM_MARLIN_USE_FP32_REDUCE: u32 = 1 << 5;

#[repr(C)]
struct FerrumMarlinLaunch {
    abi_version: u32,
    struct_size: u32,
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    c_tmp: *mut c_void,
    b_bias: *mut c_void,
    a_scales: *mut c_void,
    b_scales: *mut c_void,
    global_scale: *mut c_void,
    zero_points: *mut c_void,
    group_index: *mut c_void,
    permutation: *mut c_void,
    a_tmp: *mut c_void,
    workspace: *mut c_void,
    stream: *mut c_void,
    prob_m: i32,
    prob_n: i32,
    prob_k: i32,
    lda: i32,
    a_type: i32,
    b_type: i32,
    c_type: i32,
    scale_type: i32,
    num_groups: i32,
    group_size: i32,
    device: i32,
    thread_k_init: i32,
    thread_n_init: i32,
    sms: i32,
    flags: u32,
    reserved: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MarlinF16WeightType {
    U4,
    U4B8,
    E4M3Fn,
}

impl MarlinF16WeightType {
    const fn ffi_scalar_type(self) -> i32 {
        match self {
            Self::U4 => FERRUM_MARLIN_SCALAR_U4,
            Self::U4B8 => FERRUM_MARLIN_SCALAR_U4B8,
            Self::E4M3Fn => FERRUM_MARLIN_SCALAR_FE4M3FN,
        }
    }
}

#[derive(Clone, Copy)]
pub struct MarlinMmBuffers {
    pub a: *const c_void,
    pub b: *const c_void,
    pub c: *mut c_void,
    pub c_tmp: *mut c_void,
    pub a_scales: *mut c_void,
    pub b_scales: *mut c_void,
    pub zero_points: *mut c_void,
    pub group_index: *mut c_void,
    pub permutation: *mut c_void,
    pub a_tmp: *mut c_void,
    pub workspace: *mut c_void,
}

#[derive(Clone, Copy)]
pub struct MarlinMmProblem {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub num_groups: i32,
    pub group_size: i32,
}

#[derive(Clone, Copy)]
pub struct MarlinMmExecution {
    pub device: i32,
    pub stream: CUstream,
    pub sms: i32,
    pub has_act_order: bool,
    pub is_k_full: bool,
    pub use_atomic_add: bool,
    pub use_fp32_reduce: bool,
}

#[derive(Clone, Copy)]
pub struct MarlinMmF16WeightRequest {
    pub weight_type: MarlinF16WeightType,
    pub buffers: MarlinMmBuffers,
    pub problem: MarlinMmProblem,
    pub execution: MarlinMmExecution,
}

impl MarlinMmF16WeightRequest {
    fn into_ffi(self) -> FerrumMarlinLaunch {
        let mut flags = 0;
        if self.execution.has_act_order {
            flags |= FERRUM_MARLIN_HAS_ACT_ORDER;
        }
        if self.execution.is_k_full {
            flags |= FERRUM_MARLIN_IS_K_FULL;
        }
        if !self.buffers.zero_points.is_null() {
            flags |= FERRUM_MARLIN_HAS_ZERO_POINTS;
        }
        if self.execution.use_atomic_add {
            flags |= FERRUM_MARLIN_USE_ATOMIC_ADD;
        }
        if self.execution.use_fp32_reduce {
            flags |= FERRUM_MARLIN_USE_FP32_REDUCE;
        }

        FerrumMarlinLaunch {
            abi_version: FERRUM_MARLIN_ABI_VERSION,
            struct_size: std::mem::size_of::<FerrumMarlinLaunch>() as u32,
            a: self.buffers.a,
            b: self.buffers.b,
            c: self.buffers.c,
            c_tmp: self.buffers.c_tmp,
            b_bias: std::ptr::null_mut(),
            a_scales: self.buffers.a_scales,
            b_scales: self.buffers.b_scales,
            global_scale: std::ptr::null_mut(),
            zero_points: self.buffers.zero_points,
            group_index: self.buffers.group_index,
            permutation: self.buffers.permutation,
            a_tmp: self.buffers.a_tmp,
            workspace: self.buffers.workspace,
            stream: self.execution.stream.cast(),
            prob_m: self.problem.m,
            prob_n: self.problem.n,
            prob_k: self.problem.k,
            lda: self.problem.lda,
            a_type: FERRUM_MARLIN_SCALAR_F16,
            b_type: self.weight_type.ffi_scalar_type(),
            c_type: FERRUM_MARLIN_SCALAR_F16,
            scale_type: FERRUM_MARLIN_SCALAR_F16,
            num_groups: self.problem.num_groups,
            group_size: self.problem.group_size,
            device: self.execution.device,
            thread_k_init: -1,
            thread_n_init: -1,
            sms: self.execution.sms,
            flags,
            reserved: 0,
        }
    }
}

extern "C" {
    fn ferrum_block_fp8_group128_repack(
        row_major: *const c_void,
        marlin_packed: *mut c_void,
        size_k: c_int,
        size_n: c_int,
        stream: CUstream,
    ) -> c_int;

    fn ferrum_block_fp8_group128_scales(
        inverse_scales_bf16: *const c_void,
        marlin_scales_f16: *mut c_void,
        size_k: c_int,
        size_n: c_int,
        stream: CUstream,
    ) -> c_int;

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

    fn ferrum_marlin_mm(launch: *const FerrumMarlinLaunch);
}

fn block_fp8_group128_launch_dimensions(
    size_k: u64,
    size_n: u64,
) -> candle_core::Result<(c_int, c_int)> {
    if size_k == 0 || size_n == 0 || !size_k.is_multiple_of(128) || !size_n.is_multiple_of(128) {
        return Err(candle_core::Error::Msg(format!(
            "block-FP8 group-128 CUDA transform requires positive K/N multiples of 128, got K={size_k}, N={size_n}"
        )));
    }
    let size_k = c_int::try_from(size_k).map_err(|_| {
        candle_core::Error::Msg("block-FP8 group-128 K exceeds native i32".to_owned())
    })?;
    let size_n = c_int::try_from(size_n).map_err(|_| {
        candle_core::Error::Msg("block-FP8 group-128 N exceeds native i32".to_owned())
    })?;
    Ok((size_k, size_n))
}

#[cfg(test)]
fn block_fp8_group128_marlin_word_source_indices(
    word: usize,
    size_k: usize,
    size_n: usize,
) -> Option<[usize; 4]> {
    if size_k == 0
        || size_n == 0
        || !size_k.is_multiple_of(128)
        || !size_n.is_multiple_of(128)
        || word >= size_k.checked_mul(size_n)?.checked_div(4)?
    {
        return None;
    }
    let words_per_tile = 16 * 64 / 4;
    let tile = word / words_per_tile;
    let word_in_tile = word % words_per_tile;
    let n_tiles = size_n / 64;
    let k_tile = tile / n_tiles;
    let n_tile = tile % n_tiles;
    let marlin_thread = word_in_tile / 8;
    let word_lane = word_in_tile % 8;
    let warp = word_lane / 2;
    let column_half = word_lane % 2;
    let tensor_core_column = marlin_thread / 4;
    let tensor_core_row = (marlin_thread % 4) * 2;
    let output = n_tile * 64 + warp * 16 + tensor_core_column + column_half * 8;
    let source_base = output * size_k + k_tile * 16;
    Some([
        source_base + tensor_core_row,
        source_base + tensor_core_row + 8,
        source_base + tensor_core_row + 1,
        source_base + tensor_core_row + 9,
    ])
}

#[cfg(test)]
fn block_fp8_group128_scale_source_index(
    destination: usize,
    size_k: usize,
    size_n: usize,
) -> Option<usize> {
    if size_k == 0 || size_n == 0 || !size_k.is_multiple_of(128) || !size_n.is_multiple_of(128) {
        return None;
    }
    let group_count = size_k / 128;
    let scale_count = group_count.checked_mul(size_n)?;
    if destination >= scale_count {
        return None;
    }
    let permutation_width = if group_count == 1 { 32 } else { 64 };
    let destination_lane = destination % permutation_width;
    let source_lane = if permutation_width == 32 {
        const COLUMNS: [usize; 8] = [0, 1, 8, 9, 16, 17, 24, 25];
        2 * (destination_lane / 8) + COLUMNS[destination_lane % 8]
    } else {
        destination_lane / 8 + 8 * (destination_lane % 8)
    };
    let logical = destination / permutation_width * permutation_width + source_lane;
    let group = logical / size_n;
    let output = logical % size_n;
    Some((output / 128) * group_count + group)
}

/// Launch the product static-weight transform from exact row-major checkpoint
/// E4M3 bits directly into the final vLLM Marlin W8A16 tile layout.
///
/// # Safety
///
/// `row_major` must address `size_n * size_k` readable device bytes and
/// `marlin_packed` the same number of writable device bytes on `stream`'s
/// context. The two ranges must not overlap.
pub(crate) unsafe fn launch_block_fp8_group128_repack(
    stream: &CudaStream,
    row_major: cudarc::driver::sys::CUdeviceptr,
    marlin_packed: cudarc::driver::sys::CUdeviceptr,
    size_k: u64,
    size_n: u64,
) -> candle_core::Result<()> {
    let (size_k, size_n) = block_fp8_group128_launch_dimensions(size_k, size_n)?;
    let ret = unsafe {
        ferrum_block_fp8_group128_repack(
            row_major as usize as *const c_void,
            marlin_packed as usize as *mut c_void,
            size_k,
            size_n,
            stream.cu_stream(),
        )
    };
    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "block-FP8 group-128 direct Marlin repack launch failed: ret={ret} (K={size_k}, N={size_n})"
        )));
    }
    Ok(())
}

/// Launch BF16 inverse-scale expansion, exponent-bias correction, and the
/// final P32/P64 Marlin scale permutation directly into F16 destination bytes.
///
/// # Safety
///
/// `inverse_scales_bf16` must address `(size_n / 128) * (size_k / 128)`
/// readable BF16 elements and `marlin_scales_f16` must address
/// `size_n * (size_k / 128)` writable F16 elements on `stream`'s context. The
/// two ranges must not overlap.
pub(crate) unsafe fn launch_block_fp8_group128_scales(
    stream: &CudaStream,
    inverse_scales_bf16: cudarc::driver::sys::CUdeviceptr,
    marlin_scales_f16: cudarc::driver::sys::CUdeviceptr,
    size_k: u64,
    size_n: u64,
) -> candle_core::Result<()> {
    let (size_k, size_n) = block_fp8_group128_launch_dimensions(size_k, size_n)?;
    let ret = unsafe {
        ferrum_block_fp8_group128_scales(
            inverse_scales_bf16 as usize as *const c_void,
            marlin_scales_f16 as usize as *mut c_void,
            size_k,
            size_n,
            stream.cu_stream(),
        )
    };
    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "block-FP8 group-128 scale transform launch failed: ret={ret} (K={size_k}, N={size_n})"
        )));
    }
    Ok(())
}

/// Launch an FP16-activation Marlin GEMM through the shared versioned FFI.
///
/// # Safety
/// - The request buffers must be valid device pointers on the requested device.
/// - The stream must be a valid CUstream associated with that device.
/// - Caller must respect Marlin shape constraints (size_n divisible by
///   min_thread_n, size_k divisible by tile_k_size, etc.). The kernel
///   abort()s otherwise.
pub unsafe fn launch_marlin_mm_f16_weight(request: MarlinMmF16WeightRequest) {
    let launch = request.into_ffi();
    ferrum_marlin_mm(&launch);
}

/// Compatibility helper for the existing GPTQ U4B8 call sites.
///
/// # Safety
/// The same pointer, stream, and shape requirements as
/// [`launch_marlin_mm_f16_weight`] apply.
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
    launch_marlin_mm_f16_weight(MarlinMmF16WeightRequest {
        weight_type: MarlinF16WeightType::U4B8,
        buffers: MarlinMmBuffers {
            a,
            b,
            c,
            c_tmp,
            a_scales: a_s,
            b_scales: b_s,
            zero_points: std::ptr::null_mut(),
            group_index: g_idx,
            permutation: perm,
            a_tmp,
            workspace,
        },
        problem: MarlinMmProblem {
            m: prob_m,
            n: prob_n,
            k: prob_k,
            lda,
            num_groups,
            group_size,
        },
        execution: MarlinMmExecution {
            device: dev,
            stream,
            sms,
            has_act_order,
            is_k_full,
            use_atomic_add,
            use_fp32_reduce,
        },
    });
}

/// Build a stacked `MarlinWeight` whose `qweight` is in the shared
/// IST-DASLab/vLLM Marlin INT4 tile format. For each expert we
/// repack the raw GPTQ qweight via `ferrum_vllm_gptq_marlin_repack`
/// and concatenate into one stacked buffer. Scales are concatenated
/// after the same Marlin scale permutation used by the vLLM kernel.
/// Asymmetric GPTQ qzeros are converted from AutoGPTQ's packed
/// `zero - 1` encoding into packed runtime zero-points while preserving
/// the kernel's `[groups, N/8]` zero-point layout.
///
/// Caller-side per-expert input:
///   qweights[e]: `[K/8, N]` i32 (GPTQ on-disk, sym=true)
///   scales[e]:   `[K/G, N]` f32 (NativeSafetensorsLoader format)
///   qzeros[e]:   `[K/G, N/8]` i32 (GPTQ on-disk, packed `zero - 1`)
pub fn load_stacked_gptq_vllm_marlin(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    qweights: &[&[i32]],
    scales_f32: &[&[f32]],
    qzeros: &[&[i32]],
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
    if num_experts == 0 || scales_f32.len() != num_experts || qzeros.len() != num_experts {
        return Err(candle_core::Error::Msg(format!(
            "vLLM stacked Marlin: shape mismatch qw={} sc={} qz={}",
            num_experts,
            scales_f32.len(),
            qzeros.len()
        )));
    }
    if group_size == 0 || k % group_size != 0 {
        return Err(candle_core::Error::Msg(format!(
            "vLLM stacked Marlin: K={k} not divisible by group_size={group_size}"
        )));
    }
    if n_per_expert % 8 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "vLLM stacked Marlin: N={n_per_expert} must be divisible by 8 for INT4 qzeros"
        )));
    }
    let qw_per = (k / 8) * n_per_expert;
    let groups = k / group_size;
    let sc_per = groups * n_per_expert;
    let qz_per = groups * (n_per_expert / 8);

    let total_qw = num_experts * qw_per;
    let total_sc = num_experts * sc_per;
    let qw_out: cudarc::driver::CudaSlice<i32> = stream
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
        if qzeros[e].len() != qz_per {
            return Err(candle_core::Error::Msg(format!(
                "vLLM stacked Marlin: qzeros[{e}].len()={} expected {qz_per}",
                qzeros[e].len()
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

    let has_asymmetric_qzeros = qzeros.iter().any(|qz| !gptq_qzeros_are_symmetric_code7(qz));
    let qzeros_dev = if has_asymmetric_qzeros {
        let mut qz_flat: Vec<i32> = Vec::with_capacity(num_experts * qz_per);
        for (e, qz) in qzeros.iter().enumerate() {
            let qz_repacked = repack_gptq_qzeros_to_marlin(qz, k, n_per_expert, group_size)
                .map_err(|err| {
                    candle_core::Error::Msg(format!("vLLM stacked Marlin qzeros[{e}]: {err}"))
                })?;
            qz_flat.extend(qz_repacked);
        }
        Some(
            stream
                .clone_htod(qz_flat.as_slice())
                .map_err(|err| candle_core::Error::Msg(format!("htod stacked qzeros: {err}")))?,
        )
    } else {
        None
    };

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
        qzeros: qzeros_dev,
        workspace,
        k,
        n: n_per_expert * num_experts, // stacked N (per-expert tiles concatenated)
        group_size: group_size as i32,
        vllm_moe: true,
        perm: None,
    })
}

pub(crate) fn gptq_qzeros_are_symmetric_code7(qzeros: &[i32]) -> bool {
    !qzeros.is_empty()
        && qzeros.iter().all(|&word| {
            let word = word as u32;
            (0..8).all(|i| ((word >> (i * 4)) & 0xF) == 7)
        })
}

pub(crate) fn repack_gptq_qzeros_to_marlin(
    qzeros: &[i32],
    k: usize,
    n: usize,
    group_size: usize,
) -> candle_core::Result<Vec<i32>> {
    if group_size == 0 || k % group_size != 0 {
        return Err(candle_core::Error::Msg(format!(
            "K={k} not divisible by group_size={group_size}"
        )));
    }
    if n % 8 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "N={n} must be divisible by 8 for INT4 qzeros"
        )));
    }
    let groups = k / group_size;
    let qz_per = groups * (n / 8);
    if qzeros.len() != qz_per {
        return Err(candle_core::Error::Msg(format!(
            "qzeros len={} expected {qz_per} for groups={groups} N={n}",
            qzeros.len()
        )));
    }
    let packed_cols = n / 8;
    let mut packed = vec![0i32; qz_per];
    for group in 0..groups {
        for packed_col in 0..packed_cols {
            let word = qzeros[group * packed_cols + packed_col] as u32;
            let mut out_word = 0u32;
            for lane in 0..8 {
                let raw = ((word >> (lane * 4)) & 0xF) as u8;
                if raw == 15 {
                    return Err(candle_core::Error::Msg(format!(
                        "qzeros group={group} packed_col={packed_col} lane={lane} has code 15; \
                         AutoGPTQ zero+1 would exceed INT4 range"
                    )));
                }
                out_word |= ((raw + 1) as u32) << (lane * 4);
            }
            packed[group * packed_cols + packed_col] = out_word as i32;
        }
    }
    Ok(packed)
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

fn validate_vllm_fp8_marlin_repack_raw_bits(
    input_elements: usize,
    output_elements: usize,
    size_k: i32,
    size_n: i32,
) -> candle_core::Result<()> {
    if size_k <= 0 || size_k % 16 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "vLLM FP8 Marlin repack size_k must be a positive multiple of 16, got {size_k}"
        )));
    }
    if size_n <= 0 || size_n % 64 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "vLLM FP8 Marlin repack size_n must be a positive multiple of 64, got {size_n}"
        )));
    }
    let size_k = usize::try_from(size_k).expect("positive i32 size_k fits usize");
    let size_n = usize::try_from(size_n).expect("positive i32 size_n fits usize");
    let expected_elements = size_k
        .checked_mul(size_n)
        .and_then(|elements| elements.checked_div(4))
        .ok_or_else(|| {
            candle_core::Error::Msg("vLLM FP8 Marlin repack element count exceeds usize".to_owned())
        })?;
    if input_elements != expected_elements || output_elements != expected_elements {
        return Err(candle_core::Error::Msg(format!(
            "vLLM FP8 Marlin repack requires [K/4, N] input and equal-size output: \
             expected {expected_elements} u32 elements, got input={input_elements}, output={output_elements}"
        )));
    }
    Ok(())
}

/// Repack raw E4M3 bytes from the GPTQ-compatible K-major input ABI into the
/// vLLM Marlin W8A16 tile ABI without decoding or requantizing the weights.
///
/// `raw_bits_k_major` has shape `[size_k / 4, size_n]` in `u32` elements. Each
/// little-endian word contains four consecutive K-axis E4M3 bytes for one
/// output channel. A row-major checkpoint matrix `[N, K]` must therefore be
/// transposed and packed into this shape before calling this function. The
/// output contains the same number of `u32` elements in Marlin tile order.
pub fn vllm_fp8_marlin_repack_raw_bits(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    raw_bits_k_major: &cudarc::driver::CudaSlice<u32>,
    marlin_packed: &mut cudarc::driver::CudaSlice<u32>,
    size_k: i32,
    size_n: i32,
) -> candle_core::Result<()> {
    validate_vllm_fp8_marlin_repack_raw_bits(
        raw_bits_k_major.len(),
        marlin_packed.len(),
        size_k,
        size_n,
    )?;
    let stream_ordinal = stream.context().ordinal();
    if raw_bits_k_major.ordinal() != stream_ordinal || marlin_packed.ordinal() != stream_ordinal {
        return Err(candle_core::Error::Msg(
            "vLLM FP8 Marlin repack buffers and stream must belong to the same CUDA device"
                .to_owned(),
        ));
    }
    let device_ordinal = i32::try_from(stream_ordinal)
        .map_err(|_| candle_core::Error::Msg("CUDA device ordinal exceeds i32".to_owned()))?;

    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let (input_pointer, _input_guard) = raw_bits_k_major.device_ptr(stream);
    let (output_pointer, _output_guard) = marlin_packed.device_ptr_mut(stream);
    let ret = unsafe {
        ferrum_vllm_gptq_marlin_repack(
            input_pointer as *const _,
            std::ptr::null(),
            output_pointer as *mut _,
            size_k,
            size_n,
            8,
            0,
            device_ordinal,
            stream.cu_stream(),
        )
    };
    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "vLLM FP8 gptq_marlin_repack failed: ret={ret} (size_k={size_k}, size_n={size_n})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        block_fp8_group128_marlin_word_source_indices, block_fp8_group128_scale_source_index,
        gptq_qzeros_are_symmetric_code7, repack_gptq_qzeros_to_marlin,
        validate_vllm_fp8_marlin_repack_raw_bits, FerrumMarlinLaunch, MarlinF16WeightType,
        MarlinMmBuffers, MarlinMmExecution, MarlinMmF16WeightRequest, MarlinMmProblem,
        FERRUM_MARLIN_HAS_ACT_ORDER, FERRUM_MARLIN_HAS_ZERO_POINTS, FERRUM_MARLIN_IS_K_FULL,
        FERRUM_MARLIN_SCALAR_FE4M3FN, FERRUM_MARLIN_SCALAR_U4, FERRUM_MARLIN_SCALAR_U4B8,
        FERRUM_MARLIN_USE_ATOMIC_ADD, FERRUM_MARLIN_USE_FP32_REDUCE,
    };

    #[test]
    fn marlin_launch_ffi_layout_and_weight_types_are_stable() {
        assert_eq!(std::mem::size_of::<FerrumMarlinLaunch>(), 184);
        assert_eq!(std::mem::align_of::<FerrumMarlinLaunch>(), 8);
        assert_eq!(
            MarlinF16WeightType::U4.ffi_scalar_type(),
            FERRUM_MARLIN_SCALAR_U4
        );
        assert_eq!(
            MarlinF16WeightType::U4B8.ffi_scalar_type(),
            FERRUM_MARLIN_SCALAR_U4B8
        );
        assert_eq!(
            MarlinF16WeightType::E4M3Fn.ffi_scalar_type(),
            FERRUM_MARLIN_SCALAR_FE4M3FN
        );
    }

    #[test]
    fn typed_marlin_request_maps_to_versioned_ffi() {
        let request = MarlinMmF16WeightRequest {
            weight_type: MarlinF16WeightType::U4,
            buffers: MarlinMmBuffers {
                a: 1_usize as *const _,
                b: 2_usize as *const _,
                c: 3_usize as *mut _,
                c_tmp: 4_usize as *mut _,
                a_scales: 5_usize as *mut _,
                b_scales: 6_usize as *mut _,
                zero_points: 20_usize as *mut _,
                group_index: 7_usize as *mut _,
                permutation: 8_usize as *mut _,
                a_tmp: 9_usize as *mut _,
                workspace: 10_usize as *mut _,
            },
            problem: MarlinMmProblem {
                m: 11,
                n: 12,
                k: 13,
                lda: 14,
                num_groups: 15,
                group_size: 16,
            },
            execution: MarlinMmExecution {
                device: 17,
                stream: 18_usize as _,
                sms: 19,
                has_act_order: true,
                is_k_full: true,
                use_atomic_add: true,
                use_fp32_reduce: true,
            },
        };

        let launch = request.into_ffi();
        assert_eq!(launch.a, request.buffers.a);
        assert_eq!(launch.b, request.buffers.b);
        assert_eq!(launch.c, request.buffers.c);
        assert_eq!(launch.c_tmp, request.buffers.c_tmp);
        assert_eq!(launch.a_scales, request.buffers.a_scales);
        assert_eq!(launch.b_scales, request.buffers.b_scales);
        assert_eq!(launch.zero_points, request.buffers.zero_points);
        assert_eq!(launch.group_index, request.buffers.group_index);
        assert_eq!(launch.permutation, request.buffers.permutation);
        assert_eq!(launch.a_tmp, request.buffers.a_tmp);
        assert_eq!(launch.workspace, request.buffers.workspace);
        assert_eq!(launch.prob_m, request.problem.m);
        assert_eq!(launch.prob_n, request.problem.n);
        assert_eq!(launch.prob_k, request.problem.k);
        assert_eq!(launch.lda, request.problem.lda);
        assert_eq!(launch.num_groups, request.problem.num_groups);
        assert_eq!(launch.group_size, request.problem.group_size);
        assert_eq!(launch.device, request.execution.device);
        assert_eq!(launch.sms, request.execution.sms);
        assert_eq!(
            launch.flags,
            FERRUM_MARLIN_HAS_ACT_ORDER
                | FERRUM_MARLIN_IS_K_FULL
                | FERRUM_MARLIN_HAS_ZERO_POINTS
                | FERRUM_MARLIN_USE_ATOMIC_ADD
                | FERRUM_MARLIN_USE_FP32_REDUCE
        );

        let fp8_request = MarlinMmF16WeightRequest {
            weight_type: MarlinF16WeightType::E4M3Fn,
            buffers: MarlinMmBuffers {
                zero_points: std::ptr::null_mut(),
                ..request.buffers
            },
            problem: MarlinMmProblem {
                m: 1,
                n: 128,
                k: 256,
                lda: 256,
                num_groups: 2,
                group_size: 128,
            },
            execution: MarlinMmExecution {
                has_act_order: false,
                ..request.execution
            },
        };
        let fp8_launch = fp8_request.into_ffi();
        assert_eq!(fp8_launch.b_type, FERRUM_MARLIN_SCALAR_FE4M3FN);
        assert_eq!(fp8_launch.num_groups, 2);
        assert_eq!(fp8_launch.group_size, 128);
        assert_eq!(
            fp8_launch.flags & (FERRUM_MARLIN_HAS_ACT_ORDER | FERRUM_MARLIN_HAS_ZERO_POINTS),
            0
        );
    }

    #[test]
    fn fp8_raw_bit_repack_requires_exact_k_major_u32_extents() {
        let expected = 128 * 64 / 4;
        validate_vllm_fp8_marlin_repack_raw_bits(expected, expected, 128, 64).unwrap();

        for (input, output) in [(expected - 1, expected), (expected, expected - 1)] {
            let error = validate_vllm_fp8_marlin_repack_raw_bits(input, output, 128, 64)
                .expect_err("mismatched raw-bit extent must fail");
            assert!(error.to_string().contains("[K/4, N]"));
        }
    }

    #[test]
    fn fp8_raw_bit_repack_rejects_non_tile_shapes() {
        let cases = [(0, 64, "size_k"), (127, 64, "size_k"), (128, 32, "size_n")];
        for (size_k, size_n, expected) in cases {
            let error = validate_vllm_fp8_marlin_repack_raw_bits(0, 0, size_k, size_n)
                .expect_err("non-tile shape must fail");
            assert!(error.to_string().contains(expected));
        }
    }

    #[test]
    fn block_fp8_direct_repack_indices_match_nested_marlin_tile_oracle() {
        let n = 256_usize;
        let k = 256_usize;
        let source = (0..n * k)
            .map(|index| (index as u8).wrapping_mul(73).wrapping_add(0x5b))
            .collect::<Vec<_>>();
        let direct = (0..n * k / 4)
            .flat_map(|word| {
                block_fp8_group128_marlin_word_source_indices(word, k, n)
                    .expect("validated fixture word")
                    .map(|source_index| source[source_index])
            })
            .collect::<Vec<_>>();

        let mut nested = vec![0_u8; n * k];
        for k_tile in 0..k / 16 {
            for n_tile in 0..n / 64 {
                let output_base = (k_tile * (n / 64) + n_tile) * 16 * 64;
                for thread in 0..32 {
                    let tensor_core_column = thread / 4;
                    let tensor_core_row = (thread % 4) * 2;
                    for warp in 0..4 {
                        for half in 0..2 {
                            let output = n_tile * 64 + warp * 16 + tensor_core_column + half * 8;
                            let rows = [
                                tensor_core_row,
                                tensor_core_row + 8,
                                tensor_core_row + 1,
                                tensor_core_row + 9,
                            ];
                            let output_word = thread * 8 + warp * 2 + half;
                            for (byte, row) in rows.into_iter().enumerate() {
                                nested[output_base + output_word * 4 + byte] =
                                    source[output * k + k_tile * 16 + row];
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(direct, nested);
    }

    #[test]
    fn block_fp8_direct_repack_fuses_gate_up_before_tiling() {
        let source_n = 128_usize;
        let fused_n = source_n * 2;
        let k = 128_usize;
        let gate = (0..source_n * k)
            .map(|index| (index as u8).wrapping_mul(17).wrapping_add(3))
            .collect::<Vec<_>>();
        let up = (0..source_n * k)
            .map(|index| (index as u8).wrapping_mul(29).wrapping_add(11))
            .collect::<Vec<_>>();
        let fused = [gate, up].concat();
        let packed = (0..fused_n * k / 4)
            .flat_map(|word| {
                block_fp8_group128_marlin_word_source_indices(word, k, fused_n)
                    .expect("validated fused fixture word")
                    .map(|source_index| fused[source_index])
            })
            .collect::<Vec<_>>();
        assert_eq!(packed.len(), fused.len());

        let up_source_position = (0..fused_n * k / 4)
            .find_map(|word| {
                block_fp8_group128_marlin_word_source_indices(word, k, fused_n)
                    .expect("validated fused fixture word")
                    .into_iter()
                    .position(|source_index| source_index == source_n * k)
                    .map(|byte| word * 4 + byte)
            })
            .expect("fused up row is represented in final tiles");
        assert_eq!(packed[up_source_position], fused[source_n * k]);
    }

    #[test]
    fn block_fp8_direct_scale_indices_match_p32_and_p64_oracles() {
        for (n, k) in [(128_usize, 128_usize), (256_usize, 256_usize)] {
            let source = (0..(n / 128) * (k / 128))
                .flat_map(|index| half::bf16::from_f32((index + 1) as f32 / 8.0).to_le_bytes())
                .collect::<Vec<_>>();
            let actual = (0..n * (k / 128))
                .map(|destination| {
                    let source_index = block_fp8_group128_scale_source_index(destination, k, n)
                        .expect("validated scale fixture index");
                    let offset = source_index * 2;
                    let inverse =
                        half::bf16::from_le_bytes([source[offset], source[offset + 1]]).to_f32();
                    half::f16::from_f32(inverse * 256.0)
                })
                .collect::<Vec<_>>();
            let expected = crate::marlin_repack::block_fp8_group128_scales_to_marlin_f16_reference(
                &source, n, k,
            )
            .unwrap();
            assert_eq!(actual, expected, "N={n}, K={k}");
        }
    }

    #[test]
    fn qzeros_code7_detects_symmetric_gptq() {
        assert!(gptq_qzeros_are_symmetric_code7(&[0x7777_7777]));
        assert!(!gptq_qzeros_are_symmetric_code7(&[0x7777_7778]));
        assert!(!gptq_qzeros_are_symmetric_code7(&[]));
    }

    #[test]
    fn qzeros_code8_repack_converts_to_actual_zero_point_9() {
        let qzeros = vec![0x8888_8888u32 as i32; 8];
        let packed = repack_gptq_qzeros_to_marlin(&qzeros, 128, 64, 128).unwrap();
        assert_eq!(packed, vec![0x9999_9999u32 as i32; 8]);
    }

    #[test]
    fn qzeros_repack_preserves_kernel_layout() {
        let actual = [1u8, 2, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15];
        let mut qzeros = vec![0i32; 8];
        for packed_col in 0..2 {
            let mut word = 0u32;
            for lane in 0..8 {
                let raw = actual[packed_col * 8 + lane] - 1;
                word |= (raw as u32) << (lane * 4);
            }
            qzeros[packed_col] = word as i32;
        }
        qzeros[2..].fill(0x7777_7777);

        let packed = repack_gptq_qzeros_to_marlin(&qzeros, 128, 64, 128).unwrap();
        assert_eq!(packed[0] as u32, 0x8765_4321);
        assert_eq!(packed[1] as u32, 0xFEDC_BA98);
        assert_eq!(packed[2] as u32, 0x8888_8888);
    }
}
