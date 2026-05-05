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
