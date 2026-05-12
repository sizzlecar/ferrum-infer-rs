//! Q4_K_M MoE indirect-dispatch GEMV — Rust glue for `q4_k_moe_id_gemv.metal`.
//!
//! Single Metal dispatch covers all `n_selected` (token, expert) pairs at
//! decode m=1, replacing the per-expert gemv loop in `moe_forward`. See
//! `q4_k_moe_id_gemv.metal` for the algorithmic notes.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q4_k_moe_id_gemv.metal");
const KERNEL_NAME: &str = "gemv_q4kw_moe_id_f32";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q4_k_moe_id_gemv.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemv_q4kw_moe_id_f32 function");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemv_q4kw_moe_id_f32 pipeline")
    })
}

/// Dispatch a Q4_K MoE GEMV on an existing compute encoder.
///
/// `weights_stacked` : `[num_experts, n, k/256]` super-blocks contiguous,
///                     stride `nb02_bytes = n * (k/256) * 144` per expert.
/// `a`               : `[k]` activations (single token).
/// `ids`             : `[n_selected]` selected expert IDs (i32).
/// `out`             : `[n_selected, n]` output rows.
/// `n`               : per-expert out_features (must be divisible by 4).
/// `k`               : in_features (multiple of 256).
/// `n_selected`      : number of selected experts (= top_k for decode m=1).
/// `src1_stride` is the per-slot activation stride in **floats**:
/// `0` means every slot reads the same input row (broadcast — used for
/// MoE `gate` and `up`); `k` means each slot reads its own input row
/// (used for `down`, where each expert consumes its own silu·up output).
pub fn dispatch_gemv_q4k_moe_id_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    weights_stacked: &Buffer,
    weights_byte_offset: u64,
    ids: &Buffer,
    out: &Buffer,
    n: usize,
    k: usize,
    n_selected: usize,
    src1_stride: usize,
) {
    debug_assert!(k % 256 == 0, "K must be a multiple of 256 (got {k})");
    debug_assert!(n % 4 == 0, "N must be a multiple of 4 (got {n})");

    let nb01_bytes = (k / 256) * 144;
    let nb02_bytes = n * nb01_bytes;

    #[repr(C)]
    struct P {
        n: i32,
        k: i32,
        nb01: i32,
        nb02: i32,
        n_selected: i32,
        src1_stride: i32,
    }
    let params = P {
        n: n as i32,
        k: k as i32,
        nb01: nb01_bytes as i32,
        nb02: nb02_bytes as i32,
        n_selected: n_selected as i32,
        src1_stride: src1_stride as i32,
    };

    let pipe = pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(weights_stacked), weights_byte_offset);
    enc.set_buffer(1, Some(a), 0);
    enc.set_buffer(2, Some(ids), 0);
    enc.set_buffer(3, Some(out), 0);
    enc.set_bytes(
        4,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    const TILE_ROWS: u64 = 4;
    let grid = MTLSize::new((n as u64).div_ceil(TILE_ROWS), 1, n_selected as u64);
    let tg = MTLSize::new(32, 2, 1);
    enc.dispatch_thread_groups(grid, tg);
}

/// Offset-aware variant of [`dispatch_gemv_q4k_moe_id_on_encoder`].
///
/// `a_byte_offset` lets `a` start at a per-item position in a stacked
/// `[M, K]` buffer (eliminates the `copy_slice` from M-batched
/// `norm_out` into a single-row scratch on the per-item decode loop).
///
/// `ids_byte_offset` lets `ids` start at the i-th `top_k` block of a
/// stacked `[M, top_k]` selected-experts buffer (eliminates the
/// `copy_slice` from M-batched `selected_ids_buf`).
///
/// `out_byte_offset` is reserved for symmetry; output is currently
/// always written to offset 0 (per-iter scratch). Pass 0.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemv_q4k_moe_id_offset_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_byte_offset: u64,
    weights_stacked: &Buffer,
    weights_byte_offset: u64,
    ids: &Buffer,
    ids_byte_offset: u64,
    out: &Buffer,
    n: usize,
    k: usize,
    n_selected: usize,
    src1_stride: usize,
) {
    debug_assert!(k % 256 == 0, "K must be a multiple of 256 (got {k})");
    debug_assert!(n % 4 == 0, "N must be a multiple of 4 (got {n})");

    let nb01_bytes = (k / 256) * 144;
    let nb02_bytes = n * nb01_bytes;

    #[repr(C)]
    struct P {
        n: i32,
        k: i32,
        nb01: i32,
        nb02: i32,
        n_selected: i32,
        src1_stride: i32,
    }
    let params = P {
        n: n as i32,
        k: k as i32,
        nb01: nb01_bytes as i32,
        nb02: nb02_bytes as i32,
        n_selected: n_selected as i32,
        src1_stride: src1_stride as i32,
    };

    let pipe = pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(weights_stacked), weights_byte_offset);
    enc.set_buffer(1, Some(a), a_byte_offset);
    enc.set_buffer(2, Some(ids), ids_byte_offset);
    enc.set_buffer(3, Some(out), 0);
    enc.set_bytes(
        4,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    const TILE_ROWS: u64 = 4;
    let grid = MTLSize::new((n as u64).div_ceil(TILE_ROWS), 1, n_selected as u64);
    let tg = MTLSize::new(32, 2, 1);
    enc.dispatch_thread_groups(grid, tg);
}
