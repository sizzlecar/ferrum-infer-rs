//! Q6_K MoE indirect-dispatch GEMV — Rust glue for `q6_k_moe_id_gemv.metal`.
//!
//! Counterpart to `q4_k_moe_id_gemv` for Q6_K-quantised expert weights
//! (typically used for `ffn_down_exps` in Qwen3-30B-A3B Q4_K_M, where
//! down is Q6_K and gate/up are Q4_K). One dispatch covers all
//! `n_selected` (token, expert) pairs.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q6_k_moe_id_gemv.metal");
const KERNEL_NAME: &str = "gemv_q6kw_moe_id_f32";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q6_k_moe_id_gemv.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemv_q6kw_moe_id_f32 function");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemv_q6kw_moe_id_f32 pipeline")
    })
}

/// Q6_K MoE block stride in bytes (same as the dense Q6_K GEMV uses).
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// See `q4_k_moe_id_gemv::dispatch_gemv_q4k_moe_id_on_encoder` for the
/// `src1_stride` contract.
pub fn dispatch_gemv_q6k_moe_id_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    weights_stacked: &Buffer,
    ids: &Buffer,
    out: &Buffer,
    n: usize,
    k: usize,
    n_selected: usize,
    src1_stride: usize,
) {
    debug_assert!(k % 256 == 0);
    debug_assert!(n % 4 == 0);

    let nb01_bytes = (k / 256) * Q6_K_BLOCK_BYTES;
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
    enc.set_buffer(0, Some(weights_stacked), 0);
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
