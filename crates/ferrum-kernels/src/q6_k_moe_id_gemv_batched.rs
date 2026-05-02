//! Q6_K MoE indirect-dispatch GEMV — **batched over M tokens**.
//!
//! Counterpart of `q4_k_moe_id_gemv_batched` for Q6_K expert weights.
//! Single Metal dispatch covers all `m * top_k` (token, expert) pairs;
//! same 2D src1 indexing contract (`src1_outer_stride` /
//! `src1_inner_stride`) so callers can switch between the two block
//! formats with no model-side changes.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

use crate::q6_k_moe_id_gemv::Q6_K_BLOCK_BYTES;

const SHADER_SRC: &str = include_str!("q6_k_moe_id_gemv_batched.metal");
const KERNEL_NAME: &str = "gemv_q6kw_moe_id_batched_f32";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q6_k_moe_id_gemv_batched.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemv_q6kw_moe_id_batched_f32 function");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemv_q6kw_moe_id_batched_f32 pipeline")
    })
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemv_q6k_moe_id_batched_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    weights_stacked: &Buffer,
    weights_byte_offset: u64,
    ids: &Buffer,
    out: &Buffer,
    n: usize,
    k: usize,
    m: usize,
    top_k: usize,
    src1_outer_stride: usize,
    src1_inner_stride: usize,
) {
    debug_assert!(k % 256 == 0);
    debug_assert!(n % 4 == 0);
    debug_assert!(top_k > 0 && m > 0);

    let nb01_bytes = (k / 256) * Q6_K_BLOCK_BYTES;
    let nb02_bytes = n * nb01_bytes;
    let n_pairs = m * top_k;

    #[repr(C)]
    struct P {
        n: i32,
        k: i32,
        nb01: i32,
        nb02: i32,
        top_k: i32,
        n_pairs: i32,
        src1_outer_stride: i32,
        src1_inner_stride: i32,
    }
    let params = P {
        n: n as i32,
        k: k as i32,
        nb01: nb01_bytes as i32,
        nb02: nb02_bytes as i32,
        top_k: top_k as i32,
        n_pairs: n_pairs as i32,
        src1_outer_stride: src1_outer_stride as i32,
        src1_inner_stride: src1_inner_stride as i32,
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
    let grid = MTLSize::new((n as u64).div_ceil(TILE_ROWS), 1, n_pairs as u64);
    let tg = MTLSize::new(32, 2, 1);
    enc.dispatch_thread_groups(grid, tg);
}
