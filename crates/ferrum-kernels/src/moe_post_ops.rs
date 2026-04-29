//! Stacked SiLU·gate + weighted-sum kernels for the MoE decode fast
//! path. See `moe_post_ops.metal` for the algorithmic notes.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("moe_post_ops.metal");

static SILU_PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();
static WSUM_PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn silu_pipeline(device: &Device) -> &'static ComputePipelineState {
    SILU_PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile moe_post_ops.metal");
        let function = lib
            .get_function("silu_mul_stacked_f32", None)
            .expect("find silu_mul_stacked_f32");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build silu_mul_stacked_f32 pipeline")
    })
}

fn wsum_pipeline(device: &Device) -> &'static ComputePipelineState {
    WSUM_PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile moe_post_ops.metal");
        let function = lib
            .get_function("weighted_sum_stacked_f32", None)
            .expect("find weighted_sum_stacked_f32");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build weighted_sum_stacked_f32 pipeline")
    })
}

/// Stacked SiLU·gate dispatch.
///
/// `gate`, `up`: `[n_slots, ffn]`. `out`: `[n_slots, ffn]`.
pub fn dispatch_silu_mul_stacked(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    gate: &Buffer,
    up: &Buffer,
    out: &Buffer,
    n_slots: usize,
    ffn: usize,
) {
    #[repr(C)]
    struct P {
        ffn: i32,
        n_slots: i32,
    }
    let params = P {
        ffn: ffn as i32,
        n_slots: n_slots as i32,
    };

    let pipe = silu_pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(gate), 0);
    enc.set_buffer(1, Some(up), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_bytes(
        3,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    let grid_x = (ffn as u64).div_ceil(256);
    let grid = MTLSize::new(grid_x, n_slots as u64, 1);
    let tg = MTLSize::new(256, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
}

/// Weighted-sum across slots dispatch.
///
/// `slots`: `[n_slots, hidden]`. `weights`: `[n_slots]`. `out`: `[hidden]`.
pub fn dispatch_weighted_sum_stacked(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    slots: &Buffer,
    weights: &Buffer,
    out: &Buffer,
    n_slots: usize,
    hidden: usize,
) {
    #[repr(C)]
    struct P {
        hidden: i32,
        n_slots: i32,
    }
    let params = P {
        hidden: hidden as i32,
        n_slots: n_slots as i32,
    };

    let pipe = wsum_pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(slots), 0);
    enc.set_buffer(1, Some(weights), 0);
    enc.set_buffer(2, Some(out), 0);
    enc.set_bytes(
        3,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    let grid_x = (hidden as u64).div_ceil(256);
    let grid = MTLSize::new(grid_x, 1, 1);
    let tg = MTLSize::new(256, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
}
