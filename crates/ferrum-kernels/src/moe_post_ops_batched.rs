//! Stacked SiLU·gate + weighted-sum kernels for batch > 1 (prefill).
//! See `moe_post_ops_batched.metal` for the algorithmic notes.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("moe_post_ops_batched.metal");

static SILU_PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();
static WSUM_PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn silu_pipeline(device: &Device) -> &'static ComputePipelineState {
    SILU_PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile moe_post_ops_batched.metal");
        let function = lib
            .get_function("silu_mul_batched_f32", None)
            .expect("find silu_mul_batched_f32");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build silu_mul_batched_f32 pipeline")
    })
}

fn wsum_pipeline(device: &Device) -> &'static ComputePipelineState {
    WSUM_PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile moe_post_ops_batched.metal");
        let function = lib
            .get_function("weighted_sum_batched_f32", None)
            .expect("find weighted_sum_batched_f32");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build weighted_sum_batched_f32 pipeline")
    })
}

pub fn dispatch_silu_mul_batched(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    gate: &Buffer,
    up: &Buffer,
    out: &Buffer,
    total_pairs: usize,
    ffn: usize,
) {
    #[repr(C)]
    struct P {
        total_pairs: i32,
        ffn: i32,
    }
    let params = P {
        total_pairs: total_pairs as i32,
        ffn: ffn as i32,
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

    let grid = MTLSize::new((ffn as u64).div_ceil(256), total_pairs as u64, 1);
    let tg = MTLSize::new(256, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
}

pub fn dispatch_weighted_sum_batched(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    slots: &Buffer,
    weights: &Buffer,
    out: &Buffer,
    batch: usize,
    top_k: usize,
    hidden: usize,
) {
    #[repr(C)]
    struct P {
        batch: i32,
        top_k: i32,
        hidden: i32,
    }
    let params = P {
        batch: batch as i32,
        top_k: top_k as i32,
        hidden: hidden as i32,
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

    let grid = MTLSize::new((hidden as u64).div_ceil(256), batch as u64, 1);
    let tg = MTLSize::new(256, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
}

// `compute_ids_tpe` moved to `crate::moe_host` (non-cfg-gated) so non-Metal
// builds can still call into it. See `moe_host::compute_ids_tpe`.
