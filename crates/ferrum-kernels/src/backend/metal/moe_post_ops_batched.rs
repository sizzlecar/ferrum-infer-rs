//! Stacked SiLU·gate + weighted-sum kernels for batch > 1 (prefill).
//! See `moe_post_ops_batched.metal` for the algorithmic notes.

#![cfg(all(target_os = "macos", feature = "metal"))]

use crate::attention::metal::pipelines::MetalPipelines;
use std::ffi::c_void;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("moe_post_ops_batched.metal");

pub(crate) fn build_silu_pipeline(device: &Device) -> ComputePipelineState {
    let lib = device
        .new_library_with_source(SHADER_SRC, &CompileOptions::new())
        .expect("compile moe_post_ops_batched.metal");
    let function = lib
        .get_function("silu_mul_batched_f32", None)
        .expect("find silu_mul_batched_f32");
    device
        .new_compute_pipeline_state_with_function(&function)
        .expect("build silu_mul_batched_f32 pipeline")
}

pub(crate) fn build_wsum_pipeline(device: &Device) -> ComputePipelineState {
    let lib = device
        .new_library_with_source(SHADER_SRC, &CompileOptions::new())
        .expect("compile moe_post_ops_batched.metal");
    let function = lib
        .get_function("weighted_sum_batched_f32", None)
        .expect("find weighted_sum_batched_f32");
    device
        .new_compute_pipeline_state_with_function(&function)
        .expect("build weighted_sum_batched_f32 pipeline")
}

pub(crate) fn dispatch_silu_mul_batched(
    pipes: &MetalPipelines,
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

    let pipe = pipes.moe_post_ops_batched_silu_pipeline();
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

pub(crate) fn dispatch_weighted_sum_batched(
    pipes: &MetalPipelines,
    enc: &ComputeCommandEncoderRef,
    slots: &Buffer,
    weights: &Buffer,
    out: &Buffer,
    batch: usize,
    top_k: usize,
    hidden: usize,
) {
    dispatch_weighted_sum_batched_offset(
        pipes, enc, slots, 0, weights, 0, out, 0, batch, top_k, hidden,
    );
}

/// Offset-aware weighted_sum_batched. `weights_byte_offset` /
/// `out_byte_offset` let `weights` start mid-buffer (e.g. read row i
/// from a stacked `[M, top_k]` buffer at `i*top_k*4`) and `out` start
/// mid-buffer (e.g. write row i into a stacked `[M, hidden]` buffer
/// at `i*hidden*4`). Eliminates `copy_slice` round-trips on the
/// per-item batched-decode path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_weighted_sum_batched_offset(
    pipes: &MetalPipelines,
    enc: &ComputeCommandEncoderRef,
    slots: &Buffer,
    slots_byte_offset: u64,
    weights: &Buffer,
    weights_byte_offset: u64,
    out: &Buffer,
    out_byte_offset: u64,
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

    let pipe = pipes.moe_post_ops_batched_wsum_pipeline();
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(slots), slots_byte_offset);
    enc.set_buffer(1, Some(weights), weights_byte_offset);
    enc.set_buffer(2, Some(out), out_byte_offset);
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
