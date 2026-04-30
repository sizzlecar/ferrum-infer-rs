//! Q6_K MoE 2-D GEMM with indirect dispatch — Rust glue.
//!
//! See `q6_k_moe_id_gemm.metal`. Same shape as `q4_k_moe_id_gemm` but
//! the per-expert-slab byte layout uses Q6_K's 210-byte super-block.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q6_k_moe_id_gemm.metal");
const KERNEL_NAME: &str = "gemm_q6kw_moe_id_f32";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q6_k_moe_id_gemm.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemm_q6kw_moe_id_f32");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemm_q6kw_moe_id_f32 pipeline")
    })
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemm_q6k_moe_id_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    weights_stacked: &Buffer,
    weights_byte_offset: u64,
    src1: &Buffer,
    ids: &Buffer,
    tpe: &Buffer,
    out: &Buffer,
    num_experts: usize,
    m: usize,
    k: usize,
    ne11: usize,
    top_k: usize,
    max_per_expert: usize,
    batch: usize,
) {
    debug_assert!(k % 256 == 0);
    debug_assert!(m % 4 == 0);

    let nb01_bytes = (k / 256) * crate::q6_k_gemv::Q6_K_BLOCK_BYTES;
    let nb02_bytes = m * nb01_bytes;

    #[repr(C)]
    struct P {
        m: i32,
        k: i32,
        nb01: i32,
        nb02: i32,
        ne11: i32,
        top_k: i32,
        max_per_expert: i32,
        batch: i32,
    }
    let params = P {
        m: m as i32,
        k: k as i32,
        nb01: nb01_bytes as i32,
        nb02: nb02_bytes as i32,
        ne11: ne11 as i32,
        top_k: top_k as i32,
        max_per_expert: max_per_expert as i32,
        batch: batch as i32,
    };

    let pipe = pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(weights_stacked), weights_byte_offset);
    enc.set_buffer(1, Some(src1), 0);
    enc.set_buffer(2, Some(ids), 0);
    enc.set_buffer(3, Some(tpe), 0);
    enc.set_buffer(4, Some(out), 0);
    enc.set_bytes(
        5,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );
    enc.set_threadgroup_memory_length(0, 8192);

    const NR0: u64 = 64;
    const NR1: u64 = 32;
    let grid = MTLSize::new(
        (max_per_expert as u64).div_ceil(NR1),
        (m as u64).div_ceil(NR0),
        num_experts as u64,
    );
    let tg = MTLSize::new(128, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
}
