//! Q6_K fused GEMV — adapted from llama.cpp's
//! `kernel_mul_mv_q6_K_f32_impl`. Same N_R0=2, N_SG=2 threadgroup layout
//! as `q4_k_gemv_v2`. See `q6_k_gemv.metal` for the kernel.
//!
//! Q6_K block layout (256 weights / 210 bytes / 6.5 bits/w):
//!   - `ql[128]`: lower 4 bits of each weight
//!   - `qh[64]`:  upper 2 bits, 4 weights packed per byte
//!   - `scales[16]`: 16 sub-block int8 scales
//!   - `d`:        super-block half-precision scale
//!
//! Used for Qwen3-8B Q4_K_M's `attn_v` / `ffn_down` / `output` tensors,
//! which llama.cpp's quantizer upgrades from Q4_K → Q6_K. Without this
//! kernel they fall through to the eager-fp32 GgufLinear path, blowing
//! a 5 GB Q6 footprint up to ~10 GB of fp32 in MTLBuffer.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q6_k_gemv.metal");
const KERNEL_NAME: &str = "gemv_f32a_q6kw_v2";

/// Q6_K super-block size in bytes (matches llama.cpp's `block_q6_K`).
pub const Q6_K_BLOCK_BYTES: usize = 128 + 64 + 16 + 2;
pub const Q6_K_BLOCK_ELEMENTS: usize = 256;

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q6_k_gemv.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemv_f32a_q6kw_v2 function");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemv_f32a_q6kw_v2 pipeline")
    })
}

/// Dispatch the Q6_K fused GEMV on an existing compute encoder.
///
/// `src0`: `block_q6_K[N * (K/256)]` row-major super-blocks
/// `a`:    `f32[K]` activations (single row)
/// `c`:    `f32[N]` output (single row)
pub fn dispatch_gemv_q6k_v2_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    src0: &Buffer,
    c: &Buffer,
    n: usize,
    k: usize,
) {
    dispatch_gemv_q6k_v2_offset(device, enc, a, 0, src0, c, 0, n, k);
}

/// Same as [`dispatch_gemv_q6k_v2_on_encoder`] but with byte offsets into
/// `a` and `c`. Used by the m>1 prefill path to batch a multi-row matmul
/// as a sequence of gemv calls without having to write a separate gemm
/// kernel just for Q6_K.
pub fn dispatch_gemv_q6k_v2_offset(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_offset_bytes: u64,
    src0: &Buffer,
    c: &Buffer,
    c_offset_bytes: u64,
    n: usize,
    k: usize,
) {
    debug_assert!(k % 256 == 0, "K must be a multiple of 256 (got {k})");

    let nb01_bytes = (k / 256) * Q6_K_BLOCK_BYTES;

    #[repr(C)]
    struct P {
        n: i32,
        k: i32,
        nb01: i32,
    }
    let params = P {
        n: n as i32,
        k: k as i32,
        nb01: nb01_bytes as i32,
    };

    let pipe = pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(src0), 0);
    enc.set_buffer(1, Some(a), a_offset_bytes);
    enc.set_buffer(2, Some(c), c_offset_bytes);
    enc.set_bytes(
        3,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    const TILE_ROWS: u64 = 4; // N_R0 * N_SG
    let grid = MTLSize::new((n as u64).div_ceil(TILE_ROWS), 1, 1);
    let tg = MTLSize::new(32, 2, 1);
    enc.dispatch_thread_groups(grid, tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CandleDevice, Tensor};
    use metal::MTLResourceOptions;

    /// Verify Q6_K GEMV matches candle's CPU reference within fp16
    /// quantisation tolerance, on a Qwen3-8B `down_proj`-shaped
    /// 4096×12288 tensor.
    #[test]
    fn fused_gemv_q6k_matches_cpu_reference() {
        let n: usize = 4096;
        let k: usize = 12288;

        let raw_w: Vec<f32> = (0..n * k)
            .map(|i| ((((i % 313) as f32) * 0.0173).sin()
                + (((i % 251) as f32) * 0.0091).cos()) * 0.5)
            .collect();
        let cpu = CandleDevice::Cpu;
        let t_w = Tensor::from_vec(raw_w, (n, k), &cpu).unwrap();
        let qt_w = QTensor::quantize(&t_w, GgmlDType::Q6K).unwrap();
        let dense_w = qt_w.dequantize(&cpu).unwrap();

        let raw_a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.0007).sin()).collect();
        let t_a = Tensor::from_vec(raw_a.clone(), (1, k), &cpu).unwrap();
        let ref_t = t_a.matmul(&dense_w.transpose(0, 1).unwrap()).unwrap();
        let ref_c: Vec<f32> = ref_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let bytes = qt_w.data().unwrap();

        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device — skipping");
            return;
        };
        let queue = device.new_command_queue();

        let a_buf = device.new_buffer_with_data(
            raw_a.as_ptr() as *const _,
            (raw_a.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w_buf = device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let c_buf = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        dispatch_gemv_q6k_v2_on_encoder(&device, enc, &a_buf, &w_buf, &c_buf, n, k);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let our_ptr = c_buf.contents() as *const f32;
        let our_c: &[f32] = unsafe { std::slice::from_raw_parts(our_ptr, n) };

        let mut max_abs = 0.0_f32;
        let mut mismatches = 0usize;
        for (i, (&our, &refv)) in our_c.iter().zip(ref_c.iter()).enumerate() {
            let diff = (our - refv).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            let denom = our.abs().max(refv.abs()).max(1e-3);
            let rel = diff / denom;
            if diff > 0.5 && rel > 0.05 {
                mismatches += 1;
                if mismatches < 5 {
                    eprintln!("[{i}] our={our} ref={refv} diff={diff} rel={rel}");
                }
            }
        }
        eprintln!("q6k max_abs={max_abs:.4} mismatches={mismatches}/{n}");
        assert!(
            mismatches == 0,
            "q6k: {mismatches}/{n} elements outside fp16 tolerance"
        );
    }
}
