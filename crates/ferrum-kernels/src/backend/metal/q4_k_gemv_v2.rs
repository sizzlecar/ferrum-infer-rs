//! Q4_K_M GEMV — v2, modeled on llama.cpp's `kernel_mul_mv_q4_K_f32_impl`.
//!
//! See `q4_k_gemv_v2.metal` for the kernel-side comments. Compared to v1
//! (`q4_k_gemv.rs`), this kernel:
//!
//! - Processes **2 output rows per simdgroup** (`N_R0 = 2`), reusing
//!   activations between rows.
//! - Uses **2 simdgroups per threadgroup** (`N_SG = 2`), so each
//!   threadgroup is 64 threads handling 4 consecutive output rows.
//! - Halves activation bandwidth and quadruples occupancy per
//!   threadgroup vs the 1-row-per-simdgroup v1.
//!
//! Layout assumption: `K % 256 == 0`. `src0` is row-major
//! `block_q4_K[N * (K/256)]`.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q4_k_gemv_v2.metal");
const KERNEL_NAME: &str = "gemv_f32a_q4kw_v2";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q4_k_gemv_v2.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemv_f32a_q4kw_v2 function");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemv_f32a_q4kw_v2 pipeline")
    })
}

/// Dispatch the v2 fused GEMV on an existing compute encoder.
///
/// `src0`: `block_q4_K[N * (K/256)]` row-major super-blocks
/// `a`:    `f32[K]` activations
/// `c`:    `f32[N]` output (written by the kernel)
/// `n`:    out_features (must satisfy n % 4 == 0 — block size is 4 rows)
/// `k`:    in_features (must satisfy k % 256 == 0)
pub fn dispatch_gemv_q4k_v2_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    src0: &Buffer,
    src0_byte_offset: u64,
    c: &Buffer,
    n: usize,
    k: usize,
) {
    dispatch_gemv_q4k_v2_offset(device, enc, a, 0, src0, src0_byte_offset, c, 0, n, k);
}

/// Same as [`dispatch_gemv_q4k_v2_on_encoder`] with byte offsets into
/// `a` and `c`. `src0_byte_offset` is the byte offset into the weight
/// buffer where this tensor's super-blocks start (non-zero when `src0`
/// is a shared zero-copy mmap buffer). Used by `MultiQuantLinear` to
/// write each fused-projection part into a disjoint slice of the shared
/// output buffer.
pub fn dispatch_gemv_q4k_v2_offset(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    a_offset_bytes: u64,
    src0: &Buffer,
    src0_byte_offset: u64,
    c: &Buffer,
    c_offset_bytes: u64,
    n: usize,
    k: usize,
) {
    debug_assert!(k % 256 == 0, "K must be a multiple of 256 (got {k})");

    let nb01_bytes = (k / 256) * 144;

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
    enc.set_buffer(0, Some(src0), src0_byte_offset);
    enc.set_buffer(1, Some(a), a_offset_bytes);
    enc.set_buffer(2, Some(c), c_offset_bytes);
    enc.set_bytes(
        3,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    const TILE_ROWS: u64 = 4;
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

    /// Compare v2 GEMV against candle's CPU `dequantize → matmul` reference
    /// at a 4K×4K shape (Qwen3-8B o_proj exactly).
    #[test]
    fn fused_gemv_q4k_v2_matches_cpu_reference_4096x4096() {
        let n: usize = 4096;
        let k: usize = 4096;

        let raw_w: Vec<f32> = (0..n * k)
            .map(|i| {
                ((((i % 313) as f32) * 0.0173).sin() + (((i % 251) as f32) * 0.0091).cos()) * 0.5
            })
            .collect();
        let cpu = CandleDevice::Cpu;
        let t_w = Tensor::from_vec(raw_w, (n, k), &cpu).unwrap();
        let qt_w = QTensor::quantize(&t_w, GgmlDType::Q4K).unwrap();
        let dense_w = qt_w.dequantize(&cpu).unwrap();

        let raw_a: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.0007).sin()).collect();
        let t_a = Tensor::from_vec(raw_a.clone(), (1, k), &cpu).unwrap();

        // Reference: A @ W^T via candle CPU.
        let ref_t = t_a.matmul(&dense_w.transpose(0, 1).unwrap()).unwrap();
        let ref_c: Vec<f32> = ref_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(ref_c.len(), n);

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
        dispatch_gemv_q4k_v2_on_encoder(&device, enc, &a_buf, &w_buf, 0, &c_buf, n, k);
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
        eprintln!("v2 max_abs={max_abs:.4} mismatches={mismatches}/{n}");
        assert!(
            mismatches == 0,
            "v2: {mismatches}/{n} elements outside fp16 tolerance"
        );
    }
}
