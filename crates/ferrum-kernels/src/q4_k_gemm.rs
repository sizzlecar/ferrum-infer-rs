//! Q4_K_M GEMM (m>1 prefill path) — ported from llama.cpp's
//! `kernel_mul_mm_q4_K_f32` (legacy non-tensor-API variant). Replaces
//! ferrum's old `dequant_q4_k → fp16 transient → gemm_v2_f16w`
//! sequence with a single fused dispatch:
//!
//!   - Output tile: 64 weight rows × 32 activation rows per
//!     threadgroup, computed via `simdgroup_half8x8` matrix multiply
//!   - 4 simdgroups × 32 threads = 128 threads per threadgroup
//!   - Q4_K dequant inlined into the threadgroup-memory load (no fp16
//!     transient buffer)
//!   - shmem: 8 KiB (4096 B for half-A + 4096 B for half-B)
//!
//! Expected wins over the existing prefill path:
//!   - No fp16 transient write+read (≈ 2× memory traffic saved)
//!   - simdgroup matrix multiply (M1 GPU compute units' fast path) vs
//!     gemm_v2_f16w which loops over scalars
//!
//! Layout assumptions: src0 (weights) is row-major
//! `block_q4_K[N * (K/256)]`, src1 (activations) is row-major `f32[M, K]`,
//! dst is row-major `f32[M, N]`. K must be a multiple of 256.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q4_k_gemm.metal");
const KERNEL_NAME: &str = "gemm_q4kw_f32a_f32o";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q4_k_gemm.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemm_q4kw_f32a_f32o");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemm_q4kw_f32a_f32o pipeline")
    })
}

/// Dispatch the fused Q4_K GEMM on an existing compute encoder.
///
/// `src0`: `block_q4_K[N * (K/256)]` row-major
/// `a`:    `f32[M, K]` activations (row-major, M = batch tokens)
/// `c`:    `f32[M, N]` output
/// `n`:    out_features (must be divisible by 64 for the fast write
///         path; smaller is handled but with a slightly slower
///         partial-tile store)
/// `k`:    in_features (multiple of 256)
/// `m`:    activation rows (batch size)
pub fn dispatch_gemm_q4k_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    src0: &Buffer,
    src0_byte_offset: u64,
    c: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) {
    // Standalone case: dst stride = n (out_features), no column offset.
    dispatch_gemm_q4k_part(device, enc, a, src0, src0_byte_offset, c, 0, m, n, n, k);
}

/// Strided variant for `MultiQuantLinear` Fused: writes one part of a
/// fused output, occupying `[c_offset_cols, c_offset_cols + n)` columns
/// of a `[m, stride_c]` output buffer. `n` is the part's row count
/// (write width per output row); `stride_c` is the total fused-output
/// row stride. `src0_byte_offset` is the byte offset into the weight
/// buffer (non-zero when `src0` is a shared zero-copy mmap buffer).
pub fn dispatch_gemm_q4k_part(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    src0: &Buffer,
    src0_byte_offset: u64,
    c: &Buffer,
    c_offset_cols: usize,
    m: usize,
    n: usize,
    stride_c: usize,
    k: usize,
) {
    debug_assert!(k % 256 == 0, "K must be a multiple of 256 (got {k})");
    debug_assert!(
        n <= stride_c,
        "n ({n}) must fit within stride_c ({stride_c})"
    );
    debug_assert!(
        c_offset_cols + n <= stride_c,
        "part [{c_offset_cols}, {}) overflows stride_c {stride_c}",
        c_offset_cols + n
    );

    let nb01_bytes = (k / 256) * 144;

    #[repr(C)]
    struct P {
        m: i32,
        n: i32,
        k: i32,
        nb01: i32,
        stride_c: i32,
    }
    let params = P {
        m: n as i32, // write width = part rows
        n: m as i32, // batch (activation rows)
        k: k as i32,
        nb01: nb01_bytes as i32,
        stride_c: stride_c as i32, // dst row stride = total output rows
    };

    let pipe = pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(src0), src0_byte_offset);
    enc.set_buffer(1, Some(a), 0);
    enc.set_buffer(2, Some(c), (c_offset_cols * 4) as u64);
    enc.set_bytes(
        3,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );
    enc.set_threadgroup_memory_length(0, 8192);

    const NR0: u64 = 64;
    const NR1: u64 = 32;
    let grid = MTLSize::new((m as u64).div_ceil(NR1), (n as u64).div_ceil(NR0), 1);
    let tg = MTLSize::new(128, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CandleDevice, Tensor};
    use metal::MTLResourceOptions;

    /// 64×32 single-tile correctness test (smallest meaningful shape).
    #[test]
    fn fused_gemm_q4k_smoke_64x4096_x_32() {
        let n: usize = 64; // weight rows / out_features
        let k: usize = 256; // smallest valid (1 super-block per row)
        let m: usize = 32; // batch tokens

        let raw_w: Vec<f32> = (0..n * k)
            .map(|i| {
                ((((i % 313) as f32) * 0.0173).sin() + (((i % 251) as f32) * 0.0091).cos()) * 0.5
            })
            .collect();
        let cpu = CandleDevice::Cpu;
        let t_w = Tensor::from_vec(raw_w, (n, k), &cpu).unwrap();
        let qt_w = QTensor::quantize(&t_w, GgmlDType::Q4K).unwrap();
        let dense_w = qt_w.dequantize(&cpu).unwrap(); // [n, k]

        let raw_a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.0007).sin()).collect();
        let t_a = Tensor::from_vec(raw_a.clone(), (m, k), &cpu).unwrap();

        // Reference: A @ W^T → [m, n]
        let ref_t = t_a.matmul(&dense_w.transpose(0, 1).unwrap()).unwrap();
        let ref_c: Vec<f32> = ref_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(ref_c.len(), m * n);

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
        let c_buf = device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        dispatch_gemm_q4k_on_encoder(&device, enc, &a_buf, &w_buf, 0, &c_buf, m, n, k);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let our_ptr = c_buf.contents() as *const f32;
        let our_c: &[f32] = unsafe { std::slice::from_raw_parts(our_ptr, m * n) };

        // GEMM precision: K=256 multiplies, fp16 storage → tolerance loose.
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
        eprintln!(
            "q4k mul_mm 64x32 max_abs={max_abs:.4} mismatches={mismatches}/{}",
            m * n
        );
        assert!(
            mismatches == 0,
            "q4k mul_mm: {mismatches}/{} elements outside tolerance",
            m * n
        );
    }

    /// Real prefill-shape correctness test: 4096×4096 weight, m=11 prompt.
    #[test]
    fn fused_gemm_q4k_4096x4096_x_11() {
        let n: usize = 4096;
        let k: usize = 4096;
        let m: usize = 11;

        let raw_w: Vec<f32> = (0..n * k)
            .map(|i| {
                ((((i % 313) as f32) * 0.0173).sin() + (((i % 251) as f32) * 0.0091).cos()) * 0.5
            })
            .collect();
        let cpu = CandleDevice::Cpu;
        let t_w = Tensor::from_vec(raw_w, (n, k), &cpu).unwrap();
        let qt_w = QTensor::quantize(&t_w, GgmlDType::Q4K).unwrap();
        let dense_w = qt_w.dequantize(&cpu).unwrap();

        let raw_a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.0007).sin()).collect();
        let t_a = Tensor::from_vec(raw_a.clone(), (m, k), &cpu).unwrap();

        let ref_t = t_a.matmul(&dense_w.transpose(0, 1).unwrap()).unwrap();
        let ref_c: Vec<f32> = ref_t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let bytes = qt_w.data().unwrap();
        let Some(device) = Device::system_default() else {
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
        let c_buf = device.new_buffer((m * n * 4) as u64, MTLResourceOptions::StorageModeShared);

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        dispatch_gemm_q4k_on_encoder(&device, enc, &a_buf, &w_buf, 0, &c_buf, m, n, k);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let our_ptr = c_buf.contents() as *const f32;
        let our_c: &[f32] = unsafe { std::slice::from_raw_parts(our_ptr, m * n) };

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
                    eprintln!(
                        "[{i}] m={} n={} our={our} ref={refv} diff={diff}",
                        i / n,
                        i % n
                    );
                }
            }
        }
        eprintln!(
            "q4k mul_mm 4096x4096 m=11: max_abs={max_abs:.4} mismatches={mismatches}/{}",
            m * n
        );
        assert!(
            mismatches == 0,
            "q4k mul_mm 4096x4096: {mismatches} elements diverge"
        );
    }
}
