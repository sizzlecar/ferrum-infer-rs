//! Q6_K GEMM (m > 1 prefill path) — same template as `q4_k_gemm.rs`,
//! adapted with `dequantize_q6_K`. Eliminates the per-row gemv loop
//! used by `gemm_quant Q6K m>1` (which scaled linearly with prompt
//! length and was the dominant remaining prefill bottleneck after
//! Q4_K mul_mm landed).

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q6_k_gemm.metal");
const KERNEL_NAME: &str = "gemm_q6kw_f32a_f32o";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q6_k_gemm.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemm_q6kw_f32a_f32o");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemm_q6kw_f32a_f32o pipeline")
    })
}

pub fn dispatch_gemm_q6k_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    src0: &Buffer,
    c: &Buffer,
    m: usize,
    n: usize,
    k: usize,
) {
    dispatch_gemm_q6k_part(device, enc, a, src0, c, 0, m, n, n, k);
}

/// Strided variant — see `q4_k_gemm::dispatch_gemm_q4k_part` for
/// rationale. Writes part columns `[c_offset_cols, c_offset_cols + n)`
/// of a `[m, stride_c]` output buffer.
pub fn dispatch_gemm_q6k_part(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    src0: &Buffer,
    c: &Buffer,
    c_offset_cols: usize,
    m: usize,
    n: usize,
    stride_c: usize,
    k: usize,
) {
    debug_assert!(k % 256 == 0, "K must be a multiple of 256 (got {k})");
    debug_assert!(c_offset_cols + n <= stride_c);

    let nb01_bytes = (k / 256) * crate::q6_k_gemv::Q6_K_BLOCK_BYTES;

    #[repr(C)]
    struct P {
        m: i32,
        n: i32,
        k: i32,
        nb01: i32,
        stride_c: i32,
    }
    let params = P {
        m: n as i32,
        n: m as i32,
        k: k as i32,
        nb01: nb01_bytes as i32,
        stride_c: stride_c as i32,
    };

    let pipe = pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(src0), 0);
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

    /// Real prefill-shape correctness test for Q6_K mul_mm at the
    /// Qwen3-8B `down_proj` shape (4096 out × 12288 in, m=11).
    #[test]
    fn fused_gemm_q6k_4096x12288_x_11() {
        let n: usize = 4096;
        let k: usize = 12288;
        let m: usize = 11;

        let raw_w: Vec<f32> = (0..n * k)
            .map(|i| {
                ((((i % 313) as f32) * 0.0173).sin() + (((i % 251) as f32) * 0.0091).cos()) * 0.5
            })
            .collect();
        let cpu = CandleDevice::Cpu;
        let t_w = Tensor::from_vec(raw_w, (n, k), &cpu).unwrap();
        let qt_w = QTensor::quantize(&t_w, GgmlDType::Q6K).unwrap();
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
        dispatch_gemm_q6k_on_encoder(&device, enc, &a_buf, &w_buf, &c_buf, m, n, k);
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
            "q6k mul_mm 4096x12288 m=11: max_abs={max_abs:.4} mismatches={mismatches}/{}",
            m * n
        );
        assert!(
            mismatches == 0,
            "q6k mul_mm: {mismatches} elements outside tolerance"
        );
    }
}
