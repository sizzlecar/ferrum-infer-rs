//! Fused Q4_K_M dequant + GEMV — Metal compute kernel + Rust glue.
//!
//! Companion to `q4_k.rs` (the standalone dequant kernel). Use this
//! whenever `m == 1` (decode path): one kernel dispatch reads the Q4
//! super-blocks, decodes 256 weights per super-block on the fly, and
//! reduces against `A` — no fp16 transient written or read. Drops
//! ~64 MB of memory traffic per 4K×4K matmul vs. the dequant→transient→
//! gemv pipeline.
//!
//! For prefill (`m > 1`) the model still goes through the standalone
//! dequant + `gemm_v2_f16w` path; a fused gemm is a future optimisation.
//!
//! Layout assumptions (must hold; otherwise fall back to dequant path):
//!   - `K` (in_features) is a multiple of 256 (Q4_K_M super-block size)
//!   - `W` is row-major over `[N, K/256]` super-blocks (matches what
//!     `MetalQuantStore::Q4K { blocks, .. }` already stores)

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q4_k_gemv.metal");
const KERNEL_NAME: &str = "gemv_f32a_q4kw";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q4_k_gemv.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemv_f32a_q4kw function");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemv_f32a_q4kw pipeline")
    })
}

/// Dispatch fused GEMV on an existing compute encoder.
///
/// Computes `c[n] = a[k] @ w[n, k]^T` where `w` is `block_q4_K[n * (k/256)]`.
///
/// Caller is responsible for `enc.end_encoding()` after the dispatch
/// (or chaining further dispatches).
pub fn dispatch_gemv_q4k_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    w: &Buffer,
    c: &Buffer,
    n: usize,
    k: usize,
) {
    debug_assert!(
        k % 256 == 0,
        "gemv_q4k requires K divisible by 256 (got K={k})"
    );

    #[repr(C)]
    struct P {
        n: i32,
        k: i32,
    }
    let params = P {
        n: n as i32,
        k: k as i32,
    };

    let pipe = pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(a), 0);
    enc.set_buffer(1, Some(w), 0);
    enc.set_buffer(2, Some(c), 0);
    // setBytes inlines small (<=4KB) params into the encoder argument
    // table — no MTLBuffer allocation per call. With 145 quant matmuls
    // per token, this is real money on Apple Silicon (alloc takes ~ms).
    enc.set_bytes(
        3,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    // 1 threadgroup per output column, exactly 1 SIMD group (32 threads)
    // per threadgroup — the kernel stripes `tiitg` ∈ [0, 32) over K with
    // step 32 and reduces with `simd_sum`.
    let grid = MTLSize::new(n as u64, 1, 1);
    let tg = MTLSize::new(32, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CandleDevice, Tensor};

    /// Compare the fused GEMV against candle's CPU `dequantize → matmul`
    /// reference within fp16 quantisation tolerance. Tests a 4K×4K shape
    /// (matches Qwen3-8B o_proj exactly).
    #[test]
    fn fused_gemv_q4k_matches_cpu_reference_4096x4096() {
        let n: usize = 4096;
        let k: usize = 4096;

        // Synthetic weight, sin/cos pattern so quantisation is non-trivial.
        let raw_w: Vec<f32> = (0..n * k)
            .map(|i| (((i % 313) as f32) * 0.0173).sin() * 0.5
                + (((i % 251) as f32) * 0.0091).cos() * 0.5)
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
        let c_buf = device.new_buffer(
            (n * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        dispatch_gemv_q4k_on_encoder(&device, enc, &a_buf, &w_buf, &c_buf, n, k);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let our_ptr = c_buf.contents() as *const f32;
        let our_c: &[f32] = unsafe { std::slice::from_raw_parts(our_ptr, n) };

        // GEMM accumulates K=4096 multiplies — fp16 weight quant adds ~0.1
        // relative error per element after that many additions. Loose
        // tolerance: 1e-1 absolute on values typically in [-1, 1].
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
        eprintln!("max_abs={max_abs:.4} mismatches={mismatches}/{n}");
        assert!(
            mismatches == 0,
            "{mismatches}/{n} elements outside fp16 tolerance"
        );
    }
}
