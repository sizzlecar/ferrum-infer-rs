//! Q4_K_M dequant — ferrum-native Metal compute kernel + Rust glue.
//!
//! The Metal shader (`q4_k_dequant.metal`) is embedded via `include_str!`
//! and compiled at runtime on first call (cached for the process). No
//! `build.rs` plumbing required.
//!
//! Block layout matches GGML / llama.cpp / candle's `BlockQ4K`
//! (see `candle_core::quantized::k_quants`):
//!   - 144 bytes per super-block of 256 weights (~4.5 bits / weight)
//!   - `[d (fp16) | dmin (fp16) | scales[12] | qs[128]]`
//!
//! Public entry: [`dequant_q4_k_to_f16`]. Drop in raw block bytes (a
//! `Buffer` viewed as `block_q4_K[]`), get back fp16 elements
//! (`Buffer` of `n_blocks * 256` halves).
//!
//! This module is the foundation for keeping Q4_K_M weights quantised
//! in Metal memory and dequant'ing on-the-fly per matmul, instead of
//! eager-dequanting at load time and burning 8× the RAM.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::sync::OnceLock;

use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState,
    Device, MTLSize,
};

/// Bytes per super-block (144 = 4 fp16 + 12 scales + 128 qs).
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Weights per super-block.
pub const Q4_K_BLOCK_ELEMENTS: usize = 256;

const SHADER_SRC: &str = include_str!("q4_k_dequant.metal");
const KERNEL_NAME: &str = "dequantize_q4_k_f16";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

/// Lazy-compile the kernel on first use; reuse the pipeline forever.
/// Per-process compilation is ~5-10 ms, dwarfed by any actual workload.
fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q4_k_dequant.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find dequantize_q4_k_f16 function in library");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build compute pipeline")
    })
}

/// Encode a Q4_K_M dequant pass on an existing command buffer. Caller
/// is responsible for committing + waiting.
///
/// `blocks_buf`: `[block_q4_K; n_blocks]`, total `n_blocks * 144` bytes
/// `out_buf`:    `[half; n_blocks * 256]`, total `n_blocks * 512` bytes
pub fn encode_dequant_q4_k_to_f16(
    device: &Device,
    cmd: &CommandBufferRef,
    blocks_buf: &Buffer,
    out_buf: &Buffer,
    n_blocks: usize,
) {
    if n_blocks == 0 {
        return;
    }
    let pipe = pipeline(device);
    let enc = cmd.new_compute_command_encoder();
    encode_dispatch(enc, pipe, blocks_buf, out_buf, n_blocks);
    enc.end_encoding();
}

fn encode_dispatch(
    enc: &ComputeCommandEncoderRef,
    pipe: &ComputePipelineState,
    blocks_buf: &Buffer,
    out_buf: &Buffer,
    n_blocks: usize,
) {
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(blocks_buf), 0);
    enc.set_buffer(1, Some(out_buf), 0);

    // Threadgroup width — 64 is a common Metal sweet spot for small
    // per-thread state. We're one-thread-per-super-block (256 weights /
    // thread), so this maps to 64 super-blocks per group.
    let threads_per_group = pipe.thread_execution_width().min(64) as u64;
    let total_threads = n_blocks as u64;
    let tg = MTLSize::new(threads_per_group, 1, 1);
    let grid = MTLSize::new(total_threads, 1, 1);
    enc.dispatch_threads(grid, tg);
}

/// Convenience: one-shot dequant. Allocates a fresh command buffer,
/// commits, and waits. Use this in tests; the per-forward path inside
/// the model should reuse a long-lived command buffer via
/// [`encode_dequant_q4_k_to_f16`].
pub fn dequant_q4_k_to_f16_blocking(
    device: &Device,
    queue: &metal::CommandQueue,
    blocks_buf: &Buffer,
    out_buf: &Buffer,
    n_blocks: usize,
) {
    let cmd = queue.new_command_buffer();
    encode_dequant_q4_k_to_f16(device, cmd, blocks_buf, out_buf, n_blocks);
    cmd.commit();
    cmd.wait_until_completed();
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CandleDevice, Tensor};
    use half::f16;
    use metal::MTLResourceOptions;

    fn close_enough(a: f32, b: f32, rel: f32, abs: f32) -> bool {
        let diff = (a - b).abs();
        diff <= abs || diff <= rel * b.abs().max(a.abs())
    }

    /// End-to-end correctness check: ferrum's Metal Q4_K_M dequant kernel
    /// must match candle's CPU reference dequant within fp16 quantisation
    /// error. The kernel is one-thread-per-super-block; this test pushes
    /// 4 super-blocks (1024 weights total) through.
    #[test]
    fn metal_q4_k_dequant_matches_candle_cpu_reference() {
        // Sample data — non-trivial pattern (not all-zero, not constant).
        let n_blocks: usize = 4;
        let n_elem: usize = n_blocks * Q4_K_BLOCK_ELEMENTS; // 1024
        let raw: Vec<f32> = (0..n_elem)
            .map(|i| ((i as f32 * 0.0173).sin() + (i as f32 * 0.0091).cos()) * 0.5)
            .collect();

        // Quantise with candle to get a real Q4_K_M block payload.
        let cpu = CandleDevice::Cpu;
        let t = Tensor::from_vec(raw.clone(), n_elem, &cpu).unwrap();
        let qt = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();

        // Reference: candle's CPU dequant (also goes through k_quants::to_float).
        let dense_ref = qt.dequantize(&cpu).unwrap();
        let ref_f32: Vec<f32> = dense_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(ref_f32.len(), n_elem);

        // Extract the raw on-disk block bytes from the QTensor.
        let bytes = qt.data().expect("read QTensor bytes");
        assert_eq!(
            bytes.len(),
            n_blocks * Q4_K_BLOCK_BYTES,
            "expected {n_blocks} super-blocks × 144 bytes"
        );

        // Set up Metal: device, queue, input/output buffers.
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device available — skipping");
            return;
        };
        let queue = device.new_command_queue();

        let blocks_buf = device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (n_elem * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Run our kernel.
        dequant_q4_k_to_f16_blocking(&device, &queue, &blocks_buf, &out_buf, n_blocks);

        // Read back fp16 → fp32.
        let out_ptr = out_buf.contents() as *const f16;
        let our_f16: Vec<f16> = unsafe { std::slice::from_raw_parts(out_ptr, n_elem) }.to_vec();
        let our_f32: Vec<f32> = our_f16.iter().map(|h| h.to_f32()).collect();

        // Compare. fp16 quant adds ~2^-10 relative error on top of the
        // Q4_K_M quant itself; the reference is fp32 of the same Q4 data,
        // so any divergence is just our kernel rounding to fp16.
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        let mut mismatches = 0;
        for (i, (&our, &refv)) in our_f32.iter().zip(ref_f32.iter()).enumerate() {
            let diff = (our - refv).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            let denom = refv.abs().max(our.abs()).max(1e-6);
            let rel = diff / denom;
            if rel > max_rel {
                max_rel = rel;
            }
            // Tolerance: fp16 has ~3-4 decimal digits, weights are ~[-1,1] range.
            // 1e-2 absolute is safe; tighter would risk flake on edge values.
            if !close_enough(our, refv, 1e-2, 1e-3) {
                mismatches += 1;
                if mismatches < 5 {
                    eprintln!("[{i}] our={our} ref={refv} diff={diff}");
                }
            }
        }
        eprintln!(
            "max_abs_diff={max_abs:.6} max_rel_diff={max_rel:.6} mismatches={mismatches}/{n_elem}"
        );
        assert!(
            mismatches == 0,
            "{mismatches}/{n_elem} elements outside fp16 tolerance"
        );
    }

    /// Larger workload — 4 KiB super-blocks (1M weights), exercise grid
    /// dispatch sizes that won't fit in one threadgroup.
    #[test]
    fn metal_q4_k_dequant_handles_thousands_of_blocks() {
        let n_blocks: usize = 4096; // 1M weights
        let n_elem: usize = n_blocks * Q4_K_BLOCK_ELEMENTS;
        let raw: Vec<f32> = (0..n_elem)
            .map(|i| (i as f32 * 1.7e-4).sin() * 0.7)
            .collect();

        let cpu = CandleDevice::Cpu;
        let t = Tensor::from_vec(raw, n_elem, &cpu).unwrap();
        let qt = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
        let dense_ref = qt.dequantize(&cpu).unwrap();
        let ref_f32: Vec<f32> = dense_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bytes = qt.data().unwrap();

        let Some(device) = Device::system_default() else {
            return;
        };
        let queue = device.new_command_queue();
        let blocks_buf = device.new_buffer_with_data(
            bytes.as_ptr() as *const _,
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (n_elem * std::mem::size_of::<f16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        dequant_q4_k_to_f16_blocking(&device, &queue, &blocks_buf, &out_buf, n_blocks);

        let out_ptr = out_buf.contents() as *const f16;
        let our_f16: &[f16] = unsafe { std::slice::from_raw_parts(out_ptr, n_elem) };
        let mut mismatches = 0;
        for (i, h) in our_f16.iter().enumerate() {
            let our = h.to_f32();
            let r = ref_f32[i];
            if (our - r).abs() > 1e-2 && (our - r).abs() / r.abs().max(1e-6) > 1e-2 {
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "{mismatches}/{n_elem} elements diverged from candle CPU reference"
        );
    }
}
