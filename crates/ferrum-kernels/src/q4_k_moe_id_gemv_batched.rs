//! Q4_K_M MoE indirect-dispatch GEMV — **batched over M tokens**.
//!
//! See `q4_k_moe_id_gemv_batched.metal` for algorithmic notes. Single
//! Metal dispatch covers all `m * top_k` (token, expert) pairs at once,
//! eliminating ferrum's per-token outer loop in the engine that
//! currently emits ~16× the dispatches llama.cpp emits at c=16.
//!
//! The src1 indexing is a 2D walk (token × slot) controlled by
//! `src1_outer_stride` and `src1_inner_stride` (both in **floats**):
//!
//!   gate / up : src1 = `norm_out[m, K]`
//!               outer = K, inner = 0
//!               (slots within a token broadcast — same activation row)
//!   down      : src1 = `silu_stacked[m, top_k, K]`
//!               outer = top_k * K, inner = K
//!               (each pair has its own row)

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q4_k_moe_id_gemv_batched.metal");
const KERNEL_NAME: &str = "gemv_q4kw_moe_id_batched_f32";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q4_k_moe_id_gemv_batched.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemv_q4kw_moe_id_batched_f32 function");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemv_q4kw_moe_id_batched_f32 pipeline")
    })
}

/// Dispatch the batched MoE GEMV on an existing compute encoder.
///
/// `weights_stacked` : `[num_experts, n, k/256]` Q4_K block buffer,
///                     stride `nb02 = n * (k/256) * 144` bytes per
///                     expert.
/// `a`               : activation buffer; the per-pair row is selected
///                     by `(token_idx, slot_idx)` decomposition of the
///                     pair index, with offsets controlled by
///                     `src1_outer_stride` and `src1_inner_stride`
///                     (both in elements / floats).
/// `ids`             : `[m * top_k]` flat selected-expert IDs (i32),
///                     pair_idx = token_idx * top_k + slot_idx.
/// `out`             : `[m * top_k, n]` output rows.
/// `n`, `k`          : per-expert out_features / in_features.
/// `m`, `top_k`      : token batch size and selected experts per token.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemv_q4k_moe_id_batched_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    weights_stacked: &Buffer,
    weights_byte_offset: u64,
    ids: &Buffer,
    out: &Buffer,
    n: usize,
    k: usize,
    m: usize,
    top_k: usize,
    src1_outer_stride: usize,
    src1_inner_stride: usize,
) {
    debug_assert!(k % 256 == 0, "K must be a multiple of 256 (got {k})");
    debug_assert!(n % 4 == 0, "N must be a multiple of 4 (got {n})");
    debug_assert!(top_k > 0 && m > 0);

    let nb01_bytes = (k / 256) * 144;
    let nb02_bytes = n * nb01_bytes;
    let n_pairs = m * top_k;

    #[repr(C)]
    struct P {
        n: i32,
        k: i32,
        nb01: i32,
        nb02: i32,
        top_k: i32,
        n_pairs: i32,
        src1_outer_stride: i32,
        src1_inner_stride: i32,
    }
    let params = P {
        n: n as i32,
        k: k as i32,
        nb01: nb01_bytes as i32,
        nb02: nb02_bytes as i32,
        top_k: top_k as i32,
        n_pairs: n_pairs as i32,
        src1_outer_stride: src1_outer_stride as i32,
        src1_inner_stride: src1_inner_stride as i32,
    };

    let pipe = pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(weights_stacked), weights_byte_offset);
    enc.set_buffer(1, Some(a), 0);
    enc.set_buffer(2, Some(ids), 0);
    enc.set_buffer(3, Some(out), 0);
    enc.set_bytes(
        4,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    const TILE_ROWS: u64 = 4;
    let grid = MTLSize::new((n as u64).div_ceil(TILE_ROWS), 1, n_pairs as u64);
    let tg = MTLSize::new(32, 2, 1);
    enc.dispatch_thread_groups(grid, tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CDevice, Tensor};
    use metal::MTLResourceOptions;

    /// Batched kernel must produce the same outputs as M sequential
    /// invocations of the per-token `gemv_q4kw_moe_id_f32` kernel —
    /// they share the inner Q4_K decode loop verbatim.
    #[test]
    fn batched_matches_per_token_q4k_moe_gemv_gate_up() {
        const NUM_EXPERTS: usize = 4;
        const N: usize = 64;
        const K: usize = 256;
        const M: usize = 3;
        const TOP_K: usize = 2;

        let cpu = CDevice::Cpu;

        // Per-expert quantized weight slabs (Q4_K bytes), concatenated
        // into a single stack matching `load_q4k_experts` layout.
        let mut weights_bytes = Vec::new();
        for e in 0..NUM_EXPERTS {
            let raw: Vec<f32> = (0..N * K)
                .map(|i| ((((i + e * 313) % 251) as f32) * 0.013).sin() * 0.4 + 0.05)
                .collect();
            let t = Tensor::from_vec(raw, (N, K), &cpu).unwrap();
            let qt = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
            weights_bytes.extend_from_slice(&qt.data().unwrap());
        }
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 144;
        assert_eq!(
            weights_bytes.len(),
            NUM_EXPERTS * N * (K / QK_K) * BLOCK_BYTES
        );

        // Activation: M tokens × K floats.
        let act: Vec<f32> = (0..M * K)
            .map(|i| ((i as f32) * 0.0021).cos() * 0.7)
            .collect();
        // Selected experts per (token, slot). Flat layout [M * TOP_K].
        let ids: Vec<i32> = vec![1, 3, 0, 2, 3, 1];
        assert_eq!(ids.len(), M * TOP_K);

        let Some(device) = metal::Device::system_default() else {
            eprintln!("no Metal device — skipping");
            return;
        };
        let queue = device.new_command_queue();

        let a_buf = device.new_buffer_with_data(
            act.as_ptr() as *const _,
            (act.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ids_buf = device.new_buffer_with_data(
            ids.as_ptr() as *const _,
            (ids.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w_buf = device.new_buffer_with_data(
            weights_bytes.as_ptr() as *const _,
            weights_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Per-token reference: for each token `t`, run the existing
        // per-token kernel with src1 starting at row `t*K` and
        // `src1_stride=0` (broadcast across slots within token), write
        // into a dst slice of shape [TOP_K, N] at offset `t*TOP_K*N`.
        // Concatenated, that's the [M*TOP_K, N] output the batched
        // kernel produces.
        let out_size = (M * TOP_K * N * 4) as u64;
        let dst_per_token = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);
        let dst_batched = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

        // Per-token reference path.
        let cmd1 = queue.new_command_buffer();
        let enc1 = cmd1.new_compute_command_encoder();
        for t in 0..M {
            // a slice → at offset t*K floats, ids slice → at offset t*TOP_K i32s.
            // The existing kernel doesn't take offsets natively; both buffers
            // are addressed via the `_offset_on_encoder` variant.
            crate::q4_k_moe_id_gemv::dispatch_gemv_q4k_moe_id_offset_on_encoder(
                &device,
                enc1,
                &a_buf,
                (t * K * std::mem::size_of::<f32>()) as u64,
                &w_buf,
                0,
                &ids_buf,
                (t * TOP_K * std::mem::size_of::<i32>()) as u64,
                &dst_per_token,
                N,
                K,
                TOP_K,
                0, // broadcast within token (gate/up)
            );
            // The offset-aware kernel always writes `out` from offset 0;
            // we need the output spread across [M*TOP_K, N], so swap
            // the dst buffer per iteration via a copy. Easiest: separate
            // small dst slices and a copy kernel — but for a unit test
            // we can just give each iteration a fresh buffer slice
            // through Metal's argument bindings. Since
            // `dispatch_gemv_q4k_moe_id_offset_on_encoder` always sets
            // `dst[3]` to offset 0, we'd overwrite each call. Workaround:
            // allocate per-token dst buffers, then memcpy.
            //
            // For simplicity, we end the encoder, run the per-token
            // dispatch into a small scratch, copy out, repeat. Below is
            // the actual mechanics outside this loop.
        }
        enc1.end_encoding();
        cmd1.commit();
        cmd1.wait_until_completed();

        // Re-do per-token reference, this time with a dedicated scratch
        // and per-iteration copy into the consolidated buffer. Cleanest
        // way given the offset-aware kernel writes dst at offset 0.
        let scratch = device.new_buffer(
            (TOP_K * N * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        for t in 0..M {
            let cmd_t = queue.new_command_buffer();
            let enc_t = cmd_t.new_compute_command_encoder();
            crate::q4_k_moe_id_gemv::dispatch_gemv_q4k_moe_id_offset_on_encoder(
                &device,
                enc_t,
                &a_buf,
                (t * K * std::mem::size_of::<f32>()) as u64,
                &w_buf,
                0,
                &ids_buf,
                (t * TOP_K * std::mem::size_of::<i32>()) as u64,
                &scratch,
                N,
                K,
                TOP_K,
                0,
            );
            enc_t.end_encoding();
            // Copy scratch → dst_per_token at the right offset.
            let blit = cmd_t.new_blit_command_encoder();
            blit.copy_from_buffer(
                &scratch,
                0,
                &dst_per_token,
                (t * TOP_K * N * 4) as u64,
                (TOP_K * N * 4) as u64,
            );
            blit.end_encoding();
            cmd_t.commit();
            cmd_t.wait_until_completed();
        }

        // Batched path: one dispatch covering all M*TOP_K pairs.
        let cmd2 = queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        dispatch_gemv_q4k_moe_id_batched_on_encoder(
            &device,
            enc2,
            &a_buf,
            &w_buf,
            0,
            &ids_buf,
            &dst_batched,
            N,
            K,
            M,
            TOP_K,
            K, // outer stride: K floats per token
            0, // inner stride: 0 (gate/up broadcast within token)
        );
        enc2.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        let len = M * TOP_K * N;
        let per_token: &[f32] =
            unsafe { std::slice::from_raw_parts(dst_per_token.contents() as *const f32, len) };
        let batched: &[f32] =
            unsafe { std::slice::from_raw_parts(dst_batched.contents() as *const f32, len) };

        let mut max_abs = 0f32;
        let mut mismatches = 0usize;
        for (i, (&a, &b)) in per_token.iter().zip(batched.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            let denom = a.abs().max(b.abs()).max(1e-3);
            let rel = diff / denom;
            if rel > 1e-5 && diff > 1e-5 {
                mismatches += 1;
                if mismatches < 5 {
                    eprintln!("[{i}] per_token={a} batched={b} diff={diff} rel={rel}");
                }
            }
        }
        eprintln!("max_abs={max_abs:.6} mismatches={mismatches}/{len}");
        assert!(
            mismatches == 0,
            "batched output diverges from per-token — max_abs={max_abs:.6} \
             ({mismatches}/{len} mismatches)"
        );
    }

    /// Same as the gate/up case but exercise the per-pair row layout
    /// (`src1_inner_stride = K`, `src1_outer_stride = top_k * K`) used
    /// by the down projection.
    #[test]
    fn batched_matches_per_token_q4k_moe_gemv_down() {
        const NUM_EXPERTS: usize = 4;
        const N: usize = 64;
        const K: usize = 256;
        const M: usize = 3;
        const TOP_K: usize = 2;

        let cpu = CDevice::Cpu;
        let mut weights_bytes = Vec::new();
        for e in 0..NUM_EXPERTS {
            let raw: Vec<f32> = (0..N * K)
                .map(|i| ((((i + e * 251) % 199) as f32) * 0.011).cos() * 0.3 - 0.1)
                .collect();
            let t = Tensor::from_vec(raw, (N, K), &cpu).unwrap();
            let qt = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
            weights_bytes.extend_from_slice(&qt.data().unwrap());
        }

        // Down activation: [M, TOP_K, K] — each pair has its own row.
        let act: Vec<f32> = (0..M * TOP_K * K)
            .map(|i| ((i as f32) * 0.0017).sin() * 0.5)
            .collect();
        let ids: Vec<i32> = vec![1, 3, 0, 2, 3, 1];

        let Some(device) = metal::Device::system_default() else {
            eprintln!("no Metal device — skipping");
            return;
        };
        let queue = device.new_command_queue();

        let a_buf = device.new_buffer_with_data(
            act.as_ptr() as *const _,
            (act.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ids_buf = device.new_buffer_with_data(
            ids.as_ptr() as *const _,
            (ids.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w_buf = device.new_buffer_with_data(
            weights_bytes.as_ptr() as *const _,
            weights_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let out_size = (M * TOP_K * N * 4) as u64;
        let dst_per_token = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);
        let dst_batched = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);
        let scratch = device.new_buffer(
            (TOP_K * N * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        for t in 0..M {
            let cmd_t = queue.new_command_buffer();
            let enc_t = cmd_t.new_compute_command_encoder();
            // Per-token: src1 row at offset t*top_k*K, src1_stride=K
            // (per-slot row inside the token — matches down projection's
            // [top_k, K] block per token).
            crate::q4_k_moe_id_gemv::dispatch_gemv_q4k_moe_id_offset_on_encoder(
                &device,
                enc_t,
                &a_buf,
                (t * TOP_K * K * std::mem::size_of::<f32>()) as u64,
                &w_buf,
                0,
                &ids_buf,
                (t * TOP_K * std::mem::size_of::<i32>()) as u64,
                &scratch,
                N,
                K,
                TOP_K,
                K, // per-slot stride within token
            );
            enc_t.end_encoding();
            let blit = cmd_t.new_blit_command_encoder();
            blit.copy_from_buffer(
                &scratch,
                0,
                &dst_per_token,
                (t * TOP_K * N * 4) as u64,
                (TOP_K * N * 4) as u64,
            );
            blit.end_encoding();
            cmd_t.commit();
            cmd_t.wait_until_completed();
        }

        let cmd2 = queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        dispatch_gemv_q4k_moe_id_batched_on_encoder(
            &device,
            enc2,
            &a_buf,
            &w_buf,
            0,
            &ids_buf,
            &dst_batched,
            N,
            K,
            M,
            TOP_K,
            TOP_K * K, // outer stride: top_k*K floats per token
            K,         // inner stride: K floats per slot
        );
        enc2.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        let len = M * TOP_K * N;
        let per_token: &[f32] =
            unsafe { std::slice::from_raw_parts(dst_per_token.contents() as *const f32, len) };
        let batched: &[f32] =
            unsafe { std::slice::from_raw_parts(dst_batched.contents() as *const f32, len) };

        let mut max_abs = 0f32;
        let mut mismatches = 0usize;
        for (i, (&a, &b)) in per_token.iter().zip(batched.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            let denom = a.abs().max(b.abs()).max(1e-3);
            let rel = diff / denom;
            if rel > 1e-5 && diff > 1e-5 {
                mismatches += 1;
                if mismatches < 5 {
                    eprintln!("[{i}] per_token={a} batched={b} diff={diff} rel={rel}");
                }
            }
        }
        eprintln!("max_abs={max_abs:.6} mismatches={mismatches}/{len}");
        assert!(
            mismatches == 0,
            "batched(down) diverges from per-token — max_abs={max_abs:.6}"
        );
    }
}
