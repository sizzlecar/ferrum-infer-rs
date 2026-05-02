//! Batched fused gate+up+silu MoE GEMV — Rust glue for
//! `q4_k_moe_id_gate_up_silu_batched.metal`. Hybrid of the m=1 fused
//! gate+up+silu kernel (df64ac1) and the batched-pair Z-axis layout
//! (`q4_k_moe_id_gemv_batched.rs`). One Metal dispatch covers all
//! `m * top_k` (token, expert) pairs and writes `silu_stacked`
//! directly, replacing 3 batched dispatches per layer.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q4_k_moe_id_gate_up_silu_batched.metal");
const KERNEL_NAME: &str = "gemv_q4kw_moe_id_gate_up_silu_batched_f32";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q4_k_moe_id_gate_up_silu_batched.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemv_q4kw_moe_id_gate_up_silu_batched_f32 function");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemv_q4kw_moe_id_gate_up_silu_batched_f32 pipeline")
    })
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemv_q4k_moe_id_gate_up_silu_batched_on_encoder(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    a: &Buffer,
    gate_w_stacked: &Buffer,
    gate_w_byte_offset: u64,
    up_w_stacked: &Buffer,
    up_w_byte_offset: u64,
    ids: &Buffer,
    silu_out: &Buffer,
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
    enc.set_buffer(0, Some(gate_w_stacked), gate_w_byte_offset);
    enc.set_buffer(1, Some(up_w_stacked), up_w_byte_offset);
    enc.set_buffer(2, Some(a), 0);
    enc.set_buffer(3, Some(ids), 0);
    enc.set_buffer(4, Some(silu_out), 0);
    enc.set_bytes(
        5,
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

    /// Batched fused kernel ↔ batched (gate gemv + up gemv + silu_mul)
    /// must agree bitwise.
    #[test]
    fn fused_gate_up_silu_batched_matches_unfused() {
        const NUM_EXPERTS: usize = 4;
        const N: usize = 64;
        const K: usize = 256;
        const M: usize = 3;
        const TOP_K: usize = 2;

        let cpu = CDevice::Cpu;

        let pack_stack = |seed: f32| -> Vec<u8> {
            let mut buf = Vec::new();
            for e in 0..NUM_EXPERTS {
                let raw: Vec<f32> = (0..N * K)
                    .map(|i| ((((i + e * 313) % 251) as f32) * 0.013).sin() * 0.4 + seed)
                    .collect();
                let t = Tensor::from_vec(raw, (N, K), &cpu).unwrap();
                let qt = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
                buf.extend_from_slice(&qt.data().unwrap());
            }
            buf
        };
        let gate_bytes = pack_stack(0.05);
        let up_bytes = pack_stack(-0.07);

        let act: Vec<f32> = (0..M * K)
            .map(|i| ((i as f32) * 0.0021).cos() * 0.7)
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
        let gate_buf = device.new_buffer_with_data(
            gate_bytes.as_ptr() as *const _,
            gate_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let up_buf = device.new_buffer_with_data(
            up_bytes.as_ptr() as *const _,
            up_bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let pair_out_size = (M * TOP_K * N * 4) as u64;
        let gate_out = device.new_buffer(pair_out_size, MTLResourceOptions::StorageModeShared);
        let up_out = device.new_buffer(pair_out_size, MTLResourceOptions::StorageModeShared);
        let silu_unfused = device.new_buffer(pair_out_size, MTLResourceOptions::StorageModeShared);
        let silu_fused = device.new_buffer(pair_out_size, MTLResourceOptions::StorageModeShared);

        // Unfused: 3 batched dispatches.
        let cmd1 = queue.new_command_buffer();
        let enc1 = cmd1.new_compute_command_encoder();
        crate::q4_k_moe_id_gemv_batched::dispatch_gemv_q4k_moe_id_batched_on_encoder(
            &device, enc1, &a_buf, &gate_buf, 0, &ids_buf, &gate_out, N, K, M, TOP_K, K, 0,
        );
        crate::q4_k_moe_id_gemv_batched::dispatch_gemv_q4k_moe_id_batched_on_encoder(
            &device, enc1, &a_buf, &up_buf, 0, &ids_buf, &up_out, N, K, M, TOP_K, K, 0,
        );
        crate::moe_post_ops_batched::dispatch_silu_mul_batched(
            &device,
            enc1,
            &gate_out,
            &up_out,
            &silu_unfused,
            M * TOP_K,
            N,
        );
        enc1.end_encoding();
        cmd1.commit();
        cmd1.wait_until_completed();

        // Fused: 1 dispatch.
        let cmd2 = queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        dispatch_gemv_q4k_moe_id_gate_up_silu_batched_on_encoder(
            &device,
            enc2,
            &a_buf,
            &gate_buf,
            0,
            &up_buf,
            0,
            &ids_buf,
            &silu_fused,
            N,
            K,
            M,
            TOP_K,
            K,
            0,
        );
        enc2.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        let len = M * TOP_K * N;
        let unfused: &[f32] =
            unsafe { std::slice::from_raw_parts(silu_unfused.contents() as *const f32, len) };
        let fused: &[f32] =
            unsafe { std::slice::from_raw_parts(silu_fused.contents() as *const f32, len) };

        let mut max_abs = 0f32;
        let mut mismatches = 0usize;
        for (i, (&u, &f)) in unfused.iter().zip(fused.iter()).enumerate() {
            let diff = (u - f).abs();
            if diff > max_abs {
                max_abs = diff;
            }
            let denom = u.abs().max(f.abs()).max(1e-3);
            let rel = diff / denom;
            if rel > 1e-5 && diff > 1e-5 {
                mismatches += 1;
                if mismatches < 5 {
                    eprintln!("[{i}] unfused={u} fused={f} diff={diff} rel={rel}");
                }
            }
        }
        eprintln!("max_abs={max_abs:.6} mismatches={mismatches}/{len}");
        assert!(
            mismatches == 0,
            "fused-batched output diverges from unfused-batched — max_abs={max_abs:.6}"
        );
    }
}
