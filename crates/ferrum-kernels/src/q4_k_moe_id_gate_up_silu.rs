//! Q4_K_M MoE fused gate+up gemv with in-register SiLU·gate — Rust glue
//! for `q4_k_moe_id_gate_up_silu.metal`.
//!
//! See the .metal file for the algorithmic notes and bandwidth ledger.
//! This dispatch replaces three back-to-back dispatches in the decode
//! m=1 stacked MoE FFN path:
//!
//!   1. `gemv_q4kw_moe_id_f32` (gate) → gate_out_stacked
//!   2. `gemv_q4kw_moe_id_f32` (up)   → up_out_stacked
//!   3. `silu_mul_stacked_f32`        → silu_stacked
//!
//! → one dispatch that writes `silu_stacked` directly. `gate_w` and
//! `up_w` must share `n` (out_features), `k` (in_features), and the
//! `nb01/nb02` block strides (true for Qwen3-MoE GGUFs — gate and up
//! always have matching shapes).

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("q4_k_moe_id_gate_up_silu.metal");
const KERNEL_NAME: &str = "gemv_q4kw_moe_id_gate_up_silu_f32";

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile q4_k_moe_id_gate_up_silu.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find gemv_q4kw_moe_id_gate_up_silu_f32 function");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build gemv_q4kw_moe_id_gate_up_silu_f32 pipeline")
    })
}

/// Dispatch the fused gate+up+silu MoE GEMV on an existing compute encoder.
///
/// `gate_w_stacked` / `up_w_stacked`: `[num_experts, n, k/256]` Q4_K
///   block buffers, both with stride `nb02 = n * (k/256) * 144` bytes
///   per expert (must match — this kernel asserts shape-equality
///   structurally by using the same `n / k` for both).
/// `a`               : `[k]` activations (single token, broadcast across slots).
/// `ids`             : `[n_selected]` selected expert IDs (i32).
/// `silu_out`        : `[n_selected, n]` output rows, equal to
///                     `silu(gate · a) * (up · a)` per slot.
/// `n`               : per-expert out_features (must be divisible by 4).
/// `k`               : in_features (multiple of 256).
/// `n_selected`      : number of selected experts (= top_k for decode m=1).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gemv_q4k_moe_id_gate_up_silu_on_encoder(
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
    n_selected: usize,
) {
    debug_assert!(k % 256 == 0, "K must be a multiple of 256 (got {k})");
    debug_assert!(n % 4 == 0, "N must be a multiple of 4 (got {n})");

    let nb01_bytes = (k / 256) * 144;
    let nb02_bytes = n * nb01_bytes;

    #[repr(C)]
    struct P {
        n: i32,
        k: i32,
        nb01: i32,
        nb02: i32,
        n_selected: i32,
    }
    let params = P {
        n: n as i32,
        k: k as i32,
        nb01: nb01_bytes as i32,
        nb02: nb02_bytes as i32,
        n_selected: n_selected as i32,
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
    let grid = MTLSize::new((n as u64).div_ceil(TILE_ROWS), 1, n_selected as u64);
    let tg = MTLSize::new(32, 2, 1);
    enc.dispatch_thread_groups(grid, tg);
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device as CDevice, Tensor};
    use metal::MTLResourceOptions;

    /// Fused gate+up+silu must produce the same outputs as the legacy
    /// 3-dispatch sequence (gate gemv → up gemv → silu_mul_stacked) on
    /// the same Q4_K weights. The inner Q4_K decode is byte-for-byte
    /// identical, the simd_sum reduce is identical, and silu·mul is the
    /// same arithmetic — so the two paths should match within fp32
    /// reordering noise (typically bitwise on M1).
    #[test]
    fn fused_matches_unfused_q4k_moe_gate_up_silu() {
        const NUM_EXPERTS: usize = 4;
        const N: usize = 64; // expert_inter
        const K: usize = 256; // hidden
        const TOP_K: usize = 2;

        let cpu = CDevice::Cpu;

        // Pack a [num_experts, n, k] stack: per-expert quantize to Q4_K,
        // concat into one byte slab matching the layout `load_q4k_experts`
        // expects (`expected = num_experts * n * (k/256) * 144`).
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

        // Sanity: layout matches `load_q4k_experts` expectation.
        const QK_K: usize = 256;
        const BLOCK_BYTES: usize = 144;
        let expected = NUM_EXPERTS * N * (K / QK_K) * BLOCK_BYTES;
        assert_eq!(gate_bytes.len(), expected);
        assert_eq!(up_bytes.len(), expected);

        let act: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.0021).cos() * 0.7).collect();
        let ids: Vec<i32> = vec![1, 3];

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

        let out_size = (TOP_K * N * 4) as u64;
        let gate_out = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);
        let up_out = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);
        let silu_unfused = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);
        let silu_fused = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

        // ── Unfused: 3 dispatches ──────────────────────────────────────
        let cmd1 = queue.new_command_buffer();
        let enc1 = cmd1.new_compute_command_encoder();
        crate::q4_k_moe_id_gemv::dispatch_gemv_q4k_moe_id_on_encoder(
            &device, enc1, &a_buf, &gate_buf, 0, &ids_buf, &gate_out, N, K, TOP_K, 0,
        );
        crate::q4_k_moe_id_gemv::dispatch_gemv_q4k_moe_id_on_encoder(
            &device, enc1, &a_buf, &up_buf, 0, &ids_buf, &up_out, N, K, TOP_K, 0,
        );
        crate::moe_post_ops::dispatch_silu_mul_stacked(
            &device,
            enc1,
            &gate_out,
            &up_out,
            &silu_unfused,
            TOP_K,
            N,
        );
        enc1.end_encoding();
        cmd1.commit();
        cmd1.wait_until_completed();

        // ── Fused: 1 dispatch ──────────────────────────────────────────
        let cmd2 = queue.new_command_buffer();
        let enc2 = cmd2.new_compute_command_encoder();
        dispatch_gemv_q4k_moe_id_gate_up_silu_on_encoder(
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
            TOP_K,
        );
        enc2.end_encoding();
        cmd2.commit();
        cmd2.wait_until_completed();

        let len = TOP_K * N;
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
            // Same Q4_K decode + same simd_sum reduce + same silu·mul ops
            // — only the *order* differs between fused and unfused (the
            // unfused path round-trips through fp32 memory). Bitwise
            // identity is expected on M1; a tiny tolerance protects
            // against future compiler reordering.
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
            "fused output diverges from unfused — max_abs={max_abs:.6} \
             ({mismatches}/{len} mismatches)"
        );
    }
}
