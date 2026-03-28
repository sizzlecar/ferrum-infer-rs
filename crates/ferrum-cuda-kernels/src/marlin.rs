//! Marlin INT4xFP16 fused GEMM kernel (IST Austria).
//!
//! Near-ideal 3.9x speedup over FP16 cuBLAS for INT4 quantized weights.
//! Weights must be in Marlin packed format (different from GPTQ).
//!
//! Constraints: K % 128 == 0, N % 256 == 0, SM >= 8.0 (Ampere+).

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::sync::Arc;

// FFI declaration for the Marlin CUDA kernel.
// Only linked when the "marlin" feature is enabled (requires nvcc + SM >= 8.0).
#[cfg(feature = "marlin")]
extern "C" {
    fn marlin_cuda(
        A: *const std::ffi::c_void,
        B: *const std::ffi::c_void,
        C: *mut std::ffi::c_void,
        s: *const std::ffi::c_void,
        prob_m: i32,
        prob_n: i32,
        prob_k: i32,
        workspace: *mut std::ffi::c_void,
        groupsize: i32,
        dev: i32,
        stream: cudarc::driver::sys::CUstream,
        thread_k: i32,
        thread_n: i32,
        sms: i32,
        max_par: i32,
    ) -> i32;
}

/// Check if Marlin kernel is available at compile time.
pub fn is_available() -> bool {
    cfg!(feature = "marlin")
}

/// Marlin-format quantized weight for one linear layer.
pub struct MarlinWeight {
    /// Repacked INT4 weights in Marlin tile format: varies by K, N
    pub qweight: CudaSlice<i32>,
    /// Per-group FP16 scales (permuted for Marlin access pattern)
    pub scales: CudaSlice<half::f16>,
    /// Workspace for Marlin kernel: [N/128 * max_par] int32, zeroed
    pub workspace: CudaSlice<i32>,
    pub k: usize,
    pub n: usize,
    pub group_size: i32,
}

/// Run Marlin INT4xFP16 fused GEMM.
///
/// Computes: C[m, n] = A[m, k] @ dequant(B[k, n])
/// where B is in Marlin packed INT4 format.
///
/// Only available when compiled with `--features marlin`.
#[cfg(feature = "marlin")]
pub fn marlin_gemm(
    stream: &Arc<CudaStream>,
    input: &CudaSlice<half::f16>,
    weight: &MarlinWeight,
    output: &mut CudaSlice<half::f16>,
    m: i32,
) -> candle_core::Result<()> {
    let n = weight.n as i32;
    let k = weight.k as i32;

    // Get raw device pointers
    let (a_ptr, _a_guard) = input.device_ptr(stream);
    let (b_ptr, _b_guard) = weight.qweight.device_ptr(stream);
    let (c_ptr, _c_guard) = output.device_ptr(stream);
    let (s_ptr, _s_guard) = weight.scales.device_ptr(stream);
    let (ws_ptr, _ws_guard) = weight.workspace.device_ptr(stream);

    let raw_stream = stream.cu_stream();

    let ret = unsafe {
        marlin_cuda(
            a_ptr as *const _,
            b_ptr as *const _,
            c_ptr as *mut _,
            s_ptr as *const _,
            m,
            n,
            k,
            ws_ptr as *mut _,
            weight.group_size,
            0, // dev
            raw_stream,
            -1, // auto thread_k
            -1, // auto thread_n
            -1, // auto sms
            16, // max_par
        )
    };

    if ret != 0 {
        return Err(candle_core::Error::Msg(format!(
            "marlin_cuda failed: ret={ret} (m={m}, n={n}, k={k}, gs={})",
            weight.group_size
        )));
    }
    Ok(())
}

/// Stub when Marlin feature is not enabled.
#[cfg(not(feature = "marlin"))]
pub fn marlin_gemm(
    _stream: &Arc<CudaStream>,
    _input: &CudaSlice<half::f16>,
    _weight: &MarlinWeight,
    _output: &mut CudaSlice<half::f16>,
    _m: i32,
) -> candle_core::Result<()> {
    Err(candle_core::Error::Msg(
        "Marlin kernel not available (compile with --features marlin)".into(),
    ))
}

// ===================== Weight Repacking (GPTQ → Marlin) =====================

/// Repack GPTQ INT4 weights to Marlin format on CPU.
///
/// GPTQ format: qweight [K/8, N] int32, 8 values packed per word, row-major
/// Marlin format: tiled, permuted, and repacked for tensor core access
///
/// This is a one-time CPU operation during model loading.
pub fn repack_gptq_to_marlin(
    qweight_gptq: &[i32], // [K/8, N]
    k: usize,
    n: usize,
) -> Vec<i32> {
    // Step 1: Unpack GPTQ to individual INT4 values [K, N]
    let packed_rows = k / 8;
    let mut unpacked = vec![0u8; k * n];
    for pr in 0..packed_rows {
        for col in 0..n {
            let packed = qweight_gptq[pr * n + col];
            for i in 0..8 {
                unpacked[(pr * 8 + i) * n + col] = ((packed >> (i * 4)) & 0xF) as u8;
            }
        }
    }

    // Step 2: Tile transpose — reshape [K, N] into [K/16, 16, N/16, 16]
    // then permute to [K/16, N/16, 16, 16], flatten to [K/16, N*16]
    let tile_k = k / 16;
    let tile_n = n / 16;
    let mut tiled = vec![0u8; k * n];
    for tk in 0..tile_k {
        for tn in 0..tile_n {
            for ik in 0..16 {
                for in_ in 0..16 {
                    let src_idx = (tk * 16 + ik) * n + (tn * 16 + in_);
                    let dst_idx = (tk * tile_n + tn) * 256 + ik * 16 + in_;
                    tiled[dst_idx] = unpacked[src_idx];
                }
            }
        }
    }

    // Step 3: Apply interleave permutation within each group of 8
    // [0,2,4,6,1,3,5,7] — separates even/odd for half2 pairs
    let interleave = [0usize, 2, 4, 6, 1, 3, 5, 7];
    let total = k * n;
    let groups = total / 8;
    let mut permuted = vec![0u8; total];
    for g in 0..groups {
        for i in 0..8 {
            permuted[g * 8 + i] = tiled[g * 8 + interleave[i]];
        }
    }

    // Step 4: Pack 8 INT4 values into int32 (every 8th element shares a word)
    let packed_len = total / 8;
    let mut result = vec![0i32; packed_len];
    for i in 0..packed_len {
        let mut word = 0u32;
        for j in 0..8 {
            word |= (permuted[i * 8 + j] as u32) << (j * 4);
        }
        result[i] = word as i32;
    }

    result
}

/// Permute scales from GPTQ layout to Marlin access pattern.
pub fn repack_scales_to_marlin(
    scales_gptq: &[half::f16], // [K/group_size, N]
    k: usize,
    n: usize,
    group_size: usize,
) -> Vec<half::f16> {
    let num_groups = k / group_size;
    if num_groups == 1 {
        // Per-channel: simpler permutation
        return repack_scales_perchannel(scales_gptq, n);
    }

    // For grouped (group_size=128): tile and permute scales
    // The Marlin kernel reads scales in a specific order matching its tile layout
    let mut result = vec![half::f16::ZERO; num_groups * n];

    // Scale permutation for group_size=128 (group_blocks=8):
    // Groups are accessed in 8-row tiles, so reorder accordingly
    let scale_perm: Vec<usize> = (0..8)
        .flat_map(|i| (0..8).map(move |j| i + 8 * j))
        .collect();

    for col in 0..n {
        for (dst_g, &src_g) in scale_perm.iter().enumerate().take(num_groups.min(64)) {
            if src_g < num_groups {
                result[dst_g * n + col] = scales_gptq[src_g * n + col];
            }
        }
        // For num_groups > 64, copy remaining in order
        for g in 64..num_groups {
            result[g * n + col] = scales_gptq[g * n + col];
        }
    }

    result
}

fn repack_scales_perchannel(scales: &[half::f16], n: usize) -> Vec<half::f16> {
    // Per-channel scales: [1, N] → apply single permutation
    let perm: Vec<usize> = (0..4)
        .flat_map(|i| [0, 1, 8, 9, 16, 17, 24, 25].map(move |j| 2 * i + j))
        .collect();

    let mut result = vec![half::f16::ZERO; n];
    let groups = n / 32;
    for g in 0..groups {
        for (dst, &src) in perm.iter().enumerate().take(32) {
            if g * 32 + src < n {
                result[g * 32 + dst] = scales[g * 32 + src];
            }
        }
    }
    result
}
