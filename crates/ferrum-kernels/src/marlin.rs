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
    /// Activation gather permutation for desc_act=true (act-order) GPTQ.
    /// `perm[i]` = original column index that should appear at position i
    /// after gather. Computed at load time as `argsort(g_idx_disk)`.
    /// `qweight` rows have already been permuted by this; runtime gathers
    /// input columns by the same perm so the standard Marlin kernel
    /// produces the un-permuted GEMM result. None for desc_act=false.
    pub perm: Option<CudaSlice<i32>>,
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

    let raw_stream = stream.cu_stream();

    // Zero workspace on the runner's stream — Marlin uses it as mutex locks.
    // All operations (memset + kernel) on same stream → naturally ordered.
    {
        let (ws_ptr, _guard) = weight.workspace.device_ptr(stream);
        unsafe {
            cudarc::driver::sys::cuMemsetD32Async(ws_ptr, 0, weight.workspace.len(), raw_stream);
        }
    }

    // Get raw device pointers
    let (a_ptr, _a_guard) = input.device_ptr(stream);
    let (b_ptr, _b_guard) = weight.qweight.device_ptr(stream);
    let (c_ptr, _c_guard) = output.device_ptr(stream);
    let (s_ptr, _s_guard) = weight.scales.device_ptr(stream);
    let (ws_ptr, _ws_guard) = weight.workspace.device_ptr(stream);

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

    // No per-call sync needed — all operations (memset + kernel) are on the
    // runner's stream. decode_step syncs once at the end before returning logits.
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

/// Permute GPTQ INT4 qweight rows by `perm` (one-time, CPU work at load).
/// Input qweight is [K/8, N] packed (8 INT4s per i32 along K). Output is
/// the same shape, with rows reordered: out[i, n] = in[perm[i], n] (after
/// unpacking + repacking). Used to put rows in g_idx-sorted order so the
/// downstream Marlin repack sees a layout where group_idx = i / group_size.
pub fn permute_gptq_qweight_rows(
    qweight_gptq: &[i32], // [K/8, N] packed
    perm: &[usize],       // [K]
    k: usize,
    n: usize,
) -> Vec<i32> {
    debug_assert_eq!(perm.len(), k);
    debug_assert_eq!(qweight_gptq.len(), (k / 8) * n);

    // Step 1: Unpack [K/8, N] → [K, N] (one INT4 per byte slot, low 4 bits).
    let mut kn = vec![0u8; k * n];
    let packed_rows = k / 8;
    for pr in 0..packed_rows {
        for col in 0..n {
            let packed = qweight_gptq[pr * n + col] as u32;
            for i in 0..8 {
                kn[(pr * 8 + i) * n + col] = ((packed >> (i * 4)) & 0xF) as u8;
            }
        }
    }

    // Step 2: Permute rows. out[i, n] = in[perm[i], n].
    let mut sorted = vec![0u8; k * n];
    for i in 0..k {
        let src_row = perm[i];
        for col in 0..n {
            sorted[i * n + col] = kn[src_row * n + col];
        }
    }

    // Step 3: Repack [K, N] → [K/8, N] (8 INT4s per i32 along K).
    let mut packed = vec![0i32; (k / 8) * n];
    for pr in 0..packed_rows {
        for col in 0..n {
            let mut word = 0u32;
            for i in 0..8 {
                word |= (sorted[(pr * 8 + i) * n + col] as u32) << (i * 4);
            }
            packed[pr * n + col] = word as i32;
        }
    }
    packed
}

/// Repack GPTQ INT4 weights to Marlin format on CPU.
///
/// GPTQ format: qweight [K/8, N] int32 (in_features packed, out_features columns)
/// Marlin format: [N/16, K*16/8] int32, tiled and permuted for tensor core access
///
/// Key: Marlin operates on [N, K] layout (out_features first, like PyTorch Linear.weight).
/// GPTQ stores [K, N]. Must transpose before tiling.
///
/// Reference: IST-DASLab/marlin __init__.py Layer.pack()
pub fn repack_gptq_to_marlin(
    qweight_gptq: &[i32], // [K/8, N]
    k: usize,
    n: usize,
) -> Vec<i32> {
    // Step 1: Unpack GPTQ [K/8, N] → individual INT4 values [K, N]
    let packed_rows = k / 8;
    let mut kn = vec![0u8; k * n]; // [K, N] layout
    for pr in 0..packed_rows {
        for col in 0..n {
            let packed = qweight_gptq[pr * n + col];
            for i in 0..8 {
                kn[(pr * 8 + i) * n + col] = ((packed >> (i * 4)) & 0xF) as u8;
            }
        }
    }

    // Step 2: Transpose [K, N] to get w = linear.weight.data.t() = [K, N]
    // (GPTQ stores [K, N] already, so kn IS [K, N] — no transpose needed!)
    // Marlin's pack() does: w = linear.weight.data.t() which gives [K, N].
    // Our kn is already [K, N].

    // Step 3: Tile [K, N] → [K/16, 16, N/16, 16] → permute(0,2,1,3) → [K/16, N*16]
    let tile = 16;
    let kt = k / tile;
    let nt = n / tile;
    let mut tiled = vec![0u8; k * n]; // [K/16, N*16]
    for tk in 0..kt {
        for tn in 0..nt {
            for ik in 0..tile {
                for in_ in 0..tile {
                    let src = (tk * tile + ik) * n + (tn * tile + in_);
                    let dst = tk * (n * tile) + tn * (tile * tile) + ik * tile + in_;
                    tiled[dst] = kn[src];
                }
            }
        }
    }

    // Step 4: Apply _perm in blocks of 1024
    let perm = build_marlin_perm();
    let total = k * n;
    let mut permuted = vec![0u8; total];
    let num_blocks = total / 1024;
    for blk in 0..num_blocks {
        let base = blk * 1024;
        for (dst, &src) in perm.iter().enumerate() {
            permuted[base + dst] = tiled[base + src];
        }
    }

    // Step 4: Pack 8 INT4 values → int32, taking every 8th element
    //         result shape: [N/16, K*16/8] = [N/16, K*2]
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
///
/// GPTQ: [num_groups, N] row-major (groups along K, columns are out_features)
/// Marlin: [num_groups, N] but reshuffled to match the kernel's tile access.
///
/// Reference: IST-DASLab/marlin __init__.py _scale_perm / _scale_perm_single
pub fn repack_scales_to_marlin(
    scales_gptq: &[half::f16], // [num_groups, N]
    k: usize,
    n: usize,
    group_size: usize,
) -> Vec<half::f16> {
    let num_groups = k / group_size;

    // Build permutation table matching Marlin's scale access pattern
    let scale_perm: Vec<usize> = if num_groups > 1 {
        // Grouped quantization (group_size=128, group_blocks=8)
        // _scale_perm = [i + 8*j for i in range(8) for j in range(8)]
        (0..8)
            .flat_map(|i| (0..8).map(move |j| i + 8 * j))
            .collect()
    } else {
        // Per-channel (group_size=-1, group_blocks=-1)
        // _scale_perm_single = [2*i+j for i in range(4) for j in [0,1,8,9,16,17,24,25]]
        (0..4)
            .flat_map(|i| [0, 1, 8, 9, 16, 17, 24, 25].map(move |j| 2 * i + j))
            .collect()
    };

    // Flatten scales, apply permutation in blocks
    let total = num_groups * n;
    let perm_len = scale_perm.len();
    let mut result = vec![half::f16::ZERO; total];

    // Reshape scales as flat array, permute in blocks of perm_len
    for blk in 0..(total / perm_len) {
        let base = blk * perm_len;
        for (dst, &src) in scale_perm.iter().enumerate() {
            result[base + dst] = scales_gptq[base + src];
        }
    }
    // Remainder (if total not divisible by perm_len)
    let rem_start = (total / perm_len) * perm_len;
    for i in rem_start..total {
        result[i] = scales_gptq[i];
    }
    result
}

/// Build the 1024-element Marlin weight permutation array.
///
/// This encodes the m16n8k16 tensor core mma fragment layout.
/// Each 1024-element block of the tiled weight [N/16, K*16] is
/// permuted to match how the Marlin kernel loads data into
/// tensor core fragments via shared memory.
///
/// Reference: IST-DASLab/marlin __init__.py _perm construction
fn build_marlin_perm() -> Vec<usize> {
    let mut perm = Vec::with_capacity(1024);

    for i in 0..32 {
        let col = i / 4;
        let mut perm1 = Vec::with_capacity(8);

        for _block in 0..2 {
            for &row_off in &[0, 1, 8, 9] {
                let row = 2 * (i % 4) + row_off / 8 * 8 + row_off % 8;
                // Actually, the original Python is:
                // for row in [2*(i%4), 2*(i%4)+1, 2*(i%4+4), 2*(i%4+4)+1]:
                //     perm1.append(16*row + col + 8*block)
                let _ = row; // ignore, use direct construction below
            }
        }

        // Direct from Python: for block in [0,1]: for row in [...]: perm1.append(...)
        perm1.clear();
        for block in 0..2 {
            for &row in &[
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ] {
                perm1.push(16 * row + col + 8 * block);
            }
        }

        for j in 0..4 {
            for &p in &perm1 {
                perm.push(p + 256 * j);
            }
        }
    }

    assert_eq!(perm.len(), 1024);

    // KEY: apply interleave [0,2,4,6,1,3,5,7] within each group of 8
    let interleave = [0usize, 2, 4, 6, 1, 3, 5, 7];
    let mut perm_interleaved = vec![0usize; 1024];
    for g in 0..128 {
        for i in 0..8 {
            perm_interleaved[g * 8 + i] = perm[g * 8 + interleave[i]];
        }
    }

    perm_interleaved
}
