//! Triton-rs GPTQ INT4xFP16 fused GEMM — alternative to Marlin.
//!
//! Loads PTX produced by triton-rs's `compile_mlir` from `triton_ptx/`
//! (see `crates/triton-dsl/examples/w4a16_gptq_gemm_typed.rs` for the
//! MLIR source) and launches via cudarc with the Triton 3.6 ABI:
//!
//!   user args:  (a, qweight, scales, qzeros, c, M, N, K, group_size,
//!                stride_am, stride_ak, stride_qwk, stride_qwn,
//!                stride_sk,  stride_sn,
//!                stride_qzk, stride_qzn,
//!                stride_cm,  stride_cn)
//!   implicit:   (global_scratch, profile_scratch)
//!
//! Operates on the **on-disk GPTQ tensor layout**:
//!   - qweight: `[K/8, N]` int32 (8 nibbles per K, packed along K)
//!   - scales:  `[K/G, N]` f16
//!   - qzeros:  `[K/G, N/8]` int32 (packed along N)
//!
//! No host-side repack — Marlin's path repacks GPTQ → Marlin tile format
//! at load (`crate::marlin::repack_gptq_to_marlin`); this kernel avoids
//! that step and runs directly against the GPTQ layout.
//!
//! Block tile: `<BM=64, BN=64, BK=32>` (matches the dump default in the
//! triton-dsl example). Grid: `(cdiv(M, 64), cdiv(N, 64), 1)`.
//!
//! Only compiled with `--features cuda,triton-kernels`. The dispatch
//! decision lives in `backend::cuda::CudaBackend::load_gptq` (which
//! reads `FERRUM_TRITON_INT4=1` once at weight load) and
//! `gemm_gptq` (which routes by `GptqStoreCuda` variant).

#![cfg(all(feature = "cuda", feature = "triton-kernels"))]

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::triton_meta::parse_meta;
use crate::triton_ptx::w4a16_gptq_f16;

/// Block sizes baked into the precompiled PTX. Must match the `mlir()` dump
/// defaults in `crates/triton-dsl/examples/w4a16_gptq_gemm_typed.rs` —
/// changing them here without regenerating the PTX is a silent miscompile.
pub const BM: usize = 64;
pub const BN: usize = 64;
pub const BK: usize = 32;

/// Precompiled PTX text — re-exported from the embedded `triton_ptx`
/// module so external callers (tests, custom dispatch) don't have to
/// reach into a `pub(crate)` symbol.
pub const W4A16_PTX: &str = w4a16_gptq_f16::PTX;

/// GPTQ weights on GPU in the on-disk layout (no Marlin repack).
///
/// All three tensors live on the same stream as the rest of the runtime
/// so the Triton kernel launches cleanly without cross-stream sync.
pub struct TritonGptqWeight {
    /// Packed INT4 weights `[K/8, N]` int32 (8 nibbles per K).
    pub qweight: CudaSlice<i32>,
    /// Per-group fp16 scales `[K/G, N]`.
    pub scales: CudaSlice<half::f16>,
    /// Per-group packed zeros `[K/G, N/8]` int32 (packed along N).
    pub qzeros: CudaSlice<i32>,
    pub k: usize,
    pub n: usize,
    pub group_size: i32,
}

/// Parsed launch params (warps + shared mem) — derived from JSON metadata
/// shipped alongside the PTX. Read once and stashed in a OnceLock so the
/// hot path doesn't re-parse JSON or walk the metadata text every step.
struct LaunchParams {
    num_warps: u32,
    shared_mem: u32,
    fn_name: &'static str,
}

fn launch_params() -> &'static LaunchParams {
    static CACHE: std::sync::OnceLock<LaunchParams> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        let meta = parse_meta(w4a16_gptq_f16::META)
            .unwrap_or_else(|e| panic!("triton w4a16 meta parse: {e}"));
        // The function name comes from the kernel JSON ("name" field).
        // Leak it once so it can live as a 'static str without per-call allocs.
        let fn_name: &'static str = Box::leak(meta.name.into_boxed_str());
        LaunchParams {
            num_warps: meta.num_warps,
            shared_mem: meta.shared_mem as u32,
            fn_name,
        }
    })
}

/// Fused INT4-weight × FP16-activation GEMM via the Triton-rs PTX kernel.
///
/// Computes `C[m, n] = A[m, k] @ dequant(B[k, n])` with
/// `B = (qweight4 - qzero4) * scale`, the standard GPTQ formula.
///
/// `m` = number of rows in A and C. Pass `1` for batch-1 decode.
///
/// `func` is a pre-loaded `CudaFunction` for `w4a16_gptq_gemm_typed`,
/// usually produced by the caller through `CudaState::func()` so the PTX
/// module load happens once and is cached for the life of the process.
pub fn launch_w4a16_gptq_triton(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    input: &CudaSlice<half::f16>,
    weight: &TritonGptqWeight,
    output: &mut CudaSlice<half::f16>,
    m: i32,
) -> candle_core::Result<()> {
    let k = weight.k as i32;
    let n = weight.n as i32;
    let gs = weight.group_size;

    let lp = launch_params();

    // Strides for row-major contiguous tensors (matches how the Marlin /
    // GPTQ load path stores them on the device).
    //   A:       [M, K]      → stride_am=K, stride_ak=1
    //   qweight: [K/8, N]    → stride_qwk=N, stride_qwn=1
    //   scales:  [K/G, N]    → stride_sk=N, stride_sn=1
    //   qzeros:  [K/G, N/8]  → stride_qzk=N/8, stride_qzn=1
    //   C:       [M, N]      → stride_cm=N, stride_cn=1
    let stride_am = k;
    let stride_ak = 1i32;
    let stride_qwk = n;
    let stride_qwn = 1i32;
    let stride_sk = n;
    let stride_sn = 1i32;
    let stride_qzk = n / 8;
    let stride_qzn = 1i32;
    let stride_cm = n;
    let stride_cn = 1i32;

    // Implicit Triton 3.6 scratch buffers. Both are zero-sized for this
    // kernel (no global atomic state, no profiler), so we bump to 1B.
    // Allocated on the runtime stream — same as the rest of the args, so
    // ordering is natural (no cross-stream sync).
    let global_scratch: CudaSlice<u8> = stream
        .alloc_zeros::<u8>(1)
        .map_err(|e| candle_core::Error::Msg(format!("triton w4a16 scratch: {e}")))?;
    let profile_scratch: CudaSlice<u8> = stream
        .alloc_zeros::<u8>(1)
        .map_err(|e| candle_core::Error::Msg(format!("triton w4a16 profile: {e}")))?;

    let qw = weight.qweight.slice(..);
    let sc = weight.scales.slice(..);
    let qz = weight.qzeros.slice(..);
    let inp = input.slice(..);

    let mut b = stream.launch_builder(func);
    b.arg(&inp);
    b.arg(&qw);
    b.arg(&sc);
    b.arg(&qz);
    b.arg(output);
    b.arg(&m);
    b.arg(&n);
    b.arg(&k);
    b.arg(&gs);
    b.arg(&stride_am);
    b.arg(&stride_ak);
    b.arg(&stride_qwk);
    b.arg(&stride_qwn);
    b.arg(&stride_sk);
    b.arg(&stride_sn);
    b.arg(&stride_qzk);
    b.arg(&stride_qzn);
    b.arg(&stride_cm);
    b.arg(&stride_cn);
    b.arg(&global_scratch);
    b.arg(&profile_scratch);

    // Grid: ceil(M / BM) × ceil(N / BN). Block: num_warps * 32 threads.
    let grid_m = ((m as usize + BM - 1) / BM) as u32;
    let grid_n = ((n as usize + BN - 1) / BN) as u32;
    let block_size = lp.num_warps * 32;

    let cfg = LaunchConfig {
        grid_dim: (grid_m, grid_n, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: lp.shared_mem,
    };

    unsafe { b.launch(cfg) }.map(|_| ()).map_err(|e| {
        candle_core::Error::Msg(format!(
            "triton w4a16 launch: {e} (m={m}, n={n}, k={k}, gs={gs})"
        ))
    })
}

/// Helper: kernel function name (matches the `tt.func` name in the MLIR).
/// Used by the dispatcher in `backend::cuda::CudaBackend::gemm_gptq` to
/// look up the cached `CudaFunction` from the `CudaState` module map.
pub fn fn_name() -> &'static str {
    launch_params().fn_name
}
