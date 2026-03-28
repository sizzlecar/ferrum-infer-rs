//! INT4 quantization support: dequantization kernel + linear dispatch.
//!
//! Phase 1: dequant INT4→FP16 + cuBLAS GEMM (simple, correct)
//! Phase 2: Marlin INT4xFP16 fused kernel (TODO, 3.9x faster)

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use candle_core::cuda_backend::CudaDevice;

use crate::ptx;
use crate::weight_store::{GpuQuantWeight, LinearWeight};

/// Dequantize INT4 packed weights to FP16.
///
/// `qw`: quantized weight (GPTQ format)
/// `output`: pre-allocated FP16 buffer [K, N]
pub fn dequant_int4(
    device: &CudaDevice,
    qw: &GpuQuantWeight,
    output: &mut CudaSlice<half::f16>,
) -> candle_core::Result<()> {
    let k = qw.k as i32;
    let n = qw.n as i32;
    let gs = qw.group_size as i32;

    if qw.symmetric {
        let func = device.get_or_load_custom_func(
            "dequant_int4_sym_to_fp16",
            "dequant_int4",
            ptx::DEQUANT_INT4,
        )?;
        let qw_v = qw.qweight.slice(..);
        let sc_v = qw.scales.slice(..);
        let mut b = func.builder();
        b.arg(&qw_v);
        b.arg(&sc_v);
        b.arg(output);
        b.arg(&k);
        b.arg(&n);
        b.arg(&gs);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (((qw.n + 255) / 256) as u32, (qw.k / 8) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("dequant_int4_sym: {e}")))?;
    } else {
        let func = device.get_or_load_custom_func(
            "dequant_int4_to_fp16",
            "dequant_int4",
            ptx::DEQUANT_INT4,
        )?;
        let qw_v = qw.qweight.slice(..);
        let sc_v = qw.scales.slice(..);
        let qz_v = qw
            .qzeros
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("non-symmetric quant requires qzeros".into()))?
            .slice(..);
        let mut b = func.builder();
        b.arg(&qw_v);
        b.arg(&sc_v);
        b.arg(&qz_v);
        b.arg(output);
        b.arg(&k);
        b.arg(&n);
        b.arg(&gs);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (((qw.n + 255) / 256) as u32, (qw.k / 8) as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|e| candle_core::Error::Msg(format!("dequant_int4: {e}")))?;
    }

    Ok(())
}

/// Dispatch linear projection: FP16 cuBLAS or INT4 dequant+cuBLAS.
///
/// For INT4 weights, dequantizes to a temp FP16 buffer first, then runs cuBLAS GEMM.
/// `temp_fp16` must be pre-allocated with size >= K * N (for the largest quantized weight).
pub fn linear_dispatch(
    blas: &CudaBlas,
    device: &CudaDevice,
    input: &CudaSlice<half::f16>,
    weight: &LinearWeight,
    output: &mut CudaSlice<half::f16>,
    temp_fp16: &mut CudaSlice<half::f16>,
    m: i32,
    n: i32,
    k: i32,
) -> candle_core::Result<()> {
    match weight {
        LinearWeight::Fp16(w) => crate::cublas::linear_f16(blas, input, &w.slice, output, m, n, k),
        LinearWeight::Int4(qw) => {
            // Step 1: dequant INT4 → temp FP16 [K, N]
            dequant_int4(device, qw, temp_fp16)?;
            // Step 2: cuBLAS GEMM with dequantized weight
            crate::cublas::linear_f16(blas, input, temp_fp16, output, m, n, k)
        }
    }
}
