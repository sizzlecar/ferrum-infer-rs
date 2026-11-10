//! Enqueue on the caller's lane and retain every scratch/static owner.
use super::*;
use cudarc::cublas::{result::gemm_ex, sys, CudaBlas};
use cudarc::driver::CudaStream;
use std::ffi::c_void;

fn flat(elements: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (elements.div_ceil(256).min(65535) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn groups(count: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count.min(65535) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(super) fn enqueue(
    stream: &CudaStream,
    blas: &CudaBlas,
    kernels: &RetainedFunctions,
    regions: &[CudaBufferRegion],
    shape: Shape,
    storage: Storage,
    prepared: Option<&fused_native::Prepared>,
) -> Result<(), CudaDeviceRuntimeError> {
    let m = shape.rows;
    let k = shape.inputs;
    let n = shape.outputs;
    let scratch = regions[storage.inputs() + 1].device_ptr();
    let rotated = scratch + shape.rotated_activation;
    let quantized = scratch + shape.quantized_activation;
    let row_scales = scratch + shape.row_scales;
    let weight = shape
        .decoded_weight
        .map_or(regions[1].device_ptr(), |offset| scratch + offset);
    let x = regions[0].device_ptr();
    let channel = regions[if storage.codebook() { 3 } else { 2 }].device_ptr();
    let y = regions[storage.inputs()].device_ptr();
    let group_count = m * k / 256;
    let output_count = m * n;
    let mut rotate = stream.launch_builder(kernels.comfy_convrot_function(if storage.f32() {
        5
    } else if storage.swiglu() {
        4
    } else {
        0
    }));
    rotate.arg(&x).arg(&rotated).arg(&group_count);
    if storage.swiglu() {
        rotate.arg(&k);
    }
    unsafe { rotate.launch(groups(group_count)) }
        .map_err(|error| CudaDeviceRuntimeError::driver("ConvRot regular H256", error))?;
    let mut quantize =
        stream.launch_builder(kernels.comfy_convrot_function(if storage.f32() { 6 } else { 1 }));
    quantize
        .arg(&rotated)
        .arg(&quantized)
        .arg(&row_scales)
        .arg(&m)
        .arg(&k);
    unsafe { quantize.launch(groups(m)) }.map_err(|error| {
        CudaDeviceRuntimeError::driver("ConvRot activation INT8 quantization", error)
    })?;
    if storage.codebook() {
        let packed = regions[1].device_ptr();
        let relative = regions[2].device_ptr();
        let codebook = regions[4].device_ptr();
        let weight_count = n * k;
        let mut decode = stream.launch_builder(kernels.comfy_convrot_function(2));
        decode
            .arg(&packed)
            .arg(&relative)
            .arg(&codebook)
            .arg(&weight)
            .arg(&weight_count);
        unsafe { decode.launch(flat(weight_count)) }.map_err(|error| {
            CudaDeviceRuntimeError::driver("ConvRot W4 codebook INT8 decode", error)
        })?;
    }
    if let Some(prepared) = prepared.filter(|p| p.fused()) {
        // Every input/output/scratch span and the native owner are retained by
        // the enclosing command. Failure after enqueue never retries another path.
        return unsafe {
            prepared.run(
                quantized,
                weight,
                row_scales,
                channel,
                y,
                stream.cu_stream() as _,
            )
        };
    }
    let accumulator = scratch + shape.accumulator;
    let handle = *blas.handle();
    let one = 1_i32;
    let zero = 0_i32;
    unsafe {
        sys::cublasSetStream_v2(handle, stream.cu_stream() as _)
            .result()
            .map_err(|error| CudaDeviceRuntimeError::blas("ConvRot INT8 GEMM stream", error))?;
        sys::cublasSetWorkspace_v2(handle, std::ptr::null_mut(), 0)
            .result()
            .map_err(|error| {
                CudaDeviceRuntimeError::blas("ConvRot INT8 GEMM zero workspace", error)
            })?;
        sys::cublasSetPointerMode_v2(handle, sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST)
            .result()
            .map_err(|error| {
                CudaDeviceRuntimeError::blas("ConvRot INT8 GEMM host scalars", error)
            })?;
        sys::cublasSetMathMode(handle, sys::cublasMath_t::CUBLAS_DEFAULT_MATH)
            .result()
            .map_err(|error| CudaDeviceRuntimeError::blas("ConvRot INT8 GEMM math mode", error))?;
        sys::cublasSetAtomicsMode(handle, sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED)
            .result()
            .map_err(|error| CudaDeviceRuntimeError::blas("ConvRot INT8 GEMM atomics", error))?;
        gemm_ex(
            handle,
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            &one as *const i32 as *const c_void,
            weight as *const c_void,
            sys::cudaDataType_t::CUDA_R_8I,
            k as i32,
            quantized as *const c_void,
            sys::cudaDataType_t::CUDA_R_8I,
            k as i32,
            &zero as *const i32 as *const c_void,
            accumulator as *mut c_void,
            sys::cudaDataType_t::CUDA_R_32I,
            n as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .map_err(|error| CudaDeviceRuntimeError::blas("ConvRot INT8 x INT8 to INT32 GEMM", error))?;
    let mut epilogue =
        stream.launch_builder(kernels.comfy_convrot_function(if storage.f32() { 7 } else { 3 }));
    epilogue
        .arg(&accumulator)
        .arg(&row_scales)
        .arg(&channel)
        .arg(&y)
        .arg(&output_count)
        .arg(&n);
    unsafe { epilogue.launch(flat(output_count)) }
        .map_err(|error| CudaDeviceRuntimeError::driver("ConvRot INT32 epilogue", error))?;
    Ok(())
}
