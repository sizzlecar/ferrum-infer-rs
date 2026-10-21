//! CUDA-only assembly; family, compilation, admission, execution and oracle are shared.

use super::super::arguments::CudaOptions;
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_dense_bf16_cublas_version, cuda_tensor_runtime_config, cuda_vnext_dense_bf16_provider,
        CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
};

pub(super) fn create(
    config: &QwenImage21Config,
    directory: &Path,
    shape: ProjectionShape,
    options: &CudaOptions,
) -> Result<ManagedProjection<CudaDeviceRuntime>> {
    if QwenImage21ImageProjectionFamily::CUDA_PROFILE_ID != CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID {
        bail!("image projection family and CUDA provider numerical profiles differ");
    }
    let runtime = Arc::new(CudaDeviceRuntime::new_tensor_eager(
        CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(
                options.device,
                id(format!("device.cuda.{}", options.device))?,
            )?,
            usable_memory_bytes: options.usable_memory_bytes,
            bootstrap_memory_bytes: options.bootstrap_memory_bytes,
            stream_memory_bytes: options.stream_memory_bytes,
            maximum_streams: options.maximum_streams,
            maximum_in_flight_per_stream: options.maximum_in_flight_per_stream,
            opaque_memory_profile: options.opaque_memory_profile.clone(),
        },
    )?);
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(dense_linear_bf16_contract()?)],
        vec![cuda_vnext_dense_bf16_provider(&runtime)?],
    )?;
    let materializers = WeightMaterializerRegistry::identity_only()?;
    let descriptor = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.projection")?,
        ContractVersion::new(1, 0),
        descriptor.runtime_implementation_fingerprint.clone(),
        descriptor.id.clone(),
        descriptor.capabilities.clone(),
    )?;
    let catalog = materializers
        .augment_catalog(registry.capability_catalog(descriptor.clone(), vec![engine])?)?;
    let settings = ExecutionSettings {
        profile: CUDA_DENSE_BF16_NUMERICAL_PROFILE_ID,
        backend: "CUDA",
        capacity_bytes: options.usable_memory_bytes,
        staging_bytes: options.transfer_staging_bytes,
        backend_configuration: json!({
            "declared_allowances": options,
            "cublas_version": cuda_dense_bf16_cublas_version()?,
            "arithmetic": "CUBLAS_COMPUTE_32F; reduced-precision reduction disabled; BF16 operands and output; no user/default GEMM workspace",
            "qualification": "sampled mathematical oracle only; opaque allowances unqualified; no Reference bitwise or Torch/vLLM equality claim"
        }),
        live_allocations: None,
    };
    ManagedProjection::new(
        config,
        directory,
        shape,
        runtime,
        &registry,
        &catalog,
        &materializers,
        settings,
    )
}
