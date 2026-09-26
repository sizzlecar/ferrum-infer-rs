//! Real-history diagnostics composed from each production backend factory.
use ferrum_interfaces::vnext::DeviceId;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use ferrum_interfaces::vnext::WeightMaterializerSelection;
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use ferrum_kernels::backend::metal::{
    vnext_ops::MetalVNextComposition, vnext_runtime::MetalDeviceRuntime,
};
use ferrum_models::{vnext::DefinedProductionModel, VNextModelExecutor};
use ferrum_types::{EngineConfig, FerrumError, Result};

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub fn create_metal_vnext_teacher_collector(
    engine: &EngineConfig,
    defined: &DefinedProductionModel,
) -> Result<VNextModelExecutor<MetalDeviceRuntime>> {
    let device_id =
        DeviceId::new("device.metal.0").map_err(|error| FerrumError::device(error.to_string()))?;
    let composition = MetalVNextComposition::create_with_memory_sampling(
        device_id,
        engine.runtime.device_memory_sampling.as_ref(),
    )
    .map_err(|error| {
        FerrumError::device(format!("create production Metal teacher runtime: {error}"))
    })?;
    let (runtime, operations, materializers, materializer_id, catalog) = composition.into_parts();
    let selection = WeightMaterializerSelection::exact(materializer_id);
    crate::product_composition::create_vnext_executor(
        engine,
        defined,
        runtime,
        operations,
        materializers,
        catalog,
        |_| Ok(selection.clone()),
    )
}

#[cfg(feature = "cuda")]
pub fn create_cuda_vnext_teacher_collector(
    engine: &EngineConfig,
    defined: &DefinedProductionModel,
    ordinal: usize,
) -> Result<VNextModelExecutor<ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime>> {
    use ferrum_kernels::backend::cuda::vnext_ops::{
        cuda_weight_materializer_selection, CudaVNextComposition,
    };
    let device_id = DeviceId::new(format!("device.cuda.{ordinal}"))
        .map_err(|error| FerrumError::device(error.to_string()))?;
    let composition = CudaVNextComposition::create(
        ordinal,
        device_id,
        crate::product_composition::cuda_attention_policy_for_kv(
            engine.runtime.attention_execution_policy,
            engine.kv_cache.dtype,
        )?,
    )
    .map_err(|error| {
        FerrumError::device(format!("create production CUDA teacher runtime: {error}"))
    })?;
    let (runtime, operations, materializers, catalog) = composition.into_parts();
    crate::product_composition::create_vnext_executor(
        engine,
        defined,
        runtime,
        operations,
        materializers,
        catalog,
        cuda_weight_materializer_selection,
    )
}
