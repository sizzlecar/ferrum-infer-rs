//! Production composition boundary for CUDA vNext determinism collection.
//!
//! The collector owns the same concrete executor used by `run` and `serve`.
//! Diagnostic code can inspect the resolved plan and live capability catalog,
//! while execution resources remain private to the executor.

use ferrum_interfaces::vnext::{
    CapabilityCatalog, DeviceId, ResolvedModelPlan, SubmissionWaveDeterminismEvidence,
};
use ferrum_interfaces::ModelExecutor;
use ferrum_kernels::backend::cuda::{
    vnext_ops::CudaVNextComposition, vnext_runtime::CudaDeviceRuntime,
};
use ferrum_models::vnext::DefinedProductionModel;
use ferrum_models::{VNextDeterminismExecutionSpec, VNextExecutorConfig, VNextModelExecutor};
use ferrum_types::{Device, EngineConfig, FerrumError, Result};

pub struct CudaVNextDeterminismCollector {
    executor: VNextModelExecutor<CudaDeviceRuntime>,
}

impl CudaVNextDeterminismCollector {
    pub(crate) fn new(executor: VNextModelExecutor<CudaDeviceRuntime>) -> Self {
        Self { executor }
    }

    pub async fn prepare(&self) -> Result<()> {
        self.executor.prepare_startup().await
    }

    pub fn resolved_model_plan(&self) -> &ResolvedModelPlan {
        self.executor.resolved_plan()
    }

    pub fn model_info(&self) -> &ferrum_types::ModelInfo {
        self.executor.info()
    }

    pub fn capability_catalog(&self) -> &CapabilityCatalog {
        self.executor.capability_catalog()
    }

    pub async fn collect_execution(
        &self,
        spec: &VNextDeterminismExecutionSpec,
    ) -> Result<SubmissionWaveDeterminismEvidence> {
        self.executor.collect_determinism_execution(spec).await
    }
}

pub fn create_cuda_vnext_determinism_collector(
    engine: &EngineConfig,
    defined: &DefinedProductionModel,
    ordinal: usize,
) -> Result<CudaVNextDeterminismCollector> {
    let device_id = DeviceId::new(format!("device.cuda.{ordinal}"))
        .map_err(|error| FerrumError::device(error.to_string()))?;
    let composition = CudaVNextComposition::create(
        ordinal,
        device_id,
        engine.runtime.attention_execution_policy,
    )
    .map_err(|error| FerrumError::device(format!("create vNext CUDA runtime: {error}")))?;
    let (runtime, operation_registry, weight_materializers, catalog) = composition.into_parts();
    let executor = crate::product_composition::create_vnext_executor_with_configuration(
        engine,
        defined,
        runtime,
        operation_registry,
        weight_materializers,
        catalog,
        ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection,
        |info, runtime| VNextExecutorConfig::for_determinism_collection(engine, info, runtime),
    )?;
    Ok(CudaVNextDeterminismCollector::new(executor))
}
