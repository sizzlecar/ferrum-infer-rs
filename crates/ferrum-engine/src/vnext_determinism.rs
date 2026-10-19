//! Production composition boundary for CUDA vNext determinism collection.
//!
//! The collector owns the same concrete executor used by `run` and `serve`.
//! Diagnostic code can inspect the resolved plan and live capability catalog,
//! while execution resources remain private to the executor.

#[cfg(feature = "cuda")]
use ferrum_execution::ModelExecutor;
#[cfg(feature = "cuda")]
use ferrum_execution::{DeterminismExecutionSpec, ProgramExecutor, ProgramExecutorConfig};
use ferrum_execution::{ExecutorBuildError, ExecutorBuildResult, ExecutorOwner};
#[cfg(feature = "cuda")]
use ferrum_kernels::backend::cuda::{
    vnext_ops::CudaVNextComposition, vnext_runtime::CudaDeviceRuntime,
};
#[cfg(feature = "cuda")]
use ferrum_models::vnext::DefinedProductionModel;
#[cfg(feature = "cuda")]
use ferrum_runtime::{
    CapabilityCatalog, DeviceId, ResolvedModelPlan, SubmissionWaveDeterminismEvidence,
};
#[cfg(feature = "cuda")]
use ferrum_types::EngineConfig;
use ferrum_types::{FerrumError, Result};

#[cfg(feature = "cuda")]
pub struct CudaVNextDeterminismCollector {
    executor: Option<ProgramExecutor<CudaDeviceRuntime>>,
}

#[cfg(feature = "cuda")]
impl CudaVNextDeterminismCollector {
    pub(crate) fn new(executor: ProgramExecutor<CudaDeviceRuntime>) -> Self {
        Self {
            executor: Some(executor),
        }
    }

    fn executor(&self) -> &ProgramExecutor<CudaDeviceRuntime> {
        self.executor
            .as_ref()
            .expect("live collector owns its original program")
    }

    /// Finish this original collection and observe its checked retirement.
    /// A collection failure remains the primary error even if cleanup fails.
    pub async fn finish_collection<T>(mut self, result: Result<T>) -> ExecutorBuildResult<T> {
        let executor = self
            .executor
            .take()
            .expect("collector closes its original program once");
        finish_owned_collection(ExecutorOwner::new(Box::new(executor)), result).await
    }

    pub async fn prepare(&self) -> Result<()> {
        self.executor().prepare_startup().await
    }

    pub fn resolved_model_plan(&self) -> &ResolvedModelPlan {
        self.executor().resolved_plan()
    }

    pub fn model_info(&self) -> &ferrum_types::ModelInfo {
        self.executor().info()
    }

    pub fn capability_catalog(&self) -> &CapabilityCatalog {
        self.executor().capability_catalog()
    }

    pub async fn collect_execution(
        &self,
        spec: &DeterminismExecutionSpec,
    ) -> Result<SubmissionWaveDeterminismEvidence> {
        self.executor().collect_determinism_execution(spec).await
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaVNextDeterminismCollector {
    fn drop(&mut self) {
        if let Some(executor) = self.executor.take() {
            // Cancellation uses the same final-close authority and service.
            // It never retries collection or treats ordinary Drop as retirement.
            let owner = ExecutorOwner::new(Box::new(executor));
            let _observation = owner.begin_shutdown();
        }
    }
}

async fn finish_owned_collection<T>(
    owner: ExecutorOwner,
    result: Result<T>,
) -> ExecutorBuildResult<T> {
    let cleanup = owner.begin_shutdown();
    drop(owner);
    let report = cleanup.wait().await;
    match result {
        Err(cause) => Err(ExecutorBuildError::with_cleanup(cause, cleanup)),
        Ok(value) => match report.result() {
            Ok(()) => Ok(value),
            Err(cause) => Err(ExecutorBuildError::with_cleanup(cause, cleanup)),
        },
    }
}

#[cfg(feature = "cuda")]
pub fn create_cuda_vnext_determinism_collector(
    engine: &EngineConfig,
    defined: &DefinedProductionModel,
    ordinal: usize,
) -> ferrum_execution::ExecutorBuildResult<CudaVNextDeterminismCollector> {
    let device_id = DeviceId::new(format!("device.cuda.{ordinal}"))
        .map_err(|error| FerrumError::device(error.to_string()))?;
    let composition = CudaVNextComposition::create(
        ordinal,
        device_id,
        engine.runtime.attention_execution_policy,
    )
    .map_err(|error| FerrumError::device(format!("create vNext CUDA runtime: {error}")))?;
    let (runtime, operation_registry, weight_materializers, catalog) = composition.into_parts();
    let executor = crate::product_composition::create_program_executor_with_configuration(
        engine,
        defined,
        runtime,
        operation_registry,
        weight_materializers,
        catalog,
        ferrum_kernels::backend::cuda::vnext_ops::cuda_weight_materializer_selection,
        None,
        |engine, info, runtime| {
            ProgramExecutorConfig::for_determinism_collection(engine, info, runtime)
        },
    )?;
    Ok(CudaVNextDeterminismCollector::new(executor))
}

#[cfg(test)]
mod retirement_tests;
