use super::*;

pub struct PreparedDenoisingBatch<R: DeviceRuntime> {
    pub(super) plan: FastH3DenoisingBatchPlan,
    pub(super) compiled: Arc<CompiledTensorComponent<R>>,
}
impl<R: DeviceRuntime> Clone for PreparedDenoisingBatch<R> {
    fn clone(&self) -> Self {
        Self {
            plan: self.plan.clone(),
            compiled: self.compiled.clone(),
        }
    }
}
impl<R: DeviceRuntime> PreparedDenoisingBatch<R> {
    pub fn compile(
        config: FastH3DenoisingConfig,
        limits: FastH3DenoisingLimits,
        regions: Arc<CheckedVsaRegions>,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        policy: &TensorComponentPolicy,
        materializers: &WeightMaterializerRegistry,
    ) -> Result<Self, VNextError> {
        let (plan, family) =
            FastH3DenoisingBatchPlan::prepare_with_family(config, limits, regions)?;
        let compiled = Arc::new(CompiledTensorComponent::compile(
            family,
            registry,
            catalog,
            policy,
            materializers,
            plan.compile_options(),
        )?);
        Ok(Self { plan, compiled })
    }
    pub fn plan(&self) -> &FastH3DenoisingBatchPlan {
        &self.plan
    }
    pub fn compiled(&self) -> &Arc<CompiledTensorComponent<R>> {
        &self.compiled
    }
}

/// One actual evaluation-specific PDD composition, compiled once. Its source
/// remains an explicit caller-owned view and is admitted only while predicting.
pub struct PreparedPrediction<R: DeviceRuntime> {
    pub(super) config: FastH3FinalOutputConfig,
    pub(super) compiled: Arc<CompiledTensorComponent<R>>,
    pub(super) device: Option<(FastH3FinalOutputConfig, Arc<CompiledTensorComponent<R>>)>,
}
impl<R: DeviceRuntime> Clone for PreparedPrediction<R> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            compiled: self.compiled.clone(),
            device: self.device.clone(),
        }
    }
}
impl<R: DeviceRuntime> PreparedPrediction<R> {
    pub fn compile(
        config: FastH3FinalOutputConfig,
        limits: FastH3FinalOutputLimits,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        policy: &TensorComponentPolicy,
        materializers: &WeightMaterializerRegistry,
    ) -> Result<Self, VNextError> {
        let family = definition(
            FastH3FinalOutputFamily::new(limits)?,
            &config,
            FastH3FinalOutputFamily::PROFILE_ID,
        )?;
        let compiled = Arc::new(CompiledTensorComponent::compile(
            family,
            registry,
            catalog,
            policy,
            materializers,
            &config.compile_options(limits)?,
        )?);
        let device_config = config
            .clone()
            .with_input_residency(FastH3PredictionInput::RequestState);
        let device_family = definition(
            FastH3FinalOutputFamily::new(limits)?,
            &device_config,
            FastH3FinalOutputFamily::PROFILE_ID,
        )?;
        let device = match CompiledTensorComponent::compile(
            device_family,
            registry,
            catalog,
            policy,
            materializers,
            &device_config.compile_options(limits)?,
        ) {
            Ok(component) => {
                if compiled.family().weights() != component.family().weights() {
                    return Err(invalid(
                        "prediction input residency changed the source schema",
                    ));
                }
                Some((device_config, Arc::new(component)))
            }
            Err(VNextError::PlanCapacityExceeded { .. }) => None,
            Err(error) => return Err(error),
        };
        Ok(Self {
            config,
            compiled,
            device,
        })
    }
    pub fn configuration(&self) -> &FastH3FinalOutputConfig {
        &self.config
    }
}
fn definition<F: ModelFamilyProvider>(
    family: F,
    config: &F::Config,
    profile: &str,
) -> Result<PreparedModelFamily, VNextError> {
    TypedFamilyRegistration::new(family).prepare_with_profile(
        &serde_json::to_value(config).map_err(invalid)?,
        &NumericalProfileId::new(profile).map_err(invalid)?,
    )
}
