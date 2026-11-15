use super::*;
use ferrum_execution::component::TensorComponentPolicy;
use ferrum_models::fast_h3::*;

/// Borrowed compilation inputs from the same product composition; this is not
/// an execution owner, allocation authority or a cache of model instances.
pub struct ConditioningCompilation<'a, R: DeviceRuntime> {
    pub registry: &'a OperationRuntimeRegistry<R>,
    pub catalog: &'a CapabilityCatalog,
    pub policy: &'a TensorComponentPolicy,
    pub materializers: &'a WeightMaterializerRegistry,
}
impl<R: DeviceRuntime> ConditioningCompilation<'_, R> {
    pub(super) fn compile_prepared(
        &self,
        prepared: PreparedModelFamily,
        options: &ProgramPlanCompileOptions,
    ) -> Result<Arc<CompiledTensorComponent<R>>, ConditioningFailure> {
        Ok(Arc::new(
            CompiledTensorComponent::compile(
                prepared,
                self.registry,
                self.catalog,
                self.policy,
                self.materializers,
                options,
            )
            .map_err(|e| prep("conditioning compile", e))?,
        ))
    }
    fn compile<F: ModelFamilyProvider>(
        &self,
        family: F,
        config: &F::Config,
        profile: &str,
        options: ProgramPlanCompileOptions,
    ) -> Result<Arc<CompiledTensorComponent<R>>, ConditioningFailure> {
        let raw = serde_json::to_value(config).map_err(|e| prep("conditioning config", e))?;
        let profile =
            NumericalProfileId::new(profile).map_err(|e| prep("conditioning profile", e))?;
        let prepared = TypedFamilyRegistration::new(family)
            .prepare_with_profile(&raw, &profile)
            .map_err(|e| prep("conditioning family", e))?;
        self.compile_prepared(prepared, &options)
    }
}
fn prep(stage: &'static str, e: impl fmt::Display) -> ConditioningFailure {
    ConditioningFailure::preparation(stage, e)
}
fn execution_ids(
    initialization: &ComponentInitializationConfig,
    suffix: &str,
) -> Result<(RunId, RequestIdentity), ConditioningFailure> {
    Ok((
        RunId::new(format!("{}.{}", initialization.run_id.as_str(), suffix))
            .map_err(|e| prep("conditioning identity", e))?,
        initialization.request_id.clone(),
    ))
}

/// One cold computation per checkpoint/schedule. Every request borrows these
/// same Arc owners; per-request recreation would defeat checked batch binding.
pub struct ConditioningTables {
    pub layers: Arc<[FastH3ConditioningTable]>,
    pub output: Arc<FastH3OutputConditioningTable>,
}
#[allow(clippy::too_many_arguments)]
pub fn precompute_tables<R: DeviceRuntime>(
    package: Arc<FastH3TransformerPackage>,
    schedule: &FastH3ConditioningSchedule,
    compilation: &ConditioningCompilation<'_, R>,
    runtime: Arc<R>,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<R>,
    limits: FastH3TimeConditioningLimits,
    output_limits: FastH3OutputConditioningLimits,
    mut initialization: impl FnMut(&str) -> Result<ComponentInitializationConfig, VNextError>,
) -> Result<ConditioningTables, ConditioningFailure> {
    let mut tables = Vec::with_capacity(package.config().layers());
    for layer in 0..package.config().layers() {
        let config = FastH3TimeConditioningConfig::from_package(
            &package,
            layer,
            schedule.timesteps().len() as u64,
            limits,
        )
        .map_err(|e| prep("time config", e))?;
        let source = FastH3TimeConditioningSource::new(&package, &config, host, limits)
            .map_err(|e| prep("time source", e))?;
        let compiled = compilation.compile(
            FastH3TimeConditioningFamily::new(limits).map_err(|e| prep("time family", e))?,
            &config,
            FastH3TimeConditioningFamily::PROFILE_ID,
            config
                .compile_options()
                .map_err(|e| prep("time options", e))?,
        )?;
        let input = config
            .managed_schedule_features(host, schedule)
            .map_err(|e| prep("time features", e))?;
        let init = initialization(&format!("time.{layer}"))
            .map_err(|e| prep("time initialization policy", e))?;
        let (run, request) = execution_ids(&init, "forward")?;
        let outputs = forward(
            compiled,
            runtime.clone(),
            host,
            actor,
            init,
            run,
            request,
            &source,
            package.clone(),
            vec![input],
        )?;
        tables.push(
            FastH3ConditioningTable::from_execution(config, schedule, outputs, limits)
                .map_err(|e| prep("time table", e))?,
        );
    }
    let config = FastH3OutputTimeConfig::from_package(&package, schedule, output_limits)
        .map_err(|e| prep("output time config", e))?;
    let source =
        FastH3OutputConditioningSource::for_time_projection(&package, &config, output_limits)
            .map_err(|e| prep("output time source", e))?;
    let compiled = compilation.compile(
        FastH3OutputTimeFamily::new(output_limits).map_err(|e| prep("output time family", e))?,
        &config,
        FastH3OutputTimeFamily::PROFILE_ID,
        config
            .compile_options()
            .map_err(|e| prep("output time options", e))?,
    )?;
    let first = tables
        .first()
        .ok_or_else(|| prep("time table", "empty transformer"))?;
    let input = config
        .embedding_input(first, host)
        .map_err(|e| prep("output time input", e))?;
    let init =
        initialization("output.time").map_err(|e| prep("output time initialization policy", e))?;
    let (run, request) = execution_ids(&init, "forward")?;
    let outputs = forward(
        compiled,
        runtime,
        host,
        actor,
        init,
        run,
        request,
        &source,
        package.clone(),
        vec![input],
    )?;
    let output = FastH3OutputConditioningTable::from_execution(config, outputs, output_limits)
        .map_err(|e| prep("output time table", e))?;
    Ok(ConditioningTables {
        layers: tables.into(),
        output: Arc::new(output),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn refine_text<R: DeviceRuntime>(
    package: Arc<FastH3TransformerPackage>,
    text: &TensorExecutionOutput,
    compilation: &ConditioningCompilation<'_, R>,
    runtime: Arc<R>,
    host: &HostMemoryAccount,
    actor: &mut TensorSerialActor<R>,
    input_limits: FastH3InputLimits,
    refiner_limits: FastH3TextRefinerLimits,
    mut initialization: impl FnMut(&str) -> Result<ComponentInitializationConfig, VNextError>,
) -> Result<TensorExecutionOutput, ConditioningFailure> {
    let dimensions = text.tensor().dimensions();
    if dimensions.len() != 2
        || dimensions[1] != package.config().text_dim() as u64
        || text.tensor().element_type() != ElementType::Bf16
        || text.value_id().as_str()
            != ferrum_models::qwen3_vl_text::Qwen3VLTextEncoderFamily::OUTPUT_ID
        || !host.owns(text.bytes())
    {
        return Err(prep(
            "text projection input",
            "original completed encoder features differ",
        ));
    }
    let config =
        FastH3InputProjectionConfig::text_from_package(&package, dimensions[0], input_limits)
            .map_err(|e| prep("text projection config", e))?;
    let source = FastH3InputProjectionSource::new(&package, &config, host, input_limits)
        .map_err(|e| prep("text projection source", e))?;
    let compiled = compilation.compile(
        FastH3InputProjectionFamily::new(input_limits)
            .map_err(|e| prep("text projection family", e))?,
        &config,
        FastH3InputProjectionFamily::TEXT_PROFILE_ID,
        config
            .compile_options()
            .map_err(|e| prep("text projection options", e))?,
    )?;
    // An admitted copy bridges completion-owned Output to the next component's
    // InputPreprocess. No owner is reclassified and no completion is forged.
    let mut owner = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, text.bytes().len())
        .map_err(|e| prep("text projection admission", e))?;
    owner.as_mut().copy_from_slice(text.bytes().bytes());
    let input = ManagedHostTensor::new(
        ProgramValueId::new(FastH3InputProjectionFamily::TEXT_INPUT_ID)
            .map_err(|e| prep("text projection input", e))?,
        text.tensor().clone(),
        owner.freeze(),
    )
    .map_err(|e| prep("text projection input", e))?;
    let init = initialization("text.projection")
        .map_err(|e| prep("text projection initialization policy", e))?;
    let (run, request) = execution_ids(&init, "forward")?;
    let projected = forward(
        compiled,
        runtime.clone(),
        host,
        actor,
        init,
        run,
        request,
        &source,
        package.clone(),
        vec![input],
    )?;
    let config =
        FastH3TextRefinerConfig::from_package(&package, vec![dimensions[0]], refiner_limits)
            .map_err(|e| prep("text refiner config", e))?;
    let source = FastH3TextRefinerSource::new(&package, &config, refiner_limits)
        .map_err(|e| prep("text refiner source", e))?;
    let compiled = compilation.compile(
        FastH3TextRefinerFamily::new(refiner_limits).map_err(|e| prep("text refiner family", e))?,
        &config,
        FastH3TextRefinerFamily::PROFILE_ID,
        config
            .compile_options()
            .map_err(|e| prep("text refiner options", e))?,
    )?;
    if projected.len() != 1 {
        return Err(prep("text projection", "expected one original output"));
    }
    let input = FastH3DenoisingRequest::projected_text_input(&config, &projected[0], host)
        .map_err(|e| prep("text refiner input", e))?;
    drop(projected);
    let init = initialization("text.refiner")
        .map_err(|e| prep("text refiner initialization policy", e))?;
    let (run, request) = execution_ids(&init, "forward")?;
    let mut outputs = forward(
        compiled,
        runtime,
        host,
        actor,
        init,
        run,
        request,
        &source,
        package.clone(),
        vec![input],
    )?;
    if outputs.len() != 1 {
        return Err(prep("text refiner", "expected one original output"));
    }
    Ok(outputs.remove(0))
}
