use super::*;
use ferrum_engine::{
    builder::EngineBuilder,
    continuous_engine::{CalibrationLimits, CalibrationSession},
};
use ferrum_types::{ExecutionResourceAuthority, RuntimeConfigSnapshot, RuntimeConfigSource};

pub(super) async fn prepare(
    cmd: &CalibrateSloCommand,
    config: &CliConfig,
    manifest: &manifest::Manifest,
) -> Result<(
    CalibrationSession,
    serde_json::Value,
    inputs::PreparedInputs,
)> {
    let policy = super::super::slo::load(Some(&cmd.slo_config), None)
        .await?
        .ok_or_else(|| FerrumError::config("calibration requires an explicit SLO policy"))?;
    if policy.config.mode != ferrum_types::SloMode::Observe
        || policy.config.admission.time_policy
            != ferrum_types::SloTimeAdmissionPolicy::CompleteRequests
        || policy.config.output.transport != ferrum_types::SloOutputTransport::Credited
    {
        return Err(FerrumError::config("calibration policy must explicitly select Observe, CompleteRequests and credited output"));
    }
    validate_export_configuration(cmd, manifest, &policy.config)?;
    let mut requested = match cmd.startup_usage {
        CalibrationStartupUsage::Run => super::super::run::run_base_runtime_config(
            config,
            RuntimeConfigSnapshot::capture_current(),
        ),
        CalibrationStartupUsage::Serve => super::super::serve::merge_runtime_config_sources(
            config.runtime.runtime_config_entries(),
            RuntimeConfigSnapshot::capture_current(),
            Vec::new(),
        ),
    };
    policy.apply_to_snapshot(&mut requested);
    if let Some(budget) = cmd.runtime_memory_budget_bytes {
        requested.upsert(
            "FERRUM_RUNTIME_MEMORY_BUDGET_BYTES",
            budget.to_string(),
            RuntimeConfigSource::Cli,
        );
    }
    let mut device = super::super::run::select_device(&cmd.backend)?;
    let mut devices =
        crate::gpu_devices::resolve_cuda_gpu_devices(cmd.gpu_devices.as_deref(), &device)?;
    if let Some(selected) = &devices {
        device = selected.primary_device();
    }
    if let Some(kv) = &cmd.kv_dtype {
        requested.upsert("FERRUM_KV_DTYPE", kv, RuntimeConfigSource::Cli);
    }
    let prepared = super::super::run::model_startup::prepare_product_source(
        &cmd.model,
        &cmd.product_sources,
        config,
        cmd.numerical_profile.as_ref(),
        &requested,
    )
    .await?;
    let input = prepared.input;
    let defined = prepared.defined_model.ok_or_else(|| {
        FerrumError::unsupported("calibration requires a registered native PlanRuntime model")
    })?;
    if let Some(selected) = devices.as_mut() {
        selected.apply_model_layer_count(defined.descriptor().layer_count())?;
        for entry in selected.runtime_config_entries() {
            requested.upsert_entry(entry);
        }
    }
    let mut engine = input.engine_config;
    engine.backend.device = device.clone();
    engine.scheduler.policy = ferrum_types::SchedulingPolicy::ContinuousBatch;
    engine.backend.backend_options.insert(
        "model_path".into(),
        serde_json::json!(input.source.local_path),
    );
    if let Some(selected) = &devices {
        selected.insert_backend_options(&mut engine.backend.backend_options);
    }
    crate::layer_split_pipeline::insert_backend_option_from_runtime(
        &requested,
        &mut engine.backend.backend_options,
    )?;
    let authority = ExecutionResourceAuthority::PlanRuntime;
    let memory =
        crate::startup::memory_request(&device, authority, cmd.gpu_memory_utilization, &requested)?;
    let hardware = crate::startup::hardware_for_request(&device, memory.as_ref());
    let capabilities = defined.model_capabilities(
        &engine.numerical_execution,
        ferrum_types::KvStorageFormat::try_from(engine.kv_cache.dtype)
            .map_err(FerrumError::config)?,
    )?;
    let mut workload = ferrum_types::WorkloadProfile::serving_default();
    let usage = match cmd.startup_usage {
        CalibrationStartupUsage::Run => {
            workload.serving_mode = "interactive".into();
            workload.priority = ferrum_types::WorkloadPriority::Latency;
            crate::startup::StartupUsage::SingleRequest
        }
        CalibrationStartupUsage::Serve => crate::startup::StartupUsage::PersistentServing,
    };
    let mut resolved = crate::startup::resolve_config(
        requested,
        capabilities,
        hardware,
        workload,
        authority,
        usage,
    )?;
    crate::runtime_env::materialize_runtime_env_effective(&resolved.runtime_config);
    engine
        .apply_runtime_config_snapshot(&resolved.runtime_config)
        .map_err(FerrumError::config)?;
    engine.runtime.startup_memory_request = memory;
    engine.sampling.default_params.model_output_protocol = defined.descriptor().output_protocol();
    let template = crate::source_resolver::load_defined_product_chat_template(&defined)?;
    let source_identity = crate::source_resolver::product_source_identity(
        Some(&defined),
        input.model_sources.as_deref(),
        &input.requested_model,
        &input.public_model_id,
        Some(&template),
    )?;
    let inputs = inputs::PreparedInputs::prepare(manifest, template)?;
    if let Some(reference) = &manifest.reference {
        reference.validate(manifest, &inputs)?;
    }
    let session = EngineBuilder::new(engine)
        .with_defined_model(defined)
        .build_calibration(CalibrationLimits::new(manifest.protocol.maximum_requests)?)
        .await?;
    crate::startup::apply_engine_plan(&mut resolved, session.configuration());
    let provenance = serde_json::json!({
        "requested_model": input.requested_model, "resolved_model": input.public_model_id,
        "product_sources": source_identity, "runtime": resolved.runtime_config,
        "engine": session.configuration(), "prompt_rendering": inputs.provenance(),
        "startup_usage": format!("{:?}", cmd.startup_usage),
        "cli_version": env!("CARGO_PKG_VERSION"),
    });
    Ok((session, provenance, inputs))
}

pub(super) fn validate_export_configuration(
    cmd: &CalibrateSloCommand,
    manifest: &manifest::Manifest,
    policy: &ferrum_types::SloConfig,
) -> Result<()> {
    paths::distinct(paths::outputs(cmd, manifest))?;
    let selected = manifest.validation_model.selected();
    if selected.map(|(predictor, _, _)| predictor)
        != policy
            .cost_observation
            .predictor
            .is_selected()
            .then_some(policy.cost_observation.predictor)
    {
        return Err(FerrumError::config(
            "selected predictor version and independent calibration protocol must match exactly",
        ));
    }
    if let Some((_, export, _)) = selected {
        export.validate().map_err(FerrumError::config)?;
        if policy.cost_observation.profile_export.is_some() {
            return Err(FerrumError::config(
                "selected capture owns its fit/residual files; legacy online export must be absent",
            ));
        }
        if policy
            .cost_observation
            .profile_import
            .declared_local_clock_max_error_ns
            .is_none()
        {
            return Err(FerrumError::config(
                "selected whole-wave calibration requires explicit local import clock accuracy",
            ));
        }
    }
    if matches!(
        manifest.validation_model,
        manifest::ValidationSource::ExportedProfile { .. }
    ) {
        if policy.cost_observation.profile_export.is_none() {
            return Err(FerrumError::config("exported_profile validation requires cost_observation.profile_export to retain original training observations"));
        }
        if policy
            .cost_observation
            .profile_import
            .declared_local_clock_max_error_ns
            .is_none()
        {
            return Err(FerrumError::config("exported_profile validation requires an explicit profile_import.declared_local_clock_max_error_ns"));
        }
    }
    if let Some(export) = &policy.cost_observation.profile_export {
        let mut all = paths::outputs(cmd, manifest);
        all.extend([export.path.as_path(), export.observations_path.as_path()]);
        paths::distinct(all)?;
        paths::validate([&export.path, &export.observations_path])?;
    }
    Ok(())
}
