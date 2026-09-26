use super::*;
use ferrum_interfaces::ModelExecutor;
use ferrum_types::{
    teacher_capture::VNextTeacherCaptureIdentity, RuntimeConfigSnapshot, RuntimeConfigSource,
};

pub(super) async fn collect(
    cmd: &VNextTeacherCommand,
    config: &CliConfig,
    spec: &VNextTeacherExecutionSpec,
    artifacts: &mut artifacts::Artifacts,
) -> Result<()> {
    let device = crate::backend_selection::select_device(&cmd.backend)?;
    let supported = match &device {
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        ferrum_types::Device::Metal => true,
        #[cfg(feature = "cuda")]
        ferrum_types::Device::CUDA(_) => true,
        _ => false,
    };
    if !supported {
        return Err(FerrumError::unsupported(
            "real-history teacher capture requires a Metal or CUDA production runtime",
        ));
    }
    let mut runtime = RuntimeConfigSnapshot::default();
    runtime.upsert("FERRUM_KV_DTYPE", "fp16", RuntimeConfigSource::Cli);
    let policy = ferrum_types::NumericalExecutionPolicy::Require(cmd.numerical_profile.clone());
    let prepared = super::super::run::model_startup::prepare_product_source(
        &cmd.model,
        &cmd.product_sources,
        config,
        Some(&policy),
        &runtime,
    )
    .await?;
    let input = prepared.input;
    let defined = prepared.defined_model.ok_or_else(|| {
        FerrumError::unsupported("real-history capture requires a registered production model")
    })?;
    // Retain and validate the same template source as the typed product family.
    // The already-tokenized teacher histories remain unchanged; this template
    // is provenance only and is not applied to their prompt token IDs.
    let template = crate::source_resolver::load_defined_product_chat_template(&defined)?;
    let source_identity = crate::source_resolver::product_source_identity(
        Some(&defined),
        input.model_sources.as_deref(),
        &input.requested_model,
        &input.public_model_id,
        Some(&template),
    )?
    .ok_or_else(|| FerrumError::model("teacher model has no retained product source identity"))?;
    let mut engine = input.engine_config;
    engine.backend.device = device.clone();
    engine.backend.dtype = defined.descriptor().execution_dtype();
    engine.backend.backend_options.insert(
        "model_path".into(),
        serde_json::json!(input.source.local_path),
    );
    engine.scheduler.policy = ferrum_types::SchedulingPolicy::ContinuousBatch;
    engine.scheduler.max_running_requests = spec.owners.len();
    engine.scheduler.sequence_fit_policy = ferrum_types::SequenceFitPolicy::FullInputMustFit;
    engine.batching.max_num_batched_tokens = spec.prefill_chunk_tokens.max(spec.owners.len());
    engine.runtime.max_model_len = Some(spec.maximum_sequence_tokens);
    engine.runtime.kv_capacity = Some(spec.maximum_sequence_tokens);
    engine.runtime.chunked_prefill_size = Some(spec.prefill_chunk_tokens);
    engine.memory.usable_capacity_bytes = Some(cmd.runtime_memory_budget_bytes.get());
    engine.kv_cache.dtype = ferrum_types::KvCacheDtype::Fp16;
    let executable = std::env::current_exe().map_err(|error| FerrumError::io(error.to_string()))?;
    artifacts.manifest.configuration =
        serde_json::to_value(&engine).map_err(|error| FerrumError::internal(error.to_string()))?;
    artifacts.manifest.identity = Some(VNextTeacherCaptureIdentity {
        model_id: input.public_model_id,
        model_source: serde_json::to_value(source_identity)
            .map_err(|error| FerrumError::internal(error.to_string()))?,
        numerical_profile: cmd.numerical_profile.to_string(),
        kv_storage: "fp16".into(),
        family_fingerprint: String::new(),
        program_fingerprint: String::new(),
        resolved_plan_fingerprint: String::new(),
        binary: artifacts::file_identity(&executable)?,
        history_file: artifacts::file_identity(&cmd.history_file)?,
    });
    match device {
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        ferrum_types::Device::Metal => {
            let executor = ferrum_engine::vnext_teacher::create_metal_vnext_teacher_collector(
                &engine, &defined,
            )?;
            collect_from_executor(&executor, spec, artifacts).await
        }
        #[cfg(feature = "cuda")]
        ferrum_types::Device::CUDA(ordinal) => {
            let executor = ferrum_engine::vnext_teacher::create_cuda_vnext_teacher_collector(
                &engine, &defined, ordinal,
            )?;
            collect_from_executor(&executor, spec, artifacts).await
        }
        _ => Err(FerrumError::unsupported(
            "teacher production backend is absent from this binary",
        )),
    }
}

async fn collect_from_executor<R: ferrum_interfaces::vnext::DeviceRuntime>(
    executor: &ferrum_models::VNextModelExecutor<R>,
    spec: &VNextTeacherExecutionSpec,
    artifacts: &mut artifacts::Artifacts,
) -> Result<()> {
    artifacts.manifest.vocabulary_size = executor.info().vocab_size;
    artifacts
        .manifest
        .identity
        .as_mut()
        .ok_or_else(|| FerrumError::internal("teacher identity was lost"))?
        .resolved_plan_fingerprint = executor.resolved_plan().fingerprint().to_owned();
    artifacts.persist()?;
    let summary = executor.collect_teacher_history(spec, artifacts).await?;
    let identity = artifacts
        .manifest
        .identity
        .as_mut()
        .ok_or_else(|| FerrumError::internal("teacher identity was lost"))?;
    identity.family_fingerprint = summary.family_fingerprint;
    identity.program_fingerprint = summary.program_fingerprint;
    if summary.decision_count != artifacts.manifest.decisions.len()
        || summary.physical_wave_count != artifacts.manifest.waves.len()
        || summary.vocabulary_size != artifacts.manifest.vocabulary_size
    {
        return Err(FerrumError::internal(
            "teacher persisted inventory differs from completed execution",
        ));
    }
    Ok(())
}
