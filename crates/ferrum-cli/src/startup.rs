//! Shared typed startup policy for the product entrypoints.

use ferrum_types::{
    EngineConfig, ExecutionResourceAuthority, FerrumConfigBuilder, FerrumError,
    HardwareCapabilities, ModelCapabilities, ResolvedFerrumConfig, Result, RuntimeConfigSnapshot,
    RuntimeConfigSource, StartupMemoryRequest, WorkloadProfile,
};

pub(crate) fn resolve_config(
    mut requested: RuntimeConfigSnapshot,
    model: ModelCapabilities,
    hardware: HardwareCapabilities,
    workload: WorkloadProfile,
    authority: ExecutionResourceAuthority,
) -> Result<ResolvedFerrumConfig> {
    if authority == ExecutionResourceAuthority::PlanRuntime
        && crate::runtime_env::runtime_snapshot_value(&requested, "FERRUM_MAX_BATCHED_TOKENS")
            .is_none()
    {
        // A common starting target; the compiled plan fits its real workspace
        // to available memory before allocating model resources.
        let max_sequences =
            crate::runtime_env::runtime_snapshot_value(&requested, "FERRUM_PAGED_MAX_SEQS")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(workload.target_concurrency);
        requested.upsert(
            "FERRUM_MAX_BATCHED_TOKENS",
            2048.max(max_sequences).to_string(),
            RuntimeConfigSource::Default,
        );
    }
    FerrumConfigBuilder::new(requested)
        .with_model_capabilities(model)
        .with_hardware_capabilities(hardware)
        .with_workload_profile(workload)
        .with_execution_resource_authority(authority)
        .resolve()
        .map_err(|error| FerrumError::config(format!("invalid auto config: {error}")))
}

pub(crate) fn memory_request(
    device: &ferrum_types::Device,
    authority: ExecutionResourceAuthority,
    utilization: f32,
    requested: &RuntimeConfigSnapshot,
) -> Result<Option<StartupMemoryRequest>> {
    if authority == ExecutionResourceAuthority::PlanRuntime {
        let memory = ferrum_kernels::backend::probe_device_memory(device)?;
        return StartupMemoryRequest::from_snapshot(memory, utilization, requested)
            .map(Some)
            .map_err(FerrumError::config);
    }
    Ok(None)
}

pub(crate) fn hardware_for_request(
    device: &ferrum_types::Device,
    request: Option<&StartupMemoryRequest>,
) -> HardwareCapabilities {
    let mut hardware = crate::commands::serve::hardware_capabilities_for_device(device);
    if let Some(request) = request {
        hardware.vram_bytes = Some(request.device.capacity_bytes);
    }
    hardware
}

pub(crate) fn apply_engine_plan(resolved: &mut ResolvedFerrumConfig, engine_config: &EngineConfig) {
    if let Some(plan) = engine_config.runtime.startup_memory_plan.as_ref() {
        resolved.apply_startup_memory_plan(plan);
        eprintln!(
            "Memory plan: context={} tokens, concurrent requests={}, batch={} tokens, budget={:.2} GiB",
            plan.selected.context_tokens,
            plan.selected.max_sequences,
            plan.selected.max_batch_tokens,
            plan.request.usable_capacity_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        for reason in &plan.reasons {
            eprintln!("  {reason}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{DeviceMemorySnapshot, RuntimeConfigEntry};

    fn hardware() -> HardwareCapabilities {
        HardwareCapabilities {
            backend: "metal".into(),
            vram_bytes: Some(32 * 1024 * 1024 * 1024),
            supported_dtypes: vec!["fp16".into(), "fp32".into()],
            supported_kv_dtypes: vec!["fp16".into()],
            ..HardwareCapabilities::unknown()
        }
    }

    fn model() -> ModelCapabilities {
        let mut model = ModelCapabilities::unknown();
        model.max_context_len = Some(262_144);
        model
    }

    #[test]
    fn native_workloads_share_context_policy_without_legacy_kv_defaults() {
        for workload in [
            WorkloadProfile::serving_default(),
            WorkloadProfile::serving_default_for_hardware(&hardware()),
        ] {
            let concurrency = workload.target_concurrency;
            let resolved = resolve_config(
                RuntimeConfigSnapshot::default(),
                model(),
                hardware(),
                workload,
                ExecutionResourceAuthority::PlanRuntime,
            )
            .unwrap();
            let mut engine = EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert_eq!(engine.runtime.max_model_len, Some(262_144));
            assert_eq!(engine.runtime.kv_capacity, None);
            assert_eq!(engine.scheduler.max_running_requests, concurrency);
            assert_eq!(engine.batching.max_num_batched_tokens, 2048);
        }
        assert!(WorkloadProfile::serving_default_for_hardware(&hardware()).target_concurrency > 1);
    }

    #[test]
    fn native_batch_default_covers_explicit_admission_width() {
        let requested = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
            "FERRUM_PAGED_MAX_SEQS",
            "3000",
            RuntimeConfigSource::Cli,
        )]);
        let resolved = resolve_config(
            requested,
            model(),
            hardware(),
            WorkloadProfile::serving_default_for_hardware(&hardware()),
            ExecutionResourceAuthority::PlanRuntime,
        )
        .unwrap();
        let mut engine = EngineConfig::default();
        engine
            .apply_runtime_config_snapshot(&resolved.runtime_config)
            .unwrap();
        assert_eq!(engine.scheduler.max_running_requests, 3000);
        assert_eq!(engine.batching.max_num_batched_tokens, 3000);
    }

    #[test]
    fn explicit_batch_limits_automatic_concurrency_before_memory_fitting() {
        let requested = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
            "FERRUM_MAX_BATCHED_TOKENS",
            "1",
            RuntimeConfigSource::Cli,
        )]);
        let workload = WorkloadProfile::serving_default_for_hardware(&hardware());
        assert!(workload.target_concurrency > 1);
        let resolved = resolve_config(
            requested.clone(),
            model(),
            hardware(),
            workload.clone(),
            ExecutionResourceAuthority::PlanRuntime,
        )
        .unwrap();
        let mut engine = EngineConfig::default();
        engine
            .apply_runtime_config_snapshot(&resolved.runtime_config)
            .unwrap();
        assert_eq!(engine.scheduler.max_running_requests, 1);
        assert_eq!(engine.batching.max_num_batched_tokens, 1);
        assert!(resolved
            .runtime_config
            .entries
            .contains(&requested.entries[0]));

        // Two explicit limits retain the normal incompatibility error.
        let mut incompatible = requested;
        incompatible.upsert("FERRUM_PAGED_MAX_SEQS", "2", RuntimeConfigSource::Cli);
        assert!(resolve_config(
            incompatible,
            model(),
            hardware(),
            workload,
            ExecutionResourceAuthority::PlanRuntime,
        )
        .is_err());
    }

    #[test]
    fn native_startup_preserves_explicit_limits_and_their_sources() {
        for source in [
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Env,
            RuntimeConfigSource::Cli,
            RuntimeConfigSource::ScriptCase,
        ] {
            let requested = RuntimeConfigSnapshot::from_entries([
                RuntimeConfigEntry::new("FERRUM_MAX_MODEL_LEN", "8192", source),
                RuntimeConfigEntry::new("FERRUM_PAGED_MAX_SEQS", "7", source),
                RuntimeConfigEntry::new("FERRUM_MAX_BATCHED_TOKENS", "256", source),
            ]);
            let memory = StartupMemoryRequest::from_snapshot(
                DeviceMemorySnapshot {
                    capacity_bytes: 32 * 1024 * 1024 * 1024,
                    available_bytes: 24 * 1024 * 1024 * 1024,
                    source: "fixture".into(),
                },
                0.9,
                &requested,
            )
            .unwrap();
            assert!(memory.context_is_explicit);
            assert!(memory.sequences_is_explicit);
            assert!(memory.batch_is_explicit);
            let resolved = resolve_config(
                requested.clone(),
                model(),
                hardware(),
                WorkloadProfile::serving_default_for_hardware(&hardware()),
                ExecutionResourceAuthority::PlanRuntime,
            )
            .unwrap();
            for entry in requested.entries {
                assert!(resolved.runtime_config.entries.contains(&entry));
            }
        }
    }

    #[test]
    fn legacy_startup_does_not_probe_native_device_memory() {
        assert!(memory_request(
            &ferrum_types::Device::ROCm(0),
            ExecutionResourceAuthority::LegacyEngine,
            0.9,
            &RuntimeConfigSnapshot::default(),
        )
        .unwrap()
        .is_none());
    }
}
