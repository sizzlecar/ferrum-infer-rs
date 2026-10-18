//! Shared typed startup policy for the product entrypoints.

use ferrum_types::{
    EngineConfig, ExecutionResourceAuthority, FerrumConfigBuilder, FerrumError,
    HardwareCapabilities, ModelCapabilities, ResolvedFerrumConfig, Result, RuntimeConfigSnapshot,
    RuntimeConfigSource, StartupMemoryRequest, WorkloadProfile,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StartupUsage {
    SingleRequest,
    PersistentServing,
}

pub(crate) fn resolve_config(
    mut requested: RuntimeConfigSnapshot,
    model: ModelCapabilities,
    hardware: HardwareCapabilities,
    workload: WorkloadProfile,
    authority: ExecutionResourceAuthority,
    usage: StartupUsage,
) -> Result<ResolvedFerrumConfig> {
    if authority == ExecutionResourceAuthority::PlanRuntime
        && crate::runtime_env::runtime_snapshot_value(&requested, "FERRUM_PREFIX_CACHE").is_none()
    {
        // This requests optional native state retention within the existing
        // runtime memory budget. The compiled plan and selected providers
        // determine whether checkpoints are usable; unsupported plans keep
        // executing without a cache. Explicit values, including presets, win.
        requested.upsert(
            "FERRUM_PREFIX_CACHE",
            match usage {
                StartupUsage::SingleRequest => "0",
                StartupUsage::PersistentServing => "1",
            },
            RuntimeConfigSource::Default,
        );
    }
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
    #[cfg(feature = "cuda")]
    if let ferrum_types::Device::CUDA(ordinal) = device {
        if let Err(error) = apply_cuda_device_attributes(
            &mut hardware,
            ferrum_kernels::backend::probe_cuda_device_attributes(*ordinal),
        ) {
            tracing::warn!(
                device_ordinal = *ordinal,
                %error,
                "CUDA device attributes unavailable; hardware qualification remains unknown"
            );
        }
    }
    if let Some(request) = request {
        hardware.vram_bytes = Some(request.device.capacity_bytes);
    }
    hardware
}

#[cfg(any(feature = "cuda", test))]
fn apply_cuda_device_attributes(
    hardware: &mut HardwareCapabilities,
    observed: Result<ferrum_kernels::backend::CudaDeviceAttributes>,
) -> Result<()> {
    // A failed native query must not retain a shell observation or GPU-name
    // guess as authority for automatic execution choices.
    hardware.compute_capability = None;
    hardware.sm_count = None;
    let observed = observed?;
    hardware.compute_capability = Some(format!(
        "{}.{}",
        observed.compute_capability_major, observed.compute_capability_minor
    ));
    hardware.sm_count = Some(observed.multiprocessor_count);
    Ok(())
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

    #[test]
    fn native_cuda_attributes_replace_shell_values_without_changing_versions_or_memory() {
        let mut hardware = HardwareCapabilities {
            backend: "cuda".into(),
            cuda_runtime: Some("12.8".into()),
            compute_capability: Some("8.0".into()),
            sm_count: Some(1),
            vram_bytes: Some(16 * 1024 * 1024 * 1024),
            ..HardwareCapabilities::unknown()
        };
        let mut expected = hardware.clone();
        expected.compute_capability = Some("12.0".into());
        expected.sm_count = Some(32);

        apply_cuda_device_attributes(
            &mut hardware,
            Ok(ferrum_kernels::backend::CudaDeviceAttributes {
                compute_capability_major: 12,
                compute_capability_minor: 0,
                multiprocessor_count: 32,
            }),
        )
        .unwrap();

        assert_eq!(hardware, expected);
    }

    #[test]
    fn native_cuda_attribute_failure_clears_guessed_qualification() {
        let mut hardware = HardwareCapabilities {
            backend: "cuda".into(),
            cuda_runtime: Some("12.8".into()),
            compute_capability: Some("8.9".into()),
            sm_count: Some(128),
            vram_bytes: Some(16 * 1024 * 1024 * 1024),
            ..HardwareCapabilities::unknown()
        };
        let mut expected = hardware.clone();
        expected.compute_capability = None;
        expected.sm_count = None;

        assert!(apply_cuda_device_attributes(
            &mut hardware,
            Err(FerrumError::device("driver attribute query failed")),
        )
        .is_err());

        assert_eq!(hardware, expected);
    }

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
    fn retired_moe_options_fail_for_run_and_serve_startup() {
        for usage in [StartupUsage::SingleRequest, StartupUsage::PersistentServing] {
            for authority in [
                ExecutionResourceAuthority::LegacyEngine,
                ExecutionResourceAuthority::PlanRuntime,
            ] {
                for key in ["FERRUM_MOE_GRAPH", "FERRUM_MOE_BATCH_THRESHOLD"] {
                    let error = resolve_config(
                        RuntimeConfigSnapshot::from_env_vars([(key, "0")]),
                        model(),
                        hardware(),
                        WorkloadProfile::serving_default(),
                        authority,
                        usage,
                    )
                    .expect_err("removed controls must not disappear at either startup boundary");
                    assert!(error.to_string().contains(key), "{error}");
                    assert!(error.to_string().contains("retired"), "{error}");
                }
            }
        }
    }

    #[test]
    fn native_cuda_execution_defaults_are_shared_by_run_and_serve() {
        use ferrum_types::{
            CausalDecodeAttentionGrouping, CausalDecodeKvReuse, CompiledKernelFeatures,
            InvocationPreparationStrategy, SamplingReadbackMode, SequenceBackingGrowth,
        };
        let hardware = HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures {
            cuda: true,
            cuda_graph: true,
            vllm_paged_attn: true,
            cuda_indexed_causal_attention: true,
            cuda_causal_gqa2: true,
            ..CompiledKernelFeatures::default()
        });
        for usage in [StartupUsage::SingleRequest, StartupUsage::PersistentServing] {
            for explicit_uniform in [false, true] {
                let mut requested = RuntimeConfigSnapshot::default();
                if explicit_uniform {
                    requested.upsert(
                        "FERRUM_SAMPLING_READBACK_MODE",
                        "uniform",
                        RuntimeConfigSource::ConfigFile,
                    );
                }
                let resolved = resolve_config(
                    requested,
                    model(),
                    hardware.clone(),
                    WorkloadProfile::serving_default(),
                    ExecutionResourceAuthority::PlanRuntime,
                    usage,
                )
                .unwrap();
                let mut engine = EngineConfig::default();
                let numerical = engine.numerical_execution.clone();
                engine
                    .apply_runtime_config_snapshot(&resolved.runtime_config)
                    .unwrap();
                assert_eq!(
                    engine.runtime.invocation_preparation_strategy,
                    InvocationPreparationStrategy::ResidentDecodeProgram
                );
                assert_eq!(
                    engine.runtime.causal_decode_attention_grouping,
                    CausalDecodeAttentionGrouping::FixedExecutionCapacity
                );
                assert_eq!(
                    engine.runtime.causal_decode_kv_reuse,
                    CausalDecodeKvReuse::PairedQueryHeadsUnifiedV2
                );
                assert_eq!(
                    engine.runtime.sequence_backing_growth,
                    SequenceBackingGrowth::PagedGroup4GrowOnly
                );
                assert_eq!(
                    engine.runtime.sampling_readback_mode,
                    if explicit_uniform {
                        SamplingReadbackMode::Uniform
                    } else {
                        SamplingReadbackMode::Participant
                    }
                );
                assert_eq!(engine.numerical_execution, numerical);
                assert_prefix_state_cache(
                    &resolved,
                    usage == StartupUsage::PersistentServing,
                    RuntimeConfigSource::Default,
                );
            }
        }
    }

    fn assert_prefix_state_cache(
        resolved: &ResolvedFerrumConfig,
        enabled: bool,
        source: RuntimeConfigSource,
    ) {
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|entry| entry.key == "FERRUM_PREFIX_CACHE")
            .expect("resolved prefix-state request");
        assert_eq!(entry.effective_value, if enabled { "1" } else { "0" });
        assert_eq!(entry.source, source);
        let decision = resolved
            .decisions
            .iter()
            .find(|decision| decision.selection == "prefix_cache_policy")
            .expect("prefix-cache decision");
        assert_eq!(
            decision.selected,
            if enabled {
                "prefix_cache_enabled"
            } else {
                "prefix_cache_disabled"
            }
        );
        assert_eq!(decision.source_key.as_deref(), Some("FERRUM_PREFIX_CACHE"));
        let mut engine = EngineConfig::default();
        engine.runtime.prefix_state_cache_enabled = !enabled;
        engine
            .apply_runtime_config_snapshot(&resolved.runtime_config)
            .unwrap();
        assert_eq!(engine.runtime.prefix_state_cache_enabled, enabled);
        assert!(!engine.runtime.prefix_cache_enabled);
    }

    #[test]
    fn native_prefix_default_matches_product_usage_and_engine_config() {
        for (usage, enabled) in [
            (StartupUsage::SingleRequest, false),
            (StartupUsage::PersistentServing, true),
        ] {
            let resolved = resolve_config(
                RuntimeConfigSnapshot::default(),
                model(),
                HardwareCapabilities::unknown(),
                WorkloadProfile::serving_default(),
                ExecutionResourceAuthority::PlanRuntime,
                usage,
            )
            .unwrap();
            assert_prefix_state_cache(&resolved, enabled, RuntimeConfigSource::Default);
        }
    }

    #[test]
    fn native_prefix_defaults_preserve_explicit_values_and_preset_sources() {
        for source in [
            RuntimeConfigSource::ConfigFile,
            RuntimeConfigSource::Env,
            RuntimeConfigSource::Cli,
            RuntimeConfigSource::ScriptCase,
            RuntimeConfigSource::Default,
        ] {
            for (usage, enabled) in [
                (StartupUsage::SingleRequest, true),
                (StartupUsage::PersistentServing, false),
            ] {
                let requested = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
                    "FERRUM_PREFIX_CACHE",
                    if enabled { "1" } else { "0" },
                    source,
                )]);
                let resolved = resolve_config(
                    requested,
                    model(),
                    hardware(),
                    WorkloadProfile::serving_default(),
                    ExecutionResourceAuthority::PlanRuntime,
                    usage,
                )
                .unwrap();
                assert_prefix_state_cache(&resolved, enabled, source);
            }
        }
    }

    #[test]
    fn legacy_prefix_default_remains_disabled_for_both_product_usages() {
        for usage in [StartupUsage::SingleRequest, StartupUsage::PersistentServing] {
            let resolved = resolve_config(
                RuntimeConfigSnapshot::default(),
                model(),
                hardware(),
                WorkloadProfile::serving_default(),
                ExecutionResourceAuthority::LegacyEngine,
                usage,
            )
            .unwrap();
            assert!(crate::runtime_env::runtime_snapshot_value(
                &resolved.runtime_config,
                "FERRUM_PREFIX_CACHE",
            )
            .is_none());
            let mut engine = EngineConfig::default();
            engine
                .apply_runtime_config_snapshot(&resolved.runtime_config)
                .unwrap();
            assert!(!engine.runtime.prefix_state_cache_enabled);
            assert!(!engine.runtime.prefix_cache_enabled);
        }
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
                StartupUsage::PersistentServing,
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
            StartupUsage::PersistentServing,
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
            StartupUsage::PersistentServing,
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
            StartupUsage::PersistentServing,
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
                StartupUsage::PersistentServing,
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
