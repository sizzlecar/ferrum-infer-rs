use super::*;

fn cuda() -> HardwareCapabilities {
    HardwareCapabilities::rtx4090_cuda(CompiledKernelFeatures {
        cuda: true,
        cuda_graph: true,
        vllm_paged_attn: true,
        cuda_indexed_causal_attention: true,
        cuda_causal_gqa2: true,
        cuda_q4_shared_weight_decode: true,
        cuda_q5_shared_weight_decode: true,
        ..CompiledKernelFeatures::default()
    })
}

fn resolve(hardware: HardwareCapabilities, entries: &[(&str, &str)]) -> ResolvedFerrumConfig {
    builder(hardware, entries).resolve().unwrap()
}

fn builder(hardware: HardwareCapabilities, entries: &[(&str, &str)]) -> FerrumConfigBuilder {
    FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries(entries.iter().map(
        |(key, value)| RuntimeConfigEntry::new(*key, *value, RuntimeConfigSource::Cli),
    )))
    .with_hardware_capabilities(hardware)
    .with_execution_resource_authority(ExecutionResourceAuthority::PlanRuntime)
}

fn engine(resolved: &ResolvedFerrumConfig) -> crate::EngineConfig {
    let mut engine = crate::EngineConfig::default();
    engine
        .apply_runtime_config_snapshot(&resolved.runtime_config)
        .unwrap();
    engine
}

#[test]
fn cuda_execution_defaults_materialize_the_validated_bundle() {
    let resolved = resolve(cuda(), &[]);
    let runtime = engine(&resolved).runtime;
    assert_eq!(
        runtime.invocation_preparation_strategy,
        InvocationPreparationStrategy::ResidentDecodeProgram
    );
    assert_eq!(
        runtime.program_binding_upload_strategy,
        ProgramBindingUploadStrategy::CompactScatter
    );
    assert_eq!(
        runtime.segment_binding_owner_view_mode,
        SegmentBindingOwnerViewMode::Indexed
    );
    assert_eq!(
        runtime.causal_decode_preparation_mode,
        CausalDecodePreparationMode::Packed
    );
    assert_eq!(
        runtime.causal_decode_attention_grouping,
        CausalDecodeAttentionGrouping::FixedExecutionCapacity
    );
    assert_eq!(
        runtime.causal_decode_kv_reuse,
        CausalDecodeKvReuse::PairedQueryHeadsUnifiedV2
    );
    assert_eq!(
        runtime.sequence_backing_growth,
        SequenceBackingGrowth::PagedGroup4GrowOnly
    );
    assert_eq!(
        runtime.completion_readback_policy,
        CompletionReadbackPolicy::BatchedU32
    );
    assert_eq!(
        runtime.sampling_readback_mode,
        SamplingReadbackMode::Participant
    );
    assert_eq!(runtime.q4_mmvq_weight_decode, Q4MmvqWeightDecode::Baseline);
    assert_eq!(runtime.q5_mmvq_weight_decode, Q5MmvqWeightDecode::Baseline);
    assert_eq!(
        runtime.q8_projection_weight_preparation,
        Q8ProjectionWeightPreparation::Compressed
    );
    assert!(resolved
        .runtime_config
        .entries
        .iter()
        .all(|entry| entry.key != "FERRUM_ACTIVE_DECODE_PREFILL_CHUNK"));
    for (selection, key) in [
        (
            "invocation_preparation_strategy",
            "FERRUM_INVOCATION_PREPARATION_STRATEGY",
        ),
        (
            "program_binding_upload_strategy",
            "FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY",
        ),
        (
            "segment_binding_owner_view_mode",
            crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY,
        ),
        (
            "causal_decode_preparation_mode",
            crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
        ),
        (
            "causal_decode_attention_grouping",
            crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY,
        ),
        (
            "causal_decode_kv_reuse",
            crate::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY,
        ),
        (
            "sequence_backing_growth",
            crate::SEQUENCE_BACKING_GROWTH_CONFIG_KEY,
        ),
        ("sampling_readback_mode", "FERRUM_SAMPLING_READBACK_MODE"),
        (
            "completion_readback_policy",
            crate::COMPLETION_READBACK_POLICY_CONFIG_KEY,
        ),
    ] {
        let decision = resolved
            .decisions
            .iter()
            .find(|d| d.selection == selection)
            .unwrap();
        let entry = resolved
            .runtime_config
            .entries
            .iter()
            .find(|e| e.key == key)
            .unwrap();
        assert_eq!(decision.selected, entry.effective_value);
        assert_eq!(decision.source, AutoConfigSource::Default);
        assert_eq!(decision.source_key, None);
    }
}

#[test]
fn cuda_execution_defaults_preserve_explicit_opt_outs_and_provenance() {
    let entries = [
        ("FERRUM_INVOCATION_PREPARATION_STRATEGY", "full"),
        ("FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY", "sparse"),
        (crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY, "legacy"),
        (
            crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
            "per-participant",
        ),
        (
            crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY,
            "contiguous",
        ),
        (crate::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY, "none"),
        (crate::SEQUENCE_BACKING_GROWTH_CONFIG_KEY, "exact"),
        ("FERRUM_SAMPLING_READBACK_MODE", "uniform"),
        (crate::COMPLETION_READBACK_POLICY_CONFIG_KEY, "synchronous"),
    ];
    for source in [
        RuntimeConfigSource::Cli,
        RuntimeConfigSource::ConfigFile,
        RuntimeConfigSource::ScriptCase,
        RuntimeConfigSource::Default,
    ] {
        let requested = RuntimeConfigSnapshot::from_entries(
            entries
                .iter()
                .map(|(key, value)| RuntimeConfigEntry::new(*key, *value, source)),
        );
        let resolved = FerrumConfigBuilder::new(requested)
            .with_hardware_capabilities(cuda())
            .with_execution_resource_authority(ExecutionResourceAuthority::PlanRuntime)
            .resolve()
            .unwrap();
        let runtime = engine(&resolved).runtime;
        assert_eq!(
            runtime.invocation_preparation_strategy,
            InvocationPreparationStrategy::Full
        );
        assert_eq!(
            runtime.sequence_backing_growth,
            SequenceBackingGrowth::Exact
        );
        for (key, value) in entries {
            let entry = resolved
                .runtime_config
                .entries
                .iter()
                .find(|e| e.key == key)
                .unwrap();
            assert_eq!(entry.effective_value, value);
            assert_eq!(entry.source, source);
            let decision = resolved
                .decisions
                .iter()
                .find(|d| d.source_key.as_deref() == Some(key))
                .unwrap();
            assert_eq!(decision.selected, value);
            assert_eq!(decision.source, auto_config_source_from_runtime(source));
        }
    }
}

#[test]
fn cuda_execution_defaults_respect_dependency_opt_outs() {
    for (key, value) in [
        ("FERRUM_INVOCATION_PREPARATION_STRATEGY", "full"),
        ("FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY", "sparse"),
        (crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY, "legacy"),
        ("FERRUM_REUSABLE_EXECUTION", "0"),
    ] {
        let runtime = engine(&resolve(cuda(), &[(key, value)])).runtime;
        assert_eq!(
            runtime.invocation_preparation_strategy,
            InvocationPreparationStrategy::Full
        );
        assert_eq!(
            runtime.segment_binding_owner_view_mode,
            SegmentBindingOwnerViewMode::Legacy
        );
        assert_eq!(
            runtime.sequence_backing_growth,
            SequenceBackingGrowth::Exact
        );
    }
    let runtime = engine(&resolve(
        cuda(),
        &[(
            crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
            "per-participant",
        )],
    ))
    .runtime;
    assert_eq!(
        runtime.causal_decode_attention_grouping,
        CausalDecodeAttentionGrouping::Contiguous
    );
    assert_eq!(runtime.causal_decode_kv_reuse, CausalDecodeKvReuse::None);
    let runtime = engine(&resolve(
        cuda(),
        &[(
            crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY,
            "contiguous",
        )],
    ))
    .runtime;
    assert_eq!(
        runtime.causal_decode_preparation_mode,
        CausalDecodePreparationMode::Packed
    );
    assert_eq!(runtime.causal_decode_kv_reuse, CausalDecodeKvReuse::None);
    for grouping in ["indexed", "fixed-capacity", "fixed-execution-capacity"] {
        let runtime = engine(&resolve(
            cuda(),
            &[(crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY, grouping)],
        ))
        .runtime;
        assert_eq!(
            runtime.causal_decode_preparation_mode,
            CausalDecodePreparationMode::Packed
        );
        assert_eq!(
            runtime.causal_decode_attention_grouping.as_runtime_value(),
            grouping
        );
    }
}

#[test]
fn cuda_execution_defaults_fall_back_when_compiled_capabilities_are_missing() {
    let mut hardware = cuda();
    hardware.compiled_features.cuda = false;
    let runtime = engine(&resolve(hardware, &[])).runtime;
    assert_eq!(
        runtime.invocation_preparation_strategy,
        InvocationPreparationStrategy::Full
    );
    assert_eq!(
        runtime.sampling_readback_mode,
        SamplingReadbackMode::Uniform
    );
    assert_eq!(
        runtime.causal_decode_preparation_mode,
        CausalDecodePreparationMode::PerParticipant
    );
    let mut hardware = cuda();
    hardware.compiled_features.cuda_graph = false;
    assert_eq!(
        engine(&resolve(hardware, &[]))
            .runtime
            .invocation_preparation_strategy,
        InvocationPreparationStrategy::Full
    );
    let mut hardware = cuda();
    hardware.compiled_features.vllm_paged_attn = false;
    let runtime = engine(&resolve(hardware, &[])).runtime;
    assert_eq!(
        runtime.causal_decode_preparation_mode,
        CausalDecodePreparationMode::PerParticipant
    );
    assert_eq!(
        runtime.causal_decode_attention_grouping,
        CausalDecodeAttentionGrouping::Contiguous
    );
    let mut hardware = cuda();
    hardware.compiled_features.cuda_indexed_causal_attention = false;
    let runtime = engine(&resolve(hardware, &[])).runtime;
    assert_eq!(
        runtime.causal_decode_preparation_mode,
        CausalDecodePreparationMode::Packed
    );
    assert_eq!(
        runtime.causal_decode_attention_grouping,
        CausalDecodeAttentionGrouping::Contiguous
    );
    assert_eq!(runtime.causal_decode_kv_reuse, CausalDecodeKvReuse::None);
    let mut hardware = cuda();
    hardware.compiled_features.cuda_causal_gqa2 = false;
    let runtime = engine(&resolve(hardware, &[])).runtime;
    assert_eq!(
        runtime.causal_decode_attention_grouping,
        CausalDecodeAttentionGrouping::FixedExecutionCapacity
    );
    assert_eq!(runtime.causal_decode_kv_reuse, CausalDecodeKvReuse::None);
    for entries in [
        vec![("FERRUM_ATTENTION_POLICY", "portable")],
        vec![("FERRUM_KV_DTYPE", "int8")],
    ] {
        let runtime = engine(&resolve(cuda(), &entries)).runtime;
        assert_eq!(
            runtime.causal_decode_preparation_mode,
            CausalDecodePreparationMode::PerParticipant
        );
        assert_eq!(
            runtime.causal_decode_attention_grouping,
            CausalDecodeAttentionGrouping::Contiguous
        );
        assert_eq!(runtime.causal_decode_kv_reuse, CausalDecodeKvReuse::None);
    }
}

#[test]
fn cuda_execution_defaults_do_not_change_other_backends_or_legacy() {
    for backend in ["cpu", "metal", "cuda"] {
        let mut hardware = cuda();
        hardware.backend = backend.into();
        let authority = if backend == "cuda" {
            ExecutionResourceAuthority::LegacyEngine
        } else {
            ExecutionResourceAuthority::PlanRuntime
        };
        let resolved = builder(hardware, &[])
            .with_execution_resource_authority(authority)
            .resolve()
            .unwrap();
        let runtime = engine(&resolved).runtime;
        assert_eq!(
            runtime.invocation_preparation_strategy,
            InvocationPreparationStrategy::Full
        );
        assert_eq!(
            runtime.program_binding_upload_strategy,
            ProgramBindingUploadStrategy::Sparse
        );
        assert_eq!(
            runtime.completion_readback_policy,
            CompletionReadbackPolicy::Synchronous
        );
        assert_eq!(
            runtime.causal_decode_attention_grouping,
            CausalDecodeAttentionGrouping::Contiguous
        );
        assert_eq!(
            runtime.sequence_backing_growth,
            SequenceBackingGrowth::Exact
        );
        assert_eq!(
            runtime.sampling_readback_mode,
            SamplingReadbackMode::Uniform
        );
    }
}

#[test]
fn cuda_execution_defaults_do_not_hide_explicit_incompatibilities() {
    for entries in [
        vec![
            (
                "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                "resident-decode-program",
            ),
            (crate::SEGMENT_BINDING_OWNER_VIEW_MODE_CONFIG_KEY, "legacy"),
        ],
        vec![
            (
                "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                "resident-decode-program",
            ),
            ("FERRUM_PROGRAM_BINDING_UPLOAD_STRATEGY", "sparse"),
        ],
        vec![
            (
                crate::CAUSAL_DECODE_PREPARATION_MODE_CONFIG_KEY,
                "per-participant",
            ),
            (
                crate::CAUSAL_DECODE_ATTENTION_GROUPING_CONFIG_KEY,
                "fixed-execution-capacity",
            ),
        ],
    ] {
        assert!(matches!(
            builder(cuda(), &entries).resolve(),
            Err(AutoConfigError::InvalidOverride { .. })
        ));
    }
    let mut hardware = cuda();
    hardware.compiled_features.cuda_causal_gqa2 = false;
    assert!(matches!(
        builder(
            hardware,
            &[(
                crate::CAUSAL_DECODE_KV_REUSE_CONFIG_KEY,
                "paired-query-heads-unified-v2"
            )]
        )
        .resolve(),
        Err(AutoConfigError::InvalidOverride { .. })
    ));
    let requested = RuntimeConfigSnapshot::from_entries([RuntimeConfigEntry::new(
        crate::SEQUENCE_BACKING_GROWTH_CONFIG_KEY,
        "paged-group4-grow-only",
        RuntimeConfigSource::Env,
    )]);
    assert!(matches!(
        FerrumConfigBuilder::new(requested)
            .with_hardware_capabilities(cuda())
            .with_execution_resource_authority(ExecutionResourceAuthority::PlanRuntime)
            .resolve(),
        Err(AutoConfigError::InvalidOverride { .. })
    ));
}
