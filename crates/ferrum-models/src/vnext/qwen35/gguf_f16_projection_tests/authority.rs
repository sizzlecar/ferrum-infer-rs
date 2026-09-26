//! Real Qwen prepared-family and live materializer registry. The CPU catalog
//! exercises cold conversion authority, not execution of the CUDA-only aliases.
use super::*;
use ferrum_interfaces::vnext::*;
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_kernels::gguf_f16_projection_materializer::*;

fn native_mixed_config(intermediate: u64) -> Qwen35FamilyConfig {
    let mut config = native_config(intermediate);
    for (role, format, values, bytes) in [
        ("mlp_gate", "quantization.gguf.q4-k", 256, 144),
        ("mlp_up", "quantization.gguf.q5-k", 256, 176),
        ("mlp_down", "quantization.gguf.q6-k", 256, 210),
        ("linear_attn_qkv", "quantization.gguf.q8-0", 32, 34),
    ] {
        let weight = config.weights.iter_mut().find(|w| w.role == role).unwrap();
        weight.source_encoding =
            FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: QuantizationFormatId::new(format).unwrap(),
                logical_values_per_block: values,
                bytes_per_block: bytes,
            });
    }
    config
}

fn prepare(config: &Qwen35FamilyConfig, profile: &str) -> PreparedModelFamily {
    TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(config).unwrap(),
            &NumericalProfileId::new(profile).unwrap(),
        )
        .unwrap()
}

fn registry_catalog() -> (
    CpuVNextComposition,
    WeightMaterializerRegistry,
    CapabilityCatalog,
) {
    let cpu = CpuVNextComposition::create(
        DeviceId::new("device.test.gguf-f16-authority").unwrap(),
        128 << 20,
    )
    .unwrap();
    let original = cpu.catalog();
    let mut device = original.device().clone();
    // Only the cold CPU materializer capability is added to this test catalog.
    // No CUDA alias provider, GPU implementation or execution claim is added.
    device
        .capabilities
        .insert(CapabilityId::new(GGUF_F16_PROJECTION_CAPABILITY_ID).unwrap());
    let catalog = CapabilityCatalog::new(
        device,
        original.operations().values().cloned().collect(),
        original.providers().clone(),
        original.engine_providers().values().cloned().collect(),
    )
    .unwrap();
    let registry =
        WeightMaterializerRegistry::new(vec![gguf_f16_projection_materializer().unwrap()]).unwrap();
    let catalog = registry.augment_catalog(catalog).unwrap();
    (cpu, registry, catalog)
}

#[test]
fn gguf_f16_live_registry_authorizes_only_explicit_profile_and_approximate_artifact() {
    let config = native_mixed_config(256);
    let family = prepare(
        &config,
        F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID,
    );
    let strict = prepare(&config, F32_MASTER_NUMERICAL_PROFILE_ID);
    assert_eq!(family.weight_schema(), strict.weight_schema());
    assert!(!requests_gguf_f16_projection_materialization(&strict));
    assert!(requests_gguf_f16_projection_materialization(&family));
    let (_, registry, catalog) = registry_catalog();
    let selection = gguf_f16_projection_materializer_selection(&family).unwrap();
    assert!(selection.has_numeric_quality_artifact());
    assert_eq!(
        catalog
            .weight_materializer(selection.materializer_id())
            .unwrap()
            .fidelity(),
        WeightMaterializationFidelity::Approximate
    );
    assert!(matches!(
        registry.select(
            &family,
            &catalog,
            &WeightMaterializerSelection::exact(selection.materializer_id().clone())
        ),
        Err(VNextError::WeightMaterializerQualityApprovalRequired { .. })
    ));
    assert!(gguf_f16_projection_materializer_selection(&strict).is_err());
    assert!(registry.select(&strict, &catalog, &selection).is_err());
    let trusted = registry.select(&family, &catalog, &selection).unwrap();
    let plan = trusted.plan();
    let approval = plan.approximate_quality_approval().unwrap();
    assert_eq!(
        approval.source_schema_fingerprint(),
        family.weight_schema().fingerprint().unwrap()
    );
    assert_eq!(
        approval.execution_schema_fingerprint(),
        plan.schema().fingerprint().unwrap()
    );
    assert_eq!(
        plan.source_schema_fingerprint(),
        approval.source_schema_fingerprint()
    );
    let inventory = gguf_f16_projection_inventory(&family).unwrap();
    assert_eq!(
        inventory.source_schema_fingerprint,
        approval.source_schema_fingerprint()
    );
    assert_eq!(
        inventory.execution_schema_fingerprint,
        approval.execution_schema_fingerprint()
    );
    let mut operations = BTreeSet::new();
    for weight in inventory.weights {
        let source = family
            .weight_schema()
            .tensor(&weight.logical_weight)
            .unwrap();
        let execution = plan.schema().tensor(&weight.logical_weight).unwrap();
        if weight
            .consumers
            .iter()
            .any(|consumer| consumer.role.is_some())
        {
            operations.extend(
                weight
                    .consumers
                    .iter()
                    .map(|c| c.operation.as_str().to_owned()),
            );
            assert!(weight.consumers.iter().all(|c| c.role.is_some()));
            assert!(matches!(
                execution.physical_layout,
                PhysicalWeightLayout::Dense { .. }
            ));
            assert_eq!(execution.dimensions, source.dimensions);
            for component in plan
                .schema()
                .physical_component_refs(&weight.logical_weight)
                .unwrap()
            {
                assert_eq!(
                    component.encoding,
                    WeightEncoding::Dense {
                        element_type: ElementType::F16
                    }
                );
            }
        } else {
            assert_eq!(
                execution, source,
                "head, embedding and non-projection storage are unchanged"
            );
        }
    }
    assert_eq!(
        operations,
        BTreeSet::from([
            DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID.to_owned(),
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID.to_owned(),
            CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID.to_owned(),
        ])
    );
}

fn policy(catalog: &CapabilityCatalog) -> ResolvedRuntimePolicy {
    ResolvedRuntimePolicy::new(
        "runtime-policy.test.gguf-f16-authority",
        ContractVersion::new(1, 0),
        SchedulingDiscipline::FirstReady,
        RuntimeMemoryPolicy {
            capacity_bytes: 128 << 20,
            reserve_bytes: 1 << 20,
            maximum_active_sequences: 1,
            dynamic_storage_profile_order: catalog
                .device()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
            checkpoint_capacity: None,
        },
        AdmissionPolicy {
            maximum_queue_depth: 1,
            maximum_scheduled_tokens: 4,
            sequence_fit_policy: AdmissionFitPolicy::ImmediateOnly,
            allow_defer: false,
            cancellation_check_interval_steps: 1,
        },
        ferrum_types::AttentionExecutionPolicy::Portable,
        ExecutionDeterminismRequirement::BitwiseSameRuntime,
        None,
    )
    .unwrap()
}

fn compile_options(config: &Qwen35FamilyConfig) -> ProgramPlanCompileOptions {
    let inputs = [
        ("value.input.token_ids", ElementType::U32, vec![4]),
        (
            "value.input.greedy_token_mask",
            ElementType::U8,
            vec![config.vocab_size],
        ),
        (
            "value.input.greedy_repetition_token_ids",
            ElementType::U32,
            vec![4],
        ),
        (
            "value.input.greedy_repetition_offsets",
            ElementType::U32,
            vec![2],
        ),
        (
            "value.input.greedy_repetition_penalty",
            ElementType::F32,
            vec![1],
        ),
    ]
    .into_iter()
    .map(|(id, element_type, dimensions)| {
        (
            ProgramValueId::new(id).unwrap(),
            ProgramTensorSpec {
                dimensions,
                element_type,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )
    })
    .collect();
    ProgramPlanCompileOptions::new(inputs).unwrap()
}

#[test]
fn gguf_f16_trusted_plan_cannot_be_reused_for_strict_consumers_or_changed_source() {
    let config = native_mixed_config(256);
    let family = prepare(
        &config,
        F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID,
    );
    let strict = prepare(&config, F32_MASTER_NUMERICAL_PROFILE_ID);
    let (cpu, registry, catalog) = registry_catalog();
    let policy = policy(&catalog);
    let strict_compilation = ProgramPlanCompiler::compile(
        &strict,
        &catalog,
        &policy,
        &cpu.registry().planning(),
        &compile_options(&config),
    )
    .unwrap();
    let selection = gguf_f16_projection_materializer_selection(&family).unwrap();
    let trusted = registry.select(&family, &catalog, &selection).unwrap();
    let binding = |family: &PreparedModelFamily, plan| {
        PlanBuildRequest::new(
            family,
            &catalog,
            &policy,
            strict_compilation.node_resolutions().to_vec(),
        )
        .unwrap()
        .with_execution_weights(plan)
        .map(|_| ())
    };
    // This is only the public weight-authority boundary, not a completed RN
    // model compile. The resolutions are real CPU strict-program resolutions.
    assert!(binding(&family, trusted.clone()).is_ok());
    assert!(binding(&strict, trusted.clone()).is_err(),
        "the same source schema must not let RN-F16 authority escape its typed projection consumers");
    let changed = native_mixed_config(512);
    let changed = prepare(
        &changed,
        F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID,
    );
    assert_ne!(family.weight_schema(), changed.weight_schema());
    assert!(binding(&changed, trusted.clone()).is_err());
    // The conversion artifact can qualify the same math on another supported
    // schema; a fresh live selection must bind a distinct source/plan/approval.
    let fresh = registry.select(&changed, &catalog, &selection).unwrap();
    assert_ne!(
        fresh.plan().source_schema_fingerprint(),
        trusted.plan().source_schema_fingerprint()
    );
    assert_ne!(
        fresh.plan().fingerprint().unwrap(),
        trusted.plan().fingerprint().unwrap()
    );
    assert_eq!(
        fresh
            .plan()
            .approximate_quality_approval()
            .unwrap()
            .source_schema_fingerprint(),
        changed.weight_schema().fingerprint().unwrap()
    );
    assert!(binding(&changed, fresh).is_ok());
}

#[path = "hybrid.rs"]
mod hybrid;
