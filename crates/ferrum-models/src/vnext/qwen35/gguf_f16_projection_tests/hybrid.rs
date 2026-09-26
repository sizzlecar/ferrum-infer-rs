//! CPU source/schema/authority tests; these grant no CUDA or full-model quality.
use super::*;

const HYBRID: &str = F32_MASTER_GGUF_F16_ATTENTION_RESIDUAL2_FFN_M2TO8_NUMERICAL_PROFILE_ID;

fn config() -> Qwen35FamilyConfig {
    let mut config = native_mixed_config(256);
    let up = config
        .weights
        .iter_mut()
        .find(|w| w.role == "mlp_up")
        .unwrap();
    up.source_encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new("quantization.gguf.q4-k").unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: 144,
    });
    config
}

#[test]
fn hybrid_rn_attention_is_explicit_and_preserves_old_profiles_states_and_kv() {
    let config = config();
    let provider = Qwen35FamilyProvider::new().unwrap();
    provider.validate_typed_config(&config).unwrap();
    let catalog = provider.numerical_profiles(&config).unwrap();
    let id = NumericalProfileId::new(HYBRID).unwrap();
    assert!(!catalog.auto_preference().contains(&id));
    assert_eq!(
        catalog
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap()
            .iter()
            .map(|p| p.id.as_str())
            .collect::<Vec<_>>(),
        [F32_MASTER_NUMERICAL_PROFILE_ID]
    );
    assert!(catalog
        .candidates(
            &NumericalExecutionPolicy::Require(id.clone()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    let strict = prepare(&config, F32_MASTER_NUMERICAL_PROFILE_ID);
    let rounded = prepare(
        &config,
        F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID,
    );
    let hybrid = prepare(&config, HYBRID);
    assert_eq!(strict.weight_schema(), hybrid.weight_schema());
    assert_eq!(strict.program().states(), hybrid.program().states());
    let mut expected = rounded.numerical_profile().clone();
    expected.id = id;
    let ffn = expected
        .operations
        .iter_mut()
        .find(|op| op.operation_id.as_str() == DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID)
        .unwrap();
    ffn.operation_id = operation_id(DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_OPERATION_ID).unwrap();
    ffn.multiplication_type = Some(ElementType::I8);
    expected
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(&expected, hybrid.numerical_profile());
    for (before, after) in rounded
        .program()
        .blocks()
        .iter()
        .zip(hybrid.program().blocks())
    {
        let mut expected = before.clone();
        for node in &mut expected.nodes {
            if node.operation_id.as_str() == DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID {
                node.operation_id =
                    operation_id(DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_OPERATION_ID).unwrap();
            }
        }
        assert_eq!(
            &expected, after,
            "only FFN changes relative to the all-RN program"
        );
    }
    // Removing the eligible pair removes only this candidate, never old policy
    // bytes or Auto behavior.
    let without_pair = native_config(256);
    let old_catalog = provider.numerical_profiles(&without_pair).unwrap();
    assert!(old_catalog
        .resolve(&NumericalProfileId::new(HYBRID).unwrap())
        .is_err());
    assert_eq!(old_catalog.auto_preference(), catalog.auto_preference());
    for old in [
        F16_NUMERICAL_PROFILE_ID,
        F32_MASTER_NUMERICAL_PROFILE_ID,
        F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID,
        F16_INT8_KV_NUMERICAL_PROFILE_ID,
        F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID,
    ] {
        let old = NumericalProfileId::new(old).unwrap();
        assert_eq!(
            serde_json::to_vec(old_catalog.resolve(&old).unwrap()).unwrap(),
            serde_json::to_vec(catalog.resolve(&old).unwrap()).unwrap()
        );
    }
}

#[test]
fn hybrid_rn_attention_rejects_unqualified_source_or_forged_ffn_and_state_contracts() {
    let config = config();
    let provider = Qwen35FamilyProvider::new().unwrap();
    let explicit = provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&NumericalProfileId::new(HYBRID).unwrap())
        .unwrap()
        .clone();
    let mut no_pair = config.clone();
    no_pair
        .weights
        .iter_mut()
        .find(|w| w.role == "mlp_up")
        .unwrap()
        .source_encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new("quantization.gguf.q5-k").unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: 176,
    });
    let mut wrong_abi = config.clone();
    wrong_abi.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
    for rejected in [
        no_pair,
        wrong_abi,
        tests::test_config(),
        tests::test_moe_gguf_config(),
    ] {
        assert!(provider.semantic_program(&rejected, &explicit).is_err());
    }
    for mode in 0..2 {
        let mut forged = explicit.clone();
        if mode == 0 {
            forged
                .operations
                .iter_mut()
                .find(|op| {
                    op.operation_id.as_str() == DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_OPERATION_ID
                })
                .unwrap()
                .multiplication_type = Some(ElementType::F16);
        } else {
            forged
                .boundaries
                .insert(value_id("value.output.logits").unwrap(), ElementType::F16);
        }
        assert!(provider.semantic_program(&config, &forged).is_err());
    }
}

#[test]
fn hybrid_rn_attention_live_registry_retains_compressed_ffn_without_dual_residency() {
    let family = prepare(&config(), HYBRID);
    let rounded = prepare(
        &config(),
        F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID,
    );
    let (_, registry, catalog) = registry_catalog();
    let selection = gguf_f16_projection_materializer_selection(&family).unwrap();
    assert!(matches!(
        registry.select(
            &family,
            &catalog,
            &WeightMaterializerSelection::exact(selection.materializer_id().clone())
        ),
        Err(VNextError::WeightMaterializerQualityApprovalRequired { .. })
    ));
    let trusted = registry.select(&family, &catalog, &selection).unwrap();
    let schema = trusted.plan().schema();
    let inventory = gguf_f16_projection_inventory(&family).unwrap();
    let full_inventory = gguf_f16_projection_inventory(&rounded).unwrap();
    assert_eq!(
        inventory.source_consumed_bytes,
        full_inventory.source_consumed_bytes
    );
    assert!(inventory.converted_f16_bytes > 0);
    assert!(inventory.converted_f16_bytes < full_inventory.converted_f16_bytes);
    assert_eq!(
        inventory.unique_consumed_execution_bytes,
        inventory.converted_f16_bytes + inventory.retained_consumed_bytes
    );
    assert!(
        inventory.unique_consumed_execution_bytes < full_inventory.unique_consumed_execution_bytes
    );
    let mut retained_ffn = 0;
    let mut compressed_formats = BTreeSet::new();
    let mut attention_roles = BTreeSet::new();
    let mut expected_components = BTreeSet::new();
    for weight in &inventory.weights {
        expected_components.extend(weight.execution_components.iter().cloned());
        let source = family
            .weight_schema()
            .tensor(&weight.logical_weight)
            .unwrap();
        let execution = schema.tensor(&weight.logical_weight).unwrap();
        let ffn = weight
            .consumers
            .iter()
            .any(|c| c.operation.as_str() == DENSE_SWIGLU_Q8_RESIDUAL2_FFN_M2_TO8_OPERATION_ID);
        if ffn {
            retained_ffn += 1;
            assert!(weight.consumers.iter().all(|c| c.role.is_none()));
            assert_eq!(weight.source_components, weight.execution_components);
            assert_eq!(source, execution);
            let components = schema
                .physical_component_refs(&weight.logical_weight)
                .unwrap();
            for component in components {
                if let WeightEncoding::BlockQuantized(spec) = &component.encoding {
                    compressed_formats.insert(spec.format_id.as_str().to_owned());
                }
            }
        } else if weight.consumers.iter().any(|c| c.role.is_some()) {
            assert!(weight.consumers.iter().all(|c| c.role.is_some()));
            attention_roles.extend(
                weight
                    .consumers
                    .iter()
                    .map(|c| c.operation.as_str().to_owned()),
            );
            for component in schema
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
                source, execution,
                "head/embedding/non-projection storage stays original"
            );
        }
    }
    assert!(retained_ffn > 0);
    assert!(compressed_formats.contains("quantization.gguf.q4-k"));
    assert!(compressed_formats.contains("quantization.gguf.q6-k"));
    assert_eq!(
        attention_roles,
        BTreeSet::from([
            GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID.to_owned(),
            CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID.to_owned(),
        ])
    );
    assert_eq!(
        schema
            .components
            .iter()
            .map(|c| c.id.clone())
            .collect::<BTreeSet<_>>(),
        expected_components,
        "no unused second weight view remains resident"
    );
}

#[test]
fn hybrid_rn_attention_trusted_materialization_cannot_cross_full_rn_or_strict_profile() {
    let config = config();
    let hybrid = prepare(&config, HYBRID);
    let rounded = prepare(
        &config,
        F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID,
    );
    let strict = prepare(&config, F32_MASTER_NUMERICAL_PROFILE_ID);
    let (cpu, registry, catalog) = registry_catalog();
    let policy = policy(&catalog);
    let compilation = ProgramPlanCompiler::compile(
        &strict,
        &catalog,
        &policy,
        &cpu.registry().planning(),
        &compile_options(&config),
    )
    .unwrap();
    let selected = |family: &PreparedModelFamily| {
        let selection = gguf_f16_projection_materializer_selection(family).unwrap();
        registry.select(family, &catalog, &selection).unwrap()
    };
    let hybrid_plan = selected(&hybrid);
    let rounded_plan = selected(&rounded);
    assert_ne!(hybrid_plan.plan().schema(), rounded_plan.plan().schema());
    let bind = |family: &PreparedModelFamily, plan| {
        PlanBuildRequest::new(
            family,
            &catalog,
            &policy,
            compilation.node_resolutions().to_vec(),
        )
        .unwrap()
        .with_execution_weights(plan)
        .map(|_| ())
    };
    assert!(bind(&hybrid, hybrid_plan.clone()).is_ok());
    assert!(bind(&rounded, rounded_plan.clone()).is_ok());
    assert!(bind(&hybrid, rounded_plan).is_err());
    assert!(bind(&rounded, hybrid_plan.clone()).is_err());
    assert!(bind(&strict, hybrid_plan).is_err());
}
