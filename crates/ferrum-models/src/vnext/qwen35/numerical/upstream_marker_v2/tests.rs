use super::*;
use ferrum_interfaces::vnext::{
    NumericalArithmeticStage, NumericalExecutionPolicy, ProjectionBlockFormat,
    TypedFamilyRegistration, UpstreamProjectionArithmetic,
};

fn source() -> Qwen35FamilyConfig {
    super::super::q8act_attention_tests::source()
}
fn candidate_id() -> NumericalProfileId {
    NumericalProfileId::new(
        F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_NUMERICAL_PROFILE_ID,
    )
    .unwrap()
}

#[test]
fn upstream_prefill_require_lowers_new_ids_without_changing_auto_or_old_profiles() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let config = source();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let id = NumericalProfileId::new(
        F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_UPSTREAM_MARKER_V2_PREFILL_NUMERICAL_PROFILE_ID,
    )
    .unwrap();
    let profiles = definition.numerical_profiles();
    let selected = profiles
        .candidates(
            &NumericalExecutionPolicy::Require(id.clone()),
            KvStorageFormat::F16,
        )
        .unwrap()[0];
    assert!(profiles
        .candidates(
            &NumericalExecutionPolicy::Require(id.clone()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    let strict = profiles
        .resolve(&NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap())
        .unwrap();
    assert_eq!(
        profiles
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap(),
        [strict]
    );
    let prepared = registration.prepare(&definition, &id).unwrap();
    selected.validate_program(prepared.program()).unwrap();
    let prior = registration.prepare(&definition, &candidate_id()).unwrap();
    assert_eq!(prepared.program().states(), prior.program().states());
    assert_eq!(prepared.weight_schema(), prior.weight_schema());
    for kind in [
        UpstreamMarkerV2Profile::SwiGluPrefill,
        UpstreamMarkerV2Profile::GatedDeltaPrefill,
        UpstreamMarkerV2Profile::CausalPrefill,
    ] {
        assert!(prepared
            .program()
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .any(|n| n.operation_id.as_str() == kind.operation_id()));
    }
    let mut tampered = selected.clone();
    tampered
        .operations
        .iter_mut()
        .find_map(|o| o.composite_arithmetic.as_mut())
        .unwrap()
        .projections
        .pop();
    assert!(Qwen35FamilyProvider::new()
        .unwrap()
        .semantic_program(&config, &tampered)
        .is_err());
}

#[test]
fn upstream_marker_v2_require_lowers_real_trio_and_preserves_auto_state_and_head() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let config = source();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let catalog = definition.numerical_profiles();
    let strict_id = NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap();
    let strict = catalog.resolve(&strict_id).unwrap();
    let selected = catalog
        .candidates(
            &NumericalExecutionPolicy::Require(candidate_id()),
            KvStorageFormat::F16,
        )
        .unwrap()[0];
    assert_eq!(
        catalog
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap(),
        [strict]
    );
    assert!(catalog
        .candidates(
            &NumericalExecutionPolicy::Require(candidate_id()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    let mut restored = selected.clone();
    restored.id = strict.id.clone();
    for kind in [
        UpstreamMarkerV2Profile::SwiGlu,
        UpstreamMarkerV2Profile::GatedDelta,
        UpstreamMarkerV2Profile::Causal,
    ] {
        let op = restored
            .operations
            .iter_mut()
            .find(|op| op.operation_id.as_str() == kind.operation_id())
            .unwrap();
        assert_eq!(op.composite_arithmetic, Some(kind.arithmetic()));
        *op = strict
            .operations
            .iter()
            .find(|op| op.operation_id.as_str() == kind.strict_operation_id())
            .unwrap()
            .clone();
    }
    restored
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(
        &restored, strict,
        "outside declared projections state/boundary/head semantics stay strict"
    );
    let prepared = registration.prepare(&definition, &candidate_id()).unwrap();
    let baseline = registration.prepare(&definition, &strict_id).unwrap();
    assert_eq!(prepared.weight_schema(), baseline.weight_schema());
    assert_eq!(prepared.program().states(), baseline.program().states());
    selected.validate_program(prepared.program()).unwrap();
    for kind in [
        UpstreamMarkerV2Profile::SwiGlu,
        UpstreamMarkerV2Profile::GatedDelta,
        UpstreamMarkerV2Profile::Causal,
    ] {
        assert!(prepared
            .program()
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .any(|n| n.operation_id.as_str() == kind.operation_id()));
    }
    let mut missing_kv = selected.clone();
    missing_kv.kv_storage.clear();
    assert!(missing_kv
        .validate_program(prepared.program())
        .unwrap_err()
        .to_string()
        .contains("is missing from the numerical storage contract"));
}

#[test]
fn upstream_marker_v2_rejects_modified_policy_and_unconsumed_quantized_sources() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let mut config = source();
    let mut profile = provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&candidate_id())
        .unwrap()
        .clone();
    let op = profile
        .operations
        .iter_mut()
        .find(|op| op.operation_id.as_str() == UpstreamMarkerV2Profile::Causal.operation_id())
        .unwrap();
    let composite = op.composite_arithmetic.as_mut().unwrap();
    // Removing a projection is a valid strict fallback declaration in isolation,
    // but it is not the exact family profile requested by this identifier.
    composite.projections.pop();
    profile.validate().unwrap();
    assert!(provider.semantic_program(&config, &profile).is_err());
    for weight in &mut config.weights {
        weight.source_encoding = FamilyWeightSourceEncoding::Dense {
            element_type: ElementType::F16,
        };
    }
    let text = Qwen35FamilyProvider::text_config(&config).unwrap();
    assert!(!eligible(&config, &text));
    let mut unused = config
        .weights
        .iter()
        .find(|w| w.role == "self_attn_q")
        .unwrap()
        .clone();
    unused.source_encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new("quantization.gguf.q4-k").unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: 144,
    });
    unused.layer_index = Some(0);
    config.weights.push(unused);
    assert!(!eligible(&config, &text));
    config.weights.last_mut().unwrap().layer_index = Some(text.num_hidden_layers as u32);
    assert!(!eligible(&config, &text));
    assert!(provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&candidate_id())
        .is_err());
}

#[test]
fn upstream_marker_v2_declares_exact_physical_rows_and_preserves_old_g32_contracts() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let config = source();
    let catalog = provider.numerical_profiles(&config).unwrap();
    let old = catalog
        .resolve(
            &NumericalProfileId::new(
                F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID,
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        old.operations
            .iter()
            .find(|op| op.operation_id.as_str() == Q8ActAttentionProfile::Causal.operation_id())
            .unwrap()
            .composite_arithmetic,
        Some(Q8ActAttentionProfile::Causal.arithmetic())
    );
    for kind in [
        UpstreamMarkerV2Profile::SwiGlu,
        UpstreamMarkerV2Profile::GatedDelta,
        UpstreamMarkerV2Profile::Causal,
    ] {
        for projection in kind.arithmetic().projections {
            for leaf in projection.leaves {
                let [NumericalArithmeticStage::UpstreamProjection { policy }] =
                    leaf.arithmetic.stages.as_slice()
                else {
                    panic!("actual upstream contract")
                };
                for rows in [1, 4] {
                    assert_eq!(
                        policy
                            .routes
                            .iter()
                            .find(|r| r.local_rows.contains(&rows))
                            .unwrap()
                            .arithmetic,
                        UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2
                    );
                }
                let row8 = policy
                    .routes
                    .iter()
                    .find(|route| route.local_rows.contains(&8))
                    .unwrap();
                let mmq_eight = matches!(
                    projection.role,
                    ProjectionRole::SwiGluDown | ProjectionRole::SwiGluGateUp
                );
                assert_eq!(
                    row8.arithmetic,
                    if mmq_eight {
                        if leaf.format == ProjectionBlockFormat::Iq4Xs {
                            UpstreamProjectionArithmetic::MmqD4MarkerV2
                        } else {
                            UpstreamProjectionArithmetic::MmqDs4MarkerV2
                        }
                    } else {
                        UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2
                    }
                );
                for rows in [2, 3, 5, 6, 7, 9, 15, 16, 17, 31, 32] {
                    assert!(matches!(
                        policy
                            .routes
                            .iter()
                            .find(|r| r.local_rows.contains(&rows))
                            .unwrap()
                            .arithmetic,
                        UpstreamProjectionArithmetic::MmqD4MarkerV2
                            | UpstreamProjectionArithmetic::MmqDs4MarkerV2
                    ));
                }
                assert!(!policy.routes.iter().any(|r| r.local_rows.contains(&33)));
            }
        }
    }
}

#[test]
fn hybrid_g32_mmq_require_lowers_new_ids_without_changing_auto_or_old_profiles() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let config = source();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let id = NumericalProfileId::new(
        F32_MASTER_FFN_ATTENTION_Q4K_Q5K_IQ4XS_G32_MMQ_PREFILL_MARKER_V1_NUMERICAL_PROFILE_ID,
    )
    .unwrap();
    let profiles = definition.numerical_profiles();
    let selected = profiles
        .candidates(
            &NumericalExecutionPolicy::Require(id.clone()),
            KvStorageFormat::F16,
        )
        .unwrap()[0];
    assert!(profiles
        .candidates(
            &NumericalExecutionPolicy::Require(id.clone()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    let strict = profiles
        .resolve(&NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap())
        .unwrap();
    assert_eq!(
        profiles
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap(),
        [strict]
    );
    let prepared = registration.prepare(&definition, &id).unwrap();
    selected.validate_program(prepared.program()).unwrap();
    let prior = registration.prepare(&definition, &candidate_id()).unwrap();
    assert_eq!(prepared.program().states(), prior.program().states());
    assert_eq!(prepared.weight_schema(), prior.weight_schema());
    for kind in [
        UpstreamMarkerV2Profile::SwiGluG32MmqPrefill,
        UpstreamMarkerV2Profile::GatedDeltaG32MmqPrefill,
        UpstreamMarkerV2Profile::CausalG32MmqPrefill,
    ] {
        assert!(prepared
            .program()
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .any(|n| n.operation_id.as_str() == kind.operation_id()));
    }
    let mut tampered = selected.clone();
    tampered
        .operations
        .iter_mut()
        .find_map(|o| o.composite_arithmetic.as_mut())
        .unwrap()
        .projections
        .pop();
    assert!(Qwen35FamilyProvider::new()
        .unwrap()
        .semantic_program(&config, &tampered)
        .is_err());
}
