use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, TypedFamilyRegistration};

fn source() -> Qwen35FamilyConfig {
    let mut config = super::super::q8act_attention_tests::source();
    for weight in &mut config.weights {
        let (format, qk, bytes) = match weight.role.as_str() {
            "linear_attn_qkv" => ("quantization.gguf.q3-k", 256, 110),
            "linear_attn_z" => ("quantization.gguf.iq3-s", 256, 110),
            "self_attn_q" => ("quantization.gguf.iq4-nl", 32, 18),
            _ => continue,
        };
        weight.source_encoding =
            FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
                format_id: QuantizationFormatId::new(format).unwrap(),
                logical_values_per_block: qk,
                bytes_per_block: bytes,
            });
    }
    config
}
fn candidate_id() -> NumericalProfileId {
    NumericalProfileId::new(F32_MASTER_FFN_ATTENTION_Q3K_Q4K_Q5K_IQ3S_IQ4NL_IQ4XS_UPSTREAM_MARKER_V2_EXTRA_PREFILL_NUMERICAL_PROFILE_ID).unwrap()
}

#[test]
fn upstream_extra_prefill_require_lowers_trio_with_real_kv_and_leaves_auto_strict() {
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
    let prepared = registration.prepare(&definition, &candidate_id()).unwrap();
    let baseline = registration.prepare(&definition, &strict_id).unwrap();
    assert_eq!(prepared.program().states(), baseline.program().states());
    assert_eq!(prepared.weight_schema(), baseline.weight_schema());
    selected.validate_program(prepared.program()).unwrap();
    let mut restored = selected.clone();
    restored.id = strict.id.clone();
    for kind in [
        UpstreamMarkerV2Profile::SwiGluExtraLargePrefill,
        UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill,
        UpstreamMarkerV2Profile::CausalExtraLargePrefill,
    ] {
        assert!(prepared
            .program()
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .any(|n| n.operation_id.as_str() == kind.operation_id()));
        let op = restored
            .operations
            .iter_mut()
            .find(|op| op.operation_id.as_str() == kind.operation_id())
            .unwrap();
        assert_eq!(op.composite_arithmetic, Some(kind.arithmetic()));
        *op = strict
            .operations
            .iter()
            .find(|o| o.operation_id.as_str() == kind.strict_operation_id())
            .unwrap()
            .clone();
    }
    restored
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(
        &restored, strict,
        "head, state, outer arithmetic and nonprojection operations are unchanged"
    );
    let mut missing_kv = selected.clone();
    missing_kv.kv_storage.clear();
    assert!(missing_kv.validate_program(prepared.program()).is_err());
    let mut tampered = selected.clone();
    tampered
        .operations
        .iter_mut()
        .find_map(|o| o.composite_arithmetic.as_mut())
        .unwrap()
        .projections
        .pop();
    tampered.validate().unwrap();
    assert!(Qwen35FamilyProvider::new()
        .unwrap()
        .semantic_program(&config, &tampered)
        .is_err());
}

#[test]
fn upstream_extra_prefill_eligibility_uses_consumed_roles_exact_abi_and_k256_not_weight_qk32() {
    let mut config = source();
    let text = Qwen35FamilyProvider::text_config(&config).unwrap();
    assert!(extra_eligible(&config, &text));
    assert!(
        !eligible(&config, &text),
        "the old profile remains closed to extra-only weights"
    );
    for weight in &mut config.weights {
        weight.source_encoding = FamilyWeightSourceEncoding::Dense {
            element_type: ElementType::F16,
        };
    }
    assert!(!extra_eligible(&config, &text));
    let index = config
        .weights
        .iter()
        .position(|w| w.role == "self_attn_q")
        .unwrap();
    let valid = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new("quantization.gguf.iq4-nl").unwrap(),
        logical_values_per_block: 32,
        bytes_per_block: 18,
    });
    config.weights[index].source_encoding = valid.clone();
    assert!(extra_eligible(&config, &text));
    config.weights[index].dimensions[1] = 32;
    assert!(!extra_eligible(&config, &text));
    config.weights[index].dimensions[1] = 256;
    let FamilyWeightSourceEncoding::BlockQuantized(block) =
        &mut config.weights[index].source_encoding
    else {
        unreachable!()
    };
    block.logical_values_per_block = 256;
    assert!(!extra_eligible(&config, &text));
    config.weights[index].source_encoding = valid;
    config.weights[index].layer_index = Some(0); // The source's layer zero is GDN, not causal.
    assert!(!extra_eligible(&config, &text));
    config.weights[index].layer_index = Some(text.num_hidden_layers as u32);
    assert!(!extra_eligible(&config, &text));
    assert!(Qwen35FamilyProvider::new()
        .unwrap()
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&candidate_id())
        .is_err());
}

#[test]
fn upstream_extra_prefill_added_profile_preserves_existing_three_format_profiles_exactly() {
    let config = super::super::q8act_attention_tests::source();
    let family = ModelFamilyId::new(FAMILY_ID).unwrap();
    let text = Qwen35FamilyProvider::text_config(&config).unwrap();
    let (states, kv) =
        super::super::states(&text, config.max_position_embeddings, KvStorageFormat::F16).unwrap();
    let master = super::super::profile(&family, &text, &states, &kv, true, false).unwrap();
    let catalog = super::super::profiles(&family, &config).unwrap();
    for original in [
        profile(&master).unwrap(),
        profile_for(&master, true).unwrap(),
        hybrid_profile(&master).unwrap(),
        extra_profile(&master).unwrap(),
    ] {
        assert_eq!(
            serde_json::to_vec(catalog.resolve(&original.id).unwrap()).unwrap(),
            serde_json::to_vec(&original).unwrap()
        );
    }
    assert!(catalog.resolve(&candidate_id()).is_ok());
}
