use super::q8_swiglu_tests::native_config;
use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, NumericalProfileId};

fn selected_id() -> NumericalProfileId {
    NumericalProfileId::new(F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID).unwrap()
}
fn replacement(operation: &str) -> Option<&'static str> {
    match operation {
        DENSE_SWIGLU_OPERATION_ID => Some(DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID),
        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID => {
            Some(GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID)
        }
        CAUSAL_PAGED_ATTENTION_F32_MASTER_OPERATION_ID => {
            Some(CAUSAL_PAGED_ATTENTION_F32_MASTER_GGUF_F16_PROJECTIONS_OPERATION_ID)
        }
        _ => None,
    }
}

#[test]
fn gguf_f16_projections_change_all_three_projection_operations_and_no_other_boundaries() {
    let config = native_config(256);
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let catalog = definition.numerical_profiles();
    assert_eq!(catalog.version(), ContractVersion::new(1, 7));
    assert!(!catalog.auto_preference().contains(&selected_id()));
    let candidates = catalog
        .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
        .unwrap();
    assert_eq!(
        candidates.iter().map(|p| p.id.as_str()).collect::<Vec<_>>(),
        vec![F32_MASTER_NUMERICAL_PROFILE_ID]
    );
    assert!(catalog
        .candidates(
            &NumericalExecutionPolicy::Require(selected_id()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    let old = registration
        .prepare(
            &definition,
            &NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let new = registration.prepare(&definition, &selected_id()).unwrap();
    assert_eq!(
        old.weight_schema(),
        new.weight_schema(),
        "conversion belongs to separately authorized materialization"
    );
    assert_eq!(old.canonical_config(), new.canonical_config());
    assert_eq!(old.config_fingerprint(), new.config_fingerprint());
    assert_eq!(old.program().states(), new.program().states());
    let mut expected = old.numerical_profile().clone();
    expected.id = selected_id();
    let mut changed = BTreeSet::new();
    for operation in &mut expected.operations {
        if let Some(id) = replacement(operation.operation_id.as_str()) {
            changed.insert(id);
            operation.operation_id = operation_id(id).unwrap();
            operation.version = ContractVersion::new(1, 0);
            operation.multiplication_type = Some(ElementType::F16);
            operation.accumulation_type = Some(ElementType::F32);
        }
    }
    assert_eq!(
        changed.len(),
        3,
        "fixture must exercise FFN, GDN and causal attention"
    );
    expected
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(&expected, new.numerical_profile());
    assert_ne!(
        old.numerical_profile().fingerprint().unwrap(),
        new.numerical_profile().fingerprint().unwrap()
    );
    assert_eq!(old.program().blocks().len(), new.program().blocks().len());
    let mut changed_nodes = 0;
    for (before, after) in old.program().blocks().iter().zip(new.program().blocks()) {
        let mut expected = before.clone();
        for node in &mut expected.nodes {
            if let Some(id) = replacement(node.operation_id.as_str()) {
                node.operation_id = operation_id(id).unwrap();
                node.required_version = ContractVersion::new(1, 0);
                changed_nodes += 1;
            }
        }
        assert_eq!(
            &expected, after,
            "head, embedding, states, KV and every unrelated node retain strict semantics"
        );
    }
    assert_eq!(
        changed_nodes,
        Qwen35FamilyProvider::text_config(&config)
            .unwrap()
            .num_hidden_layers
            * 2
    );
}

#[test]
fn gguf_f16_projections_reject_unsupported_source_and_forged_state_policy() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let config = native_config(256);
    let explicit = provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&selected_id())
        .unwrap()
        .clone();
    let mut unsupported = config.clone();
    let projection = unsupported
        .weights
        .iter_mut()
        .find(|w| w.role == "mlp_gate")
        .unwrap();
    projection.source_encoding =
        FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
            format_id: QuantizationFormatId::new("quantization.gguf.q4-0").unwrap(),
            logical_values_per_block: 32,
            bytes_per_block: 18,
        });
    let mut f32_weight = config.clone();
    f32_weight
        .weights
        .iter_mut()
        .find(|w| w.role == "mlp_up")
        .unwrap()
        .source_encoding = FamilyWeightSourceEncoding::Dense {
        element_type: ElementType::F32,
    };
    let mut wrong_abi = config.clone();
    wrong_abi.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
    for rejected in [
        unsupported,
        f32_weight,
        wrong_abi,
        tests::test_config(),
        tests::test_moe_gguf_config(),
        tests::test_block_fp8_config(),
    ] {
        // Valid source combinations can be outside this first materializer's scope.
        assert!(provider
            .numerical_profiles(&rejected)
            .unwrap()
            .resolve(&selected_id())
            .is_err());
        assert!(provider.semantic_program(&rejected, &explicit).is_err());
    }
    let mut forged = explicit.clone();
    forged
        .boundaries
        .insert(value_id("value.output.logits").unwrap(), ElementType::F16);
    assert!(provider.semantic_program(&config, &forged).is_err());
    let mut forged = explicit;
    forged
        .operations
        .iter_mut()
        .find(|o| o.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID)
        .unwrap()
        .multiplication_type = Some(ElementType::F16);
    assert!(provider.semantic_program(&config, &forged).is_err());
}

#[test]
fn gguf_f16_projection_source_whitelist_applies_only_to_declared_projection_leaves() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    for (format, values, bytes) in [
        ("quantization.gguf.q4-k", 256, 144),
        ("quantization.gguf.q5-k", 256, 176),
        ("quantization.gguf.q6-k", 256, 210),
        ("quantization.gguf.q8-0", 32, 34),
    ] {
        for role in ["mlp_gate", "linear_attn_qkv", "self_attn_q"] {
            let mut config = native_config(256);
            let weight = config.weights.iter_mut().find(|w| w.role == role).unwrap();
            weight.source_encoding =
                FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: QuantizationFormatId::new(format).unwrap(),
                    logical_values_per_block: values,
                    bytes_per_block: bytes,
                });
            provider.validate_typed_config(&config).unwrap();
            assert!(provider
                .numerical_profiles(&config)
                .unwrap()
                .resolve(&selected_id())
                .is_ok());
        }
    }
    let mut config = native_config(256);
    let head = config
        .weights
        .iter_mut()
        .find(|w| w.role == "embed_tokens")
        .unwrap();
    head.source_encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new("quantization.gguf.q4-0").unwrap(),
        logical_values_per_block: 32,
        bytes_per_block: 18,
    });
    provider.validate_typed_config(&config).unwrap();
    assert!(provider.numerical_profiles(&config).unwrap().resolve(&selected_id()).is_ok(),
        "untouched head storage is not permission to convert it, nor subject to projection whitelist");
}

#[path = "gguf_f16_projection_tests/authority.rs"]
mod authority;
