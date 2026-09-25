use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, NumericalProfileId};

fn head_id() -> NumericalProfileId {
    NumericalProfileId::new(F32_MASTER_F16_HEAD_NUMERICAL_PROFILE_ID).unwrap()
}

fn native_config(tied: bool) -> Qwen35FamilyConfig {
    let mut config = q8_swiglu_tests::native_config(256);
    if !tied {
        config.hf_config["text_config"]["tie_word_embeddings"] = false.into();
        let mut head = required_weight(&config, None, "embed_tokens")
            .unwrap()
            .clone();
        head.role = "lm_head".into();
        head.external_name = "output.weight".into();
        config.weights.push(head);
        config.weights.sort_by(|left, right| {
            (left.layer_index, left.role.as_str()).cmp(&(right.layer_index, right.role.as_str()))
        });
    }
    config
}

fn quantize_global(config: &mut Qwen35FamilyConfig, role: &str, dtype: GgmlDType, bytes: u32) {
    let weight = config
        .weights
        .iter_mut()
        .find(|weight| weight.layer_index.is_none() && weight.role == role)
        .unwrap();
    weight.source_encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new(block_quantization_format(dtype).unwrap()).unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: bytes,
    });
}

fn qualified_config(tied: bool) -> Qwen35FamilyConfig {
    let mut config = native_config(tied);
    quantize_global(
        &mut config,
        if tied { "embed_tokens" } else { "lm_head" },
        GgmlDType::Q6K,
        210,
    );
    config
}

#[test]
fn f16_head_keeps_existing_profiles_and_auto_preferences() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let mut dense_head = native_config(false);
    // Preserve the already declared Q8 profiles as well as F16/F32 and KV
    // variants when the newly qualified leaf is the output head.
    q8_swiglu_tests::quantize(&mut dense_head, "mlp_gate", GgmlDType::Q4K, 144);
    q8_swiglu_tests::quantize(&mut dense_head, "linear_attn_qkv", GgmlDType::Q6K, 210);
    provider.validate_typed_config(&dense_head).unwrap();
    let before = provider.numerical_profiles(&dense_head).unwrap();
    let mut qualified = dense_head;
    quantize_global(&mut qualified, "lm_head", GgmlDType::Q6K, 210);
    provider.validate_typed_config(&qualified).unwrap();
    let after = provider.numerical_profiles(&qualified).unwrap();
    assert!(before.resolve(&head_id()).is_err());
    assert!(after.resolve(&head_id()).is_ok());
    assert_eq!(before.auto_preference(), after.auto_preference());
    assert!(!after.auto_preference().contains(&head_id()));
    for old in before.profiles() {
        let current = after.resolve(&old.id).unwrap();
        assert_eq!(
            serde_json::to_vec(old).unwrap(),
            serde_json::to_vec(current).unwrap()
        );
        assert_eq!(old.fingerprint().unwrap(), current.fingerprint().unwrap());
    }
    for kv in [
        KvStorageFormat::F16,
        KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
    ] {
        assert_eq!(
            before
                .candidates(&NumericalExecutionPolicy::Auto, kv)
                .unwrap(),
            after
                .candidates(&NumericalExecutionPolicy::Auto, kv)
                .unwrap()
        );
    }
}

#[test]
fn f16_head_require_changes_only_selected_head_arithmetic_and_node() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    for tied in [false, true] {
        let config = qualified_config(tied);
        let definition = registration
            .define(&serde_json::to_value(&config).unwrap())
            .unwrap();
        let requested = definition
            .numerical_profiles()
            .candidates(
                &NumericalExecutionPolicy::Require(head_id()),
                KvStorageFormat::F16,
            )
            .unwrap();
        assert_eq!(
            requested
                .iter()
                .map(|profile| &profile.id)
                .collect::<Vec<_>>(),
            vec![&head_id()]
        );
        let original = registration
            .prepare(
                &definition,
                &NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap(),
            )
            .unwrap();
        let selected = registration.prepare(&definition, &head_id()).unwrap();
        assert_eq!(original.weight_schema(), selected.weight_schema());
        assert_eq!(original.canonical_config(), selected.canonical_config());
        assert_eq!(original.config_fingerprint(), selected.config_fingerprint());
        assert_eq!(original.program().states(), selected.program().states());
        assert_eq!(original.program().inputs(), selected.program().inputs());
        assert_eq!(original.program().outputs(), selected.program().outputs());
        let mut expected = original.numerical_profile().clone();
        expected.id = head_id();
        let head = expected
            .operations
            .iter_mut()
            .find(|operation| {
                operation.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID
            })
            .unwrap();
        head.operation_id =
            operation_id(LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID).unwrap();
        head.version = ContractVersion::new(1, 0);
        head.multiplication_type = Some(ElementType::F16);
        head.accumulation_type = Some(ElementType::F32);
        expected
            .operations
            .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
        assert_eq!(&expected, selected.numerical_profile());
        assert_ne!(
            original.numerical_profile().fingerprint().unwrap(),
            selected.numerical_profile().fingerprint().unwrap()
        );
        assert_ne!(
            original.program().fingerprint().unwrap(),
            selected.program().fingerprint().unwrap()
        );
        assert_ne!(
            original.fingerprint().unwrap(),
            selected.fingerprint().unwrap()
        );
        let expected_weight = weight_value_id(output_projection_weight(&config).unwrap()).unwrap();
        let mut found_head = false;
        assert_eq!(
            original.program().blocks().len(),
            selected.program().blocks().len()
        );
        for (old, new) in original
            .program()
            .blocks()
            .iter()
            .zip(selected.program().blocks())
        {
            let mut expected_block = old.clone();
            for node in &mut expected_block.nodes {
                if node.operation_id.as_str() == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID {
                    assert_eq!(node.inputs[1], expected_weight);
                    node.operation_id =
                        operation_id(LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID)
                            .unwrap();
                    assert_eq!(node.required_version, ContractVersion::new(1, 0));
                    found_head = true;
                }
            }
            assert_eq!(&expected_block, new);
        }
        assert!(found_head);
        assert!(definition
            .numerical_profiles()
            .candidates(
                &NumericalExecutionPolicy::Require(head_id()),
                KvStorageFormat::Int8PerTokenHeadF32ScaleV1
            )
            .is_err());
    }
}

#[test]
fn f16_head_requires_the_actual_selected_q6_head_and_qualified_semantics() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let qualified = qualified_config(false);
    let explicit = provider
        .numerical_profiles(&qualified)
        .unwrap()
        .resolve(&head_id())
        .unwrap()
        .clone();
    let mut unused_embedding = native_config(false);
    quantize_global(&mut unused_embedding, "embed_tokens", GgmlDType::Q6K, 210);
    let mut wrong_qtype = native_config(false);
    quantize_global(&mut wrong_qtype, "lm_head", GgmlDType::Q4K, 144);
    let mut wrong_abi = qualified.clone();
    wrong_abi.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
    let mut hadamard = qualified.clone();
    let name = output_projection_weight(&hadamard)
        .unwrap()
        .external_name
        .clone();
    hadamard.gguf_hadamard = Some(
        serde_json::from_value(serde_json::json!({
            "version": 1, "block_size": 4,
            "signs": {"mode": "explicit", "by_width": {"256": vec![1_i8; 256]}},
            "gdn_v_grouped": true,
            "weights": {name: {"direction": "before_matmul", "input_width": 256}}
        }))
        .unwrap(),
    );
    for config in [
        native_config(false),
        unused_embedding,
        wrong_qtype,
        wrong_abi,
        hadamard,
        tests::test_block_fp8_config(),
    ] {
        let definition = registration
            .define(&serde_json::to_value(&config).unwrap())
            .unwrap();
        assert!(definition.numerical_profiles().resolve(&head_id()).is_err());
        assert!(registration.prepare(&definition, &head_id()).is_err());
        assert!(provider.semantic_program(&config, &explicit).is_err());
    }
    let catalog = provider.numerical_profiles(&qualified).unwrap();
    let mut wrong_kv = explicit;
    wrong_kv.kv_storage = catalog
        .resolve(&NumericalProfileId::new(F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID).unwrap())
        .unwrap()
        .kv_storage
        .clone();
    assert!(provider.semantic_program(&qualified, &wrong_kv).is_err());
}

#[test]
fn f16_head_eligibility_checks_real_q6_block_metadata_and_shape() {
    let original = qualified_config(false);
    for (block_values, block_bytes, width) in [(128, 210, 256), (256, 209, 256), (256, 210, 255)] {
        let mut config = original.clone();
        let head = config
            .weights
            .iter_mut()
            .find(|weight| weight.role == "lm_head")
            .unwrap();
        let FamilyWeightSourceEncoding::BlockQuantized(spec) = &mut head.source_encoding else {
            unreachable!()
        };
        spec.logical_values_per_block = block_values;
        spec.bytes_per_block = block_bytes;
        head.dimensions[1] = width;
        let text = Qwen35FamilyProvider::text_config(&config).unwrap();
        assert!(!numerical::f16_head_eligible(&config, &text));
    }
}
