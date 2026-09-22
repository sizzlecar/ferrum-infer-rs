use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, NumericalProfileId};

fn q8_id() -> NumericalProfileId {
    NumericalProfileId::new(F32_MASTER_Q8_SWIGLU_NUMERICAL_PROFILE_ID).unwrap()
}

pub(super) fn native_config(intermediate: u64) -> Qwen35FamilyConfig {
    let mut config = tests::test_dense_gguf_config();
    config.hf_config["text_config"]["hidden_size"] = 256.into();
    config.hf_config["text_config"]["intermediate_size"] = intermediate.into();
    let text = Qwen35FamilyProvider::text_config(&config).unwrap();
    for weight in &mut config.weights {
        weight.dimensions = expected_weight_dimensions(&text, config.vocab_size, weight)
            .unwrap()
            .remove(0);
    }
    config
}

pub(super) fn quantize(config: &mut Qwen35FamilyConfig, role: &str, dtype: GgmlDType, bytes: u32) {
    let weight = config
        .weights
        .iter_mut()
        .find(|weight| weight.role == role && weight.layer_index == Some(0))
        .unwrap();
    weight.source_encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new(block_quantization_format(dtype).unwrap()).unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: bytes,
    });
}

#[test]
fn q8_swiglu_preserves_old_profile_wire_fingerprints_and_auto_preferences() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let dense = native_config(256);
    provider.validate_typed_config(&dense).unwrap();
    let before = provider.numerical_profiles(&dense).unwrap();
    let mut config = dense;
    quantize(&mut config, "mlp_gate", GgmlDType::Q4K, 144);
    provider.validate_typed_config(&config).unwrap();
    let after = provider.numerical_profiles(&config).unwrap();
    assert_eq!(before.auto_preference(), after.auto_preference());
    assert!(!after.auto_preference().contains(&q8_id()));
    for id in [
        F16_NUMERICAL_PROFILE_ID,
        F32_MASTER_NUMERICAL_PROFILE_ID,
        F16_INT8_KV_NUMERICAL_PROFILE_ID,
        F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID,
    ] {
        let id = NumericalProfileId::new(id).unwrap();
        let old = before.resolve(&id).unwrap();
        let current = after.resolve(&id).unwrap();
        assert_eq!(
            serde_json::to_vec(old).unwrap(),
            serde_json::to_vec(current).unwrap()
        );
        assert_eq!(old.fingerprint().unwrap(), current.fingerprint().unwrap());
    }
    for kv in [
        ferrum_types::KvStorageFormat::F16,
        ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
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
fn q8_swiglu_require_changes_only_dense_arithmetic_and_nodes() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    for (role, dtype, bytes, intermediate) in [
        // A dense down projection need not have a K256 input axis.
        ("mlp_gate", GgmlDType::Q4K, 144, 32),
        ("mlp_up", GgmlDType::Q5K, 176, 256),
        ("mlp_down", GgmlDType::Q6K, 210, 256),
    ] {
        let mut config = native_config(intermediate);
        quantize(&mut config, role, dtype, bytes);
        let definition = registration
            .define(&serde_json::to_value(&config).unwrap())
            .unwrap();
        let chosen = definition
            .numerical_profiles()
            .candidates(
                &NumericalExecutionPolicy::Require(q8_id()),
                ferrum_types::KvStorageFormat::F16,
            )
            .unwrap();
        assert_eq!(chosen.len(), 1);
        assert_eq!(chosen[0].id, q8_id());
        let original = registration
            .prepare(
                &definition,
                &NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap(),
            )
            .unwrap();
        let selected = registration.prepare(&definition, &q8_id()).unwrap();
        assert_eq!(original.weight_schema(), selected.weight_schema());
        assert_eq!(original.canonical_config(), selected.canonical_config());
        assert_eq!(original.config_fingerprint(), selected.config_fingerprint());
        assert_eq!(original.program().states(), selected.program().states());
        let mut expected = original.numerical_profile().clone();
        expected.id = q8_id();
        let arithmetic = expected
            .operations
            .iter_mut()
            .find(|operation| operation.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID)
            .unwrap();
        arithmetic.operation_id = operation_id(DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID).unwrap();
        arithmetic.multiplication_type = Some(ElementType::I8);
        arithmetic.accumulation_type = Some(ElementType::F32);
        expected
            .operations
            .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
        assert_eq!(&expected, selected.numerical_profile());
        assert_eq!(
            original.program().blocks().len(),
            selected.program().blocks().len()
        );
        let mut changed = 0;
        for (old, new) in original
            .program()
            .blocks()
            .iter()
            .zip(selected.program().blocks())
        {
            let mut expected = old.clone();
            for node in &mut expected.nodes {
                if node.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID {
                    node.operation_id =
                        operation_id(DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID).unwrap();
                    assert_eq!(node.required_version, ContractVersion::new(1, 0));
                    changed += 1;
                }
            }
            assert_eq!(&expected, new);
        }
        assert_eq!(
            changed,
            Qwen35FamilyProvider::text_config(&config)
                .unwrap()
                .num_hidden_layers
        );
        assert!(definition
            .numerical_profiles()
            .candidates(
                &NumericalExecutionPolicy::Require(q8_id()),
                ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1
            )
            .is_err());
    }
}

#[test]
fn q8_swiglu_is_not_offered_for_unqualified_sources_or_combinations() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let dense = native_config(256);
    let mut eligible = dense.clone();
    quantize(&mut eligible, "mlp_gate", GgmlDType::Q4K, 144);
    let explicit = provider
        .numerical_profiles(&eligible)
        .unwrap()
        .resolve(&q8_id())
        .unwrap()
        .clone();
    let mut wrong_abi = eligible.clone();
    wrong_abi.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
    let mut unrelated = dense.clone();
    quantize(&mut unrelated, "linear_attn_qkv", GgmlDType::Q4K, 144);
    let mut hadamard = eligible;
    let name = hadamard
        .weights
        .iter()
        .find(|weight| weight.layer_index == Some(0) && weight.role == "mlp_gate")
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
        dense,
        wrong_abi,
        unrelated,
        hadamard,
        tests::test_moe_gguf_config(),
        tests::test_block_fp8_config(),
    ] {
        // Use valid family definitions, rather than making malformed weights
        // stand in for a supported but unqualified execution combination.
        let definition = registration
            .define(&serde_json::to_value(&config).unwrap())
            .unwrap();
        assert!(definition.numerical_profiles().resolve(&q8_id()).is_err());
        assert!(registration.prepare(&definition, &q8_id()).is_err());
        assert!(provider.semantic_program(&config, &explicit).is_err());
    }
}
