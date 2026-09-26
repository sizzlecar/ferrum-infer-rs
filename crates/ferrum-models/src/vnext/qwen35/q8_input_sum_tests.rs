use super::q8_swiglu_tests::{native_config, quantize};
use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, NumericalProfileId};

fn input_sum_id() -> NumericalProfileId {
    NumericalProfileId::new(F32_MASTER_Q8_SWIGLU_INPUT_SUM_NUMERICAL_PROFILE_ID).unwrap()
}

#[test]
fn q8_input_sum_is_explicit_and_preserves_existing_profile_values() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let mut config = native_config(256);
    let dense = provider.numerical_profiles(&config).unwrap();
    quantize(&mut config, "mlp_gate", GgmlDType::Q4K, 144);
    let catalog = provider.numerical_profiles(&config).unwrap();
    assert_eq!(dense.auto_preference(), catalog.auto_preference());
    assert!(!catalog.auto_preference().contains(&input_sum_id()));
    for id in [
        F16_NUMERICAL_PROFILE_ID,
        F32_MASTER_NUMERICAL_PROFILE_ID,
        F16_INT8_KV_NUMERICAL_PROFILE_ID,
        F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID,
    ] {
        let id = NumericalProfileId::new(id).unwrap();
        assert_eq!(
            serde_json::to_vec(dense.resolve(&id).unwrap()).unwrap(),
            serde_json::to_vec(catalog.resolve(&id).unwrap()).unwrap()
        );
    }
    // Derive the old opt-in policy from its unchanged strict declarations,
    // independently of the new profile constructor.
    let mut old_expected = catalog
        .resolve(&NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap())
        .unwrap()
        .clone();
    old_expected.id = NumericalProfileId::new(F32_MASTER_Q8_SWIGLU_NUMERICAL_PROFILE_ID).unwrap();
    let arithmetic = old_expected
        .operations
        .iter_mut()
        .find(|o| o.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID)
        .unwrap();
    arithmetic.operation_id = operation_id(DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID).unwrap();
    arithmetic.multiplication_type = Some(ElementType::I8);
    arithmetic.accumulation_type = Some(ElementType::F32);
    old_expected
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(&old_expected, catalog.resolve(&old_expected.id).unwrap());
    let new = catalog.resolve(&input_sum_id()).unwrap();
    assert_ne!(
        new.fingerprint().unwrap(),
        old_expected.fingerprint().unwrap()
    );
    let decoded: NumericalExecutionProfile =
        serde_json::from_slice(&serde_json::to_vec(new).unwrap()).unwrap();
    assert_eq!(&decoded, new);
    for kv in [
        ferrum_types::KvStorageFormat::F16,
        ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
    ] {
        assert_eq!(
            dense
                .candidates(&NumericalExecutionPolicy::Auto, kv)
                .unwrap(),
            catalog
                .candidates(&NumericalExecutionPolicy::Auto, kv)
                .unwrap()
        );
    }
}

#[test]
fn q8_input_sum_require_changes_only_ffn_from_strict_and_original_q8() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    for (role, dtype, bytes, intermediate) in [
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
                &NumericalExecutionPolicy::Require(input_sum_id()),
                ferrum_types::KvStorageFormat::F16,
            )
            .unwrap();
        assert_eq!(chosen.len(), 1);
        assert_eq!(chosen[0].id, input_sum_id());
        let selected = registration.prepare(&definition, &input_sum_id()).unwrap();
        for (old_id, old_op) in [
            (F32_MASTER_NUMERICAL_PROFILE_ID, DENSE_SWIGLU_OPERATION_ID),
            (
                F32_MASTER_Q8_SWIGLU_NUMERICAL_PROFILE_ID,
                DENSE_SWIGLU_Q8_F32SCALE_OPERATION_ID,
            ),
        ] {
            let original = registration
                .prepare(&definition, &NumericalProfileId::new(old_id).unwrap())
                .unwrap();
            assert_eq!(original.weight_schema(), selected.weight_schema());
            assert_eq!(original.canonical_config(), selected.canonical_config());
            assert_eq!(original.config_fingerprint(), selected.config_fingerprint());
            assert_eq!(original.program().states(), selected.program().states());
            let mut expected = original.numerical_profile().clone();
            expected.id = input_sum_id();
            let arithmetic = expected
                .operations
                .iter_mut()
                .find(|o| o.operation_id.as_str() == old_op)
                .unwrap();
            arithmetic.operation_id =
                operation_id(DENSE_SWIGLU_Q8_F32SCALE_INPUT_SUM_OPERATION_ID).unwrap();
            arithmetic.multiplication_type = Some(ElementType::I8);
            arithmetic.accumulation_type = Some(ElementType::F32);
            expected
                .operations
                .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
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
                    if node.operation_id.as_str() == old_op {
                        node.operation_id =
                            operation_id(DENSE_SWIGLU_Q8_F32SCALE_INPUT_SUM_OPERATION_ID).unwrap();
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
        }
        assert!(definition
            .numerical_profiles()
            .candidates(
                &NumericalExecutionPolicy::Require(input_sum_id()),
                ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1
            )
            .is_err());
    }
}

#[test]
fn q8_input_sum_rejects_unqualified_sources_and_foreign_profiles() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let dense = native_config(256);
    let mut eligible = dense.clone();
    quantize(&mut eligible, "mlp_gate", GgmlDType::Q4K, 144);
    let explicit = provider
        .numerical_profiles(&eligible)
        .unwrap()
        .resolve(&input_sum_id())
        .unwrap()
        .clone();
    let mut wrong_abi = eligible.clone();
    wrong_abi.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
    let mut unrelated = dense.clone();
    quantize(&mut unrelated, "linear_attn_qkv", GgmlDType::Q4K, 144);
    let mut hadamard = eligible.clone();
    let name = hadamard
        .weights
        .iter()
        .find(|w| w.layer_index == Some(0) && w.role == "mlp_gate")
        .unwrap()
        .external_name
        .clone();
    hadamard.gguf_hadamard = Some(
        serde_json::from_value(serde_json::json!({
            "version": 1, "block_size": 4,
            "signs": {"mode":"explicit", "by_width":{"256":vec![1_i8;256]}},
            "gdn_v_grouped":true,
            "weights":{name:{"direction":"before_matmul", "input_width":256}}
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
        let definition = registration
            .define(&serde_json::to_value(&config).unwrap())
            .unwrap();
        assert!(definition
            .numerical_profiles()
            .resolve(&input_sum_id())
            .is_err());
        assert!(registration.prepare(&definition, &input_sum_id()).is_err());
        assert!(provider.semantic_program(&config, &explicit).is_err());
    }
    let mut changed_kv = explicit;
    changed_kv.kv_storage = provider
        .numerical_profiles(&eligible)
        .unwrap()
        .resolve(&NumericalProfileId::new(F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID).unwrap())
        .unwrap()
        .kv_storage
        .clone();
    assert!(provider.semantic_program(&eligible, &changed_kv).is_err());
}
