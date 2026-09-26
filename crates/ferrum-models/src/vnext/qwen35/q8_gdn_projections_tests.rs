use super::q8_swiglu_tests::{native_config, quantize};
use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, NumericalProfileId};

fn profile_id() -> NumericalProfileId {
    NumericalProfileId::new(F32_MASTER_Q8_SWIGLU_GDN_PROJECTIONS_NUMERICAL_PROFILE_ID).unwrap()
}

fn ff_only() -> Qwen35FamilyConfig {
    let mut config = native_config(256);
    quantize(&mut config, "mlp_gate", GgmlDType::Q4K, 144);
    config
}

#[test]
fn q8_gdn_projections_preserve_old_profiles_and_auto_preferences() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let mut config = ff_only();
    provider.validate_typed_config(&config).unwrap();
    let before = provider.numerical_profiles(&config).unwrap();
    quantize(&mut config, "linear_attn_qkv", GgmlDType::Q5K, 176);
    provider.validate_typed_config(&config).unwrap();
    let after = provider.numerical_profiles(&config).unwrap();
    assert_eq!(after.version(), ContractVersion::new(1, 7));
    assert_eq!(
        after.resolve(&profile_id()).unwrap().version,
        ContractVersion::new(1, 0)
    );
    assert!(before.resolve(&profile_id()).is_err());
    assert_eq!(before.auto_preference(), after.auto_preference());
    assert!(!after.auto_preference().contains(&profile_id()));
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
    let candidates = after
        .candidates(
            &NumericalExecutionPolicy::Require(profile_id()),
            KvStorageFormat::F16,
        )
        .unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].id, profile_id());
    assert!(after
        .candidates(
            &NumericalExecutionPolicy::Require(profile_id()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
}

#[test]
fn q8_gdn_projections_require_changes_only_gdn_arithmetic_and_nodes() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    for (role, format, bytes) in [
        ("linear_attn_qkv", GgmlDType::Q4K, 144),
        ("linear_attn_z", GgmlDType::Q5K, 176),
        ("linear_attn_b", GgmlDType::Q6K, 210),
        ("linear_attn_a", GgmlDType::Q4K, 144),
        ("linear_attn_out", GgmlDType::Q6K, 210),
    ] {
        let mut config = ff_only();
        if role == "linear_attn_out" {
            config.hf_config["text_config"]["linear_value_head_dim"] = 128.into();
            let text = Qwen35FamilyProvider::text_config(&config).unwrap();
            for weight in &mut config.weights {
                weight.dimensions = expected_weight_dimensions(&text, config.vocab_size, weight)
                    .unwrap()
                    .remove(0);
            }
        }
        // Every other physical leaf remains dense. In the input-projection
        // cases the dense output projection deliberately has a non-K256 width.
        quantize(&mut config, role, format, bytes);
        let definition = registration
            .define(&serde_json::to_value(&config).unwrap())
            .unwrap();
        let old = registration
            .prepare(
                &definition,
                &NumericalProfileId::new(F32_MASTER_Q8_SWIGLU_NUMERICAL_PROFILE_ID).unwrap(),
            )
            .unwrap();
        let new = registration.prepare(&definition, &profile_id()).unwrap();
        assert_eq!(old.weight_schema(), new.weight_schema());
        assert_eq!(old.canonical_config(), new.canonical_config());
        assert_eq!(old.config_fingerprint(), new.config_fingerprint());
        assert_eq!(old.program().states(), new.program().states());
        assert_ne!(
            old.program().fingerprint().unwrap(),
            new.program().fingerprint().unwrap()
        );
        let mut expected_profile = old.numerical_profile().clone();
        expected_profile.id = profile_id();
        let arithmetic = expected_profile
            .operations
            .iter_mut()
            .find(|op| {
                op.operation_id.as_str() == GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID
            })
            .unwrap();
        arithmetic.operation_id =
            operation_id(GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q8_PROJECTIONS_OPERATION_ID)
                .unwrap();
        arithmetic.multiplication_type = Some(ElementType::I8);
        arithmetic.accumulation_type = Some(ElementType::F32);
        expected_profile
            .operations
            .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
        assert_eq!(&expected_profile, new.numerical_profile());
        assert_eq!(old.program().blocks().len(), new.program().blocks().len());
        let mut changed = 0;
        for (old, new) in old.program().blocks().iter().zip(new.program().blocks()) {
            let mut expected = old.clone();
            for node in &mut expected.nodes {
                if node.operation_id.as_str()
                    == GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_OPERATION_ID
                {
                    node.operation_id = operation_id(
                        GATED_DELTA_RECURRENT_ATTENTION_F32_MASTER_Q8_PROJECTIONS_OPERATION_ID,
                    )
                    .unwrap();
                    assert_eq!(node.required_version, ContractVersion::new(1, 0));
                    changed += 1;
                }
            }
            assert_eq!(&expected, new);
        }
        let text = Qwen35FamilyProvider::text_config(&config).unwrap();
        assert_eq!(
            changed,
            text.layer_types
                .iter()
                .filter(|&&kind| kind == Qwen35LayerType::LinearAttention)
                .count()
        );
    }
}

#[test]
fn q8_gdn_projections_reject_unqualified_sources_and_int8_kv() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let mut eligible = ff_only();
    quantize(&mut eligible, "linear_attn_qkv", GgmlDType::Q5K, 176);
    let explicit = provider
        .numerical_profiles(&eligible)
        .unwrap()
        .resolve(&profile_id())
        .unwrap()
        .clone();
    let mut no_ffn = native_config(256);
    quantize(&mut no_ffn, "linear_attn_qkv", GgmlDType::Q5K, 176);
    let mut wrong_abi = eligible.clone();
    wrong_abi.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
    let mut other_format = ff_only();
    other_format
        .weights
        .iter_mut()
        .find(|w| w.role == "linear_attn_qkv" && w.layer_index == Some(0))
        .unwrap()
        .source_encoding = FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new(block_quantization_format(GgmlDType::Q8_0).unwrap())
            .unwrap(),
        logical_values_per_block: 32,
        bytes_per_block: 34,
    });
    let mut hadamard = eligible.clone();
    let name = hadamard
        .weights
        .iter()
        .find(|w| w.role == "linear_attn_qkv" && w.layer_index == Some(0))
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
        ff_only(),
        no_ffn,
        wrong_abi,
        other_format,
        hadamard,
        tests::test_moe_gguf_config(),
        tests::test_block_fp8_config(),
    ] {
        let definition = registration
            .define(&serde_json::to_value(&config).unwrap())
            .unwrap();
        assert!(definition
            .numerical_profiles()
            .resolve(&profile_id())
            .is_err());
        assert!(registration.prepare(&definition, &profile_id()).is_err());
        assert!(provider.semantic_program(&config, &explicit).is_err());
    }
    // Even an imported profile must not pair this policy with INT8 KV.
    let int8 = provider
        .numerical_profiles(&eligible)
        .unwrap()
        .resolve(&NumericalProfileId::new(F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID).unwrap())
        .unwrap()
        .clone();
    let mut altered = explicit;
    altered.kv_storage = int8.kv_storage;
    altered.states = int8.states;
    assert!(provider.semantic_program(&eligible, &altered).is_err());
}

#[test]
fn q8_gdn_projections_qualification_checks_role_layer_shape_and_block_metadata() {
    let mut eligible = ff_only();
    quantize(&mut eligible, "linear_attn_qkv", GgmlDType::Q5K, 176);
    let text = Qwen35FamilyProvider::text_config(&eligible).unwrap();
    assert!(numerical::q8_gdn_projections_eligible(&eligible, &text));
    let leaf = eligible
        .weights
        .iter()
        .position(|w| w.role == "linear_attn_qkv" && w.layer_index == Some(0))
        .unwrap();
    // Malformed metadata must not grant qualification. These are deliberately
    // not presented as valid family definitions or provider-executable models.
    for variant in 0..9 {
        let mut config = eligible.clone();
        let weight = &mut config.weights[leaf];
        match variant {
            0 => weight.role = "linear_attn_conv".to_owned(),
            1 => weight.layer_index = None,
            2 => weight.layer_index = Some(text.layer_types.len() as u32),
            3 => {
                weight.layer_index = Some(
                    text.layer_types
                        .iter()
                        .position(|&kind| kind == Qwen35LayerType::FullAttention)
                        .unwrap() as u32,
                )
            }
            4 => weight.dimensions[1] = 255,
            5 => weight.dimensions[0] = 0,
            6 => weight.dimensions.push(1),
            7 | 8 => {
                let FamilyWeightSourceEncoding::BlockQuantized(spec) = &mut weight.source_encoding
                else {
                    unreachable!()
                };
                if variant == 7 {
                    spec.bytes_per_block = 144;
                } else {
                    spec.logical_values_per_block = 32;
                }
            }
            _ => unreachable!(),
        }
        assert!(
            !numerical::q8_gdn_projections_eligible(&config, &text),
            "variant {variant}"
        );
    }
}
