//! Profile/program eligibility only; real-model quality is a separate gate.
use super::*;
const FRAGMENT: &str = F32_MASTER_GGUF_F16_RN_FRAGMENT_M1_TO8_NUMERICAL_PROFILE_ID;

fn encoding(format: &str, bytes: u32) -> FamilyWeightSourceEncoding {
    FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
        format_id: QuantizationFormatId::new(format).unwrap(),
        logical_values_per_block: 256,
        bytes_per_block: bytes,
    })
}
fn config(intermediate: u64) -> Qwen35FamilyConfig {
    let mut c = native_config(intermediate);
    for w in &mut c.weights {
        match w.role.as_str() {
            "mlp_gate" | "mlp_up" => w.source_encoding = encoding("quantization.gguf.q4-k", 144),
            "mlp_down" => w.source_encoding = encoding("quantization.gguf.q6-k", 210),
            _ => {}
        }
    }
    c
}
fn prepare(c: &Qwen35FamilyConfig, profile: &str) -> ferrum_interfaces::vnext::PreparedModelFamily {
    TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &NumericalProfileId::new(profile).unwrap(),
        )
        .unwrap()
}

#[test]
fn rn_fragment_profile_is_explicit_and_only_changes_ffn_relative_to_rn() {
    let c = config(256);
    let provider = Qwen35FamilyProvider::new().unwrap();
    provider.validate_typed_config(&c).unwrap();
    let catalog = provider.numerical_profiles(&c).unwrap();
    let id = NumericalProfileId::new(FRAGMENT).unwrap();
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
    let rounded = prepare(&c, F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID);
    let fragment = prepare(&c, FRAGMENT);
    assert_eq!(rounded.weight_schema(), fragment.weight_schema());
    assert_eq!(rounded.canonical_config(), fragment.canonical_config());
    assert_eq!(rounded.program().states(), fragment.program().states());
    let mut expected = rounded.numerical_profile().clone();
    expected.id = id;
    let ffn = expected
        .operations
        .iter_mut()
        .find(|o| o.operation_id.as_str() == DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID)
        .unwrap();
    ffn.operation_id = operation_id(DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID).unwrap();
    expected
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(&expected, fragment.numerical_profile());
    let mut changed = 0;
    for (before, after) in rounded
        .program()
        .blocks()
        .iter()
        .zip(fragment.program().blocks())
    {
        let mut expected = before.clone();
        for node in &mut expected.nodes {
            if node.operation_id.as_str() == DENSE_SWIGLU_GGUF_F16_WEIGHTS_OPERATION_ID {
                node.operation_id =
                    operation_id(DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID).unwrap();
                changed += 1;
            }
        }
        assert_eq!(
            &expected, after,
            "attention/head/embedding/KV/residual remain identical to RN"
        );
    }
    assert_eq!(
        changed,
        Qwen35FamilyProvider::text_config(&c)
            .unwrap()
            .num_hidden_layers
    );
    let without = provider.numerical_profiles(&native_config(256)).unwrap();
    assert!(without
        .resolve(&NumericalProfileId::new(FRAGMENT).unwrap())
        .is_err());
    for old in [
        F16_NUMERICAL_PROFILE_ID,
        F32_MASTER_NUMERICAL_PROFILE_ID,
        F32_MASTER_GGUF_F16_PROJECTIONS_NUMERICAL_PROFILE_ID,
        F16_INT8_KV_NUMERICAL_PROFILE_ID,
        F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID,
    ] {
        let old = NumericalProfileId::new(old).unwrap();
        assert_eq!(
            serde_json::to_vec(without.resolve(&old).unwrap()).unwrap(),
            serde_json::to_vec(catalog.resolve(&old).unwrap()).unwrap()
        );
    }
}

#[test]
fn rn_fragment_eligibility_checks_every_layer_format_shape_and_forged_contract() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let c = config(256);
    let id = NumericalProfileId::new(FRAGMENT).unwrap();
    let explicit = provider
        .numerical_profiles(&c)
        .unwrap()
        .resolve(&id)
        .unwrap()
        .clone();
    for case in 0..6 {
        let mut bad = c.clone();
        let target = bad
            .weights
            .iter_mut()
            .rev()
            .find(|w| w.role == "mlp_up")
            .unwrap();
        match case {
            0 => target.source_encoding = encoding("quantization.gguf.q5-k", 176),
            1 => target.source_encoding = encoding("quantization.gguf.q4-k", 145),
            2 => {
                target.source_encoding = FamilyWeightSourceEncoding::Dense {
                    element_type: ElementType::F16,
                }
            }
            3 => target.dimensions[1] -= 1,
            4 => bad.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped,
            5 => {
                bad = config(288);
            }
            _ => unreachable!(),
        }
        let text = Qwen35FamilyProvider::text_config(&bad).unwrap();
        assert!(
            !numerical::gguf_rn_f16_fragment_eligible(&bad, &text),
            "case {case}"
        );
        assert!(
            provider.semantic_program(&bad, &explicit).is_err(),
            "case {case}"
        );
    }
    for case in 0..3 {
        let mut forged = explicit.clone();
        match case {
            0 => {
                forged
                    .operations
                    .iter_mut()
                    .find(|o| {
                        o.operation_id.as_str()
                            == DENSE_SWIGLU_GGUF_RN_F16_FRAGMENT_M1_TO8_OPERATION_ID
                    })
                    .unwrap()
                    .multiplication_type = Some(ElementType::I8)
            }
            1 => {
                forged
                    .boundaries
                    .insert(value_id("value.output.logits").unwrap(), ElementType::F16);
            }
            2 => forged.kv_storage.clear(),
            _ => unreachable!(),
        }
        assert!(provider.semantic_program(&c, &forged).is_err());
    }
}

#[test]
fn rn_fragment_eligibility_uses_declared_formats_and_general_k256_shapes() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    for (format, bytes) in [
        ("quantization.gguf.q4-k", 144),
        ("quantization.gguf.q5-k", 176),
        ("quantization.gguf.q6-k", 210),
    ] {
        for intermediate in [256, 512, 768] {
            let mut c = config(intermediate);
            for w in &mut c.weights {
                if matches!(w.role.as_str(), "mlp_gate" | "mlp_up" | "mlp_down") {
                    w.source_encoding = encoding(format, bytes);
                }
            }
            provider.validate_typed_config(&c).unwrap();
            assert!(provider
                .numerical_profiles(&c)
                .unwrap()
                .resolve(&NumericalProfileId::new(FRAGMENT).unwrap())
                .is_ok());
            prepare(&c, FRAGMENT);
        }
    }
}
