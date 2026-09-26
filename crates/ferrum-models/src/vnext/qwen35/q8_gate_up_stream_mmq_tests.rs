use super::q8_swiglu_tests::{native_config, quantize};
use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, NumericalProfileId};

fn policy_id() -> NumericalProfileId {
    NumericalProfileId::new(F32_MASTER_Q8_GATE_UP_STREAM_MMQ_NUMERICAL_PROFILE_ID).unwrap()
}

fn eligible_config() -> Qwen35FamilyConfig {
    let mut config = native_config(256);
    quantize(&mut config, "mlp_gate", GgmlDType::Q4K, 144);
    quantize(&mut config, "mlp_up", GgmlDType::Q4K, 144);
    config
}

#[test]
fn stream_mmq_is_opt_in_and_does_not_rewrite_existing_profile_contracts() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let mut partial = native_config(256);
    quantize(&mut partial, "mlp_gate", GgmlDType::Q4K, 144);
    let old = provider.numerical_profiles(&partial).unwrap();
    assert!(old.resolve(&policy_id()).is_err());
    let new = provider.numerical_profiles(&eligible_config()).unwrap();
    assert_eq!(old.auto_preference(), new.auto_preference());
    assert!(!new.auto_preference().contains(&policy_id()));
    for id in [
        F16_NUMERICAL_PROFILE_ID,
        F32_MASTER_NUMERICAL_PROFILE_ID,
        F16_INT8_KV_NUMERICAL_PROFILE_ID,
        F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID,
        F32_MASTER_Q8_SWIGLU_NUMERICAL_PROFILE_ID,
        F32_MASTER_Q8_SWIGLU_INPUT_SUM_NUMERICAL_PROFILE_ID,
    ] {
        let id = NumericalProfileId::new(id).unwrap();
        assert_eq!(
            serde_json::to_vec(old.resolve(&id).unwrap()).unwrap(),
            serde_json::to_vec(new.resolve(&id).unwrap()).unwrap()
        );
    }
    let selected = new.resolve(&policy_id()).unwrap();
    let round_trip: NumericalExecutionProfile =
        serde_json::from_slice(&serde_json::to_vec(selected).unwrap()).unwrap();
    assert_eq!(&round_trip, selected);
    assert!(new
        .candidates(
            &NumericalExecutionPolicy::Require(policy_id()),
            ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
}

#[test]
fn stream_mmq_changes_all_ffn_identities_but_no_other_program_or_state_contract() {
    // Only the first layer is eligible; fallback layers must still retain the
    // explicitly selected operation identity. Runtime eligibility is not a
    // claim that every node or every invocation takes the approximate branch.
    let config = eligible_config();
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let old = registration
        .prepare(
            &definition,
            &NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let new = registration.prepare(&definition, &policy_id()).unwrap();
    assert_eq!(old.weight_schema(), new.weight_schema());
    assert_eq!(old.canonical_config(), new.canonical_config());
    assert_eq!(old.config_fingerprint(), new.config_fingerprint());
    assert_eq!(old.program().states(), new.program().states());
    assert_eq!(old.program().blocks().len(), new.program().blocks().len());
    let mut changed = 0;
    for (a, b) in old.program().blocks().iter().zip(new.program().blocks()) {
        let mut expected = a.clone();
        for node in &mut expected.nodes {
            if node.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID {
                node.operation_id =
                    operation_id(DENSE_SWIGLU_Q8_GATE_UP_STREAM_MMQ_OPERATION_ID).unwrap();
                changed += 1;
            }
        }
        assert_eq!(&expected, b);
    }
    assert_eq!(
        changed,
        Qwen35FamilyProvider::text_config(&config)
            .unwrap()
            .num_hidden_layers
    );
    let mut expected = old.numerical_profile().clone();
    expected.id = policy_id();
    let dense = expected
        .operations
        .iter_mut()
        .find(|o| o.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID)
        .unwrap();
    dense.operation_id = operation_id(DENSE_SWIGLU_Q8_GATE_UP_STREAM_MMQ_OPERATION_ID).unwrap();
    dense.multiplication_type = Some(ElementType::I8);
    dense.accumulation_type = Some(ElementType::F32);
    expected
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(&expected, new.numerical_profile());
}

#[test]
fn stream_mmq_requires_same_layer_q4_pair_and_rejects_foreign_policy_injection() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let eligible = eligible_config();
    let profile = provider
        .numerical_profiles(&eligible)
        .unwrap()
        .resolve(&policy_id())
        .unwrap()
        .clone();
    let dense = native_config(256);
    let mut wrong_up = eligible.clone();
    quantize(&mut wrong_up, "mlp_up", GgmlDType::Q5K, 176);
    let mut down_only = dense.clone();
    quantize(&mut down_only, "mlp_down", GgmlDType::Q4K, 144);
    let mut wrong_abi = eligible.clone();
    wrong_abi.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
    let mut different_layers = eligible.clone();
    let dense_up = dense
        .weights
        .iter()
        .find(|w| w.role == "mlp_up" && w.layer_index == Some(0))
        .unwrap()
        .source_encoding
        .clone();
    different_layers
        .weights
        .iter_mut()
        .find(|w| w.role == "mlp_up" && w.layer_index == Some(0))
        .unwrap()
        .source_encoding = dense_up;
    let q4 = eligible
        .weights
        .iter()
        .find(|w| w.role == "mlp_up" && w.layer_index == Some(0))
        .unwrap()
        .source_encoding
        .clone();
    different_layers
        .weights
        .iter_mut()
        .find(|w| w.role == "mlp_up" && w.layer_index == Some(1))
        .unwrap()
        .source_encoding = q4;
    for config in [dense, wrong_up, down_only, wrong_abi, different_layers] {
        assert!(provider
            .numerical_profiles(&config)
            .unwrap()
            .resolve(&policy_id())
            .is_err());
        assert!(provider.semantic_program(&config, &profile).is_err());
    }
    let mut wrong_kv = profile;
    wrong_kv.kv_storage = provider
        .numerical_profiles(&eligible)
        .unwrap()
        .resolve(&NumericalProfileId::new(F32_MASTER_INT8_KV_NUMERICAL_PROFILE_ID).unwrap())
        .unwrap()
        .kv_storage
        .clone();
    assert!(provider.semantic_program(&eligible, &wrong_kv).is_err());
}
