use super::*;
use ferrum_interfaces::vnext::{NumericalExecutionPolicy, TypedFamilyRegistration};

fn source() -> Qwen35FamilyConfig {
    let mut config = super::super::tests::test_dense_gguf_config();
    // Only the FFN width changes; a down projection has one complete IQ4_XS
    // block while recurrent, attention and external activation shapes stay fixed.
    config.hf_config["text_config"]["intermediate_size"] = 256.into();
    for weight in &mut config.weights {
        match weight.role.as_str() {
            "mlp_gate" | "mlp_up" => weight.dimensions[0] = 256,
            "mlp_down" => {
                weight.dimensions[1] = 256;
                if weight.layer_index == Some(0) {
                    weight.source_encoding =
                        FamilyWeightSourceEncoding::BlockQuantized(BlockQuantizationSpec {
                            format_id: QuantizationFormatId::new("quantization.gguf.iq4-xs")
                                .unwrap(),
                            logical_values_per_block: 256,
                            bytes_per_block: 136,
                        });
                }
            }
            _ => {}
        }
    }
    config
}

#[test]
fn q8act_ffn_require_preserves_auto_state_boundaries_and_non_ffn_operations() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let config = source();
    let definition = registration
        .define(&serde_json::to_value(&config).unwrap())
        .unwrap();
    let catalog = definition.numerical_profiles();
    let id = NumericalProfileId::new(F32_MASTER_FFN_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID).unwrap();
    let strict_id = NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap();
    let q8 = catalog
        .candidates(
            &NumericalExecutionPolicy::Require(id.clone()),
            KvStorageFormat::F16,
        )
        .unwrap()[0];
    assert_eq!(q8.id, id);
    assert_eq!(
        catalog
            .candidates(&NumericalExecutionPolicy::Auto, KvStorageFormat::F16)
            .unwrap()
            .iter()
            .map(|p| &p.id)
            .collect::<Vec<_>>(),
        [&strict_id]
    );
    assert!(catalog
        .candidates(
            &NumericalExecutionPolicy::Require(id.clone()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    let strict = catalog.resolve(&strict_id).unwrap();
    assert_eq!(q8.boundaries, strict.boundaries);
    assert_eq!(q8.states, strict.states);
    assert_eq!(q8.kv_storage, strict.kv_storage);
    let mut restored = q8.clone();
    restored.id = strict.id.clone();
    let candidate_operation = restored
        .operations
        .iter_mut()
        .find(|op| {
            op.operation_id.as_str()
                == ferrum_interfaces::vnext::DENSE_SWIGLU_IQ4XS_Q8ACT_G32_OPERATION_ID
        })
        .unwrap();
    *candidate_operation = strict
        .operations
        .iter()
        .find(|op| op.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID)
        .unwrap()
        .clone();
    restored
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(
        &restored, strict,
        "the explicit profile changes only SwiGLU arithmetic"
    );
    let prepared = registration.prepare(&definition, &id).unwrap();
    let baseline = registration.prepare(&definition, &strict_id).unwrap();
    assert_eq!(prepared.weight_schema(), baseline.weight_schema());
    assert_eq!(prepared.canonical_config(), baseline.canonical_config());
    assert_eq!(prepared.program().states(), baseline.program().states());
    q8.validate_program(prepared.program()).unwrap();
    assert_ne!(
        prepared.program().fingerprint().unwrap(),
        baseline.program().fingerprint().unwrap()
    );
}

#[test]
fn q8act_ffn_rejects_mutated_arithmetic_and_unqualified_sources() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let mut config = source();
    let id = NumericalProfileId::new(F32_MASTER_FFN_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID).unwrap();
    let mut profile = provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&id)
        .unwrap()
        .clone();
    let leaf = &mut profile
        .operations
        .iter_mut()
        .find_map(|op| op.composite_arithmetic.as_mut())
        .unwrap()
        .projections[0]
        .leaves[0];
    leaf.shape.minimum_input_features *= 2;
    profile.validate().unwrap();
    assert!(provider.semantic_program(&config, &profile).is_err());
    for weight in &mut config.weights {
        if matches!(
            weight.source_encoding,
            FamilyWeightSourceEncoding::BlockQuantized(_)
        ) {
            weight.source_encoding = FamilyWeightSourceEncoding::Dense {
                element_type: ElementType::F16,
            };
        }
    }
    assert!(provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&id)
        .is_err());
    let mut config = source();
    config.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
    assert!(provider
        .numerical_profiles(&config)
        .unwrap()
        .resolve(&id)
        .is_err());
}

#[test]
fn q8act_three_format_require_changes_only_ffn_and_preserves_old_policy() {
    let registration = TypedFamilyRegistration::new(Qwen35FamilyProvider::new().unwrap());
    let definition = registration
        .define(&serde_json::to_value(source()).unwrap())
        .unwrap();
    let catalog = definition.numerical_profiles();
    let strict_id = NumericalProfileId::new(F32_MASTER_NUMERICAL_PROFILE_ID).unwrap();
    let old_id =
        NumericalProfileId::new(F32_MASTER_FFN_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID).unwrap();
    let new_id =
        NumericalProfileId::new(F32_MASTER_FFN_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID)
            .unwrap();
    let strict = catalog.resolve(&strict_id).unwrap();
    let old = catalog.resolve(&old_id).unwrap();
    let new = catalog
        .candidates(
            &NumericalExecutionPolicy::Require(new_id.clone()),
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
            &NumericalExecutionPolicy::Require(new_id.clone()),
            KvStorageFormat::Int8PerTokenHeadF32ScaleV1
        )
        .is_err());
    let old_arithmetic = old
        .operations
        .iter()
        .find_map(|op| op.composite_arithmetic.as_ref())
        .unwrap();
    let new_arithmetic = new
        .operations
        .iter()
        .find_map(|op| op.composite_arithmetic.as_ref())
        .unwrap();
    assert_eq!(old_arithmetic, &Q8ActSwiGluProfile::Iq4Xs.arithmetic());
    for (format, bytes) in [
        ("quantization.gguf.q4-k", 144),
        ("quantization.gguf.q5-k", 176),
    ] {
        let block = BlockQuantizationSpec {
            format_id: QuantizationFormatId::new(format).unwrap(),
            logical_values_per_block: 256,
            bytes_per_block: bytes,
        };
        assert!(matches!(
            old_arithmetic
                .declared_projection_arithmetic(
                    ProjectionRole::SwiGluDown,
                    Some(&block),
                    256,
                    4,
                    false
                )
                .unwrap(),
            DeclaredProjectionArithmetic::StrictBase(_)
        ));
        assert!(matches!(
            new_arithmetic
                .declared_projection_arithmetic(
                    ProjectionRole::SwiGluDown,
                    Some(&block),
                    256,
                    4,
                    false
                )
                .unwrap(),
            DeclaredProjectionArithmetic::Staged(_)
        ));
    }
    let mut restored = new.clone();
    restored.id = strict.id.clone();
    *restored
        .operations
        .iter_mut()
        .find(|op| op.composite_arithmetic.is_some())
        .unwrap() = strict
        .operations
        .iter()
        .find(|op| op.operation_id.as_str() == DENSE_SWIGLU_OPERATION_ID)
        .unwrap()
        .clone();
    restored
        .operations
        .sort_by(|a, b| a.operation_id.cmp(&b.operation_id));
    assert_eq!(
        &restored, strict,
        "new profile changes no non-FFN numerical contract"
    );
    let prepared = registration.prepare(&definition, &new_id).unwrap();
    let old_prepared = registration.prepare(&definition, &old_id).unwrap();
    assert_eq!(prepared.weight_schema(), old_prepared.weight_schema());
    assert_eq!(prepared.program().states(), old_prepared.program().states());
    new.validate_program(prepared.program()).unwrap();
    assert_ne!(new.fingerprint().unwrap(), old.fingerprint().unwrap());
}

#[test]
fn q8act_three_format_qualifies_affine_only_source_without_enabling_old_profile() {
    let provider = Qwen35FamilyProvider::new().unwrap();
    let old_id =
        NumericalProfileId::new(F32_MASTER_FFN_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID).unwrap();
    let new_id =
        NumericalProfileId::new(F32_MASTER_FFN_Q4K_Q5K_IQ4XS_Q8ACT_G32_NUMERICAL_PROFILE_ID)
            .unwrap();
    for (format, bytes) in [
        ("quantization.gguf.q4-k", 144),
        ("quantization.gguf.q5-k", 176),
    ] {
        let mut config = source();
        for weight in &mut config.weights {
            if let FamilyWeightSourceEncoding::BlockQuantized(spec) = &mut weight.source_encoding {
                spec.format_id = QuantizationFormatId::new(format).unwrap();
                spec.logical_values_per_block = 256;
                spec.bytes_per_block = bytes;
            }
        }
        let catalog = provider.numerical_profiles(&config).unwrap();
        assert!(catalog.resolve(&old_id).is_err());
        let profile = catalog.resolve(&new_id).unwrap();
        provider.semantic_program(&config, profile).unwrap();
        let mut tampered = profile.clone();
        tampered
            .operations
            .iter_mut()
            .find_map(|op| op.composite_arithmetic.as_mut())
            .unwrap()
            .projections[0]
            .leaves[0]
            .shape
            .minimum_input_features *= 2;
        tampered.validate().unwrap();
        assert!(provider.semantic_program(&config, &tampered).is_err());
        config.recurrent_weight_abi = RecurrentWeightAbi::LogRateGrouped;
        assert!(provider
            .numerical_profiles(&config)
            .unwrap()
            .resolve(&new_id)
            .is_err());
    }
}
