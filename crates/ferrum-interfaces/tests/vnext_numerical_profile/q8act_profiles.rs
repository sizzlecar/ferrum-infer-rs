use super::*;

fn profile(
    selected: Q8ActSwiGluProfile,
    arithmetic: CompositeNumericalArithmetic,
) -> NumericalExecutionProfile {
    NumericalExecutionProfile {
        id: id("fixture.q8act-policy"),
        version: ContractVersion::new(1, 0),
        family_id: id("family.fixture.q8act-policy"),
        primary_activation: id("value.output"),
        boundaries: BTreeMap::from([(id("value.output"), ElementType::F16)]),
        states: vec![],
        kv_storage: vec![],
        operations: vec![NumericalOperationContract {
            operation_id: id(selected.operation_id()),
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: None,
            composite_arithmetic: Some(arithmetic),
        }],
    }
}

fn block(format: &str, bytes: u32) -> BlockQuantizationSpec {
    BlockQuantizationSpec {
        format_id: id(format),
        logical_values_per_block: 256,
        bytes_per_block: bytes,
    }
}

#[test]
fn iq4_only_q8act_wire_and_numerical_fingerprint_remain_unchanged() {
    // A fixed wire fixture is independent of the shared/new constructor, so a
    // simultaneous accidental expansion of both builders cannot pass this test.
    let legacy: CompositeNumericalArithmetic =
        serde_json::from_str(include_str!("iq4xs_q8act_legacy.json")).unwrap();
    let selected = Q8ActSwiGluProfile::Iq4Xs;
    let actual = selected.arithmetic();
    assert_eq!(
        serde_json::to_vec(&actual).unwrap(),
        serde_json::to_vec(&legacy).unwrap()
    );
    let expected = profile(selected, legacy);
    let current = profile(selected, actual.clone());
    assert_eq!(
        current.fingerprint().unwrap(),
        expected.fingerprint().unwrap()
    );
    assert_eq!(
        selected.operation_id(),
        DENSE_SWIGLU_IQ4XS_Q8ACT_G32_OPERATION_ID
    );
    assert_eq!(
        selected.capability_id(),
        DENSE_SWIGLU_IQ4XS_Q8ACT_G32_CAPABILITY_ID
    );
    for physical in [
        block("quantization.gguf.q4-k", 144),
        block("quantization.gguf.q5-k", 176),
    ] {
        for role in [ProjectionRole::SwiGluGateUp, ProjectionRole::SwiGluDown] {
            assert_eq!(
                actual
                    .declared_projection_arithmetic(role, Some(&physical), 5120, 17408, false)
                    .unwrap(),
                DeclaredProjectionArithmetic::StrictBase(StrictProjectionReason::FormatNotDeclared)
            );
        }
    }
}

#[test]
fn three_format_q8act_has_distinct_identity_and_exact_affine_abis() {
    let selected = Q8ActSwiGluProfile::Q4KQ5KIq4Xs;
    let old = Q8ActSwiGluProfile::Iq4Xs;
    assert_ne!(selected.operation_id(), old.operation_id());
    assert_ne!(selected.capability_id(), old.capability_id());
    let strict = dense_swiglu_contract().unwrap();
    let contract = selected.contract().unwrap();
    assert_eq!(contract.descriptor().id.as_str(), selected.operation_id());
    assert_eq!(contract.descriptor().version, ContractVersion::new(1, 0));
    assert_eq!(contract.descriptor().inputs, strict.descriptor().inputs);
    assert_eq!(contract.descriptor().outputs, strict.descriptor().outputs);
    assert_eq!(
        contract.descriptor().attributes,
        strict.descriptor().attributes
    );
    assert_eq!(
        contract.descriptor().resources,
        strict.descriptor().resources
    );
    let arithmetic = selected.arithmetic();
    arithmetic.validate().unwrap();
    for projection in &arithmetic.projections {
        assert_eq!(
            projection
                .leaves
                .iter()
                .map(|leaf| leaf.format)
                .collect::<Vec<_>>(),
            [
                ProjectionBlockFormat::Q4K,
                ProjectionBlockFormat::Q5K,
                ProjectionBlockFormat::Iq4Xs
            ]
        );
        for (index, (format, bytes)) in [
            ("quantization.gguf.q4-k", 144),
            ("quantization.gguf.q5-k", 176),
            ("quantization.gguf.iq4-xs", 136),
        ]
        .into_iter()
        .enumerate()
        {
            let exact = block(format, bytes);
            assert_eq!(
                arithmetic
                    .declared_projection_arithmetic(projection.role, Some(&exact), 256, 1, false)
                    .unwrap(),
                DeclaredProjectionArithmetic::Staged(&projection.leaves[index].arithmetic)
            );
            assert!(matches!(
                projection.leaves[index].arithmetic.stages[1],
                NumericalArithmeticStage::IntegerDot {
                    values_per_partial: 32,
                    ..
                }
            ));
            for wrong in [
                block(format, bytes + 1),
                BlockQuantizationSpec {
                    logical_values_per_block: 128,
                    ..exact.clone()
                },
            ] {
                assert_eq!(
                    arithmetic
                        .declared_projection_arithmetic(
                            projection.role,
                            Some(&wrong),
                            256,
                            1,
                            false
                        )
                        .unwrap(),
                    DeclaredProjectionArithmetic::StrictBase(
                        StrictProjectionReason::FormatNotDeclared
                    )
                );
            }
        }
    }
    let wire = serde_json::to_value(&arithmetic).unwrap();
    assert_eq!(wire["projections"][0]["leaves"][1]["format"], json!("q5_k"));
    let restored: CompositeNumericalArithmetic = serde_json::from_value(wire).unwrap();
    assert_eq!(restored, arithmetic);
    assert_ne!(
        profile(selected, arithmetic).fingerprint().unwrap(),
        profile(old, old.arithmetic()).fingerprint().unwrap()
    );
}

#[test]
fn affine_q8act_rejects_missing_or_noninteger_code_sum_and_wrong_group_extent() {
    let base = Q8ActSwiGluProfile::Q4KQ5KIq4Xs.arithmetic();
    for projection_index in 0..2 {
        for leaf_index in 0..2 {
            for wrong in [
                AffineMinCorrection::None {},
                AffineMinCorrection::QuantizedPartialSum {
                    accumulation_type: ElementType::F32,
                },
            ] {
                let mut changed = base.clone();
                let NumericalArithmeticStage::Rescale { min_correction, .. } =
                    &mut changed.projections[projection_index].leaves[leaf_index]
                        .arithmetic
                        .stages[2]
                else {
                    unreachable!()
                };
                *min_correction = wrong;
                assert!(changed.validate().is_err());
                assert!(profile(Q8ActSwiGluProfile::Q4KQ5KIq4Xs, changed)
                    .fingerprint()
                    .is_err());
            }
            let mut wrong_extent = base.clone();
            let stages = &mut wrong_extent.projections[projection_index].leaves[leaf_index]
                .arithmetic
                .stages;
            if let NumericalArithmeticStage::ActivationQuantization { group_values, .. } =
                &mut stages[0]
            {
                *group_values = 64;
            }
            if let NumericalArithmeticStage::IntegerDot {
                values_per_partial, ..
            } = &mut stages[1]
            {
                *values_per_partial = 64;
            }
            assert!(
                wrong_extent.validate().is_err(),
                "a partial must not cross weight coefficient groups"
            );
        }
    }
    let mut iq4_with_affine = base.clone();
    if let NumericalArithmeticStage::Rescale { min_correction, .. } =
        &mut iq4_with_affine.projections[0].leaves[2].arithmetic.stages[2]
    {
        *min_correction = AffineMinCorrection::QuantizedPartialSum {
            accumulation_type: ElementType::I32,
        };
    }
    assert!(iq4_with_affine.validate().is_err());
    let mut duplicate = base.clone();
    duplicate.projections[0]
        .leaves
        .push(base.projections[0].leaves[1].clone());
    assert!(duplicate.validate().is_err());
    for key in ["original_activation_sum", "different_code_group"] {
        let mut wire = serde_json::to_value(&base).unwrap();
        wire["projections"][0]["leaves"][1]["arithmetic"]["stages"][2]["min_correction"][key] =
            json!(true);
        assert!(serde_json::from_value::<CompositeNumericalArithmetic>(wire).is_err());
    }
}
