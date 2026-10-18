use super::*;

pub(super) fn arithmetic() -> StagedNumericalArithmetic {
    StagedNumericalArithmetic {
        schema_version: NUMERICAL_ARITHMETIC_SCHEMA_VERSION,
        stages: vec![
            NumericalArithmeticStage::ActivationQuantization {
                input_type: ElementType::F16,
                code_type: ElementType::I8,
                scale_type: ElementType::F32,
                group_values: 32,
                max_code: 127,
                scale_rule: ActivationScaleRule::AbsMaxOverMaxCode,
                rounding: IntegerQuantizationRounding::NearestTiesAwayFromZero,
                zero: ZeroQuantizationPolicy::PositiveZeroScaleAndZeroCodes,
                non_finite: NonFiniteQuantizationPolicy::NanScaleAndZeroCodes,
            },
            NumericalArithmeticStage::IntegerDot {
                activation_type: ElementType::I8,
                weight_type: ElementType::I8,
                accumulation_type: ElementType::I32,
                values_per_partial: 4,
            },
            NumericalArithmeticStage::Rescale {
                integer_input_type: ElementType::I32,
                activation_scale_type: ElementType::F32,
                weight_coefficient_type: ElementType::F32,
                arithmetic_type: ElementType::F32,
                min_correction: AffineMinCorrection::QuantizedPartialSum {
                    accumulation_type: ElementType::I32,
                },
                contraction: FloatingPointContraction::Disallowed,
            },
            NumericalArithmeticStage::FloatingReduction {
                input_type: ElementType::F32,
                accumulation_type: ElementType::F32,
                order: FloatingReductionOrder::OperationDefined,
            },
            NumericalArithmeticStage::OutputRounding {
                input_type: ElementType::F32,
                output_type: ElementType::F16,
                rounding: FloatingStorageRounding::NearestTiesToEven,
            },
        ],
    }
}

fn family() -> Family {
    Family {
        family_id: id("family.numerical-fixture"),
        program_calls: Arc::default(),
        corrupt_physical_identity: false,
    }
}

fn staged_profile() -> NumericalExecutionProfile {
    let mut profile = family().profile("fixture.quantized-projection", ElementType::F16);
    let operation = &mut profile.operations[0];
    // A changed arithmetic policy receives a distinct versioned identity;
    // this fixture does not register it as an executable operation/provider.
    operation.operation_id = id("operation.fixture.quantized-projection");
    operation.multiplication_type = None;
    operation.accumulation_type = None;
    operation.staged_arithmetic = Some(arithmetic());
    profile
}

#[test]
fn legacy_float_json_and_fingerprint_remain_unchanged() {
    // Canonical pre-stages bytes, including actual old field order and an
    // absent new field. Verify decoding AND byte/fingerprint compatibility.
    let legacy = r#"{"id":"fixture.legacy","version":{"major":1,"minor":0},"family_id":"family.fixture","primary_activation":"value.output","boundaries":{"value.output":"f16"},"states":[],"kv_storage":[],"operations":[{"operation_id":"operation.fixture","version":{"major":1,"minor":0},"multiplication_type":"f32","accumulation_type":"f32"}]}"#;
    let profile: NumericalExecutionProfile = serde_json::from_str(legacy).unwrap();
    profile.validate().unwrap();
    assert_eq!(serde_json::to_string(&profile).unwrap(), legacy);
    assert_eq!(
        profile.fingerprint().unwrap(),
        format!("{:x}", Sha256::digest(legacy.as_bytes()))
    );
    assert!(profile.operations[0].staged_arithmetic.is_none());
    assert!(profile.operations[0].composite_arithmetic.is_none());
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyOperationReader {
    operation_id: OperationId,
    version: ContractVersion,
    multiplication_type: Option<ElementType>,
    accumulation_type: Option<ElementType>,
}

#[test]
fn old_reader_rejects_stages_instead_of_silently_reading_a_float_summary() {
    let legacy = family().profile("fixture.f16", ElementType::F16);
    let old: LegacyOperationReader =
        serde_json::from_value(serde_json::to_value(&legacy.operations[0]).unwrap()).unwrap();
    assert_eq!(old.operation_id, legacy.operations[0].operation_id);
    assert_eq!(old.version, legacy.operations[0].version);
    assert_eq!(
        old.multiplication_type,
        legacy.operations[0].multiplication_type
    );
    assert_eq!(
        old.accumulation_type,
        legacy.operations[0].accumulation_type
    );
    let staged = staged_profile();
    assert!(serde_json::from_value::<LegacyOperationReader>(
        serde_json::to_value(&staged.operations[0]).unwrap()
    )
    .is_err());
}

#[test]
fn staged_projection_roundtrip_preserves_actual_dot4_and_group32_boundaries() {
    let profile = staged_profile();
    profile.validate().unwrap();
    let wire = serde_json::to_value(&profile).unwrap();
    let stages = &wire["operations"][0]["staged_arithmetic"]["stages"];
    assert_eq!(stages[0]["group_values"], 32);
    assert_eq!(stages[1]["values_per_partial"], 4);
    assert_eq!(stages[1]["accumulation_type"], "i32");
    assert_eq!(stages[2]["arithmetic_type"], "f32");
    assert_eq!(stages[4]["output_type"], "f16");
    let restored: NumericalExecutionProfile = serde_json::from_value(wire).unwrap();
    assert_eq!(restored, profile);
    assert_eq!(
        restored.fingerprint().unwrap(),
        profile.fingerprint().unwrap()
    );
}

#[test]
fn numerical_policy_changes_have_distinct_fingerprints() {
    let base = staged_profile();
    let fingerprint = base.fingerprint().unwrap();
    let changes: [fn(&mut Vec<NumericalArithmeticStage>); 5] = [
        |stages| {
            if let NumericalArithmeticStage::Rescale { contraction, .. } = &mut stages[2] {
                *contraction = FloatingPointContraction::Permitted;
            }
        },
        |stages| {
            if let NumericalArithmeticStage::ActivationQuantization { rounding, .. } =
                &mut stages[0]
            {
                *rounding = IntegerQuantizationRounding::NearestTiesToEven;
            }
        },
        |stages| {
            if let NumericalArithmeticStage::IntegerDot {
                values_per_partial, ..
            } = &mut stages[1]
            {
                *values_per_partial = 32;
            }
        },
        |stages| {
            if let NumericalArithmeticStage::Rescale { min_correction, .. } = &mut stages[2] {
                *min_correction = AffineMinCorrection::None {};
            }
        },
        |stages| {
            if let NumericalArithmeticStage::FloatingReduction { order, .. } = &mut stages[3] {
                *order = FloatingReductionOrder::Sequential;
            }
        },
    ];
    for change in changes {
        let mut candidate = base.clone();
        change(
            &mut candidate.operations[0]
                .staged_arithmetic
                .as_mut()
                .unwrap()
                .stages,
        );
        candidate.validate().unwrap();
        assert_ne!(candidate.fingerprint().unwrap(), fingerprint);
    }
    // Schema identity is not a claim of equal output bits across these policies.
    assert_eq!(base.fingerprint().unwrap(), fingerprint);
}

#[test]
fn stages_reject_missing_reordered_disconnected_and_overflowing_arithmetic() {
    let invalid: [fn(&mut StagedNumericalArithmetic); 13] = [
        |a| a.schema_version += 1,
        |a| {
            a.stages.remove(2);
        },
        |a| a.stages.swap(1, 2),
        |a| {
            if let NumericalArithmeticStage::ActivationQuantization { scale_type, .. } =
                &mut a.stages[0]
            {
                *scale_type = ElementType::F16;
            }
        },
        |a| {
            if let NumericalArithmeticStage::ActivationQuantization { max_code, .. } =
                &mut a.stages[0]
            {
                *max_code = 128;
            }
        },
        |a| {
            if let NumericalArithmeticStage::ActivationQuantization { group_values, .. } =
                &mut a.stages[0]
            {
                *group_values = 0;
            }
        },
        |a| {
            if let NumericalArithmeticStage::IntegerDot {
                values_per_partial, ..
            } = &mut a.stages[1]
            {
                *values_per_partial = 0;
            }
        },
        |a| {
            if let NumericalArithmeticStage::IntegerDot {
                accumulation_type, ..
            } = &mut a.stages[1]
            {
                *accumulation_type = ElementType::F32;
            }
        },
        |a| {
            if let NumericalArithmeticStage::IntegerDot {
                values_per_partial, ..
            } = &mut a.stages[1]
            {
                *values_per_partial = 3;
            }
        },
        |a| {
            if let NumericalArithmeticStage::Rescale { min_correction, .. } = &mut a.stages[2] {
                *min_correction = AffineMinCorrection::QuantizedPartialSum {
                    accumulation_type: ElementType::F32,
                };
            }
        },
        |a| {
            if let NumericalArithmeticStage::FloatingReduction {
                accumulation_type, ..
            } = &mut a.stages[3]
            {
                *accumulation_type = ElementType::I32;
            }
        },
        |a| {
            if let NumericalArithmeticStage::OutputRounding { input_type, .. } = &mut a.stages[4] {
                *input_type = ElementType::I32;
            }
        },
        |a| {
            if let NumericalArithmeticStage::ActivationQuantization { group_values, .. } =
                &mut a.stages[0]
            {
                *group_values = u32::MAX;
            }
            if let NumericalArithmeticStage::IntegerDot {
                values_per_partial, ..
            } = &mut a.stages[1]
            {
                *values_per_partial = u32::MAX;
            }
        },
    ];
    for mutate in invalid {
        let mut profile = staged_profile();
        mutate(profile.operations[0].staged_arithmetic.as_mut().unwrap());
        assert!(
            profile.validate().is_err(),
            "invalid stages accepted: {:?}",
            profile.operations[0]
        );
        assert!(profile.fingerprint().is_err());
    }
}

#[test]
fn local_dot_i32_bound_accepts_its_limit_and_rejects_the_next_value() {
    let safe = (i32::MAX as u64 / (127 * 128)) as u32;
    for values in [safe, safe + 1] {
        let mut stages = arithmetic();
        if let NumericalArithmeticStage::ActivationQuantization { group_values, .. } =
            &mut stages.stages[0]
        {
            *group_values = values;
        }
        if let NumericalArithmeticStage::IntegerDot {
            values_per_partial, ..
        } = &mut stages.stages[1]
        {
            *values_per_partial = values;
        }
        assert_eq!(stages.validate().is_ok(), values == safe);
    }
}

#[test]
fn integer_pair_and_ambiguous_old_summary_remain_invalid() {
    let mut legacy = family().profile("fixture.float", ElementType::F16);
    legacy.operations[0].multiplication_type = Some(ElementType::I8);
    legacy.operations[0].accumulation_type = Some(ElementType::I32);
    assert!(legacy.validate().is_err());
    let mut staged = staged_profile();
    staged.operations[0].multiplication_type = Some(ElementType::F32);
    assert!(staged.validate().is_err());
}

#[test]
fn unknown_stage_fields_and_tags_are_not_silently_discarded() {
    let wire = serde_json::to_value(arithmetic()).unwrap();
    let mut unknown_field = wire.clone();
    unknown_field["stages"][1]["hidden_rescale"] = json!("f16");
    assert!(serde_json::from_value::<StagedNumericalArithmetic>(unknown_field).is_err());
    let mut unknown_stage = wire;
    unknown_stage["stages"][1]["stage"] = json!("unversioned_mma");
    assert!(serde_json::from_value::<StagedNumericalArithmetic>(unknown_stage).is_err());
}

#[test]
fn schema_one_rejects_input_types_without_a_finite_scale_underflow_policy() {
    for dtype in [ElementType::F32, ElementType::Bf16] {
        let mut profile = staged_profile();
        if let NumericalArithmeticStage::ActivationQuantization { input_type, .. } = &mut profile
            .operations[0]
            .staged_arithmetic
            .as_mut()
            .unwrap()
            .stages[0]
        {
            *input_type = dtype;
        }
        assert!(profile.validate().is_err());
        assert!(profile.fingerprint().is_err());
    }
    // F64 is not an ElementType at all; reject the wire spelling rather than
    // widening the shared storage enum solely for a negative test.
    let mut wire = serde_json::to_value(arithmetic()).unwrap();
    wire["stages"][0]["input_type"] = json!("f64");
    assert!(serde_json::from_value::<StagedNumericalArithmetic>(wire).is_err());
}

#[test]
fn no_min_correction_keeps_its_wire_shape_and_rejects_hidden_fields() {
    let none = AffineMinCorrection::None {};
    assert_eq!(serde_json::to_value(none).unwrap(), json!({"kind":"none"}));
    assert_eq!(
        serde_json::from_value::<AffineMinCorrection>(json!({"kind":"none"})).unwrap(),
        none
    );
    let mut wire = serde_json::to_value(arithmetic()).unwrap();
    wire["stages"][2]["min_correction"] = json!({"kind":"none"});
    let decoded: StagedNumericalArithmetic = serde_json::from_value(wire.clone()).unwrap();
    decoded.validate().unwrap();
    wire["stages"][2]["min_correction"]["hidden_rescale"] = json!("f16");
    assert!(serde_json::from_value::<StagedNumericalArithmetic>(wire).is_err());
}

#[test]
fn staged_store_dtype_must_match_the_actual_program_boundary() {
    let family = family();
    let mut profile = staged_profile();
    let program = family
        .semantic_program(&Config { width: 256 }, &profile)
        .unwrap();
    profile.validate_program(&program).unwrap();
    if let NumericalArithmeticStage::OutputRounding { output_type, .. } = &mut profile.operations[0]
        .staged_arithmetic
        .as_mut()
        .unwrap()
        .stages[4]
    {
        *output_type = ElementType::F32;
    }
    profile.validate().unwrap();
    assert!(profile.validate_program(&program).is_err());
}
