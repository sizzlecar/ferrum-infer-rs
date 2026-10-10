use super::*;

fn q6_policy() -> UpstreamProjectionPolicy {
    policy(
        ProjectionBlockFormat::Q6K,
        UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1,
        UpstreamProjectionLayout::Columns,
        &(1..=32).collect::<Vec<_>>(),
    )
}

#[test]
fn q6_f16_arithmetic_is_an_independent_boundary_with_explicit_marker_support() {
    let q6 = UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1;
    let semantics = q6.semantics(ProjectionBlockFormat::Q6K).unwrap();
    assert_eq!(
        semantics.pack,
        UpstreamActivationPack::MmqTransposedD4_144BytesPer128
    );
    assert_eq!(semantics.activation_scale_type, ElementType::F32);
    assert_eq!(
        semantics.coefficient_rule,
        UpstreamCoefficientRule::HalfBaseTimesIntegerInF32
    );
    assert_eq!(semantics.min_source, UpstreamMinSource::None);
    assert_eq!(semantics.reduction, UpstreamReduction::MmqMmaStreamKF32);
    let marker = semantics.marker.unwrap();
    assert_eq!(marker.canonical_f16_nan_bits, 0x7e00);
    assert!(marker.nonfinite_input_or_consumed_metadata_poison_entire_input_row);
    assert!(marker.nonfinite_consumed_weight_coefficient_poisons_entire_physical_leaf);
    assert!(marker.nonfinite_f32_result_or_f16_overflow_poisons_output_element);
    assert!(marker.zero_logical_and_padding_groups_use_positive_zero_metadata_and_codes);
    for format in [
        ProjectionBlockFormat::Q4K,
        ProjectionBlockFormat::Iq4Xs,
        ProjectionBlockFormat::Q3K,
    ] {
        assert!(q6.semantics(format).is_err());
    }
    for arithmetic in [
        UpstreamProjectionArithmetic::MmqD4MarkerV2,
        UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2,
        UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
    ] {
        assert!(arithmetic.semantics(ProjectionBlockFormat::Q6K).is_err());
    }
    let staged = q6_policy().staged().unwrap();
    assert_eq!(
        staged.schema_version,
        NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_Q6_F16
    );
    for version in [
        NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM,
        NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA,
        NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_GEOMETRY,
        NUMERICAL_ARITHMETIC_SCHEMA_VERSION_Q6_MMQ_F32,
    ] {
        let mut wrong = staged.clone();
        wrong.schema_version = version;
        assert!(wrong.validate().is_err());
    }
    let mut invalid = q6_policy();
    invalid.routes[0].layout = UpstreamProjectionLayout::Channels;
    assert!(invalid.validate().is_err());
    invalid = q6_policy();
    invalid.routes[0].local_rows.insert(33);
    assert!(invalid.validate().is_err());
    invalid = q6_policy();
    invalid.routes[0].prefill_rows = Some(UpstreamPrefillRows {
        first: 33,
        last: 64,
    });
    assert!(invalid.validate().is_err());
}

#[test]
fn q6_f16_profiles_preserve_all_inherited_leaves_and_strict_operation_ports() {
    for (new, old) in [
        (
            UpstreamMarkerV2Profile::SwiGluQ6F16,
            UpstreamMarkerV2Profile::SwiGluExtraAllRows,
        ),
        (
            UpstreamMarkerV2Profile::GatedDeltaQ6F16,
            UpstreamMarkerV2Profile::GatedDeltaM8Geometry,
        ),
        (
            UpstreamMarkerV2Profile::CausalQ6F16,
            UpstreamMarkerV2Profile::CausalM8Geometry,
        ),
    ] {
        let new_contract = new.arithmetic();
        new_contract.validate().unwrap();
        let old_contract = old.arithmetic();
        assert_ne!(new.operation_id(), old.operation_id());
        assert_ne!(new.capability_id(), old.capability_id());
        assert!(new.q6_f16() && new.extra_all_rows() && new.prefill());
        assert!(!old.q6_f16());
        assert_eq!(new.strict_operation_id(), old.strict_operation_id());
        let mut descriptor = new.contract().unwrap().descriptor().clone();
        let old_descriptor = old.contract().unwrap().descriptor().clone();
        descriptor.id = old_descriptor.id.clone();
        descriptor.provider = old_descriptor.provider.clone();
        assert_eq!(descriptor, old_descriptor);
        let mut restored = new_contract.clone();
        restored.schema_version = old_contract.schema_version;
        for projection in &mut restored.projections {
            projection
                .leaves
                .retain(|leaf| leaf.format != ProjectionBlockFormat::Q6K);
        }
        assert_eq!(
            serde_json::to_vec(&restored).unwrap(),
            serde_json::to_vec(&old_contract).unwrap()
        );
        let mut wrong = new_contract.clone();
        wrong.schema_version = old_contract.schema_version;
        assert!(wrong.validate().is_err());
        restored.schema_version = new_contract.schema_version;
        assert!(
            restored.validate().is_err(),
            "schema cannot falsely declare absent Q6 arithmetic"
        );
    }
}

#[test]
fn q6_f16_selection_uses_physical_leaf_dimensions_and_preserves_strict_edges() {
    let contract = UpstreamMarkerV2Profile::SwiGluQ6F16.arithmetic();
    let selected = contract.projections[0]
        .leaves
        .iter()
        .find(|leaf| leaf.format == ProjectionBlockFormat::Q6K)
        .unwrap()
        .arithmetic
        .upstream_policy()
        .unwrap();
    for (rows, k, n, expected) in [
        (3, 5120, 5120, false),
        (4, 5120, 5120, true),
        (4, 5120, 5119, false),
        (7, 5120, 1024, false),
        (8, 5120, 1024, true),
        (32, 5120, 1024, true),
        (33, 5120, 5120, false),
        (8, 4864, 5120, false),
        (8, 5121, 5120, false),
        (8, 5120, 1023, false),
    ] {
        assert_eq!(
            selected.select_arithmetic(UpstreamProjectionLayout::Columns, rows, k, n),
            expected.then_some(UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1)
        );
    }
    assert_eq!(
        selected.select_arithmetic(UpstreamProjectionLayout::Channels, 8, 5120, 5120),
        None
    );
    // ABI capability is broader than the product's performance selection.
    assert_eq!(
        q6_policy().select_arithmetic(UpstreamProjectionLayout::Columns, 1, 256, 1),
        Some(UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1)
    );
    let mut wrong = selected.clone();
    wrong.geometry_selection = Some(UpstreamProjectionGeometrySelection::Q6F16Mmq {
        minimum_input_features: 256,
        minimum_output_features: 1,
        minimum_rows_for_small_outputs: 0,
        minimum_output_features_for_small_rows: 1,
    });
    assert!(wrong.validate().is_err());
}

fn joined_ffn() -> (
    CompositeNumericalArithmetic,
    Vec<ResolvedValueBinding>,
    PreparedProjectionNumerics,
) {
    let contract = UpstreamMarkerV2Profile::SwiGluQ6F16.arithmetic();
    let values = vec![
        binding(
            1,
            &[
                Some("quantization.gguf.q4-k"),
                Some("quantization.gguf.q6-k"),
            ],
            5120,
            17408,
            false,
        ),
        binding(2, &[Some("quantization.gguf.q6-k")], 17408, 5120, false),
    ];
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
    (contract, values, prepared)
}

fn joined_facts(rows: u32) -> UpstreamProjectionWaveFacts {
    UpstreamProjectionWaveFacts {
        role: ProjectionRole::SwiGluGateUp,
        component_id: WeightId::new("component.1.1").unwrap(),
        local_rows: rows,
        layout: UpstreamProjectionLayout::Columns,
        input_stride: 5123,
        output_stride: 34816,
        input_byte_offset: 6,
        output_byte_offset: 10,
        weight_byte_offset: 12,
        input_available_bytes: u64::from(rows) * 5123 * 2,
        output_available_bytes: u64::from(rows) * 34816 * 2,
        weight_available_bytes: 17408 * 20 * 210,
        retained_zero_padded_weight_rows: 17408,
    }
}

fn joined_native(rows: u32) -> UpstreamNativePlanFacts {
    let mut n = native(rows, UpstreamProjectionLayout::Columns, 17408, true);
    n.implementation_fingerprint = "native.fixture.q6-f16.abi2".into();
    if let UpstreamNativeGeometry::Mmq { padded_inputs, .. } = &mut n.geometry {
        *padded_inputs = 5120;
    }
    n
}

#[test]
fn q6_f16_wave_owns_complete_strided_leaf_and_fresh_native_marker_requirements() {
    let (contract, values, prepared) = joined_ffn();
    let projection = prepared.projection(ProjectionRole::SwiGluGateUp).unwrap();
    assert_eq!(projection.leaves()[1].output_offset(), 17408);
    let facts = joined_facts(8);
    let native = joined_native(8);
    assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &facts, None).is_err());
    let wave = PreparedUpstreamProjectionWave::prepare(&prepared, &facts, Some(&native)).unwrap();
    let PreparedUpstreamProjectionRoute::Selected {
        arithmetic,
        scratch,
        scratch_bytes,
        retained_weight_validation,
        ..
    } = wave.route()
    else {
        panic!("Q6 selected")
    };
    assert_eq!(
        *arithmetic,
        UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1
    );
    let validation = retained_weight_validation.as_ref().unwrap();
    assert_eq!(validation.component_id, facts.component_id);
    assert_eq!((validation.byte_len, validation.alignment), (4, 4));
    assert!(scratch
        .windows(2)
        .all(|pair| pair[0].byte_offset + pair[0].byte_len <= pair[1].byte_offset));
    assert!(scratch
        .iter()
        .all(|r| r.byte_offset % 16 == 0 && r.byte_offset + r.byte_len <= *scratch_bytes));
    assert_eq!(
        scratch
            .iter()
            .find(|r| r.role == UpstreamScratchRole::RowPoisonFlags)
            .unwrap()
            .byte_len,
        8 * 4
    );
    wave.validate_reconstructed(&contract, &values, &facts, Some(&native))
        .unwrap();
    let available = BTreeSet::from([(ProjectionBlockFormat::Q6K, *arithmetic)]);
    wave.require_execution_support(
        &available,
        UpstreamDynamicDomainSupport::CanonicalNanMarkerV2Device,
    )
    .unwrap();
    assert!(wave
        .require_execution_support(&available, UpstreamDynamicDomainSupport::EnforcedBeforeDot)
        .is_err());
    assert!(wave
        .require_execution_support(
            &BTreeSet::new(),
            UpstreamDynamicDomainSupport::CanonicalNanMarkerV2Device
        )
        .is_err());
    let mut altered = native.clone();
    altered.implementation_fingerprint.push_str(".other");
    assert!(wave
        .validate_reconstructed(&contract, &values, &facts, Some(&altered))
        .is_err());
    let mut serialized = serde_json::to_value(&wave).unwrap();
    serialized["route"]["scratch"][1]["byte_offset"] = 0.into();
    let overlapping: PreparedUpstreamProjectionWave = serde_json::from_value(serialized).unwrap();
    assert!(overlapping
        .validate_reconstructed(&contract, &values, &facts, Some(&native))
        .is_err());
}

#[test]
fn q6_f16_wave_rejects_short_ranges_invalid_stride_and_native_tail_bounds() {
    let (_, _, prepared) = joined_ffn();
    let good = joined_facts(8);
    let native = joined_native(8);
    for change in [
        |f: &mut UpstreamProjectionWaveFacts| f.input_stride = 5119,
        |f: &mut UpstreamProjectionWaveFacts| f.output_stride = 34815,
        |f: &mut UpstreamProjectionWaveFacts| f.input_byte_offset += 1,
        |f: &mut UpstreamProjectionWaveFacts| f.output_byte_offset += 1,
        |f: &mut UpstreamProjectionWaveFacts| f.input_available_bytes = (7 * 5123 + 5120) * 2 - 1,
        |f: &mut UpstreamProjectionWaveFacts| f.output_available_bytes -= 1,
        |f: &mut UpstreamProjectionWaveFacts| f.weight_available_bytes -= 1,
        |f: &mut UpstreamProjectionWaveFacts| f.output_byte_offset = u64::MAX - 1,
    ] {
        let mut bad = good.clone();
        change(&mut bad);
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &bad, Some(&native)).is_err());
    }
    for change in [
        |n: &mut UpstreamNativePlanFacts| {
            if let UpstreamNativeGeometry::Mmq { padded_inputs, .. } = &mut n.geometry {
                *padded_inputs = 5121;
            }
        },
        |n: &mut UpstreamNativePlanFacts| {
            if let UpstreamNativeGeometry::Mmq {
                packed_guard_blocks,
                ..
            } = &mut n.geometry
            {
                *packed_guard_blocks = 0;
            }
        },
    ] {
        // M7 leaves unused cooperative-load lanes in the final tile.
        let mut bad = joined_native(7);
        change(&mut bad);
        assert!(
            PreparedUpstreamProjectionWave::prepare(&prepared, &joined_facts(7), Some(&bad))
                .is_err()
        );
    }
    let mut misaligned = good.clone();
    misaligned.weight_byte_offset += 2;
    assert!(matches!(
        PreparedUpstreamProjectionWave::prepare(&prepared, &misaligned, None)
            .unwrap()
            .route(),
        PreparedUpstreamProjectionRoute::StrictBase {
            reason: UpstreamStrictReason::WeightAlignmentNotSupported
        }
    ));
    for rows in [1, 3, 33] {
        assert!(matches!(
            PreparedUpstreamProjectionWave::prepare(&prepared, &joined_facts(rows), None)
                .unwrap()
                .route(),
            PreparedUpstreamProjectionRoute::StrictBase {
                reason: UpstreamStrictReason::RowsOrLayoutNotDeclared
            }
        ));
    }
}

#[test]
fn q6_f16_small_output_fallback_needs_no_native_plan_but_eligible_rows_do() {
    let contract = UpstreamMarkerV2Profile::CausalQ6F16.arithmetic();
    let values = vec![
        binding(2, &[Some("quantization.gguf.q6-k")], 5120, 1024, false),
        binding(3, &[None], 5120, 1024, false),
        binding(4, &[None], 5120, 1024, false),
        binding(5, &[None], 5120, 1024, false),
    ];
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
    for rows in [1, 3, 4, 7, 8, 32, 33] {
        let mut f = facts(rows, UpstreamProjectionLayout::Columns, 1024);
        f.input_stride = 5120;
        f.input_available_bytes = u64::from(rows) * 5120 * 2;
        f.weight_available_bytes = 1024 * 20 * 210;
        let without_native = PreparedUpstreamProjectionWave::prepare(&prepared, &f, None);
        if matches!(rows, 8..=32) {
            assert!(without_native.is_err());
            let n = joined_native(rows);
            assert!(matches!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n))
                    .unwrap()
                    .route(),
                PreparedUpstreamProjectionRoute::Selected {
                    arithmetic: UpstreamProjectionArithmetic::MmqD4Q6F16MarkerV1,
                    ..
                }
            ));
        } else {
            assert!(matches!(
                without_native.unwrap().route(),
                PreparedUpstreamProjectionRoute::StrictBase {
                    reason: UpstreamStrictReason::RowsOrLayoutNotDeclared
                }
            ));
        }
    }
}
