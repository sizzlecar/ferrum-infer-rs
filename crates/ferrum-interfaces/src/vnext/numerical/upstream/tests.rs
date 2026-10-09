use super::*;
use crate::vnext::*;
use std::collections::BTreeSet;

use super::super::prepared::tests::binding;

#[path = "extra_tests.rs"]
mod extra;

#[test]
fn upstream_prefill_interval_preserves_legacy_wire_and_rejects_ambiguous_domains() {
    for (old, new) in [
        (
            UpstreamMarkerV2Profile::SwiGlu,
            UpstreamMarkerV2Profile::SwiGluPrefill,
        ),
        (
            UpstreamMarkerV2Profile::GatedDelta,
            UpstreamMarkerV2Profile::GatedDeltaPrefill,
        ),
        (
            UpstreamMarkerV2Profile::Causal,
            UpstreamMarkerV2Profile::CausalPrefill,
        ),
    ] {
        let original = old.arithmetic();
        let wire = serde_json::to_string(&original).unwrap();
        assert!(!wire.contains("prefill_rows"));
        assert_eq!(
            serde_json::from_str::<CompositeNumericalArithmetic>(&wire).unwrap(),
            original
        );
        let mut expanded = new.arithmetic();
        expanded.validate().unwrap();
        for projection in &mut expanded.projections {
            for leaf in &mut projection.leaves {
                let [NumericalArithmeticStage::UpstreamProjection { policy }] =
                    leaf.arithmetic.stages.as_mut_slice()
                else {
                    panic!("upstream")
                };
                for m in [1, 2, 4, 8, 16, 32, 33, 34, 64, 127, 2048] {
                    assert_eq!(
                        policy.routes.iter().filter(|r| r.contains_rows(m)).count(),
                        1
                    );
                }
                assert!(!policy.routes.iter().any(|r| r.contains_rows(2049)));
                let range = policy.routes.pop().unwrap();
                assert_eq!(
                    range.prefill_rows,
                    Some(UpstreamPrefillRows {
                        first: 33,
                        last: 2048
                    })
                );
                let mut invalid = policy.clone();
                invalid.routes.extend([range.clone(), range.clone()]);
                assert!(invalid.validate().is_err());
                for (first, last) in [(0, 32), (32, 33), (33, 2049), (64, 33)] {
                    let mut invalid = policy.clone();
                    let mut r = range.clone();
                    r.prefill_rows = Some(UpstreamPrefillRows { first, last });
                    invalid.routes.push(r);
                    assert!(invalid.validate().is_err());
                }
                for arithmetic in [
                    UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
                    UpstreamProjectionArithmetic::MmqD4V1,
                ] {
                    let mut invalid = policy.clone();
                    let mut r = range.clone();
                    r.arithmetic = arithmetic;
                    invalid.routes.push(r);
                    assert!(invalid.validate().is_err());
                }
            }
        }
        assert_eq!(
            serde_json::to_string(&expanded).unwrap(),
            wire,
            "old route serialization is unchanged"
        );
        assert_ne!(old.operation_id(), new.operation_id());
    }
}

#[test]
fn upstream_prefill_wave_requires_fixed_geometry_and_bounded_scratch() {
    let contract = UpstreamMarkerV2Profile::CausalPrefill.arithmetic();
    let values = vec![
        binding(2, &[Some("quantization.gguf.iq4-xs")], 256, 17, false),
        binding(3, &[None], 256, 17, false),
        binding(4, &[None], 256, 17, false),
        binding(5, &[None], 256, 17, false),
    ];
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
    for rows in [33, 64, 128, 256, 2048] {
        let f = facts(rows, UpstreamProjectionLayout::Columns, 17);
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, None).is_err());
        let mut n = native(rows, f.layout, 17, true);
        n.geometry = UpstreamNativeGeometry::Mmq {
            padded_inputs: 512,
            row_tile: 32,
            column_tile: 128,
            threads: 256,
            shared_bytes: 32768,
            packed_guard_blocks: rows.min(512) / 8 * 8,
            blocks: n.multiprocessors,
            fixup: true,
        };
        let wave = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
        let PreparedUpstreamProjectionRoute::Selected { scratch_bytes, .. } = wave.route() else {
            panic!("selected")
        };
        assert!(
            *scratch_bytes
                <= UpstreamScratchEstimate::mmq_prefill(256, 17, n.multiprocessors)
                    .unwrap()
                    .bytes(u64::from(rows))
                    .unwrap()
        );
        if let UpstreamNativeGeometry::Mmq { row_tile, .. } = &mut n.geometry {
            *row_tile = 16;
        }
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).is_err());
    }
    let f = facts(2049, UpstreamProjectionLayout::Columns, 17);
    assert!(matches!(
        PreparedUpstreamProjectionWave::prepare(&prepared, &f, None)
            .unwrap()
            .route(),
        PreparedUpstreamProjectionRoute::StrictBase {
            reason: UpstreamStrictReason::RowsOrLayoutNotDeclared
        }
    ));
}

fn policy(
    format: ProjectionBlockFormat,
    arithmetic: UpstreamProjectionArithmetic,
    layout: UpstreamProjectionLayout,
    rows: &[u32],
) -> UpstreamProjectionPolicy {
    UpstreamProjectionPolicy {
        format,
        routes: vec![UpstreamProjectionRouteDeclaration {
            arithmetic,
            layout,
            local_rows: rows.iter().copied().collect(),
            prefill_rows: None,
        }],
        fallback: StrictProjectionFallback::RetainBaseArithmetic {},
    }
}

fn fixture(
    arithmetic: UpstreamProjectionArithmetic,
    layout: UpstreamProjectionLayout,
    rows: &[u32],
    n: u64,
) -> (
    CompositeNumericalArithmetic,
    Vec<ResolvedValueBinding>,
    PreparedProjectionNumerics,
) {
    let mut contract = Q8ActAttentionProfile::Causal.arithmetic();
    contract.schema_version = COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM;
    for projection in &mut contract.projections {
        projection.leaves = vec![QuantizedProjectionLeafContract {
            format: ProjectionBlockFormat::Iq4Xs,
            shape: ProjectionShapeEligibility {
                minimum_input_features: 256,
                minimum_output_features: 1,
                input_features_multiple: 256,
            },
            arithmetic: policy(ProjectionBlockFormat::Iq4Xs, arithmetic, layout, rows)
                .staged()
                .unwrap(),
        }];
    }
    let values = vec![
        binding(2, &[Some("quantization.gguf.iq4-xs")], 256, n, false),
        binding(3, &[Some("quantization.gguf.q6-k")], 256, n, false),
        binding(4, &[None], 256, n, false),
        binding(5, &[Some("quantization.gguf.iq4-xs")], 256, n, true),
    ];
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
    (contract, values, prepared)
}

fn facts(rows: u32, layout: UpstreamProjectionLayout, n: u64) -> UpstreamProjectionWaveFacts {
    UpstreamProjectionWaveFacts {
        role: ProjectionRole::CausalQuery,
        component_id: WeightId::new("component.2.0").unwrap(),
        local_rows: rows,
        layout,
        input_stride: 259,
        output_stride: n + 3,
        input_byte_offset: 6,
        output_byte_offset: 10,
        weight_byte_offset: 12,
        input_available_bytes: u64::from(rows) * 259 * 2,
        output_available_bytes: u64::from(rows) * (n + 3) * 2,
        weight_available_bytes: (n + 3) * 136,
        retained_zero_padded_weight_rows: n + 3,
    }
}

fn native(
    rows: u32,
    layout: UpstreamProjectionLayout,
    n: u32,
    mmq: bool,
) -> UpstreamNativePlanFacts {
    let geometry = if mmq {
        UpstreamNativeGeometry::Mmq {
            padded_inputs: 512,
            row_tile: 8,
            column_tile: 128,
            threads: 128,
            shared_bytes: 32768,
            packed_guard_blocks: rows / 8 * 8,
            blocks: 4,
            fixup: true,
        }
    } else {
        UpstreamNativeGeometry::Mmvq {
            padded_inputs: 512,
            padded_outputs: (n + 1) / 2 * 2,
            columns: if layout == UpstreamProjectionLayout::Columns {
                rows
            } else {
                1
            },
            channels: if layout == UpstreamProjectionLayout::Channels {
                rows
            } else {
                1
            },
            warps: 4,
            rows_per_block: 2,
        }
    };
    UpstreamNativePlanFacts {
        implementation_fingerprint: "native.fixture.abi-v1".into(),
        device_architecture: 1200,
        multiprocessors: 128,
        geometry,
        maximum_dynamic_shared_bytes: 65536,
    }
}

#[test]
fn upstream_marker_v2_partial_rows_prepare_native_and_keep_large_prefill_strict() {
    let contract = UpstreamMarkerV2Profile::Causal.arithmetic();
    for format in [
        ProjectionBlockFormat::Iq4Xs,
        ProjectionBlockFormat::Q4K,
        ProjectionBlockFormat::Q5K,
    ] {
        let values = vec![
            binding(2, &[Some(format.abi().0)], 256, 17, false),
            binding(3, &[None], 256, 17, false),
            binding(4, &[None], 256, 17, false),
            binding(5, &[None], 256, 17, false),
        ];
        let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
        let mut previous = contract.clone();
        for projection in &mut previous.projections {
            for leaf in &mut projection.leaves {
                let [NumericalArithmeticStage::UpstreamProjection { policy }] =
                    leaf.arithmetic.stages.as_mut_slice()
                else {
                    panic!("upstream policy");
                };
                for route in &mut policy.routes {
                    route
                        .local_rows
                        .retain(|row| matches!(row, 1 | 4 | 8 | 16 | 32));
                }
            }
        }
        let old = PreparedProjectionNumerics::prepare(&previous, &values).unwrap();
        assert_ne!(prepared.fingerprint(), old.fingerprint());
        for rows in 1..=32 {
            let mut f = facts(rows, UpstreamProjectionLayout::Columns, 17);
            f.weight_available_bytes =
                f.retained_zero_padded_weight_rows * u64::from(format.abi().2);
            let n = native(rows, f.layout, 17, !matches!(rows, 1 | 4 | 8));
            assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, None).is_err());
            let plan = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
            let PreparedUpstreamProjectionRoute::Selected {
                scratch,
                scratch_bytes,
                retained_weight_validation,
                ..
            } = plan.route()
            else {
                panic!("native partial-row route");
            };
            let mut end = 0;
            for region in scratch {
                assert_eq!(region.byte_offset % 16, 0);
                assert!(region.byte_offset >= end);
                end = region.byte_offset + region.byte_len;
            }
            assert_eq!(end, *scratch_bytes);
            assert_eq!(
                scratch.last().unwrap().role,
                UpstreamScratchRole::RowPoisonFlags
            );
            assert_eq!(scratch.last().unwrap().byte_len, u64::from(rows) * 4);
            assert!(retained_weight_validation.is_some());
            plan.validate_reconstructed(&contract, &values, &f, Some(&n))
                .unwrap();
        }
        for rows in [33, 1025, u32::from(u16::MAX)] {
            let f = facts(rows, UpstreamProjectionLayout::Columns, 17);
            assert!(matches!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &f, None)
                    .unwrap()
                    .route(),
                PreparedUpstreamProjectionRoute::StrictBase {
                    reason: UpstreamStrictReason::RowsOrLayoutNotDeclared
                }
            ));
        }
        assert!(PreparedUpstreamProjectionWave::prepare(
            &prepared,
            &facts(0, UpstreamProjectionLayout::Columns, 17),
            None
        )
        .is_err());
    }
}

#[test]
fn upstream_semantics_distinguish_actual_pack_min_and_coefficient_boundaries() {
    let d4 = UpstreamProjectionArithmetic::MmqD4V1
        .semantics(ProjectionBlockFormat::Iq4Xs)
        .unwrap();
    let ds4 = UpstreamProjectionArithmetic::MmqDs4V1
        .semantics(ProjectionBlockFormat::Q4K)
        .unwrap();
    let mv = UpstreamProjectionArithmetic::MmvqQ8_1V1
        .semantics(ProjectionBlockFormat::Q4K)
        .unwrap();
    assert_eq!(d4.activation_scale_type, ElementType::F32);
    assert_eq!(ds4.activation_scale_type, ElementType::F16);
    assert_eq!(
        ds4.min_source,
        UpstreamMinSource::HalfRoundedOriginalActivationSum
    );
    assert_eq!(
        mv.min_source,
        UpstreamMinSource::ActualQuantizedI8SumTimesStoredScale
    );
    assert_ne!(ds4.coefficient_rule, mv.coefficient_rule);
    assert_ne!(ds4.pack, d4.pack);
    assert_ne!(d4.reduction, mv.reduction);
    assert_ne!(d4.dynamic_domain, mv.dynamic_domain);
    assert!(UpstreamProjectionArithmetic::MmqDs4V1
        .semantics(ProjectionBlockFormat::Iq4Xs)
        .is_err());
    assert!(UpstreamProjectionArithmetic::MmqD4V1
        .semantics(ProjectionBlockFormat::Q5K)
        .is_err());
}

#[test]
fn upstream_policy_rejects_ambiguous_unknown_and_uninstantiated_domains() {
    let p = policy(
        ProjectionBlockFormat::Iq4Xs,
        UpstreamProjectionArithmetic::MmvqQ8_1V1,
        UpstreamProjectionLayout::Columns,
        &[1, 4, 8],
    );
    p.validate().unwrap();
    for rows in [0, 2, 3, 5, 6, 7, 9, 33] {
        let mut bad = p.clone();
        bad.routes[0].local_rows.insert(rows);
        assert!(bad.validate().is_err());
    }
    let mut duplicate = p.clone();
    duplicate.routes.push(p.routes[0].clone());
    assert!(duplicate.validate().is_err());
    let mut wire = serde_json::to_value(&p).unwrap();
    wire["routes"][0]["local_rows"] = serde_json::json!([4, 4]);
    assert!(serde_json::from_value::<UpstreamProjectionPolicy>(wire).is_err());
    let mut wire = serde_json::to_value(&p).unwrap();
    wire["routes"][0]["offered_concurrency"] = 8.into();
    assert!(serde_json::from_value::<UpstreamProjectionPolicy>(wire).is_err());
    let mut wire = serde_json::to_value(&p).unwrap();
    wire["routes"][0]["arithmetic"] = "mmq_mystery_v1".into();
    assert!(serde_json::from_value::<UpstreamProjectionPolicy>(wire).is_err());
}

#[test]
fn upstream_schema_preserves_old_wire_and_old_acceptance_boundaries() {
    let legacy = dense_swiglu_iq4xs_q8act_g32_arithmetic();
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../tests/vnext_numerical_profile/iq4xs_q8act_legacy.json"
    ))
    .unwrap();
    assert_eq!(serde_json::to_value(&legacy).unwrap(), expected);
    let bytes = serde_json::to_vec(&legacy).unwrap();
    let old: CompositeNumericalArithmetic = serde_json::from_slice(&bytes).unwrap();
    old.validate().unwrap();
    assert_eq!(serde_json::to_vec(&old).unwrap(), bytes);
    let (contract, _, _) = fixture(
        UpstreamProjectionArithmetic::MmqD4V1,
        UpstreamProjectionLayout::Columns,
        &[8],
        17,
    );
    for version in [1, 2, 4] {
        let mut bad = contract.clone();
        bad.schema_version = version;
        assert!(bad.validate().is_err());
    }
    let mut bad = legacy;
    bad.schema_version = COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM;
    assert!(bad.validate().is_err());
    let mut bad = contract.clone();
    bad.projections[0].weight_input_ordinal = 3;
    assert!(bad.validate().is_err());
    let mut bad = contract.clone();
    bad.projections[0].activation_output_type = ElementType::F32;
    assert!(bad.validate().is_err());
    let mut bad = contract;
    bad.projections[0].leaves[0].arithmetic.schema_version = 1;
    assert!(bad.validate().is_err());
}

#[test]
fn upstream_mmq_scratch_and_trusted_reconstruction_reject_tampering() {
    let (contract, values, prepared) = fixture(
        UpstreamProjectionArithmetic::MmqD4V1,
        UpstreamProjectionLayout::Columns,
        &[1, 8],
        17,
    );
    for rows in [1, 8] {
        let f = facts(rows, UpstreamProjectionLayout::Columns, 17);
        let n = native(rows, f.layout, 17, true);
        let plan = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
        let PreparedUpstreamProjectionRoute::Selected {
            scratch,
            scratch_bytes,
            ..
        } = plan.route()
        else {
            panic!("declared MMQ route")
        };
        let mut end = 0;
        for s in scratch {
            assert_eq!(s.byte_offset % 16, 0);
            assert!(s.byte_offset >= end);
            end = s.byte_offset + s.byte_len;
        }
        assert_eq!(end, *scratch_bytes);
        assert_eq!(scratch[0].byte_len, u64::from(rows) * 512 * 4);
        assert_eq!(
            scratch[1].byte_len,
            u64::from(rows) * 512 * 144 / 128 + u64::from(rows / 8 * 8) * 144
        );
        assert_eq!(scratch[2].byte_len, u64::from(rows) * 17 * 4);
        assert_eq!(scratch[3].byte_len, 4 * 8 * 128 * 4);
        let wire = serde_json::to_value(&plan).unwrap();
        let restored: PreparedUpstreamProjectionWave =
            serde_json::from_value(wire.clone()).unwrap();
        restored
            .validate_reconstructed(&contract, &values, &f, Some(&n))
            .unwrap();
        assert_eq!(plan.fingerprint().unwrap(), restored.fingerprint().unwrap());
        for field in ["byte_offset", "byte_len"] {
            let mut corrupt = wire.clone();
            corrupt["route"]["scratch"][1][field] = 0.into();
            assert!(
                serde_json::from_value::<PreparedUpstreamProjectionWave>(corrupt)
                    .unwrap()
                    .validate_reconstructed(&contract, &values, &f, Some(&n))
                    .is_err()
            );
        }
        let mut changed = f.clone();
        changed.input_byte_offset += 2;
        assert!(plan
            .validate_reconstructed(&contract, &values, &changed, Some(&n))
            .is_err());
        let mut changed = values.clone();
        changed[0] = binding(2, &[Some("quantization.gguf.iq4-xs")], 256, 17, true);
        assert!(plan
            .validate_reconstructed(&contract, &changed, &f, Some(&n))
            .is_err());
        let mut changed = n.clone();
        changed.implementation_fingerprint.push('2');
        assert!(plan
            .validate_reconstructed(&contract, &values, &f, Some(&changed))
            .is_err());
    }
}

#[test]
fn upstream_mmvq_requires_owned_tail_padding_and_exact_layout() {
    for layout in [
        UpstreamProjectionLayout::Columns,
        UpstreamProjectionLayout::Channels,
    ] {
        let (_, _, prepared) = fixture(UpstreamProjectionArithmetic::MmvqQ8_1V1, layout, &[4], 17);
        let mut f = facts(4, layout, 17);
        let n = native(4, layout, 17, false);
        let plan = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
        let PreparedUpstreamProjectionRoute::Selected { scratch, pack, .. } = plan.route() else {
            panic!("declared MMVQ route")
        };
        assert_eq!(*pack, UpstreamActivationPack::MmvqRowMajorQ8_1_36BytesPer32);
        assert_eq!(scratch.len(), 3);
        assert_eq!(scratch[1].byte_len, 4 * 512 / 32 * 36);
        f.retained_zero_padded_weight_rows = 17;
        assert!(matches!(
            PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n))
                .unwrap()
                .route(),
            PreparedUpstreamProjectionRoute::StrictBase {
                reason: UpstreamStrictReason::RetainedWeightPaddingNotProven
            }
        ));
        f.retained_zero_padded_weight_rows = 18;
        f.weight_available_bytes = 17 * 136;
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).is_err());
        f.weight_available_bytes = 18 * 136;
        let mut wrong = n.clone();
        if let UpstreamNativeGeometry::Mmvq { channels, .. } = &mut wrong.geometry {
            *channels += 1;
        }
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&wrong)).is_err());
    }
}

#[test]
fn upstream_wave_domains_distinguish_fallback_from_invalid_or_missing_implementation() {
    let (_, _, prepared) = fixture(
        UpstreamProjectionArithmetic::MmqD4V1,
        UpstreamProjectionLayout::Columns,
        &[8],
        17,
    );
    let f = facts(8, UpstreamProjectionLayout::Columns, 17);
    let n = native(8, f.layout, 17, true);
    assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, None).is_err());
    let plan = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
    let supported = BTreeSet::from([(
        ProjectionBlockFormat::Iq4Xs,
        UpstreamProjectionArithmetic::MmqD4V1,
    )]);
    assert!(plan
        .require_execution_support(
            &BTreeSet::new(),
            UpstreamDynamicDomainSupport::EnforcedBeforeDot
        )
        .is_err());
    assert!(plan
        .require_execution_support(&supported, UpstreamDynamicDomainSupport::Unavailable)
        .is_err());
    // This is a typed support declaration, not evidence that a device checker exists.
    plan.require_execution_support(&supported, UpstreamDynamicDomainSupport::EnforcedBeforeDot)
        .unwrap();
    let mut odd = f.clone();
    odd.weight_byte_offset = 13;
    assert!(matches!(
        PreparedUpstreamProjectionWave::prepare(&prepared, &odd, None)
            .unwrap()
            .route(),
        PreparedUpstreamProjectionRoute::StrictBase {
            reason: UpstreamStrictReason::WeightAlignmentNotSupported
        }
    ));
    let undeclared = facts(7, f.layout, 17);
    assert!(matches!(
        PreparedUpstreamProjectionWave::prepare(&prepared, &undeclared, None)
            .unwrap()
            .route(),
        PreparedUpstreamProjectionRoute::StrictBase {
            reason: UpstreamStrictReason::RowsOrLayoutNotDeclared
        }
    ));
    for field in [
        "input_byte_offset",
        "output_byte_offset",
        "input_stride",
        "output_stride",
    ] {
        let mut wire = serde_json::to_value(&f).unwrap();
        wire[field] = u64::MAX.into();
        let bad: UpstreamProjectionWaveFacts = serde_json::from_value(wire).unwrap();
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &bad, Some(&n)).is_err());
    }
    let mut short = f.clone();
    short.output_available_bytes -= 10;
    assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &short, Some(&n)).is_err());
    let mut wrong = n.clone();
    if let UpstreamNativeGeometry::Mmq { padded_inputs, .. } = &mut wrong.geometry {
        *padded_inputs = 256;
    }
    assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&wrong)).is_err());
}

#[test]
fn upstream_static_strict_leaves_remain_strict_without_native_support() {
    let (_, _, prepared) = fixture(
        UpstreamProjectionArithmetic::MmqD4V1,
        UpstreamProjectionLayout::Columns,
        &[8],
        17,
    );
    for (role, ordinal, reason) in [
        (
            ProjectionRole::CausalKey,
            3,
            StrictProjectionReason::FormatNotDeclared,
        ),
        (
            ProjectionRole::CausalValue,
            4,
            StrictProjectionReason::FormatNotDeclared,
        ),
        (
            ProjectionRole::CausalOutput,
            5,
            StrictProjectionReason::TransformedWeight,
        ),
    ] {
        let mut f = facts(8, UpstreamProjectionLayout::Columns, 17);
        f.role = role;
        f.component_id = WeightId::new(format!("component.{ordinal}.0")).unwrap();
        assert!(
            matches!(PreparedUpstreamProjectionWave::prepare(&prepared,&f,None).unwrap().route(),PreparedUpstreamProjectionRoute::StrictBase { reason:UpstreamStrictReason::Static(actual) } if actual==&reason)
        );
    }
}

#[test]
fn upstream_composite_covers_standard_roles_with_separate_affine_arithmetic() {
    for mut contract in [
        dense_swiglu_q4k_q5k_iq4xs_q8act_g32_arithmetic(),
        Q8ActAttentionProfile::GatedDelta.arithmetic(),
        Q8ActAttentionProfile::Causal.arithmetic(),
    ] {
        let strict = contract.strict_base.clone();
        let ports: Vec<_> = contract
            .projections
            .iter()
            .map(|p| (p.role, p.weight_input_ordinal))
            .collect();
        contract.schema_version = COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM;
        for p in &mut contract.projections {
            for leaf in &mut p.leaves {
                let mmq = if leaf.format == ProjectionBlockFormat::Iq4Xs {
                    UpstreamProjectionArithmetic::MmqD4V1
                } else {
                    UpstreamProjectionArithmetic::MmqDs4V1
                };
                let mut declared = policy(
                    leaf.format,
                    mmq,
                    UpstreamProjectionLayout::Columns,
                    &[16, 32],
                );
                declared.routes.push(UpstreamProjectionRouteDeclaration {
                    arithmetic: UpstreamProjectionArithmetic::MmvqQ8_1V1,
                    layout: UpstreamProjectionLayout::Columns,
                    local_rows: BTreeSet::from([1, 4, 8]),
                    prefill_rows: None,
                });
                leaf.arithmetic = declared.staged().unwrap();
            }
        }
        contract.validate().unwrap();
        assert_eq!(contract.strict_base, strict);
        assert_eq!(
            contract
                .projections
                .iter()
                .map(|p| (p.role, p.weight_input_ordinal))
                .collect::<Vec<_>>(),
            ports
        );
        let mut wrong = contract.clone();
        if let NumericalArithmeticStage::UpstreamProjection { policy } =
            &mut wrong.projections[0].leaves[0].arithmetic.stages[0]
        {
            policy.format = if policy.format == ProjectionBlockFormat::Iq4Xs {
                ProjectionBlockFormat::Q4K
            } else {
                ProjectionBlockFormat::Iq4Xs
            };
        }
        assert!(wrong.validate().is_err());
    }
}

#[test]
fn upstream_marker_v2_changes_exceptional_semantics_without_changing_finite_expressions() {
    for (v1, v2, format) in [
        (
            UpstreamProjectionArithmetic::MmqD4V1,
            UpstreamProjectionArithmetic::MmqD4MarkerV2,
            ProjectionBlockFormat::Iq4Xs,
        ),
        (
            UpstreamProjectionArithmetic::MmqDs4V1,
            UpstreamProjectionArithmetic::MmqDs4MarkerV2,
            ProjectionBlockFormat::Q4K,
        ),
        (
            UpstreamProjectionArithmetic::MmqDs4V1,
            UpstreamProjectionArithmetic::MmqDs4MarkerV2,
            ProjectionBlockFormat::Q5K,
        ),
        (
            UpstreamProjectionArithmetic::MmvqQ8_1V1,
            UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
            ProjectionBlockFormat::Iq4Xs,
        ),
        (
            UpstreamProjectionArithmetic::MmvqQ8_1V1,
            UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
            ProjectionBlockFormat::Q4K,
        ),
        (
            UpstreamProjectionArithmetic::MmvqQ8_1V1,
            UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
            ProjectionBlockFormat::Q5K,
        ),
    ] {
        let old = v1.semantics(format).unwrap();
        let mut new = v2.semantics(format).unwrap();
        let marker = new.marker.take().unwrap();
        assert_eq!(marker.canonical_f16_nan_bits, 0x7e00);
        assert_eq!(marker.canonical_f32_nan_bits, 0x7fc0_0000);
        assert!(marker.zero_logical_and_padding_groups_use_positive_zero_metadata_and_codes);
        assert!(marker.nonfinite_input_or_consumed_metadata_poison_entire_input_row);
        assert!(marker.nonfinite_consumed_weight_coefficient_poisons_entire_physical_leaf);
        assert!(marker.nonfinite_f32_result_or_f16_overflow_poisons_output_element);
        assert!(marker.unused_original_sum_overflow_is_ignored);
        assert!(marker.half_scale_underflow_to_zero_is_allowed);
        assert_eq!(
            new.dynamic_domain,
            UpstreamDynamicDomain::CanonicalNanMarkerV2
        );
        new.dynamic_domain = old.dynamic_domain;
        assert_eq!(new, old);
        let old_policy = policy(format, v1, UpstreamProjectionLayout::Columns, &[8]);
        let new_policy = policy(format, v2, UpstreamProjectionLayout::Columns, &[8]);
        assert_ne!(
            serde_json::to_vec(&old_policy).unwrap(),
            serde_json::to_vec(&new_policy).unwrap()
        );
        let old_wire = serde_json::to_string(&old_policy).unwrap();
        assert!(!old_wire.contains("marker"));
        let restored: UpstreamProjectionPolicy = serde_json::from_str(&old_wire).unwrap();
        assert_eq!(serde_json::to_string(&restored).unwrap(), old_wire);
    }
}

#[test]
fn upstream_marker_v2_retains_distinct_row_and_weight_flags_and_requires_actual_protocol() {
    for (v1, v2, mmq) in [
        (
            UpstreamProjectionArithmetic::MmqD4V1,
            UpstreamProjectionArithmetic::MmqD4MarkerV2,
            true,
        ),
        (
            UpstreamProjectionArithmetic::MmvqQ8_1V1,
            UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
            false,
        ),
    ] {
        for rows in [1, 4, 8] {
            let (contract, values, prepared) =
                fixture(v2, UpstreamProjectionLayout::Columns, &[rows], 17);
            let (_, _, legacy) = fixture(v1, UpstreamProjectionLayout::Columns, &[rows], 17);
            let f = facts(rows, UpstreamProjectionLayout::Columns, 17);
            let n = native(rows, f.layout, 17, mmq);
            let plan = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
            let old = PreparedUpstreamProjectionWave::prepare(&legacy, &f, Some(&n)).unwrap();
            assert_ne!(old.fingerprint().unwrap(), plan.fingerprint().unwrap());
            assert!(serde_json::to_value(&old).unwrap()["route"]
                .get("retained_weight_validation")
                .is_none());
            let PreparedUpstreamProjectionRoute::Selected {
                scratch,
                scratch_bytes,
                retained_weight_validation,
                ..
            } = plan.route()
            else {
                panic!("V2 selected")
            };
            let flag = scratch.last().unwrap();
            assert_eq!(flag.role, UpstreamScratchRole::RowPoisonFlags);
            assert_eq!(flag.byte_len, u64::from(rows) * 4);
            assert_eq!(flag.byte_offset % 16, 0);
            assert_eq!(flag.byte_offset + flag.byte_len, *scratch_bytes);
            assert!(scratch
                .windows(2)
                .all(|v| v[0].byte_offset + v[0].byte_len <= v[1].byte_offset));
            let retained = retained_weight_validation.as_ref().unwrap();
            assert_eq!((retained.byte_len, retained.alignment), (4, 4));
            assert_eq!(
                (&retained.component_id, retained.weight_byte_offset),
                (&f.component_id, f.weight_byte_offset)
            );
            let available = BTreeSet::from([(ProjectionBlockFormat::Iq4Xs, v2)]);
            for absent in [
                UpstreamDynamicDomainSupport::Unavailable,
                UpstreamDynamicDomainSupport::EnforcedBeforeDot,
            ] {
                assert!(plan.require_execution_support(&available, absent).is_err());
            }
            assert!(plan
                .require_execution_support(
                    &BTreeSet::from([(ProjectionBlockFormat::Iq4Xs, v1)]),
                    UpstreamDynamicDomainSupport::CanonicalNanMarkerV2Device
                )
                .is_err());
            // This checks the API declaration only. Hardware tests must prove
            // actual pack, retained scan and cast wiring before registration.
            plan.require_execution_support(
                &available,
                UpstreamDynamicDomainSupport::CanonicalNanMarkerV2Device,
            )
            .unwrap();
            let mut wire = serde_json::to_value(&plan).unwrap();
            wire["route"]["retained_weight_validation"]["component_id"] = "component.3.0".into();
            assert!(
                serde_json::from_value::<PreparedUpstreamProjectionWave>(wire)
                    .unwrap()
                    .validate_reconstructed(&contract, &values, &f, Some(&n))
                    .is_err()
            );
        }
    }
}

#[test]
fn hybrid_g32_mmq_wave_preserves_byte_safe_decode_and_trusted_branch_identity() {
    for format in [
        "quantization.gguf.iq4-xs",
        "quantization.gguf.q4-k",
        "quantization.gguf.q5-k",
    ] {
        let contract = UpstreamMarkerV2Profile::CausalG32MmqPrefill.arithmetic();
        let values = vec![
            binding(2, &[Some(format)], 256, 17, false),
            binding(3, &[None], 256, 17, false),
            binding(4, &[None], 256, 17, false),
            binding(5, &[None], 256, 17, false),
        ];
        let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
        let block_bytes = match format {
            "quantization.gguf.q4-k" => 144,
            "quantization.gguf.q5-k" => 176,
            _ => 136,
        };
        for rows in [1, 3, 4, 7, 8, 9, 16, 32, 33, 2048, 2049] {
            let mut f = facts(rows, UpstreamProjectionLayout::Columns, 17);
            f.input_stride = 256;
            f.weight_available_bytes = 17 * block_bytes;
            f.weight_byte_offset = 13;
            let wave = PreparedUpstreamProjectionWave::prepare(&prepared, &f, None).unwrap();
            if rows <= 32 {
                assert!(matches!(
                    wave.route(),
                    PreparedUpstreamProjectionRoute::G32 { .. }
                ));
                wave.validate_reconstructed(&contract, &values, &f, None)
                    .unwrap();
                let mut changed = f.clone();
                changed.local_rows = 33;
                changed.input_available_bytes = 33 * 259 * 2;
                changed.output_available_bytes = 33 * 20 * 2;
                assert!(wave
                    .validate_reconstructed(&contract, &values, &changed, None)
                    .is_err());
                let mut strided = f.clone();
                strided.input_stride = 257;
                assert!(matches!(
                    PreparedUpstreamProjectionWave::prepare(&prepared, &strided, None)
                        .unwrap()
                        .route(),
                    PreparedUpstreamProjectionRoute::StrictBase {
                        reason: UpstreamStrictReason::ActivationStrideNotSupported
                    }
                ));
                assert!(wave
                    .require_execution_support(
                        &BTreeSet::new(),
                        UpstreamDynamicDomainSupport::CanonicalNanMarkerV2Device
                    )
                    .is_err());
                let mut short = f.clone();
                short.weight_available_bytes -= 1;
                assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &short, None).is_err());
                let mut bad_input = f.clone();
                bad_input.input_byte_offset += 1;
                assert!(
                    PreparedUpstreamProjectionWave::prepare(&prepared, &bad_input, None).is_err()
                );
            } else {
                assert!(matches!(
                    wave.route(),
                    PreparedUpstreamProjectionRoute::StrictBase { .. }
                ));
                if rows <= 2048 {
                    f.weight_byte_offset = 12;
                    assert!(
                        PreparedUpstreamProjectionWave::prepare(&prepared, &f, None).is_err(),
                        "eligible MMQ cannot silently use strict without its native plan"
                    );
                }
            }
        }
        let mut f = facts(33, UpstreamProjectionLayout::Columns, 17);
        f.weight_available_bytes = 17 * block_bytes;
        let mut n = native(33, f.layout, 17, true);
        n.geometry = UpstreamNativeGeometry::Mmq {
            padded_inputs: 512,
            row_tile: 32,
            column_tile: 128,
            threads: 256,
            shared_bytes: 32768,
            packed_guard_blocks: 32,
            blocks: n.multiprocessors,
            fixup: true,
        };
        let wave = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
        assert!(matches!(
            wave.route(),
            PreparedUpstreamProjectionRoute::Selected {
                retained_weight_validation: Some(_),
                ..
            }
        ));
        wave.validate_reconstructed(&contract, &values, &f, Some(&n))
            .unwrap();
        let mut changed = serde_json::to_value(&wave).unwrap();
        changed["route"]["scratch_bytes"] = 0.into();
        let changed: PreparedUpstreamProjectionWave = serde_json::from_value(changed).unwrap();
        assert!(changed
            .validate_reconstructed(&contract, &values, &f, Some(&n))
            .is_err());
    }
}

#[path = "extra_prefill_tests.rs"]
mod extra_prefill;

#[path = "all_rows_tests.rs"]
mod all_rows;
