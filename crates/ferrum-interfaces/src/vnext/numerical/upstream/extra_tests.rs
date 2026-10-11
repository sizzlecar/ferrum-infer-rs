use super::*;

const EXTRA: [ProjectionBlockFormat; 3] = [
    ProjectionBlockFormat::Q3K,
    ProjectionBlockFormat::Iq3S,
    ProjectionBlockFormat::Iq4Nl,
];

#[test]
fn upstream_extra_preserves_old_leaf_wire_and_rejects_legacy_schema_reinterpretation() {
    for (old, new) in [
        (
            UpstreamMarkerV2Profile::SwiGluPrefill,
            UpstreamMarkerV2Profile::SwiGluExtraPrefill,
        ),
        (
            UpstreamMarkerV2Profile::GatedDeltaPrefill,
            UpstreamMarkerV2Profile::GatedDeltaExtraPrefill,
        ),
        (
            UpstreamMarkerV2Profile::CausalPrefill,
            UpstreamMarkerV2Profile::CausalExtraPrefill,
        ),
    ] {
        assert!(new.extra() && new.prefill() && !new.hybrid());
        let prior = old.arithmetic();
        let current = new.arithmetic();
        current.validate().unwrap();
        assert_ne!(old.operation_id(), new.operation_id());
        let mut restored = current.clone();
        restored.schema_version = prior.schema_version;
        for projection in &mut restored.projections {
            projection.leaves.retain(|leaf| !leaf.format.is_extra());
        }
        assert_eq!(
            serde_json::to_vec(&restored).unwrap(),
            serde_json::to_vec(&prior).unwrap()
        );
        let wire = serde_json::to_vec(&current).unwrap();
        let roundtrip: CompositeNumericalArithmetic = serde_json::from_slice(&wire).unwrap();
        roundtrip.validate().unwrap();
        assert_eq!(current, roundtrip);
        for schema in 1..=COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_G32_MMQ {
            let mut invalid = current.clone();
            invalid.schema_version = schema;
            assert!(invalid.validate().is_err());
        }
        for projection in &current.projections {
            for leaf in projection
                .leaves
                .iter()
                .filter(|leaf| leaf.format.is_extra())
            {
                for schema in [
                    NUMERICAL_ARITHMETIC_SCHEMA_VERSION,
                    NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM,
                    NUMERICAL_ARITHMETIC_SCHEMA_VERSION_G32_MMQ,
                ] {
                    let mut invalid = leaf.arithmetic.clone();
                    invalid.schema_version = schema;
                    assert!(invalid.validate().is_err());
                }
            }
        }
    }
    for format in EXTRA {
        assert!(G32MmqPrefillPolicy::new(format).is_err());
    }
    let mut hybrid = G32MmqPrefillPolicy::new(ProjectionBlockFormat::Iq4Xs).unwrap();
    hybrid.format = ProjectionBlockFormat::Q3K;
    // Malformed serialized hybrid input must reject, not panic in its closed builder.
    assert!(hybrid.validate().is_err());
}

#[test]
fn upstream_extra_semantics_distinguish_d4_no_min_and_weight_qk_from_activation_abi() {
    for format in EXTRA {
        for arithmetic in [
            UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2,
            UpstreamProjectionArithmetic::MmvqQ8_1ExtraMarkerV2,
        ] {
            let semantics = arithmetic.semantics(format).unwrap();
            assert_eq!(semantics.min_source, UpstreamMinSource::None);
            assert_eq!(
                semantics.dynamic_domain,
                UpstreamDynamicDomain::CanonicalNanMarkerV2
            );
            assert_eq!(semantics.marker.unwrap().canonical_f16_nan_bits, 0x7e00);
            assert!(arithmetic.semantics(ProjectionBlockFormat::Iq4Xs).is_err());
        }
        for old in [
            UpstreamProjectionArithmetic::MmqD4V1,
            UpstreamProjectionArithmetic::MmqDs4MarkerV2,
            UpstreamProjectionArithmetic::MmvqQ8_1MarkerV2,
        ] {
            assert!(old.semantics(format).is_err());
        }
        let (name, qk, bytes) = format.abi();
        let block = BlockQuantizationSpec {
            format_id: QuantizationFormatId::new(name).unwrap(),
            logical_values_per_block: qk,
            bytes_per_block: bytes,
        };
        let contract = UpstreamMarkerV2Profile::CausalExtraPrefill.arithmetic();
        assert!(matches!(
            contract
                .declared_projection_arithmetic(
                    ProjectionRole::CausalQuery,
                    Some(&block),
                    256,
                    17,
                    false
                )
                .unwrap(),
            DeclaredProjectionArithmetic::Staged(_)
        ));
        assert!(matches!(
            contract
                .declared_projection_arithmetic(
                    ProjectionRole::CausalQuery,
                    Some(&block),
                    256,
                    17,
                    true
                )
                .unwrap(),
            DeclaredProjectionArithmetic::StrictBase(StrictProjectionReason::TransformedWeight)
        ));
        for k in [32, 64, 255, 257] {
            assert!(matches!(
                contract
                    .declared_projection_arithmetic(
                        ProjectionRole::CausalQuery,
                        Some(&block),
                        k,
                        17,
                        false
                    )
                    .unwrap(),
                DeclaredProjectionArithmetic::StrictBase(StrictProjectionReason::ShapeNotDeclared)
            ));
        }
        let mut wrong_abi = block.clone();
        wrong_abi.bytes_per_block += 1;
        assert!(matches!(
            contract
                .declared_projection_arithmetic(
                    ProjectionRole::CausalQuery,
                    Some(&wrong_abi),
                    256,
                    17,
                    false
                )
                .unwrap(),
            DeclaredProjectionArithmetic::StrictBase(StrictProjectionReason::FormatNotDeclared)
        ));
    }
    assert_eq!(
        ProjectionBlockFormat::Iq4Nl.abi(),
        ("quantization.gguf.iq4-nl", 32, 18)
    );
}

fn values(format: ProjectionBlockFormat) -> Vec<ResolvedValueBinding> {
    vec![
        binding(2, &[Some(format.abi().0)], 256, 17, false),
        binding(3, &[None], 256, 17, false),
        binding(4, &[None], 256, 17, false),
        binding(5, &[None], 256, 17, false),
    ]
}

#[test]
fn upstream_extra_wave_rebuild_checks_real_block_spans_scratch_and_missing_implementation() {
    let contract = UpstreamMarkerV2Profile::CausalExtraPrefill.arithmetic();
    for format in EXTRA {
        let values = values(format);
        let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
        let mut f = facts(16, UpstreamProjectionLayout::Columns, 17);
        let expected = 17 * (256 / u64::from(format.abi().1)) * u64::from(format.abi().2);
        f.weight_available_bytes = expected;
        f.retained_zero_padded_weight_rows = 17;
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, None).is_err());
        let n = native(16, f.layout, 17, true);
        let wave = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
        wave.validate_reconstructed(&contract, &values, &f, Some(&n))
            .unwrap();
        let PreparedUpstreamProjectionRoute::Selected {
            arithmetic,
            scratch,
            scratch_bytes,
            retained_weight_validation,
            ..
        } = wave.route()
        else {
            panic!("declared MMQ route")
        };
        assert_eq!(
            *arithmetic,
            UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
        );
        assert_eq!(retained_weight_validation.as_ref().unwrap().format, format);
        assert_eq!(retained_weight_validation.as_ref().unwrap().byte_len, 4);
        for pair in scratch.windows(2) {
            assert!(pair[0].byte_offset + pair[0].byte_len <= pair[1].byte_offset);
        }
        assert!(scratch
            .iter()
            .all(|r| r.byte_offset + r.byte_len <= *scratch_bytes));
        assert!(wave
            .require_execution_support(
                &BTreeSet::new(),
                UpstreamDynamicDomainSupport::CanonicalNanMarkerV2Device
            )
            .is_err());
        let available = BTreeSet::from([(format, *arithmetic)]);
        assert!(wave
            .require_execution_support(&available, UpstreamDynamicDomainSupport::EnforcedBeforeDot)
            .is_err());
        wave.require_execution_support(
            &available,
            UpstreamDynamicDomainSupport::CanonicalNanMarkerV2Device,
        )
        .unwrap();
        let mut short = f.clone();
        short.weight_available_bytes -= 1;
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &short, Some(&n)).is_err());
        let mut overflow = f.clone();
        overflow.weight_byte_offset = u64::MAX - 3;
        assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &overflow, Some(&n)).is_err());
        let mut odd = f.clone();
        odd.weight_byte_offset += 1;
        assert!(matches!(
            PreparedUpstreamProjectionWave::prepare(&prepared, &odd, None)
                .unwrap()
                .route(),
            PreparedUpstreamProjectionRoute::StrictBase {
                reason: UpstreamStrictReason::WeightAlignmentNotSupported
            }
        ));
        let mut changed = f.clone();
        changed.local_rows = 7;
        assert!(wave
            .validate_reconstructed(&contract, &values, &changed, None)
            .is_err());
        for rows in [2, 3, 5, 6, 7, 9, 15, 17, 31, 33, 2048, 2049] {
            let mut unsupported = facts(rows, f.layout, 17);
            unsupported.weight_available_bytes = expected;
            assert!(matches!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &unsupported, None)
                    .unwrap()
                    .route(),
                PreparedUpstreamProjectionRoute::StrictBase {
                    reason: UpstreamStrictReason::RowsOrLayoutNotDeclared
                }
            ));
        }
    }
}

#[test]
fn upstream_extra_mmvq_padding_requires_owned_full_qk32_rows() {
    let format = ProjectionBlockFormat::Iq4Nl;
    let contract = UpstreamMarkerV2Profile::CausalExtraPrefill.arithmetic();
    let values = values(format);
    let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
    let mut f = facts(8, UpstreamProjectionLayout::Columns, 17);
    let row_bytes = (256 / 32) * 18;
    f.weight_available_bytes = 18 * row_bytes;
    f.retained_zero_padded_weight_rows = 17;
    let n = native(8, f.layout, 17, false);
    assert!(matches!(
        PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n))
            .unwrap()
            .route(),
        PreparedUpstreamProjectionRoute::StrictBase {
            reason: UpstreamStrictReason::RetainedWeightPaddingNotProven
        }
    ));
    f.retained_zero_padded_weight_rows = 18;
    let wave = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
    wave.validate_reconstructed(&contract, &values, &f, Some(&n))
        .unwrap();
    f.weight_available_bytes -= 1;
    assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).is_err());
}

#[test]
fn upstream_extra_selects_only_declared_local_rows_without_role_or_shape_heuristics() {
    for profile in [
        UpstreamMarkerV2Profile::SwiGluExtraPrefill,
        UpstreamMarkerV2Profile::GatedDeltaExtraPrefill,
        UpstreamMarkerV2Profile::CausalExtraPrefill,
    ] {
        for projection in profile.arithmetic().projections {
            for leaf in projection
                .leaves
                .into_iter()
                .filter(|leaf| leaf.format.is_extra())
            {
                let policy = leaf.arithmetic.upstream_policy().unwrap();
                let mmvq_rows: &[u32] = match leaf.format {
                    ProjectionBlockFormat::Q3K => &[1],
                    ProjectionBlockFormat::Iq3S => &[1, 4],
                    ProjectionBlockFormat::Iq4Nl => &[1, 4, 8],
                    _ => unreachable!(),
                };
                for row in 0..=33 {
                    let routes = policy
                        .routes
                        .iter()
                        .filter(|route| route.contains_rows(row))
                        .collect::<Vec<_>>();
                    if ![1, 4, 8, 16, 32].contains(&row) {
                        assert!(
                            routes.is_empty(),
                            "partial local rows keep strict semantics"
                        );
                    } else {
                        assert_eq!(
                            routes.len(),
                            1,
                            "each eligible local row has one numerical rule"
                        );
                        assert_eq!(routes[0].layout, UpstreamProjectionLayout::Columns);
                        assert_eq!(
                            routes[0].arithmetic,
                            if mmvq_rows.contains(&row) {
                                UpstreamProjectionArithmetic::MmvqQ8_1ExtraMarkerV2
                            } else {
                                UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
                            }
                        );
                    }
                }
                assert!(policy
                    .routes
                    .iter()
                    .all(|route| route.prefill_rows.is_none()));
                assert!(!policy.routes.iter().any(|route| route.contains_rows(2048)));
            }
        }
    }
}
