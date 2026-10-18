use super::*;

const EXTRA: [ProjectionBlockFormat; 3] = [
    ProjectionBlockFormat::Q3K,
    ProjectionBlockFormat::Iq3S,
    ProjectionBlockFormat::Iq4Nl,
];

#[test]
fn extra_all_rows_extends_only_missing_mmq_rows_without_changing_old_wire() {
    for (old, new) in [
        (
            UpstreamMarkerV2Profile::SwiGluExtraLargePrefill,
            UpstreamMarkerV2Profile::SwiGluExtraAllRows,
        ),
        (
            UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill,
            UpstreamMarkerV2Profile::GatedDeltaExtraAllRows,
        ),
        (
            UpstreamMarkerV2Profile::CausalExtraLargePrefill,
            UpstreamMarkerV2Profile::CausalExtraAllRows,
        ),
    ] {
        assert!(!old.extra_all_rows());
        assert!(
            new.extra_all_rows()
                && new.extra_prefill()
                && new.extra()
                && new.prefill()
                && !new.hybrid()
        );
        assert_ne!(old.operation_id(), new.operation_id());
        assert_ne!(old.capability_id(), new.capability_id());
        let prior = old.arithmetic();
        let current = new.arithmetic();
        current.validate().unwrap();
        assert_eq!(current.schema_version, prior.schema_version);
        let mut restored = current.clone();
        for (projection, before) in restored.projections.iter_mut().zip(&prior.projections) {
            for (leaf, old_leaf) in projection.leaves.iter_mut().zip(&before.leaves) {
                if leaf.format.is_extra() {
                    assert_eq!(
                        leaf.arithmetic.schema_version,
                        old_leaf.arithmetic.schema_version
                    );
                    let [NumericalArithmeticStage::UpstreamProjection { policy }] =
                        leaf.arithmetic.stages.as_mut_slice()
                    else {
                        panic!("upstream")
                    };
                    let old_policy = old_leaf.arithmetic.upstream_policy().unwrap();
                    assert_eq!(policy.routes.len(), old_policy.routes.len());
                    for rows in 1..=32 {
                        let route = policy
                            .routes
                            .iter()
                            .find(|r| r.contains_rows(rows))
                            .unwrap();
                        if let Some(old_route) =
                            old_policy.routes.iter().find(|r| r.contains_rows(rows))
                        {
                            assert_eq!(route.arithmetic, old_route.arithmetic);
                            assert_eq!(route.layout, old_route.layout);
                        } else {
                            assert_eq!(
                                route.arithmetic,
                                UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
                            );
                            assert_eq!(route.layout, UpstreamProjectionLayout::Columns);
                        }
                    }
                    for (route, old_route) in policy.routes.iter_mut().zip(&old_policy.routes) {
                        assert_eq!(route.arithmetic, old_route.arithmetic);
                        assert_eq!(route.layout, old_route.layout);
                        assert_eq!(route.prefill_rows, old_route.prefill_rows);
                        route
                            .local_rows
                            .retain(|rows| old_route.local_rows.contains(rows));
                    }
                }
                assert_eq!(
                    serde_json::to_vec(leaf).unwrap(),
                    serde_json::to_vec(old_leaf).unwrap()
                );
            }
        }
        assert_eq!(
            serde_json::to_vec(&restored).unwrap(),
            serde_json::to_vec(&prior).unwrap()
        );
        let decoded: CompositeNumericalArithmetic =
            serde_json::from_slice(&serde_json::to_vec(&current).unwrap()).unwrap();
        assert_eq!(decoded, current);
        decoded.validate().unwrap();
    }
}

#[test]
fn extra_all_rows_prepares_new_widths_with_exact_native_identity_and_bounds() {
    let contract = UpstreamMarkerV2Profile::CausalExtraAllRows.arithmetic();
    let old = UpstreamMarkerV2Profile::CausalExtraLargePrefill.arithmetic();
    for format in EXTRA {
        let values = vec![
            binding(2, &[Some(format.abi().0)], 256, 17, false),
            binding(3, &[None], 256, 17, false),
            binding(4, &[None], 256, 17, false),
            binding(5, &[None], 256, 17, false),
        ];
        let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
        let prior = PreparedProjectionNumerics::prepare(&old, &values).unwrap();
        for rows in (1..=32).filter(|m| !matches!(m, 1 | 4 | 8 | 16 | 32)) {
            let mut f = facts(rows, UpstreamProjectionLayout::Columns, 17);
            f.weight_available_bytes =
                17 * (256 / u64::from(format.abi().1)) * u64::from(format.abi().2);
            let n = native(rows, f.layout, 17, true);
            assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, None).is_err());
            assert!(matches!(
                PreparedUpstreamProjectionWave::prepare(&prior, &f, None)
                    .unwrap()
                    .route(),
                PreparedUpstreamProjectionRoute::StrictBase {
                    reason: UpstreamStrictReason::RowsOrLayoutNotDeclared
                }
            ));
            let wave = PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&n)).unwrap();
            wave.validate_reconstructed(&contract, &values, &f, Some(&n))
                .unwrap();
            assert!(wave
                .validate_reconstructed(&old, &values, &f, Some(&n))
                .is_err());
            let PreparedUpstreamProjectionRoute::Selected {
                arithmetic,
                scratch,
                ..
            } = wave.route()
            else {
                panic!("new MMQ width")
            };
            assert_eq!(
                *arithmetic,
                UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
            );
            for pair in scratch.windows(2) {
                assert!(pair[0].byte_offset + pair[0].byte_len <= pair[1].byte_offset);
            }
            let mut changed = f.clone();
            changed.weight_available_bytes -= 1;
            assert!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &changed, Some(&n)).is_err()
            );
            changed = f.clone();
            changed.weight_byte_offset += 1;
            assert!(matches!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &changed, None)
                    .unwrap()
                    .route(),
                PreparedUpstreamProjectionRoute::StrictBase {
                    reason: UpstreamStrictReason::WeightAlignmentNotSupported
                }
            ));
            changed = f.clone();
            changed.input_byte_offset = u64::MAX - 1;
            assert!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &changed, Some(&n)).is_err()
            );
            changed = f.clone();
            changed.local_rows = 0;
            assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &changed, None).is_err());
            let wrong = native(rows, f.layout, 17, false);
            assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &f, Some(&wrong)).is_err());
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
}

#[test]
fn extra_all_rows_does_not_relax_mmvq_or_accept_ambiguous_domains() {
    let contract = UpstreamMarkerV2Profile::SwiGluExtraAllRows.arithmetic();
    for leaf in contract.projections[0]
        .leaves
        .iter()
        .filter(|l| l.format.is_extra())
    {
        let policy = leaf.arithmetic.upstream_policy().unwrap();
        let mut duplicate = policy.clone();
        duplicate.routes[0].local_rows.insert(31);
        assert!(duplicate.staged().is_err());
        let mut unsupported_mmvq = policy.clone();
        for route in &mut unsupported_mmvq.routes {
            route.local_rows.remove(&31);
        }
        unsupported_mmvq.routes[0].local_rows.insert(31);
        assert!(unsupported_mmvq.staged().is_err());
        for invalid_rows in [0, 33, u32::MAX] {
            let mut invalid = policy.clone();
            invalid.routes[1].local_rows.insert(invalid_rows);
            assert!(invalid.staged().is_err());
        }
    }
}
