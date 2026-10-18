use super::*;

const FORMATS: [ProjectionBlockFormat; 3] = [
    ProjectionBlockFormat::Q3K,
    ProjectionBlockFormat::Iq3S,
    ProjectionBlockFormat::Iq4Nl,
];

#[test]
fn extra_prefill_preserves_old_small_routes_and_original_leaf_wire() {
    for (old, new) in [
        (
            UpstreamMarkerV2Profile::SwiGluExtraPrefill,
            UpstreamMarkerV2Profile::SwiGluExtraLargePrefill,
        ),
        (
            UpstreamMarkerV2Profile::GatedDeltaExtraPrefill,
            UpstreamMarkerV2Profile::GatedDeltaExtraLargePrefill,
        ),
        (
            UpstreamMarkerV2Profile::CausalExtraPrefill,
            UpstreamMarkerV2Profile::CausalExtraLargePrefill,
        ),
    ] {
        assert!(!old.extra_prefill());
        assert!(new.extra_prefill() && new.extra() && new.prefill() && !new.hybrid());
        assert_ne!(old.operation_id(), new.operation_id());
        let prior = old.arithmetic();
        let current = new.arithmetic();
        current.validate().unwrap();
        assert_eq!(
            current.schema_version,
            COMPOSITE_NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA_PREFILL
        );
        let mut restored = current.clone();
        restored.schema_version = prior.schema_version;
        for (projection, old_projection) in restored.projections.iter_mut().zip(&prior.projections)
        {
            for (leaf, old_leaf) in projection.leaves.iter_mut().zip(&old_projection.leaves) {
                if leaf.format.is_extra() {
                    let [NumericalArithmeticStage::UpstreamProjection { policy }] =
                        leaf.arithmetic.stages.as_mut_slice()
                    else {
                        panic!("upstream")
                    };
                    let old_policy = old_leaf.arithmetic.upstream_policy().unwrap();
                    for rows in 0..=32 {
                        assert_eq!(
                            policy.routes.iter().find(|r| r.contains_rows(rows)),
                            old_policy.routes.iter().find(|r| r.contains_rows(rows))
                        );
                    }
                    let route = policy.routes.pop().unwrap();
                    assert_eq!(
                        route.prefill_rows,
                        Some(UpstreamPrefillRows {
                            first: 33,
                            last: 2048
                        })
                    );
                    assert_eq!(
                        route.arithmetic,
                        UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
                    );
                    assert_eq!(route.layout, UpstreamProjectionLayout::Columns);
                    assert!(route.local_rows.is_empty());
                    leaf.arithmetic.schema_version =
                        NUMERICAL_ARITHMETIC_SCHEMA_VERSION_UPSTREAM_EXTRA;
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
        for schema in 1..=6 {
            let mut invalid = current.clone();
            invalid.schema_version = schema;
            assert!(invalid.validate().is_err(), "schema {schema}");
        }
    }
}

#[test]
fn extra_prefill_rejects_legacy_stages_overlap_and_non_mmq_intervals() {
    let contract = UpstreamMarkerV2Profile::CausalExtraLargePrefill.arithmetic();
    for leaf in contract.projections[0]
        .leaves
        .iter()
        .filter(|l| l.format.is_extra())
    {
        for schema in 1..=5 {
            let mut invalid = leaf.arithmetic.clone();
            invalid.schema_version = schema;
            assert!(invalid.validate().is_err());
        }
        let policy = leaf.arithmetic.upstream_policy().unwrap();
        for (first, last) in [(0, 2048), (32, 2048), (33, 2049), (54, 33)] {
            let mut invalid = policy.clone();
            invalid.routes.last_mut().unwrap().prefill_rows =
                Some(UpstreamPrefillRows { first, last });
            assert!(invalid.staged().is_err());
        }
        let mut invalid = policy.clone();
        invalid.routes.push(invalid.routes.last().unwrap().clone());
        assert!(invalid.staged().is_err());
        let mut invalid = policy.clone();
        invalid.routes.last_mut().unwrap().arithmetic =
            UpstreamProjectionArithmetic::MmvqQ8_1ExtraMarkerV2;
        assert!(invalid.staged().is_err());
        let mut invalid = policy.clone();
        invalid.routes.last_mut().unwrap().layout = UpstreamProjectionLayout::Channels;
        assert!(invalid.staged().is_err());
    }
}

#[test]
fn extra_prefill_wave_rebuild_enforces_native_geometry_real_weight_span_and_scratch() {
    let contract = UpstreamMarkerV2Profile::CausalExtraLargePrefill.arithmetic();
    for format in FORMATS {
        let values = vec![
            binding(2, &[Some(format.abi().0)], 256, 17, false),
            binding(3, &[None], 256, 17, false),
            binding(4, &[None], 256, 17, false),
            binding(5, &[None], 256, 17, false),
        ];
        let prepared = PreparedProjectionNumerics::prepare(&contract, &values).unwrap();
        for rows in [33, 54, 2048] {
            let mut facts = facts(rows, UpstreamProjectionLayout::Columns, 17);
            facts.weight_available_bytes =
                17 * (256 / u64::from(format.abi().1)) * u64::from(format.abi().2);
            let mut native = native(rows, facts.layout, 17, true);
            native.geometry = UpstreamNativeGeometry::Mmq {
                padded_inputs: 512,
                row_tile: 32,
                column_tile: 128,
                threads: 256,
                shared_bytes: 32768,
                packed_guard_blocks: rows.min(512) / 8 * 8,
                blocks: native.multiprocessors,
                fixup: true,
            };
            assert!(PreparedUpstreamProjectionWave::prepare(&prepared, &facts, None).is_err());
            let wave =
                PreparedUpstreamProjectionWave::prepare(&prepared, &facts, Some(&native)).unwrap();
            wave.validate_reconstructed(&contract, &values, &facts, Some(&native))
                .unwrap();
            let PreparedUpstreamProjectionRoute::Selected {
                arithmetic,
                scratch,
                scratch_bytes,
                ..
            } = wave.route()
            else {
                panic!("eligible extra MMQ prefill")
            };
            assert_eq!(
                *arithmetic,
                UpstreamProjectionArithmetic::MmqD4ExtraMarkerV2
            );
            for pair in scratch.windows(2) {
                assert!(pair[0].byte_offset + pair[0].byte_len <= pair[1].byte_offset);
            }
            assert!(
                *scratch_bytes
                    <= UpstreamScratchEstimate::mmq_prefill(256, 17, native.multiprocessors)
                        .unwrap()
                        .bytes(u64::from(rows))
                        .unwrap()
            );
            let mut short = facts.clone();
            short.weight_available_bytes -= 1;
            assert!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &short, Some(&native)).is_err()
            );
            let mut odd = facts.clone();
            odd.weight_byte_offset += 1;
            assert!(matches!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &odd, None)
                    .unwrap()
                    .route(),
                PreparedUpstreamProjectionRoute::StrictBase {
                    reason: UpstreamStrictReason::WeightAlignmentNotSupported
                }
            ));
            let mut changed = facts.clone();
            changed.local_rows = 32;
            assert!(wave
                .validate_reconstructed(&contract, &values, &changed, Some(&native))
                .is_err());
            if let UpstreamNativeGeometry::Mmq { row_tile, .. } = &mut native.geometry {
                *row_tile = 16;
            }
            assert!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &facts, Some(&native)).is_err()
            );
        }
        for rows in [3, 9, 2049] {
            let f = facts(rows, UpstreamProjectionLayout::Columns, 17);
            assert!(matches!(
                PreparedUpstreamProjectionWave::prepare(&prepared, &f, None)
                    .unwrap()
                    .route(),
                PreparedUpstreamProjectionRoute::StrictBase { .. }
            ));
        }
    }
}
