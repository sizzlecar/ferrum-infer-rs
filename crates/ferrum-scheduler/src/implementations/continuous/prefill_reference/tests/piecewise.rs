use super::*;

fn specification() -> PiecewiseReferenceSpec {
    PiecewiseReferenceSpec {
        minimum_prompt_tokens: n32(1),
        maximum_prompt_tokens: n32(10),
        body_endpoints: vec![n32(4), n32(9)],
    }
}
fn builder() -> ReferenceCalibrationBuilder {
    let old = artifact();
    let mut builder = ReferenceCalibrationBuilder::new(
        old.reference_revision,
        old.fingerprint,
        old.protocol,
        old.generated_unix_ns,
        Default::default(),
    )
    .unwrap()
    .with_piecewise(specification())
    .unwrap();
    builder.set_decode_samples(old.decode_samples).unwrap();
    let mut ordinal = 20;
    for n in [1, 5, 10] {
        let mut offset = 0;
        let mut partition = Vec::new();
        while offset < n {
            let count = specification().next_count(n, offset).unwrap().get();
            partition.push(shape(ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens: n,
            }));
            offset += count;
        }
        let trials = (0..3)
            .map(|trial| {
                let mut previous = None;
                let samples = partition
                    .iter()
                    .enumerate()
                    .map(|(i, shape)| {
                        let chunk = &shape.exact.prefill_chunks[0];
                        let end = chunk.offset + chunk.count.get();
                        // Distinct body slopes; final values deliberately need not
                        // increase with N. They remain a positive final-only unit.
                        let wall = if end == n {
                            80 - u64::from(n)
                        } else {
                            u64::from(chunk.count.get()) * (10 + i as u64 * 10)
                        };
                        let value = sample(
                            ordinal,
                            shape.clone(),
                            wall,
                            ReferenceCommitReceipt {
                                owner_incarnation: n64(1000 + u64::from(n) * 10 + trial as u64),
                                work_generation: n64(i as u64 + 1),
                                origin: if i == 0 {
                                    ReferenceStateOrigin::Fresh
                                } else {
                                    ReferenceStateOrigin::CommittedContinuation
                                },
                                previous_record: previous,
                                prefix_before: chunk.offset,
                                prefix_after: end,
                                generated_before: 0,
                                generated_after: u32::from(end == n),
                            },
                        );
                        ordinal += 1;
                        previous = Some(value.record);
                        value
                    })
                    .collect();
                ReferencePrefillTrial {
                    trial_index: trial,
                    input_tokens_sha256: [n as u8; 32],
                    samples,
                }
            })
            .collect();
        builder
            .add_curve(ReferenceCurveInput {
                total_prompt_tokens: n32(n),
                partition,
                trials,
            })
            .unwrap();
    }
    builder
}
fn loaded() -> Arc<LoadedPrefillReference> {
    let bytes = builder().finish_bytes().unwrap();
    let digest = specification()
        .protocol_sha256(&artifact().protocol)
        .unwrap();
    load_prefill_reference_bytes(&bytes, &fingerprint(), digest, &Default::default()).unwrap()
}

#[test]
fn piecewise_real_segment_evidence_covers_unmeasured_lengths_and_endpoints() {
    let value = loaded();
    assert_eq!(value.piecewise_domain(), Some((n32(1), n32(10))));
    assert_eq!(
        value.supported_lengths().collect::<Vec<_>>(),
        vec![1, 5, 10]
    );
    let curve = value.curve(n32(7)).unwrap();
    assert_eq!(curve.work_at(3), Some(30));
    assert_eq!(curve.work_at(6), Some(80));
    assert_eq!(curve.work_at(7), Some(153)); // B(6)=80, actual F interpolation=73
    assert_eq!(curve.work_at(8), None);
    assert_eq!(value.curve(n32(1)).unwrap().work_at(1), Some(79));
    assert!(matches!(
        value.curve(n32(11)),
        Err(ReferenceUnknown::LengthNotCalibrated)
    ));
    assert_eq!(value.tau_ref_ns().get(), 7);
    // V1 bytes still do not interpolate an unmeasured N or endpoint.
    let old = load(&artifact()).unwrap();
    assert!(old.curve(n32(7)).is_err());
    assert_eq!(old.curve(n32(10)).unwrap().work_at(3), None);
}

#[test]
fn piecewise_quantization_is_one_legal_granule_and_recompute_earns_no_credit() {
    let value = loaded();
    let id = RequestId::new();
    let mut bound = value
        .bind(id.clone(), n64(1), n32(7), 100, 1100, 0)
        .unwrap();
    assert_eq!(
        bound.record_committed(&id, n64(1), 0, 3, false).unwrap(),
        30
    );
    assert_eq!(bound.record_committed(&id, n64(1), 0, 2, false).unwrap(), 0);
    assert_eq!(bound.logical_high_water(), 3);
    assert_eq!(bound.admitted_at_ns(), 100);
    assert!(bound.progress(0, 7, &[600]).is_err());
    let progress = bound
        .progress_with_granule(0, 7, &[600, 1100], Some(n32(1)))
        .unwrap();
    // ideal=ceil(153/2)=77; B(5)=60, B(6)=80. No 100ns measured-segment Q.
    assert_eq!(progress.milestones[0].required_reference_work_ns, 60);
    assert_eq!(progress.milestones[1].required_reference_work_ns, 153);
    assert!(bound.record_committed(&id, n64(1), 3, 7, false).is_err());
    assert_eq!(
        bound.record_committed(&id, n64(1), 3, 7, true).unwrap(),
        123
    );
}

#[test]
fn piecewise_legal_counts_are_not_limited_to_measured_endpoints() {
    let value = loaded();
    assert_eq!(
        value
            .legal_chunks(
                n32(7),
                0,
                ReferenceChunkLimits {
                    maximum_tokens: n32(6),
                    alignment: n32(2),
                    allow_final_short_chunk: true,
                    maximum_candidates: nz(8)
                }
            )
            .unwrap(),
        vec![n32(2), n32(4), n32(6)]
    );
    assert_eq!(
        value
            .legal_chunks(
                n32(7),
                6,
                ReferenceChunkLimits {
                    maximum_tokens: n32(4),
                    alignment: n32(2),
                    allow_final_short_chunk: true,
                    maximum_candidates: nz(8)
                }
            )
            .unwrap(),
        vec![n32(1)]
    );
}

#[test]
fn piecewise_chunk_ladder_only_enriches_legally_accounted_progress() {
    let value = loaded();
    let chunks = value
        .legal_chunks(
            n32(10),
            0,
            ReferenceChunkLimits {
                maximum_tokens: n32(8),
                alignment: n32(1),
                allow_final_short_chunk: true,
                maximum_candidates: nz(64),
            },
        )
        .unwrap();
    assert!(
        chunks.contains(&n32(2)),
        "an intermediate action absent from the old three anchors"
    );
    let curve = value.curve(n32(10)).unwrap();
    for chunk in chunks {
        assert!(curve.work_at(chunk.get()).is_some());
    }
    // A ladder cannot create reference coverage, bypass alignment at the
    // current frontier, or interpolate a V1 exact endpoint.
    let limits = ReferenceChunkLimits {
        maximum_tokens: n32(8),
        alignment: n32(2),
        allow_final_short_chunk: true,
        maximum_candidates: nz(64),
    };
    assert!(matches!(
        value.legal_chunks(n32(11), 0, limits),
        Err(ReferenceUnknown::LengthNotCalibrated)
    ));
    assert!(matches!(
        value.legal_chunks(n32(10), 1, limits),
        Err(ReferenceUnknown::NoLegalChunk)
    ));
    let exact = load(&artifact()).unwrap();
    assert!(!exact
        .legal_chunks(n32(10), 0, limits)
        .unwrap()
        .contains(&n32(2)));
}

#[test]
fn piecewise_wire_cannot_relabel_v1_or_extrapolate_missing_terminal_anchor() {
    let bytes = builder().finish_bytes().unwrap();
    let digest = specification()
        .protocol_sha256(&artifact().protocol)
        .unwrap();
    let mut wire: ReferenceCalibrationV2 = serde_json::from_slice(&bytes).unwrap();
    wire.curves
        .retain(|curve| curve.total_prompt_tokens.get() != 10);
    assert!(load_prefill_reference_bytes(
        &serde_json::to_vec(&wire).unwrap(),
        &fingerprint(),
        digest,
        &Default::default()
    )
    .is_err());
    assert!(load_prefill_reference_bytes(
        &bytes,
        &fingerprint(),
        artifact().protocol.sha256().unwrap(),
        &Default::default()
    )
    .is_err());
    let mut generic: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    generic["schema_version"] = 1.into();
    assert!(load_prefill_reference_bytes(
        &serde_json::to_vec(&generic).unwrap(),
        &fingerprint(),
        digest,
        &Default::default()
    )
    .is_err());
}

#[test]
fn piecewise_partition_and_integer_inverse_are_checked() {
    let mut spec = specification();
    spec.body_endpoints = vec![n32(9), n32(4)];
    assert!(spec.validate().is_err());
    let value = loaded();
    for n in 1..=10 {
        let curve = value.curve(n32(n)).unwrap();
        for p in 0..n {
            let work = curve.work_at(p).unwrap();
            assert_eq!(curve.nonfinal_prefix_at_or_below(work), Some(p));
            assert!(curve.work_at(p + 1).unwrap() > work);
        }
    }
    use super::super::super::slo_planner::interpolate_reference_work;
    let points = [
        ReferenceWorkPoint {
            prompt_tokens: 0,
            cumulative_work_ns: u64::MAX,
        },
        ReferenceWorkPoint {
            prompt_tokens: u32::MAX,
            cumulative_work_ns: u64::MAX,
        },
    ];
    assert_eq!(interpolate_reference_work(&points, 123), Some(u64::MAX));
}
