use super::*;
use ferrum_interfaces::execution_cost::{
    satisfied_completion_cost_signature, CanonicalWaveCostFeatures, CostSamplingHistoryScope,
    HostCostPolicyV2, HostCostStateV1, COST_NUMERIC_FEATURE_SCHEMA_V1,
};

fn n32(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}
fn n64(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}
fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}
fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
fn host(generated: u64) -> HostCostFeaturesV1 {
    HostCostFeaturesV1 {
        policy: HostCostPolicyV2 {
            empirical_content_domain: None,
            categorical_signature: [5; 32],
            decoder_text_bytes_per_token: 4,
            decoder_scratch_bytes_per_token: 8,
            raw_token_bytes_bound: 4,
        },
        state: HostCostStateV1 {
            generated_tokens_before: generated,
            maximum_output_tokens: 8,
            sampling_history_tokens: generated,
            sampling_history_scope: CostSamplingHistoryScope::FullGeneration,
            pending_decoded_utf8: false,
            completion_state_signature: satisfied_completion_cost_signature(),
        },
    }
}
fn shape(work: ActualRowWork) -> ProfileWaveShapeV2 {
    let (kind, decode, prefill, features) = match work {
        ActualRowWork::Decode { kv_tokens } => (
            ProfileWaveKind::Decode,
            vec![kv_tokens],
            vec![],
            project_host_cost_features(
                host(1),
                work,
                CostRowOutput::Decode {
                    requires_full_logits: false,
                    repetition_tokens: 0,
                    repetition_penalty_bits: 1.0_f32.to_bits(),
                },
            )
            .unwrap(),
        ),
        ActualRowWork::Prefill {
            offset,
            count,
            total_prompt_tokens,
        } => (
            ProfileWaveKind::Prefill,
            vec![],
            vec![ProfilePrefillShape {
                offset,
                count: n32(count),
                total_prompt_tokens: n32(total_prompt_tokens),
            }],
            project_host_cost_features(
                host(0),
                work,
                CostRowOutput::Prefill {
                    final_logits: offset + count == total_prompt_tokens,
                },
            )
            .unwrap(),
        ),
        _ => unreachable!(),
    };
    ProfileWaveShapeV2 {
        exact: ProfileWaveShape {
            kind,
            path: ProfileExecutionPath::PlanRuntime,
            provider_signature: [6; 32],
            output_policy_signature: [7; 32],
            graph_state: ProfileGraphState::Disabled,
            order: ProfileBatchOrder::Ordered,
            decode_kv_tokens: decode,
            prefill_chunks: prefill,
            recurrent_state_bytes: 16,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        },
        numeric_features: Some(CanonicalWaveCostFeatures {
            schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
            output_policy_signature: [8; 32],
            rows: vec![features],
        }),
    }
}
fn sample(
    record: u64,
    shape: ProfileWaveShapeV2,
    wall: u64,
    commit: ReferenceCommitReceipt,
) -> ReferenceObservedSample {
    ReferenceObservedSample {
        record: ReferenceRecordId {
            source_sha256: [9; 32],
            ordinal: record,
        },
        observation: ProfileSampleV2 {
            source_record: record,
            measured_unix_ns: 100 + record,
            shape,
            boundary: ProfileCostBoundary::PreparationToCommit,
            outcome: ProfileObservationOutcome::Completed {},
            timing: ProfileWaveTiming {
                wall_total_ns: wall,
                device_elapsed_ns: None,
                stages: Default::default(),
            },
        },
        commit,
    }
}
fn artifact() -> ReferenceCalibrationV1 {
    let decode_shape = shape(ActualRowWork::Decode { kv_tokens: 10 });
    let protocol = ReferenceProtocolV1 {
        granule_tokens: n32(4),
        repetitions: nz(3),
        estimator: ReferenceEstimator::UpperMedianWallV1,
        input_preprocessing_sha256: [10; 32],
        measurement_conditions_sha256: [11; 32],
        prefill_host: host(0),
        decode_host: host(1),
        decode_shape: decode_shape.clone(),
    };
    let mut builder = ReferenceCalibrationBuilder::new(
        n64(42),
        (&fingerprint()).into(),
        protocol,
        n64(1000),
        Default::default(),
    )
    .unwrap();
    builder
        .set_decode_samples(
            (0..3)
                .map(|index| {
                    sample(
                        index + 1,
                        decode_shape.clone(),
                        6 + index,
                        ReferenceCommitReceipt {
                            owner_incarnation: n64(index + 1),
                            work_generation: n64(2),
                            origin: ReferenceStateOrigin::PreparedDecode,
                            previous_record: None,
                            prefix_before: 10,
                            prefix_after: 11,
                            generated_before: 1,
                            generated_after: 2,
                        },
                    )
                })
                .collect(),
        )
        .unwrap();
    let partition = [(0, 4), (4, 4), (8, 2)]
        .into_iter()
        .map(|(offset, count)| {
            shape(ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens: 10,
            })
        })
        .collect::<Vec<_>>();
    let trials = (0..3)
        .map(|trial| {
            let mut previous = None;
            let samples = partition
                .iter()
                .enumerate()
                .map(|(index, shape)| {
                    let chunk = &shape.exact.prefill_chunks[0];
                    let output = sample(
                        (4 + trial * 3 + index) as u64,
                        shape.clone(),
                        [3, 8, 1][index] + trial as u64,
                        ReferenceCommitReceipt {
                            owner_incarnation: n64(100 + trial as u64),
                            work_generation: n64(index as u64 + 1),
                            origin: if index == 0 {
                                ReferenceStateOrigin::Fresh
                            } else {
                                ReferenceStateOrigin::CommittedContinuation
                            },
                            previous_record: previous,
                            prefix_before: chunk.offset,
                            prefix_after: chunk.offset + chunk.count.get(),
                            generated_before: 0,
                            generated_after: u32::from(index == 2),
                        },
                    );
                    previous = Some(output.record);
                    output
                })
                .collect();
            ReferencePrefillTrial {
                trial_index: trial,
                input_tokens_sha256: [12; 32],
                samples,
            }
        })
        .collect();
    builder
        .add_curve(ReferenceCurveInput {
            total_prompt_tokens: n32(10),
            partition,
            trials,
        })
        .unwrap();
    builder.finish().unwrap()
}
fn load(artifact: &ReferenceCalibrationV1) -> Result<Arc<LoadedPrefillReference>, ReferenceError> {
    load_prefill_reference_bytes(
        &serde_json::to_vec(artifact).unwrap(),
        &fingerprint(),
        artifact.protocol.sha256().unwrap(),
        &Default::default(),
    )
}

#[test]
fn measured_segments_form_nonuniform_curve_and_fixed_singleton_unit() {
    let loaded = load(&artifact()).unwrap();
    assert_eq!(loaded.tau_ref_ns().get(), 7);
    let curve = loaded.curve(n32(10)).unwrap();
    assert_eq!(curve.version, 42);
    assert_eq!(
        curve
            .points
            .iter()
            .map(|p| (p.prompt_tokens, p.cumulative_work_ns))
            .collect::<Vec<_>>(),
        vec![(0, 0), (4, 4), (8, 13), (10, 15)]
    );
    assert!(Arc::ptr_eq(&curve, &loaded.curve(n32(10)).unwrap()));
    assert_eq!(loaded.point_count(), 4);
    assert_eq!(loaded.sample_count(), 12);
    assert_eq!(loaded.generated_unix_ns(), 1000);
    assert_eq!(loaded.supported_lengths().collect::<Vec<_>>(), vec![10]);
    assert!(matches!(
        loaded.curve(n32(9)),
        Err(ReferenceUnknown::LengthNotCalibrated)
    ));
}

#[test]
fn legal_chunks_use_only_endpoints_and_real_tail_with_backend_bounds() {
    let loaded = load(&artifact()).unwrap();
    let limits = ReferenceChunkLimits {
        maximum_tokens: n32(10),
        alignment: n32(4),
        allow_final_short_chunk: true,
        maximum_candidates: nz(8),
    };
    assert_eq!(
        loaded
            .legal_chunks(n32(10), 0, limits)
            .unwrap()
            .iter()
            .map(|n| n.get())
            .collect::<Vec<_>>(),
        vec![4, 8, 10]
    );
    assert_eq!(
        loaded
            .legal_chunks(n32(10), 4, limits)
            .unwrap()
            .iter()
            .map(|n| n.get())
            .collect::<Vec<_>>(),
        vec![4, 6]
    );
    assert_eq!(
        loaded
            .legal_chunks(
                n32(10),
                0,
                ReferenceChunkLimits {
                    maximum_tokens: n32(5),
                    ..limits
                }
            )
            .unwrap(),
        vec![n32(4)]
    );
    assert_eq!(
        loaded.legal_chunks(
            n32(10),
            8,
            ReferenceChunkLimits {
                allow_final_short_chunk: false,
                ..limits
            }
        ),
        Err(ReferenceUnknown::NoLegalChunk)
    );
    assert_eq!(
        loaded.legal_chunks(n32(10), 1, limits),
        Err(ReferenceUnknown::MissingEndpoint)
    );
    assert_eq!(
        loaded.legal_chunks(
            n32(10),
            0,
            ReferenceChunkLimits {
                maximum_candidates: nz(65),
                ..limits
            }
        ),
        Err(ReferenceUnknown::PointBudget)
    );
}

#[test]
fn recompute_restored_baseline_and_original_admission_cannot_earn_duplicate_credit() {
    let loaded = load(&artifact()).unwrap();
    let id = RequestId::new();
    let mut binding = loaded
        .bind(id.clone(), n64(1), n32(10), 100, 200, 4)
        .unwrap();
    assert_eq!(binding.record_committed(&id, n64(1), 0, 4, false), Ok(0));
    assert_eq!(binding.record_committed(&id, n64(1), 4, 8, false), Ok(9));
    assert_eq!(binding.record_committed(&id, n64(1), 0, 8, false), Ok(0));
    let projected = binding.progress(0, 10, &[150, 200]).unwrap();
    assert_eq!(projected.admitted_at_ns, 100);
    assert_eq!(projected.reference_work_at_admission_ns, 4);
    assert_eq!(projected.logical_high_water, 8);
    assert_eq!(
        projected
            .milestones
            .iter()
            .map(|m| m.required_reference_work_ns)
            .collect::<Vec<_>>(),
        vec![4, 15]
    );
    assert_eq!(projected.ideal_reference_work_at(175, 200), Some(13));
    assert_eq!(
        binding.record_committed(&id, n64(2), 8, 10, true),
        Err(ReferenceUnknown::WrongIncarnation)
    );
    assert_eq!(
        binding.record_committed(&id, n64(1), 8, 10, false),
        Err(ReferenceUnknown::FinalTokenNotCommitted)
    );
    assert_eq!(binding.logical_high_water(), 8);
    assert_eq!(binding.record_committed(&id, n64(1), 8, 10, true), Ok(2));
    assert_eq!(binding.record_committed(&id, n64(1), 8, 10, true), Ok(0));
    assert!(binding.first_token_committed());
    assert_eq!(binding.admitted_at_ns(), 100);
    assert_eq!(binding.first_deadline_ns(), 200);
}

#[test]
fn merged_or_segmented_work_has_identical_net_credit() {
    let loaded = load(&artifact()).unwrap();
    let id = RequestId::new();
    let mut segmented = loaded.bind(id.clone(), n64(1), n32(10), 0, 100, 0).unwrap();
    let mut merged = segmented.clone();
    let first = segmented
        .record_committed(&id, n64(1), 0, 4, false)
        .unwrap();
    let second = segmented
        .record_committed(&id, n64(1), 4, 8, false)
        .unwrap();
    assert_eq!(
        first + second,
        merged.record_committed(&id, n64(1), 0, 8, false).unwrap()
    );
    assert_eq!(segmented.logical_high_water(), merged.logical_high_water());
    assert!(Arc::ptr_eq(segmented.reference(), merged.reference()));
}

#[test]
fn reference_is_unchanged_when_online_cost_model_publishes_new_versions() {
    use super::super::cost_model::*;
    let artifact = artifact();
    let loaded = load(&artifact).unwrap();
    let id = RequestId::new();
    let binding = loaded.bind(id, n64(1), n32(10), 0, 100, 0).unwrap();
    let identity = binding.identity();
    let curve = Arc::clone(binding.reference());
    let mut trainer = CostModelTrainer::new(fingerprint(), CostModelSettings::default()).unwrap();
    let mut shape: WaveExecutionShape = artifact.protocol.decode_shape.exact.clone().into();
    shape.numeric_features = artifact.protocol.decode_shape.numeric_features.clone();
    let observation = |at| WaveCostObservation {
        fingerprint: fingerprint(),
        actual_shape: shape.clone(),
        boundary: CostBoundary::PreparationToCommit,
        outcome: WaveObservationOutcome::Completed,
        timing: WaveTiming {
            wall_total_ns: 7,
            device_elapsed_ns: None,
            stages: Default::default(),
        },
        observed_at_ns: at,
    };
    trainer.observe(observation(1)).unwrap();
    let first = trainer.publish(2).unwrap().model_version();
    trainer.observe(observation(3)).unwrap();
    let second = trainer.publish(4).unwrap().model_version();
    assert!(second > first);
    assert_eq!(binding.identity(), identity);
    assert_eq!(binding.tau_ref_ns().get(), 7);
    assert!(Arc::ptr_eq(&curve, binding.reference()));
    assert_eq!(binding.reference().version, 42);
}

#[test]
fn mismatched_model_protocol_or_content_revision_is_not_the_same_calibration() {
    let artifact = artifact();
    let bytes = serde_json::to_vec(&artifact).unwrap();
    let mut wrong = fingerprint();
    wrong.device_runtime = [99; 32];
    assert!(matches!(
        load_prefill_reference_bytes(
            &bytes,
            &wrong,
            artifact.protocol.sha256().unwrap(),
            &Default::default()
        ),
        Err(ReferenceError::Incompatible)
    ));
    assert!(matches!(
        load_prefill_reference_bytes(&bytes, &fingerprint(), [99; 32], &Default::default()),
        Err(ReferenceError::Incompatible)
    ));
    let first = load(&artifact).unwrap();
    let mut changed = artifact.clone();
    changed.generated_unix_ns = n64(1001);
    let second = load(&changed).unwrap();
    assert_eq!(first.identity().revision, second.identity().revision);
    assert_ne!(first.identity(), second.identity());
}

#[test]
fn malformed_actual_receipt_chains_and_nonisolated_observations_cannot_build_work() {
    for case in 0..10 {
        let mut value = artifact();
        match case {
            0 => value.curves[0].trials[0].samples[1].commit.previous_record = None,
            1 => value.curves[0].trials[0].samples[2].commit.generated_after = 0,
            2 => {
                value.curves[0].trials[0].samples[0].commit.origin =
                    ReferenceStateOrigin::Recomputed
            }
            3 => {
                value.curves[0].trials[0].samples[0].observation.outcome =
                    ProfileObservationOutcome::NotSubmitted {}
            }
            4 => value.decode_samples[0].observation.boundary = ProfileCostBoundary::DeviceOnly,
            5 => value.curves[0].trials[1].input_tokens_sha256 = [99; 32],
            6 => value.curves[0].trials[0].samples[1].commit.work_generation = n64(1),
            7 => value.decode_samples[0].observation.timing.wall_total_ns = 0,
            8 => value.decode_samples[0].observation.shape.numeric_features = None,
            9 => value.curves[0].trials[1].trial_index = 0,
            _ => unreachable!(),
        }
        assert!(load(&value).is_err(), "invalid calibration case {case}");
    }
}

#[test]
fn missing_endpoints_duplicate_sources_and_cumulative_overflow_fail_closed() {
    let mut value = artifact();
    value.curves[0].partition.remove(1);
    for trial in &mut value.curves[0].trials {
        trial.samples.remove(1);
    }
    assert!(load(&value).is_err());
    let mut value = artifact();
    value.decode_samples[1] = value.decode_samples[0].clone();
    assert!(load(&value).is_err());
    let mut value = artifact();
    for trial in &mut value.curves[0].trials {
        trial.samples[0].observation.timing.wall_total_ns = u64::MAX;
    }
    assert!(matches!(load(&value), Err(ReferenceError::Overflow)));
}

#[test]
fn strict_wire_and_user_budgets_are_enforced_before_use() {
    let artifact = artifact();
    let digest = artifact.protocol.sha256().unwrap();
    let mut wire = serde_json::to_value(&artifact).unwrap();
    wire["fake_pass"] = serde_json::json!(true);
    assert!(matches!(
        load_prefill_reference_bytes(
            &serde_json::to_vec(&wire).unwrap(),
            &fingerprint(),
            digest,
            &Default::default()
        ),
        Err(ReferenceError::Json(_))
    ));
    let bytes = serde_json::to_vec(&artifact).unwrap();
    let limits = SloPrefillReferenceLimits {
        max_file_bytes: nz(bytes.len() - 1),
        ..Default::default()
    };
    assert!(matches!(
        load_prefill_reference_bytes(&bytes, &fingerprint(), digest, &limits),
        Err(ReferenceError::Limit(_))
    ));
    let limits = SloPrefillReferenceLimits {
        max_points_per_curve: nz(3),
        ..Default::default()
    };
    assert!(matches!(
        load_prefill_reference_bytes(&bytes, &fingerprint(), digest, &limits),
        Err(ReferenceError::Limit(_))
    ));
    let limits = SloPrefillReferenceLimits {
        max_samples: nz(11),
        ..Default::default()
    };
    assert!(matches!(
        load_prefill_reference_bytes(&bytes, &fingerprint(), digest, &limits),
        Err(ReferenceError::Limit(_))
    ));
}

#[test]
fn builder_rejects_incomplete_repetitions_and_duplicate_lengths() {
    let value = artifact();
    let mut builder = ReferenceCalibrationBuilder::new(
        value.reference_revision,
        value.fingerprint.clone(),
        value.protocol.clone(),
        value.generated_unix_ns,
        Default::default(),
    )
    .unwrap();
    assert!(builder
        .set_decode_samples(value.decode_samples[..2].to_vec())
        .is_err());
    builder.set_decode_samples(value.decode_samples).unwrap();
    builder.add_curve(value.curves[0].clone()).unwrap();
    assert!(builder.add_curve(value.curves[0].clone()).is_err());
    assert!(builder.finish().is_ok());
}

#[path = "tests/piecewise.rs"]
mod piecewise;
