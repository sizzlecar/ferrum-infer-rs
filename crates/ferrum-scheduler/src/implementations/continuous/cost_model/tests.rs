use super::*;

fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}

fn settings() -> CostModelSettings {
    CostModelSettings {
        min_samples: NonZeroUsize::new(3).unwrap(),
        max_samples_per_bucket: NonZeroUsize::new(8).unwrap(),
        drift_margin_ns: 10,
        max_wave_ns: NonZeroU64::new(10_000).unwrap(),
        max_sample_age_ns: NonZeroU64::new(1000).unwrap(),
        ..Default::default()
    }
}

fn decode(contexts: &[u32]) -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: None,
        numeric_features: None,
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [5; 32],
        output_policy_signature: [6; 32],
        graph_state: WaveGraphState::Warm,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: contexts.to_vec(),
        prefill_chunks: Vec::new(),
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}

fn prefill(offset: u32, count: u32, total: u32) -> WaveExecutionShape {
    WaveExecutionShape {
        kind: WaveKind::Prefill,
        decode_kv_tokens: Vec::new(),
        prefill_chunks: vec![PrefillShape {
            offset,
            count: NonZeroU32::new(count).unwrap(),
            total_prompt_tokens: NonZeroU32::new(total).unwrap(),
        }],
        ..decode(&[])
    }
}

fn observation(shape: WaveExecutionShape, cost: u64, at: u64) -> WaveCostObservation {
    WaveCostObservation {
        fingerprint: fingerprint(),
        actual_shape: shape,
        boundary: CostBoundary::PreparationToCommit,
        outcome: WaveObservationOutcome::Completed,
        timing: WaveTiming {
            wall_total_ns: cost,
            device_elapsed_ns: None,
            stages: WaveStageTimings::default(),
        },
        observed_at_ns: at,
    }
}

fn train(trainer: &mut CostModelTrainer, shape: &WaveExecutionShape, costs: &[u64], first_at: u64) {
    for (index, &cost) in costs.iter().enumerate() {
        assert_eq!(
            trainer
                .observe(observation(shape.clone(), cost, first_at + index as u64))
                .unwrap(),
            ObservationDisposition::Recorded
        );
    }
}

fn prediction(
    snapshot: &CostModelSnapshot,
    shape: &WaveExecutionShape,
    now: u64,
) -> WaveCostPrediction {
    match snapshot.predict(
        &fingerprint(),
        shape,
        CostBoundary::PreparationToCommit,
        now,
    ) {
        CostPrediction::Known(prediction) => prediction,
        unknown => panic!("expected calibrated prediction, got {unknown:?}"),
    }
}

#[test]
fn empirical_quantile_and_drift_use_actual_wall_measurements() {
    let shape = decode(&[1024, 4096]);
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    train(&mut trainer, &shape, &[100, 200, 300], 1);
    let snapshot = trainer.publish(4).unwrap();
    let value = prediction(&snapshot, &shape, 4);
    assert_eq!(value.typical_ns, 200);
    assert_eq!(value.residual_margin_ns, 100);
    assert_eq!(value.drift_margin_ns, 10);
    assert_eq!(value.planning_ns, 310);
    assert_eq!(value.sample_count, 3);
    assert_eq!(value.model_version, 1);
    assert_eq!(
        value.coverage.decode_context_ranges,
        [(1024, 1024), (4096, 4096)]
    );
}

#[test]
fn prediction_reports_remaining_ttl_at_the_query_time_including_exact_expiry() {
    let shape = decode(&[128]);
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    train(&mut trainer, &shape, &[100; 3], 1);
    let snapshot = trainer.publish(4).unwrap();
    assert_eq!(prediction(&snapshot, &shape, 4).valid_for_ns, 997);
    assert_eq!(prediction(&snapshot, &shape, 1000).valid_for_ns, 1);
    assert_eq!(prediction(&snapshot, &shape, 1001).valid_for_ns, 0);
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &shape,
            CostBoundary::PreparationToCommit,
            1002
        ),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}

#[test]
fn unknown_never_becomes_zero_cost_for_missing_sparse_or_mismatched_evidence() {
    let shape = decode(&[128]);
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    train(&mut trainer, &shape, &[100, 200], 1);
    let snapshot = trainer.publish(3).unwrap();
    assert_eq!(
        snapshot.predict(&fingerprint(), &shape, CostBoundary::PreparationToCommit, 3),
        CostPrediction::Unknown(CostUnknownReason::InsufficientSamples)
    );
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &decode(&[129]),
            CostBoundary::PreparationToCommit,
            3
        ),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    let mut mismatch = fingerprint();
    mismatch.numerical_policy[0] ^= 1;
    assert_eq!(
        snapshot.predict(&mismatch, &shape, CostBoundary::PreparationToCommit, 3),
        CostPrediction::Unknown(CostUnknownReason::FingerprintMismatch)
    );
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &decode(&[0]),
            CostBoundary::PreparationToCommit,
            3
        ),
        CostPrediction::Unknown(CostUnknownReason::InvalidShape)
    );
}

#[test]
fn fallback_provider_graph_and_output_policies_have_distinct_buckets() {
    let shape = decode(&[128]);
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    train(&mut trainer, &shape, &[100; 3], 1);
    let snapshot = trainer.publish(4).unwrap();
    let mut variants = vec![shape.clone(); 6];
    variants[0].path = WaveExecutionPath::UnsupportedFallback;
    variants[1].path = WaveExecutionPath::CapacityFallback;
    variants[2].provider_signature[0] ^= 1;
    variants[3].output_policy_signature[0] ^= 1;
    variants[4].graph_state = WaveGraphState::Cold;
    variants[5].recurrent_state_bytes = 4096;
    for variant in variants {
        assert_eq!(
            snapshot.predict(
                &fingerprint(),
                &variant,
                CostBoundary::PreparationToCommit,
                4
            ),
            CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
        );
    }
}

#[test]
fn only_executor_declared_independent_rows_are_canonicalized() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    let ordered = decode(&[128, 512]);
    train(&mut trainer, &ordered, &[100; 3], 1);
    let mut independent = ordered.clone();
    independent.order = BatchOrderSemantics::IndependentRows;
    train(&mut trainer, &independent, &[200; 3], 4);
    let snapshot = trainer.publish(7).unwrap();
    let mut reordered = ordered.clone();
    reordered.decode_kv_tokens.reverse();
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &reordered,
            CostBoundary::PreparationToCommit,
            7
        ),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    reordered.order = BatchOrderSemantics::IndependentRows;
    assert_eq!(prediction(&snapshot, &reordered, 7).typical_ns, 200);
}

#[test]
fn explicit_bucketing_interpolates_only_inside_observed_coverage() {
    let settings = CostModelSettings {
        context_bucket_tokens: NonZeroU32::new(128).unwrap(),
        ..settings()
    };
    let mut trainer = CostModelTrainer::new(fingerprint(), settings).unwrap();
    for (at, context) in [(1, 132), (2, 140), (3, 150)] {
        trainer
            .observe(observation(decode(&[context]), 100, at))
            .unwrap();
    }
    let snapshot = trainer.publish(4).unwrap();
    assert_eq!(prediction(&snapshot, &decode(&[144]), 4).sample_count, 3);
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &decode(&[151]),
            CostBoundary::PreparationToCommit,
            4
        ),
        CostPrediction::Unknown(CostUnknownReason::OutsideObservedCoverage)
    );
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &decode(&[256]),
            CostBoundary::PreparationToCommit,
            4
        ),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}

#[test]
fn prefill_final_fragment_is_not_merged_with_intermediate_fragment() {
    let settings = CostModelSettings {
        prefill_offset_bucket_tokens: NonZeroU32::new(128).unwrap(),
        ..settings()
    };
    let mut trainer = CostModelTrainer::new(fingerprint(), settings).unwrap();
    let intermediate = prefill(10, 10, 30);
    train(&mut trainer, &intermediate, &[100; 3], 1);
    let snapshot = trainer.publish(4).unwrap();
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &prefill(20, 10, 30),
            CostBoundary::PreparationToCommit,
            4
        ),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &prefill(11, 10, 30),
            CostBoundary::PreparationToCommit,
            4
        ),
        CostPrediction::Unknown(CostUnknownReason::OutsideObservedCoverage)
    );
}

#[test]
fn wall_and_device_boundaries_are_calibrated_independently() {
    let shape = decode(&[128]);
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    for at in 1..=3 {
        let mut obs = observation(shape.clone(), 100, at);
        obs.timing.device_elapsed_ns = Some(60);
        obs.timing.stages = WaveStageTimings {
            prepare: Some(MeasuredSpan {
                start_ns: 0,
                end_ns: 10,
            }),
            device_wait: Some(MeasuredSpan {
                start_ns: 10,
                end_ns: 75,
            }),
            commit: Some(MeasuredSpan {
                start_ns: 80,
                end_ns: 90,
            }),
            ..Default::default()
        };
        trainer.observe(obs.clone()).unwrap();
        obs.boundary = CostBoundary::DeviceOnly;
        trainer.observe(obs).unwrap();
    }
    let snapshot = trainer.publish(4).unwrap();
    assert_eq!(prediction(&snapshot, &shape, 4).typical_ns, 100);
    let CostPrediction::Known(device) =
        snapshot.predict(&fingerprint(), &shape, CostBoundary::DeviceOnly, 4)
    else {
        panic!("missing device cost")
    };
    assert_eq!(device.typical_ns, 60);
    assert_eq!(device.planning_ns, 70);
}

#[test]
fn invalid_or_overlapping_stage_timings_are_rejected_atomically() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    for stages in [
        WaveStageTimings {
            prepare: Some(MeasuredSpan {
                start_ns: 20,
                end_ns: 10,
            }),
            ..Default::default()
        },
        WaveStageTimings {
            commit: Some(MeasuredSpan {
                start_ns: 90,
                end_ns: 101,
            }),
            ..Default::default()
        },
        WaveStageTimings {
            prepare: Some(MeasuredSpan {
                start_ns: 0,
                end_ns: 30,
            }),
            device_wait: Some(MeasuredSpan {
                start_ns: 20,
                end_ns: 80,
            }),
            ..Default::default()
        },
    ] {
        let mut obs = observation(decode(&[128]), 100, 1);
        obs.timing.stages = stages;
        assert!(matches!(
            trainer.observe(obs),
            Err(CostModelError::InvalidTiming(_))
        ));
    }
    for (wall, device) in [(0, None), (10_001, None), (100, Some(0)), (100, Some(101))] {
        let mut obs = observation(decode(&[128]), wall, 1);
        obs.timing.device_elapsed_ns = device;
        assert!(matches!(
            trainer.observe(obs),
            Err(CostModelError::InvalidTiming(_))
        ));
    }
    assert_eq!(trainer.retained_sample_count(), 0);
}

#[test]
fn not_submitted_deferred_failed_and_unisolated_partial_are_not_success_samples() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    for (outcome, expected) in [
        (
            WaveObservationOutcome::NotSubmitted,
            ObservationSkipReason::NotSubmitted,
        ),
        (
            WaveObservationOutcome::Deferred,
            ObservationSkipReason::Deferred,
        ),
        (
            WaveObservationOutcome::FailedAfterSubmit,
            ObservationSkipReason::FailedAfterSubmit,
        ),
        (
            WaveObservationOutcome::PartiallyCompleted {
                timing_covers_actual_shape_only: false,
            },
            ObservationSkipReason::UnisolatedPartial,
        ),
    ] {
        let mut obs = observation(decode(&[]), 0, 1);
        obs.outcome = outcome;
        assert_eq!(
            trainer.observe(obs).unwrap(),
            ObservationDisposition::Skipped(expected)
        );
    }
    let mut missing_device = observation(decode(&[128]), 100, 1);
    missing_device.boundary = CostBoundary::DeviceOnly;
    assert_eq!(
        trainer.observe(missing_device).unwrap(),
        ObservationDisposition::Skipped(ObservationSkipReason::MissingDeviceTiming)
    );
    assert_eq!(trainer.retained_sample_count(), 0);
    assert_eq!(trainer.publish(1).unwrap().bucket_count(), 0);
}

#[test]
fn isolated_partial_trains_only_actual_completed_shape_and_actual_path() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    let mut actual = prefill(0, 8, 32);
    actual.path = WaveExecutionPath::CapacityFallback;
    for at in 1..=3 {
        let mut obs = observation(actual.clone(), 100, at);
        obs.outcome = WaveObservationOutcome::PartiallyCompleted {
            timing_covers_actual_shape_only: true,
        };
        trainer.observe(obs).unwrap();
    }
    let snapshot = trainer.publish(4).unwrap();
    assert_eq!(prediction(&snapshot, &actual, 4).sample_count, 3);
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &prefill(0, 32, 32),
            CostBoundary::PreparationToCommit,
            4
        ),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}

#[test]
fn stale_samples_and_foreign_clock_offsets_cannot_produce_predictions() {
    let shape = decode(&[128]);
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    train(&mut trainer, &shape, &[100; 3], 1);
    let snapshot = trainer.publish(4).unwrap();
    assert_eq!(
        snapshot.predict(&fingerprint(), &shape, CostBoundary::PreparationToCommit, 3),
        CostPrediction::Unknown(CostUnknownReason::ClockMovedBackwards)
    );
    assert_eq!(
        snapshot.predict(
            &fingerprint(),
            &shape,
            CostBoundary::PreparationToCommit,
            1002
        ),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
    assert!(matches!(
        trainer.observe(observation(shape.clone(), 100, 3)),
        Err(CostModelError::ClockMovedBackwards)
    ));
    assert!(matches!(
        trainer.publish(3),
        Err(CostModelError::ClockMovedBackwards)
    ));
    trainer
        .observe(observation(shape.clone(), 100, 2000))
        .unwrap();
    let sparse_fresh = trainer.publish(2000).unwrap();
    assert_eq!(
        sparse_fresh.predict(
            &fingerprint(),
            &shape,
            CostBoundary::PreparationToCommit,
            2000
        ),
        // Fully expired samples are reclaimed on valid observation. The one
        // genuinely fresh replacement still cannot satisfy min_samples.
        CostPrediction::Unknown(CostUnknownReason::InsufficientSamples)
    );
}

#[test]
fn publishing_is_immutable_and_versions_only_change_for_new_evidence() {
    let shape = decode(&[128]);
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    train(&mut trainer, &shape, &[100; 3], 1);
    let old = trainer.publish(4).unwrap();
    let same = trainer.publish(5).unwrap();
    assert!(Arc::ptr_eq(&old, &same));
    trainer.raise_drift_margin_ns(10).unwrap();
    assert!(Arc::ptr_eq(&old, &trainer.publish(6).unwrap()));
    trainer.observe(observation(shape.clone(), 500, 7)).unwrap();
    let new = trainer.publish(8).unwrap();
    assert_eq!(old.model_version(), 1);
    assert_eq!(new.model_version(), 2);
    assert_eq!(prediction(&old, &shape, 8).planning_ns, 110);
    let value = prediction(&new, &shape, 8);
    assert_eq!(value.planning_ns, 510);
    assert_eq!(value.errors.compared_samples, 1);
    assert_eq!(value.errors.underestimates, 1);
    assert_eq!(value.errors.max_underestimate_ns, 390);
}

#[test]
fn margins_and_planning_floors_only_decrease_in_a_fresh_calibration_epoch() {
    let shape = decode(&[128]);
    let settings = CostModelSettings {
        max_samples_per_bucket: NonZeroUsize::new(3).unwrap(),
        ..settings()
    };
    let mut trainer = CostModelTrainer::new(fingerprint(), settings.clone()).unwrap();
    train(&mut trainer, &shape, &[500; 3], 1);
    let old = trainer.publish(4).unwrap();
    train(&mut trainer, &shape, &[100; 3], 5);
    let new = trainer.publish(8).unwrap();
    assert_eq!(prediction(&new, &shape, 8).typical_ns, 100);
    assert_eq!(prediction(&new, &shape, 8).planning_ns, 510);
    assert_eq!(prediction(&new, &shape, 8).prior_planning_floor_ns, 510);
    assert_eq!(
        trainer.raise_drift_margin_ns(0),
        Err(CostModelError::MarginDecreaseRequiresRecalibration)
    );
    trainer.raise_drift_margin_ns(1000).unwrap();
    assert_eq!(
        prediction(&trainer.publish(9).unwrap(), &shape, 9).planning_ns,
        1100
    );
    let mut recalibrated =
        CostModelTrainer::after_version(fingerprint(), settings, old.model_version() + 2).unwrap();
    train(&mut recalibrated, &shape, &[100; 3], 1);
    let fresh = recalibrated.publish(4).unwrap();
    assert_eq!(fresh.model_version(), 4);
    assert_eq!(prediction(&fresh, &shape, 4).planning_ns, 110);
}

#[test]
fn version_exhaustion_does_not_replace_last_snapshot() {
    let shape = decode(&[128]);
    let mut trainer =
        CostModelTrainer::after_version(fingerprint(), settings(), u64::MAX - 1).unwrap();
    train(&mut trainer, &shape, &[100; 3], 1);
    let last = trainer.publish(4).unwrap();
    assert_eq!(last.model_version(), u64::MAX);
    assert!(Arc::ptr_eq(&last, &trainer.publish(5).unwrap()));
    trainer.observe(observation(shape.clone(), 200, 6)).unwrap();
    assert_eq!(
        trainer.publish(7).unwrap_err(),
        CostModelError::VersionExhausted
    );
    assert_eq!(prediction(&last, &shape, 7).planning_ns, 110);
}

#[test]
fn invalid_quantiles_and_overflowing_settings_are_rejected() {
    for q in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 0.0, 1.01] {
        let config = CostModelSettings {
            residual_quantile: q,
            ..settings()
        };
        assert!(matches!(
            CostModelTrainer::new(fingerprint(), config),
            Err(CostModelError::InvalidSettings(_))
        ));
    }
    let config = CostModelSettings {
        max_wave_ns: NonZeroU64::new(u64::MAX).unwrap(),
        ..settings()
    };
    assert_eq!(
        CostModelTrainer::new(fingerprint(), config).unwrap_err(),
        CostModelError::ArithmeticOverflow
    );
    let config = CostModelSettings {
        max_buckets: NonZeroUsize::new(usize::MAX).unwrap(),
        ..settings()
    };
    assert!(matches!(
        CostModelTrainer::new(fingerprint(), config),
        Err(CostModelError::InvalidSettings(_))
    ));
    let config = CostModelSettings {
        min_samples: NonZeroUsize::new(9).unwrap(),
        ..settings()
    };
    assert!(matches!(
        CostModelTrainer::new(fingerprint(), config),
        Err(CostModelError::InvalidSettings(_))
    ));
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    assert_eq!(
        trainer.raise_drift_margin_ns(u64::MAX),
        Err(CostModelError::ArithmeticOverflow)
    );
}

#[test]
fn malformed_shapes_and_overflow_do_not_consume_retention_capacity() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    let mut oversized_rows = decode(&vec![1; 129]);
    let mut invalids = vec![
        decode(&[]),
        decode(&[0]),
        decode(&[262_145]),
        prefill(20, 20, 30),
        prefill(u32::MAX, 1, u32::MAX),
    ];
    invalids.push(oversized_rows.clone());
    oversized_rows.decode_kv_tokens = vec![1];
    oversized_rows.recurrent_state_bytes = u64::MAX;
    oversized_rows.restore_bytes = 1;
    invalids.push(oversized_rows);
    let mut kind = decode(&[1]);
    kind.kind = WaveKind::Mixed;
    invalids.push(kind);
    for shape in invalids {
        assert!(matches!(
            trainer.observe(observation(shape, 100, 1)),
            Err(CostModelError::InvalidShape(_)) | Err(CostModelError::ArithmeticOverflow)
        ));
    }
    let mut mismatch = observation(decode(&[1]), 100, 1);
    mismatch.fingerprint.model_weights[0] ^= 1;
    assert_eq!(
        trainer.observe(mismatch),
        Err(CostModelError::FingerprintMismatch)
    );
    assert_eq!(trainer.retained_sample_count(), 0);
    assert_eq!(trainer.retained_shape_rows(), 0);
}

#[test]
fn retention_limits_are_atomic_and_per_bucket_ring_replacement_is_bounded() {
    let settings = CostModelSettings {
        max_buckets: NonZeroUsize::new(1).unwrap(),
        max_samples_per_bucket: NonZeroUsize::new(3).unwrap(),
        max_retained_samples: NonZeroUsize::new(3).unwrap(),
        ..settings()
    };
    let shape = decode(&[128]);
    let mut trainer = CostModelTrainer::new(fingerprint(), settings).unwrap();
    train(&mut trainer, &shape, &[100; 20], 1);
    assert_eq!(trainer.retained_sample_count(), 3);
    assert_eq!(trainer.retained_shape_rows(), 3);
    assert_eq!(
        trainer.observe(observation(decode(&[256]), 100, 21)),
        Err(CostModelError::CapacityExceeded("buckets"))
    );
    assert_eq!(
        prediction(&trainer.publish(21).unwrap(), &shape, 21).sample_count,
        3
    );
}

#[test]
fn global_sample_and_shape_row_limits_are_enforced_across_buckets() {
    let sample_settings = CostModelSettings {
        max_retained_samples: NonZeroUsize::new(3).unwrap(),
        ..settings()
    };
    let mut trainer = CostModelTrainer::new(fingerprint(), sample_settings).unwrap();
    train(&mut trainer, &decode(&[1]), &[100; 3], 1);
    assert_eq!(
        trainer.observe(observation(decode(&[2]), 100, 4)),
        Err(CostModelError::CapacityExceeded("retained samples"))
    );
    let mut settings = CostModelSettings {
        max_retained_shape_rows: NonZeroUsize::new(3).unwrap(),
        ..settings()
    };
    settings.shape_limits.max_rows = NonZeroUsize::new(3).unwrap();
    let mut trainer = CostModelTrainer::new(fingerprint(), settings).unwrap();
    trainer
        .observe(observation(decode(&[1, 2]), 100, 1))
        .unwrap();
    assert_eq!(
        trainer.observe(observation(decode(&[1, 2]), 100, 2)),
        Err(CostModelError::CapacityExceeded("retained shape rows"))
    );
    assert_eq!(trainer.retained_sample_count(), 1);
    assert_eq!(trainer.retained_shape_rows(), 2);
}

#[test]
fn restore_maintenance_and_mixed_waves_require_real_nonempty_work() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    let mut restore = decode(&[]);
    restore.kind = WaveKind::Restore;
    restore.restore_bytes = 4096;
    train(&mut trainer, &restore, &[100; 3], 1);
    let mut maintenance = decode(&[]);
    maintenance.kind = WaveKind::Maintenance;
    maintenance.maintenance_units = 1;
    train(&mut trainer, &maintenance, &[200; 3], 4);
    let mut mixed = prefill(0, 8, 32);
    mixed.kind = WaveKind::Mixed;
    mixed.decode_kv_tokens.push(1024);
    train(&mut trainer, &mixed, &[300; 3], 7);
    let snapshot = trainer.publish(10).unwrap();
    assert_eq!(prediction(&snapshot, &restore, 10).typical_ns, 100);
    assert_eq!(prediction(&snapshot, &maintenance, 10).typical_ns, 200);
    assert_eq!(prediction(&snapshot, &mixed, 10).typical_ns, 300);
    restore.restore_bytes = 0;
    maintenance.maintenance_units = 0;
    assert!(matches!(
        trainer.observe(observation(restore, 100, 11)),
        Err(CostModelError::InvalidShape(_))
    ));
    assert!(matches!(
        trainer.observe(observation(maintenance, 100, 11)),
        Err(CostModelError::InvalidShape(_))
    ));
}
