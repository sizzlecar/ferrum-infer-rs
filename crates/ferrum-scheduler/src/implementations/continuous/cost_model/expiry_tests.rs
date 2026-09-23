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
        max_buckets: NonZeroUsize::new(1).unwrap(),
        min_samples: NonZeroUsize::new(1).unwrap(),
        max_samples_per_bucket: NonZeroUsize::new(3).unwrap(),
        max_retained_samples: NonZeroUsize::new(3).unwrap(),
        max_retained_shape_rows: NonZeroUsize::new(4).unwrap(),
        max_sample_age_ns: NonZeroU64::new(10).unwrap(),
        max_wave_ns: NonZeroU64::new(1000).unwrap(),
        drift_margin_ns: 10,
        shape_limits: CostShapeLimits {
            max_rows: NonZeroUsize::new(2).unwrap(),
            ..CostModelSettings::default().shape_limits
        },
        ..Default::default()
    }
}

fn shape(contexts: &[u32]) -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: None,
        numeric_features: None,
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [5; 32],
        output_policy_signature: [6; 32],
        graph_state: WaveGraphState::Disabled,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: contexts.to_vec(),
        prefill_chunks: vec![],
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}

fn sample(contexts: &[u32], cost: u64, at: u64) -> WaveCostObservation {
    WaveCostObservation {
        fingerprint: fingerprint(),
        actual_shape: shape(contexts),
        boundary: CostBoundary::PreparationToCommit,
        outcome: WaveObservationOutcome::Completed,
        timing: WaveTiming {
            wall_total_ns: cost,
            device_elapsed_ns: None,
            stages: Default::default(),
        },
        observed_at_ns: at,
    }
}

fn prediction(snapshot: &CostModelSnapshot, contexts: &[u32], at: u64) -> WaveCostPrediction {
    match snapshot.predict(
        &fingerprint(),
        &shape(contexts),
        CostBoundary::PreparationToCommit,
        at,
    ) {
        CostPrediction::Known(value) => value,
        other => panic!("expected known {contexts:?} at {at}, got {other:?}"),
    }
}

fn assert_bounded(trainer: &CostModelTrainer) {
    assert!(trainer.buckets.len() <= trainer.settings.max_buckets.get());
    assert!(trainer.planning_floors.len() <= trainer.buckets.len());
    assert!(trainer
        .planning_floors
        .keys()
        .all(|key| trainer.buckets.contains_key(key)));
    let samples: usize = trainer
        .buckets
        .values()
        .map(|bucket| bucket.samples.len())
        .sum();
    let rows: usize = trainer
        .buckets
        .values()
        .flat_map(|bucket| &bucket.samples)
        .map(|sample| shape_rows(&sample.shape))
        .sum();
    assert_eq!(trainer.retained_samples, samples);
    assert_eq!(trainer.retained_rows, rows);
    assert!(samples <= trainer.settings.max_retained_samples.get());
    assert!(rows <= trainer.settings.max_retained_shape_rows.get());
    assert!(trainer.retired_floors_ns.iter().all(
        |&floor| floor <= trainer.settings.max_wave_ns.get() + trainer.settings.drift_margin_ns
    ));
}

fn retired(trainer: &CostModelTrainer) -> u64 {
    trainer.retired_planning_floor_ns(WaveKind::Decode, CostBoundary::PreparationToCommit)
}

#[test]
fn exact_ttl_boundary_is_retained_then_expiry_releases_bucket_without_refreshing_old_arc() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer.observe(sample(&[100], 100, 1)).unwrap();
    let old = trainer.publish(1).unwrap();
    assert_eq!(prediction(&old, &[100], 11).valid_for_ns, 0);
    assert_eq!(
        trainer.observe(sample(&[200], 20, 11)),
        Err(CostModelError::CapacityExceeded("buckets"))
    );
    assert_eq!(retired(&trainer), 0);
    assert!(Arc::ptr_eq(&old, trainer.published.as_ref().unwrap()));
    assert!(!trainer.dirty);
    trainer.observe(sample(&[200], 20, 12)).unwrap();
    assert_eq!(retired(&trainer), 110);
    let new = trainer.publish(12).unwrap();
    assert_eq!(prediction(&new, &[200], 12).planning_ns, 110);
    assert_eq!(prediction(&old, &[100], 11).planning_ns, 110);
    assert_eq!(
        old.predict(
            &fingerprint(),
            &shape(&[100]),
            CostBoundary::PreparationToCommit,
            12
        ),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
    assert_eq!(
        new.predict(
            &fingerprint(),
            &shape(&[100]),
            CostBoundary::PreparationToCommit,
            12
        ),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    assert!(!Arc::ptr_eq(&old, &new));
    assert_bounded(&trainer);
}

#[test]
fn expired_prefix_reclaims_global_sample_and_row_capacity_in_a_still_live_bucket() {
    let mut config = settings();
    config.max_buckets = NonZeroUsize::new(2).unwrap();
    config.max_retained_samples = NonZeroUsize::new(2).unwrap();
    let mut trainer = CostModelTrainer::new(fingerprint(), config).unwrap();
    trainer.observe(sample(&[100, 101], 100, 1)).unwrap();
    trainer.observe(sample(&[100, 101], 80, 9)).unwrap();
    trainer.publish(9).unwrap();
    trainer.observe(sample(&[200, 201], 20, 12)).unwrap();
    assert_eq!(trainer.retained_sample_count(), 2);
    assert_eq!(trainer.retained_shape_rows(), 4);
    assert_eq!(
        retired(&trainer),
        0,
        "partially live bucket retains its own floor"
    );
    let snapshot = trainer.publish(12).unwrap();
    assert_eq!(prediction(&snapshot, &[100, 101], 12).sample_count, 1);
    assert_eq!(
        prediction(&snapshot, &[100, 101], 12).oldest_sample_at_ns,
        9
    );
    assert_eq!(prediction(&snapshot, &[100, 101], 12).planning_ns, 110);
    assert_bounded(&trainer);
}

#[test]
fn per_bucket_replacement_uses_fresh_count_after_expiry() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    for at in [1, 2, 9] {
        trainer.observe(sample(&[100], 30, at)).unwrap();
    }
    trainer.observe(sample(&[100], 20, 12)).unwrap();
    let timestamps: Vec<_> = trainer
        .buckets
        .values()
        .next()
        .unwrap()
        .samples
        .iter()
        .map(|sample| sample.observed_at_ns)
        .collect();
    assert_eq!(
        timestamps,
        [2, 9, 12],
        "do not evict an extra fresh sample because pre-expiry length was full"
    );
    trainer.observe(sample(&[100], 20, 12)).unwrap();
    let timestamps: Vec<_> = trainer
        .buckets
        .values()
        .next()
        .unwrap()
        .samples
        .iter()
        .map(|sample| sample.observed_at_ns)
        .collect();
    assert_eq!(timestamps, [9, 12, 12]);
    assert_bounded(&trainer);
}

#[test]
fn recreated_and_already_existing_buckets_both_take_their_retired_population_floor() {
    let mut config = settings();
    config.max_buckets = NonZeroUsize::new(2).unwrap();
    let mut trainer = CostModelTrainer::new(fingerprint(), config.clone()).unwrap();
    trainer.observe(sample(&[100], 500, 1)).unwrap();
    trainer.observe(sample(&[200], 20, 8)).unwrap();
    let old = trainer.publish(8).unwrap();
    assert_eq!(prediction(&old, &[200], 8).planning_ns, 30);
    trainer.observe(sample(&[300], 30, 12)).unwrap();
    let current = trainer.publish(12).unwrap();
    assert_eq!(prediction(&current, &[200], 12).planning_ns, 510);
    assert_eq!(
        prediction(&current, &[300], 12).prior_planning_floor_ns,
        510
    );
    assert_eq!(
        prediction(&old, &[200], 12).planning_ns,
        30,
        "published snapshots are immutable"
    );
    trainer.observe(sample(&[100], 1, 23)).unwrap();
    let recreated = trainer.publish(23).unwrap();
    assert_eq!(prediction(&recreated, &[100], 23).planning_ns, 510);
    assert_eq!(retired(&trainer), 510);
    assert_bounded(&trainer);
    let mut new_epoch =
        CostModelTrainer::after_version(fingerprint(), config, recreated.model_version()).unwrap();
    assert_eq!(retired(&new_epoch), 0);
    new_epoch.observe(sample(&[100], 1, 1)).unwrap();
    let recalibrated = new_epoch.publish(1).unwrap();
    assert_eq!(prediction(&recalibrated, &[100], 1).planning_ns, 11);
    assert!(recalibrated.model_version() > recreated.model_version());
}

#[test]
fn repeated_expiry_keeps_floor_metadata_bounded_and_never_decreases_cost() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    let mut floor = 0;
    // Different shapes force real identity retirement; this is storage and
    // monotonicity coverage, not a workload/benchmark repetition gate.
    for (index, cost) in [100, 40, 700, 1, 600].into_iter().enumerate() {
        let at = 1 + index as u64 * 11;
        let context = 100 + index as u32;
        trainer.observe(sample(&[context], cost, at)).unwrap();
        let snapshot = trainer.publish(at).unwrap();
        let plan = prediction(&snapshot, &[context], at).planning_ns;
        assert!(plan >= floor);
        floor = plan;
        assert_bounded(&trainer);
        assert_eq!(trainer.buckets.len(), 1);
        assert_eq!(trainer.planning_floors.len(), 1);
    }
}

#[test]
fn malformed_skipped_and_clock_reversed_observations_do_not_trigger_cleanup() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer.observe(sample(&[100], 100, 1)).unwrap();
    let original = trainer.publish(5).unwrap();
    let mut cases = vec![
        sample(&[0], 20, 20),
        sample(&[200], 0, 20),
        sample(&[200], 20, 4),
    ];
    let mut wrong_identity = sample(&[200], 20, 20);
    wrong_identity.fingerprint.execution_config[0] ^= 1;
    cases.push(wrong_identity);
    let mut skipped = sample(&[200], 20, 20);
    skipped.outcome = WaveObservationOutcome::Deferred;
    cases.push(skipped);
    let mut missing_device = sample(&[200], 20, 20);
    missing_device.boundary = CostBoundary::DeviceOnly;
    cases.push(missing_device);
    for input in cases {
        let before = format!("{trainer:?}");
        assert_ne!(trainer.observe(input), Ok(ObservationDisposition::Recorded));
        assert_eq!(format!("{trainer:?}"), before);
        assert!(Arc::ptr_eq(&original, trainer.published.as_ref().unwrap()));
        assert_bounded(&trainer);
    }
}

#[test]
fn insufficient_capacity_after_prospective_cleanup_is_transactional() {
    let mut config = settings();
    config.max_retained_samples = NonZeroUsize::new(2).unwrap();
    let mut trainer = CostModelTrainer::new(fingerprint(), config).unwrap();
    trainer.observe(sample(&[100], 100, 1)).unwrap();
    trainer.observe(sample(&[100], 90, 9)).unwrap();
    trainer.publish(9).unwrap();
    let before = format!("{trainer:?}");
    assert_eq!(
        trainer.observe(sample(&[200], 20, 12)),
        Err(CostModelError::CapacityExceeded("buckets"))
    );
    assert_eq!(
        format!("{trainer:?}"),
        before,
        "one expired sample must not be removed when the live bucket still blocks a new key"
    );
    assert_bounded(&trainer);
}

#[test]
fn sample_and_row_capacity_still_reject_fresh_full_storage_without_mutation() {
    let mut config = settings();
    config.max_buckets = NonZeroUsize::new(3).unwrap();
    config.max_retained_samples = NonZeroUsize::new(2).unwrap();
    for rows_limited in [false, true] {
        let mut config = config.clone();
        if rows_limited {
            config.max_retained_samples = NonZeroUsize::new(3).unwrap();
        }
        let mut trainer = CostModelTrainer::new(fingerprint(), config).unwrap();
        trainer.observe(sample(&[100, 101], 100, 1)).unwrap();
        trainer.observe(sample(&[200, 201], 100, 2)).unwrap();
        trainer.publish(2).unwrap();
        let before = format!("{trainer:?}");
        let expected = if rows_limited {
            "retained shape rows"
        } else {
            "retained samples"
        };
        assert_eq!(
            trainer.observe(sample(&[300], 20, 3)),
            Err(CostModelError::CapacityExceeded(expected))
        );
        assert_eq!(format!("{trainer:?}"), before);
        assert_bounded(&trainer);
    }
}

#[test]
fn a_retired_floor_supplies_neither_sample_coverage_nor_artificial_freshness() {
    let mut config = settings();
    config.min_samples = NonZeroUsize::new(2).unwrap();
    let mut trainer = CostModelTrainer::new(fingerprint(), config).unwrap();
    trainer.observe(sample(&[100], 100, 1)).unwrap();
    trainer.observe(sample(&[100], 100, 2)).unwrap();
    trainer.publish(2).unwrap();
    trainer.observe(sample(&[200], 10, 13)).unwrap();
    let sparse = trainer.publish(13).unwrap();
    assert_eq!(retired(&trainer), 110);
    assert_eq!(
        sparse.predict(
            &fingerprint(),
            &shape(&[200]),
            CostBoundary::PreparationToCommit,
            13
        ),
        CostPrediction::Unknown(CostUnknownReason::InsufficientSamples)
    );
    assert_eq!(
        sparse.predict(
            &fingerprint(),
            &shape(&[300]),
            CostBoundary::PreparationToCommit,
            13
        ),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    trainer.observe(sample(&[200], 10, 14)).unwrap();
    let known = trainer.publish(14).unwrap();
    assert_eq!(prediction(&known, &[200], 23).valid_for_ns, 0);
    assert_eq!(
        known.predict(
            &fingerprint(),
            &shape(&[200]),
            CostBoundary::PreparationToCommit,
            24
        ),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}

#[test]
fn maximum_valid_planning_floor_survives_reclamation_without_overflow() {
    let mut config = settings();
    config.max_wave_ns = NonZeroU64::new(u64::MAX - config.drift_margin_ns).unwrap();
    let mut trainer = CostModelTrainer::new(fingerprint(), config.clone()).unwrap();
    trainer
        .observe(sample(&[100], config.max_wave_ns.get(), 1))
        .unwrap();
    assert_eq!(
        prediction(&trainer.publish(1).unwrap(), &[100], 1).planning_ns,
        u64::MAX
    );
    trainer.observe(sample(&[200], 1, 12)).unwrap();
    let snapshot = trainer.publish(12).unwrap();
    assert_eq!(retired(&trainer), u64::MAX);
    assert_eq!(prediction(&snapshot, &[200], 12).planning_ns, u64::MAX);
    assert_bounded(&trainer);
}

#[test]
fn long_prefill_retirement_does_not_contaminate_decode_or_device_only_costs() {
    let prefill = |cost, at| WaveCostObservation {
        actual_shape: WaveExecutionShape {
            kind: WaveKind::Prefill,
            decode_kv_tokens: vec![],
            prefill_chunks: vec![PrefillShape {
                offset: 0,
                count: NonZeroU32::new(128).unwrap(),
                total_prompt_tokens: NonZeroU32::new(256).unwrap(),
            }],
            ..shape(&[])
        },
        ..sample(&[], cost, at)
    };
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    let old_prefill = prefill(900, 1);
    let prefill_shape = old_prefill.actual_shape.clone();
    trainer.observe(old_prefill).unwrap();
    trainer.publish(1).unwrap();
    trainer.observe(sample(&[100], 10, 12)).unwrap();
    let decode = trainer.publish(12).unwrap();
    assert_eq!(prediction(&decode, &[100], 12).planning_ns, 20);
    assert_eq!(
        trainer.retired_planning_floor_ns(WaveKind::Prefill, CostBoundary::PreparationToCommit),
        910
    );
    assert_eq!(retired(&trainer), 0);
    let mut device = sample(&[100], 20, 23);
    device.boundary = CostBoundary::DeviceOnly;
    device.timing.device_elapsed_ns = Some(5);
    trainer.observe(device).unwrap();
    let snapshot = trainer.publish(23).unwrap();
    let CostPrediction::Known(device) =
        snapshot.predict(&fingerprint(), &shape(&[100]), CostBoundary::DeviceOnly, 23)
    else {
        panic!("device timing must calibrate independently")
    };
    assert_eq!(device.planning_ns, 15);
    assert_eq!(retired(&trainer), 20);
    assert_eq!(
        trainer.retired_planning_floor_ns(WaveKind::Decode, CostBoundary::DeviceOnly),
        0
    );
    trainer.observe(prefill(1, 34)).unwrap();
    let snapshot = trainer.publish(34).unwrap();
    let CostPrediction::Known(recreated) = snapshot.predict(
        &fingerprint(),
        &prefill_shape,
        CostBoundary::PreparationToCommit,
        34,
    ) else {
        panic!("recreated prefill must retain its own floor")
    };
    assert_eq!(recreated.planning_ns, 910);
    assert_eq!(
        trainer.retired_planning_floor_ns(WaveKind::Decode, CostBoundary::DeviceOnly),
        15
    );
    assert_bounded(&trainer);
}

#[test]
fn a_clock_reversal_cannot_reclaim_samples_even_if_the_proposed_time_would_expire_them() {
    let mut trainer = CostModelTrainer::new(fingerprint(), settings()).unwrap();
    trainer.observe(sample(&[100], 100, 1)).unwrap();
    trainer.publish(1).unwrap();
    let original = trainer.publish(20).unwrap();
    let before = format!("{trainer:?}");
    assert_eq!(
        trainer.observe(sample(&[200], 20, 12)),
        Err(CostModelError::ClockMovedBackwards)
    );
    assert_eq!(format!("{trainer:?}"), before);
    assert!(Arc::ptr_eq(&original, trainer.published.as_ref().unwrap()));
    assert_bounded(&trainer);
}
