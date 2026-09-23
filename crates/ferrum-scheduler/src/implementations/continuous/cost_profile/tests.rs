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
        drift_margin_ns: 10,
        max_wave_ns: NonZeroU64::new(10_000).unwrap(),
        max_sample_age_ns: NonZeroU64::new(1000).unwrap(),
        ..Default::default()
    }
}

fn clock() -> ProfileLoadClock {
    ProfileLoadClock {
        wall_unix_ns: Some(10_000),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 0,
    }
}

fn shape() -> ProfileWaveShape {
    ProfileWaveShape {
        kind: ProfileWaveKind::Decode,
        path: ProfileExecutionPath::PlanRuntime,
        provider_signature: [5; 32],
        output_policy_signature: [6; 32],
        graph_state: ProfileGraphState::Warm,
        order: ProfileBatchOrder::Ordered,
        decode_kv_tokens: vec![128],
        prefill_chunks: Vec::new(),
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}

fn profile() -> CostProfileFile {
    CostProfileFile {
        schema_version: COST_PROFILE_SCHEMA_VERSION,
        fingerprint: ProfileFingerprint::from(&fingerprint()),
        settings: ProfileModelSettings::from(&settings()),
        generated_unix_ns: 9900,
        source_clock_max_error_ns: Some(0),
        source: ProfileSource {
            generator: "ferrum-calibration".into(),
            generator_revision: "source-revision".into(),
            measurement_protocol: "actual-wave-preparation-through-commit".into(),
            observation_artifact_sha256: [7; 32],
        },
        samples: (0..3)
            .map(|index| ProfileSample {
                source_record: index,
                measured_unix_ns: 9800 + index,
                shape: shape(),
                boundary: ProfileCostBoundary::PreparationToCommit,
                outcome: ProfileObservationOutcome::Completed {},
                timing: ProfileWaveTiming {
                    wall_total_ns: 100 * (index + 1),
                    device_elapsed_ns: None,
                    stages: ProfileStageTimings::default(),
                },
            })
            .collect(),
    }
}

fn bytes(profile: &CostProfileFile) -> Vec<u8> {
    serde_json::to_vec(profile).unwrap()
}

fn load(profile: &CostProfileFile) -> Result<LoadedCostProfile, CostProfileError> {
    load_cost_profile_bytes(
        &bytes(profile),
        &fingerprint(),
        &settings(),
        &CostProfileLoadLimits::default(),
        clock(),
    )
}

fn predict(loaded: &LoadedCostProfile, at: u64) -> CostPrediction {
    loaded.snapshot.predict(
        &fingerprint(),
        &shape().into(),
        CostBoundary::PreparationToCommit,
        at,
    )
}

#[test]
fn new_process_import_preserves_wall_age_and_expires_on_the_monotonic_clock() {
    let mut input = profile();
    input.samples.reverse();
    let loaded = load(&input).unwrap();
    assert_eq!(loaded.snapshot.clock.source_monotonic_anchor_ns, 0);
    assert_eq!(loaded.snapshot.clock.model_anchor_ns, 1000);
    let CostPrediction::Known(prediction) = predict(&loaded, 0) else {
        panic!("fresh profile should predict");
    };
    assert_eq!(prediction.planning_ns, 310);
    assert_eq!(prediction.valid_for_ns, 800);
    assert_eq!(prediction.oldest_sample_at_ns, 800);
    assert_eq!(prediction.newest_sample_at_ns, 802);
    assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(200));
    assert!(matches!(predict(&loaded, 800), CostPrediction::Known(_)));
    let CostPrediction::Known(at_expiry) = predict(&loaded, 800) else {
        unreachable!()
    };
    assert_eq!(at_expiry.valid_for_ns, 0);
    assert_eq!(
        predict(&loaded, 801),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}

#[test]
fn wall_clock_uncertainty_consumes_remaining_freshness_instead_of_extending_it() {
    let mut input = profile();
    input.source_clock_max_error_ns = Some(20);
    let mut now = clock();
    now.wall_max_error_ns = Some(30);
    let loaded = load_cost_profile_bytes(
        &bytes(&input),
        &fingerprint(),
        &settings(),
        &CostProfileLoadLimits::default(),
        now,
    )
    .unwrap();
    assert_eq!(loaded.provenance.conservative_clock_error_ns, 50);
    assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(250));
    assert!(matches!(predict(&loaded, 750), CostPrediction::Known(_)));
    assert_eq!(
        predict(&loaded, 751),
        CostPrediction::Unknown(CostUnknownReason::StaleSamples)
    );
}

#[test]
fn newly_generated_files_do_not_refresh_old_measurements() {
    let mut input = profile();
    input.generated_unix_ns = 10_000;
    for sample in &mut input.samples {
        sample.measured_unix_ns = 1000;
    }
    let loaded = load(&input).unwrap();
    assert_eq!(loaded.provenance.counts.stale_samples, 3);
    assert_eq!(loaded.provenance.counts.recorded_samples, 0);
    assert_eq!(loaded.provenance.oldest_imported_age_ns, None);
    assert_eq!(
        predict(&loaded, 0),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    assert_eq!(loaded.snapshot.bucket_count(), 0);
    let limits = CostProfileLoadLimits {
        max_profile_age_ns: NonZeroU64::new(99).unwrap(),
        ..Default::default()
    };
    assert!(matches!(
        load_cost_profile_bytes(
            &bytes(&profile()),
            &fingerprint(),
            &settings(),
            &limits,
            clock()
        ),
        Err(CostProfileError::Clock(_))
    ));
}

#[test]
fn untrusted_future_or_inconsistent_wall_clocks_are_rejected() {
    let mut input = profile();
    input.source_clock_max_error_ns = None;
    assert!(matches!(load(&input), Err(CostProfileError::Clock(_))));
    input = profile();
    input.generated_unix_ns = 10_001;
    assert!(matches!(load(&input), Err(CostProfileError::Clock(_))));
    input = profile();
    input.samples[0].measured_unix_ns = 9901;
    assert!(matches!(load(&input), Err(CostProfileError::Clock(_))));
    input = profile();
    input.samples[0].measured_unix_ns = 0;
    assert!(matches!(load(&input), Err(CostProfileError::Clock(_))));
    for now in [
        ProfileLoadClock {
            wall_unix_ns: None,
            ..clock()
        },
        ProfileLoadClock {
            wall_max_error_ns: None,
            ..clock()
        },
    ] {
        assert!(matches!(
            load_cost_profile_bytes(
                &bytes(&profile()),
                &fingerprint(),
                &settings(),
                &CostProfileLoadLimits::default(),
                now
            ),
            Err(CostProfileError::Clock(_))
        ));
    }
    input = profile();
    input.source_clock_max_error_ns = Some(1_000_000_001);
    assert!(matches!(load(&input), Err(CostProfileError::Clock(_))));
}

#[test]
fn all_four_fingerprints_are_required_and_execution_shapes_do_not_cross_predict() {
    for component in 0..4 {
        let mut input = profile();
        match component {
            0 => input.fingerprint.model_weights[0] ^= 1,
            1 => input.fingerprint.numerical_policy[0] ^= 1,
            2 => input.fingerprint.device_runtime[0] ^= 1,
            _ => input.fingerprint.execution_config[0] ^= 1,
        }
        assert!(matches!(
            load(&input),
            Err(CostProfileError::FingerprintMismatch)
        ));
    }
    let loaded = load(&profile()).unwrap();
    let mut variants = vec![shape(); 5];
    variants[0].path = ProfileExecutionPath::UnsupportedFallback;
    variants[1].provider_signature[0] ^= 1;
    variants[2].output_policy_signature[0] ^= 1;
    variants[3].graph_state = ProfileGraphState::Cold;
    variants[4].decode_kv_tokens[0] = 129;
    for variant in variants {
        assert_eq!(
            loaded.snapshot.predict(
                &fingerprint(),
                &variant.into(),
                CostBoundary::PreparationToCommit,
                0
            ),
            CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
        );
    }
    let mut numerical = fingerprint();
    numerical.numerical_policy[0] ^= 1;
    assert_eq!(
        loaded.snapshot.predict(
            &numerical,
            &shape().into(),
            CostBoundary::PreparationToCommit,
            0
        ),
        CostPrediction::Unknown(CostUnknownReason::FingerprintMismatch)
    );
}

#[test]
fn schema_is_versioned_strict_and_requires_explicit_settings() {
    let mut input = profile();
    input.schema_version = 99;
    assert!(matches!(
        load(&input),
        Err(CostProfileError::UnsupportedVersion(99))
    ));
    // V2 is supported through its own strict envelope. Relabeling a v1 body
    // cannot invent its explicit model selection or numeric evidence.
    input.schema_version = v2::COST_PROFILE_SCHEMA_VERSION_V2;
    assert!(matches!(load(&input), Err(CostProfileError::Json(_))));
    for pointer in [
        "",
        "/settings",
        "/samples/0/shape",
        "/samples/0/timing/stages",
        "/samples/0/outcome",
    ] {
        let mut value = serde_json::to_value(profile()).unwrap();
        value
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unknown_field".into(), serde_json::json!(1));
        assert!(
            serde_json::from_value::<CostProfileFile>(value).is_err(),
            "{pointer}"
        );
    }
    let mut value = serde_json::to_value(profile()).unwrap();
    value["settings"]
        .as_object_mut()
        .unwrap()
        .remove("min_samples");
    assert!(serde_json::from_value::<CostProfileFile>(value).is_err());
    let encoded = bytes(&profile());
    assert_eq!(
        serde_json::from_slice::<CostProfileFile>(&encoded).unwrap(),
        profile()
    );
    for outcome in [
        ProfileObservationOutcome::Completed {},
        ProfileObservationOutcome::NotSubmitted {},
        ProfileObservationOutcome::Deferred {},
        ProfileObservationOutcome::FailedAfterSubmit {},
        ProfileObservationOutcome::PartiallyCompleted {
            timing_covers_actual_shape_only: true,
        },
    ] {
        let mut encoded = serde_json::to_value(outcome).unwrap();
        encoded.as_object_mut().unwrap().insert(
            "unexpected_metadata".into(),
            serde_json::json!("must reject"),
        );
        assert!(serde_json::from_value::<ProfileObservationOutcome>(encoded).is_err());
    }
}

#[test]
fn profile_cannot_override_runtime_settings_or_bypass_their_validation() {
    let mut input = profile();
    input.settings.min_samples = NonZeroUsize::new(1).unwrap();
    assert!(matches!(
        load(&input),
        Err(CostProfileError::SettingsMismatch)
    ));
    input = profile();
    input.settings.residual_quantile = 2.0;
    assert!(matches!(
        load(&input),
        Err(CostProfileError::Model(CostModelError::InvalidSettings(_)))
    ));
    let mut value = serde_json::to_value(profile()).unwrap();
    value["settings"]["max_sample_age_ns"] = serde_json::json!(0);
    assert!(serde_json::from_value::<CostProfileFile>(value).is_err());
}

#[test]
fn file_sample_and_shape_limits_are_independent_and_hard_bounds_apply_during_parse() {
    let encoded = bytes(&profile());
    let limits = [
        CostProfileLoadLimits {
            max_file_bytes: NonZeroUsize::new(encoded.len() - 1).unwrap(),
            ..Default::default()
        },
        CostProfileLoadLimits {
            max_samples: NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        },
        CostProfileLoadLimits {
            max_total_shape_rows: NonZeroUsize::new(2).unwrap(),
            ..Default::default()
        },
        CostProfileLoadLimits {
            max_file_bytes: NonZeroUsize::new(HARD_FILE_BYTES + 1).unwrap(),
            ..Default::default()
        },
    ];
    for limits in limits {
        assert!(matches!(
            load_cost_profile_bytes(&encoded, &fingerprint(), &settings(), &limits, clock()),
            Err(CostProfileError::Limit(_))
        ));
    }
    let mut input = profile();
    input.samples[0].shape.decode_kv_tokens = vec![1; settings().shape_limits.max_rows.get() + 1];
    assert!(matches!(load(&input), Err(CostProfileError::Limit(_))));
    input.samples[0].shape.decode_kv_tokens = vec![1; HARD_ROWS_PER_VECTOR + 1];
    assert!(matches!(load(&input), Err(CostProfileError::Json(_))));
}

#[test]
fn imported_completed_samples_pass_the_trainers_timing_and_actual_shape_checks() {
    let mut input = profile();
    input.samples[0].timing.wall_total_ns = 0;
    assert!(matches!(
        load(&input),
        Err(CostProfileError::Sample {
            reason: CostModelError::InvalidTiming(_),
            ..
        })
    ));
    input = profile();
    input.samples[0].timing.stages.prepare = Some(ProfileMeasuredSpan {
        start_ns: 0,
        end_ns: 90,
    });
    input.samples[0].timing.stages.commit = Some(ProfileMeasuredSpan {
        start_ns: 80,
        end_ns: 100,
    });
    assert!(matches!(
        load(&input),
        Err(CostProfileError::Sample {
            reason: CostModelError::InvalidTiming(_),
            ..
        })
    ));
    input = profile();
    input.samples[0].shape.decode_kv_tokens = vec![0];
    assert!(matches!(
        load(&input),
        Err(CostProfileError::Sample {
            reason: CostModelError::InvalidShape(_),
            ..
        })
    ));
}

#[test]
fn failures_deferred_and_unisolated_partial_observations_are_audited_without_training_zero_cost() {
    let mut input = profile();
    for (index, outcome) in [
        ProfileObservationOutcome::NotSubmitted {},
        ProfileObservationOutcome::Deferred {},
        ProfileObservationOutcome::FailedAfterSubmit {},
        ProfileObservationOutcome::PartiallyCompleted {
            timing_covers_actual_shape_only: false,
        },
    ]
    .into_iter()
    .enumerate()
    {
        let mut sample = input.samples[0].clone();
        sample.source_record = index as u64 + 10;
        sample.outcome = outcome;
        sample.timing.wall_total_ns = 0;
        input.samples.push(sample);
    }
    let loaded = load(&input).unwrap();
    assert_eq!(loaded.provenance.counts.offered_samples, 7);
    assert_eq!(loaded.provenance.counts.recorded_samples, 3);
    for reason in [
        ProfileSkippedObservation::NotSubmitted,
        ProfileSkippedObservation::Deferred,
        ProfileSkippedObservation::FailedAfterSubmit,
        ProfileSkippedObservation::UnisolatedPartial,
    ] {
        assert_eq!(loaded.provenance.counts.skipped_samples[&reason], 1);
    }
    let CostPrediction::Known(value) = predict(&loaded, 0) else {
        panic!("known");
    };
    assert_eq!(value.planning_ns, 310);
    input.samples.drain(..3);
    let failed_only = load(&input).unwrap();
    assert_eq!(
        predict(&failed_only, 0),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
}

#[test]
fn device_diagnostics_never_stand_in_for_full_preparation_to_commit_cost() {
    let mut input = profile();
    for sample in &mut input.samples {
        sample.boundary = ProfileCostBoundary::DeviceOnly;
        sample.timing.device_elapsed_ns = Some(50);
    }
    let loaded = load(&input).unwrap();
    assert_eq!(
        predict(&loaded, 0),
        CostPrediction::Unknown(CostUnknownReason::UnobservedBucket)
    );
    let CostPrediction::Known(value) =
        loaded
            .snapshot
            .predict(&fingerprint(), &shape().into(), CostBoundary::DeviceOnly, 0)
    else {
        panic!("known device measurement");
    };
    assert_eq!(value.planning_ns, 60);
    for sample in &mut input.samples {
        sample.timing.device_elapsed_ns = None;
    }
    let missing = load(&input).unwrap();
    assert_eq!(
        missing.provenance.counts.skipped_samples[&ProfileSkippedObservation::MissingDeviceTiming],
        3
    );
}

#[test]
fn isolated_partial_shape_training_is_explicit() {
    let mut input = profile();
    for sample in &mut input.samples {
        sample.outcome = ProfileObservationOutcome::PartiallyCompleted {
            timing_covers_actual_shape_only: true,
        };
    }
    assert!(matches!(
        predict(&load(&input).unwrap(), 0),
        CostPrediction::Known(_)
    ));
}

#[test]
fn live_observation_and_publication_continue_the_same_epoch_and_old_snapshots_stay_immutable() {
    let mut now = clock();
    now.monotonic_now_ns = 400;
    let mut loaded = load_cost_profile_bytes(
        &bytes(&profile()),
        &fingerprint(),
        &settings(),
        &CostProfileLoadLimits::default(),
        now,
    )
    .unwrap();
    let old = loaded.snapshot.clone();
    assert_eq!(old.clock.model_now_ns(410).unwrap(), 1010);
    assert_eq!(
        old.predict(
            &fingerprint(),
            &shape().into(),
            CostBoundary::PreparationToCommit,
            399
        ),
        CostPrediction::Unknown(CostUnknownReason::ClockMovedBackwards)
    );
    loaded
        .observe_live(
            WaveCostObservation {
                fingerprint: fingerprint(),
                actual_shape: shape().into(),
                boundary: CostBoundary::PreparationToCommit,
                outcome: WaveObservationOutcome::Completed,
                timing: WaveTiming {
                    wall_total_ns: 1000,
                    device_elapsed_ns: None,
                    stages: WaveStageTimings::default(),
                },
                observed_at_ns: 0,
            },
            410,
        )
        .unwrap();
    let new = loaded.publish(410).unwrap();
    assert_eq!(new.model_version(), old.model_version() + 1);
    let CostPrediction::Known(old_value) = old.predict(
        &fingerprint(),
        &shape().into(),
        CostBoundary::PreparationToCommit,
        410,
    ) else {
        panic!("old");
    };
    let CostPrediction::Known(new_value) = new.predict(
        &fingerprint(),
        &shape().into(),
        CostBoundary::PreparationToCommit,
        410,
    ) else {
        panic!("new");
    };
    assert_eq!(old_value.planning_ns, 310);
    assert_eq!(new_value.planning_ns, 1010);
    assert!(new.clock.model_now_ns(u64::MAX).is_err());
}

#[test]
fn exact_file_digest_and_bounded_source_identity_are_retained() {
    let encoded = bytes(&profile());
    let loaded = load(&profile()).unwrap();
    assert_eq!(
        loaded.provenance.file_sha256,
        format!("sha256:{:x}", Sha256::digest(&encoded))
    );
    assert_eq!(loaded.provenance.file_bytes, encoded.len());
    assert_eq!(loaded.provenance.source, profile().source);
    let mut whitespace = encoded;
    whitespace.push(b' ');
    let alternate = load_cost_profile_bytes(
        &whitespace,
        &fingerprint(),
        &settings(),
        &CostProfileLoadLimits::default(),
        clock(),
    )
    .unwrap();
    assert_ne!(
        alternate.provenance.file_sha256,
        loaded.provenance.file_sha256
    );
    let mut input = profile();
    input.samples[1].source_record = input.samples[0].source_record;
    assert!(matches!(load(&input), Err(CostProfileError::Metadata(_))));
    input = profile();
    input.source.generator.clear();
    assert!(matches!(load(&input), Err(CostProfileError::Metadata(_))));
}

#[test]
fn file_loader_caps_reads_and_retains_canonical_location_as_provenance_only() {
    struct TempFile(PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let temp = TempFile(
        std::env::temp_dir().join(format!("ferrum-cost-profile-{}.json", uuid::Uuid::new_v4())),
    );
    let encoded = bytes(&profile());
    std::fs::write(&temp.0, &encoded).unwrap();
    let expanded_limits = CostProfileLoadLimits {
        max_file_bytes: NonZeroUsize::new(128 * 1024 * 1024).unwrap(),
        ..Default::default()
    };
    let loaded = load_cost_profile(
        &temp.0,
        &fingerprint(),
        &settings(),
        &expanded_limits,
        clock(),
    )
    .unwrap();
    assert_eq!(
        loaded.provenance.loaded_from,
        Some(temp.0.canonicalize().unwrap())
    );
    assert!(matches!(predict(&loaded, 0), CostPrediction::Known(_)));
    assert_eq!(
        loaded.provenance.file_sha256,
        format!("sha256:{:x}", Sha256::digest(&encoded))
    );
    assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(200));
    let exact_limit = CostProfileLoadLimits {
        max_file_bytes: NonZeroUsize::new(encoded.len()).unwrap(),
        ..expanded_limits.clone()
    };
    assert!(
        load_cost_profile(&temp.0, &fingerprint(), &settings(), &exact_limit, clock(),).is_ok()
    );
    let limits = CostProfileLoadLimits {
        max_file_bytes: NonZeroUsize::new(encoded.len() - 1).unwrap(),
        ..Default::default()
    };
    assert!(matches!(
        load_cost_profile(&temp.0, &fingerprint(), &settings(), &limits, clock()),
        Err(CostProfileError::Limit(_))
    ));
    // Sparse metadata exercises the actual pre-allocation rejection: no large
    // fixture, payload allocation or source parse is necessary.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&temp.0)
        .unwrap();
    file.set_len(expanded_limits.max_file_bytes.get() as u64 + 1)
        .unwrap();
    assert!(matches!(
        load_cost_profile(
            &temp.0,
            &fingerprint(),
            &settings(),
            &expanded_limits,
            clock()
        ),
        Err(CostProfileError::Limit("file byte limit exceeded"))
    ));
    let largest_limits = CostProfileLoadLimits {
        max_file_bytes: NonZeroUsize::new(HARD_FILE_BYTES).unwrap(),
        ..expanded_limits
    };
    file.set_len(HARD_FILE_BYTES as u64 + 1).unwrap();
    assert!(matches!(
        load_cost_profile(
            &temp.0,
            &fingerprint(),
            &settings(),
            &largest_limits,
            clock()
        ),
        Err(CostProfileError::Limit("file byte limit exceeded"))
    ));
}
