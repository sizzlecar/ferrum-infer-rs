use super::*;

#[test]
fn explicit_larger_file_budget_roundtrips_cut_and_final_artifacts_with_original_age() {
    let mut f = Fixture::new();
    f.options.max_file_bytes =
        NonZeroUsize::new(ferrum_types::SloCostProfileImportConfig::MAX_FILE_BYTES).unwrap();
    let mut exporter = f.exporter().unwrap();
    exporter
        .record_numbered(
            1,
            &observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    let cut = exporter
        .write_cut(
            CostProfileCutPaths {
                profile: f.dir.join("cut-profile.json"),
                source: f.dir.join("cut-source.jsonl"),
            },
            1,
            ExportClockReading {
                wall_unix_ns: 1300,
                monotonic_ns: 400,
            },
            stats(),
            &training_stats(),
        )
        .unwrap();
    let receipt = f.finish(exporter).unwrap();
    assert_eq!(cut.retained_samples, 1);
    assert_eq!(cut.accepted_ordinal, 1);
    for (path, source_hash) in [
        (&cut.profile, cut.source_digest),
        (&f.options.path, receipt.source.digest),
    ] {
        let limits = profile::CostProfileLoadLimits {
            max_file_bytes: f.options.max_file_bytes,
            ..Default::default()
        };
        let loaded = profile::load_cost_profile(
            path,
            &fingerprint(),
            &f.settings,
            &limits,
            profile::ProfileLoadClock {
                wall_unix_ns: Some(1400),
                wall_max_error_ns: Some(0),
                monotonic_now_ns: 0,
            },
        )
        .unwrap();
        assert_eq!(loaded.provenance.oldest_imported_age_ns, Some(380));
        let encoded = fs::read(path).unwrap();
        let file: profile_v2::CostProfileFileV2 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(file.samples.len(), 1);
        assert_eq!(file.samples[0].measured_unix_ns, 1020);
        assert_eq!(file.source.observation_artifact_sha256, source_hash);
        let small = profile::CostProfileLoadLimits {
            max_file_bytes: NonZeroUsize::new(encoded.len() - 1).unwrap(),
            ..limits
        };
        assert!(matches!(
            profile::load_cost_profile(
                path,
                &fingerprint(),
                &f.settings,
                &small,
                profile::ProfileLoadClock {
                    wall_unix_ns: Some(1400),
                    wall_max_error_ns: Some(0),
                    monotonic_now_ns: 0,
                },
            ),
            Err(profile::CostProfileError::Limit(_))
        ));
    }
}

#[test]
fn larger_file_budget_does_not_raise_retained_memory_or_entry_limits() {
    let mut f = Fixture::new();
    f.options.max_file_bytes =
        NonZeroUsize::new(ferrum_types::SloCostProfileImportConfig::MAX_FILE_BYTES).unwrap();
    f.options.max_samples = NonZeroUsize::new(131_072).unwrap();
    f.options.max_total_shape_rows = NonZeroUsize::new(1_048_576).unwrap();
    f.options.validate().unwrap();
    assert!(matches!(
        f.exporter(),
        Err(ExportError::Source(
            "retained export allocation exceeds 128 MiB hard bound"
        ))
    ));
    assert!(!f.options.path.exists());
    assert!(!f.options.observations_path.exists());

    f.options.max_samples = NonZeroUsize::new(1).unwrap();
    f.options.max_total_shape_rows = NonZeroUsize::new(1).unwrap();
    let mut exporter = f.exporter().unwrap();
    for at in [120, 121] {
        exporter
            .record(
                &observation(at),
                TrainingDisposition::Recorded,
                PreUpdatePrediction::NoPublishedModel,
            )
            .unwrap();
    }
    f.finish(exporter).unwrap();
    let rows = f.records();
    assert_eq!(rows[2]["counts"]["received_observations"], 2);
    assert_eq!(rows[2]["counts"]["dropped_sample_limit"], 1);
    let file: profile_v2::CostProfileFileV2 =
        serde_json::from_slice(&fs::read(&f.options.path).unwrap()).unwrap();
    assert_eq!(file.samples.len(), 1);
}
