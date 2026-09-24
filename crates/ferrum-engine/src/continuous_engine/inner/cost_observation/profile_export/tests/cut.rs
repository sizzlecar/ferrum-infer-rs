use super::*;

fn paths(f: &Fixture) -> CostProfileCutPaths {
    CostProfileCutPaths {
        profile: f.dir.join("training-profile.json"),
        source: f.dir.join("training-source.jsonl"),
    }
}

#[tokio::test]
async fn calibration_profile_cut_excludes_heldout_and_retains_rejected_ordinal() {
    let f = Fixture::new();
    let mut config = SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::MIN;
    config.profile_export = Some(f.options.clone());
    let runtime = runtime_with_fixed_clock(&config, false);
    runtime.sink.offer(observation(100)).unwrap();
    let mut invalid = observation(100);
    invalid.timing.wall_total_ns = 0;
    runtime.sink.offer(invalid).unwrap();
    let waiter = runtime.request_profile_cut(paths(&f)).unwrap();
    let mut heldout = observation(100);
    heldout.timing.wall_total_ns = 90;
    runtime.sink.offer(heldout).unwrap();
    runtime.consume_samples();
    let frozen = waiter.wait().await.unwrap();
    let receipt = frozen.profile_cut.unwrap().unwrap();
    assert_eq!(receipt.accepted_ordinal, 2);
    assert_eq!(receipt.retained_samples, 1);
    assert_eq!(receipt.raw_retained_observations, 2);
    assert_eq!(runtime.sink.stats().drained, 2);
    let source = fs::read(&receipt.source).unwrap();
    assert_eq!(
        receipt.source_digest,
        <[u8; 32]>::from(Sha256::digest(&source))
    );
    let raw: Vec<serde_json::Value> = std::str::from_utf8(&source)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(raw[1]["accepted_ordinal"], 1);
    assert_eq!(raw[2]["accepted_ordinal"], 2);
    assert_eq!(raw[2]["training"]["reason"], "invalid_timing");
    let cut_bytes = fs::read(&receipt.profile).unwrap();
    let cut: profile_v2::CostProfileFileV2 = serde_json::from_slice(&cut_bytes).unwrap();
    assert_eq!(cut.samples.len(), 1);
    assert_eq!(cut.samples[0].source_record, 0);
    assert_eq!(cut.samples[0].timing.wall_total_ns, 20);
    assert_eq!(
        cut.source.observation_artifact_sha256,
        receipt.source_digest
    );
    assert!(
        !f.options.path.exists(),
        "cut must not finalize live export"
    );
    runtime.shutdown().await.unwrap();
    let live: profile_v2::CostProfileFileV2 =
        serde_json::from_slice(&fs::read(&f.options.path).unwrap()).unwrap();
    assert_eq!(live.samples.len(), 2);
    assert_eq!(live.samples[1].timing.wall_total_ns, 90);
    assert_eq!(fs::read(&receipt.profile).unwrap(), cut_bytes);
    assert_eq!(fs::read(&receipt.source).unwrap(), source);
}

#[test]
fn calibration_profile_cut_roundtrips_product_loader_without_renewing_age() {
    let f = Fixture::new();
    let mut exporter = f.exporter().unwrap();
    exporter
        .record_numbered(
            1,
            &observation(120),
            TrainingDisposition::Recorded,
            PreUpdatePrediction::NoPublishedModel,
        )
        .unwrap();
    let receipt = exporter
        .write_cut(
            paths(&f),
            1,
            ExportClockReading {
                wall_unix_ns: 1300,
                monotonic_ns: 400,
            },
            stats(),
            &training_stats(),
        )
        .unwrap();
    let file: profile_v2::CostProfileFileV2 =
        serde_json::from_slice(&fs::read(&receipt.profile).unwrap()).unwrap();
    assert_eq!(file.samples[0].measured_unix_ns, 1020);
    let imported = profile::load_cost_profile(
        &receipt.profile,
        &fingerprint(),
        &f.settings,
        &Default::default(),
        profile::ProfileLoadClock {
            wall_unix_ns: Some(1400),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 0,
        },
    )
    .unwrap();
    assert_eq!(imported.provenance.oldest_imported_age_ns, Some(380));
    let model::CostPrediction::Known(prediction) = imported.snapshot.predict(
        &fingerprint(),
        &observation(0).actual_shape,
        model::CostBoundary::PreparationToCommit,
        0,
    ) else {
        panic!("valid cut must load through the product loader");
    };
    assert_eq!(prediction.valid_for_ns, 620);
    assert!(matches!(
        imported.snapshot.predict(
            &fingerprint(),
            &observation(0).actual_shape,
            model::CostBoundary::PreparationToCommit,
            621
        ),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
    f.finish(exporter).unwrap();
}

#[tokio::test]
async fn calibration_profile_cut_failure_keeps_live_worker_and_files_intact() {
    let f = Fixture::new();
    let mut config = SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::MIN;
    config.profile_export = Some(f.options.clone());
    let runtime = runtime_with_fixed_clock(&config, false);
    runtime.sink.offer(observation(100)).unwrap();
    let paths = paths(&f);
    fs::write(&paths.profile, b"existing evidence").unwrap();
    let waiter = runtime.request_profile_cut(paths.clone()).unwrap();
    runtime.consume_samples();
    let frozen = waiter.wait().await.unwrap();
    assert!(frozen.profile_cut.unwrap().is_err());
    assert_eq!(fs::read(&paths.profile).unwrap(), b"existing evidence");
    assert!(!paths.source.exists());
    runtime.sink.offer(observation(100)).unwrap();
    runtime.shutdown().await.unwrap();
    assert_eq!(runtime.trained_samples(), 2);
    assert!(f.options.path.exists());
    assert_eq!(f.records().len(), 4);
}

#[tokio::test]
async fn calibration_profile_cut_dropped_waiter_still_publishes_once() {
    let f = Fixture::new();
    let mut config = SloCostObservationConfig::default();
    config.model.min_samples = NonZeroUsize::MIN;
    config.profile_export = Some(f.options.clone());
    let runtime = runtime_with_fixed_clock(&config, false);
    runtime.sink.offer(observation(100)).unwrap();
    drop(runtime.request_profile_cut(paths(&f)).unwrap());
    assert!(runtime.request_checkpoint().is_err());
    runtime.consume_samples();
    let source = fs::read(paths(&f).source).unwrap();
    assert!(paths(&f).profile.exists());
    let waiter = runtime.request_checkpoint().unwrap();
    runtime.consume_samples();
    assert_eq!(waiter.wait().await.unwrap().accepted_ordinal, 1);
    assert_eq!(fs::read(paths(&f).source).unwrap(), source);
    runtime.shutdown().await.unwrap();
}
