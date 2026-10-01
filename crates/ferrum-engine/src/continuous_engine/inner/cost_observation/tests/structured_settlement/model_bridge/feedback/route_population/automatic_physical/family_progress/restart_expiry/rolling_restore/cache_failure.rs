use super::*;

#[tokio::test]
async fn automatic_reuse_codec_capacity_failure_does_not_poison_live_qualification() {
    let directory = CacheDirectory(std::env::temp_dir().join(format!(
        "ferrum-cache-failure-live-{}",
        uuid::Uuid::new_v4()
    )));
    let identity = identity();
    let ExecutorCostIdentityAvailability::Known(executor) = &identity else {
        unreachable!()
    };
    let domain = CostWorkloadDomainV1::new_vnext(
        executor,
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(2).unwrap(),
            maximum_context_tokens: NonZeroU32::new(128).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(2).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(32).unwrap(),
            repetition_slot_capacity: 0,
            fixed_state_bytes_per_row: 64,
        },
    )
    .unwrap();
    let local = Arc::new(VirtualClock(AtomicU64::new(1)));
    let clock = Arc::new(BootClock {
        local: local.clone(),
        domain: CostMonotonicDomainV1::new_macos_continuous([85; 16]).unwrap(),
    });
    let mut settings = SloAutomaticCalibrationSettingsV1 {
        population_schedule:
            ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::OwnerBlocksRollingV2,
        maximum_retained_generations: NonZeroUsize::new(2).unwrap(),
        discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
        phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
        ..Default::default()
    };
    let SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 { location, limits } =
        &mut settings.reuse
    else {
        unreachable!()
    };
    *location = SloAutomaticCalibrationCacheLocationV1::Directory {
        path: directory.0.clone(),
    };
    // Enough for owner/dirty/lease metadata so the real optional cache opens;
    // smaller than this original source header, which fails only its sink.
    limits.maximum_total_bytes = NonZeroU64::new(256).unwrap();
    let mut config = SloCostObservationConfig::structured_whole_wave_v2();
    config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 { settings };
    let runtime = EngineCostRuntime::build_with_manual_worker_and_reuse(
        identity,
        clock,
        &config,
        domain.clone(),
    )
    .unwrap();
    assert!(
        runtime.audit_snapshot().automatic_reuse.is_some(),
        "cache really opened before the source failed"
    );
    runtime.begin_automatic_calibration().unwrap();
    runtime.consume_samples();
    let mut f = Families {
        runtime,
        clock: local,
        domain,
        recorded: 0,
    };
    for _ in 0..4 {
        complete_block(&mut f, A);
    }
    assert!(f.known(A), "{:#?}", f.live().audit());
    let audit = f.live().audit();
    assert!(audit.qualified_publications > 0, "{audit:#?}");
    assert_eq!(audit.failed_generations, 0, "{audit:#?}");
    assert!(audit.publication_error.is_none(), "{audit:#?}");
    f.runtime.shutdown().await.unwrap();
    f.runtime.acknowledge_restart_shutdown();
    let cache = f.runtime.audit_snapshot().automatic_reuse.unwrap();
    assert!(!cache.clean_shutdown);
    assert_eq!(
        serde_json::to_value(cache.first_failure).unwrap()["reason"],
        "capacity"
    );
    assert!(directory.0.join("cache-v1/dirty").is_file());
    assert!(!directory.0.join("cache-v1/manifest.json").exists());
}
