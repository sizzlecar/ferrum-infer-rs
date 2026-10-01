//! Private ordinary-call tickets -> original FIFO -> worker fit/publication.
//! CPU clock fixtures test lifecycle contracts, not backend speed or accuracy.
use super::*;

struct LiveFixture {
    directory: PathBuf,
    evidence: PathBuf,
    clock: Arc<VirtualClock>,
    config: SloCostObservationConfig,
    query: StructuredQueryV2,
}
impl LiveFixture {
    fn new() -> Self {
        #[derive(serde::Serialize)]
        struct Declaration {
            schema_version: u32,
            phase_offered_waves: [usize; 3],
            maximum_window_ns: u64,
            settings: StructuredSettingsV2,
            scopes: Vec<StructuredScopeV2>,
        }
        let directory =
            std::env::temp_dir().join(format!("ferrum-live-worker-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&directory).unwrap();
        let w = wave("fixture.feedback.a");
        let input = StructuredInputV2::from_actual(
            &w.prepared.exact,
            &w.prepared.selected,
            &w.prepared.recipe,
        )
        .unwrap();
        let declaration = Declaration {
            schema_version: 1,
            phase_offered_waves: [8; 3],
            maximum_window_ns: 10_000_000_000,
            settings: StructuredSettingsV2 {
                static_margin_ns: 1,
                ..Default::default()
            },
            scopes: vec![StructuredScopeV2 {
                numerical_family: None,
                owner: input.owner().clone(),
                coverage: StructuredCoverageV2 {
                    pending_eligible_positions: vec![],
                    authorized_pending_constraints: vec![HostPendingConstraintV2::AnySubset],
                    pending_counts: vec![0],
                    length_counts: vec![1],
                    pending_positions: vec![],
                    length_positions: vec![0],
                    joint_counts: vec![(0, 1)],
                },
            }],
        };
        let path = directory.join("declaration.json");
        fs::write(&path, serde_json::to_vec(&declaration).unwrap()).unwrap();
        let evidence = directory.join("evidence");
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        // Same-process publication needs no operator clock-accuracy declaration.
        assert_eq!(
            config.profile_import.declared_local_clock_max_error_ns,
            None
        );
        config.live_structured_calibration =
            ferrum_types::SloLiveStructuredCalibration::ServiceWindowsV1 {
                declaration: path,
                evidence_directory: evidence.clone(),
                maximum_generations: NonZeroUsize::new(2).unwrap(),
                maximum_source_bytes: NonZeroU64::new(8 * 1024 * 1024).unwrap(),
                maximum_retained_numeric_bytes: NonZeroUsize::new(16 * 1024 * 1024).unwrap(),
            };
        Self {
            directory,
            evidence,
            clock: Arc::new(VirtualClock(AtomicU64::new(1))),
            config,
            query: StructuredQueryV2::exact(input),
        }
    }
    fn build(&self) -> EngineCostRuntime {
        EngineCostRuntime::build(identity(), self.clock.clone(), &self.config, false).unwrap()
    }
    fn generation(&self, runtime: &EngineCostRuntime, generation: u64) {
        let error = self.generation_with_hook(runtime, generation, || {});
        assert!(error.is_none(), "{error:?}");
    }
    fn generation_with_hook(
        &self,
        runtime: &EngineCostRuntime,
        generation: u64,
        mut before_close: impl FnMut(),
    ) -> Option<String> {
        runtime.consume_samples();
        for p in 0..3 {
            for n in 0..8 {
                let mut w = wave("fixture.feedback.a");
                // Independent complete owners, with identical declared work.
                w.actual.rows[0].owner_incarnation = generation * 100 + p * 8 + n + 1;
                record_with_hooks(
                    &runtime.ids,
                    &runtime.sink,
                    &self.clock,
                    w.actual,
                    w.host,
                    None,
                    0,
                    None,
                    |at| {
                        Some(
                            runtime
                                .reserve_live_ticket(Some(at))
                                .expect("declared offer"),
                        )
                    },
                    |_| {},
                );
                if p == 2 && n == 7 {
                    before_close();
                }
                runtime.consume_samples();
            }
            let audit = runtime.training.live.as_ref().unwrap().audit();
            assert!(audit.failure.is_none(), "{audit:?}");
            if p < 2 {
                assert!(audit.publication_error.is_none(), "{audit:?}");
                assert_eq!(audit.population.phase, (p + 1) as usize);
            }
        }
        runtime
            .training
            .live
            .as_ref()
            .unwrap()
            .audit()
            .publication_error
    }
}
impl Drop for LiveFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[tokio::test]
async fn live_service_worker_qualifies_without_initial_profile_and_replaces_epoch() {
    let f = LiveFixture::new();
    let runtime = f.build();
    assert!(runtime.snapshot().is_none());
    f.generation(&runtime, 1);
    let first = runtime
        .snapshot()
        .expect("original worker published first live catalog");
    let now = f.clock.now_ns().unwrap();
    first.audit_structured_query_v2(&f.query, now).unwrap();
    let receipt = runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(receipt.schema_version, 13);
    assert_eq!(
        receipt.clock_basis,
        ferrum_types::SloCostProfileClockBasis::SameProcessMonotonic
    );
    assert_eq!(receipt.declared_local_clock_max_error_ns, None);
    assert_eq!(receipt.conservative_clock_error_ns, 0);
    let child = &receipt.structured_whole_wave_v2.as_ref().unwrap().children[0];
    assert_eq!(child.clock_basis, receipt.clock_basis);
    assert_eq!(child.source_monotonic_anchor_ns, child.model_anchor_ns);
    assert_eq!(
        runtime
            .training
            .live
            .as_ref()
            .unwrap()
            .audit()
            .qualified_publications,
        1
    );
    f.generation(&runtime, 2);
    let second = runtime.snapshot().unwrap();
    assert!(second.model_version() > first.model_version());
    assert!(!first.current());
    assert_eq!(
        runtime.try_model_version_current(first.model_version()),
        Some(false)
    );
    second
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(
        runtime
            .training
            .live
            .as_ref()
            .unwrap()
            .audit()
            .qualified_publications,
        2
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn live_service_worker_write_failure_preserves_the_previous_qualified_snapshot() {
    let f = LiveFixture::new();
    let runtime = f.build();
    f.generation(&runtime, 1);
    let old = runtime.snapshot().unwrap();
    fs::rename(&f.evidence, f.directory.join("preserved-evidence")).unwrap();
    fs::write(
        &f.evidence,
        b"block new generation directory without losing earlier evidence",
    )
    .unwrap();
    runtime.consume_samples();
    let audit = runtime.training.live.as_ref().unwrap().audit();
    assert!(audit.publication_error.is_some());
    assert_eq!(audit.qualified_publications, 1);
    assert!(Arc::ptr_eq(&old, &runtime.snapshot().unwrap()));
    old.audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn live_service_profile_write_failure_after_source_seal_preserves_old_predictor() {
    let f = LiveFixture::new();
    let runtime = f.build();
    f.generation(&runtime, 1);
    let old = runtime.snapshot().unwrap();
    let old_receipt = runtime.training.published_catalog_receipt().unwrap();
    let mut occupied = None;
    let error = f.generation_with_hook(&runtime, 2, || {
        let destination = runtime
            .training
            .live
            .as_ref()
            .unwrap()
            .pending_profile_path()
            .unwrap();
        fs::write(&destination, b"existing file must not be replaced").unwrap();
        occupied = Some(destination);
    });
    assert!(error.is_some());
    assert!(old.current());
    assert!(Arc::ptr_eq(&old, &runtime.snapshot().unwrap()));
    assert_eq!(
        runtime.training.published_catalog_receipt().unwrap(),
        old_receipt
    );
    assert_eq!(
        fs::read(occupied.unwrap()).unwrap(),
        b"existing file must not be replaced"
    );
    let source_count = fs::read_dir(&f.evidence)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jsonl"))
        .count();
    assert_eq!(
        source_count, 2,
        "both original sealed generations remain inspectable"
    );
    old.audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn live_service_background_worker_advances_and_publishes_from_producer_notifications() {
    let f = LiveFixture::new();
    let runtime = EngineCostRuntime::build(identity(), f.clock.clone(), &f.config, true).unwrap();
    for p in 0..3 {
        // These are synchronous producer calls with no calibration capture
        // or FIFO acknowledgement wait. Pause only competing consumer pops
        // while offering the declared eight attempts exactly once; a legal
        // try-lock drop is not a replacement-sample opportunity.
        runtime.with_training_paused(|| {
            for n in 0..8 {
                let mut w = wave("fixture.feedback.a");
                w.actual.rows[0].owner_incarnation = 100 + p * 8 + n;
                record_with_hooks(
                    &runtime.ids,
                    &runtime.sink,
                    &f.clock,
                    w.actual,
                    w.host,
                    None,
                    0,
                    None,
                    |at| {
                        // Production reservation deliberately never waits for the
                        // worker's active-window lock. This fixture has not begun
                        // preparation yet: wait for that brief contention without
                        // inventing or replacing an issued offer.
                        let deadline =
                            std::time::Instant::now() + std::time::Duration::from_secs(10);
                        loop {
                            if let Some(ticket) = runtime.reserve_live_ticket(Some(at)) {
                                break Some(ticket);
                            }
                            let audit = runtime.training.live.as_ref().unwrap().audit();
                            assert!(!audit.population.failed, "{audit:?}");
                            assert!(!audit.population.closed, "{audit:?}");
                            assert!(std::time::Instant::now() < deadline, "{audit:?}");
                            std::thread::yield_now();
                        }
                    },
                    |_| {},
                );
            }
        });
        // Release the trainer before any await. The real background worker
        // must resolve the original entries and publish from their wakes.
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let audit = runtime.training.live.as_ref().unwrap().audit();
                assert!(audit.publication_error.is_none(), "{audit:?}");
                if (p < 2 && audit.population.phase == (p + 1) as usize)
                    || (p == 2 && audit.qualified_publications == 1)
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("original worker must progress without another inference wave");
    }
    let stats = runtime.sink.stats();
    assert_eq!(stats.raw_offered, 3 * 8);
    assert_eq!(stats.raw_accepted, stats.raw_offered);
    assert_eq!(stats.raw_resolved, stats.raw_offered);
    assert_eq!(stats.raw_resolution_failed, 0);
    assert_eq!(stats.raw_pending, 0);
    assert!(!stats.has_lost_samples(), "{stats:?}");
    runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .unwrap();
    runtime.shutdown().await.unwrap();
}
