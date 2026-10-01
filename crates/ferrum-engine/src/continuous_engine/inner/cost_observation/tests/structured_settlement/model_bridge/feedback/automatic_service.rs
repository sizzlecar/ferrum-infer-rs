//! Original private tickets and host settlements exercise the automatic flow.
use super::*;
use ferrum_types::{
    SloAutomaticCalibrationDiagnosticsV1, SloAutomaticCalibrationSettingsV1, SloCostProfileStorage,
    SloLiveStructuredCalibration,
};
#[path = "automatic_service/consumer_capture.rs"]
mod consumer_capture;
#[path = "automatic_service/optional_diagnostics.rs"]
mod optional_diagnostics;
#[path = "automatic_service/shadow_discovery.rs"]
mod shadow_discovery;

struct AutomaticFixture {
    config: SloCostObservationConfig,
    runtime: EngineCostRuntime,
    clock: Arc<VirtualClock>,
    query: StructuredQueryV2,
}
impl AutomaticFixture {
    fn new() -> Self {
        Self::with_diagnostics(SloAutomaticCalibrationDiagnosticsV1::MemoryOnly)
    }
    fn with_diagnostics(diagnostics: SloAutomaticCalibrationDiagnosticsV1) -> Self {
        Self::with_capture_policy(
            diagnostics,
            ferrum_types::SloStructuredActualCapturePolicy::LegacyEveryWave,
        )
    }
    fn with_capture_policy(
        diagnostics: SloAutomaticCalibrationDiagnosticsV1,
        capture_policy: ferrum_types::SloStructuredActualCapturePolicy,
    ) -> Self {
        let clock = Arc::new(VirtualClock(AtomicU64::new(1)));
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        config.structured_actual_capture = capture_policy;
        config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 {
            settings: SloAutomaticCalibrationSettingsV1 {
                // This fixture verifies the original source6 fixed-population protocol.
                population_schedule:
                    ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1,
                route_population: ferrum_types::SloCalibrationRoutePopulationV1::AllAttempts,
                discovery_offered_waves: NonZeroUsize::new(8).unwrap(),
                phase_offered_waves: [NonZeroUsize::new(8).unwrap(); 3],
                maximum_retained_generations: NonZeroUsize::new(2).unwrap(),
                maximum_retained_numeric_bytes: NonZeroUsize::new(16 * 1024 * 1024).unwrap(),
                diagnostics,
                ..Default::default()
            },
        };
        let runtime = EngineCostRuntime::build(identity(), clock.clone(), &config, false).unwrap();
        let w = wave("fixture.feedback.a");
        let query = StructuredQueryV2::exact(
            StructuredInputV2::from_actual(
                &w.prepared.exact,
                &w.prepared.selected,
                &w.prepared.recipe,
            )
            .unwrap(),
        );
        Self {
            config,
            runtime,
            clock,
            query,
        }
    }
    fn live(
        &self,
    ) -> &Arc<crate::continuous_engine::inner::cost_observation::live_calibration::LiveCalibration>
    {
        self.runtime.training.live.as_ref().unwrap()
    }
    fn start(&self) {
        self.runtime.begin_automatic_calibration().unwrap();
        self.runtime.consume_samples();
    }
    fn record(&self, incarnation: u64, algorithm: &'static str, ticket: bool) {
        self.record_wave(incarnation, wave(algorithm), ticket);
    }
    fn record_wave(&self, incarnation: u64, mut w: Wave, ticket: bool) {
        w.actual.rows[0].owner_incarnation = incarnation;
        let capture_demand = std::cell::Cell::new(None);
        record_with_hooks(
            &self.runtime.ids,
            &self.runtime.sink,
            &self.clock,
            w.actual,
            w.host,
            None,
            0,
            None,
            |at| {
                let ticket = if ticket {
                    Some(
                        self.runtime
                            .reserve_live_ticket(Some(at))
                            .expect("fixed offered ticket"),
                    )
                } else {
                    None
                };
                capture_demand.set(Some(self.runtime.actual_capture_demand(
                    &self.config,
                    false,
                    ticket.is_some(),
                    false,
                )));
                ticket
            },
            |call| {
                let demand = capture_demand.get().expect("capture demand precedes call");
                call.structured_capture = demand.structured_sample;
                call.source_generation = demand.source_generation;
                if ticket {
                    assert!(
                        call.structured_capture,
                        "a real reserved ticket must retain all evidence"
                    );
                }
            },
        );
        self.runtime.consume_samples();
    }
    fn generation(&self, generation: u64) {
        self.generation_for(generation, "fixture.feedback.a");
    }
    fn generation_for(&self, generation: u64, algorithm: &'static str) {
        self.runtime.consume_samples();
        let before = self.live().audit();
        assert_eq!(before.population.generation, generation);
        assert_eq!(
            before.population.phase, 3,
            "each generation rediscovers owners"
        );
        for p in 0..4 {
            for n in 0..8 {
                self.record(generation * 100 + p * 8 + n + 1, algorithm, true);
            }
            let audit = self.live().audit();
            assert!(audit.publication_error.is_none(), "{audit:?}");
            if p < 3 {
                assert_eq!(audit.population.phase, p as usize);
                assert_eq!(audit.population.issued, 0, "next phase needs fresh tickets");
                assert_eq!(
                    audit.retained_waves, 0,
                    "previous wave population is not reused"
                );
                assert_eq!(audit.qualified_publications, generation - 1);
            }
        }
        assert_eq!(self.live().audit().qualified_publications, generation);
    }
}

#[tokio::test]
async fn automatic_service_bootstrap_capture_is_deferred_and_does_not_expire_dummy_window() {
    let f = AutomaticFixture::new();
    f.record(1, "fixture.feedback.a", false);
    f.live().poll(Some(1_000_000_000_000));
    f.live().poll(None);
    assert!(f.runtime.reserve_live_ticket(Some(u64::MAX)).is_none());
    let before = f.live().audit();
    assert_eq!(before.population.generation, 0);
    assert_eq!(before.population.issued, 0);
    assert!(!before.population.failed);
    assert_eq!(before.failed_generations, 0);
    f.start();
    assert_eq!(f.live().audit().population.generation, 1);
    assert_eq!(f.live().audit().population.phase, 3);
    assert_eq!(f.live().audit().retained_waves, 0);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_service_fresh_phases_publish_memory_and_continue_beyond_retention_count() {
    let f = AutomaticFixture::new();
    f.start();
    let mut previous = None;
    for generation in 1..=4 {
        f.generation(generation);
        let snapshot = f.runtime.snapshot().expect("qualified memory catalog");
        snapshot
            .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
            .unwrap();
        if let Some(old) = previous.replace(snapshot) {
            assert!(!old.current());
        }
        let receipt = f.runtime.training.published_catalog_receipt().unwrap();
        assert_eq!(receipt.storage, SloCostProfileStorage::Memory);
        assert!(receipt.path.is_none());
        assert_eq!(
            receipt.offered_samples, 24,
            "the eight discovery waves are excluded"
        );
        assert_eq!(receipt.recorded_samples, 24);
        let children = &receipt.structured_whole_wave_v2.as_ref().unwrap().children;
        assert!(children
            .iter()
            .all(|child| child.storage == SloCostProfileStorage::Memory
                && child.source_path.is_none()));
        let audit = f.live().audit();
        assert!(audit.automatic.unwrap().retained_origins <= 2);
    }
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_service_failed_private_offer_retains_evidence_and_restarts_discovery() {
    let f = AutomaticFixture::new();
    f.start();
    f.record(1, "fixture.feedback.a", true);
    let lost = f.runtime.reserve_live_ticket(f.clock.now_ns()).unwrap();
    drop(lost);
    f.runtime.consume_samples();
    let audit = f.live().audit();
    assert_eq!(audit.failed_generations, 1);
    let automatic = audit.automatic.unwrap();
    assert_eq!(automatic.retained_failed_wave_count, 1);
    let failed = automatic.retained_failed_ticket_population.unwrap();
    assert_eq!(failed.issued, 2);
    assert_eq!(failed.retired, 2);
    assert!(failed.failed);
    f.runtime.consume_samples();
    let audit = f.live().audit();
    assert_eq!(audit.population.generation, 2);
    assert_eq!(audit.population.phase, 3);
    assert_eq!(audit.population.issued, 0);
    assert_eq!(audit.retained_waves, 0);
    assert_eq!(audit.automatic.unwrap().retained_failed_wave_count, 1);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_service_unseen_owner_cannot_expand_frozen_scope() {
    let f = AutomaticFixture::new();
    f.start();
    for n in 0..8 {
        f.record(n + 1, "fixture.feedback.a", true);
    }
    for n in 0..8 {
        f.record(n + 100, "fixture.feedback.b", true);
    }
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 0);
    assert_eq!(audit.failed_generations, 1);
    assert!(f.runtime.snapshot().is_none());
    f.runtime.consume_samples();
    assert_eq!(f.live().audit().population.generation, 2);
    assert_eq!(f.live().audit().population.phase, 0);
    assert_eq!(f.live().audit().population.issued, 0);
    // The new catalogue comes from the whole old Fit population. It still
    // needs three disjoint numerical populations of its own before publication.
    for n in 0..24 {
        f.record(n + 200, "fixture.feedback.b", true);
        if n < 23 {
            assert_eq!(f.live().audit().qualified_publications, 0);
        }
    }
    assert_eq!(f.live().audit().qualified_publications, 1);
    let snapshot = f.runtime.snapshot().unwrap();
    assert!(snapshot
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .is_err());
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_service_distinct_owner_origins_obey_retention_without_stopping() {
    let f = AutomaticFixture::new();
    f.start();
    let algorithms = [
        "fixture.feedback.a",
        "fixture.feedback.b",
        "fixture.feedback.c",
    ];
    for (index, algorithm) in algorithms.into_iter().enumerate() {
        f.generation_for(index as u64 + 1, algorithm);
    }
    let snapshot = f.runtime.snapshot().unwrap();
    assert!(snapshot
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .is_err());
    for algorithm in &algorithms[1..] {
        let w = wave(algorithm);
        let query = StructuredQueryV2::exact(
            StructuredInputV2::from_actual(
                &w.prepared.exact,
                &w.prepared.selected,
                &w.prepared.recipe,
            )
            .unwrap(),
        );
        snapshot
            .audit_structured_query_v2(&query, f.clock.now_ns().unwrap())
            .unwrap();
    }
    assert_eq!(f.live().audit().automatic.unwrap().retained_origins, 2);
    let measured = f
        .runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap()
        .iter()
        .map(|child| child.retained_payload_bytes().unwrap())
        .sum::<usize>();
    let accounted = f
        .live()
        .audit()
        .automatic
        .unwrap()
        .retained_payload_bytes_upper_bound;
    assert!(accounted >= measured && accounted > 0);
    assert!(accounted <= 16 * 1024 * 1024 * 2 / 5 * 2);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_service_new_owner_stays_unknown_without_revoking_an_unrelated_owner() {
    let f = AutomaticFixture::new();
    f.start();
    f.generation(1);
    let previous = f.runtime.snapshot().unwrap();
    let before = f.runtime.audit_snapshot().structured_feedback.unwrap();
    f.runtime.consume_samples();
    f.record(201, "fixture.feedback.b", true);
    assert!(previous.current());
    previous
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .unwrap();
    let wave = wave("fixture.feedback.b");
    let unknown = StructuredQueryV2::exact(
        StructuredInputV2::from_actual(
            &wave.prepared.exact,
            &wave.prepared.selected,
            &wave.prepared.recipe,
        )
        .unwrap(),
    );
    assert!(matches!(
        previous.audit_structured_query_v2(&unknown, f.clock.now_ns().unwrap()),
        Err(StructuredUnknownV2::WrongDomain)
    ));
    let after = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(
        after.outside_catalog_observations,
        before.outside_catalog_observations + 1
    );
    assert_eq!(
        after.uncomparable_observations,
        before.uncomparable_observations
    );
    assert_eq!(f.live().audit().qualified_publications, 1);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_service_new_kv_range_is_unknown_without_revoking_qualified_same_owner() {
    let f = AutomaticFixture::new();
    f.start();
    f.generation(1);
    let previous = f.runtime.snapshot().unwrap();
    let before = f.runtime.audit_snapshot().structured_feedback.unwrap();
    let outside = wave_with_kv("fixture.feedback.a", 16);
    let query = StructuredQueryV2::exact(
        StructuredInputV2::from_actual(
            &outside.prepared.exact,
            &outside.prepared.selected,
            &outside.prepared.recipe,
        )
        .unwrap(),
    );
    assert_eq!(query.owner(), f.query.owner());
    assert!(matches!(
        previous.audit_structured_query_v2(&query, f.clock.now_ns().unwrap()),
        Err(StructuredUnknownV2::JointSupport | StructuredUnknownV2::UnidentifiedDirection)
    ));
    f.runtime.consume_samples();
    f.record_wave(201, outside, true);
    assert!(previous.current());
    previous
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .unwrap();
    let after = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(
        after.outside_support_observations,
        before.outside_support_observations + 1
    );
    assert_eq!(
        after.outside_catalog_observations,
        before.outside_catalog_observations
    );
    assert_eq!(after.compared, before.compared);
    assert_eq!(
        after.uncomparable_observations,
        before.uncomparable_observations
    );
    assert_eq!(after.maximum_margin_ns, before.maximum_margin_ns);
    assert_eq!(f.live().audit().qualified_publications, 1);
    f.runtime.shutdown().await.unwrap();
}

struct Directory {
    path: PathBuf,
}
impl Directory {
    fn new() -> Self {
        Self {
            path: std::env::temp_dir().join(format!(
                "ferrum-automatic-live-directory-{}",
                uuid::Uuid::new_v4()
            )),
        }
    }
    fn policy(&self) -> SloAutomaticCalibrationDiagnosticsV1 {
        SloAutomaticCalibrationDiagnosticsV1::Directory {
            directory: self.path.clone(),
            maximum_source_bytes: NonZeroU64::new(8 * 1024 * 1024).unwrap(),
            maximum_total_bytes: NonZeroU64::new(32 * 1024 * 1024).unwrap(),
            maximum_retained_generations: NonZeroUsize::MIN,
        }
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[tokio::test]
async fn automatic_service_directory_publishes_and_rotates_across_process_lifetimes() {
    use sha2::Digest;
    let directory = Directory::new();
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    f.generation(1);
    let receipt = f.runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(receipt.storage, SloCostProfileStorage::Memory);
    assert!(receipt.path.is_none());
    let diagnostic = f
        .live()
        .audit()
        .automatic
        .unwrap()
        .last_diagnostic_publication
        .unwrap();
    assert_eq!(diagnostic.generation, 1);
    let source = diagnostic.source;
    let raw = fs::read(&source.path).unwrap();
    assert_eq!(raw.len() as u64, source.bytes);
    assert_eq!(<[u8; 32]>::from(sha2::Sha256::digest(&raw)), source.digest);
    let child = &receipt.structured_whole_wave_v2.as_ref().unwrap().children[0];
    assert_eq!(child.storage, SloCostProfileStorage::Memory);
    assert!(child.profile_path.is_none() && child.source_path.is_none());
    assert_eq!(child.source_sha256, source.digest);
    assert_eq!(child.source_bytes, source.bytes);
    f.generation(2);
    assert!(
        !source.path.exists(),
        "the bounded archive rotates independently of prediction"
    );
    let second = f
        .live()
        .audit()
        .automatic
        .unwrap()
        .last_diagnostic_publication
        .unwrap()
        .source
        .path;
    f.runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .unwrap();
    f.runtime.shutdown().await.unwrap();
    drop(f);
    let restarted = AutomaticFixture::with_diagnostics(directory.policy());
    restarted.start();
    restarted.generation(1);
    assert!(!second.exists());
    let fresh = restarted
        .live()
        .audit()
        .automatic
        .unwrap()
        .last_diagnostic_publication
        .unwrap()
        .source
        .path;
    assert!(fresh.is_file());
    assert_ne!(
        fresh, second,
        "restarts never overwrite original source names"
    );
    restarted.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_service_directory_keeps_failed_discovery_originals() {
    let directory = Directory::new();
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    f.record(1, "fixture.feedback.a", true);
    drop(f.runtime.reserve_live_ticket(f.clock.now_ns()).unwrap());
    f.runtime.consume_samples();
    let failed = f.live().audit().last_failed_source.unwrap();
    assert!(failed.footer_complete);
    assert!(!failed.retained_unpublished);
    let bytes = fs::read(failed.path.unwrap()).unwrap();
    let rows: Vec<serde_json::Value> = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(
        rows.iter()
            .filter(|row| row.get("stages").is_some())
            .count(),
        1
    );
    assert_eq!(
        rows.iter()
            .filter(|row| row.get("failed_ticket").is_some())
            .count(),
        1
    );
    assert!(rows.last().unwrap().get("failure").is_some());
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_service_directory_keeps_failed_numerical_private_population() {
    let directory = Directory::new();
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    for n in 0..11 {
        f.record(n + 1, "fixture.feedback.a", true);
    }
    assert_eq!(f.live().audit().population.phase, 0);
    drop(f.runtime.reserve_live_ticket(f.clock.now_ns()).unwrap());
    f.runtime.consume_samples();
    let failed = f.live().audit().last_failed_source.unwrap();
    assert!(failed.footer_complete);
    assert!(!failed.retained_unpublished);
    let bytes = fs::read(failed.path.unwrap()).unwrap();
    let rows: Vec<serde_json::Value> = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(
        rows.iter()
            .filter(|row| row.get("kind").and_then(|value| value.as_str()) == Some("completed"))
            .count(),
        3
    );
    assert_eq!(
        rows.iter()
            .filter(|row| row.get("kind").and_then(|value| value.as_str()) == Some("ticket_failed"))
            .count(),
        1
    );
    assert!(rows
        .last()
        .unwrap()
        .get("failure")
        .is_some_and(|value| value.is_string()));
    assert_eq!(f.live().audit().qualified_publications, 0);
    f.runtime.shutdown().await.unwrap();
}
