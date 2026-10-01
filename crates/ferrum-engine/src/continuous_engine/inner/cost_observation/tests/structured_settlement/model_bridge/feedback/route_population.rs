//! Real private core selector + bound submission -> engine settlement -> live
//! fixed populations -> original source6/profile13. Controlled provider clocks
//! are protocol fixtures, not measured GPU performance.
use super::*;
#[path = "route_population/automatic_physical.rs"]
mod automatic_physical;
#[path = "route_population/diagnostics.rs"]
mod diagnostics;
#[path = "route_population/fixture.rs"]
mod fixture;
#[path = "route_population/no_submission.rs"]
mod no_submission;
#[path = "route_population/private_capture.rs"]
mod private_capture;
#[path = "../../../../../../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../../../../../../../ferrum-interfaces/tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;
use ferrum_types::{
    SloAutomaticCalibrationDiagnosticsV1, SloAutomaticCalibrationSettingsV1,
    SloCalibrationRoutePopulationV1, SloLiveStructuredCalibration,
};
use fixture::record_route;
use sha2::Digest;

struct Automatic {
    runtime: EngineCostRuntime,
    clock: Arc<VirtualClock>,
    directory: PathBuf,
    outside_without_layout: bool,
}
impl Automatic {
    fn new(numeric_offers: usize) -> Self {
        Self::with_population(
            numeric_offers,
            SloCalibrationRoutePopulationV1::WarmOrGraphDisabledV1,
        )
    }
    fn with_population(numeric_offers: usize, population: SloCalibrationRoutePopulationV1) -> Self {
        let directory =
            std::env::temp_dir().join(format!("ferrum-outside-route-{}", uuid::Uuid::new_v4()));
        let clock = Arc::new(VirtualClock(AtomicU64::new(1)));
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 {
            settings: SloAutomaticCalibrationSettingsV1 {
                // This fixture verifies the original source6 fixed-population protocol.
                population_schedule:
                    ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1,
                route_population: population,
                discovery_offered_waves: NonZeroUsize::new(9).unwrap(),
                phase_offered_waves: [NonZeroUsize::new(numeric_offers).unwrap(); 3],
                maximum_retained_numeric_bytes: NonZeroUsize::new(32 * 1024 * 1024).unwrap(),
                diagnostics: SloAutomaticCalibrationDiagnosticsV1::Directory {
                    directory: directory.clone(),
                    maximum_source_bytes: std::num::NonZeroU64::new(16 * 1024 * 1024).unwrap(),
                    maximum_total_bytes: std::num::NonZeroU64::new(128 * 1024 * 1024).unwrap(),
                    maximum_retained_generations: NonZeroUsize::new(4).unwrap(),
                },
                ..Default::default()
            },
        };
        let runtime = EngineCostRuntime::build(identity(), clock.clone(), &config, false).unwrap();
        runtime.begin_automatic_calibration().unwrap();
        runtime.consume_samples();
        Self {
            runtime,
            clock,
            directory,
            outside_without_layout: false,
        }
    }
    fn record(&self, outside: bool) -> Arc<HostStageEvidenceV1> {
        if outside && self.outside_without_layout {
            return fixture::record_route_with_hook(
                &self.runtime,
                &self.clock,
                wave("fixture.outside-route"),
                true,
                false,
                true,
                |_| {},
            )
            .unwrap();
        }
        record_route(
            &self.runtime,
            &self.clock,
            wave("fixture.outside-route"),
            outside,
            false,
        )
        .unwrap()
    }
    fn live(
        &self,
    ) -> &Arc<crate::continuous_engine::inner::cost_observation::live_calibration::LiveCalibration>
    {
        self.runtime.training.live.as_ref().unwrap()
    }
    fn phase(&self, offers: usize, outside: usize) {
        for n in 0..offers {
            self.record(n < outside);
            self.runtime.consume_samples();
        }
    }
}
impl Drop for Automatic {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[tokio::test]
async fn route_population_original_attempts_publish_and_profile13_replays_all_three_phases() {
    original_attempts_publish_and_replay(false).await;
}

#[tokio::test]
async fn route_population_no_program_layout_preserves_attempts_and_profile13_replay() {
    original_attempts_publish_and_replay(true).await;
}

async fn original_attempts_publish_and_replay(outside_without_layout: bool) {
    let mut f = Automatic::new(9);
    f.outside_without_layout = outside_without_layout;
    f.phase(9, 1); // discovery does not train
    assert_eq!(f.live().audit().population.phase, 0);
    for p in 0..3 {
        f.phase(9, 1);
        assert!(
            f.live().audit().publication_error.is_none(),
            "{:#?}",
            f.live().audit()
        );
        if p < 2 {
            assert_eq!(f.live().audit().population.phase, p + 1);
        }
    }
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 1, "{audit:#?}");
    let json = serde_json::to_value(&audit).unwrap();
    let published = &json["automatic"]["last_published_population"];
    for phase in published["phase_populations"].as_array().unwrap() {
        assert_eq!(phase["issued"], 9);
        assert_eq!(phase["eligible_route"], 8);
        assert_eq!(phase["outside_declared_route"], 1);
        assert_eq!(phase["retired"], 9);
    }
    let source = audit
        .automatic
        .unwrap()
        .last_diagnostic_publication
        .unwrap()
        .source
        .path;
    let bytes = fs::read(&source).unwrap();
    let records: Vec<serde_json::Value> = bytes
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(
        records
            .iter()
            .filter(|r| r["kind"] == "outside_declared_route")
            .count(),
        3
    );
    let profile = f.directory.join("strict-profile13.json");
    let limits = file::CostProfileLoadLimits::default();
    let exported = file::export_structured_profile_v13(
        &source,
        sha2::Sha256::digest(&bytes).into(),
        &profile,
        0,
        &limits,
    )
    .unwrap();
    assert_eq!(exported.children.len(), 1);
    let fingerprint = f.runtime.snapshot().unwrap().fingerprint().clone();
    let closing = records.last().unwrap()["closing"]["wall_unix_ns"]
        .as_u64()
        .unwrap();
    let imported = file::load_structured_profile_v13(
        &profile,
        &fingerprint,
        &limits,
        file::ProfileLoadClock {
            wall_unix_ns: Some(closing + 10),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: 1000,
        },
    )
    .unwrap();
    assert_eq!(imported.offered_attempts, 27);
    assert_eq!(
        imported.children[0]
            .provenance()
            .phases
            .each_ref()
            .map(|p| p.members),
        [8; 3]
    );
    let original = records
        .iter()
        .find(|r| r["kind"] == "outside_declared_route")
        .unwrap();
    for edit in 0..if outside_without_layout { 10 } else { 5 } {
        let mut changed = original.clone();
        let e = &mut changed["wave"]["evidence"];
        match edit {
            0 => {
                e["selected_at_ns"] =
                    (e["submitted"]["submission_started_at_ns"].as_u64().unwrap() + 1).into()
            }
            1 => e["submitted"]["plan_hash"] = "00".repeat(32).into(),
            2 => e["physical"]["lost_observations"] = 1.into(),
            3 => e["host_stages"]["rows"][0]["terminal"]["generated_tokens"] = 0.into(),
            4 => e["submitted"]["batch_invocation"] = 0.into(),
            5 => e["selection"]["non_reusable_wave"] = serde_json::Value::Null,
            6 => e["selection"]["non_reusable_wave"]["immediate_sequences"] = 2.into(),
            7 => e["selection"]["non_reusable_wave"]["immediate_tokens"] = 2.into(),
            8 => e["submitted"]["graph"]["capture_requested"] = true.into(),
            9 => e["selection"]["class"] = "warm".into(),
            _ => unreachable!(),
        }
        let file::StructuredServiceRecordV6::OutsideDeclaredRoute { wave: changed } =
            serde_json::from_value(changed).unwrap()
        else {
            unreachable!()
        };
        assert!(
            changed
                .validate_settlement(&file::ProfileFingerprint::from(&fingerprint), 1)
                .is_err(),
            "tamper {edit}"
        );
    }
    let query_wave = wave("fixture.outside-route");
    let query = StructuredQueryV2::exact(
        StructuredInputV2::from_actual(
            &query_wave.prepared.exact,
            &query_wave.prepared.selected,
            &query_wave.prepared.recipe,
        )
        .unwrap(),
    );
    let replayed = imported.children[0]
        .predict_query_local(&fingerprint, &query, 1000)
        .unwrap();
    let now = f.clock.now_ns().unwrap();
    let memory = f
        .runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&query, now)
        .unwrap();
    assert_eq!(memory.planning_ns, replayed.planning_ns);
    fs::remove_file(&profile).unwrap();
    // More failed generations cannot erase the latest actual published population.
    f.runtime.consume_samples();
    let feedback_before = f.runtime.audit_snapshot().structured_feedback.unwrap();
    for _ in 0..6 {
        f.phase(9, 9);
        f.runtime.consume_samples();
    }
    let later = serde_json::to_value(f.live().audit()).unwrap();
    assert_eq!(later["automatic"]["last_published_population"], *published);
    let feedback_after = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(
        feedback_after.outside_route_observations,
        feedback_before.outside_route_observations + 54
    );
    assert_eq!(
        feedback_after.uncomparable_observations,
        feedback_before.uncomparable_observations
    );
    assert!(feedback_after.revoked.is_none());
    f.runtime
        .snapshot()
        .unwrap()
        .audit_structured_query_v2(&query, f.clock.now_ns().unwrap())
        .unwrap();
}

#[tokio::test]
async fn route_population_outside_does_not_replace_numeric_minimum_or_swallow_unknown() {
    let f = Automatic::new(8);
    f.phase(9, 1);
    f.phase(8, 1); // seven numerical members; never replenish excluded offers
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 0);
    assert!(audit.failed_generations > 0, "{audit:#?}");
    let failed = serde_json::to_value(&audit).unwrap();
    let phase = &failed["automatic"]["history"][0]["phase_populations"][0];
    assert_eq!(phase["issued"], 8);
    assert_eq!(phase["eligible_route"], 7);
    assert_eq!(phase["outside_declared_route"], 1);
    assert!(failed["automatic"]["history"][0]["phase_populations"][1].is_null());
    assert!(failed["automatic"]["history"][0]["phase_populations"][2].is_null());
    f.runtime.consume_samples();
    record_route(
        &f.runtime,
        &f.clock,
        wave("fixture.outside-route"),
        true,
        true,
    );
    f.runtime.consume_samples();
    let audit = f.live().audit();
    assert_eq!(audit.qualified_publications, 0);
    assert!(
        audit.failed_generations >= 2,
        "later non-GraphPath unknown remains failed: {audit:#?}"
    );
}

#[tokio::test]
async fn route_population_outside_terminal_advances_the_same_original_owner_frontier() {
    let f = Automatic::new(9);
    let original = f.record(true);
    f.runtime.consume_samples();
    let before = f.live().audit();
    assert_eq!(
        (
            before.population.issued,
            before.population.outside_declared_route
        ),
        (1, 1)
    );
    let mut duplicate = wave("fixture.outside-route");
    duplicate.actual.rows[0].request_id = original.rows[0].request_id.clone();
    duplicate.actual.rows[0].owner_incarnation = original.rows[0].owner_incarnation;
    duplicate.actual.rows[0].work_generation = original.rows[0].work_generation;
    record_route(&f.runtime, &f.clock, duplicate, false, false).unwrap();
    f.runtime.consume_samples();
    let after = f.live().audit();
    assert_eq!(after.qualified_publications, 0);
    assert!(after.failed_generations > 0);
    assert_eq!(
        after.automatic.unwrap().retained_failed_reason,
        Some("within_window_frontier_discontinuity")
    );
}
