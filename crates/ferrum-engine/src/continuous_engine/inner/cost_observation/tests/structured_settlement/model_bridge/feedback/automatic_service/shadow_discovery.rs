//! Real settled waves discover a later generation; numerical samples never do.
use super::*;

fn phase(f: &AutomaticFixture, start: u64, algorithm: &'static str) {
    for n in 0..8 {
        f.record(start + n, algorithm, true);
    }
}

fn query(algorithm: &'static str) -> StructuredQueryV2 {
    let w = wave(algorithm);
    StructuredQueryV2::exact(
        StructuredInputV2::from_actual(&w.prepared.exact, &w.prepared.selected, &w.prepared.recipe)
            .unwrap(),
    )
}

#[tokio::test]
async fn automatic_shadow_uses_latest_complete_residual_and_retains_failed_originals() {
    let directory = Directory::new();
    let f = AutomaticFixture::with_diagnostics(directory.policy());
    f.start();
    phase(&f, 1, "fixture.feedback.a"); // independent discovery
    phase(&f, 101, "fixture.feedback.a"); // Fit remains viable
    assert_eq!(f.live().audit().population.phase, 1);
    assert_eq!(f.live().audit().failed_generations, 0);
    phase(&f, 201, "fixture.feedback.b"); // all old children fail Residual
    let failed = f.live().audit();
    assert_eq!(failed.failed_generations, 1);
    assert_eq!(failed.qualified_publications, 0);
    assert_eq!(failed.population.phase, 1);
    assert_eq!(
        (failed.population.issued, failed.population.retired),
        (8, 8)
    );
    let bytes = fs::read(failed.last_failed_source.unwrap().path.unwrap()).unwrap();
    let records: Vec<serde_json::Value> = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert_eq!(
        records.iter().filter(|r| r["kind"] == "completed").count(),
        16
    );
    assert_eq!(
        records
            .iter()
            .filter(|r| r["kind"] == "phase_freeze")
            .count(),
        2
    );
    assert_eq!(
        records.iter().filter(|r| r["kind"] == "phase_open").count(),
        2
    );
    assert!(records.last().unwrap()["failure"].is_string());

    f.runtime.consume_samples();
    let next = f.live().audit();
    assert_eq!((next.population.generation, next.population.phase), (2, 0));
    assert_eq!((next.population.issued, next.retained_waves), (0, 0));
    let json = serde_json::to_value(&next).unwrap();
    let origin = &json["automatic"]["discovery_origin"];
    assert_eq!(origin["generation"], 1);
    assert_eq!(origin["phase"], 1);
    assert_eq!(origin["population"]["issued"], 8);
    assert_eq!(origin["population"]["retired"], 8);
    assert_eq!(origin["population"]["failed"], false);
    let discovery_cutoff = origin["closing_fifo_cutoff"].as_u64().unwrap();
    for p in 0..3 {
        phase(&f, 301 + p * 100, "fixture.feedback.b");
        assert_eq!(f.live().audit().qualified_publications, u64::from(p == 2));
    }
    let receipt = f.runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(receipt.offered_samples, 24);
    assert_eq!(receipt.recorded_samples, 24);
    let source = f
        .live()
        .audit()
        .automatic
        .unwrap()
        .last_diagnostic_publication
        .unwrap()
        .source;
    let bytes = fs::read(source.path).unwrap();
    let records: Vec<serde_json::Value> = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();
    assert!(records
        .iter()
        .filter(|r| r["kind"] == "completed")
        .all(|r| r["wave"]["fifo"].as_u64().unwrap() > discovery_cutoff));
    let snapshot = f.runtime.snapshot().unwrap();
    snapshot
        .audit_structured_query_v2(&query("fixture.feedback.b"), f.clock.now_ns().unwrap())
        .unwrap();
    assert!(snapshot
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .is_err());
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_shadow_partial_or_lost_population_restarts_original_discovery() {
    let f = AutomaticFixture::new();
    f.start();
    phase(&f, 1, "fixture.feedback.a");
    for n in 0..7 {
        f.record(101 + n, "fixture.feedback.b", true);
    }
    drop(f.runtime.reserve_live_ticket(f.clock.now_ns()).unwrap());
    f.runtime.consume_samples();
    let failed = f.live().audit();
    assert_eq!(failed.failed_generations, 1);
    assert_eq!(failed.qualified_publications, 0);
    assert_eq!(failed.automatic.unwrap().retained_failed_wave_count, 7);
    f.runtime.consume_samples();
    let next = f.live().audit();
    assert_eq!((next.population.generation, next.population.phase), (2, 3));
    assert!(serde_json::to_value(next).unwrap()["automatic"]["discovery_origin"].is_null());
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_shadow_expired_seed_restarts_original_discovery() {
    let f = AutomaticFixture::new();
    f.start();
    phase(&f, 1, "fixture.feedback.a");
    phase(&f, 101, "fixture.feedback.b");
    assert_eq!(f.live().audit().failed_generations, 1);
    let SloLiveStructuredCalibration::AutomaticV1 { settings } =
        &f.config.live_structured_calibration
    else {
        unreachable!()
    };
    f.clock
        .set(f.clock.now_ns().unwrap() + settings.maximum_window_ns.get() + 1);
    f.runtime.consume_samples();
    let next = f.live().audit();
    assert_eq!((next.population.generation, next.population.phase), (2, 3));
    assert!(serde_json::to_value(next).unwrap()["automatic"]["discovery_origin"].is_null());
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_shadow_failed_new_candidate_preserves_published_owner_until_new_qualification() {
    let f = AutomaticFixture::new();
    f.start();
    f.generation(1);
    let original = f.runtime.snapshot().unwrap();
    f.runtime.consume_samples();
    phase(&f, 201, "fixture.feedback.a");
    phase(&f, 301, "fixture.feedback.b");
    assert_eq!(f.live().audit().failed_generations, 1);
    assert_eq!(f.live().audit().qualified_publications, 1);
    assert!(original.current());
    original
        .audit_structured_query_v2(&f.query, f.clock.now_ns().unwrap())
        .unwrap();
    assert!(original
        .audit_structured_query_v2(&query("fixture.feedback.b"), f.clock.now_ns().unwrap())
        .is_err());
    f.runtime.consume_samples();
    assert_eq!(
        (
            f.live().audit().population.generation,
            f.live().audit().population.phase
        ),
        (3, 0)
    );
    for p in 0..3 {
        phase(&f, 401 + p * 100, "fixture.feedback.b");
        if p < 2 {
            assert!(original.current());
            assert_eq!(f.live().audit().qualified_publications, 1);
        }
    }
    assert_eq!(f.live().audit().qualified_publications, 2);
    let next = f.runtime.snapshot().unwrap();
    for q in [&f.query, &query("fixture.feedback.b")] {
        next.audit_structured_query_v2(q, f.clock.now_ns().unwrap())
            .unwrap();
    }
    f.runtime.shutdown().await.unwrap();
}
