//! Default automatic policy, no profile path or diagnostics prerequisite:
//! original source8 -> orderly worker shutdown -> a new runtime -> real witness.
use super::*;
mod subprocess;
use ferrum_interfaces::execution_cost::CostMonotonicDomainV1;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1;

struct Clock {
    now: AtomicU64,
    domain: CostMonotonicDomainV1,
}
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.now.fetch_add(1, Ordering::Relaxed))
    }
    fn monotonic_domain(&self) -> Option<&CostMonotonicDomainV1> {
        Some(&self.domain)
    }
}
fn cache_policy(path: &std::path::Path) -> ferrum_types::SloAutomaticCalibrationReuseV1 {
    let mut policy = ferrum_types::SloAutomaticCalibrationReuseV1::default();
    let ferrum_types::SloAutomaticCalibrationReuseV1::SameBootCleanShutdownV1 { location, .. } =
        &mut policy
    else {
        unreachable!()
    };
    *location =
        ferrum_types::SloAutomaticCalibrationCacheLocationV1::Directory { path: path.into() };
    policy
}
fn native(executor: &ControlledExecutor) {
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    executor.enable_structured_query_route();
}

/// Capture actual provider inputs before any numerical sample or publication.
/// This declaration must survive independently of the qualified child roster.
async fn install_cold_seed(
    session: &mut CalibrationSession,
    executor: &ControlledExecutor,
) -> DeclaredAlgorithmUniverseV1 {
    use crate::continuous_engine::inner::calibration::geometry_projection::{
        GeometryInputTarget, GeometryProjectionLimits, GeometryProjectionPoint,
    };
    executor.prepare_structured_query_resources();
    // The numerical source fixture intentionally mixes CLI and Completion SSE.
    // A homogeneous numerical U must instead use distinct real owners from
    // the same installed output/sampling template, as the production inventory.
    let probes = crate::continuous_engine::inner::calibration::geometry_projection::tests::requests(
        session, 2,
    );
    let report = session
        .project_geometry_inputs(
            probes,
            &[GeometryInputTarget::Decode(GeometryProjectionPoint {
                rows: 2,
                sequence_tokens: 2,
            })],
            GeometryProjectionLimits {
                deadline: tokio::time::Instant::now() + Duration::from_secs(10),
                maximum_projections: 64,
                maximum_route_states: 8,
                maximum_retained_bytes: 4 * 1024 * 1024,
                prefill_chunk: NonZeroU32::new(2).unwrap(),
                prefill_row_ceiling: None,
            },
            &[],
        )
        .await
        .unwrap();
    assert!(
        report.outcomes.iter().all(|o| o.unknown.is_none()),
        "{:?}",
        report
            .outcomes
            .iter()
            .map(|o| &o.unknown)
            .collect::<Vec<_>>()
    );
    let seed = DeclaredAlgorithmUniverseV1::from_inputs(
        report
            .outcomes
            .iter()
            .flat_map(|o| o.branches.iter())
            .map(|b| b.query.input()),
        4096,
    )
    .unwrap();
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(session.frontiers().unwrap().is_empty());
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap();
    assert!(
        runtime.snapshot().is_none(),
        "coordinates grant no numerical qualification"
    );
    let mut corrupt = serde_json::to_value(&seed).unwrap();
    corrupt["algorithms"][0]["kind"] = serde_json::json!(255);
    assert!(serde_json::from_value::<DeclaredAlgorithmUniverseV1>(corrupt).is_err());
    let mut wrong_domain = serde_json::to_value(&seed).unwrap();
    wrong_domain["workload_domain"] = serde_json::to_value([90u8; 32]).unwrap();
    let wrong_domain = serde_json::from_value::<DeclaredAlgorithmUniverseV1>(wrong_domain).unwrap();
    assert!(runtime
        .install_startup_algorithm_seed(wrong_domain)
        .is_err());
    assert!(runtime.startup_algorithm_seed().is_none());
    runtime
        .install_startup_algorithm_seed(seed.clone())
        .unwrap();
    assert_eq!(runtime.startup_algorithm_seed(), Some(seed.clone()));
    seed
}

#[tokio::test]
async fn automatic_reuse_runtime_source8_restores_without_diagnostic_directory_or_manual_import() {
    let directory = Directory::new();
    let clock = Arc::new(Clock {
        now: AtomicU64::new(100),
        domain: CostMonotonicDomainV1::new_macos_continuous([21; 16]).unwrap(),
    });
    let policy = cache_policy(&directory.0);
    let (mut first, executor) =
        automatic_session_with_storage(2, clock.clone(), Default::default(), policy.clone()).await;
    native(&executor);
    let runtime = first.engine.inner.cost_runtime.as_ref().unwrap().clone();
    assert!(!runtime.reused_cost_is_fresh());
    let original_seed = install_cold_seed(&mut first, &executor).await;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
    first.begin_startup_owner_series(1, deadline).unwrap();
    first
        .begin_prepared_owner_source(full_population(&first), CostProfileLoadLimits::default())
        .await
        .unwrap();
    assert!(
        first
            .prepared_owner_capture
            .as_ref()
            .unwrap()
            .journal_observer()
            .is_none(),
        "diagnostics remains MemoryOnly"
    );
    collect_source(&mut first, [16, 8, 8], deadline).await;
    first.activate_prepared_owner_source().await.unwrap();
    first.retire_startup_owner_source().await.unwrap();
    first.finish_startup_owner_series().unwrap();
    let old = runtime.startup_series_children_for_test().unwrap();
    assert!(!old.is_empty());
    // Existing helper checks fresh actual query, original CostWitness,
    // once-only backend submission, host reconciliation, then real shutdown.
    future::verify(first, executor).await;
    drop(runtime);
    assert!(directory.0.join("cache-v1/manifest.json").is_file());
    assert!(!directory.0.join("cache-v1/dirty").exists());

    let (second, executor) =
        automatic_session_with_storage(2, clock.clone(), Default::default(), policy).await;
    native(&executor);
    let runtime = second.engine.inner.cost_runtime.as_ref().unwrap().clone();
    assert!(
        runtime.reused_cost_is_fresh(),
        "normal runtime constructor must hit the clean cache"
    );
    assert_eq!(
        executor.physical.load(Ordering::Acquire),
        0,
        "cache load and independent numerical replay submit no probe wave"
    );
    assert_eq!(runtime.startup_algorithm_seed(), Some(original_seed));
    let restored = runtime.startup_series_children_for_test().unwrap();
    assert_eq!(restored.len(), old.len());
    for child in &restored {
        let before = old
            .iter()
            .find(|c| c.domain_signature() == child.domain_signature())
            .unwrap();
        assert_eq!(
            before.provenance().source_sha256,
            child.provenance().source_sha256
        );
        assert_eq!(
            before.provenance().parameters_sha256,
            child.provenance().parameters_sha256
        );
        assert_eq!(before.provenance().clock, child.provenance().clock);
    }
    future::verify(second, executor).await;
    drop(runtime);
    assert!(
        !directory.0.join("cache-v1/dirty").exists(),
        "an unchanged restored File catalog can be handed off again without timestamp renewal"
    );
}
