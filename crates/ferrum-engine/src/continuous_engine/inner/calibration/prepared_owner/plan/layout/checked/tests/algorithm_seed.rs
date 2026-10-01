//! Real CPU cold inputs survive zero numerical-source allocations and bind the
//! next original source7 header. A declaration alone never installs a model.
use super::*;
use crate::continuous_engine::inner::{
    calibration::geometry_projection::{
        GeometryInputTarget, GeometryProjectionLimits, GeometryProjectionPoint,
    },
    cost_observation::EngineCostRuntime,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    DeclaredAlgorithmUniverseV1, OwnerAlgorithmUniversePolicyV1,
};
use std::num::{NonZeroU32, NonZeroUsize};

#[tokio::test]
async fn cold_inventory_seed_without_publication_reaches_original_online_header() {
    let (mut session, executor) = fixture(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    executor
        .row_selected_cpu_fill
        .store(true, Ordering::Release);
    let inner = Arc::get_mut(&mut session.engine.inner).unwrap();
    inner.tokenizer = Arc::new(tokenizer().await);
    let old_runtime = inner.cost_runtime.as_ref().unwrap().clone();
    let mut settings = SloAutomaticCalibrationSettingsV1::default();
    settings.reuse = ferrum_types::SloAutomaticCalibrationReuseV1::Disabled {};
    let mut config = ferrum_types::SloCostObservationConfig::structured_whole_wave_v2();
    config.live_structured_calibration = ferrum_types::SloLiveStructuredCalibration::AutomaticV1 {
        settings: settings.clone(),
    };
    let runtime = Arc::new(
        EngineCostRuntime::new_with_domain(
            old_runtime.identity.clone(),
            &config,
            None,
            old_runtime.workload_domain().cloned(),
        )
        .unwrap(),
    );
    inner.config.scheduler.slo.cost_observation = config;
    inner.cost_runtime = Some(runtime.clone());
    old_runtime.shutdown().await.unwrap();

    let model = session.configuration().model.model_id.clone();
    let roots = [
        template("test", &model).unwrap(),
        template("test ok a", &model).unwrap(),
    ];
    let inputs = Box::pin(PreparedProbeInputs::new(&mut session, &settings, &roots))
        .await
        .unwrap();
    let mut cursor = inputs.into_cursor().unwrap();
    // A single remaining request cannot supply any complete independent
    // Fit/Residual/Qualification source. Planning has its original own quota.
    let mut budget = ProbeExecutionBudget::new_with_input_projection_limit(
        Instant::now() + Duration::from_secs(30),
        NonZeroUsize::new(1).unwrap(),
        settings.cost_probe.maximum_offered_waves,
        settings.cost_probe.maximum_input_projection_requests,
    );
    assert!(Box::pin(cursor.next(&mut session, &mut budget))
        .await
        .unwrap()
        .is_none());
    assert_eq!(cursor.pending_input_units(), 0);
    assert_eq!(budget.selection_requests_remaining(), 1);
    assert!(budget.preflight_charge().planning_admitted_requests > 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert!(runtime.snapshot().is_none());
    let seed = cursor
        .take_algorithm_seed()
        .expect("checked classes survive no source allocation");
    assert!(cursor.take_algorithm_seed().is_none());

    let probes = crate::continuous_engine::inner::calibration::geometry_projection::tests::requests(
        &session, 2,
    );
    let targets = [2, 3].map(|sequence_tokens| {
        GeometryInputTarget::Decode(GeometryProjectionPoint {
            rows: 2,
            sequence_tokens,
        })
    });
    let report = session
        .project_geometry_inputs(
            probes,
            &targets,
            GeometryProjectionLimits {
                deadline: Instant::now() + Duration::from_secs(10),
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
    let early = report.outcomes[0].branches[0].query.input();
    let later = report.outcomes[1].branches[0].query.input();
    let first_only = DeclaredAlgorithmUniverseV1::from_inputs([early], 4096).unwrap();
    assert!(
        !first_only.contains_checked_algorithms(later).unwrap(),
        "real provider changes class across the declared KV partition"
    );
    assert!(seed.contains_checked_algorithms(early).unwrap());
    assert!(seed.contains_checked_algorithms(later).unwrap());
    runtime
        .install_startup_algorithm_seed(seed.clone())
        .unwrap();
    assert_eq!(runtime.startup_algorithm_seed(), Some(seed.clone()));
    assert!(
        runtime.snapshot().is_none(),
        "algorithm seed confers no numeric qualification"
    );
    runtime.begin_automatic_calibration().unwrap();
    let header = tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Some(header) = runtime.automatic_original_header_for_test() {
                break header;
            }
            runtime.wake_trainer();
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("original source7 opened before serving");
    assert_eq!(
        header.declaration.schedule.algorithm_universe,
        Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1)
    );
    assert_eq!(
        header
            .declaration
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .algorithm_universe,
        Some(seed.clone())
    );
    let roundtrip = serde_json::from_slice::<
        ferrum_scheduler::implementations::continuous::cost_profile::StructuredServiceHeaderV7,
    >(&serde_json::to_vec(&header).unwrap())
    .unwrap();
    assert_eq!(header.protocol, roundtrip.protocol);
    assert!(
        runtime.install_startup_algorithm_seed(seed).is_err(),
        "a live generation is immutable"
    );
    assert!(runtime.snapshot().is_none());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    session.shutdown().await.unwrap();
}
