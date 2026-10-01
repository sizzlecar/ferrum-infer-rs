//! Desired automatic lifecycle: independent families can resume after complete
//! intervening input blocks. The old global phase schedule is expected to fail
//! the positive test; it must not be made an expected-panic or ignored test.
use super::*;
#[path = "family_progress/failure_archive.rs"]
mod failure_archive;
#[path = "family_progress/restart_expiry.rs"]
mod restart_expiry;
#[path = "family_progress/unticketed_boundary.rs"]
mod unticketed_boundary;
use std::num::NonZeroUsize;

#[path = "family_progress/shutdown.rs"]
mod shutdown;

#[path = "family_progress/successful_archive.rs"]
mod successful_archive;

#[path = "family_progress/incremental.rs"]
mod incremental;

#[path = "family_progress/input_coverage_repro.rs"]
mod input_coverage_repro;

#[path = "family_progress/startup_activation.rs"]
mod startup_activation;

#[path = "family_progress/rolling.rs"]
mod rolling;
#[path = "family_progress/runtime_renewal.rs"]
mod runtime_renewal;

const OFFERS: usize = 8;
const A: &str = "fixture.family-progress.a";
const B: &str = "fixture.family-progress.b";

struct Families {
    runtime: EngineCostRuntime,
    clock: Arc<VirtualClock>,
    domain: CostWorkloadDomainV1,
    recorded: u64,
}

impl Families {
    fn new() -> Self {
        Self::new_with_diagnostics(SloAutomaticCalibrationDiagnosticsV1::MemoryOnly)
    }

    fn new_with_diagnostics(diagnostics: SloAutomaticCalibrationDiagnosticsV1) -> Self {
        Self::configured(diagnostics, true)
    }

    fn unstarted() -> Self {
        Self::configured(SloAutomaticCalibrationDiagnosticsV1::MemoryOnly, false)
    }

    fn configured(diagnostics: SloAutomaticCalibrationDiagnosticsV1, start_live: bool) -> Self {
        Self::configured_with_input_readiness(diagnostics, start_live, Default::default())
    }

    fn configured_with_input_readiness(
        diagnostics: SloAutomaticCalibrationDiagnosticsV1,
        start_live: bool,
        input_readiness: ferrum_types::SloAutomaticCalibrationInputReadinessV1,
    ) -> Self {
        Self::configured_with_settings(
            SloAutomaticCalibrationSettingsV1 {
                discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
                phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
                diagnostics,
                input_readiness,
                ..Default::default()
            },
            start_live,
        )
    }

    fn configured_with_settings(
        settings: SloAutomaticCalibrationSettingsV1,
        start_live: bool,
    ) -> Self {
        let identity = identity();
        let ExecutorCostIdentityAvailability::Known(executor) = &identity else {
            panic!("fixture identity");
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
        let clock = Arc::new(VirtualClock(AtomicU64::new(1)));
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        assert_eq!(config.model.min_samples.get(), OFFERS);
        config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 { settings };
        let runtime = EngineCostRuntime::build_with_profile_and_domain(
            identity,
            clock.clone(),
            &config,
            false,
            None,
            None,
            Some(domain.clone()),
        )
        .unwrap();
        if start_live {
            runtime.begin_automatic_calibration().unwrap();
            runtime.consume_samples();
        }
        Self {
            runtime,
            clock,
            domain,
            recorded: 0,
        }
    }

    fn live(
        &self,
    ) -> &Arc<crate::continuous_engine::inner::cost_observation::live_calibration::LiveCalibration>
    {
        self.runtime.training.live.as_ref().unwrap()
    }

    fn wave(algorithm: &'static str) -> Wave {
        // The controlled selected-cost class is the only changing model input.
        // Both alternatives execute the same genuine two-participant CPU
        // submission fixture and use the original private host-settled proof.
        let host = wave(algorithm).host;
        wave_with_host_rows(algorithm, 7, host, 2)
    }

    fn query(&self, algorithm: &'static str) -> StructuredQueryV2 {
        let w = Self::wave(algorithm);
        StructuredQueryV2::from_future_with_domain(
            &w.prepared.exact,
            &w.prepared.selected,
            &w.prepared.recipe,
            &ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
            &self.domain,
        )
        .unwrap()
    }

    fn record(&mut self, algorithm: &'static str) {
        let w = Self::wave(algorithm);
        let expected = w.actual.rows.clone();
        let stages = fixture::record_cohort_route(&self.runtime, &self.clock, w)
            .expect("original CPU submission and host settlement");
        assert_eq!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert_eq!(stages.rows.len(), expected.len());
        assert!(stages
            .structured_evidence
            .as_ref()
            .is_some_and(Result::is_ok));
        for (row, settled) in expected.iter().zip(&stages.rows) {
            assert_eq!(settled.request_id, row.request_id);
            assert_eq!(settled.owner_incarnation, row.owner_incarnation);
            assert_eq!(settled.work_generation, row.work_generation);
            assert_eq!(settled.input_index, row.input_index);
        }
        self.runtime.consume_samples();
        self.recorded += 1;
        let sink = self.runtime.audit_snapshot().sink;
        assert_eq!(sink.raw_offered, self.recorded);
        assert_eq!(sink.raw_accepted, self.recorded);
        assert_eq!(sink.raw_resolved, self.recorded);
        assert_eq!(sink.raw_resolution_failed, 0);
        assert_eq!(sink.raw_lost, 0);
        assert_eq!(sink.raw_pending, 0);
    }

    fn block(&mut self, algorithm: &'static str) {
        // Deterministic tests have no background worker. This empty turn opens
        // the next block after a closed publication/failure without new offers.
        self.runtime.consume_samples();
        for _ in 0..OFFERS {
            self.record(algorithm);
        }
    }

    fn known(&self, algorithm: &'static str) -> bool {
        self.runtime.snapshot().is_some_and(|snapshot| {
            snapshot
                .audit_structured_query_v2(&self.query(algorithm), self.clock.now_ns().unwrap())
                .is_ok()
        })
    }

    fn assert_unpublished(&self) {
        assert_eq!(self.live().audit().qualified_publications, 0);
        assert!(!self.known(A));
        assert!(!self.known(B));
    }
}

#[tokio::test]
async fn automatic_family_progress_interleaved_complete_blocks_publish_both_families() {
    let mut f = Families::new();
    // A: discovery(1), Fit(2), Residual(5), Qualification(6).
    // B: discovery(3), Fit(4), Residual(7), Qualification(8).
    // The old global schedule repeatedly sees the other family in Residual;
    // its actual valid observations must not be blamed on missing evidence.
    for algorithm in [A, A, B, B, A] {
        f.block(algorithm);
        f.assert_unpublished();
    }
    f.block(A);
    let first = f.live().audit();
    assert_eq!(
        first.qualified_publications, 1,
        "A completed its own three independent phases despite intervening B blocks: {first:#?}"
    );
    assert!(f.known(A));
    let first_children = f
        .runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(first_children.len(), 1);
    let first_domain = *first_children[0].domain_signature();
    let first_provenance = first_children[0].provenance().clone();
    assert!(
        !f.known(B),
        "B still lacks independent Residual and Qualification"
    );
    f.block(B);
    assert_eq!(f.live().audit().qualified_publications, 1);
    assert!(f.known(A));
    assert!(!f.known(B), "Residual must not double as Qualification");
    f.block(B);
    let last = f.live().audit();
    assert_eq!(last.qualified_publications, 2, "{last:#?}");
    assert!(f.known(A));
    assert!(f.known(B));
    let children = f
        .runtime
        .training
        .live_catalog_children(f.clock.now_ns().unwrap())
        .unwrap();
    assert_eq!(children.len(), 2);
    let retained = children
        .iter()
        .find(|child| *child.domain_signature() == first_domain)
        .unwrap();
    assert_eq!(
        retained.provenance().file_sha256,
        first_provenance.file_sha256
    );
    assert_eq!(
        retained.provenance().source_sha256,
        first_provenance.source_sha256
    );
    assert_eq!(
        retained.provenance().parameters_sha256,
        first_provenance.parameters_sha256
    );
    assert_eq!(
        retained.provenance().loaded_unix_ns,
        first_provenance.loaded_unix_ns
    );
    assert_ne!(
        children[0].owner().algorithm_domain,
        children[1].owner().algorithm_domain
    );
    for child in children {
        assert_eq!(child.owner().rows, 2);
        assert_eq!(
            child.owner().cost_template_policy(),
            StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
        );
        assert_eq!(child.workload_domain(), Some(&f.domain));
    }
    assert_eq!(f.recorded, 8 * OFFERS as u64);
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_family_progress_missing_independent_qualification_cannot_publish() {
    let mut f = Families::new();
    for algorithm in [A, A, B, B, A] {
        f.block(algorithm);
        f.assert_unpublished();
    }
    // A has Discovery/Fit/Residual only. More unrelated B members cannot serve
    // as its Qualification; shutting down cannot convert a prefix to a model.
    f.block(B);
    f.assert_unpublished();
    f.runtime.shutdown().await.unwrap();
    f.assert_unpublished();
    assert!(f.runtime.training.published_catalog_receipt().is_none());
}

#[tokio::test]
async fn automatic_family_progress_lost_original_qualification_ticket_cannot_publish() {
    let mut f = Families::new();
    for algorithm in [A, A, B, B, A] {
        f.block(algorithm);
        f.assert_unpublished();
    }
    f.runtime.consume_samples();
    for _ in 0..OFFERS - 1 {
        f.record(A);
    }
    let failures = f.live().audit().failed_generations;
    let at = f.clock.now_ns().unwrap() + 100;
    f.clock.set(at);
    // Retire an original issued ticket without its execution observation. No
    // replacement offer or successful zero-duration sample is manufactured.
    drop(
        f.runtime
            .reserve_live_ticket(Some(at))
            .expect("eighth original offer"),
    );
    f.runtime.consume_samples();
    let failed = f.live().audit();
    assert!(failed.failed_generations > failures, "{failed:#?}");
    assert_eq!(failed.population.declared_offers, OFFERS);
    assert_eq!(failed.population.issued, OFFERS);
    assert_eq!(failed.population.retired, OFFERS);
    assert!(failed.population.failed);
    f.assert_unpublished();
    f.runtime.shutdown().await.unwrap();
    f.assert_unpublished();
    assert!(f.runtime.training.published_catalog_receipt().is_none());
}

#[path = "family_progress/consumer_proofs.rs"]
mod consumer_proofs;

#[tokio::test]
async fn global_residual_strategy_reaches_actual_source7_header_and_fit_target_stopping() {
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
        NonNegativePlanningEstimatorV1 as Estimator, OwnerInputReadinessV1,
    };
    use ferrum_types::{
        SloAutomaticCalibrationInputReadinessV1 as Readiness,
        SloAutomaticCalibrationNumericalStrategyV1 as Strategy,
    };
    let mut settings = SloAutomaticCalibrationSettingsV1 {
        discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
        phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
        ..Default::default()
    };
    let old = Families::configured_with_settings(settings.clone(), true)
        .runtime
        .automatic_original_header_for_test()
        .unwrap();
    settings.numerical_strategy = Strategy::IdentifiedFitGlobalResidualV1;
    settings.validate().unwrap();
    let Readiness::WorkAxesAndBranchesV3 {
        maximum_phase_blocks,
        maximum_geometry_visits,
    } = settings.input_readiness
    else {
        panic!("fixture uses actual automatic default geometry")
    };
    let expected = OwnerInputReadinessV1::new_fit_target_v4(
        maximum_phase_blocks.map(NonZeroUsize::get),
        maximum_geometry_visits.get(),
    )
    .unwrap();
    let current = Families::configured_with_settings(settings, true);
    let new = current
        .runtime
        .automatic_original_header_for_test()
        .unwrap();
    assert_eq!(
        new.declaration
            .nonnegative_envelope
            .as_ref()
            .unwrap()
            .planning_estimator,
        Estimator::IdentifiedFitGlobalResidualV1
    );
    assert_eq!(new.declaration.schedule.input_readiness, Some(expected));
    assert_ne!(
        new.declaration.schedule.input_readiness,
        old.declaration.schedule.input_readiness
    );
    assert_ne!(new.declaration_sha256, old.declaration_sha256);
    assert_eq!(
        serde_json::to_value(&new.declaration.settings).unwrap(),
        serde_json::to_value(&old.declaration.settings).unwrap()
    );
    assert_eq!(
        new.declaration.maximum_window_ns,
        old.declaration.maximum_window_ns
    );
    assert_eq!(
        new.declaration.maximum_retained_numeric_bytes,
        old.declaration.maximum_retained_numeric_bytes
    );
    assert_eq!(
        new.declaration.schedule.phase_support,
        old.declaration.schedule.phase_support
    );
    assert_eq!(
        new.declaration.route_population,
        old.declaration.route_population
    );
}

#[path = "family_progress/prospective_eos_feedback.rs"]
mod prospective_eos_feedback;
