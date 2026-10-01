//! A real CPU cohort changes width while the installed algorithm and finite
//! physical descriptor remain fixed. Private route/host receipts come from the
//! original submission and settlement APIs, never constructed wire evidence.
use super::*;
use std::num::NonZeroUsize;

#[path = "shadow_batch/domain_guard.rs"]
mod domain_guard;

const ALGORITHM: &str = "fixture.batch-width";
const OFFERS: usize = 8;

struct Cohort {
    runtime: EngineCostRuntime,
    clock: Arc<VirtualClock>,
    domain: CostWorkloadDomainV1,
}

impl Cohort {
    fn new() -> Self {
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
        // Declare the complete populations before observing any input. Keep
        // all original numerical, age, feedback and retention defaults.
        config.live_structured_calibration = SloLiveStructuredCalibration::AutomaticV1 {
            settings: SloAutomaticCalibrationSettingsV1 {
                // This fixture verifies the original source6 fixed-population protocol.
                population_schedule:
                    ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1,
                discovery_offered_waves: NonZeroUsize::new(OFFERS).unwrap(),
                phase_offered_waves: [NonZeroUsize::new(OFFERS).unwrap(); 3],
                ..Default::default()
            },
        };
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
        runtime.begin_automatic_calibration().unwrap();
        runtime.consume_samples();
        Self {
            runtime,
            clock,
            domain,
        }
    }

    fn live(
        &self,
    ) -> &Arc<crate::continuous_engine::inner::cost_observation::live_calibration::LiveCalibration>
    {
        self.runtime.training.live.as_ref().unwrap()
    }

    fn wave(rows: u32) -> Wave {
        let host = wave(ALGORITHM).host;
        wave_with_host_rows(ALGORITHM, 7, host, rows)
    }

    fn query(&self, rows: u32) -> StructuredQueryV2 {
        let w = Self::wave(rows);
        StructuredQueryV2::from_future_with_domain(
            &w.prepared.exact,
            &w.prepared.selected,
            &w.prepared.recipe,
            &ferrum_interfaces::execution_cost::HostContentForecastV2::Exact,
            &self.domain,
        )
        .unwrap()
    }

    fn record(&self, rows: u32) {
        let w = Self::wave(rows);
        let expected = w.actual.rows.clone();
        let stages = fixture::record_cohort_route(&self.runtime, &self.clock, w)
            .expect("original completed physical cohort");
        assert_eq!(
            stages.completeness,
            HostStageCompleteness::CompleteSingleWave
        );
        assert_eq!(stages.rows.len(), rows as usize);
        assert!(stages
            .structured_evidence
            .as_ref()
            .is_some_and(Result::is_ok));
        for (actual, settled) in expected.iter().zip(&stages.rows) {
            assert_eq!(settled.request_id, actual.request_id);
            assert_eq!(settled.owner_incarnation, actual.owner_incarnation);
            assert_eq!(settled.work_generation, actual.work_generation);
            assert_eq!(settled.input_index, actual.input_index);
            assert_eq!(
                settled.completeness,
                HostStageCompleteness::CompleteSingleWave
            );
            assert!(settled.terminal.is_some());
        }
        self.runtime.consume_samples();
    }

    fn phase(&self, rows: u32) {
        for _ in 0..OFFERS {
            self.record(rows);
        }
    }

    fn assert_physical_catalog(&self, count: usize) {
        let children = self
            .runtime
            .training
            .live_catalog_children(self.clock.now_ns().unwrap())
            .unwrap();
        assert_eq!(children.len(), count);
        for child in children {
            assert_eq!(child.workload_domain(), Some(&self.domain));
            assert_eq!(
                child.owner().cost_template_policy(),
                StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
            );
        }
    }

    fn open_shadow(&self, previous_generation: u64) -> u64 {
        let failed = self.live().audit();
        assert_eq!(failed.population.generation, previous_generation);
        assert_eq!(failed.population.phase, 1);
        assert_eq!(
            (failed.population.issued, failed.population.retired),
            (OFFERS, OFFERS)
        );
        // A full, genuinely recorded but outside-owner Residual population can
        // discover the next owner. It cannot supply that owner's Fit samples.
        self.runtime.consume_samples();
        let next = self.live().audit();
        assert_eq!(
            (next.population.generation, next.population.phase),
            (previous_generation + 1, 0)
        );
        assert_eq!((next.population.issued, next.population.retired), (0, 0));
        let json = serde_json::to_value(next).unwrap();
        let origin = &json["automatic"]["discovery_origin"];
        assert_eq!(origin["generation"], previous_generation);
        assert_eq!(origin["phase"], 1);
        assert_eq!(origin["population"]["issued"], OFFERS);
        assert_eq!(origin["population"]["retired"], OFFERS);
        assert_eq!(origin["population"]["failed"], false);
        origin["closing_fifo_cutoff"].as_u64().unwrap()
    }
}

#[tokio::test]
async fn automatic_physical_batch_width_shadow_requires_fresh_three_phases() {
    let f = Cohort::new();
    for _ in 0..4 {
        f.phase(2);
    }
    assert_eq!(f.live().audit().qualified_publications, 1);
    assert_eq!(f.live().audit().failed_generations, 0);
    f.assert_physical_catalog(1);
    let old = f.runtime.snapshot().unwrap();
    old.audit_structured_query_v2(&f.query(2), f.clock.now_ns().unwrap())
        .unwrap();
    assert!(old
        .audit_structured_query_v2(&f.query(1), f.clock.now_ns().unwrap())
        .is_err());
    assert!(
        old.undeclared_structured_owner(&f.query(1)),
        "the one-row query is valid in the same physical domain, but not the old family"
    );

    f.runtime.consume_samples();
    f.phase(2); // New generation's independent discovery.
    f.phase(2); // Its original Fit is viable.
    let feedback_before = f.runtime.audit_snapshot().structured_feedback.unwrap();
    f.phase(1); // The entire original Residual offers a different row family.
    let failed = f.live().audit();
    assert_eq!(failed.failed_generations, 1);
    assert_eq!(failed.qualified_publications, 1);
    let feedback_after = f.runtime.audit_snapshot().structured_feedback.unwrap();
    assert_eq!(
        feedback_after.outside_catalog_observations,
        feedback_before.outside_catalog_observations + OFFERS as u64
    );
    assert_eq!(
        feedback_after.uncomparable_observations,
        feedback_before.uncomparable_observations
    );
    assert!(feedback_after.revoked.is_none());
    assert!(old.current());
    let cutoff = f.open_shadow(2);
    assert!(cutoff > 0);

    for phase in 0..3 {
        f.phase(1);
        let audit = f.live().audit();
        assert_eq!(audit.failed_generations, 1, "{audit:#?}");
        assert_eq!(audit.qualified_publications, 1 + u64::from(phase == 2));
        if phase < 2 {
            assert_eq!(audit.population.phase, phase + 1);
            assert_eq!((audit.population.issued, audit.population.retired), (0, 0));
            assert!(
                f.runtime
                    .snapshot()
                    .unwrap()
                    .audit_structured_query_v2(&f.query(1), f.clock.now_ns().unwrap())
                    .is_err(),
                "shadow discovery and partial numerical phases cannot publish"
            );
        }
    }
    let receipt = f.runtime.training.published_catalog_receipt().unwrap();
    assert_eq!(receipt.offered_samples, 3 * OFFERS);
    assert_eq!(receipt.recorded_samples, 3 * OFFERS);
    assert_eq!(receipt.storage, SloCostProfileStorage::Memory);
    f.assert_physical_catalog(2);
    let next = f.runtime.snapshot().unwrap();
    for rows in [2, 1] {
        let prediction = next
            .audit_structured_query_v2(&f.query(rows), f.clock.now_ns().unwrap())
            .unwrap();
        assert!(prediction.planning_ns > 0);
    }
    f.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn automatic_physical_batch_width_short_shadow_fit_shutdown_never_publishes() {
    let f = Cohort::new();
    f.phase(2);
    f.phase(2);
    f.phase(1);
    assert_eq!(f.live().audit().failed_generations, 1);
    assert_eq!(f.live().audit().qualified_publications, 0);
    f.open_shadow(1);
    for _ in 0..OFFERS - 1 {
        f.record(1);
    }
    let before = f.live().audit();
    assert_eq!(before.population.phase, 0);
    assert_eq!(
        (before.population.issued, before.population.retired),
        (OFFERS - 1, OFFERS - 1)
    );
    assert_eq!(before.qualified_publications, 0);
    assert!(f.runtime.snapshot().is_none());
    f.runtime.shutdown().await.unwrap();
    let after = f.live().audit();
    assert_eq!(after.qualified_publications, 0);
    assert_eq!(after.failed_generations, 2);
    assert!(after.population.closed && after.population.failed);
    assert_eq!(after.population.issued, OFFERS - 1);
    assert_eq!(after.population.retired, after.population.issued);
    assert!(f.runtime.snapshot().is_none());
    assert!(f.runtime.training.published_catalog_receipt().is_none());
}
