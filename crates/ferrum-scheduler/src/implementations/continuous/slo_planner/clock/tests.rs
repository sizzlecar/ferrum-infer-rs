use super::super::super::cost_model::*;
use super::*;
use ferrum_types::SloLatencyBudgets;
use std::{cell::Cell, sync::Arc};

fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
}
fn ns(value: u64) -> Duration {
    Duration::from_nanos(value)
}
fn n32(value: u32) -> NonZeroU32 {
    NonZeroU32::new(value).unwrap()
}
fn n64(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}
fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}
fn state(ingress: Instant) -> RequestSloState {
    RequestSloState::new(
        ingress,
        "interactive".into(),
        SloLatencyBudgets {
            ttft_ms: n64(10),
            tpot_ms: n64(10),
            itl_ms: n64(20),
        },
    )
    .unwrap()
}

#[test]
fn origin_preserves_ingress_before_engine_start_and_roundtrips_real_instants() {
    let ingress = Instant::now();
    let engine_start = ingress + ms(50);
    let observed = ingress + ms(100);
    let first = state(ingress);
    let second = state(ingress + ms(70));
    let origin = PlanningTimeOrigin::from_ingress(observed, nz(2), [&first, &second]).unwrap();
    assert_eq!(origin.observed_at(), observed);
    assert_eq!(origin.observed_at_ns(), 100_000_000);
    assert_eq!(origin.at_ns(engine_start).unwrap(), 50_000_000);
    let projected = origin.project_request(&first, n32(8)).unwrap();
    assert_eq!(projected.ingress_at_ns, 0);
    assert_eq!(projected.next_deadline_ns(), Some(10_000_000));
    assert!(
        projected.slo_failed,
        "existing ingress wait must not be reset at engine start"
    );
    assert_eq!(
        origin
            .instant_at_ns(projected.next_deadline_ns().unwrap())
            .unwrap(),
        first.next_deadline().unwrap()
    );
    assert_eq!(
        origin.at_ns(ingress - ns(1)),
        Err(PlanningTimeError::BeforeOrigin)
    );
    let anchor = PlanningCostClockAnchor::exact(origin.at_ns(engine_start).unwrap(), 0);
    assert_eq!(
        anchor.cost_time_ns(origin.observed_at_ns()).unwrap(),
        50_000_000
    );
}

#[test]
fn ingress_iteration_is_bounded_without_truncating_obligations() {
    let at = Instant::now();
    let request = state(at);
    let visited = Cell::new(0);
    let requests = std::iter::from_fn(|| {
        visited.set(visited.get() + 1);
        Some(&request)
    });
    assert!(matches!(
        PlanningTimeOrigin::from_ingress(at, nz(3), requests),
        Err(PlanningTimeError::RequestCapacity)
    ));
    assert_eq!(visited.get(), 4);
    let empty = PlanningTimeOrigin::from_ingress(at, nz(1), []).unwrap();
    assert_eq!(empty.observed_at_ns(), 0);
}

#[test]
fn timing_projection_keeps_endpoint_deadlines_elapsed_and_sticky_violations() {
    let at = Instant::now();
    let mut request = state(at);
    request.record_commit(at + ms(5)).unwrap();
    request.record_commit(at + ms(10)).unwrap();
    request.record_commit(at + ms(25)).unwrap();
    let origin = PlanningTimeOrigin::from_ingress(at + ms(36), nz(1), [&request]).unwrap();
    let timing = origin.project_request(&request, n32(8)).unwrap();
    assert_eq!(timing.committed_tokens, 3);
    assert_eq!(timing.first_commit_at_ns, Some(5_000_000));
    assert_eq!(timing.last_commit_at_ns, Some(25_000_000));
    assert_eq!(timing.next_deadline_ns(), Some(35_000_000));
    assert!(
        timing.slo_failed,
        "unobserved blocked wait past prefix deadline is projected"
    );
    assert!(!request.violations().any(), "projection is read-only");
    request.observe_wait(at + ms(36)).unwrap();
    request.record_commit(at + ms(37)).unwrap();
    let next = PlanningTimeOrigin::from_ingress(at + ms(38), nz(1), [&request]).unwrap();
    assert!(
        next.project_request(&request, n32(8)).unwrap().slo_failed,
        "later progress cannot erase sticky failure"
    );
}

#[test]
fn completed_request_gets_no_new_wait_violation_but_keeps_historical_failure() {
    let at = Instant::now();
    let mut request = state(at);
    request.record_commit(at + ms(5)).unwrap();
    let origin = PlanningTimeOrigin::from_ingress(at + ms(1000), nz(1), [&request]).unwrap();
    let timing = origin.project_request(&request, n32(1)).unwrap();
    assert!(timing.completed());
    assert!(!timing.slo_failed);
    let mut late = state(at);
    late.record_commit(at + ms(11)).unwrap();
    assert!(origin.project_request(&late, n32(1)).unwrap().slo_failed);
}

#[test]
fn future_untrusted_hidden_observation_and_inconsistent_token_limits_are_rejected() {
    let at = Instant::now();
    let origin = PlanningTimeOrigin::from_origin(at, at + ms(10)).unwrap();
    let future = state(at + ms(11));
    assert!(matches!(
        PlanningTimeOrigin::from_ingress(at + ms(10), nz(1), [&future]),
        Err(PlanningTimeError::FutureRequestTiming)
    ));
    let mut future_commit = state(at);
    future_commit.record_commit(at + ms(11)).unwrap();
    assert_eq!(
        origin.project_request(&future_commit, n32(2)),
        Err(PlanningTimeError::FutureRequestTiming)
    );
    let mut hidden = state(at);
    hidden.observe_wait(at + ms(20)).unwrap();
    assert_eq!(
        origin.project_request(&hidden, n32(2)),
        Err(PlanningTimeError::InconsistentRequestTiming)
    );
    let mut invalid = state(at);
    invalid.record_commit(at + ms(5)).unwrap();
    assert!(invalid.record_commit(at + ms(4)).is_err());
    assert_eq!(
        origin.project_request(&invalid, n32(2)),
        Err(PlanningTimeError::UntrustedRequest)
    );
    let mut too_many = state(at);
    too_many.record_commit(at + ms(1)).unwrap();
    too_many.record_commit(at + ms(2)).unwrap();
    assert_eq!(
        origin.project_request(&too_many, n32(1)),
        Err(PlanningTimeError::InconsistentRequestTiming)
    );
    let too_late_origin = PlanningTimeOrigin::from_origin(at + ms(1), at + ms(10)).unwrap();
    assert_eq!(
        too_late_origin.project_request(&state(at), n32(1)),
        Err(PlanningTimeError::BeforeOrigin)
    );
}

#[test]
fn instant_conversion_rejects_out_of_range_nanoseconds_instead_of_saturating() {
    let at = Instant::now();
    let origin = PlanningTimeOrigin::from_origin(at, at).unwrap();
    let huge = Duration::from_secs(u64::MAX / 1_000_000_000 + 1);
    if let Some(later) = at.checked_add(huge) {
        assert_eq!(origin.at_ns(later), Err(PlanningTimeError::TimeOverflow));
        assert!(matches!(
            PlanningTimeOrigin::from_origin(at, later),
            Err(PlanningTimeError::TimeOverflow)
        ));
    } else {
        // Some platform Instant domains reject this duration before conversion.
        assert!(at.checked_add(huge).is_none());
    }
    assert!(matches!(
        PlanningTimeOrigin::from_origin(at + ns(1), at),
        Err(PlanningTimeError::BeforeOrigin)
    ));
}

fn fingerprint() -> ExecutionFingerprint {
    ExecutionFingerprint {
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }
}
fn shape() -> WaveExecutionShape {
    WaveExecutionShape {
        row_multiset_features: None,
        host_content_features: None,
        numeric_features: None,
        kind: WaveKind::Decode,
        path: WaveExecutionPath::PlanRuntime,
        provider_signature: [5; 32],
        output_policy_signature: [6; 32],
        graph_state: WaveGraphState::Disabled,
        order: BatchOrderSemantics::Ordered,
        decode_kv_tokens: vec![128],
        prefill_chunks: vec![],
        recurrent_state_bytes: 0,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}
fn model() -> Arc<CostModelSnapshot> {
    let mut trainer = CostModelTrainer::new(
        fingerprint(),
        CostModelSettings {
            min_samples: nz(1),
            max_sample_age_ns: n64(100),
            ..Default::default()
        },
    )
    .unwrap();
    trainer
        .observe(WaveCostObservation {
            fingerprint: fingerprint(),
            actual_shape: shape(),
            boundary: CostBoundary::PreparationToCommit,
            outcome: WaveObservationOutcome::Completed,
            timing: WaveTiming {
                wall_total_ns: 10,
                device_elapsed_ns: None,
                stages: Default::default(),
            },
            observed_at_ns: 10,
        })
        .unwrap();
    trainer.publish(10).unwrap()
}

#[test]
fn independent_epoch_adapter_preserves_real_model_ttl_and_unknown_shape() {
    let model = model();
    let anchor = PlanningCostClockAnchor::exact(1000, 40);
    let adapter = AnchoredPlanningCostModel::new(model.as_ref(), anchor);
    assert_eq!(adapter.model_version(), model.model_version());
    assert_eq!(
        adapter
            .predict(&fingerprint(), &shape(), 1000)
            .unwrap()
            .valid_for_ns,
        70
    );
    assert_eq!(
        adapter
            .predict(&fingerprint(), &shape(), 1070)
            .unwrap()
            .valid_for_ns,
        0
    );
    assert!(adapter.predict(&fingerprint(), &shape(), 1071).is_none());
    assert!(adapter.predict(&fingerprint(), &shape(), 999).is_none());
    let mut unknown = shape();
    unknown.provider_signature[0] ^= 1;
    assert!(adapter.predict(&fingerprint(), &unknown, 1000).is_none());
}

#[test]
fn bracketed_read_can_only_age_samples_and_rejects_queries_before_read_completion() {
    let at = Instant::now();
    let origin = PlanningTimeOrigin::from_origin(at, at + ns(1000)).unwrap();
    let anchor =
        PlanningCostClockAnchor::conservative_read(&origin, at + ns(1000), 40, at + ns(1020))
            .unwrap();
    assert_eq!(anchor.acquisition_skew_ns(), 20);
    assert_eq!(
        anchor.cost_time_ns(1019),
        Err(PlanningTimeError::BeforeCostAnchor)
    );
    // Cost read may occur anywhere in [1000,1020]. At the query, the mapped
    // cost time upper-bounds every synchronized possibility in that bracket.
    for actual_read_ns in [1000, 1007, 1020] {
        let synchronized_cost = 40 + (1070 - actual_read_ns);
        assert!(anchor.cost_time_ns(1070).unwrap() >= synchronized_cost);
    }
    let model = model();
    let adapter = AnchoredPlanningCostModel::new(model.as_ref(), anchor);
    assert_eq!(
        adapter
            .predict(&fingerprint(), &shape(), 1020)
            .unwrap()
            .valid_for_ns,
        50
    );
    assert_eq!(
        adapter
            .predict(&fingerprint(), &shape(), 1070)
            .unwrap()
            .valid_for_ns,
        0
    );
    assert!(adapter.predict(&fingerprint(), &shape(), 1071).is_none());
    assert_eq!(
        PlanningCostClockAnchor::conservative_read(&origin, at + ns(2), 0, at + ns(1)),
        Err(PlanningTimeError::ReversedCostBracket)
    );
}

#[test]
fn cost_clock_overflow_cannot_turn_into_fresh_or_zero_cost_prediction() {
    let anchor = PlanningCostClockAnchor::exact(5, u64::MAX);
    assert_eq!(anchor.cost_time_ns(5).unwrap(), u64::MAX);
    assert_eq!(anchor.cost_time_ns(6), Err(PlanningTimeError::TimeOverflow));
    assert_eq!(
        anchor.cost_time_ns(4),
        Err(PlanningTimeError::BeforeCostAnchor)
    );
    let model = model();
    assert!(AnchoredPlanningCostModel::new(model.as_ref(), anchor)
        .predict(&fingerprint(), &shape(), 6)
        .is_none());
}
