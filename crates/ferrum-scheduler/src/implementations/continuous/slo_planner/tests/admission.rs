use super::super::time_admission::*;
use super::*;
use ferrum_types::{SloAdmissionConfig, SloTimeAdmissionPolicy};
use std::cell::Cell;

mod clock;

// A finite KV reservoir with persistent growth across all waves. This models
// the resource-resolver contract; it does not substitute for backend tests.
struct KvCapacity(u64);
impl PlanningResourceResolver for KvCapacity {
    fn begin(
        &self,
        _: &SchedulerSnapshot,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<Box<dyn PlanningResourceProjection + '_>, PlanningUnknownReason> {
        poll()?;
        Ok(Box::new(KvProjection(self.0)))
    }
}
struct KvProjection(u64);
impl PlanningResourceProjection for KvProjection {
    fn apply(
        &mut self,
        query: &PlanningResourceQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<(), PlanningUnknownReason> {
        let mut growth = 0;
        for work in &query.wave.work {
            poll()?;
            let request = query
                .requests
                .iter()
                .find(|request| request.key == work.key)
                .unwrap();
            growth += match work.action {
                WaveAction::Decode => 1,
                WaveAction::Prefill { offset, count } => {
                    u64::from((offset + count.get()).saturating_sub(request.context_tokens))
                }
            };
        }
        self.0 = self
            .0
            .checked_sub(growth)
            .ok_or(PlanningUnknownReason::OutputOrResourceBlocked)?;
        Ok(())
    }
}

fn assess(
    snapshot: &SchedulerSnapshot,
    policy: &SloAdmissionConfig,
    boundary: TimeAdmissionBoundary,
    model: &dyn PlanningCostModel,
    clock: &mut dyn PlanningClock,
) -> TimeAdmissionDecision {
    let active: Vec<_> = snapshot
        .requests
        .iter()
        .skip(1)
        .filter(|request| !request.timing.completed())
        .map(|request| request.key.clone())
        .collect();
    TimeAdmissionEvaluator {
        policy,
        planner: &planner(4),
        model,
        shapes: &TestResolver,
        resources: &KvCapacity(1024),
    }
    .assess(
        TimeAdmissionQuery {
            snapshot,
            target: &snapshot.requests[0].key,
            active: &active,
            boundary,
        },
        clock,
    )
}

#[test]
fn admission_witness_must_cover_the_new_decoder_not_only_its_first_token() {
    let mut target = prefill(1);
    target.timing.maximum_output_tokens = n32(3);
    let snapshot = snapshot(vec![target]);
    let before = snapshot.clone();
    let no_decode_cost =
        Model(|shape: &WaveExecutionShape| (!shape.prefill_chunks.is_empty()).then_some(5));
    feasible(planner(4).propose_with_resources(
        &snapshot,
        &no_decode_cost,
        &TestResolver,
        &KvCapacity(1024),
        &mut Clock(100),
    ));
    assert!(matches!(
        assess(
            &snapshot,
            &SloAdmissionConfig::default(),
            TimeAdmissionBoundary::BeforeAcceptance,
            &no_decode_cost,
            &mut Clock(100)
        ),
        TimeAdmissionDecision::Unknown {
            reason: TimeAdmissionUnknown::Planning(PlanningUnknownReason::CostUnavailable),
            continuation: TimeAdmissionContinuation::BestEffort,
            ..
        }
    ));
    let TimeAdmissionDecision::Admit { witness, .. } = assess(
        &snapshot,
        &SloAdmissionConfig::default(),
        TimeAdmissionBoundary::BeforeAcceptance,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &mut Clock(100),
    ) else {
        panic!("known next decode should have a shared witness");
    };
    assert!(witness.waves >= 2);
    assert!(witness.predicted_output_tokens >= 2);
    assert_eq!(witness.validated_through_ns, snapshot.scope.horizon_end_ns);
    assert_eq!(
        snapshot, before,
        "forecasting must preserve ingress, output limit, peers and resources"
    );
}

#[test]
fn individually_feasible_requests_do_not_gain_two_incompatible_promises() {
    let mut left = prefill(1);
    let mut right = prefill(2);
    left.timing.budgets.ttft_ns = n64(120);
    right.timing.budgets.ttft_ns = n64(120);
    let model = Model(|_: &WaveExecutionShape| Some(15));
    let policy = SloAdmissionConfig {
        time_policy: SloTimeAdmissionPolicy::RequireSlo,
        ..Default::default()
    };
    for request in [&left, &right] {
        assert!(matches!(
            assess(
                &snapshot(vec![request.clone()]),
                &policy,
                TimeAdmissionBoundary::BeforeAcceptance,
                &model,
                &mut Clock(100)
            ),
            TimeAdmissionDecision::Admit { .. }
        ));
    }
    let mut together = snapshot(vec![left, right]);
    together.capabilities.prefill_chunk_sizes = vec![n32(8)];
    assert!(matches!(
        assess(
            &together,
            &policy,
            TimeAdmissionBoundary::BeforeAcceptance,
            &model,
            &mut Clock(100)
        ),
        TimeAdmissionDecision::Unknown {
            continuation: TimeAdmissionContinuation::WaitForEvidence,
            ..
        }
    ));
}

#[test]
fn late_peer_is_retained_and_does_not_block_completion_first_progress() {
    let mut peer = decode(2);
    peer.timing.slo_failed = true;
    let snapshot = snapshot(vec![prefill(1), peer]);
    let before = snapshot.clone();
    let model = Model(|_: &WaveExecutionShape| Some(5));
    let mut policy = SloAdmissionConfig::default();
    assert!(matches!(
        assess(
            &snapshot,
            &policy,
            TimeAdmissionBoundary::BeforeAcceptance,
            &model,
            &mut Clock(100)
        ),
        TimeAdmissionDecision::BestEffort {
            reason: PlanningImpossibleReason::HistoricalViolation { .. }
        }
    ));
    policy.time_policy = SloTimeAdmissionPolicy::RequireSlo;
    assert!(matches!(
        assess(
            &snapshot,
            &policy,
            TimeAdmissionBoundary::BeforeAcceptance,
            &model,
            &mut Clock(100)
        ),
        TimeAdmissionDecision::Defer {
            reason: TimeAdmissionDeferReason::ExistingObligationAtRisk,
            ..
        }
    ));
    assert!(matches!(
        assess(
            &snapshot,
            &policy,
            TimeAdmissionBoundary::Accepted,
            &model,
            &mut Clock(100)
        ),
        TimeAdmissionDecision::BestEffort { .. }
    ));
    assert_eq!(snapshot, before);
}

#[test]
fn unknown_cost_and_coverage_do_not_authorize_default_rejection() {
    let snapshot = snapshot(vec![prefill(1)]);
    let no_cost = Model(|_: &WaveExecutionShape| None);
    let mut policy = SloAdmissionConfig::default();
    for boundary in [
        TimeAdmissionBoundary::BeforeAcceptance,
        TimeAdmissionBoundary::Accepted,
    ] {
        assert!(matches!(
            assess(&snapshot, &policy, boundary, &no_cost, &mut Clock(100)),
            TimeAdmissionDecision::Unknown {
                continuation: TimeAdmissionContinuation::BestEffort,
                ..
            }
        ));
    }
    policy.max_sequence_tokens = nz(4);
    assert!(matches!(
        assess(
            &snapshot,
            &policy,
            TimeAdmissionBoundary::BeforeAcceptance,
            &no_cost,
            &mut Clock(100)
        ),
        TimeAdmissionDecision::Unknown {
            reason: TimeAdmissionUnknown::OutsideSequenceCoverage,
            continuation: TimeAdmissionContinuation::BestEffort,
            ..
        }
    ));
    policy.time_policy = SloTimeAdmissionPolicy::RequireSlo;
    assert!(matches!(
        assess(
            &snapshot,
            &policy,
            TimeAdmissionBoundary::BeforeAcceptance,
            &no_cost,
            &mut Clock(100)
        ),
        TimeAdmissionDecision::Unknown {
            continuation: TimeAdmissionContinuation::WaitForEvidence,
            ..
        }
    ));
}

fn at_expiry() -> SchedulerSnapshot {
    let mut target = prefill(1);
    target.timing.budgets.ttft_ns = n64(2_000_000);
    let mut snapshot = snapshot(vec![target]);
    snapshot.observed_at_ns = 1_000_000;
    snapshot.scope.horizon_end_ns = 1_100_000;
    snapshot
}

#[test]
fn original_ingress_expiry_only_rejects_explicit_strict_unaccepted_work() {
    let snapshot = at_expiry();
    let model = Model(|_: &WaveExecutionShape| None);
    let mut policy = SloAdmissionConfig {
        max_wait_ms: n64(1),
        ..Default::default()
    };
    for (time_policy, boundary, reject) in [
        (
            SloTimeAdmissionPolicy::CompleteRequests,
            TimeAdmissionBoundary::BeforeAcceptance,
            false,
        ),
        (
            SloTimeAdmissionPolicy::RequireSlo,
            TimeAdmissionBoundary::Accepted,
            false,
        ),
        (
            SloTimeAdmissionPolicy::RequireSlo,
            TimeAdmissionBoundary::BeforeAcceptance,
            true,
        ),
    ] {
        policy.time_policy = time_policy;
        let decision = assess(&snapshot, &policy, boundary, &model, &mut Clock(1_000_000));
        if reject {
            assert!(matches!(
                decision,
                TimeAdmissionDecision::Reject {
                    reason: TimeAdmissionRejectReason::StrictWaitExpired {
                        expiry_at_ns: 1_000_000
                    }
                }
            ));
        } else {
            let TimeAdmissionDecision::Unknown {
                continuation, wait, ..
            } = decision
            else {
                panic!("missing cost must remain unknown");
            };
            assert_eq!(continuation, TimeAdmissionContinuation::BestEffort);
            assert_eq!(wait.review_at_ns, None);
            assert_eq!(wait.strict_expiry_at_ns, None);
        }
    }
}

#[test]
fn strict_expiry_is_rechecked_after_real_search_and_forecast_unknown_keeps_its_wake() {
    let mut snapshot = at_expiry();
    snapshot.observed_at_ns = 999_990;
    let policy = SloAdmissionConfig {
        time_policy: SloTimeAdmissionPolicy::RequireSlo,
        max_wait_ms: n64(1),
        ..Default::default()
    };
    let calls = Cell::new(0);
    let model = Model(|_: &WaveExecutionShape| {
        calls.set(calls.get() + 1);
        Some(5)
    });
    let result = assess(
        &snapshot,
        &policy,
        TimeAdmissionBoundary::BeforeAcceptance,
        &model,
        &mut ScriptClock::new(&[999_990, 1_000_001]),
    );
    assert!(
        calls.get() > 0,
        "expiry crossing must occur through the search path"
    );
    assert!(matches!(
        result,
        TimeAdmissionDecision::Reject {
            reason: TimeAdmissionRejectReason::StrictWaitExpired {
                expiry_at_ns: 1_000_000
            }
        }
    ));
    snapshot.has_unmodeled_maintenance = true;
    let TimeAdmissionDecision::Unknown { wait, .. } = assess(
        &snapshot,
        &policy,
        TimeAdmissionBoundary::BeforeAcceptance,
        &model,
        &mut Clock(999_990),
    ) else {
        panic!("maintenance remains unknown");
    };
    assert_eq!(wait.strict_expiry_at_ns, Some(1_000_000));
    assert_eq!(wait.review_at_ns, Some(1_000_000));
}

#[test]
fn active_limit_defers_without_rewriting_or_discarding_the_waiting_request() {
    let snapshot = snapshot(vec![prefill(1), decode(2)]);
    let before = snapshot.clone();
    let policy = SloAdmissionConfig {
        max_active_requests: nz(1),
        ..Default::default()
    };
    let model = Model(|_: &WaveExecutionShape| -> Option<u64> {
        panic!("a full active set needs a genuine capacity wake before a new forecast");
    });
    assert!(matches!(
        assess(
            &snapshot,
            &policy,
            TimeAdmissionBoundary::Accepted,
            &model,
            &mut Clock(100)
        ),
        TimeAdmissionDecision::Defer {
            reason: TimeAdmissionDeferReason::ActiveLimit,
            ..
        }
    ));
    assert_eq!(snapshot, before);
}

#[test]
fn strict_admission_witness_cannot_be_reused_after_wait_expiry() {
    let mut snapshot = at_expiry();
    snapshot.observed_at_ns = 999_990;
    let policy = SloAdmissionConfig {
        time_policy: SloTimeAdmissionPolicy::RequireSlo,
        max_wait_ms: n64(1),
        ..Default::default()
    };
    let TimeAdmissionDecision::Admit { first_wave, .. } = assess(
        &snapshot,
        &policy,
        TimeAdmissionBoundary::BeforeAcceptance,
        &Model(|_: &WaveExecutionShape| Some(5)),
        &mut Clock(999_990),
    ) else {
        panic!("request can be accepted before its independent wait expiry");
    };
    assert!(first_wave.planning_observed_at_ns + first_wave.witness_valid_for_ns < 1_000_000);
}

#[test]
fn clock_reversal_during_prediction_cannot_produce_an_admission_or_safe_fallback() {
    use std::{cell::RefCell, rc::Rc};
    struct MutableClock {
        script: Rc<RefCell<VecDeque<u64>>>,
        last: u64,
    }
    impl PlanningClock for MutableClock {
        fn now_ns(&mut self) -> u64 {
            if let Some(next) = self.script.borrow_mut().pop_front() {
                self.last = next;
            }
            self.last
        }
    }
    let script = Rc::new(RefCell::new(VecDeque::new()));
    let injected = Cell::new(false);
    let model = Model(|_: &WaveExecutionShape| {
        if !injected.replace(true) {
            script.borrow_mut().extend([200, 150]);
        }
        Some(5)
    });
    let mut snapshot = snapshot(vec![prefill(1)]);
    snapshot.requests[0].timing.budgets.ttft_ns = n64(1_000);
    snapshot.scope.horizon_end_ns = 900;
    let mut clock = MutableClock {
        script: Rc::clone(&script),
        last: 100,
    };
    assert!(matches!(
        assess(
            &snapshot,
            &SloAdmissionConfig::default(),
            TimeAdmissionBoundary::Accepted,
            &model,
            &mut clock
        ),
        TimeAdmissionDecision::Unknown {
            reason: TimeAdmissionUnknown::Planning(PlanningUnknownReason::ClockMovedBackwards),
            continuation: TimeAdmissionContinuation::WaitForEvidence,
            ..
        }
    ));
    assert!(injected.get());
}

#[test]
fn time_policy_cannot_turn_resource_unknown_or_bad_identity_into_a_permit() {
    let snapshot = snapshot(vec![prefill(1)]);
    let model = Model(|_: &WaveExecutionShape| Some(5));
    let evaluator = TimeAdmissionEvaluator {
        policy: &SloAdmissionConfig::default(),
        planner: &planner(4),
        model: &model,
        shapes: &TestResolver,
        resources: &KvCapacity(0),
    };
    let result = evaluator.assess(
        TimeAdmissionQuery {
            snapshot: &snapshot,
            target: &snapshot.requests[0].key,
            active: &[],
            boundary: TimeAdmissionBoundary::Accepted,
        },
        &mut Clock(100),
    );
    assert!(matches!(
        result,
        TimeAdmissionDecision::Unknown {
            continuation: TimeAdmissionContinuation::WaitForEvidence,
            ..
        }
    ));
    let mut stale = snapshot.requests[0].key.clone();
    stale.incarnation += 1;
    assert!(matches!(
        evaluator.assess(
            TimeAdmissionQuery {
                snapshot: &snapshot,
                target: &stale,
                active: &[],
                boundary: TimeAdmissionBoundary::BeforeAcceptance
            },
            &mut Clock(100)
        ),
        TimeAdmissionDecision::Unknown {
            reason: TimeAdmissionUnknown::Planning(PlanningUnknownReason::InvalidSnapshot),
            continuation: TimeAdmissionContinuation::WaitForEvidence,
            ..
        }
    ));
}
