use super::*;
use std::time::{Duration, Instant};

fn long_window() -> SchedulerSnapshot {
    let mut target = prefill(1);
    target.timing.budgets.ttft_ns = n64(1_000_000);
    let mut snapshot = snapshot(vec![target]);
    snapshot.scope.horizon_end_ns = 1_000_000;
    snapshot
}

#[test]
fn admission_origin_bridge_preserves_original_request_and_finite_common_witness() {
    let snapshot = long_window();
    let at = Instant::now();
    let observed = at + Duration::from_nanos(snapshot.observed_at_ns);
    let origin = PlanningTimeOrigin::from_origin(at, observed).unwrap();
    let policy = SloAdmissionConfig::default();
    let planner = planner(4);
    let model = Model(|_: &WaveExecutionShape| Some(5));
    let evaluator = TimeAdmissionEvaluator {
        policy: &policy,
        planner: &planner,
        model: &model,
        shapes: &TestResolver,
        resources: &KvCapacity(1024),
    };
    let before = snapshot.clone();
    let result = origin
        .assess_admission_with_deadline(
            &evaluator,
            TimeAdmissionQuery {
                snapshot: &snapshot,
                target: &snapshot.requests[0].key,
                active: &[],
                boundary: TimeAdmissionBoundary::Accepted,
            },
            observed + Duration::from_micros(10),
            || observed,
        )
        .unwrap();
    let TimeAdmissionDecision::Admit {
        first_wave,
        witness,
        ..
    } = result
    else {
        panic!("known fixture should retain its finite admission witness: {result:?}");
    };
    assert_eq!(first_wave.snapshot_observed_at_ns, snapshot.observed_at_ns);
    assert_eq!(witness.validated_through_ns, snapshot.scope.horizon_end_ns);
    assert_eq!(snapshot, before);
}

#[test]
fn admission_cannot_restart_budget_after_capture_or_policy_checks() {
    let snapshot = long_window();
    let at = Instant::now();
    let observed = at + Duration::from_nanos(snapshot.observed_at_ns);
    let origin = PlanningTimeOrigin::from_origin(at, observed).unwrap();
    let policy = SloAdmissionConfig::default();
    let planner = planner(4);
    let model = Model(|_: &WaveExecutionShape| Some(5));
    let evaluator = TimeAdmissionEvaluator {
        policy: &policy,
        planner: &planner,
        model: &model,
        shapes: &TestResolver,
        resources: &KvCapacity(1024),
    };
    let reads = Cell::new(0);
    let deadline = observed + Duration::from_micros(2);
    let result = origin
        .assess_admission_with_deadline(
            &evaluator,
            TimeAdmissionQuery {
                snapshot: &snapshot,
                target: &snapshot.requests[0].key,
                active: &[],
                boundary: TimeAdmissionBoundary::Accepted,
            },
            deadline,
            || {
                let read = reads.get();
                reads.set(read + 1);
                if read < 2 {
                    observed + Duration::from_nanos(1500)
                } else {
                    deadline
                }
            },
        )
        .unwrap();
    assert!(matches!(
        result,
        TimeAdmissionDecision::Unknown {
            reason: TimeAdmissionUnknown::Planning(PlanningUnknownReason::ComputeBudgetExhausted),
            ..
        }
    ));
    assert!(reads.get() >= 3);
}

#[test]
fn admission_clock_reversal_cannot_turn_an_unknown_into_an_admit_or_reject() {
    let snapshot = long_window();
    let at = Instant::now();
    let observed = at + Duration::from_nanos(snapshot.observed_at_ns);
    let origin = PlanningTimeOrigin::from_origin(at, observed).unwrap();
    let policy = SloAdmissionConfig::default();
    let planner = planner(4);
    let model = Model(|_: &WaveExecutionShape| Some(5));
    let evaluator = TimeAdmissionEvaluator {
        policy: &policy,
        planner: &planner,
        model: &model,
        shapes: &TestResolver,
        resources: &KvCapacity(1024),
    };
    let reads = Cell::new(0);
    let result = origin.assess_admission_with_deadline(
        &evaluator,
        TimeAdmissionQuery {
            snapshot: &snapshot,
            target: &snapshot.requests[0].key,
            active: &[],
            boundary: TimeAdmissionBoundary::Accepted,
        },
        observed + Duration::from_micros(10),
        || {
            let read = reads.get();
            reads.set(read + 1);
            if read == 0 {
                observed + Duration::from_nanos(10)
            } else {
                observed
            }
        },
    );
    assert!(matches!(
        result,
        Err(PlanningTimeError::ClockMovedBackwards)
    ));
}

#[test]
fn explicit_outer_window_does_not_restart_admission_after_capture() {
    let mut snapshot = long_window();
    snapshot.observed_at_ns = 700;
    let at = Instant::now();
    let origin = PlanningTimeOrigin::from_origin(at, at + Duration::from_nanos(700)).unwrap();
    let policy = SloAdmissionConfig::default();
    let planner = planner(4);
    let predictions = std::cell::Cell::new(0);
    let model = Model(|_: &WaveExecutionShape| {
        predictions.set(predictions.get() + 1);
        Some(5)
    });
    let evaluator = TimeAdmissionEvaluator {
        policy: &policy,
        planner: &planner,
        model: &model,
        shapes: &TestResolver,
        resources: &KvCapacity(1024),
    };
    let result = origin
        .assess_admission_with_budget_window(
            &evaluator,
            TimeAdmissionQuery {
                snapshot: &snapshot,
                target: &snapshot.requests[0].key,
                active: &[],
                boundary: TimeAdmissionBoundary::Accepted,
            },
            PlanningBudgetWindow {
                started_at_ns: 0,
                deadline_ns: 1_000,
            },
            || at + Duration::from_nanos(800),
        )
        .unwrap();
    assert!(
        matches!(
            result,
            TimeAdmissionDecision::Unknown {
                reason: TimeAdmissionUnknown::Planning(
                    PlanningUnknownReason::ComputeBudgetExhausted
                ),
                ..
            }
        ),
        "{result:?}"
    );
    assert_eq!(predictions.get(), 0);
}

/// The first successful final-prefill cost lookup completes this fixture's
/// only one-token target. Decision work then crosses the outer soft deadline.
/// A partial -> final prefix is a required search path before that event, not
/// "optional" merely because its physical query has nonempty prior_waves.
struct AdmissionPhaseResolver<'a> {
    now: &'a Cell<u64>,
    completed_predictions: &'a Cell<usize>,
    replay_queries: std::cell::RefCell<Vec<(usize, u32, u32)>>,
    replay_at: Option<u64>,
}

impl PlanningShapeResolver for AdmissionPhaseResolver<'_> {
    fn resolve(
        &self,
        query: &PlanningShapeQuery<'_>,
        poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    ) -> Result<
        Option<ferrum_interfaces::execution_cost::CanonicalWaveCostShape>,
        PlanningUnknownReason,
    > {
        use ferrum_interfaces::execution_cost::ActualRowWork;
        if self.completed_predictions.get() > 0 {
            let mut replay = self.replay_queries.borrow_mut();
            if replay.is_empty() {
                if let Some(now) = self.replay_at {
                    self.now.set(now);
                }
            }
            assert_eq!(query.rows.len(), 1);
            let ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } = query.rows[0].work
            else {
                panic!("the one-token target has no decode work");
            };
            assert_eq!(total_prompt_tokens, 8);
            replay.push((query.prior_waves.len(), offset, count));
        }
        TestResolver.resolve(query, poll)
    }
}

fn admission_with_capture_and_optional_branch(
    replay_at: Option<u64>,
) -> (TimeAdmissionDecision, Vec<(usize, u32, u32)>, usize) {
    let snapshot = long_window();
    assert_eq!(snapshot.observed_at_ns, 100);
    let at = Instant::now();
    let origin = PlanningTimeOrigin::from_origin(at, at + Duration::from_nanos(100)).unwrap();
    let now = Cell::new(100);
    let completed_predictions = Cell::new(0);
    let shapes = AdmissionPhaseResolver {
        now: &now,
        completed_predictions: &completed_predictions,
        replay_queries: std::cell::RefCell::new(Vec::new()),
        replay_at,
    };
    let mut planner = planner(2);
    planner.settings.search.max_planning_us = n64(1);
    let policy = SloAdmissionConfig::default();
    let model = Model(|shape: &WaveExecutionShape| {
        if shape.prefill_chunks.iter().any(|chunk| {
            chunk.offset.checked_add(chunk.count.get()) == Some(chunk.total_prompt_tokens.get())
        }) {
            let completed = completed_predictions.get();
            completed_predictions.set(completed + 1);
            if completed == 0 {
                now.set(650);
            }
        }
        Some(5)
    });
    let evaluator = TimeAdmissionEvaluator {
        policy: &policy,
        planner: &planner,
        model: &model,
        shapes: &shapes,
        resources: &KvCapacity(1024),
    };
    let result = origin
        .assess_admission_with_budget_window(
            &evaluator,
            TimeAdmissionQuery {
                snapshot: &snapshot,
                target: &snapshot.requests[0].key,
                active: &[],
                boundary: TimeAdmissionBoundary::Accepted,
            },
            // Capture consumed [0,100]. Search/replay end at 600/800, not
            // snapshot-relative 700/900; publication still retains [800,1000].
            PlanningBudgetWindow {
                started_at_ns: 0,
                deadline_ns: 1_000,
            },
            || at + Duration::from_nanos(now.get()),
        )
        .unwrap();
    (
        result,
        shapes.replay_queries.into_inner(),
        completed_predictions.get(),
    )
}

#[test]
fn admission_preserves_outer_soft_deadline_and_replays_the_complete_witness() {
    let (result, replay_queries, completed_predictions) =
        admission_with_capture_and_optional_branch(None);
    let TimeAdmissionDecision::Admit {
        first_wave,
        witness,
        search,
    } = result
    else {
        panic!("the complete witness fits the original replay window: {result:?}");
    };
    assert_eq!(search.search_soft_stops, 1);
    assert_eq!(
        replay_queries.len(),
        witness.waves,
        "replay every wave exactly once"
    );
    let mut completed_prompt = 0;
    for (index, &(prior_waves, offset, count)) in replay_queries.iter().enumerate() {
        assert_eq!(prior_waves, index, "replay retains the complete prefix");
        assert_eq!(
            offset, completed_prompt,
            "replay starts at the original frontier"
        );
        completed_prompt += count;
    }
    assert_eq!(completed_prompt, 8);
    assert_eq!(witness.predicted_output_tokens, 1);
    assert_eq!(first_wave.planning_observed_at_ns, 650);
    assert_eq!(
        completed_predictions, 2,
        "initial complete witness and final replay"
    );
}

#[test]
fn admission_original_replay_deadline_never_returns_the_saved_witness() {
    for replay_at in [800, 1_000, 1_200] {
        let (result, replay_queries, completed_predictions) =
            admission_with_capture_and_optional_branch(Some(replay_at));
        assert!(
            matches!(
                result,
                TimeAdmissionDecision::Unknown {
                    reason: TimeAdmissionUnknown::Planning(
                        PlanningUnknownReason::ComputeBudgetExhausted
                    ),
                    ..
                }
            ),
            "{replay_at}: {result:?}"
        );
        assert_eq!(
            replay_queries.len(),
            1,
            "replay was entered but stopped at its original deadline"
        );
        assert_eq!(replay_queries[0].0, 0);
        assert_eq!(replay_queries[0].1, 0);
        assert_eq!(
            completed_predictions, 1,
            "the expired replay cannot reach another cost lookup"
        );
    }
}
