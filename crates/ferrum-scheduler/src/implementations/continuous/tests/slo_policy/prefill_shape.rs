//! Query ranges are product planning facts, never checkpoint/capacity authority.
use super::*;
use ferrum_runtime::{CheckpointBoundaryConstraint, CheckpointTokenSpanConstraint};
use std::num::NonZeroU64;

pub(super) fn tail(reserved: u64) -> PrefillShapePolicy {
    PrefillShapePolicy::PromptTail(
        CheckpointBoundaryConstraint::new(
            CheckpointTokenSpanConstraint::any_positive(),
            CheckpointTokenSpanConstraint::new(
                NonZeroU64::new(reserved).unwrap(),
                NonZeroU64::new(1).unwrap(),
            )
            .unwrap(),
        )
        .unwrap(),
    )
}

fn actual_tokens(batch: &BatchPlan, id: &RequestId) -> usize {
    batch
        .requests
        .iter()
        .find(|request| &request.request.id == id)
        .unwrap()
        .tokens_to_process
        .unwrap()
}

#[test]
fn slo_prefill_shape_candidate_ends_and_selected_suffix_use_one_projection() {
    for reserve in [1, 2] {
        let sched = scheduler(None);
        let id = waiting(&sched, 8);
        let policy = tail(reserve);
        let boundary = 8 - reserve as usize;
        let source = snapshot(11, Instant::now(), false);
        let mut planning = TimePlanningRound::new(
            source.clone(),
            targets(),
            policy,
            CostExecution::SingleInvocation,
        );
        admit(&sched, Some(&mut planning));
        // Admission already prices the first actually executable range.
        assert!(source
            .calls
            .lock()
            .iter()
            .all(|(_, work)| work[0].requested.end() == boundary));
        {
            let queue = sched.prefill_queue.read();
            for candidate in [1, boundary, boundary + 1, 8] {
                let (work, exact) = sched
                    .cost_work(&queue[0], candidate, false, policy)
                    .unwrap();
                assert!(exact);
                assert_eq!(work.requested.as_range(), 0..candidate.min(boundary));
                assert_eq!(work.context_tokens, 8);
            }
        }
        let batch = select(&sched, Some(&mut planning)).batch.unwrap();
        assert_eq!(actual_tokens(&batch, &id), boundary);
        assert_eq!(planning.selected[0].requested.as_range(), 0..boundary);
        assert_eq!(
            planning.statistics.shrunk_prefills, 0,
            "checkpoint partitioning is not time adaptation"
        );
        assert!(!sched.mark_prefill_chunk_processed(&id, 8, boundary));
        let mut suffix = TimePlanningRound::new(
            source.clone(),
            targets(),
            policy,
            CostExecution::SingleInvocation,
        );
        let batch = select(&sched, Some(&mut suffix)).batch.unwrap();
        assert_eq!(actual_tokens(&batch, &id), reserve as usize);
        assert_eq!(suffix.selected[0].requested.as_range(), boundary..8);
        assert_eq!(suffix.selected[0].context_tokens, 8);
        assert_eq!(suffix.statistics.shrunk_prefills, 0);
        assert!(sched.mark_prefill_chunk_processed(&id, 8, reserve as usize));
        assert_eq!(sched.decoding_count(), 1);
        assert!(sched.cancel_request(id).unwrap());
        assert_eq!(sched.active_count(), 0);
    }
}

#[test]
fn slo_prefill_shape_time_shrink_starts_from_projected_work() {
    let sched = scheduler(None);
    let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
    let request =
        create_test_request_with_prompt_tokens(Priority::Normal, 100).with_ingress(ingress);
    let id = request.id.clone();
    enqueue_waiting(&sched, request);
    admit(&sched, None);
    let source = snapshot(1, ingress.instant(), false);
    let mut config = targets();
    config.ttft_ms = 60.0;
    let mut planning = TimePlanningRound::new(
        source.clone(),
        config,
        tail(20),
        CostExecution::SingleInvocation,
    );
    let batch = sched
        .create_iteration_batch_with_time(
            hint(),
            WaitingAdmissionMode::Prepared {
                initial_priority: None,
                wake: None,
                observer: &mut |_| {},
            },
            ingress.instant(),
            Some(&mut planning),
        )
        .unwrap()
        .unwrap();
    // Actual candidate 80 costs 90.25ms. The next query is 40 (50.25ms),
    // never nominal 100 -> 50 or repeated queries of the same tail boundary.
    assert_eq!(actual_tokens(&batch, &id), 40);
    assert_eq!(planning.selected[0].requested.as_range(), 0..40);
    assert_eq!(planning.statistics.shrunk_prefills, 1);
    assert_eq!(planning.statistics.budget_misses, 1);
    let calls = source.calls.lock();
    assert!(calls
        .iter()
        .all(|(_, work)| matches!(work[0].requested.end(), 80 | 40)));
    assert!(calls.iter().any(|(_, work)| work[0].requested.end() == 80));
    assert!(calls.iter().any(|(_, work)| work[0].requested.end() == 40));
}

struct ExactDomain {
    at: Instant,
    work: CostWork,
    calls: Mutex<Vec<CostWork>>,
}
impl CostSnapshot for ExactDomain {
    fn query_limits(&self) -> ferrum_types::CostQueryLimits {
        ferrum_types::CostQueryLimits::default()
    }
    fn revision(&self) -> u64 {
        21
    }
    fn predict(&self, query: &CostQuery<'_>, _now: Instant) -> CostPrediction {
        assert!(query.work_is_exact);
        assert_eq!(query.work.len(), 1);
        self.calls.lock().push(query.work[0]);
        if query.work == [self.work] {
            CostPrediction::Known {
                central_service_time: Duration::from_micros(1),
                upper_service_time: Duration::from_micros(1),
                risk: CostRisk::EngineeringMargin,
                support: CostSupport::ExactObservedWork,
                samples: 2,
                observed_at: self.at,
                expires_at: self.at + Duration::from_secs(10),
            }
        } else {
            CostPrediction::Unknown(CostUnknownReason::NoCoverage)
        }
    }
}

#[test]
fn slo_prefill_shape_replay_domain_cannot_price_different_prompt_range() {
    let sched = ContinuousBatchScheduler::new(SchedulerConfig {
        prefill_step_chunk: Some(4),
        slo: Some(targets()),
        ..SchedulerConfig::default()
    });
    let policy = tail(2);
    // A confirmed replay context may legally execute 0..4/context5 without
    // reserving a prompt tail. This is the historical empirical domain.
    let mut replay =
        ContinuousBatchRequest::new(create_test_request_with_prompt_tokens(Priority::Normal, 4));
    replay.prefill_tokens = 5;
    replay.capacity_deferred_from_decode = true;
    replay.slo_first_token_emitted = true;
    let (old_domain, exact) = sched.cost_work(&replay, 4, false, policy).unwrap();
    assert!(exact);
    assert_eq!(old_domain.requested.as_range(), 0..4);
    assert_eq!(old_domain.context_tokens, 5);
    let source = Arc::new(ExactDomain {
        at: Instant::now(),
        work: old_domain,
        calls: Mutex::new(Vec::new()),
    });
    let id = waiting(&sched, 5);
    admit(&sched, None);
    let mut planning = TimePlanningRound::new(
        source.clone(),
        SchedulerSloConfig {
            unknown_cost: UnknownCostPolicy::Reject,
            ..targets()
        },
        policy,
        CostExecution::SingleInvocation,
    );
    assert!(select(&sched, Some(&mut planning)).batch.is_none());
    let calls = source.calls.lock();
    assert!(!calls.is_empty());
    assert!(calls
        .iter()
        .all(|work| work.requested.as_range() == (0..3) && work.context_tokens == 5));
    assert_eq!(planning.statistics.known_queries, 0);
    let failures = sched.take_admission_failures();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].0, id);
    assert!(matches!(
        &failures[0].1,
        FerrumError::RequestValidation { .. }
    ));
}

#[test]
fn slo_prefill_shape_restore_and_same_id_replacement_refresh_scope() {
    let sched = scheduler(None);
    let original = create_test_request_with_prompt_tokens(Priority::Normal, 8);
    let id = original.id.clone();
    enqueue_waiting(&sched, original.clone());
    let source = snapshot(9, Instant::now(), false);
    let mut planning = TimePlanningRound::new(
        source.clone(),
        targets(),
        tail(2),
        CostExecution::SingleInvocation,
    );
    admit(&sched, Some(&mut planning));
    assert_eq!(
        source.calls.lock().last().unwrap().1[0]
            .requested
            .as_range(),
        0..6
    );
    let prepared = sched.prepare_prefix_restore(&id, 0, 8).unwrap().unwrap();
    sched.commit_prefix_restored(prepared, 4).unwrap();
    let batch = select(&sched, Some(&mut planning)).batch.unwrap();
    assert_eq!(actual_tokens(&batch, &id), 2);
    assert_eq!(planning.selected[0].requested.as_range(), 4..6);
    assert_eq!(
        planning.selected[0].prefix_restore,
        AttemptPrefixRestore::Restored { tokens: 4 }
    );
    assert_eq!(planning.revision(), 9);
    assert!(sched.cancel_request(id.clone()).unwrap());

    // Same public RequestId, fresh queue incarnation. No shape plan or restore
    // fact from the cancelled request survives in the immutable-cost round.
    let replacement = original.with_metadata(PROMPT_TOKENS_METADATA_KEY, serde_json::json!(5));
    sched.submit_request(replacement).unwrap();
    admit(&sched, Some(&mut planning));
    let batch = select(&sched, Some(&mut planning)).batch.unwrap();
    assert_eq!(actual_tokens(&batch, &id), 3);
    assert_eq!(planning.selected[0].requested.as_range(), 0..3);
    assert_eq!(planning.selected[0].context_tokens, 5);
    assert_eq!(
        planning.selected[0].prefix_restore,
        AttemptPrefixRestore::NoneObserved
    );
    assert!(source
        .calls
        .lock()
        .iter()
        .all(|(revision, _)| *revision == 9));
    assert!(sched.cancel_request(id).unwrap());
}

#[test]
fn slo_prefill_shape_unknown_preserves_policy_and_existing_decode_progress() {
    for unknown_cost in [
        UnknownCostPolicy::StaticFallback,
        UnknownCostPolicy::Reject,
        UnknownCostPolicy::Wait {
            max_wait_ms: 1000,
            on_timeout: CostFallback::Reject,
        },
    ] {
        let sched = scheduler(None);
        let decode = activate_decode_requests(&sched, 1).remove(0);
        let id = waiting(&sched, 8);
        admit(&sched, None);
        let source = snapshot(1, Instant::now(), false);
        let mut planning = TimePlanningRound::new(
            source.clone(),
            SchedulerSloConfig {
                unknown_cost,
                ..targets()
            },
            PrefillShapePolicy::Unknown,
            CostExecution::SingleInvocation,
        );
        let selection = select(&sched, Some(&mut planning));
        let batch = selection.batch.unwrap();
        assert_eq!(actual_tokens(&batch, &decode), 1);
        assert!(
            source.calls.lock().is_empty(),
            "unknown shape must not look up an exact empirical domain"
        );
        match unknown_cost {
            UnknownCostPolicy::StaticFallback => assert_eq!(actual_tokens(&batch, &id), 8),
            UnknownCostPolicy::Reject => {
                assert_eq!(batch.requests.len(), 1);
                assert_eq!(sched.take_admission_failures()[0].0, id);
            }
            UnknownCostPolicy::Wait { .. } => {
                assert_eq!(batch.requests.len(), 1);
                assert!(selection.retry_at.is_some());
                assert!(selection.time_blocked);
            }
        }
    }
}

#[test]
fn slo_prefill_shape_estimated_context_never_projects_a_checkpoint() {
    let sched = scheduler(None);
    let request = ContinuousBatchRequest::new(create_test_request(Priority::Normal));
    let candidate = sched.initial_prefill_token_estimate(&request);
    let (work, exact) = sched
        .cost_work(&request, candidate, false, tail(2))
        .unwrap();
    assert!(!exact);
    assert_eq!(work.requested.as_range(), 0..candidate);
    // Unknown policy stays unknown even when replay identity is otherwise
    // known; absence of a qualification is not inferred from phase alone.
    let mut replay =
        ContinuousBatchRequest::new(create_test_request_with_prompt_tokens(Priority::Normal, 8));
    replay.prefill_tokens = 9;
    replay.capacity_deferred_from_decode = true;
    let (work, exact) = sched
        .cost_work(&replay, 4, false, PrefillShapePolicy::Unknown)
        .unwrap();
    assert!(!exact);
    assert_eq!(work.requested.as_range(), 0..4);
}
