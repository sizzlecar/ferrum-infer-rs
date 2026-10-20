use super::*;
use crate::cost_prediction::{
    CostPrediction, CostQuery, CostSnapshot, CostUnknownReason, CostWork, TimePlanningRound,
};
use ferrum_execution::execution_attempt::{AttemptPrefixRestore, AttemptWorkKind};
use ferrum_execution::model_executor::PrefillShapePolicy;

mod prefill_shape;
use ferrum_types::{CostFallback, RequestIngressAnchor, SchedulerSloConfig, UnknownCostPolicy};

struct Snapshot {
    revision: u64,
    at: Instant,
    unknown: bool,
    minimum_covered_prefill: usize,
    calls: Mutex<Vec<(u64, Vec<CostWork>)>>,
}

impl CostSnapshot for Snapshot {
    fn revision(&self) -> u64 {
        self.revision
    }
    fn predict(&self, query: &CostQuery<'_>, _now: Instant) -> CostPrediction {
        assert!(query.work_is_exact);
        self.calls.lock().push((self.revision, query.work.to_vec()));
        if self.unknown
            || query.work.iter().any(|work| {
                work.kind == AttemptWorkKind::Prefill
                    && work.requested.as_range().len() < self.minimum_covered_prefill
            })
        {
            return CostPrediction::Unknown(CostUnknownReason::NoCoverage);
        }
        // A declared immutable fixture, not a second fitting/training path.
        let micros = 10_250
            + query
                .work
                .iter()
                .map(|work| match work.kind {
                    AttemptWorkKind::Prefill => work.requested.as_range().len() as u64 * 1000,
                    AttemptWorkKind::Decode => 9_500,
                })
                .sum::<u64>();
        CostPrediction::Known {
            upper_service_time: Duration::from_micros(micros),
            samples: 2,
            observed_at: self.at,
            expires_at: self.at + Duration::from_secs(10),
        }
    }
}

fn snapshot(revision: u64, at: Instant, unknown: bool) -> Arc<Snapshot> {
    Arc::new(Snapshot {
        revision,
        at,
        unknown,
        minimum_covered_prefill: 0,
        calls: Mutex::new(Vec::new()),
    })
}

fn targets() -> SchedulerSloConfig {
    SchedulerSloConfig {
        ttft_ms: 1000.0,
        tpot_ms: 100.0,
        itl_ms: 120.0,
        reference: Default::default(),
        unknown_cost: Default::default(),
    }
}

fn scheduler(static_budget: Option<usize>) -> ContinuousBatchScheduler {
    ContinuousBatchScheduler::new(SchedulerConfig {
        max_running_requests: 8,
        active_decode_prefill_token_budget: static_budget,
        slo: Some(targets()),
        ..SchedulerConfig::default()
    })
}

fn hint() -> BatchHint {
    BatchHint {
        max_batch_size: 8,
        max_tokens: 256,
        target_latency_ms: None,
        available_memory: None,
        resource_constraints: Default::default(),
    }
}

fn waiting(scheduler: &ContinuousBatchScheduler, tokens: usize) -> RequestId {
    let request = create_test_request_with_prompt_tokens(Priority::Normal, tokens);
    let id = request.id.clone();
    enqueue_waiting(scheduler, request);
    id
}

fn admit(scheduler: &ContinuousBatchScheduler, mut planning: Option<&mut TimePlanningRound>) {
    let mut round = scheduler.begin_token_admission(8, None).unwrap();
    while let Some(candidate) = round.next(&mut |_| {}, planning.as_deref_mut()).unwrap() {
        let request_id = candidate.request().id.clone();
        assert!(matches!(
            round.apply(
                candidate,
                AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt { request_id })
            ),
            TokenAdmissionApply::Applied { accepted: Some(_) }
        ));
    }
    round.finish().unwrap();
}

fn select(
    scheduler: &ContinuousBatchScheduler,
    planning: Option<&mut TimePlanningRound>,
) -> crate::cost_prediction::BatchSelection {
    scheduler
        .next_batch_with_token_admissions(
            scheduler.prepare_token_batch_selection(hint()),
            None,
            &mut |_| {},
            planning,
        )
        .unwrap()
}

#[test]
fn slo_snapshot_budget_changes_real_batch_and_respects_static_caps() {
    for static_budget in [None, Some(13)] {
        let sched = scheduler(static_budget);
        let decode = activate_decode_requests(&sched, 1);
        let first = waiting(&sched, 100);
        waiting(&sched, 100);
        admit(&sched, None);
        let cold = select(&sched, None).batch.unwrap();
        assert_eq!(
            cold.resource_requirements.gpu_memory,
            if static_budget.is_some() { 14 } else { 201 } * 16
        );
        let source = snapshot(7, Instant::now(), false);
        let mut planning =
            TimePlanningRound::new(source.clone(), targets(), PrefillShapePolicy::Identity);
        let adapted = select(&sched, Some(&mut planning)).batch.unwrap();
        assert_eq!(
            adapted.resource_requirements.gpu_memory,
            if static_budget.is_some() { 14 } else { 101 } * 16
        );
        assert!(adapted
            .requests
            .iter()
            .any(|request| request.request.id == first));
        assert_eq!(
            adapted
                .requests
                .iter()
                .filter(|request| decode.contains(&request.request.id))
                .count(),
            1
        );
        let stats = sched.slo_snapshot().unwrap();
        assert_eq!(stats.last_revision, Some(7));
        assert_eq!(stats.planning_rounds, 1);
        assert!(stats.selections.known_queries > 0);
        let mut narrow_round =
            TimePlanningRound::new(source, targets(), PrefillShapePolicy::Identity);
        let narrow = sched
            .next_batch_with_token_admissions(
                sched.prepare_token_batch_selection(BatchHint {
                    max_tokens: 7,
                    ..hint()
                }),
                None,
                &mut |_| {},
                Some(&mut narrow_round),
            )
            .unwrap()
            .batch
            .unwrap();
        assert!(narrow.resource_requirements.gpu_memory <= 7 * 16);
    }
}

#[test]
fn slo_one_snapshot_survives_admission_subrounds_and_final_selection() {
    let sched = scheduler(None);
    let now = Instant::now();
    let old = snapshot(3, now, false);
    let published = snapshot(4, now, true);
    let mut planning = TimePlanningRound::new(old.clone(), targets(), PrefillShapePolicy::Identity);
    waiting(&sched, 32);
    admit(&sched, Some(&mut planning));
    // A source can publish here, but the already-owned planning round must
    // retain its original immutable snapshot through another admission await.
    waiting(&sched, 64);
    admit(&sched, Some(&mut planning));
    assert_eq!(
        select(&sched, Some(&mut planning))
            .batch
            .unwrap()
            .requests
            .len(),
        2
    );
    assert!(old.calls.lock().iter().all(|(revision, _)| *revision == 3));
    assert!(published.calls.lock().is_empty());
    let mut next =
        TimePlanningRound::new(published.clone(), targets(), PrefillShapePolicy::Identity);
    assert_eq!(
        select(&sched, Some(&mut next))
            .batch
            .unwrap()
            .requests
            .len(),
        2
    );
    assert!(!published.calls.lock().is_empty());
    assert_eq!(sched.slo_snapshot().unwrap().last_revision, Some(4));
}

#[test]
fn slo_unknown_wait_uses_original_ingress_and_never_renews_deadline() {
    for on_timeout in [CostFallback::StaticFallback, CostFallback::Reject] {
        let sched = scheduler(None);
        let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::HttpHandler);
        let request =
            create_test_request_with_prompt_tokens(Priority::Normal, 32).with_ingress(ingress);
        let id = request.id.clone();
        enqueue_waiting(&sched, request);
        let config = SchedulerSloConfig {
            unknown_cost: UnknownCostPolicy::Wait {
                max_wait_ms: 100,
                on_timeout,
            },
            ..targets()
        };
        let source = snapshot(1, ingress.instant(), true);
        let epoch = sched.confirmed_capacity_release_epoch();
        for elapsed in [10, 75] {
            let mut planning =
                TimePlanningRound::new(source.clone(), config, PrefillShapePolicy::Identity);
            let mut round = sched.begin_token_admission(1, None).unwrap();
            assert!(round
                .next_at(
                    &mut |_| {},
                    Some(&mut planning),
                    ingress.instant() + Duration::from_millis(elapsed)
                )
                .unwrap()
                .is_none());
            let receipt = round.finish().unwrap();
            assert_eq!(receipt.probed(), 0);
            assert_eq!(receipt.deferred(), 0);
            assert_eq!(
                planning.retry_at(),
                Some(ingress.instant() + Duration::from_millis(100))
            );
            assert!(planning.time_blocked());
            assert_eq!(sched.waiting_count(), 1);
            assert_eq!(sched.confirmed_capacity_release_epoch(), epoch);
        }
        let mut planning = TimePlanningRound::new(source, config, PrefillShapePolicy::Identity);
        let mut round = sched.begin_token_admission(1, None).unwrap();
        let candidate = round
            .next_at(
                &mut |_| {},
                Some(&mut planning),
                ingress.instant() + Duration::from_millis(100),
            )
            .unwrap();
        match on_timeout {
            CostFallback::StaticFallback => {
                let candidate =
                    candidate.expect("finite wait must expire into declared static fallback");
                assert!(matches!(
                    round.apply(
                        candidate,
                        AdmissionProbeOutcome::Admitted(ExecutorPrefillAdmissionReceipt {
                            request_id: id.clone()
                        })
                    ),
                    TokenAdmissionApply::Applied { accepted: Some(_) }
                ));
            }
            CostFallback::Reject => {
                assert!(candidate.is_none());
                let failures = sched.take_admission_failures();
                assert_eq!(failures.len(), 1);
                assert_eq!(failures[0].0, id);
                assert!(matches!(
                    &failures[0].1,
                    FerrumError::RequestValidation { .. }
                ));
            }
        }
        round.finish().unwrap();
        assert_eq!(planning.retry_at(), None);
        assert_eq!(sched.confirmed_capacity_release_epoch(), epoch);
    }
}

#[test]
fn slo_unknown_reject_uses_original_failure_without_executor_probe() {
    let sched = scheduler(None);
    let id = waiting(&sched, 32);
    let mut planning = TimePlanningRound::new(
        snapshot(1, Instant::now(), true),
        SchedulerSloConfig {
            unknown_cost: UnknownCostPolicy::Reject,
            ..targets()
        },
        PrefillShapePolicy::Identity,
    );
    let mut round = sched.begin_token_admission(1, None).unwrap();
    assert!(round
        .next(&mut |_| {}, Some(&mut planning))
        .unwrap()
        .is_none());
    assert_eq!(round.finish().unwrap().probed(), 1);
    let failures = sched.take_admission_failures();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].0, id);
    assert!(matches!(
        &failures[0].1,
        FerrumError::RequestValidation { .. }
    ));
    assert_eq!(sched.trace_phase(&id), Some(RequestPhase::AdmissionFailed));
    assert_eq!(sched.prefilling_count(), 0);
    assert!(sched.take_admission_failures().is_empty());
}

#[test]
fn slo_stale_and_inexact_are_unknown_while_existing_decode_progresses() {
    let sched = scheduler(None);
    let decode = activate_decode_requests(&sched, 1);
    waiting(&sched, 32);
    admit(&sched, None);
    let at = Instant::now();
    let mut planning = TimePlanningRound::new(
        snapshot(1, at, false),
        SchedulerSloConfig {
            unknown_cost: UnknownCostPolicy::Reject,
            ..targets()
        },
        PrefillShapePolicy::Identity,
    );
    let batch = sched
        .create_iteration_batch_with_time(
            hint(),
            WaitingAdmissionMode::Prepared {
                initial_priority: None,
                wake: None,
                observer: &mut |_| {},
            },
            at + Duration::from_secs(11),
            Some(&mut planning),
        )
        .unwrap()
        .unwrap();
    assert_eq!(batch.requests.len(), 1);
    assert_eq!(batch.requests[0].request.id, decode[0]);
    assert_eq!(sched.take_admission_failures().len(), 1);
    assert!(planning.statistics.unknown_queries > 0);

    let inexact_sched = scheduler(None);
    enqueue_waiting(&inexact_sched, create_test_request(Priority::Normal));
    admit(&inexact_sched, None);
    let source = snapshot(2, Instant::now(), false);
    let mut inexact =
        TimePlanningRound::new(source.clone(), targets(), PrefillShapePolicy::Identity);
    assert!(select(&inexact_sched, Some(&mut inexact)).batch.is_some());
    assert!(source.calls.lock().is_empty());
    assert!(inexact.statistics.static_fallbacks > 0);
}

#[test]
fn slo_query_limit_is_checked_before_projection_and_preserves_decode() {
    let sched = scheduler(None);
    activate_decode_requests(&sched, 2);
    waiting(&sched, 32);
    admit(&sched, None);
    let source = snapshot(1, Instant::now(), false);
    let mut config = targets();
    config.reference.max_participants = 1;
    let mut planning = TimePlanningRound::new(source.clone(), config, PrefillShapePolicy::Identity);
    assert_eq!(
        select(&sched, Some(&mut planning))
            .batch
            .unwrap()
            .requests
            .len(),
        3
    );
    assert!(planning.selected.len() <= 1);
    assert!(planning.selected_over_limit);
    assert!(planning.statistics.static_fallbacks > 0);
    assert!(source.calls.lock().iter().all(|(_, work)| work.len() <= 1));
}

#[test]
fn slo_projection_matches_prefill_and_decode_attempt_contexts_including_recompute() {
    let sched = scheduler(None);
    let id = waiting(&sched, 128);
    admit(&sched, None);
    let prepared = sched.prepare_prefix_restore(&id, 0, 128).unwrap().unwrap();
    sched.commit_prefix_restored(prepared, 120).unwrap();
    {
        let queue = sched.prefill_queue.read();
        let (work, exact) = sched
            .cost_work(&queue[0], 8, false, PrefillShapePolicy::Identity)
            .unwrap();
        assert!(exact);
        assert_eq!(
            (
                work.requested.start(),
                work.requested.as_range().len(),
                work.context_tokens
            ),
            (120, 8, 128)
        );
        assert_eq!(
            work.prefix_restore,
            AttemptPrefixRestore::Restored { tokens: 120 }
        );
        assert_eq!(work.internally_reused_tokens, None);
    }
    sched.mark_prefill_complete(&id, 128);
    // Engine's successful prefill commits/samples the first output without
    // update_decode_progress; prefill_context_len is input + generated count.
    {
        let queue = sched.decode_queue.read();
        let (work, exact) = sched
            .cost_work(&queue.requests[&id], 1, true, PrefillShapePolicy::Identity)
            .unwrap();
        assert!(exact);
        assert_eq!(
            (
                work.requested.start(),
                work.requested.as_range().len(),
                work.context_tokens
            ),
            (128, 1, 129)
        );
    }
    sched.update_decode_progress(&id, 2);
    {
        let queue = sched.decode_queue.read();
        let (work, exact) = sched
            .cost_work(&queue.requests[&id], 1, true, PrefillShapePolicy::Identity)
            .unwrap();
        assert!(exact);
        assert_eq!((work.requested.start(), work.context_tokens), (129, 130));
    }
    assert!(sched.defer_decode_to_waiting_for_capacity(&id, 1));
    assert!(sched.promote_to_prefill_with_empty_retry(&id, None));
    {
        let queue = sched.prefill_queue.read();
        assert!(
            !sched
                .cost_work(&queue[0], 1, false, PrefillShapePolicy::Identity)
                .unwrap()
                .1
        );
    }
    sched.mark_prefill_complete(&id, 130);
    {
        let queue = sched.decode_queue.read();
        let (work, exact) = sched
            .cost_work(&queue.requests[&id], 1, true, PrefillShapePolicy::Identity)
            .unwrap();
        assert!(exact);
        assert_eq!((work.requested.start(), work.context_tokens), (130, 131));
    }
    sched.update_decode_progress(&id, 4);
    {
        let queue = sched.decode_queue.read();
        let (work, exact) = sched
            .cost_work(&queue.requests[&id], 1, true, PrefillShapePolicy::Identity)
            .unwrap();
        assert!(exact);
        assert_eq!((work.requested.start(), work.context_tokens), (131, 132));
    }
    sched
        .decode_queue
        .write()
        .requests
        .get_mut(&id)
        .unwrap()
        .inner
        .request
        .metadata
        .remove(PROMPT_TOKENS_METADATA_KEY);
    let queue = sched.decode_queue.read();
    assert!(
        !sched
            .cost_work(&queue.requests[&id], 1, true, PrefillShapePolicy::Identity)
            .unwrap()
            .1
    );
}

#[test]
fn slo_known_oversize_then_unknown_preserves_declared_policy_and_counts_final_shrink() {
    for policy in [
        UnknownCostPolicy::StaticFallback,
        UnknownCostPolicy::Reject,
        UnknownCostPolicy::Wait {
            max_wait_ms: 1000,
            on_timeout: CostFallback::Reject,
        },
    ] {
        for mixed in [false, true] {
            let sched = scheduler(None);
            if mixed {
                activate_decode_requests(&sched, 1);
            }
            let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
            enqueue_waiting(
                &sched,
                create_test_request_with_prompt_tokens(Priority::Normal, 100).with_ingress(ingress),
            );
            admit(&sched, None);
            let source = Arc::new(Snapshot {
                revision: 1,
                at: ingress.instant(),
                unknown: false,
                minimum_covered_prefill: 100,
                calls: Mutex::new(Vec::new()),
            });
            let mut config = targets();
            config.ttft_ms = 50.0;
            config.itl_ms = 50.0;
            config.unknown_cost = policy;
            let mut planning = TimePlanningRound::new(source, config, PrefillShapePolicy::Identity);
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
                .unwrap();
            assert!(planning.statistics.budget_misses > 0);
            match policy {
                UnknownCostPolicy::StaticFallback if !mixed => {
                    assert_eq!(batch.unwrap().requests[0].tokens_to_process, Some(50));
                    assert_eq!(planning.statistics.shrunk_prefills, 1);
                    assert_eq!(planning.statistics.static_fallbacks, 1);
                }
                _ => {
                    assert_eq!(planning.statistics.shrunk_prefills, 0);
                    if mixed {
                        let batch = batch.unwrap();
                        assert_eq!(batch.requests.len(), 1);
                        assert_eq!(batch.requests[0].tokens_to_process, Some(1));
                    } else {
                        assert!(batch.is_none());
                    }
                    match policy {
                        UnknownCostPolicy::Reject => {
                            assert_eq!(sched.take_admission_failures().len(), 1)
                        }
                        UnknownCostPolicy::Wait { .. } => assert_eq!(
                            planning.retry_at(),
                            Some(ingress.instant() + Duration::from_secs(1))
                        ),
                        UnknownCostPolicy::StaticFallback => assert!(planning.time_blocked()),
                    }
                }
            }
        }
    }
}

#[test]
fn request_clock_ttft_priority_uses_ingress_and_excludes_recompute() {
    for wall_direction in [-1, 1] {
        let sched = scheduler(None);
        let ingress = RequestIngressAnchor::capture(RequestIngressBoundary::SchedulerApi);
        let short_request =
            create_test_request_with_prompt_tokens(Priority::Normal, 20).with_ingress(ingress);
        let urgent_request =
            create_test_request_with_prompt_tokens(Priority::Normal, 120).with_ingress(ingress);
        let short = short_request.id.clone();
        let urgent = urgent_request.id.clone();
        enqueue_waiting(&sched, short_request);
        enqueue_waiting(&sched, urgent_request);
        admit(&sched, None);
        let mut queue = sched.prefill_queue.write();
        for request in queue.iter_mut() {
            let direction = if request.inner.request.id == short {
                wall_direction
            } else {
                -wall_direction
            };
            let wall = chrono::Utc::now() + chrono::Duration::days(direction * 3650);
            request.inner.submitted_at = wall;
            request.inner.started_at = Some(wall);
            request.inner.request.created_at = wall;
        }
        let mut planning = TimePlanningRound::new(
            snapshot(1, ingress.instant(), false),
            targets(),
            PrefillShapePolicy::Identity,
        );
        sched.prioritize_ttft(
            &mut queue,
            ingress.instant() + Duration::from_millis(800),
            Some(&mut planning),
        );
        assert_eq!(queue.front().unwrap().inner.request.id, urgent);
        queue.front_mut().unwrap().slo_first_token_emitted = true;
        sched.prioritize_ttft(
            &mut queue,
            ingress.instant() + Duration::from_millis(950),
            Some(&mut planning),
        );
        assert_eq!(queue.front().unwrap().inner.request.id, short);
        assert_eq!(queue.back().unwrap().inner.request.id, urgent);
    }
}

#[tokio::test]
async fn slo_invalid_programmatic_targets_reject_requests() {
    let sched = ContinuousBatchScheduler::new(SchedulerConfig {
        slo: Some(SchedulerSloConfig {
            ttft_ms: f64::NAN,
            ..targets()
        }),
        ..SchedulerConfig::default()
    });
    assert!(sched
        .submit(create_test_request(Priority::Normal))
        .await
        .is_err());
    assert_eq!(sched.waiting_count(), 0);
}
