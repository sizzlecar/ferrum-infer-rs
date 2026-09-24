use super::*;
use crate::continuous_engine::inner::slo_controller::tests::fixture::{
    fixture_with_width, ControlledExecutor,
};
use crate::continuous_engine::output_flow_runtime::OutputReadinessState;
use ferrum_interfaces::execution_cost::ActualRowWork;
use ferrum_interfaces::scheduler::Scheduler;
use futures::{FutureExt, StreamExt};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::time::Duration;

#[path = "reference/tests.rs"]
mod reference;
mod terminal;
mod token_policy_residency;

async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(3), future)
        .await
        .expect("manual calibration stalled")
}

async fn fixture(width: usize) -> (CalibrationSession, Arc<ControlledExecutor>) {
    let (engine, _, executor) = fixture_with_width(width).await;
    // The shared test factory disables automatic startup for its own driver.
    // Here no request has existed; exercise the session's actual fresh gate.
    engine.inner.bg_loop_spawned.store(false, Ordering::Release);
    let session = CalibrationSession::from_fresh_engine(
        engine,
        CalibrationLimits::new(NonZeroUsize::new(width).unwrap()).unwrap(),
    )
    .unwrap();
    (session, executor)
}

async fn ready(session: &CalibrationSession, id: &RequestId, blocked: bool) {
    let mut changed = session.engine.inner.sequences.read()[id]
        .credited_output
        .as_ref()
        .unwrap()
        .port
        .subscribe();
    bounded(async {
        loop {
            let current = session.engine.inner.sequences.read()[id]
                .credited_output
                .as_ref()
                .unwrap()
                .port
                .readiness();
            if matches!(current, OutputReadinessState::OutputBlocked(_)) == blocked
                && (blocked || current == OutputReadinessState::Ready)
            {
                break;
            }
            changed.changed().await.unwrap();
        }
    })
    .await;
}

async fn add(
    session: &mut CalibrationSession,
    tokens: usize,
) -> (RequestId, CreditedOutputSession) {
    let mut request = ferrum_types::InferenceRequest::new(
        vec!["test"; tokens].join(" "),
        session.configuration().model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let id = request.id.clone();
    let output = session
        .add_request(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(session, &id, false).await;
    (id, output)
}

fn frontier(session: &CalibrationSession, id: &RequestId) -> CalibrationFrontier {
    session
        .frontiers()
        .unwrap()
        .into_iter()
        .find(|row| row.request_id() == id)
        .unwrap()
}

async fn admit(session: &mut CalibrationSession) {
    assert!(matches!(
        session.step(CalibrationAction::AdmitOne).await.unwrap(),
        CalibrationTurn::AdmittedOrMaintained
    ));
}

#[tokio::test]
async fn calibration_full_logits_route_executes_without_rewriting_sequence_policy_or_history() {
    use ferrum_interfaces::model_executor::LogitsReturnPolicy;
    let (mut session, executor) = fixture(1).await;
    let (id, mut output) = add(&mut session, 4).await;
    admit(&mut session).await;
    let prefill = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(4).unwrap())
        .unwrap();
    let first = wave(&mut session, &executor, vec![prefill]).await;
    assert!(first.error.is_none());
    drop(bounded(output.frames.next()).await.unwrap());
    ready(&session, &id, false).await;
    let (original_policy, original_tokens, original_pending) = {
        let sequences = session.engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert!(sequence.can_use_model_greedy_argmax());
        (
            serde_json::to_value(&sequence.sampling_params).unwrap(),
            sequence.generated_tokens.clone(),
            sequence.pending_decoded_utf8_bytes.clone(),
        )
    };
    let before = frontier(&session, &id);
    let selected = before
        .decode_work_with_route(CalibrationDecodeRoute::FullLogits)
        .unwrap();
    let full = wave(&mut session, &executor, vec![selected]).await;
    assert!(full.error.is_none());
    assert_eq!(full.submission, CalibrationSubmissionState::HostReconciled);
    assert!(matches!(
        full.ordered_work.participants()[0]
            .selection()
            .decode_policy,
        Some(LogitsReturnPolicy::FullLogits)
    ));
    {
        let sequences = session.engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(
            serde_json::to_value(&sequence.sampling_params).unwrap(),
            original_policy
        );
        assert_eq!(
            &sequence.generated_tokens[..original_tokens.len()],
            &original_tokens
        );
        assert_eq!(sequence.generated_tokens.len(), original_tokens.len() + 1);
        assert_eq!(sequence.pending_decoded_utf8_bytes, original_pending);
        assert!(sequence.can_use_model_greedy_argmax());
    }
    drop(bounded(output.frames.next()).await.unwrap());
    ready(&session, &id, false).await;
    let after = frontier(&session, &id);
    assert_eq!(after.owner_incarnation(), before.owner_incarnation());
    let actual = wave(&mut session, &executor, vec![after.decode_work().unwrap()]).await;
    assert!(actual.error.is_none());
    assert!(matches!(
        actual.ordered_work.participants()[0]
            .selection()
            .decode_policy,
        Some(LogitsReturnPolicy::GreedyArgmax { .. })
    ));
    drop(output);
    session.shutdown().await.unwrap();
}

async fn wave(
    session: &mut CalibrationSession,
    executor: &ControlledExecutor,
    rows: Vec<CalibrationWork>,
) -> CalibrationWaveReport {
    bounded(async {
        loop {
            executor.resource_planning_unknown.lock().take();
            executor.resource_revalidation_changed.store(false, Ordering::Release);
            let entries = executor.entries.load(Ordering::Acquire);
            let turn = session
                .step(CalibrationAction::Wave(rows.clone()))
                .await
                .unwrap();
            let resource_unknown = executor.resource_planning_unknown.lock().take();
            let resource_changed = executor.resource_revalidation_changed.load(Ordering::Acquire);
            match turn {
                CalibrationTurn::Wave(report) => return report,
                CalibrationTurn::Blocked(
                    CalibrationBlockReason::ResourceUnavailable(
                        reason @ ferrum_interfaces::vnext::ResourcePlanningUnknown::ReadUnavailable(_),
                    ),
                ) =>
                {
                    // These parallel fixtures share the real device-budget
                    // account. Retry only a proven nonblocking read miss or
                    // changed resource view; the next turn must capture and
                    // validate the same exact work against the new view.
                    // Never retry an unsupported shape or stale request owner.
                    assert_eq!(resource_unknown, Some(reason));
                    assert_eq!(executor.entries.load(Ordering::Acquire), entries);
                    tokio::task::yield_now().await;
                }
                CalibrationTurn::Blocked(CalibrationBlockReason::PublicationUnavailable)
                    if resource_changed || matches!(
                        resource_unknown,
                        Some(ferrum_interfaces::vnext::ResourcePlanningUnknown::ReadUnavailable(_))
                    ) =>
                {
                    assert_eq!(executor.entries.load(Ordering::Acquire), entries);
                    tokio::task::yield_now().await;
                }
                other => {
                    panic!("expected one exact wave, got {other:?}; resource={resource_unknown:?}, changed={resource_changed}")
                }
            }
        }
    })
    .await
}

#[tokio::test]
async fn calibration_resource_failure_is_typed_and_not_reused_by_the_next_turn() {
    use ferrum_interfaces::vnext::ResourcePlanningUnknown;
    let (mut session, executor) = fixture(1).await;
    let (id, output) = add(&mut session, 4).await;
    admit(&mut session).await;
    let before = frontier(&session, &id);
    let work = before.prefill_work(NonZeroU32::new(2).unwrap()).unwrap();
    // Terminate the actual core session, not a permissive resource-view mock.
    // Its real planning read must refuse the now-aborted owner before any call.
    executor.abort_resource_sessions();
    let turn = bounded(async {
        loop {
            let turn = session
                .step(CalibrationAction::Wave(vec![work.clone()]))
                .await
                .unwrap();
            if matches!(
                turn,
                CalibrationTurn::Blocked(CalibrationBlockReason::ResourceUnavailable(
                    ResourcePlanningUnknown::ReadUnavailable(_)
                ))
            ) {
                tokio::task::yield_now().await;
                continue;
            }
            break turn;
        }
    })
    .await;
    assert!(
        matches!(
            turn,
            CalibrationTurn::Blocked(CalibrationBlockReason::ResourceUnavailable(
                ResourcePlanningUnknown::BusyOrUnavailable
            ))
        ),
        "{turn:?}"
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    let after = frontier(&session, &id);
    assert_eq!(after.owner_incarnation(), before.owner_incarnation());
    assert_eq!(after.work_generation(), before.work_generation());
    assert_eq!(after.prefill_progress(), before.prefill_progress());
    assert_eq!(after.generated_tokens(), before.generated_tokens());
    assert_eq!(after.kv_tokens(), before.kv_tokens());
    let mut wrong_frontier = work;
    wrong_frontier.frontier.generation =
        NonZeroU64::new(before.work_generation().get() + 1).unwrap();
    assert!(matches!(
        session
            .step(CalibrationAction::Wave(vec![wrong_frontier]))
            .await
            .unwrap(),
        CalibrationTurn::Blocked(CalibrationBlockReason::SelectionUnavailable(
            "calibration_frontier_changed"
        ))
    ));
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    drop(output);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn calibration_is_manual_and_preserves_exact_partial_final_decode_work() {
    let (mut session, executor) = fixture(1).await;
    let (id, mut output) = add(&mut session, 4).await;
    tokio::task::yield_now().await;
    assert!(!session.engine.inner.bg_loop_spawned.load(Ordering::Acquire));
    assert!(!session.engine.inner.is_running.load(Ordering::Acquire));
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(
        session.configuration().scheduler.slo.mode,
        ferrum_types::SloMode::Observe
    );
    admit(&mut session).await;
    let original_input = *frontier(&session, &id).request_evidence();
    assert_eq!(original_input.original_input_tokens, 4);
    let partial = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(2).unwrap())
        .unwrap();
    let report = wave(&mut session, &executor, vec![partial.clone()]).await;
    assert_eq!(
        report.submission,
        CalibrationSubmissionState::HostReconciled
    );
    assert!(report.error.is_none());
    assert_eq!(
        report.ordered_work.participants()[0].selection().work,
        ActualRowWork::Prefill {
            offset: 0,
            count: 2,
            total_prompt_tokens: 4
        }
    );
    assert_eq!(frontier(&session, &id).prefill_progress(), Some((2, 4)));
    assert_eq!(frontier(&session, &id).generated_tokens(), 0);
    // The old observed generation cannot be reused after a partial commit.
    assert!(matches!(
        session
            .step(CalibrationAction::Wave(vec![partial]))
            .await
            .unwrap(),
        CalibrationTurn::Blocked(_)
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    let final_row = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(2).unwrap())
        .unwrap();
    assert_eq!(
        wave(&mut session, &executor, vec![final_row])
            .await
            .submission,
        CalibrationSubmissionState::HostReconciled
    );
    drop(bounded(output.frames.next()).await.unwrap());
    ready(&session, &id, false).await;
    let decode = frontier(&session, &id).decode_work().unwrap();
    assert_eq!(
        wave(&mut session, &executor, vec![decode]).await.submission,
        CalibrationSubmissionState::HostReconciled
    );
    assert_eq!(frontier(&session, &id).generated_tokens(), 2);
    assert_eq!(*frontier(&session, &id).request_evidence(), original_input);
    assert_eq!(executor.physical.load(Ordering::Acquire), 3);
    drop(output);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn calibration_never_substitutes_a_ready_subset_for_a_blocked_exact_cohort() {
    let (mut session, executor) = fixture(2).await;
    let (slow, slow_output) = add(&mut session, 1).await;
    admit(&mut session).await;
    let row = frontier(&session, &slow)
        .prefill_work(NonZeroU32::MIN)
        .unwrap();
    wave(&mut session, &executor, vec![row]).await;
    ready(&session, &slow, true).await;
    let (healthy, healthy_output) = add(&mut session, 2).await;
    admit(&mut session).await;
    let rows = vec![
        frontier(&session, &healthy)
            .prefill_work(NonZeroU32::new(2).unwrap())
            .unwrap(),
        frontier(&session, &slow).decode_work().unwrap(),
    ];
    assert!(matches!(
        session.step(CalibrationAction::Wave(rows)).await.unwrap(),
        CalibrationTurn::Blocked(_)
    ));
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(frontier(&session, &healthy).generated_tokens(), 0);
    assert_eq!(
        frontier(&session, &healthy).prefill_progress(),
        Some((0, 2))
    );
    drop((slow_output, healthy_output));
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn calibration_no_submit_receipt_and_dropped_waiter_reap_do_not_repeat_work() {
    let (mut session, executor) = fixture(1).await;
    let (id, output) = add(&mut session, 2).await;
    admit(&mut session).await;
    let row = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(2).unwrap())
        .unwrap();
    executor.replan_before_encode.store(true, Ordering::Release);
    let report = wave(&mut session, &executor, vec![row.clone()]).await;
    assert_eq!(report.submission, CalibrationSubmissionState::NotSubmitted);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    assert_eq!(frontier(&session, &id).prefill_progress(), Some((0, 2)));
    assert!(executor.entered.notified().now_or_never().is_some());

    executor
        .replan_before_encode
        .store(false, Ordering::Release);
    executor.park.store(true, Ordering::Release);
    let mut waiter = Box::pin(wave(&mut session, &executor, vec![row.clone()]));
    // A genuine try-read miss can yield before entering the executor. Keep
    // polling the wave until that entry; polling once then waiting only for
    // the notification would leave the wave itself suspended forever.
    bounded(async {
        tokio::select! {
            report = waiter.as_mut() => panic!("parked executor completed early: {report:?}"),
            _ = executor.entered.notified() => {},
        }
    })
    .await;
    assert_eq!(executor.entries.load(Ordering::Acquire), 2);
    drop(waiter);
    assert!(session.frontiers().is_err());
    let unavailable = std::env::temp_dir().join(format!(
        "ferrum-unreaped-calibration-{}",
        uuid::Uuid::new_v4()
    ));
    assert!(bounded(
        session.export_and_load_cost_profile(CalibrationProfilePaths {
            profile: unavailable.join("profile.json"),
            source: unavailable.join("source.jsonl"),
        })
    )
    .await
    .is_err());
    assert!(
        !unavailable.exists(),
        "pending work must not start artifact IO"
    );
    executor.resume.notify_one();
    let report = match bounded(session.step(CalibrationAction::Wave(vec![row])))
        .await
        .unwrap()
    {
        CalibrationTurn::Reaped(report) => report,
        other => panic!("dropped waiter must reap, not start the supplied action: {other:?}"),
    };
    assert_eq!(
        report.submission,
        CalibrationSubmissionState::HostReconciled
    );
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(executor.entries.load(Ordering::Acquire), 2);
    assert_eq!(frontier(&session, &id).generated_tokens(), 1);
    assert!(matches!(
        session.step(CalibrationAction::Reap).await.unwrap(),
        CalibrationTurn::Blocked(CalibrationBlockReason::NoPendingWork)
    ));
    drop(output);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn calibration_submitted_error_is_not_a_no_submit_receipt() {
    let (mut session, executor) = fixture(1).await;
    let (id, output) = add(&mut session, 1).await;
    admit(&mut session).await;
    let row = frontier(&session, &id)
        .prefill_work(NonZeroU32::MIN)
        .unwrap();
    executor.fail_after_submit.store(true, Ordering::Release);
    let report = wave(&mut session, &executor, vec![row]).await;
    assert_eq!(report.submission, CalibrationSubmissionState::Submitted);
    assert!(report.error.is_some());
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert!(session.frontiers().unwrap().is_empty());
    drop(output);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn calibration_rejects_other_sessions_and_does_not_narrow_chunks() {
    let (mut session, executor) = fixture(1).await;
    Arc::get_mut(&mut session.engine.inner)
        .unwrap()
        .config
        .scheduler
        .prefill_step_chunk = Some(2);
    let (id, output) = add(&mut session, 4).await;
    admit(&mut session).await;
    let row = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(4).unwrap())
        .unwrap();
    // Typed server capacity remains authoritative; the requested four tokens
    // must not silently become a successful two-token calibration sample.
    assert!(matches!(
        session
            .step(CalibrationAction::Wave(vec![row.clone()]))
            .await
            .unwrap(),
        CalibrationTurn::Blocked(_)
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    let (mut other, other_executor) = fixture(1).await;
    assert!(other
        .step(CalibrationAction::Wave(vec![row]))
        .await
        .is_err());
    assert_eq!(other_executor.entries.load(Ordering::Acquire), 0);
    drop(output);
    session.shutdown().await.unwrap();
    other.shutdown().await.unwrap();
}

#[tokio::test]
async fn calibration_publication_block_does_not_reuse_previous_frontier_rejection() {
    let (mut session, executor) = fixture(1).await;
    let (id, output) = add(&mut session, 4).await;
    admit(&mut session).await;
    let stale = frontier(&session, &id)
        .prefill_work(NonZeroU32::new(2).unwrap())
        .unwrap();
    wave(&mut session, &executor, vec![stale.clone()]).await;
    assert!(matches!(
        session
            .step(CalibrationAction::Wave(vec![stale]))
            .await
            .unwrap(),
        CalibrationTurn::Blocked(CalibrationBlockReason::SelectionUnavailable(
            "calibration_frontier_changed"
        ))
    ));
    let before = frontier(&session, &id);
    let final_row = before.prefill_work(NonZeroU32::new(2).unwrap()).unwrap();
    let changed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let signal = Arc::clone(&changed);
    let scheduler = Arc::clone(&session.engine.inner.scheduler);
    *executor.after_resource_revalidation.lock() = Some(Box::new(move || {
        scheduler.record_external_capacity_release();
        signal.store(true, Ordering::Release);
    }));
    let turn = bounded(async {
        loop {
            executor.resource_planning_unknown.lock().take();
            let turn = session
                .step(CalibrationAction::Wave(vec![final_row.clone()]))
                .await
                .unwrap();
            if changed.load(Ordering::Acquire) {
                break turn;
            }
            assert!(
                matches!(
                    executor.resource_planning_unknown.lock().take(),
                    Some(ferrum_interfaces::vnext::ResourcePlanningUnknown::ReadUnavailable(_))
                ),
                "publication hook was not reached: {turn:?}"
            );
            assert_eq!(executor.physical.load(Ordering::Acquire), 1);
            tokio::task::yield_now().await;
        }
    })
    .await;
    assert!(
        matches!(
            turn,
            CalibrationTurn::Blocked(CalibrationBlockReason::PublicationUnavailable)
        ),
        "current publication race was misreported: {turn:?}"
    );
    let after = frontier(&session, &id);
    assert_eq!(after.owner_incarnation(), before.owner_incarnation());
    assert_eq!(after.work_generation(), before.work_generation());
    assert_eq!(after.prefill_progress(), before.prefill_progress());
    assert_eq!(after.generated_tokens(), before.generated_tokens());
    assert_eq!(after.kv_tokens(), before.kv_tokens());
    assert_eq!(executor.entries.load(Ordering::Acquire), 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(
        wave(&mut session, &executor, vec![final_row])
            .await
            .submission,
        CalibrationSubmissionState::HostReconciled
    );
    drop(output);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn calibration_waiting_shutdown_releases_queue_index_and_output_without_submit() {
    for disconnected in [false, true] {
        let (mut session, executor) = fixture(1).await;
        let (id, output) = add(&mut session, 2).await;
        let inner = Arc::clone(&session.engine.inner);
        assert_eq!(inner.scheduler.waiting_count(), 1);
        assert_eq!(inner.scheduler.active_count(), 0);
        assert_eq!(
            inner.scheduler.request_state(&id),
            Some(ferrum_types::RequestState::Waiting)
        );
        let mut response = ferrum_types::InferenceResponse {
            request_id: id.clone(),
            text: String::new(),
            tokens: Vec::new(),
            finish_reason: ferrum_types::FinishReason::Length,
            usage: ferrum_types::TokenUsage::new(2, 0),
            latency_ms: 0,
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
            api_response: None,
            execution_evidence: None,
        };
        // A malformed success must leave the real waiting owner untouched.
        assert!(inner
            .scheduler
            .complete(id.clone(), &response)
            .await
            .is_err());
        assert_eq!(inner.scheduler.waiting_count(), 1);
        assert_eq!(inner.scheduler.metrics().completed_requests, 0);
        let mut output = Some(output);
        if disconnected {
            drop(output.take());
        }
        session.shutdown().await.unwrap();
        assert!(inner.sequences.read().is_empty());
        assert_eq!(inner.scheduler.waiting_count(), 0);
        assert_eq!(inner.scheduler.active_count(), 0);
        assert_eq!(inner.scheduler.request_state(&id), None);
        assert_eq!(inner.scheduler.metrics().completed_requests, 0);
        assert_eq!(inner.scheduler.metrics().failed_requests, 1);
        assert_eq!(executor.entries.load(Ordering::Acquire), 0);
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        // The same terminal request is now unknown; it cannot count twice.
        response.finish_reason = ferrum_types::FinishReason::Error;
        assert!(inner.scheduler.complete(id, &response).await.is_err());
        assert_eq!(inner.scheduler.metrics().failed_requests, 1);
        if let Some(output) = output {
            drop(output.frames);
            let completion = bounded(output.completion).await.unwrap();
            assert!(matches!(
                completion.payload(),
                ferrum_interfaces::output_flow::OutputCompletion::Failed(_)
            ));
            drop(completion);
        }
        let pool = inner.output_credit_pool.get().unwrap().as_ref().unwrap();
        let mut changed = pool.subscribe();
        bounded(async {
            loop {
                let credit = pool.snapshot();
                if credit.data_used == Default::default()
                    && credit.terminal_held == Default::default()
                {
                    break;
                }
                changed.changed().await.unwrap();
            }
        })
        .await;
    }
}

#[tokio::test]
async fn calibration_input_digest_comes_from_product_tokens_not_prompt_bytes() {
    let (mut session, executor) = fixture(3).await;
    let mut outputs = Vec::new();
    let mut evidence = Vec::new();
    for prompt in ["test test", "test \t test", "test ok"] {
        let mut request = ferrum_types::InferenceRequest::new(
            prompt,
            session.configuration().model.model_id.clone(),
        );
        request.stream = true;
        request.sampling_params.max_tokens = 4;
        let id = request.id.clone();
        outputs.push(
            session
                .add_request(
                    request,
                    InferenceRequestContext::from_ingress(slo_clock_now()),
                    Arc::new(OutputProjectionContract::cli_text()),
                )
                .await
                .unwrap(),
        );
        evidence.push(*frontier(&session, &id).request_evidence());
    }
    assert!(evidence.iter().all(|row| row.original_input_tokens == 2));
    assert_eq!(evidence[0], evidence[1]);
    assert_ne!(
        evidence[0].original_input_tokens_sha256,
        evidence[2].original_input_tokens_sha256
    );
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    drop(outputs);
    session.shutdown().await.unwrap();
}
