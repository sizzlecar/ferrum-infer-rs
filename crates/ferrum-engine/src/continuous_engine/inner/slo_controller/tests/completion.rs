//! Completion-only engine behavior, with actual scheduler/output ownership.
//! The controlled executor is not evidence of native encoding or Metal speed.
use super::*;
use ferrum_interfaces::engine::InferenceEngine;

mod deferral;
mod draft;
mod retry;
mod work_envelope;

async fn request(
    engine: &ContinuousBatchEngine,
    tokens: usize,
    maximum: usize,
    reuse: Option<RequestId>,
) -> (RequestId, CreditedOutputSession) {
    let mut request = ferrum_types::InferenceRequest::new(
        vec!["test"; tokens].join(" "),
        engine.inner.config.model.model_id.clone(),
    );
    if let Some(id) = reuse {
        request.id = id;
    }
    request.stream = true;
    request.sampling_params.max_tokens = maximum;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::from_ingress(slo_clock_now()),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    ready(engine, &id).await;
    let mut sequences = engine.inner.sequences.write();
    let sequence = sequences.get_mut(&id).unwrap();
    assert!(
        sequence.cost_frontier.is_some(),
        "real owner must remain fenced"
    );
    sequence.cost_policy_signature = None;
    assert_eq!(sequence.input_tokens.len(), tokens);
    (id, session)
}

async fn admit(engine: &ContinuousBatchEngine, maximum: usize) {
    let _iteration = engine.inner.iteration_lock.lock().await;
    assert!(engine
        .inner
        .prepare_slo_admission_turn(maximum)
        .await
        .unwrap());
}

async fn selected(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    width: usize,
    tokens: usize,
    reason: CompletionOnlyReason,
) -> owner::PreparedControllerWave {
    let mut hint = ferrum_interfaces::BatchHint::simple(width);
    hint.max_tokens = tokens;
    prefill::selected_after_retry(engine, executor, || {
        let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
        engine
            .inner
            .prepare_completion_controller(&hint, &budget, reason)
    })
    .await
}

async fn step(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    width: usize,
    tokens: usize,
) {
    let prepared = selected(
        engine,
        executor,
        width,
        tokens,
        CompletionOnlyReason::CostUnavailable,
    )
    .await;
    assert!(matches!(
        bounded(engine.inner.execute_slo_controller_wave(prepared))
            .await
            .unwrap(),
        EngineIterationOutcome::Progressed
    ));
}

async fn consume(
    engine: &ContinuousBatchEngine,
    id: &RequestId,
    session: &mut CreditedOutputSession,
) {
    drop(
        bounded(session.frames.next())
            .await
            .expect("visible output"),
    );
    ready(engine, id).await;
}

#[tokio::test]
async fn completion_without_cost_or_timing_advances_partial_final_then_decode() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, mut session) = request(&engine, 4, 2, None).await;
    assert!(engine.inner.prefill_reference_runtime.is_none());
    assert!(engine.inner.sequences.read()[&id]
        .prefill_reference
        .is_none());
    // Missing SLO/reference timing must not remove an otherwise valid owner.
    engine.inner.sequences.write().get_mut(&id).unwrap().slo = None;
    assert!(engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    admit(&engine, 1).await;
    step(&engine, &executor, 1, 2).await;
    let audit = engine.inner.slo_controller.lock().last_audit.unwrap();
    assert!(
        audit
            .stages
            .iter()
            .any(|stage| { matches!(stage.stage, ControllerStage::Capture) && stage.calls > 0 }),
        "real completion capture was omitted from controller wall-time audit: {audit:?}"
    );
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.prefill_tokens_processed, 2);
        assert!(!sequence.prefill_complete);
        assert!(sequence.generated_tokens.is_empty());
        assert!(sequence.credited_output.as_ref().unwrap().grant.is_none());
        assert!(sequence.cost_policy_signature.is_none());
    }
    step(&engine, &executor, 1, 2).await;
    consume(&engine, &id, &mut session).await;
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    step(&engine, &executor, 1, 1).await;
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert_eq!(executor.physical.load(Ordering::Acquire), 3);
    assert_eq!(
        executor.submitted_requests.lock().as_slice(),
        &[vec![id.clone()], vec![id.clone()], vec![id]]
    );
    cleanup(engine, session).await;
}

#[tokio::test]
async fn completion_mixed_work_commits_both_phases_without_a_cost_witness() {
    let (engine, _, executor) = completion_fixture(2).await;
    let (decode_id, mut decode) = request(&engine, 1, 4, None).await;
    admit(&engine, 1).await;
    step(&engine, &executor, 1, 1).await;
    consume(&engine, &decode_id, &mut decode).await;
    let (prefill_id, mut prefill) = request(&engine, 2, 4, None).await;
    admit(&engine, 1).await;
    step(&engine, &executor, 2, 3).await;
    assert_eq!(
        engine.inner.sequences.read()[&decode_id]
            .generated_tokens
            .len(),
        2
    );
    assert_eq!(
        engine.inner.sequences.read()[&prefill_id]
            .generated_tokens
            .len(),
        1
    );
    assert!(engine.inner.sequences.read()[&prefill_id].prefill_complete);
    assert_eq!(executor.physical.load(Ordering::Acquire), 2);
    let submitted = executor.submitted_requests.lock().last().unwrap().clone();
    assert_eq!(submitted.len(), 2);
    assert!(submitted.contains(&decode_id) && submitted.contains(&prefill_id));
    consume(&engine, &decode_id, &mut decode).await;
    consume(&engine, &prefill_id, &mut prefill).await;
    drop(prefill);
    cleanup(engine, decode).await;
}

#[tokio::test]
async fn completion_width_one_rotates_every_live_owner_before_revisiting() {
    let (engine, _, executor) = completion_fixture(3).await;
    let mut ids = Vec::new();
    let mut sessions = Vec::new();
    for _ in 0..3 {
        let (id, session) = request(&engine, 1, 4, None).await;
        ids.push(id);
        sessions.push(session);
    }
    admit(&engine, 3).await;
    // Two complete owner rotations: the second consists of real decode work.
    for _ in 0..2 {
        let mut visited = Vec::new();
        for _ in 0..ids.len() {
            step(&engine, &executor, 1, 1).await;
            let submitted = executor.submitted_requests.lock().last().unwrap().clone();
            assert_eq!(submitted.len(), 1);
            assert!(
                !visited.contains(&submitted[0]),
                "one ready owner monopolized the wave"
            );
            let index = ids.iter().position(|id| id == &submitted[0]).unwrap();
            consume(&engine, &ids[index], &mut sessions[index]).await;
            visited.push(submitted[0].clone());
        }
        assert!(ids.iter().all(|id| visited.contains(id)));
    }
    assert!(ids
        .iter()
        .all(|id| engine.inner.sequences.read()[id].generated_tokens.len() == 2));
    let last = sessions.pop().unwrap();
    drop(sessions);
    cleanup(engine, last).await;
}

#[tokio::test]
async fn completion_slow_output_does_not_block_a_healthy_peer() {
    use crate::continuous_engine::output_flow_runtime::OutputReadinessState;
    let (engine, _, executor) = completion_fixture(2).await;
    let (slow_id, mut slow) = request(&engine, 1, 4, None).await;
    admit(&engine, 1).await;
    step(&engine, &executor, 1, 1).await;
    let mut changes = engine.inner.sequences.read()[&slow_id]
        .credited_output
        .as_ref()
        .unwrap()
        .port
        .subscribe();
    bounded(async {
        loop {
            if matches!(
                engine.inner.sequences.read()[&slow_id]
                    .credited_output
                    .as_ref()
                    .unwrap()
                    .port
                    .readiness(),
                OutputReadinessState::OutputBlocked(_)
            ) {
                break;
            }
            changes.changed().await.unwrap();
        }
    })
    .await;
    let (healthy_id, mut healthy) = request(&engine, 1, 3, None).await;
    admit(&engine, 1).await;
    step(&engine, &executor, 1, 1).await;
    assert_eq!(
        executor.submitted_requests.lock().last().unwrap(),
        &vec![healthy_id.clone()]
    );
    assert_eq!(
        engine.inner.sequences.read()[&slow_id]
            .generated_tokens
            .len(),
        1
    );
    assert_eq!(
        engine.inner.sequences.read()[&healthy_id]
            .generated_tokens
            .len(),
        1
    );
    // Releasing the real queued frame restores the previously blocked peer.
    consume(&engine, &slow_id, &mut slow).await;
    consume(&engine, &healthy_id, &mut healthy).await;
    step(&engine, &executor, 1, 1).await;
    assert_eq!(
        executor.submitted_requests.lock().last().unwrap(),
        &vec![slow_id.clone()]
    );
    assert_eq!(
        engine.inner.sequences.read()[&slow_id]
            .generated_tokens
            .len(),
        2
    );
    drop(healthy);
    cleanup(engine, slow).await;
}

#[tokio::test(start_paused = true)]
async fn completion_expired_wait_and_sticky_slo_miss_do_not_shorten_output() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, mut session) = request(&engine, 1, 3, None).await;
    // Longer than both the original 10s TTFT and the configured 30s wait.
    tokio::time::advance(Duration::from_secs(31)).await;
    {
        let mut sequences = engine.inner.sequences.write();
        let sequence = sequences.get_mut(&id).unwrap();
        let timing = sequence.slo.as_mut().unwrap();
        timing.observe_wait(slo_clock_now()).unwrap();
        assert!(timing.violations().ttft);
        assert_eq!(sequence.sampling_params.max_tokens, 3);
    }
    admit(&engine, 1).await;
    let prepared = selected(
        &engine,
        &executor,
        1,
        1,
        CompletionOnlyReason::ExistingSloMiss,
    )
    .await;
    engine
        .inner
        .execute_slo_controller_wave(prepared)
        .await
        .unwrap();
    consume(&engine, &id, &mut session).await;
    assert!(
        engine.inner.sequences.read()[&id]
            .slo
            .as_ref()
            .unwrap()
            .violations()
            .ttft
    );
    assert_eq!(
        engine.inner.sequences.read()[&id]
            .sampling_params
            .max_tokens,
        3
    );
    step(&engine, &executor, 1, 1).await;
    consume(&engine, &id, &mut session).await;
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 2);
    step(&engine, &executor, 1, 1).await;
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert_eq!(executor.physical.load(Ordering::Acquire), 3);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn completion_cancelled_same_id_cannot_submit_or_modify_replacement() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, old) = request(&engine, 1, 3, None).await;
    admit(&engine, 1).await;
    let old_incarnation = engine.inner.sequences.read()[&id]
        .cost_frontier
        .unwrap()
        .owner_incarnation;
    let prepared = selected(
        &engine,
        &executor,
        1,
        1,
        CompletionOnlyReason::CostUnavailable,
    )
    .await;
    drop(old);
    bounded(async {
        while engine.inner.sequences.read().contains_key(&id) {
            engine.inner.cancel_abandoned_requests().await.unwrap();
            tokio::task::yield_now().await;
        }
    })
    .await;
    let (_, replacement) = request(&engine, 1, 4, Some(id.clone())).await;
    assert_ne!(
        engine.inner.sequences.read()[&id]
            .cost_frontier
            .unwrap()
            .owner_incarnation,
        old_incarnation
    );
    assert!(matches!(
        engine
            .inner
            .execute_slo_controller_wave(prepared)
            .await
            .unwrap(),
        EngineIterationOutcome::Idle
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert!(sequence.generated_tokens.is_empty());
        assert!(sequence.model_kv.is_none());
        assert_eq!(sequence.sampling_params.max_tokens, 4);
        assert!(sequence.credited_output.as_ref().unwrap().grant.is_none());
    }
    admit(&engine, 1).await;
    step(&engine, &executor, 1, 1).await;
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    cleanup(engine, replacement).await;
}

#[tokio::test]
async fn completion_caller_drop_preserves_one_dispatch_and_one_commit() {
    let (engine, _, executor) = completion_fixture(1).await;
    let (id, mut session) = request(&engine, 1, 4, None).await;
    admit(&engine, 1).await;
    step(&engine, &executor, 1, 1).await;
    consume(&engine, &id, &mut session).await;
    // Discard the first completed wave's notification; the wait below must
    // observe the second dispatch actually entering the parked executor.
    assert!(executor.entered.notified().now_or_never().is_some());
    let prepared = selected(
        &engine,
        &executor,
        1,
        1,
        CompletionOnlyReason::WitnessExpired,
    )
    .await;
    executor.park.store(true, Ordering::Release);
    let mut waiting = Box::pin(engine.inner.execute_slo_controller_wave(prepared));
    assert!(waiting.as_mut().now_or_never().is_none());
    bounded(executor.entered.notified()).await;
    assert_eq!(executor.entries.load(Ordering::Acquire), 2);
    assert_eq!(executor.physical.load(Ordering::Acquire), 1);
    drop(waiting);
    assert_eq!(scheduled(&engine), 1);
    executor.resume.notify_one();
    assert!(matches!(
        bounded(engine.inner.drain_slo_execution()).await.unwrap(),
        Some(EngineIterationOutcome::Progressed)
    ));
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 2);
    assert_eq!(executor.physical.load(Ordering::Acquire), 2);
    assert_eq!(executor.entries.load(Ordering::Acquire), 2);
    assert_eq!(scheduled(&engine), 0);
    assert!(engine.inner.drain_slo_execution().await.unwrap().is_none());
    assert_eq!(executor.physical.load(Ordering::Acquire), 2);
    cleanup(engine, session).await;
}

#[tokio::test]
async fn completion_requires_enforce_and_complete_requests_policy() {
    let (mut engine, _, executor) = fixture().await;
    let hint = ferrum_interfaces::BatchHint::simple(1);
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap();
    assert!(matches!(
        engine
            .inner
            .prepare_completion_controller(&hint, &budget, CompletionOnlyReason::CostUnavailable)
            .unwrap(),
        SloIterationPlan::Idle
    ));
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.scheduler.slo.admission.time_policy =
        ferrum_types::SloTimeAdmissionPolicy::RequireSlo;
    assert!(!inner.completion_allowed());
    assert!(matches!(
        inner
            .prepare_completion_controller(&hint, &budget, CompletionOnlyReason::CostUnavailable)
            .unwrap(),
        SloIterationPlan::Idle
    ));
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    engine.shutdown().await.unwrap();
}
