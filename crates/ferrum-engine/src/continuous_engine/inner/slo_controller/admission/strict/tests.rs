use super::*;
use crate::continuous_engine::inner::slo_controller::tests::{
    bounded,
    fixture::{fixture_with_width, ControlledExecutor},
    prefill, ready,
};
use ferrum_interfaces::{
    engine::{InferenceEngine, LlmInferenceEngine},
    output_flow::{OutputCompletion, OutputProjectionContract},
    InferenceRequestContext,
};
use ferrum_scheduler::implementations::continuous::ContinuousBatchScheduler;
use ferrum_types::{SloMode, SloTimeAdmissionPolicy};
use futures::{FutureExt, StreamExt};
use std::{sync::atomic::Ordering, time::Duration};

mod feasible;

async fn fixture(
    width: usize,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    let (mut engine, scheduler, executor) = fixture_with_width(width).await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = SloMode::Enforce;
    inner.config.scheduler.slo.admission.time_policy = SloTimeAdmissionPolicy::RequireSlo;
    inner.config.scheduler.slo.admission.max_wait_ms = NonZeroU64::new(30_000).unwrap();
    (engine, scheduler, executor)
}

fn input(engine: &ContinuousBatchEngine, tokens: usize, maximum: usize) -> InferenceRequest {
    let mut request = InferenceRequest::new(
        vec!["test"; tokens].join(" "),
        engine.inner.config.model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.max_tokens = maximum;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    request
}

async fn cancel(engine: &ContinuousBatchEngine) {
    let _iteration = engine.inner.iteration_lock.lock().await;
    engine.inner.cancel_abandoned_requests().await.unwrap();
    engine.inner.expire_strict_admissions().await.unwrap();
}

#[tokio::test]
async fn strict_installed_repetition_sampling_can_be_selected_without_acceptance() {
    use ferrum_interfaces::execution_cost::PlainTextSamplingRouteV2;
    for temperature in [0.0, 0.8] {
        let (engine, _, executor) = fixture(1).await;
        let mut request = input(&engine, 4, 2);
        request.sampling_params.temperature = temperature;
        request.sampling_params.repetition_penalty = 1.1;
        let id = request.id.clone();
        let mut submitted = Box::pin(engine.infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text()),
        ));
        assert!(submitted.as_mut().now_or_never().is_none());
        ready(&engine, &id).await;
        prefill::admit(&engine, 1).await;
        let captured = prefill::captured(&engine, &executor).await;
        let sequences = engine.inner.sequences.read();
        let route = if temperature == 0.0 {
            PlainTextSamplingRouteV2::Greedy {
                repetition_penalty: true,
            }
        } else {
            PlainTextSamplingRouteV2::FullLogits
        };
        assert_eq!(
            sampling::capability(&sequences[&id]),
            Ok(sampling::SamplingCapability::InstalledPlainText(route))
        );
        assert_eq!(
            engine.inner.strict_candidate(
                &sequences,
                &captured.queue,
                captured.model.model_version()
            ),
            Some(id.clone())
        );
        assert!(sequences[&id]
            .time_admission
            .as_ref()
            .unwrap()
            .before_acceptance());
        assert!(sequences[&id].generated_tokens.is_empty());
        assert_eq!(captured.snapshot.requests[0].key.request_id, id);
        assert!(captured.waiting_fences.is_empty());
        assert_eq!(
            captured.fences[0].future_repetition.is_some(),
            temperature == 0.0
        );
        if temperature == 0.0 {
            let (unique, penalty, vocabulary) = captured.fences[0].future_repetition.unwrap();
            assert_eq!((unique, penalty), (0, 1.1));
            assert_eq!(
                vocabulary,
                engine.inner.model_executor.info().vocab_size as u64
            );
        }
        drop(sequences);
        assert!(submitted.as_mut().now_or_never().is_none());
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        drop(captured);
        drop(submitted);
        cancel(&engine).await;
        engine.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn strict_repetition_without_installed_identity_or_capability_stays_waiting() {
    let (engine, _, executor) = fixture(1).await;
    let mut request = input(&engine, 4, 2);
    request.sampling_params.repetition_penalty = 1.1;
    let id = request.id.clone();
    let mut submitted = Box::pin(engine.infer_credited_stream(
        request,
        InferenceRequestContext::capture(),
        Arc::new(OutputProjectionContract::cli_text()),
    ));
    assert!(submitted.as_mut().now_or_never().is_none());
    ready(&engine, &id).await;
    prefill::admit(&engine, 1).await;
    let captured = prefill::captured(&engine, &executor).await;
    for missing_identity in [true, false] {
        let (identity, numeric_policy) = {
            let mut sequences = engine.inner.sequences.write();
            let sequence = sequences.get_mut(&id).unwrap();
            let original = (sequence.cost_policy_signature, sequence.cost_numeric_policy);
            if missing_identity {
                sequence.cost_policy_signature = None;
            } else {
                sequence.cost_numeric_policy = None;
            }
            original
        };
        {
            let sequences = engine.inner.sequences.read();
            assert_eq!(
                sampling::capability(&sequences[&id]),
                Err(if missing_identity {
                    sampling::SamplingUnavailable::MissingIdentity
                } else {
                    sampling::SamplingUnavailable::UnsupportedRoute
                })
            );
            assert!(engine
                .inner
                .strict_candidate(&sequences, &captured.queue, captured.model.model_version())
                .is_none());
            assert!(sequences[&id]
                .time_admission
                .as_ref()
                .unwrap()
                .before_acceptance());
        }
        let unavailable = engine
            .inner
            .capture_slo_controller_snapshot(
                &ferrum_interfaces::BatchHint::simple(1),
                ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap(),
            )
            .err()
            .unwrap();
        assert_eq!(unavailable.reason, "strict_waiting_evidence");
        assert!(submitted.as_mut().now_or_never().is_none());
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        let mut sequences = engine.inner.sequences.write();
        let sequence = sequences.get_mut(&id).unwrap();
        sequence.cost_policy_signature = identity;
        sequence.cost_numeric_policy = numeric_policy;
    }
    drop(captured);
    drop(submitted);
    cancel(&engine).await;
    engine.shutdown().await.unwrap();
}

/// Explicit finite lifecycle evidence, never used as a measured cost sample
/// or as a claim that the controlled backend's missing projector is covered.
async fn witness(
    engine: &ContinuousBatchEngine,
    executor: &ControlledExecutor,
    id: &RequestId,
) -> PendingTimeWitness {
    let captured = prefill::captured(engine, executor).await;
    let sequences = engine.inner.sequences.read();
    let sequence = &sequences[id];
    let state = sequence.time_admission.as_ref().unwrap();
    PendingTimeWitness {
        request_id: id.clone(),
        owner: sequence.stream_projection_identity.clone(),
        work_generation: sequence.cost_frontier.unwrap().work_generation.get(),
        ingress: state.ingress,
        original_input_tokens: state.original_input_tokens,
        maximum_output_tokens: state.maximum_output_tokens,
        evidence: StartedTimeWitness {
            snapshot_generation: captured.snapshot.generation,
            model_version: captured.snapshot.cost_model_version,
            validated_through: slo_clock_now() + Duration::from_secs(30),
            obligations_beyond_horizon: 1,
        },
    }
}

#[tokio::test(start_paused = true)]
async fn strict_unknown_waits_without_execution_and_expires_on_original_ingress() {
    let (mut engine, scheduler, executor) = fixture(1).await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.cost_runtime = None;
    inner.config.scheduler.slo.admission.max_wait_ms = NonZeroU64::new(100).unwrap();
    let request = input(&engine, 4, 2);
    let id = request.id.clone();
    let ingress = slo_clock_now();
    let mut submitted = Box::pin(engine.infer_credited_stream(
        request,
        InferenceRequestContext::from_ingress(ingress),
        Arc::new(OutputProjectionContract::cli_text()),
    ));
    assert!(submitted.as_mut().now_or_never().is_none());
    ready(&engine, &id).await;
    assert!(engine.inner.sequences.read()[&id]
        .time_admission
        .as_ref()
        .unwrap()
        .before_acceptance());
    let mut wait = Box::pin(engine.inner.wait_for_slo_time_admission());
    assert!(wait.as_mut().now_or_never().is_none());
    tokio::time::advance(Duration::from_millis(99)).await;
    assert!(wait.as_mut().now_or_never().is_none());
    assert!(submitted.as_mut().now_or_never().is_none());
    tokio::time::advance(Duration::from_millis(1)).await;
    assert!(wait.as_mut().now_or_never().is_some());
    drop(wait);
    engine.inner.run_iteration().await.unwrap();
    assert!(matches!(
        submitted.await,
        Err(FerrumError::SloTimeAdmissionRejected {
            reason: SloTimeAdmissionRejection::WaitExpired,
        })
    ));
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(scheduler.trace_phase(&id), None);
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn strict_cancel_and_shutdown_resolve_pending_owners_without_acceptance() {
    let (engine, scheduler, executor) = fixture(2).await;
    let request = input(&engine, 4, 2);
    let id = request.id.clone();
    let mut submitted = Box::pin(engine.infer_credited_stream(
        request,
        InferenceRequestContext::capture(),
        Arc::new(OutputProjectionContract::cli_text()),
    ));
    assert!(submitted.as_mut().now_or_never().is_none());
    drop(submitted);
    cancel(&engine).await;
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(scheduler.trace_phase(&id), None);
    let request = input(&engine, 4, 2);
    let mut submitted = Box::pin(engine.infer_credited_stream(
        request,
        InferenceRequestContext::capture(),
        Arc::new(OutputProjectionContract::cli_text()),
    ));
    assert!(submitted.as_mut().now_or_never().is_none());
    engine.shutdown().await.unwrap();
    assert!(matches!(
        submitted.await,
        Err(FerrumError::Cancelled { .. })
    ));
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(executor.entries.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn strict_physical_preparation_does_not_release_waiting_prompt_capacity() {
    for limits in [(1, 100, 1000), (10, 4, 1000), (10, 100, 19)] {
        let (mut engine, scheduler, executor) = fixture(2).await;
        let policy = &mut Arc::get_mut(&mut engine.inner)
            .unwrap()
            .config
            .scheduler
            .slo
            .admission;
        policy.max_waiting_requests = NonZeroUsize::new(limits.0).unwrap();
        policy.max_waiting_prompt_tokens = NonZeroUsize::new(limits.1).unwrap();
        policy.max_waiting_prompt_bytes = NonZeroUsize::new(limits.2).unwrap();
        let first = input(&engine, 4, 2);
        let id = first.id.clone();
        let mut submitted = Box::pin(engine.infer_credited_stream(
            first,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text()),
        ));
        assert!(submitted.as_mut().now_or_never().is_none());
        ready(&engine, &id).await;
        prefill::admit(&engine, 1).await;
        assert_eq!(scheduler.waiting_count(), 0);
        let second = input(&engine, 1, 2);
        assert!(matches!(
            engine
                .infer_credited_stream(
                    second,
                    InferenceRequestContext::capture(),
                    Arc::new(OutputProjectionContract::cli_text()),
                )
                .await,
            Err(FerrumError::ResourceExhausted { .. })
        ));
        assert_eq!(executor.physical.load(Ordering::Acquire), 0);
        drop(submitted);
        cancel(&engine).await;
        engine.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn strict_snapshot_keeps_full_owner_fence_but_only_one_pending_candidate() {
    let (engine, _, executor) = fixture(2).await;
    let first = input(&engine, 4, 2);
    let first_id = first.id.clone();
    let second = input(&engine, 4, 2);
    let second_id = second.id.clone();
    let mut one = Box::pin(engine.infer_credited_stream(
        first,
        InferenceRequestContext::capture(),
        Arc::new(OutputProjectionContract::cli_text()),
    ));
    let mut two = Box::pin(engine.infer_credited_stream(
        second,
        InferenceRequestContext::capture(),
        Arc::new(OutputProjectionContract::cli_text()),
    ));
    assert!(one.as_mut().now_or_never().is_none());
    assert!(two.as_mut().now_or_never().is_none());
    ready(&engine, &first_id).await;
    ready(&engine, &second_id).await;
    prefill::admit(&engine, 2).await;
    let captured = prefill::captured(&engine, &executor).await;
    assert_eq!(captured.queue.requests().len(), 2);
    assert_eq!(captured.snapshot.requests.len(), 1);
    assert_eq!(captured.fences.len(), 1);
    assert_eq!(captured.waiting_fences.len(), 1);
    assert!(engine.inner.controller_frontiers_match(&captured));
    let omitted = captured.waiting_fences[0].key.request_id.clone();
    let (original_reference, original_identity) = {
        let mut sequences = engine.inner.sequences.write();
        let sequence = sequences.get_mut(&omitted).unwrap();
        sequence.sampling_params.repetition_penalty = 1.25;
        (
            sequence.prefill_reference.take(),
            sequence.cost_policy_signature.take(),
        )
    };
    let unaffected = prefill::captured(&engine, &executor).await;
    assert_eq!(
        unaffected.snapshot.requests[0].key,
        captured.snapshot.requests[0].key
    );
    assert_eq!(unaffected.waiting_fences[0].key.request_id, omitted);
    assert!(unaffected.waiting_fences[0].future_greedy_policy.is_none());
    assert!(unaffected.waiting_fences[0].future_repetition.is_none());
    drop(unaffected);
    {
        let mut sequences = engine.inner.sequences.write();
        let sequence = sequences.get_mut(&omitted).unwrap();
        sequence.sampling_params.repetition_penalty = 1.0;
        sequence.prefill_reference = original_reference;
        sequence.cost_policy_signature = original_identity;
    }
    engine
        .inner
        .sequences
        .write()
        .get_mut(&omitted)
        .unwrap()
        .prefill_tokens_processed += 1;
    assert!(!engine.inner.controller_frontiers_match(&captured));
    engine
        .inner
        .sequences
        .write()
        .get_mut(&omitted)
        .unwrap()
        .prefill_tokens_processed -= 1;
    let proposal = engine.inner.propose_slo_time_admission(&captured).unwrap();
    assert!(matches!(
        proposal.decision,
        Some(PlanningDecision::Unknown { .. })
    ));
    assert!(proposal.pending.is_none());
    assert_eq!(executor.physical.load(Ordering::Acquire), 0);
    let next = prefill::captured(&engine, &executor).await;
    assert_eq!(next.snapshot.requests[0].key.request_id, omitted);
    let next_proposal = engine.inner.propose_slo_time_admission(&next).unwrap();
    assert!(matches!(
        next_proposal.decision,
        Some(PlanningDecision::Unknown { .. })
    ));
    assert!(
        engine
            .inner
            .strict_candidate(
                &engine.inner.sequences.read(),
                &next.queue,
                next.model.model_version(),
            )
            .is_none(),
        "unchanged Unknown evidence cannot create an endless candidate rotation"
    );
    drop(next);
    let no_completion = engine
        .inner
        .prepare_completion_controller(
            &ferrum_interfaces::BatchHint::simple(2),
            &ControllerBudget::new(slo_clock_now(), Duration::from_secs(30)).unwrap(),
            ferrum_interfaces::execution_cost::CompletionOnlyReason::CostUnavailable,
        )
        .unwrap();
    assert!(matches!(no_completion, SloIterationPlan::Idle));
    assert!(one.as_mut().now_or_never().is_none());
    assert!(two.as_mut().now_or_never().is_none());
    // An omitted waiter still belongs to the output/owner safety seal. Moving
    // its actual credit invalidates publication even though it was not forecast.
    let grant = {
        let sequences = engine.inner.sequences.read();
        match sequences[&omitted]
            .credited_output
            .as_ref()
            .unwrap()
            .port
            .try_take()
        {
            crate::continuous_engine::output_flow_runtime::OutputReadiness::Ready(grant) => grant,
            _ => panic!("ready omitted owner retains its original output credit"),
        }
    };
    assert!(!engine.inner.controller_frontiers_match(&captured));
    grant.return_unsubmitted();
    drop(captured);
    drop(one);
    drop(two);
    cancel(&engine).await;
    engine.shutdown().await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn strict_stale_receipt_cannot_accept_and_accepted_owner_survives_later_cost_loss() {
    let (mut engine, scheduler, executor) = fixture(1).await;
    let request = input(&engine, 4, 2);
    let id = request.id.clone();
    // This test pauses Tokio's monotonic clock. Transport capture() uses
    // std::Instant and can then lie in the controller's virtual future.
    // Supply the original ingress from the same explicit virtual clock.
    let ingress = slo_clock_now();
    let mut submitted = Box::pin(engine.infer_credited_stream(
        request,
        InferenceRequestContext::from_ingress(ingress),
        Arc::new(OutputProjectionContract::cli_text()),
    ));
    assert!(submitted.as_mut().now_or_never().is_none());
    ready(&engine, &id).await;
    prefill::admit(&engine, 1).await;
    let receipt = witness(&engine, &executor, &id).await;
    let mut stale = receipt.clone();
    stale.owner = Arc::new(());
    assert!(!engine
        .inner
        .accept_strict_witness(&stale, slo_clock_now() + Duration::from_secs(1)));
    stale = receipt.clone();
    stale.evidence.model_version += 1;
    assert!(!engine
        .inner
        .accept_strict_witness(&stale, slo_clock_now() + Duration::from_secs(1)));
    assert!(!engine
        .inner
        .accept_strict_witness(&receipt, slo_clock_now() - Duration::from_nanos(1)));
    assert!(submitted.as_mut().now_or_never().is_none());
    let prepared = prefill::install_with_admission(
        &engine,
        &executor,
        &scheduler,
        &[(
            id.clone(),
            PlanningWorkAction::Prefill {
                offset: 0,
                count: NonZeroUsize::new(4).unwrap(),
            },
        )],
        Some(receipt.clone()),
    )
    .await;
    assert!(engine
        .inner
        .accept_strict_witness(&receipt, slo_clock_now() + Duration::from_secs(1)));
    let mut session = submitted.await.unwrap();
    assert!(!engine.inner.sequences.read()[&id]
        .time_admission
        .as_ref()
        .unwrap()
        .before_acceptance());
    engine
        .inner
        .execute_slo_controller_wave(prepared)
        .await
        .unwrap();
    drop(bounded(session.frames.next()).await.unwrap());
    ready(&engine, &id).await;
    let accepted = prefill::captured(&engine, &executor).await;
    assert!(accepted.before_acceptance_candidate.is_none());
    assert!(
        engine.inner.propose_slo_time_admission(&accepted).is_none(),
        "accepted-only work must retain the ordinary finite SLO planner"
    );
    drop(accepted);
    Arc::get_mut(&mut engine.inner).unwrap().cost_runtime = None;
    tokio::time::advance(Duration::from_secs(31)).await;
    engine.inner.expire_strict_admissions().await.unwrap();
    assert_eq!(
        engine.inner.sequences.read()[&id]
            .slo
            .as_ref()
            .unwrap()
            .ingress(),
        ingress,
        "acceptance and later cost loss must preserve original ingress"
    );
    assert!(
        engine.inner.sequences.read().contains_key(&id),
        "accepted owner cannot expire"
    );
    let next = prefill::selected_after_retry(&engine, &executor, || {
        engine
            .inner
            .prepare_slo_controller(&ferrum_interfaces::BatchHint::simple(1))
    })
    .await;
    engine
        .inner
        .execute_slo_controller_wave(next)
        .await
        .unwrap();
    while let Some(frame) = bounded(session.frames.next()).await {
        drop(frame);
    }
    let completed = bounded(session.completion).await.unwrap();
    assert!(
        matches!(completed.payload(), OutputCompletion::Succeeded { usage, .. } if usage.completion_tokens == 2)
    );
    drop(completed);
    assert!(engine.inner.sequences.read().is_empty());
    engine.shutdown().await.unwrap();
}
