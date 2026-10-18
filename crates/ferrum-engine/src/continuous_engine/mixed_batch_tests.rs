//! Controlled mixed-work protocol tests. Fake logits/cache handles are test
//! plumbing; these checks do not claim model or backend numerical coverage.

use super::controlled_token_fixture::{FixtureInput, FixtureOutcome};
use super::*;
use ferrum_execution::token_execution::{ExecutorSequenceIssuer, TokenWork};

#[derive(Clone, Copy)]
pub(super) enum MixedBehavior {
    Exact,
    NotSubmitted,
    SubmittedError,
    WrongDecodeCache,
    ShortPrefill,
}

impl PlanRuntimeBatchDecodeTestExecutor {
    pub(super) fn mock_mixed_tokens(
        &self,
        inputs: &[FixtureInput<'_, '_>],
    ) -> Result<FixtureOutcome> {
        self.mixed_calls.fetch_add(1, Ordering::Relaxed);
        let Some(behavior) = *self.mixed_behavior.lock().unwrap() else {
            return Ok(FixtureOutcome::Unsupported);
        };
        match behavior {
            MixedBehavior::NotSubmitted => {
                let observed = ferrum_runtime::CapacityAvailabilityEpoch::new(
                    ferrum_runtime::CapacityAvailabilitySource::ActiveSequenceSlots,
                    1,
                )
                .unwrap();
                let wait =
                    ferrum_runtime::CapacityWaitCondition::from_observation(47, vec![observed])
                        .unwrap();
                return Ok(FixtureOutcome::Deferred(
                    test_execution_capacity_deferral(
                        ExecutorAdmissionEpochs::new(std::num::NonZeroU64::new(47).unwrap(), 0, 0),
                        wait,
                        ExecutorExecutionCapacityStage::StepAdmission,
                    )
                    .into(),
                ));
            }
            MixedBehavior::SubmittedError => {
                return Err(FerrumError::backend(
                    "mixed device failure after submission",
                ))
            }
            _ => {}
        }
        let mut rows = inputs
            .iter()
            .map(|input| input.exact(6, self.info().vocab_size))
            .collect::<Vec<_>>();
        if matches!(behavior, MixedBehavior::WrongDecodeCache) {
            let index = inputs
                .iter()
                .position(|input| matches!(input.work.work(), TokenWork::Decode { .. }))
                .unwrap();
            // A real foreign opaque identity, never a fabricated cache string.
            rows[index].sequence = ExecutorSequenceIssuer::default().issue_sequence()?;
        }
        if matches!(behavior, MixedBehavior::ShortPrefill) {
            let index = inputs
                .iter()
                .rposition(|input| matches!(input.work.work(), TokenWork::Prefill(_)))
                .unwrap();
            rows.remove(index);
        }
        Ok(FixtureOutcome::Rows(rows))
    }
}

type MixedFixture = (
    ContinuousBatchEngine,
    Arc<PlanRuntimeBatchDecodeTestExecutor>,
    ferrum_interfaces::BatchPlan,
    RequestId,
    RequestId,
);

async fn mixed_fixture(behavior: Option<MixedBehavior>, final_prefill: bool) -> MixedFixture {
    mixed_fixture_with_execution(
        behavior,
        final_prefill,
        Some(ferrum_types::PrefillDecodeExecution::Mixed),
    )
    .await
}

async fn mixed_fixture_with_execution(
    behavior: Option<MixedBehavior>,
    final_prefill: bool,
    execution: Option<ferrum_types::PrefillDecodeExecution>,
) -> MixedFixture {
    let (mut engine, scheduler, executor, _) = plan_runtime_batch_decode_test_engine_with_trace(
        PlanRuntimeBatchDecodeBehavior::Exact,
        None,
    );
    if let Some(execution) = execution {
        Arc::get_mut(&mut engine.inner)
            .unwrap()
            .config
            .batching
            .prefill_decode_execution = execution;
    }
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .runtime
        .profile_detail = ferrum_types::ObservabilityProfileDetail::Latency;
    *executor.mixed_behavior.lock().unwrap() = behavior;
    let decode_id = install_mixed_request(&engine, &scheduler, vec![TokenId::new(5)], true).await;
    let first = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    assert_eq!(first.requests.len(), 1);
    engine.inner.process_batch(&first).await.unwrap();
    {
        let sequences = engine.inner.sequences.read();
        let decode = &sequences[&decode_id];
        assert!(decode.prefill_complete);
        assert_eq!(decode.generated_tokens, [TokenId::new(6)]);
        assert_eq!(
            decode
                .execution_frontier
                .as_ref()
                .unwrap()
                .consumed_tokens(),
            1
        );
    }
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    let prefill_id =
        install_mixed_request(&engine, &scheduler, vec![TokenId::new(5); 4], false).await;
    let mut batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .unwrap();
    assert_eq!(batch.requests.len(), 2);
    let scheduled = batch
        .requests
        .iter_mut()
        .find(|item| item.request.id == prefill_id)
        .unwrap();
    scheduled.tokens_to_process = Some(if final_prefill { 4 } else { 2 });
    (engine, executor, batch, prefill_id, decode_id)
}

async fn install_mixed_request(
    engine: &ContinuousBatchEngine,
    scheduler: &ContinuousBatchScheduler,
    tokens: Vec<TokenId>,
    timing: bool,
) -> RequestId {
    let mut request = policy_request();
    request.prompt = "test".to_owned();
    request.sampling_params.max_tokens = 4;
    request.evidence_request.capture_engine_token_timing = timing;
    request.metadata.insert(
        PROMPT_TOKENS_METADATA_KEY.to_owned(),
        serde_json::json!(tokens.len()),
    );
    let request_id = request.id.clone();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        tokens,
        Some(engine.inner.tokenizer.clone()),
        Some(64),
    );
    sequence.request_slot = Some(RequestSlotLease::open(&engine.inner, request_id.clone()));
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), sequence);
    assert!(
        engine
            .inner
            .probe_executor_prefill_admission_for_test(&request)
            .await
    );
    engine
        .inner
        .sequences
        .write()
        .get_mut(&request_id)
        .unwrap()
        .request_slot
        .as_mut()
        .unwrap()
        .admit(&engine.inner);
    scheduler.submit(request).await.unwrap();
    request_id
}

async fn close_mixed_requests(engine: &ContinuousBatchEngine, ids: &[&RequestId]) {
    for id in ids {
        // Explicit cancellation uses the same product close registry; do not
        // disarm leases or clear a fixture map to manufacture clean teardown.
        engine
            .inner
            .complete_request_for_test(id, FinishReason::Error)
            .await
            .unwrap();
    }
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
}

#[tokio::test]
async fn mixed_batch_default_split_does_not_invoke_mixed_executor() {
    let (engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture_with_execution(Some(MixedBehavior::SubmittedError), true, None).await;
    assert_eq!(
        engine.config().batching.prefill_decode_execution,
        ferrum_types::PrefillDecodeExecution::Split
    );
    engine.inner.process_batch(&batch).await.unwrap();
    let sequences = engine.inner.sequences.read();
    assert_eq!(sequences[&prefill_id].generated_tokens.len(), 1);
    assert_eq!(sequences[&decode_id].generated_tokens.len(), 2);
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
    drop(sequences);
    close_mixed_requests(&engine, &[&prefill_id, &decode_id]).await;
}

#[tokio::test]
async fn mixed_batch_commits_both_phases_and_preserves_nonfinal_prefill() {
    for final_prefill in [false, true] {
        let (engine, executor, batch, prefill_id, decode_id) =
            mixed_fixture(Some(MixedBehavior::Exact), final_prefill).await;
        engine.inner.process_batch(&batch).await.unwrap();
        let sequences = engine.inner.sequences.read();
        let prefill = &sequences[&prefill_id];
        assert_eq!(
            prefill.prefill_tokens_processed,
            if final_prefill { 4 } else { 2 }
        );
        assert_eq!(prefill.prefill_complete, final_prefill);
        assert_eq!(prefill.generated_tokens.len(), usize::from(final_prefill));
        assert_eq!(sequences[&decode_id].generated_tokens.len(), 2);
        let prefill_frontier = prefill.execution_frontier.as_ref().unwrap();
        assert_eq!(
            prefill_frontier.consumed_tokens(),
            if final_prefill { 4 } else { 2 }
        );
        assert_eq!(
            executor.token_frontier_for_request(&prefill_id).as_ref(),
            Some(prefill_frontier)
        );
        let decode_frontier = sequences[&decode_id].execution_frontier.as_ref().unwrap();
        assert_eq!(decode_frontier.consumed_tokens(), 2);
        assert_eq!(
            executor.token_frontier_for_request(&decode_id).as_ref(),
            Some(decode_frontier)
        );
        assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
        drop(sequences);
        close_mixed_requests(&engine, &[&prefill_id, &decode_id]).await;
    }
}

#[tokio::test]
async fn mixed_batch_zero_submission_falls_back_once() {
    for behavior in [None, Some(MixedBehavior::NotSubmitted)] {
        let (engine, executor, batch, prefill_id, decode_id) = mixed_fixture(behavior, true).await;
        engine.inner.process_batch(&batch).await.unwrap();
        let sequences = engine.inner.sequences.read();
        assert_eq!(sequences[&prefill_id].generated_tokens.len(), 1);
        assert_eq!(sequences[&decode_id].generated_tokens.len(), 2);
        assert_eq!(
            sequences[&prefill_id]
                .execution_frontier
                .as_ref()
                .unwrap()
                .consumed_tokens(),
            4
        );
        assert_eq!(
            sequences[&decode_id]
                .execution_frontier
                .as_ref()
                .unwrap()
                .consumed_tokens(),
            2
        );
        assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
        drop(sequences);
        close_mixed_requests(&engine, &[&prefill_id, &decode_id]).await;
    }
}

#[tokio::test]
async fn mixed_batch_zero_submission_preserves_decode_scheduling_interval() {
    for behavior in [None, Some(MixedBehavior::NotSubmitted)] {
        let (engine, executor, batch, prefill_id, decode_id) = mixed_fixture(behavior, true).await;
        assert!(engine.inner.sequences.read()[&decode_id].closes_pending_decode_scheduling());
        engine.inner.process_batch(&batch).await.unwrap();
        // A zero-submission mixed attempt is not a decode execution interval.
        // Its successful single-phase fallback records exactly one scheduling,
        // execution and postprocess span, then opens the next scheduling span.
        assert!(engine.inner.sequences.read()[&decode_id].closes_pending_decode_scheduling());
        let evidence = engine
            .inner
            .sequences
            .write()
            .get_mut(&decode_id)
            .unwrap()
            .take_execution_evidence()
            .unwrap()
            .unwrap();
        assert_eq!(evidence.output_token_ids.len(), 2);
        let timing = evidence.engine_token_timing.unwrap();
        assert_eq!(
            timing
                .decode_stage_intervals
                .iter()
                .map(|span| span.stage)
                .collect::<Vec<_>>(),
            [
                EngineDecodeStage::DecodeScheduling,
                EngineDecodeStage::DecodeExecution,
                EngineDecodeStage::DecodePostprocess,
            ]
        );
        assert_eq!(timing.token_commit_nanos_since_request_start.len(), 2);
        assert_eq!(
            timing.decode_stage_intervals[0].start_nanos_since_request_start,
            timing.token_commit_nanos_since_request_start[0]
        );
        assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
        close_mixed_requests(&engine, &[&prefill_id, &decode_id]).await;
    }
}

#[tokio::test]
async fn mixed_batch_submitted_failure_never_replays() {
    let (engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture(Some(MixedBehavior::SubmittedError), true).await;
    let prefill = executor.token_frontier_for_request(&prefill_id).unwrap();
    let decode = executor.token_frontier_for_request(&decode_id).unwrap();
    let error = engine.inner.process_batch(&batch).await.unwrap_err();
    assert!(error
        .to_string()
        .contains("mixed device failure after submission"));
    assert!(!engine.inner.sequences.read().contains_key(&prefill_id));
    assert!(!engine.inner.sequences.read().contains_key(&decode_id));
    assert!(prefill.sequence().observed_end().is_some());
    assert!(decode.sequence().observed_end().is_some());
    assert!(executor.token_frontier_for_request(&prefill_id).is_none());
    assert!(executor.token_frontier_for_request(&decode_id).is_none());
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn mixed_batch_submitted_failure_terminates_peers_when_first_cleanup_fails() {
    let (engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture(Some(MixedBehavior::SubmittedError), true).await;
    let prefill = executor.token_frontier_for_request(&prefill_id).unwrap();
    let decode = executor.token_frontier_for_request(&decode_id).unwrap();
    // Simulate a scheduler/engine terminal-state disagreement for the first
    // participant. Its scheduler completion will fail after engine removal.
    assert!(engine
        .inner
        .scheduler
        .cancel(prefill_id.clone())
        .await
        .unwrap());
    assert!(engine.inner.process_batch(&batch).await.is_err());
    assert!(!engine.inner.sequences.read().contains_key(&prefill_id));
    assert!(!engine.inner.sequences.read().contains_key(&decode_id));
    assert!(prefill.sequence().observed_end().is_some());
    assert!(decode.sequence().observed_end().is_some());
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert!(engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .is_none());
}

#[tokio::test]
async fn mixed_batch_rejects_invalid_phase_without_discarding_valid_peer() {
    for behavior in [MixedBehavior::WrongDecodeCache, MixedBehavior::ShortPrefill] {
        let (engine, executor, batch, prefill_id, decode_id) =
            mixed_fixture(Some(behavior), true).await;
        let prefill = executor.token_frontier_for_request(&prefill_id).unwrap();
        let decode = executor.token_frontier_for_request(&decode_id).unwrap();
        assert!(engine.inner.process_batch(&batch).await.is_err());
        let sequences = engine.inner.sequences.read();
        let (
            healthy_id,
            healthy_before,
            failed_id,
            failed_before,
            expected_tokens,
            expected_position,
        ) = match behavior {
            MixedBehavior::WrongDecodeCache => (&prefill_id, &prefill, &decode_id, &decode, 1, 4),
            MixedBehavior::ShortPrefill => (&decode_id, &decode, &prefill_id, &prefill, 2, 2),
            _ => unreachable!(),
        };
        let healthy = &sequences[healthy_id];
        assert_eq!(healthy.generated_tokens.len(), expected_tokens);
        let frontier = healthy.execution_frontier.as_ref().unwrap();
        assert_eq!(frontier.sequence(), healthy_before.sequence());
        assert_eq!(frontier.consumed_tokens(), expected_position);
        assert!(frontier.sequence().observed_end().is_none());
        assert_eq!(
            executor.token_frontier_for_request(healthy_id).as_ref(),
            Some(frontier)
        );
        assert!(!sequences.contains_key(failed_id));
        assert!(failed_before.sequence().observed_end().is_some());
        assert!(executor.token_frontier_for_request(failed_id).is_none());
        assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
        drop(sequences);
        close_mixed_requests(&engine, &[healthy_id]).await;
    }
}
