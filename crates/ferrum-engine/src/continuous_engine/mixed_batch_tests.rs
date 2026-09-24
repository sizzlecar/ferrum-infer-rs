use super::*;
use ferrum_interfaces::model_executor::PlanRuntimeMixedBatchOutcome;

#[path = "bounded_legacy_prefill_tests.rs"]
mod bounded_legacy_prefill_tests;
#[path = "bounded_wave_tests.rs"]
mod bounded_wave_tests;

#[derive(Clone, Copy)]
pub(super) enum MixedBehavior {
    Exact,
    NotSubmitted,
    SubmittedError,
    WrongDecodeCache,
    ShortPrefill,
    NarrowPrefill,
    YieldBeforeCompletion,
}

impl PlanRuntimeBatchDecodeTestExecutor {
    pub(super) async fn mock_mixed_batch(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
    ) -> Result<PlanRuntimeMixedBatchOutcome> {
        self.mixed_calls.fetch_add(1, Ordering::Relaxed);
        let behavior = *self.mixed_behavior.lock().unwrap();
        let Some(behavior) = behavior else {
            return Ok(PlanRuntimeMixedBatchOutcome::Unsupported);
        };
        if matches!(behavior, MixedBehavior::YieldBeforeCompletion) {
            // Model a completion that is not yet terminal on the first poll.
            tokio::task::yield_now().await;
        }
        match behavior {
            MixedBehavior::NotSubmitted => {
                let observed = ferrum_interfaces::vnext::CapacityAvailabilityEpoch::new(
                    ferrum_interfaces::vnext::CapacityAvailabilitySource::ActiveSequenceSlots,
                    1,
                )
                .unwrap();
                let wait = ferrum_interfaces::vnext::CapacityWaitCondition::from_observation(
                    47,
                    vec![observed],
                )
                .unwrap();
                return Ok(PlanRuntimeMixedBatchOutcome::NotSubmitted(
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
        let mut prefill_outputs = Vec::new();
        for input in prefills {
            let completed_chunk = if matches!(behavior, MixedBehavior::NarrowPrefill) {
                PrefillChunk::new(input.chunk.tokens_processed(), 1, input.input_tokens.len())?
            } else {
                input.chunk
            };
            let cache: Arc<dyn KvCacheHandle> = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
                input.request_id.clone(),
                1,
                completed_chunk.end(),
            ));
            let output = if completed_chunk.is_final() {
                let mut logits = vec![0.0; self.info().vocab_size];
                logits[6] = 1.0;
                PlanRuntimePrefillOutput::final_logits(
                    input.request_id.clone(),
                    completed_chunk.end(),
                    logits,
                    cache,
                )?
            } else {
                PlanRuntimePrefillOutput::intermediate(
                    input.request_id.clone(),
                    completed_chunk.end(),
                    cache,
                )
            };
            prefill_outputs.push(PlanRuntimePrefillCompletion::new(
                output,
                input.chunk,
                completed_chunk,
                u32::from(matches!(behavior, MixedBehavior::NarrowPrefill)),
            )?);
        }
        let mut decode_outputs = decodes
            .iter()
            .map(|input| {
                let mut logits = vec![0.0; self.info().vocab_size];
                logits[6] = 1.0;
                PlanRuntimeDecodeOutput::new(
                    ExecutorSamplingOutput::FullLogits(logits),
                    input.kv_cache.clone(),
                )
            })
            .collect::<Vec<_>>();
        if matches!(behavior, MixedBehavior::WrongDecodeCache) {
            decode_outputs[0].kv_cache = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
                RequestId::new(),
                1,
                1,
            ));
        }
        if matches!(behavior, MixedBehavior::ShortPrefill) {
            prefill_outputs.pop();
        }
        Ok(PlanRuntimeMixedBatchOutcome::Completed {
            prefills: prefill_outputs,
            decodes: decode_outputs,
        })
    }
}

type MixedFixture = (
    ContinuousBatchEngine,
    Arc<PlanRuntimeBatchDecodeTestExecutor>,
    ferrum_interfaces::BatchPlan,
    RequestId,
    RequestId,
);

fn prefill_test_frontier(
    engine: &ContinuousBatchEngine,
    id: &RequestId,
) -> ferrum_scheduler::implementations::continuous::planning_state::PlanningRequestState {
    use ferrum_scheduler::vnext::{AdmissionWakeEpochs, AdmissionWakeSnapshot};
    engine
        .inner
        .scheduler
        .planning_state(
            std::num::NonZeroUsize::new(32).unwrap(),
            AdmissionWakeSnapshot::new(
                AdmissionWakeEpochs::new(std::num::NonZeroU64::new(47).unwrap(), 0, 0, 0),
                &[],
            ),
        )
        .unwrap()
        .requests()
        .iter()
        .find(|row| &row.key.request_id == id)
        .unwrap()
        .clone()
}

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
    let (mut engine, scheduler, executor, tokenizer) =
        plan_runtime_batch_decode_test_engine_with_trace(
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
    let decode_id =
        install_profiled_plan_runtime_decode_frontier(&engine, &scheduler, tokenizer).await;
    let mut request = policy_request();
    request.prompt = "test".to_owned();
    request.sampling_params.max_tokens = 4;
    request
        .metadata
        .insert(PROMPT_TOKENS_METADATA_KEY.to_owned(), serde_json::json!(4));
    let prefill_id = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    let sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(5); 4],
        Some(engine.inner.tokenizer.clone()),
        Some(64),
    );
    engine
        .inner
        .sequences
        .write()
        .insert(prefill_id.clone(), sequence);
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
        assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
        let row = prefill_test_frontier(&engine, &prefill_id);
        assert_eq!(row.committed_output_tokens, usize::from(final_prefill));
        assert_eq!(row.computed_tokens, if final_prefill { 4 } else { 2 });
        assert_eq!(row.resident_tokens, row.computed_tokens);
        assert_eq!(row.scheduled_tokens, row.computed_tokens);
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
        assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
    }
}

#[tokio::test]
async fn mixed_batch_zero_submission_preserves_decode_scheduling_interval() {
    for behavior in [None, Some(MixedBehavior::NotSubmitted)] {
        let (engine, _, batch, prefill_id, decode_id) = mixed_fixture(behavior, true).await;
        assert!(engine.inner.sequences.read()[&decode_id].closes_pending_decode_scheduling());
        engine
            .inner
            .run_plan_runtime_mixed_batch(
                &batch,
                std::slice::from_ref(&prefill_id),
                std::slice::from_ref(&decode_id),
            )
            .await
            .unwrap();
        assert!(engine.inner.sequences.read()[&decode_id].closes_pending_decode_scheduling());
    }
}

#[tokio::test]
async fn mixed_batch_submitted_failure_never_replays() {
    let (engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture(Some(MixedBehavior::SubmittedError), true).await;
    engine.inner.process_batch(&batch).await.unwrap();
    assert!(!engine.inner.sequences.read().contains_key(&prefill_id));
    assert!(!engine.inner.sequences.read().contains_key(&decode_id));
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn mixed_batch_submitted_failure_terminates_peers_when_first_cleanup_fails() {
    let (engine, executor, batch, prefill_id, decode_id) =
        mixed_fixture(Some(MixedBehavior::SubmittedError), true).await;
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
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert!(engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .is_none());
}

#[tokio::test]
async fn mixed_batch_validates_all_outputs_before_publishing_any_tokens() {
    for behavior in [MixedBehavior::WrongDecodeCache, MixedBehavior::ShortPrefill] {
        let (engine, executor, batch, prefill_id, decode_id) =
            mixed_fixture(Some(behavior), true).await;
        assert!(engine
            .inner
            .run_plan_runtime_mixed_batch(
                &batch,
                std::slice::from_ref(&prefill_id),
                std::slice::from_ref(&decode_id)
            )
            .await
            .is_err());
        let sequences = engine.inner.sequences.read();
        assert!(sequences[&prefill_id].generated_tokens.is_empty());
        assert_eq!(sequences[&decode_id].generated_tokens.len(), 1);
        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    }
}
