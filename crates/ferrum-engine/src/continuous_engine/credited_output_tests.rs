use super::*;
use crate::continuous_engine::output_flow_runtime::{
    OutputReadiness, OutputReadinessState, ReadyOutputGrant,
};
use ferrum_interfaces::{
    output_credit::{OutputCreditAmount, OutputCreditPool},
    output_flow::{CreditedOutputSession, OutputCompletion, OutputProjectionContract},
    tokenizer::{BoundedDecodeBound, BoundedDecodeError, DecodedTextBound},
    InferenceRequestContext,
};
use futures::{Future, StreamExt};
use std::num::NonZeroUsize;
use std::sync::atomic::AtomicUsize;

#[path = "credited_chat_tests.rs"]
mod chat;

#[path = "credited_prepared_completion_tests.rs"]
mod prepared_completion;

#[derive(Default)]
struct CreditedLegacyDecodeCalls {
    decode: AtomicUsize,
    incremental: AtomicUsize,
}

impl CreditedLegacyDecodeCalls {
    fn reset(&self) {
        self.decode.store(0, Ordering::Relaxed);
        self.incremental.store(0, Ordering::Relaxed);
    }

    fn assert_unused(&self) {
        assert_eq!(
            self.decode.load(Ordering::Relaxed),
            0,
            "legacy decode called"
        );
        assert_eq!(
            self.incremental.load(Ordering::Relaxed),
            0,
            "legacy incremental decode called"
        );
    }
}

/// This fixture's decoder only concatenates literal vocabulary entries. The
/// four-byte bound is derived from those entries, with no hidden postprocessor.
struct CreditedPolicyTokenizer(PolicyTokenizer, Arc<CreditedLegacyDecodeCalls>);
impl Tokenizer for CreditedPolicyTokenizer {
    fn encode(&self, text: &str, special: bool) -> Result<Vec<TokenId>> {
        self.0.encode(text, special)
    }
    fn decode(&self, tokens: &[TokenId], special: bool) -> Result<String> {
        self.1.decode.fetch_add(1, Ordering::Relaxed);
        self.0
            .decode(tokens, special)
            .map(|text| text.into_boxed_str().into_string())
    }
    fn decode_incremental(&self, previous: &[TokenId], token: TokenId) -> Result<String> {
        self.1.incremental.fetch_add(1, Ordering::Relaxed);
        self.0.decode_incremental(previous, token)
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        self.0.special_tokens()
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.0.token_id(text)
    }
    fn token_text(&self, token: TokenId) -> Option<&str> {
        self.0.token_text(token)
    }
    fn info(&self) -> TokenizerInfo {
        self.0.info()
    }
    fn decoded_text_bound(&self) -> Option<DecodedTextBound> {
        Some(DecodedTextBound::new(NonZeroUsize::new(4).unwrap()))
    }
    fn bounded_decode_bound(&self) -> Option<BoundedDecodeBound> {
        // This fixture deliberately declares only its literal test vocabulary;
        // the parent's synthetic byte/reasoning cases have separate semantics.
        self.0
            .texts
            .iter()
            .flatten()
            .all(|text| matches!(text.as_str(), "test" | "ok"))
            .then(|| BoundedDecodeBound::new(NonZeroUsize::new(4).unwrap(), 0))
    }
    fn bounded_token_bytes_bound(&self) -> Option<NonZeroUsize> {
        self.bounded_decode_bound()
            .map(|_| NonZeroUsize::new(4).unwrap())
    }
    fn token_bytes_bounded_into(
        &self,
        token: TokenId,
        output: &mut [u8],
    ) -> std::result::Result<Option<usize>, BoundedDecodeError> {
        let required = self
            .bounded_token_bytes_bound()
            .ok_or(BoundedDecodeError::Unsupported)?
            .get();
        if output.len() < required {
            return Err(BoundedDecodeError::InsufficientScratch {
                required,
                available: output.len(),
            });
        }
        let Some(text) = self.0.token_text(token) else {
            return Ok(None);
        };
        output[..text.len()].copy_from_slice(text.as_bytes());
        Ok(Some(text.len()))
    }
    fn decode_bounded_into(
        &self,
        tokens: &[TokenId],
        skip_special: bool,
        _scratch: &mut [u8],
        output: &mut String,
    ) -> std::result::Result<(), BoundedDecodeError> {
        let required = self
            .bounded_decode_bound()
            .ok_or(BoundedDecodeError::Unsupported)?
            .requirements(tokens.len())?;
        if output.capacity() < required.text_bytes {
            return Err(BoundedDecodeError::InsufficientOutput {
                required: required.text_bytes,
                available: output.capacity(),
            });
        }
        output.clear();
        for token in tokens {
            if skip_special && self.0.hidden_decode_token_ids.contains(&token.get()) {
                continue;
            }
            if let Some(text) = self.0.token_text(*token) {
                output.push_str(text);
            }
        }
        Ok(())
    }
}

type CreditedFixture = (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<PlanRuntimeBatchDecodeTestExecutor>,
);
fn credited_fixture() -> CreditedFixture {
    credited_counted_fixture().0
}

fn credited_counted_fixture() -> (CreditedFixture, Arc<CreditedLegacyDecodeCalls>) {
    let (mut engine, scheduler, executor, _) =
        plan_runtime_batch_decode_test_engine(PlanRuntimeBatchDecodeBehavior::Exact);
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    let calls = Arc::new(CreditedLegacyDecodeCalls::default());
    inner.tokenizer = Arc::new(CreditedPolicyTokenizer(
        PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]),
        calls.clone(),
    ));
    // One reusable data event and one independently reserved terminal event.
    inner
        .config
        .scheduler
        .slo
        .output
        .max_queued_events_per_request = NonZeroUsize::new(2).unwrap();
    // These tests drive the real scheduler/process_batch boundary explicitly.
    // Prevent infer_credited_stream from creating a competing background loop.
    inner.bg_loop_spawned.store(true, Ordering::Release);
    ((engine, scheduler, executor), calls)
}

async fn credited_bound<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(3), future)
        .await
        .expect("credited engine made no progress")
}

async fn credited_submit(
    engine: &ContinuousBatchEngine,
    max_tokens: usize,
) -> (RequestId, CreditedOutputSession) {
    let mut request = policy_request();
    request.stream = true;
    request.sampling_params.max_tokens = max_tokens;
    let id = request.id.clone();
    let session = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await
        .unwrap();
    (id, session)
}

fn credited_pool(engine: &ContinuousBatchEngine) -> &OutputCreditPool {
    engine
        .inner
        .output_credit_pool
        .get()
        .unwrap()
        .as_ref()
        .unwrap()
}

fn credited_state(engine: &ContinuousBatchEngine, id: &RequestId) -> OutputReadinessState {
    engine.inner.sequences.read()[id]
        .credited_output
        .as_ref()
        .unwrap()
        .port
        .readiness()
}

async fn credited_wait(
    engine: &ContinuousBatchEngine,
    id: &RequestId,
    predicate: impl Fn(OutputReadinessState) -> bool,
) {
    let mut changes = engine.inner.sequences.read()[id]
        .credited_output
        .as_ref()
        .unwrap()
        .port
        .subscribe();
    credited_bound(async {
        while !predicate(credited_state(engine, id)) {
            changes.changed().await.unwrap();
        }
    })
    .await;
}

async fn credited_ready(engine: &ContinuousBatchEngine, id: &RequestId) {
    credited_wait(engine, id, |state| state == OutputReadinessState::Ready).await;
}

fn credited_take(engine: &ContinuousBatchEngine, id: &RequestId) -> ReadyOutputGrant {
    match engine.inner.sequences.read()[id]
        .credited_output
        .as_ref()
        .unwrap()
        .port
        .try_take()
    {
        OutputReadiness::Ready(grant) => grant,
        _ => panic!("fixture requires a real prepared grant"),
    }
}

async fn credited_next(
    scheduler: &ContinuousBatchScheduler,
    count: usize,
) -> ferrum_interfaces::BatchPlan {
    scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(count))
        .await
        .unwrap()
}

async fn credited_drained(engine: &ContinuousBatchEngine) {
    let pool = credited_pool(engine);
    let mut wake = pool.subscribe();
    credited_bound(async {
        while pool.snapshot().retained_accounts != 0 {
            wake.changed().await.unwrap();
        }
    })
    .await;
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    assert_eq!(pool.snapshot().terminal_held, OutputCreditAmount::ZERO);
}

async fn credited_cancel(engine: &ContinuousBatchEngine, session: CreditedOutputSession) {
    drop(session.frames);
    engine.inner.run_iteration().await.unwrap();
    drop(credited_bound(session.completion).await.unwrap());
}

#[tokio::test]
async fn credited_engine_budget_rejection_submits_no_model_work() {
    let (mut engine, scheduler, executor) = credited_fixture();
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .output
        .max_projection_bytes_per_request = NonZeroUsize::new(1).unwrap();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 4;
    let result = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await;
    assert!(result.is_err());
    assert_eq!(executor.inner.prefill_count(), 0);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 0);
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(credited_pool(&engine).snapshot().retained_accounts, 0);
}

#[tokio::test]
async fn credited_engine_rejects_delayed_completion_before_decode_or_admission() {
    let ((engine, scheduler, executor), calls) = credited_counted_fixture();
    let mut request = policy_request();
    request.stream = true;
    request.sampling_params.max_tokens = 4;
    request.sampling_params.response_completion_boundary =
        ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "test".to_owned(),
            alternate_envelope: None,
        };
    let result = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await;
    assert!(
        result.is_err(),
        "unproved completion decoder must not be admitted"
    );
    calls.assert_unused();
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(executor.inner.prefill_count(), 0);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 0);
    if let Some(Ok(pool)) = engine.inner.output_credit_pool.get() {
        assert_eq!(pool.snapshot().retained_accounts, 0);
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    }
}

#[tokio::test]
async fn credited_engine_greedy_error_and_resample_diagnostics_never_decode() {
    let ((engine, _scheduler, _executor), calls) = credited_counted_fixture();
    let (id, session) = credited_submit(&engine, 4).await;
    credited_ready(&engine, &id).await;
    // Admission may construct a shared vocabulary policy with the ordinary
    // decoder. The live credited request's error handling may not call it.
    calls.reset();
    {
        let mut sequences = engine.inner.sequences.write();
        let sequence = sequences.get_mut(&id).unwrap();
        let forbidden = sequence
            .forbidden_token_ids
            .iter()
            .next()
            .copied()
            .expect("fixture has an UNK/PAD generation prohibition");
        assert!(sequence.can_use_model_greedy_argmax());
        let result = sequence.validate_and_commit_model_greedy_argmax_token(
            Some(engine.inner.tokenizer.as_ref()),
            TokenId::new(forbidden),
        );
        assert!(result.is_err());
        assert!(sequence.generated_tokens.is_empty());
        calls.assert_unused();

        // Exercise the actual failure diagnostic with known and unknown IDs
        // beyond its retained tail. No sampling/output frontier is changed.
        sequence.log_forbidden_decode_resample_failure(
            Some(engine.inner.tokenizer.as_ref()),
            0,
            &[
                TokenId::new(5),
                TokenId::new(6),
                TokenId::new(forbidden),
                TokenId::MAX,
                TokenId::new(5),
                TokenId::new(6),
                TokenId::new(forbidden),
                TokenId::MAX,
                TokenId::new(5),
            ],
        );
        calls.assert_unused();
        assert!(sequence.generated_tokens.is_empty());
    }
    credited_cancel(&engine, session).await;
    credited_drained(&engine).await;
    calls.assert_unused();
}

#[tokio::test]
async fn credited_engine_without_grant_defers_before_prefill_and_reuses_original_credit() {
    let (engine, scheduler, executor) = credited_fixture();
    let (id, session) = credited_submit(&engine, 4).await;
    credited_ready(&engine, &id).await;
    let held = credited_take(&engine, &id);
    let before = credited_pool(&engine).snapshot().data_used;
    let batch = credited_next(&scheduler, 1).await;
    engine.inner.process_batch(&batch).await.unwrap();
    assert_eq!(executor.inner.prefill_count(), 0);
    assert!(engine.inner.sequences.read()[&id]
        .generated_tokens
        .is_empty());
    assert_eq!(
        scheduler
            .trace_snapshot()
            .execution_readiness_blocked_prefill_len,
        1
    );
    assert_eq!(credited_pool(&engine).snapshot().data_used, before);
    held.return_unsubmitted();
    credited_ready(&engine, &id).await;
    assert_eq!(credited_pool(&engine).snapshot().data_used, before);
    engine.inner.refresh_credited_output_readiness();
    let replanned = credited_next(&scheduler, 1).await;
    assert_eq!(replanned.requests[0].request.id, id);
    engine.inner.process_batch(&replanned).await.unwrap();
    assert_eq!(executor.inner.prefill_count(), 1);
    assert_eq!(
        engine.inner.sequences.read()[&id].generated_tokens,
        vec![TokenId::new(6)]
    );
    credited_cancel(&engine, session).await;
    credited_drained(&engine).await;
}

#[tokio::test]
async fn credited_engine_slow_output_replans_and_healthy_request_really_commits() {
    let (engine, scheduler, executor) = credited_fixture();
    let (slow_id, mut slow) = credited_submit(&engine, 4).await;
    credited_ready(&engine, &slow_id).await;
    let first = credited_next(&scheduler, 1).await;
    engine.inner.process_batch(&first).await.unwrap();
    credited_wait(&engine, &slow_id, |state| {
        matches!(state, OutputReadinessState::OutputBlocked(_))
    })
    .await;
    let (healthy_id, mut healthy) = credited_submit(&engine, 2).await;
    credited_ready(&engine, &healthy_id).await;
    let combined = credited_next(&scheduler, 2).await;
    assert_eq!(combined.requests.len(), 2);
    assert!(combined
        .requests
        .iter()
        .any(|scheduled| scheduled.request.id == slow_id));
    assert!(combined
        .requests
        .iter()
        .any(|scheduled| scheduled.request.id == healthy_id));
    engine.inner.process_batch(&combined).await.unwrap();
    assert_eq!(
        executor.inner.prefill_count(),
        1,
        "blocked cohort performs no partial submission"
    );
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert!(engine.inner.sequences.read()[&healthy_id]
        .generated_tokens
        .is_empty());
    assert_eq!(
        scheduler
            .trace_snapshot()
            .execution_readiness_blocked_decode_len,
        1
    );
    credited_ready(&engine, &healthy_id).await;
    engine.inner.refresh_credited_output_readiness();
    let replanned = credited_next(&scheduler, 2).await;
    assert_eq!(replanned.requests.len(), 1);
    assert_eq!(replanned.requests[0].request.id, healthy_id);
    engine.inner.process_batch(&replanned).await.unwrap();
    assert_eq!(executor.inner.prefill_count(), 2);
    assert_eq!(
        engine.inner.sequences.read()[&slow_id]
            .generated_tokens
            .len(),
        1
    );
    assert_eq!(
        engine.inner.sequences.read()[&healthy_id].generated_tokens,
        vec![TokenId::new(6)]
    );
    let healthy_first = credited_bound(healthy.frames.next()).await.unwrap();
    assert_eq!(healthy_first.wire().payload(), b"ok");
    drop(healthy_first);
    credited_ready(&engine, &healthy_id).await;
    let healthy_decode = credited_next(&scheduler, 2).await;
    assert_eq!(healthy_decode.requests.len(), 1);
    engine.inner.process_batch(&healthy_decode).await.unwrap();
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
    assert!(!engine.inner.sequences.read().contains_key(&healthy_id));
    assert_eq!(
        engine.inner.sequences.read()[&slow_id]
            .generated_tokens
            .len(),
        1
    );
    drop(credited_bound(healthy.frames.next()).await.unwrap());
    assert!(
        credited_bound(healthy.frames.next())
            .await
            .unwrap()
            .metadata()
            .terminal
    );
    let healthy_completion = credited_bound(healthy.completion).await.unwrap();
    match healthy_completion.payload() {
        OutputCompletion::Succeeded {
            history: Some(history),
            usage,
            ..
        } => {
            assert_eq!(history.text, "okok");
            assert_eq!(usage.completion_tokens, 2);
        }
        _ => panic!("healthy request failed while slow output was full"),
    }
    drop((healthy_completion, healthy.frames));
    let slow_first = credited_bound(slow.frames.next()).await.unwrap();
    assert_eq!(slow_first.wire().payload(), b"ok");
    drop(slow_first);
    credited_ready(&engine, &slow_id).await;
    engine.inner.refresh_credited_output_readiness();
    let resumed = credited_next(&scheduler, 1).await;
    assert_eq!(resumed.requests[0].request.id, slow_id);
    engine.inner.process_batch(&resumed).await.unwrap();
    assert_eq!(
        engine.inner.sequences.read()[&slow_id]
            .generated_tokens
            .len(),
        2
    );
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 2);
    credited_cancel(&engine, slow).await;
    credited_drained(&engine).await;
}

#[tokio::test]
async fn credited_engine_one_token_terminal_settles_grant_before_sequence_removal() {
    let ((engine, scheduler, executor), calls) = credited_counted_fixture();
    let (id, mut session) = credited_submit(&engine, 1).await;
    credited_ready(&engine, &id).await;
    calls.reset();
    let batch = credited_next(&scheduler, 1).await;
    engine.inner.process_batch(&batch).await.unwrap();
    assert_eq!(executor.inner.prefill_count(), 1);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert!(!engine.inner.sequences.read().contains_key(&id));
    let data = credited_bound(session.frames.next()).await.unwrap();
    assert_eq!(data.wire().payload(), b"ok");
    assert_eq!(data.metadata().token, Some(TokenId::new(6)));
    drop(data);
    let terminal = credited_bound(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    assert_eq!(terminal.metadata().generated_tokens, 1);
    drop(terminal);
    let completion = credited_bound(session.completion).await.unwrap();
    match completion.payload() {
        OutputCompletion::Succeeded {
            history: Some(history),
            reason,
            usage,
        } => {
            assert_eq!(history.text, "ok");
            assert_eq!(history.tokens, vec![TokenId::new(6)]);
            assert_eq!(*reason, FinishReason::Length);
            assert_eq!(usage.completion_tokens, 1);
        }
        _ => panic!("terminal token lost its pre-wave grant"),
    }
    assert!(credited_pool(&engine).snapshot().data_used.projection_bytes > 0);
    drop((completion, session.frames));
    credited_drained(&engine).await;
    calls.assert_unused();
}

#[tokio::test]
async fn credited_engine_deferred_wave_preserves_credit_and_frontiers_until_fresh_plan() {
    let (mut engine, scheduler, executor) = credited_fixture();
    let (decode_id, mut decode) = credited_submit(&engine, 4).await;
    credited_ready(&engine, &decode_id).await;
    engine
        .inner
        .process_batch(&credited_next(&scheduler, 1).await)
        .await
        .unwrap();
    drop(credited_bound(decode.frames.next()).await.unwrap());
    credited_ready(&engine, &decode_id).await;
    let (prefill_id, prefill) = credited_submit(&engine, 4).await;
    credited_ready(&engine, &prefill_id).await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.batching.prefill_decode_execution = ferrum_types::PrefillDecodeExecution::Mixed;
    *executor.mixed_behavior.lock().unwrap() = Some(mixed_batch_tests::MixedBehavior::NotSubmitted);
    let before = credited_pool(&engine).snapshot().data_used;
    let batch = credited_next(&scheduler, 2).await;
    engine.inner.process_batch(&batch).await.unwrap();
    credited_ready(&engine, &decode_id).await;
    credited_ready(&engine, &prefill_id).await;
    assert_eq!(credited_pool(&engine).snapshot().data_used, before);
    assert_eq!(
        engine.inner.sequences.read()[&decode_id]
            .generated_tokens
            .len(),
        1
    );
    assert!(engine.inner.sequences.read()[&prefill_id]
        .generated_tokens
        .is_empty());
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    *executor.mixed_behavior.lock().unwrap() = Some(mixed_batch_tests::MixedBehavior::Exact);
    let replanned = credited_next(&scheduler, 2).await;
    engine.inner.process_batch(&replanned).await.unwrap();
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 2);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
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
    drop((decode.frames, prefill.frames));
    engine.inner.run_iteration().await.unwrap();
    drop(credited_bound(decode.completion).await.unwrap());
    drop(credited_bound(prefill.completion).await.unwrap());
    credited_drained(&engine).await;
}

#[tokio::test]
async fn credited_engine_disconnect_keeps_history_charged_until_sequence_cleanup() {
    let (engine, scheduler, executor) = credited_fixture();
    let (id, session) = credited_submit(&engine, 4).await;
    credited_ready(&engine, &id).await;
    engine
        .inner
        .process_batch(&credited_next(&scheduler, 1).await)
        .await
        .unwrap();
    let projection = credited_pool(&engine).snapshot().data_used.projection_bytes;
    assert!(projection > 0);
    drop(session.frames);
    credited_wait(&engine, &id, |state| {
        matches!(state, OutputReadinessState::Closing(_))
    })
    .await;
    assert_eq!(engine.inner.sequences.read()[&id].generated_tokens.len(), 1);
    assert_eq!(
        credited_pool(&engine).snapshot().data_used.projection_bytes,
        projection
    );
    let mut completion = session.completion;
    assert!(futures::poll!(&mut completion).is_pending());
    engine.inner.run_iteration().await.unwrap();
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert_eq!(scheduler.trace_phase(&id), None);
    assert_eq!(executor.released_cache_count.load(Ordering::Relaxed), 1);
    let completion = credited_bound(completion).await.unwrap();
    assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    assert_eq!(
        credited_pool(&engine).snapshot().data_used.projection_bytes,
        projection
    );
    drop(completion);
    credited_drained(&engine).await;
}

#[tokio::test]
async fn credited_engine_submitted_failure_cleans_both_participants_without_replay() {
    let (mut engine, scheduler, executor) = credited_fixture();
    let (decode_id, mut decode) = credited_submit(&engine, 4).await;
    credited_ready(&engine, &decode_id).await;
    engine
        .inner
        .process_batch(&credited_next(&scheduler, 1).await)
        .await
        .unwrap();
    drop(credited_bound(decode.frames.next()).await.unwrap());
    credited_ready(&engine, &decode_id).await;
    let (prefill_id, mut prefill) = credited_submit(&engine, 4).await;
    credited_ready(&engine, &prefill_id).await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.batching.prefill_decode_execution = ferrum_types::PrefillDecodeExecution::Mixed;
    *executor.mixed_behavior.lock().unwrap() =
        Some(mixed_batch_tests::MixedBehavior::SubmittedError);
    let batch = credited_next(&scheduler, 2).await;
    assert!(engine.inner.process_batch(&batch).await.is_err());
    assert!(!engine.inner.sequences.read().contains_key(&decode_id));
    assert!(!engine.inner.sequences.read().contains_key(&prefill_id));
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert_eq!(executor.inner.prefill_count(), 1);
    assert!(scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .is_none());
    assert!(engine.inner.process_batch(&batch).await.is_err());
    assert_eq!(executor.mixed_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
    assert!(
        credited_bound(decode.frames.next())
            .await
            .unwrap()
            .metadata()
            .terminal
    );
    assert!(
        credited_bound(prefill.frames.next())
            .await
            .unwrap()
            .metadata()
            .terminal
    );
    let decode_completion = credited_bound(decode.completion).await.unwrap();
    let prefill_completion = credited_bound(prefill.completion).await.unwrap();
    assert!(matches!(
        decode_completion.payload(),
        OutputCompletion::Failed(_)
    ));
    assert!(matches!(
        prefill_completion.payload(),
        OutputCompletion::Failed(_)
    ));
    drop((
        decode_completion,
        prefill_completion,
        decode.frames,
        prefill.frames,
    ));
    credited_drained(&engine).await;
}

#[tokio::test]
async fn credited_engine_shutdown_cleans_model_without_waiting_for_full_output_queue() {
    let (engine, scheduler, executor) = credited_fixture();
    let (id, mut session) = credited_submit(&engine, 4).await;
    credited_ready(&engine, &id).await;
    engine
        .inner
        .process_batch(&credited_next(&scheduler, 1).await)
        .await
        .unwrap();
    credited_wait(&engine, &id, |state| {
        matches!(state, OutputReadinessState::OutputBlocked(_))
    })
    .await;
    credited_bound(engine.shutdown()).await.unwrap();
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert_eq!(scheduler.trace_phase(&id), None);
    assert_eq!(executor.released_cache_count.load(Ordering::Relaxed), 1);
    // Shutdown is complete even though no wire frame has been consumed.
    let completion = credited_bound(session.completion).await.unwrap();
    assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    let data = credited_bound(session.frames.next()).await.unwrap();
    assert_eq!(data.wire().payload(), b"ok");
    let terminal = credited_bound(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    drop((terminal, completion, session.frames));
    assert_eq!(credited_pool(&engine).snapshot().data_used.events, 1);
    drop(data);
    credited_drained(&engine).await;
}

#[tokio::test]
async fn credited_engine_rejects_requests_after_shutdown_before_budget_or_execution() {
    let (engine, scheduler, executor) = credited_fixture();
    engine.shutdown().await.unwrap();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let result = engine
        .infer_credited_stream(
            request,
            InferenceRequestContext::capture(),
            Arc::new(OutputProjectionContract::cli_text()),
        )
        .await;
    let error = match result {
        Ok(_) => panic!("shutdown engine accepted a credited request"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("shutting down"));
    assert!(engine.inner.output_credit_pool.get().is_none());
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(executor.inner.prefill_count(), 0);
    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 0);
}
