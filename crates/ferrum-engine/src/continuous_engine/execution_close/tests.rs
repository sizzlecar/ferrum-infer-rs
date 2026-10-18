use super::*;
use ferrum_execution::{
    model_executor::{ExecutorCapabilities, ExecutorStatus},
    token_execution::ExecutorSequenceIssuer,
};
use ferrum_kv::{PagedKvCacheConfig, PagedKvCacheManager};
use ferrum_models::test_support::{tiny_llama_executor, TinyLlamaConfig, TinyTokenizer};
use std::sync::atomic::{AtomicU64, AtomicUsize};

#[path = "tests/recompute_failure.rs"]
mod recompute_failure;

struct CloseProbe {
    inner: Arc<dyn ModelExecutor>,
    issuer: ExecutorSequenceIssuer,
    terminal: Mutex<Option<tokio::sync::oneshot::Receiver<Result<ExecutionTerminalOutcome>>>>,
    calls: AtomicUsize,
    initial_error: bool,
    end_then_error: AtomicBool,
    capacity_epoch: AtomicU64,
    immediate_ends: Mutex<std::collections::HashSet<ExecutorSequence>>,
    retained_cpu_model: Mutex<
        Option<ferrum_models::models::LlamaFamilyModel<ferrum_kernels::backend::cpu::CpuBackend>>,
    >,
}

impl CloseProbe {
    fn close(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.immediate_ends.lock().remove(sequence) {
            return Ok(ExecutionCloseOutcome::ended(self.issuer.ended(sequence)?));
        }
        if self.end_then_error.swap(false, Ordering::SeqCst) {
            self.issuer.ended(sequence)?;
            return Err(FerrumError::backend(
                "close failed after publishing actual terminal fact",
            ));
        }
        if let Some(ended) = sequence.observed_end() {
            return Ok(ExecutionCloseOutcome::ended(ended));
        }
        let receiver = self.terminal.lock().take().expect("close invoked once");
        let held = self
            .issuer
            .held(self.issuer.recovery_key(sequence)?, async move {
                receiver
                    .await
                    .map_err(|_| FerrumError::internal("test terminal sender lost"))?
            })?;
        let outcome = ExecutionCloseOutcome::held(held);
        Ok(if self.initial_error {
            outcome.with_error(FerrumError::backend("initial publication failure"))
        } else {
            outcome
        })
    }
}

#[async_trait::async_trait]
impl ModelExecutor for CloseProbe {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        _reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.close(sequence)
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.close(sequence)
    }
    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<ferrum_runtime::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        let epoch = self.capacity_epoch.load(Ordering::Acquire);
        availability.clear();
        availability.push(
            ferrum_runtime::CapacityAvailabilityEpoch::new(
                ferrum_runtime::CapacityAvailabilitySource::ActiveSequenceSlots,
                epoch,
            )
            .unwrap(),
        );
        Ok(Some(ExecutorAdmissionEpochs {
            coordinator_id: std::num::NonZeroU64::new(41).unwrap(),
            release_epoch: epoch,
            capacity_epoch: epoch,
        }))
    }
}

fn fixture(
    initial_error: bool,
) -> (
    ContinuousBatchEngine,
    Arc<CloseProbe>,
    ExecutorSequence,
    RequestId,
    RequestSlotLease,
    tokio::sync::oneshot::Sender<Result<ExecutionTerminalOutcome>>,
) {
    let cfg = TinyLlamaConfig::default();
    fixture_with_inner(initial_error, tiny_llama_executor(&cfg))
}

fn fixture_with_inner(
    initial_error: bool,
    inner: Arc<dyn ModelExecutor>,
) -> (
    ContinuousBatchEngine,
    Arc<CloseProbe>,
    ExecutorSequence,
    RequestId,
    RequestSlotLease,
    tokio::sync::oneshot::Sender<Result<ExecutionTerminalOutcome>>,
) {
    let cfg = TinyLlamaConfig::default();
    let (sender, terminal) = tokio::sync::oneshot::channel();
    let executor = Arc::new(CloseProbe {
        inner,
        issuer: ExecutorSequenceIssuer::new(),
        terminal: Mutex::new(Some(terminal)),
        calls: AtomicUsize::new(0),
        initial_error,
        end_then_error: AtomicBool::new(false),
        capacity_epoch: AtomicU64::new(1),
        immediate_ends: Mutex::new(std::collections::HashSet::new()),
        retained_cpu_model: Mutex::new(None),
    });
    let sequence = executor.issuer.issue_sequence().unwrap();
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let kv = Arc::new(
        PagedKvCacheManager::new(
            Device::CPU,
            PagedKvCacheConfig {
                block_size: 16,
                max_gpu_blocks: 16,
                max_cpu_blocks: 0,
                enable_swapping: false,
                num_layers: cfg.num_layers,
                num_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        Arc::new(TinyTokenizer::new(cfg.vocab_size)),
        Arc::new(crate::registry::GreedySampler),
        kv,
        executor.clone(),
        Arc::new(crate::tensor_factory::CandleTensorFactory::new(Device::CPU)),
    )
    .unwrap();
    let request_id = RequestId::new();
    let mut slot = RequestSlotLease::open(&engine.inner, request_id.clone());
    slot.admit(&engine.inner);
    (engine, executor, sequence, request_id, slot, sender)
}

async fn install_recompute_request(
    engine: &ContinuousBatchEngine,
    executor: &CloseProbe,
    identity: &ExecutorSequence,
    request_id: &RequestId,
    slot: RequestSlotLease,
) -> PressureYieldTransaction {
    use ferrum_runtime::{
        CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition,
    };
    let mut request = InferenceRequest::new("recompute", "tiny-llama-test");
    request.id = request_id.clone();
    request
        .metadata
        .insert(PROMPT_TOKENS_METADATA_KEY.into(), serde_json::json!(1));
    engine
        .inner
        .scheduler
        .submit(request.clone())
        .await
        .unwrap();
    let batch = engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    assert_eq!(batch.requests.len(), 1);
    engine.inner.scheduler.mark_prefill_complete(request_id, 1);
    let mut state = SequenceState::new(request, vec![TokenId(1)]);
    state.execution_frontier = Some(executor.issuer.frontier(identity, 1).unwrap());
    state.prefill_complete = true;
    state.prefill_tokens_processed = 1;
    state.generated_tokens.push(TokenId(2));
    state.phase = RequestPhase::Decoding;
    state.request_slot = Some(slot);
    engine
        .inner
        .sequences
        .write()
        .insert(request_id.clone(), state);
    let source = CapacityAvailabilitySource::ActiveSequenceSlots;
    let condition = CapacityWaitCondition::from_observation(
        41,
        vec![CapacityAvailabilityEpoch::new(source, 1).unwrap()],
    )
    .unwrap();
    let release = ExecutionCapacityReleaseSnapshot::new([(request_id.clone(), vec![source])]);
    let action = engine
        .inner
        .scheduler
        .defer_decode_for_execution_capacity(
            std::slice::from_ref(request_id),
            AdmissionDeferral::new(
                DeferredAction::WaitForRelease,
                AdmissionWakeEpochs::new(std::num::NonZeroU64::new(41).unwrap(), 1, 1, 0),
                condition,
            ),
            &release,
        )
        .unwrap();
    let ExecutionCapacityAction::YieldPlanned { transaction } = action else {
        panic!("the sole exact-source owner must yield for recompute")
    };
    transaction
}

#[tokio::test]
async fn execution_close_recompute_waits_for_ended_before_requeue_and_release() {
    let (engine, executor, identity, request_id, slot, sender) = fixture(false);
    let transaction =
        install_recompute_request(&engine, &executor, &identity, &request_id, slot).await;
    assert!(!engine
        .inner
        .execute_capacity_yield(&transaction, 1, None)
        .await
        .unwrap());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert!(engine
        .inner
        .execution_closes
        .contains_incarnation(&identity));
    assert_eq!(
        engine.inner.scheduler.trace_phase(&request_id),
        Some(RequestPhase::Decoding)
    );
    assert!(engine
        .inner
        .scheduler
        .all_active_execution_readiness_blocked());
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&request_id];
        assert_eq!(
            sequence.execution_frontier.as_ref().unwrap().sequence(),
            &identity
        );
        assert!(sequence.request_slot.is_none());
        assert_eq!(sequence.generated_tokens, vec![TokenId(2)]);
    }
    executor.capacity_epoch.store(2, Ordering::Release);
    sender
        .send(Ok(ExecutionTerminalOutcome::ended(
            executor.issuer.ended(&identity).unwrap(),
        )))
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert!(!engine
        .inner
        .execution_closes
        .contains_incarnation(&identity));
    assert_eq!(
        engine.inner.scheduler.trace_phase(&request_id),
        Some(RequestPhase::Waiting)
    );
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&request_id];
        assert!(sequence.execution_frontier.is_none());
        assert!(sequence.request_slot.as_ref().unwrap().armed);
        assert!(!sequence.prefill_complete);
        assert_eq!(sequence.prefill_tokens_processed, 0);
        assert_eq!(sequence.generated_tokens, vec![TokenId(2)]);
    }
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    assert!(engine
        .inner
        .scheduler
        .cancel_request(request_id.clone())
        .unwrap());
    let mut sequence = engine.inner.sequences.write().remove(&request_id).unwrap();
    sequence.request_slot.take().unwrap().close(&engine.inner);
}

#[tokio::test]
async fn managed_driver_closing_does_not_revive_a_completed_recompute() {
    // Protocol/resource ownership fixture: no numerical claim. The original
    // observer publishes Ended before Closing, but product consumption occurs
    // after the gate closes and before the newly spawned cleanup task polls.
    let (engine, executor, identity, request_id, slot, sender) = fixture(false);
    let transaction =
        install_recompute_request(&engine, &executor, &identity, &request_id, slot).await;
    assert!(!engine
        .inner
        .execute_capacity_yield(&transaction, 1, None)
        .await
        .unwrap());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );
    executor.capacity_epoch.store(2, Ordering::Release);
    sender
        .send(Ok(ExecutionTerminalOutcome::ended(
            executor.issuer.ended(&identity).unwrap(),
        )))
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    let observer = engine.driver.shutdown(&engine.inner).unwrap();
    engine.inner.drain_execution_closes().unwrap();
    assert_ne!(
        engine.inner.scheduler.trace_phase(&request_id),
        Some(RequestPhase::Waiting)
    );
    assert_eq!(engine.inner.scheduler.waiting_count(), 0);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    observer.wait().await.result().unwrap();
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(engine.inner.scheduler.trace_phase(&request_id), None);
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1,
        "consuming the old Ended during cleanup must not announce a second release"
    );
    assert_eq!(
        executor.calls.load(Ordering::SeqCst),
        1,
        "the original recompute closure already owns the physical terminal"
    );
}

#[tokio::test]
async fn execution_close_recompute_source_error_preserves_feedback_and_once_release() {
    let (engine, executor, identity, request_id, slot, sender) = fixture(false);
    let transaction =
        install_recompute_request(&engine, &executor, &identity, &request_id, slot).await;
    assert!(!engine
        .inner
        .execute_capacity_yield(&transaction, 1, None)
        .await
        .unwrap());
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );

    // The exact incarnation really ends, but its controlled capacity source
    // deliberately does not advance. Reconciliation must retain this error
    // while reporting the queue mutation and the actual Ended exactly once.
    sender
        .send(Ok(ExecutionTerminalOutcome::ended(
            executor.issuer.ended(&identity).unwrap(),
        )))
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    let error = engine
        .inner
        .drain_execution_closes()
        .expect_err("unchanged source must be reported");
    assert!(error
        .to_string()
        .contains("did not advance the failed exact source"));
    let snapshot = engine.inner.scheduler.trace_snapshot();
    assert_eq!(snapshot.capacity_release_epoch, 1);
    assert_eq!(snapshot.decode_capacity_backpressure_admit_limit, Some(1));
    assert_eq!(snapshot.capacity_backpressure_admit_limit, Some(1));
    assert_eq!(snapshot.pressure_pending_release_fences, 0);
    assert_eq!(
        engine.inner.scheduler.trace_phase(&request_id),
        Some(RequestPhase::Waiting)
    );
    assert!(engine.inner.sequences.read()[&request_id]
        .execution_frontier
        .is_none());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );

    assert!(engine
        .inner
        .scheduler
        .cancel_request(request_id.clone())
        .unwrap());
    let mut sequence = engine.inner.sequences.write().remove(&request_id).unwrap();
    sequence.request_slot.take().unwrap().close(&engine.inner);
}

#[tokio::test]
async fn execution_close_recompute_cancel_while_held_does_not_revive_or_repeat_close() {
    for (initial_error, terminal_before_cancel) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let (engine, executor, identity, request_id, slot, sender) = fixture(initial_error);
        let mut sender = Some(sender);
        let transaction =
            install_recompute_request(&engine, &executor, &identity, &request_id, slot).await;
        let yielded = engine
            .inner
            .execute_capacity_yield(&transaction, 1, None)
            .await;
        assert_eq!(yielded.is_err(), initial_error);
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        if terminal_before_cancel {
            executor.capacity_epoch.store(2, Ordering::Release);
            sender
                .take()
                .unwrap()
                .send(Ok(ExecutionTerminalOutcome::ended(
                    executor.issuer.ended(&identity).unwrap(),
                )))
                .unwrap();
            engine.inner.execution_closes.join_observers().await;
        }
        assert!(engine
            .inner
            .scheduler
            .cancel_request(request_id.clone())
            .unwrap());
        let mut state = engine.inner.sequences.write().remove(&request_id).unwrap();
        let frontier = state.execution_frontier.take().unwrap();
        assert!(state.request_slot.is_none());
        let ticket = engine
            .inner
            .begin_execution_close(
                request_id.clone(),
                frontier.sequence().clone(),
                None,
                ExecutionEndReason::Cancelled,
            )
            .unwrap();
        let first = ticket.initial_result().await;
        assert_eq!(first.is_err(), initial_error);
        if let Err(error) = first {
            assert!(error.to_string().contains("initial publication failure"));
        }
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .capacity_release_epoch,
            0
        );
        if let Some(sender) = sender {
            executor.capacity_epoch.store(2, Ordering::Release);
            sender
                .send(Ok(ExecutionTerminalOutcome::ended(
                    executor.issuer.ended(&identity).unwrap(),
                )))
                .unwrap();
        }
        engine.inner.execution_closes.join_observers().await;
        engine.inner.drain_execution_closes().unwrap();
        assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
        assert!(engine.inner.sequences.read().get(&request_id).is_none());
        assert_eq!(engine.inner.scheduler.trace_phase(&request_id), None);
        assert_eq!(engine.inner.scheduler.waiting_count(), 0);
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .capacity_release_epoch,
            1
        );
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .pressure_pending_release_fences,
            0
        );
        engine.inner.drain_execution_closes().unwrap();
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .capacity_release_epoch,
            1
        );
    }
}

#[tokio::test]
async fn execution_close_held_preserves_slot_until_exact_ended_and_notifies_once() {
    let (engine, executor, sequence, request_id, slot, sender) = fixture(false);
    let epoch = engine
        .inner
        .scheduler
        .trace_snapshot()
        .capacity_release_epoch;
    let ticket = engine
        .inner
        .begin_execution_close(
            request_id,
            sequence.clone(),
            Some(slot),
            ExecutionEndReason::Cancelled,
        )
        .unwrap();
    ticket.initial_result().await.unwrap();
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        epoch
    );
    {
        let state = engine.inner.execution_closes.state.lock();
        let entry = &state.entries[&sequence];
        assert!(entry.slot.as_ref().unwrap().armed);
        assert_eq!(entry.recovery.as_ref().unwrap().sequence(), &sequence);
    }
    sender
        .send(Ok(ExecutionTerminalOutcome::ended(
            executor.issuer.ended(&sequence).unwrap(),
        )))
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        epoch + 1
    );
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        epoch + 1
    );
}

#[tokio::test]
async fn execution_close_real_cpu_held_terminal_preserves_error_and_releases_once() {
    use super::super::token_completion_tests::observed_cpu_model;
    use ferrum_execution::model_executor::{
        ExecutorPrefillAdmission, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
    };
    use ferrum_execution::token_execution::{
        ExecutionAdmission, ExecutionBatchOutcome, ExecutionProduct, ExecutionWork,
        TokenAdmissionOutcome, TokenWork,
    };
    use ferrum_models::{test_support::tiny_llama_model, DecoderOnlyLLM};

    let (inner, model, cfg) = observed_cpu_model();
    let (engine, executor, sequence, request_id, slot, sender) =
        fixture_with_inner(false, inner.clone());
    let prompt = [TokenId(1), TokenId(2), TokenId(3)];
    let TokenAdmissionOutcome::Admitted(frontier) = inner
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&request_id, &prompt, cfg.max_seq_len),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("real tiny model must admit the test prompt")
    };
    let policy = LogitsReturnPolicy::FullLogits;
    let work = ExecutionWork::new(
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, prompt.len(), prompt.len()).unwrap()),
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::Completed(batch) = inner.execute_tokens(&[work]).await.unwrap()
    else {
        panic!("real CPU prefill must complete")
    };
    let completion = batch.into_completions().pop().unwrap();
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
        completion.product()
    else {
        panic!("full vocabulary reference is required")
    };
    let reference = tiny_llama_model(&cfg).prefill("independent-reference", &[1, 2, 3]);
    assert_eq!(logits, &reference);
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|value| value.is_finite()));

    // This test wrapper owns an explicit outer -> actual-inner binding. It
    // may publish its outer Ended only after the real inner executor closes
    // this exact incarnation, not merely when this task or its gate finishes.
    let binding = (
        sequence.clone(),
        completion.next_frontier().sequence().clone(),
    );
    let issuer = executor.issuer.clone();
    let (release, gate) = tokio::sync::oneshot::channel();
    let closer = tokio::spawn(async move {
        gate.await.unwrap();
        let close = inner
            .finish_tokens(
                &binding.1,
                ExecutionEndReason::Completed {
                    input_tokens: 3,
                    output_tokens: 1,
                },
            )
            .await
            .unwrap();
        assert!(close.error.is_none());
        let ExecutionCloseDisposition::Ended(actual) = close.disposition else {
            panic!("real CPU cache must be released before publication")
        };
        assert_eq!(actual.sequence(), &binding.1);
        let terminal = ExecutionTerminalOutcome::ended(issuer.ended(&binding.0).unwrap())
            .with_error(FerrumError::backend(
                "deferred post-close publication failure",
            ));
        sender.send(Ok(terminal)).unwrap();
    });
    engine
        .inner
        .begin_execution_close(
            request_id,
            sequence.clone(),
            Some(slot),
            ExecutionEndReason::Completed {
                input_tokens: prompt.len(),
                output_tokens: 1,
            },
        )
        .unwrap()
        .initial_result()
        .await
        .unwrap();
    assert!(sequence.observed_end().is_none());
    {
        let state = model.lock();
        assert_eq!(state.model.kv_caches.len(), 1);
        assert!(state.model.kv_caches.values().all(|layers| {
            layers.len() == cfg.num_layers && layers.iter().all(|layer| layer.len == prompt.len())
        }));
        assert!(state.releases.is_empty());
    }
    {
        let state = engine.inner.execution_closes.state.lock();
        let entry = &state.entries[&sequence];
        assert!(matches!(entry.observation, CloseObservation::Pending));
        assert!(entry.slot.as_ref().unwrap().armed);
    }
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );
    release.send(()).unwrap();
    closer.await.unwrap();
    engine.inner.execution_closes.join_observers().await;
    {
        let state = engine.inner.execution_closes.state.lock();
        let entry = &state.entries[&sequence];
        assert!(entry.operation_error.is_none());
        let CloseObservation::Ended(ended, Some(FerrumError::Backend { message })) =
            &entry.observation
        else {
            panic!("Held must deliver both authentic Ended and its deferred error")
        };
        assert_eq!(ended.sequence(), &sequence);
        assert!(message.contains("deferred post-close publication failure"));
    }
    let error = engine.inner.drain_execution_closes().unwrap_err();
    assert!(matches!(&error, FerrumError::Backend { message }
        if message.contains("deferred post-close publication failure")));
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    {
        let state = model.lock();
        assert!(state.model.kv_caches.is_empty());
        assert_eq!(state.releases, vec![vec![prompt.len(); cfg.num_layers]]);
    }
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(model.lock().releases.len(), 1);
}

#[tokio::test]
async fn execution_close_dropped_ticket_and_join_do_not_cancel_owner() {
    let (engine, executor, sequence, request_id, slot, sender) = fixture(false);
    let ticket = engine
        .inner
        .begin_execution_close(
            request_id,
            sequence.clone(),
            Some(slot),
            ExecutionEndReason::Completed {
                input_tokens: 1,
                output_tokens: 1,
            },
        )
        .unwrap();
    drop(ticket);
    {
        let join = engine.inner.execution_closes.join_observers();
        tokio::pin!(join);
        assert!(futures::poll!(&mut join).is_pending());
    }
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );
    loop {
        let notified = engine.inner.work_notify.notified();
        if engine.inner.execution_closes.state.lock().entries[&sequence]
            .recovery
            .is_some()
        {
            break;
        }
        tokio::time::timeout(Duration::from_secs(2), notified)
            .await
            .unwrap();
    }
    sender
        .send(Ok(ExecutionTerminalOutcome::ended(
            executor.issuer.ended(&sequence).unwrap(),
        )))
        .unwrap();
    // join cancellation removed only the JoinHandle; completion is stored in
    // the same retained record and still wakes the engine.
    loop {
        let notified = engine.inner.work_notify.notified();
        if matches!(
            engine.inner.execution_closes.state.lock().entries[&sequence].observation,
            CloseObservation::Ended(..)
        ) {
            break;
        }
        tokio::time::timeout(Duration::from_secs(2), notified)
            .await
            .unwrap();
    }
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
}

#[tokio::test]
async fn execution_close_keeps_initial_error_and_later_unproved_owner_on_shutdown() {
    let (engine, executor, sequence, request_id, slot, sender) = fixture(true);
    let ticket = engine
        .inner
        .begin_execution_close(
            request_id,
            sequence.clone(),
            Some(slot),
            ExecutionEndReason::Cancelled,
        )
        .unwrap();
    assert!(ticket
        .initial_result()
        .await
        .unwrap_err()
        .to_string()
        .contains("initial publication failure"));
    assert!(engine
        .inner
        .drain_execution_closes()
        .unwrap_err()
        .to_string()
        .contains("initial publication failure"));
    sender
        .send(Err(FerrumError::backend("terminal observation failed")))
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    assert!(engine
        .inner
        .drain_execution_closes()
        .unwrap_err()
        .to_string()
        .contains("terminal observation failed"));
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
    assert!(
        engine.inner.execution_closes.state.lock().entries[&sequence]
            .slot
            .as_ref()
            .unwrap()
            .armed
    );
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );
    assert!(engine
        .shutdown()
        .await
        .unwrap_err()
        .to_string()
        .contains("unproved execution closures"));
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    // Explicit unreclaimed exit logs the retained obligation, never close/release.
    let scheduler = Arc::clone(&engine.inner.scheduler);
    drop(engine);
    assert_eq!(scheduler.trace_snapshot().capacity_release_epoch, 0);
}

#[tokio::test]
async fn execution_close_observed_ended_avoids_reclosing_and_preserves_operation_error() {
    let (engine, executor, sequence, request_id, slot, _sender) = fixture(false);
    let observed = ExecutionCloseOutcome::ended(executor.issuer.ended(&sequence).unwrap())
        .with_error(FerrumError::backend("terminal publication failed"));
    let ticket = engine
        .inner
        .begin_observed_close(request_id, sequence, Some(slot), observed)
        .unwrap();
    assert!(ticket
        .initial_result()
        .await
        .unwrap_err()
        .to_string()
        .contains("terminal publication failed"));
    engine.inner.execution_closes.join_observers().await;
    assert!(engine
        .inner
        .drain_execution_closes()
        .unwrap_err()
        .to_string()
        .contains("terminal publication failed"));
    assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
}

#[tokio::test]
async fn execution_close_rejects_other_incarnation_without_freeing_original_slot() {
    let (engine, executor, sequence, request_id, slot, _sender) = fixture(false);
    let foreign = executor.issuer.issue_sequence().unwrap();
    let observed = ExecutionCloseOutcome::ended(executor.issuer.ended(&foreign).unwrap());
    let ticket = engine
        .inner
        .begin_observed_close(request_id, sequence.clone(), Some(slot), observed)
        .unwrap();
    assert!(ticket
        .initial_result()
        .await
        .unwrap_err()
        .to_string()
        .contains("another incarnation"));
    engine.inner.execution_closes.join_observers().await;
    assert!(engine.inner.drain_execution_closes().is_err());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
    assert!(
        engine.inner.execution_closes.state.lock().entries[&sequence]
            .slot
            .as_ref()
            .unwrap()
            .armed
    );
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );
}

#[tokio::test]
async fn execution_close_published_end_still_runs_executor_close_validation() {
    for reason in [
        ExecutionEndReason::Cancelled,
        ExecutionEndReason::Completed {
            input_tokens: 1,
            output_tokens: 1,
        },
    ] {
        let (engine, executor, sequence, request_id, slot, _sender) = fixture(false);
        drop(executor.issuer.ended(&sequence).unwrap());
        let ticket = engine
            .inner
            .begin_execution_close(request_id, sequence, Some(slot), reason)
            .unwrap();
        ticket.initial_result().await.unwrap();
        engine.inner.execution_closes.join_observers().await;
        engine.inner.drain_execution_closes().unwrap();
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .capacity_release_epoch,
            1
        );
    }
}

#[tokio::test]
async fn execution_close_retains_error_when_end_is_published_during_close() {
    let (engine, executor, sequence, request_id, slot, _sender) = fixture(false);
    assert!(sequence.observed_end().is_none());
    executor.end_then_error.store(true, Ordering::SeqCst);
    let ticket = engine
        .inner
        .begin_execution_close(
            request_id,
            sequence.clone(),
            Some(slot),
            ExecutionEndReason::Cancelled,
        )
        .unwrap();
    assert!(ticket
        .initial_result()
        .await
        .unwrap_err()
        .to_string()
        .contains("failed after publishing actual terminal fact"));
    engine.inner.execution_closes.join_observers().await;
    assert!(engine
        .inner
        .drain_execution_closes()
        .unwrap_err()
        .to_string()
        .contains("failed after publishing actual terminal fact"));
    assert!(sequence.observed_end().is_some());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
}

#[tokio::test]
async fn execution_close_observer_error_does_not_erase_independently_published_end() {
    let (engine, executor, sequence, request_id, slot, sender) = fixture(true);
    let ticket = engine
        .inner
        .begin_execution_close(
            request_id,
            sequence.clone(),
            Some(slot),
            ExecutionEndReason::Cancelled,
        )
        .unwrap();
    assert!(ticket
        .initial_result()
        .await
        .unwrap_err()
        .to_string()
        .contains("initial publication failure"));
    assert!(engine
        .inner
        .drain_execution_closes()
        .unwrap_err()
        .to_string()
        .contains("initial publication failure"));
    drop(executor.issuer.ended(&sequence).unwrap());
    sender
        .send(Err(FerrumError::backend(
            "observer transport failed after terminal",
        )))
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    assert!(engine
        .inner
        .drain_execution_closes()
        .unwrap_err()
        .to_string()
        .contains("observer transport failed after terminal"));
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
}

#[tokio::test]
async fn execution_close_duplicate_retains_the_second_real_observer() {
    let (engine, executor, sequence, request_id, slot, sender) = fixture(false);
    let ticket = engine
        .inner
        .begin_execution_close(
            request_id.clone(),
            sequence.clone(),
            Some(slot),
            ExecutionEndReason::Cancelled,
        )
        .unwrap();
    ticket.initial_result().await.unwrap();
    let (second_sender, second_receiver) = tokio::sync::oneshot::channel();
    let second = executor
        .issuer
        .held(
            executor.issuer.recovery_key(&sequence).unwrap(),
            async move {
                second_receiver
                    .await
                    .map_err(|_| FerrumError::internal("lost duplicate observer"))?
            },
        )
        .unwrap();
    assert!(engine
        .inner
        .begin_observed_close(
            request_id,
            sequence.clone(),
            None,
            ExecutionCloseOutcome::held(second)
                .with_error(FerrumError::backend("duplicate publication error")),
        )
        .is_err());
    assert!(
        !second_sender.is_closed(),
        "rejection must retain the incoming Held future"
    );
    assert!(engine
        .inner
        .drain_execution_closes()
        .unwrap_err()
        .to_string()
        .contains("duplicate publication error"));
    sender
        .send(Ok(ExecutionTerminalOutcome::ended(
            executor.issuer.ended(&sequence).unwrap(),
        )))
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    assert!(!second_sender.is_closed());
    drop(engine);
    assert!(
        second_sender.is_closed(),
        "only explicit unreclaimed registry exit drops the rejected observer"
    );
}

#[tokio::test]
async fn execution_close_foreign_held_survives_original_terminal_consumption() {
    let (engine, executor, sequence, request_id, slot, _sender) = fixture(false);
    let foreign = executor.issuer.issue_sequence().unwrap();
    let (foreign_sender, foreign_receiver) = tokio::sync::oneshot::channel();
    let held = executor
        .issuer
        .held(
            executor.issuer.recovery_key(&foreign).unwrap(),
            async move {
                foreign_receiver
                    .await
                    .map_err(|_| FerrumError::internal("lost foreign observer"))?
            },
        )
        .unwrap();
    // A separate real published fact permits only the original slot's release.
    drop(executor.issuer.ended(&sequence).unwrap());
    let ticket = engine
        .inner
        .begin_observed_close(
            request_id,
            sequence,
            Some(slot),
            ExecutionCloseOutcome::held(held)
                .with_error(FerrumError::backend("foreign publication error")),
        )
        .unwrap();
    assert!(ticket
        .initial_result()
        .await
        .unwrap_err()
        .to_string()
        .contains("foreign publication error"));
    engine.inner.execution_closes.join_observers().await;
    assert!(engine.inner.drain_execution_closes().is_err());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
    assert!(
        !foreign_sender.is_closed(),
        "releasing the original slot cannot erase a foreign Held"
    );
    assert!(foreign.observed_end().is_none());
    drop(engine);
    assert!(foreign_sender.is_closed());
}

#[tokio::test]
async fn execution_close_wrong_request_slot_retains_incoming_observer() {
    let (engine, executor, sequence, _request_id, slot, _sender) = fixture(false);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let held = executor
        .issuer
        .held(
            executor.issuer.recovery_key(&sequence).unwrap(),
            async move {
                receiver
                    .await
                    .map_err(|_| FerrumError::internal("lost invalid slot observer"))?
            },
        )
        .unwrap();
    assert!(engine
        .inner
        .begin_observed_close(
            RequestId::new(),
            sequence,
            Some(slot),
            ExecutionCloseOutcome::held(held),
        )
        .is_err());
    assert!(!sender.is_closed());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
    assert!(engine.inner.drain_execution_closes().is_err());
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );
    drop(engine);
    assert!(sender.is_closed());
}

#[tokio::test]
async fn execution_close_unassigned_retains_observer_without_registering_foreign_identity() {
    let (engine, executor, sequence, request_id, slot, sender) = fixture(false);
    let ticket = engine
        .inner
        .begin_execution_close(
            request_id.clone(),
            sequence.clone(),
            Some(slot),
            ExecutionEndReason::Cancelled,
        )
        .unwrap();
    ticket.initial_result().await.unwrap();
    let foreign = executor.issuer.issue_sequence().unwrap();
    let (foreign_sender, foreign_receiver) = tokio::sync::oneshot::channel();
    let held = executor
        .issuer
        .held(
            executor.issuer.recovery_key(&foreign).unwrap(),
            async move {
                foreign_receiver
                    .await
                    .map_err(|_| FerrumError::internal("lost unassigned observer"))?
            },
        )
        .unwrap();
    engine.inner.retain_unassigned_execution_close(
        request_id,
        ExecutionCloseOutcome::held(held)
            .with_error(FerrumError::backend("unassigned publication error")),
        FerrumError::internal("unassigned batch disposition"),
    );
    {
        let state = engine.inner.execution_closes.state.lock();
        assert_eq!(state.entries.len(), 1);
        assert!(state.entries.contains_key(&sequence));
        assert!(!state.entries.contains_key(&foreign));
        assert_eq!(state.rejected.len(), 1);
        assert!(state.rejected[0].slot.is_none());
    }
    assert!(engine
        .inner
        .drain_execution_closes()
        .unwrap_err()
        .to_string()
        .contains("unassigned publication error"));
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );
    assert!(!foreign_sender.is_closed());
    sender
        .send(Ok(ExecutionTerminalOutcome::ended(
            executor.issuer.ended(&sequence).unwrap(),
        )))
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert!(!foreign_sender.is_closed());
    assert!(foreign.observed_end().is_none());
    drop(engine);
    assert!(foreign_sender.is_closed());
}

#[tokio::test]
async fn execution_close_detached_joins_held_without_downgrading_release_obligation() {
    let (engine, executor, sequence, request_id, slot, sender) = fixture(false);
    engine
        .inner
        .begin_execution_close(
            request_id.clone(),
            sequence.clone(),
            Some(slot),
            ExecutionEndReason::Cancelled,
        )
        .unwrap()
        .initial_result()
        .await
        .unwrap();
    let (additional_sender, additional_receiver) = tokio::sync::oneshot::channel();
    let additional = executor
        .issuer
        .held(
            executor.issuer.recovery_key(&sequence).unwrap(),
            async move {
                additional_receiver
                    .await
                    .map_err(|_| FerrumError::internal("lost detached observer"))?
            },
        )
        .unwrap();
    let ticket = engine
        .inner
        .begin_detached_execution_close(
            request_id,
            sequence.clone(),
            Some(
                ExecutionCloseOutcome::held(additional)
                    .with_error(FerrumError::backend("detached observation error")),
            ),
        )
        .unwrap();
    ticket.initial_result().await.unwrap();
    assert!(engine
        .inner
        .drain_execution_closes()
        .unwrap_err()
        .to_string()
        .contains("detached observation error"));
    assert!(!additional_sender.is_closed());
    assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        0
    );
    {
        let state = engine.inner.execution_closes.state.lock();
        let entry = &state.entries[&sequence];
        assert!(matches!(entry.purpose, ClosePurpose::Finalize { .. }));
        assert!(entry.slot.as_ref().unwrap().armed);
    }
    sender
        .send(Ok(ExecutionTerminalOutcome::ended(
            executor.issuer.ended(&sequence).unwrap(),
        )))
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .capacity_release_epoch,
        1
    );
    // Only the same incarnation's actual terminal proof makes this secondary
    // observer dispensable; it was retained throughout Held.
    assert!(additional_sender.is_closed());
}

#[tokio::test]
async fn execution_close_detached_held_and_fresh_slotless_release_have_distinct_obligations() {
    for detached in [true, false] {
        let (engine, executor, sequence, request_id, slot, sender) = fixture(false);
        // The fixture slot is not transferred. The two paths intentionally
        // share slot=None, so a slot-presence heuristic cannot satisfy this test.
        slot.close(&engine.inner);
        let ticket = if detached {
            engine
                .inner
                .begin_detached_execution_close(request_id, sequence.clone(), None)
        } else {
            // A just-admitted physical owner that could not be installed into
            // the product retains its new release obligation without a slot.
            engine.inner.begin_execution_close(
                request_id,
                sequence.clone(),
                None,
                ExecutionEndReason::Cancelled,
            )
        }
        .unwrap();
        ticket.initial_result().await.unwrap();
        engine.inner.drain_execution_closes().unwrap();
        assert!(!sender.is_closed());
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 1);
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .capacity_release_epoch,
            0
        );
        sender
            .send(Ok(ExecutionTerminalOutcome::ended(
                executor.issuer.ended(&sequence).unwrap(),
            )))
            .unwrap();
        engine.inner.execution_closes.join_observers().await;
        engine.inner.drain_execution_closes().unwrap();
        assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .capacity_release_epoch,
            if detached { 0 } else { 1 }
        );
        engine.inner.drain_execution_closes().unwrap();
        assert_eq!(
            engine
                .inner
                .scheduler
                .trace_snapshot()
                .capacity_release_epoch,
            if detached { 0 } else { 1 }
        );
    }
}
