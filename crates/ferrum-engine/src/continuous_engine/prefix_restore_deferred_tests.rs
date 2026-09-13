//! Optional restore waits must preserve progress and exact checkpoint ownership.
use super::*;
use ferrum_interfaces::model_executor::{
    PlanRuntimePrefixRestoreDeferral, PlanRuntimePrefixRestoreInput,
    PlanRuntimePrefixRestoreOutcome, PlanRuntimePrefixRestoreOutput, PrefixCaptureLease,
    PrefixCaptureStatus, PrefixRestoreSource,
};
use ferrum_testkit::MockKvCacheHandle;

#[derive(Debug)]
struct Checkpoint {
    request_id: RequestId,
    input: Vec<TokenId>,
}

impl PrefixCaptureLease for Checkpoint {
    fn boundary(&self) -> usize {
        3
    }
    fn status(&self) -> PrefixCaptureStatus {
        PrefixCaptureStatus::Ready
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Default)]
pub(super) struct RestoreState {
    targets: Mutex<HashSet<RequestId>>,
    releasers: Mutex<HashSet<RequestId>>,
    ready: AtomicBool,
    attempts: Mutex<Vec<(RequestId, bool)>>,
    chunks: Mutex<Vec<(RequestId, PrefillChunk)>>,
    leases: Mutex<Vec<Weak<Checkpoint>>>,
    acknowledged: Mutex<Vec<RequestId>>,
}

impl RestoreState {
    pub(super) fn release_capable(&self, id: &RequestId) -> bool {
        self.releasers.lock().contains(id)
    }

    pub(super) fn computed(&self, id: &RequestId, chunk: PrefillChunk) {
        self.chunks.lock().push((id.clone(), chunk));
    }

    pub(super) fn restore(
        self: &Arc<Self>,
        input: PlanRuntimePrefixRestoreInput<'_>,
        epochs: ExecutorAdmissionEpochs,
        wait: ferrum_interfaces::vnext::CapacityWaitCondition,
    ) -> Result<PlanRuntimePrefixRestoreOutcome> {
        if !self.targets.lock().contains(input.request_id) {
            return Ok(PlanRuntimePrefixRestoreOutcome::Unavailable);
        }
        self.attempts
            .lock()
            .push((input.request_id.clone(), input.retry.is_some()));
        let checkpoint: Arc<dyn PrefixCaptureLease> = if let Some(retry) = input.retry {
            let lease = retry
                .checkpoint()
                .as_any()
                .downcast_ref::<Checkpoint>()
                .expect("retry must retain the original source, not reselect it");
            assert_eq!(&lease.request_id, input.request_id);
            assert_eq!(lease.input, input.input_tokens);
            Arc::clone(retry.checkpoint())
        } else {
            let lease = Arc::new(Checkpoint {
                request_id: input.request_id.clone(),
                input: input.input_tokens.to_vec(),
            });
            self.leases.lock().push(Arc::downgrade(&lease));
            lease
        };
        if !self.ready.load(Ordering::Acquire) {
            return Ok(PlanRuntimePrefixRestoreOutcome::Deferred(
                PlanRuntimePrefixRestoreDeferral::new(
                    test_execution_capacity_deferral(
                        epochs,
                        wait,
                        ExecutorExecutionCapacityStage::SequenceExtension,
                    ),
                    checkpoint,
                    PrefixRestoreSource::Index,
                ),
            ));
        }
        let id = input.request_id.clone();
        let cache = Arc::new(MockKvCacheHandle::new(id.clone(), 1, checkpoint.boundary()));
        let state = Arc::clone(self);
        PlanRuntimePrefixRestoreOutput::new(
            id.clone(),
            checkpoint.boundary(),
            input.input_tokens.len(),
            cache,
            move || {
                assert_eq!(checkpoint.status(), PrefixCaptureStatus::Ready);
                state.acknowledged.lock().push(id);
                Ok(())
            },
        )
        .map(PlanRuntimePrefixRestoreOutcome::Restored)
    }

    fn computed_for(&self, id: &RequestId) -> Vec<PrefillChunk> {
        self.chunks
            .lock()
            .iter()
            .filter(|(request, _)| request == id)
            .map(|(_, chunk)| *chunk)
            .collect()
    }

    fn assert_unpinned(&self) {
        assert!(self
            .leases
            .lock()
            .iter()
            .all(|lease| lease.upgrade().is_none()));
    }
}

fn engine() -> (
    ContinuousBatchEngine,
    Arc<PlanRuntimeChunkedPrefillTestExecutor>,
    Arc<RestoreState>,
) {
    let mut config = EngineConfig::default();
    config.scheduler.max_running_requests = 3;
    config.scheduler.prefill_step_chunk = Some(2);
    let state = Arc::new(RestoreState::default());
    let mut executor = PlanRuntimeChunkedPrefillTestExecutor::new(false);
    executor.prefix_retry = Some(Arc::clone(&state));
    let executor = Arc::new(executor);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        Arc::new(ferrum_testkit::MockTokenizer::new(128)),
        Arc::new(ferrum_testkit::MockSampler),
        executor.clone(),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    (engine, executor, state)
}

async fn submit(
    engine: &ContinuousBatchEngine,
    id: RequestId,
    tokens: &[u32],
) -> tokio::sync::oneshot::Receiver<Result<InferenceResponse>> {
    let mut request = policy_request();
    request.id = id.clone();
    request.sampling_params.max_tokens = 1;
    request.metadata.insert(
        PROMPT_TOKENS_METADATA_KEY.to_owned(),
        serde_json::json!(tokens.len()),
    );
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        tokens.iter().copied().map(TokenId::new).collect(),
        Some(Arc::clone(&engine.inner.tokenizer)),
        Some(128),
    );
    let (sender, receiver) = tokio::sync::oneshot::channel();
    sequence.response_sender = Some(sender);
    assert!(engine
        .inner
        .sequences
        .write()
        .insert(id, sequence)
        .is_none());
    engine.inner.scheduler.submit(request).await.unwrap();
    receiver
}

async fn finish(engine: &ContinuousBatchEngine) {
    tokio::time::timeout(Duration::from_secs(2), async {
        for _ in 0..64 {
            if engine.inner.sequences.read().is_empty() {
                return;
            }
            engine.inner.run_iteration().await.unwrap();
        }
        panic!("optional prefix restore prevented runnable cold progress");
    })
    .await
    .expect("capacity deferral must not await inside an iteration");
}

async fn prime_peer(engine: &ContinuousBatchEngine, state: &RestoreState, peer: &RequestId) {
    for _ in 0..8 {
        if !state.computed_for(peer).is_empty() {
            return;
        }
        engine.inner.run_iteration().await.unwrap();
    }
    panic!("peer must own actual prefill state before it can release capacity");
}

#[tokio::test]
async fn prefix_restore_capacity_without_releaser_falls_back_and_releases_checkpoint() {
    let (engine, _, state) = engine();
    let id = RequestId::new();
    state.targets.lock().insert(id.clone());
    let response = submit(&engine, id.clone(), &[11, 12, 13, 14, 15]).await;
    finish(&engine).await;
    assert!(response.await.unwrap().is_ok());
    assert_eq!(state.computed_for(&id)[0].tokens_processed(), 0);
    assert_eq!(engine.inner.prefix_cache_hits.load(Ordering::Relaxed), 0);
    assert!(state.acknowledged.lock().is_empty());
    state.assert_unpinned();
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_restore_capacity_retries_retained_source_before_any_cold_work() {
    let (engine, executor, state) = engine();
    let peer = RequestId::new();
    state.releasers.lock().insert(peer.clone());
    let peer_response = submit(&engine, peer.clone(), &[31; 21]).await;
    prime_peer(&engine, &state, &peer).await;
    let id = RequestId::new();
    state.targets.lock().insert(id.clone());
    let response = submit(&engine, id.clone(), &[11, 12, 13, 14, 15]).await;
    engine.inner.run_iteration().await.unwrap();
    engine.inner.run_iteration().await.unwrap();
    assert!(state.computed_for(&id).is_empty());
    assert_eq!(*state.attempts.lock(), [(id.clone(), false)]);
    assert!(state
        .leases
        .lock()
        .iter()
        .any(|lease| lease.upgrade().is_some()));

    state.ready.store(true, Ordering::Release);
    executor.publish_release();
    finish(&engine).await;
    assert!(response.await.unwrap().is_ok());
    assert!(peer_response.await.unwrap().is_ok());
    assert_eq!(
        state.computed_for(&id),
        [PrefillChunk::new(3, 2, 5).unwrap()]
    );
    assert_eq!(
        *state.attempts.lock(),
        [(id.clone(), false), (id.clone(), true)]
    );
    assert_eq!(*state.acknowledged.lock(), [id]);
    state.assert_unpinned();
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_restore_cancelled_wait_cannot_leak_into_reused_request_id() {
    let (engine, executor, state) = engine();
    let peer = RequestId::new();
    state.releasers.lock().insert(peer.clone());
    let peer_response = submit(&engine, peer.clone(), &[31; 21]).await;
    prime_peer(&engine, &state, &peer).await;
    let id = RequestId::new();
    state.targets.lock().insert(id.clone());
    let cancelled = submit(&engine, id.clone(), &[11, 12, 13, 14, 15]).await;
    engine.inner.run_iteration().await.unwrap();
    assert!(state.computed_for(&id).is_empty());
    drop(cancelled);
    engine.inner.run_iteration().await.unwrap();
    assert!(!engine.inner.sequences.read().contains_key(&id));
    state.assert_unpinned();

    state.ready.store(true, Ordering::Release);
    executor.publish_release();
    let replacement = submit(&engine, id.clone(), &[81, 82, 83, 84, 85]).await;
    finish(&engine).await;
    assert!(replacement.await.unwrap().is_ok());
    assert!(peer_response.await.unwrap().is_ok());
    assert_eq!(
        *state.attempts.lock(),
        [(id.clone(), false), (id.clone(), false)]
    );
    assert_eq!(
        state.computed_for(&id),
        [PrefillChunk::new(3, 2, 5).unwrap()]
    );
    state.assert_unpinned();
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_restore_all_requests_deferred_cannot_form_a_wait_cycle() {
    let (engine, _, state) = engine();
    let mut responses = Vec::new();
    for tokens in [
        [11, 12, 13, 14, 15],
        [21, 22, 23, 24, 25],
        [31, 32, 33, 34, 35],
    ] {
        let id = RequestId::new();
        state.targets.lock().insert(id.clone());
        state.releasers.lock().insert(id.clone());
        responses.push(submit(&engine, id, &tokens).await);
    }
    finish(&engine).await;
    for response in responses {
        assert!(response.await.unwrap().is_ok());
    }
    assert!(state.acknowledged.lock().is_empty());
    state.assert_unpinned();
    engine.shutdown().await.unwrap();
}
