//! Engine publication tests. Native copy/numerical correctness is exercised
//! by the runtime tests; this fixture detects scheduling and ownership errors.
use super::*;
use ferrum_interfaces::model_executor::{
    PlanRuntimePrefixRestoreInput, PlanRuntimePrefixRestoreOutput,
};

#[derive(Clone, Copy)]
enum RestoreBehavior {
    Hit,
    Miss,
    ForeignRequest,
    FailedAcknowledgement,
}

pub(super) struct RestoreState {
    behavior: RestoreBehavior,
    tokens: usize,
    acknowledgements: AtomicU64,
    cancellations: AtomicU64,
}

struct PublicationGuard {
    state: Arc<RestoreState>,
    acknowledged: bool,
}

impl Drop for PublicationGuard {
    fn drop(&mut self) {
        if !self.acknowledged {
            self.state.cancellations.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl RestoreState {
    pub(super) fn restore(
        self: &Arc<Self>,
        input: PlanRuntimePrefixRestoreInput<'_>,
    ) -> Result<Option<PlanRuntimePrefixRestoreOutput>> {
        if matches!(self.behavior, RestoreBehavior::Miss) {
            return Ok(None);
        }
        let request_id = if matches!(self.behavior, RestoreBehavior::ForeignRequest) {
            RequestId::new()
        } else {
            input.request_id.clone()
        };
        let mut guard = PublicationGuard {
            state: Arc::clone(self),
            acknowledged: false,
        };
        let cache = Arc::new(ferrum_testkit::MockKvCacheHandle::new(
            request_id.clone(),
            1,
            self.tokens,
        ));
        PlanRuntimePrefixRestoreOutput::new(
            request_id,
            self.tokens,
            input.input_tokens.len(),
            cache,
            move || {
                if matches!(guard.state.behavior, RestoreBehavior::FailedAcknowledgement) {
                    return Err(FerrumError::cancelled("restored target was cancelled"));
                }
                guard.acknowledged = true;
                guard.state.acknowledgements.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        )
        .map(Some)
    }
}

fn engine(
    behavior: RestoreBehavior,
    tokens: usize,
    trace_path: Option<PathBuf>,
) -> (
    Arc<ContinuousBatchEngine>,
    Arc<PlanRuntimeChunkedPrefillTestExecutor>,
    Arc<RestoreState>,
) {
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = trace_path;
    config.scheduler.max_running_requests = 1;
    config.scheduler.prefill_step_chunk = Some(2);
    let state = Arc::new(RestoreState {
        behavior,
        tokens,
        acknowledgements: AtomicU64::new(0),
        cancellations: AtomicU64::new(0),
    });
    let mut executor = PlanRuntimeChunkedPrefillTestExecutor::new(false);
    executor.prefix_restore = Some(Arc::clone(&state));
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
    (Arc::new(engine), executor, state)
}

fn request() -> InferenceRequest {
    let mut request = policy_request();
    request.prompt = "one two three four".to_owned();
    request.sampling_params.max_tokens = 1;
    request
}

#[tokio::test]
async fn prefix_restore_executes_only_suffix_and_samples_after_real_prefill() {
    let (engine, executor, state) = engine(RestoreBehavior::Hit, 3, None);
    let response = tokio::time::timeout(Duration::from_secs(2), engine.infer(request()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.tokens.len(), 1);
    assert_eq!(state.acknowledgements.load(Ordering::Relaxed), 1);
    assert_eq!(state.cancellations.load(Ordering::Relaxed), 0);
    assert_eq!(
        *executor.attempted_chunks.lock().unwrap(),
        [PrefillChunk::new(3, 2, 5).unwrap()]
    );
    assert_eq!(engine.inner.total_prefill_tokens.load(Ordering::Relaxed), 2);
    assert_eq!(engine.inner.prefix_cache_hits.load(Ordering::Relaxed), 1);
    assert!(executor.retained.lock().unwrap().is_empty());
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_restore_miss_preserves_all_cold_prefill_work() {
    let (engine, executor, state) = engine(RestoreBehavior::Miss, 3, None);
    tokio::time::timeout(Duration::from_secs(2), engine.infer(request()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        *executor.attempted_chunks.lock().unwrap(),
        [
            PrefillChunk::new(0, 2, 5).unwrap(),
            PrefillChunk::new(2, 2, 5).unwrap(),
            PrefillChunk::new(4, 1, 5).unwrap(),
        ]
    );
    assert_eq!(engine.inner.total_prefill_tokens.load(Ordering::Relaxed), 5);
    assert_eq!(engine.inner.prefix_cache_hits.load(Ordering::Relaxed), 0);
    assert_eq!(state.acknowledgements.load(Ordering::Relaxed), 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn prefix_restore_rejects_foreign_full_or_cancelled_state_before_execution() {
    for (behavior, tokens) in [
        (RestoreBehavior::ForeignRequest, 3),
        (RestoreBehavior::Hit, 0),
        (RestoreBehavior::Hit, 5),
        (RestoreBehavior::FailedAcknowledgement, 3),
    ] {
        let (engine, executor, state) = engine(behavior, tokens, None);
        assert!(
            tokio::time::timeout(Duration::from_secs(2), engine.infer(request()))
                .await
                .expect("invalid restoration must finish cleanup")
                .is_err()
        );
        assert!(executor.attempted_chunks.lock().unwrap().is_empty());
        assert!(executor.retained.lock().unwrap().is_empty());
        assert_eq!(state.acknowledgements.load(Ordering::Relaxed), 0);
        assert_eq!(state.cancellations.load(Ordering::Relaxed), 1);
        assert_eq!(engine.inner.total_prefill_tokens.load(Ordering::Relaxed), 0);
        assert_eq!(engine.inner.prefix_cache_hits.load(Ordering::Relaxed), 0);
        assert!(engine.inner.sequences.read().is_empty());
        engine.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn prefix_restore_profile_accounts_only_for_committed_prefill_work() {
    for behavior in [
        RestoreBehavior::Hit,
        RestoreBehavior::Miss,
        RestoreBehavior::FailedAcknowledgement,
    ] {
        let path = resource_trace_temp_path("prefix-work-accounting");
        let (engine, executor, _) = engine(behavior, 3, Some(path.clone()));
        let result = tokio::time::timeout(Duration::from_secs(2), engine.infer(request()))
            .await
            .unwrap();
        let failed = matches!(behavior, RestoreBehavior::FailedAcknowledgement);
        assert_eq!(result.is_err(), failed);
        flush_engine_profile_events(&engine);
        let events = read_engine_profile_events(&path);
        let restored = events
            .iter()
            .filter(|event| event.phase == "vnext.prefix_restore")
            .map(|event| event.shape["restored_tokens"].as_u64().unwrap())
            .sum::<u64>();
        let committed = events
            .iter()
            .filter(|event| event.phase == "vnext.prefill_chunk_completed")
            .map(|event| {
                (
                    event.shape["start_token"].as_u64().unwrap(),
                    event.shape["end_token"].as_u64().unwrap(),
                    event.shape["computed_tokens"].as_u64().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let executed = executor
            .attempted_chunks
            .lock()
            .unwrap()
            .iter()
            .map(|chunk| {
                (
                    chunk.tokens_processed() as u64,
                    chunk.end() as u64,
                    chunk.tokens_to_process() as u64,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(committed, executed);
        if failed {
            assert_eq!(restored, 0);
            assert!(committed.is_empty());
        } else {
            assert_eq!(
                restored + committed.iter().map(|(_, _, n)| n).sum::<u64>(),
                5
            );
            assert_eq!(committed.first().unwrap().0, restored);
        }
        engine.shutdown().await.unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
