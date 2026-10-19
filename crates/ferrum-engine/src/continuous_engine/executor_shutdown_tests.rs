use super::*;
use ferrum_models::test_support::{tiny_llama_executor_owned, tiny_tokenizer, TinyLlamaConfig};
use ferrum_testkit::{MockKvCacheManager, MockSampler, MockTensorFactory};

/// A rejected observer must outlive every public Engine handle, and its own
/// terminal event must wake the original cleanup service without another request.
pub(super) async fn retire_rejected_observer_after_drop(
    engine: ContinuousBatchEngine,
    issuer: &ferrum_execution::token_execution::ExecutorSequenceIssuer,
    sequence: &ferrum_execution::token_execution::ExecutorSequence,
    sender: tokio::sync::oneshot::Sender<
        Result<ferrum_execution::token_execution::ExecutionTerminalOutcome>,
    >,
) {
    let (original_inner, original_scope, journal) = {
        let inner = engine.active_inner().unwrap();
        (
            Arc::downgrade(&inner),
            Arc::downgrade(&inner.model_executor),
            inner.scheduler_trace_jsonl.clone(),
        )
    };
    let scheduler = Arc::clone(&engine.scheduler);
    let observer = engine.driver.shutdown_active().unwrap();
    let report = tokio::time::timeout(Duration::from_secs(3), observer.wait())
        .await
        .unwrap();
    assert!(report.result().is_err());
    assert!(report.retains_unproved_owners());
    let mut completion = report.cleanup_completion();
    let epoch = scheduler.trace_snapshot().capacity_release_epoch;
    drop((report, observer, engine));
    assert!(original_inner.upgrade().is_some());
    assert!(original_scope.upgrade().is_some());
    assert!(!sender.is_closed());
    if let Some(journal) = &journal {
        assert!(!journal.delivery_stats().worker_finished);
    }

    sender
        .send(Ok(
            ferrum_execution::token_execution::ExecutionTerminalOutcome::ended(
                issuer.ended(sequence).unwrap(),
            ),
        ))
        .unwrap();
    tokio::time::timeout(Duration::from_secs(3), async {
        while !*completion.borrow_and_update() {
            completion.changed().await.unwrap();
        }
    })
    .await
    .expect("the original Held terminal must wake retained cleanup");
    assert!(original_inner.upgrade().is_none());
    assert!(original_scope.upgrade().is_none());
    assert_eq!(
        scheduler.trace_snapshot().capacity_release_epoch,
        epoch,
        "retiring a rejected observation cannot create a capacity release"
    );
    if let Some(journal) = &journal {
        assert!(journal.delivery_stats().worker_finished);
    }
}

fn engine(executor: impl IntoEngineExecutor) -> ContinuousBatchEngine {
    let config = TinyLlamaConfig::default();
    ContinuousBatchEngine::new(
        EngineConfig::default(),
        Arc::new(ContinuousBatchScheduler::new(Default::default())),
        tiny_tokenizer(&config),
        Arc::new(MockSampler),
        Arc::new(MockKvCacheManager::new(4096)),
        executor,
        Arc::new(MockTensorFactory),
    )
    .unwrap()
}

fn request() -> InferenceRequest {
    let mut request = InferenceRequest::new("hello", "tiny-llama-test");
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request
}

#[tokio::test]
async fn owned_engine_retires_real_cpu_model_before_shutdown_returns() {
    let owner = ExecutorOwner::new(tiny_llama_executor_owned(&TinyLlamaConfig::default()));
    let engine = engine(owner);
    let output = engine.infer(request()).await.unwrap();
    assert!(!output.response().tokens.is_empty());
    let observer = engine.driver.shutdown_active().unwrap();
    let report = observer.wait().await;
    report.result().unwrap();
    assert_eq!(
        report.executor_shutdown().unwrap().snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert!(!report.retains_unproved_owners());
    assert!(engine.driver.active.lock().is_none());
    assert!(!engine.status().await.is_ready);
    assert!(engine.infer(request()).await.is_err());
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn shared_engine_close_preserves_peer_until_its_actual_scope_ends() {
    let owner = ExecutorOwner::new(tiny_llama_executor_owned(&TinyLlamaConfig::default()));
    let first = engine(owner.scope().unwrap());
    let second = engine(owner.scope().unwrap());
    assert!(!first
        .infer(request())
        .await
        .unwrap()
        .response()
        .tokens
        .is_empty());
    first.shutdown().await.unwrap();
    let close = owner.begin_shutdown();
    assert_eq!(close.snapshot().phase(), ExecutorShutdownPhase::Draining);
    assert!(owner.scope().is_err());
    // The original second scope still authorizes real model work after the
    // first engine has fully closed and final scope registration is sealed.
    assert!(!second
        .infer(request())
        .await
        .unwrap()
        .response()
        .tokens
        .is_empty());
    second.shutdown().await.unwrap();
    let terminal = tokio::time::timeout(Duration::from_secs(5), close.wait())
        .await
        .unwrap();
    assert_eq!(terminal.phase(), ExecutorShutdownPhase::Retired);
    terminal.result().unwrap();
}
