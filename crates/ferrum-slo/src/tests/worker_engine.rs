//! Actual model progress while the independently owned estimator loses samples.
use super::*;
use ferrum_engine::EngineBuilder;
use ferrum_execution::executor_lifecycle::ExecutorOwner;
use ferrum_interfaces::engine::LlmInferenceEngine;
use ferrum_models::test_support::{
    tiny_llama_model_with_execution_config, tiny_model_info, tiny_tokenizer, TinyLlamaConfig,
};
use ferrum_types::{EngineConfig, InferenceRequest, LlamaExecutionConfig, SchedulerSloConfig};

struct Capture {
    target: Arc<dyn ExecutionAttemptObserver>,
    samples: Mutex<Vec<ExecutionAttemptSummary>>,
}

impl Capture {
    fn new(target: Arc<dyn ExecutionAttemptObserver>) -> Arc<Self> {
        Arc::new(Self {
            target,
            samples: Mutex::new(Vec::new()),
        })
    }

    fn take(&self) -> Vec<ExecutionAttemptSummary> {
        std::mem::take(&mut *self.samples.lock().unwrap())
    }
}

impl ExecutionAttemptObserver for Capture {
    fn limits(&self) -> ExecutionAttemptLimits {
        self.target.limits()
    }

    fn try_record(&self, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        let result = self.target.try_record(summary);
        let mut samples = self
            .samples
            .try_lock()
            .expect("test observations read between requests");
        assert!(samples.len() < 32, "bounded test witness exhausted");
        samples.push(summary.clone());
        result
    }

    fn record_drop(&self, reason: ExecutionAttemptDrop) {
        self.target.record_drop(reason);
    }
}

fn request() -> InferenceRequest {
    let mut request = InferenceRequest::new("hello world", "tiny-llama-test");
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request
}

fn executor(cfg: &TinyLlamaConfig) -> ExecutorOwner {
    let model = tiny_llama_model_with_execution_config(
        cfg,
        LlamaExecutionConfig {
            prefix_cache: false,
            ..Default::default()
        },
    );
    ExecutorOwner::new(Box::new(ferrum_models::LlmExecutor::new(
        Box::new(model),
        tiny_model_info(cfg),
    )))
}

async fn engine(
    executor: &ExecutorOwner,
    cfg: &TinyLlamaConfig,
    feedback: Option<(&ReferenceEstimatorOwner, Arc<Capture>)>,
) -> Box<dyn LlmInferenceEngine> {
    let mut config = EngineConfig::default();
    if feedback.is_some() {
        config.scheduler.slo = Some(SchedulerSloConfig {
            ttft_ms: 10_000.0,
            tpot_ms: 10_000.0,
            itl_ms: 10_000.0,
            query: Default::default(),
            planning: Default::default(),
            joint: Default::default(),
            strategy: Default::default(),
            unknown_stopping: Default::default(),
            unknown_cost: Default::default(),
        });
    }
    let mut builder = EngineBuilder::new(config)
        .with_custom_tokenizer(tiny_tokenizer(cfg))
        .with_custom_sampler(Arc::new(ferrum_engine::registry::GreedySampler))
        .with_custom_executor(executor.scope().unwrap());
    if let Some((owner, observer)) = feedback {
        builder = builder.with_execution_feedback(observer, owner.source());
    }
    builder.build().await.unwrap()
}

async fn close(engine: Box<dyn LlmInferenceEngine>) {
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_real_cpu_full_queue_preserves_output_and_later_publication() {
    let cfg = TinyLlamaConfig::default();
    let executor = executor(&cfg);
    let baseline_engine = engine(&executor, &cfg, None).await;
    let expected = baseline_engine.infer(request()).await.unwrap();
    close(baseline_engine).await;

    let gate = Arc::new(Gate::default());
    let release = Release(Arc::clone(&gate));
    let owner = ReferenceEstimator::spawn_inner(
        ReferenceEstimatorConfig {
            queue_capacity: 1,
            ..config()
        },
        worker::Hooks {
            gate: Some(gate.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let observer = Capture::new(owner.observer());
    let actual_engine = engine(&executor, &cfg, Some((&owner, observer.clone()))).await;
    for _ in 0..2 {
        let actual = actual_engine.infer(request()).await.unwrap();
        assert_eq!(actual.response().tokens, expected.response().tokens);
        assert_eq!(actual.response().text, expected.response().text);
        assert_eq!(
            actual.response().finish_reason,
            expected.response().finish_reason
        );
    }
    tokio::time::timeout(Duration::from_secs(5), gate.wait_entered())
        .await
        .unwrap();
    let samples = observer.take();
    let stats = owner.stats();
    assert_eq!(stats.queued_records, 1);
    assert_eq!(stats.accepted, 1);
    assert_eq!(stats.dropped_full as usize, samples.len() - 1);
    assert_eq!(stats.published_snapshots, 0);
    assert!(stats.queued_bytes <= owner.shared.config.max_queue_bytes);
    assert!(samples
        .iter()
        .all(|sample| sample.eligibility == ExecutionAttemptEligibility::Completed));

    drop(release);
    revision(owner.source().as_ref(), 1).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while owner.stats().queued_records != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let actual = actual_engine.infer(request()).await.unwrap();
    assert_eq!(actual.response().tokens, expected.response().tokens);
    assert_eq!(actual.response().text, expected.response().text);
    assert_eq!(
        actual.response().finish_reason,
        expected.response().finish_reason
    );
    close(actual_engine).await;
    let exit = owner.shutdown().await.unwrap();
    assert_eq!(exit.stats.queued_records, 0);
    assert_eq!(exit.stats.queued_bytes, 0);
    assert!(
        exit.stats.trained_samples > 1,
        "the same worker must consume later real work"
    );
    assert!(exit.revision > 1);
    drop((actual, expected));
    executor.begin_shutdown().wait().await.result().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_real_cpu_worker_panic_preserves_output_and_original_snapshot_expiry() {
    let cfg = TinyLlamaConfig::default();
    let executor = executor(&cfg);
    let baseline_engine = engine(&executor, &cfg, None).await;
    let expected = baseline_engine.infer(request()).await.unwrap();
    close(baseline_engine).await;

    let panic_next = Arc::new(AtomicBool::new(false));
    let owner = ReferenceEstimator::spawn_inner(
        ReferenceEstimatorConfig {
            snapshot_ttl_ms: 20,
            ..config()
        },
        worker::Hooks {
            panic_next: Some(panic_next.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let observer = Capture::new(owner.observer());
    let actual_engine = engine(&executor, &cfg, Some((&owner, observer.clone()))).await;
    let first = actual_engine.infer(request()).await.unwrap();
    assert_eq!(first.response().tokens, expected.response().tokens);
    let samples = observer.take();
    // Wait for accepted observations, rather than assuming request completion
    // synchronously joins the independent worker.
    tokio::time::timeout(Duration::from_secs(5), async {
        while owner.stats().queued_records != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let snapshot = owner.source().load();
    let prefill = samples
        .iter()
        .find(|sample| sample.participants[0].kind == AttemptWorkKind::Prefill)
        .unwrap();
    // Query at this actual observation's timestamp to avoid requiring a slow
    // test host to read the snapshot before its intentionally short TTL ends.
    let CostPrediction::Known { expires_at, .. } =
        predict(snapshot.as_ref(), prefill, prefill.completed_at)
    else {
        panic!("the real prefill observation was not learned");
    };
    let revision_before_failure = snapshot.revision();
    let mut changes = owner.source().subscribe();
    changes.borrow_and_update();
    panic_next.store(true, Ordering::Release);
    let second = actual_engine.infer(request()).await.unwrap();
    assert_eq!(second.response().tokens, expected.response().tokens);
    assert_eq!(second.response().text, expected.response().text);
    assert_eq!(
        second.response().finish_reason,
        expected.response().finish_reason
    );
    tokio::time::timeout(Duration::from_secs(5), async {
        while changes.changed().await.is_ok() {}
    })
    .await
    .unwrap();
    assert_eq!(owner.source().load().revision(), revision_before_failure);
    tokio::time::sleep_until(tokio::time::Instant::from_std(expires_at)).await;
    assert_eq!(
        predict(snapshot.as_ref(), prefill, Instant::now()),
        CostPrediction::Unknown(CostUnknownReason::Stale)
    );
    let before = actual_engine.scheduler_slo_snapshot().unwrap()["static_fallbacks"]
        .as_u64()
        .unwrap();
    let third = actual_engine.infer(request()).await.unwrap();
    assert_eq!(third.response().tokens, expected.response().tokens);
    assert_eq!(third.response().text, expected.response().text);
    assert_eq!(
        third.response().finish_reason,
        expected.response().finish_reason
    );
    assert!(
        actual_engine.scheduler_slo_snapshot().unwrap()["static_fallbacks"]
            .as_u64()
            .unwrap()
            > before
    );
    close(actual_engine).await;
    assert_eq!(owner.shutdown().await, Err(EstimatorError::WorkerPanicked));
    assert_eq!(owner.stats().worker_failures, 1);
    assert_eq!(owner.stats().queued_records, 0);
    assert_eq!(owner.stats().queued_bytes, 0);
    drop((first, second, third, expected));
    executor.begin_shutdown().wait().await.result().unwrap();
}
