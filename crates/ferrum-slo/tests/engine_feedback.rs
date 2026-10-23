//! Real model -> Engine observer -> reference worker -> later Engine selection.
//! This proves a host scheduling decision, not a client latency SLO or speedup.

use ferrum_engine::EngineBuilder;
use ferrum_execution::{execution_attempt::*, executor_lifecycle::ExecutorOwner};
use ferrum_interfaces::{engine::LlmInferenceEngine, Tokenizer};
use ferrum_models::test_support::{
    tiny_llama_model_with_execution_config, tiny_model_info, tiny_tokenizer, TinyLlamaConfig,
};
use ferrum_scheduler::cost_prediction::*;
use ferrum_slo::{ReferenceEstimator, ReferenceEstimatorConfig};
use ferrum_types::{
    EngineConfig, InferenceRequest, LlamaExecutionConfig, ObservabilityProfileDetail,
    SchedulerSloConfig,
};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

/// A bounded test witness delegates every observation to the real worker.
/// It neither invents samples nor supplies scheduler predictions.
struct Witness {
    observer: Arc<dyn ExecutionAttemptObserver>,
    summaries: Mutex<Vec<ExecutionAttemptSummary>>,
}

impl Witness {
    fn new(observer: Arc<dyn ExecutionAttemptObserver>) -> Arc<Self> {
        Arc::new(Self {
            observer,
            summaries: Mutex::new(Vec::new()),
        })
    }
    fn take(&self) -> Vec<ExecutionAttemptSummary> {
        std::mem::take(&mut *self.summaries.lock().unwrap())
    }
}

impl ExecutionAttemptObserver for Witness {
    fn limits(&self) -> ExecutionAttemptLimits {
        self.observer.limits()
    }
    fn try_record(&self, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        let delivery = self.observer.try_record(summary);
        let mut samples = self
            .summaries
            .try_lock()
            .expect("witness is never read during model work");
        assert!(self.limits().accepts(summary.participants.len()));
        assert!(samples.len() < 32, "bounded test witness exhausted");
        samples.push(summary.clone());
        delivery
    }
    fn record_drop(&self, reason: ExecutionAttemptDrop) {
        self.observer.record_drop(reason);
    }
}

fn request() -> InferenceRequest {
    let mut request = InferenceRequest::new("hello world", "tiny-llama-test");
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request
}

async fn engine(
    owner: &ExecutorOwner,
    cfg: &TinyLlamaConfig,
    slo: Option<SchedulerSloConfig>,
    observer: Option<Arc<dyn ExecutionAttemptObserver>>,
    source: Option<Arc<dyn CostSnapshotSource>>,
) -> Box<dyn LlmInferenceEngine> {
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Off;
    config.scheduler.slo = slo;
    let mut builder = EngineBuilder::new(config)
        .with_custom_tokenizer(tiny_tokenizer(cfg))
        .with_custom_sampler(Arc::new(ferrum_engine::registry::GreedySampler))
        .with_custom_executor(owner.scope().unwrap());
    if let Some(observer) = observer {
        builder = if let Some(source) = source {
            builder.with_execution_feedback(observer, source)
        } else {
            builder.with_execution_attempt_observer(observer)
        };
    } else {
        assert!(source.is_none());
    }
    builder.build().await.unwrap()
}

async fn close(engine: Box<dyn LlmInferenceEngine>) {
    engine.shutdown().await.unwrap();
    engine.wait_for_shutdown_completion().await.unwrap();
    drop(engine);
}

fn work(summary: &ExecutionAttemptSummary) -> Vec<CostWork> {
    summary
        .participants
        .iter()
        .map(|member| CostWork {
            kind: member.kind,
            requested: member.requested,
            context_tokens: member.context_tokens,
            prefix_restore: member.prefix_restore,
            internally_reused_tokens: member.internally_reused_tokens,
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_actual_cpu_feedback_changes_next_prefill_and_preserves_output_and_close() {
    let cfg = TinyLlamaConfig::default();
    let model = tiny_llama_model_with_execution_config(
        &cfg,
        LlamaExecutionConfig {
            prefix_cache: false,
            ..Default::default()
        },
    );
    let executor = ExecutorOwner::new(Box::new(ferrum_models::LlmExecutor::new(
        Box::new(model),
        tiny_model_info(&cfg),
    )));
    let prompt_tokens = tiny_tokenizer(&cfg)
        .encode(&request().prompt, true)
        .unwrap()
        .len();
    assert!(prompt_tokens > 1);

    let baseline_engine = engine(&executor, &cfg, None, None, None).await;
    let baseline = baseline_engine.infer(request()).await.unwrap();
    assert!(
        baseline.response().tokens.len() >= 3,
        "fixture must exercise two decode positions"
    );
    close(baseline_engine).await;

    let estimator = ReferenceEstimator::spawn(ReferenceEstimatorConfig {
        min_samples: 2,
        ..Default::default()
    })
    .unwrap();
    let source = estimator.source();
    let training = Witness::new(estimator.observer());
    let training_engine = engine(&executor, &cfg, None, Some(training.clone()), None).await;
    for _ in 0..2 {
        let actual = training_engine.infer(request()).await.unwrap();
        assert_eq!(actual.response().tokens, baseline.response().tokens);
        assert_eq!(actual.response().text, baseline.response().text);
        assert_eq!(
            actual.response().finish_reason,
            baseline.response().finish_reason
        );
    }
    close(training_engine).await;
    let samples = training.take();
    assert!(samples.iter().all(|sample| sample.eligibility
        == ExecutionAttemptEligibility::Completed
        && sample.participants.len() == 1));
    let prefill = samples
        .iter()
        .find(|sample| sample.participants[0].kind == AttemptWorkKind::Prefill)
        .unwrap();
    assert_eq!(prefill.participants[0].requested.start(), 0);
    assert_eq!(prefill.participants[0].requested.end(), prompt_tokens);
    assert_eq!(prefill.participants[0].context_tokens, prompt_tokens);
    let prefill_work = work(prefill);
    let mut changes = source.subscribe();
    let snapshot = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let snapshot = source.load();
            if samples.iter().all(|sample| {
                matches!(
                    snapshot.predict(
                        &CostQuery {
                            work: &work(sample),
                            work_is_exact: true,
                            execution:
                                ferrum_scheduler::cost_prediction::CostExecution::SingleInvocation,
                        },
                        Instant::now()
                    ),
                    CostPrediction::Known { samples: 2, .. }
                )
            }) {
                break snapshot;
            }
            changes.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
    for generated in [1, 2] {
        let decode = samples
            .iter()
            .find(|sample| {
                let member = &sample.participants[0];
                member.kind == AttemptWorkKind::Decode
                    && member.requested.start() == prompt_tokens + generated - 1
            })
            .expect("actual decode position was observed");
        assert_eq!(
            decode.participants[0].context_tokens,
            prompt_tokens + generated
        );
        assert_eq!(
            decode.participants[0].requested.end(),
            prompt_tokens + generated
        );
    }
    let CostPrediction::Known {
        upper_service_time, ..
    } = snapshot.predict(
        &CostQuery {
            work: &prefill_work,
            work_is_exact: true,
            execution: ferrum_scheduler::cost_prediction::CostExecution::SingleInvocation,
        },
        Instant::now(),
    )
    else {
        panic!("real full-prefill domain was not learned");
    };
    assert!(!upper_service_time.is_zero());
    let learned_revision = snapshot.revision();

    // Freeze a real consumer budget below the already learned full-prefill
    // cost. Smaller uncovered work uses the explicitly selected static fallback.
    let slo = SchedulerSloConfig {
        ttft_ms: upper_service_time.as_secs_f64() * 1000.0 / 2.0,
        tpot_ms: 10_000.0,
        itl_ms: 10_000.0,
        query: Default::default(),
        planning: Default::default(),
        joint: Default::default(),
        strategy: ferrum_types::SloStrategy::ReferenceStepDiagnostic,
        unknown_stopping: Default::default(),
        unknown_cost: Default::default(),
    };
    let consumer = Witness::new(estimator.observer());
    let consumer_engine = engine(
        &executor,
        &cfg,
        Some(slo),
        Some(consumer.clone()),
        Some(source.clone()),
    )
    .await;
    let actual = consumer_engine.infer(request()).await.unwrap();
    assert_eq!(actual.response().tokens, baseline.response().tokens);
    assert_eq!(actual.response().text, baseline.response().text);
    assert_eq!(
        actual.response().finish_reason,
        baseline.response().finish_reason
    );
    let selections = consumer_engine.scheduler_slo_snapshot().unwrap();
    assert!(selections["last_revision"].as_u64().unwrap() >= learned_revision);
    assert!(selections["known_queries"].as_u64().unwrap() > 0);
    assert!(selections["shrunk_prefills"].as_u64().unwrap() > 0);
    let consumed = consumer.take();
    let prefills: Vec<_> = consumed
        .iter()
        .flat_map(|sample| &sample.participants)
        .filter(|member| member.kind == AttemptWorkKind::Prefill)
        .collect();
    assert!(
        prefills.len() > 1,
        "learned full-prefill cost must change actual executed ranges"
    );
    assert!(prefills[0].requested.end() < prompt_tokens);
    let mut end = 0;
    for member in prefills {
        assert_eq!(member.requested.start(), end);
        assert!(matches!(
            member.outcome,
            AttemptParticipantOutcome::Completed {
                frontier_committed: true,
                ..
            }
        ));
        end = member.requested.end();
    }
    assert_eq!(end, prompt_tokens);
    close(consumer_engine).await;
    drop((actual, baseline));
    let exit = estimator.shutdown().await.unwrap();
    assert_eq!(exit.stats.queued_records, 0);
    assert_eq!(exit.stats.queued_bytes, 0);
    assert_eq!(
        exit.stats.dropped_full
            + exit.stats.dropped_bytes
            + exit.stats.dropped_oversize
            + exit.stats.dropped_closed,
        0
    );
    assert_eq!(
        exit.stats.invalid_attempts + exit.stats.excluded_attempts + exit.stats.worker_failures,
        0
    );
    executor.begin_shutdown().wait().await.result().unwrap();
}
