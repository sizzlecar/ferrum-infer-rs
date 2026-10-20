//! Real CPU execution checks the collection path; protocol fault tests live
//! beside their existing consumer fixtures and make no numerical claims.

use super::*;
use ferrum_execution::execution_attempt::*;
use ferrum_execution::token_execution::ExecutedTokenWork;
use ferrum_models::test_support::{
    tiny_llama_model_with_execution_config, tiny_model_info, tiny_tokenizer, TinyLlamaConfig,
};

pub(super) struct AttemptRecorder {
    pub(super) limits: ExecutionAttemptLimits,
    pub(super) dropped: AtomicU64,
    pub(super) panic_on_record: bool,
    summaries: Mutex<Vec<ExecutionAttemptSummary>>,
}

impl Default for AttemptRecorder {
    fn default() -> Self {
        Self {
            limits: ExecutionAttemptLimits {
                max_participants: 64,
                max_summary_bytes: 128 * 1024,
            },
            dropped: AtomicU64::new(0),
            panic_on_record: false,
            summaries: Mutex::new(Vec::new()),
        }
    }
}

impl AttemptRecorder {
    pub(super) fn take(&self) -> Vec<ExecutionAttemptSummary> {
        std::mem::take(&mut *self.summaries.lock())
    }
}

impl ExecutionAttemptObserver for AttemptRecorder {
    fn limits(&self) -> ExecutionAttemptLimits {
        self.limits
    }

    fn try_record(&self, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        assert!(!self.panic_on_record, "injected optional observer panic");
        assert!(self.limits.accepts(summary.participants.len()));
        assert!(summary.completed_at <= Instant::now());
        let Some(mut samples) = self.summaries.try_lock() else {
            return ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::QueueFull);
        };
        if samples.len() == 128 {
            return ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::QueueFull);
        }
        samples.push(summary.clone());
        ExecutionAttemptDelivery::Recorded
    }

    fn record_drop(&self, reason: ExecutionAttemptDrop) {
        assert_eq!(reason, ExecutionAttemptDrop::ProjectionLimit);
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

async fn real_cpu_engine(
    observer: Option<Arc<dyn ExecutionAttemptObserver>>,
) -> Box<dyn LlmInferenceEngine> {
    real_cpu_engine_config(observer, EngineConfig::default()).await
}

async fn real_cpu_engine_config(
    observer: Option<Arc<dyn ExecutionAttemptObserver>>,
    config: EngineConfig,
) -> Box<dyn LlmInferenceEngine> {
    let cfg = TinyLlamaConfig::default();
    let model = tiny_llama_model_with_execution_config(
        &cfg,
        ferrum_types::LlamaExecutionConfig {
            prefix_cache: false,
            ..Default::default()
        },
    );
    let owner = ExecutorOwner::new(Box::new(ferrum_models::LlmExecutor::new(
        Box::new(model),
        tiny_model_info(&cfg),
    )));
    let mut builder = crate::EngineBuilder::new(config)
        .with_custom_tokenizer(tiny_tokenizer(&cfg))
        .with_custom_sampler(Arc::new(crate::registry::GreedySampler))
        .with_owned_executor(owner)
        .unwrap();
    if let Some(observer) = observer {
        builder = builder.with_execution_attempt_observer(observer);
    }
    builder.build().await.unwrap()
}

#[tokio::test]
async fn actual_cpu_attempt_observer_does_not_enable_product_timing_or_charge_output_quota() {
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Off;
    config.output.semantic_bytes_total = 4096;
    config.output.semantic_bytes_per_session = 4096;
    let input = request().with_engine_token_timing_evidence();
    let off = real_cpu_engine_config(None, config.clone()).await;
    let baseline = off.infer(input.clone()).await.unwrap();
    let recorder = Arc::new(AttemptRecorder::default());
    let on = real_cpu_engine_config(Some(recorder.clone()), config).await;
    let observed = on.infer(input).await.unwrap();
    assert_eq!(observed.response().tokens, baseline.response().tokens);
    assert_eq!(observed.response().text, baseline.response().text);
    assert_eq!(
        observed.response().finish_reason,
        baseline.response().finish_reason
    );
    for output in [&baseline, &observed] {
        let evidence = output.response().execution_evidence.as_ref().unwrap();
        assert_eq!(evidence.output_token_ids, output.response().tokens);
        let timing = evidence.engine_token_timing.as_ref().unwrap();
        timing.validate(output.response().tokens.len()).unwrap();
        assert!(
            timing.decode_stage_intervals.is_empty(),
            "attempt clocks cannot enable product decode stages with profile Off"
        );
    }
    let baseline_session = baseline.output_session().clone();
    let observed_session = observed.output_session().clone();
    assert_eq!(
        observed_session.semantic_usage(),
        baseline_session.semantic_usage()
    );
    let samples = recorder.take();
    assert!(samples.iter().any(|sample| sample
        .participants
        .iter()
        .any(|member| member.kind == AttemptWorkKind::Decode)));
    assert!(samples.iter().all(|sample| sample.executor_wall().is_some()
        && sample.eligibility == ExecutionAttemptEligibility::Completed));
    off.shutdown().await.unwrap();
    on.shutdown().await.unwrap();
    drop((baseline, observed));
    assert_eq!(baseline_session.semantic_usage(), 0);
    assert_eq!(observed_session.semantic_usage(), 0);
}

fn request() -> InferenceRequest {
    let mut request = InferenceRequest::new("hello world", "tiny-llama-test");
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request
}

#[tokio::test]
async fn actual_cpu_attempts_preserve_outputs_and_each_committed_range_without_journals() {
    let off = real_cpu_engine(None).await;
    let input = request();
    let baseline = off.infer(input.clone()).await.unwrap();
    off.shutdown().await.unwrap();
    off.wait_for_shutdown_completion().await.unwrap();

    let recorder = Arc::new(AttemptRecorder::default());
    let on = real_cpu_engine(Some(recorder.clone())).await;
    let observed = on.infer(input.clone()).await.unwrap();
    assert_eq!(observed.response().tokens, baseline.response().tokens);
    assert_eq!(observed.response().text, baseline.response().text);
    assert_eq!(
        observed.response().finish_reason,
        baseline.response().finish_reason
    );
    let summaries = recorder.take();
    let prompt_len = tiny_tokenizer(&TinyLlamaConfig::default())
        .encode(&input.prompt, true)
        .unwrap()
        .len();
    let mut position = 0;
    let mut previous_end = None;
    for summary in &summaries {
        assert_eq!(summary.eligibility, ExecutionAttemptEligibility::Completed);
        assert!(summary.completed_at >= summary.started_at);
        if let Some(previous_end) = previous_end {
            assert!(summary.started_at >= previous_end);
        }
        previous_end = Some(summary.completed_at);
        assert_eq!(summary.participants.len(), 1);
        let member = &summary.participants[0];
        assert_eq!(member.request_id, input.id);
        assert_eq!(member.requested.start(), position);
        assert_eq!(member.prefix_restore, AttemptPrefixRestore::NoneObserved);
        assert_eq!(member.internally_reused_tokens, None);
        assert!(!member.product_failed);
        let AttemptParticipantOutcome::Completed {
            work,
            frontier_committed: true,
        } = member.outcome
        else {
            panic!("actual model progress must be retained");
        };
        assert_eq!(work.range().unwrap(), member.requested);
        if position < prompt_len {
            assert_eq!(member.kind, AttemptWorkKind::Prefill);
            assert!(matches!(work, ExecutedTokenWork::Prefill { .. }));
        } else {
            assert_eq!(member.kind, AttemptWorkKind::Decode);
            assert!(matches!(work, ExecutedTokenWork::Decode { .. }));
        }
        position = work.range().unwrap().end();
    }
    assert_eq!(position, prompt_len + observed.response().tokens.len() - 1);
    on.shutdown().await.unwrap();
    on.wait_for_shutdown_completion().await.unwrap();
}

#[tokio::test]
async fn actual_cpu_output_survives_projection_limits_and_observer_panic() {
    let off = real_cpu_engine(None).await;
    let baseline = off.infer(request()).await.unwrap();
    off.shutdown().await.unwrap();
    for (participants, bytes, panic_on_record) in [
        (0, usize::MAX, false),
        (64, 0, false),
        (64, 128 * 1024, true),
    ] {
        let recorder = Arc::new(AttemptRecorder {
            limits: ExecutionAttemptLimits {
                max_participants: participants,
                max_summary_bytes: bytes,
            },
            panic_on_record,
            ..Default::default()
        });
        let engine = real_cpu_engine(Some(recorder.clone())).await;
        let output = engine.infer(request()).await.unwrap();
        assert_eq!(output.response().tokens, baseline.response().tokens);
        assert_eq!(output.response().text, baseline.response().text);
        assert!(recorder.take().is_empty());
        assert_eq!(
            recorder.dropped.load(Ordering::Relaxed) > 0,
            !panic_on_record
        );
        engine.shutdown().await.unwrap();
        engine.wait_for_shutdown_completion().await.unwrap();
    }
}
