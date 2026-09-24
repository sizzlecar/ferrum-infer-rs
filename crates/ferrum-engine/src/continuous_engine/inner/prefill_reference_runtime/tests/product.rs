//! Common engine construction/admission and the actual PlanRuntime commit leaf.
use super::*;
use crate::continuous_engine::ContinuousBatchEngine;
use ferrum_interfaces::{
    engine::InferenceEngine, model_executor::*, scheduler::Scheduler, InferenceRequestContext,
    ModelExecutor,
};
use ferrum_scheduler::implementations::continuous::ContinuousBatchScheduler;
use ferrum_testkit::{MockModelExecutor, MockTensorFactory, MockTokenizer};
use ferrum_types::{EngineConfig, ServiceSloConfig, SloMode};

struct ReferenceExecutor {
    base: MockModelExecutor,
}
#[async_trait::async_trait]
impl ModelExecutor for ReferenceExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.base.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.base.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.base.status()
    }
    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }
    fn execution_cost_identity(&self) -> ExecutorCostIdentityAvailability {
        identity()
    }
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.base.prefill(input).await
    }
    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.base.decode(input).await
    }
    async fn plan_runtime_prefill_with_capacity(
        &self,
        input: &PlanRuntimePrefillInput,
    ) -> Result<PlanRuntimePrefillOutcome> {
        let actual = PrefillChunk::new(
            input.chunk.tokens_processed(),
            input.chunk.tokens_to_process().min(2),
            input.input_tokens.len(),
        )?;
        let cache = Arc::new(MockKvCacheHandle::new(
            input.request_id.clone(),
            1,
            actual.end(),
        ));
        let output = if actual.is_final() {
            let mut logits = vec![0.0; 128];
            logits[4] = 10.0;
            PlanRuntimePrefillOutput::final_logits(
                input.request_id.clone(),
                actual.end(),
                logits,
                cache,
            )?
        } else {
            PlanRuntimePrefillOutput::intermediate(input.request_id.clone(), actual.end(), cache)
        };
        Ok(PlanRuntimePrefillOutcome::Completed(
            PlanRuntimePrefillCompletion::new(
                output,
                input.chunk,
                actual,
                u32::from(actual != input.chunk),
            )?,
        ))
    }
}

fn engine(config: EngineConfig) -> Result<ContinuousBatchEngine> {
    ContinuousBatchEngine::new_plan_runtime(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        Arc::new(MockTokenizer::new(128)),
        Arc::new(crate::registry::GreedySampler),
        Arc::new(ReferenceExecutor {
            base: MockModelExecutor::instant(128),
        }),
        Arc::new(MockTensorFactory),
    )
}
fn config(file: &ArtifactFile) -> EngineConfig {
    let mut config = EngineConfig::default();
    config.scheduler.slo.mode = SloMode::Observe;
    config.scheduler.slo.prefill_reference = Some(file.0.clone());
    config.scheduler.slo.default_service_class = Some("reference".into());
    config.scheduler.slo.services.push(ServiceSloConfig {
        id: "reference".into(),
        server_token_commit: budgets(),
        client_visible: None,
        attainment: Default::default(),
    });
    config
}

#[tokio::test]
async fn shared_constructor_loads_once_without_online_profile_then_real_leaf_credits_actual_chunks()
{
    let file = ArtifactFile::new();
    let engine = engine(config(&file)).unwrap();
    assert!(engine
        .inner
        .cost_runtime
        .as_ref()
        .unwrap()
        .snapshot()
        .is_none());
    let calibration = engine
        .inner
        .prefill_reference_runtime
        .as_ref()
        .unwrap()
        .calibration()
        .clone();
    // No background scheduling: drive the same production leaf deterministically.
    engine.inner.bg_loop_spawned.store(true, Ordering::Release);
    std::fs::write(&file.0.artifact_path, b"corrupted after successful startup").unwrap();
    let mut request = InferenceRequest::new("one two three", "reference-test");
    request.sampling_params.max_tokens = 8;
    request.sampling_params.temperature = 0.0;
    let id = request.id.clone();
    let ingress = Instant::now() - Duration::from_secs(1);
    let stream = engine
        .infer_stream_with_context(request, InferenceRequestContext::from_ingress(ingress))
        .await
        .unwrap();
    let admission_ns = {
        let sequences = engine.inner.sequences.read();
        let bound = known(&sequences[&id]);
        assert_eq!(bound.identity(), calibration.identity());
        assert_eq!(bound.total_prompt_tokens().get(), 4);
        assert!(bound.binding().admitted_at_ns() >= 1_000_000_000);
        bound.binding().admitted_at_ns()
    };
    let batch = engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    engine.inner.process_batch(&batch).await.unwrap();
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.prefill_tokens_processed, 2);
        assert!(sequence.generated_tokens.is_empty());
        assert_eq!(known(sequence).binding().logical_high_water(), 2);
        assert!(!known(sequence).binding().first_token_committed());
    }
    let batch = engine
        .inner
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    engine.inner.process_batch(&batch).await.unwrap();
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.generated_tokens.len(), 1);
        assert_eq!(known(sequence).binding().logical_high_water(), 4);
        assert!(known(sequence).binding().first_token_committed());
        assert_eq!(known(sequence).binding().admitted_at_ns(), admission_ns);
    }
    drop(stream);
    // This fixture disabled the background iteration that normally observes
    // the receiver-drop wake. Drive its real cancellation leaf as well as its
    // prefill leaf; shutdown's credited-output drain does not own this legacy
    // stream. The request slot must be explicitly closed, never dropped away.
    {
        let _iteration = engine.inner.iteration_lock.lock().await;
        assert!(engine.inner.sequences.read()[&id].client_receiver_closed());
        engine.inner.cancel_abandoned_requests().await.unwrap();
    }
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert_eq!(engine.inner.scheduler.active_count(), 0);
    assert_eq!(engine.inner.scheduler.waiting_count(), 0);
    assert_eq!(engine.inner.scheduler.trace_phase(&id), None);
    engine.shutdown().await.unwrap();
}

#[test]
fn configured_bad_reference_fails_common_constructor_but_off_without_reference_is_unchanged() {
    let file = ArtifactFile::new();
    let mut bad = config(&file);
    bad.scheduler
        .slo
        .prefill_reference
        .as_mut()
        .unwrap()
        .expected_protocol_sha256 = [88; 32];
    assert!(matches!(engine(bad), Err(FerrumError::Config { .. })));
    let off = engine(EngineConfig::default()).unwrap();
    assert!(off.inner.prefill_reference_runtime.is_none());
    assert!(off.inner.cost_runtime.is_none());
}
