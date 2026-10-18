use super::*;
use ferrum_testkit::MockTokenizer;
use futures::StreamExt;

struct CountingTokenizer {
    inner: MockTokenizer,
    encodes: AtomicU64,
}

impl Tokenizer for CountingTokenizer {
    fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>> {
        self.encodes.fetch_add(1, Ordering::Relaxed);
        self.inner.encode(text, add_special)
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        self.inner.decode(tokens, skip_special)
    }

    fn decode_incremental(&self, previous: &[TokenId], next: TokenId) -> Result<String> {
        self.inner.decode_incremental(previous, next)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        self.inner.special_tokens()
    }

    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.inner.token_id(text)
    }

    fn token_text(&self, token: TokenId) -> Option<&str> {
        self.inner.token_text(token)
    }

    fn info(&self) -> TokenizerInfo {
        self.inner.info()
    }
}

// This probe tests admission and metadata transport, not adapter arithmetic.
struct AdapterCapableExecutor {
    inner: MockModelExecutor,
    metadata: Mutex<Vec<HashMap<String, serde_json::Value>>>,
}

#[async_trait::async_trait]
impl ModelExecutor for AdapterCapableExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn adapter_execution_support(&self) -> AdapterExecutionSupport {
        AdapterExecutionSupport::PerSequenceLora
    }

    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.metadata.lock().push(input.metadata.clone());
        self.inner.prefill(input).await
    }

    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

fn engine_with_executor(
    executor: Arc<dyn ModelExecutor + Send + Sync>,
) -> (
    ContinuousBatchEngine,
    Arc<CountingTokenizer>,
    Arc<MockKvCacheManager>,
) {
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer = Arc::new(CountingTokenizer {
        inner: MockTokenizer::new(128),
        encodes: AtomicU64::new(0),
    });
    let kv = Arc::new(MockKvCacheManager::new(256));
    let engine = match executor.execution_resource_authority() {
        ExecutionResourceAuthority::LegacyEngine => ContinuousBatchEngine::new(
            config,
            scheduler,
            tokenizer.clone(),
            Arc::new(ferrum_testkit::MockSampler),
            kv.clone(),
            executor,
            Arc::new(MockTensorFactory),
        ),
        ExecutionResourceAuthority::PlanRuntime => ContinuousBatchEngine::new_plan_runtime(
            config,
            scheduler,
            tokenizer.clone(),
            Arc::new(ferrum_testkit::MockSampler),
            executor,
            Arc::new(MockTensorFactory),
        ),
    }
    .unwrap();
    (engine, tokenizer, kv)
}

fn adapter_request() -> InferenceRequest {
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    request
        .metadata
        .insert("ferrum_lora_adapter".into(), serde_json::json!("adapter-a"));
    request
        .metadata
        .insert("ferrum_lora_path".into(), serde_json::json!("/adapters/a"));
    request
}

async fn rejected_request(
    engine: &ContinuousBatchEngine,
    mut request: InferenceRequest,
    streaming: bool,
) -> FerrumError {
    request.stream = streaming;
    request.sampling_params.max_tokens = 1;
    if streaming {
        engine
            .infer_stream(request)
            .await
            .err()
            .expect("adapter request must fail before a stream is returned")
    } else {
        engine
            .infer(request)
            .await
            .expect_err("adapter request must fail before inference")
    }
}

fn assert_no_admission(
    engine: &ContinuousBatchEngine,
    tokenizer: &CountingTokenizer,
    kv: &MockKvCacheManager,
    request_id: &RequestId,
) {
    assert_eq!(tokenizer.encodes.load(Ordering::Relaxed), 0);
    assert!(engine.inner.sequences.read().is_empty());
    assert_eq!(engine.inner.scheduler.waiting_count(), 0);
    assert_eq!(engine.inner.scheduler.prefilling_count(), 0);
    assert_eq!(engine.inner.scheduler.decoding_count(), 0);
    assert_eq!(engine.inner.scheduler.trace_phase(request_id), None);
    assert_eq!(kv.stats().allocation_count, 0);
    assert_eq!(kv.active_count(), 0);
    assert_eq!(engine.inner.model_executor.status().prefill_operations, 0);
    assert_eq!(engine.inner.model_executor.status().decode_operations, 0);
    assert_eq!(
        engine
            .inner
            .resource_trace_event_counter
            .load(Ordering::Relaxed),
        0
    );
    assert!(!engine.inner.bg_loop_spawned.load(Ordering::Acquire));
    assert_eq!(engine.inner.iteration_count.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn unsupported_adapter_requests_are_rejected_before_admission() {
    let native = Arc::new(PlanRuntimeAdmissionTestExecutor::new(128));
    let executors: [Arc<dyn ModelExecutor + Send + Sync>; 2] =
        [Arc::new(MockModelExecutor::instant(128)), native.clone()];
    for executor in executors {
        let (engine, tokenizer, kv) = engine_with_executor(executor);
        assert_eq!(
            engine.adapter_execution_support(),
            AdapterExecutionSupport::Unsupported
        );
        for streaming in [false, true] {
            let request = adapter_request();
            let id = request.id.clone();
            let error = rejected_request(&engine, request, streaming).await;
            assert!(matches!(error, FerrumError::Unsupported { .. }));
            assert_no_admission(&engine, &tokenizer, &kv, &id);
        }
    }
    assert_eq!(native.admission_probes.load(Ordering::Relaxed), 0);
    assert!(native.retained.lock().unwrap().is_empty());
    assert_eq!(native.prefill_calls.load(Ordering::Relaxed), 0);
    assert_eq!(native.maintenance_calls.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn malformed_adapter_requests_never_reach_tokenization() {
    let capable = Arc::new(AdapterCapableExecutor {
        inner: MockModelExecutor::instant(128),
        metadata: Mutex::new(Vec::new()),
    });
    let executors: [Arc<dyn ModelExecutor + Send + Sync>; 2] =
        [Arc::new(MockModelExecutor::instant(128)), capable.clone()];
    for executor in executors {
        let (engine, tokenizer, kv) = engine_with_executor(executor);
        for metadata in [
            serde_json::json!({"ferrum_lora_adapter": "adapter-a"}),
            serde_json::json!({"ferrum_lora_path": "/adapters/a"}),
            serde_json::json!({"ferrum_lora_adapter": null, "ferrum_lora_path": "/adapters/a"}),
            serde_json::json!({"ferrum_lora_adapter": "adapter-a", "ferrum_lora_path": 7}),
            serde_json::json!({"ferrum_lora_adapter": "", "ferrum_lora_path": "/adapters/a"}),
        ] {
            for streaming in [false, true] {
                let mut request = policy_request();
                request.metadata = serde_json::from_value(metadata.clone()).unwrap();
                let id = request.id.clone();
                rejected_request(&engine, request, streaming).await;
                assert_no_admission(&engine, &tokenizer, &kv, &id);
            }
        }
    }
    assert!(capable.metadata.lock().is_empty());
}

async fn execute_one_token(
    engine: &ContinuousBatchEngine,
    mut request: InferenceRequest,
    streaming: bool,
) {
    request.stream = streaming;
    request.sampling_params.max_tokens = 1;
    // Exercise both admissions without relying on prefix-cache reuse behavior.
    request.prompt = if streaming {
        "stream request"
    } else {
        "sync request"
    }
    .into();
    tokio::time::timeout(Duration::from_secs(2), async {
        if streaming {
            let mut stream = engine.infer_stream(request).await.unwrap();
            let mut finished = false;
            while let Some(chunk) = stream.next().await {
                finished |= chunk.unwrap().finish_reason.is_some();
            }
            assert!(finished);
        } else {
            let response = engine.infer(request).await.unwrap();
            assert_eq!(response.tokens.len(), 1);
        }
    })
    .await
    .expect("accepted request must complete");
}

#[tokio::test]
async fn adapter_support_is_forwarded_and_base_requests_still_execute() {
    let capable = Arc::new(AdapterCapableExecutor {
        inner: MockModelExecutor::instant(128),
        metadata: Mutex::new(Vec::new()),
    });
    let (engine, _, _) = engine_with_executor(capable.clone());
    let public: &dyn LlmInferenceEngine = &engine;
    assert_eq!(
        public.adapter_execution_support(),
        AdapterExecutionSupport::PerSequenceLora
    );
    for streaming in [false, true] {
        execute_one_token(&engine, adapter_request(), streaming).await;
    }
    let observed = capable.metadata.lock().clone();
    assert_eq!(observed.len(), 2);
    for metadata in observed {
        assert_eq!(metadata["ferrum_lora_adapter"], "adapter-a");
        assert_eq!(metadata["ferrum_lora_path"], "/adapters/a");
    }
    engine.shutdown().await.unwrap();

    let executor = Arc::new(MockModelExecutor::instant(128));
    let (base, _, _) = engine_with_executor(executor.clone());
    for streaming in [false, true] {
        execute_one_token(&base, policy_request(), streaming).await;
    }
    assert_eq!(executor.prefill_count(), 2);
    base.shutdown().await.unwrap();
}
