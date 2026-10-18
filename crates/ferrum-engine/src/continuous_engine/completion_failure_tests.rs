//! Terminal errors must not strand unrelated owners or either client channel.

use super::*;
use ferrum_kv::{PagedKvCacheConfig, PagedKvCacheManager};
use ferrum_models::test_support::{tiny_llama_executor, TinyLlamaConfig, TinyTokenizer};
use std::sync::atomic::{AtomicBool, AtomicUsize};

struct CompletionProbe {
    inner: Arc<dyn ModelExecutor>,
    fail_confirmation: AtomicBool,
    completions: AtomicUsize,
    releases: AtomicUsize,
}

#[async_trait::async_trait]
impl ModelExecutor for CompletionProbe {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.inner.prefill(input).await
    }
    async fn batch_prefill(&self, inputs: &[PrefillInput]) -> Result<Vec<PrefillOutput>> {
        self.inner.batch_prefill(inputs).await
    }
    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.inner.decode(input).await
    }
    async fn complete_cache(&self, completion: ExecutorSequenceCompletion) -> Result<()> {
        self.completions.fetch_add(1, Ordering::SeqCst);
        if self.fail_confirmation.swap(false, Ordering::SeqCst) {
            // The real inner executor keeps its cache. The engine is not
            // entitled to replace this failed confirmation with release_cache.
            return Err(FerrumError::backend(
                "injected terminal confirmation failure",
            ));
        }
        self.inner.complete_cache(completion).await
    }
    fn release_cache(&self, id: &str) {
        self.releases.fetch_add(1, Ordering::SeqCst);
        self.inner.release_cache(id);
    }
}

#[derive(Clone, Copy)]
enum CompletionFault {
    Evidence,
    Confirmation,
    Scheduler,
}

async fn completion_failure_closes_known_owners(fault: CompletionFault) {
    let cfg = TinyLlamaConfig::default();
    // Parallel cases can start on the same system-clock tick. Each journal
    // needs its own identity; a shared path legitimately shares its writer.
    let trace_path = resource_trace_temp_path(&format!("completion-failure-{}", RequestId::new()));
    let mut config = EngineConfig::default();
    config.backend.device = Device::CPU;
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());
    config.runtime.profile_detail = ObservabilityProfileDetail::Replay;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer = Arc::new(TinyTokenizer::new(cfg.vocab_size));
    let executor = Arc::new(CompletionProbe {
        inner: tiny_llama_executor(&cfg),
        fail_confirmation: AtomicBool::new(false),
        completions: AtomicUsize::new(0),
        releases: AtomicUsize::new(0),
    });
    let allocations = Arc::new(
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
    let recurrent = Arc::new(InMemoryRecurrentStateManager::new(
        InMemoryRecurrentStateConfig {
            total_memory_bytes: 64,
            total_batch_slots: 1,
        },
    ));
    let factory = Arc::new(crate::tensor_factory::CandleTensorFactory::new(Device::CPU));
    let engine = ContinuousBatchEngine::new_with_recurrent_state_manager(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        Arc::new(crate::registry::GreedySampler),
        allocations.clone(),
        executor.clone(),
        factory.clone(),
        Some(recurrent.clone()),
    )
    .unwrap();
    let mut request = InferenceRequest::new("actual CPU completion", "tiny-llama-test");
    request.sampling_params.max_tokens = 4;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.top_k = Some(cfg.vocab_size);
    request.evidence_request.capture_engine_token_timing = true;
    request
        .metadata
        .insert("ferrum_ignore_eos".into(), serde_json::json!(true));
    let id = request.id.clone();
    let prompt = tokenizer.encode(&request.prompt, true).unwrap();
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        prompt.clone(),
        Some(tokenizer),
        Some(cfg.vocab_size),
    );
    let (response_sender, mut response_receiver) = tokio::sync::oneshot::channel();
    let (stream_sender, mut stream_receiver) = tokio::sync::mpsc::channel(16);
    sequence.response_sender = Some(response_sender);
    sequence.stream_sender = Some(stream_sender);
    let mut slot = RequestSlotLease::open(&engine.inner, id.clone());
    scheduler.submit(request).await.unwrap();
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    slot.admit(&engine.inner);
    sequence.request_slot = Some(slot);
    engine.inner.sequences.write().insert(id.clone(), sequence);
    engine
        .inner
        .process_batch_legacy_split_with_test_sequences(&batch)
        .await
        .unwrap();
    assert_eq!(allocations.list_handles().len(), 1);
    assert!(allocations.stats().used_memory_bytes > 0);
    let initial_chunk = stream_receiver.try_recv().unwrap().unwrap();
    assert!(!initial_chunk.text.is_empty());
    let (cache, pending_token) = {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&id];
        assert_eq!(sequence.generated_tokens.len(), 1);
        (
            sequence.kv_cache_handle().unwrap().clone(),
            sequence.generated_tokens[0],
        )
    };
    let recurrent_handle = recurrent
        .allocate(&RecurrentStateSpec {
            request_id: id.clone(),
            num_layers: 1,
            tensors: vec![RecurrentStateTensorSpec::new(
                0,
                "state",
                vec![1],
                DataType::FP32,
            )],
            device: Device::CPU,
            max_batch_slots: 1,
        })
        .await
        .unwrap();
    engine
        .inner
        .sequences
        .write()
        .get_mut(&id)
        .unwrap()
        .commit_recurrent_state_admission(recurrent_handle, 1);
    engine.inner.trace_recurrent_allocate(&id, 1, Some(1));
    match fault {
        CompletionFault::Evidence => {
            // Exercise the real timing validator: two commit observations for
            // one generated token, after successful actual model prefill.
            engine
                .inner
                .sequences
                .write()
                .get_mut(&id)
                .unwrap()
                .record_generated_token_commit();
        }
        CompletionFault::Confirmation => executor.fail_confirmation.store(true, Ordering::SeqCst),
        CompletionFault::Scheduler => {
            // A real scheduler invariant error, not a fake successful complete.
            assert!(scheduler.cancel(id.clone()).await.unwrap());
        }
    }
    let error = engine
        .inner
        .complete_request_for_test(&id, FinishReason::Length)
        .await
        .unwrap_err();
    match fault {
        CompletionFault::Evidence => assert!(
            matches!(&error, FerrumError::Internal { message } if message.contains("invalid engine token timing evidence"))
        ),
        CompletionFault::Confirmation => assert!(
            matches!(&error, FerrumError::Backend { message } if message == "injected terminal confirmation failure")
        ),
        CompletionFault::Scheduler => assert!(matches!(&error, FerrumError::Scheduler { .. })),
    }
    assert!(!engine.inner.sequences.read().contains_key(&id));
    assert!(allocations.list_handles().is_empty());
    assert_eq!(allocations.stats().used_memory_bytes, 0);
    assert_eq!(recurrent.stats().active_states, 0);
    assert_eq!(recurrent.stats().used_batch_slots, 0);
    assert_eq!(scheduler.trace_snapshot().active_len, 0);
    let response_error = response_receiver
        .try_recv()
        .expect("synchronous client must receive a terminal result")
        .unwrap_err();
    assert_eq!(response_error.to_string(), error.to_string());
    let stream_error = stream_receiver
        .try_recv()
        .expect("streaming client must receive a terminal result")
        .unwrap_err();
    assert_eq!(stream_error.to_string(), error.to_string());
    assert!(
        stream_receiver.try_recv().is_err(),
        "no success terminal after an error"
    );
    let expected_completions = usize::from(!matches!(fault, CompletionFault::Evidence));
    let expected_releases = usize::from(matches!(fault, CompletionFault::Evidence));
    assert_eq!(
        executor.completions.load(Ordering::SeqCst),
        expected_completions
    );
    assert_eq!(executor.releases.load(Ordering::SeqCst), expected_releases);
    // Re-observing the terminal request must not duplicate ownership operations.
    engine
        .inner
        .complete_request_for_test(&id, FinishReason::Length)
        .await
        .unwrap();
    assert_eq!(
        executor.completions.load(Ordering::SeqCst),
        expected_completions
    );
    assert_eq!(executor.releases.load(Ordering::SeqCst), expected_releases);

    flush_engine_profile_events(&engine);
    let events = read_engine_profile_events(&trace_path);
    let resources: Vec<_> = events
        .iter()
        .filter_map(|event| event.resource.as_ref())
        .collect();
    let count = |kind: &str, action| {
        resources
            .iter()
            .filter(|resource| resource.resource_kind == kind && resource.action == action)
            .count()
    };
    assert_eq!(count("request_slot", ResourceAction::Release), 1);
    assert_eq!(count("request_slot", ResourceAction::RequestClose), 1);
    assert_eq!(count("kv_block", ResourceAction::Release), 1);
    assert_eq!(count("recurrent_state_slot", ResourceAction::Release), 1);
    if matches!(fault, CompletionFault::Confirmation) {
        assert_eq!(
            count("model_cache_ref", ResourceAction::Release),
            0,
            "failed terminal confirmation is not a release receipt"
        );
        assert!(events
            .iter()
            .any(|event| event.phase == "engine_model_cache_completion_failed"));
        // The lower executor still owns actual model KV. Continuing it solely
        // in this CPU fixture proves no engine fallback secretly released it.
        // This is not a claim that arbitrary failed GPU work can be resumed.
        let reference = tiny_llama_executor(&cfg);
        let prompt_values: Vec<f32> = prompt.iter().map(|token| token.get() as f32).collect();
        let prompt_tensor = factory
            .from_slice(
                &prompt_values,
                &[1, prompt.len()],
                DataType::FP32,
                Device::CPU,
            )
            .unwrap();
        let reference_prefill = reference
            .prefill(&PrefillInput::new(prompt_tensor))
            .await
            .unwrap();
        let token_tensor = factory
            .from_slice(
                &[pending_token.get() as f32],
                &[1, 1],
                DataType::FP32,
                Device::CPU,
            )
            .unwrap();
        let actual = executor
            .inner
            .decode(&DecodeInput::new(token_tensor.clone(), cache.clone()))
            .await
            .unwrap();
        let expected = reference
            .decode(&DecodeInput::new(
                token_tensor,
                reference_prefill.kv_cache.clone(),
            ))
            .await
            .unwrap();
        let actual = actual.logits.to_vec_f32().unwrap();
        let expected = expected.logits.to_vec_f32().unwrap();
        assert_eq!(actual.len(), cfg.vocab_size);
        assert_eq!(expected.len(), cfg.vocab_size);
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!(actual.is_finite() && expected.is_finite());
            assert!((actual - expected).abs() <= 1e-5);
        }
        executor.inner.release_cache(&cache.cache_id());
        reference.release_cache(&reference_prefill.kv_cache.cache_id());
    } else {
        assert_eq!(count("model_cache_ref", ResourceAction::Release), 1);
    }
    drop(engine);
    let _ = std::fs::remove_file(trace_path);
}

#[tokio::test]
async fn completion_evidence_error_still_releases_resources_and_notifies_clients() {
    completion_failure_closes_known_owners(CompletionFault::Evidence).await;
}

#[tokio::test]
async fn completion_confirmation_error_preserves_lower_owner_and_closes_clients() {
    completion_failure_closes_known_owners(CompletionFault::Confirmation).await;
}

#[tokio::test]
async fn completion_scheduler_error_still_closes_slot_and_notifies_clients() {
    completion_failure_closes_known_owners(CompletionFault::Scheduler).await;
}
