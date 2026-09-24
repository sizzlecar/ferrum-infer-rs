use super::*;
use crate::continuous_engine::inner::batch::legacy_prefill::LegacyPrefillWaveOutcome;

struct IncrementalLegacyModel {
    config: LlmRuntimeConfig,
    caches: Arc<Mutex<HashMap<String, Vec<u32>>>>,
    declared: bool,
}

impl DecoderOnlyLLM for IncrementalLegacyModel {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.config
    }
    fn supports_bounded_incremental_prefill(&self) -> bool {
        self.declared
    }
    fn incremental_prefill_cache_len(&self, cache: &str) -> Result<usize> {
        Ok(self.caches.lock().get(cache).map(Vec::len).unwrap_or(0))
    }
    fn prefill(&mut self, cache: &str, tokens: &[u32]) -> Vec<f32> {
        self.caches
            .lock()
            .entry(cache.to_owned())
            .or_default()
            .extend_from_slice(tokens);
        let mut logits = vec![0.0; 64];
        logits[6] = 1.0;
        logits
    }
    fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
        panic!("unexpected future decode")
    }
    fn release(&mut self, cache: &str) {
        self.caches.lock().remove(cache);
    }
}

async fn legacy_fixture(
    declared: bool,
) -> (
    ContinuousBatchEngine,
    Arc<MockKvCacheManager>,
    Arc<Mutex<HashMap<String, Vec<u32>>>>,
    ferrum_interfaces::BatchPlan,
    RequestId,
) {
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let caches = Arc::new(Mutex::new(HashMap::new()));
    let model = IncrementalLegacyModel {
        config: LlmRuntimeConfig {
            hidden_size: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 4,
            vocab_size: 64,
            max_seq_len: 16,
        },
        caches: caches.clone(),
        declared,
    };
    let info = ferrum_types::ModelInfo {
        model_id: ferrum_types::ModelId::new("bounded-legacy-fixture"),
        model_type: ferrum_types::ModelType::Custom("bounded-legacy-fixture".to_owned()),
        num_parameters: 0,
        hidden_size: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 64,
        max_sequence_length: 16,
        dtype: DataType::FP32,
        device: Device::CPU,
        version: None,
        license: None,
        metadata: HashMap::new(),
    };
    let kv = Arc::new(MockKvCacheManager::new(128));
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        Arc::new(crate::registry::GreedySampler),
        kv.clone(),
        Arc::new(LlmExecutor::new(Box::new(model), info)),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    let mut request = policy_request();
    request.prompt = "test".to_owned();
    request.sampling_params.max_tokens = 4;
    request
        .metadata
        .insert(PROMPT_TOKENS_METADATA_KEY.to_owned(), serde_json::json!(4));
    let rid = request.id.clone();
    scheduler.submit(request.clone()).await.unwrap();
    engine.inner.sequences.write().insert(
        rid.clone(),
        SequenceState::new_with_tokenizer_and_model_vocab_size(
            request,
            vec![TokenId::new(5); 4],
            Some(tokenizer),
            Some(64),
        ),
    );
    let mut batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    batch.requests[0].tokens_to_process = Some(2);
    (engine, kv, caches, batch, rid)
}

#[tokio::test]
async fn bounded_legacy_prefill_retains_one_allocation_and_samples_only_final_chunk_once() {
    let (engine, kv, caches, mut batch, rid) = legacy_fixture(true).await;
    let LegacyPrefillWaveOutcome::Completed {
        completed_chunk, ..
    } = engine
        .inner
        .execute_legacy_prefill_wave(&batch.requests[0])
        .await
        .unwrap()
    else {
        panic!("not completed")
    };
    assert_eq!(completed_chunk.range(), 0..2);
    let cache_id = {
        let sequences = engine.inner.sequences.read();
        let seq = &sequences[&rid];
        assert_eq!(seq.prefill_tokens_processed, 2);
        assert!(!seq.prefill_complete);
        assert!(seq.generated_tokens.is_empty());
        let cache = seq.kv_cache_handle().unwrap();
        assert_eq!(cache.block_table().sequence_length, 2);
        cache.cache_id()
    };
    assert_eq!(caches.lock()[&cache_id], vec![5, 5]);
    assert_eq!(kv.stats().allocation_count, 1);
    let partial = prefill_test_frontier(&engine, &rid);
    assert_eq!(
        (
            partial.computed_tokens,
            partial.resident_tokens,
            partial.scheduled_tokens,
            partial.committed_output_tokens
        ),
        (2, 2, 2, 0)
    );
    batch.requests[0].tokens_processed = 2;
    let LegacyPrefillWaveOutcome::Completed {
        completed_chunk, ..
    } = engine
        .inner
        .execute_legacy_prefill_wave(&batch.requests[0])
        .await
        .unwrap()
    else {
        panic!("not completed")
    };
    assert_eq!(completed_chunk.range(), 2..4);
    let seqs = engine.inner.sequences.read();
    assert_eq!(seqs[&rid].generated_tokens, vec![TokenId::new(6)]);
    assert!(seqs[&rid].prefill_complete);
    assert_eq!(seqs[&rid].prefill_tokens_processed, 4);
    assert_eq!(seqs[&rid].kv_cache_handle().unwrap().cache_id(), cache_id);
    drop(seqs);
    let final_row = prefill_test_frontier(&engine, &rid);
    assert_eq!(
        (
            final_row.computed_tokens,
            final_row.resident_tokens,
            final_row.scheduled_tokens,
            final_row.committed_output_tokens
        ),
        (4, 4, 4, 1)
    );
    assert_eq!(caches.lock()[&cache_id], vec![5; 4]);
    assert_eq!(kv.stats().allocation_count, 1);
    assert!(engine
        .inner
        .execute_legacy_prefill_wave(&batch.requests[0])
        .await
        .is_err());
    assert_eq!(
        engine.inner.sequences.read()[&rid].generated_tokens.len(),
        1
    );
    assert_eq!(prefill_test_frontier(&engine, &rid), final_row);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn bounded_legacy_prefill_capability_and_prefix_exclusions_are_before_allocation() {
    let (engine, kv, caches, batch, rid) = legacy_fixture(false).await;
    assert!(engine
        .inner
        .execute_legacy_prefill_wave(&batch.requests[0])
        .await
        .is_err());
    assert!(caches.lock().is_empty());
    assert_eq!(kv.stats().allocation_count, 0);
    assert_eq!(
        engine.inner.sequences.read()[&rid].prefill_tokens_processed,
        0
    );

    let (mut engine, kv, caches, batch, _) = legacy_fixture(true).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .runtime_config
        .prefix_cache_enabled = true;
    assert!(engine
        .inner
        .execute_legacy_prefill_wave(&batch.requests[0])
        .await
        .is_err());
    assert!(caches.lock().is_empty());
    assert_eq!(kv.stats().allocation_count, 0);
}

#[tokio::test]
async fn bounded_legacy_prefill_enforce_compatibility_returns_after_one_chunk() {
    let (mut engine, kv, caches, batch, rid) = legacy_fixture(true).await;
    Arc::get_mut(&mut engine.inner)
        .unwrap()
        .config
        .scheduler
        .slo
        .mode = ferrum_types::SloMode::Enforce;
    engine.inner.process_batch(&batch).await.unwrap();
    let seqs = engine.inner.sequences.read();
    assert_eq!(seqs[&rid].prefill_tokens_processed, 2);
    assert!(!seqs[&rid].prefill_complete);
    assert!(seqs[&rid].generated_tokens.is_empty());
    assert_eq!(caches.lock().values().next().unwrap().len(), 2);
    assert_eq!(kv.stats().allocation_count, 1);
}
