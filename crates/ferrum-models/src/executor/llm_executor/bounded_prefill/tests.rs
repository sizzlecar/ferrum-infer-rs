use super::*;
use crate::common::LlmRuntimeConfig;
use ferrum_interfaces::model_executor::PrefillChunk;
use ferrum_testkit::MockTensor;
use std::collections::HashMap;

struct IncrementalModel {
    config: LlmRuntimeConfig,
    caches: Arc<Mutex<HashMap<String, Vec<u32>>>>,
    declared: bool,
    drop_last: bool,
}

impl DecoderOnlyLLM for IncrementalModel {
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
        let len = tokens.len() - usize::from(self.drop_last);
        self.caches
            .lock()
            .entry(cache.to_owned())
            .or_default()
            .extend_from_slice(&tokens[..len]);
        vec![0.0, 1.0, 2.0, 3.0]
    }
    fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
        panic!("unexpected future decode")
    }
    fn release(&mut self, cache: &str) {
        self.caches.lock().remove(cache);
    }
}

fn fixture(
    declared: bool,
    drop_last: bool,
) -> (LlmExecutor, Arc<Mutex<HashMap<String, Vec<u32>>>>) {
    let caches = Arc::new(Mutex::new(HashMap::new()));
    let model = IncrementalModel {
        config: LlmRuntimeConfig {
            hidden_size: 4,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 4,
            vocab_size: 4,
            max_seq_len: 16,
        },
        caches: caches.clone(),
        declared,
        drop_last,
    };
    let info = ModelInfo {
        model_id: ferrum_types::ModelId("incremental-fixture".to_owned()),
        model_type: ferrum_types::ModelType::Custom("incremental-fixture".to_owned()),
        num_parameters: 0,
        hidden_size: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 4,
        max_sequence_length: 16,
        dtype: DataType::FP32,
        device: ferrum_types::Device::CPU,
        version: None,
        license: None,
        metadata: HashMap::new(),
    };
    (LlmExecutor::new(Box::new(model), info), caches)
}

fn input(request: RequestId, chunk: PrefillChunk) -> PrefillInput {
    PrefillInput::new(MockTensor::from_f32(vec![0.0, 1.0, 2.0, 3.0], &[1, 4]).into_ref())
        .with_request_context(request, 8)
        .with_chunk(chunk)
}

#[tokio::test]
async fn bounded_legacy_prefill_slices_exact_range_and_retains_actual_model_cache() {
    let (executor, caches) = fixture(true, false);
    assert!(executor.supports_bounded_incremental_prefill());
    let request = RequestId::new();
    let first = PrefillChunk::new(0, 2, 4).unwrap();
    let ExecutorPrefillOutcome::Completed(completion) = executor
        .bounded_incremental_prefill(&input(request.clone(), first))
        .await
        .unwrap()
    else {
        panic!("not completed")
    };
    assert_eq!(completion.completed_chunk(), first);
    let (output, _, _, _) = completion.into_parts();
    let cache_id = output.kv_cache.cache_id();
    assert_eq!(output.kv_cache.block_table().sequence_length, 2);
    assert_eq!(caches.lock()[&cache_id], vec![0, 1]);
    let last = PrefillChunk::new(2, 2, 4).unwrap();
    let continuation = input(request, last).with_kv_cache(output.kv_cache);
    let ExecutorPrefillOutcome::Completed(completion) = executor
        .bounded_incremental_prefill(&continuation)
        .await
        .unwrap()
    else {
        panic!("not completed")
    };
    assert_eq!(completion.planned_chunk(), last);
    assert_eq!(completion.completed_chunk(), last);
    let (output, _, _, _) = completion.into_parts();
    assert_eq!(output.kv_cache.cache_id(), cache_id);
    assert_eq!(output.kv_cache.block_table().sequence_length, 4);
    assert_eq!(caches.lock()[&cache_id], vec![0, 1, 2, 3]);
    // A stale handle cannot make the model append the second range twice.
    assert!(executor
        .bounded_incremental_prefill(&continuation)
        .await
        .is_err());
    assert_eq!(caches.lock()[&cache_id], vec![0, 1, 2, 3]);
}

#[tokio::test]
async fn bounded_legacy_prefill_rejects_unobserved_progress_and_cleans_fresh_state() {
    let (executor, caches) = fixture(true, true);
    let chunk = PrefillChunk::new(0, 2, 4).unwrap();
    assert!(executor
        .bounded_incremental_prefill(&input(RequestId::new(), chunk))
        .await
        .is_err());
    assert!(
        caches.lock().is_empty(),
        "failed first chunk has no escaped model owner"
    );
}

#[tokio::test]
async fn bounded_legacy_prefill_rejects_undeclared_models_and_missing_continuations() {
    let (executor, caches) = fixture(false, false);
    assert!(!executor.supports_bounded_incremental_prefill());
    assert!(executor
        .bounded_incremental_prefill(&input(
            RequestId::new(),
            PrefillChunk::new(0, 2, 4).unwrap()
        ))
        .await
        .is_err());
    assert!(caches.lock().is_empty());
    let (executor, caches) = fixture(true, false);
    assert!(executor
        .bounded_incremental_prefill(&input(
            RequestId::new(),
            PrefillChunk::new(2, 2, 4).unwrap()
        ))
        .await
        .is_err());
    assert!(caches.lock().is_empty());
}
