//! Post-forward output failures must never replay an already consumed batch.

use super::*;
use ferrum_execution::model_executor::{KvSlotRequest, KvSlotReservation, LogitsReturnPolicy};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kv::{PagedKvCacheConfig, PagedKvCacheManager};
use ferrum_models::models::LlamaFamilyModel;
use ferrum_models::test_support::{
    tiny_llama_model, tiny_model_info, TinyLlamaConfig, TinyTokenizer,
};
use parking_lot::Mutex;
use std::sync::atomic::AtomicBool;

struct DecodeObservation {
    cache_id: String,
    position: u32,
    logits: Vec<f32>,
    layer_lengths: Vec<usize>,
}

struct Observation {
    model: LlamaFamilyModel<CpuBackend>,
    decodes: Vec<DecodeObservation>,
    released: Vec<(String, Vec<usize>)>,
}

impl Observation {
    fn lengths(&self, cache_id: &str) -> Vec<usize> {
        self.model.kv_caches[cache_id]
            .iter()
            .map(|layer| layer.len)
            .collect()
    }

    fn record(&mut self, cache_id: &str, position: u32, logits: &[f32]) {
        self.decodes.push(DecodeObservation {
            cache_id: cache_id.into(),
            position,
            logits: logits.to_vec(),
            layer_lengths: self.lengths(cache_id),
        });
    }
}

// The observer owns no synthetic cache. Every forward/reservation/release is
// delegated to the real model, including its native unified/fallback choice.
struct ObservedLlama {
    state: Arc<Mutex<Observation>>,
    config: LlmRuntimeConfig,
}

impl DecoderOnlyLLM for ObservedLlama {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.config
    }

    fn adapter_execution_support(&self) -> ferrum_interfaces::adapter::AdapterExecutionSupport {
        self.state.lock().model.adapter_execution_support()
    }

    fn preload_lora_adapter(
        &mut self,
        adapter: &ferrum_models::lora::ActiveLoraAdapter,
    ) -> Result<()> {
        self.state.lock().model.preload_lora_adapter(adapter)
    }

    fn set_lora_adapter_for_cache(
        &mut self,
        id: &str,
        adapter: Option<ferrum_models::lora::ActiveLoraAdapter>,
    ) -> Result<()> {
        self.state
            .lock()
            .model
            .set_lora_adapter_for_cache(id, adapter)
    }

    fn prepare_kv_capacity(&mut self, id: &str, capacity: usize) {
        self.state.lock().model.prepare_kv_capacity(id, capacity);
    }

    fn reserve_kv_slots(
        &mut self,
        requests: &[KvSlotRequest],
    ) -> Result<Option<KvSlotReservation>> {
        self.state.lock().model.reserve_kv_slots(requests)
    }

    fn prefill(&mut self, id: &str, tokens: &[u32]) -> Vec<f32> {
        self.state.lock().model.prefill(id, tokens)
    }

    fn decode(&mut self, id: &str, token: u32, position: u32) -> Vec<f32> {
        let mut state = self.state.lock();
        let logits = state.model.decode(id, token, position);
        state.record(id, position, &logits);
        logits
    }

    fn decode_batch_with_full_logits(
        &mut self,
        batch: &[(String, u32, u32)],
        full: bool,
    ) -> Vec<Vec<f32>> {
        let mut state = self.state.lock();
        let outputs = state.model.decode_batch_with_full_logits(batch, full);
        assert_eq!(outputs.len(), batch.len());
        for ((id, _, position), logits) in batch.iter().zip(&outputs) {
            state.record(id, *position, logits);
        }
        outputs
    }

    fn decode_batch_with_logits_policy(
        &mut self,
        batch: &[(String, u32, u32)],
        policies: &[LogitsReturnPolicy],
    ) -> Vec<Vec<f32>> {
        let mut state = self.state.lock();
        let outputs = state.model.decode_batch_with_logits_policy(batch, policies);
        assert_eq!(outputs.len(), batch.len());
        for ((id, _, position), logits) in batch.iter().zip(&outputs) {
            state.record(id, *position, logits);
        }
        outputs
    }

    fn unified_forward_can_return_full_logits(&self) -> bool {
        self.state
            .lock()
            .model
            .unified_forward_can_return_full_logits()
    }

    fn unified_forward(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        self.state.lock().model.unified_forward(items)
    }

    fn unified_forward_with_logits_policy(
        &mut self,
        items: &[(String, Vec<u32>, usize, bool)],
        policies: &[LogitsReturnPolicy],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        let mut state = self.state.lock();
        let outputs = state
            .model
            .unified_forward_with_logits_policy(items, policies)?;
        assert_eq!(outputs.len(), items.len());
        for ((id, tokens, position, _), output) in items.iter().zip(&outputs) {
            if *position > 0 && tokens.len() == 1 {
                state.record(
                    id,
                    *position as u32,
                    output.as_ref().expect("decode logits"),
                );
            }
        }
        Ok(outputs)
    }

    fn release(&mut self, id: &str) {
        let mut state = self.state.lock();
        let lengths = state.lengths(id);
        state.released.push((id.into(), lengths));
        state.model.release(id);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultKind {
    InvalidLogits,
    MissingLogits,
    ShortResult,
}

struct FaultAfterForward {
    inner: LlmExecutor,
    armed: AtomicBool,
    injected: Mutex<Option<String>>,
    fault: FaultKind,
    fault_index: usize,
}

#[async_trait::async_trait]
impl ModelExecutor for FaultAfterForward {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
    fn kv_capacity(&self) -> Option<usize> {
        self.inner.kv_capacity()
    }
    fn reserve_kv_slots(&self, requests: &[KvSlotRequest]) -> Result<Option<KvSlotReservation>> {
        self.inner.reserve_kv_slots(requests)
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
    async fn unified_decode(&self, batch: &UnifiedBatch) -> Result<Vec<Option<Vec<f32>>>> {
        let mut outputs = self.inner.unified_decode(batch).await?;
        if self.armed.swap(false, Ordering::SeqCst) {
            assert_eq!(
                batch.items.len(),
                2,
                "fault targets exactly the admitted A/B cohort"
            );
            assert_eq!(outputs.len(), 2);
            for (item, output) in batch.items.iter().zip(&outputs) {
                assert!(item.pos_offset > 0 && item.q_tokens.len() == 1 && item.is_final_chunk);
                let logits = output.as_ref().expect("real CPU decode returned logits");
                assert_eq!(logits.len(), self.info().vocab_size);
                assert!(logits.iter().all(|value| value.is_finite()));
            }
            *self.injected.lock() = Some(batch.items[self.fault_index].seq_id.clone());
            match self.fault {
                FaultKind::InvalidLogits => outputs[self.fault_index]
                    .as_mut()
                    .unwrap()
                    .fill(f32::NEG_INFINITY),
                FaultKind::MissingLogits => outputs[self.fault_index] = None,
                FaultKind::ShortResult => {
                    let _ = outputs.remove(self.fault_index);
                }
            }
        }
        Ok(outputs)
    }
    fn release_cache(&self, id: &str) {
        self.inner.release_cache(id);
    }
}

fn assert_full_logits(expected: &[f32], actual: &[f32], vocab: usize) {
    assert_eq!(expected.len(), vocab);
    assert_eq!(actual.len(), vocab);
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        assert!(
            expected.is_finite() && actual.is_finite(),
            "nonfinite logit {index}"
        );
        assert!(
            (expected - actual).abs() <= 1e-5,
            "logit {index}: {expected} vs {actual}"
        );
    }
}

fn drain(
    receiver: &mut tokio::sync::mpsc::Receiver<Result<StreamChunk>>,
) -> Vec<Result<StreamChunk>> {
    let mut chunks = Vec::new();
    while let Ok(chunk) = receiver.try_recv() {
        chunks.push(chunk);
    }
    chunks
}

fn assert_fault_error(result: &Result<StreamChunk>, fault: FaultKind) {
    let error = match result {
        Err(error) => error,
        Ok(_) => {
            panic!("faulted participant must receive an error, not a token or success terminal")
        }
    };
    let expected = match fault {
        FaultKind::InvalidLogits => "engine sampling policies have no finite token",
        FaultKind::MissingLogits => "unified_decode returned None for decode item",
        FaultKind::ShortResult => "unified_decode returned 1 results for 2 requests",
    };
    assert!(
        error.to_string().contains(expected),
        "unexpected error: {error}"
    );
}

#[tokio::test]
async fn cpu_batch_sampling_failure_does_not_replay_committed_peer() {
    assert_batch_fault(FaultKind::InvalidLogits, 1).await;
}

#[tokio::test]
async fn cpu_batch_first_sampling_failure_still_consumes_later_peer() {
    assert_batch_fault(FaultKind::InvalidLogits, 0).await;
}

#[tokio::test]
async fn cpu_batch_missing_logits_does_not_replay_committed_peer() {
    assert_batch_fault(FaultKind::MissingLogits, 1).await;
}

#[tokio::test]
async fn cpu_batch_short_result_terminates_consumed_cohort_without_replay() {
    assert_batch_fault(FaultKind::ShortResult, 1).await;
}

async fn assert_batch_fault(fault: FaultKind, fault_index: usize) {
    assert!(fault_index < 2);
    let terminates_cohort = fault == FaultKind::ShortResult;
    let cfg = TinyLlamaConfig::default();
    let model = tiny_llama_model(&cfg);
    let model_config = model.config().clone();
    let observed = Arc::new(Mutex::new(Observation {
        model,
        decodes: vec![],
        released: vec![],
    }));
    let executor = Arc::new(FaultAfterForward {
        inner: LlmExecutor::new(
            Box::new(ObservedLlama {
                state: observed.clone(),
                config: model_config,
            }),
            tiny_model_info(&cfg),
        ),
        armed: AtomicBool::new(false),
        injected: Mutex::new(None),
        fault,
        fault_index,
    });
    let tokenizer = Arc::new(TinyTokenizer::new(cfg.vocab_size));
    let mut config = EngineConfig::default();
    config.backend.device = Device::CPU;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    // Real manager-owned allocations are distinct from the model's actual KV.
    let allocations = Arc::new(
        PagedKvCacheManager::new(
            Device::CPU,
            PagedKvCacheConfig {
                block_size: 16,
                max_gpu_blocks: 32,
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
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        Arc::new(crate::registry::GreedySampler),
        allocations.clone(),
        executor.clone(),
        Arc::new(crate::tensor_factory::CandleTensorFactory::new(Device::CPU)),
    )
    .unwrap();
    let mut streams = HashMap::new();
    for prompt in [
        "first actual request",
        "second actual request",
        "independent live peer",
    ] {
        let mut request = InferenceRequest::new(prompt, "tiny-llama-test");
        request.sampling_params.max_tokens = 4;
        request.sampling_params.temperature = 0.0;
        request.sampling_params.top_k = Some(cfg.vocab_size);
        request
            .metadata
            .insert("ferrum_ignore_eos".into(), serde_json::json!(true));
        let tokens = tokenizer.encode(prompt, true).unwrap();
        let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            tokens,
            Some(tokenizer.clone()),
            Some(cfg.vocab_size),
        );
        let (sender, receiver) = tokio::sync::mpsc::channel(16);
        sequence.stream_sender = Some(sender);
        streams.insert(request.id.clone(), receiver);
        engine
            .inner
            .sequences
            .write()
            .insert(request.id.clone(), sequence);
        scheduler.submit(request).await.unwrap();
    }
    let prefill = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(3))
        .await
        .unwrap();
    assert_eq!(prefill.requests.len(), 3);
    engine
        .inner
        .process_batch_legacy_split_with_test_sequences(&prefill)
        .await
        .unwrap();
    assert_eq!(allocations.list_handles().len(), 3);
    assert!(allocations.stats().used_memory_bytes > 0);

    let mut reference = tiny_llama_model(&cfg);
    let mut initial = HashMap::new();
    for scheduled in &prefill.requests {
        let id = &scheduled.request.id;
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[id];
        assert!(sequence.prefill_complete);
        assert_eq!(sequence.generated_tokens.len(), 1);
        let cache_id = sequence.model_cache_id().unwrap().to_owned();
        let prompt: Vec<_> = sequence
            .input_tokens
            .iter()
            .map(|token| token.get())
            .collect();
        let generated = sequence.generated_tokens.clone();
        assert_eq!(
            observed.lock().lengths(&cache_id),
            vec![prompt.len(); cfg.num_layers]
        );
        let reference_logits = reference.prefill(&cache_id, &prompt);
        assert_eq!(reference_logits.len(), cfg.vocab_size);
        let chunks = drain(streams.get_mut(id).unwrap());
        assert_eq!(chunks.len(), 1);
        let chunk = chunks[0].as_ref().unwrap();
        assert_eq!(chunk.token, Some(generated[0]));
        assert!(!chunk.text.is_empty());
        initial.insert(id.clone(), (cache_id, prompt.len(), generated));
    }
    let scheduled_decode = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(3))
        .await
        .unwrap();
    assert_eq!(scheduled_decode.requests.len(), 3);
    // Preserve the actual scheduler order; only hold its third participant out
    // of this dispatch so it can independently demonstrate resource isolation.
    let a = scheduled_decode.requests[1 - fault_index]
        .request
        .id
        .clone();
    let b = scheduled_decode.requests[fault_index].request.id.clone();
    let peer = scheduled_decode.requests[2].request.id.clone();
    let mut failed_batch = scheduled_decode.clone();
    failed_batch.requests.truncate(2);
    executor.armed.store(true, Ordering::SeqCst);
    engine
        .inner
        .process_batch_legacy_split_with_test_sequences(&failed_batch)
        .await
        .unwrap();
    assert_eq!(
        engine.inner.total_decode_tokens.load(Ordering::Relaxed),
        if terminates_cohort { 0 } else { 1 },
        "malformed row counts must not publish a guessed output prefix"
    );
    let (a_cache, a_len, a_tokens) = &initial[&a];
    let (b_cache, b_len, b_tokens) = &initial[&b];
    assert_eq!(executor.injected.lock().as_deref(), Some(b_cache.as_str()));
    let a_expected = reference.decode(a_cache, a_tokens[0].get(), *a_len as u32);
    let b_expected = reference.decode(b_cache, b_tokens[0].get(), *b_len as u32);
    {
        let state = observed.lock();
        // Check real per-layer KV before call-count diagnostics: a replay is
        // a physical advance, not harmless duplicate host dispatch.
        if terminates_cohort {
            assert!(!state.model.kv_caches.contains_key(a_cache));
        } else {
            assert_eq!(
                state.lengths(a_cache),
                vec![a_len + 1; cfg.num_layers],
                "unaffected participant must consume exactly one input"
            );
        }
        let a_outputs: Vec<_> = state
            .decodes
            .iter()
            .filter(|row| row.cache_id == *a_cache)
            .collect();
        assert_eq!(a_outputs.len(), 1);
        assert_eq!(a_outputs[0].position as usize, *a_len);
        assert_eq!(a_outputs[0].layer_lengths, vec![a_len + 1; cfg.num_layers]);
        assert_full_logits(&a_expected, &a_outputs[0].logits, cfg.vocab_size);
        let b_outputs: Vec<_> = state
            .decodes
            .iter()
            .filter(|row| row.cache_id == *b_cache)
            .collect();
        assert_eq!(
            b_outputs.len(),
            1,
            "B consumed input must not be silently retried"
        );
        assert_eq!(b_outputs[0].position as usize, *b_len);
        assert_eq!(b_outputs[0].layer_lengths, vec![b_len + 1; cfg.num_layers]);
        assert_full_logits(&b_expected, &b_outputs[0].logits, cfg.vocab_size);
        assert!(!state.model.kv_caches.contains_key(b_cache));
        let mut expected_released = vec![(b_cache.clone(), vec![b_len + 1; cfg.num_layers])];
        if terminates_cohort {
            expected_released.push((a_cache.clone(), vec![a_len + 1; cfg.num_layers]));
        }
        expected_released.sort();
        let mut actual_released = state.released.clone();
        actual_released.sort();
        assert_eq!(actual_released, expected_released);
    }
    {
        let sequences = engine.inner.sequences.read();
        if terminates_cohort {
            assert!(!sequences.contains_key(&a));
        } else {
            let sequence = &sequences[&a];
            assert_eq!(sequence.generated_tokens.len(), 2);
            assert_eq!(&sequence.generated_tokens[..1], a_tokens);
            assert_eq!(sequence.kv_cache_handle().unwrap().num_tokens(), a_len + 1);
        }
        assert!(!sequences.contains_key(&b));
    }
    let a_updates = drain(streams.get_mut(&a).unwrap());
    assert_eq!(a_updates.len(), 1);
    if terminates_cohort {
        assert_fault_error(&a_updates[0], fault);
        assert!(allocations.get_handle(a.clone()).is_none());
        assert_eq!(scheduler.trace_phase(&a), None);
    } else {
        let update = a_updates[0].as_ref().unwrap();
        assert!(!update.text.is_empty());
        assert_eq!(
            update.token,
            engine.inner.sequences.read()[&a]
                .generated_tokens
                .last()
                .copied()
        );
    }
    let b_updates = drain(streams.get_mut(&b).unwrap());
    assert_eq!(b_updates.len(), 1);
    assert_fault_error(&b_updates[0], fault);
    assert!(allocations.get_handle(b.clone()).is_none());
    assert_eq!(
        allocations.list_handles().len(),
        if terminates_cohort { 1 } else { 2 }
    );
    assert_eq!(scheduler.trace_phase(&b), None);

    let (peer_cache, peer_len, peer_tokens) = &initial[&peer];
    assert_eq!(
        observed.lock().lengths(peer_cache),
        vec![*peer_len; cfg.num_layers]
    );
    let peer_expected = reference.decode(peer_cache, peer_tokens[0].get(), *peer_len as u32);
    let mut peer_batch = scheduled_decode;
    peer_batch
        .requests
        .retain(|scheduled| scheduled.request.id == peer);
    engine
        .inner
        .process_batch_legacy_split_with_test_sequences(&peer_batch)
        .await
        .unwrap();
    {
        let state = observed.lock();
        let outputs: Vec<_> = state
            .decodes
            .iter()
            .filter(|row| row.cache_id == *peer_cache)
            .collect();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].layer_lengths, vec![peer_len + 1; cfg.num_layers]);
        assert_full_logits(&peer_expected, &outputs[0].logits, cfg.vocab_size);
    }
    assert_eq!(
        engine.inner.sequences.read()[&peer].generated_tokens.len(),
        2
    );
    assert!(allocations.get_handle(peer.clone()).is_some());
    // All survivors finish through normal scheduling/max_tokens; no reset or
    // direct cache deletion is used to make the ownership checks pass.
    for _ in 2..4 {
        let finish = scheduler
            .next_batch(ferrum_interfaces::BatchHint::simple(3))
            .await
            .unwrap();
        assert_eq!(finish.requests.len(), if terminates_cohort { 1 } else { 2 });
        engine
            .inner
            .process_batch_legacy_split_with_test_sequences(&finish)
            .await
            .unwrap();
    }
    assert!(engine.inner.sequences.read().is_empty());
    assert!(observed.lock().model.kv_caches.is_empty());
    let mut released: Vec<_> = observed
        .lock()
        .released
        .iter()
        .map(|(id, _)| id.clone())
        .collect();
    released.sort();
    let mut expected = vec![a_cache.clone(), b_cache.clone(), peer_cache.clone()];
    expected.sort();
    assert_eq!(
        released, expected,
        "each physical model cache released exactly once"
    );
    assert!(allocations.list_handles().is_empty());
    assert_eq!(allocations.stats().used_memory_bytes, 0);
    assert_eq!(scheduler.trace_snapshot().active_len, 0);
}
