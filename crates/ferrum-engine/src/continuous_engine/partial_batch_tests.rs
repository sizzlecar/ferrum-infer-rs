//! Post-forward output failures must never replay an already consumed batch.

use super::*;
use ferrum_execution::model_executor::{
    DecodeInput, DecodeOutput, ExecutorCapabilities, ExecutorStatus, KvSlotRequest,
    KvSlotReservation, PrefillInput, PrefillOutput,
};
use ferrum_execution::token_execution::*;
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kv::{PagedKvCacheConfig, PagedKvCacheManager};
use ferrum_models::common::llm::{ModelKvReservation, ModelTokenOutcome, ModelTokenWork};
use ferrum_models::models::LlamaFamilyModel;
use ferrum_models::test_support::{
    tiny_llama_model, tiny_model_info, TinyLlamaConfig, TinyTokenizer,
};
use ferrum_models::{DecoderOnlyLLM, LlmExecutor, LlmRuntimeConfig};
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
    prepared: Vec<String>,
    prefills: Vec<(String, Vec<f32>)>,
    decodes: Vec<DecodeObservation>,
    released: Vec<(String, Vec<usize>)>,
    inject: Option<(String, FaultKind)>,
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

// The observer owns no synthetic cache. Every token forward/reservation/close
// is delegated to the real model. Corrupt only the returned row after retaining
// its full numerical output and actual per-layer KV lengths.
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
        let mut state = self.state.lock();
        state.prepared.push(id.to_owned());
        state.model.prepare_kv_capacity(id, capacity);
    }

    fn reserve_kv_slots(
        &mut self,
        requests: &[KvSlotRequest],
    ) -> Result<Option<KvSlotReservation>> {
        self.state.lock().model.reserve_kv_slots(requests)
    }

    fn reserve_token_kv_slots(&mut self, requests: &[KvSlotRequest]) -> Result<ModelKvReservation> {
        self.state.lock().model.reserve_token_kv_slots(requests)
    }

    fn supports_direct_token_execution(&self) -> bool {
        self.state.lock().model.supports_direct_token_execution()
    }

    fn execute_token_work(&mut self, id: &str, work: ModelTokenWork<'_>) -> ModelTokenOutcome {
        let mut state = self.state.lock();
        let ModelTokenOutcome::Terminal(Ok(mut logits)) = state.model.execute_token_work(id, work)
        else {
            panic!("real CPU model must complete before injected output failure")
        };
        assert_eq!(logits.len(), self.config.vocab_size);
        assert!(logits.iter().all(|value| value.is_finite()));
        match work {
            ModelTokenWork::Prefill { .. } => state.prefills.push((id.to_owned(), logits.clone())),
            ModelTokenWork::Decode { position, .. } => {
                state.record(id, position, &logits);
                if state
                    .inject
                    .as_ref()
                    .is_some_and(|(target, _)| target == id)
                {
                    match state.inject.take().unwrap().1 {
                        FaultKind::InvalidLogits => logits.fill(f32::NEG_INFINITY),
                        FaultKind::MissingLogits => logits.clear(),
                        FaultKind::ShortResult => {
                            unreachable!("row-count fault is after the real batch")
                        }
                    }
                }
            }
        }
        ModelTokenOutcome::Terminal(Ok(logits))
    }

    fn close_token_sequence(&mut self, id: &str) -> Result<()> {
        let mut state = self.state.lock();
        let lengths = state.lengths(id);
        state.model.close_token_sequence(id)?;
        state.released.push((id.into(), lengths));
        Ok(())
    }

    fn prefill(&mut self, _: &str, _: &[u32]) -> Vec<f32> {
        panic!("no legacy model entry")
    }
    fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
        panic!("no legacy model entry")
    }
    fn release(&mut self, _: &str) {
        panic!("token closure must use the checked model operation")
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
    observed: Arc<Mutex<Observation>>,
    caches: Mutex<HashMap<ExecutorSequence, String>>,
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
    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        let before = self.observed.lock().prepared.len();
        let outcome = self.inner.admit_tokens(input).await?;
        if let TokenAdmissionOutcome::Admitted(frontier) = &outcome {
            let state = self.observed.lock();
            assert_eq!(state.prepared.len(), before + 1);
            self.caches
                .lock()
                .insert(frontier.sequence().clone(), state.prepared[before].clone());
        }
        Ok(outcome)
    }
    async fn execute_tokens(&self, inputs: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        let armed = self.armed.swap(false, Ordering::SeqCst);
        if armed {
            assert_eq!(
                inputs.len(),
                2,
                "fault targets exactly the admitted A/B cohort"
            );
            assert!(inputs
                .iter()
                .all(|input| matches!(input.work(), TokenWork::Decode { .. })));
            let cache = self.caches.lock()[inputs[self.fault_index].sequence()].clone();
            *self.injected.lock() = Some(cache.clone());
            if self.fault != FaultKind::ShortResult {
                self.observed.lock().inject = Some((cache, self.fault));
            }
        }
        let outcome = self.inner.execute_tokens(inputs).await?;
        if armed && self.fault == FaultKind::ShortResult {
            let ExecutionBatchOutcome::Completed(batch) = outcome else {
                panic!("real CPU cohort must complete before row-count corruption")
            };
            let mut completed = batch.into_completions();
            assert_eq!(completed.len(), 2);
            for completion in &completed {
                let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
                    completion.product()
                else {
                    panic!("real full-vocabulary decode output")
                };
                assert_eq!(logits.len(), self.info().vocab_size);
                assert!(logits.iter().all(|value| value.is_finite()));
            }
            // Drop exactly one authentic result after both actual model calls.
            // Other rows keep their original index and incarnation; the new
            // protocol can preserve them without guessing a positional prefix.
            drop(completed.remove(self.fault_index));
            return Ok(ExecutionBatchOutcome::ContractViolation(
                ExecutionContractViolation::new(
                    FerrumError::backend("injected short token result after actual CPU decode"),
                    (completed, Vec::new(), Vec::new()),
                ),
            ));
        }
        Ok(outcome)
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.inner.finish_tokens(sequence, reason).await
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.inner.cancel_tokens(sequence)
    }
    async fn prefill(&self, _: &PrefillInput) -> Result<PrefillOutput> {
        panic!("no Tensor bridge")
    }
    async fn decode(&self, _: &DecodeInput) -> Result<DecodeOutput> {
        panic!("no Tensor bridge")
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
    assert_fault_cause(error, fault);
}

fn assert_fault_cause(error: &FerrumError, fault: FaultKind) {
    let expected = match fault {
        FaultKind::InvalidLogits => "engine sampling policies have no finite token",
        FaultKind::MissingLogits => "returned 0 logits for vocabulary",
        FaultKind::ShortResult => "injected short token result after actual CPU decode",
    };
    assert!(
        error.to_string().contains(expected),
        "unexpected error: {error}"
    );
}

fn sample_reference(
    sequence: &mut SequenceState,
    logits: &[f32],
    tokenizer: &TinyTokenizer,
) -> TokenId {
    // Only the product sampling policy is shared. Numerical logits and KV
    // come from a separate tiny model that never sees an injected fault.
    sequence
        .sample_and_commit_with_processors_and_tokenizer(&mut logits.to_vec(), Some(tokenizer))
        .unwrap()
}

fn assert_visible_update(chunk: &StreamChunk, expected: TokenId, tokenizer: &TinyTokenizer) {
    assert_eq!(chunk.token, Some(expected));
    assert_eq!(chunk.text, tokenizer.decode(&[expected], true).unwrap());
    assert!(!chunk.text.is_empty());
    assert!(chunk.finish_reason.is_none());
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
async fn cpu_batch_short_result_preserves_identified_peer_without_replay() {
    assert_batch_fault(FaultKind::ShortResult, 1).await;
}

async fn assert_batch_fault(fault: FaultKind, fault_index: usize) {
    assert!(fault_index < 2);
    let cfg = TinyLlamaConfig::default();
    let model = tiny_llama_model(&cfg);
    let model_config = model.config().clone();
    let observed = Arc::new(Mutex::new(Observation {
        model,
        prepared: vec![],
        prefills: vec![],
        decodes: vec![],
        released: vec![],
        inject: None,
    }));
    let executor = Arc::new(FaultAfterForward {
        inner: LlmExecutor::new(
            Box::new(ObservedLlama {
                state: observed.clone(),
                config: model_config,
            }),
            tiny_model_info(&cfg),
        ),
        observed: observed.clone(),
        caches: Mutex::new(HashMap::new()),
        armed: AtomicBool::new(false),
        injected: Mutex::new(None),
        fault,
        fault_index,
    });
    let tokenizer = Arc::new(TinyTokenizer::new(cfg.vocab_size));
    let mut config = EngineConfig::default();
    config.backend.device = Device::CPU;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    // Retained constructor dependency only: the token path must not create a
    // duplicate Engine KV allocation beside the model's real physical cache.
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
    let mut reference_sequences = HashMap::new();
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
        request.metadata.insert(
            PROMPT_TOKENS_METADATA_KEY.into(),
            serde_json::json!(tokens.len()),
        );
        reference_sequences.insert(
            request.id.clone(),
            SequenceState::new_with_tokenizer_and_model_vocab_size(
                request.clone(),
                tokens.clone(),
                Some(tokenizer.clone()),
                Some(cfg.vocab_size),
            ),
        );
        let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            tokens,
            Some(tokenizer.clone()),
            Some(cfg.vocab_size),
        );
        let (sender, receiver) = tokio::sync::mpsc::channel(16);
        sequence.stream_sender = Some(sender);
        sequence.request_slot = Some(RequestSlotLease::open(&engine.inner, request.id.clone()));
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
    for scheduled in &prefill.requests {
        assert!(
            engine
                .inner
                .probe_executor_prefill_admission_for_test(&scheduled.request)
                .await
        );
        engine
            .inner
            .sequences
            .write()
            .get_mut(&scheduled.request.id)
            .unwrap()
            .request_slot
            .as_mut()
            .unwrap()
            .admit(&engine.inner);
    }
    engine.inner.process_batch(&prefill).await.unwrap();
    assert!(allocations.list_handles().is_empty());
    assert_eq!(allocations.stats().used_memory_bytes, 0);
    assert_eq!(observed.lock().model.kv_caches.len(), 3);

    let mut reference = tiny_llama_model(&cfg);
    let mut initial = HashMap::new();
    let mut identities = HashMap::new();
    for scheduled in &prefill.requests {
        let id = &scheduled.request.id;
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[id];
        assert!(sequence.prefill_complete);
        assert_eq!(sequence.generated_tokens.len(), 1);
        let frontier = sequence.execution_frontier.as_ref().unwrap();
        let cache_id = executor.caches.lock()[frontier.sequence()].clone();
        identities.insert(id.clone(), frontier.sequence().clone());
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
        let reference_token = sample_reference(
            reference_sequences.get_mut(id).unwrap(),
            &reference_logits,
            &tokenizer,
        );
        assert_eq!(generated, [reference_token]);
        let state = observed.lock();
        let prefills: Vec<_> = state
            .prefills
            .iter()
            .filter(|(cache, _)| cache == &cache_id)
            .collect();
        assert_eq!(prefills.len(), 1);
        assert_full_logits(&reference_logits, &prefills[0].1, cfg.vocab_size);
        assert_eq!(frontier.consumed_tokens(), prompt.len());
        drop(state);
        let chunks = drain(streams.get_mut(id).unwrap());
        assert_eq!(chunks.len(), 1);
        let chunk = chunks[0].as_ref().unwrap();
        assert_visible_update(chunk, reference_token, &tokenizer);
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
    let batch_result = engine.inner.process_batch(&failed_batch).await;
    if fault == FaultKind::MissingLogits {
        // The producer already ended this incarnation and retained its
        // product-validation error alongside the actual close disposition.
        assert_fault_cause(&batch_result.unwrap_err(), fault);
    } else {
        batch_result.unwrap();
    }
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(
        engine.inner.total_decode_tokens.load(Ordering::Relaxed),
        1,
        "the authentic healthy completion must survive each output fault"
    );
    let (a_cache, a_len, a_tokens) = &initial[&a];
    let (b_cache, b_len, b_tokens) = &initial[&b];
    assert_eq!(executor.injected.lock().as_deref(), Some(b_cache.as_str()));
    let a_expected = reference.decode(a_cache, a_tokens[0].get(), *a_len as u32);
    let b_expected = reference.decode(b_cache, b_tokens[0].get(), *b_len as u32);
    let a_expected_token = sample_reference(
        reference_sequences.get_mut(&a).unwrap(),
        &a_expected,
        &tokenizer,
    );
    {
        let state = observed.lock();
        // Check real per-layer KV before call-count diagnostics: a replay is
        // a physical advance, not harmless duplicate host dispatch.
        assert_eq!(
            state.lengths(a_cache),
            vec![a_len + 1; cfg.num_layers],
            "unaffected participant must consume exactly one input"
        );
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
        expected_released.sort();
        let mut actual_released = state.released.clone();
        actual_released.sort();
        assert_eq!(actual_released, expected_released);
    }
    {
        let sequences = engine.inner.sequences.read();
        let sequence = &sequences[&a];
        assert_eq!(sequence.generated_tokens.len(), 2);
        assert_eq!(&sequence.generated_tokens[..1], a_tokens);
        let frontier = sequence.execution_frontier.as_ref().unwrap();
        assert_eq!(frontier.sequence(), &identities[&a]);
        assert_eq!(frontier.consumed_tokens(), a_len + 1);
        assert!(!sequences.contains_key(&b));
    }
    let a_updates = drain(streams.get_mut(&a).unwrap());
    assert_eq!(a_updates.len(), 1);
    let update = a_updates[0].as_ref().unwrap();
    assert_visible_update(update, a_expected_token, &tokenizer);
    assert_eq!(
        update.token,
        engine.inner.sequences.read()[&a]
            .generated_tokens
            .last()
            .copied()
    );
    let b_updates = drain(streams.get_mut(&b).unwrap());
    assert_eq!(b_updates.len(), 1);
    assert_fault_error(&b_updates[0], fault);
    assert!(identities[&b].observed_end().is_some());
    assert!(identities[&a].observed_end().is_none());
    assert!(identities[&peer].observed_end().is_none());
    assert!(allocations.list_handles().is_empty());
    assert_eq!(scheduler.trace_phase(&b), None);

    let (peer_cache, peer_len, peer_tokens) = &initial[&peer];
    assert_eq!(
        observed.lock().lengths(peer_cache),
        vec![*peer_len; cfg.num_layers]
    );
    let peer_expected = reference.decode(peer_cache, peer_tokens[0].get(), *peer_len as u32);
    let peer_expected_token = sample_reference(
        reference_sequences.get_mut(&peer).unwrap(),
        &peer_expected,
        &tokenizer,
    );
    let mut peer_batch = scheduled_decode;
    peer_batch
        .requests
        .retain(|scheduled| scheduled.request.id == peer);
    engine.inner.process_batch(&peer_batch).await.unwrap();
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
    assert_eq!(
        engine.inner.sequences.read()[&peer]
            .execution_frontier
            .as_ref()
            .unwrap()
            .consumed_tokens(),
        peer_len + 1
    );
    let peer_updates = drain(streams.get_mut(&peer).unwrap());
    assert_eq!(peer_updates.len(), 1);
    assert_visible_update(
        peer_updates[0].as_ref().unwrap(),
        peer_expected_token,
        &tokenizer,
    );
    // All survivors finish through normal scheduling/max_tokens; no reset or
    // direct cache deletion is used to make the ownership checks pass.
    for previous_outputs in 2..4 {
        let finish = scheduler
            .next_batch(ferrum_interfaces::BatchHint::simple(3))
            .await
            .unwrap();
        assert_eq!(finish.requests.len(), 2);
        let mut expected = HashMap::new();
        for scheduled in &finish.requests {
            let id = &scheduled.request.id;
            let (cache_id, prompt_len, _) = &initial[id];
            let oracle = reference_sequences.get_mut(id).unwrap();
            assert_eq!(oracle.generated_tokens.len(), previous_outputs);
            let position = prompt_len + previous_outputs - 1;
            {
                let sequences = engine.inner.sequences.read();
                let sequence = &sequences[id];
                assert_eq!(sequence.generated_tokens, oracle.generated_tokens);
                let frontier = sequence.execution_frontier.as_ref().unwrap();
                assert_eq!(frontier.sequence(), &identities[id]);
                assert_eq!(frontier.consumed_tokens(), position);
            }
            let logits = reference.decode(
                cache_id,
                oracle.generated_tokens.last().unwrap().get(),
                position as u32,
            );
            let token = sample_reference(oracle, &logits, &tokenizer);
            expected.insert(id.clone(), (position, logits, token));
        }
        engine.inner.process_batch(&finish).await.unwrap();
        for scheduled in &finish.requests {
            let id = &scheduled.request.id;
            let (cache_id, prompt_len, _) = &initial[id];
            let (position, logits, token) = &expected[id];
            {
                let state = observed.lock();
                let outputs: Vec<_> = state
                    .decodes
                    .iter()
                    .filter(|row| row.cache_id == *cache_id)
                    .collect();
                assert_eq!(
                    outputs.len(),
                    previous_outputs,
                    "no retry or skipped forward"
                );
                let output = outputs.last().unwrap();
                assert_eq!(output.position as usize, *position);
                assert_eq!(output.layer_lengths, vec![position + 1; cfg.num_layers]);
                assert_full_logits(logits, &output.logits, cfg.vocab_size);
                if previous_outputs < 3 {
                    assert_eq!(state.lengths(cache_id), vec![position + 1; cfg.num_layers]);
                } else {
                    assert!(!state.model.kv_caches.contains_key(cache_id));
                    assert_eq!(
                        state
                            .released
                            .iter()
                            .filter(|(cache, _)| cache == cache_id)
                            .map(|(_, lengths)| lengths.clone())
                            .collect::<Vec<_>>(),
                        [vec![prompt_len + 3; cfg.num_layers]]
                    );
                }
            }
            let chunks = drain(streams.get_mut(id).unwrap());
            assert_eq!(chunks.len(), if previous_outputs == 3 { 2 } else { 1 });
            assert_visible_update(chunks[0].as_ref().unwrap(), *token, &tokenizer);
            if previous_outputs == 3 {
                let terminal = chunks[1].as_ref().unwrap();
                assert!(terminal.token.is_none());
                assert!(terminal.text.is_empty());
                assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
                let usage = terminal.usage.as_ref().unwrap();
                assert_eq!(usage.prompt_tokens, *prompt_len);
                assert_eq!(usage.completion_tokens, 4);
                assert!(!engine.inner.sequences.read().contains_key(id));
                assert!(identities[id].observed_end().is_some());
            } else {
                let sequences = engine.inner.sequences.read();
                assert_eq!(
                    sequences[id].generated_tokens,
                    reference_sequences[id].generated_tokens
                );
                assert_eq!(
                    sequences[id]
                        .execution_frontier
                        .as_ref()
                        .unwrap()
                        .consumed_tokens(),
                    position + 1
                );
                assert!(identities[id].observed_end().is_none());
            }
        }
    }
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
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
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
    assert!(identities
        .values()
        .all(|identity| identity.observed_end().is_some()));
    assert_eq!(engine.inner.total_decode_tokens.load(Ordering::Relaxed), 6);
    let state = observed.lock();
    assert_eq!(state.prepared.len(), 3);
    assert_eq!(state.prefills.len(), 3);
    assert_eq!(
        state.decodes.len(),
        7,
        "three forwards per survivor and one failed peer"
    );
    assert!(state.inject.is_none());
}
