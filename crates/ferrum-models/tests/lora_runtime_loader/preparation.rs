use super::execution::{metadata, model_adapter, tokens};
use super::*;
use ferrum_execution::model_executor::{
    DecodeInput, KvSlotRequest, KvSlotReservation, PrefillInput, UnifiedBatch, UnifiedBatchItem,
};
use ferrum_execution::ModelExecutor;
use ferrum_interfaces::adapter::AdapterExecutionSupport;
use ferrum_models::common::{DecoderOnlyLLM, LlmRuntimeConfig};
use ferrum_models::models::LlamaFamilyModel;
use ferrum_models::test_support::{tiny_llama_model, tiny_model_info, TinyLlamaConfig};
use ferrum_models::LlmExecutor;
use ferrum_types::Result;
use parking_lot::Mutex;
use std::sync::Arc;

// Keep access to the actual model's live KV map after it is boxed by the
// executor. Forward, loading, binding and reservation remain real CPU paths.
struct Observation {
    model: LlamaFamilyModel<CpuBackend>,
    forwards: usize,
    fail_reservation: bool,
}

struct ObservedLlama {
    state: Arc<Mutex<Observation>>,
    config: LlmRuntimeConfig,
}

impl DecoderOnlyLLM for ObservedLlama {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.config
    }

    fn adapter_execution_support(&self) -> AdapterExecutionSupport {
        self.state.lock().model.adapter_execution_support()
    }

    fn preload_lora_adapter(&mut self, adapter: &ActiveLoraAdapter) -> Result<()> {
        self.state.lock().model.preload_lora_adapter(adapter)
    }

    fn set_lora_adapter_for_cache(
        &mut self,
        cache_id: &str,
        adapter: Option<ActiveLoraAdapter>,
    ) -> Result<()> {
        self.state
            .lock()
            .model
            .set_lora_adapter_for_cache(cache_id, adapter)
    }

    fn prepare_kv_capacity(&mut self, cache_id: &str, capacity_hint: usize) {
        self.state
            .lock()
            .model
            .prepare_kv_capacity(cache_id, capacity_hint);
    }

    fn reserve_kv_slots(
        &mut self,
        requests: &[KvSlotRequest],
    ) -> Result<Option<KvSlotReservation>> {
        let mut state = self.state.lock();
        let reservation = state.model.reserve_kv_slots(requests)?;
        if state.fail_reservation {
            Err(ferrum_types::FerrumError::resource_exhausted(
                "injected capacity rejection after cache preparation",
            ))
        } else {
            Ok(reservation)
        }
    }

    fn prefill(&mut self, cache_id: &str, tokens: &[u32]) -> Vec<f32> {
        let mut state = self.state.lock();
        state.forwards += 1;
        state.model.prefill(cache_id, tokens)
    }

    fn decode(&mut self, cache_id: &str, token: u32, pos: u32) -> Vec<f32> {
        let mut state = self.state.lock();
        state.forwards += 1;
        state.model.decode(cache_id, token, pos)
    }

    fn release(&mut self, cache_id: &str) {
        self.state.lock().model.release(cache_id);
    }
}

fn observed(cfg: &TinyLlamaConfig) -> (LlmExecutor, Arc<Mutex<Observation>>) {
    let model = tiny_llama_model(cfg);
    let config = model.config().clone();
    let state = Arc::new(Mutex::new(Observation {
        model,
        forwards: 0,
        fail_reservation: false,
    }));
    let executor = LlmExecutor::new(
        Box::new(ObservedLlama {
            state: state.clone(),
            config,
        }),
        tiny_model_info(cfg),
    );
    (executor, state)
}

fn bad_adapter(path: &std::path::Path) -> HashMap<String, serde_json::Value> {
    let mut selected = metadata(&path.join("missing-adapter"));
    selected.insert("ferrum_lora_adapter".into(), serde_json::json!("missing"));
    selected
}

fn hinted(mut selected: HashMap<String, serde_json::Value>) -> HashMap<String, serde_json::Value> {
    selected.insert("ferrum_kv_capacity_hint".into(), serde_json::json!(16));
    selected
}

fn snapshot(state: &Mutex<Observation>) -> (Vec<(String, Vec<usize>)>, u64, usize) {
    let state = state.lock();
    let mut caches: Vec<_> = state
        .model
        .kv_caches
        .iter()
        .map(|(id, layers)| (id.clone(), layers.iter().map(|layer| layer.len).collect()))
        .collect();
    caches.sort_by(|a, b| a.0.cmp(&b.0));
    let bindings = state.model.lora_metrics_snapshot().unwrap()["active_cache_bindings"]
        .as_u64()
        .unwrap();
    (caches, bindings, state.forwards)
}

#[tokio::test]
async fn lora_preparation_load_failure_leaves_no_hidden_prefill_cache() {
    let cfg = TinyLlamaConfig::default();
    let directory = model_adapter(&cfg);
    let (executor, state) = observed(&cfg);
    let valid =
        PrefillInput::new(tokens(&[0, 1])).with_metadata(hinted(metadata(directory.path())));
    let invalid =
        PrefillInput::new(tokens(&[0, 1])).with_metadata(hinted(bad_adapter(directory.path())));

    assert!(executor.prefill(&invalid).await.is_err());
    assert_eq!(snapshot(&state), (vec![], 0, 0));
    assert!(executor
        .batch_prefill(&[valid.clone(), invalid])
        .await
        .is_err());
    assert_eq!(snapshot(&state), (vec![], 0, 0));
    // A successfully preloaded adapter may remain cached and must be reusable.
    let output = executor.prefill(&valid).await.unwrap();
    assert_eq!(snapshot(&state).1, 1);
    executor.release_cache(&output.kv_cache.cache_id());
    assert!(snapshot(&state).0.is_empty());
    assert_eq!(snapshot(&state).1, 0);
}

#[tokio::test]
async fn lora_preparation_failed_batches_preserve_existing_adapter_and_kv() {
    let cfg = TinyLlamaConfig::default();
    let directory = model_adapter(&cfg);
    let selected = metadata(directory.path());
    let (executor, state) = observed(&cfg);
    let (reference, _) = observed(&cfg);
    let prompt = PrefillInput::new(tokens(&[0, 1])).with_metadata(selected.clone());
    let initial = executor.prefill(&prompt).await.unwrap();
    let expected = reference.prefill(&prompt).await.unwrap();
    let before = snapshot(&state);
    assert_eq!(before.1, 1);
    let bad = bad_adapter(directory.path());
    let continuation = PrefillInput::new(tokens(&[2])).with_kv_cache(initial.kv_cache.clone());
    assert!(executor
        .prefill(&continuation.clone().with_metadata(bad.clone()))
        .await
        .is_err());
    assert!(executor
        .batch_prefill(&[
            continuation,
            PrefillInput::new(tokens(&[2])).with_metadata(bad.clone()),
        ])
        .await
        .is_err());
    let base = DecodeInput::new(tokens(&[2]), initial.kv_cache.clone());
    let invalid = base.clone().with_metadata(bad.clone());
    assert!(executor.decode(&invalid).await.is_err());
    assert!(executor.batch_decode(&[base, invalid]).await.is_err());
    let item = |id: &str, metadata| UnifiedBatchItem {
        seq_id: id.into(),
        q_tokens: vec![2],
        kv_cache: initial.kv_cache.clone(),
        recurrent_state: None,
        pos_offset: 2,
        is_final_chunk: true,
        metadata,
        logits_policy: Default::default(),
    };
    assert!(executor
        .unified_decode(&UnifiedBatch {
            items: vec![
                item(&initial.kv_cache.cache_id(), HashMap::new()),
                item("caller-owned-not-yet-executed", bad),
            ]
        })
        .await
        .is_err());
    assert_eq!(snapshot(&state), before);

    let actual = executor
        .decode(&DecodeInput::new(tokens(&[2]), initial.kv_cache).with_metadata(selected.clone()))
        .await
        .unwrap();
    let expected = reference
        .decode(&DecodeInput::new(tokens(&[2]), expected.kv_cache).with_metadata(selected))
        .await
        .unwrap();
    assert_eq!(
        actual.logits.to_vec_f32().unwrap(),
        expected.logits.to_vec_f32().unwrap()
    );
    executor.release_cache(&actual.kv_cache.cache_id());
    reference.release_cache(&expected.kv_cache.cache_id());
}

#[tokio::test]
async fn lora_preparation_capacity_failure_preserves_existing_and_discards_new_caches() {
    let cfg = TinyLlamaConfig::default();
    let directory = model_adapter(&cfg);
    let selected = metadata(directory.path());
    let (executor, state) = observed(&cfg);
    let (reference, _) = observed(&cfg);
    let prompt = PrefillInput::new(tokens(&[0, 1])).with_metadata(selected.clone());
    let initial = executor.prefill(&prompt).await.unwrap();
    let expected = reference.prefill(&prompt).await.unwrap();
    let before = snapshot(&state);
    state.lock().fail_reservation = true;
    let fresh = || PrefillInput::new(tokens(&[2])).with_metadata(hinted(selected.clone()));
    assert!(executor.prefill(&fresh()).await.is_err());
    assert_eq!(snapshot(&state), before);
    assert!(executor
        .batch_prefill(&[
            PrefillInput::new(tokens(&[2])).with_kv_cache(initial.kv_cache.clone()),
            fresh(),
            fresh(),
        ])
        .await
        .is_err());
    assert_eq!(snapshot(&state), before);
    let base = DecodeInput::new(tokens(&[2]), initial.kv_cache.clone());
    assert!(executor.decode(&base).await.is_err());
    assert!(executor.batch_decode(&[base]).await.is_err());
    assert!(executor
        .unified_decode(&UnifiedBatch {
            items: vec![UnifiedBatchItem {
                seq_id: initial.kv_cache.cache_id(),
                q_tokens: vec![2],
                kv_cache: initial.kv_cache.clone(),
                recurrent_state: None,
                pos_offset: 2,
                is_final_chunk: true,
                metadata: HashMap::new(),
                logits_policy: Default::default(),
            }]
        })
        .await
        .is_err());
    assert_eq!(snapshot(&state), before);
    state.lock().fail_reservation = false;
    let actual = executor
        .decode(&DecodeInput::new(tokens(&[2]), initial.kv_cache).with_metadata(selected.clone()))
        .await
        .unwrap();
    let expected = reference
        .decode(&DecodeInput::new(tokens(&[2]), expected.kv_cache).with_metadata(selected))
        .await
        .unwrap();
    assert_eq!(
        actual.logits.to_vec_f32().unwrap(),
        expected.logits.to_vec_f32().unwrap()
    );
    executor.release_cache(&actual.kv_cache.cache_id());
    reference.release_cache(&expected.kv_cache.cache_id());
}
