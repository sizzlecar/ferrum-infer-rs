use super::execution::{
    admit, decode, finish, logits, model_adapter, prefill, selection, try_admit,
};
use super::*;
use ferrum_execution::model_executor::{KvSlotRequest, KvSlotReservation};
use ferrum_execution::token_execution::ExecutionProduct;
use ferrum_interfaces::adapter::{AdapterExecutionSupport, LoraAdapterSelection};
use ferrum_models::common::llm::{ModelKvReservation, ModelTokenOutcome, ModelTokenWork};
use ferrum_models::common::{DecoderOnlyLLM, LlmRuntimeConfig};
use ferrum_models::models::LlamaFamilyModel;
use ferrum_models::test_support::{tiny_llama_model, tiny_model_info, TinyLlamaConfig};
use ferrum_models::LlmExecutor;
use ferrum_types::{FerrumError, RequestId, Result};
use parking_lot::Mutex;
use std::sync::Arc;

// Retain observation of the actual model after boxing it. Preparation,
// forward, adapter binding and checked release remain real CPU operations.
struct Observation {
    model: LlamaFamilyModel<CpuBackend>,
    forwards: usize,
    fail_preparation: bool,
    failed_preparations: Vec<(String, Vec<usize>)>,
    releases: Vec<(String, Vec<usize>)>,
}

struct ObservedLlama {
    state: Arc<Mutex<Observation>>,
    config: LlmRuntimeConfig,
}

impl DecoderOnlyLLM for ObservedLlama {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.config
    }

    fn supports_direct_token_execution(&self) -> bool {
        self.state.lock().model.supports_direct_token_execution()
    }

    fn adapter_execution_support(&self) -> AdapterExecutionSupport {
        self.state.lock().model.adapter_execution_support()
    }

    fn lora_metrics_snapshot(&self) -> Option<serde_json::Value> {
        self.state.lock().model.lora_metrics_snapshot()
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

    fn prepare_token_kv(
        &mut self,
        cache_id: &str,
        capacity_hint: usize,
        prompt_tokens: usize,
    ) -> Result<ModelKvReservation> {
        let mut state = self.state.lock();
        let reservation = state
            .model
            .prepare_token_kv(cache_id, capacity_hint, prompt_tokens)?;
        if state.fail_preparation {
            assert!(matches!(reservation, ModelKvReservation::Reserved(_)));
            let lengths = state
                .model
                .kv_caches
                .get(cache_id)
                .expect("fault must follow actual cache preparation")
                .iter()
                .map(|layer| layer.len)
                .collect();
            state
                .failed_preparations
                .push((cache_id.to_owned(), lengths));
            Err(FerrumError::resource_exhausted(
                "injected capacity rejection after actual cache preparation",
            ))
        } else {
            Ok(reservation)
        }
    }

    fn reserve_kv_slots(
        &mut self,
        requests: &[KvSlotRequest],
    ) -> Result<Option<KvSlotReservation>> {
        self.state.lock().model.reserve_kv_slots(requests)
    }

    fn execute_token_work(
        &mut self,
        cache_id: &str,
        work: ModelTokenWork<'_>,
    ) -> ModelTokenOutcome {
        let mut state = self.state.lock();
        state.forwards += 1;
        state.model.execute_token_work(cache_id, work)
    }

    fn close_token_sequence(&mut self, cache_id: &str) -> Result<()> {
        let mut state = self.state.lock();
        let lengths = state
            .model
            .kv_caches
            .get(cache_id)
            .expect("checked close must retain the actual prepared cache")
            .iter()
            .map(|layer| layer.len)
            .collect();
        state.model.close_token_sequence(cache_id)?;
        assert!(!state.model.kv_caches.contains_key(cache_id));
        state.releases.push((cache_id.to_owned(), lengths));
        Ok(())
    }

    fn prefill(&mut self, _cache_id: &str, _tokens: &[u32]) -> Vec<f32> {
        panic!("token test must not call the old model prefill entry")
    }

    fn decode(&mut self, _cache_id: &str, _token: u32, _pos: u32) -> Vec<f32> {
        panic!("token test must not call the old model decode entry")
    }

    fn release(&mut self, _cache_id: &str) {
        panic!("token test requires checked model closure")
    }
}

fn observed(cfg: &TinyLlamaConfig) -> (LlmExecutor, Arc<Mutex<Observation>>) {
    let model = tiny_llama_model(cfg);
    let config = model.config().clone();
    let state = Arc::new(Mutex::new(Observation {
        model,
        forwards: 0,
        fail_preparation: false,
        failed_preparations: Vec::new(),
        releases: Vec::new(),
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
    let missing_path = directory.path().join("missing-adapter");
    let invalid = LoraAdapterSelection {
        name: "missing",
        path: missing_path.to_str().unwrap(),
    };
    let (executor, state) = observed(&cfg);
    let rejected_request = RequestId::new();
    assert!(
        try_admit(&executor, &rejected_request, &[0, 1], Some(invalid))
            .await
            .is_err()
    );
    assert_eq!(snapshot(&state), (vec![], 0, 0));
    assert!(state.lock().releases.is_empty());

    // A failed unpublished admission does not poison its request identity.
    let start = admit(
        &executor,
        &rejected_request,
        &[0, 1],
        Some(selection(directory.path())),
    )
    .await;
    assert_eq!(snapshot(&state).1, 1);
    let output = prefill(&executor, &start, 2, 2).await;
    assert!(!logits(&output).is_empty());
    finish(&executor, output.next_frontier(), 2, 1).await;
    assert!(snapshot(&state).0.is_empty());
    assert_eq!(snapshot(&state).1, 0);
    assert_eq!(state.lock().releases.len(), 1);

    // Reusable adapter weights may stay cached, but a subsequent load failure
    // must not create a new sequence cache or change any execution state.
    let before = snapshot(&state);
    assert!(
        try_admit(&executor, &RequestId::new(), &[0, 1], Some(invalid))
            .await
            .is_err()
    );
    assert_eq!(snapshot(&state), before);
    assert_eq!(state.lock().releases.len(), 1);
}

#[tokio::test]
async fn lora_failed_admission_preserves_existing_adapter_kv_and_logits() {
    let cfg = TinyLlamaConfig::default();
    let directory = model_adapter(&cfg);
    let selected = Some(selection(directory.path()));
    let (executor, state) = observed(&cfg);
    let (reference, reference_state) = observed(&cfg);
    let request = RequestId::new();
    let start = admit(&executor, &request, &[0, 1, 2], selected).await;
    let reference_start = admit(&reference, &RequestId::new(), &[0, 1, 2], selected).await;
    let initial = prefill(&executor, &start, 2, 3).await;
    let expected = prefill(&reference, &reference_start, 2, 3).await;
    assert!(matches!(initial.product(), ExecutionProduct::Intermediate));
    assert!(matches!(expected.product(), ExecutionProduct::Intermediate));
    let before = snapshot(&state);
    assert_eq!(before.0.len(), 1);
    assert!(before.0[0].1.iter().all(|length| *length == 2));
    assert_eq!(before.1, 1);
    let missing_path = directory.path().join("missing-adapter");

    assert!(try_admit(
        &executor,
        &RequestId::new(),
        &[2],
        Some(LoraAdapterSelection {
            name: "missing",
            path: missing_path.to_str().unwrap()
        }),
    )
    .await
    .is_err());
    // A new base admission cannot rebind the existing adapter incarnation.
    assert!(matches!(
        try_admit(&executor, &request, &[0, 1, 2], None).await,
        Err(FerrumError::RequestValidation { .. })
    ));
    assert_eq!(snapshot(&state), before);
    assert!(state.lock().releases.is_empty());

    // Complete the original immutable prompt after the failed admission and
    // compare all vocabulary logits to an independently executed model.
    let actual = prefill(&executor, initial.next_frontier(), 1, 3).await;
    let expected = prefill(&reference, expected.next_frontier(), 1, 3).await;
    assert_eq!(logits(&actual), logits(&expected));
    let actual = decode(&executor, actual.next_frontier(), 3).await;
    let expected = decode(&reference, expected.next_frontier(), 3).await;
    assert_eq!(logits(&actual), logits(&expected));
    finish(&executor, actual.next_frontier(), 3, 2).await;
    finish(&reference, expected.next_frontier(), 3, 2).await;
    for observation in [&state, &reference_state] {
        let after = snapshot(observation);
        assert!(after.0.is_empty());
        assert_eq!(after.1, 0);
        let observation = observation.lock();
        assert_eq!(observation.releases.len(), 1);
        assert_eq!(observation.releases[0].1, vec![4; cfg.num_layers]);
    }
}

#[tokio::test]
async fn lora_admission_capacity_failure_preserves_existing_and_discards_new_caches() {
    let cfg = TinyLlamaConfig::default();
    let directory = model_adapter(&cfg);
    let selected = Some(selection(directory.path()));
    let (executor, state) = observed(&cfg);
    let (reference, reference_state) = observed(&cfg);
    let start = admit(&executor, &RequestId::new(), &[0, 1], selected).await;
    let reference_start = admit(&reference, &RequestId::new(), &[0, 1], selected).await;
    let initial = prefill(&executor, &start, 2, 2).await;
    let expected = prefill(&reference, &reference_start, 2, 2).await;
    assert_eq!(logits(&initial), logits(&expected));
    let before = snapshot(&state);
    state.lock().fail_preparation = true;
    for _ in 0..2 {
        assert!(matches!(
            try_admit(&executor, &RequestId::new(), &[2], selected).await,
            Err(FerrumError::ResourceExhausted { .. })
        ));
        assert_eq!(snapshot(&state), before);
    }
    {
        let observation = state.lock();
        assert_eq!(observation.failed_preparations.len(), 2);
        assert_ne!(
            observation.failed_preparations[0].0,
            observation.failed_preparations[1].0
        );
        assert_eq!(observation.releases, observation.failed_preparations);
        for (cache_id, lengths) in &observation.failed_preparations {
            assert_ne!(cache_id, &before.0[0].0);
            assert_eq!(lengths, &vec![0; cfg.num_layers]);
        }
    }
    // This injected preparation failure covers unpublished fresh admissions.
    // No retry permission is inferred for an execution that itself failed.
    state.lock().fail_preparation = false;
    let actual = decode(&executor, initial.next_frontier(), 2).await;
    let expected = decode(&reference, expected.next_frontier(), 2).await;
    assert_eq!(logits(&actual), logits(&expected));
    finish(&executor, actual.next_frontier(), 2, 2).await;
    finish(&reference, expected.next_frontier(), 2, 2).await;
    assert!(snapshot(&state).0.is_empty());
    assert_eq!(snapshot(&state).1, 0);
    assert_eq!(state.lock().releases.len(), 3);
    assert_eq!(
        state.lock().releases.last().unwrap().1,
        vec![3; cfg.num_layers]
    );
    assert!(snapshot(&reference_state).0.is_empty());
    assert_eq!(snapshot(&reference_state).1, 0);
    assert_eq!(reference_state.lock().releases.len(), 1);
    assert_eq!(
        reference_state.lock().releases[0].1,
        vec![3; cfg.num_layers]
    );
}
