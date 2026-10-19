//! Model-owned protocol tests. Real numerical oracles remain in
//! `token_execution::{tests,managed_tests}` and the LoRA integration fixtures.

use super::*;
use crate::common::llm::{
    ModelExecutionRecovery, ModelTokenBatchOutcome, ModelTokenInput, ModelTokenOutcome,
    ModelTokenProduct, ModelTokenWork,
};
use ferrum_execution::model_executor::{
    ExecutorPrefillAdmission, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
};
use ferrum_execution::token_execution::*;
use ferrum_types::{Device, ModelId, ModelType};
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
struct RecordedWork {
    cache_id: String,
    tokens: Vec<u32>,
    position: usize,
    sample: bool,
    full_logits: bool,
}

#[derive(Default)]
struct RecordingCalls {
    prepared: Vec<(String, usize)>,
    live_caches: HashMap<String, usize>,
    released: Vec<String>,
    work: Vec<RecordedWork>,
    cohorts: usize,
    fail_reservation: bool,
}

struct RecordingLlm {
    calls: Arc<Mutex<RecordingCalls>>,
    config: crate::common::LlmRuntimeConfig,
    worker: bool,
}

impl RecordingLlm {
    fn new(calls: Arc<Mutex<RecordingCalls>>) -> Self {
        Self {
            calls,
            config: crate::common::LlmRuntimeConfig {
                hidden_size: 4,
                num_layers: 1,
                num_kv_heads: 1,
                head_dim: 4,
                vocab_size: 4,
                max_seq_len: 16,
            },
            worker: false,
        }
    }
}

impl DecoderOnlyLLM for RecordingLlm {
    fn retire_model(&mut self) -> crate::common::ModelRetirementOutcome {
        // This fixture owns only synchronous host values and call history.
        crate::common::ModelRetirementOutcome::Retired { error: None }
    }
    fn config(&self) -> &crate::common::LlmRuntimeConfig {
        &self.config
    }
    fn token_execution_mode(&self) -> ModelTokenExecutionMode {
        if self.worker {
            ModelTokenExecutionMode::ManagedWorker
        } else {
            ModelTokenExecutionMode::InlineSynchronous
        }
    }
    fn recover_token_execution(&mut self) -> ModelExecutionRecovery {
        ModelExecutionRecovery::Quiescent { error: None }
    }
    fn prepare_kv_capacity(&mut self, cache_id: &str, capacity_hint: usize) {
        let mut calls = self.calls.lock();
        calls.prepared.push((cache_id.to_owned(), capacity_hint));
        assert!(calls.live_caches.insert(cache_id.to_owned(), 0).is_none());
    }
    fn reserve_kv_slots(
        &mut self,
        requests: &[KvSlotRequest],
    ) -> Result<Option<KvSlotReservation>> {
        let mut calls = self.calls.lock();
        for request in requests {
            // Models can create records before reporting a capacity failure.
            calls
                .live_caches
                .entry(request.cache_id.clone())
                .or_insert(0);
        }
        if calls.fail_reservation {
            Err(FerrumError::resource_exhausted(
                "injected KV capacity failure",
            ))
        } else {
            Ok(None)
        }
    }
    fn execute_token_work(
        &mut self,
        input: &ModelTokenInput<'_>,
    ) -> ModelTokenOutcome<ModelTokenProduct> {
        let (tokens, position, logits) = match input.work {
            ModelTokenWork::Prefill { tokens, position } => {
                (tokens.to_vec(), position, vec![0.0, 1.0, 2.0, 3.0])
            }
            ModelTokenWork::Decode { token, position } => {
                (vec![token], position as usize, vec![3.0, 2.0, 1.0, 0.0])
            }
        };
        let mut calls = self.calls.lock();
        let length = calls
            .live_caches
            .get_mut(input.cache_id)
            .expect("admitted owner");
        assert_eq!(*length, position);
        *length += tokens.len();
        calls.work.push(RecordedWork {
            cache_id: input.cache_id.to_owned(),
            tokens,
            position,
            sample: input.sample,
            full_logits: input.logits_policy.requires_full_logits(),
        });
        let output = input.sample.then(|| {
            if input.logits_policy.requires_full_logits() {
                ExecutorSamplingOutput::FullLogits(logits)
            } else {
                let token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0;
                ExecutorSamplingOutput::GreedyToken(TokenId::new(token as u32))
            }
        });
        ModelTokenOutcome::Terminal(Ok(output))
    }
    fn execute_token_batch(&mut self, inputs: &[ModelTokenInput<'_>]) -> ModelTokenBatchOutcome {
        self.calls.lock().cohorts += 1;
        let products = inputs
            .iter()
            .map(|input| {
                let ModelTokenOutcome::Terminal(Ok(product)) = self.execute_token_work(input)
                else {
                    panic!("synchronous recording operation")
                };
                product
            })
            .collect();
        ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Terminal(Ok(products)))
    }
    fn close_token_sequence(&mut self, cache_id: &str) -> Result<()> {
        let mut calls = self.calls.lock();
        assert!(calls.live_caches.remove(cache_id).is_some());
        calls.released.push(cache_id.to_owned());
        Ok(())
    }
    fn prefill(&mut self, _: &str, _: &[u32]) -> Vec<f32> {
        panic!("executor must use direct token work")
    }
    fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
        panic!("executor must use direct token work")
    }
    fn release(&mut self, _: &str) {
        panic!("executor must use checked token closure")
    }
    fn cache_metrics_snapshot(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({ "position": "recording-test-cache" }))
    }
}

fn test_model_info() -> ModelInfo {
    ModelInfo {
        model_id: ModelId("recording".to_owned()),
        model_type: ModelType::Custom("recording".to_owned()),
        num_parameters: 0,
        hidden_size: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 4,
        max_sequence_length: 16,
        dtype: DataType::FP32,
        device: Device::CPU,
        version: None,
        license: None,
        metadata: HashMap::new(),
    }
}

fn recording_executor(calls: Arc<Mutex<RecordingCalls>>) -> LlmExecutor {
    LlmExecutor::new(Box::new(RecordingLlm::new(calls)), test_model_info())
}

async fn admit(executor: &LlmExecutor, prompt: &[TokenId], maximum: usize) -> ExecutionFrontier {
    let request = RequestId::new();
    let TokenAdmissionOutcome::Admitted(frontier) = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_diagnostic(&request, prompt, maximum),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("expected admission")
    };
    frontier
}

async fn close(executor: &LlmExecutor, sequence: &ExecutorSequence) {
    let outcome = executor.cancel_tokens(sequence).unwrap();
    assert!(outcome.error.is_none());
    let ended = match outcome.disposition {
        ExecutionCloseDisposition::Ended(ended) => ended,
        ExecutionCloseDisposition::Held(held) => {
            let terminal = held.wait().await.unwrap();
            assert!(terminal.error.is_none());
            terminal.ended
        }
    };
    assert_eq!(ended.sequence(), sequence);
}

fn sampling(completion: &ExecutionCompletion) -> &ExecutorSamplingOutput {
    let ExecutionProduct::Sampling(output) = completion.product() else {
        panic!("expected sampling product")
    };
    output
}

async fn execute(executor: &LlmExecutor, inputs: &[ExecutionWork<'_>]) -> Vec<ExecutionCompletion> {
    let ExecutionBatchOutcome::Completed(batch) = executor.execute_tokens(inputs).await.unwrap()
    else {
        panic!("expected completed work")
    };
    batch.into_completions()
}

#[tokio::test]
async fn token_preparation_failure_releases_only_new_owner_and_preserves_continuation() {
    let calls = Arc::new(Mutex::new(RecordingCalls::default()));
    let executor = recording_executor(calls.clone());
    let original = admit(&executor, &[TokenId(1), TokenId(2)], 16).await;
    let original_cache = calls.lock().prepared[0].0.clone();
    calls.lock().fail_reservation = true;
    let request = RequestId::new();
    let error = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_diagnostic(&request, &[TokenId(3)], 16),
            adapter: None,
        })
        .await
        .err()
        .expect("capacity failure");
    assert!(matches!(error, FerrumError::ResourceExhausted { .. }));
    {
        let calls = calls.lock();
        assert_eq!(calls.prepared.len(), 2);
        assert_eq!(calls.released, vec![calls.prepared[1].0.clone()]);
        assert_eq!(
            calls.live_caches,
            HashMap::from([(original_cache.clone(), 0)])
        );
        assert!(calls.work.is_empty());
    }
    assert!(original.sequence().observed_end().is_none());
    calls.lock().fail_reservation = false;
    let policy = LogitsReturnPolicy::FullLogits;
    let output = execute(
        &executor,
        &[ExecutionWork::new(
            &original,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
            &policy,
        )
        .unwrap()],
    )
    .await;
    assert_eq!(
        sampling(&output[0]),
        &ExecutorSamplingOutput::FullLogits(vec![0.0, 1.0, 2.0, 3.0])
    );
    assert_eq!(calls.lock().live_caches[&original_cache], 2);
    close(&executor, original.sequence()).await;
    assert!(calls.lock().live_caches.is_empty());
}

#[tokio::test]
async fn token_worker_preserves_mixed_partial_full_and_greedy_products() {
    let calls = Arc::new(Mutex::new(RecordingCalls::default()));
    let executor = LlmExecutor::new(
        Box::new(RecordingLlm {
            worker: true,
            ..RecordingLlm::new(calls.clone())
        }),
        test_model_info(),
    );
    let full = LogitsReturnPolicy::FullLogits;
    let greedy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    };
    let old = admit(&executor, &[TokenId(1), TokenId(2)], 8).await;
    let first = execute(
        &executor,
        &[ExecutionWork::new(
            &old,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
            &full,
        )
        .unwrap()],
    )
    .await;
    assert_eq!(
        sampling(&first[0]),
        &ExecutorSamplingOutput::FullLogits(vec![0.0, 1.0, 2.0, 3.0])
    );
    let fresh = admit(&executor, &[TokenId(3), TokenId(2), TokenId(1)], 7).await;
    let mixed = execute(
        &executor,
        &[
            ExecutionWork::new(
                &fresh,
                TokenWork::Prefill(PrefillChunk::new(0, 1, 3).unwrap()),
                &greedy,
            )
            .unwrap(),
            ExecutionWork::new(
                first[0].next_frontier(),
                TokenWork::Decode {
                    token: TokenId(3),
                    position: 2,
                },
                &full,
            )
            .unwrap(),
        ],
    )
    .await;
    assert!(matches!(mixed[0].product(), ExecutionProduct::Intermediate));
    assert_eq!(mixed[0].next_frontier().consumed_tokens(), 1);
    assert_eq!(mixed[1].next_frontier().consumed_tokens(), 3);
    assert_eq!(
        sampling(&mixed[1]),
        &ExecutorSamplingOutput::FullLogits(vec![3.0, 2.0, 1.0, 0.0])
    );
    let final_batch = execute(
        &executor,
        &[
            ExecutionWork::new(
                mixed[0].next_frontier(),
                TokenWork::Prefill(PrefillChunk::new(1, 2, 3).unwrap()),
                &greedy,
            )
            .unwrap(),
            ExecutionWork::new(
                mixed[1].next_frontier(),
                TokenWork::Decode {
                    token: TokenId(0),
                    position: 3,
                },
                &greedy,
            )
            .unwrap(),
        ],
    )
    .await;
    assert_eq!(
        sampling(&final_batch[0]),
        &ExecutorSamplingOutput::GreedyToken(TokenId(3))
    );
    assert_eq!(
        sampling(&final_batch[1]),
        &ExecutorSamplingOutput::GreedyToken(TokenId(0))
    );
    {
        let calls = calls.lock();
        let old_cache = &calls.prepared[0].0;
        let fresh_cache = &calls.prepared[1].0;
        assert_eq!(
            calls.prepared,
            vec![(old_cache.clone(), 8), (fresh_cache.clone(), 7)]
        );
        assert_eq!(calls.cohorts, 3);
        assert_eq!(
            calls.work,
            vec![
                RecordedWork {
                    cache_id: old_cache.clone(),
                    tokens: vec![1, 2],
                    position: 0,
                    sample: true,
                    full_logits: true
                },
                RecordedWork {
                    cache_id: fresh_cache.clone(),
                    tokens: vec![3],
                    position: 0,
                    sample: false,
                    full_logits: false
                },
                RecordedWork {
                    cache_id: old_cache.clone(),
                    tokens: vec![3],
                    position: 2,
                    sample: true,
                    full_logits: true
                },
                RecordedWork {
                    cache_id: fresh_cache.clone(),
                    tokens: vec![2, 1],
                    position: 1,
                    sample: true,
                    full_logits: false
                },
                RecordedWork {
                    cache_id: old_cache.clone(),
                    tokens: vec![0],
                    position: 3,
                    sample: true,
                    full_logits: false
                },
            ]
        );
    }
    close(&executor, old.sequence()).await;
    close(&executor, fresh.sequence()).await;
    assert!(calls.lock().live_caches.is_empty());
    assert_eq!(calls.lock().released.len(), 2);
}

#[cfg(feature = "test-support")]
#[tokio::test]
async fn lora_prefix_cache_instance_rejects_before_binding_or_prefill() {
    use crate::test_support::{tiny_llama_model, tiny_model_info, TinyLlamaConfig};
    use ferrum_interfaces::adapter::LoraAdapterSelection;

    let cfg = TinyLlamaConfig::default();
    let mut model = tiny_llama_model(&cfg);
    assert_eq!(
        model.adapter_execution_support(),
        AdapterExecutionSupport::PerSequenceLora
    );
    model.runtime_env.prefix_cache = true;
    model.runtime_env.metal_paged_kv = Some(false);
    assert_eq!(
        model.adapter_execution_support(),
        AdapterExecutionSupport::Unsupported
    );
    let error = model
        .set_lora_adapter_for_cache(
            "adapter-cache",
            Some(ActiveLoraAdapter {
                name: "adapter".into(),
                path: "unused-adapter-path".into(),
            }),
        )
        .unwrap_err();
    assert!(matches!(error, FerrumError::Unsupported { .. }));
    assert!(model.kv_caches.is_empty());

    let executor = LlmExecutor::new(Box::new(model), tiny_model_info(&cfg));
    assert_eq!(
        executor.adapter_execution_support(),
        AdapterExecutionSupport::Unsupported
    );
    let request = RequestId::new();
    let prompt = [TokenId(0), TokenId(1)];
    let error = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_diagnostic(&request, &prompt, cfg.max_seq_len),
            adapter: Some(LoraAdapterSelection {
                name: "adapter",
                path: "unused-adapter-path",
            }),
        })
        .await
        .err()
        .expect("unsupported adapter");
    assert!(matches!(error, FerrumError::Unsupported { .. }));
    assert_eq!(executor.next_cache_id.load(Ordering::Relaxed), 0);
    let metrics = executor.lora_metrics_snapshot().unwrap();
    assert_eq!(metrics["active_cache_bindings"].as_u64(), Some(0));
    assert_eq!(metrics["adapter_count"].as_u64(), Some(0));
    assert_eq!(
        executor
            .lock_model()
            .unwrap()
            .cache_metrics_snapshot()
            .unwrap()["enabled"],
        true
    );
    let base = admit(&executor, &prompt, cfg.max_seq_len).await;
    let full = LogitsReturnPolicy::FullLogits;
    let output = execute(
        &executor,
        &[ExecutionWork::new(
            &base,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
            &full,
        )
        .unwrap()],
    )
    .await;
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values)) =
        output[0].product()
    else {
        panic!("base execution must retain full vocabulary logits")
    };
    assert_eq!(values.len(), cfg.vocab_size);
    close(&executor, base.sequence()).await;
}

#[test]
fn llm_executor_reports_retained_model_sources_without_a_runtime_plan() {
    let calls = Arc::new(Mutex::new(RecordingCalls::default()));
    let unbound = recording_executor(calls.clone());
    assert!(unbound.bound_resolved_sources().is_none());

    // This fixture exercises the source lease, not weight loading.
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("config.json"), b"{}").unwrap();
    std::fs::write(root.path().join("tokenizer.json"), b"{}").unwrap();
    std::fs::write(root.path().join("model.safetensors"), b"fixture-weights").unwrap();
    let sources =
        Arc::new(ProductionModelSourceBundle::open_colocated_safetensors(root.path()).unwrap());
    let retained = Arc::downgrade(&sources);
    let executor = LlmExecutor::new_with_model_sources(
        Box::new(RecordingLlm::new(calls)),
        test_model_info(),
        sources,
    );
    let retained = retained
        .upgrade()
        .expect("executor retains its source bundle");
    assert!(std::ptr::eq(
        executor.bound_resolved_sources().unwrap(),
        retained.resolved_sources(),
    ));
    assert!(executor.resolved_model_plan().is_none());
    assert_eq!(
        executor.execution_resource_authority(),
        ferrum_types::ExecutionResourceAuthority::LegacyEngine,
    );
}
#[test]
fn cache_metrics_snapshot_includes_model_lock_wait_metrics() {
    let calls = Arc::new(Mutex::new(RecordingCalls::default()));
    let executor = recording_executor(calls);

    assert_eq!(executor.kv_capacity(), Some(16));
    // Capacity is frozen metadata; acquire the original model through a real
    // snapshot operation before checking accumulated lock measurements.
    assert!(executor.cache_metrics_snapshot().is_some());
    let metrics = executor.cache_metrics_snapshot().unwrap();

    assert_eq!(metrics["position"], "recording-test-cache");
    assert_eq!(metrics["executor_model_lock"]["schema_version"], 1);
    assert!(
        metrics["executor_model_lock"]["samples"].as_u64().unwrap() >= 2,
        "metrics: {metrics}"
    );
    assert!(
        metrics["executor_model_lock"]["total_wait_time_us"]
            .as_u64()
            .is_some(),
        "metrics: {metrics}"
    );
    assert!(
        metrics["executor_model_lock"]["avg_wait_time_ms"].is_number(),
        "metrics: {metrics}"
    );
}
