use super::*;
use crate::common::llm::ModelTokenBatchOutcome;
use crate::common::LlmRuntimeConfig;
use crate::models::LlamaFamilyModel;
use crate::test_support::{tiny_llama_model, tiny_model_info, TinyLlamaConfig};
use ferrum_execution::model_executor::{ExecutorPrefillAdmission, PrefillChunk};
use ferrum_kernels::backend::cpu::CpuBackend;

fn model(config: &TinyLlamaConfig) -> LlamaFamilyModel<CpuBackend> {
    let mut model = tiny_llama_model(config);
    model.runtime_env.metal_paged_kv = Some(false);
    model.runtime_env.prefix_cache = false;
    model.runtime_env.kv_capacity = Some(config.max_seq_len);
    model
}

#[derive(Clone, Copy)]
enum Fault {
    ModelFailure,
    InvalidOutput,
}

struct Observation {
    model: LlamaFamilyModel<CpuBackend>,
    forwards: Vec<(String, usize, usize)>,
    releases: Vec<(String, Vec<usize>)>,
    fail_on_forward: Option<(usize, Fault)>,
    fail_prepare_after_allocation: bool,
    fail_close: bool,
}

struct ObservedModel {
    state: Arc<Mutex<Observation>>,
    config: LlmRuntimeConfig,
}

impl DecoderOnlyLLM for ObservedModel {
    fn config(&self) -> &LlmRuntimeConfig {
        &self.config
    }
    fn supports_direct_token_execution(&self) -> bool {
        true
    }
    fn prepare_kv_capacity(&mut self, cache_id: &str, capacity_hint: usize) {
        let mut state = self.state.lock();
        state.model.prepare_kv_capacity(cache_id, capacity_hint);
        assert!(
            !state.fail_prepare_after_allocation,
            "injected preparation panic after actual KV allocation"
        );
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
    fn execute_token_work(
        &mut self,
        input: &ModelTokenInput<'_>,
    ) -> ModelTokenOutcome<ModelTokenProduct> {
        let mut state = self.state.lock();
        let cache_id = input.cache_id;
        let before = state.model.cache_len(cache_id);
        let result = state.model.execute_token_work(input);
        let after = state.model.cache_len(cache_id);
        state.forwards.push((cache_id.to_owned(), before, after));
        if let Some((call, fault)) = state.fail_on_forward {
            if call == state.forwards.len() {
                assert!(matches!(result, ModelTokenOutcome::Terminal(Ok(_))));
                // Fault injection is after a real complete forward. Actual KV
                // and full model execution are observed, not inferred from a
                // mock call counter or fabricated terminal receipt.
                return match fault {
                    Fault::ModelFailure => ModelTokenOutcome::Terminal(Err(FerrumError::backend(
                        "injected post-forward model failure",
                    ))),
                    Fault::InvalidOutput => ModelTokenOutcome::Terminal(Ok(Some(
                        ExecutorSamplingOutput::FullLogits(vec![0.0]),
                    ))),
                };
            }
        }
        result
    }
    fn close_token_sequence(&mut self, cache_id: &str) -> Result<()> {
        let mut state = self.state.lock();
        if state.fail_close {
            return Err(FerrumError::device("injected cleanup observation failure"));
        }
        let lengths = state
            .model
            .kv_caches
            .get(cache_id)
            .map(|layers| layers.iter().map(|cache| cache.len).collect())
            .unwrap_or_default();
        state.model.close_token_sequence(cache_id)?;
        state.releases.push((cache_id.to_owned(), lengths));
        Ok(())
    }
    fn prefill(&mut self, _: &str, _: &[u32]) -> Vec<f32> {
        panic!("token execution must not call the old model adapter")
    }
    fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
        panic!("token execution must not call the old model adapter")
    }
    fn release(&mut self, _: &str) {
        panic!("token closure must use the checked model boundary")
    }
    fn reset(&mut self) {
        panic!("one participant must never reset its peers")
    }
}

fn fixture(
    fault: Option<(usize, Fault)>,
) -> (LlmExecutor, Arc<Mutex<Observation>>, TinyLlamaConfig) {
    let config = TinyLlamaConfig::default();
    let real = model(&config);
    let runtime_config = real.config().clone();
    let observation = Arc::new(Mutex::new(Observation {
        model: real,
        forwards: Vec::new(),
        releases: Vec::new(),
        fail_on_forward: fault,
        fail_prepare_after_allocation: false,
        fail_close: false,
    }));
    let executor = LlmExecutor::new(
        Box::new(ObservedModel {
            state: observation.clone(),
            config: runtime_config,
        }),
        tiny_model_info(&config),
    );
    (executor, observation, config)
}

async fn admit(
    executor: &LlmExecutor,
    request: &RequestId,
    tokens: &[TokenId],
) -> ExecutionFrontier {
    let TokenAdmissionOutcome::Admitted(frontier) = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(request, tokens, 16),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("expected admission")
    };
    frontier
}

fn logits(completion: &ExecutionCompletion) -> &[f32] {
    let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) =
        completion.product()
    else {
        panic!("expected complete vocabulary logits")
    };
    logits
}

async fn prefill(
    executor: &LlmExecutor,
    frontier: &ExecutionFrontier,
    count: usize,
    total: usize,
) -> ExecutionCompletion {
    let policy = LogitsReturnPolicy::FullLogits;
    let input = ExecutionWork::new(
        frontier,
        TokenWork::Prefill(PrefillChunk::new(frontier.consumed_tokens(), count, total).unwrap()),
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::Completed(batch) = executor.execute_tokens(&[input]).await.unwrap()
    else {
        panic!("expected terminal completion")
    };
    batch.into_completions().pop().unwrap()
}

#[tokio::test]
async fn direct_tokens_real_llama_partial_final_decode_and_close() {
    let (executor, observation, config) = fixture(None);
    let mut reference = model(&config);
    let request = RequestId::new();
    let prompt = [TokenId(1), TokenId(2), TokenId(3)];
    let start = admit(&executor, &request, &prompt).await;
    assert!(executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&request, &prompt, 16),
            adapter: None
        })
        .await
        .is_err());
    {
        let state = observation.lock();
        assert_eq!(state.model.kv_caches.len(), 1);
        assert!(state
            .model
            .kv_caches
            .values()
            .flatten()
            .all(|cache| cache.len == 0 && cache.capacity >= 16));
        assert!(state.forwards.is_empty());
    }
    let partial = prefill(&executor, &start, 2, 3).await;
    assert!(matches!(partial.product(), ExecutionProduct::Intermediate));
    reference.prefill("reference", &[1, 2]);
    let final_ = prefill(&executor, partial.next_frontier(), 1, 3).await;
    assert_eq!(logits(&final_), reference.prefill("reference", &[3]));
    assert!(matches!(
        final_.executed(),
        ExecutedTokenWork::Prefill { .. }
    ));
    let policy = LogitsReturnPolicy::FullLogits;
    let work = ExecutionWork::new(
        final_.next_frontier(),
        TokenWork::Decode {
            token: TokenId(4),
            position: 3,
        },
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::Completed(batch) = executor.execute_tokens(&[work]).await.unwrap()
    else {
        panic!("expected decode")
    };
    assert_eq!(
        logits(&batch.completions()[0]),
        reference.decode("reference", 4, 3)
    );
    let stale = ExecutionWork::new(
        &start,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 3).unwrap()),
        &policy,
    )
    .unwrap();
    assert!(executor.execute_tokens(&[stale]).await.is_err());
    let other = ExecutorSequenceIssuer::new();
    let foreign = other.frontier(&other.issue_sequence().unwrap(), 0).unwrap();
    let work = ExecutionWork::new(
        &foreign,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 3).unwrap()),
        &policy,
    )
    .unwrap();
    assert!(executor.execute_tokens(&[work]).await.is_err());
    assert_eq!(observation.lock().forwards.len(), 3);
    let close = executor
        .finish_tokens(
            start.sequence(),
            ExecutionEndReason::Completed {
                input_tokens: 3,
                output_tokens: 2,
            },
        )
        .await
        .unwrap();
    assert!(close.error.is_none());
    assert!(matches!(
        close.disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    let state = observation.lock();
    assert!(state.model.kv_caches.is_empty());
    assert_eq!(state.releases.len(), 1);
    assert_eq!(state.releases[0].1, vec![4; config.num_layers]);
    assert!(executor.token_sequences.lock().live.is_empty());
}

#[test]
fn direct_tokens_real_llama_default_batch_preserves_optional_products() {
    let (executor, observation, config) = fixture(None);
    let mut reference = model(&config);
    reference.prefill("reference", &[1, 2]);
    let expected = reference.prefill("reference", &[3]);
    let policy = LogitsReturnPolicy::FullLogits;
    let prompt: Arc<[u32]> = Arc::from([1, 2, 3]);
    let mut actual = executor.lock_model();
    assert!(matches!(
        actual
            .prepare_token_kv(ModelKvPreparation {
                cache_id: "model-default-batch",
                capacity_hint: 16,
                prompt: &prompt,
                adapter: None,
            })
            .unwrap(),
        ModelKvReservation::Reserved(_)
    ));
    // ObservedModel inherits the real trait's serial fallback. Each scalar
    // call reaches tiny Llama, including the intermediate no-product chunk.
    let ModelTokenBatchOutcome::Serial(rows) = actual.execute_token_batch(&[
        ModelTokenInput {
            cache_id: "model-default-batch",
            work: ModelTokenWork::Prefill {
                tokens: &[1, 2],
                position: 0,
            },
            logits_policy: &policy,
            sample: false,
        },
        ModelTokenInput {
            cache_id: "model-default-batch",
            work: ModelTokenWork::Prefill {
                tokens: &[3],
                position: 2,
            },
            logits_policy: &policy,
            sample: true,
        },
    ]) else {
        panic!("default batch must retain the serial outcomes");
    };
    assert_eq!(rows.len(), 2);
    assert!(matches!(&rows[0], ModelTokenOutcome::Terminal(Ok(None))));
    assert!(matches!(
        &rows[1],
        ModelTokenOutcome::Terminal(Ok(Some(ExecutorSamplingOutput::FullLogits(logits))))
            if logits == &expected
    ));
    actual.close_token_sequence("model-default-batch").unwrap();
    drop(actual);
    let state = observation.lock();
    assert_eq!(state.forwards.len(), 2);
    assert_eq!(state.forwards[0].1, 0);
    assert_eq!(state.forwards[0].2, 2);
    assert_eq!(state.forwards[1].1, 2);
    assert_eq!(state.forwards[1].2, 3);
    assert_eq!(
        state.releases,
        [("model-default-batch".to_owned(), vec![3; config.num_layers])]
    );
    assert!(state.model.kv_caches.is_empty());
}

#[tokio::test]
async fn direct_tokens_real_llama_partial_product_is_not_silently_discarded() {
    let (executor, observation, config) = fixture(Some((1, Fault::InvalidOutput)));
    let frontier = admit(
        &executor,
        &RequestId::new(),
        &[TokenId(1), TokenId(2), TokenId(3)],
    )
    .await;
    let policy = LogitsReturnPolicy::FullLogits;
    let work = ExecutionWork::new(
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 2, 3).unwrap()),
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::ContractViolation(violation) =
        executor.execute_tokens(&[work]).await.unwrap()
    else {
        panic!("an unexpected sampling product must not become Intermediate");
    };
    assert!(violation.completed.is_empty());
    assert_eq!(violation.rejected_completions.len(), 1);
    let (index, executed, product) = &violation.rejected_completions[0];
    assert_eq!(*index, 0);
    assert_eq!(executed.range().unwrap().as_range(), 0..2);
    assert!(
        matches!(product, ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values)) if values == &[0.0])
    );
    assert_eq!(violation.remaining.len(), 1);
    assert!(
        matches!(violation.remaining[0].continuation(), ContinuationDisposition::Ended(ended) if ended.sequence() == frontier.sequence())
    );
    let state = observation.lock();
    assert_eq!(state.forwards.len(), 1);
    assert_eq!((state.forwards[0].1, state.forwards[0].2), (0, 2));
    assert_eq!(state.releases.len(), 1);
    assert_eq!(state.releases[0].1, vec![2; config.num_layers]);
    assert!(state.model.kv_caches.is_empty());
    assert!(executor.token_sequences.lock().live.is_empty());
}

#[tokio::test]
async fn direct_tokens_real_llama_completion_accepts_last_output_at_physical_capacity() {
    for output_tokens in [1, 3] {
        let (executor, observation, config) = fixture(None);
        let mut reference = model(&config);
        let request = RequestId::new();
        let prompt = [TokenId(1), TokenId(2), TokenId(3)];
        let capacity = prompt.len() + output_tokens - 1;
        let TokenAdmissionOutcome::Admitted(start) = executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    &request,
                    &prompt,
                    capacity,
                    prompt.len(),
                    0,
                )
                .unwrap(),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("exact-capacity CPU admission must succeed");
        };
        let cache_id = executor.token_sequences.lock().live[start.sequence()]
            .cache_id
            .clone();
        let mut result = prefill(&executor, &start, prompt.len(), prompt.len()).await;
        assert_eq!(logits(&result), reference.prefill("reference", &[1, 2, 3]));
        for generated in 1..output_tokens {
            let token = TokenId((generated + 3) as u32);
            let position = prompt.len() + generated - 1;
            let policy = LogitsReturnPolicy::FullLogits;
            let work = ExecutionWork::new(
                result.next_frontier(),
                TokenWork::Decode { token, position },
                &policy,
            )
            .unwrap();
            let ExecutionBatchOutcome::Completed(batch) =
                executor.execute_tokens(&[work]).await.unwrap()
            else {
                panic!("CPU decode must complete");
            };
            result = batch.into_completions().pop().unwrap();
            assert_eq!(
                logits(&result),
                reference.decode("reference", token.get(), position as u32)
            );
        }
        assert_eq!(result.next_frontier().consumed_tokens(), capacity);
        {
            let state = observation.lock();
            assert_eq!(state.forwards.len(), output_tokens);
            assert!(state.model.kv_caches[&cache_id]
                .iter()
                .all(|layer| layer.len == capacity && layer.capacity == capacity));
            assert!(state.releases.is_empty());
        }
        let closed = executor
            .finish_tokens(
                start.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: prompt.len(),
                    output_tokens,
                },
            )
            .await
            .unwrap();
        assert!(closed.error.is_none());
        assert!(
            matches!(closed.disposition, ExecutionCloseDisposition::Ended(ref ended)
            if ended.sequence() == start.sequence())
        );
        assert!(executor.token_sequences.lock().live.is_empty());
        let state = observation.lock();
        assert!(state.model.kv_caches.is_empty());
        assert_eq!(
            state.releases,
            vec![(cache_id, vec![capacity; config.num_layers])]
        );
    }
}

#[tokio::test]
async fn direct_tokens_real_llama_completion_rejects_excess_and_overflow_without_closing_owner() {
    let (executor, observation, config) = fixture(None);
    let mut reference = model(&config);
    let request = RequestId::new();
    let prompt = [TokenId(1), TokenId(2), TokenId(3)];
    let capacity = 4;
    let TokenAdmissionOutcome::Admitted(start) = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_product_request(
                &request,
                &prompt,
                capacity,
                prompt.len(),
                0,
            )
            .unwrap(),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("CPU admission must succeed");
    };
    let cache_id = executor.token_sequences.lock().live[start.sequence()]
        .cache_id
        .clone();
    let first = prefill(&executor, &start, prompt.len(), prompt.len()).await;
    assert_eq!(logits(&first), reference.prefill("reference", &[1, 2, 3]));
    for (input_tokens, output_tokens) in [(3, 3), (3, usize::MAX), (2, 1)] {
        let error = executor
            .finish_tokens(
                start.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens,
                    output_tokens,
                },
            )
            .await
            .unwrap_err();
        assert!(
            matches!(error, FerrumError::RequestValidation { ref message }
            if message == "token completion usage differs from immutable admission")
        );
        assert!(start.sequence().observed_end().is_none());
        {
            let entries = executor.token_sequences.lock();
            let entry = &entries.live[start.sequence()];
            assert!(!entry.closing);
            assert_eq!(&entry.frontier, first.next_frontier());
        }
        let state = observation.lock();
        assert_eq!(state.forwards.len(), 1);
        assert!(state.releases.is_empty());
        assert!(state.model.kv_caches[&cache_id]
            .iter()
            .all(|layer| layer.len == 3));
    }
    let policy = LogitsReturnPolicy::FullLogits;
    let work = ExecutionWork::new(
        first.next_frontier(),
        TokenWork::Decode {
            token: TokenId(4),
            position: 3,
        },
        &policy,
    )
    .unwrap();
    let ExecutionBatchOutcome::Completed(batch) = executor.execute_tokens(&[work]).await.unwrap()
    else {
        panic!("the unchanged owner must remain executable");
    };
    assert_eq!(
        logits(&batch.completions()[0]),
        reference.decode("reference", 4, 3)
    );
    assert_eq!(
        batch.completions()[0].next_frontier().consumed_tokens(),
        capacity
    );
    let closed = executor
        .finish_tokens(
            start.sequence(),
            ExecutionEndReason::Completed {
                input_tokens: 3,
                output_tokens: 2,
            },
        )
        .await
        .unwrap();
    assert!(closed.error.is_none());
    assert!(
        matches!(closed.disposition, ExecutionCloseDisposition::Ended(ref ended)
        if ended.sequence() == start.sequence())
    );
    assert!(executor.token_sequences.lock().live.is_empty());
    let state = observation.lock();
    assert!(state.model.kv_caches.is_empty());
    assert_eq!(state.forwards.len(), 2);
    assert_eq!(
        state.releases,
        vec![(cache_id, vec![capacity; config.num_layers])]
    );
}

#[tokio::test]
async fn direct_tokens_real_llama_completion_preserves_zero_output_and_replay_accounting() {
    // Product usage is not an equality assertion about the physical frontier:
    // recompute can already contain output tokens, and a caller can close after
    // prefill without publishing a sampled output. Match the native contract.
    for replayed in [0, 2] {
        let (executor, observation, config) = fixture(None);
        let request = RequestId::new();
        let prompt = vec![TokenId(1); 2 + replayed];
        let TokenAdmissionOutcome::Admitted(start) = executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    &request,
                    &prompt,
                    prompt.len(),
                    2,
                    replayed,
                )
                .unwrap(),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("CPU replay admission must succeed");
        };
        let result = prefill(&executor, &start, prompt.len(), prompt.len()).await;
        assert_eq!(result.next_frontier().consumed_tokens(), prompt.len());
        if replayed > 0 {
            assert!(executor
                .finish_tokens(
                    start.sequence(),
                    ExecutionEndReason::Completed {
                        input_tokens: 2,
                        output_tokens: replayed - 1,
                    }
                )
                .await
                .is_err());
            assert!(start.sequence().observed_end().is_none());
            assert!(!executor.token_sequences.lock().live[start.sequence()].closing);
            assert!(observation.lock().releases.is_empty());
        }
        let closed = executor
            .finish_tokens(
                start.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 2,
                    output_tokens: replayed,
                },
            )
            .await
            .unwrap();
        assert!(closed.error.is_none());
        assert!(matches!(
            closed.disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
        assert!(executor.token_sequences.lock().live.is_empty());
        let state = observation.lock();
        assert!(state.model.kv_caches.is_empty());
        assert_eq!(state.releases.len(), 1);
        assert_eq!(state.releases[0].1, vec![prompt.len(); config.num_layers]);
    }
}

#[tokio::test]
async fn direct_tokens_real_llama_admission_reserves_actual_blocks_and_retries_after_release() {
    let (executor, observation, _) = fixture(None);
    {
        let mut state = observation.lock();
        state.model.runtime_env.metal_paged_kv = Some(true);
        state.model.runtime_env.kv_max_blocks = Some(3);
    }
    let prompt = vec![TokenId(1); 17];
    let first_request = RequestId::new();
    let second_request = RequestId::new();
    fn admission<'a>(request: &'a RequestId, prompt: &'a [TokenId]) -> ExecutionAdmission<'a> {
        ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(request, prompt, 64),
            adapter: None,
        }
    }
    let TokenAdmissionOutcome::Admitted(first) = executor
        .admit_tokens(admission(&first_request, &prompt))
        .await
        .unwrap()
    else {
        panic!("first prompt should reserve two physical blocks");
    };
    let TokenAdmissionOutcome::CapacityDeferred(deferred) = executor
        .admit_tokens(admission(&second_request, &prompt))
        .await
        .unwrap()
    else {
        panic!("second prompt must observe the occupied physical pool");
    };
    assert_eq!(deferred.required_blocks(), 2);
    assert_eq!(deferred.snapshot().block_size, 16);
    assert_eq!(deferred.snapshot().total_blocks, 3);
    assert_eq!(deferred.snapshot().free_blocks, 1);
    {
        let state = observation.lock();
        assert!(state.forwards.is_empty());
        assert_eq!(state.model.kv_caches.len(), 1);
        for cache in state.model.kv_caches.values().flatten() {
            assert_eq!(cache.len, 0);
            assert_eq!(cache.paged_block_indices.len(), 2);
        }
        assert_eq!(
            state.model.kv_slot_capacity_snapshot().unwrap().free_blocks,
            1
        );
    }
    assert_eq!(executor.token_sequences.lock().live.len(), 1);
    let closed = executor.cancel_tokens(first.sequence()).unwrap();
    assert!(closed.error.is_none());
    assert!(matches!(
        closed.disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    assert_eq!(
        observation
            .lock()
            .model
            .kv_slot_capacity_snapshot()
            .unwrap()
            .free_blocks,
        3
    );
    let TokenAdmissionOutcome::Admitted(second) = executor
        .admit_tokens(admission(&second_request, &prompt))
        .await
        .unwrap()
    else {
        panic!("actual release must permit the unchanged second request");
    };
    assert!(matches!(
        executor
            .cancel_tokens(second.sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    let impossible = vec![TokenId(1); 49];
    assert!(executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&RequestId::new(), &impossible, 64),
            adapter: None,
        })
        .await
        .is_err());
    assert!(executor.token_sequences.lock().live.is_empty());
    let state = observation.lock();
    assert!(state.model.kv_caches.is_empty());
    assert!(state.forwards.is_empty());
    assert_eq!(
        state.model.kv_slot_capacity_snapshot().unwrap().free_blocks,
        3
    );
}

#[tokio::test]
async fn direct_tokens_real_llama_admission_preserves_primary_failure_and_original_owner() {
    let (executor, observation, _) = fixture(None);
    {
        let mut state = observation.lock();
        state.fail_prepare_after_allocation = true;
        state.fail_close = true;
    }
    let prompt = [TokenId(1), TokenId(2)];
    let request = RequestId::new();
    let error = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&request, &prompt, 16),
            adapter: None,
        })
        .await
        .unwrap_err();
    assert!(matches!(error, FerrumError::Backend { .. }));
    assert!(error
        .to_string()
        .contains("injected preparation panic after actual KV allocation"));
    let sequence = {
        let sequences = executor.token_sequences.lock();
        assert_eq!(sequences.live.len(), 1);
        let (sequence, entry) = sequences.live.iter().next().unwrap();
        assert!(entry.closing);
        let state = observation.lock();
        assert_eq!(state.model.kv_caches.len(), 1);
        assert!(state.model.kv_caches[&entry.cache_id]
            .iter()
            .all(|cache| cache.len == 0));
        assert!(state.forwards.is_empty());
        assert!(state.releases.is_empty());
        sequence.clone()
    };
    observation.lock().fail_close = false;
    let closed = executor.cancel_tokens(&sequence).unwrap();
    assert!(closed.error.is_none());
    assert!(matches!(
        closed.disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    assert!(executor.token_sequences.lock().live.is_empty());
    let state = observation.lock();
    assert!(state.model.kv_caches.is_empty());
    assert_eq!(state.releases.len(), 1);
}

async fn partial_failure(fault: Fault) {
    let (executor, observation, config) = fixture(Some((2, fault)));
    let mut reference = model(&config);
    let prompts = [
        [TokenId(1), TokenId(2)],
        [TokenId(3), TokenId(4)],
        [TokenId(5), TokenId(6)],
    ];
    let mut frontiers = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        frontiers.push(admit(&executor, &RequestId::new(), prompt).await);
    }
    let cache_ids = frontiers
        .iter()
        .map(|frontier| {
            executor.token_sequences.lock().live[frontier.sequence()]
                .cache_id
                .clone()
        })
        .collect::<Vec<_>>();
    let policy = LogitsReturnPolicy::FullLogits;
    let work = frontiers
        .iter()
        .map(|frontier| {
            ExecutionWork::new(
                frontier,
                TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
                &policy,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let outcome = executor.execute_tokens(&work).await.unwrap();
    let (completed, remaining) = match (fault, outcome) {
        (Fault::ModelFailure, ExecutionBatchOutcome::Interrupted(batch)) => {
            let (completed, remaining, _) = batch.into_parts();
            (completed, remaining)
        }
        (Fault::InvalidOutput, ExecutionBatchOutcome::ContractViolation(violation)) => {
            assert_eq!(violation.rejected_completions.len(), 1);
            let (index, executed, product) = &violation.rejected_completions[0];
            assert_eq!(*index, 1);
            assert_eq!(executed.range().unwrap().as_range(), 0..2);
            assert!(
                matches!(product, ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values)) if values == &[0.0])
            );
            (violation.completed, violation.remaining)
        }
        _ => panic!("expected retained partial progress and the original invalid product"),
    };
    assert_eq!(
        completed
            .iter()
            .map(ExecutionCompletion::input_index)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(logits(&completed[0]), reference.prefill("a", &[1, 2]));
    assert_eq!(logits(&completed[1]), reference.prefill("peer", &[5, 6]));
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].input_index(), 1);
    assert!(matches!(
        remaining[0].continuation(),
        ContinuationDisposition::Ended(_)
    ));
    assert!(matches!(
        remaining[0].progress(),
        ExecutionProgress::Terminal { .. }
    ));
    {
        let state = observation.lock();
        assert_eq!(state.forwards.len(), 3);
        assert!(state
            .forwards
            .iter()
            .all(|(_, before, after)| (*before, *after) == (0, 2)));
        assert_eq!(
            state.releases,
            [(cache_ids[1].clone(), vec![2; config.num_layers])]
        );
        assert!(!state.model.kv_caches.contains_key(&cache_ids[1]));
        assert_eq!(state.model.cache_len(&cache_ids[0]), 2);
        assert_eq!(state.model.cache_len(&cache_ids[2]), 2);
    }
    let work = completed
        .iter()
        .map(|completion| {
            ExecutionWork::new(
                completion.next_frontier(),
                TokenWork::Decode {
                    token: TokenId(7),
                    position: 2,
                },
                &policy,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let ExecutionBatchOutcome::Completed(next) = executor.execute_tokens(&work).await.unwrap()
    else {
        panic!("successful peers must continue")
    };
    assert_eq!(logits(&next.completions()[0]), reference.decode("a", 7, 2));
    assert_eq!(
        logits(&next.completions()[1]),
        reference.decode("peer", 7, 2)
    );
    for index in [0, 2] {
        let close = executor.cancel_tokens(frontiers[index].sequence()).unwrap();
        assert!(close.error.is_none());
        assert!(matches!(
            close.disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
    }
    assert!(executor
        .execute_tokens(&[ExecutionWork::new(
            &frontiers[1],
            TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
            &policy
        )
        .unwrap()])
        .await
        .is_err());
    let state = observation.lock();
    assert_eq!(state.forwards.len(), 5);
    assert_eq!(state.releases.len(), 3);
    for cache in &cache_ids {
        assert_eq!(
            state.releases.iter().filter(|(id, _)| id == cache).count(),
            1
        );
    }
    assert!(state.model.kv_caches.is_empty());
    assert!(executor.token_sequences.lock().live.is_empty());
}

#[tokio::test]
async fn direct_tokens_real_llama_later_model_failure_does_not_replay_success() {
    partial_failure(Fault::ModelFailure).await;
}

#[tokio::test]
async fn direct_tokens_real_llama_later_invalid_output_preserves_terminal_progress() {
    partial_failure(Fault::InvalidOutput).await;
}

#[tokio::test]
async fn direct_tokens_real_llama_inconsistent_layer_frontier_is_not_retryable() {
    let (executor, observation, config) = fixture(None);
    let mut reference = model(&config);
    let start = admit(&executor, &RequestId::new(), &[TokenId(1), TokenId(2)]).await;
    let first = prefill(&executor, &start, 2, 2).await;
    let cache_id = executor.token_sequences.lock().live[start.sequence()]
        .cache_id
        .clone();
    // Corrupt one real layer's logical frontier, without replacing its backing
    // or mocking the model check. This call must not retry the stale frontier.
    observation
        .lock()
        .model
        .kv_caches
        .get_mut(&cache_id)
        .unwrap()[0]
        .len = 3;
    let peer = admit(&executor, &RequestId::new(), &[TokenId(5), TokenId(6)]).await;
    let policy = LogitsReturnPolicy::FullLogits;
    let work = [
        ExecutionWork::new(
            first.next_frontier(),
            TokenWork::Decode {
                token: TokenId(3),
                position: 2,
            },
            &policy,
        )
        .unwrap(),
        ExecutionWork::new(
            &peer,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
            &policy,
        )
        .unwrap(),
    ];
    let ExecutionBatchOutcome::Interrupted(batch) = executor.execute_tokens(&work).await.unwrap()
    else {
        panic!("expected explicit unusable participant")
    };
    assert_eq!(batch.remaining().len(), 1);
    assert!(matches!(
        batch.remaining()[0].progress(),
        ExecutionProgress::NotSubmitted { .. }
    ));
    assert!(matches!(
        batch.remaining()[0].continuation(),
        ContinuationDisposition::Ended(_)
    ));
    assert_eq!(batch.completed().len(), 1);
    assert_eq!(batch.completed()[0].input_index(), 1);
    assert_eq!(
        logits(&batch.completed()[0]),
        reference.prefill("peer", &[5, 6])
    );
    {
        let state = observation.lock();
        assert_eq!(state.forwards[1], (cache_id.clone(), 3, 3));
        assert_eq!(state.releases, [(cache_id.clone(), vec![3, 2])]);
        assert!(!state.model.kv_caches.contains_key(&cache_id));
    }
    assert!(executor.execute_tokens(&work[..1]).await.is_err());
    assert!(matches!(
        executor.cancel_tokens(peer.sequence()).unwrap().disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    assert!(observation.lock().model.kv_caches.is_empty());
}
