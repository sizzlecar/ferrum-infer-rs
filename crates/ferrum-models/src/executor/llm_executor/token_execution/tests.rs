use super::*;
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
        self.state
            .lock()
            .model
            .prepare_kv_capacity(cache_id, capacity_hint);
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
        let before = state.model.cache_len(cache_id);
        let result = state.model.execute_token_work(cache_id, work);
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
                    Fault::InvalidOutput => ModelTokenOutcome::Terminal(Ok(vec![0.0])),
                };
            }
        }
        result
    }
    fn close_token_sequence(&mut self, cache_id: &str) -> Result<()> {
        let mut state = self.state.lock();
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

fn admit(executor: &LlmExecutor, request: &RequestId, tokens: &[TokenId]) -> ExecutionFrontier {
    let TokenAdmissionOutcome::Admitted(frontier) = executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(request, tokens, 16),
            adapter: None,
        })
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
    let start = admit(&executor, &request, &prompt);
    assert!(executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&request, &prompt, 16),
            adapter: None
        })
        .is_err());
    assert!(observation.lock().model.kv_caches.is_empty());
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

async fn partial_failure(fault: Fault) {
    let (executor, observation, config) = fixture(Some((2, fault)));
    let mut reference = model(&config);
    let prompts = [
        [TokenId(1), TokenId(2)],
        [TokenId(3), TokenId(4)],
        [TokenId(5), TokenId(6)],
    ];
    let frontiers = prompts
        .iter()
        .map(|prompt| admit(&executor, &RequestId::new(), prompt))
        .collect::<Vec<_>>();
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
    let start = admit(&executor, &RequestId::new(), &[TokenId(1), TokenId(2)]);
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
    let peer = admit(&executor, &RequestId::new(), &[TokenId(5), TokenId(6)]);
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
