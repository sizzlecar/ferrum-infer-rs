//! Real CPU allocator/table shortfalls coupled to real contiguous tiny-model
//! computation. CPU has no paged-attention kernel: these tests qualify typed
//! capacity propagation and numerical continuation, not CPU paged attention.

use super::*;

#[tokio::test]
async fn physical_sequence_slots_inline_preserves_typed_shortfall_without_new_cache() {
    let fixture = fixture(false, false, 8);
    fixture.storage.lock().runtime_env.paged_max_seqs = 1;
    let prompt = [TokenId(1), TokenId(2)];
    let first = admit(&fixture.executor, &RequestId::new(), &prompt).await;
    let waiting = RequestId::new();
    let outcome = fixture
        .executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(&waiting, &prompt, 32),
            adapter: None,
        })
        .await
        .unwrap();
    assert!(matches!(
        outcome,
        TokenAdmissionOutcome::SequenceSlotsDeferred(_)
    ));
    assert_eq!(fixture.storage.lock().kv_caches.len(), 1);
    assert_eq!(fixture.executor.token_sequences.lock().live.len(), 1);
    assert!(
        terminal(fixture.executor.cancel_tokens(first.sequence()).unwrap())
            .await
            .error
            .is_none()
    );
    let replacement = admit(&fixture.executor, &waiting, &prompt).await;
    assert!(terminal(
        fixture
            .executor
            .cancel_tokens(replacement.sequence())
            .unwrap()
    )
    .await
    .error
    .is_none());
}

struct CapacityModel {
    storage: Arc<Mutex<LlamaFamilyModel<CpuBackend>>>,
    numerical: Arc<Mutex<LlamaFamilyModel<CpuBackend>>>,
    control: Arc<Control>,
    forwards: Arc<AtomicU64>,
    config: LlmRuntimeConfig,
    worker: bool,
    cohort: bool,
}

impl CapacityModel {
    fn reservation(&self, inputs: &[ModelTokenInput<'_>]) -> ModelTokenOutcome<()> {
        let requests: Vec<_> = inputs
            .iter()
            .map(|input| KvSlotRequest {
                cache_id: input.cache_id.to_owned(),
                target_len: match input.work {
                    ModelTokenWork::Prefill { tokens, position } => position + tokens.len(),
                    ModelTokenWork::Decode { position, .. } => position as usize + 1,
                },
                admission_target_len: None,
            })
            .collect();
        // The actual production allocator and preparation frame decide fit.
        // No fabricated OOM, diagnostic-only snapshot or injected threshold.
        self.storage
            .lock()
            .prepare_managed_token_capacity(&requests)
    }
}

impl DecoderOnlyLLM for CapacityModel {
    fn retire_model(&mut self) -> crate::common::ModelRetirementOutcome {
        use crate::common::ModelRetirementOutcome;
        match self.storage.lock().retire_model() {
            ModelRetirementOutcome::Retained { error } => {
                ModelRetirementOutcome::Retained { error }
            }
            ModelRetirementOutcome::Retired { error: first } => {
                match self.numerical.lock().retire_model() {
                    ModelRetirementOutcome::Retained { error } => {
                        ModelRetirementOutcome::Retained { error }
                    }
                    ModelRetirementOutcome::Retired { error } => ModelRetirementOutcome::Retired {
                        error: first.or(error),
                    },
                }
            }
        }
    }
    fn config(&self) -> &LlmRuntimeConfig {
        &self.config
    }
    fn token_execution_mode(&self) -> ModelTokenExecutionMode {
        if self.worker {
            ModelTokenExecutionMode::ManagedWorker
        } else {
            ModelTokenExecutionMode::InlineSynchronous
        }
    }
    fn prepare_token_kv(&mut self, request: ModelKvPreparation<'_>) -> Result<ModelKvReservation> {
        let reserved = self.storage.lock().prepare_token_kv(request)?;
        if matches!(reserved, ModelKvReservation::Reserved(_)) {
            assert!(matches!(
                self.numerical.lock().prepare_token_kv(request)?,
                ModelKvReservation::Reserved(_)
            ));
        }
        Ok(reserved)
    }
    fn reserve_token_kv_slots(&mut self, requests: &[KvSlotRequest]) -> Result<ModelKvReservation> {
        self.storage.lock().reserve_token_kv_slots(requests)
    }
    fn execute_token_work(
        &mut self,
        input: &ModelTokenInput<'_>,
    ) -> ModelTokenOutcome<ModelTokenProduct> {
        self.forwards.fetch_add(1, Ordering::Relaxed);
        self.numerical.lock().execute_token_work(input)
    }
    fn execute_token_batch(&mut self, inputs: &[ModelTokenInput<'_>]) -> ModelTokenBatchOutcome {
        if let Some(gate) = self.control.gate.lock().take() {
            gate.entered.send(()).unwrap();
            gate.release.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        self.control.batches.fetch_add(1, Ordering::Relaxed);
        if self.cohort {
            match self.reservation(inputs) {
                ModelTokenOutcome::CapacityDeferred(deferred) => {
                    return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::CapacityDeferred(
                        deferred,
                    ))
                }
                ModelTokenOutcome::NotSubmitted(error)
                | ModelTokenOutcome::Terminal(Err(error)) => {
                    return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::NotSubmitted(error))
                }
                ModelTokenOutcome::Held(error) => {
                    return ModelTokenBatchOutcome::Cohort(ModelTokenOutcome::Held(error))
                }
                ModelTokenOutcome::Terminal(Ok(())) => {}
            }
            self.forwards
                .fetch_add(inputs.len() as u64, Ordering::Relaxed);
            return self.numerical.lock().execute_token_batch(inputs);
        }
        let mut rows = Vec::with_capacity(inputs.len());
        for input in inputs {
            let row = match self.reservation(std::slice::from_ref(input)) {
                ModelTokenOutcome::CapacityDeferred(deferred) => {
                    ModelTokenOutcome::CapacityDeferred(deferred)
                }
                ModelTokenOutcome::NotSubmitted(error)
                | ModelTokenOutcome::Terminal(Err(error)) => ModelTokenOutcome::NotSubmitted(error),
                ModelTokenOutcome::Held(error) => ModelTokenOutcome::Held(error),
                ModelTokenOutcome::Terminal(Ok(())) => self.execute_token_work(input),
            };
            rows.push(row);
        }
        ModelTokenBatchOutcome::Serial(rows)
    }
    fn recover_token_execution(&mut self) -> ModelExecutionRecovery {
        assert!(matches!(
            self.storage.lock().recover_token_execution(),
            ModelExecutionRecovery::Quiescent { error: None }
        ));
        self.numerical.lock().recover_token_execution()
    }
    fn close_token_execution(&mut self, id: &str) -> ModelTokenCloseOutcome {
        assert!(matches!(
            self.storage.lock().close_token_execution(id),
            ModelTokenCloseOutcome::Ended { error: None }
        ));
        self.numerical.lock().close_token_execution(id)
    }
    fn close_token_sequence(&mut self, id: &str) -> Result<()> {
        self.storage.lock().close_token_sequence(id)?;
        self.numerical.lock().close_token_sequence(id)
    }
    fn prefill(&mut self, _: &str, _: &[u32]) -> Vec<f32> {
        panic!("token path required")
    }
    fn decode(&mut self, _: &str, _: u32, _: u32) -> Vec<f32> {
        panic!("token path required")
    }
    fn release(&mut self, _: &str) {
        panic!("checked token close required")
    }
    fn reset(&mut self) {
        panic!("peer storage must remain intact")
    }
}

struct CapacityFixture {
    executor: LlmExecutor,
    storage: Arc<Mutex<LlamaFamilyModel<CpuBackend>>>,
    numerical: Arc<Mutex<LlamaFamilyModel<CpuBackend>>>,
    control: Arc<Control>,
    forwards: Arc<AtomicU64>,
}

fn fixture(worker: bool, cohort: bool, blocks: usize) -> CapacityFixture {
    let mut storage = real_model::<CpuBackend>(true);
    storage.runtime_env.kv_max_blocks = Some(blocks);
    let numerical = real_model::<CpuBackend>(false);
    let config = numerical.config().clone();
    let storage = Arc::new(Mutex::new(storage));
    let numerical = Arc::new(Mutex::new(numerical));
    let control = Arc::new(Control::default());
    let forwards = Arc::new(AtomicU64::new(0));
    let info = tiny_model_info(&TinyLlamaConfig {
        hidden_size: 256,
        intermediate_size: 512,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 128,
        num_layers: 1,
        vocab_size: 128,
        max_seq_len: 32,
    });
    let executor = LlmExecutor::new(
        Box::new(CapacityModel {
            storage: Arc::clone(&storage),
            numerical: Arc::clone(&numerical),
            control: Arc::clone(&control),
            forwards: Arc::clone(&forwards),
            config,
            worker,
            cohort,
        }),
        info,
    );
    CapacityFixture {
        executor,
        storage,
        numerical,
        control,
        forwards,
    }
}

fn free_blocks(fixture: &CapacityFixture) -> usize {
    fixture
        .storage
        .lock()
        .paged_block_alloc
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .free_count()
}

async fn prefill(fixture: &CapacityFixture, count: usize) -> Vec<ExecutionFrontier> {
    let prompt: Vec<_> = (1..=16).map(TokenId).collect();
    let mut frontiers = Vec::new();
    for _ in 0..count {
        frontiers.push(admit(&fixture.executor, &RequestId::new(), &prompt).await);
    }
    let policy = LogitsReturnPolicy::FullLogits;
    let work: Vec<_> = frontiers
        .iter()
        .map(|frontier| {
            ExecutionWork::new(
                frontier,
                TokenWork::Prefill(PrefillChunk::new(0, 16, 16).unwrap()),
                &policy,
            )
            .unwrap()
        })
        .collect();
    let rows = completed(fixture.executor.execute_tokens(&work).await.unwrap());
    let mut oracle = real_model::<CpuBackend>(false);
    let expected = oracle.prefill_internal("reference", &(1..=16).collect::<Vec<_>>());
    for row in &rows {
        compare_logits(full_logits(row), &expected);
    }
    assert_eq!(fixture.forwards.load(Ordering::Relaxed), count as u64);
    rows.into_iter()
        .map(|row| row.next_frontier().clone())
        .collect()
}

fn verify_deferred(cause: &ExecutionInterruptionCause, required: usize, free: usize) {
    let ExecutionInterruptionCause::Deferred(deferral) = cause else {
        panic!("allocator rejection must remain typed: {cause:?}")
    };
    let deferred = deferral.as_allocator_capacity().unwrap();
    assert_eq!(deferred.required_blocks(), required);
    assert_eq!(deferred.snapshot().free_blocks, free);
    assert_eq!(deferred.snapshot().block_size, 16);
}

fn assert_live(fixture: &CapacityFixture, frontier: &ExecutionFrontier, physical: usize) {
    let cache_id = {
        let sequences = fixture.executor.token_sequences.lock();
        let entry = &sequences.live[frontier.sequence()];
        assert_eq!(&entry.frontier, frontier);
        assert!(!entry.closing);
        entry.cache_id.clone()
    };
    assert!(frontier.sequence().observed_end().is_none());
    assert!(fixture.numerical.lock().kv_caches[&cache_id]
        .iter()
        .all(|cache| cache.len == physical));
}

async fn decode_capacity_case(worker: bool, cohort: bool) {
    let fixture = fixture(worker, cohort, 3);
    let frontiers = prefill(&fixture, 2).await;
    assert_eq!(free_blocks(&fixture), 1);
    let policy = LogitsReturnPolicy::FullLogits;
    let work: Vec<_> = frontiers
        .iter()
        .map(|frontier| {
            ExecutionWork::new(
                frontier,
                TokenWork::Decode {
                    token: TokenId(9),
                    position: 16,
                },
                &policy,
            )
            .unwrap()
        })
        .collect();
    let outcome = fixture.executor.execute_tokens(&work).await.unwrap();
    let a = if cohort {
        let ExecutionBatchOutcome::NotSubmitted(not_submitted) = outcome else {
            panic!("whole unsubmitted cohort must retain its fit result")
        };
        let NoSubmissionReason::Deferred(deferral) = not_submitted.into_reason() else {
            panic!("typed deferral required")
        };
        verify_deferred(&ExecutionInterruptionCause::Deferred(deferral), 2, 1);
        assert_eq!(fixture.forwards.load(Ordering::Relaxed), 2);
        for frontier in &frontiers {
            assert_live(&fixture, frontier, 16);
        }
        let rows = completed(fixture.executor.execute_tokens(&work[..1]).await.unwrap());
        rows.into_iter().next().unwrap()
    } else {
        let ExecutionBatchOutcome::Interrupted(interrupted) = outcome else {
            panic!("serial fit must preserve earlier actual output")
        };
        let (mut completed, remaining, causes) = interrupted.into_parts();
        assert_eq!(completed.len(), 1);
        assert_eq!(remaining.len(), 1);
        let (index, sequence, progress, continuation, cause_index) =
            remaining.into_iter().next().unwrap().into_parts();
        assert_eq!(index, 1);
        assert_eq!(&sequence, frontiers[1].sequence());
        assert!(matches!(progress, ExecutionProgress::NotSubmitted { .. }));
        let ContinuationDisposition::Retryable(frontier) = continuation else {
            panic!("real fit shortfall must not close the unchanged owner")
        };
        assert_eq!(frontier, frontiers[1]);
        verify_deferred(&causes[cause_index], 1, 0);
        completed.remove(0)
    };
    assert_eq!(fixture.forwards.load(Ordering::Relaxed), 3);
    assert_live(&fixture, a.next_frontier(), 17);
    assert_live(&fixture, &frontiers[1], 16);
    assert_eq!(free_blocks(&fixture), 0);
    let mut oracle = real_model::<CpuBackend>(false);
    oracle.prefill_internal("reference", &(1..=16).collect::<Vec<_>>());
    let expected = oracle.decode_internal("reference", 9, 16);
    compare_logits(full_logits(&a), &expected);
    assert!(terminal(
        fixture
            .executor
            .cancel_tokens(a.next_frontier().sequence())
            .unwrap()
    )
    .await
    .error
    .is_none());
    assert_eq!(free_blocks(&fixture), 2);
    let rows = completed(fixture.executor.execute_tokens(&work[1..]).await.unwrap());
    compare_logits(full_logits(&rows[0]), &expected);
    assert_eq!(
        fixture.forwards.load(Ordering::Relaxed),
        4,
        "successful A must never replay"
    );
    assert_live(&fixture, rows[0].next_frontier(), 17);
    assert!(terminal(
        fixture
            .executor
            .cancel_tokens(rows[0].next_frontier().sequence())
            .unwrap()
    )
    .await
    .error
    .is_none());
    assert_eq!(free_blocks(&fixture), 3);
    assert!(fixture.storage.lock().kv_caches.is_empty());
    assert!(fixture.numerical.lock().kv_caches.is_empty());
    assert!(fixture.executor.token_sequences.lock().live.is_empty());
}

#[tokio::test]
async fn managed_worker_cpu_capacity_whole_cohort_retains_original_frontiers() {
    decode_capacity_case(true, true).await;
}

#[tokio::test]
async fn managed_worker_cpu_capacity_serial_preserves_completed_peer() {
    decode_capacity_case(true, false).await;
}

#[tokio::test]
async fn managed_worker_cpu_capacity_inline_preserves_completed_peer() {
    decode_capacity_case(false, false).await;
}

#[tokio::test]
async fn managed_worker_cpu_capacity_cancelled_subset_keeps_one_fit_cause() {
    let fixture = fixture(true, true, 4);
    let frontiers = prefill(&fixture, 4).await;
    assert_eq!(free_blocks(&fixture), 0);
    let policy = LogitsReturnPolicy::FullLogits;
    let work: Vec<_> = frontiers
        .iter()
        .map(|frontier| {
            ExecutionWork::new(
                frontier,
                TokenWork::Decode {
                    token: TokenId(9),
                    position: 16,
                },
                &policy,
            )
            .unwrap()
        })
        .collect();
    let (started, release) = fixture.control.block_next();
    let mut blocker = Box::pin(fixture.executor.execute_tokens(&work[..1]));
    poll_pending(blocker.as_mut());
    started.await.unwrap();
    let mut queued = Box::pin(fixture.executor.execute_tokens(&work[1..]));
    poll_pending(queued.as_mut());
    let closing = fixture
        .executor
        .cancel_tokens(frontiers[1].sequence())
        .unwrap();
    release.send(()).unwrap();
    assert!(matches!(
        blocker.await.unwrap(),
        ExecutionBatchOutcome::NotSubmitted(_)
    ));
    let ExecutionBatchOutcome::Interrupted(interrupted) = queued.await.unwrap() else {
        panic!("cancelled member must coexist with the deferred live subgroup")
    };
    let (completed, remaining, causes) = interrupted.into_parts();
    assert!(completed.is_empty());
    assert_eq!(remaining.len(), 3);
    let mut shared_cause = None;
    for participant in remaining {
        let (index, sequence, progress, continuation, cause) = participant.into_parts();
        assert_eq!(&sequence, frontiers[index + 1].sequence());
        if index == 0 {
            match continuation {
                ContinuationDisposition::Ended(ended) => assert_eq!(ended.sequence(), &sequence),
                ContinuationDisposition::Held(held) => {
                    let observed = tokio::time::timeout(Duration::from_secs(5), held.wait())
                        .await
                        .unwrap()
                        .unwrap();
                    assert_eq!(observed.ended.sequence(), &sequence);
                    assert!(observed.error.is_none());
                }
                ContinuationDisposition::Retryable(_) => panic!("cancelled member cannot retry"),
            }
        } else {
            assert!(matches!(progress, ExecutionProgress::NotSubmitted { .. }));
            let ContinuationDisposition::Retryable(frontier) = continuation else {
                panic!("healthy subgroup must remain live")
            };
            assert_eq!(frontier, frontiers[index + 1]);
            assert_live(&fixture, &frontier, 16);
            if let Some(prior) = shared_cause {
                assert_eq!(cause, prior);
            }
            shared_cause = Some(cause);
        }
    }
    let deferred = causes[shared_cause.unwrap()]
        .as_deferral()
        .unwrap()
        .as_allocator_capacity()
        .unwrap();
    assert_eq!(
        deferred.required_blocks(),
        2,
        "fit belongs to the selected two-member cohort"
    );
    assert!(deferred.snapshot().free_blocks <= 1);
    assert_eq!(fixture.forwards.load(Ordering::Relaxed), 4);
    assert!(terminal(closing).await.error.is_none());
    for index in [0, 2, 3] {
        assert!(terminal(
            fixture
                .executor
                .cancel_tokens(frontiers[index].sequence())
                .unwrap()
        )
        .await
        .error
        .is_none());
    }
    assert_eq!(free_blocks(&fixture), 4);
    assert!(fixture.storage.lock().kv_caches.is_empty());
    assert!(fixture.numerical.lock().kv_caches.is_empty());
    assert!(fixture.executor.token_sequences.lock().live.is_empty());
}
