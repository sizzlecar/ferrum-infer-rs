use super::*;
use ferrum_execution::{
    model_executor::{ExecutorSamplingOutput, KvSlotCapacitySnapshot},
    token_execution::*,
};
use ferrum_interfaces::kv_cache::AllocationRequest;
use ferrum_types::{Priority, TokenId};
use parking_lot::Mutex;

struct Entry {
    request: RequestId,
    prompt: Arc<[TokenId]>,
    product_prompt: usize,
    replayed: usize,
    maximum: usize,
    frontier: ExecutionFrontier,
    recovery: ExecutionRecoveryKey,
    // A view of the manager's registered owner, not a separate block allocator.
    cache: Arc<dyn KvCacheHandle>,
}

#[derive(Default)]
pub(super) struct PagedTokenState {
    issuer: ExecutorSequenceIssuer,
    entries: Mutex<HashMap<ExecutorSequence, Entry>>,
}

impl PagedAttentionExecutor {
    pub(super) fn shutdown_token_sequences(&mut self) -> Result<()> {
        // This executor may share its manager with a peer. Release only exact
        // handles in its original registry, never the manager's other entries.
        let sequences = self
            .tokens
            .entries
            .get_mut()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let mut failure = None;
        for sequence in sequences {
            match self.close_token_sequence(&sequence, ExecutionEndReason::Cancelled) {
                Ok(outcome) => {
                    if let Some(error) = outcome.error {
                        failure.get_or_insert(error);
                    }
                    if !matches!(outcome.disposition, ExecutionCloseDisposition::Ended(_)) {
                        failure.get_or_insert_with(|| {
                            FerrumError::backend(
                                "paged host executor retains an unreleased physical cache",
                            )
                        });
                    }
                }
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }

    pub(super) fn admit_token_sequence(
        &self,
        input: ExecutionAdmission<'_>,
    ) -> Result<TokenAdmissionOutcome> {
        input.validate()?;
        if input.adapter.is_some() {
            return Err(FerrumError::unsupported(
                "paged fixture does not implement adapters",
            ));
        }
        let context = input.context;
        if self.config.num_layers == 0
            || self.config.num_kv_heads == 0
            || self.config.head_dim == 0
            || self.config.num_heads == 0
            || self.config.num_heads % self.config.num_kv_heads != 0
            || self.config.vocab_size == 0
            || context.maximum_sequence_tokens > self.config.max_sequence_length
            || context
                .input_tokens
                .iter()
                .any(|token| token.get() as usize >= self.config.vocab_size)
        {
            return Err(FerrumError::request_validation(
                "invalid paged model admission",
            ));
        }
        let mut entries = self.tokens.entries.lock();
        if let Some(entry) = entries
            .values()
            .find(|entry| &entry.request == context.request_id)
        {
            if entry.frontier.consumed_tokens() == 0
                && entry.prompt.as_ref() == context.input_tokens
                && entry.maximum == context.maximum_sequence_tokens
                && entry.product_prompt == context.product_prompt_tokens
                && entry.replayed == context.replayed_output_tokens
            {
                self.kv_manager.with_registered_handle(
                    &entry.request,
                    &entry.cache,
                    |handle| {
                        if handle.num_tokens() != 0 {
                            return Err(FerrumError::request_validation(
                                "paged admission owner changed its frontier",
                            ));
                        }
                        Ok(())
                    },
                )??;
                return Ok(TokenAdmissionOutcome::Admitted(entry.frontier.clone()));
            }
            return Err(FerrumError::request_validation(
                "duplicate live paged request",
            ));
        }
        let pool = self.kv_manager.gpu_pool();
        let stats = pool.stats();
        let required_blocks = context.maximum_sequence_tokens.div_ceil(pool.block_size());
        // The pool creates blocks lazily: uncreated slots are also allocatable.
        let free_blocks = stats.max_blocks - stats.allocated_blocks;
        if required_blocks > stats.max_blocks {
            return Err(FerrumError::resource_exhausted(
                "paged request cannot fit the entire block pool",
            ));
        }
        if required_blocks > free_blocks {
            return Ok(TokenAdmissionOutcome::CapacityDeferred(
                TokenAdmissionCapacityDeferral::new(
                    pool.capacity_domain().clone(),
                    required_blocks,
                    KvSlotCapacitySnapshot {
                        free_blocks,
                        total_blocks: stats.max_blocks,
                        block_size: pool.block_size(),
                    },
                )?,
            ));
        }
        let sequence = self.tokens.issuer.issue_sequence()?;
        let frontier = self.tokens.issuer.frontier(&sequence, 0)?;
        let recovery = self.tokens.issuer.recovery_key(&sequence)?;
        // Reserve the declared extent in the original manager. Future CPU work
        // cannot exhaust the pool halfway through an already admitted sequence.
        let cache = self.kv_manager.allocate_sync(&AllocationRequest {
            request_id: context.request_id.clone(),
            initial_tokens: context.maximum_sequence_tokens,
            max_sequence_length: context.maximum_sequence_tokens,
            num_layers: self.config.num_layers,
            num_heads: self.config.num_kv_heads,
            head_dim: self.config.head_dim,
            device: Device::CPU,
            dtype: DataType::FP32,
            priority: Priority::Normal,
        })?;
        self.kv_manager
            .with_registered_handle(context.request_id, &cache, |handle| {
                handle.set_num_tokens(0);
            })?;
        entries.insert(
            sequence,
            Entry {
                request: context.request_id.clone(),
                prompt: context.input_tokens.into(),
                product_prompt: context.product_prompt_tokens,
                replayed: context.replayed_output_tokens,
                maximum: context.maximum_sequence_tokens,
                frontier: frontier.clone(),
                recovery,
                cache,
            },
        );
        Ok(TokenAdmissionOutcome::Admitted(frontier))
    }

    fn apply_tokens(
        &self,
        entry: &Entry,
        handle: &PagedKvCacheHandle,
        work: TokenWork,
    ) -> Result<ExecutionProduct> {
        let range = work.range()?;
        let mut query = Vec::with_capacity(
            (range.end() - range.start()) * self.config.num_heads * self.config.head_dim,
        );
        for position in range.as_range() {
            let token = match work {
                TokenWork::Prefill(_) => entry.prompt[position],
                TokenWork::Decode { token, .. } => token,
            };
            let embedding = self.token_embedding(token.get());
            for layer in 0..self.config.num_layers {
                self.kv_manager
                    .write_kv(handle, layer, position, &embedding, &embedding)?;
            }
            for kv_head in 0..self.config.num_kv_heads {
                for _ in 0..self.config.num_heads / self.config.num_kv_heads {
                    query.extend_from_slice(
                        &embedding
                            [kv_head * self.config.head_dim..(kv_head + 1) * self.config.head_dim],
                    );
                }
            }
        }
        handle.set_num_tokens(range.end());
        match work {
            TokenWork::Prefill(_) => {
                self.counters.prefill.fetch_add(1, Ordering::Relaxed);
            }
            TokenWork::Decode { .. } => {
                self.counters.decode.fetch_add(1, Ordering::Relaxed);
            }
        }
        if matches!(work, TokenWork::Prefill(chunk) if !chunk.is_final()) {
            return Ok(ExecutionProduct::Intermediate);
        }
        let output = paged_attention(
            &query,
            range.end() - range.start(),
            self.config.num_heads,
            self.config.num_kv_heads,
            self.config.head_dim,
            &self.kv_manager,
            handle,
            self.config.num_layers - 1,
            range.end(),
        )?;
        let width = self.config.num_heads * self.config.head_dim;
        Ok(ExecutionProduct::Sampling(
            ExecutorSamplingOutput::FullLogits(
                self.attention_to_logits(&output[output.len() - width..]),
            ),
        ))
    }

    pub(super) fn execute_token_work(
        &self,
        work: &[ExecutionWork<'_>],
    ) -> Result<ExecutionBatchOutcome> {
        let batch = self.tokens.issuer.batch(work, self.config.vocab_size)?;
        let mut entries = self.tokens.entries.lock();
        for input in work {
            let entry = entries
                .get(input.sequence())
                .ok_or_else(|| FerrumError::request_validation("unknown paged incarnation"))?;
            let registered_length =
                self.kv_manager
                    .with_registered_handle(&entry.request, &entry.cache, |handle| {
                        handle.num_tokens()
                    })?;
            if &entry.frontier != input.expected_frontier()
                || input.work().range()?.end() > entry.maximum
                || registered_length != entry.frontier.consumed_tokens()
            {
                return Err(FerrumError::request_validation("stale paged frontier"));
            }
            match input.work() {
                TokenWork::Prefill(chunk) if chunk.total_prompt_tokens() == entry.prompt.len() => {}
                TokenWork::Decode { position, .. } if position >= entry.prompt.len() => {}
                _ => {
                    return Err(FerrumError::request_validation(
                        "paged work does not match admitted prompt",
                    ))
                }
            }
        }
        let prefill = matches!(work[0].work(), TokenWork::Prefill(_));
        if work
            .iter()
            .any(|input| matches!(input.work(), TokenWork::Prefill(_)) != prefill)
        {
            return Ok(batch.not_submitted(NoSubmissionReason::UnsupportedShape));
        }
        let mut completed = Vec::with_capacity(work.len());
        let mut failed = Vec::new();
        let mut causes = Vec::new();
        for (index, input) in work.iter().enumerate() {
            let entry = entries
                .get_mut(input.sequence())
                .expect("validated paged incarnation");
            // Revalidate under the read guard covering all physical accesses:
            // a replacement after the whole-batch precheck cannot be written.
            let result =
                self.kv_manager
                    .with_registered_handle(&entry.request, &entry.cache, |handle| {
                        if handle.num_tokens() != entry.frontier.consumed_tokens() {
                            return (
                                Err(FerrumError::request_validation(
                                    "paged cache length changed before execution",
                                )),
                                ExecutionProgress::NotSubmitted {
                                    stage: NoSubmissionStage::Validation,
                                },
                            );
                        }
                        (
                            self.apply_tokens(entry, handle, input.work()),
                            ExecutionProgress::Terminal {
                                disposition: TerminalModelDisposition::Failed,
                                executed_range: None,
                            },
                        )
                    });
            let (result, failed_progress) = match result {
                Ok(result) => result,
                Err(error) => (
                    Err(error),
                    ExecutionProgress::NotSubmitted {
                        stage: NoSubmissionStage::Validation,
                    },
                ),
            };
            match result {
                Ok(product) => {
                    let executed = match input.work() {
                        TokenWork::Prefill(chunk) => ExecutedTokenWork::Prefill {
                            chunk,
                            capacity_probe_count: 0,
                        },
                        TokenWork::Decode { position, .. } => {
                            ExecutedTokenWork::Decode { position }
                        }
                    };
                    // The local reference produces exactly one vocabulary row
                    // (or Intermediate) for the already validated exact range.
                    let completion = batch
                        .completion(index, executed, product)
                        .expect("paged reference product shape");
                    entry.frontier = completion.next_frontier().clone();
                    completed.push(completion);
                }
                Err(error) => {
                    let cause_index = causes.len();
                    causes.push(error.into());
                    let continuation = match self
                        .kv_manager
                        .deallocate_handle_sync(&entry.request, &entry.cache)
                    {
                        Ok(()) => {
                            entries.remove(input.sequence());
                            ContinuationDisposition::Ended(
                                self.tokens
                                    .issuer
                                    .ended(input.sequence())
                                    .expect("validated issuer"),
                            )
                        }
                        Err(release_error) => {
                            // No asynchronous device work exists. The manager
                            // retains the unreleased blocks; an observation error
                            // must not fabricate successful resource closure.
                            let owner = Arc::clone(&self.kv_manager);
                            let held = self
                                .tokens
                                .issuer
                                .held(entry.recovery.clone(), async move {
                                    let _owner = owner;
                                    Err(release_error)
                                })
                                .expect("admitted recovery key");
                            ContinuationDisposition::Held(held)
                        }
                    };
                    failed.push(
                        batch
                            .participant(index, failed_progress, continuation, cause_index)
                            .expect("exact failed paged participant"),
                    );
                }
            }
        }
        Ok(if failed.is_empty() {
            batch
                .completed(completed)
                .unwrap_or_else(|error| error.into_outcome())
        } else {
            batch
                .interrupted(completed, failed, causes)
                .unwrap_or_else(|error| error.into_outcome())
        })
    }

    pub(super) fn close_token_sequence(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.tokens.issuer.frontier(sequence, 0)?;
        let mut entries = self.tokens.entries.lock();
        let Some(entry) = entries.get(sequence) else {
            return sequence
                .observed_end()
                .map(ExecutionCloseOutcome::ended)
                .ok_or_else(|| FerrumError::request_validation("unknown paged close incarnation"));
        };
        if let ExecutionEndReason::Completed {
            input_tokens,
            output_tokens,
        } = reason
        {
            if input_tokens != entry.product_prompt
                || output_tokens < entry.replayed
                || input_tokens
                    .checked_add(output_tokens.saturating_sub(1))
                    .is_none_or(|end| end > entry.maximum)
            {
                return Err(FerrumError::request_validation(
                    "paged completion usage mismatch",
                ));
            }
        }
        match self
            .kv_manager
            .deallocate_handle_sync(&entry.request, &entry.cache)
        {
            Ok(()) => {
                entries.remove(sequence);
                Ok(ExecutionCloseOutcome::ended(
                    self.tokens.issuer.ended(sequence)?,
                ))
            }
            Err(error) => {
                let owner = Arc::clone(&self.kv_manager);
                let observed_error = error.clone();
                Ok(ExecutionCloseOutcome::held(self.tokens.issuer.held(
                    entry.recovery.clone(),
                    async move {
                        let _owner = owner;
                        Err(observed_error)
                    },
                )?)
                .with_error(error))
            }
        }
    }
}

impl Drop for PagedAttentionExecutor {
    fn drop(&mut self) {
        // CPU operations are synchronous. The original manager owns all blocks;
        // dropping this fixture returns each still-admitted cache through it.
        for entry in self.tokens.entries.get_mut().values() {
            if let Err(error) = self
                .kv_manager
                .deallocate_handle_sync(&entry.request, &entry.cache)
            {
                tracing::error!(%error, request_id = %entry.request, "paged fixture retained unreleased cache");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_execution::model_executor::{
        ExecutorPrefillAdmission, LogitsReturnPolicy, PrefillChunk,
    };
    use ferrum_interfaces::KvCacheManager;
    use ferrum_kv::managers::paged::PagedKvCacheConfig;

    fn fixture() -> (PagedAttentionExecutor, Arc<PagedKvCacheManager>) {
        let config = PagedExecutorConfig {
            vocab_size: 64,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            max_sequence_length: 8,
        };
        let manager = Arc::new(
            PagedKvCacheManager::new(
                Device::CPU,
                PagedKvCacheConfig {
                    block_size: 2,
                    max_gpu_blocks: 4,
                    max_cpu_blocks: 0,
                    enable_cow: false,
                    enable_swapping: false,
                    num_layers: 1,
                    num_heads: 1,
                    head_dim: 2,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        (
            PagedAttentionExecutor::new(config, manager.clone()),
            manager,
        )
    }

    async fn admit(
        executor: &PagedAttentionExecutor,
        request: &RequestId,
        maximum: usize,
    ) -> TokenAdmissionOutcome {
        let prompt = [TokenId::new(3), TokenId::new(4), TokenId::new(5)];
        executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_diagnostic(request, &prompt, maximum),
                adapter: None,
            })
            .await
            .unwrap()
    }

    async fn run(
        executor: &PagedAttentionExecutor,
        frontier: &ExecutionFrontier,
        work: TokenWork,
    ) -> ExecutionCompletion {
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [ExecutionWork::new(frontier, work, &policy).unwrap()];
        let ExecutionBatchOutcome::Completed(output) =
            executor.execute_tokens(&inputs).await.unwrap()
        else {
            panic!("paged work failed");
        };
        output.into_completions().remove(0)
    }

    fn logits(completion: &ExecutionCompletion) -> &[f32] {
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values)) =
            completion.product()
        else {
            panic!("full logits required");
        };
        values
    }

    fn close(executor: &PagedAttentionExecutor, sequence: &ExecutorSequence) {
        let outcome = executor.cancel_tokens(sequence).unwrap();
        assert!(outcome.error.is_none());
        let ExecutionCloseDisposition::Ended(ended) = outcome.disposition else {
            panic!("physical paged owner must be released");
        };
        assert_eq!(ended.sequence(), sequence);
    }

    // Independent dense f64 attention. It does not call paged_attention,
    // executor embedding/projection helpers, or a second executor path. Inputs
    // use the fixture's specified f32 embeddings; accumulation/softmax are f64.
    fn assert_dense_oracle(actual: &[f32], tokens: &[u32], config: &PagedExecutorConfig) {
        let embeddings: Vec<Vec<f64>> = tokens
            .iter()
            .map(|token| {
                (0..config.num_kv_heads * config.head_dim)
                    .map(|dimension| {
                        (((*token as f32 + 1.0) * (dimension as f32 + 1.0)).sin()) as f64
                    })
                    .collect()
            })
            .collect();
        let query = embeddings.last().unwrap();
        let mut expected = vec![0.0_f64; config.vocab_size];
        for head in 0..config.num_heads {
            let kv_head = head / (config.num_heads / config.num_kv_heads);
            let offset = kv_head * config.head_dim;
            let scores: Vec<f64> = embeddings
                .iter()
                .map(|key| {
                    (0..config.head_dim)
                        .map(|dimension| query[offset + dimension] * key[offset + dimension])
                        .sum::<f64>()
                        / (config.head_dim as f64).sqrt()
                })
                .collect();
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
            let denominator: f64 = weights.iter().sum();
            for dimension in 0..config.head_dim {
                let value: f64 = embeddings
                    .iter()
                    .zip(&weights)
                    .map(|(value, weight)| value[offset + dimension] * weight / denominator)
                    .sum();
                expected[(head * config.head_dim + dimension) % config.vocab_size] += value;
            }
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (f64::from(actual) - expected).abs() <= 2e-6 * (1.0 + expected.abs()),
                "logit {index}: {actual} differs from dense f64 reference {expected}"
            );
        }
    }

    #[tokio::test]
    async fn host_shutdown_paged_releases_only_original_handles_and_preserves_live_peer() {
        use ferrum_execution::executor_lifecycle::{ExecutorOwner, ExecutorShutdownPhase};
        use std::time::Duration;
        let (first, manager) = fixture();
        let peer = PagedAttentionExecutor::new(first.config.clone(), manager.clone());
        let first_observation = first.observer();
        let peer_observation = peer.observer();
        let first_id = RequestId::new();
        let peer_id = RequestId::new();
        let TokenAdmissionOutcome::Admitted(first_frontier) = admit(&first, &first_id, 4).await
        else {
            panic!("first admission");
        };
        let TokenAdmissionOutcome::Admitted(peer_frontier) = admit(&peer, &peer_id, 4).await else {
            panic!("peer admission");
        };
        let first_prefill = run(
            &first,
            &first_frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        assert_dense_oracle(logits(&first_prefill), &[3, 4, 5], &first.config);
        let peer_prefill = run(
            &peer,
            &peer_frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        assert_dense_oracle(logits(&peer_prefill), &[3, 4, 5], &peer.config);
        let peer_handle = manager.get_handle(peer_id.clone()).unwrap();
        assert_eq!(manager.stats().active_caches, 2);
        assert_eq!(manager.gpu_pool().stats().allocated_blocks, 4);

        let owner = ExecutorOwner::new(Box::new(first));
        let report = tokio::time::timeout(Duration::from_secs(5), owner.begin_shutdown().wait())
            .await
            .unwrap();
        assert_eq!(report.phase(), ExecutorShutdownPhase::Retired);
        report.result().unwrap();
        assert!(first_frontier.sequence().observed_end().is_some());
        assert!(peer_frontier.sequence().observed_end().is_none());
        assert!(manager.get_handle(first_id).is_none());
        assert!(Arc::ptr_eq(
            &peer_handle,
            &manager.get_handle(peer_id.clone()).unwrap()
        ));
        assert_eq!(manager.stats().active_caches, 1);
        assert_eq!(manager.gpu_pool().stats().allocated_blocks, 2);
        let peer_decode = run(
            &peer,
            peer_prefill.next_frontier(),
            TokenWork::Decode {
                token: TokenId::new(6),
                position: 3,
            },
        )
        .await;
        assert_dense_oracle(logits(&peer_decode), &[3, 4, 5, 6], &peer.config);
        let peer_owner = ExecutorOwner::new(Box::new(peer));
        let report =
            tokio::time::timeout(Duration::from_secs(5), peer_owner.begin_shutdown().wait())
                .await
                .unwrap();
        assert_eq!(report.phase(), ExecutorShutdownPhase::Retired);
        report.result().unwrap();
        assert!(peer_frontier.sequence().observed_end().is_some());
        assert_eq!(manager.stats().active_caches, 0);
        assert_eq!(manager.gpu_pool().stats().allocated_blocks, 0);
        drop(peer_handle);
        drop(owner);
        drop(peer_owner);
        let original_manager = Arc::downgrade(&manager);
        drop(manager);
        assert!(original_manager.upgrade().is_none());
        assert_eq!(first_observation.prefill_count(), 1);
        assert_eq!(peer_observation.prefill_count(), 1);
        assert_eq!(peer_observation.decode_count(), 1);
    }

    #[tokio::test]
    async fn paged_tokens_external_manager_replacement_rejects_old_access_and_close() {
        let (executor, manager) = fixture();
        let request = RequestId::new();
        let TokenAdmissionOutcome::Admitted(old) = admit(&executor, &request, 5).await else {
            panic!("original admission");
        };
        let old_handle = manager.get_handle(request.clone()).unwrap();
        manager.deallocate_sync(&request).unwrap();
        let replacement = manager
            .allocate_sync(&AllocationRequest {
                request_id: request.clone(),
                initial_tokens: 3,
                max_sequence_length: 3,
                num_layers: 1,
                num_heads: 1,
                head_dim: 2,
                device: Device::CPU,
                dtype: DataType::FP32,
                priority: Priority::Normal,
            })
            .unwrap();
        assert!(!Arc::ptr_eq(&old_handle, &replacement));
        let handle = PagedAttentionExecutor::as_paged_handle(replacement.as_ref()).unwrap();
        for position in 0..3 {
            manager
                .write_kv(handle, 0, position, &[11.0, 13.0], &[17.0, 19.0])
                .unwrap();
        }
        let expected = manager.read_kv(handle, 0, 0, 3).unwrap();
        let allocated = manager.gpu_pool().stats().allocated_blocks;
        let policy = LogitsReturnPolicy::FullLogits;
        let work = [ExecutionWork::new(
            &old,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
            &policy,
        )
        .unwrap()];
        assert!(executor.execute_tokens(&work).await.is_err());
        assert_eq!(
            executor.prefill_count(),
            0,
            "stale work must not enter the physical operation"
        );
        let closed = executor.cancel_tokens(old.sequence()).unwrap();
        assert!(closed.error.is_some());
        let ExecutionCloseDisposition::Held(held) = closed.disposition else {
            panic!("a replacement is not proof that this issuer closed the original owner");
        };
        assert_eq!(held.key().sequence(), old.sequence());
        assert!(held.wait().await.is_err());
        assert!(old.sequence().observed_end().is_none());
        assert!(Arc::ptr_eq(
            &replacement,
            &manager.get_handle(request.clone()).unwrap()
        ));
        assert_eq!(manager.read_kv(handle, 0, 0, 3).unwrap(), expected);
        assert_eq!(manager.gpu_pool().stats().allocated_blocks, allocated);
        assert_eq!(manager.stats().active_caches, 1);
        drop(executor);
        assert!(Arc::ptr_eq(
            &replacement,
            &manager.get_handle(request.clone()).unwrap()
        ));
        assert_eq!(manager.gpu_pool().stats().allocated_blocks, allocated);
        manager
            .deallocate_handle_sync(&request, &replacement)
            .unwrap();
        assert_eq!(manager.stats().active_caches, 0);
    }

    #[tokio::test]
    async fn paged_tokens_preserve_real_chunked_attention_capacity_and_exact_release() {
        let (executor, manager) = fixture();
        let id = RequestId::new();
        let TokenAdmissionOutcome::Admitted(frontier) = admit(&executor, &id, 5).await else {
            panic!("admit");
        };
        assert_eq!(manager.stats().active_caches, 1);
        assert_eq!(manager.gpu_pool().stats().allocated_blocks, 3);
        let peer = RequestId::new();
        let TokenAdmissionOutcome::CapacityDeferred(shortfall) = admit(&executor, &peer, 3).await
        else {
            panic!("real capacity rejection");
        };
        assert_eq!(shortfall.required_blocks(), 2);
        assert_eq!(shortfall.snapshot().free_blocks, 1);
        assert!(manager.get_handle(peer.clone()).is_none());

        let first = run(
            &executor,
            &frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 3).unwrap()),
        )
        .await;
        assert!(matches!(first.product(), ExecutionProduct::Intermediate));
        let last = run(
            &executor,
            first.next_frontier(),
            TokenWork::Prefill(PrefillChunk::new(2, 1, 3).unwrap()),
        )
        .await;
        assert_dense_oracle(logits(&last), &[3, 4, 5], &executor.config);
        let decode = run(
            &executor,
            last.next_frontier(),
            TokenWork::Decode {
                token: TokenId::new(6),
                position: 3,
            },
        )
        .await;
        assert_dense_oracle(logits(&decode), &[3, 4, 5, 6], &executor.config);
        let handle = manager.get_handle(id.clone()).unwrap();
        assert_eq!(handle.num_tokens(), 4);
        let (keys, values) = manager
            .read_kv(
                PagedAttentionExecutor::as_paged_handle(handle.as_ref()).unwrap(),
                0,
                0,
                4,
            )
            .unwrap();
        let expected_kv: Vec<f32> = [3_u32, 4, 5, 6]
            .into_iter()
            .flat_map(|token| {
                (0..2).map(move |dimension| ((token as f32 + 1.0) * (dimension as f32 + 1.0)).sin())
            })
            .collect();
        assert_eq!(keys, expected_kv);
        assert_eq!(values, expected_kv);

        let closed = executor.cancel_tokens(frontier.sequence()).unwrap();
        assert!(closed.error.is_none());
        assert!(matches!(
            closed.disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
        assert_eq!(manager.stats().active_caches, 0);
        assert_eq!(manager.gpu_pool().stats().allocated_blocks, 0);
        let TokenAdmissionOutcome::Admitted(replacement) = admit(&executor, &id, 5).await else {
            panic!("replacement");
        };
        assert_ne!(frontier.sequence(), replacement.sequence());
        close(&executor, frontier.sequence());
        assert_eq!(manager.stats().active_caches, 1);
        assert!(replacement.sequence().observed_end().is_none());
        close(&executor, replacement.sequence());
        let TokenAdmissionOutcome::Admitted(peer) = admit(&executor, &peer, 3).await else {
            panic!("released capacity must admit");
        };
        close(&executor, peer.sequence());
        assert_eq!(manager.gpu_pool().stats().allocated_blocks, 0);
    }
}
