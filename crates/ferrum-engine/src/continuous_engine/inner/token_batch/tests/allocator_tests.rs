//! Actual allocator reference closure through the Engine's recompute registry.
//! Products below are protocol fixtures, not model numerical evidence.

use super::*;
use ferrum_execution::model_executor::{ExecutorSamplingOutput, KvSlotCapacitySnapshot};
use ferrum_execution::token_execution::TokenAllocatorReclaim;
use ferrum_models::common::paged_pool::BlockAllocator;
use ferrum_types::AllocatorCapacityDomain;

pub(super) struct AllocatorState {
    pool: BlockAllocator,
    owners: HashMap<ExecutorSequence, Vec<u32>>,
}

impl AllocatorState {
    fn snapshot(&self) -> KvSlotCapacitySnapshot {
        KvSlotCapacitySnapshot {
            block_size: 16,
            total_blocks: self.pool.capacity() as usize,
            free_blocks: self.pool.free_count(),
        }
    }

    pub(super) fn release(&mut self, sequence: &ExecutorSequence) -> Result<()> {
        let blocks = self.owners.remove(sequence).ok_or_else(|| {
            FerrumError::internal("allocator fixture closed a missing or already released owner")
        })?;
        self.pool.free(&blocks);
        Ok(())
    }

    pub(super) fn observe(
        &self,
        blocked: &ExecutionFrontier,
        domain: &AllocatorCapacityDomain,
        candidates: &[ExecutorSequence],
    ) -> Result<Option<TokenAllocatorReclaim>> {
        if self.pool.capacity_domain() != domain {
            return Ok(None);
        }
        if !self.owners.contains_key(blocked.sequence()) {
            return Err(FerrumError::internal(
                "blocked allocator owner is no longer live",
            ));
        }
        let mut seen = std::collections::HashSet::new();
        let mut mappings = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if candidate == blocked.sequence() || !seen.insert(candidate) {
                return Err(FerrumError::internal(
                    "invalid allocator candidate identity",
                ));
            }
            mappings.push(self.owners.get(candidate).map_or(&[][..], Vec::as_slice));
        }
        let reclaimed = self.pool.observe_reclaimable_blocks(mappings)?;
        Ok(Some(TokenAllocatorReclaim {
            snapshot: self.snapshot(),
            reclaimable_blocks: reclaimed.reclaimable_blocks,
            contributing_candidates: reclaimed.contributing_candidates,
        }))
    }
}

fn rejection(executor: &ProtocolExecutor) -> NoSubmissionReason {
    let state = executor.allocator.lock();
    let state = state.as_ref().unwrap();
    NoSubmissionReason::Deferred(ExecutorExecutionDeferral::AllocatorCapacity(
        TokenAdmissionCapacityDeferral::new(
            state.pool.capacity_domain().clone(),
            1,
            state.snapshot(),
        )
        .unwrap(),
    ))
}

pub(super) async fn commit_initial_products(
    engine: &ContinuousBatchEngine,
    executor: &ProtocolExecutor,
    inputs: &mut [TokenBatchInput],
) {
    let chunk = PrefillChunk::new(0, 2, 2).unwrap();
    for input in inputs.iter_mut() {
        input.work = TokenWork::Prefill(chunk);
        input.scheduled_tokens = 2;
    }
    let works = inputs
        .iter()
        .map(|input| input.as_work().unwrap())
        .collect::<Vec<_>>();
    let scope = executor
        .issuer
        .batch(&works, executor.info().vocab_size)
        .unwrap();
    let completions = (0..inputs.len())
        .map(|index| {
            let mut logits = vec![f32::NEG_INFINITY; executor.info().vocab_size];
            logits[7] = 1.0;
            scope
                .completion(
                    index,
                    ExecutedTokenWork::Prefill {
                        chunk,
                        capacity_probe_count: 1,
                    },
                    ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)),
                )
                .unwrap()
        })
        .collect();
    let mut deferred = Vec::new();
    engine
        .inner
        .consume_token_outcome(
            &inputs.iter().collect::<Vec<_>>(),
            &works,
            scope.completed(completions).unwrap(),
            None,
            &mut deferred,
        )
        .await
        .unwrap();
    assert!(deferred.is_empty());
    drop(works);
    for input in inputs {
        let sequences = &engine.inner.sequences;
        let sequence_handle = sequences.get(&input.request_id).unwrap();
        let mut sequence_guard = sequence_handle.lock();
        let sequence = sequence_guard.as_mut().unwrap();
        assert_eq!(sequence.generated_tokens, vec![TokenId(7)]);
        assert_eq!(sequence.prefill_tokens_processed, 2);
        input.frontier = sequence.execution_frontier.clone().unwrap();
        input.work = TokenWork::Decode {
            token: TokenId(7),
            position: 2,
        };
        input.scheduled_tokens = 1;
        input.context_tokens = 3;
        let mut slot = RequestSlotLease::open(&engine.inner, input.request_id.clone());
        slot.admit(&engine.inner);
        sequence.request_slot = Some(slot);
    }
}

fn assert_waiting_owner(engine: &ContinuousBatchEngine, input: &TokenBatchInput) {
    assert!(input.frontier.sequence().observed_end().is_some());
    let sequences = &engine.inner.sequences;
    let sequence_handle = sequences.get(&input.request_id).unwrap();
    let sequence_guard = sequence_handle.lock();
    let sequence = sequence_guard.as_ref().unwrap();
    assert!(sequence.execution_frontier.is_none());
    assert!(sequence.execution_close.is_none());
    assert_eq!(sequence.phase, RequestPhase::Waiting);
    assert_eq!(sequence.original_request.id, input.request_id);
    assert_eq!(sequence.input_tokens, vec![TokenId(3), TokenId(4)]);
    assert_eq!(sequence.generated_tokens, vec![TokenId(7)]);
    assert_eq!(
        sequence
            .sampling_history
            .token_frequencies()
            .get(&TokenId(7)),
        Some(&1)
    );
    assert_eq!(sequence.prefill_tokens_processed, 0);
    assert!(!sequence.prefill_complete);
    assert!(sequence.request_slot.as_ref().unwrap().armed);
}

fn assert_no_peer_readmission(engine: &ContinuousBatchEngine) {
    let tick = engine
        .inner
        .scheduler
        .prepare_token_admission_observed(
            3,
            None,
            &mut |_| {
                panic!(
                    "yielded peer must not consume capacity before the pressure owner progresses"
                )
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(tick.probed(), 0);
    assert_eq!(tick.admitted(), 0);
    assert!(engine
        .inner
        .scheduler
        .passive_capacity_wait_condition()
        .unwrap()
        .is_none());
}

#[tokio::test]
async fn token_allocator_shared_peers_close_in_order_before_recompute_can_readmit() {
    let (engine, executor, mut inputs) = fixture_with_clock(3, true).await;
    commit_initial_products(&engine, &executor, &mut inputs).await;
    let ingress = inputs
        .iter()
        .map(|input| {
            input
                .handle
                .lock()
                .as_ref()
                .unwrap()
                .original_request
                .ingress
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut pool = BlockAllocator::new(2);
    let blocked_block = pool.allocate().unwrap();
    let shared_block = pool.allocate().unwrap();
    pool.acquire(shared_block);
    let domain = pool.capacity_domain().clone();
    *executor.allocator.lock() = Some(AllocatorState {
        pool,
        owners: HashMap::from([
            (inputs[0].frontier.sequence().clone(), vec![blocked_block]),
            (inputs[1].frontier.sequence().clone(), vec![shared_block]),
            (inputs[2].frontier.sequence().clone(), vec![shared_block]),
        ]),
    });
    let peer_identities = inputs[1..]
        .iter()
        .map(|input| input.frontier.sequence().clone())
        .collect::<Vec<_>>();
    for peer in &peer_identities {
        let solo = executor
            .observe_token_allocator_reclaim(
                &inputs[0].frontier,
                &domain,
                std::slice::from_ref(peer),
            )
            .unwrap()
            .unwrap();
        assert_eq!(solo.reclaimable_blocks, 0);
        assert!(solo.contributing_candidates.is_empty());
    }
    let joint = executor
        .observe_token_allocator_reclaim(&inputs[0].frontier, &domain, &peer_identities)
        .unwrap()
        .unwrap();
    assert_eq!(joint.snapshot.free_blocks, 0);
    assert_eq!(joint.reclaimable_blocks, 1);
    assert_eq!(joint.contributing_candidates, vec![0, 1]);

    // Each peer first encounters the same actual pool while A can still run.
    // This creates genuine scheduler pressure; no direct epoch manipulation.
    let mut root_width = None;
    for peer in &inputs[1..] {
        assert!(engine
            .inner
            .handle_token_deferral(&[peer], rejection(&executor), &mut root_width, 0,)
            .await
            .unwrap()
            .is_empty());
    }
    assert!(executor.closed.lock().is_empty());
    assert_eq!(engine.inner.scheduler.confirmed_capacity_release_epoch(), 0);

    assert!(engine
        .inner
        .handle_token_deferral(&[&inputs[0]], rejection(&executor), &mut root_width, 0,)
        .await
        .unwrap()
        .is_empty());
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    let first_identity = executor.closed.lock()[0].clone();
    let first = inputs[1..]
        .iter()
        .find(|input| input.frontier.sequence() == &first_identity)
        .unwrap();
    let remaining = inputs[1..]
        .iter()
        .find(|input| input.frontier.sequence() != &first_identity)
        .unwrap();
    assert_eq!(executor.closed.lock().len(), 1);
    {
        let state = executor.allocator.lock();
        let state = state.as_ref().unwrap();
        assert_eq!(state.pool.free_count(), 0);
        assert_eq!(state.pool.ref_count(shared_block), 1);
        assert_eq!(state.pool.ref_count(blocked_block), 1);
        assert!(!state.owners.contains_key(&first_identity));
    }
    assert_eq!(engine.inner.scheduler.confirmed_capacity_release_epoch(), 1);
    assert_waiting_owner(&engine, first);
    assert_no_peer_readmission(&engine);
    // Follow the actual next-iteration path before reporting another failed
    // fit. A release notification must not clear the remaining contributor's
    // block and let it compete with this episode's original progress owner.
    let retry_batch = engine
        .inner
        .scheduler
        .next_batch_with_token_admissions(
            ferrum_interfaces::BatchHint::simple(3),
            None,
            &mut |_| {},
        )
        .unwrap()
        .expect("the original progress owner must be allowed to re-probe");
    assert_eq!(
        retry_batch.requests.len(),
        1,
        "a shared reference release must not wake the remaining contributor"
    );
    assert_eq!(retry_batch.requests[0].request.id, inputs[0].request_id);
    assert_eq!(
        engine.inner.scheduler.trace_phase(&remaining.request_id),
        Some(RequestPhase::Decoding)
    );
    {
        let sequences = &engine.inner.sequences;
        let peer_handle = sequences.get(&remaining.request_id).unwrap();
        let peer_guard = peer_handle.lock();
        let peer = peer_guard.as_ref().unwrap();
        assert_eq!(peer.execution_frontier.as_ref(), Some(&remaining.frontier));
        assert_eq!(peer.generated_tokens, vec![TokenId(7)]);
        assert!(remaining.frontier.sequence().observed_end().is_none());
    }
    assert_eq!(
        engine.inner.scheduler.confirmed_capacity_release_epoch(),
        1,
        "selecting a re-probe is not another physical release"
    );
    let remaining_observation = executor
        .observe_token_allocator_reclaim(&inputs[0].frontier, &domain, &peer_identities)
        .unwrap()
        .unwrap();
    assert_eq!(remaining_observation.reclaimable_blocks, 1);
    assert_eq!(
        remaining_observation.contributing_candidates,
        vec![peer_identities
            .iter()
            .position(|identity| identity == remaining.frontier.sequence())
            .unwrap()]
    );

    // The first Ended withdrew one real reference, not a free block. A's next
    // authoritative fit check must select the remaining contributor.
    let before_second_release = rejection(&executor);
    assert!(engine
        .inner
        .handle_token_deferral(&[&inputs[0]], rejection(&executor), &mut root_width, 1,)
        .await
        .unwrap()
        .is_empty());
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(
        executor.closed.lock().as_slice(),
        &[first_identity, remaining.frontier.sequence().clone()]
    );
    assert_eq!(engine.inner.scheduler.confirmed_capacity_release_epoch(), 2);
    assert_waiting_owner(&engine, first);
    assert_waiting_owner(&engine, remaining);
    assert_no_peer_readmission(&engine);
    for (input, original) in inputs[1..].iter().zip(&ingress[1..]) {
        let mut state = input.handle.lock();
        let sequence = state.as_mut().unwrap();
        assert_eq!(sequence.start_time, original.instant());
        assert_eq!(sequence.original_request.ingress, Some(*original));
        assert!(
            !sequence.closes_pending_decode_scheduling(),
            "Held and recompute wait are not decode scheduling"
        );
        let timing = sequence
            .take_execution_evidence()
            .unwrap()
            .unwrap()
            .engine_token_timing
            .unwrap();
        assert_eq!(timing.token_commit_nanos_since_request_start.len(), 1);
        assert!(
            timing.decode_ready_nanos_since_request_start.unwrap()
                <= timing.token_commit_nanos_since_request_start[0]
        );
        assert_eq!(timing.decode_stage_intervals.len(), 1);
        assert_eq!(
            timing.decode_stage_intervals[0].stage,
            EngineDecodeStage::DecodeScheduling
        );
        assert_eq!(
            timing.decode_stage_intervals[0].start_nanos_since_request_start,
            timing.token_commit_nanos_since_request_start[0]
        );
        assert!(
            timing.decode_stage_intervals[0].end_nanos_since_request_start
                <= original.elapsed().as_nanos() as u64
        );
    }
    let runnable = engine
        .inner
        .scheduler
        .next_batch_with_token_admissions(
            ferrum_interfaces::BatchHint::simple(3),
            None,
            &mut |_| {},
        )
        .unwrap()
        .unwrap();
    assert_eq!(runnable.requests.len(), 1);
    assert_eq!(runnable.requests[0].request.id, inputs[0].request_id);
    assert_eq!(
        engine
            .inner
            .handle_token_deferral(&[&inputs[0]], before_second_release, &mut root_width, 2,)
            .await
            .unwrap(),
        vec![vec![0]],
        "a fresh allocator query, not the close return, permits A's retry"
    );
    {
        let mut state = executor.allocator.lock();
        let state = state.as_mut().unwrap();
        assert_eq!(state.pool.free_count(), 1);
        assert_eq!(state.pool.ref_count(shared_block), 0);
        let reserved = state.pool.allocate_n(1).unwrap();
        assert_eq!(reserved, vec![shared_block]);
        state
            .owners
            .get_mut(inputs[0].frontier.sequence())
            .unwrap()
            .extend(reserved);
    }
    assert_eq!(
        engine.inner.scheduler.confirmed_capacity_release_epoch(),
        2,
        "a successful allocation is not a release notification"
    );
    let works = [inputs[0].as_work().unwrap()];
    let scope = executor
        .issuer
        .batch(&works, executor.info().vocab_size)
        .unwrap();
    let mut logits = vec![f32::NEG_INFINITY; executor.info().vocab_size];
    logits[8] = 1.0;
    let completion = scope
        .completion(
            0,
            ExecutedTokenWork::Decode { position: 2 },
            ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)),
        )
        .unwrap();
    let mut deferred = Vec::new();
    engine
        .inner
        .consume_token_outcome(
            &[&inputs[0]],
            &works,
            scope.completed(vec![completion]).unwrap(),
            None,
            &mut deferred,
        )
        .await
        .unwrap();
    assert!(deferred.is_empty());
    {
        let sequences = &engine.inner.sequences;
        let owner_handle = sequences.get(&inputs[0].request_id).unwrap();
        let owner_guard = owner_handle.lock();
        let owner = owner_guard.as_ref().unwrap();
        let frontier = owner.execution_frontier.as_ref().unwrap();
        assert_eq!(frontier.sequence(), inputs[0].frontier.sequence());
        assert_eq!(frontier.consumed_tokens(), 3);
        assert_eq!(owner.generated_tokens, vec![TokenId(7), TokenId(8)]);
    }
    assert_waiting_owner(&engine, first);
    assert_waiting_owner(&engine, remaining);
    assert_eq!(engine.inner.scheduler.confirmed_capacity_release_epoch(), 2);
    // A is still active and owns both blocks. The retired peers are waiting
    // for recompute, so terminate them through real client cancellation, not
    // active-work completion. Neither path may re-admit their old incarnations.
    engine
        .inner
        .complete_request_with_error(
            &inputs[0].request_id,
            FerrumError::request_validation("allocator test cleanup"),
        )
        .await
        .unwrap();
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert_eq!(engine.inner.scheduler.confirmed_capacity_release_epoch(), 3);
    assert_eq!(engine.inner.scheduler.waiting_count(), 2);
    {
        let sequences = &engine.inner.sequences;
        for input in &inputs[1..] {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            drop(receiver);
            let sequence_handle = sequences.get(&input.request_id).unwrap();
            let mut sequence_guard = sequence_handle.lock();
            let sequence = sequence_guard.as_mut().unwrap();
            sequence.response_sender = Some(sender);
            assert!(sequence.client_receiver_closed());
        }
    }
    engine.inner.cancel_abandoned_requests().await.unwrap();
    assert_eq!(engine.inner.scheduler.waiting_count(), 0);
    assert_eq!(
        engine.inner.scheduler.confirmed_capacity_release_epoch(),
        3,
        "cancelling already-retired waiting peers is not a second physical release"
    );
    for input in &inputs {
        assert_eq!(engine.inner.scheduler.trace_phase(&input.request_id), None);
    }
    let state = executor.allocator.lock();
    let state = state.as_ref().unwrap();
    assert!(state.owners.is_empty());
    assert_eq!(state.pool.free_count(), 2);
    assert_eq!(executor.closed.lock().len(), 3);
    assert!(inputs
        .iter()
        .all(|input| input.frontier.sequence().observed_end().is_some()));
    assert!(engine.inner.sequences.is_empty());
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
}

#[tokio::test]
async fn token_allocator_sole_owner_impossible_growth_errors_without_parking() {
    let (engine, executor, inputs) = fixture_with_request_count(1).await;
    let mut pool = BlockAllocator::new(2);
    let blocks = pool.allocate_n(2).unwrap();
    *executor.allocator.lock() = Some(AllocatorState {
        pool,
        owners: HashMap::from([(inputs[0].frontier.sequence().clone(), blocks)]),
    });
    let (sender, receiver) = tokio::sync::oneshot::channel();
    engine
        .inner
        .sequences
        .get(&inputs[0].request_id)
        .unwrap()
        .lock()
        .as_mut()
        .unwrap()
        .response_sender = Some(sender);
    let mut width = None;
    let error = engine
        .inner
        .handle_token_deferral(&[&inputs[0]], rejection(&executor), &mut width, 0)
        .await
        .unwrap_err();
    assert!(matches!(&error, FerrumError::ResourceExhausted { .. }));
    assert!(!engine
        .inner
        .scheduler
        .all_active_allocator_capacity_blocked());
    assert_eq!(
        engine
            .inner
            .scheduler
            .trace_snapshot()
            .pressure_active_episodes,
        0
    );
    assert_eq!(engine.inner.scheduler.confirmed_capacity_release_epoch(), 0);
    assert!(executor.closed.lock().is_empty());
    assert!(inputs[0].frontier.sequence().observed_end().is_none());
    engine
        .inner
        .complete_request_with_error(&inputs[0].request_id, error)
        .await
        .unwrap();
    assert!(matches!(
        receiver.await.unwrap(),
        Err(FerrumError::ResourceExhausted { .. })
    ));
    engine.inner.execution_closes.join_observers().await;
    engine.inner.drain_execution_closes().unwrap();
    assert!(inputs[0].frontier.sequence().observed_end().is_some());
    assert_eq!(executor.closed.lock().len(), 1);
    let state = executor.allocator.lock();
    let state = state.as_ref().unwrap();
    assert_eq!(state.pool.free_count(), 2);
    assert!(state.owners.is_empty());
    assert_eq!(engine.inner.scheduler.confirmed_capacity_release_epoch(), 1);
    assert!(engine.inner.sequences.is_empty());
    assert_eq!(engine.inner.execution_closes.unreclaimed_count(), 0);
}
