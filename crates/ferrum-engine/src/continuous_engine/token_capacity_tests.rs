//! Controlled token-protocol capacity tests. The real scheduler queues,
//! pressure coordinator and Engine close registry execute; no native allocator,
//! model forward or GPU numerical qualification is claimed.
use super::*;
use ferrum_execution::token_execution::{
    ExecutedTokenWork, ExecutionAdmission, ExecutionBatchOutcome, ExecutionCloseOutcome,
    ExecutionEndReason, ExecutionFrontier, ExecutionProduct, ExecutionWork, ExecutorSequence,
    ExecutorSequenceIssuer, NoSubmissionReason, TokenAdmissionOutcome, TokenWork,
};

struct TokenCapacityExecutor {
    source: PlanRuntimeBatchDecodeTestExecutor,
    issuer: ExecutorSequenceIssuer,
    live: Arc<Mutex<HashMap<ExecutorSequence, ExecutionFrontier>>>,
}

struct TokenCapacityObservation {
    source: Arc<PlanRuntimeBatchDecodeObservation>,
    live: std::sync::Weak<Mutex<HashMap<ExecutorSequence, ExecutionFrontier>>>,
}

impl TokenCapacityObservation {
    fn contains(&self, sequence: &ExecutorSequence) -> bool {
        self.live
            .upgrade()
            .is_some_and(|live| live.lock().contains_key(sequence))
    }
}

impl ObservedTestExecutor for TokenCapacityExecutor {
    type Observation = TokenCapacityObservation;

    fn observer(&self) -> Arc<Self::Observation> {
        Arc::new(TokenCapacityObservation {
            source: self.source.observer(),
            live: Arc::downgrade(&self.live),
        })
    }

    fn retire_fixture(&mut self) -> Result<()> {
        let sequences = self.live.lock().keys().cloned().collect::<Vec<_>>();
        for sequence in sequences {
            let outcome = self.close(&sequence)?;
            if let Some(error) = outcome.error {
                return Err(error);
            }
            if sequence.observed_end().is_none() {
                return Err(FerrumError::internal(
                    "capacity fixture retained an unproved incarnation",
                ));
            }
        }
        self.source.retire_fixture()
    }
}

impl TokenCapacityExecutor {
    fn close(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.issuer.frontier(sequence, 0)?;
        if self.live.lock().remove(sequence).is_some() {
            // The existing fixture's availability changes only after removing
            // the exact retained incarnation, never from logical cancellation.
            self.source
                .released_cache_count
                .fetch_add(1, Ordering::AcqRel);
            return Ok(ExecutionCloseOutcome::ended(self.issuer.ended(sequence)?));
        }
        sequence
            .observed_end()
            .map(ExecutionCloseOutcome::ended)
            .ok_or_else(|| FerrumError::internal("unknown capacity-test incarnation"))
    }
}

#[async_trait::async_trait]
impl ModelExecutor for TokenCapacityExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::executor_lifecycle::ExecutorShutdownPermit,
    ) -> ferrum_execution::executor_lifecycle::ExecutorShutdownObserver {
        begin_controlled_shutdown(self, permit)
    }
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.source.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.source.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.source.status()
    }
    fn execution_planning_model(&self) -> ExecutionPlanningModel {
        ExecutionPlanningModel::CompiledPlan
    }
    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        self.source.plan_runtime_resource_snapshot()
    }
    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<ferrum_runtime::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        self.source.write_execution_capacity_snapshot(availability)
    }
    fn write_token_capacity_release_sources(
        &self,
        sequence: &ExecutorSequence,
        sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        self.issuer.frontier(sequence, 0)?;
        sources.clear();
        if !self.live.lock().contains_key(sequence) {
            return Ok(false);
        }
        sources.push(ferrum_runtime::CapacityAvailabilitySource::ActiveSequenceSlots);
        Ok(true)
    }
    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        input.validate()?;
        let sequence = self.issuer.issue_sequence()?;
        let frontier = self.issuer.frontier(&sequence, 0)?;
        self.live.lock().insert(sequence, frontier.clone());
        Ok(TokenAdmissionOutcome::Admitted(frontier))
    }
    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        let scope = self.issuer.batch(work, self.info().vocab_size)?;
        let mut live = self.live.lock();
        for input in work {
            if live.get(input.sequence()) != Some(input.expected_frontier()) {
                return Err(FerrumError::internal(
                    "capacity fixture received stale token work",
                ));
            }
        }
        let decode_only = work
            .iter()
            .all(|input| matches!(input.work(), TokenWork::Decode { .. }));
        if decode_only {
            let call = self
                .source
                .batch_decode_calls
                .fetch_add(1, Ordering::Relaxed);
            if matches!(
                self.source.behavior,
                PlanRuntimeBatchDecodeBehavior::DeferUntilPeerCacheRelease
            ) && self.source.released_cache_count.load(Ordering::Acquire) == 0
            {
                let mut availability = Vec::new();
                let observed = self
                    .source
                    .write_execution_capacity_snapshot(&mut availability)?
                    .expect("the existing controlled native fixture has a coordinator");
                let condition = ferrum_runtime::CapacityWaitCondition::from_observation(
                    observed.coordinator_id.get(),
                    availability,
                )
                .unwrap();
                return Ok(scope.not_submitted(NoSubmissionReason::Deferred(
                    test_execution_capacity_deferral(
                        observed,
                        condition,
                        self.source.capacity_stage_for_call(call),
                    )
                    .into(),
                )));
            }
        }
        let mut completions = Vec::with_capacity(work.len());
        for (index, input) in work.iter().enumerate() {
            let executed = match input.work() {
                TokenWork::Prefill(chunk) => ExecutedTokenWork::Prefill {
                    chunk,
                    capacity_probe_count: 1,
                },
                TokenWork::Decode { position, .. } => ExecutedTokenWork::Decode { position },
            };
            let product = match input.work() {
                TokenWork::Prefill(chunk) if !chunk.is_final() => ExecutionProduct::Intermediate,
                _ => {
                    let output = if input.logits_policy().requires_full_logits() {
                        let mut logits = vec![0.0; self.info().vocab_size];
                        logits[6] = 1.0;
                        ExecutorSamplingOutput::FullLogits(logits)
                    } else {
                        ExecutorSamplingOutput::GreedyToken(TokenId(6))
                    };
                    ExecutionProduct::Sampling(output)
                }
            };
            completions.push(scope.completion(index, executed, product).unwrap());
        }
        let outcome = scope.completed(completions).unwrap();
        let ExecutionBatchOutcome::Completed(completed) = &outcome else {
            unreachable!()
        };
        for completion in completed.completions() {
            live.insert(
                completion.next_frontier().sequence().clone(),
                completion.next_frontier().clone(),
            );
        }
        Ok(outcome)
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        _reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.close(sequence)
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.close(sequence)
    }
}

fn fixture(
    behavior: PlanRuntimeBatchDecodeBehavior,
    stage: ExecutorExecutionCapacityStage,
    next_stage: Option<ExecutorExecutionCapacityStage>,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<TokenCapacityObservation>,
) {
    let source = PlanRuntimeBatchDecodeTestExecutor::with_capacity_stage_transition(
        behavior, stage, next_stage,
    );
    let executor = TokenCapacityExecutor {
        source,
        issuer: ExecutorSequenceIssuer::new(),
        live: Arc::new(Mutex::new(HashMap::new())),
    };
    let (engine, scheduler, executor, _) =
        plan_runtime_test_engine(plan_runtime_test_config(None), executor);
    (engine, scheduler, executor)
}

async fn install_decode_cohort(
    engine: &ContinuousBatchEngine,
    scheduler: &ContinuousBatchScheduler,
    executor: &TokenCapacityObservation,
) -> Vec<RequestId> {
    let mut ids = Vec::new();
    for _ in 0..8 {
        let mut request = policy_request();
        request.sampling_params.max_tokens = 32;
        request
            .metadata
            .insert(PROMPT_TOKENS_METADATA_KEY.into(), serde_json::json!(1));
        let tokens = vec![TokenId(5)];
        let admitted = engine
            .active_inner()
            .expect("engine execution is open")
            .model_executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    &request.id,
                    &tokens,
                    64,
                    tokens.len(),
                    0,
                )
                .unwrap(),
                adapter: None,
            })
            .await
            .unwrap();
        let TokenAdmissionOutcome::Admitted(frontier) = admitted else {
            panic!("controlled admission")
        };
        let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            tokens,
            Some(
                engine
                    .active_inner()
                    .expect("engine execution is open")
                    .tokenizer
                    .clone(),
            ),
            Some(64),
        );
        state.execution_frontier = Some(frontier);
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .insert(request.id.clone(), state);
        scheduler.submit(request.clone()).await.unwrap();
        ids.push(request.id);
    }
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(ids.len()))
        .await
        .unwrap();
    assert_eq!(batch.requests.len(), ids.len());
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .unwrap();
    for id in &ids {
        let sequences = &engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        let state_handle = sequences.get(id).unwrap();
        let state_guard = state_handle.lock();
        let state = state_guard.as_ref().unwrap();
        assert!(state.prefill_complete);
        assert_eq!(
            state.execution_frontier.as_ref().unwrap().consumed_tokens(),
            1
        );
        assert_eq!(state.generated_tokens, [TokenId(6)]);
    }
    assert_eq!(
        executor.source.batch_decode_calls.load(Ordering::Acquire),
        0
    );
    ids
}

async fn run_decode_width(engine: &ContinuousBatchEngine, width: usize) {
    let batch = engine
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(width))
        .await
        .expect("live decode cohort");
    assert_eq!(batch.requests.len(), width);
    assert!(batch.requests.iter().all(|request| {
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .get(&request.request.id)
            .unwrap()
            .lock()
            .as_ref()
            .unwrap()
            .prefill_complete
    }));
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .unwrap();
}

async fn close_auxiliary_incarnation(
    engine: &ContinuousBatchEngine,
    executor: &TokenCapacityObservation,
) {
    let request = policy_request();
    let tokens = [TokenId(5)];
    let admitted = engine
        .active_inner()
        .expect("engine execution is open")
        .model_executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_product_request(&request.id, &tokens, 64, 1, 0)
                .unwrap(),
            adapter: None,
        })
        .await
        .unwrap();
    let TokenAdmissionOutcome::Admitted(frontier) = admitted else {
        unreachable!()
    };
    engine.scheduler.submit(request.clone()).await.unwrap();
    assert!(engine.scheduler.cancel_request(request.id.clone()).unwrap());
    let before = engine.scheduler.trace_snapshot().capacity_release_epoch;
    let ticket = engine
        .active_inner()
        .expect("engine execution is open")
        .begin_execution_close(
            request.id,
            frontier.sequence().clone(),
            None,
            ExecutionEndReason::Cancelled,
        )
        .unwrap();
    ticket.initial_result().await.unwrap();
    engine
        .active_inner()
        .expect("engine execution is open")
        .execution_closes
        .join_observers()
        .await;
    engine
        .active_inner()
        .expect("engine execution is open")
        .drain_execution_closes()
        .unwrap();
    assert!(frontier.sequence().observed_end().is_some());
    assert!(!executor.contains(frontier.sequence()));
    assert_eq!(
        engine.scheduler.trace_snapshot().capacity_release_epoch,
        before + 1
    );
}

#[tokio::test]
async fn token_capacity_execution_pressure_preserves_root_width_across_recursive_split_and_yield() {
    for stage in [
        ExecutorExecutionCapacityStage::StepAdmission,
        ExecutorExecutionCapacityStage::SubmissionWave,
    ] {
        let (engine, scheduler, executor) = fixture(
            PlanRuntimeBatchDecodeBehavior::DeferUntilPeerCacheRelease,
            stage,
            None,
        );
        let ids = install_decode_cohort(&engine, &scheduler, &executor).await;
        run_decode_width(&engine, ids.len()).await;
        let pressure = scheduler.trace_snapshot();
        assert_eq!(
            executor.source.released_cache_count.load(Ordering::Acquire),
            1
        );
        assert_eq!(pressure.decode_capacity_backpressure_admit_limit, Some(4));
        assert!(pressure.decode_execution_pressure_enforced);
        assert_eq!(pressure.waiting_queue_len, 1);
        assert_eq!(pressure.decode_queue_len, ids.len() - 1);
        assert_eq!(pressure.capacity_release_epoch, 1);
    }
}

#[tokio::test]
async fn token_capacity_sequence_extension_pressure_does_not_cap_survivors_during_recompute() {
    let (engine, scheduler, executor) = fixture(
        PlanRuntimeBatchDecodeBehavior::DeferUntilPeerCacheRelease,
        ExecutorExecutionCapacityStage::SequenceExtension,
        None,
    );
    let ids = install_decode_cohort(&engine, &scheduler, &executor).await;
    run_decode_width(&engine, ids.len()).await;
    let pressure = scheduler.trace_snapshot();
    assert_eq!(
        executor.source.released_cache_count.load(Ordering::Acquire),
        1
    );
    assert_eq!(pressure.decode_capacity_backpressure_admit_limit, Some(4));
    assert!(!pressure.decode_execution_pressure_enforced);
    assert_eq!(pressure.waiting_queue_len, 1);
    assert_eq!(pressure.decode_queue_len, ids.len() - 1);
    let mut availability = Vec::new();
    let epochs = engine
        .active_inner()
        .expect("engine execution is open")
        .model_executor
        .write_execution_capacity_snapshot(&mut availability)
        .unwrap()
        .unwrap();
    let wake = AdmissionWakeSnapshot::new(
        AdmissionWakeEpochs::new(
            epochs.coordinator_id,
            epochs.release_epoch,
            epochs.capacity_epoch,
            0,
        ),
        &availability,
    );
    let survivors = scheduler
        .next_batch_with_token_admissions(
            scheduler
                .prepare_token_batch_selection(ferrum_interfaces::BatchHint::simple(ids.len())),
            Some(wake),
            &mut |_| {},
            None,
        )
        .unwrap()
        .batch
        .expect("already admitted survivors remain schedulable");
    assert_eq!(survivors.requests.len(), ids.len() - 1);
    assert!(survivors.requests.iter().all(|scheduled| {
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .get(&scheduled.request.id)
            .unwrap()
            .lock()
            .as_ref()
            .unwrap()
            .prefill_complete
    }));
}

#[tokio::test]
async fn token_capacity_recursive_cohort_pressure_upgrades_sequence_extension_feedback() {
    let (engine, scheduler, executor) = fixture(
        PlanRuntimeBatchDecodeBehavior::DeferUntilPeerCacheRelease,
        ExecutorExecutionCapacityStage::SequenceExtension,
        Some(ExecutorExecutionCapacityStage::StepAdmission),
    );
    let ids = install_decode_cohort(&engine, &scheduler, &executor).await;
    run_decode_width(&engine, ids.len()).await;
    let pressure = scheduler.trace_snapshot();
    assert_eq!(
        executor.source.released_cache_count.load(Ordering::Acquire),
        1
    );
    assert_eq!(pressure.decode_capacity_backpressure_admit_limit, Some(4));
    assert!(pressure.decode_execution_pressure_enforced);
    assert_eq!(pressure.waiting_queue_len, 1);
    assert_eq!(pressure.decode_queue_len, ids.len() - 1);
}

#[tokio::test]
async fn token_capacity_saturated_root_success_recovers_execution_pressure_additively() {
    let (engine, scheduler, executor) = fixture(
        PlanRuntimeBatchDecodeBehavior::Exact,
        ExecutorExecutionCapacityStage::StepAdmission,
        None,
    );
    install_decode_cohort(&engine, &scheduler, &executor).await;
    scheduler.record_decode_execution_capacity_pressure(8);
    assert_eq!(
        scheduler
            .trace_snapshot()
            .decode_capacity_backpressure_admit_limit,
        Some(4)
    );

    // Even an actual full-width successful call cannot grow the learned bound
    // before any retained execution incarnation has been released.
    run_decode_width(&engine, 4).await;
    assert_eq!(
        scheduler
            .trace_snapshot()
            .decode_capacity_backpressure_admit_limit,
        Some(4)
    );
    assert!(
        scheduler
            .trace_snapshot()
            .decode_execution_pressure_enforced
    );

    close_auxiliary_incarnation(&engine, &executor).await;
    run_decode_width(&engine, 4).await;
    assert_eq!(
        scheduler
            .trace_snapshot()
            .decode_capacity_backpressure_admit_limit,
        Some(5)
    );
    assert!(
        scheduler
            .trace_snapshot()
            .decode_execution_pressure_enforced
    );

    run_decode_width(&engine, 3).await;
    assert_eq!(
        scheduler
            .trace_snapshot()
            .decode_capacity_backpressure_admit_limit,
        Some(5)
    );

    close_auxiliary_incarnation(&engine, &executor).await;
    run_decode_width(&engine, 5).await;
    assert_eq!(
        scheduler
            .trace_snapshot()
            .decode_capacity_backpressure_admit_limit,
        Some(6)
    );
    assert!(
        scheduler
            .trace_snapshot()
            .decode_execution_pressure_enforced
    );
}
