use super::*;
use crate::continuous_engine::inner::cost_observation::*;
use ferrum_interfaces::vnext::{self, ExecutionCostRouteAvailability, ExecutionCostRouteView};
use ferrum_interfaces::{engine::InferenceEngine, KvCacheHandle, ModelExecutor, Tokenizer};
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use std::sync::atomic::{AtomicBool, AtomicU64};

mod snapshot_epoch;

#[path = "../../../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod contract;

struct CoreEvidence {
    fixture: Option<contract::Fixture>,
    sessions: Vec<Arc<vnext::SequenceSession<contract::TestRuntime>>>,
    lane: Option<Arc<vnext::ExecutionLane<contract::TestRuntime>>>,
}
impl CoreEvidence {
    fn new(width: usize) -> Self {
        let fixture = contract::fixture();
        let sessions = (0..width)
            .map(|index| {
                let sequence = contract::logical_resources_with_work(
                    &fixture.plan_resources,
                    &format!("run.controller.{index}"),
                    &format!("request.controller.{index}"),
                    vnext::TokenSpanWork::from_token_ids(&[1; 8], 0..8).unwrap(),
                );
                sequence.open_session().unwrap()
            })
            .collect::<Vec<_>>();
        let borrowed = sessions.iter().map(Arc::as_ref).collect::<Vec<_>>();
        let lane = fixture.plan_resources.create_execution_lane().unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            match fixture.plan_resources.execution_cost_route_view(
                &borrowed,
                &vec![1; width],
                lane.as_ref(),
                ResourcePlanningLimits::default(),
                &mut || true,
            ) {
                ExecutionCostRouteAvailability::Known(_) => break,
                ExecutionCostRouteAvailability::Unknown(
                    vnext::ExecutionCostRouteUnknown::Resource(
                        ResourcePlanningUnknown::ReadUnavailable(stage),
                    ),
                ) => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "route fixture read stayed contended: {stage:?}"
                    );
                    std::thread::yield_now();
                }
                other => panic!("real fixture route view: {other:?}"),
            }
        }
        Self {
            fixture: Some(fixture),
            sessions,
            lane: Some(lane),
        }
    }
}
impl Drop for CoreEvidence {
    fn drop(&mut self) {
        self.sessions.clear();
        self.lane.take();
        if let Some(fixture) = self.fixture.take() {
            drop(fixture.registry);
            drop(fixture.impostor_registry);
            drop(fixture.runtime);
            assert!(matches!(
                vnext::PlanRuntimeResources::close(fixture.plan_resources),
                Ok(vnext::PlanRuntimeCloseOutcome::Closed(_))
            ));
        }
    }
}

pub(in crate::continuous_engine) struct ControlledExecutor {
    base: MockModelExecutor,
    evidence: CoreEvidence,
    /// Models PlanRuntime-owned state without installing a legacy handle.
    pub typed_sequence_state: Mutex<Option<TypedSequenceStateMemory>>,
    pub entries: AtomicUsize,
    pub physical: AtomicUsize,
    pub prefill_granularity: AtomicUsize,
    pub park: AtomicBool,
    pub fail_after_submit: AtomicBool,
    pub panic_before_submit: AtomicBool,
    pub narrow_last_mixed_prefill: AtomicBool,
    pub replan_before_encode: AtomicBool,
    /// Optional controlled-backend observation of the inputs actually executed.
    /// This is protocol evidence in tests, not a Metal timing/route claim.
    pub emit_cost_observations: AtomicBool,
    pub completion_work_known: AtomicBool,
    pub completion_fail: AtomicBool,
    pub completion_calls: AtomicUsize,
    pub discarded_prefills: Mutex<Vec<String>>,
    pub produced_caches: Mutex<Vec<std::sync::Weak<MockKvCacheHandle>>>,
    session_bindings: Mutex<Vec<RequestId>>,
    pub submitted_requests: Mutex<Vec<Vec<RequestId>>>,
    pub before_prefill_discard: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    pub after_resource_revalidation: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    pub resource_planning_unknown: Mutex<Option<ResourcePlanningUnknown>>,
    /// Last real route result, cleared by each capture attempt. This is
    /// diagnostic evidence, not an injected answer or permission to retry.
    pub cost_route_unknown: Mutex<Option<vnext::ExecutionCostRouteUnknown>>,
    pub resource_revalidation_changed: AtomicBool,
    pub entered: Notify,
    pub resume: Notify,
}

#[async_trait::async_trait]
impl ModelExecutor for ControlledExecutor {
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.base.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        let mut caps = self.base.capabilities();
        if let Some(state) = *self.typed_sequence_state.lock() {
            caps.memory_requirements.typed_sequence_state = Some(state);
        }
        caps
    }
    fn status(&self) -> ExecutorStatus {
        self.base.status()
    }
    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }
    fn slo_execution_capability(&self) -> ExecutorSloCapability {
        // The controlled backend executes all three guarded eager wave kinds
        // below. Missing future route evidence still returns Unknown; this is
        // an algorithm capability, never a promise of predictive coverage.
        ExecutorSloCapability::GuardedEagerWaves
    }
    fn guarded_prefill_granularity(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(self.prefill_granularity.load(Ordering::Acquire))
    }

    fn execution_cost_identity(&self) -> ExecutorCostIdentityAvailability {
        identity()
    }
    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        Ok(Some(ExecutorAdmissionEpochs::new(
            NonZeroU64::new(47).unwrap(),
            0,
            0,
        )))
    }
    fn write_execution_capacity_snapshot(
        &self,
        sources: &mut Vec<vnext::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        sources.clear();
        sources.push(
            vnext::CapacityAvailabilityEpoch::new(
                vnext::CapacityAvailabilitySource::ActiveSequenceSlots,
                1,
            )
            .unwrap(),
        );
        self.execution_capacity_epochs()
    }
    fn execution_resource_planning_view(
        &self,
        requests: &[ExecutorResourcePlanningRequest<'_>],
        limits: ResourcePlanningLimits,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningView> {
        *self.resource_planning_unknown.lock() = None;
        match self.route_for_requests(requests, limits, budget) {
            ExecutionCostRouteAvailability::Known(route) => {
                // The ordinary resource-only snapshot has no execution lane.
                // Use the real lane-bearing evidence required by exact work.
                ResourcePlanningAvailability::Known(route.resource_view().clone())
            }
            ExecutionCostRouteAvailability::Unknown(
                vnext::ExecutionCostRouteUnknown::Resource(reason),
            ) => {
                *self.resource_planning_unknown.lock() = Some(reason);
                ResourcePlanningAvailability::Unknown(reason)
            }
            ExecutionCostRouteAvailability::Unknown(_) => {
                *self.resource_planning_unknown.lock() =
                    Some(ResourcePlanningUnknown::InvalidInput);
                ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::InvalidInput)
            }
        }
    }
    fn execution_cost_route_view(
        &self,
        requests: &[ExecutorResourcePlanningRequest<'_>],
        limits: ResourcePlanningLimits,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<ExecutionCostRouteView> {
        self.route_for_requests(requests, limits, budget)
    }
    fn revalidate_execution_resource_planning_view(
        &self,
        requests: &[ExecutorResourcePlanningRequest<'_>],
        view: &ResourcePlanningView,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<bool> {
        // Preserve the real resource comparison; the one-shot hook only lets
        // tests race a real owner/epoch change between capture and publication.
        let result = match self.execution_resource_planning_view(requests, view.limits(), budget) {
            ResourcePlanningAvailability::Known(current) => {
                ResourcePlanningAvailability::Known(view.same_live_evidence(&current))
            }
            ResourcePlanningAvailability::Unknown(reason) => {
                ResourcePlanningAvailability::Unknown(reason)
            }
        };
        if matches!(result, ResourcePlanningAvailability::Known(false)) {
            self.resource_revalidation_changed
                .store(true, Ordering::Release);
        }
        if let Some(hook) = self.after_resource_revalidation.lock().take() {
            hook();
        }
        result
    }
    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        input.validate()?;
        Ok(ExecutorPrefillAdmissionDecision::Admitted(
            ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            },
        ))
    }
    fn cancel_prefill_admission(&self, _: &RequestId) -> bool {
        true
    }
    fn cancel_prefill_admission_observed(
        &self,
        id: &RequestId,
    ) -> ExecutorAdmissionCancellationObservation {
        let released = self.cancel_prefill_admission(id);
        ExecutorAdmissionCancellationObservation {
            released,
            work: if self.completion_work_known.load(Ordering::Acquire) {
                // This controlled executor's cancellation above performs no work.
                ExecutorCompletionWork::NoAdditionalWork
            } else {
                ExecutorCompletionWork::Unknown
            },
        }
    }
    async fn complete_cache(&self, completion: ExecutorSequenceCompletion) -> Result<()> {
        self.completion_calls.fetch_add(1, Ordering::AcqRel);
        self.release_cache(completion.cache_id());
        if self.completion_fail.load(Ordering::Acquire) {
            Err(FerrumError::backend("controlled completion failure"))
        } else {
            Ok(())
        }
    }
    async fn complete_cache_observed(
        &self,
        completion: ExecutorSequenceCompletion,
    ) -> ExecutorCompletionObservation {
        let result = self.complete_cache(completion).await;
        ExecutorCompletionObservation {
            result,
            work: if self.completion_work_known.load(Ordering::Acquire) {
                ExecutorCompletionWork::NoAdditionalWork
            } else {
                ExecutorCompletionWork::Unknown
            },
        }
    }
    fn discard_plan_runtime_prefill(&self, authority: PlanRuntimePrefillAuthority) -> Result<()> {
        if let Some(check) = self.before_prefill_discard.lock().as_ref() {
            check();
        }
        self.discarded_prefills
            .lock()
            .push(authority.kv_cache().cache_id());
        self.release_cache(&authority.kv_cache().cache_id());
        Ok(())
    }
    async fn prefill(&self, input: &PrefillInput) -> Result<PrefillOutput> {
        self.base.prefill(input).await
    }
    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        self.base.decode(input).await
    }
    async fn plan_runtime_prefill_with_capacity(
        &self,
        input: &PlanRuntimePrefillInput,
    ) -> Result<PlanRuntimePrefillOutcome> {
        self.prefill_output(input)
            .map(PlanRuntimePrefillOutcome::Completed)
    }
    async fn plan_runtime_batch_prefill_guarded_work_observed(
        &self,
        inputs: &[PlanRuntimePrefillInput],
        _: &ExpectedExecutionWave,
        guard: &dyn NonblockingHostSubmissionGuard,
        mut observation: GuardedCostObservation<'_, '_>,
    ) -> GuardedDispatchOutcome<Vec<PlanRuntimePrefillCompletion>> {
        if !self.enter_guarded(guard).await {
            return GuardedDispatchOutcome::ReplanBeforeEncode;
        }
        self.submitted_requests.lock().push(
            inputs
                .iter()
                .map(|input| input.request_id.clone())
                .collect(),
        );
        self.begin_cost_observation(observation.as_deref_mut(), inputs, &[]);
        let result = if self.fail_after_submit.load(Ordering::Acquire) {
            Err(FerrumError::backend("injected submitted prefill failure"))
        } else {
            inputs
                .iter()
                .map(|input| self.prefill_output(input))
                .collect()
        };
        self.end_cost_observation(observation, result.is_ok());
        GuardedDispatchOutcome::Submitted(result)
    }
    async fn plan_runtime_mixed_batch_guarded_work_observed(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
        _: &ExpectedExecutionWave,
        guard: &dyn NonblockingHostSubmissionGuard,
        mut observation: GuardedCostObservation<'_, '_>,
    ) -> GuardedDispatchOutcome<PlanRuntimeMixedBatchOutput> {
        if !self.enter_guarded(guard).await {
            return GuardedDispatchOutcome::ReplanBeforeEncode;
        }
        self.submitted_requests.lock().push(
            prefills
                .iter()
                .map(|input| input.request_id.clone())
                .chain(decodes.iter().map(|input| input.request_id.clone()))
                .collect(),
        );
        self.begin_cost_observation(observation.as_deref_mut(), prefills, decodes);
        let result = if self.fail_after_submit.load(Ordering::Acquire) {
            Err(FerrumError::backend("injected submitted mixed failure"))
        } else {
            prefills
                .iter()
                .enumerate()
                .map(|(index, input)| {
                    if index + 1 == prefills.len()
                        && self.narrow_last_mixed_prefill.load(Ordering::Acquire)
                    {
                        let planned = input.chunk;
                        let completed = PrefillChunk::new(
                            planned.tokens_processed(),
                            planned.tokens_to_process() - 1,
                            planned.total_prompt_tokens(),
                        )?;
                        let output = PlanRuntimePrefillOutput::intermediate(
                            input.request_id.clone(),
                            completed.end(),
                            self.cache(&input.request_id, completed.end()),
                        );
                        let completion =
                            PlanRuntimePrefillCompletion::new(output, planned, completed, 1)?;
                        completion.validate_for(
                            &input.request_id,
                            planned,
                            self.info().vocab_size,
                        )?;
                        Ok(completion)
                    } else {
                        self.prefill_output(input)
                    }
                })
                .collect::<Result<Vec<_>>>()
                .map(|prefills| PlanRuntimeMixedBatchOutput {
                    prefills,
                    decodes: self.decode_outputs(decodes),
                })
        };
        self.end_cost_observation(observation, result.is_ok());
        GuardedDispatchOutcome::Submitted(result)
    }
    async fn plan_runtime_batch_decode_guarded_work_observed(
        &self,
        inputs: &[PlanRuntimeDecodeInput],
        _: &ExpectedExecutionWave,
        guard: &dyn NonblockingHostSubmissionGuard,
        mut observation: GuardedCostObservation<'_, '_>,
    ) -> GuardedDispatchOutcome<Vec<PlanRuntimeDecodeOutput>> {
        if !self.enter_guarded(guard).await {
            return GuardedDispatchOutcome::ReplanBeforeEncode;
        }
        self.submitted_requests.lock().push(
            inputs
                .iter()
                .map(|input| input.request_id.clone())
                .collect(),
        );
        self.begin_cost_observation(observation.as_deref_mut(), &[], inputs);
        if self.fail_after_submit.load(Ordering::Acquire) {
            self.end_cost_observation(observation, false);
            return GuardedDispatchOutcome::Submitted(Err(FerrumError::backend(
                "injected submitted failure",
            )));
        }
        let result = self.decode_outputs(inputs);
        self.end_cost_observation(observation, true);
        GuardedDispatchOutcome::Submitted(Ok(result))
    }
}

impl ControlledExecutor {
    pub(in crate::continuous_engine) fn abort_resource_sessions(&self) {
        for session in &self.evidence.sessions {
            session.try_abort_if_quiescent().unwrap();
        }
    }

    fn begin_cost_observation(
        &self,
        context: GuardedCostObservation<'_, '_>,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
    ) {
        if !self.emit_cost_observations.load(Ordering::Acquire) {
            return;
        }
        let Some(context) = context else {
            return;
        };
        if self.narrow_last_mixed_prefill.load(Ordering::Acquire) {
            context.mark_unknown(ActualWaveEvidenceUnknown::ProviderPath);
            return;
        }
        let mut rows = Vec::with_capacity(prefills.len() + decodes.len());
        let mut numeric = Vec::with_capacity(rows.capacity());
        let mut output_identity = sha2::Sha256::new();
        use sha2::Digest;
        output_identity.update(b"ferrum.controlled-backend.actual-output.v1");
        for (id, work, output) in prefills
            .iter()
            .map(|input| {
                (
                    &input.request_id,
                    ActualRowWork::Prefill {
                        offset: input.chunk.tokens_processed() as u32,
                        count: input.chunk.tokens_to_process() as u32,
                        total_prompt_tokens: input.chunk.total_prompt_tokens() as u32,
                    },
                    CostRowOutput::Prefill {
                        final_logits: input.chunk.is_final(),
                    },
                )
            })
            .chain(decodes.iter().map(|input| {
                (
                    &input.request_id,
                    ActualRowWork::Decode {
                        kv_tokens: input.kv_cache.num_tokens() as u32,
                    },
                    CostRowOutput::Decode {
                        requires_full_logits: input.logits_policy.requires_full_logits(),
                        repetition_tokens: 0,
                        repetition_penalty_bits: 1_f32.to_bits(),
                    },
                )
            }))
        {
            let participant = context
                .participant(id)
                .expect("actual input has caller correlation");
            let host = participant
                .host_features
                .expect("bounded fixture host features");
            output_identity.update(participant.output_policy_signature.unwrap());
            output_identity.update(format!("{work:?}/{output:?}").as_bytes());
            numeric.push(project_host_cost_features(host, work, output).unwrap());
            rows.push(ActualWaveRow {
                request_id: id.clone(),
                owner_incarnation: participant.owner_incarnation,
                work_generation: participant.work_generation,
                input_index: participant.input_index,
                work,
            });
        }
        let signature: [u8; 32] = output_identity.finalize().into();
        let shape = ActualWaveShape {
            statistical_evidence: None,
            kind: match (prefills.is_empty(), decodes.is_empty()) {
                (true, _) => ActualWaveKind::Decode,
                (_, true) => ActualWaveKind::Prefill,
                _ => ActualWaveKind::Mixed,
            },
            path: ActualWavePath::PlanRuntime,
            graph: ActualWaveGraphState::Disabled,
            row_order: ActualWaveRowOrder::Ordered,
            provider_signature: sha2::Sha256::digest(b"controlled-backend.host-greedy-cache.v1")
                .into(),
            output_policy_signature: signature,
            numeric_features: Some(CanonicalWaveCostFeatures {
                schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
                output_policy_signature: signature,
                rows: numeric,
            }),
            host_content_features: None,
            row_multiset_features: None,
            rows,
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        };
        let start = context.now_ns();
        context.physical_wave(Ok(shape), start);
    }
    fn end_cost_observation(&self, context: GuardedCostObservation<'_, '_>, success: bool) {
        if !self.emit_cost_observations.load(Ordering::Acquire) {
            return;
        }
        if let Some(context) = context {
            context.terminal(
                if success {
                    ActualWaveOutcome::Completed
                } else {
                    ActualWaveOutcome::FailedAfterSubmit
                },
                None,
            );
            context.finish_call(if success {
                ObservedCallOutcome::Completed
            } else {
                ObservedCallOutcome::Failed
            });
        }
    }
    fn route_for_requests(
        &self,
        requests: &[ExecutorResourcePlanningRequest<'_>],
        limits: ResourcePlanningLimits,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> ExecutionCostRouteAvailability<ExecutionCostRouteView> {
        *self.cost_route_unknown.lock() = None;
        if requests.is_empty()
            || requests.len() > self.evidence.sessions.len()
            || !budget.has_budget()
        {
            return ExecutionCostRouteAvailability::Unknown(
                vnext::ExecutionCostRouteUnknown::InvalidInput,
            );
        }
        let mut bindings = self.session_bindings.lock();
        let mut indices = Vec::with_capacity(requests.len());
        for request in requests {
            let index = match bindings.iter().position(|id| id == request.request_id) {
                Some(index) => index,
                None if bindings.len() < self.evidence.sessions.len() => {
                    bindings.push(request.request_id.clone());
                    bindings.len() - 1
                }
                None => {
                    return ExecutionCostRouteAvailability::Unknown(
                        vnext::ExecutionCostRouteUnknown::InvalidInput,
                    )
                }
            };
            if indices.contains(&index) {
                return ExecutionCostRouteAvailability::Unknown(
                    vnext::ExecutionCostRouteUnknown::InvalidInput,
                );
            }
            indices.push(index);
        }
        drop(bindings);
        let sessions = indices
            .iter()
            .map(|&index| self.evidence.sessions[index].as_ref())
            .collect::<Vec<_>>();
        let route = self
            .evidence
            .fixture
            .as_ref()
            .unwrap()
            .plan_resources
            .execution_cost_route_view(
                &sessions,
                &vec![1; sessions.len()],
                self.evidence.lane.as_ref().unwrap().as_ref(),
                limits,
                budget,
            );
        if let ExecutionCostRouteAvailability::Unknown(reason) = &route {
            *self.cost_route_unknown.lock() = Some(*reason);
            eprintln!("controlled executor cost route unavailable: {reason:?}");
        }
        route
    }

    fn cache(&self, request: &RequestId, tokens: usize) -> Arc<MockKvCacheHandle> {
        let cache = Arc::new(MockKvCacheHandle::new(request.clone(), 1, tokens));
        self.produced_caches.lock().push(Arc::downgrade(&cache));
        cache
    }
    async fn enter_guarded(&self, guard: &dyn NonblockingHostSubmissionGuard) -> bool {
        self.entries.fetch_add(1, Ordering::AcqRel);
        self.entered.notify_one();
        if self.park.load(Ordering::Acquire) {
            self.resume.notified().await;
        }
        assert!(
            !self.panic_before_submit.load(Ordering::Acquire),
            "injected executor panic before physical counter"
        );
        // This fake gate is before encode; native post-encode cleanup has its
        // own real Metal tests and private receipt, never fabricated here.
        if guard.check().is_err() {
            return false;
        }
        // The controlled backend can decline before any encode. This is not a
        // fabricated native post-encode rollback receipt.
        if self.replan_before_encode.load(Ordering::Acquire) {
            return false;
        }
        self.physical.fetch_add(1, Ordering::AcqRel);
        true
    }
    fn prefill_output(
        &self,
        input: &PlanRuntimePrefillInput,
    ) -> Result<PlanRuntimePrefillCompletion> {
        let cache = self.cache(&input.request_id, input.chunk.end());
        let output = if input.chunk.is_final() {
            let mut logits = vec![0.0; self.info().vocab_size];
            logits[6] = 1.0;
            PlanRuntimePrefillOutput::final_logits(
                input.request_id.clone(),
                input.chunk.end(),
                logits,
                cache,
            )?
        } else {
            PlanRuntimePrefillOutput::intermediate(
                input.request_id.clone(),
                input.chunk.end(),
                cache,
            )
        };
        PlanRuntimePrefillCompletion::new(output, input.chunk, input.chunk, 0)
    }
    fn decode_outputs(&self, inputs: &[PlanRuntimeDecodeInput]) -> Vec<PlanRuntimeDecodeOutput> {
        inputs
            .iter()
            .map(|input| {
                let output = if input.logits_policy.requires_full_logits() {
                    let mut logits = vec![0.0; self.info().vocab_size];
                    logits[6] = 1.0;
                    ExecutorSamplingOutput::FullLogits(logits)
                } else {
                    ExecutorSamplingOutput::GreedyToken(ferrum_types::TokenId::new(6))
                };
                let cache: Arc<dyn KvCacheHandle> =
                    self.cache(&input.request_id, input.kv_cache.num_tokens() + 1);
                PlanRuntimeDecodeOutput::new(output, cache)
            })
            .collect()
    }
}

fn identity() -> ExecutorCostIdentityAvailability {
    ExecutorCostIdentityAvailability::Known(Arc::new(ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }))
}
pub(super) fn canonical(context: u32) -> CanonicalWaveCostShape {
    CanonicalWaveCostShape {
        kind: ActualWaveKind::Decode,
        path: ActualWavePath::PlanRuntime,
        graph: ActualWaveGraphState::Disabled,
        row_order: ActualWaveRowOrder::Ordered,
        provider_signature: [5; 32],
        output_policy_signature: [6; 32],
        numeric_features: None,
        host_content_features: None,
        row_multiset_features: None,
        rows: vec![ActualRowWork::Decode { kv_tokens: context }],
        recurrent_state_bytes: 0,
    }
}
struct Clock(AtomicU64);
impl CostObservationClock for Clock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(Ordering::Relaxed))
    }
}

fn trained_runtime() -> Arc<EngineCostRuntime> {
    let clock = Arc::new(Clock(AtomicU64::new(11)));
    let runtime = Arc::new(EngineCostRuntime::with_clock(identity(), clock).unwrap());
    for _ in 0
        ..ferrum_scheduler::implementations::continuous::cost_model::CostModelSettings::default()
            .min_samples
            .get()
    {
        let id = RequestId::new();
        let sample_clock = Arc::new(Clock(AtomicU64::new(2)));
        let mut call = EngineCostCall::begin(
            &runtime.ids,
            sample_clock.clone(),
            runtime.sink.clone(),
            EngineCostCallSpec {
                identity: identity(),
                participants: vec![CostObservationParticipant {
                    request_id: id.clone(),
                    owner_incarnation: 1,
                    work_generation: 1,
                    input_index: 0,
                    output_policy_signature: Some([6; 32]),
                    host_features: None,
                }],
                prepare_started_at_ns: Some(1),
                boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
                recorder_limits: runtime.recorder_limits,
            },
        )
        .unwrap();
        {
            let mut context = call.context().unwrap();
            context.physical_wave(
                Ok(ActualWaveShape {
                    statistical_evidence: None,
                    kind: ActualWaveKind::Decode,
                    path: ActualWavePath::PlanRuntime,
                    graph: ActualWaveGraphState::Disabled,
                    row_order: ActualWaveRowOrder::Ordered,
                    provider_signature: [5; 32],
                    output_policy_signature: [6; 32],
                    numeric_features: None,
                    host_content_features: None,
                    row_multiset_features: None,
                    rows: vec![ActualWaveRow {
                        request_id: id.clone(),
                        owner_incarnation: 1,
                        work_generation: 1,
                        input_index: 0,
                        work: ActualRowWork::Decode { kv_tokens: 1 },
                    }],
                    recurrent_state_bytes: 0,
                    restore_bytes: 0,
                    maintenance_bytes: 0,
                    maintenance_units: 0,
                }),
                Some(3),
            );
            sample_clock.0.store(6, Ordering::Relaxed);
            context.terminal(ActualWaveOutcome::Completed, None);
            context.finish_call(ObservedCallOutcome::Completed);
        }
        call.record_host_result(HostCommitEvidence {
            request_id: id,
            owner_incarnation: 1,
            work_generation: 1,
            input_index: 0,
            outcome: HostCommitOutcome::Committed(HostCommittedWork::Decode {
                kv_tokens_before: 1,
                kv_tokens_after: 2,
                generated_tokens_before: 1,
                generated_tokens_after: 2,
            }),
            committed_at_ns: Some(9),
        });
        sample_clock.0.store(10, Ordering::Relaxed);
        assert_eq!(call.finish(), CostCallDisposition::Published);
    }
    runtime.consume_samples();
    assert!(runtime.snapshot().is_some());
    runtime
}

pub(super) async fn fixture() -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    fixture_with_width(1).await
}

pub(in crate::continuous_engine) async fn startup_components(
    width: usize,
) -> (Arc<dyn Tokenizer + Send + Sync>, Arc<ControlledExecutor>) {
    let vocab = (0..64)
        .map(|id| {
            (
                match id {
                    5 => "test".into(),
                    6 => "ok".into(),
                    _ => format!("v{id}"),
                },
                id,
            )
        })
        .collect();
    let mut raw = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("v0".into())
            .build()
            .unwrap(),
    );
    raw.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    raw.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::Whitespace::default(),
    ));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(HuggingFaceTokenizer::new(raw).await.unwrap());
    let executor = Arc::new(ControlledExecutor {
        base: MockModelExecutor::instant(64),
        evidence: CoreEvidence::new(width),
        typed_sequence_state: Mutex::new(None),
        entries: AtomicUsize::new(0),
        physical: AtomicUsize::new(0),
        prefill_granularity: AtomicUsize::new(1),
        park: AtomicBool::new(false),
        fail_after_submit: AtomicBool::new(false),
        panic_before_submit: AtomicBool::new(false),
        narrow_last_mixed_prefill: AtomicBool::new(false),
        replan_before_encode: AtomicBool::new(false),
        emit_cost_observations: AtomicBool::new(false),
        completion_work_known: AtomicBool::new(false),
        completion_fail: AtomicBool::new(false),
        completion_calls: AtomicUsize::new(0),
        discarded_prefills: Mutex::new(Vec::new()),
        produced_caches: Mutex::new(Vec::new()),
        session_bindings: Mutex::new(Vec::new()),
        submitted_requests: Mutex::new(Vec::new()),
        before_prefill_discard: Mutex::new(None),
        after_resource_revalidation: Mutex::new(None),
        resource_planning_unknown: Mutex::new(None),
        cost_route_unknown: Mutex::new(None),
        resource_revalidation_changed: AtomicBool::new(false),
        entered: Notify::new(),
        resume: Notify::new(),
    });
    (tokenizer, executor)
}

pub(in crate::continuous_engine::inner) async fn fixture_with_width(
    width: usize,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    let (tokenizer, executor) = startup_components(width).await;
    let mut config = ferrum_types::EngineConfig::default();
    config.scheduler.slo.output.max_queued_events_per_request = NonZeroUsize::new(2).unwrap();
    // Functional branch tests have no wall-clock performance threshold. The
    // dedicated virtual-clock tests exercise actual planning budget exhaustion.
    config.scheduler.slo.planner.max_planning_us = NonZeroU64::new(30_000_000).unwrap();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let mut engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler.clone(),
        tokenizer,
        Arc::new(crate::registry::GreedySampler),
        executor.clone(),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Observe;
    inner.config.scheduler.slo.default_service_class = Some("controller-test".into());
    inner
        .config
        .scheduler
        .slo
        .services
        .push(ferrum_types::ServiceSloConfig {
            id: "controller-test".into(),
            server_token_commit: ferrum_types::SloLatencyBudgets {
                ttft_ms: NonZeroU64::new(10_000).unwrap(),
                tpot_ms: NonZeroU64::new(10_000).unwrap(),
                itl_ms: NonZeroU64::new(10_000).unwrap(),
            },
            client_visible: None,
            attainment: Default::default(),
        });
    inner.cost_runtime = Some(trained_runtime());
    inner.prefill_reference_runtime = Some(
        crate::continuous_engine::inner::prefill_reference_runtime::test_calibration_runtime(),
    );
    inner.bg_loop_spawned.store(true, Ordering::Release);
    (engine, scheduler, executor)
}

pub(super) async fn completion_fixture(
    width: usize,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    let (mut engine, scheduler, executor) = fixture_with_width(width).await;
    let inner = Arc::get_mut(&mut engine.inner).unwrap();
    inner.config.scheduler.slo.mode = ferrum_types::SloMode::Enforce;
    inner.config.scheduler.slo.admission.time_policy =
        ferrum_types::SloTimeAdmissionPolicy::CompleteRequests;
    inner.prefill_reference_runtime = None;
    // Retain the real frontier/identity allocator, but publish no learned cost.
    inner.cost_runtime = Some(Arc::new(
        EngineCostRuntime::with_clock(identity(), Arc::new(Clock(AtomicU64::new(11)))).unwrap(),
    ));
    assert!(inner.cost_runtime.as_ref().unwrap().snapshot().is_none());
    (engine, scheduler, executor)
}

pub(super) fn pool(
    engine: &ContinuousBatchEngine,
) -> &ferrum_interfaces::output_credit::OutputCreditPool {
    engine
        .inner
        .output_credit_pool
        .get()
        .unwrap()
        .as_ref()
        .unwrap()
}
pub(in crate::continuous_engine::inner::slo_controller) async fn cleanup(
    engine: ContinuousBatchEngine,
    session: CreditedOutputSession,
) {
    drop(session);
    engine.shutdown().await.unwrap();
    let mut changes = pool(&engine).subscribe();
    bounded(async {
        loop {
            let snapshot = pool(&engine).snapshot();
            if snapshot.data_used == Default::default()
                && snapshot.terminal_held == Default::default()
            {
                break;
            }
            changes.changed().await.unwrap();
        }
    })
    .await;
}
