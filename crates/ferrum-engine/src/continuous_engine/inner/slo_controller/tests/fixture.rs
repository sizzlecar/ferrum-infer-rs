use super::*;
use crate::continuous_engine::inner::cost_observation::*;
use ferrum_interfaces::vnext::{self, ExecutionCostRouteAvailability, ExecutionCostRouteView};
use ferrum_interfaces::{engine::InferenceEngine, KvCacheHandle, ModelExecutor, Tokenizer};
use ferrum_tokenizer::implementations::HuggingFaceTokenizer;
use std::sync::atomic::{AtomicBool, AtomicU64};

mod admission_capacity;
mod deferral;
mod native_sessions;
mod native_structured;
mod prefix_checkpoint;
mod query_projection;
mod query_resources;
mod revalidation_retry;
mod snapshot_epoch;
mod structured;
pub(in crate::continuous_engine::inner) use structured::private_outside_pending;
mod token_policy_residency;

#[path = "../../../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod contract;

struct CoreEvidence {
    prefix: Option<prefix_checkpoint::NativePrefix>,
    fixture: Option<contract::Fixture>,
    sessions: native_sessions::Sessions,
    session_serial: AtomicU64,
    lane: Option<Arc<vnext::ExecutionLane<contract::TestRuntime>>>,
}
impl CoreEvidence {
    fn new(width: usize) -> Self {
        Self::new_options(width, false)
    }
    fn new_options(width: usize, checkpoint: bool) -> Self {
        // These fixtures test one controller's behavior. Unrelated parallel
        // tests must not change its process-wide device capacity evidence.
        // Shared-account resource tests continue to declare a shared DeviceId.
        static NEXT_DEVICE: AtomicU64 = AtomicU64::new(0);
        let instance = NEXT_DEVICE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("controller fixture device identities exhausted");
        let device = vnext::DeviceId::new(format!("device.controller-fixture.{instance}")).unwrap();
        let fixture = if checkpoint {
            contract::fixture_with_checkpoint(device)
        } else {
            contract::fixture_with_device_id(device)
        };
        let sessions = (0..width)
            .map(|index| {
                let sequence = contract::logical_resources_with_work(
                    &fixture.plan_resources,
                    &format!("run.controller.{index}"),
                    &format!("request.controller.{index}"),
                    if checkpoint {
                        // Exact initial prompt used by this opt-in fixture.
                        // The separate generation ceiling remains eight.
                        vnext::TokenSpanWork::from_token_ids_with_fit(&[5; 4], 0..4, 8).unwrap()
                    } else {
                        vnext::TokenSpanWork::from_token_ids(&[1; 8], 0..8).unwrap()
                    },
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
            prefix: checkpoint.then(prefix_checkpoint::NativePrefix::new),
            fixture: Some(fixture),
            sessions: native_sessions::Sessions::new(sessions),
            session_serial: AtomicU64::new(0),
            lane: Some(lane),
        }
    }
}
impl Drop for CoreEvidence {
    fn drop(&mut self) {
        self.prefix.take();
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
    pub startup_capability_override: Mutex<Option<ExecutorSloCapability>>,
    pub profile_sink: Mutex<Option<Arc<dyn vnext::ExecutionEventSink>>>,
    /// Synthetic input obligations for planner coverage tests only. Actual
    /// CPU fill projections and resources remain independently authoritative;
    /// this does not claim that the fill operator changes kernels here.
    pub decode_context_coverage_override: Mutex<Option<Arc<vnext::ExecutorDecodeContextCoverage>>>,
    pub entries: AtomicUsize,
    pub physical: AtomicUsize,
    /// Opt-in implementation of the ordinary tensor-free wave protocol for
    /// controlled adaptive/single-wave comparisons. Existing guard fixtures
    /// keep their original unsupported batch defaults.
    pub plain_wave_protocol: AtomicBool,
    pub prefill_granularity: AtomicUsize,
    pub park: AtomicBool,
    /// Opt-in gate after successful physical observation, before the host
    /// receives outputs. Unlike `park`, this cannot exercise a zero-submit path.
    pub park_after_submit: AtomicBool,
    pub fail_after_submit: AtomicBool,
    pub panic_before_submit: AtomicBool,
    pub narrow_last_mixed_prefill: AtomicBool,
    pub replan_before_encode: AtomicBool,
    /// Optional controlled-backend observation of the inputs actually executed.
    /// This is protocol evidence in tests, not a Metal timing/route claim.
    pub emit_cost_observations: AtomicBool,
    /// A controlled backend may lack an actual route while its real submitted
    /// work still completes. This hook feeds the original bounded recorder;
    /// tests never manufacture a CalibrationWaveReport or accepted sample.
    pub actual_observation_unknown: Mutex<
        Option<
            Box<
                dyn Fn(
                        &[PlanRuntimePrefillInput],
                        &[PlanRuntimeDecodeInput],
                    ) -> Option<ActualWaveEvidenceUnknown>
                    + Send
                    + Sync,
            >,
        >,
    >,
    pub emit_structured_cost_observations: AtomicBool,
    pub native_structured_submission: AtomicBool,
    /// Opt-in original outside selector for real private-prefix CPU preparation.
    pub native_prefix_preparation_outside: AtomicBool,
    pub native_prefix_preparation_outside_submissions: AtomicUsize,
    native_structured_history: Arc<Mutex<std::collections::HashMap<RequestId, Vec<u32>>>>,
    token_policy_lifecycle: Mutex<token_policy_residency::Lifecycle>,
    /// Opt-in future algebra for this fixture's actual CPU logits fill.
    pub project_structured_cpu_fill: AtomicBool,
    /// A real constrained CPU route: only single-row prefill is installed,
    /// while the existing joint decode remains supported.
    pub single_row_prefill_only: AtomicBool,
    /// Test-only route readiness fault at the actual pure projection boundary.
    /// The callback observes original completed CPU submissions; returning
    /// None resumes the normal physical resource/provider projection.
    pub projection_readiness_fault: Mutex<
        Option<Box<dyn Fn(usize, usize) -> Option<vnext::ExecutionCostRouteUnknown> + Send + Sync>>,
    >,
    /// Select real CPU implementations from each decode row's context.
    pub context_partitioned_cpu_fill: AtomicBool,
    pub row_selected_cpu_fill: AtomicBool,
    pub completion_work_known: AtomicBool,
    pub completion_fail: AtomicBool,
    pub completion_calls: AtomicUsize,
    pub discarded_prefills: Mutex<Vec<String>>,
    pub produced_caches: Mutex<Vec<std::sync::Weak<MockKvCacheHandle>>>,
    session_bindings: Mutex<Vec<RequestId>>,
    /// Sequential calibration can reuse the one physical fixture slot only
    /// after its old request completed and every actual KV owner was dropped.
    pub recycle_completed_bindings: AtomicBool,
    completed_bindings: Mutex<std::collections::HashSet<RequestId>>,
    pub submitted_requests: Mutex<Vec<Vec<RequestId>>>,
    pub before_prefill_discard: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    pub before_resource_revalidation: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    pub after_resource_revalidation: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    pub resource_planning_unknown: Mutex<Option<ResourcePlanningUnknown>>,
    /// Last real route result, cleared by each capture attempt. This is
    /// diagnostic evidence, not an injected answer or permission to retry.
    pub cost_route_unknown: Mutex<Option<vnext::ExecutionCostRouteUnknown>>,
    pub resource_revalidation_changed: AtomicBool,
    pub deferrals: deferral::ControlledDeferrals,
    admission_capacity: Mutex<Option<admission_capacity::RealAdmission>>,
    pub entered: Notify,
    pub resume: Notify,
    pub submitted: Notify,
    pub resume_after_submit: Notify,
}

#[async_trait::async_trait]
impl ModelExecutor for ControlledExecutor {
    fn decode_context_coverage(&self) -> Arc<vnext::ExecutorDecodeContextCoverage> {
        self.decode_context_coverage_override
            .lock()
            .clone()
            .unwrap_or_default()
    }

    fn calibration_invalidate_token_policy_residency(
        &self,
    ) -> ferrum_interfaces::model_executor::TokenPolicyResidencyInvalidation {
        token_policy_residency::invalidate(self)
    }

    fn supports_guarded_prefix_maintenance(&self) -> bool {
        self.evidence.prefix.is_some()
    }
    fn supports_guarded_prefix_maintenance_for(
        &self,
        purpose: ferrum_interfaces::model_executor::PrefixCapturePurpose,
    ) -> bool {
        match purpose {
            ferrum_interfaces::model_executor::PrefixCapturePurpose::SharedCache
            | ferrum_interfaces::model_executor::PrefixCapturePurpose::PrivateCalibration => {
                self.evidence.prefix.is_some()
            }
        }
    }
    fn install_checkpoint_observation_sink(
        &self,
        sink: std::sync::Weak<dyn vnext::NativeCheckpointObservationSink>,
    ) -> Result<bool> {
        let Some(prefix) = self.evidence.prefix.as_ref() else {
            return Ok(false);
        };
        prefix.install_observer(sink)?;
        Ok(true)
    }
    fn plan_prefix_capture_boundary(
        &self,
        input: ferrum_interfaces::model_executor::PrefixCaptureBoundary<'_>,
    ) -> Option<ferrum_interfaces::model_executor::PrefixCapturePlan> {
        self.prefix_boundary(input)
    }
    fn plan_prefix_capture_boundary_for(
        &self,
        purpose: ferrum_interfaces::model_executor::PrefixCapturePurpose,
        input: ferrum_interfaces::model_executor::PrefixCaptureBoundary<'_>,
    ) -> Option<ferrum_interfaces::model_executor::PrefixCapturePlan> {
        self.supports_guarded_prefix_maintenance_for(purpose)
            .then(|| self.prefix_boundary(input))
            .flatten()
    }
    fn plan_prompt_tail_capture_boundary(
        &self,
        chunk: ferrum_interfaces::model_executor::PrefillChunk,
    ) -> Option<ferrum_interfaces::model_executor::PrefixCapturePlan> {
        self.prefix_prompt_tail(chunk)
    }
    fn try_retain_ready_prefix(
        &self,
        input: ferrum_interfaces::model_executor::PrefixReadyRestoreRequest<'_>,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<
        Option<Arc<dyn ferrum_interfaces::model_executor::PrefixCaptureLease>>,
    > {
        self.prefix_ready(input, budget)
    }
    fn bind_execution_ready_checkpoint(
        &self,
        view: &vnext::ExecutionCostRouteView,
        state: &vnext::ExecutionCostRouteState,
        lease: &dyn ferrum_interfaces::model_executor::PrefixCaptureLease,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<vnext::FutureRetainedCheckpointBinding> {
        self.prefix_ready_bind(view, state, lease, budget)
    }
    fn retain_prefix_capture_interest(
        &self,
        input: ferrum_interfaces::model_executor::PrefixCaptureRequest<'_>,
    ) -> Result<Option<Arc<dyn ferrum_interfaces::model_executor::PrefixCaptureLease>>> {
        Ok(self.prefix_interest(input))
    }
    async fn try_capture_plan_runtime_prefix_guarded(
        &self,
        input: ferrum_interfaces::model_executor::PrefixCaptureRequest<'_>,
        guard: Arc<dyn vnext::CheckpointTransferSubmissionGuard>,
    ) -> Result<bool> {
        self.prefix_capture(input, guard.as_ref())
    }
    async fn try_restore_plan_runtime_prefix_guarded(
        &self,
        input: ferrum_interfaces::model_executor::PlanRuntimePrefixRestoreInput<'_>,
        guard: Arc<dyn vnext::CheckpointTransferSubmissionGuard>,
    ) -> Result<ferrum_interfaces::model_executor::PlanRuntimePrefixRestoreOutcome> {
        self.prefix_restore(input, guard.as_ref())
    }
    fn project_execution_checkpoint(
        &self,
        view: &vnext::ExecutionCostRouteView,
        state: &vnext::ExecutionCostRouteState,
        query: vnext::FutureCheckpointCostQuery<'_>,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<vnext::FutureCheckpointCostProjection> {
        self.prefix_project(view, state, query, budget)
    }
    fn bind_execution_retained_checkpoint(
        &self,
        view: &vnext::ExecutionCostRouteView,
        state: &vnext::ExecutionCostRouteState,
        lease: &dyn ferrum_interfaces::model_executor::PrefixCaptureLease,
        source: usize,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<vnext::FutureRetainedCheckpointBinding> {
        self.prefix_bind(view, state, lease, source, budget)
    }
    fn execution_checkpoint_restore_completed(
        &self,
        view: &vnext::ExecutionCostRouteView,
        lease: &dyn ferrum_interfaces::model_executor::PrefixCaptureLease,
        target: usize,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<bool> {
        self.prefix_restored(view, lease, target, budget)
    }
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
        if self
            .profile_sink
            .lock()
            .as_ref()
            .is_some_and(|sink| !sink.device_timing_mode().guarded_completion_compatible())
        {
            return ExecutorSloCapability::Unavailable;
        }
        // The controlled backend executes all three guarded eager wave kinds
        // below. Missing future route evidence still returns Unknown; this is
        // an algorithm capability, never a promise of predictive coverage.
        self.startup_capability_override
            .lock()
            .unwrap_or(ExecutorSloCapability::GuardedEagerWaves)
    }
    fn attach_execution_event_sink(&self, sink: Arc<dyn vnext::ExecutionEventSink>) {
        *self.profile_sink.lock() = Some(sink);
    }
    fn guarded_prefill_granularity(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(self.prefill_granularity.load(Ordering::Acquire))
    }

    fn execution_cost_identity(&self) -> ExecutorCostIdentityAvailability {
        identity()
    }
    fn cost_workload_domain(&self) -> CostWorkloadDomainAvailability {
        self.evidence
            .prefix
            .as_ref()
            .and_then(|prefix| prefix.cost_domain.get().cloned())
            .map(CostWorkloadDomainAvailability::Known)
            .unwrap_or_default()
    }
    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        if let Some(admission) = self.admission_capacity.lock().as_ref() {
            return Ok(Some(admission.epochs(&mut Vec::new())));
        }
        Ok(Some(ExecutorAdmissionEpochs::new(
            NonZeroU64::new(47).unwrap(),
            0,
            self.deferrals.capacity_epoch.load(Ordering::Acquire),
        )))
    }
    fn write_execution_capacity_snapshot(
        &self,
        sources: &mut Vec<vnext::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        if let Some(admission) = self.admission_capacity.lock().as_ref() {
            return Ok(Some(admission.epochs(sources)));
        }
        sources.clear();
        sources.push(
            vnext::CapacityAvailabilityEpoch::new(
                vnext::CapacityAvailabilitySource::ActiveSequenceSlots,
                self.deferrals.capacity_epoch.load(Ordering::Acquire) + 1,
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
        self.deferrals
            .planning_captures
            .fetch_add(1, Ordering::AcqRel);
        self.route_for_requests(requests, limits, budget)
    }

    fn project_execution_cost_wave(
        &self,
        view: &vnext::ExecutionCostRouteView,
        state: &vnext::ExecutionCostRouteState,
        query: &vnext::FutureWaveCostQuery<'_>,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<vnext::ExecutionCostRouteProjection> {
        query_projection::project(self, view, state, query, budget)
    }

    fn project_execution_cost_wave_with_host_content(
        &self,
        view: &vnext::ExecutionCostRouteView,
        state: &vnext::ExecutionCostRouteState,
        query: &vnext::FutureWaveCostQuery<'_>,
        host: &vnext::FutureHostPendingQueryV2<'_>,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<vnext::ExecutionCostRouteForecastV2> {
        query_projection::project_with_host_content(self, view, state, query, host, budget)
    }

    fn maintain_execution_capacity_once(
        &self,
        ticket: ExecutorExecutionMaintenanceTicket,
        guard: &dyn NonblockingHostSubmissionGuard,
    ) -> Result<ExecutorExecutionMaintenanceOutcome> {
        self.deferrals.maintain(ticket, guard)
    }
    fn revalidate_execution_resource_planning_view(
        &self,
        requests: &[ExecutorResourcePlanningRequest<'_>],
        view: &ResourcePlanningView,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<bool> {
        // Preserve the real resource comparison; the one-shot hook only lets
        // tests race a real owner/epoch change between capture and publication.
        if let Some(hook) = self.before_resource_revalidation.lock().take() {
            hook();
        }
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
        let mut lifecycle = self.token_policy_lifecycle.lock();
        let decision = match self.admission_capacity.lock().as_mut() {
            Some(admission) => admission.admit(input),
            None => ExecutorPrefillAdmissionDecision::Admitted(ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            }),
        };
        if let ExecutorPrefillAdmissionDecision::Admitted(receipt) = &decision {
            self.admit_checkpoint_native_session(input)?;
            lifecycle.admitted.insert(receipt.request_id.clone());
        }
        Ok(decision)
    }
    fn cancel_prefill_admission(&self, id: &RequestId) -> bool {
        if let Some(prefix) = self.evidence.prefix.as_ref() {
            prefix.cancel_pending(id);
        }
        let mut lifecycle = self.token_policy_lifecycle.lock();
        let released = match self.admission_capacity.lock().as_mut() {
            Some(admission) => admission.cancel(id),
            None => true,
        };
        if released {
            lifecycle.admitted.remove(id);
        }
        if released && self.recycle_completed_bindings.load(Ordering::Acquire) {
            // A cancelled fresh root has no cache and never calls complete_cache.
            // Existing KV owners still require their normal completion path.
            let cache_id = format!("mock_{id}");
            let no_cache = self
                .produced_caches
                .lock()
                .iter()
                .filter_map(std::sync::Weak::upgrade)
                .all(|cache| cache.cache_id() != cache_id);
            if no_cache {
                self.completed_bindings.lock().insert(id.clone());
            }
        }
        released
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
    fn release_cache(&self, cache_id: &str) {
        // This opt-in fixture owns real native histories. Production cancels
        // its registry owner when the engine releases an errored/cancelled
        // cache; the inherited mock no-op must not leave that owner live.
        if self.evidence.prefix.is_none() {
            return;
        }
        let lifecycle = self.token_policy_lifecycle.lock();
        if lifecycle.executing != 0 {
            return;
        }
        let request = self
            .session_bindings
            .lock()
            .iter()
            .find(|request| format!("mock_{request}") == cache_id)
            .cloned();
        let Some(request) = request else {
            return;
        };
        // Ordinary cache-reference updates cannot retire an admitted owner.
        // Completion/cancellation first retires its exact admission as usual.
        if lifecycle.admitted.contains(&request) {
            return;
        }
        self.native_structured_history.lock().remove(&request);
        if self.recycle_completed_bindings.load(Ordering::Acquire) {
            self.completed_bindings.lock().insert(request);
        }
    }
    async fn complete_cache(&self, completion: ExecutorSequenceCompletion) -> Result<()> {
        let _work = self.token_policy_work();
        self.completion_calls.fetch_add(1, Ordering::AcqRel);
        self.release_cache(completion.cache_id());
        if self.completion_fail.load(Ordering::Acquire) {
            Err(FerrumError::backend("controlled completion failure"))
        } else {
            self.native_structured_history
                .lock()
                .remove(completion.request_id());
            if self.recycle_completed_bindings.load(Ordering::Acquire) {
                self.completed_bindings
                    .lock()
                    .insert(completion.request_id().clone());
            }
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
        let _work = self.token_policy_work();
        self.base.prefill(input).await
    }
    async fn decode(&self, input: &DecodeInput) -> Result<DecodeOutput> {
        let _work = self.token_policy_work();
        self.base.decode(input).await
    }
    async fn plan_runtime_prefill_with_capacity(
        &self,
        input: &PlanRuntimePrefillInput,
    ) -> Result<PlanRuntimePrefillOutcome> {
        let _work = self.token_policy_work();
        if self.plain_wave_protocol.load(Ordering::Acquire) {
            if let Some(deferral) = self.deferrals.take_plain(&self.entries) {
                return Ok(PlanRuntimePrefillOutcome::Deferred(deferral));
            }
            self.record_plain_wave(std::slice::from_ref(input), &[]);
        }
        self.prefill_output(input)
            .map(PlanRuntimePrefillOutcome::Completed)
    }
    async fn plan_runtime_batch_prefill_with_capacity(
        &self,
        inputs: &[PlanRuntimePrefillInput],
    ) -> Result<PlanRuntimeBatchPrefillOutcome> {
        let _work = self.token_policy_work();
        if !self.plain_wave_protocol.load(Ordering::Acquire) {
            return Ok(PlanRuntimeBatchPrefillOutcome::Unsupported);
        }
        if let Some(deferral) = self.deferrals.take_plain(&self.entries) {
            return Ok(PlanRuntimeBatchPrefillOutcome::NotSubmitted(deferral));
        }
        self.record_plain_wave(inputs, &[]);
        inputs
            .iter()
            .map(|input| self.prefill_output(input))
            .collect::<Result<Vec<_>>>()
            .map(PlanRuntimeBatchPrefillOutcome::Completed)
    }
    async fn plan_runtime_batch_decode_with_capacity(
        &self,
        inputs: &[PlanRuntimeDecodeInput],
    ) -> Result<PlanRuntimeBatchDecodeOutcome> {
        let _work = self.token_policy_work();
        if !self.plain_wave_protocol.load(Ordering::Acquire) {
            return Err(FerrumError::unsupported(
                "tensor-free plan-runtime batch decode is not implemented",
            ));
        }
        if let Some(deferral) = self.deferrals.take_plain(&self.entries) {
            return Ok(PlanRuntimeBatchDecodeOutcome::Deferred(deferral));
        }
        self.record_plain_wave(&[], inputs);
        Ok(PlanRuntimeBatchDecodeOutcome::Completed(
            self.decode_outputs(inputs),
        ))
    }
    async fn plan_runtime_batch_prefill_guarded_work_observed(
        &self,
        inputs: &[PlanRuntimePrefillInput],
        _: &ExpectedExecutionWave,
        guard: &dyn NonblockingHostSubmissionGuard,
        mut observation: GuardedCostObservation<'_, '_>,
    ) -> GuardedDispatchOutcome<Vec<PlanRuntimePrefillCompletion>> {
        let _work = self.token_policy_work();
        self.deferrals.before_prefill(inputs);
        if let Some(deferred) =
            self.deferrals
                .take_observed(guard, &self.entries, observation.as_deref_mut())
        {
            return deferred;
        }
        if !self.enter_guarded(guard).await {
            return GuardedDispatchOutcome::ReplanBeforeEncode;
        }
        self.submitted_requests.lock().push(
            inputs
                .iter()
                .map(|input| input.request_id.clone())
                .collect(),
        );
        let native = self.native_structured_output(observation.as_deref_mut(), inputs, &[], guard);
        if native.is_none() {
            self.begin_cost_observation(observation.as_deref_mut(), inputs, &[]);
        }
        let result = if self.fail_after_submit.load(Ordering::Acquire) {
            Err(FerrumError::backend("injected submitted prefill failure"))
        } else {
            if let Some(native) = native {
                native.and_then(|rows| {
                    inputs
                        .iter()
                        .zip(rows)
                        .map(|(input, logits)| self.prefill_output_from_logits(input, logits))
                        .collect()
                })
            } else {
                inputs
                    .iter()
                    .map(|input| self.prefill_output(input))
                    .collect()
            }
        };
        self.end_cost_observation(observation, result.is_ok());
        if result.is_ok() {
            self.wait_after_submit().await;
        }
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
        let _work = self.token_policy_work();
        if let Some(deferred) =
            self.deferrals
                .take_observed(guard, &self.entries, observation.as_deref_mut())
        {
            return deferred;
        }
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
        if result.is_ok() {
            self.wait_after_submit().await;
        }
        GuardedDispatchOutcome::Submitted(result)
    }
    async fn plan_runtime_batch_decode_guarded_work_observed(
        &self,
        inputs: &[PlanRuntimeDecodeInput],
        _: &ExpectedExecutionWave,
        guard: &dyn NonblockingHostSubmissionGuard,
        mut observation: GuardedCostObservation<'_, '_>,
    ) -> GuardedDispatchOutcome<Vec<PlanRuntimeDecodeOutput>> {
        let _work = self.token_policy_work();
        if let Some(deferred) =
            self.deferrals
                .take_observed(guard, &self.entries, observation.as_deref_mut())
        {
            return deferred;
        }
        if !self.enter_guarded(guard).await {
            return GuardedDispatchOutcome::ReplanBeforeEncode;
        }
        self.submitted_requests.lock().push(
            inputs
                .iter()
                .map(|input| input.request_id.clone())
                .collect(),
        );
        let native = self.native_structured_output(observation.as_deref_mut(), &[], inputs, guard);
        if native.is_none() {
            self.begin_cost_observation(observation.as_deref_mut(), &[], inputs);
        }
        if self.fail_after_submit.load(Ordering::Acquire) {
            self.end_cost_observation(observation, false);
            return GuardedDispatchOutcome::Submitted(Err(FerrumError::backend(
                "injected submitted failure",
            )));
        }
        let result = if let Some(native) = native {
            native.map(|rows| {
                inputs
                    .iter()
                    .zip(rows)
                    .map(|(input, logits)| self.decode_output_from_logits(input, logits))
                    .collect()
            })
        } else {
            Ok(self.decode_outputs(inputs))
        };
        self.end_cost_observation(observation, result.is_ok());
        if result.is_ok() {
            self.wait_after_submit().await;
        }
        GuardedDispatchOutcome::Submitted(result)
    }
}

impl ControlledExecutor {
    async fn wait_after_submit(&self) {
        if self.park_after_submit.load(Ordering::Acquire) {
            self.submitted.notify_one();
            self.resume_after_submit.notified().await;
        }
    }

    fn record_plain_wave(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
    ) {
        self.entries.fetch_add(1, Ordering::AcqRel);
        self.physical.fetch_add(1, Ordering::AcqRel);
        self.submitted_requests.lock().push(
            prefills
                .iter()
                .map(|input| input.request_id.clone())
                .chain(decodes.iter().map(|input| input.request_id.clone()))
                .collect(),
        );
    }
    pub(in crate::continuous_engine) fn abort_resource_sessions(&self) {
        for session in self.evidence.sessions.snapshot() {
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
        let unknown = self
            .actual_observation_unknown
            .lock()
            .as_ref()
            .and_then(|hook| hook(prefills, decodes));
        if let Some(reason) = unknown {
            let started_at = context.now_ns();
            context.physical_wave(Err(reason), started_at);
            return;
        }
        if self
            .emit_structured_cost_observations
            .load(Ordering::Acquire)
            && prefills
                .iter()
                .map(|r| &r.request_id)
                .chain(decodes.iter().map(|r| &r.request_id))
                .all(|id| {
                    context
                        .participant(id)
                        .and_then(|p| p.host_features)
                        .is_some_and(|host| host.supports_installed_plain_text_content())
                })
        {
            structured::record(context, prefills, decodes, None);
            return;
        }
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
    pub(in crate::continuous_engine::inner) fn enable_structured_query_route(&self) {
        let fixture = self.evidence.fixture.as_ref().unwrap();
        let mut trace = fixture.provider_trace.lock().unwrap();
        trace.cost_route_supported = true;
        trace.cost_route_statistics = true;
        let mut runtime = fixture.runtime_trace.lock().unwrap();
        runtime.structured_cost_capture_enabled = true;
        runtime.cost_route_projection_enabled = true;
    }

    pub(in crate::continuous_engine::inner) fn enable_cost_route_eager_boundary(&self) {
        let fixture = self.evidence.fixture.as_ref().unwrap();
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .cost_route_eager_boundary = true;
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .cost_direct_replay_projection_enabled = true;
    }

    pub(in crate::continuous_engine::inner) fn set_cost_graph_evidence(
        &self,
        state: Option<vnext::DeviceCostGraphStreamState>,
        catalog: Option<vnext::DeviceCostGraphCatalog>,
    ) {
        let fixture = self.evidence.fixture.as_ref().unwrap();
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.cost_graph_state = state;
        trace.cost_graph_catalog = catalog;
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
                None if self.evidence.prefix.is_some() => {
                    // A read-only projection cannot mint a new admission.
                    return ExecutionCostRouteAvailability::Unknown(
                        vnext::ExecutionCostRouteUnknown::InvalidInput,
                    );
                }
                None if bindings.len() < self.evidence.sessions.len() => {
                    bindings.push(request.request_id.clone());
                    bindings.len() - 1
                }
                None if self.recycle_completed_bindings.load(Ordering::Acquire) => {
                    let mut completed = self.completed_bindings.lock();
                    let caches = self.produced_caches.lock();
                    let free = bindings.iter().enumerate().find_map(|(index, old)| {
                        let cache_id = format!("mock_{old}");
                        (completed.contains(old)
                            && !indices.contains(&index)
                            && caches
                                .iter()
                                .filter_map(std::sync::Weak::upgrade)
                                .all(|cache| cache.cache_id() != cache_id))
                        .then_some(index)
                    });
                    let Some(index) = free else {
                        return ExecutionCostRouteAvailability::Unknown(
                            vnext::ExecutionCostRouteUnknown::InvalidInput,
                        );
                    };
                    completed.remove(&bindings[index]);
                    bindings[index] = request.request_id.clone();
                    index
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
        let owned_sessions = indices
            .iter()
            .map(|&index| self.evidence.sessions.get(index).unwrap())
            .collect::<Vec<_>>();
        let sessions = owned_sessions.iter().map(Arc::as_ref).collect::<Vec<_>>();
        let resources = &self.evidence.fixture.as_ref().unwrap().plan_resources;
        let initial_query = self
            .evidence
            .fixture
            .as_ref()
            .unwrap()
            .runtime_trace
            .lock()
            .unwrap()
            .cost_route_projection_enabled;
        let frontiers = if self.project_structured_cpu_fill.load(Ordering::Acquire)
            || self.native_structured_submission.load(Ordering::Acquire)
        {
            // Derive every current frontier from this actual CPU executor's
            // retained KV handle; never reuse the old initial-only constant.
            let caches = self.produced_caches.lock();
            let mut frontiers = Vec::with_capacity(requests.len());
            for request in requests {
                let frontier = match request.cache_id {
                    None if self.evidence.prefix.is_some() => {
                        // Production reads its admitted sequence's completed
                        // prefill frontier when the caller has no decode cache
                        // ID. Our real native completion publishes that same
                        // frontier in its retained output handle, even for an
                        // intermediate prefill. Input-history length is the
                        // full prompt and is not a completed-work frontier.
                        let cache_id = format!("mock_{}", request.request_id);
                        match caches
                            .iter()
                            .rev()
                            .filter_map(std::sync::Weak::upgrade)
                            .find(|cache| cache.cache_id() == cache_id)
                        {
                            Some(cache) => cache.num_tokens() as u64,
                            None if self
                                .native_structured_history
                                .lock()
                                .contains_key(request.request_id) =>
                            {
                                return ExecutionCostRouteAvailability::Unknown(
                                    vnext::ExecutionCostRouteUnknown::StaleView,
                                );
                            }
                            None => 0, // The admitted owner has no completed work.
                        }
                    }
                    None => 0,
                    Some(cache_id) => {
                        let Some(cache) = caches
                            .iter()
                            .rev()
                            .filter_map(std::sync::Weak::upgrade)
                            .find(|cache| cache.cache_id() == cache_id)
                        else {
                            return ExecutionCostRouteAvailability::Unknown(
                                vnext::ExecutionCostRouteUnknown::InvalidInput,
                            );
                        };
                        cache.num_tokens() as u64
                    }
                };
                frontiers.push(frontier);
            }
            frontiers
        } else if initial_query {
            // This opt-in projection fixture covers only genuine initial
            // prefills. It has no native ledger for post-submit KV frontiers.
            if self.physical.load(Ordering::Acquire) != 0
                || requests.iter().any(|request| request.cache_id.is_some())
            {
                return ExecutionCostRouteAvailability::Unknown(
                    vnext::ExecutionCostRouteUnknown::Unsupported,
                );
            }
            vec![0; sessions.len()]
        } else {
            vec![1; sessions.len()]
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        let route = loop {
            let route = resources.execution_cost_route_view(
                &sessions,
                &frontiers,
                self.evidence.lane.as_ref().unwrap().as_ref(),
                limits,
                budget,
            );
            match route {
                // Distinct fixture cleanup domains still share one status
                // registry mutex. Wait only for this unrelated try-read race,
                // before any executor entry; recapture the full real view.
                // Saturation, stale/unsupported evidence and all other read
                // failures remain Unknown. Never reset the caller's budget.
                ExecutionCostRouteAvailability::Unknown(
                    vnext::ExecutionCostRouteUnknown::Resource(
                        ResourcePlanningUnknown::ReadUnavailable(
                            vnext::ResourcePlanningReadStage::DeferredCleanup,
                        ),
                    ),
                ) if std::time::Instant::now() < deadline && budget.has_budget() => {
                    std::thread::yield_now();
                }
                // On either deadline keep the final actual Unknown; do not
                // substitute a cached view or turn contention into Known.
                other => break other,
            }
        };
        if self.evidence.prefix.is_some() {
            if let ExecutionCostRouteAvailability::Known(route) = &route {
                for ((request, participant), frontier) in requests
                    .iter()
                    .zip(route.resource_view().participants())
                    .zip(&frontiers)
                {
                    if request.cache_id.is_none()
                        && participant
                            .completed_checkpoint_boundary()
                            .map_or(0, |boundary| boundary.completed_tokens())
                            != *frontier
                    {
                        // An older output handle can outlive its replacement.
                        // It cannot override this capture's native completion.
                        return ExecutionCostRouteAvailability::Unknown(
                            vnext::ExecutionCostRouteUnknown::StaleView,
                        );
                    }
                }
            }
        }
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
        if !self.native_structured_submission.load(Ordering::Acquire) {
            self.physical.fetch_add(1, Ordering::AcqRel);
        }
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
    train_runtime(&runtime);
    runtime
}

pub(super) fn train_runtime(runtime: &Arc<EngineCostRuntime>) {
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
        assert_eq!(call.finish(), CostCallDisposition::Queued);
    }
    runtime.consume_samples();
    assert!(runtime.snapshot().is_some());
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
    startup_components_options(width, false).await
}

async fn startup_components_options(
    width: usize,
    checkpoint: bool,
) -> (Arc<dyn Tokenizer + Send + Sync>, Arc<ControlledExecutor>) {
    startup_components_with_generation_config(width, checkpoint, None).await
}

async fn startup_components_with_generation_config(
    width: usize,
    checkpoint: bool,
    generation_config: Option<&[u8]>,
) -> (Arc<dyn Tokenizer + Send + Sync>, Arc<ControlledExecutor>) {
    let vocab = (0..64)
        .map(|id| {
            (
                match id {
                    5 => "test".into(),
                    6 => "ok".into(),
                    10 if checkpoint => "a".into(),
                    11 if checkpoint => "Ã".into(),
                    12 if checkpoint => "©".into(),
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
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(match generation_config {
        Some(config) => HuggingFaceTokenizer::from_source_bytes(
            raw.to_string(false).unwrap().as_bytes(),
            None,
            Some(config),
        )
        .await
        .unwrap(),
        None => HuggingFaceTokenizer::new(raw).await.unwrap(),
    });
    let executor = Arc::new(ControlledExecutor {
        base: MockModelExecutor::instant(64),
        evidence: CoreEvidence::new_options(width, checkpoint),
        typed_sequence_state: Mutex::new(None),
        startup_capability_override: Mutex::new(None),
        profile_sink: Mutex::new(None),
        decode_context_coverage_override: Mutex::new(None),
        entries: AtomicUsize::new(0),
        physical: AtomicUsize::new(0),
        plain_wave_protocol: AtomicBool::new(false),
        prefill_granularity: AtomicUsize::new(1),
        park: AtomicBool::new(false),
        park_after_submit: AtomicBool::new(false),
        fail_after_submit: AtomicBool::new(false),
        panic_before_submit: AtomicBool::new(false),
        narrow_last_mixed_prefill: AtomicBool::new(false),
        replan_before_encode: AtomicBool::new(false),
        emit_cost_observations: AtomicBool::new(false),
        actual_observation_unknown: Mutex::new(None),
        emit_structured_cost_observations: AtomicBool::new(false),
        native_structured_submission: AtomicBool::new(false),
        native_prefix_preparation_outside: AtomicBool::new(false),
        native_prefix_preparation_outside_submissions: AtomicUsize::new(0),
        native_structured_history: Arc::new(Mutex::new(Default::default())),
        token_policy_lifecycle: Mutex::new(Default::default()),
        project_structured_cpu_fill: AtomicBool::new(false),
        single_row_prefill_only: AtomicBool::new(false),
        projection_readiness_fault: Mutex::new(None),
        context_partitioned_cpu_fill: AtomicBool::new(false),
        row_selected_cpu_fill: AtomicBool::new(false),
        completion_work_known: AtomicBool::new(false),
        completion_fail: AtomicBool::new(false),
        completion_calls: AtomicUsize::new(0),
        discarded_prefills: Mutex::new(Vec::new()),
        produced_caches: Mutex::new(Vec::new()),
        session_bindings: Mutex::new(Vec::new()),
        recycle_completed_bindings: AtomicBool::new(false),
        completed_bindings: Mutex::new(std::collections::HashSet::new()),
        submitted_requests: Mutex::new(Vec::new()),
        before_prefill_discard: Mutex::new(None),
        before_resource_revalidation: Mutex::new(None),
        after_resource_revalidation: Mutex::new(None),
        resource_planning_unknown: Mutex::new(None),
        cost_route_unknown: Mutex::new(None),
        resource_revalidation_changed: AtomicBool::new(false),
        deferrals: Default::default(),
        admission_capacity: Mutex::new(None),
        entered: Notify::new(),
        resume: Notify::new(),
        submitted: Notify::new(),
        resume_after_submit: Notify::new(),
    });
    (tokenizer, executor)
}

/// Real native CPU checkpoint executor. Callers still construct the ordinary
/// engine/runtime and must learn costs from its acknowledged transfer receipts.
pub(in crate::continuous_engine) async fn startup_checkpoint_components(
    width: usize,
) -> (Arc<dyn Tokenizer + Send + Sync>, Arc<ControlledExecutor>) {
    startup_checkpoint_components_with_generation_config(width, None).await
}

pub(in crate::continuous_engine) async fn startup_checkpoint_components_with_generation_config(
    width: usize,
    generation_config: Option<&[u8]>,
) -> (Arc<dyn Tokenizer + Send + Sync>, Arc<ControlledExecutor>) {
    let (tokenizer, executor) =
        startup_components_with_generation_config(width, true, generation_config).await;
    executor
        .native_structured_submission
        .store(true, Ordering::Release);
    executor
        .project_structured_cpu_fill
        .store(true, Ordering::Release);
    executor.enable_structured_query_route();
    (tokenizer, executor)
}

pub(in crate::continuous_engine::inner) async fn fixture_with_width(
    width: usize,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    fixture_with_custom_config(width, |_| {}).await
}

pub(in crate::continuous_engine::inner) async fn fixture_with_custom_config(
    width: usize,
    configure: impl FnOnce(&mut ferrum_types::EngineConfig),
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    fixture_with_custom_config_options(width, None, configure).await
}

/// Opt in to the existing native completed-boundary state program. Stateless
/// guard fixtures keep their original plan and cost identity by default.
pub(in crate::continuous_engine::inner) async fn fixture_with_checkpoint_config(
    width: usize,
    maximum_scheduled_tokens_per_wave: NonZeroU64,
    configure: impl FnOnce(&mut ferrum_types::EngineConfig),
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    fixture_with_custom_config_options(width, Some(maximum_scheduled_tokens_per_wave), configure)
        .await
}

/// Read the original bounded controller capture, rather than replacing a
/// request's dynamic executor feedback with a fixture-supplied private cap.
pub(in crate::continuous_engine::inner) fn assert_prefill_step_work_policy(
    engine: &ContinuousBatchEngine,
    maximum_wave_tokens: NonZeroUsize,
    per_row_step: NonZeroUsize,
) {
    let mut hint = ferrum_interfaces::BatchHint::simple(maximum_wave_tokens.get());
    hint.max_tokens = maximum_wave_tokens.get();
    let budget = ControllerBudget::new(slo_clock_now(), Duration::from_secs(1)).unwrap();
    let captured = engine
        .inner
        .capture_slo_controller_snapshot(&hint, budget)
        .expect("original admitted controller snapshot");
    assert!(!captured.snapshot.requests.is_empty());
    assert!(captured
        .snapshot
        .requests
        .iter()
        .all(|row| matches!(row.phase, RequestPhaseView::Prefill(_))));
    let policy = captured.snapshot.capabilities.work_policy;
    assert_eq!(
        policy.prefill_step_chunk,
        Some(u64::try_from(per_row_step.get()).unwrap())
    );
    assert_eq!(
        policy.maximum_wave_tokens,
        u64::try_from(maximum_wave_tokens.get()).unwrap()
    );
    let envelope = policy.for_ready_decoders(0);
    assert_eq!(envelope.maximum_prefill_chunk, policy.prefill_step_chunk);
    let mut usage =
        ferrum_scheduler::implementations::continuous::work_policy::WaveWorkUsage::default();
    let step = NonZeroU64::new(u64::try_from(per_row_step.get()).unwrap()).unwrap();
    assert!(envelope.include(&mut usage, Some(step)));
    assert!(envelope.include(&mut usage, Some(step)));
    assert!(!envelope.include(&mut usage, Some(step)));
    let oversized = NonZeroU64::new(step.get().checked_add(1).unwrap()).unwrap();
    assert!(!envelope.include(
        &mut ferrum_scheduler::implementations::continuous::work_policy::WaveWorkUsage::default(),
        Some(oversized),
    ));
}

async fn fixture_with_custom_config_options(
    width: usize,
    checkpoint_tokens: Option<NonZeroU64>,
    configure: impl FnOnce(&mut ferrum_types::EngineConfig),
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    let (tokenizer, executor) =
        startup_components_options(width, checkpoint_tokens.is_some()).await;
    let checkpoint_context = checkpoint_tokens.map(|tokens| {
        // This is the original logical request ceiling. The Mock model's
        // larger capability cannot confer additional resident authority.
        let context = NonZeroU32::new(
            executor
                .native_request_fit_tokens()
                .unwrap()
                .try_into()
                .unwrap(),
        )
        .unwrap();
        executor.declare_checkpoint_cost_domain(context, tokens);
        context
    });
    let mut config = ferrum_types::EngineConfig::default();
    if let Some(context) = checkpoint_context {
        config.runtime.max_model_len = Some(context.get() as usize);
    }
    // This fixture exercises mixed guards explicitly; the product default is Split.
    config.batching.prefill_decode_execution = ferrum_types::PrefillDecodeExecution::Mixed;
    config.scheduler.slo.output.max_queued_events_per_request = NonZeroUsize::new(2).unwrap();
    // Functional branch tests have no wall-clock performance threshold. The
    // dedicated virtual-clock tests exercise actual planning budget exhaustion.
    config.scheduler.slo.planner.max_planning_us = NonZeroU64::new(30_000_000).unwrap();
    configure(&mut config);
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
    completion_fixture_with_config(width, |_| {}).await
}

pub(super) async fn completion_fixture_with_config(
    width: usize,
    configure: impl FnOnce(&mut ferrum_types::EngineConfig),
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<ControlledExecutor>,
) {
    let (mut engine, scheduler, executor) = fixture_with_custom_config(width, configure).await;
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
