use super::*;
use ferrum_execution::executor_lifecycle::ExecutorOwner;
use ferrum_execution::token_execution::{
    ExecutionAdmission, ExecutionBatchOutcome, ExecutionCloseOutcome, ExecutionEndReason,
    ExecutionFrontier, ExecutionWork, ExecutorSequence, TokenAdmissionOutcome,
};
use ferrum_execution::{
    model_executor::{
        ExecutorAdmissionEpochs, ExecutorCapabilities, ExecutorCapacityWaitRegistration,
        ExecutorExecutionCapacityDeferral, ExecutorExecutionCapacityPreemption,
        ExecutorExecutionCapacityStage, ExecutorPrefillAdmission, ExecutorPrefillAdmissionDecision,
        ExecutorPrefillAdmissionReceipt, ExecutorPrefillMaintenanceBlocker,
        ExecutorPrefillMaintenanceDeferral, ExecutorPrefillMaintenanceOutcome,
        ExecutorPrefillMaintenanceStage, ExecutorSamplingOutput, ExecutorSequenceCompletion,
        ExecutorStatus, PlanRuntimeResourceSnapshot, PrefillChunk,
    },
    ModelExecutor,
};
use ferrum_interfaces::kv_cache::AllocationRequest;
use ferrum_interfaces::tokenizer::{TokenizerInfo, TokenizerType};
use ferrum_interfaces::{KvCacheManager, TensorRef};
use ferrum_models::DecoderOnlyLLM;
use ferrum_testkit::{MockKvCacheManager, MockModelExecutor, MockTensorFactory};
use ferrum_types::{DataType, Priority};
use std::time::Duration;

#[path = "controlled_token_fixture.rs"]
mod controlled_token_fixture;
use controlled_token_fixture::ControlledTokenState;

struct ControlledTokenObservation(std::sync::Weak<ControlledTokenState>);

impl ControlledTokenObservation {
    fn frontier_for_request(&self, request: &RequestId) -> Option<ExecutionFrontier> {
        self.0.upgrade()?.frontier_for_request(request)
    }
}

trait ObservedTestExecutor: ModelExecutor {
    type Observation;
    fn observer(&self) -> Arc<Self::Observation>;
    fn retire_fixture(&mut self) -> Result<()>;
}

fn begin_controlled_shutdown<E: ObservedTestExecutor>(
    executor: Arc<E>,
    permit: &ferrum_execution::executor_lifecycle::ExecutorShutdownPermit,
) -> ferrum_execution::executor_lifecycle::ExecutorShutdownObserver {
    use ferrum_execution::executor_lifecycle::{
        ExecutorShutdownObserver, ExecutorShutdownReporter,
    };
    if let Err(error) = permit.check(executor.as_ref()) {
        return ExecutorShutdownObserver::retained(Arc::new(executor), error);
    }
    let executor = match Arc::try_unwrap(executor) {
        Ok(executor) => executor,
        Err(original) => {
            return ExecutorShutdownObserver::retained(
                Arc::new(original),
                FerrumError::internal(
                    "controlled fixture still has an unscoped execution reference",
                ),
            )
        }
    };
    enum Original<E> {
        Live(E),
        Dropping,
        Retired,
    }
    fn advance<E: ObservedTestExecutor>(
        original: &Mutex<Original<E>>,
        reporter: &ExecutorShutdownReporter,
    ) {
        let mut original = original.lock();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            match &mut *original {
                Original::Live(executor) => executor.retire_fixture()?,
                Original::Dropping => {
                    return Err(FerrumError::internal(
                        "controlled fixture destruction unwound without a retirement receipt",
                    ))
                }
                Original::Retired => return Ok(()),
            }
            drop(std::mem::replace(&mut *original, Original::Dropping));
            *original = Original::Retired;
            Ok(())
        }));
        match result {
            Ok(Ok(())) => reporter.retired(None),
            Ok(Err(error)) => reporter.retained(error),
            Err(_) => reporter.retained(FerrumError::internal("controlled fixture close panicked")),
        }
    }
    let original = Arc::new(Mutex::new(Original::Live(executor)));
    let (reporter, observer) = ExecutorShutdownObserver::channel(original.clone());
    let weak = Arc::downgrade(&original);
    let retry_reporter = reporter.clone();
    let observer = observer.with_retry(Arc::new(move || {
        if let Some(original) = weak.upgrade() {
            advance(&original, &retry_reporter);
        }
    }));
    advance(&original, &reporter);
    observer
}

#[path = "adapter_admission_tests.rs"]
mod adapter_admission_tests;

#[path = "stop_boundary_tests.rs"]
mod stop_boundary_tests;

#[path = "prefix_rendezvous_tests.rs"]
mod prefix_rendezvous_tests;
#[path = "prefix_restore_deferred_tests.rs"]
mod prefix_restore_deferred_tests;
#[path = "prefix_restore_tests.rs"]
mod prefix_restore_tests;

#[path = "completion_decode_tests.rs"]
mod completion_decode_tests;
#[path = "managed_driver_tests.rs"]
mod managed_driver_tests;
#[path = "metrics_only_tests.rs"]
mod metrics_only_tests;
#[path = "mixed_batch_tests.rs"]
mod mixed_batch_tests;
#[path = "partial_batch_tests.rs"]
mod partial_batch_tests;
#[path = "prefix_prompt_tail_tests.rs"]
mod prefix_prompt_tail_tests;
#[path = "token_capacity_tests.rs"]
mod token_capacity_tests;

fn test_execution_capacity_deferral(
    observed: ExecutorAdmissionEpochs,
    wait_condition: ferrum_runtime::CapacityWaitCondition,
    stage: ExecutorExecutionCapacityStage,
) -> ExecutorExecutionCapacityDeferral {
    let pressure = ferrum_runtime::DeviceCapacityPressure::new(
        ferrum_runtime::DeviceCapacityPressureScope::PlanBudget,
        "device.test-execution-capacity".to_owned(),
        1,
        1,
        1,
        1,
        1,
    )
    .unwrap();
    ExecutorExecutionCapacityDeferral::from_backing_pressure(
        observed,
        wait_condition,
        pressure.into(),
        stage,
    )
    .unwrap()
}

struct PlanRuntimeAdmissionTestExecutor {
    tokens: Arc<ControlledTokenState>,
    inner: MockModelExecutor,
    maintenance_gated: bool,
    observation: Arc<PlanRuntimeAdmissionObservation>,
}

struct PlanRuntimeAdmissionObservation {
    tokens: ControlledTokenObservation,
    event_sink: std::sync::Mutex<Option<Arc<dyn ProductExecutionEventSink>>>,
    retained: std::sync::Mutex<HashSet<RequestId>>,
    admission_accounting: std::sync::Mutex<Vec<(RequestId, usize, usize, usize)>>,
    completions: std::sync::Mutex<Vec<ExecutorSequenceCompletion>>,
    available_admissions: AtomicU64,
    capacity_epoch: AtomicU64,
    admission_probes: AtomicU64,
    maintenance_calls: AtomicU64,
    prefill_calls: AtomicU64,
    batch_prefill_calls: AtomicU64,
    defer_batch_prefill_once: AtomicBool,
    fail_next_prefill_after_admission: AtomicBool,
}

impl std::ops::Deref for PlanRuntimeAdmissionTestExecutor {
    type Target = PlanRuntimeAdmissionObservation;
    fn deref(&self) -> &Self::Target {
        &self.observation
    }
}

impl ObservedTestExecutor for PlanRuntimeAdmissionTestExecutor {
    type Observation = PlanRuntimeAdmissionObservation;
    fn observer(&self) -> Arc<Self::Observation> {
        Arc::clone(&self.observation)
    }
    fn retire_fixture(&mut self) -> Result<()> {
        controlled_token_fixture::shutdown(self, &self.tokens)?;
        let pending = self
            .retained
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for request in pending {
            self.cancel_prefill_admission(&request);
        }
        drop(self.event_sink.lock().unwrap().take());
        Ok(())
    }
}

struct PlanRuntimeChunkedPrefillTestExecutor {
    tokens: Arc<ControlledTokenState>,
    inner: MockModelExecutor,
    prefix_restore: Option<Arc<prefix_restore_tests::RestoreState>>,
    prefix_retry: Option<Arc<prefix_restore_deferred_tests::RestoreState>>,
    rendezvous: Option<Arc<prefix_rendezvous_tests::CaptureState>>,
    prompt_tail_reserve: Option<usize>,
    observation: Arc<PlanRuntimeChunkedPrefillObservation>,
}

struct PlanRuntimeChunkedPrefillObservation {
    tokens: ControlledTokenObservation,
    retained: std::sync::Mutex<HashSet<RequestId>>,
    attempted_chunks: std::sync::Mutex<Vec<PrefillChunk>>,
    completed_chunks: AtomicU64,
    defer_next_prefill: AtomicBool,
    narrow_next_prefill: AtomicBool,
    release_epoch: Arc<AtomicU64>,
    capacity_wait_registrations: AtomicU64,
    capacity_signal: tokio::sync::watch::Sender<u64>,
    tail_batch_prefill_calls: AtomicU64,
}

impl std::ops::Deref for PlanRuntimeChunkedPrefillTestExecutor {
    type Target = PlanRuntimeChunkedPrefillObservation;
    fn deref(&self) -> &Self::Target {
        &self.observation
    }
}

impl ObservedTestExecutor for PlanRuntimeChunkedPrefillTestExecutor {
    type Observation = PlanRuntimeChunkedPrefillObservation;
    fn observer(&self) -> Arc<Self::Observation> {
        Arc::clone(&self.observation)
    }
    fn retire_fixture(&mut self) -> Result<()> {
        controlled_token_fixture::shutdown(self, &self.tokens)?;
        let pending = self
            .retained
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for request in pending {
            self.cancel_prefill_admission(&request);
        }
        Ok(())
    }
}

impl PlanRuntimeChunkedPrefillTestExecutor {
    const COORDINATOR_ID: u64 = 61;

    fn new(defer_first_prefill: bool) -> Self {
        let (capacity_signal, _) = tokio::sync::watch::channel(0);
        let tokens = Arc::new(ControlledTokenState::default());
        Self {
            tokens: Arc::clone(&tokens),
            inner: MockModelExecutor::instant(128),
            prefix_restore: None,
            prefix_retry: None,
            rendezvous: None,
            prompt_tail_reserve: None,
            observation: Arc::new(PlanRuntimeChunkedPrefillObservation {
                tokens: ControlledTokenObservation(Arc::downgrade(&tokens)),
                retained: std::sync::Mutex::new(HashSet::new()),
                attempted_chunks: std::sync::Mutex::new(Vec::new()),
                completed_chunks: AtomicU64::new(0),
                defer_next_prefill: AtomicBool::new(defer_first_prefill),
                narrow_next_prefill: AtomicBool::new(false),
                release_epoch: Arc::new(AtomicU64::new(0)),
                capacity_wait_registrations: AtomicU64::new(0),
                capacity_signal,
                tail_batch_prefill_calls: AtomicU64::new(0),
            }),
        }
    }

    fn new_with_partial_first_prefill() -> Self {
        let executor = Self::new(false);
        executor.narrow_next_prefill.store(true, Ordering::Release);
        executor
    }

    fn epochs(&self) -> ExecutorAdmissionEpochs {
        ExecutorAdmissionEpochs::new(
            std::num::NonZeroU64::new(Self::COORDINATOR_ID).unwrap(),
            self.release_epoch.load(Ordering::Acquire),
            0,
        )
    }

    fn availability(&self) -> ferrum_runtime::CapacityAvailabilityEpoch {
        ferrum_runtime::CapacityAvailabilityEpoch::new(
            ferrum_runtime::CapacityAvailabilitySource::ActiveSequenceSlots,
            self.release_epoch.load(Ordering::Acquire) + 1,
        )
        .unwrap()
    }

    fn wait_condition(&self) -> ferrum_runtime::CapacityWaitCondition {
        ferrum_runtime::CapacityWaitCondition::from_observation(
            Self::COORDINATOR_ID,
            vec![self.availability()],
        )
        .unwrap()
    }
}

#[derive(Clone, Copy)]
enum PlanRuntimeBatchDecodeBehavior {
    Exact,
    DeferUntilPeerCacheRelease,
    DeferThenPreemptionFails,
}

struct PlanRuntimeBatchDecodeTestExecutor {
    tokens: Arc<ControlledTokenState>,
    inner: MockModelExecutor,
    behavior: PlanRuntimeBatchDecodeBehavior,
    capacity_stage: ExecutorExecutionCapacityStage,
    capacity_stage_after_first_call: Option<ExecutorExecutionCapacityStage>,
    observation: Arc<PlanRuntimeBatchDecodeObservation>,
}

struct PlanRuntimeBatchDecodeObservation {
    tokens: ControlledTokenObservation,
    decode_calls: AtomicU64,
    batch_decode_calls: AtomicU64,
    released_cache_count: AtomicU64,
    mixed_behavior: std::sync::Mutex<Option<mixed_batch_tests::MixedBehavior>>,
    mixed_calls: AtomicU64,
}

impl std::ops::Deref for PlanRuntimeBatchDecodeTestExecutor {
    type Target = PlanRuntimeBatchDecodeObservation;
    fn deref(&self) -> &Self::Target {
        &self.observation
    }
}

impl ObservedTestExecutor for PlanRuntimeBatchDecodeTestExecutor {
    type Observation = PlanRuntimeBatchDecodeObservation;
    fn observer(&self) -> Arc<Self::Observation> {
        Arc::clone(&self.observation)
    }
    fn retire_fixture(&mut self) -> Result<()> {
        controlled_token_fixture::shutdown(self, &self.tokens)?;
        Ok(())
    }
}

impl PlanRuntimeBatchDecodeTestExecutor {
    fn with_capacity_stage_transition(
        behavior: PlanRuntimeBatchDecodeBehavior,
        capacity_stage: ExecutorExecutionCapacityStage,
        capacity_stage_after_first_call: Option<ExecutorExecutionCapacityStage>,
    ) -> Self {
        let tokens = Arc::new(ControlledTokenState::default());
        Self {
            tokens: Arc::clone(&tokens),
            inner: MockModelExecutor::instant(64),
            behavior,
            capacity_stage,
            capacity_stage_after_first_call,
            observation: Arc::new(PlanRuntimeBatchDecodeObservation {
                tokens: ControlledTokenObservation(Arc::downgrade(&tokens)),
                decode_calls: AtomicU64::new(0),
                batch_decode_calls: AtomicU64::new(0),
                released_cache_count: AtomicU64::new(0),
                mixed_behavior: std::sync::Mutex::new(None),
                mixed_calls: AtomicU64::new(0),
            }),
        }
    }

    fn capacity_stage_for_call(&self, call_index: u64) -> ExecutorExecutionCapacityStage {
        if call_index > 0 {
            self.capacity_stage_after_first_call
                .unwrap_or(self.capacity_stage)
        } else {
            self.capacity_stage
        }
    }
}

#[async_trait::async_trait]
impl ModelExecutor for PlanRuntimeChunkedPrefillTestExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::executor_lifecycle::ExecutorShutdownPermit,
    ) -> ferrum_execution::executor_lifecycle::ExecutorShutdownObserver {
        begin_controlled_shutdown(self, permit)
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        controlled_token_fixture::admit(self, &self.tokens, input)
    }
    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        controlled_token_fixture::execute(self, &self.tokens, work).await
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        controlled_token_fixture::finish(self, &self.tokens, sequence, reason).await
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        controlled_token_fixture::cancel(self, &self.tokens, sequence)
    }
    fn write_token_capacity_release_sources(
        &self,
        sequence: &ExecutorSequence,
        sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        controlled_token_fixture::release_sources(self, &self.tokens, sequence, sources)
    }
    fn plan_prompt_tail_capture_boundary(
        &self,
        chunk: PrefillChunk,
    ) -> Option<ferrum_execution::model_executor::PrefixCapturePlan> {
        let boundary = chunk
            .total_prompt_tokens()
            .checked_sub(self.prompt_tail_reserve?)?;
        if boundary <= chunk.tokens_processed() || boundary > chunk.end() {
            return None;
        }
        Some(ferrum_execution::model_executor::PrefixCapturePlan {
            boundary,
            span: ferrum_runtime::CheckpointTokenSpanConstraint::any_positive(),
        })
    }

    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }

    fn supports_plan_runtime_prefix_restore(&self) -> bool {
        self.prefix_restore.is_some() || self.prefix_retry.is_some() || self.rendezvous.is_some()
    }

    fn plan_prefix_capture_boundary(
        &self,
        input: ferrum_execution::model_executor::PrefixCaptureBoundary<'_>,
    ) -> Option<ferrum_execution::model_executor::PrefixCapturePlan> {
        self.rendezvous.as_ref()?.boundary(input)
    }

    fn retain_prefix_capture_interest(
        &self,
        input: ferrum_execution::model_executor::PrefixCaptureRequest<'_>,
    ) -> Result<Option<Arc<dyn ferrum_execution::model_executor::PrefixCaptureLease>>> {
        assert!(self
            .retained
            .lock()
            .unwrap()
            .contains(input.source_request_id));
        Ok(self.rendezvous.as_ref().and_then(|state| {
            state.arm(
                input,
                Arc::clone(&self.release_epoch),
                self.capacity_signal.clone(),
            )
        }))
    }

    async fn try_restore_plan_runtime_prefix(
        &self,
        input: ferrum_execution::model_executor::PlanRuntimePrefixRestoreInput<'_>,
    ) -> Result<ferrum_execution::model_executor::PlanRuntimePrefixRestoreOutcome> {
        assert!(self.retained.lock().unwrap().contains(input.request_id));
        if let Some(state) = &self.prefix_retry {
            return state.restore(
                input,
                self.tokens.clone(),
                self.epochs(),
                self.wait_condition(),
            );
        }
        if let Some(state) = &self.rendezvous {
            return state.restore(input, self.tokens.clone());
        }
        self.prefix_restore
            .as_ref()
            .unwrap()
            .restore(input, self.tokens.clone())
    }

    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        PlanRuntimeResourceSnapshot::new(1_000, 900, 700, 700, 400, 300, 200, 0, 0).map(Some)
    }

    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        Ok(Some(self.epochs()))
    }

    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<ferrum_runtime::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        availability.clear();
        availability.push(self.availability());
        Ok(Some(self.epochs()))
    }

    fn write_execution_capacity_release_sources(
        &self,
        preemption: &ExecutorExecutionCapacityPreemption,
        sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        sources.clear();
        if self
            .prefix_retry
            .as_ref()
            .is_some_and(|state| state.release_capable(preemption.request_id()))
        {
            sources.push(ferrum_runtime::CapacityAvailabilitySource::ActiveSequenceSlots);
            return Ok(true);
        }
        Ok(false)
    }

    fn register_execution_capacity_waiter(
        &self,
        observed: &ferrum_runtime::CapacityWaitCondition,
    ) -> Result<Option<ExecutorCapacityWaitRegistration>> {
        if observed.coordinator_id().get() != Self::COORDINATOR_ID {
            return Err(FerrumError::internal(
                "chunked prefill test received a foreign coordinator",
            ));
        }
        let observed_epoch = observed
            .observed()
            .first()
            .ok_or_else(|| FerrumError::internal("chunked prefill wait has no source"))?
            .epoch();
        let mut signal = self.capacity_signal.subscribe();
        self.capacity_wait_registrations
            .fetch_add(1, Ordering::Relaxed);
        Ok(Some(ExecutorCapacityWaitRegistration::new(async move {
            loop {
                let release_epoch = *signal.borrow_and_update();
                if release_epoch + 1 > observed_epoch {
                    return Ok(ExecutorAdmissionEpochs::new(
                        std::num::NonZeroU64::new(Self::COORDINATOR_ID).unwrap(),
                        release_epoch,
                        0,
                    ));
                }
                signal.changed().await.map_err(|_| {
                    FerrumError::internal("chunked prefill capacity signal closed while waiting")
                })?;
            }
        })))
    }

    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        let inserted = self
            .retained
            .lock()
            .expect("chunked prefill retained mutex poisoned")
            .insert(input.request_id.clone());
        if !inserted {
            return Err(FerrumError::already_exists(
                "duplicate chunked prefill test admission",
            ));
        }
        if let Some(state) = &self.rendezvous {
            if state.defer_admission_once() {
                return Ok(ExecutorPrefillAdmissionDecision::MaintenanceDeferred(
                    ExecutorPrefillMaintenanceDeferral::new(
                        input.request_id.clone(),
                        self.epochs(),
                        self.wait_condition(),
                        ExecutorPrefillMaintenanceStage::PhysicalBacking,
                        vec![ExecutorPrefillMaintenanceBlocker::Capacity {
                            domain_id: Some(1),
                            kind: ferrum_runtime::CapacityShortfallKind::BackingGrowthRequired,
                            requested: 4096,
                            available: 0,
                            current_total: 0,
                            maximum_total: 8192,
                        }],
                    )?,
                ));
            }
            state.admitted(input.request_id);
        }
        Ok(ExecutorPrefillAdmissionDecision::Admitted(
            ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            },
        ))
    }

    fn cancel_prefill_admission(&self, request_id: &RequestId) -> bool {
        self.retained
            .lock()
            .expect("chunked prefill retained mutex poisoned")
            .remove(request_id)
    }

    fn maintain_prefill_backing(
        &self,
        request_id: &RequestId,
    ) -> Result<ExecutorPrefillMaintenanceOutcome> {
        let state = self
            .rendezvous
            .as_ref()
            .ok_or_else(|| FerrumError::unsupported("no fixture maintenance"))?;
        assert!(self.retained.lock().unwrap().remove(request_id));
        state.maintained(request_id);
        if state.wait_for_release.load(Ordering::Acquire) && state.has_ready_capture() {
            return Ok(ExecutorPrefillMaintenanceOutcome::WaitForRelease {
                current: self.epochs(),
                wait_condition: self.wait_condition(),
                pressure: ferrum_runtime::DeviceCapacityPressure::new(
                    ferrum_runtime::DeviceCapacityPressureScope::PlanBudget,
                    "device.prefix-fixture".to_owned(),
                    4096,
                    4096,
                    4096,
                    4096,
                    4096,
                )
                .unwrap()
                .into(),
            });
        }
        self.publish_release();
        Ok(ExecutorPrefillMaintenanceOutcome::RetryAdmission {
            current: self.epochs(),
        })
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for PlanRuntimeBatchDecodeTestExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::executor_lifecycle::ExecutorShutdownPermit,
    ) -> ferrum_execution::executor_lifecycle::ExecutorShutdownObserver {
        begin_controlled_shutdown(self, permit)
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        controlled_token_fixture::admit(self, &self.tokens, input)
    }
    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        controlled_token_fixture::execute(self, &self.tokens, work).await
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        controlled_token_fixture::finish(self, &self.tokens, sequence, reason).await
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        if matches!(
            self.behavior,
            PlanRuntimeBatchDecodeBehavior::DeferThenPreemptionFails
        ) {
            return Err(FerrumError::internal(
                "synthetic request-scoped close failure",
            ));
        }
        controlled_token_fixture::cancel(self, &self.tokens, sequence)
    }
    fn write_token_capacity_release_sources(
        &self,
        sequence: &ExecutorSequence,
        sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        controlled_token_fixture::release_sources(self, &self.tokens, sequence, sources)
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

    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }

    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        PlanRuntimeResourceSnapshot::new(1_000, 900, 700, 700, 400, 300, 200, 0, 0).map(Some)
    }

    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        Ok(Some(ExecutorAdmissionEpochs::new(
            std::num::NonZeroU64::new(47).unwrap(),
            self.released_cache_count.load(Ordering::Acquire),
            0,
        )))
    }

    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<ferrum_runtime::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        availability.clear();
        availability.push(
            ferrum_runtime::CapacityAvailabilityEpoch::new(
                ferrum_runtime::CapacityAvailabilitySource::ActiveSequenceSlots,
                self.released_cache_count.load(Ordering::Acquire) + 1,
            )
            .map_err(|error| FerrumError::internal(error.to_string()))?,
        );
        self.execution_capacity_epochs()
    }

    fn write_execution_capacity_release_sources(
        &self,
        _preemption: &ExecutorExecutionCapacityPreemption,
        sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        sources.clear();
        sources.push(ferrum_runtime::CapacityAvailabilitySource::ActiveSequenceSlots);
        Ok(true)
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

impl PlanRuntimeAdmissionTestExecutor {
    fn new(vocab_size: usize) -> Self {
        let tokens = Arc::new(ControlledTokenState::default());
        Self {
            tokens: Arc::clone(&tokens),
            inner: MockModelExecutor::instant(vocab_size),
            maintenance_gated: false,
            observation: Arc::new(PlanRuntimeAdmissionObservation {
                tokens: ControlledTokenObservation(Arc::downgrade(&tokens)),
                event_sink: std::sync::Mutex::new(None),
                retained: std::sync::Mutex::new(HashSet::new()),
                admission_accounting: std::sync::Mutex::new(Vec::new()),
                completions: std::sync::Mutex::new(Vec::new()),
                available_admissions: AtomicU64::new(0),
                capacity_epoch: AtomicU64::new(0),
                admission_probes: AtomicU64::new(0),
                maintenance_calls: AtomicU64::new(0),
                prefill_calls: AtomicU64::new(0),
                batch_prefill_calls: AtomicU64::new(0),
                defer_batch_prefill_once: AtomicBool::new(false),
                fail_next_prefill_after_admission: AtomicBool::new(false),
            }),
        }
    }

    fn new_with_resource_prefill_failure(vocab_size: usize) -> Self {
        let executor = Self::new(vocab_size);
        executor
            .fail_next_prefill_after_admission
            .store(true, Ordering::Release);
        executor
    }

    fn new_with_serial_backing_growth(vocab_size: usize) -> Self {
        Self {
            maintenance_gated: true,
            ..Self::new(vocab_size)
        }
    }

    fn new_with_prefill_latency(vocab_size: usize, latency: Duration) -> Self {
        Self {
            inner: MockModelExecutor::new(vocab_size, latency, Duration::ZERO),
            ..Self::new(vocab_size)
        }
    }

    fn new_with_batch_prefill_deferral(vocab_size: usize) -> Self {
        let executor = Self::new(vocab_size);
        executor
            .defer_batch_prefill_once
            .store(true, Ordering::Release);
        executor
    }

    fn capacity_epochs(&self) -> ExecutorAdmissionEpochs {
        ExecutorAdmissionEpochs::new(
            std::num::NonZeroU64::new(41).unwrap(),
            0,
            self.capacity_epoch.load(Ordering::Acquire),
        )
    }

    fn capacity_wait_condition(&self) -> ferrum_runtime::CapacityWaitCondition {
        ferrum_runtime::CapacityWaitCondition::from_observation(
            41,
            vec![ferrum_runtime::CapacityAvailabilityEpoch::new(
                ferrum_runtime::CapacityAvailabilitySource::Domain(
                    ferrum_runtime::CapacityDomainId::new(1).unwrap(),
                ),
                self.capacity_epoch.load(Ordering::Acquire) + 1,
            )
            .unwrap()],
        )
        .unwrap()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for PlanRuntimeAdmissionTestExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::executor_lifecycle::ExecutorShutdownPermit,
    ) -> ferrum_execution::executor_lifecycle::ExecutorShutdownObserver {
        begin_controlled_shutdown(self, permit)
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        controlled_token_fixture::admit(self, &self.tokens, input)
    }
    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        controlled_token_fixture::execute(self, &self.tokens, work).await
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        controlled_token_fixture::finish(self, &self.tokens, sequence, reason).await
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        controlled_token_fixture::cancel(self, &self.tokens, sequence)
    }
    fn write_token_capacity_release_sources(
        &self,
        sequence: &ExecutorSequence,
        sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        controlled_token_fixture::release_sources(self, &self.tokens, sequence, sources)
    }
    fn attach_execution_event_sink(&self, sink: Arc<dyn ProductExecutionEventSink>) {
        *self.event_sink.lock().unwrap() = Some(sink);
    }

    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }

    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        PlanRuntimeResourceSnapshot::new(1_000, 900, 700, 700, 400, 300, 200, 0, 0).map(Some)
    }

    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        Ok(Some(self.capacity_epochs()))
    }

    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<ferrum_runtime::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        availability.clear();
        availability.extend_from_slice(self.capacity_wait_condition().observed());
        Ok(Some(self.capacity_epochs()))
    }

    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        input.validate()?;
        self.admission_probes.fetch_add(1, Ordering::Relaxed);
        self.admission_accounting
            .lock()
            .expect("admission accounting mutex poisoned")
            .push((
                input.request_id.clone(),
                input.input_tokens.len(),
                input.product_prompt_tokens,
                input.replayed_output_tokens,
            ));
        let inserted = self
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .insert(input.request_id.clone());
        if !inserted {
            return Err(FerrumError::already_exists("duplicate test admission"));
        }
        if self.maintenance_gated
            && self
                .available_admissions
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |available| {
                    (available > 0).then(|| available - 1)
                })
                .is_err()
        {
            return Ok(ExecutorPrefillAdmissionDecision::MaintenanceDeferred(
                ExecutorPrefillMaintenanceDeferral::new(
                    input.request_id.clone(),
                    self.capacity_epochs(),
                    self.capacity_wait_condition(),
                    ExecutorPrefillMaintenanceStage::PhysicalBacking,
                    vec![ExecutorPrefillMaintenanceBlocker::Capacity {
                        domain_id: Some(1),
                        kind: ferrum_runtime::CapacityShortfallKind::BackingGrowthRequired,
                        requested: 4096,
                        available: 0,
                        current_total: 0,
                        maximum_total: 8192,
                    }],
                )?,
            ));
        }
        Ok(ExecutorPrefillAdmissionDecision::Admitted(
            ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            },
        ))
    }

    fn cancel_prefill_admission(&self, request_id: &RequestId) -> bool {
        self.retained
            .lock()
            .expect("retained admission mutex poisoned")
            .remove(request_id)
    }

    fn maintain_prefill_backing(
        &self,
        request_id: &RequestId,
    ) -> Result<ExecutorPrefillMaintenanceOutcome> {
        if !self
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .remove(request_id)
        {
            return Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending);
        }
        self.maintenance_calls.fetch_add(1, Ordering::Relaxed);
        if self
            .available_admissions
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.capacity_epoch.fetch_add(1, Ordering::AcqRel);
            Ok(ExecutorPrefillMaintenanceOutcome::Maintained {
                current: self.capacity_epochs(),
                pools_grown: 1,
                allocated_bytes: 4096,
                pools_reclaimed: 0,
                chunks_reclaimed: 0,
                reclaimed_bytes: 0,
                rebalance: None,
            })
        } else {
            Ok(ExecutorPrefillMaintenanceOutcome::RetryAdmission {
                current: self.capacity_epochs(),
            })
        }
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TestMaintenanceBehavior {
    Advance,
    WaitForRelease,
    RetryAdmission,
    IncoherentRebalance,
    Fail,
}

struct PlanRuntimeMaintenanceTestExecutor {
    tokens: Arc<ControlledTokenState>,
    inner: MockModelExecutor,
    maintenance_behavior: TestMaintenanceBehavior,
    observation: Arc<PlanRuntimeMaintenanceObservation>,
}

struct PlanRuntimeMaintenanceObservation {
    tokens: ControlledTokenObservation,
    retained: std::sync::Mutex<HashSet<RequestId>>,
    release_epoch: AtomicU64,
    capacity_epoch: AtomicU64,
    admission_probes: AtomicU64,
    maintenance_calls: AtomicU64,
    capacity_wait_registrations: AtomicU64,
    prefill_calls: AtomicU64,
    call_order: std::sync::Mutex<Vec<&'static str>>,
    capacity_signal: tokio::sync::watch::Sender<u64>,
}

impl std::ops::Deref for PlanRuntimeMaintenanceTestExecutor {
    type Target = PlanRuntimeMaintenanceObservation;
    fn deref(&self) -> &Self::Target {
        &self.observation
    }
}

impl ObservedTestExecutor for PlanRuntimeMaintenanceTestExecutor {
    type Observation = PlanRuntimeMaintenanceObservation;
    fn observer(&self) -> Arc<Self::Observation> {
        Arc::clone(&self.observation)
    }
    fn retire_fixture(&mut self) -> Result<()> {
        controlled_token_fixture::shutdown(self, &self.tokens)?;
        let pending = self
            .retained
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for request in pending {
            self.cancel_prefill_admission(&request);
        }
        Ok(())
    }
}

impl PlanRuntimeMaintenanceTestExecutor {
    fn new(vocab_size: usize) -> Self {
        Self::with_behavior(vocab_size, TestMaintenanceBehavior::Advance)
    }

    fn with_behavior(vocab_size: usize, maintenance_behavior: TestMaintenanceBehavior) -> Self {
        let (capacity_signal, _) = tokio::sync::watch::channel(0);
        let tokens = Arc::new(ControlledTokenState::default());
        Self {
            tokens: Arc::clone(&tokens),
            inner: MockModelExecutor::instant(vocab_size),
            maintenance_behavior,
            observation: Arc::new(PlanRuntimeMaintenanceObservation {
                tokens: ControlledTokenObservation(Arc::downgrade(&tokens)),
                retained: std::sync::Mutex::new(HashSet::new()),
                release_epoch: AtomicU64::new(0),
                capacity_epoch: AtomicU64::new(0),
                admission_probes: AtomicU64::new(0),
                maintenance_calls: AtomicU64::new(0),
                capacity_wait_registrations: AtomicU64::new(0),
                prefill_calls: AtomicU64::new(0),
                call_order: std::sync::Mutex::new(Vec::new()),
                capacity_signal,
            }),
        }
    }

    fn epochs(&self) -> ExecutorAdmissionEpochs {
        ExecutorAdmissionEpochs::new(
            std::num::NonZeroU64::new(43).unwrap(),
            self.release_epoch.load(Ordering::Acquire),
            self.capacity_epoch.load(Ordering::Acquire),
        )
    }

    fn wait_condition(&self) -> ferrum_runtime::CapacityWaitCondition {
        ferrum_runtime::CapacityWaitCondition::from_observation(
            43,
            vec![ferrum_runtime::CapacityAvailabilityEpoch::new(
                ferrum_runtime::CapacityAvailabilitySource::Domain(
                    ferrum_runtime::CapacityDomainId::new(1).unwrap(),
                ),
                self.release_epoch
                    .load(Ordering::Acquire)
                    .checked_add(self.capacity_epoch.load(Ordering::Acquire))
                    .and_then(|epoch| epoch.checked_add(1))
                    .unwrap(),
            )
            .unwrap()],
        )
        .unwrap()
    }
}

#[async_trait::async_trait]
impl ModelExecutor for PlanRuntimeMaintenanceTestExecutor {
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ferrum_execution::executor_lifecycle::ExecutorShutdownPermit,
    ) -> ferrum_execution::executor_lifecycle::ExecutorShutdownObserver {
        begin_controlled_shutdown(self, permit)
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        controlled_token_fixture::admit(self, &self.tokens, input)
    }
    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        controlled_token_fixture::execute(self, &self.tokens, work).await
    }
    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        controlled_token_fixture::finish(self, &self.tokens, sequence, reason).await
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        controlled_token_fixture::cancel(self, &self.tokens, sequence)
    }
    fn write_token_capacity_release_sources(
        &self,
        sequence: &ExecutorSequence,
        sources: &mut Vec<ferrum_runtime::CapacityAvailabilitySource>,
    ) -> Result<bool> {
        controlled_token_fixture::release_sources(self, &self.tokens, sequence, sources)
    }
    fn info(&self) -> &ferrum_types::ModelInfo {
        self.inner.info()
    }

    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        ExecutionResourceAuthority::PlanRuntime
    }

    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        PlanRuntimeResourceSnapshot::new(1_000, 900, 700, 700, 400, 300, 200, 0, 0).map(Some)
    }

    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        Ok(Some(self.epochs()))
    }

    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<ferrum_runtime::CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        availability.clear();
        availability.extend_from_slice(self.wait_condition().observed());
        Ok(Some(self.epochs()))
    }

    fn register_execution_capacity_waiter(
        &self,
        observed: &ferrum_runtime::CapacityWaitCondition,
    ) -> Result<Option<ExecutorCapacityWaitRegistration>> {
        if observed.coordinator_id().get() != 43 {
            return Err(FerrumError::internal(
                "test capacity waiter received a foreign coordinator",
            ));
        }
        let expected_source = ferrum_runtime::CapacityAvailabilitySource::Domain(
            ferrum_runtime::CapacityDomainId::new(1).unwrap(),
        );
        if observed
            .observed()
            .iter()
            .any(|entry| entry.source() != expected_source)
        {
            return Err(FerrumError::internal(
                "test capacity waiter received an unknown source",
            ));
        }
        let observed_generation = observed
            .observed()
            .iter()
            .map(|entry| entry.epoch())
            .min()
            .expect("capacity wait conditions are non-empty");
        let capacity_epoch = self.capacity_epoch.load(Ordering::Acquire);
        let mut signal = self.capacity_signal.subscribe();
        self.capacity_wait_registrations
            .fetch_add(1, Ordering::Relaxed);
        Ok(Some(ExecutorCapacityWaitRegistration::new(async move {
            loop {
                let release_epoch = *signal.borrow_and_update();
                let current_generation = release_epoch
                    .checked_add(capacity_epoch)
                    .and_then(|generation| generation.checked_add(1))
                    .ok_or_else(|| FerrumError::internal("test capacity generation overflowed"))?;
                if current_generation > observed_generation {
                    return Ok(ExecutorAdmissionEpochs::new(
                        std::num::NonZeroU64::new(43).unwrap(),
                        release_epoch,
                        capacity_epoch,
                    ));
                }
                signal.changed().await.map_err(|_| {
                    FerrumError::internal("test capacity signal closed while waiting")
                })?;
            }
        })))
    }

    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        self.admission_probes.fetch_add(1, Ordering::Relaxed);
        let inserted = self
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .insert(input.request_id.clone());
        if !inserted {
            return Err(FerrumError::already_exists("duplicate test admission"));
        }
        if self.capacity_epoch.load(Ordering::Acquire) == 0
            && !(self.maintenance_behavior == TestMaintenanceBehavior::WaitForRelease
                && self.release_epoch.load(Ordering::Acquire) > 0)
            && !(self.maintenance_behavior == TestMaintenanceBehavior::RetryAdmission
                && self.maintenance_calls.load(Ordering::Acquire) > 0)
        {
            self.call_order
                .lock()
                .expect("call order mutex poisoned")
                .push("defer");
            return Ok(ExecutorPrefillAdmissionDecision::MaintenanceDeferred(
                ExecutorPrefillMaintenanceDeferral::new(
                    input.request_id.clone(),
                    self.epochs(),
                    self.wait_condition(),
                    ExecutorPrefillMaintenanceStage::LogicalCapacity,
                    vec![ExecutorPrefillMaintenanceBlocker::Capacity {
                        domain_id: Some(1),
                        kind: ferrum_runtime::CapacityShortfallKind::BackingGrowthRequired,
                        requested: 4096,
                        available: 0,
                        current_total: 0,
                        maximum_total: 4096,
                    }],
                )?,
            ));
        }
        self.call_order
            .lock()
            .expect("call order mutex poisoned")
            .push("admit");
        Ok(ExecutorPrefillAdmissionDecision::Admitted(
            ExecutorPrefillAdmissionReceipt {
                request_id: input.request_id.clone(),
            },
        ))
    }

    fn cancel_prefill_admission(&self, request_id: &RequestId) -> bool {
        self.retained
            .lock()
            .expect("retained admission mutex poisoned")
            .remove(request_id)
    }

    fn maintain_prefill_backing(
        &self,
        request_id: &RequestId,
    ) -> Result<ExecutorPrefillMaintenanceOutcome> {
        if !self
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .remove(request_id)
        {
            return Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending);
        }
        self.maintenance_calls.fetch_add(1, Ordering::Relaxed);
        self.call_order
            .lock()
            .expect("call order mutex poisoned")
            .push("maintain");
        match self.maintenance_behavior {
            TestMaintenanceBehavior::Advance => {
                self.capacity_epoch.store(1, Ordering::Release);
                Ok(ExecutorPrefillMaintenanceOutcome::Maintained {
                    current: self.epochs(),
                    pools_grown: 1,
                    allocated_bytes: 4096,
                    pools_reclaimed: 0,
                    chunks_reclaimed: 0,
                    reclaimed_bytes: 0,
                    rebalance: None,
                })
            }
            TestMaintenanceBehavior::WaitForRelease => {
                Ok(ExecutorPrefillMaintenanceOutcome::WaitForRelease {
                    current: self.epochs(),
                    wait_condition: self.wait_condition(),
                    pressure: ferrum_runtime::DeviceCapacityPressure::new(
                        ferrum_runtime::DeviceCapacityPressureScope::PlanBudget,
                        "device.test-capacity-pressure".to_owned(),
                        4096,
                        4096,
                        4096,
                        4096,
                        4096,
                    )
                    .unwrap()
                    .into(),
                })
            }
            TestMaintenanceBehavior::RetryAdmission => {
                Ok(ExecutorPrefillMaintenanceOutcome::RetryAdmission {
                    current: self.epochs(),
                })
            }
            TestMaintenanceBehavior::IncoherentRebalance => {
                self.capacity_epoch.store(1, Ordering::Release);
                Ok(ExecutorPrefillMaintenanceOutcome::Maintained {
                    current: self.epochs(),
                    pools_grown: 1,
                    allocated_bytes: 4096,
                    pools_reclaimed: 1,
                    chunks_reclaimed: 1,
                    reclaimed_bytes: 4096,
                    rebalance: None,
                })
            }
            TestMaintenanceBehavior::Fail => {
                Err(FerrumError::backend("synthetic backing allocation failure"))
            }
        }
    }

    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }

    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
}

struct PolicyTokenizer {
    vocab_size: usize,
    special: ferrum_types::SpecialTokens,
    ids: HashMap<String, TokenId>,
    texts: Vec<Option<String>>,
    encoded_sequences: HashMap<String, Vec<TokenId>>,
    hidden_decode_token_ids: HashSet<u32>,
}

impl PolicyTokenizer {
    fn new(vocab_size: usize, pairs: &[(&str, u32)]) -> Self {
        let max_id = pairs.iter().map(|(_, id)| *id as usize).max().unwrap_or(0);
        let mut texts = vec![None; max_id + 1];
        let mut ids = HashMap::new();
        for (text, id) in pairs {
            ids.insert((*text).to_string(), TokenId::new(*id));
            texts[*id as usize] = Some((*text).to_string());
        }
        Self {
            vocab_size,
            special: ferrum_types::SpecialTokens {
                bos_token: Some(TokenId::new(1)),
                eos_token: Some(TokenId::new(3)),
                unk_token: Some(TokenId::new(2)),
                pad_token: Some(TokenId::new(4)),
                sep_token: None,
                cls_token: None,
                mask_token: None,
                extra_eos_tokens: Vec::new(),
            },
            ids,
            texts,
            encoded_sequences: HashMap::new(),
            hidden_decode_token_ids: HashSet::new(),
        }
    }
}

impl Tokenizer for PolicyTokenizer {
    fn encode(&self, text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
        if let Some(tokens) = self.encoded_sequences.get(text) {
            return Ok(tokens.clone());
        }
        if let Some(id) = self.ids.get(text) {
            return Ok(vec![*id]);
        }
        let split_tokens = text
            .split_whitespace()
            .map(|part| self.ids.get(part).copied())
            .collect::<Option<Vec<_>>>();
        if let Some(tokens) = split_tokens.filter(|tokens| !tokens.is_empty()) {
            return Ok(tokens);
        }
        Ok(vec![TokenId::new(0)])
    }

    fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String> {
        let mut output = String::new();
        let mut pending_bad_byte = false;
        for token in tokens {
            if skip_special && self.hidden_decode_token_ids.contains(&token.get()) {
                continue;
            }
            let Some(text) = self.token_text(*token) else {
                continue;
            };
            if skip_special && matches!(text, "<think>" | "</think>") {
                continue;
            }
            match text {
                "byte-fallback" => output.push('\u{FFFD}'),
                "bad-byte-lead" => pending_bad_byte = true,
                "valid-byte-cont" if pending_bad_byte => {
                    output.push('好');
                    pending_bad_byte = false;
                }
                text => {
                    if pending_bad_byte {
                        output.push('\u{FFFD}');
                        pending_bad_byte = false;
                    }
                    output.push_str(text);
                }
            }
        }
        Ok(output)
    }

    fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
        Ok(self.token_text(next).unwrap_or_default().to_string())
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        &self.special
    }

    fn token_id(&self, text: &str) -> Option<TokenId> {
        self.ids.get(text).copied()
    }

    fn token_text(&self, token_id: TokenId) -> Option<&str> {
        self.texts
            .get(token_id.get() as usize)
            .and_then(|text| text.as_deref())
    }

    fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            tokenizer_type: TokenizerType::Custom,
            vocab_size: self.vocab_size,
            special_tokens: self.special.clone(),
            supports_incremental: true,
            supports_chat_template: false,
            max_token_length: None,
            model_name: Some("policy-tokenizer-test".to_string()),
        }
    }
}

struct FragmentedUtf8PolicyTokenizer {
    special: ferrum_types::SpecialTokens,
    texts: Vec<Option<String>>,
}

impl FragmentedUtf8PolicyTokenizer {
    const FIRE_HEAD: u32 = 128;
    const FIRE_TAIL: u32 = 129;
    const REPLACEMENT: u32 = 130;
    const EOS: u32 = 131;
    const BASE_VOCAB_SIZE: usize = 132;
    const MODEL_VOCAB_SIZE: usize = 133;

    fn new() -> Self {
        let mut texts = (0u8..=127)
            .map(|byte| Some((byte as char).to_string()))
            .collect::<Vec<_>>();
        texts.extend([
            Some("fire-head".to_string()),
            Some("fire-tail".to_string()),
            Some("\u{FFFD}".to_string()),
            Some("<eos>".to_string()),
        ]);
        Self {
            special: ferrum_types::SpecialTokens {
                eos_token: Some(TokenId::new(Self::EOS)),
                ..ferrum_types::SpecialTokens::default()
            },
            texts,
        }
    }

    fn raw_bytes(token: TokenId) -> Option<Vec<u8>> {
        match token.get() {
            byte @ 0..=127 => Some(vec![byte as u8]),
            Self::FIRE_HEAD => Some(vec![0xF0, 0x9F]),
            Self::FIRE_TAIL => Some(vec![0x94, 0xA5]),
            Self::REPLACEMENT => Some("\u{FFFD}".as_bytes().to_vec()),
            _ => None,
        }
    }
}

impl Tokenizer for FragmentedUtf8PolicyTokenizer {
    fn encode(&self, text: &str, _add_special: bool) -> Result<Vec<TokenId>> {
        let mut tokens = Vec::new();
        let mut bytes = text.as_bytes();
        while let Some((&byte, remaining)) = bytes.split_first() {
            if bytes.starts_with(&[0xF0, 0x9F, 0x94, 0xA5]) {
                tokens.push(TokenId::new(Self::FIRE_HEAD));
                tokens.push(TokenId::new(Self::FIRE_TAIL));
                bytes = &bytes[4..];
            } else if byte.is_ascii() {
                tokens.push(TokenId::new(byte as u32));
                bytes = remaining;
            } else {
                return Err(FerrumError::tokenizer(
                    "fragmented UTF-8 policy tokenizer received unsupported input",
                ));
            }
        }
        Ok(tokens)
    }

    fn decode(&self, tokens: &[TokenId], _skip_special: bool) -> Result<String> {
        let bytes = tokens
            .iter()
            .filter_map(|token| Self::raw_bytes(*token))
            .flatten()
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn decode_incremental(&self, _prev: &[TokenId], next: TokenId) -> Result<String> {
        self.decode(&[next], true)
    }

    fn vocab_size(&self) -> usize {
        Self::BASE_VOCAB_SIZE
    }

    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        &self.special
    }

    fn token_id(&self, text: &str) -> Option<TokenId> {
        match text {
            "\u{FFFD}" => Some(TokenId::new(Self::REPLACEMENT)),
            "<eos>" => Some(TokenId::new(Self::EOS)),
            _ if text.len() == 1 && text.is_ascii() => {
                Some(TokenId::new(text.as_bytes()[0] as u32))
            }
            _ => None,
        }
    }

    fn token_text(&self, token_id: TokenId) -> Option<&str> {
        self.texts
            .get(token_id.get() as usize)
            .and_then(|text| text.as_deref())
    }

    fn token_bytes(&self, token_id: TokenId) -> Option<Vec<u8>> {
        Self::raw_bytes(token_id)
    }

    fn info(&self) -> TokenizerInfo {
        TokenizerInfo {
            tokenizer_type: TokenizerType::BPE,
            vocab_size: Self::BASE_VOCAB_SIZE,
            special_tokens: self.special.clone(),
            supports_incremental: true,
            supports_chat_template: false,
            max_token_length: Some(2),
            model_name: Some("fragmented-utf8-policy-test".to_string()),
        }
    }
}

fn policy_request() -> InferenceRequest {
    InferenceRequest {
        id: RequestId::new(),
        prompt: "test".to_string(),
        model_id: ferrum_types::ModelId::new("test"),
        sampling_params: SamplingParams::greedy(),
        stream: false,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        ingress: None,
        api_request: None,
        evidence_request: Default::default(),
        metadata: HashMap::new(),
    }
}

async fn install_execution_readiness_wake(
    scheduler: &ContinuousBatchScheduler,
) -> (RequestId, ExecutionReadinessWake) {
    let request = policy_request();
    let request_id = request.id.clone();
    scheduler.submit(request).await.unwrap();
    let initial = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .expect("request should enter prefill before readiness deferral");
    assert_eq!(initial.requests.len(), 1);
    scheduler.mark_prefill_complete(&request_id, 1);
    let receipt = scheduler
        .defer_for_execution_readiness(std::slice::from_ref(&request_id))
        .unwrap();
    (request_id, receipt.into_wake())
}

fn readiness_wait_target(
    request_id: &RequestId,
) -> (Arc<SequenceHandle>, ExecutionReadinessWaitTarget) {
    let issuer = ferrum_execution::token_execution::ExecutorSequenceIssuer::new();
    let sequence = issuer.issue_sequence().unwrap();
    let mut request = policy_request();
    request.id = request_id.clone();
    let mut state = SequenceState::new(request, vec![TokenId(1)]);
    state.execution_frontier = Some(issuer.frontier(&sequence, 0).unwrap());
    let sequences = SequenceRegistry::default();
    assert!(sequences.insert(request_id.clone(), state).is_none());
    let handle = sequences.get(request_id).unwrap();
    let target = ExecutionReadinessWaitTarget {
        request: Arc::downgrade(&handle),
        sequence,
    };
    (handle, target)
}

#[test]
fn execution_readiness_registry_enforces_an_independent_fixed_cap() {
    let registry = ExecutionReadinessWaitRegistry::new();
    assert_eq!(MAX_EXECUTION_READINESS_WAITERS, 64);
    let mut slots = Vec::with_capacity(MAX_EXECUTION_READINESS_WAITERS);
    for _ in 0..MAX_EXECUTION_READINESS_WAITERS {
        slots.push(Arc::clone(&registry.slots).try_acquire_owned().unwrap());
    }
    assert!(Arc::clone(&registry.slots).try_acquire_owned().is_err());
}

#[tokio::test]
async fn execution_readiness_registry_resumes_only_after_successful_wait() {
    let scheduler = ContinuousBatchScheduler::new(ferrum_types::SchedulerConfig::default());
    let (request_id, wake) = install_execution_readiness_wake(&scheduler).await;
    let registry = ExecutionReadinessWaitRegistry::new();
    let (handle, target) = readiness_wait_target(&request_id);
    let sequence = target.sequence.clone();
    let work_notify = Arc::new(Notify::new());
    let notified = work_notify.notified();
    tokio::pin!(notified);
    let slot = registry.reserve().await.unwrap();
    registry
        .track(
            wake,
            async { Ok(()) },
            vec![target],
            Arc::clone(&work_notify),
            slot,
        )
        .unwrap();

    tokio::time::timeout(Duration::from_secs(1), &mut notified)
        .await
        .expect("successful waiter must wake the engine");
    assert_eq!(registry.pending_count(), 0);
    assert!(registry.take_failures().is_empty());
    assert_eq!(
        handle
            .lock()
            .as_ref()
            .unwrap()
            .execution_frontier
            .as_ref()
            .unwrap()
            .sequence(),
        &sequence
    );
    let resumed = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .expect("successful readiness must authorize an exact reprobe");
    assert_eq!(resumed.requests[0].request.id, request_id);

    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            registry.reap_finished();
            if registry.tasks.lock().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("finished waiter must be reapable");
}

#[tokio::test]
async fn execution_readiness_registry_reports_wait_errors_without_rescheduling() {
    let scheduler = ContinuousBatchScheduler::new(ferrum_types::SchedulerConfig::default());
    let (request_id, wake) = install_execution_readiness_wake(&scheduler).await;
    let registry = ExecutionReadinessWaitRegistry::new();
    let (handle, target) = readiness_wait_target(&request_id);
    let sequence = target.sequence.clone();
    let work_notify = Arc::new(Notify::new());
    let notified = work_notify.notified();
    tokio::pin!(notified);
    let slot = registry.reserve().await.unwrap();
    registry
        .track(
            wake,
            async { Err("hazard channel closed".to_string()) },
            vec![target],
            Arc::clone(&work_notify),
            slot,
        )
        .unwrap();

    tokio::time::timeout(Duration::from_secs(1), &mut notified)
        .await
        .expect("failed waiter must wake the terminal owner");
    let failures = registry.take_failures();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].requests.len(), 1);
    assert!(failures[0].requests[0]
        .request
        .ptr_eq(&Arc::downgrade(&handle)));
    assert_eq!(failures[0].requests[0].sequence, sequence);
    assert_eq!(failures[0].message, "hazard channel closed");
    assert!(scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .is_none());
}

#[tokio::test]
async fn execution_readiness_registry_converts_waiter_panic_to_terminal_failure() {
    let scheduler = ContinuousBatchScheduler::new(ferrum_types::SchedulerConfig::default());
    let (request_id, wake) = install_execution_readiness_wake(&scheduler).await;
    let registry = ExecutionReadinessWaitRegistry::new();
    let (handle, target) = readiness_wait_target(&request_id);
    let sequence = target.sequence.clone();
    let work_notify = Arc::new(Notify::new());
    let notified = work_notify.notified();
    tokio::pin!(notified);
    let slot = registry.reserve().await.unwrap();
    registry
        .track(
            wake,
            async {
                panic!("synthetic readiness panic");
                #[allow(unreachable_code)]
                Ok::<(), String>(())
            },
            vec![target],
            Arc::clone(&work_notify),
            slot,
        )
        .unwrap();

    tokio::time::timeout(Duration::from_secs(1), &mut notified)
        .await
        .expect("panicked waiter must wake the terminal owner");
    let failures = registry.take_failures();
    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].requests.len(), 1);
    assert!(failures[0].requests[0]
        .request
        .ptr_eq(&Arc::downgrade(&handle)));
    assert_eq!(failures[0].requests[0].sequence, sequence);
    assert_eq!(
        failures[0].message,
        "Request-state readiness waiter panicked"
    );
    assert!(scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .is_none());
}

#[tokio::test]
async fn execution_readiness_registry_cancels_ticket_before_shutdown_join() {
    let scheduler = ContinuousBatchScheduler::new(ferrum_types::SchedulerConfig::default());
    let (request_id, wake) = install_execution_readiness_wake(&scheduler).await;
    let registry = ExecutionReadinessWaitRegistry::new();
    let (handle, target) = readiness_wait_target(&request_id);
    let sequence = target.sequence.clone();
    let work_notify = Arc::new(Notify::new());
    let slot = registry.reserve().await.unwrap();
    registry
        .track(
            wake,
            std::future::pending::<std::result::Result<(), String>>(),
            vec![target],
            work_notify,
            slot,
        )
        .unwrap();
    assert_eq!(registry.pending_count(), 1);

    tokio::time::timeout(Duration::from_secs(1), registry.abort_and_join())
        .await
        .expect("shutdown must not leak readiness waiter tasks")
        .unwrap();
    assert_eq!(registry.pending_count(), 0);
    assert!(registry.tasks.lock().is_empty());
    assert!(registry.take_failures().is_empty());
    assert_eq!(
        handle
            .lock()
            .as_ref()
            .unwrap()
            .execution_frontier
            .as_ref()
            .unwrap()
            .sequence(),
        &sequence
    );
    let resumed = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .expect("cancelled shutdown ticket must not strand the frontier");
    assert_eq!(resumed.requests[0].request.id, request_id);
}

fn policy_tool_request(
    protocol: ferrum_types::ApiToolCallProtocol,
    tool_choice: ferrum_types::ApiToolChoice,
) -> InferenceRequest {
    let mut request = policy_request();
    request.api_request = Some(ferrum_types::ApiRequest::Chat(
        ferrum_types::ApiChatRequest {
            messages: Vec::new(),
            tools: vec![ferrum_types::ApiTool {
                tool_type: "function".to_string(),
                function: ferrum_types::ApiFunction {
                    name: "lookup_weather".to_string(),
                    description: None,
                    parameters: None,
                    strict: None,
                },
            }],
            tool_choice: Some(tool_choice),
            tool_call_protocol: protocol,
            legacy_functions: Vec::new(),
            legacy_function_call: None,
            response_format: None,
            stream_options: None,
        },
    ));
    request
}

#[derive(Debug, Clone, Default)]
struct TestResourceTraceState {
    reserved: i64,
    committed: i64,
    released: i64,
    rolled_back: i64,
}

impl TestResourceTraceState {
    fn outstanding_reserved(&self) -> i64 {
        self.reserved - self.released - self.rolled_back
    }

    fn outstanding_committed(&self) -> i64 {
        self.committed - self.released - self.rolled_back
    }

    fn assert_zero_outstanding(&self, key: &(String, String, String)) {
        assert_eq!(
            self.outstanding_reserved(),
            0,
            "resource {key:?} closed with outstanding reserved state: {self:?}"
        );
        assert_eq!(
            self.outstanding_committed(),
            0,
            "resource {key:?} closed with outstanding committed state: {self:?}"
        );
    }
}

fn resource_trace_temp_path(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock should be after UNIX_EPOCH")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ferrum-engine-{label}-{}-{nanos}.jsonl",
        std::process::id()
    ))
}

fn read_engine_profile_events(path: &Path) -> Vec<FerrumProfileEvent> {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("failed to read resource trace {}: {error}", path.display())
    });
    assert!(
        !contents.trim().is_empty(),
        "resource trace {} should not be empty",
        path.display()
    );

    contents
        .lines()
        .enumerate()
        .filter_map(|(line_no, line)| {
            if line.trim().is_empty() {
                return None;
            }
            let event: FerrumProfileEvent = serde_json::from_str(line).unwrap_or_else(|error| {
                panic!(
                    "line {} in {} is not a FerrumProfileEvent: {error}\n{line}",
                    line_no + 1,
                    path.display()
                )
            });
            event.validate().unwrap_or_else(|error| {
                panic!(
                    "line {} in {} failed profile validation: {error}\n{line}",
                    line_no + 1,
                    path.display()
                )
            });
            Some(event)
        })
        .collect()
}

fn flush_engine_profile_events(engine: &ContinuousBatchEngine) {
    if let Some(journal) = &engine
        .active_inner()
        .expect("engine execution is open")
        .scheduler_trace_jsonl
    {
        journal.flush().expect("scheduler trace flush barrier");
    }
}

fn assert_engine_resource_trace_balanced(path: &Path) -> Vec<ResourceTraceEvent> {
    let mut states: HashMap<(String, String, String), TestResourceTraceState> = HashMap::new();
    let mut resources = Vec::new();

    for event in read_engine_profile_events(path) {
        let Some(resource) = event.resource else {
            continue;
        };
        resource.validate().unwrap();
        let key = (
            resource.owner_kind.clone(),
            resource.owner_id.clone(),
            resource.resource_kind.clone(),
        );
        match resource.action {
            ResourceAction::Reserve => {
                let amount = resource.amount.expect("reserve amount");
                let state = states.entry(key).or_default();
                assert!(amount > 0, "reserve amount must be positive: {resource:?}");
                assert_eq!(resource.before, Some(state.outstanding_reserved()));
                state.reserved += amount;
                assert_eq!(resource.after, Some(state.outstanding_reserved()));
            }
            ResourceAction::Commit => {
                let amount = resource.amount.expect("commit amount");
                let state = states.entry(key).or_default();
                assert!(amount > 0, "commit amount must be positive: {resource:?}");
                assert_eq!(resource.before, Some(state.outstanding_committed()));
                state.committed += amount;
                assert_eq!(resource.after, Some(state.outstanding_committed()));
            }
            ResourceAction::Release => {
                let amount = resource.amount.expect("release amount");
                let state = states.entry(key).or_default();
                assert!(amount > 0, "release amount must be positive: {resource:?}");
                let before = state.outstanding_committed();
                assert!(
                    amount <= before,
                    "release underflow for {resource:?}; state before release: {state:?}"
                );
                assert_eq!(resource.before, Some(before));
                state.released += amount;
                assert_eq!(resource.after, Some(state.outstanding_committed()));
            }
            ResourceAction::Rollback => {
                let amount = resource.amount.expect("rollback amount");
                let state = states.entry(key).or_default();
                assert!(amount > 0, "rollback amount must be positive: {resource:?}");
                let before = state.outstanding_reserved();
                assert!(
                    amount <= before,
                    "rollback underflow for {resource:?}; state before rollback: {state:?}"
                );
                assert_eq!(resource.before, Some(before));
                state.rolled_back += amount;
                assert_eq!(resource.after, Some(state.outstanding_reserved()));
            }
            ResourceAction::RequestClose => {
                let owner_kind = resource.owner_kind.clone();
                let owner_id = resource.owner_id.clone();
                for (state_key, state) in states
                    .iter()
                    .filter(|(state_key, _)| state_key.0 == owner_kind && state_key.1 == owner_id)
                {
                    state.assert_zero_outstanding(state_key);
                }
                states.retain(|state_key, _| state_key.0 != owner_kind || state_key.1 != owner_id);
            }
            ResourceAction::RequestOpen
            | ResourceAction::Defer
            | ResourceAction::Reject
            | ResourceAction::CapacitySnapshot => {}
        }
        resources.push(resource);
    }

    for (key, state) in &states {
        state.assert_zero_outstanding(key);
    }

    resources
}

struct FailingFromSliceTensorFactory;

impl TensorFactory for FailingFromSliceTensorFactory {
    fn empty(&self, shape: &[usize], dtype: DataType, device: Device) -> Result<TensorRef> {
        MockTensorFactory.empty(shape, dtype, device)
    }

    fn zeros_like(&self, tensor: &TensorRef) -> Result<TensorRef> {
        MockTensorFactory.zeros_like(tensor)
    }

    fn from_slice(
        &self,
        _data: &[f32],
        _shape: &[usize],
        _dtype: DataType,
        _device: Device,
    ) -> Result<TensorRef> {
        Err(FerrumError::backend("synthetic tokens_to_tensor failure"))
    }

    fn to_device(&self, tensor: &TensorRef, device: Device) -> Result<TensorRef> {
        MockTensorFactory.to_device(tensor, device)
    }

    fn narrow(
        &self,
        tensor: &TensorRef,
        dim: usize,
        start: usize,
        length: usize,
    ) -> Result<TensorRef> {
        MockTensorFactory.narrow(tensor, dim, start, length)
    }

    fn reshape(&self, tensor: &TensorRef, shape: &[usize]) -> Result<TensorRef> {
        MockTensorFactory.reshape(tensor, shape)
    }

    fn zeros(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef> {
        MockTensorFactory.zeros(shape, dtype, device)
    }

    fn ones(&self, shape: &[usize], dtype: DataType, device: &Device) -> Result<TensorRef> {
        MockTensorFactory.ones(shape, dtype, device)
    }

    fn uniform(
        &self,
        shape: &[usize],
        low: f32,
        high: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef> {
        MockTensorFactory.uniform(shape, low, high, dtype, device)
    }

    fn normal(
        &self,
        shape: &[usize],
        mean: f32,
        std: f32,
        dtype: DataType,
        device: &Device,
    ) -> Result<TensorRef> {
        MockTensorFactory.normal(shape, mean, std, dtype, device)
    }

    fn from_tensor(&self, tensor: &TensorRef, device: &Device) -> Result<TensorRef> {
        MockTensorFactory.from_tensor(tensor, device)
    }
}

#[test]
fn continuous_engine_runtime_config_parses_env_snapshot() {
    let cfg = ContinuousEngineRuntimeConfig::from_env_vars(
        Some(64),
        [
            (BATCH_DECODE_PROF_ENV, "1"),
            (CHUNKED_PREFILL_ENV, "128"),
            (KV_CAPACITY_ENV, "2048"),
            (MAX_MODEL_LEN_ENV, "4096"),
            ("FERRUM_PROFILE_JSONL", "/tmp/profile.jsonl"),
            ("FERRUM_SCHEDULER_TRACE_JSONL", "/tmp/scheduler-trace.jsonl"),
        ],
    );

    assert_eq!(cfg.active_decode_prefill_chunk, Some(64));
    assert!(cfg.batch_decode_prof);
    assert!(cfg.chunked_prefill_present);
    assert_eq!(cfg.chunked_prefill_size, Some(128));
    // This snapshot preserves resolved configuration. Chunk extent now comes
    // from the scheduler's typed work and is tested at that execution boundary.
    let mut typed = EngineConfig::default();
    typed.scheduler.active_decode_prefill_chunk = Some(64);
    typed.runtime.chunked_prefill_size = Some(128);
    let resolved = ContinuousEngineRuntimeConfig::from_engine_config(&typed);
    assert_eq!(
        resolved.active_decode_prefill_chunk,
        cfg.active_decode_prefill_chunk
    );
    assert_eq!(resolved.chunked_prefill_size, cfg.chunked_prefill_size);
    assert_eq!(
        resolved.chunked_prefill_present,
        cfg.chunked_prefill_present
    );
    assert_eq!(cfg.kv_capacity, Some(2048));
    assert_eq!(cfg.max_model_len, Some(4096));
    assert_eq!(
        cfg.profile_jsonl.as_deref(),
        Some(std::path::Path::new("/tmp/profile.jsonl"))
    );
    assert_eq!(
        cfg.scheduler_trace_jsonl.as_deref(),
        Some(std::path::Path::new("/tmp/scheduler-trace.jsonl"))
    );
}

#[test]
fn continuous_engine_runtime_config_keeps_invalid_chunk_presence() {
    for value in ["invalid", "", "0"] {
        let cfg =
            ContinuousEngineRuntimeConfig::from_env_vars(None, [(CHUNKED_PREFILL_ENV, value)]);
        assert!(cfg.chunked_prefill_present);
        assert_eq!(cfg.chunked_prefill_size, None);
    }
    let absent = ContinuousEngineRuntimeConfig::from_env_vars(None, [] as [(&str, &str); 0]);
    assert!(!absent.chunked_prefill_present);
    assert_eq!(absent.chunked_prefill_size, None);
}

#[test]
fn performance_breakdown_reports_engine_timing_counters() {
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> =
        Arc::new(ferrum_testkit::MockKvCacheManager::new(256));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(ferrum_testkit::MockTensorFactory);
    let model_executor =
        ExecutorOwner::new(Box::new(ferrum_testkit::MockModelExecutor::instant(128)));
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        model_executor,
        tensor_factory,
    )
    .expect("legacy engine composition must match executor authority");

    engine
        .active_inner()
        .expect("engine execution is open")
        .record_scheduling_time(Duration::from_micros(1500));
    engine
        .active_inner()
        .expect("engine execution is open")
        .record_scheduling_time(Duration::from_micros(2500));
    engine
        .active_inner()
        .expect("engine execution is open")
        .record_model_execution_time(Duration::from_micros(10_000));
    engine
        .active_inner()
        .expect("engine execution is open")
        .record_model_execution_time(Duration::from_micros(14_000));
    engine
        .active_inner()
        .expect("engine execution is open")
        .record_iteration_lock_wait(Duration::from_micros(300));
    engine
        .active_inner()
        .expect("engine execution is open")
        .record_iteration_lock_wait(Duration::from_micros(700));

    let breakdown = engine.metrics().performance_breakdown;
    assert_eq!(breakdown.scheduling_time_ms, 2.0);
    assert_eq!(breakdown.model_execution_time_ms, 12.0);
    assert_eq!(breakdown.other_overhead_time_ms, 0.5);
}

fn test_continuous_engine() -> ContinuousBatchEngine {
    test_continuous_engine_with_config(EngineConfig::default())
}

fn test_continuous_engine_with_config(config: EngineConfig) -> ContinuousBatchEngine {
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        Arc::new(ferrum_testkit::MockTokenizer::new(128)),
        Arc::new(ferrum_testkit::MockSampler),
        ExecutorOwner::new(Box::new(PlanRuntimeAdmissionTestExecutor::new(128))),
        Arc::new(MockTensorFactory),
    )
    .unwrap()
}

#[test]
fn cache_metrics_do_not_claim_a_retired_whole_prompt_cache() {
    let engine = test_continuous_engine();
    assert!(engine.cache_metrics_snapshot().is_none());
    assert_eq!(engine.prefix_cache_hits(), 0);
}

#[test]
fn product_profile_path_constructs_a_vnext_execution_journal() {
    let trace_path = resource_trace_temp_path("profile-runtime-journal");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.runtime.profile_jsonl = Some(trace_path.clone());

    let engine = test_continuous_engine_with_config(config);

    assert!(engine
        .active_inner()
        .expect("engine execution is open")
        .profile_trace_journal()
        .is_some());
    assert!(engine
        .active_inner()
        .expect("engine execution is open")
        .scheduler_trace_jsonl
        .is_none());
    assert!(trace_path.exists());
    drop(engine);
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn profile_journal_budget_reaches_profile_scheduler_and_legacy_writers() {
    let directory = tempfile::tempdir().unwrap();
    let paths = ["profile.jsonl", "scheduler.jsonl", "legacy.jsonl"]
        .map(|name| directory.path().join(name));
    let mut config = EngineConfig::default();
    config.runtime.profile_journal.queue_capacity = 2;
    config.runtime.profile_journal.max_batch_records = 2048;
    let expected: ferrum_bench_core::JsonlJournalConfig = config.runtime.profile_journal.into();
    config.runtime.profile_jsonl = Some(paths[0].clone());
    config.runtime.scheduler_trace_jsonl = Some(paths[1].clone());
    config.runtime.legacy_scheduler_trace_jsonl = Some(paths[2].clone());
    let engine = test_continuous_engine_with_config(config);
    assert!(engine
        .active_inner()
        .expect("engine execution is open")
        .profile_trace_journal()
        .is_some());
    assert!(engine
        .active_inner()
        .expect("engine execution is open")
        .scheduler_trace_jsonl
        .is_some());
    assert!(engine
        .active_inner()
        .expect("engine execution is open")
        .legacy_scheduler_trace_jsonl
        .is_some());
    for path in &paths {
        let original = engine
            .active_inner()
            .expect("engine execution is open")
            .journal_scopes
            .lock()
            .producer::<serde_json::Value>(path)
            .unwrap();
        assert_eq!(original.config(), expected);
    }
    drop(engine);
}

#[tokio::test]
async fn idle_background_loop_parks_and_new_work_resumes_it() {
    let engine = Arc::new(test_continuous_engine());
    let mut first = policy_request();
    first.sampling_params.max_tokens = 1;
    tokio::time::timeout(Duration::from_secs(2), engine.infer(first))
        .await
        .expect("first request must complete")
        .unwrap();

    let parked_iteration = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let before = engine
                .active_inner()
                .expect("engine execution is open")
                .iteration_count
                .load(Ordering::Acquire);
            tokio::time::sleep(Duration::from_millis(10)).await;
            let after = engine
                .active_inner()
                .expect("engine execution is open")
                .iteration_count
                .load(Ordering::Acquire);
            if before == after {
                break after;
            }
        }
    })
    .await
    .expect("background loop must reach its idle wait");
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .iteration_count
            .load(Ordering::Acquire),
        parked_iteration,
        "an idle engine must not poll for work"
    );

    let mut second = policy_request();
    second.sampling_params.max_tokens = 1;
    tokio::time::timeout(Duration::from_secs(2), engine.infer(second))
        .await
        .expect("new work must wake the parked background loop")
        .unwrap();
    assert!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .iteration_count
            .load(Ordering::Acquire)
            > parked_iteration
    );
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_wakes_an_idle_background_loop() {
    let engine = test_continuous_engine();
    let background = engine.start_loop().unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        while engine
            .active_inner()
            .expect("engine execution is open")
            .iteration_count
            .load(Ordering::Acquire)
            == 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("background loop must enter one idle iteration");

    engine.shutdown().await.unwrap();
    tokio::time::timeout(Duration::from_secs(1), background.wait())
        .await
        .expect("shutdown must wake the idle loop")
        .result()
        .expect("background loop must not panic");
}

#[tokio::test]
async fn shutdown_signal_retains_wakes_before_wait_registration() {
    let engine = test_continuous_engine();

    let observer = engine
        .driver
        .shutdown(&engine.active_inner().expect("engine execution is open"))
        .unwrap();

    assert!(engine.driver.is_closing());
    assert!(!engine.driver.is_running());
    tokio::time::timeout(
        Duration::from_millis(100),
        engine
            .active_inner()
            .expect("engine execution is open")
            .shutdown_notify
            .notified(),
    )
    .await
    .expect("shutdown wake must survive registration after the signal");
    tokio::time::timeout(
        Duration::from_millis(100),
        engine
            .active_inner()
            .expect("engine execution is open")
            .work_notify
            .notified(),
    )
    .await
    .expect("work wake must survive registration after the signal");
    observer.wait().await.result().unwrap();
}

#[tokio::test]
async fn process_batch_rejects_a_scheduler_item_without_published_sequence_state() {
    let engine = test_continuous_engine_with_config(EngineConfig::default());
    let request = policy_request();
    let request_id = request.id.clone();
    let batch = ferrum_interfaces::BatchPlan {
        batch_id: ferrum_types::BatchId::new(),
        requests: vec![ferrum_interfaces::scheduler::ScheduledRequest::new(request)],
        max_sequence_length: 1,
        estimated_time_ms: None,
        resource_requirements: ferrum_interfaces::scheduler::BatchResourceRequirements::default(),
        created_at: chrono::Utc::now(),
    };

    let error = engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .expect_err("batch processing must not create a second tokenized request state");

    assert!(matches!(error, FerrumError::Internal { .. }));
    assert!(engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&request_id)
        .is_none());
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        0
    );
}

#[test]
fn resource_composition_mismatch_is_a_configuration_error() {
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);

    let legacy_executor = ExecutorOwner::new(Box::new(MockModelExecutor::instant(128)));
    let plan_runtime_error = ContinuousBatchEngine::new_plan_runtime(
        config.clone(),
        Arc::clone(&scheduler),
        Arc::clone(&tokenizer),
        Arc::clone(&sampler),
        legacy_executor,
        Arc::clone(&tensor_factory),
    )
    .err()
    .expect("legacy executor must not enter a plan-runtime composition");
    assert!(plan_runtime_error
        .to_string()
        .contains("does not match executor authority"));

    let kv_cache: Arc<dyn KvCacheManager + Send + Sync> = Arc::new(MockKvCacheManager::new(128));
    let plan_runtime_executor =
        ExecutorOwner::new(Box::new(PlanRuntimeAdmissionTestExecutor::new(128)));
    let legacy_error = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        plan_runtime_executor,
        tensor_factory,
    )
    .err()
    .expect("plan-runtime executor must not enter a legacy composition");
    assert!(legacy_error
        .to_string()
        .contains("does not match executor authority"));
}

#[test]
fn public_engine_authority_reports_validated_resource_composition() {
    let config = EngineConfig::default();
    let legacy = ContinuousBatchEngine::new(
        config.clone(),
        Arc::new(ContinuousBatchScheduler::new(config.scheduler)),
        Arc::new(ferrum_testkit::MockTokenizer::new(128)),
        Arc::new(ferrum_testkit::MockSampler),
        Arc::new(MockKvCacheManager::new(256)),
        ExecutorOwner::new(Box::new(MockModelExecutor::instant(128))),
        Arc::new(MockTensorFactory),
    )
    .unwrap();
    let (native, _, _, _) =
        plan_runtime_batch_decode_test_engine(PlanRuntimeBatchDecodeBehavior::Exact);
    for (engine, expected) in [
        (&legacy, ExecutionResourceAuthority::LegacyEngine),
        (&native, ExecutionResourceAuthority::PlanRuntime),
    ] {
        let engine: &dyn LlmInferenceEngine = engine;
        assert_eq!(engine.execution_resource_authority(), expected);
    }
}

fn plan_runtime_batch_decode_test_engine(
    behavior: PlanRuntimeBatchDecodeBehavior,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<PlanRuntimeBatchDecodeObservation>,
    Arc<dyn Tokenizer + Send + Sync>,
) {
    plan_runtime_batch_decode_test_engine_with_trace(behavior, None)
}

fn plan_runtime_batch_decode_profile_test_engine(
    profile_detail: ObservabilityProfileDetail,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<PlanRuntimeBatchDecodeObservation>,
    Arc<dyn Tokenizer + Send + Sync>,
) {
    let mut config = plan_runtime_test_config(None);
    config.runtime.profile_detail = profile_detail;
    let executor = PlanRuntimeBatchDecodeTestExecutor::with_capacity_stage_transition(
        PlanRuntimeBatchDecodeBehavior::Exact,
        ExecutorExecutionCapacityStage::StepAdmission,
        None,
    );
    plan_runtime_test_engine(config, executor, Arc::new(MockTensorFactory))
}

fn plan_runtime_batch_decode_test_engine_with_trace(
    behavior: PlanRuntimeBatchDecodeBehavior,
    trace_path: Option<PathBuf>,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<PlanRuntimeBatchDecodeObservation>,
    Arc<dyn Tokenizer + Send + Sync>,
) {
    plan_runtime_batch_decode_test_engine_with_trace_and_factory(
        behavior,
        trace_path,
        Arc::new(MockTensorFactory),
    )
}

fn plan_runtime_batch_decode_test_engine_with_trace_and_factory(
    behavior: PlanRuntimeBatchDecodeBehavior,
    trace_path: Option<PathBuf>,
    tensor_factory: Arc<dyn TensorFactory>,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<PlanRuntimeBatchDecodeObservation>,
    Arc<dyn Tokenizer + Send + Sync>,
) {
    plan_runtime_batch_decode_test_engine_with_trace_factory_and_stage(
        behavior,
        trace_path,
        tensor_factory,
        ExecutorExecutionCapacityStage::StepAdmission,
        None,
    )
}

fn plan_runtime_batch_decode_test_engine_with_trace_factory_and_stage(
    behavior: PlanRuntimeBatchDecodeBehavior,
    trace_path: Option<PathBuf>,
    tensor_factory: Arc<dyn TensorFactory>,
    capacity_stage: ExecutorExecutionCapacityStage,
    capacity_stage_after_first_call: Option<ExecutorExecutionCapacityStage>,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<PlanRuntimeBatchDecodeObservation>,
    Arc<dyn Tokenizer + Send + Sync>,
) {
    let executor = PlanRuntimeBatchDecodeTestExecutor::with_capacity_stage_transition(
        behavior,
        capacity_stage,
        capacity_stage_after_first_call,
    );
    plan_runtime_test_engine(
        plan_runtime_test_config(trace_path),
        executor,
        tensor_factory,
    )
}

fn plan_runtime_test_config(trace_path: Option<PathBuf>) -> EngineConfig {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.runtime.scheduler_trace_jsonl = trace_path;
    config.runtime.profile_entrypoint = Some(ProfileEntrypoint::Synthetic);
    config
}

fn plan_runtime_test_engine<E: ObservedTestExecutor>(
    config: EngineConfig,
    executor: E,
    tensor_factory: Arc<dyn TensorFactory>,
) -> (
    ContinuousBatchEngine,
    Arc<ContinuousBatchScheduler>,
    Arc<E::Observation>,
    Arc<dyn Tokenizer + Send + Sync>,
) {
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(64, &[("test", 5), ("ok", 6)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler.clone(),
        tokenizer.clone(),
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();
    (engine, scheduler, observation, tokenizer)
}

async fn install_plan_runtime_decode_cohort(
    engine: &ContinuousBatchEngine,
    scheduler: &ContinuousBatchScheduler,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
) -> (Vec<RequestId>, Vec<TokenId>, Vec<ExecutorSequence>) {
    install_plan_runtime_decode_frontiers(engine, scheduler, tokenizer, &["test", "ok"]).await
}

async fn install_plan_runtime_decode_frontiers(
    engine: &ContinuousBatchEngine,
    scheduler: &ContinuousBatchScheduler,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
    prompts: &[&str],
) -> (Vec<RequestId>, Vec<TokenId>, Vec<ExecutorSequence>) {
    let mut ids = Vec::new();
    for prompt in prompts {
        let mut request = policy_request();
        request.prompt = (*prompt).to_string();
        request.sampling_params.max_tokens = 4;
        request
            .metadata
            .insert(PROMPT_TOKENS_METADATA_KEY.to_string(), serde_json::json!(1));
        let tokens = tokenizer.encode(prompt, true).unwrap();
        assert_eq!(tokens.len(), 1);
        let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            tokens.clone(),
            Some(tokenizer.clone()),
            Some(64),
        );
        let TokenAdmissionOutcome::Admitted(frontier) = engine
            .active_inner()
            .expect("engine execution is open")
            .model_executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    &request.id,
                    &tokens,
                    sequence.model_maximum_sequence_tokens(),
                    tokens.len(),
                    0,
                )
                .unwrap(),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("controlled cohort admission must succeed");
        };
        sequence.execution_frontier = Some(frontier);
        ids.push(request.id.clone());
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .insert(request.id.clone(), sequence);
        scheduler.submit(request).await.unwrap();
    }
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(prompts.len().max(1)))
        .await
        .unwrap();
    assert_eq!(batch.requests.len(), ids.len());
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .unwrap();
    let sequences = &engine
        .active_inner()
        .expect("engine execution is open")
        .sequences;
    let initial = ids
        .iter()
        .map(|id| {
            sequences
                .get(id)
                .unwrap()
                .lock()
                .as_ref()
                .unwrap()
                .generated_tokens[0]
        })
        .collect();
    let identities = ids
        .iter()
        .map(|id| {
            let frontier = sequences
                .get(id)
                .unwrap()
                .lock()
                .as_ref()
                .unwrap()
                .execution_frontier
                .clone()
                .unwrap();
            assert_eq!(frontier.consumed_tokens(), 1);
            frontier.sequence().clone()
        })
        .collect();
    (ids, initial, identities)
}

async fn install_profiled_plan_runtime_decode_frontier(
    engine: &ContinuousBatchEngine,
    scheduler: &ContinuousBatchScheduler,
    tokenizer: Arc<dyn Tokenizer + Send + Sync>,
) -> RequestId {
    let mut request = policy_request();
    request.prompt = "test".to_owned();
    request.sampling_params.max_tokens = 4;
    request.evidence_request.capture_engine_token_timing = true;
    request
        .metadata
        .insert(PROMPT_TOKENS_METADATA_KEY.to_owned(), serde_json::json!(1));
    let id = request.id.clone();
    let tokens = vec![TokenId(5)];
    let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request.clone(),
        tokens.clone(),
        Some(tokenizer),
        Some(64),
    );
    let TokenAdmissionOutcome::Admitted(frontier) = engine
        .active_inner()
        .expect("engine execution is open")
        .model_executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_product_request(
                &id,
                &tokens,
                sequence.model_maximum_sequence_tokens(),
                tokens.len(),
                0,
            )
            .unwrap(),
            adapter: None,
        })
        .await
        .unwrap()
    else {
        panic!("controlled profile admission must succeed");
    };
    sequence.execution_frontier = Some(frontier);
    engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .insert(id.clone(), sequence);
    scheduler.submit(request).await.unwrap();
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(1))
        .await
        .unwrap();
    assert_eq!(batch.requests.len(), 1);
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .unwrap();
    id
}

#[tokio::test]
async fn token_lone_decode_self_recompute_releases_exact_incarnation_before_requeue() {
    let (engine, scheduler, executor, tokenizer) =
        plan_runtime_batch_decode_test_engine(PlanRuntimeBatchDecodeBehavior::Exact);
    let (ids, initial_tokens, incarnations) =
        install_plan_runtime_decode_frontiers(&engine, &scheduler, tokenizer, &["test"]).await;
    let id = &ids[0];
    let mut availability = Vec::new();
    let epochs = engine
        .active_inner()
        .expect("engine execution is open")
        .model_executor
        .write_execution_capacity_snapshot(&mut availability)
        .unwrap()
        .unwrap();
    let condition = ferrum_runtime::CapacityWaitCondition::from_observation(
        epochs.coordinator_id.get(),
        availability,
    )
    .unwrap();
    let deferral = AdmissionDeferral::new(
        DeferredAction::WaitForRelease,
        AdmissionWakeEpochs::new(
            epochs.coordinator_id,
            epochs.release_epoch,
            epochs.capacity_epoch,
            0,
        ),
        condition,
    );
    let action = scheduler
        .defer_decode_for_execution_capacity(
            &ids,
            deferral,
            &engine
                .active_inner()
                .expect("engine execution is open")
                .execution_capacity_release_snapshot()
                .unwrap(),
        )
        .unwrap();
    let ExecutionCapacityAction::YieldPlanned { transaction } = action else {
        panic!("self recompute must be planned");
    };
    assert_eq!(transaction.kind(), PressureYieldKind::SelfRecompute);
    assert_eq!(transaction.victim_request_id(), id);
    assert!(!engine
        .active_inner()
        .expect("engine execution is open")
        .execute_capacity_yield(&transaction, 1, None)
        .await
        .unwrap());
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
    assert!(incarnations[0].observed_end().is_some());
    assert!(executor.token_frontier_for_request(id).is_none());
    assert_eq!(executor.released_cache_count.load(Ordering::Acquire), 1);
    let trace = scheduler.trace_snapshot();
    assert_eq!(trace.active_len, 0);
    assert_eq!(trace.waiting_queue_len, 1);
    assert_eq!(trace.pressure_pending_release_fences, 0);
    assert_eq!(trace.capacity_release_epoch, 1);
    let sequences = &engine
        .active_inner()
        .expect("engine execution is open")
        .sequences;
    let sequence_handle = sequences.get(id).unwrap();
    let sequence_guard = sequence_handle.lock();
    let sequence = sequence_guard.as_ref().unwrap();
    assert!(sequence.execution_frontier.is_none());
    assert_eq!(sequence.generated_tokens, initial_tokens);
    assert_eq!(sequence.phase, RequestPhase::Waiting);
    assert_eq!(sequence.preemption_count, 1);
}

#[tokio::test]
async fn plan_runtime_batch_decode_process_batch_submits_one_cohort() {
    let trace_path = resource_trace_temp_path("plan-runtime-authoritative-batch-decode");
    let _ = std::fs::remove_file(&trace_path);
    let (engine, scheduler, executor, tokenizer) = plan_runtime_batch_decode_test_engine_with_trace(
        PlanRuntimeBatchDecodeBehavior::Exact,
        Some(trace_path.clone()),
    );
    let (request_ids, initial_tokens, cache_ids) =
        install_plan_runtime_decode_cohort(&engine, &scheduler, tokenizer).await;
    let decode_batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .expect("ready decode cohort");

    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&decode_batch)
        .await
        .unwrap();

    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.decode_calls.load(Ordering::Relaxed), 0);
    let sequences = &engine
        .active_inner()
        .expect("engine execution is open")
        .sequences;
    for ((request_id, initial_token), cache_id) in
        request_ids.iter().zip(initial_tokens).zip(cache_ids)
    {
        let sequence_handle = sequences.get(request_id).expect("decoded sequence");
        let sequence_guard = sequence_handle.lock();
        let sequence = sequence_guard.as_ref().unwrap();
        assert_eq!(sequence.generated_tokens.first(), Some(&initial_token));
        assert_eq!(sequence.generated_tokens.len(), 2);
        assert_eq!(
            sequence.execution_frontier.as_ref().unwrap().sequence(),
            &cache_id
        );
        assert_eq!(sequence.tokens_this_iteration, 1);
    }

    flush_engine_profile_events(&engine);
    let trace = std::fs::read_to_string(&trace_path).unwrap();
    let events = trace
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<FerrumProfileEvent>(line).unwrap())
        .collect::<Vec<_>>();
    assert!(
        events.iter().all(|event| {
            event
                .resource
                .as_ref()
                .is_none_or(|resource| resource.resource_kind != "backend_workspace")
        }),
        "PlanRuntime must not publish the legacy synthetic backend_workspace ledger"
    );
    let _ = std::fs::remove_file(trace_path);
}

#[tokio::test]
async fn token_capacity_unproved_close_keeps_the_release_fence_and_original_owner() {
    let (engine, scheduler, executor, tokenizer) = plan_runtime_batch_decode_test_engine(
        PlanRuntimeBatchDecodeBehavior::DeferThenPreemptionFails,
    );
    let (_, _, incarnations) =
        install_plan_runtime_decode_cohort(&engine, &scheduler, tokenizer).await;
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .unwrap();
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .unwrap();
    engine
        .active_inner()
        .expect("engine execution is open")
        .execution_closes
        .join_observers()
        .await;
    engine
        .active_inner()
        .expect("engine execution is open")
        .output_tasks
        .join_current()
        .await;
    let error = engine
        .active_inner()
        .expect("engine execution is open")
        .run_iteration()
        .await
        .err()
        .expect("the driver must observe the managed close failure");
    assert!(error
        .to_string()
        .contains("synthetic request-scoped close failure"));
    let _ = engine
        .active_inner()
        .expect("engine execution is open")
        .drain_execution_closes();
    assert_eq!(executor.released_cache_count.load(Ordering::Acquire), 0);
    assert!(incarnations
        .iter()
        .all(|identity| identity.observed_end().is_none()));
    assert_eq!(scheduler.trace_snapshot().capacity_release_epoch, 0);
    assert!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count()
            > 0
    );
    assert!(scheduler.trace_snapshot().pressure_pending_release_fences > 0);
}

#[tokio::test]
async fn plan_runtime_profile_records_only_explicit_decode_host_stages() {
    for (profile_detail, expected_stage_count) in [
        (ObservabilityProfileDetail::Full, 3),
        (ObservabilityProfileDetail::Latency, 3),
        (ObservabilityProfileDetail::Off, 0),
    ] {
        let (engine, scheduler, executor, tokenizer) =
            plan_runtime_batch_decode_profile_test_engine(profile_detail);
        let request_id =
            install_profiled_plan_runtime_decode_frontier(&engine, &scheduler, tokenizer).await;

        let batch = scheduler
            .next_batch(ferrum_interfaces::BatchHint::simple(1))
            .await
            .unwrap();
        engine
            .active_inner()
            .expect("engine execution is open")
            .process_batch(&batch)
            .await
            .unwrap();

        assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
        let timing = engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .get(&request_id)
            .unwrap()
            .lock()
            .as_mut()
            .unwrap()
            .take_execution_evidence()
            .unwrap()
            .unwrap()
            .engine_token_timing
            .unwrap();
        assert_eq!(timing.token_commit_nanos_since_request_start.len(), 2);
        assert_eq!(
            timing.decode_stage_intervals.len(),
            expected_stage_count,
            "profile detail {} stage capture",
            profile_detail.as_str()
        );
        if matches!(
            profile_detail,
            ObservabilityProfileDetail::Full | ObservabilityProfileDetail::Latency
        ) {
            let scheduling = &timing.decode_stage_intervals[0];
            let execution = &timing.decode_stage_intervals[1];
            let postprocess = &timing.decode_stage_intervals[2];
            assert_eq!(scheduling.stage, EngineDecodeStage::DecodeScheduling);
            assert_eq!(execution.stage, EngineDecodeStage::DecodeExecution);
            assert_eq!(postprocess.stage, EngineDecodeStage::DecodePostprocess);
            assert_eq!(
                scheduling.start_nanos_since_request_start,
                timing.token_commit_nanos_since_request_start[0]
            );
            assert_eq!(
                scheduling.end_nanos_since_request_start,
                execution.start_nanos_since_request_start
            );
            assert_eq!(
                execution.end_nanos_since_request_start,
                postprocess.start_nanos_since_request_start
            );
            assert_eq!(
                postprocess.end_nanos_since_request_start,
                timing.token_commit_nanos_since_request_start[1]
            );
        }
    }
}

#[tokio::test]
async fn plan_runtime_batch_decode_does_not_materialize_host_token_tensors() {
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(FailingFromSliceTensorFactory);
    let (engine, scheduler, executor, tokenizer) =
        plan_runtime_batch_decode_test_engine_with_trace_and_factory(
            PlanRuntimeBatchDecodeBehavior::Exact,
            None,
            tensor_factory,
        );
    let (request_ids, _, _) =
        install_plan_runtime_decode_cohort(&engine, &scheduler, tokenizer).await;

    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(request_ids.len()))
        .await
        .unwrap();
    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .expect("common decode must not call TensorFactory::from_slice");

    assert_eq!(executor.batch_decode_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.decode_calls.load(Ordering::Relaxed), 0);
    let sequences = &engine
        .active_inner()
        .expect("engine execution is open")
        .sequences;
    for request_id in request_ids {
        assert_eq!(
            sequences
                .get(&request_id)
                .expect("decoded sequence must remain active")
                .lock()
                .as_ref()
                .unwrap()
                .generated_tokens
                .len(),
            2
        );
    }
}

#[tokio::test]
async fn plan_runtime_status_separates_static_and_dynamic_memory() {
    let config = EngineConfig::default();
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = ExecutorOwner::new(Box::new(PlanRuntimeAdmissionTestExecutor::new(128)));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        tokenizer,
        sampler,
        executor,
        tensor_factory,
    )
    .unwrap();

    let status = engine.status().await;

    assert_eq!(status.memory_usage.total_bytes, 900);
    assert_eq!(status.memory_usage.used_bytes, 500);
    assert_eq!(status.memory_usage.free_bytes, 400);
    assert_eq!(status.memory_usage.cache_memory_bytes, 100);
    assert!((status.memory_usage.utilization_percent - 55.555557).abs() < 0.001);
}

#[tokio::test]
async fn plan_runtime_product_path_requires_typed_admission_before_prefill() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeAdmissionTestExecutor::new(128);
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(FailingFromSliceTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        tokenizer,
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;

    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.response().finish_reason, FinishReason::Length);
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 1);
    let completions = executor
        .completions
        .lock()
        .expect("completion receipt mutex poisoned");
    assert_eq!(completions.len(), 1);
    assert_eq!(completions[0].request_id(), &response.response().request_id);
    assert_eq!(
        completions[0].input_tokens(),
        response.response().usage.prompt_tokens as u64
    );
    assert_eq!(
        completions[0].output_tokens(),
        response.response().usage.completion_tokens as u64
    );
    drop(completions);
    assert!(executor
        .retained
        .lock()
        .expect("retained admission mutex poisoned")
        .is_empty());
    let trace = scheduler.trace_snapshot();
    assert_eq!(trace.legacy_waiting_admission_ticks, 0);
    assert!(trace.dynamic_admission_ticks >= 1);
}

#[tokio::test]
async fn plan_runtime_prefill_resource_failure_preserves_terminal_error_and_releases_request() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeAdmissionTestExecutor::new_with_resource_prefill_failure(128);
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(FailingFromSliceTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        tokenizer,
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.clone();

    let error = engine
        .infer(request)
        .await
        .expect_err("injected resource failure must terminate the request");

    assert!(
        matches!(error, FerrumError::ResourceExhausted { .. }),
        "unexpected terminal error: {error:?}"
    );
    assert!(error
        .to_string()
        .contains("typed diagnostic resource failure after prefill submission"));
    assert!(!engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&request_id)
        .is_some());
    assert!(executor
        .retained
        .lock()
        .expect("retained admission mutex poisoned")
        .is_empty());
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn plan_runtime_recompute_admission_preserves_product_usage_accounting() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeAdmissionTestExecutor::new(128);
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        scheduler,
        tokenizer,
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 8;
    let request_id = request.id.clone();
    let mut sequence =
        SequenceState::new(request.clone(), vec![TokenId::new(11), TokenId::new(12)]);
    sequence.generated_tokens.push(TokenId::new(13));
    sequence.phase = RequestPhase::Waiting;
    engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .insert(request_id.clone(), sequence);

    let admitted = engine
        .active_inner()
        .expect("engine execution is open")
        .probe_executor_prefill_admission_for_test(&request)
        .await;

    assert!(admitted);
    assert_eq!(
        executor
            .admission_accounting
            .lock()
            .expect("admission accounting mutex poisoned")
            .as_slice(),
        &[(request_id.clone(), 3, 2, 1)]
    );
    let frontier = executor
        .tokens
        .frontier_for_request(&request_id)
        .expect("admitted identity");
    let close = engine
        .active_inner()
        .expect("engine execution is open")
        .model_executor
        .cancel_tokens(frontier.sequence())
        .expect("close exact admission");
    assert!(matches!(
        close.disposition,
        ferrum_execution::token_execution::ExecutionCloseDisposition::Ended(_)
    ));
    assert!(executor.tokens.frontier_for_request(&request_id).is_none());
}

#[tokio::test]
async fn plan_runtime_prefill_cohort_uses_one_capacity_aware_batch() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.scheduler.max_running_requests = 2;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        128,
        &[("test", 5), ("ok", 6), ("generated", 42)],
    ));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let executor = PlanRuntimeAdmissionTestExecutor::new(128);
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(FailingFromSliceTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        Arc::clone(&tokenizer),
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();

    let mut request_ids = Vec::new();
    for prompt in ["test", "ok"] {
        let mut request = policy_request();
        request.prompt = prompt.to_string();
        request.sampling_params.max_tokens = 2;
        request
            .metadata
            .insert(PROMPT_TOKENS_METADATA_KEY.to_string(), serde_json::json!(1));
        scheduler.submit(request.clone()).await.unwrap();
        let tokens = tokenizer.encode(&request.prompt, true).unwrap();
        let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            tokens.clone(),
            Some(Arc::clone(&tokenizer)),
            Some(128),
        );
        let TokenAdmissionOutcome::Admitted(frontier) = engine
            .active_inner()
            .expect("engine execution is open")
            .model_executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    &request.id,
                    &tokens,
                    sequence.model_maximum_sequence_tokens(),
                    tokens.len(),
                    0,
                )
                .unwrap(),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("fixture batch admission must succeed");
        };
        sequence.execution_frontier = Some(frontier);
        request_ids.push(request.id.clone());
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .insert(request.id.clone(), sequence);
    }
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .expect("two admitted prefills should form one scheduler cohort");
    assert_eq!(batch.requests.len(), 2);

    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .unwrap();

    assert_eq!(executor.batch_prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 0);
    assert!(executor
        .retained
        .lock()
        .expect("retained admission mutex poisoned")
        .is_empty());
    let sequences = &engine
        .active_inner()
        .expect("engine execution is open")
        .sequences;
    let remaining_request_ids = sequences
        .snapshot()
        .iter()
        .map(|handle| handle.request_id().clone())
        .collect::<Vec<_>>();
    let scheduler_trace = scheduler.trace_snapshot();
    for request_id in request_ids {
        let sequence_handle = sequences.get(&request_id).unwrap_or_else(|| {
            panic!(
                "prefilled sequence {request_id} disappeared; remaining={remaining_request_ids:?}; scheduler={scheduler_trace:?}"
            )
        });
        let sequence_guard = sequence_handle.lock();
        let sequence = sequence_guard.as_ref().unwrap();
        assert!(sequence.prefill_complete);
        assert_eq!(sequence.generated_tokens.len(), 1);
        assert!(sequence.execution_frontier.is_some());
    }
}

#[tokio::test]
async fn plan_runtime_batch_prefill_deferral_falls_back_without_failing_requests() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.scheduler.max_running_requests = 2;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        128,
        &[("test", 5), ("ok", 6), ("generated", 42)],
    ));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let executor = PlanRuntimeAdmissionTestExecutor::new_with_batch_prefill_deferral(128);
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(FailingFromSliceTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        Arc::clone(&tokenizer),
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();

    let mut request_ids = Vec::new();
    for prompt in ["test", "ok"] {
        let mut request = policy_request();
        request.prompt = prompt.to_string();
        request.sampling_params.max_tokens = 2;
        request
            .metadata
            .insert(PROMPT_TOKENS_METADATA_KEY.to_string(), serde_json::json!(1));
        scheduler.submit(request.clone()).await.unwrap();
        let tokens = tokenizer.encode(&request.prompt, true).unwrap();
        let mut sequence = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request.clone(),
            tokens.clone(),
            Some(Arc::clone(&tokenizer)),
            Some(128),
        );
        let TokenAdmissionOutcome::Admitted(frontier) = engine
            .active_inner()
            .expect("engine execution is open")
            .model_executor
            .admit_tokens(ExecutionAdmission {
                context: ExecutorPrefillAdmission::for_product_request(
                    &request.id,
                    &tokens,
                    sequence.model_maximum_sequence_tokens(),
                    tokens.len(),
                    0,
                )
                .unwrap(),
                adapter: None,
            })
            .await
            .unwrap()
        else {
            panic!("fixture batch admission must succeed");
        };
        sequence.execution_frontier = Some(frontier);
        request_ids.push(request.id.clone());
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .insert(request.id.clone(), sequence);
    }
    let batch = scheduler
        .next_batch(ferrum_interfaces::BatchHint::simple(2))
        .await
        .expect("two admitted prefills should form one scheduler cohort");

    engine
        .active_inner()
        .expect("engine execution is open")
        .process_batch(&batch)
        .await
        .unwrap();

    assert_eq!(executor.batch_prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 2);
    assert!(executor
        .retained
        .lock()
        .expect("retained admission mutex poisoned")
        .is_empty());
    let sequences = &engine
        .active_inner()
        .expect("engine execution is open")
        .sequences;
    for request_id in request_ids {
        let sequence_handle = sequences
            .get(&request_id)
            .expect("deferred batch participant must survive per-request fallback");
        let sequence_guard = sequence_handle.lock();
        let sequence = sequence_guard.as_ref().unwrap();
        assert!(sequence.prefill_complete);
        assert_eq!(sequence.generated_tokens.len(), 1);
    }
}

#[tokio::test]
async fn plan_runtime_serial_backing_growth_converges_before_batch_submission() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.scheduler.max_running_requests = 2;
    config.scheduler.prefill_first_until_active = Some(2);
    config.batching.max_batch_size = 2;
    config.batching.max_num_batched_tokens = 16;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        128,
        &[("test", 5), ("ok", 6), ("generated", 42)],
    ));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let executor = PlanRuntimeAdmissionTestExecutor::new_with_serial_backing_growth(128);
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        Arc::clone(&tokenizer),
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();

    for prompt in ["test", "ok"] {
        let mut request = policy_request();
        request.prompt = prompt.to_string();
        request.sampling_params.max_tokens = 2;
        request
            .metadata
            .insert(PROMPT_TOKENS_METADATA_KEY.to_string(), serde_json::json!(1));
        scheduler.submit(request.clone()).await.unwrap();
        let tokens = tokenizer.encode(&request.prompt, true).unwrap();
        engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .insert(
                request.id.clone(),
                SequenceState::new_with_tokenizer_and_model_vocab_size(
                    request,
                    tokens,
                    Some(Arc::clone(&tokenizer)),
                    Some(128),
                ),
            );
    }

    engine
        .active_inner()
        .expect("engine execution is open")
        .run_iteration()
        .await
        .unwrap();

    assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 3);
    assert_eq!(executor.batch_prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 0);
    assert!(executor
        .retained
        .lock()
        .expect("retained admission mutex poisoned")
        .is_empty());
    assert_eq!(scheduler.prefilling_count(), 0);
    assert_eq!(scheduler.decoding_count(), 2);
}

#[tokio::test]
async fn request_publication_lock_is_released_during_device_execution() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.scheduler.max_running_requests = 2;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(128, &[("test", 5)]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let executor =
        PlanRuntimeAdmissionTestExecutor::new_with_prefill_latency(128, Duration::from_millis(100));
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        Arc::clone(&tokenizer),
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();

    let mut request = policy_request();
    request.prompt = "test".to_string();
    request.sampling_params.max_tokens = 2;
    request
        .metadata
        .insert(PROMPT_TOKENS_METADATA_KEY.to_string(), serde_json::json!(1));
    scheduler.submit(request.clone()).await.unwrap();
    let tokens = tokenizer.encode(&request.prompt, true).unwrap();
    engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .insert(
            request.id.clone(),
            SequenceState::new_with_tokenizer_and_model_vocab_size(
                request,
                tokens,
                Some(tokenizer),
                Some(128),
            ),
        );

    let inner = Arc::clone(&engine.active_inner().expect("engine execution is open"));
    let iteration = tokio::spawn(async move { inner.run_iteration().await });
    tokio::time::timeout(Duration::from_millis(50), async {
        while executor.prefill_calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("test prefill must enter device execution");

    let publication_inner = engine.active_inner().expect("engine execution is open");
    let publication_guard = tokio::time::timeout(
        Duration::from_millis(20),
        publication_inner.iteration_lock.lock(),
    )
    .await
    .expect("request publication must not wait for the device wave");
    drop(publication_guard);
    drop(publication_inner);
    iteration.await.unwrap().unwrap();
}

#[tokio::test]
async fn plan_runtime_chunked_prefill_retries_exact_chunk_and_samples_only_final_chunk() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.scheduler.max_running_requests = 1;
    config.scheduler.prefill_step_chunk = Some(2);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeChunkedPrefillTestExecutor::new(true);
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = Arc::new(
        ContinuousBatchEngine::new_plan_runtime(
            config,
            Arc::clone(&scheduler),
            tokenizer,
            sampler,
            executor_owner,
            tensor_factory,
        )
        .unwrap(),
    );
    let mut request = policy_request();
    request.prompt = "one two three four".to_string();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.clone();
    let infer_engine = Arc::clone(&engine);
    let inference = tokio::spawn(async move { infer_engine.infer(request).await });

    tokio::time::timeout(Duration::from_secs(1), async {
        while executor.capacity_wait_registrations.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("first prefill chunk must park on exact execution capacity");

    assert_eq!(
        *executor
            .attempted_chunks
            .lock()
            .expect("chunked prefill attempt mutex poisoned"),
        vec![PrefillChunk::new(0, 2, 5).unwrap()]
    );
    {
        let sequences = &engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        let sequence_handle = sequences
            .get(&request_id)
            .expect("deferred prefill retains engine sequence state");
        let sequence_guard = sequence_handle.lock();
        let sequence = sequence_guard.as_ref().unwrap();
        assert_eq!(sequence.prefill_tokens_processed, 0);
        assert!(sequence.generated_tokens.is_empty());
    }
    assert_eq!(scheduler.prefilling_count(), 1);
    assert_eq!(
        scheduler
            .trace_snapshot()
            .execution_capacity_blocked_prefill_len,
        1
    );

    executor.publish_release();
    let response = tokio::time::timeout(Duration::from_secs(2), inference)
        .await
        .expect("capacity release must resume the exact prefill chunk")
        .expect("chunked prefill inference task must not panic")
        .unwrap();

    assert_eq!(response.response().finish_reason, FinishReason::Length);
    assert_eq!(
        response.response().tokens.len(),
        1,
        "only the final chunk may sample"
    );
    assert_eq!(executor.completed_chunks.load(Ordering::Relaxed), 3);
    assert_eq!(
        *executor
            .attempted_chunks
            .lock()
            .expect("chunked prefill attempt mutex poisoned"),
        vec![
            PrefillChunk::new(0, 2, 5).unwrap(),
            PrefillChunk::new(0, 2, 5).unwrap(),
            PrefillChunk::new(2, 2, 5).unwrap(),
            PrefillChunk::new(4, 1, 5).unwrap(),
        ]
    );
    assert!(executor
        .retained
        .lock()
        .expect("chunked prefill retained mutex poisoned")
        .is_empty());
    assert_eq!(
        scheduler
            .trace_snapshot()
            .execution_capacity_blocked_prefill_len,
        0
    );
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn plan_runtime_partial_prefill_completion_commits_only_the_executor_prefix() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.scheduler.max_running_requests = 1;
    config.scheduler.prefill_step_chunk = Some(4);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeChunkedPrefillTestExecutor::new_with_partial_first_prefill();
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        tokenizer,
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();
    let mut request = policy_request();
    request.prompt = "one two three four".to_string();
    request.sampling_params.max_tokens = 1;

    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.response().finish_reason, FinishReason::Length);
    assert_eq!(
        response.response().tokens.len(),
        1,
        "only the final prefix may sample"
    );
    assert_eq!(executor.completed_chunks.load(Ordering::Relaxed), 3);
    assert_eq!(
        *executor
            .attempted_chunks
            .lock()
            .expect("chunked prefill attempt mutex poisoned"),
        vec![
            PrefillChunk::new(0, 4, 5).unwrap(),
            PrefillChunk::new(2, 2, 5).unwrap(),
            PrefillChunk::new(4, 1, 5).unwrap(),
        ]
    );
    assert!(executor
        .retained
        .lock()
        .expect("chunked prefill retained mutex poisoned")
        .is_empty());
    assert_eq!(
        scheduler
            .trace_snapshot()
            .execution_capacity_blocked_prefill_len,
        0
    );
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn plan_runtime_one_token_prefill_without_external_releaser_fails_closed() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    config.scheduler.max_running_requests = 1;
    config.scheduler.prefill_step_chunk = Some(1);
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeChunkedPrefillTestExecutor::new(true);
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        tokenizer,
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();
    let mut request = policy_request();
    request.prompt = "one two three four".to_string();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.clone();

    let error = tokio::time::timeout(Duration::from_secs(2), engine.infer(request))
        .await
        .expect("minimum prefill frontier must not wait on itself")
        .expect_err("a minimum prefill frontier without an external releaser must fail closed");

    assert!(
        matches!(error, FerrumError::ResourceExhausted { .. }),
        "unexpected terminal error: {error:?}"
    );
    assert!(error
        .to_string()
        .contains("one-token prefill does not fit and has no external capacity releaser"));
    assert_eq!(
        executor.capacity_wait_registrations.load(Ordering::Relaxed),
        0,
        "self-only capacity failure must not register a passive waiter"
    );
    assert_eq!(
        scheduler
            .trace_snapshot()
            .execution_capacity_blocked_prefill_len,
        0
    );
    assert_eq!(executor.completed_chunks.load(Ordering::Relaxed), 0);
    assert_eq!(
        executor
            .attempted_chunks
            .lock()
            .expect("chunked prefill attempt mutex poisoned")
            .len(),
        1,
        "a terminal minimum frontier must not be retried"
    );
    assert!(executor.tokens.frontier_for_request(&request_id).is_none());
    assert!(executor
        .retained
        .lock()
        .expect("chunked prefill retained mutex poisoned")
        .is_empty());
    assert!(!engine
        .active_inner()
        .expect("engine execution is open")
        .sequences
        .get(&request_id)
        .is_some());
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .execution_closes
            .unreclaimed_count(),
        0
    );
    assert_eq!(scheduler.active_count(), 0);
    assert_eq!(scheduler.waiting_count(), 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn plan_runtime_backing_maintenance_advances_epoch_before_prefill() {
    let trace_path = resource_trace_temp_path("executor-prefill-maintenance");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());
    config.runtime.profile_entrypoint = Some(ProfileEntrypoint::Serve);
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeMaintenanceTestExecutor::new(128);
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        tokenizer,
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.clone();

    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.response().finish_reason, FinishReason::Length);
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 2);
    assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        *executor
            .call_order
            .lock()
            .expect("call order mutex poisoned"),
        vec!["defer", "maintain", "admit", "prefill"]
    );
    assert!(executor
        .retained
        .lock()
        .expect("retained admission mutex poisoned")
        .is_empty());
    let snapshot = scheduler.trace_snapshot();
    assert_eq!(snapshot.legacy_waiting_admission_ticks, 0);
    assert_eq!(snapshot.dynamic_backing_growth_requested, 1);
    assert_eq!(snapshot.dynamic_admission_deferred, 1);

    flush_engine_profile_events(&engine);
    let events = read_engine_profile_events(&trace_path);
    assert!(
        events.iter().all(|event| {
            event
                .resource
                .as_ref()
                .is_none_or(|resource| resource.resource_kind != "backend_workspace")
        }),
        "PlanRuntime must not publish the legacy synthetic backend_workspace ledger"
    );
    let deferred_index = events
        .iter()
        .position(|event| {
            event.request_id == request_id.to_string()
                && event.phase == "vnext.prefill_admission"
                && event.shape.get("decision") == Some(&serde_json::json!("maintenance_deferred"))
        })
        .expect("maintenance deferral trace event");
    let maintenance_index = events
        .iter()
        .position(|event| {
            event.request_id == request_id.to_string()
                && event.phase == "vnext.prefill_backing_maintenance"
                && event.shape.get("outcome") == Some(&serde_json::json!("maintained"))
        })
        .expect("maintenance completion trace event");
    let admitted_index = events
        .iter()
        .position(|event| {
            event.request_id == request_id.to_string()
                && event.phase == "vnext.prefill_admission"
                && event.shape.get("decision") == Some(&serde_json::json!("admitted"))
        })
        .expect("admitted trace event");
    assert!(deferred_index < maintenance_index);
    assert!(maintenance_index < admitted_index);
    assert_eq!(
        events[deferred_index].shape.get("prefill_submit_observed"),
        Some(&serde_json::json!(false))
    );
    let input_token_count = events[deferred_index]
        .shape
        .get("input_token_count")
        .and_then(serde_json::Value::as_u64)
        .expect("typed prefill input token count");
    let maximum_sequence_tokens = events[deferred_index]
        .shape
        .get("maximum_sequence_tokens")
        .and_then(serde_json::Value::as_u64)
        .expect("typed maximum sequence horizon");
    assert!(input_token_count > 0);
    assert!(maximum_sequence_tokens >= input_token_count);
    assert_eq!(
        events[maintenance_index]
            .attributes
            .get("maintenance_evidence")
            .and_then(|evidence| evidence.get("rebalance")),
        Some(&serde_json::Value::Null)
    );
    assert!(events[deferred_index]
        .attributes
        .get("monotonic_nanos")
        .and_then(serde_json::Value::as_u64)
        .is_some());

    let _ = std::fs::remove_file(trace_path);
}

fn plan_runtime_wait_for_release_engine() -> (
    Arc<ContinuousBatchEngine>,
    Arc<ContinuousBatchScheduler>,
    Arc<PlanRuntimeMaintenanceObservation>,
) {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeMaintenanceTestExecutor::with_behavior(
        128,
        TestMaintenanceBehavior::WaitForRelease,
    );
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = Arc::new(
        ContinuousBatchEngine::new_plan_runtime(
            config,
            Arc::clone(&scheduler),
            tokenizer,
            sampler,
            executor_owner,
            tensor_factory,
        )
        .unwrap(),
    );
    (engine, scheduler, executor)
}

async fn wait_for_capacity_wait(executor: &PlanRuntimeMaintenanceObservation) {
    tokio::time::timeout(Duration::from_secs(1), async {
        while executor.maintenance_calls.load(Ordering::Acquire) == 0
            || executor.capacity_wait_registrations.load(Ordering::Acquire) == 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("typed capacity maintenance must park the request");
}

#[tokio::test]
async fn plan_runtime_capacity_pressure_waits_without_spinning_then_retries_on_release() {
    let (engine, scheduler, executor) = plan_runtime_wait_for_release_engine();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let infer_engine = Arc::clone(&engine);
    let inference = tokio::spawn(async move { infer_engine.infer(request).await });

    wait_for_capacity_wait(&executor).await;
    let parked_iteration = engine
        .active_inner()
        .expect("engine execution is open")
        .iteration_count
        .load(Ordering::Acquire);
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .iteration_count
            .load(Ordering::Acquire),
        parked_iteration,
        "a registered capacity wait must park the background loop"
    );
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 1);
    assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 0);
    assert_eq!(scheduler.waiting_count(), 1);
    assert_eq!(scheduler.trace_snapshot().dynamic_admission_failed, 0);

    executor.publish_release();
    let response = tokio::time::timeout(Duration::from_secs(2), inference)
        .await
        .expect("release epoch must wake the waiting request")
        .expect("inference task must not panic")
        .unwrap();
    assert_eq!(response.response().finish_reason, FinishReason::Length);
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 2);
    assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(
        *executor
            .call_order
            .lock()
            .expect("call order mutex poisoned"),
        vec!["defer", "maintain", "admit", "prefill"]
    );
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn plan_runtime_capacity_wait_wakes_and_cancels_when_stream_is_dropped() {
    let (engine, scheduler, executor) = plan_runtime_wait_for_release_engine();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.clone();
    let stream = engine.infer_stream(request).await.unwrap();

    wait_for_capacity_wait(&executor).await;
    let parked_iteration = engine
        .active_inner()
        .expect("engine execution is open")
        .iteration_count
        .load(Ordering::Acquire);
    drop(stream);

    tokio::time::timeout(Duration::from_secs(1), async {
        while engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .get(&request_id)
            .is_some()
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping the stream must wake cancellation without a capacity release");
    assert!(
        engine
            .active_inner()
            .expect("engine execution is open")
            .iteration_count
            .load(Ordering::Acquire)
            > parked_iteration
    );
    assert_eq!(scheduler.trace_phase(&request_id), None);
    assert_eq!(scheduler.trace_snapshot().cancelled_total, 1);
    assert_eq!(scheduler.waiting_count(), 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn plan_runtime_capacity_wait_wakes_and_cancels_when_sync_future_is_aborted() {
    let (engine, scheduler, executor) = plan_runtime_wait_for_release_engine();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.clone();
    let infer_engine = Arc::clone(&engine);
    let inference = tokio::spawn(async move { infer_engine.infer(request).await });

    wait_for_capacity_wait(&executor).await;
    inference.abort();
    assert!(inference
        .await
        .expect_err("inference task must be cancelled")
        .is_cancelled());

    tokio::time::timeout(Duration::from_secs(1), async {
        while engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .get(&request_id)
            .is_some()
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("aborting sync inference must wake cancellation without a capacity release");
    assert_eq!(scheduler.trace_phase(&request_id), None);
    assert_eq!(scheduler.trace_snapshot().cancelled_total, 1);
    assert_eq!(scheduler.waiting_count(), 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn plan_runtime_capacity_wait_accepts_new_work_and_shutdown_without_reprobing_old_work() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeMaintenanceTestExecutor::with_behavior(
        128,
        TestMaintenanceBehavior::WaitForRelease,
    );
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = Arc::new(
        ContinuousBatchEngine::new_plan_runtime(
            config,
            scheduler,
            tokenizer,
            sampler,
            executor_owner,
            tensor_factory,
        )
        .unwrap(),
    );
    let original_inner = Arc::downgrade(&engine.active_inner().expect("engine execution is open"));
    let background = engine.start_loop().unwrap();

    let mut first = policy_request();
    first.sampling_params.max_tokens = 1;
    let first_engine = Arc::clone(&engine);
    let first_task = tokio::spawn(async move { first_engine.infer(first).await });
    tokio::time::timeout(Duration::from_secs(1), async {
        while executor.capacity_wait_registrations.load(Ordering::Acquire) < 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("first request must park on capacity");
    assert_eq!(executor.admission_probes.load(Ordering::Acquire), 1);

    let mut second = policy_request();
    second.sampling_params.max_tokens = 1;
    let second_engine = Arc::clone(&engine);
    let second_task = tokio::spawn(async move { second_engine.infer(second).await });
    tokio::time::timeout(Duration::from_secs(1), async {
        while executor.admission_probes.load(Ordering::Acquire) < 2
            || executor.capacity_wait_registrations.load(Ordering::Acquire) < 2
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("new work must interrupt and then rebuild the aggregate capacity wait");
    assert_eq!(
        executor.admission_probes.load(Ordering::Acquire),
        2,
        "the first unchanged request must not be probed again"
    );
    assert_eq!(executor.prefill_calls.load(Ordering::Acquire), 0);

    engine.shutdown().await.unwrap();
    let report = tokio::time::timeout(Duration::from_secs(1), background.wait())
        .await
        .expect("shutdown must interrupt the capacity waiter");
    report
        .result()
        .expect("capacity-waiting background loop must not panic");
    assert_eq!(
        report
            .executor_shutdown()
            .expect("standalone Engine closes its original executor")
            .snapshot()
            .phase(),
        ferrum_execution::executor_lifecycle::ExecutorShutdownPhase::Retired,
    );
    for task in [first_task, second_task] {
        assert!(tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("shutdown must notify every parked request")
            .expect("inference task must not panic")
            .is_err());
    }
    assert!(matches!(
        engine.active_inner(),
        Err(FerrumError::Scheduler { message }) if message.contains("engine is closing")
    ));
    assert!(
        original_inner.upgrade().is_none(),
        "the original sequence registry and Engine scope are retired"
    );
    assert!(
        executor.tokens.0.upgrade().is_none(),
        "observation must not keep the original executor token state alive"
    );
    assert!(executor.retained.lock().unwrap().is_empty());
    assert_eq!(engine.scheduler.waiting_count(), 0);
    assert_eq!(engine.scheduler.trace_snapshot().active_len, 0);
    assert_eq!(executor.admission_probes.load(Ordering::Acquire), 2);
    assert_eq!(executor.prefill_calls.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn plan_runtime_retry_admission_does_not_require_a_published_epoch() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(ferrum_testkit::MockTokenizer::new(128));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
    let executor = PlanRuntimeMaintenanceTestExecutor::with_behavior(
        128,
        TestMaintenanceBehavior::RetryAdmission,
    );
    let observation = executor.observer();
    let executor_owner = ExecutorOwner::new(Box::new(executor));
    let executor = observation;
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new_plan_runtime(
        config,
        Arc::clone(&scheduler),
        tokenizer,
        sampler,
        executor_owner,
        tensor_factory,
    )
    .unwrap();
    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;

    let response = tokio::time::timeout(Duration::from_secs(2), engine.infer(request))
        .await
        .expect("physical recheck must force a bounded admission retry")
        .unwrap();

    assert_eq!(response.response().finish_reason, FinishReason::Length);
    assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 2);
    assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 1);
    assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 1);
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(scheduler.trace_snapshot().dynamic_admission_failed, 0);
    engine.shutdown().await.unwrap();
}

#[tokio::test]
async fn plan_runtime_invalid_maintenance_fails_waiting_request_without_retry_loop() {
    for behavior in [
        TestMaintenanceBehavior::Fail,
        TestMaintenanceBehavior::IncoherentRebalance,
    ] {
        let mut config = EngineConfig::default();
        config.kv_cache.max_blocks = 128;
        let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
        let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
            Arc::new(ferrum_testkit::MockTokenizer::new(128));
        let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(ferrum_testkit::MockSampler);
        let executor = PlanRuntimeMaintenanceTestExecutor::with_behavior(128, behavior);
        let observation = executor.observer();
        let executor_owner = ExecutorOwner::new(Box::new(executor));
        let executor = observation;
        let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
        let engine = ContinuousBatchEngine::new_plan_runtime(
            config,
            Arc::clone(&scheduler),
            tokenizer,
            sampler,
            executor_owner,
            tensor_factory,
        )
        .unwrap();
        let mut request = policy_request();
        request.sampling_params.max_tokens = 1;
        let request_id = request.id.clone();

        let error = tokio::time::timeout(Duration::from_secs(2), engine.infer(request))
            .await
            .expect("invalid maintenance must not leave the request waiting forever")
            .expect_err("maintenance failure must preserve the terminal error for the caller");

        match (behavior, &error) {
            (TestMaintenanceBehavior::Fail, FerrumError::Backend { message }) => {
                assert_eq!(message, "synthetic backing allocation failure");
            }
            (TestMaintenanceBehavior::IncoherentRebalance, FerrumError::Internal { message }) => {
                assert!(message.contains("reported growth without installed capacity"));
            }
            _ => panic!("maintenance returned an unrelated terminal error: {error:?}"),
        }
        assert_eq!(executor.admission_probes.load(Ordering::Relaxed), 1);
        assert_eq!(executor.maintenance_calls.load(Ordering::Relaxed), 1);
        assert_eq!(executor.prefill_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            executor.capacity_wait_registrations.load(Ordering::Relaxed),
            0
        );
        assert!(executor.tokens.frontier_for_request(&request_id).is_none());
        assert!(!engine
            .active_inner()
            .expect("engine execution is open")
            .sequences
            .get(&request_id)
            .is_some());
        assert_eq!(
            engine
                .active_inner()
                .expect("engine execution is open")
                .execution_closes
                .unreclaimed_count(),
            0
        );
        assert_eq!(scheduler.active_count(), 0);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.trace_snapshot().dynamic_admission_failed, 1);
        assert!(executor
            .retained
            .lock()
            .expect("retained admission mutex poisoned")
            .is_empty());
        engine.shutdown().await.unwrap();
    }
}

#[test]
#[should_panic(expected = "request slot lease dropped without explicit reject or close")]
fn request_slot_lease_drop_without_consumption_panics_in_tests() {
    let engine = test_continuous_engine();
    let _lease = RequestSlotLease::open(
        &engine.active_inner().expect("engine execution is open"),
        RequestId::new(),
    );
}

#[test]
#[should_panic(expected = "sequence state dropped with owned request slot")]
fn sequence_state_drop_with_owned_request_slot_panics_in_tests() {
    let engine = test_continuous_engine();
    let request = policy_request();
    let mut sequence = SequenceState::new(request.clone(), vec![TokenId::new(1)]);
    sequence.request_slot = Some(RequestSlotLease::open(
        &engine.active_inner().expect("engine execution is open"),
        request.id,
    ));
}

#[tokio::test]
async fn scheduler_trace_plan_stats_reports_request_details() {
    let engine = test_continuous_engine();
    let request = policy_request();
    let request_id = request.id.clone();

    engine.scheduler.submit(request.clone()).await.unwrap();
    let batch = engine
        .scheduler
        .next_batch(ferrum_interfaces::BatchHint {
            max_batch_size: 4,
            max_tokens: 4,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        })
        .await
        .expect("batch should schedule submitted request");

    let mut seq = SequenceState::new(
        request,
        vec![
            TokenId::new(10),
            TokenId::new(11),
            TokenId::new(12),
            TokenId::new(13),
        ],
    );
    seq.prefill_tokens_processed = 1;
    {
        let sequences = &engine
            .active_inner()
            .expect("engine execution is open")
            .sequences;
        sequences.insert(request_id.clone(), seq);
    }

    let stats = engine
        .active_inner()
        .expect("engine execution is open")
        .scheduler_trace_plan_stats(&batch);
    assert_eq!(stats.batch_size, 1);
    assert_eq!(stats.prefill_items, 1);
    assert_eq!(stats.prefill_tokens, 4);
    assert_eq!(stats.requests.len(), 1);

    let request_stats = &stats.requests[0];
    assert_eq!(request_stats.request_id, request_id.to_string());
    assert_eq!(request_stats.phase.as_deref(), Some("Prefilling"));
    assert_eq!(request_stats.scheduled_tokens, 4);
    assert_eq!(request_stats.prompt_tokens, Some(4));
    assert_eq!(request_stats.generated_tokens, Some(0));
    assert_eq!(request_stats.prefill_tokens_processed, Some(1));
    assert_eq!(request_stats.prefill_tokens_remaining_before, Some(3));
    assert_eq!(request_stats.is_final_prefill_chunk, Some(true));
}

#[tokio::test]
async fn scheduler_trace_jsonl_resource_events_balance_successful_infer() {
    let trace_path = resource_trace_temp_path("successful-infer");
    let _ = std::fs::remove_file(&trace_path);
    let mut config = EngineConfig::default();
    config.runtime.scheduler_trace_jsonl = Some(trace_path.clone());
    config.runtime.profile_entrypoint = Some(ProfileEntrypoint::Run);
    config.runtime.profile_detail = ObservabilityProfileDetail::Replay;
    config.kv_cache.max_blocks = 128;
    let engine = test_continuous_engine_with_config(config);

    let mut request = policy_request();
    request.sampling_params.max_tokens = 1;
    let request_id = request.id.clone();
    let response = engine.infer(request).await.unwrap();

    assert_eq!(response.response().request_id, request_id);
    assert_eq!(response.response().finish_reason, FinishReason::Length);

    engine.shutdown().await.unwrap();
    let resources = assert_engine_resource_trace_balanced(&trace_path);
    let saw = |kind: &str, action: ResourceAction| {
        resources
            .iter()
            .any(|resource| resource.resource_kind == kind && resource.action == action)
    };

    assert!(saw("request_slot", ResourceAction::RequestOpen));
    assert!(saw("request_slot", ResourceAction::Reserve));
    assert!(saw("request_slot", ResourceAction::Commit));
    assert!(saw("request_slot", ResourceAction::Release));
    assert!(saw("request_slot", ResourceAction::RequestClose));
    assert!(resources
        .iter()
        .all(|resource| resource.resource_kind == "request_slot"));

    let events = read_engine_profile_events(&trace_path);
    let request_open = events
        .iter()
        .find(|event| event.phase == "engine_request_open")
        .expect("request-open timeline event");
    let request_close = events
        .iter()
        .find(|event| event.phase == "engine_request_close")
        .expect("request-close timeline event");
    let terminal_evidence = events
        .iter()
        .find(|event| {
            event.request_id == request_id.to_string()
                && event.phase == "engine_sequence_terminal_evidence"
        })
        .expect("completion token evidence must be traced in replay mode");
    for event in [request_open, request_close] {
        assert!(event
            .attributes
            .get("monotonic_nanos")
            .and_then(serde_json::Value::as_u64)
            .is_some());
        assert!(event
            .attributes
            .get("active_sequence_count")
            .and_then(serde_json::Value::as_u64)
            .is_some());
        assert!(event
            .attributes
            .get("scheduler_snapshot")
            .and_then(serde_json::Value::as_object)
            .is_some());
        assert_eq!(
            event.attributes["active_sequence_count"],
            event.attributes["scheduler_snapshot"]["active_len"]
        );
    }
    assert_eq!(
        request_close.attributes["active_sequence_count"],
        serde_json::json!(0)
    );
    assert_eq!(
        request_close.attributes["scheduler_snapshot"]["active_len"],
        serde_json::json!(0)
    );
    assert_eq!(
        request_close.attributes["scheduler_snapshot"]["completed_total"],
        serde_json::json!(1)
    );
    assert_eq!(
        terminal_evidence.attributes["termination"],
        serde_json::json!("completed")
    );
    assert_eq!(
        terminal_evidence.attributes["finish_reason"],
        serde_json::json!("length")
    );
    assert_eq!(
        terminal_evidence.shape["generated_token_count"],
        serde_json::json!(1)
    );

    let _ = std::fs::remove_file(trace_path);
}

fn vnext_profile_test_event() -> (
    ferrum_runtime::RunId,
    ferrum_runtime::RequestIdentity,
    ferrum_runtime::ExecutionEvent,
) {
    vnext_profile_test_event_for_request("request.vnext.engine-profile-test")
}

fn vnext_profile_test_event_for_request(
    request_identity: &str,
) -> (
    ferrum_runtime::RunId,
    ferrum_runtime::RequestIdentity,
    ferrum_runtime::ExecutionEvent,
) {
    use ferrum_runtime::{
        ExecutionEvent, ExecutionEventDetail, ExecutionEventKind, ExecutionIdentityEnvelope,
        ExecutionIdentityParts, ExecutionPhase, MonotonicTimestamp, RequestIdentity, RunId, SpanId,
        EXECUTION_IDENTITY_VERSION,
    };

    let run_id = RunId::new("run.vnext.engine-profile-test").unwrap();
    let request_id = RequestIdentity::new(request_identity).unwrap();
    let event = ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: 1,
        },
        ExecutionPhase::Resolution,
        ExecutionEventKind::RequestAccepted,
        ExecutionIdentityEnvelope::new(ExecutionIdentityParts {
            version: EXECUTION_IDENTITY_VERSION,
            run_id: run_id.clone(),
            request_id: request_id.clone(),
            sequence: 1,
            plan_id: None,
            plan_hash: None,
            frame_id: None,
            node_invocation_id: None,
            node_id: None,
            operation_id: None,
            provider_id: None,
            device_id: None,
            resource_pool_id: None,
            resource_pool_identity_fingerprint: None,
            provisioning_run_id: None,
            provisioning_request_id: None,
            transaction_id: None,
            active_sequence_slot: None,
            admission_generation: None,
            activation_epoch: None,
            runtime_implementation_fingerprint: None,
            active_sequence_fingerprint: None,
            completed_sequence_fingerprint: None,
            aborted_sequence_fingerprint: None,
            resource_id: None,
            resource_generation: None,
            resource_batch_fingerprint: None,
            span_id: SpanId::new("vnext/request/engine-profile-test").unwrap(),
            parent_span_id: None,
            async_links: Vec::new(),
        })
        .unwrap(),
        ExecutionEventDetail::MonotonicClockAnchor {
            clock_source: "rust_std_instant".to_string(),
            wall_anchor_unix_nanos: 1_700_000_000_000_000_000,
            max_error_nanos: 400,
        },
    )
    .unwrap();
    (run_id, request_id, event)
}

fn vnext_profile_test_operation_event() -> ferrum_runtime::ExecutionEvent {
    use ferrum_runtime::{
        DeviceId, ExecutionEvent, ExecutionEventDetail, ExecutionEventKind, ExecutionFrameId,
        ExecutionIdentityEnvelope, ExecutionIdentityParts, ExecutionPhase, MonotonicTimestamp,
        NodeId, NodeInvocationId, OperationId, PlanHash, PlanId, ProviderId, RequestIdentity,
        ResourcePoolId, RunId, SpanId, TransactionId, EXECUTION_IDENTITY_VERSION,
    };

    let fingerprint = |digit: char| digit.to_string().repeat(64);
    let plan_hash: PlanHash = serde_json::from_value(serde_json::json!(fingerprint('0'))).unwrap();
    ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: 3,
        },
        ExecutionPhase::Execution,
        ExecutionEventKind::OperationSubmitted,
        ExecutionIdentityEnvelope::new(ExecutionIdentityParts {
            version: EXECUTION_IDENTITY_VERSION,
            run_id: RunId::new("run.vnext.engine-profile-test").unwrap(),
            request_id: RequestIdentity::new("request.vnext.engine-profile-test").unwrap(),
            sequence: 3,
            plan_id: Some(PlanId::new("plan/vnext/engine-profile-test").unwrap()),
            plan_hash: Some(plan_hash),
            frame_id: Some(ExecutionFrameId::try_from(1).unwrap()),
            node_invocation_id: Some(NodeInvocationId::try_from(1).unwrap()),
            node_id: Some(NodeId::new("node/vnext/engine-profile-test").unwrap()),
            operation_id: Some(OperationId::new("operation/vnext/engine-profile-test").unwrap()),
            provider_id: Some(ProviderId::new("provider/vnext/engine-profile-test").unwrap()),
            device_id: Some(DeviceId::new("device/vnext/engine-profile-test").unwrap()),
            resource_pool_id: Some(ResourcePoolId::try_from(7).unwrap()),
            resource_pool_identity_fingerprint: Some(fingerprint('3')),
            provisioning_run_id: Some(RunId::new("run.vnext.pool-provisioning").unwrap()),
            provisioning_request_id: Some(
                RequestIdentity::new("request.vnext.pool-provisioning").unwrap(),
            ),
            transaction_id: Some(
                TransactionId::new("transaction.vnext.pool-provisioning").unwrap(),
            ),
            active_sequence_slot: Some(1),
            admission_generation: Some(1),
            activation_epoch: Some(1),
            runtime_implementation_fingerprint: Some(fingerprint('1')),
            active_sequence_fingerprint: Some(fingerprint('2')),
            completed_sequence_fingerprint: None,
            aborted_sequence_fingerprint: None,
            resource_id: None,
            resource_generation: None,
            resource_batch_fingerprint: None,
            span_id: SpanId::new("vnext/request/engine-profile-test/operation/1").unwrap(),
            parent_span_id: Some(SpanId::new("vnext/request/engine-profile-test/node/1").unwrap()),
            async_links: vec![SpanId::new("vnext/request/engine-profile-test/async/1").unwrap()],
        })
        .unwrap(),
        ExecutionEventDetail::None,
    )
    .unwrap()
}

fn vnext_profile_test_node_event(
    kind: ferrum_runtime::ExecutionEventKind,
) -> ferrum_runtime::ExecutionEvent {
    use ferrum_runtime::{
        ExecutionEvent, ExecutionEventDetail, ExecutionEventKind, ExecutionIdentityEnvelope,
        ExecutionPhase, MonotonicTimestamp, SpanId,
    };

    assert!(matches!(
        kind,
        ExecutionEventKind::NodeStarted | ExecutionEventKind::NodeRetired
    ));
    let mut identity = vnext_profile_test_operation_event()
        .identity()
        .parts()
        .clone();
    identity.sequence = if kind == ExecutionEventKind::NodeStarted {
        2
    } else {
        4
    };
    identity.span_id = SpanId::new("vnext/request/engine-profile-test/node/1").unwrap();
    identity.parent_span_id =
        Some(SpanId::new("vnext/request/engine-profile-test/frame/1").unwrap());
    identity.async_links.clear();
    ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: identity.sequence,
        },
        ExecutionPhase::Execution,
        kind,
        ExecutionIdentityEnvelope::new(identity).unwrap(),
        ExecutionEventDetail::None,
    )
    .unwrap()
}

fn vnext_profile_test_resource_failure_event() -> ferrum_runtime::ExecutionEvent {
    use ferrum_runtime::{
        ExecutionEvent, ExecutionEventDetail, ExecutionEventKind, ExecutionIdentityEnvelope,
        ExecutionPhase, FailureDomain, FailureEnvelope, IdentifiedFailure, MonotonicTimestamp,
        SpanId,
    };

    let submitted = vnext_profile_test_operation_event();
    let failure = IdentifiedFailure::new(
        submitted.identity().clone(),
        FailureEnvelope::new(
            FailureDomain::Resource,
            "diagnostic_resource_after_submit",
            "injected resource failure after operation submission",
            false,
        )
        .unwrap()
        .with_resource_snapshot(
            PlanRuntimeResourceSnapshot::new(1_000, 900, 700, 700, 400, 300, 200, 0, 0).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut observation = submitted.identity().parts().clone();
    observation.sequence = 4;
    observation.parent_span_id = Some(observation.span_id.clone());
    observation.span_id = SpanId::new("vnext/request/engine-profile-test/failure/1").unwrap();

    ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: 4,
        },
        ExecutionPhase::Execution,
        ExecutionEventKind::FailureObserved,
        ExecutionIdentityEnvelope::new(observation).unwrap(),
        ExecutionEventDetail::Failure(failure),
    )
    .unwrap()
}

#[test]
fn vnext_profile_preserves_the_complete_canonical_execution_identity() {
    let trace_path = resource_trace_temp_path("vnext-complete-execution-identity");
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Latency;
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);

    let profile = sink
        .profile_event(&vnext_profile_test_operation_event())
        .unwrap();
    let identity = &profile.attributes["execution_identity"];

    assert_eq!(
        profile.attributes["execution_identity_version"],
        ferrum_runtime::EXECUTION_IDENTITY_VERSION.to_string()
    );
    assert_eq!(profile.attributes["resource_pool_id"], "resource-pool:7");
    assert_eq!(profile.attributes["active_sequence_slot"], 1);
    assert_eq!(profile.attributes["admission_generation"], 1);
    assert_eq!(profile.attributes["activation_epoch"], 1);
    assert_eq!(
        profile.attributes["async_links"].as_array().unwrap().len(),
        1
    );
    assert_eq!(identity["resource_pool_id"], 7);
    assert_eq!(identity["active_sequence_slot"], 1);
    assert_eq!(identity["resource_id"], serde_json::Value::Null);
    assert_eq!(identity["async_links"], profile.attributes["async_links"]);

    drop(sink);
    journal.close().unwrap();
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn basic_vnext_profile_serializes_identity_once_and_stays_within_request_budget() {
    assert_compact_vnext_profile(ObservabilityProfileDetail::Basic);
}

#[test]
fn off_vnext_profile_compacts_nodes_and_preserves_lifecycle_identity() {
    assert_compact_vnext_profile(ObservabilityProfileDetail::Off);
}

fn assert_compact_vnext_profile(detail: ObservabilityProfileDetail) {
    use ferrum_runtime::ExecutionEventKind;

    const NODES_PER_REQUEST: usize = 132;
    const LIFECYCLE_EVENTS_PER_REQUEST: usize = 6;
    const MAX_BYTES_PER_REQUEST: usize = 1024 * 1024;

    let trace_path = resource_trace_temp_path(&format!(
        "vnext-{}-compact-execution-identity",
        detail.as_str()
    ));
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = detail;
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Serve, &config);

    assert_eq!(
        sink.capture_policy(),
        ExecutionEventCapturePolicy::FirstFramePerRequest
    );
    assert_eq!(
        sink.device_timing_mode(),
        match detail {
            ObservabilityProfileDetail::Off => ferrum_runtime::DeviceTimingMode::Off,
            ObservabilityProfileDetail::Basic => ferrum_runtime::DeviceTimingMode::Completion,
            _ => unreachable!("this helper checks Off and Basic projections"),
        }
    );
    assert_eq!(
        sink.execution_facts_capture(),
        ferrum_runtime::ExecutionFactsCapture::Off
    );

    let operation = sink
        .profile_event(&vnext_profile_test_operation_event())
        .unwrap();
    operation.validate().unwrap();
    for field in [
        "execution_identity",
        "execution_request_id",
        "backend_device",
        "backend_type",
    ] {
        assert_eq!(
            operation.attributes.contains_key(field),
            detail == ObservabilityProfileDetail::Off,
            "{detail:?} operation projection changed {field}"
        );
    }
    assert_eq!(
        operation.attributes["execution_identity_version"],
        ferrum_runtime::EXECUTION_IDENTITY_VERSION.to_string()
    );
    assert_eq!(operation.request_id, "request.vnext.engine-profile-test");
    assert_eq!(operation.shape["execution_sequence"], 3);
    assert_eq!(operation.shape["frame_id"], 1);
    assert_eq!(operation.shape["node_invocation_id"], 1);
    for field in [
        "plan_id",
        "plan_hash",
        "node_id",
        "operation_id",
        "provider_id",
        "device_id",
        "resource_pool_id",
        "resource_pool_identity_fingerprint",
        "provisioning_run_id",
        "provisioning_request_id",
        "transaction_id",
        "runtime_implementation_fingerprint",
        "active_sequence_fingerprint",
        "run_id",
        "span_id",
        "parent_span_id",
        "async_links",
    ] {
        assert!(
            operation.attributes.contains_key(field),
            "basic operation identity omitted {field}"
        );
    }
    assert_eq!(
        operation.backend_detail.as_ref().unwrap()["backend_device"],
        serde_json::json!(format!("{:?}", config.backend.device))
    );

    let started = sink
        .profile_event(&vnext_profile_test_node_event(
            ExecutionEventKind::NodeStarted,
        ))
        .unwrap();
    let retired = sink
        .profile_event(&vnext_profile_test_node_event(
            ExecutionEventKind::NodeRetired,
        ))
        .unwrap();
    for event in [&started, &retired] {
        event.validate().unwrap();
        assert_eq!(event.request_id, operation.request_id);
        assert_eq!(event.shape["frame_id"], operation.shape["frame_id"]);
        assert_eq!(
            event.shape["node_invocation_id"],
            operation.shape["node_invocation_id"]
        );
        for field in [
            "plan_id",
            "plan_hash",
            "node_id",
            "operation_id",
            "provider_id",
            "device_id",
            "resource_pool_id",
            "active_sequence_slot",
            "admission_generation",
            "activation_epoch",
            "run_id",
            "span_id",
            "parent_span_id",
        ] {
            assert!(
                event.attributes.contains_key(field),
                "node linkage omitted {field}"
            );
        }
        for field in [
            "execution_identity",
            "execution_request_id",
            "backend_device",
            "backend_type",
            "resource_pool_identity_fingerprint",
            "provisioning_run_id",
            "provisioning_request_id",
            "transaction_id",
            "runtime_implementation_fingerprint",
            "active_sequence_fingerprint",
            "async_links",
        ] {
            assert!(
                !event.attributes.contains_key(field),
                "repeated node identity retained {field}"
            );
        }
    }

    let (_, _, accepted) = vnext_profile_test_event();
    let lifecycle = sink.profile_event(&accepted).unwrap();
    lifecycle.validate().unwrap();
    assert_eq!(lifecycle.phase, "vnext.request_accepted");
    assert_eq!(
        lifecycle.attributes["vnext_monotonic_wall_anchor_unix_nanos"],
        1_700_000_000_000_000_000_i64
    );
    assert_eq!(
        lifecycle.attributes.contains_key("execution_identity"),
        detail == ObservabilityProfileDetail::Off
    );
    let serialized_request_upper_bound = NODES_PER_REQUEST
        * (serde_json::to_vec(&started).unwrap().len()
            + serde_json::to_vec(&operation).unwrap().len()
            + serde_json::to_vec(&retired).unwrap().len())
        + LIFECYCLE_EVENTS_PER_REQUEST
            * serde_json::to_vec(&operation)
                .unwrap()
                .len()
                .max(serde_json::to_vec(&lifecycle).unwrap().len());
    if detail == ObservabilityProfileDetail::Basic {
        assert!(
            serialized_request_upper_bound <= MAX_BYTES_PER_REQUEST,
            "basic execution trace upper bound is {serialized_request_upper_bound} bytes"
        );
    }

    let failure = sink
        .profile_event(&vnext_profile_test_resource_failure_event())
        .unwrap();
    assert!(failure.attributes.contains_key("execution_identity"));
    assert!(failure.attributes.contains_key("execution_request_id"));

    let mut terminal_identity = accepted.identity().parts().clone();
    terminal_identity.sequence = 5;
    let terminal = ferrum_runtime::ExecutionEvent::new(
        ferrum_runtime::MonotonicTimestamp {
            nanos_since_run_start: 5,
        },
        ferrum_runtime::ExecutionPhase::Completion,
        ExecutionEventKind::RequestFailed,
        ferrum_runtime::ExecutionIdentityEnvelope::new(terminal_identity).unwrap(),
        ferrum_runtime::ExecutionEventDetail::FailureTerminal {
            first_failure_fingerprint: failure.attributes["first_failure_fingerprint"]
                .as_str()
                .unwrap()
                .to_owned(),
        },
    )
    .unwrap();
    let terminal = sink.profile_event(&terminal).unwrap();
    terminal.validate().unwrap();
    assert_eq!(terminal.phase, "vnext.request_failed");
    assert_eq!(terminal.status, ProfileStatus::Failure);
    assert_eq!(
        terminal.attributes["first_failure_fingerprint"],
        failure.attributes["first_failure_fingerprint"]
    );
    assert!(terminal.attributes.contains_key("execution_identity"));
    assert!(terminal.attributes.contains_key("execution_request_id"));

    drop(sink);
    journal.close().unwrap();
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn vnext_profile_preserves_real_resource_failure_evidence() {
    let trace_path = resource_trace_temp_path("vnext-resource-failure-evidence");
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Latency;
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Serve, &config);

    let profile = sink
        .profile_event(&vnext_profile_test_resource_failure_event())
        .unwrap();
    let resource = profile.resource.as_ref().expect("resource trace evidence");
    let snapshot = &profile.attributes["plan_runtime_resource_snapshot"];

    profile.validate().unwrap();
    assert_eq!(profile.phase, "vnext.failure_observed");
    assert_eq!(profile.status, ProfileStatus::Failure);
    assert_eq!(resource.action, ResourceAction::Reject);
    assert_eq!(resource.owner_kind, "resource_pool");
    assert_eq!(resource.owner_id, "resource-pool:7");
    assert_eq!(resource.before, Some(400));
    assert_eq!(resource.after, Some(400));
    assert_eq!(resource.capacity, Some(900));
    assert_eq!(snapshot["device_capacity_bytes"], 1_000);
    assert_eq!(snapshot["usable_capacity_bytes"], 900);
    assert_eq!(snapshot["dynamic_resident_bytes"], 300);
    assert_eq!(
        profile.error.as_ref().unwrap().kind,
        "diagnostic_resource_after_submit"
    );

    drop(sink);
    journal.close().unwrap();
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn vnext_execution_events_use_the_canonical_scheduler_trace_schema() {
    use ferrum_runtime::{ExecutionEventEmitter, TrustedExecutionEventContext};

    let trace_path = resource_trace_temp_path("vnext-execution-event");
    let _ = std::fs::remove_file(&trace_path);
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let config = EngineConfig::default();
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);
    assert_eq!(
        sink.capture_policy(),
        ExecutionEventCapturePolicy::FirstFramePerRequest
    );
    assert_eq!(
        sink.device_timing_mode(),
        ferrum_runtime::DeviceTimingMode::Off
    );
    assert_eq!(sink.capture_policy().as_str(), "first_frame_per_request");
    let (run_id, request_id, event) = vnext_profile_test_event();
    let mut emitter = ExecutionEventEmitter::new(&sink, run_id.clone(), request_id.clone());
    emitter
        .emit(
            event,
            &TrustedExecutionEventContext::pre_plan(&run_id, &request_id),
        )
        .unwrap();
    journal.flush().unwrap();

    let rows = std::fs::read_to_string(&trace_path).unwrap();
    let profile: FerrumProfileEvent = serde_json::from_str(rows.trim()).unwrap();
    profile.validate().unwrap();
    assert_eq!(profile.phase, "vnext.request_accepted");
    assert_eq!(profile.event_kind, ProfileEventKind::Instant);
    assert_eq!(profile.request_id, request_id.to_string());
    assert_eq!(
        profile.attributes.get("execution_trace_source"),
        Some(&serde_json::json!("vnext"))
    );
    assert_eq!(
        profile.attributes.get("execution_capture_policy"),
        Some(&serde_json::json!("first_frame_per_request"))
    );
    assert_eq!(
        profile.attributes.get("vnext_monotonic_clock_source"),
        Some(&serde_json::json!("rust_std_instant"))
    );
    assert_eq!(
        profile
            .attributes
            .get("vnext_monotonic_wall_anchor_unix_nanos"),
        Some(&serde_json::json!(1_700_000_000_000_000_000_i64))
    );
    assert_eq!(
        profile
            .attributes
            .get("vnext_clock_conversion_max_error_nanos"),
        Some(&serde_json::json!(400))
    );

    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn basic_vnext_profile_keeps_startup_lifecycle_without_repeated_frame_detail() {
    let trace_path = resource_trace_temp_path("vnext-startup-lifecycle-profile");
    let _ = std::fs::remove_file(&trace_path);
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Basic;
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);

    assert_eq!(
        sink.capture_policy_for_request(ExecutorRequestOrigin::Startup),
        ExecutionEventCapturePolicy::LifecycleOnly
    );
    assert_eq!(
        sink.capture_policy_for_request(ExecutorRequestOrigin::Product),
        ExecutionEventCapturePolicy::FirstFramePerRequest
    );
    let (_, _, startup) =
        vnext_profile_test_event_for_request("request.startup.engine-profile-test");
    let profile = sink.profile_event(&startup).unwrap();
    assert_eq!(
        profile.attributes.get("execution_request_origin"),
        Some(&serde_json::json!("startup"))
    );
    assert_eq!(
        profile.attributes.get("execution_capture_policy"),
        Some(&serde_json::json!("lifecycle_only"))
    );

    drop(sink);
    journal.close().unwrap();
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn staged_vnext_profiles_separate_resource_latency_and_kernel_costs() {
    let cases = [
        (
            ObservabilityProfileDetail::Resource,
            ExecutionEventCapturePolicy::AllFrames,
            ferrum_runtime::DeviceTimingMode::Off,
            false,
        ),
        (
            ObservabilityProfileDetail::Latency,
            ExecutionEventCapturePolicy::FirstFramePerRequest,
            ferrum_runtime::DeviceTimingMode::Off,
            false,
        ),
        (
            ObservabilityProfileDetail::Kernel,
            ExecutionEventCapturePolicy::AllFrames,
            ferrum_runtime::DeviceTimingMode::Kernel,
            true,
        ),
    ];

    for (detail, expected_capture_policy, expected_timing, diagnostic_only) in cases {
        let trace_path = resource_trace_temp_path(detail.as_str());
        let _ = std::fs::remove_file(&trace_path);
        let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
        let mut config = EngineConfig::default();
        config.runtime.profile_detail = detail;
        let sink =
            VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);

        assert_eq!(sink.capture_policy(), expected_capture_policy);
        assert_eq!(sink.device_timing_mode(), expected_timing);
        let (_, _, accepted) = vnext_profile_test_event();
        let profile = sink.profile_event(&accepted).unwrap();
        assert_eq!(
            profile.attributes.get("profile_detail"),
            Some(&serde_json::json!(detail.as_str()))
        );
        assert_eq!(
            profile.attributes.get("diagnostic_only"),
            Some(&serde_json::json!(diagnostic_only))
        );

        drop(sink);
        journal.close().unwrap();
        let _ = std::fs::remove_file(trace_path);
    }
}

#[test]
fn sequence_engine_timing_is_opt_in_and_matches_committed_tokens() {
    let input = vec![TokenId::new(1), TokenId::new(2)];
    let mut disabled = SequenceState::new(policy_request(), input.clone());
    disabled.generated_tokens.push(TokenId::new(3));
    disabled.record_generated_token_commit();
    assert!(disabled.take_execution_evidence().unwrap().is_none());

    let mut request = policy_request();
    request.evidence_request.capture_prompt_token_ids = true;
    request.evidence_request.capture_engine_token_timing = true;
    let mut enabled = SequenceState::new(request, input.clone());
    enabled.record_decode_ready();
    enabled.generated_tokens.push(TokenId::new(3));
    enabled.record_generated_token_commit();
    let execution_started_at = Instant::now();
    enabled.close_decode_scheduling(execution_started_at);
    let execution_ended_at = Instant::now();
    enabled.record_decode_execution(execution_started_at, execution_ended_at);
    enabled.generated_tokens.push(TokenId::new(4));
    enabled.record_generated_token_commit_with_decode_postprocess(Some(execution_ended_at));

    let evidence = enabled
        .take_execution_evidence()
        .unwrap()
        .expect("requested execution evidence");
    assert_eq!(evidence.prompt_token_ids, input);
    assert_eq!(
        evidence.output_token_ids,
        vec![TokenId::new(3), TokenId::new(4)]
    );
    let timing = evidence.engine_token_timing.expect("engine token timing");
    timing.validate(2).unwrap();
    assert_eq!(timing.token_commit_nanos_since_request_start.len(), 2);
    assert!(
        timing.token_commit_nanos_since_request_start[1]
            >= timing.token_commit_nanos_since_request_start[0]
    );
    assert!(timing.decode_wall_nanos().is_some());
    assert_eq!(timing.decode_stage_intervals.len(), 3);
    assert_eq!(
        timing.decode_stage_intervals[0].stage,
        EngineDecodeStage::DecodeScheduling
    );
    assert_eq!(
        timing.decode_stage_intervals[1].stage,
        EngineDecodeStage::DecodeExecution
    );
    assert_eq!(
        timing.decode_stage_intervals[2].stage,
        EngineDecodeStage::DecodePostprocess
    );
    assert_eq!(
        timing.decode_stage_intervals[2].end_nanos_since_request_start,
        timing.token_commit_nanos_since_request_start[1]
    );
}

#[test]
fn vnext_profile_marks_only_failure_observed_as_blocking_first_failure() {
    use ferrum_runtime::{
        ExecutionEvent, ExecutionEventDetail, ExecutionEventKind, ExecutionIdentityEnvelope,
        ExecutionPhase, FailureDomain, FailureEnvelope, IdentifiedFailure, MonotonicTimestamp,
    };

    let trace_path = resource_trace_temp_path("vnext-first-failure");
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let config = EngineConfig::default();
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);
    let (_, _, accepted) =
        vnext_profile_test_event_for_request("request.product.engine-profile-test");
    let mut observed_parts = accepted.identity().parts().clone();
    observed_parts.sequence = 2;
    let observed_identity = ExecutionIdentityEnvelope::new(observed_parts).unwrap();
    let failure = IdentifiedFailure::new(
        observed_identity.clone(),
        FailureEnvelope::new(
            FailureDomain::Product,
            "capacity_exhausted",
            "capacity exhausted before admission",
            false,
        )
        .unwrap(),
    )
    .unwrap();
    let fingerprint = failure.fingerprint();
    let observed = ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: 2,
        },
        ExecutionPhase::Resolution,
        ExecutionEventKind::FailureObserved,
        observed_identity,
        ExecutionEventDetail::Failure(failure),
    )
    .unwrap();

    let mut terminal_parts = accepted.identity().parts().clone();
    terminal_parts.sequence = 3;
    let terminal = ExecutionEvent::new(
        MonotonicTimestamp {
            nanos_since_run_start: 3,
        },
        ExecutionPhase::Resolution,
        ExecutionEventKind::RequestFailed,
        ExecutionIdentityEnvelope::new(terminal_parts).unwrap(),
        ExecutionEventDetail::FailureTerminal {
            first_failure_fingerprint: fingerprint.clone(),
        },
    )
    .unwrap();

    let observed_profile = sink.profile_event(&observed).unwrap();
    assert!(observed_profile.error.as_ref().unwrap().blocking);
    assert_eq!(
        observed_profile.attributes["first_failure_event"],
        serde_json::json!(true)
    );
    assert_eq!(
        observed_profile.attributes["first_failure_fingerprint"],
        serde_json::json!(fingerprint)
    );
    assert_eq!(
        observed_profile.attributes["failure_domain"],
        serde_json::json!("product")
    );
    assert_eq!(
        observed_profile.attributes["execution_request_origin"],
        serde_json::json!("product")
    );
    assert_eq!(
        observed_profile.attributes["execution_request_id"],
        serde_json::json!("request.product.engine-profile-test")
    );

    let terminal_profile = sink.profile_event(&terminal).unwrap();
    assert!(!terminal_profile.error.as_ref().unwrap().blocking);
    assert!(terminal_profile
        .attributes
        .get("first_failure_event")
        .is_none());
    assert_eq!(
        terminal_profile.attributes["terminal_failure_event"],
        serde_json::json!(true)
    );
    drop(sink);
    journal.close().unwrap();
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn vnext_execution_events_fan_out_to_profile_and_scheduler_journals() {
    let profile_path = resource_trace_temp_path("vnext-profile-fanout");
    let scheduler_path = resource_trace_temp_path("vnext-scheduler-fanout");
    let _ = std::fs::remove_file(&profile_path);
    let _ = std::fs::remove_file(&scheduler_path);
    let profile_journal = create_scheduler_trace_sink(Some(&profile_path)).unwrap();
    let scheduler_journal = create_scheduler_trace_sink(Some(&scheduler_path)).unwrap();
    let config = EngineConfig::default();
    let sink = VNextProfileExecutionEventSink::with_journals(
        vec![profile_journal.clone(), scheduler_journal.clone()],
        ProfileEntrypoint::Run,
        &config,
    );
    let (_, _, event) = vnext_profile_test_event();

    sink.enqueue_events(vec![event]).unwrap();
    profile_journal.flush().unwrap();
    scheduler_journal.flush().unwrap();

    let profile_events = read_engine_profile_events(&profile_path);
    let scheduler_events = read_engine_profile_events(&scheduler_path);
    assert_eq!(profile_events, scheduler_events);
    assert_eq!(profile_events.len(), 1);
    assert_eq!(profile_events[0].phase, "vnext.request_accepted");

    profile_journal.close().unwrap();
    scheduler_journal.close().unwrap();
    let _ = std::fs::remove_file(profile_path);
    let _ = std::fs::remove_file(scheduler_path);
}

#[test]
fn full_vnext_profile_enables_kernel_attribution_and_all_frames() {
    let trace_path = resource_trace_temp_path("vnext-full-profile");
    let _ = std::fs::remove_file(&trace_path);
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Full;
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);

    assert_eq!(
        sink.capture_policy(),
        ExecutionEventCapturePolicy::AllFrames
    );
    assert_eq!(
        sink.device_timing_mode(),
        ferrum_runtime::DeviceTimingMode::Kernel
    );
    let (_, _, accepted) = vnext_profile_test_event();
    let profile = sink.profile_event(&accepted).unwrap();
    assert_eq!(
        profile.attributes.get("profile_detail"),
        Some(&serde_json::json!("full"))
    );
    assert_eq!(
        profile.attributes.get("diagnostic_only"),
        Some(&serde_json::json!(true))
    );

    drop(sink);
    journal.close().unwrap();
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn replay_vnext_profile_preserves_physical_timing_without_kernel_attribution() {
    let trace_path = resource_trace_temp_path("vnext-replay-profile");
    let _ = std::fs::remove_file(&trace_path);
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Replay;
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);

    assert_eq!(
        sink.capture_policy(),
        ExecutionEventCapturePolicy::AllFrames
    );
    assert_eq!(
        sink.device_timing_mode(),
        ferrum_runtime::DeviceTimingMode::Replay
    );
    assert!(sink
        .device_timing_mode()
        .physical_span_attribution_enabled());
    assert!(!sink.device_timing_mode().kernel_attribution_enabled());

    drop(sink);
    journal.close().unwrap();
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn verification_vnext_profile_requires_eager_kernel_attribution() {
    let trace_path = resource_trace_temp_path("vnext-verification-profile");
    let _ = std::fs::remove_file(&trace_path);
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Verify;
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);

    assert_eq!(
        sink.capture_policy(),
        ExecutionEventCapturePolicy::AllFrames
    );
    assert_eq!(
        sink.device_timing_mode(),
        ferrum_runtime::DeviceTimingMode::Verification
    );
    assert!(sink.device_timing_mode().kernel_attribution_enabled());
    assert!(!sink
        .device_timing_mode()
        .direct_reusable_execution_allowed());

    drop(sink);
    journal.close().unwrap();
    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn replay_device_timing_is_owned_once_by_its_physical_span() {
    use ferrum_runtime::{DeviceExecutionInterval, DeviceExecutionIntervalKind};

    let eager = DeviceSubmissionExecutionSpan::measured(
        0,
        1,
        DeviceExecutionSpanKind::EagerCommand,
        vec![DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Transfer, 0, 10).unwrap()],
    )
    .unwrap();
    let replay = DeviceSubmissionExecutionSpan::measured(
        1,
        4,
        DeviceExecutionSpanKind::ReusableExecutable,
        vec![DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Compute, 10, 40).unwrap()],
    )
    .unwrap()
    .with_reusable_executable_fingerprint("a".repeat(64))
    .unwrap();
    let timing = DeviceSubmissionExecutionTiming::from_spans(4, vec![eager, replay]).unwrap();
    let terminal = DeviceTimingMeasurement::Measured(timing);

    let eager_projection = project_device_command_timing(&terminal, 0);
    assert_eq!(eager_projection.status, "measured");
    assert_eq!(
        eager_projection
            .command_measurement
            .and_then(DeviceExecutionSpanMeasurement::elapsed_ns),
        Some(10)
    );

    for command_index in 1..4 {
        let replay_projection = project_device_command_timing(&terminal, command_index);
        assert_eq!(replay_projection.status, "covered_by_physical_span");
        assert!(replay_projection.command_measurement.is_none());
        assert_eq!(
            replay_projection
                .physical_span
                .and_then(|span| span.measurement().elapsed_ns()),
            Some(30)
        );
        assert_eq!(
            replay_projection
                .physical_span
                .and_then(DeviceSubmissionExecutionSpan::reusable_executable_fingerprint),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        );
    }

    let DeviceTimingMeasurement::Measured(timing) = &terminal else {
        unreachable!()
    };
    assert_eq!(
        timing
            .spans()
            .iter()
            .filter_map(|span| span.measurement().elapsed_ns())
            .sum::<u64>(),
        40
    );
}

#[test]
fn scheduler_trace_journal_preserves_ready_and_deferred_fifo_order() {
    let trace_path = resource_trace_temp_path("vnext-deferred-fifo");
    let _ = std::fs::remove_file(&trace_path);
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let config = EngineConfig::default();
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);
    let (_, _, accepted) = vnext_profile_test_event();
    let operation = vnext_profile_test_operation_event();

    journal
        .enqueue(sink.profile_event(&accepted).unwrap())
        .unwrap();
    sink.enqueue_events(vec![operation]).unwrap();
    journal
        .enqueue(sink.profile_event(&accepted).unwrap())
        .unwrap();
    journal.flush().unwrap();

    let phases = std::fs::read_to_string(&trace_path)
        .unwrap()
        .lines()
        .map(|line| {
            serde_json::from_str::<FerrumProfileEvent>(line)
                .unwrap()
                .phase
        })
        .collect::<Vec<_>>();
    assert_eq!(
        phases,
        [
            "vnext.request_accepted",
            "vnext.operation_submitted",
            "vnext.request_accepted",
        ]
    );

    let _ = std::fs::remove_file(trace_path);
}

#[test]
#[ignore = "release-mode diagnostic for the canonical trace producer"]
fn vnext_profile_event_producer_cost_probe() {
    const EVENTS_PER_FRAME: usize = 399;
    const SAMPLE_COUNT: usize = 11;

    let trace_path = resource_trace_temp_path("vnext-profile-producer-cost");
    let _ = std::fs::remove_file(&trace_path);
    let journal = create_scheduler_trace_sink(Some(&trace_path)).unwrap();
    let config = EngineConfig::default();
    let sink =
        VNextProfileExecutionEventSink::new(journal.clone(), ProfileEntrypoint::Run, &config);
    let event = vnext_profile_test_operation_event();
    let mut materialize_nanos = Vec::with_capacity(SAMPLE_COUNT);
    let mut enqueue_nanos = Vec::with_capacity(SAMPLE_COUNT);
    let mut deferred_enqueue_nanos = Vec::with_capacity(SAMPLE_COUNT);

    for _ in 0..SAMPLE_COUNT {
        let started = std::time::Instant::now();
        let mut profiles = Vec::with_capacity(EVENTS_PER_FRAME);
        for _ in 0..EVENTS_PER_FRAME {
            profiles.push(
                sink.profile_event(std::hint::black_box(&event))
                    .expect("profile event materialization must remain valid"),
            );
        }
        materialize_nanos.push(started.elapsed().as_nanos());

        let started = std::time::Instant::now();
        journal
            .enqueue_batch(std::hint::black_box(profiles))
            .expect("profile batch enqueue must succeed");
        enqueue_nanos.push(started.elapsed().as_nanos());
    }

    let deferred_batches = (0..SAMPLE_COUNT)
        .map(|_| vec![event.clone(); EVENTS_PER_FRAME])
        .collect::<Vec<_>>();
    for events in deferred_batches {
        let started = std::time::Instant::now();
        sink.enqueue_events(std::hint::black_box(events))
            .expect("deferred profile batch enqueue must succeed");
        deferred_enqueue_nanos.push(started.elapsed().as_nanos());
    }

    let started = std::time::Instant::now();
    journal.flush().unwrap();
    let drain_nanos = started.elapsed().as_nanos();
    materialize_nanos.sort_unstable();
    enqueue_nanos.sort_unstable();
    deferred_enqueue_nanos.sort_unstable();
    let median_index = SAMPLE_COUNT / 2;
    let receipt = serde_json::json!({
        "schema_version": 1,
        "artifact_type": "runtime_vnext_trace_producer_cost",
        "representative_event": "operation_submitted",
        "events_per_frame": EVENTS_PER_FRAME,
        "samples": SAMPLE_COUNT,
        "materialize_validate_median_ns_per_frame": materialize_nanos[median_index],
        "materialize_validate_median_ns_per_event": materialize_nanos[median_index]
            / EVENTS_PER_FRAME as u128,
        "enqueue_median_ns_per_frame": enqueue_nanos[median_index],
        "deferred_capture_enqueue_median_ns_per_frame": deferred_enqueue_nanos[median_index],
        "producer_cost_reduction_ratio": materialize_nanos[median_index] as f64
            / deferred_enqueue_nanos[median_index] as f64,
        "writer_drain_ns": drain_nanos,
        "diagnostic_only": true,
    });
    println!("FERRUM RUNTIME VNEXT TRACE PRODUCER COST: {receipt}");

    let _ = std::fs::remove_file(trace_path);
}

#[test]
fn request_context_capacity_uses_executor_kv_capacity_when_smaller() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 2048;
    let runtime = ContinuousEngineRuntimeConfig::from_env_vars(None, Vec::<(&str, &str)>::new());

    assert_eq!(
        effective_request_context_capacity(&config, &runtime, Some(512)),
        Some(512)
    );
}

#[test]
fn advertised_context_capacity_matches_the_request_admission_boundary() {
    let mut config = EngineConfig::default();
    config.runtime.max_model_len = Some(512);
    let engine = test_continuous_engine_with_config(config);
    let capacity = engine.context_capacity().expect("known engine capacity");
    assert!(capacity <= 512);
    let mut request = InferenceRequest::new("test", "test").with_sampling_params(SamplingParams {
        max_tokens: 1,
        ..SamplingParams::default()
    });
    validate_request_context_budget(
        &request,
        capacity - 1,
        &engine.config,
        &engine
            .active_inner()
            .expect("engine execution is open")
            .runtime_config,
        engine
            .active_inner()
            .expect("engine execution is open")
            .model_executor
            .kv_capacity(),
    )
    .expect("advertised capacity must be accepted");
    request.sampling_params.max_tokens = 2;
    assert!(matches!(
        validate_request_context_budget(
            &request,
            capacity - 1,
            &engine.config,
            &engine.active_inner().expect("engine execution is open").runtime_config,
            engine.active_inner().expect("engine execution is open").model_executor.kv_capacity(),
        ),
        Err(FerrumError::ContextLengthExceeded { capacity: actual, .. }) if actual == capacity
    ));
}

#[test]
fn explicit_request_budget_accepts_exact_capacity_and_rejects_one_token_over() {
    let capacity = 512;
    let prompt_tokens = 59;
    let config = EngineConfig::default();
    let runtime = ContinuousEngineRuntimeConfig::from_env_vars(None, Vec::<(&str, &str)>::new());
    let mut request = InferenceRequest::new("test", "test").with_sampling_params(SamplingParams {
        max_tokens: capacity - prompt_tokens,
        ..SamplingParams::default()
    });

    validate_request_context_budget(&request, prompt_tokens, &config, &runtime, Some(capacity))
        .expect("an explicit budget may exactly fill the available context");

    request.sampling_params.max_tokens += 1;
    clamp_default_max_tokens_to_context(
        &mut request,
        prompt_tokens,
        &config,
        &runtime,
        Some(capacity),
    );
    assert_eq!(
        request.sampling_params.max_tokens,
        capacity - prompt_tokens + 1
    );
    let error =
        validate_request_context_budget(&request, prompt_tokens, &config, &runtime, Some(capacity))
            .expect_err("one explicit output token beyond the context must be rejected");
    assert!(matches!(
        error,
        FerrumError::ContextLengthExceeded {
            capacity: 512,
            input_tokens: 59,
            output_tokens: 454,
        }
    ));
}

#[test]
fn opencode_sized_prompt_rejects_an_explicit_4096_context() {
    const INPUT_TOKENS: usize = 7_470;
    const OUTPUT_TOKENS: usize = 512;

    let config = EngineConfig::default();
    let runtime = ContinuousEngineRuntimeConfig::from_env_vars(None, [(MAX_MODEL_LEN_ENV, "4096")]);
    let request = InferenceRequest::new("test", "test").with_sampling_params(SamplingParams {
        max_tokens: OUTPUT_TOKENS,
        ..SamplingParams::default()
    });

    let error =
        validate_request_context_budget(&request, INPUT_TOKENS, &config, &runtime, Some(262_144))
            .expect_err("an explicit 4096-token product ceiling cannot admit the prompt");

    assert!(error.to_string().contains(
        "This model context is limited to 4096 tokens, but this request needs 7470 input tokens + 512 output tokens"
    ));
}

#[test]
fn opencode_sized_prompt_fits_default_model_context_and_explicit_8192_context() {
    const INPUT_TOKENS: usize = 7_470;
    const OUTPUT_TOKENS: usize = 512;

    let config = EngineConfig::default();
    let request = InferenceRequest::new("test", "test").with_sampling_params(SamplingParams {
        max_tokens: OUTPUT_TOKENS,
        ..SamplingParams::default()
    });
    let default_runtime =
        ContinuousEngineRuntimeConfig::from_env_vars(None, Vec::<(&str, &str)>::new());
    validate_request_context_budget(
        &request,
        INPUT_TOKENS,
        &config,
        &default_runtime,
        Some(262_144),
    )
    .expect("the model-derived default context must admit the prompt");

    let bounded_runtime =
        ContinuousEngineRuntimeConfig::from_env_vars(None, [(MAX_MODEL_LEN_ENV, "8192")]);
    validate_request_context_budget(
        &request,
        INPUT_TOKENS,
        &config,
        &bounded_runtime,
        Some(262_144),
    )
    .expect("an explicit 8192-token context must admit the prompt and output budget");
}

#[test]
fn opencode_sized_prompt_clamps_only_an_automatic_output_budget() {
    const INPUT_TOKENS: usize = 7_470;

    let config = EngineConfig::default();
    let runtime = ContinuousEngineRuntimeConfig::from_env_vars(None, [(MAX_MODEL_LEN_ENV, "8192")]);
    let mut request = InferenceRequest::new("test", "test")
        .with_sampling_params(SamplingParams {
            max_tokens: 4_096,
            ..SamplingParams::default()
        })
        .with_metadata(
            DEFAULT_MAX_TOKENS_METADATA_KEY.to_string(),
            serde_json::json!(true),
        );

    clamp_default_max_tokens_to_context(
        &mut request,
        INPUT_TOKENS,
        &config,
        &runtime,
        Some(262_144),
    );

    assert_eq!(request.sampling_params.max_tokens, 722);
    validate_request_context_budget(&request, INPUT_TOKENS, &config, &runtime, Some(262_144))
        .unwrap();
}

#[test]
fn automatic_output_budget_clamps_to_remaining_executor_capacity() {
    let config = EngineConfig::default();
    let runtime = ContinuousEngineRuntimeConfig::from_env_vars(None, Vec::<(&str, &str)>::new());
    let mut request = InferenceRequest::new("test", "test")
        .with_sampling_params(SamplingParams {
            max_tokens: 4096,
            ..SamplingParams::default()
        })
        .with_metadata(
            DEFAULT_MAX_TOKENS_METADATA_KEY.to_string(),
            serde_json::json!(true),
        );

    clamp_default_max_tokens_to_context(&mut request, 100, &config, &runtime, Some(4096));

    assert_eq!(request.sampling_params.max_tokens, 3996);
    validate_request_context_budget(&request, 100, &config, &runtime, Some(4096)).unwrap();
}

#[test]
fn explicit_output_budget_is_not_silently_clamped() {
    let config = EngineConfig::default();
    let runtime = ContinuousEngineRuntimeConfig::from_env_vars(None, Vec::<(&str, &str)>::new());
    let mut request = InferenceRequest::new("test", "test").with_sampling_params(SamplingParams {
        max_tokens: 4096,
        ..SamplingParams::default()
    });

    clamp_default_max_tokens_to_context(&mut request, 100, &config, &runtime, Some(4096));

    assert_eq!(request.sampling_params.max_tokens, 4096);
    let error = validate_request_context_budget(&request, 100, &config, &runtime, Some(4096))
        .expect_err("explicit output budget must remain a hard request contract");
    assert!(error
        .to_string()
        .contains("100 input tokens + 4096 output tokens"));
}

#[test]
fn test_sequence_state() {
    let request = InferenceRequest {
        id: RequestId::new(),
        prompt: "test".to_string(),
        model_id: ferrum_types::ModelId::new("test"),
        sampling_params: SamplingParams::default(),
        stream: false,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: chrono::Utc::now(),
        ingress: None,
        api_request: None,
        evidence_request: Default::default(),
        metadata: HashMap::new(),
    };

    let tokens = vec![TokenId::new(1), TokenId::new(2)];
    let state = SequenceState::new(request, tokens);

    assert_eq!(state.phase, RequestPhase::Waiting);
    assert_eq!(state.total_tokens(), 2);
    assert!(!state.prefill_complete);
}

#[test]
fn sequence_terminal_token_trace_is_bounded_and_stable() {
    let request = policy_request();
    let prompt = (0..300).map(TokenId::new).collect::<Vec<_>>();
    let mut state = SequenceState::new(request, prompt);
    state.generated_tokens = (1_000..1_400).map(TokenId::new).collect();

    let first = SequenceTokenTraceEvidence::capture(&state);
    let second = SequenceTokenTraceEvidence::capture(&state);

    assert_eq!(first, second);
    assert_eq!(first.prompt_token_count, 300);
    assert_eq!(
        first.prompt_token_prefix.len(),
        TOKEN_TRACE_PROMPT_PREFIX_LIMIT
    );
    assert_eq!(first.prompt_token_prefix.first(), Some(&0));
    assert_eq!(first.prompt_token_tail.len(), TOKEN_TRACE_PROMPT_TAIL_LIMIT);
    assert_eq!(first.prompt_token_tail.first(), Some(&172));
    assert_eq!(first.prompt_token_tail.last(), Some(&299));
    assert_eq!(first.generated_token_count, 400);
    assert_eq!(
        first.generated_token_prefix.len(),
        TOKEN_TRACE_GENERATED_PREFIX_LIMIT
    );
    assert_eq!(first.generated_token_prefix.first(), Some(&1_000));
    assert_eq!(first.generated_token_prefix.last(), Some(&1_255));
    assert_eq!(
        first.generated_token_tail.len(),
        TOKEN_TRACE_GENERATED_TAIL_LIMIT
    );
    assert_eq!(first.generated_token_tail.first(), Some(&1_368));
    assert_eq!(first.generated_token_tail.last(), Some(&1_399));

    state.generated_tokens[0] = TokenId::new(2_000);
    let changed = SequenceTokenTraceEvidence::capture(&state);
    assert_ne!(first.generated_token_sha256, changed.generated_token_sha256);
}

#[test]
fn sequence_state_detects_closed_run_and_serve_receivers() {
    let mut run_sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
    let (response_sender, response_receiver) = tokio::sync::oneshot::channel();
    run_sequence.response_sender = Some(response_sender);
    assert!(!run_sequence.client_receiver_closed());
    drop(response_receiver);
    assert!(run_sequence.client_receiver_closed());

    let mut serve_sequence = SequenceState::new(policy_request(), vec![TokenId::new(1)]);
    let budget = OutputBudget::new(ferrum_types::OutputConfig::default()).unwrap();
    let (stream_sender, stream_receiver) = OutputSender::new(
        Weak::new(),
        serve_sequence.original_request.id.clone(),
        budget.try_open_session().unwrap(),
        1,
    );
    serve_sequence.stream_sender = Some(stream_sender);
    assert!(!serve_sequence.client_receiver_closed());
    drop(stream_receiver);
    assert!(serve_sequence.client_receiver_closed());
}

#[tokio::test]
async fn engine_status_reports_kv_cache_memory_from_manager_stats() {
    let mut config = EngineConfig::default();
    config.kv_cache.max_blocks = 128;
    let scheduler = Arc::new(ContinuousBatchScheduler::new(config.scheduler.clone()));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(64, &[]));
    let sampler: Arc<dyn Sampler + Send + Sync> = Arc::new(crate::registry::GreedySampler);
    let kv_cache = Arc::new(MockKvCacheManager::new(128));
    kv_cache
        .allocate(&AllocationRequest {
            request_id: RequestId::new(),
            initial_tokens: 17,
            max_sequence_length: 32,
            num_layers: 1,
            num_heads: 1,
            head_dim: 1,
            device: Device::CPU,
            dtype: DataType::FP16,
            priority: Priority::Normal,
        })
        .await
        .unwrap();
    let executor = ExecutorOwner::new(Box::new(MockModelExecutor::instant(64)));
    let tensor_factory: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
    let engine = ContinuousBatchEngine::new(
        config,
        scheduler,
        tokenizer,
        sampler,
        kv_cache,
        executor,
        tensor_factory,
    )
    .expect("legacy engine composition must match executor authority");

    let status = engine.status().await;

    assert_eq!(status.memory_usage.total_bytes, 128 * 16 * 1024);
    assert_eq!(status.memory_usage.used_bytes, 2 * 16 * 1024);
    assert_eq!(status.memory_usage.cache_memory_bytes, 2 * 16 * 1024);
    assert_eq!(
        status.memory_usage.free_bytes,
        status.memory_usage.total_bytes - status.memory_usage.used_bytes
    );
    assert_eq!(status.memory_usage.cpu_memory_bytes, Some(2 * 16 * 1024));
    assert_eq!(status.memory_usage.gpu_memory_bytes, None);
    assert!(status.memory_usage.utilization_percent > 0.0);
}

#[test]
fn sequence_stop_reason_distinguishes_user_stop_from_natural_eos() {
    let mut tokenizer = PolicyTokenizer::new(
        8,
        &[
            ("prompt", 0),
            ("<eos>", 3),
            ("STOP", 6),
            ("answerSTOPtail", 7),
        ],
    );
    tokenizer.special.extra_eos_tokens = vec![TokenId::new(7)];
    tokenizer.hidden_decode_token_ids.insert(3);
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    assert_eq!(tokenizer.decode(&[TokenId::new(3)], true).unwrap(), "");
    for (label, ignore_eos, stops, token, max_tokens, expected) in [
        (
            "natural EOS precedes length",
            false,
            vec![],
            3,
            1,
            Some(FinishReason::EOS),
        ),
        ("ignored natural EOS", true, vec![], 3, 2, None),
        (
            "ignored EOS reaches length",
            true,
            vec![],
            3,
            1,
            Some(FinishReason::Length),
        ),
        (
            "user stop precedes length",
            false,
            vec!["STOP"],
            6,
            1,
            Some(FinishReason::Stop),
        ),
        (
            "hidden EOS is explicitly a user stop",
            false,
            vec!["<eos>"],
            3,
            1,
            Some(FinishReason::Stop),
        ),
        (
            "ignore EOS preserves colliding user stop",
            true,
            vec!["<eos>"],
            3,
            1,
            Some(FinishReason::Stop),
        ),
        (
            "text stop inside EOS precedes EOS and length",
            false,
            vec!["STOP"],
            7,
            1,
            Some(FinishReason::Stop),
        ),
    ] {
        let mut request = policy_request();
        request.sampling_params.stop_sequences = stops.into_iter().map(str::to_string).collect();
        request.sampling_params.max_tokens = max_tokens;
        request.metadata.insert(
            "ferrum_ignore_eos".to_string(),
            serde_json::json!(ignore_eos),
        );
        let mut state = SequenceState::new_with_tokenizer(
            request,
            vec![TokenId::new(0)],
            Some(Arc::clone(&tokenizer)),
        );
        state.generated_tokens.push(TokenId::new(token));
        assert_eq!(
            state.stop_reason(Some(tokenizer.as_ref())),
            expected,
            "{label}"
        );
    }
}

#[test]
fn sequence_state_detects_text_stop_before_length() {
    let tokenizer = PolicyTokenizer::new(8, &[("OK", 5), ("<END>", 6), ("TAIL", 7)]);
    let mut request = policy_request();
    request.sampling_params.max_tokens = 3;
    let mut state = SequenceState::new(request, vec![TokenId::new(0)]);
    state.generated_tokens = vec![TokenId::new(5), TokenId::new(6), TokenId::new(7)];
    state.stop_text_seqs = vec!["<END>".to_string()];

    assert_eq!(
        state.stop_reason(Some(&tokenizer)),
        Some(FinishReason::Stop)
    );
}

#[test]
fn model_decode_metadata_marks_structured_requests_for_full_logits() {
    let plain = SequenceState::new(policy_request(), vec![TokenId::new(0)]);
    assert_eq!(
        plain
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        None
    );
    assert_eq!(
        plain
            .model_decode_metadata()
            .get("ferrum_kv_capacity_hint")
            .and_then(|value| value.as_u64()),
        Some((1 + plain.sampling_params.max_tokens.saturating_sub(1)) as u64)
    );
    assert_eq!(
        plain
            .model_decode_metadata()
            .get("ferrum_kv_admission_target_len")
            .and_then(|value| value.as_u64()),
        Some(plain.prefill_context_len() as u64)
    );

    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    let structured = SequenceState::new_with_tokenizer(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );
    assert_eq!(
        structured
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        Some(true)
    );

    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
            .to_string(),
    );
    let json_schema =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));
    assert_eq!(
        json_schema
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[test]
fn structured_sequence_requires_a_tokenizer_aware_grammar_factory() {
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;

    let error = SequenceState::try_new_with_tokenizer_model_vocab_and_structured_factory(
        request,
        vec![TokenId::new(0)],
        None,
        None,
        None,
        OutputBudget::new(Default::default())
            .unwrap()
            .try_open_session()
            .unwrap(),
    )
    .unwrap_err()
    .to_string();

    assert!(
        error.contains("structured output requires a tokenizer-aware grammar factory"),
        "{error}"
    );
}

#[test]
fn sequence_state_prefill_context_preserves_generated_tokens_for_kv_recompute() {
    let mut state = SequenceState::new(policy_request(), vec![TokenId::new(10), TokenId::new(11)]);
    state.generated_tokens = vec![TokenId::new(12), TokenId::new(13)];

    assert_eq!(
        state.prefill_context_tokens(),
        vec![
            TokenId::new(10),
            TokenId::new(11),
            TokenId::new(12),
            TokenId::new(13)
        ]
    );
    assert_eq!(state.prefill_context_len(), 4);
    assert!(
        state
            .model_decode_metadata()
            .get("ferrum_kv_capacity_hint")
            .and_then(|value| value.as_u64())
            .unwrap()
            >= state.prefill_context_len() as u64
    );
    assert_eq!(
        state
            .model_decode_metadata()
            .get("ferrum_kv_admission_target_len")
            .and_then(|value| value.as_u64()),
        Some(state.prefill_context_len() as u64)
    );
}

/// Recompute resumes from token history, not from serialized/swapped KV. The
/// scheduler/deferral tests above cover requeueing; this exercises the actual CPU
/// model cache and logical token history. The common close-registry tests cover
/// the actual Engine requeue transition and exact incarnation ownership.
#[test]
fn cpu_kv_recompute_matches_uninterrupted_logits_and_preserves_peer() {
    use ferrum_models::test_support::{tiny_llama_model, TinyLlamaConfig};

    fn assert_logits_equal(label: &str, expected: &[f32], actual: &[f32], vocab: usize) {
        assert_eq!(expected.len(), vocab, "{label}: reference vocabulary");
        assert_eq!(actual.len(), vocab, "{label}: actual vocabulary");
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            assert!(
                expected.is_finite() && actual.is_finite(),
                "{label}: nonfinite logit at {index}: {expected} vs {actual}"
            );
            assert!(
                (expected - actual).abs() <= 1e-5,
                "{label}: logit {index} differs: {expected} vs {actual}"
            );
        }
    }

    // Both the first sampled token and a later decode boundary leave one
    // generated token outside KV. Neither may be omitted when rebuilding it.
    for completed_decodes in [0, 2] {
        let cfg = TinyLlamaConfig::default();
        let mut model = tiny_llama_model(&cfg);
        let mut reference = tiny_llama_model(&cfg);
        let mut peer_reference = tiny_llama_model(&cfg);
        let cache_id = "recompute-request";
        let peer_id = "independent-request";
        let prompt = [3, 11, 6];
        let peer_prompt = [7, 2, 13, 5];
        let mut state = SequenceState::new(
            policy_request(),
            prompt.iter().copied().map(TokenId::new).collect(),
        );

        let mut logits = model.prefill(cache_id, &prompt);
        assert_logits_equal(
            "initial prefill",
            &reference.prefill(cache_id, &prompt),
            &logits,
            cfg.vocab_size,
        );
        let mut pending = state
            .sample_and_commit_with_processors(&mut logits)
            .expect("sample initial CPU logits");
        for offset in 0..completed_decodes {
            let position = (prompt.len() + offset) as u32;
            logits = model.decode(cache_id, pending.get(), position);
            assert_logits_equal(
                "decode before preemption",
                &reference.decode(cache_id, pending.get(), position),
                &logits,
                cfg.vocab_size,
            );
            pending = state
                .sample_and_commit_with_processors(&mut logits)
                .expect("sample CPU decode logits");
        }

        assert_logits_equal(
            "peer prefill",
            &peer_reference.prefill(peer_id, &peer_prompt),
            &model.prefill(peer_id, &peer_prompt),
            cfg.vocab_size,
        );
        let peer_position = peer_prompt.len() as u32;
        assert_logits_equal(
            "peer decode before release",
            &peer_reference.decode(peer_id, 17, peer_position),
            &model.decode(peer_id, 17, peer_position),
            cfg.vocab_size,
        );
        let peer_len = model.cache_len(peer_id);
        let cached_len = model.cache_len(cache_id);
        assert_eq!(cached_len, prompt.len() + completed_decodes);
        assert_eq!(state.prefill_context_len(), cached_len + 1);
        assert_eq!(state.generated_tokens.last(), Some(&pending));
        let generated_before = state.generated_tokens.clone();
        state.tokens_this_iteration = 1;

        model.release(cache_id);
        assert_eq!(model.cache_len(cache_id), 0, "old physical KV must be gone");
        assert_eq!(
            model.cache_len(peer_id),
            peer_len,
            "release must be isolated"
        );
        assert_eq!(state.generated_tokens, generated_before);

        // An independent request continues while the first has no physical KV.
        assert_logits_equal(
            "peer decode while request is released",
            &peer_reference.decode(peer_id, 19, peer_len as u32),
            &model.decode(peer_id, 19, peer_len as u32),
            cfg.vocab_size,
        );
        assert_eq!(model.cache_len(cache_id), 0);
        let history: Vec<u32> = state
            .prefill_context_tokens()
            .iter()
            .map(|token| token.get())
            .collect();
        assert_eq!(history.len(), cached_len + 1);
        assert_eq!(history.last(), Some(&pending.get()));
        let expected = reference.decode(cache_id, pending.get(), cached_len as u32);
        logits = model.prefill(cache_id, &history);
        assert_logits_equal("recomputed prefill", &expected, &logits, cfg.vocab_size);
        assert_eq!(model.cache_len(cache_id), history.len());
        assert_eq!(model.cache_len(peer_id), peer_len + 1);

        // Compare the whole distribution after further autoregressive steps,
        // not merely the greedy choice at the restoration point.
        for _ in 0..2 {
            let next = state
                .sample_and_commit_with_processors(&mut logits)
                .expect("sample resumed CPU logits");
            let position = (state.prefill_context_len() - 1) as u32;
            logits = model.decode(cache_id, next.get(), position);
            assert_logits_equal(
                "decode after recompute",
                &reference.decode(cache_id, next.get(), position),
                &logits,
                cfg.vocab_size,
            );
        }
        assert_logits_equal(
            "peer decode after recompute",
            &peer_reference.decode(peer_id, 23, (peer_len + 1) as u32),
            &model.decode(peer_id, 23, (peer_len + 1) as u32),
            cfg.vocab_size,
        );
        let peer_len = model.cache_len(peer_id);
        model.release(cache_id);
        assert_eq!(model.cache_len(cache_id), 0);
        assert_eq!(model.cache_len(peer_id), peer_len);
        model.release(peer_id);
        reference.release(cache_id);
        peer_reference.release(peer_id);
        assert_eq!(model.cache_len(peer_id), 0);
    }
}

#[test]
fn model_decode_metadata_keeps_sampling_masks_on_model_argmax_path() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));

    assert_eq!(
        state
            .model_decode_metadata()
            .get("ferrum_require_full_logits")
            .and_then(|value| value.as_bool()),
        None
    );
    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = state.model_decode_logits_policy()
    else {
        panic!("plain greedy decode should use a masked model-side argmax policy");
    };
    assert_eq!(mask.valid_token_mask[0], 1);
    assert_eq!(mask.valid_token_mask[2], 0);
}

#[test]
fn model_decode_argmax_mask_uses_model_vocab_for_extended_stop_tokens() {
    let mut tok = PolicyTokenizer::new(
        4,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("ok", 3),
            ("<pad>", 4),
            ("<|im_end|>", 5),
        ],
    );
    tok.special.eos_token = Some(TokenId::new(5));
    tok.special.pad_token = Some(TokenId::new(4));
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tok);

    let state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer),
        Some(6),
    );

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = state.model_decode_logits_policy()
    else {
        panic!("plain greedy decode should use a model-side argmax policy");
    };
    assert_eq!(mask.len(), 6);
    assert_eq!(
        mask.valid_token_mask[5], 1,
        "extended EOS must remain selectable"
    );
    assert_eq!(
        mask.valid_token_mask[4], 0,
        "unallowed extended PAD must stay masked"
    );
}

fn response_completion_tokenizer() -> Arc<dyn Tokenizer + Send + Sync> {
    Arc::new(PolicyTokenizer::new(
        8,
        &[
            ("reason", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("<eos>", 3),
            ("<pad>", 4),
            ("END", 5),
            ("\n", 6),
            ("answer", 7),
        ],
    ))
}

fn response_completion_request() -> InferenceRequest {
    let mut request = policy_request();
    request.sampling_params.response_completion_boundary =
        ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "END".to_string(),
            alternate_envelope: None,
        };
    request
}

fn response_completion_tool_tokenizer() -> Arc<dyn Tokenizer + Send + Sync> {
    Arc::new(PolicyTokenizer::new(
        10,
        &[
            ("reason", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("<eos>", 3),
            ("<pad>", 4),
            ("END", 5),
            ("\n", 6),
            ("<tool_call>", 7),
            ("</tool_call>", 8),
            ("answer", 9),
        ],
    ))
}

fn response_completion_multitoken_tool_tokenizer() -> Arc<dyn Tokenizer + Send + Sync> {
    Arc::new(PolicyTokenizer::new(
        12,
        &[
            ("reason", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("<eos>", 3),
            ("<pad>", 4),
            ("END", 5),
            ("\\n", 6),
            ("OPEN", 7),
            ("START", 8),
            ("CLOSE", 9),
            ("FINISH", 10),
            ("answer", 11),
        ],
    ))
}

fn response_completion_tool_request(tool_choice: ferrum_types::ApiToolChoice) -> InferenceRequest {
    let mut request = policy_tool_request(
        ferrum_types::ApiToolCallProtocol::FunctionParameterXml,
        tool_choice,
    );
    let alternate_envelope = request
        .api_request
        .as_ref()
        .and_then(ferrum_types::ApiRequest::generated_response_envelope);
    request.sampling_params.response_completion_boundary =
        ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload {
            delimiter: "END".to_string(),
            alternate_envelope,
        };
    request
}

#[test]
fn response_completion_boundary_masks_eos_until_delimiter_and_payload() {
    let tokenizer = response_completion_tokenizer();
    let mut state = SequenceState::new_with_tokenizer(
        response_completion_request(),
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );

    for token_id in [0u32, 5, 6, 7] {
        let mut logits = vec![f32::NEG_INFINITY; 8];
        logits[3] = 100.0;
        logits[token_id as usize] = 10.0;
        let token = state
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap();
        assert_eq!(token, TokenId::new(token_id));
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    let mut eos_logits = vec![f32::NEG_INFINITY; 8];
    eos_logits[3] = 100.0;
    let eos = state
        .sample_and_commit_with_processors_and_tokenizer(&mut eos_logits, Some(tokenizer.as_ref()))
        .unwrap();
    assert_eq!(eos, TokenId::new(3));
}

#[test]
fn response_completion_boundary_accepts_complete_typed_tool_envelope() {
    let tokenizer = response_completion_tool_tokenizer();
    let mut state = SequenceState::new_with_tokenizer(
        response_completion_tool_request(ferrum_types::ApiToolChoice::Mode("auto".to_string())),
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );

    for token_id in [7u32, 0, 8] {
        let mut logits = vec![f32::NEG_INFINITY; 10];
        logits[3] = 100.0;
        logits[token_id as usize] = 10.0;
        let token = state
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap();
        assert_eq!(token, TokenId::new(token_id));
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    assert_eq!(state.stop_reason(Some(tokenizer.as_ref())), None);
    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        ..
    } = state.model_decode_logits_policy()
    else {
        panic!("completed tool envelope must retain model-side greedy argmax");
    };
    assert_eq!(mask.valid_token_mask[3], 1);

    let mut second_envelope_logits = vec![f32::NEG_INFINITY; 10];
    second_envelope_logits[3] = 10.0;
    second_envelope_logits[7] = 100.0;
    let second_open = state
        .sample_and_commit_with_processors_and_tokenizer(
            &mut second_envelope_logits,
            Some(tokenizer.as_ref()),
        )
        .unwrap();
    assert_eq!(
        second_open,
        TokenId::new(7),
        "completion capability must not force-stop a legal following envelope"
    );

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(open_mask),
        ..
    } = state.model_decode_logits_policy()
    else {
        panic!("open second envelope must retain model-side greedy argmax");
    };
    assert_eq!(open_mask.valid_token_mask[3], 0);

    for token_id in [0u32, 8] {
        let mut logits = vec![f32::NEG_INFINITY; 10];
        logits[3] = 100.0;
        logits[token_id as usize] = 10.0;
        let token = state
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap();
        assert_eq!(token, TokenId::new(token_id));
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    let mut eos_logits = vec![f32::NEG_INFINITY; 10];
    eos_logits[3] = 100.0;
    let eos = state
        .sample_and_commit_with_processors_and_tokenizer(&mut eos_logits, Some(tokenizer.as_ref()))
        .unwrap();
    assert_eq!(eos, TokenId::new(3));
    assert!(matches!(
        state.stop_reason(Some(tokenizer.as_ref())),
        Some(FinishReason::Stop | FinishReason::EOS)
    ));
}

#[test]
fn response_completion_boundary_keeps_eos_masked_inside_tool_after_reasoning_close() {
    let tokenizer = response_completion_tool_tokenizer();
    let mut state = SequenceState::new_with_tokenizer(
        response_completion_tool_request(ferrum_types::ApiToolChoice::Mode("auto".to_string())),
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );

    for token_id in [5u32, 6, 7, 0] {
        let mut logits = vec![f32::NEG_INFINITY; 10];
        logits[3] = 100.0;
        logits[token_id as usize] = 10.0;
        let token = state
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap();
        assert_eq!(token, TokenId::new(token_id));
        assert_eq!(
            logits[3],
            f32::NEG_INFINITY,
            "EOS must remain masked until the opened tool envelope closes"
        );
    }

    let mut close_logits = vec![f32::NEG_INFINITY; 10];
    close_logits[3] = 100.0;
    close_logits[8] = 10.0;
    let close = state
        .sample_and_commit_with_processors_and_tokenizer(
            &mut close_logits,
            Some(tokenizer.as_ref()),
        )
        .unwrap();
    assert_eq!(close, TokenId::new(8));
    assert_eq!(close_logits[3], f32::NEG_INFINITY);

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        ..
    } = state.model_decode_logits_policy()
    else {
        panic!("completed tool envelope must retain model-side greedy argmax");
    };
    assert_eq!(mask.valid_token_mask[3], 1);
}

#[test]
fn response_completion_boundary_tracks_multitoken_envelope_without_false_payload() {
    let tokenizer = response_completion_multitoken_tool_tokenizer();
    let mut request =
        response_completion_tool_request(ferrum_types::ApiToolChoice::Mode("auto".to_string()));
    let ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload {
        alternate_envelope: Some(envelope),
        ..
    } = &mut request.sampling_params.response_completion_boundary
    else {
        panic!("tool request must carry its typed completion envelope");
    };
    envelope.open_token_text = "OPEN START".to_string();
    envelope.close_token_text = "CLOSE FINISH".to_string();
    let mut state = SequenceState::new_with_tokenizer(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );

    for token_id in [5u32, 7, 8, 0, 9, 10] {
        let mut logits = vec![f32::NEG_INFINITY; 12];
        logits[3] = 100.0;
        logits[token_id as usize] = 10.0;
        let token = state
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap();
        assert_eq!(token, TokenId::new(token_id));
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        ..
    } = state.model_decode_logits_policy()
    else {
        panic!("completed multi-token envelope must retain model-side greedy argmax");
    };
    assert_eq!(mask.valid_token_mask[3], 1);
}

#[test]
fn response_completion_boundary_rejects_tool_envelopes_beyond_typed_limit() {
    let tokenizer = response_completion_multitoken_tool_tokenizer();
    let mut request =
        response_completion_tool_request(ferrum_types::ApiToolChoice::Mode("auto".to_string()));
    let ferrum_types::ResponseCompletionBoundary::AfterDelimiterAndPayload {
        alternate_envelope: Some(envelope),
        ..
    } = &mut request.sampling_params.response_completion_boundary
    else {
        panic!("tool request must carry its typed completion envelope");
    };
    envelope.open_token_text = "OPEN START".to_string();
    envelope.close_token_text = "CLOSE FINISH".to_string();
    envelope.max_envelopes = 1;
    let mut state = SequenceState::new_with_tokenizer(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );

    for token_id in [7u32, 8, 0, 9, 10, 7] {
        let mut logits = vec![f32::NEG_INFINITY; 12];
        logits[token_id as usize] = 100.0;
        let token = state
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap();
        assert_eq!(token, TokenId::new(token_id));
    }

    let committed_len = state.generated_tokens.len();
    for _ in 0..2 {
        let mut repeated_open_finish = vec![f32::NEG_INFINITY; 12];
        repeated_open_finish[8] = 100.0;
        let error = state
            .sample_and_commit_with_processors_and_tokenizer(
                &mut repeated_open_finish,
                Some(tokenizer.as_ref()),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("1-envelope protocol limit"),
            "{error}"
        );
        assert_eq!(
            state.generated_tokens.len(),
            committed_len,
            "a rejected envelope token must not commit or advance its matcher"
        );
    }
}

#[test]
fn response_completion_boundary_rejects_unopened_or_disabled_tool_envelope() {
    let tokenizer = response_completion_tool_tokenizer();
    let mut unopened = SequenceState::new_with_tokenizer(
        response_completion_tool_request(ferrum_types::ApiToolChoice::Mode("auto".to_string())),
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );
    let mut close_logits = vec![f32::NEG_INFINITY; 10];
    close_logits[3] = 100.0;
    close_logits[8] = 10.0;
    let close = unopened
        .sample_and_commit_with_processors_and_tokenizer(
            &mut close_logits,
            Some(tokenizer.as_ref()),
        )
        .unwrap();
    assert_eq!(close, TokenId::new(8));
    assert_eq!(close_logits[3], f32::NEG_INFINITY);

    let mut still_blocked = vec![f32::NEG_INFINITY; 10];
    still_blocked[3] = 100.0;
    still_blocked[0] = 10.0;
    let fallback = unopened
        .sample_and_commit_with_processors_and_tokenizer(
            &mut still_blocked,
            Some(tokenizer.as_ref()),
        )
        .unwrap();
    assert_eq!(fallback, TokenId::new(0));
    assert_eq!(still_blocked[3], f32::NEG_INFINITY);

    let mut disabled = SequenceState::new_with_tokenizer(
        response_completion_tool_request(ferrum_types::ApiToolChoice::Mode("none".to_string())),
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );
    for token_id in [7u32, 0, 8] {
        let mut logits = vec![f32::NEG_INFINITY; 10];
        logits[3] = 100.0;
        logits[token_id as usize] = 10.0;
        let token = disabled
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap();
        assert_eq!(token, TokenId::new(token_id));
        assert_eq!(logits[3], f32::NEG_INFINITY);
    }
}

#[test]
fn token_sequence_matcher_handles_overlaps_reuse_and_empty_input() {
    assert!(TokenSequenceMatcher::new(Vec::new(), "empty matcher").is_err());

    let mut matcher = TokenSequenceMatcher::new(vec![1, 2, 1], "overlapping matcher").unwrap();
    assert!(!matcher.observe(1));
    assert!(!matcher.observe(2));
    assert!(matcher.observe(1));
    assert!(
        !matcher.observe(2),
        "successful matches must retain their KMP overlap"
    );
    assert!(matcher.observe(1));
    matcher.reset();
    assert!(matcher.is_at_boundary());
}

#[test]
fn response_completion_boundary_updates_greedy_mask_at_phase_transition() {
    let tokenizer = response_completion_tokenizer();
    let mut state = SequenceState::new_with_tokenizer(
        response_completion_request(),
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(initial_mask),
        ..
    } = state.model_decode_logits_policy()
    else {
        panic!("completion boundary must preserve model-side greedy argmax");
    };
    assert_eq!(initial_mask.valid_token_mask[3], 0);
    assert!(state.generated_tokens.is_empty());
    assert!(state.sampling_history.token_frequencies().is_empty());
    assert!(state
        .validate_and_commit_model_greedy_argmax_token(Some(tokenizer.as_ref()), TokenId::new(3))
        .unwrap_err()
        .to_string()
        .contains("before satisfying the response completion boundary"));
    assert!(
        state.generated_tokens.is_empty(),
        "a rejected greedy token must not partially commit generated history"
    );
    assert!(
        state.sampling_history.token_frequencies().is_empty(),
        "a rejected greedy token must not partially commit sampling history"
    );

    for token_id in [0u32, 5, 6, 7] {
        let token = TokenId::new(token_id);
        state
            .validate_and_commit_model_greedy_argmax_token(Some(tokenizer.as_ref()), token)
            .unwrap();
    }
    assert_eq!(state.generated_tokens, [0u32, 5, 6, 7].map(TokenId::new));
    assert_eq!(
        state.sampling_history.token_count(&state.generated_tokens),
        state.generated_tokens.len()
    );

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(completed_mask),
        ..
    } = state.model_decode_logits_policy()
    else {
        panic!("completed boundary must retain model-side greedy argmax");
    };
    assert_eq!(completed_mask.valid_token_mask[3], 1);
    assert_ne!(completed_mask.fingerprint, initial_mask.fingerprint);
    state
        .validate_and_commit_model_greedy_argmax_token(Some(tokenizer.as_ref()), TokenId::new(3))
        .unwrap();
}

#[test]
fn response_completion_boundary_preserves_explicit_user_stop() {
    let tokenizer = response_completion_tokenizer();
    let mut request = response_completion_request();
    request.sampling_params.stop_sequences = vec!["reason".to_string()];
    let mut state = SequenceState::new_with_tokenizer(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );

    let mut logits = vec![f32::NEG_INFINITY; 8];
    logits[0] = 10.0;
    logits[3] = 100.0;
    let token = state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();
    assert_eq!(token, TokenId::new(0));
    assert_eq!(
        state.stop_reason(Some(tokenizer.as_ref())),
        Some(FinishReason::Stop)
    );
}

#[test]
fn response_completion_boundary_preserves_user_stop_colliding_with_model_eos() {
    let tokenizer = response_completion_tokenizer();
    for ignore_eos in [false, true] {
        for model_argmax in [false, true] {
            let mut request = response_completion_request();
            request.sampling_params.stop_sequences = vec!["<eos>".to_string()];
            request.metadata.insert(
                "ferrum_ignore_eos".to_string(),
                serde_json::json!(ignore_eos),
            );
            if !model_argmax {
                request.sampling_params.presence_penalty = 1.0;
            }
            let mut state = SequenceState::new_with_tokenizer(
                request,
                vec![TokenId::new(0)],
                Some(Arc::clone(&tokenizer)),
            );
            if model_argmax {
                let LogitsReturnPolicy::GreedyArgmax {
                    token_mask: Some(mask),
                    ..
                } = state.model_decode_logits_policy()
                else {
                    panic!("ordinary greedy sampling must retain its model argmax path");
                };
                assert_eq!(
                    mask.valid_token_mask[3], 1,
                    "an explicit stop must remain selectable before the completion boundary"
                );
                state
                    .validate_and_commit_model_greedy_argmax_token(
                        Some(tokenizer.as_ref()),
                        TokenId::new(3),
                    )
                    .unwrap();
            } else {
                assert!(matches!(
                    state.model_decode_logits_policy(),
                    LogitsReturnPolicy::FullLogits
                ));
                let mut logits = vec![f32::NEG_INFINITY; tokenizer.vocab_size()];
                logits[0] = 1.0;
                logits[3] = 100.0;
                let token = state
                    .sample_and_commit_with_processors_and_tokenizer(
                        &mut logits,
                        Some(tokenizer.as_ref()),
                    )
                    .unwrap();
                assert_eq!(token, TokenId::new(3));
            }
            let stop = state.stop_reason(Some(tokenizer.as_ref()));
            assert_eq!(stop, Some(FinishReason::Stop));
            assert!(!state.should_stream_generated_token(
                Some(tokenizer.as_ref()),
                TokenId::new(3),
                stop
            ));
        }
    }
}

#[test]
fn response_completion_boundary_preserves_ignore_eos() {
    let tokenizer = response_completion_tokenizer();
    let mut request = response_completion_request();
    request
        .metadata
        .insert("ferrum_ignore_eos".to_string(), serde_json::json!(true));
    let mut state = SequenceState::new_with_tokenizer(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        ..
    } = state.model_decode_logits_policy()
    else {
        panic!("ignore_eos must retain model-side greedy argmax");
    };
    assert_eq!(mask.valid_token_mask[3], 1);
    state
        .validate_and_commit_model_greedy_argmax_token(Some(tokenizer.as_ref()), TokenId::new(3))
        .unwrap();
    assert_eq!(state.stop_reason(Some(tokenizer.as_ref())), None);
}

#[test]
fn model_decode_logits_policy_keeps_repetition_penalty_on_greedy_argmax_path() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let mut request = policy_request();
    request.sampling_params.repetition_penalty = 1.1;
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));
    state.generated_tokens = vec![TokenId::new(3), TokenId::new(3), TokenId::new(0)];

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: Some(penalty),
    } = state.model_decode_logits_policy()
    else {
        panic!("greedy repetition penalty should use model-side argmax policy");
    };
    assert_eq!(mask.valid_token_mask[2], 0);
    assert_eq!(penalty.penalty(), 1.1);
    assert_eq!(penalty.token_ids(), &[3, 0]);
}

#[test]
fn request_sampling_plan_applies_presence_and_frequency_penalties() {
    let mut request = policy_request();
    request.sampling_params.temperature = 0.0;
    request.sampling_params.repetition_penalty = 1.0;
    request.sampling_params.presence_penalty = 1.5;
    request.sampling_params.frequency_penalty = 0.5;
    let mut state = SequenceState::new(request, vec![TokenId::new(0)]);
    state.generated_tokens = vec![TokenId::new(1), TokenId::new(1)];
    state
        .sampling_history
        .record(TokenId::new(1), &mut state.output_semantic)
        .unwrap();
    state
        .sampling_history
        .record(TokenId::new(1), &mut state.output_semantic)
        .unwrap();

    assert_eq!(
        state.sampling_plan.processor_chain.processor_names(),
        vec!["presence_frequency_penalty"]
    );

    let mut logits = vec![0.0, 3.0, 2.0];
    let token = state
        .sample_and_commit_with_processors(&mut logits)
        .unwrap();

    assert_eq!(token, TokenId::new(2));
    assert_eq!(
        state.generated_tokens,
        [TokenId::new(1), TokenId::new(1), token]
    );
    assert!((logits[1] - 0.5).abs() < 1e-6);
    assert_eq!(
        state
            .sampling_history
            .token_frequencies()
            .get(&TokenId::new(2)),
        Some(&1)
    );
}

#[test]
fn sampling_history_transitions_keep_scope_and_frequency_table_consistent() {
    let mut credit = OutputBudget::new(Default::default())
        .unwrap()
        .try_open_session()
        .unwrap()
        .try_reserve_semantic(0)
        .unwrap();
    let mut history = SequenceSamplingHistory::default();
    let generated = vec![
        TokenId::new(1),
        TokenId::new(1),
        TokenId::new(2),
        TokenId::new(3),
    ];
    for &token in &generated {
        history.record(token, &mut credit).unwrap();
    }
    assert_eq!(
        history.scope(),
        SequenceSamplingHistoryScope::FullGeneration
    );
    assert_eq!(history.previous_tokens(&generated).unwrap(), &generated);
    assert_eq!(history.token_frequencies().get(&TokenId::new(1)), Some(&2));

    assert_eq!(
        credit.bytes(),
        3 * (std::mem::size_of::<TokenId>() + std::mem::size_of::<usize>())
    );
    assert!(history.enter_hidden_structured_output(&mut credit).unwrap());
    assert_eq!(history.previous_tokens(&generated).unwrap(), &[]);
    assert!(history.token_frequencies().is_empty());
    assert_eq!(credit.bytes(), 0);
    history.record(TokenId::new(4), &mut credit).unwrap();
    assert!(history.token_frequencies().is_empty());

    assert!(history
        .activate_visible_structured_output(2, &generated, &mut credit)
        .unwrap());
    assert_eq!(
        history.scope(),
        SequenceSamplingHistoryScope::VisibleStructuredOutput {
            start_token_index: 2
        }
    );
    assert_eq!(
        history.previous_tokens(&generated).unwrap(),
        &[TokenId::new(2), TokenId::new(3)]
    );
    assert_eq!(history.token_frequencies().len(), 2);
    assert_eq!(
        credit.bytes(),
        2 * (std::mem::size_of::<TokenId>() + std::mem::size_of::<usize>())
    );
    assert!(!history
        .activate_visible_structured_output(2, &generated, &mut credit)
        .unwrap());
    assert!(history
        .activate_visible_structured_output(generated.len() + 1, &generated, &mut credit)
        .is_err());
}

#[test]
fn sequence_rejects_unrepresented_sampling_modes_before_execution() {
    let mut request = policy_request();
    request.sampling_params.typical_p = Some(0.9);

    let error = SequenceState::try_new_with_tokenizer_model_vocab_and_structured_factory(
        request,
        vec![TokenId::new(0)],
        None,
        None,
        None,
        OutputBudget::new(Default::default())
            .unwrap()
            .try_open_session()
            .unwrap(),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("tfs, typical_p, and mirostat are not supported"));
}

#[test]
fn model_decode_logits_policy_requires_full_for_structured_output() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    let state = SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));

    assert!(matches!(
        state.model_decode_logits_policy(),
        LogitsReturnPolicy::FullLogits
    ));
    assert!(state.has_structured_output_constraint());
}

#[test]
fn token_policy_cache_keeps_request_control_allowances_isolated() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        8,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
        ],
    ));
    let no_allowances = HashSet::new();
    let forbidden = cached_forbidden_generation_tokens(&tokenizer, &no_allowances);
    assert!(forbidden.contains(&2));

    let allowed = HashSet::from([2]);
    let request_forbidden = cached_forbidden_generation_tokens(&tokenizer, &allowed);
    assert!(!request_forbidden.contains(&2));

    let next_request_forbidden = cached_forbidden_generation_tokens(&tokenizer, &no_allowances);
    assert!(
        next_request_forbidden.contains(&2),
        "one request's allowed controls must not mutate the shared base policy"
    );
}

#[test]
fn token_policy_cache_rejects_an_expired_tokenizer_identity() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        9,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
        ],
    ));
    let expired_tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(PolicyTokenizer::new(9, &[("stale", 0)]));
    let expired_identity = Arc::downgrade(&expired_tokenizer);
    drop(expired_tokenizer);

    let key = (tokenizer_cache_key(&tokenizer), tokenizer.vocab_size());
    let cache = TOKEN_POLICY_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    cache.lock().expect("token policy cache poisoned").insert(
        key,
        TokenPolicyCacheEntry {
            tokenizer: expired_identity,
            forbidden: HashSet::from([u32::MAX]),
            model_greedy_forbidden: HashSet::from([u32::MAX]),
        },
    );

    let forbidden = cached_forbidden_generation_tokens(&tokenizer, &HashSet::new());
    assert!(
        !forbidden.contains(&u32::MAX),
        "an expired tokenizer identity must not return stale policy data"
    );
    assert!(forbidden.contains(&2));

    cache
        .lock()
        .expect("token policy cache poisoned")
        .remove(&key);
}

#[test]
fn model_greedy_argmax_output_accepts_masked_policy_token_and_tracks_frequency() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );

    assert!(state.requires_full_logits_for_sampling());
    state
        .validate_and_commit_model_greedy_argmax_token(Some(tokenizer.as_ref()), TokenId::new(0))
        .unwrap();
    assert_eq!(
        state
            .sampling_history
            .token_frequencies()
            .get(&TokenId::new(0)),
        Some(&1)
    );
    let err = state
        .validate_and_commit_model_greedy_argmax_token(Some(tokenizer.as_ref()), TokenId::new(2))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("model greedy argmax returned a forbidden token"),
        "model-side greedy argmax must not bypass forbidden token masks: {err}"
    );
    assert!(err.contains("token_id=2"), "{err}");
    assert!(err.contains("token_text=\"<unk>\""), "{err}");
    assert!(err.contains("forbidden_count="), "{err}");
    assert!(err.contains("argmax_mask="), "{err}");
    assert!(err.contains("value=0"), "{err}");
}

#[test]
fn model_greedy_argmax_output_rejects_non_greedy_request() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        4,
        &[("normal", 0), ("<s>", 1), ("<unk>", 2), ("ok", 3)],
    ));
    let mut request = policy_request();
    request.sampling_params.top_p = 0.8;
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer.clone()));

    assert!(state
        .validate_and_commit_model_greedy_argmax_token(Some(tokenizer.as_ref()), TokenId::new(0))
        .is_err());
}

#[test]
fn single_token_stop_sequence_also_matches_composite_decoded_token() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        8,
        &[("OK", 5), ("\n", 6), ("OK \n\n", 7)],
    ));
    let mut request = policy_request();
    request.sampling_params.stop_sequences = vec!["\n".to_string()];
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer.clone()));

    assert!(state.stop_token_ids.contains(&6));
    assert!(state.stop_text_seqs.contains(&"\n".to_string()));

    state.generated_tokens.push(TokenId::new(7));

    assert_eq!(
        state.stop_reason(Some(tokenizer.as_ref())),
        Some(FinishReason::Stop)
    );
}

#[test]
fn schema_guided_sampling_masks_extended_stop_tokens_before_accept() {
    let mut tokenizer = PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("<|eot_id|>", 8),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    for user_stop in [None, Some("</s>"), Some("<|eot_id|>")] {
        let mut request = policy_request();
        request.sampling_params.stop_sequences =
            user_stop.into_iter().map(str::to_string).collect();
        request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
            r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
                .to_string(),
        );
        let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request,
            vec![TokenId::new(0)],
            Some(Arc::clone(&tokenizer)),
            Some(9),
        );

        assert!(state.structured_output_processor.is_some());
        assert!(
            state.stop_token_ids.contains(&8),
            "common eot token should be a resolved stop token"
        );

        let mut logits = vec![f32::NEG_INFINITY; 9];
        logits[0] = 1.0;
        logits[1] = 0.5;
        logits[3] = 200.0;
        logits[8] = 100.0;

        let token = state
            .sample_and_commit_with_processors(&mut logits)
            .unwrap();

        assert_eq!(token.get(), 0);
        for eos in [3, 8] {
            assert!(
                logits[eos].is_infinite() && logits[eos].is_sign_negative(),
                "hard JSON must mask EOS {eos} before acceptance even with user stop {user_stop:?}"
            );
        }
    }
}

#[test]
fn schema_guided_sampling_masks_extended_control_tokens_before_accept() {
    let mut tokenizer = PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("<think>", 7),
            ("<|eot_id|>", 8),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#
            .to_string(),
    );
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(tokenizer),
        Some(9),
    );

    assert!(state.structured_output_processor.is_some());
    assert!(
        state.allowed_extended_token_ids.contains(&7),
        "think token should be an allowed generated control token outside base vocab"
    );
    assert!(
        !state.stop_token_ids.contains(&7),
        "think token should not be treated as a terminator"
    );

    let mut logits = vec![f32::NEG_INFINITY; 9];
    logits[0] = 1.0;
    logits[1] = 0.5;
    logits[7] = 100.0;
    logits[8] = 90.0;

    let token = state
        .sample_and_commit_with_processors(&mut logits)
        .unwrap();

    assert_eq!(token.get(), 0);
    assert!(
        logits[7].is_infinite() && logits[7].is_sign_negative(),
        "schema-guided generation must not sample invisible control tokens before accept"
    );
    assert!(
        logits[8].is_infinite() && logits[8].is_sign_negative(),
        "schema-guided generation must not sample stop tokens before accept"
    );
}

#[test]
fn schema_guided_sampling_preserves_required_reasoning_delimiter_control() {
    let mut tokenizer = PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("</think>", 7),
            ("<think>", 8),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    request.sampling_params.structured_output_start =
        ferrum_types::StructuredOutputStart::AfterDelimiter("</think>".to_string());
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
        Some(9),
    );

    let mut reasoning_logits = vec![f32::NEG_INFINITY; 9];
    reasoning_logits[0] = 1.0;
    reasoning_logits[7] = 100.0;
    reasoning_logits[8] = 90.0;
    let delimiter = state
        .sample_and_commit_with_processors_and_tokenizer(
            &mut reasoning_logits,
            Some(tokenizer.as_ref()),
        )
        .expect("the activation delimiter must remain selectable");
    assert_eq!(delimiter, TokenId::new(7));
    assert_eq!(reasoning_logits[7], 100.0);
    assert_eq!(reasoning_logits[8], f32::NEG_INFINITY);

    let mut grammar_logits = vec![0.0; 9];
    grammar_logits[0] = 1.0;
    grammar_logits[7] = 100.0;
    grammar_logits[8] = 90.0;
    let first_json_token = state
        .sample_and_commit_with_processors(&mut grammar_logits)
        .expect("the grammar must activate after the delimiter");
    assert_eq!(first_json_token, TokenId::new(0));
    assert_eq!(grammar_logits[7], f32::NEG_INFINITY);
    assert_eq!(grammar_logits[8], f32::NEG_INFINITY);
}

#[test]
fn schema_guided_sampling_scopes_penalties_to_visible_output_after_reasoning() {
    let mut tokenizer = PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("</think>", 7),
            ("<think>", 8),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let mut request = policy_request();
    request.sampling_params.repetition_penalty = 2.0;
    request.sampling_params.presence_penalty = 1.5;
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    request.sampling_params.structured_output_start =
        ferrum_types::StructuredOutputStart::AfterDelimiter("</think>".to_string());
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
        Some(9),
    );

    // The hidden reasoning domain deliberately repeats the same opening token
    // that the visible JSON grammar will need later. User-facing penalties are
    // scoped away from hidden reasoning, so neither hidden step is changed.
    for _ in 0..2 {
        let mut reasoning_logits = vec![f32::NEG_INFINITY; 9];
        reasoning_logits[0] = 3.0;
        let reasoning_token = state
            .sample_and_commit_with_processors_and_tokenizer(
                &mut reasoning_logits,
                Some(tokenizer.as_ref()),
            )
            .unwrap();
        assert_eq!(reasoning_token, TokenId::new(0));
        assert_eq!(reasoning_logits[0], 3.0);
        assert_eq!(
            state.sampling_history.scope(),
            SequenceSamplingHistoryScope::HiddenStructuredOutput
        );
        assert!(state.sampling_history.token_frequencies().is_empty());
    }
    assert_eq!(state.generated_tokens.len(), 2);

    let mut delimiter_logits = vec![f32::NEG_INFINITY; 9];
    delimiter_logits[7] = 3.0;
    let delimiter = state
        .sample_and_commit_with_processors_and_tokenizer(
            &mut delimiter_logits,
            Some(tokenizer.as_ref()),
        )
        .unwrap();
    assert_eq!(delimiter, TokenId::new(7));
    assert_eq!(
        state.sampling_history.scope(),
        SequenceSamplingHistoryScope::HiddenStructuredOutput
    );
    assert!(state.sampling_history.token_frequencies().is_empty());
    assert_eq!(state.generated_tokens.len(), 3);

    let mut grammar_logits = vec![f32::NEG_INFINITY; 9];
    grammar_logits[0] = 1.0;
    let first_json_token = state
        .sample_and_commit_with_processors_and_tokenizer(
            &mut grammar_logits,
            Some(tokenizer.as_ref()),
        )
        .unwrap();
    assert_eq!(first_json_token, TokenId::new(0));
    assert_eq!(state.generated_tokens.len(), 4);
    assert_eq!(
        grammar_logits[0], 1.0,
        "hidden reasoning must not apply repetition or presence penalty to visible JSON"
    );
    assert_eq!(
        state.sampling_history.scope(),
        SequenceSamplingHistoryScope::VisibleStructuredOutput {
            start_token_index: 3
        }
    );
    assert_eq!(
        state.sampling_history.token_frequencies().len(),
        1,
        "the active penalty table must contain only visible JSON tokens"
    );
    assert_eq!(
        state
            .sampling_history
            .token_frequencies()
            .get(&TokenId::new(0)),
        Some(&1)
    );
    assert!(!state
        .sampling_history
        .token_frequencies()
        .contains_key(&TokenId::new(7)));

    // A replay reset reconstructs the same typed scope from generated history
    // instead of leaking hidden-token counts back into the visible domain.
    state.reset_guided_processors().unwrap();
    assert_eq!(
        state.sampling_history.scope(),
        SequenceSamplingHistoryScope::HiddenStructuredOutput
    );
    assert!(state.sampling_history.token_frequencies().is_empty());
    let mut close_logits = vec![f32::NEG_INFINITY; 9];
    close_logits[4] = 1.0;
    let close = state
        .sample_and_commit_with_processors_and_tokenizer(
            &mut close_logits,
            Some(tokenizer.as_ref()),
        )
        .unwrap();
    assert_eq!(close, TokenId::new(4));
    assert_eq!(state.generated_tokens.len(), 5);
    assert_eq!(
        state.sampling_history.scope(),
        SequenceSamplingHistoryScope::VisibleStructuredOutput {
            start_token_index: 3
        }
    );
    assert_eq!(
        state
            .sampling_history
            .previous_tokens(&state.generated_tokens)
            .unwrap(),
        &[TokenId::new(0), TokenId::new(4)]
    );
    assert_eq!(
        state
            .sampling_history
            .token_frequencies()
            .get(&TokenId::new(0)),
        Some(&1)
    );
    assert_eq!(
        state
            .sampling_history
            .token_frequencies()
            .get(&TokenId::new(4)),
        Some(&1)
    );
    assert_eq!(
        state
            .sampling_history
            .token_frequencies()
            .values()
            .sum::<usize>(),
        2
    );
}

#[test]
fn schema_guided_sampling_suppresses_penalties_through_multi_token_delimiter() {
    let mut tokenizer = PolicyTokenizer::new(
        9,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("END", 7),
            ("THINK", 8),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let mut request = policy_request();
    request.sampling_params.repetition_penalty = 2.0;
    request.sampling_params.presence_penalty = 1.5;
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    request.sampling_params.structured_output_start =
        ferrum_types::StructuredOutputStart::AfterDelimiter("END THINK".to_string());
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
        Some(9),
    );

    let mut reasoning_logits = vec![f32::NEG_INFINITY; 9];
    reasoning_logits[0] = 3.0;
    let reasoning = state
        .sample_and_commit_with_processors_and_tokenizer(
            &mut reasoning_logits,
            Some(tokenizer.as_ref()),
        )
        .unwrap();
    assert_eq!(reasoning, TokenId::new(0));

    for delimiter_token in [TokenId::new(7), TokenId::new(8)] {
        let mut delimiter_logits = vec![f32::NEG_INFINITY; 9];
        delimiter_logits[delimiter_token.get() as usize] = 3.0;
        let token = state
            .sample_and_commit_with_processors_and_tokenizer(
                &mut delimiter_logits,
                Some(tokenizer.as_ref()),
            )
            .unwrap();
        assert_eq!(token, delimiter_token);
        assert_eq!(
            state.sampling_history.scope(),
            SequenceSamplingHistoryScope::HiddenStructuredOutput
        );
        assert!(state.sampling_history.token_frequencies().is_empty());
    }

    let mut grammar_logits = vec![f32::NEG_INFINITY; 9];
    grammar_logits[0] = 1.0;
    let first_json_token = state
        .sample_and_commit_with_processors_and_tokenizer(
            &mut grammar_logits,
            Some(tokenizer.as_ref()),
        )
        .unwrap();
    assert_eq!(first_json_token, TokenId::new(0));
    assert_eq!(grammar_logits[0], 1.0);
    assert_eq!(
        state.sampling_history.scope(),
        SequenceSamplingHistoryScope::VisibleStructuredOutput {
            start_token_index: 3
        }
    );
}

#[test]
fn schema_guided_sampling_forces_reasoning_delimiter_at_budget() {
    let mut tokenizer = PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("</think>", 7),
            ("<think>", 8),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let mut request = policy_request();
    request.sampling_params.max_tokens = 40;
    request.sampling_params.temperature = 0.8;
    request.sampling_params.top_p = 0.5;
    request.sampling_params.top_k = Some(1);
    request.sampling_params.repetition_penalty = 1.2;
    request.sampling_params.seed = Some(9271);
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonObject;
    request.sampling_params.structured_output_start =
        ferrum_types::StructuredOutputStart::AfterDelimiter("</think>".to_string());
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
        Some(9),
    );
    state.generated_tokens = vec![TokenId::new(0); 7];

    let mut reasoning_logits = vec![f32::NEG_INFINITY; 9];
    reasoning_logits[0] = 100.0;
    let delimiter = state
        .sample_and_commit_with_processors_and_tokenizer(
            &mut reasoning_logits,
            Some(tokenizer.as_ref()),
        )
        .expect("the output budget must force the reasoning delimiter");
    assert_eq!(delimiter, TokenId::new(7));
    assert_eq!(reasoning_logits[7], 0.0);
    assert!(reasoning_logits
        .iter()
        .enumerate()
        .all(|(token, logit)| token == 7 || !logit.is_finite()));

    let mut grammar_logits = vec![0.0; 9];
    grammar_logits[0] = 1.0;
    grammar_logits[7] = 100.0;
    grammar_logits[8] = 90.0;
    let first_json_token = state
        .sample_and_commit_with_processors(&mut grammar_logits)
        .expect("the grammar must activate after the forced delimiter");
    assert_eq!(first_json_token, TokenId::new(0));
    assert_eq!(grammar_logits[7], f32::NEG_INFINITY);
    assert_eq!(grammar_logits[8], f32::NEG_INFINITY);
}

#[test]
fn schema_guided_sampling_allows_extended_stop_after_accept() {
    let mut tokenizer = PolicyTokenizer::new(
        6,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("</s>", 3),
            ("}", 4),
            ("\"", 5),
            ("<think>", 7),
            ("<|eot_id|>", 8),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let mut request = policy_request();
    request.sampling_params.response_format =
        ferrum_types::ResponseFormat::JsonSchema(r#"{"enum":["x"]}"#.to_string());
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(tokenizer),
        Some(9),
    );
    state.generated_tokens = vec![TokenId::new(5), TokenId::new(2), TokenId::new(5)];

    let mut logits = vec![f32::NEG_INFINITY; 9];
    logits[1] = 80.0;
    logits[7] = 100.0;
    logits[8] = 90.0;

    let token = state
        .sample_and_commit_with_processors(&mut logits)
        .unwrap();

    assert_eq!(token.get(), 8);
    assert!(state
        .structured_output_processor
        .as_ref()
        .unwrap()
        .is_accepting_with_terminals(&state.generated_tokens, &state.stop_token_ids)
        .unwrap());
    assert!(
        logits[7].is_infinite() && logits[7].is_sign_negative(),
        "completed schema output should still reject non-stop control tokens"
    );
    assert!(
        logits[8].is_finite(),
        "completed schema output should allow the resolved stop token"
    );
}

#[test]
fn structured_output_terminal_error_reports_privacy_safe_lexical_progress() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(FragmentedUtf8PolicyTokenizer::new());
    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":false}"#
            .to_string(),
    );
    let mut state = SequenceState::new_with_tokenizer(
        request,
        vec![TokenId::new(b'0' as u32)],
        Some(tokenizer),
    );
    state.generated_tokens =
        br#"{"value":12777"#.iter().map(|byte| TokenId::new(*byte as u32)).collect();

    let error = state
        .structured_output_terminal_error(FinishReason::Length)
        .expect("an incomplete number must produce a terminal error")
        .to_string();

    assert!(error.contains("grammar_tokens=14"), "{error}");
    assert!(
        error.contains("sampling_history_scope=FullGeneration"),
        "{error}"
    );
    assert!(error.contains("sampling_history_tokens=14"), "{error}");
    assert!(
        error.contains("sampling_history_unique_tokens=0"),
        "{error}"
    );
    assert!(error.contains("trailing_class=Some(Number)"), "{error}");
    assert!(error.contains("trailing_class_tokens=5"), "{error}");
    assert!(
        error.contains(&format!("trailing_token_id=Some({})", b'7')),
        "{error}"
    );
    assert!(error.contains("trailing_identical_tokens=3"), "{error}");
    assert!(error.contains("liveness_identical_token_limit="), "{error}");
    assert!(error.contains("liveness_interventions=0"), "{error}");
    assert!(!error.contains(r#"{"value":12777"#), "{error}");
}

#[test]
fn sample_masks_unknown_pad_reserved_and_bos_tokens() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        10,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("[PAD151935]", 4),
            ("<|reserved_special_token_0|>", 5),
            ("ok", 6),
            ("other", 7),
            ("byte-fallback", 8),
            ("\u{00ef}\u{00bf}\u{00bd}", 9),
        ],
    ));
    let mut state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));
    let mut logits = vec![0.0f32; 10];
    logits[1] = 100.0;
    logits[2] = 99.0;
    logits[4] = 98.0;
    logits[5] = 97.0;
    logits[8] = 96.0;
    logits[9] = 95.0;
    logits[6] = 1.0;

    let token = state
        .sample_and_commit_with_processors(&mut logits)
        .unwrap();

    assert_eq!(token.get(), 6);
    for token_id in [1usize, 2, 4, 5, 8, 9] {
        assert_eq!(logits[token_id], f32::NEG_INFINITY);
    }
}

#[test]
fn sample_masks_tokenizer_vocab_holes() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        12,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
            ("ok", 6),
        ],
    ));
    let mut state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));
    let mut logits = vec![0.0f32; 12];
    logits[11] = 100.0;
    logits[6] = 1.0;

    let token = state
        .sample_and_commit_with_processors(&mut logits)
        .unwrap();

    assert_eq!(token.get(), 6);
    assert_eq!(logits[11], f32::NEG_INFINITY);
}

#[test]
fn sampling_rejects_when_engine_masks_remove_the_last_finite_candidate() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
            ("ok", 5),
        ],
    ));
    let mut state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));
    let mut logits = vec![f32::NEG_INFINITY; 6];
    logits[2] = 100.0;

    let error = state
        .sample_and_commit_with_processors(&mut logits)
        .expect_err("a forbidden token must not become an arbitrary greedy fallback");

    assert_eq!(logits[2], f32::NEG_INFINITY);
    assert!(
        error
            .to_string()
            .contains("engine sampling policies have no finite token"),
        "{error}"
    );
}

#[test]
fn structured_sampling_preserves_split_utf8_fragments_and_masks_model_vocab_holes() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(FragmentedUtf8PolicyTokenizer::new());
    let fire_head = TokenId::new(FragmentedUtf8PolicyTokenizer::FIRE_HEAD);
    let fire_tail = TokenId::new(FragmentedUtf8PolicyTokenizer::FIRE_TAIL);
    assert!(tokenizer
        .decode(&[fire_head], true)
        .unwrap()
        .contains('\u{FFFD}'));
    assert!(tokenizer
        .decode(&[fire_tail], true)
        .unwrap()
        .contains('\u{FFFD}'));
    assert_eq!(
        tokenizer.decode(&[fire_head, fire_tail], true).unwrap(),
        "🔥"
    );

    let forbidden = cached_forbidden_generation_tokens(&tokenizer, &HashSet::new());
    assert!(!forbidden.contains(&fire_head.get()));
    assert!(!forbidden.contains(&fire_tail.get()));
    assert!(forbidden.contains(&FragmentedUtf8PolicyTokenizer::REPLACEMENT));

    let mut request = policy_request();
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","properties":{"value":{"const":"\ud83d\udd25"}},"required":["value"],"additionalProperties":false}"#
            .to_string(),
    );
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
        Some(FragmentedUtf8PolicyTokenizer::MODEL_VOCAB_SIZE),
    );
    let prefix = "{\"value\":\"";
    state.generated_tokens = tokenizer.encode(prefix, false).unwrap();

    let hole = FragmentedUtf8PolicyTokenizer::MODEL_VOCAB_SIZE - 1;
    let mut head_logits = vec![f32::NEG_INFINITY; FragmentedUtf8PolicyTokenizer::MODEL_VOCAB_SIZE];
    head_logits[usize::from(fire_head)] = 10.0;
    head_logits[hole] = 100.0;
    let sampled_head = state
        .sample_and_commit_with_processors_and_tokenizer(&mut head_logits, Some(tokenizer.as_ref()))
        .expect("the legal UTF-8 head fragment must remain selectable");
    assert_eq!(sampled_head, fire_head);
    assert_eq!(head_logits[hole], f32::NEG_INFINITY);

    let mut tail_logits = vec![f32::NEG_INFINITY; FragmentedUtf8PolicyTokenizer::MODEL_VOCAB_SIZE];
    tail_logits[usize::from(fire_tail)] = 10.0;
    tail_logits[hole] = 100.0;
    let sampled_tail = state
        .sample_and_commit_with_processors_and_tokenizer(&mut tail_logits, Some(tokenizer.as_ref()))
        .expect("the legal UTF-8 tail fragment must remain selectable");
    assert_eq!(sampled_tail, fire_tail);
    assert_eq!(tail_logits[hole], f32::NEG_INFINITY);

    assert_eq!(
        tokenizer.decode(&state.generated_tokens, true).unwrap(),
        format!("{prefix}🔥")
    );
}

#[test]
fn utf8_fragment_classifier_rejects_bytes_that_cannot_form_valid_text() {
    for bytes in [
        &[0xF0, 0x9F][..],
        &[0x94, 0xA5][..],
        &[0x94, b'x'][..],
        &[0xF0, 0x9F, 0x94, 0xA5][..],
        b"Ferrum".as_slice(),
    ] {
        assert!(is_potential_utf8_fragment(bytes), "bytes={bytes:?}");
    }

    for bytes in [
        &[][..],
        &[0xFF][..],
        &[0xC0, 0x80][..],
        &[0xF0, 0x80][..],
        &[0xED, 0xA0][..],
        &[0xF4, 0x90][..],
        &[0x80, 0x80, 0x80, 0x80][..],
        &[0xF0, 0x9F, b'x'][..],
    ] {
        assert!(!is_potential_utf8_fragment(bytes), "bytes={bytes:?}");
    }
}

#[test]
fn sample_resamples_candidate_that_would_flush_replacement_char() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        8,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
            ("bad-byte-lead", 5),
            ("ok", 6),
            ("valid-byte-cont", 7),
        ],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );
    state.generated_tokens.push(TokenId::new(5));

    let mut logits = vec![0.0f32; 8];
    logits[6] = 100.0;
    logits[7] = 1.0;

    let token = state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();

    assert_eq!(token.get(), 7);
    assert_eq!(logits[6], f32::NEG_INFINITY);
}

#[test]
fn sample_resamples_orphan_utf8_continuation_without_pending_lead() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(FragmentedUtf8PolicyTokenizer::new());
    let orphan_continuation = TokenId::new(FragmentedUtf8PolicyTokenizer::FIRE_TAIL);
    let normal = TokenId::new(b'x' as u32);
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        policy_request(),
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
        Some(FragmentedUtf8PolicyTokenizer::MODEL_VOCAB_SIZE),
    );
    let mut logits = vec![f32::NEG_INFINITY; FragmentedUtf8PolicyTokenizer::MODEL_VOCAB_SIZE];
    logits[usize::from(orphan_continuation)] = 100.0;
    logits[usize::from(normal)] = 1.0;

    let sampled = state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .expect("full-logits sampling must recover from an orphan UTF-8 continuation");

    assert_eq!(sampled, normal);
    assert_eq!(logits[usize::from(orphan_continuation)], f32::NEG_INFINITY);
}

#[test]
fn pending_utf8_fragment_temporarily_requires_full_logits_for_resampling() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> =
        Arc::new(FragmentedUtf8PolicyTokenizer::new());
    let fire_head = TokenId::new(FragmentedUtf8PolicyTokenizer::FIRE_HEAD);
    let fire_tail = TokenId::new(FragmentedUtf8PolicyTokenizer::FIRE_TAIL);
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        policy_request(),
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
        Some(FragmentedUtf8PolicyTokenizer::MODEL_VOCAB_SIZE),
    );

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = state.model_decode_logits_policy()
    else {
        panic!("initial greedy decode must carry the byte-validity mask");
    };
    assert_eq!(
        mask.valid_token_mask[usize::from(fire_head)],
        1,
        "a UTF-8 lead fragment must remain selectable"
    );
    assert_eq!(
        mask.valid_token_mask[usize::from(fire_tail)],
        0,
        "an orphan UTF-8 continuation must be masked without a pending lead"
    );

    state
        .validate_and_commit_model_greedy_argmax_token(Some(tokenizer.as_ref()), fire_head)
        .expect("the first half of a valid UTF-8 token pair must remain selectable");
    assert!(state.pending_decoded_utf8_fragment);
    assert_eq!(state.pending_decoded_utf8_bytes, vec![0xF0, 0x9F]);
    assert!(matches!(
        state.model_decode_logits_policy(),
        LogitsReturnPolicy::FullLogits
    ));

    let mut logits = vec![f32::NEG_INFINITY; FragmentedUtf8PolicyTokenizer::MODEL_VOCAB_SIZE];
    logits[b'x' as usize] = 100.0;
    logits[usize::from(fire_tail)] = 1.0;
    let sampled = state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .expect("full-logits fallback must resample the UTF-8 continuation");

    assert_eq!(sampled, fire_tail);
    assert_eq!(logits[b'x' as usize], f32::NEG_INFINITY);
    assert!(!state.pending_decoded_utf8_fragment);
    assert!(state.pending_decoded_utf8_bytes.is_empty());
    assert!(matches!(
        state.model_decode_logits_policy(),
        LogitsReturnPolicy::GreedyArgmax { .. }
    ));
    assert_eq!(
        tokenizer.decode(&state.generated_tokens, true).unwrap(),
        "🔥"
    );
}

#[test]
fn sample_candidate_checks_from_streamed_text_boundary() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        7,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("<pad>", 4),
            ("byte-fallback", 5),
            ("ok", 6),
        ],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );
    state.generated_tokens.push(TokenId::new(5));
    state.streamed_text_len = 0;

    assert!(state.sample_candidate_decodes_to_forbidden_output(
        Some(tokenizer.as_ref()),
        state.streamed_text_len,
        TokenId::new(6),
        None,
    ));
}

#[test]
fn sample_allows_generated_control_tokens_above_base_vocab() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("</think>", 5),
            ("[PAD151935]", 6),
        ],
    ));
    let mut state =
        SequenceState::new_with_tokenizer(policy_request(), vec![TokenId::new(0)], Some(tokenizer));
    assert!(
        state.requires_full_logits_for_sampling(),
        "extended control-token masks require full logits; GPU argmax would bypass them"
    );
    let mut logits = vec![0.0f32; 7];
    logits[4] = 1.0;
    logits[5] = 90.0;
    logits[6] = 100.0;

    let token = state
        .sample_and_commit_with_processors(&mut logits)
        .unwrap();

    assert_eq!(token.get(), 5);
    assert_eq!(logits[5], 90.0);
    assert_eq!(logits[6], f32::NEG_INFINITY);
}

#[test]
fn tool_protocol_allows_only_its_added_tokens_for_auto_calls() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("x", 5),
            ("<tool_call>", 7),
            ("</tool_call>", 8),
            ("<|fim_prefix|>", 9),
        ],
    ));
    let request = policy_tool_request(
        ferrum_types::ApiToolCallProtocol::FunctionParameterXml,
        ferrum_types::ApiToolChoice::Mode("auto".to_string()),
    );
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(tokenizer),
        Some(10),
    );

    assert!(state.allowed_extended_token_ids.contains(&7));
    assert!(state.allowed_extended_token_ids.contains(&8));
    assert!(
        !state.allowed_extended_token_ids.contains(&9),
        "unrelated added-vocabulary controls must remain masked"
    );

    let mut logits = vec![f32::NEG_INFINITY; 10];
    logits[5] = 1.0;
    logits[7] = 90.0;
    logits[9] = 100.0;

    let token = state
        .sample_and_commit_with_processors(&mut logits)
        .unwrap();

    assert_eq!(token.get(), 7);
    assert_eq!(logits[7], 90.0);
    assert_eq!(logits[9], f32::NEG_INFINITY);
}

#[test]
fn harmony_protocol_allows_only_its_typed_generation_controls() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("x", 5),
            ("<|channel|>", 7),
            ("<|message|>", 8),
            ("<|fim_prefix|>", 9),
        ],
    ));
    let mut request = policy_request();
    request.sampling_params.model_output_protocol =
        ferrum_types::ModelOutputProtocol::HarmonyGptOss;
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(tokenizer),
        Some(10),
    );

    assert!(state.allowed_extended_token_ids.contains(&7));
    assert!(state.allowed_extended_token_ids.contains(&8));
    assert!(!state.allowed_extended_token_ids.contains(&9));
    let mut logits = vec![f32::NEG_INFINITY; 10];
    logits[5] = 1.0;
    logits[7] = 90.0;
    logits[9] = 100.0;
    let token = state
        .sample_and_commit_with_processors(&mut logits)
        .unwrap();
    assert_eq!(token.get(), 7);
    assert_eq!(logits[9], f32::NEG_INFINITY);
}

fn gemma_thought_policy_tokenizer() -> Arc<dyn Tokenizer + Send + Sync> {
    let mut tokenizer = PolicyTokenizer::new(
        9,
        &[
            ("{", 0),
            ("}", 1),
            ("<|channel>", 2),
            ("<turn|>", 3),
            ("<channel|>", 4),
            ("thought", 5),
            ("\n", 6),
            ("x", 7),
            ("\"", 8),
            // The unrelated control exercises extended-vocabulary rejection;
            // the native channel controls remain below the base vocabulary.
            ("<|fim_prefix|>", 9),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    tokenizer.encoded_sequences.insert(
        "<|channel>thought\n".to_string(),
        [2, 5, 6].into_iter().map(TokenId::new).collect(),
    );
    Arc::new(tokenizer)
}

#[test]
fn gemma_thought_sampling_allows_native_controls_below_base_vocab() {
    let tokenizer = gemma_thought_policy_tokenizer();
    let mut request = policy_request();
    request.sampling_params.model_output_protocol = ferrum_types::ModelOutputProtocol::GemmaThought;
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(7)],
        Some(Arc::clone(&tokenizer)),
        Some(10),
    );
    for native_control in [2, 4] {
        assert!((native_control as usize) < tokenizer.vocab_size());
        assert!(state.allowed_extended_token_ids.contains(&native_control));
    }
    assert!(!state.allowed_extended_token_ids.contains(&9));

    for expected in [2, 5, 6, 4, 7, 3] {
        let mut logits = vec![f32::NEG_INFINITY; 10];
        logits[expected] = 1.0;
        logits[9] = 100.0;
        let token = state
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap();
        assert_eq!(token, TokenId::new(expected as u32));
        assert_eq!(logits[9], f32::NEG_INFINITY);
        let stop = state.stop_reason(Some(tokenizer.as_ref()));
        if expected == 3 {
            assert_eq!(stop, Some(FinishReason::EOS));
            assert!(!state.should_stream_generated_token(Some(tokenizer.as_ref()), token, stop));
        } else {
            assert_eq!(stop, None);
            assert!(state.should_stream_generated_token(Some(tokenizer.as_ref()), token, stop));
        }
    }
}

#[test]
fn gemma_thought_structured_sampling_preserves_header_controls_and_turn_eos() {
    // Both markers are below base_vocab_size, as in Gemma's real tokenizer.
    // Exercise the SequenceState processor chain shared by run and serve:
    // protocol controls must be selectable before JSON and masked within it.
    let tokenizer = gemma_thought_policy_tokenizer();
    for allow_reasoning in [true, false] {
        let mut request = policy_request();
        request.sampling_params.max_tokens = 64;
        request.sampling_params.repetition_penalty = 2.0;
        request.sampling_params.presence_penalty = 1.5;
        request.sampling_params.model_output_protocol =
            ferrum_types::ModelOutputProtocol::GemmaThought;
        request.sampling_params.structured_output_start =
            ferrum_types::StructuredOutputStart::AfterReasoningEnvelope {
                opening: "<|channel>thought\n".to_string(),
                closing: "<channel|>".to_string(),
                allow_reasoning,
            };
        request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
            r#"{"type":"object","maxProperties":0}"#.to_string(),
        );
        let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
            request,
            vec![TokenId::new(7)],
            Some(Arc::clone(&tokenizer)),
            Some(10),
        );
        assert!(matches!(
            state.model_decode_logits_policy(),
            LogitsReturnPolicy::FullLogits
        ));

        let sample = |state: &mut SequenceState, expected: usize, forced: bool, grammar: bool| {
            let mut logits = vec![f32::NEG_INFINITY; 10];
            logits[if forced { 7 } else { expected }] = 1.0;
            logits[9] = 400.0;
            logits[3] = 300.0;
            if grammar {
                logits[2] = 200.0;
                logits[4] = 200.0;
            }
            let token = state
                .sample_and_commit_with_processors_and_tokenizer(
                    &mut logits,
                    Some(tokenizer.as_ref()),
                )
                .unwrap();
            assert_eq!(token, TokenId::new(expected as u32));
            assert_eq!(logits[9], f32::NEG_INFINITY);
            if expected != 3 {
                assert_eq!(logits[3], f32::NEG_INFINITY);
            }
            if grammar {
                assert_eq!(logits[2], f32::NEG_INFINITY);
                assert_eq!(logits[4], f32::NEG_INFINITY);
            }
            logits
        };

        // The ordinary thought/newline tokens are part of the required header,
        // even when the model assigns them negative-infinity logits.
        for token in [2, 5, 6] {
            sample(&mut state, token, true, false);
            assert_eq!(state.stop_reason(Some(tokenizer.as_ref())), None);
        }
        if allow_reasoning {
            sample(&mut state, 0, false, false);
        }
        sample(&mut state, 4, !allow_reasoning, false);
        assert!(state.sampling_history.token_frequencies().is_empty());
        assert!(state
            .structured_output_terminal_error(FinishReason::Length)
            .is_some());

        let grammar_start = state.generated_tokens.len();
        let logits = sample(&mut state, 0, false, true);
        assert_eq!(
            logits[0], 1.0,
            "reasoning must not penalize the visible JSON opener"
        );
        assert_eq!(
            state.sampling_history.scope(),
            SequenceSamplingHistoryScope::VisibleStructuredOutput {
                start_token_index: grammar_start
            }
        );
        state.reset_guided_processors().unwrap();
        sample(&mut state, 1, false, true);
        assert_eq!(
            tokenizer
                .decode(&state.generated_tokens[grammar_start..], true)
                .unwrap(),
            "{}"
        );
        assert!(state
            .structured_output_terminal_error(FinishReason::Length)
            .is_none());
        sample(&mut state, 3, false, true);
        let stop = state.stop_reason(Some(tokenizer.as_ref()));
        assert_eq!(stop, Some(FinishReason::EOS));
        assert!(!state.should_stream_generated_token(
            Some(tokenizer.as_ref()),
            TokenId::new(3),
            stop
        ));
        assert!(state
            .structured_output_terminal_error(FinishReason::EOS)
            .is_none());
    }
}

#[test]
fn harmony_structured_sampling_preserves_framing_budget_history_and_terminal() {
    const CHANNEL: u32 = 9;
    const MESSAGE: u32 = 10;
    const START: u32 = 11;
    const END: u32 = 12;
    const RETURN: u32 = 13;
    const CALL: u32 = 14;
    const UNRELATED_CONTROL: u32 = 16;
    const MODEL_VOCAB: usize = 17;
    let mut tokenizer = PolicyTokenizer::new(
        9,
        &[
            ("{", 0),
            (" ", 1),
            ("x", 2),
            ("<eos>", 3),
            ("}", 4),
            ("\"", 5),
            ("analysis", 6),
            ("final", 7),
            ("assistant", 8),
            ("<|channel|>", CHANNEL),
            ("<|message|>", MESSAGE),
            ("<|start|>", START),
            ("<|end|>", END),
            ("<|return|>", RETURN),
            ("<|call|>", CALL),
            ("<|constrain|>", 15),
            ("<|fim_prefix|>", UNRELATED_CONTROL),
        ],
    );
    tokenizer.special.bos_token = None;
    tokenizer.special.unk_token = None;
    tokenizer.special.pad_token = None;
    tokenizer.special.extra_eos_tokens = vec![TokenId::new(RETURN), TokenId::new(CALL)];
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let mut request = policy_request();
    request.sampling_params.max_tokens = 64;
    request.sampling_params.repetition_penalty = 2.0;
    request.sampling_params.presence_penalty = 1.5;
    request.sampling_params.model_output_protocol =
        ferrum_types::ModelOutputProtocol::HarmonyGptOss;
    request.sampling_params.structured_output_start =
        ferrum_types::StructuredOutputStart::HarmonyFinal;
    request.sampling_params.response_format = ferrum_types::ResponseFormat::JsonSchema(
        r#"{"type":"object","maxProperties":0}"#.to_string(),
    );
    let mut state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(8)],
        Some(Arc::clone(&tokenizer)),
        Some(MODEL_VOCAB),
    );
    assert!(matches!(
        state.model_decode_logits_policy(),
        LogitsReturnPolicy::FullLogits
    ));

    let sample = |state: &mut SequenceState, expected: u32, forcing: bool| {
        let mut logits = vec![f32::NEG_INFINITY; MODEL_VOCAB];
        logits[UNRELATED_CONTROL as usize] = 400.0;
        logits[3] = 300.0;
        logits[CALL as usize] = 250.0;
        // A forced boundary must become finite even when the model insists
        // on more reasoning. Ordinary steps retain the requested logit.
        logits[if forcing { 0 } else { expected as usize }] = 1.0;
        let token = state
            .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
            .unwrap();
        assert_eq!(token, TokenId::new(expected));
        assert_eq!(logits[UNRELATED_CONTROL as usize], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[CALL as usize], f32::NEG_INFINITY);
        logits
    };
    for token in [CHANNEL, 6, MESSAGE] {
        sample(&mut state, token, false);
    }
    let reasoning_limit = state
        .structured_output_processor
        .as_ref()
        .unwrap()
        .progress_with_terminals(&state.generated_tokens, &state.stop_token_ids)
        .unwrap()
        .budget
        .unwrap()
        .reasoning_token_limit;
    while state.generated_tokens.len() < reasoning_limit {
        sample(&mut state, 0, false);
        assert_eq!(state.stop_reason(Some(tokenizer.as_ref())), None);
        assert!(state.sampling_history.token_frequencies().is_empty());
    }
    for token in [END, START, 8, CHANNEL, 7, MESSAGE] {
        sample(&mut state, token, true);
        assert_eq!(state.stop_reason(Some(tokenizer.as_ref())), None);
    }
    assert!(state
        .structured_output_terminal_error(FinishReason::Length)
        .is_some());
    let grammar_start = state.generated_tokens.len();
    let logits = sample(&mut state, 0, false);
    assert_eq!(logits[0], 1.0, "analysis must not penalize the JSON opener");
    assert_eq!(
        state.sampling_history.scope(),
        SequenceSamplingHistoryScope::VisibleStructuredOutput {
            start_token_index: grammar_start
        }
    );

    state.reset_guided_processors().unwrap();
    sample(&mut state, 4, false);
    assert_eq!(
        state
            .sampling_history
            .previous_tokens(&state.generated_tokens)
            .unwrap(),
        &[TokenId::new(0), TokenId::new(4)]
    );
    assert!(state
        .structured_output_terminal_error(FinishReason::Length)
        .is_none());
    assert_eq!(state.stop_reason(Some(tokenizer.as_ref())), None);

    sample(&mut state, RETURN, false);
    let stop = state.stop_reason(Some(tokenizer.as_ref()));
    assert_eq!(stop, Some(FinishReason::EOS));
    assert!(state.should_stream_generated_token(
        Some(tokenizer.as_ref()),
        TokenId::new(RETURN),
        stop,
    ));
    assert!(state
        .structured_output_terminal_error(FinishReason::EOS)
        .is_none());
    assert!(state.generated_tokens.len() <= state.sampling_params.max_tokens);
    let output = tokenizer.decode(&state.generated_tokens, true).unwrap();
    let parsed = ferrum_types::parse_harmony_response(&output).unwrap();
    assert_eq!(parsed.content, "{}");
    assert!(!parsed.reasoning_content.unwrap().is_empty());
}

#[test]
fn harmony_call_and_return_terminals_are_sent_to_the_incremental_stream() {
    let mut tokenizer = PolicyTokenizer::new(
        9,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("<eos>", 3),
            ("<pad>", 4),
            ("<|call|>", 7),
            ("<|return|>", 8),
        ],
    );
    tokenizer.special.extra_eos_tokens = vec![TokenId::new(7), TokenId::new(8)];
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let mut request = policy_request();
    request.sampling_params.model_output_protocol =
        ferrum_types::ModelOutputProtocol::HarmonyGptOss;
    let mut state = SequenceState::new_with_tokenizer(
        request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );

    for (token, marker) in [
        (TokenId::new(7), "<|call|>"),
        (TokenId::new(8), "<|return|>"),
    ] {
        state.generated_tokens.clear();
        state.generated_tokens.push(token);
        let stop_reason = state.stop_reason(Some(tokenizer.as_ref()));

        assert_eq!(stop_reason, Some(FinishReason::EOS));
        assert!(state.should_stream_generated_token(Some(tokenizer.as_ref()), token, stop_reason,));
        assert_eq!(
            tokenizer.decode(&state.generated_tokens, true).unwrap(),
            marker
        );
    }
}

#[test]
fn ordinary_eos_and_untyped_harmony_terminal_are_not_sent_to_the_stream() {
    let mut tokenizer = PolicyTokenizer::new(
        8,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("<eos>", 3),
            ("<pad>", 4),
            ("<|return|>", 7),
        ],
    );
    tokenizer.special.extra_eos_tokens = vec![TokenId::new(7)];
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);

    let mut text_state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );
    for token in [TokenId::new(3), TokenId::new(7)] {
        text_state.generated_tokens.clear();
        text_state.generated_tokens.push(token);
        let stop_reason = text_state.stop_reason(Some(tokenizer.as_ref()));
        assert_eq!(stop_reason, Some(FinishReason::EOS));
        assert!(
            !text_state
                .should_stream_generated_token(Some(tokenizer.as_ref()), token, stop_reason,),
            "text protocol must not leak decoded stop token {:?}",
            tokenizer.decode(&[token], true).unwrap()
        );
    }

    let mut harmony_request = policy_request();
    harmony_request.sampling_params.model_output_protocol =
        ferrum_types::ModelOutputProtocol::HarmonyGptOss;
    let mut harmony_state = SequenceState::new_with_tokenizer(
        harmony_request,
        vec![TokenId::new(0)],
        Some(Arc::clone(&tokenizer)),
    );
    harmony_state.generated_tokens.push(TokenId::new(3));
    let stop_reason = harmony_state.stop_reason(Some(tokenizer.as_ref()));
    assert_eq!(stop_reason, Some(FinishReason::EOS));
    assert!(
        !harmony_state.should_stream_generated_token(
            Some(tokenizer.as_ref()),
            TokenId::new(3),
            stop_reason,
        ),
        "Harmony must not expose an ordinary model EOS token"
    );
}

#[test]
fn tool_protocol_does_not_widen_sampling_when_tool_choice_is_none() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("x", 5),
            ("<tool_call>", 7),
            ("</tool_call>", 8),
        ],
    ));
    let request = policy_tool_request(
        ferrum_types::ApiToolCallProtocol::FunctionParameterXml,
        ferrum_types::ApiToolChoice::Mode("none".to_string()),
    );
    let state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(tokenizer),
        Some(9),
    );

    assert!(!state.allowed_extended_token_ids.contains(&7));
    assert!(!state.allowed_extended_token_ids.contains(&8));
}

#[test]
fn json_tool_protocol_does_not_allow_xml_added_tokens() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("x", 5),
            ("<tool_call>", 7),
            ("</tool_call>", 8),
        ],
    ));
    let request = policy_tool_request(
        ferrum_types::ApiToolCallProtocol::Json,
        ferrum_types::ApiToolChoice::Mode("auto".to_string()),
    );
    let state = SequenceState::new_with_tokenizer_and_model_vocab_size(
        request,
        vec![TokenId::new(0)],
        Some(tokenizer),
        Some(9),
    );

    assert!(!state.allowed_extended_token_ids.contains(&7));
    assert!(!state.allowed_extended_token_ids.contains(&8));
}

#[test]
fn sample_resamples_hidden_non_stop_control_tokens_above_base_vocab() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("x", 5),
            ("<think>", 7),
        ],
    ));
    let mut state = SequenceState::new_with_tokenizer(
        policy_request(),
        vec![TokenId::new(0)],
        Some(tokenizer.clone()),
    );
    state.generated_tokens.push(TokenId::new(4));
    state.decoded_text_len = tokenizer
        .decode(&state.generated_tokens, true)
        .expect("generated prefix decodes")
        .len();

    assert!(
        state.allowed_extended_token_ids.contains(&7),
        "think token should be whitelisted as a generated control token"
    );
    assert!(
        !state.stop_token_ids.contains(&7),
        "think token should not be treated as a stop token"
    );

    let mut logits = vec![f32::NEG_INFINITY; 8];
    logits[5] = 1.0;
    logits[7] = 100.0;

    let token = state
        .sample_and_commit_with_processors_and_tokenizer(&mut logits, Some(tokenizer.as_ref()))
        .unwrap();

    assert_eq!(token.get(), 5);
    assert_eq!(logits[7], f32::NEG_INFINITY);
}

#[test]
fn sample_masks_metadata_initial_token_text_only_before_first_generation() {
    let tokenizer: Arc<dyn Tokenizer + Send + Sync> = Arc::new(PolicyTokenizer::new(
        6,
        &[
            ("normal", 0),
            ("<s>", 1),
            ("<unk>", 2),
            ("</s>", 3),
            ("ok", 4),
            ("blocked-once", 5),
        ],
    ));
    let mut request = policy_request();
    request.metadata.insert(
        "ferrum_initial_forbidden_token_texts".to_string(),
        serde_json::json!(["blocked-once"]),
    );
    let mut state =
        SequenceState::new_with_tokenizer(request, vec![TokenId::new(0)], Some(tokenizer));
    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = state.model_decode_logits_policy()
    else {
        panic!("first greedy decode should use the initial token mask");
    };
    assert_eq!(mask.valid_token_mask[5], 0);

    let mut first_logits = vec![0.0f32; 6];
    first_logits[0] = 1.0;
    first_logits[5] = 100.0;
    let first = state
        .sample_and_commit_with_processors(&mut first_logits)
        .unwrap();
    assert_eq!(first.get(), 0);
    assert_eq!(first_logits[5], f32::NEG_INFINITY);

    let LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(mask),
        repetition_penalty: None,
    } = state.model_decode_logits_policy()
    else {
        panic!("subsequent greedy decode should use the regular token mask");
    };
    assert_eq!(mask.valid_token_mask[5], 1);
    let mut next_logits = vec![0.0f32; 6];
    next_logits[0] = 1.0;
    next_logits[5] = 100.0;
    let next = state
        .sample_and_commit_with_processors(&mut next_logits)
        .unwrap();
    assert_eq!(next.get(), 5);
    assert_eq!(next_logits[5], 100.0);
}

mod auto_tools_json;

impl PlanRuntimeChunkedPrefillObservation {
    fn publish_release(&self) {
        let epoch = self.release_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.capacity_signal.send_replace(epoch);
    }
}

impl PlanRuntimeMaintenanceObservation {
    fn publish_release(&self) {
        let release_epoch = self.release_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.capacity_signal.send_replace(release_epoch);
    }
}

impl PlanRuntimeBatchDecodeObservation {
    fn token_frontier_for_request(&self, request: &RequestId) -> Option<ExecutionFrontier> {
        self.tokens.frontier_for_request(request)
    }
}
