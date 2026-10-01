//! Opt-in engine fixture backed by native checkpoint transfer/publication.
//! All cost observations are emitted by CompletionReaper after actual ack.
use super::*;
use ferrum_interfaces::model_executor::{
    PlanRuntimePrefixRestoreInput, PlanRuntimePrefixRestoreOutcome, PlanRuntimePrefixRestoreOutput,
    PrefixCaptureBoundary, PrefixCaptureLease, PrefixCapturePlan, PrefixCaptureRequest,
    PrefixCaptureStatus,
};
use std::{any::Any, time::Instant};
use vnext::DeviceRuntime;
mod ready;

pub(super) struct NativePrefix {
    pub(super) cost_domain: std::sync::OnceLock<Arc<CostWorkloadDomainV1>>,
    pub(super) reaper: Arc<vnext::CompletionReaper<contract::TestRuntime>>,
    entries: Mutex<Vec<Arc<Lease>>>,
    attempts: Mutex<std::collections::VecDeque<NativeAttempt>>,
    publications: Arc<Mutex<std::collections::VecDeque<NativePublishedCheckpoint>>>,
    observer: Mutex<Option<Arc<PublishedCheckpointObserver>>>,
    cancelled_sources: Mutex<std::collections::HashSet<RequestId>>,
    pub(super) admitted: Mutex<std::collections::HashMap<RequestId, vnext::TokenSpanWork>>,
}
impl NativePrefix {
    pub(super) fn new() -> Self {
        Self {
            cost_domain: std::sync::OnceLock::new(),
            reaper: vnext::CompletionReaper::new(),
            entries: Mutex::new(Vec::new()),
            attempts: Mutex::new(Default::default()),
            publications: Arc::new(Mutex::new(Default::default())),
            observer: Mutex::new(None),
            cancelled_sources: Mutex::new(Default::default()),
            admitted: Mutex::new(Default::default()),
        }
    }
    pub(super) fn install_observer(
        &self,
        sink: std::sync::Weak<dyn vnext::NativeCheckpointObservationSink>,
    ) -> Result<()> {
        let mut installed = self.observer.lock();
        if let Some(old) = installed.as_ref() {
            return if std::sync::Weak::ptr_eq(&old.destination, &sink) {
                Ok(())
            } else {
                Err(failure(
                    "checkpoint fixture cannot redirect an installed receipt sink",
                ))
            };
        }
        let observer = Arc::new(PublishedCheckpointObserver {
            destination: sink,
            publications: Arc::clone(&self.publications),
        });
        let erased: Arc<dyn vnext::NativeCheckpointObservationSink> = observer.clone();
        self.reaper
            .install_checkpoint_observation_sink(Arc::downgrade(&erased))
            .map_err(failure)?;
        *installed = Some(observer);
        Ok(())
    }
    fn record_attempt(
        &self,
        request: &RequestId,
        kind: vnext::NativeCheckpointTransferKind,
        outcome: NativeAttemptOutcome,
    ) {
        let mut attempts = self.attempts.lock();
        // Diagnostic-only ring: discarded history never affects work or samples.
        if attempts.len() == 32 {
            attempts.pop_front();
        }
        attempts.push_back(NativeAttempt {
            request: request.clone(),
            kind,
            outcome,
        });
    }
    fn retain(&self, input: PrefixCaptureRequest<'_>) -> Option<Arc<Lease>> {
        if self
            .cancelled_sources
            .lock()
            .contains(input.source_request_id)
            || input.boundary == 0
            || input.boundary >= input.source_tokens.len()
            || Instant::now() >= input.expires_at
        {
            return None;
        }
        let mut entries = self.entries.lock();
        entries.retain(|entry| entry.status() != PrefixCaptureStatus::Unavailable);
        if let Some(entry) = entries.iter().find(|entry| {
            entry.source == *input.source_request_id
                && entry.boundary == input.boundary
                && entry.tokens.as_ref() == input.source_tokens
        }) {
            return Some(entry.clone()); // Original expiry is never renewed.
        }
        // Small declared fixture population; dropping the index never fabricates
        // success and external leases continue retaining their actual checkpoint.
        if entries.len() >= 64 {
            return None;
        }
        let entry = Arc::new(Lease {
            source: input.source_request_id.clone(),
            tokens: Arc::from(input.source_tokens),
            boundary: input.boundary,
            maximum: input.maximum_sequence_tokens,
            expires: input.expires_at,
            checkpoint: Mutex::new(None),
            cancelled: AtomicBool::new(false),
        });
        entries.push(entry.clone());
        Some(entry)
    }
    pub(super) fn cancel_pending(&self, source: &RequestId) {
        self.cancelled_sources.lock().insert(source.clone());
        for entry in self.entries.lock().iter() {
            if &entry.source == source && entry.checkpoint.lock().is_none() {
                entry.cancelled.store(true, Ordering::Release);
            }
        }
    }
}
// Read-only diagnostic copies of authenticated *published* receipts. This
// wrapper is installed only through the constructor's original sink handoff;
// the original weak destination still receives each owned receipt exactly once.
#[derive(Debug, Clone)]
struct NativePublishedCheckpoint {
    identity: vnext::NativeCheckpointTransferIdentity,
    source: Option<vnext::NativeCheckpointTransferIdentity>,
    domain: vnext::NativeCheckpointTransferCostDomain,
    host_work: Option<vnext::NativeCheckpointTransferHostWork>,
}
struct PublishedCheckpointObserver {
    destination: std::sync::Weak<dyn vnext::NativeCheckpointObservationSink>,
    publications: Arc<Mutex<std::collections::VecDeque<NativePublishedCheckpoint>>>,
}
impl vnext::NativeCheckpointObservationSink for PublishedCheckpointObserver {
    fn try_record(&self, observation: vnext::NativeCheckpointTransferObservation) -> bool {
        let record = NativePublishedCheckpoint {
            identity: observation.identity().clone(),
            source: observation.source_capture_identity().cloned(),
            domain: observation.cost_domain().clone(),
            host_work: observation.host_work().copied(),
        };
        // Metadata owns no device lease and cannot grant execution or training.
        // Diagnostic contention never changes the original queue's outcome.
        if let Some(mut published) = self.publications.try_lock() {
            if published.len() == 32 {
                published.pop_front();
            }
            published.push_back(record);
        }
        self.destination
            .upgrade()
            .is_some_and(|sink| sink.try_record(observation))
    }
}
#[derive(Debug)]
struct NativeAttempt {
    request: RequestId,
    kind: vnext::NativeCheckpointTransferKind,
    outcome: NativeAttemptOutcome,
}
#[derive(Debug)]
enum NativeAttemptOutcome {
    TerminalReady,
    Capacity {
        reason: vnext::CheckpointAccessSkipReason,
        result: vnext::CheckpointCapacityMaintenanceOutcome,
    },
    CapacityReprobeExhausted(vnext::CheckpointAccessSkipReason),
    Skipped(vnext::CheckpointAccessSkipReason),
    GuardRejected(ferrum_interfaces::execution_cost::GuardedNotSubmittedReason),
}
struct Lease {
    source: RequestId,
    tokens: Arc<[ferrum_types::TokenId]>,
    boundary: usize,
    maximum: usize,
    expires: Instant,
    checkpoint: Mutex<Option<vnext::SequenceCheckpoint<contract::TestRuntime>>>,
    cancelled: AtomicBool,
}
impl std::fmt::Debug for Lease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCpuPrefixLease")
            .field("source", &self.source)
            .field("boundary", &self.boundary)
            .field("status", &self.status())
            .finish()
    }
}
impl PrefixCaptureLease for Lease {
    fn boundary(&self) -> usize {
        self.boundary
    }
    fn status(&self) -> PrefixCaptureStatus {
        if self.cancelled.load(Ordering::Acquire) || Instant::now() >= self.expires {
            PrefixCaptureStatus::Unavailable
        } else if self.checkpoint.lock().is_some() {
            PrefixCaptureStatus::Ready
        } else {
            PrefixCaptureStatus::Pending
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
fn failure(error: impl std::fmt::Display) -> FerrumError {
    FerrumError::backend(error.to_string())
}
fn unknown<T>() -> vnext::ExecutionCostRouteAvailability<T> {
    vnext::ExecutionCostRouteAvailability::Unknown(vnext::ExecutionCostRouteUnknown::Unsupported)
}
fn terminal(
    start: vnext::NativeCheckpointStart<contract::TestRuntime>,
    prefix: &NativePrefix,
    request: &RequestId,
    kind: vnext::NativeCheckpointTransferKind,
) -> Result<Option<vnext::NativeCheckpointResult<contract::TestRuntime>>> {
    use vnext::{NativeCheckpointObservation as O, NativeCheckpointStart as S};
    let mut transfer = match start {
        S::Submitted(transfer) => transfer,
        S::CapacityMaintenance {
            reason,
            maintenance,
        } => {
            let result = maintenance.try_maintain().map_err(failure)?;
            prefix.record_attempt(
                request,
                kind,
                NativeAttemptOutcome::Capacity { reason, result },
            );
            return Ok(None);
        }
        S::Skipped(reason) => {
            prefix.record_attempt(request, kind, NativeAttemptOutcome::Skipped(reason));
            return Ok(None);
        }
        S::GuardRejected(reason) => {
            prefix.record_attempt(request, kind, NativeAttemptOutcome::GuardRejected(reason));
            return Ok(None);
        }
        S::NotSubmitted(error) => return Err(failure(error)),
        S::Indeterminate(_) | S::ContractAfterSubmission { .. } => {
            return Err(failure("CPU checkpoint submission indeterminate"))
        }
    };
    // The actual CPU fixture submits synchronously. A pending/quarantined
    // operation is never promoted to a successful observation by this adapter.
    if transfer.poll().map_err(failure)? != O::Ready {
        return Err(failure("CPU checkpoint has no terminal receipt"));
    }
    let result = transfer.take_result().map_err(failure)?;
    if result.is_some() {
        prefix.record_attempt(request, kind, NativeAttemptOutcome::TerminalReady);
    }
    Ok(result)
}
impl ControlledExecutor {
    /// Actual native terminal operations, independent of observation queue
    /// availability. The proof separately checks engine/native full ACK.
    pub(in crate::continuous_engine::inner) fn native_prefix_terminal_counts(
        &self,
    ) -> (usize, usize) {
        let Some(prefix) = self.evidence.prefix.as_ref() else {
            return (0, 0);
        };
        let attempts = prefix.attempts.lock();
        let count = |kind| {
            attempts
                .iter()
                .filter(|value| {
                    value.kind == kind
                        && matches!(value.outcome, NativeAttemptOutcome::TerminalReady)
                })
                .count()
        };
        (
            count(vnext::NativeCheckpointTransferKind::Capture),
            count(vnext::NativeCheckpointTransferKind::Restore),
        )
    }

    /// Immutable numerical capability of the opt-in actual CPU program. This
    /// supplies no samples, learned model, route permission or backing lease.
    pub(super) fn declare_checkpoint_cost_domain(
        &self,
        maximum_context_tokens: NonZeroU32,
        maximum_scheduled_tokens_per_wave: NonZeroU64,
    ) {
        assert!(
            self.session_bindings.lock().is_empty(),
            "declare before requests"
        );
        assert!(maximum_context_tokens.get() as usize <= self.capabilities().max_sequence_length);
        let prefix = self
            .evidence
            .prefix
            .as_ref()
            .expect("checkpoint CPU program");
        let plan = self
            .evidence
            .fixture
            .as_ref()
            .unwrap()
            .resolved
            .execution_plan();
        let first = plan.checkpoint_byte_plan(1).unwrap();
        let maximum = plan
            .checkpoint_byte_plan(maximum_context_tokens.get().into())
            .unwrap();
        assert_eq!(
            first.logical_bytes(),
            maximum.logical_bytes(),
            "fixture state is fixed, not per-token storage"
        );
        // The controller reads logical sequence-state memory from the same
        // executor capability as a real PlanRuntime model. Its fixed native
        // state has no legacy recurrent handle and must not become zero merely
        // because that optional handle is absent on ordinary user requests.
        *self.typed_sequence_state.lock() = Some(TypedSequenceStateMemory {
            kv_bytes_per_token: 0,
            other_token_scaled_bytes_per_token: 0,
            fixed_bytes_per_sequence: first.logical_bytes(),
        });
        let ExecutorCostIdentityAvailability::Known(identity) = self.execution_cost_identity()
        else {
            panic!("actual CPU execution identity unavailable")
        };
        let domain = CostWorkloadDomainV1::new_vnext(
            &identity,
            CostWorkloadLimitsV1 {
                maximum_rows: NonZeroU32::new(self.evidence.sessions.len().try_into().unwrap())
                    .unwrap(),
                maximum_context_tokens,
                maximum_scheduled_tokens_per_wave,
                output_vocabulary_elements: NonZeroU64::new(self.info().vocab_size as u64).unwrap(),
                repetition_slot_capacity: 0,
                fixed_state_bytes_per_row: first.logical_bytes(),
            },
        )
        .unwrap();
        assert!(
            prefix.cost_domain.set(Arc::new(domain)).is_ok(),
            "domain cannot change after construction"
        );
    }
    fn prefix_session(
        &self,
        id: &RequestId,
    ) -> Option<Arc<vnext::SequenceSession<contract::TestRuntime>>> {
        let index = self
            .session_bindings
            .lock()
            .iter()
            .position(|bound| bound == id)?;
        self.evidence.sessions.get(index)
    }
    pub(super) fn prefix_boundary(
        &self,
        input: PrefixCaptureBoundary<'_>,
    ) -> Option<PrefixCapturePlan> {
        self.evidence.prefix.as_ref()?;
        let vnext::SequenceCheckpointCapability::Enabled(layout) = self
            .evidence
            .fixture
            .as_ref()?
            .plan
            .sequence_checkpoint_capability()
        else {
            return None;
        };
        let boundary = layout.shared_prefix_boundary(
            input.processed_tokens as u64,
            input.source_prompt_tokens as u64,
            input.common_prefix_tokens as u64,
            &input
                .follower_prompt_tokens
                .iter()
                .map(|n| *n as u64)
                .collect::<Vec<_>>(),
        )?;
        Some(PrefixCapturePlan {
            boundary: usize::try_from(boundary).ok()?,
            span: layout.capture_span_constraint(),
        })
    }
    pub(super) fn prefix_interest(
        &self,
        input: PrefixCaptureRequest<'_>,
    ) -> Option<Arc<dyn PrefixCaptureLease>> {
        self.prefix_session(input.source_request_id)?;
        self.evidence
            .prefix
            .as_ref()?
            .retain(input)
            .map(|lease| lease as Arc<dyn PrefixCaptureLease>)
    }
    pub(super) fn prefix_capture(
        &self,
        input: PrefixCaptureRequest<'_>,
        guard: &dyn vnext::CheckpointTransferSubmissionGuard,
    ) -> Result<bool> {
        let started = vnext::CheckpointTransferObservationStart::now();
        let Some(prefix) = self.evidence.prefix.as_ref() else {
            return Ok(false);
        };
        let Some(source) = self.prefix_session(input.source_request_id) else {
            return Ok(false);
        };
        let Some(lease) = prefix.retain(input) else {
            return Ok(false);
        };
        if lease.maximum != input.maximum_sequence_tokens
            || lease.status() == PrefixCaptureStatus::Unavailable
        {
            return Ok(false);
        }
        // The guarded production entry performs a new native capture even
        // when its index already has this boundary. A ready lookup is not a
        // new full-publication receipt for automatic maintenance calibration.
        if !self
            .native_structured_history
            .lock()
            .get(input.source_request_id)
            .is_some_and(|tokens| {
                tokens
                    .iter()
                    .copied()
                    .eq(input.source_tokens.iter().map(|token| token.get()))
            })
        {
            return Ok(false);
        }
        let fixture = self.evidence.fixture.as_ref().unwrap();
        let binding = fixture
            .plan_resources
            .trusted_runtime_binding()
            .map_err(failure)?;
        let capture = || {
            prefix
                .reaper
                .try_capture_sequence_checkpoint_guarded(
                    &fixture.plan,
                    &binding,
                    Arc::clone(&source),
                    self.evidence.lane.as_ref().unwrap().clone(),
                    vnext::DeviceTimingMode::Off,
                    started,
                    guard,
                )
                .map_err(failure)
        };
        // Match production capture_with_capacity: Ready permits one complete
        // fresh claim/encode/guard attempt. It is neither a transfer receipt nor
        // permission to chase a second shortage or retry a rejected guard.
        let start = match capture()? {
            vnext::NativeCheckpointStart::CapacityMaintenance {
                reason,
                maintenance,
            } => {
                let result = maintenance.try_maintain().map_err(failure)?;
                let ready = matches!(
                    &result,
                    vnext::CheckpointCapacityMaintenanceOutcome::Ready(_)
                );
                prefix.record_attempt(
                    input.source_request_id,
                    vnext::NativeCheckpointTransferKind::Capture,
                    NativeAttemptOutcome::Capacity { reason, result },
                );
                if !ready {
                    return Ok(false);
                }
                match capture()? {
                    vnext::NativeCheckpointStart::CapacityMaintenance { reason, .. } => {
                        prefix.record_attempt(
                            input.source_request_id,
                            vnext::NativeCheckpointTransferKind::Capture,
                            NativeAttemptOutcome::CapacityReprobeExhausted(reason),
                        );
                        return Ok(false);
                    }
                    start => start,
                }
            }
            start => start,
        };
        let Some(vnext::NativeCheckpointResult::Captured(mut checkpoint)) = terminal(
            start,
            prefix,
            input.source_request_id,
            vnext::NativeCheckpointTransferKind::Capture,
        )?
        else {
            return Ok(false);
        };
        if checkpoint.completed_tokens() != input.boundary
            || !checkpoint
                .token_prefix()
                .iter()
                .copied()
                .eq(input.source_tokens[..input.boundary]
                    .iter()
                    .map(|token| token.get()))
        {
            return Err(failure("native capture differs from requested boundary"));
        }
        let acknowledgement = checkpoint.take_publication_acknowledgement();
        // Publish exact usable lease before acknowledging the same native owner.
        *lease.checkpoint.lock() = Some(checkpoint);
        if let Some(acknowledgement) = acknowledgement {
            let mut published = lease.checkpoint.lock();
            if let Err(error) = acknowledgement.acknowledge(published.as_ref().unwrap()) {
                published.take();
                lease.cancelled.store(true, Ordering::Release);
                return Err(failure(error));
            }
        }
        Ok(true)
    }
    pub(super) fn prefix_restore(
        &self,
        input: PlanRuntimePrefixRestoreInput<'_>,
        guard: &dyn vnext::CheckpointTransferSubmissionGuard,
    ) -> Result<PlanRuntimePrefixRestoreOutcome> {
        let started = vnext::CheckpointTransferObservationStart::now();
        let Some(prefix) = self.evidence.prefix.as_ref() else {
            return Ok(PlanRuntimePrefixRestoreOutcome::Unavailable);
        };
        let Some(target) = self.prefix_session(input.request_id) else {
            return Ok(PlanRuntimePrefixRestoreOutcome::Unavailable);
        };
        let checkpoint = if let Some(lease) = input.checkpoint {
            self.prefix_checkpoint_from_lease(lease)
        } else {
            prefix
                .entries
                .lock()
                .iter()
                .filter(|lease| {
                    lease.status() == PrefixCaptureStatus::Ready
                        && input
                            .input_tokens
                            .starts_with(&lease.tokens[..lease.boundary])
                })
                .max_by_key(|lease| lease.boundary)
                .and_then(|lease| lease.checkpoint.lock().clone())
        };
        let Some(checkpoint) = checkpoint else {
            return Ok(PlanRuntimePrefixRestoreOutcome::Unavailable);
        };
        let fixture = self.evidence.fixture.as_ref().unwrap();
        let tokens: Arc<[u32]> = input.input_tokens.iter().map(|token| token.get()).collect();
        let start = prefix
            .reaper
            .try_restore_sequence_checkpoint_guarded(
                &fixture.plan,
                target,
                &checkpoint,
                tokens,
                self.evidence.lane.as_ref().unwrap().clone(),
                vnext::DeviceTimingMode::Off,
                started,
                guard,
            )
            .map_err(failure)?;
        let Some(vnext::NativeCheckpointResult::Restored(publication)) = terminal(
            start,
            prefix,
            input.request_id,
            vnext::NativeCheckpointTransferKind::Restore,
        )?
        else {
            return Ok(PlanRuntimePrefixRestoreOutcome::Unavailable);
        };
        let boundary = publication.completed_tokens();
        let request = input.request_id.clone();
        let restored = input.input_tokens[..boundary]
            .iter()
            .map(|token| token.get())
            .collect();
        let history = self.native_structured_history.clone();
        let output = PlanRuntimePrefixRestoreOutput::new(
            request.clone(),
            boundary,
            input.input_tokens.len(),
            self.cache(&request, boundary),
            move || {
                // Match production: publish the executor's usable frontier
                // before the native full-ack observer measures its endpoint.
                // Release the history lock before invoking the nonblocking sink.
                let previous = history.lock().insert(request.clone(), restored);
                if let Err(error) = publication.acknowledge() {
                    // Failed acknowledgement must not leave newly visible
                    // history that the caller could mistake for a usable restore.
                    let mut history = history.lock();
                    match previous {
                        Some(previous) => {
                            history.insert(request, previous);
                        }
                        None => {
                            history.remove(&request);
                        }
                    }
                    return Err(failure(error));
                }
                Ok(())
            },
        )?;
        Ok(PlanRuntimePrefixRestoreOutcome::Restored(output))
    }
    pub(super) fn prefix_project(
        &self,
        view: &vnext::ExecutionCostRouteView,
        state: &vnext::ExecutionCostRouteState,
        query: vnext::FutureCheckpointCostQuery<'_>,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<vnext::FutureCheckpointCostProjection> {
        if self.evidence.prefix.is_none() {
            return unknown();
        }
        let f = self.evidence.fixture.as_ref().unwrap();
        match query {
            vnext::FutureCheckpointCostQuery::Capture {
                source,
                span_start,
                boundary,
                prompt_tokens,
            } => f.plan_resources.project_future_checkpoint_capture(
                view,
                state,
                &f.plan,
                f.runtime.descriptor(),
                source,
                span_start,
                boundary,
                prompt_tokens,
                budget,
            ),
            vnext::FutureCheckpointCostQuery::Restore {
                checkpoint,
                target,
                prompt_tokens,
            } => f.plan_resources.project_future_checkpoint_restore(
                view,
                state,
                &f.plan,
                f.runtime.descriptor(),
                checkpoint,
                target,
                prompt_tokens,
                budget,
            ),
        }
    }
    pub(super) fn prefix_bind(
        &self,
        view: &vnext::ExecutionCostRouteView,
        state: &vnext::ExecutionCostRouteState,
        lease: &dyn PrefixCaptureLease,
        source: usize,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<vnext::FutureRetainedCheckpointBinding> {
        let Some(lease) = lease.as_any().downcast_ref::<Lease>() else {
            return unknown();
        };
        if lease.status() != PrefixCaptureStatus::Ready {
            return unknown();
        }
        let checkpoint = lease.checkpoint.lock();
        let Some(checkpoint) = checkpoint.as_ref() else {
            return unknown();
        };
        let f = self.evidence.fixture.as_ref().unwrap();
        f.plan_resources
            .bind_future_retained_checkpoint(view, state, &f.plan, checkpoint, source, budget)
    }
    pub(super) fn prefix_restored(
        &self,
        view: &vnext::ExecutionCostRouteView,
        lease: &dyn PrefixCaptureLease,
        target: usize,
        budget: &mut dyn vnext::ResourcePlanningBudget,
    ) -> vnext::ExecutionCostRouteAvailability<bool> {
        let Some(checkpoint) = self.prefix_checkpoint_from_lease(lease) else {
            return unknown();
        };
        match self
            .evidence
            .fixture
            .as_ref()
            .unwrap()
            .plan_resources
            .checkpoint_restore_completed(view.resource_view(), &checkpoint, target, budget)
        {
            vnext::ResourcePlanningAvailability::Known(value) => {
                vnext::ExecutionCostRouteAvailability::Known(value)
            }
            vnext::ResourcePlanningAvailability::Unknown(reason) => {
                vnext::ExecutionCostRouteAvailability::Unknown(
                    vnext::ExecutionCostRouteUnknown::Resource(reason),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests;
