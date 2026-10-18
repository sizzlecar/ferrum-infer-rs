//! Cold-path observation of exact executor closure. This registry owns product
//! obligations, not native release authority; the executor/reaper independently
//! retains physical resources while closure is held or unproved.

use super::*;
use ferrum_execution::token_execution::{
    ExecutionCloseDisposition, ExecutionCloseOutcome, ExecutionEndReason, ExecutionEnded,
    ExecutionFrontier, ExecutionRecoveryKey, ExecutionTerminalOutcome, ExecutorSequence,
};

mod recompute;

pub(super) struct RecomputeClose {
    request: Weak<SequenceHandle>,
    frontier: ExecutionFrontier,
    transaction: PressureYieldTransaction,
    attempted_width: usize,
    observed_free_blocks: Option<usize>,
    readiness: ExecutionReadinessWake,
    failure_targets: std::collections::VecDeque<RecomputeFailureTarget>,
    failure_error: Option<FerrumError>,
    failure_marked: bool,
}

#[derive(Clone)]
struct RecomputeFailureTarget {
    request: Weak<SequenceHandle>,
    key: ferrum_scheduler::implementations::continuous::PrefixRequestKey,
    frontier: Option<ExecutionFrontier>,
}

struct ReadyExecutionClose {
    sequence: ExecutorSequence,
    request: Option<Weak<SequenceHandle>>,
}

enum ClosePurpose {
    Finalize { recompute: Option<RecomputeClose> },
    Recompute(RecomputeClose),
    // A stale callback has no product slot or new release obligation. Keep its
    // real executor observation, but do not count the same physical end again.
    DetachedObservation,
}

impl Default for ClosePurpose {
    fn default() -> Self {
        Self::Finalize { recompute: None }
    }
}

impl ClosePurpose {
    fn recompute(&self) -> Option<&RecomputeClose> {
        match self {
            Self::Recompute(recompute)
            | Self::Finalize {
                recompute: Some(recompute),
            } => Some(recompute),
            _ => None,
        }
    }

    fn recompute_mut(&mut self) -> Option<&mut RecomputeClose> {
        match self {
            Self::Recompute(recompute)
            | Self::Finalize {
                recompute: Some(recompute),
            } => Some(recompute),
            _ => None,
        }
    }
}

enum CloseObservation {
    Pending,
    Ended(ExecutionEnded, Option<FerrumError>),
    Unproved(FerrumError),
}

struct CloseEntry {
    request_id: RequestId,
    slot: Option<RequestSlotLease>,
    // Retain the original executor independently of its close/wait future.
    _executor: Arc<dyn ModelExecutor + Send + Sync>,
    recovery: Option<ExecutionRecoveryKey>,
    observation: CloseObservation,
    operation_error: Option<FerrumError>,
    operation_error_reported: bool,
    observation_error_reported: bool,
    // Invalid protocol output still owns its original observation future.
    // It cannot close the expected slot, and must not be discarded on rejection.
    retained_invalid_outcome: Option<ExecutionCloseOutcome>,
    purpose: ClosePurpose,
    initial_result: Option<Result<()>>,
    initial_waiters: Vec<tokio::sync::oneshot::Sender<Result<()>>>,
    additional_observations: Vec<ExecutionCloseOutcome>,
    additional_errors_reported: usize,
    report_unreclaimed: bool,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl CloseEntry {
    fn observe(&mut self, observation: CloseObservation) {
        if let CloseObservation::Unproved(error) = &observation {
            if let Some(recompute) = self.purpose.recompute_mut() {
                recompute.failure_error.get_or_insert_with(|| error.clone());
            }
        }
        self.observation = observation;
    }

    fn has_pending_product_failure(&self) -> bool {
        self.purpose.recompute().is_some_and(|recompute| {
            recompute.failure_error.is_some() && !recompute.failure_targets.is_empty()
        })
    }

    fn publish_initial(&mut self, result: Result<()>) {
        if self.initial_result.is_none() {
            for waiter in self.initial_waiters.drain(..) {
                let _ = waiter.send(result.clone());
            }
            self.initial_result = Some(result);
        }
    }

    fn subscribe_initial(&mut self) -> ExecutionCloseTicket {
        let (sender, first_result) = tokio::sync::oneshot::channel();
        if let Some(result) = &self.initial_result {
            let _ = sender.send(result.clone());
        } else {
            self.initial_waiters.push(sender);
        }
        ExecutionCloseTicket { first_result }
    }
}

impl Drop for CloseEntry {
    fn drop(&mut self) {
        let terminal_proved = matches!(self.observation, CloseObservation::Ended(..));
        if self.report_unreclaimed && (self.slot.is_some() || !terminal_proved) {
            // Engine destruction is not a release proof. Suppress only the
            // lease's misuse panic after explicitly reporting unreclaimed exit;
            // do not emit RequestClose, Release, or a scheduler capacity wake.
            warn!(
                request_id = %self.request_id,
                admitted = self.slot.as_ref().is_some_and(|slot| slot.admitted),
                terminal_proved,
                invalid_observation_retained = self.retained_invalid_outcome.is_some(),
                "engine exited with an unconsumed execution-close obligation"
            );
        }
        if let Some(mut slot) = self.slot.take() {
            slot.armed = false;
        }
    }
}

#[derive(Default)]
struct CloseState {
    entries: HashMap<ExecutorSequence, CloseEntry>,
    // An invalid duplicate must not replace the original owner or drop the
    // newly transferred slot. Keep this error-only ownership until engine exit.
    rejected: Vec<CloseEntry>,
}

#[derive(Default)]
pub(super) struct ExecutionCloseRegistry {
    state: Arc<Mutex<CloseState>>,
}

enum CloseStart {
    Execute(ExecutionEndReason),
    Observed(ExecutionCloseOutcome),
}

pub(super) struct ExecutionCloseTicket {
    first_result: tokio::sync::oneshot::Receiver<Result<()>>,
}

impl ExecutionCloseTicket {
    /// Initial close-operation error, independent of eventual physical closure.
    pub(super) async fn initial_result(self) -> Result<()> {
        self.first_result.await.map_err(|_| {
            FerrumError::internal(
                "execution close task lost its initial result; ownership remains unproved",
            )
        })?
    }
}

impl ExecutionCloseRegistry {
    /// Cold pressure observation of original in-flight close owners. An
    /// unproved failed close is not a promise of a future release wake.
    pub(super) fn pending_incarnations(&self) -> Vec<(RequestId, ExecutorSequence)> {
        self.state
            .lock()
            .entries
            .iter()
            .filter_map(|(sequence, entry)| {
                matches!(entry.observation, CloseObservation::Pending)
                    .then(|| (entry.request_id.clone(), sequence.clone()))
            })
            .collect()
    }

    pub(super) fn contains_incarnation(&self, sequence: &ExecutorSequence) -> bool {
        self.state.lock().entries.contains_key(sequence)
    }

    fn start(
        &self,
        request_id: RequestId,
        sequence: ExecutorSequence,
        slot: Option<RequestSlotLease>,
        executor: Arc<dyn ModelExecutor + Send + Sync>,
        start: CloseStart,
        purpose: ClosePurpose,
        work_notify: Arc<Notify>,
    ) -> Result<ExecutionCloseTicket> {
        let mut entry = CloseEntry {
            request_id,
            slot,
            _executor: Arc::clone(&executor),
            recovery: None,
            observation: CloseObservation::Pending,
            operation_error: None,
            operation_error_reported: false,
            observation_error_reported: false,
            retained_invalid_outcome: None,
            purpose,
            initial_result: None,
            initial_waiters: Vec::new(),
            additional_observations: Vec::new(),
            additional_errors_reported: 0,
            report_unreclaimed: true,
            task: None,
        };
        let mut state = self.state.lock();
        if let Some(existing) = state.entries.get_mut(&sequence) {
            if matches!(entry.purpose, ClosePurpose::DetachedObservation)
                && existing.request_id == entry.request_id
                && entry.slot.is_none()
            {
                // The original close owns any capacity notification. Joining
                // it cannot downgrade Finalize/Recompute or start a new close.
                if let CloseStart::Observed(outcome) = start {
                    let actual = match &outcome.disposition {
                        ExecutionCloseDisposition::Ended(ended) => ended.sequence(),
                        ExecutionCloseDisposition::Held(held) => held.key().sequence(),
                    };
                    if actual != &sequence {
                        let error =
                            FerrumError::internal("detached close received another incarnation");
                        entry.observation = CloseObservation::Unproved(error.clone());
                        entry.operation_error = outcome.error.clone();
                        entry.retained_invalid_outcome = Some(outcome);
                        state.rejected.push(entry);
                        work_notify.notify_one();
                        return Err(error);
                    }
                    existing.additional_observations.push(outcome);
                }
                entry.report_unreclaimed = false;
                return Ok(existing.subscribe_initial());
            }
            if matches!(existing.purpose, ClosePurpose::Recompute(_))
                && matches!(entry.purpose, ClosePurpose::Finalize { .. })
                && existing.request_id == entry.request_id
                && entry.slot.is_none()
            {
                // Cancellation upgrades the original close task. Its actual
                // first result and retained owner are shared, never restarted.
                if let CloseStart::Observed(outcome) = start {
                    let actual = match &outcome.disposition {
                        ExecutionCloseDisposition::Ended(ended) => ended.sequence(),
                        ExecutionCloseDisposition::Held(held) => held.key().sequence(),
                    };
                    if actual != &sequence {
                        let error = FerrumError::internal(
                            "recompute finalization received another incarnation",
                        );
                        entry.observation = CloseObservation::Unproved(error.clone());
                        entry.retained_invalid_outcome = Some(outcome);
                        state.rejected.push(entry);
                        work_notify.notify_one();
                        return Err(error);
                    }
                    existing.additional_observations.push(outcome);
                }
                let ClosePurpose::Recompute(recompute) = std::mem::take(&mut existing.purpose)
                else {
                    unreachable!()
                };
                existing.purpose = ClosePurpose::Finalize {
                    recompute: Some(recompute),
                };
                // This temporary registration never owned a physical close.
                entry.report_unreclaimed = false;
                return Ok(existing.subscribe_initial());
            }
        }
        let invalid = if entry
            .slot
            .as_ref()
            .is_some_and(|slot| slot.request_id != entry.request_id)
        {
            Some(FerrumError::internal(
                "execution close received another request's slot",
            ))
        } else if state.entries.contains_key(&sequence) {
            Some(FerrumError::internal(
                "execution close registered the same incarnation twice",
            ))
        } else {
            None
        };
        if let Some(error) = invalid {
            entry.observation = CloseObservation::Unproved(error.clone());
            if let CloseStart::Observed(outcome) = start {
                entry.operation_error = outcome.error.clone();
                entry.retained_invalid_outcome = Some(outcome);
            }
            state.rejected.push(entry);
            work_notify.notify_one();
            return Err(error);
        }
        let ticket = entry.subscribe_initial();
        state.entries.insert(sequence.clone(), entry);

        let shared = Arc::clone(&self.state);
        let task_sequence = sequence.clone();
        let task = tokio::spawn(async move {
            let observed = std::panic::AssertUnwindSafe(async {
                let outcome = match start {
                    CloseStart::Execute(reason) => {
                        // A prior terminal fact proves release, but does not
                        // replace executor bookkeeping or finish validation.
                        let result = match reason {
                            ExecutionEndReason::Cancelled => executor.cancel_tokens(&task_sequence),
                            completed => executor.finish_tokens(&task_sequence, completed).await,
                        };
                        match result {
                            Ok(outcome) => outcome,
                            Err(error) => match task_sequence.observed_end() {
                                Some(ended) => {
                                    ExecutionCloseOutcome::ended(ended).with_error(error)
                                }
                                None => return Err(error),
                            },
                        }
                    }
                    CloseStart::Observed(outcome) => outcome,
                };
                {
                    let mut state = shared.lock();
                    let entry = state
                        .entries
                        .get_mut(&task_sequence)
                        .expect("registered close must remain owned until terminal consumption");
                    entry.operation_error = outcome.error.clone();
                }
                let actual_sequence = match &outcome.disposition {
                    ExecutionCloseDisposition::Ended(ended) => ended.sequence(),
                    ExecutionCloseDisposition::Held(held) => held.key().sequence(),
                };
                if actual_sequence != &task_sequence {
                    let error =
                        FerrumError::internal("execution close returned another incarnation");
                    let mut state = shared.lock();
                    let request_id = state.entries[&task_sequence].request_id.clone();
                    // Keep the unexpected observation separately: a genuine
                    // Ended for the expected identity may still arrive, but
                    // consuming that slot must not drop this foreign Held.
                    state.rejected.push(CloseEntry {
                        request_id,
                        slot: None,
                        _executor: Arc::clone(&executor),
                        recovery: None,
                        observation: CloseObservation::Unproved(error.clone()),
                        operation_error: None,
                        operation_error_reported: false,
                        observation_error_reported: false,
                        retained_invalid_outcome: Some(outcome),
                        purpose: ClosePurpose::default(),
                        initial_result: None,
                        initial_waiters: Vec::new(),
                        additional_observations: Vec::new(),
                        additional_errors_reported: 0,
                        report_unreclaimed: true,
                        task: None,
                    });
                    return Err(error);
                }
                let initial_result = outcome.error.clone().map_or(Ok(()), Err);
                match outcome.disposition {
                    ExecutionCloseDisposition::Ended(ended) => {
                        Ok(ExecutionTerminalOutcome::ended(ended))
                    }
                    ExecutionCloseDisposition::Held(held) => {
                        {
                            let mut state = shared.lock();
                            let entry = state
                                .entries
                                .get_mut(&task_sequence)
                                .expect("held close remains registered");
                            entry.recovery = Some(held.key().clone());
                            entry.publish_initial(initial_result);
                        }
                        work_notify.notify_one();
                        held.wait().await
                    }
                }
            })
            .catch_unwind()
            .await;
            let result = match observed {
                Ok(result) => result,
                Err(_) => Err(FerrumError::internal("execution close observer panicked")),
            };
            let result_error = match &result {
                Ok(terminal) => terminal.error.clone(),
                Err(error) => Some(error.clone()),
            };
            let observation = match result {
                Ok(terminal) if terminal.ended.sequence() == &task_sequence => {
                    CloseObservation::Ended(terminal.ended, terminal.error)
                }
                Ok(_) => CloseObservation::Unproved(FerrumError::internal(
                    "execution close returned another incarnation",
                )),
                Err(error) => match task_sequence.observed_end() {
                    Some(ended) => CloseObservation::Ended(ended, Some(error)),
                    None => CloseObservation::Unproved(error),
                },
            };
            {
                let mut state = shared.lock();
                let entry = state
                    .entries
                    .get_mut(&task_sequence)
                    .expect("close observer retains its registered incarnation");
                entry.observe(observation);
                let initial_result = entry
                    .operation_error
                    .clone()
                    .or(result_error)
                    .map_or(Ok(()), Err);
                // Publish only after the actual terminal/unproved fact is
                // retained; a woken consumer may immediately drain this entry.
                entry.publish_initial(initial_result);
            }
            work_notify.notify_one();
        });
        state
            .entries
            .get_mut(&sequence)
            .expect("just registered")
            .task = Some(task);
        Ok(ticket)
    }

    pub(super) async fn join_observers(&self) {
        // Never abort a close task to manufacture quiescence. Dropping this
        // join future detaches only observation; records and executor owners
        // remain retained by the registry and running tasks.
        let tasks = {
            let mut state = self.state.lock();
            state
                .entries
                .iter_mut()
                .filter_map(|(sequence, entry)| {
                    entry.task.take().map(|task| (sequence.clone(), task))
                })
                .collect::<Vec<_>>()
        };
        for (sequence, task) in tasks {
            if let Err(error) = task.await {
                if let Some(entry) = self.state.lock().entries.get_mut(&sequence) {
                    if !matches!(entry.observation, CloseObservation::Ended(..)) {
                        let error = FerrumError::internal(format!(
                            "execution close task did not establish closure: {error}"
                        ));
                        let observation = match sequence.observed_end() {
                            Some(ended) => CloseObservation::Ended(ended, Some(error.clone())),
                            None => CloseObservation::Unproved(error.clone()),
                        };
                        entry.observe(observation);
                        entry.publish_initial(Err(error));
                    }
                }
            }
        }
    }

    fn observe_ended_and_errors(
        &self,
        scheduler: &ContinuousBatchScheduler,
    ) -> (Vec<ReadyExecutionClose>, Vec<FerrumError>, bool) {
        let mut state = self.state.lock();
        let mut errors = Vec::new();
        for entry in state.entries.values_mut() {
            if let Some(recompute) = entry.purpose.recompute_mut() {
                if recompute.failure_error.is_some() && !recompute.failure_marked {
                    // A failed observer retains its fence. Record that it is
                    // no longer a future release witness for new pressure.
                    if let Err(error) =
                        scheduler.mark_execution_capacity_yield_unproved(&recompute.transaction)
                    {
                        errors.push(error);
                    }
                    recompute.failure_marked = true;
                }
            }
        }
        // A failed observer is not a terminal proof. If the original executor
        // subsequently publishes its actual end, consume that fact without
        // erasing the observer's error or issuing another physical close.
        for (sequence, entry) in &mut state.entries {
            if let CloseObservation::Unproved(error) = &entry.observation {
                if let Some(ended) = sequence.observed_end() {
                    entry.observation = CloseObservation::Ended(ended, Some(error.clone()));
                }
            }
        }
        let ended = state
            .entries
            .iter()
            .filter_map(|(sequence, entry)| {
                (matches!(entry.observation, CloseObservation::Ended(..))
                    && !entry.has_pending_product_failure())
                .then(|| ReadyExecutionClose {
                    sequence: sequence.clone(),
                    request: entry
                        .purpose
                        .recompute()
                        .map(|recompute| recompute.request.clone()),
                })
            })
            .collect::<Vec<_>>();
        let mut newly_unproved = false;
        let CloseState { entries, rejected } = &mut *state;
        for entry in entries.values_mut().chain(rejected.iter_mut()) {
            for outcome in &entry.additional_observations[entry.additional_errors_reported..] {
                if let Some(error) = &outcome.error {
                    errors.push(error.clone());
                }
            }
            entry.additional_errors_reported = entry.additional_observations.len();

            if !entry.operation_error_reported {
                if let Some(error) = &entry.operation_error {
                    errors.push(error.clone());
                    entry.operation_error_reported = true;
                }
            }
            if !entry.observation_error_reported {
                if let CloseObservation::Unproved(error) | CloseObservation::Ended(_, Some(error)) =
                    &entry.observation
                {
                    errors.push(error.clone());
                    newly_unproved |= matches!(entry.observation, CloseObservation::Unproved(_));
                    entry.observation_error_reported = true;
                }
            }
        }
        (ended, errors, newly_unproved)
    }

    /// Recheck and claim only after the original recompute request is locked.
    /// Removing a recompute before that lock would let concurrent completion
    /// register a second close instead of upgrading the retained obligation.
    fn take_ended(&self, sequence: &ExecutorSequence) -> Option<CloseEntry> {
        let mut state = self.state.lock();
        let ready = state.entries.get(sequence).is_some_and(|entry| {
            matches!(entry.observation, CloseObservation::Ended(..))
                && !entry.has_pending_product_failure()
        });
        ready.then(|| {
            state
                .entries
                .remove(sequence)
                .expect("checked close remains registered")
        })
    }

    fn next_recompute_failure(
        &self,
    ) -> Option<(ExecutorSequence, RecomputeFailureTarget, FerrumError)> {
        let state = self.state.lock();
        state.entries.iter().find_map(|(sequence, entry)| {
            let recompute = entry.purpose.recompute()?;
            Some((
                sequence.clone(),
                recompute.failure_targets.front()?.clone(),
                recompute.failure_error.as_ref()?.clone(),
            ))
        })
    }

    fn acknowledge_recompute_failure(
        &self,
        sequence: &ExecutorSequence,
        key: &ferrum_scheduler::implementations::continuous::PrefixRequestKey,
    ) {
        let mut state = self.state.lock();
        if let Some(recompute) = state
            .entries
            .get_mut(sequence)
            .and_then(|entry| entry.purpose.recompute_mut())
        {
            if recompute
                .failure_targets
                .front()
                .is_some_and(|target| &target.key == key)
            {
                recompute.failure_targets.pop_front();
            }
        }
    }

    pub(super) fn unreclaimed_count(&self) -> usize {
        let state = self.state.lock();
        state.entries.len() + state.rejected.len()
    }
}

impl EngineInner {
    /// Preserve an unassignable executor observation without associating it
    /// with a live incarnation or transferring any request-slot authority.
    /// The request ID is diagnostic provenance only.
    pub(super) fn retain_unassigned_execution_close(
        &self,
        request_id: RequestId,
        outcome: ExecutionCloseOutcome,
        error: FerrumError,
    ) {
        self.execution_closes
            .state
            .lock()
            .rejected
            .push(CloseEntry {
                request_id,
                slot: None,
                _executor: Arc::clone(&self.model_executor),
                recovery: None,
                observation: CloseObservation::Unproved(error),
                operation_error: outcome.error.clone(),
                operation_error_reported: false,
                observation_error_reported: false,
                retained_invalid_outcome: Some(outcome),
                purpose: ClosePurpose::default(),
                initial_result: None,
                initial_waiters: Vec::new(),
                additional_observations: Vec::new(),
                additional_errors_reported: 0,
                report_unreclaimed: true,
                task: None,
            });
        self.work_notify.notify_one();
    }

    /// Call after logical scheduler termination. Ownership transfers before any
    /// async close work; cancellation of the client/driver future cannot lose it.
    pub(super) fn begin_execution_close(
        &self,
        request_id: RequestId,
        sequence: ExecutorSequence,
        slot: Option<RequestSlotLease>,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseTicket> {
        self.execution_closes.start(
            request_id,
            sequence,
            slot,
            Arc::clone(&self.model_executor),
            CloseStart::Execute(reason),
            ClosePurpose::default(),
            Arc::clone(&self.work_notify),
        )
    }

    pub(super) fn begin_observed_close(
        &self,
        request_id: RequestId,
        sequence: ExecutorSequence,
        slot: Option<RequestSlotLease>,
        outcome: ExecutionCloseOutcome,
    ) -> Result<ExecutionCloseTicket> {
        self.execution_closes.start(
            request_id,
            sequence,
            slot,
            Arc::clone(&self.model_executor),
            CloseStart::Observed(outcome),
            ClosePurpose::default(),
            Arc::clone(&self.work_notify),
        )
    }

    /// Observe a stale incarnation after its product ownership has moved to
    /// an existing close obligation or a replacement. This does not represent
    /// a newly admitted resource owner. In particular, unexpected *fresh*
    /// admission cleanup must still use `begin_execution_close`, even without
    /// a request slot.
    pub(super) fn begin_detached_execution_close(
        &self,
        request_id: RequestId,
        sequence: ExecutorSequence,
        outcome: Option<ExecutionCloseOutcome>,
    ) -> Result<ExecutionCloseTicket> {
        self.execution_closes.start(
            request_id,
            sequence,
            None,
            Arc::clone(&self.model_executor),
            outcome.map_or(
                CloseStart::Execute(ExecutionEndReason::Cancelled),
                CloseStart::Observed,
            ),
            ClosePurpose::DetachedObservation,
            Arc::clone(&self.work_notify),
        )
    }

    pub(super) fn drain_execution_closes(&self) -> Result<()> {
        // Snapshot without consuming the close owner. Completion may still
        // upgrade Recompute to Finalize while holding its original request.
        // Never wait for that request while holding the close registry lock.
        let (ended, mut errors, newly_unproved) = self
            .execution_closes
            .observe_ended_and_errors(&self.scheduler);
        for ready in ended {
            let request = ready.request.and_then(|request| request.upgrade());
            let mut state = request.as_ref().map(|request| request.lock());
            let Some(mut entry) = self.execution_closes.take_ended(&ready.sequence) else {
                continue;
            };
            let CloseObservation::Ended(ended, _) = &entry.observation else {
                unreachable!("only checked terminal entries are removed")
            };
            debug!(request_id = %entry.request_id, sequence = ?ended.sequence(), "execution close confirmed");
            let identity = ended.sequence().clone();
            let mut victim_requeued = false;
            let notify_capacity = !matches!(entry.purpose, ClosePurpose::DetachedObservation);
            match std::mem::take(&mut entry.purpose) {
                ClosePurpose::Recompute(recompute) => {
                    let (requeued, result) = self.finish_execution_recompute(
                        request.as_ref(),
                        state.as_mut().and_then(|state| state.as_mut()),
                        &entry.request_id,
                        &identity,
                        recompute,
                        &mut entry.slot,
                    );
                    victim_requeued = requeued;
                    if let Err(error) = result {
                        errors.push(error);
                    }
                }
                ClosePurpose::Finalize {
                    recompute: Some(recompute),
                } => {
                    recompute.readiness.cancel();
                    if let Err(error) = self.scheduler.abort_execution_capacity_yield(
                        &recompute.transaction,
                        false,
                        recompute.attempted_width,
                        recompute.observed_free_blocks,
                    ) {
                        errors.push(error);
                    }
                }
                ClosePurpose::Finalize { recompute: None } => {}
                ClosePurpose::DetachedObservation => {}
            }
            // Recompute installs its new waiting frontier before this wake.
            // Detached observations retain actual Ended too, but have no new
            // product release obligation to notify after the original drain.
            if notify_capacity {
                if victim_requeued {
                    self.scheduler
                        .record_external_capacity_release_after_recompute();
                } else {
                    self.scheduler.record_external_capacity_release();
                }
            }
            if let Some(slot) = entry.slot.take() {
                slot.close(self);
            }
        }
        if newly_unproved && self.scheduler.invalidate_allocator_capacity_waits() {
            // Losing a pending release witness is not a physical release.
            // Re-probe the allocator and re-evaluate actual contributors;
            // keep the failed close owner and all original fences retained.
            self.work_notify.notify_one();
        }
        for error in &errors {
            warn!(error = %error, "execution close reported failure; unproved owners remain retained");
        }
        errors.into_iter().next().map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests;
