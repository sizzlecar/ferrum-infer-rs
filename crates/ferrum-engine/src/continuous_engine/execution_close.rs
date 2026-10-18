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
    frontier: ExecutionFrontier,
    transaction: PressureYieldTransaction,
    attempted_width: usize,
    observed_free_blocks: Option<usize>,
    readiness: ExecutionReadinessWake,
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
                entry.observation = observation;
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
                        entry.observation = match sequence.observed_end() {
                            Some(ended) => CloseObservation::Ended(ended, Some(error.clone())),
                            None => CloseObservation::Unproved(error.clone()),
                        };
                        entry.publish_initial(Err(error));
                    }
                }
            }
        }
    }

    fn take_ended_and_errors(&self) -> (Vec<CloseEntry>, Vec<FerrumError>) {
        let mut state = self.state.lock();
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
                matches!(entry.observation, CloseObservation::Ended(..)).then(|| sequence.clone())
            })
            .collect::<Vec<_>>();
        let mut errors = Vec::new();
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
                    entry.observation_error_reported = true;
                }
            }
        }
        let ended = ended
            .into_iter()
            .map(|sequence| state.entries.remove(&sequence).unwrap())
            .collect();
        (ended, errors)
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
        // Serialize terminal consumption with the product's synchronous
        // frontier/slot handoff. A completion must register its original
        // incarnation while holding this same outer lock, before awaiting.
        let mut sequences = self.sequences.write();
        let (ended, mut errors) = self.execution_closes.take_ended_and_errors();
        for mut entry in ended {
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
                        &mut sequences,
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
        drop(sequences);
        for error in &errors {
            warn!(error = %error, "execution close reported failure; unproved owners remain retained");
        }
        errors.into_iter().next().map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests;
