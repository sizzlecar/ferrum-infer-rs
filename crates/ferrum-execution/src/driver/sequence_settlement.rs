//! Ownership retained by the existing sequence registry when a wave is abandoned.
//!
//! Completion handles observe the original reaper; none of these product types
//! manufacture native authority or maintain a second physical resource ledger.

use super::*;

pub(super) struct SequenceSettlement<R: DeviceRuntime> {
    pub(super) closing: bool,
    pub(super) cancel_requested: bool,
    pub(super) terminal: Option<SequenceSessionTerminalReceipt>,
    held: Option<Arc<HeldCohort<R>>>,
    observation_error: Option<FerrumError>,
    terminal_waiters:
        Vec<tokio::sync::oneshot::Sender<crate::token_execution::ExecutionTerminalOutcome>>,
}

impl<R: DeviceRuntime> Default for SequenceSettlement<R> {
    fn default() -> Self {
        Self {
            closing: false,
            cancel_requested: false,
            terminal: None,
            held: None,
            observation_error: None,
            terminal_waiters: Vec::new(),
        }
    }
}

enum WaveObserver<R: DeviceRuntime> {
    Completion(CompletionHandle<R>),
    Indeterminate(IndeterminateSubmissionHandle<R>),
}

impl<R: DeviceRuntime> Clone for WaveObserver<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Completion(handle) => Self::Completion(handle.clone()),
            Self::Indeterminate(handle) => Self::Indeterminate(handle.clone()),
        }
    }
}

impl<R: DeviceRuntime> WaveObserver<R> {
    fn identity(&self) -> &BatchOperationIdentity {
        match self {
            Self::Completion(handle) => handle.batch_identity(),
            Self::Indeterminate(handle) => handle.batch_identity(),
        }
    }
}

pub(super) enum RetainedWaveObservation {
    Readbacks(std::result::Result<CompletionReadbackBatchObservation, VNextError>),
    Completion(std::result::Result<CompletionObservation, VNextError>),
    Recovery(std::result::Result<CompletionRecoveryOutcome, VNextError>),
}

impl RetainedWaveObservation {
    fn terminal_completion(&self) -> Option<&OperationCompletionReceipt> {
        match self {
            Self::Readbacks(Ok(CompletionReadbackBatchObservation::Terminal(receipt))) => {
                Some(receipt.completion())
            }
            Self::Completion(Ok(CompletionObservation::Terminal(receipt))) => Some(receipt),
            _ => None,
        }
    }
    fn release_readback_payload(self) -> Self {
        match self {
            Self::Readbacks(Ok(CompletionReadbackBatchObservation::Terminal(receipt))) => {
                Self::Completion(Ok(CompletionObservation::Terminal(
                    receipt.into_completion(),
                )))
            }
            other => other,
        }
    }
    fn failure_progress(&self) -> WaveProgress {
        match self {
            Self::Readbacks(Ok(CompletionReadbackBatchObservation::Terminal(receipt))) => {
                WaveProgress::Terminal(receipt.completion().clone())
            }
            Self::Completion(Ok(CompletionObservation::Terminal(receipt))) => {
                WaveProgress::Terminal(receipt.clone())
            }
            Self::Recovery(Ok(CompletionRecoveryOutcome::Drained(receipt))) => {
                WaveProgress::Drained(receipt.clone())
            }
            _ => WaveProgress::Unknown,
        }
    }
    fn terminal_fact(&self) -> Option<bool> {
        let completion = match self {
            Self::Readbacks(Ok(CompletionReadbackBatchObservation::Terminal(receipt))) => {
                receipt.completion()
            }
            Self::Completion(Ok(CompletionObservation::Terminal(receipt))) => receipt,
            Self::Recovery(Ok(CompletionRecoveryOutcome::Drained(_))) => return Some(false),
            _ => return None,
        };
        Some(matches!(
            completion.disposition(),
            OperationCompletionDisposition::Succeeded
        ))
    }
}

pub(super) trait RetainWaveObservation: Send + 'static {
    fn into_retained(self) -> RetainedWaveObservation;
}

impl RetainWaveObservation for std::result::Result<CompletionReadbackBatchObservation, VNextError> {
    fn into_retained(self) -> RetainedWaveObservation {
        RetainedWaveObservation::Readbacks(self)
    }
}
impl RetainWaveObservation for std::result::Result<CompletionObservation, VNextError> {
    fn into_retained(self) -> RetainedWaveObservation {
        RetainedWaveObservation::Completion(self)
    }
}
impl RetainWaveObservation for std::result::Result<CompletionRecoveryOutcome, VNextError> {
    fn into_retained(self) -> RetainedWaveObservation {
        RetainedWaveObservation::Recovery(self)
    }
}

/// This route also handles a closure dropped before the bounded queue accepts it
/// or while unwinding. An accepted observation must finish before recovery may
/// attempt to consume that same completion slot.
pub(super) struct DetachedWaveObservation<R: DeviceRuntime> {
    registry: Arc<Mutex<VNextSequenceRegistry<R>>>,
    reaper: Arc<CompletionReaper<R>>,
    observer: WaveObserver<R>,
    submission_observation: Option<PendingSubmissionObservation>,
    armed: bool,
}

impl<R: DeviceRuntime> DetachedWaveObservation<R> {
    pub(super) fn wrap<T: RetainWaveObservation>(self, value: T) -> WorkerOutcomeEnvelope<R, T> {
        WorkerOutcomeEnvelope {
            route: self,
            value: Some(value),
        }
    }

    fn retain(&mut self, observation: Option<RetainedWaveObservation>) {
        if !self.armed {
            return;
        }
        self.armed = false;
        let mut registry = self.registry.lock();
        let sequences = registry.sequences_for_identity(self.observer.identity());
        registry.hold_wave(
            &sequences,
            self.observer.identity().batch_step_id(),
            None,
            None,
            Some(self.observer.clone()),
            observation,
            self.submission_observation.take(),
            false,
            Arc::clone(&self.reaper),
        );
    }
}

impl<R: DeviceRuntime> Drop for DetachedWaveObservation<R> {
    fn drop(&mut self) {
        self.retain(None);
    }
}

/// Protects both send failure and successful send followed by receiver drop.
/// Ordinary receipt delivery moves its existing payload without cloning it.
pub(super) struct WorkerOutcomeEnvelope<R: DeviceRuntime, T: RetainWaveObservation> {
    route: DetachedWaveObservation<R>,
    value: Option<T>,
}

impl<R: DeviceRuntime, T: RetainWaveObservation> WorkerOutcomeEnvelope<R, T> {
    pub(super) fn into_inner(mut self, guard: &mut PreparedWaveGuard<'_, R>) -> T {
        self.route.armed = false;
        guard.submission_observation = self.route.submission_observation.take();
        self.value.take().expect("worker envelope consumed once")
    }
}

impl<R: DeviceRuntime, T: RetainWaveObservation> Drop for WorkerOutcomeEnvelope<R, T> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            self.route.retain(Some(value.into_retained()));
        }
    }
}

#[derive(Debug)]
pub(super) enum WaveProgress {
    BeforeSubmit,
    Terminal(OperationCompletionReceipt),
    Drained(CompletionDrainReceipt),
    Unknown,
    Retired(StepRetirementReceipt),
}

#[derive(Debug)]
pub(super) struct WaveFailure {
    error: FerrumError,
    progress: WaveProgress,
    // A terminal callback can fail alongside the original device/contract
    // failure. This is returned evidence, not another completion owner.
    observation_error: Option<FerrumError>,
    observation_error_in_primary: bool,
}

impl WaveFailure {
    pub(super) fn into_parts(self) -> (FerrumError, WaveProgress, Option<FerrumError>, bool) {
        let observation_reported =
            self.observation_error_in_primary || self.observation_error.is_some();
        (
            self.error,
            self.progress,
            self.observation_error,
            observation_reported,
        )
    }
    pub(super) fn not_submitted(error: FerrumError) -> Self {
        Self {
            error,
            progress: WaveProgress::BeforeSubmit,
            observation_error: None,
            observation_error_in_primary: false,
        }
    }
    pub(super) fn into_error(self) -> FerrumError {
        // This compatibility boundary is reached only with the exact wave
        // owner held by its stack guard or transferred to the registry.
        match &self.progress {
            WaveProgress::BeforeSubmit => tracing::debug!("vNext wave failed before submission"),
            WaveProgress::Terminal(receipt) => {
                tracing::debug!(disposition = ?receipt.disposition(), "vNext wave failed after observed completion")
            }
            WaveProgress::Drained(receipt) => {
                tracing::debug!(identity = ?receipt.batch_identity(), "vNext wave failed after lane drain")
            }
            WaveProgress::Unknown => {
                tracing::debug!("vNext wave failed with unresolved submission")
            }
            WaveProgress::Retired(receipt) => tracing::debug!(
                participants = receipt.participants().len(),
                "vNext product processing failed after retirement"
            ),
        }
        match self.observation_error {
            Some(observation) => FerrumError::backend(format!(
                "{}; additional terminal observation failure: {observation}",
                self.error
            )),
            None => self.error,
        }
    }
}

pub(super) struct RetiredWave<T, R: DeviceRuntime> {
    pub(super) output: T,
    pub(super) retirement: RetiredWaveGuard<R>,
}

pub(super) struct RetiredWaveGuard<R: DeviceRuntime> {
    registry: Arc<Mutex<VNextSequenceRegistry<R>>>,
    receipt: Option<StepRetirementReceipt>,
}

impl<R: DeviceRuntime> RetiredWaveGuard<R> {
    pub(super) fn receipt(&self) -> &StepRetirementReceipt {
        self.receipt.as_ref().expect("live retired wave guard")
    }
    pub(super) fn finish(mut self) -> StepRetirementReceipt {
        self.receipt.take().expect("retired wave consumed once")
    }
    pub(super) fn fail(mut self, error: FerrumError) -> WaveFailure {
        let receipt = self.receipt.take().expect("retired wave consumed once");
        self.registry.lock().cancel_retired_participants(&receipt);
        WaveFailure {
            error,
            progress: WaveProgress::Retired(receipt),
            observation_error: None,
            observation_error_in_primary: false,
        }
    }
}

impl<R: DeviceRuntime> Drop for RetiredWaveGuard<R> {
    fn drop(&mut self) {
        if let Some(receipt) = self.receipt.take() {
            self.registry.lock().cancel_retired_participants(&receipt);
        }
    }
}

pub(super) struct PreparedWaveGuard<'a, R: DeviceRuntime> {
    executor: &'a ProgramExecutor<R>,
    participants: &'a [VNextExecutionParticipant<'a, R>],
    retirement: Option<PreparedRetirement<R>>,
    observer: Option<WaveObserver<R>>,
    observation: Option<RetainedWaveObservation>,
    submission_observation: Option<PendingSubmissionObservation>,
    observation_error: Option<FerrumError>,
    terminal_fact: Option<bool>,
    worker_pending: bool,
}

impl<'a, R: DeviceRuntime> PreparedWaveGuard<'a, R> {
    pub(super) fn new(
        executor: &'a ProgramExecutor<R>,
        retirement: PreparedRetirement<R>,
        participants: &'a [VNextExecutionParticipant<'a, R>],
    ) -> Self {
        Self {
            executor,
            participants,
            retirement: Some(retirement),
            observer: None,
            observation: None,
            submission_observation: None,
            observation_error: None,
            terminal_fact: None,
            worker_pending: false,
        }
    }

    pub(super) fn set_submission_observation(
        &mut self,
        sink: Option<Arc<dyn ProductExecutionEventSink>>,
        capture: ExecutionFactsCapture,
        observation: SubmissionObservation,
        legacy_attribution: bool,
    ) {
        self.submission_observation = sink.map(|sink| {
            PendingSubmissionObservation::new(sink, capture, observation, legacy_attribution)
        });
    }

    pub(super) fn observation_error(&self) -> Option<&FerrumError> {
        self.observation_error.as_ref()
    }
    pub(super) fn attach_observation_failure(
        &mut self,
        mut failure: WaveFailure,
        is_primary: bool,
    ) -> WaveFailure {
        let observation = self.observation_error.take();
        if is_primary {
            // The ordinary successful-device path already uses this error as
            // its primary cause; retain provenance without a second copy.
            failure.observation_error_in_primary = observation.is_some();
        } else {
            failure.observation_error = observation;
        }
        failure
    }

    pub(super) fn observe_submission(&mut self, handle: &CompletionHandle<R>) {
        self.observer = Some(WaveObserver::Completion(handle.clone()));
    }
    pub(super) fn observe_indeterminate(&mut self, handle: &IndeterminateSubmissionHandle<R>) {
        self.observer = Some(WaveObserver::Indeterminate(handle.clone()));
    }
    pub(super) fn worker_route(&mut self) -> DetachedWaveObservation<R> {
        self.worker_pending = true;
        DetachedWaveObservation {
            registry: Arc::clone(&self.executor.sequences),
            reaper: Arc::clone(&self.executor.reaper),
            observer: self
                .observer
                .as_ref()
                .expect("submitted wave owns its observer")
                .clone(),
            submission_observation: self.submission_observation.take(),
            armed: true,
        }
    }
    fn accept(&mut self, observation: RetainedWaveObservation) {
        self.worker_pending = false;
        self.terminal_fact = observation.terminal_fact();
        self.observation = Some(observation);
        // Store the real observation before any fallible external callback.
        if let Some(receipt) = self
            .observation
            .as_ref()
            .and_then(RetainedWaveObservation::terminal_completion)
        {
            if let Some(delivery) = self.submission_observation.take() {
                if let Some(error) = delivery.deliver(receipt) {
                    for participant in self.participants {
                        participant
                            .sequence
                            .settlement
                            .lock()
                            .observation_error
                            .get_or_insert_with(|| error.clone());
                    }
                    self.observation_error.get_or_insert(error);
                }
            }
        }
    }
    pub(super) fn accept_readback_observation(
        &mut self,
        observation: std::result::Result<CompletionReadbackBatchObservation, VNextError>,
    ) {
        self.accept(RetainedWaveObservation::Readbacks(observation));
    }
    pub(super) fn accept_observation(
        &mut self,
        observation: std::result::Result<CompletionObservation, VNextError>,
    ) {
        self.accept(RetainedWaveObservation::Completion(observation));
    }
    pub(super) fn accept_recovery(
        &mut self,
        observation: std::result::Result<CompletionRecoveryOutcome, VNextError>,
    ) {
        self.accept(RetainedWaveObservation::Recovery(observation));
    }
    pub(super) fn readbacks(&self) -> &CompletionReadbackBatchReceipt {
        match self.observation.as_ref() {
            Some(RetainedWaveObservation::Readbacks(Ok(
                CompletionReadbackBatchObservation::Terminal(receipt),
            ))) => receipt,
            _ => panic!("readback access requires the actual terminal receipt"),
        }
    }
    pub(super) fn rollback_unsubmitted(&mut self) -> Result<StepRetirementReceipt> {
        let retirement = self
            .retirement
            .take()
            .expect("guard owns prepared retirement");
        match retirement.try_rollback_unsubmitted() {
            Ok(receipt) => Ok(receipt),
            Err(failure) => {
                let error = FerrumError::backend(failure.error().to_string());
                self.retirement = Some(failure.into_retirement());
                Err(error)
            }
        }
    }
    pub(super) fn replace_retirement(&mut self, retirement: PreparedRetirement<R>) {
        assert!(
            self.retirement.is_none(),
            "replacement requires finalized old step"
        );
        self.retirement = Some(retirement);
        self.observer = None;
        self.observation = None;
        self.submission_observation = None;
        self.observation_error = None;
        self.terminal_fact = None;
        self.worker_pending = false;
    }
    pub(super) fn fail(&mut self, error: FerrumError) -> WaveFailure {
        let progress = self
            .observation
            .as_ref()
            .map(RetainedWaveObservation::failure_progress)
            .unwrap_or(if self.observer.is_none() {
                WaveProgress::BeforeSubmit
            } else {
                WaveProgress::Unknown
            });
        // Every caller returns immediately. The dispatch owner is declared
        // after this guard and must drop before Drop publishes recovery work.
        // Publishing here could race its still-live StepResourceLease owner.
        WaveFailure {
            error,
            progress,
            observation_error: self.observation_error.take(),
            observation_error_in_primary: false,
        }
    }
    fn retain(&mut self) {
        let Some(retirement) = self.retirement.take() else {
            return;
        };
        self.executor.release_resident_decode();
        let step = retirement.batch_step_id();
        // Cold only: no success-path participant allocation is introduced.
        let sequences = self
            .participants
            .iter()
            .map(|p| Arc::clone(p.sequence))
            .collect::<Vec<_>>();
        self.executor.sequences.lock().hold_wave(
            &sequences,
            step,
            Some(retirement),
            self.terminal_fact,
            self.observer.take(),
            self.observation.take(),
            self.submission_observation.take(),
            self.worker_pending,
            Arc::clone(&self.executor.reaper),
        );
    }
    pub(super) fn retire_normal<T>(
        &mut self,
        output: T,
    ) -> std::result::Result<RetiredWave<T, R>, WaveFailure> {
        // The compact fact was derived from the original receipt, not from an
        // error string or a missing reaper slot. Release output packet owners.
        self.observation = self
            .observation
            .take()
            .map(RetainedWaveObservation::release_readback_payload);
        let retirement = self
            .retirement
            .take()
            .expect("guard owns prepared retirement");
        match retirement.try_retire_normal() {
            Ok(retirement) => Ok(RetiredWave {
                output,
                retirement: RetiredWaveGuard {
                    registry: Arc::clone(&self.executor.sequences),
                    receipt: Some(retirement),
                },
            }),
            Err(failure) => {
                let error = FerrumError::backend(format!(
                    "vNext step retirement failed: {}",
                    failure.error()
                ));
                self.retirement = Some(failure.into_retirement());
                Err(self.fail(error))
            }
        }
    }
}

impl<R: DeviceRuntime> Drop for PreparedWaveGuard<'_, R> {
    fn drop(&mut self) {
        self.retain();
    }
}

struct HeldCohort<R: DeviceRuntime> {
    step: BatchStepId,
    state: Mutex<HeldCohortState<R>>,
    reaper: Arc<CompletionReaper<R>>,
}

struct HeldCohortState<R: DeviceRuntime> {
    retirement: Option<PreparedRetirement<R>>,
    observer: Option<WaveObserver<R>>,
    observation: Option<RetainedWaveObservation>,
    submission_observation: Option<PendingSubmissionObservation>,
    observation_error: Option<FerrumError>,
    terminal_fact: Option<bool>,
    worker_pending: bool,
    observer_finished: bool,
    recovering: bool,
    finalized: bool,
}

impl<R: DeviceRuntime> VNextSequenceRegistry<R> {
    fn cancel_retired_participants(&mut self, receipt: &StepRetirementReceipt) {
        for sequence in self.retained_sequences() {
            if receipt.participants().iter().any(|participant| {
                participant.assignment().sequence_authority()
                    == sequence.session.sequence_authority()
            }) {
                sequence.abort();
                self.remove_ended_exact(&sequence);
            }
        }
    }
    pub(super) fn retained_sequences(&self) -> Vec<Arc<VNextSequence<R>>> {
        let mut sequences = self.active.values().cloned().collect::<Vec<_>>();
        for slot in self.prefills.values() {
            if let VNextPrefillSlotState::Ready(sequence)
            | VNextPrefillSlotState::Executing(sequence) = &*slot.state.lock()
            {
                if !sequences
                    .iter()
                    .any(|current| Arc::ptr_eq(current, sequence))
                {
                    sequences.push(Arc::clone(sequence));
                }
            }
        }
        sequences
    }

    fn sequences_for_identity(
        &self,
        identity: &BatchOperationIdentity,
    ) -> Vec<Arc<VNextSequence<R>>> {
        self.retained_sequences()
            .into_iter()
            .filter(|sequence| {
                identity.participants().iter().any(|participant| {
                    let parts = participant.identity().parts();
                    let authority = sequence.session.sequence_authority();
                    parts.active_sequence_slot == Some(authority.sparse_id())
                        && parts.admission_generation == Some(authority.generation())
                        && parts.activation_epoch
                            == Some(sequence.active_binding.activation_epoch())
                        && parts.active_sequence_fingerprint.as_deref()
                            == Some(sequence.active_binding.fingerprint())
                })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn hold_wave(
        &mut self,
        sequences: &[Arc<VNextSequence<R>>],
        step: BatchStepId,
        retirement: Option<PreparedRetirement<R>>,
        terminal_fact: Option<bool>,
        observer: Option<WaveObserver<R>>,
        observation: Option<RetainedWaveObservation>,
        submission_observation: Option<PendingSubmissionObservation>,
        worker_pending: bool,
        reaper: Arc<CompletionReaper<R>>,
    ) {
        let existing = sequences.iter().find_map(|sequence| {
            sequence
                .settlement
                .lock()
                .held
                .as_ref()
                .filter(|held| held.step == step)
                .cloned()
        });
        let held = existing.unwrap_or_else(|| {
            Arc::new(HeldCohort {
                step,
                reaper,
                state: Mutex::new(HeldCohortState {
                    retirement: None,
                    observer: None,
                    observation: None,
                    submission_observation: None,
                    observation_error: None,
                    terminal_fact: None,
                    worker_pending: false,
                    observer_finished: false,
                    recovering: false,
                    finalized: false,
                }),
            })
        });
        {
            let mut state = held.state.lock();
            if retirement.is_some() {
                assert!(
                    state.retirement.is_none(),
                    "exact retirement cannot be installed twice"
                );
                state.retirement = retirement;
                state.worker_pending = worker_pending && !state.observer_finished;
            } else {
                state.observer_finished = true;
                state.worker_pending = false;
            }
            if observer.is_some() {
                state.observer = observer;
            }
            if let Some(delivery) = submission_observation {
                debug_assert!(state.submission_observation.is_none());
                state.submission_observation = Some(delivery);
            }
            if let Some(observation) = observation {
                state.terminal_fact = observation.terminal_fact().or(state.terminal_fact);
                state.observation = Some(observation);
            }
            state.terminal_fact = terminal_fact.or(state.terminal_fact);
        }
        for sequence in sequences {
            sequence.active.store(false, Ordering::Release);
            let mut settlement = sequence.settlement.lock();
            settlement.closing = true;
            settlement.cancel_requested = true;
            if let Some(previous) = &settlement.held {
                assert!(
                    Arc::ptr_eq(previous, &held),
                    "one sequence cannot own two unsettled waves"
                );
            }
            settlement.held = Some(Arc::clone(&held));
            if let Some(slot) = self.prefills.get(sequence.request_id()) {
                slot.cancelled.store(true, Ordering::Release);
            }
        }
        if let Some(wake) = &self.recovery_wake {
            wake.request();
        }
    }

    pub(super) fn remove_ended_exact(&mut self, sequence: &Arc<VNextSequence<R>>) {
        if sequence.settlement.lock().terminal.is_none() {
            return;
        }
        sequence.publish_token_terminal();
        self.token_index.remove(&sequence.token_identity);
        if self
            .active
            .get(&sequence.cache_id)
            .is_some_and(|current| Arc::ptr_eq(current, sequence))
        {
            self.active.remove(&sequence.cache_id);
        }
        let Some(slot) = self.prefills.get(sequence.request_id()).cloned() else {
            return;
        };
        let matches = matches!(&*slot.state.lock(), VNextPrefillSlotState::Ready(current) | VNextPrefillSlotState::Executing(current) if Arc::ptr_eq(current, sequence));
        if matches {
            *slot.state.lock() = VNextPrefillSlotState::Terminal;
            self.prefills.remove(sequence.request_id());
        }
    }
}

impl<R: DeviceRuntime> VNextSequence<R> {
    fn publish_token_terminal(&self) {
        let mut settlement = self.settlement.lock();
        self.publish_token_terminal_locked(&mut settlement);
    }

    fn publish_token_terminal_locked(&self, settlement: &mut SequenceSettlement<R>) {
        if settlement.terminal.is_none() {
            return;
        }
        // The issuer only names the authentic receipt already retained above.
        if let Ok(ended) = self.token_issuer.ended(&self.token_identity) {
            let error = settlement.observation_error.clone();
            for waiter in settlement.terminal_waiters.drain(..) {
                let _ = waiter.send(crate::token_execution::ExecutionTerminalOutcome {
                    ended: ended.clone(),
                    error: error.clone(),
                });
            }
        }
    }

    pub(super) fn record_token_terminal(&self, receipt: SequenceSessionTerminalReceipt) {
        let mut settlement = self.settlement.lock();
        settlement.terminal = Some(receipt);
        self.publish_token_terminal_locked(&mut settlement);
    }

    pub(super) fn token_close_observation(
        &self,
    ) -> Result<crate::token_execution::ExecutionCloseOutcome> {
        use crate::token_execution::ExecutionCloseOutcome;
        let mut settlement = self.settlement.lock();
        if settlement.terminal.is_some() {
            return self.token_issuer.ended(&self.token_identity).map(|ended| {
                ExecutionCloseOutcome {
                    disposition: crate::token_execution::ExecutionCloseDisposition::Ended(ended),
                    error: settlement.observation_error.clone(),
                }
            });
        }
        // Reserved before admission publication; a cold close never needs a
        // new identity allocation that could exhaust after model execution.
        let key = self.token_recovery_key.clone();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        settlement.terminal_waiters.push(sender);
        self.token_issuer
            .held(key, async move {
                receiver.await.map_err(|_| {
                    FerrumError::backend(
                        "native terminal owner ended without publishing a terminal receipt",
                    )
                })
            })
            .map(ExecutionCloseOutcome::held)
    }
    pub(super) fn token_frontier_is_open(&self) -> bool {
        let settlement = self.settlement.lock();
        self.active.load(Ordering::Acquire)
            && !settlement.closing
            && !settlement.cancel_requested
            && settlement.terminal.is_none()
            && settlement.held.is_none()
    }

    pub(super) fn begin_closing(&self) {
        self.active.store(false, Ordering::Release);
        self.settlement.lock().closing = true;
    }

    pub(super) fn try_end_cancelled(&self) -> Result<()> {
        let mut settlement = self.settlement.lock();
        if settlement.terminal.is_some() {
            return Ok(());
        }
        if settlement.held.is_some() {
            return Err(FerrumError::backend("sequence retains an unsettled wave"));
        }
        // Aborting an actual Step poisons its participant sessions. Such a
        // session refuses a cancellation request but still permits checked
        // terminal abort once its native frames/flights are gone.
        let cancellation = self.session.request_cancel();
        let receipt = self.session.try_abort().map_err(|error| {
            FerrumError::backend(match cancellation {
                Ok(_) => error.to_string(),
                Err(cancel_error) => format!(
                    "sequence cancellation failed: {cancel_error}; terminal abort failed: {error}"
                ),
            })
        })?;
        settlement.terminal = Some(receipt);
        self.publish_token_terminal_locked(&mut settlement);
        Ok(())
    }
}

/// Returns checked-out owners even if a backend observation unwinds. A panic
/// while a consuming finalizer owns retirement cannot reconstruct that owner;
/// the cohort remains Held and the worker reports incomplete recovery.
struct RecoveryCheckout<'a, R: DeviceRuntime> {
    held: &'a HeldCohort<R>,
    retirement: Option<PreparedRetirement<R>>,
    observation: Option<RetainedWaveObservation>,
    submission_observation: Option<PendingSubmissionObservation>,
    terminal_fact: Option<bool>,
    finalized: bool,
}

impl<R: DeviceRuntime> Drop for RecoveryCheckout<'_, R> {
    fn drop(&mut self) {
        let mut state = self.held.state.lock();
        state.retirement = self.retirement.take();
        state.observation = self.observation.take();
        state.submission_observation = self.submission_observation.take();
        state.terminal_fact = self.terminal_fact;
        state.finalized = self.finalized;
        state.recovering = false;
    }
}

fn recover_cohort<R: DeviceRuntime>(held: &Arc<HeldCohort<R>>) -> bool {
    let (mut checkout, observer) = {
        let mut state = held.state.lock();
        if state.finalized {
            return true;
        }
        if state.worker_pending || state.recovering || state.retirement.is_none() {
            return false;
        }
        state.recovering = true;
        (
            RecoveryCheckout {
                held,
                retirement: state.retirement.take(),
                observation: state.observation.take(),
                submission_observation: state.submission_observation.take(),
                terminal_fact: state.terminal_fact,
                finalized: false,
            },
            state.observer.clone(),
        )
    };
    checkout.observation = checkout
        .observation
        .take()
        .map(RetainedWaveObservation::release_readback_payload);
    checkout.terminal_fact = checkout.terminal_fact.or_else(|| {
        checkout
            .observation
            .as_ref()
            .and_then(RetainedWaveObservation::terminal_fact)
    });
    if checkout.terminal_fact.is_none() {
        if let Some(observer) = &observer {
            checkout.observation = Some(match observer {
                WaveObserver::Completion(handle) => {
                    let result = handle.wait();
                    let terminal_result = matches!(&result, Ok(CompletionObservation::Terminal(_)));
                    if terminal_result {
                        RetainedWaveObservation::Completion(result)
                    } else {
                        // An absent/reaped slot is not terminal evidence. The
                        // failed recovery is retained, never reclassified.
                        RetainedWaveObservation::Recovery(
                            held.reaper.recover_slot_by_draining_lane(handle.slot_id()),
                        )
                    }
                }
                WaveObserver::Indeterminate(handle) => {
                    RetainedWaveObservation::Recovery(handle.recover_by_draining_lane())
                }
            });
            checkout.terminal_fact = checkout
                .observation
                .as_ref()
                .and_then(RetainedWaveObservation::terminal_fact);
        }
    }
    if let Some(receipt) = checkout
        .observation
        .as_ref()
        .and_then(RetainedWaveObservation::terminal_completion)
    {
        if let Some(delivery) = checkout.submission_observation.take() {
            if let Some(error) = delivery.deliver(receipt) {
                held.state.lock().observation_error.get_or_insert(error);
            }
        }
    }
    if observer.is_some() && checkout.terminal_fact.is_none() {
        return false;
    }
    let retirement = checkout
        .retirement
        .take()
        .expect("recovery checks out the original retirement");
    // A prepared-but-never-enqueued wave has a retired invocation tombstone
    // after its dispatch owner drops. It is not a pristine rollback. Abort
    // validates the actual native state; explicit capacity rollback remains
    // the separate rollback_unsubmitted path before preparing a wave.
    let finalized = retirement.try_abort();
    match finalized {
        Ok(_) => {
            checkout.finalized = true;
            true
        }
        Err(failure) => {
            tracing::debug!(step = ?held.step, error = %failure.error(), "vNext wave retirement remains held");
            checkout.retirement = Some(failure.into_retirement());
            false
        }
    }
}

pub(super) fn install_recovery<R: DeviceRuntime>(
    worker: &VNextCompletionWorker,
    registry: &Arc<Mutex<VNextSequenceRegistry<R>>>,
    reaper: &Arc<CompletionReaper<R>>,
) -> std::io::Result<()> {
    let registry = Arc::downgrade(registry);
    let reaper = Arc::clone(reaper);
    worker.install_recovery(move || {
        let Some(registry) = registry.upgrade() else {
            return;
        };
        let sequences = registry.lock().retained_sequences();
        let mut cohorts: Vec<Arc<HeldCohort<R>>> = Vec::new();
        for sequence in &sequences {
            if let Some(held) = sequence.settlement.lock().held.clone() {
                if !cohorts.iter().any(|previous| Arc::ptr_eq(previous, &held)) {
                    cohorts.push(held);
                }
            }
        }
        for held in cohorts {
            if recover_cohort(&held) {
                let observation_error = held.state.lock().observation_error.clone();
                for sequence in &sequences {
                    let mut settlement = sequence.settlement.lock();
                    if settlement
                        .held
                        .as_ref()
                        .is_some_and(|current| Arc::ptr_eq(current, &held))
                    {
                        if let Some(error) = &observation_error {
                            settlement.observation_error.get_or_insert_with(|| error.clone());
                        }
                        settlement.held = None;
                    }
                }
            }
        }
        // Native checkpoint ownership remains in the existing reaper.
        let _ = reaper.recover_abandoned_checkpoints(MAX_COMPLETION_SWEEP_SLOTS);
        for sequence in sequences {
            if !sequence.settlement.lock().cancel_requested {
                continue;
            }
            match sequence.try_end_cancelled() {
                Ok(()) => registry.lock().remove_ended_exact(&sequence),
                Err(error) => {
                    tracing::debug!(cache_id = %sequence.cache_id, %error, "vNext sequence remains held");
                }
            }
        }
    })
}
