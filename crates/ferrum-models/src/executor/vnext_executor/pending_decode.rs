//! One bounded worker task owns both submissions. The host must settle the
//! parent's Step before the worker observes or retires the dependent child.
use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use tokio::sync::{oneshot, Notify};

pub(super) struct SubmittedPairWave<R: DeviceRuntime> {
    completion: CompletionHandle<R>,
    readbacks: CompletionReadbackBatchRequest,
    step: Arc<StepResourceLease<R>>,
}

impl<R: DeviceRuntime> SubmittedPairWave<R> {
    /// Transfer the model's sole Step owner; core retains its own owner until
    /// the terminal observation. Callers must not keep a scheduler Step clone.
    pub(super) fn new(
        completion: CompletionHandle<R>,
        readbacks: CompletionReadbackBatchRequest,
        step: Arc<StepResourceLease<R>>,
    ) -> Self {
        Self {
            completion,
            readbacks,
            step,
        }
    }
}

/// No `into_step`: acknowledgement cannot precede retirement/failed-closed Drop.
pub(super) struct ParentRetirementGuard<R: DeviceRuntime> {
    step: Option<Arc<StepResourceLease<R>>>,
    acknowledged: Option<SyncSender<bool>>,
}

impl<R: DeviceRuntime> ParentRetirementGuard<R> {
    pub(super) fn step(&self) -> &StepResourceLease<R> {
        self.step.as_deref().expect("live parent retirement guard")
    }

    pub(super) fn retire_normal(mut self) -> Result<StepRetirementReceipt> {
        let result = retire_step(self.step.take().expect("live parent Step"), true);
        self.acknowledge(result.is_ok());
        result
    }

    pub(super) fn abort(mut self) -> Result<StepRetirementReceipt> {
        let result = retire_step(self.step.take().expect("live parent Step"), false);
        self.acknowledge(false);
        result
    }

    fn acknowledge(&mut self, committed: bool) {
        if let Some(sender) = self.acknowledged.take() {
            // Capacity one, exactly one message: Drop never blocks on the
            // worker being between readback and recv.
            let _ = sender.send(committed);
        }
    }
}

impl<R: DeviceRuntime> Drop for ParentRetirementGuard<R> {
    fn drop(&mut self) {
        if let Some(step) = self.step.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| retire_step(step, false)));
        }
        self.acknowledge(false);
    }
}

type ParentResult<R> = Result<(CompletionReadbackBatchReceipt, ParentRetirementGuard<R>)>;

pub(super) struct PendingParentReadback<R: DeviceRuntime> {
    receiver: oneshot::Receiver<ParentResult<R>>,
}

impl<R: DeviceRuntime> PendingParentReadback<R> {
    pub(super) async fn wait(self) -> ParentResult<R> {
        self.receiver
            .await
            .map_err(|_| FerrumError::backend("paired parent observation channel closed"))?
    }
}

/// The parent fence and readback have been observed exactly once, while its
/// Step remains unretired so a prepared successor can still depend on it.
pub(super) struct ObservedPairParent<R: DeviceRuntime> {
    wave: OwnedWave<R>,
    receipt: CompletionReadbackBatchReceipt,
}

impl<R: DeviceRuntime> ObservedPairParent<R> {
    pub(super) fn receipt(&self) -> &CompletionReadbackBatchReceipt {
        &self.receipt
    }

    pub(super) fn abort(mut self) -> Result<StepRetirementReceipt> {
        retire_step(self.wave.step.take().expect("observed parent Step"), false)
    }
}

pub(super) fn observe_parent<R: DeviceRuntime>(
    reservation: VNextCompletionReservation<'_>,
    parent: SubmittedPairWave<R>,
    reaper: Arc<CompletionReaper<R>>,
) -> VNextCompletionTicket<Result<ObservedPairParent<R>>> {
    reservation.submit(VNextCompletionTaskKind::WaveReadback, move || {
        let mut wave = OwnedWave::new(parent, reaper);
        let receipt = wave.observe()?;
        Ok(ObservedPairParent { wave, receipt })
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RowDisposition {
    Available,
    Consumed,
    Discarded,
}

struct CohortState {
    result: Option<Result<Arc<CompletionReadbackBatchReceipt>>>,
    rows: Vec<RowDisposition>,
}

/// Contains only completion evidence and row-consumption bookkeeping. In
/// particular, this cannot form a cycle with VNextSequence or take its locks.
pub(super) struct PendingDecodeCohort {
    state: Mutex<CohortState>,
    completed: Notify,
}

#[derive(Clone)]
pub(super) struct BufferedDecodeRow {
    receipt: Arc<CompletionReadbackBatchReceipt>,
    row: usize,
}

impl BufferedDecodeRow {
    pub(super) fn receipt(&self) -> &CompletionReadbackBatchReceipt {
        &self.receipt
    }
    pub(super) fn disposition(&self) -> &CompletionReadbackDisposition {
        &self.receipt.dispositions()[self.row]
    }
}

impl PendingDecodeCohort {
    fn new(rows: usize) -> Self {
        Self {
            state: Mutex::new(CohortState {
                result: None,
                rows: vec![RowDisposition::Available; rows],
            }),
            completed: Notify::new(),
        }
    }

    fn publish(&self, result: Result<CompletionReadbackBatchReceipt>) {
        let mut state = self.state.lock();
        if state.result.is_none() {
            state.result = Some(result.and_then(|receipt| {
                if receipt.dispositions().len() != state.rows.len() {
                    return Err(FerrumError::backend(
                        "paired child readback cardinality changed",
                    ));
                }
                if receipt
                    .dispositions()
                    .iter()
                    .enumerate()
                    .any(|(row, disposition)| {
                        let CompletionReadbackDisposition::Succeeded(output) = disposition else {
                            return true;
                        };
                        output.request().participant_index() as usize != row
                    })
                {
                    return Err(FerrumError::backend(
                        "paired child readback differs from canonical cohort rows",
                    ));
                }
                Ok(Arc::new(receipt))
            }));
            drop(state);
            self.completed.notify_waiters();
        }
    }

    pub(super) fn row_count(&self) -> usize {
        self.state.lock().rows.len()
    }

    pub(super) fn is_ready(&self) -> bool {
        self.state.lock().result.is_some()
    }

    pub(super) fn rows_remaining(&self) -> usize {
        self.state
            .lock()
            .rows
            .iter()
            .filter(|row| **row == RowDisposition::Available)
            .count()
    }

    pub(super) fn peek_row(&self, row: usize) -> Result<BufferedDecodeRow> {
        let state = self.state.lock();
        let receipt = state
            .result
            .as_ref()
            .ok_or_else(|| FerrumError::backend("pending decode is not observed"))?
            .clone()?;
        if state.rows.get(row) != Some(&RowDisposition::Available) {
            return Err(FerrumError::backend("pending decode row is unavailable"));
        }
        Ok(BufferedDecodeRow { receipt, row })
    }

    /// The caller must first retire/abort its parent guard. Awaiting this while
    /// retaining that guard would wait for the caller's own acknowledgement.
    pub(super) async fn wait(&self) -> Result<Arc<CompletionReadbackBatchReceipt>> {
        loop {
            let notified = self.completed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if let Some(result) = &self.state.lock().result {
                return result.clone();
            }
            notified.await;
        }
    }

    /// Take only after the whole caller batch is ready, so an unrelated fresh
    /// row's capacity deferral cannot consume this already executed output.
    pub(super) fn take_row(&self, row: usize) -> Result<BufferedDecodeRow> {
        let mut state = self.state.lock();
        let receipt = state
            .result
            .as_ref()
            .ok_or_else(|| FerrumError::backend("pending decode is not observed"))?
            .clone()?;
        let slot = state
            .rows
            .get_mut(row)
            .ok_or_else(|| FerrumError::backend("pending decode row is out of range"))?;
        if *slot != RowDisposition::Available {
            return Err(FerrumError::backend(
                "pending decode row was already consumed or discarded",
            ));
        }
        *slot = RowDisposition::Consumed;
        Ok(BufferedDecodeRow { receipt, row })
    }

    pub(super) fn discard_row(&self, row: usize) -> Result<()> {
        let mut state = self.state.lock();
        let slot = state
            .rows
            .get_mut(row)
            .ok_or_else(|| FerrumError::backend("pending decode row is out of range"))?;
        if *slot == RowDisposition::Consumed {
            return Err(FerrumError::backend(
                "consumed pending decode row cannot be discarded",
            ));
        }
        *slot = RowDisposition::Discarded;
        Ok(())
    }
}

/// One reservation, one accepted worker job, two exact submitted waves. All
/// channels are bounded; abandoning either receiver does not cancel this job.
pub(super) fn submit_pair<R: DeviceRuntime>(
    reservation: VNextCompletionReservation<'_>,
    parent: SubmittedPairWave<R>,
    child: SubmittedPairWave<R>,
    reaper: Arc<CompletionReaper<R>>,
) -> (PendingParentReadback<R>, Arc<PendingDecodeCohort>) {
    submit_pair_with_observer(reservation, parent, child, reaper, |_| {})
}

pub(super) fn submit_pair_with_observer<R: DeviceRuntime>(
    reservation: VNextCompletionReservation<'_>,
    parent: SubmittedPairWave<R>,
    child: SubmittedPairWave<R>,
    reaper: Arc<CompletionReaper<R>>,
    observer: impl FnOnce(&Result<CompletionReadbackBatchReceipt>) + Send + 'static,
) -> (PendingParentReadback<R>, Arc<PendingDecodeCohort>) {
    submit_owned_pair(
        reservation,
        OwnedWave::new(parent, Arc::clone(&reaper)),
        None,
        OwnedWave::new(child, reaper),
        observer,
    )
}

pub(super) fn submit_pair_with_observed_parent<R: DeviceRuntime>(
    reservation: VNextCompletionReservation<'_>,
    parent: ObservedPairParent<R>,
    child: SubmittedPairWave<R>,
    reaper: Arc<CompletionReaper<R>>,
    observer: impl FnOnce(&Result<CompletionReadbackBatchReceipt>) + Send + 'static,
) -> (PendingParentReadback<R>, Arc<PendingDecodeCohort>) {
    submit_owned_pair(
        reservation,
        parent.wave,
        Some(parent.receipt),
        OwnedWave::new(child, reaper),
        observer,
    )
}

fn submit_owned_pair<R: DeviceRuntime>(
    reservation: VNextCompletionReservation<'_>,
    parent: OwnedWave<R>,
    parent_receipt: Option<CompletionReadbackBatchReceipt>,
    child: OwnedWave<R>,
    observer: impl FnOnce(&Result<CompletionReadbackBatchReceipt>) + Send + 'static,
) -> (PendingParentReadback<R>, Arc<PendingDecodeCohort>) {
    let cohort = Arc::new(PendingDecodeCohort::new(child.readbacks.len()));
    let retained = Arc::clone(&cohort);
    let (parent_sender, receiver) = oneshot::channel();
    let ticket = reservation.submit(VNextCompletionTaskKind::WaveReadback, move || {
        let mut task = PairTask {
            parent: Some(parent),
            parent_receipt,
            child: Some(child),
            parent_sender: Some(parent_sender),
        };
        let result = catch_unwind(AssertUnwindSafe(|| task.run()));
        let result = match result {
            Ok(result) => result,
            Err(_) => Err(FerrumError::backend("paired completion task panicked")),
        };
        // Drop performs exact recovery while the worker still owns both the
        // reaper and Steps. Parent goes first, then the dependent child.
        task.cleanup();
        if let Some(sender) = task.parent_sender.take() {
            let message =
                result.as_ref().err().cloned().unwrap_or_else(|| {
                    FerrumError::backend("paired parent result was not delivered")
                });
            let _ = sender.send(Err(message));
        }
        // Metrics cannot strand a completed cohort if an observer panics.
        let _ = catch_unwind(AssertUnwindSafe(|| observer(&result)));
        retained.publish(result);
    });
    // Accepted tasks outlive their tickets. Results use the two bounded
    // channels above, so this unit-result receiver is deliberately unused.
    drop(ticket);
    (PendingParentReadback { receiver }, cohort)
}

struct PairTask<R: DeviceRuntime> {
    parent: Option<OwnedWave<R>>,
    parent_receipt: Option<CompletionReadbackBatchReceipt>,
    child: Option<OwnedWave<R>>,
    parent_sender: Option<oneshot::Sender<ParentResult<R>>>,
}

impl<R: DeviceRuntime> PairTask<R> {
    fn run(&mut self) -> Result<CompletionReadbackBatchReceipt> {
        let parent = self.parent.as_mut().expect("pair parent");
        let receipt = match self.parent_receipt.take() {
            Some(receipt) => receipt,
            None => parent.observe()?,
        };
        let (acknowledged, receiver): (SyncSender<bool>, Receiver<bool>) = sync_channel(1);
        let guard = ParentRetirementGuard {
            step: parent.step.take(),
            acknowledged: Some(acknowledged),
        };
        let sender = self.parent_sender.take().expect("one parent result");
        // If the parent caller abandoned its future, SendError owns the guard.
        // Dropping it aborts the Step and sends the acknowledgement first.
        if let Err(abandoned) = sender.send(Ok((receipt, guard))) {
            drop(abandoned);
        }
        let parent_committed = receiver.recv().unwrap_or(false);
        drop(self.parent.take());
        let child = self.child.as_mut().expect("pair child");
        let receipt = child.observe()?;
        let successful = parent_committed
            && matches!(
                receipt.completion().disposition(),
                OperationCompletionDisposition::Succeeded
            )
            && receipt
                .dispositions()
                .iter()
                .all(|row| matches!(row, CompletionReadbackDisposition::Succeeded(_)));
        let step = child
            .step
            .take()
            .expect("child Step retained until observation");
        let retirement = retire_step(step, successful);
        retirement?;
        if !successful {
            return Err(FerrumError::backend(
                "paired child did not produce a committed successful readback",
            ));
        }
        Ok(receipt)
    }

    fn cleanup(&mut self) {
        drop(self.parent.take());
        drop(self.child.take());
    }
}

impl<R: DeviceRuntime> Drop for PairTask<R> {
    fn drop(&mut self) {
        self.cleanup();
    }
}

struct OwnedWave<R: DeviceRuntime> {
    completion: CompletionHandle<R>,
    readbacks: CompletionReadbackBatchRequest,
    step: Option<Arc<StepResourceLease<R>>>,
    reaper: Arc<CompletionReaper<R>>,
    terminal: bool,
}

impl<R: DeviceRuntime> OwnedWave<R> {
    fn new(wave: SubmittedPairWave<R>, reaper: Arc<CompletionReaper<R>>) -> Self {
        Self {
            completion: wave.completion,
            readbacks: wave.readbacks,
            step: Some(wave.step),
            reaper,
            terminal: false,
        }
    }

    fn observe(&mut self) -> Result<CompletionReadbackBatchReceipt> {
        match self.completion.wait_with_readbacks(self.readbacks.clone()) {
            Ok(CompletionReadbackBatchObservation::Terminal(receipt)) => {
                self.terminal = true;
                Ok(receipt)
            }
            other => {
                let context = format!("paired wave readback was not terminal: {other:?}");
                let recovered = self.recover();
                Err(FerrumError::backend(match recovered {
                    Ok(()) => context,
                    Err(error) => format!("{context}; {error}"),
                }))
            }
        }
    }

    fn recover(&mut self) -> Result<()> {
        // A blocking indeterminate observation already authorized recovery.
        // Drain that exact slot immediately; do not replace its failure by a
        // later successful fence query before failing the dependent pair.
        if let Ok(outcome) = self
            .reaper
            .recover_slot_by_draining_lane(self.completion.slot_id())
        {
            return self.record_recovery(outcome);
        }
        // A contract error may have occurred before fence observation. The
        // recovery API requires a blocking observation before a lane drain.
        match self
            .reaper
            .wait_slot_for_recovery(self.completion.slot_id())
        {
            Ok(CompletionObservation::Terminal(_)) => {
                self.terminal = true;
                return Ok(());
            }
            _ => {}
        }
        match self
            .reaper
            .recover_slot_by_draining_lane(self.completion.slot_id())
        {
            Ok(outcome) => self.record_recovery(outcome),
            Err(error) => Err(FerrumError::backend(format!(
                "paired wave recovery failed: {error}"
            ))),
        }
    }

    fn record_recovery(&mut self, outcome: CompletionRecoveryOutcome) -> Result<()> {
        match outcome {
            CompletionRecoveryOutcome::Drained(_) => {
                self.terminal = true;
                Ok(())
            }
            CompletionRecoveryOutcome::Quarantined(_) => Err(FerrumError::backend(
                "paired wave remains quarantined; core retains its device ownership",
            )),
        }
    }
}

impl<R: DeviceRuntime> Drop for OwnedWave<R> {
    fn drop(&mut self) {
        if self.step.is_none() {
            return;
        }
        if !self.terminal {
            let _ = catch_unwind(AssertUnwindSafe(|| self.recover()));
        }
        if let Some(step) = self.step.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| retire_step(step, false)));
        }
    }
}

fn retire_step<R: DeviceRuntime>(
    step: Arc<StepResourceLease<R>>,
    commit: bool,
) -> Result<StepRetirementReceipt> {
    let result = if commit {
        step.try_retire_normal()
    } else {
        step.try_abort()
    };
    match result {
        Ok(receipt) => Ok(receipt),
        Err(failure) => {
            let message = format!("paired Step retirement failed: {}", failure.error());
            let step = failure.into_step();
            if commit {
                if let Err(failure) = step.try_abort() {
                    drop(failure.into_step());
                }
            } else {
                drop(step);
            }
            Err(FerrumError::backend(message))
        }
    }
}

#[cfg(test)]
#[path = "pending_decode/tests.rs"]
mod tests;
#[cfg(test)]
#[path = "../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[cfg(test)]
#[path = "../../../../ferrum-interfaces/tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;
