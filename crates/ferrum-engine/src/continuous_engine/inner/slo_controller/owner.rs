//! A published wave survives cancellation of its caller. Before dispatch a
//! dropped token can be withdrawn; after dispatch the same task must be joined.
use super::*;
use ferrum_interfaces::execution_cost::{
    ExpectedExecutionWave, ExpectedWorkSelection, HostSubmissionRejection,
    NonblockingHostSubmissionGuard, WaveCommitment,
};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

mod structured_prepared;

const READY: u8 = 0;
const DISPATCH_ENTERED: u8 = 1;
const FINISHED: u8 = 2;
const WITHDRAWN: u8 = 3;

pub(super) struct ControllerWork {
    pub prospective_capture: Option<Arc<super::super::cost_observation::ProspectiveCapture>>,
    pub batch: ferrum_interfaces::BatchPlan,
    pub timing: ControllerTimingCommitment,
    pub proof: ControllerSafetyProof,
    pub expected: ExpectedExecutionWave,
}

pub(super) enum ControllerTimingCommitment {
    Witness {
        valid_until: Instant,
        model_version: u64,
        predicted_wall_ns: u64,
        admission: Option<super::admission::PendingTimeWitness>,
    },
    CompleteRequests,
}

impl ControllerWork {
    pub fn rows(&self) -> impl Iterator<Item = &ExpectedWorkSelection> {
        self.expected
            .work()
            .participants()
            .iter()
            .map(|row| row.selection())
    }

    pub fn commit_fences(&self) -> Vec<ControllerCommitFence<'_>> {
        match self.expected.commitment() {
            WaveCommitment::CostWitness(witness) => witness
                .participants()
                .iter()
                .map(ControllerCommitFence::Cost)
                .collect(),
            WaveCommitment::CompleteRequests(_) => {
                self.rows().map(ControllerCommitFence::Completion).collect()
            }
        }
    }

    fn check_time(&self, engine: &EngineInner) -> std::result::Result<(), HostSubmissionRejection> {
        use HostSubmissionRejection::*;
        match (&self.timing, self.expected.commitment()) {
            (ControllerTimingCommitment::CompleteRequests, WaveCommitment::CompleteRequests(_)) => {
                Ok(())
            }
            (
                ControllerTimingCommitment::Witness {
                    valid_until,
                    model_version,
                    ..
                },
                WaveCommitment::CostWitness(_),
            ) => {
                if slo_clock_now() > *valid_until {
                    return Err(WitnessExpired);
                }
                let runtime = engine.cost_runtime.as_ref().ok_or(CostModelChanged)?;
                let model = runtime
                    .try_snapshot()
                    .ok_or(Busy)?
                    .ok_or(CostModelChanged)?;
                if model.model_version() != *model_version {
                    return Err(CostModelChanged);
                }
                Ok(())
            }
            _ => Err(CostModelChanged),
        }
    }
}

pub(super) struct ControllerFlight {
    pub work: ControllerWork,
    pub receipt: Mutex<Option<PlanningPublicationReceipt>>,
    phase: AtomicU8,
    abandoned: AtomicBool,
    pub(super) calibration: Option<Arc<super::super::calibration::CalibrationWaveReceipt>>,
    // Await this handle by mutable reference. Dropping the waiter leaves the
    // handle in this durable slot, and never aborts the backend future.
    task: tokio::sync::Mutex<Option<tokio::task::JoinHandle<Result<EngineIterationOutcome>>>>,
    #[cfg(test)]
    withdraw_gate: Mutex<Option<(Arc<Notify>, Arc<Notify>)>>,
    #[cfg(test)]
    withdraw_lost: Notify,
    #[cfg(test)]
    cleanup_pending: AtomicBool,
}

#[cfg(test)]
impl ControllerFlight {
    pub(super) fn pause_withdrawal_for_test(&self, observed: Arc<Notify>, resume: Arc<Notify>) {
        *self.withdraw_gate.lock() = Some((observed, resume));
    }
    pub(super) async fn withdrawal_lost_for_test(&self) {
        self.withdraw_lost.notified().await;
    }
    pub(super) fn cleanup_pending_for_test(&self) -> bool {
        self.cleanup_pending.load(Ordering::Acquire)
    }
}

pub(in crate::continuous_engine) struct PreparedControllerWave {
    flight: Arc<ControllerFlight>,
    wake: Arc<Notify>,
    armed: bool,
}

impl PreparedControllerWave {
    pub(in crate::continuous_engine::inner) fn calibration_receipt(
        &self,
    ) -> Option<Arc<super::super::calibration::CalibrationWaveReceipt>> {
        self.flight.calibration.clone()
    }
}

impl Drop for PreparedControllerWave {
    fn drop(&mut self) {
        if self.armed {
            self.flight.abandoned.store(true, Ordering::Release);
            self.wake.notify_one();
        }
    }
}

pub(super) struct HostGuard<'a> {
    pub engine: &'a EngineInner,
    pub work: &'a ControllerWork,
}

impl NonblockingHostSubmissionGuard for HostGuard<'_> {
    fn check(&self) -> std::result::Result<(), HostSubmissionRejection> {
        let _stage = self.work.proof.budget.stage(ControllerStage::HostGuard);
        use HostSubmissionRejection::*;
        if self.engine.shutdown_started.load(Ordering::Acquire) {
            return Err(Cancelled);
        }
        self.work.check_time(self.engine)?;
        let sequences = self.engine.sequences.try_read().ok_or(Busy)?;
        let witnessed = matches!(self.work.timing, ControllerTimingCommitment::Witness { .. });
        if witnessed && sequences.len() != self.work.proof.fences.len() {
            return Err(FrontierChanged);
        }
        for fence in &self.work.proof.fences {
            let selected = self
                .work
                .rows()
                .find(|row| row.request_id == fence.key.request_id);
            if !witnessed && selected.is_none() {
                continue;
            }
            let sequence = sequences
                .get(&fence.key.request_id)
                .ok_or(FrontierChanged)?;
            if !fence.matches_sequence(sequence) {
                return Err(FrontierChanged);
            }
            let output = sequence.credited_output.as_ref().ok_or(OutputRevoked)?;
            if output.failure.is_some() || output.port.consumer_closed() {
                return Err(OutputRevoked);
            }
            let produces_token = selected.is_some_and(|row| match row.input {
                ferrum_interfaces::execution_cost::ExpectedWaveInput::Decode { .. } => true,
                ferrum_interfaces::execution_cost::ExpectedWaveInput::Prefill { chunk } => {
                    chunk.is_final()
                }
            });
            if produces_token {
                if output.grant.is_none() || output.tokens_before_grant != fence.generated {
                    return Err(OutputRevoked);
                }
            } else if output.grant.is_some() || output.port.planning_snapshot() != fence.output {
                return Err(OutputRevoked);
            }
        }
        // The bounded reads above are part of validation, not free elapsed time.
        if let Some(scope) = &self.work.proof.protection {
            if scope.rows().len() != self.work.proof.fences.len()
                || scope.required_first_service().is_some_and(|key| {
                    !self.work.rows().any(|row| {
                        row.request_id == key.request_id
                            && row.owner_incarnation.get() == key.incarnation
                    })
                })
            {
                return Err(FrontierChanged);
            }
        }
        // Debt is part of the immutable decision input. No other successful
        // service may silently change the chosen recovery obligation.
        for peer in &self.work.proof.recovery_peers {
            let sequence = sequences.get(&peer.id).ok_or(FrontierChanged)?;
            if !Arc::ptr_eq(&peer.owner, &sequence.stream_projection_identity)
                || sequence
                    .time_admission
                    .as_ref()
                    .is_none_or(|state| state.recovery_service != peer.debt)
            {
                return Err(FrontierChanged);
            }
        }
        self.work.check_time(self.engine)?;
        Ok(())
    }
}

impl EngineInner {
    pub(super) fn install_controller_wave(
        &self,
        work: ControllerWork,
        receipt: PlanningPublicationReceipt,
    ) -> Result<PreparedControllerWave> {
        let calibration = self.manual_calibration_driver.then(|| {
            Arc::new(super::super::calibration::CalibrationWaveReceipt::new(
                work.expected.work().clone(),
            ))
        });
        let flight = Arc::new(ControllerFlight {
            work,
            receipt: Mutex::new(Some(receipt)),
            phase: AtomicU8::new(READY),
            abandoned: AtomicBool::new(false),
            calibration,
            task: tokio::sync::Mutex::new(None),
            #[cfg(test)]
            withdraw_gate: Mutex::new(None),
            #[cfg(test)]
            withdraw_lost: Notify::new(),
            #[cfg(test)]
            cleanup_pending: AtomicBool::new(false),
        });
        let mut state = self.slo_controller.lock();
        if state.pending_execution.is_some() {
            return Err(FerrumError::internal("SLO execution slot already occupied"));
        }
        state.pending_execution = Some(Arc::clone(&flight));
        Ok(PreparedControllerWave {
            flight,
            wake: Arc::clone(&self.work_notify),
            armed: true,
        })
    }

    fn clear_controller_flight(&self, flight: &Arc<ControllerFlight>) {
        let mut state = self.slo_controller.lock();
        if state
            .pending_execution
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, flight))
        {
            state.pending_execution = None;
        }
    }

    /// Identity-fenced return of the original, definitely uncommitted grants.
    /// A cancelled/reused request's replacement grant must never be touched.
    pub(super) fn return_controller_output(&self, work: &ControllerWork) {
        let mut sequences = self.sequences.write();
        for participant in work.rows() {
            let Some(sequence) = sequences.get_mut(&participant.request_id) else {
                continue;
            };
            if !sequence.cost_frontier.is_some_and(|frontier| {
                frontier.owner_incarnation == participant.owner_incarnation
                    && frontier.work_generation == participant.work_generation
            }) {
                continue;
            }
            let Some(output) = sequence.credited_output.as_mut() else {
                continue;
            };
            if output.tokens_before_grant == sequence.generated_tokens.len() {
                if let Some(grant) = output.grant.take() {
                    grant.return_unsubmitted();
                }
            }
        }
    }

    pub(super) fn withdraw_controller_flight(&self, flight: &ControllerFlight) -> Result<()> {
        let _stage = flight
            .work
            .proof
            .budget
            .stage(ControllerStage::Reconciliation);
        self.return_controller_output(&flight.work);
        if let Some(receipt) = flight.receipt.lock().take() {
            self.keep_unsubmitted_publication(receipt)?;
        }
        Ok(())
    }

    pub(in crate::continuous_engine) async fn execute_slo_controller_wave(
        self: &Arc<Self>,
        mut prepared: PreparedControllerWave,
    ) -> Result<EngineIterationOutcome> {
        let flight = Arc::clone(&prepared.flight);
        // The phase transition and task publication share the join slot lock.
        // A competing shutdown cannot observe DispatchEntered with no handle.
        let Ok(mut task_slot) = flight.task.try_lock() else {
            return Ok(EngineIterationOutcome::Idle);
        };
        // No await between phase transition and publication of the JoinHandle.
        match flight.phase.compare_exchange(
            READY,
            DISPATCH_ENTERED,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(WITHDRAWN | FINISHED) => {
                prepared.armed = false;
                return Ok(EngineIterationOutcome::Idle);
            }
            Err(_) => return Err(FerrumError::internal("SLO wave dispatched twice")),
        }
        let engine = Arc::clone(self);
        let task_flight = Arc::clone(&flight);
        *task_slot = Some(tokio::spawn(async move {
            task_flight.work.proof.budget.record_ready_queue();
            let wait = task_flight
                .work
                .proof
                .budget
                .stage(ControllerStage::IterationLock);
            // Serializes host mutation with every normal iteration. The final
            // backend guard still checks time/output after all preparation.
            let _iteration = engine.iteration_lock.lock().await;
            drop(wait);
            let result = engine.dispatch_controller_wave(&task_flight).await;
            engine.finish_controller_audit(
                &task_flight.work.proof.budget,
                match &result {
                    Ok(EngineIterationOutcome::Progressed) => "submitted",
                    Ok(_) => "withdrawn",
                    Err(_) => "failed",
                },
            );
            task_flight.phase.store(FINISHED, Ordering::Release);
            engine.work_notify.notify_one();
            result
        }));
        prepared.armed = false;
        drop(task_slot);
        self.join_controller_flight(&flight).await
    }

    async fn join_controller_flight(
        self: &Arc<Self>,
        flight: &Arc<ControllerFlight>,
    ) -> Result<EngineIterationOutcome> {
        let mut slot = flight.task.lock().await;
        let result = loop {
            match slot.as_mut() {
                Some(task) => match task.await {
                    Ok(result) => break result,
                    Err(error) => {
                        let message =
                            format!("guarded controller task failed after dispatch entry: {error}");
                        let engine = Arc::clone(self);
                        let cleanup_flight = Arc::clone(flight);
                        // Replace the failed handle before another await. Even
                        // cancellation of this joiner while cleanup waits for
                        // iteration_lock leaves a durable cleanup task to join.
                        *slot = Some(tokio::spawn(async move {
                            let wait = cleanup_flight
                                .work
                                .proof
                                .budget
                                .stage(ControllerStage::IterationLock);
                            let _iteration = engine.iteration_lock.lock().await;
                            drop(wait);
                            let stage = cleanup_flight
                                .work
                                .proof
                                .budget
                                .stage(ControllerStage::Reconciliation);
                            let cleanup = engine
                                .fail_controller_participants(&cleanup_flight.work, &message)
                                .await;
                            // Indeterminate submission is never rolled back.
                            cleanup_flight.receipt.lock().take();
                            cleanup_flight.phase.store(FINISHED, Ordering::Release);
                            drop(stage);
                            engine.finish_controller_audit(
                                &cleanup_flight.work.proof.budget,
                                "failed",
                            );
                            cleanup?;
                            Err(FerrumError::internal(message))
                        }));
                        #[cfg(test)]
                        {
                            flight.cleanup_pending.store(true, Ordering::Release);
                            self.work_notify.notify_one();
                        }
                    }
                },
                None => return Ok(EngineIterationOutcome::Idle),
            }
        };
        slot.take();
        drop(slot);
        self.clear_controller_flight(flight);
        result
    }

    /// Called before acquiring iteration_lock, and on shutdown before resource
    /// destruction. Running tasks are joined, never aborted or blindly retried.
    pub(in crate::continuous_engine) async fn drain_slo_execution(
        self: &Arc<Self>,
    ) -> Result<Option<EngineIterationOutcome>> {
        let flight = self.slo_controller.lock().pending_execution.clone();
        let Some(flight) = flight else {
            return Ok(None);
        };
        if flight.phase.load(Ordering::Acquire) == READY {
            if !flight.abandoned.load(Ordering::Acquire)
                && !self.shutdown_started.load(Ordering::Acquire)
            {
                return Ok(Some(EngineIterationOutcome::Idle));
            }
            #[cfg(test)]
            {
                let gate = flight.withdraw_gate.lock().take();
                if let Some((observed, resume)) = gate {
                    observed.notify_one();
                    resume.notified().await;
                }
            }
            let slot = flight.task.lock().await;
            // Shutdown/abandonment requests withdrawal, but only winning this
            // CAS proves dispatch did not enter. A loser must join that wave.
            if flight
                .phase
                .compare_exchange(READY, WITHDRAWN, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                flight.work.proof.budget.record_ready_queue();
                let result = self.withdraw_controller_flight(&flight);
                flight.phase.store(FINISHED, Ordering::Release);
                self.clear_controller_flight(&flight);
                self.finish_controller_audit(
                    &flight.work.proof.budget,
                    if result.is_ok() {
                        "withdrawn"
                    } else {
                        "failed"
                    },
                );
                result?;
                return Ok(Some(EngineIterationOutcome::Idle));
            }
            #[cfg(test)]
            flight.withdraw_lost.notify_one();
            drop(slot);
        }
        self.join_controller_flight(&flight).await.map(Some)
    }
}
