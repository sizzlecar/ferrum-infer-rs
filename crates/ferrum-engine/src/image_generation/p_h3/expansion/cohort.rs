//! One original static installation with independently owned token requests.
//! Every admitted participant contributes to one original native token wave;
//! independent admissions, histories and terminal proofs share its installation.
use super::*;
use ferrum_execution::{
    executor_lifecycle::{
        ExecutorOwner, ExecutorScope, ExecutorShutdownObserver, ExecutorShutdownPhase,
        ExecutorShutdownReport,
    },
    token_execution::{
        ExecutionBatchOutcome, ExecutionCloseDisposition, ExecutionEnded, ExecutionFrontier,
        ExecutionTerminalOutcome, ExecutorSequence,
    },
};
use ferrum_runtime::HostMemoryScope;
use ferrum_types::{FinishReason, RequestId};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
    time::Instant,
};

mod preparation;
pub(crate) use preparation::PreparedExpansionCohort;
#[cfg(test)]
mod tests;

type RetirementWait = Pin<Box<dyn Future<Output = ExecutorShutdownReport> + Send>>;
type CloseWait = Pin<Box<dyn Future<Output = Result<ExecutionTerminalOutcome>> + Send>>;

pub(crate) enum ExpansionParticipantResult {
    Expanded(String),
    Failed(FerrumError),
    Cancelled,
}

pub(crate) enum ExpansionCohortProgress {
    Pending { made_progress: bool },
    Retired(Vec<ExpansionParticipantResult>),
}

pub(super) enum ParticipantState {
    Admission(Option<execute::ExpansionAdmission>),
    Tokens(execute::ExpansionCursor),
    Closing {
        sequence: ExecutorSequence,
        wait: CloseWait,
    },
    Terminal,
}

pub(super) struct ExpansionParticipant {
    prepared: PreparedTextExpansion,
    request: RequestId,
    pub(super) state: Option<ParticipantState>,
    result: Option<ExpansionParticipantResult>,
    ended: Option<ExecutionEnded>,
    failure: Option<execute::ExpansionExecutionFailure>,
    completed: Option<execute::ExpansionExecution>,
    pub(super) observation: Option<super::diagnostics::PhaseObservation>,
}

pub(super) enum CohortState {
    Running,
    Retiring {
        wait: RetirementWait,
        failure: Option<FerrumError>,
    },
    Stopped,
}

pub(crate) struct NativeExpansionCohort {
    assets: Arc<NativeTextExpansion>,
    pub(super) owner: ExecutorOwner,
    scope: Option<ExecutorScope>,
    host: HostMemoryAccount,
    pub(super) participants: Vec<ExpansionParticipant>,
    pub(super) state: Option<CohortState>,
    next: usize,
    retained_waves: Vec<RetainedExpansionWave>,
    next_wave: u64,
}

struct RetainedExpansionWave {
    requests: Vec<RequestId>,
    input_frontiers: Vec<ExecutionFrontier>,
    outcome: ExecutionBatchOutcome,
}

impl NativeExpansionCohort {
    pub(crate) fn initialize<R: DeviceRuntime>(
        assets: Arc<NativeTextExpansion>,
        composition: &RuntimeComposition<R>,
        prepared: PreparedExpansionCohort,
        phase: HostMemoryScope,
    ) -> ExecutorBuildResult<Self> {
        let diagnostics = (0..prepared.len()).map(|_| None).collect();
        Self::initialize_prepared(assets, composition, prepared, phase, diagnostics)
    }

    pub(crate) fn initialize_with_diagnostics<R: DeviceRuntime>(
        assets: Arc<NativeTextExpansion>,
        composition: &RuntimeComposition<R>,
        prepared: PreparedExpansionCohort,
        phase: HostMemoryScope,
        diagnostics: Vec<Option<super::DiagnosticContext>>,
    ) -> ExecutorBuildResult<Self> {
        Self::initialize_prepared(assets, composition, prepared, phase, diagnostics)
    }

    pub(super) fn initialize_single<R: DeviceRuntime>(
        assets: Arc<NativeTextExpansion>,
        composition: &RuntimeComposition<R>,
        prepared: PreparedTextExpansion,
        phase: HostMemoryScope,
        diagnostics: Option<super::DiagnosticContext>,
    ) -> ExecutorBuildResult<Self> {
        Self::initialize_prepared(
            assets,
            composition,
            PreparedExpansionCohort::single(prepared),
            phase,
            vec![diagnostics],
        )
    }

    fn initialize_prepared<R: DeviceRuntime>(
        assets: Arc<NativeTextExpansion>,
        composition: &RuntimeComposition<R>,
        prepared: PreparedExpansionCohort,
        phase: HostMemoryScope,
        diagnostics: Vec<Option<super::DiagnosticContext>>,
    ) -> ExecutorBuildResult<Self> {
        if diagnostics.len() != prepared.len() {
            return Err(FerrumError::request_validation(
                "cohort diagnostic extent differs from its requests",
            )
            .into());
        }
        let timing = prepared.requests[0].device_timing;
        if prepared
            .requests
            .iter()
            .any(|request| request.device_timing != timing)
        {
            return Err(FerrumError::request_validation(
                "cohort device timing differs between requests",
            )
            .into());
        }
        let started = Instant::now();
        let host = phase.account();
        for request in &prepared.requests {
            assets.validate_bound_inputs(request)?;
        }
        let owner = assets.initialize_owned_with_engine(
            composition,
            &prepared.requests[0],
            &prepared.engine,
            phase,
        )?;
        let scope = owner.scope().map_err(|error| {
            ferrum_execution::ExecutorBuildError::with_cleanup(error, owner.begin_shutdown())
        })?;
        // No request has entered admission yet. Off leaves the executor's
        // original sink and timing mode untouched, including all event owners.
        if timing == ferrum_runtime::DeviceTimingMode::Kernel {
            scope.attach_execution_event_sink(Arc::new(super::diagnostics::KernelMetricsOnlySink));
        }
        Ok(Self::from_owner(
            assets,
            prepared.requests,
            owner,
            scope,
            host,
            diagnostics
                .into_iter()
                .map(|context| context.map(|context| (context, started)))
                .collect(),
        ))
    }

    fn from_owner(
        assets: Arc<NativeTextExpansion>,
        requests: Vec<PreparedTextExpansion>,
        owner: ExecutorOwner,
        scope: ExecutorScope,
        host: HostMemoryAccount,
        diagnostics: Vec<Option<(super::DiagnosticContext, Instant)>>,
    ) -> Self {
        let mut diagnostics = diagnostics.into_iter();
        let shared_owner = super::diagnostics::SharedOwner {
            id: uuid::Uuid::new_v4(),
            participants: requests.len(),
        };
        let participants = requests
            .into_iter()
            .map(|prepared| {
                let observation = diagnostics.next().flatten().map(|(context, started)| {
                    let observation = super::diagnostics::PhaseObservation::new(
                        context,
                        prepared.tokens.len(),
                        prepared.generation.eos_token_id,
                        started,
                    )
                    .with_vision_binding(prepared.vision_binding);
                    if shared_owner.participants > 1 {
                        observation.with_shared_owner(shared_owner)
                    } else {
                        observation
                    }
                });
                ExpansionParticipant {
                    prepared,
                    request: RequestId::new(),
                    state: Some(ParticipantState::Admission(None)),
                    result: None,
                    ended: None,
                    failure: None,
                    completed: None,
                    observation,
                }
            })
            .collect();
        Self {
            assets,
            owner,
            scope: Some(scope),
            host,
            participants,
            state: Some(CohortState::Running),
            next: 0,
            retained_waves: Vec::new(),
            next_wave: 0,
        }
    }

    pub(crate) fn cleanup(&self) -> ExecutorShutdownObserver {
        self.owner.shutdown_observer()
    }

    fn retire(&mut self, failure: Option<FerrumError>) {
        // Several results from the same original wave can fail. Keep the first
        // shutdown observer/reason and its still-live scope diagnostics; later
        // rows still retain their own histories and close observations below.
        if !matches!(self.state, Some(CohortState::Running)) {
            return;
        }
        for participant in &mut self.participants {
            if let Some(observation) = &mut participant.observation {
                observation.retiring(self.scope.as_ref());
            }
        }
        drop(self.scope.take());
        let observation = self.owner.begin_shutdown();
        self.state = Some(CohortState::Retiring {
            wait: Box::pin(async move { observation.wait().await }),
            failure,
        });
    }

    fn failed(&mut self, index: usize, mut failure: execute::ExpansionExecutionFailure) {
        let participant = &mut self.participants[index];
        if let (Some(observation), Some(history)) = (&mut participant.observation, &failure.history)
        {
            observation.tokens(history.ids(), failure.finish);
        }
        participant.result = Some(if failure.finish == FinishReason::Cancelled {
            ExpansionParticipantResult::Cancelled
        } else {
            ExpansionParticipantResult::Failed(failure.error.clone())
        });
        // A broken batch contract cannot certify the safety of siblings using
        // the same installation, even if one sequence subsequently closed.
        let violation = matches!(
            failure.outcome,
            Some(ExecutionBatchOutcome::ContractViolation(_))
        );
        let mut fatal = violation.then(|| failure.error.clone());
        if !violation {
            if let Some(frontier) = &failure.frontier {
                let sequence = frontier.sequence().clone();
                match failure.close.take() {
                    Some(close) => {
                        if let Some(error) = close.error {
                            failure.close_error.get_or_insert(error);
                        }
                        match close.disposition {
                            ExecutionCloseDisposition::Ended(ended)
                                if ended.sequence() == &sequence =>
                            {
                                participant.ended = Some(ended);
                                participant.state = Some(ParticipantState::Terminal);
                            }
                            ExecutionCloseDisposition::Held(held)
                                if held.key().sequence() == &sequence =>
                            {
                                participant.state = Some(ParticipantState::Closing {
                                    sequence,
                                    wait: Box::pin(async move { held.wait().await }),
                                });
                            }
                            disposition => {
                                // Preserve foreign/invalid original observations
                                // with the failed owner instead of dropping them.
                                failure.close = Some(
                                    ferrum_execution::token_execution::ExecutionCloseOutcome {
                                        disposition,
                                        error: failure.close_error.clone(),
                                    },
                                );
                                fatal = Some(FerrumError::backend(
                                    "expansion close belongs to another sequence",
                                ));
                            }
                        }
                    }
                    None => fatal = Some(failure.error.clone()),
                }
            } else if failure.admission_probe == execute::AdmissionProbe::NeverEntered {
                // Failure before any admission probe (including cancellation)
                // acquired no native sequence or retained prefill authority.
                participant.state = Some(ParticipantState::Terminal);
            } else if failure.admission.is_some()
                && self
                    .scope
                    .as_ref()
                    .is_some_and(|scope| scope.cancel_prefill_admission(&participant.request))
            {
                participant.state = Some(ParticipantState::Terminal);
            } else {
                // Absence from a registry is not terminal proof. The original
                // executor's shutdown takes custody of ambiguous admission.
                fatal = Some(failure.error.clone());
            }
        }
        participant.failure = Some(failure);
        if let Some(error) = fatal {
            self.retire(Some(error));
        }
    }

    fn apply_step(
        &mut self,
        index: usize,
        step: std::result::Result<execute::ExpansionStep, execute::ExpansionExecutionFailure>,
    ) {
        let participant = &mut self.participants[index];
        match step {
            Ok(execute::ExpansionStep::Continue(cursor)) => {
                participant.state = Some(ParticipantState::Tokens(cursor))
            }
            Ok(execute::ExpansionStep::Complete(done)) => {
                if let Some(observation) = &mut participant.observation {
                    observation.tokens(done.history.ids(), done.finish);
                }
                participant.ended = Some(done.ended.clone());
                participant.result = Some(
                    match self.assets.extract(
                        &participant.prepared,
                        done.history.ids(),
                        done.finish,
                    ) {
                        Ok(text) => ExpansionParticipantResult::Expanded(text),
                        Err(error) => ExpansionParticipantResult::Failed(error),
                    },
                );
                participant.completed = Some(done);
                participant.state = Some(ParticipantState::Terminal);
            }
            Err(failure) => self.failed(index, failure),
        }
    }

    fn advance_tokens(&mut self) {
        let Some(next_wave) = self.next_wave.checked_add(1) else {
            self.retire(Some(FerrumError::internal(
                "expansion wave identity overflow",
            )));
            return;
        };
        let wave_identity = self.next_wave;
        self.next_wave = next_wave;
        let indices: Vec<_> = self
            .participants
            .iter()
            .enumerate()
            .filter_map(|(index, participant)| {
                matches!(participant.state, Some(ParticipantState::Tokens(_))).then_some(index)
            })
            .collect();
        let cursors = indices
            .iter()
            .map(|&index| match self.participants[index].state.take() {
                Some(ParticipantState::Tokens(cursor)) => cursor,
                _ => unreachable!("selected token participant changed state"),
            })
            .collect();
        let execute::ExpansionWave {
            steps,
            input_frontiers,
            retained,
            fatal,
            native_outcome,
        } = block_on_wave(execute::advance_batch(
            self.scope.as_ref().expect("live cohort scope"),
            cursors,
        ));
        for (input_index, (&index, step)) in indices.iter().zip(&steps).enumerate() {
            if let Some(observation) = &self.participants[index].observation {
                let (outcome, after) = match step {
                    Ok(execute::ExpansionStep::Continue(cursor)) => {
                        ("completed", Some(cursor.frontier_tokens()))
                    }
                    Ok(execute::ExpansionStep::Complete(done)) => {
                        ("completed", Some(done.frontier.consumed_tokens()))
                    }
                    Err(failure) => (
                        "failed",
                        failure
                            .frontier
                            .as_ref()
                            .map(ExecutionFrontier::consumed_tokens),
                    ),
                };
                observation.wave(
                    wave_identity,
                    indices.len(),
                    input_index,
                    input_frontiers[input_index].consumed_tokens(),
                    after,
                    outcome,
                    native_outcome,
                );
            }
        }
        // Keep the unique original ledger before processing close results: one
        // failure can already seal the owner. Input indices remain mapped to
        // the exact original requests/frontiers even after peers finish.
        let retained_index = retained.map(|outcome| {
            let index = self.retained_waves.len();
            self.retained_waves.push(RetainedExpansionWave {
                requests: indices
                    .iter()
                    .map(|&i| self.participants[i].request.clone())
                    .collect(),
                input_frontiers,
                outcome,
            });
            index
        });
        for (input_index, (index, mut step)) in indices.into_iter().zip(steps).enumerate() {
            if let (Some(wave), Err(failure)) = (retained_index, &mut step) {
                failure.wave_observation =
                    Some(execute::ExpansionWaveReference { wave, input_index });
            }
            self.apply_step(index, step);
        }
        if let Some(error) = fatal {
            if matches!(self.state, Some(CohortState::Running)) {
                self.retire(Some(error));
            }
        }
    }

    /// Advance one native wave or poll one original close. A Held participant
    /// stops shared submission until its exact terminal observer settles.
    pub(crate) fn advance(
        &mut self,
        cancelled: &[bool],
        wake: &Waker,
    ) -> Result<ExpansionCohortProgress> {
        if cancelled.len() != self.participants.len() {
            return Err(FerrumError::request_validation(
                "cohort cancellation extent differs",
            ));
        }
        let state = self
            .state
            .take()
            .ok_or_else(|| FerrumError::internal("cohort state already consumed"))?;
        match state {
            CohortState::Retiring { mut wait, failure } => {
                let report = match wait.as_mut().poll(&mut Context::from_waker(wake)) {
                    Poll::Pending => {
                        self.state = Some(CohortState::Retiring { wait, failure });
                        return Ok(ExpansionCohortProgress::Pending {
                            made_progress: false,
                        });
                    }
                    Poll::Ready(report) => report,
                };
                self.state = Some(CohortState::Stopped);
                let retired = report.phase() == ExecutorShutdownPhase::Retired;
                let retirement = report.result();
                for (participant, &cancelled) in self.participants.iter_mut().zip(cancelled) {
                    if let Some(observation) = participant.observation.take() {
                        let (outcome, text) = if !retired {
                            ("retained", None)
                        } else if retirement.is_err() || failure.is_some() {
                            ("failed", None)
                        } else if cancelled {
                            ("cancelled", None)
                        } else {
                            match participant.result.as_ref() {
                                Some(ExpansionParticipantResult::Expanded(text)) => {
                                    ("expanded", Some(text.as_str()))
                                }
                                Some(ExpansionParticipantResult::Cancelled) => ("cancelled", None),
                                _ => ("failed", None),
                            }
                        };
                        observation.record(retired, outcome, text);
                    }
                }
                if !retired {
                    return Err(report.error().cloned().unwrap_or_else(|| {
                        FerrumError::backend("prompt cohort cleanup retains original resources")
                    }));
                }
                // Histories and retained failure observations may outlive an
                // individual sequence, but never the completed phase grant.
                for participant in &mut self.participants {
                    participant.failure.take();
                    participant.completed.take();
                    participant.state = Some(ParticipantState::Terminal);
                }
                self.retained_waves.clear();
                retirement?;
                if let Some(error) = failure {
                    return Err(error);
                }
                let results = self
                    .participants
                    .iter_mut()
                    .zip(cancelled)
                    .map(|(p, &cancelled)| {
                        let result = p.result.take().ok_or_else(|| {
                            FerrumError::internal("retired participant has no result")
                        })?;
                        Ok(if cancelled {
                            ExpansionParticipantResult::Cancelled
                        } else {
                            result
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                return Ok(ExpansionCohortProgress::Retired(results));
            }
            CohortState::Stopped => {
                self.state = Some(CohortState::Stopped);
                return Err(FerrumError::request_validation("prompt cohort is terminal"));
            }
            CohortState::Running => self.state = Some(CohortState::Running),
        }

        let pending =
            |p: &ExpansionParticipant| !matches!(p.state, Some(ParticipantState::Terminal));
        let index = self
            .participants
            .iter()
            .position(|p| matches!(p.state, Some(ParticipantState::Closing { .. })))
            .or_else(|| {
                self.participants
                    .iter()
                    .zip(cancelled)
                    .position(|(p, &cancelled)| cancelled && pending(p))
            })
            .or_else(|| {
                // Freeze all admissions before forwarding the finite cohort,
                // so an early Ready request cannot run ahead as singleton waves.
                self.participants.iter().position(|participant| {
                    matches!(participant.state, Some(ParticipantState::Admission(_)))
                })
            })
            .or_else(|| {
                (0..self.participants.len())
                    .map(|offset| (self.next + offset) % self.participants.len())
                    .find(|&i| pending(&self.participants[i]))
            });
        let Some(index) = index else {
            self.retire(None);
            return Ok(ExpansionCohortProgress::Pending {
                made_progress: true,
            });
        };
        self.next = (index + 1) % self.participants.len();
        if self.participants.len() > 1
            && !cancelled[index]
            && matches!(
                self.participants[index].state,
                Some(ParticipantState::Tokens(_))
            )
        {
            self.advance_tokens();
            if matches!(self.state, Some(CohortState::Running))
                && self
                    .participants
                    .iter()
                    .all(|p| matches!(p.state, Some(ParticipantState::Terminal)))
            {
                self.retire(None);
            }
            return Ok(ExpansionCohortProgress::Pending {
                made_progress: true,
            });
        }
        let participant = &mut self.participants[index];
        let state = participant
            .state
            .take()
            .ok_or_else(|| FerrumError::internal("participant state already consumed"))?;
        match state {
            ParticipantState::Admission(admission) => {
                let scope = self.scope.as_ref().expect("live cohort scope");
                let admission = match admission {
                    Some(admission) => Ok(admission),
                    None => execute::ExpansionAdmission::new(
                        scope,
                        &participant.request,
                        &participant.prepared.tokens,
                        participant.prepared.generation,
                        participant.prepared.prefill_chunk_tokens,
                        &self.host,
                    )
                    .and_then(|a| {
                        a.with_inputs(participant.prepared.causal_inputs.as_ref().map(Arc::clone))
                    }),
                };
                let result = admission.and_then(|a| {
                    block_on_wave(
                        a.advance(scope, &participant.prepared.tokens, &|| cancelled[index]),
                    )
                });
                match result {
                    Ok(execute::ExpansionAdmissionStep::Ready(cursor)) => {
                        participant.state = Some(ParticipantState::Tokens(cursor))
                    }
                    Ok(execute::ExpansionAdmissionStep::Pending(admission)) => {
                        participant.state = Some(ParticipantState::Admission(Some(admission)))
                    }
                    Err(failure) => self.failed(index, failure),
                }
            }
            ParticipantState::Tokens(cursor) => {
                match block_on_wave(
                    cursor.advance(self.scope.as_ref().expect("live cohort scope"), &|| {
                        cancelled[index]
                    }),
                ) {
                    Ok(execute::ExpansionStep::Continue(cursor)) => {
                        participant.state = Some(ParticipantState::Tokens(cursor))
                    }
                    Ok(execute::ExpansionStep::Complete(done)) => {
                        if let Some(observation) = &mut participant.observation {
                            observation.tokens(done.history.ids(), done.finish);
                        }
                        participant.ended = Some(done.ended.clone());
                        participant.result = Some(
                            match self.assets.extract(
                                &participant.prepared,
                                done.history.ids(),
                                done.finish,
                            ) {
                                Ok(text) => ExpansionParticipantResult::Expanded(text),
                                Err(error) => ExpansionParticipantResult::Failed(error),
                            },
                        );
                        participant.completed = Some(done);
                        participant.state = Some(ParticipantState::Terminal);
                    }
                    Err(failure) => self.failed(index, failure),
                }
            }
            ParticipantState::Closing { sequence, mut wait } => {
                match wait.as_mut().poll(&mut Context::from_waker(wake)) {
                    Poll::Pending => {
                        participant.state = Some(ParticipantState::Closing { sequence, wait });
                        return Ok(ExpansionCohortProgress::Pending {
                            made_progress: false,
                        });
                    }
                    Poll::Ready(Ok(terminal)) if terminal.ended.sequence() == &sequence => {
                        if let Some(error) = terminal.error {
                            participant.result = Some(ExpansionParticipantResult::Failed(error));
                        }
                        participant.ended = Some(terminal.ended);
                        participant.state = Some(ParticipantState::Terminal);
                    }
                    Poll::Ready(result) => {
                        let error = result.err().unwrap_or_else(|| {
                            FerrumError::backend("held expansion close returned another sequence")
                        });
                        self.retire(Some(error));
                    }
                }
            }
            ParticipantState::Terminal => unreachable!("terminal participant is not selected"),
        }
        if matches!(self.state, Some(CohortState::Running))
            && self
                .participants
                .iter()
                .all(|p| matches!(p.state, Some(ParticipantState::Terminal)))
        {
            self.retire(None);
        }
        Ok(ExpansionCohortProgress::Pending {
            made_progress: true,
        })
    }
}

fn block_on_wave<F: Future>(future: F) -> F::Output {
    futures::executor::block_on(tokio::task::unconstrained(future))
}

impl Drop for NativeExpansionCohort {
    fn drop(&mut self) {
        self.owner.shutdown_observer().retain_until_retired((
            self.state.take(),
            std::mem::take(&mut self.participants),
            std::mem::take(&mut self.retained_waves),
        ));
        drop(self.scope.take());
        self.owner.begin_shutdown();
    }
}
