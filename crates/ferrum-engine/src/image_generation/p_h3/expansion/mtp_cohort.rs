//! A sealed cohort executes each main or draft wave as one actual Core batch.
//! The ordinary image scheduler still owns joining/refill and phase capacity.
//! This machine owns the two causal programs and their independent sequences.
use super::cohort::{ExpansionCohortProgress, ExpansionParticipantResult, PreparedExpansionCohort};
use super::*;
use ferrum_execution::driver::SharedProgramWeights;
use ferrum_execution::{CausalProgramSession, CausalRequestInputs, CausalWaveFailure};
use ferrum_runtime::{CompletionReaper, RequestIdentity, RunId, SelectedPrefix, SequenceSession};
use ferrum_types::FinishReason;
use std::{collections::BTreeSet, task::Waker, time::Instant};

mod cleanup;
mod core;
mod cursor;
mod diagnostic;
pub(crate) mod initialization;
#[cfg(test)]
mod tests;
use cursor::{Cursor, WaveKind};

struct SettlingWave<R: DeviceRuntime> {
    kind: WaveKind,
    indices: Vec<usize>,
    computed: ferrum_execution::CausalComputedWave<R>,
    observation: Option<diagnostic::WaveObservation>,
}

struct ActiveWave<R: DeviceRuntime> {
    kind: WaveKind,
    participants: Vec<usize>,
    native: core::Wave<R>,
    observation: Option<diagnostic::WaveObservation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lifecycle {
    Running,
    Retiring,
    Aborting,
    Closed,
}

/// A caller must retain this owner after an execution error until try_close
/// proves all original frames, sequences, programs and shared weights closed.
pub(crate) struct MtpExpansionCohort<R: DeviceRuntime> {
    assets: Arc<NativeTextExpansion>,
    programs: cleanup::ProgramOwners<R>,
    host: HostMemoryAccount,
    run: RunId,
    maximum_tokens: usize,
    pending: Vec<Option<PreparedTextExpansion>>,
    inputs: Vec<Arc<CausalRequestInputs>>,
    main_admissions: Vec<Option<core::Admission<R>>>,
    draft_admissions: Vec<Option<core::Admission<R>>>,
    cursors: Vec<Option<Cursor<R>>>,
    results: Vec<Option<ExpansionParticipantResult>>,
    active: Option<ActiveWave<R>>,
    settling: Option<SettlingWave<R>>,
    failures: Vec<CausalWaveFailure<R>>,
    stranded: Vec<Arc<SequenceSession<R>>>,
    reaper: Arc<CompletionReaper<R>>,
    last_kind: usize,
    lifecycle: Lifecycle,
    first_error: Option<FerrumError>,
    cancelled: Vec<bool>,
    diagnostics: Option<diagnostic::Observations>,
    diagnostics_set: bool,
}

impl<R: DeviceRuntime> MtpExpansionCohort<R> {
    /// Assets/geometry/Host demands are validated before installing programs.
    /// All initialized native owners move into this machine even if its first
    /// poll discovers an input mismatch; error paths retain the same owners.
    pub(crate) fn from_programs(
        assets: Arc<NativeTextExpansion>,
        main: CausalProgramSession<R>,
        draft: CausalProgramSession<R>,
        weights: SharedProgramWeights<R>,
        prepared: PreparedExpansionCohort,
        inputs: Vec<Arc<CausalRequestInputs>>,
    ) -> Self {
        let maximum_tokens = prepared.engine.runtime.max_model_len.unwrap_or(0);
        let count = prepared.requests.len();
        Self {
            assets,
            host: main.host_account().clone(),
            programs: cleanup::ProgramOwners::new(main, draft, weights),
            run: RunId::new(format!("run.mtp.{}", uuid::Uuid::new_v4()))
                .expect("valid run identity"),
            maximum_tokens,
            pending: prepared.requests.into_iter().map(Some).collect(),
            inputs,
            main_admissions: (0..count).map(|_| None).collect(),
            draft_admissions: (0..count).map(|_| None).collect(),
            cursors: (0..count).map(|_| None).collect(),
            results: (0..count).map(|_| None).collect(),
            active: None,
            settling: None,
            failures: Vec::new(),
            stranded: Vec::new(),
            reaper: CompletionReaper::new(),
            last_kind: 3,
            lifecycle: Lifecycle::Running,
            first_error: None,
            cancelled: vec![false; count],
            diagnostics: None,
            diagnostics_set: false,
        }
    }

    /// Install original request diagnostics once, after initialization and before
    /// the first admission. Reject late replacement rather than lose histories.
    pub(crate) fn set_diagnostics(
        &mut self,
        contexts: Vec<Option<(DiagnosticContext, Instant)>>,
    ) -> Result<()> {
        if self.diagnostics_set
            || self.lifecycle != Lifecycle::Running
            || self.first_error.is_some()
            || contexts.len() != self.pending.len()
            || self.pending.iter().any(Option::is_none)
            || self.main_admissions.iter().any(Option::is_some)
            || self.draft_admissions.iter().any(Option::is_some)
        {
            return Err(FerrumError::request_validation(
                "MTP diagnostic setup is late or has different membership",
            ));
        }
        for (context, _) in contexts.iter().flatten() {
            context
                .policy
                .validate()
                .map_err(|error| FerrumError::config(error.to_string()))?;
        }
        self.diagnostics_set = true;
        if contexts
            .iter()
            .flatten()
            .any(|(c, _)| c.policy != PH3ImagePromptExpansionDiagnosticPolicy::Disabled)
        {
            self.diagnostics = Some(diagnostic::Observations::new(contexts, &self.pending));
        }
        Ok(())
    }

    pub(crate) fn advance(
        &mut self,
        cancelled: &[bool],
        wake: &Waker,
    ) -> Result<ExpansionCohortProgress> {
        if let Some(error) = &self.first_error {
            return Err(error.clone());
        }
        let result = match self.lifecycle {
            Lifecycle::Running => self.advance_running(cancelled, wake),
            Lifecycle::Retiring => self.poll_retirement(),
            Lifecycle::Aborting | Lifecycle::Closed => {
                return Err(FerrumError::internal(
                    "MTP execution cannot resume after cleanup has started",
                ));
            }
        };
        if let Err(error) = &result {
            // Preserve the first public failure as well as every native recovery
            // owner. A pre-submit or publication error also forbids execution.
            self.first_error = Some(error.clone());
        }
        result
    }

    fn advance_running(
        &mut self,
        cancelled: &[bool],
        wake: &Waker,
    ) -> Result<ExpansionCohortProgress> {
        if cancelled.len() != self.results.len()
            || self.inputs.len() != self.results.len()
            || self.results.is_empty()
            || self.maximum_tokens == 0
        {
            return Err(FerrumError::request_validation(
                "MTP cohort geometry/input extent differs",
            ));
        }
        let mut made_progress = false;
        for (pending, &cancel) in self.cancelled.iter_mut().zip(cancelled) {
            *pending |= cancel;
        }
        if let Some(active) = &mut self.active {
            if active.participants.iter().any(|&i| self.cancelled[i])
                && active.native.discard_before_frame()
            {
                // No Core frame exists, so removing this pending wave cannot
                // poison a survivor. Re-form it from still-live sequences.
                self.active.take();
                made_progress = true;
            }
        }
        for index in 0..self.cancelled.len() {
            let cancel = self.cancelled[index];
            if cancel && self.results[index].is_none() {
                if self
                    .active
                    .as_ref()
                    .is_some_and(|wave| wave.participants.contains(&index))
                {
                    // An admitted frame keeps its original Open membership
                    // through compute. Cancel its actual session on the next
                    // poll, before selected settlement; never abort its peers.
                    continue;
                }
                if let Some(cursor) = &mut self.cursors[index] {
                    cursor.cancel()?;
                } else if self.abort_admissions(index)? {
                    self.pending[index].take();
                    self.results[index] = Some(ExpansionParticipantResult::Cancelled);
                    made_progress = true;
                }
            }
        }
        // Refresh actual session cancellation after compute and before any
        // selected copy or output publication; callers can cancel while the
        // synchronous native forward was running on the preceding poll.
        if self.settling.is_some() {
            return self.settle_wave();
        }
        // An already prepared wave is completed/retired with its original
        // membership. Core discards cancelled rows without changing siblings.
        if self.active.is_some() {
            return self.poll_wave(wake);
        }
        self.finish_participants()?;
        if self.results.iter().all(Option::is_some) {
            self.lifecycle = Lifecycle::Retiring;
            return self.poll_retirement();
        }
        let mut admission_progress = false;
        for index in 0..self.results.len() {
            if self.results[index].is_some() || self.cancelled[index] {
                continue;
            }
            if self.cursors[index].is_none() {
                if self.main_admissions[index].is_none() {
                    let prepared = self.pending[index]
                        .as_ref()
                        .ok_or_else(|| FerrumError::internal("MTP lost request preparation"))?;
                    let history: Arc<[u32]> = prepared
                        .tokens
                        .iter()
                        .map(|token| token.get())
                        .collect::<Vec<_>>()
                        .into();
                    let end = prepared.prefill_chunk_tokens.min(history.len());
                    self.main_admissions[index] = Some(core::Admission::new(
                        self.programs.main()?,
                        self.run.clone(),
                        request_id(index, "main")?,
                        history,
                        0..end,
                        self.inputs[index].maximum_tokens(),
                        Some(Arc::clone(&self.inputs[index])),
                    )?);
                }
                match self.main_admissions[index]
                    .as_mut()
                    .unwrap()
                    .poll(self.programs.main()?, wake)?
                {
                    core::AdmissionPoll::Pending {
                        made_progress: progress,
                    } => admission_progress |= progress,
                    core::AdmissionPoll::Ready(session) => {
                        // Keep a close authority if managed output allocation fails.
                        self.stranded.push(Arc::clone(&session));
                        let cursor = Cursor::new(
                            self.pending[index].take().unwrap(),
                            session,
                            Arc::clone(&self.inputs[index]),
                            &self.host,
                        )?;
                        self.stranded.pop();
                        self.cursors[index] = Some(cursor);
                        self.main_admissions[index].take();
                        admission_progress = true;
                    }
                }
            }
            if let Some(cursor) = &mut self.cursors[index] {
                if cursor.kind() == Some(WaveKind::Draft) && cursor.draft.is_none() {
                    if self.draft_admissions[index].is_none() {
                        self.draft_admissions[index] = Some(core::Admission::new(
                            self.programs.draft()?,
                            self.run.clone(),
                            request_id(index, "draft")?,
                            cursor.draft_initial_history()?,
                            0..1,
                            self.inputs[index].maximum_tokens(),
                            None,
                        )?);
                    }
                    match self.draft_admissions[index]
                        .as_mut()
                        .unwrap()
                        .poll(self.programs.draft()?, wake)?
                    {
                        core::AdmissionPoll::Pending {
                            made_progress: progress,
                        } => admission_progress |= progress,
                        core::AdmissionPoll::Ready(session) => {
                            cursor.draft = Some(session);
                            self.draft_admissions[index].take();
                            admission_progress = true;
                        }
                    }
                }
            }
        }
        if admission_progress {
            // Let maintenance and newly opened admissions finish their current
            // round before selecting a Ready subset. A genuine WaitForRelease
            // reports false and must not bar ready requests from freeing space.
            return Ok(ExpansionCohortProgress::Pending {
                made_progress: true,
            });
        }
        let kinds = [
            WaveKind::Prefill,
            WaveKind::Draft,
            WaveKind::Verify,
            WaveKind::Last,
        ];
        for offset in 1..=kinds.len() {
            let slot = (self.last_kind + offset) % kinds.len();
            let kind = kinds[slot];
            let participants = self
                .cursors
                .iter()
                .enumerate()
                .filter_map(|(index, cursor)| {
                    cursor
                        .as_ref()
                        .filter(|cursor| {
                            cursor.kind() == Some(kind)
                                && (kind != WaveKind::Draft || cursor.draft.is_some())
                        })
                        .map(|_| index)
                })
                .collect::<Vec<_>>();
            if participants.is_empty() {
                continue;
            }
            let members = participants
                .iter()
                .map(|&index| self.cursors[index].as_ref().unwrap().member(&self.host))
                .collect::<Result<Vec<_>>>()?;
            let program = if kind == WaveKind::Draft {
                self.programs.draft()?
            } else {
                self.programs.main()?
            };
            let before = members
                .iter()
                .map(|member| member.range.start)
                .collect::<Vec<_>>();
            let native = core::Wave::new(program, members, self.maximum_tokens, kind)?;
            let observation = self.diagnostics.as_mut().map(|diagnostics| {
                diagnostics.begin(
                    kind,
                    native.input_indices().iter().map(|&i| before[i]).collect(),
                )
            });
            self.active = Some(ActiveWave {
                kind,
                participants,
                native,
                observation,
            });
            self.last_kind = slot;
            return self.poll_wave(wake);
        }
        Ok(ExpansionCohortProgress::Pending { made_progress })
    }

    fn poll_wave(&mut self, wake: &Waker) -> Result<ExpansionCohortProgress> {
        let active = self.active.as_mut().expect("active wave");
        let program = if active.kind == WaveKind::Draft {
            self.programs.draft()?
        } else {
            self.programs.main()?
        };
        let started = Instant::now();
        let polled = active.native.poll(program, Arc::clone(&self.reaper), wake);
        if let Some(observation) = &mut active.observation {
            observation.polled(started.elapsed());
        }
        let computed = match polled {
            Ok(core::WavePoll::Pending { made_progress }) => {
                return Ok(ExpansionCohortProgress::Pending { made_progress })
            }
            Ok(core::WavePoll::Computed(computed)) => computed,
            Err(core::WaveError::Before(error)) => return Err(error),
            Err(core::WaveError::Execution(failure)) => {
                let error = core_error(&failure);
                self.failures.push(failure);
                return Err(error);
            }
        };
        let active = self.active.take().unwrap();
        let indices = active
            .native
            .input_indices()
            .iter()
            .map(|&i| active.participants[i])
            .collect::<Vec<_>>();
        let kind = active.kind;
        let observation = active.observation;
        if let (Some(diagnostics), Some(observation)) = (&mut self.diagnostics, &observation) {
            diagnostics.computed(observation, indices.len());
        }
        self.settling = Some(SettlingWave {
            kind,
            indices,
            computed,
            observation,
        });
        Ok(ExpansionCohortProgress::Pending {
            made_progress: true,
        })
    }

    fn settle_wave(&mut self) -> Result<ExpansionCohortProgress> {
        let SettlingWave {
            kind,
            indices,
            computed,
            observation,
        } = self.settling.take().expect("computed wave");
        let started = Instant::now();
        if kind == WaveKind::Verify {
            let stops = indices
                .iter()
                .map(|&index| {
                    BTreeSet::from([self.cursors[index]
                        .as_ref()
                        .unwrap()
                        .prepared
                        .generation
                        .eos_token_id])
                })
                .collect::<Vec<_>>();
            let remaining = indices
                .iter()
                .map(|&index| self.cursors[index].as_ref().unwrap().remaining())
                .collect::<Vec<_>>();
            let outputs = match computed.commit_one_draft(&stops, &remaining) {
                Ok(outputs) => outputs,
                Err(failure) => {
                    let error = core_error(&failure);
                    self.failures.push(failure);
                    return Err(error);
                }
            };
            let settlement_wall = started.elapsed();
            let verification = outputs
                .iter()
                .map(|output| {
                    output
                        .as_ref()
                        .map(|output| (output.accepted_draft(), output.tokens().len()))
                })
                .collect::<Vec<_>>();
            for (&index, output) in indices.iter().zip(outputs) {
                let cursor = self.cursors[index].as_mut().unwrap();
                if let Some(output) = output {
                    cursor.accept_verified(output)?;
                } else {
                    cursor.finish = Some(FinishReason::Cancelled);
                }
            }
            if let (Some(diagnostics), Some(observation)) = (&mut self.diagnostics, &observation) {
                let facts = indices
                    .iter()
                    .zip(verification)
                    .map(|(&index, verification)| {
                        verification.map(|verification| {
                            (
                                self.cursors[index].as_ref().unwrap().accepted_frontier() as u64,
                                Some(verification),
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                diagnostics.settled(observation, &indices, &facts, settlement_wall);
            }
        } else {
            let outputs = match computed.commit(vec![SelectedPrefix::All; indices.len()]) {
                Ok(outputs) => outputs.into_outputs(),
                Err(failure) => {
                    let error = core_error(&failure);
                    self.failures.push(failure);
                    return Err(error);
                }
            };
            if let (Some(diagnostics), Some(observation)) = (&mut self.diagnostics, &observation) {
                let facts = outputs
                    .iter()
                    .map(|output| output.as_ref().map(|output| (output.frontier(), None)))
                    .collect::<Vec<_>>();
                diagnostics.settled(observation, &indices, &facts, started.elapsed());
            }
            for (index, output) in indices.into_iter().zip(outputs) {
                let cursor = self.cursors[index].as_mut().unwrap();
                if let Some(output) = output {
                    if kind == WaveKind::Draft {
                        cursor.accept_draft(output)?;
                    } else {
                        cursor.accept_main(output)?;
                    }
                } else {
                    cursor.finish = Some(FinishReason::Cancelled);
                }
            }
        }
        Ok(ExpansionCohortProgress::Pending {
            made_progress: true,
        })
    }

    fn finish_participants(&mut self) -> Result<()> {
        for index in 0..self.cursors.len() {
            let Some(cursor) = &self.cursors[index] else {
                continue;
            };
            let Some(finish) = cursor.finish else {
                continue;
            };
            if !self.abort_admissions(index)? {
                continue;
            }
            self.cursors[index]
                .as_mut()
                .unwrap()
                .close_sequences(finish == FinishReason::Cancelled)?;
            let cursor = self.cursors[index].take().unwrap();
            if let Some(diagnostics) = &mut self.diagnostics {
                diagnostics.ended(index, cursor.output.ids(), finish);
            }
            self.results[index] = Some(if finish == FinishReason::Cancelled {
                ExpansionParticipantResult::Cancelled
            } else {
                match self
                    .assets
                    .extract(&cursor.prepared, cursor.output.ids(), finish)
                {
                    Ok(text) => ExpansionParticipantResult::Expanded(text),
                    Err(error) => ExpansionParticipantResult::Failed(error),
                }
            });
        }
        Ok(())
    }

    fn abort_admissions(&mut self, index: usize) -> Result<bool> {
        for admissions in [&mut self.main_admissions, &mut self.draft_admissions] {
            if let Some(admission) = &mut admissions[index] {
                if !admission.try_abort()? {
                    return Ok(false);
                }
            }
            admissions[index].take();
        }
        Ok(true)
    }

    pub(crate) fn try_close(&mut self) -> Result<bool> {
        if self.lifecycle == Lifecycle::Closed {
            return Ok(true);
        }
        // Publish the irreversible mode change before the first fallible or
        // deferred cleanup. Partial abort can never become normal retirement.
        self.lifecycle = Lifecycle::Aborting;
        if let Some(diagnostics) = &mut self.diagnostics {
            diagnostics.retiring();
        }
        let result = self.close_owned();
        match &result {
            Ok(true) => {
                self.lifecycle = Lifecycle::Closed;
                if let Some(diagnostics) = &mut self.diagnostics {
                    diagnostics.record(&self.results, false, self.first_error.is_some());
                }
            }
            Err(error) if self.first_error.is_none() => self.first_error = Some(error.clone()),
            _ => {}
        }
        result
    }

    fn poll_retirement(&mut self) -> Result<ExpansionCohortProgress> {
        if let Some(diagnostics) = &mut self.diagnostics {
            diagnostics.retiring();
        }
        if self.close_owned()? {
            self.lifecycle = Lifecycle::Closed;
            if let Some(diagnostics) = &mut self.diagnostics {
                diagnostics.record(&self.results, true, false);
            }
            Ok(ExpansionCohortProgress::Retired(
                self.results
                    .iter_mut()
                    .map(|result| result.take().expect("normal retirement has all results"))
                    .collect(),
            ))
        } else {
            Ok(ExpansionCohortProgress::Pending {
                made_progress: false,
            })
        }
    }

    fn close_owned(&mut self) -> Result<bool> {
        if let Some(active) = &mut self.active {
            if !active.native.try_abort()? {
                return Ok(false);
            }
        }
        self.active.take();
        if let Some(settling) = self.settling.take() {
            if let Err(failure) = settling.computed.try_abort() {
                self.failures.push(failure);
                return Ok(false);
            }
        }
        while let Some(failure) = self.failures.pop() {
            if let Err(failure) = failure.try_abort() {
                self.failures.push(failure);
                return Ok(false);
            }
        }
        for index in 0..self.results.len() {
            if !self.abort_admissions(index)? {
                return Ok(false);
            }
            if let Some(cursor) = &mut self.cursors[index] {
                if let Some(diagnostics) = &mut self.diagnostics {
                    diagnostics.ended(
                        index,
                        cursor.output.ids(),
                        cursor.finish.unwrap_or(if self.first_error.is_some() {
                            FinishReason::Error
                        } else {
                            FinishReason::Cancelled
                        }),
                    );
                }
                cursor.close_sequences(true)?;
            }
            self.cursors[index].take();
            self.pending[index].take();
        }
        while let Some(session) = self.stranded.last() {
            session
                .try_abort_if_quiescent()
                .or_else(|_| session.try_abort())
                .map_err(core_error)?;
            self.stranded.pop();
        }
        self.inputs.clear();
        if let Some(diagnostics) = &mut self.diagnostics {
            diagnostics.snapshot_replay(&self.programs);
        }
        self.programs.poll_close()
    }
}

fn request_id(index: usize, role: &str) -> Result<RequestIdentity> {
    RequestIdentity::new(format!("request.mtp.{role}.{index}")).map_err(core_error)
}
fn core_error(error: impl std::fmt::Display) -> FerrumError {
    FerrumError::backend(format!("MTP expansion: {error}"))
}
