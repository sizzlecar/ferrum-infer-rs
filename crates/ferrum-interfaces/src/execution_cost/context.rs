//! Explicit per-call observation context. It owns no execution authority and
//! must never affect the executor's result or physical dispatch decisions.

use super::*;

/// Unavailable proves the observed entrypoint did no work. The nested executor
/// result is intentionally distinct: an Err there may follow device submission.
pub enum ObservedDispatch<T> {
    Unavailable,
    Executed(ferrum_types::Result<T>),
}

/// Implementations must be nonblocking and must not retain a global lock across
/// an async call. None means clock conversion/overflow failed, never timestamp 0.
pub trait CostObservationClock: Send + Sync {
    fn now_ns(&self) -> Option<u64>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CostObservationParticipant {
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub input_index: u32,
    /// Actual host sampling/structured-output policy; FullLogits alone does not
    /// describe engine commit cost. Missing policy evidence stays Unknown.
    pub output_policy_signature: Option<[u8; 32]>,
    /// Additional versioned host evidence; None preserves exact-only v1.
    pub host_features: Option<HostCostFeaturesV1>,
}

/// Call control outcomes are not physical waves. In particular, Unsupported
/// and a capacity deferral have no invented zero-cost actual shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservedCallOutcome {
    Completed,
    Unsupported,
    NotSubmitted,
    Deferred,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActualWaveEvidenceUnknown {
    ProviderPath,
    GraphPath,
    RecurrentState,
    OutputPolicy,
    ParticipantCorrelation,
    Clock,
    Capacity,
    InvalidLifecycle,
    ShapeOverflow,
}

/// The caller-provided preparation boundary is a claim about this one call.
/// Only a single isolated physical wave can later become a complete wall sample
/// after the engine adds its host commit. The executor never adds that commit.
pub struct PlanRuntimeCostObservationContext<'a> {
    recorder: &'a mut BoundedWaveRecorder,
    clock: &'a dyn CostObservationClock,
    participants: &'a [CostObservationParticipant],
    prepare_started_at_ns: Option<u64>,
    boundary: WaveObservationBoundary,
    handle: Option<WaveObservationHandle>,
    waves: usize,
    outcome: Option<ObservedCallOutcome>,
    unknown: Option<ActualWaveEvidenceUnknown>,
    terminal_recorded: bool,
    structured_capture: bool,
}

impl<'a> PlanRuntimeCostObservationContext<'a> {
    pub fn new(
        recorder: &'a mut BoundedWaveRecorder,
        clock: &'a dyn CostObservationClock,
        participants: &'a [CostObservationParticipant],
        prepare_started_at_ns: Option<u64>,
        boundary: WaveObservationBoundary,
    ) -> Self {
        Self {
            recorder,
            clock,
            participants,
            prepare_started_at_ns,
            boundary,
            handle: None,
            waves: 0,
            outcome: None,
            unknown: None,
            terminal_recorded: false,
            structured_capture: false,
        }
    }

    /// Passive evidence only; execution and cost-model policy are unchanged.
    pub fn with_structured_capture(mut self, enabled: bool) -> Self {
        self.structured_capture = enabled;
        self
    }
    pub fn structured_capture_enabled(&self) -> bool {
        self.structured_capture
    }

    pub fn now_ns(&self) -> Option<u64> {
        self.clock.now_ns()
    }

    pub fn participant(&self, request_id: &RequestId) -> Option<&CostObservationParticipant> {
        if self.participants.len() > 1024 {
            return None;
        }
        let mut found = self
            .participants
            .iter()
            .filter(|participant| &participant.request_id == request_id);
        let participant = found.next()?;
        found.next().is_none().then_some(participant)
    }

    pub fn unknown_reason(&self) -> Option<ActualWaveEvidenceUnknown> {
        self.unknown
    }
    pub fn call_outcome(&self) -> Option<ObservedCallOutcome> {
        self.outcome
    }
    pub fn physical_wave_count(&self) -> usize {
        self.waves
    }

    /// Clone the instance-fenced handle for the engine's eventual commit. None
    /// is not permission to train or to reserve output after physical work.
    pub fn wave_handle(&self) -> Option<WaveObservationHandle> {
        self.handle.clone()
    }

    pub fn mark_unknown(&mut self, reason: ActualWaveEvidenceUnknown) {
        self.unknown.get_or_insert(reason);
        self.recorder.note_unknown_evidence(reason);
    }

    /// Invoked only after a real dispatch decision. `submission_started_at_ns`
    /// was sampled immediately before that attempt, not after its completion.
    /// Unknown shape retains a physical observation without guessed fields.
    pub fn physical_wave(
        &mut self,
        shape: Result<ActualWaveShape, ActualWaveEvidenceUnknown>,
        submission_started_at_ns: Option<u64>,
    ) {
        self.waves = self.waves.saturating_add(1);
        self.terminal_recorded = false;
        if self.waves > 1 {
            if let Some(handle) = &self.handle {
                let _ = self.recorder.mark_composite(handle);
            }
            self.boundary = WaveObservationBoundary::CompositeDeferredCommit;
        }
        // A failed second begin must never let its terminal update a sibling.
        self.handle = None;
        let (Some(prepare), Some(submit)) = (self.prepare_started_at_ns, submission_started_at_ns)
        else {
            self.mark_unknown(ActualWaveEvidenceUnknown::Clock);
            self.recorder.note_lost(1);
            return;
        };
        let result = match shape {
            Ok(shape) => {
                if shape.rows.iter().any(|row| {
                    self.participant(&row.request_id).is_none_or(|expected| {
                        expected.owner_incarnation != row.owner_incarnation
                            || expected.work_generation != row.work_generation
                            || expected.input_index != row.input_index
                    })
                }) {
                    self.mark_unknown(ActualWaveEvidenceUnknown::ParticipantCorrelation);
                    self.recorder.begin_unknown(
                        self.boundary,
                        prepare,
                        ActualWaveEvidenceUnknown::ParticipantCorrelation,
                    )
                } else {
                    self.recorder.begin(shape, self.boundary, prepare)
                }
            }
            Err(reason) => {
                self.mark_unknown(reason);
                self.recorder.begin_unknown(self.boundary, prepare, reason)
            }
        };
        match result {
            Ok(handle) => {
                if self.recorder.submission_started(&handle, submit).is_err() {
                    self.mark_unknown(ActualWaveEvidenceUnknown::InvalidLifecycle);
                }
                self.handle = Some(handle);
            }
            Err(_) => self.mark_unknown(ActualWaveEvidenceUnknown::Capacity),
        }
    }

    /// Leaves cancellation as an unfinished observation. No RAII guard invents
    /// a terminal fence or reclaims any executor-owned resource.
    pub fn terminal(&mut self, outcome: ActualWaveOutcome, device_elapsed_ns: Option<NonZeroU64>) {
        if self.terminal_recorded {
            return;
        }
        let Some(handle) = self.handle.clone() else {
            return;
        };
        let Some(now) = self.clock.now_ns() else {
            self.mark_unknown(ActualWaveEvidenceUnknown::Clock);
            return;
        };
        if self
            .recorder
            .terminal(&handle, outcome, now, device_elapsed_ns)
            .is_err()
        {
            self.mark_unknown(ActualWaveEvidenceUnknown::InvalidLifecycle);
        } else {
            self.terminal_recorded = true;
        }
    }

    pub fn finish_call(&mut self, outcome: ObservedCallOutcome) {
        if self.outcome.replace(outcome).is_some() {
            self.mark_unknown(ActualWaveEvidenceUnknown::InvalidLifecycle);
        }
        if self.waves > 0
            && matches!(
                outcome,
                ObservedCallOutcome::Unsupported
                    | ObservedCallOutcome::Deferred
                    | ObservedCallOutcome::NotSubmitted
            )
        {
            self.mark_unknown(ActualWaveEvidenceUnknown::InvalidLifecycle);
        }
    }
}
