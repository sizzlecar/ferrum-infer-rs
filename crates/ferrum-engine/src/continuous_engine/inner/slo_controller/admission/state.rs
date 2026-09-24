//! Acceptance and finite forecast evidence are distinct lifecycle facts.
use super::*;

/// Installed only by an actual ingress publication. This constant-size state
/// neither owns resources nor changes the request's original latency clocks.
#[derive(Debug)]
pub(in crate::continuous_engine) struct SequenceTimeAdmission {
    pub(in crate::continuous_engine::inner::slo_controller) recovery_service: RecoveryServiceDebt,
    pub(in crate::continuous_engine::inner::slo_controller) recovery_seen: bool,
    pub(super) ingress: Instant,
    pub(super) original_input_tokens: usize,
    pub(super) maximum_output_tokens: usize,
    pub(super) last_assessment: Option<TimeAdmissionAssessment>,
    pub(super) started: Option<StartedTimeWitness>,
    /// Retained across capacity yield/recompute. A continuation is not a new
    /// fresh owner competing for time admission again.
    pub(super) activated: bool,
    pub(super) deferred: Option<super::defer::TimeAdmissionReview>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TimeAdmissionAssessmentKind {
    Feasible,
    Unknown(TimeAdmissionUnknown),
    Deferred(TimeAdmissionDeferReason),
    /// Original wait review elapsed; one real excess activation proceeds
    /// without a time promise, under the completion-first physical protocol.
    OverdueCompletion,
    AlreadyImpossible,
    /// This first slice evaluates already accepted owners; it cannot apply a
    /// before-acceptance rejection to them, even under RequireSlo.
    InvalidAcceptanceBoundary,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct TimeAdmissionAssessment {
    pub at: Instant,
    pub snapshot_generation: u64,
    pub model_version: u64,
    pub kind: TimeAdmissionAssessmentKind,
}

/// This is evidence that execution of a finite common forecast began, not a
/// lifetime guarantee, a benchmark acceptance result, or a fresh SLO origin.
#[derive(Debug, Clone, Copy)]
pub(super) struct StartedTimeWitness {
    pub snapshot_generation: u64,
    pub model_version: u64,
    pub validated_through: Instant,
    pub obligations_beyond_horizon: usize,
}

/// Retained with the actual guarded flight. Its private constructor can only
/// follow a successful time assessment against the complete live snapshot.
#[derive(Debug)]
pub(in crate::continuous_engine::inner::slo_controller) struct PendingTimeWitness {
    pub request_id: RequestId,
    pub owner: Arc<()>,
    pub work_generation: u64,
    pub ingress: Instant,
    pub original_input_tokens: usize,
    pub maximum_output_tokens: usize,
    pub(super) evidence: StartedTimeWitness,
}

impl SequenceTimeAdmission {
    #[cfg(test)]
    pub(in crate::continuous_engine::inner::slo_controller) fn has_started_witness(&self) -> bool {
        self.started.is_some()
    }

    pub(in crate::continuous_engine) fn record_recovery_progress(&mut self) {
        self.activated = true;
        self.recovery_service.progressed();
    }

    pub(super) fn matches(&self, sequence: &SequenceState) -> bool {
        sequence.slo.as_ref().is_some_and(|slo| {
            slo.is_trusted()
                && slo.ingress() == self.ingress
                && sequence.input_tokens.len() == self.original_input_tokens
                && sequence.sampling_params.max_tokens == self.maximum_output_tokens
        })
    }
}

impl TimeAdmissionAssessmentKind {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Feasible => "feasible_within_horizon",
            Self::Unknown(_) => "unknown",
            Self::Deferred(_) => "deferred_time_promise",
            Self::OverdueCompletion => "overdue_completion",
            Self::AlreadyImpossible => "best_effort",
            Self::InvalidAcceptanceBoundary => "invalid_acceptance_boundary",
        }
    }
}

impl EngineInner {
    /// Called after the actual scheduler item and engine owner are installed,
    /// under the common ingress/iteration lock. Off leaves no extra state.
    pub(in crate::continuous_engine) fn initialize_sequence_time_admission(
        &self,
        sequence: &mut SequenceState,
    ) {
        if self.config.scheduler.slo.mode == ferrum_types::SloMode::Off
            || sequence.time_admission.is_some()
        {
            return;
        }
        let Some(slo) = sequence.slo.as_ref().filter(|slo| slo.is_trusted()) else {
            return;
        };
        sequence.time_admission = Some(SequenceTimeAdmission {
            recovery_service: RecoveryServiceDebt::new(
                self.config.scheduler.slo.admission.max_active_requests,
            ),
            recovery_seen: false,
            ingress: slo.ingress(),
            original_input_tokens: sequence.input_tokens.len(),
            maximum_output_tokens: sequence.sampling_params.max_tokens,
            last_assessment: None,
            started: None,
            activated: false,
            deferred: None,
        });
        counter!("ferrum.engine.slo_time_admission_accepted_untimed_total").increment(1);
    }

    /// Called only for an actual Submitted outcome, before host commit mutates
    /// a selected owner's generation. Rejection, withdrawal and Observe never
    /// call this function. The native guard already checked the same witness
    /// before irreversible submission; completion of its await may be later.
    pub(in crate::continuous_engine::inner::slo_controller) fn record_started_time_witness(
        &self,
        pending: &PendingTimeWitness,
    ) {
        let mut sequences = self.sequences.write();
        let Some(sequence) = sequences.get_mut(&pending.request_id) else {
            return;
        };
        if !Arc::ptr_eq(&sequence.stream_projection_identity, &pending.owner)
            || !sequence.generated_tokens.is_empty()
            || sequence
                .cost_frontier
                .is_none_or(|frontier| frontier.work_generation.get() != pending.work_generation)
            || sequence.input_tokens.len() != pending.original_input_tokens
            || sequence.sampling_params.max_tokens != pending.maximum_output_tokens
            || sequence
                .slo
                .as_ref()
                .is_none_or(|slo| !slo.is_trusted() || slo.ingress() != pending.ingress)
        {
            return;
        }
        let Some(state) = sequence.time_admission.as_mut() else {
            return;
        };
        if state.started.is_some()
            || state.ingress != pending.ingress
            || state.original_input_tokens != pending.original_input_tokens
            || state.maximum_output_tokens != pending.maximum_output_tokens
        {
            return;
        }
        state.started = Some(pending.evidence);
        counter!("ferrum.engine.slo_time_witness_started_total").increment(1);
        tracing::trace!(
            request_id = %pending.request_id,
            snapshot_generation = pending.evidence.snapshot_generation,
            model_version = pending.evidence.model_version,
            obligations_beyond_horizon = pending.evidence.obligations_beyond_horizon,
            validated_through = ?pending.evidence.validated_through,
            "started finite time-admission witness after guarded submission"
        );
    }
}
