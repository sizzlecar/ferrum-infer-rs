//! Explicit per-call observation context. It owns no execution authority and
//! must never affect the executor's result or physical dispatch decisions.

use super::*;

/// Demand for a dynamic actual sample from this submission only. It does not
/// enable backend capability, change logical attribution, or authorize work.
/// Public dispatch wrappers keep RuntimePolicy for compatibility.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StructuredCostSampleDemand {
    #[default]
    RuntimePolicy,
    Requested,
    NotRequested,
}

impl StructuredCostSampleDemand {
    pub fn for_call(
        policy: ferrum_types::SloStructuredActualCapturePolicy,
        observed_call: bool,
        artifact_consumer: bool,
    ) -> Self {
        if policy.is_legacy() {
            Self::RuntimePolicy
        } else if observed_call || artifact_consumer {
            Self::Requested
        } else {
            Self::NotRequested
        }
    }

    pub fn enabled(self, capability: ferrum_types::SloStructuredCostCapture) -> bool {
        capability == ferrum_types::SloStructuredCostCapture::HostSettledV1
            && self != Self::NotRequested
    }
}

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

    /// Cold identity for clocks whose nanosecond origin survives a process
    /// restart within the same boot and namespace. None preserves process-local
    /// calibration but cannot authorize persisted age/TTL comparisons.
    fn monotonic_domain(&self) -> Option<&CostMonotonicDomainV1> {
        None
    }
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
    participant_index: std::collections::HashMap<&'a RequestId, Option<usize>>,
    prepare_started_at_ns: Option<u64>,
    boundary: WaveObservationBoundary,
    handle: Option<WaveObservationHandle>,
    waves: usize,
    outcome: Option<ObservedCallOutcome>,
    unknown: Option<ActualWaveEvidenceUnknown>,
    terminal_recorded: bool,
    structured_capture: bool,
    numeric_observation: crate::vnext::DeviceCostObservationDemand,
}

impl<'a> PlanRuntimeCostObservationContext<'a> {
    pub fn new(
        recorder: &'a mut BoundedWaveRecorder,
        clock: &'a dyn CostObservationClock,
        participants: &'a [CostObservationParticipant],
        prepare_started_at_ns: Option<u64>,
        boundary: WaveObservationBoundary,
    ) -> Self {
        let mut participant_index = std::collections::HashMap::new();
        if participants.len() <= 1024 {
            for (index, participant) in participants.iter().enumerate() {
                participant_index
                    .entry(&participant.request_id)
                    .and_modify(|prior| *prior = None)
                    .or_insert(Some(index));
            }
        }
        Self {
            recorder,
            clock,
            participants,
            participant_index,
            prepare_started_at_ns,
            boundary,
            handle: None,
            waves: 0,
            outcome: None,
            unknown: None,
            terminal_recorded: false,
            structured_capture: false,
            numeric_observation: crate::vnext::DeviceCostObservationDemand::Required,
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

    /// Declared by the actual consumer before execution. Exact lifecycle and
    /// physical receipts remain required regardless of statistical demand.
    pub fn with_cost_observation_demand(
        mut self,
        demand: crate::vnext::DeviceCostObservationDemand,
    ) -> Self {
        self.numeric_observation = demand;
        self
    }
    pub fn cost_observation_demand(&self) -> crate::vnext::DeviceCostObservationDemand {
        if self.structured_capture {
            crate::vnext::DeviceCostObservationDemand::Required
        } else {
            self.numeric_observation
        }
    }

    pub fn now_ns(&self) -> Option<u64> {
        self.clock.now_ns()
    }

    pub fn participant(&self, request_id: &RequestId) -> Option<&CostObservationParticipant> {
        if self.participants.len() > 1024 {
            return None;
        }
        self.participant_index
            .get(request_id)
            .and_then(|index| *index)
            .and_then(|index| self.participants.get(index))
    }

    /// Called before encode/submit, with the receipt from the same actual
    /// program selector. No outcome or duration can influence membership.
    pub fn prepared_route(
        &mut self,
        route: PreparedCostRouteV1,
        rows: Result<Vec<ActualWaveRow>, ActualWaveEvidenceUnknown>,
    ) {
        self.recorder.diagnose_route_selection(&route);
        let proof = (|| {
            if self.waves != 0 || self.outcome.is_some() {
                return Err("selection_after_wave_or_outcome");
            }
            let selected_at_ns = self.now_ns().ok_or("selection_clock_missing")?;
            self.recorder.diagnose_route_clock(selected_at_ns);
            if self
                .prepare_started_at_ns
                .is_none_or(|started| selected_at_ns < started)
            {
                return Err("selection_before_preparation_or_missing_clock");
            }
            let rows = rows.map_err(|_| "selected_rows_unknown")?;
            if route.class().is_outside() {
                if rows.is_empty()
                    || rows.len() != self.participants.len()
                    || rows.len() > 1024
                    || rows.iter().enumerate().any(|(index, row)| {
                        rows[..index].iter().any(|prior| {
                            prior.request_id == row.request_id
                                || prior.input_index == row.input_index
                        }) || self.participant(&row.request_id).is_none_or(|host| {
                            host.owner_incarnation != row.owner_incarnation
                                || host.work_generation != row.work_generation
                                || host.input_index != row.input_index
                        })
                    })
                {
                    return Err("outside_selected_rows_invalid");
                }
            } else if !rows.is_empty() {
                return Err("eligible_selection_has_outside_rows");
            }
            Ok(PreparedCallRouteV1 {
                route,
                selected_at_ns,
                rows,
                submitted: None,
            })
        })();
        if let Err(gate) = proof.as_ref() {
            self.recorder.diagnose_route_rejection(gate);
        }
        self.recorder.record_prepared_route(proof.ok());
    }

    /// Binds the selected Outside proof to the original successful submit's
    /// private batch identity and native graph evidence. Device completion and
    /// host settlement are still required separately before retirement.
    pub fn route_submission(
        &mut self,
        attribution: Option<&crate::vnext::BoundDeviceSubmissionAttribution>,
    ) {
        self.recorder.record_route_submission(attribution);
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
        self.record_physical_wave(
            shape.map(PhysicalWaveInput::Ready),
            submission_started_at_ns,
        );
    }

    pub fn physical_wave_pending(
        &mut self,
        shape: Result<PendingActualWave, ActualWaveEvidenceUnknown>,
        submission_started_at_ns: Option<u64>,
    ) {
        self.record_physical_wave(
            shape.map(PhysicalWaveInput::Pending),
            submission_started_at_ns,
        );
    }

    fn record_physical_wave(
        &mut self,
        shape: Result<PhysicalWaveInput, ActualWaveEvidenceUnknown>,
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
                if shape.rows().iter().any(|row| {
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
                    match shape {
                        PhysicalWaveInput::Ready(shape) => {
                            self.recorder.begin(shape, self.boundary, prepare)
                        }
                        PhysicalWaveInput::Pending(shape) => {
                            self.recorder.begin_pending(shape, self.boundary, prepare)
                        }
                    }
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

    /// Finish an actual typed capacity decision whose executor contract rules
    /// out provider encode and inference submission. Generic Deferred is not
    /// sufficient. Partial physical work, unknown/lost evidence, a previously
    /// selected route, or a failed rollback cannot produce this receipt.
    pub fn finish_capacity_deferred(
        &mut self,
        deferral: &crate::model_executor::ExecutorExecutionDeferral,
    ) {
        self.finish_capacity_no_submission(deferral, ObservedCallOutcome::Deferred);
    }

    /// Batch prefill and mixed execution retain their original NotSubmitted
    /// control outcome while producing the same private capacity proof.
    pub fn finish_capacity_not_submitted(
        &mut self,
        deferral: &crate::model_executor::ExecutorExecutionDeferral,
    ) {
        self.finish_capacity_no_submission(deferral, ObservedCallOutcome::NotSubmitted);
    }

    /// Finish only the original core guard rejection after successful Step
    /// rollback. A selected route is allowed here solely through its exact
    /// Step/Invocation/lane and checked participant binding; it remains forbidden
    /// for the earlier capacity-deferral proof.
    pub fn finish_guard_rollback(&mut self, receipt: &GuardedNotSubmitted) {
        let proof = (|| {
            if self.waves != 0
                || self.handle.is_some()
                || self.outcome.is_some()
                || self.unknown.is_some()
                || self.terminal_recorded
            {
                return Err(ActualWaveEvidenceUnknown::InvalidLifecycle);
            }
            let binding = receipt
                .observation
                .as_ref()
                .ok_or(ActualWaveEvidenceUnknown::ParticipantCorrelation)?;
            let signature = super::guarded_participant_signature(self.participants.len(), |i| {
                let row = &self.participants[i];
                (&row.request_id, row.owner_incarnation, row.work_generation)
            })
            .ok_or(ActualWaveEvidenceUnknown::ParticipantCorrelation)?;
            if binding.participant_count != self.participants.len()
                || binding.participant_signature != signature
            {
                return Err(ActualWaveEvidenceUnknown::ParticipantCorrelation);
            }
            let prepare = self
                .prepare_started_at_ns
                .ok_or(ActualWaveEvidenceUnknown::Clock)?;
            let returned = self
                .now_ns()
                .filter(|now| *now >= prepare)
                .ok_or(ActualWaveEvidenceUnknown::Clock)?;
            let signature = super::no_submission::participant_signature(self.participants)
                .ok_or(ActualWaveEvidenceUnknown::ParticipantCorrelation)?;
            self.recorder
                .record_guard_rollback(
                    receipt,
                    prepare,
                    returned,
                    self.participants.len(),
                    signature,
                )
                .map_err(|_| ActualWaveEvidenceUnknown::InvalidLifecycle)
        })();
        if let Err(reason) = proof {
            self.mark_unknown(reason);
        }
        self.finish_call(ObservedCallOutcome::NotSubmitted);
    }

    fn finish_capacity_no_submission(
        &mut self,
        deferral: &crate::model_executor::ExecutorExecutionDeferral,
        outcome: ObservedCallOutcome,
    ) {
        let proof = (|| {
            if self.waves != 0
                || self.handle.is_some()
                || self.outcome.is_some()
                || self.unknown.is_some()
                || self.terminal_recorded
            {
                return Err(ActualWaveEvidenceUnknown::InvalidLifecycle);
            }
            let prepare = self
                .prepare_started_at_ns
                .ok_or(ActualWaveEvidenceUnknown::Clock)?;
            let returned = self
                .now_ns()
                .filter(|at| *at >= prepare)
                .ok_or(ActualWaveEvidenceUnknown::Clock)?;
            let signature = super::no_submission::participant_signature(self.participants)
                .ok_or(ActualWaveEvidenceUnknown::ParticipantCorrelation)?;
            let reason = CallNoSubmissionReasonV1::from_deferral(deferral)
                .ok_or(ActualWaveEvidenceUnknown::InvalidLifecycle)?;
            self.recorder
                .record_no_submission(
                    prepare,
                    returned,
                    self.participants.len(),
                    signature,
                    reason,
                )
                .map_err(|_| ActualWaveEvidenceUnknown::InvalidLifecycle)
        })();
        if let Err(reason) = proof {
            self.mark_unknown(reason);
        }
        self.finish_call(outcome);
    }
}

enum PhysicalWaveInput {
    Ready(ActualWaveShape),
    Pending(PendingActualWave),
}
impl PhysicalWaveInput {
    fn rows(&self) -> &[ActualWaveRow] {
        match self {
            Self::Ready(shape) => &shape.rows,
            Self::Pending(shape) => shape.rows(),
        }
    }
}

#[cfg(test)]
#[path = "context/structured_demand_tests.rs"]
mod structured_demand_tests;
