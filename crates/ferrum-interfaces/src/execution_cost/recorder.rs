use super::*;
mod memory;
mod pending;
pub use memory::*;
mod route_diagnostic;
pub use route_diagnostic::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostRecorderLimits {
    pub max_waves: usize,
    /// Physical row-vector capacities, not auxiliary statistical inputs.
    pub max_rows_per_wave: usize,
    /// Conservative retention units; structured auxiliary bytes are rounded
    /// up to CostRowNumericFeatures-sized units in addition to physical rows.
    pub max_retained_rows: usize,
}

impl CostRecorderLimits {
    pub fn validate(self) -> Result<(), CostRecorderError> {
        if self.max_waves == 0
            || self.max_waves > 4096
            || self.max_rows_per_wave == 0
            || self.max_rows_per_wave > 1024
            || self.max_retained_rows < self.max_rows_per_wave
            || self.max_retained_rows > 65_536
        {
            Err(CostRecorderError::InvalidLimits)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CostRecorderError {
    #[error("invalid observation retention limits")]
    InvalidLimits,
    #[error("observation allocation or wave capacity exhausted")]
    WaveCapacity,
    #[error("observation row capacity exhausted")]
    RowCapacity,
    #[error("invalid actual physical-wave shape")]
    InvalidShape,
    #[error("observation handle belongs to a different recorder")]
    HandleMismatch,
    #[error("invalid observation lifecycle transition")]
    InvalidTransition,
    #[error("observation clock or device interval is invalid")]
    InvalidTiming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostObservationUnknownReason {
    EvidenceUnavailable(ActualWaveEvidenceUnknown),
    LostObservations,
    InvalidObservation,
    IncompleteObservation,
    NoObservations,
    OutcomeNotCompleted,
    BoundaryNotIsolated,
    OverlappingSiblingWave,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostObservationCoverage {
    Complete,
    Unknown {
        reason: CostObservationUnknownReason,
        lost_observations: u64,
    },
}

/// Private instance fence prevents a handle from another recorder with the same
/// caller-supplied call ID and ordinal from being accepted. No global ID needed.
#[derive(Debug, Clone)]
pub struct WaveObservationHandle {
    fence: Arc<()>,
    index: usize,
}

#[derive(Debug)]
pub struct BoundedWaveRecorder {
    fence: Arc<()>,
    call_id: NonZeroU64,
    limits: CostRecorderLimits,
    byte_limits: CostRecorderByteLimits,
    pending_retained_bytes: usize,
    pending_working_bytes: usize,
    pending_retained_rows: usize,
    memory_audit: CostRecorderMemoryAudit,
    observations: Vec<ActualWaveObservation>,
    pending: Vec<Option<PendingActualWave>>,
    next_wave_ordinal: u64,
    retained_rows: usize,
    prepared_route: Option<PreparedCallRouteV1>,
    prepared_route_attempts: usize,
    route_unknown: bool,
    route_diagnostic: Option<RouteCaptureDiagnostic>,
    lost_observations: u64,
    invalid_observation: bool,
    unknown_evidence: Option<ActualWaveEvidenceUnknown>,
    no_submission: Option<CallNoSubmissionV1>,
}

impl BoundedWaveRecorder {
    /// A receipt remains valid only while this original recorder has never
    /// attempted a physical observation or selected/submitted a device route.
    pub fn no_submission(&self) -> Option<&CallNoSubmissionV1> {
        self.no_submission
            .as_ref()
            .filter(|proof| match proof.reason {
                CallNoSubmissionReasonV1::GuardRollback {
                    batch_step,
                    batch_invocation,
                    lane_id,
                    ..
                } => self.pristine_guard_rollback(batch_step, batch_invocation, lane_id),
                _ => self.pristine_without_wave(),
            })
    }

    fn pristine_without_wave(&self) -> bool {
        // Passive diagnostics may be enabled before any work. Only the actual
        // recorder lifecycle can disqualify a typed no-submission receipt.
        self.next_wave_ordinal == 0
            && self.observations.is_empty()
            && self.pending.is_empty()
            && self.prepared_route.is_none()
            && self.prepared_route_attempts == 0
            && !self.route_unknown
            && self.lost_observations == 0
            && !self.invalid_observation
            && self.unknown_evidence.is_none()
    }

    fn pristine_guard_rollback(&self, step: u64, invocation: u64, lane: u64) -> bool {
        self.next_wave_ordinal == 0
            && self.observations.is_empty()
            && self.pending.is_empty()
            && self.prepared_route_attempts == 1
            && !self.route_unknown
            && self.lost_observations == 0
            && !self.invalid_observation
            && self.unknown_evidence.is_none()
            && self.prepared_route.as_ref().is_some_and(|p| {
                p.submitted.is_none()
                    && p.route.class != PreparedCostRouteClassV1::Unknown
                    && p.route.batch_step == Some(step)
                    && p.route.batch_invocation == Some(invocation)
                    && p.route.lane_id == lane
            })
    }

    pub(super) fn record_guard_rollback(
        &mut self,
        receipt: &GuardedNotSubmitted,
        prepare_started_at_ns: u64,
        returned_at_ns: u64,
        participant_count: usize,
        participant_signature: [u8; 32],
    ) -> Result<(), CostRecorderError> {
        let Some(bound) = receipt.observation.as_ref() else {
            self.invalid_observation = true;
            return Err(CostRecorderError::InvalidTransition);
        };
        if !self.pristine_guard_rollback(bound.batch_step, bound.batch_invocation, bound.lane_id)
            || self.no_submission.is_some()
            || self.prepared_route.as_ref().is_none_or(|p| {
                p.selected_at_ns < prepare_started_at_ns || returned_at_ns < p.selected_at_ns
            })
        {
            self.invalid_observation = true;
            return Err(CostRecorderError::InvalidTransition);
        }
        self.no_submission = Some(CallNoSubmissionV1 {
            call_id: self.call_id.get(),
            prepare_started_at_ns,
            returned_at_ns,
            participant_count,
            participant_signature,
            reason: CallNoSubmissionReasonV1::GuardRollback {
                batch_step: bound.batch_step,
                batch_invocation: bound.batch_invocation,
                lane_id: bound.lane_id,
                reason: receipt.reason(),
            },
        });
        Ok(())
    }

    pub(super) fn record_no_submission(
        &mut self,
        prepare_started_at_ns: u64,
        returned_at_ns: u64,
        participant_count: usize,
        participant_signature: [u8; 32],
        reason: CallNoSubmissionReasonV1,
    ) -> Result<(), CostRecorderError> {
        if !self.pristine_without_wave() || self.no_submission.is_some() {
            self.invalid_observation = true;
            return Err(CostRecorderError::InvalidTransition);
        }
        if returned_at_ns < prepare_started_at_ns {
            self.invalid_observation = true;
            return Err(CostRecorderError::InvalidTiming);
        }
        self.no_submission = Some(CallNoSubmissionV1 {
            call_id: self.call_id.get(),
            prepare_started_at_ns,
            returned_at_ns,
            participant_count,
            participant_signature,
            reason,
        });
        Ok(())
    }

    /// Allocate the bounded wave slots before execution, not during callbacks.
    pub fn new(call_id: NonZeroU64, limits: CostRecorderLimits) -> Result<Self, CostRecorderError> {
        Self::new_with_byte_limits(call_id, limits, CostRecorderByteLimits::default())
    }

    pub fn new_with_byte_limits(
        call_id: NonZeroU64,
        limits: CostRecorderLimits,
        byte_limits: CostRecorderByteLimits,
    ) -> Result<Self, CostRecorderError> {
        limits.validate()?;
        byte_limits.validate()?;
        let mut observations = Vec::new();
        observations
            .try_reserve_exact(limits.max_waves)
            .map_err(|_| CostRecorderError::WaveCapacity)?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(limits.max_waves)
            .map_err(|_| CostRecorderError::WaveCapacity)?;
        Ok(Self {
            fence: Arc::new(()),
            call_id,
            limits,
            byte_limits,
            pending_retained_bytes: 0,
            pending_working_bytes: 0,
            pending_retained_rows: 0,
            memory_audit: CostRecorderMemoryAudit::default(),
            observations,
            pending,
            next_wave_ordinal: 0,
            retained_rows: 0,
            prepared_route: None,
            prepared_route_attempts: 0,
            route_unknown: false,
            route_diagnostic: None,
            lost_observations: 0,
            invalid_observation: false,
            unknown_evidence: None,
            no_submission: None,
        })
    }

    pub fn begin(
        &mut self,
        shape: ActualWaveShape,
        boundary: WaveObservationBoundary,
        prepare_started_at_ns: u64,
    ) -> Result<WaveObservationHandle, CostRecorderError> {
        if self.no_submission.is_some() {
            return self.reject_begin(CostRecorderError::InvalidTransition);
        }
        let ordinal = self.next_wave_ordinal;
        self.next_wave_ordinal = self.next_wave_ordinal.saturating_add(1);
        let Ok(physical_wave_ordinal) = u32::try_from(ordinal) else {
            return self.reject_begin(CostRecorderError::WaveCapacity);
        };
        let validation = shape.validate(self.limits.max_rows_per_wave);
        if let Err(error) = validation {
            return self.reject_begin(error);
        }
        if self.observations.len() >= self.limits.max_waves {
            return self.reject_begin(CostRecorderError::WaveCapacity);
        }
        // Retention bounds allocated storage, not only the number of live rows.
        // An oversized caller-owned Vec must not smuggle unbounded spare space.
        if shape.rows.capacity() > self.limits.max_rows_per_wave {
            return self.reject_begin(CostRecorderError::RowCapacity);
        }
        let numeric_rows = shape
            .numeric_features
            .as_ref()
            .map_or(0, |features| features.rows.capacity());
        if numeric_rows > self.limits.max_rows_per_wave {
            return self.reject_begin(CostRecorderError::RowCapacity);
        }
        let static_rows = shape
            .row_multiset_features
            .as_ref()
            .map_or(0, |features| features.rows.capacity());
        if static_rows > self.limits.max_rows_per_wave {
            return self.reject_begin(CostRecorderError::RowCapacity);
        }
        let structured_rows = shape
            .statistical_evidence
            .as_ref()
            .map_or(0, |evidence| evidence.structured_retained_rows());
        if shape
            .statistical_evidence
            .as_ref()
            .map_or(0, |evidence| evidence.structured_physical_host_capacity())
            > self.limits.max_rows_per_wave
        {
            return self.reject_begin(CostRecorderError::RowCapacity);
        }
        let Some(rows) = shape
            .rows
            .capacity()
            .checked_add(numeric_rows)
            .and_then(|rows| rows.checked_add(static_rows))
            .and_then(|rows| rows.checked_add(structured_rows))
            .and_then(|rows| self.retained_rows.checked_add(rows))
        else {
            return self.reject_begin(CostRecorderError::RowCapacity);
        };
        if rows > self.limits.max_retained_rows {
            return self.reject_begin(CostRecorderError::RowCapacity);
        }
        let index = self.observations.len();
        self.observations.push(ActualWaveObservation {
            call_id: self.call_id,
            physical_wave_ordinal,
            shape: Some(shape),
            shape_unknown: None,
            boundary,
            prepare_started_at_ns,
            submission_started_at_ns: None,
            terminal_at_ns: None,
            host_committed_at_ns: None,
            device_elapsed_ns: None,
            outcome: None,
        });
        self.retained_rows = rows;
        self.pending.push(None);
        Ok(WaveObservationHandle {
            fence: Arc::clone(&self.fence),
            index,
        })
    }

    /// A dropped callback / trace must be reported even when no handle exists.
    /// Saturation is intentional: loss stays visible and can never wrap to zero.
    pub fn note_lost(&mut self, count: u64) {
        self.lost_observations = self.lost_observations.saturating_add(count);
    }

    pub fn note_unknown_evidence(&mut self, reason: ActualWaveEvidenceUnknown) {
        self.unknown_evidence.get_or_insert(reason);
        // A first GraphPath cannot hide a later lifecycle/capacity/identity
        // error from the independent Outside population proof.
        self.route_unknown |= reason != ActualWaveEvidenceUnknown::GraphPath;
    }

    /// Preserve a real physical wave without inventing unavailable shape fields.
    pub fn begin_unknown(
        &mut self,
        boundary: WaveObservationBoundary,
        prepare_started_at_ns: u64,
        reason: ActualWaveEvidenceUnknown,
    ) -> Result<WaveObservationHandle, CostRecorderError> {
        self.note_unknown_evidence(reason);
        let ordinal = self.next_wave_ordinal;
        self.next_wave_ordinal = self.next_wave_ordinal.saturating_add(1);
        let Ok(physical_wave_ordinal) = u32::try_from(ordinal) else {
            return self.reject_begin(CostRecorderError::WaveCapacity);
        };
        if self.observations.len() >= self.limits.max_waves {
            return self.reject_begin(CostRecorderError::WaveCapacity);
        }
        let index = self.observations.len();
        self.observations.push(ActualWaveObservation {
            call_id: self.call_id,
            physical_wave_ordinal,
            shape: None,
            shape_unknown: Some(reason),
            boundary,
            prepare_started_at_ns,
            submission_started_at_ns: None,
            terminal_at_ns: None,
            host_committed_at_ns: None,
            device_elapsed_ns: None,
            outcome: None,
        });
        self.pending.push(None);
        Ok(WaveObservationHandle {
            fence: Arc::clone(&self.fence),
            index,
        })
    }

    /// Only downgrades a boundary when an outer call reveals sibling work.
    pub fn mark_composite(
        &mut self,
        handle: &WaveObservationHandle,
    ) -> Result<(), CostRecorderError> {
        self.edit(handle, |wave| {
            wave.boundary = WaveObservationBoundary::CompositeDeferredCommit;
            Ok(())
        })
    }

    fn reject_begin<T>(&mut self, error: CostRecorderError) -> Result<T, CostRecorderError> {
        self.note_lost(1);
        if error == CostRecorderError::InvalidShape {
            self.invalid_observation = true;
        }
        Err(error)
    }

    fn edit(
        &mut self,
        handle: &WaveObservationHandle,
        update: impl FnOnce(&mut ActualWaveObservation) -> Result<(), CostRecorderError>,
    ) -> Result<(), CostRecorderError> {
        if !Arc::ptr_eq(&self.fence, &handle.fence) || handle.index >= self.observations.len() {
            self.invalid_observation = true;
            return Err(CostRecorderError::HandleMismatch);
        }
        let result = update(&mut self.observations[handle.index]);
        if result.is_err() {
            self.invalid_observation = true;
        }
        result
    }

    pub fn submission_started(
        &mut self,
        handle: &WaveObservationHandle,
        at_ns: u64,
    ) -> Result<(), CostRecorderError> {
        self.edit(handle, |wave| {
            if wave.submission_started_at_ns.is_some() || wave.outcome.is_some() {
                return Err(CostRecorderError::InvalidTransition);
            }
            if at_ns < wave.prepare_started_at_ns {
                return Err(CostRecorderError::InvalidTiming);
            }
            wave.submission_started_at_ns = Some(at_ns);
            Ok(())
        })
    }

    pub fn terminal(
        &mut self,
        handle: &WaveObservationHandle,
        outcome: ActualWaveOutcome,
        at_ns: u64,
        device_elapsed_ns: Option<NonZeroU64>,
    ) -> Result<(), CostRecorderError> {
        self.edit(handle, |wave| {
            if wave.outcome.is_some() {
                return Err(CostRecorderError::InvalidTransition);
            }
            let not_submitted = matches!(
                outcome,
                ActualWaveOutcome::NotSubmitted | ActualWaveOutcome::Deferred
            );
            // A submission attempt can return DefinitelyNotSubmitted. Deferred
            // may occur before any attempt. Neither may report device work.
            if (!not_submitted && wave.submission_started_at_ns.is_none())
                || (not_submitted && device_elapsed_ns.is_some())
            {
                return Err(CostRecorderError::InvalidTransition);
            }
            let start = wave
                .submission_started_at_ns
                .unwrap_or(wave.prepare_started_at_ns);
            let Some(elapsed) = at_ns.checked_sub(start) else {
                return Err(CostRecorderError::InvalidTiming);
            };
            if device_elapsed_ns.is_some_and(|duration| duration.get() > elapsed) {
                return Err(CostRecorderError::InvalidTiming);
            }
            wave.outcome = Some(outcome);
            wave.terminal_at_ns = Some(at_ns);
            wave.device_elapsed_ns = device_elapsed_ns;
            Ok(())
        })
    }

    pub fn host_committed(
        &mut self,
        handle: &WaveObservationHandle,
        at_ns: u64,
    ) -> Result<(), CostRecorderError> {
        self.edit(handle, |wave| {
            if wave.host_committed_at_ns.is_some()
                || wave.outcome != Some(ActualWaveOutcome::Completed)
            {
                return Err(CostRecorderError::InvalidTransition);
            }
            if at_ns
                < wave
                    .terminal_at_ns
                    .ok_or(CostRecorderError::InvalidTransition)?
                || at_ns <= wave.prepare_started_at_ns
            {
                return Err(CostRecorderError::InvalidTiming);
            }
            wave.host_committed_at_ns = Some(at_ns);
            Ok(())
        })
    }

    /// Passive pre-submit evidence. Repeated preparations (including a
    /// definitely-not-submitted retry) invalidate the one-attempt scope proof.
    pub(crate) fn record_prepared_route(&mut self, route: Option<PreparedCallRouteV1>) {
        self.prepared_route_attempts = self.prepared_route_attempts.saturating_add(1);
        if self.prepared_route_attempts != 1 || !self.observations.is_empty() {
            self.diagnose_route_rejection("multiple_or_late_preparation");
            self.prepared_route = None;
            return;
        }
        let Some(route) = route else { return };
        let Some(retained) = self.retained_rows.checked_add(route.rows.capacity()) else {
            self.diagnose_route_rejection("prepared_rows_capacity_overflow");
            return;
        };
        if route.rows.capacity() > self.limits.max_rows_per_wave
            || retained > self.limits.max_retained_rows
        {
            self.diagnose_route_rejection("prepared_rows_capacity_exceeded");
            return;
        }
        self.retained_rows = retained;
        self.prepared_route = Some(route);
    }

    pub(crate) fn record_route_submission(
        &mut self,
        attribution: Option<&crate::vnext::BoundDeviceSubmissionAttribution>,
    ) {
        self.diagnose_route_submission(attribution);
        let Some(prepared) = self.prepared_route.as_mut() else {
            // A submission callback without its preparation is a real invalid
            // transition even when passive route diagnostics are disabled.
            self.invalid_observation = true;
            self.diagnose_route_rejection("prepared_route_missing_at_submission");
            return;
        };
        if self.prepared_route_attempts != 1
            || self.observations.len() != 1
            || prepared.submitted.is_some()
        {
            self.diagnose_route_rejection("submission_not_one_preparation_one_wave");
            self.prepared_route = None;
            return;
        }
        let Some(attribution) = attribution else {
            self.diagnose_route_rejection("submission_attribution_missing");
            self.prepared_route = None;
            return;
        };
        let Some(submitted_at) = self.observations[0].submission_started_at_ns else {
            self.diagnose_route_rejection("submission_clock_missing");
            self.prepared_route = None;
            return;
        };
        let identity = attribution.batch_identity();
        let graph = attribution.device().graph_evidence();
        if prepared.selected_at_ns > submitted_at
            || prepared.route.lane_id != identity.lane_id().get()
        {
            let gate = if prepared.selected_at_ns > submitted_at {
                "selection_after_submission"
            } else {
                "submitted_lane_mismatch"
            };
            self.diagnose_route_rejection(gate);
            self.prepared_route = None;
            return;
        }
        if let Some(program) = prepared.route.program_id() {
            if prepared.route.batch_step != Some(identity.batch_step_id().get())
                || prepared.route.batch_invocation != Some(identity.batch_invocation_id().get())
                || program.lane_id() != identity.lane_id()
                || program.plan_hash() != identity.plan_hash()
                || program.runtime_implementation_fingerprint()
                    != identity.runtime_implementation_fingerprint()
                || prepared.route.graph_state.is_none()
                || prepared.route.graph_state != graph.map(|g| g.before_state())
            {
                let gate = if prepared.route.batch_step != Some(identity.batch_step_id().get()) {
                    "submitted_step_mismatch"
                } else if prepared.route.batch_invocation
                    != Some(identity.batch_invocation_id().get())
                {
                    "submitted_invocation_mismatch"
                } else if program.lane_id() != identity.lane_id() {
                    "submitted_program_lane_mismatch"
                } else if program.plan_hash() != identity.plan_hash() {
                    "submitted_plan_mismatch"
                } else if program.runtime_implementation_fingerprint()
                    != identity.runtime_implementation_fingerprint()
                {
                    "submitted_runtime_mismatch"
                } else if prepared.route.graph_state.is_none() {
                    "selected_graph_missing"
                } else {
                    "submitted_graph_before_mismatch"
                };
                self.diagnose_route_rejection(gate);
                self.prepared_route = None;
                return;
            }
        } else if let Some(eager) = &prepared.route.non_reusable_wave {
            if prepared.route.class() != PreparedCostRouteClassV1::OutsideProgramLayoutAbsent
                || prepared.route.reason() != PreparedCostRouteReasonV1::ProgramLayoutAbsent
                || prepared.route.batch_step != Some(identity.batch_step_id().get())
                || prepared.route.batch_invocation != Some(identity.batch_invocation_id().get())
                || eager.plan_hash != identity.plan_hash().as_str()
                || eager.runtime_implementation_fingerprint
                    != identity.runtime_implementation_fingerprint()
                || prepared.route.catalog_epoch != Some(prepared.route.lane_epoch)
                || graph.is_none_or(|actual| {
                    prepared.route.graph_state != Some(actual.before_state())
                        || !actual.proves_configured_eager_observation()
                })
            {
                self.diagnose_route_rejection("submitted_non_reusable_wave_mismatch");
                self.prepared_route = None;
                return;
            }
        } else if prepared.route.class() != PreparedCostRouteClassV1::GraphDisabled {
            self.diagnose_route_rejection("program_identity_missing_for_selected_class");
            self.prepared_route = None;
            return;
        }
        prepared.submitted = Some(SubmittedRouteEvidenceV1 {
            batch_step: identity.batch_step_id().get(),
            batch_invocation: identity.batch_invocation_id().get(),
            plan_hash: identity.plan_hash().as_str().to_owned(),
            runtime_implementation_fingerprint: identity
                .runtime_implementation_fingerprint()
                .to_owned(),
            lane_id: identity.lane_id().get(),
            submission_started_at_ns: submitted_at,
            graph,
        });
    }

    pub fn prepared_route(&self) -> Option<&PreparedCallRouteV1> {
        (self.prepared_route_attempts == 1 && !self.route_unknown)
            .then_some(self.prepared_route.as_ref())
            .flatten()
    }

    pub fn observations(&self) -> &[ActualWaveObservation] {
        &self.observations
    }

    pub fn coverage(&self) -> CostObservationCoverage {
        let reason = if self.lost_observations > 0 {
            Some(CostObservationUnknownReason::LostObservations)
        } else if self.invalid_observation {
            Some(CostObservationUnknownReason::InvalidObservation)
        } else if let Some(reason) = self.unknown_evidence {
            Some(CostObservationUnknownReason::EvidenceUnavailable(reason))
        } else if self.pending.iter().any(Option::is_some) {
            Some(CostObservationUnknownReason::IncompleteObservation)
        } else if self.observations.is_empty() {
            Some(CostObservationUnknownReason::NoObservations)
        } else if self.observations.iter().any(|wave| {
            wave.outcome.is_none()
                || (wave.outcome == Some(ActualWaveOutcome::Completed)
                    && wave.boundary == WaveObservationBoundary::IsolatedPreparationToCommit
                    && wave.host_committed_at_ns.is_none())
        }) {
            Some(CostObservationUnknownReason::IncompleteObservation)
        } else {
            None
        };
        match reason {
            Some(reason) => CostObservationCoverage::Unknown {
                reason,
                lost_observations: self.lost_observations,
            },
            None => CostObservationCoverage::Complete,
        }
    }

    /// The full measured wall sample, never reconstructed from stage sums.
    /// Callers must use this gate before training, not merely inspect outcome.
    pub fn trainable_wall_ns(
        &self,
        handle: &WaveObservationHandle,
    ) -> Result<NonZeroU64, CostObservationUnknownReason> {
        if !Arc::ptr_eq(&self.fence, &handle.fence) {
            return Err(CostObservationUnknownReason::InvalidObservation);
        }
        if let CostObservationCoverage::Unknown { reason, .. } = self.coverage() {
            return Err(reason);
        }
        let wave = self
            .observations
            .get(handle.index)
            .ok_or(CostObservationUnknownReason::InvalidObservation)?;
        if wave.outcome != Some(ActualWaveOutcome::Completed) {
            return Err(CostObservationUnknownReason::OutcomeNotCompleted);
        }
        if wave.boundary != WaveObservationBoundary::IsolatedPreparationToCommit {
            return Err(CostObservationUnknownReason::BoundaryNotIsolated);
        }
        let end = wave
            .host_committed_at_ns
            .ok_or(CostObservationUnknownReason::IncompleteObservation)?;
        for (index, sibling) in self.observations.iter().enumerate() {
            if index == handle.index {
                continue;
            }
            let sibling_end = sibling
                .host_committed_at_ns
                .or(sibling.terminal_at_ns)
                .ok_or(CostObservationUnknownReason::IncompleteObservation)?;
            if sibling.prepare_started_at_ns < end && wave.prepare_started_at_ns < sibling_end {
                return Err(CostObservationUnknownReason::OverlappingSiblingWave);
            }
        }
        end.checked_sub(wave.prepare_started_at_ns)
            .and_then(NonZeroU64::new)
            .ok_or(CostObservationUnknownReason::InvalidObservation)
    }
}
