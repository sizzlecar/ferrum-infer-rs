use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostRecorderLimits {
    pub max_waves: usize,
    pub max_rows_per_wave: usize,
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
    observations: Vec<ActualWaveObservation>,
    next_wave_ordinal: u64,
    retained_rows: usize,
    lost_observations: u64,
    invalid_observation: bool,
    unknown_evidence: Option<ActualWaveEvidenceUnknown>,
}

impl BoundedWaveRecorder {
    /// Allocate the bounded wave slots before execution, not during callbacks.
    pub fn new(call_id: NonZeroU64, limits: CostRecorderLimits) -> Result<Self, CostRecorderError> {
        limits.validate()?;
        let mut observations = Vec::new();
        observations
            .try_reserve_exact(limits.max_waves)
            .map_err(|_| CostRecorderError::WaveCapacity)?;
        Ok(Self {
            fence: Arc::new(()),
            call_id,
            limits,
            observations,
            next_wave_ordinal: 0,
            retained_rows: 0,
            lost_observations: 0,
            invalid_observation: false,
            unknown_evidence: None,
        })
    }

    pub fn begin(
        &mut self,
        shape: ActualWaveShape,
        boundary: WaveObservationBoundary,
        prepare_started_at_ns: u64,
    ) -> Result<WaveObservationHandle, CostRecorderError> {
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
        if structured_rows > self.limits.max_rows_per_wave {
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
