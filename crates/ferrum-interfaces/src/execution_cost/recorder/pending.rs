use super::*;

impl BoundedWaveRecorder {
    /// Transfer call-owned evidence into the CPU consumer without allocation or
    /// keeping a second copy. The source recorder becomes an empty tombstone.
    pub fn take_frozen(&mut self) -> Self {
        Self {
            fence: Arc::clone(&self.fence),
            call_id: self.call_id,
            limits: self.limits,
            byte_limits: self.byte_limits,
            pending_retained_bytes: std::mem::take(&mut self.pending_retained_bytes),
            pending_working_bytes: std::mem::take(&mut self.pending_working_bytes),
            pending_retained_rows: std::mem::take(&mut self.pending_retained_rows),
            memory_audit: self.memory_audit,
            observations: std::mem::take(&mut self.observations),
            pending: std::mem::take(&mut self.pending),
            next_wave_ordinal: std::mem::replace(&mut self.next_wave_ordinal, u64::MAX),
            retained_rows: std::mem::take(&mut self.retained_rows),
            prepared_route: self.prepared_route.take(),
            prepared_route_attempts: self.prepared_route_attempts,
            route_unknown: self.route_unknown,
            route_diagnostic: self.route_diagnostic.take(),
            lost_observations: self.lost_observations,
            invalid_observation: self.invalid_observation,
            unknown_evidence: self.unknown_evidence,
            no_submission: self.no_submission.take(),
        }
    }
    /// Preserve the real wave lifecycle before its CPU projection is resolved.
    /// Pending is neither Known nor an invented zero-work observation.
    pub fn begin_pending(
        &mut self,
        pending: PendingActualWave,
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
        if self.observations.len() >= self.limits.max_waves {
            return self.reject_begin(CostRecorderError::WaveCapacity);
        }
        let bounds = pending.bounds();
        let retained = self.retained_rows.checked_add(bounds.retained_rows);
        let raw = self
            .pending_retained_bytes
            .checked_add(bounds.retained_bytes);
        let working = self
            .pending_working_bytes
            .checked_add(bounds.maximum_resolved_bytes);
        self.memory_audit.observe(retained, raw, working);
        let Some(retained) = retained
            .filter(|n| *n <= self.limits.max_retained_rows)
            .filter(|_| pending.rows().len() <= self.limits.max_rows_per_wave)
            .filter(|_| bounds.retained_rows <= self.limits.max_rows_per_wave)
        else {
            self.note_memory_rejection(
                CostRecorderCapacityResource::Rows,
                retained,
                self.limits.max_retained_rows,
            );
            return self.reject_begin(CostRecorderError::RowCapacity);
        };
        let Some(raw) = raw.filter(|n| *n <= self.byte_limits.maximum_retained_bytes) else {
            self.note_memory_rejection(
                CostRecorderCapacityResource::RawBytes,
                raw,
                self.byte_limits.maximum_retained_bytes,
            );
            return self.reject_begin(CostRecorderError::RowCapacity);
        };
        let Some(working) = working.filter(|n| *n <= self.byte_limits.maximum_working_bytes) else {
            self.note_memory_rejection(
                CostRecorderCapacityResource::WorkingBytes,
                working,
                self.byte_limits.maximum_working_bytes,
            );
            return self.reject_begin(CostRecorderError::RowCapacity);
        };
        let index = self.observations.len();
        self.observations.push(ActualWaveObservation {
            call_id: self.call_id,
            physical_wave_ordinal,
            shape: None,
            shape_unknown: None,
            boundary,
            prepare_started_at_ns,
            submission_started_at_ns: None,
            terminal_at_ns: None,
            host_committed_at_ns: None,
            device_elapsed_ns: None,
            outcome: None,
        });
        self.pending.push(Some(pending));
        self.retained_rows = retained;
        self.pending_retained_rows += bounds.retained_rows;
        self.pending_retained_bytes = raw;
        self.pending_working_bytes = working;
        Ok(WaveObservationHandle {
            fence: Arc::clone(&self.fence),
            index,
        })
    }

    pub fn has_pending_projection(&self) -> bool {
        self.pending.iter().any(Option::is_some)
    }

    /// Called only by the evidence consumer; retains original lifecycle clocks.
    pub fn resolve_pending(&mut self) -> Result<(), ActualWaveEvidenceUnknown> {
        let mut failure = None;
        for index in 0..self.pending.len() {
            let Some(pending) = self.pending[index].take() else {
                continue;
            };
            match pending.resolve() {
                Ok(shape) => self.observations[index].shape = Some(shape),
                Err(reason) => {
                    self.observations[index].shape_unknown = Some(reason);
                    self.note_unknown_evidence(reason);
                    failure.get_or_insert(reason);
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }

    /// Already checked during capture; includes reserved expansion capacity.
    pub fn retained_payload_bytes_upper_bound(&self) -> Option<usize> {
        let row_bytes = std::mem::size_of::<CostRowNumericFeatures>()
            .max(std::mem::size_of::<ActualWaveRow>())
            .max(std::mem::size_of::<HostRowStaticCostFeaturesV2>())
            .max(std::mem::size_of::<StructuredHostRowV1>());
        self.retained_rows
            .checked_sub(self.pending_retained_rows)?
            .checked_mul(row_bytes)?
            .checked_add(self.pending_retained_bytes)?
            .checked_add(
                self.observations
                    .capacity()
                    .checked_mul(std::mem::size_of::<ActualWaveObservation>())?,
            )?
            .checked_add(
                self.pending
                    .capacity()
                    .checked_mul(std::mem::size_of::<Option<PendingActualWave>>())?,
            )
    }
}
