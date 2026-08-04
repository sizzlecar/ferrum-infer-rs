use super::{
    invalid_resource, Arc, BTreeMap, BTreeSet, BackingInitializationCell,
    BackingInitializationStatus, BatchParticipantAuthority, DeviceCommandBatch, DeviceRuntime,
    LogicalBackingSliceAuthority, StateInitialization, StepResourceLease, VNextError,
};

pub(crate) enum BackingInitializationEncodeError<E> {
    Contract(VNextError),
    Runtime {
        participant: BatchParticipantAuthority,
        error: E,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedBackingInitializationPhase {
    Prepared,
    InFlight,
    Terminal,
}

struct PreparedBackingInitializationClaim {
    participant: BatchParticipantAuthority,
    cell: Arc<BackingInitializationCell>,
    slices: Vec<LogicalBackingSliceAuthority>,
}

pub(super) struct PreparedBackingInitializations {
    wave_fingerprint: Option<String>,
    claims: Vec<PreparedBackingInitializationClaim>,
    phase: PreparedBackingInitializationPhase,
}

impl PreparedBackingInitializations {
    pub(super) fn prepare<R>(
        step: &StepResourceLease<R>,
        wave_fingerprint: &str,
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime,
    {
        let mut grouped = BTreeMap::<
            String,
            (
                BatchParticipantAuthority,
                Arc<BackingInitializationCell>,
                Vec<LogicalBackingSliceAuthority>,
            ),
        >::new();
        for participant in &step.participants {
            let owner = BatchParticipantAuthority::new(
                participant.session.sequence_authority(),
                participant.session.request_authority(),
            );
            for authority in
                participant
                    .backing_snapshot
                    .backing_slices()
                    .iter()
                    .filter(|authority| {
                        authority.evidence().initialization() == StateInitialization::Zero
                    })
            {
                let cell = authority.initialization_cell().ok_or_else(|| {
                    invalid_resource(
                        "zero-initialized backing slice has no initialization authority",
                    )
                })?;
                match cell.status()? {
                    BackingInitializationStatus::Initialized => continue,
                    BackingInitializationStatus::Poisoned => {
                        return Err(invalid_resource(
                            "backing initialization authority is fail-closed",
                        ));
                    }
                    BackingInitializationStatus::Pending
                    | BackingInitializationStatus::Prepared
                    | BackingInitializationStatus::InFlight => {}
                }
                let entry = grouped
                    .entry(cell.target_fingerprint().to_owned())
                    .or_insert_with(|| (owner, Arc::clone(cell), Vec::new()));
                if !Arc::ptr_eq(&entry.1, cell) {
                    return Err(invalid_resource(
                        "distinct backing initialization authorities share a target fingerprint",
                    ));
                }
                if !entry
                    .2
                    .iter()
                    .any(|existing| existing.evidence() == authority.evidence())
                {
                    entry.2.push(authority.retained());
                }
            }
        }

        let mut prepared = Self {
            wave_fingerprint: None,
            claims: Vec::new(),
            phase: PreparedBackingInitializationPhase::Prepared,
        };
        for (_, (participant, cell, mut slices)) in grouped {
            slices.sort_by(|left, right| {
                left.resource_id().cmp(right.resource_id()).then_with(|| {
                    left.evidence()
                        .physical_offset_bytes()
                        .cmp(&right.evidence().physical_offset_bytes())
                })
            });
            if cell.prepare(wave_fingerprint)? {
                prepared.claims.push(PreparedBackingInitializationClaim {
                    participant,
                    cell,
                    slices,
                });
            }
        }
        if !prepared.claims.is_empty() {
            prepared.wave_fingerprint = Some(wave_fingerprint.to_owned());
        }
        Ok(prepared)
    }

    pub(super) fn ensure_wave(&self, wave_fingerprint: &str) -> Result<(), VNextError> {
        if self
            .wave_fingerprint
            .as_deref()
            .is_some_and(|current| current != wave_fingerprint)
        {
            return Err(invalid_resource(
                "backing initialization permit belongs to another submission wave",
            ));
        }
        Ok(())
    }

    pub(super) fn encode<R>(
        &self,
        step: &StepResourceLease<R>,
        runtime: &R,
        commands: &mut DeviceCommandBatch<R::Command>,
    ) -> Result<usize, BackingInitializationEncodeError<R::Error>>
    where
        R: DeviceRuntime,
    {
        if self.phase != PreparedBackingInitializationPhase::Prepared {
            return Err(BackingInitializationEncodeError::Contract(
                invalid_resource("backing initialization is not prepared for encoding"),
            ));
        }
        let pools = step.participants[0]
            .session
            .resources()
            .request
            .plan
            .dynamic_pools();
        let mut command_count = 0_usize;
        for claim in &self.claims {
            let mut encoded_ranges = BTreeSet::new();
            for authority in &claim.slices {
                if authority.evidence().initialization() != StateInitialization::Zero
                    || authority
                        .initialization_cell()
                        .is_none_or(|cell| !Arc::ptr_eq(cell, &claim.cell))
                {
                    return Err(BackingInitializationEncodeError::Contract(
                        invalid_resource(
                            "backing initialization target differs from its prepared authority",
                        ),
                    ));
                }
                let view = pools
                    .view(authority)
                    .map_err(BackingInitializationEncodeError::Contract)?;
                for binding in view.segment_bindings() {
                    let segment = binding.segment();
                    let range = (
                        segment.chunk_ordinal(),
                        segment.chunk_generation(),
                        segment.offset_bytes(),
                        segment.length_bytes(),
                    );
                    if !encoded_ranges.insert(range) {
                        continue;
                    }
                    let actual = runtime.buffer_descriptor(binding.buffer());
                    if &actual != binding.descriptor()
                        || segment
                            .offset_bytes()
                            .checked_add(segment.length_bytes())
                            .is_none_or(|end| end > actual.size_bytes)
                    {
                        return Err(BackingInitializationEncodeError::Contract(
                            invalid_resource("backing initialization buffer descriptor drifted"),
                        ));
                    }
                    let command = runtime
                        .encode_zero(
                            binding.buffer(),
                            segment.offset_bytes(),
                            segment.length_bytes(),
                        )
                        .map_err(|error| BackingInitializationEncodeError::Runtime {
                            participant: claim.participant,
                            error,
                        })?;
                    commands.push_initialization(command);
                    command_count = command_count.checked_add(1).ok_or_else(|| {
                        BackingInitializationEncodeError::Contract(invalid_resource(
                            "backing initialization command count overflows usize",
                        ))
                    })?;
                }
            }
        }
        Ok(command_count)
    }

    pub(super) fn mark_in_flight(&mut self) -> Result<(), VNextError> {
        if self.phase != PreparedBackingInitializationPhase::Prepared {
            return Err(invalid_resource(
                "backing initialization cannot install a second fence",
            ));
        }
        for claim in &self.claims {
            let wave_fingerprint = self
                .wave_fingerprint
                .as_deref()
                .expect("non-empty initialization claims own a wave fingerprint");
            if let Err(error) = claim.cell.mark_in_flight(wave_fingerprint) {
                self.mark_indeterminate();
                return Err(error);
            }
        }
        self.phase = PreparedBackingInitializationPhase::InFlight;
        Ok(())
    }

    pub(super) fn finish(&mut self, succeeded: bool) -> Result<(), VNextError> {
        if self.phase != PreparedBackingInitializationPhase::InFlight {
            self.mark_indeterminate();
            return Err(invalid_resource(
                "backing initialization reached terminal without an installed fence",
            ));
        }
        for claim in &self.claims {
            let wave_fingerprint = self
                .wave_fingerprint
                .as_deref()
                .expect("non-empty initialization claims own a wave fingerprint");
            if let Err(error) = claim.cell.finish(wave_fingerprint, succeeded) {
                self.mark_indeterminate();
                return Err(error);
            }
        }
        self.phase = PreparedBackingInitializationPhase::Terminal;
        Ok(())
    }

    pub(super) fn mark_indeterminate(&mut self) {
        for claim in &self.claims {
            claim.cell.mark_indeterminate();
        }
        self.phase = PreparedBackingInitializationPhase::Terminal;
    }
}

impl Drop for PreparedBackingInitializations {
    fn drop(&mut self) {
        match self.phase {
            PreparedBackingInitializationPhase::Prepared => {
                for claim in &self.claims {
                    claim.cell.rollback_prepared(
                        self.wave_fingerprint
                            .as_deref()
                            .expect("non-empty initialization claims own a wave fingerprint"),
                    );
                }
            }
            PreparedBackingInitializationPhase::InFlight => self.mark_indeterminate(),
            PreparedBackingInitializationPhase::Terminal => {}
        }
    }
}
