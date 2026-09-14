use super::{
    invalid_resource, Arc, BTreeMap, BTreeSet, BackingInitializationCell,
    BackingInitializationStatus, BatchParticipantAuthority, DeviceCommandBatch, DeviceRuntime,
    DynamicPoolSet, LogicalBackingSliceAuthority, StateInitialization, StepResourceLease,
    VNextError,
};
use crate::vnext::{
    AllocationKind, AllocationLifetime, BufferUsage, PreparedSequenceStateTransfer,
    SequenceBackingGeneration, SequenceCheckpointBytePlan, SequenceCheckpointLayout,
    SequenceSessionEpoch, SequenceSessionFingerprint,
};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_RESTORE_INITIALIZATION_OWNER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
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

struct RestoreInitializationTarget {
    slot: Arc<super::SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
    generation: SequenceBackingGeneration,
    reservation: NonZeroU64,
}

impl RestoreInitializationTarget {
    fn from_guard<R: DeviceRuntime>(guard: &PreparedSequenceStateTransfer<R>) -> Self {
        Self {
            slot: Arc::clone(&guard.session().slot),
            epoch: guard.session().epoch(),
            fingerprint: guard.session().fingerprint().clone(),
            generation: guard.backing().generation(),
            reservation: guard.reservation_serial(),
        }
    }

    fn ensure_guard<R: DeviceRuntime>(
        &self,
        guard: &PreparedSequenceStateTransfer<R>,
    ) -> Result<(), VNextError> {
        guard.ensure_fresh_restore_target()?;
        if !Arc::ptr_eq(&self.slot, &guard.session().slot)
            || self.epoch != guard.session().epoch()
            || self.fingerprint != *guard.session().fingerprint()
            || self.generation != guard.backing().generation()
            || self.reservation != guard.reservation_serial()
        {
            return Err(invalid_resource(
                "restore initialization belongs to another exact target reservation",
            ));
        }
        Ok(())
    }
}

pub(crate) struct PreparedBackingInitializations {
    wave_fingerprint: Option<String>,
    cell_fingerprint: Option<String>,
    claims: Vec<PreparedBackingInitializationClaim>,
    phase: PreparedBackingInitializationPhase,
    restore_target: Option<RestoreInitializationTarget>,
}

impl PreparedBackingInitializations {
    pub(super) fn prepare<R>(
        step: &StepResourceLease<R>,
        wave_fingerprint: &str,
    ) -> Result<Self, VNextError>
    where
        R: DeviceRuntime,
    {
        Self::prepare_slices(
            step.participants.iter().map(|participant| {
                (
                    BatchParticipantAuthority::new(
                        participant.session.sequence_authority(),
                        participant.session.request_authority(),
                    ),
                    participant.backing_snapshot.backing_slices(),
                )
            }),
            wave_fingerprint,
            None,
        )
    }

    /// Prepare the target's complete zero-initialization obligations, without
    /// constructing a model step. The caller must encode these zero commands
    /// before all checkpoint copies in the same ordered submission. A successful
    /// fence for the entire submission, not encoding alone, permits `finish`.
    pub(crate) fn prepare_restore<R: DeviceRuntime>(
        guard: &PreparedSequenceStateTransfer<R>,
        layout: &SequenceCheckpointLayout,
        byte_plan: &SequenceCheckpointBytePlan,
        transfer_fingerprint: &str,
    ) -> Result<Self, VNextError> {
        guard.ensure_fresh_restore_target()?;
        let plan = &guard.session().resources().request.plan;
        let _lifecycle = plan
            .resources
            .read_lifecycle("prepare restore initialization")?;
        if transfer_fingerprint.is_empty()
            || byte_plan.plan_hash() != plan.plan_hash()
            || layout.fingerprint()? != byte_plan.layout_fingerprint()
        {
            return Err(invalid_resource(
                "restore initialization does not match the trusted plan and complete byte layout",
            ));
        }
        let expected_resources = layout
            .states()
            .iter()
            .map(|state| state.resource_id())
            .collect::<BTreeSet<_>>();
        let actual_resources = guard
            .backing()
            .backing_slices()
            .iter()
            .map(LogicalBackingSliceAuthority::resource_id)
            .collect::<BTreeSet<_>>();
        if expected_resources != actual_resources {
            return Err(invalid_resource(
                "restore initialization must cover the entire target state resource closure",
            ));
        }
        let pools = plan.dynamic_pools();
        for state in layout.states() {
            let descriptor = pools
                .domains
                .iter()
                .flat_map(|domain| &domain.descriptors)
                .find(|descriptor| descriptor.base_resource_id() == state.resource_id())
                .ok_or_else(|| invalid_resource("restore state has no target descriptor"))?;
            if descriptor.lifetime() != AllocationLifetime::Sequence
                || descriptor.usage() != BufferUsage::State
                || descriptor.kind() != &AllocationKind::Value
                || descriptor.initialization() != state.initialization()
                || descriptor.storage() != state.storage()
                || descriptor.element_type() != state.tensor().element_type()
            {
                return Err(invalid_resource(
                    "restore state initialization or physical ABI differs from its target",
                ));
            }
        }
        for resource in byte_plan.resources() {
            let view =
                pools.view_many(guard.backing().backing_slices_for(resource.resource_id()))?;
            if resource
                .ranges()
                .iter()
                .any(|range| range.source().end > view.size_bytes())
            {
                return Err(invalid_resource(
                    "restore state copy exceeds the concrete target backing",
                ));
            }
        }
        Self::prepare_slices(
            std::iter::once((
                BatchParticipantAuthority::new(
                    guard.session().sequence_authority(),
                    guard.session().request_authority(),
                ),
                guard.backing().backing_slices(),
            )),
            transfer_fingerprint,
            Some(RestoreInitializationTarget::from_guard(guard)),
        )
    }

    fn prepare_slices<'a>(
        participants: impl IntoIterator<
            Item = (
                BatchParticipantAuthority,
                &'a [LogicalBackingSliceAuthority],
            ),
        >,
        wave_fingerprint: &str,
        restore_target: Option<RestoreInitializationTarget>,
    ) -> Result<Self, VNextError> {
        let mut grouped = BTreeMap::<
            String,
            (
                BatchParticipantAuthority,
                Arc<BackingInitializationCell>,
                Vec<LogicalBackingSliceAuthority>,
            ),
        >::new();
        for (owner, slices) in participants {
            for authority in slices.iter().filter(|authority| {
                authority.evidence().initialization() == StateInitialization::Zero
            }) {
                let cell = authority.initialization_cell().ok_or_else(|| {
                    invalid_resource(
                        "zero-initialized backing slice has no initialization authority",
                    )
                })?;
                if restore_target.is_some()
                    && cell.status()? != BackingInitializationStatus::Pending
                {
                    return Err(invalid_resource(
                        "fresh restore requires pending target initialization cells",
                    ));
                }
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
                if !Arc::ptr_eq(&entry.1, cell) || entry.0 != owner {
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

        if restore_target.is_some() {
            for (_, (_, _, slices)) in &grouped {
                validate_complete_cell_capacity(slices)?;
            }
        }
        // A second concurrent preparation of the same transfer may observe
        // Pending before the first claim lands. Give each preparation its own
        // cell owner so `prepare` cannot mistake it for a same-wave retry.
        let cell_fingerprint = if restore_target.is_some() {
            let serial = NEXT_RESTORE_INITIALIZATION_OWNER
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |serial| {
                    serial.checked_add(1)
                })
                .map_err(|_| invalid_resource("restore initialization identities are exhausted"))?;
            format!("{wave_fingerprint}/restore-initialization/{serial}")
        } else {
            wave_fingerprint.to_owned()
        };

        let mut prepared = Self {
            wave_fingerprint: None,
            cell_fingerprint: None,
            claims: Vec::new(),
            phase: PreparedBackingInitializationPhase::Prepared,
            restore_target,
        };
        for (_, (participant, cell, mut slices)) in grouped {
            slices.sort_by(|left, right| {
                left.resource_id().cmp(right.resource_id()).then_with(|| {
                    left.evidence()
                        .physical_offset_bytes()
                        .cmp(&right.evidence().physical_offset_bytes())
                })
            });
            if cell.prepare(&cell_fingerprint)? {
                // Install ownership before another preparation can fail, so
                // Drop always rolls back every cell already claimed here.
                prepared.wave_fingerprint = Some(wave_fingerprint.to_owned());
                prepared.cell_fingerprint = Some(cell_fingerprint.clone());
                prepared.claims.push(PreparedBackingInitializationClaim {
                    participant,
                    cell,
                    slices,
                });
            }
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
        if self.restore_target.is_some() {
            return Err(BackingInitializationEncodeError::Contract(
                invalid_resource("restore initialization cannot be encoded as a model step"),
            ));
        }
        let pools = step.participants[0]
            .session
            .resources()
            .request
            .plan
            .dynamic_pools();
        self.encode_in_pools(pools, runtime, commands)
    }

    pub(crate) fn encode_restore<R: DeviceRuntime>(
        &self,
        guard: &PreparedSequenceStateTransfer<R>,
        runtime: &R,
        commands: &mut DeviceCommandBatch<R::Command>,
    ) -> Result<usize, BackingInitializationEncodeError<R::Error>> {
        let target = self.restore_target.as_ref().ok_or_else(|| {
            BackingInitializationEncodeError::Contract(invalid_resource(
                "model initialization cannot be encoded as a restore",
            ))
        })?;
        target
            .ensure_guard(guard)
            .map_err(BackingInitializationEncodeError::Contract)?;
        if !std::ptr::eq(runtime, guard.runtime_arc().as_ref()) {
            return Err(BackingInitializationEncodeError::Contract(
                invalid_resource(
                    "restore initialization must use the target's exact runtime instance",
                ),
            ));
        }
        self.encode_in_pools(
            guard.session().resources().request.plan.dynamic_pools(),
            runtime,
            commands,
        )
    }

    fn encode_in_pools<R: DeviceRuntime>(
        &self,
        pools: &DynamicPoolSet<R>,
        runtime: &R,
        commands: &mut DeviceCommandBatch<R::Command>,
    ) -> Result<usize, BackingInitializationEncodeError<R::Error>> {
        if self.phase != PreparedBackingInitializationPhase::Prepared {
            return Err(BackingInitializationEncodeError::Contract(
                invalid_resource("backing initialization is not prepared for encoding"),
            ));
        }
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

    pub(crate) fn mark_in_flight(&mut self) -> Result<(), VNextError> {
        if self.phase != PreparedBackingInitializationPhase::Prepared {
            return Err(invalid_resource(
                "backing initialization cannot install a second fence",
            ));
        }
        for claim in &self.claims {
            let wave_fingerprint = self
                .cell_fingerprint
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

    pub(crate) fn finish(&mut self, succeeded: bool) -> Result<(), VNextError> {
        if self.phase != PreparedBackingInitializationPhase::InFlight {
            self.mark_indeterminate();
            return Err(invalid_resource(
                "backing initialization reached terminal without an installed fence",
            ));
        }
        for claim in &self.claims {
            let wave_fingerprint = self
                .cell_fingerprint
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

    pub(crate) fn mark_indeterminate(&mut self) {
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
                        self.cell_fingerprint
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

/// A cell is indivisible: completing one projection must not suppress the
/// first-use zeroing of an uninitialized part of the same physical claim.
/// Initialization includes allocator slack inside these owned projections;
/// it never copies slack from a checkpoint or touches another owner's extent.
fn validate_complete_cell_capacity(
    slices: &[LogicalBackingSliceAuthority],
) -> Result<(), VNextError> {
    let first = slices
        .first()
        .ok_or_else(|| invalid_resource("restore initialization cell has no projections"))?;
    let mut ranges = Vec::with_capacity(slices.len());
    for slice in slices {
        if !Arc::ptr_eq(&slice.segment_lease, &first.segment_lease) {
            return Err(invalid_resource(
                "restore initialization cell spans different physical claims",
            ));
        }
        let start = slice.evidence().physical_offset_bytes();
        let end = start
            .checked_add(slice.capacity_size_bytes())
            .ok_or_else(|| invalid_resource("restore initialization projection overflows"))?;
        if end > first.segment_lease.size_bytes {
            return Err(invalid_resource(
                "restore initialization projection exceeds its claim",
            ));
        }
        ranges.push(start..end);
    }
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut covered = 0;
    for range in ranges {
        if range.start > covered {
            return Err(invalid_resource(
                "restore initialization has an uncovered physical claim range",
            ));
        }
        covered = covered.max(range.end);
    }
    if covered != first.segment_lease.size_bytes {
        return Err(invalid_resource(
            "restore initialization does not cover its complete physical claim",
        ));
    }
    Ok(())
}
