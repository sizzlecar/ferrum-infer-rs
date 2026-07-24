use super::{
    fmt, invalid_resource, ActiveSequenceFrame, AdmissionDeferred, AdmissionDemand,
    AdmissionRejected, AdmittedSequenceResources, Arc, BTreeMap, BatchInvocationId,
    BatchParticipantAuthority, BatchParticipantTokenSpan, BatchStepId, BatchWorkShape,
    CapacityDomainId, CapacityEntry, CapacityUnits, CapacityVector, DeviceRuntime, Digest,
    DynamicBackingDeferred, DynamicDeferredMaintenanceOutcome, ExecutionFrameId, ExecutionLane,
    LaneStableArenaSlotIdentity, LaneStableArenaSlotLease, LogicalBackingSliceAuthority,
    LogicalBatchCapacityLease, Mutex, NodeId, ParticipantFlightPhase, ParticipantNodeKey,
    PhysicalBackingClaimIdentity, PlanBackingDeferral, PlanCapacityWaitRegistration, PlanHash,
    ProgramBindingExecutionBinding, ProgramBindingLayout, ProgramBindingNodeBinding,
    RequestAuthorityId, SequenceAuthorityId, SequenceBackingSnapshot, SequenceSession,
    SequenceSessionEpoch, SequenceSessionFingerprint, SequenceSessionPhase, SequenceSessionSlot,
    SequenceSessionSlotState, Serialize, Sha256, StepParticipantFrameAssignment, TokenSpanWork,
    TrustedPlanRuntimeEvidence, VNextError,
};
use crate::vnext::DeviceReusableExecutionProgramId;
use crate::vnext::{ReusableExecutionBucketId, ReusableExecutionBucketSpec};

/// Resources whose lifetime is one exact continuous-batch execution frame.
/// Child invocation leases retain this scope through `Arc`, so shared frame
/// capacity and every participant authority outlive asynchronous device work.
#[must_use = "step resources must live through every child invocation"]
pub struct StepResourceLease<R>
where
    R: DeviceRuntime,
{
    // The transaction releases physical extents before its logical claim,
    // then per-sequence frame guards release before their parent sessions.
    pub(super) claimed_backing: ClaimedBackingTransaction,
    pub(super) participants: Vec<AdmittedStepParticipant<R>>,
    pub(super) invocation_registry: Arc<InvocationRegistry>,
    pub(super) execution_lane: Arc<ExecutionLane<R>>,
    pub(super) reusable_execution_bucket: Option<ReusableExecutionBucketSpec>,
    pub(super) batch_step_id: BatchStepId,
    pub(super) finalized: bool,
}

/// Canonical non-empty set selected by the scheduler for one continuous
/// batch. Membership is exact; capacity shapes may not claim a different
/// sequence count and no global concurrency ceiling is embedded here.
#[must_use = "a batch participant set is required to admit one execution frame"]
pub struct ExecutionBatchParticipants<R>
where
    R: DeviceRuntime,
{
    pub(super) sessions: Vec<Arc<SequenceSession<R>>>,
    plan_evidence: TrustedPlanRuntimeEvidence,
}

fn sequence_participant_key<R: DeviceRuntime>(
    sequence: &AdmittedSequenceResources<R>,
) -> (u32, u64, u32, u64) {
    let sequence_authority = sequence.sequence_authority();
    let request_authority = sequence.request_authority();
    (
        sequence_authority.sparse_id(),
        sequence_authority.generation(),
        request_authority.sparse_id(),
        request_authority.generation(),
    )
}

pub(super) fn session_participant_key<R: DeviceRuntime>(
    session: &SequenceSession<R>,
) -> (u32, u64, u32, u64) {
    sequence_participant_key(session.resources())
}

impl<R> ExecutionBatchParticipants<R>
where
    R: DeviceRuntime,
{
    pub fn new(mut sessions: Vec<Arc<SequenceSession<R>>>) -> Result<Self, VNextError> {
        sessions.sort_by_key(|session| session_participant_key(session));
        if sessions.is_empty()
            || sessions
                .windows(2)
                .any(|pair| session_participant_key(&pair[0]) == session_participant_key(&pair[1]))
        {
            return Err(invalid_resource(
                "execution batch participants must be non-empty and unique",
            ));
        }
        u32::try_from(sessions.len())
            .map_err(|_| invalid_resource("execution batch participant count exceeds u32"))?;
        let plan_evidence = sessions[0].resources().plan_evidence();
        if sessions.iter().any(|session| {
            session.resources().is_poisoned()
                || session.resources().plan_evidence() != plan_evidence
        }) {
            return Err(invalid_resource(
                "execution batch participants differ in plan, runtime, pool, coordinator, or health",
            ));
        }
        let resources = Arc::clone(&sessions[0].resources().request.plan.resources);
        if sessions
            .iter()
            .any(|session| !Arc::ptr_eq(&resources, &session.resources().request.plan.resources))
        {
            return Err(invalid_resource(
                "execution batch participants belong to distinct plan runtime roots",
            ));
        }
        let _lifecycle = resources.read_lifecycle("create execution batch participants")?;
        Ok(Self {
            sessions,
            plan_evidence,
        })
    }

    pub fn len(&self) -> u32 {
        u32::try_from(self.sessions.len())
            .expect("execution batch participant count is validated at construction")
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn sessions(&self) -> &[Arc<SequenceSession<R>>] {
        &self.sessions
    }

    pub fn plan_evidence(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan_evidence
    }

    pub fn bind_work_shape(
        &self,
        token_spans: Vec<TokenSpanWork>,
    ) -> Result<BatchWorkShape, VNextError> {
        if token_spans.len() != self.sessions.len() {
            return Err(invalid_resource(
                "batch token work count differs from its exact participant set",
            ));
        }
        BatchWorkShape::new(
            self.sessions
                .iter()
                .zip(token_spans)
                .map(|(session, token_span)| {
                    BatchParticipantTokenSpan::new(
                        BatchParticipantAuthority::new(
                            session.sequence_authority(),
                            session.request_authority(),
                        ),
                        token_span,
                    )
                })
                .collect(),
        )
    }
}

#[derive(Clone)]
pub(super) struct SequenceFrameCandidate {
    pub(super) slot: Arc<SequenceSessionSlot>,
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
}

pub(super) struct SequenceFrameCaptureCandidate<R>
where
    R: DeviceRuntime,
{
    frame: SequenceFrameCandidate,
    resources: Arc<AdmittedSequenceResources<R>>,
}

pub(super) struct SessionFrameHold {
    pub(super) slot: Arc<SequenceSessionSlot>,
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
    pub(super) frame_id: ExecutionFrameId,
    pub(super) batch_step_id: BatchStepId,
    pub(super) finalized: bool,
}

pub(super) struct CapturedSessionFrame<R>
where
    R: DeviceRuntime,
{
    pub(super) hold: SessionFrameHold,
    pub(super) backing_snapshot: Arc<SequenceBackingSnapshot<R>>,
}

#[derive(Clone)]
pub(super) struct ParticipantFlightCandidate {
    pub(super) slot: Arc<SequenceSessionSlot>,
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
    pub(super) frame: ActiveSequenceFrame,
    pub(super) participant: BatchParticipantAuthority,
}

#[derive(Clone)]
struct ParticipantNodeFlightCandidate {
    candidate: ParticipantFlightCandidate,
    node_id: NodeId,
}

impl ParticipantNodeFlightCandidate {
    fn new(candidate: ParticipantFlightCandidate, node_id: NodeId) -> Self {
        Self { candidate, node_id }
    }

    fn key(&self) -> ParticipantNodeKey {
        ParticipantNodeKey::new(
            self.candidate.participant,
            self.candidate.frame.frame_id,
            self.node_id.clone(),
        )
    }
}

/// One participant-local flight owned by an exact invocation. Dropping this
/// hold removes the sequence-flight count, while the physical ledger guard
/// independently retires its topology. Only the sealed device DNF path may
/// reset an in-flight hold to Prepared for retry.
pub(super) struct PreparedParticipantFlightHold {
    slot: Arc<SequenceSessionSlot>,
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
    key: ParticipantNodeKey,
    batch_step_id: BatchStepId,
    pub(super) phase: ParticipantFlightPhase,
}

impl Drop for PreparedParticipantFlightHold {
    fn drop(&mut self) {
        let mut state = match self.slot.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                *state = SequenceSessionSlotState::FailClosed;
                return;
            }
        };
        match &mut *state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                if active.participant_flights.remove(&self.key) != Some(self.phase) {
                    active.phase = SequenceSessionPhase::Poisoned;
                }
            }
            _ => *state = SequenceSessionSlotState::FailClosed,
        }
    }
}

pub(super) fn prepare_participant_flights(
    candidates: &[ParticipantFlightCandidate],
    node_id: &NodeId,
) -> Result<Vec<PreparedParticipantFlightHold>, VNextError> {
    prepare_participant_node_flights(
        candidates
            .iter()
            .cloned()
            .map(|candidate| ParticipantNodeFlightCandidate::new(candidate, node_id.clone()))
            .collect(),
    )
}

fn prepare_participant_node_flights(
    candidates: Vec<ParticipantNodeFlightCandidate>,
) -> Result<Vec<PreparedParticipantFlightHold>, VNextError> {
    let mut candidates = candidates
        .into_iter()
        .map(|candidate| {
            let key = candidate.key();
            (candidate.candidate, key)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.1.cmp(&right.1));
    if candidates.is_empty() || candidates.windows(2).any(|pair| pair[0].1 >= pair[1].1) {
        return Err(invalid_resource(
            "prepared submission wave requires canonical non-empty unique participant-node keys",
        ));
    }

    type ParticipantFrameKey = (u32, u64, u32, u64, u64);
    let participant_frame_key = |key: &ParticipantNodeKey| -> ParticipantFrameKey {
        (
            key.sequence_authority().sparse_id(),
            key.sequence_authority().generation(),
            key.request_authority().sparse_id(),
            key.request_authority().generation(),
            key.frame_id().get(),
        )
    };
    let mut slot_by_participant = BTreeMap::<ParticipantFrameKey, usize>::new();
    let mut participant_by_slot = BTreeMap::<usize, ParticipantFrameKey>::new();
    let mut unique_slots = Vec::<Arc<SequenceSessionSlot>>::new();
    let mut slot_indices = Vec::with_capacity(candidates.len());
    for (candidate, key) in &candidates {
        let participant = participant_frame_key(key);
        let slot_identity = Arc::as_ptr(&candidate.slot) as usize;
        if slot_by_participant
            .get(&participant)
            .is_some_and(|known| *known != slot_identity)
            || participant_by_slot
                .get(&slot_identity)
                .is_some_and(|known| *known != participant)
        {
            return Err(invalid_resource(
                "submission wave participant/frame authority maps to inconsistent session slots",
            ));
        }
        slot_by_participant.insert(participant, slot_identity);
        participant_by_slot.insert(slot_identity, participant);
        let slot_index = if let Some(index) = unique_slots
            .iter()
            .position(|slot| Arc::ptr_eq(slot, &candidate.slot))
        {
            index
        } else {
            unique_slots.push(Arc::clone(&candidate.slot));
            unique_slots.len() - 1
        };
        slot_indices.push(slot_index);
    }

    let mut states = Vec::with_capacity(unique_slots.len());
    for slot in &unique_slots {
        states.push(
            slot.state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    for ((candidate, key), &slot_index) in candidates.iter().zip(&slot_indices) {
        match &*states[slot_index] {
            SequenceSessionSlotState::Active(active)
                if active.epoch == candidate.epoch
                    && active.fingerprint == candidate.fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame == Some(candidate.frame)
                    && active.submission_wave_flight.is_none()
                    && !active.participant_flights.contains_key(key) => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != candidate.epoch
                    || active.fingerprint != candidate.fingerprint =>
            {
                return Err(invalid_resource(
                    "stale prepared submission wave participant authority",
                ));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "cancelled, poisoned, duplicate, or cross-frame participant cannot enter a submission wave",
                ));
            }
            _ => {
                return Err(invalid_resource(
                    "inactive or terminal participant cannot enter a submission wave",
                ));
            }
        }
    }

    let mut inserted = Vec::<(usize, ParticipantNodeKey)>::new();
    for ((_, key), &slot_index) in candidates.iter().zip(&slot_indices) {
        let previous = {
            let SequenceSessionSlotState::Active(active) = &mut *states[slot_index] else {
                unreachable!("all prepared wave participants were validated");
            };
            active
                .participant_flights
                .insert(key.clone(), ParticipantFlightPhase::Prepared)
        };
        if let Some(previous) = previous {
            for (rollback_slot, rollback_key) in inserted.into_iter().rev() {
                if let SequenceSessionSlotState::Active(rollback) = &mut *states[rollback_slot] {
                    rollback.participant_flights.remove(&rollback_key);
                    rollback.phase = SequenceSessionPhase::Poisoned;
                }
            }
            if let SequenceSessionSlotState::Active(active) = &mut *states[slot_index] {
                active.participant_flights.insert(key.clone(), previous);
                active.phase = SequenceSessionPhase::Poisoned;
            }
            return Err(invalid_resource(
                "prepared participant wave changed during atomic insertion",
            ));
        }
        inserted.push((slot_index, key.clone()));
    }
    drop(states);

    Ok(candidates
        .into_iter()
        .map(|(candidate, key)| PreparedParticipantFlightHold {
            slot: candidate.slot,
            epoch: candidate.epoch,
            fingerprint: candidate.fingerprint,
            key,
            batch_step_id: candidate.frame.batch_step_id,
            phase: ParticipantFlightPhase::Prepared,
        })
        .collect())
}

/// One participant-local owner for a physical all-node submission wave. Node
/// coverage remains in the immutable plan topology and invocation registry;
/// the sequence session only tracks whether this participant has device work.
pub(super) struct PreparedSubmissionWaveParticipantFlightHold {
    slot: Arc<SequenceSessionSlot>,
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
    participant: BatchParticipantAuthority,
    frame: ActiveSequenceFrame,
    pub(super) phase: ParticipantFlightPhase,
}

impl PreparedSubmissionWaveParticipantFlightHold {
    fn canonical_key(&self) -> (u32, u64, u32, u64, u64) {
        let (sequence, sequence_generation, request, request_generation) =
            self.participant.canonical_key();
        (
            sequence,
            sequence_generation,
            request,
            request_generation,
            self.frame.frame_id.get(),
        )
    }
}

impl Drop for PreparedSubmissionWaveParticipantFlightHold {
    fn drop(&mut self) {
        let mut state = match self.slot.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                *state = SequenceSessionSlotState::FailClosed;
                return;
            }
        };
        match &mut *state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                if active.submission_wave_flight.take() != Some(self.phase) {
                    active.phase = SequenceSessionPhase::Poisoned;
                }
            }
            _ => *state = SequenceSessionSlotState::FailClosed,
        }
    }
}

pub(super) fn prepare_submission_wave_participant_flights(
    candidates: &[ParticipantFlightCandidate],
) -> Result<Vec<PreparedSubmissionWaveParticipantFlightHold>, VNextError> {
    type ParticipantFrameKey = (u32, u64, u32, u64, u64);

    let mut candidates = candidates
        .iter()
        .cloned()
        .map(|candidate| {
            let (sequence, sequence_generation, request, request_generation) =
                candidate.participant.canonical_key();
            let key = (
                sequence,
                sequence_generation,
                request,
                request_generation,
                candidate.frame.frame_id.get(),
            );
            (candidate, key)
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|(_, key)| *key);
    if candidates.is_empty() || candidates.windows(2).any(|pair| pair[0].1 >= pair[1].1) {
        return Err(invalid_resource(
            "submission wave requires canonical non-empty unique participant/frame authorities",
        ));
    }

    let mut slot_by_participant = BTreeMap::<ParticipantFrameKey, usize>::new();
    let mut participant_by_slot = BTreeMap::<usize, ParticipantFrameKey>::new();
    let mut unique_slots = Vec::<Arc<SequenceSessionSlot>>::new();
    let mut unique_slot_indices = BTreeMap::<usize, usize>::new();
    let mut slot_indices = Vec::with_capacity(candidates.len());
    for (candidate, participant) in &candidates {
        let slot_identity = Arc::as_ptr(&candidate.slot) as usize;
        if slot_by_participant
            .get(participant)
            .is_some_and(|known| *known != slot_identity)
            || participant_by_slot
                .get(&slot_identity)
                .is_some_and(|known| known != participant)
        {
            return Err(invalid_resource(
                "submission wave participant/frame authority maps to inconsistent session slots",
            ));
        }
        slot_by_participant.insert(*participant, slot_identity);
        participant_by_slot.insert(slot_identity, *participant);
        let slot_index = match unique_slot_indices.get(&slot_identity).copied() {
            Some(index) => index,
            None => {
                let index = unique_slots.len();
                unique_slots.push(Arc::clone(&candidate.slot));
                unique_slot_indices.insert(slot_identity, index);
                index
            }
        };
        slot_indices.push(slot_index);
    }

    let mut states = Vec::with_capacity(unique_slots.len());
    for slot in &unique_slots {
        states.push(
            slot.state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    for ((candidate, _), &slot_index) in candidates.iter().zip(&slot_indices) {
        match &*states[slot_index] {
            SequenceSessionSlotState::Active(active)
                if active.epoch == candidate.epoch
                    && active.fingerprint == candidate.fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame == Some(candidate.frame)
                    && active.participant_flights.is_empty()
                    && active.submission_wave_flight.is_none() => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != candidate.epoch
                    || active.fingerprint != candidate.fingerprint =>
            {
                return Err(invalid_resource(
                    "stale submission wave participant authority",
                ));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "cancelled, poisoned, duplicate, cross-frame, or node-busy participant cannot enter a submission wave",
                ));
            }
            _ => {
                return Err(invalid_resource(
                    "inactive or terminal participant cannot enter a submission wave",
                ));
            }
        }
    }

    let mut inserted_slots = Vec::<usize>::with_capacity(candidates.len());
    for &slot_index in &slot_indices {
        let previous = {
            let SequenceSessionSlotState::Active(active) = &mut *states[slot_index] else {
                unreachable!("all submission wave participants were validated");
            };
            active
                .submission_wave_flight
                .replace(ParticipantFlightPhase::Prepared)
        };
        if previous.is_some() {
            for rollback_slot in inserted_slots.into_iter().rev() {
                if let SequenceSessionSlotState::Active(rollback) = &mut *states[rollback_slot] {
                    rollback.submission_wave_flight = None;
                    rollback.phase = SequenceSessionPhase::Poisoned;
                }
            }
            if let SequenceSessionSlotState::Active(active) = &mut *states[slot_index] {
                active.submission_wave_flight = previous;
                active.phase = SequenceSessionPhase::Poisoned;
            }
            return Err(invalid_resource(
                "submission wave participant flights changed during atomic insertion",
            ));
        }
        inserted_slots.push(slot_index);
    }
    drop(states);

    Ok(candidates
        .into_iter()
        .map(
            |(candidate, _)| PreparedSubmissionWaveParticipantFlightHold {
                slot: candidate.slot,
                epoch: candidate.epoch,
                fingerprint: candidate.fingerprint,
                participant: candidate.participant,
                frame: candidate.frame,
                phase: ParticipantFlightPhase::Prepared,
            },
        )
        .collect())
}

pub(super) fn begin_submission_wave_participant_flights_dispatch(
    holds: &mut [PreparedSubmissionWaveParticipantFlightHold],
) -> Result<(), VNextError> {
    transition_submission_wave_participant_flights(
        holds,
        ParticipantFlightPhase::Prepared,
        ParticipantFlightPhase::InFlight,
        "begin submission wave dispatch",
    )
}

pub(super) fn reset_submission_wave_participant_flights_after_definitely_not_submitted(
    holds: &mut [PreparedSubmissionWaveParticipantFlightHold],
) -> Result<(), VNextError> {
    transition_submission_wave_participant_flights(
        holds,
        ParticipantFlightPhase::InFlight,
        ParticipantFlightPhase::Prepared,
        "reset definitely-not-submitted submission wave dispatch",
    )
}

fn transition_submission_wave_participant_flights(
    holds: &mut [PreparedSubmissionWaveParticipantFlightHold],
    expected: ParticipantFlightPhase,
    next: ParticipantFlightPhase,
    context: &'static str,
) -> Result<(), VNextError> {
    if holds.is_empty()
        || holds.iter().any(|hold| hold.phase != expected)
        || holds
            .windows(2)
            .any(|pair| pair[0].canonical_key() >= pair[1].canonical_key())
    {
        return Err(invalid_resource(format!(
            "{context} requires canonical non-empty unique participant flights in the expected phase"
        )));
    }

    let slots = holds
        .iter()
        .map(|hold| Arc::clone(&hold.slot))
        .collect::<Vec<_>>();
    let mut states = Vec::with_capacity(slots.len());
    for slot in &slots {
        states.push(
            slot.state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    for (hold, state) in holds.iter().zip(&states) {
        match &**state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == hold.epoch
                    && active.fingerprint == hold.fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame == Some(hold.frame)
                    && active.participant_flights.is_empty()
                    && active.submission_wave_flight == Some(expected) => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != hold.epoch || active.fingerprint != hold.fingerprint =>
            {
                return Err(invalid_resource(
                    "stale submission wave participant authority during phase transition",
                ));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(format!(
                    "cancelled, poisoned, wrong-phase, cross-frame, or node-busy participant cannot {context}"
                )));
            }
            _ => {
                return Err(invalid_resource(format!(
                    "inactive or terminal participant cannot {context}"
                )));
            }
        }
    }
    for state in &mut states {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all submission wave participants were validated while locked");
        };
        active.submission_wave_flight = Some(next);
    }
    drop(states);
    for hold in holds {
        hold.phase = next;
    }
    Ok(())
}

pub(super) fn begin_participant_flights_dispatch(
    holds: &mut [PreparedParticipantFlightHold],
) -> Result<(), VNextError> {
    transition_participant_flights(
        holds,
        ParticipantFlightPhase::Prepared,
        ParticipantFlightPhase::InFlight,
        "begin dispatch",
    )
}

pub(super) fn reset_participant_flights_after_definitely_not_submitted(
    holds: &mut [PreparedParticipantFlightHold],
) -> Result<(), VNextError> {
    transition_participant_flights(
        holds,
        ParticipantFlightPhase::InFlight,
        ParticipantFlightPhase::Prepared,
        "reset definitely-not-submitted dispatch",
    )
}

fn transition_participant_flights(
    holds: &mut [PreparedParticipantFlightHold],
    expected: ParticipantFlightPhase,
    next: ParticipantFlightPhase,
    context: &'static str,
) -> Result<(), VNextError> {
    if holds.is_empty()
        || holds.iter().any(|hold| hold.phase != expected)
        || holds.windows(2).any(|pair| pair[0].key >= pair[1].key)
    {
        return Err(invalid_resource(format!(
            "{context} requires canonical non-empty unique participant-node flights in the expected phase"
        )));
    }

    let mut unique_slots = Vec::<Arc<SequenceSessionSlot>>::new();
    let mut slot_indices = Vec::with_capacity(holds.len());
    for hold in holds.iter() {
        let slot_index = if let Some(index) = unique_slots
            .iter()
            .position(|slot| Arc::ptr_eq(slot, &hold.slot))
        {
            index
        } else {
            unique_slots.push(Arc::clone(&hold.slot));
            unique_slots.len() - 1
        };
        slot_indices.push(slot_index);
    }
    let mut states = Vec::with_capacity(unique_slots.len());
    for slot in &unique_slots {
        states.push(
            slot.state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    for (hold, &slot_index) in holds.iter().zip(&slot_indices) {
        match &*states[slot_index] {
            SequenceSessionSlotState::Active(active)
                if active.epoch == hold.epoch
                    && active.fingerprint == hold.fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame
                        == Some(ActiveSequenceFrame {
                            frame_id: hold.key.frame_id(),
                            batch_step_id: hold.batch_step_id,
                        })
                    && active.submission_wave_flight.is_none()
                    && active.participant_flights.get(&hold.key) == Some(&expected) => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != hold.epoch || active.fingerprint != hold.fingerprint =>
            {
                return Err(invalid_resource(
                    "stale invocation participant authority during phase transition",
                ));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(format!(
                    "cancelled, poisoned, wrong-phase, or cross-frame participant cannot {context}"
                )));
            }
            _ => {
                return Err(invalid_resource(format!(
                    "inactive or terminal participant cannot {context}"
                )));
            }
        }
    }
    for (hold, &slot_index) in holds.iter().zip(&slot_indices) {
        let SequenceSessionSlotState::Active(active) = &mut *states[slot_index] else {
            unreachable!("all dispatch participants were validated while locked");
        };
        let phase = active
            .participant_flights
            .get_mut(&hold.key)
            .expect("validated participant flight remains present while locked");
        *phase = next;
    }
    for hold in holds {
        hold.phase = next;
    }
    Ok(())
}

impl Drop for SessionFrameHold {
    fn drop(&mut self) {
        if self.finalized {
            return;
        }
        let mut state = match self.slot.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let SequenceSessionSlotState::Active(active) = &mut *state {
            if active.epoch == self.epoch
                && active.fingerprint == self.fingerprint
                && active.active_frame
                    == Some(ActiveSequenceFrame {
                        frame_id: self.frame_id,
                        batch_step_id: self.batch_step_id,
                    })
            {
                active.phase = SequenceSessionPhase::Poisoned;
            }
        }
    }
}

pub(super) struct AdmittedStepParticipant<R>
where
    R: DeviceRuntime,
{
    pub(super) frame: SessionFrameHold,
    // Drop the independently retained backing before the session parent. The
    // snapshot also owns a runtime keepalive for fence/reaper handoff.
    pub(super) backing_snapshot: Arc<SequenceBackingSnapshot<R>>,
    pub(super) session: Arc<SequenceSession<R>>,
}

fn execution_frame_successor(frame_id: ExecutionFrameId) -> Option<ExecutionFrameId> {
    frame_id
        .get()
        .checked_add(1)
        .and_then(|next| ExecutionFrameId::try_from(next).ok())
}

pub(super) fn acquire_session_frames(
    candidates: &[SequenceFrameCandidate],
    batch_step_id: BatchStepId,
) -> Result<Vec<SessionFrameHold>, VNextError> {
    if candidates.is_empty()
        || candidates.iter().enumerate().any(|(index, candidate)| {
            candidates[..index]
                .iter()
                .any(|prior| Arc::ptr_eq(&prior.slot, &candidate.slot))
        })
    {
        return Err(invalid_resource(
            "step frame acquisition requires non-empty unique session slots",
        ));
    }
    let mut holds = Vec::with_capacity(candidates.len());
    let mut states = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        states.push(
            candidate
                .slot
                .state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    for (candidate, state) in candidates.iter().zip(&states) {
        match &**state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == candidate.epoch
                    && active.fingerprint == candidate.fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame.is_none()
                    && active.next_frame.is_some() => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != candidate.epoch
                    || active.fingerprint != candidate.fingerprint =>
            {
                return Err(invalid_resource("stale sequence session frame authority"));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "sequence session cannot acquire a frame in its current phase",
                ));
            }
            _ => {
                return Err(invalid_resource(
                    "inactive or terminal sequence session cannot acquire a frame",
                ));
            }
        }
    }
    for (candidate, state) in candidates.iter().zip(&mut states) {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all session frame candidates were validated");
        };
        let frame_id = active
            .next_frame
            .take()
            .expect("validated session has a next execution frame");
        active.next_frame = execution_frame_successor(frame_id);
        active.active_frame = Some(ActiveSequenceFrame {
            frame_id,
            batch_step_id,
        });
        holds.push(SessionFrameHold {
            slot: Arc::clone(&candidate.slot),
            epoch: candidate.epoch,
            fingerprint: candidate.fingerprint.clone(),
            frame_id,
            batch_step_id,
            finalized: false,
        });
    }
    Ok(holds)
}

pub(super) fn session_frame_candidates<R: DeviceRuntime>(
    sessions: &[Arc<SequenceSession<R>>],
) -> Vec<SequenceFrameCandidate> {
    sessions
        .iter()
        .map(|session| SequenceFrameCandidate {
            slot: Arc::clone(&session.slot),
            epoch: session.epoch,
            fingerprint: session.fingerprint.clone(),
        })
        .collect()
}

pub(super) fn acquire_session_frames_with_backing<R>(
    candidates: &[SequenceFrameCaptureCandidate<R>],
    batch_step_id: BatchStepId,
) -> Result<Vec<CapturedSessionFrame<R>>, VNextError>
where
    R: DeviceRuntime,
{
    if candidates.is_empty()
        || candidates.iter().enumerate().any(|(index, candidate)| {
            candidates[..index]
                .iter()
                .any(|prior| Arc::ptr_eq(&prior.frame.slot, &candidate.frame.slot))
        })
    {
        return Err(invalid_resource(
            "step frame acquisition requires non-empty unique session slots",
        ));
    }
    let mut states = candidates
        .iter()
        .map(|candidate| {
            candidate
                .frame
                .slot
                .state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (candidate, state) in candidates.iter().zip(&states) {
        match &**state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == candidate.frame.epoch
                    && active.fingerprint == candidate.frame.fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame.is_none()
                    && active.next_frame.is_some() => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != candidate.frame.epoch
                    || active.fingerprint != candidate.frame.fingerprint =>
            {
                return Err(invalid_resource("stale sequence session frame authority"));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "sequence session cannot acquire a frame in its current phase",
                ));
            }
            _ => {
                return Err(invalid_resource(
                    "inactive or terminal sequence session cannot acquire a frame",
                ));
            }
        }
    }
    // Frame capture and extension publication use the same slot -> backing
    // lock order. Allocator, device, copy, and fence work stay outside it.
    let backing_states = candidates
        .iter()
        .map(|candidate| candidate.resources.lock_backing_state())
        .collect::<Result<Vec<_>, _>>()?;
    let mut captured = Vec::with_capacity(candidates.len());
    for ((candidate, state), backing_state) in
        candidates.iter().zip(&mut states).zip(&backing_states)
    {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all session frame candidates were validated");
        };
        let frame_id = active
            .next_frame
            .take()
            .expect("validated session has a next execution frame");
        active.next_frame = execution_frame_successor(frame_id);
        active.active_frame = Some(ActiveSequenceFrame {
            frame_id,
            batch_step_id,
        });
        captured.push(CapturedSessionFrame {
            hold: SessionFrameHold {
                slot: Arc::clone(&candidate.frame.slot),
                epoch: candidate.frame.epoch,
                fingerprint: candidate.frame.fingerprint.clone(),
                frame_id,
                batch_step_id,
                finalized: false,
            },
            backing_snapshot: Arc::clone(&backing_state.current),
        });
    }
    Ok(captured)
}

pub(super) fn session_frame_capture_candidates<R: DeviceRuntime>(
    sessions: &[Arc<SequenceSession<R>>],
) -> Vec<SequenceFrameCaptureCandidate<R>> {
    sessions
        .iter()
        .map(|session| SequenceFrameCaptureCandidate {
            frame: SequenceFrameCandidate {
                slot: Arc::clone(&session.slot),
                epoch: session.epoch,
                fingerprint: session.fingerprint.clone(),
            },
            resources: Arc::clone(session.resources()),
        })
        .collect()
}

pub(super) fn poison_session_frame(hold: &SessionFrameHold) {
    let mut state = match hold.slot.state.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    if let SequenceSessionSlotState::Active(active) = &mut *state {
        if active.epoch == hold.epoch
            && active.fingerprint == hold.fingerprint
            && active.active_frame
                == Some(ActiveSequenceFrame {
                    frame_id: hold.frame_id,
                    batch_step_id: hold.batch_step_id,
                })
        {
            active.phase = SequenceSessionPhase::Poisoned;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StepFrameFinalization {
    Commit,
    Abort,
    RollbackUnsubmitted,
}

pub(super) fn finalize_session_frames(
    holds: &mut [&mut SessionFrameHold],
    finalization: StepFrameFinalization,
) -> Result<Vec<StepParticipantRetirementDisposition>, VNextError> {
    let slots = holds
        .iter()
        .map(|hold| Arc::clone(&hold.slot))
        .collect::<Vec<_>>();
    let mut states = Vec::with_capacity(slots.len());
    for slot in &slots {
        states.push(
            slot.state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    let mut dispositions = Vec::with_capacity(holds.len());
    for (hold, state) in holds.iter().zip(&states) {
        let hold = &**hold;
        let SequenceSessionSlotState::Active(active) = &**state else {
            return Err(invalid_resource(
                "step participant session is no longer active",
            ));
        };
        if active.epoch != hold.epoch
            || active.fingerprint != hold.fingerprint
            || active.active_frame
                != Some(ActiveSequenceFrame {
                    frame_id: hold.frame_id,
                    batch_step_id: hold.batch_step_id,
                })
            || active.next_frame != execution_frame_successor(hold.frame_id)
            || active.has_participant_flights()
            || (finalization == StepFrameFinalization::Commit
                && active.phase == SequenceSessionPhase::Poisoned)
            || (finalization == StepFrameFinalization::Commit && active.retired_frames == u64::MAX)
            || (finalization == StepFrameFinalization::RollbackUnsubmitted
                && active.phase != SequenceSessionPhase::Open)
        {
            return Err(invalid_resource(
                "step finalization differs from its exact session frame or has live participant work",
            ));
        }
        dispositions.push(match finalization {
            StepFrameFinalization::Abort => StepParticipantRetirementDisposition::Aborted,
            StepFrameFinalization::RollbackUnsubmitted => {
                StepParticipantRetirementDisposition::RolledBackUnsubmitted
            }
            StepFrameFinalization::Commit
                if active.phase == SequenceSessionPhase::CancelRequested =>
            {
                StepParticipantRetirementDisposition::DiscardedCancelled
            }
            StepFrameFinalization::Commit => StepParticipantRetirementDisposition::Committed,
        });
    }
    for (hold, state) in holds.iter().zip(&mut states) {
        let hold = &**hold;
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all step participant sessions were validated");
        };
        active.active_frame = None;
        match finalization {
            StepFrameFinalization::Abort => active.phase = SequenceSessionPhase::Poisoned,
            StepFrameFinalization::Commit => active.retired_frames += 1,
            StepFrameFinalization::RollbackUnsubmitted => active.next_frame = Some(hold.frame_id),
        }
    }
    drop(states);
    for hold in holds {
        hold.finalized = true;
    }
    Ok(dispositions)
}

#[derive(Default)]
pub(super) struct InvocationRegistryState {
    pub(super) entries: BTreeMap<ParticipantNodeKey, ParticipantNodeLedgerEntry>,
    pub(super) submission_wave: Option<SubmissionWaveLedgerEntry>,
    pub(super) poisoned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PhysicalInvocationPhase {
    Prepared,
    NotSubmitted,
    InFlight,
    Retired,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct ParticipantNodeLedgerEntry {
    pub(super) batch_invocation_id: BatchInvocationId,
    pub(super) work_fingerprint: String,
    pub(super) phase: PhysicalInvocationPhase,
}

#[derive(Clone, PartialEq, Eq)]
pub(super) struct SubmissionWaveLedgerEntry {
    // Full-plan preparation has already proved the exact node set. This compact entry is
    // therefore the unexpanded participant x plan-node Cartesian tombstone for the step.
    pub(super) ledger: ParticipantNodeLedgerEntry,
    pub(super) covered_participant_nodes: usize,
}

#[derive(Default)]
pub(super) struct InvocationRegistry {
    pub(super) state: Mutex<InvocationRegistryState>,
}

impl InvocationRegistry {
    pub(super) fn ensure_pristine_for_step_rollback(&self) -> Result<(), VNextError> {
        let state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("invocation registry is poisoned"))?;
        if state.poisoned || !state.entries.is_empty() || state.submission_wave.is_some() {
            return Err(invalid_resource(
                "unsubmitted step rollback requires a pristine invocation registry",
            ));
        }
        Ok(())
    }

    pub(super) fn enter(
        self: &Arc<Self>,
        keys: Vec<ParticipantNodeKey>,
        batch_invocation_id: BatchInvocationId,
        work_fingerprint: &str,
    ) -> Result<ActiveInvocationWaveGuard, VNextError> {
        if keys.is_empty() || keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid_resource(
                "physical invocation ledger requires canonical non-empty unique participant-node keys",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("invocation registry is poisoned"))?;
        if state.poisoned {
            return Err(invalid_resource("invocation registry is fail-closed"));
        }
        if state.submission_wave.is_some() || keys.iter().any(|key| state.entries.contains_key(key))
        {
            return Err(invalid_resource(
                "participant/frame/node topology is already prepared, in flight, or retired in this step",
            ));
        }
        let entry = ParticipantNodeLedgerEntry {
            batch_invocation_id,
            work_fingerprint: work_fingerprint.to_owned(),
            phase: PhysicalInvocationPhase::Prepared,
        };
        for key in &keys {
            if state.entries.insert(key.clone(), entry.clone()).is_some() {
                state.poisoned = true;
                return Err(invalid_resource(
                    "physical invocation ledger changed during atomic prepare",
                ));
            }
        }
        Ok(ActiveInvocationWaveGuard {
            registry: Arc::clone(self),
            topology: ActiveInvocationLedgerTopology::ParticipantNodes(keys),
            work_fingerprint: work_fingerprint.to_owned(),
            batch_invocation_id,
            phase: PhysicalInvocationPhase::Prepared,
        })
    }

    pub(super) fn enter_submission_wave(
        self: &Arc<Self>,
        covered_participant_nodes: usize,
        batch_invocation_id: BatchInvocationId,
        topology_fingerprint: &str,
    ) -> Result<ActiveInvocationWaveGuard, VNextError> {
        if covered_participant_nodes == 0 {
            return Err(invalid_resource(
                "full-plan submission wave ledger requires non-zero participant-node coverage",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| invalid_resource("invocation registry is poisoned"))?;
        if state.poisoned {
            return Err(invalid_resource("invocation registry is fail-closed"));
        }
        if state.submission_wave.is_some() || !state.entries.is_empty() {
            return Err(invalid_resource(
                "full-plan submission wave overlaps prepared, in-flight, or retired participant-node topology in this step",
            ));
        }
        let ledger = ParticipantNodeLedgerEntry {
            batch_invocation_id,
            work_fingerprint: topology_fingerprint.to_owned(),
            phase: PhysicalInvocationPhase::Prepared,
        };
        state.submission_wave = Some(SubmissionWaveLedgerEntry {
            ledger,
            covered_participant_nodes,
        });
        Ok(ActiveInvocationWaveGuard {
            registry: Arc::clone(self),
            topology: ActiveInvocationLedgerTopology::FullPlanSubmissionWave {
                covered_participant_nodes,
            },
            work_fingerprint: topology_fingerprint.to_owned(),
            batch_invocation_id,
            phase: PhysicalInvocationPhase::Prepared,
        })
    }
}

enum ActiveInvocationLedgerTopology {
    ParticipantNodes(Vec<ParticipantNodeKey>),
    FullPlanSubmissionWave { covered_participant_nodes: usize },
}

pub(super) struct ActiveInvocationWaveGuard {
    registry: Arc<InvocationRegistry>,
    topology: ActiveInvocationLedgerTopology,
    work_fingerprint: String,
    batch_invocation_id: BatchInvocationId,
    phase: PhysicalInvocationPhase,
}

impl ActiveInvocationWaveGuard {
    pub(super) const fn physical_entry_count(&self) -> usize {
        match &self.topology {
            ActiveInvocationLedgerTopology::ParticipantNodes(keys) => keys.len(),
            ActiveInvocationLedgerTopology::FullPlanSubmissionWave { .. } => 1,
        }
    }

    fn transition(
        &mut self,
        expected: PhysicalInvocationPhase,
        next: PhysicalInvocationPhase,
        next_attempt: BatchInvocationId,
    ) -> Result<(), VNextError> {
        if self.phase != expected {
            return Err(invalid_resource(
                "physical invocation guard is not in the expected phase",
            ));
        }
        let mut state = self
            .registry
            .state
            .lock()
            .map_err(|_| invalid_resource("invocation registry is poisoned"))?;
        let expected_entry = ParticipantNodeLedgerEntry {
            batch_invocation_id: self.batch_invocation_id,
            work_fingerprint: self.work_fingerprint.clone(),
            phase: expected,
        };
        let authority_matches = match &self.topology {
            ActiveInvocationLedgerTopology::ParticipantNodes(keys) => {
                state.submission_wave.is_none()
                    && keys
                        .iter()
                        .all(|key| state.entries.get(key) == Some(&expected_entry))
            }
            ActiveInvocationLedgerTopology::FullPlanSubmissionWave {
                covered_participant_nodes,
            } => {
                state.entries.is_empty()
                    && state.submission_wave
                        == Some(SubmissionWaveLedgerEntry {
                            ledger: expected_entry.clone(),
                            covered_participant_nodes: *covered_participant_nodes,
                        })
            }
        };
        if state.poisoned || !authority_matches {
            state.poisoned = true;
            return Err(invalid_resource(
                "physical invocation ledger differs from its exact transition authority",
            ));
        }
        match &self.topology {
            ActiveInvocationLedgerTopology::ParticipantNodes(keys) => {
                for key in keys {
                    let entry = state
                        .entries
                        .get_mut(key)
                        .expect("validated physical invocation key remains present");
                    entry.batch_invocation_id = next_attempt;
                    entry.phase = next;
                }
            }
            ActiveInvocationLedgerTopology::FullPlanSubmissionWave { .. } => {
                let entry = &mut state
                    .submission_wave
                    .as_mut()
                    .expect("validated full-plan submission wave remains present")
                    .ledger;
                entry.batch_invocation_id = next_attempt;
                entry.phase = next;
            }
        }
        self.batch_invocation_id = next_attempt;
        self.phase = next;
        Ok(())
    }

    pub(super) fn mark_not_submitted(&mut self) -> Result<(), VNextError> {
        self.transition(
            PhysicalInvocationPhase::Prepared,
            PhysicalInvocationPhase::NotSubmitted,
            self.batch_invocation_id,
        )
    }

    pub(super) fn prepare_retry(
        &mut self,
        fresh_attempt: BatchInvocationId,
    ) -> Result<(), VNextError> {
        if fresh_attempt == self.batch_invocation_id {
            return Err(invalid_resource(
                "definitely-not-submitted retry requires a fresh physical attempt id",
            ));
        }
        self.transition(
            PhysicalInvocationPhase::NotSubmitted,
            PhysicalInvocationPhase::Prepared,
            fresh_attempt,
        )
    }

    pub(super) fn mark_in_flight(&mut self) -> Result<(), VNextError> {
        self.transition(
            PhysicalInvocationPhase::Prepared,
            PhysicalInvocationPhase::InFlight,
            self.batch_invocation_id,
        )
    }
}

impl Drop for ActiveInvocationWaveGuard {
    fn drop(&mut self) {
        if self.phase == PhysicalInvocationPhase::Retired {
            return;
        }
        let mut state = match self.registry.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                state
            }
        };
        let expected = ParticipantNodeLedgerEntry {
            batch_invocation_id: self.batch_invocation_id,
            work_fingerprint: self.work_fingerprint.clone(),
            phase: self.phase,
        };
        let authority_matches = match &self.topology {
            ActiveInvocationLedgerTopology::ParticipantNodes(keys) => {
                state.submission_wave.is_none()
                    && keys
                        .iter()
                        .all(|key| state.entries.get(key) == Some(&expected))
            }
            ActiveInvocationLedgerTopology::FullPlanSubmissionWave {
                covered_participant_nodes,
            } => {
                state.entries.is_empty()
                    && state.submission_wave
                        == Some(SubmissionWaveLedgerEntry {
                            ledger: expected,
                            covered_participant_nodes: *covered_participant_nodes,
                        })
            }
        };
        if !authority_matches {
            state.poisoned = true;
            return;
        }
        match &self.topology {
            ActiveInvocationLedgerTopology::ParticipantNodes(keys) => {
                for key in keys {
                    state
                        .entries
                        .get_mut(key)
                        .expect("validated physical invocation key remains present")
                        .phase = PhysicalInvocationPhase::Retired;
                }
            }
            ActiveInvocationLedgerTopology::FullPlanSubmissionWave { .. } => {
                state
                    .submission_wave
                    .as_mut()
                    .expect("validated full-plan submission wave remains present")
                    .ledger
                    .phase = PhysicalInvocationPhase::Retired;
            }
        }
        self.phase = PhysicalInvocationPhase::Retired;
    }
}

pub enum StepResourceAdmissionDecision<R>
where
    R: DeviceRuntime,
{
    Admitted(Arc<StepResourceLease<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(StepAdmissionBackingDeferral<R>),
    PermanentRejected(AdmissionRejected),
}

/// Non-cloneable physical-backing authority for one exact batch participant
/// set and immutable step work shape.
#[must_use = "step backing deferral retains its exact participant parents"]
pub struct StepAdmissionBackingDeferral<R>
where
    R: DeviceRuntime,
{
    backing: PlanBackingDeferral<R>,
    participants: Vec<Arc<SequenceSession<R>>>,
    work_fingerprint: String,
}

impl<R> StepAdmissionBackingDeferral<R>
where
    R: DeviceRuntime,
{
    pub(super) fn new(
        evidence: DynamicBackingDeferred,
        participants: Vec<Arc<SequenceSession<R>>>,
        work_fingerprint: String,
    ) -> Result<Self, VNextError> {
        let first = participants
            .first()
            .ok_or_else(|| invalid_resource("step backing deferral requires participants"))?;
        let resources = Arc::clone(&first.resources().request.plan.resources);
        if participants.iter().any(|participant| {
            !Arc::ptr_eq(&resources, &participant.resources().request.plan.resources)
        }) {
            return Err(invalid_resource(
                "step backing deferral participants belong to different plans",
            ));
        }
        Ok(Self {
            backing: PlanBackingDeferral::new(resources, evidence)?,
            participants,
            work_fingerprint,
        })
    }

    pub fn evidence(&self) -> &DynamicBackingDeferred {
        self.backing.evidence()
    }

    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    pub fn work_fingerprint(&self) -> &str {
        &self.work_fingerprint
    }

    pub fn maintain(&self) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        for participant in &self.participants {
            participant.ensure_open_identity()?;
        }
        self.backing.maintain()
    }

    pub fn register_waiter(&self) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        self.backing.register_waiter()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StepParticipantRetirementDisposition {
    Committed,
    DiscardedCancelled,
    RolledBackUnsubmitted,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StepParticipantRetirement {
    pub(super) assignment: StepParticipantFrameAssignment,
    pub(super) disposition: StepParticipantRetirementDisposition,
}

impl StepParticipantRetirement {
    pub const fn assignment(&self) -> StepParticipantFrameAssignment {
        self.assignment
    }

    pub const fn disposition(&self) -> StepParticipantRetirementDisposition {
        self.disposition
    }
}

/// One atomic physical/logical backing claim bound to an immutable batch work
/// authority. Even an empty resource demand retains the work shape and claim
/// fingerprint through dispatch and fence ownership.
#[must_use = "claimed backing must remain owned through its device fence"]
pub struct ClaimedBackingTransaction {
    // Physical extents release before the logical capacity claim.
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    logical_capacity: Option<LogicalBatchCapacityLease>,
    work_shape: Arc<BatchWorkShape>,
    demand: AdmissionDemand,
    fingerprint: String,
    // Occupancy releases after every physical/logical claim field above.
    _lane_slot_lease: Option<LaneStableArenaSlotLease>,
}

fn validate_backing_claim(
    backing_slices: &[LogicalBackingSliceAuthority],
    demand: &AdmissionDemand,
) -> Result<(), VNextError> {
    let reusable_execution_bucket_id = backing_slices
        .first()
        .and_then(|slice| slice.evidence().reusable_execution_bucket_id());
    if backing_slices.iter().any(|slice| {
        slice.evidence().reusable_execution_bucket_id() != reusable_execution_bucket_id
    }) {
        return Err(invalid_resource(
            "one backing transaction cannot mix reusable execution buckets",
        ));
    }
    let mut backing_by_domain = BTreeMap::<CapacityDomainId, u64>::new();
    let mut physical_claims = BTreeMap::<
        PhysicalBackingClaimIdentity,
        (Arc<super::BackingSegmentLease>, CapacityDomainId, u64),
    >::new();
    for slice in backing_slices {
        let evidence = slice.evidence();
        let claim_identity = evidence.physical_claim_identity();
        if claim_identity.pool_id() != evidence.pool_id()
            || claim_identity
                .resource_ids()
                .binary_search(evidence.resource_id())
                .is_err()
            || slice.segment_lease.claim_identity != *claim_identity
            || slice.segment_lease.segment_generation != evidence.segment_generation()
            || slice.segment_lease.size_bytes != evidence.physical_size_bytes()
            || evidence.size_bytes() == 0
            || evidence.size_bytes() > evidence.capacity_size_bytes()
            || evidence
                .physical_offset_bytes()
                .checked_add(evidence.capacity_size_bytes())
                .is_none_or(|end| end > evidence.physical_size_bytes())
        {
            return Err(invalid_resource(
                "logical backing projection differs from its physical claim authority",
            ));
        }
        match physical_claims.entry(claim_identity.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                let total = backing_by_domain.entry(slice.domain_id()).or_default();
                *total = total
                    .checked_add(evidence.physical_size_bytes())
                    .ok_or_else(|| invalid_resource("claimed backing domain bytes overflow u64"))?;
                entry.insert((
                    Arc::clone(&slice.segment_lease),
                    slice.domain_id(),
                    evidence.physical_size_bytes(),
                ));
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                let (lease, domain_id, size_bytes) = entry.get();
                if !Arc::ptr_eq(lease, &slice.segment_lease)
                    || *domain_id != slice.domain_id()
                    || *size_bytes != evidence.physical_size_bytes()
                {
                    return Err(invalid_resource(
                        "shared logical projections do not retain one physical claim",
                    ));
                }
            }
        }
    }
    let backing_claim = if backing_by_domain.is_empty() {
        CapacityVector::empty()
    } else {
        CapacityVector::new(
            backing_by_domain
                .into_iter()
                .map(|(domain, bytes)| CapacityEntry::new(domain, CapacityUnits::new(bytes)))
                .collect::<Result<Vec<_>, _>>()?,
        )?
    };
    let physical_covers_logical = backing_claim.entries().len()
        == demand.immediate_claim().entries().len()
        && backing_claim.entries().iter().all(|physical| {
            demand
                .immediate_claim()
                .units_for(physical.domain())
                .is_some_and(|logical| physical.units().get() >= logical.get())
        });
    let claim_matches = if reusable_execution_bucket_id.is_some() {
        physical_covers_logical
    } else {
        backing_claim == *demand.immediate_claim()
    };
    if !claim_matches {
        return Err(invalid_resource(
            "physical backing does not cover the exact evaluated logical demand",
        ));
    }
    Ok(())
}

fn logical_capacity_matches(
    logical_capacity: &Option<LogicalBatchCapacityLease>,
    demand: &AdmissionDemand,
    participants: &[BatchParticipantAuthority],
    backing_slices: &[LogicalBackingSliceAuthority],
) -> bool {
    match logical_capacity {
        Some(capacity) => {
            capacity.claims() == demand.immediate_claim()
                && capacity.parents().len() == participants.len()
                && capacity
                    .parents()
                    .iter()
                    .zip(participants)
                    .all(|(parent, participant)| {
                        parent.sequence() == participant.sequence_authority()
                            && parent.request() == participant.request_authority()
                    })
        }
        None => demand.immediate_claim().is_empty() && backing_slices.is_empty(),
    }
}

impl ClaimedBackingTransaction {
    pub(super) fn new(
        work_shape: Arc<BatchWorkShape>,
        demand: AdmissionDemand,
        logical_capacity: Option<LogicalBatchCapacityLease>,
        backing_slices: Vec<LogicalBackingSliceAuthority>,
        lane_slot_lease: Option<LaneStableArenaSlotLease>,
    ) -> Result<Self, VNextError> {
        validate_backing_claim(&backing_slices, &demand)?;
        if !logical_capacity_matches(
            &logical_capacity,
            &demand,
            work_shape.participants(),
            &backing_slices,
        ) {
            return Err(invalid_resource(
                "logical backing claim differs from work participants or evaluated demand",
            ));
        }
        #[derive(Serialize)]
        struct BackingFingerprint<'a> {
            allocation_fingerprint: &'a str,
            logical_size_bytes: u64,
        }

        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            work_fingerprint: &'a str,
            demand: &'a AdmissionDemand,
            backing: Vec<BackingFingerprint<'a>>,
            capacity_parents: Vec<(SequenceAuthorityId, RequestAuthorityId)>,
        }
        let input = FingerprintInput {
            domain: "ferrum.runtime-vnext.claimed-backing.v4",
            work_fingerprint: work_shape.fingerprint(),
            demand: &demand,
            backing: backing_slices
                .iter()
                .map(LogicalBackingSliceAuthority::evidence)
                .map(|evidence| BackingFingerprint {
                    allocation_fingerprint: evidence.allocation_fingerprint(),
                    logical_size_bytes: evidence.size_bytes(),
                })
                .collect(),
            capacity_parents: logical_capacity
                .as_ref()
                .map(|capacity| {
                    capacity
                        .parents()
                        .iter()
                        .map(|parent| (parent.sequence(), parent.request()))
                        .collect()
                })
                .unwrap_or_default(),
        };
        let bytes = serde_json::to_vec(&input).map_err(|error| {
            invalid_resource(format!(
                "claimed backing fingerprint encode failed: {error}"
            ))
        })?;
        Ok(Self {
            backing_slices,
            logical_capacity,
            work_shape,
            demand,
            fingerprint: format!("{:x}", Sha256::digest(bytes)),
            _lane_slot_lease: lane_slot_lease,
        })
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work_shape.as_ref()
    }

    pub(super) fn work_shape_arc(&self) -> &Arc<BatchWorkShape> {
        &self.work_shape
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        &self.backing_slices
    }

    pub fn logical_capacity(&self) -> Option<&LogicalBatchCapacityLease> {
        self.logical_capacity.as_ref()
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn demand(&self) -> &AdmissionDemand {
        &self.demand
    }

    pub fn physical_claim_count(&self) -> usize {
        self.backing_slices
            .iter()
            .map(|slice| slice.evidence().physical_claim_identity())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    }

    pub fn has_shared_physical_claims(&self) -> bool {
        self.backing_slices
            .iter()
            .any(|slice| slice.evidence().physical_claim_identity().is_shared())
    }

    pub fn lane_stable_slot_identity(&self) -> Option<LaneStableArenaSlotIdentity> {
        self._lane_slot_lease
            .as_ref()
            .map(LaneStableArenaSlotLease::identity)
    }
}

/// One physical/logical Invocation backing transaction shared by every node
/// in an immutable-plan submission wave. Physical demand is charged once for
/// the liveness-derived peak and retained until the wave's terminal fence.
#[must_use = "submission wave backing must remain owned through its device fence"]
pub struct ClaimedSubmissionWaveBacking {
    // Physical extents release before the logical capacity claim.
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    logical_capacity: Option<LogicalBatchCapacityLease>,
    plan_hash: PlanHash,
    node_count: usize,
    work_shape: Arc<BatchWorkShape>,
    demand: AdmissionDemand,
    fingerprint: String,
    program_binding: Option<Arc<ProgramBindingExecutionBinding>>,
    // Occupancy releases only after terminal completion readback drops this claim.
    _lane_slot_lease: Option<LaneStableArenaSlotLease>,
}

impl ClaimedSubmissionWaveBacking {
    pub(super) fn new(
        plan_hash: PlanHash,
        node_count: usize,
        work_shape: Arc<BatchWorkShape>,
        demand: AdmissionDemand,
        logical_capacity: Option<LogicalBatchCapacityLease>,
        backing_slices: Vec<LogicalBackingSliceAuthority>,
        program_binding_layout: Option<Arc<ProgramBindingLayout>>,
        lane_slot_lease: Option<LaneStableArenaSlotLease>,
    ) -> Result<Self, VNextError> {
        let participants = work_shape.participants();
        if node_count == 0
            || participants.is_empty()
            || participants
                .windows(2)
                .any(|pair| pair[0].canonical_key() >= pair[1].canonical_key())
        {
            return Err(invalid_resource(
                "submission wave backing requires ordered unique nodes and participants",
            ));
        }
        validate_backing_claim(&backing_slices, &demand)?;
        if !logical_capacity_matches(&logical_capacity, &demand, participants, &backing_slices) {
            return Err(invalid_resource(
                "submission wave capacity differs from its participants or evaluated demand",
            ));
        }
        let program_binding = program_binding_layout
            .map(|layout| {
                let lane_slot_identity = lane_slot_lease
                    .as_ref()
                    .map(LaneStableArenaSlotLease::identity)
                    .ok_or_else(|| {
                        invalid_resource(
                            "compiled program binding layout has no lane-stable slot authority",
                        )
                    })?;
                ProgramBindingExecutionBinding::bind(
                    plan_hash.clone(),
                    node_count,
                    layout,
                    lane_slot_identity,
                    &backing_slices,
                )
            })
            .transpose()?;

        #[derive(Serialize)]
        struct BackingFingerprint<'a> {
            allocation_fingerprint: &'a str,
            logical_size_bytes: u64,
        }

        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            plan_hash: &'a PlanHash,
            node_count: usize,
            work_fingerprint: &'a str,
            demand: &'a AdmissionDemand,
            backing: Vec<BackingFingerprint<'a>>,
            capacity_parents: Vec<(SequenceAuthorityId, RequestAuthorityId)>,
        }
        let input = FingerprintInput {
            domain: "ferrum.runtime-vnext.claimed-submission-wave-backing.v4",
            plan_hash: &plan_hash,
            node_count,
            work_fingerprint: work_shape.fingerprint(),
            demand: &demand,
            backing: backing_slices
                .iter()
                .map(LogicalBackingSliceAuthority::evidence)
                .map(|evidence| BackingFingerprint {
                    allocation_fingerprint: evidence.allocation_fingerprint(),
                    logical_size_bytes: evidence.size_bytes(),
                })
                .collect(),
            capacity_parents: logical_capacity
                .as_ref()
                .map(|capacity| {
                    capacity
                        .parents()
                        .iter()
                        .map(|parent| (parent.sequence(), parent.request()))
                        .collect()
                })
                .unwrap_or_default(),
        };
        let bytes = serde_json::to_vec(&input).map_err(|error| {
            invalid_resource(format!(
                "submission wave backing fingerprint encode failed: {error}"
            ))
        })?;
        Ok(Self {
            backing_slices,
            logical_capacity,
            plan_hash,
            node_count,
            work_shape,
            demand,
            fingerprint: format!("{:x}", Sha256::digest(bytes)),
            program_binding,
            _lane_slot_lease: lane_slot_lease,
        })
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        &self.backing_slices
    }

    pub fn logical_capacity(&self) -> Option<&LogicalBatchCapacityLease> {
        self.logical_capacity.as_ref()
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work_shape.as_ref()
    }

    pub fn participants(&self) -> &[BatchParticipantAuthority] {
        self.work_shape.participants()
    }

    pub fn demand(&self) -> &AdmissionDemand {
        &self.demand
    }

    pub fn program_binding_node(&self, node_index: usize) -> Option<ProgramBindingNodeBinding> {
        self.program_binding
            .as_ref()
            .and_then(|binding| binding.node(node_index))
    }

    pub fn program_binding_layout(&self) -> Option<&ProgramBindingLayout> {
        self.program_binding
            .as_ref()
            .map(|binding| binding.layout())
    }

    pub fn program_binding_lane_slot_identity(&self) -> Option<&LaneStableArenaSlotIdentity> {
        self.program_binding
            .as_ref()
            .map(|binding| binding.lane_slot_identity())
    }

    pub fn reusable_execution_bucket_id(&self) -> Option<&ReusableExecutionBucketId> {
        self.program_binding_layout()
            .map(ProgramBindingLayout::reusable_execution_bucket_id)
            .or_else(|| {
                self.backing_slices
                    .iter()
                    .find_map(|slice| slice.evidence().reusable_execution_bucket_id())
            })
    }

    pub fn reusable_execution_program_id(
        &self,
        runtime_implementation_fingerprint: &str,
        lane_id: super::ExecutionLaneId,
    ) -> Result<Option<DeviceReusableExecutionProgramId>, VNextError> {
        let Some(layout) = self.program_binding_layout() else {
            if self.program_binding_lane_slot_identity().is_some() {
                return Err(invalid_resource(
                    "reusable execution lane slot has no compiled program binding layout",
                ));
            }
            return Ok(None);
        };
        let lane_slot = self.program_binding_lane_slot_identity().ok_or_else(|| {
            invalid_resource("compiled program binding layout has no lane-stable slot identity")
        })?;
        if lane_slot.lane_id() != lane_id
            || lane_slot.reusable_execution_bucket_id() != layout.reusable_execution_bucket_id()
        {
            return Err(invalid_resource(
                "reusable execution program layout differs from its lane-stable slot",
            ));
        }
        DeviceReusableExecutionProgramId::new(
            self.plan_hash.clone(),
            runtime_implementation_fingerprint.to_owned(),
            lane_id,
            layout.reusable_execution_bucket_id().clone(),
            layout.fingerprint().to_owned(),
            lane_slot.layout_fingerprint().to_owned(),
            lane_slot.slot_id(),
            self.work_shape.immediate_sequences(),
            self.work_shape.immediate_tokens(),
            self.work_shape.immediate_pages(),
        )
        .map(Some)
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn physical_claim_count(&self) -> usize {
        self.backing_slices
            .iter()
            .map(|slice| slice.evidence().physical_claim_identity())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StepRetirementReceipt {
    pub(super) batch_step_id: BatchStepId,
    pub(super) participants: Vec<StepParticipantRetirement>,
}

impl StepRetirementReceipt {
    pub const fn batch_step_id(&self) -> BatchStepId {
        self.batch_step_id
    }

    pub fn participants(&self) -> &[StepParticipantRetirement] {
        &self.participants
    }
}

#[must_use = "failed step finalization retains the exact step authority"]
pub struct StepFinalizationFailure<R>
where
    R: DeviceRuntime,
{
    pub(super) step: Arc<StepResourceLease<R>>,
    pub(super) error: VNextError,
}

impl<R> fmt::Debug for StepFinalizationFailure<R>
where
    R: DeviceRuntime,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StepFinalizationFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<R> StepFinalizationFailure<R>
where
    R: DeviceRuntime,
{
    pub fn error(&self) -> &VNextError {
        &self.error
    }

    pub fn into_step(self) -> Arc<StepResourceLease<R>> {
        self.step
    }
}
