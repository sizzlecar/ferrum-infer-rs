use super::*;

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

pub(super) struct SessionFrameHold {
    pub(super) slot: Arc<SequenceSessionSlot>,
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
    pub(super) frame_id: ExecutionFrameId,
    pub(super) batch_step_id: BatchStepId,
    pub(super) finalized: bool,
}

#[derive(Clone)]
pub(super) struct ParticipantFlightCandidate {
    pub(super) slot: Arc<SequenceSessionSlot>,
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
    pub(super) frame: ActiveSequenceFrame,
    pub(super) participant: BatchParticipantAuthority,
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
    if candidates.is_empty()
        || candidates.iter().enumerate().any(|(index, candidate)| {
            candidates[..index]
                .iter()
                .any(|prior| Arc::ptr_eq(&prior.slot, &candidate.slot))
        })
    {
        return Err(invalid_resource(
            "prepared invocation requires non-empty unique participant sessions",
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
        let key = ParticipantNodeKey::new(
            candidate.participant,
            candidate.frame.frame_id,
            node_id.clone(),
        );
        match &**state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == candidate.epoch
                    && active.fingerprint == candidate.fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame == Some(candidate.frame)
                    && !active.participant_flights.contains_key(&key) => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != candidate.epoch
                    || active.fingerprint != candidate.fingerprint =>
            {
                return Err(invalid_resource(
                    "stale prepared invocation participant authority",
                ));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "cancelled, poisoned, duplicate, or cross-frame participant cannot enter an invocation",
                ));
            }
            _ => {
                return Err(invalid_resource(
                    "inactive or terminal participant cannot enter an invocation",
                ));
            }
        }
    }
    let mut insertion_failure = None;
    for (index, (candidate, state)) in candidates.iter().zip(&mut states).enumerate() {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all prepared invocation participants were validated");
        };
        let key = ParticipantNodeKey::new(
            candidate.participant,
            candidate.frame.frame_id,
            node_id.clone(),
        );
        if let Some(previous) = active
            .participant_flights
            .insert(key.clone(), ParticipantFlightPhase::Prepared)
        {
            active.participant_flights.insert(key, previous);
            active.phase = SequenceSessionPhase::Poisoned;
            insertion_failure = Some(index);
            break;
        }
    }
    if let Some(index) = insertion_failure {
        for rollback_index in 0..index {
            let rollback_key = ParticipantNodeKey::new(
                candidates[rollback_index].participant,
                candidates[rollback_index].frame.frame_id,
                node_id.clone(),
            );
            let SequenceSessionSlotState::Active(rollback) = &mut *states[rollback_index] else {
                continue;
            };
            rollback.participant_flights.remove(&rollback_key);
            rollback.phase = SequenceSessionPhase::Poisoned;
        }
        return Err(invalid_resource(
            "prepared participant flight changed during atomic insertion",
        ));
    }
    drop(states);
    for candidate in candidates {
        let key = ParticipantNodeKey::new(
            candidate.participant,
            candidate.frame.frame_id,
            node_id.clone(),
        );
        holds.push(PreparedParticipantFlightHold {
            slot: Arc::clone(&candidate.slot),
            epoch: candidate.epoch,
            fingerprint: candidate.fingerprint.clone(),
            key,
            batch_step_id: candidate.frame.batch_step_id,
            phase: ParticipantFlightPhase::Prepared,
        });
    }
    Ok(holds)
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
        || holds.iter().enumerate().any(|(index, hold)| {
            hold.phase != expected
                || holds[..index]
                    .iter()
                    .any(|prior| Arc::ptr_eq(&prior.slot, &hold.slot))
        })
    {
        return Err(invalid_resource(format!(
            "{context} requires non-empty unique participant flights in the expected phase"
        )));
    }

    // Holds originate from the canonically ordered batch participant set. Keep
    // every session lock until all checks and transitions have completed so a
    // concurrent cancellation is ordered wholly before or after this change.
    let candidates = holds
        .iter()
        .map(|hold| {
            (
                Arc::clone(&hold.slot),
                hold.epoch,
                hold.fingerprint.clone(),
                hold.key.clone(),
                hold.batch_step_id,
            )
        })
        .collect::<Vec<_>>();
    let mut states = Vec::with_capacity(candidates.len());
    for (slot, _, _, _, _) in &candidates {
        states.push(
            slot.state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?,
        );
    }
    for ((_, epoch, fingerprint, key, batch_step_id), state) in candidates.iter().zip(&states) {
        match &**state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == *epoch
                    && active.fingerprint == *fingerprint
                    && active.phase == SequenceSessionPhase::Open
                    && active.active_frame
                        == Some(ActiveSequenceFrame {
                            frame_id: key.frame_id(),
                            batch_step_id: *batch_step_id,
                        })
                    && active.participant_flights.get(key) == Some(&expected) => {}
            SequenceSessionSlotState::Active(active)
                if active.epoch != *epoch || active.fingerprint != *fingerprint =>
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
    for ((_, _, _, key, _), state) in candidates.iter().zip(&mut states) {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all dispatch participants were validated while locked");
        };
        let phase = active
            .participant_flights
            .get_mut(key)
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
    pub(super) session: Arc<SequenceSession<R>>,
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
        active.next_frame = frame_id
            .get()
            .checked_add(1)
            .and_then(|next| ExecutionFrameId::try_from(next).ok());
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

pub(super) fn finalize_session_frames(
    holds: &mut [&mut SessionFrameHold],
    abort: bool,
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
            || !active.participant_flights.is_empty()
            || (!abort && active.phase == SequenceSessionPhase::Poisoned)
            || (!abort && active.retired_frames == u64::MAX)
        {
            return Err(invalid_resource(
                "step finalization differs from its exact session frame or has live participant work",
            ));
        }
        dispositions.push(if abort {
            StepParticipantRetirementDisposition::Aborted
        } else if active.phase == SequenceSessionPhase::CancelRequested {
            StepParticipantRetirementDisposition::DiscardedCancelled
        } else {
            StepParticipantRetirementDisposition::Committed
        });
    }
    for state in &mut states {
        let SequenceSessionSlotState::Active(active) = &mut **state else {
            unreachable!("all step participant sessions were validated");
        };
        active.active_frame = None;
        if abort {
            active.phase = SequenceSessionPhase::Poisoned;
        } else {
            active.retired_frames += 1;
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

#[derive(Default)]
pub(super) struct InvocationRegistry {
    pub(super) state: Mutex<InvocationRegistryState>,
}

impl InvocationRegistry {
    pub(super) fn enter(
        self: &Arc<Self>,
        keys: Vec<ParticipantNodeKey>,
        batch_invocation_id: BatchInvocationId,
        work_fingerprint: &str,
    ) -> Result<ActiveInvocationGuard, VNextError> {
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
        if keys.iter().any(|key| state.entries.contains_key(key)) {
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
        Ok(ActiveInvocationGuard {
            registry: Arc::clone(self),
            keys,
            work_fingerprint: work_fingerprint.to_owned(),
            batch_invocation_id,
            phase: PhysicalInvocationPhase::Prepared,
        })
    }
}

pub(super) struct ActiveInvocationGuard {
    registry: Arc<InvocationRegistry>,
    keys: Vec<ParticipantNodeKey>,
    work_fingerprint: String,
    batch_invocation_id: BatchInvocationId,
    phase: PhysicalInvocationPhase,
}

impl ActiveInvocationGuard {
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
        if state.poisoned
            || self
                .keys
                .iter()
                .any(|key| state.entries.get(key) != Some(&expected_entry))
        {
            state.poisoned = true;
            return Err(invalid_resource(
                "physical invocation ledger differs from its exact transition authority",
            ));
        }
        for key in &self.keys {
            let entry = state
                .entries
                .get_mut(key)
                .expect("validated physical invocation key remains present");
            entry.batch_invocation_id = next_attempt;
            entry.phase = next;
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

impl Drop for ActiveInvocationGuard {
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
        if self
            .keys
            .iter()
            .any(|key| state.entries.get(key) != Some(&expected))
        {
            state.poisoned = true;
            return;
        }
        for key in &self.keys {
            state
                .entries
                .get_mut(key)
                .expect("validated physical invocation key remains present")
                .phase = PhysicalInvocationPhase::Retired;
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
    BackingDeferred(DynamicBackingDeferred),
    PermanentRejected(AdmissionRejected),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StepParticipantRetirementDisposition {
    Committed,
    DiscardedCancelled,
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
    work_shape: BatchWorkShape,
    demand: AdmissionDemand,
    fingerprint: String,
}

impl ClaimedBackingTransaction {
    pub(super) fn new(
        work_shape: BatchWorkShape,
        demand: AdmissionDemand,
        logical_capacity: Option<LogicalBatchCapacityLease>,
        backing_slices: Vec<LogicalBackingSliceAuthority>,
    ) -> Result<Self, VNextError> {
        let mut backing_by_domain = BTreeMap::<CapacityDomainId, u64>::new();
        for slice in &backing_slices {
            let total = backing_by_domain.entry(slice.domain_id()).or_default();
            *total = total
                .checked_add(slice.size_bytes())
                .ok_or_else(|| invalid_resource("claimed backing domain bytes overflow u64"))?;
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
        if backing_claim != *demand.immediate_claim() {
            return Err(invalid_resource(
                "physical backing differs from the exact evaluated immediate demand",
            ));
        }
        match &logical_capacity {
            Some(capacity)
                if capacity.claims() == demand.immediate_claim()
                    && capacity.parents().len() == work_shape.participants().len()
                    && capacity
                        .parents()
                        .iter()
                        .zip(work_shape.participants())
                        .all(|(parent, participant)| {
                            parent.sequence() == participant.sequence_authority()
                                && parent.request() == participant.request_authority()
                        }) => {}
            None if demand.immediate_claim().is_empty() && backing_slices.is_empty() => {}
            _ => {
                return Err(invalid_resource(
                    "logical backing claim differs from work participants or evaluated demand",
                ))
            }
        }
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            work_fingerprint: &'a str,
            demand: &'a AdmissionDemand,
            backing: Vec<&'a LogicalBackingSliceEvidence>,
            capacity_parents: Vec<(SequenceAuthorityId, RequestAuthorityId)>,
        }
        let input = FingerprintInput {
            domain: "ferrum.runtime-vnext.claimed-backing.v1",
            work_fingerprint: work_shape.fingerprint(),
            demand: &demand,
            backing: backing_slices
                .iter()
                .map(LogicalBackingSliceAuthority::evidence)
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
        })
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
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
