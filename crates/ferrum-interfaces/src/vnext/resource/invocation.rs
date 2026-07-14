use super::*;

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
    claimed_backing: ClaimedBackingTransaction,
    participants: Vec<AdmittedStepParticipant<R>>,
    invocation_registry: Arc<InvocationRegistry>,
    batch_step_id: BatchStepId,
    finalized: bool,
}

impl<R> StepResourceLease<R>
where
    R: DeviceRuntime,
{
    fn new(
        participants: Vec<AdmittedStepParticipant<R>>,
        batch_step_id: BatchStepId,
        claimed_backing: ClaimedBackingTransaction,
    ) -> Result<Self, VNextError> {
        if participants.is_empty() {
            return Err(invalid_resource(
                "step resources require a non-empty participant set",
            ));
        }
        let coordinator = participants[0]
            .session
            .resources()
            .request
            .plan
            .logical_admission();
        if claimed_backing.work_shape().participants().len() != participants.len()
            || claimed_backing
                .work_shape()
                .participants()
                .iter()
                .zip(&participants)
                .any(|(authority, participant)| {
                    authority.sequence_authority() != participant.session.sequence_authority()
                        || authority.request_authority() != participant.session.request_authority()
                })
        {
            return Err(invalid_resource(
                "step work shape differs from its exact batch participants",
            ));
        }
        if let Some(capacity) = claimed_backing.logical_capacity() {
            let parents_match = capacity
                .parents()
                .iter()
                .map(|parent| (parent.sequence(), parent.request()))
                .eq(participants.iter().map(|participant| {
                    (
                        participant.session.sequence_authority(),
                        participant.session.request_authority(),
                    )
                }));
            if !coordinator.owns_batch_capacity_claim(capacity) || !parents_match {
                return Err(invalid_resource(
                    "step capacity authority differs from its exact batch participants",
                ));
            }
        }
        Ok(Self {
            claimed_backing,
            participants,
            invocation_registry: Arc::new(InvocationRegistry::default()),
            batch_step_id,
            finalized: false,
        })
    }

    pub const fn batch_step_id(&self) -> BatchStepId {
        self.batch_step_id
    }

    pub fn try_retire_normal(
        self: Arc<Self>,
    ) -> Result<StepRetirementReceipt, StepFinalizationFailure<R>> {
        Self::try_finalize(self, false)
    }

    pub fn try_abort(self: Arc<Self>) -> Result<StepRetirementReceipt, StepFinalizationFailure<R>> {
        Self::try_finalize(self, true)
    }

    fn try_finalize(
        step: Arc<Self>,
        abort: bool,
    ) -> Result<StepRetirementReceipt, StepFinalizationFailure<R>> {
        let mut step = match Arc::try_unwrap(step) {
            Ok(step) => step,
            Err(step) => {
                return Err(StepFinalizationFailure {
                    step,
                    error: invalid_resource(
                        "step cannot finalize while an invocation or scheduler clone retains it",
                    ),
                });
            }
        };
        let dispositions = match step.finalize_participants(abort) {
            Ok(dispositions) => dispositions,
            Err(error) => {
                return Err(StepFinalizationFailure {
                    step: Arc::new(step),
                    error,
                });
            }
        };
        let participants = step
            .participants
            .iter()
            .zip(dispositions)
            .map(|(participant, disposition)| StepParticipantRetirement {
                assignment: StepParticipantFrameAssignment::new(
                    participant.session.sequence_authority(),
                    participant.session.request_authority(),
                    participant.frame.frame_id,
                ),
                disposition,
            })
            .collect();
        Ok(StepRetirementReceipt {
            batch_step_id: step.batch_step_id,
            participants,
        })
    }

    fn finalize_participants(
        &mut self,
        abort: bool,
    ) -> Result<Vec<StepParticipantRetirementDisposition>, VNextError> {
        if self.finalized {
            return Err(invalid_resource("step resources are already finalized"));
        }
        let mut holds = self
            .participants
            .iter_mut()
            .map(|participant| &mut participant.frame)
            .collect::<Vec<_>>();
        let dispositions = finalize_session_frames(&mut holds, abort)?;
        self.finalized = true;
        Ok(dispositions)
    }

    pub fn participant_count(&self) -> u32 {
        u32::try_from(self.participants.len())
            .expect("step participant count was validated before admission")
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.participants[0].session.resources().coordinator_id()
    }

    pub fn participants(
        &self,
    ) -> impl ExactSizeIterator<Item = &Arc<AdmittedSequenceResources<R>>> {
        self.participants
            .iter()
            .map(|participant| participant.session.resources())
    }

    pub fn participant_frames(
        &self,
    ) -> impl ExactSizeIterator<Item = StepParticipantFrameAssignment> + '_ {
        self.participants.iter().map(|participant| {
            StepParticipantFrameAssignment::new(
                participant.session.sequence_authority(),
                participant.session.request_authority(),
                participant.frame.frame_id,
            )
        })
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        self.claimed_backing.backing_slices()
    }

    pub fn logical_capacity(&self) -> Option<&LogicalBatchCapacityLease> {
        self.claimed_backing.logical_capacity()
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.claimed_backing.work_shape()
    }

    pub fn bind_invocation_work_shape(
        &self,
        mut participant_tokens: Vec<(BatchParticipantAuthority, TokenSpanWork)>,
    ) -> Result<BatchWorkShape, VNextError> {
        participant_tokens.sort_by_key(|(participant, _)| participant.canonical_key());
        if participant_tokens.is_empty()
            || participant_tokens
                .windows(2)
                .any(|pair| pair[0].0.canonical_key() == pair[1].0.canonical_key())
            || participant_tokens.iter().any(|(authority, _)| {
                self.participants
                    .binary_search_by_key(&authority.canonical_key(), |participant| {
                        session_participant_key(&participant.session)
                    })
                    .is_err()
            })
        {
            return Err(invalid_resource(
                "invocation token work must bind a unique non-empty step participant subset",
            ));
        }
        BatchWorkShape::new(
            participant_tokens
                .into_iter()
                .map(|(participant, token_span)| {
                    BatchParticipantTokenSpan::new(participant, token_span)
                })
                .collect(),
        )
    }

    pub fn bind_all_invocation_work_shape(
        &self,
        token_spans: Vec<TokenSpanWork>,
    ) -> Result<BatchWorkShape, VNextError> {
        if token_spans.len() != self.participants.len() {
            return Err(invalid_resource(
                "invocation token work count differs from all step participants",
            ));
        }
        self.bind_invocation_work_shape(
            self.participants
                .iter()
                .zip(token_spans)
                .map(|(participant, token_span)| {
                    (
                        BatchParticipantAuthority::new(
                            participant.session.sequence_authority(),
                            participant.session.request_authority(),
                        ),
                        token_span,
                    )
                })
                .collect(),
        )
    }

    pub fn claimed_backing(&self) -> &ClaimedBackingTransaction {
        &self.claimed_backing
    }

    pub fn static_provisioning(&self) -> Option<&StaticProvisioningLease<R>> {
        self.participants[0]
            .session
            .resources()
            .static_provisioning()
    }

    pub fn plan_evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.participants[0].session.resources().plan_evidence()
    }

    pub(crate) fn backing_view(
        &self,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        if let Some(authority) = self
            .claimed_backing
            .backing_slices()
            .iter()
            .find(|authority| authority.resource_id() == resource_id)
        {
            return self.participants[0]
                .session
                .resources()
                .request
                .plan
                .dynamic_pools()
                .view(authority);
        }
        Err(invalid_resource(format!(
            "resource `{resource_id}` is not step-shared backing"
        )))
    }

    pub(crate) fn participant_backing_views(
        &self,
        resource_id: &ResourceId,
    ) -> Result<Vec<(SequenceAuthorityId, LogicalBackingBufferView<'_, R::Buffer>)>, VNextError>
    {
        self.participants
            .iter()
            .map(|participant| {
                Ok((
                    participant.session.sequence_authority(),
                    participant.session.resources().backing_view(resource_id)?,
                ))
            })
            .collect()
    }

    pub fn try_admit_invocation(
        self: &Arc<Self>,
        request: InvocationResourceAdmissionRequest,
    ) -> Result<InvocationResourceAdmissionDecision<R>, VNextError> {
        let _lifecycle = self.participants[0]
            .session
            .resources()
            .request
            .plan
            .resources
            .read_lifecycle("admit a step invocation")?;
        let InvocationResourceAdmissionRequest {
            node_id,
            work_shape,
            fit_policy,
            pressure_action,
        } = request;
        let immediate_shape = work_shape.immediate_shape();
        let fit_shape = match fit_policy {
            AdmissionFitPolicy::ImmediateOnly => immediate_shape,
            AdmissionFitPolicy::FullInputMustFit => work_shape.fit_shape(),
        };
        let participant_sessions = work_shape
            .participants()
            .iter()
            .map(|authority| {
                self.participants
                    .binary_search_by_key(&authority.canonical_key(), |participant| {
                        session_participant_key(&participant.session)
                    })
                    .map(|index| Arc::clone(&self.participants[index].session))
                    .map_err(|_| {
                        invalid_resource(
                            "invocation participant is not a member of its execution frame",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut participant_frames = Vec::with_capacity(participant_sessions.len());
        let mut flight_candidates = Vec::with_capacity(participant_sessions.len());
        for participant in &participant_sessions {
            let key = session_participant_key(participant);
            let index = self
                .participants
                .binary_search_by_key(&key, |step_participant| {
                    session_participant_key(&step_participant.session)
                })
                .map_err(|_| {
                    invalid_resource("invocation participant lost its execution-frame assignment")
                })?;
            let step_participant = &self.participants[index];
            participant_frames.push(StepParticipantFrameAssignment::new(
                participant.sequence_authority(),
                participant.request_authority(),
                step_participant.frame.frame_id,
            ));
            flight_candidates.push(ParticipantFlightCandidate {
                slot: Arc::clone(&participant.slot),
                epoch: participant.epoch,
                fingerprint: participant.fingerprint.clone(),
                frame: ActiveSequenceFrame {
                    frame_id: step_participant.frame.frame_id,
                    batch_step_id: self.batch_step_id,
                },
                participant: BatchParticipantAuthority::new(
                    participant.sequence_authority(),
                    participant.request_authority(),
                ),
            });
        }
        let participants = participant_sessions
            .iter()
            .map(|participant| Arc::clone(participant.resources()))
            .collect::<Vec<_>>();
        let participant_count = u32::try_from(participants.len())
            .map_err(|_| invalid_resource("invocation participant count exceeds u32"))?;
        if participant_count == 0
            || immediate_shape.sequences() != participant_count
            || fit_shape.sequences() != participant_count
        {
            return Err(invalid_resource(
                "invocation shape sequence count differs from its exact participant set",
            ));
        }
        let plan = &participants[0].request.plan;
        let (demand, requested_slices) = plan.scoped_demand(
            AllocationLifetime::Invocation,
            Some(&node_id),
            immediate_shape,
            fit_shape,
            fit_policy,
            pressure_action,
        )?;
        let prepared = match plan.prepare_backing_slices(requested_slices)? {
            BackingPrepareDecision::Prepared(prepared) => prepared,
            BackingPrepareDecision::Deferred(deferred) => {
                return Ok(InvocationResourceAdmissionDecision::BackingDeferred(
                    deferred,
                ));
            }
        };
        let participant_authorities = participants
            .iter()
            .map(|participant| {
                BatchParticipantAuthority::new(
                    participant.sequence_authority(),
                    participant.request_authority(),
                )
            })
            .collect::<Vec<_>>();
        if work_shape.participants() != participant_authorities {
            return Err(invalid_resource(
                "invocation work authority differs from selected participants",
            ));
        }
        let logical_capacity = if demand.immediate_claim().is_empty() {
            None
        } else {
            let parents = participants
                .iter()
                .map(|participant| &*participant.logical_lease)
                .collect::<Vec<_>>();
            match plan
                .logical_admission()
                .try_claim_for_sequences(&parents, &demand)?
            {
                BatchCapacityClaimDecision::Claimed(capacity) => {
                    let parents_match = capacity
                        .parents()
                        .iter()
                        .map(|parent| (parent.sequence(), parent.request()))
                        .eq(participants.iter().map(|participant| {
                            (
                                participant.sequence_authority(),
                                participant.request_authority(),
                            )
                        }));
                    if !plan
                        .logical_admission()
                        .owns_batch_capacity_claim(&capacity)
                        || !parents_match
                    {
                        return Err(invalid_resource(
                            "invocation admission returned capacity for another participant set",
                        ));
                    }
                    Some(capacity)
                }
                BatchCapacityClaimDecision::Deferred(deferred) => {
                    return Ok(InvocationResourceAdmissionDecision::Deferred(deferred));
                }
                BatchCapacityClaimDecision::PermanentRejected(rejected) => {
                    return Ok(InvocationResourceAdmissionDecision::PermanentRejected(
                        rejected,
                    ));
                }
            }
        };
        let backing_slices = prepared.commit();
        let claimed_backing =
            ClaimedBackingTransaction::new(work_shape, demand, logical_capacity, backing_slices)?;
        let batch_invocation_id = issue_batch_invocation_id()?;
        let prepared_participant_flights =
            prepare_participant_flights(&flight_candidates, &node_id)?;
        let topology_keys = participant_frames
            .iter()
            .map(|assignment| {
                ParticipantNodeKey::new(
                    assignment.participant(),
                    assignment.frame_id(),
                    node_id.clone(),
                )
            })
            .collect();
        let active_invocation = self.invocation_registry.enter(
            topology_keys,
            batch_invocation_id,
            claimed_backing.work_shape().fingerprint(),
        )?;
        Ok(InvocationResourceAdmissionDecision::Admitted(
            InvocationResourceLease::new(
                Arc::clone(self),
                participants,
                participant_frames,
                node_id,
                batch_invocation_id,
                claimed_backing,
                prepared_participant_flights,
                active_invocation,
            )?,
        ))
    }
}

impl<R> Drop for StepResourceLease<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if !self.finalized {
            for participant in &self.participants {
                poison_session_frame(&participant.frame);
            }
        }
    }
}

pub enum InvocationResourceAdmissionDecision<R>
where
    R: DeviceRuntime,
{
    Admitted(InvocationResourceLease<R>),
    Deferred(AdmissionDeferred),
    BackingDeferred(DynamicBackingDeferred),
    PermanentRejected(AdmissionRejected),
}

/// Exact prepared batch node/provider invocation authority. No device command
/// has been submitted at this layer; dropping it performs the typed
/// definitely-not-submitted participant-flight rollback.
#[must_use = "prepared invocation resources must be dispatched or explicitly dropped"]
pub struct InvocationResourceLease<R>
where
    R: DeviceRuntime,
{
    // Claimed backing is returned before participant-flight and parent frame
    // authorities. It retains the immutable work fingerprint even when empty.
    claimed_backing: ClaimedBackingTransaction,
    prepared_participant_flights: Vec<PreparedParticipantFlightHold>,
    active_invocation: ActiveInvocationGuard,
    participants: Vec<Arc<AdmittedSequenceResources<R>>>,
    participant_frames: Vec<StepParticipantFrameAssignment>,
    step: Arc<StepResourceLease<R>>,
    node_id: NodeId,
    batch_invocation_id: BatchInvocationId,
}

impl<R> InvocationResourceLease<R>
where
    R: DeviceRuntime,
{
    fn new(
        step: Arc<StepResourceLease<R>>,
        participants: Vec<Arc<AdmittedSequenceResources<R>>>,
        participant_frames: Vec<StepParticipantFrameAssignment>,
        node_id: NodeId,
        batch_invocation_id: BatchInvocationId,
        claimed_backing: ClaimedBackingTransaction,
        prepared_participant_flights: Vec<PreparedParticipantFlightHold>,
        active_invocation: ActiveInvocationGuard,
    ) -> Result<Self, VNextError> {
        if participants.is_empty() {
            return Err(invalid_resource(
                "invocation resources require a non-empty participant set",
            ));
        }
        if participant_frames.len() != participants.len()
            || prepared_participant_flights.len() != participants.len()
            || participant_frames
                .iter()
                .zip(&participants)
                .any(|(assignment, participant)| {
                    assignment.sequence_authority() != participant.sequence_authority()
                        || assignment.request_authority() != participant.request_authority()
                })
        {
            return Err(invalid_resource(
                "invocation frame mapping differs from its exact participant set",
            ));
        }
        if claimed_backing.work_shape().participants().len() != participants.len()
            || claimed_backing
                .work_shape()
                .participants()
                .iter()
                .zip(&participants)
                .any(|(authority, participant)| {
                    authority.sequence_authority() != participant.sequence_authority()
                        || authority.request_authority() != participant.request_authority()
                })
            || claimed_backing.work_shape().immediate_tokens()
                > step.work_shape().immediate_tokens()
            || claimed_backing.work_shape().immediate_pages() > step.work_shape().immediate_pages()
            || claimed_backing.work_shape().fit_tokens() > step.work_shape().fit_tokens()
            || claimed_backing.work_shape().fit_pages() > step.work_shape().fit_pages()
        {
            return Err(invalid_resource(
                "invocation work shape differs from participants or exceeds its step",
            ));
        }
        if let Some(capacity) = claimed_backing.logical_capacity() {
            let coordinator = participants[0].request.plan.logical_admission();
            let parents_match = capacity
                .parents()
                .iter()
                .map(|parent| (parent.sequence(), parent.request()))
                .eq(participants.iter().map(|participant| {
                    (
                        participant.sequence_authority(),
                        participant.request_authority(),
                    )
                }));
            if !coordinator.owns_batch_capacity_claim(capacity) || !parents_match {
                return Err(invalid_resource(
                    "invocation capacity authority differs from its exact participants",
                ));
            }
        }
        Ok(Self {
            claimed_backing,
            prepared_participant_flights,
            active_invocation,
            participants,
            participant_frames,
            step,
            node_id,
            batch_invocation_id,
        })
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub const fn batch_invocation_id(&self) -> BatchInvocationId {
        self.batch_invocation_id
    }

    pub fn batch_step_id(&self) -> BatchStepId {
        self.step.batch_step_id()
    }

    pub fn participant_count(&self) -> u32 {
        u32::try_from(self.participants.len())
            .expect("invocation participant count was validated before admission")
    }

    pub fn prepared_participant_count(&self) -> u32 {
        u32::try_from(self.prepared_participant_flights.len())
            .expect("prepared participant count was validated at construction")
    }

    pub(crate) fn participant_session_identities(
        &self,
    ) -> impl ExactSizeIterator<Item = (SequenceSessionEpoch, &SequenceSessionFingerprint)> {
        self.prepared_participant_flights
            .iter()
            .map(|hold| (hold.epoch, &hold.fingerprint))
    }

    pub(crate) fn participant_node_keys(&self) -> Vec<ParticipantNodeKey> {
        self.participant_frames
            .iter()
            .map(|assignment| {
                ParticipantNodeKey::new(
                    assignment.participant(),
                    assignment.frame_id(),
                    self.node_id.clone(),
                )
            })
            .collect()
    }

    pub(crate) fn begin_dispatch(&mut self) -> Result<(), VNextError> {
        begin_participant_flights_dispatch(&mut self.prepared_participant_flights)
    }

    pub(crate) fn mark_submission_fence_installed(&mut self) -> Result<(), VNextError> {
        self.active_invocation.mark_in_flight()
    }

    pub(crate) fn definitely_not_submitted(
        mut self,
    ) -> Result<DefinitelyNotSubmittedRetryAuthority<R>, VNextError> {
        self.active_invocation.mark_not_submitted()?;
        reset_participant_flights_after_definitely_not_submitted(
            &mut self.prepared_participant_flights,
        )?;
        let topology_fingerprint = self.retry_topology_fingerprint()?;
        let work_fingerprint = self.work_shape().fingerprint().to_owned();
        let prior_attempt = self.batch_invocation_id;
        Ok(DefinitelyNotSubmittedRetryAuthority {
            invocation: Some(self),
            topology_fingerprint,
            work_fingerprint,
            prior_attempt,
        })
    }

    fn prepare_definitely_not_submitted_retry(
        &mut self,
        fresh_attempt: BatchInvocationId,
        topology_fingerprint: &str,
        work_fingerprint: &str,
    ) -> Result<(), VNextError> {
        if self.retry_topology_fingerprint()? != topology_fingerprint
            || self.work_shape().fingerprint() != work_fingerprint
            || self
                .prepared_participant_flights
                .iter()
                .any(|hold| hold.phase != ParticipantFlightPhase::Prepared)
        {
            return Err(invalid_resource(
                "definitely-not-submitted retry topology or work fingerprint changed",
            ));
        }
        self.active_invocation.prepare_retry(fresh_attempt)?;
        self.batch_invocation_id = fresh_attempt;
        Ok(())
    }

    fn retry_topology_fingerprint(&self) -> Result<String, VNextError> {
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            node_id: &'a NodeId,
            participant_frames: &'a [StepParticipantFrameAssignment],
            work_fingerprint: &'a str,
        }
        let bytes = serde_json::to_vec(&FingerprintInput {
            domain: "ferrum.runtime-vnext.invocation-retry-topology.v1",
            node_id: &self.node_id,
            participant_frames: &self.participant_frames,
            work_fingerprint: self.work_shape().fingerprint(),
        })
        .map_err(|error| {
            invalid_resource(format!(
                "invocation retry topology fingerprint encode failed: {error}"
            ))
        })?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.participants[0].coordinator_id()
    }

    pub fn participants(
        &self,
    ) -> impl ExactSizeIterator<Item = &Arc<AdmittedSequenceResources<R>>> {
        self.participants.iter()
    }

    pub fn participant_frames(&self) -> &[StepParticipantFrameAssignment] {
        &self.participant_frames
    }

    pub fn step_resources(&self) -> &Arc<StepResourceLease<R>> {
        &self.step
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        self.claimed_backing.backing_slices()
    }

    pub fn logical_capacity(&self) -> Option<&LogicalBatchCapacityLease> {
        self.claimed_backing.logical_capacity()
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.claimed_backing.work_shape()
    }

    pub fn claimed_backing(&self) -> &ClaimedBackingTransaction {
        &self.claimed_backing
    }

    pub fn plan_evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.step.plan_evidence()
    }

    pub(crate) fn runtime(&self) -> &Arc<R> {
        self.participants[0].request.plan.runtime()
    }

    pub(crate) fn deferred_cleanup_domain(&self) -> DeferredDeviceCleanupDomainId {
        self.participants[0]
            .request
            .plan
            .resources
            .deferred_cleanup_domain
    }

    pub(crate) fn backing_view(
        &self,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        if let Some(authority) = self
            .claimed_backing
            .backing_slices()
            .iter()
            .find(|authority| authority.resource_id() == resource_id)
        {
            return self.step.participants[0]
                .session
                .resources()
                .request
                .plan
                .dynamic_pools()
                .view(authority);
        }
        self.step.backing_view(resource_id)
    }

    pub(crate) fn participant_backing_views(
        &self,
        resource_id: &ResourceId,
    ) -> Result<Vec<(SequenceAuthorityId, LogicalBackingBufferView<'_, R::Buffer>)>, VNextError>
    {
        self.participants
            .iter()
            .map(|participant| {
                Ok((
                    participant.sequence_authority(),
                    participant.backing_view(resource_id)?,
                ))
            })
            .collect()
    }
}

/// The sole retry edge after a device runtime proves that submit did not
/// happen. It owns the exact invocation, topology and work evidence; dropping
/// it retires the ledger tombstone and cannot be relabeled as retryable later.
#[must_use = "a definitely-not-submitted retry authority must be retried or retired"]
pub struct DefinitelyNotSubmittedRetryAuthority<R>
where
    R: DeviceRuntime,
{
    invocation: Option<InvocationResourceLease<R>>,
    topology_fingerprint: String,
    work_fingerprint: String,
    prior_attempt: BatchInvocationId,
}

impl<R> fmt::Debug for DefinitelyNotSubmittedRetryAuthority<R>
where
    R: DeviceRuntime,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DefinitelyNotSubmittedRetryAuthority")
            .field("prior_attempt", &self.prior_attempt)
            .field("topology_fingerprint", &self.topology_fingerprint)
            .field("work_fingerprint", &self.work_fingerprint)
            .finish_non_exhaustive()
    }
}

impl<R> DefinitelyNotSubmittedRetryAuthority<R>
where
    R: DeviceRuntime,
{
    pub const fn prior_attempt(&self) -> BatchInvocationId {
        self.prior_attempt
    }

    pub fn topology_fingerprint(&self) -> &str {
        &self.topology_fingerprint
    }

    pub fn work_fingerprint(&self) -> &str {
        &self.work_fingerprint
    }

    pub fn retry(mut self) -> Result<InvocationResourceLease<R>, VNextError> {
        let fresh_attempt = issue_batch_invocation_id()?;
        let invocation = self
            .invocation
            .as_mut()
            .ok_or_else(|| invalid_resource("retry authority no longer owns its invocation"))?;
        invocation.prepare_definitely_not_submitted_retry(
            fresh_attempt,
            &self.topology_fingerprint,
            &self.work_fingerprint,
        )?;
        Ok(self
            .invocation
            .take()
            .expect("validated retry authority still owns its invocation"))
    }
}

impl<R> ExecutionBatchParticipants<R>
where
    R: DeviceRuntime,
{
    pub fn try_begin_step(
        &self,
        request: StepResourceAdmissionRequest,
    ) -> Result<StepResourceAdmissionDecision<R>, VNextError> {
        let _lifecycle = self.sessions[0]
            .resources()
            .request
            .plan
            .resources
            .read_lifecycle("begin an execution step")?;
        let StepResourceAdmissionRequest {
            work_shape,
            fit_policy,
            pressure_action,
        } = request;
        let expected_participants = self
            .sessions
            .iter()
            .map(|session| {
                BatchParticipantAuthority::new(
                    session.sequence_authority(),
                    session.request_authority(),
                )
            })
            .collect::<Vec<_>>();
        if work_shape.participants() != expected_participants {
            return Err(invalid_resource(
                "step work authority differs from its exact participant set",
            ));
        }
        let immediate_shape = work_shape.immediate_shape();
        let fit_shape = match fit_policy {
            AdmissionFitPolicy::ImmediateOnly => immediate_shape,
            AdmissionFitPolicy::FullInputMustFit => work_shape.fit_shape(),
        };
        let plan = &self.sessions[0].resources().request.plan;
        let (demand, requested_slices) = plan.scoped_demand(
            AllocationLifetime::Step,
            None,
            immediate_shape,
            fit_shape,
            fit_policy,
            pressure_action,
        )?;
        let prepared = match plan.prepare_backing_slices(requested_slices)? {
            BackingPrepareDecision::Prepared(prepared) => prepared,
            BackingPrepareDecision::Deferred(deferred) => {
                return Ok(StepResourceAdmissionDecision::BackingDeferred(deferred));
            }
        };
        let logical_capacity = if demand.immediate_claim().is_empty() {
            None
        } else {
            let parents = self
                .sessions
                .iter()
                .map(|session| &*session.resources().logical_lease)
                .collect::<Vec<_>>();
            match plan
                .logical_admission()
                .try_claim_for_sequences(&parents, &demand)?
            {
                BatchCapacityClaimDecision::Claimed(capacity) => {
                    let parents_match = capacity
                        .parents()
                        .iter()
                        .map(|parent| (parent.sequence(), parent.request()))
                        .eq(self.sessions.iter().map(|session| {
                            (session.sequence_authority(), session.request_authority())
                        }));
                    if !plan
                        .logical_admission()
                        .owns_batch_capacity_claim(&capacity)
                        || !parents_match
                    {
                        return Err(invalid_resource(
                            "step admission returned capacity for another participant set",
                        ));
                    }
                    Some(capacity)
                }
                BatchCapacityClaimDecision::Deferred(deferred) => {
                    return Ok(StepResourceAdmissionDecision::Deferred(deferred));
                }
                BatchCapacityClaimDecision::PermanentRejected(rejected) => {
                    return Ok(StepResourceAdmissionDecision::PermanentRejected(rejected));
                }
            }
        };
        let backing_slices = prepared.commit();
        let claimed_backing =
            ClaimedBackingTransaction::new(work_shape, demand, logical_capacity, backing_slices)?;
        let batch_step_id = issue_batch_step_id()?;
        let candidates = session_frame_candidates(&self.sessions);
        let frames = acquire_session_frames(&candidates, batch_step_id)?;
        let participants = self
            .sessions
            .iter()
            .cloned()
            .zip(frames)
            .map(|(session, frame)| AdmittedStepParticipant { frame, session })
            .collect();
        Ok(StepResourceAdmissionDecision::Admitted(Arc::new(
            StepResourceLease::new(participants, batch_step_id, claimed_backing)?,
        )))
    }
}

pub(super) const fn sequence_slot_active(epoch: u64) -> u64 {
    (epoch << 2) | 1
}

pub(super) const fn sequence_slot_poisoned_drained(epoch: u64) -> u64 {
    (epoch << 2) | 2
}

pub(super) const fn sequence_slot_poisoned_undrained(epoch: u64) -> u64 {
    (epoch << 2) | 3
}

pub(super) const fn sequence_slot_is_poisoned(state: u64) -> bool {
    matches!(state & 3, 2 | 3)
}

pub(super) const SEQUENCE_DISPATCH_POISONED_BIT: u64 = 1 << 63;
const SEQUENCE_DISPATCH_COUNT_MASK: u64 = SEQUENCE_DISPATCH_POISONED_BIT - 1;

pub(super) fn sequence_dispatch_is_poisoned(gate: &AtomicU64) -> bool {
    gate.load(Ordering::Acquire) & SEQUENCE_DISPATCH_POISONED_BIT != 0
}

struct SequenceDispatchGuard<'a> {
    gate: &'a AtomicU64,
}

impl Drop for SequenceDispatchGuard<'_> {
    fn drop(&mut self) {
        let previous = self.gate.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous & SEQUENCE_DISPATCH_COUNT_MASK > 0);
    }
}

fn enter_sequence_dispatch(gate: &AtomicU64) -> Result<SequenceDispatchGuard<'_>, VNextError> {
    gate.fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
        if state & SEQUENCE_DISPATCH_POISONED_BIT != 0
            || state & SEQUENCE_DISPATCH_COUNT_MASK == SEQUENCE_DISPATCH_COUNT_MASK
        {
            None
        } else {
            Some(state + 1)
        }
    })
    .map_err(|state| {
        if state & SEQUENCE_DISPATCH_POISONED_BIT != 0 {
            invalid_resource("poisoned resource pool cannot dispatch another operation")
        } else {
            invalid_resource("resource pool dispatch counter is exhausted")
        }
    })?;
    Ok(SequenceDispatchGuard { gate })
}

impl<R> AdmittedSequenceResources<R>
where
    R: DeviceRuntime,
{
    fn validate_runtime(&self, context: &'static str) -> Result<(), VNextError> {
        let descriptor = self.request.plan.runtime().descriptor();
        descriptor.validate()?;
        if descriptor.id != *self.request.plan.device_id()
            || descriptor.runtime_implementation_fingerprint
                != self.request.plan.runtime_implementation_fingerprint()
        {
            return Err(invalid_resource(format!(
                "{context} runtime differs from the trusted plan/runtime binding"
            )));
        }
        Ok(())
    }

    pub fn create_execution_stream(
        self: &Arc<Self>,
    ) -> Result<BoundExecutionStream<R>, ExecutionStreamCreationError<R::Error>> {
        let _lifecycle = self
            .request
            .plan
            .resources
            .read_lifecycle("create an execution stream")
            .map_err(ExecutionStreamCreationError::Contract)?;
        if self.is_poisoned() {
            return Err(ExecutionStreamCreationError::Contract(invalid_resource(
                "poisoned logical sequence cannot create an execution stream",
            )));
        }
        self.validate_runtime("execution stream creation preflight")
            .map_err(ExecutionStreamCreationError::Contract)?;
        let stream = self
            .request
            .plan
            .runtime()
            .create_stream()
            .map_err(ExecutionStreamCreationError::Runtime)?;
        self.validate_runtime("execution stream creation completion")
            .map_err(ExecutionStreamCreationError::Contract)?;
        if self.request.plan.runtime().stream_state(&stream) != StreamState::Ready {
            return Err(ExecutionStreamCreationError::Contract(invalid_resource(
                "new execution stream is not ready",
            )));
        }
        Ok(BoundExecutionStream {
            runtime: Arc::clone(self.request.plan.runtime()),
            coordinator_id: self.coordinator_id(),
            sequence_authority: self.sequence_authority(),
            stream: Some(stream),
            state: BoundExecutionStreamState::Ready,
            sequence_recovery: Arc::clone(&self.sequence_recovery),
            sequence_dispatch_gate: Arc::clone(&self.sequence_dispatch_gate),
            abandoned_sequence: None,
            resources: Arc::clone(self),
        })
    }

    pub fn activate<'resources, 'exec>(
        &'resources self,
        stream: &'exec mut BoundExecutionStream<R>,
    ) -> Result<ActiveSequencePermit<'resources, 'exec, R>, VNextError> {
        let _lifecycle = self
            .request
            .plan
            .resources
            .read_lifecycle("activate an execution stream")?;
        if self.is_poisoned() {
            return Err(invalid_resource(
                "poisoned logical sequence cannot be activated",
            ));
        }
        self.validate_runtime("logical sequence activation")?;
        if !Arc::ptr_eq(self.request.plan.runtime(), &stream.runtime)
            || !std::ptr::eq(self, Arc::as_ref(&stream.resources))
            || stream.coordinator_id != self.coordinator_id()
            || stream.sequence_authority != self.sequence_authority()
            || !Arc::ptr_eq(&self.sequence_recovery, &stream.sequence_recovery)
            || !Arc::ptr_eq(&self.sequence_dispatch_gate, &stream.sequence_dispatch_gate)
        {
            return Err(invalid_resource(
                "execution stream belongs to another logical sequence authority",
            ));
        }
        if stream.state != BoundExecutionStreamState::Ready
            || stream.abandoned_sequence.is_some()
            || self.request.plan.runtime().stream_state(stream.stream()) != StreamState::Ready
        {
            return Err(invalid_resource(
                "logical sequence activation requires one core-ready stream",
            ));
        }
        let mut authority_source = self.lock_authority_source()?;
        let selecting_legacy = match *authority_source {
            SequenceExecutionAuthoritySource::Unselected => true,
            SequenceExecutionAuthoritySource::LegacyStream => false,
            SequenceExecutionAuthoritySource::SequenceSession => {
                return Err(invalid_resource(
                    "logical sequence execution authority is permanently selected for sequence sessions",
                ));
            }
            SequenceExecutionAuthoritySource::FailClosed => {
                return Err(invalid_resource(
                    "logical sequence execution authority selector is fail-closed",
                ));
            }
        };
        let epoch = match self.next_activation_epoch.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |epoch| epoch.checked_add(1).filter(|next| *next <= (u64::MAX >> 2)),
        ) {
            Ok(epoch) => epoch,
            Err(_) => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(invalid_resource("active sequence epoch space is exhausted"));
            }
        };
        let active_state = sequence_slot_active(epoch);
        if let Err(actual) =
            self.state
                .compare_exchange(0, active_state, Ordering::AcqRel, Ordering::Acquire)
        {
            if selecting_legacy {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
            }
            return Err(if sequence_slot_is_poisoned(actual) {
                invalid_resource("logical sequence was abandoned and is poisoned")
            } else {
                invalid_resource("logical sequence already owns an active stream")
            });
        }
        let slot = self.sequence_authority().sparse_id();
        let recovery_metadata = AbandonedSequenceMetadata {
            plan: self.request.plan.evidence(),
            sequence_authority: self.sequence_authority(),
            run_id: self.run_id().clone(),
            request_id: self.request_id().clone(),
            slot,
            activation_epoch: epoch,
            runtime_implementation_fingerprint: self
                .request
                .plan
                .runtime_implementation_fingerprint()
                .to_owned(),
            state: Arc::clone(&self.state),
            sequence_dispatch_gate: Arc::clone(&self.sequence_dispatch_gate),
            drained: false,
        };
        let recovery_key = recovery_metadata.key();
        self.sequence_recovery.register(recovery_metadata);
        stream.abandoned_sequence = Some(recovery_key);
        stream.state = BoundExecutionStreamState::InUse;
        *authority_source = SequenceExecutionAuthoritySource::LegacyStream;
        Ok(ActiveSequencePermit {
            resources: self,
            epoch,
            state: Arc::clone(&self.state),
            stream,
            runtime_fingerprint: self
                .request
                .plan
                .runtime_implementation_fingerprint()
                .to_owned(),
            stream_drained: false,
            completed: false,
        })
    }

    pub fn recover_abandoned_sequence(
        &self,
    ) -> Result<ActiveSequenceAbortReceipt, AbandonedSequenceRecoveryError<R::Error>> {
        self.sequence_recovery.recover(
            self.request.plan.runtime(),
            self.sequence_authority().sparse_id(),
        )
    }
}

/// Non-cloneable guard for an admitted active-sequence slot. Dispatch borrows
/// this permit; the sequence owner retains it until all asynchronous work is
/// synchronized or cancelled.
#[must_use = "an active sequence permit must live until asynchronous work is complete"]
pub struct ActiveSequencePermit<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    resources: &'resources AdmittedSequenceResources<R>,
    epoch: u64,
    state: Arc<AtomicU64>,
    stream: &'exec mut BoundExecutionStream<R>,
    runtime_fingerprint: String,
    stream_drained: bool,
    completed: bool,
}

impl<'resources, 'exec, R> ActiveSequencePermit<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    pub fn resources(&self) -> &'resources AdmittedSequenceResources<R> {
        self.resources
    }

    pub fn run_id(&self) -> &RunId {
        self.resources.run_id()
    }

    pub fn request_id(&self) -> &RequestIdentity {
        self.resources.request_id()
    }

    pub fn sequence_authority(&self) -> SequenceAuthorityId {
        self.resources.sequence_authority()
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.resources.coordinator_id()
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        self.resources.backing_slices()
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_fingerprint
    }

    pub(crate) fn with_runtime_and_stream<T>(
        &mut self,
        action: impl FnOnce(&R, &mut R::Stream) -> T,
    ) -> Result<T, VNextError> {
        if self.stream.state != BoundExecutionStreamState::InUse {
            return Err(invalid_resource(
                "operation dispatch requires one core-owned in-use stream",
            ));
        }
        let _dispatch_guard = enter_sequence_dispatch(&self.resources.sequence_dispatch_gate)?;
        Ok(action(
            self.resources.request.plan.runtime(),
            self.stream.stream_mut(),
        ))
    }

    /// Consumes dispatch authority before draining the exact bound stream.
    /// Successful synchronization returns a different typestate that cannot
    /// be passed back to `OperationDispatch`.
    pub fn synchronize(
        mut self,
    ) -> Result<
        SynchronizedSequencePermit<'resources, 'exec, R>,
        SequenceSynchronizationFailure<'resources, 'exec, R>,
    > {
        let preflight = self
            .resources
            .validate_runtime("sequence synchronization preflight")
            .and_then(|()| {
                if self
                    .resources
                    .request
                    .plan
                    .runtime()
                    .descriptor()
                    .runtime_implementation_fingerprint
                    == self.runtime_fingerprint
                {
                    Ok(())
                } else {
                    Err(invalid_resource(
                        "sequence synchronization runtime differs from its activation snapshot",
                    ))
                }
            });

        // Draining is attempted even when descriptor validation fails. The
        // stream/runtime pair is privately bound, while skipping the drain
        // could make later buffer quarantine unsafe.
        let runtime_error = match self
            .resources
            .request
            .plan
            .runtime()
            .synchronize(self.stream.stream_mut())
        {
            Ok(()) => None,
            Err(error) => Some(error),
        };
        let stream_ready = self
            .resources
            .request
            .plan
            .runtime()
            .stream_state(self.stream.stream())
            == StreamState::Ready;
        self.stream_drained = runtime_error.is_none() && stream_ready;
        if self.stream_drained {
            self.stream
                .sequence_recovery
                .set_drained((self.sequence_authority().sparse_id(), self.epoch), true);
        }
        let completion = self
            .resources
            .validate_runtime("sequence synchronization completion")
            .and_then(|()| {
                if stream_ready {
                    Ok(())
                } else {
                    Err(invalid_resource(
                        "sequence synchronization did not return the bound stream to ready",
                    ))
                }
            });
        let error = preflight
            .err()
            .map(SequenceSynchronizationError::Contract)
            .or_else(|| runtime_error.map(SequenceSynchronizationError::Runtime))
            .or_else(|| completion.err().map(SequenceSynchronizationError::Contract));
        if let Some(error) = error {
            return Err(SequenceSynchronizationFailure {
                permit: Some(self),
                error,
            });
        }
        self.stream.state = BoundExecutionStreamState::Ready;
        Ok(SynchronizedSequencePermit { permit: Some(self) })
    }
}

#[derive(Debug)]
pub enum SequenceSynchronizationError<E> {
    Contract(VNextError),
    Runtime(E),
}

/// Retry owner for a failed stream drain. It intentionally does not expose
/// the active dispatch permit, so no operation can be submitted between a
/// failed synchronization attempt and its retry.
#[must_use = "failed sequence synchronization must be retried or retained"]
pub struct SequenceSynchronizationFailure<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    permit: Option<ActiveSequencePermit<'resources, 'exec, R>>,
    error: SequenceSynchronizationError<R::Error>,
}

impl<R> fmt::Debug for SequenceSynchronizationFailure<'_, '_, R>
where
    R: DeviceRuntime,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SequenceSynchronizationFailure")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<'resources, 'exec, R> SequenceSynchronizationFailure<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    pub fn error(&self) -> &SequenceSynchronizationError<R::Error> {
        &self.error
    }

    pub fn retry(
        mut self,
    ) -> Result<
        SynchronizedSequencePermit<'resources, 'exec, R>,
        SequenceSynchronizationFailure<'resources, 'exec, R>,
    > {
        self.permit
            .take()
            .expect("synchronization failure owns its active permit")
            .synchronize()
    }
}

/// Stream-drained typestate. It has no dispatch API and must choose exactly
/// one terminal slot disposition.
#[must_use = "a synchronized sequence must be completed or aborted"]
pub struct SynchronizedSequencePermit<'resources, 'exec, R>
where
    R: DeviceRuntime,
{
    permit: Option<ActiveSequencePermit<'resources, 'exec, R>>,
}

impl<R> SynchronizedSequencePermit<'_, '_, R>
where
    R: DeviceRuntime,
{
    pub fn complete(mut self) -> Result<ActiveSequenceCompletionReceipt, VNextError> {
        let mut permit = self
            .permit
            .take()
            .expect("synchronized sequence owns its active permit");
        let sequence_poisoned =
            sequence_dispatch_is_poisoned(&permit.resources.sequence_dispatch_gate);
        let terminal_state = if sequence_poisoned {
            sequence_slot_poisoned_drained(permit.epoch)
        } else {
            0
        };
        permit
            .state
            .compare_exchange(
                sequence_slot_active(permit.epoch),
                terminal_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| invalid_resource("active sequence epoch is no longer completable"))?;
        permit
            .stream
            .sequence_recovery
            .clear((permit.sequence_authority().sparse_id(), permit.epoch));
        permit.stream.abandoned_sequence = None;
        permit.stream.state = BoundExecutionStreamState::Ready;
        permit.completed = true;
        if sequence_poisoned {
            return Err(invalid_resource(
                "sequence cannot complete successfully after its dispatch authority was poisoned",
            ));
        }
        Ok(ActiveSequenceCompletionReceipt {
            plan: permit.resources.request.plan.evidence(),
            sequence_authority: permit.sequence_authority(),
            run_id: permit.run_id().clone(),
            request_id: permit.request_id().clone(),
            activation_epoch: permit.epoch,
            runtime_implementation_fingerprint: permit.runtime_fingerprint.clone(),
        })
    }

    /// Produces abort evidence only after the exact bound stream was drained.
    /// Only this exact logical sequence remains poisoned after abort.
    pub fn abort(mut self) -> Result<ActiveSequenceAbortReceipt, VNextError> {
        let mut permit = self
            .permit
            .take()
            .expect("synchronized sequence owns its active permit");
        permit
            .state
            .compare_exchange(
                sequence_slot_active(permit.epoch),
                sequence_slot_poisoned_drained(permit.epoch),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| invalid_resource("active sequence epoch is no longer abortable"))?;
        permit
            .resources
            .sequence_dispatch_gate
            .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
        permit
            .stream
            .sequence_recovery
            .clear((permit.sequence_authority().sparse_id(), permit.epoch));
        permit.stream.abandoned_sequence = None;
        permit.stream.state = BoundExecutionStreamState::Ready;
        permit.completed = true;
        Ok(ActiveSequenceAbortReceipt {
            plan: permit.resources.request.plan.evidence(),
            sequence_authority: permit.sequence_authority(),
            run_id: permit.run_id().clone(),
            request_id: permit.request_id().clone(),
            activation_epoch: permit.epoch,
            runtime_implementation_fingerprint: permit.runtime_fingerprint.clone(),
            disposition: ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
        })
    }
}

/// Core-signed evidence that synchronization succeeded and the exact active
/// slot epoch was atomically cleared. It is trusted output and deliberately
/// cannot be deserialized or constructed by a caller.
#[derive(Debug, Serialize)]
#[must_use = "sequence completion evidence must be recorded by execution"]
pub struct ActiveSequenceCompletionReceipt {
    plan: TrustedPlanRuntimeEvidence,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
    activation_epoch: u64,
    runtime_implementation_fingerprint: String,
}

impl ActiveSequenceCompletionReceipt {
    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }
}

/// Terminal resource disposition produced by an explicit sequence abort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ActiveSequenceAbortDisposition {
    SynchronizedAndPoisoned,
    SequenceSessionTerminalized,
}

/// Core-signed evidence that the exact active slot epoch was atomically
/// poisoned. This type is trusted output and has no deserialization or public
/// construction path.
#[derive(Debug, Serialize)]
#[must_use = "sequence abort evidence must be recorded by execution"]
pub struct ActiveSequenceAbortReceipt {
    pub(super) plan: TrustedPlanRuntimeEvidence,
    pub(super) sequence_authority: SequenceAuthorityId,
    pub(super) run_id: RunId,
    pub(super) request_id: RequestIdentity,
    pub(super) activation_epoch: u64,
    pub(super) runtime_implementation_fingerprint: String,
    pub(super) disposition: ActiveSequenceAbortDisposition,
}

impl ActiveSequenceAbortReceipt {
    pub fn plan(&self) -> &TrustedPlanRuntimeEvidence {
        &self.plan
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn activation_epoch(&self) -> u64 {
        self.activation_epoch
    }

    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }

    pub const fn disposition(&self) -> ActiveSequenceAbortDisposition {
        self.disposition
    }
}

impl<R> Drop for ActiveSequencePermit<'_, '_, R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if !self.completed {
            let poisoned_state = if self.stream_drained {
                sequence_slot_poisoned_drained(self.epoch)
            } else {
                sequence_slot_poisoned_undrained(self.epoch)
            };
            let result = self.state.compare_exchange(
                sequence_slot_active(self.epoch),
                poisoned_state,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            debug_assert!(result.is_ok(), "active sequence slot guard lost ownership");
            if result.is_ok() {
                self.resources
                    .sequence_dispatch_gate
                    .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
                self.stream.state = BoundExecutionStreamState::Poisoned;
                self.stream.sequence_recovery.set_drained(
                    (self.sequence_authority().sparse_id(), self.epoch),
                    self.stream_drained,
                );
            }
        }
    }
}
