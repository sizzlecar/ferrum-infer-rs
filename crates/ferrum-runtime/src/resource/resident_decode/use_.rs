use super::*;
use crate::{translate_step_participant_readback_range, AllocationKind};

impl<R: DeviceRuntime> ResidentDecodeUse<R> {
    pub fn batch_step_id(&self) -> BatchStepId {
        self.frame.step_id
    }
    pub fn batch_invocation_id(&self) -> BatchInvocationId {
        self.attempt
    }
    pub fn execution_lane_id(&self) -> ExecutionLaneId {
        self.frame.reservation.lane.id()
    }
    pub fn execution_lane(&self) -> &Arc<ExecutionLane<R>> {
        &self.frame.reservation.lane
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub(crate) fn belongs_to_reservation(
        &self,
        reservation: &Arc<ResidentDecodeReservation<R>>,
    ) -> bool {
        Arc::ptr_eq(&self.frame.reservation, reservation)
    }
    pub fn work_shape(&self) -> &Arc<BatchWorkShape> {
        &self.frame.work
    }
    pub fn plan_evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.frame.participants[0]
            .session
            .resources()
            .plan_evidence()
    }
    pub fn participant_frames(&self) -> Vec<StepParticipantFrameAssignment> {
        self.frame
            .participants
            .iter()
            .map(|p| {
                StepParticipantFrameAssignment::new(
                    p.session.sequence_authority(),
                    p.session.request_authority(),
                    p.frame.frame_id,
                )
            })
            .collect()
    }
    pub fn participant_sessions(&self) -> impl ExactSizeIterator<Item = &Arc<SequenceSession<R>>> {
        self.frame.participants.iter().map(|p| &p.session)
    }
    pub fn backing_snapshots(
        &self,
    ) -> impl ExactSizeIterator<Item = &Arc<SequenceBackingSnapshot<R>>> {
        self.frame.participants.iter().map(|p| &p.backing_snapshot)
    }
    pub(crate) fn sequence_physical_tables(
        &self,
    ) -> &[Arc<SequencePhysicalBindingTable<R::Buffer>>] {
        self.frame
            .tables
            .get()
            .expect("fresh resident use has checked tables")
    }
    pub fn step_backing(&self) -> &ClaimedBackingTransaction {
        &self.frame.reservation.step_backing
    }
    pub fn lane_stable_slot_identity(&self) -> Option<LaneStableArenaSlotIdentity> {
        self.step_backing().lane_stable_slot_identity()
    }
    pub fn claimed_backing(&self) -> &ClaimedSubmissionWaveBacking {
        &self.frame.reservation.wave_backing
    }
    pub fn reusable_execution_bucket(&self) -> &ReusableExecutionBucketSpec {
        &self.frame.reservation.bucket
    }
    pub fn node_count(&self) -> usize {
        self.frame.reservation.full_nodes.len()
    }
    pub(crate) fn runtime(&self) -> &Arc<R> {
        self.frame.reservation.lane.runtime_arc()
    }
    pub(crate) fn deferred_cleanup_domain(&self) -> DeferredDeviceCleanupDomainId {
        self.frame.participants[0]
            .session
            .resources()
            .request
            .plan
            .resources
            .deferred_cleanup_domain
    }
    pub(crate) fn retention(&self) -> DeviceBufferRetention {
        DeviceBufferRetention::pair(Arc::clone(&self.frame), Arc::clone(&self.frame.reservation))
    }
    pub(crate) fn retain_resident_program(
        &self,
        retention: DeviceBufferRetention,
    ) -> Result<(), VNextError> {
        let mut stored = self
            .frame
            .program_retention
            .lock()
            .map_err(|_| invalid_resource("resident program retention poisoned"))?;
        if let Some(previous) = stored.as_ref() {
            return if previous.same_owners(&retention) {
                Ok(())
            } else {
                Err(invalid_resource(
                    "resident use already owns a different sealed program",
                ))
            };
        }
        *stored = Some(retention);
        Ok(())
    }
    pub(crate) fn dynamic_descriptor(
        &self,
        resource: &ResourceId,
    ) -> Result<&DynamicResourceDescriptor, VNextError> {
        let pools = self.frame.participants[0]
            .session
            .resources()
            .request
            .plan
            .dynamic_pools();
        let mut found = pools.domains.iter().filter_map(|d| {
            d.descriptors
                .binary_search_by(|x| x.base_resource_id().cmp(resource))
                .ok()
                .map(|i| &d.descriptors[i])
        });
        let value = found
            .next()
            .ok_or_else(|| invalid_resource("resident resource has no dynamic descriptor"))?;
        if found.next().is_some() {
            return Err(invalid_resource(
                "resident resource has duplicate descriptors",
            ));
        }
        Ok(value)
    }
    fn validate_readback_node(&self, node: &NodeId, participant: usize) -> Result<(), VNextError> {
        let plan = &self.frame.participants[0].session.resources().request.plan;
        if participant >= self.frame.participants.len()
            || !plan.nodes().iter().any(|n| n.id() == node)
        {
            return Err(invalid_resource(
                "resident readback node/participant is absent",
            ));
        }
        Ok(())
    }
    pub(crate) fn backing_view(
        &self,
        node: &NodeId,
        participant: usize,
        resource: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        self.validate_readback_node(node, participant)?;
        let pools = self.frame.participants[0]
            .session
            .resources()
            .request
            .plan
            .dynamic_pools();
        if let Some(slice) = self
            .claimed_backing()
            .backing_slices()
            .iter()
            .chain(self.step_backing().backing_slices())
            .find(|s| s.resource_id() == resource)
        {
            return pools.view(slice);
        }
        let p = &self.frame.participants[participant];
        let slices = p.backing_snapshot.backing_slices_for(resource);
        if !slices.is_empty() {
            return pools.view_many(slices);
        }
        p.session.resources().request.backing_view(resource)
    }
    pub(crate) fn readback_range(
        &self,
        node: &NodeId,
        participant: usize,
        resource: &ResourceId,
        semantic: Range<u64>,
    ) -> Result<Range<u64>, VNextError> {
        self.validate_readback_node(node, participant)?;
        let descriptor = self.dynamic_descriptor(resource)?;
        if descriptor.lifetime() == AllocationLifetime::Step
            && descriptor.kind() == &AllocationKind::Value
        {
            translate_step_participant_readback_range(
                descriptor.demand(),
                &self.frame.work,
                participant,
                semantic,
            )
        } else {
            Ok(semantic)
        }
    }
    pub fn set_terminal_readbacks(
        &mut self,
        request: CompletionReadbackBatchRequest,
    ) -> Result<(), VNextError> {
        if self.terminal_request.is_some() {
            return Err(invalid_resource(
                "resident terminal readbacks already declared",
            ));
        }
        self.terminal_request = Some(request);
        Ok(())
    }
    pub(crate) fn terminal_readback_request(&self) -> Option<&CompletionReadbackBatchRequest> {
        self.terminal_request.as_ref()
    }
    pub(crate) fn has_prepared_terminal_readbacks(&self) -> bool {
        self.terminal_payload.is_some()
    }
    pub(crate) fn install_terminal_readback_payload(
        &mut self,
        payload: Arc<dyn Any + Send + Sync>,
    ) {
        assert!(self.terminal_payload.is_none());
        self.terminal_payload = Some(payload);
    }
    pub(crate) fn take_terminal_readbacks(
        &mut self,
    ) -> Option<(CompletionReadbackBatchRequest, Arc<dyn Any + Send + Sync>)> {
        self.terminal_payload.take().map(|p| {
            (
                self.terminal_request
                    .take()
                    .expect("prepared resident packet has declaration"),
                p,
            )
        })
    }
    pub(crate) fn begin_dispatch(&mut self) -> Result<(), VNextError> {
        if let Some(initializations) = &self.initializations {
            initializations.ensure_wave(&self.fingerprint)?;
        } else {
            self.initializations = Some(PreparedBackingInitializations::prepare_resident(
                self.frame.participants.iter().map(|p| {
                    (
                        BatchParticipantAuthority::new(
                            p.session.sequence_authority(),
                            p.session.request_authority(),
                        ),
                        p.backing_snapshot.backing_slices(),
                    )
                }),
                &self.fingerprint,
            )?);
        }
        begin_submission_wave_participant_flights_dispatch(&mut self.flights)
    }
    pub(crate) fn encode_backing_initializations(
        &self,
        runtime: &R,
        commands: &mut DeviceCommandBatch<R::Command>,
    ) -> Result<usize, BackingInitializationEncodeError<R::Error>> {
        self.initializations
            .as_ref()
            .ok_or_else(|| {
                BackingInitializationEncodeError::Contract(invalid_resource(
                    "resident initialization not prepared",
                ))
            })?
            .encode_in_pools(
                self.frame.participants[0]
                    .session
                    .resources()
                    .request
                    .plan
                    .dynamic_pools(),
                runtime,
                commands,
            )
    }
    pub(crate) fn mark_submission_fence_installed(&mut self) -> Result<(), VNextError> {
        self.frame.submitted.store(true, Ordering::Release);
        self.initializations
            .as_mut()
            .ok_or_else(|| invalid_resource("resident initialization not prepared"))?
            .mark_in_flight()?;
        self.active_wave.mark_in_flight()?;
        if let Some(hazard) = &mut self.hazards {
            hazard.mark_submission_fence_installed()?;
        }
        Ok(())
    }
    pub(crate) fn mark_submission_indeterminate(&mut self) {
        self.frame.submitted.store(true, Ordering::Release);
        if let Some(initializations) = &mut self.initializations {
            initializations.mark_indeterminate();
        }
        if let Some(hazard) = &mut self.hazards {
            hazard.mark_submission_indeterminate();
        }
    }
    pub(crate) fn finish_backing_initializations(
        &mut self,
        succeeded: bool,
    ) -> Result<(), VNextError> {
        self.initializations
            .as_mut()
            .ok_or_else(|| invalid_resource("resident initialization not prepared"))?
            .finish(succeeded)
    }
    pub(crate) fn finish_request_state_hazards(
        &mut self,
        disposition: RequestStateHazardTerminalDisposition,
    ) -> Result<(), VNextError> {
        if let Some(hazard) = &mut self.hazards {
            hazard.finish(disposition)?;
        }
        Ok(())
    }
    pub(crate) fn definitely_not_submitted(mut self) -> Result<Self, VNextError> {
        self.active_wave.mark_not_submitted()?;
        reset_submission_wave_participant_flights_after_definitely_not_submitted(
            &mut self.flights,
        )?;
        self.terminal_payload.take();
        let attempt = issue_batch_invocation_id()?;
        self.active_wave.prepare_retry(attempt)?;
        self.attempt = attempt;
        // Keep the exact work/physical topology fingerprint used by initialization
        // and the registry; BatchInvocationId distinguishes the new attempt.
        Ok(self)
    }
    pub(crate) fn record_completed_boundary(
        &self,
        seal: &SuccessfulWaveCompletionSeal,
    ) -> Result<(), VNextError> {
        if seal.batch_step_id() != self.batch_step_id()
            || seal.batch_invocation_id() != self.attempt
            || seal.lane_id() != self.execution_lane_id()
            || seal.wave_fingerprint() != self.fingerprint
            || self.frame.finalized
            || !self.frame.submitted.load(Ordering::Acquire)
        {
            return Err(invalid_resource(
                "resident success seal does not match its fresh full-plan use",
            ));
        }
        let mut slot = self
            .frame
            .completed_boundary
            .lock()
            .map_err(|_| invalid_resource("resident completed boundary poisoned"))?;
        if !matches!(*slot, StepCompletedBoundarySlot::Empty) {
            *slot = StepCompletedBoundarySlot::Invalid;
            return Err(invalid_resource("resident success recorded twice"));
        }
        *slot = StepCompletedBoundarySlot::Invalid;
        let proof = validate_completed_participants(
            self.claimed_backing().plan_hash(),
            self.batch_step_id(),
            self.attempt,
            &self.frame.work,
            &self.frame.participants,
        )?;
        *slot = StepCompletedBoundarySlot::Succeeded(proof);
        Ok(())
    }
}
