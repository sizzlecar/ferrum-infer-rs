//! Exclusive ownership of stable Step and full-plan Invocation allocations.
//!
//! A reservation is not a participant, frame, flight, or submission permission.
//! Every use captures those authorities afresh and remains busy until the fresh
//! Step has retired. The ordinary path remains the fallback for a changed cohort
//! or resource shape.

use super::*;
use crate::vnext::{ReusableExecutionBucketId, ReusableExecutionBucketSpec};

const IDLE: u8 = 0;
const BUSY: u8 = 1;
const REVOKED: u8 = 2;

/// One fresh use; deliberately non-cloneable and stored last in its Step.
pub(super) struct DecodeSessionStepUse {
    state: Arc<AtomicU8>,
    finalized: bool,
}

impl DecodeSessionStepUse {
    fn acquire(state: &Arc<AtomicU8>) -> Result<Self, VNextError> {
        state
            .compare_exchange(IDLE, BUSY, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| invalid_resource("decode session is busy or revoked"))?;
        Ok(Self {
            state: Arc::clone(state),
            finalized: false,
        })
    }

    pub(super) fn mark_finalized(&mut self) {
        self.finalized = true;
    }
}

impl Drop for DecodeSessionStepUse {
    fn drop(&mut self) {
        if self.finalized {
            // Dropping an outer reservation revokes even an otherwise retired use.
            let _ = self
                .state
                .compare_exchange(BUSY, IDLE, Ordering::AcqRel, Ordering::Acquire);
        } else {
            self.state.store(REVOKED, Ordering::Release);
        }
    }
}

pub enum DecodeSessionPrepareDecision<R: DeviceRuntime> {
    Prepared {
        step: Arc<StepResourceLease<R>>,
        wave: PreparedStepSubmissionWave<R>,
    },
    Fallback,
    RequestStateDeferred(RequestStateHazardDeferral),
    RequestStateSplitRequired(RequestStateHazardSplitRequired),
    RequestStatePoisoned(RequestStateHazardPoison),
}

/// Owns capacity and slot occupancy, never an old Step or frame permission.
#[must_use = "release idle reservations when their cohort changes or capacity is needed"]
pub struct DecodeSessionReservation<R: DeviceRuntime> {
    // Child physical/capacity claims must drop before their sequence parents.
    step_backing: ClaimedBackingTransaction,
    wave_backing: ClaimedSubmissionWaveBacking,
    used_pools: Vec<Arc<DynamicBackingPool<R>>>,
    participants: ExecutionBatchParticipants<R>,
    lane: Arc<ExecutionLane<R>>,
    bucket: ReusableExecutionBucketSpec,
    bucket_id: ReusableExecutionBucketId,
    pressure_action: AdmissionPressureAction,
    state: Arc<AtomicU8>,
    prepared_steps: u64,
}

impl<R: DeviceRuntime> Drop for DecodeSessionReservation<R> {
    fn drop(&mut self) {
        // In-flight uses retain their own shared allocation owners and parents.
        // Revocation cannot release their gate or their fence-owned resources.
        self.state.store(REVOKED, Ordering::Release);
    }
}

fn supported_request(request: &StepResourceAdmissionRequest) -> bool {
    request.fit_policy() == AdmissionFitPolicy::ImmediateOnly
        && request.reusable_execution_bucket_id().is_some()
        && !request.work_shape().participants().is_empty()
        && request
            .work_shape()
            .participant_token_ranges()
            .iter()
            .all(|range| range.immediate_tokens() == 1)
}

impl<R: DeviceRuntime> ExecutionBatchParticipants<R> {
    /// Cold reservation. Only this operation evaluates demand, claims the two
    /// resource scopes, binds their certificates and builds the program binding.
    pub fn try_reserve_decode_session(
        &self,
        request: &StepResourceAdmissionRequest,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<Option<DecodeSessionReservation<R>>, VNextError> {
        if !supported_request(request) {
            return Ok(None);
        }
        let step = match self.try_begin_step(request.clone(), lane)? {
            StepResourceAdmissionDecision::Admitted(step) => step,
            _ => return Ok(None),
        };
        // No invocation is prepared in this cold frame. Roll it back after
        // collecting the allocation owners, leaving the first use a fresh frame.
        let result = (|| {
            let plan = &self.sessions[0].resources().request.plan;
            let _lifecycle = plan.resources.read_lifecycle("reserve a decode session")?;
            let bucket = step.reusable_execution_bucket().cloned().ok_or_else(|| {
                invalid_resource("decode session requires a resolved reusable bucket")
            })?;
            let shape = request.immediate_shape();
            let (demand, slices) = plan.submission_wave_demand(
                shape,
                shape,
                Some(&bucket),
                request.fit_policy(),
                request.pressure_action(),
            )?;
            let prepared = match plan.prepare_lane_stable_backing_slices(lane, slices)? {
                LaneBackingPrepareDecision::Prepared(prepared) => prepared,
                LaneBackingPrepareDecision::Deferred(_) => return Ok(None),
            };
            let capacity = if demand.immediate_claim().is_empty() {
                None
            } else {
                let parents = self
                    .sessions
                    .iter()
                    .map(|session| session.resources().logical_lease())
                    .collect::<Vec<_>>();
                match plan
                    .logical_admission()
                    .try_claim_for_sequences(&parents, &demand)?
                {
                    BatchCapacityClaimDecision::Claimed(capacity) => {
                        if !plan
                            .logical_admission()
                            .owns_batch_capacity_claim(&capacity)
                        {
                            return Err(invalid_resource(
                                "decode session received foreign capacity",
                            ));
                        }
                        Some(capacity)
                    }
                    _ => return Ok(None),
                }
            };
            let layout = if plan
                .nodes()
                .iter()
                .any(|node| node.binding_resource().is_some())
            {
                Some(
                    plan.dynamic_pools()
                        .program_binding_layout(bucket.bucket_id())
                        .cloned()
                        .ok_or_else(|| {
                            invalid_resource("decode session has no compiled binding layout")
                        })?,
                )
            } else {
                None
            };
            let wave_backing = ClaimedSubmissionWaveBacking::new(
                plan.plan_hash().clone(),
                plan.nodes().len(),
                plan.nodes().len(),
                Arc::clone(&request.work_shape),
                demand,
                capacity,
                prepared.commit(),
                layout,
            )?;
            let used_pool_ids = step
                .backing_slices()
                .iter()
                .chain(wave_backing.backing_slices())
                .map(|slice| slice.evidence().pool_id().clone())
                .collect::<BTreeSet<_>>();
            let used_pools = used_pool_ids
                .iter()
                .map(|pool_id| {
                    plan.dynamic_pools()
                        .pools
                        .get(pool_id)
                        .cloned()
                        .ok_or_else(|| invalid_resource("decode session backing pool is missing"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            // The participant constructor performs its own lifecycle check;
            // never recursively acquire a read lock while a writer may wait.
            drop(_lifecycle);
            Ok(Some(DecodeSessionReservation {
                step_backing: step
                    .claimed_backing()
                    .borrow_reserved(Arc::clone(&request.work_shape))?,
                wave_backing,
                used_pools,
                participants: ExecutionBatchParticipants::new(self.sessions.clone())?,
                lane: Arc::clone(lane),
                bucket_id: bucket.bucket_id().clone(),
                bucket,
                pressure_action: request.pressure_action(),
                state: Arc::new(AtomicU8::new(IDLE)),
                prepared_steps: 0,
            }))
        })();
        step.try_rollback_unsubmitted()
            .map_err(|failure| failure.error)?;
        result
    }
}

impl<R: DeviceRuntime> DecodeSessionReservation<R> {
    pub fn is_idle(&self) -> bool {
        self.state.load(Ordering::Acquire) == IDLE
    }

    pub const fn prepared_steps(&self) -> u64 {
        self.prepared_steps
    }

    pub fn matches_participants(&self, participants: &ExecutionBatchParticipants<R>) -> bool {
        self.participants.sessions.len() == participants.sessions.len()
            && self
                .participants
                .sessions
                .iter()
                .zip(&participants.sessions)
                .all(|(old, current)| Arc::ptr_eq(old, current))
    }

    pub fn try_prepare_step(
        &mut self,
        request: &StepResourceAdmissionRequest,
    ) -> Result<DecodeSessionPrepareDecision<R>, VNextError> {
        if self.state.load(Ordering::Acquire) != IDLE {
            return Err(invalid_resource(
                "decode session cannot be reused before its prior Step retires",
            ));
        }
        // A changed legal cohort/shape is not an authorization failure. The caller
        // drops this idle reservation before ordinary admission, so it cannot
        // starve that fallback of its own reserved capacity.
        if !supported_request(request)
            || request.reusable_execution_bucket_id() != Some(&self.bucket_id)
            || request.pressure_action() != self.pressure_action
            || request.immediate_shape() != self.step_backing.work_shape().immediate_shape()
            || request.work_shape().participants() != self.step_backing.work_shape().participants()
        {
            return Ok(DecodeSessionPrepareDecision::Fallback);
        }
        let plan = &self.participants.sessions[0].resources().request.plan;
        let _lifecycle = plan.resources.read_lifecycle("borrow a decode session")?;
        // A live child claim does not certify the coordinator's current health.
        // Preserve ordinary admission's poison boundary without claiming again.
        plan.logical_admission().epochs()?;
        if !self.lane.is_reusable() || !Arc::ptr_eq(plan.runtime(), self.lane.runtime_arc()) {
            return Err(invalid_resource(
                "decode session execution lane is no longer reusable",
            ));
        }
        for session in &self.participants.sessions {
            session.ensure_open_identity()?;
            if session.resources().is_poisoned() {
                return Err(invalid_resource("decode session participant is poisoned"));
            }
        }
        // Fresh pool health remains mandatory. Exact current backing identities
        // and each current logical window are additionally checked by dispatch.
        for pool in &self.used_pools {
            if pool
                .state
                .lock()
                .map_err(|_| invalid_resource("decode session backing pool mutex is poisoned"))?
                .poisoned
            {
                return Err(invalid_resource("decode session backing pool is poisoned"));
            }
        }
        let claimed = self
            .step_backing
            .borrow_reserved(Arc::clone(&request.work_shape))?;
        let session_use = DecodeSessionStepUse::acquire(&self.state)?;
        let step = self.participants.prepare_reserved_step(
            &self.lane,
            Some(self.bucket.clone()),
            claimed,
            session_use,
        )?;
        // Full-plan preparation acquires a fresh lifecycle guard itself.
        drop(_lifecycle);
        let decision = step.try_prepare_reserved_full_plan_submission_wave(
            Arc::clone(&request.work_shape),
            request.pressure_action(),
            &self.wave_backing,
        );
        let result = match decision {
            Ok(StepSubmissionWaveAdmissionDecision::Prepared(wave)) => {
                self.prepared_steps = self.prepared_steps.saturating_add(1);
                return Ok(DecodeSessionPrepareDecision::Prepared { step, wave });
            }
            Ok(StepSubmissionWaveAdmissionDecision::RequestStateDeferred(value)) => {
                Ok(DecodeSessionPrepareDecision::RequestStateDeferred(value))
            }
            Ok(StepSubmissionWaveAdmissionDecision::RequestStateSplitRequired(value)) => Ok(
                DecodeSessionPrepareDecision::RequestStateSplitRequired(value),
            ),
            Ok(StepSubmissionWaveAdmissionDecision::RequestStatePoisoned(value)) => {
                Ok(DecodeSessionPrepareDecision::RequestStatePoisoned(value))
            }
            Ok(_) => Err(invalid_resource(
                "reserved decode wave unexpectedly attempted resource admission",
            )),
            Err(error) => Err(error),
        };
        // No full wave was installed; restore the fresh frame through the normal
        // pristine-registry rollback. Failure retains existing poison semantics.
        step.try_rollback_unsubmitted()
            .map_err(|failure| failure.error)?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_session_use_is_exclusive_and_only_finalized_drop_reopens() {
        let state = Arc::new(AtomicU8::new(IDLE));
        let mut first = DecodeSessionStepUse::acquire(&state).unwrap();
        assert!(DecodeSessionStepUse::acquire(&state).is_err());
        first.mark_finalized();
        assert_eq!(state.load(Ordering::Acquire), BUSY);
        drop(first);
        assert_eq!(state.load(Ordering::Acquire), IDLE);
        let second = DecodeSessionStepUse::acquire(&state).unwrap();
        drop(second);
        assert_eq!(state.load(Ordering::Acquire), REVOKED);
        assert!(DecodeSessionStepUse::acquire(&state).is_err());
    }

    #[test]
    fn decode_session_outer_revocation_cannot_reopen_late_use() {
        let state = Arc::new(AtomicU8::new(IDLE));
        let mut active = DecodeSessionStepUse::acquire(&state).unwrap();
        state.store(REVOKED, Ordering::Release);
        active.mark_finalized();
        drop(active);
        assert_eq!(state.load(Ordering::Acquire), REVOKED);
    }
}
