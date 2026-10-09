//! Diagnostic copies of captured claims, never execution authority or live views.
use super::*;
use crate::vnext::{
    BackingSegment, DeviceReusableExecutionProgramId, DynamicBackingPoolId,
    LaneStableArenaSlotIdentity, PlanHash, RequestAuthorityId,
};

#[derive(PartialEq, Eq)]
struct CapturedExtent {
    resource: ResourceId,
    pool: DynamicBackingPoolId,
    pool_instance: u64,
    segments: Vec<BackingSegment>,
    offset: u64,
    capacity: u64,
    physical_size: u64,
}

fn extents(slices: &[LogicalBackingSliceAuthority]) -> Vec<CapturedExtent> {
    slices
        .iter()
        .map(|slice| {
            let evidence = slice.evidence();
            CapturedExtent {
                resource: slice.resource_id().clone(),
                pool: evidence.pool_id().clone(),
                pool_instance: evidence.pool_instance_id(),
                segments: evidence.segments().to_vec(),
                offset: evidence.physical_offset_bytes(),
                capacity: evidence.capacity_size_bytes(),
                physical_size: evidence.physical_size_bytes(),
            }
        })
        .collect()
}

/// A bounded, owned diagnostic snapshot. No Serialize/Debug implementation:
/// request/sequence identities stay in process. No buffer or lease is retained.
/// Equality reports captured-claim similarity, not current physical ownership,
/// runtime immutability, graphExec identity, or permission to reuse execution.
pub struct SubmissionWaveStructureObservation {
    lane: ExecutionLaneId,
    lane_epoch: u64,
    plan: PlanHash,
    program: Option<DeviceReusableExecutionProgramId>,
    resident_selected: bool,
    dimensions: (u32, u64),
    participants: Vec<(
        RequestAuthorityId,
        SequenceAuthorityId,
        SequenceSessionEpoch,
        SequenceSessionFingerprint,
    )>,
    step_slot: Option<LaneStableArenaSlotIdentity>,
    invocation_slot: Option<LaneStableArenaSlotIdentity>,
    shared: (Vec<CapturedExtent>, Vec<CapturedExtent>),
    participant_extents: Vec<(Vec<CapturedExtent>, Vec<CapturedExtent>)>,
}

impl SubmissionWaveStructureObservation {
    pub fn resident_program_selected(&self) -> bool {
        self.resident_selected && self.program.is_some()
    }
    pub const PROGRAM_OR_LANE: u8 = 1;
    pub const DIMENSIONS: u8 = 2;
    pub const PARTICIPANTS: u8 = 4;
    pub const PHYSICAL_SLOTS: u8 = 8;
    pub const CAPTURED_EXTENTS: u8 = 16;
    pub const PLAN: u8 = 32;

    /// Joint change mask. Zero means only these observed facts compare equal.
    pub fn change_mask(&self, prior: &Self) -> u8 {
        let mut mask = 0;
        if self.lane != prior.lane
            || self.lane_epoch != prior.lane_epoch
            || self.program != prior.program
            || self.resident_selected != prior.resident_selected
        {
            mask |= Self::PROGRAM_OR_LANE;
        }
        if self.dimensions != prior.dimensions {
            mask |= Self::DIMENSIONS;
        }
        if self.participants != prior.participants {
            mask |= Self::PARTICIPANTS;
        }
        if self.step_slot != prior.step_slot || self.invocation_slot != prior.invocation_slot {
            mask |= Self::PHYSICAL_SLOTS;
        }
        if self.shared != prior.shared || self.participant_extents != prior.participant_extents {
            mask |= Self::CAPTURED_EXTENTS;
        }
        if self.plan != prior.plan {
            mask |= Self::PLAN;
        }
        mask
    }
}

impl<R: DeviceRuntime> PreparedStepSubmissionWave<R> {
    pub(crate) fn structure_observation(
        &self,
        program: Option<DeviceReusableExecutionProgramId>,
        lane_epoch: u64,
        resident_selected: bool,
    ) -> SubmissionWaveStructureObservation {
        // Dispatch validated nonempty canonical participants before this call.
        let first = &self.nodes[0].participant_authority;
        let work = self.claimed_backing.work_shape();
        SubmissionWaveStructureObservation {
            lane: self.execution_lane_id,
            lane_epoch,
            plan: first.plan_evidence.plan_hash().clone(),
            program,
            resident_selected,
            dimensions: (work.immediate_sequences(), work.immediate_tokens()),
            participants: first
                .participants
                .iter()
                .zip(&first.participant_session_identities)
                .map(|(participant, (epoch, fingerprint))| {
                    (
                        participant.request_authority(),
                        participant.sequence_authority(),
                        *epoch,
                        fingerprint.clone(),
                    )
                })
                .collect(),
            step_slot: self.step.claimed_backing().lane_stable_slot_identity(),
            invocation_slot: self
                .claimed_backing
                .program_binding_lane_slot_identity()
                .cloned(),
            shared: (
                extents(self.step.backing_slices()),
                extents(self.claimed_backing.backing_slices()),
            ),
            participant_extents: self
                .step
                .participants
                .iter()
                .map(|participant| {
                    (
                        extents(
                            participant
                                .session
                                .resources()
                                .request_resources()
                                .backing_slices(),
                        ),
                        extents(participant.backing_snapshot.backing_slices()),
                    )
                })
                .collect(),
        }
    }
}
