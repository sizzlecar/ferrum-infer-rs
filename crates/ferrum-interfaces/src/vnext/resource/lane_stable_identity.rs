use serde::Serialize;

use super::{AllocationLifetime, ExecutionLaneId};
use crate::vnext::ReusableExecutionBucketId;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LaneStableArenaSlotIdentity {
    lane_id: ExecutionLaneId,
    lifetime: AllocationLifetime,
    reusable_execution_bucket_id: ReusableExecutionBucketId,
    layout_fingerprint: String,
    slot_id: u64,
}

impl LaneStableArenaSlotIdentity {
    pub(super) fn new(
        lane_id: ExecutionLaneId,
        lifetime: AllocationLifetime,
        reusable_execution_bucket_id: ReusableExecutionBucketId,
        layout_fingerprint: String,
        slot_id: u64,
    ) -> Self {
        Self {
            lane_id,
            lifetime,
            reusable_execution_bucket_id,
            layout_fingerprint,
            slot_id,
        }
    }

    pub const fn lane_id(&self) -> ExecutionLaneId {
        self.lane_id
    }

    pub const fn lifetime(&self) -> AllocationLifetime {
        self.lifetime
    }

    pub fn reusable_execution_bucket_id(&self) -> &ReusableExecutionBucketId {
        &self.reusable_execution_bucket_id
    }

    pub fn layout_fingerprint(&self) -> &str {
        &self.layout_fingerprint
    }

    pub const fn slot_id(&self) -> u64 {
        self.slot_id
    }
}
