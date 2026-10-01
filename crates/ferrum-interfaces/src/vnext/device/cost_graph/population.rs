//! Conservative retained payload charge for private route observations.
use crate::vnext::DeviceReusableExecutionProgramId;
impl DeviceReusableExecutionProgramId {
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.runtime_implementation_fingerprint.capacity())?
            .checked_add(self.program_binding_layout_fingerprint.capacity())?
            .checked_add(self.lane_stable_layout_fingerprint.capacity())?
            .checked_add(self.plan_hash.as_str().len())?
            .checked_add(self.bucket_id.as_str().len())?
            .checked_add(4 * std::mem::size_of::<usize>())
    }
}
