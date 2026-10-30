//! Opt-in observations of the existing exact-shape execution path.
use ferrum_runtime::{
    BoundDeviceSubmissionAttribution, DeviceSubmissionStage, DeviceSubmissionTimingSink,
    OperationCompletionReceipt, SubmissionWaveDispatchStage, SubmissionWaveDispatchTimingSink,
};
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
pub struct TensorExecutionProfile {
    pub attribution: Option<BoundDeviceSubmissionAttribution>,
    pub completion: OperationCompletionReceipt,
    /// Host wall intervals; they overlap device execution and must not be summed with it.
    pub host_wall_us: TensorHostWallTimings,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorHostWallTimings {
    pub readback_prepare: u64,
    pub sequence_admission: u64,
    pub step_admission: u64,
    pub wave_prepare: u64,
    pub identity_bind: u64,
    pub uploads_prepare: u64,
    pub lane_create: u64,
    pub encode_submit: u64,
    pub wait_readback: u64,
    pub drop_lane: u64,
    pub retire: u64,
    pub total: u64,
}

pub(super) struct NoHostDispatchTiming;
impl DeviceSubmissionTimingSink for NoHostDispatchTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        unreachable!("disabled host dispatch timing")
    }
}
impl SubmissionWaveDispatchTimingSink for NoHostDispatchTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        unreachable!("disabled host dispatch timing")
    }
}
