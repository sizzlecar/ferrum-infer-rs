//! Opt-in observations of the existing exact-shape execution path.
use ferrum_runtime::{
    BoundDeviceSubmissionAttribution, DeviceSubmissionStage, DeviceSubmissionTimingSink,
    OperationCompletionReceipt, SubmissionWaveDispatchStage, SubmissionWaveDispatchTimingSink,
};
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
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

/// Bounded observations of the original dispatch, enabled by DEBUG tracing.
/// Callbacks only accumulate durations; formatting happens after dispatch.
#[derive(Default)]
pub(super) struct HostDispatchTiming {
    contract: AtomicU64,
    backing: AtomicU64,
    provider: AtomicU64,
    identity: AtomicU64,
    invocation: AtomicU64,
    binding: AtomicU64,
    lane_reserve: AtomicU64,
    device_submit: AtomicU64,
    backend_prepare: AtomicU64,
    backend_input_sync: AtomicU64,
    backend_enqueue: AtomicU64,
    backend_fence: AtomicU64,
    completion_arm: AtomicU64,
}

fn accumulate(counter: &AtomicU64, elapsed: Duration) {
    let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
        Some(old.saturating_add(ns))
    });
}

impl HostDispatchTiming {
    pub(super) fn trace(&self, dispatch_succeeded: bool) {
        let us = |counter: &AtomicU64| counter.load(Ordering::Relaxed) / 1000;
        tracing::debug!(
            dispatch_succeeded,
            contract_us = us(&self.contract),
            backing_input_us = us(&self.backing),
            provider_nodes_us = us(&self.provider),
            node_identity_us = us(&self.identity),
            node_invocation_us = us(&self.invocation),
            binding_validation_us = us(&self.binding),
            lane_reserve_us = us(&self.lane_reserve),
            device_submit_us = us(&self.device_submit),
            backend_prepare_us = us(&self.backend_prepare),
            backend_input_sync_us = us(&self.backend_input_sync),
            backend_enqueue_us = us(&self.backend_enqueue),
            backend_fence_us = us(&self.backend_fence),
            completion_arm_us = us(&self.completion_arm),
            "tensor dispatch host timing; nested intervals overlap and are not GPU durations"
        );
    }
}

impl DeviceSubmissionTimingSink for HostDispatchTiming {
    const ENABLED: bool = true;

    fn record_device_submission(&self, stage: DeviceSubmissionStage, elapsed: Duration) {
        let counter = match stage {
            DeviceSubmissionStage::ValidateAndPrepare => &self.backend_prepare,
            DeviceSubmissionStage::InputPacketAllocationSync => &self.backend_input_sync,
            DeviceSubmissionStage::EnqueueCommands => &self.backend_enqueue,
            DeviceSubmissionStage::RecordFenceAndAccount => &self.backend_fence,
            _ => return,
        };
        accumulate(counter, elapsed);
    }
}

impl SubmissionWaveDispatchTimingSink for HostDispatchTiming {
    fn record(&self, stage: SubmissionWaveDispatchStage, elapsed: Duration) {
        let counter = match stage {
            SubmissionWaveDispatchStage::ContractValidateAndReserve => &self.contract,
            SubmissionWaveDispatchStage::BackingAndInputEncode => &self.backing,
            SubmissionWaveDispatchStage::ProviderNodeEncode => &self.provider,
            SubmissionWaveDispatchStage::NodeIdentityMaterialize => &self.identity,
            SubmissionWaveDispatchStage::NodeInvocationConstruct => &self.invocation,
            SubmissionWaveDispatchStage::BindingValidateAndCoalesce => &self.binding,
            SubmissionWaveDispatchStage::LaneReserve => &self.lane_reserve,
            SubmissionWaveDispatchStage::DeviceRuntimeSubmit => &self.device_submit,
            SubmissionWaveDispatchStage::CompletionArm => &self.completion_arm,
            _ => return,
        };
        accumulate(counter, elapsed);
    }
}
