//! Terminal device evidence, without a second clock or inferred kernel times.
use super::*;
use serde::Serialize;
use std::time::Duration;

/// Dictionary for the exact submission's native command `node_index` values.
#[derive(Debug, Clone, Serialize)]
pub struct TensorTimingNode {
    pub node_index: usize,
    pub node_id: NodeId,
    pub operation_id: OperationId,
}

/// One successful tensor execution's existing backend observations. Device
/// intervals and blocking Host wait overlap and must not be added. Span ranges
/// and labeled subwork are retained intact; one span is not duplicated across
/// the logical nodes it covers. Unsupported measurements remain unavailable.
#[derive(Debug, Clone, Serialize)]
pub struct TensorExecutionTiming {
    pub run_id: RunId,
    pub request_id: RequestIdentity,
    pub batch_invocation_id: BatchInvocationId,
    pub plan_hash: PlanHash,
    pub runtime_implementation_fingerprint: String,
    pub fence: CompletionFenceTiming,
    pub nodes: Vec<TensorTimingNode>,
    pub attribution: Option<DeviceSubmissionAttribution>,
    pub submission_timing: DeviceTimingMeasurement<DeviceSubmissionExecutionTiming>,
}

impl TensorExecutionTiming {
    pub(super) fn from_completion(
        run_id: RunId,
        request_id: RequestIdentity,
        receipt: &OperationCompletionReceipt,
        observation: SubmissionObservation,
    ) -> Self {
        let identity = receipt.submission().batch_identity();
        let nodes = (0..identity.node_count())
            .map(|index| TensorTimingNode {
                node_index: index,
                node_id: identity.node_id_at(index).expect("validated node").clone(),
                operation_id: identity
                    .operation_id_at(index)
                    .expect("validated operation")
                    .clone(),
            })
            .collect();
        Self {
            run_id,
            request_id,
            batch_invocation_id: identity.batch_invocation_id(),
            plan_hash: identity.plan_hash().clone(),
            runtime_implementation_fingerprint: identity
                .runtime_implementation_fingerprint()
                .to_owned(),
            fence: receipt.fence_timing(),
            nodes,
            attribution: observation
                .into_parts()
                .1
                .map(|attribution| attribution.device().clone()),
            submission_timing: receipt.submission_timing().clone(),
        }
    }

    pub(super) fn log(&self) {
        // Also available to both run/serve through the ordinary tracing writer.
        // The typed receipt remains available even when this target is filtered.
        if tracing::enabled!(target: "ferrum_execution::tensor::device_timing", tracing::Level::INFO)
        {
            match serde_json::to_string(self) {
                Ok(report) => tracing::info!(
                    target: "ferrum_execution::tensor::device_timing",
                    tensor_device_timing = %report,
                    "tensor execution terminal device evidence"
                ),
                Err(error) => tracing::warn!(
                    target: "ferrum_execution::tensor::device_timing",
                    %error, "could not serialize tensor device evidence"
                ),
            }
        }
    }
}

// Use the existing profiled dispatch API without enabling its separate Host
// stage timers. The Off path remains compile-time free of these clock reads.
pub(super) struct NoHostDispatchTiming;
impl DeviceSubmissionTimingSink for NoHostDispatchTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        unreachable!("disabled Host timing sink")
    }
}
impl SubmissionWaveDispatchTimingSink for NoHostDispatchTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        unreachable!("disabled Host timing sink")
    }
}
