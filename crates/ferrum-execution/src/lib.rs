//! Product execution contracts built on the shared runtime's resource authority.
//!
//! Foundation ports remain in `ferrum_interfaces`; device submission, completion,
//! and resource authorization belong to `ferrum_runtime`.

mod event_sink;
pub mod model_executor;
pub mod token_execution;

pub use event_sink::ProductExecutionEventSink;
pub use ferrum_runtime::{ExecutorRequestOrigin, PlanRuntimeResourceSnapshot};
pub use model_executor::{
    DecodeInput, DecodeOutput, ExecutorAdmissionEpochs, ExecutorPrefillAdmission,
    ExecutorPrefillAdmissionDecision, ExecutorPrefillAdmissionReceipt,
    ExecutorPrefillMaintenanceBlocker, ExecutorPrefillMaintenanceDeferral,
    ExecutorPrefillMaintenanceOutcome, ExecutorPrefillMaintenanceStage, ExecutorSamplingOutput,
    ModelExecutor, PlanRuntimeBatchDecodeOutcome, PlanRuntimeBatchPrefillOutcome,
    PlanRuntimeDecodeInput, PlanRuntimeDecodeOutput, PlanRuntimePrefillAuthority,
    PlanRuntimePrefillCompletion, PlanRuntimePrefillInput, PlanRuntimePrefillOutcome,
    PlanRuntimePrefillOutput, PlanRuntimePrefillProduct, PrefillInput, PrefillOutput,
};
