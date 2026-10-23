//! Product execution protocol and model-program driver using shared resource authority.
//!
//! Foundation ports remain in `ferrum_interfaces`; device submission, completion,
//! and resource authorization belong to `ferrum_runtime`.

mod build_error;
pub use build_error::{ExecutorBuildError, ExecutorBuildResult};

pub mod component;
pub mod driver;
mod event_sink;
pub mod execution_attempt;
pub mod executor_lifecycle;
pub mod model_executor;
pub mod prepared_model;
pub mod tensor;
pub mod token_execution;
pub use tensor::{
    PreparedTensorExecutor, TensorAdmissionFailure, TensorAdmissionStage, TensorExecutionError,
    TensorExecutionOutput, TensorExecutionRetirement, TensorExecutorConfig,
    TensorRequestCloseFailure, TensorRequestSession, TensorRequestStateMapping,
};

#[cfg(any(test, feature = "test-support"))]
pub use driver::test_support;
pub use driver::{
    CompiledProgram, DeterminismExecutionMode, DeterminismExecutionSpec, DeterminismInitialState,
    DeterminismParticipantSpec, DeterminismPhase, DeterminismWorkspacePoison,
    ProgramCompilationError, ProgramExecutor, ProgramExecutorConfig, RuntimeComposition,
    MAX_DETERMINISM_PARTICIPANTS,
};
pub use event_sink::ProductExecutionEventSink;
pub use executor_lifecycle::{
    ExecutorOwner, ExecutorScope, ExecutorShutdownObserver, ExecutorShutdownPermit,
    ExecutorShutdownPhase, ExecutorShutdownReport, ExecutorShutdownReporter,
};
pub use ferrum_runtime::{ExecutorRequestOrigin, PlanRuntimeResourceSnapshot};
pub use model_executor::{
    ExecutorAdmissionEpochs, ExecutorPrefillAdmission, ExecutorPrefillAdmissionDecision,
    ExecutorPrefillAdmissionReceipt, ExecutorPrefillMaintenanceBlocker,
    ExecutorPrefillMaintenanceDeferral, ExecutorPrefillMaintenanceOutcome,
    ExecutorPrefillMaintenanceStage, ExecutorSamplingOutput, ModelExecutor,
};
pub use prepared_model::{CausalExecutionFacts, PreparedExecutionInput};
