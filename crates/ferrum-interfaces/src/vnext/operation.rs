mod attribute;
mod buffer_view;
mod catalog;
mod compiled_submission_wave;
mod descriptor;
mod determinism;
mod determinism_artifact;
mod dispatch;
mod dispatch_contract;
mod foundation;
mod identity;
mod invocation;
mod provider;
mod registry;
mod resolved_value;
mod storage_profile;
mod tensor_contract;
mod workspace_encoding;

pub use attribute::{
    AttributeConstraint, AttributeId, AttributeSchema, AttributeSpec, AttributeValueKind,
};
pub use buffer_view::{
    OperationBufferRegionIter, OperationBufferRegions, OperationBufferStorageKind,
    OperationBufferView, OperationPhysicalRegion,
};
pub use catalog::CapabilityCatalog;
pub use compiled_submission_wave::CompiledSubmissionWaveIdentity;
pub use descriptor::{
    OperationContract, OperationDescriptor, OracleSpec, ProfilePhase, ProviderRequirement,
    ResourcePresenceRequirement, ResourceRequirements,
};
pub use determinism::{
    SubmissionWaveDeterminismEvidence, SubmissionWaveDeterminismHandle,
    SubmissionWaveDeterminismInitializationIdentity, SubmissionWaveDeterminismLogicalRange,
    SubmissionWaveDeterminismPhysicalReadback, SubmissionWaveDeterminismReadbackPlan,
    SubmissionWaveDeterminismReadbackTarget, SubmissionWaveDeterminismRestore,
    SubmissionWaveDeterminismRestoreLayout, SubmissionWaveDeterminismWitnessReadback,
};
pub use determinism_artifact::{
    SubmissionWaveDeterminismArtifactAttribution, SubmissionWaveDeterminismArtifactExecution,
    SubmissionWaveDeterminismArtifactInitializationIdentity,
    SubmissionWaveDeterminismArtifactLogicalCommand,
    SubmissionWaveDeterminismArtifactPhysicalCommand,
    SubmissionWaveDeterminismArtifactReplayedSegment, SubmissionWaveDeterminismArtifactWitness,
};
pub use dispatch::OperationDispatch;
pub use dispatch_contract::{
    BoundDeviceSubmissionAttribution, DispatchRetryAuthority, OperationDispatchError,
    ProfiledSubmissionHandle, SubmissionExecutionPolicy, SubmissionScratchInitialization,
    SubmissionWaveDispatchError, SubmissionWaveDispatchStage, SubmissionWaveDispatchTimingSink,
    SubmissionWaveInputUpload,
};
pub use identity::{
    BatchOperationIdentity, BatchOperationIdentityMaterializationSnapshot,
    BatchOperationNodeIdentity, BatchOperationParticipantIdentity,
};
pub use invocation::{BatchedOperationInvocation, OperationInvocation};
pub use provider::{
    EngineProviderDescriptor, ExecutionDeterminismRequirement, OperationFailure,
    OperationProviderDescriptor, ProviderCompatibilityRejectReason, ProviderCompatibilityRejection,
    ProviderCompatibilityReport, ProviderCompatibilityRequest,
    ProviderExecutionContractFingerprint, ProviderExecutionRepeatability,
    ProviderExecutionSemantics, ProviderReplayEquivalence, UnvalidatedOperationFailure,
    PROVIDER_EXECUTION_SEMANTICS_VERSION,
};
pub(crate) use registry::OperationRegistryAuthority;
pub use registry::{
    BoundOperationProvider, BoundOperationProviderSet, OperationPlanningHandle,
    OperationPlanningRegistry, OperationProvider, OperationResourceEstimate,
    OperationResourceEstimateRequest, OperationResourceEstimator, OperationRuntimeRegistry,
    ReusableExecutionTopology, ReusableExecutionTopologyRequest,
};
pub use resolved_value::{
    ProviderStorageBindingRequirement, ResolvedStorageComponent, ResolvedValueBinding,
    ResolvedValueRole, ResolvedValueStorage,
};
pub use storage_profile::{
    DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageRequirement, DynamicStorageView,
    ElementType,
};
pub use tensor_contract::{
    AliasPolicy, BlockedTensorPadding, DimensionConstraint, LayoutConstraint, ResolvedTensorLayout,
    ResolvedTensorSpec, StrideConstraint, TensorAccess, TensorContract,
};

pub const MAX_OPERATION_CATALOG_ROWS: usize = 4096;
pub const MAX_OPERATION_PROVIDER_ROWS: usize = 16384;
pub const MAX_ENGINE_PROVIDER_ROWS: usize = 4096;
pub const MAX_OPERATION_FAILURE_WIRE_BYTES: usize = 16 * 1024;
pub const MAX_REFERENCE_ORACLE_DEPTH: usize = 64;
