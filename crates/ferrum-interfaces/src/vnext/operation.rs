mod attribute;
mod buffer_view;
mod catalog;
mod compiled_identity;
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
mod semantic;
mod storage_profile;
mod tensor_contract;
mod weight_contract;
mod workspace_encoding;

pub use attribute::{AttributeConstraint, AttributeSchema, AttributeSpec};
pub use buffer_view::{
    OperationBufferRegionIter, OperationBufferRegions, OperationBufferStorageKind,
    OperationBufferView, OperationPhysicalRegion,
};
pub use catalog::{
    CapabilityCatalog, MAX_ENGINE_PROVIDER_ROWS, MAX_OPERATION_CATALOG_ROWS,
    MAX_OPERATION_PROVIDER_ROWS, MAX_REFERENCE_ORACLE_DEPTH,
};
pub use compiled_identity::CompiledSubmissionWaveIdentity;
pub use descriptor::{
    OperationContract, OperationDescriptor, OracleSpec, ProfilePhase, ProviderRequirement,
    ResourcePresenceRequirement, ResourceRequirements,
};
pub use determinism::{
    SubmissionWaveDeterminismEvidence, SubmissionWaveDeterminismHandle,
    SubmissionWaveDeterminismInitializationIdentity, SubmissionWaveDeterminismLogicalRange,
    SubmissionWaveDeterminismParticipantOrder, SubmissionWaveDeterminismPhysicalReadback,
    SubmissionWaveDeterminismReadbackPlan, SubmissionWaveDeterminismReadbackTarget,
    SubmissionWaveDeterminismRestore, SubmissionWaveDeterminismRestoreLayout,
    SubmissionWaveDeterminismWitnessReadback,
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
    MAX_OPERATION_FAILURE_WIRE_BYTES, PROVIDER_EXECUTION_SEMANTICS_VERSION,
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
pub use semantic::{AttributeId, AttributeValueKind, CanonicalRational, SemanticValue};
pub use storage_profile::{
    DynamicStorageAllocator, DynamicStorageProfile, DynamicStorageRequirement, DynamicStorageView,
    ElementType,
};
pub use tensor_contract::{
    AliasPolicy, BlockedTensorPadding, DimensionConstraint, LayoutConstraint, ResolvedTensorLayout,
    ResolvedTensorSpec, StrideConstraint, TensorAccess, TensorContract,
};
pub(crate) use weight_contract::{
    checked_elements, physical_component_ids, validate_physical_layout_budget,
    ResolvedWeightLogicalValidation,
};
pub use weight_contract::{
    AxisWeightComponent, BlockQuantizationSpec, CompositeWeightPart, PhysicalStorageLayout,
    PhysicalWeightComponentBinding, PhysicalWeightLayout, PhysicalWeightPadding,
    QuantizationGrouping, QuantizationPacking, QuantizationSpec, ResolvedWeightBinding,
    ResolvedWeightComponentLayout, WeightComponentRole, WeightEncoding,
    MAX_PHYSICAL_WEIGHT_LAYOUT_DEPTH, MAX_PHYSICAL_WEIGHT_LAYOUT_NODES,
};
