use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use super::{
    AliasPolicy, AttributeId, BlockedTensorPadding, BufferRequest, BufferUsage, CapabilityCatalog,
    CapabilityId, ContractVersion, DeviceId, DynamicStorageAllocator, DynamicStorageProfile,
    DynamicStorageRequirement, DynamicStorageView, ElementType, ModelFamilyId, NodeId,
    OperationDescriptor, OperationId, OperationPlanningHandle, OperationPlanningRegistry,
    OperationProviderDescriptor, OperationRegistryAuthority, OperationResourceEstimate,
    OperationResourceEstimateRequest, PlanId, PreparedModelFamily, ProgramNode, ProgramTensorSpec,
    ProgramValueId, ProviderCompatibilityRejectReason, ProviderCompatibilityRequest, ProviderId,
    QuantizationFormatId, ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueBinding,
    ResolvedValueRole, ResolvedValueStorage, ResourceId, SemanticValue, StateCapacityDemand,
    StateId, StateLifetime, TensorAccess, VNextError, WeightEncoding, WeightFormatId, WeightId,
};

mod contracts;
pub use contracts::*;

mod work;
pub use work::*;

mod storage;
pub use storage::*;

mod memory;
pub use memory::*;

mod provider;
pub use provider::*;

mod plan;
pub use plan::*;

mod solver;
use solver::*;

mod policy;
pub use policy::*;

#[cfg(test)]
#[path = "execution/tests.rs"]
mod tests;
