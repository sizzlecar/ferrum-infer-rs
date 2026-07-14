use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use super::{
    AliasPolicy, AttributeId, BlockedTensorPadding, BufferRequest, BufferUsage, CapabilityCatalog,
    CapabilityId, ContractVersion, DeviceId, DimensionConstraint, DynamicStorageAllocator,
    DynamicStorageProfile, DynamicStorageRequirement, DynamicStorageView, ElementType,
    ModelFamilyId, NodeId, OperationDescriptor, OperationId, OperationPlanningHandle,
    OperationPlanningRegistry, OperationProviderDescriptor, OperationRegistryAuthority,
    OperationResourceEstimate, OperationResourceEstimateRequest, PlanId, PreparedModelFamily,
    ProgramNode, ProgramNodeWorkSpec, ProgramTensorSpec, ProgramValueId,
    ProviderCompatibilityRejectReason, ProviderCompatibilityRequest, ProviderId,
    QuantizationFormatId, ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueBinding,
    ResolvedValueRole, ResolvedValueStorage, ResourceId, SemanticValue, StateCapacityDemand,
    StateId, StateLifetime, TensorAccess, VNextError, WeightEncoding, WeightFormatId, WeightId,
};

mod foundation;
use foundation::*;

mod binding;
pub(crate) use binding::*;

mod work;
pub use work::*;

mod workspace;
pub use workspace::*;

mod provider_resource;
pub use provider_resource::*;

mod contracts;
pub use contracts::*;

mod storage;
pub use storage::*;

mod allocation;
pub use allocation::*;

mod solver;
use solver::*;

mod memory;
pub use memory::*;

mod provider;
pub use provider::*;

mod policy;
pub use policy::*;

mod plan;
pub use plan::*;

mod resolution;

mod validation;

mod planner;
pub use planner::*;

#[cfg(test)]
#[path = "execution/tests.rs"]
mod tests;
