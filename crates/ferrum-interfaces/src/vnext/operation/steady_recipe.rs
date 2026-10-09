//! Declarative resident-node preparation. Declarations describe requirements;
//! only core can issue the fresh, retained physical table consumed by a provider.
use std::any::Any;
use std::sync::Arc;

use super::foundation::invalid_operation;
use super::{
    BatchOperationIdentity, BatchOperationNodeIdentity, OperationPhysicalRegion, ResolvedValueRole,
    RetainedPlanDependencyAuthority, RetainedPlanDependencySpec,
};
use crate::vnext::{
    BufferDescriptor, DeviceBufferRetention, ElementType, LogicalBackingSegmentBinding, NodeId,
    ProgramBindingNodeBinding, ProviderId, VNextError, WeightId,
};

/// A finite selector in the already compiled node schema. No executable
/// callbacks or arbitrary expressions are evaluated under a resource permit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SteadyRecipeRegionSelector {
    Value {
        role: ResolvedValueRole,
        ordinal: u32,
        component: Option<WeightId>,
    },
    Scratch,
    Persistent,
    ProgramBinding,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SteadyRecipeRegionRequest {
    pub selector: SteadyRecipeRegionSelector,
    /// Relative to the selected component or workspace, not a device address.
    pub offset_bytes: u64,
    pub length_bytes: u64,
    pub element_type: ElementType,
    pub alignment_bytes: u64,
}

/// Owned static dependency declaration. This is not an issued authority and
/// carries no invocation scope or request/step resource retention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SteadyRecipeDependency {
    pub input_ordinal: u32,
    pub component_id: WeightId,
    pub source_offset_bytes: u64,
    pub source_length_bytes: u64,
    pub persistent_offset_bytes: u64,
    pub persistent_length_bytes: u64,
    pub alignment_bytes: u64,
    pub validation_identity: String,
}

impl SteadyRecipeDependency {
    pub fn as_spec(&self) -> RetainedPlanDependencySpec<'_> {
        RetainedPlanDependencySpec {
            input_ordinal: self.input_ordinal,
            component_id: &self.component_id,
            source_offset_bytes: self.source_offset_bytes,
            source_length_bytes: self.source_length_bytes,
            persistent_offset_bytes: self.persistent_offset_bytes,
            persistent_length_bytes: self.persistent_length_bytes,
            alignment_bytes: self.alignment_bytes,
            validation_identity: &self.validation_identity,
        }
    }
}

/// Provider-owned cold state is restricted by contract to Plan/static owners
/// and scalar geometry. In particular it must not contain a prior invocation,
/// its scratch projections, request state, commands or dependency authorities.
pub struct SteadyRecipeDeclaration {
    pub(super) regions: Vec<SteadyRecipeRegionRequest>,
    pub(super) dependencies: Vec<SteadyRecipeDependency>,
    pub(super) state: Arc<dyn Any + Send + Sync>,
}

impl SteadyRecipeDeclaration {
    pub fn new(
        regions: Vec<SteadyRecipeRegionRequest>,
        dependencies: Vec<SteadyRecipeDependency>,
        state: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, VNextError> {
        if regions.is_empty()
            || regions.iter().any(|r| {
                r.length_bytes == 0
                    || !r.alignment_bytes.is_power_of_two()
                    || r.offset_bytes.checked_add(r.length_bytes).is_none()
            })
        {
            return Err(invalid_operation(
                "steady recipe has an invalid region declaration",
            ));
        }
        Ok(Self {
            regions,
            dependencies,
            state,
        })
    }
}

pub(super) enum CheckedSteadyRecipeStorage<'a, B> {
    Static {
        buffer: &'a B,
        retention: DeviceBufferRetention,
    },
    Dynamic(LogicalBackingSegmentBinding<B>),
}

/// An exact retained physical range issued by core for this preparation only.
/// No public constructor permits a provider to turn numeric IDs into authority.
pub struct CheckedSteadyRecipeRegion<'a, B> {
    pub(super) storage: CheckedSteadyRecipeStorage<'a, B>,
    pub(super) descriptor: BufferDescriptor,
    pub(super) logical_offset_bytes: u64,
    pub(super) physical_offset_bytes: u64,
    pub(super) length_bytes: u64,
}

impl<B> CheckedSteadyRecipeRegion<'_, B> {
    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
    pub fn physical_region(&self) -> OperationPhysicalRegion<'_, B> {
        let (buffer, retention) = match &self.storage {
            CheckedSteadyRecipeStorage::Static { buffer, retention } => {
                (*buffer, retention.clone())
            }
            CheckedSteadyRecipeStorage::Dynamic(binding) => (binding.buffer(), binding.retention()),
        };
        OperationPhysicalRegion::from_checked_parts(
            buffer,
            self.logical_offset_bytes,
            self.physical_offset_bytes,
            self.length_bytes,
            retention,
        )
    }
}

/// All regions are fresh for this node/wave. The provider may construct only
/// owned commands/retentions from them; the table cannot be moved into cold state.
pub struct PreparedSteadyRecipePatch<'a, B> {
    pub(super) batch_identity: &'a BatchOperationIdentity,
    pub(super) node_identity: &'a BatchOperationNodeIdentity,
    pub(super) program_binding: ProgramBindingNodeBinding,
    pub(super) participant_count: usize,
    pub(super) region_count: usize,
    pub(super) regions: Vec<CheckedSteadyRecipeRegion<'a, B>>,
    pub(super) state: Arc<dyn Any + Send + Sync>,
    pub(super) dependency_scope: Arc<()>,
    pub(super) dependencies: Vec<(
        SteadyRecipeDependency,
        super::retained_dependency::RetainedPlanDependencyTemplate,
    )>,
}

impl<B> PreparedSteadyRecipePatch<'_, B> {
    pub fn batch_identity(&self) -> &BatchOperationIdentity {
        self.batch_identity
    }
    pub fn node_identity(&self) -> &BatchOperationNodeIdentity {
        self.node_identity
    }
    pub fn node_id(&self) -> &NodeId {
        self.node_identity.node_id()
    }
    pub fn provider_id(&self) -> &ProviderId {
        self.node_identity.provider_id()
    }
    pub fn program_binding(&self) -> &ProgramBindingNodeBinding {
        &self.program_binding
    }
    pub fn participant_count(&self) -> usize {
        self.participant_count
    }
    pub fn cold_state<T: Any + Send + Sync>(&self) -> Result<&T, VNextError> {
        self.state
            .downcast_ref::<T>()
            .ok_or_else(|| invalid_operation("steady recipe cold state type differs"))
    }
    pub fn region(
        &self,
        participant: usize,
        declaration: usize,
    ) -> Result<&CheckedSteadyRecipeRegion<'_, B>, VNextError> {
        if participant >= self.participant_count || declaration >= self.region_count {
            return Err(invalid_operation(
                "steady recipe region index is out of range",
            ));
        }
        let index = participant
            .checked_mul(self.region_count)
            .and_then(|start| start.checked_add(declaration))
            .ok_or_else(|| invalid_operation("steady recipe region index overflows"))?;
        self.regions
            .get(index)
            .ok_or_else(|| invalid_operation("steady recipe checked table is incomplete"))
    }
    pub fn retained_plan_dependency(
        &self,
        spec: RetainedPlanDependencySpec<'_>,
    ) -> Result<RetainedPlanDependencyAuthority, VNextError> {
        let (_, template) = self
            .dependencies
            .iter()
            .find(|(d, _)| {
                d.input_ordinal == spec.input_ordinal
                    && &d.component_id == spec.component_id
                    && d.source_offset_bytes == spec.source_offset_bytes
                    && d.source_length_bytes == spec.source_length_bytes
                    && d.persistent_offset_bytes == spec.persistent_offset_bytes
                    && d.persistent_length_bytes == spec.persistent_length_bytes
                    && d.alignment_bytes == spec.alignment_bytes
                    && d.validation_identity == spec.validation_identity
            })
            .ok_or_else(|| {
                invalid_operation("steady recipe dependency was not declared and checked")
            })?;
        Ok(template.issue(&self.dependency_scope))
    }
}
