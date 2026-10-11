//! Authority for a wave prefix that reads immutable Plan weights and updates
//! an owned, preserved Plan validation bank. This is deliberately independent
//! of request-specific program binding slots.
use std::sync::Arc;

use serde::Serialize;

use super::foundation::invalid_operation;
use super::{BatchedOperationInvocation, ResolvedValueBinding, ResolvedValueRole, TensorAccess};
use crate::vnext::resource::SegmentPlanResourceView;
use crate::vnext::{
    AllocationLifetime, BufferDescriptor, BufferUsage, DeviceBufferRetention,
    DeviceReusableAddressScope, ElementType, NodeId, ProviderId, ResourceTransactionIdentity,
    VNextError, WeightId,
};

/// Exact subranges authorized by a live invocation. Source offsets are relative
/// to the selected physical weight component; destination offsets are relative
/// to this node's persistent allocation, including its admitted padding.
pub struct RetainedPlanDependencySpec<'a> {
    pub input_ordinal: u32,
    pub component_id: &'a WeightId,
    pub source_offset_bytes: u64,
    pub source_length_bytes: u64,
    pub persistent_offset_bytes: u64,
    pub persistent_length_bytes: u64,
    pub alignment_bytes: u64,
    /// Provider implementation plus the exact validation arithmetic/geometry.
    pub validation_identity: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RetainedPlanDependencyIdentity {
    node: NodeId,
    provider: ProviderId,
    source: RetainedPlanDependencyRange,
    destination: RetainedPlanDependencyRange,
    component: WeightId,
    input_ordinal: u32,
    validation_identity: String,
}

impl RetainedPlanDependencyIdentity {
    pub fn node_id(&self) -> &NodeId {
        &self.node
    }
    pub fn component_id(&self) -> &WeightId {
        &self.component
    }
    pub fn persistent_offset_bytes(&self) -> u64 {
        self.destination.offset
    }
    pub fn validation_identity(&self) -> &str {
        &self.validation_identity
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct RetainedPlanDependencyRange {
    pub descriptor: BufferDescriptor,
    pub transaction: ResourceTransactionIdentity,
    pub generation: u64,
    pub offset: u64,
    pub length: u64,
}

/// No public constructor: only a validated, live invocation can issue this
/// authority. The ownership remains alive even if a backend command forgets
/// to retain its borrowed source view.
pub struct RetainedPlanDependencyAuthority {
    pub(super) scope: Arc<()>,
    pub(crate) identity: RetainedPlanDependencyIdentity,
    _source_retention: DeviceBufferRetention,
    _destination_retention: DeviceBufferRetention,
}

pub struct EncodedRetainedPlanDependency<C> {
    pub(super) authority: RetainedPlanDependencyAuthority,
    pub(crate) command: C,
}

impl<C> EncodedRetainedPlanDependency<C> {
    pub(crate) fn identity(&self) -> &RetainedPlanDependencyIdentity {
        &self.authority.identity
    }
}

impl RetainedPlanDependencyAuthority {
    /// The command must implement only the declared immutable validation and
    /// publication/wait, never read activation/state or earlier wave outputs.
    pub fn encode<C>(self, command: C) -> EncodedRetainedPlanDependency<C> {
        EncodedRetainedPlanDependency {
            authority: self,
            command,
        }
    }
}

impl<B> BatchedOperationInvocation<'_, B> {
    pub fn retained_plan_dependency(
        &self,
        spec: RetainedPlanDependencySpec<'_>,
    ) -> Result<RetainedPlanDependencyAuthority, VNextError> {
        if !self.retained_persistent_preserve
            || spec.validation_identity.is_empty()
            || !spec.alignment_bytes.is_power_of_two()
        {
            return Err(invalid_operation("retained Plan dependency requires owned Plan Preserve storage and a validation identity"));
        }
        let mut issued = None;
        for participant in self.participants() {
            let binding = participant
                .bindings()
                .iter()
                .find(|b| b.role() == ResolvedValueRole::Input && b.ordinal() == spec.input_ordinal)
                .ok_or_else(|| invalid_operation("retained dependency weight input is absent"))?;
            if binding.usage() != BufferUsage::Weights || binding.access() != TensorAccess::Read {
                return Err(invalid_operation(
                    "retained dependency source must be a read-only weight input",
                ));
            }
            let component = binding
                .storage()
                .components()
                .iter()
                .find(|c| c.component_id() == Some(spec.component_id))
                .ok_or_else(|| {
                    invalid_operation("retained dependency physical weight component is absent")
                })?;
            checked_range(
                spec.source_offset_bytes,
                spec.source_length_bytes,
                component.length_bytes(),
            )?;
            let source_offset = component
                .offset_bytes()
                .checked_add(spec.source_offset_bytes)
                .ok_or_else(|| invalid_operation("retained dependency weight offset overflows"))?;
            let source = participant
                .views()
                .iter()
                .find(|v| v.resource_id() == component.resource_id())
                .ok_or_else(|| invalid_operation("retained dependency weight view is absent"))?;
            let destination = participant.persistent_view().ok_or_else(|| {
                invalid_operation("retained dependency persistent view is absent")
            })?;
            if source.allocation_lifetime() != AllocationLifetime::Plan
                || source.descriptor().usage != BufferUsage::Weights
                || destination.allocation_lifetime() != AllocationLifetime::Plan
                || destination.descriptor().usage != BufferUsage::Persistent
                || destination.descriptor().element_type != ElementType::U8
                || destination.descriptor().alignment_bytes < spec.alignment_bytes
                || spec.persistent_offset_bytes % spec.alignment_bytes != 0
                || spec.persistent_length_bytes % spec.alignment_bytes != 0
                || source.resource_id() == destination.resource_id()
            {
                return Err(invalid_operation(
                    "retained dependency source/destination ownership, type, or alignment differs",
                ));
            }
            let (source, source_retention) =
                source.retained_dependency_range(source_offset, spec.source_length_bytes)?;
            let (destination, destination_retention) = destination.retained_dependency_range(
                spec.persistent_offset_bytes,
                spec.persistent_length_bytes,
            )?;
            let identity = RetainedPlanDependencyIdentity {
                node: self.node_id().clone(),
                provider: self.provider_id().clone(),
                source,
                destination,
                component: spec.component_id.clone(),
                input_ordinal: spec.input_ordinal,
                validation_identity: spec.validation_identity.to_owned(),
            };
            if let Some(previous) = &issued {
                let previous: &RetainedPlanDependencyAuthority = previous;
                if previous.identity != identity {
                    return Err(invalid_operation(
                        "retained dependency participants do not share exact Plan allocations",
                    ));
                }
            } else {
                issued = Some(RetainedPlanDependencyAuthority {
                    scope: self.retained_dependency_scope.clone(),
                    identity,
                    _source_retention: source_retention,
                    _destination_retention: destination_retention,
                });
            }
        }
        issued.ok_or_else(|| invalid_operation("retained dependency has no participants"))
    }
}

/// The operation layer binds these immutable declarations to the exact Plan
/// before issuing a segment. Both views must come from this wave's bulk Plan
/// lease observation, which already proved every participant shares that lease.
/// The destination is this node's declared persistent allocation; the scope is
/// newly created for this node in this attempt, never retained by a cold recipe.
pub(super) struct SegmentPlanDependencyInput<'a, 'lease, B> {
    pub scope: &'a Arc<()>,
    pub node: &'a NodeId,
    pub provider: &'a ProviderId,
    pub binding: &'a ResolvedValueBinding,
    pub source: &'a SegmentPlanResourceView<'lease, B>,
    pub destination: &'a SegmentPlanResourceView<'lease, B>,
    pub persistent_preserve: bool,
}

/// Issue only from freshly checked Plan views. This preserves the invocation
/// path's source/component/range checks without rebuilding participant views
/// or rechecking the same shared immutable Plan allocation for each participant.
pub(super) fn issue_segment_plan_dependency<B>(
    input: SegmentPlanDependencyInput<'_, '_, B>,
    spec: RetainedPlanDependencySpec<'_>,
) -> Result<RetainedPlanDependencyAuthority, VNextError> {
    if !input.persistent_preserve
        || spec.validation_identity.is_empty()
        || !spec.alignment_bytes.is_power_of_two()
    {
        return Err(invalid_operation("retained Plan dependency requires owned Plan Preserve storage and a validation identity"));
    }
    let binding = input.binding;
    if binding.role() != ResolvedValueRole::Input || binding.ordinal() != spec.input_ordinal {
        return Err(invalid_operation(
            "retained dependency weight input is absent",
        ));
    }
    if binding.usage() != BufferUsage::Weights || binding.access() != TensorAccess::Read {
        return Err(invalid_operation(
            "retained dependency source must be a read-only weight input",
        ));
    }
    let component = binding
        .storage()
        .components()
        .iter()
        .find(|component| component.component_id() == Some(spec.component_id))
        .ok_or_else(|| {
            invalid_operation("retained dependency physical weight component is absent")
        })?;
    checked_range(
        spec.source_offset_bytes,
        spec.source_length_bytes,
        component.length_bytes(),
    )?;
    let source_offset = component
        .offset_bytes()
        .checked_add(spec.source_offset_bytes)
        .ok_or_else(|| invalid_operation("retained dependency weight offset overflows"))?;
    let source = &input.source.leased;
    let destination = &input.destination.leased;
    if source.resource_id() != component.resource_id() {
        return Err(invalid_operation(
            "retained dependency weight view is absent",
        ));
    }
    // Plan lifetime is guaranteed by prepare_segment_plan_views. Recheck the
    // selected committed metadata and exact transaction/admission here so two
    // individually valid observations from different Plans cannot be combined.
    if source.committed_descriptor().usage != BufferUsage::Weights
        || destination.committed_descriptor().usage != BufferUsage::Persistent
        || destination.committed_descriptor().element_type != ElementType::U8
        || destination.committed_descriptor().alignment_bytes < spec.alignment_bytes
        || spec.persistent_offset_bytes % spec.alignment_bytes != 0
        || spec.persistent_length_bytes % spec.alignment_bytes != 0
        || source.resource_id() == destination.resource_id()
        || !std::ptr::eq(source.identity(), destination.identity())
        || !std::ptr::eq(source.admission(), destination.admission())
        || source.identity() != destination.identity()
        || source.admission() != destination.admission()
    {
        return Err(invalid_operation(
            "retained dependency source/destination ownership, type, or alignment differs",
        ));
    }
    let (source, source_retention) =
        segment_dependency_range(input.source, source_offset, spec.source_length_bytes)?;
    let (destination, destination_retention) = segment_dependency_range(
        input.destination,
        spec.persistent_offset_bytes,
        spec.persistent_length_bytes,
    )?;
    Ok(RetainedPlanDependencyAuthority {
        scope: Arc::clone(input.scope),
        identity: RetainedPlanDependencyIdentity {
            node: input.node.clone(),
            provider: input.provider.clone(),
            source,
            destination,
            component: spec.component_id.clone(),
            input_ordinal: spec.input_ordinal,
            validation_identity: spec.validation_identity.to_owned(),
        },
        _source_retention: source_retention,
        _destination_retention: destination_retention,
    })
}

fn segment_dependency_range<B>(
    view: &SegmentPlanResourceView<'_, B>,
    offset: u64,
    length: u64,
) -> Result<(RetainedPlanDependencyRange, DeviceBufferRetention), VNextError> {
    let leased = &view.leased;
    checked_range(offset, length, leased.committed_descriptor().size_bytes)?;
    if leased.generation() == 0
        || view.retention.reusable_address_scope() != Some(DeviceReusableAddressScope::Plan)
    {
        return Err(invalid_operation(
            "retained dependency has no live Plan generation",
        ));
    }
    Ok((
        RetainedPlanDependencyRange {
            descriptor: leased.committed_descriptor().clone(),
            transaction: leased.identity().clone(),
            generation: leased.generation(),
            offset,
            length,
        },
        view.retention.clone(),
    ))
}

pub(super) fn checked_range(offset: u64, length: u64, available: u64) -> Result<(), VNextError> {
    if length == 0 || offset.checked_add(length).is_none_or(|end| end > available) {
        return Err(invalid_operation(
            "retained dependency range is empty, overflowing, or outside admission",
        ));
    }
    Ok(())
}

pub(super) fn append_dependencies<C>(
    scope: &Arc<()>,
    dependencies: Vec<EncodedRetainedPlanDependency<C>>,
    identities: &mut Vec<RetainedPlanDependencyIdentity>,
    commands: &mut Vec<C>,
    leases: &mut Vec<RetainedPlanDependencyAuthority>,
) -> Result<(), VNextError> {
    for dependency in dependencies {
        if !Arc::ptr_eq(scope, &dependency.authority.scope) {
            return Err(invalid_operation(
                "retained dependency was not issued by this live invocation",
            ));
        }
        let identity = dependency.authority.identity.clone();
        for previous in identities.iter() {
            let left = &previous.destination;
            let right = &identity.destination;
            if left.transaction == right.transaction
                && left.descriptor.resource_id == right.descriptor.resource_id
                && left.offset < right.offset + right.length
                && right.offset < left.offset + left.length
                && previous != &identity
            {
                return Err(invalid_operation(
                    "retained dependency validation banks conflict",
                ));
            }
        }
        // Within ONE wave and ONE execution lane (hence one ordered stream),
        // repeated local launches may share one exact bank. Keep one ordered
        // callback; its backend owns once-publication and each-stream waiting.
        if !identities.contains(&identity) {
            identities.push(identity);
            commands.push(dependency.command);
            leases.push(dependency.authority);
        }
    }
    Ok(())
}
