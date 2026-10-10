//! Closed binding declarations for one resident segment. Declarations contain
//! immutable rules, never a previous invocation or submission authority. Core
//! resolves their resource uses under one fresh segment permit before handing
//! the retained physical facts to the runtime's single segment encoder.
use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use super::foundation::invalid_operation;
use super::{OperationBufferStorageKind, ResolvedValueRole, RetainedPlanDependencyAuthority};
use crate::vnext::{
    BatchWorkShape, BufferDescriptor, DeviceBufferRetention, ElementType,
    EncodedReusableExecutionBindings, ProgramBindingNodeBinding, SegmentBackingWindowView,
    VNextError, WeightId,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SegmentBindingRegionSelector {
    Value {
        role: ResolvedValueRole,
        ordinal: u32,
        component: Option<WeightId>,
    },
    Scratch,
    Persistent,
    ProgramBinding,
}

/// A finite core-evaluated window rule. CurrentResource is evaluated from this
/// wave's admitted resource, so KV/scale page growth never reuses a cold extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentBindingRegionExtent {
    Exact(u64),
    CurrentResource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentBindingRegionRequest {
    pub selector: SegmentBindingRegionSelector,
    pub offset_bytes: u64,
    pub extent: SegmentBindingRegionExtent,
    pub element_type: ElementType,
    pub alignment_bytes: u64,
}

/// A declaration, not a capability. The core must issue a NEW dependency scope
/// after checking the current Plan owners and every range in the entire segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentBindingDependency {
    pub input_ordinal: u32,
    pub component_id: WeightId,
    pub source_offset_bytes: u64,
    pub source_length_bytes: u64,
    pub persistent_offset_bytes: u64,
    pub persistent_length_bytes: u64,
    pub alignment_bytes: u64,
    pub validation_identity: String,
}

impl SegmentBindingDependency {
    pub fn as_spec(&self) -> super::RetainedPlanDependencySpec<'_> {
        super::RetainedPlanDependencySpec {
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

/// Provider-private state may own Plan weights/validation banks and immutable
/// geometry. It must not own Step/Invocation/Request/Sequence allocations,
/// prior encoded commands, or dependency authorities. No callbacks are stored
/// in the declarative resource schema or run while the pool permit is held.
pub struct SegmentBindingDeclaration {
    pub(crate) regions: Vec<SegmentBindingRegionRequest>,
    pub(crate) dependencies: Vec<SegmentBindingDependency>,
    pub(crate) state: Arc<dyn Any + Send + Sync>,
}

impl SegmentBindingDeclaration {
    pub fn new(
        regions: Vec<SegmentBindingRegionRequest>,
        dependencies: Vec<SegmentBindingDependency>,
        state: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, VNextError> {
        for region in &regions {
            if !region.alignment_bytes.is_power_of_two()
                || matches!(region.extent, SegmentBindingRegionExtent::Exact(n) if n == 0 || region.offset_bytes.checked_add(n).is_none())
            {
                return Err(invalid_operation(
                    "segment binding region is empty or overflowing",
                ));
            }
        }
        for dependency in &dependencies {
            if dependency.validation_identity.is_empty()
                || !dependency.alignment_bytes.is_power_of_two()
                || dependency.source_length_bytes == 0
                || dependency.persistent_length_bytes == 0
                || dependency
                    .source_offset_bytes
                    .checked_add(dependency.source_length_bytes)
                    .is_none()
                || dependency
                    .persistent_offset_bytes
                    .checked_add(dependency.persistent_length_bytes)
                    .is_none()
            {
                return Err(invalid_operation(
                    "segment binding dependency range is invalid",
                ));
            }
        }
        Ok(Self {
            regions,
            dependencies,
            state,
        })
    }

    pub fn regions(&self) -> &[SegmentBindingRegionRequest] {
        &self.regions
    }
    pub fn dependencies(&self) -> &[SegmentBindingDependency] {
        &self.dependencies
    }
    /// Only the backend interprets its opaque Plan state. Core retains and
    /// transports this borrow without selecting a concrete provider type.
    pub fn provider_state(&self) -> &(dyn Any + Send + Sync) {
        self.state.as_ref()
    }
}

/// Explicit diagnostic only: production execution does not build a reference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SegmentBindingOracleMode {
    #[default]
    Disabled,
    CompareReference,
}

pub use ferrum_types::SegmentBindingOwnerViewMode;

/// An index into one current patch, never a cross-wave ownership identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentBindingOwnerIndex(pub(crate) usize);

impl SegmentBindingOwnerIndex {
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Borrowed exact ownership. A command must retain this capability and its
/// backend allocation before returning from the synchronous segment encoder.
pub struct SegmentBindingOwnerView<'a, B> {
    pub(crate) buffer: &'a B,
    pub(crate) retention: &'a DeviceBufferRetention,
}

impl<'a, B> SegmentBindingOwnerView<'a, B> {
    pub fn buffer_and_retention(&self) -> (&'a B, &'a DeviceBufferRetention) {
        (self.buffer, self.retention)
    }
}

/// Borrowed command channels for the same-wave diagnostic. No authority can be
/// moved, minted, or submitted through this view.
pub struct SegmentBindingOracleCommands<'a, C> {
    pub program_bindings: &'a [C],
    pub dynamic_bindings: &'a [C],
    pub result_bindings: &'a [C],
    pub retained_dependencies: Vec<&'a C>,
}

/// Core-issued retained physical fact. Providers cannot mint this from IDs.
/// The buffer borrow is tied to the owning fresh segment batch, and any command
/// must retain the returned capability before that batch is released.
pub struct SegmentBindingPhysicalRegion<'a, B> {
    pub(crate) buffer: &'a B,
    pub(crate) physical_range: Range<u64>,
    pub(crate) logical_offset_bytes: u64,
    pub(crate) retention: DeviceBufferRetention,
}

impl<'a, B> SegmentBindingPhysicalRegion<'a, B> {
    pub fn buffer_and_physical_range(&self) -> (&'a B, Range<u64>, DeviceBufferRetention) {
        (
            self.buffer,
            self.physical_range.clone(),
            self.retention.clone(),
        )
    }
    pub fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }
    pub fn length_bytes(&self) -> u64 {
        self.physical_range.end - self.physical_range.start
    }
}

pub struct PreparedSegmentBindingRegion<'a, B> {
    pub(crate) descriptor: BufferDescriptor,
    pub(crate) storage_kind: OperationBufferStorageKind,
    pub(crate) source: SegmentBindingPhysicalSource<'a, B>,
}

pub(crate) enum SegmentBindingPhysicalSource<'a, B> {
    Legacy(Vec<SegmentBindingPhysicalRegion<'a, B>>),
    Plan {
        owner_index: SegmentBindingOwnerIndex,
        buffer: &'a B,
        physical_range: Range<u64>,
        retention: &'a DeviceBufferRetention,
    },
    Dynamic {
        window: SegmentBackingWindowView<'a, B>,
        owner_base: usize,
    },
}

/// One exact checked occurrence. The owner may retain a larger allocation;
/// only this range, not that allocation's size, is authorized by this fact.
pub struct SegmentBindingPhysicalRegionView<'a, B> {
    buffer: &'a B,
    physical_range: Range<u64>,
    logical_offset_bytes: u64,
    retention: &'a DeviceBufferRetention,
    owner_index: Option<SegmentBindingOwnerIndex>,
}

impl<'a, B> SegmentBindingPhysicalRegionView<'a, B> {
    pub fn borrowed_buffer_and_physical_range(
        &self,
    ) -> (&'a B, Range<u64>, &'a DeviceBufferRetention) {
        (self.buffer, self.physical_range.clone(), self.retention)
    }

    pub fn buffer_and_physical_range(&self) -> (&'a B, Range<u64>, DeviceBufferRetention) {
        (
            self.buffer,
            self.physical_range.clone(),
            self.retention.clone(),
        )
    }

    pub const fn owner_index(&self) -> Option<SegmentBindingOwnerIndex> {
        self.owner_index
    }

    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub fn length_bytes(&self) -> u64 {
        self.physical_range.end - self.physical_range.start
    }
}

pub struct SegmentBindingPhysicalRegions<'a, B> {
    source: &'a SegmentBindingPhysicalSource<'a, B>,
    next: usize,
    len: usize,
}

impl<'a, B> Iterator for SegmentBindingPhysicalRegions<'a, B> {
    type Item = SegmentBindingPhysicalRegionView<'a, B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }
        let result = match self.source {
            SegmentBindingPhysicalSource::Legacy(regions) => {
                let region = regions.get(self.next)?;
                SegmentBindingPhysicalRegionView {
                    buffer: region.buffer,
                    physical_range: region.physical_range.clone(),
                    logical_offset_bytes: region.logical_offset_bytes,
                    retention: &region.retention,
                    owner_index: None,
                }
            }
            SegmentBindingPhysicalSource::Plan {
                owner_index,
                buffer,
                physical_range,
                retention,
            } => SegmentBindingPhysicalRegionView {
                buffer: *buffer,
                physical_range: physical_range.clone(),
                logical_offset_bytes: 0,
                retention: *retention,
                owner_index: Some(*owner_index),
            },
            SegmentBindingPhysicalSource::Dynamic { window, owner_base } => {
                let region = window.physical_region(self.next)?;
                let (buffer, physical_range, retention) =
                    region.borrowed_buffer_and_physical_range();
                SegmentBindingPhysicalRegionView {
                    buffer,
                    physical_range,
                    logical_offset_bytes: region.logical_offset_bytes(),
                    retention,
                    // The complete owner table length was checked before any
                    // projection was built; binding indices belong to it.
                    owner_index: Some(SegmentBindingOwnerIndex(
                        owner_base + region.binding_index(),
                    )),
                }
            }
        };
        self.next += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl<B> ExactSizeIterator for SegmentBindingPhysicalRegions<'_, B> {}
impl<B> std::iter::FusedIterator for SegmentBindingPhysicalRegions<'_, B> {}

impl<B> PreparedSegmentBindingRegion<'_, B> {
    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }
    pub fn storage_kind(&self) -> OperationBufferStorageKind {
        self.storage_kind
    }
    pub fn physical_regions(&self) -> SegmentBindingPhysicalRegions<'_, B> {
        let len = match &self.source {
            SegmentBindingPhysicalSource::Legacy(regions) => regions.len(),
            SegmentBindingPhysicalSource::Plan { .. } => 1,
            SegmentBindingPhysicalSource::Dynamic { window, .. } => window.region_count(),
        };
        SegmentBindingPhysicalRegions {
            source: &self.source,
            next: 0,
            len,
        }
    }
}

/// One node's finite projections into the fresh segment table. This is NOT a
/// BatchedOperationInvocation and carries no provider callback. Backend code
/// interprets only its own closed cold recipe within the whole-segment encoder.
pub struct PreparedSegmentBindingNode<'a, B> {
    pub(crate) node_index: u32,
    pub(crate) declaration: &'a SegmentBindingDeclaration,
    pub(crate) program_binding: Option<ProgramBindingNodeBinding>,
    pub(crate) work_shape: &'a BatchWorkShape,
    /// Participant-major, followed by declaration region index.
    pub(crate) regions: Vec<PreparedSegmentBindingRegion<'a, B>>,
    pub(crate) dependencies: Vec<Option<RetainedPlanDependencyAuthority>>,
}

impl<B> PreparedSegmentBindingNode<'_, B> {
    pub fn node_index(&self) -> u32 {
        self.node_index
    }
    pub fn declaration(&self) -> &SegmentBindingDeclaration {
        self.declaration
    }
    pub fn program_binding(&self) -> Option<&ProgramBindingNodeBinding> {
        self.program_binding.as_ref()
    }
    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work_shape
    }
    pub fn participant_count(&self) -> usize {
        self.work_shape.participant_token_ranges().len()
    }
    pub fn region(
        &self,
        participant: usize,
        declaration: usize,
    ) -> Result<&PreparedSegmentBindingRegion<'_, B>, VNextError> {
        if participant >= self.participant_count() || declaration >= self.declaration.regions.len()
        {
            return Err(invalid_operation(
                "segment binding region index is outside its schema",
            ));
        }
        let index = participant
            .checked_mul(self.declaration.regions.len())
            .and_then(|start| start.checked_add(declaration))
            .ok_or_else(|| invalid_operation("segment binding region index overflows"))?;
        self.regions
            .get(index)
            .ok_or_else(|| invalid_operation("segment binding facts are incomplete"))
    }
    pub fn take_dependency(
        &mut self,
        index: usize,
    ) -> Result<RetainedPlanDependencyAuthority, VNextError> {
        self.dependencies
            .get_mut(index)
            .and_then(Option::take)
            .ok_or_else(|| invalid_operation("segment binding dependency is absent or consumed"))
    }
}

pub struct PreparedSegmentBindingPatch<'a, B> {
    pub(crate) owner_view_mode: SegmentBindingOwnerViewMode,
    pub(crate) owners: Vec<SegmentBindingOwnerView<'a, B>>,
    pub(crate) nodes: Vec<PreparedSegmentBindingNode<'a, B>>,
}

impl<'a, B> PreparedSegmentBindingPatch<'a, B> {
    pub const fn owner_view_mode(&self) -> SegmentBindingOwnerViewMode {
        self.owner_view_mode
    }

    pub fn owners(&self) -> &[SegmentBindingOwnerView<'a, B>] {
        &self.owners
    }

    pub fn into_parts(
        self,
    ) -> (
        Vec<SegmentBindingOwnerView<'a, B>>,
        Vec<PreparedSegmentBindingNode<'a, B>>,
    ) {
        (self.owners, self.nodes)
    }

    pub fn nodes(&self) -> &[PreparedSegmentBindingNode<'a, B>] {
        &self.nodes
    }
    pub fn into_nodes(self) -> Vec<PreparedSegmentBindingNode<'a, B>> {
        self.nodes
    }
}

/// Preserve original node order and attribution around the single resident
/// launch. The core still validates returned node coverage and channel counts.
pub struct EncodedSegmentBindingNode<C> {
    pub node_index: u32,
    pub bindings: EncodedReusableExecutionBindings<C>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_declaration_rejects_empty_overflowing_or_unaligned_contracts() {
        let region = SegmentBindingRegionRequest {
            selector: SegmentBindingRegionSelector::ProgramBinding,
            offset_bytes: 0,
            extent: SegmentBindingRegionExtent::Exact(16),
            element_type: ElementType::U8,
            alignment_bytes: 4,
        };
        let valid =
            |request| SegmentBindingDeclaration::new(vec![request], Vec::new(), Arc::new(()));
        assert!(valid(region.clone()).is_ok());
        assert!(valid(SegmentBindingRegionRequest {
            extent: SegmentBindingRegionExtent::Exact(0),
            ..region.clone()
        })
        .is_err());
        assert!(valid(SegmentBindingRegionRequest {
            offset_bytes: u64::MAX - 15,
            ..region.clone()
        })
        .is_err());
        assert!(valid(SegmentBindingRegionRequest {
            alignment_bytes: 3,
            ..region
        })
        .is_err());
    }

    #[test]
    fn segment_dependency_keeps_exact_subranges_and_rejects_overflow() {
        let dependency = SegmentBindingDependency {
            input_ordinal: 2,
            component_id: WeightId::new("segment.weight").unwrap(),
            source_offset_bytes: 4,
            source_length_bytes: 64,
            persistent_offset_bytes: 12,
            persistent_length_bytes: 4,
            alignment_bytes: 4,
            validation_identity: "validation.segment".into(),
        };
        let declaration =
            SegmentBindingDeclaration::new(Vec::new(), vec![dependency.clone()], Arc::new(()))
                .unwrap();
        let spec = declaration.dependencies()[0].as_spec();
        assert_eq!(spec.source_offset_bytes, 4);
        assert_eq!(spec.source_length_bytes, 64);
        assert_eq!(spec.persistent_offset_bytes, 12);
        assert_eq!(spec.persistent_length_bytes, 4);
        for bad in [
            SegmentBindingDependency {
                source_offset_bytes: u64::MAX - 32,
                ..dependency.clone()
            },
            SegmentBindingDependency {
                persistent_offset_bytes: u64::MAX - 2,
                ..dependency.clone()
            },
            SegmentBindingDependency {
                source_length_bytes: 0,
                ..dependency.clone()
            },
            SegmentBindingDependency {
                validation_identity: String::new(),
                ..dependency
            },
        ] {
            assert!(SegmentBindingDeclaration::new(Vec::new(), vec![bad], Arc::new(())).is_err());
        }
    }
}
