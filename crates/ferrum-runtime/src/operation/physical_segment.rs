//! Borrowed whole-wave input backed by retained Sequence physical tables.
//! This representation never contains a PreparedSegmentBindingFrame.
use super::foundation::invalid_operation;
use super::segment_compile::CompiledSegmentBindingRecipe;
use crate::resource::{SegmentPlanResourceView, SegmentWindowResource};
use crate::*;
use std::ops::Range;
use std::sync::Arc;

pub(super) enum PhysicalFrameResource {
    Plan(usize),
    Sequence { ordinal: usize, first_size: usize },
    Fresh(SegmentWindowResource),
}

pub struct PreparedPhysicalSegmentWave<'w, B> {
    pub(super) recipe: &'w CompiledSegmentBindingRecipe,
    pub(super) plan_views: &'w [SegmentPlanResourceView<'w, B>],
    pub(super) tables: &'w [Arc<SequencePhysicalBindingTable<B>>],
    pub(super) resources: Vec<PhysicalFrameResource>,
    pub(super) sequence_bytes: Vec<u64>,
    pub(super) fresh_backing: Option<&'w SegmentBackingBatch<B>>,
    pub(super) fresh_expected: &'w [SegmentBackingExpectation],
    pub(super) work_shape: crate::TokenBatchWorkView<'w>,
    pub(super) program_bindings: Vec<Option<ProgramBindingNodeBinding>>,
    pub(super) dependencies: Vec<Vec<Option<RetainedPlanDependencyAuthority>>>,
}

impl<'w, B> PreparedPhysicalSegmentWave<'w, B> {
    pub fn node_count(&self) -> usize {
        self.recipe.nodes.len()
    }
    pub fn sequence_tables(&self) -> &[Arc<SequencePhysicalBindingTable<B>>] {
        self.tables
    }
    pub fn node(
        &self,
        index: usize,
    ) -> Result<PreparedPhysicalSegmentNodeView<'_, 'w, B>, VNextError> {
        if index >= self.node_count() {
            return Err(invalid_operation("physical segment node is absent"));
        }
        Ok(PreparedPhysicalSegmentNodeView { wave: self, index })
    }
    pub fn dependency_count(&self, node: usize) -> Option<usize> {
        self.dependencies.get(node).map(Vec::len)
    }
    pub fn dependency_identity(
        &self,
        node: usize,
        dependency: usize,
    ) -> Option<&RetainedPlanDependencyIdentity> {
        self.dependencies
            .get(node)?
            .get(dependency)?
            .as_ref()
            .map(|authority| &authority.identity)
    }
    pub fn take_dependency(
        &mut self,
        node: usize,
        dependency: usize,
    ) -> Result<RetainedPlanDependencyAuthority, VNextError> {
        self.dependencies
            .get_mut(node)
            .and_then(|row| row.get_mut(dependency))
            .and_then(Option::take)
            .ok_or_else(|| invalid_operation("physical segment dependency is absent or consumed"))
    }
}

pub struct PreparedPhysicalSegmentNodeView<'f, 'w, B> {
    wave: &'f PreparedPhysicalSegmentWave<'w, B>,
    index: usize,
}

impl<'f, 'w, B> PreparedPhysicalSegmentNodeView<'f, 'w, B> {
    pub fn node_index(&self) -> u32 {
        self.wave.recipe.nodes[self.index].node_index as u32
    }
    pub fn participant_count(&self) -> usize {
        self.wave.work_shape.participant_token_ranges().len()
    }
    pub fn work_shape(&self) -> &crate::TokenBatchWorkView<'w> {
        &self.wave.work_shape
    }
    pub fn program_binding(&self) -> Option<&'f ProgramBindingNodeBinding> {
        self.wave.program_bindings[self.index].as_ref()
    }
    pub fn declaration(&self) -> &'f SegmentBindingDeclaration {
        &self.wave.recipe.nodes[self.index].declaration
    }

    pub fn region(
        &self,
        participant: usize,
        declaration: usize,
    ) -> Result<PhysicalSegmentRegionView<'f, B>, VNextError> {
        if participant >= self.participant_count() {
            return Err(invalid_operation("physical segment participant is absent"));
        }
        let node = &self.wave.recipe.nodes[self.index];
        let region = node
            .regions
            .get(declaration)
            .ok_or_else(|| invalid_operation("physical segment region is absent"))?;
        let resource = self
            .wave
            .resources
            .get(region.resource_index)
            .ok_or_else(|| invalid_operation("physical segment resource is absent"))?;
        let resource_id = &self.wave.recipe.resources[region.resource_index].resource_id;
        let (available, alignment, usage, element, storage, source) = match resource {
            PhysicalFrameResource::Plan(index) => {
                let plan = &self.wave.plan_views[*index];
                let d = plan.leased.committed_descriptor();
                (
                    d.size_bytes,
                    d.alignment_bytes,
                    d.usage,
                    d.element_type,
                    OperationBufferStorageKind::StaticContiguous,
                    PhysicalRegionSource::Plan {
                        buffer: plan.leased.buffer(),
                        descriptor: d,
                        retention: &plan.retention,
                    },
                )
            }
            PhysicalFrameResource::Sequence {
                ordinal,
                first_size,
            } => {
                let table = &self.wave.tables[participant];
                let row = table
                    .resource(*ordinal)
                    .ok_or_else(|| invalid_operation("physical Sequence row is absent"))?;
                let d = row.descriptor();
                (
                    self.wave.sequence_bytes[first_size + participant],
                    d.alignment_bytes(),
                    d.usage(),
                    d.element_type(),
                    storage_kind(d.storage_profile()),
                    PhysicalRegionSource::Sequence {
                        table,
                        ordinal: *ordinal,
                    },
                )
            }
            PhysicalFrameResource::Fresh(rows) => {
                let (request, offset, bytes) = rows.row(participant, self.wave.fresh_expected)?;
                let d = &self.wave.fresh_expected[request];
                let batch = self
                    .wave
                    .fresh_backing
                    .ok_or_else(|| invalid_operation("physical segment fresh backing is absent"))?;
                (
                    bytes,
                    d.alignment_bytes,
                    d.usage,
                    d.element_type,
                    storage_kind(d.storage_profile),
                    PhysicalRegionSource::Fresh {
                        batch,
                        request,
                        participant_offset: offset,
                    },
                )
            }
        };
        let length = region.length_for_current_view(available)?;
        if element != region.request.element_type
            || alignment < region.request.alignment_bytes
            || !region
                .offset_bytes
                .is_multiple_of(region.request.alignment_bytes)
        {
            return Err(invalid_operation(
                "physical segment region has different type or alignment",
            ));
        }
        Ok(PhysicalSegmentRegionView {
            descriptor: SegmentBindingDescriptorView {
                resource_id,
                size_bytes: length,
                alignment_bytes: alignment,
                usage,
                element_type: element,
            },
            storage,
            source,
            offset: region.offset_bytes,
            length,
        })
    }

    /// Retained physical capacity for a resident program's paged Sequence state.
    ///
    /// This is separate from `region`, whose exact current logical window is
    /// unchanged. Only a whole-resource state declaration can retain its owned
    /// spare pages. The backend must still derive valid entries from fresh work;
    /// capacity does not certify that any future token has executed.
    pub fn resident_sequence_capacity_region(
        &self,
        participant: usize,
        declaration: usize,
    ) -> Result<PhysicalSegmentRegionView<'f, B>, VNextError> {
        let current = self.region(participant, declaration)?;
        let rule = &self.wave.recipe.nodes[self.index].regions[declaration];
        if rule.request.extent != SegmentBindingRegionExtent::CurrentResource
            || current.offset != 0
            || current.descriptor.usage != BufferUsage::State
            || current.storage != OperationBufferStorageKind::DynamicPaged
        {
            return Err(invalid_operation(
                "resident capacity requires a whole paged Sequence state declaration",
            ));
        }
        let PhysicalRegionSource::Sequence { table, ordinal } = current.source else {
            return Err(invalid_operation(
                "resident capacity requires retained Sequence ownership",
            ));
        };
        let row = table
            .resource(ordinal)
            .ok_or_else(|| invalid_operation("resident capacity Sequence row is absent"))?;
        let capacity = row.descriptor().logical_size_bytes();
        if capacity < current.length
            || row.resource_id() != current.descriptor.resource_id
            || rule
                .component_length
                .is_some_and(|maximum| capacity > maximum)
        {
            return Err(invalid_operation(
                "resident capacity exceeds its owned resource declaration",
            ));
        }
        Ok(PhysicalSegmentRegionView {
            descriptor: SegmentBindingDescriptorView {
                size_bytes: capacity,
                ..current.descriptor
            },
            storage: current.storage,
            source: PhysicalRegionSource::Sequence { table, ordinal },
            offset: 0,
            length: capacity,
        })
    }
}

fn storage_kind(profile: DynamicStorageProfile) -> OperationBufferStorageKind {
    match profile.view() {
        DynamicStorageView::Contiguous => OperationBufferStorageKind::DynamicContiguous,
        DynamicStorageView::PagedRegions { .. } => OperationBufferStorageKind::DynamicPaged,
    }
}

enum PhysicalRegionSource<'a, B> {
    Plan {
        buffer: &'a B,
        descriptor: &'a BufferDescriptor,
        retention: &'a DeviceBufferRetention,
    },
    Sequence {
        table: &'a Arc<SequencePhysicalBindingTable<B>>,
        ordinal: usize,
    },
    Fresh {
        batch: &'a SegmentBackingBatch<B>,
        request: usize,
        participant_offset: u64,
    },
}

pub struct PhysicalSegmentRegionView<'a, B> {
    descriptor: SegmentBindingDescriptorView<'a>,
    storage: OperationBufferStorageKind,
    source: PhysicalRegionSource<'a, B>,
    offset: u64,
    length: u64,
}

impl<'a, B> PhysicalSegmentRegionView<'a, B> {
    pub fn descriptor(&self) -> SegmentBindingDescriptorView<'a> {
        self.descriptor
    }
    pub fn storage_kind(&self) -> OperationBufferStorageKind {
        self.storage
    }
    pub fn sequence_table(&self) -> Option<&'a Arc<SequencePhysicalBindingTable<B>>> {
        match &self.source {
            PhysicalRegionSource::Sequence { table, .. } => Some(*table),
            _ => None,
        }
    }
    pub fn resource_ordinal(&self) -> Option<usize> {
        match &self.source {
            PhysicalRegionSource::Sequence { ordinal, .. } => Some(*ordinal),
            _ => None,
        }
    }
    pub fn offset_bytes(&self) -> u64 {
        self.offset
    }
    pub fn length_bytes(&self) -> u64 {
        self.length
    }

    /// Borrowed projection for fresh/non-Sequence ranges and reference oracles.
    /// CUDA's hot Sequence route uses the retained backend table directly.
    pub fn physical_regions(
        &self,
    ) -> impl Iterator<Item = PhysicalSegmentPhysicalRegion<'a, B>> + '_ {
        let mut index = 0usize;
        std::iter::from_fn(move || loop {
            let current = index;
            index += 1;
            let (buffer, range, retention, logical) = match &self.source {
                PhysicalRegionSource::Plan {
                    buffer,
                    descriptor,
                    retention,
                } => {
                    if current != 0 {
                        return None;
                    }
                    (*buffer, 0..descriptor.size_bytes, *retention, 0)
                }
                PhysicalRegionSource::Sequence { table, ordinal } => {
                    let region = table.resource(*ordinal)?.physical_regions().nth(current)?;
                    let (buffer, range, retention) = region.borrowed_buffer_and_physical_range();
                    // The backing descriptor is available through the retained buffer row.
                    (buffer, range, retention, region.logical_offset_bytes())
                }
                PhysicalRegionSource::Fresh { batch, request, .. } => {
                    let region = batch.window(*request)?.physical_region(current)?;
                    let (buffer, range, retention) = region.borrowed_buffer_and_physical_range();
                    (buffer, range, retention, region.logical_offset_bytes())
                }
            };
            let base = match &self.source {
                PhysicalRegionSource::Fresh {
                    participant_offset, ..
                } => *participant_offset,
                _ => 0,
            };
            let start = base + self.offset;
            let end = start + self.length;
            let from = logical.max(start);
            let to = (logical + range.end - range.start).min(end);
            if from >= to {
                continue;
            }
            return Some(PhysicalSegmentPhysicalRegion {
                buffer,
                range: range.start + from - logical..range.start + to - logical,
                retention,
                logical_offset: from - start,
            });
        })
    }
}

pub struct PhysicalSegmentPhysicalRegion<'a, B> {
    buffer: &'a B,
    range: Range<u64>,
    retention: &'a DeviceBufferRetention,
    logical_offset: u64,
}
impl<'a, B> PhysicalSegmentPhysicalRegion<'a, B> {
    pub fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset
    }
    pub fn length_bytes(&self) -> u64 {
        self.range.end - self.range.start
    }
    pub fn borrowed_buffer_and_physical_range(
        &self,
    ) -> (&'a B, Range<u64>, &'a DeviceBufferRetention) {
        (self.buffer, self.range.clone(), self.retention)
    }
}
