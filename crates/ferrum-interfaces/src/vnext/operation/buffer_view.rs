use std::ops::Range;

use super::super::{
    AllocationLifetime, BatchWorkShape, BufferDescriptor, BufferUsage, DeviceBufferRetention,
    DeviceRuntime, DynamicResourceDemand, DynamicResourceShape, LeasedBufferView,
    LogicalBackingBufferView, LogicalBackingSegmentBinding, NodeWorkContract, ResourceId,
    ResourceTransactionIdentity, VNextError,
};
use super::foundation::invalid_operation;
use super::{DynamicStorageView, ResolvedStorageComponent, ResolvedValueBinding};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ValueBindingPhysicalCoverage {
    CanonicalComponent,
    RuntimeTokenView,
}

pub(super) fn validate_value_binding_physical_coverage(
    work: &NodeWorkContract,
    binding: &ResolvedValueBinding,
    component: &ResolvedStorageComponent,
    descriptor: &BufferDescriptor,
    dynamic_demand: Option<&DynamicResourceDemand>,
    value_alignment_bytes: u64,
) -> Result<ValueBindingPhysicalCoverage, VNextError> {
    let required_end = component
        .offset_bytes()
        .checked_add(component.length_bytes())
        .ok_or_else(|| invalid_operation("bound component range overflows u64"))?;
    if descriptor.usage != binding.usage()
        || descriptor.element_type != component.element_type()
        || descriptor.alignment_bytes < value_alignment_bytes
        || descriptor.alignment_bytes % value_alignment_bytes != 0
        || component.offset_bytes() % value_alignment_bytes != 0
    {
        return Err(invalid_operation(format!(
            "resource `{}` differs from its value binding",
            component.resource_id()
        )));
    }

    let Some(projection) = work.token_projection(binding.role(), binding.ordinal()) else {
        if required_end > descriptor.size_bytes {
            return Err(invalid_operation(format!(
                "resource `{}` differs from its value binding",
                component.resource_id()
            )));
        }
        return Ok(ValueBindingPhysicalCoverage::CanonicalComponent);
    };

    let axis = usize::try_from(projection.axis())
        .map_err(|_| invalid_operation("token projection axis exceeds usize"))?;
    let canonical_extent = projection.canonical_extent();
    let bytes_per_token = component
        .length_bytes()
        .checked_div(canonical_extent)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| invalid_operation("token projection has zero canonical extent"))?;
    let demand_matches = matches!(
        dynamic_demand,
        Some(DynamicResourceDemand::Tokens {
            bytes_per_token: planned_bytes_per_token,
            ..
        }) if *planned_bytes_per_token == bytes_per_token
    );
    if binding.usage() != BufferUsage::Activations
        || component.offset_bytes() != 0
        || component.length_bytes() % canonical_extent != 0
        || usize::try_from(projection.rank()).ok() != Some(binding.tensor().dimensions().len())
        || binding.tensor().dimensions().get(axis) != Some(&canonical_extent)
        || !demand_matches
        || descriptor.size_bytes < bytes_per_token
    {
        return Err(invalid_operation(format!(
            "resource `{}` differs from its token-projected value binding",
            component.resource_id()
        )));
    }

    Ok(ValueBindingPhysicalCoverage::RuntimeTokenView)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepParticipantRangeCoordinates {
    SourceToken,
    ParticipantLocal,
}

/// Maps source-token coordinates used by product input uploads into shared
/// Step backing. Fixed participant values already use local coordinates.
pub(crate) fn translate_step_participant_upload_range(
    demand: &DynamicResourceDemand,
    work_shape: &BatchWorkShape,
    participant_index: usize,
    semantic_range: Range<u64>,
) -> Result<Range<u64>, VNextError> {
    translate_step_participant_range(
        demand,
        work_shape,
        participant_index,
        semantic_range,
        StepParticipantRangeCoordinates::SourceToken,
    )
}

/// Maps participant-local completion coordinates into shared Step backing.
pub(crate) fn translate_step_participant_readback_range(
    demand: &DynamicResourceDemand,
    work_shape: &BatchWorkShape,
    participant_index: usize,
    semantic_range: Range<u64>,
) -> Result<Range<u64>, VNextError> {
    translate_step_participant_range(
        demand,
        work_shape,
        participant_index,
        semantic_range,
        StepParticipantRangeCoordinates::ParticipantLocal,
    )
}

fn translate_step_participant_range(
    demand: &DynamicResourceDemand,
    work_shape: &BatchWorkShape,
    participant_index: usize,
    semantic_range: Range<u64>,
    coordinates: StepParticipantRangeCoordinates,
) -> Result<Range<u64>, VNextError> {
    if semantic_range.start >= semantic_range.end {
        return Err(invalid_operation(
            "participant resource projection has an empty semantic range",
        ));
    }
    if participant_index >= work_shape.participant_token_ranges().len() {
        return Err(invalid_operation(
            "participant resource projection is out of range",
        ));
    }
    match demand {
        DynamicResourceDemand::ActualSequences {
            bytes_per_sequence,
            maximum_sequences,
        } => {
            if work_shape.immediate_sequences() > *maximum_sequences
                || semantic_range.end > *bytes_per_sequence
            {
                return Err(invalid_operation(
                    "participant fixed resource projection exceeds its planned stride",
                ));
            }
            let base = bytes_per_sequence
                .checked_mul(u64::try_from(participant_index).map_err(|_| {
                    invalid_operation("participant resource projection exceeds u64")
                })?)
                .ok_or_else(|| {
                    invalid_operation("participant fixed resource projection overflows u64")
                })?;
            let translated_start = base
                .checked_add(semantic_range.start)
                .ok_or_else(|| invalid_operation("participant fixed range overflows u64"))?;
            let translated_end = base
                .checked_add(semantic_range.end)
                .ok_or_else(|| invalid_operation("participant fixed range overflows u64"))?;
            Ok(translated_start..translated_end)
        }
        DynamicResourceDemand::Tokens {
            bytes_per_token,
            maximum_tokens,
        } => {
            if work_shape.immediate_tokens() > *maximum_tokens {
                return Err(invalid_operation(
                    "participant token resource projection exceeds its planned ceiling",
                ));
            }
            let token_range = &work_shape.participant_token_ranges()[participant_index];
            let source = token_range.source_token_range();
            let packed = token_range.immediate_token_range();
            let source_start = source
                .start
                .checked_mul(*bytes_per_token)
                .ok_or_else(|| invalid_operation("source token byte offset overflows u64"))?;
            let source_end = source
                .end
                .checked_mul(*bytes_per_token)
                .ok_or_else(|| invalid_operation("source token byte range overflows u64"))?;
            let packed_start = packed
                .start
                .checked_mul(*bytes_per_token)
                .ok_or_else(|| invalid_operation("packed token byte offset overflows u64"))?;
            let relative_range = match coordinates {
                StepParticipantRangeCoordinates::SourceToken => {
                    if semantic_range.start < source_start || semantic_range.end > source_end {
                        return Err(invalid_operation(
                            "participant token resource projection is outside its source span",
                        ));
                    }
                    semantic_range.start - source_start..semantic_range.end - source_start
                }
                StepParticipantRangeCoordinates::ParticipantLocal => {
                    let span_bytes = source_end - source_start;
                    if semantic_range.end > span_bytes {
                        return Err(invalid_operation(
                            "participant token readback exceeds its immediate span",
                        ));
                    }
                    semantic_range
                }
            };
            let translated_start = packed_start
                .checked_add(relative_range.start)
                .ok_or_else(|| invalid_operation("packed token projection overflows u64"))?;
            let translated_end = translated_start
                .checked_add(relative_range.end - relative_range.start)
                .ok_or_else(|| invalid_operation("packed token range overflows u64"))?;
            Ok(translated_start..translated_end)
        }
        // Fixed/page-shaped internal activations are batch-wide resources. The
        // planner gives product-fixed I/O an ActualSequences demand, so only
        // that typed demand is participant-local here.
        _ => Ok(semantic_range),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationBufferStorageKind {
    StaticContiguous,
    DynamicContiguous,
    DynamicPaged,
}

enum OperationBufferSource<'a, B> {
    Static {
        view: LeasedBufferView<'a, B>,
        retention: DeviceBufferRetention,
    },
    Backing(LogicalBackingBufferView<'a, B>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperationBufferCoverage {
    Exact,
    /// The operation sees an exact logical prefix while resource authority
    /// retains wider physical capacity for a frontier or reusable bucket.
    BackingWindow {
        offset_bytes: u64,
    },
}

enum OperationRegionSource<'a, B> {
    Contiguous {
        buffer: &'a B,
        physical_base_offset_bytes: u64,
        retention: DeviceBufferRetention,
    },
    Paged {
        bindings: &'a [LogicalBackingSegmentBinding<B>],
        logical_origin_bytes: u64,
    },
}

/// A checked logical range translated to physical device-buffer regions.
/// Dynamic buffers never expose an arena buffer without its physical offsets.
pub struct OperationBufferRegions<'a, B> {
    storage_kind: OperationBufferStorageKind,
    logical_offset_bytes: u64,
    logical_length_bytes: u64,
    source: OperationRegionSource<'a, B>,
}

impl<'a, B> OperationBufferRegions<'a, B> {
    pub const fn storage_kind(&self) -> OperationBufferStorageKind {
        self.storage_kind
    }

    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub const fn logical_length_bytes(&self) -> u64 {
        self.logical_length_bytes
    }

    pub fn iter(&self) -> OperationBufferRegionIter<'a, B> {
        let logical_end_bytes = self
            .logical_offset_bytes
            .checked_add(self.logical_length_bytes)
            .expect("validated operation logical range does not overflow");
        match &self.source {
            OperationRegionSource::Contiguous {
                buffer,
                physical_base_offset_bytes,
                retention,
            } => OperationBufferRegionIter {
                state: OperationBufferRegionIterState::Contiguous(Some(OperationPhysicalRegion {
                    buffer: *buffer,
                    logical_offset_bytes: self.logical_offset_bytes,
                    physical_offset_bytes: physical_base_offset_bytes
                        .checked_add(self.logical_offset_bytes)
                        .expect("validated contiguous physical range does not overflow"),
                    length_bytes: self.logical_length_bytes,
                    retention: retention.clone(),
                })),
            },
            OperationRegionSource::Paged {
                bindings,
                logical_origin_bytes,
            } => OperationBufferRegionIter {
                state: OperationBufferRegionIterState::Paged {
                    bindings: *bindings,
                    logical_origin_bytes: *logical_origin_bytes,
                    requested_start_bytes: logical_origin_bytes
                        .checked_add(self.logical_offset_bytes)
                        .expect("validated paged window start does not overflow"),
                    requested_end_bytes: logical_origin_bytes
                        .checked_add(logical_end_bytes)
                        .expect("validated paged window end does not overflow"),
                    next_segment: 0,
                    next_segment_logical_offset_bytes: 0,
                },
            },
        }
    }
}

/// One indivisible physical region. The buffer reference is intentionally
/// returned only together with the physical byte range.
pub struct OperationPhysicalRegion<'a, B> {
    buffer: &'a B,
    logical_offset_bytes: u64,
    physical_offset_bytes: u64,
    length_bytes: u64,
    retention: DeviceBufferRetention,
}

impl<'a, B> OperationPhysicalRegion<'a, B> {
    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }

    pub fn buffer_and_physical_range(
        &self,
    ) -> (&'a B, std::ops::Range<u64>, DeviceBufferRetention) {
        (
            self.buffer,
            self.physical_offset_bytes
                ..self
                    .physical_offset_bytes
                    .checked_add(self.length_bytes)
                    .expect("validated physical region does not overflow"),
            self.retention.clone(),
        )
    }
}

pub struct OperationBufferRegionIter<'a, B> {
    state: OperationBufferRegionIterState<'a, B>,
}

enum OperationBufferRegionIterState<'a, B> {
    Contiguous(Option<OperationPhysicalRegion<'a, B>>),
    Paged {
        bindings: &'a [LogicalBackingSegmentBinding<B>],
        logical_origin_bytes: u64,
        requested_start_bytes: u64,
        requested_end_bytes: u64,
        next_segment: usize,
        next_segment_logical_offset_bytes: u64,
    },
}

impl<'a, B> Iterator for OperationBufferRegionIter<'a, B> {
    type Item = OperationPhysicalRegion<'a, B>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.state {
            OperationBufferRegionIterState::Contiguous(region) => region.take(),
            OperationBufferRegionIterState::Paged {
                bindings,
                logical_origin_bytes,
                requested_start_bytes,
                requested_end_bytes,
                next_segment,
                next_segment_logical_offset_bytes,
            } => {
                while let Some(binding) = bindings.get(*next_segment) {
                    *next_segment += 1;
                    let (segment_logical_end, region) = translate_paged_segment(
                        binding.buffer(),
                        binding.retention(),
                        binding.segment().offset_bytes(),
                        binding.segment().length_bytes(),
                        *next_segment_logical_offset_bytes,
                        *requested_start_bytes,
                        *requested_end_bytes,
                    );
                    *next_segment_logical_offset_bytes = segment_logical_end;
                    if let Some(mut region) = region {
                        region.logical_offset_bytes = region
                            .logical_offset_bytes
                            .checked_sub(*logical_origin_bytes)
                            .expect("paged region is inside its validated backing window");
                        return Some(region);
                    }
                }
                None
            }
        }
    }
}

fn translate_paged_segment<'a, B>(
    buffer: &'a B,
    retention: DeviceBufferRetention,
    physical_offset_bytes: u64,
    length_bytes: u64,
    logical_start_bytes: u64,
    requested_start_bytes: u64,
    requested_end_bytes: u64,
) -> (u64, Option<OperationPhysicalRegion<'a, B>>) {
    let logical_end_bytes = logical_start_bytes
        .checked_add(length_bytes)
        .expect("validated backing segments do not overflow");
    let translated_start = logical_start_bytes.max(requested_start_bytes);
    let translated_end = logical_end_bytes.min(requested_end_bytes);
    let region = (translated_start < translated_end).then(|| OperationPhysicalRegion {
        buffer,
        logical_offset_bytes: translated_start,
        physical_offset_bytes: physical_offset_bytes
            .checked_add(translated_start - logical_start_bytes)
            .expect("validated paged physical range does not overflow"),
        length_bytes: translated_end - translated_start,
        retention,
    });
    (logical_end_bytes, region)
}

const fn operation_storage_kind(view: DynamicStorageView) -> OperationBufferStorageKind {
    match view {
        DynamicStorageView::Contiguous => OperationBufferStorageKind::DynamicContiguous,
        DynamicStorageView::PagedRegions { .. } => OperationBufferStorageKind::DynamicPaged,
    }
}

fn validate_dynamic_binding_layout(
    storage_kind: OperationBufferStorageKind,
    logical_size_bytes: u64,
    mut binding_lengths: impl ExactSizeIterator<Item = u64>,
    coverage: OperationBufferCoverage,
) -> Result<(), VNextError> {
    let binding_count = binding_lengths.len();
    if storage_kind == OperationBufferStorageKind::StaticContiguous {
        return Err(invalid_operation(
            "dynamic backing cannot claim static storage kind",
        ));
    }
    if binding_count == 0 {
        return Err(invalid_operation(
            "dynamic backing has no physical segment binding",
        ));
    }
    if storage_kind == OperationBufferStorageKind::DynamicContiguous && binding_count != 1 {
        return Err(invalid_operation(
            "contiguous dynamic storage requires one physical segment binding",
        ));
    }
    let covered = binding_lengths.try_fold(0_u64, |total, length_bytes| {
        total
            .checked_add(length_bytes)
            .ok_or_else(|| invalid_operation("backing segment coverage overflows u64"))
    })?;
    let window_offset = match coverage {
        OperationBufferCoverage::Exact => 0,
        OperationBufferCoverage::BackingWindow { offset_bytes } => offset_bytes,
    };
    let required_end = window_offset
        .checked_add(logical_size_bytes)
        .ok_or_else(|| invalid_operation("operation backing window overflows u64"))?;
    if covered < required_end
        || (coverage == OperationBufferCoverage::Exact && covered != logical_size_bytes)
    {
        return Err(invalid_operation(
            "dynamic backing segments do not cover the operation's exact logical view",
        ));
    }
    Ok(())
}

pub(super) fn sequence_execution_shape(
    committed: DynamicResourceShape,
    source_end_tokens: u64,
) -> Result<DynamicResourceShape, VNextError> {
    if committed.sequences() != 1
        || source_end_tokens == 0
        || source_end_tokens > committed.tokens()
    {
        return Err(invalid_operation(
            "sequence operation frontier is empty or exceeds committed backing",
        ));
    }
    Ok(DynamicResourceShape::from_validated(
        1,
        source_end_tokens,
        committed.pages(),
    ))
}

pub struct OperationBufferView<'a, B> {
    descriptor: BufferDescriptor,
    source: OperationBufferSource<'a, B>,
    coverage: OperationBufferCoverage,
    allocation_lifetime: AllocationLifetime,
    packed_batch_coordinates: bool,
}

impl<'a, B> OperationBufferView<'a, B> {
    pub(super) fn from_static(
        view: LeasedBufferView<'a, B>,
        retention: DeviceBufferRetention,
    ) -> Self {
        Self {
            descriptor: view.committed_descriptor().clone(),
            source: OperationBufferSource::Static { view, retention },
            coverage: OperationBufferCoverage::Exact,
            allocation_lifetime: AllocationLifetime::Plan,
            packed_batch_coordinates: false,
        }
    }

    pub(super) fn from_backing_exact(
        descriptor: BufferDescriptor,
        backing: LogicalBackingBufferView<'a, B>,
        allocation_lifetime: AllocationLifetime,
    ) -> Self {
        Self::from_backing(
            descriptor,
            backing,
            OperationBufferCoverage::Exact,
            allocation_lifetime,
        )
    }

    pub(super) fn from_backing_prefix(
        descriptor: BufferDescriptor,
        backing: LogicalBackingBufferView<'a, B>,
        allocation_lifetime: AllocationLifetime,
    ) -> Self {
        Self::from_backing(
            descriptor,
            backing,
            OperationBufferCoverage::BackingWindow { offset_bytes: 0 },
            allocation_lifetime,
        )
    }

    pub(super) fn from_backing_window(
        descriptor: BufferDescriptor,
        backing: LogicalBackingBufferView<'a, B>,
        offset_bytes: u64,
        allocation_lifetime: AllocationLifetime,
    ) -> Self {
        Self::from_backing(
            descriptor,
            backing,
            OperationBufferCoverage::BackingWindow { offset_bytes },
            allocation_lifetime,
        )
    }

    fn from_backing(
        descriptor: BufferDescriptor,
        backing: LogicalBackingBufferView<'a, B>,
        coverage: OperationBufferCoverage,
        allocation_lifetime: AllocationLifetime,
    ) -> Self {
        Self {
            descriptor,
            source: OperationBufferSource::Backing(backing),
            coverage,
            allocation_lifetime,
            packed_batch_coordinates: false,
        }
    }

    pub(super) fn with_packed_batch_coordinates(mut self, packed: bool) -> Self {
        self.packed_batch_coordinates = packed;
        self
    }

    pub(super) fn validate_runtime<R>(
        &self,
        runtime: &R,
        expected_static_identity: Option<&ResourceTransactionIdentity>,
    ) -> Result<(), VNextError>
    where
        R: DeviceRuntime<Buffer = B>,
    {
        match &self.source {
            OperationBufferSource::Static { view, .. } => {
                let actual = runtime.buffer_descriptor(view.buffer());
                if Some(view.identity()) != expected_static_identity
                    || &actual != view.committed_descriptor()
                    || view.generation() == 0
                {
                    return Err(invalid_operation(format!(
                        "runtime descriptor differs from committed static resource `{}`",
                        self.resource_id()
                    )));
                }
            }
            OperationBufferSource::Backing(backing_view) => {
                let bindings = backing_view.segment_bindings();
                if bindings.is_empty()
                    || bindings.len() != backing_view.committed_evidence_segments().count()
                    || bindings
                        .iter()
                        .zip(backing_view.committed_evidence_segments())
                        .any(|(binding, evidence)| {
                            let actual = runtime.buffer_descriptor(binding.buffer());
                            binding.segment() != evidence
                                || binding.chunk() != evidence.chunk()
                                || &actual != binding.descriptor()
                                || binding
                                    .segment()
                                    .offset_bytes()
                                    .checked_add(binding.segment().length_bytes())
                                    .is_none_or(|end| end > binding.descriptor().size_bytes)
                        })
                {
                    return Err(invalid_operation(format!(
                        "runtime descriptor differs from a committed backing chunk for `{}`",
                        self.resource_id()
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.descriptor.resource_id
    }

    pub fn descriptor(&self) -> &BufferDescriptor {
        &self.descriptor
    }

    pub const fn allocation_lifetime(&self) -> AllocationLifetime {
        self.allocation_lifetime
    }

    pub const fn uses_packed_batch_coordinates(&self) -> bool {
        self.packed_batch_coordinates
    }

    pub fn storage_kind(&self) -> OperationBufferStorageKind {
        match &self.source {
            OperationBufferSource::Static { .. } => OperationBufferStorageKind::StaticContiguous,
            OperationBufferSource::Backing(view) => {
                operation_storage_kind(view.storage_profile().view())
            }
        }
    }

    pub fn translate(
        &self,
        logical_offset_bytes: u64,
        logical_length_bytes: u64,
    ) -> Result<OperationBufferRegions<'_, B>, VNextError> {
        let logical_end_bytes = logical_offset_bytes
            .checked_add(logical_length_bytes)
            .ok_or_else(|| invalid_operation("operation logical buffer range overflows u64"))?;
        if logical_length_bytes == 0 || logical_end_bytes > self.descriptor.size_bytes {
            return Err(invalid_operation(
                "operation logical buffer range is empty or outside its resource",
            ));
        }
        match &self.source {
            OperationBufferSource::Static { view, retention } => Ok(OperationBufferRegions {
                storage_kind: OperationBufferStorageKind::StaticContiguous,
                logical_offset_bytes,
                logical_length_bytes,
                source: OperationRegionSource::Contiguous {
                    buffer: view.buffer(),
                    physical_base_offset_bytes: 0,
                    retention: retention.clone(),
                },
            }),
            OperationBufferSource::Backing(view) => {
                let bindings = view.segment_bindings();
                if bindings.len() != view.committed_evidence_segments().count()
                    || bindings.iter().zip(view.committed_evidence_segments()).any(
                        |(binding, segment)| {
                            binding.segment() != segment || binding.chunk() != segment.chunk()
                        },
                    )
                {
                    return Err(invalid_operation(
                        "dynamic backing bindings differ from committed segment evidence",
                    ));
                }
                let storage_kind = operation_storage_kind(view.storage_profile().view());
                let backing_window_offset = match self.coverage {
                    OperationBufferCoverage::Exact => 0,
                    OperationBufferCoverage::BackingWindow { offset_bytes } => offset_bytes,
                };
                validate_dynamic_binding_layout(
                    storage_kind,
                    self.descriptor.size_bytes,
                    bindings
                        .iter()
                        .map(|binding| binding.segment().length_bytes()),
                    self.coverage,
                )?;
                let source = match storage_kind {
                    OperationBufferStorageKind::DynamicContiguous => {
                        let binding = &bindings[0];
                        OperationRegionSource::Contiguous {
                            buffer: binding.buffer(),
                            physical_base_offset_bytes: binding
                                .segment()
                                .offset_bytes()
                                .checked_add(backing_window_offset)
                                .ok_or_else(|| {
                                    invalid_operation(
                                        "participant backing physical offset overflows u64",
                                    )
                                })?,
                            retention: binding.retention(),
                        }
                    }
                    OperationBufferStorageKind::DynamicPaged => OperationRegionSource::Paged {
                        bindings,
                        logical_origin_bytes: backing_window_offset,
                    },
                    OperationBufferStorageKind::StaticContiguous => unreachable!(
                        "dynamic storage kind was validated before region construction"
                    ),
                };
                Ok(OperationBufferRegions {
                    storage_kind,
                    logical_offset_bytes,
                    logical_length_bytes,
                    source,
                })
            }
        }
    }
}

#[cfg(test)]
mod operation_buffer_region_tests {
    use super::{
        sequence_execution_shape, translate_paged_segment,
        translate_step_participant_readback_range, translate_step_participant_upload_range,
        validate_dynamic_binding_layout, validate_value_binding_physical_coverage,
        OperationBufferCoverage, OperationBufferRegions, OperationBufferStorageKind,
        OperationRegionSource, ValueBindingPhysicalCoverage,
    };
    use crate::vnext::{
        AliasPolicy, BatchWorkShape, BufferDescriptor, BufferUsage, DeviceBufferRetention,
        DynamicResourceDemand, DynamicResourceShape, ElementType, NodeWorkContract, ProgramValueId,
        ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueBinding, ResolvedValueRole,
        ResolvedValueStorage, ResourceId, TensorAccess, TokenSpanWork,
    };
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    #[test]
    fn token_projection_validates_runtime_view_instead_of_canonical_extent() {
        let resource_id = ResourceId::new("resource.activation.token-ids").unwrap();
        let binding = ResolvedValueBinding::new(
            ProgramValueId::new("value.input.token-ids").unwrap(),
            ResolvedValueRole::Input,
            0,
            ResolvedTensorSpec::new(
                vec![128],
                ElementType::U32,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            TensorAccess::Read,
            AliasPolicy::NoAlias,
            BufferUsage::Activations,
            None,
            ResolvedValueStorage::single(resource_id.clone(), 0, 512, ElementType::U32).unwrap(),
        )
        .unwrap();
        let work: NodeWorkContract = serde_json::from_value(serde_json::json!({
            "tokens": {
                "source": {
                    "value_id": "value.input.token-ids",
                    "role": "input",
                    "ordinal": 0,
                    "axis": 0,
                    "rank": 1,
                    "canonical_extent": 128
                },
                "projections": [{
                    "value_id": "value.input.token-ids",
                    "role": "input",
                    "ordinal": 0,
                    "axis": 0,
                    "rank": 1,
                    "canonical_extent": 128
                }]
            }
        }))
        .unwrap();
        let descriptor = BufferDescriptor {
            resource_id,
            size_bytes: 160,
            alignment_bytes: 16,
            usage: BufferUsage::Activations,
            element_type: ElementType::U32,
        };
        let component = &binding.storage().components()[0];
        let demand = DynamicResourceDemand::tokens(4, 128).unwrap();

        assert_eq!(
            validate_value_binding_physical_coverage(
                &work,
                &binding,
                component,
                &descriptor,
                Some(&demand),
                16,
            )
            .unwrap(),
            ValueBindingPhysicalCoverage::RuntimeTokenView
        );
        assert!(validate_value_binding_physical_coverage(
            &NodeWorkContract::Fixed,
            &binding,
            component,
            &descriptor,
            Some(&demand),
            16,
        )
        .is_err());
        assert!(validate_value_binding_physical_coverage(
            &work,
            &binding,
            component,
            &descriptor,
            Some(&DynamicResourceDemand::tokens(8, 128).unwrap()),
            16,
        )
        .is_err());
    }

    #[test]
    fn paged_translation_uses_each_chunks_exact_buffer() {
        struct MockBinding<'a> {
            buffer: &'a u8,
            physical_offset_bytes: u64,
            length_bytes: u64,
        }

        let first_buffer = 7_u8;
        let second_buffer = 11_u8;
        let bindings = [
            MockBinding {
                buffer: &first_buffer,
                physical_offset_bytes: 64,
                length_bytes: 8,
            },
            MockBinding {
                buffer: &second_buffer,
                physical_offset_bytes: 200,
                length_bytes: 12,
            },
        ];
        let mut next_logical_offset = 0;
        let translated = bindings
            .iter()
            .filter_map(|binding| {
                let (logical_end, region) = translate_paged_segment(
                    binding.buffer,
                    DeviceBufferRetention::plan(Arc::new(())),
                    binding.physical_offset_bytes,
                    binding.length_bytes,
                    next_logical_offset,
                    6,
                    16,
                );
                next_logical_offset = logical_end;
                region
            })
            .collect::<Vec<_>>();

        assert_eq!(translated.len(), 2);
        let (first, first_physical, _first_retention) = translated[0].buffer_and_physical_range();
        assert!(std::ptr::eq(first, &first_buffer));
        assert_eq!(translated[0].logical_offset_bytes(), 6);
        assert_eq!(first_physical, 70..72);
        let (second, second_physical, _second_retention) =
            translated[1].buffer_and_physical_range();
        assert!(std::ptr::eq(second, &second_buffer));
        assert_eq!(translated[1].logical_offset_bytes(), 8);
        assert_eq!(second_physical, 200..208);
    }

    #[test]
    fn contiguous_layout_rejects_cross_chunk_bindings() {
        let first_buffer = 7_u8;
        let second_buffer = 11_u8;
        let bindings = [(&first_buffer, 8_u64), (&second_buffer, 12_u64)];

        let error = validate_dynamic_binding_layout(
            OperationBufferStorageKind::DynamicContiguous,
            20,
            bindings.iter().map(|(_, length_bytes)| *length_bytes),
            OperationBufferCoverage::Exact,
        )
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("contiguous dynamic storage requires one physical segment binding"));
    }

    #[test]
    fn operation_prefix_view_retains_wider_backing_coverage() {
        validate_dynamic_binding_layout(
            OperationBufferStorageKind::DynamicPaged,
            64,
            [64_u64, 64].into_iter(),
            OperationBufferCoverage::BackingWindow { offset_bytes: 0 },
        )
        .unwrap();
        validate_dynamic_binding_layout(
            OperationBufferStorageKind::DynamicContiguous,
            64,
            [128_u64].into_iter(),
            OperationBufferCoverage::BackingWindow { offset_bytes: 0 },
        )
        .unwrap();

        assert!(validate_dynamic_binding_layout(
            OperationBufferStorageKind::DynamicPaged,
            64,
            [64_u64, 64].into_iter(),
            OperationBufferCoverage::Exact,
        )
        .is_err());
        assert!(validate_dynamic_binding_layout(
            OperationBufferStorageKind::DynamicPaged,
            128,
            [64_u64].into_iter(),
            OperationBufferCoverage::BackingWindow { offset_bytes: 0 },
        )
        .is_err());
    }

    #[test]
    fn participant_step_projection_keeps_upload_and_readback_coordinate_spaces_distinct() {
        let first_tokens = vec![1_u32; 20];
        let second_tokens = vec![2_u32; 9];
        let work = BatchWorkShape::test_only(vec![
            TokenSpanWork::from_token_ids(&first_tokens, 17..18).unwrap(),
            TokenSpanWork::from_token_ids(&second_tokens, 7..9).unwrap(),
        ])
        .unwrap();
        let tokens = DynamicResourceDemand::tokens(4, 32).unwrap();

        assert_eq!(
            translate_step_participant_upload_range(&tokens, &work, 0, 68..72).unwrap(),
            0..4
        );
        assert_eq!(
            translate_step_participant_upload_range(&tokens, &work, 1, 28..36).unwrap(),
            4..12
        );
        assert_eq!(
            translate_step_participant_readback_range(&tokens, &work, 1, 0..8).unwrap(),
            4..12
        );
        assert!(translate_step_participant_readback_range(&tokens, &work, 1, 28..36).is_err());
    }

    #[test]
    fn fixed_participant_step_projection_uses_disjoint_aligned_strides() {
        let tokens = [1_u32];
        let work = BatchWorkShape::test_only(vec![
            TokenSpanWork::from_token_ids(&tokens, 0..1).unwrap(),
            TokenSpanWork::from_token_ids(&tokens, 0..1).unwrap(),
        ])
        .unwrap();
        let fixed = DynamicResourceDemand::actual_sequences(16, 4).unwrap();

        assert_eq!(
            translate_step_participant_upload_range(&fixed, &work, 0, 0..4).unwrap(),
            0..4
        );
        assert_eq!(
            translate_step_participant_upload_range(&fixed, &work, 1, 0..4).unwrap(),
            16..20
        );
        assert_eq!(
            translate_step_participant_readback_range(&fixed, &work, 1, 0..4).unwrap(),
            16..20
        );
        assert!(translate_step_participant_upload_range(&fixed, &work, 1, 0..17).is_err());
        assert_eq!(
            translate_step_participant_upload_range(
                &DynamicResourceDemand::fixed(16).unwrap(),
                &work,
                1,
                0..4,
            )
            .unwrap(),
            0..4
        );
    }

    #[test]
    fn sequence_execution_shape_uses_the_executed_source_frontier() {
        let committed = DynamicResourceShape::from_validated(1, 8, 3);
        let projected = sequence_execution_shape(committed, 4).unwrap();

        assert_eq!(projected.sequences(), 1);
        assert_eq!(projected.tokens(), 4);
        assert_eq!(projected.pages(), 3);
        assert!(sequence_execution_shape(committed, 0).is_err());
        assert!(sequence_execution_shape(committed, 9).is_err());
        assert!(
            sequence_execution_shape(DynamicResourceShape::from_validated(2, 8, 3), 4).is_err()
        );
    }

    #[test]
    fn contiguous_translation_applies_physical_base_offset() {
        let buffer = 9_u8;
        let regions = OperationBufferRegions {
            storage_kind: OperationBufferStorageKind::DynamicContiguous,
            logical_offset_bytes: 16,
            logical_length_bytes: 32,
            source: OperationRegionSource::Contiguous {
                buffer: &buffer,
                physical_base_offset_bytes: 4096,
                retention: DeviceBufferRetention::plan(Arc::new(())),
            },
        };

        let translated = regions.iter().collect::<Vec<_>>();
        assert_eq!(translated.len(), 1);
        let (actual, physical, _retention) = translated[0].buffer_and_physical_range();
        assert_eq!(*actual, buffer);
        assert_eq!(translated[0].logical_offset_bytes(), 16);
        assert_eq!(physical, 4112..4144);
    }

    #[test]
    fn physical_region_retains_opaque_owner_after_translation_source_drops() {
        struct DropOwner(Arc<std::sync::atomic::AtomicBool>);

        impl Drop for DropOwner {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let buffer = 9_u8;
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let owner = Arc::new(DropOwner(Arc::clone(&dropped)));
        let regions = OperationBufferRegions {
            storage_kind: OperationBufferStorageKind::DynamicContiguous,
            logical_offset_bytes: 0,
            logical_length_bytes: 8,
            source: OperationRegionSource::Contiguous {
                buffer: &buffer,
                physical_base_offset_bytes: 64,
                retention: DeviceBufferRetention::plan(Arc::clone(&owner)),
            },
        };
        drop(owner);

        let translated = regions.iter().collect::<Vec<_>>();
        drop(regions);
        assert!(!dropped.load(Ordering::Acquire));
        let (_, physical, retention) = translated[0].buffer_and_physical_range();
        assert_eq!(physical, 64..72);
        drop(translated);
        assert!(!dropped.load(Ordering::Acquire));

        drop(retention);
        assert!(dropped.load(Ordering::Acquire));
    }
}
