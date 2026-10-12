//! One fresh resource observation for an entire compiled binding segment.
//!
//! The operation layer supplies already-bound current-wave requirements. This
//! module resolves their captured authorities once, locks the unique pools in
//! canonical order, and consumes every declared window before unlocking. No
//! provider, runtime getter, or backend callback executes under these locks.
use super::{
    invalid_resource, AllocationLifetime, Arc, BufferDescriptor, BufferUsage,
    DeviceBufferRetention, DeviceRuntime, DynamicPoolSet, DynamicStorageProfile, ElementType,
    LogicalBackingSegmentBinding, LogicalBackingSliceAuthority, ResourceId, VNextError,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::time::{Duration, Instant};

use crate::vnext::{SubmissionWaveDispatchStage, SubmissionWaveDispatchTimingSink};

/// Durations only while pool guards exist. Declare this before retained results
/// and guards: ordinary errors flush after both have dropped, while unwinding
/// follows the existing dispatch timers' rule of not calling observers.
struct SegmentBackingTimings<'a, S: SubmissionWaveDispatchTimingSink> {
    sink: &'a S,
    elapsed: [Option<Duration>; 4],
    active: usize,
    started: Option<Instant>,
}

impl<'a, S: SubmissionWaveDispatchTimingSink> SegmentBackingTimings<'a, S> {
    const STAGES: [SubmissionWaveDispatchStage; 4] = [
        SubmissionWaveDispatchStage::SegmentBackingDedupReserveAndPoolResolution,
        SubmissionWaveDispatchStage::SegmentBackingLockAcquisition,
        SubmissionWaveDispatchStage::SegmentBackingLockedValidation,
        SubmissionWaveDispatchStage::SegmentBackingWindowIntersections,
    ];

    #[inline(always)]
    fn new(sink: &'a S) -> Self {
        Self {
            sink,
            elapsed: [None; 4],
            active: 0,
            started: S::ENABLED.then(Instant::now),
        }
    }

    #[inline(always)]
    fn next(&mut self) {
        if let Some(started) = self.started {
            let now = Instant::now();
            self.elapsed[self.active] = Some(now.duration_since(started));
            self.active += 1;
            self.started = Some(now);
        }
    }

    #[inline(always)]
    fn finish(&mut self) {
        if let Some(started) = self.started.take() {
            self.elapsed[self.active] = Some(started.elapsed());
        }
    }
}

impl<S: SubmissionWaveDispatchTimingSink> Drop for SegmentBackingTimings<'_, S> {
    fn drop(&mut self) {
        if S::ENABLED && !std::thread::panicking() {
            self.finish();
            for (stage, elapsed) in Self::STAGES.into_iter().zip(self.elapsed) {
                if let Some(elapsed) = elapsed {
                    self.sink.record(stage, elapsed);
                }
            }
        }
    }
}

#[cfg(test)]
mod timing_tests {
    use super::*;
    use crate::vnext::{DeviceSubmissionStage, DeviceSubmissionTimingSink};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Sink<const ENABLED: bool>(AtomicUsize);

    impl<const ENABLED: bool> DeviceSubmissionTimingSink for Sink<ENABLED> {
        const ENABLED: bool = ENABLED;

        fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
            panic!("pool timing cannot submit device commands");
        }
    }

    impl<const ENABLED: bool> SubmissionWaveDispatchTimingSink for Sink<ENABLED> {
        fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
            assert!(ENABLED, "disabled sink cannot be called");
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn segment_backing_timing_off_has_no_clock_and_panic_does_not_flush() {
        let disabled = Sink::<false>(AtomicUsize::new(0));
        let mut timing = SegmentBackingTimings::new(&disabled);
        assert!(timing.started.is_none());
        timing.next();
        timing.finish();
        assert!(timing.started.is_none());
        assert!(timing.elapsed.iter().all(Option::is_none));
        drop(timing);
        assert_eq!(disabled.0.load(Ordering::Relaxed), 0);

        let enabled = Sink::<true>(AtomicUsize::new(0));
        assert!(std::panic::catch_unwind(|| {
            let mut timing = SegmentBackingTimings::new(&enabled);
            timing.next();
            panic!("diagnostic observers must not flush during unwinding");
        })
        .is_err());
        assert_eq!(enabled.0.load(Ordering::Relaxed), 0);
    }

    fn expectation(bytes: u64) -> SegmentBackingExpectation {
        SegmentBackingExpectation {
            logical_bytes: bytes,
            logical_size_rule: SegmentLogicalSizeRule::Exact,
            usage: BufferUsage::Activations,
            storage_profile: DynamicStorageProfile::new(
                crate::vnext::DynamicStorageAllocator::LinearArena,
                crate::vnext::DynamicStorageView::Contiguous,
            )
            .unwrap(),
            element_type: ElementType::U8,
            alignment_bytes: 1,
        }
    }

    #[test]
    fn compiled_segment_windows_preserve_node_participant_order_and_fresh_growth() {
        // One shared Step allocation with two rows, followed by each current
        // Sequence. Two nodes have different declaration counts; no flattened
        // node/participant index may refer to a different resource or row.
        let mut expected = [expectation(32), expectation(24), expectation(40)];
        let resources = [
            Some(SegmentWindowResource {
                first_request: 0,
                shared: true,
                participant_stride: Some(16),
            }),
            Some(SegmentWindowResource {
                first_request: 1,
                shared: false,
                participant_stride: None,
            }),
        ];
        let rule = |resource_index, offset_bytes, length_bytes| SegmentWindowRule {
            resource_index,
            offset_bytes,
            length_bytes,
            element_type: ElementType::U8,
            alignment_bytes: 1,
        };
        let rules = [rule(0, 4, Some(8)), rule(1, 8, None), rule(1, 0, Some(4))];
        let nodes = [0..2, 2..3];
        let collect = |expected: &[SegmentBackingExpectation]| {
            SegmentBackingWindowInput::compiled(expected, &resources, &rules, &nodes, 2)
                .unwrap()
                .iter()
                .map(|window| {
                    let window = window.unwrap();
                    (
                        window.resource_index,
                        window.offset_bytes,
                        window.length_bytes,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(
            collect(&expected),
            vec![
                (0, 0, 32),
                (1, 0, 24),
                (2, 0, 40),
                (0, 4, 8),
                (1, 8, 16),
                (0, 20, 8),
                (2, 8, 32),
                (1, 0, 4),
                (2, 0, 4),
            ]
        );
        expected[2].logical_bytes = 56;
        let grown = collect(&expected);
        assert_eq!(grown[2], (2, 0, 56));
        assert_eq!(grown[6], (2, 8, 48));
        assert_eq!(grown[8], (2, 0, 4));
        let input =
            SegmentBackingWindowInput::compiled(&expected, &resources, &rules, &nodes, 2).unwrap();
        assert!(input.get(input.len()).is_err());
    }

    #[test]
    fn compiled_segment_windows_reject_invalid_ordinals_rows_and_shape() {
        let expected = [expectation(16)];
        let resources = [Some(SegmentWindowResource {
            first_request: 0,
            shared: true,
            participant_stride: Some(16),
        })];
        let rules = [SegmentWindowRule {
            resource_index: 0,
            offset_bytes: 0,
            length_bytes: None,
            element_type: ElementType::U8,
            alignment_bytes: 1,
        }];
        let nodes = [0..1];
        let input =
            SegmentBackingWindowInput::compiled(&expected, &resources, &rules, &nodes, 2).unwrap();
        assert!(input.get(1).is_ok());
        assert!(input.get(2).is_err()); // second participant is outside this Step
        assert!(
            SegmentBackingWindowInput::compiled(&expected, &resources, &rules, &[1..2], 1,)
                .is_err()
        );
        assert!(
            SegmentBackingWindowInput::compiled(&expected, &resources, &rules, &nodes, 0,).is_err()
        );
        assert!(SegmentBackingWindowInput::compiled(
            &expected,
            &resources,
            &rules,
            &nodes,
            usize::MAX,
        )
        .is_err());
        let absent = [None];
        assert!(
            SegmentBackingWindowInput::compiled(&expected, &absent, &rules, &nodes, 1,)
                .unwrap()
                .get(1)
                .is_err()
        );
        let per_participant = SegmentWindowResource {
            first_request: 0,
            shared: false,
            participant_stride: None,
        };
        assert!(per_participant.row(1, &expected).is_err());
    }
}

pub(crate) struct SegmentPlanResourceRequest<'a> {
    pub slot_index: usize,
    pub allocation: &'a super::ResourceAllocation,
}

/// A fresh current lease observation. Its retention owns Plan resources only;
/// the borrow cannot be retargeted to a later wave or another admitted Plan.
pub(crate) struct SegmentPlanResourceView<'a, B> {
    pub leased: super::LeasedBufferView<'a, B>,
    pub retention: DeviceBufferRetention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentLogicalSizeRule {
    Exact,
    AtLeast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentBackingExpectation {
    pub logical_bytes: u64,
    pub logical_size_rule: SegmentLogicalSizeRule,
    pub usage: BufferUsage,
    pub storage_profile: DynamicStorageProfile,
    pub element_type: ElementType,
    pub alignment_bytes: u64,
}

/// A compiled unique-resource index resolved against this wave, never a node
/// invocation or a previously captured physical authority.
#[derive(Clone, Debug)]
pub struct SegmentBackingRequest {
    pub participant_index: usize,
    pub resource_id: ResourceId,
    pub lifetime: AllocationLifetime,
    pub expected: SegmentBackingExpectation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentBackingWindow {
    pub resource_index: usize,
    pub offset_bytes: u64,
    pub length_bytes: u64,
    pub element_type: ElementType,
    pub alignment_bytes: u64,
}

/// Immutable window instructions. They name resource ordinals, never owners.
pub(crate) struct SegmentWindowRule {
    pub resource_index: usize,
    pub offset_bytes: u64,
    pub length_bytes: Option<u64>,
    pub element_type: ElementType,
    pub alignment_bytes: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct SegmentWindowResource {
    pub first_request: usize,
    pub shared: bool,
    pub participant_stride: Option<u64>,
}

impl SegmentWindowResource {
    pub(crate) fn row(
        self,
        participant: usize,
        expected: &[SegmentBackingExpectation],
    ) -> Result<(usize, u64, u64), VNextError> {
        let request = self
            .first_request
            .checked_add(if self.shared { 0 } else { participant })
            .ok_or_else(|| invalid_resource("segment resource ordinal overflows"))?;
        let requirement = expected
            .get(request)
            .ok_or_else(|| invalid_resource("segment resource ordinal is absent"))?;
        let (offset, bytes) = match self.participant_stride {
            Some(stride) => (
                stride
                    .checked_mul(participant as u64)
                    .ok_or_else(|| invalid_resource("segment participant offset overflows"))?,
                stride,
            ),
            None => (0, requirement.logical_bytes),
        };
        if bytes == 0
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > requirement.logical_bytes)
        {
            return Err(invalid_resource(
                "segment participant exceeds its current resource",
            ));
        }
        Ok((request, offset, bytes))
    }
}

/// Closed, pure data access under the permit. No backend callback is invoked.
#[derive(Clone, Copy)]
pub(crate) enum SegmentBackingWindowInput<'a> {
    Slice(&'a [SegmentBackingWindow]),
    Compiled {
        expected: &'a [SegmentBackingExpectation],
        resources: &'a [Option<SegmentWindowResource>],
        rules: &'a [SegmentWindowRule],
        nodes: &'a [Range<usize>],
        participants: usize,
        length: usize,
    },
}

impl<'a> SegmentBackingWindowInput<'a> {
    pub(crate) fn compiled(
        expected: &'a [SegmentBackingExpectation],
        resources: &'a [Option<SegmentWindowResource>],
        rules: &'a [SegmentWindowRule],
        nodes: &'a [Range<usize>],
        participants: usize,
    ) -> Result<Self, VNextError> {
        let length = rules
            .len()
            .checked_mul(participants)
            .and_then(|n| n.checked_add(expected.len()))
            .ok_or_else(|| invalid_resource("compiled segment window count overflows"))?;
        let mut next = 0;
        for node in nodes {
            if node.start != next || node.start >= node.end || node.end > rules.len() {
                return Err(invalid_resource(
                    "compiled segment window order is incomplete",
                ));
            }
            next = node.end;
        }
        if next != rules.len() || participants == 0 {
            return Err(invalid_resource(
                "compiled segment windows have no complete participants",
            ));
        }
        Ok(Self::Compiled {
            expected,
            resources,
            rules,
            nodes,
            participants,
            length,
        })
    }

    pub(crate) fn len(self) -> usize {
        match self {
            Self::Slice(windows) => windows.len(),
            Self::Compiled { length, .. } => length,
        }
    }

    pub(crate) fn get(self, index: usize) -> Result<SegmentBackingWindow, VNextError> {
        match self {
            Self::Slice(windows) => windows
                .get(index)
                .copied()
                .ok_or_else(|| invalid_resource("segment window ordinal is absent")),
            Self::Compiled {
                expected,
                resources,
                rules,
                nodes,
                participants,
                length,
            } => {
                if index >= length {
                    return Err(invalid_resource(
                        "compiled segment window ordinal is absent",
                    ));
                }
                if let Some(requirement) = expected.get(index) {
                    return Ok(SegmentBackingWindow {
                        resource_index: index,
                        offset_bytes: 0,
                        length_bytes: requirement.logical_bytes,
                        element_type: requirement.element_type,
                        alignment_bytes: requirement.alignment_bytes,
                    });
                }
                let index = index - expected.len();
                let node_index = nodes
                    .partition_point(|node| node.start * participants <= index)
                    .checked_sub(1)
                    .ok_or_else(|| invalid_resource("compiled segment node window is absent"))?;
                let node = &nodes[node_index];
                let within = index - node.start * participants;
                let participant = within / node.len();
                let rule = &rules[node.start + within % node.len()];
                let resource = resources
                    .get(rule.resource_index)
                    .and_then(|r| *r)
                    .ok_or_else(|| {
                        invalid_resource("compiled segment window has no dynamic resource")
                    })?;
                let (resource_index, offset, available) = resource.row(participant, expected)?;
                let length_bytes = match rule.length_bytes {
                    Some(length) => length,
                    None => available
                        .checked_sub(rule.offset_bytes)
                        .ok_or_else(|| invalid_resource("compiled segment resource shrank"))?,
                };
                let offset_bytes = offset
                    .checked_add(rule.offset_bytes)
                    .ok_or_else(|| invalid_resource("compiled segment window offset overflows"))?;
                Ok(SegmentBackingWindow {
                    resource_index,
                    offset_bytes,
                    length_bytes,
                    element_type: rule.element_type,
                    alignment_bytes: rule.alignment_bytes,
                })
            }
        }
    }

    fn iter(self) -> impl ExactSizeIterator<Item = Result<SegmentBackingWindow, VNextError>> + 'a {
        (0..self.len()).map(move |index| self.get(index))
    }
}

/// Only the permit can construct these summaries. The effective view is the
/// requested current-wave size, not all captured capacity or committed slack.
pub struct SegmentBackingResource {
    logical_size_bytes: u64,
    capacity_size_bytes: u64,
    view_size_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
    storage_profile: DynamicStorageProfile,
    bindings: Range<usize>,
    binding_block: usize,
}

impl SegmentBackingResource {
    pub const fn logical_size_bytes(&self) -> u64 {
        self.logical_size_bytes
    }
    pub const fn capacity_size_bytes(&self) -> u64 {
        self.capacity_size_bytes
    }
    pub const fn view_size_bytes(&self) -> u64 {
        self.view_size_bytes
    }
    pub const fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }
    pub const fn usage(&self) -> BufferUsage {
        self.usage
    }
    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }
    pub const fn storage_profile(&self) -> DynamicStorageProfile {
        self.storage_profile
    }
}

struct WindowRegion {
    binding_index: usize,
    binding_block: usize,
    binding_local: usize,
    logical_offset_bytes: u64,
    physical_offset_bytes: u64,
    length_bytes: u64,
}

pub(crate) enum SegmentBackingSource<'a, B> {
    Authorities(&'a [LogicalBackingSliceAuthority]),
    Sequence(super::SequencePhysicalRowSource<'a, B>),
}

impl<'a, B> SegmentBackingSource<'a, B> {
    fn authorities(&self) -> &'a [LogicalBackingSliceAuthority] {
        match self {
            Self::Authorities(authorities) => authorities,
            Self::Sequence(source) => source.authorities(),
        }
    }
}

/// Ordered blocks retain exact bindings; cached rows are never flattened.
pub(crate) struct SegmentBindings<B> {
    blocks: Vec<BindingBlock<B>>,
    ends: Vec<usize>,
    count: usize,
}

enum BindingBlock<B> {
    Fresh(Vec<LogicalBackingSegmentBinding<B>>),
    Sequence(Arc<super::sequence_physical_rows::ValidatedSequencePhysicalRow<B>>),
}

impl<B> SegmentBindings<B> {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            ends: Vec::new(),
            count: 0,
        }
    }
    pub(crate) fn len(&self) -> usize {
        self.count
    }
    fn push(&mut self, block: BindingBlock<B>) -> Result<Range<usize>, VNextError> {
        let start = self.count;
        let length = match &block {
            BindingBlock::Fresh(bindings) => bindings.len(),
            BindingBlock::Sequence(row) => row.len(),
        };
        self.count = self
            .count
            .checked_add(length)
            .ok_or_else(|| invalid_resource("segment binding count overflows usize"))?;
        self.blocks.push(block);
        self.ends.push(self.count);
        Ok(start..self.count)
    }
    pub(crate) fn get(&self, index: usize) -> Option<&LogicalBackingSegmentBinding<B>> {
        let block = self.ends.partition_point(|end| *end <= index);
        let base = block
            .checked_sub(1)
            .map_or(0, |previous| self.ends[previous]);
        self.in_block(block, index - base)
    }
    fn in_block(&self, block: usize, local: usize) -> Option<&LogicalBackingSegmentBinding<B>> {
        match self.blocks.get(block)? {
            BindingBlock::Fresh(bindings) => bindings.get(local),
            BindingBlock::Sequence(row) => row.get(local),
        }
    }
    pub(crate) fn iter(&self) -> SegmentBindingsIter<'_, B> {
        SegmentBindingsIter {
            bindings: self,
            block: 0,
            limit: self.blocks.len(),
            part: 0,
            current: [].iter(),
            remaining: self.count,
        }
    }
    fn block_iter(&self, block: usize) -> SegmentBindingsIter<'_, B> {
        let end = self.ends[block];
        let start = block
            .checked_sub(1)
            .map_or(0, |previous| self.ends[previous]);
        SegmentBindingsIter {
            bindings: self,
            block,
            limit: block + 1,
            part: 0,
            current: [].iter(),
            remaining: end - start,
        }
    }
}

impl<B> std::ops::Index<usize> for SegmentBindings<B> {
    type Output = LogicalBackingSegmentBinding<B>;
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("validated segment binding index")
    }
}

pub(crate) struct SegmentBindingsIter<'a, B> {
    bindings: &'a SegmentBindings<B>,
    block: usize,
    limit: usize,
    part: usize,
    current: std::slice::Iter<'a, LogicalBackingSegmentBinding<B>>,
    remaining: usize,
}
impl<'a, B> Iterator for SegmentBindingsIter<'a, B> {
    type Item = &'a LogicalBackingSegmentBinding<B>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(binding) = self.current.next() {
                self.remaining -= 1;
                return Some(binding);
            }
            if self.block >= self.limit {
                return None;
            }
            let piece = match self.bindings.blocks.get(self.block)? {
                BindingBlock::Fresh(bindings) => (self.part == 0).then_some(bindings.as_slice()),
                BindingBlock::Sequence(row) => row.part(self.part),
            };
            if let Some(piece) = piece {
                self.part += 1;
                self.current = piece.iter();
            } else {
                self.block += 1;
                self.part = 0;
            }
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl<B> ExactSizeIterator for SegmentBindingsIter<'_, B> {}

struct SegmentMetadataIter<'a, B> {
    bindings: &'a SegmentBindings<B>,
    block: usize,
    entry: usize,
}
impl<'a, B> Iterator for SegmentMetadataIter<'a, B> {
    type Item = (&'a B, &'a BufferDescriptor);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let value = match self.bindings.blocks.get(self.block)? {
                BindingBlock::Fresh(bindings) => bindings
                    .get(self.entry)
                    .map(|b| (b.buffer(), b.descriptor())),
                BindingBlock::Sequence(row) => row.metadata(self.entry),
            };
            if value.is_some() {
                self.entry += 1;
                return value;
            }
            self.block += 1;
            self.entry = 0;
        }
    }
}

/// Owns real allocation/lease retention, but no pool lock or whole wave owner.
/// A later mutation is ordered after this observation, not prohibited until
/// submit. The existing submission/fence ownership must retain these regions.
pub struct SegmentBackingBatch<B> {
    resources: Vec<SegmentBackingResource>,
    bindings: SegmentBindings<B>,
    windows: Vec<Range<usize>>,
    regions: Vec<WindowRegion>,
}

impl<B> SegmentBackingBatch<B> {
    pub fn resources(&self) -> &[SegmentBackingResource] {
        &self.resources
    }
    pub fn resource(&self, index: usize) -> Option<&SegmentBackingResource> {
        self.resources.get(index)
    }
    pub fn window(&self, index: usize) -> Option<SegmentBackingWindowView<'_, B>> {
        self.windows
            .get(index)
            .map(|range| SegmentBackingWindowView {
                batch: self,
                regions: &self.regions[range.clone()],
            })
    }
    /// The operation layer uses each actual buffer once for its explicit
    /// immutable-metadata capability. This never calls a runtime under a lock.
    pub(crate) fn bindings(&self) -> &SegmentBindings<B> {
        &self.bindings
    }

    pub(crate) fn metadata_buffers(&self) -> impl Iterator<Item = (&B, &BufferDescriptor)> {
        SegmentMetadataIter {
            bindings: &self.bindings,
            block: 0,
            entry: 0,
        }
    }
}

pub struct SegmentBackingWindowView<'a, B> {
    batch: &'a SegmentBackingBatch<B>,
    regions: &'a [WindowRegion],
}

impl<'a, B> SegmentBackingWindowView<'a, B> {
    pub(crate) fn region_count(&self) -> usize {
        self.regions.len()
    }

    pub(crate) fn physical_region(&self, index: usize) -> Option<SegmentPhysicalRegion<'a, B>> {
        self.regions.get(index).map(|region| SegmentPhysicalRegion {
            binding: self
                .batch
                .bindings
                .in_block(region.binding_block, region.binding_local)
                .expect("validated window binding"),
            region,
        })
    }

    pub fn physical_regions(
        &self,
    ) -> impl ExactSizeIterator<Item = SegmentPhysicalRegion<'a, B>> + 'a {
        let batch = self.batch;
        self.regions
            .iter()
            .map(move |region| SegmentPhysicalRegion {
                binding: batch
                    .bindings
                    .in_block(region.binding_block, region.binding_local)
                    .expect("validated window binding"),
                region,
            })
    }
}

pub struct SegmentPhysicalRegion<'a, B> {
    binding: &'a LogicalBackingSegmentBinding<B>,
    region: &'a WindowRegion,
}

impl<'a, B> SegmentPhysicalRegion<'a, B> {
    pub(crate) const fn binding_index(&self) -> usize {
        self.region.binding_index
    }

    pub(crate) fn borrowed_buffer_and_physical_range(
        &self,
    ) -> (&'a B, Range<u64>, &'a DeviceBufferRetention) {
        (
            self.binding.buffer(),
            self.region.physical_offset_bytes
                ..self.region.physical_offset_bytes + self.region.length_bytes,
            self.binding.retention_ref(),
        )
    }

    pub const fn logical_offset_bytes(&self) -> u64 {
        self.region.logical_offset_bytes
    }
    pub const fn length_bytes(&self) -> u64 {
        self.region.length_bytes
    }
    pub fn descriptor(&self) -> &BufferDescriptor {
        self.binding.descriptor()
    }
    pub fn buffer_and_physical_range(&self) -> (&'a B, Range<u64>, DeviceBufferRetention) {
        (
            self.binding.buffer(),
            self.region.physical_offset_bytes
                ..self.region.physical_offset_bytes + self.region.length_bytes,
            self.binding.retention(),
        )
    }
}

impl<R: DeviceRuntime> DynamicPoolSet<R> {
    pub(super) fn segment_backing_batch(
        &self,
        groups: &[&[LogicalBackingSliceAuthority]],
        expected: &[SegmentBackingExpectation],
        windows: &[SegmentBackingWindow],
    ) -> Result<SegmentBackingBatch<R::Buffer>, VNextError> {
        self.segment_backing_batch_with_timing(
            groups,
            expected,
            windows,
            &crate::vnext::operation::DisabledSubmissionWaveDispatchTimingSink,
        )
    }

    pub(super) fn segment_backing_batch_with_timing<S: SubmissionWaveDispatchTimingSink>(
        &self,
        groups: &[&[LogicalBackingSliceAuthority]],
        expected: &[SegmentBackingExpectation],
        windows: &[SegmentBackingWindow],
        timing_sink: &S,
    ) -> Result<SegmentBackingBatch<R::Buffer>, VNextError> {
        self.segment_backing_batch_with_window_input(
            groups,
            expected,
            SegmentBackingWindowInput::Slice(windows),
            timing_sink,
        )
    }

    pub(super) fn segment_backing_batch_with_window_input<S: SubmissionWaveDispatchTimingSink>(
        &self,
        groups: &[&[LogicalBackingSliceAuthority]],
        expected: &[SegmentBackingExpectation],
        windows: SegmentBackingWindowInput<'_>,
        timing_sink: &S,
    ) -> Result<SegmentBackingBatch<R::Buffer>, VNextError> {
        let sources = groups
            .iter()
            .map(|group| SegmentBackingSource::Authorities(*group))
            .collect::<Vec<_>>();
        self.segment_backing_batch_with_sources(&sources, expected, windows, timing_sink)
    }

    pub(super) fn segment_backing_batch_with_sources<S: SubmissionWaveDispatchTimingSink>(
        &self,
        sources: &[SegmentBackingSource<'_, R::Buffer>],
        expected: &[SegmentBackingExpectation],
        windows: SegmentBackingWindowInput<'_>,
        timing_sink: &S,
    ) -> Result<SegmentBackingBatch<R::Buffer>, VNextError> {
        let groups = sources
            .iter()
            .map(SegmentBackingSource::authorities)
            .collect::<Vec<_>>();
        let mut timings = SegmentBackingTimings::new(timing_sink);
        if groups.len() != expected.len() || groups.is_empty() {
            return Err(invalid_resource(
                "segment resources require matching nonempty expectations",
            ));
        }
        // Allocate before locking. Partial retained results are also declared
        // before guards, so panic/error unwinding unlocks before owner Drop.
        let mut result = SegmentBackingBatch {
            resources: Vec::new(),
            bindings: SegmentBindings::new(),
            windows: Vec::new(),
            regions: Vec::new(),
        };
        let mut ids = BTreeSet::new();
        let mut unique = BTreeMap::new();
        let mut group_indices = Vec::new();
        let mut unique_groups = Vec::new();
        let mut unique_sources = Vec::new();
        let mut counts = Vec::new();
        let mut prefixes = Vec::new();
        for (source, group) in sources.iter().zip(&groups) {
            let first = group
                .first()
                .ok_or_else(|| invalid_resource("segment resource has no authority"))?;
            ids.insert(first.evidence.pool_id.clone());
            // Pointer identity is only a dedup key. Every unique slice is still
            // fully validated below against its exact pool and segment lease.
            let key = (group.as_ptr(), group.len());
            let index = match unique.get(&key) {
                Some(index) => *index,
                None => {
                    let index = unique_groups.len();
                    unique_groups.push(*group);
                    unique_sources.push(source);
                    unique.insert(key, index);
                    let (count, prefix) = match source {
                        SegmentBackingSource::Authorities(_) => (
                            group.iter().try_fold(0usize, |count, authority| {
                                count.checked_add(authority.evidence.segments.len())
                            }),
                            None,
                        ),
                        SegmentBackingSource::Sequence(row) => {
                            (row.segment_count(), row.cached_prefix())
                        }
                    };
                    counts.push(count.ok_or_else(|| {
                        invalid_resource("segment binding count overflows usize")
                    })?);
                    prefixes.push(prefix);
                    index
                }
            };
            group_indices.push(index);
        }
        let region_capacity = windows.iter().try_fold(0usize, |total, window| {
            let window = window?;
            let group = group_indices
                .get(window.resource_index)
                .ok_or_else(|| invalid_resource("segment window references an unknown resource"))?;
            total
                .checked_add(counts[*group])
                .ok_or_else(|| invalid_resource("segment window capacity overflows usize"))
        })?;
        result
            .resources
            .try_reserve_exact(groups.len())
            .map_err(|_| invalid_resource("segment resource allocation failed"))?;
        result
            .bindings
            .blocks
            .try_reserve_exact(unique_groups.len())
            .map_err(|_| invalid_resource("segment binding allocation failed"))?;
        result
            .bindings
            .ends
            .try_reserve_exact(unique_groups.len())
            .map_err(|_| invalid_resource("segment binding allocation failed"))?;
        result
            .windows
            .try_reserve_exact(windows.len())
            .map_err(|_| invalid_resource("segment window allocation failed"))?;
        result
            .regions
            .try_reserve_exact(region_capacity)
            .map_err(|_| invalid_resource("segment region allocation failed"))?;
        let mut summaries = Vec::new();
        summaries
            .try_reserve_exact(unique_groups.len())
            .map_err(|_| invalid_resource("segment summary allocation failed"))?;
        let mut prepared = Vec::with_capacity(unique_groups.len());
        for (index, group) in unique_groups.iter().enumerate() {
            let skip = prefixes[index]
                .as_ref()
                .map_or(0, |row| row.authority_count);
            let tail_count = counts[index]
                .checked_sub(prefixes[index].as_ref().map_or(0, |row| row.len()))
                .ok_or_else(|| {
                    invalid_resource("sequence physical prefix exceeds its captured backing")
                })?;
            let mut bindings = Vec::new();
            bindings
                .try_reserve_exact(tail_count)
                .map_err(|_| invalid_resource("segment binding allocation failed"))?;
            let mut allocations = Vec::new();
            if matches!(unique_sources[index], SegmentBackingSource::Sequence(_)) {
                allocations
                    .try_reserve_exact(group.len().checked_sub(skip).ok_or_else(|| {
                        invalid_resource(
                            "sequence physical prefix exceeds its captured authorities",
                        )
                    })?)
                    .map_err(|_| {
                        invalid_resource("sequence physical evidence allocation failed")
                    })?;
            }
            prepared.push((bindings, allocations));
        }
        let mut publish = Vec::with_capacity(unique_groups.len());
        let pools = ids
            .into_iter()
            .map(|id| {
                self.pools
                    .get(&id)
                    .map(|pool| (id, pool))
                    .ok_or_else(|| invalid_resource("segment authority has no dynamic pool"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        timings.next();
        let guards = pools
            .iter()
            .map(|(_, pool)| {
                pool.state
                    .lock()
                    .map_err(|_| invalid_resource("dynamic backing pool is poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        timings.next();
        let checked = (|| {
            for (group_index, group) in unique_groups.iter().enumerate() {
                let first = &group[0];
                let pool_index = pools
                    .binary_search_by(|(id, _)| id.cmp(&first.evidence.pool_id))
                    .map_err(|_| invalid_resource("segment pool permit is incomplete"))?;
                let pool = pools[pool_index].1;
                let state = &guards[pool_index];
                if state.poisoned {
                    return Err(invalid_resource("dynamic backing pool is fail-closed"));
                }
                let prefix = prefixes[group_index].as_ref();
                if let Some(prefix) = prefix {
                    // Snapshot producers seal the old exact allocations and
                    // leases. Observe the actual pool header/first authority
                    // again; every distinct current chunk must still be the
                    // exact Arc validated by the published prefix.
                    Self::validate_authority_identity_and_bounds(pool, first)?;
                    if !prefix.current(state) {
                        // Rebuild from original authorities after unlock,
                        // rather than carrying a stale chunk into this wave.
                        return Ok(false);
                    }
                    if prefix.authority_count < group.len() && !prefix.last_complete {
                        return Err(invalid_resource(
                            "multi-extent logical backing cannot contain interior capacity slack",
                        ));
                    }
                }
                let skip = prefix.map_or(0, |row| row.authority_count);
                let mut logical = prefix.map_or(0, |row| row.logical);
                let mut capacity = prefix.map_or(0, |row| row.capacity);
                let (bindings, allocations) = &mut prepared[group_index];
                for (index, authority) in group.iter().enumerate().skip(skip) {
                    if authority.evidence.pool_id != first.evidence.pool_id
                        || authority.evidence.resource_id != first.evidence.resource_id
                        || authority.evidence.storage_profile != first.evidence.storage_profile
                        || authority.evidence.alignment_bytes != first.evidence.alignment_bytes
                        || authority.evidence.usage != first.evidence.usage
                        || authority.evidence.element_type != first.evidence.element_type
                        || authority.evidence.initialization != first.evidence.initialization
                    {
                        return Err(invalid_resource(
                            "logical backing authorities have incompatible resource metadata",
                        ));
                    }
                    Self::validate_authority(pool, authority)?;
                    if index + 1 < group.len()
                        && authority.evidence.logical_size_bytes
                            != authority.evidence.capacity_size_bytes
                    {
                        return Err(invalid_resource(
                            "multi-extent logical backing cannot contain interior capacity slack",
                        ));
                    }
                    logical = logical
                        .checked_add(authority.evidence.logical_size_bytes)
                        .ok_or_else(|| {
                            invalid_resource("logical backing view size overflows u64")
                        })?;
                    capacity = capacity
                        .checked_add(authority.evidence.capacity_size_bytes)
                        .ok_or_else(|| {
                            invalid_resource("logical backing capacity overflows u64")
                        })?;
                    if matches!(
                        unique_sources[group_index],
                        SegmentBackingSource::Sequence(_)
                    ) {
                        allocations.push(Arc::clone(&authority.evidence.allocation));
                    }
                    for segment in &authority.evidence.segments {
                        let chunk =
                            state.chunks.get(&segment.chunk_ordinal()).ok_or_else(|| {
                                invalid_resource("logical backing references a missing chunk")
                            })?;
                        if segment.pool_id() != &authority.evidence.pool_id
                            || chunk.backing.identity != *segment.chunk()
                            || segment
                                .offset_bytes()
                                .checked_add(segment.length_bytes())
                                .is_none_or(|end| end > chunk.backing.descriptor.size_bytes)
                        {
                            return Err(invalid_resource(
                                "logical backing references a stale or out-of-bounds chunk region",
                            ));
                        }
                        let retention = match authority.reusable_lane {
                            Some(lane) => DeviceBufferRetention::lane_pair(
                                lane,
                                Arc::clone(&authority.segment_lease),
                                Arc::clone(&chunk.backing),
                            ),
                            None => DeviceBufferRetention::pair(
                                Arc::clone(&authority.segment_lease),
                                Arc::clone(&chunk.backing),
                            ),
                        };
                        bindings.push(LogicalBackingSegmentBinding {
                            segment: segment.clone(),
                            chunk: Arc::clone(&chunk.backing),
                            retention,
                        });
                    }
                }
                let range =
                    if let SegmentBackingSource::Sequence(source) = unique_sources[group_index] {
                        let row = if prefix.is_some_and(|row| row.authority_count == group.len()) {
                            Arc::clone(prefix.expect("complete prefix was observed"))
                        } else {
                            Arc::new(
                                super::sequence_physical_rows::ValidatedSequencePhysicalRow::new(
                                    group.len(),
                                    prefix.map(|row| row.as_ref()),
                                    std::mem::take(bindings),
                                    std::mem::take(allocations),
                                    logical,
                                    capacity,
                                    group.last().is_some_and(|last| {
                                        last.evidence.logical_size_bytes
                                            == last.evidence.capacity_size_bytes
                                    }),
                                ),
                            )
                        };
                        publish.push((source, Arc::clone(&row)));
                        result.bindings.push(BindingBlock::Sequence(row))?
                    } else {
                        result
                            .bindings
                            .push(BindingBlock::Fresh(std::mem::take(bindings)))?
                    };
                summaries.push((logical, capacity, range));
            }
            for (index, requirement) in expected.iter().enumerate() {
                let first = &groups[index][0].evidence;
                let (logical, capacity, bindings) = &summaries[group_indices[index]];
                let size_matches = match requirement.logical_size_rule {
                    SegmentLogicalSizeRule::Exact => *logical == requirement.logical_bytes,
                    SegmentLogicalSizeRule::AtLeast => *logical >= requirement.logical_bytes,
                };
                if requirement.logical_bytes == 0
                    || !size_matches
                    || capacity < logical
                    || first.usage != requirement.usage
                    || first.storage_profile != requirement.storage_profile
                    || first.element_type != requirement.element_type
                    || first.alignment_bytes != requirement.alignment_bytes
                {
                    return Err(invalid_resource(
                        "segment backing differs from its current-wave resource requirement",
                    ));
                }
                result.resources.push(SegmentBackingResource {
                    logical_size_bytes: *logical,
                    capacity_size_bytes: *capacity,
                    view_size_bytes: requirement.logical_bytes,
                    alignment_bytes: first.alignment_bytes,
                    usage: first.usage,
                    element_type: first.element_type,
                    storage_profile: first.storage_profile,
                    bindings: bindings.clone(),
                    binding_block: group_indices[index],
                });
            }
            timings.next();
            for window in windows.iter() {
                let window = window?;
                let resource = result.resources.get(window.resource_index).ok_or_else(|| {
                    invalid_resource("segment window references an unknown resource")
                })?;
                let end = window
                    .offset_bytes
                    .checked_add(window.length_bytes)
                    .ok_or_else(|| invalid_resource("segment window range overflows u64"))?;
                if window.length_bytes == 0
                    || end > resource.view_size_bytes
                    || window.element_type != resource.element_type
                    || window.alignment_bytes == 0
                    || !window.alignment_bytes.is_power_of_two()
                    || resource.alignment_bytes < window.alignment_bytes
                    || !window.offset_bytes.is_multiple_of(window.alignment_bytes)
                {
                    return Err(invalid_resource(
                        "segment window is outside its typed current-wave view",
                    ));
                }
                let start = result.regions.len();
                let mut logical_origin = 0u64;
                let mut covered = 0u64;
                for (binding_local, binding) in result
                    .bindings
                    .block_iter(resource.binding_block)
                    .enumerate()
                {
                    let binding_index = resource.bindings.start + binding_local;
                    let segment_end = logical_origin
                        .checked_add(binding.segment.length_bytes())
                        .ok_or_else(|| invalid_resource("segment logical range overflows u64"))?;
                    let begin = logical_origin.max(window.offset_bytes);
                    let finish = segment_end.min(end);
                    if begin < finish {
                        let physical = binding
                            .segment
                            .offset_bytes()
                            .checked_add(begin - logical_origin)
                            .ok_or_else(|| {
                                invalid_resource("segment physical offset overflows u64")
                            })?;
                        let length = finish - begin;
                        if !physical.is_multiple_of(window.alignment_bytes)
                            || physical
                                .checked_add(length)
                                .is_none_or(|limit| limit > binding.descriptor().size_bytes)
                        {
                            return Err(invalid_resource(
                                "segment physical window exceeds its actual allocation",
                            ));
                        }
                        covered = covered
                            .checked_add(length)
                            .ok_or_else(|| invalid_resource("segment covered bytes overflow"))?;
                        result.regions.push(WindowRegion {
                            binding_index,
                            binding_block: resource.binding_block,
                            binding_local,
                            logical_offset_bytes: begin - window.offset_bytes,
                            physical_offset_bytes: physical,
                            length_bytes: length,
                        });
                    }
                    logical_origin = segment_end;
                    if logical_origin >= end {
                        break;
                    }
                }
                if covered != window.length_bytes {
                    return Err(invalid_resource(
                        "segment physical backing does not cover its window",
                    ));
                }
                result.windows.push(start..result.regions.len());
            }
            Ok(true)
        })();
        // Read the last locked interval without invoking the observer. It is
        // flushed only after guard release, on both success and ordinary Err.
        timings.finish();
        drop(guards);
        if !checked? {
            // Fallback allocations and all real owner Drop occur unlocked.
            // The legacy path freshly selects current Arcs and repeats every
            // original authority/window check.
            return self.segment_backing_batch_with_window_input(
                &groups,
                expected,
                windows,
                timing_sink,
            );
        }
        for (source, row) in publish {
            source.publish(row);
        }
        Ok(result)
    }
}
