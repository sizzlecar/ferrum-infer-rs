//! The existing checkpoint allocator replayed against one immutable resource
//! capture. These values hold numbers only and never authorize a device copy.
use super::*;
use crate::vnext::{
    split_strided_copy, CheckpointInputDependency, CheckpointPartitionNumerics, DeviceDescriptor,
    ExecutionPlan, NativeCheckpointTransferGeometry, NativeCheckpointTransferGeometryBuilder,
    SequenceCheckpoint, SequenceCheckpointBytePlan, SequenceCheckpointCapability,
    StridedCopyRegion, StridedCopySplitError,
};

#[derive(Debug, Clone)]
pub struct ResourcePlanningCheckpoint {
    fence: Arc<()>,
    token: Arc<()>,
    source: Option<usize>,
    source_authority: SequenceAuthorityId,
    byte_plan: Arc<SequenceCheckpointBytePlan>,
    ranges: Arc<BTreeMap<ResourceId, Arc<Vec<BackingSegment>>>>,
}

impl ResourcePlanningCheckpoint {
    /// A numeric restore may use only a checkpoint retained by this exact
    /// predecessor, never a sibling branch with identical byte counts.
    pub fn belongs_to(&self, state: &ResourcePlanningState) -> bool {
        Arc::ptr_eq(&self.fence, &state.fence)
            && state
                .checkpoint_tokens
                .iter()
                .any(|token| Arc::ptr_eq(token, &self.token))
    }
    pub fn source_participant(&self) -> Option<usize> {
        self.source
    }
    pub fn byte_plan(&self) -> &SequenceCheckpointBytePlan {
        &self.byte_plan
    }
}

#[derive(Debug, Clone)]
pub struct ResourcePlanningCheckpointProjection {
    pub state: ResourcePlanningState,
    pub checkpoint: ResourcePlanningCheckpoint,
    pub geometry: NativeCheckpointTransferGeometry,
}

#[derive(Debug, Clone)]
pub struct ResourcePlanningRestoreProjection {
    pub state: ResourcePlanningState,
    pub geometry: NativeCheckpointTransferGeometry,
}

/// An existing successful native capture bound into a new numeric view.
/// No allocation, logical claim or projected transfer is repeated.
#[derive(Debug, Clone)]
pub struct ResourcePlanningRetainedCheckpoint {
    pub state: ResourcePlanningState,
    pub checkpoint: ResourcePlanningCheckpoint,
}

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    pub fn bind_retained_checkpoint(
        self: &Arc<Self>,
        view: &ResourcePlanningView,
        state: &ResourcePlanningState,
        plan: &ExecutionPlan,
        checkpoint: &SequenceCheckpoint<R>,
        source: usize,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningRetainedCheckpoint> {
        self.bind_owned_checkpoint(view, state, plan, checkpoint, Some(source), budget)
    }

    /// Bind a completed cache owner whose producer need not remain admitted.
    /// The checkpoint's private native owner, rather than a caller supplied
    /// source row, proves the already charged extents and capture authority.
    pub fn bind_ready_checkpoint(
        self: &Arc<Self>,
        view: &ResourcePlanningView,
        state: &ResourcePlanningState,
        plan: &ExecutionPlan,
        checkpoint: &SequenceCheckpoint<R>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningRetainedCheckpoint> {
        self.bind_owned_checkpoint(view, state, plan, checkpoint, None, budget)
    }

    fn bind_owned_checkpoint(
        self: &Arc<Self>,
        view: &ResourcePlanningView,
        state: &ResourcePlanningState,
        plan: &ExecutionPlan,
        checkpoint: &SequenceCheckpoint<R>,
        source: Option<usize>,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningRetainedCheckpoint> {
        let result = (|| {
            validate(view, state, plan, budget)?;
            let captured = checkpoint.captured_evidence();
            if state.waves != 0 || !state.checkpoint_tokens.is_empty() {
                return Err(ResourcePlanningUnknown::InvalidInput);
            }
            let owner = captured.backing();
            if let Some(source) = source {
                let participant = view
                    .participants
                    .get(source)
                    .ok_or(ResourcePlanningUnknown::InvalidInput)?;
                if captured.identity().sequence_authority() != participant.authority {
                    return Err(ResourcePlanningUnknown::StaleIdentity);
                }
            }
            if !Arc::ptr_eq(self, owner.plan_resources())
                || view.coordinator_id != self.dynamic_pools.logical_admission.id()
                || captured.byte_plan().plan_hash() != plan.plan_hash()
                || view
                    .lane_id
                    .is_some_and(|lane| lane != captured.identity().lane_id())
            {
                return Err(ResourcePlanningUnknown::StaleIdentity);
            }
            check_plan_bound(captured.byte_plan(), view.limits, budget)?;
            let retained = view
                .logical
                .checkpoint_retention()
                .ok_or(ResourcePlanningUnknown::Unsupported)?
                .0;
            if owner.extent_bytes() > retained
                || state
                    .checkpoint_retained_bytes
                    .is_none_or(|bytes| bytes < retained)
            {
                return Err(ResourcePlanningUnknown::StaleIdentity);
            }
            // Allocation generations are minted while holding the pool lock.
            // An owned extent older than this capture was already removed from
            // its free layout and charged to the shared logical ledger. The
            // actual checkpoint borrowed here pins it across these checks.
            for evidence in owner.backing_evidence() {
                poll(budget)?;
                let pool = view
                    .pools
                    .iter()
                    .find(|pool| &pool.id == evidence.pool_id())
                    .ok_or(ResourcePlanningUnknown::StaleIdentity)?;
                if pool.instance != evidence.pool_instance_id()
                    || evidence.segment_generation() >= pool.next_extent_generation
                {
                    return Err(ResourcePlanningUnknown::StaleIdentity);
                }
                for segment in evidence.segments() {
                    poll(budget)?;
                    let start = segment.offset_bytes();
                    let end = start
                        .checked_add(segment.length_bytes())
                        .ok_or(ResourcePlanningUnknown::InvalidDemand)?;
                    let chunk = segment.chunk_ordinal();
                    let free = &pool.allocator.by_offset;
                    let previous_overlaps = free.range(..=(chunk, start)).next_back().is_some_and(
                        |(&(ordinal, offset), extent)| {
                            ordinal == chunk
                                && offset
                                    .checked_add(extent.length_bytes)
                                    .is_none_or(|free_end| free_end > start)
                        },
                    );
                    if previous_overlaps
                        || free.range((chunk, start)..(chunk, end)).next().is_some()
                    {
                        return Err(ResourcePlanningUnknown::StaleIdentity);
                    }
                }
            }
            let mut count = 0;
            let ranges = sequence_ranges::capture(
                owner.backing_slices(),
                view.limits.maximum_free_extents,
                &mut count,
                budget,
            )?;
            let mut next = state.clone();
            let token = Arc::new(());
            next.checkpoint_tokens.push(Arc::clone(&token));
            poll(budget)?;
            Ok(ResourcePlanningRetainedCheckpoint {
                state: next,
                checkpoint: ResourcePlanningCheckpoint {
                    fence: Arc::clone(&view.fence),
                    token,
                    source,
                    source_authority: captured.identity().sequence_authority(),
                    byte_plan: Arc::clone(captured.byte_plan()),
                    ranges,
                },
            })
        })();
        availability(result)
    }

    /// Imported frontier proof captured after the native restore publication
    /// was acknowledged. A caller's offset alone cannot establish this fact.
    pub fn checkpoint_restore_completed(
        self: &Arc<Self>,
        view: &ResourcePlanningView,
        checkpoint: &SequenceCheckpoint<R>,
        target: usize,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<bool> {
        availability((|| {
            poll(budget)?;
            let captured = checkpoint.captured_evidence();
            if !Arc::ptr_eq(self, captured.backing().plan_resources())
                || view.coordinator_id != self.dynamic_pools.logical_admission.id()
                || view.plan_hash() != captured.byte_plan().plan_hash()
            {
                return Err(ResourcePlanningUnknown::StaleIdentity);
            }
            let participant = view
                .participants
                .get(target)
                .ok_or(ResourcePlanningUnknown::InvalidInput)?;
            let result = participant
                .completed_checkpoint_boundary
                .as_ref()
                .is_some_and(|boundary| {
                    boundary.restored_from == Some(captured.capture_attempt())
                        && boundary.completed_tokens == captured.byte_plan().boundary()
                });
            poll(budget)?;
            Ok(result)
        })())
    }

    pub(crate) fn checkpoint_descriptor_matches(&self, descriptor: &DeviceDescriptor) -> bool {
        self.runtime.descriptor() == descriptor
    }
    /// Capture only after the route state has independently reached the exact
    /// completed prefix. The returned private token binds later restores to
    /// these same allocated compact ranges and this same numeric capture.
    pub fn project_checkpoint_capture(
        self: &Arc<Self>,
        view: &ResourcePlanningView,
        state: &ResourcePlanningState,
        plan: &ExecutionPlan,
        source: usize,
        boundary: u64,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningCheckpointProjection> {
        let result = (|| {
            validate(view, state, plan, budget)?;
            if view.coordinator_id != self.dynamic_pools.logical_admission.id() {
                return Err(ResourcePlanningUnknown::StaleIdentity);
            }
            let participant = view
                .participants
                .get(source)
                .ok_or(ResourcePlanningUnknown::InvalidInput)?;
            if boundary == 0
                || boundary > participant.maximum_tokens
                || state
                    .covered_tokens(source)
                    .is_none_or(|covered| covered < boundary)
            {
                return Err(ResourcePlanningUnknown::InvalidInput);
            }
            let byte_plan = Arc::new(
                plan.checkpoint_byte_plan(boundary)
                    .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?,
            );
            check_plan_bound(&byte_plan, view.limits, budget)?;
            let requests = byte_plan
                .backing_requests()
                .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?;
            let binding = TrustedPlanRuntimeBinding {
                resources: Arc::clone(self),
            };
            let evaluated = binding
                .evaluate_checkpoint_backing(&requests)
                .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?;
            let (retained, maximum) = view
                .logical
                .checkpoint_retention()
                .ok_or(ResourcePlanningUnknown::Unsupported)?;
            let current = state
                .checkpoint_retained_bytes
                .ok_or(ResourcePlanningUnknown::StaleIdentity)?;
            if current < retained {
                return Err(ResourcePlanningUnknown::StaleIdentity);
            }
            let next_retained = current
                .checked_add(evaluated.extent_bytes)
                .ok_or(ResourcePlanningUnknown::InvalidDemand)?;
            if next_retained > maximum {
                return Err(ResourcePlanningUnknown::LogicalCapacity);
            }
            let mut next = state.clone();
            project::charge_logical(
                view,
                &mut next,
                &evaluated.demand,
                &mut BTreeMap::new(),
                true,
            )?;
            let allocated = project::reserve(&mut next, &evaluated.slices, view.limits, budget)?;
            let ranges = allocated_ranges(&evaluated.slices, &allocated, view.limits, budget)?;
            let source_ranges = state
                .sequence_ranges
                .get(source)
                .ok_or(ResourcePlanningUnknown::InvalidInput)?;
            let mut geometry = NativeCheckpointTransferGeometryBuilder::new();
            append_copies(
                &byte_plan,
                false,
                source_ranges,
                &ranges,
                &mut geometry,
                view.limits,
                budget,
            )?;
            next.checkpoint_retained_bytes = Some(next_retained);
            let token = Arc::new(());
            next.checkpoint_tokens.push(Arc::clone(&token));
            next.waves = next
                .waves
                .checked_add(1)
                .ok_or(ResourcePlanningUnknown::LimitExceeded)?;
            poll(budget)?;
            Ok(ResourcePlanningCheckpointProjection {
                state: next,
                checkpoint: ResourcePlanningCheckpoint {
                    fence: Arc::clone(&view.fence),
                    token,
                    source: Some(source),
                    source_authority: participant.authority,
                    byte_plan,
                    ranges,
                },
                geometry: geometry
                    .finish()
                    .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?,
            })
        })();
        availability(result)
    }

    /// Restore extends the target through the same sequence demand/allocator
    /// protocol as the actual restore. It never creates a model Step and keeps
    /// the checkpoint's capacity charged through the rest of this projection.
    pub fn project_checkpoint_restore(
        self: &Arc<Self>,
        view: &ResourcePlanningView,
        state: &ResourcePlanningState,
        plan: &ExecutionPlan,
        checkpoint: &ResourcePlanningCheckpoint,
        target: usize,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningRestoreProjection> {
        let result = (|| {
            validate(view, state, plan, budget)?;
            if view.coordinator_id != self.dynamic_pools.logical_admission.id() {
                return Err(ResourcePlanningUnknown::StaleIdentity);
            }
            if !Arc::ptr_eq(&view.fence, &checkpoint.fence)
                || !state
                    .checkpoint_tokens
                    .iter()
                    .any(|token| Arc::ptr_eq(token, &checkpoint.token))
                || checkpoint.byte_plan.plan_hash() != plan.plan_hash()
            {
                return Err(ResourcePlanningUnknown::StaleIdentity);
            }
            let participant = view
                .participants
                .get(target)
                .ok_or(ResourcePlanningUnknown::InvalidInput)?;
            if checkpoint.source_authority == participant.authority {
                return Err(ResourcePlanningUnknown::StaleIdentity);
            }
            if participant.retired_frames != 0
                || !participant.restore_zero_initializations_pending
                || !participant.restore_frontier_fresh
                || checkpoint.byte_plan.boundary() > participant.maximum_tokens
            {
                return Err(ResourcePlanningUnknown::InvalidInput);
            }
            check_plan_bound(&checkpoint.byte_plan, view.limits, budget)?;
            let mut next = state.clone();
            let before = next
                .covered
                .get(target)
                .copied()
                .ok_or(ResourcePlanningUnknown::InvalidInput)?;
            let end = checkpoint.byte_plan.boundary();
            let binding = TrustedPlanRuntimeBinding {
                resources: Arc::clone(self),
            };
            if end > before.tokens() {
                let after = DynamicResourceShape::from_validated(1, end, 0);
                let (demand, requests) = binding
                    .sequence_extension_demand(
                        before,
                        after,
                        AdmissionPressureAction::WaitForRelease,
                    )
                    .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?;
                // Fixed zero state uses the captured exact initialization
                // order. Growing zero cells need their own ordered-cell proof;
                // do not silently treat their initialization as free work.
                if requests
                    .iter()
                    .flat_map(|r| &r.projections)
                    .any(|p| p.descriptor.initialization() == StateInitialization::Zero)
                {
                    return Err(ResourcePlanningUnknown::Unsupported);
                }
                project::charge_logical(view, &mut next, &demand, &mut BTreeMap::new(), true)?;
                let allocated = project::reserve(&mut next, &requests, view.limits, budget)?;
                sequence_ranges::extend(
                    &mut next.sequence_ranges[target],
                    &requests,
                    &allocated,
                    view.limits.maximum_free_extents,
                    budget,
                )?;
                sequence_ranges::check_bound(
                    &next.sequence_ranges,
                    view.limits.maximum_free_extents,
                )?;
                next.covered[target] = after;
            }
            let pending = participant
                .pending_zero_initializations
                .as_deref()
                .ok_or(ResourcePlanningUnknown::Unsupported)?;
            let mut geometry = NativeCheckpointTransferGeometryBuilder::new();
            for claim in pending {
                if claim.transfer_bytes.len() != claim.transfer_resources.len() {
                    return Err(ResourcePlanningUnknown::InvalidDemand);
                }
                for (resource, &bytes) in claim
                    .transfer_resources
                    .iter()
                    .zip(claim.transfer_bytes.iter())
                {
                    poll(budget)?;
                    geometry
                        .push_initialization(resource, bytes)
                        .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?;
                }
            }
            append_copies(
                &checkpoint.byte_plan,
                true,
                &next.sequence_ranges[target],
                &checkpoint.ranges,
                &mut geometry,
                view.limits,
                budget,
            )?;
            next.waves = next
                .waves
                .checked_add(1)
                .ok_or(ResourcePlanningUnknown::LimitExceeded)?;
            poll(budget)?;
            Ok(ResourcePlanningRestoreProjection {
                state: next,
                geometry: geometry
                    .finish()
                    .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?,
            })
        })();
        availability(result)
    }
}

fn availability<T>(result: Result<T, ResourcePlanningUnknown>) -> ResourcePlanningAvailability<T> {
    match result {
        Ok(value) => ResourcePlanningAvailability::Known(value),
        Err(reason) => ResourcePlanningAvailability::Unknown(reason),
    }
}

fn validate(
    view: &ResourcePlanningView,
    state: &ResourcePlanningState,
    plan: &ExecutionPlan,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    poll(budget)?;
    if !Arc::ptr_eq(&view.fence, &state.fence) || view.plan_hash() != plan.plan_hash() {
        return Err(ResourcePlanningUnknown::StaleIdentity);
    }
    if state.waves >= view.limits.maximum_projected_waves {
        return Err(ResourcePlanningUnknown::LimitExceeded);
    }
    let SequenceCheckpointCapability::Enabled(layout) = plan.sequence_checkpoint_capability()
    else {
        return Err(ResourcePlanningUnknown::Unsupported);
    };
    if layout.input_dependency() != CheckpointInputDependency::ExactTokenPrefix
        || !layout.inputs().conditioning_inputs().is_empty()
        || layout.providers().iter().any(|provider| {
            provider.contract().partition_numerics()
                == CheckpointPartitionNumerics::SamePartitionOnly
        })
    {
        return Err(ResourcePlanningUnknown::Unsupported);
    }
    if layout.states().len() > view.limits.maximum_descriptors {
        return Err(ResourcePlanningUnknown::LimitExceeded);
    }
    Ok(())
}

fn check_plan_bound(
    plan: &SequenceCheckpointBytePlan,
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    if plan.resources().len() > limits.maximum_descriptors {
        return Err(ResourcePlanningUnknown::LimitExceeded);
    }
    let mut ranges = 0_usize;
    for resource in plan.resources() {
        poll(budget)?;
        ranges = ranges
            .checked_add(
                resource
                    .ranges()
                    .len()
                    .checked_add(resource.strided_ranges().len())
                    .ok_or(ResourcePlanningUnknown::LimitExceeded)?,
            )
            .ok_or(ResourcePlanningUnknown::LimitExceeded)?;
        if ranges > limits.maximum_free_extents {
            return Err(ResourcePlanningUnknown::LimitExceeded);
        }
    }
    Ok(())
}

fn allocated_ranges(
    requests: &[EvaluatedBackingRequest<'_>],
    allocated: &[(usize, Vec<BackingSegment>)],
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Arc<BTreeMap<ResourceId, Arc<Vec<BackingSegment>>>>, ResourcePlanningUnknown> {
    let mut requests: Vec<_> = requests.iter().collect();
    requests.sort_by(|a, b| {
        a.domain
            .pool_id()
            .cmp(b.domain.pool_id())
            .then_with(|| b.capacity_size_bytes.cmp(&a.capacity_size_bytes))
            .then_with(|| a.claim_identity.cmp(&b.claim_identity))
    });
    if requests.len() != allocated.len() {
        return Err(ResourcePlanningUnknown::InvalidDemand);
    }
    let mut result = BTreeMap::new();
    let mut count = 0_usize;
    for (request, (_, segments)) in requests.into_iter().zip(allocated) {
        for projection in &request.projections {
            poll(budget)?;
            let fragments = fragments(
                segments,
                projection.physical_offset_bytes,
                projection.logical_size_bytes,
                limits,
                budget,
            )?;
            count = count
                .checked_add(fragments.len())
                .ok_or(ResourcePlanningUnknown::LimitExceeded)?;
            if count > limits.maximum_free_extents
                || result
                    .insert(
                        projection.descriptor.base_resource_id().clone(),
                        Arc::new(fragments),
                    )
                    .is_some()
            {
                return Err(ResourcePlanningUnknown::InvalidDemand);
            }
        }
    }
    Ok(Arc::new(result))
}

fn fragments(
    segments: &[BackingSegment],
    offset: u64,
    length: u64,
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Vec<BackingSegment>, ResourcePlanningUnknown> {
    if length == 0 {
        return Err(ResourcePlanningUnknown::InvalidDemand);
    }
    let mut skip = offset;
    let mut remaining = length;
    let mut result = Vec::new();
    for segment in segments {
        poll(budget)?;
        if skip >= segment.length_bytes() {
            skip -= segment.length_bytes();
            continue;
        }
        let take = remaining.min(segment.length_bytes() - skip);
        if result.len() >= limits.maximum_free_extents {
            return Err(ResourcePlanningUnknown::LimitExceeded);
        }
        result.push(
            BackingSegment::new(
                segment.chunk().clone(),
                segment
                    .offset_bytes()
                    .checked_add(skip)
                    .ok_or(ResourcePlanningUnknown::InvalidDemand)?,
                take,
            )
            .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?,
        );
        remaining -= take;
        skip = 0;
        if remaining == 0 {
            return Ok(result);
        }
    }
    Err(ResourcePlanningUnknown::InvalidDemand)
}

fn append_copies(
    plan: &SequenceCheckpointBytePlan,
    restore: bool,
    sequence: &BTreeMap<ResourceId, Arc<Vec<BackingSegment>>>,
    checkpoint: &BTreeMap<ResourceId, Arc<Vec<BackingSegment>>>,
    geometry: &mut NativeCheckpointTransferGeometryBuilder,
    limits: ResourcePlanningLimits,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<(), ResourcePlanningUnknown> {
    let mut count = 0_usize;
    for resource in plan.resources() {
        poll(budget)?;
        let source = sequence
            .get(resource.resource_id())
            .ok_or(ResourcePlanningUnknown::Unsupported)?;
        let compact = checkpoint
            .get(resource.resource_id())
            .ok_or(ResourcePlanningUnknown::InvalidDemand)?;
        for range in resource.ranges() {
            let a = fragments(
                source,
                range.source().start,
                range.length_bytes(),
                limits,
                budget,
            )?;
            let b = fragments(
                compact,
                range.checkpoint_offset(),
                range.length_bytes(),
                limits,
                budget,
            )?;
            let (mut i, mut j, mut used_a, mut used_b) = (0, 0, 0, 0);
            while i < a.len() && j < b.len() {
                poll(budget)?;
                let bytes = (a[i].length_bytes() - used_a).min(b[j].length_bytes() - used_b);
                count = count
                    .checked_add(1)
                    .ok_or(ResourcePlanningUnknown::LimitExceeded)?;
                if count > crate::execution_cost::MAX_COST_COMMANDS {
                    return Err(ResourcePlanningUnknown::LimitExceeded);
                }
                geometry
                    .push_copy(resource.resource_id(), bytes)
                    .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?;
                used_a += bytes;
                used_b += bytes;
                if used_a == a[i].length_bytes() {
                    i += 1;
                    used_a = 0;
                }
                if used_b == b[j].length_bytes() {
                    j += 1;
                    used_b = 0;
                }
            }
            if i != a.len() || j != b.len() {
                return Err(ResourcePlanningUnknown::InvalidDemand);
            }
        }
    }
    // Native encoding submits all contiguous fragments before rectangles.
    // Reuse its exact segment splitting, including row boundaries and pitches.
    for resource in plan.resources() {
        poll(budget)?;
        let source = sequence
            .get(resource.resource_id())
            .ok_or(ResourcePlanningUnknown::Unsupported)?;
        let compact = checkpoint
            .get(resource.resource_id())
            .ok_or(ResourcePlanningUnknown::InvalidDemand)?;
        for region in resource.strided_ranges() {
            let a = fragments(
                source,
                region.source_offset_bytes(),
                region
                    .source_extent_bytes()
                    .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?,
                limits,
                budget,
            )?;
            let b = fragments(
                compact,
                region.destination_offset_bytes(),
                region
                    .destination_extent_bytes()
                    .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?,
                limits,
                budget,
            )?;
            let shape = StridedCopyRegion::new(
                0,
                0,
                region.width_bytes(),
                region.height(),
                region.source_pitch_bytes(),
                region.destination_pitch_bytes(),
            )
            .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?;
            let physical = split_strided_copy(
                shape,
                &a.iter()
                    .map(BackingSegment::length_bytes)
                    .collect::<Vec<_>>(),
                &b.iter()
                    .map(BackingSegment::length_bytes)
                    .collect::<Vec<_>>(),
                crate::execution_cost::MAX_COST_COMMANDS.saturating_sub(count),
                || poll(budget),
            )
            .map_err(|error| match error {
                StridedCopySplitError::LimitExceeded => ResourcePlanningUnknown::LimitExceeded,
                StridedCopySplitError::Poll(error) => error,
                StridedCopySplitError::Contract(_) => ResourcePlanningUnknown::InvalidDemand,
            })?;
            count = count
                .checked_add(physical.len())
                .ok_or(ResourcePlanningUnknown::LimitExceeded)?;
            if count > crate::execution_cost::MAX_COST_COMMANDS {
                return Err(ResourcePlanningUnknown::LimitExceeded);
            }
            for fragment in physical {
                poll(budget)?;
                let region = if restore {
                    fragment.region.reversed()
                } else {
                    fragment.region
                };
                geometry
                    .push_strided_copy(resource.resource_id(), region)
                    .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?;
            }
        }
    }
    Ok(())
}
