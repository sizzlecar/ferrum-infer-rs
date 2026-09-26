use super::*;
use crate::vnext::resource::sequence::{SequenceSessionPhase, SequenceSessionSlotState};

impl<R: DeviceRuntime> PlanRuntimeResources<R> {
    /// Samples existing admitted sessions without pinning their backing. Every
    /// acquired lock is nonblocking; all guards outlive the numeric copy and
    /// drop before returning. No maintenance, admission or allocation occurs.
    pub fn resource_planning_view(
        self: &Arc<Self>,
        sessions: &[&SequenceSession<R>],
        limits: ResourcePlanningLimits,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningView> {
        match self.capture_resource_planning_view(sessions, None, limits, budget) {
            Ok(view) => ResourcePlanningAvailability::Known(view),
            Err(reason) => ResourcePlanningAvailability::Unknown(reason),
        }
    }

    /// Includes this exact lane's retained Step/Invocation slots. The lane,
    /// arena and allocator reads are nonblocking and overlap in one bracket.
    pub fn resource_planning_view_on_lane(
        self: &Arc<Self>,
        sessions: &[&SequenceSession<R>],
        lane: &ExecutionLane<R>,
        limits: ResourcePlanningLimits,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> ResourcePlanningAvailability<ResourcePlanningView> {
        if !Arc::ptr_eq(&self.runtime, lane.runtime_arc()) {
            return ResourcePlanningAvailability::Unknown(ResourcePlanningUnknown::StaleIdentity);
        }
        // Scalar resource callers do not request or pay for a graph inventory.
        match lane.try_with_resource_planning_lane(|epoch, _| {
            self.capture_resource_planning_view(sessions, Some((lane.id(), epoch)), limits, budget)
        }) {
            Ok(view) => ResourcePlanningAvailability::Known(view),
            Err(reason) => ResourcePlanningAvailability::Unknown(reason),
        }
    }

    pub(super) fn resource_planning_view_with_graph_on_lane(
        self: &Arc<Self>,
        sessions: &[&SequenceSession<R>],
        lane: &ExecutionLane<R>,
        limits: ResourcePlanningLimits,
        budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<
        (
            ResourcePlanningView,
            Option<crate::vnext::DeviceCostGraphStreamState>,
            Option<crate::vnext::DeviceCostGraphCatalog>,
        ),
        ResourcePlanningUnknown,
    > {
        if !Arc::ptr_eq(&self.runtime, lane.runtime_arc()) {
            return Err(ResourcePlanningUnknown::StaleIdentity);
        }
        lane.try_with_cost_planning_lane(limits, budget, |epoch, graph, catalog, budget| {
            self.capture_resource_planning_view(sessions, Some((lane.id(), epoch)), limits, budget)
                .map(|view| (view, graph, catalog))
        })
    }

    fn capture_resource_planning_view(
        &self,
        sessions: &[&SequenceSession<R>],
        lane: Option<(ExecutionLaneId, u64)>,
        limits: ResourcePlanningLimits,
        poll_budget: &mut dyn ResourcePlanningBudget,
    ) -> Result<ResourcePlanningView, ResourcePlanningUnknown> {
        use ResourcePlanningReadStage as Stage;
        use ResourcePlanningUnknown as U;
        poll(poll_budget)?;
        if !limits.is_valid() || sessions.is_empty() {
            return Err(U::InvalidInput);
        }
        let pools = &self.dynamic_pools;
        if sessions.len() > limits.maximum_participants
            || pools.pools.len() > limits.maximum_pools
            || pools.domains.len() > limits.maximum_pools
            || pools
                .domains
                .iter()
                .try_fold(0_usize, |sum, domain| {
                    sum.checked_add(domain.descriptors.len())
                })
                .is_none_or(|count| count > limits.maximum_descriptors)
            || pools.nodes.len() > limits.maximum_descriptors
        {
            return Err(U::LimitExceeded);
        }
        if pools
            .reusable_execution
            .as_ref()
            .is_some_and(|plan| plan.buckets().len() > limits.maximum_descriptors)
        {
            return Err(U::LimitExceeded);
        }
        // Retained lane slots have different acquisition/reuse rules. Never
        // charge them as disposable eager extents or assume an empty lane.
        if pools.reusable_execution.is_some() && lane.is_none() {
            return Err(U::ReusableExecution);
        }
        let _lifecycle = self.try_read_planning_lifecycle()?;
        let mut physical_ranges = physical_ranges::capture_static(self, limits, poll_budget)?;
        let mut slots = Vec::with_capacity(sessions.len());
        let mut backings = Vec::with_capacity(sessions.len());
        let mut participants = Vec::with_capacity(sessions.len());
        let mut sequence_ranges = Vec::with_capacity(sessions.len());
        let mut sequence_segments = 0_usize;
        for session in sessions {
            poll(poll_budget)?;
            if session.resources().coordinator_id() != pools.logical_admission.id()
                || participants.iter().any(|p: &ResourcePlanningParticipant| {
                    p.authority == session.sequence_authority()
                })
            {
                return Err(U::StaleIdentity);
            }
            slots.push(
                session
                    .slot
                    .state
                    .try_lock()
                    .map_err(|error| read_lock_error(error, Stage::SequenceSession))?,
            );
            let SequenceSessionSlotState::Active(active) = &**slots.last().unwrap() else {
                return Err(U::BusyOrUnavailable);
            };
            if active.epoch != session.epoch()
                || &active.fingerprint != session.fingerprint()
                || active.phase != SequenceSessionPhase::Open
                || active.active_frame.is_some()
                || active.has_participant_flights()
                || active.state_transfer.is_reserved()
            {
                return Err(U::BusyOrUnavailable);
            }
            backings.push(
                session
                    .resources()
                    .try_lock_planning_backing()
                    .map_err(|_| U::BusyOrUnavailable)?
                    .ok_or(U::ReadUnavailable(Stage::SequenceBacking))?,
            );
            let backing = &backings.last().unwrap().current;
            // Token-span execution has no independently supplied page work.
            // Page-evidence protocols need an explicit adapter, not page guesses.
            if backing.committed_pages() != 0 {
                return Err(U::Unsupported);
            }
            sequence_ranges.push(if physical_ranges.is_some() {
                sequence_ranges::capture(
                    backing.backing_slices(),
                    limits.maximum_free_extents,
                    &mut sequence_segments,
                    poll_budget,
                )?
            } else {
                Arc::new(BTreeMap::new())
            });
            participants.push(ResourcePlanningParticipant {
                authority: session.sequence_authority(),
                epoch: session.epoch(),
                fingerprint: session.fingerprint().clone(),
                backing_generation: backing.generation(),
                covered: backing.committed_shape(),
                maximum_tokens: session
                    .resources()
                    .request_resources()
                    .work_shape()
                    .fit_tokens(),
                retired_frames: active.retired_frames,
                pending_zero_initializations: pending_zero_initializations(
                    BatchParticipantAuthority::new(
                        session.sequence_authority(),
                        session.request_authority(),
                    ),
                    backing.backing_slices(),
                    limits.maximum_free_extents,
                    poll_budget,
                )?,
            });
        }
        // Hold the arena read across logical and physical snapshots. A retained
        // slot's capacity must never also appear in copied free extents.
        let arena_guard = lane
            .map(|_| {
                pools
                    .lane_stable_arenas
                    .try_lock()
                    .map_err(|error| read_lock_error(error, Stage::LaneWorkspace))
            })
            .transpose()?;
        let workspace = match (lane, arena_guard.as_deref()) {
            (Some((id, epoch)), Some(arena)) => {
                Some(workspace::capture(arena, id, epoch, limits, poll_budget)?)
            }
            (None, None) => None,
            _ => return Err(U::InvalidDemand),
        };
        // Some release paths acquire these locks in a different order. This
        // chain only uses try_lock and never waits while retaining a guard.
        pools
            .logical_admission
            .with_planning_snapshot(limits.maximum_pools, |logical| {
                let budget = &pools.budget;
                let account = budget
                    .account
                    .state
                    .try_lock()
                    .map_err(|error| read_lock_error(error, Stage::DeviceBudget))?;
                if account.poisoned {
                    return Err(U::BusyOrUnavailable);
                }
                if account.budgets.len() > limits.maximum_device_budgets {
                    return Err(U::LimitExceeded);
                }
                let record = account
                    .budgets
                    .get(&budget.budget_id)
                    .ok_or(U::StaleIdentity)?;
                let process_ceiling = account
                    .budgets
                    .values()
                    .map(|record| record.device_wide_usable_ceiling_bytes)
                    .min()
                    .ok_or(U::StaleIdentity)?;
                let budget_view = BudgetReadView {
                    budget_id: budget.budget_id,
                    process_claimed: account.claimed_bytes,
                    plan_claimed: record.claimed_bytes,
                    plan_ceiling: budget.device_wide_usable_ceiling_bytes,
                    process_ceiling,
                    next_budget_id: account.next_budget_id,
                };
                if budget_view.process_claimed > process_ceiling
                    || budget_view.plan_claimed > budget_view.plan_ceiling
                {
                    return Err(U::MaintenanceRequired);
                }
                let mut guards = Vec::with_capacity(pools.pools.len());
                for pool in pools.pools.values() {
                    poll(poll_budget)?;
                    guards.push(
                        pool.state
                            .try_lock()
                            .map_err(|error| read_lock_error(error, Stage::PhysicalPool))?,
                    );
                }
                let mut extent_count = 0_usize;
                let mut views = Vec::with_capacity(pools.pools.len());
                for ((id, pool), state) in pools.pools.iter().zip(&guards) {
                    poll(poll_budget)?;
                    if state.poisoned {
                        return Err(U::BusyOrUnavailable);
                    }
                    if state.allocator.by_size.len() != state.allocator.by_offset.len() {
                        return Err(U::InvalidDemand);
                    }
                    if state.pending_growth_bytes != 0 {
                        return Err(U::MaintenanceRequired);
                    }
                    if let Some(physical) = physical_ranges.as_mut() {
                        if physical
                            .chunks
                            .len()
                            .checked_add(state.chunks.len())
                            .is_none_or(|count| count > limits.maximum_free_extents)
                        {
                            return Err(U::LimitExceeded);
                        }
                        for chunk in state.chunks.values() {
                            poll(poll_budget)?;
                            let backing = &chunk.backing;
                            let Some(range) = self.runtime.cost_buffer_range(&backing.buffer)
                            else {
                                physical_ranges = None;
                                break;
                            };
                            if backing.identity.pool_id() != id
                                || range.length() != backing.descriptor.size_bytes
                                || physical
                                    .chunks
                                    .insert(backing.identity.clone(), range)
                                    .is_some()
                            {
                                return Err(U::InvalidDemand);
                            }
                        }
                    }
                    extent_count = extent_count
                        .checked_add(state.allocator.by_offset.len())
                        .ok_or(U::LimitExceeded)?;
                    if extent_count > limits.maximum_free_extents {
                        return Err(U::LimitExceeded);
                    }
                    views.push(PoolReadView {
                        id: id.clone(),
                        instance: pool.instance_id,
                        next_extent_generation: pool.next_extent_generation.load(Ordering::Acquire),
                        resident_bytes: state.resident_bytes,
                        allocator: state.allocator.clone(),
                    });
                }
                poll(poll_budget)?;
                Ok(ResourcePlanningView {
                    fence: Arc::new(()),
                    plan_hash: self.planning_plan_hash().clone(),
                    coordinator_id: pools.logical_admission.id(),
                    lane_id: lane.map(|(id, _)| id),
                    limits,
                    logical,
                    budget: budget_view,
                    pools: views,
                    participants,
                    workspace,
                    physical_ranges,
                    sequence_ranges,
                })
            })
            .map_err(|_| U::BusyOrUnavailable)?
            .ok_or(U::ReadUnavailable(Stage::LogicalCapacity))?
    }
}

fn pending_zero_initializations(
    owner: BatchParticipantAuthority,
    slices: &[LogicalBackingSliceAuthority],
    maximum_segments: usize,
    budget: &mut dyn ResourcePlanningBudget,
) -> Result<Option<Arc<[PendingZeroInitialization]>>, ResourcePlanningUnknown> {
    use super::super::backing_initialization::order::{InitializationOrder, InitializationRanges};
    let mut ordering = InitializationOrder::new();
    let mut visited = 0_usize;
    for slice in slices {
        poll(budget)?;
        if slice.evidence().initialization() != StateInitialization::Zero {
            continue;
        }
        let Some(cell) = slice.initialization_cell() else {
            return Ok(None);
        };
        match cell.status() {
            Ok(BackingInitializationStatus::Initialized) => continue,
            Ok(BackingInitializationStatus::Pending) => {}
            _ => return Ok(None),
        }
        for _ in &slice.evidence().segments {
            poll(budget)?;
            visited = visited
                .checked_add(1)
                .ok_or(ResourcePlanningUnknown::LimitExceeded)?;
            if visited > maximum_segments {
                return Err(ResourcePlanningUnknown::LimitExceeded);
            }
        }
        ordering
            .insert(owner, slice)
            .map_err(|_| ResourcePlanningUnknown::InvalidDemand)?;
    }
    let mut claims = Vec::new();
    for (_, cell, ordered_slices) in ordering.finish() {
        poll(budget)?;
        let mut seen = InitializationRanges::new();
        let mut bytes = Vec::new();
        for slice in ordered_slices {
            for segment in slice.evidence().segments() {
                poll(budget)?;
                if seen.insert(segment) {
                    bytes
                        .try_reserve(1)
                        .map_err(|_| ResourcePlanningUnknown::LimitExceeded)?;
                    bytes.push(segment.length_bytes());
                }
            }
        }
        claims
            .try_reserve(1)
            .map_err(|_| ResourcePlanningUnknown::LimitExceeded)?;
        claims.push(PendingZeroInitialization {
            target_fingerprint: Arc::from(cell.target_fingerprint()),
            transfer_bytes: bytes.into(),
        });
    }
    Ok(Some(claims.into()))
}
