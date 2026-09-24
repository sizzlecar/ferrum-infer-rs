//! Bounded numerical resource simulation. A fit is evidence, never a lease.
//!
//! Views deliberately own neither sessions, backing snapshots, device buffers,
//! nor runtime plans. The runtime is borrowed again to evaluate its immutable
//! demand formulas. Only resident allocations are modeled; maintenance remains
//! unknown. Lane-stable workspace slots retain their real
//! physical capacity across numerical waves.

use super::*;
use crate::vnext::{
    CapacityDomainId, CapacitySnapshot, DynamicResourceShape, ReusableExecutionBucketId,
    SequenceAuthorityId,
};

mod capture;
mod cost_route;
mod physical_ranges;
mod project;
mod sequence_ranges;
mod workspace;
pub(crate) use physical_ranges::ResourceCostRangeProof;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourcePlanningUnknown {
    Unsupported,
    BusyOrUnavailable,
    LimitExceeded,
    InvalidInput,
    StaleIdentity,
    ReusableExecution,
    MaintenanceRequired,
    LogicalCapacity,
    PhysicalCapacity,
    InvalidDemand,
    BudgetExhausted,
    ReadUnavailable(ResourcePlanningReadStage),
}

/// A failed nonblocking read, distinct from an observed cancelled, executing,
/// poisoned or unsupported state. Callers may retry within their wall budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResourcePlanningReadStage {
    Lifecycle,
    DeferredCleanup,
    SequenceSession,
    SequenceBacking,
    LogicalCapacity,
    DeviceBudget,
    PhysicalPool,
    ExecutionLane,
    LaneWorkspace,
    ModelRegistry,
    ModelRegistrySlot,
    ModelSequenceOperation,
}

pub(super) fn read_lock_error<T>(
    error: std::sync::TryLockError<T>,
    stage: ResourcePlanningReadStage,
) -> ResourcePlanningUnknown {
    match error {
        std::sync::TryLockError::WouldBlock => ResourcePlanningUnknown::ReadUnavailable(stage),
        std::sync::TryLockError::Poisoned(_) => ResourcePlanningUnknown::BusyOrUnavailable,
    }
}

/// Must be nonblocking. Called around every row, pool and demand/allocation
/// evaluation. An individual existing formula/extent operation is bounded by
/// the snapshot limits and is not preemptible; callers also check wall time
/// after this API returns.
pub trait ResourcePlanningBudget {
    fn has_budget(&mut self) -> bool;
}

impl<F: FnMut() -> bool> ResourcePlanningBudget for F {
    fn has_budget(&mut self) -> bool {
        self()
    }
}

fn poll(budget: &mut dyn ResourcePlanningBudget) -> Result<(), ResourcePlanningUnknown> {
    if budget.has_budget() {
        Ok(())
    } else {
        Err(ResourcePlanningUnknown::BudgetExhausted)
    }
}

#[derive(Debug, Clone)]
pub enum ResourcePlanningAvailability<T> {
    Known(T),
    Unknown(ResourcePlanningUnknown),
}

/// Bounds are enforced before cloning live state and at every simulated wave.
/// These defaults are engineering limits, not performance or capacity promises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourcePlanningLimits {
    pub maximum_participants: usize,
    pub maximum_pools: usize,
    pub maximum_descriptors: usize,
    pub maximum_free_extents: usize,
    pub maximum_device_budgets: usize,
    pub maximum_projected_waves: usize,
}

impl Default for ResourcePlanningLimits {
    fn default() -> Self {
        Self {
            maximum_participants: 256,
            maximum_pools: 256,
            maximum_descriptors: 4096,
            maximum_free_extents: 8192,
            maximum_device_budgets: 64,
            maximum_projected_waves: 64,
        }
    }
}

impl ResourcePlanningLimits {
    pub fn is_valid(self) -> bool {
        (1..=1024).contains(&self.maximum_participants)
            && (1..=1024).contains(&self.maximum_pools)
            && (1..=16384).contains(&self.maximum_descriptors)
            && (1..=65536).contains(&self.maximum_free_extents)
            && (1..=1024).contains(&self.maximum_device_budgets)
            && (1..=256).contains(&self.maximum_projected_waves)
    }
}

/// A numerical future token span. It is intentionally not execution authority.
/// The controller must separately prove logical work identity and frontiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourcePlanningRow {
    pub participant_index: usize,
    pub start_token: u64,
    pub token_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePlanningParticipant {
    authority: SequenceAuthorityId,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
    backing_generation: SequenceBackingGeneration,
    covered: DynamicResourceShape,
    maximum_tokens: u64,
    retired_frames: u64,
    pending_zero_transfer_bytes: Option<Arc<[u64]>>,
}

impl ResourcePlanningParticipant {
    pub fn matches_session_identity<R: DeviceRuntime>(&self, session: &SequenceSession<R>) -> bool {
        self.authority == session.sequence_authority()
            && self.epoch == session.epoch()
            && &self.fingerprint == session.fingerprint()
    }
    pub fn authority(&self) -> SequenceAuthorityId {
        self.authority
    }
    pub fn backing_generation(&self) -> SequenceBackingGeneration {
        self.backing_generation
    }
    pub fn covered_tokens(&self) -> u64 {
        self.covered.tokens()
    }
    pub fn maximum_tokens(&self) -> u64 {
        self.maximum_tokens
    }
    pub(crate) fn pending_zero_commands(&self) -> Option<u32> {
        self.pending_zero_transfer_bytes
            .as_ref()
            .and_then(|spans| u32::try_from(spans.len()).ok())
    }
    /// One length per unique pending physical zero extent. All entries are
    /// the same Fill path; order among lengths cannot alter their class chain
    /// or checked byte sum. This is copied numeric evidence, not a lease.
    pub(crate) fn pending_zero_transfer_bytes(&self) -> Option<&[u64]> {
        self.pending_zero_transfer_bytes.as_deref()
    }
}

#[derive(Debug, Clone)]
struct PoolReadView {
    id: DynamicBackingPoolId,
    instance: u64,
    next_extent_generation: u64,
    resident_bytes: u64,
    allocator: FreeExtentIndex,
}

impl PartialEq for PoolReadView {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.instance == other.instance
            && self.next_extent_generation == other.next_extent_generation
            && self.resident_bytes == other.resident_bytes
            && self.allocator.free_bytes == other.allocator.free_bytes
            && self.allocator.by_offset == other.allocator.by_offset
            && self.allocator.by_size == other.allocator.by_size
    }
}
impl Eq for PoolReadView {}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BudgetReadView {
    budget_id: u64,
    process_claimed: u64,
    plan_claimed: u64,
    plan_ceiling: u64,
    process_ceiling: u64,
    next_budget_id: u64,
}

/// The private fence binds simulation states to this exact snapshot. Its Arc
/// owns only `()`, so retaining a beam cannot retain any physical resource.
#[derive(Debug, Clone)]
pub struct ResourcePlanningView {
    fence: Arc<()>,
    plan_hash: PlanHash,
    coordinator_id: LogicalAdmissionCoordinatorId,
    lane_id: Option<ExecutionLaneId>,
    limits: ResourcePlanningLimits,
    logical: CapacitySnapshot,
    budget: BudgetReadView,
    pools: Vec<PoolReadView>,
    participants: Vec<ResourcePlanningParticipant>,
    workspace: Option<workspace::WorkspaceReadView>,
    physical_ranges: Option<physical_ranges::PhysicalRanges>,
    sequence_ranges: sequence_ranges::SequenceRanges,
}

impl ResourcePlanningView {
    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }
    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }
    pub fn lane_id(&self) -> Option<ExecutionLaneId> {
        self.lane_id
    }
    pub fn participants(&self) -> &[ResourcePlanningParticipant] {
        &self.participants
    }
    pub fn limits(&self) -> ResourcePlanningLimits {
        self.limits
    }
    pub fn initial_state(&self) -> ResourcePlanningState {
        ResourcePlanningState {
            fence: Arc::clone(&self.fence),
            pools: self.pools.clone(),
            workspace: self.workspace.clone(),
            logical_available: self
                .logical
                .domains()
                .iter()
                .map(|domain| (domain.domain(), domain.available().get()))
                .collect(),
            covered: self.participants.iter().map(|p| p.covered).collect(),
            sequence_ranges: self.sequence_ranges.clone(),
            waves: 0,
        }
    }
    /// Structural comparison is also sensitive to allocation layout, not just
    /// wake epochs. A successful comparison still grants no Step permission.
    pub fn same_live_evidence(&self, other: &Self) -> bool {
        self.plan_hash == other.plan_hash
            && self.coordinator_id == other.coordinator_id
            && self.lane_id == other.lane_id
            && self.limits == other.limits
            && self.logical == other.logical
            && self.budget == other.budget
            && self.pools == other.pools
            && self.participants == other.participants
            && self.workspace == other.workspace
            && self.physical_ranges == other.physical_ranges
            && self.sequence_ranges == other.sequence_ranges
    }
}

#[derive(Debug, Clone)]
pub struct ResourcePlanningState {
    fence: Arc<()>,
    pools: Vec<PoolReadView>,
    workspace: Option<workspace::WorkspaceReadView>,
    logical_available: BTreeMap<CapacityDomainId, u64>,
    covered: Vec<DynamicResourceShape>,
    sequence_ranges: sequence_ranges::SequenceRanges,
    waves: usize,
}

impl ResourcePlanningState {
    pub fn projected_waves(&self) -> usize {
        self.waves
    }
    pub fn covered_tokens(&self, participant: usize) -> Option<u64> {
        self.covered.get(participant).map(|shape| shape.tokens())
    }
    pub fn available_in_domain(&self, domain: CapacityDomainId) -> Option<u64> {
        self.logical_available.get(&domain).copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePlanningDomainDemand {
    pub domain: CapacityDomainId,
    pub persistent_bytes: u64,
    pub transient_peak_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct ResourcePlanningProjection {
    pub state: ResourcePlanningState,
    pub domains: Vec<ResourcePlanningDomainDemand>,
    /// Predicted selection from the same first-idle slot transaction as the
    /// resource simulation. Identity only: this owns no slot or physical lease.
    pub(crate) step_slot: Option<LaneStableArenaSlotIdentity>,
    pub(crate) physical_ranges: Option<ResourceCostRangeProof>,
}

impl ResourcePlanningProjection {
    pub fn selected_step_slot(&self) -> Option<&LaneStableArenaSlotIdentity> {
        self.step_slot.as_ref()
    }
}
