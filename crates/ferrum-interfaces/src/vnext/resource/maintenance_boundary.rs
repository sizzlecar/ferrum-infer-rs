use serde::Serialize;

use super::{
    BackingChunkIdentity, CapacityDomainId, CapacityVector, DynamicBackingPackingEnvelope,
    DynamicBackingPoolId, DynamicPoolLiveOccupancyStatus, LogicalAdmissionCoordinatorId,
};
use crate::vnext::DeviceCapacityPressure;

pub const DYNAMIC_POOL_MAINTENANCE_BOUNDARY_SCHEMA_VERSION: u32 = 1;

/// One resident chunk as observed while every pool maintenance/state lock is
/// held and before a pressure-driven rebalance mutates residency.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolMaintenanceBoundaryChunk {
    pub(in crate::vnext::resource) identity: BackingChunkIdentity,
    pub(in crate::vnext::resource) bytes: u64,
    pub(in crate::vnext::resource) live_segments: u64,
    pub(in crate::vnext::resource) external_references: usize,
    pub(in crate::vnext::resource) protected_packing: bool,
    pub(in crate::vnext::resource) full_extent_available: bool,
    pub(in crate::vnext::resource) resident_floor_allows_reclaim: bool,
    pub(in crate::vnext::resource) reclaim_candidate: bool,
}

impl DynamicPoolMaintenanceBoundaryChunk {
    pub fn identity(&self) -> &BackingChunkIdentity {
        &self.identity
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn live_segments(&self) -> u64 {
        self.live_segments
    }

    pub const fn external_references(&self) -> usize {
        self.external_references
    }

    pub const fn protected_packing(&self) -> bool {
        self.protected_packing
    }

    pub const fn full_extent_available(&self) -> bool {
        self.full_extent_available
    }

    pub const fn resident_floor_allows_reclaim(&self) -> bool {
        self.resident_floor_allows_reclaim
    }

    pub const fn reclaim_candidate(&self) -> bool {
        self.reclaim_candidate
    }
}

/// Event-bound physical and logical state for one pool at a failed device
/// reservation. Consumers can recompute the complete reclaim frontier instead
/// of inferring it from a later health snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolMaintenanceBoundaryPool {
    pub(in crate::vnext::resource) pool_id: DynamicBackingPoolId,
    pub(in crate::vnext::resource) domain_id: CapacityDomainId,
    pub(in crate::vnext::resource) excluded_from_reclaim: bool,
    pub(in crate::vnext::resource) resident_bytes: u64,
    pub(in crate::vnext::resource) pending_growth_bytes: u64,
    pub(in crate::vnext::resource) free_bytes: u64,
    pub(in crate::vnext::resource) largest_contiguous_bytes: u64,
    pub(in crate::vnext::resource) free_extent_layout_fingerprint: String,
    pub(in crate::vnext::resource) logical_used_bytes: u64,
    pub(in crate::vnext::resource) live_occupancy: DynamicPoolLiveOccupancyStatus,
    pub(in crate::vnext::resource) minimum_resident_bytes: u64,
    pub(in crate::vnext::resource) maximum_resident_bytes: u64,
    pub(in crate::vnext::resource) protected_immediate_bytes: u64,
    pub(in crate::vnext::resource) protected_packing_satisfied: bool,
    pub(in crate::vnext::resource) coherent_runnable_floor_bytes: u64,
    pub(in crate::vnext::resource) resident_floor_bytes: u64,
    pub(in crate::vnext::resource) reclaimable_bytes: u64,
    pub(in crate::vnext::resource) chunks: Vec<DynamicPoolMaintenanceBoundaryChunk>,
}

impl DynamicPoolMaintenanceBoundaryPool {
    pub fn pool_id(&self) -> &DynamicBackingPoolId {
        &self.pool_id
    }

    pub const fn domain_id(&self) -> CapacityDomainId {
        self.domain_id
    }

    pub const fn excluded_from_reclaim(&self) -> bool {
        self.excluded_from_reclaim
    }

    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub const fn pending_growth_bytes(&self) -> u64 {
        self.pending_growth_bytes
    }

    pub const fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    pub const fn largest_contiguous_bytes(&self) -> u64 {
        self.largest_contiguous_bytes
    }

    pub fn free_extent_layout_fingerprint(&self) -> &str {
        &self.free_extent_layout_fingerprint
    }

    pub const fn logical_used_bytes(&self) -> u64 {
        self.logical_used_bytes
    }

    pub const fn live_occupancy(&self) -> &DynamicPoolLiveOccupancyStatus {
        &self.live_occupancy
    }

    pub const fn minimum_resident_bytes(&self) -> u64 {
        self.minimum_resident_bytes
    }

    pub const fn maximum_resident_bytes(&self) -> u64 {
        self.maximum_resident_bytes
    }

    pub const fn protected_immediate_bytes(&self) -> u64 {
        self.protected_immediate_bytes
    }

    pub const fn protected_packing_satisfied(&self) -> bool {
        self.protected_packing_satisfied
    }

    pub const fn coherent_runnable_floor_bytes(&self) -> u64 {
        self.coherent_runnable_floor_bytes
    }

    pub const fn resident_floor_bytes(&self) -> u64 {
        self.resident_floor_bytes
    }

    pub const fn reclaimable_bytes(&self) -> u64 {
        self.reclaimable_bytes
    }

    pub fn chunks(&self) -> &[DynamicPoolMaintenanceBoundaryChunk] {
        &self.chunks
    }
}

/// Atomic cold-path receipt for the exact capacity boundary that caused a
/// rebalance attempt. Selected chunks describe the planner decision before
/// mutation; `reclaim_sufficient=false` is the typed reason maintenance must
/// wait for a release epoch instead of retrying allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DynamicPoolMaintenanceBoundaryReceipt {
    pub(in crate::vnext::resource) schema_version: u32,
    pub(in crate::vnext::resource) coordinator_id: LogicalAdmissionCoordinatorId,
    pub(in crate::vnext::resource) logical_release_epoch: u64,
    pub(in crate::vnext::resource) logical_capacity_epoch: u64,
    pub(in crate::vnext::resource) plan_device_capacity_epoch: u64,
    pub(in crate::vnext::resource) process_device_capacity_epoch: u64,
    pub(in crate::vnext::resource) pressure: DeviceCapacityPressure,
    pub(in crate::vnext::resource) planned_domains: Vec<CapacityDomainId>,
    pub(in crate::vnext::resource) protected_immediate: CapacityVector,
    pub(in crate::vnext::resource) protected_packing_envelopes: Vec<DynamicBackingPackingEnvelope>,
    pub(in crate::vnext::resource) pools: Vec<DynamicPoolMaintenanceBoundaryPool>,
    pub(in crate::vnext::resource) reclaim_candidate_chunks: usize,
    pub(in crate::vnext::resource) reclaim_candidate_bytes: u64,
    pub(in crate::vnext::resource) selected_chunks: Vec<BackingChunkIdentity>,
    pub(in crate::vnext::resource) selected_bytes: u64,
    pub(in crate::vnext::resource) reclaim_sufficient: bool,
}

impl DynamicPoolMaintenanceBoundaryReceipt {
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub const fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub const fn logical_release_epoch(&self) -> u64 {
        self.logical_release_epoch
    }

    pub const fn logical_capacity_epoch(&self) -> u64 {
        self.logical_capacity_epoch
    }

    pub const fn plan_device_capacity_epoch(&self) -> u64 {
        self.plan_device_capacity_epoch
    }

    pub const fn process_device_capacity_epoch(&self) -> u64 {
        self.process_device_capacity_epoch
    }

    pub const fn pressure(&self) -> &DeviceCapacityPressure {
        &self.pressure
    }

    pub fn planned_domains(&self) -> &[CapacityDomainId] {
        &self.planned_domains
    }

    pub const fn protected_immediate(&self) -> &CapacityVector {
        &self.protected_immediate
    }

    pub fn protected_packing_envelopes(&self) -> &[DynamicBackingPackingEnvelope] {
        &self.protected_packing_envelopes
    }

    pub fn pools(&self) -> &[DynamicPoolMaintenanceBoundaryPool] {
        &self.pools
    }

    pub const fn reclaim_candidate_chunks(&self) -> usize {
        self.reclaim_candidate_chunks
    }

    pub const fn reclaim_candidate_bytes(&self) -> u64 {
        self.reclaim_candidate_bytes
    }

    pub fn selected_chunks(&self) -> &[BackingChunkIdentity] {
        &self.selected_chunks
    }

    pub const fn selected_bytes(&self) -> u64 {
        self.selected_bytes
    }

    pub const fn reclaim_sufficient(&self) -> bool {
        self.reclaim_sufficient
    }
}
