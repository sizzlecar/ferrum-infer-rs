//! Native storage ownership for a plan-authenticated compact state checkpoint.
//!
//! This layer allocates storage only. It does not certify state contents or
//! publish a restorable checkpoint; the state-transfer completion protocol must
//! retain this owner until the copy has a proven terminal outcome.

use super::{
    align_up_resource, invalid_resource, AdmissionDeferred, AdmissionDemand, AdmissionFitPolicy,
    AdmissionPressureAction, AdmissionRejected, AllocationKind, AllocationLifetime, Arc,
    BackingClaimCertificate, BackingPrepareDecision, BufferUsage, CapacityEntry, CapacityUnits,
    CapacityVector, DeviceBufferRetention, DeviceRuntime, DynamicBackingDeferred,
    EvaluatedBackingProjection, EvaluatedBackingRequest, LogicalBackingBufferView,
    LogicalBackingSliceAuthority, LogicalBackingSliceEvidence, PhysicalBackingClaimIdentity,
    PlanHash, PlanRuntimeResources, ResourceId, TrustedPlanRuntimeBinding, VNextError,
};
use crate::vnext::{
    CheckpointAuthorityId, CheckpointCapacityClaimDecision, CheckpointRetentionSkipReason,
    LogicalCheckpointLease,
};
use std::collections::BTreeMap;
mod transfer;
use std::sync::Mutex;

mod capture;
pub(crate) use capture::*;

/// Compact bytes for one base resource after the plan layout has merged all
/// aliases and validated the actual source ranges at the capture boundary.
#[derive(Debug)]
pub(crate) struct CheckpointBackingRequest {
    resource_id: ResourceId,
    logical_bytes: u64,
}

impl CheckpointBackingRequest {
    pub(crate) fn new(resource_id: ResourceId, logical_bytes: u64) -> Result<Self, VNextError> {
        if logical_bytes == 0 {
            return Err(invalid_resource("checkpoint backing cannot be empty"));
        }
        Ok(Self {
            resource_id,
            logical_bytes,
        })
    }

    pub(crate) fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub(crate) const fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }
}

/// Crate-internal input from the plan's certified checkpoint layout. A wire
/// payload or external caller cannot construct this allocation capability.
#[derive(Debug)]
pub(crate) struct CheckpointBackingRequests {
    plan_hash: PlanHash,
    requests: Vec<CheckpointBackingRequest>,
}

impl CheckpointBackingRequests {
    pub(crate) fn new(
        plan_hash: PlanHash,
        mut requests: Vec<CheckpointBackingRequest>,
    ) -> Result<Self, VNextError> {
        requests.sort_by(|left, right| left.resource_id.cmp(&right.resource_id));
        if requests.is_empty()
            || requests
                .windows(2)
                .any(|pair| pair[0].resource_id == pair[1].resource_id)
        {
            return Err(invalid_resource(
                "checkpoint layout must supply non-empty unique base resources",
            ));
        }
        Ok(Self {
            plan_hash,
            requests,
        })
    }

    pub(crate) fn requests(&self) -> &[CheckpointBackingRequest] {
        &self.requests
    }
}

pub(crate) enum CheckpointBackingAllocationDecision<R: DeviceRuntime> {
    Allocated(Arc<CheckpointBackingOwner<R>>),
    Skipped(CheckpointRetentionSkipReason),
    Deferred(AdmissionDeferred),
    BackingDeferred(DynamicBackingDeferred),
    PermanentRejected(AdmissionRejected),
}

/// Owned, unpublished storage. Cloning the enclosing Arc pins the complete
/// physical and logical ownership, including after cache index eviction.
///
/// Field order is part of the release contract: extents are returned first,
/// then logical availability is published, and only then may the plan close.
#[must_use = "checkpoint backing and its capacity must remain owned through device completion"]
pub(crate) struct CheckpointBackingOwner<R: DeviceRuntime> {
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    logical_lease: LogicalCheckpointLease,
    plan: Arc<PlanRuntimeResources<R>>,
    logical_bytes: u64,
    extent_bytes: u64,
    capture: Mutex<CheckpointCaptureState>,
}

impl<R: DeviceRuntime> CheckpointBackingOwner<R> {
    pub(crate) fn authority(&self) -> CheckpointAuthorityId {
        self.logical_lease.authority()
    }

    pub(crate) fn claims(&self) -> &CapacityVector {
        self.logical_lease.claims()
    }

    /// Union of valid content bytes, before allocator padding. Each base
    /// resource appears once, even if several model states alias it.
    pub(crate) const fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    /// Independently retained aligned extents, not entire shared pool chunks.
    pub(crate) const fn extent_bytes(&self) -> u64 {
        self.extent_bytes
    }

    pub(crate) fn backing_evidence(&self) -> impl Iterator<Item = &LogicalBackingSliceEvidence> {
        self.backing_slices
            .iter()
            .map(LogicalBackingSliceAuthority::evidence)
    }

    pub(in crate::vnext::resource) fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        &self.backing_slices
    }

    pub(in crate::vnext::resource) fn plan_resources(&self) -> &Arc<PlanRuntimeResources<R>> {
        &self.plan
    }

    pub(crate) fn device_buffer_retention(self: &Arc<Self>) -> DeviceBufferRetention {
        // Checkpoints have no reusable address scope. The second reference is
        // intentional: use the existing opaque retention without inventing a
        // plan-address or execution-lane identity for this transient storage.
        DeviceBufferRetention::pair(Arc::clone(self), Arc::clone(&self.plan))
    }

    pub(crate) fn view<'owner>(
        self: &'owner Arc<Self>,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'owner, R::Buffer>, VNextError> {
        let index = self
            .backing_slices
            .binary_search_by(|slice| slice.resource_id().cmp(resource_id))
            .map_err(|_| invalid_resource("checkpoint does not own this resource"))?;
        let mut view = self.plan.dynamic_pools.view(&self.backing_slices[index])?;
        for binding in &mut view.bindings {
            // A command may retain a segment after dropping the borrowed
            // view. Pin the logical lease and plan as well as physical bytes.
            binding.retention =
                DeviceBufferRetention::pair(Arc::clone(self), Arc::clone(&binding.chunk));
        }
        Ok(view)
    }
}

impl<R: DeviceRuntime> TrustedPlanRuntimeBinding<R> {
    /// Atomically prepares physical extents, claims the exact same logical
    /// capacity, then commits both under the plan's lifecycle read gate.
    /// Optional capture never grows or waits for backing in this call.
    pub(crate) fn try_allocate_checkpoint_backing(
        &self,
        request: &CheckpointBackingRequests,
    ) -> Result<CheckpointBackingAllocationDecision<R>, VNextError> {
        let _lifecycle = self
            .resources
            .read_lifecycle("allocate checkpoint backing")?;
        if request.plan_hash != *self.plan_hash() {
            return Err(invalid_resource(
                "checkpoint layout belongs to another plan",
            ));
        }
        let mut requested_slices = Vec::with_capacity(request.requests.len());
        let mut domain_bytes = BTreeMap::new();
        let mut logical_bytes = 0_u64;
        let mut extent_bytes = 0_u64;
        for resource in &request.requests {
            let (domain, descriptor) = self
                .dynamic_pools()
                .domains
                .iter()
                .find_map(|domain| {
                    domain
                        .descriptors
                        .iter()
                        .find(|descriptor| descriptor.base_resource_id() == &resource.resource_id)
                        .map(|descriptor| (domain, descriptor))
                })
                .ok_or_else(|| {
                    invalid_resource("checkpoint references an unknown plan resource")
                })?;
            if descriptor.lifetime() != AllocationLifetime::Sequence
                || descriptor.usage() != BufferUsage::State
                || *descriptor.kind() != AllocationKind::Value
                || resource.logical_bytes > descriptor.theoretical_maximum_request_bytes()?
            {
                return Err(invalid_resource(
                    "checkpoint allocation must fit the plan's Sequence state descriptor",
                ));
            }
            let pool = self
                .dynamic_pools()
                .pools
                .get(domain.pool_id())
                .ok_or_else(|| invalid_resource("checkpoint descriptor has no State pool"))?;
            let capacity_bytes =
                align_up_resource(resource.logical_bytes, pool.allocation_quantum())?;
            let total = domain_bytes.entry(domain.domain_id()).or_insert(0_u64);
            *total = total
                .checked_add(capacity_bytes)
                .ok_or_else(|| invalid_resource("checkpoint domain capacity overflows u64"))?;
            logical_bytes = logical_bytes
                .checked_add(resource.logical_bytes)
                .ok_or_else(|| invalid_resource("checkpoint logical bytes overflow u64"))?;
            extent_bytes = extent_bytes
                .checked_add(capacity_bytes)
                .ok_or_else(|| invalid_resource("checkpoint extent bytes overflow u64"))?;
            requested_slices.push(EvaluatedBackingRequest {
                domain,
                claim_identity: PhysicalBackingClaimIdentity::new(
                    domain.pool_id().clone(),
                    vec![resource.resource_id.clone()],
                )?,
                capacity_size_bytes: capacity_bytes,
                reusable_execution_bucket_id: None,
                projections: vec![EvaluatedBackingProjection {
                    descriptor,
                    physical_offset_bytes: 0,
                    logical_size_bytes: resource.logical_bytes,
                    capacity_size_bytes: capacity_bytes,
                }],
            });
        }
        let capacity = CapacityVector::new(
            domain_bytes
                .into_iter()
                .map(|(domain, bytes)| CapacityEntry::new(domain, CapacityUnits::new(bytes)))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let demand = AdmissionDemand::from_plan(
            capacity.clone(),
            capacity,
            AdmissionFitPolicy::ImmediateOnly,
            AdmissionPressureAction::WaitForRelease,
        )?;
        // Charge all domains and the aggregate retention fee atomically before
        // preparing physical extents. Later locals drop first on rollback.
        let logical_lease = match self
            .logical_admission()
            .try_claim_checkpoint(&demand, extent_bytes)?
        {
            CheckpointCapacityClaimDecision::Claimed(lease) => lease,
            CheckpointCapacityClaimDecision::Skipped(reason) => {
                return Ok(CheckpointBackingAllocationDecision::Skipped(reason));
            }
            CheckpointCapacityClaimDecision::Deferred(deferred) => {
                return Ok(CheckpointBackingAllocationDecision::Deferred(deferred));
            }
            CheckpointCapacityClaimDecision::PermanentRejected(rejected) => {
                return Ok(CheckpointBackingAllocationDecision::PermanentRejected(
                    rejected,
                ));
            }
        };
        if !self
            .logical_admission()
            .owns_checkpoint_claim(&logical_lease)
        {
            drop(logical_lease);
            return Err(invalid_resource(
                "checkpoint claim belongs to another coordinator",
            ));
        }
        let prepared = match self
            .dynamic_pools()
            .prepare_checkpoint_claim(&requested_slices)?
        {
            BackingPrepareDecision::Prepared(prepared) => prepared,
            BackingPrepareDecision::Deferred(deferred) => {
                return Ok(CheckpointBackingAllocationDecision::BackingDeferred(
                    deferred,
                ));
            }
        };
        let owner = CheckpointBackingOwner {
            backing_slices: prepared.commit(),
            logical_lease,
            plan: Arc::clone(&self.resources),
            logical_bytes,
            extent_bytes,
            capture: Mutex::new(CheckpointCaptureState::default()),
        };
        // Verify the committed extents and charged demand together before
        // returning an owner. Error drop retains physical-before-logical order.
        BackingClaimCertificate::from_slices(&owner.backing_slices)?
            .bind(&owner.backing_slices, &demand)?;
        Ok(CheckpointBackingAllocationDecision::Allocated(Arc::new(
            owner,
        )))
    }
}
