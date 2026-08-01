use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, Weak};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{
    invalid_resource, AllocationLifetime, BackingClaimCertificate, CapacityDomainId, DeviceRuntime,
    DynamicBackingDeferred, EvaluatedBackingProjection, EvaluatedBackingRequest, ExecutionLane,
    ExecutionLaneId, LaneStableArenaSlotIdentity, LogicalAdmissionCoordinator,
    LogicalBackingSliceAuthority, PhysicalBackingClaimIdentity, ResourceId, VNextError,
};
use crate::vnext::ReusableExecutionBucketId;

pub(super) struct CommittedLaneBackingClaim {
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    certificate: Arc<BackingClaimCertificate>,
    slot_lease: Option<LaneStableArenaSlotLease>,
}

impl CommittedLaneBackingClaim {
    #[cfg(test)]
    pub(super) fn certificate(&self) -> &Arc<BackingClaimCertificate> {
        &self.certificate
    }

    #[cfg(test)]
    pub(super) fn into_parts(
        self,
    ) -> (
        Vec<LogicalBackingSliceAuthority>,
        Option<LaneStableArenaSlotLease>,
    ) {
        (self.backing_slices, self.slot_lease)
    }

    pub(super) fn into_certified_parts(
        self,
    ) -> (
        Vec<LogicalBackingSliceAuthority>,
        Arc<BackingClaimCertificate>,
        Option<LaneStableArenaSlotLease>,
    ) {
        (self.backing_slices, self.certificate, self.slot_lease)
    }
}

pub(super) struct PreparedLaneBackingClaim {
    stable: Vec<LogicalBackingSliceAuthority>,
    certificate: Arc<BackingClaimCertificate>,
    slot_lease: Option<LaneStableArenaSlotLease>,
}

impl PreparedLaneBackingClaim {
    pub(super) fn new(
        stable: Vec<LogicalBackingSliceAuthority>,
        slot_lease: Option<LaneStableArenaSlotLease>,
    ) -> Result<Self, VNextError> {
        let certificate = Arc::new(BackingClaimCertificate::from_slices(&stable)?);
        Ok(Self {
            stable,
            certificate,
            slot_lease,
        })
    }

    pub(super) fn certified(
        stable: Vec<LogicalBackingSliceAuthority>,
        certificate: Arc<BackingClaimCertificate>,
        slot_lease: LaneStableArenaSlotLease,
    ) -> Self {
        Self {
            stable,
            certificate,
            slot_lease: Some(slot_lease),
        }
    }

    pub(super) fn commit(self) -> CommittedLaneBackingClaim {
        CommittedLaneBackingClaim {
            backing_slices: self.stable,
            certificate: self.certificate,
            slot_lease: self.slot_lease,
        }
    }
}

pub(super) enum LaneBackingPrepareDecision {
    Prepared(PreparedLaneBackingClaim),
    Deferred(DynamicBackingDeferred),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct LaneStableArenaKey {
    pub(super) lane_id: ExecutionLaneId,
    pub(super) lifetime: AllocationLifetime,
    pub(super) reusable_execution_bucket_id: ReusableExecutionBucketId,
    pub(super) layout_fingerprint: String,
}

pub(super) struct LaneStableProjectionBinding {
    pub(super) request_index: usize,
    pub(super) projection_index: usize,
}

pub(super) struct LaneStableArenaSlot {
    pub(super) slot_id: u64,
    pub(super) authorities: Vec<LogicalBackingSliceAuthority>,
    pub(super) certificate: Arc<BackingClaimCertificate>,
    pub(super) projection_bindings: Vec<LaneStableProjectionBinding>,
    pub(super) availability_domains: Vec<CapacityDomainId>,
    pub(super) in_use: bool,
    pub(super) last_used: u64,
}

impl LaneStableArenaSlot {
    pub(super) fn has_external_address_pins(&self) -> bool {
        self.authorities
            .iter()
            .enumerate()
            .filter(|(index, authority)| {
                !self.authorities[..*index]
                    .iter()
                    .any(|prior| Arc::ptr_eq(&prior.segment_lease, &authority.segment_lease))
            })
            .any(|(_, authority)| {
                let retained_by_slot = self
                    .authorities
                    .iter()
                    .filter(|candidate| {
                        Arc::ptr_eq(&candidate.segment_lease, &authority.segment_lease)
                    })
                    .count();
                Arc::strong_count(&authority.segment_lease) > retained_by_slot
            })
    }
}

pub(super) trait LaneStableArenaLane: Send + Sync {
    fn try_trim_reusable_executables(&self) -> Result<bool, VNextError>;
}

impl<R> LaneStableArenaLane for ExecutionLane<R>
where
    R: DeviceRuntime,
{
    fn try_trim_reusable_executables(&self) -> Result<bool, VNextError> {
        self.trim_reusable_executables_if_quiescent()
    }
}

pub(super) struct LaneStableArenaEntry {
    pub(super) lane: Weak<dyn LaneStableArenaLane>,
    pub(super) slots: BTreeMap<u64, LaneStableArenaSlot>,
}

pub(super) struct LaneStableArenaEvictionCandidate {
    pub(super) key: LaneStableArenaKey,
    pub(super) slot_id: u64,
    pub(super) last_used: u64,
    pub(super) lane: Arc<dyn LaneStableArenaLane>,
}

fn lane_stable_projection_matches(
    authority: &LogicalBackingSliceAuthority,
    request: &EvaluatedBackingRequest<'_>,
    projection: &EvaluatedBackingProjection<'_>,
) -> bool {
    request.claim_identity == *authority.evidence.physical_claim_identity()
        && projection.descriptor.base_resource_id() == authority.evidence.resource_id()
        && request.capacity_size_bytes == authority.evidence.physical_size_bytes
        && projection.physical_offset_bytes == authority.evidence.physical_offset_bytes
        && projection.capacity_size_bytes == authority.evidence.capacity_size_bytes
        && projection.logical_size_bytes != 0
        && projection.logical_size_bytes <= projection.capacity_size_bytes
}

pub(super) fn bind_lane_stable_slot_projections(
    authorities: &[LogicalBackingSliceAuthority],
    requests: &[&EvaluatedBackingRequest<'_>],
) -> Result<Vec<LaneStableProjectionBinding>, VNextError> {
    authorities
        .iter()
        .map(|authority| {
            let request_index = requests
                .binary_search_by(|request| {
                    request
                        .claim_identity
                        .cmp(authority.evidence.physical_claim_identity())
                })
                .map_err(|_| {
                    invalid_resource("lane-stable arena request lost its physical claim projection")
                })?;
            let request = requests[request_index];
            let projection_index = request
                .projections
                .binary_search_by(|projection| {
                    projection
                        .descriptor
                        .base_resource_id()
                        .cmp(authority.evidence.resource_id())
                })
                .map_err(|_| {
                    invalid_resource(
                        "lane-stable arena request lost its logical resource projection",
                    )
                })?;
            if !lane_stable_projection_matches(
                authority,
                request,
                &request.projections[projection_index],
            ) {
                return Err(invalid_resource(
                    "lane-stable arena request differs from its retained capacity layout",
                ));
            }
            Ok(LaneStableProjectionBinding {
                request_index,
                projection_index,
            })
        })
        .collect()
}

impl LaneStableArenaEntry {
    pub(super) fn claim_idle_slot(
        &mut self,
        lane_id: ExecutionLaneId,
        now: u64,
        requests: &[&EvaluatedBackingRequest<'_>],
    ) -> Result<
        Option<(
            u64,
            Vec<LogicalBackingSliceAuthority>,
            Arc<BackingClaimCertificate>,
            Vec<CapacityDomainId>,
        )>,
        VNextError,
    > {
        let Some(slot) = self.slots.values_mut().find(|slot| !slot.in_use) else {
            return Ok(None);
        };
        if slot.authorities.len() != slot.projection_bindings.len() {
            return Err(invalid_resource(
                "lane-stable arena slot lost its projection bindings",
            ));
        }
        let stable = slot
            .authorities
            .iter()
            .zip(&slot.projection_bindings)
            .map(|(authority, binding)| {
                let request = requests
                    .get(binding.request_index)
                    .copied()
                    .ok_or_else(|| {
                        invalid_resource(
                            "lane-stable arena request lost its physical claim projection",
                        )
                    })?;
                let projection = request
                    .projections
                    .get(binding.projection_index)
                    .ok_or_else(|| {
                        invalid_resource(
                            "lane-stable arena request lost its logical resource projection",
                        )
                    })?;
                if !lane_stable_projection_matches(authority, request, projection) {
                    return Err(invalid_resource(
                        "lane-stable arena request differs from its retained capacity layout",
                    ));
                }
                let mut retained = authority.retained_for_lane(lane_id);
                retained.evidence.logical_size_bytes = projection.logical_size_bytes;
                Ok(retained)
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        slot.in_use = true;
        slot.last_used = now;
        Ok(Some((
            slot.slot_id,
            stable,
            Arc::clone(&slot.certificate),
            slot.availability_domains.clone(),
        )))
    }
}

pub(super) struct LaneStableArenaState {
    pub(super) clock: u64,
    pub(super) next_slot_id: u64,
    pub(super) poisoned: bool,
    pub(super) entries: BTreeMap<LaneStableArenaKey, LaneStableArenaEntry>,
}

impl Default for LaneStableArenaState {
    fn default() -> Self {
        Self {
            clock: 0,
            next_slot_id: 1,
            poisoned: false,
            entries: BTreeMap::new(),
        }
    }
}

impl LaneStableArenaState {
    pub(super) fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

    pub(super) fn issue_slot_id(&mut self) -> Result<u64, VNextError> {
        let slot_id = self.next_slot_id;
        self.next_slot_id = self.next_slot_id.checked_add(1).ok_or_else(|| {
            invalid_resource("lane-stable arena slot identity space is exhausted")
        })?;
        Ok(slot_id)
    }

    pub(super) fn take_expired_lanes(&mut self) -> Result<Vec<LaneStableArenaEntry>, VNextError> {
        let expired = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.lane.upgrade().is_none())
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        if expired.iter().any(|key| {
            self.entries
                .get(key)
                .is_some_and(|entry| entry.slots.values().any(|slot| slot.in_use))
        }) {
            self.poisoned = true;
            return Err(invalid_resource(
                "lane-stable arena retained a busy slot after its execution lane expired",
            ));
        }
        Ok(expired
            .into_iter()
            .filter_map(|key| self.entries.remove(&key))
            .collect())
    }
}

pub(super) struct LaneStableArenaSlotLease {
    pub(super) arenas: Arc<Mutex<LaneStableArenaState>>,
    pub(super) logical_admission: LogicalAdmissionCoordinator,
    pub(super) availability_domains: Vec<CapacityDomainId>,
    pub(super) key: LaneStableArenaKey,
    pub(super) slot_id: u64,
}

impl LaneStableArenaSlotLease {
    pub(super) fn identity(&self) -> LaneStableArenaSlotIdentity {
        LaneStableArenaSlotIdentity::new(
            self.key.lane_id,
            self.key.lifetime,
            self.key.reusable_execution_bucket_id.clone(),
            self.key.layout_fingerprint.clone(),
            self.slot_id,
        )
    }
}

impl Drop for LaneStableArenaSlotLease {
    fn drop(&mut self) {
        let mut arenas = match self.arenas.lock() {
            Ok(arenas) => arenas,
            Err(poisoned) => {
                poisoned.into_inner().poisoned = true;
                return;
            }
        };
        if arenas.poisoned {
            return;
        }
        let now = arenas.tick();
        let released = arenas
            .entries
            .get_mut(&self.key)
            .and_then(|entry| entry.slots.get_mut(&self.slot_id))
            .is_some_and(|slot| {
                if !slot.in_use {
                    return false;
                }
                slot.in_use = false;
                slot.last_used = now;
                true
            });
        if !released {
            arenas.poisoned = true;
            return;
        }
        drop(arenas);

        let mut notification_failed = false;
        for domain in &self.availability_domains {
            notification_failed |= self
                .logical_admission
                .notify_domain_availability_changed(*domain)
                .is_err();
        }
        if notification_failed {
            let mut arenas = self
                .arenas
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            arenas.poisoned = true;
        }
    }
}

pub(super) fn lane_stable_layout_fingerprint(
    lifetime: AllocationLifetime,
    bucket_id: &ReusableExecutionBucketId,
    requests: &[&EvaluatedBackingRequest<'_>],
) -> Result<String, VNextError> {
    #[derive(Serialize)]
    struct Projection<'a> {
        resource_id: &'a ResourceId,
        physical_offset_bytes: u64,
        capacity_size_bytes: u64,
    }

    #[derive(Serialize)]
    struct Request<'a> {
        claim_identity: &'a PhysicalBackingClaimIdentity,
        capacity_size_bytes: u64,
        projections: Vec<Projection<'a>>,
    }

    #[derive(Serialize)]
    struct Material<'a> {
        domain: &'static str,
        lifetime: AllocationLifetime,
        reusable_execution_bucket_id: &'a ReusableExecutionBucketId,
        requests: Vec<Request<'a>>,
    }

    let material = Material {
        domain: "ferrum.runtime-vnext.lane-stable-layout.v1",
        lifetime,
        reusable_execution_bucket_id: bucket_id,
        requests: requests
            .iter()
            .map(|request| Request {
                claim_identity: &request.claim_identity,
                capacity_size_bytes: request.capacity_size_bytes,
                projections: request
                    .projections
                    .iter()
                    .map(|projection| Projection {
                        resource_id: projection.descriptor.base_resource_id(),
                        physical_offset_bytes: projection.physical_offset_bytes,
                        capacity_size_bytes: projection.capacity_size_bytes,
                    })
                    .collect(),
            })
            .collect(),
    };
    serde_json::to_vec(&material)
        .map(|bytes| format!("sha256/{:x}", Sha256::digest(bytes)))
        .map_err(|error| {
            invalid_resource(format!(
                "lane-stable layout fingerprint encode failed: {error}"
            ))
        })
}

pub(super) fn lane_stable_layout_key(
    lane_id: ExecutionLaneId,
    lifetime: AllocationLifetime,
    requests: &[&EvaluatedBackingRequest<'_>],
) -> Result<LaneStableArenaKey, VNextError> {
    if requests.is_empty()
        || requests
            .windows(2)
            .any(|pair| pair[0].claim_identity >= pair[1].claim_identity)
        || requests.iter().any(|request| {
            request.projections.is_empty()
                || request
                    .projections
                    .iter()
                    .any(|projection| projection.descriptor.lifetime() != lifetime)
        })
    {
        return Err(invalid_resource(
            "lane-stable arena layout is empty, non-canonical, or mixes lifetimes",
        ));
    }
    let bucket_id = requests[0]
        .reusable_execution_bucket_id
        .as_ref()
        .ok_or_else(|| {
            invalid_resource(
                "lane-stable arena requires an immutable-plan reusable execution bucket",
            )
        })?;
    if requests
        .iter()
        .any(|request| request.reusable_execution_bucket_id.as_ref() != Some(bucket_id))
    {
        return Err(invalid_resource(
            "lane-stable arena layout mixes reusable execution buckets",
        ));
    }
    Ok(LaneStableArenaKey {
        lane_id,
        lifetime,
        reusable_execution_bucket_id: bucket_id.clone(),
        layout_fingerprint: lane_stable_layout_fingerprint(lifetime, bucket_id, requests)?,
    })
}
