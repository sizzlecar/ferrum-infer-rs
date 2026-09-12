//! Coordinator-owned logical capacity for immutable state copies. Physical
//! backing and plan lifetime checks remain the resource layer's responsibility.

use super::*;

/// Process-local identity of one independent checkpoint capacity claim.
/// Identifiers are issued only by the owning coordinator and are never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct CheckpointAuthorityId {
    coordinator_id: LogicalAdmissionCoordinatorId,
    serial: u64,
}

impl CheckpointAuthorityId {
    pub const fn coordinator_id(self) -> LogicalAdmissionCoordinatorId {
        self.coordinator_id
    }

    pub const fn serial(self) -> u64 {
        self.serial
    }
}

#[derive(Debug)]
pub enum CheckpointCapacityClaimDecision {
    Claimed(LogicalCheckpointLease),
    Skipped(CheckpointRetentionSkipReason),
    Deferred(AdmissionDeferred),
    PermanentRejected(AdmissionRejected),
}

/// Optional retention policy decisions, independent of device pressure or
/// ordinary capacity-domain availability. A skipped capture does not wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointRetentionSkipReason {
    Disabled,
    Capacity {
        requested_bytes: u64,
        retained_bytes: u64,
        maximum_bytes: u64,
    },
}

#[derive(Debug, PartialEq, Eq)]
struct CheckpointClaimRecord {
    claims: CapacityVector,
    retained_bytes: u64,
}

#[derive(Debug)]
pub(super) struct CheckpointClaimLedger {
    next_serial: u64,
    live: BTreeMap<CheckpointAuthorityId, CheckpointClaimRecord>,
    capacity: Option<CheckpointCapacityPolicy>,
    retained_bytes: u64,
    closed: bool,
}

impl CheckpointClaimLedger {
    pub(super) fn new(capacity: Option<CheckpointCapacityPolicy>) -> Self {
        Self {
            next_serial: 1,
            live: BTreeMap::new(),
            capacity,
            retained_bytes: 0,
            closed: false,
        }
    }

    pub(super) fn count(&self) -> u64 {
        // The supported Rust targets have at most 64-bit address spaces.
        self.live.len() as u64
    }
}

/// A checkpoint's independent domain claim. This lease has no request or
/// sequence parent and confers no device allocation or execution authority.
///
/// The resource owner must release its physical extents before this logical
/// lease. A completion fence may retain that owner after cache eviction.
#[derive(Debug)]
#[must_use = "checkpoint capacity remains charged until its owner is released"]
pub struct LogicalCheckpointLease {
    inner: Arc<CoordinatorInner>,
    authority: CheckpointAuthorityId,
    claims: CapacityVector,
    retained_bytes: u64,
    released: bool,
}

impl LogicalAdmissionCoordinator {
    /// Aligned bytes reserved or retained by every live checkpoint lease,
    /// including owners outside the cache index. This is a limit within pool
    /// residency, not an additional DeviceCapacityBudget charge.
    pub fn checkpoint_retained_bytes(&self) -> Result<u64, VNextError> {
        Ok(self.inner.lock_state()?.checkpoint_claims.retained_bytes)
    }

    /// Claims plan-derived checkpoint demand in the existing domain ledger.
    /// The caller must hold the plan lifecycle guard and commit this together
    /// with prepared physical backing. This is not a product allocation API.
    pub(crate) fn try_claim_checkpoint(
        &self,
        demand: &AdmissionDemand,
        retained_bytes: u64,
    ) -> Result<CheckpointCapacityClaimDecision, VNextError> {
        if demand.immediate_claim.is_empty() {
            return Err(invalid_admission(
                "checkpoint claim requires non-empty demand",
            ));
        }
        // Resource domains are denominated in their actual aligned physical
        // bytes. Never accept a caller's smaller fee for those same claims.
        let claimed_bytes = demand
            .immediate_claim
            .entries()
            .iter()
            .try_fold(0_u64, |sum, claim| sum.checked_add(claim.units.get()))
            .ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::ArithmeticOverflow,
                    "checkpoint domain byte sum overflows u64",
                )
            })?;
        if retained_bytes == 0 || retained_bytes != claimed_bytes {
            return Err(invalid_admission(
                "checkpoint retention fee must equal the complete aligned domain claims",
            ));
        }
        let mut state = self.inner.lock_mutation()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        if state.checkpoint_claims.closed {
            return Err(invalid_admission("checkpoint admission is closed"));
        }
        let Some(policy) = state.checkpoint_claims.capacity else {
            return Ok(CheckpointCapacityClaimDecision::Skipped(
                CheckpointRetentionSkipReason::Disabled,
            ));
        };
        let evaluation = evaluate_demand(&state, demand)?;
        if !evaluation.permanent.is_empty() {
            return Ok(CheckpointCapacityClaimDecision::PermanentRejected(
                AdmissionRejected {
                    immediate_requested: demand.immediate_claim.clone(),
                    fit_requested: demand.fit_requirement.clone(),
                    maximum: state.snapshot(self.id()),
                    blockers: evaluation.permanent,
                },
            ));
        }
        if !evaluation.blockers.is_empty() {
            let action = deferred_action(demand, evaluation.growth_required);
            let wait_condition =
                state.wait_condition_for_blockers(self.id(), &evaluation.blockers)?;
            let snapshot = state.snapshot(self.id());
            return Ok(CheckpointCapacityClaimDecision::Deferred(
                AdmissionDeferred {
                    immediate_requested: demand.immediate_claim.clone(),
                    fit_requested: demand.fit_requirement.clone(),
                    release_epoch: snapshot.release_epoch,
                    capacity_epoch: snapshot.capacity_epoch,
                    available: snapshot,
                    blockers: evaluation.blockers,
                    action,
                    wait_condition,
                },
            ));
        }

        let current_retained = state.checkpoint_claims.retained_bytes;
        let maximum_bytes = policy.maximum_retained_bytes();
        let Some(remaining) = maximum_bytes.checked_sub(current_retained) else {
            state.poisoned = true;
            self.inner.epoch_tx.send_replace(state.epochs(self.id()));
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "checkpoint retained capacity exceeds its immutable policy",
            ));
        };
        if retained_bytes > remaining {
            return Ok(CheckpointCapacityClaimDecision::Skipped(
                CheckpointRetentionSkipReason::Capacity {
                    requested_bytes: retained_bytes,
                    retained_bytes: current_retained,
                    maximum_bytes,
                },
            ));
        }
        let next_retained = current_retained
            .checked_add(retained_bytes)
            .ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::ArithmeticOverflow,
                    "checkpoint retained byte usage overflows u64",
                )
            })?;
        let serial = state.checkpoint_claims.next_serial;
        let next_serial = serial
            .checked_add(1)
            .filter(|_| serial != 0)
            .ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::AuthorityExhausted,
                    "checkpoint authority serial is exhausted",
                )
            })?;
        let authority = CheckpointAuthorityId {
            coordinator_id: self.id(),
            serial,
        };
        if state.checkpoint_claims.live.contains_key(&authority) {
            return Err(invalid_admission("checkpoint authority is already live"));
        }
        state
            .release_epoch
            .checked_add(u64::from(state.active_requests))
            .and_then(|epoch| epoch.checked_add(u64::from(state.active_sequences)))
            .and_then(|epoch| epoch.checked_add(state.active_child_claims))
            .and_then(|epoch| epoch.checked_add(state.checkpoint_claims.count()))
            .and_then(|epoch| epoch.checked_add(1))
            .ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::EpochExhausted,
                    "release epoch cannot represent every outstanding lease release",
                )
            })?;
        let mut next_usage = Vec::with_capacity(demand.immediate_claim.entries().len());
        for entry in demand.immediate_claim.entries() {
            let domain = state
                .domains
                .get(&entry.domain)
                .expect("checkpoint demand domains were validated");
            let used = domain.used.checked_add(entry.units.get()).ok_or_else(|| {
                admission_fault(
                    DynamicAdmissionFaultKind::ArithmeticOverflow,
                    "checkpoint capacity usage overflows u64",
                )
            })?;
            if domain.availability_epoch == u64::MAX {
                return Err(admission_fault(
                    DynamicAdmissionFaultKind::EpochExhausted,
                    "checkpoint domain cannot publish its eventual release",
                ));
            }
            next_usage.push((entry.domain, used));
        }
        let claims = demand.immediate_claim.clone();
        // Prepare both owned vectors before the first ledger mutation.
        let recorded_claims = CheckpointClaimRecord {
            claims: claims.clone(),
            retained_bytes,
        };
        state
            .checkpoint_claims
            .live
            .insert(authority, recorded_claims);
        state.checkpoint_claims.next_serial = next_serial;
        state.checkpoint_claims.retained_bytes = next_retained;
        for (domain, used) in next_usage {
            state
                .domains
                .get_mut(&domain)
                .expect("validated domain")
                .used = used;
        }
        Ok(CheckpointCapacityClaimDecision::Claimed(
            LogicalCheckpointLease {
                inner: Arc::clone(&self.inner),
                authority,
                claims,
                retained_bytes,
                released: false,
            },
        ))
    }

    pub(crate) fn owns_checkpoint_claim(&self, lease: &LogicalCheckpointLease) -> bool {
        !lease.released
            && lease.authority.coordinator_id == self.id()
            && Arc::ptr_eq(&self.inner, &lease.inner)
    }

    /// Prevents new checkpoint claims without invalidating outstanding owners.
    /// Resource-layer plan shutdown must call this under its lifecycle gate.
    pub(crate) fn close_checkpoint_admission(&self) -> Result<(), VNextError> {
        let mut state = self.inner.lock_mutation()?;
        if state.poisoned {
            return Err(admission_fault(
                DynamicAdmissionFaultKind::Poisoned,
                "coordinator is fail-closed",
            ));
        }
        state.checkpoint_claims.closed = true;
        Ok(())
    }
}

impl LogicalCheckpointLease {
    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.inner.id
    }

    pub const fn authority(&self) -> CheckpointAuthorityId {
        self.authority
    }

    pub fn claims(&self) -> &CapacityVector {
        &self.claims
    }

    /// Independently retained aligned extents, not another device allocation.
    pub const fn retained_bytes(&self) -> u64 {
        self.retained_bytes
    }

    fn release_inner(&mut self) -> bool {
        if self.released {
            return true;
        }
        let mut state = match self.inner.lock_mutation() {
            Ok(state) => state,
            Err(_) => return false,
        };
        let valid = !state.poisoned
            && self.authority.coordinator_id == self.inner.id
            && state
                .checkpoint_claims
                .live
                .get(&self.authority)
                .is_some_and(|record| {
                    record.claims == self.claims && record.retained_bytes == self.retained_bytes
                })
            && self.retained_bytes > 0
            && self
                .claims
                .entries()
                .iter()
                .try_fold(0_u64, |sum, claim| sum.checked_add(claim.units.get()))
                == Some(self.retained_bytes)
            && state.checkpoint_claims.retained_bytes >= self.retained_bytes
            && state.checkpoint_claims.capacity.is_some_and(|policy| {
                state.checkpoint_claims.retained_bytes <= policy.maximum_retained_bytes()
            })
            && self.claims.entries().iter().all(|claim| {
                state.domains.get(&claim.domain).is_some_and(|domain| {
                    domain.used >= claim.units.get() && domain.availability_epoch < u64::MAX
                })
            });
        let next_release_epoch = state.release_epoch.checked_add(1);
        if !valid || next_release_epoch.is_none() {
            state.poisoned = true;
            self.inner
                .epoch_tx
                .send_replace(state.epochs(self.inner.id));
            return false;
        }
        for claim in self.claims.entries() {
            let domain = state
                .domains
                .get_mut(&claim.domain)
                .expect("validated domain");
            domain.used -= claim.units.get();
            domain.availability_epoch += 1;
        }
        state.checkpoint_claims.live.remove(&self.authority);
        state.checkpoint_claims.retained_bytes -= self.retained_bytes;
        state.release_epoch = next_release_epoch.expect("validated release epoch");
        // Logical availability is published here; this says nothing about
        // device residency or completion of physical backing release.
        self.inner
            .epoch_tx
            .send_replace(state.epochs(self.inner.id));
        drop(state);
        self.released = true;
        true
    }
}

impl Drop for LogicalCheckpointLease {
    fn drop(&mut self) {
        let _ = self.release_inner();
    }
}

#[cfg(test)]
mod tests;
