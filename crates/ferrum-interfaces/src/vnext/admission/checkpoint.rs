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
    Deferred(AdmissionDeferred),
    PermanentRejected(AdmissionRejected),
}

#[derive(Debug)]
pub(super) struct CheckpointClaimLedger {
    next_serial: u64,
    live: BTreeMap<CheckpointAuthorityId, CapacityVector>,
    closed: bool,
}

impl CheckpointClaimLedger {
    pub(super) fn new() -> Self {
        Self {
            next_serial: 1,
            live: BTreeMap::new(),
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
    released: bool,
}

impl LogicalAdmissionCoordinator {
    /// Claims plan-derived checkpoint demand in the existing domain ledger.
    /// The caller must hold the plan lifecycle guard and commit this together
    /// with prepared physical backing. This is not a product allocation API.
    pub(crate) fn try_claim_checkpoint(
        &self,
        demand: &AdmissionDemand,
    ) -> Result<CheckpointCapacityClaimDecision, VNextError> {
        if demand.immediate_claim.is_empty() {
            return Err(invalid_admission(
                "checkpoint claim requires non-empty demand",
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
        let recorded_claims = claims.clone();
        state
            .checkpoint_claims
            .live
            .insert(authority, recorded_claims);
        state.checkpoint_claims.next_serial = next_serial;
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
            && state.checkpoint_claims.live.get(&self.authority) == Some(&self.claims)
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
