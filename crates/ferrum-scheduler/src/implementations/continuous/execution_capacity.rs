//! Cold observations for the scheduler's single execution-pressure policy.

use super::{AdmissionDeferral, PressureWaitCondition};
use ferrum_runtime::{CapacityAvailabilitySource, CapacityWaitCondition};
use ferrum_types::{AllocatorCapacityDomain, FerrumError, RequestId, Result};
use std::collections::{HashMap, HashSet};

/// A retained retry predicate, not a reservation or a second allocator ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionCapacityDeferral {
    Native(AdmissionDeferral),
    Allocator {
        domain: AllocatorCapacityDomain,
        release_epoch: u64,
    },
}

impl ExecutionCapacityDeferral {
    pub fn as_native(&self) -> Option<&AdmissionDeferral> {
        match self {
            Self::Native(deferral) => Some(deferral),
            Self::Allocator { .. } => None,
        }
    }

    pub fn wait_condition(&self) -> PressureWaitCondition {
        match self {
            Self::Native(deferral) => deferral.wait_condition().clone().into(),
            Self::Allocator {
                domain,
                release_epoch,
            } => PressureWaitCondition::allocator(domain.clone(), *release_epoch),
        }
    }
}

impl From<AdmissionDeferral> for ExecutionCapacityDeferral {
    fn from(deferral: AdmissionDeferral) -> Self {
        Self::Native(deferral)
    }
}

/// Engine observations of exact physical owners, collected only after an
/// authoritative capacity rejection. No logical phase proves releasability.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExecutionCapacityReleaseSnapshot {
    native_sources_by_request: HashMap<RequestId, Vec<CapacityAvailabilitySource>>,
    allocator: Option<AllocatorReleaseObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AllocatorReleaseObservation {
    domain: AllocatorCapacityDomain,
    blocked_request_id: RequestId,
    contributors: HashSet<RequestId>,
    pending_release: bool,
}

impl ExecutionCapacityReleaseSnapshot {
    pub fn new(
        capabilities: impl IntoIterator<Item = (RequestId, Vec<CapacityAvailabilitySource>)>,
    ) -> Self {
        let mut native_sources_by_request = HashMap::new();
        for (request_id, mut sources) in capabilities {
            sources.sort_unstable();
            sources.dedup();
            if !sources.is_empty() {
                native_sources_by_request.insert(request_id, sources);
            }
        }
        Self {
            native_sources_by_request,
            allocator: None,
        }
    }

    /// The actual owner must first prove that current free blocks plus the
    /// external cohort's joint reclaim can satisfy the rejected allocation.
    /// Each contributor removes a reference in that reclaimable closure;
    /// individual contributors may free zero blocks until peers also end.
    /// `pending_release` refers only to contributing same-pool closes whose
    /// terminal observation is independently retained by the engine.
    pub fn with_allocator(
        mut self,
        domain: AllocatorCapacityDomain,
        blocked_request_id: RequestId,
        releasers: Vec<RequestId>,
        pending_release: bool,
    ) -> Result<Self> {
        let contributors = releasers.iter().cloned().collect::<HashSet<_>>();
        if self.allocator.is_some()
            || contributors.len() != releasers.len()
            || contributors.contains(&blocked_request_id)
        {
            return Err(FerrumError::scheduler("allocator release observation requires one exact query and unique external contributors"));
        }
        self.allocator = Some(AllocatorReleaseObservation {
            domain,
            blocked_request_id,
            contributors,
            pending_release,
        });
        Ok(self)
    }

    pub(super) fn allocator_pending_release(
        &self,
        request_id: &RequestId,
        domain: &AllocatorCapacityDomain,
    ) -> Result<bool> {
        let observation = self.allocator.as_ref().ok_or_else(|| {
            FerrumError::scheduler("allocator pressure requires an exact release observation")
        })?;
        if &observation.domain != domain || &observation.blocked_request_id != request_id {
            return Err(FerrumError::scheduler(
                "allocator release observation belongs to another pool or blocked request",
            ));
        }
        Ok(observation.pending_release)
    }

    pub(super) fn can_advance_pressure(
        &self,
        request_id: &RequestId,
        condition: &PressureWaitCondition,
    ) -> bool {
        match condition {
            PressureWaitCondition::Native(condition) => self.can_advance(request_id, condition),
            PressureWaitCondition::Allocator { domain, .. } => {
                self.allocator.as_ref().is_some_and(|observation| {
                    &observation.domain == domain && observation.contributors.contains(request_id)
                })
            }
        }
    }

    pub(super) fn can_advance(
        &self,
        request_id: &RequestId,
        condition: &CapacityWaitCondition,
    ) -> bool {
        self.native_sources_by_request
            .get(request_id)
            .is_some_and(|sources| {
                condition
                    .observed()
                    .iter()
                    .any(|observed| sources.binary_search(&observed.source()).is_ok())
            })
    }

    pub fn has_external_releaser(
        &self,
        blocked_request_id: &RequestId,
        condition: &CapacityWaitCondition,
    ) -> bool {
        self.native_sources_by_request.keys().any(|request_id| {
            request_id != blocked_request_id && self.can_advance(request_id, condition)
        })
    }
}
