//! Exact capacity observations used by the single pressure coordinator.

use ferrum_runtime::{
    CapacityAvailabilitySource, CapacityWaitCondition, LogicalAdmissionCoordinatorId,
};
use ferrum_types::AllocatorCapacityDomain;
use serde::Serialize;
use std::collections::BTreeMap;

/// An observation that may justify re-probing an actual allocator. Neither
/// variant reserves capacity or proves that a request now fits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum PressureWaitCondition {
    Native(CapacityWaitCondition),
    Allocator {
        domain: AllocatorCapacityDomain,
        /// The scheduler's existing confirmed physical-release epoch.
        release_epoch: u64,
    },
}

impl PressureWaitCondition {
    pub fn native(condition: CapacityWaitCondition) -> Self {
        Self::Native(condition)
    }

    pub fn allocator(domain: AllocatorCapacityDomain, release_epoch: u64) -> Self {
        Self::Allocator {
            domain,
            release_epoch,
        }
    }

    pub fn as_native(&self) -> Option<&CapacityWaitCondition> {
        match self {
            Self::Native(condition) => Some(condition),
            Self::Allocator { .. } => None,
        }
    }

    pub fn as_allocator(&self) -> Option<(&AllocatorCapacityDomain, u64)> {
        match self {
            Self::Allocator {
                domain,
                release_epoch,
            } => Some((domain, *release_epoch)),
            Self::Native(_) => None,
        }
    }

    pub(super) fn wait_domain(&self) -> PressureWaitDomain {
        match self {
            Self::Native(condition) => PressureWaitDomain::Native(condition.coordinator_id()),
            Self::Allocator { domain, .. } => PressureWaitDomain::Allocator(domain.clone()),
        }
    }
}

impl From<CapacityWaitCondition> for PressureWaitCondition {
    fn from(condition: CapacityWaitCondition) -> Self {
        Self::Native(condition)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum PressureWaitDomain {
    Native(LogicalAdmissionCoordinatorId),
    Allocator(AllocatorCapacityDomain),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum PressureWaitSource {
    Native(CapacityAvailabilitySource),
    AllocatorBlocks,
}

/// Device budget changes remain retry observations, but request release cannot
/// shrink plan-owned device backing. Preserve that native causality rule.
pub(super) fn is_request_release_source(source: PressureWaitSource) -> bool {
    matches!(
        source,
        PressureWaitSource::Native(CapacityAvailabilitySource::Domain(_))
            | PressureWaitSource::Native(CapacityAvailabilitySource::ActiveSequenceSlots)
            | PressureWaitSource::AllocatorBlocks
    )
}

pub(super) fn observed_map(condition: &PressureWaitCondition) -> BTreeMap<PressureWaitSource, u64> {
    match condition {
        PressureWaitCondition::Native(condition) => condition
            .observed()
            .iter()
            .map(|entry| (PressureWaitSource::Native(entry.source()), entry.epoch()))
            .collect(),
        PressureWaitCondition::Allocator { release_epoch, .. } => {
            BTreeMap::from([(PressureWaitSource::AllocatorBlocks, *release_epoch)])
        }
    }
}

pub(super) fn waits_overlap(left: &PressureWaitCondition, right: &PressureWaitCondition) -> bool {
    if left.wait_domain() != right.wait_domain() {
        return false;
    }
    let right = observed_map(right);
    observed_map(left)
        .keys()
        .any(|source| is_request_release_source(*source) && right.contains_key(source))
}

pub(super) fn same_source_topology(
    left: &PressureWaitCondition,
    right: &PressureWaitCondition,
) -> bool {
    left.wait_domain() == right.wait_domain()
        && observed_map(left).keys().eq(observed_map(right).keys())
}

pub(super) fn observed_advanced(
    prior: &BTreeMap<PressureWaitSource, u64>,
    current: &BTreeMap<PressureWaitSource, u64>,
) -> bool {
    prior.iter().any(|(source, epoch)| {
        current
            .get(source)
            .is_some_and(|current_epoch| current_epoch > epoch)
    })
}
