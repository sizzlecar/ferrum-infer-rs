//! Exact bounded cohort evidence shared by native and model-owned allocators.

#[cfg(test)]
mod tests;

use super::*;
pub use ferrum_interfaces::request_observation::RequestLifecycleId;
pub use ferrum_runtime::{PersistentCapacityInspectionBudget, PersistentCapacityUnknown};

pub const MAX_PERSISTENT_CAPACITY_MEMBERS: usize = 64;
pub const MAX_PERSISTENT_CAPACITY_RETAINED_BYTES: usize = 16 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistentCapacityState {
    Waiting,
    Resident {
        sequence: crate::token_execution::ExecutorSequence,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentCapacityMember {
    pub request_id: RequestId,
    pub lifecycle: RequestLifecycleId,
    pub state: PersistentCapacityState,
    /// The actual execution ceiling, including replay, not prompt + cap when
    /// the final sampled token is never consumed by another invocation.
    pub through_tokens: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct PersistentCapacityQuery<'a> {
    pub members: &'a [PersistentCapacityMember],
    pub max_live_sequences: usize,
    pub max_wave_sequences: usize,
    pub inspection_budget: PersistentCapacityInspectionBudget,
}

impl PersistentCapacityQuery<'_> {
    pub fn validate(&self) -> std::result::Result<(), PersistentCapacityUnknown> {
        self.inspection_budget.validate()?;
        if self.members.is_empty()
            || self.members.len() > MAX_PERSISTENT_CAPACITY_MEMBERS
            || self.max_live_sequences == 0
            || self.max_live_sequences > MAX_PERSISTENT_CAPACITY_MEMBERS
            || self.max_wave_sequences == 0
            || self.max_wave_sequences > self.max_live_sequences
        {
            return Err(PersistentCapacityUnknown::InvalidQuery);
        }
        let mut total = 0_u64;
        let mut residents = 0;
        for (index, member) in self.members.iter().enumerate() {
            if member.through_tokens == 0
                || member.lifecycle.producer == 0
                || member.lifecycle.admission == 0
                || self.members[..index].iter().any(|prior| {
                    prior.request_id == member.request_id
                        || prior.lifecycle == member.lifecycle
                        || matches!((&prior.state, &member.state),
                            (PersistentCapacityState::Resident { sequence: a },
                             PersistentCapacityState::Resident { sequence: b }) if a == b)
                })
            {
                return Err(PersistentCapacityUnknown::InvalidQuery);
            }
            total = total
                .checked_add(
                    u64::try_from(member.through_tokens)
                        .map_err(|_| PersistentCapacityUnknown::Arithmetic)?,
                )
                .ok_or(PersistentCapacityUnknown::Arithmetic)?;
            residents += usize::from(matches!(
                member.state,
                PersistentCapacityState::Resident { .. }
            ));
        }
        if residents > self.max_live_sequences {
            return Err(PersistentCapacityUnknown::InvalidQuery);
        }
        self.retained_bytes()?;
        Ok(())
    }

    pub fn retained_bytes(&self) -> std::result::Result<usize, PersistentCapacityUnknown> {
        std::mem::size_of::<PersistentCapacityMember>()
            .checked_mul(self.members.len())
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<PersistentCohortCapacitySnapshot>())
            })
            .filter(|bytes| *bytes <= MAX_PERSISTENT_CAPACITY_RETAINED_BYTES)
            .ok_or(PersistentCapacityUnknown::InspectionBudget)
    }
}

/// Original-owner support for the exact supplied cohort at bounded concurrency
/// using already allocated persistent backing. No future growth, transient
/// workspace, release time, reservation or successful execution is promised.
/// Discard on any membership, incarnation, ceiling or resource-state change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentCohortCapacitySnapshot {
    members: Arc<[PersistentCapacityMember]>,
    max_live_sequences: usize,
    max_wave_sequences: usize,
}

impl PersistentCohortCapacitySnapshot {
    /// Called by an original resource owner only after its bounded observation
    /// proves every pool and persistent metadata requirement for this query.
    /// Validation binds the observation; it does not itself prove physical fit.
    pub fn from_verified_query(
        query: &PersistentCapacityQuery<'_>,
    ) -> std::result::Result<Self, PersistentCapacityUnknown> {
        query.validate()?;
        let mut members = Vec::new();
        members
            .try_reserve_exact(query.members.len())
            .map_err(|_| PersistentCapacityUnknown::InspectionBudget)?;
        members.extend_from_slice(query.members);
        Ok(Self {
            members: members.into(),
            max_live_sequences: query.max_live_sequences,
            max_wave_sequences: query.max_wave_sequences,
        })
    }

    pub fn members(&self) -> &[PersistentCapacityMember] {
        &self.members
    }
    pub const fn max_live_sequences(&self) -> usize {
        self.max_live_sequences
    }
    pub const fn max_wave_sequences(&self) -> usize {
        self.max_wave_sequences
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.members.len() * std::mem::size_of::<PersistentCapacityMember>()
    }
}
