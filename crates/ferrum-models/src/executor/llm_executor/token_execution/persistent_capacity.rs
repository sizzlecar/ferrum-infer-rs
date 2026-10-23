//! Hold exact live registry bindings during a try-only original-owner query.

use super::*;
use crate::common::llm::{ModelCapacityResident, ModelPersistentCapacityQuery};
use ferrum_execution::model_executor::{
    PersistentCapacityQuery, PersistentCapacityState, PersistentCapacityUnknown as U,
    PersistentCohortCapacitySnapshot, MAX_PERSISTENT_CAPACITY_MEMBERS,
};

impl LlmExecutor {
    pub(in crate::executor::llm_executor) fn observe_persistent_cohort_capacity(
        &self,
        query: &PersistentCapacityQuery<'_>,
    ) -> std::result::Result<PersistentCohortCapacitySnapshot, U> {
        query.validate()?;
        let registry = self.token_sequences.try_lock().ok_or(U::Contended)?;
        if registry.closing {
            return Err(U::PendingOwner);
        }
        let resident_count = query
            .members
            .iter()
            .filter(|member| matches!(member.state, PersistentCapacityState::Resident { .. }))
            .count();
        if resident_count != registry.live.len() {
            return Err(U::MembershipChanged);
        }
        let mut budget = query.inspection_budget;
        let scratch = resident_count
            .checked_mul(std::mem::size_of::<ModelCapacityResident<'_>>())
            .and_then(|n| {
                n.checked_add(MAX_PERSISTENT_CAPACITY_MEMBERS * std::mem::size_of::<usize>())
            })
            .and_then(|n| n.checked_add(query.retained_bytes().ok()?))
            .ok_or(U::Arithmetic)?;
        budget.max_scratch_bytes = budget
            .max_scratch_bytes
            .checked_sub(scratch)
            .ok_or(U::InspectionBudget)?;
        let mut residents = Vec::new();
        residents
            .try_reserve_exact(resident_count)
            .map_err(|_| U::InspectionBudget)?;
        let mut ceilings = [0usize; MAX_PERSISTENT_CAPACITY_MEMBERS];
        for (index, member) in query.members.iter().enumerate() {
            ceilings[index] = member.through_tokens;
            match &member.state {
                PersistentCapacityState::Waiting => {
                    if registry
                        .live
                        .values()
                        .any(|entry| entry.request_id == member.request_id)
                    {
                        return Err(U::MembershipChanged);
                    }
                }
                PersistentCapacityState::Resident { sequence } => {
                    self.token_issuer
                        .frontier(sequence, 0)
                        .map_err(|_| U::MembershipChanged)?;
                    let entry = registry.live.get(sequence).ok_or(U::MembershipChanged)?;
                    if entry.request_id != member.request_id
                        || member.through_tokens != entry.maximum_tokens
                    {
                        return Err(U::MembershipChanged);
                    }
                    if entry.closing
                        || sequence.observed_end().is_some()
                        || entry
                            .device
                            .as_ref()
                            .is_some_and(|state| !state.persistent_capacity_ready())
                    {
                        return Err(U::PendingOwner);
                    }
                    residents.push(ModelCapacityResident {
                        cache_id: &entry.cache_id,
                        consumed_tokens: entry.frontier.consumed_tokens(),
                        through_tokens: member.through_tokens,
                    });
                }
            }
        }
        // Recovery takes model then registry. Never wait in the inverse order:
        // try-lock either observes one coherent owner set or fails closed.
        let model = self.model.try_lock().ok_or(U::Contended)?;
        let model = model.as_ref().ok_or(U::PendingOwner)?;
        model.persistent_capacity_for_cohort(&ModelPersistentCapacityQuery {
            residents: &residents,
            through_tokens: &ceilings[..query.members.len()],
            max_live_sequences: query.max_live_sequences,
            max_wave_sequences: query.max_wave_sequences,
            inspection_budget: budget,
        })?;
        PersistentCohortCapacitySnapshot::from_verified_query(query)
    }
}

#[cfg(test)]
mod tests;
