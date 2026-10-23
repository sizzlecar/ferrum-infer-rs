//! Exact cohort binding for an original owner's persistent capacity observation.

use super::TimePlanningRound;
use ferrum_execution::model_executor::{PersistentCapacityQuery, PersistentCohortCapacitySnapshot};

impl TimePlanningRound {
    /// Attach only a complete observation for the current request lifetimes.
    /// This grants no admission authority. Refreshing facts or capacity clears
    /// it, and selection additionally checks each request's execution ceiling.
    pub fn set_persistent_capacity(
        &mut self,
        query: &PersistentCapacityQuery<'_>,
        proof: Option<PersistentCohortCapacitySnapshot>,
    ) {
        self.admission_gate = None;
        self.persistent_capacity = proof.filter(|proof| {
            if query.validate().is_err()
                || proof.members() != query.members
                || proof.max_live_sequences() != query.max_live_sequences
                || proof.max_wave_sequences() != query.max_wave_sequences
            {
                return false;
            }
            let Some(facts) = self.request_facts.as_ref() else {
                return false;
            };
            let Some(clocks) = self.request_clocks.as_ref() else {
                return false;
            };
            let bytes = proof
                .retained_bytes()
                .checked_add(self.request_fact_bytes)
                .and_then(|bytes| {
                    clocks
                        .len()
                        .checked_mul(std::mem::size_of::<(
                            ferrum_types::RequestId,
                            super::RequestClockFacts,
                        )>())
                        .and_then(|clocks| bytes.checked_add(clocks))
                });
            !facts.is_empty()
                && proof.members().len() == facts.len()
                && bytes.is_some_and(|bytes| bytes <= self.config.planning.max_request_fact_bytes)
                && proof
                    .members()
                    .iter()
                    .zip(facts)
                    .all(|(member, (id, fact))| {
                        member.request_id == *id && member.lifecycle == fact.lifecycle
                    })
        });
    }

    pub fn persistent_capacity(&self) -> Option<&PersistentCohortCapacitySnapshot> {
        self.persistent_capacity.as_ref()
    }
}
