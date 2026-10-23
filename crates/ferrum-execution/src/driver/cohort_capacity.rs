//! Point-in-time capacity evidence from the original native plan/session owners.

use super::*;
use crate::model_executor::{
    PersistentCapacityQuery, PersistentCapacityState, PersistentCapacityUnknown,
    PersistentCohortCapacitySnapshot,
};

impl<R: DeviceRuntime> ProgramExecutor<R> {
    pub(super) fn native_persistent_cohort_capacity(
        &self,
        query: &PersistentCapacityQuery<'_>,
    ) -> std::result::Result<PersistentCohortCapacitySnapshot, PersistentCapacityUnknown> {
        use PersistentCapacityUnknown as U;
        query.validate()?;
        if query.max_live_sequences != self.policy.memory().maximum_active_sequences as usize
            || query.max_wave_sequences as u64 > self.policy.admission().maximum_scheduled_tokens
            || query
                .members
                .iter()
                .any(|m| m.through_tokens > self.maximum_model_tokens)
        {
            return Err(U::InvalidQuery);
        }
        // A future checkpoint can retain pages after the logical request ends.
        // This initial recycling proof has no bound for those independent owners.
        if self.prefix_restore_enabled() || self.checkpoint_capture.is_some() {
            return Err(U::Unsupported);
        }
        let per_member = std::mem::size_of::<Arc<VNextSequence<R>>>()
            + std::mem::size_of::<u64>()
            + std::mem::size_of::<tokio::sync::MutexGuard<'_, ()>>()
            + std::mem::size_of::<parking_lot::MutexGuard<'_, SequenceSettlement<R>>>()
            + std::mem::size_of::<(Arc<SequenceSession<R>>, u64)>();
        let scratch = query
            .members
            .len()
            .checked_mul(per_member)
            .ok_or(U::Arithmetic)?;
        let mut budget = query.inspection_budget;
        budget.max_scratch_bytes = budget
            .max_scratch_bytes
            .checked_sub(scratch)
            .filter(|n| *n > 0)
            .ok_or(U::InspectionBudget)?;
        let mut sequences = Vec::with_capacity(query.members.len());
        let mut ceilings = Vec::with_capacity(query.members.len());
        {
            let registry = self.sequences.try_lock().ok_or(U::Contended)?;
            for member in query.members {
                ceilings.push(u64::try_from(member.through_tokens).map_err(|_| U::Arithmetic)?);
                if let PersistentCapacityState::Resident { sequence: identity } = &member.state {
                    self.token_issuer
                        .frontier(identity, 0)
                        .map_err(|_| U::MembershipChanged)?;
                    let sequence = registry
                        .token_index
                        .get(identity)
                        .and_then(Weak::upgrade)
                        .ok_or(U::MembershipChanged)?;
                    if sequence.request.product_request_id() != &member.request_id
                        || member.through_tokens != sequence.maximum_tokens
                    {
                        return Err(U::MembershipChanged);
                    }
                    sequences.push(sequence);
                }
            }
        }
        let _operations = sequences
            .iter()
            .map(|sequence| sequence.operation.try_lock().ok())
            .collect::<Option<Vec<_>>>()
            .ok_or(U::Contended)?;
        let mut settlements = Vec::with_capacity(sequences.len());
        for sequence in &sequences {
            let settlement = sequence.settlement.try_lock().ok_or(U::Contended)?;
            if !sequence.active.load(Ordering::Acquire) || !settlement.token_frontier_is_open() {
                return Err(U::PendingOwner);
            }
            settlements.push(settlement);
        }
        let residents = sequences
            .iter()
            .map(|sequence| {
                (
                    Arc::clone(&sequence.session),
                    sequence.maximum_tokens as u64,
                )
            })
            .collect::<Vec<_>>();
        self.plan_resources.persistent_capacity_for_cohort(
            &residents,
            &ceilings,
            query.max_live_sequences,
            budget,
        )?;
        PersistentCohortCapacitySnapshot::from_verified_query(query)
    }
}
