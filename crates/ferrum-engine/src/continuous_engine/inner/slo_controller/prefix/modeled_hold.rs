//! Read-only recognition of the dependency owned by the current cohort.
//! This does not remove StateBlocked or create a restore/compute permission.
use super::*;

impl SloPrefixCohort {
    pub(in crate::continuous_engine::inner::slo_controller) fn modeled_hold_target<'a>(
        &self,
        queue: &'a PlanningQueueSnapshot,
        fences: &[EngineFence],
    ) -> Option<&'a PlanningRequestKey> {
        if self.restored || slo_clock_now() >= self.expires_at {
            return None;
        }
        matching_target(
            &self.hold,
            &self.target,
            self.source_incarnation,
            self.target_incarnation,
            queue,
            fences,
        )
    }
}

fn matching_target<'a>(
    hold: &PrefixRendezvousHold,
    target: &PrefixRequestKey,
    source_incarnation: u64,
    target_incarnation: u64,
    queue: &'a PlanningQueueSnapshot,
    fences: &[EngineFence],
) -> Option<&'a PlanningRequestKey> {
    let [follower] = hold.followers() else {
        return None;
    };
    if !hold.is_pending()
        || follower.request_id() != target.request_id()
        || follower.ordinal() != target.ordinal()
        || follower.work_generation() != target.work_generation()
    {
        return None;
    }
    let source = queue.requests().iter().find(|row| {
        row.key.request_id == *hold.source().request_id()
            && row.key.ticket.get() == hold.source().ordinal()
    })?;
    let target_row = queue.requests().iter().find(|row| {
        row.key.request_id == *target.request_id()
            && row.key.ticket.get() == target.ordinal()
            && row.key.generation == target.work_generation()
    })?;
    if !target_row.is_modeled_rendezvous_follower(source, hold) {
        return None;
    }
    // Full keys join the whole scheduler snapshot to this transaction's actual
    // engine owners. A same-ID replacement or an old/incomplete fence set
    // cannot classify a prefix gate as modeled.
    let source_fence = fences.iter().find(|fence| fence.key == source.key)?;
    let target_fence = fences.iter().find(|fence| fence.key == target_row.key)?;
    if source_fence.incarnation != source_incarnation
        || target_fence.incarnation != target_incarnation
        || source_fence.prefill_complete
        || target_fence.prefill_complete
        || source_fence.prefill_tokens_processed != source.prefill_offset
        || target_fence.prefill_tokens_processed != 0
        || target_fence.generated != 0
        || !hold.is_pending()
    {
        return None;
    }
    Some(&target_row.key)
}

#[cfg(test)]
mod tests;
