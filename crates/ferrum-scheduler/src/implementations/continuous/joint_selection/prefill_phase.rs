//! Atomic installation of a finite, positive prefill policy commitment.
use super::*;
use ferrum_execution::execution_attempt::AttemptWorkKind;

impl ContinuousBatchScheduler {
    pub(super) fn apply_joint_prefill_phase(
        &self,
        batch: &mut [ScheduledRequest],
        round: &mut TimePlanningRound,
        limit: NonZeroUsize,
        bindings: &[WaveBinding],
    ) -> bool {
        if bindings.len() != batch.len() || !round.cost_selection_is_exact(batch.len()) {
            return false;
        }
        let mut prefill = self.prefill_queue.write();
        let decode = self.decode_queue.read();
        let mut updates = Vec::with_capacity(batch.len());
        // Validate the complete selected wave before changing any request.
        for (index, ((scheduled, selected), binding)) in
            batch.iter().zip(&round.selected).zip(bindings).enumerate()
        {
            if binding.id != scheduled.request.id
                || binding.work != *selected
                || binding.offset != scheduled.tokens_processed
                || batch[..index]
                    .iter()
                    .any(|old| old.request.id == scheduled.request.id)
            {
                return false;
            }
            let Some(tokens) = scheduled.tokens_to_process else {
                return false;
            };
            let (request, is_decode) = match selected.kind {
                AttemptWorkKind::Prefill => {
                    let Some(request) = prefill
                        .iter()
                        .find(|request| request.inner.request.id == binding.id)
                    else {
                        return false;
                    };
                    if request.phase != RequestPhase::Prefilling
                        || request.prefill_chunk_offset != binding.offset
                    {
                        return false;
                    }
                    (request, false)
                }
                AttemptWorkKind::Decode => {
                    let Some(request) = decode.requests.get(&binding.id) else {
                        return false;
                    };
                    if request.phase != RequestPhase::Decoding
                        || request.total_tokens() != binding.offset
                        || tokens != 1
                    {
                        return false;
                    }
                    (request, true)
                }
            };
            if request.waiting_admission_ticket != Some(binding.ticket)
                || request.logical_work_frontier.progress_generation() != binding.generation
                || self.prompt_token_estimate(request) != Some(binding.prompt_tokens)
                || request.capacity_deferred_from_decode
                || request.prefix_restore.is_pending()
                || request.execution_capacity_deferral.is_some()
                || request.execution_readiness_block.is_some()
                || request.execution_maintenance_retry.is_some()
                || request.output_readiness.is_some()
                || request.capacity_deferred_until_release_epoch > 0
                || request.prefix_rendezvous.held()
                || self.cost_work(request, tokens, is_decode, round.prefill_shape_policy)
                    != Some((*selected, true))
            {
                return false;
            }
            if is_decode {
                continue;
            }
            if request
                .joint_prefill_phase_cap
                .is_some_and(|cap| cap.prompt_tokens != binding.prompt_tokens)
            {
                return false;
            }
            let cap = PrefillPhaseCap {
                limit: request
                    .joint_prefill_phase_cap
                    .map_or(limit, |old| old.limit.min(limit)),
                prompt_tokens: binding.prompt_tokens,
            };
            let Some((work, true)) = self.cost_work(
                request,
                tokens.min(cap.limit.get()),
                false,
                round.prefill_shape_policy,
            ) else {
                return false;
            };
            if work.requested.as_range().is_empty() || work.requested.as_range().len() > tokens {
                return false;
            }
            updates.push((
                index,
                work,
                cap,
                request.joint_prefill_phase_cap != Some(cap),
            ));
        }
        // A new commitment can change later waves even when this wave's budget
        // already made its chunk small. Reinstalling an identical cap is a no-op.
        if !updates.iter().any(|(_, _, _, changed)| *changed) {
            return false;
        }
        for (index, work, cap, _) in updates {
            let scheduled = &mut batch[index];
            let request = prefill
                .iter_mut()
                .find(|request| request.inner.request.id == scheduled.request.id)
                .expect("all bound members were validated under this write lock");
            let tokens = work.requested.as_range().len();
            if scheduled.tokens_to_process != Some(tokens) {
                round.statistics.shrunk_prefills += 1;
            }
            request.joint_prefill_phase_cap = Some(cap);
            request.logical_work_frontier.mark_scheduled(tokens);
            scheduled.tokens_to_process = Some(tokens);
            round.selected[index] = work;
        }
        true
    }
}
