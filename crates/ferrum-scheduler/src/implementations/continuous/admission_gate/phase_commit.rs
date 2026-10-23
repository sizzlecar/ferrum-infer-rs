//! A bounded policy commitment before admission, with no executor authority.
use super::super::joint_selection::PrefillCapProposal;
use super::*;

impl ContinuousBatchScheduler {
    pub(super) fn commit_joint_admission_phase(
        &self,
        members: &[Member],
        proposal: &PrefillCapProposal,
        apply_before: Instant,
    ) -> bool {
        if proposal.targets.is_empty() || proposal.targets.len() > members.len() {
            return false;
        }
        let deadline = apply_before.min(proposal.valid_until);
        let mut waiting = self.waiting_queue.write();
        let mut prefill = self.prefill_queue.write();
        let decode = self.decode_queue.read();
        if Instant::now() >= deadline
            || waiting.scan_active()
            || waiting.has_capacity_deferrals()
            || members.len() != waiting.len() + prefill.len() + decode.requests.len()
        {
            return false;
        }
        // Check the whole comparison population, not just capped members.
        // In particular a scheduled-but-uncommitted wave can change without
        // incrementing the logical generation.
        for request in prefill
            .iter()
            .chain(decode.requests.values())
            .chain(waiting.iter())
        {
            if !members
                .iter()
                .any(|member| member.matches_exact(self, request))
                || request.capacity_deferred_from_decode
                || request.prefix_restore.is_pending()
                || request.execution_capacity_deferral.is_some()
                || request.execution_readiness_block.is_some()
                || request.execution_maintenance_retry.is_some()
                || request.output_readiness.is_some()
                || request.capacity_deferred_until_release_epoch > 0
                || request.prefix_rendezvous.held()
            {
                return false;
            }
        }
        let mut previous = None;
        for request in waiting.iter() {
            let Some(index) = members
                .iter()
                .position(|member| member.matches_exact(self, request))
            else {
                return false;
            };
            if previous.is_some_and(|old| old >= index) {
                return false;
            }
            previous = Some(index);
        }
        let mut changes = false;
        for (index, target) in proposal.targets.iter().enumerate() {
            if proposal.targets[..index]
                .iter()
                .any(|old| old.key == target.key)
                || members
                    .get(target.request)
                    .is_none_or(|member| member.key != target.key)
            {
                return false;
            }
            let request = match target.phase {
                RequestPhase::Waiting => waiting.iter().find(|request| target.key.matches(request)),
                RequestPhase::Prefilling => {
                    prefill.iter().find(|request| target.key.matches(request))
                }
                _ => return false,
            };
            let Some(request) = request else {
                return false;
            };
            if request.phase != target.phase
                || request.logical_work_frontier.progress_generation() != target.generation
                || request.prefill_chunk_offset != target.prefilled_tokens
                || self.prompt_token_estimate(request) != Some(target.prompt_tokens)
                || request.joint_prefill_phase_cap != target.prior_cap
                || target.prefilled_tokens >= target.prompt_tokens
                || target
                    .prior_cap
                    .is_some_and(|cap| cap.prompt_tokens != target.prompt_tokens)
            {
                return false;
            }
            match target.phase {
                RequestPhase::Waiting => {
                    if request.logical_work_frontier.work_kind()
                        != pressure::LogicalWorkKind::Waiting
                        || request.logical_work_frontier != pressure::LogicalWorkFrontier::default()
                        || target.generation != LogicalWorkGeneration::ZERO
                        || target.prefilled_tokens != 0
                        || request.decode_tokens != 0
                    {
                        return false;
                    }
                }
                RequestPhase::Prefilling => {
                    if !request
                        .logical_work_frontier
                        .can_restore_at(target.prefilled_tokens)
                    {
                        return false;
                    }
                }
                _ => unreachable!("target residency was checked above"),
            }
            let cap = PrefillPhaseCap::selected(
                target.prior_cap,
                proposal.limit,
                target.prompt_tokens,
                proposal.packing_mode,
            );
            changes |= target.prior_cap != Some(cap);
        }
        if !changes || Instant::now() >= deadline {
            return false;
        }
        // No fallible operation follows this point. Each target was validated
        // under all three locks, so no prefix of the proposal can be installed.
        for target in &proposal.targets {
            let request = match target.phase {
                RequestPhase::Waiting => waiting.request_mut(target.key.ticket()),
                RequestPhase::Prefilling => prefill
                    .iter_mut()
                    .find(|request| target.key.matches(request)),
                _ => unreachable!("only waiting/prefilling targets passed validation"),
            }
            .expect("exact target remains owned under the same locks");
            request.joint_prefill_phase_cap = Some(PrefillPhaseCap::selected(
                target.prior_cap,
                proposal.limit,
                target.prompt_tokens,
                proposal.packing_mode,
            ));
        }
        true
    }
}

#[cfg(test)]
mod tests;
