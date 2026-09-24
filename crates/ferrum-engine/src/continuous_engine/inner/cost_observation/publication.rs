//! Recheck the host frontier after scheduler/output publication. No output or
//! physical authority is acquired, returned or reconstructed here.
use super::*;
use crate::continuous_engine::EngineInner;

impl EngineInner {
    pub(in crate::continuous_engine) fn publish_cost_host_commit(
        &self,
        cost: &mut Option<ObservedCostCall>,
        evidence: Option<HostCommitEvidence>,
        produced_token: bool,
        terminal_cleanup: bool,
    ) -> Option<PendingHostRow> {
        let Some(call) = cost.as_deref_mut() else {
            return None;
        };
        let Some(mut evidence) = evidence else {
            call.reject(CostCallRejection::HostMissing);
            return None;
        };
        if terminal_cleanup {
            // Final decode/release/terminal transport has no calibrated shape
            // in this single-wave model. Never copy it into sibling samples.
            call.reject(CostCallRejection::Composite);
        }
        let sequences = self.sequences.read();
        let Some(sequence) = sequences.get(&evidence.request_id) else {
            call.host_cancelled(&evidence.request_id);
            return None;
        };
        let matches_frontier = sequence.cost_frontier.is_some_and(|frontier| {
            frontier.owner_incarnation.get() == evidence.owner_incarnation
                && evidence.work_generation.checked_add(1) == Some(frontier.work_generation.get())
        });
        if !matches_frontier {
            call.reject(CostCallRejection::FrontierMismatch);
            return None;
        }
        if produced_token {
            if let Some(output) = &sequence.credited_output {
                if output.failure.is_some() || sequence.client_receiver_closed() {
                    call.host_failed(&evidence.request_id);
                    if !terminal_cleanup {
                        return None;
                    }
                }
                if output.grant.is_some() && !terminal_cleanup {
                    // A sibling/outer batch will finish this grant later.
                    call.reject(CostCallRejection::Composite);
                    return None;
                }
                if output.accepted_ordinal == 0 && !terminal_cleanup {
                    call.host_failed(&evidence.request_id);
                    return None;
                }
            } else if sequence.stream_sender.is_some() {
                // Legacy send waits on a client; no bounded consumer model is
                // asserted by the executor's preparation-to-commit shape.
                call.reject(CostCallRejection::Composite);
                return None;
            }
        }
        if terminal_cleanup && sequence.credited_output.is_none() {
            return None;
        }
        let pending = call.host_publication(
            &evidence,
            terminal_cleanup,
            produced_token && sequence.credited_output.is_some(),
        );
        evidence.committed_at_ns = call.now_ns();
        call.record_host_result(evidence);
        pending
    }
}
