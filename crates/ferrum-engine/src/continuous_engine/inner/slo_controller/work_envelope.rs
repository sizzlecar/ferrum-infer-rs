//! Same product limits for joint planning, completion proposals and publication.
use super::*;
use crate::continuous_engine::output_flow_runtime::OutputPlanningCreditView;
use ferrum_scheduler::implementations::continuous::work_policy::{
    PlanningWorkPolicy, WaveWorkEnvelope, WaveWorkUsage,
};

impl EngineInner {
    pub(super) fn controller_work_policy(
        &self,
        hint: &ferrum_interfaces::BatchHint,
        queue: &PlanningQueueSnapshot,
    ) -> PlanningWorkPolicy {
        self.scheduler.planning_work_policy(
            hint,
            self.config.batching.prefill_decode_execution
                == ferrum_types::PrefillDecodeExecution::Mixed
                && self.model_executor.capabilities().supports_dynamic_batching,
            queue.requests().iter().any(|row| {
                row.queue == PlanningQueueKind::Decode && row.readiness.capacity_blocked
            }),
        )
    }

    pub(super) fn completion_work_envelope(
        &self,
        hint: &ferrum_interfaces::BatchHint,
        queue: &PlanningQueueSnapshot,
        sequences: &HashMap<RequestId, SequenceState>,
        budget: &ControllerBudget,
    ) -> Option<WaveWorkEnvelope> {
        let mut decoders = 0;
        for row in queue.requests() {
            if !budget.poll() {
                return None;
            }
            if row.queue != PlanningQueueKind::Decode || !row.readiness.ready() {
                continue;
            }
            let sequence = sequences.get(&row.key.request_id)?;
            if !sequence.prefill_complete
                || sequence.generated_tokens.len() >= sequence.sampling_params.max_tokens
            {
                continue;
            }
            let Some(output) = &sequence.credited_output else {
                continue;
            };
            let view = output.port.planning_snapshot();
            if output.failure.is_none()
                && !output.port.consumer_closed()
                && output.grant.is_none()
                && view.future_capacity.is_some()
                && !matches!(view.readiness, OutputPlanningCreditView::OutputBlocked(_))
            {
                decoders += 1;
            }
        }
        Some(
            self.controller_work_policy(hint, queue)
                .for_ready_decoders(decoders),
        )
    }

    pub(super) fn planned_work_policy_matches(
        &self,
        captured: &ControllerSnapshot,
        work: &[PlanningWorkSelection],
        hint: &ferrum_interfaces::BatchHint,
    ) -> bool {
        let policy = self.controller_work_policy(hint, &captured.queue);
        if policy != captured.snapshot.capabilities.work_policy {
            return false;
        }
        let mut decoders = 0;
        for row in &captured.snapshot.requests {
            if !captured.budget.poll() {
                return false;
            }
            if row.readiness == RequestReadiness::Ready
                && !row.timing.completed()
                && matches!(row.phase, RequestPhaseView::Decode)
            {
                decoders += 1;
            }
        }
        selections_fit(policy.for_ready_decoders(decoders), work, &captured.budget)
    }

    pub(super) fn completion_work_policy_matches(
        &self,
        proof: &ControllerSafetyProof,
        work: &[PlanningWorkSelection],
        hint: &ferrum_interfaces::BatchHint,
    ) -> bool {
        let Some(sequences) = self.sequences.try_read() else {
            return false;
        };
        let Some(envelope) =
            self.completion_work_envelope(hint, &proof.queue, &sequences, &proof.budget)
        else {
            return false;
        };
        selections_fit(envelope, work, &proof.budget)
    }
}

fn selections_fit(
    envelope: WaveWorkEnvelope,
    work: &[PlanningWorkSelection],
    budget: &ControllerBudget,
) -> bool {
    let mut used = WaveWorkUsage::default();
    for row in work {
        if !budget.poll() {
            return false;
        }
        let prefill = match row.action {
            PlanningWorkAction::Decode => None,
            PlanningWorkAction::Prefill { count, .. } => NonZeroU64::new(count.get() as u64),
        };
        if !envelope.include(&mut used, prefill) {
            return false;
        }
    }
    true
}
