//! Fixed-size per-completion evidence, shared only with the existing worker.
use super::*;
use ferrum_interfaces::model_executor::{ExecutorCompletionActivities, ExecutorCompletionWork};
use std::sync::atomic::AtomicU32;
#[cfg(test)]
mod tests;

#[derive(Default)]
pub(super) struct CompletionProbe {
    retention_entered: AtomicBool,
    submitted: AtomicU32,
    indeterminate: AtomicBool,
    maintenance: AtomicBool,
    recovery: AtomicBool,
    lost: AtomicBool,
}

impl CompletionProbe {
    pub(super) fn retention(&self) {
        self.retention_entered.store(true, Ordering::Relaxed);
    }
    pub(super) fn recovery(&self) {
        self.recovery.store(true, Ordering::Relaxed);
    }
    pub(super) fn maintenance(&self) {
        self.maintenance.store(true, Ordering::Relaxed);
    }
    pub(super) fn checkpoint<R: DeviceRuntime>(&self, start: &NativeCheckpointStart<R>) {
        let uncertain = matches!(
            start,
            NativeCheckpointStart::Indeterminate(_)
                | NativeCheckpointStart::ContractAfterSubmission { .. }
        );
        if uncertain || matches!(start, NativeCheckpointStart::Submitted(_)) {
            self.record_submission(uncertain);
        }
    }
    fn record_submission(&self, uncertain: bool) {
        if self
            .submitted
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .is_err()
        {
            self.lost.store(true, Ordering::Relaxed);
        }
        if uncertain {
            self.indeterminate.store(true, Ordering::Relaxed);
        }
    }
    pub(super) fn work(&self) -> ExecutorCompletionWork {
        if !self.retention_entered.load(Ordering::Relaxed) {
            return ExecutorCompletionWork::NoAdditionalWork;
        }
        ExecutorCompletionWork::AdditionalOrUnproven(ExecutorCompletionActivities {
            checkpoint_submitted: self.submitted.load(Ordering::Relaxed),
            submission_indeterminate: self.indeterminate.load(Ordering::Relaxed),
            maintenance_entered: self.maintenance.load(Ordering::Relaxed),
            recovery_entered: self.recovery.load(Ordering::Relaxed),
            evidence_lost: self.lost.load(Ordering::Relaxed),
        })
    }
}

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) async fn complete_cache_inner(
        &self,
        completion: ExecutorSequenceCompletion,
        probe: Option<&Arc<CompletionProbe>>,
    ) -> Result<()> {
        let sequence = self
            .sequences
            .lock()
            .active
            .remove(completion.cache_id())
            .ok_or_else(|| {
                FerrumError::not_found(format!(
                    "vNext completion cache `{}` is not active",
                    completion.cache_id()
                ))
            })?;
        let mut pending = PendingSequenceCompletion {
            sequence: &sequence,
            operation: None,
            completed: false,
        };
        pending.operation = Some(sequence.operation.lock().await);
        validate_sequence_completion_accounting(
            sequence.request_id(),
            sequence.product_prompt_tokens,
            sequence.replayed_output_tokens,
            &completion,
        )?;
        self.retain_completed_sequence_boundary(&sequence, probe)
            .await?;
        sequence.complete(&completion)?;
        pending.completed = true;
        Ok(())
    }
}
