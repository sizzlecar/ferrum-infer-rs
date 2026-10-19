//! Rejected protocol output still owns its original asynchronous observation.
//! Its terminal receipt can end that custody, never confer a request-slot or
//! scheduler-capacity release obligation that the rejected output did not have.

use super::*;

pub(super) struct RejectedObservation {
    sequence: ExecutorSequence,
    result: Mutex<CloseObservation>,
}

impl RejectedObservation {
    /// Read error and terminal proof under the same lock. A producer may
    /// publish an error-bearing terminal between separate queries.
    pub(super) fn snapshot(&self) -> (Option<ExecutionEnded>, Option<FerrumError>) {
        match &*self.result.lock() {
            CloseObservation::Ended(ended, error) => (Some(ended.clone()), error.clone()),
            // A failed observer cannot erase its original executor's later
            // terminal fact. Rechecking this path requires its normal wake or
            // the retained report's explicit retry when no wake remains.
            CloseObservation::Unproved(error) => {
                (self.sequence.observed_end(), Some(error.clone()))
            }
            CloseObservation::Pending => (None, None),
        }
    }

    #[cfg(test)]
    pub(super) fn terminal(&self) -> Option<ExecutionEnded> {
        self.snapshot().0
    }
}

impl CloseState {
    pub(super) fn reject(&mut self, mut entry: CloseEntry, work_notify: Arc<Notify>) {
        if let Some(outcome) = entry.retained_invalid_outcome.take() {
            if entry.operation_error.is_none() {
                entry.operation_error = outcome.error;
            }
            match outcome.disposition {
                ExecutionCloseDisposition::Ended(ended) => {
                    entry.rejected_observation = Some(Arc::new(RejectedObservation {
                        sequence: ended.sequence().clone(),
                        result: Mutex::new(CloseObservation::Ended(ended, None)),
                    }));
                }
                ExecutionCloseDisposition::Held(held) => {
                    let observation = Arc::new(RejectedObservation {
                        sequence: held.key().sequence().clone(),
                        result: Mutex::new(CloseObservation::Pending),
                    });
                    entry.rejected_observation = Some(Arc::clone(&observation));
                    let executor = Arc::clone(&entry._executor);
                    entry.task = Some(tokio::spawn(async move {
                        // This is the transferred observer itself. No second
                        // finish/cancel operation or model work is issued.
                        let result = std::panic::AssertUnwindSafe(held.wait())
                            .catch_unwind()
                            .await;
                        let result = match result {
                            Ok(Ok(terminal))
                                if terminal.ended.sequence() == &observation.sequence =>
                            {
                                CloseObservation::Ended(terminal.ended, terminal.error)
                            }
                            Ok(Ok(_)) => CloseObservation::Unproved(FerrumError::internal(
                                "rejected close observer returned another incarnation",
                            )),
                            Ok(Err(error)) => CloseObservation::Unproved(error),
                            Err(_) => CloseObservation::Unproved(FerrumError::internal(
                                "rejected close observer panicked",
                            )),
                        };
                        *observation.result.lock() = result;
                        work_notify.notify_one();
                        drop(executor);
                    }));
                }
            }
        }
        self.rejected.push(entry);
    }
}
