//! Observation payload travels with the original completion owner. It never
//! owns a Step, sequence, device allocation, or a second recovery route.
use super::*;

pub(super) struct PendingSubmissionObservation {
    sink: Arc<dyn ProductExecutionEventSink>,
    capture: ExecutionFactsCapture,
    observation: SubmissionObservation,
    legacy_attribution: bool,
}

impl PendingSubmissionObservation {
    pub(super) fn new(
        sink: Arc<dyn ProductExecutionEventSink>,
        capture: ExecutionFactsCapture,
        observation: SubmissionObservation,
        legacy_attribution: bool,
    ) -> Self {
        Self {
            sink,
            capture,
            observation,
            legacy_attribution,
        }
    }

    /// Consumed exactly once, after its caller has retained the actual receipt.
    /// A sink panic is an observation contract error, never a reason to retry.
    pub(super) fn deliver(self, receipt: &OperationCompletionReceipt) -> Option<FerrumError> {
        let Self {
            sink,
            capture,
            observation,
            legacy_attribution,
        } = self;
        let (work, attribution, device_facts) = observation.into_parts();
        let mut failure = None;
        let mut record =
            |call: &mut dyn FnMut() -> std::result::Result<(), ExecutionEventSinkError>| {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let result = call();
                    crate::event_sink::resolve_event_delivery(sink.as_ref(), result)
                }))
                .unwrap_or_else(|_| {
                    Err(ExecutionEventSinkError::contract(
                        "terminal execution observer panicked",
                    ))
                });
                if let Err(error) = result {
                    if failure
                        .as_ref()
                        .is_none_or(|prior: &ExecutionEventSinkError| {
                            prior.kind() == ExecutionEventSinkErrorKind::Transport
                                && error.kind() == ExecutionEventSinkErrorKind::Contract
                        })
                    {
                        failure = Some(error);
                    }
                }
            };
        if capture == ExecutionFactsCapture::On {
            record(&mut || {
                let work = work.as_ref().ok_or_else(|| {
                    ExecutionEventSinkError::contract(
                        "facts-enabled submission has no original work observation",
                    )
                })?;
                let facts = TerminalExecutionFacts::new(
                    receipt,
                    work,
                    attribution.as_ref(),
                    device_facts.as_ref(),
                )
                .map_err(|error| ExecutionEventSinkError::contract(error.to_string()))?;
                sink.record_execution_facts(facts)
            });
        }
        if legacy_attribution && attribution.is_some() {
            // This attachment is moved, never cloned to create another ledger.
            let mut attribution = attribution;
            record(&mut || {
                let attribution = attribution
                    .take()
                    .expect("one legacy attribution delivery")
                    .bind_terminal_timing(receipt.submission_timing().clone())
                    .map_err(|error| ExecutionEventSinkError::contract(error.to_string()))?;
                sink.record_device_submission_attribution(&attribution)
            });
        } else if !matches!(
            receipt.submission_timing(),
            DeviceTimingMeasurement::NotRequested
        ) {
            record(&mut || sink.record_physical_device_submission_timing(receipt));
        }
        failure.map(|error| FerrumError::backend(format!("execution observation failed: {error}")))
    }
}
