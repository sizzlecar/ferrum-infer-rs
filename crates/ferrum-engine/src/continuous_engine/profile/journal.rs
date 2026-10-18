use super::*;
#[cfg(test)]
use ferrum_bench_core::jsonl_journal::JsonlJournalDeliveryStats;
use ferrum_bench_core::jsonl_journal::JsonlJournalErrorKind;
use ferrum_runtime::ExecutionEventDeliveryPolicy;

#[derive(Clone)]
pub(in crate::continuous_engine) struct SchedulerTraceRecord(Arc<FerrumProfileEvent>);

impl serde::Serialize for SchedulerTraceRecord {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serde::Serialize::serialize(self.0.as_ref(), serializer)
    }
}

#[derive(Clone)]
pub(in crate::continuous_engine) struct SchedulerTraceJournal {
    inner: JsonlJournal<SchedulerTraceRecord>,
    // Clones used by scheduler traces, native fanout and shutdown share this
    // first-failure latch; transient queue pressure never disables delivery.
    warned: Arc<AtomicBool>,
}

impl SchedulerTraceJournal {
    pub(in crate::continuous_engine) fn from_producer(
        inner: JsonlJournal<SchedulerTraceRecord>,
    ) -> Self {
        Self {
            inner,
            warned: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(in crate::continuous_engine) fn enqueue(
        &self,
        event: FerrumProfileEvent,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        self.enqueue_validated(Self::validated_records(vec![event])?)
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) fn enqueue_batch(
        &self,
        events: Vec<FerrumProfileEvent>,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        self.enqueue_validated(Self::validated_records(events)?)
    }

    pub(super) fn validated_records(
        events: Vec<FerrumProfileEvent>,
    ) -> std::result::Result<Vec<SchedulerTraceRecord>, ExecutionEventSinkError> {
        for event in &events {
            event
                .validate()
                .map_err(ExecutionEventSinkError::contract)?;
        }
        Ok(events
            .into_iter()
            .map(|event| SchedulerTraceRecord(Arc::new(event)))
            .collect())
    }

    pub(super) fn enqueue_validated(
        &self,
        records: Vec<SchedulerTraceRecord>,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        self.inner
            .try_enqueue_batch(records)
            .inspect_err(|error| {
                let reason = match error.kind() {
                    JsonlJournalErrorKind::Full => "full",
                    JsonlJournalErrorKind::GateBusy => "gate_busy",
                    JsonlJournalErrorKind::Closed => "closed",
                    JsonlJournalErrorKind::Worker => "worker",
                    JsonlJournalErrorKind::Panic => "panic",
                    JsonlJournalErrorKind::Encoding => "encoding",
                    JsonlJournalErrorKind::Config => "config",
                    JsonlJournalErrorKind::Limit => "limit",
                };
                counter!("ferrum.engine.journal_delivery_failures_total", "reason" => reason)
                    .increment(1);
                if !self.warned.swap(true, Ordering::Relaxed) {
                    warn!(path = %self.path().display(), reason, error = %error,
                        "execution journal delivery failed; later attempts remain enabled");
                }
            })
            .map_err(journal_observation_error)
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) fn flush(&self) -> std::result::Result<(), JsonlJournalError> {
        self.inner.flush().result()
    }

    pub(in crate::continuous_engine) fn path(&self) -> &Path {
        self.inner.path()
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) fn delivery_stats(&self) -> JsonlJournalDeliveryStats {
        self.inner.delivery_stats()
    }
}

pub(in crate::continuous_engine) fn journal_observation_error(
    error: JsonlJournalError,
) -> ExecutionEventSinkError {
    match error.kind() {
        JsonlJournalErrorKind::Encoding
        | JsonlJournalErrorKind::Panic
        | JsonlJournalErrorKind::Config => ExecutionEventSinkError::contract(error.to_string()),
        _ => ExecutionEventSinkError::transport(error.to_string()),
    }
}

pub(in crate::continuous_engine) fn observation_delivery_policy(
    detail: ObservabilityProfileDetail,
) -> ExecutionEventDeliveryPolicy {
    if detail.requires_complete_evidence() {
        ExecutionEventDeliveryPolicy::RequiredEvidence
    } else {
        ExecutionEventDeliveryPolicy::OptionalTelemetry
    }
}

#[derive(Default)]
pub(in crate::continuous_engine) struct ObservationHealth {
    state: Mutex<ObservationHealthState>,
}

#[derive(Default)]
struct ObservationHealthState {
    optional_failures: u64,
    first_optional: Option<FerrumError>,
    first_required: Option<FerrumError>,
}

impl ObservationHealth {
    pub(in crate::continuous_engine) fn record(
        &self,
        error: &ExecutionEventSinkError,
        policy: ExecutionEventDeliveryPolicy,
    ) {
        let mut state = self.state.lock();
        if error.is_optional_transport(policy) {
            state.optional_failures = state.optional_failures.saturating_add(1);
            if state.first_optional.is_none() {
                warn!(error = %error, "optional execution telemetry degraded");
                state.first_optional = Some(FerrumError::internal(format!(
                    "execution observation failed: {error}"
                )));
            }
        } else {
            state.first_required.get_or_insert_with(|| {
                FerrumError::internal(format!("execution observation failed: {error}"))
            });
        }
    }

    pub(in crate::continuous_engine) fn snapshot(
        &self,
    ) -> (Option<FerrumError>, Option<FerrumError>, u64) {
        let state = self.state.lock();
        (
            state.first_required.clone(),
            state.first_optional.clone(),
            state.optional_failures,
        )
    }
}

impl EngineInner {
    pub(in crate::continuous_engine) fn report_observation_failure(
        &self,
        error: ExecutionEventSinkError,
    ) {
        let policy = observation_delivery_policy(self.config.runtime.profile_detail);
        self.observation_health.record(&error, policy);
        if !error.is_optional_transport(policy) {
            self.output_tasks
                .record_control_error(FerrumError::internal(format!(
                    "execution observation failed: {error}"
                )));
            self.work_notify.notify_one();
        }
    }
}

pub(in crate::continuous_engine) fn open_observation_journal<T>(
    path: Option<&Path>,
    detail: ObservabilityProfileDetail,
    health: &ObservationHealth,
    open: impl FnOnce(&Path) -> std::io::Result<T>,
) -> Result<Option<T>> {
    let Some(path) = path else { return Ok(None) };
    match open(path) {
        Ok(journal) => Ok(Some(journal)),
        Err(error) => {
            // An existing shared writer may already have failed its contract.
            // io::Error is only the open API's carrier, not proof of I/O failure.
            let failure = match error
                .get_ref()
                .and_then(|source| source.downcast_ref::<JsonlJournalError>())
            {
                Some(error) => journal_observation_error(error.clone()),
                None => ExecutionEventSinkError::transport(format!(
                    "open execution journal {}: {error}",
                    path.display(),
                )),
            };
            let policy = observation_delivery_policy(detail);
            health.record(&failure, policy);
            if failure.is_optional_transport(policy) {
                Ok(None)
            } else {
                match failure.kind() {
                    ferrum_runtime::ExecutionEventSinkErrorKind::Contract => {
                        Err(FerrumError::internal(failure.to_string()))
                    }
                    ferrum_runtime::ExecutionEventSinkErrorKind::Transport => {
                        Err(FerrumError::io(failure.to_string()))
                    }
                }
            }
        }
    }
}
