//! HTTP owns handles to the existing ordered journals, never a second writer.
//! Production record delivery is bounded/nonwaiting; final flush runs only in
//! the server's explicit stop path after HTTP and engine producers have drained.
use super::*;
use ferrum_bench_core::{
    JsonlJournal, JsonlJournalDeliveryStats, JsonlJournalError, JsonlJournalErrorKind,
};

#[derive(Clone)]
struct ProfileRecord(Arc<FerrumProfileEvent>);
impl serde::Serialize for ProfileRecord {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serde::Serialize::serialize(self.0.as_ref(), serializer)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureKind {
    Contract,
    Transport,
}

struct Failure {
    kind: FailureKind,
    message: String,
}
impl Failure {
    fn contract(message: impl Into<String>) -> Self {
        Self {
            kind: FailureKind::Contract,
            message: message.into(),
        }
    }
    fn journal(error: &JsonlJournalError) -> Self {
        let kind = match error.kind() {
            JsonlJournalErrorKind::Config
            | JsonlJournalErrorKind::Encoding
            | JsonlJournalErrorKind::Panic => FailureKind::Contract,
            _ => FailureKind::Transport,
        };
        Self {
            kind,
            message: error.to_string(),
        }
    }
    fn open(error: &std::io::Error) -> Self {
        match error
            .get_ref()
            .and_then(|source| source.downcast_ref::<JsonlJournalError>())
        {
            Some(error) => Self::journal(error),
            None => Self {
                kind: FailureKind::Transport,
                message: error.to_string(),
            },
        }
    }
}

struct Destination {
    profile: bool,
    journal: JsonlJournal<ProfileRecord>,
    warned: AtomicBool,
}

#[derive(Default)]
struct Health {
    contract: Option<Error>,
    required_transport: Option<Error>,
    optional_failures: u64,
    first_optional: Option<String>,
    close_attempt_finished: bool,
}

enum CloseOwner {
    Idle,
    Running(tokio::task::JoinHandle<()>),
    Finished,
}

/// Snapshot of the original journals; a successful flush never erases loss.
#[derive(Debug, Clone)]
pub struct HttpProfileDeliveryReport {
    pub journals: Vec<HttpProfileJournalReport>,
    pub optional_failures: u64,
    pub first_optional_failure: Option<String>,
    pub required_error: Option<Error>,
    pub close_attempt_finished: bool,
}

#[derive(Debug, Clone)]
pub struct HttpProfileJournalReport {
    pub path: PathBuf,
    pub stats: JsonlJournalDeliveryStats,
}

impl HttpProfileDeliveryReport {
    pub fn result(&self) -> ferrum_types::Result<()> {
        self.required_error.clone().map_or(Ok(()), Err)
    }

    pub fn is_degraded(&self) -> bool {
        self.required_error.is_some()
            || self.optional_failures != 0
            || self.journals.iter().any(|journal| {
                journal.stats.dropped != 0
                    || journal.stats.failed != 0
                    || journal.stats.failure_kind.is_some()
            })
    }

    pub(super) fn health_json(&self) -> serde_json::Value {
        serde_json::json!({
            "required_failure": self.required_error.is_some(),
            "degraded": self.is_degraded(),
            "optional_failures": self.optional_failures,
            "close_attempt_finished": self.close_attempt_finished,
            "journals": self.journals.iter().map(|journal| {
                let stats = &journal.stats;
                serde_json::json!({
                    "accepted": stats.accepted, "dropped": stats.dropped, "failed": stats.failed,
                    "written": stats.written, "flushed": stats.flushed,
                    "unbounded_accepted": stats.unbounded_accepted,
                    "pending_payload_bytes": stats.pending_payload_bytes,
                    "peak_payload_bytes": stats.peak_payload_bytes,
                    "worker_state": format!("{:?}", stats.worker_state),
                    "worker_finished": stats.worker_finished, "flush_pending": stats.flush_pending,
                    "failure_kind": stats.failure_kind.map(|kind| format!("{kind:?}")),
                })
            }).collect::<Vec<_>>()
        })
    }
}

pub(super) struct HttpProfileDelivery {
    detail: ferrum_types::ObservabilityProfileDetail,
    destinations: Vec<Destination>,
    health: Mutex<Health>,
    close: tokio::sync::Mutex<CloseOwner>,
}

impl HttpProfileDelivery {
    pub(super) fn open(state: &AppState) -> Arc<Self> {
        let mut destinations: Vec<Destination> = Vec::with_capacity(2);
        let mut failures = Vec::new();
        let config_valid = state.profile_journal.validate();
        for (path, profile) in [
            (state.profile_jsonl.as_deref(), true),
            (state.memory_profile_jsonl.as_deref(), false),
        ] {
            let Some(path) = path else { continue };
            if config_valid.is_err() {
                continue;
            }
            // Opening is configuration work, never a handler operation. Append
            // attaches to the same normalized writer used by Engine and CLI.
            match JsonlJournal::<ProfileRecord>::open(
                path.as_path(),
                ferrum_bench_core::JsonlJournalOpenMode::Append,
                state.profile_journal.clone().into(),
            ) {
                Ok(journal) => {
                    if let Some(existing) = destinations
                        .iter_mut()
                        .find(|entry| entry.journal.path() == journal.path())
                    {
                        existing.profile |= profile;
                    } else {
                        destinations.push(Destination {
                            profile,
                            journal,
                            warned: AtomicBool::new(false),
                        });
                    }
                }
                Err(error) => failures.push((path.to_path_buf(), Failure::open(&error))),
            }
        }
        let delivery = Arc::new(Self {
            detail: state.profile_detail,
            destinations,
            health: Mutex::new(Health::default()),
            close: tokio::sync::Mutex::new(CloseOwner::Idle),
        });
        if let Err(error) = config_valid {
            delivery.record_failure(Failure::contract(error.to_string()));
        }
        for (path, failure) in failures {
            warn!(path = %path.display(), error = %failure.message, "HTTP profile journal could not be opened");
            delivery.record_failure(failure);
        }
        delivery
    }

    fn record_failure(&self, failure: Failure) {
        let mut health = self.health.lock().expect("HTTP profile health");
        if failure.kind == FailureKind::Contract {
            health.contract.get_or_insert_with(|| {
                Error::internal(format!("HTTP profile contract failed: {}", failure.message))
            });
        } else if self.detail.requires_complete_evidence() {
            health.required_transport.get_or_insert_with(|| {
                Error::io(format!("HTTP profile delivery failed: {}", failure.message))
            });
        } else {
            health.optional_failures = health.optional_failures.saturating_add(1);
            health.first_optional.get_or_insert(failure.message);
        }
    }

    pub(super) fn result(&self) -> ferrum_types::Result<()> {
        let health = self.health.lock().expect("HTTP profile health");
        let mut contract = health.contract.clone();
        let mut transport = health.required_transport.clone();
        for destination in &self.destinations {
            if let Some(kind) = destination.journal.delivery_stats().failure_kind {
                if matches!(
                    kind,
                    JsonlJournalErrorKind::Config
                        | JsonlJournalErrorKind::Encoding
                        | JsonlJournalErrorKind::Panic
                ) {
                    contract.get_or_insert_with(|| {
                        Error::internal(format!("HTTP profile writer contract failed: {kind:?}"))
                    });
                } else if self.detail.requires_complete_evidence() {
                    transport.get_or_insert_with(|| {
                        Error::io(format!("HTTP profile writer delivery failed: {kind:?}"))
                    });
                }
            }
        }
        contract.or(transport).map_or(Ok(()), Err)
    }

    pub(super) fn report(&self) -> HttpProfileDeliveryReport {
        let health = self.health.lock().expect("HTTP profile health");
        let journals: Vec<_> = self
            .destinations
            .iter()
            .map(|entry| HttpProfileJournalReport {
                path: entry.journal.path().to_path_buf(),
                stats: entry.journal.delivery_stats(),
            })
            .collect();
        let mut contract = health.contract.clone();
        let mut required_transport = health.required_transport.clone();
        // Worker failures may arrive after a successful enqueue and before the
        // next request. Expose the original writer's facts without another task
        // or waiting for shutdown to notice them.
        for journal in &journals {
            if let Some(kind) = journal.stats.failure_kind {
                if matches!(
                    kind,
                    JsonlJournalErrorKind::Config
                        | JsonlJournalErrorKind::Encoding
                        | JsonlJournalErrorKind::Panic
                ) {
                    contract.get_or_insert_with(|| {
                        Error::internal(format!(
                            "HTTP profile writer failed its contract: {kind:?}"
                        ))
                    });
                } else if self.detail.requires_complete_evidence() {
                    required_transport.get_or_insert_with(|| {
                        Error::io(format!("HTTP profile writer failed delivery: {kind:?}"))
                    });
                }
            }
        }
        HttpProfileDeliveryReport {
            journals,
            optional_failures: health.optional_failures,
            first_optional_failure: health.first_optional.clone(),
            required_error: contract.or(required_transport),
            close_attempt_finished: health.close_attempt_finished,
        }
    }

    pub(super) fn enqueue(
        &self,
        event: FerrumProfileEvent,
        include_memory: bool,
    ) -> ferrum_types::Result<()> {
        if let Err(error) = event.validate() {
            self.record_failure(Failure::contract(error));
            return self.result();
        }
        let record = ProfileRecord(Arc::new(event));
        for destination in &self.destinations {
            if !include_memory && !destination.profile {
                continue;
            }
            if let Err(error) = destination.journal.try_enqueue(record.clone()) {
                let reason = match error.kind() {
                    JsonlJournalErrorKind::Full => "full",
                    JsonlJournalErrorKind::GateBusy => "gate_busy",
                    JsonlJournalErrorKind::Closed => "closed",
                    JsonlJournalErrorKind::Worker => "worker",
                    JsonlJournalErrorKind::Encoding => "encoding",
                    JsonlJournalErrorKind::Panic => "panic",
                    JsonlJournalErrorKind::Limit => "limit",
                    JsonlJournalErrorKind::Config => "config",
                };
                metrics::counter!("ferrum.server.profile_delivery_failures_total", "reason" => reason).increment(1);
                if !destination.warned.swap(true, Ordering::Relaxed) {
                    warn!(path = %destination.journal.path().display(), reason, error = %error, "HTTP profile delivery failed; later attempts remain enabled");
                }
                // Each destination is attempted even after Contract/Required
                // failure. Health retains Contract ahead of transport errors.
                self.record_failure(Failure::journal(&error));
            }
        }
        self.result()
    }

    fn close_journals(&self) {
        for destination in &self.destinations {
            let close = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                destination.journal.close()
            }));
            let stats = destination.journal.delivery_stats();
            let failure = match close {
                Ok(Err(error)) => Some(Failure::journal(&error)),
                Err(_) => Some(Failure::contract("HTTP journal close panicked")),
                Ok(Ok(())) if stats.dropped != 0 || stats.failed != 0 || stats.flushed != stats.accepted
                    || stats.flush_pending || !stats.worker_finished || stats.pending_payload_bytes != 0
                    || stats.failure_kind.is_some() => Some(Failure {
                        kind: match stats.failure_kind {
                            Some(JsonlJournalErrorKind::Config | JsonlJournalErrorKind::Encoding | JsonlJournalErrorKind::Panic) => FailureKind::Contract,
                            _ => FailureKind::Transport,
                        },
                        message: format!("HTTP journal {} is incomplete: accepted={}, dropped={}, failed={}, flushed={}",
                            destination.journal.path().display(), stats.accepted, stats.dropped, stats.failed, stats.flushed),
                    }),
                Ok(Ok(())) => None,
            };
            if let Some(failure) = failure {
                warn!(path = %destination.journal.path().display(), error = %failure.message, "HTTP journal close reports incomplete evidence");
                self.record_failure(failure);
            }
        }
        self.health
            .lock()
            .expect("HTTP profile health")
            .close_attempt_finished = true;
    }

    pub(super) async fn close(self: &Arc<Self>) -> ferrum_types::Result<()> {
        let mut owner = self.close.lock().await;
        if matches!(*owner, CloseOwner::Idle) {
            let delivery = Arc::clone(self);
            *owner = CloseOwner::Running(tokio::task::spawn_blocking(move || {
                delivery.close_journals()
            }));
        }
        if let CloseOwner::Running(handle) = &mut *owner {
            // Borrow the original job. Cancelling stop cannot discard its join
            // or cause another close job to race this same writer.
            let result = handle.await;
            *owner = CloseOwner::Finished;
            if let Err(error) = result {
                self.record_failure(Failure::contract(format!(
                    "HTTP journal close task failed: {error}"
                )));
            }
        }
        self.result()
    }

    #[cfg(test)]
    pub(super) fn flush(&self) -> ferrum_types::Result<()> {
        for destination in &self.destinations {
            destination
                .journal
                .flush()
                .map_err(|error| Error::internal(error.to_string()))?;
        }
        self.result()
    }
}

impl AxumServer {
    pub fn profile_delivery_report(&self) -> Option<HttpProfileDeliveryReport> {
        self.lifecycle
            .profile_delivery
            .lock()
            .expect("server profile owner")
            .as_ref()
            .map(|delivery| delivery.report())
    }

    pub(super) fn prepare_profile_state(
        &self,
        mut state: AppState,
    ) -> ferrum_types::Result<AppState> {
        let mut owner = self
            .lifecycle
            .profile_delivery
            .lock()
            .expect("server profile owner");
        if owner.is_some() {
            return Err(Error::internal("HTTP profile lifecycle already prepared"));
        }
        let delivery = HttpProfileDelivery::open(&state);
        // Install custody before checking startup errors or binding the socket.
        // Failed startup remains closable through this same server lifecycle.
        *owner = Some(Arc::clone(&delivery));
        drop(owner);
        state.profile_delivery = Some(Arc::clone(&delivery));
        delivery.result()?;
        Ok(state)
    }

    pub(super) async fn close_profile_delivery(&self) -> ferrum_types::Result<()> {
        let delivery = self
            .lifecycle
            .profile_delivery
            .lock()
            .expect("server profile owner")
            .clone();
        match delivery {
            Some(delivery) => delivery.close().await,
            None => Ok(()),
        }
    }
}
