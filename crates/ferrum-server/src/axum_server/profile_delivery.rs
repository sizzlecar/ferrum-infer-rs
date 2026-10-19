//! HTTP owns handles to the existing ordered journals, never a second writer.
//! Production record delivery is bounded/nonwaiting; final flush runs only in
//! the server's explicit stop path after HTTP and engine producers have drained.
use super::*;
use ferrum_journal::{
    JsonlFinalReceipt, JsonlFlushReceipt, JsonlJournal, JsonlJournalDeliveryStats,
    JsonlJournalError, JsonlJournalErrorKind, JsonlJournalOwner, JsonlProducerScope,
};

/// Explicit component scopes supplied by the product's unique journal owner.
/// Failed destinations are part of the assembly result, never reopened by HTTP.
#[derive(Default)]
pub struct HttpJournalScopes {
    entries: BTreeMap<PathBuf, std::io::Result<JsonlProducerScope>>,
}
impl HttpJournalScopes {
    pub fn new(scopes: Vec<JsonlProducerScope>) -> ferrum_types::Result<Self> {
        let mut result = Self::default();
        for scope in scopes {
            let path = scope.path().to_path_buf();
            if result.entries.insert(path, Ok(scope)).is_some() {
                return Err(Error::config("duplicate HTTP journal scope"));
            }
        }
        Ok(result)
    }
    pub fn with_open_failure(
        mut self,
        path: impl Into<PathBuf>,
        error: std::io::Error,
    ) -> ferrum_types::Result<Self> {
        let path = path.into();
        let key = ferrum_journal::normalize_jsonl_path(&path).unwrap_or(path);
        if self.entries.insert(key, Err(error)).is_some() {
            return Err(Error::config("duplicate HTTP journal destination"));
        }
        Ok(self)
    }
}

struct DestinationClosure {
    scope: JsonlProducerScope,
    // Present only for standalone HTTP assembly. An injected scope cannot
    // close any other producer or acquire final authority from its path.
    owner: Option<JsonlJournalOwner>,
}
#[derive(Default)]
struct DestinationProof {
    flush: Option<JsonlFlushReceipt>,
    final_receipt: Option<JsonlFinalReceipt>,
}

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
    closure: Mutex<DestinationClosure>,
    proof: Mutex<DestinationProof>,
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
    pub scope_receipt: Option<JsonlFlushReceipt>,
    pub final_receipt: Option<JsonlFinalReceipt>,
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
                    "scope_receipt": journal.scope_receipt,
                    "final_receipt": journal.final_receipt,
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
    /// Retain and finish transferred scopes even when startup never reached
    /// profile assembly. This does not open paths or manufacture HTTP events.
    fn unprepared(
        scopes: HttpJournalScopes,
        detail: ferrum_types::ObservabilityProfileDetail,
    ) -> Arc<Self> {
        let mut destinations = Vec::new();
        let mut failures = Vec::new();
        for (_, entry) in scopes.entries {
            match entry {
                Ok(scope) => destinations.push(Destination {
                    profile: false,
                    journal: scope.producer(),
                    warned: AtomicBool::new(false),
                    closure: Mutex::new(DestinationClosure { scope, owner: None }),
                    proof: Mutex::new(DestinationProof::default()),
                }),
                Err(error) => failures.push(Failure::open(&error)),
            }
        }
        let delivery = Arc::new(Self {
            detail,
            destinations,
            health: Mutex::new(Health::default()),
            close: tokio::sync::Mutex::new(CloseOwner::Idle),
        });
        for failure in failures {
            delivery.record_failure(failure);
        }
        delivery
    }

    pub(super) fn open(state: &AppState, mut injected: Option<HttpJournalScopes>) -> Arc<Self> {
        let mut destinations = Vec::with_capacity(2);
        let mut failures = Vec::new();
        let mut targets = BTreeMap::<PathBuf, bool>::new();
        let config_valid = state.profile_journal.validate();
        for (path, profile) in [
            (state.profile_jsonl.as_deref(), true),
            (state.memory_profile_jsonl.as_deref(), false),
        ] {
            let Some(path) = path else { continue };
            let key =
                ferrum_journal::normalize_jsonl_path(path).unwrap_or_else(|_| path.to_path_buf());
            *targets.entry(key).or_default() |= profile;
        }
        if config_valid.is_ok() {
            for (path, profile) in targets {
                let opened = match injected.as_mut() {
                    Some(scopes) => match scopes.entries.remove(&path) {
                        Some(Ok(scope))
                            if !scope.is_retired()
                                && scope.config() == state.profile_journal.into() =>
                        {
                            Ok((scope, None))
                        }
                        Some(Ok(_)) => {
                            failures.push((path, Failure::contract("HTTP journal scope is retired or differs from declared configuration")));
                            continue;
                        }
                        Some(Err(error)) => Err(error),
                        None => {
                            failures.push((
                                path,
                                Failure::contract("missing injected HTTP journal destination"),
                            ));
                            continue;
                        }
                    },
                    None => JsonlJournalOwner::open(
                        &path,
                        ferrum_journal::JsonlJournalOpenMode::Append,
                        state.profile_journal.into(),
                    )
                    .and_then(|owner| {
                        owner
                            .producer_scope()
                            .map(|scope| (scope, Some(owner)))
                            .map_err(std::io::Error::other)
                    }),
                };
                match opened {
                    Ok((scope, owner)) => destinations.push(Destination {
                        profile,
                        journal: scope.producer(),
                        warned: AtomicBool::new(false),
                        closure: Mutex::new(DestinationClosure { scope, owner }),
                        proof: Mutex::new(DestinationProof::default()),
                    }),
                    Err(error) => failures.push((path, Failure::open(&error))),
                }
            }
        }
        let unused_scopes = injected
            .as_ref()
            .is_some_and(|scopes| !scopes.entries.is_empty());
        let delivery = Arc::new(Self {
            detail: state.profile_detail,
            destinations,
            health: Mutex::new(Health::default()),
            close: tokio::sync::Mutex::new(CloseOwner::Idle),
        });
        if let Err(error) = config_valid {
            delivery.record_failure(Failure::contract(error.to_string()));
        }
        if unused_scopes {
            delivery.record_failure(Failure::contract(
                "unused injected HTTP journal destinations",
            ));
        }
        for (path, failure) in failures {
            warn!(path = %path.display(), error = %failure.message, "HTTP profile journal could not be prepared");
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
        let journals: Vec<_> = self
            .destinations
            .iter()
            .map(|entry| {
                let proof = entry.proof.lock().expect("HTTP journal proof");
                HttpProfileJournalReport {
                    path: entry.journal.path().to_path_buf(),
                    stats: entry.journal.delivery_stats(),
                    scope_receipt: proof.flush.clone(),
                    final_receipt: proof.final_receipt.clone(),
                }
            })
            .collect();
        let health = self.health.lock().expect("HTTP profile health");
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
            let mut closure = destination.closure.lock().expect("HTTP journal closure");
            let finish =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| closure.scope.finish()));
            match finish {
                Ok(receipt) => {
                    if let Err(error) = receipt.result() {
                        self.record_failure(Failure::journal(&error));
                    }
                    // The fixed prefix, not a later global accepted/pending count,
                    // proves the HTTP producer's accepted records were observed.
                    if receipt.cut.dropped != 0 || receipt.failed != 0 {
                        self.record_failure(Failure {
                            kind: FailureKind::Transport,
                            message: "HTTP journal prefix contains lost records".to_string(),
                        });
                    }
                    destination.proof.lock().expect("HTTP journal proof").flush = Some(receipt);
                }
                Err(_) => {
                    self.record_failure(Failure::contract("HTTP journal scope finish panicked"))
                }
            }
            if let Some(owner) = closure.owner.as_mut() {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner.close())) {
                    Ok(Ok(receipt)) => {
                        if let Err(error) = receipt.result() {
                            self.record_failure(Failure::journal(&error));
                        }
                        destination
                            .proof
                            .lock()
                            .expect("HTTP journal proof")
                            .final_receipt = Some(receipt);
                    }
                    Ok(Err(error)) => self.record_failure(Failure::contract(format!(
                        "HTTP journal final close incomplete: {error}"
                    ))),
                    Err(_) => {
                        self.record_failure(Failure::contract("HTTP journal final close panicked"))
                    }
                }
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
                .result()
                .map_err(|error| Error::internal(error.to_string()))?;
        }
        self.result()
    }
}

impl AxumServer {
    /// Supply only this HTTP component's publishing scopes. The caller retains
    /// final-close authority until all component scopes and tail records finish.
    pub fn with_journal_scopes(self, scopes: HttpJournalScopes) -> Self {
        *self
            .lifecycle
            .profile_scopes
            .lock()
            .expect("HTTP injected scopes") = Some(scopes);
        self
    }

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
        let injected = self
            .lifecycle
            .profile_scopes
            .lock()
            .expect("HTTP injected scopes")
            .take();
        let delivery = HttpProfileDelivery::open(&state, injected);
        // Install custody before checking startup errors or binding the socket.
        // Failed startup remains closable through this same server lifecycle.
        *owner = Some(Arc::clone(&delivery));
        drop(owner);
        state.profile_delivery = Some(Arc::clone(&delivery));
        delivery.result()?;
        Ok(state)
    }

    pub(super) async fn close_profile_delivery(&self) -> ferrum_types::Result<()> {
        let delivery = {
            // Same lock order as prepare_profile_state. start/stop publication
            // is serialized separately by stop_lock, so an unprepared scope
            // cannot be retired while a late start takes it for publication.
            let mut delivery = self
                .lifecycle
                .profile_delivery
                .lock()
                .expect("server profile owner");
            if delivery.is_none() {
                let pending = self
                    .lifecycle
                    .profile_scopes
                    .lock()
                    .expect("HTTP injected scopes")
                    .take();
                if let Some(scopes) = pending {
                    let detail = self
                        .lifecycle
                        .profile_detail
                        .lock()
                        .expect("HTTP selected profile detail")
                        .unwrap_or(self.state.profile_detail);
                    *delivery = Some(HttpProfileDelivery::unprepared(scopes, detail));
                }
            }
            delivery.clone()
        };
        match delivery {
            Some(delivery) => delivery.close().await,
            None => Ok(()),
        }
    }
}
