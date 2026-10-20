//! Journal producer custody belongs to the engine; final close belongs to its assembly.
use super::*;
use ferrum_journal::jsonl_journal::{
    normalize_jsonl_path, JsonlFinalReceipt, JsonlJournalOwner, JsonlProducerScope,
};

enum SuppliedJournal {
    Scope(JsonlProducerScope),
    Unavailable(std::io::Error),
}

/// Explicit, process-local producer scopes supplied by the product assembly.
/// This contains no final-close authority and is deliberately not serializable.
pub struct EngineJournalScopes {
    entries: BTreeMap<PathBuf, SuppliedJournal>,
}

impl EngineJournalScopes {
    pub fn new(scopes: Vec<JsonlProducerScope>) -> Result<Self> {
        let mut entries = BTreeMap::new();
        for scope in scopes {
            let path = scope.path().to_path_buf();
            if entries.contains_key(&path) {
                return Err(FerrumError::config(format!(
                    "duplicate engine journal scope for {}",
                    path.display()
                )));
            }
            entries.insert(path, SuppliedJournal::Scope(scope));
        }
        Ok(Self { entries })
    }

    /// Preserve the assembly's actual open failure. The engine applies its
    /// configured delivery policy, and never retries by opening another owner.
    pub fn with_open_failure(
        mut self,
        path: impl Into<PathBuf>,
        error: std::io::Error,
    ) -> Result<Self> {
        let path = path.into();
        let key = normalize_jsonl_path(&path).unwrap_or(path);
        if self.entries.contains_key(&key) {
            return Err(FerrumError::config(format!(
                "duplicate engine journal destination for {}",
                key.display()
            )));
        }
        self.entries
            .insert(key, SuppliedJournal::Unavailable(error));
        Ok(self)
    }

    /// Configured names, before canonicalization. Assemblies canonicalize and
    /// mint one scope per distinct destination using their original owner.
    pub fn configured_paths(config: &EngineConfig) -> Vec<PathBuf> {
        [
            config.runtime.profile_jsonl.clone(),
            config.runtime.scheduler_trace_jsonl.clone(),
            config.runtime.legacy_scheduler_trace_jsonl.clone(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

struct EngineJournal {
    scope: JsonlProducerScope,
    // Present only for the standalone library composition.
    owner: Option<JsonlJournalOwner>,
    final_receipt: Option<JsonlFinalReceipt>,
}

#[derive(Default)]
pub(super) struct EngineJournals {
    entries: BTreeMap<PathBuf, EngineJournal>,
}

#[cfg(test)]
impl EngineInner {
    /// Test publication uses the original scope; the production profile sink
    /// already holds its producer and needs no duplicate field on EngineInner.
    pub(super) fn profile_trace_journal(&self) -> Option<SchedulerTraceJournal> {
        let path = self.runtime_config.profile_jsonl.as_deref()?;
        self.journal_scopes
            .lock()
            .producer(path)
            .map(SchedulerTraceJournal::from_producer)
    }
}

impl EngineJournals {
    pub(super) fn prepare(
        config: &EngineConfig,
        supplied: Option<EngineJournalScopes>,
        health: &ObservationHealth,
    ) -> Result<Self> {
        let mut supplied = supplied.map(|supplied| supplied.entries);
        let mut journals = Self::default();
        let mut visited = HashSet::new();
        for path in EngineJournalScopes::configured_paths(config) {
            let key = normalize_jsonl_path(&path).unwrap_or_else(|_| path.clone());
            if !visited.insert(key.clone()) {
                continue;
            }
            let requested = config.runtime.profile_journal.into();
            let open = if let Some(supplied) = supplied.as_mut() {
                match supplied.remove(&key).ok_or_else(|| {
                    FerrumError::config(format!(
                        "missing explicit engine journal scope for {}",
                        key.display()
                    ))
                })? {
                    SuppliedJournal::Scope(scope) => {
                        if scope.config() != requested || scope.is_retired() {
                            return Err(FerrumError::config(format!(
                                "engine journal scope for {} has incompatible configuration or is retired",
                                key.display()
                            )));
                        }
                        Ok(EngineJournal {
                            scope,
                            owner: None,
                            final_receipt: None,
                        })
                    }
                    SuppliedJournal::Unavailable(error) => Err(error),
                }
            } else {
                JsonlJournalOwner::open(
                    &path,
                    ferrum_journal::JsonlJournalOpenMode::Truncate,
                    requested,
                )
                .and_then(|owner| {
                    let scope = owner.producer_scope().map_err(std::io::Error::other)?;
                    Ok(EngineJournal {
                        scope,
                        owner: Some(owner),
                        final_receipt: None,
                    })
                })
            };
            if let Some(journal) = open_observation_journal(
                Some(&path),
                config.runtime.profile_detail,
                health,
                |_| open,
            )? {
                journals.entries.insert(key, journal);
            }
        }
        if supplied.as_ref().is_some_and(|entries| !entries.is_empty()) {
            return Err(FerrumError::config(
                "explicit engine journal scopes contain an unconfigured destination",
            ));
        }
        Ok(journals)
    }

    pub(super) fn producer<T: Serialize + Send + 'static>(
        &self,
        path: &Path,
    ) -> Option<JsonlJournal<T>> {
        let key = normalize_jsonl_path(path).unwrap_or_else(|_| path.to_path_buf());
        self.entries.get(&key).map(|entry| entry.scope.producer())
    }

    pub(super) fn has_unfinished_custody(&self) -> bool {
        self.entries.values().any(|entry| {
            !entry.scope.is_retired()
                || (entry.owner.is_some()
                    && !entry
                        .final_receipt
                        .as_ref()
                        .is_some_and(|receipt| receipt.worker_joined))
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(super) fn pending_deliveries(&self) -> Vec<DriverJournalDelivery> {
        self.entries
            .iter()
            .map(|(path, entry)| DriverJournalDelivery {
                path: path.clone(),
                stats: entry.scope.delivery_stats(),
                scope_receipt: None,
                final_receipt: entry.final_receipt.clone(),
            })
            .collect()
    }

    /// Called only after the engine's actual producers/observers have stopped.
    /// Original controls remain in this custody on a panic or close rejection.
    pub(super) fn finish(
        &mut self,
    ) -> Vec<(DriverJournalDelivery, Option<ExecutionEventSinkError>)> {
        self.entries.iter_mut().map(|(path, entry)| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let scope = entry.scope.finish();
                let scope_error = scope.result().err();
                let mut error = scope_error.map(journal_observation_error);
                if error.is_none() && (scope.cut.dropped != 0 || scope.failed != 0 || scope.flushed != scope.cut.accepted) {
                    error = Some(ExecutionEventSinkError::transport(format!(
                        "execution journal {} prefix is incomplete: accepted={}, dropped={}, failed={}, flushed={}",
                        path.display(), scope.cut.accepted, scope.cut.dropped, scope.failed, scope.flushed,
                    )));
                }
                if !scope.scope_retired {
                    error = Some(ExecutionEventSinkError::contract("execution journal producer scope did not retire"));
                }
                if scope.scope_retired {
                    if let Some(owner) = entry.owner.as_mut() {
                        match owner.close() {
                            Ok(receipt) => {
                                if let Err(cause) = receipt.result() {
                                    let cause = journal_observation_error(cause);
                                    if error.as_ref().is_none_or(|existing| existing.kind() != ferrum_runtime::ExecutionEventSinkErrorKind::Contract) || cause.kind() == ferrum_runtime::ExecutionEventSinkErrorKind::Contract {
                                        error = Some(cause);
                                    }
                                }
                                if error.is_none() && (receipt.flush.cut.dropped != 0
                                    || receipt.flush.failed != 0
                                    || receipt.flush.flushed != receipt.flush.cut.accepted) {
                                    error = Some(ExecutionEventSinkError::transport(format!(
                                        "execution journal {} final publication prefix is incomplete",
                                        path.display()
                                    )));
                                }
                                if !receipt.worker_joined {
                                    error = Some(ExecutionEventSinkError::contract("execution journal final owner did not join its worker"));
                                }
                                entry.final_receipt = Some(receipt);
                            }
                            Err(cause) => {
                                // A refused final close is missing lifecycle
                                // proof, not an optional telemetry loss.
                                error = Some(ExecutionEventSinkError::contract(format!(
                                    "execution journal final close was not permitted: {cause}"
                                )));
                            }
                        }
                    }
                }
                (Some(scope), error)
            }));
            let (scope_receipt, error) = match result {
                Ok(result) => result,
                Err(_) => (None, Some(ExecutionEventSinkError::contract("execution journal scope finalization panicked"))),
            };
            (DriverJournalDelivery {
                path: path.clone(),
                stats: entry.scope.delivery_stats(),
                scope_receipt,
                final_receipt: entry.final_receipt.clone(),
            }, error)
        }).collect()
    }
}
