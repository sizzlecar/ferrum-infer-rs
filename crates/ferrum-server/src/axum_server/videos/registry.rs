use super::*;
use ferrum_engine::video_generation::{
    VideoCancellationHandle, VideoGenerationResult, VideoJob, VideoJobError,
};
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use std::{
    any::Any,
    collections::HashMap,
    panic::AssertUnwindSafe,
    sync::{Mutex, MutexGuard},
};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore},
    task::JoinHandle,
};
use uuid::Uuid;

mod store;
pub(super) use store::Artifact;

pub(super) struct Registry {
    slots: Arc<Semaphore>,
    state: Mutex<State>,
    store: store::Store,
    retention: VideoJobRetention,
}
#[derive(Default)]
struct State {
    closing: bool,
    entries: HashMap<Uuid, Entry>,
    tasks: Vec<JoinHandle<()>>,
    drain_error: Option<String>,
}
struct Entry {
    snapshot: Snapshot,
    cancel: Option<VideoCancellationHandle>,
    outcome: Option<Outcome>,
    // Only live work and unreaped/panicked owners retain active admission.
    _slot: Option<Arc<OwnedSemaphorePermit>>,
}
enum Outcome {
    Finished(Result<VideoGenerationResult, VideoJobError>),
    Stored(Artifact),
    Retired,
    Panicked { _payload: Box<dyn Any + Send> },
}
impl Outcome {
    fn requires_retention(&self) -> bool {
        matches!(self, Self::Panicked { .. })
            || matches!(self, Self::Finished(Err(VideoJobError::Encoding(e))) if e.has_unreaped_process())
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Status {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Snapshot {
    pub id: Uuid,
    pub status: Status,
    pub created_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<i64>,
    pub seed: u64,
    /// A request, not a claim that device work or the encoder has stopped.
    pub cancel_requested: bool,
    pub delete_requested: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoded_bytes: Option<u64>,
}
impl Registry {
    pub fn new(
        maximum: usize,
        directory: &std::path::Path,
        retention: VideoJobRetention,
        readers: usize,
    ) -> anyhow::Result<Self> {
        let maximum_records = maximum
            .checked_add(retention.maximum_jobs)
            .and_then(|n| n.checked_add(readers))
            .ok_or_else(|| anyhow::anyhow!("video record limit overflow"))?;
        let (store, snapshots) =
            store::Store::open(directory, retention.maximum_bytes, maximum_records)?;
        let mut state = State::default();
        for snapshot in snapshots {
            let outcome = if snapshot.status == Status::Completed {
                Outcome::Stored(store.restore_artifact(&snapshot)?)
            } else {
                Outcome::Retired
            };
            state.entries.insert(
                snapshot.id,
                Entry {
                    snapshot,
                    cancel: None,
                    outcome: Some(outcome),
                    _slot: None,
                },
            );
        }
        let registry = Self {
            slots: Arc::new(Semaphore::new(maximum)),
            state: Mutex::new(state),
            store,
            retention,
        };
        registry.prune(&mut registry.state(), chrono::Utc::now().timestamp());
        if let Some(error) = registry.drain_error() {
            anyhow::bail!(error);
        }
        Ok(registry)
    }
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub fn len(&self) -> usize {
        self.state().entries.len()
    }
    pub fn available_slots(&self) -> usize {
        self.slots.available_permits()
    }
    pub fn retained_jobs(&self) -> usize {
        self.state()
            .entries
            .values()
            .filter(|entry| {
                entry
                    .outcome
                    .as_ref()
                    .is_some_and(|outcome| !outcome.requires_retention())
            })
            .count()
    }
    pub fn retained_bytes(&self) -> u64 {
        self.store.bytes()
    }
    pub fn is_closing(&self) -> bool {
        let state = self.state();
        state.closing || state.drain_error.is_some() || self.store.error().is_some()
    }
    fn accepting(&self, state: &State) -> Result<(), ServerError> {
        if state.closing || state.drain_error.is_some() || self.store.error().is_some() {
            return Err(ServerError::ServiceUnavailable(
                "video service is shutting down or its job store is unavailable".into(),
            ));
        }
        Ok(())
    }
    pub fn reserve(&self) -> Result<Arc<OwnedSemaphorePermit>, ServerError> {
        let mut state = self.state();
        self.prune(&mut state, chrono::Utc::now().timestamp());
        self.accepting(&state)?;
        state.tasks.retain(|task| !task.is_finished());
        self.slots
            .clone()
            .try_acquire_owned()
            .map(Arc::new)
            .map_err(|_| capacity("video upload/running job capacity exhausted"))
    }
    pub fn submit(
        self: &Arc<Self>,
        slot: Arc<OwnedSemaphorePermit>,
        seed: u64,
        submit: impl FnOnce() -> Result<VideoJob, ServiceError>,
    ) -> Result<Snapshot, ServerError> {
        let mut state = self.state();
        self.accepting(&state)?;
        let job = submit().map_err(service_error)?;
        let cancel = job.cancellation_handle();
        let id = job.correlation().request_id;
        let snapshot = Snapshot {
            id,
            status: Status::InProgress,
            created_at: chrono::Utc::now().timestamp(),
            finished_at: None,
            seed,
            cancel_requested: false,
            delete_requested: false,
            error: None,
            encoded_bytes: None,
        };
        // Recording failure after engine acceptance cancels the actual job but
        // still installs its original waiter before returning an HTTP error.
        let saved = self.store.write(&snapshot);
        if let Err(error) = &saved {
            cancel.cancel();
            state.drain_error = Some(format!("video job persistence failed: {error}"));
        }
        state.entries.insert(
            id,
            Entry {
                snapshot: snapshot.clone(),
                cancel: Some(cancel),
                outcome: None,
                _slot: Some(slot.clone()),
            },
        );
        let registry = self.clone();
        state.tasks.push(tokio::spawn(async move {
            let _slot = slot;
            let result = AssertUnwindSafe(job.wait()).catch_unwind().await;
            let outcome = match result {
                Ok(result) => Outcome::Finished(result),
                Err(_payload) => Outcome::Panicked { _payload },
            };
            // Durable copying/syncing runs away from the async executor. The
            // registry mutex is released throughout the MP4 copy itself.
            let finishing = registry.clone();
            if let Err(error) =
                tokio::task::spawn_blocking(move || finishing.complete(id, outcome)).await
            {
                registry.record_drain_error(format!("video persistence worker failed: {error}"));
            }
        }));
        saved.map_err(internal)?;
        Ok(snapshot)
    }
    fn complete(&self, id: Uuid, outcome: Outcome) {
        let (outcome, stored_error, result_details) = match outcome {
            Outcome::Finished(Ok(result)) => {
                let details = (result.seed, result.artifact.receipt().encoded_bytes);
                if let Some(reservation) = self.reserve_output(id, details.1) {
                    let persisted = match reservation {
                        Ok(reservation) => {
                            let reader = match result.artifact.open() {
                                Ok(reader) => reader,
                                Err(error) => {
                                    // open currently returns only file I/O
                                    // failures, but preserve its typed owner
                                    // through the ordinary encoder retirement
                                    // path rather than erasing Mp4Error.
                                    return self.record_completion(
                                        id,
                                        Outcome::Finished(Err(VideoJobError::Encoding(error))),
                                        None,
                                        None,
                                    );
                                }
                            };
                            self.store.copy_reserved(id, reservation, reader)
                        }
                        Err(error) => Err(error),
                    };
                    match persisted {
                        Ok(artifact) => (Outcome::Stored(artifact), None, Some(details)),
                        Err(error) => {
                            tracing::error!(job_id=%id, %error, "video output persistence failed");
                            (
                                Outcome::Retired,
                                Some(
                                    "video output persistence failed or retained storage is full"
                                        .to_owned(),
                                ),
                                None,
                            )
                        }
                    }
                } else {
                    (Outcome::Retired, None, Some(details))
                }
                // The original result now drops its OutputSession and spool.
            }
            outcome => (outcome, None, None),
        };
        self.record_completion(id, outcome, stored_error, result_details);
    }
    fn reserve_output(&self, id: Uuid, bytes: u64) -> Option<anyhow::Result<store::Reservation>> {
        let mut state = self.state();
        self.prune(&mut state, chrono::Utc::now().timestamp());
        if state
            .entries
            .get(&id)
            .expect("live waiter record")
            .snapshot
            .delete_requested
        {
            return None;
        }
        if bytes > self.retention.maximum_bytes {
            return Some(Err(anyhow::anyhow!(
                "encoded video exceeds the entire retained storage capacity"
            )));
        }
        // Eviction and the actual byte charge share one admission boundary.
        // Concurrent completions cannot both promise the same free extent.
        while !self.store.available(bytes) {
            let Some(oldest) = Self::oldest_retired(&state) else {
                break;
            };
            if !self.remove(&mut state, oldest) {
                break;
            }
        }
        Some(self.store.reserve(bytes))
    }
    fn record_completion(
        &self,
        id: Uuid,
        outcome: Outcome,
        stored_error: Option<String>,
        result_details: Option<(u64, u64)>,
    ) {
        let mut state = self.state();
        let closing = state.closing;
        let entry = state
            .entries
            .get_mut(&id)
            .expect("live video job cannot be removed before completion");
        entry.snapshot.finished_at = Some(chrono::Utc::now().timestamp());
        if let Some((seed, bytes)) = result_details {
            entry.snapshot.status = Status::Completed;
            entry.snapshot.seed = seed;
            entry.snapshot.encoded_bytes = Some(bytes);
        } else if let Some(error) = stored_error {
            entry.snapshot.status = Status::Failed;
            entry.snapshot.error = Some(error);
        } else {
            match &outcome {
                Outcome::Finished(Err(e)) if e.is_cancelled() && !outcome.requires_retention() => {
                    entry.snapshot.status = if closing {
                        Status::Failed
                    } else {
                        Status::Cancelled
                    };
                    if closing {
                        entry.snapshot.error =
                            Some("video generation interrupted by server shutdown".into());
                    }
                }
                Outcome::Finished(Err(error)) => {
                    tracing::error!(job_id=%id, %error, "video job failed");
                    entry.snapshot.status = Status::Failed;
                    entry.snapshot.error = Some("video generation or encoding failed".into());
                }
                Outcome::Panicked { .. } => {
                    entry.snapshot.status = Status::Failed;
                    entry.snapshot.error =
                        Some("video waiter panicked; server retains the original failure".into());
                }
                _ => unreachable!("completion must provide terminal status"),
            }
        }
        entry.cancel = None;
        let retained = outcome.requires_retention();
        if !retained {
            entry._slot.take();
        }
        let remove = entry.snapshot.delete_requested && !retained;
        // Safe failures have finished/reaped their work. Retaining their stderr
        // owners would also keep OutputSession admission indefinitely.
        entry.outcome = Some(if retained || matches!(outcome, Outcome::Stored(_)) {
            outcome
        } else {
            Outcome::Retired
        });
        let saved = self.store.write(&entry.snapshot);
        if let Err(error) = saved {
            entry.snapshot.status = Status::Failed;
            entry.snapshot.error = Some(
                "video completion could not be persisted; server storage is unavailable".into(),
            );
            state.drain_error = Some(format!("video completion persistence failed: {error}"));
            return;
        }
        if remove {
            self.remove(&mut state, id);
        }
        self.prune(&mut state, chrono::Utc::now().timestamp());
    }
    fn oldest_retired(state: &State) -> Option<Uuid> {
        state
            .entries
            .values()
            .filter(|e| e.outcome.as_ref().is_some_and(|o| !o.requires_retention()))
            .min_by_key(|e| {
                (
                    e.snapshot.finished_at.unwrap_or(e.snapshot.created_at),
                    e.snapshot.created_at,
                    e.snapshot.id,
                )
            })
            .map(|e| e.snapshot.id)
    }
    fn prune(&self, state: &mut State, now: i64) {
        loop {
            let retired = state
                .entries
                .values()
                .filter(|e| e.outcome.as_ref().is_some_and(|o| !o.requires_retention()))
                .count();
            let Some(oldest) = Self::oldest_retired(state) else {
                break;
            };
            let entry = &state.entries[&oldest];
            let age = now.saturating_sub(
                entry
                    .snapshot
                    .finished_at
                    .unwrap_or(entry.snapshot.created_at),
            );
            if retired <= self.retention.maximum_jobs && age < self.retention.ttl.as_secs() as i64 {
                break;
            }
            if !self.remove(state, oldest) {
                break;
            }
        }
    }
    fn remove(&self, state: &mut State, id: Uuid) -> bool {
        let entry = state.entries.get_mut(&id).expect("known retired job");
        entry.snapshot.delete_requested = true;
        // This tombstone survives a crash while a download owns the artifact.
        let saved = self
            .store
            .write(&entry.snapshot)
            .and_then(|()| match &entry.outcome {
                Some(Outcome::Stored(artifact)) => {
                    artifact.delete();
                    Ok(())
                }
                _ => self.store.remove_record(id),
            });
        match saved {
            Ok(()) => {
                state.entries.remove(&id);
                true
            }
            Err(error) => {
                state.drain_error = Some(format!("video record cleanup failed: {error}"));
                false
            }
        }
    }
    pub fn get(&self, id: Uuid) -> Option<Snapshot> {
        let mut state = self.state();
        self.prune(&mut state, chrono::Utc::now().timestamp());
        state.entries.get(&id).map(|e| e.snapshot.clone())
    }
    pub fn artifact(&self, id: Uuid) -> Option<Result<Artifact, Snapshot>> {
        let mut state = self.state();
        self.prune(&mut state, chrono::Utc::now().timestamp());
        state.entries.get(&id).map(|entry| match &entry.outcome {
            Some(Outcome::Stored(artifact))
                if entry.snapshot.status == Status::Completed
                    && !entry.snapshot.delete_requested =>
            {
                Ok(artifact.clone())
            }
            _ => Err(entry.snapshot.clone()),
        })
    }
    pub fn delete(&self, id: Uuid) -> Result<Option<Snapshot>, ServerError> {
        let mut state = self.state();
        let Some(entry) = state.entries.get_mut(&id) else {
            return Ok(None);
        };
        entry.snapshot.delete_requested = true;
        if let Some(cancel) = &entry.cancel {
            entry.snapshot.cancel_requested = true;
            cancel.cancel();
        }
        let response = entry.snapshot.clone();
        let terminal = entry
            .outcome
            .as_ref()
            .is_some_and(|o| !o.requires_retention());
        if let Err(error) = self.store.write(&entry.snapshot) {
            state.drain_error = Some(format!("video delete persistence failed: {error}"));
            return Err(internal("video deletion could not be persisted"));
        } else if terminal && !self.remove(&mut state, id) {
            return Err(internal("video deletion could not be completed"));
        }
        Ok(Some(response))
    }
    pub fn close(&self) -> Vec<JoinHandle<()>> {
        let mut state = self.state();
        state.closing = true;
        for entry in state.entries.values_mut() {
            if let Some(cancel) = &entry.cancel {
                entry.snapshot.cancel_requested = true;
                cancel.cancel();
            }
        }
        std::mem::take(&mut state.tasks)
    }
    pub fn clear_retired(&self) {
        let mut state = self.state();
        // Stored artifacts keep files on drop. Only explicit DELETE/TTL grants
        // deletion authority, so orderly shutdown is restartable.
        state
            .entries
            .retain(|_, e| e.outcome.as_ref().is_some_and(Outcome::requires_retention));
        if !state.entries.is_empty() {
            state.drain_error = Some(
                "video jobs retain unreaped encoder or panic owners; retirement is incomplete"
                    .into(),
            );
        }
    }
    pub fn record_drain_error(&self, error: String) {
        self.state().drain_error = Some(error);
    }
    pub fn drain_error(&self) -> Option<String> {
        self.state()
            .drain_error
            .clone()
            .or_else(|| self.store.error())
    }
    pub fn admit_reader(
        &self,
        readers: &Arc<Semaphore>,
    ) -> Result<OwnedSemaphorePermit, ServerError> {
        let state = self.state();
        self.accepting(&state)?;
        readers
            .clone()
            .try_acquire_owned()
            .map_err(|_| capacity("video download capacity exhausted"))
    }
}

#[cfg(test)]
mod tests;
