use super::*;
use ferrum_engine::video_generation::{
    VideoCancellationHandle, VideoGenerationResult, VideoJob, VideoJobError,
};
use ferrum_output::mp4::Mp4Artifact;
use futures::FutureExt;
use serde::Serialize;
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

pub(super) struct Registry {
    slots: Arc<Semaphore>,
    state: Mutex<State>,
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
    _slot: Arc<OwnedSemaphorePermit>,
}
enum Outcome {
    Finished(Result<VideoGenerationResult, VideoJobError>),
    /// Preserve a panic payload rather than losing any resources it owns.
    Panicked {
        _payload: Box<dyn Any + Send>,
    },
}
impl Outcome {
    fn requires_retention(&self) -> bool {
        matches!(self, Self::Panicked { .. })
            || matches!(self, Self::Finished(Err(VideoJobError::Encoding(e))) if e.has_unreaped_process())
    }
}
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Status {
    InProgress,
    Completed,
    Failed,
    Cancelled,
}
#[derive(Debug, Clone, Serialize)]
pub(super) struct Snapshot {
    pub id: Uuid,
    pub status: Status,
    pub created_at: i64,
    pub seed: u64,
    /// A request, not a claim that device work or the encoder has stopped.
    pub cancel_requested: bool,
    pub delete_requested: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoded_bytes: Option<u64>,
}
impl Registry {
    pub fn new(maximum: usize) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(maximum)),
            state: Mutex::new(State::default()),
        }
    }
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
    pub fn len(&self) -> usize {
        self.state().entries.len()
    }
    pub fn is_closing(&self) -> bool {
        self.state().closing
    }
    pub fn reserve(&self) -> Result<Arc<OwnedSemaphorePermit>, ServerError> {
        let mut state = self.state();
        if state.closing {
            return Err(ServerError::ServiceUnavailable(
                "video service is shutting down".into(),
            ));
        }
        // Only completed task handles are dropped. Live waiters remain owned.
        state.tasks.retain(|task| !task.is_finished());
        self.slots
            .clone()
            .try_acquire_owned()
            .map(Arc::new)
            .map_err(|_| {
                capacity("video job registry is full; delete terminal jobs to release capacity")
            })
    }
    pub fn submit(
        self: &Arc<Self>,
        slot: Arc<OwnedSemaphorePermit>,
        seed: u64,
        submit: impl FnOnce() -> Result<VideoJob, ServiceError>,
    ) -> Result<Snapshot, ServerError> {
        let mut state = self.state();
        if state.closing {
            return Err(ServerError::ServiceUnavailable(
                "video service is shutting down".into(),
            ));
        }
        let job = submit().map_err(service_error)?;
        let cancel = job.cancellation_handle();
        let id = job.correlation().request_id;
        // Submission proves acceptance, not a GPU phase. InProgress includes
        // queueing, generation and encoding until wait() actually terminates.
        let snapshot = Snapshot {
            id,
            status: Status::InProgress,
            created_at: chrono::Utc::now().timestamp(),
            seed,
            cancel_requested: false,
            delete_requested: false,
            error: None,
            encoded_bytes: None,
        };
        state.entries.insert(
            id,
            Entry {
                snapshot: snapshot.clone(),
                cancel: Some(cancel),
                outcome: None,
                _slot: slot.clone(),
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
            registry.complete(id, outcome);
        }));
        Ok(snapshot)
    }
    fn complete(&self, id: Uuid, outcome: Outcome) {
        let mut state = self.state();
        let Some(entry) = state.entries.get_mut(&id) else {
            unreachable!("live video job cannot be removed before completion");
        };
        match &outcome {
            Outcome::Finished(Ok(result)) => {
                entry.snapshot.status = Status::Completed;
                entry.snapshot.seed = result.seed;
                entry.snapshot.encoded_bytes = Some(result.artifact.receipt().encoded_bytes);
            }
            Outcome::Finished(Err(e)) if e.is_cancelled() && !outcome.requires_retention() => {
                entry.snapshot.status = Status::Cancelled
            }
            Outcome::Finished(Err(error)) => {
                tracing::error!(job_id = %id, %error, "video job failed");
                entry.snapshot.status = Status::Failed;
                entry.snapshot.error = Some(
                    "video generation or encoding failed; server retains the original failure",
                );
            }
            Outcome::Panicked { .. } => {
                entry.snapshot.status = Status::Failed;
                entry.snapshot.error =
                    Some("video waiter panicked; server retains the original failure");
            }
        }
        entry.cancel = None;
        let remove = entry.snapshot.delete_requested && !outcome.requires_retention();
        entry.outcome = Some(outcome);
        if remove {
            state.entries.remove(&id);
        }
    }
    pub fn get(&self, id: Uuid) -> Option<Snapshot> {
        self.state().entries.get(&id).map(|e| e.snapshot.clone())
    }
    pub fn artifact(&self, id: Uuid) -> Option<Result<Mp4Artifact, Snapshot>> {
        self.state()
            .entries
            .get(&id)
            .map(|entry| match &entry.outcome {
                Some(Outcome::Finished(Ok(output))) if !entry.snapshot.delete_requested => {
                    Ok(output.artifact.clone())
                }
                _ => Err(entry.snapshot.clone()),
            })
    }
    pub fn delete(&self, id: Uuid) -> Option<Snapshot> {
        let mut state = self.state();
        let entry = state.entries.get_mut(&id)?;
        entry.snapshot.delete_requested = true;
        if let Some(cancel) = &entry.cancel {
            entry.snapshot.cancel_requested = true;
            cancel.cancel();
        }
        let response = entry.snapshot.clone();
        if entry
            .outcome
            .as_ref()
            .is_some_and(|o| !o.requires_retention())
        {
            state.entries.remove(&id);
        }
        Some(response)
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
        self.state().drain_error.clone()
    }

    /// Serialize reader admission with close(). A reader already admitted owns
    /// its artifact and permit independently of a later job DELETE.
    pub fn admit_reader(
        &self,
        readers: &Arc<Semaphore>,
    ) -> Result<OwnedSemaphorePermit, ServerError> {
        let state = self.state();
        if state.closing {
            return Err(ServerError::ServiceUnavailable(
                "video service is shutting down".into(),
            ));
        }
        readers
            .clone()
            .try_acquire_owned()
            .map_err(|_| capacity("video download capacity exhausted"))
    }
}

#[cfg(test)]
mod tests;
