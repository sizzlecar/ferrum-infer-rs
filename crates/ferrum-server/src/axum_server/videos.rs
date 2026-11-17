//! Ferrum asynchronous video jobs. This is a concrete bounded job protocol,
//! not an OpenAI video compatibility claim or another inference scheduler.
//!
//! `POST /v1/videos` takes JSON `model`, `prompt`, `seed`, `width`, `height`,
//! `num_frames`, integer `fps`, and ordered `references` objects with `type`
//! (`image`/`audio`) and `data_url` (PNG/WAVE). The legacy image-only
//! `image_reference: [{image_url: "data:image/png;base64,..."}]` remains accepted
//! as an alternative. Unknown controls and mixed representations are rejected.
//! GET `/v1/videos/{id}` returns the original service request ID and one of
//! `in_progress`, `completed`, `failed`, `cancelled`. `in_progress` covers queue,
//! model and codec work; no unsupported stage/percentage estimate is emitted.
//! Content is available only after real completion at `/v1/videos/{id}/content`.
//! DELETE requests cancellation and cleanup. A live record remains visible
//! until its waiter finishes; completed deletion does not invalidate readers
//! already holding its artifact. Completed outputs have independent durable
//! retention; TTL and capacity reclaim old terminal jobs without holding active
//! output sessions. Restart recovers records, not unfinished GPU computations.
use super::{AppState, ServerError};
use crate::model_registry::{ServedModelKind, ServedModelRegistry};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use ferrum_engine::{
    iterative_service::{ServiceError, ServiceErrorKind},
    video_generation::{
        VideoGenerationParameters, VideoOutputPlan, VideoReferenceLimits, VideoService,
        VideoServiceShutdown,
    },
};
use ferrum_models::audio_input::AudioInputLimits;
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use std::sync::Arc;

pub(super) mod content;
mod registry;
mod request;
#[cfg(test)]
mod tests;

/// Bounds HTTP metadata and concurrent ingress/downloads. Encoded/decoded PNG
/// and download backing use the original generator Host account. Framework
/// network buffers, JSON control allocations and codec-internal workspace are
/// independently bounded here, not represented as total process RSS.
#[derive(Debug, Clone)]
pub struct VideoHttpConfig {
    /// Counts uploading/running jobs and retained unsafe failure owners.
    pub maximum_jobs: usize,
    /// Exclusively owned, stable directory for restartable records and MP4s.
    pub job_directory: std::path::PathBuf,
    pub retention: VideoJobRetention,
    pub maximum_body_bytes: usize,
    /// Maximum time without receiving nonempty request body data.
    pub upload_idle_timeout: std::time::Duration,
    /// Maximum total time spent receiving the request body, not generating video.
    pub upload_timeout: std::time::Duration,
    pub maximum_prompt_bytes: usize,
    pub references: VideoReferenceLimits,
    pub image_limits: ImageInputLimits,
    pub audio_limits: AudioInputLimits,
    pub maximum_readers: u32,
    pub read_buffer_bytes: usize,
}

/// Terminal job storage is separate from upload/running admission and the
/// encoder's temporary spool. Capacity pressure may shorten the retention TTL.
#[derive(Debug, Clone, Copy)]
pub struct VideoJobRetention {
    pub maximum_jobs: usize,
    pub maximum_bytes: u64,
    pub ttl: std::time::Duration,
}
impl Default for VideoJobRetention {
    fn default() -> Self {
        Self {
            maximum_jobs: 128,
            maximum_bytes: 8 << 30,
            ttl: std::time::Duration::from_secs(24 * 60 * 60),
        }
    }
}
impl VideoHttpConfig {
    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.maximum_jobs > 0 && self.maximum_jobs <= tokio::sync::Semaphore::MAX_PERMITS,
            "invalid video registry capacity"
        );
        anyhow::ensure!(
            self.job_directory.is_absolute()
                && self.retention.maximum_jobs > 0
                && self.retention.maximum_jobs <= tokio::sync::Semaphore::MAX_PERMITS
                && self.retention.maximum_bytes >= 12
                && self.retention.ttl.as_secs() > 0
                && self.retention.ttl.as_secs() <= i64::MAX as u64,
            "video job storage requires an absolute directory and positive retention bounds"
        );
        anyhow::ensure!(
            self.maximum_body_bytes > 0 && self.maximum_body_bytes <= isize::MAX as usize,
            "invalid video request body capacity"
        );
        anyhow::ensure!(
            !self.upload_idle_timeout.is_zero()
                && !self.upload_timeout.is_zero()
                && tokio::time::Instant::now()
                    .checked_add(self.upload_idle_timeout)
                    .is_some()
                && tokio::time::Instant::now()
                    .checked_add(self.upload_timeout)
                    .is_some(),
            "video upload deadlines must be positive and representable"
        );
        anyhow::ensure!(
            self.maximum_prompt_bytes > 0 && self.maximum_prompt_bytes <= self.maximum_body_bytes,
            "invalid video prompt capacity"
        );
        self.references.validate()?;
        anyhow::ensure!(
            self.references.maximum_total <= self.maximum_body_bytes,
            "invalid video reference count"
        );
        anyhow::ensure!(
            self.audio_limits.maximum_encoded_bytes > 0
                && self.audio_limits.maximum_decoded_bytes > 0,
            "invalid audio reference capacities"
        );
        anyhow::ensure!(
            self.maximum_readers > 0
                && self.maximum_readers as usize <= tokio::sync::Semaphore::MAX_PERMITS,
            "invalid video reader capacity"
        );
        anyhow::ensure!(
            (1..=1024 * 1024).contains(&self.read_buffer_bytes),
            "video download buffer must be within 1..=1048576 bytes"
        );
        Ok(())
    }
}

/// Resolves an actual model output plan and opens its one original output
/// session. In particular the model declares the precise audio sample count;
/// the HTTP layer never computes it by truncating video frames / fps.
pub type VideoOutputPlanResolver = Arc<
    dyn Fn(&VideoGenerationParameters) -> Result<VideoOutputPlan, VideoPlanError> + Send + Sync,
>;

/// The composition root can reject unsupported geometry without constructing
/// an engine-private ServiceError. Original output admission errors keep their
/// typed capacity / stopped classification.
#[derive(Debug, thiserror::Error)]
pub enum VideoPlanError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Service(#[from] ServiceError),
}

#[derive(Clone)]
pub struct VideoHttpService {
    model_name: Arc<str>,
    service: Arc<VideoService>,
    config: VideoHttpConfig,
    output_plan: VideoOutputPlanResolver,
    registry: ServedModelRegistry,
    jobs: Arc<registry::Registry>,
    readers: Arc<tokio::sync::Semaphore>,
    shutdown_signal: tokio::sync::watch::Sender<bool>,
    shutdown: Arc<tokio::sync::Mutex<Shutdown>>,
}
#[derive(Default)]
struct Shutdown {
    task: Option<tokio::task::JoinHandle<Result<VideoServiceShutdown, String>>>,
    completed: bool,
    owner: Option<VideoServiceShutdown>,
    error: Option<String>,
}
impl VideoHttpService {
    pub fn new(
        model_name: impl Into<String>,
        service: impl Into<Arc<VideoService>>,
        config: VideoHttpConfig,
        output_plan: VideoOutputPlanResolver,
    ) -> anyhow::Result<Self> {
        config.validate()?;
        let model_name = model_name.into();
        let registry = ServedModelRegistry::try_new(
            model_name.clone(),
            ServedModelKind::VideoGeneration,
            vec![model_name.clone()],
            vec![],
        )?;
        let (shutdown_signal, _) = tokio::sync::watch::channel(false);
        Ok(Self {
            model_name: model_name.into(),
            service: service.into(),
            output_plan,
            registry,
            jobs: Arc::new(registry::Registry::new(
                config.maximum_jobs,
                &config.job_directory,
                config.retention,
                config.maximum_readers as usize,
            )?),
            readers: Arc::new(tokio::sync::Semaphore::new(config.maximum_readers as usize)),
            shutdown_signal,
            shutdown: Default::default(),
            config,
        })
    }
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
    pub fn service(&self) -> &VideoService {
        &self.service
    }
    pub(super) fn is_accepting(&self) -> bool {
        self.service.snapshot().accepting && !self.jobs.is_closing()
    }
    pub(super) fn registry(&self) -> ServedModelRegistry {
        self.registry.clone()
    }
    pub(super) fn health(&self) -> serde_json::Value {
        serde_json::json!({
            "model": self.model_name(),
            "ready": self.is_accepting(),
            "protocol": "ferrum-video-jobs-v1",
            "jobs": self.jobs.len(),
            "active_jobs": self.config.maximum_jobs.saturating_sub(self.jobs.available_slots()),
            "maximum_jobs": self.config.maximum_jobs,
            "retained_jobs": self.jobs.retained_jobs(),
            "retained_bytes": self.jobs.retained_bytes(),
            "maximum_retained_jobs": self.config.retention.maximum_jobs,
            "maximum_retained_bytes": self.config.retention.maximum_bytes,
            "retention_seconds": self.config.retention.ttl.as_secs(),
            "scheduler": self.service.snapshot(),
        })
    }
    /// Cancellation of an observer leaves this sole shutdown task installed.
    /// It drains original job waiters (including kill/reap), reader operations
    /// and then the one iterative worker. Failure owners stay in shared state.
    pub async fn shutdown(&self) -> ferrum_types::Result<()> {
        let mut state = self.shutdown.lock().await;
        if !state.completed {
            if state.task.is_none() {
                let service = self.clone();
                // Stop admission synchronously before spawning the drain.
                let tasks = service.jobs.close();
                service.shutdown_signal.send_replace(true);
                state.task = Some(tokio::spawn(async move {
                    let mut error = None;
                    for task in tasks {
                        if let Err(e) = task.await {
                            error.get_or_insert_with(|| format!("video waiter failed: {e}"));
                        }
                    }
                    // A cancelled Body may leave one blocking file read. Its
                    // original reader permit remains inside that operation.
                    let _readers = service
                        .readers
                        .clone()
                        .acquire_many_owned(service.config.maximum_readers)
                        .await
                        .map_err(|e| e.to_string())?;
                    let owner = service
                        .service
                        .shutdown()
                        .await
                        .map_err(|e| e.to_string())?;
                    service.jobs.clear_retired();
                    // Preserve the actual retirement even if a waiter panicked.
                    if let Some(error) = error {
                        tracing::error!(%error, "video shutdown waiter failure");
                        service.jobs.record_drain_error(error);
                    }
                    Ok(owner)
                }));
            }
            match state.task.as_mut().expect("installed video shutdown").await {
                Ok(Ok(owner)) => {
                    if owner.has_failure() {
                        state.error = Some("video worker has a retained failure owner".into());
                    }
                    state.owner = Some(owner);
                }
                Ok(Err(e)) => state.error = Some(e),
                Err(e) => state.error = Some(format!("video shutdown task failed: {e}")),
            }
            state.task.take();
            state.error = state.error.take().or_else(|| self.jobs.drain_error());
            state.completed = true;
        }
        match &state.error {
            Some(e) => Err(ferrum_types::FerrumError::internal(e.clone())),
            None => Ok(()),
        }
    }
    /// Transfers a completed retirement receipt. This does not start shutdown;
    /// call `shutdown().await` first, including when that call reports failure.
    pub async fn take_shutdown(&self) -> Option<VideoServiceShutdown> {
        self.shutdown.lock().await.owner.take()
    }
}

pub(super) async fn create(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Result<Response, ServerError> {
    let videos = available(&state)?;
    let jobs = videos.jobs.clone();
    let reservation = tokio::task::spawn_blocking(move || jobs.reserve())
        .await
        .map_err(internal)??;
    let (input, reservation) = request::read(request, videos, reservation).await?;
    let seed = input.parameters.seed;
    let output = (videos.output_plan)(&input.parameters).map_err(|e| match e {
        VideoPlanError::Invalid(message) => invalid(message),
        VideoPlanError::Service(error) => service_error(error),
    })?;
    // Registry lock covers the synchronous submit/installation boundary so a
    // concurrent shutdown cannot miss an accepted job. The blocking operation
    // installs its independent waiter even if its HTTP observer disconnects.
    let videos = videos.clone();
    let response = tokio::task::spawn_blocking(move || {
        videos.jobs.submit(reservation, seed, || {
            videos.service.try_submit(input, output)
        })
    })
    .await
    .map_err(internal)??;
    Ok((StatusCode::ACCEPTED, Json(response)).into_response())
}
pub(super) async fn get(
    State(state): State<AppState>,
    Path(id): Path<uuid::Uuid>,
) -> Result<Response, ServerError> {
    let jobs = available(&state)?.jobs.clone();
    let snapshot = tokio::task::spawn_blocking(move || jobs.get(id))
        .await
        .map_err(internal)?;
    Ok(match snapshot {
        Some(value) => Json(value).into_response(),
        None => missing(),
    })
}
pub(super) async fn delete(
    State(state): State<AppState>,
    Path(id): Path<uuid::Uuid>,
) -> Result<Response, ServerError> {
    let jobs = available(&state)?.jobs.clone();
    let snapshot = tokio::task::spawn_blocking(move || jobs.delete(id))
        .await
        .map_err(internal)??;
    Ok(match snapshot {
        Some(value) => Json(value).into_response(),
        None => missing(),
    })
}
fn available(state: &AppState) -> Result<&VideoHttpService, ServerError> {
    state
        .videos
        .as_ref()
        .ok_or_else(|| ServerError::ServiceUnavailable("video generator not loaded".into()))
}
fn missing() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error":{"message":"video job not found","type":"not_found"}})),
    )
        .into_response()
}
fn invalid(e: impl std::fmt::Display) -> ServerError {
    ServerError::invalid_request(e.to_string(), None)
}
fn capacity(e: impl std::fmt::Display) -> ServerError {
    ServerError::CapacityExceeded(e.to_string())
}
fn internal(e: impl std::fmt::Display) -> ServerError {
    ServerError::InternalError(e.to_string())
}
fn service_error(e: ServiceError) -> ServerError {
    match e.kind() {
        ServiceErrorKind::InvalidRequest => invalid(e),
        ServiceErrorKind::QueueFull => capacity(e),
        ServiceErrorKind::Stopped => ServerError::ServiceUnavailable(e.to_string()),
        _ => internal(e),
    }
}
