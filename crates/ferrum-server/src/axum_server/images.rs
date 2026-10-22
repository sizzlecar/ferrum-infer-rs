//! Image-only HTTP admission and a managed PNG/base64 response path.
//!
//! The current generator adapter is an experimental serial functional profile.
//! An HTTP queue is not evidence of batched execution, G3 readiness or throughput.

use super::{json_rejection_detail, AppState, ServerError};
use crate::model_registry::{ServedModelKind, ServedModelRegistry};
use axum::{
    extract::{rejection::JsonRejection, State},
    http::header,
    response::{IntoResponse, Response},
    Json,
};
use ferrum_engine::image_generation::{
    service::{ImageService, ImageServiceError, ImageServiceErrorKind, ImageServiceShutdown},
    ImageGenerationRequest,
};
use ferrum_output::{
    png::{PngDimensions, PngLimits},
    OutputBudget,
};
use ferrum_types::OutputConfig;
use serde::Deserialize;
use std::sync::Arc;

mod response;

/// Explicit image response limits, independent of the text engine's output
/// budget. `max_event_bytes` bounds the complete JSON/base64 response here.
/// The same generator Host account must cover raw BF16 RGBA, conversion scratch
/// and the final encoded response simultaneously. Codec-internal workspace is
/// separately bounded as documented by `ferrum_output::png`.
#[derive(Debug, Clone)]
pub struct ImageHttpConfig {
    pub output: OutputConfig,
    pub default_width: u32,
    pub default_height: u32,
    pub default_seed: u64,
    pub rgba_scratch_bytes: usize,
    pub png_limits: PngLimits,
}

/// Owns one generator service and its independent output admission pool.
#[derive(Clone)]
pub struct ImageHttpService {
    model_name: Arc<str>,
    service: Arc<ImageService>,
    config: ImageHttpConfig,
    output: OutputBudget,
    registry: ServedModelRegistry,
    shutdown: Arc<tokio::sync::Mutex<ImageShutdownState>>,
}

#[derive(Default)]
struct ImageShutdownState {
    completed: bool,
    failed: bool,
    task: Option<tokio::task::JoinHandle<Result<ImageServiceShutdown, ImageServiceError>>>,
    error: Option<String>,
    owner: Option<ImageServiceShutdown>,
}

impl ImageHttpService {
    pub fn new(
        model_name: impl Into<String>,
        service: impl Into<Arc<ImageService>>,
        config: ImageHttpConfig,
    ) -> anyhow::Result<Self> {
        let model_name = model_name.into();
        let service = service.into();
        let registry = ServedModelRegistry::try_new(
            model_name.clone(),
            ServedModelKind::ImageGeneration,
            vec![model_name.clone()],
            vec![],
        )?;
        anyhow::ensure!(
            (1..=64 * 1024).contains(&config.rgba_scratch_bytes),
            "image RGBA conversion scratch must be within 1..=65536 bytes"
        );
        let dimensions =
            PngDimensions::new(config.default_width.into(), config.default_height.into())?;
        config.png_limits.validate(dimensions)?;
        service.validate_request(&ImageGenerationRequest {
            prompt: String::new(),
            width: config.default_width,
            height: config.default_height,
            seed: config.default_seed,
            n: 1,
        })?;
        let output = OutputBudget::new(config.output.clone())?;
        Ok(Self {
            model_name: model_name.into(),
            service,
            config,
            output,
            registry,
            shutdown: Arc::new(tokio::sync::Mutex::new(ImageShutdownState::default())),
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
    pub fn service(&self) -> &ImageService {
        &self.service
    }
    pub fn output_usage(&self) -> ferrum_output::OutputUsage {
        self.output.usage()
    }
    pub(super) fn registry(&self) -> ServedModelRegistry {
        self.registry.clone()
    }

    /// Join once and retain the actual generator/failure owner for diagnostics
    /// or explicit recovery. Repeated server shutdown observations are safe.
    pub async fn shutdown(&self) -> ferrum_types::Result<()> {
        let mut state = self.shutdown.lock().await;
        if !state.completed {
            if state.task.is_none() {
                let service = self.service.clone();
                state.task = Some(tokio::spawn(async move { service.shutdown().await }));
            }
            // The task stays in shared state while observed. Cancelling an
            // observer leaves the sole join and its original owners recoverable.
            let result = state
                .task
                .as_mut()
                .expect("image shutdown task installed")
                .await;
            state.task.take();
            match result {
                Ok(Ok(owner)) => {
                    state.failed = owner.has_failure();
                    state.owner = Some(owner);
                }
                Ok(Err(error)) => {
                    state.failed = true;
                    state.error = Some(error.to_string());
                }
                Err(error) => {
                    state.failed = true;
                    state.error = Some(error.to_string());
                }
            }
            state.completed = true;
        }
        if state.failed {
            Err(ferrum_types::FerrumError::internal(state.error.clone().unwrap_or_else(||
                "image worker stopped with a retained failure owner; inspect take_shutdown()".into())))
        } else {
            Ok(())
        }
    }

    /// Transfer the original shutdown owner after shutdown completes. A failed
    /// receipt retains the typed native failure; dropping it uses engine-defined
    /// quarantine rather than presenting normal resource retirement.
    pub async fn take_shutdown(&self) -> Option<ImageServiceShutdown> {
        self.shutdown.lock().await.owner.take()
    }

    pub(super) fn health(&self) -> serde_json::Value {
        serde_json::json!({
            "model":self.model_name(), "ready":self.service.is_accepting(),
            "profile":"experimental_serial_functional",
            "qualification":"functional pipeline only; not G3 or measured throughput",
            "scheduler":self.service.snapshot(),
            "output":{
                "sessions":self.output.usage().sessions,
                "encoded_bytes":self.output.usage().encoded_bytes,
            },
            "host":self.service.host_account().usage().ok().map(|usage| serde_json::json!({
                "committed_bytes":usage.committed_bytes(), "allocated_bytes":usage.allocated_bytes(),
            })),
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ImageRequest {
    model: String,
    prompt: String,
    #[serde(default = "one")]
    n: u32,
    #[serde(default)]
    size: Option<String>,
    #[serde(default = "b64_json")]
    response_format: String,
    #[serde(default)]
    seed: Option<u64>,
}

fn one() -> u32 {
    1
}
fn b64_json() -> String {
    "b64_json".into()
}

impl ImageRequest {
    fn convert(
        self,
        model_name: &str,
        config: &ImageHttpConfig,
    ) -> Result<ImageGenerationRequest, ServerError> {
        if self.model != model_name {
            return Err(ServerError::invalid_request(
                "unknown image model",
                Some("model"),
            ));
        }
        if self.n != 1 {
            return Err(ServerError::unsupported_feature(
                "image generation supports n=1 only",
                Some("n"),
            ));
        }
        if self.response_format != "b64_json" {
            return Err(ServerError::unsupported_feature(
                "image response_format must be b64_json",
                Some("response_format"),
            ));
        }
        let (width, height) = match self.size {
            Some(size) => parse_size(&size)?,
            None => (config.default_width, config.default_height),
        };
        let dimensions = PngDimensions::new(width.into(), height.into())
            .map_err(|error| ServerError::invalid_request(error.to_string(), Some("size")))?;
        config
            .png_limits
            .validate(dimensions)
            .map_err(|error| ServerError::invalid_request(error.to_string(), Some("size")))?;
        Ok(ImageGenerationRequest {
            prompt: self.prompt,
            width,
            height,
            seed: self.seed.unwrap_or(config.default_seed),
            n: 1,
        })
    }
}

fn parse_size(size: &str) -> Result<(u32, u32), ServerError> {
    let invalid =
        || ServerError::invalid_request("size must be positive decimal WIDTHxHEIGHT", Some("size"));
    let (width, height) = size.split_once('x').ok_or_else(invalid)?;
    let parse = |part: &str| -> Result<u32, ServerError> {
        if part.is_empty() || !part.bytes().all(|c| c.is_ascii_digit()) {
            return Err(invalid());
        }
        part.parse::<u32>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(invalid)
    };
    Ok((parse(width)?, parse(height)?))
}

fn service_error(error: ImageServiceError) -> ServerError {
    match error.kind() {
        ImageServiceErrorKind::InvalidRequest => {
            ServerError::invalid_request(error.to_string(), None)
        }
        ImageServiceErrorKind::QueueFull => ServerError::CapacityExceeded(error.to_string()),
        ImageServiceErrorKind::Stopped
        | ImageServiceErrorKind::Cancelled
        | ImageServiceErrorKind::Shutdown => ServerError::ServiceUnavailable(error.to_string()),
        ImageServiceErrorKind::Generation | ImageServiceErrorKind::WorkerPanic => {
            ServerError::InternalError(error.to_string())
        }
    }
}

pub(super) async fn generations_handler(
    State(state): State<AppState>,
    request: Result<Json<ImageRequest>, JsonRejection>,
) -> Result<Response, ServerError> {
    let Json(request) = request.map_err(|error| {
        ServerError::invalid_request(
            format!("invalid image request: {}", json_rejection_detail(&error)),
            None,
        )
    })?;
    let images = state
        .images
        .as_ref()
        .ok_or_else(|| ServerError::ServiceUnavailable("image generator not loaded".into()))?;
    let request = request.convert(images.model_name(), &images.config)?;
    images
        .service
        .validate_request(&request)
        .map_err(service_error)?;
    // Unread HTTP bodies keep this independent output session after the device
    // stages close. Capacity failure precedes admission to the generation queue.
    let session = images
        .output
        .try_open_session()
        .map_err(|error| ServerError::CapacityExceeded(error.to_string()))?;
    let result = images
        .service
        .try_submit(request)
        .map_err(service_error)?
        .wait()
        .await
        .map_err(service_error)?;
    let bytes = response::encode(
        session.clone(),
        images.service.host_account().clone(),
        result.rgba.tensor().clone(),
        result.rgba.bytes().clone(),
        images.config.png_limits,
        images.config.rgba_scratch_bytes,
    )
    .await?;
    drop(result);
    Ok((
        [(header::CONTENT_TYPE, "application/json")],
        super::output::retained_body(session, axum::body::Body::from(bytes)),
    )
        .into_response())
}

#[cfg(test)]
mod tests;
