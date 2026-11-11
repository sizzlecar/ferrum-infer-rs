//! Endpoint protocols share one arrival clock, request bound and PNG validator.

use super::*;
use ferrum_bench_core::images::{ImageFailureKind, ValidatedPngResponse};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

mod comfy;
mod execution_audit;
mod workflow;
pub use execution_audit::{
    ComfyExecutionAuditConfig, ComfyExecutionAuditEvidence, ComfyExecutionAuditLimits,
};
use workflow::PreparedWorkflow;
#[cfg(test)]
mod tests;
pub use workflow::{ComfyBinding, ComfyInputValue, ComfySteps, ComfyWorkflowSpec};

pub(super) type RequestFailure = (ImageFailureKind, &'static str);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageEndpointProtocol {
    #[default]
    OpenAiImages,
    Comfy(ComfyProtocolConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComfyProtocolConfig {
    pub generation: Option<ComfyWorkflowSpec>,
    pub edit: Option<ComfyWorkflowSpec>,
    #[serde(default)]
    pub limits: ComfyProtocolLimits,
    /// Operator declaration only. The client never clears a shared graph cache.
    #[serde(default)]
    pub graph_cache_disabled_declared: bool,
    /// Observe this request's original node execution over its own WebSocket.
    /// Mutually exclusive with the legacy disabled-cache declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_audit: Option<ComfyExecutionAuditConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ComfyProtocolLimits {
    pub maximum_workflow_bytes: usize,
    pub maximum_metadata_response_bytes: usize,
    pub maximum_metadata_bytes_per_request: usize,
    /// Encoded upload bytes, including warmup and failed/unknown uploads. This
    /// is not a bound on the server's filesystem, previews or metadata overhead.
    pub maximum_uploaded_bytes: u64,
    pub maximum_polls: u32,
    pub poll_interval_ms: u64,
    pub cleanup_timeout_ms: u64,
}

impl Default for ComfyProtocolLimits {
    fn default() -> Self {
        Self {
            maximum_workflow_bytes: 1024 * 1024,
            maximum_metadata_response_bytes: 2 * 1024 * 1024,
            maximum_metadata_bytes_per_request: 64 * 1024 * 1024,
            maximum_uploaded_bytes: 512 * 1024 * 1024,
            maximum_polls: 100_000,
            poll_interval_ms: 100,
            cleanup_timeout_ms: 3_000,
        }
    }
}

impl ComfyProtocolLimits {
    fn validate(&self) -> anyhow::Result<()> {
        for value in [
            self.maximum_workflow_bytes,
            self.maximum_metadata_response_bytes,
            self.maximum_metadata_bytes_per_request,
        ] {
            ensure!(
                value > 0 && value < isize::MAX as usize,
                "Comfy byte limits must fit the address space"
            );
        }
        ensure!(
            self.maximum_metadata_response_bytes <= self.maximum_metadata_bytes_per_request,
            "Comfy aggregate metadata limit is smaller than one response"
        );
        ensure!(
            self.maximum_uploaded_bytes > 0 && self.maximum_polls > 0,
            "Comfy upload and poll capacities must be positive"
        );
        ensure!(
            (1..=86_400_000).contains(&self.poll_interval_ms)
                && (1..=86_400_000).contains(&self.cleanup_timeout_ms),
            "Comfy polling and cleanup durations must be within one day"
        );
        Ok(())
    }
}

impl ImageEndpointProtocol {
    pub fn from_path(path: &std::path::Path, maximum_bytes: usize) -> anyhow::Result<Self> {
        let bytes = workflow::read_bounded(path, maximum_bytes)?;
        let mut protocol: Self =
            serde_json::from_slice(&bytes).context("invalid endpoint protocol JSON")?;
        if let Self::Comfy(config) = &mut protocol {
            for spec in [&mut config.generation, &mut config.edit]
                .into_iter()
                .flatten()
            {
                if spec.path.is_relative() {
                    spec.path = path
                        .parent()
                        .unwrap_or(std::path::Path::new("."))
                        .join(&spec.path);
                }
            }
        }
        Ok(protocol)
    }

    pub(super) fn is_comfy(&self) -> bool {
        matches!(self, Self::Comfy(_))
    }
}

#[derive(Debug, Clone)]
pub(super) enum PreparedProtocol {
    OpenAiImages,
    Comfy(Arc<PreparedComfy>),
}

#[derive(Debug)]
pub(super) struct PreparedComfy {
    origin: reqwest::Url,
    generation: Option<PreparedWorkflow>,
    edit: Option<PreparedWorkflow>,
    limits: ComfyProtocolLimits,
    identity: ImageProtocolIdentity,
    execution_audit: Option<ComfyExecutionAuditConfig>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageProtocolIdentity {
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_workflow_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edit_workflow_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_cache_disabled_declared: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_audit: Option<ComfyExecutionAuditConfig>,
    pub timing_scope: &'static str,
}

impl PreparedProtocol {
    pub(super) fn prepare(
        endpoint: &ImageBenchmarkEndpoint,
        config: &ImageBenchmarkConfig,
        dataset: &ImageDataset,
        references: &super::edit::PreparedReferences,
    ) -> anyhow::Result<Self> {
        let ImageEndpointProtocol::Comfy(declaration) = &endpoint.protocol else {
            return Ok(Self::OpenAiImages);
        };
        declaration.limits.validate()?;
        ensure!(
            declaration.graph_cache_disabled_declared != declaration.execution_audit.is_some(),
            "Comfy requires exactly one cache-disabled declaration or bounded execution_audit"
        );
        if let Some(audit) = &declaration.execution_audit {
            audit.validate()?;
        }
        ensure!(
            endpoint.edit_url.is_none(),
            "Comfy uses one explicit origin, not edit-endpoint URLs"
        );
        let origin = super::validate_endpoint_url(&endpoint.url)?;
        ensure!(
            origin.path() == "/",
            "Comfy endpoint must be an HTTP(S) origin with root path"
        );
        let generation = declaration
            .generation
            .as_ref()
            .map(|spec| PreparedWorkflow::load(spec, false, &declaration.limits))
            .transpose()?;
        let edit = declaration
            .edit
            .as_ref()
            .map(|spec| PreparedWorkflow::load(spec, true, &declaration.limits))
            .transpose()?;
        let mut upload_bytes = 0u64;
        for (index, sample) in dataset.samples.iter().enumerate() {
            let is_edit = !sample.effective_mode(&dataset.mode).is_generate();
            let workflow = if is_edit {
                edit.as_ref()
            } else {
                generation.as_ref()
            }
            .context("Comfy workflow missing for a dataset request mode")?;
            workflow.validate_sample(sample)?;
            if declaration.execution_audit.is_some() {
                workflow.validate_execution_audit()?;
            }
            if let Some(reference) = references.for_sample(index) {
                upload_bytes = upload_bytes
                    .checked_add(reference.identity.png_bytes as u64)
                    .context("Comfy total encoded uploads overflow")?;
            }
        }
        upload_bytes = upload_bytes
            .checked_mul(
                config
                    .repetitions
                    .checked_add(config.warmup_repetitions)
                    .context("Comfy repetition count overflows")? as u64,
            )
            .context("Comfy total encoded uploads overflow")?;
        ensure!(
            upload_bytes <= declaration.limits.maximum_uploaded_bytes,
            "Comfy planned encoded uploads exceed the declared retained-upload budget"
        );
        let identity = ImageProtocolIdentity {
            kind: "comfy_prompt_history_view_v1",
            generation_workflow_sha256: generation.as_ref().map(|w| w.sha256.clone()),
            edit_workflow_sha256: edit.as_ref().map(|w| w.sha256.clone()),
            graph_cache_disabled_declared: Some(declaration.graph_cache_disabled_declared),
            execution_audit: declaration.execution_audit.clone(),
            timing_scope: "first protocol operation through original PNG body reception; includes optional WebSocket handshake, edit upload, queue wait and history-poll granularity; excludes audit tail and local PNG validation/capture; audit tail retains a client slot and affects closed-loop throughput; server identity and disabled-cache mode are declarations",
        };
        Ok(Self::Comfy(Arc::new(PreparedComfy {
            origin,
            generation,
            edit,
            limits: declaration.limits.clone(),
            identity,
            execution_audit: declaration.execution_audit.clone(),
        })))
    }

    pub(super) fn identity(&self) -> ImageProtocolIdentity {
        match self {
            Self::OpenAiImages => ImageProtocolIdentity {
                kind: "open_ai_images", generation_workflow_sha256: None,
                edit_workflow_sha256: None, graph_cache_disabled_declared: None,
                execution_audit: None,
                timing_scope: "POST start through full OpenAI image JSON body reception, excludes local PNG validation/capture",
            },
            Self::Comfy(config) => config.identity.clone(),
        }
    }

    pub(super) async fn receive_comfy(
        &self,
        client: &reqwest::Client,
        sample: &ferrum_bench_core::images::ImageSample,
        config: &ImageBenchmarkConfig,
        reference: Option<&super::edit::PreparedReference>,
        record: &mut ImageRequestRecord,
        control: &RunControl,
    ) -> (
        Result<ReceivedPayload, RequestFailure>,
        ComfyRequestEvidence,
    ) {
        let Self::Comfy(prepared) = self else {
            unreachable!()
        };
        comfy::receive(prepared, client, sample, config, reference, record, control).await
    }
}

pub(super) enum ReceivedPayload {
    OpenAiJson(Vec<u8>),
    Png(Vec<u8>),
}

impl ReceivedPayload {
    pub(super) fn validate(
        self,
        width: u32,
        height: u32,
        limits: &ImageResponseLimits,
    ) -> Result<ValidatedPngResponse, ferrum_bench_core::images::ImageBenchError> {
        match self {
            Self::OpenAiJson(bytes) => ferrum_bench_core::images::validate_image_response_with_png(
                &bytes, width, height, limits,
            ),
            Self::Png(bytes) => {
                ferrum_bench_core::images::validate_png_response(bytes, width, height, limits)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkStopReason {
    Cancelled,
    RemoteStateUnknown,
}

#[derive(Clone)]
pub(super) struct RunControl(Arc<tokio::sync::watch::Sender<Option<BenchmarkStopReason>>>);

impl RunControl {
    pub(super) fn new() -> Self {
        let (sender, _) = tokio::sync::watch::channel(None);
        Self(Arc::new(sender))
    }
    pub(super) fn reason(&self) -> Option<BenchmarkStopReason> {
        *self.0.borrow()
    }
    pub(super) fn stop(&self, reason: BenchmarkStopReason) {
        self.0.send_if_modified(|current| {
            if current.is_none() || matches!(reason, BenchmarkStopReason::RemoteStateUnknown) {
                *current = Some(reason);
                true
            } else {
                false
            }
        });
    }
    pub(super) async fn stopped(&self) {
        let mut receiver = self.0.subscribe();
        loop {
            let stopped = receiver.borrow_and_update().is_some();
            if stopped {
                return;
            }
            if receiver.changed().await.is_err() {
                return;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ComfyRequestEvidence {
    pub sample_index: usize,
    pub phase: ImageRequestPhase,
    pub repetition: usize,
    pub prompt_id: String,
    pub submitted_workflow_sha256: Option<String>,
    pub history_execution_graph_sha256: Option<String>,
    pub submission_acknowledged: bool,
    pub protocol_elapsed_ms: f64,
    /// Full PNG receipt relative to protocol start, including audit handshake.
    pub body_complete_elapsed_ms: Option<f64>,
    /// Work after complete PNG receipt, excluded from e2e but retaining the slot.
    pub audit_tail_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_audit: Option<ComfyExecutionAuditEvidence>,
    pub cleanup_ms: f64,
    /// Completed phase durations. A phase interrupted by a failure can be zero;
    /// protocol_elapsed_ms remains the complete observed request cutoff.
    pub upload_ms: f64,
    pub submission_ms: f64,
    pub history_ms: f64,
    pub image_download_ms: f64,
    pub polls: u32,
    pub poll_interval_ms: u64,
    /// JSON bytes for upload/submission/history. Cleanup uses its own bounded
    /// metadata window and is timed separately, outside the request cutoff.
    pub metadata_bytes: usize,
    pub downloaded_png_bytes: usize,
    pub uploaded_png_bytes: usize,
    pub cleanup: ComfyCleanupOutcome,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComfyCleanupOutcome {
    NotSubmitted,
    Terminal,
    CancelledQuiescent,
    Unknown,
}
