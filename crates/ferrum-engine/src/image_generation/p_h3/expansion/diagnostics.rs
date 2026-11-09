//! Opt-in observations of the actual owned causal phase. Text is borrowed only
//! while serializing into the pipeline's original bounded Output collector.
use crate::image_generation::continuous_trace::ImageContinuousTraceHandle;
use anyhow::{ensure, Result};
use ferrum_execution::executor_lifecycle::ExecutorScope;
use ferrum_types::FinishReason;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{panic::AssertUnwindSafe, time::Instant};
use uuid::Uuid;

pub const PH3_IMAGE_PROMPT_EXPANSION_MAX_CAPTURE_BYTES: usize = 4 * 1024 * 1024;

/// Uses the existing executor/backend timing collectors without opening an
/// event journal or requesting execution facts. The bounded CUDA profile
/// already admits Kernel diagnostics within its original lane/flight opaque
/// allowances; this sink creates no alternative native budget.
pub(super) struct KernelMetricsOnlySink;

impl ferrum_runtime::ExecutionEventSink for KernelMetricsOnlySink {
    fn enablement(&self) -> ferrum_runtime::ExecutionEventSinkEnablement {
        ferrum_runtime::ExecutionEventSinkEnablement::None
    }

    fn is_enabled(&self, _kind: ferrum_runtime::ExecutionEventKind) -> bool {
        false
    }

    fn device_timing_mode(&self) -> ferrum_runtime::DeviceTimingMode {
        ferrum_runtime::DeviceTimingMode::Kernel
    }

    fn record(
        &self,
        _permit: ferrum_runtime::EventEmissionPermit,
    ) -> std::result::Result<(), ferrum_runtime::ExecutionEventSinkError> {
        Ok(())
    }
}

impl ferrum_execution::ProductExecutionEventSink for KernelMetricsOnlySink {}

/// Disabled preserves the ordinary trace's absence of prompt content. Capture
/// is an explicit diagnostic for caller-approved inputs, never an HTTP field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum PH3ImagePromptExpansionDiagnosticPolicy {
    #[default]
    Disabled,
    Summary,
    CaptureEffectivePrompt {
        maximum_utf8_bytes: usize,
    },
}
impl<'de> Deserialize<'de> for PH3ImagePromptExpansionDiagnosticPolicy {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        // Serde's internally-tagged unit variants can discard extra fields.
        // Explicit empty structs enforce the same closed schema in every mode.
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Empty {}
        #[derive(Deserialize)]
        #[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
        enum Wire {
            Disabled(Empty),
            Summary(Empty),
            CaptureEffectivePrompt { maximum_utf8_bytes: usize },
        }
        Ok(match Wire::deserialize(deserializer)? {
            Wire::Disabled(_) => Self::Disabled,
            Wire::Summary(_) => Self::Summary,
            Wire::CaptureEffectivePrompt { maximum_utf8_bytes } => {
                Self::CaptureEffectivePrompt { maximum_utf8_bytes }
            }
        })
    }
}
impl PH3ImagePromptExpansionDiagnosticPolicy {
    pub fn validate(self) -> Result<()> {
        if let Self::CaptureEffectivePrompt { maximum_utf8_bytes } = self {
            ensure!(
                maximum_utf8_bytes > 0
                    && maximum_utf8_bytes <= PH3_IMAGE_PROMPT_EXPANSION_MAX_CAPTURE_BYTES,
                "prompt expansion capture must be between 1 byte and 4 MiB"
            );
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(in crate::image_generation::p_h3) struct DiagnosticContext {
    pub policy: PH3ImagePromptExpansionDiagnosticPolicy,
    pub trace: ImageContinuousTraceHandle,
    pub job: u64,
    pub request_id: Option<Uuid>,
}

/// Identity of the completed visual output actually bound to this request.
/// Contains no image, tensor, runtime or Host backing owner.
#[derive(Clone, Copy, Serialize)]
pub(in crate::image_generation::p_h3) struct VisionBindingReceipt {
    pub(in crate::image_generation::p_h3) reused: bool,
    pub(in crate::image_generation::p_h3) producer_job: u64,
    pub(in crate::image_generation::p_h3) producer_request_id: Option<Uuid>,
    pub(in crate::image_generation::p_h3) output_bytes: usize,
}

pub(super) struct PhaseObservation {
    context: DiagnosticContext,
    prompt_tokens: usize,
    protocol_eos_token_id: u32,
    initialization_wall_us: Option<u64>,
    retirement_started: Option<Instant>,
    token_history: Option<TokenHistoryFacts>,
    metrics: Option<Value>,
    shared_owner: Option<SharedOwner>,
    vision_binding: Option<VisionBindingReceipt>,
}

#[derive(Clone, Copy, Serialize)]
pub(super) struct SharedOwner {
    pub(super) id: Uuid,
    pub(super) participants: usize,
}

#[derive(Serialize)]
struct TokenHistoryFacts {
    output_tokens_including_eos: usize,
    last_token_id: Option<u32>,
    protocol_eos_observed: bool,
    finish: FinishReason,
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum PromptCapture<'a> {
    Disabled,
    Captured {
        text: &'a str,
    },
    CapacityRefused {
        actual_utf8_bytes: usize,
        maximum_utf8_bytes: usize,
    },
    Unavailable,
}

#[derive(Serialize)]
struct ObservationEvent<'a> {
    kind: &'static str,
    job: u64,
    request_id: Option<Uuid>,
    policy: PH3ImagePromptExpansionDiagnosticPolicy,
    prompt_tokens: usize,
    protocol_eos_token_id: u32,
    initialization_wall_us: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_owner: Option<SharedOwner>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vision_binding: Option<VisionBindingReceipt>,
    retirement_wall_us: Option<u64>,
    executor_retired: bool,
    outcome: &'static str,
    token_history: &'a Option<TokenHistoryFacts>,
    executor_metrics: &'a Option<Value>,
    effective_prompt_utf8_bytes: Option<usize>,
    effective_prompt_sha256: Option<String>,
    capture: PromptCapture<'a>,
}

fn micros(start: Instant) -> Option<u64> {
    u64::try_from(start.elapsed().as_micros()).ok()
}

impl PhaseObservation {
    pub(super) fn new(
        context: DiagnosticContext,
        prompt_tokens: usize,
        eos: u32,
        started: Instant,
    ) -> Self {
        Self {
            context,
            prompt_tokens,
            protocol_eos_token_id: eos,
            initialization_wall_us: micros(started),
            retirement_started: None,
            token_history: None,
            metrics: None,
            shared_owner: None,
            vision_binding: None,
        }
    }

    pub(super) fn with_shared_owner(mut self, owner: SharedOwner) -> Self {
        self.shared_owner = Some(owner);
        self
    }

    pub(super) fn with_vision_binding(mut self, receipt: Option<VisionBindingReceipt>) -> Self {
        self.vision_binding = receipt;
        self
    }

    pub(super) fn wave(
        &self,
        wave: u64,
        participants: usize,
        input_index: usize,
        before: usize,
        after: Option<usize>,
        outcome: &'static str,
        native_outcome: super::execute::ExpansionWaveOutcome,
    ) {
        self.context.trace.record_diagnostic(&serde_json::json!({
            "kind": "prompt_expansion_token_wave", "shared_owner": self.shared_owner,
            "wave": wave, "participants": participants, "input_index": input_index,
            "job": self.context.job, "request_id": self.context.request_id,
            "before_consumed_tokens": before, "after_consumed_tokens": after,
            "outcome": outcome, "native_outcome": native_outcome,
        }));
    }

    pub(super) fn tokens(&mut self, ids: &[u32], finish: FinishReason) {
        self.token_history = Some(TokenHistoryFacts {
            output_tokens_including_eos: ids.len(),
            last_token_id: ids.last().copied(),
            protocol_eos_observed: ids.last() == Some(&self.protocol_eos_token_id),
            finish,
        });
    }

    /// Actual observations from an owned Core machine without an ExecutorScope.
    /// Callers label their clock and scope; these are not legacy executor metrics.
    pub(super) fn set_metrics(&mut self, metrics: Value) {
        self.metrics = Some(metrics);
    }

    /// Snapshot while this original scope is still alive, after its last
    /// completed wave. No executor/runtime is retained by the observation.
    pub(super) fn retiring(&mut self, scope: Option<&ExecutorScope>) {
        self.metrics = scope.and_then(|scope| {
            std::panic::catch_unwind(AssertUnwindSafe(|| project_metrics(scope))).ok()
        });
        self.retirement_started = Some(Instant::now());
    }

    pub(super) fn record(self, retired: bool, outcome: &'static str, text: Option<&str>) {
        let text = text.filter(|_| retired && outcome == "expanded");
        let capture = match (self.context.policy, text) {
            (
                PH3ImagePromptExpansionDiagnosticPolicy::CaptureEffectivePrompt {
                    maximum_utf8_bytes,
                },
                Some(text),
            ) if text.len() <= maximum_utf8_bytes => PromptCapture::Captured { text },
            (
                PH3ImagePromptExpansionDiagnosticPolicy::CaptureEffectivePrompt {
                    maximum_utf8_bytes,
                },
                Some(text),
            ) => PromptCapture::CapacityRefused {
                actual_utf8_bytes: text.len(),
                maximum_utf8_bytes,
            },
            (PH3ImagePromptExpansionDiagnosticPolicy::CaptureEffectivePrompt { .. }, None) => {
                PromptCapture::Unavailable
            }
            _ => PromptCapture::Disabled,
        };
        self.context.trace.record_diagnostic(&ObservationEvent {
            kind: "prompt_expansion_observed",
            job: self.context.job,
            request_id: self.context.request_id,
            policy: self.context.policy,
            prompt_tokens: self.prompt_tokens,
            protocol_eos_token_id: self.protocol_eos_token_id,
            initialization_wall_us: self.initialization_wall_us,
            shared_owner: self.shared_owner,
            vision_binding: self.vision_binding,
            retirement_wall_us: self.retirement_started.and_then(micros),
            executor_retired: retired,
            outcome,
            token_history: &self.token_history,
            executor_metrics: &self.metrics,
            effective_prompt_utf8_bytes: text.map(str::len),
            effective_prompt_sha256: text
                .map(|text| format!("{:x}", Sha256::digest(text.as_bytes()))),
            capture,
        });
    }
}

fn project_metrics(scope: &ExecutorScope) -> Value {
    let status = scope.status();
    let mut projected = Map::new();
    projected.insert(
        "prefill_operations".into(),
        status.prefill_operations.into(),
    );
    projected.insert("decode_operations".into(), status.decode_operations.into());
    projected.insert(
        "avg_prefill_time_ms".into(),
        serde_json::json!((status.prefill_operations > 0).then_some(status.avg_prefill_time_ms)),
    );
    projected.insert(
        "avg_decode_time_ms".into(),
        serde_json::json!((status.decode_operations > 0).then_some(status.avg_decode_time_ms)),
    );
    if let Some(Value::Object(mut snapshot)) = scope.cache_metrics_snapshot() {
        for key in [
            "family_fingerprint",
            "program_fingerprint",
            "resolved_plan_fingerprint",
            "static_bytes",
            "wave_timing_by_phase",
            "device_timing_by_phase",
            "startup_preparation",
        ] {
            if let Some(value) = snapshot.remove(key) {
                projected.insert(key.into(), value);
            }
        }
        if let Some(Value::Object(mut counters)) = snapshot.remove("counters") {
            let mut selected = Map::new();
            for key in [
                "prefill_operations",
                "decode_operations",
                "submitted_waves",
                "completed_waves",
                "failed_waves",
                "uploaded_bytes",
                "readback_bytes",
            ] {
                if let Some(value) = counters.remove(key) {
                    selected.insert(key.into(), value);
                }
            }
            projected.insert("counters".into(), Value::Object(selected));
        }
    }
    Value::Object(projected)
}

#[cfg(test)]
mod tests;
