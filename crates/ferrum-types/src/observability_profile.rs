//! Product-path observability profile event schema.

use crate::resource_trace::ResourceTraceEvent;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const OBSERVABILITY_PROFILE_SCHEMA_VERSION: u32 = 1;
pub const DEFAULT_OBSERVABILITY_PROFILE_SAMPLE_RATE: f64 = 0.01;
pub const SYNTHETIC_RUNTIME_PRESET_HASH: &str =
    "sha256:6c3b8d2c431c47cf612289b02a8c631c894f34f532508fc58841e572aedaa7bc";
pub const ENGINE_RUNTIME_TRACE_PRESET_HASH: &str =
    "sha256:30c1be62aa61858deca261ebcbfb4115918c1d6d0466f7ad5ffd7bc8d901e782";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileEntrypoint {
    Run,
    Serve,
    BenchServe,
    Synthetic,
}

impl ProfileEntrypoint {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "run" => Some(Self::Run),
            "serve" => Some(Self::Serve),
            "bench_serve" | "bench-serve" | "benchserve" => Some(Self::BenchServe),
            "synthetic" => Some(Self::Synthetic),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Serve => "serve",
            Self::BenchServe => "bench_serve",
            Self::Synthetic => "synthetic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileEventKind {
    Instant,
    TimedSpan,
    Resource,
    Memory,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileStatus {
    Ok,
    Failure,
    DiagnosticOnly,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservabilityProfileDetail {
    #[default]
    Off,
    Basic,
    Debug,
    Full,
}

impl ObservabilityProfileDetail {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "basic" => Some(Self::Basic),
            "debug" => Some(Self::Debug),
            "full" => Some(Self::Full),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Basic => "basic",
            Self::Debug => "debug",
            Self::Full => "full",
        }
    }

    pub fn diagnostic_only(self) -> bool {
        matches!(self, Self::Debug | Self::Full)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FerrumObservabilityConfig {
    pub entrypoint: ProfileEntrypoint,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_jsonl: Option<PathBuf>,
    pub profile_detail: ObservabilityProfileDetail,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_profile_jsonl: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheduler_trace_jsonl: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_dump_dir: Option<PathBuf>,
    pub profile_sample_rate: f64,
}

impl FerrumObservabilityConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        entrypoint: ProfileEntrypoint,
        model: impl Into<String>,
        profile_jsonl: Option<PathBuf>,
        profile_detail: ObservabilityProfileDetail,
        memory_profile_jsonl: Option<PathBuf>,
        scheduler_trace_jsonl: Option<PathBuf>,
        request_dump_dir: Option<PathBuf>,
        profile_sample_rate: f64,
    ) -> Self {
        Self {
            entrypoint,
            model: model.into(),
            profile_jsonl,
            profile_detail,
            memory_profile_jsonl,
            scheduler_trace_jsonl,
            request_dump_dir,
            profile_sample_rate,
        }
    }

    pub fn enabled(&self) -> bool {
        self.profile_detail != ObservabilityProfileDetail::Off
            || self.profile_jsonl.is_some()
            || self.memory_profile_jsonl.is_some()
            || self.scheduler_trace_jsonl.is_some()
            || self.request_dump_dir.is_some()
    }

    pub fn synthetic_no_weight_enabled(&self) -> bool {
        self.enabled() && self.model == "synthetic/no-weight"
    }

    pub fn unified_product_profile_enabled(&self) -> bool {
        self.enabled()
            && (self.profile_detail != ObservabilityProfileDetail::Off
                || self.memory_profile_jsonl.is_some()
                || self.scheduler_trace_jsonl.is_some()
                || self.request_dump_dir.is_some())
    }

    pub fn validate(&self) -> std::result::Result<(), String> {
        if !self.profile_sample_rate.is_finite()
            || self.profile_sample_rate < 0.0
            || self.profile_sample_rate > 1.0
        {
            return Err("profile_sample_rate must be between 0.0 and 1.0".to_string());
        }
        if self.enabled()
            && self.profile_jsonl.is_none()
            && self.memory_profile_jsonl.is_none()
            && self.scheduler_trace_jsonl.is_none()
            && self.request_dump_dir.is_none()
        {
            return Err(
                "observability profile detail requires at least one artifact path".to_string(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub scope: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    pub before_bytes: Option<u64>,
    pub after_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high_water_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub available_bytes: Option<i64>,
}

impl MemorySnapshot {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.scope.trim().is_empty() {
            return Err("memory scope must be non-empty".to_string());
        }
        if self.before_bytes.is_none() || self.after_bytes.is_none() {
            return Err("memory before_bytes and after_bytes are required".to_string());
        }
        if self.high_water_bytes.is_none() {
            return Err("memory high_water_bytes is required".to_string());
        }
        if self.current_bytes.is_none() {
            return Err("memory current_bytes is required".to_string());
        }
        if self.available_bytes.is_some_and(|bytes| bytes < 0) {
            return Err("memory available_bytes must be non-negative".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileError {
    pub kind: String,
    pub message: String,
    #[serde(default)]
    pub blocking: bool,
}

impl ProfileError {
    fn validate(&self) -> std::result::Result<(), String> {
        if self.kind.trim().is_empty() {
            return Err("error kind must be non-empty".to_string());
        }
        if self.message.trim().is_empty() {
            return Err("error message must be non-empty".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayReference {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_dir: Option<String>,
}

impl ReplayReference {
    fn validate(&self) -> std::result::Result<(), String> {
        if self.command.trim().is_empty() {
            return Err("replay command must be non-empty".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FerrumProfileEvent {
    pub schema_version: u32,
    pub ts_unix_nanos: i64,
    pub event_id: String,
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    pub entrypoint: ProfileEntrypoint,
    pub backend: String,
    pub runtime_preset_hash: String,
    pub phase: String,
    pub event_kind: ProfileEventKind,
    pub timestamp: DateTime<Utc>,
    pub status: ProfileStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemorySnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<ResourceTraceEvent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ProfileError>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<ReplayReference>,
    #[serde(default)]
    pub shape: BTreeMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_detail: Option<BTreeMap<String, Value>>,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
}

impl FerrumProfileEvent {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.schema_version != OBSERVABILITY_PROFILE_SCHEMA_VERSION {
            return Err(format!(
                "schema_version must be {OBSERVABILITY_PROFILE_SCHEMA_VERSION}"
            ));
        }
        if self.ts_unix_nanos <= 0 {
            return Err("ts_unix_nanos must be positive".to_string());
        }
        if self.event_id.trim().is_empty() {
            return Err("event_id must be non-empty".to_string());
        }
        if self.request_id.trim().is_empty() {
            return Err("request_id must be non-empty".to_string());
        }
        if self
            .correlation_id
            .as_ref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err("correlation_id must be non-empty".to_string());
        }
        if self.backend.trim().is_empty() {
            return Err("backend must be non-empty".to_string());
        }
        if self.runtime_preset_hash.trim().is_empty() {
            return Err("runtime_preset_hash must be non-empty".to_string());
        }
        if self.phase.trim().is_empty() {
            return Err("phase must be non-empty".to_string());
        }
        if self.shape.is_empty() {
            return Err("shape must contain at least one field".to_string());
        }
        if self.event_kind == ProfileEventKind::TimedSpan && self.duration_us.is_none() {
            return Err("duration_us is required for timed_span events".to_string());
        }
        if let Some(memory) = &self.memory {
            memory.validate()?;
        }
        if let Some(resource) = &self.resource {
            resource.validate()?;
        }
        if let Some(error) = &self.error {
            error.validate()?;
        }
        if self.status == ProfileStatus::Failure && self.error.is_none() {
            return Err("failure events must include error detail".to_string());
        }
        if let Some(replay) = &self.replay {
            replay.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_event() -> FerrumProfileEvent {
        FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: Utc::now()
                .timestamp_nanos_opt()
                .expect("test timestamp should fit i64 nanos"),
            event_id: "evt-1".to_string(),
            request_id: "req-1".to_string(),
            correlation_id: Some("corr-1".to_string()),
            entrypoint: ProfileEntrypoint::Synthetic,
            backend: "synthetic".to_string(),
            runtime_preset_hash: SYNTHETIC_RUNTIME_PRESET_HASH.to_string(),
            phase: "request".to_string(),
            event_kind: ProfileEventKind::TimedSpan,
            timestamp: Utc::now(),
            status: ProfileStatus::Ok,
            model: Some("synthetic/no-weight".to_string()),
            duration_us: Some(100),
            memory: None,
            resource: None,
            error: None,
            replay: None,
            shape: BTreeMap::from([("batch_size".to_string(), Value::from(1))]),
            backend_detail: None,
            attributes: BTreeMap::new(),
        }
    }

    #[test]
    fn timed_span_requires_duration_and_request_id() {
        base_event().validate().unwrap();

        let mut missing_duration = base_event();
        missing_duration.duration_us = None;
        assert!(missing_duration.validate().is_err());

        let mut missing_request = base_event();
        missing_request.request_id.clear();
        assert!(missing_request.validate().is_err());

        let mut missing_correlation = base_event();
        missing_correlation.correlation_id = None;
        assert!(missing_correlation.validate().is_err());

        let mut missing_runtime_preset = base_event();
        missing_runtime_preset.runtime_preset_hash.clear();
        assert!(missing_runtime_preset.validate().is_err());

        let mut missing_shape = base_event();
        missing_shape.shape.clear();
        assert!(missing_shape.validate().is_err());
    }

    #[test]
    fn observability_config_requires_artifact_path_when_detail_enabled() {
        let config = FerrumObservabilityConfig::new(
            ProfileEntrypoint::Run,
            "synthetic/no-weight",
            None,
            ObservabilityProfileDetail::Basic,
            None,
            None,
            None,
            DEFAULT_OBSERVABILITY_PROFILE_SAMPLE_RATE,
        );
        assert!(config.enabled());
        assert!(config.synthetic_no_weight_enabled());
        assert!(config.validate().is_err());

        let with_artifact = FerrumObservabilityConfig::new(
            ProfileEntrypoint::Serve,
            "synthetic/no-weight",
            Some(PathBuf::from("profile.jsonl")),
            ObservabilityProfileDetail::Basic,
            None,
            None,
            None,
            DEFAULT_OBSERVABILITY_PROFILE_SAMPLE_RATE,
        );
        assert!(with_artifact.validate().is_ok());
        assert!(with_artifact.unified_product_profile_enabled());

        let invalid_sample_rate = FerrumObservabilityConfig::new(
            ProfileEntrypoint::Run,
            "model",
            Some(PathBuf::from("profile.jsonl")),
            ObservabilityProfileDetail::Off,
            None,
            None,
            None,
            1.1,
        );
        assert!(invalid_sample_rate.validate().is_err());
    }
}
