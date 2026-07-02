//! Product-path observability profile event schema.

use crate::resource_trace::ResourceTraceEvent;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const OBSERVABILITY_PROFILE_SCHEMA_VERSION: u32 = 1;

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
    pub event_id: String,
    pub request_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    pub entrypoint: ProfileEntrypoint,
    pub backend: String,
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
    pub attributes: BTreeMap<String, Value>,
}

impl FerrumProfileEvent {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.schema_version != OBSERVABILITY_PROFILE_SCHEMA_VERSION {
            return Err(format!(
                "schema_version must be {OBSERVABILITY_PROFILE_SCHEMA_VERSION}"
            ));
        }
        if self.event_id.trim().is_empty() {
            return Err("event_id must be non-empty".to_string());
        }
        if self.request_id.trim().is_empty() {
            return Err("request_id must be non-empty".to_string());
        }
        if self.backend.trim().is_empty() {
            return Err("backend must be non-empty".to_string());
        }
        if self.phase.trim().is_empty() {
            return Err("phase must be non-empty".to_string());
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
            event_id: "evt-1".to_string(),
            request_id: "req-1".to_string(),
            correlation_id: None,
            entrypoint: ProfileEntrypoint::Synthetic,
            backend: "synthetic".to_string(),
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
    }
}
