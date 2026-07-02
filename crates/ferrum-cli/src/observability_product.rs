use chrono::{Duration, Utc};
use clap::ValueEnum;
use ferrum_types::{
    FerrumError, FerrumProfileEvent, MemorySnapshot, ProfileEntrypoint, ProfileEventKind,
    ProfileStatus, ReplayReference, ResourceAction, ResourceTraceEvent, Result,
    OBSERVABILITY_PROFILE_SCHEMA_VERSION,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

const SYNTHETIC_MODEL: &str = "synthetic/no-weight";
const SYNTHETIC_BACKEND: &str = "synthetic";
const DEFAULT_PROFILE_SAMPLE_RATE: f64 = 0.01;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ProfileDetailArg {
    #[default]
    Off,
    Basic,
    Debug,
    Full,
}

impl ProfileDetailArg {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Basic => "basic",
            Self::Debug => "debug",
            Self::Full => "full",
        }
    }

    fn diagnostic_only(self) -> bool {
        matches!(self, Self::Debug | Self::Full)
    }
}

#[derive(Clone, Debug)]
pub struct ProductObservabilityConfig {
    pub entrypoint: ProfileEntrypoint,
    pub model: String,
    pub profile_jsonl: Option<PathBuf>,
    pub profile_detail: ProfileDetailArg,
    pub memory_profile_jsonl: Option<PathBuf>,
    pub scheduler_trace_jsonl: Option<PathBuf>,
    pub request_dump_dir: Option<PathBuf>,
    pub profile_sample_rate: f64,
}

impl ProductObservabilityConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        entrypoint: ProfileEntrypoint,
        model: impl Into<String>,
        profile_jsonl: Option<&PathBuf>,
        profile_detail: ProfileDetailArg,
        memory_profile_jsonl: Option<&PathBuf>,
        scheduler_trace_jsonl: Option<&PathBuf>,
        request_dump_dir: Option<&PathBuf>,
        profile_sample_rate: f64,
    ) -> Self {
        Self {
            entrypoint,
            model: model.into(),
            profile_jsonl: profile_jsonl.cloned(),
            profile_detail,
            memory_profile_jsonl: memory_profile_jsonl.cloned(),
            scheduler_trace_jsonl: scheduler_trace_jsonl.cloned(),
            request_dump_dir: request_dump_dir.cloned(),
            profile_sample_rate,
        }
    }

    pub fn enabled(&self) -> bool {
        self.profile_detail != ProfileDetailArg::Off
            || self.profile_jsonl.is_some()
            || self.memory_profile_jsonl.is_some()
            || self.scheduler_trace_jsonl.is_some()
            || self.request_dump_dir.is_some()
    }

    pub fn synthetic_no_weight_enabled(&self) -> bool {
        self.enabled() && self.model == SYNTHETIC_MODEL
    }

    fn validate(&self) -> Result<()> {
        if !self.profile_sample_rate.is_finite()
            || self.profile_sample_rate < 0.0
            || self.profile_sample_rate > 1.0
        {
            return Err(FerrumError::invalid_parameter(
                "--profile-sample-rate must be between 0.0 and 1.0",
            ));
        }
        if self.enabled()
            && self.profile_jsonl.is_none()
            && self.memory_profile_jsonl.is_none()
            && self.scheduler_trace_jsonl.is_none()
            && self.request_dump_dir.is_none()
        {
            return Err(FerrumError::invalid_parameter(
                "observability profile detail requires at least one artifact path",
            ));
        }
        Ok(())
    }
}

pub fn default_profile_sample_rate() -> f64 {
    DEFAULT_PROFILE_SAMPLE_RATE
}

pub fn write_synthetic_product_observability(
    config: &ProductObservabilityConfig,
) -> Result<Vec<PathBuf>> {
    config.validate()?;
    let request_id = format!(
        "product-obs-{}-{}",
        entrypoint_label(config.entrypoint),
        Uuid::new_v4().simple()
    );
    let replay_command = replay_command(config);
    let events = product_events(config, &request_id, &replay_command);
    let mut written = Vec::new();
    if let Some(path) = &config.profile_jsonl {
        write_profile_jsonl(path, &events)?;
        written.push(path.clone());
    }
    if let Some(path) = &config.memory_profile_jsonl {
        let memory_events: Vec<_> = events
            .iter()
            .filter(|event| event.memory.is_some())
            .cloned()
            .collect();
        write_profile_jsonl(path, &memory_events)?;
        written.push(path.clone());
    }
    if let Some(path) = &config.scheduler_trace_jsonl {
        let scheduler_events: Vec<_> = events
            .iter()
            .filter(|event| event.resource.is_some())
            .cloned()
            .collect();
        write_profile_jsonl(path, &scheduler_events)?;
        written.push(path.clone());
    }
    if let Some(dir) = &config.request_dump_dir {
        fs_create_dir_all(dir)?;
        let request_path = dir.join("request.json");
        let replay_path = dir.join("replay_command.txt");
        write_json(
            &request_path,
            &request_dump(config, &request_id, &replay_command),
        )?;
        fs_write(&replay_path, format!("{replay_command}\n"))?;
        written.push(request_path);
        written.push(replay_path);
    }
    Ok(written)
}

fn product_events(
    config: &ProductObservabilityConfig,
    request_id: &str,
    replay_command: &str,
) -> Vec<FerrumProfileEvent> {
    let base = Utc::now();
    let mut open = base_event(
        config,
        request_id,
        "request_open",
        ProfileEventKind::Resource,
        base,
    );
    open.resource = Some(ResourceTraceEvent {
        owner_kind: "request".to_string(),
        owner_id: request_id.to_string(),
        resource_kind: "request_slot".to_string(),
        action: ResourceAction::RequestOpen,
        amount: None,
        before: None,
        after: None,
        capacity: Some(1),
        reason: None,
    });

    let mut scheduler = base_event(
        config,
        request_id,
        "scheduler_admission",
        ProfileEventKind::Resource,
        base + Duration::microseconds(10),
    );
    scheduler.resource = Some(ResourceTraceEvent {
        owner_kind: "request".to_string(),
        owner_id: request_id.to_string(),
        resource_kind: "scheduler_admission_slot".to_string(),
        action: ResourceAction::Reserve,
        amount: Some(1),
        before: Some(1),
        after: Some(0),
        capacity: Some(1),
        reason: None,
    });

    let mut prefill = base_event(
        config,
        request_id,
        "synthetic_prefill",
        ProfileEventKind::TimedSpan,
        base + Duration::microseconds(20),
    );
    prefill.duration_us = Some(160);
    prefill.memory = Some(MemorySnapshot {
        scope: "process".to_string(),
        backend: Some(SYNTHETIC_BACKEND.to_string()),
        before_bytes: Some(2048),
        after_bytes: Some(2304),
        high_water_bytes: Some(2304),
        available_bytes: Some(1024 * 1024),
    });
    prefill.attributes.extend(common_attrs(config));
    prefill
        .attributes
        .insert("input_tokens".to_string(), json!(8));

    let mut complete = base_event(
        config,
        request_id,
        "request_complete",
        ProfileEventKind::Instant,
        base + Duration::microseconds(190),
    );
    complete.status = if config.profile_detail.diagnostic_only() {
        ProfileStatus::DiagnosticOnly
    } else {
        ProfileStatus::Ok
    };
    complete.replay = Some(ReplayReference {
        command: replay_command.to_string(),
        bundle_dir: config
            .request_dump_dir
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
    });
    complete.attributes.extend(common_attrs(config));
    complete
        .attributes
        .insert("response_text".to_string(), json!("synthetic ok"));
    vec![open, scheduler, prefill, complete]
}

fn base_event(
    config: &ProductObservabilityConfig,
    request_id: &str,
    phase: &str,
    event_kind: ProfileEventKind,
    timestamp: chrono::DateTime<Utc>,
) -> FerrumProfileEvent {
    FerrumProfileEvent {
        schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
        event_id: format!(
            "evt-product-{}-{phase}",
            entrypoint_label(config.entrypoint)
        ),
        request_id: request_id.to_string(),
        correlation_id: Some(format!(
            "corr-product-{}",
            entrypoint_label(config.entrypoint)
        )),
        entrypoint: config.entrypoint,
        backend: SYNTHETIC_BACKEND.to_string(),
        phase: phase.to_string(),
        event_kind,
        timestamp,
        status: ProfileStatus::Ok,
        model: Some(config.model.clone()),
        duration_us: None,
        memory: None,
        resource: None,
        error: None,
        replay: None,
        attributes: common_attrs(config),
    }
}

fn common_attrs(config: &ProductObservabilityConfig) -> BTreeMap<String, Value> {
    BTreeMap::from([
        (
            "profile_detail".to_string(),
            json!(config.profile_detail.as_str()),
        ),
        (
            "profile_sample_rate".to_string(),
            json!(config.profile_sample_rate),
        ),
        (
            "diagnostic_only".to_string(),
            json!(config.profile_detail.diagnostic_only()),
        ),
        ("l0_only".to_string(), json!(true)),
    ])
}

fn request_dump(
    config: &ProductObservabilityConfig,
    request_id: &str,
    replay_command: &str,
) -> serde_json::Value {
    let entrypoint = entrypoint_label(config.entrypoint);
    match config.entrypoint {
        ProfileEntrypoint::Run => json!({
            "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            "entrypoint": entrypoint,
            "request_id": request_id,
            "model": config.model,
            "backend": SYNTHETIC_BACKEND,
            "profile_detail": config.profile_detail.as_str(),
            "profile_sample_rate": config.profile_sample_rate,
            "l0_only": true,
            "sanitized": true,
            "prompt": "product observability wiring",
            "replay_command": replay_command
        }),
        ProfileEntrypoint::Serve => json!({
            "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            "entrypoint": entrypoint,
            "request_id": request_id,
            "model": config.model,
            "backend": SYNTHETIC_BACKEND,
            "profile_detail": config.profile_detail.as_str(),
            "profile_sample_rate": config.profile_sample_rate,
            "l0_only": true,
            "sanitized": true,
            "http": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "body": {
                    "model": config.model,
                    "messages": [{"role": "user", "content": "product observability wiring"}],
                    "stream": false
                }
            },
            "replay_command": replay_command
        }),
        other => json!({
            "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            "entrypoint": entrypoint_label(other),
            "request_id": request_id,
            "model": config.model,
            "backend": SYNTHETIC_BACKEND,
            "profile_detail": config.profile_detail.as_str(),
            "profile_sample_rate": config.profile_sample_rate,
            "l0_only": true,
            "sanitized": true,
            "replay_command": replay_command
        }),
    }
}

fn replay_command(config: &ProductObservabilityConfig) -> String {
    let mut parts = vec![
        "cargo".to_string(),
        "run".to_string(),
        "-p".to_string(),
        "ferrum-cli".to_string(),
        "--".to_string(),
        entrypoint_label(config.entrypoint).to_string(),
        SYNTHETIC_MODEL.to_string(),
        "--profile-detail".to_string(),
        config.profile_detail.as_str().to_string(),
        "--profile-sample-rate".to_string(),
        config.profile_sample_rate.to_string(),
    ];
    push_path_arg(&mut parts, "--profile-jsonl", config.profile_jsonl.as_ref());
    push_path_arg(
        &mut parts,
        "--memory-profile-jsonl",
        config.memory_profile_jsonl.as_ref(),
    );
    push_path_arg(
        &mut parts,
        "--scheduler-trace-jsonl",
        config.scheduler_trace_jsonl.as_ref(),
    );
    push_path_arg(
        &mut parts,
        "--request-dump-dir",
        config.request_dump_dir.as_ref(),
    );
    parts
        .iter()
        .map(|part| shell_quote(part))
        .collect::<Vec<_>>()
        .join(" ")
}

fn push_path_arg(parts: &mut Vec<String>, flag: &str, path: Option<&PathBuf>) {
    if let Some(path) = path {
        parts.push(flag.to_string());
        parts.push(path.to_string_lossy().to_string());
    }
}

fn write_profile_jsonl(path: &Path, events: &[FerrumProfileEvent]) -> Result<()> {
    if events.is_empty() {
        return Err(FerrumError::internal("profile event set must be non-empty"));
    }
    let mut body = String::new();
    for event in events {
        event.validate().map_err(|err| {
            FerrumError::internal(format!("invalid product observability event: {err}"))
        })?;
        body.push_str(&serde_json::to_string(event).map_err(|err| {
            FerrumError::serialization(format!("failed to serialize profile event: {err}"))
        })?);
        body.push('\n');
    }
    fs_write(path, body)
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    let body = serde_json::to_string_pretty(value)
        .map_err(|err| FerrumError::serialization(format!("failed to serialize JSON: {err}")))?;
    fs_write(path, format!("{body}\n"))
}

fn fs_create_dir_all(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .map_err(|err| FerrumError::io(format!("failed to create {}: {err}", path.display())))
}

fn fs_write(path: &Path, content: impl AsRef<[u8]>) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs_create_dir_all(parent)?;
    }
    fs::write(path, content)
        .map_err(|err| FerrumError::io(format!("failed to write {}: {err}", path.display())))
}

fn entrypoint_label(entrypoint: ProfileEntrypoint) -> &'static str {
    match entrypoint {
        ProfileEntrypoint::Run => "run",
        ProfileEntrypoint::Serve => "serve",
        ProfileEntrypoint::BenchServe => "bench_serve",
        ProfileEntrypoint::Synthetic => "synthetic",
    }
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_./:".contains(ch))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_product_observability_writes_profile_paths() {
        let root = std::env::temp_dir().join(format!(
            "ferrum-product-observability-{}",
            Uuid::new_v4().simple()
        ));
        let config = ProductObservabilityConfig::new(
            ProfileEntrypoint::Run,
            SYNTHETIC_MODEL,
            Some(&root.join("profile.jsonl")),
            ProfileDetailArg::Basic,
            Some(&root.join("memory.jsonl")),
            Some(&root.join("scheduler.jsonl")),
            Some(&root.join("request_dump")),
            1.0,
        );
        let written = write_synthetic_product_observability(&config).unwrap();
        assert_eq!(written.len(), 5);
        assert!(root.join("profile.jsonl").is_file());
        assert!(root.join("memory.jsonl").is_file());
        assert!(root.join("scheduler.jsonl").is_file());
        assert!(root.join("request_dump/request.json").is_file());
        assert!(root.join("request_dump/replay_command.txt").is_file());
        fs::remove_dir_all(root).ok();
    }
}
