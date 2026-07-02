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

pub struct ActualRunObservation {
    pub request_id: String,
    pub duration_us: u64,
    pub output_tokens: usize,
    pub chunk_count: usize,
    pub finish_reason: Option<String>,
    pub prompt_chars: usize,
    pub response_chars: usize,
    pub memory: Option<crate::memory_profile::ProcessMemoryObservation>,
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

    pub fn unified_product_profile_enabled(&self) -> bool {
        self.enabled()
            && (self.profile_detail != ProfileDetailArg::Off
                || self.memory_profile_jsonl.is_some()
                || self.scheduler_trace_jsonl.is_some()
                || self.request_dump_dir.is_some())
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

pub fn write_actual_run_observability(
    config: &ProductObservabilityConfig,
    observation: &ActualRunObservation,
) -> Result<Vec<PathBuf>> {
    if !config.enabled() {
        return Ok(Vec::new());
    }
    config.validate()?;
    let replay_command = replay_command(config);
    let events = actual_run_events(config, observation, &replay_command);
    write_actual_artifacts(config, &events, &observation.request_id, &replay_command)
}

pub fn write_actual_serve_startup_observability(
    config: &ProductObservabilityConfig,
    startup_duration_us: u64,
    startup_memory: Option<crate::memory_profile::ProcessMemoryObservation>,
) -> Result<Vec<PathBuf>> {
    if !config.unified_product_profile_enabled() {
        return Ok(Vec::new());
    }
    config.validate()?;
    let request_id = format!("serve-startup-{}", Uuid::new_v4().simple());
    let replay_command = replay_command(config);
    let events = actual_serve_startup_events(
        config,
        &request_id,
        startup_duration_us,
        startup_memory.as_ref(),
        &replay_command,
    );
    write_actual_artifacts(config, &events, &request_id, &replay_command)
}

fn write_actual_artifacts(
    config: &ProductObservabilityConfig,
    events: &[FerrumProfileEvent],
    request_id: &str,
    replay_command: &str,
) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    if let Some(path) = &config.profile_jsonl {
        write_profile_jsonl(path, events)?;
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
        append_profile_jsonl(path, &scheduler_events)?;
        written.push(path.clone());
    }
    if let Some(dir) = &config.request_dump_dir {
        fs_create_dir_all(dir)?;
        let request_path = dir.join("request.json");
        let replay_path = dir.join("replay_command.txt");
        write_json(
            &request_path,
            &actual_request_dump(config, request_id, replay_command),
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
    let open = resource_event(
        config,
        request_id,
        "request",
        request_id,
        "request_slot",
        "request_open",
        ResourceAction::RequestOpen,
        base,
        None,
        None,
        None,
        Some(1),
        None,
    );
    let reserve = resource_event(
        config,
        request_id,
        "request",
        request_id,
        "request_slot",
        "request_slot_reserve",
        ResourceAction::Reserve,
        base + Duration::microseconds(10),
        Some(1),
        Some(1),
        Some(0),
        Some(1),
        None,
    );
    let commit = resource_event(
        config,
        request_id,
        "request",
        request_id,
        "request_slot",
        "request_slot_commit",
        ResourceAction::Commit,
        base + Duration::microseconds(20),
        Some(1),
        Some(0),
        Some(1),
        Some(1),
        None,
    );

    let mut prefill = base_event(
        config,
        request_id,
        "synthetic_prefill",
        ProfileEventKind::TimedSpan,
        base + Duration::microseconds(30),
    );
    prefill.duration_us = Some(160);
    prefill.memory = Some(MemorySnapshot {
        scope: "process".to_string(),
        backend: Some(SYNTHETIC_BACKEND.to_string()),
        before_bytes: Some(2048),
        after_bytes: Some(2304),
        current_bytes: Some(2304),
        high_water_bytes: Some(2304),
        available_bytes: Some(1024 * 1024),
    });
    prefill.attributes.extend(common_attrs(config));
    prefill
        .attributes
        .insert("input_tokens".to_string(), json!(8));

    let release = resource_event(
        config,
        request_id,
        "request",
        request_id,
        "request_slot",
        "request_slot_release",
        ResourceAction::Release,
        base + Duration::microseconds(180),
        Some(1),
        Some(1),
        Some(0),
        Some(1),
        None,
    );

    let mut close = resource_event(
        config,
        request_id,
        "request",
        request_id,
        "request_slot",
        "request_close",
        ResourceAction::RequestClose,
        base + Duration::microseconds(190),
        None,
        None,
        None,
        Some(1),
        None,
    );
    close.status = if config.profile_detail.diagnostic_only() {
        ProfileStatus::DiagnosticOnly
    } else {
        ProfileStatus::Ok
    };
    close.replay = Some(ReplayReference {
        command: replay_command.to_string(),
        bundle_dir: config
            .request_dump_dir
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
    });
    close.attributes.extend(common_attrs(config));
    close
        .attributes
        .insert("response_text".to_string(), json!("synthetic ok"));
    vec![open, reserve, commit, prefill, release, close]
}

fn actual_run_events(
    config: &ProductObservabilityConfig,
    observation: &ActualRunObservation,
    replay_command: &str,
) -> Vec<FerrumProfileEvent> {
    let base = Utc::now();
    let open = actual_resource_event(
        config,
        &observation.request_id,
        "request",
        &observation.request_id,
        "request_slot",
        "request_open",
        ResourceAction::RequestOpen,
        base,
        None,
        None,
        None,
        Some(1),
        None,
    );
    let reserve = actual_resource_event(
        config,
        &observation.request_id,
        "request",
        &observation.request_id,
        "request_slot",
        "request_slot_reserve",
        ResourceAction::Reserve,
        base + Duration::microseconds(5),
        Some(1),
        Some(1),
        Some(0),
        Some(1),
        None,
    );
    let commit = actual_resource_event(
        config,
        &observation.request_id,
        "request",
        &observation.request_id,
        "request_slot",
        "request_slot_commit",
        ResourceAction::Commit,
        base + Duration::microseconds(10),
        Some(1),
        Some(0),
        Some(1),
        Some(1),
        None,
    );

    let mut generation = actual_base_event(
        config,
        &observation.request_id,
        "actual_run_generation",
        ProfileEventKind::TimedSpan,
        base + Duration::microseconds(20),
    );
    generation.duration_us = Some(observation.duration_us);
    attach_process_memory(
        &mut generation,
        observation.memory.as_ref(),
        "first_request_done",
    );
    generation.attributes.insert(
        "output_tokens".to_string(),
        json!(observation.output_tokens),
    );
    generation
        .attributes
        .insert("chunk_count".to_string(), json!(observation.chunk_count));
    generation.attributes.insert(
        "finish_reason".to_string(),
        json!(observation.finish_reason.as_deref().unwrap_or("unknown")),
    );

    let release = actual_resource_event(
        config,
        &observation.request_id,
        "request",
        &observation.request_id,
        "request_slot",
        "request_slot_release",
        ResourceAction::Release,
        base + Duration::microseconds(30),
        Some(1),
        Some(1),
        Some(0),
        Some(1),
        None,
    );

    let mut close = actual_resource_event(
        config,
        &observation.request_id,
        "request",
        &observation.request_id,
        "request_slot",
        "request_close",
        ResourceAction::RequestClose,
        base + Duration::microseconds(40),
        None,
        None,
        None,
        Some(1),
        None,
    );
    close.replay = Some(ReplayReference {
        command: replay_command.to_string(),
        bundle_dir: config
            .request_dump_dir
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
    });
    close
        .attributes
        .insert("prompt_chars".to_string(), json!(observation.prompt_chars));
    close.attributes.insert(
        "response_chars".to_string(),
        json!(observation.response_chars),
    );
    vec![open, reserve, commit, generation, release, close]
}

fn actual_serve_startup_events(
    config: &ProductObservabilityConfig,
    request_id: &str,
    startup_duration_us: u64,
    startup_memory: Option<&crate::memory_profile::ProcessMemoryObservation>,
    replay_command: &str,
) -> Vec<FerrumProfileEvent> {
    let base = Utc::now();
    let open = actual_resource_event(
        config,
        request_id,
        "server",
        request_id,
        "startup_slot",
        "server_startup_open",
        ResourceAction::RequestOpen,
        base,
        None,
        None,
        None,
        Some(1),
        None,
    );
    let reserve = actual_resource_event(
        config,
        request_id,
        "server",
        request_id,
        "startup_slot",
        "server_startup_reserve",
        ResourceAction::Reserve,
        base + Duration::microseconds(5),
        Some(1),
        Some(1),
        Some(0),
        Some(1),
        None,
    );
    let commit = actual_resource_event(
        config,
        request_id,
        "server",
        request_id,
        "startup_slot",
        "server_startup_commit",
        ResourceAction::Commit,
        base + Duration::microseconds(10),
        Some(1),
        Some(0),
        Some(1),
        Some(1),
        None,
    );
    let mut startup = actual_base_event(
        config,
        request_id,
        "actual_serve_startup",
        ProfileEventKind::TimedSpan,
        base + Duration::microseconds(20),
    );
    startup.duration_us = Some(startup_duration_us);
    attach_process_memory(&mut startup, startup_memory, "model_loaded");

    let release = actual_resource_event(
        config,
        request_id,
        "server",
        request_id,
        "startup_slot",
        "server_startup_release",
        ResourceAction::Release,
        base + Duration::microseconds(30),
        Some(1),
        Some(1),
        Some(0),
        Some(1),
        None,
    );
    let mut ready = actual_resource_event(
        config,
        request_id,
        "server",
        request_id,
        "startup_slot",
        "server_ready_for_requests",
        ResourceAction::RequestClose,
        base + Duration::microseconds(40),
        None,
        None,
        None,
        Some(1),
        None,
    );
    ready.replay = Some(ReplayReference {
        command: replay_command.to_string(),
        bundle_dir: config
            .request_dump_dir
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
    });
    vec![open, reserve, commit, startup, release, ready]
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

fn actual_base_event(
    config: &ProductObservabilityConfig,
    request_id: &str,
    phase: &str,
    event_kind: ProfileEventKind,
    timestamp: chrono::DateTime<Utc>,
) -> FerrumProfileEvent {
    let mut event = base_event(config, request_id, phase, event_kind, timestamp);
    event.backend = "actual".to_string();
    event.attributes = actual_attrs(config);
    event
}

#[allow(clippy::too_many_arguments)]
fn resource_event(
    config: &ProductObservabilityConfig,
    request_id: &str,
    owner_kind: &str,
    owner_id: &str,
    resource_kind: &str,
    phase: &str,
    action: ResourceAction,
    timestamp: chrono::DateTime<Utc>,
    amount: Option<i64>,
    before: Option<i64>,
    after: Option<i64>,
    capacity: Option<i64>,
    reason: Option<&str>,
) -> FerrumProfileEvent {
    let mut event = base_event(
        config,
        request_id,
        phase,
        ProfileEventKind::Resource,
        timestamp,
    );
    event.resource = Some(ResourceTraceEvent {
        owner_kind: owner_kind.to_string(),
        owner_id: owner_id.to_string(),
        resource_kind: resource_kind.to_string(),
        action,
        amount,
        before,
        after,
        capacity,
        reason: reason.map(str::to_string),
    });
    event
}

#[allow(clippy::too_many_arguments)]
fn actual_resource_event(
    config: &ProductObservabilityConfig,
    request_id: &str,
    owner_kind: &str,
    owner_id: &str,
    resource_kind: &str,
    phase: &str,
    action: ResourceAction,
    timestamp: chrono::DateTime<Utc>,
    amount: Option<i64>,
    before: Option<i64>,
    after: Option<i64>,
    capacity: Option<i64>,
    reason: Option<&str>,
) -> FerrumProfileEvent {
    let mut event = resource_event(
        config,
        request_id,
        owner_kind,
        owner_id,
        resource_kind,
        phase,
        action,
        timestamp,
        amount,
        before,
        after,
        capacity,
        reason,
    );
    event.backend = "actual".to_string();
    event.attributes = actual_attrs(config);
    event
}

fn attach_process_memory(
    event: &mut FerrumProfileEvent,
    observation: Option<&crate::memory_profile::ProcessMemoryObservation>,
    stage: &str,
) {
    if let Some(observation) = observation {
        event.memory = Some(observation.to_snapshot("process", Some("actual")));
        event
            .attributes
            .insert("memory_measurement".to_string(), json!("process_rss"));
        event
            .attributes
            .insert("memory_stage".to_string(), json!(stage));
        event.attributes.insert(
            "process_memory_source".to_string(),
            json!(observation.source),
        );
    } else {
        event.memory = Some(MemorySnapshot {
            scope: "process".to_string(),
            backend: Some("actual".to_string()),
            before_bytes: Some(0),
            after_bytes: Some(0),
            current_bytes: Some(0),
            high_water_bytes: Some(0),
            available_bytes: None,
        });
        event
            .attributes
            .insert("memory_measurement".to_string(), json!("not_collected"));
        event
            .attributes
            .insert("memory_stage".to_string(), json!(stage));
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

fn actual_attrs(config: &ProductObservabilityConfig) -> BTreeMap<String, Value> {
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
        ("l0_only".to_string(), json!(false)),
        ("actual_model_smoke".to_string(), json!(true)),
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

fn actual_request_dump(
    config: &ProductObservabilityConfig,
    request_id: &str,
    replay_command: &str,
) -> serde_json::Value {
    json!({
        "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
        "entrypoint": entrypoint_label(config.entrypoint),
        "request_id": request_id,
        "model": config.model,
        "backend": "actual",
        "profile_detail": config.profile_detail.as_str(),
        "profile_sample_rate": config.profile_sample_rate,
        "l0_only": false,
        "actual_model_smoke": true,
        "sanitized": true,
        "replay_command": replay_command
    })
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

fn append_profile_jsonl(path: &Path, events: &[FerrumProfileEvent]) -> Result<()> {
    if events.is_empty() {
        return Err(FerrumError::internal("profile event set must be non-empty"));
    }
    if let Some(parent) = path.parent() {
        fs_create_dir_all(parent)?;
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
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| {
            FerrumError::internal(format!(
                "failed to open profile JSONL {} for append: {err}",
                path.display()
            ))
        })?;
    file.write_all(body.as_bytes()).map_err(|err| {
        FerrumError::internal(format!(
            "failed to append profile JSONL {}: {err}",
            path.display()
        ))
    })
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
