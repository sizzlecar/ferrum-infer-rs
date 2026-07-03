use chrono::{Duration, Utc};
use ferrum_types::{
    FerrumError, FerrumProfileEvent, MemorySnapshot, ProfileEntrypoint, ProfileEventKind,
    ProfileStatus, ReplayReference, ResourceAction, ResourceTraceEvent, Result,
    OBSERVABILITY_PROFILE_SCHEMA_VERSION,
};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use uuid::Uuid;

const SYNTHETIC_MODEL: &str = "synthetic/no-weight";
const SYNTHETIC_BACKEND: &str = "synthetic";

pub fn write_observability_vertical_slice(
    entrypoint: ProfileEntrypoint,
    out_dir: &Path,
) -> Result<()> {
    let request_id = format!(
        "obs-{}-{}",
        entrypoint_label(entrypoint),
        Uuid::new_v4().simple()
    );
    let request_dump_dir = out_dir.join("request_dump");
    let bundle_dir = request_dump_dir.join(&request_id);
    let replay_args = replay_command_args(entrypoint, out_dir, &request_dump_dir);
    let replay_command = replay_command(&replay_args);
    fs_create_dir_all(&request_dump_dir)?;
    fs_create_dir_all(&bundle_dir)?;

    let request_dump_path = request_dump_dir.join("request.json");
    let replay_command_path = out_dir.join("replay_command.txt");
    let request_dump_replay_command_path = request_dump_dir.join("replay_command.txt");
    let profile_path = out_dir.join("profile.jsonl");
    let summary_path = out_dir.join("observability_profile_summary.json");

    let request = request_dump(entrypoint, &request_id, &replay_command);
    write_json(&request_dump_path, &request)?;
    fs_write(&replay_command_path, format!("{replay_command}\n"))?;
    fs_write(
        &request_dump_replay_command_path,
        format!("{replay_command}\n"),
    )?;
    write_replay_bundle(
        &bundle_dir,
        entrypoint,
        &request_id,
        &replay_command,
        &replay_args,
        &request,
        out_dir,
        &request_dump_dir,
    )?;

    let events = synthetic_events(
        entrypoint,
        &request_id,
        &replay_command,
        &request_dump_dir.to_string_lossy(),
    );
    write_profile_jsonl(&profile_path, &events)?;

    let summary = json!({
        "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
        "entrypoint": entrypoint_label(entrypoint),
        "backend": SYNTHETIC_BACKEND,
        "model": SYNTHETIC_MODEL,
        "l0_only": true,
        "status": "pass",
        "request_count": 1,
        "failed_count": 0,
        "corrupted_count": 0,
        "bad_text_count": 0,
        "oom_prevented_count": 0,
        "silent_oom_count": 0,
        "latency_us": {
            "p50": 320,
            "p95": 320,
            "p99": 320
        },
        "memory_high_water_bytes": 1536,
        "resource_leak_count": 0,
        "top_slow_phases": [
            {"phase": "synthetic_decode", "duration_us": 200},
            {"phase": "synthetic_prefill", "duration_us": 120}
        ],
        "first_failure_event": null,
        "profile_jsonl": profile_path.to_string_lossy(),
        "request_dump": request_dump_path.to_string_lossy(),
        "request_dump_dir": request_dump_dir.to_string_lossy(),
        "replay_bundle_dir": bundle_dir.to_string_lossy(),
        "replay_command": replay_command,
        "replay_command_path": replay_command_path.to_string_lossy()
    });
    write_json(&summary_path, &summary)?;
    Ok(())
}

fn synthetic_events(
    entrypoint: ProfileEntrypoint,
    request_id: &str,
    replay_command: &str,
    bundle_dir: &str,
) -> Vec<FerrumProfileEvent> {
    let base = Utc::now();
    let request_open = resource_event(
        entrypoint,
        request_id,
        "request_open",
        ResourceTraceEvent {
            owner_kind: "request".to_string(),
            owner_id: request_id.to_string(),
            resource_kind: "request_slot".to_string(),
            action: ResourceAction::RequestOpen,
            amount: None,
            before: None,
            after: None,
            capacity: Some(1),
            reason: None,
        },
        base,
    );
    let reserve = resource_event(
        entrypoint,
        request_id,
        "kv_reserve",
        ResourceTraceEvent {
            owner_kind: "request".to_string(),
            owner_id: request_id.to_string(),
            resource_kind: "kv_block".to_string(),
            action: ResourceAction::Reserve,
            amount: Some(1),
            before: Some(0),
            after: Some(1),
            capacity: Some(4),
            reason: None,
        },
        base + Duration::microseconds(10),
    );
    let commit = resource_event(
        entrypoint,
        request_id,
        "kv_commit",
        ResourceTraceEvent {
            owner_kind: "request".to_string(),
            owner_id: request_id.to_string(),
            resource_kind: "kv_block".to_string(),
            action: ResourceAction::Commit,
            amount: Some(1),
            before: Some(0),
            after: Some(1),
            capacity: Some(4),
            reason: None,
        },
        base + Duration::microseconds(15),
    );
    let prefill = timed_event(
        entrypoint,
        request_id,
        "synthetic_prefill",
        120,
        1024,
        1280,
        1280,
        attrs([("input_tokens", json!(8)), ("output_tokens", json!(0))]),
        base + Duration::microseconds(20),
    );
    let decode = timed_event(
        entrypoint,
        request_id,
        "synthetic_decode",
        200,
        1280,
        1536,
        1536,
        attrs([("input_tokens", json!(8)), ("output_tokens", json!(4))]),
        base + Duration::microseconds(140),
    );
    let release = resource_event(
        entrypoint,
        request_id,
        "kv_release",
        ResourceTraceEvent {
            owner_kind: "request".to_string(),
            owner_id: request_id.to_string(),
            resource_kind: "kv_block".to_string(),
            action: ResourceAction::Release,
            amount: Some(1),
            before: Some(1),
            after: Some(0),
            capacity: Some(4),
            reason: None,
        },
        base + Duration::microseconds(340),
    );
    let mut complete = base_event(
        entrypoint,
        request_id,
        "request_complete",
        ProfileEventKind::Instant,
        base + Duration::microseconds(350),
    );
    complete.status = ProfileStatus::DiagnosticOnly;
    complete.replay = Some(ReplayReference {
        command: replay_command.to_string(),
        bundle_dir: Some(bundle_dir.to_string()),
    });
    complete.attributes = attrs([
        ("l0_only", json!(true)),
        ("response_text", json!("synthetic ok")),
    ]);
    vec![
        request_open,
        reserve,
        commit,
        prefill,
        decode,
        release,
        complete,
    ]
}

fn base_event(
    entrypoint: ProfileEntrypoint,
    request_id: &str,
    phase: &str,
    event_kind: ProfileEventKind,
    timestamp: chrono::DateTime<Utc>,
) -> FerrumProfileEvent {
    FerrumProfileEvent {
        schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
        event_id: format!("evt-{}-{phase}", entrypoint_label(entrypoint)),
        request_id: request_id.to_string(),
        correlation_id: Some(format!("corr-{}", entrypoint_label(entrypoint))),
        entrypoint,
        backend: SYNTHETIC_BACKEND.to_string(),
        phase: phase.to_string(),
        event_kind,
        timestamp,
        status: ProfileStatus::Ok,
        model: Some(SYNTHETIC_MODEL.to_string()),
        duration_us: None,
        memory: None,
        resource: None,
        error: None,
        replay: None,
        attributes: BTreeMap::new(),
    }
}

fn resource_event(
    entrypoint: ProfileEntrypoint,
    request_id: &str,
    phase: &str,
    resource: ResourceTraceEvent,
    timestamp: chrono::DateTime<Utc>,
) -> FerrumProfileEvent {
    let mut event = base_event(
        entrypoint,
        request_id,
        phase,
        ProfileEventKind::Resource,
        timestamp,
    );
    event.resource = Some(resource);
    event
}

#[allow(clippy::too_many_arguments)]
fn timed_event(
    entrypoint: ProfileEntrypoint,
    request_id: &str,
    phase: &str,
    duration_us: u64,
    before_bytes: u64,
    after_bytes: u64,
    high_water_bytes: u64,
    attributes: BTreeMap<String, Value>,
    timestamp: chrono::DateTime<Utc>,
) -> FerrumProfileEvent {
    let mut event = base_event(
        entrypoint,
        request_id,
        phase,
        ProfileEventKind::TimedSpan,
        timestamp,
    );
    event.duration_us = Some(duration_us);
    event.memory = Some(MemorySnapshot {
        scope: "process".to_string(),
        backend: Some(SYNTHETIC_BACKEND.to_string()),
        before_bytes: Some(before_bytes),
        after_bytes: Some(after_bytes),
        current_bytes: Some(after_bytes),
        high_water_bytes: Some(high_water_bytes),
        available_bytes: Some(1024 * 1024),
    });
    event.attributes = attributes;
    event
}

fn request_dump(
    entrypoint: ProfileEntrypoint,
    request_id: &str,
    replay_command: &str,
) -> serde_json::Value {
    match entrypoint {
        ProfileEntrypoint::Run => json!({
            "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            "entrypoint": "run",
            "request_id": request_id,
            "model": SYNTHETIC_MODEL,
            "backend": SYNTHETIC_BACKEND,
            "l0_only": true,
            "sanitized": true,
            "prompt": "observability vertical slice",
            "sampling": {"max_tokens": 4, "temperature": 0.0},
            "replay_command": replay_command
        }),
        ProfileEntrypoint::Serve => json!({
            "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            "entrypoint": "serve",
            "request_id": request_id,
            "model": SYNTHETIC_MODEL,
            "backend": SYNTHETIC_BACKEND,
            "l0_only": true,
            "sanitized": true,
            "http": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "body": {
                    "model": SYNTHETIC_MODEL,
                    "messages": [{"role": "user", "content": "observability vertical slice"}],
                    "max_tokens": 4,
                    "temperature": 0.0
                }
            },
            "replay_command": replay_command
        }),
        other => json!({
            "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            "entrypoint": entrypoint_label(other),
            "request_id": request_id,
            "model": SYNTHETIC_MODEL,
            "backend": SYNTHETIC_BACKEND,
            "l0_only": true,
            "sanitized": true,
            "replay_command": replay_command
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_replay_bundle(
    bundle_dir: &Path,
    entrypoint: ProfileEntrypoint,
    request_id: &str,
    replay_command: &str,
    replay_args: &[String],
    request: &serde_json::Value,
    out_dir: &Path,
    request_dump_dir: &Path,
) -> Result<()> {
    let output_text = "synthetic ok";
    let files = [
        ("request.json", request.clone()),
        (
            "prompt_token_ids.json",
            json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "model": SYNTHETIC_MODEL,
                "tokenizer_or_model": SYNTHETIC_MODEL,
                "token_ids": [101, 202, 303, 404],
                "token_count": 4,
                "sanitized": true
            }),
        ),
        (
            "sampling_params.json",
            json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "sampling_params": {"max_tokens": 4, "temperature": 0.0},
                "unavailable_reason": null
            }),
        ),
        (
            "runtime_effective_config.json",
            json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "entrypoint": entrypoint_label(entrypoint),
                "profile_detail": "basic",
                "profile_sample_rate": 1.0,
                "profile_jsonl": out_dir.join("profile.jsonl").to_string_lossy(),
                "memory_profile_jsonl": out_dir.join("memory_profile.jsonl").to_string_lossy(),
                "scheduler_trace_jsonl": out_dir.join("scheduler_trace.jsonl").to_string_lossy(),
                "request_dump_dir": request_dump_dir.to_string_lossy(),
                "sanitized": true
            }),
        ),
        (
            "backend_selection.json",
            json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "backend": SYNTHETIC_BACKEND,
                "model": SYNTHETIC_MODEL,
                "l0_only": true
            }),
        ),
        (
            "output_token_ids.json",
            json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "token_ids": [909, 808],
                "token_count": 2,
                "finish_reason": "stop"
            }),
        ),
        (
            "bad_output_scan.json",
            json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "bad_output": false,
                "bad_text_count": 0,
                "reasons": [],
                "first_bad_text_span": null,
                "failure_kind": null,
                "output_chars": output_text.chars().count(),
                "output_sha256": sha256_hex(output_text.as_bytes())
            }),
        ),
        (
            "replay.command.json",
            json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "entrypoint": entrypoint_label(entrypoint),
                "command": replay_command,
                "argv": replay_args,
                "bundle_dir": bundle_dir.to_string_lossy(),
                "sanitized": true
            }),
        ),
    ];
    for (name, value) in files {
        write_json(&bundle_dir.join(name), &value)?;
    }
    fs_write(
        bundle_dir.join("output_text.txt").as_path(),
        format!("{output_text}\n"),
    )?;
    Ok(())
}

fn replay_command_args(
    entrypoint: ProfileEntrypoint,
    out_dir: &Path,
    request_dump_dir: &Path,
) -> Vec<String> {
    let subcommand = match entrypoint {
        ProfileEntrypoint::Run => "run",
        ProfileEntrypoint::Serve => "serve",
        ProfileEntrypoint::BenchServe => "bench-serve",
        ProfileEntrypoint::Synthetic => "run",
    };
    vec![
        "cargo".to_string(),
        "run".to_string(),
        "-p".to_string(),
        "ferrum-cli".to_string(),
        "--".to_string(),
        subcommand.to_string(),
        SYNTHETIC_MODEL.to_string(),
        "--profile-detail".to_string(),
        "basic".to_string(),
        "--profile-sample-rate".to_string(),
        "1".to_string(),
        "--profile-jsonl".to_string(),
        out_dir.join("profile.jsonl").to_string_lossy().to_string(),
        "--memory-profile-jsonl".to_string(),
        out_dir
            .join("memory_profile.jsonl")
            .to_string_lossy()
            .to_string(),
        "--scheduler-trace-jsonl".to_string(),
        out_dir
            .join("scheduler_trace.jsonl")
            .to_string_lossy()
            .to_string(),
        "--request-dump-dir".to_string(),
        request_dump_dir.to_string_lossy().to_string(),
    ]
}

fn replay_command(args: &[String]) -> String {
    args.iter()
        .map(|part| shell_quote(part))
        .collect::<Vec<_>>()
        .join(" ")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
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

fn entrypoint_label(entrypoint: ProfileEntrypoint) -> &'static str {
    match entrypoint {
        ProfileEntrypoint::Run => "run",
        ProfileEntrypoint::Serve => "serve",
        ProfileEntrypoint::BenchServe => "bench_serve",
        ProfileEntrypoint::Synthetic => "synthetic",
    }
}

fn attrs<const N: usize>(entries: [(&str, Value); N]) -> BTreeMap<String, Value> {
    entries
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

fn write_profile_jsonl(path: &Path, events: &[FerrumProfileEvent]) -> Result<()> {
    let mut body = String::new();
    for event in events {
        event.validate().map_err(|err| {
            FerrumError::internal(format!(
                "invalid observability vertical slice event {}: {err}",
                event.event_id
            ))
        })?;
        body.push_str(&serde_json::to_string(event).map_err(|err| {
            FerrumError::serialization(format!(
                "failed to serialize observability event {}: {err}",
                event.event_id
            ))
        })?);
        body.push('\n');
    }
    fs_write(path, body)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_required_run_artifacts() {
        let root = std::env::temp_dir().join(format!(
            "ferrum-observability-vertical-slice-{}",
            Uuid::new_v4().simple()
        ));
        write_observability_vertical_slice(ProfileEntrypoint::Run, &root).unwrap();
        assert!(root.join("profile.jsonl").is_file());
        assert!(root.join("request_dump/request.json").is_file());
        assert!(root.join("replay_command.txt").is_file());
        assert!(root.join("observability_profile_summary.json").is_file());
        let bundle_dir = fs::read_dir(root.join("request_dump"))
            .unwrap()
            .flatten()
            .find_map(|entry| entry.path().is_dir().then_some(entry.path()))
            .expect("request replay bundle directory should exist");
        for name in [
            "request.json",
            "prompt_token_ids.json",
            "sampling_params.json",
            "runtime_effective_config.json",
            "backend_selection.json",
            "output_token_ids.json",
            "output_text.txt",
            "bad_output_scan.json",
            "replay.command.json",
        ] {
            assert!(bundle_dir.join(name).is_file(), "missing {name}");
        }
        let replay: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(bundle_dir.join("replay.command.json")).unwrap(),
        )
        .unwrap();
        let argv = replay["argv"].as_array().unwrap();
        assert!(argv.iter().any(|part| part == "--request-dump-dir"));
        let profile = fs::read_to_string(root.join("profile.jsonl")).unwrap();
        assert!(profile.contains("\"entrypoint\":\"run\""));
        assert!(profile.contains("\"replay\""));
        fs::remove_dir_all(root).ok();
    }
}
