//! Offline request replay bundle validation.

use crate::config::CliConfig;
use clap::Args;
use ferrum_types::{FerrumError, ProfileEntrypoint, Result, OBSERVABILITY_PROFILE_SCHEMA_VERSION};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};

const REQUIRED_BUNDLE_FILES: &[&str] = &[
    "request.json",
    "prompt_token_ids.json",
    "sampling_params.json",
    "runtime_effective_config.json",
    "backend_selection.json",
    "output_token_ids.json",
    "output_text.txt",
    "bad_output_scan.json",
    "replay.command.json",
];

#[derive(Args, Debug)]
pub struct ReplayBundleCommand {
    /// Request replay bundle directory.
    pub bundle_dir: PathBuf,

    /// Write an offline synthetic/no-weight replay artifact to this directory.
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,

    /// Print a JSON summary instead of a PASS line.
    #[arg(long)]
    pub json: bool,
}

pub async fn execute(cmd: ReplayBundleCommand, _config: CliConfig) -> Result<()> {
    let summary = replay_bundle(&cmd.bundle_dir, cmd.out.as_deref())?;
    if cmd.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&summary)
                .map_err(|err| FerrumError::serialization(err.to_string()))?
        );
    } else {
        println!(
            "FERRUM REPLAY BUNDLE PASS: {}",
            cmd.bundle_dir.to_string_lossy()
        );
    }
    Ok(())
}

fn replay_bundle(bundle_dir: &Path, out: Option<&Path>) -> Result<Value> {
    let validated = validate_bundle(bundle_dir)?;
    let generated = if let Some(out) = out {
        Some(write_offline_replay_artifact(out, validated.entrypoint)?)
    } else {
        None
    };
    Ok(json!({
        "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
        "status": "pass",
        "bundle_dir": bundle_dir.to_string_lossy(),
        "request_id": validated.request_id,
        "entrypoint": validated.entrypoint_label,
        "model": validated.model,
        "output_token_count": validated.output_token_count,
        "bad_output": validated.bad_output,
        "offline_replay_artifact": generated,
        "pass_line": format!("FERRUM REPLAY BUNDLE PASS: {}", bundle_dir.to_string_lossy())
    }))
}

#[derive(Debug)]
struct ValidatedBundle {
    request_id: String,
    entrypoint: ProfileEntrypoint,
    entrypoint_label: String,
    model: String,
    output_token_count: usize,
    bad_output: bool,
}

fn validate_bundle(bundle_dir: &Path) -> Result<ValidatedBundle> {
    if !bundle_dir.is_dir() {
        return Err(FerrumError::invalid_parameter(format!(
            "bundle dir does not exist or is not a directory: {}",
            bundle_dir.display()
        )));
    }
    for file in REQUIRED_BUNDLE_FILES {
        let path = bundle_dir.join(file);
        if !path.is_file() {
            return Err(FerrumError::invalid_parameter(format!(
                "missing replay bundle file: {}",
                path.display()
            )));
        }
    }

    let request = read_json(&bundle_dir.join("request.json"))?;
    let request_id = required_string(&request, "request_id", "request.json")?;
    if request.get("schema_version").and_then(Value::as_u64)
        != Some(OBSERVABILITY_PROFILE_SCHEMA_VERSION as u64)
    {
        return Err(FerrumError::invalid_parameter(
            "request.json schema_version mismatch",
        ));
    }
    if request.get("sanitized").and_then(Value::as_bool) != Some(true) {
        return Err(FerrumError::invalid_parameter(
            "request.json sanitized must be true",
        ));
    }
    let entrypoint_label = required_string(&request, "entrypoint", "request.json")?;
    let entrypoint = match entrypoint_label.as_str() {
        "run" => ProfileEntrypoint::Run,
        "serve" => ProfileEntrypoint::Serve,
        other => {
            return Err(FerrumError::invalid_parameter(format!(
                "unsupported replay entrypoint: {other}"
            )));
        }
    };
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    for file in [
        "prompt_token_ids.json",
        "sampling_params.json",
        "runtime_effective_config.json",
        "backend_selection.json",
        "output_token_ids.json",
        "bad_output_scan.json",
        "replay.command.json",
    ] {
        let value = read_json(&bundle_dir.join(file))?;
        let other_id = required_string(&value, "request_id", file)?;
        if other_id != request_id {
            return Err(FerrumError::invalid_parameter(format!(
                "{file} request_id mismatch: {other_id} != {request_id}"
            )));
        }
    }

    let output_tokens = read_json(&bundle_dir.join("output_token_ids.json"))?;
    let output_token_count = validate_token_ids(&output_tokens, "output_token_ids.json")?;
    let bad_scan = read_json(&bundle_dir.join("bad_output_scan.json"))?;
    let bad_output = bad_scan
        .get("bad_output")
        .and_then(Value::as_bool)
        .ok_or_else(|| FerrumError::invalid_parameter("bad_output_scan.bad_output missing"))?;
    let replay = read_json(&bundle_dir.join("replay.command.json"))?;
    let argv = replay
        .get("argv")
        .and_then(Value::as_array)
        .ok_or_else(|| FerrumError::invalid_parameter("replay.command argv missing"))?;
    if argv.is_empty() || !argv.iter().all(Value::is_string) {
        return Err(FerrumError::invalid_parameter(
            "replay.command argv must be a non-empty string array",
        ));
    }

    Ok(ValidatedBundle {
        request_id,
        entrypoint,
        entrypoint_label,
        model,
        output_token_count,
        bad_output,
    })
}

fn write_offline_replay_artifact(out: &Path, entrypoint: ProfileEntrypoint) -> Result<Value> {
    let profile = out.join("profile.jsonl");
    let memory = out.join("memory_profile.jsonl");
    let scheduler = out.join("scheduler_trace.jsonl");
    let request_dump = out.join("request_dump");
    let config = crate::observability_product::ProductObservabilityConfig::new(
        entrypoint,
        "synthetic/no-weight",
        Some(&profile),
        crate::observability_product::ProfileDetailArg::Basic,
        Some(&memory),
        Some(&scheduler),
        Some(&request_dump),
        1.0,
    );
    let written = crate::observability_product::write_synthetic_product_observability(&config)?;
    let summary = json!({
        "out": out.to_string_lossy(),
        "entrypoint": entrypoint.as_str(),
        "artifact_count": written.len(),
        "profile_jsonl": profile.to_string_lossy(),
        "request_dump_dir": request_dump.to_string_lossy()
    });
    fs::create_dir_all(out).map_err(|err| FerrumError::io(err.to_string()))?;
    fs::write(
        out.join("replay_bundle_summary.json"),
        serde_json::to_vec_pretty(&summary)
            .map_err(|err| FerrumError::serialization(err.to_string()))?,
    )
    .map_err(|err| FerrumError::io(err.to_string()))?;
    Ok(summary)
}

fn validate_token_ids(value: &Value, label: &str) -> Result<usize> {
    let token_ids = value
        .get("token_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| FerrumError::invalid_parameter(format!("{label}.token_ids missing")))?;
    if !token_ids
        .iter()
        .all(|item| item.as_u64().is_some_and(|token| token <= u32::MAX as u64))
    {
        return Err(FerrumError::invalid_parameter(format!(
            "{label}.token_ids must be non-negative u32 values"
        )));
    }
    let token_count = value
        .get("token_count")
        .and_then(Value::as_u64)
        .ok_or_else(|| FerrumError::invalid_parameter(format!("{label}.token_count missing")))?
        as usize;
    if token_count != token_ids.len() {
        return Err(FerrumError::invalid_parameter(format!(
            "{label}.token_count must match token_ids length"
        )));
    }
    Ok(token_count)
}

fn read_json(path: &Path) -> Result<Value> {
    let body = fs::read_to_string(path)
        .map_err(|err| FerrumError::io(format!("failed to read {}: {err}", path.display())))?;
    serde_json::from_str(&body).map_err(|err| {
        FerrumError::serialization(format!("failed to parse {}: {err}", path.display()))
    })
}

fn required_string(value: &Value, key: &str, label: &str) -> Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| FerrumError::invalid_parameter(format!("{label}.{key} must be a string")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn replay_bundle_validates_and_writes_offline_artifact() {
        let root = std::env::temp_dir().join(format!("ferrum-replay-bundle-{}", Uuid::new_v4()));
        let bundle = root.join("req-test");
        write_test_bundle(&bundle, "run");
        let out = root.join("offline");

        let summary = replay_bundle(&bundle, Some(&out)).unwrap();

        assert_eq!(summary["status"], "pass");
        assert_eq!(summary["entrypoint"], "run");
        assert!(out.join("profile.jsonl").is_file());
        assert!(out.join("request_dump").is_dir());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn replay_bundle_rejects_missing_required_file() {
        let root = std::env::temp_dir().join(format!("ferrum-replay-bundle-{}", Uuid::new_v4()));
        let bundle = root.join("req-test");
        write_test_bundle(&bundle, "serve");
        fs::remove_file(bundle.join("output_token_ids.json")).unwrap();

        let err = validate_bundle(&bundle).expect_err("missing output tokens should fail");

        assert!(err.to_string().contains("missing replay bundle file"));
        fs::remove_dir_all(root).ok();
    }

    fn write_test_bundle(bundle: &Path, entrypoint: &str) {
        fs::create_dir_all(bundle).unwrap();
        let request_id = "req-test";
        let common = json!({
            "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            "request_id": request_id,
            "sanitized": true
        });
        write_json(
            &bundle.join("request.json"),
            json!({
                "schema_version": OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                "request_id": request_id,
                "entrypoint": entrypoint,
                "model": "synthetic/no-weight",
                "backend": "synthetic",
                "sanitized": true
            }),
        );
        write_json(
            &bundle.join("prompt_token_ids.json"),
            json!({"schema_version": 1, "request_id": request_id, "token_ids": [1], "token_count": 1, "sanitized": true}),
        );
        write_json(
            &bundle.join("sampling_params.json"),
            json!({"schema_version": 1, "request_id": request_id, "sampling_params": {"max_tokens": 1}}),
        );
        write_json(
            &bundle.join("runtime_effective_config.json"),
            json!({"schema_version": 1, "request_id": request_id, "entrypoint": entrypoint, "sanitized": true}),
        );
        write_json(
            &bundle.join("backend_selection.json"),
            json!({"schema_version": 1, "request_id": request_id, "backend": "synthetic"}),
        );
        write_json(
            &bundle.join("output_token_ids.json"),
            json!({"schema_version": 1, "request_id": request_id, "token_ids": [2], "token_count": 1, "finish_reason": "stop"}),
        );
        write_json(
            &bundle.join("bad_output_scan.json"),
            json!({
                "schema_version": 1,
                "request_id": request_id,
                "bad_output": false,
                "reasons": [],
                "classified_output_sha256": "2689367b205c16ce32ed4200942b8b8b1e262dfc70d9bc9fbc77c49699a4f1df",
                "output_sha256": "dc51b8c96c2d745df3bd5590d990230a482fd247123599548e0632fdbf97fc22"
            }),
        );
        write_json(
            &bundle.join("replay.command.json"),
            json!({"schema_version": 1, "request_id": request_id, "entrypoint": entrypoint, "argv": ["ferrum", "replay-bundle", bundle], "command": "ferrum replay-bundle"}),
        );
        fs::write(bundle.join("output_text.txt"), "ok\n").unwrap();
        let _ = common;
    }

    fn write_json(path: &Path, value: Value) {
        fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    }
}
