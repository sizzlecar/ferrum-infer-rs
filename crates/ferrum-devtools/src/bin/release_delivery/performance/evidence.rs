//! Replay this one benchmark contract. The release gate independently binds
//! these artifacts to trusted CI execution; local model paths are never opened here.
use super::{
    compare, process, read_json,
    source::{Bundle, SourceManifest},
    PerformanceArgs, PerformanceReport, Status,
};
use ferrum_bench_core::{
    release_regression::performance::{ExpectedPerformanceRun, SourceIdentity},
    BenchReport,
};
use serde_json::Value;
use std::path::Path;

// ExpectedPerformanceRun binds this contract to a Metal worker. These are its
// POSIX wire paths, even when the evidence is replayed on Windows.
fn worker_path(path: &str) -> bool {
    path.starts_with('/')
        && !path.contains(['\\', '\0', '\r', '\n'])
        && (path == "/"
            || path[1..]
                .split('/')
                .all(|part| !matches!(part, "" | "." | "..")))
}

fn field_path(value: &Value, field: &str) -> Result<String, String> {
    let path = value[field]
        .as_str()
        .ok_or_else(|| format!("missing evidence path {field}"))?;
    if !worker_path(path) {
        return Err(format!(
            "evidence path {field} must be a normalized absolute POSIX worker path"
        ));
    }
    Ok(path.to_string())
}

fn worker_arguments(args: &[std::ffi::OsString]) -> Result<Value, String> {
    // serde's OsString representation is platform tagged. Reconstruct the Mac
    // worker's existing Unix byte representation, not the replay host's encoding.
    args.iter()
        .map(|arg| {
            let text = arg.to_str().ok_or("worker argument is not UTF-8")?;
            Ok(serde_json::json!({"Unix": text.as_bytes()}))
        })
        .collect::<Result<Vec<_>, String>>()
        .map(Value::Array)
}

fn require(condition: bool, error: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(error.into())
    }
}

pub(super) fn verify_evidence(
    expected: &ExpectedPerformanceRun,
    directory: &Path,
) -> Result<(), String> {
    expected.validate()?;
    let recorded_task: ExpectedPerformanceRun = read_json(&directory.join("expected-task.json"))?;
    require(
        &recorded_task == expected,
        "recorded performance task differs from prepared task",
    )?;
    let args: PerformanceArgs = read_json(&directory.join("inputs.json"))?;
    args.verify_expected(expected)?;
    let source: SourceManifest = read_json(&directory.join("source-manifest.json"))?;
    require(
        source.schema_version == 1
            && source.profile == expected.profile
            && source.identity()? == expected.source,
        "recorded model source differs from prepared content identity",
    )?;
    let report: PerformanceReport = read_json(&directory.join("performance.json"))?;
    require(
        report.schema_version == 1
            && matches!(report.status, Status::Passed)
            && report.error.is_none()
            && report.calibration_stable == Some(true)
            && report.profile.as_ref() == Some(&expected.profile)
            && !report.release_approved
            && report.elapsed_seconds.is_finite()
            && report.elapsed_seconds > 0.0
            && report.elapsed_seconds <= expected.policy.task_timeout_secs as f64
            && report.completed_phases == ["baseline-a", "baseline-b", "candidate"],
        "performance execution is failed, incomplete, over budget or mismatched",
    )?;
    let bound: Value = read_json(&directory.join("bound-source.json"))?;
    let bundle: Bundle =
        serde_json::from_value(bound["bundle"].clone()).map_err(|e| e.to_string())?;
    let tokenizer_dir = field_path(&bound["bundle"], "tokenizer_dir")?;
    let gguf = field_path(&bound["bundle"], "gguf")?;
    require(
        gguf == format!("{}/model.gguf", tokenizer_dir.trim_end_matches('/'))
            && field_path(&bound, "gguf_canonical").is_ok(),
        "invalid bound source paths",
    )?;
    // Paths name files on the execution worker. Their content is attested by the
    // checked before/after digests; trying to open them on Linux would be invalid.
    let binaries: Value = read_json(&directory.join("binaries.json"))?;
    let report_dir = args
        .report_dir
        .to_str()
        .filter(|path| worker_path(path))
        .ok_or("performance report directory must be a normalized absolute POSIX worker path")?;
    let mut canonical = Vec::new();
    for (label, sha, version, requested) in [
        (
            "baseline",
            &expected.baseline_sha256,
            &expected.baseline_version,
            &args.baseline_bin,
        ),
        (
            "candidate",
            &expected.candidate_sha256,
            &expected.candidate_version,
            &args.candidate_bin,
        ),
        (
            "client",
            &expected.client_sha256,
            &expected.client_version,
            &args.client_bin,
        ),
    ] {
        let observed = &binaries[label];
        let binary = field_path(observed, "canonical")?;
        require(
            observed["sha256"].as_str() == Some(sha.as_str())
                && observed["version"].as_str() == Some(version.as_str())
                && observed["version_exit_code"] == 0
                && observed["requested"]
                    == serde_json::to_value(requested).map_err(|e| e.to_string())?
                && observed["version_stdout"].as_str().map(str::trim)
                    == Some(format!("ferrum {version}").as_str()),
            "binary/version observation differs from prepared task",
        )?;
        let stdout = std::fs::read_to_string(directory.join(format!("{label}-version.stdout.log")))
            .map_err(|e| e.to_string())?;
        require(
            stdout.trim() == format!("ferrum {version}"),
            "version stdout differs from registered executable version",
        )?;
        canonical.push(binary);
    }
    let mut measurements = Vec::<BenchReport>::new();
    for (phase, server, sha, version) in [
        (
            "baseline-a",
            &canonical[0],
            &expected.baseline_sha256,
            &expected.baseline_version,
        ),
        (
            "baseline-b",
            &canonical[0],
            &expected.baseline_sha256,
            &expected.baseline_version,
        ),
        (
            "candidate",
            &canonical[1],
            &expected.candidate_sha256,
            &expected.candidate_version,
        ),
    ] {
        let local = directory.join(phase);
        let checks: Value = read_json(&local.join("checks.json"))?;
        for key in ["source_before", "source_after"] {
            let actual: SourceIdentity =
                serde_json::from_value(checks[key].clone()).map_err(|e| e.to_string())?;
            require(
                actual == expected.source,
                "executed source digests differ from prepared task",
            )?;
        }
        for key in ["server_sha256_before", "server_sha256_after"] {
            require(
                checks[key].as_str() == Some(sha.as_str()),
                "executed server digest differs from prepared task",
            )?;
        }
        for key in ["client_sha256_before", "client_sha256_after"] {
            require(
                checks[key].as_str() == Some(expected.client_sha256.as_str()),
                "executed client digest differs from prepared task",
            )?;
        }
        let health: Value = read_json(&local.join("health.json"))?;
        process::health_identity(&health, version, args.max_model_len)?;
        let execution: Value = read_json(&local.join("execution.json"))?;
        require(
            execution["server_pid"].as_u64().is_some_and(|pid| pid > 0)
                && execution["client_exit_code"] == 0
                && execution["client_cleanup_completed"] == true
                && execution["client_completed_successfully"] == true
                && execution["cleanup_completed"] == true
                && execution["error"].is_null()
                && execution["cleanup_error"].is_null(),
            "benchmark/server did not finish and clean up successfully",
        )?;
        let commands: Value = read_json(&local.join("commands.json"))?;
        let port = commands["port"]
            .as_u64()
            .and_then(|n| u16::try_from(n).ok())
            .filter(|n| *n > 0)
            .ok_or("missing actual server port")?;
        let server_args = process::server_arguments(&args, &bundle, port);
        let worker_report = format!("{}/{phase}/bench.json", report_dir.trim_end_matches('/'));
        let client_args =
            process::client_arguments(&args, &bundle, port, Path::new(&worker_report));
        require(
            commands["server"]["program"]
                == serde_json::to_value(server).map_err(|e| e.to_string())?
                && commands["client"]["program"]
                    == serde_json::to_value(&canonical[2]).map_err(|e| e.to_string())?
                && commands["server"]["args"] == worker_arguments(&server_args)?
                && commands["client"]["args"] == worker_arguments(&client_args)?,
            "actual server/client command does not implement prepared performance workload",
        )?;
        measurements.push(read_json(&local.join("bench.json"))?);
    }
    let recomputed = compare::compare(
        &measurements[0],
        &measurements[1],
        &measurements[2],
        &expected.policy.workload,
        &expected.policy.limits,
    )?;
    require(
        recomputed.status == compare::ComparisonStatus::Passed,
        "raw benchmark observations do not pass registered performance policy",
    )?;
    require(
        report
            .comparison
            .as_ref()
            .is_some_and(|saved| saved.matches_recomputed(&recomputed)),
        "performance summary differs from recomputed original measurements",
    )?;
    Ok(())
}
