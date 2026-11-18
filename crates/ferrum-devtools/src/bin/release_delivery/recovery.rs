//! Accept original local evidence after a failed artifact transfer, and identify
//! the independent frozen-tag Windows producer without changing product inputs.
use super::{ci_evidence, ci_get, installation, read, Backend, CudaVariant, Distributions};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const SM89_TEST: &str = "managed_fp8_projection_independent_oracle_shapes_and_retirement";
const CUDA_MODELS: &str = "Execute required local CUDA models from the staged archive";
const CUDA_NUMERICS: &str = "Execute SM89 FP8 numerical and retirement contracts";
const WINDOWS: &str = "Stage Windows x86_64 CUDA sm89";

#[derive(Default, Clone, Copy)]
pub(super) struct CiInputs<'a> {
    pub manual_local: bool,
    pub standalone_windows: Option<(u64, &'a str)>,
    pub version: &'a str,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LocalManifest {
    schema_version: u32,
    source_run_id: u64,
    candidate_sha: String,
    fp8: Fp8Execution,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Fp8Execution {
    test_name: String,
    checker_sha256: String,
    log_sha256: String,
    exit_code: i32,
}

fn verify_fp8(fp8: &Fp8Execution, checker: &Path, log: &Path) -> Result<(), String> {
    let actual = installation::sha256(checker)?;
    let checksum = fs::read_to_string(checker.with_file_name("sm89-fp8-runtime-check.sha256"))
        .map_err(|e| format!("read staged FP8 checker digest: {e}"))?;
    if fp8.exit_code != 0
        || fp8.test_name != SM89_TEST
        || fp8.checker_sha256 != actual
        || checksum.trim_end() != format!("{actual}  sm89-fp8-runtime-check")
        || fp8.log_sha256 != installation::sha256(log)?
    {
        return Err(
            "manual SM89 numerical evidence differs from the staged checker or failed execution"
                .into(),
        );
    }
    // A successful libtest exit also occurs for an empty filter. Require the
    // individual test result from the original harness as well as the exit code.
    let log = fs::read_to_string(log).map_err(|e| format!("read original FP8 runtime log: {e}"))?;
    if !log
        .lines()
        .any(|line| line.split_whitespace().eq(["test", SM89_TEST, "...", "ok"]))
    {
        return Err("original FP8 runtime log lacks the executed oracle/retirement test".into());
    }
    Ok(())
}

pub(super) fn read_local(
    directory: &Path,
    run: u64,
    candidate: &str,
    assets: &Distributions,
    reports: &mut Vec<Value>,
    installations: &mut Vec<installation::InstallationReport>,
) -> Result<Value, String> {
    let manifest: LocalManifest = read(&directory.join("manifest.json"))?;
    if manifest.schema_version != 1
        || manifest.source_run_id != run
        || manifest.candidate_sha != candidate
    {
        return Err(
            "manual local evidence belongs to another release candidate or source run".into(),
        );
    }
    for (backend, variant, lane) in [
        (Backend::Cuda, Some(CudaVariant::Sm89), "cuda"),
        (Backend::Cpu, None, "cpu"),
    ] {
        let asset = assets
            .get(&(backend, "x86_64-unknown-linux-gnu".into(), variant))
            .ok_or("missing frozen Linux distribution for manual model evidence")?;
        let install: installation::InstallationReport =
            read(&directory.join(format!("installation-{lane}.json")))?;
        if install.asset_name != asset.name || install.binary_sha256 != asset.binary_sha256 {
            return Err(format!(
                "manual {lane} installation belongs to a different distribution"
            ));
        }
        installation::verify_runtime(&install)?;
        installations.push(install);
        if backend == Backend::Cuda {
            let archive = asset
                .assets
                .iter()
                .find(|file| file.name == asset.name)
                .ok_or("missing SM89 archive")?;
            verify_fp8(
                &manifest.fp8,
                &archive.path.with_file_name("sm89-fp8-runtime-check"),
                &directory.join("sm89-fp8-runtime.log"),
            )?;
        }
        if reports
            .iter()
            .any(|report| report["binary_sha256"] == asset.binary_sha256)
        {
            return Err(format!(
                "manual {lane} reports compete with another evidence source"
            ));
        }
        let mut profiles = BTreeSet::new();
        let mut observed_sm89 = false;
        let lane_directory = directory.join(format!("model-{lane}"));
        for entry in
            fs::read_dir(&lane_directory).map_err(|e| format!("read {lane} model reports: {e}"))?
        {
            let entry = entry.map_err(|e| e.to_string())?;
            if !entry.file_type().map_err(|e| e.to_string())?.is_dir() {
                continue;
            }
            let path = entry.path().join("report.json");
            if !path.is_file() {
                continue;
            }
            let report: Value = read(&path)?;
            if report["binary_sha256"] != asset.binary_sha256 {
                return Err(format!("manual {lane} report used a different binary"));
            }
            if backend == Backend::Cuda {
                let startup: Vec<_> = report["cases"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|case| case["case"] == "serve-startup")
                    .collect();
                observed_sm89 |= !startup.is_empty();
                if startup.iter().any(|case| {
                    let hardware = &case["evidence"]["auto_config"]["hardware_capabilities"];
                    hardware["backend"] != "cuda" || hardware["compute_capability"] != "8.9"
                }) {
                    return Err(
                        "manual SM89 startup must observe CUDA compute capability 8.9".into(),
                    );
                }
            }
            let profile = report["profile_id"]
                .as_str()
                .ok_or("manual report has no profile ID")?;
            if !profiles.insert(profile.to_owned()) {
                return Err(format!("duplicate manual {lane} model report"));
            }
            reports.push(report);
        }
        if profiles.is_empty() {
            return Err(format!("missing original {lane} model reports"));
        }
        if backend == Backend::Cuda && !observed_sm89 {
            return Err("manual SM89 evidence has no observed startup device".into());
        }
        let execution: Value = read(&lane_directory.join("execution.json"))?;
        verify_execution(&execution, lane, &asset.binary_sha256, &profiles)?;
    }
    serde_json::to_value(&manifest).map_err(|e| e.to_string())
}

fn verify_execution(
    execution: &Value,
    lane: &str,
    binary: &str,
    profiles: &BTreeSet<String>,
) -> Result<(), String> {
    let completed: Vec<String> = serde_json::from_value(execution["completed_profiles"].clone())
        .map_err(|_| "invalid completed profile list")?;
    if execution["schema_version"] != 1
        || execution["binary_sha256"] != binary
        || execution["backend"] != lane
        || execution["status"] != "passed"
        || !execution["error"].is_null()
        || execution["completed_profiles"] != execution["profiles"]
        || completed.len() != profiles.len()
        || completed.into_iter().collect::<BTreeSet<_>>() != *profiles
        || !execution["runner_sha256"]
            .as_str()
            .is_some_and(|s| super::hex(s, 64))
    {
        return Err(format!(
            "manual {lane} runner did not execute every reported profile successfully"
        ));
    }
    Ok(())
}

fn job_summary(job: &Value) -> Value {
    let failed: Vec<_> = job["steps"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|step| matches!(step["conclusion"].as_str(), Some("failure" | "timed_out")))
        .map(|step| &step["name"])
        .collect();
    json!({"run_id":job["run_id"],"run_attempt":job["run_attempt"],"job_id":job["id"],
        "name":job["name"],"status":job["status"],"conclusion":job["conclusion"],"failed_steps":failed})
}

fn transport_failure(job: &Value) -> bool {
    let Some(steps) = job["steps"].as_array() else {
        return false;
    };
    let mut download = false;
    for step in steps {
        let Some(name) = step["name"].as_str() else {
            return false;
        };
        match step["conclusion"].as_str() {
            Some("failure" | "timed_out") if name.starts_with("Run actions/download-artifact@") => {
                download = true
            }
            // always() attempts an upload even when both result files are absent.
            Some("failure") if download && name.starts_with("Run actions/upload-artifact@") => (),
            Some("success" | "skipped") => (),
            _ => return false,
        }
    }
    download
        && [CUDA_MODELS, CUDA_NUMERICS].iter().all(|name| {
            let mut matches = steps.iter().filter(|step| step["name"] == *name);
            matches
                .next()
                .is_some_and(|step| step["conclusion"] == "skipped")
                && matches.next().is_none()
        })
}

pub(super) fn verify_transport(
    cuda: &BTreeMap<u64, Value>,
    cpu: &BTreeMap<u64, Value>,
) -> Result<Value, String> {
    let (attempt, last) = cuda.last_key_value().ok_or("no original SM89 job")?;
    let (cpu_attempt, last_cpu) = cpu.last_key_value().ok_or("no original CPU model job")?;
    if attempt != cpu_attempt
        || last["status"] != "completed"
        || !matches!(last["conclusion"].as_str(), Some("failure" | "timed_out"))
        || !transport_failure(last)
        || cuda.values().any(|job| {
            job["status"] != "completed"
                || (job["conclusion"] != "skipped" && !transport_failure(job))
        })
        || cpu
            .values()
            .any(|job| job["status"] != "completed" || job["conclusion"] != "skipped")
    {
        return Err("manual local evidence requires an artifact transfer failure before SM89 execution and its skipped CPU successor".into());
    }
    Ok(
        json!({"source":"manual_local", "original_cuda_job":job_summary(last), "original_cpu_job":job_summary(last_cpu)}),
    )
}

fn verify_standalone_run(
    run: &Value,
    source: &Value,
    id: u64,
    tag: &str,
    version: &str,
    repo: &str,
) -> Result<(), String> {
    let number = source["run_number"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("source run has no release candidate number")?;
    let repository_id = source["repository"]["id"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("missing release repository ID")?;
    if id == 0
        || source["id"] == id
        || tag != format!("v{version}-rc.{number}")
        || run["id"] != id
        || run["repository"]["id"] != repository_id
        || run["head_repository"]["id"] != repository_id
        || !run["repository"]["full_name"]
            .as_str()
            .is_some_and(|n| n.eq_ignore_ascii_case(repo))
        || run["head_sha"] != source["head_sha"]
        || run["head_branch"] != tag
        || run["event"] != "workflow_dispatch"
        || run["path"].as_str().and_then(|p| p.split('@').next())
            != Some(".github/workflows/release-windows.yml")
        || run["status"] != "completed"
        || run["conclusion"] != "success"
    {
        return Err("Windows recovery must use the successful same-repository staging workflow dispatched on the frozen RC tag".into());
    }
    Ok(())
}

pub(super) async fn verify_windows(
    client: &reqwest::Client,
    base: &str,
    repo: &str,
    source: &Value,
    id: u64,
    tag: &str,
    version: &str,
    attempt: u64,
    token: Option<&str>,
) -> Result<Value, String> {
    let url = format!("{base}/repos/{repo}/actions/runs/{id}");
    let run = ci_get(client, url.clone(), token).await?;
    verify_standalone_run(&run, source, id, tag, version, repo)?;
    let mut jobs = BTreeMap::new();
    let mut ids = std::collections::BTreeSet::new();
    for page in 1.. {
        let page = ci_get(
            client,
            format!("{url}/jobs?filter=all&per_page=100&page={page}"),
            token,
        )
        .await?;
        let batch = page["jobs"]
            .as_array()
            .ok_or("missing Windows staging job inventory")?;
        for job in batch {
            if !ids.insert(job["id"].as_u64().ok_or("Windows staging job lacks ID")?)
                || job["run_id"] != id
                || job["head_sha"] != run["head_sha"]
            {
                return Err("invalid or duplicated standalone staging job".into());
            }
            if job["name"] == WINDOWS
                && jobs
                    .insert(
                        job["run_attempt"]
                            .as_u64()
                            .ok_or("missing standalone attempt")?,
                        job.clone(),
                    )
                    .is_some()
            {
                return Err("ambiguous Windows CUDA producer in one attempt".into());
            }
        }
        if batch.len() < 100 {
            if page["total_count"].as_u64() != Some(ids.len() as u64) {
                return Err("incomplete Windows staging job inventory".into());
            }
            break;
        }
    }
    ci_evidence::verify_staging_job(&run, attempt, WINDOWS, &jobs)?;
    let steps = jobs[&attempt]["steps"]
        .as_array()
        .ok_or("missing Windows staging execution steps")?;
    for name in [
        "Verify Windows release candidate provenance",
        "Wrap the verified payload in the versioned current-user installer",
        "Upload Windows staged release assets",
    ] {
        let mut matching = steps.iter().filter(|s| s["name"] == name);
        if !matching
            .next()
            .is_some_and(|s| s["status"] == "completed" && s["conclusion"] == "success")
            || matching.next().is_some()
        {
            return Err(format!(
                "standalone Windows producer did not complete {name}"
            ));
        }
    }
    Ok(json!({"source":"standalone_windows_cuda","tag":tag,"job":job_summary(&jobs[&attempt])}))
}

#[cfg(test)]
#[path = "recovery_tests.rs"]
mod tests;
