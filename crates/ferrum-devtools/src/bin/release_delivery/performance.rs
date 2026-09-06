//! Same-host, fixed-source release latency comparison using the existing bench client.
use clap::Args;
use ferrum_bench_core::{
    release_regression::{
        performance::{ExpectedPerformanceRun, PerformancePolicy},
        ModelProfile,
    },
    BenchReport,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::time::Instant;

#[path = "performance/compare.rs"]
mod compare;
#[path = "performance/evidence.rs"]
mod evidence;
#[path = "performance/prepare.rs"]
mod prepare;
#[path = "performance/process.rs"]
mod process;
pub use prepare::{prepare, PerformancePrepareArgs};
#[path = "performance/source.rs"]
mod source;
pub(super) fn verify_evidence(
    expected: &ExpectedPerformanceRun,
    directory: &Path,
) -> Result<(), String> {
    evidence::verify_evidence(expected, directory)
}
use compare::{ComparisonStatus, Limits, Workload};

#[derive(Debug, Args, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceArgs {
    #[arg(long)]
    pub baseline_bin: PathBuf,
    #[arg(long)]
    pub baseline_sha256: String,
    #[arg(long)]
    pub baseline_version: String,
    #[arg(long)]
    pub candidate_bin: PathBuf,
    #[arg(long)]
    pub candidate_sha256: String,
    #[arg(long)]
    pub candidate_version: String,
    #[arg(long)]
    pub client_bin: PathBuf,
    #[arg(long)]
    pub client_sha256: String,
    #[arg(long)]
    pub client_version: String,
    /// A fixed local GGUF and the exact small tokenizer/metadata file set.
    #[arg(long)]
    pub source_manifest: PathBuf,
    /// Pre-registered performance task, including policy, source and binary identities.
    #[arg(long)]
    pub expected_task: PathBuf,
    #[arg(long)]
    pub input_tokens: u32,
    #[arg(long)]
    pub output_tokens: u32,
    #[arg(long)]
    pub measured_requests: u32,
    #[arg(long)]
    pub warmup_requests: u32,
    #[arg(long)]
    pub repeats: u32,
    #[arg(long)]
    pub seed: u64,
    #[arg(long)]
    pub max_model_len: u32,
    /// Fixed memory ceiling shared by both servers, instead of changing free-memory autosizing.
    #[arg(long)]
    pub runtime_memory_budget_bytes: u64,
    #[arg(long)]
    pub startup_timeout_secs: u64,
    #[arg(long)]
    pub request_timeout_secs: u64,
    #[arg(long)]
    pub task_timeout_secs: u64,
    /// Pre-registered acceptable increase, e.g. 0.10 means ten percent. No default.
    #[arg(long)]
    pub ttft_max_relative_increase: f64,
    #[arg(long)]
    pub tpot_max_relative_increase: f64,
    /// Must not exist. Includes original bench JSON, commands, health and process logs.
    #[arg(long)]
    pub report_dir: PathBuf,
}
impl PerformanceArgs {
    fn policy(&self) -> PerformancePolicy {
        PerformancePolicy {
            workload: self.workload(),
            limits: self.limits(),
            runtime_memory_budget_bytes: self.runtime_memory_budget_bytes,
            startup_timeout_secs: self.startup_timeout_secs,
            request_timeout_secs: self.request_timeout_secs,
            task_timeout_secs: self.task_timeout_secs,
        }
    }
    fn verify_expected(&self, expected: &ExpectedPerformanceRun) -> Result<(), String> {
        expected.validate()?;
        self.validate()?;
        if self.policy() != expected.policy
            || self.baseline_version != expected.baseline_version
            || self.candidate_version != expected.candidate_version
            || self.client_version != expected.client_version
            || self.baseline_sha256 != expected.baseline_sha256
            || self.candidate_sha256 != expected.candidate_sha256
            || self.client_sha256 != expected.client_sha256
        {
            return Err(
                "performance command differs from the pre-registered binary identities/policy"
                    .into(),
            );
        }
        Ok(())
    }

    fn workload(&self) -> Workload {
        Workload {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            measured_requests: self.measured_requests,
            warmup_requests: self.warmup_requests,
            repeats: self.repeats,
            seed: self.seed,
            max_model_len: self.max_model_len,
        }
    }
    fn limits(&self) -> Limits {
        Limits {
            ttft_max_relative_increase: self.ttft_max_relative_increase,
            tpot_max_relative_increase: self.tpot_max_relative_increase,
        }
    }
    fn validate(&self) -> Result<(), String> {
        if self.input_tokens == 0
            || self.output_tokens < 2
            || self.measured_requests == 0
            || self.warmup_requests == 0
            || self.repeats < 3
            || self.runtime_memory_budget_bytes == 0
            || self.startup_timeout_secs == 0
            || self.request_timeout_secs == 0
            || self.task_timeout_secs == 0
            || self
                .input_tokens
                .checked_add(self.output_tokens)
                .is_none_or(|n| n >= self.max_model_len)
            || self
                .measured_requests
                .checked_add(self.warmup_requests)
                .is_none()
            || self
                .measured_requests
                .checked_add(self.warmup_requests)
                .and_then(|n| n.checked_mul(self.repeats))
                .is_none()
        {
            return Err(
                "invalid performance workload, warmup, memory ceiling or time budget".into(),
            );
        }
        for limit in [
            self.ttft_max_relative_increase,
            self.tpot_max_relative_increase,
        ] {
            if !limit.is_finite() || limit <= 0.0 {
                return Err("performance limits must be finite and positive".into());
            }
        }
        for version in [
            &self.baseline_version,
            &self.candidate_version,
            &self.client_version,
        ] {
            let parsed = semver::Version::parse(version).map_err(|e| e.to_string())?;
            if parsed.to_string() != *version || !parsed.pre.is_empty() || !parsed.build.is_empty()
            {
                return Err("performance binary versions must be formal semantic versions".into());
            }
        }
        Ok(())
    }
}
fn remaining(deadline: Instant) -> Result<Duration, String> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or_else(|| "performance task deadline exceeded".into())
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    fs::write(
        path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write {}: {e}", path.display()))
}
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?)
        .map_err(|e| format!("parse {}: {e}", path.display()))
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Passed,
    Regressed,
    Inconclusive,
    Invalid,
}
#[derive(Debug, Serialize, Deserialize)]
struct PerformanceReport {
    pub schema_version: u32,
    pub status: Status,
    pub profile: Option<ModelProfile>,
    pub comparison: Option<compare::Comparison>,
    pub calibration_stable: Option<bool>,
    pub completed_phases: Vec<String>,
    pub elapsed_seconds: f64,
    pub error: Option<String>,
    pub release_approved: bool,
}
async fn run(
    args: &PerformanceArgs,
    report: &mut PerformanceReport,
    deadline: Instant,
) -> Result<(), String> {
    args.validate()?;
    let expected: ExpectedPerformanceRun = read_json(&args.expected_task)?;
    args.verify_expected(&expected)?;
    write_json(&args.report_dir.join("expected-task.json"), &expected)?;
    let source: source::SourceManifest = read_json(&args.source_manifest)?;
    report.profile = Some(expected.profile.clone());
    if source.identity() != Ok(expected.source.clone()) {
        return Err("performance source differs from prepare task".into());
    }
    source.validate(&expected.profile, deadline)?;
    write_json(&args.report_dir.join("source-manifest.json"), &source)?;
    // Keep symlinked weights outside uploaded reports. Dropping this owned directory
    // removes only the link and copied metadata, never the HF cache blob.
    let source_directory = tempfile::Builder::new()
        .prefix("ferrum-performance-model-")
        .tempdir()
        .map_err(|e| e.to_string())?;
    let bundle = source.prepare(&source_directory.path().join("model"), deadline)?;
    write_json(
        &args.report_dir.join("bound-source.json"),
        &json!({"gguf_canonical":fs::canonicalize(&source.gguf.path).map_err(|e|e.to_string())?,"bundle":bundle}),
    )?;
    let binaries = [
        (
            &args.baseline_bin,
            &args.baseline_sha256,
            &args.baseline_version,
            "baseline",
        ),
        (
            &args.candidate_bin,
            &args.candidate_sha256,
            &args.candidate_version,
            "candidate",
        ),
        (
            &args.client_bin,
            &args.client_sha256,
            &args.client_version,
            "client",
        ),
    ];
    let mut resolved = Vec::new();
    let mut binary_observations = serde_json::Map::new();
    for (binary, sha, version, label) in binaries {
        let requested = binary;
        let binary =
            fs::canonicalize(binary).map_err(|e| format!("resolve {label} binary: {e}"))?;
        let actual_sha256 = source::verify_file(&binary, None, sha, deadline)?;
        let log = args.report_dir.join(format!("{label}-version"));
        let mut command = process::clean_command(&binary, &args.report_dir);
        command.arg("--version");
        super::local::run_command(command, &log, remaining(deadline)?).await?;
        let actual =
            fs::read_to_string(log.with_extension("stdout.log")).map_err(|e| e.to_string())?;
        if actual.trim() != format!("ferrum {version}") {
            return Err(format!(
                "{label} binary version differs from registered version"
            ));
        }
        binary_observations.insert(label.into(),json!({"requested":requested,"canonical":binary,"sha256":actual_sha256,"version":version,"version_stdout":actual,"version_exit_code":0}));
        resolved.push(binary);
    }
    write_json(&args.report_dir.join("binaries.json"), &binary_observations)?;
    let mut reports = Vec::<BenchReport>::new();
    for (phase, binary, sha, version) in [
        (
            "baseline-a",
            &resolved[0],
            &args.baseline_sha256,
            &args.baseline_version,
        ),
        (
            "baseline-b",
            &resolved[0],
            &args.baseline_sha256,
            &args.baseline_version,
        ),
        (
            "candidate",
            &resolved[1],
            &args.candidate_sha256,
            &args.candidate_version,
        ),
    ] {
        let source_before = source.verify_bundle(&bundle, deadline)?;
        let server_before = source::verify_file(binary, None, sha, deadline)?;
        let client_before = source::verify_file(&resolved[2], None, &args.client_sha256, deadline)?;
        let phase_dir = args.report_dir.join(phase);
        fs::create_dir(&phase_dir).map_err(|e| e.to_string())?;
        process::phase(
            args,
            binary,
            &resolved[2],
            version,
            &bundle,
            &phase_dir,
            deadline,
        )
        .await?;
        let source_after = source.verify_bundle(&bundle, deadline)?;
        let server_after = source::verify_file(binary, None, sha, deadline)?;
        let client_after = source::verify_file(&resolved[2], None, &args.client_sha256, deadline)?;
        write_json(
            &phase_dir.join("checks.json"),
            &json!({"source_before":source_before,"source_after":source_after,
            "server_sha256_before":server_before,"server_sha256_after":server_after,
            "client_sha256_before":client_before,"client_sha256_after":client_after}),
        )?;
        reports.push(read_json(&phase_dir.join("bench.json"))?);
        report.completed_phases.push(phase.into());
        write_json(&args.report_dir.join("performance.json"), report)?;
        if reports.len() == 2 {
            let stable =
                compare::calibration(&reports[0], &reports[1], &args.workload(), &args.limits())?;
            report.calibration_stable = Some(stable);
            if !stable {
                report.status = Status::Inconclusive;
                return Err("baseline A/A noise exceeds the registered performance policy; candidate was not run".into());
            }
        }
    }
    remaining(deadline)?;
    let comparison = compare::compare(
        &reports[0],
        &reports[1],
        &reports[2],
        &args.workload(),
        &args.limits(),
    )?;
    report.status = match comparison.status {
        ComparisonStatus::Passed => Status::Passed,
        ComparisonStatus::Regressed => Status::Regressed,
        ComparisonStatus::Inconclusive => Status::Inconclusive,
    };
    report.comparison = Some(comparison);
    match report.status {
        Status::Passed => Ok(()),
        _ => Err(
            "same-host performance comparison did not pass; original observations retained".into(),
        ),
    }
}
pub async fn execute(mut args: PerformanceArgs) -> Result<(), String> {
    fs::create_dir(&args.report_dir)
        .map_err(|e| format!("create fresh performance report directory: {e}"))?;
    args.report_dir = fs::canonicalize(&args.report_dir).map_err(|e| e.to_string())?;
    write_json(&args.report_dir.join("inputs.json"), &args)?;
    let start = Instant::now();
    let mut report = PerformanceReport {
        schema_version: 1,
        status: Status::Invalid,
        profile: None,
        comparison: None,
        calibration_stable: None,
        completed_phases: vec![],
        elapsed_seconds: 0.0,
        error: Some("performance execution has not completed".into()),
        release_approved: false,
    };
    write_json(&args.report_dir.join("performance.json"), &report)?;
    let result = match start.checked_add(Duration::from_secs(args.task_timeout_secs)) {
        Some(deadline) => run(&args, &mut report, deadline).await,
        None => Err("performance task deadline is not representable".into()),
    };
    report.error = result.as_ref().err().cloned();
    report.elapsed_seconds = start.elapsed().as_secs_f64();
    write_json(&args.report_dir.join("performance.json"), &report)?;
    result
}
#[cfg(test)]
#[path = "performance/tests.rs"]
mod tests;
