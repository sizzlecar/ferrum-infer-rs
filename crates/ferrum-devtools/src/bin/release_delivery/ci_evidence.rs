//! Accept raw GPU observations only from the actual successful CI producer.
//! Artifact JSON describes observations; GitHub supplies execution provenance.
use super::super::{performance, submission};
use chrono::{DateTime, FixedOffset};
use ferrum_bench_core::release_regression::{
    performance::{
        performance_check_descriptors, performance_task_schedule, ExpectedPerformanceRun,
        PerformancePolicy,
    },
    submission::SubmissionConfig,
    Behavior, EvidenceLayer, Obligation, Plan,
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Cursor, Read},
    path::{Component, Path},
    time::Duration,
};

const SUBMISSION_JOB: &str = "Quality / GPU runtime (metal)";
const SUBMISSION_EXECUTE: &str = "Require MetalContext submission and completion outputs";
const SUBMISSION_UPLOAD: &str = "Save raw numerical evidence";
const PERFORMANCE_JOB: &str = "Metal release models";
const PERFORMANCE_PREPARE: &str = "Prepare registered performance tasks";
const PERFORMANCE_REGISTER: &str = "Save registered performance tasks";
const PERFORMANCE_EXECUTE: &str = "Execute registered performance comparison";
const PERFORMANCE_UPLOAD: &str = "Save Metal release evidence";
const BASELINE_ARCHIVE: &str = "ferrum-macos-aarch64.tar.gz";
const ZIP_LIMIT: u64 = 64 * 1024 * 1024;
const EXTRACT_LIMIT: u64 = 128 * 1024 * 1024;
const FILE_LIMIT: u64 = 16 * 1024 * 1024;
const JSON_LIMIT: u64 = 4 * 1024 * 1024;

#[derive(Default)]
pub(super) struct CiEvidence {
    submission: Option<submission::VerifiedSubmission>,
    // Keep the actual verified obligation, so evidence cannot be rebound later.
    performance: BTreeMap<usize, Obligation>,
}
impl CiEvidence {
    pub(super) fn covers(&self, index: usize, obligation: &Obligation) -> bool {
        match obligation.layer {
            EvidenceLayer::BackendNumerics => self.submission.as_ref().is_some_and(|verified| {
                obligation.behavior == Behavior::SubmissionCompletion
                    && obligation.scope == verified.scope()
                    && obligation.entrypoints.is_empty()
                    && obligation.checkers == [verified.checker_id()]
            }),
            EvidenceLayer::Performance => self.performance.get(&index) == Some(obligation),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Origin {
    schema_version: u32,
    run_id: u64,
    run_attempt: u64,
    head_sha: String,
    job_id: u64,
}
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct Step {
    name: String,
    number: u64,
    status: String,
    conclusion: Option<String>,
    started_at: Option<String>,
    completed_at: Option<String>,
}
#[derive(Debug, Clone, Deserialize)]
struct Job {
    id: u64,
    run_id: u64,
    run_attempt: u64,
    head_sha: String,
    name: String,
    status: String,
    conclusion: Option<String>,
    started_at: Option<String>,
    completed_at: Option<String>,
    steps: Vec<Step>,
}
fn timestamp(value: &str) -> Result<DateTime<FixedOffset>, String> {
    DateTime::parse_from_rfc3339(value).map_err(|_| "CI timestamp is invalid".into())
}
fn interval(start: Option<&str>, end: Option<&str>) -> Result<(i64, i64), String> {
    let start = timestamp(start.ok_or("CI execution omitted its start time")?)?;
    let end = timestamp(end.ok_or("CI execution omitted its completion time")?)?;
    if end < start {
        return Err("CI execution completion precedes start".into());
    }
    // Jobs/steps expose whole seconds, artifacts expose milliseconds. Compare
    // at the coarser API precision, without inventing a time grace period.
    Ok((start.timestamp(), end.timestamp()))
}
fn step<'a>(job: &'a Job, name: &str) -> Result<&'a Step, String> {
    let mut matches = job.steps.iter().filter(|step| step.name == name);
    let found = matches
        .next()
        .ok_or_else(|| format!("CI producer omitted step {name}"))?;
    if matches.next().is_some()
        || found.status != "completed"
        || found.conclusion.as_deref() != Some("success")
        || found.number == 0
    {
        return Err(format!(
            "CI producer step {name} is ambiguous or unsuccessful"
        ));
    }
    let (start, end) = interval(found.started_at.as_deref(), found.completed_at.as_deref())?;
    let (job_start, job_end) = interval(job.started_at.as_deref(), job.completed_at.as_deref())?;
    if start < job_start || end > job_end {
        return Err("CI step is outside its producer execution".into());
    }
    Ok(found)
}
fn ordered(job: &Job, first: &str, second: &str) -> Result<(), String> {
    let first = step(job, first)?;
    let second = step(job, second)?;
    if first.number >= second.number
        || timestamp(first.completed_at.as_deref().unwrap())?
            > timestamp(second.started_at.as_deref().unwrap())?
    {
        return Err("CI producer steps are out of order".into());
    }
    Ok(())
}

/// A failed-jobs retry can clone a successful job under a new ID/attempt while
/// preserving its original execution timestamps. Such a clone is reusable;
/// any actual later execution replaces the old evidence, even if it failed.
fn producer(
    origin: &Origin,
    original: &[Job],
    all: &[Job],
    run_id: u64,
    candidate: &str,
    current_attempt: u64,
    name: &str,
) -> Result<Job, String> {
    if origin.schema_version != 1
        || origin.run_id != run_id
        || origin.head_sha != candidate
        || origin.run_attempt == 0
        || origin.run_attempt > current_attempt
        || origin.job_id == 0
    {
        return Err("artifact origin does not identify this CI candidate/attempt".into());
    }
    let mut originals = original.iter().filter(|job| job.name == name);
    let job = originals
        .next()
        .ok_or("artifact producer is absent from its original attempt")?;
    if originals.next().is_some()
        || job.id != origin.job_id
        || job.run_attempt != origin.run_attempt
        || job.run_id != run_id
        || job.head_sha != candidate
        || job.status != "completed"
        || job.conclusion.as_deref() != Some("success")
    {
        return Err("artifact's original producer did not succeed unambiguously".into());
    }
    interval(job.started_at.as_deref(), job.completed_at.as_deref())?;
    let mut attempts = BTreeSet::new();
    for later in all.iter().filter(|later| later.name == name) {
        if later.run_id != run_id
            || later.head_sha != candidate
            || later.run_attempt == 0
            || later.run_attempt > current_attempt
            || !attempts.insert(later.run_attempt)
        {
            return Err(
                "CI producer inventory is ambiguous or belongs to another candidate".into(),
            );
        }
        if later.run_attempt < origin.run_attempt {
            continue;
        }
        if later.status != "completed"
            || later.conclusion.as_deref() != Some("success")
            || later.started_at != job.started_at
            || later.completed_at != job.completed_at
            || later.steps != job.steps
        {
            return Err("a later producer execution supersedes this artifact; require its new successful evidence".into());
        }
    }
    if attempts
        .last()
        .is_none_or(|attempt| *attempt < origin.run_attempt)
    {
        return Err("current CI inventory omitted the artifact producer".into());
    }
    Ok(job.clone())
}

fn artifact_provenance(
    metadata: &Value,
    run: &Value,
    run_id: u64,
    candidate: &str,
) -> Result<(), String> {
    if metadata["id"].as_u64().is_none_or(|id| id == 0)
        || metadata["expired"].as_bool() != Some(false)
        || metadata["workflow_run"]["id"].as_u64() != Some(run_id)
        || metadata["workflow_run"]["head_sha"].as_str() != Some(candidate)
        || run["repository"]["id"].as_u64().is_none()
        || run["head_repository"]["id"].as_u64().is_none()
        || metadata["workflow_run"]["repository_id"] != run["repository"]["id"]
        || metadata["workflow_run"]["head_repository_id"] != run["head_repository"]["id"]
        || metadata["size_in_bytes"]
            .as_u64()
            .is_none_or(|size| size == 0 || size > ZIP_LIMIT)
    {
        return Err(
            "artifact is expired, oversized, or from another repository/run/candidate".into(),
        );
    }
    Ok(())
}
fn uploaded_during(metadata: &Value, job: &Job, execute: &str, upload: &str) -> Result<(), String> {
    ordered(job, execute, upload)?;
    let upload = step(job, upload)?;
    let (start, end) = interval(upload.started_at.as_deref(), upload.completed_at.as_deref())?;
    let created = timestamp(
        metadata["created_at"]
            .as_str()
            .ok_or("artifact creation time missing")?,
    )?;
    let updated = timestamp(
        metadata["updated_at"]
            .as_str()
            .ok_or("artifact update time missing")?,
    )?;
    if created > updated || created.timestamp() < start || updated.timestamp() > end {
        return Err(
            "artifact was not created and finalized during its successful producer upload".into(),
        );
    }
    Ok(())
}
fn preregistered(metadata: &Value, job: &Job) -> Result<(), String> {
    uploaded_during(metadata, job, PERFORMANCE_PREPARE, PERFORMANCE_REGISTER)?;
    ordered(job, PERFORMANCE_REGISTER, PERFORMANCE_EXECUTE)?;
    let execution = step(job, PERFORMANCE_EXECUTE)?;
    // Step numbers and nonoverlapping successful execution intervals establish
    // ordering even when GitHub rounds both boundaries to the same second.
    if timestamp(
        metadata["updated_at"]
            .as_str()
            .ok_or("registration finalization time missing")?,
    )?
    .timestamp()
        > timestamp(execution.started_at.as_deref().unwrap())?.timestamp()
    {
        return Err(
            "performance task artifact was not finalized before measurement started".into(),
        );
    }
    Ok(())
}

struct Github {
    client: reqwest::Client,
    repo: String,
    token: Option<String>,
}
impl Github {
    async fn bytes(&self, path: &str, limit: u64) -> Result<Vec<u8>, String> {
        self.response_bytes(path, limit, "application/vnd.github+json")
            .await
    }
    async fn response_bytes(
        &self,
        path: &str,
        limit: u64,
        accept: &str,
    ) -> Result<Vec<u8>, String> {
        let mut request = self
            .client
            .get(format!("https://api.github.com/repos/{}/{path}", self.repo))
            .header("Accept", accept)
            .header("X-GitHub-Api-Version", "2022-11-28");
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let mut response = request
            .send()
            .await
            .map_err(|_| "CI evidence request failed")?;
        if !response.status().is_success() {
            return Err(format!(
                "CI evidence request returned HTTP {}",
                response.status()
            ));
        }
        if response.content_length().is_some_and(|size| size > limit) {
            return Err("CI evidence exceeds the download limit".into());
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| "CI evidence download interrupted")?
        {
            if bytes.len() as u64 + chunk.len() as u64 > limit {
                return Err("CI evidence exceeds the download limit".into());
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }
    async fn json(&self, path: &str) -> Result<Value, String> {
        serde_json::from_slice(&self.bytes(path, JSON_LIMIT).await?)
            .map_err(|_| "CI evidence metadata is not JSON".into())
    }
    async fn inventory(&self, path: &str, field: &str) -> Result<Vec<Value>, String> {
        let mut output = Vec::new();
        let mut ids = BTreeSet::new();
        let mut total = None;
        for page in 1..=100 {
            let separator = if path.contains('?') { '&' } else { '?' };
            let value = self
                .json(&format!("{path}{separator}per_page=100&page={page}"))
                .await?;
            let count = value["total_count"]
                .as_u64()
                .ok_or("CI inventory omitted total_count")?;
            if count > 10000 || total.is_some_and(|old| old != count) {
                return Err("CI inventory changed or exceeded its limit".into());
            }
            total = Some(count);
            let rows = value[field].as_array().ok_or("CI inventory omitted rows")?;
            for row in rows {
                let id = row["id"]
                    .as_u64()
                    .filter(|id| *id != 0)
                    .ok_or("CI inventory row omitted ID")?;
                if !ids.insert(id) {
                    return Err("CI inventory repeats an ID".into());
                }
                output.push(row.clone());
            }
            if rows.len() < 100 || output.len() as u64 == count {
                if output.len() as u64 != count {
                    return Err("CI inventory is incomplete".into());
                }
                return Ok(output);
            }
        }
        Err("CI inventory pagination exceeded its bound".into())
    }
    async fn jobs(&self, path: &str) -> Result<Vec<Job>, String> {
        self.inventory(path, "jobs")
            .await?
            .into_iter()
            .map(|value| serde_json::from_value(value).map_err(|e| format!("CI job metadata: {e}")))
            .collect()
    }
}

fn digest_matches(bytes: &[u8], metadata: &Value) -> Result<(), String> {
    let digest = metadata["digest"]
        .as_str()
        .and_then(|digest| digest.strip_prefix("sha256:"))
        .filter(|digest| super::hex(digest, 64))
        .ok_or("GitHub asset/artifact has no SHA-256 digest")?;
    if format!("{:x}", Sha256::digest(bytes)) != digest {
        return Err("downloaded evidence differs from GitHub's immutable digest".into());
    }
    Ok(())
}
fn unpack(bytes: &[u8]) -> Result<tempfile::TempDir, String> {
    if bytes.len() as u64 > ZIP_LIMIT {
        return Err("artifact ZIP exceeds its bound".into());
    }
    let mut archive =
        zip::ZipArchive::new(Cursor::new(bytes)).map_err(|e| format!("artifact ZIP: {e}"))?;
    if archive.len() > 4096 {
        return Err("artifact contains too many entries".into());
    }
    let directory = tempfile::tempdir().map_err(|e| e.to_string())?;
    let mut total = 0u64;
    let mut names = BTreeSet::new();
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|e| format!("artifact entry: {e}"))?;
        let relative = entry
            .enclosed_name()
            .ok_or("artifact path escapes extraction directory")?;
        // zip 7.2's enclosed_name can strip an initial root/drive prefix.
        // Reject the original archive spelling before using its derived path.
        if Path::new(entry.name())
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
            || relative.as_os_str().is_empty()
            || entry.name().contains('\\')
            || entry.name().as_bytes().get(1) == Some(&b':')
            || entry.is_symlink()
            || entry
                .unix_mode()
                .is_some_and(|mode| !matches!(mode & 0o170000, 0 | 0o100000 | 0o040000))
            || !names.insert(relative.clone())
        {
            return Err("artifact contains an unsafe or duplicate entry".into());
        }
        total = total
            .checked_add(entry.size())
            .ok_or("artifact size overflow")?;
        if entry.size() > FILE_LIMIT || total > EXTRACT_LIMIT {
            return Err("expanded artifact exceeds its bound".into());
        }
        let output = directory.path().join(relative);
        if entry.is_dir() {
            fs::create_dir_all(output).map_err(|e| e.to_string())?;
        } else {
            fs::create_dir_all(output.parent().ok_or("artifact entry has no parent")?)
                .map_err(|e| e.to_string())?;
            let expected = entry.size();
            let mut data = Vec::new();
            entry
                .by_ref()
                .take(FILE_LIMIT + 1)
                .read_to_end(&mut data)
                .map_err(|e| format!("artifact decompression: {e}"))?;
            if data.len() as u64 != expected {
                return Err("artifact decompressed size differs from its declaration".into());
            }
            fs::write(output, data).map_err(|e| e.to_string())?;
        }
    }
    Ok(directory)
}

struct Artifact {
    metadata: Value,
    origin: Origin,
    producer: Job,
    directory: tempfile::TempDir,
}
async fn artifact(
    github: &Github,
    run: &Value,
    inventory: &[Value],
    all_jobs: &[Job],
    name: &str,
    job_name: &str,
    run_id: u64,
    candidate: &str,
) -> Result<Artifact, String> {
    let mut matches = inventory.iter().filter(|item| item["name"] == name);
    let metadata = matches
        .next()
        .ok_or_else(|| format!("required CI artifact {name} is absent"))?;
    if matches.next().is_some() {
        return Err(format!("CI artifact {name} is ambiguous"));
    }
    artifact_provenance(metadata, run, run_id, candidate)?;
    let id = metadata["id"].as_u64().unwrap();
    let bytes = github
        .bytes(&format!("actions/artifacts/{id}/zip"), ZIP_LIMIT)
        .await?;
    digest_matches(&bytes, metadata)?;
    let directory = unpack(&bytes)?;
    let origin: Origin = super::read(&directory.path().join("ci-origin.json"))?;
    if origin.run_attempt == 0 {
        return Err("artifact origin has no execution attempt".into());
    }
    let original = github
        .jobs(&format!(
            "actions/runs/{run_id}/attempts/{}/jobs",
            origin.run_attempt
        ))
        .await?;
    let producer = producer(
        &origin,
        &original,
        all_jobs,
        run_id,
        candidate,
        run["run_attempt"]
            .as_u64()
            .ok_or("CI run omitted attempt")?,
        job_name,
    )?;
    Ok(Artifact {
        metadata: metadata.clone(),
        origin,
        producer,
        directory,
    })
}

fn baseline_identity(
    release: &Value,
    tag: &str,
    abi_bytes: &[u8],
    expected: &ExpectedPerformanceRun,
) -> Result<(), String> {
    if release["tag_name"] != tag
        || release["draft"] != false
        || release["prerelease"] != false
        || tag.strip_prefix('v') != Some(expected.baseline_version.as_str())
    {
        return Err("performance baseline is not the plan's official formal release".into());
    }
    let assets = release["assets"]
        .as_array()
        .ok_or("official baseline omitted its assets")?;
    let find = |name: &str| -> Result<&Value, String> {
        let mut matches = assets.iter().filter(|asset| asset["name"] == name);
        let value = matches
            .next()
            .ok_or_else(|| format!("official baseline omitted {name}"))?;
        if matches.next().is_some() || value["state"] != "uploaded" {
            return Err("official baseline asset is ambiguous/incomplete".into());
        }
        Ok(value)
    };
    digest_matches(abi_bytes, find(&format!("{BASELINE_ARCHIVE}.abi.json"))?)?;
    let abi: Value =
        serde_json::from_slice(abi_bytes).map_err(|_| "official baseline ABI is invalid")?;
    let archive = find(BASELINE_ARCHIVE)?;
    if abi["asset_name"] != BASELINE_ARCHIVE
        || abi["backend"] != "metal"
        || abi["target_triple"] != "aarch64-apple-darwin"
        || abi["binary_name"] != "ferrum"
        || abi["binary_sha256"] != expected.baseline_sha256
        || archive["digest"].as_str()
            != abi["asset_sha256"]
                .as_str()
                .map(|hash| format!("sha256:{hash}"))
                .as_deref()
        || abi["asset_sha256"]
            .as_str()
            .is_none_or(|hash| !super::hex(hash, 64))
    {
        return Err(
            "performance baseline bytes do not match the official Metal asset ABI/digest".into(),
        );
    }
    Ok(())
}

pub(super) async fn load(
    repo: &str,
    run_id: u64,
    candidate: &str,
    plan: &Plan,
    base_tag: &str,
    version: &str,
    metal_binary_sha: &str,
) -> Result<CiEvidence, String> {
    tokio::time::timeout(
        Duration::from_secs(300),
        load_inner(
            repo,
            run_id,
            candidate,
            plan,
            base_tag,
            version,
            metal_binary_sha,
        ),
    )
    .await
    .map_err(|_| "CI evidence verification exceeded its five-minute bound".to_string())?
}
async fn load_inner(
    repo: &str,
    run_id: u64,
    candidate: &str,
    plan: &Plan,
    base_tag: &str,
    version: &str,
    metal_binary_sha: &str,
) -> Result<CiEvidence, String> {
    let needs_submission = plan.obligations.iter().any(|o| {
        o.layer == EvidenceLayer::BackendNumerics && o.behavior == Behavior::SubmissionCompletion
    });
    let schedule = performance_task_schedule(plan);
    if !schedule.unsupported_obligations.is_empty() {
        return Err("release performance schedule contains unsupported obligations".into());
    }
    if !needs_submission && schedule.runs.is_empty() {
        return Ok(CiEvidence::default());
    }
    let github = Github {
        client: reqwest::Client::builder()
            .user_agent("ferrum-release-evidence")
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|_| "cannot construct CI evidence client")?,
        repo: repo.into(),
        token: std::env::var("GITHUB_TOKEN").ok(),
    };
    let run_path = format!("actions/runs/{run_id}");
    let run = github.json(&run_path).await?;
    verify_run(&run, repo, run_id, candidate)?;
    let inventory = github
        .inventory(&format!("{run_path}/artifacts"), "artifacts")
        .await?;
    let jobs = github.jobs(&format!("{run_path}/jobs?filter=all")).await?;
    let mut evidence = CiEvidence::default();
    let mut consumed = Vec::new();
    if needs_submission {
        let raw = artifact(
            &github,
            &run,
            &inventory,
            &jobs,
            "backend-numerics-metal",
            SUBMISSION_JOB,
            run_id,
            candidate,
        )
        .await?;
        uploaded_during(
            &raw.metadata,
            &raw.producer,
            SUBMISSION_EXECUTE,
            SUBMISSION_UPLOAD,
        )?;
        evidence.submission = Some(submission::verify(
            &fs::read(raw.directory.path().join("submission.json")).map_err(|e| e.to_string())?,
            &SubmissionConfig {
                tokens: 3,
                intermediate: 33,
                k: 35,
                seed: 7,
                max_nmse: 1e-7,
            },
        )?);
        consumed.push(raw);
    }
    if !schedule.runs.is_empty() {
        let registered = artifact(
            &github,
            &run,
            &inventory,
            &jobs,
            "release-performance-tasks",
            PERFORMANCE_JOB,
            run_id,
            candidate,
        )
        .await?;
        let measured = artifact(
            &github,
            &run,
            &inventory,
            &jobs,
            "release-metal-evidence",
            PERFORMANCE_JOB,
            run_id,
            candidate,
        )
        .await?;
        if registered.origin != measured.origin {
            return Err("performance registration and measurement have different producers".into());
        }
        preregistered(&registered.metadata, &registered.producer)?;
        uploaded_during(
            &measured.metadata,
            &measured.producer,
            PERFORMANCE_EXECUTE,
            PERFORMANCE_UPLOAD,
        )?;
        let release = github.json(&format!("releases/tags/{base_tag}")).await?;
        let policy_bytes = github
            .response_bytes(
                &format!("contents/.github/release-performance.json?ref={candidate}"),
                JSON_LIMIT,
                "application/vnd.github.raw+json",
            )
            .await?;
        let policy = candidate_policy(&policy_bytes)?;
        let task_root = registered.directory.path();
        let abi = fs::read(
            task_root
                .join("baseline")
                .join(format!("{BASELINE_ARCHIVE}.abi.json")),
        )
        .map_err(|e| format!("registered official baseline ABI: {e}"))?;
        let expected_ids: BTreeSet<_> = schedule
            .runs
            .iter()
            .map(|task| task.profile.id.as_str())
            .collect();
        check_profile_directories(task_root, &expected_ids, true)?;
        check_profile_directories(
            &measured.directory.path().join("performance"),
            &expected_ids,
            false,
        )?;
        for requirement in schedule.runs {
            if !super::file_name(&requirement.profile.id)
                || matches!(requirement.profile.id.as_str(), "baseline" | "candidate")
            {
                return Err("performance profile id is not a safe task directory".into());
            }
            let prepared = task_root.join(&requirement.profile.id);
            let raw = measured
                .directory
                .path()
                .join("performance")
                .join(&requirement.profile.id);
            let expected: ExpectedPerformanceRun =
                super::read(&prepared.join("expected-task.json"))?;
            expected.validate()?;
            if expected.profile != requirement.profile
                || expected.obligations != requirement.obligations
                || expected.candidate_version != version
                || expected.client_version != version
                || expected.candidate_sha256 != metal_binary_sha
                || expected.client_sha256 != metal_binary_sha
                || expected.policy != policy
            {
                return Err("registered performance task differs from release scope or staged candidate/client bytes".into());
            }
            baseline_identity(&release, base_tag, &abi, &expected)?;
            let registered_source: Value = super::read(&prepared.join("source-manifest.json"))?;
            let measured_source: Value = super::read(&raw.join("source-manifest.json"))?;
            if registered_source != measured_source {
                return Err("performance source changed after registration".into());
            }
            let descriptors = performance_check_descriptors(&[expected.profile.target.clone()]);
            for index in &requirement.obligations {
                let obligation = &plan.obligations[*index];
                if descriptors.len() != 1 || obligation.checkers != [descriptors[0].id.clone()] {
                    return Err(
                        "performance obligation contains an unknown/unexecuted checker".into(),
                    );
                }
            }
            performance::verify_evidence(&expected, &raw)?;
            for index in requirement.obligations {
                evidence
                    .performance
                    .insert(index, plan.obligations[index].clone());
            }
        }
        consumed.extend([registered, measured]);
    }
    // Detect retries, replacement artifacts or new real producer executions
    // racing the download/replay, without rejecting unchanged copied jobs.
    let current = github.json(&run_path).await?;
    verify_run(&current, repo, run_id, candidate)?;
    let latest = github.jobs(&format!("{run_path}/jobs?filter=all")).await?;
    for raw in consumed {
        producer(
            &raw.origin,
            std::slice::from_ref(&raw.producer),
            &latest,
            run_id,
            candidate,
            current["run_attempt"].as_u64().unwrap(),
            &raw.producer.name,
        )?;
        let refreshed = github
            .json(&format!(
                "actions/artifacts/{}",
                raw.metadata["id"].as_u64().unwrap()
            ))
            .await?;
        if refreshed != raw.metadata {
            return Err("CI evidence artifact metadata changed during verification".into());
        }
    }
    Ok(evidence)
}
fn verify_run(run: &Value, repo: &str, run_id: u64, candidate: &str) -> Result<(), String> {
    if run["id"].as_u64() != Some(run_id)
        || run_id == 0
        || run["head_sha"].as_str() != Some(candidate)
        || run["repository"]["full_name"]
            .as_str()
            .is_none_or(|name| !name.eq_ignore_ascii_case(repo))
        || !matches!(run["event"].as_str(), Some("push" | "workflow_dispatch"))
        || run["path"].as_str().and_then(|path| path.split('@').next())
            != Some(".github/workflows/release-delivery.yml")
        || run["run_attempt"]
            .as_u64()
            .is_none_or(|attempt| attempt == 0)
    {
        return Err(
            "CI evidence run does not belong to this candidate/repository/release workflow".into(),
        );
    }
    Ok(())
}
fn check_profile_directories(
    root: &Path,
    expected: &BTreeSet<&str>,
    baseline: bool,
) -> Result<(), String> {
    let mut actual = BTreeSet::new();
    for entry in fs::read_dir(root).map_err(|e| format!("performance artifact directories: {e}"))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry
            .file_name()
            .to_str()
            .ok_or("non-UTF8 performance directory")?
            .to_owned();
        if baseline
            && name == "ci-origin.json"
            && entry.file_type().map_err(|e| e.to_string())?.is_file()
        {
            continue;
        }
        if !entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            return Err("unexpected file in performance task root".into());
        }
        // Preparation retains both original metadata bundles. Candidate byte
        // identity is independently pinned to the already verified distribution.
        if baseline && matches!(name.as_str(), "baseline" | "candidate") {
            continue;
        }
        actual.insert(name);
    }
    if actual.iter().map(String::as_str).collect::<BTreeSet<_>>() != *expected {
        return Err("performance artifact has missing or unregistered profile directories".into());
    }
    Ok(())
}

fn candidate_policy(bytes: &[u8]) -> Result<PerformancePolicy, String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|_| "candidate performance configuration is invalid JSON")?;
    if value["schema_version"] != 1 {
        return Err("candidate performance configuration has an unsupported schema".into());
    }
    let policy: PerformancePolicy = serde_json::from_value(value["policy"].clone())
        .map_err(|e| format!("candidate performance policy: {e}"))?;
    policy.validate()?;
    Ok(policy)
}
#[cfg(test)]
#[path = "ci_evidence_tests.rs"]
mod tests;
