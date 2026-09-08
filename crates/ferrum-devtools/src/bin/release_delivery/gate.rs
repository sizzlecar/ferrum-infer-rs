//! Concrete release evidence gate. Plans and saved summaries are not execution.
use super::{installation, AcceptedAsset, AcceptedRelease};
use clap::Args;
use ferrum_bench_core::{
    release_candidate::staging::{self, AbiInput, CandidateInput},
    release_regression::{
        contracts::{
            contract_check_descriptors, contract_groups, verify_contract_report, ContractReport,
        },
        distribution::distribution_check_descriptors,
        model_schedule::{model_check_descriptors, model_task_schedule, ModelTaskSchedule},
        model_tasks::{
            verify_model_reports, ExpectedModelRun, ModelCheck, DEFAULT_FUNCTIONAL_CAPACITY,
        },
        Backend, Behavior, CheckDescriptor, EvidenceLayer, Gap, Obligation, ObligationScope, Plan,
        Stage,
    },
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::process::Command;

#[path = "ci_evidence.rs"]
mod ci_evidence;

#[path = "windows_assets.rs"]
pub(super) mod windows_assets;

#[derive(Debug, Args)]
pub struct GateArgs {
    #[arg(long)]
    pub plan: PathBuf,
    #[arg(long)]
    pub tasks: PathBuf,
    #[arg(long)]
    pub contracts: PathBuf,
    #[arg(long, required = true)]
    pub installation: Vec<PathBuf>,
    #[arg(long)]
    pub report: Vec<PathBuf>,
    /// Explicit staged metadata paths; remote installation paths are not local inputs.
    #[arg(long, required = true)]
    pub abi: Vec<PathBuf>,
    /// Windows assets from this release's successful staging job, kept outside Unix ABI inputs.
    #[arg(long)]
    pub windows_staged: PathBuf,
    #[arg(long)]
    pub version: String,
    #[arg(long)]
    pub workspace: PathBuf,
    #[arg(long)]
    pub notes: PathBuf,
    #[arg(long, default_value = "sizzlecar/ferrum-infer-rs")]
    pub repo: String,
    #[arg(long)]
    pub ci_run_id: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedTasks {
    schema_version: u32,
    expectations: Vec<ExpectedModelRun>,
    unsupported_obligations: Vec<usize>,
    remaining_plan_gaps: Vec<Gap>,
}
#[derive(Debug)]
struct Distribution {
    backend: Backend,
    name: String,
    sha256: String,
    binary_sha256: String,
    target: String,
    assets: Vec<AcceptedAsset>,
}

pub async fn verify(args: GateArgs) -> Result<AcceptedRelease, String> {
    validate_repo(&args.repo)?;
    let document: Value = read(&args.plan)?;
    let (plan, candidate, schedule) = release_plan(&document)?;
    let version = semver::Version::parse(&args.version).map_err(|_| "invalid release version")?;
    if !version.pre.is_empty() || !version.build.is_empty() || version.to_string() != args.version {
        return Err("release gate requires a canonical formal version".into());
    }
    verify_plan_version(&document, &args.version)?;
    let workspace = args
        .workspace
        .canonicalize()
        .map_err(|e| format!("candidate workspace: {e}"))?;
    let head = Command::new("git")
        .arg("-C")
        .arg(&workspace)
        .args(["rev-parse", "--verify", "HEAD^{commit}"])
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .stdin(Stdio::null())
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|_| "cannot resolve publication workspace")?;
    if !head.status.success() || String::from_utf8_lossy(&head.stdout).trim() != candidate {
        return Err("workspace HEAD does not match the plan candidate".into());
    }
    // The candidate's reachable base can be older than today's main releases.
    // Reject a historical downgrade before any public channel can be mutated.
    verify_public_version(&args.repo, &args.version).await?;
    let windows = windows_assets::verify(
        &args.windows_staged,
        &args.version,
        &candidate,
        args.ci_run_id,
        &text(&document["provenance"], "release_base_tag")?,
        &args.repo,
    )?;
    let mut distributions = BTreeMap::new();
    let mut names = BTreeSet::new();
    for path in &args.abi {
        let distribution = distribution(path, &args.version, &candidate).await?;
        for asset in &distribution.assets {
            if !names.insert(asset.name.clone()) {
                return Err("duplicate declared distribution asset".into());
            }
        }
        if distributions
            .insert(distribution.backend, distribution)
            .is_some()
        {
            return Err("multiple distribution archives declare the same backend".into());
        }
    }
    // These are the product's three formal distributions, not a model matrix.
    if distributions.keys().copied().collect::<BTreeSet<_>>()
        != BTreeSet::from([Backend::Cpu, Backend::Metal, Backend::Cuda])
    {
        return Err("formal distribution inventory must include CPU, Metal and CUDA".into());
    }
    let tasks: PreparedTasks = read(&args.tasks)?;
    validate_tasks(&plan, &schedule, &tasks, &distributions, &args.version)?;
    let reports = args
        .report
        .iter()
        .map(|path| read::<Value>(path))
        .collect::<Result<Vec<_>, _>>()?;
    let models = VerifiedModels::new(&tasks.expectations, &reports)?;
    let installations = args
        .installation
        .iter()
        .map(|path| read::<installation::InstallationReport>(path))
        .collect::<Result<Vec<_>, _>>()?;
    verify_installations(
        &distributions,
        &installations,
        &models,
        &args.version,
        &candidate,
    )?;
    let contracts: ContractReport = read(&args.contracts)?;
    verify_contract_report(&contract_groups(), &contracts)
        .map_err(|issues| format!("CPU contract evidence: {}", issues.join("; ")))?;
    verify_ci(&args.repo, args.ci_run_id, &candidate, windows.attempt).await?;
    let ci_evidence = ci_evidence::load(
        &args.repo,
        args.ci_run_id,
        &candidate,
        &plan,
        &text(&document["provenance"], "release_base_tag")?,
        &args.version,
        &distributions[&Backend::Metal].binary_sha256,
    )
    .await?;
    verify_obligations_with(&plan, &ci_evidence)?;
    // A rerun that began while artifacts were inspected must not reuse old Quality.
    verify_ci(&args.repo, args.ci_run_id, &candidate, windows.attempt).await?;
    let notes = fs::read_to_string(&args.notes).map_err(|e| format!("release notes: {e}"))?;
    if notes.trim().is_empty() {
        return Err("release notes are empty".into());
    }
    let mut assets: Vec<_> = distributions
        .into_values()
        .flat_map(|distribution| distribution.assets)
        .collect();
    for asset in windows.assets {
        if !names.insert(asset.name.clone()) {
            return Err("duplicate declared release asset".into());
        }
        assets.push(asset);
    }
    Ok(AcceptedRelease {
        version: args.version,
        candidate_sha: candidate,
        workspace,
        notes,
        assets,
    })
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?)
        .map_err(|e| format!("parse {}: {e}", path.display()))
}
fn text(value: &Value, field: &str) -> Result<String, String> {
    value[field]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| format!("missing metadata {field}"))
}
fn hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn file_name(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && !value.starts_with('-')
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
fn validate_repo(repo: &str) -> Result<(), String> {
    let parts: Vec<_> = repo.split('/').collect();
    if parts.len() != 2 || parts.iter().any(|part| !file_name(part)) {
        Err("repository must be an owner/name pair".into())
    } else {
        Ok(())
    }
}

fn verify_plan_version(document: &Value, target: &str) -> Result<(), String> {
    let tag = text(&document["provenance"], "release_base_tag")?;
    let previous = tag
        .strip_prefix('v')
        .ok_or("release plan base is not a formal v-version tag")?;
    staging::validate_version_progression(previous, target)
}

fn release_plan(document: &Value) -> Result<(Plan, String, ModelTaskSchedule), String> {
    if document["schema_version"] != 2 || document["stage"] != "release" {
        return Err("release delivery requires a schema 2 release plan".into());
    }
    let plan: Plan = serde_json::from_value(document["plan"].clone())
        .map_err(|e| format!("invalid typed release plan: {e}"))?;
    let candidate = text(&document["provenance"], "candidate")?;
    if !hex(&candidate, 40) || plan.stage != Stage::Release {
        return Err("release plan has invalid candidate or stage".into());
    }
    if !plan.gaps.is_empty() {
        return Err(format!(
            "release plan still has coverage gaps: {:?}",
            plan.gaps
        ));
    }
    if plan.obligations.is_empty() || plan.selected.is_empty() {
        return Err("release plan omits its required obligations/model representatives".into());
    }
    let mut profiles = BTreeSet::new();
    for selected in &plan.selected {
        if !profiles.insert(&selected.profile.id)
            || selected
                .obligations
                .iter()
                .any(|index| *index >= plan.obligations.len())
            || selected.obligations.iter().collect::<BTreeSet<_>>().len()
                != selected.obligations.len()
        {
            return Err("invalid or duplicate selected profile assignments".into());
        }
    }
    let schedule = model_task_schedule(&plan);
    let saved: ModelTaskSchedule = serde_json::from_value(document["model_tasks"].clone())
        .map_err(|_| "plan omits its typed model schedule")?;
    if saved != schedule || !schedule.unsupported_obligations.is_empty() || schedule.runs.is_empty()
    {
        return Err("plan model schedule is stale, unsupported or empty".into());
    }
    Ok((plan, candidate, schedule))
}

async fn distribution(path: &Path, version: &str, candidate: &str) -> Result<Distribution, String> {
    let abi: Value = read(path)?;
    let name = text(&abi, "asset_name")?;
    if !file_name(&name)
        || path.file_name().and_then(|v| v.to_str()) != Some(format!("{name}.abi.json").as_str())
    {
        return Err("--abi must name the metadata for its declared archive".into());
    }
    let directory = path.parent().ok_or("ABI path has no directory")?;
    let version_record: Value = read(&directory.join(format!("{name}.version.json")))?;
    let dependencies: Value = read(&directory.join(format!("{name}.dependency.json")))?;
    if version_record["version"] != version || version_record["release_candidate_sha"] != candidate
    {
        return Err("staged distribution version/source differs from the release plan".into());
    }
    let input = CandidateInput {
        version: version.into(),
        release_candidate_sha: candidate.into(),
        release_candidate_tag: text(&version_record, "release_candidate_tag")?,
        staging_label: text(&version_record, "staging_label")?,
        workflow_run_id: text(&version_record, "workflow_run_id")?,
        workflow_run_attempt: text(&version_record, "workflow_run_attempt")?,
    };
    let abi_input = AbiInput {
        backend: serde_json::from_value(abi["backend"].clone())
            .map_err(|_| "invalid distribution backend")?,
        target_triple: text(&abi, "target_triple")?,
        cargo_features: serde_json::from_value(abi["cargo_features"].clone())
            .map_err(|_| "missing staged cargo_features")?,
        cuda_compute_capability: abi
            .get("cuda_compute_capability")
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or("invalid CUDA compute capability")
            })
            .transpose()?,
        cuda_toolkit_image: abi
            .get("cuda_toolkit_image")
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or("invalid CUDA toolkit image")
            })
            .transpose()?,
    };
    let audit_name = text(&dependencies, "audit_file")?;
    if !file_name(&audit_name) {
        return Err("dependency audit must be an adjacent declared filename".into());
    }
    let audit = fs::read_to_string(directory.join(&audit_name))
        .map_err(|e| format!("read raw dependency audit: {e}"))?;
    let archive_path = directory.join(&name);
    let bytes = fs::read(&archive_path).map_err(|e| format!("read distribution archive: {e}"))?;
    if format!("{:x}", Sha256::digest(&bytes)) != text(&abi, "asset_sha256")? {
        return Err("distribution archive checksum differs from staged metadata".into());
    }
    let extracted = tokio::time::timeout(
        Duration::from_secs(120),
        Command::new("tar")
            .arg("-xOzf")
            .arg(&archive_path)
            .args(["--", "ferrum"])
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output(),
    )
    .await
    .map_err(|_| "distribution inspection timed out")?
    .map_err(|_| "cannot inspect distribution archive")?;
    if !extracted.status.success() {
        return Err("distribution archive does not contain ferrum".into());
    }
    let regenerated = staging::generate_manifests(
        &input,
        &abi_input,
        &name,
        &bytes,
        &extracted.stdout,
        &audit_name,
        &audit,
    )?;
    if regenerated.abi != abi
        || regenerated.version != version_record
        || regenerated.dependency != dependencies
    {
        return Err(
            "staged metadata does not match actual archive/binary/dependency audit inputs".into(),
        );
    }
    for (suffix, expected) in [
        (".sha256", &regenerated.asset_checksum),
        (".binary.sha256", &regenerated.binary_checksum),
    ] {
        if fs::read_to_string(directory.join(format!("{name}{suffix}")))
            .map_err(|e| e.to_string())?
            != *expected
        {
            return Err("staged checksum sidecar disagrees with actual bytes".into());
        }
    }
    let mut files = vec![name.clone()];
    for suffix in [
        ".sha256",
        ".binary.sha256",
        ".version.json",
        ".dependency.json",
        ".abi.json",
    ] {
        files.push(format!("{name}{suffix}"));
    }
    files.push(audit_name);
    let mut assets = Vec::new();
    for filename in files {
        let path = directory.join(&filename);
        let content =
            fs::read(&path).map_err(|e| format!("read accepted distribution member: {e}"))?;
        let actual = format!("{:x}", Sha256::digest(&content));
        let unchanged = if filename == name {
            actual == text(&abi, "asset_sha256")?
        } else if filename == text(&dependencies, "audit_file")? {
            content == audit.as_bytes()
        } else if filename == format!("{name}.sha256") {
            content == regenerated.asset_checksum.as_bytes()
        } else if filename == format!("{name}.binary.sha256") {
            content == regenerated.binary_checksum.as_bytes()
        } else {
            let expected = if filename.ends_with(".abi.json") {
                &abi
            } else if filename.ends_with(".version.json") {
                &version_record
            } else {
                &dependencies
            };
            serde_json::from_slice::<Value>(&content).is_ok_and(|parsed| parsed == *expected)
        };
        if !unchanged {
            return Err(
                "distribution files changed while constructing accepted asset inventory".into(),
            );
        }
        assets.push(AcceptedAsset {
            path,
            name: filename,
            sha256: actual,
        });
    }
    let backend = match abi_input.backend {
        staging::Backend::Cpu => Backend::Cpu,
        staging::Backend::Metal => Backend::Metal,
        staging::Backend::Cuda => Backend::Cuda,
    };
    Ok(Distribution {
        backend,
        name,
        sha256: text(&abi, "asset_sha256")?,
        binary_sha256: text(&abi, "binary_sha256")?,
        target: abi_input.target_triple,
        assets,
    })
}

fn validate_tasks(
    plan: &Plan,
    schedule: &ModelTaskSchedule,
    tasks: &PreparedTasks,
    distributions: &BTreeMap<Backend, Distribution>,
    version: &str,
) -> Result<(), String> {
    if tasks.schema_version != 1
        || tasks.remaining_plan_gaps != plan.gaps
        || tasks.unsupported_obligations != schedule.unsupported_obligations
        || tasks.expectations.len() != schedule.runs.len()
    {
        return Err("prepared tasks do not preserve the current plan/schedule/gaps".into());
    }
    let mut seen = BTreeSet::new();
    for expected in &tasks.expectations {
        if !seen.insert(&expected.profile.id) {
            return Err("duplicate prepared model task".into());
        }
        let run = schedule
            .runs
            .iter()
            .find(|run| run.profile.id == expected.profile.id)
            .ok_or("unexpected model task")?;
        let binary = distributions
            .get(&expected.profile.target.backend)
            .ok_or("model task has no declared distribution")?;
        if expected.profile != run.profile
            || expected.version != version
            || expected.binary_sha256 != binary.binary_sha256
            || expected.disable_thinking != run.quick_start
            || expected.use_default_backend != run.quick_start
            || expected.runtime_capacity
                != (!run.quick_start).then_some(DEFAULT_FUNCTIONAL_CAPACITY)
            || expected.checks.iter().copied().collect::<BTreeSet<_>>()
                != run.checks.iter().copied().collect::<BTreeSet<_>>()
            || expected.checks.len() != run.checks.len()
        {
            return Err(format!(
                "model task {} differs from selected profile/checks/Quick Start/capacity/staged bytes",
                expected.profile.id
            ));
        }
    }
    Ok(())
}

struct VerifiedModels<'a> {
    tasks: &'a [ExpectedModelRun],
    reports: &'a [Value],
}
impl<'a> VerifiedModels<'a> {
    fn new(tasks: &'a [ExpectedModelRun], reports: &'a [Value]) -> Result<Self, String> {
        verify_model_reports(tasks, reports)
            .map_err(|issues| format!("model evidence: {}", issues.join("; ")))?;
        Ok(Self { tasks, reports })
    }
}
fn verify_installations(
    distributions: &BTreeMap<Backend, Distribution>,
    installations: &[installation::InstallationReport],
    models: &VerifiedModels<'_>,
    version: &str,
    candidate: &str,
) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for report in installations {
        let backend: Backend = serde_json::from_value(Value::String(report.backend.clone()))
            .map_err(|_| "invalid installation backend")?;
        let expected = distributions
            .get(&backend)
            .ok_or("unexpected installation report")?;
        if !seen.insert(backend)
            || report.version != version
            || report.candidate_sha != candidate
            || report.asset_name != expected.name
            || report.asset_sha256 != expected.sha256
            || report.binary_sha256 != expected.binary_sha256
            || report.target_triple != expected.target
        {
            return Err(
                "installation report is duplicate, stale or bound to another archive".into(),
            );
        }
        if backend == Backend::Cuda && report.status == "not_run" {
            if report.schema_version != 1
                || report.error.is_some()
                || !report.observations.is_empty()
            {
                return Err(
                    "CUDA extract-only report contains contradictory execution claims".into(),
                );
            }
            let fallback = models
                .tasks
                .iter()
                .filter(|task| {
                    task.profile.target.backend == Backend::Cuda
                        && task.binary_sha256 == expected.binary_sha256
                        && task.checks.contains(&ModelCheck::Basic)
                })
                .any(|task| {
                    models.reports.iter().any(|model| {
                        model["profile_id"] == task.profile.id
                            && model["cases"].as_array().is_some_and(|cases| {
                                ["run-basic", "serve-startup"].iter().all(|name| {
                                    cases.iter().any(|case| {
                                        case["case"] == *name && case["status"] == "passed"
                                    })
                                })
                            })
                    })
                });
            if !fallback {
                return Err("CUDA extract-only inspection needs verified same-binary RunBasic and ServeStartup model evidence".into());
            }
            println!("CUDA archive inspection remains not_run; same-binary verified model RunBasic and ServeStartup provide actual startup evidence.");
        } else {
            installation::verify_runtime(report)?;
        }
    }
    if seen != distributions.keys().copied().collect() {
        return Err("a declared distribution is missing installation evidence".into());
    }
    Ok(())
}

fn descriptors_cover(obligation: &Obligation, descriptors: &[CheckDescriptor]) -> bool {
    if obligation.checkers.is_empty() {
        return false;
    }
    let mut covered = BTreeSet::new();
    for id in &obligation.checkers {
        let Some(descriptor) = descriptors.iter().find(|descriptor| &descriptor.id == id) else {
            return false;
        };
        if descriptor.behavior != obligation.behavior
            || descriptor.layer != obligation.layer
            || descriptor.target.is_some()
        {
            return false;
        }
        covered.extend(descriptor.entrypoints.iter().copied());
    }
    obligation
        .entrypoints
        .iter()
        .all(|entry| covered.contains(entry))
}
#[cfg(test)]
fn verify_obligations(plan: &Plan) -> Result<(), String> {
    verify_obligations_with(plan, &ci_evidence::CiEvidence::default())
}
fn verify_obligations_with(plan: &Plan, evidence: &ci_evidence::CiEvidence) -> Result<(), String> {
    let contracts = contract_check_descriptors();
    let models = model_check_descriptors();
    let distribution = distribution_check_descriptors();
    for (index, obligation) in plan.obligations.iter().enumerate() {
        let covered = match obligation.layer {
            EvidenceLayer::Contract => {
                matches!(obligation.scope, ObligationScope::Global)
                    && descriptors_cover(obligation, &contracts)
            }
            EvidenceLayer::ModelRuntime => descriptors_cover(obligation, &models),
            EvidenceLayer::Compilation => {
                obligation.behavior == Behavior::WorkspaceChecks
                    && matches!(obligation.scope, ObligationScope::Global)
                    && descriptors_cover(obligation, &distribution)
            }
            EvidenceLayer::Installation => {
                obligation.behavior == Behavior::Installation
                    && matches!(obligation.scope, ObligationScope::Backend { .. })
                    && descriptors_cover(obligation, &distribution)
            }
            EvidenceLayer::BackendNumerics | EvidenceLayer::Performance => {
                evidence.covers(index, obligation)
            }
        };
        if !covered {
            return Err(format!(
                "release obligation {index} has no executed known checker for {:?}/{:?}",
                obligation.behavior, obligation.layer
            ));
        }
    }
    Ok(())
}

async fn verify_public_version(repo: &str, target: &str) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .user_agent("ferrum-release-delivery")
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|_| "cannot construct release inventory client")?;
    let token = std::env::var("GITHUB_TOKEN").ok();
    public_version_at(
        &client,
        "https://api.github.com",
        repo,
        target,
        token.as_deref(),
    )
    .await
}

/// Equality permits reconciling the current published version. Candidate/tag
/// and byte conflicts are independently rejected by the publisher on resume.
async fn public_version_at(
    client: &reqwest::Client,
    base: &str,
    repo: &str,
    target: &str,
    token: Option<&str>,
) -> Result<(), String> {
    let target_version =
        semver::Version::parse(target).map_err(|_| "invalid formal target version")?;
    if !target_version.pre.is_empty()
        || !target_version.build.is_empty()
        || target_version.to_string() != target
    {
        return Err("release target must be a canonical formal version".into());
    }
    let mut highest: Option<semver::Version> = None;
    let mut page = 1u64;
    loop {
        let mut request = client
            .get(format!(
                "{base}/repos/{repo}/releases?per_page=100&page={page}"
            ))
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28");
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .map_err(|_| "published release inventory unavailable; version ordering unknown")?;
        if !response.status().is_success() {
            return Err(format!(
                "published release inventory returned HTTP {}; version ordering unknown",
                response.status()
            ));
        }
        let value: Value = response
            .json()
            .await
            .map_err(|_| "published release inventory is not valid JSON")?;
        let rows = value
            .as_array()
            .ok_or("published release inventory is not an array")?;
        for row in rows {
            let draft = row["draft"]
                .as_bool()
                .ok_or("release inventory omits draft state")?;
            let prerelease = row["prerelease"]
                .as_bool()
                .ok_or("release inventory omits prerelease state")?;
            if draft || prerelease {
                continue;
            }
            let tag = row["tag_name"]
                .as_str()
                .ok_or("release inventory omits tag name")?;
            // Native operator assets have their own non-version tags. They are
            // releases, but not a formal Ferrum version in the v{semver} scheme.
            let Some(version) = tag
                .strip_prefix('v')
                .and_then(|value| semver::Version::parse(value).ok())
            else {
                continue;
            };
            if !version.pre.is_empty() || !version.build.is_empty() || tag != format!("v{version}")
            {
                continue;
            }
            if row["published_at"]
                .as_str()
                .filter(|date| !date.is_empty())
                .is_none()
            {
                return Err("formal release inventory omits publication confirmation".into());
            }
            if highest.as_ref().is_none_or(|current| &version > current) {
                highest = Some(version);
            }
        }
        if rows.len() < 100 {
            break;
        }
        page = page
            .checked_add(1)
            .ok_or("release inventory pagination overflow")?;
    }
    if let Some(highest) = highest {
        if target_version < highest {
            return Err(format!("target release {target} is older than already published formal release {highest}; refusing downgrade"));
        }
    }
    Ok(())
}

async fn verify_ci(
    repo: &str,
    run_id: u64,
    candidate: &str,
    windows_attempt: u64,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .user_agent("ferrum-release-delivery")
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|_| "cannot construct CI client")?;
    let token = std::env::var("GITHUB_TOKEN").ok();
    ci_at_with_windows(
        &client,
        "https://api.github.com",
        repo,
        run_id,
        candidate,
        token.as_deref(),
        Some(windows_attempt),
    )
    .await
}
#[cfg(test)]
async fn ci_at(
    client: &reqwest::Client,
    base: &str,
    repo: &str,
    run_id: u64,
    candidate: &str,
    token: Option<&str>,
) -> Result<(), String> {
    ci_at_with_windows(client, base, repo, run_id, candidate, token, None).await
}
async fn ci_at_with_windows(
    client: &reqwest::Client,
    base: &str,
    repo: &str,
    run_id: u64,
    candidate: &str,
    token: Option<&str>,
    windows_attempt: Option<u64>,
) -> Result<(), String> {
    if run_id == 0 {
        return Err("CI run id must be explicit and nonzero".into());
    }
    async fn get(
        client: &reqwest::Client,
        url: String,
        token: Option<&str>,
    ) -> Result<Value, String> {
        let mut request = client
            .get(url)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28");
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .map_err(|_| "CI query failed; status unknown")?;
        if !response.status().is_success() {
            return Err(format!(
                "CI query returned HTTP {}; status unknown",
                response.status()
            ));
        }
        response
            .json()
            .await
            .map_err(|_| "CI response was not valid JSON".into())
    }
    let run = get(
        client,
        format!("{base}/repos/{repo}/actions/runs/{run_id}"),
        token,
    )
    .await?;
    if run["id"].as_u64() != Some(run_id)
        || run["head_sha"].as_str() != Some(candidate)
        || run["repository"]["full_name"]
            .as_str()
            .is_none_or(|name| !name.eq_ignore_ascii_case(repo))
        || !matches!(run["event"].as_str(), Some("push" | "workflow_dispatch"))
        || run["path"].as_str().and_then(|path| path.split('@').next())
            != Some(".github/workflows/release-delivery.yml")
    {
        return Err("CI run does not belong to this candidate/repository/release workflow".into());
    }
    let attempt = run["run_attempt"]
        .as_u64()
        .filter(|attempt| *attempt > 0)
        .ok_or("CI run omits its current attempt")?;
    // GitHub documents filter=all as including previous executions. Select
    // the latest occurrence of this specific job: a publish-only retry need
    // not rerun Quality, but a newer pending/failed Quality replaces old success.
    // https://docs.github.com/en/rest/actions/workflow-jobs#list-jobs-for-a-workflow-run
    let mut page = 1;
    let mut occurrences = BTreeMap::new();
    let mut windows_occurrences = BTreeMap::new();
    let mut ids = BTreeSet::new();
    let mut total = None;
    loop {
        let value = get(client, format!("{base}/repos/{repo}/actions/runs/{run_id}/jobs?filter=all&per_page=100&page={page}"), token).await?;
        let jobs = value["jobs"].as_array().ok_or("CI response omits jobs")?;
        let count = value["total_count"]
            .as_u64()
            .ok_or("CI response omits total job count")?;
        if total.is_some_and(|previous| previous != count) {
            return Err("CI job inventory changed while reading it; retry reconciliation".into());
        }
        total = Some(count);
        for job in jobs {
            let id = job["id"]
                .as_u64()
                .filter(|id| *id > 0)
                .ok_or("CI job has no identity")?;
            if !ids.insert(id) {
                return Err("CI response repeats a job across pages".into());
            }
            let windows_job = job["name"] == "stage-cuda / Stage Windows x86_64 CUDA sm89";
            if job["name"] != "Quality / CI required" && !(windows_attempt.is_some() && windows_job)
            {
                continue;
            }
            let job_attempt = job["run_attempt"]
                .as_u64()
                .filter(|value| *value > 0 && *value <= attempt)
                .ok_or("Quality job omits its actual attempt or the CI run changed")?;
            if job["run_id"].as_u64() != Some(run_id) || job["head_sha"].as_str() != Some(candidate)
            {
                return Err("Quality job belongs to another run or candidate".into());
            }
            let selected = if windows_job {
                &mut windows_occurrences
            } else {
                &mut occurrences
            };
            if selected.insert(job_attempt, job.clone()).is_some() {
                return Err("Quality job is ambiguous within one execution attempt".into());
            }
        }
        if jobs.len() < 100 {
            break;
        }
        page += 1;
    }
    if total != Some(ids.len() as u64) {
        return Err("CI returned an incomplete job inventory".into());
    }
    if let Some(expected) = windows_attempt {
        let (attempt, job) = windows_occurrences
            .last_key_value()
            .ok_or("Windows release staging job is missing")?;
        if *attempt != expected || job["status"] != "completed" || job["conclusion"] != "success" {
            return Err(
                "latest Windows staging did not succeed for the accepted artifact attempt".into(),
            );
        }
    }
    let (job_attempt, required) = occurrences
        .last_key_value()
        .ok_or("release CI has not completed its required quality job")?;
    if required["status"] != "completed" || required["conclusion"] != "success" {
        return Err(
            "latest Quality / CI required did not complete successfully for this candidate".into(),
        );
    }
    println!(
        "Verified Quality / CI required job {} from attempt {} of run {} (current run attempt {}).",
        required["id"], job_attempt, run_id, attempt
    );
    Ok(())
}

#[cfg(test)]
#[path = "gate_tests.rs"]
mod tests;
