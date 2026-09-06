//! Freeze the scheduled performance task before execution, from existing cache bytes.
use super::{read_json, write_json};
use clap::Args;
use ferrum_bench_core::{
    release_candidate::staging::{
        validate_candidate, validate_version_progression, CandidateInput,
    },
    release_regression::{
        performance::{
            performance_task_schedule, ExpectedPerformanceRun, PerformancePolicy,
            PerformanceRunRequirements,
        },
        Plan, Stage,
    },
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::time::Instant;
#[path = "cache.rs"]
mod cache;

#[derive(Debug, Args)]
pub struct PerformancePrepareArgs {
    #[arg(long)]
    pub plan: PathBuf,
    #[arg(long)]
    pub profile_id: String,
    #[arg(long)]
    pub candidate_abi: PathBuf,
    #[arg(long)]
    pub baseline_abi: PathBuf,
    /// Explicit workload, time budgets and acceptable TTFT/TPOT increases.
    #[arg(long)]
    pub policy: PathBuf,
    /// Existing HF hub cache (or its parent huggingface directory). No downloads.
    #[arg(long)]
    pub hf_cache: PathBuf,
    /// Logical model/repositories/filename, with optional immutable revisions.
    #[arg(long)]
    pub source_selector: PathBuf,
    /// Task root may contain CI provenance; the selected profile directory must be new.
    #[arg(long)]
    pub output_dir: PathBuf,
}
#[derive(Deserialize)]
struct Provenance {
    base: String,
    candidate: String,
    release_base_tag: String,
}
#[derive(Deserialize)]
struct PlanDocument {
    schema_version: u32,
    stage: Stage,
    provenance: Provenance,
    plan: Plan,
}
fn requirements(
    document: &PlanDocument,
    profile: &str,
) -> Result<PerformanceRunRequirements, String> {
    if document.schema_version != 2
        || document.stage != Stage::Release
        || document.plan.stage != Stage::Release
    {
        return Err("performance prepare requires a schema-2 release plan".into());
    }
    if !cache::safe_component(profile) {
        return Err("profile ID must be a safe directory component".into());
    }
    let schedule = performance_task_schedule(&document.plan);
    if !schedule.unsupported_obligations.is_empty() {
        return Err(format!(
            "plan contains unsupported performance obligations: {:?}",
            schedule.unsupported_obligations
        ));
    }
    let mut selected = schedule
        .runs
        .into_iter()
        .filter(|run| run.profile.id == profile);
    let run = selected
        .next()
        .ok_or("profile has no assigned implemented performance task")?;
    if selected.next().is_some() {
        return Err("ambiguous performance profile assignment".into());
    }
    Ok(run)
}
struct Metadata {
    abi_bytes: Vec<u8>,
    version_bytes: Vec<u8>,
    identity: CandidateInput,
    asset_name: String,
    binary_sha256: String,
    target: String,
}
fn field(value: &Value, key: &str) -> Result<String, String> {
    value[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("asset metadata missing {key}"))
}
fn metadata(path: &Path, expected_commit: &str) -> Result<Metadata, String> {
    let abi_bytes = fs::read(path).map_err(|e| format!("read ABI metadata: {e}"))?;
    let abi: Value =
        serde_json::from_slice(&abi_bytes).map_err(|e| format!("parse ABI metadata: {e}"))?;
    let filename = path
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or("invalid ABI filename")?;
    let asset_name = filename
        .strip_suffix(".abi.json")
        .filter(|s| cache::safe_component(s))
        .ok_or("ABI path must be <asset>.abi.json")?
        .to_owned();
    let version_bytes = fs::read(path.with_file_name(format!("{asset_name}.version.json")))
        .map_err(|e| format!("read version metadata: {e}"))?;
    let version: Value = serde_json::from_slice(&version_bytes)
        .map_err(|e| format!("parse version metadata: {e}"))?;
    if abi["schema_version"] != 1
        || version["schema_version"] != 1
        || abi["asset_name"] != asset_name
        || abi["binary_name"] != "ferrum"
        || abi["backend"] != "metal"
    {
        return Err("metadata must describe one schema-1 Ferrum Metal archive".into());
    }
    for key in [
        "asset_name",
        "asset_sha256",
        "binary_name",
        "binary_sha256",
        "release_candidate_sha",
        "release_candidate_tag",
        "staging_label",
        "workflow_run_id",
        "workflow_run_attempt",
    ] {
        field(&abi, key)?;
        if abi[key] != version[key] {
            return Err(format!("ABI/version metadata disagree on {key}"));
        }
    }
    for key in ["asset_sha256", "binary_sha256"] {
        let digest = field(&abi, key)?;
        if digest.len() != 64 || !cache::revision(&digest) {
            return Err(format!("invalid metadata {key}"));
        }
    }
    let identity = CandidateInput {
        version: field(&version, "version")?,
        release_candidate_sha: field(&abi, "release_candidate_sha")?,
        release_candidate_tag: field(&abi, "release_candidate_tag")?,
        staging_label: field(&abi, "staging_label")?,
        workflow_run_id: field(&abi, "workflow_run_id")?,
        workflow_run_attempt: field(&abi, "workflow_run_attempt")?,
    };
    validate_candidate(&identity)?;
    if identity.release_candidate_sha != expected_commit {
        return Err("binary metadata source differs from planned commit".into());
    }
    let target = field(&abi, "target_triple")?;
    if !["aarch64-apple-darwin", "x86_64-apple-darwin"].contains(&target.as_str()) {
        return Err("performance metadata requires a macOS Metal target".into());
    }
    let binary_sha256 = field(&abi, "binary_sha256")?;
    Ok(Metadata {
        abi_bytes,
        version_bytes,
        identity,
        asset_name,
        binary_sha256,
        target,
    })
}
fn binary_bindings(
    document: &PlanDocument,
    baseline: &Metadata,
    candidate: &Metadata,
) -> Result<(), String> {
    if document.provenance.release_base_tag != format!("v{}", baseline.identity.version)
        || baseline.target != candidate.target
    {
        return Err(
            "baseline formal tag or ABI differs from the planned same-host comparison".into(),
        );
    }
    validate_version_progression(&baseline.identity.version, &candidate.identity.version)
}
fn preserve_metadata(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(mut file) => file.write_all(bytes).map_err(|e| e.to_string()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if fs::read(path).map_err(|e| e.to_string())? == bytes {
                Ok(())
            } else {
                Err(format!("frozen metadata differs: {}", path.display()))
            }
        }
        Err(error) => Err(error.to_string()),
    }
}
/// This only freezes expected inputs. Neither a cache hit nor an archive digest is a runtime pass.
pub async fn prepare(args: PerformancePrepareArgs) -> Result<(), String> {
    if args.output_dir.exists() && !args.output_dir.is_dir() {
        return Err("performance task root must be a directory".into());
    }
    if !cache::safe_component(&args.profile_id) || args.output_dir.join(&args.profile_id).exists() {
        return Err("performance profile directory must be new and have a safe name".into());
    }
    let plan_bytes = fs::read(&args.plan).map_err(|e| format!("read performance plan: {e}"))?;
    let document: PlanDocument =
        serde_json::from_slice(&plan_bytes).map_err(|e| format!("performance plan: {e}"))?;
    let selected = requirements(&document, &args.profile_id)?;
    let policy: PerformancePolicy = read_json(&args.policy)?;
    policy.validate()?;
    let baseline = metadata(&args.baseline_abi, &document.provenance.base)?;
    let candidate = metadata(&args.candidate_abi, &document.provenance.candidate)?;
    binary_bindings(&document, &baseline, &candidate)?;
    let selector: cache::SourceSelector = read_json(&args.source_selector)?;
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(policy.task_timeout_secs))
        .ok_or("performance preparation deadline overflow")?;
    let (manifest, resolved) =
        cache::resolve(&args.hf_cache, &selector, &selected.profile, deadline)?;
    let expected = ExpectedPerformanceRun {
        schema_version: 1,
        profile: selected.profile,
        baseline_version: baseline.identity.version.clone(),
        baseline_sha256: baseline.binary_sha256.clone(),
        candidate_version: candidate.identity.version.clone(),
        candidate_sha256: candidate.binary_sha256.clone(),
        client_version: candidate.identity.version.clone(),
        client_sha256: candidate.binary_sha256.clone(),
        policy,
        source: manifest.identity()?,
        obligations: selected.obligations,
    };
    expected.validate()?;
    fs::create_dir_all(&args.output_dir)
        .map_err(|e| format!("create performance task root: {e}"))?;
    // Exclusive profile creation prevents a resumed command overwriting frozen inputs.
    let task = args.output_dir.join(&args.profile_id);
    fs::create_dir(&task).map_err(|e| e.to_string())?;
    write_json(&task.join("source-manifest.json"), &manifest)?;
    write_json(
        &task.join("prepare-evidence.json"),
        &json!({
            "schema_version": 1, "status": "prepared", "release_approved": false,
            "plan_sha256": format!("{:x}", Sha256::digest(&plan_bytes)),
            "base_commit": document.provenance.base, "candidate_commit": document.provenance.candidate,
            "release_base_tag": document.provenance.release_base_tag,
            "source_selector": selector, "resolved_source": resolved,
            "source": expected.source, "obligations": expected.obligations,
            "archive_bytes_verified": false, "models_executed": false
        }),
    )?;
    for (role, metadata) in [("baseline", &baseline), ("candidate", &candidate)] {
        let directory = args.output_dir.join(role);
        fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        preserve_metadata(
            &directory.join(format!("{}.abi.json", metadata.asset_name)),
            &metadata.abi_bytes,
        )?;
        preserve_metadata(
            &directory.join(format!("{}.version.json", metadata.asset_name)),
            &metadata.version_bytes,
        )?;
    }
    // Written last: errors above cannot leave a complete expected task behind.
    write_json(&task.join("expected-task.json"), &expected)
}
#[cfg(test)]
#[path = "prepare_tests.rs"]
mod tests;
