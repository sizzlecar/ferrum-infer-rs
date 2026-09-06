//! Prepare model tasks from a release plan and verify actual runner reports.
//! This gate covers model execution only; it never authorizes publication.
use clap::{Parser, Subcommand};
use ferrum_bench_core::release_regression::model_schedule::model_task_schedule;
use ferrum_bench_core::release_regression::model_tasks::{
    verify_model_reports, ExpectedModelRun, DEFAULT_FUNCTIONAL_CAPACITY, DEFAULT_STOP_PROMPT,
};
use ferrum_bench_core::release_regression::{Backend, Gap, Plan};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

#[derive(Parser)]
#[command(about = "Prepare and verify required model tasks; model-runtime scope only")]
struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Emit one expectation per selected profile, binding the staged binary bytes.
    Prepare {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        version: String,
        /// ABI metadata from staging; reads its adjacent .version.json record too.
        #[arg(long, required = true)]
        abi: Vec<PathBuf>,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..))]
        max_tokens: u32,
    },
    /// Missing, failed, duplicate and unfinished reports fail this model gate.
    Verify {
        #[arg(long)]
        tasks: PathBuf,
        #[arg(long)]
        report: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedTasks {
    schema_version: u32,
    expectations: Vec<ExpectedModelRun>,
    unsupported_obligations: Vec<usize>,
    remaining_plan_gaps: Vec<Gap>,
}
#[derive(Debug)]
struct StagedBinary {
    backend: Backend,
    sha256: String,
}
fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?)
        .map_err(|e| format!("parse {}: {e}", path.display()))
}
fn write_new(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| format!("create {}: {e}", path.display()))?;
    file.write_all(&bytes)
        .map_err(|e| format!("write {}: {e}", path.display()))
}
fn staged_binary(
    abi: &Value,
    version: &Value,
    expected_version: &str,
    expected_candidate: &str,
) -> Result<StagedBinary, String> {
    if abi["schema_version"] != 1 || version["schema_version"] != 1 {
        return Err("unsupported staged metadata schema".into());
    }
    for field in [
        "asset_name",
        "asset_sha256",
        "binary_sha256",
        "release_candidate_sha",
    ] {
        if abi[field].as_str().is_none_or(str::is_empty) || abi[field] != version[field] {
            return Err(format!("staged ABI and version disagree or omit {field}"));
        }
    }
    // Bind the scope calculation to the artifact's source inputs. This association
    // does not establish runtime correctness; the runner must still execute cases.
    if expected_candidate.is_empty() || abi["release_candidate_sha"] != expected_candidate {
        return Err("staged binary source does not match the plan candidate".into());
    }
    if version["version"] != expected_version {
        return Err("staged binary version does not match task version".into());
    }
    let sha256 = abi["binary_sha256"]
        .as_str()
        .ok_or("missing binary digest")?;
    if sha256.len() != 64
        || !sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("invalid staged binary SHA-256".into());
    }
    let backend =
        serde_json::from_value(abi["backend"].clone()).map_err(|_| "invalid staged backend")?;
    Ok(StagedBinary {
        backend,
        sha256: sha256.into(),
    })
}
fn prepare(
    plan: &Plan,
    assets: &[StagedBinary],
    version: &str,
    max_tokens: u32,
) -> Result<PreparedTasks, String> {
    let version_value = semver::Version::parse(version).map_err(|e| e.to_string())?;
    if !version_value.pre.is_empty() || !version_value.build.is_empty() || max_tokens == 0 {
        return Err("task version must be formal and output budget positive".into());
    }
    let mut by_backend = BTreeMap::new();
    for asset in assets {
        if by_backend.insert(asset.backend, asset).is_some() {
            return Err(format!(
                "multiple staged binaries for {:?}; choose the intended asset explicitly",
                asset.backend
            ));
        }
    }
    let schedule = model_task_schedule(plan);
    let mut expectations = Vec::new();
    for run in schedule.runs {
        let asset = by_backend
            .get(&run.profile.target.backend)
            .ok_or_else(|| format!("no staged binary for profile {}", run.profile.id))?;
        let runtime_capacity = if run.quick_start {
            None
        } else {
            Some(DEFAULT_FUNCTIONAL_CAPACITY)
        };
        if let Some(capacity) = runtime_capacity {
            capacity.validate(max_tokens)?;
        }
        expectations.push(ExpectedModelRun {
            profile: run.profile,
            binary_sha256: asset.sha256.clone(),
            version: version.into(),
            checks: run.checks,
            disable_thinking: run.quick_start,
            use_default_backend: run.quick_start,
            max_tokens,
            runtime_capacity,
            reasoning_alias_replay: false,
            stop_prompt: DEFAULT_STOP_PROMPT.into(),
        });
    }
    Ok(PreparedTasks {
        schema_version: 1,
        expectations,
        unsupported_obligations: schedule.unsupported_obligations,
        remaining_plan_gaps: plan.gaps.clone(),
    })
}
fn run(action: Action) -> Result<(), String> {
    match action {
        Action::Prepare {
            plan,
            version,
            abi,
            output_dir,
            max_tokens,
        } => {
            let document = read_json(&plan)?;
            if document["schema_version"] != 2 {
                return Err("model tasks require regression plan schema 2".into());
            }
            let candidate = document["provenance"]["candidate"]
                .as_str()
                .filter(|value| !value.is_empty())
                .ok_or("plan omits candidate provenance")?;
            let plan: Plan = serde_json::from_value(document["plan"].clone())
                .map_err(|e| format!("read typed plan: {e}"))?;
            let mut assets = Vec::new();
            for path in abi {
                let name = path
                    .file_name()
                    .and_then(|v| v.to_str())
                    .and_then(|v| v.strip_suffix(".abi.json"))
                    .ok_or("--abi must refer to a staged .abi.json metadata file")?;
                let version_path = path.with_file_name(format!("{name}.version.json"));
                assets.push(staged_binary(
                    &read_json(&path)?,
                    &read_json(&version_path)?,
                    &version,
                    candidate,
                )?);
            }
            let tasks = prepare(&plan, &assets, &version, max_tokens)?;
            // Reserve a fresh directory so a retry cannot reuse partial or stale tasks.
            fs::create_dir(&output_dir).map_err(|e| format!("create task directory: {e}"))?;
            write_new(&output_dir.join("tasks.json"), &tasks)?;
            for (index, task) in tasks.expectations.iter().enumerate() {
                write_new(&output_dir.join(format!("task-{index}.json")), task)?;
            }
            if !tasks.unsupported_obligations.is_empty() {
                return Err(format!("model checks do not implement obligations {:?}; complete the bindings before allocating hardware; inventory saved in {}",tasks.unsupported_obligations,output_dir.display()));
            }
            println!(
                "Prepared model tasks in {}; other plan gaps remain separate",
                output_dir.display()
            );
        }
        Action::Verify {
            tasks,
            report,
            output,
        } => {
            let tasks: PreparedTasks = serde_json::from_value(read_json(&tasks)?)
                .map_err(|e| format!("read model tasks: {e}"))?;
            if tasks.schema_version != 1 {
                return Err("unsupported model task schema".into());
            }
            let reports = report
                .iter()
                .map(|p| read_json(p))
                .collect::<Result<Vec<_>, _>>()?;
            let mut issues = verify_model_reports(&tasks.expectations, &reports)
                .err()
                .unwrap_or_default();
            if !tasks.unsupported_obligations.is_empty() {
                issues.push(format!(
                    "unsupported model obligations: {:?}",
                    tasks.unsupported_obligations
                ));
            }
            let passed = issues.is_empty();
            write_new(
                &output,
                &json!({"schema_version":1,"scope":"model_runtime","model_tasks_passed":passed,
                "issues":issues,"remaining_plan_gaps":tasks.remaining_plan_gaps,"release_approved":false}),
            )?;
            if !passed {
                return Err(format!(
                    "model task verification failed; see {}",
                    output.display()
                ));
            }
            println!("Required model tasks passed; installation, CI, numerical and other release checks remain separate");
        }
    }
    Ok(())
}
fn main() -> ExitCode {
    match run(Args::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("model gate: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
#[path = "model_gate/tests.rs"]
mod tests;
