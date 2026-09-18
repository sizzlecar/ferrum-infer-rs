//! Cross-process continuation of one already verified, task-private session.
//! The original evidence is immutable; a separate copy is the only writable journal.
use super::{
    config::{self, Manifest},
    orchestral, orchestral_evidence, orchestral_wire,
    process::{self, ManagedChild, Outcome},
    proxy::Proxy,
    write_json,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::process::Command;

#[path = "orchestral_resume/storage.rs"]
mod storage;

#[derive(clap::Args)]
pub(crate) struct Args {
    #[arg(long)]
    pub source_report: PathBuf,
    #[arg(long)]
    pub task_id: String,
    #[arg(long)]
    pub follow_up_file: PathBuf,
    /// Must not exist, and must be outside the original report and workspace.
    #[arg(long)]
    pub report_dir: PathBuf,
    #[arg(long, default_value_t = 600)]
    pub timeout_secs: u64,
    /// Optional semantic assertion, independent of the exact history proof.
    #[arg(long)]
    pub expect_output_contains: Option<String>,
}

pub(crate) async fn run(args: &Args) -> Result<i32> {
    ensure!(args.timeout_secs > 0, "timeout must be positive");
    ensure!(
        args.expect_output_contains
            .as_ref()
            .is_none_or(|text| !text.trim().is_empty()),
        "semantic assertion cannot be empty"
    );
    let source = args.source_report.canonicalize()?;
    let manifest = Manifest::load(&source.join("manifest.json"))?;
    let task = manifest
        .tasks
        .iter()
        .find(|task| task.id == args.task_id)
        .context("task is not in the source manifest")?;
    let output = storage::new_report(&args.report_dir, &source, &task.workdir)?;
    let frozen = config::snapshot(std::slice::from_ref(&source))?;
    write_json(output.join("source-sha256.json"), &frozen)?;
    let result = execute(args, &source, &output, &manifest).await;
    let unchanged =
        config::snapshot(std::slice::from_ref(&source)).is_ok_and(|current| current == frozen);
    let (accepted, evidence) = match result {
        Ok(evidence) => (evidence["accepted"] == true && unchanged, evidence),
        Err(error) => (false, json!({"error":format!("{error:#}")})),
    };
    write_json(
        output.join("report.json"),
        &json!({
            "schema_version":1, "mode":"orchestral_cross_process_resume", "accepted":accepted,
            "source_report":source, "task_id":args.task_id, "source_report_unchanged":unchanged,
            "scope":"One finished private session copied into a new process; exact uncompacted HTTP history and immutable public journal prefix. No new code-task validation or performance claim.",
            "evidence":evidence
        }),
    )?;
    Ok(if accepted { 0 } else { 1 })
}

async fn execute(args: &Args, source: &Path, output: &Path, manifest: &Manifest) -> Result<Value> {
    let task = manifest
        .tasks
        .iter()
        .find(|task| task.id == args.task_id)
        .context("missing task")?;
    let spec = manifest
        .orchestral()
        .context("source agent must be Orchestral")?;
    let summary = storage::read_json(&source.join("report.json"))?;
    let prior_result = storage::read_json(&source.join(&task.id).join("result.json"))?;
    ensure!(
        summary["accepted"] == true && summary["tasks_completed"] == true,
        "source task report was not accepted"
    );
    ensure!(
        prior_result["id"] == task.id
            && prior_result["completed"] == true
            && prior_result["process"]["exit_code"] == 0
            && prior_result["process"]["timed_out"] == false
            && prior_result["validation"]["exit_code"] == 0
            && prior_result["validation"]["timed_out"] == false,
        "source task/process/independent validation is incomplete"
    );
    let session = prior_result["orchestral"]["session_id"]
        .as_str()
        .context("missing source session")?;
    let task_source = source.join(&task.id);
    ensure!(
        orchestral_wire::audit_saved(source, &task.id, &output.join("source-audit.json"), None)?
            == 0,
        "source public/HTTP evidence did not pass a fresh audit"
    );
    let prior = orchestral_evidence::read(&task_source.join("journals"), session);
    ensure!(prior.complete(), "source public lifecycle is incomplete");
    let prior_records = storage::read_json(
        prior
            .session_path
            .as_ref()
            .context("missing Session path")?,
    )?;
    let prior_records = prior_records.as_array().context("Session array")?;
    let (records, _) = orchestral_wire::load_records(&source.join("requests"), &task.id)?;
    let wire = orchestral_wire::bind(
        &prior,
        &records.iter().collect::<Vec<_>>(),
        &source.join("requests"),
        spec.tool_result_format,
    );
    let terminal_index = wire
        .terminal
        .context("missing verified source terminal")?
        .request_index;
    let terminal = records
        .iter()
        .find(|record| record.request_index == terminal_index)
        .context("missing terminal HTTP record")?;
    let followup = fs::read_to_string(&args.follow_up_file)?;
    ensure!(!followup.trim().is_empty(), "followup must not be empty");
    fs::write(output.join("follow-up.txt"), &followup)?;
    let expected =
        orchestral_wire::continuation_history(terminal, &source.join("requests"), &followup)?;
    write_json(output.join("expected-history.json"), &expected)?;
    let journal = output.join("journals");
    let copied = storage::copy_session(&prior, &journal)?;
    write_json(output.join("copied-journal-sha256.json"), &copied.hashes)?;
    let home = output.join("orchestral-home");
    let artifacts = output.join("artifacts");
    fs::create_dir(&home)?;
    fs::create_dir(&artifacts)?;
    // This bounded mode handles inline evidence, not external artifact relocation.
    ensure!(
        fs::read_dir(task_source.join("artifacts"))?
            .next()
            .is_none(),
        "resume requires inline evidence; source artifacts need an explicit relocation contract"
    );
    let workspace_before = config::snapshot(std::slice::from_ref(&task.workdir))?;
    let binary_before = config::hash_file(manifest.program())?;
    let clock = Instant::now();
    let proxy = Proxy::start(
        &manifest.server,
        &format!("{}-resume", manifest.run_id),
        BTreeSet::from([task.id.clone()]),
        output.join("requests"),
        clock,
    )
    .await?;
    let base_url = proxy.task_base_url(&task.id);
    let mut config = storage::read_json(&task_source.join("orchestral-config.json"))?;
    config["providers"]["backends"][0]["endpoint"] = json!(base_url);
    config["journal"]["root_dir"] = json!(journal);
    config["artifacts"]["root_dir"] = json!(artifacts);
    config["observability"]["log_file"] = json!(output.join("orchestral.log"));
    orchestral::validate_config(
        &config,
        &manifest.server,
        spec.tool_result_format,
        &base_url,
        &journal,
        &artifacts,
        &output.join("orchestral.log"),
    )?;
    ensure!(
        config["agent"]["compaction"]["enabled"] == false
            && config["agent"]["input_requests_enabled"] == false
            && config["agent"]["project_instructions"]["enabled"] == false,
        "resume requires explicit compaction/input-requests/project-instructions disabled"
    );
    let config_path = output.join("orchestral-config.json");
    write_json(&config_path, &config)?;
    let argv = orchestral::argv(&config_path, session, &followup)?;
    write_json(
        output.join("command.json"),
        &json!({"program":manifest.program(), "program_sha256":binary_before,
        "args":argv,"cwd":task.workdir,"ORCHESTRAL_HOME":home,"session_id":session,"config_sha256":config::hash_file(&config_path)?}),
    )?;
    let mut command = Command::new(manifest.program());
    process::isolated_local(&mut command);
    command
        .args(&argv)
        .current_dir(&task.workdir)
        .env("ORCHESTRAL_HOME", &home)
        .stdin(Stdio::null())
        .stdout(File::create(output.join("orchestral-stdout.txt"))?)
        .stderr(File::create(output.join("orchestral-stderr.txt"))?);
    let mut child = ManagedChild::spawn(&mut command)?;
    let pid = child.pid;
    if let Err(error) = write_json(
        output.join("process-start.json"),
        &json!({"pid":pid,"session_id":session}),
    ) {
        child.interrupt_then_stop().await;
        return Err(error);
    }
    let waited =
        tokio::time::timeout(Duration::from_secs(args.timeout_secs), child.child.wait()).await;
    let (exit_code, timed_out) = match waited {
        Ok(Ok(status)) => {
            child.stop().await;
            (status.code(), false)
        }
        Ok(Err(error)) => {
            child.interrupt_then_stop().await;
            return Err(error.into());
        }
        Err(_) => {
            child.interrupt_then_stop().await;
            (None, true)
        }
    };
    let process = Outcome {
        exit_code,
        timed_out,
        elapsed_ms: clock.elapsed().as_millis(),
    };
    write_json(output.join("process-result.json"), &process)?;
    proxy.drain(manifest.server.request_timeout_secs).await?;
    let requests = proxy.state.records.lock().unwrap().clone();
    write_json(output.join("requests.json"), &requests)?;
    let new_run = storage::check_copied_session(&copied)?;
    let public = orchestral_evidence::read_continuation(
        &new_run,
        &copied.session_path,
        session,
        prior_records,
    );
    let wire = orchestral_wire::bind_continuation(
        &public,
        &requests.iter().collect::<Vec<_>>(),
        &output.join("requests"),
        spec.tool_result_format,
        &expected,
    );
    let workspace_unchanged =
        config::snapshot(std::slice::from_ref(&task.workdir))? == workspace_before;
    let binary_unchanged = config::hash_file(manifest.program())? == binary_before;
    let semantic_match = args.expect_output_contains.as_ref().map(|text| {
        public
            .output
            .as_ref()
            .is_some_and(|output| output.contains(text))
    });
    let accepted = exit_code == Some(0)
        && !timed_out
        && workspace_unchanged
        && binary_unchanged
        && public.complete()
        && public.compaction_events == 0
        && public.tool_exchanges.is_empty()
        && public.input.as_deref() == Some(followup.as_str())
        && wire.complete()
        && requests.len() == 1
        && semantic_match != Some(false);
    Ok(
        json!({"accepted":accepted,"process":process,"pid":pid,"session_id":session,
        "prior_run_id":prior.run_id,"prior_session_records":prior_records.len(),"public":public,"wire":wire,
        "requests":requests,"workspace_unchanged":workspace_unchanged,"binary_unchanged":binary_unchanged,
        "expected_output_contains":args.expect_output_contains,"semantic_match":semantic_match,
        "semantic_scope":"Optional substring assertion only; exact HTTP history is required independently, even when the answer contains the expected text."}),
    )
}
