use super::{
    config::{self, Manifest, Task},
    events::Events,
    orchestral, orchestral_evidence, orchestral_wire,
    process::{self, ManagedChild, Outcome},
    proxy::{self, Proxy},
    repair, replay, write_json, Mode,
};
use anyhow::{ensure, Context, Result};
use futures::future::join_all;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
    sync::Barrier,
};

#[derive(Serialize)]
pub(crate) struct TaskResult {
    pub id: String,
    pub pid: u32,
    pub workdir: PathBuf,
    pub started_ns: u64,
    pub finished_ns: u64,
    pub process: Outcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub events: Option<Events>,
    pub validation: Option<Outcome>,
    pub source_changed: bool,
    pub closed_loop: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replay: Option<replay::Evidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orchestral: Option<orchestral_evidence::Evidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orchestral_transport: Option<orchestral_wire::Evidence>,
    pub completed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_repair: Option<repair::Report>,
}

pub(crate) async fn run(
    manifest_path: &Path,
    report_dir: &Path,
    mode: Mode,
    profile: Option<&Path>,
    validation_repairs: u32,
) -> Result<i32> {
    let manifest = Manifest::load(manifest_path)?;
    ensure!(
        manifest.orchestral().is_none() || validation_repairs == 0,
        "Orchestral supports validation_repairs=0 only; same-session repair is not implemented"
    );
    ensure!(
        !report_dir.exists() || fs::read_dir(report_dir)?.next().is_none(),
        "report directory must be new or empty"
    );
    fs::create_dir_all(report_dir)?;
    let report_dir = report_dir.canonicalize()?;
    let mut protected = vec![manifest_path.canonicalize()?, manifest.program().to_owned()];
    if let Some(spec) = manifest.orchestral() {
        protected.push(spec.config_template.clone());
    } else {
        for arg in &manifest.pi_program()?.args {
            let path = Path::new(arg);
            if path.is_absolute() && path.is_file() {
                protected.push(path.canonicalize()?);
            }
        }
    }
    for task in &manifest.tasks {
        ensure!(
            config::disjoint(&report_dir, &task.workdir),
            "report directory overlaps agent workdir"
        );
        protected.extend(task.validation.protected_paths.iter().cloned());
        protected.extend([task.validation.program.clone(), task.prompt_file.clone()]);
    }
    for path in &protected {
        ensure!(
            config::disjoint(&report_dir, path),
            "report directory overlaps protected input"
        );
    }
    let mut frozen = config::snapshot(&protected)?;
    write_json(report_dir.join("protected-before.json"), &frozen)?;
    write_json(report_dir.join("manifest.json"), &manifest)?;
    let control = report_dir.join("control-agent");
    fs::create_dir(&control)?;
    let mut version_args = manifest
        .pi_program()
        .map(|pi| pi.args.clone())
        .unwrap_or_default();
    version_args.push("--version".into());
    let version = process::logged(
        manifest.program(),
        &version_args,
        &report_dir,
        &control,
        &report_dir.join(if manifest.orchestral().is_some() {
            "orchestral-version"
        } else {
            "pi-version"
        }),
        30,
    )
    .await?;
    ensure!(
        version.exit_code == Some(0) && !version.timed_out,
        "agent version probe failed"
    );
    let mut initial = BTreeMap::new();
    for task in &manifest.tasks {
        let outcome = validate(
            task,
            &control,
            &report_dir.join(&task.id).join("initial-validation"),
            validation_repairs > 0,
        )
        .await?;
        let valid = !outcome.timed_out
            && outcome.exit_code == Some(task.validation.expected_initial_exit_code);
        initial.insert(task.id.clone(), outcome);
        write_json(report_dir.join("initial-validations.json"), &initial)?;
        ensure!(
            valid,
            "{} initial validator did not demonstrate the expected behavior failure",
            task.id
        );
    }
    ensure!(
        config::snapshot(&protected)? == frozen,
        "protected input changed during preflight"
    );
    let task_ids: BTreeSet<_> = manifest.tasks.iter().map(|t| t.id.clone()).collect();
    let clock = Instant::now();
    let wall_start = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let proxy = Proxy::start(
        &manifest.server,
        &manifest.run_id,
        task_ids.clone(),
        report_dir.join("requests"),
        clock,
    )
    .await?;
    let barrier = (mode == Mode::Concurrent).then(|| Arc::new(Barrier::new(manifest.tasks.len())));
    // All process setup happens before the shared start barrier. The barrier is
    // launch coordination only; acceptance additionally checks engine progress.
    let mut prepared = Vec::new();
    for task in &manifest.tasks {
        let output = report_dir.join(&task.id);
        if let Some(spec) = manifest.orchestral() {
            let client = orchestral::prepare(
                &manifest,
                spec,
                task,
                &output,
                &proxy.task_base_url(&task.id),
            )?;
            protected.push(client.config_path.clone());
            let before = source_snapshot(&task.workdir, &output.join("source-before"))?;
            prepared.push((task, output, client.home.clone(), before, Some(client)));
            continue;
        }
        let agent_dir = output.join("pi-agent");
        fs::create_dir(&agent_dir)?;
        fs::create_dir(output.join("sessions"))?;
        let s = &manifest.server;
        write_json(
            agent_dir.join("models.json"),
            &json!({"providers":{"ferrum-local":{
                "baseUrl":proxy.base_url,"api":"openai-completions","apiKey":"local",
                "headers":{proxy::TASK_HEADER:task.id},
                "compat":{"supportsDeveloperRole":false,"supportsReasoningEffort":false,"maxTokensField":"max_tokens"},
                "models":[{"id":s.model,"name":s.model,"reasoning":s.reasoning,"input":["text"],
                    "contextWindow":s.context_window,"maxTokens":s.max_tokens,"samplingParams":s.sampling_params}]
            }}}),
        )?;
        // pi's default 16k compaction reserve can consume an entire small local
        // context. Reserve the declared response budget, consistently for all tasks.
        write_json(
            agent_dir.join("settings.json"),
            &json!({"compaction":{"enabled":true,"reserveTokens":s.max_tokens,"keepRecentTokens":s.context_window / 4},
                "httpIdleTimeoutMs":s.request_timeout_secs.saturating_mul(1000),
                "retry":{"enabled":true,"maxRetries":3,"baseDelayMs":2000,
                    "provider":{"timeoutMs":s.request_timeout_secs.saturating_mul(1000),"maxRetries":0}}}),
        )?;
        let before = source_snapshot(&task.workdir, &output.join("source-before"))?;
        if manifest.schema_version == 2 {
            protected.extend([
                agent_dir.join("models.json"),
                agent_dir.join("settings.json"),
            ]);
        }
        prepared.push((task, output, agent_dir, before, None));
    }
    // Generated configurations are frozen before any participant crosses the barrier.
    // Legacy schema 1 keeps its original protected-input/report behavior.
    if manifest.schema_version == 2 {
        frozen = config::snapshot(&protected)?;
        write_json(report_dir.join("protected-before.json"), &frozen)?;
    }
    let results = if mode == Mode::Concurrent {
        join_all(
            prepared
                .into_iter()
                .map(|(task, output, agent_dir, before, orchestral)| {
                    run_task(
                        &manifest,
                        task,
                        output,
                        agent_dir,
                        before,
                        clock,
                        barrier.clone(),
                        validation_repairs,
                        Arc::clone(&proxy.state),
                        &protected,
                        &frozen,
                        orchestral,
                    )
                }),
        )
        .await
    } else {
        let mut results = Vec::new();
        for (task, output, agent_dir, before, orchestral) in prepared {
            results.push(
                run_task(
                    &manifest,
                    task,
                    output,
                    agent_dir,
                    before,
                    clock,
                    None,
                    validation_repairs,
                    Arc::clone(&proxy.state),
                    &protected,
                    &frozen,
                    orchestral,
                )
                .await,
            );
        }
        results
    };
    let elapsed_ms = clock.elapsed().as_millis();
    let mut tasks = Vec::new();
    let mut infrastructure_errors = Vec::new();
    for (task, result) in manifest.tasks.iter().zip(results) {
        match result {
            Ok(result) => tasks.push(result),
            Err(error) => infrastructure_errors.push(format!("{}: {error:#}", task.id)),
        }
    }
    if let Err(error) = proxy.drain(manifest.server.request_timeout_secs).await {
        infrastructure_errors.push(format!("proxy terminal evidence: {error:#}"));
    }
    let requests = proxy.state.records.lock().expect("request records").clone();
    let after = match config::snapshot(&protected) {
        Ok(after) => after,
        Err(error) if manifest.schema_version == 2 => {
            infrastructure_errors.push(format!("protected input inspection: {error:#}"));
            BTreeMap::new()
        }
        Err(error) => return Err(error),
    };
    let protected_unchanged = frozen == after;
    write_json(report_dir.join("protected-after.json"), &after)?;
    for task in &mut tasks {
        let records: Vec<_> = requests.iter().filter(|r| r.task_id == task.id).collect();
        if let Some(evidence) = &mut task.orchestral {
            let binding = orchestral_wire::bind(evidence, &records, &report_dir.join("requests"));
            task.closed_loop = evidence.complete() && binding.complete();
            task.orchestral_transport = Some(binding);
        } else if let Some(events) = &task.events {
            task.replay = Some(
                match task
                    .validation_repair
                    .as_ref()
                    .and_then(repair::Report::session_file)
                {
                    Some(path) => replay::verify_with_session(events, &records, path),
                    None => replay::verify(events, &records),
                },
            );
            if let Some(repair) = &mut task.validation_repair {
                repair.bind_requests(&records);
                write_json(
                    report_dir.join(&task.id).join("validation-repair.json"),
                    repair,
                )?;
                if repair.termination == repair::Termination::InfrastructureFailure {
                    infrastructure_errors.push(format!(
                        "{} validation repair: {}",
                        task.id,
                        repair
                            .error
                            .as_deref()
                            .unwrap_or("independent validation infrastructure failed")
                    ));
                }
            }
            task.closed_loop = events.completed_loop(&manifest.server.model)
                && task
                    .replay
                    .as_ref()
                    .is_some_and(|replay| replay.complete(events));
        }
        let last = records.iter().max_by_key(|r| r.request_index);
        task.completed &= protected_unchanged
            && task.closed_loop
            && last.is_some_and(|r| r.error.is_none() && r.saw_done && r.http_status == Some(200));
        write_json(report_dir.join(&task.id).join("result.json"), task)?;
    }
    let distinct_sessions: BTreeSet<_> = tasks
        .iter()
        .filter_map(|t| match (&t.events, &t.orchestral) {
            (Some(events), None) => events.session_id.as_deref(),
            (None, Some(evidence)) => evidence.session_id.as_deref(),
            _ => None,
        })
        .collect();
    let distinct_processes: BTreeSet<_> = tasks.iter().map(|t| t.pid).collect();
    if distinct_sessions.len() != tasks.len() || distinct_processes.len() != tasks.len() {
        infrastructure_errors.push("task processes/sessions were not distinct".into());
    }
    let agent_wall_ms = tasks
        .iter()
        .map(|t| t.started_ns)
        .min()
        .zip(tasks.iter().map(|t| t.finished_ns).max())
        .map(|(start, end)| end.saturating_sub(start) / 1_000_000);
    let transport: Vec<_> = requests
        .iter()
        .filter_map(|r| {
            r.server_request_id.as_ref().map(|id| proxy::ProgressSpan {
                task_id: r.task_id.clone(),
                request_id: id.clone(),
                timestamps_ns: r.progress_ns.clone(),
                clock_error_ns: 0,
            })
        })
        .collect();
    let transport_overlap = proxy::overlap(&task_ids, &transport);
    let engine = match profile {
        Some(path) => match proxy::engine_spans(path, &requests) {
            Ok(spans) => spans,
            Err(error) => {
                infrastructure_errors.push(format!("engine evidence: {error:#}"));
                Vec::new()
            }
        },
        None => Vec::new(),
    };
    let engine_overlap = proxy::overlap(&task_ids, &engine);
    let tasks_completed = tasks.len() == manifest.tasks.len() && tasks.iter().all(|t| t.completed);
    let accepted = tasks_completed
        && infrastructure_errors.is_empty()
        && (mode == Mode::Sequential || engine_overlap.is_some());
    let mut report = json!({"schema_version":manifest.schema_version,"run_id":manifest.run_id,"mode":mode,
        "server":manifest.server,"pi":manifest.pi,"wall_start_unix_ns":wall_start,"elapsed_ms":elapsed_ms,
        "validation_repairs":validation_repairs,"agent_mode":if validation_repairs == 0 { "print" } else { "rpc" },
        "validation_repair_cleanup": if validation_repairs == 0 { None } else { Some(if cfg!(unix) {
            "ManagedChild owns Unix process groups; structured validators propagate controller SIGTERM to their current Cargo/test group through cancellation. Cleanup is not additional agent work."
        } else { "Existing ManagedChild direct-child termination; descendant process-group cleanup is not certified on this platform." }) },
        "agent_wall_ms":agent_wall_ms,
        "initial_validations":initial,"tasks":tasks,"requests":requests,
        "protected_unchanged":protected_unchanged,"tasks_completed":tasks_completed,
        "transport_progress_overlap":transport_overlap,"engine_progress_overlap":engine_overlap,
        "server_profile_jsonl":profile,"engine_progress_spans":engine,"infrastructure_errors":infrastructure_errors,
        "accepted":accepted,
        "concurrency_evidence_note":"HTTP and SSE overlap alone do not certify inference overlap. Engine token commits are matched by request id. Work directories are isolated inputs, not an OS sandbox.",
        "replay_evidence_note":"Every invocation requires an origin response id and a successful following request carrying its exact assistant turn and tool result. Compacted history additionally requires the observed session identity, retained ancestor chain and complete following wire projection; missing evidence stays unproven. Independent semantic validation is reported separately.",
        "retry_policy":{"agent":{"enabled":true,"max_retries":3,"base_delay_ms":2000},"provider":{"max_retries":0,"request_timeout_secs":manifest.server.request_timeout_secs},
            "task_deadline_note":if validation_repairs == 0 { "Each task timeout covers the entire pi process including outer retries; the per-request timeout is not a total task budget." }
                else { "One absolute task deadline covers the Pi RPC process, automatic retries, all independent validations and repair rounds; it is never reset. Shutdown cleanup does not authorize additional agent work." }}});
    if manifest.schema_version == 2 {
        report["agent"] = serde_json::to_value(&manifest.agent)?;
        report.as_object_mut().expect("report object").remove("pi");
        if validation_repairs == 0 {
            report["retry_policy"]["task_deadline_note"] = json!("One absolute task deadline covers the agent process and independent final validation. Shutdown cleanup does not authorize additional agent work.");
        }
        if manifest.orchestral().is_some() {
            report["agent_mode"] = json!("orchestral_headless");
            report["retry_policy"] = json!({"validation_repairs":0,
                "task_deadline_note":"One absolute deadline includes the headless process and independent validation. Timeout cleanup cannot authorize new agent work.",
                "agent_retry_note":"No harness repair/retry rounds. Product retry/context behavior remains its protected native configuration.",
                "request_timeout_secs":manifest.server.request_timeout_secs});
            report["replay_evidence_note"] = json!("Public Orchestral Run/Session journal and exact native tool exchanges are bound to original response and following request bytes. No Pi events or private checkpoints are interpreted; unproven compaction remains incomplete. Delivery is distinct from independent semantic acceptance.");
        }
    }
    write_json(report_dir.join("report.json"), &report)?;
    println!(
        "{}",
        serde_json::to_string(
            &json!({"report":report_dir.join("report.json"),"tasks_completed":tasks_completed,"accepted":accepted})
        )?
    );
    Ok(if accepted { 0 } else { 1 })
}

async fn validate(
    task: &Task,
    agent_dir: &Path,
    output: &Path,
    structured: bool,
) -> Result<Outcome> {
    let mut args = validation_args(task);
    let evidence_path = output.join("evidence.json");
    if structured {
        args.extend([
            "--result-json".into(),
            evidence_path.to_string_lossy().into_owned(),
        ]);
    }
    let outcome = process::logged(
        &task.validation.program,
        &args,
        &task.validation.cwd,
        agent_dir,
        output,
        task.validation.timeout_secs,
    )
    .await?;
    if structured && outcome.exit_code == Some(1) && !outcome.timed_out {
        let evidence: super::validator::Evidence =
            serde_json::from_slice(&fs::read(evidence_path).context(
                "repair mode requires a validator with structured --result-json support",
            )?)?;
        ensure!(
            evidence.schema_version == 1
                && evidence.class == super::validator::Class::SemanticFailure,
            "initial validator did not provide a known semantic failure"
        );
    }
    Ok(outcome)
}

pub(crate) fn validation_args(task: &Task) -> Vec<String> {
    task.validation
        .args
        .iter()
        .map(|arg| {
            if arg == "{workdir}" {
                task.workdir.to_string_lossy().into_owned()
            } else {
                arg.clone()
            }
        })
        .collect()
}

/// Schema 2 uses the same absolute task budget for both products, including validation.
pub(crate) async fn validate_within(
    task: &Task,
    agent_dir: &Path,
    output: &Path,
    remaining: Duration,
) -> Result<Outcome> {
    if remaining.is_zero() {
        let outcome = Outcome {
            exit_code: None,
            timed_out: true,
            elapsed_ms: 0,
        };
        fs::create_dir_all(output)?;
        write_json(output.join("result.json"), &outcome)?;
        write_json(
            output.join("not-started.json"),
            &json!({"reason":"task deadline exhausted; no validator process started"}),
        )?;
        return Ok(outcome);
    }
    process::logged_for(
        &task.validation.program,
        &validation_args(task),
        &task.validation.cwd,
        agent_dir,
        output,
        remaining.min(Duration::from_secs(task.validation.timeout_secs)),
    )
    .await
}

async fn run_task(
    manifest: &Manifest,
    task: &Task,
    output: PathBuf,
    agent_dir: PathBuf,
    before: BTreeMap<PathBuf, String>,
    clock: Instant,
    barrier: Option<Arc<Barrier>>,
    validation_repairs: u32,
    proxy: Arc<proxy::ProxyState>,
    protected: &[PathBuf],
    frozen: &BTreeMap<PathBuf, String>,
    orchestral: Option<orchestral::Prepared>,
) -> Result<TaskResult> {
    if let Some(client) = orchestral {
        return orchestral::run_task(
            manifest, task, output, client, before, clock, barrier, protected, frozen,
        )
        .await;
    }
    if validation_repairs > 0 {
        return repair::run_task(
            manifest,
            task,
            output,
            agent_dir,
            before,
            clock,
            barrier,
            validation_repairs,
            proxy,
            protected,
            frozen,
        )
        .await;
    }
    if let Some(barrier) = barrier {
        barrier.wait().await;
    }
    let pi = manifest.pi_program()?;
    let mut args = pi.args.clone();
    args.extend(
        [
            "--mode",
            "json",
            "--provider",
            "ferrum-local",
            "--model",
            &manifest.server.model,
            "--thinking",
            &manifest.server.thinking,
            "--session-dir",
            output.join("sessions").to_str().context("session path")?,
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--print",
            "--",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    args.push(fs::read_to_string(&task.prompt_file)?);
    write_json(
        output.join("command.json"),
        &json!({"program":pi.program,"args":args,"cwd":task.workdir}),
    )?;
    let mut command = Command::new(&pi.program);
    process::isolated(&mut command, &agent_dir);
    command
        .args(&args)
        .current_dir(&task.workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(File::create(output.join("pi-stderr.txt"))?);
    let started_ns = clock.elapsed().as_nanos() as u64;
    let start = Instant::now();
    let mut process = ManagedChild::spawn(&mut command)?;
    let pid = process.pid;
    let stdout = process.child.stdout.take().context("pi stdout")?;
    let journal_path = output.join("pi-events.jsonl");
    let raw_path = output.join("pi-stdout.jsonl");
    let reader = tokio::spawn(async move {
        let mut events = Events::default();
        let mut lines = BufReader::new(stdout).lines();
        let mut raw = File::create(raw_path)?;
        let mut journal = File::create(journal_path)?;
        while let Some(line) = lines.next_line().await? {
            writeln!(raw, "{line}")?;
            let at = clock.elapsed().as_nanos() as u64;
            match serde_json::from_str::<Value>(&line) {
                Ok(event) => {
                    events.observe(&event, at);
                    writeln!(journal, "{}", json!({"elapsed_ns":at,"event":event}))?;
                }
                Err(error) => events
                    .protocol_errors
                    .push(format!("invalid pi JSON: {error}")),
            }
        }
        Ok::<_, anyhow::Error>(events)
    });
    let (exit_code, timed_out) =
        match tokio::time::timeout(Duration::from_secs(task.timeout_secs), process.child.wait())
            .await
        {
            Ok(status) => (status?.code(), false),
            Err(_) => {
                process.stop().await;
                (None, true)
            }
        };
    if manifest.schema_version == 2 && !timed_out {
        process.stop().await;
    }
    let outcome = Outcome {
        exit_code,
        timed_out,
        elapsed_ms: start.elapsed().as_millis(),
    };
    let events = tokio::time::timeout(Duration::from_secs(10), reader)
        .await
        .context("pi output did not close")???;
    let finished_ns = clock.elapsed().as_nanos() as u64;
    let validation_source = if manifest.schema_version == 2 {
        Some(source_snapshot(
            &task.workdir,
            &output.join("validation-source"),
        )?)
    } else {
        None
    };
    let validation = if manifest.schema_version == 1 {
        validate(task, &agent_dir, &output.join("final-validation"), false).await?
    } else {
        validate_within(
            task,
            &agent_dir,
            &output.join("final-validation"),
            Duration::from_secs(task.timeout_secs).saturating_sub(start.elapsed()),
        )
        .await?
    };
    let validation_within_deadline =
        manifest.schema_version == 1 || start.elapsed() <= Duration::from_secs(task.timeout_secs);
    let after = source_snapshot(&task.workdir, &output.join("source-after"))?;
    let source_changed = before != after;
    let completed = !timed_out
        && exit_code == Some(0)
        && events.completed_loop(&manifest.server.model)
        && !validation.timed_out
        && validation.exit_code == Some(0)
        && validation_within_deadline
        && validation_source
            .as_ref()
            .is_none_or(|snapshot| snapshot == &after)
        && source_changed;
    let result = TaskResult {
        id: task.id.clone(),
        pid,
        workdir: task.workdir.clone(),
        started_ns,
        finished_ns,
        process: outcome,
        events: Some(events),
        validation: Some(validation),
        source_changed,
        closed_loop: false,
        replay: Some(replay::Evidence::default()),
        orchestral: None,
        orchestral_transport: None,
        completed,
        validation_repair: None,
    };
    write_json(output.join("result.json"), &result)?;
    Ok(result)
}

/// Preserve actual candidate edits, excluding build and version-control output.
pub(crate) fn source_snapshot(root: &Path, output: &Path) -> Result<BTreeMap<PathBuf, String>> {
    fn visit(
        root: &Path,
        relative: &Path,
        output: &Path,
        result: &mut BTreeMap<PathBuf, String>,
    ) -> Result<()> {
        for entry in fs::read_dir(root.join(relative))? {
            let entry = entry?;
            if entry.file_name() == "target" || entry.file_name() == ".git" {
                continue;
            }
            let path = relative.join(entry.file_name());
            let meta = entry.file_type()?;
            ensure!(
                !meta.is_symlink(),
                "candidate contains symlink {}",
                path.display()
            );
            if meta.is_dir() {
                visit(root, &path, output, result)?;
            } else if meta.is_file() {
                result.insert(path.clone(), config::hash_file(&root.join(&path))?);
                fs::create_dir_all(output.join(&path).parent().context("snapshot parent")?)?;
                fs::copy(root.join(&path), output.join(&path))?;
            }
        }
        Ok(())
    }
    fs::create_dir_all(output)?;
    let mut result = BTreeMap::new();
    visit(root, Path::new(""), output, &mut result)?;
    write_json(output.with_extension("hashes.json"), &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn expired_task_budget_does_not_start_a_validator_or_invent_its_exit_code() {
        let directory = tempfile::tempdir().unwrap();
        let task = Task {
            id: "expired".into(),
            workdir: directory.path().into(),
            prompt_file: directory.path().join("prompt"),
            timeout_secs: 1,
            validation: config::Validation {
                // This path cannot execute: an attempted spawn would return an error.
                program: directory.path().join("does-not-exist"),
                args: Vec::new(),
                cwd: directory.path().into(),
                timeout_secs: 120,
                expected_initial_exit_code: 1,
                protected_paths: Vec::new(),
            },
        };
        let output = directory.path().join("validation");
        let result = validate_within(&task, directory.path(), &output, Duration::ZERO)
            .await
            .unwrap();
        assert!(result.timed_out);
        assert_eq!(result.exit_code, None);
        assert_eq!(result.elapsed_ms, 0);
        assert!(output.join("not-started.json").is_file());
        assert!(!output.join("command.json").exists());
    }
}
