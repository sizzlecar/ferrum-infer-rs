//! Optional external validation feedback; every round shares one task deadline.
use super::{
    config::{self, Manifest, Task},
    process::{self, Outcome},
    proxy::{ProxyState, RequestRecord},
    replay,
    rpc::{self, Agent},
    runner::{self, TaskResult},
    validator::{Class, Evidence},
    write_json,
};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Barrier;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Termination {
    Running,
    Passed,
    RepairsExhausted,
    TaskDeadline,
    AgentFailure,
    InfrastructureFailure,
}

#[derive(Serialize)]
pub(crate) struct Round {
    index: u32,
    started_ns: u64,
    settled_ns: Option<u64>,
    finished_ns: Option<u64>,
    request_start: u32,
    request_end: u32,
    request_indices: Vec<u32>,
    idle_state: Option<Value>,
    source_hashes: BTreeMap<PathBuf, String>,
    protected_unchanged: bool,
    validation: Option<Outcome>,
    validation_evidence: Option<Evidence>,
    feedback_file: Option<PathBuf>,
    feedback_sha256: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct Report {
    maximum_repairs: u32,
    task_deadline_ns: u64,
    pub termination: Termination,
    pub error: Option<String>,
    rounds: Vec<Round>,
}
impl Report {
    fn bind_final_source(&mut self, source: &BTreeMap<PathBuf, String>) {
        if self.termination == Termination::Passed
            && !self.rounds.last().is_some_and(|round| {
                round.source_hashes == *source
                    && round.protected_unchanged
                    && round
                        .validation_evidence
                        .as_ref()
                        .is_some_and(|evidence| evidence.class == Class::Passed)
            })
        {
            self.termination = Termination::InfrastructureFailure;
            self.error = Some(
                "Final candidate differs from the independently validated source after Pi shutdown"
                    .into(),
            );
        }
    }

    pub(crate) fn session_file(&self) -> Option<&Path> {
        self.rounds.iter().rev().find_map(|round| {
            round.idle_state.as_ref()?["sessionFile"]
                .as_str()
                .filter(|path| !path.is_empty())
                .map(Path::new)
        })
    }

    pub(crate) fn bind_requests(&mut self, requests: &[&RequestRecord]) {
        for round in &mut self.rounds {
            round.request_indices = requests
                .iter()
                .filter(|r| {
                    r.request_index >= round.request_start && r.request_index < round.request_end
                })
                .map(|r| r.request_index)
                .collect();
            round.request_indices.sort_unstable();
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_task(
    manifest: &Manifest,
    task: &Task,
    output: PathBuf,
    agent_dir: PathBuf,
    before: BTreeMap<PathBuf, String>,
    clock: Instant,
    barrier: Option<Arc<Barrier>>,
    maximum_repairs: u32,
    proxy: Arc<ProxyState>,
    protected: &[PathBuf],
    frozen: &BTreeMap<PathBuf, String>,
) -> Result<TaskResult> {
    if let Some(barrier) = barrier {
        barrier.wait().await;
    }
    let started = Instant::now();
    let deadline = started
        .checked_add(Duration::from_secs(task.timeout_secs))
        .context("task deadline overflow")?;
    let started_ns = clock.elapsed().as_nanos() as u64;
    let mut report = Report {
        maximum_repairs,
        task_deadline_ns: started_ns
            .saturating_add(task.timeout_secs.saturating_mul(1_000_000_000)),
        termination: Termination::Running,
        error: None,
        rounds: Vec::new(),
    };
    let mut agent = Agent::spawn(manifest, task, &output, &agent_dir, clock, started)?;
    let pid = agent.pid();
    let result = run_rounds(
        &mut agent,
        manifest,
        task,
        &output,
        &agent_dir,
        clock,
        deadline,
        &proxy,
        protected,
        frozen,
        &mut report,
    )
    .await;
    if let Err(error) = result {
        report.termination = if Instant::now() >= deadline {
            Termination::TaskDeadline
        } else if agent.unsuccessful_stop_reason().is_some() {
            // A settled length/error/abort result is recorded as an agent
            // failure, not an independently diagnosed candidate defect or a
            // made-up validation infrastructure failure.
            Termination::AgentFailure
        } else {
            Termination::InfrastructureFailure
        };
        report.error = Some(format!("{error:#}"));
    }
    if let Some(round) = report.rounds.last_mut() {
        round.request_end = proxy.request_cursor();
        round
            .finished_ns
            .get_or_insert(clock.elapsed().as_nanos() as u64);
    }
    write_json(output.join("validation-repair.json"), &report)?;
    let abort = matches!(
        report.termination,
        Termination::TaskDeadline | Termination::InfrastructureFailure
    );
    let finished = agent.finish(deadline, abort).await?;
    let process = finished.process;
    let events = finished.events;
    if process.timed_out {
        report.termination = Termination::TaskDeadline;
        report
            .error
            .get_or_insert_with(|| "Pi task/shutdown exceeded the total task deadline".into());
    } else if finished.shutdown_timed_out {
        report.termination = Termination::InfrastructureFailure;
        report
            .error
            .get_or_insert_with(|| "Pi RPC did not finish its idle shutdown".into());
    }
    let finished_ns = clock.elapsed().as_nanos() as u64;
    let after = runner::source_snapshot(&task.workdir, &output.join("source-after"))?;
    report.bind_final_source(&after);
    let validation = report
        .rounds
        .last()
        .and_then(|round| round.validation.clone());
    let source_changed = before != after;
    let completed = report.termination == Termination::Passed
        && !process.timed_out
        && process.exit_code == Some(0)
        && events.completed_loop(&manifest.server.model)
        && source_changed;
    write_json(output.join("validation-repair.json"), &report)?;
    let result = TaskResult {
        id: task.id.clone(),
        pid,
        workdir: task.workdir.clone(),
        started_ns,
        finished_ns,
        process,
        events,
        validation,
        source_changed,
        closed_loop: false,
        replay: replay::Evidence::default(),
        completed,
        validation_repair: Some(report),
    };
    write_json(output.join("result.json"), &result)?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
async fn run_rounds(
    agent: &mut Agent,
    manifest: &Manifest,
    task: &Task,
    output: &Path,
    agent_dir: &Path,
    clock: Instant,
    deadline: Instant,
    proxy: &ProxyState,
    protected: &[PathBuf],
    frozen: &BTreeMap<PathBuf, String>,
    report: &mut Report,
) -> Result<()> {
    let mut message = fs::read_to_string(&task.prompt_file)?;
    for index in 0..=report.maximum_repairs {
        rpc::remaining(deadline)?;
        ensure!(
            &config::snapshot(protected)? == frozen,
            "protected input changed before repair round"
        );
        let round_dir = output.join("validation-rounds").join(index.to_string());
        fs::create_dir_all(&round_dir)?;
        report.rounds.push(Round {
            index,
            started_ns: clock.elapsed().as_nanos() as u64,
            settled_ns: None,
            finished_ns: None,
            request_start: proxy.request_cursor(),
            request_end: proxy.request_cursor(),
            request_indices: Vec::new(),
            idle_state: None,
            source_hashes: BTreeMap::new(),
            protected_unchanged: false,
            validation: None,
            validation_evidence: None,
            feedback_file: None,
            feedback_sha256: None,
        });
        write_json(output.join("validation-repair.json"), report)?;
        let state = agent.prompt(&message, manifest, deadline).await?;
        let round = report.rounds.last_mut().expect("current validation round");
        round.settled_ns = Some(clock.elapsed().as_nanos() as u64);
        round.request_end = proxy.request_cursor();
        round.idle_state = Some(state);
        ensure!(
            &config::snapshot(protected)? == frozen,
            "protected input changed before validation"
        );
        round.source_hashes = runner::source_snapshot(&task.workdir, &round_dir.join("source"))?;
        let validation_dir = round_dir.join("validation");
        fs::create_dir_all(&validation_dir)?;
        let evidence_path = validation_dir.join("evidence.json");
        ensure!(
            !evidence_path.exists(),
            "validator result must not pre-exist independent validation"
        );
        let mut args = runner::validation_args(task);
        args.extend([
            "--result-json".into(),
            evidence_path.to_string_lossy().into_owned(),
        ]);
        let remaining =
            rpc::remaining(deadline)?.min(Duration::from_secs(task.validation.timeout_secs));
        let validation = process::logged_for(
            &task.validation.program,
            &args,
            &task.validation.cwd,
            agent_dir,
            &validation_dir,
            remaining,
        )
        .await?;
        round.validation = Some(validation.clone());
        let after =
            runner::source_snapshot(&task.workdir, &round_dir.join("source-after-validation"))?;
        ensure!(
            round.source_hashes == after,
            "candidate changed while Pi was idle and independent validation ran"
        );
        let protected_after = config::snapshot(protected)?;
        write_json(round_dir.join("protected-after.json"), &protected_after)?;
        round.protected_unchanged = &protected_after == frozen;
        ensure!(
            round.protected_unchanged,
            "protected input changed during validation"
        );
        ensure!(
            !validation.timed_out,
            "independent validator timed out; no semantic result is known"
        );
        let evidence: Evidence = serde_json::from_slice(&fs::read(&evidence_path)
            .context("validator supplied no typed result (old/unsupported validator or infrastructure failure)")?)?;
        ensure!(
            evidence.schema_version == 1 && validation.exit_code == Some(evidence.exit_code()),
            "validator result/exit status disagree"
        );
        round.validation_evidence = Some(evidence.clone());
        // Idle is rechecked before any feedback is allowed; no queued agent work
        // can edit the candidate concurrently with independent validation.
        agent.idle(manifest, deadline).await?;
        round.finished_ns = Some(clock.elapsed().as_nanos() as u64);
        if evidence.class == Class::Passed {
            report.termination = Termination::Passed;
            return Ok(());
        }
        ensure!(
            evidence.repairable(),
            "independent validator reported {:?}; no candidate semantic diagnosis is assumed",
            evidence.class
        );
        if index == report.maximum_repairs {
            report.termination = Termination::RepairsExhausted;
            return Ok(());
        }
        message = feedback(&evidence, &validation_dir)?;
        let feedback_path = round_dir.join("feedback.txt");
        fs::write(&feedback_path, &message)?;
        round.feedback_sha256 = Some(config::hash_file(&feedback_path)?);
        round.feedback_file = Some(feedback_path);
        write_json(output.join("validation-repair.json"), report)?;
    }
    unreachable!("inclusive repair budget always terminates on a validation result")
}

fn feedback(evidence: &Evidence, output: &Path) -> Result<String> {
    ensure!(
        evidence.repairable(),
        "only a known candidate failure can be fed back"
    );
    let diagnosis = if evidence.class == Class::CandidateCompilationFailure {
        ensure!(
            !evidence.diagnostics.is_empty(),
            "candidate compilation failure lacks compiler diagnostics"
        );
        evidence.diagnostics.join("\n")
    } else {
        format!(
            "stdout:\n{}\nstderr:\n{}",
            fs::read_to_string(output.join("stdout.txt"))?,
            fs::read_to_string(output.join("stderr.txt"))?
        )
    };
    Ok(format!("Independent validation of your original task reported {:?}. Continue fixing the same task in the same working directory. The original requirements and total time budget are unchanged. Do not change the validator, protected inputs, or acceptance contract. These are the actual validation diagnostics:\n\n{diagnosis}", evidence.class))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounds_bind_actual_request_indices_in_their_interval() {
        let mut report = Report {
            maximum_repairs: 1,
            task_deadline_ns: 10,
            termination: Termination::Running,
            error: None,
            rounds: vec![Round {
                index: 0,
                started_ns: 0,
                settled_ns: None,
                finished_ns: None,
                request_start: 2,
                request_end: 7,
                request_indices: vec![],
                idle_state: None,
                source_hashes: BTreeMap::new(),
                protected_unchanged: false,
                validation: None,
                validation_evidence: None,
                feedback_file: None,
                feedback_sha256: None,
            }],
        };
        let records = [
            RequestRecord {
                request_index: 1,
                ..Default::default()
            },
            RequestRecord {
                request_index: 3,
                ..Default::default()
            },
            RequestRecord {
                request_index: 7,
                ..Default::default()
            },
        ];
        report.bind_requests(&records.iter().collect::<Vec<_>>());
        assert_eq!(report.rounds[0].request_indices, [3]);
    }

    #[test]
    fn passing_validation_cannot_certify_source_changed_during_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let workdir = dir.path().join("candidate");
        fs::create_dir_all(workdir.join("src")).unwrap();
        let source = workdir.join("src/lib.rs");
        fs::write(&source, "pub fn value() -> u32 { 2 }\n").unwrap();
        let validated = runner::source_snapshot(&workdir, &dir.path().join("validated")).unwrap();
        let mut report = Report {
            maximum_repairs: 1,
            task_deadline_ns: 10,
            termination: Termination::Passed,
            error: None,
            rounds: vec![Round {
                index: 0,
                started_ns: 0,
                settled_ns: Some(1),
                finished_ns: Some(2),
                request_start: 0,
                request_end: 1,
                request_indices: vec![0],
                idle_state: None,
                source_hashes: validated.clone(),
                protected_unchanged: true,
                validation: Some(Outcome {
                    exit_code: Some(0),
                    timed_out: false,
                    elapsed_ms: 1,
                }),
                validation_evidence: Some(Evidence {
                    schema_version: 1,
                    class: Class::Passed,
                    diagnostics: vec![],
                }),
                feedback_file: None,
                feedback_sha256: None,
            }],
        };
        report.bind_final_source(&validated);
        assert_eq!(report.termination, Termination::Passed);

        fs::write(source, "pub fn value() -> u32 { 3 }\n").unwrap();
        let final_source = runner::source_snapshot(&workdir, &dir.path().join("final")).unwrap();
        report.bind_final_source(&final_source);
        assert_eq!(report.termination, Termination::InfrastructureFailure);
        assert!(report.error.is_some());

        report.termination = Termination::TaskDeadline;
        report.bind_final_source(&validated);
        assert_eq!(report.termination, Termination::TaskDeadline);
    }
    #[test]
    fn timeout_or_unknown_exit_two_cannot_produce_repair_feedback() {
        let dir = tempfile::tempdir().unwrap();
        for class in [Class::InfrastructureFailure, Class::TimedOut] {
            let evidence = Evidence {
                schema_version: 1,
                class,
                diagnostics: vec!["not a candidate error".into()],
            };
            assert!(feedback(&evidence, dir.path()).is_err());
        }
    }
    #[test]
    fn semantic_feedback_preserves_real_diagnostics_and_compiler_feedback_requires_evidence() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("stdout.txt"),
            "assertion `left == right` failed\n  left: 2\n right: 3\n",
        )
        .unwrap();
        fs::write(
            dir.path().join("stderr.txt"),
            "diagnostic with \\ and \"quotes\"\n",
        )
        .unwrap();
        let evidence = Evidence {
            schema_version: 1,
            class: Class::SemanticFailure,
            diagnostics: vec![],
        };
        let result = feedback(&evidence, dir.path()).unwrap();
        assert!(result.contains(&fs::read_to_string(dir.path().join("stdout.txt")).unwrap()));
        assert!(result.contains(&fs::read_to_string(dir.path().join("stderr.txt")).unwrap()));
        let absent = Evidence {
            schema_version: 1,
            class: Class::CandidateCompilationFailure,
            diagnostics: vec![],
        };
        assert!(feedback(&absent, dir.path()).is_err());
    }
}
