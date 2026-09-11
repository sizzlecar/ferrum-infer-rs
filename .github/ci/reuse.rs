//! Reuse individual successful PR checks only when the complete source diff
//! leaves their inputs unaffected. Release/manual workflows always run fresh.
use super::checks::{affected, Check, Checks, Plan};
use std::{collections::BTreeMap, env, fs, path::PathBuf, process::Command};

fn command(program: &str, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("{program}: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} failed while reading prior check evidence"
        ));
    }
    Ok(output.stdout)
}
fn api(path: &str, query: &str) -> Result<String, String> {
    String::from_utf8(command("gh", &["api", path, "--paginate", "--jq", query])?)
        .map_err(|_| "non-UTF-8 Actions metadata".into())
}
fn id(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit())
}
fn sha(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

fn origin(repo: &str, run_id: &str) -> Result<String, String> {
    let ids = api(
        &format!("repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"),
        ".artifacts[] | select(.name == \"ci-plan\" and .expired == false) | .id",
    )?;
    let ids: Vec<_> = ids.lines().collect();
    if ids.len() != 1 || !id(ids[0]) {
        return Err("missing or ambiguous prior CI origin".into());
    }
    let archive = command(
        "gh",
        &[
            "api",
            &format!("repos/{repo}/actions/artifacts/{}/zip", ids[0]),
        ],
    )?;
    let path = PathBuf::from(env::var("RUNNER_TEMP").map_err(|_| "RUNNER_TEMP missing")?)
        .join(format!("ci-origin-{}.zip", ids[0]));
    fs::write(&path, archive).map_err(|e| e.to_string())?;
    let bytes = command(
        "unzip",
        &[
            "-p",
            path.to_str().ok_or("invalid temporary path")?,
            "origin.txt",
        ],
    );
    let _ = fs::remove_file(path);
    parse_origin(
        &String::from_utf8(bytes?).map_err(|_| "invalid prior origin encoding")?,
        run_id,
    )
}

fn parse_origin(input: &str, run_id: &str) -> Result<String, String> {
    let mut revision = None;
    let mut origin_run = None;
    for line in input.lines() {
        match line.split_once('=') {
            Some(("revision", value)) if revision.is_none() && sha(value) => revision = Some(value),
            Some(("run_id", value)) if origin_run.is_none() && id(value) => {
                origin_run = Some(value)
            }
            _ => return Err("invalid prior origin fields".into()),
        }
    }
    if origin_run != Some(run_id) {
        return Err("origin belongs to another Actions run".into());
    }
    revision
        .map(str::to_owned)
        .ok_or_else(|| "missing checked revision".into())
}

fn changes_since(revision: &str) -> Result<Checks, String> {
    if command(
        "git",
        &["cat-file", "-e", &format!("{revision}^{{commit}}")],
    )
    .is_err()
    {
        command(
            "git",
            &["fetch", "--no-tags", "--depth=1", "origin", revision],
        )?;
    }
    let diff = command(
        "git",
        &[
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            revision,
            "HEAD",
            "--",
        ],
    )?;
    // Here empty is an observed successful Git comparison of two revisions,
    // unlike missing/truncated stdin to the public path classifier.
    Ok(if diff.is_empty() {
        Checks::default()
    } else {
        affected(&diff)
    })
}

#[derive(Debug, PartialEq, Eq)]
enum Decision {
    Reuse,
    Run,
    Older,
}

#[derive(Debug)]
struct Execution {
    run_id: String,
    job_id: u64,
    status: String,
    conclusion: String,
    completed_at: String,
}

fn latest(executions: &[Execution]) -> Result<Option<&Execution>, String> {
    let mut result: Option<&Execution> = None;
    for execution in executions {
        if execution.status == "completed" && execution.conclusion == "skipped" {
            continue;
        }
        if execution.status != "completed" {
            return Err("another execution is pending".into());
        }
        let time = execution.completed_at.as_bytes();
        if time.len() != 20
            || time[4] != b'-'
            || time[7] != b'-'
            || time[10] != b'T'
            || time[13] != b':'
            || time[16] != b':'
            || time[19] != b'Z'
            || time.iter().enumerate().any(|(index, byte)| {
                ![4, 7, 10, 13, 16, 19].contains(&index) && !byte.is_ascii_digit()
            })
        {
            return Err("unknown execution timestamp".into());
        }
        // Rerunning Windows must not make an older GPU success newer than
        // another GPU failure. Order actual check verdicts, not workflow runs.
        if result.is_none_or(|old| {
            (&execution.completed_at, execution.job_id) > (&old.completed_at, old.job_id)
        }) {
            result = Some(execution);
        }
    }
    Ok(result)
}
fn decide(status: &str, conclusion: &str, changed: Option<bool>) -> Decision {
    match (status, conclusion) {
        ("completed", "skipped") => Decision::Older,
        ("completed", "success") if changed == Some(false) => Decision::Reuse,
        _ => Decision::Run,
    }
}

fn collect_jobs(
    repo: &str,
    run_id: &str,
    endpoint: &str,
    required: Checks,
    executions: &mut BTreeMap<u8, Vec<Execution>>,
) -> Result<(), String> {
    let jobs = api(
        &format!("repos/{repo}/actions/runs/{run_id}/{endpoint}"),
        ".jobs[] | [.name, .status, (.conclusion // \"\"), .id, (.completed_at // \"\"), .run_attempt] | @tsv",
    )?;
    record_jobs(run_id, &jobs, required, executions)
}

fn record_jobs(
    run_id: &str,
    jobs: &str,
    required: Checks,
    executions: &mut BTreeMap<u8, Vec<Execution>>,
) -> Result<(), String> {
    for check in Check::ALL {
        if !required.has(check) {
            continue;
        }
        let matches: Vec<_> = jobs
            .lines()
            .map(|row| row.split('\t').collect::<Vec<_>>())
            .filter(|row| row.first().copied() == Some(check.name()))
            .collect();
        if matches.is_empty() {
            continue;
        }
        let mut attempts = std::collections::BTreeSet::new();
        for row in matches {
            if row.len() != 6 || !id(row[3]) || !id(row[5]) || !attempts.insert(row[5]) {
                return Err("incomplete or ambiguous prior check execution".into());
            }
            executions.entry(check.bit()).or_default().push(Execution {
                run_id: run_id.to_owned(),
                job_id: row[3].parse().map_err(|_| "invalid prior job ID")?,
                status: row[1].to_owned(),
                conclusion: row[2].to_owned(),
                completed_at: row[4].to_owned(),
            });
        }
    }
    Ok(())
}

pub fn reuse(plan: &mut Plan, notes: &mut Vec<String>) -> Result<(), String> {
    let repo = env::var("GITHUB_REPOSITORY").map_err(|_| "repository missing")?;
    let branch = env::var("PR_BRANCH").map_err(|_| "PR branch missing")?;
    let pr = env::var("PR_NUMBER").map_err(|_| "PR number missing")?;
    let current_run = env::var("GITHUB_RUN_ID").map_err(|_| "run ID missing")?;
    let current_attempt = env::var("GITHUB_RUN_ATTEMPT")
        .map_err(|_| "run attempt missing")?
        .parse::<u64>()
        .map_err(|_| "invalid run attempt")?;
    if current_attempt == 0 {
        return Err("invalid run attempt".into());
    }
    if !id(&pr) || !id(&current_run) || repo.split('/').count() != 2 {
        return Err("invalid PR context".into());
    }
    if env::var("PR_REPOSITORY").as_deref() != Ok(repo.as_str()) {
        return Ok(());
    }
    // Do not use a prior pull request with a recycled branch name. No secrets
    // or untrusted source code are read from archived artifacts.
    let runs = String::from_utf8(command("gh", &[
        "api", &format!("repos/{repo}/actions/workflows/ci.yml/runs"), "--method", "GET", "--paginate",
        "-f", "event=pull_request", "-f", &format!("branch={branch}"), "-F", "per_page=100", "--jq",
        ".workflow_runs[] | [.id, .head_repository.full_name, .head_branch, ([.pull_requests[].number | tostring] | join(\",\"))] | @tsv",
    ])?).map_err(|_| "invalid prior run metadata")?;
    let mut executions: BTreeMap<u8, Vec<Execution>> = BTreeMap::new();
    for row in runs.lines() {
        let fields: Vec<_> = row.split('\t').collect();
        if fields.len() != 4 || !id(fields[0]) {
            return Err("incomplete prior run metadata".into());
        }
        if fields[1] != repo || fields[2] != branch {
            continue;
        }
        if fields[3].is_empty() {
            return Err("same-branch run has no verifiable PR association".into());
        }
        if !fields[3].split(',').any(|number| number == pr) {
            continue;
        }
        let run_id = fields[0];
        if run_id == current_run {
            continue;
        }
        collect_jobs(
            &repo,
            run_id,
            "jobs?filter=all&per_page=100",
            plan.required,
            &mut executions,
        )?;
    }
    // Rerunning prepare must not erase failures from earlier attempts of this
    // same run. Only the current attempt is excluded while its plan executes.
    for attempt in 1..current_attempt {
        collect_jobs(
            &repo,
            &current_run,
            &format!("attempts/{attempt}/jobs?per_page=100"),
            plan.required,
            &mut executions,
        )?;
    }
    let mut origins = BTreeMap::new();
    for check in Check::ALL {
        let Some(executions) = executions.get(&check.bit()) else {
            continue;
        };
        match latest(executions) {
            Ok(Some(execution)) => {
                let run_id = &execution.run_id;
                let changed = origins.entry(run_id.clone()).or_insert_with(|| {
                    origin(&repo, run_id)
                        .and_then(|revision| changes_since(&revision))
                        .ok()
                });
                match decide(&execution.status, &execution.conclusion, changed.map(|paths| paths.has(check))) {
                    Decision::Older => unreachable!("latest excludes skipped jobs"),
                    Decision::Run => notes.push(format!("{}: run; latest executed check is {} or its unchanged inputs cannot be established.", check.name(), execution.conclusion)),
                    Decision::Reuse => {
                        plan.reused.0 |= check.bit();
                        notes.push(format!("{}: reuse successful [job](https://github.com/{repo}/actions/runs/{run_id}/job/{}); its input scope is unchanged.", check.name(), execution.job_id));
                    }
                }
            }
            Ok(None) => (),
            Err(reason) => notes.push(format!("{}: run; {reason}.", check.name())),
        }
    }
    Ok(())
}

pub fn write_origin() -> Result<(), String> {
    let run_id = env::var("GITHUB_RUN_ID").map_err(|_| "run ID missing")?;
    let revision = String::from_utf8(command("git", &["rev-parse", "HEAD"])?)
        .map_err(|_| "invalid checkout revision")?;
    if !id(&run_id) || !sha(revision.trim()) {
        return Err("invalid current origin".into());
    }
    let directory =
        PathBuf::from(env::var("RUNNER_TEMP").map_err(|_| "RUNNER_TEMP missing")?).join("ci-plan");
    fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
    fs::write(
        directory.join("origin.txt"),
        format!("revision={}\nrun_id={run_id}\n", revision.trim()),
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actions_history_keeps_failed_attempts_and_backdated_success_copies() {
        // GitHub copies unchanged successful jobs into later attempts, with
        // new job IDs but their original completion timestamps.
        let input = concat!(
            "CUDA (Linux)\tcompleted\tsuccess\t11\t2026-09-10T09:00:00Z\t1\n",
            "Windows (MSVC contracts)\tcompleted\tfailure\t12\t2026-09-10T09:01:00Z\t1\n",
            "CUDA (Linux)\tcompleted\tsuccess\t21\t2026-09-10T09:00:00Z\t2\n",
            "Windows (MSVC contracts)\tcompleted\tsuccess\t22\t2026-09-10T10:00:00Z\t2\n",
            "GPU runtime (cuda)\tcompleted\tfailure\t13\t2026-09-10T09:02:00Z\t1\n",
            "GPU runtime (cuda)\tcompleted\tskipped\t23\t2026-09-10T10:01:00Z\t2\n",
        );
        let mut executions = BTreeMap::new();
        record_jobs("100", input, Checks::ALL, &mut executions).unwrap();
        assert_eq!(
            latest(&executions[&Check::Windows.bit()])
                .unwrap()
                .unwrap()
                .job_id,
            22
        );
        assert_eq!(
            latest(&executions[&Check::Cuda.bit()])
                .unwrap()
                .unwrap()
                .completed_at,
            "2026-09-10T09:00:00Z"
        );
        assert_eq!(
            latest(&executions[&Check::CudaRuntime.bit()])
                .unwrap()
                .unwrap()
                .conclusion,
            "failure"
        );
        for bad in [
            "CUDA (Linux)\tcompleted\tsuccess\t31\n",
            "CUDA (Linux)\tcompleted\tsuccess\t31\t2026-09-10T10:00:00Z\tnull\n",
            "CUDA (Linux)\tcompleted\tsuccess\t31\t2026-09-10T10:00:00Z\t1\nCUDA (Linux)\tcompleted\tfailure\t32\t2026-09-10T10:00:00Z\t1\n",
        ] {
            assert!(record_jobs("100", bad, Checks::ALL, &mut BTreeMap::new()).is_err());
        }
    }
    #[test]
    fn per_job_completion_order_survives_unrelated_reruns_and_skips() {
        let execution = |run: &str, job, time: &str, conclusion: &str| Execution {
            run_id: run.into(),
            job_id: job,
            status: "completed".into(),
            conclusion: conclusion.into(),
            completed_at: time.into(),
        };
        let mut jobs = vec![
            execution("300", 1, "2026-09-10T09:00:00Z", "success"),
            execution("200", 2, "2026-09-10T10:00:00Z", "failure"),
            execution("400", 3, "2026-09-10T11:00:00Z", "skipped"),
        ];
        assert_eq!(latest(&jobs).unwrap().unwrap().conclusion, "failure");
        jobs.reverse();
        assert_eq!(latest(&jobs).unwrap().unwrap().conclusion, "failure");
        jobs.push(execution("100", 4, "2026-09-10T12:00:00Z", "success"));
        assert_eq!(latest(&jobs).unwrap().unwrap().job_id, 4);
        jobs.push(Execution {
            status: "in_progress".into(),
            ..execution("500", 5, "", "")
        });
        assert!(latest(&jobs).is_err());
    }
    #[test]
    fn reuse_requires_success_and_an_observed_unchanged_scope() {
        assert_eq!(decide("completed", "success", Some(false)), Decision::Reuse);
        for changed in [None, Some(true)] {
            assert_eq!(decide("completed", "success", changed), Decision::Run);
        }
        for state in ["failure", "cancelled", "timed_out", "neutral", ""] {
            assert_eq!(decide("completed", state, Some(false)), Decision::Run);
        }
        assert_eq!(decide("in_progress", "", Some(false)), Decision::Run);
        assert_eq!(decide("completed", "skipped", Some(false)), Decision::Older);
    }
    #[test]
    fn newer_failed_execution_blocks_older_success_but_expected_skips_can_trace_origin() {
        for first in ["failure", "cancelled", "success"] {
            let result = [first, "success"]
                .into_iter()
                .map(|value| decide("completed", value, Some(false)))
                .find(|value| *value != Decision::Older)
                .unwrap();
            assert_eq!(
                result,
                if first == "success" {
                    Decision::Reuse
                } else {
                    Decision::Run
                }
            );
        }
        assert_eq!(
            ["skipped", "success"]
                .into_iter()
                .map(|value| decide("completed", value, Some(false)))
                .find(|value| *value != Decision::Older)
                .unwrap(),
            Decision::Reuse
        );
    }
    #[test]
    fn prior_origin_is_bound_to_its_run_and_full_checked_revision() {
        let revision = "a".repeat(40);
        assert_eq!(
            parse_origin(&format!("revision={revision}\nrun_id=123\n"), "123").unwrap(),
            revision
        );
        for origin in [
            format!("revision={revision}\nrun_id=124\n"),
            "revision=main\nrun_id=123\n".into(),
            format!("revision={revision}\nrevision={revision}\nrun_id=123\n"),
        ] {
            assert!(parse_origin(&origin, "123").is_err());
        }
    }
}
