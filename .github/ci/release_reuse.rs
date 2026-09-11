//! Formal releases retain full quality coverage. Each skipped check names its
//! original successful execution; publication revalidates that frozen source.
use super::{
    checks::{Check, Checks, Plan},
    reuse,
};
use std::{collections::BTreeMap, env, fs, path::Path};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub check: Check,
    pub run_id: String,
    pub job_id: u64,
    pub revision: String,
    pub completed_at: String,
    pub execution: String,
}

#[derive(Clone, Debug)]
pub struct ReleasePlan {
    pub plan: Plan,
    pub sources: Vec<Source>,
}

impl ReleasePlan {
    pub fn fresh() -> Self {
        Self {
            plan: Plan::fresh(Checks::ALL),
            sources: Vec::new(),
        }
    }
    pub fn encode(&self) -> String {
        let mut text = format!("{}\n", self.plan.encode());
        for source in &self.sources {
            text.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\n",
                source.check.bit(),
                source.run_id,
                source.job_id,
                source.revision,
                source.completed_at,
                source.execution
            ));
        }
        text
    }
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        let plan = Plan::parse(lines.next().ok_or("missing formal quality plan")?)?;
        if plan.required != Checks::ALL {
            return Err("a formal release requires complete quality coverage".into());
        }
        let mut found = Checks::default();
        let mut sources = Vec::new();
        for line in lines {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 6
                || fields[1].parse::<u64>().ok().is_none_or(|id| id == 0)
                || !reuse::sha(fields[3])
                || fields[5].is_empty()
                || !fields[5]
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"+/=".contains(&byte))
            {
                return Err("invalid frozen quality source".into());
            }
            let check = Check::ALL
                .into_iter()
                .find(|check| fields[0] == check.bit().to_string())
                .ok_or("unknown quality check")?;
            let job_id = fields[2]
                .parse::<u64>()
                .ok()
                .filter(|id| *id > 0)
                .ok_or("invalid source job ID")?;
            if found.has(check) || !plan.reused.has(check) {
                return Err("duplicate or unexpected quality source".into());
            }
            // Reuse the execution parser's timestamp validation.
            reuse::latest(&[reuse::Execution {
                run_id: fields[1].into(),
                job_id,
                status: "completed".into(),
                conclusion: "success".into(),
                completed_at: fields[4].into(),
            }])?;
            found.0 |= check.bit();
            sources.push(Source {
                check,
                run_id: fields[1].into(),
                job_id,
                revision: fields[3].into(),
                completed_at: fields[4].into(),
                execution: fields[5].into(),
            });
        }
        if found != plan.reused {
            return Err("a reused check has no frozen source".into());
        }
        Ok(Self { plan, sources })
    }
    pub fn source(&self, check: Check) -> Option<&Source> {
        self.sources.iter().find(|source| source.check == check)
    }
}

fn release_row<'a>(row: &'a str, repo: &str) -> Result<Option<(&'a str, &'a str)>, String> {
    let fields: Vec<_> = row.split('\t').collect();
    if fields.len() != 6 || !reuse::id(fields[0]) {
        return Err("incomplete release run inventory".into());
    }
    if fields[1] != repo || fields[2] != "main" {
        return Ok(None);
    }
    if !reuse::sha(fields[3])
        || !matches!(fields[4], "push" | "workflow_dispatch")
        || fields[5].split('@').next() != Some(".github/workflows/release-delivery.yml")
    {
        return Err("unverifiable main release run".into());
    }
    Ok(Some((fields[0], fields[3])))
}

fn jobs(
    repo: &str,
    run: &str,
    endpoint: &str,
    required: Checks,
    executions: &mut BTreeMap<u8, Vec<reuse::Execution>>,
) -> Result<(), String> {
    let rows = reuse::api(&format!("repos/{repo}/actions/runs/{run}/{endpoint}"), "([\"total\", .total_count] | @tsv), (.jobs[] | [\"item\", .name, .status, (.conclusion // \"\"), .id, (.completed_at // \"\"), .run_attempt] | @tsv)")?;
    let quality: String = reuse::inventory_rows(&rows, 3)?
        .iter()
        .filter_map(|row| row.strip_prefix("Quality / "))
        .map(|row| format!("{row}\n"))
        .collect();
    reuse::record_jobs(run, &quality, required, executions)
}

fn required_artifacts(check: Check) -> &'static [&'static str] {
    match check {
        Check::Cpu => &["contract-checks", "backend-numerics-cpu"],
        Check::MetalRuntime => &["backend-numerics-metal"],
        Check::CudaRuntime => &["backend-numerics-cuda"],
        _ => &[],
    }
}

fn available_artifacts(repo: &str, run: &str, check: Check) -> Result<(), String> {
    if required_artifacts(check).is_empty() {
        return Ok(());
    }
    let inventory = reuse::api(
        &format!("repos/{repo}/actions/runs/{run}/artifacts?per_page=100"),
        ".artifacts[] | [.name, .id, .expired, (.digest // \"\")] | @tsv",
    )?;
    for name in required_artifacts(check) {
        let matches: Vec<_> = inventory
            .lines()
            .map(|row| row.split('\t').collect::<Vec<_>>())
            .filter(|fields| fields.first().copied() == Some(name))
            .collect();
        if matches.len() != 1
            || matches[0].len() != 4
            || !reuse::id(matches[0][1])
            || matches[0][2] != "false"
            || !matches[0][3]
                .strip_prefix("sha256:")
                .is_some_and(|digest| digest.len() == 64 && reuse::sha(digest))
        {
            return Err(format!(
                "required source artifact {name} is unavailable or unverifiable"
            ));
        }
    }
    Ok(())
}

fn execution_identity(repo: &str, source: &Source) -> Result<String, String> {
    // Canonical arrays avoid JSON object ordering differences. IDs/attempts are
    // excluded from the copy signature, while every executed step is retained.
    let row = reuse::api(&format!("repos/{repo}/actions/jobs/{}", source.job_id),
        "[.id, .run_id, .head_sha, .name, .status, .conclusion, .completed_at, ([.started_at, .completed_at, (.steps | map([.name, .number, .status, .conclusion, .started_at, .completed_at]))] | @base64)] | @tsv")?;
    let fields: Vec<_> = row.trim_end_matches('\n').split('\t').collect();
    if fields.len() != 8
        || fields[0] != source.job_id.to_string()
        || fields[1] != source.run_id
        || fields[2] != source.revision
        || fields[3] != format!("Quality / {}", source.check.name())
        || fields[4] != "completed"
        || fields[5] != "success"
        || fields[6] != source.completed_at
    {
        return Err("frozen quality job has different identity or outcome".into());
    }
    Ok(fields[7].to_owned())
}

pub fn resolve(
    repo: &str,
    current_run: &str,
    current_attempt: u64,
    workspace: &str,
    required: Checks,
    notes: &mut Vec<String>,
) -> Result<Vec<Source>, String> {
    if !reuse::id(current_run) || current_attempt == 0 || repo.split('/').count() != 2 {
        return Err("invalid formal release context".into());
    }
    let runs = String::from_utf8(reuse::command("gh", &[
        "api", &format!("repos/{repo}/actions/workflows/release-delivery.yml/runs"), "--method", "GET", "--paginate",
        "-f", "branch=main", "-F", "per_page=100", "--jq",
        "([\"total\", .total_count] | @tsv), (.workflow_runs[] | [\"item\", .id, .head_repository.full_name, .head_branch, .head_sha, .event, .path] | @tsv)",
    ])?).map_err(|_| "invalid release run metadata")?;
    let mut revisions = BTreeMap::new();
    let mut executions = BTreeMap::new();
    for row in reuse::inventory_rows(&runs, 0)? {
        let Some((run, revision)) = release_row(&row, repo)? else {
            continue;
        };
        revisions.insert(run.to_owned(), revision.to_owned());
        if run != current_run {
            jobs(
                repo,
                run,
                "jobs?filter=all&per_page=100",
                required,
                &mut executions,
            )?;
        }
    }
    for attempt in 1..current_attempt {
        jobs(
            repo,
            current_run,
            &format!("attempts/{attempt}/jobs?per_page=100"),
            required,
            &mut executions,
        )?;
    }
    let mut changes = BTreeMap::new();
    let mut sources = Vec::new();
    for check in Check::ALL {
        let Some(history) = executions.get(&check.bit()) else {
            continue;
        };
        let execution = match reuse::latest(history) {
            Ok(Some(value)) if value.conclusion == "success" => value,
            _ => {
                notes.push(format!(
                    "{}: run; latest execution is unsuccessful, pending or unavailable.",
                    check.name()
                ));
                continue;
            }
        };
        let changed = changes.entry(execution.run_id.clone()).or_insert_with(|| {
            let revision = reuse::origin(repo, &execution.run_id).ok()?;
            if revisions.get(&execution.run_id) != Some(&revision) {
                return None;
            }
            reuse::changes_since_at(&revision, workspace)
                .ok()
                .map(|affected| (revision, affected))
        });
        let Some((revision, _)) = changed
            .as_ref()
            .filter(|(_, affected)| !affected.has(check))
        else {
            continue;
        };
        if let Err(reason) = available_artifacts(repo, &execution.run_id, check) {
            notes.push(format!("{}: run; {reason}.", check.name()));
            continue;
        }
        let mut source = Source {
            check,
            run_id: execution.run_id.clone(),
            job_id: execution.job_id,
            revision: revision.clone(),
            completed_at: execution.completed_at.clone(),
            execution: String::new(),
        };
        source.execution = match execution_identity(repo, &source) {
            Ok(identity) => identity,
            Err(reason) => {
                notes.push(format!("{}: run; {reason}.", check.name()));
                continue;
            }
        };
        notes.push(format!("{}: reuse [original execution](https://github.com/{repo}/actions/runs/{}/job/{}); its required execution scope is unchanged.", check.name(), execution.run_id, execution.job_id));
        sources.push(source);
    }
    Ok(sources)
}

fn same_execution(expected: &Source, current: &Source) -> bool {
    // GitHub may copy a successful job under another ID on an unrelated retry.
    // The raw artifact verifier separately binds its original job/attempt/steps.
    expected.check == current.check
        && expected.run_id == current.run_id
        && expected.revision == current.revision
        && expected.completed_at == current.completed_at
        && expected.execution == current.execution
}

pub fn verify(
    accepted: &ReleasePlan,
    repo: &str,
    current_run: &str,
    current_attempt: u64,
    workspace: &str,
) -> Result<(), String> {
    if accepted.plan.reused.0 == 0 {
        return Ok(());
    }
    let current = resolve(
        repo,
        current_run,
        current_attempt,
        workspace,
        accepted.plan.reused,
        &mut Vec::new(),
    )?;
    for expected in &accepted.sources {
        if !current
            .iter()
            .any(|source| same_execution(expected, source))
            || execution_identity(repo, expected)? != expected.execution
        {
            return Err(format!("{}: frozen source was superseded or no longer proves success; rerun the quality plan", expected.check.name()));
        }
    }
    Ok(())
}

pub fn from_environment(notes: &mut Vec<String>) -> Result<ReleasePlan, String> {
    let mut result = ReleasePlan::fresh();
    result.sources = resolve(
        &env::var("GITHUB_REPOSITORY").map_err(|_| "repository missing")?,
        &env::var("GITHUB_RUN_ID").map_err(|_| "run missing")?,
        env::var("GITHUB_RUN_ATTEMPT")
            .map_err(|_| "attempt missing")?
            .parse()
            .map_err(|_| "invalid attempt")?,
        ".",
        Checks::ALL,
        notes,
    )?;
    for source in &result.sources {
        result.plan.reused.0 |= source.check.bit();
    }
    Ok(result)
}

pub fn save(plan: &ReleasePlan) -> Result<(), String> {
    let root = env::var("RUNNER_TEMP").map_err(|_| "RUNNER_TEMP missing")?;
    fs::write(
        Path::new(&root).join("ci-plan/release-plan.txt"),
        plan.encode(),
    )
    .map_err(|e| e.to_string())
}

pub fn write_producer() -> Result<(), String> {
    let repo = env::var("GITHUB_REPOSITORY").map_err(|_| "repository missing")?;
    let run = env::var("GITHUB_RUN_ID").map_err(|_| "run missing")?;
    let attempt = env::var("GITHUB_RUN_ATTEMPT").map_err(|_| "attempt missing")?;
    let revision = String::from_utf8(reuse::command("git", &["rev-parse", "HEAD"])?)
        .map_err(|_| "invalid checked revision")?
        .trim()
        .to_owned();
    if !reuse::id(&run)
        || !reuse::id(&attempt)
        || !reuse::sha(&revision)
        || env::var("RELEASE_CANDIDATE").as_deref() != Ok(revision.as_str())
    {
        return Err("release plan checkout differs from its immutable candidate".into());
    }
    let jobs = reuse::api(&format!("repos/{repo}/actions/runs/{run}/attempts/{attempt}/jobs?per_page=100"), ".jobs[] | select(.name == \"Quality / Plan and validate docs\" and .status == \"in_progress\") | .id")?;
    let ids: Vec<_> = jobs.lines().collect();
    if ids.len() != 1 || !reuse::id(ids[0]) {
        return Err("release plan producer is ambiguous".into());
    }
    let root = env::var("RUNNER_TEMP").map_err(|_| "RUNNER_TEMP missing")?;
    fs::write(Path::new(&root).join("ci-plan/ci-origin.json"), format!("{{\"schema_version\":1,\"run_id\":{run},\"run_attempt\":{attempt},\"head_sha\":\"{revision}\",\"job_id\":{}}}", ids[0])).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> Source {
        Source {
            check: Check::CudaRuntime,
            run_id: "12".into(),
            job_id: 91,
            revision: "a".repeat(40),
            completed_at: "2026-09-11T01:00:00Z".into(),
            execution: "W10=".into(),
        }
    }
    #[test]
    fn frozen_plan_requires_exact_complete_coverage_and_one_origin_per_skip() {
        let mut plan = ReleasePlan::fresh();
        plan.plan.reused.0 = Check::CudaRuntime.bit();
        plan.sources.push(source());
        assert_eq!(
            ReleasePlan::parse(&plan.encode()).unwrap().sources,
            plan.sources
        );
        for text in [
            "v1:0:0\n".into(),
            "v1:63:16\n".into(),
            format!("{}{}", plan.encode(), plan.encode().lines().nth(1).unwrap()),
            plan.encode().replace("v1:63:16", "v1:63:0"),
            plan.encode().replace("01:00:00Z", "invalid"),
        ] {
            assert!(ReleasePlan::parse(&text).is_err(), "{text}");
        }
    }
    #[test]
    fn only_same_repository_main_release_history_can_supply_a_source() {
        let row = format!(
            "12\towner/repo\tmain\t{}\tworkflow_dispatch\t.github/workflows/release-delivery.yml",
            "a".repeat(40)
        );
        assert!(release_row(&row, "owner/repo").unwrap().is_some());
        assert!(
            release_row(&row.replace("owner/repo", "fork/repo"), "owner/repo")
                .unwrap()
                .is_none()
        );
        assert!(
            release_row(&row.replace("\tmain\t", "\tfeature\t"), "owner/repo")
                .unwrap()
                .is_none()
        );
        for text in [
            row.replace("workflow_dispatch", "pull_request"),
            row.replace("release-delivery.yml", "ci.yml"),
            row.replace(&"a".repeat(40), "main"),
        ] {
            assert!(release_row(&text, "owner/repo").is_err());
        }
    }
    #[test]
    fn frozen_source_cannot_silently_switch_to_another_success() {
        let original = source();
        let mut copy = original.clone();
        copy.job_id += 1;
        assert!(same_execution(&original, &copy));
        copy.run_id = "13".into();
        assert!(!same_execution(&original, &copy));
        copy = original.clone();
        copy.completed_at = "2026-09-11T02:00:00Z".into();
        assert!(!same_execution(&original, &copy));
    }
}
