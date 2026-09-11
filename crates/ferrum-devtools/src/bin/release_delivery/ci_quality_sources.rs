//! Cross-candidate quality reuse is separate from raw producer verification.
//! Original reports retain their source run/commit; final package and model
//! evidence always belongs to the candidate being published.
use super::*;

// The publisher and rustc-only Actions planner must use the same scope and
// failure-history rules. This development tool is not a published crate.
#[allow(dead_code)]
#[path = "../../../../../.github/ci/policy.rs"]
mod policy;
pub(super) use policy::checks::Check;
use policy::release_reuse::{ReleasePlan, Source};

const PLAN_JOB: &str = "Quality / Plan and validate docs";
const PLAN_EXECUTE: &str = "Classify changes and check diff";
const PLAN_UPLOAD: &str = "Save checked source identity";
const CONTRACT_EXECUTE: &str = "Execute registered correctness boundaries";
const CONTRACT_UPLOAD: &str = "Save executed contract evidence";

struct Context {
    run: Value,
    inventory: Vec<Value>,
    jobs: Vec<Job>,
}
pub(super) struct QualitySources {
    plan: ReleasePlan,
    contexts: BTreeMap<u64, Context>,
    current_run: u64,
    workspace: String,
}

impl QualitySources {
    pub(super) async fn load(
        github: &Github,
        run: &Value,
        inventory: &[Value],
        jobs: &[Job],
        workspace: &Path,
        contracts: &Path,
    ) -> Result<(Self, Vec<Artifact>), String> {
        let current_run = run["id"].as_u64().ok_or("quality run ID missing")?;
        let candidate = run["head_sha"]
            .as_str()
            .ok_or("quality candidate missing")?;
        let frozen = artifact(
            github,
            run,
            inventory,
            jobs,
            "ci-plan",
            PLAN_JOB,
            current_run,
            candidate,
        )
        .await?;
        uploaded_during(
            &frozen.metadata,
            &frozen.producer,
            PLAN_EXECUTE,
            PLAN_UPLOAD,
        )?;
        let plan = ReleasePlan::parse(
            &fs::read_to_string(frozen.directory.path().join("release-plan.txt"))
                .map_err(|e| format!("frozen quality plan: {e}"))?,
        )?;
        let workspace = workspace
            .to_str()
            .ok_or("non-UTF8 release workspace")?
            .to_owned();
        let checkout = std::process::Command::new("git")
            .args(["-C", &workspace, "rev-parse", "HEAD"])
            .output()
            .map_err(|e| e.to_string())?;
        if !checkout.status.success()
            || String::from_utf8_lossy(&checkout.stdout).trim() != candidate
        {
            return Err("quality input comparison must use the release candidate checkout".into());
        }
        let mut contexts = BTreeMap::from([(
            current_run,
            Context {
                run: run.clone(),
                inventory: inventory.to_vec(),
                jobs: jobs.to_vec(),
            },
        )]);
        for source in &plan.sources {
            let source_run: u64 = source
                .run_id
                .parse()
                .map_err(|_| "invalid frozen source run")?;
            if contexts.contains_key(&source_run) {
                continue;
            }
            let path = format!("actions/runs/{source_run}");
            let run = github.json(&path).await?;
            verify_run(&run, &github.repo, source_run, &source.revision)?;
            if run["head_branch"] != "main" {
                return Err("quality source is not a main release run".into());
            }
            contexts.insert(
                source_run,
                Context {
                    run,
                    inventory: github
                        .inventory(&format!("{path}/artifacts"), "artifacts")
                        .await?,
                    jobs: github.jobs(&format!("{path}/jobs?filter=all")).await?,
                },
            );
        }
        let selected = Self {
            plan,
            contexts,
            current_run,
            workspace,
        };
        selected.revalidate(github).await?;
        let raw = selected
            .artifact(github, Check::Cpu, "contract-checks")
            .await?;
        uploaded_during(
            &raw.metadata,
            &raw.producer,
            CONTRACT_EXECUTE,
            CONTRACT_UPLOAD,
        )?;
        if fs::read(raw.directory.path().join("contract-checks.json")).map_err(|e| e.to_string())?
            != fs::read(contracts).map_err(|e| e.to_string())?
        {
            return Err(
                "provided CPU contracts differ from the verified original producer artifact".into(),
            );
        }
        Ok((selected, vec![frozen, raw]))
    }

    pub(super) async fn revalidate(&self, github: &Github) -> Result<(), String> {
        let plan = self.plan.clone();
        let repo = github.repo.clone();
        let run = self.current_run.to_string();
        let attempt = self.contexts[&self.current_run].run["run_attempt"]
            .as_u64()
            .ok_or("quality attempt missing")?;
        let workspace = self.workspace.clone();
        tokio::task::spawn_blocking(move || {
            policy::release_reuse::verify(&plan, &repo, &run, attempt, &workspace)
        })
        .await
        .map_err(|e| e.to_string())?
    }

    pub(super) async fn artifact(
        &self,
        github: &Github,
        check: Check,
        name: &str,
    ) -> Result<Artifact, String> {
        let source = self.plan.source(check);
        let run_id = source
            .map(|source| source.run_id.parse::<u64>().unwrap())
            .unwrap_or(self.current_run);
        let context = &self.contexts[&run_id];
        let history = execution_history(&context.jobs, source.is_some());
        let mut raw = artifact(
            github,
            &context.run,
            &context.inventory,
            &history,
            name,
            &format!("Quality / {}", check.name()),
            run_id,
            context.run["head_sha"].as_str().unwrap(),
        )
        .await?;
        if let Some(source) = source {
            verify_frozen_producer(source, &raw, &context.jobs)?;
            raw.reused_quality = true;
        }
        Ok(raw)
    }
}

pub(super) fn execution_history(jobs: &[Job], reused_quality: bool) -> Vec<Job> {
    jobs.iter()
        .filter(|job| {
            !reused_quality
                || job.status != "completed"
                || job.conclusion.as_deref() != Some("skipped")
        })
        .cloned()
        .collect()
}

fn verify_frozen_producer(source: &Source, raw: &Artifact, jobs: &[Job]) -> Result<(), String> {
    let frozen = jobs
        .iter()
        .find(|job| job.id == source.job_id)
        .ok_or("frozen quality job disappeared")?;
    if raw.origin.run_id.to_string() != source.run_id
        || raw.origin.head_sha != source.revision
        || raw.producer.completed_at.as_deref() != Some(source.completed_at.as_str())
        || frozen.completed_at != raw.producer.completed_at
        || frozen.started_at != raw.producer.started_at
        || frozen.steps != raw.producer.steps
        || frozen.name != raw.producer.name
        || frozen.status != "completed"
        || frozen.conclusion.as_deref() != Some("success")
        || raw.producer.name != format!("Quality / {}", source.check.name())
    {
        return Err("raw quality report does not belong to the frozen original execution".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw() -> Artifact {
        Artifact {
            metadata: Value::Null,
            origin: Origin {
                schema_version: 1,
                run_id: 12,
                run_attempt: 1,
                head_sha: "a".repeat(40),
                job_id: 31,
            },
            producer: Job {
                id: 31,
                run_id: 12,
                run_attempt: 1,
                head_sha: "a".repeat(40),
                name: "Quality / CPU (Linux)".into(),
                status: "completed".into(),
                conclusion: Some("success".into()),
                started_at: Some("2026-09-11T01:00:00Z".into()),
                completed_at: Some("2026-09-11T01:01:00Z".into()),
                steps: Vec::new(),
            },
            directory: tempfile::tempdir().unwrap(),
            reused_quality: true,
        }
    }
    fn source() -> Source {
        Source {
            check: Check::Cpu,
            run_id: "12".into(),
            job_id: 31,
            revision: "a".repeat(40),
            completed_at: "2026-09-11T01:01:00Z".into(),
            execution: "W10=".into(),
        }
    }

    #[test]
    fn frozen_quality_source_keeps_original_run_revision_and_execution() {
        let raw = raw();
        let mut copied = raw.producer.clone();
        copied.id = 32;
        copied.run_attempt = 2;
        let mut source = source();
        source.job_id = copied.id;
        assert!(verify_frozen_producer(&source, &raw, &[copied.clone()]).is_ok());
        for mutation in ["failure", "later", "revision", "run", "check", "job"] {
            let mut source = source.clone();
            let mut job = copied.clone();
            match mutation {
                "failure" => job.conclusion = Some("failure".into()),
                "later" => job.started_at = Some("2026-09-11T01:00:01Z".into()),
                "revision" => source.revision = "b".repeat(40),
                "run" => source.run_id = "13".into(),
                "check" => source.check = Check::CudaRuntime,
                "job" => source.job_id += 1,
                _ => unreachable!(),
            }
            assert!(
                verify_frozen_producer(&source, &raw, &[job]).is_err(),
                "{mutation}"
            );
        }
    }

    #[test]
    fn expected_quality_skip_does_not_erase_a_later_real_failure() {
        let raw = raw();
        let mut skipped = raw.producer.clone();
        skipped.id = 32;
        skipped.run_attempt = 2;
        skipped.conclusion = Some("skipped".into());
        let mut failed = skipped.clone();
        failed.id = 33;
        failed.run_attempt = 3;
        failed.conclusion = Some("failure".into());
        let verify = |jobs: &[Job], reused| {
            producer(
                &raw.origin,
                std::slice::from_ref(&raw.producer),
                &execution_history(jobs, reused),
                12,
                &"a".repeat(40),
                3,
                "Quality / CPU (Linux)",
            )
        };
        assert!(verify(&[raw.producer.clone(), skipped.clone()], true).is_ok());
        assert!(verify(&[raw.producer.clone(), skipped.clone()], false).is_err());
        assert!(verify(&[raw.producer.clone(), skipped, failed], true).is_err());
    }
}
