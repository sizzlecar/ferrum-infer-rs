//! Independent cleanup after the owning workflow finishes, or its lease expires.
use super::{api, destroy_confirm, now, write_json, Ownership};
use clap::Args;
use reqwest::Client;
use serde_json::{json, Value};
use std::{path::PathBuf, time::Duration};

#[derive(Debug, Args)]
pub struct ReapArgs {
    #[arg(long)]
    pub repository_id: u64,
    /// Omit for the scheduled repository-wide expiry reaper.
    #[arg(long)]
    pub run_id: Option<u64>,
    /// Check the real GitHub run before reclaiming an unexpired lease. This is
    /// not a force-delete flag: active or unrecognized runs never authorize it.
    #[arg(long, requires = "run_id")]
    pub completed_run_repository: Option<String>,
    #[arg(long)]
    pub output: PathBuf,
}

struct Github {
    http: Client,
    base: String,
    token: String,
    repository: String,
}

fn validate_repository(repository: &str) -> Result<(), String> {
    let parts: Vec<_> = repository.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || matches!(*part, "." | "..")
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        })
    {
        return Err("completed run repository must be owner/name".into());
    }
    Ok(())
}

impl Github {
    fn new(repository: String, token: String) -> Result<Self, String> {
        validate_repository(&repository)?;
        if token.trim().is_empty() {
            return Err("GITHUB_TOKEN is empty".into());
        }
        Ok(Self {
            http: Client::builder()
                .timeout(Duration::from_secs(30))
                .redirect(reqwest::redirect::Policy::none())
                .user_agent("ferrum-release-lease-reaper")
                .build()
                .map_err(|_| "build GitHub HTTP client")?,
            base: "https://api.github.com".into(),
            token,
            repository,
        })
    }

    async fn completed_attempt(&self, repository_id: u64, run: u64) -> Result<Option<u32>, String> {
        let mut response = self
            .http
            .get(format!(
                "{}/repos/{}/actions/runs/{run}",
                self.base, self.repository
            ))
            .bearer_auth(&self.token)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .map_err(|_| "GitHub run lookup transport failure")?;
        if !response.status().is_success() {
            return Err(format!("GitHub run lookup HTTP {}", response.status()));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| "GitHub run lookup incomplete response")?
        {
            if bytes.len() + chunk.len() > 1024 * 1024 {
                return Err("GitHub run response exceeds 1 MiB".into());
            }
            bytes.extend_from_slice(&chunk);
        }
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|_| "GitHub run lookup invalid JSON")?;
        completed_attempt(&value, &self.repository, repository_id, run)
    }
}

fn completed_attempt(
    value: &Value,
    repository: &str,
    repository_id: u64,
    run: u64,
) -> Result<Option<u32>, String> {
    if value["id"].as_u64() != Some(run)
        || value["repository"]["id"].as_u64() != Some(repository_id)
        || value["repository"]["full_name"].as_str() != Some(repository)
        || value["head_repository"]["id"].as_u64() != Some(repository_id)
        || value["path"] != ".github/workflows/release-delivery.yml"
        || !matches!(value["event"].as_str(), Some("push" | "workflow_dispatch"))
    {
        return Err("GitHub run does not identify the owned release workflow".into());
    }
    let attempt = value["run_attempt"]
        .as_u64()
        .and_then(|attempt| u32::try_from(attempt).ok())
        .filter(|attempt| *attempt > 0)
        .ok_or("GitHub run omitted a valid attempt")?;
    match value["status"].as_str() {
        Some("completed")
            if matches!(
                value["conclusion"].as_str(),
                Some(
                    "success"
                        | "failure"
                        | "cancelled"
                        | "timed_out"
                        | "action_required"
                        | "neutral"
                        | "skipped"
                        | "stale"
                )
            ) =>
        {
            Ok(Some(attempt))
        }
        Some("queued" | "in_progress" | "waiting" | "requested" | "pending") => Ok(None),
        _ => Err("GitHub run has an unknown or incomplete terminal state".into()),
    }
}

async fn reap_instances(
    args: &ReapArgs,
    provider: &api::Client,
    github: Option<&Github>,
    observed_at: u64,
) -> Result<Vec<Value>, String> {
    let mut results = Vec::new();
    for instance in provider.instances().await? {
        let Some(owner) = instance.label.as_deref().and_then(Ownership::parse) else {
            continue;
        };
        if owner.repository != args.repository_id || args.run_id.is_some_and(|run| run != owner.run)
        {
            continue;
        }
        let expired = owner.expiry <= observed_at;
        if !expired && github.is_none() {
            continue;
        }
        // The list is only a discovery mechanism, never delete authorization.
        let current = provider.instance(instance.id).await?;
        if current
            .as_ref()
            .is_none_or(|current| current.label != instance.label)
        {
            continue;
        }
        let completed = if expired {
            None
        } else {
            // Read the latest run immediately before deleting. An old completed
            // event cannot authorize deletion during a rerun of that same run ID.
            match github
                .expect("unexpired cleanup requires GitHub")
                .completed_attempt(args.repository_id, owner.run)
                .await?
            {
                Some(attempt) if owner.attempt <= attempt => Some(attempt),
                Some(_) => return Err("lease attempt is newer than the verified GitHub run".into()),
                None => {
                    results.push(json!({"instance_id":instance.id,"label":instance.label,
                        "retained":"owning_workflow_is_active"}));
                    continue;
                }
            }
        };
        let result = tokio::time::timeout(
            Duration::from_secs(120),
            destroy_confirm(provider, instance.id),
        )
        .await
        .map_err(|_| "cleanup deadline exceeded".to_string())
        .and_then(|value| value);
        results.push(json!({"instance_id":instance.id,"label":instance.label,
            "reason":if expired {"lease_expired"} else {"owning_workflow_completed"},
            "completed_run_attempt":completed,
            "destroyed_and_absent":result.is_ok(),"error":result.err()}));
    }
    Ok(results)
}

pub async fn reap(args: ReapArgs) -> Result<(), String> {
    if args.repository_id == 0
        || args.run_id == Some(0)
        || (args.completed_run_repository.is_some() && args.run_id.is_none())
    {
        return Err("positive repository/run ID required for completed-run cleanup".into());
    }
    if args.output.exists() {
        return Err("reaper output must be new".into());
    }
    let github = args
        .completed_run_repository
        .as_ref()
        .map(|repository| {
            Github::new(
                repository.clone(),
                std::env::var("GITHUB_TOKEN")
                    .map_err(|_| "GITHUB_TOKEN is required for completed-run verification")?,
            )
        })
        .transpose()?;
    let client =
        api::Client::new(std::env::var("VAST_API_KEY").map_err(|_| "VAST_API_KEY is required")?)?;
    let observed_at = now()?;
    let result = reap_instances(&args, &client, github.as_ref(), observed_at).await;
    let (results, error) = match result {
        Ok(results) => (results, None),
        Err(error) => (Vec::new(), Some(error)),
    };
    let failed = error.is_some() || results.iter().any(|result| !result["error"].is_null());
    write_json(
        &args.output,
        &json!({"schema_version":1,"repository_id":args.repository_id,"run_id":args.run_id,
            "completed_run_repository":args.completed_run_repository,"observed_at":observed_at,
            "results":results,"error":error,"passed":!failed}),
    )?;
    if failed {
        return Err("owned lease cleanup failed; see reaper report".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "reaper_tests.rs"]
mod tests;
