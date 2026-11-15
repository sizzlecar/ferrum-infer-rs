//! Declared lifecycle experiments share the timed client's bounded HTTP and
//! media handling, but never contribute cancellation samples to its SLO table.
//! A deleted record is deliberately not described as GPU retirement.
use super::{Client, Job, Observation, Status};
use crate::{input, validation, Arguments, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, path::Path, time::Instant};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Plan {
    actions: Vec<Action>,
    #[serde(skip)]
    source_sha256: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum Action {
    /// Each submission obtains the next zero-based slot, even on failure.
    Submit {
        request_index: usize,
    },
    /// Wait for declared integer observations from the public /health JSON.
    /// The report retains the actual response; no model-name dispatch occurs.
    WaitHealth {
        conditions: Vec<Condition>,
    },
    Delete {
        slot: usize,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Condition {
    pointer: String,
    comparison: Comparison,
    value: u64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Comparison {
    Equal,
    AtLeast,
}

impl Condition {
    fn matches(&self, response: &Value) -> Result<bool> {
        let actual = response
            .pointer(&self.pointer)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("health pointer {} is not an unsigned integer", self.pointer))?;
        Ok(match self.comparison {
            Comparison::Equal => actual == self.value,
            Comparison::AtLeast => actual >= self.value,
        })
    }
}

impl Plan {
    pub(crate) fn load(path: &Path, arguments: &Arguments, requests: usize) -> Result<Self> {
        let bytes = input::read_bounded(path, arguments.maximum_json_bytes)?;
        let mut plan: Self = serde_json::from_slice(&bytes)?;
        plan.source_sha256 = input::sha256(&bytes);
        if arguments.resources.resource_pid.is_some()
            || !arguments.resources.resource_gpu_index.is_empty()
        {
            return Err("lifecycle mode retains protocol observations; use timed cells for resource sampling".into());
        }
        plan.validate(requests)?;
        Ok(plan)
    }

    fn validate(&self, requests: usize) -> Result<()> {
        let mut submissions = 0;
        let mut deleted = BTreeSet::new();
        for action in &self.actions {
            match action {
                Action::Submit { request_index } => {
                    if *request_index >= requests {
                        return Err(
                            "lifecycle request_index is outside the supplied requests".into()
                        );
                    }
                    submissions += 1;
                }
                Action::Delete { slot } => {
                    if *slot >= submissions || !deleted.insert(*slot) {
                        return Err(
                            "delete must reference one earlier, not-yet-deleted slot".into()
                        );
                    }
                }
                Action::WaitHealth { conditions } => {
                    if conditions.is_empty()
                        || conditions.iter().any(|c| !c.pointer.starts_with('/'))
                    {
                        return Err("health gates require nonempty absolute JSON pointers".into());
                    }
                }
            }
        }
        if submissions == 0 {
            return Err("lifecycle plan must submit at least one request".into());
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct Entry {
    observation: Observation,
    delete_requested: bool,
    record_removed_after_delete: bool,
    observed_http_status: Option<u16>,
    #[serde(skip)]
    started: Instant,
    #[serde(skip)]
    observation_finished: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    mode: &'static str,
    execution_state: &'static str,
    client_executable_sha256: String,
    arguments: Arguments,
    input: input::Inputs,
    plan: Plan,
    plan_sha256: String,
    evidence_contract: &'static str,
    actions: Vec<Value>,
    requests: Vec<Entry>,
    stopped_reason: Option<String>,
}

impl Report {
    async fn save(&self) -> Result<()> {
        input::write_json_atomic(&self.arguments.output.join("report.json"), self).await
    }
}

async fn health_gate(client: &Client, conditions: &[Condition]) -> Result<Value> {
    let mut url = client.endpoint.clone();
    url.set_path("/health");
    let mut polls = 0u64;
    loop {
        let response = client
            .json(client.http.get(url.clone()).send().await?)
            .await?;
        polls += 1;
        let mut matched = true;
        for condition in conditions {
            matched &= condition.matches(&response)?;
        }
        if matched {
            return Ok(serde_json::json!({"polls":polls,"response":response}));
        }
        tokio::time::sleep(client.poll_interval).await;
    }
}

async fn action(
    client: &Client,
    report: &mut Report,
    index: usize,
    start: Instant,
) -> Result<Value> {
    match &report.plan.actions[index] {
        Action::Submit { request_index } => {
            let slot = report.requests.len();
            let started = Instant::now();
            report.requests.push(Entry {
                observation: Observation::new(slot, *request_index, start.elapsed().as_secs_f64()),
                delete_requested: false,
                record_removed_after_delete: false,
                observed_http_status: None,
                started,
                observation_finished: false,
            });
            // Install the unknown-submission owner before awaiting POST, so
            // cancellation of this future cannot erase a possibly live job.
            let entry = report.requests.last_mut().unwrap();
            let path = report.arguments.output.join(format!("request-{slot}.mp4"));
            let submitted = client
                .submit(
                    &report.input.requests[*request_index],
                    &path,
                    started,
                    &mut entry.observation,
                )
                .await;
            if let Err(error) = &submitted {
                entry.observation.error = Some(error.to_string());
            }
            submitted
        }
        Action::WaitHealth { conditions } => health_gate(client, conditions).await,
        Action::Delete { slot } => {
            let entry = &mut report.requests[*slot];
            let id = entry
                .observation
                .job_id
                .as_deref()
                .ok_or("delete has no original job ID")?;
            let url = client.job_url(id, false)?;
            entry.delete_requested = true;
            let response = client.json(client.http.delete(url).send().await?).await?;
            if response
                .get("id")
                .is_some_and(|value| value.as_str() != Some(id))
            {
                return Err("DELETE response job ID differs from the original submission".into());
            }
            entry.observation.cancellation = Some(response.clone());
            Ok(response)
        }
    }
}

async fn poll_entry(client: &Client, directory: &Path, entry: &mut Entry) -> Result<bool> {
    let observation = &mut entry.observation;
    let Some(id) = observation.job_id.as_deref() else {
        entry.observation_finished = true;
        return Ok(true);
    };
    let response = client.http.get(client.job_url(id, false)?).send().await?;
    entry.observed_http_status = Some(response.status().as_u16());
    observation.polls += 1;
    if response.status() == reqwest::StatusCode::NOT_FOUND && entry.delete_requested {
        entry.record_removed_after_delete = true;
        entry.observation_finished = true;
        // Other implementations may remove records while CUDA is still live.
        // Keep unknown=true until independent server retirement evidence exists.
        observation.request_returned_seconds = entry.started.elapsed().as_secs_f64();
        return Ok(true);
    }
    let raw = client.json(response).await?;
    let job: Job = serde_json::from_value(raw.clone())?;
    observation.last_job = Some(raw);
    if observation.job_id.as_deref() != Some(&job.id) {
        return Err("polled lifecycle job ID differs from the original submission".into());
    }
    match job.status {
        Status::Queued | Status::InProgress => return Ok(false),
        Status::Completed => {
            observation.remote_terminal_unknown = false;
            observation.terminal_observed_seconds = Some(entry.started.elapsed().as_secs_f64());
            if entry.delete_requested {
                return Err("job completed instead of observing the requested cancellation".into());
            }
            let path = directory.join(format!("request-{}.mp4", observation.index));
            client.download(&path, entry.started, observation).await?;
        }
        Status::Cancelled | Status::Canceled if entry.delete_requested => {
            observation.remote_terminal_unknown = false;
            observation.terminal_observed_seconds = Some(entry.started.elapsed().as_secs_f64());
        }
        status => {
            observation.remote_terminal_unknown = false;
            observation.terminal_observed_seconds = Some(entry.started.elapsed().as_secs_f64());
            return Err(format!("unexpected lifecycle terminal status {status:?}").into());
        }
    }
    observation.request_returned_seconds = entry.started.elapsed().as_secs_f64();
    entry.observation_finished = true;
    Ok(true)
}

async fn drain(client: &Client, report: &mut Report) -> Result<()> {
    while report
        .requests
        .iter()
        .any(|entry| !entry.observation_finished)
    {
        let mut changed = false;
        for entry in &mut report.requests {
            if entry.observation_finished {
                continue;
            }
            match poll_entry(client, &report.arguments.output, entry).await {
                Ok(updated) => changed |= updated,
                Err(error) => {
                    entry
                        .observation
                        .error
                        .get_or_insert_with(|| error.to_string());
                    entry.observation_finished = true;
                    changed = true;
                }
            }
        }
        if changed {
            report.save().await?;
        }
        if report
            .requests
            .iter()
            .any(|entry| !entry.observation_finished)
        {
            tokio::time::sleep(client.poll_interval).await;
        }
    }
    Ok(())
}

pub(crate) async fn run(
    client: Client,
    arguments: Arguments,
    input: input::Inputs,
    plan: Plan,
) -> Result<()> {
    std::fs::create_dir(&arguments.output)?;
    let plan_sha256 = plan.source_sha256.clone();
    let mut report = Report {
        schema_version: 1,
        mode: "lifecycle_observation",
        execution_state: "running",
        client_executable_sha256: input::file_sha256(&std::env::current_exe()?)?,
        arguments, input, plan, plan_sha256,
        evidence_contract: "Declared lifecycle actions; no throughput or SLO claim. GET 404 after DELETE proves only record removal, and remote_terminal_unknown remains true. Native fences, member retirement and replacement forwards require independently retained server execution evidence. Completed artifacts receive the same full media validation as timed requests. On action failure no further submissions occur; all known submitted jobs are still observed within the configured deadline.",
        actions: Vec::new(), requests: Vec::new(), stopped_reason: None,
    };
    report.save().await?;
    let start = Instant::now();
    for index in 0..report.plan.actions.len() {
        let before = start.elapsed().as_secs_f64();
        let result =
            tokio::time::timeout(client.timeout, action(&client, &mut report, index, start)).await;
        let result = match result {
            Ok(result) => result,
            Err(_) => Err("lifecycle action deadline exceeded; inspect original receipts".into()),
        };
        let receipt = match result {
            Ok(value) => serde_json::json!({"response":value}),
            Err(error) => {
                report.stopped_reason = Some(error.to_string());
                serde_json::json!({"error":error.to_string()})
            }
        };
        report.actions.push(serde_json::json!({"index":index,"started_seconds":before,"finished_seconds":start.elapsed().as_secs_f64(),"receipt":receipt}));
        report.save().await?;
        if report.stopped_reason.is_some() {
            break;
        }
    }
    match tokio::time::timeout(client.timeout, drain(&client, &mut report)).await {
        Ok(Ok(())) => {}
        result => {
            let error = match result {
                Ok(Err(error)) => error.to_string(),
                Err(_) => {
                    "lifecycle observation deadline exceeded; jobs may still be running".into()
                }
                Ok(Ok(())) => unreachable!(),
            };
            report.stopped_reason.get_or_insert(error);
        }
    }
    for entry in &mut report.requests {
        if let Some(path) = &entry.observation.artifact {
            let expected = &report.input.requests[entry.observation.request_index].expected;
            match validation::validate(&report.arguments, path, expected).await {
                Ok(media) => entry.observation.media = Some(media),
                Err(error) => {
                    entry.observation.error.get_or_insert(error.to_string());
                }
            }
        }
    }
    let failed = report.stopped_reason.is_some()
        || report.requests.iter().any(|entry| {
            entry.observation.error.is_some()
                || entry.observation.evidence_error.is_some()
                || !entry.observation_finished
        });
    report.execution_state = if failed {
        "stopped"
    } else {
        "observations_complete"
    };
    report.save().await?;
    println!("{}", report.arguments.output.join("report.json").display());
    if failed {
        return Err(
            "lifecycle observations failed; inspect retained report before more GPU work".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod tests;
