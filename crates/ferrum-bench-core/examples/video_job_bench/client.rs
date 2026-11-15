use super::{
    input::Request,
    report::{Cell, Observation, Summary},
    resources::{RunningWindow, ServerSampler},
    validation, Arguments, Result,
};
use reqwest::{header, Url};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::{io::AsyncWriteExt, task::JoinSet};

#[path = "lifecycle.rs"]
pub(super) mod lifecycle;

pub struct Client {
    http: reqwest::Client,
    endpoint: Url,
    poll_interval: Duration,
    timeout: Duration,
    json_limit: usize,
    artifact_limit: u64,
    cancel_on_timeout: bool,
}

#[derive(Debug, Deserialize)]
struct Job {
    id: String,
    status: Status,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Canceled,
}

impl Client {
    pub async fn new(arguments: &Arguments) -> Result<Self> {
        let mut endpoint = Url::parse(&arguments.base_url)?;
        if !matches!(endpoint.scheme(), "http" | "https")
            || endpoint.host_str().is_none()
            || !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
            || endpoint.path() != "/"
        {
            return Err(
                "base-url must be an HTTP(S) origin without credentials, path or query".into(),
            );
        }
        endpoint.set_path("/v1/videos");
        let mut headers = header::HeaderMap::new();
        if let Some(path) = &arguments.bearer_token_file {
            let token = tokio::fs::read_to_string(path).await?;
            let mut value = header::HeaderValue::from_str(&format!("Bearer {}", token.trim()))?;
            value.set_sensitive(true);
            headers.insert(header::AUTHORIZATION, value);
        }
        Ok(Self {
            http: reqwest::Client::builder()
                .default_headers(headers)
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            endpoint,
            poll_interval: Duration::from_millis(arguments.poll_interval_ms),
            timeout: Duration::from_secs(arguments.timeout_seconds),
            json_limit: arguments.maximum_json_bytes,
            artifact_limit: arguments.maximum_artifact_bytes,
            cancel_on_timeout: arguments.cancel_on_timeout,
        })
    }

    fn job_url(&self, id: &str, content: bool) -> Result<Url> {
        if id.is_empty()
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return Err("server returned an invalid job ID".into());
        }
        let mut url = self.endpoint.clone();
        let mut segments = url.path_segments_mut().map_err(|_| "invalid endpoint")?;
        segments.push(id);
        if content {
            segments.push("content");
        }
        drop(segments);
        Ok(url)
    }

    async fn json(&self, mut response: reqwest::Response) -> Result<Value> {
        let status = response.status();
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if chunk.len() > self.json_limit.saturating_sub(bytes.len()) {
                return Err("response JSON exceeds configured byte limit".into());
            }
            bytes.extend_from_slice(&chunk);
        }
        if !status.is_success() {
            return Err(format!("HTTP {status}: {}", String::from_utf8_lossy(&bytes)).into());
        }
        Ok(serde_json::from_slice(&bytes)?)
    }

    async fn submit(
        &self,
        request: &Request,
        output: &PathBuf,
        start: Instant,
        observation: &mut Observation,
    ) -> Result<Value> {
        // The POST may have reached the server even if its response is lost.
        observation.remote_terminal_unknown = true;
        let response = self
            .http
            .post(self.endpoint.clone())
            .json(&request.body)
            .send()
            .await?;
        if response.status().is_client_error() {
            observation.remote_terminal_unknown = false;
        }
        let raw = self.json(response).await?;
        observation.submission_seconds = Some(start.elapsed().as_secs_f64());
        observation.last_job = Some(raw.clone());
        let initial: Job = serde_json::from_value(raw.clone())?;
        self.job_url(&initial.id, false)?;
        observation.job_id = Some(initial.id);
        // Retain the receipt even if this process is later interrupted.
        if let Err(error) =
            super::input::write_json_atomic(&output.with_extension("submission.json"), &raw).await
        {
            observation.evidence_error = Some(format!("submission receipt: {error}"));
            return Err(error);
        }
        Ok(raw)
    }

    async fn execute(
        &self,
        request: &Request,
        output: &PathBuf,
        start: Instant,
        observation: &mut Observation,
    ) -> Result<()> {
        let mut raw = self.submit(request, output, start, observation).await?;
        loop {
            observation.last_job = Some(raw.clone());
            let job: Job = serde_json::from_value(raw)?;
            let url = self.job_url(&job.id, false)?;
            if observation.job_id.as_ref().is_some_and(|id| id != &job.id) {
                return Err("polled job ID does not match submitted ID".into());
            }
            observation.job_id = Some(job.id.clone());
            match job.status {
                Status::Completed => {
                    observation.remote_terminal_unknown = false;
                    observation.terminal_observed_seconds = Some(start.elapsed().as_secs_f64());
                    break;
                }
                Status::Failed | Status::Cancelled | Status::Canceled => {
                    observation.remote_terminal_unknown = false;
                    observation.terminal_observed_seconds = Some(start.elapsed().as_secs_f64());
                    return Err(format!("terminal job status {:?}", job.status).into());
                }
                Status::Queued | Status::InProgress => {}
            }
            tokio::time::sleep(self.poll_interval).await;
            observation.polls += 1;
            raw = self.json(self.http.get(url).send().await?).await?;
        }
        self.download(output, start, observation).await
    }

    async fn download(
        &self,
        output: &PathBuf,
        start: Instant,
        observation: &mut Observation,
    ) -> Result<()> {
        let url = self.job_url(observation.job_id.as_deref().ok_or("missing job ID")?, true)?;
        let mut response = self.http.get(url).send().await?.error_for_status()?;
        if response
            .content_length()
            .is_some_and(|size| size > self.artifact_limit)
        {
            return Err("artifact Content-Length exceeds configured byte limit".into());
        }
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .await?;
        let mut bytes = 0u64;
        let mut hash = Sha256::new();
        while let Some(chunk) = response.chunk().await? {
            if chunk.len() as u64 > self.artifact_limit.saturating_sub(bytes) {
                return Err("artifact exceeds configured byte limit".into());
            }
            file.write_all(&chunk).await?;
            hash.update(&chunk);
            bytes += chunk.len() as u64;
        }
        file.flush().await?;
        if bytes == 0 {
            return Err("downloaded artifact is empty".into());
        }
        observation.artifact_received_seconds = Some(start.elapsed().as_secs_f64());
        observation.artifact = Some(output.clone());
        observation.artifact_bytes = Some(bytes);
        observation.artifact_sha256 = Some(format!("{:x}", hash.finalize()));
        Ok(())
    }

    async fn request(
        &self,
        request: &Request,
        output: PathBuf,
        index: usize,
        request_index: usize,
        cell_start: Instant,
    ) -> Observation {
        let start = Instant::now();
        let mut observation =
            Observation::new(index, request_index, cell_start.elapsed().as_secs_f64());
        let execution = tokio::time::timeout(
            self.timeout,
            self.execute(request, &output, start, &mut observation),
        )
        .await;
        match execution {
            Ok(Ok(())) => {}
            Ok(Err(error)) => observation.error = Some(error.to_string()),
            Err(_) => {
                observation.error = Some(format!(
                    "client deadline exceeded ({} seconds)",
                    self.timeout.as_secs()
                ));
                if self.cancel_on_timeout && observation.remote_terminal_unknown {
                    if let Some(id) = &observation.job_id {
                        let cancellation = async {
                            let url = self.job_url(id, false)?;
                            self.json(self.http.delete(url).send().await?).await
                        };
                        observation.cancellation = Some(
                            match tokio::time::timeout(Duration::from_secs(10), cancellation).await
                            {
                                Ok(Ok(value)) => value,
                                Ok(Err(error)) => serde_json::json!({"error":error.to_string()}),
                                Err(_) => serde_json::json!({"error":"DELETE timeout"}),
                            },
                        );
                    }
                }
            }
        }
        observation.request_returned_seconds = start.elapsed().as_secs_f64();
        observation
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn run_cell(
    client: Arc<Client>,
    arguments: &Arguments,
    requests: &[Request],
    phase: &str,
    concurrency: usize,
    width: usize,
    repeat: usize,
    count: usize,
    sampler: Option<Arc<ServerSampler>>,
) -> Result<Cell> {
    let directory = arguments
        .output
        .join(format!("c{concurrency}-r{repeat}-{phase}"));
    tokio::fs::create_dir(&directory).await?;
    let resource_window = match sampler {
        Some(sampler) => Some(RunningWindow::begin(sampler).await),
        None => None,
    };
    let counter = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let requests = Arc::new(requests.to_vec());
    let observations = Arc::new(Mutex::new(Vec::new()));
    let start = Instant::now();
    let mut tasks = JoinSet::new();
    for _ in 0..width.min(count) {
        let (client, counter, requests, directory, stop, observations) = (
            client.clone(),
            counter.clone(),
            requests.clone(),
            directory.clone(),
            stop.clone(),
            observations.clone(),
        );
        tasks.spawn(async move {
            loop {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                let index = counter.fetch_add(1, Ordering::AcqRel);
                if index >= count {
                    break;
                }
                let request_index = index % requests.len();
                let mut observation = client
                    .request(
                        &requests[request_index],
                        directory.join(format!("request-{index}.mp4")),
                        index,
                        request_index,
                        start,
                    )
                    .await;
                if let Err(error) = super::input::write_json_atomic(
                    &directory.join(format!("request-{index}.observation.json")),
                    &observation,
                )
                .await
                {
                    observation.evidence_error = Some(format!("observation receipt: {error}"));
                }
                if observation.remote_terminal_unknown || observation.evidence_error.is_some() {
                    stop.store(true, Ordering::Release);
                }
                observations
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(observation);
            }
        });
    }
    let mut client_errors = Vec::new();
    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result {
            stop.store(true, Ordering::Release);
            client_errors.push(format!("request worker failed: {error}"));
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let resources = match resource_window {
        Some(window) => Some(window.finish().await),
        None => None,
    };
    let mut observations = std::mem::take(
        &mut *observations
            .lock()
            .unwrap_or_else(|error| error.into_inner()),
    );
    let first_dispatch = observations
        .iter()
        .map(|o| o.dispatch_offset_seconds)
        .reduce(f64::min)
        .unwrap_or(0.0);
    let last_return = observations
        .iter()
        .map(|o| o.dispatch_offset_seconds + o.request_returned_seconds)
        .fold(0.0, f64::max);
    let seconds = if client_errors.is_empty() {
        last_return - first_dispatch
    } else {
        elapsed
    };
    let present: std::collections::BTreeSet<_> = observations.iter().map(|o| o.index).collect();
    let assigned = counter.load(Ordering::Acquire).min(count);
    // Worker failure cannot erase earlier completed observations or change the denominator.
    for index in (0..count).filter(|index| !present.contains(index)) {
        let mut record = Observation::new(index, index % requests.len(), seconds);
        record.dispatched = index < assigned;
        record.remote_terminal_unknown = record.dispatched;
        record.error = Some(
            if record.dispatched {
                "worker failed after claiming request; submission and terminal state are unknown"
            } else {
                "not dispatched after a prior request or evidence failure"
            }
            .into(),
        );
        if record.dispatched {
            if let Ok(bytes) =
                tokio::fs::read(directory.join(format!("request-{index}.submission.json"))).await
            {
                if let Ok(raw) = serde_json::from_slice::<Value>(&bytes) {
                    record.job_id = raw.get("id").and_then(Value::as_str).map(str::to_owned);
                    record.last_job = Some(raw);
                }
            }
        }
        observations.push(record);
    }
    observations.sort_by_key(|o| o.index);
    for observation in &mut observations {
        if observation.error.is_none() {
            let result = match observation.artifact.as_ref() {
                Some(artifact) => {
                    validation::validate(
                        arguments,
                        artifact,
                        &requests[observation.request_index].expected,
                    )
                    .await
                }
                None => Err("successful request missing artifact".into()),
            };
            match result {
                Ok(media) => observation.media = Some(media),
                Err(error) => observation.error = Some(format!("media validation: {error}")),
            }
        }
        if let Err(error) = super::input::write_json_atomic(
            &directory.join(format!("request-{}.observation.json", observation.index)),
            observation,
        )
        .await
        {
            observation.evidence_error = Some(format!("final observation receipt: {error}"));
        }
    }
    let mut summary = Summary::calculate(&observations, seconds, arguments.latency_slo_p99_seconds);
    if let Some(resources) = &resources {
        summary.peak_vram_bytes = resources
            .device
            .as_ref()
            .and_then(|m| m.observed_peak_bytes);
        summary.peak_server_rss_bytes = resources.host.as_ref().and_then(|m| m.observed_peak_bytes);
    }
    Ok(Cell {
        phase: phase.into(),
        configured_concurrency: concurrency,
        actual_client_width: width.min(count),
        repeat,
        summary,
        requests: observations,
        resources,
        client_errors,
    })
}

#[cfg(test)]
#[path = "client_tests.rs"]
mod tests;
