//! Bounded passive /health sampler, independent of open-arrival dispatch.
use super::*;
use tokio::sync::oneshot;

const MAX_RESPONSE_BYTES: usize = 128 * 1024;

struct TimedAttempt {
    started: Instant,
    completed: Instant,
    observation: std::result::Result<ferrum_types::ExecutorQueueObservation, ServerQueueFailure>,
}
impl TimedAttempt {
    fn relative(self, origin: Instant) -> ServerQueueAttempt {
        fn offset(at: Instant, origin: Instant) -> f64 {
            at.checked_duration_since(origin)
                .map(|d| d.as_secs_f64())
                .unwrap_or_else(|| -origin.duration_since(at).as_secs_f64())
        }
        ServerQueueAttempt {
            request_started_seconds: offset(self.started, origin),
            response_completed_seconds: offset(self.completed, origin),
            observation: self.observation,
        }
    }
}

pub(super) struct Prepared {
    client: reqwest::Client,
    url: String,
    period: Duration,
    baseline: TimedAttempt,
}
impl Prepared {
    pub async fn new(ctx: &RunContext, maximum_gap: f64) -> Result<Self> {
        // Freeze derives both cadence and HTTP deadline from the declared
        // maximum gap. Four subintervals allow one missed read without hiding
        // its error; actual bracket coverage is always evaluated from raw data.
        let period =
            Duration::try_from_secs_f64(maximum_gap / 4.0).map_err(|e| err(e.to_string()))?;
        if period.is_zero() {
            return Err(err("server queue cadence is below clock resolution"));
        }
        Instant::now()
            .checked_add(period)
            .ok_or_else(|| err("server queue cadence exceeds monotonic clock range"))?;
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(1)
            .timeout(period)
            .build()
            .map_err(|e| err(e.to_string()))?;
        let url = format!("{}/health", ctx.base_url.trim_end_matches('/'));
        let baseline = sample(&client, &url, period).await;
        Ok(Self {
            client,
            url,
            period,
            baseline,
        })
    }
    pub fn start(self, origin: Instant, maximum: usize) -> Sampler {
        let (stop, mut stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            let mut attempts = Vec::new();
            let mut complete = true;
            retain(
                &mut attempts,
                self.baseline.relative(origin),
                maximum,
                &mut complete,
            );
            let mut next = origin + self.period;
            loop {
                let final_read = tokio::select! {
                    _ = &mut stopped => true,
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(next)) => false,
                };
                if attempts.len() < maximum {
                    let attempt = sample(&self.client, &self.url, self.period)
                        .await
                        .relative(origin);
                    retain(&mut attempts, attempt, maximum, &mut complete);
                } else {
                    complete = false;
                    if !final_read {
                        let _ = stopped.await;
                    }
                    break;
                }
                if final_read {
                    break;
                }
                next += self.period;
                // A delayed observer skips past ticks, rather than producing
                // an artificial burst of almost identical samples.
                if next <= Instant::now() {
                    next = Instant::now() + self.period;
                }
            }
            (attempts, complete)
        });
        Sampler {
            stop: Some(stop),
            task: Some(task),
        }
    }
}

type Task = tokio::task::JoinHandle<(Vec<ServerQueueAttempt>, bool)>;
pub(super) struct Sampler {
    stop: Option<oneshot::Sender<()>>,
    task: Option<Task>,
}
impl Sampler {
    pub async fn finish(mut self) -> Result<(Vec<ServerQueueAttempt>, bool)> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let result = self
            .task
            .as_mut()
            .expect("owned queue sampler")
            .await
            .map_err(|e| err(format!("server queue sampler failed: {e}")));
        self.task.take();
        result
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
fn retain(
    attempts: &mut Vec<ServerQueueAttempt>,
    attempt: ServerQueueAttempt,
    maximum: usize,
    complete: &mut bool,
) {
    if attempts.len() < maximum {
        attempts.push(attempt);
    } else {
        *complete = false;
    }
}
async fn sample(client: &reqwest::Client, url: &str, timeout: Duration) -> TimedAttempt {
    let started = Instant::now();
    let observation = match tokio::time::timeout(timeout, receive(client, url)).await {
        Ok(result) => result,
        Err(_) => Err(ServerQueueFailure::Timeout),
    };
    TimedAttempt {
        started,
        completed: Instant::now(),
        observation,
    }
}
async fn receive(
    client: &reqwest::Client,
    url: &str,
) -> std::result::Result<ferrum_types::ExecutorQueueObservation, ServerQueueFailure> {
    let mut response = client.get(url).send().await.map_err(|e| {
        if e.is_timeout() {
            ServerQueueFailure::Timeout
        } else {
            ServerQueueFailure::Transport
        }
    })?;
    if !response.status().is_success() {
        return Err(ServerQueueFailure::HttpStatus(response.status().as_u16()));
    }
    if response
        .content_length()
        .is_some_and(|n| n > MAX_RESPONSE_BYTES as u64)
    {
        return Err(ServerQueueFailure::ResponseTooLarge);
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| ServerQueueFailure::Transport)?
    {
        if bytes
            .len()
            .checked_add(chunk.len())
            .is_none_or(|n| n > MAX_RESPONSE_BYTES)
        {
            return Err(ServerQueueFailure::ResponseTooLarge);
        }
        bytes.extend_from_slice(&chunk);
    }
    decode(&bytes)
}
fn decode(
    bytes: &[u8],
) -> std::result::Result<ferrum_types::ExecutorQueueObservation, ServerQueueFailure> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| ServerQueueFailure::Malformed)?;
    let admission = &value["admission"];
    if let Some(message) = admission["queue_observation_error"].as_str() {
        let mut end = message.len().min(512);
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        return Err(ServerQueueFailure::Runtime(message[..end].to_owned()));
    }
    if admission["runtime_snapshot_available"] != true || admission["queue_observation"].is_null() {
        return Err(ServerQueueFailure::Unavailable);
    }
    let observation: ferrum_types::ExecutorQueueObservation =
        serde_json::from_value(admission["queue_observation"].clone())
            .map_err(|_| ServerQueueFailure::Malformed)?;
    observation
        .validate()
        .map_err(|_| ServerQueueFailure::Malformed)?;
    if admission["queue_depth"].as_u64() != Some(u64::from(observation.waiting_requests))
        || admission["active_prefill"].as_u64()
            != Some(u64::from(observation.active_prefill_sequences))
        || admission["active_decode"].as_u64()
            != Some(u64::from(observation.active_decode_sequences))
    {
        return Err(ServerQueueFailure::Malformed);
    }
    Ok(observation)
}

#[cfg(test)]
#[path = "server_queue/tests.rs"]
mod tests;
