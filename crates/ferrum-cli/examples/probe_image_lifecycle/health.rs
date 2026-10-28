use super::*;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

const NUMBERS: &[(&str, &str)] = &[
    (
        "scheduler.active_requests",
        "/images/scheduler/active_requests",
    ),
    (
        "scheduler.pending_requests",
        "/images/scheduler/pending_requests",
    ),
    (
        "scheduler.preparing_requests",
        "/images/scheduler/preparing_requests",
    ),
    ("scheduler.pending_bytes", "/images/scheduler/pending_bytes"),
    (
        "scheduler.rejected_requests",
        "/images/scheduler/rejected_requests",
    ),
    (
        "scheduler.executed_batches",
        "/images/scheduler/executed_batches",
    ),
    (
        "scheduler.executed_requests",
        "/images/scheduler/executed_requests",
    ),
    (
        "scheduler.last_batch_size",
        "/images/scheduler/last_batch_size",
    ),
    (
        "scheduler.capacity_reductions",
        "/images/scheduler/capacity_reductions",
    ),
    (
        "scheduler.maximum_pending",
        "/images/scheduler/maximum_pending",
    ),
    (
        "scheduler.maximum_pending_bytes",
        "/images/scheduler/maximum_pending_bytes",
    ),
    ("scheduler.max_num_seqs", "/images/scheduler/max_num_seqs"),
    ("output.sessions", "/images/output/sessions"),
    ("output.encoded_bytes", "/images/output/encoded_bytes"),
    ("host.committed_bytes", "/images/host/committed_bytes"),
    ("host.allocated_bytes", "/images/host/allocated_bytes"),
];
#[derive(Debug, Clone, Serialize)]
pub(super) struct Values {
    pub accepting: bool,
    pub numbers: BTreeMap<String, u64>,
}
impl Values {
    pub(super) fn get(&self, name: &str) -> Option<u64> {
        self.numbers.get(name).copied()
    }
}
pub(super) fn parse(bytes: &[u8]) -> Result<Values, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| "health JSON is invalid".to_string())?;
    let accepting = value
        .pointer("/images/scheduler/accepting")
        .and_then(|v| v.as_bool())
        .ok_or_else(|| "health lacks images.scheduler.accepting".to_string())?;
    let mut numbers = BTreeMap::new();
    for &(name, path) in NUMBERS {
        let number = value
            .pointer(path)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("health lacks unsigned {name}"))?;
        numbers.insert(name.into(), number);
    }
    Ok(Values { accepting, numbers })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Failure {
    Transport,
    HttpStatus,
    InvalidResponse,
}
#[derive(Debug, Clone, Serialize)]
pub(super) struct Snapshot {
    pub phase: &'static str,
    pub started_ms: f64,
    pub finished_ms: f64,
    pub http_status: Option<u16>,
    pub values: Option<Values>,
    pub failure: Option<Failure>,
    pub error: Option<String>,
}

pub(super) async fn snapshot(
    client: &reqwest::Client,
    args: &Arguments,
    phase: &'static str,
    origin: Instant,
) -> Snapshot {
    let mut result = Snapshot {
        phase,
        started_ms: origin.elapsed().as_secs_f64() * 1000.0,
        finished_ms: 0.0,
        http_status: None,
        values: None,
        failure: None,
        error: None,
    };
    let operation = async {
        let response = client
            .get(&args.health_endpoint)
            .send()
            .await
            .map_err(|e| (Failure::Transport, e.to_string()))?;
        result.http_status = Some(response.status().as_u16());
        if !response.status().is_success() {
            return Err((
                Failure::HttpStatus,
                format!("health HTTP {}", response.status()),
            ));
        }
        let body = request::health_body(response, args.maximum_health_bytes)
            .await
            .map_err(|(outcome, error)| {
                let failure = if outcome == request::Outcome::TransportError {
                    Failure::Transport
                } else {
                    Failure::InvalidResponse
                };
                (failure, error)
            })?;
        parse(&body).map_err(|e| (Failure::InvalidResponse, e))
    };
    let observed = match tokio::time::timeout(
        Duration::from_millis(args.health_timeout_ms),
        operation,
    )
    .await
    {
        Ok(value) => value,
        Err(_) => Err((Failure::Transport, "health request deadline expired".into())),
    };
    match observed {
        Ok(values) => result.values = Some(values),
        Err((failure, error)) => {
            result.failure = Some(failure);
            result.error = Some(error.chars().take(512).collect());
        }
    }
    result.finished_ms = origin.elapsed().as_secs_f64() * 1000.0;
    result
}

#[derive(Debug, Default, Serialize)]
pub(super) struct Log {
    pub snapshots: Vec<Snapshot>,
    pub omitted: u64,
}
impl Log {
    pub(super) fn push(&mut self, value: Snapshot, limit: usize) {
        if self.snapshots.len() < limit {
            self.snapshots.push(value);
        } else {
            self.omitted += 1;
        }
    }
}
pub(super) type SharedLog = Arc<Mutex<Log>>;

pub(super) async fn append(
    client: &reqwest::Client,
    args: &Arguments,
    log: &SharedLog,
    phase: &'static str,
    origin: Instant,
) {
    let value = snapshot(client, args, phase, origin).await;
    log.lock()
        .expect("health collector is unpoisoned")
        .push(value, args.maximum_health_snapshots);
}

#[derive(Debug, Serialize)]
pub(super) struct Comparison {
    pub baseline: Option<u64>,
    pub maximum_observed: Option<u64>,
    pub final_observed: Option<u64>,
    pub final_minus_baseline: Option<i128>,
}
pub(super) fn comparisons(log: &Log) -> BTreeMap<String, Comparison> {
    NUMBERS
        .iter()
        .map(|&(name, _)| {
            let baseline = log
                .snapshots
                .first()
                .and_then(|s| s.values.as_ref())
                .and_then(|v| v.get(name));
            let final_observed = log
                .snapshots
                .last()
                .and_then(|s| s.values.as_ref())
                .and_then(|v| v.get(name));
            let maximum_observed = log
                .snapshots
                .iter()
                .filter_map(|s| s.values.as_ref().and_then(|v| v.get(name)))
                .max();
            (
                name.into(),
                Comparison {
                    baseline,
                    maximum_observed,
                    final_observed,
                    final_minus_baseline: baseline
                        .zip(final_observed)
                        .map(|(b, f)| i128::from(f) - i128::from(b)),
                },
            )
        })
        .collect()
}
