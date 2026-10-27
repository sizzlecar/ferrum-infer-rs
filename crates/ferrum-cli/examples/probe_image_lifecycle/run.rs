use super::*;
use ferrum_bench_core::images::ValidatedImage;
use request::{Outcome, Phase, Record};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Finding {
    Observed,
    NotObserved,
    Failed,
}
#[derive(Debug, Serialize)]
pub(super) struct Evidence {
    pub status: Finding,
    pub detail: &'static str,
}
#[derive(Debug, Serialize)]
pub(super) struct Report {
    schema_version: u32,
    client_version: &'static str,
    started_unix_ms: u64,
    configuration: Arguments,
    dataset_sha256: String,
    reference_images: Vec<ValidatedImage>,
    requests: Vec<Record>,
    health: health::Log,
    comparisons: BTreeMap<String, health::Comparison>,
    findings: BTreeMap<&'static str, Evidence>,
    pub unexpected_failures: usize,
    expected_client_cancellations: usize,
    expected_client_body_drops: usize,
    server_429_rejections: usize,
    expected_shutdown_transport_or_503: usize,
    scope: &'static str,
}

pub(super) async fn run(args: &Arguments, prepared: request::Prepared) -> anyhow::Result<Report> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .pool_max_idle_per_host(0)
        .connect_timeout(Duration::from_millis(args.health_timeout_ms))
        .build()?;
    let started_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis()
        .try_into()?;
    let origin = Instant::now();
    let log = Arc::new(Mutex::new(health::Log::default()));
    health::append(&client, args, &log, "baseline", origin).await;
    eprintln!("Lifecycle probe dispatching {} requests in {:?} mode; started_unix_ms={started_unix_ms}; the probe never stops/restarts the server.", args.requests, args.mode);
    let (stop, mut stopped) = tokio::sync::oneshot::channel();
    let observed_client = client.clone();
    let observed_args = args.clone();
    let observed_log = Arc::clone(&log);
    let observer = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(observed_args.poll_ms));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                biased;
                _ = &mut stopped => break,
                _ = interval.tick() => health::append(&observed_client, &observed_args, &observed_log, "observation", origin).await,
            }
        }
    });
    let mut pending = tokio::task::JoinSet::new();
    for (index, input) in prepared.requests.iter().cloned().enumerate() {
        let client = client.clone();
        let args = args.clone();
        pending.spawn(async move {
            request::execute(&client, &args, &input, index, Phase::Perturbation, origin).await
        });
    }
    let mut records = Vec::new();
    let mut task_failures = 0;
    while let Some(result) = pending.join_next().await {
        match result {
            Ok(record) => records.push(record),
            Err(_) => task_failures += 1,
        }
    }
    tokio::time::sleep_until((origin + Duration::from_millis(args.observation_ms)).into()).await;
    let _ = stop.send(());
    if observer.await.is_err() {
        task_failures += 1;
    }
    health::append(&client, args, &log, "after_perturbation", origin).await;
    tokio::time::sleep(Duration::from_millis(args.recovery_delay_ms)).await;
    health::append(&client, args, &log, "before_recovery", origin).await;
    for index in 0..args.recovery_requests {
        let input = &prepared.requests[index % prepared.requests.len()];
        records.push(request::execute(&client, args, input, index, Phase::Recovery, origin).await);
        health::append(&client, args, &log, "after_recovery_request", origin).await;
    }
    records.sort_by_key(|record| {
        (
            if record.phase == Phase::Perturbation {
                0
            } else {
                1
            },
            record.request_index,
        )
    });
    let health = Arc::try_unwrap(log)
        .expect("observer released health log")
        .into_inner()
        .expect("health collector unpoisoned");
    let comparisons = health::comparisons(&health);
    let (findings, unexpected_failures, expected_shutdown) =
        assess(args, &records, &health, task_failures);
    Ok(Report {
        schema_version: 1, client_version: env!("CARGO_PKG_VERSION"), started_unix_ms,
        configuration: args.clone(), dataset_sha256: prepared.dataset_sha256,
        reference_images: prepared.references,
        expected_client_cancellations: records.iter().filter(|r| r.outcome == Outcome::ClientFutureDropped).count(),
        expected_client_body_drops: records.iter().filter(|r| r.outcome == Outcome::ClientBodyDropped).count(),
        server_429_rejections: records.iter().filter(|r| r.outcome == Outcome::HttpRejected).count(),
        expected_shutdown_transport_or_503: expected_shutdown,
        requests: records, health, comparisons, findings, unexpected_failures,
        scope: "isolated loopback client probe; no service start/stop/signal, no retries, fixed dataset prefix; cancellation drops only the client HTTP future and does not establish server/backend cancellation; held headers mean the application did not poll the body, TCP/HTTP buffers may still consume bytes; drop_body records dropping that unread response, not proof of a server-observed disconnect; deliberately dropped responses cannot be PNG-validated; shared health counters do not identify individual requests, prove causality or prove a blocked write; Host totals may legitimately retain warm caches; health polling competes for server resources, gaps may miss short transitions; only independently observed health changes and validated recovery PNGs are evidence; observe_shutdown expects an externally controlled stop/restart, never infers graceful cleanup from connection refusal alone",
    })
}

fn assess(
    args: &Arguments,
    records: &[Record],
    health: &health::Log,
    task_failures: usize,
) -> (BTreeMap<&'static str, Evidence>, usize, usize) {
    let baseline = health.snapshots.first().and_then(|s| s.values.as_ref());
    let shutdown = baseline.is_some_and(|b| b.accepting)
        && health.snapshots.iter().skip(1).any(|s| {
            s.values.as_ref().is_some_and(|v| !v.accepting)
                || s.failure == Some(health::Failure::Transport)
                || s.http_status == Some(503)
        });
    let expected_shutdown = |r: &&Record| {
        args.mode == Mode::ObserveShutdown
            && shutdown
            && (r.outcome == Outcome::TransportError
                || (r.outcome == Outcome::HttpError && r.http_status == Some(503)))
    };
    let expected = records.iter().filter(expected_shutdown).count();
    let unexpected = task_failures
        + usize::from(health.omitted > 0)
        + records
            .iter()
            .filter(|r| {
                !matches!(
                    r.outcome,
                    Outcome::Validated
                        | Outcome::ClientFutureDropped
                        | Outcome::ClientBodyDropped
                        | Outcome::HttpRejected
                ) && !expected_shutdown(r)
            })
            .count()
        + health
            .snapshots
            .iter()
            .filter(|s| {
                s.failure.is_some()
                    && !(args.mode == Mode::ObserveShutdown
                        && shutdown
                        && (s.failure == Some(health::Failure::Transport)
                            || s.http_status == Some(503)))
            })
            .count();
    let mut findings = BTreeMap::new();
    let mut add = |name, observed, detail| {
        findings.insert(
            name,
            Evidence {
                status: if observed {
                    Finding::Observed
                } else {
                    Finding::NotObserved
                },
                detail,
            },
        );
    };
    add(
        "client_future_dropped",
        records
            .iter()
            .any(|r| r.outcome == Outcome::ClientFutureDropped),
        "client-only action; server cancellation and device abortion are unproven",
    );
    add("client_body_hold", records.iter().any(|r| r.body_hold_started_ms.is_some()), "complete headers received, application body polling delayed; buffering can hide server pressure");
    add("client_body_dropped", records.iter().any(|r| r.outcome == Outcome::ClientBodyDropped), "client dropped the unread response after complete headers; a server-observed disconnect and cleanup require independent observations");
    for (name, metric) in [
        ("scheduler_active", "scheduler.active_requests"),
        ("scheduler_pending", "scheduler.pending_requests"),
    ] {
        add(
            name,
            health.snapshots.iter().skip(1).any(|s| {
                s.values
                    .as_ref()
                    .and_then(|v| v.get(metric))
                    .is_some_and(|n| n > 0)
            }),
            "direct aggregate health observation, not per-request attribution",
        );
    }
    let inside_hold = |s: &health::Snapshot| {
        records.iter().any(|r| {
            r.body_hold_started_ms.is_some_and(|start| {
                s.started_ms >= start
                    && s.finished_ms <= r.body_hold_finished_ms.unwrap_or(r.finished_ms)
            })
        })
    };
    let retained = health.snapshots.iter().any(|s| {
        inside_hold(s)
            && ["output.sessions", "output.encoded_bytes"]
                .iter()
                .any(|name| {
                    baseline
                        .and_then(|b| b.get(name))
                        .zip(s.values.as_ref().and_then(|v| v.get(name)))
                        .is_some_and(|(b, current)| current > b)
                })
    });
    add("output_retained_during_client_hold", retained, "output session/encoded-byte health exceeded baseline inside a client body-hold interval; this does not by itself prove a blocked network write");
    add("inference_active_while_body_held", health.snapshots.iter().any(|s| inside_hold(s) && s.values.as_ref().and_then(|v| v.get("scheduler.active_requests")).is_some_and(|n| n > 0)), "health reported active inference inside a client body-hold interval; aggregate counters do not identify which request executed");
    add("shutdown_or_unavailability", shutdown, "acceptance stopped or health became unavailable after an accepting baseline; connection loss alone does not prove graceful shutdown");
    let recovery = records
        .iter()
        .filter(|r| r.phase == Phase::Recovery)
        .collect::<Vec<_>>();
    add("validated_recovery_requests", !recovery.is_empty() && recovery.iter().all(|r| r.outcome == Outcome::Validated), "all explicitly issued recovery requests completed with independently decoded PNGs of expected dimensions");
    let final_values = health.snapshots.last().and_then(|s| s.values.as_ref());
    let idle = final_values.is_some_and(|v| {
        v.accepting
            && [
                "scheduler.active_requests",
                "scheduler.pending_requests",
                "scheduler.preparing_requests",
            ]
            .iter()
            .all(|name| v.get(name) == Some(0))
    });
    let output_returned = baseline.zip(final_values).is_some_and(|(b, f)| {
        ["output.sessions", "output.encoded_bytes"]
            .iter()
            .all(|name| b.get(name).zip(f.get(name)).is_some_and(|(b, f)| f <= b))
    });
    add("accepting_idle_and_output_returned", idle && output_returned, "final health accepts requests, has no active/pending/preparing work, and output counters are at or below baseline; Host cache retention is compared separately");
    drop(add);
    if unexpected > 0 {
        findings.insert("unexpected_failures", Evidence { status: Finding::Failed, detail: "protocol/PNG/client/health failures or incomplete observation; expected client cancellation/body drop, HTTP 429 and externally observed shutdown transport are counted separately" });
    }
    (findings, unexpected, expected)
}
