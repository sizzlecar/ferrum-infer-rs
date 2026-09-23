//! Direct HTTP measurements reuse the agent proxy's exact SSE observer.
use super::{
    proxy::{RequestRecord, SseObserver, VisibleTextRecord},
    write_json,
};
use anyhow::{ensure, Context, Result};
use axum::body::Bytes;
use ferrum_bench_core::{
    BENCHMARK_CELL_ID_HEADER, BENCHMARK_PHASE_HEADER, BENCHMARK_REPEAT_INDEX_HEADER,
    BENCHMARK_REQUEST_INDEX_HEADER, BENCHMARK_RUN_ID_HEADER,
};
use futures::StreamExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::Barrier;

#[path = "http_replay/text_timing.rs"]
mod text_timing;

#[derive(clap::Args)]
pub(crate) struct Args {
    /// OpenAI base URL, including /v1, on this machine.
    #[arg(long)]
    pub base_url: String,
    /// One body copied to every request, or one body per request, sent byte-for-byte.
    #[arg(long, required = true)]
    pub body_file: Vec<PathBuf>,
    /// Number of requests released together in this single wave.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub concurrency: u32,
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(1..))]
    pub timeout_secs: u64,
    /// Must not exist, even as an empty directory.
    #[arg(long)]
    pub report_dir: PathBuf,
}

fn at(clock: Instant) -> u64 {
    clock.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

#[derive(Clone)]
struct FrozenBody {
    source: PathBuf,
    bytes: Bytes,
    sha256: String,
    single_choice_requested: bool,
}

pub(crate) async fn run(args: &Args) -> Result<i32> {
    ensure!(
        args.concurrency > 0 && args.timeout_secs > 0,
        "concurrency and timeout must be positive"
    );
    let url = reqwest::Url::parse(&args.base_url)?;
    ensure!(
        url.scheme() == "http"
            && matches!(
                url.host_str(),
                Some("127.0.0.1" | "localhost" | "[::1]" | "::1")
            ),
        "base_url must use HTTP loopback"
    );
    ensure!(
        url.query().is_none() && url.fragment().is_none(),
        "base_url cannot contain a query or fragment"
    );
    ensure!(
        args.body_file.len() == 1 || args.body_file.len() == args.concurrency as usize,
        "supply one body file or exactly one per concurrent request"
    );
    let inputs = args
        .body_file
        .iter()
        .map(|path| -> Result<FrozenBody> {
            let source = path.canonicalize()?;
            let bytes = Bytes::from(fs::read(&source)?);
            let parsed: Value =
                serde_json::from_slice(&bytes).context("parse OpenAI request body")?;
            ensure!(
                parsed["stream"] == true,
                "body must request stream=true; it is never rewritten"
            );
            ensure!(
                parsed["model"].as_str().is_some_and(|s| !s.is_empty()),
                "body lacks a model"
            );
            ensure!(parsed["messages"].is_array(), "body lacks messages");
            let sha256 = format!("{:x}", Sha256::digest(&bytes));
            Ok(FrozenBody {
                source,
                bytes,
                sha256,
                single_choice_requested: parsed.get("n").is_none_or(|n| n.as_u64() == Some(1)),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let bodies = if inputs.len() == 1 {
        vec![inputs[0].clone(); args.concurrency as usize]
    } else {
        inputs
    };
    let endpoint = format!("{}/chat/completions", args.base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(args.timeout_secs))
        .build()?;
    // Exclusive directory creation also rejects an existing empty directory.
    fs::create_dir(&args.report_dir).context("create new replay report directory")?;
    for (index, body) in bodies.iter().enumerate() {
        fs::write(
            args.report_dir.join(format!("request-{index}.body.json")),
            &body.bytes,
        )?;
    }
    let wall_start = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run_id = format!("http-replay-{wall_start}");
    write_json(
        args.report_dir.join("input.json"),
        &json!({
            "run_id": run_id, "base_url": args.base_url,
            "bodies": bodies.iter().enumerate().map(|(index, b)| json!({
                "request_index": index, "body_file": b.source, "body_sha256": b.sha256, "body_bytes": b.bytes.len()
            })).collect::<Vec<_>>(),
            "concurrency": args.concurrency, "timeout_secs": args.timeout_secs,
            "request_mutations": [], "scope": "one wave of bound raw HTTP requests; no agent or tool execution"
        }),
    )?;
    // Finish fallible file setup before any participant waits on the barrier.
    let outputs = (0..args.concurrency)
        .map(|index| {
            File::create(args.report_dir.join(format!("response-{index}.sse"))).map(BufWriter::new)
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    let clock = Instant::now();
    let barrier = Arc::new(Barrier::new(args.concurrency as usize));
    let records = futures::future::join_all(outputs.into_iter().enumerate().map(|(index, raw)| {
        capture(
            &client,
            &endpoint,
            &run_id,
            bodies[index].bytes.clone(),
            bodies[index].single_choice_requested,
            index as u32,
            raw,
            clock,
            barrier.clone(),
        )
    }))
    .await;
    for record in &records {
        let mut bound = serde_json::to_value(record)?;
        bound["body_sha256"] = json!(bodies[record.request_index as usize].sha256);
        write_json(
            args.report_dir
                .join(format!("request-{}.json", record.request_index)),
            &bound,
        )?;
    }
    let transport_complete = records
        .iter()
        .all(|r| r.http_status == Some(200) && r.saw_done && r.error.is_none());
    let usage_complete = records.iter().all(|r| {
        r.usage.as_ref().is_some_and(|u| {
            u["prompt_tokens"].as_u64().is_some() && u["completion_tokens"].as_u64().is_some()
        })
    });
    let start = records.iter().map(|r| r.submitted_ns).min().unwrap();
    let end = records.iter().filter_map(|r| r.ended_ns).max().unwrap();
    let total_output = usage_complete.then(|| {
        records
            .iter()
            .map(|r| {
                r.usage.as_ref().unwrap()["completion_tokens"]
                    .as_u64()
                    .unwrap()
            })
            .sum::<u64>()
    });
    let successful: Vec<_> = records
        .iter()
        .filter(|r| r.http_status == Some(200) && r.saw_done && r.error.is_none())
        .collect();
    let successful_output = successful.iter().try_fold(0_u64, |total, r| {
        total.checked_add(r.usage.as_ref()?["completion_tokens"].as_u64()?)
    });
    let summary: Vec<_> = records.iter().map(|r| json!({
        "request_index": r.request_index, "request_id": r.server_request_id,
        "body_sha256": bodies[r.request_index as usize].sha256,
        "submitted_ns": r.submitted_ns, "ended_ns": r.ended_ns,
        "elapsed_ns": r.ended_ns.map(|end| end.saturating_sub(r.submitted_ns)),
        "ttft_ns": r.progress_ns.first().map(|first| first.saturating_sub(r.submitted_ns)),
        "first_sse_latency_ns": r.first_sse_ns.map(|first| first.saturating_sub(r.submitted_ns)),
        "progress_events": r.progress_ns.len(), "usage": r.usage,
        "visible_text": text_timing::measure(r),
        "http_status": r.http_status, "saw_done": r.saw_done,
        "finish_reasons": r.finish_reasons, "error": r.error
    })).collect();
    let report = json!({
        "run_id": run_id, "concurrency": args.concurrency,
        "transport_complete": transport_complete, "usage_complete": usage_complete,
        "elapsed_ns": end.saturating_sub(start), "total_output_tokens": total_output,
        "output_tokens_per_second_including_prefill": total_output.filter(|_| end > start).map(|n| n as f64 * 1e9 / (end - start) as f64),
        "successful_request_count": successful.len(),
        "successful_output_tokens": successful_output,
        "successful_output_tokens_per_second_including_prefill": successful_output.filter(|_| end > start).map(|n| n as f64 * 1e9 / (end - start) as f64),
        "requests": summary,
        "visible_text": text_timing::summarize(&records),
        "measurement_notes": [
            "TTFT is first nonempty content, reasoning or tool name/argument progress; a role-only SSE frame is not progress.",
            "visible_text reports separate first-choice content/reasoning text TTFT, last-visible TPOT using usage tokens, and pooled SSE text-event ITL. Tool-only, role-only, empty and finish-only events are excluded.",
            "Visible text gaps retain stalls, transport-coalesced events and usage/event mismatches. Strict single-token timing is not established by this replay.",
            "SSE progress events are not tokens. Output counts come only from server usage; missing usage remains unavailable.",
            "The original throughput field includes all reported usage. Successful throughput includes only completed error-free responses, over the same full-wave wall interval.",
            "Client latency includes transport, queueing and inference. Concurrent HTTP streams do not prove simultaneous GPU execution.",
            "Prefix-cache state and server configuration are controlled externally; this command does not warm, clear or reset caches.",
            "Finish reasons, including length, are retained. Transport completion is not semantic task validation."
        ]
    });
    write_json(args.report_dir.join("report.json"), &report)?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(if transport_complete && usage_complete {
        0
    } else {
        1
    })
}

#[allow(clippy::too_many_arguments)]
async fn capture(
    client: &reqwest::Client,
    endpoint: &str,
    run_id: &str,
    body: Bytes,
    single_choice_requested: bool,
    index: u32,
    mut raw: BufWriter<File>,
    clock: Instant,
    barrier: Arc<Barrier>,
) -> RequestRecord {
    barrier.wait().await;
    let mut record = RequestRecord {
        task_id: "raw-body".into(),
        request_index: index,
        submitted_ns: at(clock),
        visible_text: Some(VisibleTextRecord {
            single_choice_requested: Some(single_choice_requested),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut observer = SseObserver::default();
    let result: Result<()> = async {
        let response = client
            .post(endpoint)
            .header("content-type", "application/json")
            .header(BENCHMARK_RUN_ID_HEADER, run_id)
            .header(BENCHMARK_CELL_ID_HEADER, "raw-body")
            .header(BENCHMARK_REPEAT_INDEX_HEADER, "0")
            .header(BENCHMARK_PHASE_HEADER, "measured")
            .header(BENCHMARK_REQUEST_INDEX_HEADER, index.to_string())
            .body(body)
            .send()
            .await?;
        record.response_headers_ns = Some(at(clock));
        record.http_status = Some(response.status().as_u16());
        if !response.status().is_success() {
            record.error = Some(format!("HTTP {}", response.status()));
        }
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            let received_ns = at(clock);
            raw.write_all(&bytes)?;
            observer.push(&bytes, received_ns, &mut record)?;
        }
        Ok(())
    }
    .await;
    record.ended_ns = Some(at(clock));
    if let Err(error) = result {
        record.error.get_or_insert_with(|| format!("{error:#}"));
    }
    if let Err(error) = raw.flush() {
        record
            .error
            .get_or_insert_with(|| format!("save SSE: {error}"));
    }
    if !record.saw_done {
        record
            .error
            .get_or_insert_with(|| "stream ended without [DONE]".into());
    }
    record
}

#[cfg(test)]
#[path = "http_replay/tests.rs"]
mod tests;
