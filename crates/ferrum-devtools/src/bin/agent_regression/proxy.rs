use super::{config::Server, write_json};
use anyhow::{Context, Result};
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::Response,
    routing::post,
    Router,
};
use ferrum_bench_core::{
    BENCHMARK_CELL_ID_HEADER, BENCHMARK_PHASE_HEADER, BENCHMARK_REPEAT_INDEX_HEADER,
    BENCHMARK_REQUEST_INDEX_HEADER, BENCHMARK_RUN_ID_HEADER,
};
use futures::StreamExt;
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::{
    net::TcpListener,
    sync::{mpsc, Notify},
    task::JoinHandle,
};
use tokio_stream::wrappers::ReceiverStream;

pub(crate) const TASK_HEADER: &str = "x-ferrum-agent-task";

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct RequestRecord {
    pub task_id: String,
    pub request_index: u32,
    pub submitted_ns: u64,
    pub response_headers_ns: Option<u64>,
    pub first_sse_ns: Option<u64>,
    /// Nonempty content, reasoning or tool-argument/name fragments, not role frames.
    pub progress_ns: Vec<u64>,
    pub ended_ns: Option<u64>,
    pub server_request_id: Option<String>,
    pub http_status: Option<u16>,
    pub saw_done: bool,
    pub finish_reasons: Vec<String>,
    pub usage: Option<Value>,
    pub error: Option<String>,
    pub tool_result_ids: Vec<String>,
    /// Full original wire bodies are saved separately; retain messages in memory
    /// to prove which assistant turn each returned tool result belongs to.
    #[serde(skip)]
    pub messages: Vec<Value>,
}

pub(crate) struct Proxy {
    pub base_url: String,
    pub state: Arc<ProxyState>,
    server_task: JoinHandle<()>,
}
impl Drop for Proxy {
    fn drop(&mut self) {
        self.server_task.abort();
    }
}

pub(crate) struct ProxyState {
    client: reqwest::Client,
    upstream: Server,
    run_id: String,
    tasks: BTreeSet<String>,
    output: PathBuf,
    clock: Instant,
    next_request: AtomicU32,
    active: AtomicU32,
    finished: Notify,
    pub records: Mutex<Vec<RequestRecord>>,
}

struct FlightRecord {
    state: Arc<ProxyState>,
    record: RequestRecord,
    saved: bool,
}
impl FlightRecord {
    fn new(state: Arc<ProxyState>, record: RequestRecord) -> Self {
        state.active.fetch_add(1, Ordering::AcqRel);
        Self {
            state,
            record,
            saved: false,
        }
    }
    fn finish(&mut self) {
        if self.saved {
            return;
        }
        self.record.ended_ns.get_or_insert_with(|| self.state.at());
        self.state.save(&self.record);
        self.saved = true;
        self.state.active.fetch_sub(1, Ordering::AcqRel);
        self.state.finished.notify_one();
    }
}
impl std::ops::Deref for FlightRecord {
    type Target = RequestRecord;
    fn deref(&self) -> &RequestRecord {
        &self.record
    }
}
impl std::ops::DerefMut for FlightRecord {
    fn deref_mut(&mut self) -> &mut RequestRecord {
        &mut self.record
    }
}
impl Drop for FlightRecord {
    fn drop(&mut self) {
        if !self.saved {
            self.record
                .error
                .get_or_insert_with(|| "request handler ended before terminal evidence".into());
            self.finish();
        }
    }
}
impl ProxyState {
    fn at(&self) -> u64 {
        self.clock.elapsed().as_nanos().min(u64::MAX as u128) as u64
    }
    fn save(&self, record: &RequestRecord) {
        let mut record = record.clone();
        if let Err(error) = write_json(
            self.output
                .join(format!("{}-{}.json", record.task_id, record.request_index)),
            &record,
        ) {
            eprintln!("write request evidence: {error:#}");
            record.error = Some(format!("write request evidence: {error:#}"));
        }
        self.records.lock().expect("request records").push(record);
    }
}

impl Proxy {
    pub async fn start(
        server: &Server,
        run_id: &str,
        tasks: BTreeSet<String>,
        output: PathBuf,
        clock: Instant,
    ) -> Result<Self> {
        fs::create_dir_all(&output)?;
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let base_url = format!("http://{}/v1", listener.local_addr()?);
        let state = Arc::new(ProxyState {
            client: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(server.request_timeout_secs))
                .build()?,
            upstream: server.clone(),
            run_id: run_id.into(),
            tasks,
            output,
            clock,
            next_request: AtomicU32::new(0),
            active: AtomicU32::new(0),
            finished: Notify::new(),
            records: Mutex::new(Vec::new()),
        });
        let app = Router::new()
            .route("/v1/chat/completions", post(forward))
            .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
            .with_state(state.clone());
        let server_task = tokio::spawn(async move {
            if let Err(error) = axum::serve(listener, app).await {
                eprintln!("agent proxy: {error}");
            }
        });
        Ok(Self {
            base_url,
            state,
            server_task,
        })
    }

    pub async fn drain(&self, timeout_secs: u64) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(timeout_secs), async {
            while self.state.active.load(Ordering::Acquire) != 0 {
                self.state.finished.notified().await;
            }
        })
        .await
        .context("proxy requests did not reach terminal evidence within request budget")?;
        let records = self.state.records.lock().expect("request records");
        let issued = self.state.next_request.load(Ordering::Acquire) as usize;
        let unique: BTreeSet<_> = records.iter().map(|r| r.request_index).collect();
        anyhow::ensure!(
            records.len() == issued && unique.len() == issued,
            "issued request lacks unique terminal evidence"
        );
        Ok(())
    }
}

async fn forward(
    State(state): State<Arc<ProxyState>>,
    headers: HeaderMap,
    body: Bytes,
) -> std::result::Result<Response, (StatusCode, String)> {
    let bad = |message: &str| (StatusCode::BAD_REQUEST, message.to_owned());
    let task = headers
        .get(TASK_HEADER)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| bad("missing task identity"))?;
    if !state.tasks.contains(task) {
        return Err(bad("unknown task identity"));
    }
    let request: Value = serde_json::from_slice(&body).map_err(|_| bad("invalid JSON body"))?;
    if request["model"] != state.upstream.model {
        return Err(bad("unexpected inference model"));
    }
    if request["stream"] != true {
        return Err(bad("pi evidence requires streaming"));
    }
    let index = state.next_request.fetch_add(1, Ordering::Relaxed);
    let mut record = FlightRecord::new(
        state.clone(),
        RequestRecord {
            task_id: task.to_owned(),
            request_index: index,
            submitted_ns: state.at(),
            messages: request["messages"].as_array().cloned().unwrap_or_default(),
            tool_result_ids: request["messages"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|m| m["role"] == "tool")
                .filter_map(|m| m["tool_call_id"].as_str().map(str::to_owned))
                .collect(),
            ..Default::default()
        },
    );
    let request_path = state.output.join(format!("{task}-{index}.request.json"));
    if let Err(error) = fs::write(&request_path, &body) {
        record.error = Some(format!("request evidence: {error}"));
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("request evidence: {error}"),
        ));
    }
    let upstream = state
        .client
        .post(format!(
            "{}/chat/completions",
            state.upstream.base_url.trim_end_matches('/')
        ))
        .header("content-type", "application/json")
        .header(BENCHMARK_RUN_ID_HEADER, &state.run_id)
        .header(BENCHMARK_CELL_ID_HEADER, task)
        .header(BENCHMARK_REPEAT_INDEX_HEADER, "0")
        .header(BENCHMARK_PHASE_HEADER, "measured")
        .header(BENCHMARK_REQUEST_INDEX_HEADER, index.to_string())
        .body(body)
        .send()
        .await;
    let upstream = match upstream {
        Ok(response) => response,
        Err(error) => {
            record.error = Some(format!("{error:?}"));
            record.ended_ns = Some(state.at());
            record.finish();
            return Err((
                StatusCode::BAD_GATEWAY,
                "local Ferrum request failed; see evidence".into(),
            ));
        }
    };
    record.response_headers_ns = Some(state.at());
    record.http_status = Some(upstream.status().as_u16());
    let status = upstream.status();
    let content_type = upstream.headers().get("content-type").cloned();
    let raw_path = state.output.join(format!("{task}-{index}.response.sse"));
    let raw_file = File::create(raw_path)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    let (sender, receiver) = mpsc::channel::<std::result::Result<Bytes, std::io::Error>>(8);
    tokio::spawn(async move {
        let mut raw = BufWriter::new(raw_file);
        let mut stream = upstream.bytes_stream();
        let mut observer = SseObserver::default();
        loop {
            let chunk = tokio::select! {
                _=sender.closed()=>{
                    if !record.saw_done {record.error=Some("pi disconnected before stream completed".into());}
                    break;
                },
                chunk=stream.next()=>chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
            match chunk {
                Ok(bytes) => {
                    if let Err(error) = raw
                        .write_all(&bytes)
                        .and_then(|_| observer.push(&bytes, state.at(), &mut record))
                    {
                        record.error = Some(format!("response evidence: {error}"));
                        let _ = sender.send(Err(error)).await;
                        break;
                    }
                    if sender.send(Ok(bytes)).await.is_err() {
                        if !record.saw_done {
                            record.error = Some("pi disconnected before stream completed".into());
                        }
                        break;
                    }
                }
                Err(error) => {
                    record.error = Some(error.to_string());
                    let _ = sender
                        .send(Err(std::io::Error::other(error.to_string())))
                        .await;
                    break;
                }
            }
        }
        if let Err(error) = raw.flush() {
            record.error = Some(error.to_string());
        }
        if !status.is_success() {
            record.error.get_or_insert_with(|| format!("HTTP {status}"));
        }
        if !record.saw_done {
            record
                .error
                .get_or_insert_with(|| "stream ended without [DONE]".into());
        }
        record.ended_ns = Some(state.at());
        record.finish();
    });
    let mut response = Response::builder().status(status);
    if let Some(content_type) = content_type {
        response = response.header("content-type", content_type);
    }
    response
        .body(Body::from_stream(ReceiverStream::new(receiver)))
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))
}

#[derive(Default)]
struct SseObserver {
    buffer: Vec<u8>,
    data: Vec<u8>,
}
impl SseObserver {
    fn push(&mut self, bytes: &[u8], at: u64, record: &mut RequestRecord) -> std::io::Result<()> {
        self.buffer.extend_from_slice(bytes);
        if self.buffer.len() > 32 * 1024 * 1024 {
            return Err(std::io::Error::other("unterminated oversized SSE record"));
        }
        while let Some(end) = self.buffer.iter().position(|b| *b == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=end).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() {
                self.dispatch(at, record)?;
            } else if let Some(data) = line.strip_prefix(b"data:") {
                if !self.data.is_empty() {
                    self.data.push(b'\n');
                }
                self.data
                    .extend_from_slice(data.strip_prefix(b" ").unwrap_or(data));
            }
        }
        Ok(())
    }
    fn dispatch(&mut self, at: u64, record: &mut RequestRecord) -> std::io::Result<()> {
        if self.data.is_empty() {
            return Ok(());
        }
        record.first_sse_ns.get_or_insert(at);
        let data = std::mem::take(&mut self.data);
        if data == b"[DONE]" {
            record.saw_done = true;
            return Ok(());
        }
        let event: Value = serde_json::from_slice(&data).map_err(std::io::Error::other)?;
        if let Some(id) = event["id"].as_str() {
            if record
                .server_request_id
                .as_deref()
                .is_some_and(|previous| previous != id)
            {
                return Err(std::io::Error::other(
                    "response request id changed midstream",
                ));
            }
            record.server_request_id = Some(id.to_owned());
        }
        if !event["error"].is_null() {
            record.error = Some(event["error"].to_string());
        }
        if event["usage"].is_object() {
            record.usage = Some(event["usage"].clone());
        }
        let mut progress = false;
        for choice in event["choices"].as_array().into_iter().flatten() {
            if let Some(reason) = choice["finish_reason"].as_str() {
                record.finish_reasons.push(reason.into());
            }
            let delta = &choice["delta"];
            progress |= ["content", "reasoning", "reasoning_content"]
                .iter()
                .any(|key| delta[key].as_str().is_some_and(|s| !s.is_empty()));
            progress |= delta["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|tool| {
                    ["name", "arguments"].iter().any(|key| {
                        tool["function"][key]
                            .as_str()
                            .is_some_and(|s| !s.is_empty())
                    })
                });
        }
        if progress {
            record.progress_ns.push(at);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProgressSpan {
    pub task_id: String,
    pub request_id: String,
    pub timestamps_ns: Vec<u64>,
    pub clock_error_ns: u64,
}

/// Find a real common generation interval, not overlapping HTTP queue times.
pub(crate) fn overlap(
    tasks: &BTreeSet<String>,
    spans: &[ProgressSpan],
) -> Option<Vec<ProgressSpan>> {
    for at in spans.iter().filter_map(|s| {
        s.timestamps_ns
            .first()
            .map(|first| first.saturating_add(s.clock_error_ns))
    }) {
        let mut chosen = Vec::new();
        for task in tasks {
            let span = spans
                .iter()
                .filter(|s| {
                    &s.task_id == task
                        && s.timestamps_ns.len() >= 2
                        && s.timestamps_ns[0].saturating_add(s.clock_error_ns) <= at
                        && s.timestamps_ns
                            .last()
                            .unwrap()
                            .saturating_sub(s.clock_error_ns)
                            > at
                })
                .max_by_key(|s| s.timestamps_ns.last().copied());
            if let Some(span) = span {
                chosen.push(span.clone());
            } else {
                break;
            }
        }
        if chosen.len() != tasks.len() {
            continue;
        }
        let end = chosen
            .iter()
            .map(|s| {
                s.timestamps_ns
                    .last()
                    .unwrap()
                    .saturating_sub(s.clock_error_ns)
            })
            .min()?;
        if chosen.iter().all(|s| {
            s.timestamps_ns.iter().any(|t| {
                t.saturating_sub(s.clock_error_ns) >= at
                    && t.saturating_add(s.clock_error_ns) <= end
            })
        }) {
            return Some(chosen);
        }
    }
    None
}

pub(crate) fn engine_spans(
    path: &std::path::Path,
    requests: &[RequestRecord],
) -> Result<Vec<ProgressSpan>> {
    use std::io::{BufRead, BufReader};
    let mut ids = std::collections::BTreeMap::new();
    for request in requests {
        if let Some(id) = &request.server_request_id {
            anyhow::ensure!(
                ids.insert(id.clone(), request.task_id.clone()).is_none(),
                "server reused a request id; profile correlation would be ambiguous"
            );
        }
    }
    let mut spans = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let event: Value = serde_json::from_str(&line?).context("parse Ferrum profile JSONL")?;
        let Some(id) = event["correlation_id"].as_str() else {
            continue;
        };
        let Some(task_id) = ids.get(id) else {
            continue;
        };
        let a = &event["attributes"];
        let (Some(anchor), Some(commits)) = (
            a["engine_token_wall_anchor_unix_nanos"].as_u64(),
            a["engine_token_commit_nanos_since_request_start"].as_array(),
        ) else {
            continue;
        };
        let timestamps_ns: Vec<u64> = commits
            .iter()
            .map(|t| {
                anchor
                    .checked_add(t.as_u64().context("invalid engine token timestamp")?)
                    .context("engine token timestamp overflow")
            })
            .collect::<Result<_>>()?;
        if timestamps_ns.windows(2).any(|w| w[0] > w[1]) {
            anyhow::bail!("nonmonotonic engine token evidence");
        }
        spans.push(ProgressSpan {
            task_id: task_id.clone(),
            request_id: id.into(),
            timestamps_ns,
            clock_error_ns: a["clock_conversion_max_error_nanos"]
                .as_u64()
                .context("engine timing lacks clock uncertainty")?,
        });
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn engine_timing_requires_unambiguous_ids_and_explicit_clock_error() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut event = json!({"correlation_id":"actual-id","attributes":{
            "engine_token_wall_anchor_unix_nanos":100,
            "engine_token_commit_nanos_since_request_start":[1,4,9],
            "clock_conversion_max_error_nanos":2}});
        let request = RequestRecord {
            task_id: "a".into(),
            server_request_id: Some("actual-id".into()),
            ..Default::default()
        };
        fs::write(file.path(), event.to_string()).unwrap();
        let spans = engine_spans(file.path(), std::slice::from_ref(&request)).unwrap();
        assert_eq!(spans[0].timestamps_ns, vec![101, 104, 109]);
        assert_eq!(spans[0].clock_error_ns, 2);
        assert!(engine_spans(file.path(), &[request.clone(), request.clone()]).is_err());
        event["attributes"]
            .as_object_mut()
            .unwrap()
            .remove("clock_conversion_max_error_nanos");
        fs::write(file.path(), event.to_string()).unwrap();
        assert!(engine_spans(file.path(), &[request]).is_err());
    }

    #[tokio::test]
    async fn forwards_unchanged_local_requests_and_records_response_identity() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let received = Arc::new(Mutex::new(None));
        let capture = received.clone();
        let payload = concat!(
            "data: {\"id\":\"req-actual\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
            "data: {\"id\":\"req-actual\",\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}],\"usage\":{\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n");
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap, body: Bytes| {
                let capture = capture.clone();
                async move {
                    *capture.lock().unwrap() = Some((headers, body));
                    ([("content-type", "text/event-stream")], payload)
                }
            }),
        );
        let upstream = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let output = tempfile::tempdir().unwrap();
        let server = Server {
            base_url,
            model: "local-model".into(),
            context_window: 4096,
            max_tokens: 512,
            request_timeout_secs: 10,
            reasoning: false,
            thinking: "off".into(),
            sampling_params: Default::default(),
        };
        let proxy = Proxy::start(
            &server,
            "local-test",
            BTreeSet::from(["task-a".into()]),
            output.path().to_owned(),
            Instant::now(),
        )
        .await
        .unwrap();
        let body = json!({"model":"local-model","stream":true,"messages":[{"role":"tool","tool_call_id":"call-a","content":"done"}],"temperature":0.3});
        let wire = serde_json::to_vec(&body).unwrap();
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let response = client
            .post(format!("{}/chat/completions", proxy.base_url))
            .header(TASK_HEADER, "task-a")
            .header("content-type", "application/json")
            .body(wire.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.text().await.unwrap(), payload);
        proxy.drain(10).await.unwrap();
        let (headers, actual) = received.lock().unwrap().take().unwrap();
        assert_eq!(actual.as_ref(), wire.as_slice());
        assert_eq!(headers[BENCHMARK_CELL_ID_HEADER], "task-a");
        assert_eq!(headers[BENCHMARK_RUN_ID_HEADER], "local-test");
        assert_eq!(headers[BENCHMARK_REQUEST_INDEX_HEADER], "0");
        let records = proxy.state.records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].server_request_id.as_deref(), Some("req-actual"));
        assert_eq!(records[0].usage.as_ref().unwrap()["completion_tokens"], 1);
        assert_eq!(records[0].tool_result_ids, vec!["call-a"]);
        assert!(records[0].saw_done && records[0].error.is_none());
        drop(records);
        // Even cancellation before a response must leave one terminal record;
        // aggregation waits for its guard rather than racing a final save.
        let index = proxy.state.next_request.fetch_add(1, Ordering::Relaxed);
        let guard = FlightRecord::new(
            proxy.state.clone(),
            RequestRecord {
                request_index: index,
                task_id: "task-a".into(),
                ..Default::default()
            },
        );
        let finishing = tokio::spawn(async move {
            tokio::task::yield_now().await;
            drop(guard);
        });
        proxy.drain(10).await.unwrap();
        finishing.await.unwrap();
        let records = proxy.state.records.lock().unwrap();
        assert_eq!(records.len(), 2);
        assert!(records
            .iter()
            .find(|r| r.request_index == index)
            .unwrap()
            .error
            .is_some());
        drop(records);
        upstream.abort();
    }

    #[test]
    fn clock_uncertainty_cannot_invent_an_overlap() {
        let tasks = BTreeSet::from(["a".into(), "b".into()]);
        let mut a = span("a", &[10, 20, 30]);
        let mut b = span("b", &[20, 30, 40]);
        a.clock_error_ns = 10;
        b.clock_error_ns = 10;
        assert!(overlap(&tasks, &[a, b]).is_none());
    }
    #[test]
    fn sse_fragmentation_and_role_only_frames_do_not_invent_generation() {
        let mut parser = SseObserver::default();
        let mut r = RequestRecord::default();
        parser
            .push(
                b"data: {\"id\":\"r\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\r\n\r\n",
                1,
                &mut r,
            )
            .unwrap();
        assert!(r.progress_ns.is_empty());
        let data = "data: {\"id\":\"r\",\"choices\":[{\"delta\":{\"content\":\"你好\"}}]}\n\ndata: [DONE]\n\n";
        for byte in data.as_bytes() {
            parser.push(&[*byte], 2, &mut r).unwrap();
        }
        assert_eq!(r.progress_ns, vec![2]);
        assert!(r.saw_done);
    }
    fn span(task: &str, times: &[u64]) -> ProgressSpan {
        ProgressSpan {
            task_id: task.into(),
            request_id: task.into(),
            timestamps_ns: times.into(),
            clock_error_ns: 0,
        }
    }
    #[test]
    fn queued_serial_responses_are_not_concurrent_but_interleaving_is() {
        let tasks = BTreeSet::from(["a".into(), "b".into(), "c".into()]);
        assert!(overlap(
            &tasks,
            &[
                span("a", &[1, 2, 3]),
                span("b", &[4, 5, 6]),
                span("c", &[7, 8, 9])
            ]
        )
        .is_none());
        assert!(overlap(
            &tasks,
            &[
                span("a", &[1, 5, 9]),
                span("b", &[2, 6, 10]),
                span("c", &[3, 7, 11])
            ]
        )
        .is_some());
        assert!(overlap(
            &tasks,
            &[
                span("a", &[1, 9]),
                span("b", &[2, 6, 10]),
                span("c", &[3, 7, 11])
            ]
        )
        .is_some());
    }
}
