//! G2 agent/tool-call real-model smoke for `ferrum serve`.
//!
//! Run with a cached small model:
//!
//!     ferrum pull qwen3:0.6b
//!     cargo test --release -p ferrum-cli --test server_agent_tools -- --ignored --test-threads=1

use reqwest::Client;
use serde_json::{json, Value};
use std::fs;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SMOKE_MODEL: &str = "qwen3:0.6b";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);

fn ferrum_bin() -> PathBuf {
    if let Ok(bin) = std::env::var("CARGO_BIN_EXE_ferrum") {
        return PathBuf::from(bin);
    }
    let current = std::env::current_exe().expect("test exe path");
    let dir = current
        .parent()
        .and_then(|p| p.parent())
        .expect("target dir");
    let mut bin = dir.join("ferrum");
    if cfg!(windows) {
        bin.set_extension("exe");
    }
    assert!(bin.exists(), "ferrum binary not found at {}", bin.display());
    bin
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local_addr").port()
}

fn unique_log_path(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("ferrum-g2-{name}-{}-{now}.log", std::process::id()))
}

struct ServerFixture {
    base_url: String,
    child: Child,
    log_path: PathBuf,
}

impl ServerFixture {
    async fn spawn() -> Self {
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let log_path = unique_log_path("agent-tools-server");
        let log = fs::File::create(&log_path).expect("create server log");
        let child = Command::new(ferrum_bin())
            .args([
                "serve",
                SMOKE_MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
            ])
            .env("NO_COLOR", "1")
            .stdout(Stdio::from(log.try_clone().expect("clone server log")))
            .stderr(Stdio::from(log))
            .spawn()
            .expect("spawn ferrum serve");

        let client = Client::new();
        let healthz = format!("{base_url}/health");
        let start = Instant::now();
        loop {
            if start.elapsed() > STARTUP_TIMEOUT {
                panic!("server did not become healthy within {STARTUP_TIMEOUT:?}");
            }
            let ok = client
                .get(&healthz)
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            if ok {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }

        Self {
            base_url,
            child,
            log_path,
        }
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }
}

impl Drop for ServerFixture {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Ok(text) = fs::read_to_string(&self.log_path) {
            for bad in [
                "panicked",
                "KV cache overflow",
                "failed to render model chat template",
                "<unk>",
                "[PAD]",
            ] {
                assert!(!text.contains(bad), "server log contains {bad}: {text}");
            }
        }
        let _ = fs::remove_file(&self.log_path);
    }
}

fn parse_sse(body: &str) -> (Vec<Value>, usize) {
    let mut chunks = Vec::new();
    let mut done = 0usize;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            done += 1;
        } else if !data.is_empty() {
            chunks.push(serde_json::from_str(data).expect("valid SSE JSON"));
        }
    }
    (chunks, done)
}

fn calc_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a deterministic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "enum": ["123+456"]}},
                "required": ["expression"]
            }
        }
    })
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads qwen3:0.6b real model"]
async fn g2_agent_tools_real_model_smoke() {
    let fx = ServerFixture::spawn().await;
    let client = Client::new();

    let bad_tool = client
        .post(fx.chat_url())
        .json(&json!({
            "model": SMOKE_MODEL,
            "messages": [{"role": "user", "content": "search"}],
            "tools": [{"type": "retrieval", "function": {"name": "search", "parameters": {"type": "object"}}}]
        }))
        .send()
        .await
        .expect("post bad tool");
    assert_eq!(bad_tool.status(), 400);
    let bad_tool_body: Value = bad_tool.json().await.expect("bad tool json");
    assert_eq!(bad_tool_body["error"]["param"], "tools");

    let undeclared = client
        .post(fx.chat_url())
        .json(&json!({
            "model": SMOKE_MODEL,
            "messages": [{"role": "user", "content": "call calc"}],
            "tools": [calc_tool()],
            "tool_choice": {"type": "function", "function": {"name": "missing"}}
        }))
        .send()
        .await
        .expect("post undeclared tool_choice");
    assert_eq!(undeclared.status(), 400);
    let undeclared_body: Value = undeclared.json().await.expect("undeclared json");
    assert_eq!(undeclared_body["error"]["param"], "tool_choice");

    for i in 0..10 {
        let response = client
            .post(fx.chat_url())
            .json(&json!({
                "model": SMOKE_MODEL,
                "messages": [{"role": "user", "content": "Use the calc tool. Return only JSON arguments: {\"expression\":\"123+456\"}"}],
                "tools": [calc_tool()],
                "tool_choice": "required",
                "temperature": 0.0,
                "max_tokens": 128
            }))
            .send()
            .await
            .unwrap_or_else(|e| panic!("post tool iteration {i}: {e}"));
        assert_eq!(response.status(), 200, "iteration {i}");
        let body: Value = response.json().await.expect("tool json");
        assert_eq!(
            body["choices"][0]["finish_reason"], "tool_calls",
            "iteration {i}: {body}"
        );
        let call = &body["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["type"], "function", "iteration {i}: {body}");
        assert_eq!(call["function"]["name"], "calc", "iteration {i}: {body}");
        let args: Value = serde_json::from_str(
            call["function"]["arguments"]
                .as_str()
                .unwrap_or_else(|| panic!("missing arguments iteration {i}: {body}")),
        )
        .unwrap_or_else(|e| panic!("bad arguments JSON iteration {i}: {body}: {e}"));
        assert_eq!(args["expression"], "123+456", "iteration {i}: {body}");
    }

    let stream = client
        .post(fx.chat_url())
        .json(&json!({
            "model": SMOKE_MODEL,
            "messages": [{"role": "user", "content": "Use the calc tool. Return only JSON arguments: {\"expression\":\"123+456\"}"}],
            "tools": [calc_tool()],
            "tool_choice": "required",
            "temperature": 0.0,
            "max_tokens": 128,
            "stream": true
        }))
        .send()
        .await
        .expect("post tool stream");
    assert_eq!(stream.status(), 200);
    let stream_body = stream.text().await.expect("stream body");
    let (chunks, done) = parse_sse(&stream_body);
    assert_eq!(done, 1, "stream body: {stream_body}");
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["choices"][0]["delta"].get("tool_calls").is_some()),
        "missing tool_calls delta: {stream_body}"
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["choices"][0]["finish_reason"] == "tool_calls"),
        "missing tool_calls finish: {stream_body}"
    );
}
