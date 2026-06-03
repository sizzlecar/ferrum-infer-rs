//! G3 prefix-cache product smoke for `ferrum serve`.
//!
//!     ferrum pull qwen3:0.6b
//!     cargo test --release -p ferrum-cli --test server_prefix_cache_product -- --ignored --test-threads=1

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
    let dir = current.parent().and_then(|p| p.parent()).expect("target dir");
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
    std::env::temp_dir().join(format!("ferrum-g3-{name}-{}-{now}.log", std::process::id()))
}

struct ServerFixture {
    base_url: String,
    child: Child,
    log_path: PathBuf,
}

impl ServerFixture {
    async fn spawn(extra_args: &[&str], name: &str) -> Self {
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let log_path = unique_log_path(name);
        let log = fs::File::create(&log_path).expect("create server log");
        let mut args = vec!["serve", SMOKE_MODEL, "--host", "127.0.0.1", "--port"];
        let port_string = port.to_string();
        args.push(&port_string);
        args.extend_from_slice(extra_args);
        let child = Command::new(ferrum_bin())
            .args(args)
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

    fn metrics_url(&self) -> String {
        format!("{}/metrics", self.base_url)
    }
}

impl Drop for ServerFixture {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Ok(text) = fs::read_to_string(&self.log_path) {
            for bad in ["panicked", "KV cache overflow", "failed to render model chat template", "<unk>", "[PAD]"] {
                assert!(!text.contains(bad), "server log contains {bad}: {text}");
            }
        }
        let _ = fs::remove_file(&self.log_path);
    }
}

async fn chat(client: &Client, fx: &ServerFixture, content: &str) -> String {
    let response = client
        .post(fx.chat_url())
        .json(&json!({
            "model": SMOKE_MODEL,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 256
        }))
        .send()
        .await
        .expect("chat post");
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("chat json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("missing content: {body}"))
        .trim()
        .to_string();
    assert!(!content.is_empty(), "empty visible content: {body}");
    content
}

async fn strict_json_answer(
    client: &Client,
    fx: &ServerFixture,
    prompt: &str,
    expected: &str,
) -> Value {
    let response = client
        .post(fx.chat_url())
        .json(&json!({
            "model": SMOKE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": strict_answer_schema()
        }))
        .send()
        .await
        .expect("strict answer post");
    let status = response.status();
    let raw = response.text().await.expect("strict answer body");
    assert_eq!(status, 200, "strict answer failed: {raw}");
    let body: Value = serde_json::from_str(&raw).expect("strict answer json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("missing strict content: {body}"));
    let parsed: Value = serde_json::from_str(content)
        .unwrap_or_else(|e| panic!("invalid strict JSON {content:?}: {e}"));
    assert_eq!(parsed["answer"], expected, "body: {body}");
    parsed
}

async fn metrics(client: &Client, fx: &ServerFixture) -> String {
    client
        .get(fx.metrics_url())
        .send()
        .await
        .expect("metrics")
        .text()
        .await
        .expect("metrics text")
}

fn metric_value(metrics: &str, name: &str) -> f64 {
    metrics
        .lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            (parts.next()? == name).then(|| parts.next()?.parse::<f64>().ok()).flatten()
        })
        .unwrap_or_else(|| panic!("missing metric {name}:\n{metrics}"))
}

fn strict_answer_schema() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "Answer",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"]
            }
        }
    })
}

fn calc_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "calc",
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
async fn g3_prefix_cache_product_real_model_smoke() {
    let client = Client::new();
    let enabled = ServerFixture::spawn(
        &["--enable-prefix-cache", "--session-cache", "off", "--session-cache-max-entries", "32"],
        "prefix-enabled",
    )
    .await;

    let prompt = "Reply with exactly: ferrum-cache-ok";
    let first = chat(&client, &enabled, prompt).await;
    let second = chat(&client, &enabled, prompt).await;
    assert_eq!(first, second, "greedy output changed with prefix cache");

    let alpha = strict_json_answer(
        &client,
        &enabled,
        "Shared prefix: marker request. Return exactly this JSON object and nothing else: {\"answer\":\"alpha-token\"}",
        "alpha-token",
    )
    .await;
    let beta = strict_json_answer(
        &client,
        &enabled,
        "Shared prefix: marker request. Return exactly this JSON object and nothing else: {\"answer\":\"beta-token\"}",
        "beta-token",
    )
    .await;
    assert_ne!(alpha["answer"], beta["answer"], "shared-prefix outputs cross-talked");

    let strict = client
        .post(enabled.chat_url())
        .json(&json!({
            "model": SMOKE_MODEL,
            "messages": [{"role": "user", "content": "Return exactly this JSON object and nothing else: {\"answer\":\"cache-ok\"}"}],
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": strict_answer_schema()
        }))
        .send()
        .await
        .expect("strict post");
    assert_eq!(strict.status(), 200);
    let strict_body: Value = strict.json().await.expect("strict json");
    let strict_content = strict_body["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(serde_json::from_str::<Value>(strict_content).is_ok());

    let tool = client
        .post(enabled.chat_url())
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
        .expect("tool post");
    assert_eq!(tool.status(), 200);
    let tool_body: Value = tool.json().await.expect("tool json");
    assert_eq!(tool_body["choices"][0]["finish_reason"], "tool_calls");

    let enabled_metrics = metrics(&client, &enabled).await;
    assert!(metric_value(&enabled_metrics, "ferrum_prefix_cache_hits_total") > 0.0);
    assert!(metric_value(&enabled_metrics, "ferrum_prefix_cache_saved_prefill_tokens_total") > 0.0);
    drop(enabled);

    let disabled = ServerFixture::spawn(&["--disable-prefix-cache", "--session-cache", "off"], "prefix-disabled").await;
    let _ = chat(&client, &disabled, prompt).await;
    let _ = chat(&client, &disabled, prompt).await;
    let disabled_metrics = metrics(&client, &disabled).await;
    assert_eq!(metric_value(&disabled_metrics, "ferrum_prefix_cache_hits_total"), 0.0);
}
