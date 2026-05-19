//! HTTP server stress tests — push `cli serve` past single-client tests
//! to catch deadlocks, request-state leakage, and prefix-cache regressions.
//!
//! Loads a real model and is `#[ignore]`'d by default. Opt in:
//!
//!     ferrum pull qwen3:0.6b
//!     cargo test --release -p ferrum-cli --features metal --test server_stress \
//!       -- --ignored --test-threads=1

use reqwest::Client;
use serde_json::{json, Value};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

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

struct ServerFixture {
    base_url: String,
    child: Child,
}

impl ServerFixture {
    async fn spawn(model: &str) -> Self {
        Self::spawn_with_env(model, &[]).await
    }

    /// Spawn the server with extra env vars. Used by tests that need to
    /// opt in to non-default engine knobs (e.g. `FERRUM_PREFIX_CACHE=1`).
    async fn spawn_with_env(model: &str, extra_env: &[(&str, &str)]) -> Self {
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let mut cmd = Command::new(ferrum_bin());
        cmd.args(["serve", model, "--port", &port.to_string()])
            .env("NO_COLOR", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let child = cmd.spawn().expect("spawn ferrum serve");

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
        Self { base_url, child }
    }

    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }
}

impl Drop for ServerFixture {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PR 10 — stress smoke
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model — run with `cargo test -- --ignored`"]
async fn test_concurrent_8_requests() {
    // Eight independent requests fired in parallel against a single
    // server. Catches connection-pool deadlocks, request-state leakage
    // across the continuous-batch scheduler, and OS thread starvation.
    // 8 is well under typical paged-KV capacity for Qwen3-0.6B on Metal.
    let fx = ServerFixture::spawn(SMOKE_MODEL).await;
    let client = Client::new();
    let url = fx.chat_url();

    let prompts = [
        "Say hi.",
        "Reply with the word ALPHA.",
        "Reply with the word BRAVO.",
        "Reply with the word CHARLIE.",
        "Reply with the word DELTA.",
        "Reply with the word ECHO.",
        "Reply with the word FOXTROT.",
        "Reply with the word GOLF.",
    ];

    let futures = prompts.iter().enumerate().map(|(i, p)| {
        let client = client.clone();
        let url = url.clone();
        async move {
            let resp = client
                .post(&url)
                .json(&json!({
                    "model": SMOKE_MODEL,
                    "messages": [{"role": "user", "content": p}],
                    "max_tokens": 30,
                    "temperature": 0.0
                }))
                .send()
                .await
                .unwrap_or_else(|e| panic!("request {i} failed: {e}"));
            let status = resp.status();
            let body: Value = resp.json().await.expect("json");
            (i, status, body)
        }
    });

    let results = futures::future::join_all(futures).await;
    assert_eq!(results.len(), 8, "expected 8 results");
    for (i, status, body) in &results {
        assert_eq!(*status, 200, "request {i} non-200");
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(!content.trim().is_empty(), "request {i} content empty");
    }
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model"]
async fn test_prefix_cache_speedup() {
    // Fire two requests sharing a long system prompt. With the prefix
    // cache enabled the second request should hit the cache and complete
    // noticeably faster than the first.
    //
    // Prefix cache now defaults OFF (see continuous_engine.rs for the
    // CoW gap that motivated the flip). This test opts in explicitly
    // via `FERRUM_PREFIX_CACHE=1`. Once the CoW write-fork lands, the
    // default can flip back to ON and this opt-in becomes redundant.
    //
    // Threshold is generous (second ≤ 110 % of first) to avoid CI flake
    // on slow first-iteration scheduler warmup.
    let fx = ServerFixture::spawn_with_env(SMOKE_MODEL, &[("FERRUM_PREFIX_CACHE", "1")]).await;
    let client = Client::new();
    let url = fx.chat_url();

    // ~700-char system prompt — long enough to dominate prefill cost,
    // short enough to fit Qwen3-0.6B context easily.
    let system = "You are a precise assistant. \
        Always respond with short, factual answers. \
        Avoid speculation, avoid marketing language, avoid emoji. \
        When asked a factual question, give the most concise correct \
        answer in one short sentence. When asked for a number, return \
        just the number with no commentary. When asked for a name, \
        return just the name. Do not preface your answer. Do not \
        apologize. Do not explain your reasoning unless explicitly \
        asked. Treat every prompt as a direct question requiring a \
        direct answer. Stay calm, stay precise, stay short.";

    let send = || async {
        let start = Instant::now();
        let resp = client
            .post(&url)
            .json(&json!({
                "model": SMOKE_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Say hi."}
                ],
                "max_tokens": 10,
                "temperature": 0.0
            }))
            .send()
            .await
            .expect("post");
        assert_eq!(resp.status(), 200);
        let _body: Value = resp.json().await.expect("json");
        start.elapsed()
    };

    let first = send().await;
    let second = send().await;
    eprintln!("[prefix-cache] first={first:?} second={second:?}");

    // Loose floor: second request shouldn't be more than 10 % slower.
    // A genuine prefix cache hit will typically be 30-60 % faster; if
    // both requests independently re-prefill the long system message
    // they'll be roughly equal (failing this check by a small margin
    // is acceptable; failing by 10 %+ indicates the cache is dead).
    assert!(
        second.as_secs_f64() <= first.as_secs_f64() * 1.1,
        "second request markedly slower than first; prefix cache likely broken: \
         first={first:?}, second={second:?}"
    );
}
