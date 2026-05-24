//! Reference-output snapshot tests — byte-equal regression gate.
//!
//! Catches tokenizer / kernel / sampler regressions that change byte-level
//! output (e.g. the PR #204 prefix-cache greedy-determinism class). Each
//! case is a frozen chat completion request + ferrum's expected reply;
//! any drift in `content` / `finish_reason` / `completion_tokens` fails
//! the test.
//!
//! ## What this is (v0)
//!
//! Self-snapshot baseline, modelled on vLLM's `test_fingerprint.py`
//! (`tests/entrypoints/openai/test_fingerprint.py`): not a vs-reference
//! oracle, but a "drift gate" — once frozen, ANY change to a case's
//! output requires an explicit human-reviewed re-run with
//! `FERRUM_UPDATE_FIXTURES=1`.
//!
//! ## What's next (v1, not yet done)
//!
//! Replace `expected` with HF-transformers outputs (the true oracle —
//! vLLM's `tests/models/utils.py::check_outputs_equal` pattern). Until
//! then, this test catches "ferrum drifted from its own past self".
//!
//! ## How to use
//!
//! Update fixture (only after human-reviewing the diff — e.g. you
//! intentionally changed inference behaviour):
//!
//!     FERRUM_UPDATE_FIXTURES=1 cargo test --release -p ferrum-cli \
//!         --features metal --test reference_match -- --ignored --test-threads=1
//!
//! Verify (CI nightly):
//!
//!     cargo test --release -p ferrum-cli --features metal \
//!         --test reference_match -- --ignored --test-threads=1

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const FIXTURE_PATH: &str = "tests/fixtures/reference_outputs.json";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const UPDATE_ENV: &str = "FERRUM_UPDATE_FIXTURES";

// ─────────────────────────────────────────────────────────────────────────────
// Fixture schema — keep in sync with tests/fixtures/reference_outputs.json
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fixture {
    #[serde(
        rename = "$schema_doc",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    schema_doc: String,
    #[serde(rename = "$schema_version")]
    schema_version: u32,
    #[serde(
        rename = "$update_command",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    update_command: String,
    cases: Vec<Case>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Case {
    id: String,
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
    expected: Option<ExpectedOutput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ExpectedOutput {
    content: String,
    finish_reason: String,
    completion_tokens: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Server fixture — duplicates the pattern from server_smoke.rs /
// server_openai_compat.rs. Worth extracting to tests/common/ once a 4th
// HTTP test file lands.
// ─────────────────────────────────────────────────────────────────────────────

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
        let port = free_port();
        let base_url = format!("http://127.0.0.1:{port}");
        let child = Command::new(ferrum_bin())
            .args(["serve", model, "--port", &port.to_string()])
            .env("NO_COLOR", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn ferrum serve");

        let probe = Client::new();
        let healthz = format!("{base_url}/health");
        let start = Instant::now();
        loop {
            if start.elapsed() > STARTUP_TIMEOUT {
                panic!("server did not become healthy within {STARTUP_TIMEOUT:?}");
            }
            let ok = probe
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
// Fixture I/O
// ─────────────────────────────────────────────────────────────────────────────

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_PATH)
}

fn read_fixture(path: &Path) -> Fixture {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse fixture {}: {e}", path.display()))
}

fn write_fixture(path: &Path, fixture: &Fixture) {
    let pretty = serde_json::to_string_pretty(fixture).expect("serialize fixture");
    std::fs::write(path, format!("{pretty}\n"))
        .unwrap_or_else(|e| panic!("write fixture {}: {e}", path.display()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-case execution
// ─────────────────────────────────────────────────────────────────────────────

async fn run_case(client: &Client, fx: &ServerFixture, case: &Case) -> ExpectedOutput {
    let body = json!({
        "model": case.model,
        "messages": case.messages.iter().map(|m| json!({
            "role": m.role,
            "content": m.content,
        })).collect::<Vec<_>>(),
        "max_tokens": case.max_tokens,
        "temperature": case.temperature,
        "stream": false,
    });

    let resp = client
        .post(fx.chat_url())
        .timeout(REQUEST_TIMEOUT)
        .json(&body)
        .send()
        .await
        .unwrap_or_else(|e| panic!("case {}: HTTP send failed: {e}", case.id));
    assert!(
        resp.status().is_success(),
        "case {}: HTTP {} from /v1/chat/completions",
        case.id,
        resp.status()
    );

    let v: Value = resp
        .json()
        .await
        .unwrap_or_else(|e| panic!("case {}: JSON decode failed: {e}", case.id));

    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("case {}: missing choices[0].message.content", case.id))
        .to_string();
    let finish_reason = v["choices"][0]["finish_reason"]
        .as_str()
        .unwrap_or_else(|| panic!("case {}: missing choices[0].finish_reason", case.id))
        .to_string();
    let completion_tokens = v["usage"]["completion_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("case {}: missing usage.completion_tokens", case.id))
        as u32;

    ExpectedOutput {
        content,
        finish_reason,
        completion_tokens,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main test entry
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model — run with `cargo test -- --ignored`"]
async fn reference_match() {
    let path = fixture_path();
    let mut fixture = read_fixture(&path);
    assert_eq!(
        fixture.schema_version, 1,
        "unknown fixture schema_version {}; this test only understands v1",
        fixture.schema_version
    );

    let update_mode = std::env::var(UPDATE_ENV).as_deref() == Ok("1");
    let mut drift_report = Vec::new();

    // Group cases by model so each model is loaded once (cold-load is the
    // dominant cost — keep server alive across all of its cases).
    let mut by_model: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (idx, case) in fixture.cases.iter().enumerate() {
        by_model.entry(case.model.clone()).or_default().push(idx);
    }

    let client = Client::new();
    for (model, case_indices) in by_model {
        eprintln!("──── model: {} ({} cases) ────", model, case_indices.len());
        let fx = ServerFixture::spawn(&model).await;
        for idx in case_indices {
            let case_id = fixture.cases[idx].id.clone();
            let case_clone = fixture.cases[idx].clone();
            let actual = run_case(&client, &fx, &case_clone).await;
            eprintln!(
                "  case {}: content={:?} finish={} tok={}",
                case_id, actual.content, actual.finish_reason, actual.completion_tokens
            );

            if update_mode {
                fixture.cases[idx].expected = Some(actual);
                continue;
            }

            match &fixture.cases[idx].expected {
                None => {
                    drift_report.push(format!(
                        "case {}: NO expected baseline — run with {UPDATE_ENV}=1",
                        case_id
                    ));
                }
                Some(expected) if expected != &actual => {
                    drift_report.push(format!(
                        "case {}: DRIFT\n  expected: {:?}\n  actual:   {:?}",
                        case_id, expected, actual
                    ));
                }
                Some(_) => {} // match — no entry in drift report
            }
        }
        drop(fx);
    }

    if update_mode {
        write_fixture(&path, &fixture);
        eprintln!(
            "✓ Fixture updated: {} ({} cases)",
            path.display(),
            fixture.cases.len()
        );
        return;
    }

    if !drift_report.is_empty() {
        panic!(
            "reference-output drift in {} case(s):\n\n{}\n\n\
             To re-baseline after human-reviewed change, run:\n  {}",
            drift_report.len(),
            drift_report.join("\n\n"),
            fixture.update_command
        );
    }

    eprintln!(
        "✓ All {} reference cases match baseline.",
        fixture.cases.len()
    );
}
