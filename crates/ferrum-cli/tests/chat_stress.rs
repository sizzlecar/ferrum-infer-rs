//! Stress smoke tests — exercise paths that single-turn or short-session
//! tests can't reach:
//!
//! - Long REPL sessions (30 turns) don't accumulate state or OOM.
//! - Multiple `ferrum run` processes can coexist on the same Metal device.
//!
//! Loads a real model; `#[ignore]` by default. Opt in:
//!
//!     ferrum pull qwen3:0.6b
//!     cargo test --release -p ferrum-cli --features metal --test chat_stress \
//!       -- --ignored --test-threads=1

use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;

const SMOKE_MODEL: &str = "qwen3:0.6b";

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

/// Drive a long REPL session via stdin pipe; parse JSONL stdout into the
/// list of assistant events. Used by the stress tests below.
fn run_long_session(turns: usize, max_tokens: u32) -> Vec<Value> {
    let max_tokens_s = max_tokens.to_string();
    let mut cmd = Command::new(ferrum_bin());
    cmd.args([
        "run",
        SMOKE_MODEL,
        "--output-format",
        "jsonl",
        "--temperature",
        "0",
        "--max-tokens",
        &max_tokens_s,
    ]);
    cmd.env("NO_COLOR", "1");
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn ferrum");
    {
        let stdin = child.stdin.as_mut().expect("child stdin");
        for i in 1..=turns {
            writeln!(stdin, "Reply with just the number {i}.").expect("write stdin");
        }
        writeln!(stdin, "/bye").expect("write /bye");
    }
    let output = child.wait_with_output().expect("wait child");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "ferrum run exited non-zero ({:?}); stderr:\n{stderr}",
            output.status
        );
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v["event"].as_str() == Some("assistant"))
        .collect()
}

#[test]
#[ignore = "loads real model — run with `cargo test -- --ignored`"]
fn test_stress_30_turns_no_oom() {
    // 30 turns is well past the history cap of 10, so the truncation
    // path executes ~20 times. Guards against unbounded growth in
    // `history: Vec<(String, String)>` or downstream KV state.
    let assistants = run_long_session(30, 20);
    assert_eq!(
        assistants.len(),
        30,
        "expected 30 assistant turns, got {}",
        assistants.len()
    );
    // All turns must have non-empty content and a valid finish_reason —
    // catches regressions where late turns silently produce empty results.
    for (i, a) in assistants.iter().enumerate() {
        let content = a["content"].as_str().unwrap_or("");
        assert!(!content.trim().is_empty(), "turn {i} content empty: {a:?}");
        let fr = a["finish_reason"].as_str();
        assert!(
            matches!(fr, Some("stop" | "eos" | "length")),
            "turn {i} bad finish_reason: {fr:?}"
        );
    }
}

#[test]
#[ignore = "loads real model"]
fn test_concurrent_three_instances() {
    // Three independent `ferrum run` subprocesses on the same Metal device.
    // Each writes a distinct prompt and must get a non-empty response —
    // guards against accidental global state (PID file, shared singleton,
    // Metal device contention) that would serialize or corrupt cohabiters.
    let prompts: Arc<Vec<&'static str>> = Arc::new(vec![
        "Say the word ALPHA once.",
        "Say the word BRAVO once.",
        "Say the word CHARLIE once.",
    ]);

    let handles: Vec<_> = (0..3)
        .map(|i| {
            let prompts = Arc::clone(&prompts);
            thread::spawn(move || {
                let prompt = prompts[i];
                let output = Command::new(ferrum_bin())
                    .args([
                        "run",
                        SMOKE_MODEL,
                        "--output-format",
                        "jsonl",
                        "--temperature",
                        "0",
                        "--max-tokens",
                        "40",
                        "--prompt",
                        prompt,
                    ])
                    .env("NO_COLOR", "1")
                    .output()
                    .expect("spawn ferrum");
                assert!(
                    output.status.success(),
                    "instance {i} ({prompt:?}) exited non-zero: {:?}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let stdout = String::from_utf8_lossy(&output.stdout);
                let assistant_line = stdout
                    .lines()
                    .find(|l| l.contains("\"event\":\"assistant\""))
                    .unwrap_or_else(|| {
                        panic!("instance {i} produced no assistant event; stdout: {stdout:?}")
                    });
                let v: Value = serde_json::from_str(assistant_line).expect("json");
                let content = v["content"].as_str().unwrap_or("").to_string();
                (i, content)
            })
        })
        .collect();

    let results: Vec<(usize, String)> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.len(), 3, "all 3 instances must complete");
    for (i, content) in &results {
        assert!(
            !content.trim().is_empty(),
            "instance {i} produced empty content"
        );
    }
}
