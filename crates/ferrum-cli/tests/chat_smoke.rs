//! Chat REPL smoke tests via `--output-format jsonl`.
//!
//! Drives `ferrum run` with stdin pipe + JSONL stdout, then asserts
//! structural properties per turn. Catches the d67fbbb-class regressions:
//! EOS / stop_sequences / stream / multi-turn KV / chat template.
//!
//! These tests load a real model and default to `#[ignore]` so a bare
//! `cargo test` stays fast. Pre-pull the model once, then opt in:
//!
//!     ferrum pull qwen3:0.6b
//!     cargo test -p ferrum-cli --features metal --test chat_smoke -- --ignored

use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const SMOKE_MODEL: &str = "qwen3:0.6b";

/// One JSONL record from `ferrum run --output-format jsonl`.
#[derive(Debug)]
struct ChatEvent {
    raw: Value,
}

impl ChatEvent {
    fn event(&self) -> &str {
        self.raw["event"].as_str().unwrap_or("")
    }
    fn content(&self) -> Option<&str> {
        self.raw["content"].as_str()
    }
    fn finish_reason(&self) -> Option<&str> {
        self.raw["finish_reason"].as_str()
    }
}

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

/// Spawn `ferrum run --output-format jsonl`, feed `stdin_lines` (one per line),
/// then parse JSONL stdout into events. Panics on non-zero exit.
fn run_chat(model: &str, system: Option<&str>, stdin_lines: &[&str]) -> Vec<ChatEvent> {
    let mut cmd = Command::new(ferrum_bin());
    cmd.args([
        "run",
        model,
        "--output-format",
        "jsonl",
        "--temperature",
        "0",
        "--max-tokens",
        "100",
    ]);
    if let Some(s) = system {
        cmd.args(["--system", s]);
    }
    cmd.env("NO_COLOR", "1");
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("spawn ferrum");
    {
        let stdin = child.stdin.as_mut().expect("child stdin");
        for line in stdin_lines {
            writeln!(stdin, "{line}").expect("write stdin");
        }
    }
    let output = child.wait_with_output().expect("wait child");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "ferrum run exited non-zero ({:?}); stderr:\n{stderr}",
            output.status
        );
    }
    parse_jsonl(&String::from_utf8_lossy(&output.stdout))
}

/// One-shot: `ferrum run <model> --prompt X --output-format jsonl`.
fn run_chat_oneshot(model: &str, prompt: &str) -> Vec<ChatEvent> {
    let mut cmd = Command::new(ferrum_bin());
    cmd.args([
        "run",
        model,
        "--output-format",
        "jsonl",
        "--temperature",
        "0",
        "--max-tokens",
        "50",
        "--prompt",
        prompt,
    ]);
    cmd.env("NO_COLOR", "1");
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let output = cmd.output().expect("run ferrum one-shot");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("ferrum --prompt exited non-zero; stderr:\n{stderr}");
    }
    parse_jsonl(&String::from_utf8_lossy(&output.stdout))
}

fn parse_jsonl(stdout: &str) -> Vec<ChatEvent> {
    stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let raw: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("bad JSONL line: {line:?}\nerror: {e}"));
            ChatEvent { raw }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// PR 1 — 4 critical tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "loads real model — run with `cargo test -- --ignored` after `ferrum pull qwen3:0.6b`"]
fn test_oneshot_prompt() {
    let events = run_chat_oneshot(SMOKE_MODEL, "Say hi in one short sentence.");
    let assistant: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(
        assistant.len(),
        1,
        "expected exactly one assistant event, got {}",
        assistant.len()
    );
    let a = &assistant[0];
    assert!(
        a.content().map(|c| !c.trim().is_empty()).unwrap_or(false),
        "assistant content was empty"
    );
    assert!(
        matches!(a.finish_reason(), Some("stop" | "eos" | "length")),
        "unexpected finish_reason: {:?}",
        a.finish_reason()
    );
}

#[test]
#[ignore = "loads real model"]
fn test_repl_natural_eos() {
    let events = run_chat(
        SMOKE_MODEL,
        None,
        &["Say hi in one short sentence.", "/bye"],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(
        assistants.len(),
        1,
        "expected 1 assistant turn, got {}",
        assistants.len()
    );
    let a = &assistants[0];
    // Short prompt with max-tokens=100 → should hit natural stop, not length truncation.
    // This catches the d67fbbb-class regression where EOS detection placeholder
    // (vocab_size-10) never fires, so finish_reason was always 'length'.
    assert!(
        matches!(a.finish_reason(), Some("stop" | "eos")),
        "expected natural stop / eos; got finish_reason={:?}, content={:?}",
        a.finish_reason(),
        a.content()
    );
}

#[test]
#[ignore = "loads real model"]
fn test_repl_multi_turn_recall() {
    let events = run_chat(
        SMOKE_MODEL,
        None,
        &[
            "Remember this fact: my name is XiaoMing.",
            "What is my name? Reply with just the name.",
            "/bye",
        ],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(
        assistants.len(),
        2,
        "expected 2 assistant turns, got {}",
        assistants.len()
    );
    let second = assistants[1].content().unwrap_or("");
    // Catches d67fbbb-class regression where history wasn't carried into turn 2.
    assert!(
        second.to_lowercase().contains("xiaoming"),
        "second turn should recall 'XiaoMing'; got: {second:?}"
    );
}

#[test]
#[ignore = "loads real model"]
fn test_repl_no_template_leak() {
    let events = run_chat(
        SMOKE_MODEL,
        None,
        &["Hi there.", "Tell me a small number.", "/bye"],
    );
    // Sampler / stop-sequence path should swallow these, never expose them
    // to user-visible content. If a regression lets them through, it's a
    // chat-template double-application or sampler stop bug.
    let leaky_markers = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "</s>", "<s>"];
    for e in events.iter().filter(|e| e.event() == "assistant") {
        let content = e.content().unwrap_or("");
        for marker in &leaky_markers {
            assert!(
                !content.contains(marker),
                "assistant leaked special token {marker:?} in content: {content:?}"
            );
        }
    }
}
