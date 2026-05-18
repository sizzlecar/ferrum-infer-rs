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
    fn chunk_count(&self) -> Option<u64> {
        self.raw["chunk_count"].as_u64()
    }
    fn exit_reason(&self) -> Option<&str> {
        self.raw["reason"].as_str()
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
/// then parse JSONL stdout into events. Panics on non-zero exit. Defaults to
/// `--max-tokens 100`. Use `run_chat_full` to override.
fn run_chat(model: &str, system: Option<&str>, stdin_lines: &[&str]) -> Vec<ChatEvent> {
    run_chat_full(model, system, 100, stdin_lines)
}

fn run_chat_full(
    model: &str,
    system: Option<&str>,
    max_tokens: u32,
    stdin_lines: &[&str],
) -> Vec<ChatEvent> {
    let max_tokens_s = max_tokens.to_string();
    let mut cmd = Command::new(ferrum_bin());
    cmd.args([
        "run",
        model,
        "--output-format",
        "jsonl",
        "--temperature",
        "0",
        "--max-tokens",
        &max_tokens_s,
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

// ─────────────────────────────────────────────────────────────────────────────
// PR 2 — remaining 7 Tier 1 tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "loads real model"]
fn test_repl_max_tokens_truncation() {
    // --max-tokens=5 forces length truncation (model won't naturally finish
    // a "long story" prompt in 5 tokens). Catches regressions where
    // max-tokens isn't propagated to SamplingParams.
    let events = run_chat_full(
        SMOKE_MODEL,
        None,
        5,
        &["Tell me a long story about dragons.", "/bye"],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 1, "expected 1 assistant turn");
    assert_eq!(
        assistants[0].finish_reason(),
        Some("length"),
        "max-tokens=5 should hit length truncation; got {:?}; content={:?}",
        assistants[0].finish_reason(),
        assistants[0].content()
    );
}

#[test]
#[ignore = "loads real model"]
fn test_repl_stop_seq_each_turn() {
    // 3 short turns, every one must hit a natural stop. This is the
    // d67fbbb regression-net for stop-detection across multiple turns —
    // a stop-state-leak between turns would manifest as missing/wrong
    // finish_reason on later turns.
    let events = run_chat(
        SMOKE_MODEL,
        None,
        &[
            "Hi.",
            "What's 2 plus 2?",
            "What's the capital of France?",
            "/bye",
        ],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 3, "expected 3 assistant turns");
    for (i, a) in assistants.iter().enumerate() {
        assert!(
            matches!(a.finish_reason(), Some("stop" | "eos")),
            "turn {} unexpected finish_reason: {:?}",
            i,
            a.finish_reason()
        );
    }
}

#[test]
#[ignore = "loads real model"]
fn test_repl_chunk_count_matches() {
    // chunk_count must be > 0 when content is non-empty, and content must
    // be at least chunk_count bytes (each emitted chunk has ≥1 byte).
    // Catches stream-counter regressions and "last chunk swallowed" bugs.
    let events = run_chat(
        SMOKE_MODEL,
        None,
        &["Write one short sentence about cats.", "/bye"],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 1);
    let a = &assistants[0];
    let chunk_count = a.chunk_count().expect("chunk_count field present");
    let content = a.content().unwrap_or("");
    assert!(
        chunk_count > 0,
        "chunk_count should be > 0 with non-empty content; got 0, content={content:?}"
    );
    assert!(
        content.len() >= chunk_count as usize,
        "content.len()={} should be >= chunk_count={chunk_count} (each chunk ≥ 1 byte); content={content:?}",
        content.len()
    );
}

#[test]
#[ignore = "loads real model"]
fn test_repl_history_truncation_at_10() {
    // CLI caps history at 10 entries. Drive 12 short turns and verify
    // each turn still produces a non-empty response with a valid
    // finish_reason — proves the truncation logic doesn't crash mid-flight
    // and doesn't corrupt the prompt structure for later turns.
    let mut lines: Vec<String> = (1..=12)
        .map(|i| format!("Reply with just the number {i}."))
        .collect();
    lines.push("/bye".to_string());
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();

    let events = run_chat_full(SMOKE_MODEL, None, 30, &line_refs);
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(
        assistants.len(),
        12,
        "expected 12 assistant turns, got {}",
        assistants.len()
    );
    for (i, a) in assistants.iter().enumerate() {
        assert!(
            a.content().map(|c| !c.trim().is_empty()).unwrap_or(false),
            "turn {i} content empty: {:?}",
            a.content()
        );
        assert!(
            matches!(a.finish_reason(), Some("stop" | "eos" | "length")),
            "turn {i} finish_reason: {:?}",
            a.finish_reason()
        );
    }
}

#[test]
#[ignore = "loads real model"]
fn test_repl_system_prompt() {
    // System prompt should constrain generation. Catches regression where
    // --system flag is parsed but not actually plumbed through to
    // build_chat_prompt.
    let events = run_chat(
        SMOKE_MODEL,
        Some("You must always reply with exactly the word YES, nothing else."),
        &["What is the weather like?", "/bye"],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 1);
    let content = assistants[0].content().unwrap_or("");
    // Loose check: response must contain "yes" (case-insensitive). Small
    // models won't be perfectly obedient but should at least mention it.
    assert!(
        content.to_lowercase().contains("yes"),
        "system prompt asked for 'YES'; assistant said: {content:?}"
    );
}

#[test]
#[ignore = "loads real model"]
fn test_repl_utf8_chinese() {
    // Pure-Chinese round-trip. Verifies UTF-8 byte boundaries don't crack
    // in tokenizer / decoder / JSONL serialization paths.
    let events = run_chat(
        SMOKE_MODEL,
        Some("请用中文回答。"),
        &["你好，请用一句话问候我。", "/bye"],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 1);
    let content = assistants[0].content().unwrap_or("");
    // Expect at least one CJK Unified Ideograph character in response.
    let has_cjk = content
        .chars()
        .any(|c| (0x4E00..=0x9FFF).contains(&(c as u32)));
    assert!(
        has_cjk,
        "expected Chinese characters in response; got: {content:?}"
    );
}

#[test]
#[ignore = "loads real model"]
fn test_repl_clean_exit_bye() {
    // The final JSONL record must be an `exit` event with reason="bye"
    // when the user typed /bye. Catches regressions in exit-path bookkeeping
    // (e.g., always emitting "eof" regardless of how loop terminated).
    let events = run_chat(SMOKE_MODEL, None, &["Hi.", "/bye"]);
    let last = events.last().expect("at least one event");
    assert_eq!(
        last.event(),
        "exit",
        "last event should be 'exit'; got: {:?}",
        last.event()
    );
    assert_eq!(
        last.exit_reason(),
        Some("bye"),
        "exit reason should be 'bye'; got: {:?}",
        last.exit_reason()
    );
}
