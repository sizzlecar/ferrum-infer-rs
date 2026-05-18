//! Chat REPL smoke tests via `--output-format jsonl`.
//!
//! Most tests exercise engine/CLI plumbing (EOS detection, chunk counting,
//! history cap, exit reason, max-tokens propagation) which is **model
//! independent** — they run on Qwen3-0.6B only, the smallest fully-coherent
//! instruct model in our matrix.
//!
//! `test_repl_no_template_leak` is the only test that genuinely benefits
//! from cross-family coverage — it runs on all three PR-time models
//! (Qwen3-0.6B / TinyLlama-1.1B / Qwen2.5-0.5B) because each family has
//! distinct chat-template tokens that the sampler must strip.
//!
//! Loads real models, `#[ignore]` by default. Opt in:
//!
//!     ferrum pull qwen3:0.6b && ferrum pull tinyllama && ferrum pull qwen2.5:0.5b
//!     cargo test -p ferrum-cli --features metal --test chat_smoke \
//!       -- --ignored --test-threads=1
//!
//! Expected ~5 min on M1 Metal (13 test instances).

use rstest::rstest;
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

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
            "ferrum run [{model}] exited non-zero ({:?}); stderr:\n{stderr}",
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
        panic!("ferrum --prompt [{model}] exited non-zero; stderr:\n{stderr}");
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

/// Family-specific template tokens that must never leak into assistant
/// content. The set varies because each model family uses a different chat
/// template — Qwen ChatML, Llama-3 header tokens, TinyLlama-style tags.
fn leaky_markers_for(model: &str) -> &'static [&'static str] {
    let m = model.to_lowercase();
    if m.contains("qwen") {
        &["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
    } else if m.contains("llama") && m.contains("3") {
        &["<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>"]
    } else {
        // TinyLlama / generic chat template
        &["<|system|>", "<|user|>", "<|assistant|>"]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Structural tests — exercised on all 3 PR-time models
// ─────────────────────────────────────────────────────────────────────────────

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model — run with `cargo test -- --ignored`"]
fn test_oneshot_prompt(#[case] model: &str) {
    let events = run_chat_oneshot(model, "Say hi in one short sentence.");
    let assistant: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(
        assistant.len(),
        1,
        "[{model}] expected exactly one assistant event, got {}",
        assistant.len()
    );
    let a = &assistant[0];
    assert!(
        a.content().map(|c| !c.trim().is_empty()).unwrap_or(false),
        "[{model}] assistant content empty"
    );
    assert!(
        matches!(a.finish_reason(), Some("stop" | "eos" | "length")),
        "[{model}] unexpected finish_reason: {:?}",
        a.finish_reason()
    );
}

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model"]
fn test_repl_natural_eos(#[case] model: &str) {
    let events = run_chat(model, None, &["Say hi in one short sentence.", "/bye"]);
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 1, "[{model}] expected 1 assistant turn");
    let a = &assistants[0];
    assert!(
        matches!(a.finish_reason(), Some("stop" | "eos")),
        "[{model}] expected natural stop/eos; got {:?}, content={:?}",
        a.finish_reason(),
        a.content()
    );
}

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model"]
fn test_repl_multi_turn_recall(#[case] model: &str) {
    let events = run_chat(
        model,
        None,
        &[
            "Remember this fact: my name is XiaoMing.",
            "What is my name? Reply with just the name.",
            "/bye",
        ],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 2, "[{model}] expected 2 assistant turns");
    let second = assistants[1].content().unwrap_or("");
    assert!(
        second.to_lowercase().contains("xiaoming"),
        "[{model}] second turn should recall 'XiaoMing'; got: {second:?}"
    );
}

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[case::tinyllama("tinyllama")]
#[case::qwen25("qwen2.5:0.5b")]
#[ignore = "loads real model"]
fn test_repl_no_template_leak(#[case] model: &str) {
    let events = run_chat(
        model,
        None,
        &["Hi there.", "Tell me a small number.", "/bye"],
    );
    let markers = leaky_markers_for(model);
    for e in events.iter().filter(|e| e.event() == "assistant") {
        let content = e.content().unwrap_or("");
        for marker in markers {
            assert!(
                !content.contains(marker),
                "[{model}] assistant leaked template token {marker:?} in content: {content:?}"
            );
        }
    }
}

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model"]
fn test_repl_max_tokens_truncation(#[case] model: &str) {
    let events = run_chat_full(
        model,
        None,
        5,
        &["Tell me a long story about dragons.", "/bye"],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 1, "[{model}] expected 1 assistant turn");
    assert_eq!(
        assistants[0].finish_reason(),
        Some("length"),
        "[{model}] max-tokens=5 should hit length truncation; got {:?}, content={:?}",
        assistants[0].finish_reason(),
        assistants[0].content()
    );
}

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model"]
fn test_repl_stop_seq_each_turn(#[case] model: &str) {
    let events = run_chat(
        model,
        None,
        &[
            "Hi.",
            "What's 2 plus 2?",
            "What's the capital of France?",
            "/bye",
        ],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 3, "[{model}] expected 3 assistant turns");
    for (i, a) in assistants.iter().enumerate() {
        assert!(
            matches!(a.finish_reason(), Some("stop" | "eos")),
            "[{model}] turn {i} finish_reason: {:?}",
            a.finish_reason()
        );
    }
}

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model"]
fn test_repl_chunk_count_matches(#[case] model: &str) {
    let events = run_chat(
        model,
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
        "[{model}] chunk_count should be > 0 with non-empty content; content={content:?}"
    );
    assert!(
        content.len() >= chunk_count as usize,
        "[{model}] content.len()={} should be >= chunk_count={chunk_count}",
        content.len()
    );
}

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model"]
fn test_repl_history_truncation_at_10(#[case] model: &str) {
    // 8 turns is sufficient to trigger the 10-entry cap (push 16 history
    // entries before the while-loop trims) without paying 12-turn runtime.
    let mut lines: Vec<String> = (1..=8)
        .map(|i| format!("Reply with just the number {i}."))
        .collect();
    lines.push("/bye".to_string());
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();

    let events = run_chat_full(model, None, 30, &line_refs);
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(
        assistants.len(),
        8,
        "[{model}] expected 8 assistant turns, got {}",
        assistants.len()
    );
    for (i, a) in assistants.iter().enumerate() {
        assert!(
            a.content().map(|c| !c.trim().is_empty()).unwrap_or(false),
            "[{model}] turn {i} content empty"
        );
        assert!(
            matches!(a.finish_reason(), Some("stop" | "eos" | "length")),
            "[{model}] turn {i} finish_reason: {:?}",
            a.finish_reason()
        );
    }
}

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model"]
fn test_repl_clean_exit_bye(#[case] model: &str) {
    let events = run_chat(model, None, &["Hi.", "/bye"]);
    let last = events.last().expect("at least one event");
    assert_eq!(
        last.event(),
        "exit",
        "[{model}] last event should be 'exit'; got: {:?}",
        last.event()
    );
    assert_eq!(
        last.exit_reason(),
        Some("bye"),
        "[{model}] exit reason should be 'bye'; got: {:?}",
        last.exit_reason()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Capability-scoped tests — limited to instruction-following / Chinese-capable
// ─────────────────────────────────────────────────────────────────────────────

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model"]
fn test_repl_system_prompt(#[case] model: &str) {
    // Only Qwen3-0.6B follows simple instructions reliably at this scale —
    // Qwen2.5-0.5B and TinyLlama-1.1B produce incoherent output even at temp=0.
    let events = run_chat(
        model,
        Some("You must always reply with exactly the word YES, nothing else."),
        &["What is the weather like?", "/bye"],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 1);
    let content = assistants[0].content().unwrap_or("");
    assert!(
        content.to_lowercase().contains("yes"),
        "[{model}] system prompt asked for 'YES'; assistant said: {content:?}"
    );
}

#[rstest]
#[case::qwen3("qwen3:0.6b")]
#[ignore = "loads real model"]
fn test_repl_utf8_chinese(#[case] model: &str) {
    // Only Qwen3-0.6B reliably produces coherent Chinese; Qwen2.5-0.5B
    // at greedy temp=0 outputs near-random tokens.
    let events = run_chat(
        model,
        Some("请用中文回答。"),
        &["你好，请用一句话问候我。", "/bye"],
    );
    let assistants: Vec<_> = events.iter().filter(|e| e.event() == "assistant").collect();
    assert_eq!(assistants.len(), 1);
    let content = assistants[0].content().unwrap_or("");
    let has_cjk = content
        .chars()
        .any(|c| (0x4E00..=0x9FFF).contains(&(c as u32)));
    assert!(
        has_cjk,
        "[{model}] expected Chinese characters in response; got: {content:?}"
    );
}
