//! Tokenizer-aware HTTP bench client — emits the canonical schema
//! defined in `crates/ferrum-bench-core` and gated by
//! `docs/bench/PLAYBOOK.md` § 7.
//!
//! Scenarios (PLAYBOOK § 2):
//!   - **Closed-loop** — `--concurrency K`, K workers in tight send→wait
//!     loop. Headline = throughput. The "capacity knee" measurement.
//!   - **Concurrency sweep** — `--concurrency-sweep 1,4,16,32`, runs N
//!     closed-loop cells back-to-back to find the knee.
//!   - **Open-loop** — `--request-rate R`, Poisson(R) arrivals. The
//!     ONLY scenario in which goodput is meaningful (§ 0.4).
//!
//! Each cell runs `--n-repeats` independent times; the per-run percentiles
//! are aggregated with mean + sample stddev + Student-t 95% CI half-width.
//! Cells where `n_repeats < 3` emit `mean` only (PLAYBOOK § 0.4 contract).

use clap::Args;
use colored::*;
use ferrum_bench_core::{
    arrivals::poisson_arrival_times, compute_metrics, BenchReport, Env, ItlEvidenceSource,
    OutputTokenCountSource, QualityIssueCounts, RequestItlEvidence, RequestRecord, RunRecord,
    Scenario, Slo, TokenLengthStats, WarmupSummary,
};
use ferrum_types::Result;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio_stream::StreamExt;

use crate::config::CliConfig;

#[derive(Args, Clone)]
pub struct BenchServeCommand {
    /// Base URL of the ferrum (or other OpenAI-compatible) server.
    #[arg(long)]
    pub base_url: String,

    /// Model identifier sent in request body's `model` field.
    /// Use the local path so `vllm bench serve` numbers compare 1:1.
    #[arg(long)]
    pub model: String,

    /// Path to the model directory containing `tokenizer.json`. Used to
    /// generate exact-length random token sequences.
    #[arg(long)]
    pub tokenizer: PathBuf,

    // ─── Workload selection (pick one mode) ────────────────────────
    /// Closed-loop concurrency (single cell). Default when no other
    /// mode is given. Alias: `--max-concurrency` (legacy vLLM naming).
    #[arg(long, default_value_t = 32, alias = "max-concurrency")]
    pub concurrency: u32,

    /// Closed-loop concurrency sweep. Overrides `--concurrency`. E.g.
    /// `--concurrency-sweep 1,4,16,32` runs four closed-loop cells.
    #[arg(long, value_delimiter = ',')]
    pub concurrency_sweep: Vec<u32>,

    /// Open-loop arrival rate (req/s, Poisson). When set, overrides
    /// the closed-loop modes — this is the goodput-relevant scenario.
    #[arg(long)]
    pub request_rate: Option<f64>,

    // ─── Dataset ───────────────────────────────────────────────────
    /// Dataset: `random` (tokenizer-aware), `sharegpt` (load from JSONL),
    /// `shared-prefix` (1024-tok shared prefix + unique suffix).
    /// PLAYBOOK § 2 Scenario A (sharegpt) / Scenario C (shared-prefix).
    #[arg(long, default_value = "random")]
    pub dataset: String,

    /// Number of *tokens* per random prompt (`--dataset random` only).
    #[arg(long, default_value_t = 256)]
    pub random_input_len: usize,

    /// Max output tokens per request.
    #[arg(long, default_value_t = 128)]
    pub random_output_len: usize,

    /// Send vLLM-compatible `ignore_eos=true` so fixed-output benchmark
    /// requests run until `max_tokens` instead of stopping on model EOS.
    #[arg(long)]
    pub ignore_eos: bool,

    /// Typed chat-template thinking control. Omitted from payloads unless set.
    #[arg(long, action = clap::ArgAction::Set)]
    pub enable_thinking: Option<bool>,

    /// Path to a ShareGPT-format JSONL file (`--dataset sharegpt`).
    /// Each line should be a `{"conversations": [{"from": "...", "value":
    /// "..."}, ...]}` object (HF anon8231489123/ShareGPT_Vicuna format).
    #[arg(long)]
    pub sharegpt_path: Option<PathBuf>,

    /// Shared prefix length in *tokens* (`--dataset shared-prefix` only).
    #[arg(long, default_value_t = 1024)]
    pub shared_prefix_len: usize,

    /// Per-request unique suffix length in *tokens* (`--dataset shared-prefix`).
    #[arg(long, default_value_t = 64)]
    pub shared_suffix_len: usize,

    // ─── Run shape ─────────────────────────────────────────────────
    /// Total prompts sent per run (warmup is counted separately).
    #[arg(long, default_value_t = 100)]
    pub num_prompts: u32,

    /// Warmup requests sent before measurement begins each run.
    /// Discarded from the metrics; PLAYBOOK § 0.3 mandates ≥ 10 for
    /// committed reports.
    #[arg(long, default_value_t = 10)]
    pub warmup_requests: u32,

    /// Independent repeats per cell. PLAYBOOK § 0.4: ≥ 3 unlocks
    /// stddev + CI95; n < 3 emits mean only.
    #[arg(long, default_value_t = 1)]
    pub n_repeats: u32,

    /// SLO triple for goodput. Format: `ttft:500 tpot:50 e2el:30000`
    /// (or comma-separated). Goodput is reported only when all three
    /// are set.
    #[arg(long, value_parser = parse_slo)]
    pub goodput: Option<Slo>,

    /// Per-request HTTP timeout in seconds.
    #[arg(long, default_value_t = 600.0)]
    pub timeout: f64,

    /// Exit non-zero when any measured request errors. Warmup failures always fail.
    /// Release gates must set this.
    #[arg(long)]
    pub fail_on_error: bool,

    /// Maximum measured request error rate allowed when error enforcement is active.
    #[arg(long)]
    pub max_error_rate: Option<f64>,

    /// Require n_repeats >= 3 so reports include CI/stddev evidence.
    #[arg(long)]
    pub require_ci: bool,

    /// Deterministic prompt-generation seed. Repeat i uses a stable derivation.
    #[arg(long)]
    pub seed: Option<u64>,

    // ─── Output ────────────────────────────────────────────────────
    /// Output format: `json` (BenchReport), `jsonl` (append one
    /// `BenchReport` per line — used by `scripts/compare-commits.sh`),
    /// `md` (human-readable markdown).
    #[arg(long, default_value = "json")]
    pub output: String,

    /// Output file path. For `jsonl`, the file is opened append-mode so
    /// repeated invocations against the same DB accumulate rows.
    /// Alias: `--result-file` (legacy).
    #[arg(long, alias = "result-file")]
    pub out: Option<PathBuf>,

    // ─── Env / parity ──────────────────────────────────────────────
    /// Override `env.hw_id` (defaults to auto-detected).
    #[arg(long)]
    pub hw_id: Option<String>,

    /// Override `env.commit_sha` (defaults to `git rev-parse --short HEAD`).
    #[arg(long)]
    pub commit_sha: Option<String>,

    /// Tag string written into the report's `model` field's suffix —
    /// useful for the `bench_vs_vllm.sh` script which tags ferrum/vllm.
    #[arg(long)]
    pub tag: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────
// SLO parser
// ─────────────────────────────────────────────────────────────────────

pub(super) fn parse_slo(s: &str) -> std::result::Result<Slo, String> {
    let mut ttft: Option<f64> = None;
    let mut tpot: Option<f64> = None;
    let mut e2el: Option<f64> = None;
    for tok in s.split(|c: char| c == ',' || c.is_whitespace()) {
        if tok.is_empty() {
            continue;
        }
        let (k, v) = tok
            .split_once(':')
            .ok_or_else(|| format!("bad SLO token '{tok}', expected key:value"))?;
        let v: f64 = v.parse().map_err(|e| format!("bad SLO value '{v}': {e}"))?;
        if !v.is_finite() || v <= 0.0 {
            return Err(format!(
                "bad SLO value '{v}': expected a positive finite number"
            ));
        }
        match k {
            "ttft" => ttft = Some(v),
            "tpot" => tpot = Some(v),
            "e2el" | "e2e" => e2el = Some(v),
            other => return Err(format!("unknown SLO key '{other}'")),
        }
    }
    Ok(Slo {
        ttft_p99_ms: ttft.ok_or("missing ttft in --goodput")?,
        tpot_p99_ms: tpot.ok_or("missing tpot in --goodput")?,
        e2e_p99_ms: e2el.ok_or("missing e2el in --goodput")?,
    })
}

// ─────────────────────────────────────────────────────────────────────
// OpenAI SSE chunk types
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Option<Vec<OpenAiStreamChoice>>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: Option<OpenAiStreamDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamDelta {
    content: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    completion_tokens: Option<u32>,
}

#[derive(Clone)]
struct PromptCase {
    text: String,
    input_tokens: u32,
}

// ─────────────────────────────────────────────────────────────────────
// Single-request streamer
// ─────────────────────────────────────────────────────────────────────

async fn stream_one(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    prompt: PromptCase,
    max_tokens: usize,
    ignore_eos: bool,
    enable_thinking: Option<bool>,
    timeout_s: f64,
) -> RequestRecord {
    let body = chat_completion_body(model, &prompt.text, max_tokens, ignore_eos, enable_thinking);
    let start = Instant::now();
    let mut state = StreamState::new(start, prompt.input_tokens);

    let resp = match client
        .post(format!("{}/v1/chat/completions", base_url))
        .json(&body)
        .timeout(Duration::from_secs_f64(timeout_s))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[err] post: {}", e);
            let mut quality_issues = QualityIssueCounts::default();
            quality_issues.malformed_stream = 1;
            return failed_record(prompt.input_tokens, start, quality_issues);
        }
    };
    if !resp.status().is_success() {
        let status = resp.status();
        let txt = resp.text().await.unwrap_or_default();
        eprintln!("[err] http {}: {}", status, &txt[..txt.len().min(200)]);
        let mut quality_issues = QualityIssueCounts::default();
        if status.as_u16() == 500 {
            quality_issues.http_500 = 1;
        }
        if looks_like_panic(&txt) {
            quality_issues.panic = 1;
        }
        return failed_record(prompt.input_tokens, start, quality_issues);
    }

    let mut stream = resp.bytes_stream();
    let mut sse = SseLineBuffer::default();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[err] stream: {}", e);
                state.stream_error = Some(e.to_string());
                state.quality_issues.malformed_stream = 1;
                break;
            }
        };
        let before_output_events = state.output_delta_events;
        sse.push(&chunk, &mut state);
        state.note_transport_chunk(
            state
                .output_delta_events
                .saturating_sub(before_output_events),
        );
    }
    sse.finish(&mut state);
    state.finish()
}

#[derive(Default)]
struct SseLineBuffer {
    pending: Vec<u8>,
}

impl SseLineBuffer {
    fn push(&mut self, chunk: &[u8], state: &mut StreamState) {
        self.pending.extend_from_slice(chunk);
        let mut consumed = 0;
        while let Some(relative_newline) = self.pending[consumed..]
            .iter()
            .position(|byte| *byte == b'\n')
        {
            let newline = consumed + relative_newline;
            let line = self.pending[consumed..newline].to_vec();
            consumed = newline + 1;
            Self::process_line(&line, state);
        }
        if consumed > 0 {
            self.pending.drain(..consumed);
        }
    }

    fn finish(&mut self, state: &mut StreamState) {
        if !self.pending.is_empty() {
            let line = std::mem::take(&mut self.pending);
            Self::process_line(&line, state);
        }
    }

    fn process_line(raw_line: &[u8], state: &mut StreamState) {
        let raw_line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        let line = match std::str::from_utf8(raw_line) {
            Ok(line) => line.trim(),
            Err(error) => {
                let message = format!("invalid UTF-8 in SSE line: {error}");
                eprintln!("[err] {message}");
                state.stream_error = Some(message);
                state.quality_issues.malformed_stream = 1;
                state.quality_issues.bad_output = 1;
                return;
            }
        };
        let Some(payload) = line.strip_prefix("data:").map(str::trim) else {
            return;
        };
        if payload == "[DONE]" {
            state.done_count = state
                .done_count
                .checked_add(1)
                .expect("SSE done count overflow");
            if state.done_count > 1 {
                state.quality_issues.duplicate_done = 1;
            }
            return;
        }
        if let Err(error) = state.handle_payload(payload) {
            eprintln!("[err] malformed sse json: {error}");
            state.stream_error = Some(error);
            state.quality_issues.malformed_stream = 1;
        }
    }
}

fn chat_completion_body(
    model: &str,
    prompt_text: &str,
    max_tokens: usize,
    ignore_eos: bool,
    enable_thinking: Option<bool>,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": 0.0,
    });
    if let Some(enable_thinking) = enable_thinking {
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": enable_thinking});
    }
    if ignore_eos {
        body["ignore_eos"] = serde_json::json!(true);
    }
    body
}

fn failed_record(
    input_tokens: u32,
    start: Instant,
    quality_issues: QualityIssueCounts,
) -> RequestRecord {
    RequestRecord {
        success: false,
        ttft_ms: 0.0,
        e2e_ms: start.elapsed().as_secs_f64() * 1000.0,
        input_tokens,
        output_tokens: 0,
        output_token_count_source: OutputTokenCountSource::None,
        itl_evidence: RequestItlEvidence::failed(ItlEvidenceSource::SseDeltaEvents),
        quality_issues,
        itl_ms: vec![],
    }
}

fn join_failed_record(input_tokens: u32) -> RequestRecord {
    let mut quality_issues = QualityIssueCounts::default();
    quality_issues.panic = 1;
    RequestRecord {
        success: false,
        ttft_ms: 0.0,
        e2e_ms: 0.0,
        input_tokens,
        output_tokens: 0,
        output_token_count_source: OutputTokenCountSource::None,
        itl_evidence: RequestItlEvidence::failed(ItlEvidenceSource::SseDeltaEvents),
        quality_issues,
        itl_ms: vec![],
    }
}

async fn collect_measured_handles(
    handles: Vec<(u32, tokio::task::JoinHandle<RequestRecord>)>,
) -> Vec<RequestRecord> {
    let mut records = Vec::with_capacity(handles.len());
    for (input_tokens, handle) in handles {
        match handle.await {
            Ok(record) => records.push(record),
            Err(error) => {
                eprintln!("[err] measured request task: {error}");
                records.push(join_failed_record(input_tokens));
            }
        }
    }
    records
}

struct StreamState {
    start: Instant,
    input_tokens: u32,
    first_token_time: Option<Instant>,
    last_token_time: Option<Instant>,
    output_delta_events: u32,
    usage_completion_tokens: Option<u32>,
    itl_ms: Vec<f64>,
    transport_coalesced_output_chunks: u32,
    done_count: u32,
    stream_error: Option<String>,
    quality_issues: QualityIssueCounts,
}

impl StreamState {
    fn new(start: Instant, input_tokens: u32) -> Self {
        Self {
            start,
            input_tokens,
            first_token_time: None,
            last_token_time: None,
            output_delta_events: 0,
            usage_completion_tokens: None,
            itl_ms: Vec::new(),
            transport_coalesced_output_chunks: 0,
            done_count: 0,
            stream_error: None,
            quality_issues: QualityIssueCounts::default(),
        }
    }

    fn handle_payload(&mut self, payload: &str) -> std::result::Result<(), String> {
        let chunk: OpenAiStreamChunk =
            serde_json::from_str(payload).map_err(|e| format!("{e}: {payload}"))?;
        if let Some(usage) = chunk.usage {
            if let Some(tokens) = usage.completion_tokens {
                self.usage_completion_tokens = Some(tokens);
            }
        }
        if let Some(choices) = chunk.choices {
            if let Some(first) = choices.into_iter().next() {
                if let Some(delta) = first.delta {
                    if let Some(text) = first_non_empty_delta_text(&delta) {
                        let now = Instant::now();
                        if self.first_token_time.is_none() {
                            self.first_token_time = Some(now);
                        } else if let Some(prev) = self.last_token_time {
                            self.itl_ms.push((now - prev).as_secs_f64() * 1000.0);
                        }
                        self.last_token_time = Some(now);
                        self.output_delta_events = self
                            .output_delta_events
                            .checked_add(1)
                            .expect("stream output event count overflow");
                        if let Some(reason) = bad_output_reason(text) {
                            eprintln!(
                                "[err] bad output {reason}: {}",
                                clipped_debug_text(text, 160)
                            );
                            self.quality_issues.bad_output = 1;
                        }
                        if looks_like_panic(text) {
                            self.quality_issues.panic = 1;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn note_transport_chunk(&mut self, output_events: u32) {
        if output_events > 1 {
            self.transport_coalesced_output_chunks = self
                .transport_coalesced_output_chunks
                .checked_add(1)
                .expect("coalesced output transport chunk count overflow");
        }
    }

    fn finish(mut self) -> RequestRecord {
        let (output_tokens, source) = match self.usage_completion_tokens {
            Some(tokens) => (tokens, OutputTokenCountSource::Usage),
            None if self.output_delta_events > 0 => (
                self.output_delta_events,
                OutputTokenCountSource::StreamChunks,
            ),
            None => (0, OutputTokenCountSource::None),
        };
        if self.done_count == 0 {
            self.quality_issues.missing_done = 1;
        } else if self.done_count > 1 {
            self.quality_issues.duplicate_done = 1;
        }
        if output_tokens == 0 {
            self.quality_issues.zero_output_tokens = 1;
        }
        let e2e_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        let ttft_ms = self
            .first_token_time
            .map(|t| t.duration_since(self.start).as_secs_f64() * 1000.0)
            .unwrap_or(e2e_ms);
        let success = self.done_count == 1
            && output_tokens > 0
            && self.stream_error.is_none()
            && self.quality_issues.request_error_count() == 0;
        let observed_intervals =
            u32::try_from(self.itl_ms.len()).expect("stream ITL interval count overflow");
        let itl_evidence = RequestItlEvidence::sse(
            success,
            self.output_delta_events,
            self.usage_completion_tokens,
            observed_intervals,
            self.transport_coalesced_output_chunks,
        );
        RequestRecord {
            success,
            ttft_ms,
            e2e_ms,
            input_tokens: self.input_tokens,
            output_tokens,
            output_token_count_source: source,
            itl_evidence,
            quality_issues: self.quality_issues,
            itl_ms: self.itl_ms,
        }
    }
}

fn first_non_empty_delta_text(delta: &OpenAiStreamDelta) -> Option<&str> {
    delta
        .content
        .as_deref()
        .filter(|s| !s.is_empty())
        .or_else(|| delta.reasoning.as_deref().filter(|s| !s.is_empty()))
        .or_else(|| delta.reasoning_content.as_deref().filter(|s| !s.is_empty()))
}

#[cfg(test)]
fn has_bad_output_text(text: &str) -> bool {
    bad_output_reason(text).is_some()
}

fn bad_output_reason(text: &str) -> Option<&'static str> {
    const BAD_FRAGMENTS: &[(&str, &str)] = &[
        ("<unk>", "reserved-token"),
        ("[PAD", "reserved-token"),
        ("<pad>", "reserved-token"),
        ("<|endoftext|>", "reserved-token"),
        ("<|im_start|>", "reserved-token"),
        ("<|im_end|>", "reserved-token"),
        ("<|reserved_special_token", "reserved-token"),
        ("\u{fffd}", "invalid-utf8"),
    ];
    for (fragment, reason) in BAD_FRAGMENTS {
        if text.contains(fragment) {
            return Some(reason);
        }
    }
    contains_mojibake_sequence(text).then_some("mojibake")
}

fn contains_mojibake_sequence(text: &str) -> bool {
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            // Common UTF-8-as-Latin-1/Windows-1252 mojibake starts. Treat
            // standalone lead characters as ordinary text; require the
            // following non-ASCII continuation to avoid false positives from
            // tokenizer byte-fallback fragments in random benchmark prompts.
            '\u{00c2}' | '\u{00c3}' => {
                if chars.peek().is_some_and(|next| !next.is_ascii()) {
                    return true;
                }
            }
            // Most smart quote, dash, ellipsis, and bullet mojibake starts
            // with "â€" after UTF-8 bytes are decoded through Windows-1252.
            '\u{00e2}' => {
                if chars.peek().is_some_and(|next| *next == '\u{20ac}') {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn clipped_debug_text(text: &str, max_chars: usize) -> String {
    text.escape_debug().take(max_chars).collect()
}

fn looks_like_panic(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("panicked at") || lower.contains("thread '") && lower.contains("panicked")
}

// ─────────────────────────────────────────────────────────────────────
// Dataset generation
// ─────────────────────────────────────────────────────────────────────

/// Draw a random token sequence of exact length `n_tokens`.
fn gen_random_prompt(
    tok: &tokenizers::Tokenizer,
    n_tokens: usize,
    rng: &mut (impl Rng + ?Sized),
) -> String {
    if let Some(text) = gen_random_prompt_target_len(tok, n_tokens, rng) {
        return text;
    }
    // Last-resort fallback for tokenizers whose decode/encode roundtrip is
    // hostile to arbitrary ids. Keep the old behavior rather than failing the
    // whole benchmark; the JSON report still records actual token lengths.
    let vocab_size = tok.get_vocab_size(false) as u32;
    let lo: u32 = 256.min(vocab_size.saturating_sub(1));
    let hi: u32 = vocab_size.saturating_sub(1);
    let ids: Vec<u32> = (0..n_tokens).map(|_| rng.random_range(lo..=hi)).collect();
    tok.decode(&ids, false)
        .unwrap_or_else(|_| "hello world ".repeat(n_tokens / 2))
}

fn gen_random_prompt_target_len(
    tok: &tokenizers::Tokenizer,
    target_len: usize,
    rng: &mut (impl Rng + ?Sized),
) -> Option<String> {
    if target_len == 0 {
        return Some(String::new());
    }
    let vocab_size = tok.get_vocab_size(false) as u32;
    let lo: u32 = 256.min(vocab_size.saturating_sub(1));
    let hi: u32 = vocab_size.saturating_sub(1);
    if hi < lo {
        return None;
    }

    let mut sample_len = target_len;
    let mut best_under: Option<(usize, String)> = None;
    let mut best_any: Option<(usize, String)> = None;
    for _ in 0..64 {
        let ids: Vec<u32> = (0..sample_len).map(|_| rng.random_range(lo..=hi)).collect();
        let text = match tok.decode(&ids, false) {
            Ok(text) if !text.is_empty() => text,
            _ => continue,
        };
        let len = match token_count(tok, &text) {
            Some(len) => len,
            None => continue,
        };
        if len == target_len {
            return Some(text);
        }
        let delta = len.abs_diff(target_len);
        if best_any
            .as_ref()
            .map(|(best_len, _)| delta < best_len.abs_diff(target_len))
            .unwrap_or(true)
        {
            best_any = Some((len, text.clone()));
        }
        if len < target_len
            && best_under
                .as_ref()
                .map(|(best_len, _)| len > *best_len)
                .unwrap_or(true)
        {
            best_under = Some((len, text));
        }

        // Decoding arbitrary tokenizer ids often changes the token count on
        // re-encode. Move the next draw in the direction of the target.
        sample_len = if len > target_len {
            sample_len.saturating_sub(len - target_len).max(1)
        } else {
            sample_len + (target_len - len).max(1)
        };
    }

    if let Some((len, text)) = best_under {
        return fill_random_prompt_to_len(tok, text, len, target_len, rng, lo, hi);
    }
    best_any.map(|(_, text)| text)
}

fn token_count(tok: &tokenizers::Tokenizer, text: &str) -> Option<usize> {
    tok.encode(text, false).ok().map(|enc| enc.len())
}

fn fill_random_prompt_to_len(
    tok: &tokenizers::Tokenizer,
    mut text: String,
    mut len: usize,
    target_len: usize,
    rng: &mut (impl Rng + ?Sized),
    lo: u32,
    hi: u32,
) -> Option<String> {
    for _ in 0..target_len.saturating_mul(8).max(16) {
        if len == target_len {
            return Some(text);
        }
        let next = random_one_token_extension(tok, &text, len, rng, lo, hi)?;
        text = next.0;
        len = next.1;
    }
    (len == target_len).then_some(text)
}

fn random_one_token_extension(
    tok: &tokenizers::Tokenizer,
    base: &str,
    base_len: usize,
    rng: &mut (impl Rng + ?Sized),
    lo: u32,
    hi: u32,
) -> Option<(String, usize)> {
    // Try random vocab pieces first so prompts remain high-entropy.
    for _ in 0..128 {
        let id = rng.random_range(lo..=hi);
        if let Some(candidate) = append_decoded_piece(tok, base, base_len, id) {
            return Some(candidate);
        }
    }
    // Deterministic fallbacks cover tokenizers where most high vocab ids are
    // byte-fallback fragments that merge or expand at text boundaries.
    for piece in [" x", " y", " z", ".", ",", "\n"] {
        let candidate = format!("{base}{piece}");
        if token_count(tok, &candidate) == Some(base_len + 1) {
            return Some((candidate, base_len + 1));
        }
    }
    None
}

fn append_decoded_piece(
    tok: &tokenizers::Tokenizer,
    base: &str,
    base_len: usize,
    id: u32,
) -> Option<(String, usize)> {
    let piece = tok.decode(&[id], false).ok()?;
    if piece.is_empty() {
        return None;
    }
    let candidate = format!("{base}{piece}");
    (token_count(tok, &candidate) == Some(base_len + 1)).then_some((candidate, base_len + 1))
}

fn build_prompts(
    cmd: &BenchServeCommand,
    tok: &tokenizers::Tokenizer,
    rng: &mut (impl Rng + ?Sized),
    count: usize,
) -> Result<Vec<PromptCase>> {
    match cmd.dataset.as_str() {
        "random" => (0..count)
            .map(|_| {
                let text = gen_random_prompt(tok, cmd.random_input_len, rng);
                prompt_case(tok, text)
            })
            .collect(),
        "shared-prefix" => gen_shared_prefix_prompts(
            tok,
            count,
            cmd.shared_prefix_len,
            cmd.shared_suffix_len,
            rng,
        ),
        "sharegpt" => {
            let p = cmd.sharegpt_path.as_ref().ok_or_else(|| {
                ferrum_types::FerrumError::model("--dataset sharegpt requires --sharegpt-path PATH")
            })?;
            load_sharegpt_prompts(p, tok, count, rng)
        }
        other => Err(ferrum_types::FerrumError::model(format!(
            "unknown --dataset '{}': allowed values are random, sharegpt, shared-prefix",
            other
        ))),
    }
}

fn prompt_case(tok: &tokenizers::Tokenizer, text: String) -> Result<PromptCase> {
    let encoding = tok
        .encode(text.as_str(), false)
        .map_err(|e| ferrum_types::FerrumError::model(format!("tokenize generated prompt: {e}")))?;
    let input_tokens = u32::try_from(encoding.len()).map_err(|_| {
        ferrum_types::FerrumError::model("generated prompt token count exceeds report capacity")
    })?;
    Ok(PromptCase { text, input_tokens })
}

/// Generate `count` prompts that all share a 1024-token (or whatever
/// `shared_prefix_len` is) prefix, with a unique random suffix per
/// request. The shared prefix is sampled ONCE from the tokenizer's
/// mid-vocab range and decoded into a UTF-8 string the server can
/// re-tokenize back to ~the same length.
///
/// Used by PLAYBOOK § 2 Scenario C (prefix-cache thundering herd):
/// closed-loop ShareGPT can't reproduce this because each ShareGPT
/// conversation has a different prefix.
fn gen_shared_prefix_prompts(
    tok: &tokenizers::Tokenizer,
    count: usize,
    prefix_len: usize,
    suffix_len: usize,
    rng: &mut (impl Rng + ?Sized),
) -> Result<Vec<PromptCase>> {
    let vocab_size = tok.get_vocab_size(false) as u32;
    let lo: u32 = 256.min(vocab_size.saturating_sub(1));
    let hi: u32 = vocab_size.saturating_sub(1);
    // Shared prefix sampled once.
    let prefix_ids: Vec<u32> = (0..prefix_len).map(|_| rng.random_range(lo..=hi)).collect();
    let prefix = tok
        .decode(&prefix_ids, false)
        .unwrap_or_else(|_| "hello world ".repeat(prefix_len / 2));
    (0..count)
        .map(|_| {
            let suffix_ids: Vec<u32> = (0..suffix_len).map(|_| rng.random_range(lo..=hi)).collect();
            let suffix = tok
                .decode(&suffix_ids, false)
                .unwrap_or_else(|_| "x ".repeat(suffix_len / 2));
            // Insert a newline so the prefix is a clear boundary — helps
            // server-side prefix-cache hashing key on the same prefix
            // even when suffixes differ.
            prompt_case(tok, format!("{prefix}\n{suffix}"))
        })
        .collect()
}

/// Load up to `count` user-turn prompts from a ShareGPT-format JSONL.
///
/// Accepts either:
///   - HF `anon8231489123/ShareGPT_Vicuna` format: each line is
///     `{"id": "...", "conversations": [{"from": "human"/"gpt", "value": "..."}]}`
///   - vLLM-style: `{"input": "..."}` per line
///
/// Picks the first `human` turn per conversation (matches vLLM's
/// benchmark_serving.py heuristic). If `count` < records available,
/// randomly samples; if `count` > available, cycles with replacement.
fn load_sharegpt_prompts(
    path: &std::path::Path,
    tok: &tokenizers::Tokenizer,
    count: usize,
    rng: &mut (impl Rng + ?Sized),
) -> Result<Vec<PromptCase>> {
    use std::io::BufRead;
    let f = std::fs::File::open(path).map_err(|e| {
        ferrum_types::FerrumError::model(format!("open sharegpt {}: {e}", path.display()))
    })?;
    let mut prompts: Vec<String> = Vec::new();
    for (idx, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| {
            ferrum_types::FerrumError::model(format!("read line {idx} of {}: {e}", path.display()))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[warn] sharegpt line {idx}: skip (parse error: {e})");
                continue;
            }
        };
        // Try ShareGPT-Vicuna format first.
        let prompt: Option<String> = v
            .get("conversations")
            .and_then(|c| c.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find(|t| t.get("from").and_then(|f| f.as_str()) == Some("human"))
                    .and_then(|t| {
                        t.get("value")
                            .and_then(|x| x.as_str())
                            .map(|s| s.to_string())
                    })
            })
            // Fallback: simple {"input": "..."} format.
            .or_else(|| {
                v.get("input")
                    .and_then(|s| s.as_str())
                    .map(|s| s.to_string())
            });
        if let Some(p) = prompt {
            if !p.is_empty() {
                prompts.push(p);
            }
        }
    }
    if prompts.is_empty() {
        return Err(ferrum_types::FerrumError::model(format!(
            "sharegpt {}: no usable prompts found",
            path.display()
        )));
    }
    // Sample `count` with replacement (cycles deterministically for the
    // given seed via rng).
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let idx = rng.random_range(0..prompts.len());
        out.push(prompt_case(tok, prompts[idx].clone())?);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────
// Scenario runners
// ─────────────────────────────────────────────────────────────────────

struct RunContext {
    client: Arc<reqwest::Client>,
    base_url: Arc<String>,
    model: Arc<String>,
    max_out: usize,
    ignore_eos: bool,
    enable_thinking: Option<bool>,
    timeout_s: f64,
}

fn summarize_warmup(
    expected: usize,
    records: &[RequestRecord],
    join_errors: usize,
) -> WarmupSummary {
    assert_eq!(records.len() + join_errors, expected);
    let completed = records.iter().filter(|record| record.success).count() as u32;
    let mut quality_issues = QualityIssueCounts::default();
    for record in records {
        quality_issues.add_assign(&record.quality_issues);
    }
    quality_issues.panic = quality_issues
        .panic
        .checked_add(u32::try_from(join_errors).expect("warmup join error count overflow"))
        .expect("warmup panic count overflow");
    let expected = u32::try_from(expected).expect("warmup expected count overflow");
    WarmupSummary {
        expected,
        completed,
        errored: expected
            .checked_sub(completed)
            .expect("warmup completed count exceeds expected"),
        quality_issues,
    }
}

/// Closed-loop: K workers in a tight loop. Warmup prompts run sequentially
/// at full concurrency (just to load caches), then the measurement window
/// begins.
async fn run_closed_loop(
    ctx: &RunContext,
    prompts: Vec<PromptCase>,
    warmup_requests: u32,
    concurrency: u32,
) -> RunRecord {
    let n_warmup = warmup_requests as usize;
    let total = prompts.len();
    assert!(
        total > n_warmup,
        "num_prompts ({total}) must exceed warmup_requests ({n_warmup})"
    );

    // Warmup window — fire and discard.
    let warmup = {
        let sem = Arc::new(Semaphore::new(concurrency as usize));
        let mut handles = Vec::new();
        for prompt in prompts.iter().take(n_warmup) {
            let permit = sem.clone().acquire_owned().await.expect("semaphore");
            let ctx_c = ctx.clone_inner();
            let p = prompt.clone();
            handles.push(tokio::spawn(async move {
                let _g = permit;
                stream_one(
                    &ctx_c.client,
                    &ctx_c.base_url,
                    &ctx_c.model,
                    p,
                    ctx_c.max_out,
                    ctx_c.ignore_eos,
                    ctx_c.enable_thinking,
                    ctx_c.timeout_s,
                )
                .await
            }));
        }
        let mut records = Vec::with_capacity(n_warmup);
        let mut join_errors = 0;
        for handle in handles {
            match handle.await {
                Ok(record) => records.push(record),
                Err(_) => join_errors += 1,
            }
        }
        summarize_warmup(n_warmup, &records, join_errors)
    };

    // Measurement window.
    let sem = Arc::new(Semaphore::new(concurrency as usize));
    let start = Instant::now();
    let mut handles = Vec::with_capacity(total - n_warmup);
    for prompt in prompts.into_iter().skip(n_warmup) {
        let input_tokens = prompt.input_tokens;
        let permit = sem.clone().acquire_owned().await.expect("semaphore");
        let ctx_c = ctx.clone_inner();
        handles.push((
            input_tokens,
            tokio::spawn(async move {
                let _g = permit;
                stream_one(
                    &ctx_c.client,
                    &ctx_c.base_url,
                    &ctx_c.model,
                    prompt,
                    ctx_c.max_out,
                    ctx_c.ignore_eos,
                    ctx_c.enable_thinking,
                    ctx_c.timeout_s,
                )
                .await
            }),
        ));
    }
    let records = collect_measured_handles(handles).await;
    let duration_s = start.elapsed().as_secs_f64();
    RunRecord {
        records,
        expected_requests: u32::try_from(total - n_warmup)
            .expect("measured request count overflow"),
        duration_s,
        warmup,
    }
}

/// Open-loop: Poisson(rate) arrivals. The arrival schedule is fixed
/// before sending so that slow responses don't push later arrivals.
async fn run_open_loop(
    ctx: &RunContext,
    prompts: Vec<PromptCase>,
    warmup_requests: u32,
    rate: f64,
) -> RunRecord {
    let n_warmup = warmup_requests as usize;
    let total = prompts.len();
    assert!(total > n_warmup);

    // Warmup: send a few sequentially to load caches.
    let mut warmup_records = Vec::with_capacity(n_warmup);
    for prompt in prompts.iter().take(n_warmup) {
        warmup_records.push(
            stream_one(
                &ctx.client,
                &ctx.base_url,
                &ctx.model,
                prompt.clone(),
                ctx.max_out,
                ctx.ignore_eos,
                ctx.enable_thinking,
                ctx.timeout_s,
            )
            .await,
        );
    }
    let warmup = summarize_warmup(n_warmup, &warmup_records, 0);

    // Pre-compute arrival schedule (re-zeroed after warmup).
    let mut rng = rand::rng();
    let measurement_count = total - n_warmup;
    let schedule = poisson_arrival_times(rate, measurement_count, &mut rng);

    let start = Instant::now();
    let mut handles = Vec::with_capacity(measurement_count);
    for (i, prompt) in prompts.into_iter().skip(n_warmup).enumerate() {
        let target = schedule[i];
        let now = start.elapsed().as_secs_f64();
        if target > now {
            tokio::time::sleep(Duration::from_secs_f64(target - now)).await;
        }
        let ctx_c = ctx.clone_inner();
        let input_tokens = prompt.input_tokens;
        handles.push((
            input_tokens,
            tokio::spawn(async move {
                stream_one(
                    &ctx_c.client,
                    &ctx_c.base_url,
                    &ctx_c.model,
                    prompt,
                    ctx_c.max_out,
                    ctx_c.ignore_eos,
                    ctx_c.enable_thinking,
                    ctx_c.timeout_s,
                )
                .await
            }),
        ));
    }
    let records = collect_measured_handles(handles).await;
    let duration_s = start.elapsed().as_secs_f64();
    RunRecord {
        records,
        expected_requests: u32::try_from(measurement_count)
            .expect("measured request count overflow"),
        duration_s,
        warmup,
    }
}

impl RunContext {
    fn clone_inner(&self) -> Self {
        Self {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            max_out: self.max_out,
            ignore_eos: self.ignore_eos,
            enable_thinking: self.enable_thinking,
            timeout_s: self.timeout_s,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Env construction
// ─────────────────────────────────────────────────────────────────────

fn build_env(cmd: &BenchServeCommand, features: Vec<String>) -> Env {
    let commit_sha = cmd
        .commit_sha
        .clone()
        .or_else(|| {
            std::process::Command::new("git")
                .args(["rev-parse", "--short", "HEAD"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    let mut env = Env::capture_minimal(commit_sha, features);
    if let Some(hw) = cmd.hw_id.clone() {
        env.hw_id = hw;
    }
    env
}

fn detect_features() -> Vec<String> {
    let mut v = Vec::new();
    #[cfg(feature = "metal")]
    v.push("metal".to_string());
    #[cfg(feature = "cuda")]
    v.push("cuda".to_string());
    #[cfg(feature = "vllm-marlin")]
    v.push("vllm-marlin".to_string());
    #[cfg(feature = "vllm-moe-marlin")]
    v.push("vllm-moe-marlin".to_string());
    #[cfg(feature = "vllm-paged-attn-v2")]
    v.push("vllm-paged-attn-v2".to_string());
    #[cfg(feature = "triton-kernels")]
    v.push("triton-kernels".to_string());
    v.sort();
    v.dedup();
    v
}

// ─────────────────────────────────────────────────────────────────────
// Cell execution + multi-cell driver
// ─────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Cell {
    Closed(u32),
    Open(f64),
}

async fn execute_cell(
    cmd: &BenchServeCommand,
    ctx: &RunContext,
    cell: Cell,
) -> Result<BenchReport> {
    // Backend tag is a build-time fact, not a runtime env-var check. The earlier
    // `CUDA_VISIBLE_DEVICES.is_ok()` heuristic silently mistagged every CUDA-built
    // run as "cpu" when that env var was unset, polluting bench artifacts.
    let backend = if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "metal") {
        "metal"
    } else {
        "cpu"
    };

    let tokenizer_path = cmd.tokenizer.join("tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        ferrum_types::FerrumError::model(format!(
            "Load tokenizer at {}: {}",
            tokenizer_path.display(),
            e
        ))
    })?;
    let total_prompts_u32 = cmd
        .num_prompts
        .checked_add(cmd.warmup_requests)
        .ok_or_else(|| {
            ferrum_types::FerrumError::model("num_prompts + warmup_requests overflow")
        })?;
    let total_prompts = usize::try_from(total_prompts_u32)
        .map_err(|_| ferrum_types::FerrumError::model("prompt count exceeds platform capacity"))?;

    let mut runs: Vec<RunRecord> = Vec::with_capacity(cmd.n_repeats as usize);
    let mut actual_input_lengths: Vec<u32> = Vec::new();
    let mut actual_input_tokens_per_request: Vec<Vec<u32>> =
        Vec::with_capacity(cmd.n_repeats as usize);
    for repeat_idx in 0..cmd.n_repeats {
        let mut seeded_rng;
        let mut thread_rng;
        let rng: &mut dyn rand::RngCore = if let Some(seed) = cmd.seed {
            seeded_rng =
                StdRng::seed_from_u64(seed ^ ((repeat_idx as u64) << 32) ^ cell_seed(cell));
            &mut seeded_rng
        } else {
            thread_rng = rand::rng();
            &mut thread_rng
        };
        let prompts = build_prompts(cmd, &tok, rng, total_prompts)?;
        let measured_input_lengths: Vec<u32> = prompts
            .iter()
            .skip(cmd.warmup_requests as usize)
            .map(|p| p.input_tokens)
            .collect();
        actual_input_lengths.extend(measured_input_lengths.iter().copied());
        actual_input_tokens_per_request.push(measured_input_lengths);
        eprintln!(
            "{}",
            format!(
                "  cell {} — repeat {}/{} ({} prompts including {} warmup)",
                cell_label(cell),
                repeat_idx + 1,
                cmd.n_repeats,
                total_prompts,
                cmd.warmup_requests
            )
            .dimmed()
        );
        let run = match cell {
            Cell::Closed(c) => run_closed_loop(ctx, prompts, cmd.warmup_requests, c).await,
            Cell::Open(r) => run_open_loop(ctx, prompts, cmd.warmup_requests, r).await,
        };
        eprintln!(
            "    {} completed / {} errored / {:.1}s",
            run.n_completed(),
            run.n_errored(),
            run.duration_s
        );
        runs.push(run);
    }
    let requested_input_len = requested_input_len(cmd)?;
    let requested_output_len = u32::try_from(cmd.random_output_len).map_err(|_| {
        ferrum_types::FerrumError::model("random output length exceeds report capacity")
    })?;
    let actual_input_tokens = input_token_stats(&actual_input_lengths, requested_input_len);
    let token_count_source = output_token_count_source_from_runs(&runs);

    let env = build_env(cmd, detect_features());
    let slo = cmd.goodput.unwrap_or_else(Slo::unbounded);
    let model_field = match &cmd.tag {
        Some(t) => format!("{}#{}", cmd.model, t),
        None => cmd.model.clone(),
    };

    let (scenario, concurrency, request_rate) = match cell {
        Cell::Closed(c) => (Scenario::ClosedLoop, Some(c), None),
        Cell::Open(r) => (Scenario::OpenLoop, None, Some(r)),
    };

    let mut report = compute_metrics(
        model_field,
        backend.to_string(),
        scenario,
        concurrency,
        request_rate,
        requested_input_len,
        requested_output_len,
        cmd.warmup_requests,
        slo,
        runs,
        env,
    );
    report.actual_input_tokens = Some(actual_input_tokens);
    report.actual_input_tokens_per_request = Some(actual_input_tokens_per_request);
    report.output_token_count_source = Some(token_count_source);
    Ok(report)
}

fn requested_input_len(cmd: &BenchServeCommand) -> Result<u32> {
    match cmd.dataset.as_str() {
        "shared-prefix" => cmd
            .shared_prefix_len
            .checked_add(cmd.shared_suffix_len)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                ferrum_types::FerrumError::model(
                    "shared prefix + suffix length exceeds report capacity",
                )
            }),
        _ => u32::try_from(cmd.random_input_len).map_err(|_| {
            ferrum_types::FerrumError::model("random input length exceeds report capacity")
        }),
    }
}

fn input_token_stats(lengths: &[u32], requested: u32) -> TokenLengthStats {
    let min = lengths.iter().copied().min().unwrap_or(0);
    let max = lengths.iter().copied().max().unwrap_or(0);
    let mean = if lengths.is_empty() {
        0.0
    } else {
        lengths.iter().map(|&n| n as f64).sum::<f64>() / lengths.len() as f64
    };
    TokenLengthStats {
        requested,
        min,
        max,
        mean,
    }
}

fn output_token_count_source_from_runs(runs: &[RunRecord]) -> String {
    let mut saw_usage = false;
    let mut saw_stream_chunks = false;
    let mut saw_none = false;
    for record in runs.iter().flat_map(|run| &run.records) {
        match record.output_token_count_source {
            OutputTokenCountSource::Usage => saw_usage = true,
            OutputTokenCountSource::StreamChunks => saw_stream_chunks = true,
            OutputTokenCountSource::None => saw_none = true,
        }
    }
    match (saw_usage, saw_stream_chunks, saw_none) {
        (true, false, false) => "usage".to_string(),
        (false, true, false) => "stream_chunks".to_string(),
        (false, false, true) => "none".to_string(),
        _ => "mixed".to_string(),
    }
}

fn cell_seed(cell: Cell) -> u64 {
    match cell {
        Cell::Closed(c) => 0xC10C_ED00_0000_0000u64 ^ c as u64,
        Cell::Open(r) => 0x0FEE_D000_0000_0000u64 ^ r.to_bits(),
    }
}

fn cell_label(cell: Cell) -> String {
    match cell {
        Cell::Closed(c) => format!("closed_loop c={c}"),
        Cell::Open(r) => format!("open_loop rate={r}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Top-level entry
// ─────────────────────────────────────────────────────────────────────

pub async fn execute(cmd: BenchServeCommand, _cfg: CliConfig) -> Result<()> {
    validate_command(&cmd)?;
    eprintln!(
        "{}",
        format!(
            "ferrum bench-serve — dataset={} num_prompts={} warmup={} n_repeats={}",
            cmd.dataset, cmd.num_prompts, cmd.warmup_requests, cmd.n_repeats
        )
        .dimmed()
    );
    if cmd.n_repeats < 3 {
        eprintln!(
            "{}",
            "  [warn] n_repeats < 3 — emitting mean only, no stddev/CI95 (PLAYBOOK § 0.4)".yellow()
        );
    }

    // Build cells from CLI flags. Order: request_rate > concurrency_sweep > concurrency.
    let cells: Vec<Cell> = if let Some(rate) = cmd.request_rate {
        vec![Cell::Open(rate)]
    } else if !cmd.concurrency_sweep.is_empty() {
        cmd.concurrency_sweep
            .iter()
            .copied()
            .map(Cell::Closed)
            .collect()
    } else {
        vec![Cell::Closed(cmd.concurrency)]
    };

    let client = Arc::new(
        reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .build()
            .map_err(|e| ferrum_types::FerrumError::model(format!("reqwest client: {e}")))?,
    );
    let ctx = RunContext {
        client,
        base_url: Arc::new(cmd.base_url.clone()),
        model: Arc::new(cmd.model.clone()),
        max_out: cmd.random_output_len,
        ignore_eos: cmd.ignore_eos,
        enable_thinking: cmd.enable_thinking,
        timeout_s: cmd.timeout,
    };

    let mut reports: Vec<BenchReport> = Vec::with_capacity(cells.len());
    for cell in cells {
        eprintln!("{}", format!("→ {}", cell_label(cell)).bold());
        let r = execute_cell(&cmd, &ctx, cell).await?;
        emit_summary_line(&r);
        reports.push(r);
    }

    emit_then_enforce_error_policy(&cmd, &reports)
}

fn emit_then_enforce_error_policy(cmd: &BenchServeCommand, reports: &[BenchReport]) -> Result<()> {
    // Emit final report before enforcing error policy so failed release cells
    // still leave quality/error-count evidence in --out artifacts.
    emit_reports(cmd, reports)?;

    // PLAYBOOK § 1.5: static globals don't drop on Rust process exit.
    ferrum_bench_core::trace::flush_global_trace();

    enforce_error_policy(cmd, reports)
}

fn emit_reports(cmd: &BenchServeCommand, reports: &[BenchReport]) -> Result<()> {
    match cmd.output.as_str() {
        "json" => emit_json(cmd, reports)?,
        "jsonl" => emit_jsonl(cmd, reports)?,
        "md" => emit_markdown(cmd, reports)?,
        other => {
            return Err(ferrum_types::FerrumError::model(format!(
                "unknown --output '{other}': allowed values are json, jsonl, md"
            )))
        }
    }
    Ok(())
}

fn validate_command(cmd: &BenchServeCommand) -> Result<()> {
    if let Some(rate) = cmd.request_rate {
        if rate <= 0.0 || !rate.is_finite() {
            return Err(ferrum_types::FerrumError::model(
                "--request-rate must be a positive finite number",
            ));
        }
    }
    if cmd.timeout <= 0.0
        || !cmd.timeout.is_finite()
        || Duration::try_from_secs_f64(cmd.timeout).is_err()
    {
        return Err(ferrum_types::FerrumError::model(
            "--timeout must be a positive finite duration",
        ));
    }
    if cmd.concurrency == 0 {
        return Err(ferrum_types::FerrumError::model(
            "--concurrency must be > 0",
        ));
    }
    if cmd.num_prompts == 0 {
        return Err(ferrum_types::FerrumError::model(
            "--num-prompts must be > 0",
        ));
    }
    if cmd.n_repeats == 0 {
        return Err(ferrum_types::FerrumError::model("--n-repeats must be > 0"));
    }
    if cmd.random_input_len == 0 {
        return Err(ferrum_types::FerrumError::model(
            "--random-input-len must be > 0",
        ));
    }
    if cmd.random_output_len == 0 {
        return Err(ferrum_types::FerrumError::model(
            "--random-output-len must be > 0",
        ));
    }
    u32::try_from(cmd.random_output_len).map_err(|_| {
        ferrum_types::FerrumError::model("--random-output-len exceeds report capacity")
    })?;
    requested_input_len(cmd)?;
    if cmd.goodput.is_some_and(|slo| !slo.is_valid()) {
        return Err(ferrum_types::FerrumError::model(
            "--goodput values must be positive finite numbers",
        ));
    }
    if cmd.require_ci && cmd.n_repeats < 3 {
        return Err(ferrum_types::FerrumError::model(
            "--require-ci requires --n-repeats >= 3",
        ));
    }
    if let Some(max_error_rate) = cmd.max_error_rate {
        if !(0.0..=1.0).contains(&max_error_rate) || !max_error_rate.is_finite() {
            return Err(ferrum_types::FerrumError::model(
                "--max-error-rate must be in [0.0, 1.0]",
            ));
        }
    }
    for cell in &cmd.concurrency_sweep {
        if *cell == 0 {
            return Err(ferrum_types::FerrumError::model(
                "--concurrency-sweep values must be > 0",
            ));
        }
    }
    let _total_prompts = cmd
        .num_prompts
        .checked_add(cmd.warmup_requests)
        .ok_or_else(|| {
            ferrum_types::FerrumError::model("num_prompts + warmup_requests overflow")
        })?;
    Ok(())
}

fn enforce_error_policy(cmd: &BenchServeCommand, reports: &[BenchReport]) -> Result<()> {
    let max_error_rate = cmd.max_error_rate.unwrap_or(0.0);
    for report in reports {
        let warmup_errored: u64 = report
            .repeat_metrics
            .iter()
            .map(|repeat| repeat.warmup_errored as u64)
            .sum();
        let warmup_has_quality_issue = report
            .repeat_metrics
            .iter()
            .any(|repeat| repeat.warmup_quality_issues != QualityIssueCounts::default());
        if warmup_errored > 0 || warmup_has_quality_issue {
            return Err(ferrum_types::FerrumError::model(format!(
                "bench-serve warmup failed for {}: {} errored request(s)",
                report.model, warmup_errored
            )));
        }
        if !cmd.fail_on_error && cmd.max_error_rate.is_none() {
            continue;
        }

        let completed: u64 = report
            .completed_per_run
            .iter()
            .map(|value| *value as u64)
            .sum();
        let errored: u64 = report
            .errored_per_run
            .iter()
            .map(|value| *value as u64)
            .sum();
        let total = completed
            .checked_add(errored)
            .ok_or_else(|| ferrum_types::FerrumError::model("measured request count overflow"))?;
        let error_rate = if total == 0 {
            1.0
        } else {
            errored as f64 / total as f64
        };
        if error_rate > max_error_rate {
            return Err(ferrum_types::FerrumError::model(format!(
                "bench-serve error rate {:.4} exceeds max {:.4} for {}",
                error_rate, max_error_rate, report.model
            )));
        }
    }
    Ok(())
}

fn emit_jsonl(cmd: &BenchServeCommand, reports: &[BenchReport]) -> Result<()> {
    use std::io::Write as _;
    let out_path = cmd.out.as_ref().ok_or_else(|| {
        ferrum_types::FerrumError::model("--output jsonl requires --out PATH (append-mode log)")
    })?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .map_err(|e| {
            ferrum_types::FerrumError::model(format!("open {}: {e}", out_path.display()))
        })?;
    for r in reports {
        let line = serde_json::to_string(r).expect("serialize");
        writeln!(f, "{}", line)
            .map_err(|e| ferrum_types::FerrumError::model(format!("write jsonl: {e}")))?;
    }
    eprintln!(
        "\n→ appended {} record(s) to {}",
        reports.len(),
        out_path.display()
    );
    Ok(())
}

fn emit_markdown(cmd: &BenchServeCommand, reports: &[BenchReport]) -> Result<()> {
    use ferrum_bench_core::report::{render_single, render_sweep};
    let md = if reports.len() == 1 {
        render_single(&reports[0])
    } else {
        render_sweep(reports)
    };
    if let Some(out) = cmd.out.as_ref() {
        std::fs::write(out, &md).map_err(|e| {
            ferrum_types::FerrumError::model(format!("write {}: {e}", out.display()))
        })?;
        eprintln!("\n→ wrote {}", out.display());
    } else {
        println!("{}", md);
    }
    Ok(())
}

fn emit_summary_line(r: &BenchReport) {
    let scenario_str = match r.scenario {
        Scenario::ClosedLoop => format!("c={}", r.concurrency.unwrap_or(0)),
        Scenario::OpenLoop => format!("rate={}", r.request_rate.unwrap_or(0.0)),
        Scenario::SharedPrefix => "shared_prefix".to_string(),
        Scenario::Cli => "cli".to_string(),
    };
    let ci = if r.n_repeats >= 3 {
        format!(" (n_repeats={}, ± = ci95_hw)", r.n_repeats)
    } else {
        format!(" (n_repeats={}, no CI)", r.n_repeats)
    };
    eprintln!("    {} {}{}", "summary".bold(), scenario_str, ci);
    fmt_metric("TTFT_ms ", &r.ttft_ms, r.n_repeats);
    fmt_metric("TPOT_ms ", &r.tpot_ms, r.n_repeats);
    if r.has_complete_itl_evidence() {
        fmt_metric("ITL_ms  ", &r.itl_ms, r.n_repeats);
    } else {
        eprintln!("      ITL_ms   unavailable");
    }
    let thr = &r.output_throughput_tps;
    let good = &r.goodput_rps;
    if r.n_repeats >= 3 {
        eprintln!(
            "      throughput      {:.1} ± {:.1} tok/s",
            thr.mean, thr.ci95_hw
        );
        eprintln!(
            "      goodput         {:.2} ± {:.2} req/s",
            good.mean, good.ci95_hw
        );
    } else {
        eprintln!("      throughput      {:.1} tok/s", thr.mean);
        eprintln!("      goodput         {:.2} req/s", good.mean);
    }
}

fn fmt_metric(name: &str, m: &ferrum_bench_core::MetricSet, n_repeats: u32) {
    if n_repeats >= 3 {
        eprintln!(
            "      {} p50={:.1}±{:.1}  p95={:.1}±{:.1}  p99={:.1}±{:.1}",
            name, m.p50.mean, m.p50.ci95_hw, m.p95.mean, m.p95.ci95_hw, m.p99.mean, m.p99.ci95_hw
        );
    } else {
        eprintln!(
            "      {} p50={:.1}  p95={:.1}  p99={:.1}",
            name, m.p50.mean, m.p95.mean, m.p99.mean
        );
    }
}

fn emit_json(cmd: &BenchServeCommand, reports: &[BenchReport]) -> Result<()> {
    let value = if reports.len() == 1 {
        serde_json::to_value(&reports[0]).expect("serialize")
    } else {
        serde_json::to_value(reports).expect("serialize")
    };
    let pretty = serde_json::to_string_pretty(&value).expect("pretty");
    if let Some(out) = cmd.out.as_ref() {
        std::fs::write(out, &pretty).map_err(|e| {
            ferrum_types::FerrumError::model(format!("write {}: {e}", out.display()))
        })?;
        eprintln!("\n→ wrote {}", out.display());
    } else {
        println!("{}", pretty);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_bench_core::ItlEligibility;

    fn parse_sse_chunks<'a>(chunks: impl IntoIterator<Item = &'a [u8]>) -> RequestRecord {
        let mut state = StreamState::new(Instant::now(), 7);
        let mut parser = SseLineBuffer::default();
        for chunk in chunks {
            let before_output_events = state.output_delta_events;
            parser.push(chunk, &mut state);
            let output_events = state
                .output_delta_events
                .saturating_sub(before_output_events);
            state.note_transport_chunk(output_events);
        }
        parser.finish(&mut state);
        state.finish()
    }

    fn stream_semantics(
        record: &RequestRecord,
    ) -> (
        bool,
        u32,
        OutputTokenCountSource,
        QualityIssueCounts,
        RequestItlEvidence,
    ) {
        (
            record.success,
            record.output_tokens,
            record.output_token_count_source,
            record.quality_issues.clone(),
            record.itl_evidence.clone(),
        )
    }

    #[test]
    fn slo_parses_space_separated() {
        let s = parse_slo("ttft:500 tpot:50 e2el:30000").unwrap();
        assert_eq!(s.ttft_p99_ms, 500.0);
        assert_eq!(s.tpot_p99_ms, 50.0);
        assert_eq!(s.e2e_p99_ms, 30000.0);
    }

    #[test]
    fn slo_parses_comma_separated() {
        let s = parse_slo("ttft:500,tpot:50,e2el:30000").unwrap();
        assert_eq!(s.ttft_p99_ms, 500.0);
    }

    #[test]
    fn slo_accepts_e2e_alias() {
        let s = parse_slo("ttft:1 tpot:2 e2e:3").unwrap();
        assert_eq!(s.e2e_p99_ms, 3.0);
    }

    #[test]
    fn slo_rejects_missing_key() {
        assert!(parse_slo("ttft:500 tpot:50").is_err());
    }

    #[test]
    fn slo_rejects_unknown_key() {
        assert!(parse_slo("ttft:500 tpot:50 e2el:30000 bogus:1").is_err());
    }

    #[test]
    fn slo_rejects_nonfinite_and_nonpositive_values() {
        for value in ["NaN", "inf", "0", "-1"] {
            assert!(
                parse_slo(&format!("ttft:{value} tpot:50 e2el:30000")).is_err(),
                "SLO value {value} should be rejected"
            );
        }
    }

    #[test]
    fn chat_completion_body_omits_ignore_eos_by_default() {
        let body = chat_completion_body("model", "prompt", 128, false, None);
        assert_eq!(body["model"], serde_json::json!("model"));
        assert_eq!(body["max_tokens"], serde_json::json!(128));
        assert_eq!(body["stream"], serde_json::json!(true));
        assert!(body.get("chat_template_kwargs").is_none());
        assert_eq!(
            body["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
        assert!(body.get("ignore_eos").is_none());
    }

    #[test]
    fn chat_completion_body_sends_ignore_eos_when_requested() {
        let body = chat_completion_body("model", "prompt", 128, true, None);
        assert_eq!(body["ignore_eos"], serde_json::json!(true));
    }

    #[test]
    fn chat_completion_body_sends_typed_thinking_values() {
        let disabled = chat_completion_body("model", "prompt", 128, false, Some(false));
        assert_eq!(
            disabled["chat_template_kwargs"]["enable_thinking"],
            serde_json::json!(false)
        );
        let enabled = chat_completion_body("model", "prompt", 128, false, Some(true));
        assert_eq!(
            enabled["chat_template_kwargs"]["enable_thinking"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn enable_thinking_cli_is_tri_state() {
        use clap::Parser as _;

        #[derive(clap::Parser)]
        struct TestCli {
            #[command(flatten)]
            command: BenchServeCommand,
        }

        let base = [
            "test",
            "--base-url",
            "http://127.0.0.1:8000",
            "--model",
            "model",
            "--tokenizer",
            ".",
        ];
        let absent = TestCli::parse_from(base).command;
        assert_eq!(absent.enable_thinking, None);

        let disabled =
            TestCli::parse_from(base.into_iter().chain(["--enable-thinking", "false"])).command;
        assert_eq!(disabled.enable_thinking, Some(false));

        let enabled =
            TestCli::parse_from(base.into_iter().chain(["--enable-thinking", "true"])).command;
        assert_eq!(enabled.enable_thinking, Some(true));
    }

    #[test]
    fn incremental_sse_utf8_semantics_are_invariant_to_every_byte_cut() {
        let stream = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好🙂\"}}],\"usage\":null}\n\n",
            "data: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n",
        )
        .as_bytes();
        let baseline = parse_sse_chunks([stream]);
        assert!(baseline.success);
        assert_eq!(baseline.quality_issues, QualityIssueCounts::default());

        for cut in 0..=stream.len() {
            let split = parse_sse_chunks([&stream[..cut], &stream[cut..]]);
            assert_eq!(
                stream_semantics(&split),
                stream_semantics(&baseline),
                "SSE semantics changed at byte cut {cut}"
            );
        }

        let bytewise = parse_sse_chunks(stream.chunks(1));
        assert_eq!(stream_semantics(&bytewise), stream_semantics(&baseline));
    }

    #[test]
    fn transport_coalescing_is_diagnostic_not_stream_bulk_failure() {
        const FIRST: &[u8] =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}],\"usage\":null}\n\n";
        const SECOND: &[u8] =
            b"data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}],\"usage\":null}\n\n";
        const TAIL: &[u8] =
            b"data: {\"choices\":[],\"usage\":{\"completion_tokens\":2}}\n\ndata: [DONE]\n\n";
        let together_bytes = [FIRST, SECOND, TAIL].concat();
        let together = parse_sse_chunks([together_bytes.as_slice()]);
        assert!(together.success);
        assert_eq!(together.quality_issues.stream_bulk_flush, 0);
        assert_eq!(together.itl_evidence.transport_coalesced_output_chunks, 1);
        assert_eq!(
            together.itl_evidence.eligibility,
            ItlEligibility::TransportCoalesced
        );

        let separated = parse_sse_chunks([FIRST, SECOND, TAIL]);
        assert!(separated.success);
        assert_eq!(separated.quality_issues, QualityIssueCounts::default());
        assert_eq!(separated.itl_evidence.transport_coalesced_output_chunks, 0);
        assert_eq!(separated.itl_evidence.eligibility, ItlEligibility::Eligible);
    }

    #[test]
    fn stream_done_with_usage_succeeds() {
        let mut state = StreamState::new(Instant::now(), 7);
        state
            .handle_payload(r#"{"choices":[{"delta":{"content":"hello"}}],"usage":null}"#)
            .unwrap();
        state
            .handle_payload(r#"{"choices":[],"usage":{"completion_tokens":3}}"#)
            .unwrap();
        state.done_count = 1;
        let record = state.finish();
        assert!(record.success);
        assert_eq!(record.output_tokens, 3);
        assert_eq!(
            record.output_token_count_source,
            OutputTokenCountSource::Usage
        );
        assert_eq!(
            record.itl_evidence.eligibility,
            ItlEligibility::EventUsageMismatch
        );
        assert_eq!(record.itl_evidence.output_events, 1);
        assert_eq!(record.itl_evidence.usage_output_tokens, Some(3));
    }

    #[test]
    fn matching_sse_events_usage_and_intervals_are_itl_eligible() {
        let mut state = StreamState::new(Instant::now(), 7);
        for content in ["a", "b", "c"] {
            state
                .handle_payload(&format!(
                    r#"{{"choices":[{{"delta":{{"content":"{content}"}}}}]}}"#
                ))
                .unwrap();
        }
        state
            .handle_payload(r#"{"choices":[],"usage":{"completion_tokens":3}}"#)
            .unwrap();
        state.done_count = 1;
        let record = state.finish();
        assert!(record.success);
        assert_eq!(record.itl_ms.len(), 2);
        assert_eq!(record.itl_evidence.eligibility, ItlEligibility::Eligible);
    }

    #[tokio::test]
    async fn measured_join_error_becomes_failed_evidence() {
        let good = tokio::spawn(async {
            RequestRecord {
                success: true,
                ttft_ms: 1.0,
                e2e_ms: 2.0,
                input_tokens: 7,
                output_tokens: 1,
                output_token_count_source: OutputTokenCountSource::Usage,
                itl_evidence: RequestItlEvidence::sse(true, 1, Some(1), 0, 0),
                quality_issues: QualityIssueCounts::default(),
                itl_ms: vec![],
            }
        });
        let panicked = tokio::spawn(async {
            if true {
                panic!("measured task panic");
            }
            join_failed_record(0)
        });
        let records = collect_measured_handles(vec![(7, good), (11, panicked)]).await;
        assert_eq!(records.len(), 2);
        assert!(records[0].success);
        assert!(!records[1].success);
        assert_eq!(records[1].input_tokens, 11);
        assert_eq!(records[1].quality_issues.panic, 1);
        assert_eq!(
            records[1].output_token_count_source,
            OutputTokenCountSource::None
        );
    }

    #[test]
    fn stream_reasoning_chunk_counts_as_token_event() {
        let mut state = StreamState::new(Instant::now(), 7);
        state
            .handle_payload(r#"{"choices":[{"delta":{"reasoning":"thinking"}}]}"#)
            .unwrap();
        state.done_count = 1;
        let record = state.finish();
        assert!(record.success);
        assert_eq!(record.output_tokens, 1);
        assert_eq!(
            record.output_token_count_source,
            OutputTokenCountSource::StreamChunks
        );
    }

    #[test]
    fn stream_error_after_chunk_fails() {
        let mut state = StreamState::new(Instant::now(), 7);
        state
            .handle_payload(r#"{"choices":[{"delta":{"content":"hello"}}]}"#)
            .unwrap();
        state.stream_error = Some("broken stream".into());
        state.done_count = 1;
        let record = state.finish();
        assert!(!record.success);
        assert_eq!(
            record.output_token_count_source,
            OutputTokenCountSource::StreamChunks
        );
    }

    #[test]
    fn eof_before_done_after_chunk_fails() {
        let mut state = StreamState::new(Instant::now(), 7);
        state
            .handle_payload(r#"{"choices":[{"delta":{"content":"hello"}}]}"#)
            .unwrap();
        let record = state.finish();
        assert!(!record.success);
        assert_eq!(record.output_tokens, 1);
        assert_eq!(record.quality_issues.missing_done, 1);
    }

    #[test]
    fn malformed_sse_json_fails() {
        let mut state = StreamState::new(Instant::now(), 7);
        assert!(state.handle_payload("{bad json}").is_err());
        state.stream_error = Some("malformed json".into());
        state.quality_issues.malformed_stream = 1;
        state.done_count = 1;
        let record = state.finish();
        assert!(!record.success);
        assert_eq!(record.output_tokens, 0);
        assert_eq!(record.quality_issues.malformed_stream, 1);
    }

    #[test]
    fn done_with_zero_content_tokens_fails() {
        let mut state = StreamState::new(Instant::now(), 7);
        state.done_count = 1;
        let record = state.finish();
        assert!(!record.success);
        assert_eq!(record.output_tokens, 0);
        assert_eq!(record.quality_issues.zero_output_tokens, 1);
        assert_eq!(
            record.output_token_count_source,
            OutputTokenCountSource::None
        );
    }

    #[test]
    fn duplicate_done_fails() {
        let mut state = StreamState::new(Instant::now(), 7);
        state
            .handle_payload(r#"{"choices":[{"delta":{"content":"hello"}}]}"#)
            .unwrap();
        state.done_count = 2;
        let record = state.finish();
        assert!(!record.success);
        assert_eq!(record.quality_issues.duplicate_done, 1);
    }

    #[test]
    fn bad_output_text_fails() {
        let mut state = StreamState::new(Instant::now(), 7);
        state
            .handle_payload(r#"{"choices":[{"delta":{"content":"<unk>"}}]}"#)
            .unwrap();
        state.done_count = 1;
        let record = state.finish();
        assert!(!record.success);
        assert_eq!(record.quality_issues.bad_output, 1);
    }

    #[test]
    fn mojibake_sequences_fail_but_standalone_leads_do_not() {
        assert!(has_bad_output_text("caf\u{00c3}\u{00a9}"));
        assert!(has_bad_output_text("copyright \u{00c2}\u{00a9}"));
        assert!(has_bad_output_text("quote\u{00e2}\u{20ac}\u{2122}"));
        assert!(!has_bad_output_text("\u{00c2}"));
        assert!(!has_bad_output_text("\u{00c3}"));
        assert!(!has_bad_output_text("Grade \u{00c2} report"));
    }

    #[test]
    fn random_prompt_generation_targets_reencoded_length_when_fixture_is_set() {
        let Some(path) = ferrum_env_value("FERRUM_BENCH_TOKENIZER_FIXTURE") else {
            return;
        };
        let tok = tokenizers::Tokenizer::from_file(path).expect("load tokenizer fixture");
        let mut rng = StdRng::seed_from_u64(9271);
        for _ in 0..16 {
            let text = gen_random_prompt(&tok, 256, &mut rng);
            assert_eq!(token_count(&tok, &text), Some(256));
        }
    }

    fn ferrum_env_value(key: &str) -> Option<String> {
        ferrum_types::RuntimeConfigSnapshot::capture_current()
            .entries
            .into_iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.effective_value)
    }

    fn test_command() -> BenchServeCommand {
        BenchServeCommand {
            base_url: "http://127.0.0.1:9".to_string(),
            model: "test-model".to_string(),
            tokenizer: std::path::PathBuf::from("."),
            concurrency: 1,
            concurrency_sweep: vec![],
            request_rate: None,
            dataset: "random".to_string(),
            random_input_len: 2,
            random_output_len: 3,
            ignore_eos: false,
            enable_thinking: None,
            sharegpt_path: None,
            shared_prefix_len: 1024,
            shared_suffix_len: 64,
            num_prompts: 1,
            warmup_requests: 0,
            n_repeats: 1,
            goodput: None,
            timeout: 1.0,
            fail_on_error: false,
            max_error_rate: None,
            require_ci: false,
            seed: Some(9271),
            output: "json".to_string(),
            out: None,
            hw_id: None,
            commit_sha: None,
            tag: None,
        }
    }

    fn policy_report(
        completed: u32,
        errored: u32,
        warmup_completed: u32,
        warmup_errored: u32,
    ) -> BenchReport {
        let mut records = Vec::with_capacity((completed + errored) as usize);
        for _ in 0..completed {
            records.push(RequestRecord {
                success: true,
                ttft_ms: 10.0,
                e2e_ms: 30.0,
                input_tokens: 2,
                output_tokens: 3,
                output_token_count_source: OutputTokenCountSource::Usage,
                itl_evidence: RequestItlEvidence::sse(true, 3, Some(3), 2, 0),
                quality_issues: QualityIssueCounts::default(),
                itl_ms: vec![10.0, 10.0],
            });
        }
        for _ in 0..errored {
            let mut quality = QualityIssueCounts::default();
            quality.missing_done = 1;
            records.push(RequestRecord {
                success: false,
                ttft_ms: 0.0,
                e2e_ms: 30.0,
                input_tokens: 2,
                output_tokens: 0,
                output_token_count_source: OutputTokenCountSource::None,
                itl_evidence: RequestItlEvidence::failed(ItlEvidenceSource::SseDeltaEvents),
                quality_issues: quality,
                itl_ms: vec![],
            });
        }
        let warmup_expected = warmup_completed.checked_add(warmup_errored).unwrap();
        let mut warmup_quality = QualityIssueCounts::default();
        warmup_quality.missing_done = warmup_errored;
        compute_metrics(
            "test-model".to_string(),
            "test-backend".to_string(),
            Scenario::ClosedLoop,
            Some(1),
            None,
            2,
            3,
            warmup_expected,
            Slo::default(),
            vec![RunRecord {
                expected_requests: completed.checked_add(errored).unwrap(),
                records,
                duration_s: 1.0,
                warmup: WarmupSummary {
                    expected: warmup_expected,
                    completed: warmup_completed,
                    errored: warmup_errored,
                    quality_issues: warmup_quality,
                },
            }],
            Env::default(),
        )
    }

    #[test]
    fn validate_command_rejects_zero_repeats_and_invalid_programmatic_slo() {
        let mut cmd = test_command();
        cmd.n_repeats = 0;
        assert!(validate_command(&cmd).is_err());
        cmd.n_repeats = 1;
        cmd.goodput = Some(Slo {
            ttft_p99_ms: f64::NAN,
            ..Slo::default()
        });
        assert!(validate_command(&cmd).is_err());
        cmd.goodput = None;
        cmd.timeout = f64::MAX;
        assert!(validate_command(&cmd).is_err());
    }

    #[test]
    fn measured_error_rate_excludes_successful_warmups() {
        let report = policy_report(99, 1, 10, 0);
        let mut cmd = test_command();
        cmd.max_error_rate = Some(0.0095);
        let err = enforce_error_policy(&cmd, &[report]).expect_err("measured rate is one percent");
        assert!(err.to_string().contains("bench-serve error rate"));
    }

    #[test]
    fn warmup_failure_is_independent_of_measured_error_allowance() {
        let report = policy_report(100, 0, 9, 1);
        let mut cmd = test_command();
        cmd.max_error_rate = Some(1.0);
        let err = enforce_error_policy(&cmd, &[report]).expect_err("warmup must be perfect");
        assert!(err.to_string().contains("bench-serve warmup failed"));
    }

    #[test]
    fn fail_on_error_still_writes_json_report() {
        let out = std::env::temp_dir().join(format!(
            "ferrum-bench-serve-failed-report-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&out);

        let mut failed_quality = QualityIssueCounts::default();
        failed_quality.http_500 = 1;
        let run = RunRecord {
            records: vec![
                RequestRecord {
                    success: true,
                    ttft_ms: 10.0,
                    e2e_ms: 30.0,
                    input_tokens: 4,
                    output_tokens: 3,
                    output_token_count_source: OutputTokenCountSource::Usage,
                    itl_evidence: RequestItlEvidence::sse(true, 3, Some(3), 2, 0),
                    quality_issues: QualityIssueCounts::default(),
                    itl_ms: vec![10.0, 10.0],
                },
                RequestRecord {
                    success: false,
                    ttft_ms: 0.0,
                    e2e_ms: 50.0,
                    input_tokens: 4,
                    output_tokens: 0,
                    output_token_count_source: OutputTokenCountSource::None,
                    itl_evidence: RequestItlEvidence::failed(ItlEvidenceSource::SseDeltaEvents),
                    quality_issues: failed_quality,
                    itl_ms: vec![],
                },
            ],
            expected_requests: 2,
            duration_s: 1.0,
            warmup: Default::default(),
        };
        let report = compute_metrics(
            "test-model".to_string(),
            "test-backend".to_string(),
            Scenario::ClosedLoop,
            Some(2),
            None,
            2,
            3,
            0,
            Slo::default(),
            vec![run],
            Env::default(),
        );
        let cmd = BenchServeCommand {
            base_url: "http://127.0.0.1:9".to_string(),
            model: "test-model".to_string(),
            tokenizer: std::path::PathBuf::from("."),
            concurrency: 2,
            concurrency_sweep: vec![],
            request_rate: None,
            dataset: "random".to_string(),
            random_input_len: 4,
            random_output_len: 3,
            ignore_eos: false,
            enable_thinking: None,
            sharegpt_path: None,
            shared_prefix_len: 1024,
            shared_suffix_len: 64,
            num_prompts: 2,
            warmup_requests: 0,
            n_repeats: 1,
            goodput: None,
            timeout: 1.0,
            fail_on_error: true,
            max_error_rate: None,
            require_ci: false,
            seed: Some(9271),
            output: "json".to_string(),
            out: Some(out.clone()),
            hw_id: None,
            commit_sha: None,
            tag: None,
        };

        let err = emit_then_enforce_error_policy(&cmd, &[report]).expect_err("error policy");
        assert!(
            err.to_string().contains("bench-serve error rate"),
            "unexpected error: {err}"
        );
        let raw = std::fs::read_to_string(&out).expect("report written before error");
        let json: serde_json::Value = serde_json::from_str(&raw).expect("json report");
        assert_eq!(json["completed_per_run"], serde_json::json!([1]));
        assert_eq!(json["errored_per_run"], serde_json::json!([1]));
        assert_eq!(json["http_500_per_run"], serde_json::json!([1]));
        assert_eq!(
            json["output_tokens_per_request"],
            serde_json::json!([[3, 0]])
        );
        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn fail_on_error_rejects_warmup_only_failure() {
        let mut warmup_quality = QualityIssueCounts::default();
        warmup_quality.missing_done = 1;
        let report = compute_metrics(
            "test-model".to_string(),
            "test-backend".to_string(),
            Scenario::ClosedLoop,
            Some(1),
            None,
            2,
            3,
            1,
            Slo::default(),
            vec![RunRecord {
                records: vec![RequestRecord {
                    success: true,
                    ttft_ms: 10.0,
                    e2e_ms: 30.0,
                    input_tokens: 2,
                    output_tokens: 3,
                    output_token_count_source: OutputTokenCountSource::Usage,
                    itl_evidence: RequestItlEvidence::sse(true, 3, Some(3), 2, 0),
                    quality_issues: QualityIssueCounts::default(),
                    itl_ms: vec![10.0, 10.0],
                }],
                expected_requests: 1,
                duration_s: 1.0,
                warmup: WarmupSummary {
                    expected: 1,
                    completed: 0,
                    errored: 1,
                    quality_issues: warmup_quality,
                },
            }],
            Env::default(),
        );
        let cmd = BenchServeCommand {
            base_url: "http://127.0.0.1:9".to_string(),
            model: "test-model".to_string(),
            tokenizer: std::path::PathBuf::from("."),
            concurrency: 1,
            concurrency_sweep: vec![],
            request_rate: None,
            dataset: "random".to_string(),
            random_input_len: 2,
            random_output_len: 3,
            ignore_eos: false,
            enable_thinking: None,
            sharegpt_path: None,
            shared_prefix_len: 1024,
            shared_suffix_len: 64,
            num_prompts: 1,
            warmup_requests: 1,
            n_repeats: 1,
            goodput: None,
            timeout: 1.0,
            fail_on_error: true,
            max_error_rate: None,
            require_ci: false,
            seed: Some(9271),
            output: "json".to_string(),
            out: None,
            hw_id: None,
            commit_sha: None,
            tag: None,
        };

        let err = enforce_error_policy(&cmd, &[report]).expect_err("warmup failure");
        assert!(err.to_string().contains("bench-serve warmup failed"));
    }
}
