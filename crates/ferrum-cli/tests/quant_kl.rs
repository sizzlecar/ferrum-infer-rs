//! Quantization-drift / determinism gate (PLAYBOOK § 3 L2).
//!
//! Two sub-tests, both built on the same divergence-rate primitive:
//!
//! 1. **Self-determinism** — run the SAME model twice with greedy
//!    `temperature=0.0`; expect 100% token-level agreement. Failure
//!    indicates a non-determinism bug like the prefix-cache CoW
//!    contamination class fixed in PR #204. Always available — only
//!    needs one model. This is the test that runs by default.
//!
//! 2. **Paired-quant drift** — run FP16 ferrum and INT4 ferrum on the
//!    same prompts; expect divergence_rate < 0.10. Auto-skipped when
//!    the paired INT4 variant isn't in the HF cache (most setups).
//!
//! Both tests use the same divergence_rate metric: across N prompts × M
//! tokens, what fraction of (prompt, token-position) cells produced
//! different tokens between the two runs.
//!
//! llama.cpp's `tools/perplexity/README.md` explicitly recommends KL on
//! raw logits over PPL-vs-FP16; this test approximates KL via sampled
//! tokens (a coarser but always-available signal — see § "Why sampled
//! tokens" below).
//!
//! Run:
//!     cargo test --release -p ferrum-cli --features metal \
//!         --test quant_kl -- --ignored --test-threads=1
//!
//! ### Why sampled tokens vs raw logits
//!
//! Real KL needs the full per-step logit vector. ferrum's
//! `InferenceResponse.tokens` is the sampled `TokenId` sequence —
//! logits aren't exposed. Adding a logits-capture path is a separate,
//! engine-surface change. Until that lands, this test uses
//! **token-divergence rate** as a proxy: at temp=0 greedy, divergent
//! tokens at position k means "the argmax differs at position k",
//! which implies the logits differ enough to flip the top-1. It's a
//! coarser signal than KL (a 1-bit drift can be invisible if the top-1
//! is robust), but it catches every regression that matters in practice
//! (a kernel that flips even 1% of greedy tokens is broken).

use reqwest::Client;
use serde_json::json;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

const SELF_PROMPTS: &[&str] = &[
    "Explain entropy in one paragraph.",
    "List the planets in our solar system.",
    "Translate 'good morning' to French and Japanese.",
    "What is 17 times 23?",
    "Write a haiku about autumn leaves.",
    "Describe what gravity is.",
    "Name three classical composers.",
    "What does TCP stand for?",
    "Define the word 'serendipity'.",
    "What is the capital of Brazil?",
];

const MAX_TOKENS: u32 = 50;

// ─────────────────────────────────────────────────────────────────────
// Server fixture (mirrors reference_match.rs)
// ─────────────────────────────────────────────────────────────────────

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
            .env("FERRUM_PREFIX_CACHE", "0") // PLAYBOOK § 0.5 — determinism contract
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

// ─────────────────────────────────────────────────────────────────────
// Per-prompt greedy completion
// ─────────────────────────────────────────────────────────────────────

async fn greedy_complete(client: &Client, fx: &ServerFixture, model: &str, prompt: &str) -> String {
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": false,
    });
    let resp = client
        .post(fx.chat_url())
        .timeout(REQUEST_TIMEOUT)
        .json(&body)
        .send()
        .await
        .expect("HTTP send");
    assert!(resp.status().is_success(), "HTTP {}", resp.status());
    let v: serde_json::Value = resp.json().await.expect("JSON parse");
    v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

// ─────────────────────────────────────────────────────────────────────
// Divergence-rate primitive
// ─────────────────────────────────────────────────────────────────────

/// Word-level divergence rate between two corpora. Returns the fraction
/// of `(prompt, word-position)` cells where the two corpora differ.
///
/// We use word-level rather than tokenizer-level because:
///   - we don't have the tokenizer loaded in this test
///   - whitespace-split words ≈ tokens for English greedy decode
///   - the proxy nature (sampled-token argmax) doesn't need exact alignment
fn divergence_rate(a: &[String], b: &[String]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut total = 0usize;
    let mut diff = 0usize;
    for (sa, sb) in a.iter().zip(b.iter()) {
        let wa: Vec<&str> = sa.split_whitespace().collect();
        let wb: Vec<&str> = sb.split_whitespace().collect();
        let n = wa.len().min(wb.len());
        let extra = (wa.len() as i64 - wb.len() as i64).unsigned_abs() as usize;
        total += n + extra;
        for i in 0..n {
            if wa[i] != wb[i] {
                diff += 1;
            }
        }
        diff += extra; // length mismatch counts as divergence in the tail
    }
    if total == 0 {
        0.0
    } else {
        diff as f64 / total as f64
    }
}

async fn run_corpus(model: &str) -> Vec<String> {
    let fx = ServerFixture::spawn(model).await;
    let client = Client::new();
    let mut out = Vec::with_capacity(SELF_PROMPTS.len());
    for prompt in SELF_PROMPTS {
        out.push(greedy_complete(&client, &fx, model, prompt).await);
    }
    drop(fx);
    out
}

fn hf_cached(repo_dir_substring: &str) -> bool {
    let hf = std::env::var("HF_HOME").unwrap_or_else(|_| {
        format!("{}/.cache/huggingface", std::env::var("HOME").unwrap_or_default())
    });
    let hub = Path::new(&hf).join("hub");
    if !hub.exists() {
        return false;
    }
    std::fs::read_dir(&hub)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .any(|e| e.file_name().to_string_lossy().contains(repo_dir_substring))
        })
        .unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────────
// Sub-test 1 — self-determinism on Qwen3-0.6B
// ─────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "current_thread")]
#[ignore = "loads real model — run with `cargo test -- --ignored`"]
async fn self_determinism_qwen3_0p6b() {
    eprintln!("\n── L2 sub-test 1: self-determinism (Qwen3-0.6B greedy temp=0) ──");
    let model = "qwen3:0.6b";
    let a = run_corpus(model).await;
    let b = run_corpus(model).await;
    let rate = divergence_rate(&a, &b);
    eprintln!(
        "  {} prompts × ~{} tokens → divergence_rate = {:.4}",
        a.len(),
        MAX_TOKENS,
        rate
    );
    for (i, (sa, sb)) in a.iter().zip(b.iter()).enumerate() {
        if sa != sb {
            eprintln!("  prompt {} diverged:", i);
            eprintln!("    run A: {sa:?}");
            eprintln!("    run B: {sb:?}");
        }
    }
    // PR #204-style prefix-cache contamination would push this above 0.
    // Anything > 0.0 is a regression — greedy with prefix cache OFF is
    // expected to be bit-identical across runs (PLAYBOOK § 0.5).
    assert!(
        rate < 0.001,
        "greedy non-determinism detected: divergence_rate={rate:.4} (should be 0)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Sub-test 2 — FP16 vs INT4 drift (paired-quant, auto-skip if unavailable)
// ─────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "current_thread")]
#[ignore = "needs paired FP16/INT4 model variants in HF cache"]
async fn paired_quant_drift_qwen2p5_3b() {
    let fp16 = "Qwen/Qwen2.5-3B-Instruct";
    let int4 = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4";

    if !hf_cached("Qwen2.5-3B-Instruct-GPTQ-Int4") {
        eprintln!(
            "\n⚠ skip: {int4} not in HF cache. To enable, download both variants:"
        );
        eprintln!("    huggingface-cli download {fp16}");
        eprintln!("    huggingface-cli download {int4}");
        return;
    }
    if !hf_cached("Qwen2.5-3B-Instruct") {
        eprintln!("⚠ skip: paired FP16 variant {fp16} not in HF cache.");
        return;
    }

    eprintln!("\n── L2 sub-test 2: paired-quant drift ({fp16} ↔ {int4}) ──");
    let a = run_corpus(fp16).await;
    let b = run_corpus(int4).await;
    let rate = divergence_rate(&a, &b);
    eprintln!(
        "  {} prompts × ~{} tokens → divergence_rate = {:.4}",
        a.len(),
        MAX_TOKENS,
        rate
    );
    // Empirical: well-quantized INT4 should agree with FP16 on > 90% of
    // greedy tokens for short completions. 10% threshold = "the quant
    // is in the right ballpark; if it drops below, the quant kernel
    // (Marlin tile, GPTQ packer) has a real bug."
    assert!(
        rate < 0.10,
        "INT4 quant drift exceeds 10%: divergence_rate={rate:.4}"
    );
}
