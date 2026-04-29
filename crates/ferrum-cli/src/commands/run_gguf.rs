//! GGUF inference path — invoked by `ferrum run <path.gguf>` when the
//! model argument is a `.gguf` file. Bypasses ferrum's `Backend<B>`
//! abstraction in favour of candle-transformers' quantized model loaders
//! so we get Metal Q4_K_M kernels for free (same kernels mistral.rs and
//! llama.cpp use).
//!
//! Two modes share most of the implementation:
//!   - **one-shot** (`--prompt` set): prefill + decode + exit; emits
//!     timing for benchmarks (see `--bench-mode`).
//!   - **interactive REPL** (no `--prompt`): multi-turn chat with proper
//!     per-arch chat template, repeat penalty, top-K + top-P sampling,
//!     and a `/clear` command that re-opens the model for a fresh KV
//!     cache (since candle's quantized_llama / quantized_qwen3_moe don't
//!     yet expose `clear_kv_cache` themselves).
//!
//! Tokenizer auto-discovery rules (in order):
//!   1. Explicit `--tokenizer <path>` if given
//!   2. `<gguf-stem>.tokenizer.json` next to the `.gguf` file
//!   3. `tokenizer.json` in the same directory
//!   4. Bail with a clear error pointing at `--tokenizer`

use std::collections::VecDeque;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::time::Instant;

use colored::*;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;

use crate::commands::run::RunCommand;
use crate::config::CliConfig;
use candle_core::{DType, Device, Tensor};
use ferrum_models::gguf_runtime::{GgufArch, GgufRuntime};
use ferrum_types::{FerrumError, Result};
use tokenizers::Tokenizer;

// ────────────────────────────────────────────────────────────────────────
// Public entry — dispatched from `run::execute` when the model arg is
// a `.gguf` path.
// ────────────────────────────────────────────────────────────────────────

pub async fn run_gguf_one_shot(cmd: RunCommand, _config: CliConfig) -> Result<()> {
    let gguf_path = PathBuf::from(&cmd.model);
    let device = select_device(&cmd.backend)?;
    eprintln!("{} backend: {}", "→".cyan(), device_label(&device).bold());

    let dtype_for_moe = match &device {
        Device::Cpu => DType::F32,
        _ => DType::F16,
    };

    let load_start = Instant::now();
    eprintln!(
        "{} loading {} ...",
        "→".cyan(),
        gguf_path.display().to_string().bold()
    );
    let mut runtime = GgufRuntime::open(&gguf_path, &device, dtype_for_moe)
        .map_err(|e| FerrumError::model(format!("GgufRuntime::open: {e}")))?;
    let arch = runtime.arch();
    eprintln!(
        "{} loaded in {:.2}s (arch: {})",
        "✓".green(),
        load_start.elapsed().as_secs_f64(),
        arch_label(arch).bold()
    );

    let tokenizer_path = cmd
        .tokenizer
        .clone()
        .or_else(|| auto_discover_tokenizer(&gguf_path))
        .ok_or_else(|| {
            FerrumError::model(format!(
                "could not find tokenizer for {} — pass --tokenizer <tokenizer.json> or place it next to the .gguf file",
                gguf_path.display()
            ))
        })?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| FerrumError::model(format!("tokenizer load: {e}")))?;
    eprintln!(
        "{} tokenizer: {}",
        "→".cyan(),
        tokenizer_path.display().to_string().dimmed()
    );

    if let Some(prompt) = cmd.prompt.clone() {
        // One-shot mode: encode the prompt as-is (no chat template) and
        // run a single prefill + decode pass. Bench-friendly.
        run_one_shot(&mut runtime, &tokenizer, &device, &prompt, &cmd).await?;
    } else {
        // Interactive REPL with chat template.
        run_repl(
            &mut runtime,
            &tokenizer,
            &device,
            &gguf_path,
            tokenizer_path,
            &cmd,
        )
        .await?;
    }

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────
// Generation modes
// ────────────────────────────────────────────────────────────────────────

async fn run_one_shot(
    runtime: &mut GgufRuntime,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    cmd: &RunCommand,
) -> Result<()> {
    let prompt_tokens: Vec<u32> = tokenizer
        .encode(prompt, true)
        .map_err(|e| FerrumError::model(format!("encode: {e}")))?
        .get_ids()
        .to_vec();
    eprintln!(
        "{} prompt: {} tokens",
        "→".cyan(),
        prompt_tokens.len().to_string().bold()
    );

    let prefill_start = Instant::now();
    let prefill_tensor = build_input_tensor(&prompt_tokens, device)?;
    let mut logits = runtime
        .forward(&prefill_tensor, 0)
        .map_err(|e| FerrumError::model(format!("prefill forward: {e}")))?;
    let prefill_secs = prefill_start.elapsed().as_secs_f64();
    eprintln!(
        "{} prefill: {} tok in {:.3}s ({:.1} tok/s)",
        "✓".green(),
        prompt_tokens.len(),
        prefill_secs,
        prompt_tokens.len() as f64 / prefill_secs.max(1e-6)
    );

    let mut rng = StdRng::seed_from_u64(cmd.seed);
    let sampling = SamplingParams::from(cmd);
    let eos_id = infer_eos_token(tokenizer, runtime.arch());

    // Recent-token ring for repetition penalty.
    let mut recent: VecDeque<u32> = VecDeque::with_capacity(cmd.repeat_last_n.max(1));
    seed_recent(&mut recent, &prompt_tokens, cmd.repeat_last_n);

    let mut next_token = sample_step(&logits, &recent, &sampling, &mut rng)?;
    let mut generated: Vec<u32> = Vec::with_capacity(cmd.max_tokens as usize);
    generated.push(next_token);
    push_recent(&mut recent, next_token, cmd.repeat_last_n);

    if !cmd.bench_mode {
        eprintln!("{} generating ...", "→".cyan());
        print!("{}", prompt);
        if let Ok(s) = tokenizer.decode(&[next_token], true) {
            print!("{}", s);
            io::stdout().flush().ok();
        }
    }

    let decode_start = Instant::now();
    let max_new = cmd.max_tokens as usize;
    for step in 1..max_new {
        if Some(next_token) == eos_id {
            break;
        }
        let pos = prompt_tokens.len() + step - 1;
        let input = build_input_tensor(&[next_token], device)?;
        logits = runtime
            .forward(&input, pos)
            .map_err(|e| FerrumError::model(format!("decode forward: {e}")))?;
        next_token = sample_step(&logits, &recent, &sampling, &mut rng)?;
        generated.push(next_token);
        push_recent(&mut recent, next_token, cmd.repeat_last_n);

        if !cmd.bench_mode {
            if let Ok(s) = tokenizer.decode(&[next_token], true) {
                print!("{}", s);
                io::stdout().flush().ok();
            }
        }
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();
    if !cmd.bench_mode {
        println!();
    }

    print_timing_report(
        prompt_tokens.len(),
        generated.len(),
        prefill_secs,
        decode_secs,
    );
    Ok(())
}

async fn run_repl(
    runtime: &mut GgufRuntime,
    tokenizer: &Tokenizer,
    device: &Device,
    gguf_path: &PathBuf,
    tokenizer_path: PathBuf,
    cmd: &RunCommand,
) -> Result<()> {
    if !io::stdin().is_terminal() {
        return Err(FerrumError::model(
            "REPL needs a TTY. Pipe input via --prompt instead.".to_string(),
        ));
    }

    let arch = runtime.arch();
    let template = ChatTemplate::for_arch(arch);
    let system_prompt = cmd
        .system
        .clone()
        .unwrap_or_else(|| "You are a helpful assistant.".to_string());

    eprintln!();
    eprintln!(
        "{}  arch={} system={:?}",
        "REPL".bold().green(),
        arch_label(arch).bold(),
        system_prompt
    );
    eprintln!(
        "{}  type {} to exit, {} to reset KV cache, {} to switch system prompt",
        "tips:".dimmed(),
        "/exit".bold(),
        "/clear".bold(),
        "/system <text>".bold()
    );
    eprintln!();

    // Prefill the system prompt + assistant priming so subsequent user
    // turns don't have to re-emit it.
    let mut current_system = system_prompt;
    let mut cache_len = prefill_system(runtime, tokenizer, device, &template, &current_system)?;

    let mut rng = StdRng::seed_from_u64(cmd.seed);
    let sampling = SamplingParams::from(cmd);
    let eos_id = infer_eos_token(tokenizer, arch);
    let dtype_for_moe = match device {
        Device::Cpu => DType::F32,
        _ => DType::F16,
    };
    let stdin = io::stdin();
    let mut user_input = String::new();

    loop {
        // Prompt
        print!("{} ", "❯".cyan().bold());
        io::stdout().flush().ok();

        user_input.clear();
        let bytes = stdin
            .lock()
            .read_line(&mut user_input)
            .map_err(|e| FerrumError::model(format!("REPL stdin read: {e}")))?;
        if bytes == 0 {
            // EOF (Ctrl-D)
            println!();
            break;
        }
        let line = user_input.trim();
        if line.is_empty() {
            continue;
        }

        // ── REPL commands ─────────────────────────────────────────────
        if line == "/exit" || line == "/quit" {
            break;
        }
        if line == "/clear" {
            eprintln!("{} resetting KV cache + reloading model ...", "↻".yellow());
            *runtime = GgufRuntime::open(gguf_path, device, dtype_for_moe)
                .map_err(|e| FerrumError::model(format!("reload GgufRuntime: {e}")))?;
            cache_len = prefill_system(runtime, tokenizer, device, &template, &current_system)?;
            continue;
        }
        if let Some(rest) = line.strip_prefix("/system ") {
            current_system = rest.trim().to_string();
            eprintln!(
                "{} system prompt updated → reloading for clean cache",
                "↻".yellow()
            );
            *runtime = GgufRuntime::open(gguf_path, device, dtype_for_moe)
                .map_err(|e| FerrumError::model(format!("reload GgufRuntime: {e}")))?;
            cache_len = prefill_system(runtime, tokenizer, device, &template, &current_system)?;
            continue;
        }
        if line == "/help" {
            eprintln!(
                "{}  /exit  /clear  /system <text>  /tokenizer (path: {})",
                "commands:".bold(),
                tokenizer_path.display()
            );
            continue;
        }

        // ── Append user turn ──────────────────────────────────────────
        let user_text = template.user_turn(line);
        let user_ids: Vec<u32> = tokenizer
            .encode(user_text.as_str(), false)
            .map_err(|e| FerrumError::model(format!("encode user: {e}")))?
            .get_ids()
            .to_vec();
        let user_input_tensor = build_input_tensor(&user_ids, device)?;
        let mut logits = runtime
            .forward(&user_input_tensor, cache_len)
            .map_err(|e| FerrumError::model(format!("user forward: {e}")))?;
        cache_len += user_ids.len();

        // ── Decode assistant response ─────────────────────────────────
        let mut recent: VecDeque<u32> = VecDeque::with_capacity(cmd.repeat_last_n.max(1));
        seed_recent(&mut recent, &user_ids, cmd.repeat_last_n);

        print!("{} ", "▎".magenta());
        io::stdout().flush().ok();

        let decode_start = Instant::now();
        let max_new = cmd.max_tokens as usize;
        let mut produced: Vec<u32> = Vec::new();
        let mut next = sample_step(&logits, &recent, &sampling, &mut rng)?;
        for step in 0..max_new {
            if Some(next) == eos_id {
                break;
            }
            produced.push(next);
            push_recent(&mut recent, next, cmd.repeat_last_n);
            if let Ok(s) = tokenizer.decode(&[next], true) {
                print!("{}", s);
                io::stdout().flush().ok();
            }
            let pos = cache_len + step;
            let next_input = build_input_tensor(&[next], device)?;
            logits = runtime
                .forward(&next_input, pos)
                .map_err(|e| FerrumError::model(format!("decode forward: {e}")))?;
            next = sample_step(&logits, &recent, &sampling, &mut rng)?;
        }
        let decode_secs = decode_start.elapsed().as_secs_f64();
        cache_len += produced.len();

        println!();
        eprintln!(
            "{} {} tok / {:.2}s = {:.1} tok/s (cache: {} tok)",
            "↳".dimmed(),
            produced.len(),
            decode_secs,
            produced.len() as f64 / decode_secs.max(1e-6),
            cache_len
        );
    }

    Ok(())
}

/// Run the system message + assistant priming through the model so the
/// KV cache contains everything up to the assistant's first response
/// position. Returns the new cache length.
fn prefill_system(
    runtime: &mut GgufRuntime,
    tokenizer: &Tokenizer,
    device: &Device,
    template: &ChatTemplate,
    system: &str,
) -> Result<usize> {
    let opener = template.system_open(system);
    if opener.is_empty() {
        return Ok(0);
    }
    let ids: Vec<u32> = tokenizer
        .encode(opener.as_str(), true)
        .map_err(|e| FerrumError::model(format!("encode system: {e}")))?
        .get_ids()
        .to_vec();
    let tensor = build_input_tensor(&ids, device)?;
    runtime
        .forward(&tensor, 0)
        .map_err(|e| FerrumError::model(format!("system prefill: {e}")))?;
    Ok(ids.len())
}

// ────────────────────────────────────────────────────────────────────────
// Sampling
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct SamplingParams {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
}

impl From<&RunCommand> for SamplingParams {
    fn from(c: &RunCommand) -> Self {
        Self {
            temperature: c.temperature,
            top_k: c.top_k,
            top_p: c.top_p,
            repeat_penalty: c.repeat_penalty,
        }
    }
}

/// One sampling step: pull the last position's logits, apply repetition
/// penalty, optional top-K + top-P, then either greedy (temp=0) or
/// multinomial sample.
fn sample_step(
    logits: &Tensor,
    recent: &VecDeque<u32>,
    p: &SamplingParams,
    rng: &mut StdRng,
) -> Result<u32> {
    let last = logits
        .squeeze(0)
        .and_then(|t| {
            let dims = t.dims();
            if dims.len() == 2 {
                t.get(dims[0] - 1)
            } else {
                Ok(t)
            }
        })
        .map_err(|e| FerrumError::model(format!("logits squeeze: {e}")))?;
    let mut row: Vec<f32> = last
        .to_vec1()
        .map_err(|e| FerrumError::model(format!("to_vec1: {e}")))?;

    apply_repeat_penalty(&mut row, recent, p.repeat_penalty);

    if p.temperature <= 0.0 {
        // Greedy — fast path, deterministic.
        let (idx, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| FerrumError::model("empty logits".to_string()))?;
        return Ok(idx as u32);
    }

    // Temperature scaling
    if (p.temperature - 1.0).abs() > 1e-6 {
        for v in row.iter_mut() {
            *v /= p.temperature;
        }
    }

    // Stable softmax → probs
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }

    // Top-K filter
    if p.top_k > 0 && p.top_k < probs.len() {
        // Find threshold = K-th largest
        let mut sorted: Vec<f32> = probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[p.top_k - 1];
        for v in probs.iter_mut() {
            if *v < threshold {
                *v = 0.0;
            }
        }
    }

    // Top-P (nucleus) filter
    if p.top_p > 0.0 && p.top_p < 1.0 {
        // Sort indices by prob desc, accumulate, drop tail outside nucleus
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|a, b| {
            probs[*b]
                .partial_cmp(&probs[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cum = 0.0_f32;
        let mut keep = vec![false; probs.len()];
        for &i in &indices {
            keep[i] = true;
            cum += probs[i];
            if cum >= p.top_p {
                break;
            }
        }
        for (i, k) in keep.iter().enumerate() {
            if !*k {
                probs[i] = 0.0;
            }
        }
    }

    // Renormalise
    let s: f32 = probs.iter().sum();
    if s > 0.0 {
        for v in probs.iter_mut() {
            *v /= s;
        }
    } else {
        // All filtered out — fallback to uniform over remaining 1-elem set:
        // pick the argmax of the original logits.
        return greedy_argmax(&row);
    }

    // Multinomial sample
    let dist = WeightedIndex::new(&probs)
        .map_err(|e| FerrumError::model(format!("WeightedIndex (probs={}): {e}", probs.len())))?;
    Ok(dist.sample(rng) as u32)
}

fn greedy_argmax(row: &[f32]) -> Result<u32> {
    let (idx, _) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| FerrumError::model("empty logits".to_string()))?;
    Ok(idx as u32)
}

/// Repetition penalty: divide logit by `penalty` if penalty > 1 (suppress)
/// or multiply if < 1 (encourage). Match the OpenAI/llama.cpp behaviour
/// of applying to the absolute value: `logit = logit / penalty` for
/// positive logits, `logit * penalty` for negative.
fn apply_repeat_penalty(row: &mut [f32], recent: &VecDeque<u32>, penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 {
        return;
    }
    for &id in recent.iter() {
        let i = id as usize;
        if i < row.len() {
            row[i] = if row[i] >= 0.0 {
                row[i] / penalty
            } else {
                row[i] * penalty
            };
        }
    }
}

fn seed_recent(recent: &mut VecDeque<u32>, tokens: &[u32], capacity: usize) {
    let cap = capacity.max(1);
    let start = tokens.len().saturating_sub(cap);
    for &t in &tokens[start..] {
        recent.push_back(t);
    }
}

fn push_recent(recent: &mut VecDeque<u32>, token: u32, capacity: usize) {
    if recent.len() >= capacity.max(1) {
        recent.pop_front();
    }
    recent.push_back(token);
}

// ────────────────────────────────────────────────────────────────────────
// Chat templates
// ────────────────────────────────────────────────────────────────────────

/// Per-arch chat template. Just enough to drive multi-turn correctly for
/// the three benchmark families.
struct ChatTemplate {
    system_open: fn(&str) -> String,
    user_turn: fn(&str) -> String,
}

impl ChatTemplate {
    fn for_arch(arch: GgufArch) -> Self {
        match arch {
            GgufArch::Qwen3 | GgufArch::Qwen3Moe => Self {
                system_open: qwen_system_open,
                user_turn: qwen_user_turn,
            },
            GgufArch::Llama => Self {
                system_open: llama_system_open,
                user_turn: llama_user_turn,
            },
        }
    }

    fn system_open(&self, system: &str) -> String {
        (self.system_open)(system)
    }

    fn user_turn(&self, user: &str) -> String {
        (self.user_turn)(user)
    }
}

fn qwen_system_open(system: &str) -> String {
    if system.is_empty() {
        // Many Qwen3 GGUFs default to a nameless system; emit a minimal
        // template anyway so the assistant tag closes properly.
        return "<|im_start|>assistant\n".to_string();
    }
    format!("<|im_start|>system\n{system}<|im_end|>\n")
}

fn qwen_user_turn(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

fn llama_system_open(system: &str) -> String {
    if system.is_empty() {
        return "<|begin_of_text|>".to_string();
    }
    format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>")
}

fn llama_user_turn(user: &str) -> String {
    format!(
        "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

// ────────────────────────────────────────────────────────────────────────
// Misc helpers
// ────────────────────────────────────────────────────────────────────────

fn build_input_tensor(tokens: &[u32], device: &Device) -> Result<Tensor> {
    Tensor::new(tokens, device)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| FerrumError::model(format!("input tensor: {e}")))
}

fn auto_discover_tokenizer(gguf_path: &PathBuf) -> Option<PathBuf> {
    let dir = gguf_path.parent()?;
    if let Some(stem) = gguf_path.file_stem() {
        let candidate = dir.join(format!("{}.tokenizer.json", stem.to_string_lossy()));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let bare = dir.join("tokenizer.json");
    if bare.is_file() {
        return Some(bare);
    }
    None
}

fn infer_eos_token(tokenizer: &Tokenizer, arch: GgufArch) -> Option<u32> {
    // Search per-arch first, then fall back to common variants.
    let preferred: &[&str] = match arch {
        GgufArch::Qwen3 | GgufArch::Qwen3Moe => &["<|im_end|>", "<|endoftext|>"],
        GgufArch::Llama => &["<|eot_id|>", "<|end_of_text|>"],
    };
    for c in preferred {
        if let Some(id) = tokenizer.token_to_id(c) {
            return Some(id);
        }
    }
    for c in [
        "<|im_end|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|endoftext|>",
    ] {
        if let Some(id) = tokenizer.token_to_id(c) {
            return Some(id);
        }
    }
    None
}

fn select_device(backend: &str) -> Result<Device> {
    match backend.to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "metal" => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                Device::new_metal(0).map_err(|e| FerrumError::model(format!("metal device: {e}")))
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Err(FerrumError::model(
                    "Metal backend requested but build was not configured with --features metal on macOS"
                        .to_string(),
                ))
            }
        }
        "auto" | "" => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                Device::new_metal(0)
                    .or_else(|_| Ok::<_, candle_core::Error>(Device::Cpu))
                    .map_err(|e: candle_core::Error| {
                        FerrumError::model(format!("auto device: {e}"))
                    })
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Ok(Device::Cpu)
            }
        }
        other => Err(FerrumError::model(format!(
            "unknown backend '{other}' — use auto / cpu / metal"
        ))),
    }
}

fn device_label(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA".to_string(),
        Device::Metal(_) => "Metal".to_string(),
    }
}

fn arch_label(arch: GgufArch) -> &'static str {
    match arch {
        GgufArch::Qwen3 => "qwen3",
        GgufArch::Qwen3Moe => "qwen3moe",
        GgufArch::Llama => "llama (or qwen2/mistral via llama loader)",
    }
}

fn print_timing_report(
    prompt_tokens: usize,
    total_generated: usize,
    prefill_secs: f64,
    decode_secs: f64,
) {
    let decode_tokens = total_generated.saturating_sub(1); // first came from prefill
    let decode_tok_per_sec = decode_tokens as f64 / decode_secs.max(1e-6);
    eprintln!();
    eprintln!("{}", "─".repeat(60).dimmed());
    eprintln!(
        "{}: {} prompt + {} generated tok",
        "tokens".bold(),
        prompt_tokens,
        total_generated
    );
    eprintln!(
        "{}: {:.3}s prefill + {:.3}s decode",
        "time".bold(),
        prefill_secs,
        decode_secs
    );
    eprintln!(
        "{}: {:.1} tok/s (decode only)",
        "throughput".bold().green(),
        decode_tok_per_sec
    );
    eprintln!(
        "{}: {:.2}ms / token",
        "latency".bold(),
        (decode_secs * 1000.0) / decode_tokens.max(1) as f64
    );
}
