//! GGUF inference path — invoked by `ferrum run <path.gguf>` when the
//! model argument is a `.gguf` file. Routes through ferrum's own
//! `LlamaFamilyModel<B>` runtime (Phase 1B/1C `GgufLoader<B>` →
//! `Backend<B>` kernels), **not** candle-transformers' forward path.
//! candle is touched only inside `GgufLoader`'s tensor read +
//! dequantize step — every line of math from `.forward()` onward
//! runs through ferrum's own kernels.
//!
//! Two modes share the implementation:
//!   - **One-shot** (`--prompt` set): prefill + decode + exit. `--bench-mode`
//!     suppresses generation output, prints only timing.
//!   - **REPL** (no `--prompt`): multi-turn chat with proper per-arch chat
//!     templates (Qwen3 `<|im_start|>` / Llama-3 `<|start_header_id|>`),
//!     `/clear` to reset the KV cache for the active model.
//!
//! Tokenizer auto-discovery (in order): `--tokenizer <path>` →
//! `<gguf-stem>.tokenizer.json` → `tokenizer.json` next to the gguf.

use std::collections::VecDeque;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use colored::*;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rngs::StdRng;

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_models::common::llm::DecoderOnlyLLM;
use ferrum_models::models::llama_family::{LlamaFamilyConfig, LlamaFamilyModel};
use ferrum_models::models::qwen3_moe::Qwen3MoeModel;
use ferrum_models::moe_config::Qwen3MoeConfig;
use ferrum_quantization::gguf::{GgufFile, GgufLoader};
use ferrum_types::{FerrumError, Result};
use tokenizers::Tokenizer;

use crate::commands::run::RunCommand;
use crate::config::CliConfig;

#[cfg(all(target_os = "macos", feature = "metal"))]
use ferrum_kernels::backend::metal::MetalBackend;

// ────────────────────────────────────────────────────────────────────────
// Public entry — dispatched from `run::execute` when the model arg is
// a `.gguf` path.
// ────────────────────────────────────────────────────────────────────────

pub async fn run_gguf_one_shot(cmd: RunCommand, _config: CliConfig) -> Result<()> {
    let gguf_path = PathBuf::from(&cmd.model);
    let backend_kind = resolve_backend(&cmd.backend)?;

    eprintln!("{} backend: {}", "→".cyan(), backend_kind.label().bold());

    let load_start = Instant::now();
    let gguf = GgufFile::open(&gguf_path)
        .map_err(|e| FerrumError::model(format!("GgufFile::open: {e}")))?;

    // Architecture-aware config parse: dense Qwen3 / Llama goes through
    // `LlamaFamilyConfig::from_gguf`; MoE variants (qwen3moe) need
    // `Qwen3MoeConfig::from_gguf` since the MoE-specific keys (expert
    // count, top-K, expert FFN width) live alongside the dense fields.
    let arch_str = gguf
        .architecture()
        .map_err(|e| FerrumError::model(format!("read arch: {e}")))?
        .to_string();
    let is_moe = arch_str == "qwen3moe";

    // Parse the right config flavour. Both end up exposing num_layers /
    // hidden_size / kv_heads for the load-time banner.
    let (dense_cfg, moe_cfg) = if is_moe {
        let mc = Qwen3MoeConfig::from_gguf(&gguf)?;
        (None, Some(mc))
    } else {
        let dc = LlamaFamilyConfig::from_gguf(&gguf)?;
        (Some(dc), None)
    };
    let (n_layers, hidden, kv_heads) = if let Some(c) = dense_cfg.as_ref() {
        (c.num_layers, c.hidden_size, c.num_kv_heads)
    } else {
        let c = moe_cfg.as_ref().unwrap();
        (c.base.num_layers, c.base.hidden_size, c.base.num_kv_heads)
    };
    let cfg = dense_cfg
        .clone()
        .unwrap_or_else(|| moe_cfg.as_ref().unwrap().base.clone());

    if let Some(c) = moe_cfg.as_ref() {
        eprintln!(
            "{} GGUF parsed in {:.2}s — MoE arch detected, {} layers, hidden={}, kv_heads={}, experts={}, top_k={}, expert_inter={}",
            "✓".green(),
            load_start.elapsed().as_secs_f64(),
            n_layers,
            hidden,
            kv_heads,
            c.num_experts,
            c.num_experts_per_tok,
            c.expert_intermediate_size,
        );
    } else {
        eprintln!(
            "{} GGUF parsed in {:.2}s — arch detected, {} layers, hidden={}, kv_heads={}",
            "✓".green(),
            load_start.elapsed().as_secs_f64(),
            n_layers,
            hidden,
            kv_heads
        );
    }

    // Tokenizer (auto-discover if not provided)
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

    // Detect arch label (for chat template dispatch in REPL mode)
    let arch_label = detect_arch(&gguf)?;
    eprintln!(
        "{} loading weights into {} (arch: {}) ...",
        "→".cyan(),
        backend_kind.label(),
        arch_label.bold()
    );

    let model_load_start = Instant::now();
    let gguf_arc = Arc::new(gguf);

    match backend_kind {
        BackendKind::Cpu => {
            let loader = GgufLoader::<CpuBackend>::from_file(gguf_arc.clone());
            if let Some(mc) = moe_cfg.clone() {
                let mut model = Qwen3MoeModel::<CpuBackend>::new(mc, &loader, &gguf_arc)?;
                let model_load_secs = model_load_start.elapsed().as_secs_f64();
                eprintln!("{} model ready in {:.2}s", "✓".green(), model_load_secs);
                run_inference(
                    &mut model,
                    &tokenizer,
                    &cmd,
                    &arch_label,
                    tokenizer_path,
                    &gguf_path,
                    BackendKind::Cpu,
                )?;
            } else {
                let mut model = LlamaFamilyModel::<CpuBackend>::new(cfg, &loader)?;
                let model_load_secs = model_load_start.elapsed().as_secs_f64();
                eprintln!("{} model ready in {:.2}s", "✓".green(), model_load_secs);
                run_inference(
                    &mut model,
                    &tokenizer,
                    &cmd,
                    &arch_label,
                    tokenizer_path,
                    &gguf_path,
                    BackendKind::Cpu,
                )?;
            }
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        BackendKind::Metal => {
            let loader = GgufLoader::<MetalBackend>::from_file(gguf_arc.clone());
            if let Some(mc) = moe_cfg.clone() {
                let mut model = Qwen3MoeModel::<MetalBackend>::new(mc, &loader, &gguf_arc)?;
                let model_load_secs = model_load_start.elapsed().as_secs_f64();
                eprintln!("{} model ready in {:.2}s", "✓".green(), model_load_secs);
                run_inference(
                    &mut model,
                    &tokenizer,
                    &cmd,
                    &arch_label,
                    tokenizer_path,
                    &gguf_path,
                    BackendKind::Metal,
                )?;
            } else {
                let mut model = LlamaFamilyModel::<MetalBackend>::new(cfg, &loader)?;
                let model_load_secs = model_load_start.elapsed().as_secs_f64();
                eprintln!("{} model ready in {:.2}s", "✓".green(), model_load_secs);
                run_inference(
                    &mut model,
                    &tokenizer,
                    &cmd,
                    &arch_label,
                    tokenizer_path,
                    &gguf_path,
                    BackendKind::Metal,
                )?;
            }
        }
    }

    Ok(())
}

fn detect_arch(gguf: &GgufFile) -> Result<String> {
    gguf.architecture()
        .map(|s| s.to_string())
        .map_err(|e| FerrumError::model(format!("read arch: {e}")))
}

// ────────────────────────────────────────────────────────────────────────
// Run loop (generic over any model implementing DecoderOnlyLLM)
// ────────────────────────────────────────────────────────────────────────

fn run_inference<M: DecoderOnlyLLM>(
    model: &mut M,
    tokenizer: &Tokenizer,
    cmd: &RunCommand,
    arch_label: &str,
    tokenizer_path: PathBuf,
    gguf_path: &PathBuf,
    _backend: BackendKind,
) -> Result<()> {
    if let Some(prompt) = cmd.prompt.clone() {
        run_one_shot(model, tokenizer, &prompt, cmd)
    } else {
        run_repl(model, tokenizer, cmd, arch_label, tokenizer_path, gguf_path)
    }
}

fn run_one_shot<M: DecoderOnlyLLM>(
    model: &mut M,
    tokenizer: &Tokenizer,
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

    let cache_id = "default";

    let prefill_start = Instant::now();
    let logits = model.prefill(cache_id, &prompt_tokens);
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

    let mut recent: VecDeque<u32> = VecDeque::with_capacity(cmd.repeat_last_n.max(1));
    seed_recent(&mut recent, &prompt_tokens, cmd.repeat_last_n);

    let mut next_token = sample_logits(&logits, &recent, &sampling, &mut rng)?;
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

    let eos_id = infer_eos_token(tokenizer);

    let decode_start = Instant::now();
    let max_new = cmd.max_tokens as usize;
    for step in 1..max_new {
        if Some(next_token) == eos_id {
            break;
        }
        let pos = (prompt_tokens.len() + step - 1) as u32;
        let logits = model.decode(cache_id, next_token, pos);
        next_token = sample_logits(&logits, &recent, &sampling, &mut rng)?;
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

fn run_repl<M: DecoderOnlyLLM>(
    model: &mut M,
    tokenizer: &Tokenizer,
    cmd: &RunCommand,
    arch_label: &str,
    tokenizer_path: PathBuf,
    _gguf_path: &PathBuf,
) -> Result<()> {
    if !io::stdin().is_terminal() {
        return Err(FerrumError::model(
            "REPL needs a TTY. Use --prompt for non-interactive runs.".to_string(),
        ));
    }

    let template = ChatTemplate::for_arch(arch_label);
    let system_prompt = cmd
        .system
        .clone()
        .unwrap_or_else(|| "You are a helpful assistant.".to_string());

    eprintln!();
    eprintln!(
        "{}  arch={} system={:?}",
        "REPL".bold().green(),
        arch_label.bold(),
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

    let cache_id = "repl";

    // Prefill the system prompt + assistant priming
    let mut current_system = system_prompt;
    let mut cache_len = prefill_system(model, tokenizer, &template, &current_system, cache_id)?;

    let mut rng = StdRng::seed_from_u64(cmd.seed);
    let sampling = SamplingParams::from(cmd);
    let eos_id = infer_eos_token(tokenizer);
    let stdin = io::stdin();
    let mut user_input = String::new();

    loop {
        print!("{} ", "❯".cyan().bold());
        io::stdout().flush().ok();

        user_input.clear();
        let bytes = stdin
            .lock()
            .read_line(&mut user_input)
            .map_err(|e| FerrumError::model(format!("REPL stdin: {e}")))?;
        if bytes == 0 {
            println!();
            break;
        }
        let line = user_input.trim();
        if line.is_empty() {
            continue;
        }

        if line == "/exit" || line == "/quit" {
            break;
        }
        if line == "/clear" {
            // Switch to a fresh cache_id so the model's internal KV map drops
            // the old one when collapsing per-cache state. Re-prefill system.
            eprintln!("{} resetting KV cache ...", "↻".yellow());
            // Use a unique cache_id every reset so we don't accidentally reuse
            // stale entries. The model's internal map will allocate fresh.
            // (LlamaFamilyModel keeps caches in a HashMap keyed by string.)
            let _ = cache_len;
            cache_len = prefill_system(model, tokenizer, &template, &current_system, "repl_reset")?;
            continue;
        }
        if let Some(rest) = line.strip_prefix("/system ") {
            current_system = rest.trim().to_string();
            eprintln!("{} system prompt updated, fresh cache ...", "↻".yellow());
            cache_len = prefill_system(
                model,
                tokenizer,
                &template,
                &current_system,
                "repl_sysreset",
            )?;
            continue;
        }
        if line == "/help" {
            eprintln!(
                "{}  /exit  /clear  /system <text>  (tokenizer: {})",
                "commands:".bold(),
                tokenizer_path.display()
            );
            continue;
        }

        // ── User turn → encode → forward ─────────────────────────────
        let user_text = template.user_turn(line);
        let user_ids: Vec<u32> = tokenizer
            .encode(user_text.as_str(), false)
            .map_err(|e| FerrumError::model(format!("encode user: {e}")))?
            .get_ids()
            .to_vec();
        // We can't call `prefill` again on the same cache_id without
        // resetting — but we can use `decode` per-token to extend.
        // Cleaner: split the user turn into prefill semantics by feeding
        // tokens one-at-a-time via `decode`. (LlamaFamilyModel's prefill
        // expects the WHOLE prompt; for incremental we use decode.)
        let mut logits = vec![0.0_f32];
        for (i, &t) in user_ids.iter().enumerate() {
            let pos = (cache_len + i) as u32;
            logits = model.decode(cache_id, t, pos);
        }
        cache_len += user_ids.len();

        let mut recent: VecDeque<u32> = VecDeque::with_capacity(cmd.repeat_last_n.max(1));
        seed_recent(&mut recent, &user_ids, cmd.repeat_last_n);

        print!("{} ", "▎".magenta());
        io::stdout().flush().ok();

        let decode_start = Instant::now();
        let max_new = cmd.max_tokens as usize;
        let mut produced: Vec<u32> = Vec::new();
        let mut next = sample_logits(&logits, &recent, &sampling, &mut rng)?;
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
            let pos = (cache_len + step) as u32;
            let logits = model.decode(cache_id, next, pos);
            next = sample_logits(&logits, &recent, &sampling, &mut rng)?;
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

fn prefill_system<M: DecoderOnlyLLM>(
    model: &mut M,
    tokenizer: &Tokenizer,
    template: &ChatTemplate,
    system: &str,
    cache_id: &str,
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
    if ids.is_empty() {
        return Ok(0);
    }
    let _ = model.prefill(cache_id, &ids);
    Ok(ids.len())
}

// ────────────────────────────────────────────────────────────────────────
// Sampling — operates on Vec<f32> logits (not candle Tensor)
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

fn sample_logits(
    logits: &[f32],
    recent: &VecDeque<u32>,
    p: &SamplingParams,
    rng: &mut StdRng,
) -> Result<u32> {
    let mut row = logits.to_vec();
    apply_repeat_penalty(&mut row, recent, p.repeat_penalty);

    if p.temperature <= 0.0 {
        let (idx, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| FerrumError::model("empty logits".to_string()))?;
        return Ok(idx as u32);
    }

    if (p.temperature - 1.0).abs() > 1e-6 {
        for v in row.iter_mut() {
            *v /= p.temperature;
        }
    }

    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }

    if p.top_k > 0 && p.top_k < probs.len() {
        let mut sorted: Vec<f32> = probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[p.top_k - 1];
        for v in probs.iter_mut() {
            if *v < threshold {
                *v = 0.0;
            }
        }
    }

    if p.top_p > 0.0 && p.top_p < 1.0 {
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

    let s: f32 = probs.iter().sum();
    if s > 0.0 {
        for v in probs.iter_mut() {
            *v /= s;
        }
    } else {
        let (idx, _) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| FerrumError::model("all probs zero".to_string()))?;
        return Ok(idx as u32);
    }

    let dist = WeightedIndex::new(&probs)
        .map_err(|e| FerrumError::model(format!("WeightedIndex: {e}")))?;
    Ok(dist.sample(rng) as u32)
}

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

struct ChatTemplate {
    system_open: fn(&str) -> String,
    user_turn: fn(&str) -> String,
}

impl ChatTemplate {
    fn for_arch(arch: &str) -> Self {
        match arch {
            "qwen3" | "qwen3moe" | "qwen2" => Self {
                system_open: qwen_system_open,
                user_turn: qwen_user_turn,
            },
            "llama" | "tinyllama" | "mistral" => Self {
                system_open: llama_system_open,
                user_turn: llama_user_turn,
            },
            _ => Self {
                system_open: qwen_system_open, // sane default
                user_turn: qwen_user_turn,
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
        return String::new();
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
// Misc
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum BackendKind {
    Cpu,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal,
}

impl BackendKind {
    fn label(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Self::Metal => "Metal",
        }
    }
}

fn resolve_backend(s: &str) -> Result<BackendKind> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(BackendKind::Cpu),
        "metal" => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                Ok(BackendKind::Metal)
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Err(FerrumError::model(
                    "Metal backend requires --features metal on macOS".to_string(),
                ))
            }
        }
        "auto" | "" => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                Ok(BackendKind::Metal)
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Ok(BackendKind::Cpu)
            }
        }
        other => Err(FerrumError::model(format!(
            "unknown backend '{other}' — use auto / cpu / metal"
        ))),
    }
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

fn infer_eos_token(tokenizer: &Tokenizer) -> Option<u32> {
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

fn print_timing_report(
    prompt_tokens: usize,
    total_generated: usize,
    prefill_secs: f64,
    decode_secs: f64,
) {
    let decode_tokens = total_generated.saturating_sub(1);
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
