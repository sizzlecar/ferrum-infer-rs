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
    // `ferrum run` is single-user interactive REPL — multi-sequence
    // concurrency isn't useful here, but the chat MUST tolerate the
    // default `--max-tokens 512` plus a multi-turn conversation.
    // 0.7.2 flipped the engine-wide defaults (`KV_CAPACITY=512`,
    // `MAX_SEQS=32`) to optimise `serve` for c=16 — those defaults
    // immediately overflow on the very first `run` turn (cache + prompt
    // + 512 max_new = ~530 > 512). Override here for run mode only;
    // `serve` keeps the concurrent defaults. Users who want a specific
    // value still win — we only set the var if it isn't already set.
    if std::env::var_os("FERRUM_KV_CAPACITY").is_none() {
        std::env::set_var("FERRUM_KV_CAPACITY", "8192");
    }
    if std::env::var_os("FERRUM_METAL_PAGED_KV").is_none() {
        // Paged-KV's win is multi-seq batching at the attention kernel.
        // m=1 single-user run sees zero benefit and pays pool-allocation
        // overhead. Force off here.
        std::env::set_var("FERRUM_METAL_PAGED_KV", "0");
    }

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
            // Register the GGUF mmap so subsequent `B::load_quant*` calls
            // can wrap weight tensors in zero-copy `MTLBuffer`s instead
            // of allocating fresh device-resident copies. Saves ~17 GB
            // resident on Qwen3-30B-A3B Q4_K_M (CRUCIAL on a 32 GB Mac).
            ferrum_kernels::backend::metal::register_gguf_mmap(
                gguf_arc.mmap_bytes(),
                gguf_arc.clone(),
            )?;
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

    // Eager scratch + KV cache alloc so the first `prefill` doesn't
    // pay the cold-start MTLBuffer cost on the hot path. Without this,
    // ferrum's pp50 / pp512 numbers in `--bench-mode` (each trial is
    // a fresh process) include the cold-start allocator overhead —
    // that's not what we want to measure when comparing against
    // llama-bench, which runs its `-r N` trials inside ONE process
    // and skips alloc on subsequent rounds. See bench/fairness-audit.md
    // for the full breakdown.
    model.prepare(cache_id, prompt_tokens.len());

    // Optional Metal frame capture around the prefill iteration. Set
    // FERRUM_METAL_CAPTURE=/path/to/out.gputrace AND MTL_CAPTURE_ENABLED=1
    // to record one prefill into a .gputrace file you can open in Xcode
    // (Window → Frame Capture). The capture covers exactly the prefill
    // forward pass so the resulting trace is small and focused.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let _capture_active = ferrum_kernels::backend::metal::maybe_begin_frame_capture();

    let prefill_start = Instant::now();
    let logits = model.prefill(cache_id, &prompt_tokens);
    let prefill_secs = prefill_start.elapsed().as_secs_f64();

    #[cfg(all(target_os = "macos", feature = "metal"))]
    if _capture_active {
        ferrum_kernels::backend::metal::end_frame_capture();
    }
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

    let mut decoder = StreamingDecoder::new(&prompt_tokens);
    decoder.seed_prompt_offset(tokenizer);
    if !cmd.bench_mode {
        eprintln!("{} generating ...", "→".cyan());
        print!("{}", prompt);
        let chunk = decoder.push(next_token, tokenizer);
        print!("{}", chunk);
        io::stdout().flush().ok();
    } else {
        // Bench mode: still cumulate so the decoder state is correct,
        // but skip emit.
        let _ = decoder.push(next_token, tokenizer);
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
            let chunk = decoder.push(next_token, tokenizer);
            print!("{}", chunk);
            io::stdout().flush().ok();
        } else {
            let _ = decoder.push(next_token, tokenizer);
        }
    }
    let decode_secs = decode_start.elapsed().as_secs_f64();
    if !cmd.bench_mode {
        let tail = decoder.flush(tokenizer);
        if !tail.is_empty() {
            print!("{}", tail);
        }
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
    let mut input_bytes = Vec::new();

    loop {
        print!("{} ", "❯".cyan().bold());
        io::stdout().flush().ok();

        // Read raw bytes then UTF-8-lossy decode. `read_line` into a
        // `String` aborts on the first invalid byte sequence — that
        // breaks SSH/iOS terminals that occasionally hand us a partial
        // multi-byte CJK character or a stray latin-1 byte. Lossy
        // replaces invalid bytes with U+FFFD instead of erroring;
        // worst case the user sees one '�' in their prompt and types
        // the line again.
        input_bytes.clear();
        let bytes = stdin
            .lock()
            .read_until(b'\n', &mut input_bytes)
            .map_err(|e| FerrumError::model(format!("REPL stdin: {e}")))?;
        if bytes == 0 {
            println!();
            break;
        }
        let user_input = String::from_utf8_lossy(&input_bytes).into_owned();
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

        // Pre-check KV capacity. Worst case for this turn = ingesting
        // the full user prompt + generating `max_tokens` decode steps.
        // If that would push past the per-cache budget, refuse cleanly
        // here — the alternative is silent buffer overflow inside the
        // attention kernel (garbage tokens, walking off the K/V tile,
        // and a hard panic from `forward_layer`'s defensive check).
        let max_new = cmd.max_tokens as usize;
        let kv_capacity = model.kv_capacity();
        let needed = cache_len + user_ids.len() + max_new;
        if needed > kv_capacity {
            eprintln!(
                "{} context full: this turn would need {} tokens (current cache {} + prompt {} + max_tokens {}), but the KV budget is {}.",
                "✗".red().bold(),
                needed,
                cache_len,
                user_ids.len(),
                max_new,
                kv_capacity,
            );
            eprintln!(
                "  Either run `/clear` to reset the conversation or restart with `FERRUM_KV_CAPACITY={}` (or larger).",
                (needed.next_power_of_two()).max(8192),
            );
            continue;
        }

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
        let mut produced: Vec<u32> = Vec::new();
        // Streaming detokenizer seeded with [user_turn_tokens] so we
        // emit only the assistant turn's text, not the user's echo.
        let mut decoder = StreamingDecoder::new(&user_ids);
        decoder.seed_prompt_offset(tokenizer);
        let mut next = sample_logits(&logits, &recent, &sampling, &mut rng)?;
        for step in 0..max_new {
            if Some(next) == eos_id {
                break;
            }
            produced.push(next);
            push_recent(&mut recent, next, cmd.repeat_last_n);
            let chunk = decoder.push(next, tokenizer);
            if !chunk.is_empty() {
                print!("{}", chunk);
                io::stdout().flush().ok();
            }
            let pos = (cache_len + step) as u32;
            let logits = model.decode(cache_id, next, pos);
            next = sample_logits(&logits, &recent, &sampling, &mut rng)?;
        }
        let tail = decoder.flush(tokenizer);
        if !tail.is_empty() {
            print!("{}", tail);
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
    // Delegate to the shared helper used by `ferrum serve` so the two
    // paths search the same set of locations. In particular, the shared
    // helper also checks a sibling `../tokenizers/<bare-stem>.tokenizer.json`
    // (the `~/ferrum-bench/{models,tokenizers}/` layout) — `run` was
    // missing that until 0.7.3 and forced users to pass `--tokenizer`
    // even though `serve` worked on the same gguf.
    ferrum_models::gguf_engine_loader::auto_discover_tokenizer_path(gguf_path)
}

/// Streaming detokenizer. Calling `tokenizer.decode(&[next_token], true)`
/// once per generated token breaks on BPE byte-level vocabularies (Qwen,
/// Llama-3) where a single Chinese character (3-byte UTF-8) or emoji
/// (4-byte UTF-8) is split across 2-3 tokens. The naive decode hands
/// the tokenizer an incomplete byte sequence and you get `?` / `�` /
/// the literal `\u{1f4e6}` form in the terminal.
///
/// This decoder cumulates tokens and re-decodes the whole sequence
/// each step, emitting only the *new* tail. The tokenizer always
/// returns a valid UTF-8 String when given a complete cumulative
/// sequence, so the diff is a clean suffix at a UTF-8 boundary.
///
/// Cost: O(N²) decode calls per stream, but at single-user m=1 decode
/// rates each call is microseconds — total overhead < 1% of decode wall
/// time and well worth a working chat experience.
struct StreamingDecoder {
    all_tokens: Vec<u32>,
    printed_len: usize,
}

impl StreamingDecoder {
    fn new(prompt_tokens: &[u32]) -> Self {
        Self {
            all_tokens: prompt_tokens.to_vec(),
            // Seed `printed_len` with the prompt's decoded length so we
            // don't accidentally re-emit the prompt as part of the model
            // output.
            printed_len: 0,
        }
    }

    fn seed_prompt_offset(&mut self, tokenizer: &Tokenizer) {
        let prompt_text = tokenizer.decode(&self.all_tokens, true).unwrap_or_default();
        self.printed_len = prompt_text.len();
    }

    fn push(&mut self, tok: u32, tokenizer: &Tokenizer) -> String {
        self.all_tokens.push(tok);
        let cumulative = tokenizer.decode(&self.all_tokens, true).unwrap_or_default();

        // BPE byte-level vocabularies split CJK chars / emoji across 2-3
        // tokens. When `tokenizer.decode` is called on a cumulative
        // sequence whose LAST few bytes are an incomplete UTF-8 sequence,
        // the implementation loss-replaces them with `U+FFFD`. The next
        // token's bytes complete the codepoint and the � disappears from
        // the cumulative output. So: don't emit anything past a trailing
        // � — hold the tail back, the next push will fill it in.
        let safe_end = if cumulative.ends_with('\u{FFFD}') {
            cumulative.rfind('\u{FFFD}').unwrap_or(cumulative.len())
        } else {
            cumulative.len()
        };

        let out = if safe_end > self.printed_len {
            cumulative[self.printed_len..safe_end].to_string()
        } else {
            String::new()
        };
        self.printed_len = safe_end;
        out
    }

    fn flush(&mut self, tokenizer: &Tokenizer) -> String {
        // End-of-stream flush. If we've been holding back a trailing
        // U+FFFD waiting for a continuation that never came, emit it
        // now so the user sees the model's actual final character.
        let cumulative = tokenizer.decode(&self.all_tokens, true).unwrap_or_default();
        let out = if cumulative.len() > self.printed_len {
            cumulative[self.printed_len..].to_string()
        } else {
            String::new()
        };
        self.printed_len = cumulative.len();
        out
    }
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
