//! Run command - Interactive chat with a model (ollama-style)

use crate::config::CliConfig;
use chrono::Utc;
use clap::Args;
use colored::*;
use ferrum_models::source::{ModelFormat, ResolvedModelSource};
use ferrum_models::HfDownloader;
use ferrum_types::{InferenceRequest, Priority, RequestId, Result, SamplingParams};
use futures::StreamExt;
use std::collections::HashMap;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Args)]
pub struct RunCommand {
    /// Model name (alias like `qwen3:8b`, HF repo id, or path to a `.gguf` file).
    /// When the argument is a `.gguf` path, ferrum routes to candle-transformers'
    /// quantized loaders for the M1 Max bench path (Qwen3 / Qwen3-MoE / Llama).
    #[arg(default_value = "tinyllama")]
    pub model: String,

    /// System prompt (interactive chat mode only).
    #[arg(long)]
    pub system: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "512")]
    pub max_tokens: u32,

    /// Sampling temperature (0.0–2.0). 0.0 = greedy / argmax (deterministic,
    /// what you want for benchmarks). >0 = softmax sample with `--top-k`
    /// and `--top-p` filtering applied.
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Backend: auto, cpu, metal (default: auto)
    #[arg(long, default_value = "auto")]
    pub backend: String,

    /// One-shot prompt (skip interactive REPL). When supplied, ferrum runs a
    /// single prefill+decode and exits — useful for benchmarking and shell
    /// scripting. For `.gguf` paths, omitting this drops into the GGUF REPL.
    #[arg(long)]
    pub prompt: Option<String>,

    /// Path to a HuggingFace `tokenizer.json` (only used for `.gguf` paths).
    /// If omitted, ferrum looks for `<gguf-stem>.tokenizer.json` and then
    /// `tokenizer.json` next to the `.gguf` file.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Bench mode: skip generated text output, print only timing summary.
    /// Implies one-shot (`--prompt` is required).
    #[arg(long)]
    pub bench_mode: bool,

    /// Top-K sampling cutoff (0 disables — keep all). Only the K highest-
    /// probability tokens compete in the softmax sample. Default 50, a
    /// conservative value that filters obvious garbage without flattening
    /// the distribution.
    #[arg(long, default_value = "50")]
    pub top_k: usize,

    /// Top-P (nucleus) sampling cutoff (0.0 disables, 1.0 keeps all).
    /// Smallest set of tokens whose cumulative probability exceeds P
    /// is kept; the rest are zeroed before sampling. Default 0.95.
    #[arg(long, default_value = "0.95")]
    pub top_p: f32,

    /// Repetition penalty applied to logits before sampling. >1 discourages
    /// repeats, <1 encourages, 1.0 disables. OpenAI uses 1.1 typically.
    #[arg(long, default_value = "1.0")]
    pub repeat_penalty: f32,

    /// Number of recent tokens that the repetition penalty considers.
    /// Smaller = local repeat avoidance only.
    #[arg(long, default_value = "64")]
    pub repeat_last_n: usize,

    /// Random seed for sampling (when temperature > 0). Default 42.
    #[arg(long, default_value = "42")]
    pub seed: u64,
}

pub async fn execute(cmd: RunCommand, config: CliConfig) -> Result<()> {
    // GGUF fast path — no HF download, no candle weight loader, just hand
    // the file to candle-transformers' quantized loaders.
    if looks_like_gguf_path(&cmd.model) {
        return crate::commands::run_gguf::run_gguf_one_shot(cmd, config).await;
    }

    // Resolve model
    let model_id = resolve_model_alias(&cmd.model);
    eprintln!("{}", format!("Loading {}...", model_id).dimmed());

    // Find cached model or auto-download
    let cache_dir = get_hf_cache_dir(&config);
    let source = match find_cached_model(&cache_dir, &model_id) {
        Some(source) => source,
        None => {
            // Model not found, try to download automatically
            eprintln!(
                "{} Model '{}' not found locally, downloading...",
                "📥".cyan(),
                model_id
            );

            // Get HF token from environment
            let token = std::env::var("HF_TOKEN")
                .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
                .ok();

            // Create downloader and download
            let downloader = HfDownloader::new(cache_dir.clone(), token)?;
            let snapshot_path = downloader.download(&model_id, None).await?;

            // Now find the downloaded model
            let format = detect_format(&snapshot_path);
            if format == ModelFormat::Unknown {
                return Err(ferrum_types::FerrumError::model(
                    "Downloaded model has unknown format",
                ));
            }

            ResolvedModelSource {
                original: model_id.clone(),
                local_path: snapshot_path,
                format,
                from_cache: false,
            }
        }
    };

    // Set model path for engine
    // NOTE: std::env::set_var is unsafe on Rust 2024; keep it minimal and explicit.
    unsafe {
        std::env::set_var(
            "FERRUM_MODEL_PATH",
            source.local_path.to_string_lossy().to_string(),
        );
    }

    // Select device
    let device = select_device(&cmd.backend);
    eprintln!("{}", format!("Using {:?} backend", device).dimmed());

    // Create engine
    let engine_config = ferrum_engine::simple_engine_config(model_id.clone(), device);
    let engine = ferrum_engine::create_mvp_engine(engine_config).await?;

    // Print ready message
    eprintln!();
    eprintln!("{}", "Ready. Type your message and press Enter.".green());
    eprintln!("{}", "Use /bye or Ctrl+D to exit.".dimmed());
    eprintln!();

    // Interactive loop
    let mut history: Vec<(String, String)> = Vec::new(); // (role, content)
    let generating = Arc::new(AtomicBool::new(false));

    // If stdin is not a TTY (piped input), don't print prompts and just consume lines.
    // This enables: `printf "hi\n/bye\n" | ferrum run ...` for automation/profiling.
    let stdin_is_tty = io::stdin().is_terminal();
    let mut stdin = io::stdin().lock();

    loop {
        if stdin_is_tty {
            // Show prompt
            print!("{} ", ">>>".bright_green().bold());
            io::stdout().flush().unwrap();
        }

        // Read input
        let mut input = String::new();
        match stdin.read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let input = input.trim();
                if input.is_empty() {
                    continue;
                }
                if input == "/bye" || input == "exit" || input == "quit" {
                    break;
                }

                // Build prompt with history
                let prompt = build_chat_prompt(&history, input, cmd.system.as_deref(), &model_id);

                // Model-specific sampling defaults
                let model_lower = model_id.to_lowercase();
                let (default_top_p, default_top_k) = if model_lower.contains("qwen3") {
                    // Qwen3 non-thinking mode: top_p=0.8, top_k=20
                    (0.8, Some(20))
                } else {
                    (0.9, None)
                };

                // Create request
                let request = InferenceRequest {
                    id: RequestId(Uuid::new_v4()),
                    model_id: ferrum_types::ModelId(model_id.clone()),
                    prompt,
                    sampling_params: SamplingParams {
                        max_tokens: cmd.max_tokens as usize,
                        temperature: cmd.temperature,
                        top_p: default_top_p,
                        top_k: default_top_k,
                        repetition_penalty: 1.1,
                        stop_sequences: vec![
                            "<|im_end|>".to_string(),
                            "</s>".to_string(),
                            "<|endoftext|>".to_string(),
                        ],
                        ..Default::default()
                    },
                    stream: true,
                    priority: Priority::Normal,
                    client_id: None,
                    session_id: None,
                    created_at: Utc::now(),
                    metadata: HashMap::new(),
                };

                // Start generation
                generating.store(true, Ordering::SeqCst);
                let mut stream = engine.infer_stream(request).await?;
                let mut response = String::new();
                let start = std::time::Instant::now();
                let mut token_count = 0;

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            if !chunk.text.is_empty() {
                                print!("{}", chunk.text);
                                io::stdout().flush().unwrap();
                                response.push_str(&chunk.text);
                            }
                            if chunk.token.is_some() {
                                token_count += 1;
                            }
                            if chunk.finish_reason.is_some() {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("\n{} {}", "Error:".red(), e);
                            break;
                        }
                    }
                }

                generating.store(false, Ordering::SeqCst);

                // Print stats
                let elapsed = start.elapsed();
                let tps = if elapsed.as_secs_f64() > 0.0 {
                    token_count as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };
                println!();
                eprintln!(
                    "{}",
                    format!(
                        "[{} tokens, {:.1} tok/s, {:.1}s]",
                        token_count,
                        tps,
                        elapsed.as_secs_f64()
                    )
                    .dimmed()
                );
                eprintln!();

                // In non-interactive mode, don't wait for terminal formatting/spacing.
                if !stdin_is_tty {
                    io::stdout().flush().ok();
                    io::stderr().flush().ok();
                }

                // Add to history
                history.push(("user".to_string(), input.to_string()));
                let clean_response = response.trim().to_string();
                if !clean_response.is_empty() {
                    history.push(("assistant".to_string(), clean_response));
                }

                // Limit history
                while history.len() > 10 {
                    history.remove(0);
                }
            }
            Err(e) => {
                eprintln!("{} {}", "Error reading input:".red(), e);
                break;
            }
        }
    }

    eprintln!("{}", "Goodbye!".bright_yellow());
    Ok(())
}

pub fn resolve_model_alias(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "tinyllama" | "tiny" => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        "qwen2.5:0.5b" | "qwen:0.5b" => "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        "qwen2.5:1.5b" | "qwen:1.5b" => "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        "qwen2.5:3b" | "qwen:3b" => "Qwen/Qwen2.5-3B-Instruct".to_string(),
        "qwen2.5:7b" | "qwen:7b" => "Qwen/Qwen2.5-7B-Instruct".to_string(),
        "qwen3:0.6b" => "Qwen/Qwen3-0.6B".to_string(),
        "qwen3:1.7b" => "Qwen/Qwen3-1.7B".to_string(),
        "qwen3:4b" => "Qwen/Qwen3-4B".to_string(),
        "qwen2.5:3b-gptq" | "qwen2.5-3b-instruct-gptq-int4" => {
            "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4".to_string()
        }
        "llama3.2:1b" => "meta-llama/Llama-3.2-1B-Instruct".to_string(),
        "llama3.2:3b" => "meta-llama/Llama-3.2-3B-Instruct".to_string(),
        "whisper-tiny" | "whisper:tiny" => "openai/whisper-tiny".to_string(),
        "whisper-base" | "whisper:base" => "openai/whisper-base".to_string(),
        "whisper-small" | "whisper:small" => "openai/whisper-small".to_string(),
        "whisper-medium" | "whisper:medium" => "openai/whisper-medium".to_string(),
        "whisper-large-v3" | "whisper:large-v3" => "openai/whisper-large-v3".to_string(),
        "whisper-turbo" | "whisper:turbo" | "whisper-large-v3-turbo" => {
            "openai/whisper-large-v3-turbo".to_string()
        }
        _ => name.to_string(),
    }
}

pub fn get_hf_cache_dir(config: &CliConfig) -> PathBuf {
    // Check environment variable first
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home);
    }

    // Use config value
    let configured = shellexpand::tilde(&config.models.download.hf_cache_dir).to_string();
    PathBuf::from(configured)
}

pub fn find_cached_model(cache_dir: &PathBuf, model_id: &str) -> Option<ResolvedModelSource> {
    let repo_dir = cache_dir
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")));
    let snapshots_dir = repo_dir.join("snapshots");

    // Try refs/main first
    let ref_main = repo_dir.join("refs").join("main");
    if let Ok(rev) = std::fs::read_to_string(&ref_main) {
        let rev = rev.trim();
        if !rev.is_empty() {
            let snapshot = snapshots_dir.join(rev);
            if snapshot.exists() {
                let format = detect_format(&snapshot);
                if format != ModelFormat::Unknown {
                    return Some(ResolvedModelSource {
                        original: model_id.to_string(),
                        local_path: snapshot,
                        format,
                        from_cache: true,
                    });
                }
            }
        }
    }

    // Fallback: first snapshot directory
    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let format = detect_format(&path);
                if format != ModelFormat::Unknown {
                    return Some(ResolvedModelSource {
                        original: model_id.to_string(),
                        local_path: path,
                        format,
                        from_cache: true,
                    });
                }
            }
        }
    }

    None
}

pub fn detect_format(path: &PathBuf) -> ModelFormat {
    if path.join("model.safetensors").exists() || path.join("model.safetensors.index.json").exists()
    {
        ModelFormat::SafeTensors
    } else if path.join("pytorch_model.bin").exists() {
        ModelFormat::PyTorchBin
    } else {
        ModelFormat::Unknown
    }
}

/// Treat the model argument as a `.gguf` file path if it ends in `.gguf`
/// (case-insensitive) and the file actually exists. Anything else falls
/// through to the alias / HF repo path.
pub fn looks_like_gguf_path(model: &str) -> bool {
    let p = PathBuf::from(model);
    p.extension()
        .map(|e| e.eq_ignore_ascii_case("gguf"))
        .unwrap_or(false)
        && p.is_file()
}

pub fn select_device(backend: &str) -> ferrum_types::Device {
    match backend.to_lowercase().as_str() {
        "cpu" => ferrum_types::Device::CPU,
        "metal" => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return ferrum_types::Device::Metal;
            }
            #[allow(unreachable_code)]
            {
                eprintln!("Metal not available, falling back to CPU");
                ferrum_types::Device::CPU
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                return ferrum_types::Device::CUDA(0);
            }
            #[allow(unreachable_code)]
            {
                eprintln!("CUDA not available, falling back to CPU");
                ferrum_types::Device::CPU
            }
        }
        "auto" | _ => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return ferrum_types::Device::Metal;
            }
            #[cfg(feature = "cuda")]
            {
                return ferrum_types::Device::CUDA(0);
            }
            #[allow(unreachable_code)]
            ferrum_types::Device::CPU
        }
    }
}

fn build_chat_prompt(
    history: &[(String, String)],
    user_input: &str,
    system: Option<&str>,
    model_id: &str,
) -> String {
    // Detect model type and use appropriate template
    let model_lower = model_id.to_lowercase();

    if model_lower.contains("qwen") {
        // Qwen ChatML format (Qwen2, Qwen2.5, Qwen3)
        let mut prompt = String::new();
        if let Some(sys) = system {
            prompt.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
        }
        for (role, content) in history {
            prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }
        prompt.push_str(&format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_input
        ));
        // Qwen3: disable thinking mode by inserting empty think block
        if model_lower.contains("qwen3") {
            prompt.push_str("<think>\n\n</think>\n\n");
        }
        prompt
    } else if model_lower.contains("llama") && model_lower.contains("3") {
        // Llama 3 format
        let mut prompt = String::new();
        prompt.push_str("<|begin_of_text|>");
        if let Some(sys) = system {
            prompt.push_str(&format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                sys
            ));
        }
        for (role, content) in history {
            prompt.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                role, content
            ));
        }
        prompt.push_str(&format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            user_input
        ));
        prompt
    } else {
        // TinyLlama / generic chat format
        let sys = system.unwrap_or("You are a helpful assistant.");
        let mut prompt = format!("<|system|>\n{}</s>\n", sys);
        for (role, content) in history {
            let tag = if role == "user" { "user" } else { "assistant" };
            prompt.push_str(&format!("<|{}|>\n{}</s>\n", tag, content));
        }
        prompt.push_str(&format!("<|user|>\n{}</s>\n<|assistant|>\n", user_input));
        prompt
    }
}
