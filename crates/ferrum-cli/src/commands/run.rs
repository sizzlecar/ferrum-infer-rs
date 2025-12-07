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
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Args)]
pub struct RunCommand {
    /// Model name (e.g., tinyllama, qwen2.5:7b, or full path)
    #[arg(default_value = "tinyllama")]
    pub model: String,

    /// System prompt
    #[arg(long)]
    pub system: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "512")]
    pub max_tokens: u32,

    /// Temperature (0.0-2.0)
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Backend: auto, cpu, metal (default: auto)
    #[arg(long, default_value = "auto")]
    pub backend: String,
}

pub async fn execute(cmd: RunCommand, config: CliConfig) -> Result<()> {
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
                    "Downloaded model has unknown format"
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
    std::env::set_var("FERRUM_MODEL_PATH", source.local_path.to_string_lossy().to_string());

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

    loop {
        // Show prompt
        print!("{} ", ">>>".bright_green().bold());
        io::stdout().flush().unwrap();

        // Read input
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF (Ctrl+D)
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

                // Create request
                let request = InferenceRequest {
                    id: RequestId(Uuid::new_v4()),
                    model_id: ferrum_types::ModelId(model_id.clone()),
                    prompt,
                    sampling_params: SamplingParams {
                        max_tokens: cmd.max_tokens as usize,
                        temperature: cmd.temperature,
                        top_p: 0.9,
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

fn resolve_model_alias(name: &str) -> String {
    match name.to_lowercase().as_str() {
        "tinyllama" | "tiny" => "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
        "qwen2.5:0.5b" | "qwen:0.5b" => "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        "qwen2.5:1.5b" | "qwen:1.5b" => "Qwen/Qwen2.5-1.5B-Instruct".to_string(),
        "qwen2.5:3b" | "qwen:3b" => "Qwen/Qwen2.5-3B-Instruct".to_string(),
        "qwen2.5:7b" | "qwen:7b" => "Qwen/Qwen2.5-7B-Instruct".to_string(),
        "llama3.2:1b" => "meta-llama/Llama-3.2-1B-Instruct".to_string(),
        "llama3.2:3b" => "meta-llama/Llama-3.2-3B-Instruct".to_string(),
        _ => name.to_string(),
    }
}

fn get_hf_cache_dir(config: &CliConfig) -> PathBuf {
    // Check environment variable first
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home);
    }

    // Use config value
    let configured = shellexpand::tilde(&config.models.download.hf_cache_dir).to_string();
    PathBuf::from(configured)
}

fn find_cached_model(cache_dir: &PathBuf, model_id: &str) -> Option<ResolvedModelSource> {
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

fn detect_format(path: &PathBuf) -> ModelFormat {
    if path.join("model.safetensors").exists() || path.join("model.safetensors.index.json").exists()
    {
        ModelFormat::SafeTensors
    } else if path.join("pytorch_model.bin").exists() {
        ModelFormat::PyTorchBin
    } else {
        ModelFormat::Unknown
    }
}

fn select_device(backend: &str) -> ferrum_types::Device {
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
        "auto" | _ => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                return ferrum_types::Device::Metal;
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
        // Qwen ChatML format
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

