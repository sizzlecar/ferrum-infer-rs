//! Pull command - Download a model from HuggingFace Hub

use crate::config::CliConfig;
use clap::Args;
use colored::*;
use ferrum_models::HfDownloader;
use ferrum_types::Result;
use std::path::PathBuf;

#[derive(Args)]
pub struct PullCommand {
    /// Model name to download (e.g., tinyllama, qwen2.5:7b, TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    pub model: String,
}

pub async fn execute(cmd: PullCommand, config: CliConfig) -> Result<()> {
    // GGUF alias path — pulls just the requested quantization plus the
    // sidecar tokenizer.json so `serve` / `bench` can pick it up. Doing
    // this BEFORE `resolve_model_alias` keeps GGUF aliases like
    // `qwen3:8b-q4_k_m` from accidentally falling through to the
    // safetensors HF repo (`Qwen/Qwen3-8B-Q4_K_M`, which doesn't exist).
    if let Some((repo, filename)) = super::run::resolve_gguf_alias(&cmd.model) {
        println!(
            "{} {} (file: {})",
            "Pulling GGUF".cyan().bold(),
            repo,
            filename
        );
        let cache_dir = get_hf_cache_dir(&config);
        println!("{}", format!("Cache: {}", cache_dir.display()).dimmed());
        let token = std::env::var("HF_TOKEN")
            .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
            .ok();
        let downloader = HfDownloader::new(cache_dir.clone(), token.clone())?;
        let gguf_path = match downloader.download_gguf(&repo, None, &filename).await {
            Ok(path) => path,
            Err(e) => {
                eprintln!();
                eprintln!("{} Failed to pull GGUF: {}", "✗".red().bold(), e);
                return Err(e);
            }
        };

        // Some GGUF repos (e.g. Qwen/Qwen3-*-GGUF) don't host
        // tokenizer.json. Pull it from the safetensors sibling repo
        // (`<repo>` minus `-GGUF`) and drop it next to the gguf file so
        // serve / bench's auto_discover_tokenizer_path picks it up.
        let snapshot_dir = gguf_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| cache_dir.clone());
        if !snapshot_dir.join("tokenizer.json").is_file() {
            if let Some(sibling) = super::run::tokenizer_sibling_repo(&repo) {
                println!();
                println!(
                    "{} tokenizer.json missing in GGUF repo — fetching from {}",
                    "→".cyan(),
                    sibling.dimmed()
                );
                let dl2 = HfDownloader::new(cache_dir.clone(), token.clone())?;
                match dl2.download(&sibling, None).await {
                    Ok(sibling_dir) => {
                        // Copy tokenizer.json + tokenizer_config.json into
                        // the GGUF snapshot dir so they live next to the
                        // gguf file.
                        for tok_file in [
                            "tokenizer.json",
                            "tokenizer_config.json",
                            "special_tokens_map.json",
                            "chat_template.json",
                        ] {
                            let src = sibling_dir.join(tok_file);
                            if src.is_file() {
                                let dst = snapshot_dir.join(tok_file);
                                if let Err(e) = std::fs::copy(&src, &dst) {
                                    eprintln!(
                                        "{} could not copy {}: {}",
                                        "⚠".yellow(),
                                        tok_file,
                                        e
                                    );
                                }
                            }
                        }
                        println!(
                            "{} tokenizer placed at {}",
                            "✓".green(),
                            snapshot_dir.display().to_string().dimmed()
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "{} sibling tokenizer pull failed: {} — \
                             you'll need to provide tokenizer.json manually",
                            "⚠".yellow().bold(),
                            e
                        );
                    }
                }
            }
        }

        println!();
        println!("{} Model ready at:", "✓".green().bold());
        println!("  {}", gguf_path.display());
        println!();
        println!("{}", "Run with:".dimmed());
        println!("  ferrum serve {}", cmd.model.cyan());
        println!("  ferrum bench {} --concurrency 8", cmd.model.cyan());
        return Ok(());
    }

    let model_id = resolve_model_alias(&cmd.model);

    println!("{} {}", "Pulling".cyan().bold(), model_id);

    // Get cache directory
    let cache_dir = get_hf_cache_dir(&config);
    println!("{}", format!("Cache: {}", cache_dir.display()).dimmed());

    // Get HF token
    let token = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok();

    // Create downloader with proxy support
    let downloader = HfDownloader::new(cache_dir, token)?;

    // Download
    match downloader.download(&model_id, None).await {
        Ok(path) => {
            println!();
            println!("{} Model ready at:", "✓".green().bold());
            println!("  {}", path.display());
            Ok(())
        }
        Err(e) => {
            eprintln!();
            eprintln!("{} Failed to pull model: {}", "✗".red().bold(), e);
            eprintln!();
            eprintln!("Tips:");
            eprintln!("  • Check your internet connection");
            eprintln!("  • Set proxy: HTTPS_PROXY=socks5h://host:port");
            eprintln!("  • For private models, set HF_TOKEN environment variable");
            Err(e)
        }
    }
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

fn get_hf_cache_dir(config: &CliConfig) -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home);
    }
    let configured = shellexpand::tilde(&config.models.download.hf_cache_dir).to_string();
    PathBuf::from(configured)
}
