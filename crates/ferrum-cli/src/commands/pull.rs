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
