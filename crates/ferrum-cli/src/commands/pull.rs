//! Pull command - Download a model from HuggingFace Hub

use crate::config::CliConfig;
use clap::Args;
use colored::*;
use ferrum_types::Result;

#[derive(Args)]
pub struct PullCommand {
    /// Model name to download (e.g., tinyllama, qwen2.5:7b, TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    pub model: String,
}

pub async fn execute(cmd: PullCommand, config: CliConfig) -> Result<()> {
    let cache_dir = crate::source_resolver::hf_cache_dir(&config);
    println!(
        "{} {}",
        "Pulling".cyan().bold(),
        crate::source_resolver::resolve_model_alias(&cmd.model)
    );
    println!("{}", format!("Cache: {}", cache_dir.display()).dimmed());

    match crate::source_resolver::resolve_model_source(
        &cmd.model,
        &cache_dir,
        crate::source_resolver::DownloadPolicy::AutoDownload,
        None,
    )
    .await
    {
        Ok(resolved) => {
            println!();
            println!("{} Model ready at:", "✓".green().bold());
            println!("  {}", resolved.source.local_path.display());
            Ok(())
        }
        Err(error) => {
            eprintln!();
            eprintln!("{} Failed to pull model: {}", "✗".red().bold(), error);
            eprintln!("Set HF_TOKEN for private models and verify network/proxy settings.");
            Err(error)
        }
    }
}
