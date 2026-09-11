//! Pull command - Download a model from HuggingFace Hub

use crate::config::CliConfig;
use clap::Args;
use colored::*;
use ferrum_types::Result;

#[derive(Args)]
pub struct PullCommand {
    /// Model alias, owner/repository, or owner/repository:QUANT (for example :Q4_K_M).
    pub model: String,
    /// Exact repository-relative GGUF filename; MODEL may include @<40-hex-commit>.
    #[arg(long, value_name = "FILE")]
    pub gguf_file: Option<String>,
}

pub async fn execute(cmd: PullCommand, config: CliConfig) -> Result<()> {
    let cache_dir = crate::source_resolver::hf_cache_dir(&config);
    println!(
        "{} {}",
        "Pulling".cyan().bold(),
        crate::source_resolver::resolve_model_alias(&cmd.model)
    );
    println!("{}", format!("Cache: {}", cache_dir.display()).dimmed());

    let result = if let Some(filename) = cmd.gguf_file {
        crate::source_resolver::resolve_model_source_with_product_sources(
            &cmd.model,
            &cache_dir,
            crate::source_resolver::DownloadPolicy::AutoDownload,
            None,
            &crate::source_resolver::ProductSourceArgs {
                gguf_file: Some(filename),
                ..Default::default()
            },
        )
        .await
    } else {
        crate::source_resolver::resolve_model_source(
            &cmd.model,
            &cache_dir,
            crate::source_resolver::DownloadPolicy::AutoDownload,
            None,
        )
        .await
    };
    match result {
        Ok(resolved) => {
            println!();
            println!("{} Model ready at:", "✓".green().bold());
            println!("  {}", resolved.local_path().display());
            Ok(())
        }
        Err(error) => {
            eprintln!();
            eprintln!("{} Failed to pull model: {}", "✗".red().bold(), error);
            Err(error)
        }
    }
}
