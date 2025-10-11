//! Models command implementation - Enhanced with vLLM-inspired model maintenance

use crate::{config::CliConfig, output::OutputFormat};
use clap::Args;
use colored::*;
use ferrum_types::Result;
use ferrum_models::ModelSourceResolver;

#[derive(Args)]
pub struct ModelsCommand {
    /// List available models
    #[arg(short, long)]
    pub list: bool,

    /// Show model details
    #[arg(long)]
    pub info: Option<String>,

    /// Download a model
    #[arg(long)]
    pub download: Option<String>,

    /// Remove a model
    #[arg(long)]
    pub remove: Option<String>,

    /// Validate model files
    #[arg(long)]
    pub validate: Option<String>,

    /// Show model aliases
    #[arg(long)]
    pub aliases: bool,

    /// Add model alias
    #[arg(long)]
    pub add_alias: Option<String>,

    /// Alias target (used with --add-alias)
    #[arg(long)]
    pub alias_target: Option<String>,
}

pub async fn execute(cmd: ModelsCommand, config: CliConfig, _format: OutputFormat) -> Result<()> {
    if cmd.list {
        return list_models(&config).await;
    }

    if let Some(model_name) = cmd.info {
        return show_model_info(&model_name, &config).await;
    }

    if let Some(model_name) = cmd.download {
        return download_model(&model_name, &config).await;
    }

    if let Some(model_name) = cmd.remove {
        return remove_model(&model_name, &config).await;
    }

    if let Some(model_name) = cmd.validate {
        return validate_model(&model_name, &config).await;
    }

    if cmd.aliases {
        return show_aliases(&config).await;
    }

    if let Some(alias_name) = cmd.add_alias {
        let target = cmd.alias_target.ok_or_else(|| {
            ferrum_types::FerrumError::invalid_request(
                "--alias-target is required with --add-alias".to_string(),
            )
        })?;
        return add_alias(&alias_name, &target, &config).await;
    }

    // Default: list models
    list_models(&config).await
}

async fn list_models(config: &CliConfig) -> Result<()> {
    println!("{} Available models", "📋".bright_blue());
    
    // Search in both model_dir and hf_cache_dir
    let mut registry = ferrum_models::DefaultModelRegistry::with_defaults();
    
    let mut all_models = Vec::new();
    
    // Search in configured model directory
    let models_dir = std::path::PathBuf::from(&config.models.model_dir);
    if models_dir.exists() {
        println!("📁 Searching: {}", models_dir.display());
        if let Ok(models) = registry.discover_models(&models_dir).await {
            all_models.extend(models);
        }
    }
    
    // Also search in HF cache directory
    let hf_cache_dir = expand_home_dir(&config.models.download.hf_cache_dir);
    if hf_cache_dir.exists() && hf_cache_dir != models_dir {
        println!("📁 Searching: {}", hf_cache_dir.display());
        
        // HF cache structure: ~/.cache/huggingface/hub/models--org--name/snapshots/hash/
        let hub_dir = hf_cache_dir.join("hub");
        if hub_dir.exists() {
            println!("  🔍 扫描 HuggingFace hub 目录...");
            let mut scanned = 0;
            if let Ok(entries) = std::fs::read_dir(&hub_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_dir() && path.file_name().and_then(|n| n.to_str()).map(|s| s.starts_with("models--")).unwrap_or(false) {
                        scanned += 1;
                        // Found a model directory, check snapshots
                        if let Ok(snapshot_entries) = std::fs::read_dir(path.join("snapshots")) {
                            for snapshot in snapshot_entries.filter_map(|e| e.ok()) {
                                if snapshot.path().is_dir() {
                                    if let Ok(sub_models) = registry.discover_models(&snapshot.path()).await {
                                        all_models.extend(sub_models);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            println!("  📊 扫描了 {} 个模型目录", scanned);
        }
    }

    if all_models.is_empty() {
        println!("\nℹ️  No models found");
    } else {
        println!("\n{} Found {} model(s):", "✅".bright_green(), all_models.len());
        for model in all_models {
            let status_icon = if model.is_valid { "✅" } else { "⚠️" };
            let arch_str = model
                .architecture
                .map(|a| format!("{:?}", a))
                .unwrap_or_else(|| "Unknown".to_string());

            println!(
                "  {} {} ({})",
                status_icon,
                model.id.cyan(),
                arch_str.yellow()
            );
            println!(
                "    格式: {:?}, 路径: {}",
                model.format,
                model.path.display()
            );
        }
    }

    // Show aliases
    let aliases = registry.list_aliases();
    if !aliases.is_empty() {
        println!("\n{} Available aliases:", "🔗".bright_blue());
        for alias in aliases {
            println!(
                "  {} {} -> {}",
                "🔗",
                alias.name.cyan(),
                alias.target.yellow()
            );
            if let Some(desc) = alias.description {
                println!("    {}", desc.dimmed());
            }
        }
    }

    Ok(())
}

/// Expand ~ in paths to home directory
fn expand_home_dir(path: &str) -> std::path::PathBuf {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(&path[2..]);
        }
    }
    std::path::PathBuf::from(path)
}

async fn show_model_info(model_name: &str, _config: &CliConfig) -> Result<()> {
    println!(
        "{} Model information: {}",
        "ℹ️".bright_blue(),
        model_name.cyan()
    );

    // TODO: Implement model info display
    println!("{} Model info display not yet implemented", "⚠️".yellow());

    Ok(())
}

async fn download_model(model_name: &str, config: &CliConfig) -> Result<()> {
    println!(
        "{} Downloading model: {}",
        "⬇️".bright_blue(),
        model_name.cyan()
    );
    println!(
        "Cache directory: {}",
        config.models.download.hf_cache_dir.cyan()
    );

    let registry = ferrum_models::DefaultModelRegistry::with_defaults();
    let resolved_id = registry.resolve_model_id(model_name);

    println!("Resolving model source for: {}", resolved_id.yellow());

    // Create model source resolver
    let source_config = ferrum_models::ModelSourceConfig {
        cache_dir: Some(expand_home_dir(&config.models.download.hf_cache_dir)),
        hf_token: ferrum_models::ModelSourceConfig::get_hf_token(),
        offline_mode: false,
        max_retries: 3,
        download_timeout: 300,
        use_file_lock: true,
    };

    let resolver = ferrum_models::DefaultModelSourceResolver::new(source_config);

    match resolver.resolve(&resolved_id, None).await {
        Ok(source) => {
            println!("{} Successfully resolved model!", "✅".bright_green());
            println!("  Local path: {}", source.local_path.display());
            println!("  Format: {:?}", source.format);
            println!(
                "  From cache: {}",
                if source.from_cache { "Yes" } else { "No" }
            );
        }
        Err(e) => {
            println!("{} Failed to download model: {}", "❌".bright_red(), e);
            println!("\nTips:");
            println!("  - Check your internet connection");
            println!("  - Verify the model name/ID is correct");
            println!("  - Set HF_TOKEN environment variable for private models");
        }
    }

    Ok(())
}

async fn remove_model(model_name: &str, _config: &CliConfig) -> Result<()> {
    println!(
        "{} Removing model: {}",
        "🗑️".bright_red(),
        model_name.cyan()
    );

    // TODO: Implement model removal
    println!("{} Model removal not yet implemented", "⚠️".yellow());

    Ok(())
}

async fn validate_model(model_name: &str, _config: &CliConfig) -> Result<()> {
    println!(
        "{} Validating model: {}",
        "✅".bright_green(),
        model_name.cyan()
    );

    // TODO: Implement model validation
    println!("{} Model validation not yet implemented", "⚠️".yellow());

    Ok(())
}

async fn show_aliases(_config: &CliConfig) -> Result<()> {
    println!("{} Model aliases", "🔗".bright_blue());

    let registry = ferrum_models::DefaultModelRegistry::with_defaults();
    let aliases = registry.list_aliases();

    if aliases.is_empty() {
        println!("No aliases configured");
    } else {
        println!("\nConfigured aliases:");
        for alias in aliases {
            println!(
                "  {} {} -> {}",
                "🔗",
                alias.name.cyan(),
                alias.target.yellow()
            );
            if let Some(desc) = alias.description {
                println!("    {}", desc.dimmed());
            }
        }
    }

    println!("\nTo add an alias: ferrum models --add-alias <name> --alias-target <model-id>");
    Ok(())
}

async fn add_alias(alias_name: &str, target: &str, _config: &CliConfig) -> Result<()> {
    println!(
        "{} Adding alias: {} -> {}",
        "➕".bright_blue(),
        alias_name.cyan(),
        target.yellow()
    );

    let mut registry = ferrum_models::DefaultModelRegistry::with_defaults();
    let alias = ferrum_models::ModelAlias {
        name: alias_name.to_string(),
        target: target.to_string(),
        description: None,
    };

    match registry.add_alias(alias) {
        Ok(_) => {
            println!("{} Alias added successfully!", "✅".bright_green());
            println!("Note: This alias is only active for this session. For persistent aliases, configure them in your settings.");
        }
        Err(e) => {
            println!("{} Failed to add alias: {}", "❌".bright_red(), e);
        }
    }

    Ok(())
}
