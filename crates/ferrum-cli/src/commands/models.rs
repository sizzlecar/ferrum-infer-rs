//! Models command implementation - Enhanced with vLLM-inspired model maintenance

use crate::{config::CliConfig, output::OutputFormat};
use clap::Args;
use colored::*;
use ferrum_core::Result;
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
            ferrum_core::Error::invalid_request(
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
    println!("Model directory: {}", config.models.model_dir.cyan());

    // Use enhanced model registry for discovery
    let mut registry = ferrum_models::DefaultModelRegistry::with_defaults();
    let models_dir = std::path::PathBuf::from(&config.models.model_dir);
    
    match registry.discover_models(&models_dir).await {
        Ok(models) => {
            if models.is_empty() {
                println!("{} No models found in directory", "ℹ️".bright_blue());
            } else {
                println!("\n{} Found {} models:", "✅".bright_green(), models.len());
                for model in models {
                    let status_icon = if model.is_valid { "✅" } else { "⚠️" };
                    let arch_str = model.architecture
                        .map(|a| format!("{:?}", a))
                        .unwrap_or_else(|| "Unknown".to_string());
                    
                    println!("  {} {} ({})", status_icon, model.id.cyan(), arch_str.yellow());
                    println!("    Format: {:?}, Path: {}", model.format, model.path.display());
                }
            }
        }
        Err(e) => {
            println!("{} Failed to discover models: {}", "❌".bright_red(), e);
        }
    }
    
    // Show aliases
    let aliases = registry.list_aliases();
    if !aliases.is_empty() {
        println!("\n{} Available aliases:", "🔗".bright_blue());
        for alias in aliases {
            println!("  {} {} -> {}", "🔗", alias.name.cyan(), alias.target.yellow());
            if let Some(desc) = alias.description {
                println!("    {}", desc.dimmed());
            }
        }
    }

    Ok(())
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
        cache_dir: Some(std::path::PathBuf::from(&config.models.download.hf_cache_dir)),
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
            println!("  From cache: {}", if source.from_cache { "Yes" } else { "No" });
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
            println!("  {} {} -> {}", "🔗", alias.name.cyan(), alias.target.yellow());
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
