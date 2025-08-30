//! Configuration command implementation

use clap::Args;
use ferrum_core::Result;
use crate::{config::CliConfig, output::OutputFormat};
use colored::*;

#[derive(Args)]
pub struct ConfigCommand {
    /// Show current configuration
    #[arg(short, long)]
    pub show: bool,
    
    /// Validate configuration
    #[arg(short, long)]
    pub validate: bool,
    
    /// Generate default configuration
    #[arg(long)]
    pub generate: bool,
    
    /// Configuration key to get/set
    #[arg(long)]
    pub key: Option<String>,
    
    /// Value to set (used with --key)
    #[arg(long)]
    pub value: Option<String>,
    
    /// Output file for generated config
    #[arg(short, long)]
    pub output: Option<String>,
}

pub async fn execute(cmd: ConfigCommand, config: CliConfig, format: OutputFormat) -> Result<()> {
    if cmd.show {
        return show_config(&config, &format).await;
    }
    
    if cmd.validate {
        return validate_config(&config).await;
    }
    
    if cmd.generate {
        return generate_config(cmd.output.as_deref()).await;
    }
    
    if let Some(key) = cmd.key {
        if let Some(value) = cmd.value {
            return set_config_value(&key, &value, &config).await;
        } else {
            return get_config_value(&key, &config).await;
        }
    }
    
    // Default: show config
    show_config(&config, &format).await
}

async fn show_config(config: &CliConfig, format: &OutputFormat) -> Result<()> {
    println!("{} Current configuration", "⚙️".bright_blue());
    
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(config)
                .map_err(|e| ferrum_core::Error::serialization(format!("Failed to serialize config: {}", e)))?;
            println!("{}", json);
        }
        OutputFormat::Yaml => {
            let yaml = serde_yaml::to_string(config)
                .map_err(|e| ferrum_core::Error::serialization(format!("Failed to serialize config: {}", e)))?;
            println!("{}", yaml);
        }
        _ => {
            println!("Server:");
            println!("  Host: {}", config.server.host.cyan());
            println!("  Port: {}", config.server.port.to_string().cyan());
            println!("  Log level: {}", config.server.log_level.cyan());
            
            println!("Models:");
            println!("  Directory: {}", config.models.model_dir.cyan());
            println!("  Cache directory: {}", config.models.cache_dir.cyan());
            if let Some(default) = &config.models.default_model {
                println!("  Default: {}", default.cyan());
            }
            
            println!("Benchmark:");
            println!("  Requests: {}", config.benchmark.num_requests.to_string().cyan());
            println!("  Concurrency: {}", config.benchmark.concurrency.to_string().cyan());
            
            println!("Client:");
            println!("  Base URL: {}", config.client.base_url.cyan());
        }
    }
    
    Ok(())
}

async fn validate_config(config: &CliConfig) -> Result<()> {
    println!("{} Validating configuration", "🔍".bright_blue());
    
    match config.validate() {
        Ok(_) => {
            println!("{} Configuration is valid", "✅".green());
        }
        Err(e) => {
            println!("{} Configuration validation failed: {}", "❌".red(), e);
            return Err(e);
        }
    }
    
    Ok(())
}

async fn generate_config(output_path: Option<&str>) -> Result<()> {
    let default_config = CliConfig::default();
    let output_path = output_path.unwrap_or("ferrum.toml");
    
    println!("{} Generating default configuration", "📝".bright_blue());
    println!("Output: {}", output_path.cyan());
    
    default_config.save(output_path).await?;
    
    println!("{} Configuration generated successfully", "✅".green());
    
    Ok(())
}

async fn get_config_value(key: &str, config: &CliConfig) -> Result<()> {
    println!("{} Getting configuration value: {}", "🔍".bright_blue(), key.cyan());
    
    // TODO: Implement config key lookup
    println!("{} Config key lookup not yet implemented", "⚠️".yellow());
    println!("Key: {}", key);
    
    Ok(())
}

async fn set_config_value(key: &str, value: &str, _config: &CliConfig) -> Result<()> {
    println!("{} Setting configuration value", "✏️".bright_blue());
    println!("Key: {}", key.cyan());
    println!("Value: {}", value.cyan());
    
    // TODO: Implement config value setting
    println!("{} Config value setting not yet implemented", "⚠️".yellow());
    
    Ok(())
}
