//! CLI utility functions

use colored::*;
use ferrum_types::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Setup logging based on verbosity level
pub fn setup_logging(verbose: bool, quiet: bool) -> Result<()> {
    let log_level = if quiet {
        tracing::Level::ERROR
    } else if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::WARN // 改为WARN减少噪音
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level.to_string())),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    Ok(())
}

/// Print application banner
pub fn print_banner() {
    println!();
    println!("{}", "  ______                            ".bright_red());
    println!("{}", " |  ____|                           ".bright_red());
    println!("{}", " | |__ ___ _ __ _ __ _   _ _ __ ___   ".bright_red());
    println!(
        "{}",
        " |  __/ _ \\ '__| '__| | | | '_ ` _ \\  ".bright_red()
    );
    println!("{}", " | | |  __/ |  | |  | |_| | | | | | ".bright_red());
    println!(
        "{}",
        " |_|  \\___|_|  |_|   \\__,_|_| |_| |_| ".bright_red()
    );
    println!();
    println!(
        "   {}",
        "🦀 Rust LLM Inference Framework".bright_cyan().bold()
    );
    println!(
        "   {}",
        format!("Version: {}", env!("CARGO_PKG_VERSION")).bright_white()
    );
    println!();
}

/// Format duration in human-readable format
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_ms = duration.as_millis();

    if total_ms < 1000 {
        format!("{}ms", total_ms)
    } else if total_ms < 60_000 {
        format!("{:.2}s", total_ms as f64 / 1000.0)
    } else {
        let minutes = total_ms / 60_000;
        let seconds = (total_ms % 60_000) as f64 / 1000.0;
        format!("{}m {:.1}s", minutes, seconds)
    }
}

/// Format bytes in human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Format rate (requests per second)
pub fn format_rate(requests: f64, duration: std::time::Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds == 0.0 {
        return "∞ req/s".to_string();
    }

    let rate = requests / seconds;
    if rate >= 1000.0 {
        format!("{:.1}k req/s", rate / 1000.0)
    } else {
        format!("{:.1} req/s", rate)
    }
}

/// Show progress bar
pub fn create_progress_bar(total: u64, message: &str) -> indicatif::ProgressBar {
    let pb = indicatif::ProgressBar::new(total);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_message(message.to_string());
    pb
}

/// Confirm action with user
pub fn confirm_action(message: &str) -> Result<bool> {
    let result = dialoguer::Confirm::new()
        .with_prompt(message)
        .default(false)
        .interact()
        .map_err(|e| {
            ferrum_types::FerrumError::io_str(format!("Failed to get user confirmation: {}", e))
        })?;

    Ok(result)
}

/// Select from options
pub fn select_option<T: std::fmt::Display>(prompt: &str, options: &[T]) -> Result<usize> {
    let selection = dialoguer::Select::new()
        .with_prompt(prompt)
        .items(options)
        .interact()
        .map_err(|e| {
            ferrum_types::FerrumError::io_str(format!("Failed to get user selection: {}", e))
        })?;

    Ok(selection)
}

/// Get text input from user
pub fn get_input(prompt: &str, default: Option<&str>) -> Result<String> {
    let mut input = dialoguer::Input::new().with_prompt(prompt);

    if let Some(default_value) = default {
        input = input.default(default_value.to_string());
    }

    let result = input.interact_text().map_err(|e| {
        ferrum_types::FerrumError::io_str(format!("Failed to get user input: {}", e))
    })?;

    Ok(result)
}
