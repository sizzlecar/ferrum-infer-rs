//! Output formatting utilities

use colored::*;
use ferrum_core::Result;
use serde::Serialize;

/// Output format options
#[derive(Debug, Clone, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    Pretty,
    Json,
    Yaml,
    Table,
}

/// Output formatter trait
pub trait OutputFormatter {
    fn format<T: Serialize>(&self, data: &T, format: &OutputFormat) -> Result<String>;
}

/// Default output formatter
pub struct DefaultFormatter;

impl OutputFormatter for DefaultFormatter {
    fn format<T: Serialize>(&self, data: &T, format: &OutputFormat) -> Result<String> {
        match format {
            OutputFormat::Json => serde_json::to_string_pretty(data).map_err(|e| {
                ferrum_core::Error::serialization(format!("JSON serialization failed: {}", e))
            }),
            OutputFormat::Yaml => serde_yaml::to_string(data).map_err(|e| {
                ferrum_core::Error::serialization(format!("YAML serialization failed: {}", e))
            }),
            OutputFormat::Pretty | OutputFormat::Table => {
                // For pretty/table format, just use JSON as fallback
                serde_json::to_string_pretty(data).map_err(|e| {
                    ferrum_core::Error::serialization(format!("Pretty formatting failed: {}", e))
                })
            }
        }
    }
}

/// Print formatted output
pub fn print_output<T: Serialize>(data: &T, format: &OutputFormat) -> Result<()> {
    let formatter = DefaultFormatter;
    let output = formatter.format(data, format)?;
    println!("{}", output);
    Ok(())
}

/// Print error with formatting
pub fn print_error(error: &ferrum_core::Error) {
    eprintln!("{} {}", "Error:".red().bold(), error);
}

/// Print warning with formatting
pub fn print_warning(message: &str) {
    eprintln!("{} {}", "Warning:".yellow().bold(), message);
}

/// Print success message
pub fn print_success(message: &str) {
    println!("{} {}", "✅".green(), message.green());
}

/// Print info message
pub fn print_info(message: &str) {
    println!("{} {}", "ℹ️".blue(), message);
}

/// Create a table header
pub fn table_header(columns: &[&str]) -> String {
    let header = columns.join(" | ");
    let separator = "─".repeat(header.len());
    format!("{}\n{}", header.bold(), separator)
}

/// Create a table row
pub fn table_row(values: &[&str]) -> String {
    values.join(" | ")
}
