//! Utility functions and helpers for the LLM inference engine
//!
//! This module contains common utility functions, helpers, and convenience
//! methods used throughout the inference engine.

use crate::error::{EngineError, Result};
use std::path::Path;
use tracing::{info, warn};

/// Initialize logging based on configuration
pub fn init_logging(level: &str, format: &str) -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    let level = match level.to_lowercase().as_str() {
        "trace" => tracing::Level::TRACE,
        "debug" => tracing::Level::DEBUG,
        "info" => tracing::Level::INFO,
        "warn" => tracing::Level::WARN,
        "error" => tracing::Level::ERROR,
        _ => tracing::Level::INFO,
    };

    let env_filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(level.into());

    match format.to_lowercase().as_str() {
        "json" => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(tracing_subscriber::fmt::layer().json())
                .init();
        }
        "pretty" | _ => {
            tracing_subscriber::registry()
                .with(env_filter)
                .with(tracing_subscriber::fmt::layer().pretty())
                .init();
        }
    }

    info!("Logging initialized with level: {} and format: {}", level, format);
    Ok(())
}

/// Check if a file exists and is readable
pub fn check_file_exists<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(EngineError::config(format!(
            "File does not exist: {}",
            path.display()
        )));
    }

    if !path.is_file() {
        return Err(EngineError::config(format!(
            "Path is not a file: {}",
            path.display()
        )));
    }

    // Try to read the file to check permissions
    if let Err(e) = std::fs::File::open(path) {
        return Err(EngineError::config(format!(
            "Cannot read file {}: {}",
            path.display(),
            e
        )));
    }

    Ok(())
}

/// Create directory if it doesn't exist
pub fn ensure_directory<P: AsRef<Path>>(path: P) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        std::fs::create_dir_all(path)?;
        info!("Created directory: {}", path.display());
    } else if !path.is_dir() {
        return Err(EngineError::config(format!(
            "Path exists but is not a directory: {}",
            path.display()
        )));
    }
    Ok(())
}

/// Format file size in human-readable format
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Format duration in human-readable format
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let millis = duration.subsec_millis();

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else if seconds > 0 {
        format!("{}.{:03}s", seconds, millis)
    } else {
        format!("{}ms", millis)
    }
}

/// Get system memory information
pub fn get_memory_info() -> Result<MemoryInfo> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo")?;
        let mut total = None;
        let mut available = None;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(value) = line.split_whitespace().nth(1) {
                    total = value.parse::<u64>().ok().map(|v| v * 1024); // Convert from kB to bytes
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(value) = line.split_whitespace().nth(1) {
                    available = value.parse::<u64>().ok().map(|v| v * 1024); // Convert from kB to bytes
                }
            }
        }

        match (total, available) {
            (Some(total), Some(available)) => Ok(MemoryInfo {
                total_bytes: total,
                available_bytes: available,
                used_bytes: total - available,
            }),
            _ => Err(EngineError::internal("Failed to parse memory information")),
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        warn!("Memory information not available on this platform");
        Ok(MemoryInfo {
            total_bytes: 0,
            available_bytes: 0,
            used_bytes: 0,
        })
    }
}

/// System memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_bytes: u64,
}

impl MemoryInfo {
    /// Get memory usage as a percentage
    pub fn usage_percentage(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
        }
    }

    /// Format memory info as a string
    pub fn format(&self) -> String {
        format!(
            "Memory: {} used / {} total ({:.1}%)",
            format_bytes(self.used_bytes as usize),
            format_bytes(self.total_bytes as usize),
            self.usage_percentage()
        )
    }
}

/// Validate model name format
pub fn validate_model_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(EngineError::invalid_request("Model name cannot be empty"));
    }

    // Check for invalid characters
    if name.contains(|c: char| c.is_control() || "\"'<>&".contains(c)) {
        return Err(EngineError::invalid_request(
            "Model name contains invalid characters",
        ));
    }

    Ok(())
}

/// Generate a unique request ID
pub fn generate_request_id() -> String {
    format!("req_{}", uuid::Uuid::new_v4())
}

/// Sanitize text for logging (remove sensitive information)
pub fn sanitize_for_logging(text: &str, max_length: usize) -> String {
    let truncated = if text.len() > max_length {
        format!("{}...", &text[..max_length])
    } else {
        text.to_string()
    };

    // Remove potential sensitive patterns
    truncated
        .replace("password", "***")
        .replace("token", "***")
        .replace("key", "***")
}

/// Retry a function with exponential backoff
pub async fn retry_with_backoff<F, Fut, T, E>(
    mut operation: F,
    max_retries: usize,
    initial_delay: std::time::Duration,
) -> std::result::Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, E>>,
{
    let mut delay = initial_delay;
    
    for attempt in 0..=max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if attempt == max_retries {
                    return Err(e);
                }
                
                warn!("Operation failed, retrying in {:?} (attempt {}/{})", delay, attempt + 1, max_retries);
                tokio::time::sleep(delay).await;
                delay *= 2; // Exponential backoff
            }
        }
    }
    
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512.00 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        use std::time::Duration;

        assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(Duration::from_secs(1)), "1.000s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format_duration(Duration::from_secs(3665)), "1h 1m 5s");
    }

    #[test]
    fn test_validate_model_name() {
        assert!(validate_model_name("gpt-3.5-turbo").is_ok());
        assert!(validate_model_name("my_model_v1").is_ok());
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name("model\"with\"quotes").is_err());
        assert!(validate_model_name("model<with>brackets").is_err());
    }

    #[test]
    fn test_generate_request_id() {
        let id1 = generate_request_id();
        let id2 = generate_request_id();
        
        assert!(id1.starts_with("req_"));
        assert!(id2.starts_with("req_"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_sanitize_for_logging() {
        let text = "This is a password: secret123 and a token: abc123";
        let sanitized = sanitize_for_logging(text, 100);
        
        assert!(sanitized.contains("***"));
        assert!(!sanitized.contains("secret123"));
        assert!(!sanitized.contains("abc123"));
    }

    #[test]
    fn test_memory_info_percentage() {
        let memory = MemoryInfo {
            total_bytes: 1000,
            available_bytes: 300,
            used_bytes: 700,
        };
        
        assert_eq!(memory.usage_percentage(), 70.0);
    }
}