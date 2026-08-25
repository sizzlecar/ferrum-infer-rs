//! List command - Show downloaded models

use crate::config::CliConfig;
use clap::Args;
use colored::*;
use ferrum_types::Result;
use std::fs;
use std::path::PathBuf;

#[derive(Args)]
pub struct ListCommand {}

pub async fn execute(_cmd: ListCommand, config: CliConfig) -> Result<()> {
    let cache_dir = get_hf_cache_dir(&config);
    let hub_dir = cache_dir.join("hub");

    if !hub_dir.exists() {
        println!("{}", "No models downloaded yet.".dimmed());
        println!();
        println!("Run {} to download a model.", "ferrum pull <model>".cyan());
        return Ok(());
    }

    let mut models: Vec<ModelInfo> = Vec::new();

    // Scan hub directory for models
    if let Ok(entries) = fs::read_dir(&hub_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("models--") {
                if let Some(info) = get_model_info(&entry.path()) {
                    models.push(info);
                }
            }
        }
    }

    if models.is_empty() {
        println!("{}", "No models downloaded yet.".dimmed());
        println!();
        println!("Run {} to download a model.", "ferrum pull <model>".cyan());
        return Ok(());
    }

    // Sort: complete models first, then by name
    models.sort_by(|a, b| match (a.is_complete, b.is_complete) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.cmp(&b.name),
    });

    // Print header
    println!(
        "{:<40} {:<12} {:<10} {:<16}",
        "NAME".bold(),
        "SIZE".bold(),
        "STATUS".bold(),
        "MODIFIED".bold()
    );

    // Print models
    for model in models {
        let status = if model.is_complete {
            "ready".green().to_string()
        } else {
            "incomplete".yellow().to_string()
        };

        let name_display = if model.is_complete {
            model.name.normal().to_string()
        } else {
            model.name.dimmed().to_string()
        };

        println!(
            "{:<40} {:<12} {:<10} {:<16}",
            name_display,
            format_size(model.size),
            status,
            model.modified
        );
    }

    Ok(())
}

struct ModelInfo {
    name: String,
    size: u64,
    modified: String,
    is_complete: bool,
}

fn get_model_info(model_dir: &PathBuf) -> Option<ModelInfo> {
    // Parse name from directory: models--Org--ModelName -> Org/ModelName
    let dir_name = model_dir.file_name()?.to_string_lossy().to_string();
    let name = dir_name.strip_prefix("models--")?.replace("--", "/");

    // Get size of blobs
    let blobs_dir = model_dir.join("blobs");
    let size = if blobs_dir.exists() {
        get_dir_size(&blobs_dir)
    } else {
        0
    };

    // Check if model files exist (complete model)
    let snapshots_dir = model_dir.join("snapshots");
    let is_complete = check_model_complete(&snapshots_dir);

    // Get modification time
    let modified = if let Ok(metadata) = fs::metadata(model_dir) {
        if let Ok(time) = metadata.modified() {
            let datetime: chrono::DateTime<chrono::Local> = time.into();
            datetime.format("%Y-%m-%d %H:%M").to_string()
        } else {
            "unknown".to_string()
        }
    } else {
        "unknown".to_string()
    };

    Some(ModelInfo {
        name,
        size,
        modified,
        is_complete,
    })
}

/// Check if model has actual weight files (not just tokenizer)
fn check_model_complete(snapshots_dir: &PathBuf) -> bool {
    if !snapshots_dir.exists() {
        return false;
    }

    // Check each snapshot directory
    if let Ok(entries) = fs::read_dir(snapshots_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // A snapshot may contain a single file, sharded weights plus
                // an index, or a curated GGUF filename.
                let has_weights = fs::read_dir(&path)
                    .is_ok_and(|files| files.flatten().any(|file| is_weight_file(&file.path())));
                if has_weights {
                    return true;
                }
            }
        }
    }

    false
}

fn is_weight_file(path: &std::path::Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".safetensors")
        || lower.ends_with(".safetensors.index.json")
        || lower.ends_with(".gguf")
        || (lower.starts_with("pytorch_model")
            && (lower.ends_with(".bin") || lower.ends_with(".bin.index.json")))
}

fn get_dir_size(path: &PathBuf) -> u64 {
    let mut size = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Ok(metadata) = fs::metadata(&path) {
                    size += metadata.len();
                }
            } else if path.is_dir() {
                size += get_dir_size(&path);
            }
        }
    }
    size
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

fn get_hf_cache_dir(config: &CliConfig) -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home);
    }
    let configured = shellexpand::tilde(&config.models.download.hf_cache_dir).to_string();
    PathBuf::from(configured)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_snapshot(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("ferrum-list-{name}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(root.join("snapshots/revision")).unwrap();
        root
    }

    #[test]
    fn gguf_snapshot_is_complete() {
        let root = temporary_snapshot("gguf");
        fs::write(
            root.join("snapshots/revision/Qwen3.5-4B-Q4_K_M.gguf"),
            b"weight",
        )
        .unwrap();

        assert!(check_model_complete(&root.join("snapshots")));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tokenizer_only_snapshot_is_incomplete() {
        let root = temporary_snapshot("tokenizer");
        fs::write(root.join("snapshots/revision/tokenizer.json"), b"{}").unwrap();

        assert!(!check_model_complete(&root.join("snapshots")));
        fs::remove_dir_all(root).unwrap();
    }
}
