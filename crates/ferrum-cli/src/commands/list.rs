//! List command - Show downloaded models

use crate::config::CliConfig;
use clap::Args;
use colored::*;
use ferrum_types::Result;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

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

fn get_model_info(model_dir: &Path) -> Option<ModelInfo> {
    // Parse name from directory: models--Org--ModelName -> Org/ModelName
    let dir_name = model_dir.file_name()?.to_string_lossy().to_string();
    let encoded = dir_name.strip_prefix("models--")?;
    let parts: Vec<_> = encoded.split("--").collect();
    if !(1..=2).contains(&parts.len())
        || parts.iter().any(|part| {
            part.is_empty()
                || part.starts_with(['.', '-'])
                || part.ends_with(['.', '-'])
                || part.contains("..")
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
        })
    {
        return None;
    }
    let name = parts.join("/");

    // HF may store bytes directly in snapshots or through links into blobs.
    // Count each target once across both layouts, including partial downloads.
    let size = cache_size(model_dir);

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
    snapshots_dir
        .parent()
        .is_some_and(crate::source_resolver::cache::model_cache_ready)
}

fn cache_size(model_dir: &Path) -> u64 {
    let mut size = 0u64;
    let mut seen = HashSet::new();
    let mut directories = vec![model_dir.join("blobs"), model_dir.join("snapshots")];
    while let Some(directory) = directories.pop() {
        if directory.is_symlink() {
            continue;
        }
        if let Ok(entries) = fs::read_dir(directory) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Ok(metadata) = fs::metadata(&path) else {
                    continue;
                };
                if metadata.is_dir() {
                    if !path.is_symlink() {
                        directories.push(path);
                    }
                } else if metadata.is_file() {
                    if let Ok(canonical) = path.canonicalize() {
                        if seen.insert(canonical) {
                            size = size.saturating_add(metadata.len());
                        }
                    }
                }
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

    #[test]
    fn model_size_includes_direct_snapshot_files_and_skips_malformed_names() {
        let cache = tempfile::tempdir().unwrap();
        let model = cache.path().join("models--fixture--model");
        fs::create_dir_all(model.join("snapshots/revision")).unwrap();
        fs::write(model.join("snapshots/revision/model.gguf"), b"weights").unwrap();
        let info = get_model_info(&model).unwrap();
        assert_eq!(info.name, "fixture/model");
        assert_eq!(info.size, 7);
        assert!(info.is_complete);
        for name in [
            "models--qwen3:4b",
            "models--owner--repo--snapshot",
            "models--..",
            "models--",
        ] {
            assert!(get_model_info(&cache.path().join(name)).is_none());
        }
        assert_eq!(
            get_model_info(&cache.path().join("models--gpt2"))
                .unwrap()
                .name,
            "gpt2"
        );
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_links_share_blob_size_without_following_directory_cycles() {
        use std::os::unix::fs::symlink;
        let cache = tempfile::tempdir().unwrap();
        let model = cache.path();
        fs::create_dir_all(model.join("blobs")).unwrap();
        fs::create_dir_all(model.join("snapshots/revision")).unwrap();
        fs::write(model.join("blobs/hash"), b"weights").unwrap();
        symlink(
            "../../blobs/hash",
            model.join("snapshots/revision/model.gguf"),
        )
        .unwrap();
        symlink(
            "../../blobs/hash",
            model.join("snapshots/revision/another.gguf"),
        )
        .unwrap();
        symlink("../..", model.join("snapshots/revision/cycle")).unwrap();
        symlink("missing", model.join("snapshots/revision/missing.gguf")).unwrap();
        assert_eq!(cache_size(model), 7);
    }

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
