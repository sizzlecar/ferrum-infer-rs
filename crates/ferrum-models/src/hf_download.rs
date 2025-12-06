//! HuggingFace model downloader with proxy and resume support
//!
//! Features:
//! - SOCKS5/HTTP proxy support via environment variables
//! - Resumable downloads (断点续传)
//! - Progress bar with speed display
//! - HuggingFace token authentication

use ferrum_types::{FerrumError, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs::{self, File, OpenOptions};
use tokio::io::AsyncWriteExt;

/// HuggingFace API base URL
const HF_API_URL: &str = "https://huggingface.co";

/// Files required for a complete model
const REQUIRED_FILES: &[&str] = &["config.json"];
const MODEL_FILES: &[&str] = &[
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
];
const TOKENIZER_FILES: &[&str] = &["tokenizer.json", "tokenizer.model", "tokenizer_config.json"];

/// HuggingFace file metadata from API
#[derive(Debug, Deserialize)]
struct HfFileInfo {
    path: String,
    size: Option<u64>,
    #[serde(rename = "type")]
    file_type: Option<String>,
}

/// HuggingFace downloader with resume support
#[derive(Clone)]
pub struct HfDownloader {
    client: Client,
    cache_dir: PathBuf,
    token: Option<String>,
}

impl HfDownloader {
    /// Create a new downloader with proxy support
    pub fn new(cache_dir: PathBuf, token: Option<String>) -> Result<Self> {
        let mut builder = Client::builder()
            .timeout(std::time::Duration::from_secs(3600))
            .connect_timeout(std::time::Duration::from_secs(30));

        // Check for proxy environment variables
        if let Ok(proxy_url) = std::env::var("HTTPS_PROXY")
            .or_else(|_| std::env::var("https_proxy"))
            .or_else(|_| std::env::var("ALL_PROXY"))
            .or_else(|_| std::env::var("all_proxy"))
        {
            if !proxy_url.is_empty() {
                eprintln!("🌐 Using proxy: {}", proxy_url);
                let proxy = reqwest::Proxy::all(&proxy_url)
                    .map_err(|e| FerrumError::config(format!("Invalid proxy URL: {}", e)))?;
                builder = builder.proxy(proxy);
            }
        }

        let client = builder
            .build()
            .map_err(|e| FerrumError::config(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            cache_dir,
            token,
        })
    }

    /// Download a model from HuggingFace
    pub async fn download(&self, model_id: &str, revision: Option<&str>) -> Result<PathBuf> {
        let revision = revision.unwrap_or("main");

        // Create cache directory structure: hub/models--org--name/snapshots/revision/
        let model_cache_name = format!("models--{}", model_id.replace('/', "--"));
        let model_dir = self.cache_dir.join("hub").join(&model_cache_name);
        let snapshots_dir = model_dir.join("snapshots");
        let blobs_dir = model_dir.join("blobs");
        let refs_dir = model_dir.join("refs");

        fs::create_dir_all(&snapshots_dir).await?;
        fs::create_dir_all(&blobs_dir).await?;
        fs::create_dir_all(&refs_dir).await?;

        // Get file list from HuggingFace API
        let files = self.list_files(model_id, revision).await?;

        // Determine which files to download (only files, not directories)
        let files_to_download: Vec<_> = files
            .iter()
            .filter(|f| {
                if f.file_type.as_deref() == Some("directory") {
                    return false;
                }
                REQUIRED_FILES.contains(&f.path.as_str())
                    || MODEL_FILES.iter().any(|m| f.path.starts_with(m) || f.path == *m)
                    || TOKENIZER_FILES.contains(&f.path.as_str())
                    || f.path == "generation_config.json"
                    || f.path == "special_tokens_map.json"
            })
            .collect();

        if files_to_download.is_empty() {
            return Err(FerrumError::model("No model files found in repository"));
        }

        // Get commit SHA for snapshot directory
        let commit_sha = self.get_commit_sha(model_id, revision).await?;
        let snapshot_dir = snapshots_dir.join(&commit_sha);
        fs::create_dir_all(&snapshot_dir).await?;

        // Calculate total size
        let total_size: u64 = files_to_download.iter().filter_map(|f| f.size).sum();
        let file_count = files_to_download.len();
        println!(
            "📦 Downloading {} files ({:.2} GB)",
            file_count,
            total_size as f64 / 1_073_741_824.0
        );

        // Use concurrent downloads for multiple files (up to 3 concurrent)
        let concurrency = std::cmp::min(3, file_count);
        
        if concurrency > 1 && file_count > 1 {
            // Concurrent download with MultiProgress
            let mp = Arc::new(MultiProgress::new());
            let self_arc = Arc::new(self.clone());
            
            let mut handles = Vec::new();
            for file_info in files_to_download {
                let downloader = self_arc.clone();
                let mp = mp.clone();
                let model_id = model_id.to_string();
                let revision = revision.to_string();
                let filename = file_info.path.clone();
                let size = file_info.size.unwrap_or(0);
                let blobs = blobs_dir.clone();
                let snapshot = snapshot_dir.clone();
                
                let handle = tokio::spawn(async move {
                    downloader
                        .download_file_concurrent(&model_id, &revision, &filename, size, &blobs, &snapshot, Some(&mp))
                        .await
                });
                handles.push(handle);
            }
            
            // Wait for all downloads
            for handle in handles {
                handle.await.map_err(|e| FerrumError::model(format!("Task error: {}", e)))??;
            }
        } else {
            // Sequential download for single file
            for file_info in &files_to_download {
                self.download_file_concurrent(
                    model_id,
                    revision,
                    &file_info.path,
                    file_info.size.unwrap_or(0),
                    &blobs_dir,
                    &snapshot_dir,
                    None,
                )
                .await?;
            }
        }

        // Write refs/main to point to this snapshot
        let ref_file = refs_dir.join(revision);
        fs::write(&ref_file, &commit_sha).await?;

        println!();
        println!("✅ Download complete: {}", snapshot_dir.display());
        Ok(snapshot_dir)
    }

    /// List files in a HuggingFace repository
    async fn list_files(&self, model_id: &str, revision: &str) -> Result<Vec<HfFileInfo>> {
        let url = format!("{}/api/models/{}/tree/{}", HF_API_URL, model_id, revision);

        let mut request = self.client.get(&url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request
            .send()
            .await
            .map_err(|e| FerrumError::model(format!("Failed to list files: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(FerrumError::model(format!(
                "API error ({}): {}",
                status, text
            )));
        }

        let files: Vec<HfFileInfo> = response
            .json()
            .await
            .map_err(|e| FerrumError::model(format!("Failed to parse file list: {}", e)))?;

        Ok(files)
    }

    /// Get the commit SHA for a revision
    async fn get_commit_sha(&self, model_id: &str, revision: &str) -> Result<String> {
        let url = format!("{}/api/models/{}/revision/{}", HF_API_URL, model_id, revision);

        let mut request = self.client.get(&url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let response = request
            .send()
            .await
            .map_err(|e| FerrumError::model(format!("Failed to get revision info: {}", e)))?;

        if !response.status().is_success() {
            // Fallback: use revision as SHA if it looks like one
            if revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit()) {
                return Ok(revision.to_string());
            }
            return Err(FerrumError::model(format!(
                "Failed to get commit SHA for revision '{}'",
                revision
            )));
        }

        #[derive(Deserialize)]
        struct RevisionInfo {
            sha: String,
        }

        let info: RevisionInfo = response
            .json()
            .await
            .map_err(|e| FerrumError::model(format!("Failed to parse revision info: {}", e)))?;

        Ok(info.sha)
    }

    /// Download a single file with resume support (concurrent-safe)
    async fn download_file_concurrent(
        &self,
        model_id: &str,
        revision: &str,
        filename: &str,
        expected_size: u64,
        blobs_dir: &Path,
        snapshot_dir: &Path,
        mp: Option<&MultiProgress>,
    ) -> Result<()> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            HF_API_URL, model_id, revision, filename
        );

        // Use a short display name for progress
        let display_name = if filename.len() > 30 {
            format!("...{}", &filename[filename.len()-27..])
        } else {
            filename.to_string()
        };

        // First, do a HEAD request to get file info
        let mut head_req = self.client.head(&url);
        if let Some(token) = &self.token {
            head_req = head_req.header("Authorization", format!("Bearer {}", token));
        }

        let head_resp = head_req
            .send()
            .await
            .map_err(|e| FerrumError::model(format!("Failed to get file info for {}: {}", filename, e)))?;

        if !head_resp.status().is_success() {
            return Err(FerrumError::model(format!(
                "Failed to access {} ({})",
                filename,
                head_resp.status()
            )));
        }

        // Get content length - prefer HEAD response, fallback to API size
        let head_size = head_resp.content_length().unwrap_or(0);
        let total_size = if head_size > 0 { head_size } else { expected_size };
        
        let etag = head_resp
            .headers()
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.trim_matches('"').replace('/', "_"))
            .unwrap_or_else(|| format!("{:016x}", simple_hash(filename)));

        let blob_path = blobs_dir.join(&etag);
        let incomplete_path = blobs_dir.join(format!("{}.incomplete", etag));
        let snapshot_file = snapshot_dir.join(filename);

        // Check if already complete
        if blob_path.exists() {
            if let Ok(meta) = fs::metadata(&blob_path).await {
                if total_size == 0 || meta.len() == total_size || (total_size == 0 && meta.len() > 0) {
                    create_symlink(&blob_path, &snapshot_file).await?;
                    println!("  ✓ {} (cached)", display_name);
                    return Ok(());
                }
            }
        }

        // Check for incomplete download
        let resume_from = if incomplete_path.exists() {
            fs::metadata(&incomplete_path)
                .await
                .map(|m| m.len())
                .unwrap_or(0)
        } else {
            0
        };

        // Create progress bar - use spinner mode if size unknown
        let pb = if total_size > 0 {
            let pb = if let Some(mp) = mp {
                mp.add(ProgressBar::new(total_size))
            } else {
                ProgressBar::new(total_size)
            };
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("  {spinner:.green} {msg:<30} [{bar:30.cyan/blue}] {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>12} ETA {eta}")
                    .unwrap()
                    .progress_chars("━╸─"),
            );
            pb
        } else {
            let pb = if let Some(mp) = mp {
                mp.add(ProgressBar::new_spinner())
            } else {
                ProgressBar::new_spinner()
            };
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("  {spinner:.green} {msg:<30} {bytes:>10} {bytes_per_sec:>12}")
                    .unwrap(),
            );
            pb
        };
        pb.set_message(display_name.clone());

        // If resuming, set initial position
        if resume_from > 0 && (total_size == 0 || resume_from < total_size) {
            pb.set_position(resume_from);
            pb.set_message(format!("{} (续传)", display_name));
        }

        // Build download request with optional Range header
        let mut request = self.client.get(&url);
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        let (mut file, start_pos) = if resume_from > 0 && resume_from < total_size {
            // Resume download
            request = request.header("Range", format!("bytes={}-", resume_from));
            let file = OpenOptions::new()
                .write(true)
                .append(true)
                .open(&incomplete_path)
                .await?;
            (file, resume_from)
        } else {
            // Fresh download
            let file = File::create(&incomplete_path).await?;
            (file, 0u64)
        };

        // Send request
        let response = request
            .send()
            .await
            .map_err(|e| FerrumError::model(format!("Failed to download {}: {}", filename, e)))?;

        let status = response.status();
        if !status.is_success() && status.as_u16() != 206 {
            // 206 = Partial Content (for range requests)
            return Err(FerrumError::model(format!(
                "Failed to download {} ({})",
                filename, status
            )));
        }

        // Update total size from GET response if we didn't have it
        let content_length = response.content_length().unwrap_or(0);
        let actual_total = if start_pos > 0 {
            // For range requests, add start position to content-length
            start_pos + content_length
        } else if content_length > 0 {
            content_length
        } else {
            total_size
        };
        
        // Update progress bar with correct total
        if actual_total > 0 && actual_total != total_size {
            pb.set_length(actual_total);
        }

        // Stream download
        let mut stream = response.bytes_stream();
        let mut downloaded = start_pos;

        use futures_util::StreamExt;
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| FerrumError::model(format!("Download error for {}: {}", filename, e)))?;
            
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            pb.set_position(downloaded);
        }

        file.flush().await?;
        drop(file);

        // Verify size
        let final_size = fs::metadata(&incomplete_path).await?.len();
        if total_size > 0 && final_size != total_size {
            pb.finish_with_message(format!("{} ⚠ 不完整", display_name));
            return Err(FerrumError::model(format!(
                "Incomplete download for {}: got {} bytes, expected {}",
                filename, final_size, total_size
            )));
        }

        // Rename to final path
        fs::rename(&incomplete_path, &blob_path).await?;
        
        pb.finish_with_message(format!("{} ✓ {}", display_name, format_size(final_size)));

        // Create symlink in snapshot directory
        create_symlink(&blob_path, &snapshot_file).await?;

        Ok(())
    }
}

/// Create a symlink (or copy on Windows)
async fn create_symlink(src: &Path, dst: &Path) -> Result<()> {
    // Remove existing file/link
    if dst.exists() || dst.is_symlink() {
        fs::remove_file(dst).await.ok();
    }

    // Create parent directory
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent).await?;
    }

    // Create relative symlink
    let relative_src =
        pathdiff::diff_paths(src, dst.parent().unwrap()).unwrap_or_else(|| src.to_path_buf());

    #[cfg(unix)]
    {
        tokio::fs::symlink(&relative_src, dst).await?;
    }

    #[cfg(windows)]
    {
        // On Windows, copy instead of symlink (symlinks require admin)
        fs::copy(src, dst).await?;
    }

    Ok(())
}

/// Simple hash for fallback blob naming
fn simple_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Format file size for display
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
