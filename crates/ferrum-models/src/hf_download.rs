//! HuggingFace model downloader with proxy and resume support
//!
//! Features:
//! - SOCKS5/HTTP proxy support via environment variables
//! - Resumable downloads (断点续传)
//! - Progress bar with speed display
//! - HuggingFace token authentication

use ferrum_types::{FerrumError, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncSeekExt, AsyncWriteExt, SeekFrom};

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
        println!(
            "📦 Downloading {} files ({:.2} GB)",
            files_to_download.len(),
            total_size as f64 / 1_073_741_824.0
        );

        // Download each file
        for file_info in &files_to_download {
            self.download_file_with_resume(
                model_id,
                revision,
                &file_info.path,
                file_info.size.unwrap_or(0),
                &blobs_dir,
                &snapshot_dir,
            )
            .await?;
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

    /// Download a single file with resume support
    async fn download_file_with_resume(
        &self,
        model_id: &str,
        revision: &str,
        filename: &str,
        expected_size: u64,
        blobs_dir: &Path,
        snapshot_dir: &Path,
    ) -> Result<()> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            HF_API_URL, model_id, revision, filename
        );

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

        // Get content length and ETag
        let total_size = head_resp
            .content_length()
            .unwrap_or(expected_size);
        
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
                if total_size == 0 || meta.len() == total_size {
                    create_symlink(&blob_path, &snapshot_file).await?;
                    println!("  ✓ {} (cached)", filename);
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

        // Create progress bar
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner:.green} {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("█▓▒░"),
        );
        pb.set_message(filename.to_string());

        // If resuming, set initial position
        if resume_from > 0 && resume_from < total_size {
            pb.set_position(resume_from);
            pb.set_message(format!("{} (resuming)", filename));
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
            pb.finish_with_message(format!("{} ⚠ incomplete ({}/{})", filename, final_size, total_size));
            return Err(FerrumError::model(format!(
                "Incomplete download for {}: got {} bytes, expected {}",
                filename, final_size, total_size
            )));
        }

        // Rename to final path
        fs::rename(&incomplete_path, &blob_path).await?;
        
        pb.finish_with_message(format!("{} ✓", filename));

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
