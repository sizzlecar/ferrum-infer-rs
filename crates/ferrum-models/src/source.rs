//! Model source resolution and downloading with progress tracking

use ferrum_types::{FerrumError, ModelSource, Result};
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelSourceRuntimeEnv {
    hf_home: Option<String>,
    hf_token: Option<String>,
}

impl ModelSourceRuntimeEnv {
    fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<String>,
    {
        let mut hf_home = None;
        let mut hf_token = None;
        let mut hf_hub_token = None;

        for (key, value) in vars {
            let value = value.into();
            match key.as_ref() {
                "HF_HOME" => hf_home = Some(value),
                "HF_TOKEN" => hf_token = Some(value),
                "HUGGING_FACE_HUB_TOKEN" => hf_hub_token = Some(value),
                _ => {}
            }
        }

        Self {
            hf_home,
            hf_token: hf_token.or(hf_hub_token),
        }
    }
}

fn model_source_runtime_env() -> &'static ModelSourceRuntimeEnv {
    static CONFIG: OnceLock<ModelSourceRuntimeEnv> = OnceLock::new();
    CONFIG.get_or_init(ModelSourceRuntimeEnv::from_env)
}

/// Configuration for model source resolution
#[derive(Debug, Clone)]
pub struct ModelSourceConfig {
    pub cache_dir: Option<PathBuf>,
    pub hf_token: Option<String>,
    pub offline_mode: bool,
    pub max_retries: usize,
    pub download_timeout: u64,
    pub use_file_lock: bool,
}

impl Default for ModelSourceConfig {
    fn default() -> Self {
        // Use HuggingFace standard cache directory
        let default_cache = model_source_runtime_env()
            .hf_home
            .clone()
            .or_else(|| {
                dirs::home_dir()
                    .map(|h| h.join(".cache/huggingface"))
                    .and_then(|p| p.to_str().map(String::from))
            })
            .map(PathBuf::from);

        Self {
            cache_dir: default_cache,
            hf_token: Self::get_hf_token(),
            offline_mode: false,
            max_retries: 3,
            download_timeout: 300,
            use_file_lock: true,
        }
    }
}

impl ModelSourceConfig {
    pub fn get_hf_token() -> Option<String> {
        model_source_runtime_env().hf_token.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    SafeTensors,
    PyTorchBin,
    GGUF,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ResolvedModelSource {
    pub original: String,
    pub local_path: PathBuf,
    pub format: ModelFormat,
    pub from_cache: bool,
}

impl From<ResolvedModelSource> for ModelSource {
    fn from(value: ResolvedModelSource) -> Self {
        ModelSource::Local(value.local_path.display().to_string())
    }
}

#[async_trait::async_trait]
pub trait ModelSourceResolver: Send + Sync {
    async fn resolve(&self, id: &str, revision: Option<&str>) -> Result<ResolvedModelSource>;
}

pub struct DefaultModelSourceResolver {
    _config: ModelSourceConfig,
    api: Api,
}

impl DefaultModelSourceResolver {
    pub fn new(config: ModelSourceConfig) -> Self {
        let mut builder = ApiBuilder::new();

        if let Some(cache_dir) = &config.cache_dir {
            builder = builder.with_cache_dir(cache_dir.clone());
        }

        if let Some(token) = &config.hf_token {
            builder = builder.with_token(Some(token.clone()));
        }

        let api = builder.build().unwrap_or_else(|e| {
            warn!("Failed to build HF API: {}, using default", e);
            Api::new().expect("Failed to create default HF API")
        });

        Self {
            _config: config,
            api,
        }
    }

    fn is_local_path(id: &str) -> bool {
        Path::new(id).exists()
    }

    fn detect_format(path: &Path) -> ModelFormat {
        if path.join("model.safetensors").exists()
            || path.join("model.safetensors.index.json").exists()
        {
            ModelFormat::SafeTensors
        } else if path.join("pytorch_model.bin").exists() {
            ModelFormat::PyTorchBin
        } else {
            ModelFormat::Unknown
        }
    }

    async fn resolve_local(&self, path: &str) -> Result<ResolvedModelSource> {
        let path_buf = PathBuf::from(path);

        if !path_buf.exists() {
            return Err(FerrumError::model(format!("Path does not exist: {}", path)));
        }

        let format = Self::detect_format(&path_buf);

        Ok(ResolvedModelSource {
            original: path.to_string(),
            local_path: path_buf,
            format,
            from_cache: true,
        })
    }

    /// Download file with progress monitoring
    async fn download_with_monitor(
        &self,
        repo: &ApiRepo,
        filename: &str,
        expected_cache_dir: &Path,
    ) -> Result<PathBuf> {
        info!("📥 下载中: {}...", filename);

        let done = Arc::new(AtomicBool::new(false));
        let done_clone = done.clone();
        let filename_str = filename.to_string();

        // Start monitor task
        let monitor_task = tokio::spawn({
            let done = done.clone();
            let filename = filename_str.clone();
            let cache_dir = expected_cache_dir.to_path_buf();

            async move {
                tokio::time::sleep(Duration::from_millis(1000)).await;

                let start_time = Instant::now();
                let mut last_size = 0u64;
                let mut last_time = Instant::now();
                let mut last_print = Instant::now();

                while !done.load(Ordering::SeqCst) {
                    // Try to find downloading file
                    if let Some(current_size) = find_downloading_file(&cache_dir, &filename) {
                        let elapsed_since_last = last_time.elapsed().as_secs_f64();

                        if elapsed_since_last > 0.5 && current_size > last_size {
                            let delta = current_size - last_size;
                            let speed_mbps = delta as f64 / elapsed_since_last / 1024.0 / 1024.0;
                            let current_mb = current_size as f64 / 1024.0 / 1024.0;

                            // Only print every 2 seconds to avoid spam
                            if last_print.elapsed().as_secs() >= 2 {
                                info!(
                                    "  📊 已下载: {:.2} MB (速度: {:.1} MB/s)",
                                    current_mb, speed_mbps
                                );
                                last_print = Instant::now();
                            }

                            last_size = current_size;
                            last_time = Instant::now();
                        }
                    }

                    tokio::time::sleep(Duration::from_millis(500)).await;
                }

                // Final statistics
                let total_time = start_time.elapsed().as_secs_f64();
                if last_size > 0 && total_time > 0.0 {
                    let avg_speed = last_size as f64 / total_time / 1024.0 / 1024.0;
                    info!(
                        "  ✅ 下载完成: {:.2} MB (平均速度: {:.1} MB/s, 耗时: {:.1}s)",
                        last_size as f64 / 1024.0 / 1024.0,
                        avg_speed,
                        total_time
                    );
                }
            }
        });

        // Do the actual download (blocking, but monitored)
        let path = repo
            .get(&filename_str)
            .await
            .map_err(|e| FerrumError::model(format!("Download failed: {}", e)))?;

        // Signal completion
        done_clone.store(true, Ordering::SeqCst);

        // Wait for monitor to finish
        let _ = monitor_task.await;

        Ok(path)
    }

    async fn resolve_huggingface(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<ResolvedModelSource> {
        info!("🔍 正在解析模型: {}", repo_id);

        let repo = if let Some(rev) = revision {
            self.api.repo(hf_hub::Repo::with_revision(
                repo_id.to_string(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            ))
        } else {
            self.api.repo(hf_hub::Repo::new(
                repo_id.to_string(),
                hf_hub::RepoType::Model,
            ))
        };

        // Download config first (small file, no need for progress)
        info!("📥 下载中: config.json...");
        let config_path = repo
            .get("config.json")
            .await
            .map_err(|e| FerrumError::model(format!("Failed to download config: {}", e)))?;

        info!("✅ config.json 下载完成");

        let model_dir = config_path
            .parent()
            .ok_or_else(|| FerrumError::model("Invalid cache path"))?
            .to_path_buf();

        info!("📁 缓存目录: {:?}", model_dir);

        // Download tokenizer files (critical for inference)
        self.download_tokenizer_files(&repo).await?;

        // Download weights
        let format = self.download_weights(&repo, &model_dir).await?;

        Ok(ResolvedModelSource {
            original: repo_id.to_string(),
            local_path: model_dir,
            format,
            from_cache: false,
        })
    }

    async fn download_tokenizer_files(&self, repo: &ApiRepo) -> Result<()> {
        info!("📥 下载 tokenizer 文件...");

        // List of common tokenizer files
        let tokenizer_files = vec![
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
        ];

        let mut downloaded_count = 0;
        for filename in &tokenizer_files {
            match repo.get(filename).await {
                Ok(_path) => {
                    info!("  ✅ {}", filename);
                    downloaded_count += 1;
                }
                Err(e) => {
                    debug!("  ⏭️  {} (optional): {}", filename, e);
                }
            }
        }

        if downloaded_count > 0 {
            info!("✅ Tokenizer 文件下载完成 ({} 个文件)", downloaded_count);
        } else {
            warn!("⚠️  未找到 tokenizer 文件，可能影响推理");
        }

        Ok(())
    }

    async fn download_weights(&self, repo: &ApiRepo, model_dir: &Path) -> Result<ModelFormat> {
        // Try SafeTensors single file
        info!("🔍 检查 model.safetensors...");
        match self
            .download_with_monitor(repo, "model.safetensors", model_dir)
            .await
        {
            Ok(path) => {
                if let Ok(metadata) = std::fs::metadata(&path) {
                    info!(
                        "✅ model.safetensors 完成 ({:.2} GB)",
                        metadata.len() as f64 / 1e9
                    );
                }
                return Ok(ModelFormat::SafeTensors);
            }
            Err(e) => debug!("model.safetensors not found: {}", e),
        }

        // Try sharded SafeTensors
        info!("🔍 检查分片模型...");
        match repo.get("model.safetensors.index.json").await {
            Ok(index_path) => {
                info!("✅ 发现分片 SafeTensors 模型");

                let content = std::fs::read_to_string(&index_path)
                    .map_err(|e| FerrumError::io(format!("Failed to read index: {}", e)))?;

                let index: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|e| FerrumError::model(format!("Failed to parse index: {}", e)))?;

                if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
                    let shards: std::collections::HashSet<_> =
                        weight_map.values().filter_map(|v| v.as_str()).collect();

                    let total = shards.len();
                    info!("📦 需要下载 {} 个分片", total);

                    let mut total_bytes = 0u64;
                    for (i, shard) in shards.iter().enumerate() {
                        info!("📥 [{}/{}] {}", i + 1, total, shard);

                        let shard_path = self.download_with_monitor(repo, shard, model_dir).await?;

                        if let Ok(meta) = std::fs::metadata(&shard_path) {
                            let size = meta.len();
                            total_bytes += size;
                            info!(
                                "📊 进度: [{}/{}] 分片, 累计 {:.2} GB",
                                i + 1,
                                total,
                                total_bytes as f64 / 1e9
                            );
                        }
                    }

                    info!(
                        "🎉 全部下载完成! 总大小: {:.2} GB",
                        total_bytes as f64 / 1e9
                    );
                }

                return Ok(ModelFormat::SafeTensors);
            }
            Err(e) => debug!("Sharded model not found: {}", e),
        }

        // Try PyTorch
        info!("🔍 检查 pytorch_model.bin...");
        match self
            .download_with_monitor(repo, "pytorch_model.bin", model_dir)
            .await
        {
            Ok(path) => {
                warn!("⚠️  使用 PyTorch 格式 (推荐使用 SafeTensors)");
                if let Ok(meta) = std::fs::metadata(&path) {
                    info!(
                        "✅ pytorch_model.bin 完成 ({:.2} GB)",
                        meta.len() as f64 / 1e9
                    );
                }
                return Ok(ModelFormat::PyTorchBin);
            }
            Err(e) => debug!("pytorch_model.bin not found: {}", e),
        }

        if Self::detect_format(model_dir) == ModelFormat::GGUF {
            return Ok(ModelFormat::GGUF);
        }

        Err(FerrumError::model("未找到支持的模型格式"))
    }
}

/// Find downloading file in cache directory
fn find_downloading_file(cache_dir: &Path, _filename: &str) -> Option<u64> {
    // Just search for ANY .part file in the cache directory tree
    // This is more reliable than trying to match filenames

    // Check blobs directory
    if let Ok(entries) = std::fs::read_dir(cache_dir.join("blobs")) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let path_str = path.to_string_lossy();

            if path_str.ends_with(".part") || path_str.contains(".sync.part") {
                if let Ok(metadata) = std::fs::metadata(&path) {
                    return Some(metadata.len());
                }
            }
        }
    }

    // Also try to find in parent directories
    let mut current = cache_dir.to_path_buf();
    for _ in 0..3 {
        if let Ok(entries) = std::fs::read_dir(&current) {
            for entry in entries.filter_map(|e| e.ok()) {
                if entry.path().is_dir() {
                    if let Some(size) = scan_dir_for_part_files(&entry.path()) {
                        return Some(size);
                    }
                }
            }
        }

        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }

    None
}

/// Recursively scan directory for .part files
fn scan_dir_for_part_files(dir: &Path) -> Option<u64> {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let path_str = path.to_string_lossy();

            if path_str.ends_with(".part") || path_str.contains(".sync.part") {
                if let Ok(metadata) = std::fs::metadata(&path) {
                    return Some(metadata.len());
                }
            }

            if path.is_dir() {
                if let Some(size) = scan_dir_for_part_files(&path) {
                    return Some(size);
                }
            }
        }
    }
    None
}

#[async_trait::async_trait]
impl ModelSourceResolver for DefaultModelSourceResolver {
    async fn resolve(&self, id: &str, revision: Option<&str>) -> Result<ResolvedModelSource> {
        if Self::is_local_path(id) {
            return self.resolve_local(id).await;
        }

        self.resolve_huggingface(id, revision).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_source_runtime_env_parses_hf_cache_and_token() {
        let env = ModelSourceRuntimeEnv::from_env_vars([
            ("HF_HOME", "/tmp/hf"),
            ("HF_TOKEN", "primary"),
            ("HUGGING_FACE_HUB_TOKEN", "fallback"),
        ]);

        assert_eq!(env.hf_home.as_deref(), Some("/tmp/hf"));
        assert_eq!(env.hf_token.as_deref(), Some("primary"));
    }

    #[test]
    fn model_source_runtime_env_uses_hub_token_fallback() {
        let env = ModelSourceRuntimeEnv::from_env_vars([("HUGGING_FACE_HUB_TOKEN", "fallback")]);

        assert_eq!(env.hf_home, None);
        assert_eq!(env.hf_token.as_deref(), Some("fallback"));
    }
}
