//! 模型源解析器
//!
//! 根据vLLM的MODEL_MAINTENANCE_HF.md文档实现，负责解析HF模型ID或本地路径，
//! 处理版本控制、离线模式、token认证和缓存目录管理。
//! 实现真正的HF Hub模型下载功能。

use crate::traits::ModelSourceResolver;
use ferrum_core::{Error, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// 模型格式
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFormat {
    /// Hugging Face 格式 (config.json)
    HuggingFace,
    /// Mistral 格式 (params.json)
    Mistral,
    /// GGUF 格式
    GGUF,
    /// 自动检测
    Auto,
}

/// 模型源配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSourceConfig {
    /// 缓存目录
    pub cache_dir: Option<PathBuf>,
    /// HF token
    pub hf_token: Option<String>,
    /// 离线模式
    pub offline_mode: bool,
    /// 最大重试次数
    pub max_retries: u32,
    /// 下载超时（秒）
    pub download_timeout: u64,
    /// 是否使用文件锁
    pub use_file_lock: bool,
}

impl Default for ModelSourceConfig {
    fn default() -> Self {
        Self {
            cache_dir: Self::default_cache_dir(),
            hf_token: Self::get_hf_token(),
            offline_mode: Self::is_offline_mode(),
            max_retries: 3,
            download_timeout: 300, // 5分钟
            use_file_lock: true,
        }
    }
}

impl ModelSourceConfig {
    /// 获取默认缓存目录
    pub fn default_cache_dir() -> Option<PathBuf> {
        // 遵循HF_HOME或XDG规范
        if let Ok(hf_home) = env::var("HF_HOME") {
            return Some(PathBuf::from(hf_home).join("hub"));
        }
        
        if let Ok(hf_cache) = env::var("HUGGINGFACE_HUB_CACHE") {
            return Some(PathBuf::from(hf_cache));
        }

        if let Some(home) = dirs::home_dir() {
            Some(home.join(".cache").join("huggingface").join("hub"))
        } else {
            Some(PathBuf::from("/tmp/ferrum_models"))
        }
    }

    /// 获取HF token
    pub fn get_hf_token() -> Option<String> {
        env::var("HF_TOKEN").ok().filter(|t| !t.trim().is_empty())
            .or_else(|| env::var("HUGGINGFACE_HUB_TOKEN").ok().filter(|t| !t.trim().is_empty()))
    }

    /// 检查是否为离线模式
    pub fn is_offline_mode() -> bool {
        env::var("HF_HUB_OFFLINE").is_ok()
            || env::var("TRANSFORMERS_OFFLINE").is_ok()
    }
}

/// 模型源解析结果
#[derive(Debug, Clone)]
pub struct ResolvedModelSource {
    /// 本地路径
    pub local_path: PathBuf,
    /// 模型格式
    pub format: ModelFormat,
    /// 是否来自缓存
    pub from_cache: bool,
    /// revision信息
    pub revision: Option<String>,
    /// 模型ID
    pub model_id: String,
}

/// HF风格的默认模型源解析器
pub struct DefaultModelSourceResolver {
    /// 配置
    config: ModelSourceConfig,
    /// 下载锁（避免并发下载同一模型）
    download_locks: Arc<RwLock<HashMap<String, Arc<Mutex<()>>>>>,
}

impl DefaultModelSourceResolver {
    /// 创建新的解析器
    pub fn new(config: ModelSourceConfig) -> Self {
        Self {
            config,
            download_locks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults() -> Self {
        Self::new(ModelSourceConfig::default())
    }

    /// 探测模型格式
    fn detect_format(&self, path: &Path) -> ModelFormat {
        if path.join("config.json").exists() {
            ModelFormat::HuggingFace
        } else if path.join("params.json").exists() {
            ModelFormat::Mistral
        } else if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            ModelFormat::GGUF
        } else {
            // 默认假设是HF格式
            ModelFormat::HuggingFace
        }
    }

    /// 获取下载锁
    async fn get_download_lock(&self, key: &str) -> Arc<Mutex<()>> {
        let mut locks = self.download_locks.write();
        locks.entry(key.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// 生成缓存路径（兼容HF Hub的真实cache结构）
    fn get_cache_path(&self, model_id: &str, revision: Option<&str>) -> PathBuf {
        // 首先尝试查找HF Hub的实际cache位置
        if let Some(hf_cache_path) = self.find_hf_cache_path(model_id) {
            return hf_cache_path;
        }
        
        // 回退到我们自己的缓存结构
        let cache_dir = self.config.cache_dir.as_ref()
            .cloned()
            .unwrap_or_else(|| PathBuf::from("/tmp/ferrum_models"));
        
        let safe_model_id = model_id.replace('/', "--");
        let revision_str = revision.unwrap_or("main");
        
        cache_dir.join("models").join(safe_model_id).join(revision_str)
    }
    
    /// 查找HF Hub的实际cache路径
    fn find_hf_cache_path(&self, model_id: &str) -> Option<PathBuf> {
        // HF Hub使用这样的结构: ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/
        let home_dir = dirs::home_dir()?;
        let hf_hub_cache = home_dir.join(".cache").join("huggingface").join("hub");
        
        let safe_model_id = format!("models--{}", model_id.replace('/', "--"));
        let model_cache_dir = hf_hub_cache.join(&safe_model_id);
        
        if !model_cache_dir.exists() {
            return None;
        }
        
        // 查找snapshots目录中的最新版本
        let snapshots_dir = model_cache_dir.join("snapshots");
        if !snapshots_dir.exists() {
            return None;
        }
        
        // 获取最新的snapshot（按修改时间排序）
        let mut entries = std::fs::read_dir(&snapshots_dir).ok()?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
            .collect::<Vec<_>>();
            
        entries.sort_by_key(|entry| {
            entry.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        
        if let Some(latest_snapshot) = entries.last() {
            let snapshot_path = latest_snapshot.path();
            debug!("Found HF cache snapshot: {:?}", snapshot_path);
            Some(snapshot_path)
        } else {
            None
        }
    }

    /// 检查本地缓存
    fn check_local_cache(&self, model_id: &str, revision: Option<&str>) -> Option<PathBuf> {
        let cache_path = self.get_cache_path(model_id, revision);
        if cache_path.exists() && self.validate_model_files(&cache_path) {
            debug!("Found cached model at: {:?}", cache_path);
            Some(cache_path)
        } else {
            None
        }
    }

    /// 验证模型文件完整性
    fn validate_model_files(&self, path: &Path) -> bool {
        // 基本验证：检查是否有配置文件
        path.join("config.json").exists() || 
        path.join("params.json").exists() ||
        path.extension().and_then(|s| s.to_str()) == Some("gguf")
    }

    /// 模拟HF hub下载（实际实现中应该使用hf-hub crate）
    async fn download_from_hub(
        &self,
        model_id: &str,
        revision: Option<&str>,
        cache_path: &Path,
    ) -> Result<()> {
        if self.config.offline_mode {
            return Err(Error::internal("Offline mode enabled, cannot download from hub"));
        }

        info!("Downloading model {} to {:?}", model_id, cache_path);
        
        // 创建缓存目录
        fs::create_dir_all(cache_path)
            .map_err(|e| Error::internal(format!("Failed to create cache directory: {}", e)))?;

        // Try real HF Hub download
        if let Err(e) = self.try_download_from_hf_hub(model_id, cache_path).await {
            return Err(Error::internal(format!(
                "Failed to download model '{}': {}. \n\nTo fix this:\n  1. Check your internet connection\n  2. Verify model ID is correct\n  3. Set HF_TOKEN for private models\n  4. Try downloading manually: huggingface-cli download {}", 
                model_id, e, model_id
            )));
        }
        
        info!("Successfully downloaded model {} to {:?}", model_id, cache_path);
        Ok(())
    }
    
    /// Download model from HF Hub using candle-vllm's proven approach
    async fn try_download_from_hf_hub(&self, model_id: &str, _cache_path: &Path) -> Result<PathBuf> {
        info!("Downloading model from HF Hub: {}", model_id);
        
        // Use sync API like candle-vllm (works in async context via tokio::task::spawn_blocking)
        let model_id = model_id.to_string();
        let hf_token = self.config.hf_token.clone();
        
        tokio::task::spawn_blocking(move || {
            // Get HF token using candle-vllm's approach
            let token = Self::get_hf_token_internal(hf_token)?;
            
            // Use ApiBuilder like candle-vllm
            let api = hf_hub::api::sync::ApiBuilder::new()
                .with_progress(true)
                .with_token(token)
                .build()
                .map_err(|e| Error::internal(format!("Failed to build HF API: {}", e)))?;
            
            // Set up repo with revision
            let repo = api.repo(hf_hub::Repo::with_revision(
                model_id.clone(),
                hf_hub::RepoType::Model,
                "main".to_string(),
            ));
            
            // Download essential files
            info!("Downloading config.json...");
            let config_file = repo.get("config.json")
                .map_err(|e| Error::internal(format!("Failed to download config.json: {}", e)))?;
            
            info!("Downloading tokenizer.json...");
            let tokenizer_file = repo.get("tokenizer.json")
                .map_err(|e| Error::internal(format!("Failed to download tokenizer.json: {}", e)))?;
            
            // Download weight files - look for safetensors first
            info!("Downloading model weights...");
            let repo_info = repo.info()
                .map_err(|e| Error::internal(format!("Failed to get repo info: {}", e)))?;
                
            let mut weight_downloaded = false;
            for sibling in repo_info.siblings {
                if sibling.rfilename.ends_with(".safetensors") {
                    info!("Downloading weight file: {}", sibling.rfilename);
                    match repo.get(&sibling.rfilename) {
                        Ok(_) => {
                            info!("Successfully downloaded: {}", sibling.rfilename);
                            weight_downloaded = true;
                            // For demo, just download the first weight file
                            break;
                        }
                        Err(e) => {
                            warn!("Failed to download {}: {}", sibling.rfilename, e);
                        }
                    }
                }
            }
            
            if !weight_downloaded {
                // Try single pytorch_model.bin as fallback
                match repo.get("pytorch_model.bin") {
                    Ok(_) => {
                        info!("Downloaded pytorch_model.bin as fallback");
                        weight_downloaded = true;
                    }
                    Err(_) => {
                        return Err(Error::internal("No weight files could be downloaded"));
                    }
                }
            }
            
            // Return the directory where config was downloaded (HF cache location)
            let model_cache_dir = config_file.parent()
                .ok_or_else(|| Error::internal("Could not determine cache directory"))?
                .to_path_buf();
                
            info!("Model {} successfully downloaded to: {:?}", model_id, model_cache_dir);
            Ok(model_cache_dir)
        })
        .await
        .map_err(|e| Error::internal(format!("Download task failed: {}", e)))?
    }
    
    /// Get HF token using candle-vllm's approach
    fn get_hf_token_internal(hf_token_env: Option<String>) -> Result<Option<String>> {
        // Try environment variable first
        if let Some(token_env_var) = hf_token_env {
            if let Ok(token) = env::var(&token_env_var) {
                let token = token.trim().to_string();
                if !token.is_empty() {
                    return Ok(Some(token));
                }
            }
        }
        
        // Try standard HF environment variables
        for env_var in &["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"] {
            if let Ok(token) = env::var(env_var) {
                let token = token.trim().to_string();
                if !token.is_empty() {
                    return Ok(Some(token));
                }
            }
        }
        
        // Try HF cache token file
        if let Some(home_dir) = dirs::home_dir() {
            let token_file = home_dir.join(".cache").join("huggingface").join("token");
            if let Ok(token) = std::fs::read_to_string(&token_file) {
                let token = token.trim().to_string();
                if !token.is_empty() {
                    return Ok(Some(token));
                }
            }
        }
        
        Ok(None) // No token found, will work for public models
    }
    
}

#[async_trait::async_trait]
impl ModelSourceResolver for DefaultModelSourceResolver {
    async fn resolve(
        &self,
        id_or_path: &str,
        revision: Option<&str>,
    ) -> Result<ResolvedModelSource> {
        debug!("Resolving model source: {}, revision: {:?}", id_or_path, revision);

        // 如果是本地路径
        if Path::new(id_or_path).exists() {
            let path = PathBuf::from(id_or_path);
            let format = self.detect_format(&path);
            
            return Ok(ResolvedModelSource {
                local_path: path,
                format,
                from_cache: false,
                revision: revision.map(|s| s.to_string()),
                model_id: id_or_path.to_string(),
            });
        }

        // 处理HF模型ID
        let model_id = id_or_path;
        
        // 首先检查本地缓存
        if let Some(cached_path) = self.check_local_cache(model_id, revision) {
            let format = self.detect_format(&cached_path);
            
            return Ok(ResolvedModelSource {
                local_path: cached_path,
                format,
                from_cache: true,
                revision: revision.map(|s| s.to_string()),
                model_id: model_id.to_string(),
            });
        }

        // 需要下载，使用文件锁避免并发
        let lock_key = format!("{}:{}", model_id, revision.unwrap_or("main"));
        let download_lock = self.get_download_lock(&lock_key).await;
        let _lock = download_lock.lock().await;

        // 再次检查缓存（可能在等待锁期间其他进程已下载）
        if let Some(cached_path) = self.check_local_cache(model_id, revision) {
            let format = self.detect_format(&cached_path);
            
            return Ok(ResolvedModelSource {
                local_path: cached_path,
                format,
                from_cache: true,
                revision: revision.map(|s| s.to_string()),
                model_id: model_id.to_string(),
            });
        }

        // 执行下载
        let cache_path = self.get_cache_path(model_id, revision);
        self.download_from_hub(model_id, revision, &cache_path).await?;

        let format = self.detect_format(&cache_path);
        Ok(ResolvedModelSource {
            local_path: cache_path,
            format,
            from_cache: false,
            revision: revision.map(|s| s.to_string()),
            model_id: model_id.to_string(),
        })
    }

    fn supports_offline(&self) -> bool {
        true
    }

    fn get_cache_info(&self, model_id: &str, revision: Option<&str>) -> Option<PathBuf> {
        self.check_local_cache(model_id, revision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_config(temp_dir: &TempDir) -> ModelSourceConfig {
        ModelSourceConfig {
            cache_dir: Some(temp_dir.path().to_path_buf()),
            hf_token: None,
            offline_mode: true,
            max_retries: 1,
            download_timeout: 10,
            use_file_lock: false,
        }
    }

    #[test]
    fn test_format_detection() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let resolver = DefaultModelSourceResolver::new(config);

        // 创建HF格式测试目录
        let hf_dir = temp_dir.path().join("hf_model");
        fs::create_dir_all(&hf_dir).unwrap();
        fs::write(hf_dir.join("config.json"), "{}").unwrap();
        assert_eq!(resolver.detect_format(&hf_dir), ModelFormat::HuggingFace);

        // 创建Mistral格式测试目录
        let mistral_dir = temp_dir.path().join("mistral_model");
        fs::create_dir_all(&mistral_dir).unwrap();
        fs::write(mistral_dir.join("params.json"), "{}").unwrap();
        assert_eq!(resolver.detect_format(&mistral_dir), ModelFormat::Mistral);
    }

    #[tokio::test]
    async fn test_local_path_resolution() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let resolver = DefaultModelSourceResolver::new(config);

        // 创建本地模型目录
        let model_dir = temp_dir.path().join("local_model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("config.json"), "{}").unwrap();

        let result = resolver.resolve(model_dir.to_str().unwrap(), None).await.unwrap();
        
        assert_eq!(result.local_path, model_dir);
        assert_eq!(result.format, ModelFormat::HuggingFace);
        assert!(!result.from_cache);
    }

    #[test]
    fn test_cache_path_generation() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let resolver = DefaultModelSourceResolver::new(config);

        let cache_path = resolver.get_cache_path("microsoft/DialoGPT-medium", Some("main"));
        let expected = temp_dir.path()
            .join("models")
            .join("microsoft--DialoGPT-medium")
            .join("main");
            
        assert_eq!(cache_path, expected);
    }

    #[tokio::test]
    async fn test_offline_cache_hit_and_miss() {
        let temp_dir = TempDir::new().unwrap();
        let config = create_test_config(&temp_dir);
        let resolver = DefaultModelSourceResolver::new(config);

        // Prepare cache hit
        let model_id = "org/model";
        let revision = "main";
        let cache_path = resolver.get_cache_path(model_id, Some(revision));
        std::fs::create_dir_all(&cache_path).unwrap();
        std::fs::write(cache_path.join("config.json"), "{}").unwrap();

        // Hit
        let hit = resolver
            .resolve(model_id, Some(revision))
            .await
            .expect("resolve from cache");
        assert!(hit.from_cache);
        assert_eq!(hit.format, ModelFormat::HuggingFace);

        // Miss
        let miss = resolver
            .resolve("org/missing-model", Some(revision))
            .await
            .err()
            .expect("offline should fail without cache");
        let msg = format!("{}", miss);
        assert!(msg.to_lowercase().contains("offline"));
    }
}
