//! Tokenizer caching and sharing

use crate::{IncrementalTokenizer, Tokenizer};
use dashmap::DashMap;
use ferrum_types::Result;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Global tokenizer cache for sharing across requests
pub struct TokenizerCache {
    /// Cache of tokenizers by identifier
    cache: DashMap<String, Arc<dyn Tokenizer + Send + Sync>>,
    /// Cache of incremental tokenizers by identifier  
    incremental_cache: DashMap<String, Arc<dyn IncrementalTokenizer + Send + Sync>>,
    /// Maximum cache size
    max_size: usize,
}

impl TokenizerCache {
    /// Create new tokenizer cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: DashMap::new(),
            incremental_cache: DashMap::new(),
            max_size,
        }
    }

    /// Get or create tokenizer
    pub async fn get_or_create<F, Fut>(
        &self,
        key: &str,
        factory: F,
    ) -> Result<Arc<dyn Tokenizer + Send + Sync>>
    where
        F: FnOnce() -> Fut + Send,
        Fut: std::future::Future<Output = Result<Box<dyn Tokenizer + Send + Sync>>> + Send,
    {
        // Fast path: check if already cached
        if let Some(tokenizer) = self.cache.get(key) {
            debug!("Cache hit for tokenizer: {}", key);
            return Ok(tokenizer.clone());
        }

        // Slow path: create and cache
        debug!("Cache miss for tokenizer: {}, creating new instance", key);

        // Check cache size and evict if necessary
        if self.cache.len() >= self.max_size {
            self.evict_oldest();
        }

        let tokenizer = factory().await?;
        let arc_tokenizer = Arc::from(tokenizer);

        self.cache.insert(key.to_string(), arc_tokenizer.clone());
        info!("Cached tokenizer: {}", key);

        Ok(arc_tokenizer)
    }

    /// Get or create incremental tokenizer
    pub async fn get_or_create_incremental<F, Fut>(
        &self,
        key: &str,
        factory: F,
    ) -> Result<Arc<dyn IncrementalTokenizer + Send + Sync>>
    where
        F: FnOnce() -> Fut + Send,
        Fut: std::future::Future<Output = Result<Box<dyn IncrementalTokenizer + Send + Sync>>>
            + Send,
    {
        // Fast path: check if already cached
        if let Some(tokenizer) = self.incremental_cache.get(key) {
            debug!("Cache hit for incremental tokenizer: {}", key);
            return Ok(tokenizer.clone());
        }

        // Slow path: create and cache
        debug!(
            "Cache miss for incremental tokenizer: {}, creating new instance",
            key
        );

        // Check cache size and evict if necessary
        if self.incremental_cache.len() >= self.max_size {
            self.evict_oldest_incremental();
        }

        let tokenizer = factory().await?;
        let arc_tokenizer = Arc::from(tokenizer);

        self.incremental_cache
            .insert(key.to_string(), arc_tokenizer.clone());
        info!("Cached incremental tokenizer: {}", key);

        Ok(arc_tokenizer)
    }

    /// Remove tokenizer from cache
    pub fn remove(&self, key: &str) -> Option<Arc<dyn Tokenizer + Send + Sync>> {
        debug!("Removing tokenizer from cache: {}", key);
        self.cache.remove(key).map(|(_, v)| v)
    }

    /// Remove incremental tokenizer from cache
    pub fn remove_incremental(
        &self,
        key: &str,
    ) -> Option<Arc<dyn IncrementalTokenizer + Send + Sync>> {
        debug!("Removing incremental tokenizer from cache: {}", key);
        self.incremental_cache.remove(key).map(|(_, v)| v)
    }

    /// Clear all cached tokenizers
    pub fn clear(&self) {
        info!("Clearing tokenizer cache");
        self.cache.clear();
        self.incremental_cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            incremental_size: self.incremental_cache.len(),
            max_size: self.max_size,
            utilization: (self.cache.len() + self.incremental_cache.len()) as f32
                / (self.max_size * 2) as f32,
        }
    }

    /// Evict oldest tokenizer (simple LRU approximation)
    fn evict_oldest(&self) {
        if let Some(entry) = self.cache.iter().next() {
            let key = entry.key().clone();
            warn!("Evicting tokenizer from cache: {}", key);
            self.cache.remove(&key);
        }
    }

    /// Evict oldest incremental tokenizer
    fn evict_oldest_incremental(&self) {
        if let Some(entry) = self.incremental_cache.iter().next() {
            let key = entry.key().clone();
            warn!("Evicting incremental tokenizer from cache: {}", key);
            self.incremental_cache.remove(&key);
        }
    }
}

impl Default for TokenizerCache {
    fn default() -> Self {
        Self::new(32) // Default cache size of 32 tokenizers
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached regular tokenizers
    pub size: usize,
    /// Number of cached incremental tokenizers
    pub incremental_size: usize,
    /// Maximum cache size
    pub max_size: usize,
    /// Cache utilization (0.0 - 1.0)
    pub utilization: f32,
}

/// Global tokenizer cache instance
static GLOBAL_CACHE: once_cell::sync::Lazy<TokenizerCache> = once_cell::sync::Lazy::new(|| {
    let cache_size = std::env::var("FERRUM_TOKENIZER_CACHE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    info!(
        "Initializing global tokenizer cache with size: {}",
        cache_size
    );
    TokenizerCache::new(cache_size)
});

/// Get global tokenizer cache
pub fn global_cache() -> &'static TokenizerCache {
    &GLOBAL_CACHE
}

/// Cache key builder for consistent caching
pub struct CacheKeyBuilder {
    components: Vec<String>,
}

impl CacheKeyBuilder {
    /// Create new cache key builder
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Add component to key
    pub fn component(mut self, component: &str) -> Self {
        self.components.push(component.to_string());
        self
    }

    /// Add optional component to key
    pub fn optional_component(mut self, component: Option<&str>) -> Self {
        if let Some(comp) = component {
            self.components.push(comp.to_string());
        } else {
            self.components.push("none".to_string());
        }
        self
    }

    /// Build the cache key
    pub fn build(self) -> String {
        self.components.join(":")
    }
}

impl Default for CacheKeyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Cached tokenizer manager for easy access
pub struct TokenizerManager {
    cache: Arc<TokenizerCache>,
    factory: Arc<dyn ferrum_interfaces::TokenizerFactory + Send + Sync>,
}

impl TokenizerManager {
    /// Create new tokenizer manager
    pub fn new(
        cache: Arc<TokenizerCache>,
        factory: Arc<dyn ferrum_interfaces::TokenizerFactory + Send + Sync>,
    ) -> Self {
        Self { cache, factory }
    }

    /// Get tokenizer for model
    pub async fn get_tokenizer(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<Arc<dyn Tokenizer + Send + Sync>> {
        let cache_key = CacheKeyBuilder::new()
            .component("tokenizer")
            .component(repo_id)
            .optional_component(revision)
            .build();

        self.cache
            .get_or_create(&cache_key, || async {
                let config = ferrum_interfaces::tokenizer::TokenizerConfig {
                    revision: revision.map(|s| s.to_string()),
                    auth_token: None,
                    cache_dir: None,
                    local_files_only: false,
                    trust_remote_code: false,
                };

                self.factory
                    .create_from_pretrained(repo_id, Some(&config))
                    .await
            })
            .await
    }

    /// Get incremental tokenizer for model
    pub async fn get_incremental_tokenizer(
        &self,
        repo_id: &str,
        revision: Option<&str>,
    ) -> Result<Arc<dyn IncrementalTokenizer + Send + Sync>> {
        let cache_key = CacheKeyBuilder::new()
            .component("incremental_tokenizer")
            .component(repo_id)
            .optional_component(revision)
            .build();

        self.cache
            .get_or_create_incremental(&cache_key, || async {
                let config = ferrum_interfaces::tokenizer::TokenizerConfig {
                    revision: revision.map(|s| s.to_string()),
                    auth_token: None,
                    cache_dir: None,
                    local_files_only: false,
                    trust_remote_code: false,
                };

                self.factory
                    .create_incremental_from_pretrained(repo_id, Some(&config))
                    .await
            })
            .await
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_builder() {
        let key = CacheKeyBuilder::new()
            .component("model")
            .component("gpt2")
            .optional_component(Some("main"))
            .build();

        assert_eq!(key, "model:gpt2:main");

        let key2 = CacheKeyBuilder::new()
            .component("model")
            .component("gpt2")
            .optional_component(None)
            .build();

        assert_eq!(key2, "model:gpt2:none");
    }

    #[test]
    fn test_cache_stats() {
        let cache = TokenizerCache::new(10);
        let stats = cache.stats();

        assert_eq!(stats.size, 0);
        assert_eq!(stats.max_size, 10);
        assert_eq!(stats.utilization, 0.0);
    }
}
