//! Prefix caching for shared prompt optimization

use ferrum_types::{Result, TokenId, RequestId, FerrumError};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, trace};

/// Prefix identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixId(Vec<TokenId>);

impl PrefixId {
    /// Create new prefix ID from tokens
    pub fn new(tokens: Vec<TokenId>) -> Self {
        Self(tokens)
    }

    /// Get tokens
    pub fn tokens(&self) -> &[TokenId] {
        &self.0
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<Vec<TokenId>> for PrefixId {
    fn from(tokens: Vec<TokenId>) -> Self {
        Self::new(tokens)
    }
}

impl From<&[TokenId]> for PrefixId {
    fn from(tokens: &[TokenId]) -> Self {
        Self::new(tokens.to_vec())
    }
}

/// Cached prefix information
#[derive(Debug, Clone)]
pub struct CachedPrefix {
    /// Prefix tokens
    pub prefix_id: PrefixId,
    /// KV cache handle for this prefix
    pub kv_handle: Arc<dyn ferrum_interfaces::KvCacheHandle + Send + Sync>,
    /// Reference count
    pub ref_count: usize,
    /// Last access time
    pub last_access: std::time::Instant,
    /// Size in tokens
    pub size: usize,
}

impl CachedPrefix {
    /// Create new cached prefix
    pub fn new(
        prefix_id: PrefixId,
        kv_handle: Arc<dyn ferrum_interfaces::KvCacheHandle + Send + Sync>,
    ) -> Self {
        let size = prefix_id.len();
        Self {
            prefix_id,
            kv_handle,
            ref_count: 1,
            last_access: std::time::Instant::now(),
            size,
        }
    }

    /// Add reference
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
        self.touch();
    }

    /// Remove reference
    pub fn remove_ref(&mut self) -> Result<()> {
        if self.ref_count == 0 {
            return Err(FerrumError::invalid_state("Cannot remove ref from zero-ref prefix"));
        }
        self.ref_count -= 1;
        Ok(())
    }

    /// Update access time
    pub fn touch(&mut self) {
        self.last_access = std::time::Instant::now();
    }

    /// Check if can be evicted
    pub fn can_evict(&self) -> bool {
        self.ref_count == 0
    }
}

/// Prefix cache for shared prompt optimization
#[derive(Debug)]
pub struct PrefixCache {
    /// Cached prefixes by prefix ID
    prefixes: RwLock<HashMap<PrefixId, CachedPrefix>>,
    /// Maximum number of cached prefixes
    max_prefixes: usize,
    /// Minimum prefix length to cache
    min_prefix_length: usize,
    /// Statistics
    hits: parking_lot::Mutex<usize>,
    misses: parking_lot::Mutex<usize>,
    evictions: parking_lot::Mutex<usize>,
}

impl PrefixCache {
    /// Create new prefix cache
    pub fn new(max_prefixes: usize, min_prefix_length: usize) -> Self {
        Self {
            prefixes: RwLock::new(HashMap::new()),
            max_prefixes,
            min_prefix_length,
            hits: parking_lot::Mutex::new(0),
            misses: parking_lot::Mutex::new(0),
            evictions: parking_lot::Mutex::new(0),
        }
    }

    /// Find matching prefix for given tokens
    pub fn find_prefix(&self, tokens: &[TokenId]) -> Option<(PrefixId, Arc<dyn ferrum_interfaces::KvCacheHandle + Send + Sync>)> {
        if tokens.len() < self.min_prefix_length {
            return None;
        }

        let prefixes = self.prefixes.read();
        
        // Find longest matching prefix
        let mut best_match = None;
        let mut best_len = 0;

        for (prefix_id, cached_prefix) in prefixes.iter() {
            if tokens.starts_with(prefix_id.tokens()) && prefix_id.len() > best_len {
                best_match = Some((prefix_id.clone(), cached_prefix.kv_handle.clone()));
                best_len = prefix_id.len();
            }
        }

        if let Some(ref match_info) = best_match {
            *self.hits.lock() += 1;
            trace!("Prefix cache hit: {} tokens", best_len);
            
            // Update access time
            drop(prefixes); // Release read lock
            let mut prefixes = self.prefixes.write();
            if let Some(cached_prefix) = prefixes.get_mut(&match_info.0) {
                cached_prefix.touch();
            }
        } else {
            *self.misses.lock() += 1;
            trace!("Prefix cache miss for {} tokens", tokens.len());
        }

        best_match
    }

    /// Store prefix in cache
    pub fn store_prefix(
        &self,
        prefix_tokens: &[TokenId],
        kv_handle: Arc<dyn ferrum_interfaces::KvCacheHandle + Send + Sync>,
    ) -> Result<()> {
        if prefix_tokens.len() < self.min_prefix_length {
            return Ok(()); // Don't cache short prefixes
        }

        let prefix_id = PrefixId::from(prefix_tokens);
        let cached_prefix = CachedPrefix::new(prefix_id.clone(), kv_handle);

        let mut prefixes = self.prefixes.write();
        
        // Check if we need to evict
        if prefixes.len() >= self.max_prefixes && !prefixes.contains_key(&prefix_id) {
            self.evict_lru(&mut prefixes);
        }

        // Store or update prefix
        if let Some(existing) = prefixes.get_mut(&prefix_id) {
            existing.add_ref();
        } else {
            prefixes.insert(prefix_id, cached_prefix);
            debug!("Stored new prefix: {} tokens", prefix_tokens.len());
        }

        Ok(())
    }

    /// Remove reference to prefix
    pub fn remove_ref(&self, prefix_tokens: &[TokenId]) -> Result<()> {
        let prefix_id = PrefixId::from(prefix_tokens);
        let mut prefixes = self.prefixes.write();
        
        if let Some(cached_prefix) = prefixes.get_mut(&prefix_id) {
            cached_prefix.remove_ref()?;
            
            // Remove if no more references
            if cached_prefix.ref_count == 0 {
                prefixes.remove(&prefix_id);
                debug!("Removed unreferenced prefix: {} tokens", prefix_tokens.len());
            }
        }

        Ok(())
    }

    /// Evict least recently used prefix
    fn evict_lru(&self, prefixes: &mut HashMap<PrefixId, CachedPrefix>) {
        let mut oldest_id = None;
        let mut oldest_time = std::time::Instant::now();

        // Find least recently used evictable prefix
        for (prefix_id, cached_prefix) in prefixes.iter() {
            if cached_prefix.can_evict() && cached_prefix.last_access < oldest_time {
                oldest_time = cached_prefix.last_access;
                oldest_id = Some(prefix_id.clone());
            }
        }

        if let Some(prefix_id) = oldest_id {
            prefixes.remove(&prefix_id);
            *self.evictions.lock() += 1;
            debug!("Evicted LRU prefix: {} tokens", prefix_id.len());
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> PrefixCacheStats {
        let prefixes = self.prefixes.read();
        let total_size: usize = prefixes.values().map(|p| p.size).sum();
        let active_prefixes = prefixes.len();
        
        PrefixCacheStats {
            hits: *self.hits.lock(),
            misses: *self.misses.lock(),
            evictions: *self.evictions.lock(),
            active_prefixes,
            total_cached_tokens: total_size,
            hit_rate: {
                let hits = *self.hits.lock();
                let misses = *self.misses.lock();
                if hits + misses > 0 {
                    hits as f32 / (hits + misses) as f32
                } else {
                    0.0
                }
            },
        }
    }

    /// Clear all cached prefixes
    pub fn clear(&self) {
        let mut prefixes = self.prefixes.write();
        prefixes.clear();
        *self.hits.lock() = 0;
        *self.misses.lock() = 0;
        *self.evictions.lock() = 0;
        debug!("Cleared prefix cache");
    }

    /// Get configuration
    pub fn config(&self) -> (usize, usize) {
        (self.max_prefixes, self.min_prefix_length)
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(100, 8) // Default: cache up to 100 prefixes, minimum 8 tokens
    }
}

/// Prefix cache statistics
#[derive(Debug, Clone)]
pub struct PrefixCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub active_prefixes: usize,
    pub total_cached_tokens: usize,
    pub hit_rate: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocks::DefaultKvCacheHandle;

    // Mock KV cache handle for testing
    #[derive(Debug, Clone)]
    struct MockKvHandle {
        tokens: usize,
        device: ferrum_types::Device,
        block_table: ferrum_interfaces::BlockTable,
    }

    impl MockKvHandle {
        fn new(tokens: usize) -> Self {
            Self {
                tokens,
                device: ferrum_types::Device::Cpu,
                block_table: ferrum_interfaces::BlockTable::default(),
            }
        }
    }

    impl ferrum_interfaces::KvCacheHandle for MockKvHandle {
        fn block_table(&self) -> &ferrum_interfaces::BlockTable {
            &self.block_table
        }

        fn device(&self) -> ferrum_types::Device {
            self.device.clone()
        }

        fn num_tokens(&self) -> usize {
            self.tokens
        }
    }

    #[test]
    fn test_prefix_cache_creation() {
        let cache = PrefixCache::new(50, 4);
        let (max_prefixes, min_len) = cache.config();
        assert_eq!(max_prefixes, 50);
        assert_eq!(min_len, 4);
    }

    #[test]
    fn test_prefix_storage_and_retrieval() {
        let cache = PrefixCache::new(10, 2);
        
        let tokens = vec![TokenId::new(1), TokenId::new(2), TokenId::new(3)];
        let handle = Arc::new(MockKvHandle::new(3));
        
        // Store prefix
        cache.store_prefix(&tokens, handle.clone()).unwrap();
        
        // Should find exact match
        let result = cache.find_prefix(&tokens);
        assert!(result.is_some());
        
        // Should find prefix for longer sequence
        let longer_tokens = vec![TokenId::new(1), TokenId::new(2), TokenId::new(3), TokenId::new(4)];
        let result = cache.find_prefix(&longer_tokens);
        assert!(result.is_some());
        let (found_prefix, _) = result.unwrap();
        assert_eq!(found_prefix.tokens(), &tokens);
    }

    #[test]
    fn test_prefix_length_filtering() {
        let cache = PrefixCache::new(10, 5); // Minimum 5 tokens
        
        let short_tokens = vec![TokenId::new(1), TokenId::new(2)]; // Too short
        let handle = Arc::new(MockKvHandle::new(2));
        
        // Should not store short prefix
        cache.store_prefix(&short_tokens, handle).unwrap();
        
        let result = cache.find_prefix(&short_tokens);
        assert!(result.is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let cache = PrefixCache::new(2, 1); // Max 2 prefixes
        
        let tokens1 = vec![TokenId::new(1)];
        let tokens2 = vec![TokenId::new(2)];  
        let tokens3 = vec![TokenId::new(3)];
        
        let handle = Arc::new(MockKvHandle::new(1));
        
        // Store 2 prefixes
        cache.store_prefix(&tokens1, handle.clone()).unwrap();
        cache.store_prefix(&tokens2, handle.clone()).unwrap();
        
        // Access first one to make it more recent
        cache.find_prefix(&tokens1);
        
        // Store third - should evict tokens2 (LRU)
        cache.store_prefix(&tokens3, handle.clone()).unwrap();
        
        // tokens1 and tokens3 should exist, tokens2 should be evicted
        assert!(cache.find_prefix(&tokens1).is_some());
        assert!(cache.find_prefix(&tokens2).is_none());
        assert!(cache.find_prefix(&tokens3).is_some());
    }

    #[test]
    fn test_cache_stats() {
        let cache = PrefixCache::new(10, 2);
        let tokens = vec![TokenId::new(1), TokenId::new(2)];
        let handle = Arc::new(MockKvHandle::new(2));
        
        cache.store_prefix(&tokens, handle).unwrap();
        
        // Hit
        cache.find_prefix(&tokens);
        
        // Miss
        let other_tokens = vec![TokenId::new(3), TokenId::new(4)];
        cache.find_prefix(&other_tokens);
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
        assert_eq!(stats.active_prefixes, 1);
    }
}
