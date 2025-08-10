//! KV caching system for the LLM inference engine
//!
//! This module provides a high-performance key-value caching system optimized
//! for LLM inference with support for different eviction policies and memory management.

use crate::config::CacheConfig;
use crate::error::{EngineError, Result};
use crate::models::{CacheEntry, CacheStats, InferenceCache};
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// LRU (Least Recently Used) cache implementation
pub struct LRUCache {
    config: CacheConfig,
    entries: RwLock<HashMap<String, CacheNode>>,
    access_order: RwLock<VecDeque<String>>,
    stats: RwLock<CacheStatsInternal>,
}

/// Internal cache node
#[derive(Debug, Clone)]
struct CacheNode {
    entry: CacheEntry,
    size_bytes: usize,
}

/// Internal cache statistics with additional tracking
#[derive(Debug, Default)]
struct CacheStatsInternal {
    total_requests: u64,
    cache_hits: u64,
    cache_misses: u64,
    evictions: u64,
    total_size_bytes: usize,
}

/// Cache factory for creating different cache types
pub struct CacheFactory;

impl CacheFactory {
    /// Create a cache based on configuration
    pub fn create_cache(config: CacheConfig) -> Result<Box<dyn InferenceCache>> {
        match config.eviction_policy.as_str() {
            "lru" => Ok(Box::new(LRUCache::new(config))),
            "lfu" => Ok(Box::new(LFUCache::new(config))),
            "fifo" => Ok(Box::new(FIFOCache::new(config))),
            _ => Err(EngineError::cache(format!(
                "Unsupported eviction policy: {}",
                config.eviction_policy
            ))),
        }
    }
}

impl LRUCache {
    /// Create a new LRU cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            access_order: RwLock::new(VecDeque::new()),
            stats: RwLock::new(CacheStatsInternal::default()),
        }
    }

    /// Evict entries to make room for new ones
    fn evict_if_needed(&self, new_entry_size: usize) {
        let max_size_bytes = self.config.max_size_mb * 1024 * 1024;
        let mut stats = self.stats.write();

        while stats.total_size_bytes + new_entry_size > max_size_bytes {
            if let Some(oldest_key) = self.access_order.write().pop_front() {
                if let Some(node) = self.entries.write().remove(&oldest_key) {
                    stats.total_size_bytes -= node.size_bytes;
                    stats.evictions += 1;
                } else {
                    break; // Should not happen, but prevents infinite loop
                }
            } else {
                break; // No more entries to evict
            }
        }
    }

    /// Estimate the size of a cache entry in bytes
    fn estimate_entry_size(entry: &CacheEntry) -> usize {
        // Rough estimation: tensor size + metadata
        let tensor_size = entry.key_cache.len() * 1024 + entry.value_cache.len() * 1024; // Simplified
        tensor_size + std::mem::size_of::<CacheEntry>()
    }

    /// Clean up expired entries based on TTL
    fn cleanup_expired(&self) {
        if let Some(ttl) = self.config.ttl_seconds {
            let ttl_duration = Duration::from_secs(ttl);
            let now = Instant::now();
            let mut expired_keys = Vec::new();

            // Find expired entries
            {
                let entries = self.entries.read();
                for (key, node) in entries.iter() {
                    if now.duration_since(node.entry.last_accessed) > ttl_duration {
                        expired_keys.push(key.clone());
                    }
                }
            }

            // Remove expired entries
            if !expired_keys.is_empty() {
                let mut entries = self.entries.write();
                let mut access_order = self.access_order.write();
                let mut stats = self.stats.write();

                for key in expired_keys {
                    if let Some(node) = entries.remove(&key) {
                        stats.total_size_bytes -= node.size_bytes;
                        stats.evictions += 1;

                        // Remove from access order
                        if let Some(pos) = access_order.iter().position(|x| x == &key) {
                            access_order.remove(pos);
                        }
                    }
                }
            }
        }
    }

    /// Update access order for LRU tracking
    fn update_access_order(&self, key: &str) {
        let mut access_order = self.access_order.write();

        // Remove from current position if exists
        if let Some(pos) = access_order.iter().position(|x| x == key) {
            access_order.remove(pos);
        }

        // Add to back (most recently used)
        access_order.push_back(key.to_string());
    }
}

impl InferenceCache for LRUCache {
    fn get_cache(&self, sequence_id: &str) -> Option<CacheEntry> {
        self.cleanup_expired();

        let mut stats = self.stats.write();
        stats.total_requests += 1;

        let entry = {
            let entries = self.entries.read();
            entries.get(sequence_id).map(|node| {
                let mut entry = node.entry.clone();
                entry.touch();
                entry
            })
        };

        if let Some(mut entry) = entry {
            // Update the entry's last accessed time
            {
                let mut entries = self.entries.write();
                if let Some(node) = entries.get_mut(sequence_id) {
                    node.entry.last_accessed = entry.last_accessed;
                }
            }

            self.update_access_order(sequence_id);
            stats.cache_hits += 1;
            Some(entry)
        } else {
            stats.cache_misses += 1;
            None
        }
    }

    fn store_cache(&mut self, sequence_id: &str, cache: CacheEntry) {
        let entry_size = Self::estimate_entry_size(&cache);

        // Check if we need to evict entries
        self.evict_if_needed(entry_size);

        // Check max sequences limit
        if self.entries.read().len() >= self.config.max_sequences {
            if let Some(oldest_key) = self.access_order.write().pop_front() {
                if let Some(node) = self.entries.write().remove(&oldest_key) {
                    let mut stats = self.stats.write();
                    stats.total_size_bytes -= node.size_bytes;
                    stats.evictions += 1;
                }
            }
        }

        // Store the new entry
        let node = CacheNode {
            entry: cache,
            size_bytes: entry_size,
        };

        {
            let mut entries = self.entries.write();
            entries.insert(sequence_id.to_string(), node);
        }

        {
            let mut stats = self.stats.write();
            stats.total_size_bytes += entry_size;
        }

        self.update_access_order(sequence_id);
    }

    fn remove_cache(&mut self, sequence_id: &str) {
        if let Some(node) = self.entries.write().remove(sequence_id) {
            let mut stats = self.stats.write();
            stats.total_size_bytes -= node.size_bytes;

            // Remove from access order
            let mut access_order = self.access_order.write();
            if let Some(pos) = access_order.iter().position(|x| x == sequence_id) {
                access_order.remove(pos);
            }
        }
    }

    fn clear_cache(&mut self) {
        self.entries.write().clear();
        self.access_order.write().clear();
        let mut stats = self.stats.write();
        stats.total_size_bytes = 0;
    }

    fn cache_stats(&self) -> CacheStats {
        let stats = self.stats.read();
        let hit_rate = if stats.total_requests > 0 {
            stats.cache_hits as f32 / stats.total_requests as f32
        } else {
            0.0
        };
        let miss_rate = 1.0 - hit_rate;

        CacheStats {
            total_entries: self.entries.read().len(),
            total_size_bytes: stats.total_size_bytes,
            hit_rate,
            miss_rate,
            eviction_count: stats.evictions,
        }
    }
}

/// LFU (Least Frequently Used) cache implementation
pub struct LFUCache {
    config: CacheConfig,
    entries: RwLock<HashMap<String, CacheNode>>,
    frequency: RwLock<HashMap<String, u64>>,
    stats: RwLock<CacheStatsInternal>,
}

impl LFUCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            frequency: RwLock::new(HashMap::new()),
            stats: RwLock::new(CacheStatsInternal::default()),
        }
    }

    fn find_least_frequent_key(&self) -> Option<String> {
        let frequency = self.frequency.read();
        frequency
            .iter()
            .min_by_key(|(_, &freq)| freq)
            .map(|(key, _)| key.clone())
    }
}

impl InferenceCache for LFUCache {
    fn get_cache(&self, sequence_id: &str) -> Option<CacheEntry> {
        let mut stats = self.stats.write();
        stats.total_requests += 1;

        let entry = {
            let entries = self.entries.read();
            entries.get(sequence_id).map(|node| {
                let mut entry = node.entry.clone();
                entry.touch();
                entry
            })
        };

        if let Some(entry) = entry {
            // Increment frequency
            {
                let mut frequency = self.frequency.write();
                *frequency.entry(sequence_id.to_string()).or_insert(0) += 1;
            }

            stats.cache_hits += 1;
            Some(entry)
        } else {
            stats.cache_misses += 1;
            None
        }
    }

    fn store_cache(&mut self, sequence_id: &str, cache: CacheEntry) {
        let entry_size = LRUCache::estimate_entry_size(&cache);
        let max_size_bytes = self.config.max_size_mb * 1024 * 1024;

        // Evict least frequent entries if needed
        while self.stats.read().total_size_bytes + entry_size > max_size_bytes {
            if let Some(lf_key) = self.find_least_frequent_key() {
                if let Some(node) = self.entries.write().remove(&lf_key) {
                    self.frequency.write().remove(&lf_key);
                    let mut stats = self.stats.write();
                    stats.total_size_bytes -= node.size_bytes;
                    stats.evictions += 1;
                }
            } else {
                break;
            }
        }

        let node = CacheNode {
            entry: cache,
            size_bytes: entry_size,
        };

        self.entries.write().insert(sequence_id.to_string(), node);
        self.frequency.write().insert(sequence_id.to_string(), 1);
        self.stats.write().total_size_bytes += entry_size;
    }

    fn remove_cache(&mut self, sequence_id: &str) {
        if let Some(node) = self.entries.write().remove(sequence_id) {
            self.frequency.write().remove(sequence_id);
            self.stats.write().total_size_bytes -= node.size_bytes;
        }
    }

    fn clear_cache(&mut self) {
        self.entries.write().clear();
        self.frequency.write().clear();
        self.stats.write().total_size_bytes = 0;
    }

    fn cache_stats(&self) -> CacheStats {
        let stats = self.stats.read();
        let hit_rate = if stats.total_requests > 0 {
            stats.cache_hits as f32 / stats.total_requests as f32
        } else {
            0.0
        };

        CacheStats {
            total_entries: self.entries.read().len(),
            total_size_bytes: stats.total_size_bytes,
            hit_rate,
            miss_rate: 1.0 - hit_rate,
            eviction_count: stats.evictions,
        }
    }
}

/// FIFO (First In, First Out) cache implementation
pub struct FIFOCache {
    config: CacheConfig,
    entries: RwLock<HashMap<String, CacheNode>>,
    insertion_order: RwLock<VecDeque<String>>,
    stats: RwLock<CacheStatsInternal>,
}

impl FIFOCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(HashMap::new()),
            insertion_order: RwLock::new(VecDeque::new()),
            stats: RwLock::new(CacheStatsInternal::default()),
        }
    }
}

impl InferenceCache for FIFOCache {
    fn get_cache(&self, sequence_id: &str) -> Option<CacheEntry> {
        let mut stats = self.stats.write();
        stats.total_requests += 1;

        let entry = {
            let entries = self.entries.read();
            entries.get(sequence_id).map(|node| {
                let mut entry = node.entry.clone();
                entry.touch();
                entry
            })
        };

        if entry.is_some() {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }

        entry
    }

    fn store_cache(&mut self, sequence_id: &str, cache: CacheEntry) {
        let entry_size = LRUCache::estimate_entry_size(&cache);
        let max_size_bytes = self.config.max_size_mb * 1024 * 1024;

        // Evict oldest entries if needed
        while self.stats.read().total_size_bytes + entry_size > max_size_bytes {
            if let Some(oldest_key) = self.insertion_order.write().pop_front() {
                if let Some(node) = self.entries.write().remove(&oldest_key) {
                    let mut stats = self.stats.write();
                    stats.total_size_bytes -= node.size_bytes;
                    stats.evictions += 1;
                }
            } else {
                break;
            }
        }

        let node = CacheNode {
            entry: cache,
            size_bytes: entry_size,
        };

        self.entries.write().insert(sequence_id.to_string(), node);
        self.insertion_order
            .write()
            .push_back(sequence_id.to_string());
        self.stats.write().total_size_bytes += entry_size;
    }

    fn remove_cache(&mut self, sequence_id: &str) {
        if let Some(node) = self.entries.write().remove(sequence_id) {
            self.stats.write().total_size_bytes -= node.size_bytes;

            let mut insertion_order = self.insertion_order.write();
            if let Some(pos) = insertion_order.iter().position(|x| x == sequence_id) {
                insertion_order.remove(pos);
            }
        }
    }

    fn clear_cache(&mut self) {
        self.entries.write().clear();
        self.insertion_order.write().clear();
        self.stats.write().total_size_bytes = 0;
    }

    fn cache_stats(&self) -> CacheStats {
        let stats = self.stats.read();
        let hit_rate = if stats.total_requests > 0 {
            stats.cache_hits as f32 / stats.total_requests as f32
        } else {
            0.0
        };

        CacheStats {
            total_entries: self.entries.read().len(),
            total_size_bytes: stats.total_size_bytes,
            hit_rate,
            miss_rate: 1.0 - hit_rate,
            eviction_count: stats.evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CacheConfig;
    use candle_core::{Device, Tensor};

    fn create_test_config() -> CacheConfig {
        CacheConfig {
            enabled: true,
            max_size_mb: 10,
            eviction_policy: "lru".to_string(),
            max_sequences: 5,
            ttl_seconds: Some(3600),
        }
    }

    fn create_test_cache_entry() -> CacheEntry {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((2, 4), candle_core::DType::F32, &device).unwrap();
        CacheEntry::new(vec![tensor.clone()], vec![tensor], 10)
    }

    #[test]
    fn test_lru_cache_basic_operations() {
        let config = create_test_config();
        let mut cache = LRUCache::new(config);

        let entry = create_test_cache_entry();
        cache.store_cache("test_key", entry);

        assert!(cache.get_cache("test_key").is_some());
        assert!(cache.get_cache("nonexistent").is_none());

        cache.remove_cache("test_key");
        assert!(cache.get_cache("test_key").is_none());
    }

    #[test]
    fn test_cache_stats() {
        let config = create_test_config();
        let mut cache = LRUCache::new(config);

        let entry = create_test_cache_entry();
        cache.store_cache("test_key", entry);

        cache.get_cache("test_key"); // Hit
        cache.get_cache("nonexistent"); // Miss

        let stats = cache.cache_stats();
        assert_eq!(stats.total_entries, 1);
        assert!(stats.hit_rate > 0.0);
        assert!(stats.miss_rate > 0.0);
    }

    #[test]
    fn test_cache_factory() {
        let config = create_test_config();
        let cache = CacheFactory::create_cache(config);
        assert!(cache.is_ok());

        let invalid_config = CacheConfig {
            eviction_policy: "invalid".to_string(),
            ..create_test_config()
        };
        let cache = CacheFactory::create_cache(invalid_config);
        assert!(cache.is_err());
    }
}
