//! Memory statistics and monitoring

use ferrum_types::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Detailed memory statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMemoryStats {
    /// Device this stats belong to
    pub device: Device,
    /// Total memory allocated (lifetime)
    pub total_allocated_bytes: u64,
    /// Total memory deallocated (lifetime)  
    pub total_deallocated_bytes: u64,
    /// Current memory usage
    pub current_usage_bytes: usize,
    /// Peak memory usage
    pub peak_usage_bytes: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Number of allocation requests
    pub allocation_count: u64,
    /// Number of deallocation requests
    pub deallocation_count: u64,
    /// Number of allocation failures
    pub allocation_failures: u64,
    /// Average allocation size
    pub avg_allocation_size: f64,
    /// Fragmentation ratio (0.0 - 1.0)
    pub fragmentation_ratio: f32,
    /// Time since last reset
    pub uptime: Duration,
    /// Allocation size histogram
    pub size_histogram: HashMap<String, u64>,
}

/// Memory statistics tracker
pub struct MemoryStatsTracker {
    device: Device,
    total_allocated: AtomicU64,
    total_deallocated: AtomicU64,
    current_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    active_allocations: AtomicUsize,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
    allocation_failures: AtomicU64,
    start_time: Instant,
    size_buckets: parking_lot::Mutex<[u64; 16]>, // Size histogram buckets
}

impl MemoryStatsTracker {
    /// Create new memory stats tracker
    pub fn new(device: Device) -> Self {
        Self {
            device,
            total_allocated: AtomicU64::new(0),
            total_deallocated: AtomicU64::new(0),
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            active_allocations: AtomicUsize::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
            allocation_failures: AtomicU64::new(0),
            start_time: Instant::now(),
            size_buckets: parking_lot::Mutex::new([0; 16]),
        }
    }

    /// Record allocation
    pub fn record_allocation(&self, size: usize) {
        self.total_allocated.fetch_add(size as u64, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.active_allocations.fetch_add(1, Ordering::Relaxed);
        
        let new_usage = self.current_usage.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak if necessary
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while new_usage > peak {
            match self.peak_usage.compare_exchange_weak(
                peak, 
                new_usage, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(current_peak) => peak = current_peak,
            }
        }
        
        // Update size histogram
        self.update_size_histogram(size);
    }

    /// Record deallocation
    pub fn record_deallocation(&self, size: usize) {
        self.total_deallocated.fetch_add(size as u64, Ordering::Relaxed);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
        self.active_allocations.fetch_sub(1, Ordering::Relaxed);
        self.current_usage.fetch_sub(size, Ordering::Relaxed);
    }

    /// Record allocation failure
    pub fn record_allocation_failure(&self) {
        self.allocation_failures.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current statistics
    pub fn stats(&self) -> DetailedMemoryStats {
        let total_allocated = self.total_allocated.load(Ordering::Relaxed);
        let allocation_count = self.allocation_count.load(Ordering::Relaxed);
        
        let avg_allocation_size = if allocation_count > 0 {
            total_allocated as f64 / allocation_count as f64
        } else {
            0.0
        };

        // Build size histogram
        let buckets = self.size_buckets.lock();
        let mut size_histogram = HashMap::new();
        let bucket_labels = [
            "0-1KB", "1KB-4KB", "4KB-16KB", "16KB-64KB", 
            "64KB-256KB", "256KB-1MB", "1MB-4MB", "4MB-16MB",
            "16MB-64MB", "64MB-256MB", "256MB-1GB", "1GB-4GB",
            "4GB-16GB", "16GB-64GB", "64GB+", "Other"
        ];
        
        for (i, &count) in buckets.iter().enumerate() {
            if count > 0 {
                size_histogram.insert(bucket_labels[i].to_string(), count);
            }
        }

        DetailedMemoryStats {
            device: self.device,
            total_allocated_bytes: total_allocated,
            total_deallocated_bytes: self.total_deallocated.load(Ordering::Relaxed),
            current_usage_bytes: self.current_usage.load(Ordering::Relaxed),
            peak_usage_bytes: self.peak_usage.load(Ordering::Relaxed),
            active_allocations: self.active_allocations.load(Ordering::Relaxed),
            allocation_count,
            deallocation_count: self.deallocation_count.load(Ordering::Relaxed),
            allocation_failures: self.allocation_failures.load(Ordering::Relaxed),
            avg_allocation_size,
            fragmentation_ratio: self.calculate_fragmentation_ratio(),
            uptime: self.start_time.elapsed(),
            size_histogram,
        }
    }

    /// Reset statistics
    pub fn reset(&self) {
        self.total_allocated.store(0, Ordering::Relaxed);
        self.total_deallocated.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);
        self.allocation_failures.store(0, Ordering::Relaxed);
        
        let mut buckets = self.size_buckets.lock();
        buckets.fill(0);
    }

    fn update_size_histogram(&self, size: usize) {
        let bucket_index = match size {
            0..=1024 => 0,              // 0-1KB
            1025..=4096 => 1,           // 1KB-4KB
            4097..=16384 => 2,          // 4KB-16KB
            16385..=65536 => 3,         // 16KB-64KB
            65537..=262144 => 4,        // 64KB-256KB
            262145..=1048576 => 5,      // 256KB-1MB
            1048577..=4194304 => 6,     // 1MB-4MB
            4194305..=16777216 => 7,    // 4MB-16MB
            16777217..=67108864 => 8,   // 16MB-64MB
            67108865..=268435456 => 9,  // 64MB-256MB
            268435457..=1073741824 => 10, // 256MB-1GB
            1073741825..=4294967296 => 11, // 1GB-4GB
            4294967297..=17179869184 => 12, // 4GB-16GB
            17179869185..=68719476736 => 13, // 16GB-64GB
            68719476737.. => 14,        // 64GB+
            _ => 15,                    // Other
        };

        let mut buckets = self.size_buckets.lock();
        buckets[bucket_index] += 1;
    }

    fn calculate_fragmentation_ratio(&self) -> f32 {
        // Simplified fragmentation calculation
        // In a real implementation, this would analyze actual memory layout
        let total_allocated = self.total_allocated.load(Ordering::Relaxed);
        let total_deallocated = self.total_deallocated.load(Ordering::Relaxed);
        let current_usage = self.current_usage.load(Ordering::Relaxed);
        
        if total_allocated == 0 {
            return 0.0;
        }
        
        // Fragmentation increases with the ratio of deallocated to allocated memory
        let dealloc_ratio = total_deallocated as f32 / total_allocated as f32;
        let usage_ratio = current_usage as f32 / total_allocated as f32;
        
        // Simple heuristic: fragmentation is higher when we've deallocated a lot
        // but still have high current usage
        (dealloc_ratio * (1.0 - usage_ratio)).min(1.0)
    }
}

/// Global memory statistics registry
pub struct GlobalMemoryStatsRegistry {
    trackers: parking_lot::RwLock<HashMap<Device, MemoryStatsTracker>>,
}

impl GlobalMemoryStatsRegistry {
    /// Create new global registry
    pub fn new() -> Self {
        Self {
            trackers: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Get or create tracker for device
    pub fn get_or_create_tracker(&self, device: Device) -> parking_lot::RwLockReadGuard<'_, MemoryStatsTracker> {
        // First try to get existing tracker
        {
            let trackers = self.trackers.read();
            if trackers.contains_key(&device) {
                // Can't return the reference directly due to lifetime issues
                // This is a simplified implementation
            }
        }
        
        // Create new tracker if needed
        {
            let mut trackers = self.trackers.write();
            trackers.entry(device).or_insert_with(|| MemoryStatsTracker::new(device));
        }
        
        // Return read guard (simplified)
        unimplemented!("Simplified implementation - would need proper lifetime management")
    }

    /// Get stats for device
    pub fn get_stats(&self, device: Device) -> Option<DetailedMemoryStats> {
        let trackers = self.trackers.read();
        trackers.get(&device).map(|t| t.stats())
    }

    /// Get stats for all devices
    pub fn get_all_stats(&self) -> Vec<DetailedMemoryStats> {
        let trackers = self.trackers.read();
        trackers.values().map(|t| t.stats()).collect()
    }

    /// Reset stats for device
    pub fn reset_stats(&self, device: Device) {
        let trackers = self.trackers.read();
        if let Some(tracker) = trackers.get(&device) {
            tracker.reset();
        }
    }

    /// Reset all stats
    pub fn reset_all_stats(&self) {
        let trackers = self.trackers.read();
        for tracker in trackers.values() {
            tracker.reset();
        }
    }
}

impl Default for GlobalMemoryStatsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Global instance of memory stats registry
use once_cell::sync::Lazy;
static GLOBAL_MEMORY_STATS: Lazy<GlobalMemoryStatsRegistry> = 
    Lazy::new(|| GlobalMemoryStatsRegistry::new());

/// Get global memory stats registry
pub fn global_memory_stats() -> &'static GlobalMemoryStatsRegistry {
    &GLOBAL_MEMORY_STATS
}
