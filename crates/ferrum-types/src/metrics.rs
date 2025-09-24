//! Metrics and observability types

use crate::{ids::*, RequestId};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Engine status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    /// Whether the engine is ready to accept requests
    pub is_ready: bool,
    /// Currently loaded models
    pub loaded_models: Vec<ModelId>,
    /// Number of active requests
    pub active_requests: usize,
    /// Number of queued requests
    pub queued_requests: usize,
    /// Current memory usage
    pub memory_usage: MemoryUsage,
    /// Engine uptime
    pub uptime_seconds: u64,
    /// Last heartbeat timestamp
    pub last_heartbeat: DateTime<Utc>,
    /// Engine version
    pub version: String,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Total system memory in bytes
    pub total_bytes: usize,
    /// Used memory in bytes
    pub used_bytes: usize,
    /// Free memory in bytes
    pub free_bytes: usize,
    /// GPU memory usage (if applicable)
    pub gpu_memory_bytes: Option<usize>,
    /// CPU memory usage
    pub cpu_memory_bytes: Option<usize>,
    /// Cache memory usage
    pub cache_memory_bytes: usize,
    /// Memory utilization percentage
    pub utilization_percent: f32,
}

impl MemoryUsage {
    /// Calculate memory utilization percentage
    pub fn calculate_utilization(&mut self) {
        if self.total_bytes > 0 {
            self.utilization_percent = (self.used_bytes as f32 / self.total_bytes as f32) * 100.0;
        } else {
            self.utilization_percent = 0.0;
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Number of waiting requests
    pub waiting_requests: usize,
    /// Number of running requests
    pub running_requests: usize,
    /// Number of preempted requests
    pub preempted_requests: usize,
    /// Total completed requests
    pub completed_requests: u64,
    /// Total failed requests
    pub failed_requests: u64,
    /// Total cancelled requests
    pub cancelled_requests: u64,
    /// Average wait time in milliseconds
    pub avg_wait_time_ms: f64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Current throughput (requests per second)
    pub throughput_rps: f64,
    /// Queue utilization percentage
    pub queue_utilization: f32,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of cache blocks
    pub total_blocks: usize,
    /// Number of used blocks
    pub used_blocks: usize,
    /// Number of free blocks
    pub free_blocks: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f32,
    /// Total number of cache hits
    pub cache_hits: u64,
    /// Total number of cache misses
    pub cache_misses: u64,
    /// Number of cache evictions
    pub eviction_count: u64,
    /// Average block utilization
    pub avg_block_utilization: f32,
    /// Prefix cache statistics
    pub prefix_cache_stats: Option<PrefixCacheStats>,
}

/// Prefix cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    /// Number of cached prefixes
    pub cached_prefixes: usize,
    /// Prefix hit rate
    pub prefix_hit_rate: f32,
    /// Average prefix length
    pub avg_prefix_length: f32,
    /// Memory saved by prefix caching (bytes)
    pub memory_saved_bytes: u64,
}

/// Batch processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingMetrics {
    /// Average batch size
    pub avg_batch_size: f32,
    /// Batch utilization rate
    pub batch_utilization: f32,
    /// Number of batches created
    pub batches_created: u64,
    /// Number of batches completed
    pub batches_completed: u64,
    /// Average batch processing time
    pub avg_batch_time_ms: f64,
    /// Tokens per second across all batches
    pub tokens_per_second: f64,
}

/// Request latency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Average end-to-end latency
    pub avg_latency_ms: f64,
    /// P50 latency
    pub p50_latency_ms: f64,
    /// P90 latency
    pub p90_latency_ms: f64,
    /// P95 latency
    pub p95_latency_ms: f64,
    /// P99 latency
    pub p99_latency_ms: f64,
    /// P99.9 latency
    pub p999_latency_ms: f64,
    /// Time to first token (TTFT)
    pub avg_ttft_ms: f64,
    /// Inter-token latency
    pub avg_inter_token_latency_ms: f64,
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Requests per second
    pub requests_per_second: f64,
    /// Tokens per second
    pub tokens_per_second: f64,
    /// Characters per second
    pub characters_per_second: f64,
    /// Batch throughput
    pub batches_per_second: f64,
    /// Peak throughput achieved
    pub peak_tokens_per_second: f64,
}

/// Model execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Model identifier
    pub model_id: ModelId,
    /// Number of forward passes
    pub forward_passes: u64,
    /// Average forward pass time
    pub avg_forward_time_ms: f64,
    /// Prefill metrics
    pub prefill_metrics: PhaseMetrics,
    /// Decode metrics
    pub decode_metrics: PhaseMetrics,
    /// Total tokens generated
    pub tokens_generated: u64,
}

/// Metrics for different execution phases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    /// Number of operations
    pub operations: u64,
    /// Average time per operation
    pub avg_time_ms: f64,
    /// Total time spent
    pub total_time_ms: f64,
    /// Tokens processed
    pub tokens_processed: u64,
}

/// Request-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    /// Request identifier
    pub request_id: RequestId,
    /// Client identifier
    pub client_id: Option<ClientId>,
    /// Model used
    pub model_id: ModelId,
    /// Request creation time
    pub created_at: DateTime<Utc>,
    /// Request completion time
    pub completed_at: Option<DateTime<Utc>>,
    /// Total processing time
    pub total_time_ms: u64,
    /// Time waiting in queue
    pub queue_time_ms: u64,
    /// Time spent in prefill phase
    pub prefill_time_ms: u64,
    /// Time spent in decode phase
    pub decode_time_ms: u64,
    /// Number of input tokens
    pub input_tokens: usize,
    /// Number of output tokens
    pub output_tokens: usize,
    /// Whether request was preempted
    pub was_preempted: bool,
    /// Number of preemptions
    pub preemption_count: u32,
}

/// System resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage_percent: f32,
    /// Memory usage
    pub memory_usage: MemoryUsage,
    /// GPU utilization (if applicable)
    pub gpu_utilization: Option<GpuMetrics>,
    /// Network I/O statistics
    pub network_io: NetworkMetrics,
    /// Disk I/O statistics
    pub disk_io: DiskMetrics,
    /// System load average
    pub load_average: [f32; 3], // 1min, 5min, 15min
}

/// GPU-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// GPU utilization percentage
    pub utilization_percent: f32,
    /// GPU memory usage in bytes
    pub memory_used_bytes: usize,
    /// GPU memory total in bytes
    pub memory_total_bytes: usize,
    /// GPU temperature in Celsius
    pub temperature_celsius: f32,
    /// Power consumption in watts
    pub power_watts: f32,
}

/// Network I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Bytes received per second
    pub rx_bytes_per_sec: u64,
    /// Bytes transmitted per second
    pub tx_bytes_per_sec: u64,
    /// Packets received per second
    pub rx_packets_per_sec: u64,
    /// Packets transmitted per second
    pub tx_packets_per_sec: u64,
}

/// Disk I/O metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMetrics {
    /// Bytes read per second
    pub read_bytes_per_sec: u64,
    /// Bytes written per second
    pub write_bytes_per_sec: u64,
    /// Read operations per second
    pub read_ops_per_sec: u64,
    /// Write operations per second
    pub write_ops_per_sec: u64,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    /// Total number of errors
    pub total_errors: u64,
    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,
    /// Error rate (errors per request)
    pub error_rate: f32,
    /// Recent errors
    pub recent_errors: Vec<ErrorEvent>,
}

/// Individual error event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    /// When the error occurred
    pub timestamp: DateTime<Utc>,
    /// Error type/category
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Request ID that caused the error (if applicable)
    pub request_id: Option<RequestId>,
    /// Additional context
    pub context: HashMap<String, serde_json::Value>,
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall health status
    pub status: HealthStatusType,
    /// Individual component health
    pub components: HashMap<String, ComponentHealth>,
    /// Last health check time
    pub last_check: DateTime<Utc>,
}

/// Health status types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HealthStatusType {
    /// System is healthy
    Healthy,
    /// System has warnings but is functional
    Warning,
    /// System is unhealthy
    Unhealthy,
}

/// Individual component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component health status
    pub status: HealthStatusType,
    /// Health message
    pub message: String,
    /// Component-specific metrics
    pub metrics: HashMap<String, f64>,
    /// Last check time
    pub last_check: DateTime<Utc>,
}
