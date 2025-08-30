//! Type definitions for scheduling and batching
//!
//! This module defines the core types used throughout the scheduler system.

use chrono::{DateTime, Utc};
use ferrum_core::{BatchId, InferenceRequest, ModelId, Priority};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Scheduling policy types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    /// First In First Out
    FIFO,

    /// Priority-based scheduling
    Priority,

    /// Fair scheduling (prevent starvation)
    Fair,

    /// Shortest Job First
    SJF,

    /// Round Robin
    RoundRobin,

    /// Weighted Fair Queuing
    WFQ,

    /// Custom policy with name
    Custom(String),
}

/// Batching strategy types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BatchingStrategy {
    /// Static batching with fixed size
    Static { batch_size: usize },

    /// Dynamic batching based on load
    Dynamic {
        min_batch_size: usize,
        max_batch_size: usize,
        timeout_ms: u64,
    },

    /// Continuous batching (vLLM style)
    Continuous {
        max_batch_size: usize,
        max_waiting_time_ms: u64,
    },

    /// Adaptive batching based on model and load
    Adaptive {
        target_latency_ms: u64,
        utilization_target: f32,
    },

    /// Custom strategy
    Custom(String),
}

/// Preemption policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PreemptionPolicy {
    /// No preemption
    None,

    /// Preempt based on priority
    Priority,

    /// Preempt based on SLA deadlines
    Deadline,

    /// Preempt long-running requests
    TimeLimit { max_time_ms: u64 },

    /// Custom preemption logic
    Custom(String),
}

/// Batch request containing multiple inference requests
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Batch identifier
    pub batch_id: BatchId,

    /// Requests in this batch
    pub requests: Vec<InferenceRequest>,

    /// Model ID for this batch
    pub model_id: ModelId,

    /// Batch creation time
    pub created_at: DateTime<Utc>,

    /// Maximum sequence length in batch
    pub max_sequence_length: usize,

    /// Total tokens in batch
    pub total_tokens: usize,

    /// Estimated execution time
    pub estimated_time_ms: u64,

    /// Batch priority (derived from requests)
    pub priority: Priority,
}

/// SLA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaConfig {
    /// Maximum response time in milliseconds
    pub max_response_time_ms: u64,

    /// Maximum queue wait time in milliseconds
    pub max_queue_time_ms: u64,

    /// Throughput guarantee (requests per second)
    pub min_throughput: Option<f32>,

    /// Availability percentage (0.0 - 1.0)
    pub availability: f32,

    /// Penalty for SLA violation
    pub violation_penalty: f32,
}

/// Queue configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// Maximum queue size
    pub max_size: usize,

    /// Queue timeout in milliseconds
    pub timeout_ms: u64,

    /// Drop policy when queue is full
    pub drop_policy: DropPolicy,

    /// Priority levels
    pub priority_levels: Vec<Priority>,

    /// Fair scheduling parameters
    pub fairness_config: Option<FairnessConfig>,
}

/// Drop policy when queue is full
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DropPolicy {
    /// Drop newest requests
    DropNewest,

    /// Drop oldest requests
    DropOldest,

    /// Drop lowest priority requests
    DropLowestPriority,

    /// Reject new requests
    RejectNew,
}

/// Fairness configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessConfig {
    /// Fairness algorithm
    pub algorithm: FairnessAlgorithm,

    /// Time window for fairness calculation
    pub time_window_ms: u64,

    /// Minimum share per client
    pub min_share: f32,

    /// Maximum share per client
    pub max_share: f32,
}

/// Fairness algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FairnessAlgorithm {
    /// Round Robin
    RoundRobin,

    /// Weighted Fair Queuing
    WFQ,

    /// Deficit Round Robin
    DRR,

    /// Stochastic Fair Queuing
    SFQ,
}

/// Queue statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueueStats {
    /// Total requests in queue
    pub total_requests: usize,

    /// Requests by priority
    pub requests_by_priority: HashMap<Priority, usize>,

    /// Average wait time in milliseconds
    pub avg_wait_time_ms: f64,

    /// Maximum wait time in milliseconds
    pub max_wait_time_ms: u64,

    /// Queue utilization (0.0 - 1.0)
    pub utilization: f32,

    /// Throughput (requests per second)
    pub throughput: f32,
}

/// Scheduler metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerMetrics {
    /// Total requests processed
    pub total_requests: u64,

    /// Total requests completed
    pub completed_requests: u64,

    /// Total requests failed
    pub failed_requests: u64,

    /// Total requests cancelled
    pub cancelled_requests: u64,

    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,

    /// 95th percentile response time
    pub p95_response_time_ms: f64,

    /// 99th percentile response time
    pub p99_response_time_ms: f64,

    /// Current queue length
    pub current_queue_length: usize,

    /// Average queue length
    pub avg_queue_length: f64,

    /// Scheduler efficiency (0.0 - 1.0)
    pub efficiency: f32,
}

/// Load information for load balancing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadInfo {
    /// Current CPU utilization (0.0 - 1.0)
    pub cpu_utilization: f32,

    /// Current memory utilization (0.0 - 1.0)
    pub memory_utilization: f32,

    /// Current GPU utilization (0.0 - 1.0)
    pub gpu_utilization: f32,

    /// Current queue length
    pub queue_length: usize,

    /// Current active requests
    pub active_requests: usize,

    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,

    /// Error rate (0.0 - 1.0)
    pub error_rate: f32,
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round Robin
    RoundRobin,

    /// Least Loaded
    LeastLoaded,

    /// Weighted Round Robin
    WeightedRoundRobin { weights: Vec<f32> },

    /// Consistent Hashing
    ConsistentHashing,

    /// Random selection
    Random,

    /// Custom strategy
    Custom(String),
}

/// Preemption result
#[derive(Debug, Clone)]
pub struct PreemptionResult {
    /// Whether preemption was successful
    pub success: bool,

    /// Saved state for resumption
    pub saved_state: Option<Vec<u8>>,

    /// Reason for preemption
    pub reason: PreemptionReason,

    /// Estimated resumption time
    pub estimated_resume_time: Option<Duration>,
}

/// Reasons for preemption
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreemptionReason {
    /// Higher priority request arrived
    HigherPriority,

    /// SLA deadline approaching
    SlaDeadline,

    /// Resource contention
    ResourceContention,

    /// Request timeout
    Timeout,

    /// Manual preemption
    Manual,

    /// System shutdown
    Shutdown,
}

/// Preemption statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PreemptionStats {
    /// Total preemptions
    pub total_preemptions: u64,

    /// Successful preemptions
    pub successful_preemptions: u64,

    /// Failed preemptions
    pub failed_preemptions: u64,

    /// Average preemption time in milliseconds
    pub avg_preemption_time_ms: f64,

    /// Average resumption time in milliseconds
    pub avg_resumption_time_ms: f64,

    /// Preemptions by reason
    pub preemptions_by_reason: HashMap<PreemptionReason, u64>,
}

/// SLA status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlaStatus {
    /// SLA is being met
    Compliant,

    /// SLA is at risk
    AtRisk,

    /// SLA has been violated
    Violated,

    /// No SLA configured
    NotConfigured,
}

/// SLA metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SlaMetrics {
    /// Total requests with SLA
    pub total_sla_requests: u64,

    /// SLA compliant requests
    pub compliant_requests: u64,

    /// SLA violated requests
    pub violated_requests: u64,

    /// Average SLA compliance time
    pub avg_compliance_time_ms: f64,

    /// SLA compliance rate (0.0 - 1.0)
    pub compliance_rate: f32,
}

/// Resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    /// Available GPU memory in bytes
    pub gpu_memory: usize,

    /// Available CPU memory in bytes
    pub cpu_memory: usize,

    /// Available GPU compute units
    pub gpu_compute: f32,

    /// Available CPU cores
    pub cpu_cores: f32,

    /// Network bandwidth
    pub network_bandwidth: Option<f32>,
}

/// Resource requirement for a request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    /// Required GPU memory in bytes
    pub gpu_memory: usize,

    /// Required CPU memory in bytes
    pub cpu_memory: usize,

    /// Required GPU compute units
    pub gpu_compute: f32,

    /// Required CPU cores
    pub cpu_cores: f32,

    /// Estimated execution time in milliseconds
    pub estimated_time_ms: u64,
}

/// Batching metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchingMetrics {
    /// Total batches created
    pub total_batches: u64,

    /// Average batch size
    pub avg_batch_size: f64,

    /// Maximum batch size
    pub max_batch_size: usize,

    /// Minimum batch size
    pub min_batch_size: usize,

    /// Batch utilization (0.0 - 1.0)
    pub batch_utilization: f32,

    /// Time spent waiting for batching
    pub avg_batch_wait_time_ms: f64,
}

/// Admission decision
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionDecision {
    /// Admit the request
    Admit,

    /// Reject the request
    Reject { reason: String },

    /// Defer the request
    Defer { retry_after_ms: u64 },
}

/// Admission policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionPolicy {
    /// Maximum queue size
    pub max_queue_size: usize,

    /// Maximum concurrent requests
    pub max_concurrent: usize,

    /// Rate limit per client
    pub rate_limit: Option<RateLimitConfig>,

    /// Resource-based admission
    pub resource_based: bool,

    /// Priority-based admission
    pub priority_based: bool,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per second
    pub requests_per_second: f32,

    /// Burst capacity
    pub burst_size: u32,

    /// Refill rate
    pub refill_rate: f32,

    /// Time window in seconds
    pub window_seconds: u64,
}

/// Rate limit status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    /// Available tokens
    pub available_tokens: u32,

    /// Next refill time
    pub next_refill_ms: u64,

    /// Rate limit exceeded
    pub exceeded: bool,

    /// Reset time if exceeded
    pub reset_time_ms: Option<u64>,
}

/// Admission statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AdmissionStats {
    /// Total admission requests
    pub total_requests: u64,

    /// Admitted requests
    pub admitted_requests: u64,

    /// Rejected requests
    pub rejected_requests: u64,

    /// Deferred requests
    pub deferred_requests: u64,

    /// Admission rate (0.0 - 1.0)
    pub admission_rate: f32,

    /// Average queue size at admission
    pub avg_queue_size_at_admission: f64,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduling policy
    pub policy: SchedulingPolicy,

    /// Batching strategy
    pub batching: BatchingStrategy,

    /// Preemption policy
    pub preemption: PreemptionPolicy,

    /// Queue configuration
    pub queue: QueueConfig,

    /// Enable SLA management
    pub enable_sla: bool,

    /// Enable resource awareness
    pub enable_resource_awareness: bool,

    /// Enable admission control
    pub enable_admission_control: bool,

    /// Metrics collection interval
    pub metrics_interval_ms: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Fair,
            batching: BatchingStrategy::Continuous {
                max_batch_size: 256,
                max_waiting_time_ms: 100,
            },
            preemption: PreemptionPolicy::Priority,
            queue: QueueConfig::default(),
            enable_sla: true,
            enable_resource_awareness: true,
            enable_admission_control: true,
            metrics_interval_ms: 1000,
        }
    }
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_size: 10000,
            timeout_ms: 30000,
            drop_policy: DropPolicy::DropOldest,
            priority_levels: vec![
                Priority::Low,
                Priority::Normal,
                Priority::High,
                Priority::Critical,
            ],
            fairness_config: Some(FairnessConfig {
                algorithm: FairnessAlgorithm::WFQ,
                time_window_ms: 10000,
                min_share: 0.1,
                max_share: 0.7,
            }),
        }
    }
}

impl Default for SlaConfig {
    fn default() -> Self {
        Self {
            max_response_time_ms: 5000,
            max_queue_time_ms: 1000,
            min_throughput: None,
            availability: 0.999,
            violation_penalty: 1.0,
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 100.0,
            burst_size: 200,
            refill_rate: 100.0,
            window_seconds: 1,
        }
    }
}
