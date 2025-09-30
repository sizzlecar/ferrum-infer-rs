//! Unified scheduler interface with resource awareness and SLA support
//!
//! This module provides the unified scheduler interface that replaces the
//! conflicting scheduler definitions in the original codebase.

use async_trait::async_trait;
use ferrum_types::{
    BatchId, InferenceRequest, InferenceResponse, Priority, RequestId, RequestState, Result,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Main scheduler trait for request management and batching
#[async_trait]
pub trait Scheduler: Send + Sync {
    /// Submit new inference request
    async fn submit(&self, request: InferenceRequest) -> Result<RequestId>;

    /// Get next batch of requests to execute
    async fn next_batch(&self, hint: BatchHint) -> Option<BatchPlan>;

    /// Mark request as completed
    async fn complete(&self, request_id: RequestId, response: &InferenceResponse) -> Result<()>;

    /// Cancel pending request
    async fn cancel(&self, request_id: RequestId) -> Result<bool>;

    /// Update request priority
    async fn update_priority(&self, request_id: RequestId, priority: Priority) -> Result<()>;

    /// Get scheduler metrics
    fn metrics(&self) -> SchedulerMetrics;

    /// Get scheduler configuration
    fn config(&self) -> &SchedulerConfig;

    /// Preempt running request (if supported)
    async fn preempt(&self, request_id: RequestId) -> Result<PreemptionResult> {
        // Default implementation: preemption not supported
        Err(ferrum_types::FerrumError::unsupported(
            "Preemption not supported",
        ))
    }

    /// Resume preempted request
    async fn resume(&self, request_id: RequestId) -> Result<()> {
        // Default implementation: resumption not supported
        Err(ferrum_types::FerrumError::unsupported(
            "Resumption not supported",
        ))
    }
}

/// Batch hint for scheduler optimization
#[derive(Debug, Clone)]
pub struct BatchHint {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum total tokens in batch
    pub max_tokens: usize,
    /// Target latency for batch formation
    pub target_latency_ms: Option<u64>,
    /// Available memory for batch
    pub available_memory: Option<u64>,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
}

impl BatchHint {
    /// Create simple batch hint with size limit
    pub fn simple(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            max_tokens: max_batch_size * 2048, // Default reasonable token limit
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: ResourceConstraints::default(),
        }
    }
}

/// Resource constraints for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum GPU memory usage
    pub max_gpu_memory: Option<u64>,
    /// Maximum CPU memory usage
    pub max_cpu_memory: Option<u64>,
    /// Maximum compute units
    pub max_compute_units: Option<usize>,
    /// Required device types
    pub required_devices: Vec<ferrum_types::Device>,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_gpu_memory: None,
            max_cpu_memory: None,
            max_compute_units: None,
            required_devices: vec![],
        }
    }
}

/// Batch execution plan
#[derive(Debug, Clone)]
pub struct BatchPlan {
    /// Unique batch identifier
    pub batch_id: BatchId,
    /// Requests included in this batch
    pub requests: Vec<ScheduledRequest>,
    /// Maximum sequence length in batch
    pub max_sequence_length: usize,
    /// Estimated execution time
    pub estimated_time_ms: Option<u64>,
    /// Resource requirements
    pub resource_requirements: BatchResourceRequirements,
    /// Batch creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl BatchPlan {
    /// Get total number of tokens in batch
    pub fn total_tokens(&self) -> usize {
        self.requests
            .iter()
            .map(|req| req.request.sampling_params.max_tokens)
            .sum()
    }

    /// Get batch size
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    /// Get highest priority in batch
    pub fn max_priority(&self) -> Priority {
        self.requests
            .iter()
            .map(|req| req.request.priority)
            .max()
            .unwrap_or(Priority::Low)
    }
}

/// Scheduled request with additional metadata
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    /// Original inference request
    pub request: InferenceRequest,
    /// Current scheduling state
    pub state: RequestState,
    /// Queue position when waiting
    pub queue_position: Option<usize>,
    /// Estimated wait time
    pub estimated_wait_time: Option<Duration>,
    /// Number of tokens processed so far
    pub tokens_processed: usize,
    /// Allocated resources
    pub allocated_resources: AllocatedResources,
    /// Request submission time
    pub submitted_at: chrono::DateTime<chrono::Utc>,
    /// Request start time (when moved from waiting to running)
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl ScheduledRequest {
    /// Create new scheduled request
    pub fn new(request: InferenceRequest) -> Self {
        Self {
            request,
            state: RequestState::Waiting,
            queue_position: None,
            estimated_wait_time: None,
            tokens_processed: 0,
            allocated_resources: AllocatedResources::default(),
            submitted_at: chrono::Utc::now(),
            started_at: None,
        }
    }

    /// Get request age since submission
    pub fn age(&self) -> Duration {
        (chrono::Utc::now() - self.submitted_at)
            .to_std()
            .unwrap_or_default()
    }

    /// Get processing time (if started)
    pub fn processing_time(&self) -> Option<Duration> {
        self.started_at
            .map(|start| (chrono::Utc::now() - start).to_std().unwrap_or_default())
    }
}

/// Allocated resources for a request
#[derive(Debug, Clone, Default)]
pub struct AllocatedResources {
    /// KV cache blocks allocated
    pub kv_cache_blocks: Vec<ferrum_types::BlockId>,
    /// GPU memory allocated (bytes)
    pub gpu_memory: u64,
    /// CPU memory allocated (bytes)
    pub cpu_memory: u64,
    /// Compute units reserved
    pub compute_units: usize,
}

/// Resource requirements for batch execution
#[derive(Debug, Clone)]
pub struct BatchResourceRequirements {
    /// Required GPU memory
    pub gpu_memory: u64,
    /// Required CPU memory
    pub cpu_memory: u64,
    /// Required KV cache blocks
    pub kv_cache_blocks: usize,
    /// Required compute units
    pub compute_units: usize,
}

/// Preemption result
#[derive(Debug, Clone)]
pub struct PreemptionResult {
    /// Whether preemption was successful
    pub success: bool,
    /// Saved state for resumption (if any)
    pub saved_state: Option<PreemptionState>,
    /// Resources freed by preemption
    pub freed_resources: AllocatedResources,
}

/// State saved during preemption
#[derive(Debug, Clone)]
pub struct PreemptionState {
    /// KV cache checkpoint
    pub kv_cache_checkpoint: Vec<u8>,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Generation state
    pub generation_state: HashMap<String, serde_json::Value>,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduling policy
    pub policy: SchedulingPolicy,
    /// Maximum waiting requests
    pub max_waiting_requests: usize,
    /// Maximum concurrent requests  
    pub max_running_requests: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch formation timeout
    pub batch_timeout_ms: u64,
    /// Enable preemption
    pub enable_preemption: bool,
    /// Enable load balancing
    pub enable_load_balancing: bool,
    /// Fair share configuration
    pub fair_share_config: Option<FairShareConfig>,
    /// SLA configuration
    pub sla_config: Option<SlaConfig>,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Priority,
            max_waiting_requests: 1000,
            max_running_requests: 100,
            max_batch_size: 32,
            batch_timeout_ms: 10,
            enable_preemption: false,
            enable_load_balancing: false,
            fair_share_config: None,
            sla_config: None,
            resource_limits: ResourceLimits::default(),
        }
    }
}

/// Scheduling policies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SchedulingPolicy {
    /// First-Come-First-Served
    FCFS,
    /// Priority-based scheduling
    Priority,
    /// Fair-share scheduling
    FairShare,
    /// Shortest-Job-First
    SJF,
    /// Resource-aware scheduling
    ResourceAware,
    /// SLA-driven scheduling
    SlaAware,
}

/// Fair share configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairShareConfig {
    /// Share weights per client
    pub client_shares: HashMap<String, f32>,
    /// Default share for unspecified clients
    pub default_share: f32,
    /// Share enforcement strictness (0.0 - 1.0)
    pub enforcement_strictness: f32,
}

/// SLA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaConfig {
    /// Enable SLA enforcement
    pub enabled: bool,
    /// Default SLA for requests without specific SLA
    pub default_sla: SlaRequirements,
    /// Per-client SLA overrides
    pub client_slas: HashMap<String, SlaRequirements>,
}

/// SLA requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaRequirements {
    /// Maximum latency (P95)
    pub max_latency_p95_ms: u64,
    /// Maximum latency (P99)
    pub max_latency_p99_ms: u64,
    /// Minimum throughput
    pub min_throughput_rps: f32,
    /// Availability requirement
    pub availability_percent: f32,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum total GPU memory
    pub max_gpu_memory: Option<u64>,
    /// Maximum total CPU memory
    pub max_cpu_memory: Option<u64>,
    /// Maximum KV cache blocks
    pub max_kv_cache_blocks: Option<usize>,
    /// Per-client resource limits
    pub per_client_limits: HashMap<String, ClientResourceLimits>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_gpu_memory: None,
            max_cpu_memory: None,
            max_kv_cache_blocks: None,
            per_client_limits: HashMap::new(),
        }
    }
}

/// Per-client resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientResourceLimits {
    /// Max concurrent requests per client
    pub max_concurrent_requests: usize,
    /// Max GPU memory per client
    pub max_gpu_memory: Option<u64>,
    /// Max requests per minute
    pub max_requests_per_minute: Option<u32>,
}

/// Scheduler performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerMetrics {
    /// Currently waiting requests
    pub waiting_requests: usize,
    /// Currently running requests
    pub running_requests: usize,
    /// Total completed requests
    pub completed_requests: u64,
    /// Total failed requests
    pub failed_requests: u64,
    /// Total cancelled requests
    pub cancelled_requests: u64,
    /// Average wait time in queue (ms)
    pub avg_wait_time_ms: f64,
    /// Average execution time (ms)
    pub avg_execution_time_ms: f64,
    /// P95 wait time (ms)
    pub p95_wait_time_ms: f64,
    /// P95 execution time (ms)
    pub p95_execution_time_ms: f64,
    /// Current throughput (requests/second)
    pub throughput_rps: f64,
    /// Queue utilization (0.0 - 1.0)
    pub queue_utilization: f32,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Batch statistics
    pub batch_stats: BatchStats,
    /// SLA compliance (if enabled)
    pub sla_compliance: Option<SlaCompliance>,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// GPU memory utilization
    pub gpu_memory_utilization: f32,
    /// CPU memory utilization
    pub cpu_memory_utilization: f32,
    /// KV cache utilization
    pub kv_cache_utilization: f32,
    /// Compute unit utilization
    pub compute_utilization: f32,
}

/// Batch processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchStats {
    /// Average batch size
    pub avg_batch_size: f32,
    /// Batch utilization efficiency
    pub batch_efficiency: f32,
    /// Total batches created
    pub batches_created: u64,
    /// Total batches completed
    pub batches_completed: u64,
    /// Average batch formation time (ms)
    pub avg_batch_formation_time_ms: f64,
}

/// SLA compliance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaCompliance {
    /// Overall SLA compliance rate
    pub overall_compliance_rate: f32,
    /// Per-client compliance rates
    pub client_compliance_rates: HashMap<String, f32>,
    /// Recent SLA violations
    pub recent_violations: Vec<SlaViolation>,
}

/// SLA violation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaViolation {
    /// Request that violated SLA
    pub request_id: RequestId,
    /// Client identifier
    pub client_id: Option<String>,
    /// Violation type
    pub violation_type: SlaViolationType,
    /// Actual vs required metric
    pub actual_value: f64,
    /// Required value
    pub required_value: f64,
    /// Violation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Types of SLA violations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SlaViolationType {
    /// Latency exceeded threshold
    LatencyViolation,
    /// Throughput below threshold
    ThroughputViolation,
    /// Availability violation
    AvailabilityViolation,
}

/// Advanced scheduler capabilities
#[async_trait]
pub trait AdvancedScheduler: Scheduler {
    /// Enable resource-aware scheduling
    async fn enable_resource_awareness(&mut self, config: ResourceAwarenessConfig) -> Result<()>;

    /// Set custom admission policy
    async fn set_admission_policy(&mut self, policy: Box<dyn AdmissionPolicy>) -> Result<()>;

    /// Configure dynamic batching
    async fn configure_dynamic_batching(&mut self, config: DynamicBatchingConfig) -> Result<()>;

    /// Get detailed queue analysis
    fn queue_analysis(&self) -> QueueAnalysis;

    /// Simulate scheduling for capacity planning
    async fn simulate_load(
        &self,
        workload: &SimulatedWorkload,
    ) -> Result<SchedulingSimulationResult>;
}

/// Resource awareness configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAwarenessConfig {
    /// Enable memory-aware scheduling
    pub enable_memory_awareness: bool,
    /// Enable compute-aware scheduling
    pub enable_compute_awareness: bool,
    /// Resource prediction horizon
    pub prediction_horizon_ms: u64,
    /// Resource safety margin (0.0 - 1.0)
    pub safety_margin: f32,
}

/// Admission policy for request acceptance
pub trait AdmissionPolicy: Send + Sync {
    /// Decide whether to admit a request
    fn should_admit(
        &self,
        request: &InferenceRequest,
        current_metrics: &SchedulerMetrics,
    ) -> AdmissionDecision;

    /// Get policy name
    fn name(&self) -> &str;
}

/// Admission decision
#[derive(Debug, Clone)]
pub enum AdmissionDecision {
    /// Accept the request
    Accept,
    /// Reject the request with reason
    Reject(String),
    /// Accept but suggest delay
    AcceptWithDelay(Duration),
}

/// Dynamic batching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchingConfig {
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch formation timeout
    pub batch_timeout_ms: u64,
    /// Enable adaptive batch sizing
    pub enable_adaptive_sizing: bool,
    /// Target batch utilization
    pub target_utilization: f32,
}

/// Queue analysis results
#[derive(Debug, Clone)]
pub struct QueueAnalysis {
    /// Queue depth over time
    pub queue_depth_history: Vec<(chrono::DateTime<chrono::Utc>, usize)>,
    /// Wait time distribution
    pub wait_time_distribution: WaitTimeDistribution,
    /// Request pattern analysis
    pub request_patterns: RequestPatternAnalysis,
    /// Bottleneck identification
    pub bottlenecks: Vec<BottleneckAnalysis>,
}

/// Wait time distribution
#[derive(Debug, Clone)]
pub struct WaitTimeDistribution {
    /// P50 wait time
    pub p50_ms: f64,
    /// P95 wait time
    pub p95_ms: f64,
    /// P99 wait time
    pub p99_ms: f64,
    /// Maximum wait time
    pub max_ms: f64,
    /// Average wait time
    pub mean_ms: f64,
}

/// Request pattern analysis
#[derive(Debug, Clone)]
pub struct RequestPatternAnalysis {
    /// Peak request times
    pub peak_times: Vec<chrono::DateTime<chrono::Utc>>,
    /// Request rate trend
    pub rate_trend: RateTrend,
    /// Seasonality patterns
    pub seasonality: SeasonalityPattern,
}

/// Request rate trend
#[derive(Debug, Clone, Copy)]
pub enum RateTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Seasonality patterns
#[derive(Debug, Clone)]
pub struct SeasonalityPattern {
    /// Hourly patterns
    pub hourly_pattern: Vec<f32>,
    /// Daily patterns  
    pub daily_pattern: Vec<f32>,
    /// Weekly patterns
    pub weekly_pattern: Vec<f32>,
}

/// Bottleneck analysis
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity (0.0 - 1.0)
    pub severity: f32,
    /// Description
    pub description: String,
    /// Suggested mitigation
    pub mitigation: String,
}

/// Types of bottlenecks
#[derive(Debug, Clone, Copy)]
pub enum BottleneckType {
    /// Memory bottleneck
    Memory,
    /// Compute bottleneck
    Compute,
    /// I/O bottleneck
    IO,
    /// Scheduling bottleneck
    Scheduling,
    /// Network bottleneck
    Network,
}

/// Simulated workload for capacity planning
#[derive(Debug, Clone)]
pub struct SimulatedWorkload {
    /// Request arrival pattern
    pub arrival_pattern: ArrivalPattern,
    /// Request size distribution
    pub size_distribution: SizeDistribution,
    /// Simulation duration
    pub duration_seconds: u64,
}

/// Request arrival patterns
#[derive(Debug, Clone)]
pub enum ArrivalPattern {
    /// Constant rate
    Constant { rate_rps: f32 },
    /// Poisson process
    Poisson { lambda: f32 },
    /// Bursty pattern
    Bursty {
        burst_rate: f32,
        quiet_rate: f32,
        burst_duration_s: f32,
    },
    /// Seasonal pattern
    Seasonal {
        base_rate: f32,
        peaks: Vec<(f32, f32)>,
    }, // (time, multiplier)
}

/// Request size distribution
#[derive(Debug, Clone)]
pub enum SizeDistribution {
    /// Fixed size
    Fixed { tokens: usize },
    /// Uniform distribution
    Uniform {
        min_tokens: usize,
        max_tokens: usize,
    },
    /// Normal distribution
    Normal { mean: f32, std_dev: f32 },
    /// Log-normal distribution
    LogNormal { mu: f32, sigma: f32 },
}

/// Scheduling simulation results
#[derive(Debug, Clone)]
pub struct SchedulingSimulationResult {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed/rejected requests
    pub failed_requests: u64,
    /// Average latency
    pub avg_latency_ms: f64,
    /// P95 latency
    pub p95_latency_ms: f64,
    /// P99 latency
    pub p99_latency_ms: f64,
    /// Throughput achieved
    pub throughput_rps: f32,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Predicted bottlenecks
    pub bottlenecks: Vec<BottleneckAnalysis>,
}
