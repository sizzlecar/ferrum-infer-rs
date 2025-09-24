//! Inference engine interface with streaming and batch support
//!
//! This module provides the top-level inference engine interface that
//! orchestrates all other components: tokenizer, model executor, scheduler,
//! and sampler.

use ferrum_types::{
    InferenceRequest,
    InferenceResponse,
    Result,
    StreamChunk,
    EngineConfig,
    EngineStatus as TypesEngineStatus,
    EngineConfig as TypesEngineConfig,
    EngineModelConfig,
    BatchConfig,
    SamplingConfig,
    MonitoringConfig,
};
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, pin::Pin, time::Duration};

/// Core inference engine trait
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Execute single inference request
    async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse>;
    
    /// Execute streaming inference request
    async fn infer_stream(
        &self,
        request: InferenceRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;
    
    /// Get current engine status
    async fn status(&self) -> EngineStatus;
    
    /// Shutdown engine gracefully
    async fn shutdown(&self) -> Result<()>;
    
    /// Get engine configuration
    fn config(&self) -> &EngineConfig;
    
    /// Get engine metrics
    fn metrics(&self) -> EngineMetrics;
    
    /// Health check
    async fn health_check(&self) -> HealthStatus;
}

/// Engine status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    /// Whether engine is ready to accept requests
    pub is_ready: bool,
    /// Currently loaded models
    pub loaded_models: Vec<ferrum_types::ModelId>,
    /// Active requests count
    pub active_requests: usize,
    /// Queued requests count
    pub queued_requests: usize,
    /// Current memory usage
    pub memory_usage: ferrum_types::MemoryUsage,
    /// Engine uptime in seconds
    pub uptime_seconds: u64,
    /// Last heartbeat timestamp
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    /// Engine version
    pub version: String,
    /// Component status
    pub component_status: ComponentStatus,
}

impl From<TypesEngineStatus> for EngineStatus {
    fn from(status: TypesEngineStatus) -> Self {
        Self {
            is_ready: status.is_ready,
            loaded_models: status.loaded_models,
            active_requests: status.active_requests,
            queued_requests: status.queued_requests,
            memory_usage: status.memory_usage,
            uptime_seconds: status.uptime_seconds,
            last_heartbeat: status.last_heartbeat,
            version: status.version,
            component_status: status.component_status.into(),
        }
    }
}

/// Status of engine components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    /// Scheduler status
    pub scheduler: ComponentHealth,
    /// Model executor status
    pub model_executor: ComponentHealth,
    /// Tokenizer status
    pub tokenizer: ComponentHealth,
    /// KV cache manager status
    pub kv_cache: ComponentHealth,
    /// Memory manager status
    pub memory_manager: ComponentHealth,
    /// Backend status
    pub backend: ComponentHealth,
}

impl From<ferrum_types::ComponentStatus> for ComponentStatus {
    fn from(status: ferrum_types::ComponentStatus) -> Self {
        Self {
            scheduler: status.scheduler.into(),
            model_executor: status.model_executor.into(),
            tokenizer: status.tokenizer.into(),
            kv_cache: status.kv_cache.into(),
            memory_manager: status.memory_manager.into(),
            backend: status.backend.into(),
        }
    }
}

/// Individual component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component status
    pub status: ComponentHealthStatus,
    /// Health message
    pub message: String,
    /// Last check time
    pub last_check: chrono::DateTime<chrono::Utc>,
    /// Component-specific metrics
    pub metrics: HashMap<String, f64>,
}

impl ComponentHealth {
    pub fn healthy(component: &str) -> Self {
        Self {
            status: ComponentHealthStatus::Healthy,
            message: format!("{} healthy", component),
            last_check: chrono::Utc::now(),
            metrics: HashMap::new(),
        }
    }
}

impl From<ferrum_types::ComponentHealth> for ComponentHealth {
    fn from(health: ferrum_types::ComponentHealth) -> Self {
        Self {
            status: health.status.into(),
            message: health.message,
            last_check: health.last_check,
            metrics: health.metrics,
        }
    }
}

/// Component health status
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComponentHealthStatus {
    /// Component is healthy
    Healthy,
    /// Component has warnings but is functional
    Warning,
    /// Component is degraded
    Degraded,
    /// Component is unhealthy
    Unhealthy,
}

impl From<ferrum_types::ComponentHealthStatus> for ComponentHealthStatus {
    fn from(status: ferrum_types::ComponentHealthStatus) -> Self {
        match status {
            ferrum_types::ComponentHealthStatus::Healthy => Self::Healthy,
            ferrum_types::ComponentHealthStatus::Warning => Self::Warning,
            ferrum_types::ComponentHealthStatus::Degraded => Self::Degraded,
            ferrum_types::ComponentHealthStatus::Unhealthy => Self::Unhealthy,
        }
    }
}

/// Sampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Default sampling parameters
    pub default_params: ferrum_types::SamplingParams,
    /// Pre-configured sampling presets
    pub presets: HashMap<String, ferrum_types::SamplingParams>,
    /// Enable custom logits processors
    pub enable_custom_processors: bool,
}

/// Memory configuration for engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable memory optimization
    pub enable_optimization: bool,
    /// Memory pool size
    pub pool_size: Option<u64>,
    /// Enable memory monitoring
    pub enable_monitoring: bool,
    /// Memory pressure thresholds
    pub pressure_thresholds: crate::kv_cache::MemoryPressureThresholds,
}

/// Engine performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineMetrics {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average request latency (ms)
    pub avg_request_latency_ms: f64,
    /// P95 request latency (ms)
    pub p95_request_latency_ms: f64,
    /// P99 request latency (ms)
    pub p99_request_latency_ms: f64,
    /// Current throughput (requests/second)
    pub throughput_rps: f32,
    /// Tokens per second
    pub tokens_per_second: f32,
    /// Queue metrics
    pub queue_metrics: QueueMetrics,
    /// Resource utilization
    pub resource_utilization: ResourceMetrics,
    /// Error statistics
    pub error_stats: ErrorStats,
    /// Performance breakdown
    pub performance_breakdown: PerformanceBreakdown,
}

/// Queue-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueMetrics {
    /// Current queue length
    pub current_queue_length: usize,
    /// Average queue wait time (ms)
    pub avg_queue_wait_time_ms: f64,
    /// Queue throughput (requests/second)
    pub queue_throughput_rps: f32,
    /// Queue rejection rate
    pub queue_rejection_rate: f32,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// CPU utilization percentage
    pub cpu_utilization: f32,
    /// Memory utilization percentage
    pub memory_utilization: f32,
    /// GPU utilization percentage (if applicable)
    pub gpu_utilization: Option<f32>,
    /// Network I/O utilization
    pub network_utilization: f32,
    /// Disk I/O utilization
    pub disk_utilization: f32,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    /// Total error count
    pub total_errors: u64,
    /// Error rate (errors per request)
    pub error_rate: f32,
    /// Errors by category
    pub errors_by_category: HashMap<String, u64>,
    /// Recent error events
    pub recent_errors: Vec<ErrorEvent>,
}

/// Error event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    /// Error timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Error category
    pub category: String,
    /// Error message
    pub message: String,
    /// Request ID (if applicable)
    pub request_id: Option<ferrum_types::RequestId>,
    /// Additional context
    pub context: HashMap<String, serde_json::Value>,
}

/// Performance breakdown by phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBreakdown {
    /// Time spent in tokenization (ms)
    pub tokenization_time_ms: f64,
    /// Time spent in model execution (ms)
    pub model_execution_time_ms: f64,
    /// Time spent in sampling (ms)
    pub sampling_time_ms: f64,
    /// Time spent in scheduling (ms)
    pub scheduling_time_ms: f64,
    /// Time spent in memory operations (ms)
    pub memory_operations_time_ms: f64,
    /// Other overhead time (ms)
    pub other_overhead_time_ms: f64,
}

/// Health status for engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall health status
    pub overall_status: OverallHealthStatus,
    /// Individual component status
    pub component_status: ComponentStatus,
    /// Health check timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Health score (0.0 - 1.0)
    pub health_score: f32,
    /// Health issues detected
    pub issues: Vec<HealthIssue>,
}

impl HealthStatus {
    pub fn healthy() -> Self {
        Self {
            overall_status: OverallHealthStatus::Healthy,
            component_status: ComponentStatus {
                scheduler: ComponentHealth::healthy("scheduler"),
                model_executor: ComponentHealth::healthy("model"),
                tokenizer: ComponentHealth::healthy("tokenizer"),
                kv_cache: ComponentHealth::healthy("kv_cache"),
                memory_manager: ComponentHealth::healthy("memory"),
                backend: ComponentHealth::healthy("backend"),
            },
            timestamp: chrono::Utc::now(),
            health_score: 1.0,
            issues: Vec::new(),
        }
    }
}

/// Overall health status
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OverallHealthStatus {
    /// All systems operational
    Healthy,
    /// Minor issues but still functional
    Warning,
    /// Significant issues affecting performance
    Degraded,
    /// Critical issues, service unavailable
    Critical,
}

/// Health issue record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: String,
    /// Issue description
    pub description: String,
    /// Suggested remediation
    pub remediation: Option<String>,
    /// Issue timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Low severity, informational
    Info,
    /// Medium severity, warning
    Warning,
    /// High severity, error
    Error,
    /// Critical severity, service impacting
    Critical,
}

/// Advanced engine capabilities
#[async_trait]
pub trait AdvancedInferenceEngine: InferenceEngine {
    /// Execute batch inference
    async fn infer_batch(
        &self,
        requests: Vec<InferenceRequest>,
    ) -> Result<Vec<Result<InferenceResponse>>>;
    
    /// Execute speculative inference
    async fn infer_speculative(
        &self,
        request: InferenceRequest,
        speculation_config: SpeculationConfig,
    ) -> Result<InferenceResponse>;
    
    /// Warm up engine with sample requests
    async fn warmup(&mut self, warmup_requests: Vec<InferenceRequest>) -> Result<WarmupResult>;
    
    /// Configure engine at runtime
    async fn reconfigure(&mut self, config: EngineConfig) -> Result<()>;
    
    /// Get detailed diagnostics
    async fn diagnostics(&self) -> DiagnosticsReport;
    
    /// Export engine state for debugging
    async fn export_state(&self) -> Result<EngineState>;
    
    /// Import engine state for debugging/testing
    async fn import_state(&mut self, state: EngineState) -> Result<()>;
}

/// Speculation configuration for speculative decoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationConfig {
    /// Number of tokens to speculate ahead
    pub speculation_depth: usize,
    /// Acceptance threshold for speculative tokens
    pub acceptance_threshold: f32,
    /// Draft model configuration (if different)
    pub draft_model_config: Option<ModelConfig>,
}

/// Warmup result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupResult {
    /// Number of warmup requests processed
    pub requests_processed: usize,
    /// Total warmup time (ms)
    pub total_time_ms: u64,
    /// Average latency during warmup (ms)
    pub avg_latency_ms: f64,
    /// Memory allocated during warmup
    pub memory_allocated_bytes: u64,
    /// Whether warmup was successful
    pub success: bool,
    /// Warmup issues (if any)
    pub issues: Vec<String>,
}

/// Diagnostics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsReport {
    /// Engine configuration snapshot
    pub config_snapshot: EngineConfig,
    /// Current metrics
    pub current_metrics: EngineMetrics,
    /// Resource usage details
    pub resource_usage: DetailedResourceUsage,
    /// Performance analysis
    pub performance_analysis: PerformanceAnalysis,
    /// Component diagnostics
    pub component_diagnostics: HashMap<String, serde_json::Value>,
    /// System information
    pub system_info: SystemInfo,
}

/// Detailed resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedResourceUsage {
    /// Memory usage by component
    pub memory_by_component: HashMap<String, u64>,
    /// CPU usage by thread
    pub cpu_by_thread: HashMap<String, f32>,
    /// GPU memory usage details
    pub gpu_memory_details: Option<GpuMemoryDetails>,
    /// Network I/O details
    pub network_io_details: NetworkIODetails,
}

/// GPU memory usage details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryDetails {
    /// Total GPU memory
    pub total_memory: u64,
    /// Used GPU memory
    pub used_memory: u64,
    /// Memory by allocation type
    pub memory_by_type: HashMap<String, u64>,
    /// Large allocations
    pub large_allocations: Vec<AllocationInfo>,
}

/// Memory allocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    /// Allocation size
    pub size: u64,
    /// Allocation type
    pub allocation_type: String,
    /// Allocation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Network I/O details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIODetails {
    /// Bytes received per second
    pub bytes_received_per_sec: u64,
    /// Bytes sent per second
    pub bytes_sent_per_sec: u64,
    /// Connection count
    pub connection_count: usize,
    /// Request rate
    pub request_rate_per_sec: f32,
}

/// Performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysis {
    /// Identified bottlenecks
    pub bottlenecks: Vec<PerformanceBottleneck>,
    /// Performance recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
    /// Performance trends
    pub trends: PerformanceTrends,
}

/// Performance bottleneck identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    /// Bottleneck type
    pub bottleneck_type: String,
    /// Severity (0.0 - 1.0)
    pub severity: f32,
    /// Description
    pub description: String,
    /// Impact on performance
    pub performance_impact: f32,
}

/// Performance optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRecommendation {
    /// Recommendation category
    pub category: String,
    /// Recommendation description
    pub description: String,
    /// Expected impact
    pub expected_impact: f32,
    /// Implementation complexity (0.0 - 1.0)
    pub complexity: f32,
}

/// Performance trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// Latency trend
    pub latency_trend: TrendDirection,
    /// Throughput trend
    pub throughput_trend: TrendDirection,
    /// Error rate trend
    pub error_rate_trend: TrendDirection,
    /// Resource utilization trend
    pub resource_utilization_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Improving performance
    Improving,
    /// Stable performance
    Stable,
    /// Degrading performance
    Degrading,
    /// Volatile/unpredictable
    Volatile,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system
    pub os: String,
    /// CPU information
    pub cpu_info: String,
    /// Total system memory
    pub total_memory: u64,
    /// Available devices
    pub devices: Vec<ferrum_types::Device>,
    /// Runtime information
    pub runtime_info: RuntimeInfo,
}

/// Runtime information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    /// Rust version
    pub rust_version: String,
    /// Engine version
    pub engine_version: String,
    /// Build information
    pub build_info: BuildInfo,
    /// Feature flags enabled
    pub feature_flags: Vec<String>,
}

/// Build information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    /// Build timestamp
    pub build_timestamp: String,
    /// Git commit hash
    pub git_commit: Option<String>,
    /// Build configuration (debug/release)
    pub build_config: String,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
}

/// Engine state for export/import
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineState {
    /// Configuration state
    pub config: EngineConfig,
    /// Metrics snapshot
    pub metrics: EngineMetrics,
    /// Component states
    pub component_states: HashMap<String, serde_json::Value>,
    /// Active requests state
    pub active_requests: Vec<RequestState>,
    /// Export timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Request state for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestState {
    /// Request ID
    pub request_id: ferrum_types::RequestId,
    /// Current phase
    pub current_phase: String,
    /// Progress information
    pub progress: RequestProgress,
    /// Allocated resources
    pub allocated_resources: HashMap<String, serde_json::Value>,
}

/// Request progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestProgress {
    /// Tokens processed
    pub tokens_processed: usize,
    /// Tokens remaining
    pub tokens_remaining: usize,
    /// Elapsed time (ms)
    pub elapsed_time_ms: u64,
    /// Estimated time remaining (ms)
    pub estimated_remaining_ms: Option<u64>,
}

/// Engine factory for creating engine instances
#[async_trait]
pub trait EngineFactory: Send + Sync {
    /// Create standard inference engine
    async fn create_engine(&self, config: EngineConfig) -> Result<Box<dyn InferenceEngine>>;
    
    /// Create advanced inference engine
    async fn create_advanced_engine(
        &self,
        config: EngineConfig,
    ) -> Result<Box<dyn AdvancedInferenceEngine>>;
    
    /// Validate engine configuration
    fn validate_config(&self, config: &EngineConfig) -> Result<Vec<String>>;
    
    /// Get recommended configuration for hardware
    async fn recommend_config(&self, constraints: HardwareConstraints) -> Result<EngineConfig>;
}

/// Hardware constraints for configuration recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConstraints {
    /// Available devices
    pub available_devices: Vec<ferrum_types::Device>,
    /// Total memory available
    pub total_memory: u64,
    /// Expected request rate
    pub expected_request_rate: f32,
    /// Expected request characteristics
    pub request_characteristics: RequestCharacteristics,
}

/// Expected request characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestCharacteristics {
    /// Average input tokens
    pub avg_input_tokens: usize,
    /// Average output tokens  
    pub avg_output_tokens: usize,
    /// Typical batch size
    pub typical_batch_size: usize,
    /// Latency requirements
    pub latency_requirements: LatencyRequirements,
}

/// Latency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRequirements {
    /// Target P95 latency (ms)
    pub target_p95_latency_ms: u64,
    /// Target P99 latency (ms)
    pub target_p99_latency_ms: u64,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: u64,
}
