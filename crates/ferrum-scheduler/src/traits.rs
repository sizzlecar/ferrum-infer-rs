//! Core scheduler traits
//!
//! This module defines the abstract interfaces for request scheduling,
//! batching, and resource management in LLM inference.

use async_trait::async_trait;
use ferrum_core::{Result, RequestId, InferenceRequest, BatchId, Priority};
use crate::types::*;
use std::time::Duration;

/// Main scheduler trait for managing inference requests
#[async_trait]
pub trait Scheduler: Send + Sync {
    /// Schedule a new inference request
    async fn schedule(&self, request: InferenceRequest) -> Result<RequestId>;
    
    /// Cancel a scheduled request
    async fn cancel(&self, request_id: RequestId) -> Result<()>;
    
    /// Get next batch of requests to execute
    async fn get_next_batch(&self, max_batch_size: usize) -> Result<Option<BatchRequest>>;
    
    /// Complete a request
    async fn complete_request(&self, request_id: RequestId) -> Result<()>;
    
    /// Update request priority
    async fn update_priority(&self, request_id: RequestId, priority: Priority) -> Result<()>;
    
    /// Get queue statistics
    fn get_queue_stats(&self) -> QueueStats;
    
    /// Get scheduler metrics
    fn get_metrics(&self) -> SchedulerMetrics;
    
    /// Get scheduling policy
    fn get_policy(&self) -> &SchedulingPolicy;
}

/// Batch scheduler for continuous batching
#[async_trait]
pub trait BatchScheduler: Send + Sync {
    /// Create a new batch from pending requests
    async fn create_batch(&self, strategy: &BatchingStrategy) -> Result<Option<BatchRequest>>;
    
    /// Add request to existing batch if possible
    async fn try_add_to_batch(&self, batch_id: BatchId, request: InferenceRequest) -> Result<bool>;
    
    /// Remove request from batch
    async fn remove_from_batch(&self, batch_id: BatchId, request_id: RequestId) -> Result<()>;
    
    /// Split batch if needed
    async fn split_batch(&self, batch_id: BatchId) -> Result<Vec<BatchRequest>>;
    
    /// Merge compatible batches
    async fn merge_batches(&self, batch_ids: &[BatchId]) -> Result<BatchRequest>;
    
    /// Get optimal batch size for current load
    fn get_optimal_batch_size(&self) -> usize;
    
    /// Get batching efficiency metrics
    fn get_batching_metrics(&self) -> BatchingMetrics;
}

/// Request queue management trait
#[async_trait]
pub trait RequestQueue: Send + Sync {
    /// Enqueue a request
    async fn enqueue(&self, request: InferenceRequest) -> Result<()>;
    
    /// Dequeue the next request based on policy
    async fn dequeue(&self) -> Result<Option<InferenceRequest>>;
    
    /// Peek at the next request without removing it
    async fn peek(&self) -> Result<Option<InferenceRequest>>;
    
    /// Remove a specific request
    async fn remove(&self, request_id: RequestId) -> Result<Option<InferenceRequest>>;
    
    /// Get queue length
    fn len(&self) -> usize;
    
    /// Check if queue is empty
    fn is_empty(&self) -> bool;
    
    /// Get requests by priority
    async fn get_by_priority(&self, priority: Priority) -> Vec<InferenceRequest>;
    
    /// Clear all requests
    async fn clear(&self) -> Result<Vec<InferenceRequest>>;
}

/// Load balancer for distributing requests
#[async_trait]
pub trait LoadBalancer: Send + Sync {
    /// Select the best scheduler for a request
    async fn select_scheduler(&self, request: &InferenceRequest) -> Result<SchedulerId>;
    
    /// Report scheduler load
    async fn report_load(&self, scheduler_id: SchedulerId, load: LoadInfo) -> Result<()>;
    
    /// Get load balancing strategy
    fn get_strategy(&self) -> &LoadBalancingStrategy;
    
    /// Get current load distribution
    fn get_load_distribution(&self) -> Vec<(SchedulerId, LoadInfo)>;
}

/// Preemption manager for handling request preemption
#[async_trait]
pub trait PreemptionManager: Send + Sync {
    /// Check if a request should be preempted
    async fn should_preempt(&self, request_id: RequestId) -> Result<bool>;
    
    /// Preempt a request
    async fn preempt(&self, request_id: RequestId) -> Result<PreemptionResult>;
    
    /// Resume a preempted request
    async fn resume(&self, request_id: RequestId) -> Result<()>;
    
    /// Get preemption candidates
    async fn get_preemption_candidates(&self) -> Vec<RequestId>;
    
    /// Set preemption policy
    fn set_policy(&mut self, policy: PreemptionPolicy);
    
    /// Get preemption statistics
    fn get_preemption_stats(&self) -> PreemptionStats;
}

/// SLA (Service Level Agreement) manager
#[async_trait]
pub trait SlaManager: Send + Sync {
    /// Register SLA for a request
    async fn register_sla(&self, request_id: RequestId, sla: SlaConfig) -> Result<()>;
    
    /// Check SLA compliance
    async fn check_compliance(&self, request_id: RequestId) -> SlaStatus;
    
    /// Get requests violating SLA
    async fn get_sla_violations(&self) -> Vec<RequestId>;
    
    /// Update SLA configuration
    async fn update_sla(&self, request_id: RequestId, sla: SlaConfig) -> Result<()>;
    
    /// Get SLA metrics
    fn get_sla_metrics(&self) -> SlaMetrics;
}

/// Resource-aware scheduler
#[async_trait]
pub trait ResourceAwareScheduler: Scheduler {
    /// Schedule based on current resource availability
    async fn schedule_with_resources(
        &self,
        request: InferenceRequest,
        resources: &ResourceInfo,
    ) -> Result<RequestId>;
    
    /// Get resource requirements for a request
    fn estimate_resources(&self, request: &InferenceRequest) -> ResourceRequirement;
    
    /// Check if resources are available for a request
    fn has_resources(&self, requirement: &ResourceRequirement) -> bool;
    
    /// Reserve resources for a request
    async fn reserve_resources(&self, request_id: RequestId, resources: ResourceRequirement) -> Result<()>;
    
    /// Release reserved resources
    async fn release_resources(&self, request_id: RequestId) -> Result<()>;
}

/// Scheduler ID type
pub type SchedulerId = String;

/// Request position in queue
#[derive(Debug, Clone)]
pub struct QueuePosition {
    pub position: usize,
    pub estimated_wait_time: Duration,
    pub ahead_of_you: usize,
}

/// Admission control trait
#[async_trait]
pub trait AdmissionController: Send + Sync {
    /// Decide whether to admit a new request
    async fn should_admit(&self, request: &InferenceRequest) -> Result<AdmissionDecision>;
    
    /// Get current admission rate
    fn get_admission_rate(&self) -> f32;
    
    /// Update admission policy
    fn update_policy(&mut self, policy: AdmissionPolicy);
    
    /// Get admission statistics
    fn get_admission_stats(&self) -> AdmissionStats;
}

/// Rate limiter trait
#[async_trait]
pub trait RateLimiter: Send + Sync {
    /// Check if request is within rate limits
    async fn check_rate(&self, client_id: &str) -> Result<bool>;
    
    /// Consume rate limit tokens
    async fn consume(&self, client_id: &str, tokens: u32) -> Result<()>;
    
    /// Get rate limit status for a client
    async fn get_status(&self, client_id: &str) -> RateLimitStatus;
    
    /// Reset rate limits for a client
    async fn reset(&self, client_id: &str) -> Result<()>;
    
    /// Configure rate limits for a client
    async fn configure(&self, client_id: &str, config: RateLimitConfig) -> Result<()>;
}
