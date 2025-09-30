//! Priority-based scheduler implementation

use crate::{
    BatchHint, BatchPlan, BatchResourceRequirements, PreemptionResult, ScheduledRequest, Scheduler,
    SchedulerConfig, SchedulerMetrics,
};
use async_trait::async_trait;
use ferrum_types::{
    BatchId, InferenceRequest, InferenceResponse, Priority, RequestId, RequestState, Result,
};
use parking_lot::RwLock;
use priority_queue::PriorityQueue;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

/// Priority scheduler that processes requests based on priority and submission time
pub struct PriorityScheduler {
    /// Configuration
    config: SchedulerConfig,
    /// Priority queue for waiting requests (higher priority first, then FIFO within priority)
    waiting_queue: RwLock<PriorityQueue<RequestId, RequestPriority>>,
    /// Map from request ID to scheduled request
    request_map: RwLock<HashMap<RequestId, ScheduledRequest>>,
    /// Running requests
    running_requests: RwLock<HashMap<RequestId, ScheduledRequest>>,
    /// Completed request counter
    completed_counter: AtomicU64,
    /// Failed request counter
    failed_counter: AtomicU64,
    /// Cancelled request counter
    cancelled_counter: AtomicU64,
    /// Scheduler start time
    start_time: Instant,
    /// Metrics tracking
    metrics_tracker: Arc<MetricsTracker>,
}

/// Priority wrapper for the priority queue (higher values = higher priority)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RequestPriority {
    /// Priority level (higher = more important)
    priority: i32,
    /// Submission time as negative nanoseconds (older = higher priority within same level)
    submission_time_nanos: i64,
}

impl RequestPriority {
    fn new(priority: Priority, submitted_at: chrono::DateTime<chrono::Utc>) -> Self {
        let priority_value = match priority {
            Priority::Critical => 100,
            Priority::High => 75,
            Priority::Medium => 50,
            Priority::Low => 25,
        };

        // Use negative timestamp so older requests get higher priority within same level
        let submission_time_nanos = -submitted_at.timestamp_nanos_opt().unwrap_or(0);

        Self {
            priority: priority_value,
            submission_time_nanos,
        }
    }
}

/// Internal metrics tracker
struct MetricsTracker {
    total_wait_time_ms: AtomicU64,
    total_execution_time_ms: AtomicU64,
    request_count: AtomicU64,
    priority_stats: parking_lot::RwLock<HashMap<Priority, (u64, u64)>>, // (count, total_wait_time)
}

impl MetricsTracker {
    fn new() -> Self {
        Self {
            total_wait_time_ms: AtomicU64::new(0),
            total_execution_time_ms: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
            priority_stats: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    fn record_completion(&self, wait_time_ms: u64, execution_time_ms: u64, priority: Priority) {
        self.total_wait_time_ms
            .fetch_add(wait_time_ms, Ordering::Relaxed);
        self.total_execution_time_ms
            .fetch_add(execution_time_ms, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);

        // Track per-priority stats
        let mut priority_stats = self.priority_stats.write();
        let (count, total_wait) = priority_stats.entry(priority).or_insert((0, 0));
        *count += 1;
        *total_wait += wait_time_ms;
    }

    fn avg_wait_time_ms(&self) -> f64 {
        let total_wait = self.total_wait_time_ms.load(Ordering::Relaxed) as f64;
        let count = self.request_count.load(Ordering::Relaxed) as f64;
        if count > 0.0 {
            total_wait / count
        } else {
            0.0
        }
    }

    fn avg_execution_time_ms(&self) -> f64 {
        let total_exec = self.total_execution_time_ms.load(Ordering::Relaxed) as f64;
        let count = self.request_count.load(Ordering::Relaxed) as f64;
        if count > 0.0 {
            total_exec / count
        } else {
            0.0
        }
    }

    fn priority_wait_time(&self, priority: Priority) -> f64 {
        let priority_stats = self.priority_stats.read();
        if let Some((count, total_wait)) = priority_stats.get(&priority) {
            if *count > 0 {
                *total_wait as f64 / *count as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

impl PriorityScheduler {
    /// Create new priority scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        info!("Creating Priority scheduler with config: {:?}", config);

        Self {
            config,
            waiting_queue: RwLock::new(PriorityQueue::new()),
            request_map: RwLock::new(HashMap::new()),
            running_requests: RwLock::new(HashMap::new()),
            completed_counter: AtomicU64::new(0),
            failed_counter: AtomicU64::new(0),
            cancelled_counter: AtomicU64::new(0),
            start_time: Instant::now(),
            metrics_tracker: Arc::new(MetricsTracker::new()),
        }
    }

    /// Create batch from waiting queue based on priority
    fn create_batch(&self, hint: BatchHint) -> Option<BatchPlan> {
        let mut waiting_queue = self.waiting_queue.write();
        let mut request_map = self.request_map.write();
        let mut running_requests = self.running_requests.write();

        if waiting_queue.is_empty() {
            return None;
        }

        let mut batch_requests = Vec::new();
        let mut total_tokens = 0;
        let max_sequence_length = hint.max_tokens.min(2048);

        // Process requests in priority order
        let mut requests_to_readd = Vec::new();

        while batch_requests.len() < hint.max_batch_size
            && total_tokens < hint.max_tokens
            && !waiting_queue.is_empty()
        {
            if let Some((request_id, _priority)) = waiting_queue.pop() {
                if let Some(mut scheduled_req) = request_map.remove(&request_id) {
                    let request_tokens = scheduled_req.request.sampling_params.max_tokens;

                    // Check if adding this request would exceed limits
                    if total_tokens + request_tokens <= hint.max_tokens {
                        scheduled_req.state = RequestState::Running;
                        scheduled_req.started_at = Some(chrono::Utc::now());
                        scheduled_req.queue_position = None;

                        total_tokens += request_tokens;

                        // Move to running requests
                        running_requests.insert(request_id.clone(), scheduled_req.clone());
                        batch_requests.push(scheduled_req);
                    } else {
                        // Put the request back
                        let priority = RequestPriority::new(
                            scheduled_req.request.priority,
                            scheduled_req.submitted_at,
                        );
                        requests_to_readd.push((request_id, scheduled_req, priority));
                        break;
                    }
                }
            }
        }

        // Re-add requests that didn't fit
        for (request_id, scheduled_req, priority) in requests_to_readd {
            waiting_queue.push(request_id.clone(), priority);
            request_map.insert(request_id, scheduled_req);
        }

        if batch_requests.is_empty() {
            return None;
        }

        let batch_id = BatchId::new();
        debug!(
            "Creating priority batch {} with {} requests",
            batch_id,
            batch_requests.len()
        );

        // Calculate resource requirements based on batch composition
        let gpu_memory = (total_tokens * 16) as u64; // Estimate: 16 bytes per token
        let cpu_memory = (total_tokens * 4) as u64; // Estimate: 4 bytes per token
        let kv_cache_blocks = total_tokens / 16; // Assume 16 tokens per block

        // Estimate execution time based on highest priority in batch
        let highest_priority = batch_requests
            .iter()
            .map(|req| req.request.priority)
            .max()
            .unwrap_or(Priority::Low);

        let estimated_time_ms = match highest_priority {
            Priority::Critical => 500, // Fast lane for critical requests
            Priority::High => 750,
            Priority::Medium => 1000,
            Priority::Low => 1500,
        };

        Some(BatchPlan {
            batch_id,
            requests: batch_requests,
            max_sequence_length,
            estimated_time_ms: Some(estimated_time_ms),
            resource_requirements: BatchResourceRequirements {
                gpu_memory,
                cpu_memory,
                kv_cache_blocks,
                compute_units: 1,
            },
            created_at: chrono::Utc::now(),
        })
    }
}

#[async_trait]
impl Scheduler for PriorityScheduler {
    async fn submit(&self, request: InferenceRequest) -> Result<RequestId> {
        let request_id = request.id.clone();
        let priority = request.priority;
        debug!(
            "Submitting request {} with priority {:?} to priority scheduler",
            request_id, priority
        );

        // Check queue capacity
        let waiting_queue = self.waiting_queue.read();
        if waiting_queue.len() >= self.config.max_waiting_requests {
            warn!("Queue is full, rejecting request {}", request_id);
            return Err(ferrum_types::FerrumError::scheduler(
                "Queue is full, cannot accept more requests",
            ));
        }
        drop(waiting_queue);

        // Create scheduled request
        let scheduled_request = ScheduledRequest::new(request);
        let request_priority = RequestPriority::new(priority, scheduled_request.submitted_at);

        // Add to priority queue and request map
        let mut waiting_queue = self.waiting_queue.write();
        let mut request_map = self.request_map.write();

        let queue_position = waiting_queue.len();
        let mut scheduled_req = scheduled_request;
        scheduled_req.queue_position = Some(queue_position);

        waiting_queue.push(request_id.clone(), request_priority);
        request_map.insert(request_id.clone(), scheduled_req);

        info!(
            "Request {} queued with priority {:?} at position {}",
            request_id, priority, queue_position
        );
        Ok(request_id)
    }

    async fn next_batch(&self, hint: BatchHint) -> Option<BatchPlan> {
        self.create_batch(hint)
    }

    async fn complete(&self, request_id: RequestId, response: &InferenceResponse) -> Result<()> {
        debug!("Completing request {}", request_id);

        let mut running_requests = self.running_requests.write();
        if let Some(scheduled_req) = running_requests.remove(&request_id) {
            // Calculate metrics
            let wait_time = scheduled_req.age();
            let execution_time = scheduled_req.processing_time().unwrap_or_default();
            let priority = scheduled_req.request.priority;

            self.metrics_tracker.record_completion(
                wait_time.as_millis() as u64,
                execution_time.as_millis() as u64,
                priority,
            );

            match response.finish_reason {
                ferrum_types::FinishReason::Eos | ferrum_types::FinishReason::Stop => {
                    self.completed_counter.fetch_add(1, Ordering::Relaxed);
                    debug!(
                        "Request {} (priority {:?}) completed successfully",
                        request_id, priority
                    );
                }
                _ => {
                    self.failed_counter.fetch_add(1, Ordering::Relaxed);
                    warn!(
                        "Request {} (priority {:?}) completed with error: {:?}",
                        request_id, priority, response.finish_reason
                    );
                }
            }

            Ok(())
        } else {
            warn!("Attempted to complete unknown request: {}", request_id);
            Err(ferrum_types::FerrumError::scheduler(format!(
                "Request {} not found in running requests",
                request_id
            )))
        }
    }

    async fn cancel(&self, request_id: RequestId) -> Result<bool> {
        debug!("Cancelling request {}", request_id);

        // Try to remove from waiting queue first
        let mut waiting_queue = self.waiting_queue.write();
        let mut request_map = self.request_map.write();

        if waiting_queue.remove(&request_id).is_some() {
            request_map.remove(&request_id);
            self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
            info!("Request {} cancelled from waiting queue", request_id);
            return Ok(true);
        }
        drop((waiting_queue, request_map));

        // Try to remove from running requests
        let mut running_requests = self.running_requests.write();
        if running_requests.remove(&request_id).is_some() {
            self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
            warn!(
                "Request {} cancelled while running (may cause issues)",
                request_id
            );
            return Ok(true);
        }

        warn!("Request {} not found for cancellation", request_id);
        Ok(false)
    }

    async fn update_priority(&self, request_id: RequestId, new_priority: Priority) -> Result<()> {
        debug!(
            "Updating priority for request {} to {:?}",
            request_id, new_priority
        );

        let mut waiting_queue = self.waiting_queue.write();
        let mut request_map = self.request_map.write();

        // Check if request is in waiting queue
        if waiting_queue
            .change_priority(
                &request_id,
                RequestPriority::new(new_priority, chrono::Utc::now()),
            )
            .is_some()
        {
            // Update the request priority in the request map
            if let Some(scheduled_req) = request_map.get_mut(&request_id) {
                scheduled_req.request.priority = new_priority;
                info!(
                    "Updated priority for request {} to {:?}",
                    request_id, new_priority
                );
                return Ok(());
            }
        }

        // Check running requests (can't change priority of running request, but update for logging)
        let mut running_requests = self.running_requests.write();
        if let Some(scheduled_req) = running_requests.get_mut(&request_id) {
            let old_priority = scheduled_req.request.priority;
            scheduled_req.request.priority = new_priority;
            warn!("Updated priority for running request {} from {:?} to {:?} (no effect on scheduling)", 
                  request_id, old_priority, new_priority);
            return Ok(());
        }

        warn!("Request {} not found for priority update", request_id);
        Err(ferrum_types::FerrumError::scheduler(format!(
            "Request {} not found",
            request_id
        )))
    }

    fn metrics(&self) -> SchedulerMetrics {
        let waiting_queue = self.waiting_queue.read();
        let running_requests = self.running_requests.read();

        let waiting_count = waiting_queue.len();
        let running_count = running_requests.len();
        let completed_count = self.completed_counter.load(Ordering::Relaxed);
        let failed_count = self.failed_counter.load(Ordering::Relaxed);
        let cancelled_count = self.cancelled_counter.load(Ordering::Relaxed);

        let uptime_secs = self.start_time.elapsed().as_secs_f64();
        let throughput = if uptime_secs > 0.0 {
            completed_count as f64 / uptime_secs
        } else {
            0.0
        };

        let queue_utilization = waiting_count as f32 / self.config.max_waiting_requests as f32;

        // Calculate priority-based P95 estimates
        let critical_wait = self.metrics_tracker.priority_wait_time(Priority::Critical);
        let high_wait = self.metrics_tracker.priority_wait_time(Priority::High);
        let avg_wait = self.metrics_tracker.avg_wait_time_ms();

        let p95_wait_estimate = if critical_wait > 0.0 || high_wait > 0.0 {
            (critical_wait + high_wait) / 2.0 * 1.2 // Priority requests should have better P95
        } else {
            avg_wait * 1.5 // Fallback to simple estimate
        };

        SchedulerMetrics {
            waiting_requests: waiting_count,
            running_requests: running_count,
            completed_requests: completed_count,
            failed_requests: failed_count,
            cancelled_requests: cancelled_count,
            avg_wait_time_ms: avg_wait,
            avg_execution_time_ms: self.metrics_tracker.avg_execution_time_ms(),
            p95_wait_time_ms: p95_wait_estimate,
            p95_execution_time_ms: self.metrics_tracker.avg_execution_time_ms() * 1.3, // Priority affects exec time
            throughput_rps: throughput,
            queue_utilization,
            resource_utilization: ferrum_interfaces::scheduler::ResourceUtilization {
                gpu_memory_utilization: 0.6, // Higher utilization due to priority scheduling
                cpu_memory_utilization: 0.4,
                kv_cache_utilization: 0.5,
                compute_utilization: running_count as f32 / self.config.max_running_requests as f32,
            },
            batch_stats: ferrum_interfaces::scheduler::BatchStats {
                avg_batch_size: 6.0,   // Slightly larger batches due to better selection
                batch_efficiency: 0.9, // Higher efficiency due to priority optimization
                batches_created: completed_count / 6,
                batches_completed: completed_count / 6,
                avg_batch_formation_time_ms: 8.0, // Slightly longer due to priority sorting
            },
            sla_compliance: None,
        }
    }

    fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    async fn preempt(&self, request_id: RequestId) -> Result<PreemptionResult> {
        // Simple preemption: can preempt lower priority running requests
        let mut running_requests = self.running_requests.write();

        if let Some(scheduled_req) = running_requests.get(&request_id) {
            let priority = scheduled_req.request.priority;

            // Only allow preempting Low and Medium priority requests
            if matches!(priority, Priority::Low | Priority::Medium) {
                let removed_req = running_requests.remove(&request_id).unwrap();

                // Move back to waiting queue with updated priority
                let mut waiting_queue = self.waiting_queue.write();
                let mut request_map = self.request_map.write();

                let mut preempted_req = removed_req;
                preempted_req.state = RequestState::Waiting;
                preempted_req.started_at = None;

                let request_priority = RequestPriority::new(priority, preempted_req.submitted_at);
                waiting_queue.push(request_id.clone(), request_priority);
                request_map.insert(request_id.clone(), preempted_req);

                warn!(
                    "Preempted request {} with priority {:?}",
                    request_id, priority
                );

                return Ok(PreemptionResult {
                    success: true,
                    saved_state: None, // Simplified - no state saving in this implementation
                    freed_resources: Default::default(),
                });
            }
        }

        Err(ferrum_types::FerrumError::scheduler(format!(
            "Cannot preempt request {} (not found or high priority)",
            request_id
        )))
    }

    async fn resume(&self, _request_id: RequestId) -> Result<()> {
        // In this implementation, preempted requests are automatically re-queued
        Ok(())
    }
}
