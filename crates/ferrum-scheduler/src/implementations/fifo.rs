//! FIFO (First-In-First-Out) scheduler implementation

use crate::{
    BatchHint, BatchPlan, BatchResourceRequirements, PreemptionResult, ScheduledRequest, Scheduler,
};
use async_trait::async_trait;
use ferrum_interfaces::scheduler::SchedulerMetrics;
use ferrum_types::SchedulerConfig;
use ferrum_types::{
    BatchId, InferenceRequest, InferenceResponse, Priority, RequestId, RequestState, Result,
};
use parking_lot::RwLock;
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

/// FIFO scheduler that processes requests in first-come-first-served order
pub struct FifoScheduler {
    /// Configuration
    config: SchedulerConfig,
    /// Waiting queue (FIFO order)
    waiting_queue: RwLock<VecDeque<ScheduledRequest>>,
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

/// Internal metrics tracker
struct MetricsTracker {
    total_wait_time_ms: AtomicU64,
    total_execution_time_ms: AtomicU64,
    request_count: AtomicU64,
}

impl MetricsTracker {
    fn new() -> Self {
        Self {
            total_wait_time_ms: AtomicU64::new(0),
            total_execution_time_ms: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
        }
    }

    fn record_completion(&self, wait_time_ms: u64, execution_time_ms: u64) {
        self.total_wait_time_ms
            .fetch_add(wait_time_ms, Ordering::Relaxed);
        self.total_execution_time_ms
            .fetch_add(execution_time_ms, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);
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
}

impl FifoScheduler {
    /// Create new FIFO scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        info!("Creating FIFO scheduler with config: {:?}", config);

        Self {
            config,
            waiting_queue: RwLock::new(VecDeque::new()),
            running_requests: RwLock::new(HashMap::new()),
            completed_counter: AtomicU64::new(0),
            failed_counter: AtomicU64::new(0),
            cancelled_counter: AtomicU64::new(0),
            start_time: Instant::now(),
            metrics_tracker: Arc::new(MetricsTracker::new()),
        }
    }

    /// Create batch from waiting queue
    fn create_batch(&self, hint: BatchHint) -> Option<BatchPlan> {
        let mut waiting_queue = self.waiting_queue.write();
        let mut running_requests = self.running_requests.write();

        if waiting_queue.is_empty() {
            return None;
        }

        let mut batch_requests = Vec::new();
        let mut total_tokens = 0;
        let max_sequence_length = hint.max_tokens.min(2048); // reasonable default

        // Take requests from front of queue (FIFO)
        while batch_requests.len() < hint.max_batch_size
            && total_tokens < hint.max_tokens
            && !waiting_queue.is_empty()
        {
            if let Some(mut scheduled_req) = waiting_queue.pop_front() {
                let request_tokens = scheduled_req.request.sampling_params.max_tokens;

                // Check if adding this request would exceed limits
                if total_tokens + request_tokens <= hint.max_tokens {
                    scheduled_req.state = RequestState::Running;
                    scheduled_req.started_at = Some(chrono::Utc::now());
                    scheduled_req.queue_position = None;

                    total_tokens += request_tokens;

                    // Move to running requests
                    let request_id = scheduled_req.request.id.clone();
                    running_requests.insert(request_id, scheduled_req.clone());
                    batch_requests.push(scheduled_req);
                } else {
                    // Put the request back
                    waiting_queue.push_front(scheduled_req);
                    break;
                }
            }
        }

        if batch_requests.is_empty() {
            return None;
        }

        let batch_id = BatchId::new();
        debug!(
            "Creating batch {} with {} requests",
            batch_id,
            batch_requests.len()
        );

        Some(BatchPlan {
            batch_id,
            requests: batch_requests,
            max_sequence_length,
            estimated_time_ms: Some(1000), // Simplified estimate
            resource_requirements: BatchResourceRequirements {
                gpu_memory: (total_tokens * 16) as u64, // Rough estimate: 16 bytes per token
                cpu_memory: (total_tokens * 4) as u64,  // Rough estimate: 4 bytes per token
                kv_cache_blocks: total_tokens / 16,     // Assume 16 tokens per block
                compute_units: 1,
            },
            created_at: chrono::Utc::now(),
        })
    }
}

#[async_trait]
impl Scheduler for FifoScheduler {
    async fn submit(&self, request: InferenceRequest) -> Result<RequestId> {
        let request_id = request.id.clone();
        debug!("Submitting request {} to FIFO scheduler", request_id);

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

        // Add to waiting queue
        let mut waiting_queue = self.waiting_queue.write();
        let queue_position = waiting_queue.len();

        let mut scheduled_req = scheduled_request;
        scheduled_req.queue_position = Some(queue_position);

        waiting_queue.push_back(scheduled_req);

        info!(
            "Request {} queued at position {}",
            request_id, queue_position
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

            self.metrics_tracker.record_completion(
                wait_time.as_millis() as u64,
                execution_time.as_millis() as u64,
            );

            match response.finish_reason {
                ferrum_types::FinishReason::EOS | ferrum_types::FinishReason::Stop => {
                    self.completed_counter.fetch_add(1, Ordering::Relaxed);
                    debug!("Request {} completed successfully", request_id);
                }
                _ => {
                    self.failed_counter.fetch_add(1, Ordering::Relaxed);
                    warn!(
                        "Request {} completed with error: {:?}",
                        request_id, response.finish_reason
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
        if let Some(pos) = waiting_queue
            .iter()
            .position(|req| req.request.id == request_id)
        {
            waiting_queue.remove(pos);
            self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
            info!("Request {} cancelled from waiting queue", request_id);
            return Ok(true);
        }
        drop(waiting_queue);

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

    async fn update_priority(&self, request_id: RequestId, _priority: Priority) -> Result<()> {
        // FIFO scheduler ignores priority updates by design
        debug!(
            "Priority update ignored for request {} in FIFO scheduler",
            request_id
        );
        Ok(())
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

        ferrum_types::SchedulerStats {
            waiting_requests: waiting_count,
            running_requests: running_count,
            preempted_requests: 0, // FIFO doesn't support preemption
            completed_requests: completed_count,
            failed_requests: failed_count,
            cancelled_requests: cancelled_count,
            avg_wait_time_ms: self.metrics_tracker.avg_wait_time_ms(),
            avg_execution_time_ms: self.metrics_tracker.avg_execution_time_ms(),
            throughput_rps: throughput,
            queue_utilization,
        }
    }

    fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    async fn preempt(&self, _request_id: RequestId) -> Result<PreemptionResult> {
        Err(ferrum_types::FerrumError::unsupported(
            "FIFO scheduler does not support preemption",
        ))
    }

    async fn resume(&self, _request_id: RequestId) -> Result<()> {
        Err(ferrum_types::FerrumError::unsupported(
            "FIFO scheduler does not support resumption",
        ))
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{ModelId, SamplingParams};

    fn create_test_request(priority: Priority) -> InferenceRequest {
        InferenceRequest {
            id: RequestId::new(),
            prompt: "test".to_string(),
            model_id: ModelId::new("test-model"),
            sampling_params: SamplingParams::default(),
            stream: false,
            priority,
            client_id: None,
            session_id: None,
            created_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_fifo_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = FifoScheduler::new(config);
        assert_eq!(scheduler.waiting_queue.read().len(), 0);
    }

    #[tokio::test]
    async fn test_fifo_submit_and_batch() {
        let config = SchedulerConfig::default();
        let scheduler = FifoScheduler::new(config);

        // Submit requests
        let _id1 = scheduler
            .submit(create_test_request(Priority::Normal))
            .await
            .unwrap();
        let _id2 = scheduler
            .submit(create_test_request(Priority::High))
            .await
            .unwrap();

        // Should have 2 waiting
        assert_eq!(scheduler.waiting_queue.read().len(), 2);

        // Get batch
        let batch = scheduler.next_batch(BatchHint::simple(5)).await;
        assert!(batch.is_some());
    }

    #[tokio::test]
    async fn test_fifo_cancel() {
        let config = SchedulerConfig::default();
        let scheduler = FifoScheduler::new(config);

        let request = create_test_request(Priority::Normal);
        let id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        let result = scheduler.cancel(id).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_metrics_tracker() {
        let tracker = MetricsTracker::new();
        tracker.record_completion(100, 500);

        assert!(tracker.avg_wait_time_ms() > 0.0);
        assert!(tracker.avg_execution_time_ms() > 0.0);
    }
}
