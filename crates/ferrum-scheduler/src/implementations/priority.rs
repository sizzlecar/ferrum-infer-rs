//! Priority-based scheduler implementation

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
            Priority::Normal => 50,
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
            Priority::Normal => 1000,
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
                ferrum_types::FinishReason::EOS | ferrum_types::FinishReason::Stop => {
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

        let _p95_wait_estimate = if critical_wait > 0.0 || high_wait > 0.0 {
            (critical_wait + high_wait) / 2.0 * 1.2 // Priority requests should have better P95
        } else {
            avg_wait * 1.5 // Fallback to simple estimate
        };

        ferrum_types::SchedulerStats {
            waiting_requests: waiting_count,
            running_requests: running_count,
            preempted_requests: 0, // MVP: no preemption tracking
            completed_requests: completed_count,
            failed_requests: failed_count,
            cancelled_requests: cancelled_count,
            avg_wait_time_ms: avg_wait,
            avg_execution_time_ms: self.metrics_tracker.avg_execution_time_ms(),
            throughput_rps: throughput,
            queue_utilization,
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
            if matches!(priority, Priority::Low | Priority::Normal) {
                if let Some(removed_req) = running_requests.remove(&request_id) {
                    // Move back to waiting queue with updated priority
                    let mut waiting_queue = self.waiting_queue.write();
                    let mut request_map = self.request_map.write();

                    let mut preempted_req = removed_req;
                    preempted_req.state = RequestState::Waiting;
                    preempted_req.started_at = None;

                    let request_priority =
                        RequestPriority::new(priority, preempted_req.submitted_at);
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

// ============================================================================
// 内联单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_types::{ModelId, SamplingParams};

    fn create_test_request_with_priority(priority: Priority) -> InferenceRequest {
        InferenceRequest {
            id: RequestId::new(),
            prompt: "test prompt".to_string(),
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

    #[test]
    fn test_request_priority_ordering() {
        let now = chrono::Utc::now();
        let later = now + chrono::Duration::seconds(1);

        let critical = RequestPriority::new(Priority::Critical, now);
        let high = RequestPriority::new(Priority::High, now);
        let normal = RequestPriority::new(Priority::Normal, now);
        let low = RequestPriority::new(Priority::Low, now);

        // 优先级应该是：Critical > High > Normal > Low
        assert!(critical > high);
        assert!(high > normal);
        assert!(normal > low);
    }

    #[test]
    fn test_request_priority_fifo_within_same_level() {
        let now = chrono::Utc::now();
        let later = now + chrono::Duration::seconds(1);

        let high1 = RequestPriority::new(Priority::High, now);
        let high2 = RequestPriority::new(Priority::High, later);

        // 相同优先级内，早提交的请求优先级更高（负时间戳）
        assert!(high1 > high2);
    }

    #[tokio::test]
    async fn test_priority_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config.clone());

        assert_eq!(
            scheduler.config().max_waiting_requests,
            config.max_waiting_requests
        );
    }

    #[tokio::test]
    async fn test_priority_scheduler_submit() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        let request = create_test_request_with_priority(Priority::Normal);
        let request_id = request.id.clone();

        let result = scheduler.submit(request).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), request_id);

        // 验证请求在队列中
        let waiting_queue = scheduler.waiting_queue.read();
        assert_eq!(waiting_queue.len(), 1);
    }

    #[tokio::test]
    async fn test_priority_ordering_in_queue() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        // 提交不同优先级的请求
        let low_req = create_test_request_with_priority(Priority::Low);
        let normal_req = create_test_request_with_priority(Priority::Normal);
        let high_req = create_test_request_with_priority(Priority::High);
        let critical_req = create_test_request_with_priority(Priority::Critical);

        scheduler.submit(low_req).await.unwrap();
        scheduler.submit(normal_req).await.unwrap();
        scheduler.submit(high_req).await.unwrap();
        scheduler.submit(critical_req.clone()).await.unwrap();

        // 获取批次，应该优先处理高优先级请求
        let batch = scheduler.next_batch(BatchHint::simple(10)).await;
        assert!(batch.is_some());

        let batch = batch.unwrap();
        // 第一个请求应该是 Critical 优先级
        assert_eq!(batch.requests[0].request.priority, Priority::Critical);
    }

    #[tokio::test]
    async fn test_priority_scheduler_cancel() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        let request = create_test_request_with_priority(Priority::Normal);
        let request_id = request.id.clone();

        scheduler.submit(request).await.unwrap();

        // 取消请求
        let result = scheduler.cancel(request_id).await;
        assert!(result.is_ok());
        assert!(result.unwrap());

        // 验证请求不在队列中
        let waiting_queue = scheduler.waiting_queue.read();
        assert_eq!(waiting_queue.len(), 0);
    }

    #[tokio::test]
    async fn test_priority_update() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        let request = create_test_request_with_priority(Priority::Low);
        let request_id = request.id.clone();

        scheduler.submit(request).await.unwrap();

        // 更新优先级
        let result = scheduler
            .update_priority(request_id.clone(), Priority::High)
            .await;
        assert!(result.is_ok());

        // 验证优先级已更新
        let request_map = scheduler.request_map.read();
        if let Some(scheduled_req) = request_map.get(&request_id) {
            assert_eq!(scheduled_req.request.priority, Priority::High);
        }
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        let request = create_test_request_with_priority(Priority::Normal);
        scheduler.submit(request).await.unwrap();

        let metrics = scheduler.metrics();
        assert_eq!(metrics.waiting_requests, 1);
        assert_eq!(metrics.running_requests, 0);
        assert_eq!(metrics.completed_requests, 0);
    }

    #[tokio::test]
    async fn test_batch_creation() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        // 提交多个请求
        for i in 0..5 {
            let priority = if i % 2 == 0 {
                Priority::High
            } else {
                Priority::Normal
            };
            let request = create_test_request_with_priority(priority);
            scheduler.submit(request).await.unwrap();
        }

        // 创建批次
        let batch = scheduler.next_batch(BatchHint::simple(3)).await;
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert!(batch.requests.len() <= 3);
        assert!(batch.requests.len() > 0);
    }

    #[tokio::test]
    async fn test_preemption_low_priority() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        let low_request = create_test_request_with_priority(Priority::Low);
        let request_id = low_request.id.clone();

        scheduler.submit(low_request).await.unwrap();

        // 获取批次，将请求移到运行队列
        let _batch = scheduler.next_batch(BatchHint::simple(10)).await;

        // 尝试抢占
        let result = scheduler.preempt(request_id).await;
        assert!(result.is_ok());

        let preemption_result = result.unwrap();
        assert!(preemption_result.success);
    }

    #[tokio::test]
    async fn test_cannot_preempt_high_priority() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        let high_request = create_test_request_with_priority(Priority::High);
        let request_id = high_request.id.clone();

        scheduler.submit(high_request).await.unwrap();

        // 获取批次
        let _batch = scheduler.next_batch(BatchHint::simple(10)).await;

        // 尝试抢占高优先级请求应该失败
        let result = scheduler.preempt(request_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_queue_full() {
        let mut config = SchedulerConfig::default();
        config.max_waiting_requests = 2;
        let scheduler = PriorityScheduler::new(config);

        // 填满队列
        scheduler
            .submit(create_test_request_with_priority(Priority::Normal))
            .await
            .unwrap();
        scheduler
            .submit(create_test_request_with_priority(Priority::Normal))
            .await
            .unwrap();

        // 第三个请求应该失败
        let result = scheduler
            .submit(create_test_request_with_priority(Priority::Normal))
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_metrics_tracker_priority_stats() {
        let tracker = MetricsTracker::new();

        tracker.record_completion(100, 500, Priority::High);
        tracker.record_completion(200, 600, Priority::High);
        tracker.record_completion(300, 700, Priority::Normal);

        let high_wait = tracker.priority_wait_time(Priority::High);
        assert_eq!(high_wait, 150.0); // (100 + 200) / 2

        let avg_wait = tracker.avg_wait_time_ms();
        assert_eq!(avg_wait, 200.0); // (100 + 200 + 300) / 3
    }

    #[tokio::test]
    async fn test_complete_request() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        let request = create_test_request_with_priority(Priority::Normal);
        let request_id = request.id.clone();

        scheduler.submit(request).await.unwrap();

        // 获取批次，移到运行队列
        let _batch = scheduler.next_batch(BatchHint::simple(10)).await;

        // 完成请求
        let response = InferenceResponse {
            request_id: request_id.clone(),
            text: "test".to_string(),
            tokens: vec![],
            finish_reason: ferrum_types::FinishReason::EOS,
            usage: ferrum_types::TokenUsage::new(10, 5),
            latency_ms: 100,
            created_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        };

        let result = scheduler.complete(request_id, &response).await;
        assert!(result.is_ok());

        // 验证已完成计数增加
        let metrics = scheduler.metrics();
        assert_eq!(metrics.completed_requests, 1);
    }

    #[tokio::test]
    async fn test_resume_request() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        let request_id = RequestId::new();

        // Resume 应该总是成功（在这个实现中）
        let result = scheduler.resume(request_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_multiple_priority_levels() {
        let config = SchedulerConfig::default();
        let scheduler = PriorityScheduler::new(config);

        // 提交各种优先级的请求
        let priorities = vec![
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Critical,
            Priority::Normal,
            Priority::Low,
        ];

        for priority in priorities {
            scheduler
                .submit(create_test_request_with_priority(priority))
                .await
                .unwrap();
        }

        let metrics = scheduler.metrics();
        assert_eq!(metrics.waiting_requests, 6);

        // 获取批次
        let batch = scheduler.next_batch(BatchHint::simple(10)).await;
        assert!(batch.is_some());

        // 批次中的请求应该按优先级排序
        let batch = batch.unwrap();
        if batch.requests.len() >= 2 {
            // 第一个应该是最高优先级
            assert_eq!(batch.requests[0].request.priority, Priority::Critical);
        }
    }
}
