//! Continuous Batching Scheduler
//!
//! This scheduler implements iteration-level scheduling that allows dynamic
//! addition and removal of requests from running batches. Key features:
//!
//! - Iteration-level granularity: can add/remove requests between decode steps
//! - Separate prefill and decode queues for optimal scheduling
//! - Request state machine: Waiting -> Prefilling -> Decoding -> Completed
//! - Memory-aware scheduling based on KV cache usage
//! - Preemption support for long-running requests

use crate::{
    BatchHint, BatchPlan, BatchResourceRequirements, PreemptionResult, PreemptionState,
    ScheduledRequest, Scheduler,
};
use async_trait::async_trait;
use ferrum_interfaces::scheduler::SchedulerMetrics;
use ferrum_types::SchedulerConfig;
use ferrum_types::{
    BatchId, FerrumError, InferenceRequest, InferenceResponse, Priority, RequestId, RequestState,
    Result,
};
use parking_lot::RwLock;
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};
use tracing::{debug, info, warn};

/// Request phase in continuous batching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    /// Waiting in queue
    Waiting,
    /// Currently in prefill phase
    Prefilling,
    /// In decode phase (generating tokens)
    Decoding,
    /// Request completed
    Completed,
    /// Request was preempted
    Preempted,
    /// Request was cancelled
    Cancelled,
}

/// Extended scheduled request with continuous batching metadata
#[derive(Debug, Clone)]
pub struct ContinuousBatchRequest {
    /// Base scheduled request
    pub inner: ScheduledRequest,
    /// Current phase
    pub phase: RequestPhase,
    /// Number of prefill tokens
    pub prefill_tokens: usize,
    /// Number of decode tokens generated
    pub decode_tokens: usize,
    /// KV cache blocks allocated
    pub kv_blocks: Vec<ferrum_types::BlockId>,
    /// Whether prefill is chunked
    pub chunked_prefill: bool,
    /// Current chunk offset for chunked prefill
    pub prefill_chunk_offset: usize,
    /// Last iteration this request was processed
    pub last_iteration: u64,
    /// Time spent in prefill (ms)
    pub prefill_time_ms: u64,
    /// Time spent in decode (ms)
    pub decode_time_ms: u64,
}

impl ContinuousBatchRequest {
    /// Create from inference request
    pub fn new(request: InferenceRequest) -> Self {
        Self {
            inner: ScheduledRequest::new(request),
            phase: RequestPhase::Waiting,
            prefill_tokens: 0,
            decode_tokens: 0,
            kv_blocks: Vec::new(),
            chunked_prefill: false,
            prefill_chunk_offset: 0,
            last_iteration: 0,
            prefill_time_ms: 0,
            decode_time_ms: 0,
        }
    }

    /// Get total tokens processed
    pub fn total_tokens(&self) -> usize {
        self.prefill_tokens + self.decode_tokens
    }

    /// Check if request is active (prefilling or decoding)
    pub fn is_active(&self) -> bool {
        matches!(self.phase, RequestPhase::Prefilling | RequestPhase::Decoding)
    }

    /// Check if request is finished
    pub fn is_finished(&self) -> bool {
        matches!(
            self.phase,
            RequestPhase::Completed | RequestPhase::Cancelled
        )
    }
}

/// Continuous batching scheduler
///
/// This scheduler manages requests through their lifecycle in a continuous
/// batching system, allowing for iteration-level scheduling decisions.
pub struct ContinuousBatchScheduler {
    /// Configuration
    config: SchedulerConfig,

    /// Waiting queue (requests waiting to start)
    waiting_queue: RwLock<VecDeque<ContinuousBatchRequest>>,

    /// Prefill queue (requests in prefill phase)
    prefill_queue: RwLock<VecDeque<ContinuousBatchRequest>>,

    /// Decode queue (requests in decode phase)
    decode_queue: RwLock<HashMap<RequestId, ContinuousBatchRequest>>,

    /// Preempted requests (can be resumed)
    preempted_requests: RwLock<HashMap<RequestId, ContinuousBatchRequest>>,

    /// Request lookup table
    request_index: RwLock<HashMap<RequestId, RequestPhase>>,

    /// Current iteration number
    current_iteration: AtomicU64,

    /// Statistics
    completed_counter: AtomicU64,
    failed_counter: AtomicU64,
    cancelled_counter: AtomicU64,
    preempted_counter: AtomicU64,

    /// Start time
    start_time: Instant,

    /// Metrics tracker
    metrics_tracker: Arc<ContinuousBatchMetrics>,

    /// Continuous batching specific config
    cb_config: ContinuousBatchConfig,
}

/// Continuous batching specific configuration
#[derive(Debug, Clone)]
pub struct ContinuousBatchConfig {
    /// Maximum batch size for prefill
    pub max_prefill_batch: usize,
    /// Maximum batch size for decode
    pub max_decode_batch: usize,
    /// Enable chunked prefill
    pub enable_chunked_prefill: bool,
    /// Chunk size for chunked prefill (tokens)
    pub prefill_chunk_size: usize,
    /// Maximum KV cache blocks per request
    pub max_kv_blocks_per_request: usize,
    /// Enable request swapping (preemption)
    pub enable_swapping: bool,
    /// Swap priority threshold
    pub swap_priority_threshold: Priority,
    /// Target iteration time (ms)
    pub target_iteration_time_ms: u64,
}

impl Default for ContinuousBatchConfig {
    fn default() -> Self {
        Self {
            max_prefill_batch: 8,
            max_decode_batch: 256,
            enable_chunked_prefill: true,
            prefill_chunk_size: 512,
            max_kv_blocks_per_request: 1024,
            enable_swapping: true,
            swap_priority_threshold: Priority::Low,
            target_iteration_time_ms: 50,
        }
    }
}

/// Metrics tracker for continuous batching
struct ContinuousBatchMetrics {
    total_prefill_tokens: AtomicU64,
    total_decode_tokens: AtomicU64,
    total_prefill_time_ms: AtomicU64,
    total_decode_time_ms: AtomicU64,
    request_count: AtomicU64,
    iteration_count: AtomicU64,
}

impl ContinuousBatchMetrics {
    fn new() -> Self {
        Self {
            total_prefill_tokens: AtomicU64::new(0),
            total_decode_tokens: AtomicU64::new(0),
            total_prefill_time_ms: AtomicU64::new(0),
            total_decode_time_ms: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
            iteration_count: AtomicU64::new(0),
        }
    }

    fn record_completion(&self, req: &ContinuousBatchRequest) {
        self.total_prefill_tokens
            .fetch_add(req.prefill_tokens as u64, Ordering::Relaxed);
        self.total_decode_tokens
            .fetch_add(req.decode_tokens as u64, Ordering::Relaxed);
        self.total_prefill_time_ms
            .fetch_add(req.prefill_time_ms, Ordering::Relaxed);
        self.total_decode_time_ms
            .fetch_add(req.decode_time_ms, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }

    fn record_iteration(&self) {
        self.iteration_count.fetch_add(1, Ordering::Relaxed);
    }

    fn avg_prefill_tokens(&self) -> f64 {
        let total = self.total_prefill_tokens.load(Ordering::Relaxed) as f64;
        let count = self.request_count.load(Ordering::Relaxed) as f64;
        if count > 0.0 {
            total / count
        } else {
            0.0
        }
    }

    fn avg_decode_tokens(&self) -> f64 {
        let total = self.total_decode_tokens.load(Ordering::Relaxed) as f64;
        let count = self.request_count.load(Ordering::Relaxed) as f64;
        if count > 0.0 {
            total / count
        } else {
            0.0
        }
    }
}

impl ContinuousBatchScheduler {
    /// Create new continuous batch scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        Self::with_cb_config(config, ContinuousBatchConfig::default())
    }

    /// Create with specific continuous batching configuration
    pub fn with_cb_config(config: SchedulerConfig, cb_config: ContinuousBatchConfig) -> Self {
        info!(
            "Creating continuous batch scheduler: max_prefill={}, max_decode={}",
            cb_config.max_prefill_batch, cb_config.max_decode_batch
        );

        Self {
            config,
            waiting_queue: RwLock::new(VecDeque::new()),
            prefill_queue: RwLock::new(VecDeque::new()),
            decode_queue: RwLock::new(HashMap::new()),
            preempted_requests: RwLock::new(HashMap::new()),
            request_index: RwLock::new(HashMap::new()),
            current_iteration: AtomicU64::new(0),
            completed_counter: AtomicU64::new(0),
            failed_counter: AtomicU64::new(0),
            cancelled_counter: AtomicU64::new(0),
            preempted_counter: AtomicU64::new(0),
            start_time: Instant::now(),
            metrics_tracker: Arc::new(ContinuousBatchMetrics::new()),
            cb_config,
        }
    }

    /// Get number of active requests (prefilling + decoding)
    pub fn active_count(&self) -> usize {
        self.prefill_queue.read().len() + self.decode_queue.read().len()
    }

    /// Get number of waiting requests
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.read().len()
    }

    /// Get number of decoding requests
    pub fn decoding_count(&self) -> usize {
        self.decode_queue.read().len()
    }

    /// Get number of prefilling requests
    pub fn prefilling_count(&self) -> usize {
        self.prefill_queue.read().len()
    }

    /// Move request from waiting to prefill queue
    fn promote_to_prefill(&self, request_id: &RequestId) -> bool {
        let mut waiting_queue = self.waiting_queue.write();
        let mut prefill_queue = self.prefill_queue.write();
        let mut request_index = self.request_index.write();

        if let Some(pos) = waiting_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let mut req = waiting_queue.remove(pos).unwrap();
            req.phase = RequestPhase::Prefilling;
            req.inner.state = RequestState::Running;
            req.inner.started_at = Some(chrono::Utc::now());

            request_index.insert(request_id.clone(), RequestPhase::Prefilling);
            prefill_queue.push_back(req);

            debug!("Promoted request {} to prefill queue", request_id);
            true
        } else {
            false
        }
    }

    /// Move request from prefill to decode queue
    fn promote_to_decode(&self, request_id: &RequestId) -> bool {
        let mut prefill_queue = self.prefill_queue.write();
        let mut decode_queue = self.decode_queue.write();
        let mut request_index = self.request_index.write();

        if let Some(pos) = prefill_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let mut req = prefill_queue.remove(pos).unwrap();
            req.phase = RequestPhase::Decoding;

            request_index.insert(request_id.clone(), RequestPhase::Decoding);
            decode_queue.insert(request_id.clone(), req);

            debug!("Promoted request {} to decode queue", request_id);
            true
        } else {
            false
        }
    }

    /// Create batch plan for current iteration
    fn create_iteration_batch(&self, hint: BatchHint) -> Option<BatchPlan> {
        let iteration = self.current_iteration.fetch_add(1, Ordering::Relaxed);
        self.metrics_tracker.record_iteration();

        let mut batch_requests = Vec::new();
        let mut total_tokens = 0;

        // First, collect decode requests (they have priority)
        let decode_queue = self.decode_queue.read();
        for (_, req) in decode_queue.iter() {
            if batch_requests.len() >= hint.max_batch_size {
                break;
            }

            // Each decode step is 1 token per request
            if total_tokens + 1 <= hint.max_tokens {
                let mut scheduled = req.inner.clone();
                scheduled.tokens_processed = req.total_tokens();
                batch_requests.push(scheduled);
                total_tokens += 1;
            }
        }
        drop(decode_queue);

        // Then, add prefill requests if we have capacity
        let prefill_remaining = hint
            .max_batch_size
            .saturating_sub(batch_requests.len())
            .min(self.cb_config.max_prefill_batch);

        if prefill_remaining > 0 {
            let prefill_queue = self.prefill_queue.read();
            for req in prefill_queue.iter().take(prefill_remaining) {
                let prefill_chunk_tokens = if self.cb_config.enable_chunked_prefill {
                    self.cb_config
                        .prefill_chunk_size
                        .min(req.prefill_tokens.saturating_sub(req.prefill_chunk_offset))
                } else {
                    req.prefill_tokens
                };

                if total_tokens + prefill_chunk_tokens <= hint.max_tokens {
                    let mut scheduled = req.inner.clone();
                    scheduled.tokens_processed = req.total_tokens();
                    batch_requests.push(scheduled);
                    total_tokens += prefill_chunk_tokens;
                }
            }
        }

        // Check if we should admit new requests from waiting queue
        let waiting_queue = self.waiting_queue.read();
        let available_slots = self
            .cb_config
            .max_decode_batch
            .saturating_sub(self.decoding_count());

        let requests_to_admit: Vec<RequestId> = waiting_queue
            .iter()
            .take(available_slots)
            .map(|r| r.inner.request.id.clone())
            .collect();
        drop(waiting_queue);

        // Promote waiting requests to prefill
        for req_id in requests_to_admit {
            self.promote_to_prefill(&req_id);
        }

        // After promotion, add newly prefilling requests to batch
        if batch_requests.is_empty() {
            let prefill_queue = self.prefill_queue.read();
            for req in prefill_queue.iter().take(hint.max_batch_size) {
                let prefill_chunk_tokens = if self.cb_config.enable_chunked_prefill {
                    self.cb_config
                        .prefill_chunk_size
                        .min(req.prefill_tokens.saturating_sub(req.prefill_chunk_offset))
                } else {
                    // For new requests, prefill_tokens might be 0, use a default
                    req.inner.request.sampling_params.max_tokens.min(512)
                };

                if total_tokens + prefill_chunk_tokens <= hint.max_tokens {
                    let mut scheduled = req.inner.clone();
                    scheduled.tokens_processed = req.total_tokens();
                    batch_requests.push(scheduled);
                    total_tokens += prefill_chunk_tokens;
                }
            }
        }

        if batch_requests.is_empty() {
            return None;
        }

        let batch_id = BatchId::new();
        let max_seq_len = batch_requests
            .iter()
            .map(|r| r.request.sampling_params.max_tokens)
            .max()
            .unwrap_or(2048);

        debug!(
            "Created iteration {} batch: {} requests, {} tokens",
            iteration,
            batch_requests.len(),
            total_tokens
        );

        Some(BatchPlan {
            batch_id,
            requests: batch_requests,
            max_sequence_length: max_seq_len,
            estimated_time_ms: Some(self.cb_config.target_iteration_time_ms),
            resource_requirements: BatchResourceRequirements {
                gpu_memory: (total_tokens * 16) as u64,
                cpu_memory: (total_tokens * 4) as u64,
                kv_cache_blocks: total_tokens / 16,
                compute_units: 1,
            },
            created_at: chrono::Utc::now(),
        })
    }

    /// Mark a request as having completed prefill
    pub fn mark_prefill_complete(&self, request_id: &RequestId, tokens: usize) {
        let mut prefill_queue = self.prefill_queue.write();
        if let Some(pos) = prefill_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let req = &mut prefill_queue[pos];
            req.prefill_tokens = tokens;
        }
        drop(prefill_queue);

        // Promote to decode
        self.promote_to_decode(request_id);
    }

    /// Update decode progress for a request
    pub fn update_decode_progress(&self, request_id: &RequestId, tokens_generated: usize) {
        let mut decode_queue = self.decode_queue.write();
        if let Some(req) = decode_queue.get_mut(request_id) {
            req.decode_tokens = tokens_generated;
            req.last_iteration = self.current_iteration.load(Ordering::Relaxed);
        }
    }
}

#[async_trait]
impl Scheduler for ContinuousBatchScheduler {
    async fn submit(&self, request: InferenceRequest) -> Result<RequestId> {
        let request_id = request.id.clone();
        debug!(
            "Submitting request {} to continuous batch scheduler",
            request_id
        );

        // Check queue capacity
        let waiting_count = self.waiting_count();
        if waiting_count >= self.config.max_waiting_requests {
            warn!("Queue is full, rejecting request {}", request_id);
            return Err(FerrumError::scheduler(
                "Queue is full, cannot accept more requests",
            ));
        }

        // Create continuous batch request
        let cb_request = ContinuousBatchRequest::new(request);

        // Add to waiting queue
        let mut waiting_queue = self.waiting_queue.write();
        let queue_position = waiting_queue.len();

        let mut req = cb_request;
        req.inner.queue_position = Some(queue_position);

        waiting_queue.push_back(req);

        // Update index
        self.request_index
            .write()
            .insert(request_id.clone(), RequestPhase::Waiting);

        info!(
            "Request {} queued at position {}",
            request_id, queue_position
        );
        Ok(request_id)
    }

    async fn next_batch(&self, hint: BatchHint) -> Option<BatchPlan> {
        self.create_iteration_batch(hint)
    }

    async fn complete(&self, request_id: RequestId, response: &InferenceResponse) -> Result<()> {
        debug!("Completing request {}", request_id);

        // Remove from decode queue
        let mut decode_queue = self.decode_queue.write();
        if let Some(req) = decode_queue.remove(&request_id) {
            // Record metrics
            self.metrics_tracker.record_completion(&req);

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

            // Remove from index
            self.request_index.write().remove(&request_id);

            Ok(())
        } else {
            // Try removing from prefill queue
            let mut prefill_queue = self.prefill_queue.write();
            if let Some(pos) = prefill_queue
                .iter()
                .position(|r| r.inner.request.id == request_id)
            {
                prefill_queue.remove(pos);
                self.request_index.write().remove(&request_id);
                self.completed_counter.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }

            warn!("Attempted to complete unknown request: {}", request_id);
            Err(FerrumError::scheduler(format!(
                "Request {} not found in active queues",
                request_id
            )))
        }
    }

    async fn cancel(&self, request_id: RequestId) -> Result<bool> {
        debug!("Cancelling request {}", request_id);

        // Check and remove from waiting queue
        {
            let mut waiting_queue = self.waiting_queue.write();
            if let Some(pos) = waiting_queue
                .iter()
                .position(|r| r.inner.request.id == request_id)
            {
                waiting_queue.remove(pos);
                self.request_index.write().remove(&request_id);
                self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
                info!("Request {} cancelled from waiting queue", request_id);
                return Ok(true);
            }
        }

        // Check and remove from prefill queue
        {
            let mut prefill_queue = self.prefill_queue.write();
            if let Some(pos) = prefill_queue
                .iter()
                .position(|r| r.inner.request.id == request_id)
            {
                prefill_queue.remove(pos);
                self.request_index.write().remove(&request_id);
                self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
                warn!("Request {} cancelled during prefill", request_id);
                return Ok(true);
            }
        }

        // Check and remove from decode queue
        {
            let mut decode_queue = self.decode_queue.write();
            if decode_queue.remove(&request_id).is_some() {
                self.request_index.write().remove(&request_id);
                self.cancelled_counter.fetch_add(1, Ordering::Relaxed);
                warn!("Request {} cancelled during decode", request_id);
                return Ok(true);
            }
        }

        warn!("Request {} not found for cancellation", request_id);
        Ok(false)
    }

    async fn update_priority(&self, request_id: RequestId, priority: Priority) -> Result<()> {
        debug!(
            "Updating priority for request {} to {:?}",
            request_id, priority
        );

        // Update in waiting queue
        {
            let mut waiting_queue = self.waiting_queue.write();
            if let Some(req) = waiting_queue
                .iter_mut()
                .find(|r| r.inner.request.id == request_id)
            {
                req.inner.request.priority = priority;
                return Ok(());
            }
        }

        // Update in prefill queue
        {
            let mut prefill_queue = self.prefill_queue.write();
            if let Some(req) = prefill_queue
                .iter_mut()
                .find(|r| r.inner.request.id == request_id)
            {
                req.inner.request.priority = priority;
                return Ok(());
            }
        }

        // Update in decode queue
        {
            let mut decode_queue = self.decode_queue.write();
            if let Some(req) = decode_queue.get_mut(&request_id) {
                req.inner.request.priority = priority;
                return Ok(());
            }
        }

        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        let waiting_count = self.waiting_count();
        let prefill_count = self.prefilling_count();
        let decode_count = self.decoding_count();
        let running_count = prefill_count + decode_count;

        let completed_count = self.completed_counter.load(Ordering::Relaxed);
        let failed_count = self.failed_counter.load(Ordering::Relaxed);
        let cancelled_count = self.cancelled_counter.load(Ordering::Relaxed);
        let preempted_count = self.preempted_counter.load(Ordering::Relaxed);

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
            preempted_requests: preempted_count as usize,
            completed_requests: completed_count,
            failed_requests: failed_count,
            cancelled_requests: cancelled_count,
            avg_wait_time_ms: 0.0, // TODO: track wait times
            avg_execution_time_ms: 0.0,
            throughput_rps: throughput,
            queue_utilization,
        }
    }

    fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    async fn preempt(&self, request_id: RequestId) -> Result<PreemptionResult> {
        if !self.cb_config.enable_swapping {
            return Err(FerrumError::unsupported("Swapping is not enabled"));
        }

        debug!("Preempting request {}", request_id);

        // Remove from decode queue
        let mut decode_queue = self.decode_queue.write();
        if let Some(mut req) = decode_queue.remove(&request_id) {
            req.phase = RequestPhase::Preempted;

            // Save preemption state
            let state = PreemptionState {
                kv_cache_checkpoint: Vec::new(), // TODO: implement actual checkpoint
                tokens_processed: req.total_tokens(),
                generation_state: HashMap::new(),
            };

            let freed_resources = req.inner.allocated_resources.clone();

            // Move to preempted queue
            self.preempted_requests
                .write()
                .insert(request_id.clone(), req);
            self.request_index
                .write()
                .insert(request_id, RequestPhase::Preempted);
            self.preempted_counter.fetch_add(1, Ordering::Relaxed);

            Ok(PreemptionResult {
                success: true,
                saved_state: Some(state),
                freed_resources,
            })
        } else {
            Err(FerrumError::scheduler(format!(
                "Request {} not found in decode queue",
                request_id
            )))
        }
    }

    async fn resume(&self, request_id: RequestId) -> Result<()> {
        debug!("Resuming request {}", request_id);

        let mut preempted = self.preempted_requests.write();
        if let Some(mut req) = preempted.remove(&request_id) {
            req.phase = RequestPhase::Decoding;

            self.decode_queue
                .write()
                .insert(request_id.clone(), req);
            self.request_index
                .write()
                .insert(request_id, RequestPhase::Decoding);

            Ok(())
        } else {
            Err(FerrumError::scheduler(format!(
                "Request {} not found in preempted queue",
                request_id
            )))
        }
    }
}

impl std::fmt::Debug for ContinuousBatchScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContinuousBatchScheduler")
            .field("waiting", &self.waiting_count())
            .field("prefilling", &self.prefilling_count())
            .field("decoding", &self.decoding_count())
            .field("iteration", &self.current_iteration.load(Ordering::Relaxed))
            .finish()
    }
}

// ============================================================================
// Tests
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
    async fn test_scheduler_creation() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(scheduler.active_count(), 0);
    }

    #[tokio::test]
    async fn test_submit_and_counts() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);

        scheduler
            .submit(create_test_request(Priority::Normal))
            .await
            .unwrap();
        scheduler
            .submit(create_test_request(Priority::High))
            .await
            .unwrap();

        assert_eq!(scheduler.waiting_count(), 2);
        assert_eq!(scheduler.active_count(), 0);
    }

    #[tokio::test]
    async fn test_batch_creation() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);

        // Submit some requests
        for _ in 0..5 {
            scheduler
                .submit(create_test_request(Priority::Normal))
                .await
                .unwrap();
        }

        // Get batch
        let batch = scheduler.next_batch(BatchHint::simple(10)).await;
        assert!(batch.is_some());

        // Requests should have been promoted
        assert!(scheduler.prefilling_count() > 0 || scheduler.decoding_count() > 0);
    }

    #[tokio::test]
    async fn test_cancel_waiting() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);

        let request = create_test_request(Priority::Normal);
        let id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        assert_eq!(scheduler.waiting_count(), 1);

        let result = scheduler.cancel(id).await.unwrap();
        assert!(result);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[tokio::test]
    async fn test_metrics() {
        let config = SchedulerConfig::default();
        let scheduler = ContinuousBatchScheduler::new(config);

        scheduler
            .submit(create_test_request(Priority::Normal))
            .await
            .unwrap();

        let metrics = scheduler.metrics();
        assert_eq!(metrics.waiting_requests, 1);
    }

    #[test]
    fn test_cb_request_states() {
        let request = create_test_request(Priority::Normal);
        let cb_req = ContinuousBatchRequest::new(request);

        assert_eq!(cb_req.phase, RequestPhase::Waiting);
        assert!(!cb_req.is_active());
        assert!(!cb_req.is_finished());
    }
}

