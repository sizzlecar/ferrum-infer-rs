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
use ferrum_types::{
    BatchId, FerrumError, InferenceRequest, InferenceResponse, Priority, RequestId, RequestState,
    Result, SchedulerConfig, PROMPT_TOKENS_METADATA_KEY,
};
use parking_lot::RwLock;
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};
use tracing::{debug, info, warn};

const NO_CAPACITY_BACKPRESSURE_LIMIT: usize = usize::MAX;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct ContinuousBatchRuntimeConfig {
    prompt_token_estimate: bool,
    prefill_first_until_active: Option<usize>,
    prefill_step_chunk: Option<usize>,
    active_decode_prefill_chunk: Option<usize>,
    scheduler_none_prof: bool,
}

impl ContinuousBatchRuntimeConfig {
    fn from_scheduler_config(config: &SchedulerConfig) -> Self {
        Self {
            prompt_token_estimate: config.prompt_token_estimate,
            prefill_first_until_active: config.prefill_first_until_active,
            prefill_step_chunk: config.prefill_step_chunk,
            active_decode_prefill_chunk: config.active_decode_prefill_chunk,
            scheduler_none_prof: config.scheduler_none_prof,
        }
    }
}

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
    /// Capacity-deferred requests wait for real capacity release before re-admission.
    pub capacity_deferred_until_release_epoch: u64,
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
            capacity_deferred_until_release_epoch: 0,
        }
    }

    /// Get total tokens processed
    pub fn total_tokens(&self) -> usize {
        self.prefill_tokens + self.decode_tokens
    }

    /// Check if request is active (prefilling or decoding)
    pub fn is_active(&self) -> bool {
        matches!(
            self.phase,
            RequestPhase::Prefilling | RequestPhase::Decoding
        )
    }

    /// Check if request is finished
    pub fn is_finished(&self) -> bool {
        matches!(
            self.phase,
            RequestPhase::Completed | RequestPhase::Cancelled
        )
    }
}

/// Read-only scheduler counters for explicit engine diagnostics.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ContinuousSchedulerTraceSnapshot {
    pub current_iteration: u64,
    pub waiting_queue_len: usize,
    pub prefill_queue_len: usize,
    pub decode_queue_len: usize,
    pub preempted_queue_len: usize,
    pub active_len: usize,
    pub completed_total: u64,
    pub failed_total: u64,
    pub cancelled_total: u64,
    pub preempted_total: u64,
    pub admitted_total: u64,
    pub capacity_deferred_total: u64,
    pub capacity_backpressure_admit_limit: Option<usize>,
    pub capacity_blocked_waiting_len: usize,
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
    admitted_counter: AtomicU64,
    capacity_deferred_counter: AtomicU64,
    capacity_backpressure_limit: AtomicUsize,
    capacity_backpressure_iteration: AtomicU64,
    capacity_release_epoch: AtomicU64,
    total_wait_time_us: AtomicU64,

    /// Start time
    start_time: Instant,

    /// Metrics tracker
    metrics_tracker: Arc<ContinuousBatchMetrics>,

    /// Continuous batching specific config
    cb_config: ContinuousBatchConfig,

    /// Runtime env-derived switches parsed once at scheduler construction.
    runtime_config: ContinuousBatchRuntimeConfig,
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
        let runtime_config = ContinuousBatchRuntimeConfig::from_scheduler_config(&config);

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
            admitted_counter: AtomicU64::new(0),
            capacity_deferred_counter: AtomicU64::new(0),
            capacity_backpressure_limit: AtomicUsize::new(NO_CAPACITY_BACKPRESSURE_LIMIT),
            capacity_backpressure_iteration: AtomicU64::new(u64::MAX),
            capacity_release_epoch: AtomicU64::new(0),
            total_wait_time_us: AtomicU64::new(0),
            start_time: Instant::now(),
            metrics_tracker: Arc::new(ContinuousBatchMetrics::new()),
            cb_config,
            runtime_config,
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

    /// Snapshot queue lengths and counters for explicit scheduler trace artifacts.
    pub fn trace_snapshot(&self) -> ContinuousSchedulerTraceSnapshot {
        let waiting_queue_len = self.waiting_queue.read().len();
        let prefill_queue_len = self.prefill_queue.read().len();
        let decode_queue_len = self.decode_queue.read().len();
        let preempted_queue_len = self.preempted_requests.read().len();

        ContinuousSchedulerTraceSnapshot {
            current_iteration: self.current_iteration.load(Ordering::Relaxed),
            waiting_queue_len,
            prefill_queue_len,
            decode_queue_len,
            preempted_queue_len,
            active_len: prefill_queue_len + decode_queue_len,
            completed_total: self.completed_counter.load(Ordering::Relaxed),
            failed_total: self.failed_counter.load(Ordering::Relaxed),
            cancelled_total: self.cancelled_counter.load(Ordering::Relaxed),
            preempted_total: self.preempted_counter.load(Ordering::Relaxed),
            admitted_total: self.admitted_counter.load(Ordering::Relaxed),
            capacity_deferred_total: self.capacity_deferred_counter.load(Ordering::Relaxed),
            capacity_backpressure_admit_limit: self.capacity_backpressure_admit_limit(),
            capacity_blocked_waiting_len: self.capacity_blocked_waiting_len(),
        }
    }

    /// Return the scheduler phase for trace-only plan classification.
    pub fn trace_phase(&self, request_id: &RequestId) -> Option<RequestPhase> {
        self.request_index.read().get(request_id).copied()
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
            req.capacity_deferred_until_release_epoch = 0;
            let started_at = chrono::Utc::now();
            let wait_us = started_at
                .signed_duration_since(req.inner.submitted_at)
                .num_microseconds()
                .unwrap_or(0)
                .max(0) as u64;
            req.inner.started_at = Some(started_at);
            self.total_wait_time_us
                .fetch_add(wait_us, Ordering::Relaxed);
            self.admitted_counter.fetch_add(1, Ordering::Relaxed);

            request_index.insert(request_id.clone(), RequestPhase::Prefilling);
            prefill_queue.push_back(req);

            debug!("Promoted request {} to prefill queue", request_id);
            true
        } else {
            false
        }
    }

    fn capacity_backpressure_admit_limit(&self) -> Option<usize> {
        let limit = self.capacity_backpressure_limit.load(Ordering::Relaxed);
        if limit == NO_CAPACITY_BACKPRESSURE_LIMIT {
            None
        } else {
            Some(limit.max(1))
        }
    }

    fn capacity_blocked_waiting_len(&self) -> usize {
        if self.active_count() == 0 {
            return 0;
        }
        let release_epoch = self.capacity_release_epoch.load(Ordering::Relaxed);
        self.waiting_queue
            .read()
            .iter()
            .filter(|req| req.capacity_deferred_until_release_epoch > release_epoch)
            .count()
    }

    fn record_capacity_defer_feedback(&self, attempted_prefill_width: usize) {
        self.capacity_deferred_counter
            .fetch_add(1, Ordering::Relaxed);

        let iteration = self.current_iteration.load(Ordering::Relaxed);
        let previous_iteration = self
            .capacity_backpressure_iteration
            .swap(iteration, Ordering::Relaxed);
        if previous_iteration == iteration {
            return;
        }

        let max_running = self.config.max_running_requests.max(1);
        let proposed = attempted_prefill_width
            .max(1)
            .div_ceil(2)
            .max(1)
            .min(max_running);
        let _ = self.capacity_backpressure_limit.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |current| {
                let current = if current == NO_CAPACITY_BACKPRESSURE_LIMIT {
                    max_running
                } else {
                    current.max(1).min(max_running)
                };
                let next = proposed.min(current).max(1);
                if next >= max_running {
                    Some(NO_CAPACITY_BACKPRESSURE_LIMIT)
                } else {
                    Some(next)
                }
            },
        );
    }

    fn record_resource_progress(&self) {
        let max_running = self.config.max_running_requests.max(1);
        let current = self.capacity_backpressure_limit.load(Ordering::Relaxed);
        if current == NO_CAPACITY_BACKPRESSURE_LIMIT {
            return;
        }

        let current = current.max(1).min(max_running);
        let grown = current.saturating_mul(2).min(max_running);
        let next = if grown >= max_running {
            NO_CAPACITY_BACKPRESSURE_LIMIT
        } else {
            grown.max(1)
        };
        self.capacity_backpressure_limit
            .store(next, Ordering::Relaxed);
    }

    fn record_capacity_release_progress(&self) {
        self.capacity_release_epoch.fetch_add(1, Ordering::Relaxed);
        self.record_resource_progress();
    }

    /// Move a capacity-deferred prefill back to the waiting queue.
    ///
    /// The engine uses this when it could not allocate physical KV or
    /// recurrent state for a prefill. Leaving the request in `prefill_queue`
    /// would make `next_batch` schedule the same un-runnable work every
    /// iteration, which can starve decode and spin the scheduler.
    pub fn defer_prefill_to_waiting(&self, request_id: &RequestId) -> bool {
        let mut prefill_queue = self.prefill_queue.write();
        let mut waiting_queue = self.waiting_queue.write();
        let mut request_index = self.request_index.write();
        let attempted_prefill_width = prefill_queue.len();

        if let Some(pos) = prefill_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let mut req = prefill_queue.remove(pos).unwrap();
            req.phase = RequestPhase::Waiting;
            req.inner.state = RequestState::Waiting;
            req.inner.started_at = None;
            req.last_iteration = self.current_iteration.load(Ordering::Relaxed);
            request_index.insert(request_id.clone(), RequestPhase::Waiting);
            waiting_queue.push_back(req);
            self.record_capacity_defer_feedback(attempted_prefill_width);
            debug!("Deferred prefill request {} back to waiting", request_id);
            true
        } else {
            false
        }
    }

    /// Move a capacity-deferred decode request back to waiting for KV recompute.
    ///
    /// The engine calls this after releasing the request's physical KV/cache
    /// state. Logical output lives in the engine sequence state; scheduler
    /// token counters are reset so the next prefill rebuilds from that logical
    /// context instead of resuming the stale physical decode phase.
    pub fn defer_decode_to_waiting_for_capacity(
        &self,
        request_id: &RequestId,
        attempted_decode_width: usize,
    ) -> bool {
        let mut decode_queue = self.decode_queue.write();
        let mut waiting_queue = self.waiting_queue.write();
        let mut request_index = self.request_index.write();

        if let Some(mut req) = decode_queue.remove(request_id) {
            req.phase = RequestPhase::Waiting;
            req.inner.state = RequestState::Waiting;
            req.inner.started_at = None;
            req.prefill_tokens = 0;
            req.decode_tokens = 0;
            req.kv_blocks.clear();
            req.chunked_prefill = false;
            req.prefill_chunk_offset = 0;
            req.capacity_deferred_until_release_epoch = self
                .capacity_release_epoch
                .load(Ordering::Relaxed)
                .saturating_add(1);
            req.last_iteration = self.current_iteration.load(Ordering::Relaxed);
            request_index.insert(request_id.clone(), RequestPhase::Waiting);
            waiting_queue.push_back(req);
            self.record_capacity_defer_feedback(attempted_decode_width.max(1));
            debug!("Deferred decode request {} back to waiting", request_id);
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

    fn initial_prefill_token_estimate(&self, req: &ContinuousBatchRequest) -> usize {
        if !self.runtime_config.prompt_token_estimate {
            return self.cb_config.prefill_chunk_size;
        }

        self.prompt_token_estimate(req)
            .unwrap_or(self.cb_config.prefill_chunk_size)
    }

    fn prompt_token_estimate(&self, req: &ContinuousBatchRequest) -> Option<usize> {
        req.inner
            .request
            .metadata
            .get(PROMPT_TOKENS_METADATA_KEY)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .filter(|&v| v > 0)
    }

    fn default_active_decode_prefill_chunk(&self) -> usize {
        self.cb_config.prefill_chunk_size.div_ceil(8).max(1)
    }

    fn decode_pressure_prefill_cap_threshold(&self, hint: &BatchHint) -> usize {
        hint.max_batch_size
            .min(self.cb_config.max_decode_batch)
            .min(self.config.max_running_requests)
            .max(1)
            .div_ceil(2)
            .max(1)
    }

    fn active_decode_prefill_chunk_for_iteration(
        &self,
        hint: &BatchHint,
        scheduled_decode_count: usize,
    ) -> Option<usize> {
        if scheduled_decode_count == 0 {
            return None;
        }
        if let Some(chunk) = self.runtime_config.active_decode_prefill_chunk {
            return Some(chunk);
        }
        if scheduled_decode_count < self.decode_pressure_prefill_cap_threshold(hint) {
            return None;
        }
        Some(self.default_active_decode_prefill_chunk())
    }

    fn maybe_active_decode_prefill_chunk(
        &self,
        req: &ContinuousBatchRequest,
        active_decode_prefill_chunk: Option<usize>,
    ) -> Option<usize> {
        let chunk = active_decode_prefill_chunk?;
        if !req.chunked_prefill && self.decoding_count() == 0 {
            return None;
        }
        Some(chunk)
    }

    fn remaining_prefill_tokens(&self, req: &ContinuousBatchRequest) -> usize {
        if req.prefill_tokens == 0 {
            self.initial_prefill_token_estimate(req)
        } else {
            req.prefill_tokens.saturating_sub(req.prefill_chunk_offset)
        }
    }

    fn chunked_prefill_budget_tokens(&self, req: &ContinuousBatchRequest, chunk: usize) -> usize {
        let remaining = if req.prefill_tokens == 0 {
            self.prompt_token_estimate(req)
                .unwrap_or(self.cb_config.prefill_chunk_size)
        } else {
            req.prefill_tokens.saturating_sub(req.prefill_chunk_offset)
        };
        chunk.min(remaining).max(1)
    }

    fn prefill_budget_tokens(
        &self,
        req: &ContinuousBatchRequest,
        active_decode_prefill_chunk: Option<usize>,
        prefill_step_chunk: Option<usize>,
        step_tokens_remaining: usize,
    ) -> usize {
        if step_tokens_remaining == 0 {
            return 0;
        }
        if let Some(chunk) =
            self.maybe_active_decode_prefill_chunk(req, active_decode_prefill_chunk)
        {
            let chunk = prefill_step_chunk
                .map(|step_chunk| step_chunk.min(chunk))
                .unwrap_or(chunk);
            return self
                .chunked_prefill_budget_tokens(req, chunk)
                .min(step_tokens_remaining)
                .max(1);
        }

        let remaining = self.remaining_prefill_tokens(req);
        if let Some(chunk) = prefill_step_chunk {
            return self
                .chunked_prefill_budget_tokens(req, chunk)
                .min(step_tokens_remaining)
                .max(1);
        }
        if self.cb_config.enable_chunked_prefill {
            remaining.min(step_tokens_remaining).max(1)
        } else {
            remaining.max(1)
        }
    }

    fn active_decode_prefill_budget_tokens(
        &self,
        hint: &BatchHint,
        scheduled_decode_count: usize,
        active_decode_prefill_chunk: Option<usize>,
        prefill_step_chunk: Option<usize>,
    ) -> Option<usize> {
        let chunk = active_decode_prefill_chunk?;
        let chunk = prefill_step_chunk
            .map(|step_chunk| step_chunk.min(chunk))
            .unwrap_or(chunk);
        if scheduled_decode_count == 0 {
            return None;
        }

        let remaining_step_tokens = hint.max_tokens.saturating_sub(scheduled_decode_count);
        let free_batch_slots = hint.max_batch_size.saturating_sub(scheduled_decode_count);
        if remaining_step_tokens == 0 || free_batch_slots == 0 {
            return Some(0);
        }

        let prefill_backlog = self.prefilling_count().saturating_add(self.waiting_count());
        if prefill_backlog == 0 {
            return Some(0);
        }

        // vLLM's scheduler uses the per-step token budget dynamically rather
        // than a fixed "N prefill chunks while decoding" rule. Ferrum keeps a
        // conservative mixed-prefill guard only once decode pressure is high,
        // scaling by same-iteration batch headroom.
        let max_mixed_prefill_chunks = self.cb_config.max_prefill_batch.div_ceil(2).max(1);
        let scaled_chunks = free_batch_slots
            .saturating_mul(max_mixed_prefill_chunks)
            .div_ceil(hint.max_batch_size.max(1))
            .max(1);
        let target_chunks = scaled_chunks
            .min(max_mixed_prefill_chunks)
            .min(prefill_backlog)
            .min(free_batch_slots);

        Some(
            chunk
                .saturating_mul(target_chunks)
                .min(remaining_step_tokens),
        )
    }

    fn add_prefill_requests_to_batch(
        &self,
        hint: &BatchHint,
        batch_requests: &mut Vec<ScheduledRequest>,
        total_tokens: &mut usize,
        scheduled_request_ids: &mut HashSet<RequestId>,
        active_decode_prefill_tokens_remaining: &mut Option<usize>,
        active_decode_prefill_chunk: Option<usize>,
        prefill_step_chunk: Option<usize>,
    ) {
        if batch_requests.len() >= hint.max_batch_size || *total_tokens >= hint.max_tokens {
            return;
        }

        let prefill_queue = self.prefill_queue.read();
        for req in prefill_queue.iter() {
            if batch_requests.len() >= hint.max_batch_size {
                break;
            }
            if scheduled_request_ids.contains(&req.inner.request.id) {
                continue;
            }

            let mut step_tokens_remaining = hint.max_tokens.saturating_sub(*total_tokens);
            if let Some(remaining) = active_decode_prefill_tokens_remaining.as_ref() {
                step_tokens_remaining = step_tokens_remaining.min(*remaining);
            }
            let prefill_chunk_tokens = self.prefill_budget_tokens(
                req,
                active_decode_prefill_chunk,
                prefill_step_chunk,
                step_tokens_remaining,
            );
            // Skip fully-prefilled requests that are still in the queue
            // (they'll be promoted by mark_prefill_chunk_processed on the
            // next iteration boundary).
            if prefill_chunk_tokens == 0 {
                continue;
            }
            if let Some(remaining) = active_decode_prefill_tokens_remaining.as_mut() {
                if *remaining == 0 {
                    break;
                }
            }

            if *total_tokens + prefill_chunk_tokens <= hint.max_tokens {
                let mut scheduled = req.inner.clone();
                scheduled.tokens_processed = req.total_tokens();
                scheduled.tokens_to_process = Some(prefill_chunk_tokens);
                scheduled_request_ids.insert(scheduled.request.id.clone());
                batch_requests.push(scheduled);
                *total_tokens += prefill_chunk_tokens;
                if let Some(remaining) = active_decode_prefill_tokens_remaining.as_mut() {
                    *remaining = remaining.saturating_sub(prefill_chunk_tokens);
                }
            }
        }
    }

    /// Create batch plan for current iteration
    fn create_iteration_batch(&self, hint: BatchHint) -> Option<BatchPlan> {
        let iteration = self.current_iteration.fetch_add(1, Ordering::Relaxed);
        self.metrics_tracker.record_iteration();

        let mut batch_requests = Vec::new();
        let mut scheduled_request_ids = HashSet::new();
        let mut total_tokens = 0;
        let prefill_first_target = self
            .runtime_config
            .prefill_first_until_active
            .map(|target| {
                target
                    .min(hint.max_batch_size)
                    .min(self.cb_config.max_decode_batch)
            })
            .unwrap_or(0);
        let decoding_count = self.decoding_count();
        let active_count = self.active_count();
        let capacity_backpressure_active = self.capacity_backpressure_admit_limit().is_some();
        let skip_decode_for_prefill_first = prefill_first_target > 0
            && decoding_count < prefill_first_target
            && active_count < prefill_first_target
            && !(capacity_backpressure_active && decoding_count > 0)
            && (self.prefilling_count() > 0 || self.waiting_count() > 0);

        // First, collect decode requests (they have priority). The opt-in
        // fill-first experiment skips decodes until the active decode cohort
        // reaches the requested target, reducing early mixed prefill+decode
        // spikes in c=32 closed-loop runs.
        if !skip_decode_for_prefill_first {
            let decode_queue = self.decode_queue.read();
            for (_, req) in decode_queue.iter() {
                if batch_requests.len() >= hint.max_batch_size {
                    break;
                }

                // Each decode step is 1 token per request
                if total_tokens < hint.max_tokens {
                    let mut scheduled = req.inner.clone();
                    scheduled.tokens_processed = req.total_tokens();
                    scheduled.tokens_to_process = Some(1);
                    scheduled_request_ids.insert(scheduled.request.id.clone());
                    batch_requests.push(scheduled);
                    total_tokens += 1;
                }
            }
            drop(decode_queue);
        }
        let scheduled_decode_count = batch_requests.len();
        let active_decode_prefill_chunk =
            self.active_decode_prefill_chunk_for_iteration(&hint, scheduled_decode_count);
        let prefill_step_chunk = self.runtime_config.prefill_step_chunk;
        let mut active_decode_prefill_tokens_remaining = self.active_decode_prefill_budget_tokens(
            &hint,
            scheduled_decode_count,
            active_decode_prefill_chunk,
            prefill_step_chunk,
        );
        let allow_capacity_deferred_mixed_recompute = scheduled_decode_count > 0
            && active_decode_prefill_chunk.is_some()
            && active_decode_prefill_tokens_remaining.unwrap_or(0) > 0;

        // Then, add prefill requests up to the per-iter token budget.
        // Phase 3: `max_prefill_batch=8` no longer caps the count —
        // the only budget is `hint.max_tokens` (= EngineConfig's
        // `max_num_batched_tokens`, default 4096). Decodes contribute
        // 1 token each; prefill chunks contribute their chunk size.
        // This is what lets the Qwen3MoE `unified_forward` path
        // activate for cohort prefills (m_total must stay ≤ scratch
        // max_tokens, which is pre-allocated to the same budget).
        self.add_prefill_requests_to_batch(
            &hint,
            &mut batch_requests,
            &mut total_tokens,
            &mut scheduled_request_ids,
            &mut active_decode_prefill_tokens_remaining,
            active_decode_prefill_chunk,
            prefill_step_chunk,
        );

        // Check if we should admit new requests from waiting queue
        let waiting_queue = self.waiting_queue.read();
        let active_capacity = self
            .config
            .max_running_requests
            .saturating_sub(self.active_count());
        let decode_capacity = self
            .cb_config
            .max_decode_batch
            .saturating_sub(self.decoding_count());
        let available_slots = active_capacity.min(decode_capacity);
        let available_slots = self
            .capacity_backpressure_admit_limit()
            .map(|limit| available_slots.min(limit))
            .unwrap_or(available_slots);
        let active_count_for_capacity_wait = self.active_count();
        let capacity_release_epoch = self.capacity_release_epoch.load(Ordering::Relaxed);

        let requests_to_admit: Vec<RequestId> = waiting_queue
            .iter()
            .filter(|r| {
                active_count_for_capacity_wait == 0
                    || r.capacity_deferred_until_release_epoch <= capacity_release_epoch
                    || allow_capacity_deferred_mixed_recompute
            })
            .take(available_slots)
            .map(|r| r.inner.request.id.clone())
            .collect();
        drop(waiting_queue);

        // Promote waiting requests to prefill
        for req_id in requests_to_admit {
            self.promote_to_prefill(&req_id);
        }

        // vLLM's scheduler spends the remaining per-step token budget on
        // waiting requests after running requests. Mirror that behavior so
        // newly admitted prefills do not wait an extra iteration just because
        // the current batch already contains decode work.
        self.add_prefill_requests_to_batch(
            &hint,
            &mut batch_requests,
            &mut total_tokens,
            &mut scheduled_request_ids,
            &mut active_decode_prefill_tokens_remaining,
            active_decode_prefill_chunk,
            prefill_step_chunk,
        );

        // FERRUM_SCHED_NONE_PROF=1: log when next_batch is about to return SOME.
        if self.runtime_config.scheduler_none_prof && !batch_requests.is_empty() {
            use std::sync::atomic::AtomicU64;
            static SOME_PROF_N: AtomicU64 = AtomicU64::new(0);
            let n = SOME_PROF_N.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(64) {
                let d_len = self.decode_queue.read().len();
                let p_len = self.prefill_queue.read().len();
                let w_len = self.waiting_queue.read().len();
                eprintln!(
                    "[sched-some] n={} returning_batch={} | decode_queue={} prefill_queue={} waiting_queue={}",
                    n,
                    batch_requests.len(),
                    d_len,
                    p_len,
                    w_len,
                );
            }
        }
        if batch_requests.is_empty() {
            // FERRUM_SCHED_NONE_PROF=1: log why we returned None. Rate-limited.
            if self.runtime_config.scheduler_none_prof {
                use std::sync::atomic::AtomicU64;
                static NONE_PROF_N: AtomicU64 = AtomicU64::new(0);
                let n = NONE_PROF_N.fetch_add(1, Ordering::Relaxed);
                if n.is_multiple_of(512) {
                    let d_len = self.decode_queue.read().len();
                    let p_len = self.prefill_queue.read().len();
                    let w_len = self.waiting_queue.read().len();
                    let d_count = self.decoding_count();
                    eprintln!(
                        "[sched-none] n={} decode_queue={} prefill_queue={} waiting_queue={} decoding_count={} hint.max_batch={}",
                        n,
                        d_len,
                        p_len,
                        w_len,
                        d_count,
                        hint.max_batch_size,
                    );
                }
            }
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
                recurrent_state_bytes: 0,
                recurrent_state_slots: 0,
                compute_units: 1,
            },
            created_at: chrono::Utc::now(),
        })
    }

    /// Mark a request as having completed prefill
    pub fn mark_prefill_complete(&self, request_id: &RequestId, tokens: usize) {
        let mut prefill_queue = self.prefill_queue.write();
        let mut found = false;
        if let Some(pos) = prefill_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let req = &mut prefill_queue[pos];
            req.prefill_tokens = tokens;
            req.prefill_chunk_offset = tokens;
            req.chunked_prefill = false;
            found = true;
        }
        drop(prefill_queue);

        // Promote to decode
        self.promote_to_decode(request_id);
        if found {
            self.record_resource_progress();
        }
    }

    /// Mark a chunk of prefill as processed. Used by engines that split a
    /// long prompt across multiple iterations to reduce TTFT under load.
    ///
    /// `total_prompt_tokens` should be the full prompt length — pass it
    /// every call (idempotent: the scheduler uses the last value it sees).
    /// `chunk_tokens` is how many tokens were processed *this iteration*.
    ///
    /// Returns `true` if the request is now fully prefilled and has been
    /// promoted to the decode queue.
    pub fn mark_prefill_chunk_processed(
        &self,
        request_id: &RequestId,
        total_prompt_tokens: usize,
        chunk_tokens: usize,
    ) -> bool {
        let mut prefill_queue = self.prefill_queue.write();
        let mut fully_done = false;
        let mut made_progress = false;
        if let Some(pos) = prefill_queue
            .iter()
            .position(|r| r.inner.request.id == *request_id)
        {
            let req = &mut prefill_queue[pos];
            req.prefill_tokens = total_prompt_tokens;
            req.chunked_prefill = true;
            req.prefill_chunk_offset = req
                .prefill_chunk_offset
                .saturating_add(chunk_tokens)
                .min(total_prompt_tokens);
            fully_done = req.prefill_chunk_offset >= total_prompt_tokens;
            made_progress = chunk_tokens > 0;
        }
        drop(prefill_queue);

        if fully_done {
            self.promote_to_decode(request_id);
        }
        if made_progress {
            self.record_resource_progress();
        }
        fully_done
    }

    /// Update decode progress for a request
    pub fn update_decode_progress(&self, request_id: &RequestId, tokens_generated: usize) {
        let mut decode_queue = self.decode_queue.write();
        if let Some(req) = decode_queue.get_mut(request_id) {
            req.decode_tokens = tokens_generated;
            req.last_iteration = self.current_iteration.load(Ordering::Relaxed);
        }
        // Decode progress consumes KV capacity; only actual prefill progress or
        // completion should relax capacity backpressure.
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
                ferrum_types::FinishReason::EOS
                | ferrum_types::FinishReason::Stop
                | ferrum_types::FinishReason::Length => {
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
            self.record_capacity_release_progress();

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
                match response.finish_reason {
                    ferrum_types::FinishReason::EOS
                    | ferrum_types::FinishReason::Stop
                    | ferrum_types::FinishReason::Length => {
                        self.completed_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        self.failed_counter.fetch_add(1, Ordering::Relaxed);
                        warn!(
                            "Request {} completed with error during prefill: {:?}",
                            request_id, response.finish_reason
                        );
                    }
                }
                self.record_capacity_release_progress();
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
                self.record_capacity_release_progress();
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
                self.record_capacity_release_progress();
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
        let admitted_count = self.admitted_counter.load(Ordering::Relaxed);
        let total_wait_time_us = self.total_wait_time_us.load(Ordering::Relaxed);

        let uptime_secs = self.start_time.elapsed().as_secs_f64();
        let throughput = if uptime_secs > 0.0 {
            completed_count as f64 / uptime_secs
        } else {
            0.0
        };

        let queue_utilization = waiting_count as f32 / self.config.max_waiting_requests as f32;
        let avg_wait_time_ms = if admitted_count > 0 {
            total_wait_time_us as f64 / admitted_count as f64 / 1000.0
        } else {
            0.0
        };

        ferrum_types::SchedulerStats {
            waiting_requests: waiting_count,
            running_requests: running_count,
            preempted_requests: preempted_count as usize,
            completed_requests: completed_count,
            failed_requests: failed_count,
            cancelled_requests: cancelled_count,
            avg_wait_time_ms,
            avg_execution_time_ms: 0.0,
            throughput_rps: throughput,
            queue_utilization,
        }
    }

    fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    fn request_state(&self, request_id: &RequestId) -> Option<RequestState> {
        self.request_index
            .read()
            .get(request_id)
            .copied()
            .map(|phase| match phase {
                RequestPhase::Waiting => RequestState::Waiting,
                RequestPhase::Prefilling | RequestPhase::Decoding => RequestState::Running,
                RequestPhase::Completed => RequestState::Completed,
                RequestPhase::Preempted => RequestState::Preempted,
                RequestPhase::Cancelled => RequestState::Cancelled,
            })
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
            self.record_capacity_release_progress();

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

            self.decode_queue.write().insert(request_id.clone(), req);
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
            api_request: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    fn create_test_request_with_prompt_tokens(
        priority: Priority,
        prompt_tokens: usize,
    ) -> InferenceRequest {
        create_test_request(priority).with_metadata(
            PROMPT_TOKENS_METADATA_KEY,
            serde_json::Value::from(prompt_tokens as u64),
        )
    }

    fn enqueue_waiting(scheduler: &ContinuousBatchScheduler, request: InferenceRequest) {
        let request_id = request.id.clone();
        scheduler
            .waiting_queue
            .write()
            .push_back(ContinuousBatchRequest::new(request));
        scheduler
            .request_index
            .write()
            .insert(request_id, RequestPhase::Waiting);
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
    async fn trace_snapshot_reports_queue_counters_and_phase() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request = create_test_request(Priority::Normal);
        let request_id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        let before = scheduler.trace_snapshot();
        assert_eq!(before.waiting_queue_len, 1);
        assert_eq!(before.active_len, 0);
        assert_eq!(before.admitted_total, 0);
        assert_eq!(
            scheduler.trace_phase(&request_id),
            Some(RequestPhase::Waiting)
        );

        let batch = scheduler.next_batch(BatchHint::simple(4)).await.unwrap();
        assert_eq!(batch.size(), 1);

        let after = scheduler.trace_snapshot();
        assert_eq!(after.waiting_queue_len, 0);
        assert_eq!(after.prefill_queue_len, 1);
        assert_eq!(after.active_len, 1);
        assert_eq!(after.admitted_total, 1);
        assert_eq!(
            scheduler.trace_phase(&request_id),
            Some(RequestPhase::Prefilling)
        );
    }

    #[tokio::test]
    async fn defer_prefill_to_waiting_frees_active_slot_without_cancelling() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request = create_test_request(Priority::Normal);
        let request_id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        let batch = scheduler.next_batch(BatchHint::simple(4)).await.unwrap();
        assert_eq!(batch.size(), 1);
        let active = scheduler.trace_snapshot();
        assert_eq!(active.waiting_queue_len, 0);
        assert_eq!(active.prefill_queue_len, 1);
        assert_eq!(active.active_len, 1);

        assert!(scheduler.defer_prefill_to_waiting(&request_id));
        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 1);
        assert_eq!(deferred.prefill_queue_len, 0);
        assert_eq!(deferred.active_len, 0);
        assert_eq!(deferred.cancelled_total, 0);
        assert_eq!(
            scheduler.trace_phase(&request_id),
            Some(RequestPhase::Waiting)
        );
        assert_eq!(
            scheduler.request_state(&request_id),
            Some(RequestState::Waiting)
        );
    }

    #[test]
    fn capacity_defer_halves_next_waiting_admission_width() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 4);
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            assert!(scheduler.defer_prefill_to_waiting(request_id));
        }

        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 4);
        assert_eq!(deferred.active_len, 0);
        assert_eq!(deferred.capacity_deferred_total, 4);
        assert_eq!(deferred.capacity_backpressure_admit_limit, Some(2));

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            second_batch.requests.len(),
            2,
            "capacity-deferred waiting requests should not be immediately re-admitted at the failed width"
        );
        let after = scheduler.trace_snapshot();
        assert_eq!(after.waiting_queue_len, 2);
        assert_eq!(after.prefill_queue_len, 2);
        assert_eq!(after.active_len, 2);
        assert_eq!(after.admitted_total, 6);
        assert_eq!(after.capacity_backpressure_admit_limit, Some(2));
    }

    #[test]
    fn decode_capacity_defer_requeues_for_recompute_without_cancelling() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 4);
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }
        assert_eq!(scheduler.trace_snapshot().decode_queue_len, 4);

        for request_id in &first_ids {
            assert!(scheduler.defer_decode_to_waiting_for_capacity(request_id, 4));
        }

        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 4);
        assert_eq!(deferred.decode_queue_len, 0);
        assert_eq!(deferred.active_len, 0);
        assert_eq!(deferred.cancelled_total, 0);
        assert_eq!(deferred.capacity_deferred_total, 4);
        assert_eq!(deferred.capacity_backpressure_admit_limit, Some(2));
        for request_id in &first_ids {
            assert_eq!(
                scheduler.trace_phase(request_id),
                Some(RequestPhase::Waiting)
            );
            assert_eq!(
                scheduler.request_state(request_id),
                Some(RequestState::Waiting)
            );
        }

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            second_batch.requests.len(),
            2,
            "capacity-deferred decodes should recompute at a lower admission width"
        );
        let after = scheduler.trace_snapshot();
        assert_eq!(after.waiting_queue_len, 2);
        assert_eq!(after.prefill_queue_len, 2);
        assert_eq!(after.active_len, 2);
        assert_eq!(after.capacity_backpressure_admit_limit, Some(2));
    }

    #[test]
    fn capacity_deferred_decode_recomputes_as_bounded_mixed_prefill_under_decode_pressure() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));
        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 1);
        assert_eq!(deferred.decode_queue_len, 3);
        assert_eq!(deferred.active_len, 3);
        assert_eq!(deferred.capacity_blocked_waiting_len, 1);
        assert_eq!(deferred.capacity_backpressure_admit_limit, Some(2));

        let decode_only = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let scheduled_ids: HashSet<RequestId> = decode_only
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert_eq!(decode_only.requests.len(), 4);
        assert!(
            scheduled_ids.contains(&first_ids[0]),
            "capacity-deferred recompute should use bounded mixed prefill budget under decode pressure"
        );
        let prefill_tokens = decode_only
            .requests
            .iter()
            .find(|request| request.request.id == first_ids[0])
            .and_then(|request| request.tokens_to_process);
        assert_eq!(
            prefill_tokens,
            Some(64),
            "the recompute prefill should still be capped by the mixed-prefill token budget"
        );
        assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 0);
    }

    #[tokio::test]
    async fn capacity_deferred_decode_waits_for_release_without_bounded_mixed_budget() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..2 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            scheduler.mark_prefill_complete(request_id, 128);
        }

        assert!(scheduler.defer_decode_to_waiting_for_capacity(&first_ids[0], 4));
        let deferred = scheduler.trace_snapshot();
        assert_eq!(deferred.waiting_queue_len, 1);
        assert_eq!(deferred.decode_queue_len, 1);
        assert_eq!(deferred.active_len, 1);
        assert_eq!(deferred.capacity_blocked_waiting_len, 1);

        let response = InferenceResponse {
            request_id: first_ids[1].clone(),
            text: String::new(),
            tokens: Vec::new(),
            finish_reason: ferrum_types::FinishReason::Length,
            usage: ferrum_types::TokenUsage::new(0, 0),
            latency_ms: 0,
            created_at: chrono::Utc::now(),
            metadata: Default::default(),
            api_response: None,
        };
        scheduler
            .complete(first_ids[1].clone(), &response)
            .await
            .unwrap();

        let after_release = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_ids: HashSet<RequestId> = after_release
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert!(
            scheduled_ids.contains(&first_ids[0]),
            "a real capacity release should make the deferred recompute eligible again"
        );
        assert_eq!(scheduler.trace_snapshot().capacity_blocked_waiting_len, 0);
    }

    #[test]
    fn capacity_backpressure_grows_after_prefill_progress() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        for request in &first_batch.requests {
            assert!(scheduler.defer_prefill_to_waiting(&request.request.id));
        }
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(2)
        );

        let second_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(second_batch.requests.len(), 2);
        let progressed_id = second_batch.requests[0].request.id.clone();
        assert!(!scheduler.mark_prefill_chunk_processed(&progressed_id, 128, 1));
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            None,
            "real prefill progress should relax the capacity backpressure window"
        );

        let third_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            third_batch.requests.len(),
            4,
            "after progress, remaining waiting prefills can use available active slots again"
        );
        let after = scheduler.trace_snapshot();
        assert_eq!(after.waiting_queue_len, 0);
        assert_eq!(after.active_len, 4);
    }

    #[tokio::test]
    async fn capacity_backpressure_survives_cancel_without_token_progress() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let first_batch = scheduler.create_iteration_batch(hint).unwrap();
        let first_ids: Vec<_> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for request_id in &first_ids {
            assert!(scheduler.defer_prefill_to_waiting(request_id));
        }
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(2)
        );

        assert!(scheduler.cancel(first_ids[0].clone()).await.unwrap());
        let after_cancel = scheduler.trace_snapshot();
        assert_eq!(after_cancel.cancelled_total, 1);
        assert_eq!(
            after_cancel.capacity_backpressure_admit_limit,
            Some(2),
            "cancellation frees a slot but is not evidence that the failed admission width now fits"
        );
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

    #[test]
    fn prompt_token_metadata_expands_prefill_admission() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });

        for _ in 0..16 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let hint = BatchHint {
            max_batch_size: 32,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 8);

        for request in first_batch.requests {
            scheduler.mark_prefill_complete(&request.request.id, 256);
        }

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(mixed_batch.requests.len(), 16);
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, 2048 * 16);
    }

    #[test]
    fn prompt_token_metadata_can_be_disabled_for_prefill_admission() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: false,
            ..SchedulerConfig::default()
        });

        for _ in 0..16 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 32,
                max_tokens: 2048,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();
        assert_eq!(batch.requests.len(), 4);
    }

    #[test]
    fn scheduler_runtime_config_is_captured_at_construction() {
        let mut config = SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        };
        let scheduler = ContinuousBatchScheduler::new(config.clone());
        config.prompt_token_estimate = false;

        for _ in 0..16 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 32,
                max_tokens: 2048,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();
        assert_eq!(batch.requests.len(), 8);
    }

    #[test]
    fn max_running_requests_limits_waiting_admission() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 1,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });

        for _ in 0..3 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 1);
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.waiting_count(), 2);

        let active_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(active_batch.requests.len(), 1);
        assert_eq!(
            active_batch.requests[0].request.id, first_batch.requests[0].request.id,
            "scheduler must not admit another waiting request while the active cap is full"
        );
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.waiting_count(), 2);
    }

    #[test]
    fn newly_admitted_prefill_uses_remaining_budget_with_decode() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 4,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let first = create_test_request_with_prompt_tokens(Priority::Normal, 2);
        let first_id = first.id.clone();
        enqueue_waiting(&scheduler, first);
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 1);
        scheduler.mark_prefill_complete(&first_id, 2);

        let second = create_test_request_with_prompt_tokens(Priority::Normal, 2);
        let second_id = second.id.clone();
        enqueue_waiting(&scheduler, second);

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        let ids: HashSet<RequestId> = mixed_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        assert_eq!(mixed_batch.requests.len(), 2);
        assert!(
            ids.contains(&first_id),
            "decode request should remain scheduled"
        );
        assert!(
            ids.contains(&second_id),
            "newly admitted prefill should use remaining same-iteration budget"
        );
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, 3 * 16);
    }

    #[test]
    fn default_scheduler_caps_mixed_prefill_only_under_decode_pressure() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 8,
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let first = create_test_request_with_prompt_tokens(Priority::Normal, 256);
        let first_id = first.id.clone();
        enqueue_waiting(&scheduler, first);
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 1);
        scheduler.mark_prefill_complete(&first_id, 256);

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let low_decode_pressure = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(
            low_decode_pressure.requests.len(),
            5,
            "small decode cohorts should use remaining token budget to build concurrency"
        );
        assert_eq!(
            low_decode_pressure.resource_requirements.gpu_memory,
            (1 + 4 * 256) * 16
        );
        assert_eq!(
            low_decode_pressure
                .requests
                .iter()
                .filter(|request| request.request.id != first_id)
                .map(|request| request.tokens_to_process)
                .collect::<Vec<_>>(),
            vec![Some(256), Some(256), Some(256), Some(256)]
        );

        for request in low_decode_pressure
            .requests
            .iter()
            .filter(|request| request.request.id != first_id)
        {
            scheduler.mark_prefill_complete(&request.request.id, 256);
        }
        assert_eq!(scheduler.decoding_count(), 5);

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let high_decode_pressure = scheduler.create_iteration_batch(hint).unwrap();
        let prefill_tokens: Vec<_> = high_decode_pressure
            .requests
            .iter()
            .filter(|request| request.tokens_to_process != Some(1))
            .map(|request| request.tokens_to_process)
            .collect();
        assert_eq!(
            high_decode_pressure.requests.len(),
            7,
            "high decode pressure should admit only bounded partial prefills"
        );
        assert_eq!(prefill_tokens, vec![Some(64), Some(64)]);
        assert_eq!(
            high_decode_pressure.resource_requirements.gpu_memory,
            (5 + 128) * 16
        );
    }

    #[test]
    fn max_batched_tokens_limits_prefill_admission_by_prompt_tokens() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 8,
                max_tokens: 512,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();

        assert_eq!(batch.requests.len(), 2);
        assert_eq!(batch.resource_requirements.gpu_memory, 512 * 16);
        assert_eq!(
            scheduler.prefilling_count(),
            4,
            "max_tokens limits the emitted iteration batch, not waiting-to-prefill promotion"
        );
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn long_prefill_uses_remaining_step_budget_instead_of_fixed_chunk() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            ..SchedulerConfig::default()
        });

        for _ in 0..2 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 1536),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 8,
                max_tokens: 2048,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();

        assert_eq!(batch.requests.len(), 2);
        assert_eq!(
            batch
                .requests
                .iter()
                .map(|request| request.tokens_to_process)
                .collect::<Vec<_>>(),
            vec![Some(1536), Some(512)]
        );
        assert_eq!(batch.resource_requirements.gpu_memory, 2048 * 16);
    }

    #[test]
    fn prefill_first_until_active_skips_early_decodes() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prefill_first_until_active: Some(4),
            ..SchedulerConfig::default()
        });

        for _ in 0..3 {
            enqueue_waiting(&scheduler, create_test_request(Priority::Normal));
        }

        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 2);
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for id in &first_ids {
            scheduler.mark_prefill_complete(id, 512);
        }
        assert_eq!(scheduler.decoding_count(), 2);

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(second_batch.requests.len(), 1);
        assert!(
            second_batch
                .requests
                .iter()
                .all(|request| !first_ids.contains(&request.request.id)),
            "fill-first should schedule more prefills before decoding early requests"
        );
    }

    #[test]
    fn prefill_first_until_active_resumes_decodes_at_active_target() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prefill_first_until_active: Some(4),
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(&scheduler, create_test_request(Priority::Normal));
        }

        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 1024,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 2);
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for id in &first_ids {
            scheduler.mark_prefill_complete(id, 512);
        }

        assert_eq!(scheduler.decoding_count(), 2);
        assert_eq!(scheduler.prefilling_count(), 2);
        assert_eq!(scheduler.active_count(), 4);

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_decodes = second_batch
            .requests
            .iter()
            .filter(|request| {
                first_ids.contains(&request.request.id) && request.tokens_to_process == Some(1)
            })
            .count();
        assert_eq!(
            scheduled_decodes, 2,
            "fill-first must not starve decode once the active target is reached"
        );
    }

    #[test]
    fn capacity_backpressure_disables_prefill_first_decode_skip() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            prefill_first_until_active: Some(4),
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 512,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 4);
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for id in first_ids.iter().take(3) {
            scheduler.mark_prefill_complete(id, 128);
        }
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[3]));

        let after_defer = scheduler.trace_snapshot();
        assert_eq!(after_defer.decode_queue_len, 3);
        assert_eq!(after_defer.waiting_queue_len, 1);
        assert_eq!(after_defer.active_len, 3);
        assert_eq!(after_defer.capacity_backpressure_admit_limit, Some(1));

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_decodes = second_batch
            .requests
            .iter()
            .filter(|request| {
                first_ids[..3].contains(&request.request.id) && request.tokens_to_process == Some(1)
            })
            .count();
        assert_eq!(
            scheduled_decodes, 3,
            "capacity backpressure must let decode-ready requests run instead of repeatedly admitting a capacity-blocked prefill"
        );
    }

    #[test]
    fn decode_progress_does_not_relax_capacity_backpressure() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 4,
            prompt_token_estimate: true,
            prefill_first_until_active: Some(4),
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 128),
            );
        }

        let hint = BatchHint {
            max_batch_size: 4,
            max_tokens: 512,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };
        let first_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(first_batch.requests.len(), 4);
        let first_ids: Vec<RequestId> = first_batch
            .requests
            .iter()
            .map(|request| request.request.id.clone())
            .collect();
        for id in first_ids.iter().take(3) {
            scheduler.mark_prefill_complete(id, 128);
        }
        assert!(scheduler.defer_prefill_to_waiting(&first_ids[3]));
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(1)
        );

        for id in first_ids.iter().take(3) {
            scheduler.update_decode_progress(id, 1);
        }
        assert_eq!(
            scheduler.trace_snapshot().capacity_backpressure_admit_limit,
            Some(1),
            "decode progress consumes KV capacity and must not reopen waiting admission"
        );

        let second_batch = scheduler.create_iteration_batch(hint).unwrap();
        let scheduled_decodes = second_batch
            .requests
            .iter()
            .filter(|request| {
                first_ids[..3].contains(&request.request.id) && request.tokens_to_process == Some(1)
            })
            .count();
        assert_eq!(
            scheduled_decodes, 3,
            "capacity backpressure should keep fill-first from skipping decode after decode progress"
        );
        assert_eq!(
            second_batch.requests.len(),
            4,
            "one capacity-limited prefill may still refill the remaining batch slot"
        );
    }

    #[test]
    fn prefill_step_chunk_caps_prefill_first_batches() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            prefill_first_until_active: Some(4),
            prefill_step_chunk: Some(128),
            ..SchedulerConfig::default()
        });

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 512),
            );
        }

        let batch = scheduler
            .create_iteration_batch(BatchHint {
                max_batch_size: 8,
                max_tokens: 2048,
                target_latency_ms: None,
                available_memory: None,
                resource_constraints: Default::default(),
            })
            .unwrap();

        assert_eq!(batch.requests.len(), 4);
        assert_eq!(
            batch
                .requests
                .iter()
                .map(|request| request.tokens_to_process)
                .collect::<Vec<_>>(),
            vec![Some(128), Some(128), Some(128), Some(128)]
        );
        assert_eq!(batch.resource_requirements.gpu_memory, (4 * 128) * 16);
    }

    #[test]
    fn active_decode_prefill_chunk_only_caps_when_decode_is_active() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            active_decode_prefill_chunk: Some(64),
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 2,
            max_tokens: 512,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let first = create_test_request_with_prompt_tokens(Priority::Normal, 256);
        let first_id = first.id.clone();
        enqueue_waiting(&scheduler, first);
        let initial_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(initial_batch.requests.len(), 1);
        assert_eq!(initial_batch.resource_requirements.gpu_memory, 256 * 16);
        scheduler.mark_prefill_complete(&first_id, 256);

        enqueue_waiting(
            &scheduler,
            create_test_request_with_prompt_tokens(Priority::Normal, 256),
        );
        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(mixed_batch.requests.len(), 2);
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, (1 + 64) * 16);
    }

    #[test]
    fn active_decode_prefill_chunk_caps_aggregate_mixed_prefill_tokens() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            active_decode_prefill_chunk: Some(64),
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let first = create_test_request_with_prompt_tokens(Priority::Normal, 256);
        let first_id = first.id.clone();
        enqueue_waiting(&scheduler, first);
        let initial_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(initial_batch.requests.len(), 1);
        scheduler.mark_prefill_complete(&first_id, 256);

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            mixed_batch.requests.len(),
            5,
            "low decode pressure should admit more prefill chunks from batch headroom"
        );
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, (1 + 256) * 16);
        assert_eq!(
            scheduler.prefilling_count(),
            4,
            "waiting requests may be promoted, but scheduling must respect the mixed-prefill budget"
        );
    }

    #[test]
    fn active_decode_prefill_budget_scales_down_with_decode_pressure() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            prompt_token_estimate: true,
            active_decode_prefill_chunk: Some(64),
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 8,
            max_tokens: 2048,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let mut decode_ids = Vec::new();
        for _ in 0..6 {
            let request = create_test_request_with_prompt_tokens(Priority::Normal, 128);
            decode_ids.push(request.id.clone());
            enqueue_waiting(&scheduler, request);
        }
        let initial_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(initial_batch.requests.len(), 6);
        for id in &decode_ids {
            scheduler.mark_prefill_complete(id, 128);
        }

        for _ in 0..4 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        assert_eq!(
            mixed_batch.requests.len(),
            7,
            "high decode pressure should admit only one prefill chunk"
        );
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, (6 + 64) * 16);
    }

    #[test]
    fn active_decode_prefill_budget_uses_effective_step_chunk_for_aggregate_cap() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig {
            max_running_requests: 32,
            prompt_token_estimate: true,
            active_decode_prefill_chunk: Some(8192),
            prefill_step_chunk: Some(64),
            ..SchedulerConfig::default()
        });
        let hint = BatchHint {
            max_batch_size: 32,
            max_tokens: 8192,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: Default::default(),
        };

        let mut decode_ids = Vec::new();
        for _ in 0..7 {
            let request = create_test_request_with_prompt_tokens(Priority::Normal, 128);
            decode_ids.push(request.id.clone());
            enqueue_waiting(&scheduler, request);
        }
        let initial_batch = scheduler.create_iteration_batch(hint.clone()).unwrap();
        assert_eq!(initial_batch.requests.len(), 7);
        for id in &decode_ids {
            scheduler.mark_prefill_complete(id, 128);
        }

        for _ in 0..25 {
            enqueue_waiting(
                &scheduler,
                create_test_request_with_prompt_tokens(Priority::Normal, 256),
            );
        }

        let mixed_batch = scheduler.create_iteration_batch(hint).unwrap();
        let prefill_tokens: Vec<_> = mixed_batch
            .requests
            .iter()
            .filter(|request| request.tokens_to_process != Some(1))
            .map(|request| request.tokens_to_process)
            .collect();
        assert_eq!(
            mixed_batch.requests.len(),
            11,
            "large explicit active chunks must not bypass the prefill-step aggregate cap"
        );
        assert_eq!(prefill_tokens, vec![Some(64), Some(64), Some(64), Some(64)]);
        assert_eq!(mixed_batch.resource_requirements.gpu_memory, (7 + 256) * 16);
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

    #[tokio::test]
    async fn metrics_track_queue_wait_time_on_admission() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        scheduler
            .submit(create_test_request(Priority::Normal))
            .await
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));

        let batch = scheduler.next_batch(BatchHint::simple(1)).await;
        assert!(batch.is_some());

        let metrics = scheduler.metrics();
        assert_eq!(metrics.waiting_requests, 0);
        assert_eq!(metrics.running_requests, 1);
        assert!(
            metrics.avg_wait_time_ms >= 1.0,
            "expected non-zero wait time, got {}",
            metrics.avg_wait_time_ms
        );
    }

    #[test]
    fn test_cb_request_states() {
        let request = create_test_request(Priority::Normal);
        let cb_req = ContinuousBatchRequest::new(request);

        assert_eq!(cb_req.phase, RequestPhase::Waiting);
        assert!(!cb_req.is_active());
        assert!(!cb_req.is_finished());
    }

    /// Chunked prefill state machine: advance across multiple iterations,
    /// transition Prefilling → Decoding only on the final chunk.
    #[tokio::test]
    async fn chunked_prefill_advances_across_iterations() {
        let cb_cfg = ContinuousBatchConfig {
            enable_chunked_prefill: true,
            prefill_chunk_size: 128,
            ..ContinuousBatchConfig::default()
        };
        let scheduler =
            ContinuousBatchScheduler::with_cb_config(SchedulerConfig::default(), cb_cfg);

        let request = create_test_request(Priority::Normal);
        let req_id = request.id.clone();
        scheduler.submit(request).await.unwrap();

        // Pull a batch to promote waiting → prefilling
        let _ = scheduler.next_batch(BatchHint::simple(1024)).await;
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.decoding_count(), 0);

        // Engine reports: prompt is 400 tokens, first chunk processed 128.
        // 128 < 400 → still prefilling, no phase transition.
        let done = scheduler.mark_prefill_chunk_processed(&req_id, 400, 128);
        assert!(!done, "first chunk should not finish prefill");
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.decoding_count(), 0);

        // Second chunk — 256 of 400.
        let done = scheduler.mark_prefill_chunk_processed(&req_id, 400, 128);
        assert!(!done);
        assert_eq!(scheduler.prefilling_count(), 1);
        assert_eq!(scheduler.decoding_count(), 0);

        // Final chunk — covers remaining 144 (saturates at 400).
        let done = scheduler.mark_prefill_chunk_processed(&req_id, 400, 200);
        assert!(done, "last chunk should complete prefill");
        assert_eq!(scheduler.prefilling_count(), 0);
        assert_eq!(scheduler.decoding_count(), 1);
    }

    /// Legacy one-shot `mark_prefill_complete` still promotes correctly and
    /// sets offset to total (so the request won't be double-scheduled for
    /// more prefill if somehow still in the queue).
    #[tokio::test]
    async fn mark_prefill_complete_sets_offset_to_total() {
        let scheduler = ContinuousBatchScheduler::new(SchedulerConfig::default());
        let request = create_test_request(Priority::Normal);
        let req_id = request.id.clone();
        scheduler.submit(request).await.unwrap();
        let _ = scheduler.next_batch(BatchHint::simple(1024)).await;

        scheduler.mark_prefill_complete(&req_id, 256);

        assert_eq!(scheduler.prefilling_count(), 0);
        assert_eq!(scheduler.decoding_count(), 1);
    }
}
