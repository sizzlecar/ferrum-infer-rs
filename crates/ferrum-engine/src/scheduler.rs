//! Request scheduler implementation with priority and fairness

use async_trait::async_trait;
use ferrum_core::{
    Scheduler, InferenceRequest, RequestId, InferenceResponse,
    ScheduledBatch, SchedulerStats, Result, Error,
    RequestState, ScheduledRequest,
};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug, warn};

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of waiting requests
    pub max_waiting_requests: usize,
    
    /// Maximum number of running requests
    pub max_running_requests: usize,
    
    /// Enable preemption
    pub enable_preemption: bool,
    
    /// Scheduling policy
    pub policy: SchedulingPolicy,
}

/// Scheduling policy
#[derive(Debug, Clone)]
pub enum SchedulingPolicy {
    /// First come first serve
    FCFS,
    /// Priority-based scheduling
    Priority,
    /// Fair scheduling with user quotas
    Fair,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_waiting_requests: 1000,
            max_running_requests: 256,
            enable_preemption: true,
            policy: SchedulingPolicy::Priority,
        }
    }
}

/// Fair scheduler implementation
pub struct FairScheduler {
    config: SchedulerConfig,
    waiting_queue: Arc<RwLock<BinaryHeap<PrioritizedRequest>>>,
    running_requests: Arc<RwLock<HashMap<RequestId, ScheduledRequest>>>,
    preempted_requests: Arc<RwLock<VecDeque<ScheduledRequest>>>,
    completed_requests: Arc<RwLock<HashMap<RequestId, InferenceResponse>>>,
    stats: Arc<RwLock<SchedulerStatsInternal>>,
}

/// Request with priority for heap ordering
#[derive(Clone)]
struct PrioritizedRequest {
    request: InferenceRequest,
    arrival_time: std::time::Instant,
}

impl Eq for PrioritizedRequest {}

impl PartialEq for PrioritizedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request.id == other.request.id
    }
}

impl Ord for PrioritizedRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier arrival time
        other.request.priority.cmp(&self.request.priority)
            .then_with(|| self.arrival_time.cmp(&other.arrival_time))
    }
}

impl PartialOrd for PrioritizedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Internal scheduler statistics
struct SchedulerStatsInternal {
    total_scheduled: u64,
    total_completed: u64,
    total_failed: u64,
    total_preempted: u64,
    total_wait_time_ms: u64,
    total_execution_time_ms: u64,
}

impl FairScheduler {
    /// Create a new fair scheduler
    pub fn new(config: SchedulerConfig) -> Self {
        info!("Initializing FairScheduler with policy: {:?}", config.policy);
        
        Self {
            config,
            waiting_queue: Arc::new(RwLock::new(BinaryHeap::new())),
            running_requests: Arc::new(RwLock::new(HashMap::new())),
            preempted_requests: Arc::new(RwLock::new(VecDeque::new())),
            completed_requests: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SchedulerStatsInternal {
                total_scheduled: 0,
                total_completed: 0,
                total_failed: 0,
                total_preempted: 0,
                total_wait_time_ms: 0,
                total_execution_time_ms: 0,
            })),
        }
    }
    
    /// Check if we can schedule more requests
    fn can_schedule_more(&self) -> bool {
        let running = self.running_requests.read().len();
        running < self.config.max_running_requests
    }
    
    /// Select next request to schedule based on policy
    fn select_next_request(&self) -> Option<InferenceRequest> {
        // First check preempted requests (they have priority)
        if let Some(preempted) = self.preempted_requests.write().pop_front() {
            debug!("Resuming preempted request {:?}", preempted.request.id);
            return Some(preempted.request);
        }
        
        // Then check waiting queue
        let mut queue = self.waiting_queue.write();
        if let Some(prioritized) = queue.pop() {
            return Some(prioritized.request);
        }
        
        None
    }
    
    /// Preempt lower priority requests if needed
    fn preempt_if_needed(&self, new_request: &InferenceRequest) -> Result<()> {
        if !self.config.enable_preemption {
            return Ok(());
        }
        
        let running = self.running_requests.read();
        
        // Find lowest priority running request
        let lowest_priority = running.values()
            .filter(|r| r.request.priority < new_request.priority)
            .min_by_key(|r| r.request.priority);
        
        if let Some(to_preempt) = lowest_priority {
            let request_id = to_preempt.request.id.clone();
            drop(running);
            
            warn!("Preempting request {:?} for higher priority request {:?}", 
                  request_id, new_request.id);
            
            // Move to preempted queue
            let mut running = self.running_requests.write();
            if let Some(preempted) = running.remove(&request_id) {
                self.preempted_requests.write().push_back(preempted);
                self.stats.write().total_preempted += 1;
            }
        }
        
        Ok(())
    }
}

#[async_trait]
impl Scheduler for FairScheduler {
    async fn schedule_request(&self, request: InferenceRequest) -> Result<RequestId> {
        let request_id = request.id.clone();
        
        // Check queue limits
        {
            let queue_size = self.waiting_queue.read().len();
            if queue_size >= self.config.max_waiting_requests {
                return Err(Error::invalid_request("Request queue is full"));
            }
        }
        
        debug!("Scheduling request {:?} with priority {:?}", request_id, request.priority);
        
        // Update stats
        self.stats.write().total_scheduled += 1;
        
        // Add to waiting queue
        let prioritized = PrioritizedRequest {
            request,
            arrival_time: std::time::Instant::now(),
        };
        
        self.waiting_queue.write().push(prioritized);
        
        Ok(request_id)
    }
    
    async fn get_next_batch(&self) -> Option<ScheduledBatch> {
        if !self.can_schedule_more() {
            return None;
        }
        
        let mut batch_requests = Vec::new();
        let batch_id = ferrum_core::BatchId(uuid::Uuid::new_v4());
        
        // Try to fill a batch up to max running requests
        while self.can_schedule_more() && batch_requests.len() < 32 {
            if let Some(request) = self.select_next_request() {
                // Check if we need to preempt
                if self.running_requests.read().len() >= self.config.max_running_requests {
                    if let Err(e) = self.preempt_if_needed(&request) {
                        warn!("Failed to preempt for request {:?}: {}", request.id, e);
                        // Put request back in queue
                        self.waiting_queue.write().push(PrioritizedRequest {
                            request,
                            arrival_time: std::time::Instant::now(),
                        });
                        break;
                    }
                }
                
                let scheduled = ScheduledRequest {
                    request: request.clone(),
                    state: RequestState::Running,
                    allocated_blocks: Vec::new(), // Will be allocated by cache manager
                };
                
                // Add to running requests
                self.running_requests.write().insert(request.id.clone(), scheduled.clone());
                batch_requests.push(scheduled);
            } else {
                break;
            }
        }
        
        if batch_requests.is_empty() {
            None
        } else {
            debug!("Created batch {:?} with {} requests", batch_id, batch_requests.len());
            Some(ScheduledBatch {
                batch_id,
                requests: batch_requests,
                created_at: chrono::Utc::now(),
            })
        }
    }
    
    async fn preempt_request(&self, request_id: RequestId) -> Result<()> {
        let mut running = self.running_requests.write();
        
        if let Some(mut request) = running.remove(&request_id) {
            request.state = RequestState::Preempted;
            self.preempted_requests.write().push_back(request);
            self.stats.write().total_preempted += 1;
            
            info!("Preempted request {:?}", request_id);
            Ok(())
        } else {
            Err(Error::not_found(format!("Request {:?} not found in running requests", request_id)))
        }
    }
    
    async fn complete_request(&self, request_id: RequestId, response: InferenceResponse) -> Result<()> {
        let mut running = self.running_requests.write();
        
        if let Some(mut request) = running.remove(&request_id) {
            request.state = RequestState::Completed;
            self.completed_requests.write().insert(request_id.clone(), response);
            self.stats.write().total_completed += 1;
            
            debug!("Completed request {:?}", request_id);
            Ok(())
        } else {
            Err(Error::not_found(format!("Request {:?} not found in running requests", request_id)))
        }
    }
    
    async fn get_stats(&self) -> SchedulerStats {
        let stats = self.stats.read();
        let waiting = self.waiting_queue.read().len();
        let running = self.running_requests.read().len();
        let preempted = self.preempted_requests.read().len();
        
        let avg_wait_time = if stats.total_scheduled > 0 {
            stats.total_wait_time_ms as f64 / stats.total_scheduled as f64
        } else {
            0.0
        };
        
        let avg_execution_time = if stats.total_completed > 0 {
            stats.total_execution_time_ms as f64 / stats.total_completed as f64
        } else {
            0.0
        };
        
        SchedulerStats {
            waiting_requests: waiting,
            running_requests: running,
            preempted_requests: preempted,
            completed_requests: stats.total_completed,
            failed_requests: stats.total_failed,
            avg_wait_time_ms: avg_wait_time,
            avg_execution_time_ms: avg_execution_time,
        }
    }
}
