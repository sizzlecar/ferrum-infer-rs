//! Metrics collection and reporting for schedulers

use ferrum_types::{Priority, RequestId};
use parking_lot::RwLock;
use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

/// Real-time metrics collector for scheduler performance
pub struct SchedulerMetricsCollector {
    /// Request completion metrics
    completions: AtomicU64,
    /// Request failure metrics
    failures: AtomicU64,
    /// Request cancellation metrics
    cancellations: AtomicU64,
    /// Total wait time across all requests
    total_wait_time_ms: AtomicU64,
    /// Total execution time across all requests
    total_execution_time_ms: AtomicU64,
    /// Start time for throughput calculation
    start_time: Instant,
    /// Per-priority metrics
    priority_metrics: RwLock<HashMap<Priority, PriorityMetrics>>,
    /// Batch metrics
    batch_metrics: RwLock<BatchMetrics>,
    /// Recent wait times for percentile calculation
    recent_wait_times: RwLock<Vec<u64>>,
    /// Recent execution times for percentile calculation
    recent_execution_times: RwLock<Vec<u64>>,
}

/// Metrics specific to each priority level
#[derive(Debug, Clone, Default)]
struct PriorityMetrics {
    count: u64,
    total_wait_time_ms: u64,
    total_execution_time_ms: u64,
}

/// Batch processing metrics
#[derive(Debug, Clone, Default)]
struct BatchMetrics {
    total_batches_created: u64,
    total_batches_completed: u64,
    total_batch_formation_time_ms: u64,
    total_batch_size: u64,
    batch_count: u64,
}

impl SchedulerMetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            completions: AtomicU64::new(0),
            failures: AtomicU64::new(0),
            cancellations: AtomicU64::new(0),
            total_wait_time_ms: AtomicU64::new(0),
            total_execution_time_ms: AtomicU64::new(0),
            start_time: Instant::now(),
            priority_metrics: RwLock::new(HashMap::new()),
            batch_metrics: RwLock::new(BatchMetrics::default()),
            recent_wait_times: RwLock::new(Vec::new()),
            recent_execution_times: RwLock::new(Vec::new()),
        }
    }

    /// Record successful request completion
    pub fn record_completion(
        &self,
        wait_time: Duration,
        execution_time: Duration,
        priority: Priority,
    ) {
        let wait_time_ms = wait_time.as_millis() as u64;
        let execution_time_ms = execution_time.as_millis() as u64;

        // Update global counters
        self.completions.fetch_add(1, Ordering::Relaxed);
        self.total_wait_time_ms
            .fetch_add(wait_time_ms, Ordering::Relaxed);
        self.total_execution_time_ms
            .fetch_add(execution_time_ms, Ordering::Relaxed);

        // Update priority-specific metrics
        {
            let mut priority_metrics = self.priority_metrics.write();
            let metrics = priority_metrics.entry(priority).or_default();
            metrics.count += 1;
            metrics.total_wait_time_ms += wait_time_ms;
            metrics.total_execution_time_ms += execution_time_ms;
        }

        // Store recent times for percentile calculation (keep last 1000)
        {
            let mut recent_wait = self.recent_wait_times.write();
            recent_wait.push(wait_time_ms);
            if recent_wait.len() > 1000 {
                recent_wait.remove(0);
            }
        }

        {
            let mut recent_exec = self.recent_execution_times.write();
            recent_exec.push(execution_time_ms);
            if recent_exec.len() > 1000 {
                recent_exec.remove(0);
            }
        }
    }

    /// Record request failure
    pub fn record_failure(&self, priority: Priority) {
        self.failures.fetch_add(1, Ordering::Relaxed);

        // Update priority-specific metrics
        let mut priority_metrics = self.priority_metrics.write();
        let metrics = priority_metrics.entry(priority).or_default();
        metrics.count += 1;
    }

    /// Record request cancellation
    pub fn record_cancellation(&self, priority: Priority) {
        self.cancellations.fetch_add(1, Ordering::Relaxed);

        // Update priority-specific metrics
        let mut priority_metrics = self.priority_metrics.write();
        let metrics = priority_metrics.entry(priority).or_default();
        metrics.count += 1;
    }

    /// Record batch creation
    pub fn record_batch_created(&self, batch_size: usize, formation_time: Duration) {
        let mut batch_metrics = self.batch_metrics.write();
        batch_metrics.total_batches_created += 1;
        batch_metrics.total_batch_formation_time_ms += formation_time.as_millis() as u64;
        batch_metrics.total_batch_size += batch_size as u64;
        batch_metrics.batch_count += 1;
    }

    /// Record batch completion
    pub fn record_batch_completed(&self) {
        let mut batch_metrics = self.batch_metrics.write();
        batch_metrics.total_batches_completed += 1;
    }

    /// Get current throughput (requests per second)
    pub fn throughput_rps(&self) -> f64 {
        let elapsed = self.start_time.elapsed();
        let completions = self.completions.load(Ordering::Relaxed);

        if elapsed.as_secs_f64() > 0.0 {
            completions as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get average wait time
    pub fn avg_wait_time_ms(&self) -> f64 {
        let total_wait = self.total_wait_time_ms.load(Ordering::Relaxed);
        let completions = self.completions.load(Ordering::Relaxed);

        if completions > 0 {
            total_wait as f64 / completions as f64
        } else {
            0.0
        }
    }

    /// Get average execution time
    pub fn avg_execution_time_ms(&self) -> f64 {
        let total_exec = self.total_execution_time_ms.load(Ordering::Relaxed);
        let completions = self.completions.load(Ordering::Relaxed);

        if completions > 0 {
            total_exec as f64 / completions as f64
        } else {
            0.0
        }
    }

    /// Get P95 wait time
    pub fn p95_wait_time_ms(&self) -> f64 {
        let recent_times = self.recent_wait_times.read();
        calculate_percentile(&recent_times, 95.0)
    }

    /// Get P95 execution time
    pub fn p95_execution_time_ms(&self) -> f64 {
        let recent_times = self.recent_execution_times.read();
        calculate_percentile(&recent_times, 95.0)
    }

    /// Get average batch size
    pub fn avg_batch_size(&self) -> f32 {
        let batch_metrics = self.batch_metrics.read();
        if batch_metrics.batch_count > 0 {
            batch_metrics.total_batch_size as f32 / batch_metrics.batch_count as f32
        } else {
            0.0
        }
    }

    /// Get batch efficiency (completed / created)
    pub fn batch_efficiency(&self) -> f32 {
        let batch_metrics = self.batch_metrics.read();
        if batch_metrics.total_batches_created > 0 {
            batch_metrics.total_batches_completed as f32
                / batch_metrics.total_batches_created as f32
        } else {
            0.0
        }
    }

    /// Get average batch formation time
    pub fn avg_batch_formation_time_ms(&self) -> f64 {
        let batch_metrics = self.batch_metrics.read();
        if batch_metrics.total_batches_created > 0 {
            batch_metrics.total_batch_formation_time_ms as f64
                / batch_metrics.total_batches_created as f64
        } else {
            0.0
        }
    }

    /// Get metrics for specific priority
    pub fn priority_metrics(&self, priority: Priority) -> Option<(f64, f64)> {
        let priority_metrics = self.priority_metrics.read();
        priority_metrics.get(&priority).map(|metrics| {
            let avg_wait = if metrics.count > 0 {
                metrics.total_wait_time_ms as f64 / metrics.count as f64
            } else {
                0.0
            };
            let avg_exec = if metrics.count > 0 {
                metrics.total_execution_time_ms as f64 / metrics.count as f64
            } else {
                0.0
            };
            (avg_wait, avg_exec)
        })
    }

    /// Get total request counts
    pub fn request_counts(&self) -> (u64, u64, u64) {
        (
            self.completions.load(Ordering::Relaxed),
            self.failures.load(Ordering::Relaxed),
            self.cancellations.load(Ordering::Relaxed),
        )
    }

    /// Get batch counts
    pub fn batch_counts(&self) -> (u64, u64) {
        let batch_metrics = self.batch_metrics.read();
        (
            batch_metrics.total_batches_created,
            batch_metrics.total_batches_completed,
        )
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.completions.store(0, Ordering::Relaxed);
        self.failures.store(0, Ordering::Relaxed);
        self.cancellations.store(0, Ordering::Relaxed);
        self.total_wait_time_ms.store(0, Ordering::Relaxed);
        self.total_execution_time_ms.store(0, Ordering::Relaxed);

        self.priority_metrics.write().clear();
        *self.batch_metrics.write() = BatchMetrics::default();
        self.recent_wait_times.write().clear();
        self.recent_execution_times.write().clear();
    }
}

impl Default for SchedulerMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate percentile from a sorted list of values
fn calculate_percentile(values: &[u64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_unstable();

    let index = (percentile / 100.0 * (sorted_values.len() - 1) as f64).round() as usize;
    let index = index.min(sorted_values.len() - 1);

    sorted_values[index] as f64
}

/// Request lifecycle tracker for detailed analysis
pub struct RequestTracker {
    /// Active requests with their start times
    active_requests: RwLock<HashMap<RequestId, RequestLifecycle>>,
}

#[derive(Debug, Clone)]
struct RequestLifecycle {
    submitted_at: Instant,
    started_at: Option<Instant>,
    priority: Priority,
}

impl RequestTracker {
    /// Create new request tracker
    pub fn new() -> Self {
        Self {
            active_requests: RwLock::new(HashMap::new()),
        }
    }

    /// Track new request submission
    pub fn track_submission(&self, request_id: RequestId, priority: Priority) {
        let mut active = self.active_requests.write();
        active.insert(
            request_id,
            RequestLifecycle {
                submitted_at: Instant::now(),
                started_at: None,
                priority,
            },
        );
    }

    /// Track request start (moved from waiting to running)
    pub fn track_start(&self, request_id: &RequestId) {
        let mut active = self.active_requests.write();
        if let Some(lifecycle) = active.get_mut(request_id) {
            lifecycle.started_at = Some(Instant::now());
        }
    }

    /// Track request completion and return timing info
    pub fn track_completion(
        &self,
        request_id: &RequestId,
    ) -> Option<(Duration, Option<Duration>, Priority)> {
        let mut active = self.active_requests.write();
        active.remove(request_id).map(|lifecycle| {
            let total_time = lifecycle.submitted_at.elapsed();
            let execution_time = lifecycle.started_at.map(|start| start.elapsed());
            (total_time, execution_time, lifecycle.priority)
        })
    }

    /// Get count of currently tracked requests
    pub fn active_count(&self) -> usize {
        self.active_requests.read().len()
    }

    /// Clean up old requests (safety mechanism)
    pub fn cleanup_old_requests(&self, max_age: Duration) {
        let mut active = self.active_requests.write();
        let cutoff = Instant::now() - max_age;
        active.retain(|_, lifecycle| lifecycle.submitted_at > cutoff);
    }
}

impl Default for RequestTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_calculation() {
        let values = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

        assert_eq!(calculate_percentile(&values, 50.0), 50.0);
        assert_eq!(calculate_percentile(&values, 95.0), 100.0);
        assert_eq!(calculate_percentile(&values, 0.0), 10.0);
    }

    #[test]
    fn test_metrics_collector_basic() {
        let collector = SchedulerMetricsCollector::new();

        collector.record_completion(
            Duration::from_millis(100),
            Duration::from_millis(50),
            Priority::High,
        );

        assert_eq!(collector.avg_wait_time_ms(), 100.0);
        assert_eq!(collector.avg_execution_time_ms(), 50.0);

        let (completed, failed, cancelled) = collector.request_counts();
        assert_eq!(completed, 1);
        assert_eq!(failed, 0);
        assert_eq!(cancelled, 0);
    }
}
