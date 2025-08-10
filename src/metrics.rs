//! Metrics and monitoring for the LLM inference engine
//!
//! This module provides performance metrics, telemetry, and monitoring
//! capabilities for the inference engine.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Metrics collector for the inference engine
pub struct MetricsCollector {
    counters: Arc<RwLock<HashMap<String, u64>>>,
    gauges: Arc<RwLock<HashMap<String, f64>>>,
    histograms: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    start_time: Instant,
}

/// Metrics snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub timestamp: u64,
    pub uptime_seconds: f64,
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histograms: HashMap<String, HistogramStats>,
}

/// Histogram statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        }
    }

    /// Increment a counter metric
    pub fn increment_counter(&self, name: &str, value: u64) {
        let mut counters = self.counters.write();
        *counters.entry(name.to_string()).or_insert(0) += value;
    }

    /// Set a gauge metric
    pub fn set_gauge(&self, name: &str, value: f64) {
        let mut gauges = self.gauges.write();
        gauges.insert(name.to_string(), value);
    }

    /// Record a value in a histogram
    pub fn record_histogram(&self, name: &str, value: f64) {
        let mut histograms = self.histograms.write();
        histograms
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
    }

    /// Record the duration of an operation
    pub fn record_duration(&self, name: &str, duration: Duration) {
        self.record_histogram(name, duration.as_secs_f64() * 1000.0); // Convert to milliseconds
    }

    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        let counters = self.counters.read().clone();
        let gauges = self.gauges.read().clone();

        let histograms = {
            let histograms = self.histograms.read();
            histograms
                .iter()
                .map(|(name, values)| {
                    let stats = Self::calculate_histogram_stats(values);
                    (name.clone(), stats)
                })
                .collect()
        };

        MetricsSnapshot {
            timestamp: chrono::Utc::now().timestamp() as u64,
            uptime_seconds: self.start_time.elapsed().as_secs_f64(),
            counters,
            gauges,
            histograms,
        }
    }

    /// Clear all histogram data (useful for periodic cleanup)
    pub fn clear_histograms(&self) {
        let mut histograms = self.histograms.write();
        histograms.clear();
    }

    /// Calculate statistics for a histogram
    fn calculate_histogram_stats(values: &[f64]) -> HistogramStats {
        if values.is_empty() {
            return HistogramStats {
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count = sorted.len();
        let min = sorted[0];
        let max = sorted[count - 1];
        let mean = sorted.iter().sum::<f64>() / count as f64;

        let p50 = Self::percentile(&sorted, 0.5);
        let p95 = Self::percentile(&sorted, 0.95);
        let p99 = Self::percentile(&sorted, 0.99);

        HistogramStats {
            count,
            min,
            max,
            mean,
            p50,
            p95,
            p99,
        }
    }

    /// Calculate a percentile from sorted values
    fn percentile(sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        if sorted_values.len() == 1 {
            return sorted_values[0];
        }

        let index = percentile * (sorted_values.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            sorted_values[lower_index]
        } else {
            let lower_value = sorted_values[lower_index];
            let upper_value = sorted_values[upper_index];
            let weight = index - lower_index as f64;
            lower_value + weight * (upper_value - lower_value)
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// A timer for measuring operation duration
pub struct Timer {
    start: Instant,
    name: String,
    collector: Arc<MetricsCollector>,
}

impl Timer {
    /// Create a new timer
    pub fn new(name: String, collector: Arc<MetricsCollector>) -> Self {
        Self {
            start: Instant::now(),
            name,
            collector,
        }
    }

    /// Stop the timer and record the duration
    pub fn stop(self) {
        let duration = self.start.elapsed();
        self.collector.record_duration(&self.name, duration);
    }
}

/// Macro for timing code blocks
#[macro_export]
macro_rules! time_block {
    ($collector:expr, $name:expr, $block:block) => {{
        let _timer = Timer::new($name.to_string(), $collector.clone());
        let result = $block;
        // Timer automatically records when dropped
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();

        // Test counters
        collector.increment_counter("requests", 1);
        collector.increment_counter("requests", 2);
        collector.increment_counter("errors", 1);

        // Test gauges
        collector.set_gauge("memory_usage", 1024.0);
        collector.set_gauge("cpu_usage", 50.5);

        // Test histograms
        collector.record_histogram("latency", 10.5);
        collector.record_histogram("latency", 20.0);
        collector.record_histogram("latency", 15.2);

        let snapshot = collector.snapshot();

        assert_eq!(snapshot.counters.get("requests"), Some(&3));
        assert_eq!(snapshot.counters.get("errors"), Some(&1));
        assert_eq!(snapshot.gauges.get("memory_usage"), Some(&1024.0));
        assert_eq!(snapshot.gauges.get("cpu_usage"), Some(&50.5));

        let latency_stats = snapshot.histograms.get("latency").unwrap();
        assert_eq!(latency_stats.count, 3);
        assert_eq!(latency_stats.min, 10.5);
        assert_eq!(latency_stats.max, 20.0);
    }

    #[test]
    fn test_histogram_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = MetricsCollector::calculate_histogram_stats(&values);

        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert_eq!(stats.mean, 5.5);
        assert_eq!(stats.p50, 5.5); // Median of 10 values should be average of 5th and 6th elements
        assert_eq!(stats.p95, 9.5); // 95th percentile
        assert_eq!(stats.p99, 9.9); // 99th percentile
    }
}
