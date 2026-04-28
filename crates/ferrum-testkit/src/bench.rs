//! Benchmark result types and utilities.
//!
//! Provides structured benchmark output (JSON), percentile calculation,
//! and comparison helpers for performance regression testing.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Structured benchmark result for JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub version: String,
    pub timestamp: String,
    pub model: String,
    pub backend: String,
    pub config: BenchmarkConfig,
    pub results: BenchmarkMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub concurrency: usize,
    pub max_tokens: usize,
    pub rounds: usize,
    pub prompt_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub throughput_tps: StatSummary,
    pub ttft_ms: PercentileSummary,
    pub tpot_ms: PercentileSummary,
    pub total_tokens: usize,
    pub total_time_ms: f64,
    pub requests_completed: usize,
    pub requests_failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatSummary {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileSummary {
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub peak_kv_blocks_used: usize,
    pub total_kv_blocks: usize,
    pub peak_kv_utilization: f64,
}

impl BenchmarkResult {
    /// Write result to JSON file.
    pub fn write_json(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::other(format!("JSON serialize: {e}")))?;
        std::fs::write(path, json)
    }

    /// Write result to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

// ── Percentile calculation ───────────────────────────────────────────────

/// Calculate percentile from sorted data. Uses linear interpolation.
pub fn percentile(data: &mut [f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if data.len() == 1 {
        return data[0];
    }
    let idx = (p / 100.0) * (data.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    if hi >= data.len() {
        return data[data.len() - 1];
    }
    let frac = idx - lo as f64;
    data[lo] * (1.0 - frac) + data[hi] * frac
}

/// Build a PercentileSummary from raw latency samples.
pub fn percentile_summary(samples: &[f64]) -> PercentileSummary {
    if samples.is_empty() {
        return PercentileSummary {
            mean: 0.0,
            p50: 0.0,
            p90: 0.0,
            p95: 0.0,
            p99: 0.0,
        };
    }
    let mut data = samples.to_vec();
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    PercentileSummary {
        mean,
        p50: percentile(&mut data, 50.0),
        p90: percentile(&mut data, 90.0),
        p95: percentile(&mut data, 95.0),
        p99: percentile(&mut data, 99.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&mut data, 50.0) - 5.5).abs() < 0.01);
        assert!((percentile(&mut data, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile(&mut data, 100.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile_summary() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let ps = percentile_summary(&samples);
        assert!((ps.mean - 50.5).abs() < 0.01);
        assert!((ps.p50 - 50.5).abs() < 0.6);
        assert!((ps.p99 - 99.0).abs() < 1.0);
    }
}
