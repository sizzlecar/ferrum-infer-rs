//! Shared boilerplate for modality-only engines (embedding / transcription / TTS).
//!
//! These engines wrap a per-modality executor and expose its functionality
//! through the `InferenceEngine` trait. They do NOT do text generation,
//! so `infer / infer_stream` return Err and the orchestration accessors
//! (status / metrics / health / shutdown) return inert defaults.
//!
//! Each modality engine called these with copy-pasted bodies (~80 lines
//! of identical boilerplate per engine). This module hosts the canonical
//! versions so the 3 fake engines can shrink.

use ferrum_types::{EngineMetrics, EngineStatus, HealthStatus, MemoryUsage};

/// Inert engine status — modality engines aren't request-driven, so most
/// fields are zeroed. `is_ready: true` because the executor is loaded.
pub fn inert_status() -> EngineStatus {
    EngineStatus {
        is_ready: true,
        loaded_models: vec![],
        active_requests: 0,
        queued_requests: 0,
        memory_usage: MemoryUsage {
            total_bytes: 0,
            used_bytes: 0,
            free_bytes: 0,
            gpu_memory_bytes: None,
            cpu_memory_bytes: None,
            cache_memory_bytes: 0,
            utilization_percent: 0.0,
        },
        uptime_seconds: 0,
        last_heartbeat: chrono::Utc::now(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    }
}

/// Inert engine metrics — all zeros, defaults on nested types.
pub fn inert_metrics() -> EngineMetrics {
    EngineMetrics {
        total_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        avg_request_latency_ms: 0.0,
        p95_request_latency_ms: 0.0,
        p99_request_latency_ms: 0.0,
        throughput_rps: 0.0,
        tokens_per_second: 0.0,
        queue_metrics: Default::default(),
        resource_utilization: Default::default(),
        error_stats: Default::default(),
        performance_breakdown: Default::default(),
    }
}

/// Health: always healthy if executor loaded.
pub fn inert_health() -> HealthStatus {
    HealthStatus::healthy()
}
