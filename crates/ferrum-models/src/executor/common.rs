//! Shared executor status initialization.

use std::time::Instant;

/// Default executor status (all executors return the same thing).
pub fn default_executor_status() -> ferrum_execution::model_executor::ExecutorStatus {
    use ferrum_execution::model_executor::*;
    ExecutorStatus {
        state: ExecutorState::Ready,
        is_ready: true,
        current_batch_size: 0,
        prefill_operations: 0,
        decode_operations: 0,
        avg_prefill_time_ms: 0.0,
        avg_decode_time_ms: 0.0,
        memory_usage: ExecutorMemoryUsage {
            allocated_bytes: 0,
            used_bytes: 0,
            peak_bytes: 0,
            utilization_percent: 0.0,
        },
        last_operation: Some(Instant::now()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_executor_status_is_ready() {
        let status = default_executor_status();
        assert!(status.is_ready);
        assert_eq!(
            status.state,
            ferrum_execution::model_executor::ExecutorState::Ready
        );
    }
}
