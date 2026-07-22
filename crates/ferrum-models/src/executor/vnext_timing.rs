use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ferrum_interfaces::vnext::StaticInitializationReceipt;

pub(super) struct StartupPhaseTimer {
    phase: &'static str,
    started: Instant,
}

impl StartupPhaseTimer {
    pub(super) fn start(phase: &'static str) -> Self {
        Self {
            phase,
            started: Instant::now(),
        }
    }

    pub(super) fn finish(self) -> u64 {
        let duration_us = self.started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        tracing::info!(
            target: "ferrum.startup",
            phase = self.phase,
            duration_us,
            "vNext startup phase completed"
        );
        duration_us
    }
}

pub(super) fn log_static_initialization_receipt(receipt: &StaticInitializationReceipt) {
    tracing::info!(
        target: "ferrum.startup",
        initialized_resource_count = receipt.initialized_resource_count(),
        uploaded_component_count = receipt.uploaded_component_count(),
        uploaded_bytes = receipt.uploaded_bytes(),
        imported_component_count = receipt.imported_component_count(),
        imported_bytes = receipt.imported_bytes(),
        upload_command_count = receipt.upload_command_count(),
        submission_batch_count = receipt.submission_batch_count(),
        total_duration_us = receipt.total_duration_us(),
        setup_duration_us = receipt.setup_duration_us(),
        source_materialization_duration_us = receipt.source_materialization_duration_us(),
        device_encode_duration_us = receipt.device_encode_duration_us(),
        device_import_duration_us = receipt.device_import_duration_us(),
        submission_wait_duration_us = receipt.submission_wait_duration_us(),
        import_seal_duration_us = receipt.import_seal_duration_us(),
        slowest_component_id = receipt
            .slowest_component_id()
            .map(ToString::to_string)
            .as_deref(),
        slowest_component_materialization_duration_us = receipt
            .slowest_component_materialization_duration_us(),
        "vNext static initialization completed"
    );
}

#[derive(Default)]
pub(super) struct AtomicDurationMetrics {
    samples: AtomicU64,
    total_ns: AtomicU64,
    max_ns: AtomicU64,
}

impl AtomicDurationMetrics {
    pub(super) fn start(&self) -> AtomicDurationTimer<'_> {
        AtomicDurationTimer {
            metrics: self,
            started: Instant::now(),
        }
    }

    pub(super) fn start_if(&self, enabled: bool) -> Option<AtomicDurationTimer<'_>> {
        enabled.then(|| self.start())
    }

    pub(super) fn record(&self, duration: Duration) {
        let nanoseconds = duration.as_nanos().min(u64::MAX as u128) as u64;
        self.samples.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(nanoseconds, Ordering::Relaxed);
        self.max_ns.fetch_max(nanoseconds, Ordering::Relaxed);
    }

    pub(super) fn reset(&self) {
        self.samples.store(0, Ordering::Relaxed);
        self.total_ns.store(0, Ordering::Relaxed);
        self.max_ns.store(0, Ordering::Relaxed);
    }

    pub(super) fn snapshot(&self) -> serde_json::Value {
        let samples = self.samples.load(Ordering::Relaxed);
        let total_ns = self.total_ns.load(Ordering::Relaxed);
        let max_ns = self.max_ns.load(Ordering::Relaxed);
        serde_json::json!({
            "samples": samples,
            "total_ns": total_ns,
            "average_us": if samples == 0 {
                0.0
            } else {
                total_ns as f64 / samples as f64 / 1_000.0
            },
            "max_us": max_ns as f64 / 1_000.0,
        })
    }
}

pub(super) struct AtomicDurationTimer<'a> {
    metrics: &'a AtomicDurationMetrics,
    started: Instant,
}

impl Drop for AtomicDurationTimer<'_> {
    fn drop(&mut self) {
        self.metrics.record(self.started.elapsed());
    }
}

#[cfg(test)]
mod tests {
    use super::AtomicDurationMetrics;
    use std::time::Duration;

    #[test]
    fn duration_metrics_are_constant_space_and_saturating() {
        let metrics = AtomicDurationMetrics::default();
        metrics.record(Duration::from_micros(2));
        metrics.record(Duration::from_micros(6));

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot["samples"], 2);
        assert_eq!(snapshot["total_ns"], 8_000);
        assert_eq!(snapshot["average_us"], 4.0);
        assert_eq!(snapshot["max_us"], 6.0);
    }

    #[test]
    fn disabled_duration_metrics_do_not_read_or_record_a_sample() {
        let metrics = AtomicDurationMetrics::default();

        assert!(metrics.start_if(false).is_none());
        assert_eq!(metrics.snapshot()["samples"], 0);

        drop(metrics.start_if(true));
        assert_eq!(metrics.snapshot()["samples"], 1);
    }

    #[test]
    fn reset_removes_cold_path_samples() {
        let metrics = AtomicDurationMetrics::default();
        metrics.record(Duration::from_micros(4));

        metrics.reset();

        assert_eq!(metrics.snapshot()["samples"], 0);
        assert_eq!(metrics.snapshot()["total_ns"], 0);
        assert_eq!(metrics.snapshot()["max_us"], 0.0);
    }
}
