use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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

    pub(super) fn record(&self, duration: Duration) {
        let nanoseconds = duration.as_nanos().min(u64::MAX as u128) as u64;
        self.samples.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(nanoseconds, Ordering::Relaxed);
        self.max_ns.fetch_max(nanoseconds, Ordering::Relaxed);
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
}
