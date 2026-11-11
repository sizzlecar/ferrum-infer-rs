//! Bounded observation timing and cadence over the actual workload phase.
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default, Serialize)]
pub struct ImageResourceQueryTiming {
    pub queries: u64,
    pub total_query_ms: f64,
    pub maximum_query_ms: f64,
    pub periodic_queries: u64,
    pub maximum_periodic_query_ms: f64,
    pub overlong_periodic_queries: u64,
    pub maximum_query_start_gap_ms: f64,
    pub first_query_started_offset_ms: Option<f64>,
    pub last_query_started_offset_ms: Option<f64>,
    pub last_query_finished_offset_ms: Option<f64>,
    /// The peak was observed somewhere in this query interval, not necessarily
    /// at its start. Procfs reads and driver queries are not atomic snapshots.
    pub peak_query_started_offset_ms: Option<f64>,
    pub peak_query_finished_offset_ms: Option<f64>,
}

impl ImageResourceQueryTiming {
    pub(super) fn record(
        &mut self,
        epoch: Instant,
        started: Instant,
        finished: Instant,
        new_peak: bool,
        periodic_interval: Option<Duration>,
    ) {
        let start = milliseconds(started.saturating_duration_since(epoch));
        let end = milliseconds(finished.saturating_duration_since(epoch));
        let elapsed = milliseconds(finished.saturating_duration_since(started));
        self.queries += 1;
        self.total_query_ms += elapsed;
        self.maximum_query_ms = self.maximum_query_ms.max(elapsed);
        if let Some(interval) = periodic_interval {
            self.periodic_queries += 1;
            self.maximum_periodic_query_ms = self.maximum_periodic_query_ms.max(elapsed);
            if finished.saturating_duration_since(started) > interval {
                self.overlong_periodic_queries += 1;
            }
        }
        if let Some(last) = self.last_query_started_offset_ms {
            self.maximum_query_start_gap_ms = self.maximum_query_start_gap_ms.max(start - last);
        }
        self.first_query_started_offset_ms.get_or_insert(start);
        self.last_query_started_offset_ms = Some(start);
        self.last_query_finished_offset_ms = Some(end);
        if new_peak {
            self.peak_query_started_offset_ms = Some(start);
            self.peak_query_finished_offset_ms = Some(end);
        }
    }
}

pub(super) fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Boundary probes are outside this clock. A periodic observation consumes one
/// due slot; any intervening slots remain missed even if the query succeeds.
pub(super) struct PhaseCadence {
    interval: Duration,
    due: Instant,
    skipped: u64,
}

impl PhaseCadence {
    pub(super) fn new(phase_started: Instant, interval: Duration) -> Self {
        Self {
            interval,
            due: phase_started + interval,
            skipped: 0,
        }
    }

    pub(super) fn due(&self) -> Instant {
        self.due
    }

    pub(super) fn observed(&mut self, started: Instant) {
        while self.due + self.interval <= started {
            self.skipped += 1;
            self.due += self.interval;
        }
        self.due += self.interval;
    }

    pub(super) fn finish(mut self, phase_finished: Instant) -> u64 {
        // A deadline exactly at completion is outside the half-open workload
        // phase. Never count time spent waiting for an outstanding/final query.
        while self.due < phase_finished {
            self.skipped += 1;
            self.due += self.interval;
        }
        self.skipped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_baseline_and_final_queries_do_not_create_workload_deadlines() {
        let epoch = Instant::now();
        let phase_started = epoch + Duration::from_secs(8);
        let cadence = PhaseCadence::new(phase_started, Duration::from_secs(2));
        assert_eq!(cadence.due(), epoch + Duration::from_secs(10));
        // The final query can finish at 17s; only the phase ending at 9s counts.
        assert_eq!(cadence.finish(epoch + Duration::from_secs(9)), 0);
    }

    #[test]
    fn successful_slow_query_leaves_real_in_phase_holes() {
        let epoch = Instant::now();
        let mut cadence = PhaseCadence::new(epoch, Duration::from_secs(2));
        cadence.observed(epoch + Duration::from_secs(2));
        // The 2s query returns at 9s, after the workload finished at 7s.
        // Its success cannot recover the missed 4s and 6s observations.
        assert_eq!(cadence.finish(epoch + Duration::from_secs(7)), 2);
    }

    #[test]
    fn delayed_periodic_query_preserves_missed_slots_and_resumes_original_clock() {
        let epoch = Instant::now();
        let mut cadence = PhaseCadence::new(epoch, Duration::from_secs(2));
        cadence.observed(epoch + Duration::from_secs(2));
        cadence.observed(epoch + Duration::from_secs(9));
        assert_eq!(cadence.due(), epoch + Duration::from_secs(10));
        assert_eq!(cadence.finish(epoch + Duration::from_secs(10)), 2);
    }

    #[test]
    fn missing_last_in_phase_deadline_is_not_repaired_by_final_probe() {
        let epoch = Instant::now();
        let cadence = PhaseCadence::new(epoch, Duration::from_secs(2));
        assert_eq!(cadence.finish(epoch + Duration::from_millis(2001)), 1);
        let cadence = PhaseCadence::new(epoch, Duration::from_secs(2));
        assert_eq!(cadence.finish(epoch + Duration::from_secs(2)), 0);
    }

    #[test]
    fn source_timing_preserves_slow_query_and_peak_bracket_without_raw_series() {
        let epoch = Instant::now();
        let mut timing = ImageResourceQueryTiming::default();
        timing.record(epoch, epoch, epoch + Duration::from_millis(5), true, None);
        timing.record(
            epoch,
            epoch + Duration::from_secs(2),
            epoch + Duration::from_secs(9),
            true,
            Some(Duration::from_secs(2)),
        );
        timing.record(
            epoch,
            epoch + Duration::from_secs(10),
            epoch + Duration::from_millis(10010),
            false,
            None,
        );
        assert_eq!(timing.queries, 3);
        assert_eq!(timing.maximum_query_ms, 7000.0);
        assert_eq!(timing.periodic_queries, 1);
        assert_eq!(timing.maximum_periodic_query_ms, 7000.0);
        assert_eq!(timing.overlong_periodic_queries, 1);
        assert_eq!(timing.total_query_ms, 7015.0);
        assert_eq!(timing.maximum_query_start_gap_ms, 8000.0);
        assert_eq!(timing.peak_query_started_offset_ms, Some(2000.0));
        assert_eq!(timing.peak_query_finished_offset_ms, Some(9000.0));
        assert_eq!(timing.last_query_finished_offset_ms, Some(10010.0));
    }
}
