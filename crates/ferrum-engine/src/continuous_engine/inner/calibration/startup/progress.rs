//! Bounded, numeric-only diagnostics for the private startup driver. This is
//! never reference evidence and never changes a probe, retry or qualification.
//! Controller intervals reuse the existing finalized once-only audit. They are
//! inclusive wall intervals, not additive CPU time or a device measurement.
use super::*;
use ferrum_types::{ControllerTimingMetrics, WallTimingAggregate};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage {
    AddRequest,
    Frontier,
    Maintenance,
    OutputReadiness,
    Admission,
    Wave,
    ReferenceCapture,
    OutputCompletion,
    Complete,
    Drain,
    Drained,
}

#[derive(Default, Debug)]
struct DriverWallMs {
    add_request: f64,
    frontier: f64,
    maintenance: f64,
    output_readiness: f64,
    admission: f64,
    wave: f64,
    reference_capture: f64,
    output_completion: f64,
    drain: f64,
}

impl DriverWallMs {
    fn add(&mut self, stage: Stage, elapsed: f64) {
        let value = match stage {
            Stage::AddRequest => &mut self.add_request,
            Stage::Frontier => &mut self.frontier,
            Stage::Maintenance => &mut self.maintenance,
            Stage::OutputReadiness => &mut self.output_readiness,
            Stage::Admission => &mut self.admission,
            Stage::Wave => &mut self.wave,
            Stage::ReferenceCapture => &mut self.reference_capture,
            Stage::OutputCompletion => &mut self.output_completion,
            Stage::Drain => &mut self.drain,
            Stage::Complete | Stage::Drained => return,
        };
        *value += elapsed;
    }
}

#[derive(Default, Debug, PartialEq, Eq)]
pub(super) struct Counts {
    pub wave_attempts: usize,
    pub host_reconciled: usize,
    pub submitted_without_reconciliation: usize,
    pub indeterminate: usize,
    pub not_submitted: usize,
    pub zero_submit_retries: usize,
    pub wave_blocked: usize,
    // Original recorder counts or validated singleton/NotSubmitted receipts.
    // Other missing physical evidence is never inferred as zero.
    pub known_physical_waves: usize,
    pub reports_without_physical_count: usize,
    pub maintenance_turns: usize,
    pub maintenance_reconciled: usize,
    pub maintenance_blocked: usize,
    pub admission_turns: usize,
    pub admission_progressed: usize,
    pub admission_blocked: usize,
}

#[derive(Clone, Debug)]
pub(super) struct TimingDelta {
    pub finalized_transactions: u64,
    pub backend_submitted: u64,
    pub invalid_clock_transactions: u64,
    pub unfinished_planning_transactions: u64,
    pub planning: WallTimingAggregate,
    pub transaction: WallTimingAggregate,
    pub release: WallTimingAggregate,
    pub capture: WallTimingAggregate,
    pub search_replay: WallTimingAggregate,
    pub publication: WallTimingAggregate,
    pub ready_queue: WallTimingAggregate,
    pub iteration_lock_wait: WallTimingAggregate,
    pub input_preparation: WallTimingAggregate,
    pub host_guard: WallTimingAggregate,
    pub executor_await: WallTimingAggregate,
    pub reconciliation: WallTimingAggregate,
}

impl TimingDelta {
    fn between(
        before: Option<&ControllerTimingMetrics>,
        after: Option<&ControllerTimingMetrics>,
    ) -> Option<Self> {
        // None before means no audit had ever finalized, i.e. the true baseline
        // is empty. None after means no finalized timing is available.
        let empty = ControllerTimingMetrics::default();
        let before = before.unwrap_or(&empty);
        let after = after?;
        fn counter(before: u64, after: u64) -> Option<u64> {
            // Saturated totals no longer identify a bounded interval.
            (after != u64::MAX).then_some(())?;
            after.checked_sub(before)
        }
        fn timing(
            before: WallTimingAggregate,
            after: WallTimingAggregate,
        ) -> Option<WallTimingAggregate> {
            Some(WallTimingAggregate {
                wall_ns_total: counter(before.wall_ns_total, after.wall_ns_total)?,
                calls: counter(before.calls, after.calls)?,
            })
        }
        Some(Self {
            finalized_transactions: counter(
                before.finalized_transactions,
                after.finalized_transactions,
            )?,
            backend_submitted: counter(before.backend_submitted, after.backend_submitted)?,
            invalid_clock_transactions: counter(
                before.invalid_clock_transactions,
                after.invalid_clock_transactions,
            )?,
            unfinished_planning_transactions: counter(
                before.unfinished_planning_transactions,
                after.unfinished_planning_transactions,
            )?,
            planning: timing(before.planning, after.planning)?,
            transaction: timing(before.transaction, after.transaction)?,
            release: timing(before.release, after.release)?,
            capture: timing(before.capture, after.capture)?,
            search_replay: timing(before.search_replay, after.search_replay)?,
            publication: timing(before.publication, after.publication)?,
            ready_queue: timing(before.ready_queue, after.ready_queue)?,
            iteration_lock_wait: timing(before.iteration_lock_wait, after.iteration_lock_wait)?,
            input_preparation: timing(before.input_preparation, after.input_preparation)?,
            host_guard: timing(before.host_guard, after.host_guard)?,
            executor_await: timing(before.executor_await, after.executor_await)?,
            reconciliation: timing(before.reconciliation, after.reconciliation)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct WavePosition {
    pub attempt: usize,
    pub prefill_progress: Option<(usize, usize)>,
    pub generated_tokens: usize,
}

struct WaveInProgress {
    position: WavePosition,
    started: Instant,
    timing_before: Option<ControllerTimingMetrics>,
}

#[derive(Clone, Debug)]
pub(super) struct FinishedWave {
    pub position: WavePosition,
    pub elapsed_ms: f64,
    pub submission: Option<CalibrationSubmissionState>,
    pub physical_waves: Option<usize>,
    pub controller: Option<TimingDelta>,
}

pub(super) struct ProbeProgress {
    pub request_ordinal: usize,
    pub anchor_tokens: u32,
    pub capture: &'static str,
    pub trial: Option<CalibrationReferenceTrial>,
    pub discovery_attempt: usize,
    pub counts: Counts,
    pub stage: Stage,
    pub slowest_wave: Option<FinishedWave>,
    started: Instant,
    finished_elapsed_ms: Option<f64>,
    stage_started: Instant,
    driver_wall_ms: DriverWallMs,
    timing_before: Option<ControllerTimingMetrics>,
    wave: Option<WaveInProgress>,
}

#[derive(Default)]
pub(super) struct StartupProgress {
    pub current: Option<ProbeProgress>,
    pub completed_requests: usize,
    pub phase: &'static str,
}

impl StartupProgress {
    pub fn begin(
        &mut self,
        request_ordinal: usize,
        anchor_tokens: u32,
        capture: &probes::Capture<'_>,
        discovery_attempt: usize,
        timing_before: Option<ControllerTimingMetrics>,
    ) {
        let (capture, trial) = match capture {
            probes::Capture::Warmup => ("warmup", None),
            probes::Capture::PrefillDiscovery => ("prefill_discovery", None),
            probes::Capture::DecodeDiscovery => ("decode_discovery", None),
            probes::Capture::Trial { key, .. } => ("frozen_trial", Some(*key)),
        };
        let started = Instant::now();
        self.phase = "probe";
        self.current = Some(ProbeProgress {
            request_ordinal,
            anchor_tokens,
            capture,
            trial,
            discovery_attempt,
            counts: Counts::default(),
            stage: Stage::AddRequest,
            slowest_wave: None,
            started,
            finished_elapsed_ms: None,
            stage_started: started,
            driver_wall_ms: DriverWallMs::default(),
            timing_before,
            wave: None,
        });
        tracing::info!(
            request_ordinal,
            anchor_tokens,
            capture,
            ?trial,
            discovery_attempt,
            "Automatic reference probe started"
        );
    }

    pub fn stage(&mut self, stage: Stage) {
        if let Some(probe) = &mut self.current {
            let now = Instant::now();
            probe.driver_wall_ms.add(
                probe.stage,
                now.duration_since(probe.stage_started).as_secs_f64() * 1_000.0,
            );
            probe.stage = stage;
            probe.stage_started = now;
        }
    }

    pub fn counts(&mut self) -> &mut Counts {
        &mut self
            .current
            .as_mut()
            .expect("probe diagnostics begin before execution")
            .counts
    }

    pub fn begin_wave(
        &mut self,
        before: &CalibrationFrontier,
        timing_before: Option<ControllerTimingMetrics>,
    ) {
        self.stage(Stage::Wave);
        let probe = self.current.as_mut().unwrap();
        probe.counts.wave_attempts += 1;
        probe.wave = Some(WaveInProgress {
            position: WavePosition {
                attempt: probe.counts.wave_attempts,
                prefill_progress: before.prefill_progress(),
                generated_tokens: before.generated_tokens(),
            },
            started: probe.stage_started,
            timing_before,
        });
    }

    pub fn end_wave(
        &mut self,
        report: Option<&CalibrationWaveReport>,
        timing_after: Option<ControllerTimingMetrics>,
        reaped: bool,
    ) {
        let Some(probe) = &mut self.current else {
            return;
        };
        // A dropped timeout waiter does not finalize this slot. Its one Reap
        // can finalize it later. A second cleanup cannot double count the wave.
        let Some(wave) = probe.wave.take() else {
            return;
        };
        if let Some(report) = report {
            match report.submission {
                CalibrationSubmissionState::NotSubmitted => probe.counts.not_submitted += 1,
                CalibrationSubmissionState::InFlightUnknown => probe.counts.indeterminate += 1,
                CalibrationSubmissionState::Submitted => {
                    probe.counts.submitted_without_reconciliation += 1
                }
                CalibrationSubmissionState::HostReconciled => probe.counts.host_reconciled += 1,
            }
            match physical_waves(report) {
                Some(count) => probe.counts.known_physical_waves += count,
                None => probe.counts.reports_without_physical_count += 1,
            }
        }
        let finished = FinishedWave {
            position: wave.position,
            elapsed_ms: wave.started.elapsed().as_secs_f64() * 1_000.0,
            submission: report.map(|report| report.submission),
            physical_waves: report.and_then(physical_waves),
            controller: TimingDelta::between(wave.timing_before.as_ref(), timing_after.as_ref()),
        };
        tracing::debug!(
            request_ordinal = probe.request_ordinal,
            anchor_tokens = probe.anchor_tokens, capture = probe.capture,
            trial = ?probe.trial, reaped, wave = ?finished,
            "Automatic reference probe wave finished (inclusive controller wall ns)"
        );
        if probe
            .slowest_wave
            .as_ref()
            .is_none_or(|previous| finished.elapsed_ms > previous.elapsed_ms)
        {
            probe.slowest_wave = Some(finished);
        }
    }

    pub fn finish(&mut self, timing_after: Option<ControllerTimingMetrics>, outcome: &'static str) {
        if outcome != "failed" {
            self.stage(Stage::Complete);
        }
        if let Some(probe) = &mut self.current {
            probe.finished_elapsed_ms = Some(probe.started.elapsed().as_secs_f64() * 1_000.0);
            if outcome != "failed" {
                self.completed_requests += 1;
            }
        }
        self.log(timing_after, outcome);
    }

    pub fn log(&self, timing_after: Option<ControllerTimingMetrics>, outcome: &'static str) {
        if let Some(probe) = &self.current {
            let controller =
                TimingDelta::between(probe.timing_before.as_ref(), timing_after.as_ref());
            let pending_wave = probe.wave.as_ref().map(|wave| {
                (
                    wave.position,
                    wave.started.elapsed().as_secs_f64() * 1_000.0,
                )
            });
            tracing::info!(
                bootstrap_phase = self.phase, outcome,
                request_ordinal = probe.request_ordinal,
                anchor_tokens = probe.anchor_tokens, capture = probe.capture,
                trial = ?probe.trial, discovery_attempt = probe.discovery_attempt,
                completed_requests = self.completed_requests,
                probe_elapsed_ms = probe.finished_elapsed_ms.unwrap_or_else(|| probe.started.elapsed().as_secs_f64() * 1_000.0),
                current_stage = ?probe.stage,
                stage_elapsed_ms = probe.stage_started.elapsed().as_secs_f64() * 1_000.0,
                completed_driver_stage_wall_ms = ?probe.driver_wall_ms,
                counts = ?probe.counts,
                pending_wave = ?pending_wave,
                slowest_wave = ?probe.slowest_wave,
                controller = ?controller,
                "Automatic reference probe progress (inclusive finalized controller wall ns)"
            );
        } else {
            tracing::info!(
                bootstrap_phase = self.phase,
                outcome,
                completed_requests = self.completed_requests,
                "Automatic reference startup progress before first probe"
            );
        }
    }
}

fn physical_waves(report: &CalibrationWaveReport) -> Option<usize> {
    if let Some(actual) = &report.actual_evidence_diagnostic {
        Some(actual.physical_waves)
    } else if matches!(report.observation, CalibrationObservation::Observed { .. }) {
        // make_sample accepts exactly one original physical recorder wave.
        // A known sample need not allocate the unknown-only diagnostic DTO.
        Some(1)
    } else if report.submission == CalibrationSubmissionState::NotSubmitted {
        Some(0)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_timing_delta_preserves_nested_wall_intervals_and_missing_evidence() {
        let before = ControllerTimingMetrics::default();
        let after = ControllerTimingMetrics {
            finalized_transactions: 1,
            transaction: WallTimingAggregate {
                wall_ns_total: 100,
                calls: 1,
            },
            executor_await: WallTimingAggregate {
                wall_ns_total: 80,
                calls: 1,
            },
            host_guard: WallTimingAggregate {
                wall_ns_total: 40,
                calls: 2,
            },
            ..Default::default()
        };
        let delta = TimingDelta::between(Some(&before), Some(&after)).unwrap();
        assert_eq!(delta.transaction.wall_ns_total, 100);
        assert_eq!(delta.executor_await.wall_ns_total, 80);
        assert_eq!(delta.host_guard.wall_ns_total, 40);
        assert_eq!(delta.host_guard.calls, 2);
        assert_eq!(delta.finalized_transactions, 1);
        assert!(TimingDelta::between(None, None).is_none());
        assert!(TimingDelta::between(Some(&after), Some(&before)).is_none());
        let saturated = ControllerTimingMetrics {
            executor_await: WallTimingAggregate {
                wall_ns_total: u64::MAX,
                calls: 1,
            },
            ..after
        };
        assert!(TimingDelta::between(None, Some(&saturated)).is_none());
    }
}
