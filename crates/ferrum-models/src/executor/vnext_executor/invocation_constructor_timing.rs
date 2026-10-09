//! Exclusive constructor intervals, published once per node by a typed sink.
//! This module aggregates observations; it never reads a clock or authority.

use super::*;

#[derive(Default)]
struct PhaseMetrics {
    total_ns: AtomicU64,
    visits: AtomicU64,
}

impl PhaseMetrics {
    fn record(&self, phase: &InvocationConstructorPhaseTiming) {
        if phase.visits != 0 {
            let ns = phase.elapsed.as_nanos().min(u64::MAX as u128) as u64;
            self.total_ns.fetch_add(ns, Ordering::Relaxed);
            self.visits.fetch_add(phase.visits, Ordering::Relaxed);
        }
    }

    fn reset(&self) {
        self.total_ns.store(0, Ordering::Relaxed);
        self.visits.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> (u64, serde_json::Value) {
        let total_ns = self.total_ns.load(Ordering::Relaxed);
        let visits = self.visits.load(Ordering::Relaxed);
        (
            total_ns,
            serde_json::json!({
                "total_ns": total_ns,
                "visits": visits,
                "average_us_per_visit": if visits == 0 { 0.0 } else {
                    total_ns as f64 / visits as f64 / 1_000.0
                },
            }),
        )
    }
}

#[derive(Default)]
pub(super) struct InvocationConstructorTimingMetrics {
    node_attempts: AtomicU64,
    node_successes: AtomicU64,
    node_errors: AtomicU64,
    node_setup_assemble: PhaseMetrics,
    participant_common_validate: PhaseMetrics,
    resource_view_build: PhaseMetrics,
    runtime_full_coverage: PhaseMetrics,
    component_workspace_validate: PhaseMetrics,
}

impl InvocationConstructorTimingMetrics {
    pub(super) fn record(&self, breakdown: &InvocationConstructorTimingBreakdown) {
        self.node_attempts.fetch_add(1, Ordering::Relaxed);
        match breakdown.outcome {
            InvocationConstructorOutcome::Success => &self.node_successes,
            InvocationConstructorOutcome::Error => &self.node_errors,
        }
        .fetch_add(1, Ordering::Relaxed);
        self.node_setup_assemble
            .record(&breakdown.node_setup_assemble);
        self.participant_common_validate
            .record(&breakdown.participant_common_validate);
        self.resource_view_build
            .record(&breakdown.resource_view_build);
        self.runtime_full_coverage
            .record(&breakdown.runtime_full_coverage);
        self.component_workspace_validate
            .record(&breakdown.component_workspace_validate);
    }

    pub(super) fn reset(&self) {
        for counter in [&self.node_attempts, &self.node_successes, &self.node_errors] {
            counter.store(0, Ordering::Relaxed);
        }
        for phase in [
            &self.node_setup_assemble,
            &self.participant_common_validate,
            &self.resource_view_build,
            &self.runtime_full_coverage,
            &self.component_workspace_validate,
        ] {
            phase.reset();
        }
    }

    pub(super) fn snapshot(&self, outer_total_ns: u64) -> serde_json::Value {
        let (setup_ns, setup) = self.node_setup_assemble.snapshot();
        let (common_ns, common) = self.participant_common_validate.snapshot();
        let (resources_ns, resources) = self.resource_view_build.snapshot();
        let (coverage_ns, coverage) = self.runtime_full_coverage.snapshot();
        let (components_ns, components) = self.component_workspace_validate.snapshot();
        let exclusive_total = [
            setup_ns,
            common_ns,
            resources_ns,
            coverage_ns,
            components_ns,
        ]
        .into_iter()
        .map(u128::from)
        .sum::<u128>();
        let residual = i128::from(outer_total_ns) - exclusive_total as i128;
        let reported_residual = residual.clamp(i128::from(i64::MIN), i128::from(i64::MAX));
        serde_json::json!({
            "collection": "typed_host_timing_sink_only",
            "clock": "host_monotonic",
            "scope": "exclusive_constructor_phases_per_node_attempt_including_returned_errors_excluding_unwinding",
            "node_attempts": self.node_attempts.load(Ordering::Relaxed),
            "node_successes": self.node_successes.load(Ordering::Relaxed),
            "node_errors": self.node_errors.load(Ordering::Relaxed),
            "node_setup_assemble": setup,
            "participant_common_validate": common,
            "resource_view_build": resources,
            "runtime_full_coverage": coverage,
            "component_workspace_validate": components,
            "exclusive_total_ns": exclusive_total.min(u128::from(u64::MAX)) as u64,
            "exclusive_total_saturated": exclusive_total > u128::from(u64::MAX),
            "outer_minus_exclusive_total_ns": reported_residual as i64,
            "residual_saturated": reported_residual != residual,
            "limitations": [
                "each typed callback aggregates one node locally; phase visits are not node or wave counts",
                "phase intervals are mutually exclusive and nested inside the existing invocation_construct total; do not add child totals to the parent",
                "divide phase total_ns by the corresponding submitted wave count for time per wave; visit averages are different denominators",
                "returned errors include work through the failing phase; unwinding emits no constructor breakdown",
                "clock reads and local accounting perturb enabled execution and can be charged to phases; callback/caller overhead and uncovered work remain in the outer residual, which is not total observer overhead",
                "health fields use independent atomic loads and are not a transactional snapshot; in-flight publication can produce a negative residual",
                "host wall intervals include allocator and lock wait or preemption, and do not measure pure CPU, GPU activity, provider encode or engine sampling/commit/stream-send"
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase(ns: u64, visits: u64) -> InvocationConstructorPhaseTiming {
        InvocationConstructorPhaseTiming {
            elapsed: Duration::from_nanos(ns),
            visits,
        }
    }

    fn success() -> InvocationConstructorTimingBreakdown {
        InvocationConstructorTimingBreakdown {
            outcome: InvocationConstructorOutcome::Success,
            node_setup_assemble: phase(11, 2),
            participant_common_validate: phase(13, 3),
            resource_view_build: phase(17, 3),
            runtime_full_coverage: phase(19, 3),
            component_workspace_validate: phase(23, 3),
        }
    }

    fn snapshot(metrics: &VNextWaveTimingMetrics) -> serde_json::Value {
        metrics.snapshot()["host_encode_submit_breakdown"]["provider_encode_submit_breakdown"]
            ["provider_node_encode_breakdown"]["invocation_construct_breakdown"]
            .clone()
    }

    #[test]
    fn constructor_timing_aggregates_once_per_node_and_preserves_phase_scope() {
        let metrics = VNextExecutorMetrics::default();
        let sink = VNextWaveTimingSink {
            aggregate: &metrics.wave_timing,
            phase: &metrics.decode_wave_timing,
            structure_attempt: None,
        };
        sink.record_invocation_construct_breakdown(success());
        sink.record_invocation_construct_breakdown(InvocationConstructorTimingBreakdown {
            outcome: InvocationConstructorOutcome::Error,
            node_setup_assemble: phase(5, 1),
            participant_common_validate: phase(7, 1),
            resource_view_build: phase(0, 0),
            runtime_full_coverage: phase(0, 0),
            component_workspace_validate: phase(0, 0),
        });
        sink.record(
            SubmissionWaveDispatchStage::NodeInvocationConstruct,
            Duration::from_nanos(103),
        );
        for timing in [&metrics.wave_timing, &metrics.decode_wave_timing] {
            let value = snapshot(timing);
            assert_eq!(value["node_attempts"], 2);
            assert_eq!(value["node_successes"], 1);
            assert_eq!(value["node_errors"], 1);
            assert_eq!(value["node_setup_assemble"]["total_ns"], 16);
            assert_eq!(value["participant_common_validate"]["visits"], 4);
            assert_eq!(value["resource_view_build"]["visits"], 3);
            assert_eq!(value["component_workspace_validate"]["total_ns"], 23);
            assert_eq!(value["exclusive_total_ns"], 95);
            assert_eq!(value["outer_minus_exclusive_total_ns"], 8);
            assert_eq!(timing.snapshot()["submitted_wave_total"]["samples"], 0);
        }
        assert_eq!(snapshot(&metrics.prefill_wave_timing)["node_attempts"], 0);
        assert_eq!(snapshot(&metrics.mixed_wave_timing)["node_attempts"], 0);
    }

    #[test]
    fn constructor_timing_startup_reset_clears_both_aggregate_and_phase_totals() {
        let metrics = VNextExecutorMetrics::default();
        VNextWaveTimingSink {
            aggregate: &metrics.wave_timing,
            phase: &metrics.prefill_wave_timing,
            structure_attempt: None,
        }
        .record_invocation_construct_breakdown(success());
        metrics.reset_after_startup();
        for timing in [
            &metrics.wave_timing,
            &metrics.prefill_wave_timing,
            &metrics.decode_wave_timing,
            &metrics.mixed_wave_timing,
        ] {
            let value = snapshot(timing);
            assert_eq!(value["node_attempts"], 0);
            assert_eq!(value["node_successes"], 0);
            assert_eq!(value["node_errors"], 0);
            assert_eq!(value["exclusive_total_ns"], 0);
            assert_eq!(value["participant_common_validate"]["visits"], 0);
        }
    }

    #[test]
    fn constructor_timing_reports_signed_residual_without_inventing_parent_time() {
        let metrics = VNextWaveTimingMetrics::default();
        metrics.record_invocation_construct_breakdown(success());
        let value = snapshot(&metrics);
        assert_eq!(value["exclusive_total_ns"], 83);
        assert_eq!(value["outer_minus_exclusive_total_ns"], -83);
        assert_eq!(value["residual_saturated"], false);
        assert_eq!(metrics.node_invocation_construct.snapshot()["samples"], 0);
        assert_eq!(metrics.submitted_wave_total.snapshot()["samples"], 0);
    }
}
