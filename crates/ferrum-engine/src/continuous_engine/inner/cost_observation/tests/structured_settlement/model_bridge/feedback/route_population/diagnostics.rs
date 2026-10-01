use super::*;
use ferrum_interfaces::execution_cost::{PreparedCostRouteClassV1, PreparedCostRouteReasonV1};

// Thread-scoped filter exercises the product DEBUG gate without a dependency
// or a process-global subscriber. Events are discarded; assertions inspect
// the same bounded window audit that production exports.
pub(super) struct RuntimeDebug(pub(super) bool);
impl tracing::Subscriber for RuntimeDebug {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        self.0
            && metadata.target()
                == "ferrum_engine::continuous_engine::inner::cost_observation::runtime"
    }
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

#[tokio::test]
async fn route_diagnostic_real_submission_retains_missing_program_reason_without_eligibility() {
    for debug in [false, true] {
        tracing::subscriber::with_default(RuntimeDebug(debug), || {
            let f = Automatic::new(9);
            let stages = fixture::record_route_with_hook(
                &f.runtime,
                &f.clock,
                wave("fixture.route-diagnostic"),
                false,
                false,
                true,
                |_| {},
            )
            .unwrap();
            assert!(stages.route_evidence.is_none());
            let audit = f.live().audit();
            assert_eq!(audit.population.first_route_failure.is_some(), debug);
            if let Some(diagnostic) = audit.population.first_route_failure {
                assert_eq!(diagnostic.call_id, stages.call_id);
                assert_eq!(diagnostic.ticket, 1);
                assert_eq!(diagnostic.generation, audit.population.generation);
                assert_eq!(diagnostic.phase, audit.population.phase);
                assert_eq!(diagnostic.gate, "private_prepared_route_missing");
                let selected = diagnostic.capture.prepared.unwrap();
                assert_eq!(selected.class, PreparedCostRouteClassV1::Unknown);
                assert_eq!(
                    selected.reason,
                    PreparedCostRouteReasonV1::ProgramIdentityUnavailable
                );
                assert!(!selected.has_program_identity);
                assert_eq!(
                    diagnostic.capture.first_rejection,
                    Some("program_identity_missing_for_selected_class")
                );
                // Actual core submission facts survive the original mandatory
                // removal of the unusable private route.
                assert!(diagnostic.capture.submitted.unwrap().graph.is_some());
                assert_eq!(diagnostic.physical_waves, 1);
            }
            f.runtime.consume_samples();
            assert_eq!(f.live().audit().qualified_publications, 0);
            assert!(f.live().audit().failed_generations > 0);
        });
    }
}

#[tokio::test]
async fn route_diagnostic_settlement_distinguishes_clock_boundary_and_actual_graph() {
    for expected in [
        "prepare_clock_mismatch",
        "call_boundary",
        "actual_graph_mismatch",
        "recorder_coverage_or_shape",
    ] {
        tracing::subscriber::with_default(RuntimeDebug(true), || {
            let f = Automatic::new(9);
            let mut w = wave("fixture.route-diagnostic");
            if expected == "actual_graph_mismatch" {
                // Deliberately inconsistent shape; the real private selector
                // and original native submit both remain GraphDisabled.
                w.actual.graph = ActualWaveGraphState::Warm;
            }
            let stages = fixture::record_route_with_hook(
                &f.runtime,
                &f.clock,
                w,
                false,
                false,
                false,
                |call| match expected {
                    "prepare_clock_mismatch" => call.prepare_started_at_ns = None,
                    "call_boundary" => call.boundary = WaveObservationBoundary::Unknown,
                    "recorder_coverage_or_shape" => call.recorder.note_lost(1),
                    _ => {}
                },
            )
            .unwrap();
            assert!(stages.route_evidence.is_none());
            let diagnostic = f.live().audit().population.first_route_failure.unwrap();
            assert_eq!(diagnostic.gate, expected);
            assert_eq!(
                diagnostic.lost_observations,
                u64::from(expected == "recorder_coverage_or_shape")
            );
            assert_eq!(
                diagnostic.capture.prepared.unwrap().class,
                PreparedCostRouteClassV1::GraphDisabled
            );
            assert!(diagnostic.capture.submitted.is_some());
            assert!(diagnostic.capture.first_rejection.is_none());
            f.runtime.consume_samples();
            assert_eq!(f.live().audit().qualified_publications, 0);
        });
    }
}

#[tokio::test]
async fn route_diagnostic_first_window_failure_is_not_overwritten_and_success_has_none() {
    tracing::subscriber::with_default(RuntimeDebug(true), || {
        let f = Automatic::new(9);
        f.record(false);
        assert!(f.live().audit().population.first_route_failure.is_none());
        fixture::record_route_with_hook(
            &f.runtime,
            &f.clock,
            wave("fixture.route-diagnostic-first"),
            false,
            false,
            false,
            |call| call.boundary = WaveObservationBoundary::Unknown,
        );
        let original =
            serde_json::to_value(f.live().audit().population.first_route_failure).unwrap();
        fixture::record_route_with_hook(
            &f.runtime,
            &f.clock,
            wave("fixture.route-diagnostic-later"),
            false,
            false,
            true,
            |_| {},
        );
        assert_eq!(
            serde_json::to_value(f.live().audit().population.first_route_failure).unwrap(),
            original,
        );
        let audit = f.live().audit();
        assert_eq!(audit.population.issued, 3);
        assert_eq!(
            audit.population.retired, 0,
            "diagnostics grant no retirement"
        );
        f.runtime.consume_samples();
        assert_eq!(f.live().audit().qualified_publications, 0);
    });
}
