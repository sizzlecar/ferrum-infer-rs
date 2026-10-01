//! First rejected route gate per original window. Scalar diagnostics only;
//! never a source record, route proof, ticket retirement or query input.
use super::*;
use ferrum_interfaces::execution_cost::RouteCaptureDiagnostic;
use serde::Serializer;

#[derive(Debug, Clone, Copy, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct CallRouteFailureAudit {
    pub generation: u64,
    pub phase: usize,
    pub ticket: u64,
    pub call_id: u64,
    pub gate: &'static str,
    pub capture: RouteCaptureDiagnostic,
    pub physical_waves: usize,
    pub retained_waves: usize,
    pub lost_observations: u64,
    pub coverage: &'static str,
    #[serde(serialize_with = "optional_debug")]
    pub coverage_unknown: Option<CostObservationUnknownReason>,
    #[serde(serialize_with = "optional_debug")]
    pub dispatch_outcome: Option<ObservedCallOutcome>,
    #[serde(serialize_with = "optional_debug")]
    pub dispatch_unknown: Option<ActualWaveEvidenceUnknown>,
    pub prepare_started_at_ns: Option<u64>,
    pub returned_at_ns: Option<u64>,
    pub physical_call_id: Option<u64>,
    pub physical_wave_ordinal: Option<u32>,
    #[serde(serialize_with = "optional_debug")]
    pub physical_outcome: Option<ActualWaveOutcome>,
    #[serde(serialize_with = "optional_debug")]
    pub call_boundary: Option<WaveObservationBoundary>,
    #[serde(serialize_with = "optional_debug")]
    pub physical_boundary: Option<WaveObservationBoundary>,
    pub physical_prepare_started_at_ns: Option<u64>,
    pub physical_submission_started_at_ns: Option<u64>,
    pub physical_terminal_at_ns: Option<u64>,
    #[serde(serialize_with = "optional_debug")]
    pub actual_graph: Option<ActualWaveGraphState>,
    #[serde(serialize_with = "optional_debug")]
    pub shape_unknown: Option<ActualWaveEvidenceUnknown>,
    pub stage_rejection: Option<CostCallRejection>,
}

fn optional_debug<T: std::fmt::Debug, S: Serializer>(
    value: &Option<T>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match value {
        Some(value) => serializer.serialize_some(&format!("{value:?}")),
        None => serializer.serialize_none(),
    }
}

impl EngineCostCall {
    pub(in crate::continuous_engine::inner::cost_observation) fn capture_live_route_failure_diagnostic(
        &self,
    ) {
        if self.recorder.route_diagnostic().is_none() {
            return;
        }
        if let Err(gate) = self.route_for_settlement_checked() {
            self.capture_route_failure_gate(gate);
        }
    }

    pub(super) fn capture_route_failure_gate(&self, gate: &'static str) {
        let Some(ticket) = self.live_ticket.as_ref() else {
            return;
        };
        if ticket.route_population().is_all_attempts()
            || !tracing::enabled!(
                target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                tracing::Level::DEBUG
            )
        {
            return;
        }
        let Some(capture) = self.recorder.route_diagnostic() else {
            return;
        };
        let wave = self.recorder.observations().first();
        let (coverage, coverage_unknown, lost_observations) = match self.recorder.coverage() {
            CostObservationCoverage::Complete => ("complete", None, 0),
            CostObservationCoverage::Unknown {
                reason,
                lost_observations,
            } => ("unknown", Some(reason), lost_observations),
        };
        ticket.record_route_failure(CallRouteFailureAudit {
            generation: 0, // Bound by the original ticket, not caller-supplied.
            phase: 0,
            ticket: 0,
            call_id: self.call_id.get(),
            gate,
            capture,
            physical_waves: self.dispatch.waves,
            retained_waves: self.recorder.observations().len(),
            lost_observations,
            coverage,
            coverage_unknown,
            dispatch_outcome: self.dispatch.outcome,
            dispatch_unknown: self.dispatch.unknown,
            prepare_started_at_ns: self.prepare_started_at_ns,
            returned_at_ns: self.dispatch.returned_at_ns,
            physical_call_id: wave.map(|w| w.call_id.get()),
            physical_wave_ordinal: wave.map(|w| w.physical_wave_ordinal),
            physical_outcome: wave.and_then(|w| w.outcome),
            call_boundary: Some(self.boundary),
            physical_boundary: wave.map(|w| w.boundary),
            physical_prepare_started_at_ns: wave.map(|w| w.prepare_started_at_ns),
            physical_submission_started_at_ns: wave.and_then(|w| w.submission_started_at_ns),
            physical_terminal_at_ns: wave.and_then(|w| w.terminal_at_ns),
            actual_graph: wave.and_then(|w| w.shape.as_ref().map(|s| s.graph)),
            shape_unknown: wave.and_then(|w| w.shape_unknown),
            stage_rejection: self.stage_rejection,
        });
    }
}
