//! Fixed-size passive diagnostics. These scalar copies cannot reconstruct a
//! private selector or authorize a submission/sample. Disabled by default.
use super::*;
use crate::vnext::{DeviceCostGraphStreamState, DeviceSubmissionGraphEvidence};
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PreparedRouteDiagnostic {
    pub class: PreparedCostRouteClassV1,
    pub reason: PreparedCostRouteReasonV1,
    pub has_program_identity: bool,
    pub lane_id: u64,
    pub lane_epoch: u64,
    pub catalog_epoch: Option<u64>,
    pub batch_step: Option<u64>,
    pub batch_invocation: Option<u64>,
    pub graph: Option<DeviceCostGraphStreamState>,
}
impl From<&PreparedCostRouteV1> for PreparedRouteDiagnostic {
    fn from(route: &PreparedCostRouteV1) -> Self {
        Self {
            class: route.class,
            reason: route.reason,
            has_program_identity: route.program_id.is_some(),
            lane_id: route.lane_id,
            lane_epoch: route.lane_epoch,
            catalog_epoch: route.catalog_epoch,
            batch_step: route.batch_step,
            batch_invocation: route.batch_invocation,
            graph: route.graph_state,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct SubmittedRouteDiagnostic {
    pub batch_step: u64,
    pub batch_invocation: u64,
    pub lane_id: u64,
    pub submission_started_at_ns: Option<u64>,
    pub graph: Option<DeviceSubmissionGraphEvidence>,
    pub program_plan_matches: Option<bool>,
    pub program_runtime_matches: Option<bool>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RouteCaptureDiagnostic {
    pub prepared: Option<PreparedRouteDiagnostic>,
    pub selected_at_ns: Option<u64>,
    pub submitted: Option<SubmittedRouteDiagnostic>,
    pub first_rejection: Option<&'static str>,
    pub preparation_attempts: usize,
    pub other_evidence_unknown: bool,
}

impl BoundedWaveRecorder {
    /// Diagnostic-only opt-in. It changes no recorder acceptance or retention
    /// limit, and the returned values never act as a route receipt.
    pub fn enable_route_diagnostics(&mut self) {
        self.route_diagnostic.get_or_insert_with(Default::default);
    }

    pub fn route_diagnostic(&self) -> Option<RouteCaptureDiagnostic> {
        self.route_diagnostic.map(|mut value| {
            value.preparation_attempts = self.prepared_route_attempts;
            value.other_evidence_unknown = self.route_unknown;
            value
        })
    }

    pub(crate) fn diagnose_route_selection(&mut self, route: &PreparedCostRouteV1) {
        if let Some(diagnostic) = &mut self.route_diagnostic {
            diagnostic.prepared.get_or_insert_with(|| route.into());
        }
    }

    pub(crate) fn diagnose_route_rejection(&mut self, gate: &'static str) {
        if let Some(diagnostic) = &mut self.route_diagnostic {
            diagnostic.first_rejection.get_or_insert(gate);
        }
    }

    pub(crate) fn diagnose_route_clock(&mut self, at: u64) {
        if let Some(diagnostic) = &mut self.route_diagnostic {
            diagnostic.selected_at_ns.get_or_insert(at);
        }
    }

    pub(super) fn diagnose_route_submission(
        &mut self,
        attribution: Option<&crate::vnext::BoundDeviceSubmissionAttribution>,
    ) {
        let Some(diagnostic) = &mut self.route_diagnostic else {
            return;
        };
        let Some(attribution) = attribution else {
            diagnostic
                .first_rejection
                .get_or_insert("submission_attribution_missing");
            return;
        };
        let identity = attribution.batch_identity();
        let program = self
            .prepared_route
            .as_ref()
            .and_then(|p| p.route.program_id());
        diagnostic
            .submitted
            .get_or_insert(SubmittedRouteDiagnostic {
                batch_step: identity.batch_step_id().get(),
                batch_invocation: identity.batch_invocation_id().get(),
                lane_id: identity.lane_id().get(),
                submission_started_at_ns: self
                    .observations
                    .first()
                    .and_then(|w| w.submission_started_at_ns),
                graph: attribution.device().graph_evidence(),
                program_plan_matches: program.map(|p| p.plan_hash() == identity.plan_hash()),
                program_runtime_matches: program.map(|p| {
                    p.runtime_implementation_fingerprint()
                        == identity.runtime_implementation_fingerprint()
                }),
            });
    }
}
