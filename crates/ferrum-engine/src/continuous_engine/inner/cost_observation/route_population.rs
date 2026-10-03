//! Private same-call route evidence. Selection remains distinct from a usable
//! numerical shape; an outside attempt never manufactures a warm cost sample.
use super::*;
use ferrum_interfaces::execution_cost::{
    CostObservationCoverage, CostObservationUnknownReason, HostCostFeaturesV1, PreparedCallRouteV1,
    PreparedCostRouteClassV1,
};
use ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::StructuredPhaseV2, cost_profile::StructuredServiceOutsideRouteV6,
};
use ferrum_types::FerrumError;
use serde::Serialize;
pub(super) mod diagnostic;

#[derive(Debug, Clone, Serialize)]
struct PreparedRow {
    request_id: RequestId,
    owner_incarnation: u64,
    work_generation: u64,
    input_index: u32,
    actual_work: HostStageWork,
    host_features: HostCostFeaturesV1,
}
#[derive(Debug, Clone, Serialize)]
struct Physical {
    call_id: u64,
    physical_wave_ordinal: u32,
    physical_waves: usize,
    retained_waves: usize,
    lost_observations: u64,
    boundary: &'static str,
    prepare_started_at_ns: u64,
    submission_started_at_ns: u64,
    terminal_at_ns: u64,
    outcome: &'static str,
    call_outcome: &'static str,
    shape_unknown: Option<&'static str>,
}
#[derive(Debug, Clone)]
pub(super) struct LiveRouteEvidence {
    prepared: PreparedCallRouteV1,
    rows: Vec<PreparedRow>,
    physical: Physical,
}
impl LiveRouteEvidence {
    pub fn is_outside(&self) -> bool {
        self.prepared.route().class().is_outside()
    }
    pub fn retained_rows(&self) -> usize {
        self.rows
            .capacity()
            .saturating_add(self.prepared.rows().len())
    }
    pub fn retained_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>()
            .checked_add(self.prepared.retained_payload_bytes()?)?
            .checked_add(
                self.rows
                    .capacity()
                    .checked_mul(std::mem::size_of::<PreparedRow>())?,
            )?
            .checked_add(2 * std::mem::size_of::<usize>())
    }
    pub fn frontier_states(&self) -> impl Iterator<Item = &HostCostFeaturesV1> {
        self.rows.iter().map(|row| &row.host_features)
    }
    pub fn eligible_diagnostic(&self) -> impl Serialize + '_ {
        #[derive(Serialize)]
        struct View<'a> {
            selection: &'a ferrum_interfaces::execution_cost::PreparedCostRouteV1,
            selected_at_ns: u64,
            submitted: Option<&'a ferrum_interfaces::execution_cost::SubmittedRouteEvidenceV1>,
        }
        View {
            selection: self.prepared.route(),
            selected_at_ns: self.prepared.selected_at_ns(),
            submitted: self.prepared.submitted(),
        }
    }
    pub fn outside_diagnostic<'a>(
        &'a self,
        stages: &'a HostStageEvidenceV1,
    ) -> impl Serialize + 'a {
        #[derive(Serialize)]
        struct View<'a, S: Serialize, H: Serialize> {
            protocol: &'static str,
            #[serde(flatten)]
            prepared: S,
            prepared_rows: &'a [PreparedRow],
            physical: &'a Physical,
            host_stages: H,
        }
        View {
            protocol: "ferrum.outside-declared-route-settled.v1",
            prepared: self.eligible_diagnostic(),
            prepared_rows: &self.rows,
            physical: &self.physical,
            host_stages: stages.source_view(),
        }
    }
    pub(super) fn feedback_observed_at(
        &self,
        stages: &HostStageEvidenceV1,
        fingerprint: &ExecutionFingerprint,
    ) -> Option<u64> {
        if !self.is_outside() {
            return None;
        }
        let (_, observed) = StructuredServiceOutsideRouteV6::validate_original_diagnostic(
            &ferrum_scheduler::implementations::continuous::cost_profile::ProfileFingerprint::from(
                fingerprint,
            ),
            stages.prepare_started_at_ns?,
            serde_json::to_value(self.outside_diagnostic(stages)).ok()?,
        )
        .ok()?;
        Some(observed)
    }
    pub fn outside_record(
        &self,
        stages: &HostStageEvidenceV1,
        ticket: u64,
        phase: StructuredPhaseV2,
        fifo: u64,
    ) -> Result<StructuredServiceOutsideRouteV6, FerrumError> {
        if !self.is_outside() {
            return Err(FerrumError::config(
                "route is not outside declared population",
            ));
        }
        StructuredServiceOutsideRouteV6::from_diagnostic(
            ticket,
            phase,
            stages
                .prepare_started_at_ns
                .ok_or_else(|| FerrumError::config("missing original route clock"))?,
            fifo,
            serde_json::to_value(self.outside_diagnostic(stages))
                .map_err(|e| FerrumError::config(e.to_string()))?,
        )
        .map_err(|e| FerrumError::config(e.to_string()))
    }
}
impl EngineCostCall {
    /// This gate accepts only the private selector that the actual submission
    /// consumed. Unknown graph evidence is allowed solely for an already
    /// outside selection, and only the specific actual GraphPath rejection.
    pub(super) fn route_for_settlement(&self) -> Option<&PreparedCallRouteV1> {
        self.route_for_settlement_checked().ok()
    }

    fn route_for_settlement_checked(&self) -> Result<&PreparedCallRouteV1, &'static str> {
        match self.live_ticket.as_ref() {
            Some(ticket) if ticket.route_population().is_all_attempts() => {
                return Err("route_population_not_requested");
            }
            Some(_) => {}
            None if self
                .calibration_capture
                .as_ref()
                .is_some_and(|capture| capture.requests_original_route()) => {}
            None if self.source_generation != 0
                && self
                    .feedback_population
                    .is_some_and(|p| !p.is_all_attempts()) => {}
            None => return Err("live_ticket_missing"),
        }
        let prepared = self
            .recorder
            .prepared_route()
            .ok_or("private_prepared_route_missing")?;
        let wave = self
            .recorder
            .observations()
            .first()
            .ok_or("physical_wave_missing")?;
        let submitted = prepared.submitted().ok_or("private_submission_missing")?;
        let outside = prepared.route().class().is_outside();
        // Same ordered predicates as the original Option gate, now preserving
        // which predicate rejected. Diagnostics cannot change their outcome.
        macro_rules! reject_if {
            ($condition:expr, $gate:literal) => {
                if $condition {
                    return Err($gate);
                }
            };
        }
        reject_if!(
            self.recorder.observations().len() != 1,
            "retained_wave_count"
        );
        reject_if!(self.dispatch.waves != 1, "physical_wave_count");
        reject_if!(
            self.dispatch.outcome != Some(ObservedCallOutcome::Completed),
            "call_not_completed"
        );
        reject_if!(
            self.dispatch
                .unknown
                .is_some_and(|reason| !outside || reason != ActualWaveEvidenceUnknown::GraphPath),
            "dispatch_unknown"
        );
        reject_if!(wave.call_id != self.call_id, "physical_call_mismatch");
        reject_if!(wave.physical_wave_ordinal != 0, "physical_wave_ordinal");
        reject_if!(
            wave.outcome != Some(ActualWaveOutcome::Completed),
            "wave_not_completed"
        );
        reject_if!(
            wave.boundary != WaveObservationBoundary::IsolatedPreparationToCommit,
            "physical_boundary"
        );
        reject_if!(
            self.boundary != WaveObservationBoundary::IsolatedPreparationToCommit,
            "call_boundary"
        );
        reject_if!(
            self.prepare_started_at_ns != Some(wave.prepare_started_at_ns),
            "prepare_clock_mismatch"
        );
        reject_if!(
            prepared.selected_at_ns() < wave.prepare_started_at_ns,
            "selection_before_prepare"
        );
        reject_if!(
            wave.submission_started_at_ns != Some(submitted.submission_started_at_ns()),
            "submission_clock_mismatch"
        );
        reject_if!(
            prepared.selected_at_ns() > submitted.submission_started_at_ns(),
            "selection_after_submission"
        );
        reject_if!(
            wave.terminal_at_ns
                .is_none_or(|at| at < submitted.submission_started_at_ns()),
            "terminal_clock_missing_or_before_submit"
        );
        reject_if!(
            self.dispatch
                .returned_at_ns
                .zip(wave.terminal_at_ns)
                .is_none_or(|(returned, end)| returned < end),
            "return_clock_missing_or_before_terminal"
        );
        reject_if!(
            !outside
                && !matches!(
                    prepared.route().class(),
                    PreparedCostRouteClassV1::Warm | PreparedCostRouteClassV1::GraphDisabled
                ),
            "selected_route_unknown"
        );
        match self.recorder.coverage() {
            CostObservationCoverage::Complete if wave.shape.is_some() => {}
            // make_host_stages precedes the legacy make_sample host_committed
            // marker. Its original host-progress checks below provide the full
            // settlement; no other recorder uncertainty is accepted here.
            CostObservationCoverage::Unknown {
                reason: CostObservationUnknownReason::IncompleteObservation,
                lost_observations: 0,
            } if wave.shape.is_some() && wave.host_committed_at_ns.is_none() => {}
            CostObservationCoverage::Unknown {
                reason:
                    CostObservationUnknownReason::EvidenceUnavailable(
                        ActualWaveEvidenceUnknown::GraphPath,
                    ),
                lost_observations: 0,
            } if outside
                && wave.shape.is_none()
                && wave.shape_unknown == Some(ActualWaveEvidenceUnknown::GraphPath) => {}
            _ => return Err("recorder_coverage_or_shape"),
        }
        if !outside {
            let graph = wave.shape.as_ref().ok_or("eligible_shape_missing")?.graph;
            if PreparedCostRouteClassV1::from_eligible_graph(graph)
                != Some(prepared.route().class())
            {
                return Err("actual_graph_mismatch");
            }
        }
        Ok(prepared)
    }
    pub(super) fn make_route_evidence(&self) -> Option<Arc<LiveRouteEvidence>> {
        self.make_route_evidence_recording_failure(&mut None)
    }

    pub(super) fn make_route_evidence_recording_failure(
        &self,
        failure: &mut Option<&'static str>,
    ) -> Option<Arc<LiveRouteEvidence>> {
        match self.make_route_evidence_checked() {
            Ok(evidence) => Some(evidence),
            Err(gate) => {
                if self.recorder.route_diagnostic().is_some() {
                    *failure = Some(gate);
                }
                self.capture_route_failure_gate(gate);
                None
            }
        }
    }

    fn make_route_evidence_checked(&self) -> Result<Arc<LiveRouteEvidence>, &'static str> {
        let prepared = self.route_for_settlement_checked()?;
        let wave = self
            .recorder
            .observations()
            .first()
            .ok_or("physical_wave_missing")?;
        let mut rows = Vec::new();
        if prepared.route().class().is_outside() {
            rows.try_reserve_exact(prepared.rows().len())
                .map_err(|_| "outside_row_allocation")?;
            for actual in prepared.rows() {
                let index = self
                    .participants
                    .iter()
                    .position(|p| {
                        p.request_id == actual.request_id
                            && p.owner_incarnation == actual.owner_incarnation
                            && p.work_generation == actual.work_generation
                            && p.input_index == actual.input_index
                    })
                    .ok_or("outside_participant_identity")?;
                let host = self.participants[index]
                    .host_features
                    .ok_or("outside_host_features_missing")?;
                let work = self.host_stages[index]
                    .committed_work()
                    .ok_or("outside_committed_work_missing")?;
                let before = match work {
                    HostCommittedWork::Decode {
                        generated_tokens_before,
                        ..
                    }
                    | HostCommittedWork::Prefill {
                        generated_tokens_before,
                        ..
                    } => generated_tokens_before,
                };
                if !host.supports_installed_plain_text_content()
                    || before != host.state.generated_tokens_before
                    || sample::validate_work(actual.work, work).is_err()
                {
                    return Err("outside_host_work_or_policy");
                }
                rows.push(PreparedRow {
                    request_id: actual.request_id.clone(),
                    owner_incarnation: actual.owner_incarnation,
                    work_generation: actual.work_generation,
                    input_index: actual.input_index,
                    actual_work: match actual.work {
                        ActualRowWork::Decode { kv_tokens } => HostStageWork::Decode { kv_tokens },
                        ActualRowWork::Prefill {
                            offset,
                            count,
                            total_prompt_tokens,
                        } => HostStageWork::Prefill {
                            offset,
                            count,
                            total_prompt_tokens,
                        },
                        _ => return Err("outside_work_kind"),
                    },
                    host_features: host,
                });
            }
        }
        Ok(Arc::new(LiveRouteEvidence {
            prepared: prepared.clone(),
            rows,
            physical: Physical {
                call_id: wave.call_id.get(),
                physical_wave_ordinal: wave.physical_wave_ordinal,
                physical_waves: self.dispatch.waves,
                retained_waves: self.recorder.observations().len(),
                lost_observations: 0,
                boundary: "isolated_preparation_to_commit",
                prepare_started_at_ns: wave.prepare_started_at_ns,
                submission_started_at_ns: wave
                    .submission_started_at_ns
                    .ok_or("submission_clock_missing")?,
                terminal_at_ns: wave.terminal_at_ns.ok_or("terminal_clock_missing")?,
                outcome: "completed",
                call_outcome: "completed",
                shape_unknown: wave.shape_unknown.map(|_| "graph_path"),
            },
        }))
    }
}
