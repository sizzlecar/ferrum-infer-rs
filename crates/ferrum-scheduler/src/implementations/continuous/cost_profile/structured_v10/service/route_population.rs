//! Complete raw attempts excluded by a route declaration frozen before offers.
//! These DTOs can replay evidence; none can manufacture a live selector receipt.
use super::*;
use ferrum_interfaces::execution_cost::{HostContentDomainV1, HostCostFeaturesV1};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceRouteCountsV1 {
    pub attempted: usize,
    pub eligible_route: usize,
    pub outside_declared_route: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub no_submission: usize,
}
fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StructuredServicePreparedRouteV6 {
    selection: RouteSelection,
    selected_at_ns: u64,
    submitted: SubmittedIdentity,
}
pub(super) fn validate_eligible(wave: &StructuredServiceWaveV6) -> Result<(), CostProfileError> {
    validate_eligible_parts(
        wave.issued_at_ns,
        wave.prepared_route.as_ref(),
        &wave.host_stages,
    )
}
pub(super) fn validate_eligible_parts(
    issued_at_ns: u64,
    prepared_route: Option<&StructuredServicePreparedRouteV6>,
    stages: &Stages,
) -> Result<(), CostProfileError> {
    let fail = || invalid("source6 eligible route lacks original pre-submit declaration");
    let proof = prepared_route.ok_or_else(fail)?;
    let r = &proof.selection;
    let shape = stages.actual_shape.as_ref().ok_or_else(fail)?;
    if r.non_reusable_wave.is_some()
        || proof.selected_at_ns < issued_at_ns
        || proof.submitted.submission_started_at_ns < proof.selected_at_ns
        || stages
            .executor_returned_at_ns
            .is_none_or(|returned| proof.submitted.submission_started_at_ns > returned)
        || !submitted_matches(r, &proof.submitted)
    {
        return Err(fail());
    }
    match (r.class.as_str(), shape.exact.graph_state) {
        ("warm", ProfileGraphState::Warm)
            if r.reason == "resident_program"
                && r.catalog_epoch == Some(r.lane_epoch)
                && r.program_id.is_some()
                && r.batch_step.is_some_and(|v| v != 0)
                && r.batch_invocation.is_some_and(|v| v != 0)
                && r.graph_state.as_ref().is_some_and(|s| {
                    matches!(s.configuration.as_str(), "startup_ready" | "on_demand")
                }) =>
        {
            Ok(())
        }
        ("graph_disabled", ProfileGraphState::Disabled)
            if r.program_id.is_none()
                && (r.reason == "declared_graph_unsupported"
                    || (r.reason == "unconfigured_stream"
                        && r.graph_state.as_ref().is_some_and(|s| {
                            s.configuration == "unconfigured"
                                && s.resident_executables == 0
                                && s.resident_programs == 0
                                && s.rejected_executables == 0
                        }))) =>
        {
            Ok(())
        }
        _ => Err(fail()),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredServiceOutsideRouteV6 {
    pub ticket: u64,
    pub phase: StructuredPhaseV2,
    pub issued_at_ns: u64,
    pub fifo: u64,
    evidence: OutsideEvidence,
}
impl StructuredServiceOutsideRouteV6 {
    /// Validate original execution/host facts independently of any source ticket.
    /// This grants neither source membership nor a live selector capability.
    pub fn validate_original_diagnostic(
        fingerprint: &ProfileFingerprint,
        issued_at_ns: u64,
        evidence: serde_json::Value,
    ) -> Result<(usize, u64), CostProfileError> {
        let evidence: OutsideEvidence = serde_json::from_value(evidence)?;
        validate_settlement_parts(
            fingerprint,
            issued_at_ns,
            issued_at_ns,
            issued_at_ns,
            &evidence,
            &mut physical::Frontiers::default(),
        )
    }
    pub fn from_diagnostic(
        ticket: u64,
        phase: StructuredPhaseV2,
        issued_at_ns: u64,
        fifo: u64,
        evidence: serde_json::Value,
    ) -> Result<Self, CostProfileError> {
        Ok(Self {
            ticket,
            phase,
            issued_at_ns,
            fifo,
            evidence: serde_json::from_value(evidence)?,
        })
    }
    /// Validate the complete raw settlement without granting source membership
    /// or a prediction capability. Live callers must separately hold the
    /// original private selector and issued ticket; replay also checks policy.
    pub fn validate_settlement(
        &self,
        fingerprint: &ProfileFingerprint,
        opened_at_ns: u64,
    ) -> Result<(usize, u64), CostProfileError> {
        validate_settlement(
            fingerprint,
            opened_at_ns,
            opened_at_ns,
            self,
            &mut physical::Frontiers::default(),
        )
    }

    pub(super) fn call_id(&self) -> u64 {
        self.evidence.host_stages.call_id
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OutsideEvidence {
    protocol: String,
    selection: RouteSelection,
    selected_at_ns: u64,
    submitted: SubmittedIdentity,
    #[serde(deserialize_with = "bounded_rows")]
    prepared_rows: Vec<OutsidePreparedRow>,
    physical: Physical,
    host_stages: Stages,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutsidePreparedRow {
    request_id: String,
    owner_incarnation: u64,
    work_generation: u64,
    input_index: u32,
    actual_work: RowWork,
    host_features: HostCostFeaturesV1,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Physical {
    call_id: u64,
    physical_wave_ordinal: u32,
    physical_waves: usize,
    retained_waves: usize,
    lost_observations: u64,
    boundary: String,
    prepare_started_at_ns: u64,
    submission_started_at_ns: u64,
    terminal_at_ns: u64,
    outcome: String,
    call_outcome: String,
    shape_unknown: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteSelection {
    class: String,
    reason: String,
    program_id: Option<Program>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    non_reusable_wave: Option<NonReusableWave>,
    lane_id: u64,
    lane_epoch: u64,
    catalog_epoch: Option<u64>,
    graph_state: Option<GraphState>,
    batch_step: Option<u64>,
    batch_invocation: Option<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NonReusableWave {
    plan_hash: String,
    runtime_implementation_fingerprint: String,
    immediate_sequences: u32,
    immediate_tokens: u64,
    immediate_pages: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Program {
    plan_hash: String,
    runtime_implementation_fingerprint: String,
    lane_id: u64,
    bucket_id: String,
    program_binding_layout_fingerprint: String,
    lane_stable_layout_fingerprint: String,
    lane_slot_id: u64,
    immediate_sequences: u32,
    immediate_tokens: u64,
    immediate_pages: u64,
    topology_fingerprint: [u8; 32],
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GraphState {
    configuration: String,
    resident_executables: u64,
    resident_programs: u64,
    rejected_executables: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SubmittedIdentity {
    batch_step: u64,
    batch_invocation: u64,
    plan_hash: String,
    runtime_implementation_fingerprint: String,
    lane_id: u64,
    submission_started_at_ns: u64,
    graph: Option<SubmittedGraph>,
}
fn submitted_matches(route: &RouteSelection, submitted: &SubmittedIdentity) -> bool {
    if submitted.batch_step == 0
        || submitted.batch_invocation == 0
        || submitted.lane_id == 0
        || submitted.lane_id != route.lane_id
        || !sha256(&submitted.plan_hash)
        || !sha256(&submitted.runtime_implementation_fingerprint)
    {
        return false;
    }
    if let Some(program) = &route.program_id {
        route.non_reusable_wave.is_none()
            && route.batch_step == Some(submitted.batch_step)
            && route.batch_invocation == Some(submitted.batch_invocation)
            && program.plan_hash == submitted.plan_hash
            && program.runtime_implementation_fingerprint
                == submitted.runtime_implementation_fingerprint
            && program.lane_id == submitted.lane_id
            && route
                .graph_state
                .as_ref()
                .zip(submitted.graph.as_ref())
                .is_some_and(|(before, actual)| before == &actual.before)
    } else if let Some(wave) = &route.non_reusable_wave {
        route.class == "outside_program_layout_absent"
            && route.reason == "program_layout_absent"
            && route.batch_step == Some(submitted.batch_step)
            && route.batch_invocation == Some(submitted.batch_invocation)
            && wave.plan_hash == submitted.plan_hash
            && wave.runtime_implementation_fingerprint
                == submitted.runtime_implementation_fingerprint
            && route
                .graph_state
                .as_ref()
                .zip(submitted.graph.as_ref())
                .is_some_and(|(before, actual)| {
                    before == &actual.before
                        && before == &actual.after_preparation
                        && before.configuration == "on_demand"
                        && !actual.capture_requested
                        && actual.candidate_segments == 0
                        && actual.captured_segments == 0
                        && actual.capture_rejected_segments == 0
                        && actual.uploaded_segments == 0
                        && actual.replayed_segments == 0
                })
    } else {
        route.class == "graph_disabled"
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SubmittedGraph {
    before: GraphState,
    after_preparation: GraphState,
    capture_requested: bool,
    candidate_segments: u64,
    captured_segments: u64,
    capture_rejected_segments: u64,
    uploaded_segments: u64,
    replayed_segments: u64,
}
fn sha256(value: &str) -> bool {
    let raw = value.strip_prefix("sha256/").unwrap_or(value);
    raw.len() == 64
        && raw
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(super) fn validate(
    header: &StructuredServiceHeaderV6,
    opened: u64,
    wave: &StructuredServiceOutsideRouteV6,
    frontiers: &mut physical::Frontiers,
) -> Result<(usize, u64), CostProfileError> {
    if header.declaration.route_population.is_all_attempts() {
        return Err(invalid(
            "source6 AllAttempts cannot retire OutsideDeclaredRoute",
        ));
    }
    validate_settlement(
        &header.fingerprint,
        header.opening.monotonic_ns,
        opened,
        wave,
        frontiers,
    )
}

fn validate_settlement(
    fingerprint: &ProfileFingerprint,
    source_opened: u64,
    opened: u64,
    wave: &StructuredServiceOutsideRouteV6,
    frontiers: &mut physical::Frontiers,
) -> Result<(usize, u64), CostProfileError> {
    validate_settlement_parts(
        fingerprint,
        source_opened,
        opened,
        wave.issued_at_ns,
        &wave.evidence,
        frontiers,
    )
}
pub(super) fn validate_settlement_parts(
    fingerprint: &ProfileFingerprint,
    source_opened: u64,
    opened: u64,
    issued_at_ns: u64,
    e: &OutsideEvidence,
    frontiers: &mut physical::Frontiers,
) -> Result<(usize, u64), CostProfileError> {
    let fail =
        || invalid("source6 OutsideDeclaredRoute lacks original selection/physical/host proof");
    let r = &e.selection;
    let s = &e.host_stages;
    let p = &e.physical;
    let graph = r.graph_state.as_ref().ok_or_else(fail)?;
    let submitted_graph = e.submitted.graph.as_ref().ok_or_else(fail)?;
    let (immediate_sequences, immediate_tokens) =
        match (r.program_id.as_ref(), r.non_reusable_wave.as_ref()) {
            (Some(program), None)
                if matches!(
                    r.class.as_str(),
                    "outside_program_absent" | "outside_program_non_resident"
                ) && sha256(&program.plan_hash)
                    && sha256(&program.runtime_implementation_fingerprint)
                    && sha256(&program.program_binding_layout_fingerprint)
                    && sha256(&program.lane_stable_layout_fingerprint)
                    && program.lane_id != 0
                    && !program.bucket_id.is_empty()
                    && program.bucket_id.len() <= 160 =>
            {
                (program.immediate_sequences, program.immediate_tokens)
            }
            (None, Some(wave))
                if r.class == "outside_program_layout_absent"
                    && r.reason == "program_layout_absent"
                    && sha256(&wave.plan_hash)
                    && sha256(&wave.runtime_implementation_fingerprint) =>
            {
                (wave.immediate_sequences, wave.immediate_tokens)
            }
            _ => return Err(fail()),
        };
    let outside = matches!(
        (r.class.as_str(), r.reason.as_str()),
        ("outside_program_absent", "catalog_empty" | "program_absent")
            | ("outside_program_non_resident", "program_non_resident")
            | ("outside_program_layout_absent", "program_layout_absent")
    );
    if e.protocol != "ferrum.outside-declared-route-settled.v1"
        || !outside
        || !submitted_matches(r, &e.submitted)
        || e.submitted.submission_started_at_ns != p.submission_started_at_ns
        || r.catalog_epoch != Some(r.lane_epoch)
        || r.batch_step.is_none_or(|v| v == 0)
        || r.batch_invocation.is_none_or(|v| v == 0)
        || !matches!(graph.configuration.as_str(), "startup_ready" | "on_demand")
        || &submitted_graph.before != graph
        || !matches!(
            submitted_graph.after_preparation.configuration.as_str(),
            "startup_ready" | "on_demand"
        )
        || submitted_graph
            .captured_segments
            .checked_add(submitted_graph.capture_rejected_segments)
            .is_none_or(|n| n > submitted_graph.candidate_segments)
        || submitted_graph.uploaded_segments > submitted_graph.captured_segments
        || e.prepared_rows.is_empty()
        || e.prepared_rows.len() > 128
        || immediate_sequences as usize != e.prepared_rows.len()
        || s.rows.len() != e.prepared_rows.len()
        || s.schema_version != 1
        || s.call_id == 0
        || s.fingerprint.as_ref() != Some(fingerprint)
        || s.completeness != "complete_single_wave"
        || p.call_id != s.call_id
        || p.physical_wave_ordinal != 0
        || p.physical_waves != 1
        || p.retained_waves != 1
        || p.lost_observations != 0
        || p.outcome != "completed"
        || p.call_outcome != "completed"
        || p.boundary != "isolated_preparation_to_commit"
        || p.shape_unknown
            .as_deref()
            .is_some_and(|v| v != "graph_path")
        || (p.shape_unknown.is_some() && s.actual_shape.is_some())
        || (p.shape_unknown.is_none() && s.actual_shape.is_none())
        || p.prepare_started_at_ns != issued_at_ns
        || s.prepare_started_at_ns != Some(issued_at_ns)
        || issued_at_ns < opened
        || issued_at_ns < source_opened
        || e.selected_at_ns < issued_at_ns
        || p.submission_started_at_ns < e.selected_at_ns
        || p.terminal_at_ns < p.submission_started_at_ns
    {
        return Err(fail());
    }
    let returned = s.executor_returned_at_ns.ok_or_else(fail)?;
    let finalized = s.finalized_at_ns.ok_or_else(fail)?;
    if returned < p.terminal_at_ns || finalized < returned {
        return Err(fail());
    }
    let mut end = returned;
    let mut tokens = 0_u64;
    let mut request_ids = HashSet::new();
    let mut inputs = HashSet::new();
    let mut ordinals = HashSet::new();
    for (before, row) in e.prepared_rows.iter().zip(&s.rows) {
        let host = &before.host_features;
        let generated = host.state.generated_tokens_before;
        let maximum = host.state.maximum_output_tokens;
        let (start, finish_kv, emits, work_tokens) = match before.actual_work {
            RowWork::Decode { kv_tokens } if kv_tokens > 0 => (
                kv_tokens,
                kv_tokens.checked_add(1).ok_or_else(fail)?,
                true,
                1,
            ),
            RowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } if count > 0 => {
                let finish = offset
                    .checked_add(count)
                    .filter(|v| *v <= total_prompt_tokens)
                    .ok_or_else(fail)?;
                (
                    offset,
                    finish,
                    finish == total_prompt_tokens,
                    u64::from(count),
                )
            }
            _ => return Err(fail()),
        };
        tokens = tokens.checked_add(work_tokens).ok_or_else(fail)?;
        if !host.supports_installed_plain_text_content()
            || maximum == 0
            || generated >= maximum
            || before.request_id.is_empty()
            || before.request_id.len() > 512
            || before.owner_incarnation == 0
            || before.work_generation == 0
            || !request_ids.insert(&before.request_id)
            || !inputs.insert(before.input_index)
            || row.request_id != before.request_id
            || row.owner_incarnation != before.owner_incarnation
            || row.work_generation != before.work_generation
            || row.input_index != before.input_index
            || row.actual_work.actual() != before.actual_work.actual()
            || row.completeness != "complete_single_wave"
        {
            return Err(fail());
        }
        let ordinal = row.host_processing_ordinal.ok_or_else(fail)?;
        let begin = row.host_started_at_ns.ok_or_else(fail)?;
        let committed = row.token_committed_at_ns.ok_or_else(fail)?;
        let settled = row.settled_at_ns.ok_or_else(fail)?;
        if ordinal as usize >= s.rows.len()
            || !ordinals.insert(ordinal)
            || begin < returned
            || committed < begin
            || settled < committed
            || settled > finalized
            || row.output_published_at_ns.is_some() != emits
            || row
                .output_published_at_ns
                .is_some_and(|at| at < committed || at > settled)
            || row
                .completion_started_at_ns
                .is_some_and(|at| at < committed || at > settled)
            || row.completion_started_at_ns.is_some() != row.terminal.is_some()
            || s.rows
                .iter()
                .find(|prior| {
                    prior.host_processing_ordinal.and_then(|v| v.checked_add(1)) == Some(ordinal)
                })
                .is_some_and(|prior| prior.settled_at_ns.is_none_or(|at| at > begin))
        {
            return Err(fail());
        }
        let after = generated.checked_add(u64::from(emits)).ok_or_else(fail)?;
        match &row.terminal {
            Some(t) if emits => {
                observation::terminal_service(t)?;
                let terminal_supported =
                    match (host.policy.empirical_content_domain, t.finish_reason) {
                        (_, ferrum_types::FinishReason::Length) => true,
                        (
                            Some(HostContentDomainV1::PlainTextInstalledV2(cap)),
                            ferrum_types::FinishReason::EOS,
                        ) => cap.model_eos,
                        (
                            Some(HostContentDomainV1::PlainTextInstalledV2(cap)),
                            ferrum_types::FinishReason::Stop,
                        ) => cap.user_stop,
                        _ => false,
                    };
                if !terminal_supported
                    || t.generated_tokens != after
                    || after > maximum
                    || (t.finish_reason == ferrum_types::FinishReason::Length && after != maximum)
                {
                    return Err(fail());
                }
            }
            None if after < maximum => {}
            _ => return Err(fail()),
        }
        frontiers.advance(
            s.call_id,
            &before.request_id,
            before.owner_incarnation,
            before.work_generation,
            generated,
            maximum,
            start,
            finish_kv,
            emits,
            row.terminal.is_some(),
        )?;
        end = end.max(settled);
    }
    if tokens != immediate_tokens
        || s.full_wall_ns != end.checked_sub(issued_at_ns).filter(|v| *v > 0)
    {
        return Err(fail());
    }
    Ok((s.rows.len(), finalized))
}

impl OutsideEvidence {
    pub(super) fn original_cohort_rows(
        &self,
    ) -> Result<Vec<super::super::lifecycle::OriginalCohortRow<'_>>, CostProfileError> {
        self.prepared_rows
            .iter()
            .map(|r| {
                let (work, context_before) = match r.actual_work {
                    RowWork::Decode { kv_tokens } => {
                        (PreparedWorkV2::Decode { kv_tokens }, u64::from(kv_tokens))
                    }
                    RowWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    } => (
                        PreparedWorkV2::Prefill {
                            offset,
                            count,
                            total_prompt_tokens,
                        },
                        u64::from(offset),
                    ),
                    _ => return Err(invalid("source8 outside row lacks inference work")),
                };
                let frontier = PreparedRowFactsV2 {
                    physical_position: r.input_index,
                    generated_before: r.host_features.state.generated_tokens_before,
                    maximum_output: r.host_features.state.maximum_output_tokens,
                    context_before,
                    work,
                };
                frontier.validate().map_err(numeric_error)?;
                Ok(super::super::lifecycle::OriginalCohortRow {
                    request_id: &r.request_id,
                    owner_incarnation: r.owner_incarnation,
                    work_generation: r.work_generation,
                    frontier,
                    policy: r.host_features.policy,
                    pending_decoded_utf8: r.host_features.state.pending_decoded_utf8,
                })
            })
            .collect()
    }
    pub(super) fn original_host_stages(&self) -> &Stages {
        &self.host_stages
    }
    pub(super) fn call_id(&self) -> u64 {
        self.host_stages.call_id
    }
}
