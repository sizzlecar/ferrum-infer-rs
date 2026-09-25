//! Original whole-wall and terminal replay. No deserialized live receipt exists.
use super::*;
use ferrum_interfaces::execution_cost::*;

pub(super) fn outside(p: &Prepared, s: OutsideSettlement) -> Stages {
    Stages {
        schema_version: 1,
        call_id: s.call_id,
        presubmit_prediction: None,
        fingerprint: s.fingerprint,
        actual_shape: Some(p.exact.clone()),
        statistical_evidence: Some(p.selected.clone()),
        structured_evidence: None,
        prepare_started_at_ns: s.prepare_started_at_ns,
        executor_returned_at_ns: s.executor_returned_at_ns,
        rows: s.rows,
        finalized_at_ns: s.finalized_at_ns,
        full_wall_ns: s.full_wall_ns,
        completeness: s.completeness,
    }
}
pub(super) fn validate(
    h: &Header,
    p: &Prepared,
    s: &Stages,
    independent: Option<&IndependentAttentionWaveEvidenceWireV2>,
    binding: [u8; 32],
) -> Result<(u64, u64), CostProfileError> {
    let fail = || invalid("original V2 host settlement differs from Prepared or terminal protocol");
    if s.schema_version != 1
        || s.call_id == 0
        || s.fingerprint.as_ref() != Some(&h.fingerprint)
        || s.completeness != "complete_single_wave"
        || s.rows.len() != p.rows.len()
        || s.actual_shape.as_ref() != Some(&p.exact)
        || s.statistical_evidence.as_ref() != Some(&p.selected)
        || independent != p.selected_independent_attention_v2.as_ref()
        || stage_binding(s, independent)? != binding
    {
        return Err(fail());
    }
    let prepare = s.prepare_started_at_ns.ok_or_else(fail)?;
    let returned = s.executor_returned_at_ns.ok_or_else(fail)?;
    let finalized = s.finalized_at_ns.ok_or_else(fail)?;
    if prepare < h.opening.monotonic_ns
        || prepare < h.opened_at_ns
        || returned < prepare
        || finalized < returned
    {
        return Err(fail());
    }
    let mut end = returned;
    let mut ordinals = HashSet::new();
    let mut inputs = HashSet::new();
    for ((row, before), declared) in s.rows.iter().zip(&p.rows).zip(&p.recipe.physical_host_rows) {
        if row.request_id != before.request_id
            || row.owner_incarnation != before.owner_incarnation
            || row.work_generation != before.work_generation
            || row.actual_work.actual() != prepared::actual_work(before.frontier.work)
            || row.completeness != "complete_single_wave"
            || !inputs.insert(row.input_index)
        {
            return Err(fail());
        }
        let ordinal = row.host_processing_ordinal.ok_or_else(fail)?;
        let start = row.host_started_at_ns.ok_or_else(fail)?;
        let committed = row.token_committed_at_ns.ok_or_else(fail)?;
        let finish = row.settled_at_ns.ok_or_else(fail)?;
        let emits = before.frontier.work.emits_token().map_err(numeric_error)?;
        if ordinal as usize >= s.rows.len()
            || !ordinals.insert(ordinal)
            || start < returned
            || committed < start
            || finish < committed
            || finish > finalized
            || row.output_published_at_ns.is_some() != emits
            || row
                .output_published_at_ns
                .is_some_and(|v| v < committed || v > finish)
            || row
                .completion_started_at_ns
                .is_some_and(|v| v < committed || v > finish)
        {
            return Err(fail());
        }
        if let Some(prior) = s
            .rows
            .iter()
            .find(|p| p.host_processing_ordinal.and_then(|v| v.checked_add(1)) == Some(ordinal))
        {
            if prior.settled_at_ns.is_none_or(|v| v > start) {
                return Err(fail());
            }
        }
        match (declared.terminal_expectation, &row.terminal) {
            (Expectation::NoTokenProduced, None) | (Expectation::TokenMayTerminate, None) => {}
            (Expectation::LengthBoundary, Some(t)) => {
                if t.generated_tokens
                    != before
                        .frontier
                        .generated_before
                        .checked_add(1)
                        .ok_or_else(fail)?
                    || t.generated_tokens != before.frontier.maximum_output
                    || row.completion_started_at_ns.is_none()
                {
                    return Err(fail());
                }
                terminal(t)?;
            }
            _ => return Err(fail()),
        }
        end = end.max(finish);
    }
    let wall = end
        .checked_sub(prepare)
        .filter(|v| *v > 0)
        .ok_or_else(fail)?;
    if s.full_wall_ns != Some(wall) {
        return Err(fail());
    }
    if let Some(settled) = &s.structured_evidence {
        let settled = settled.as_ref().map_err(|_| fail())?;
        if settled.protocol != "ferrum.structured-host-settled-capture.v1"
            || settled.call_id != s.call_id
            || settled.recipe != p.recipe
            || settled.stage_binding != binding
            || settled.full_wall_ns != wall
            || settled.executor_envelope_ns != returned - prepare
            || settled.host_settled_after_executor_ns != end - returned
        {
            return Err(fail());
        }
    }
    Ok((wall, finalized))
}
pub(super) fn terminal(t: &Terminal) -> Result<(), CostProfileError> {
    if t.finish_reason != ferrum_types::FinishReason::Length
        || t.generated_tokens == 0
        || t.output_failed
        || t.physical_failed
        || t.scheduler_failed
        || !t.terminal_handoff_succeeded
        || t.pending_restore_removed
        || t.other_physical_resources
        || !t.request_slot_closed
        || !t.owner_matched
        || t.admission_cancellation_work != serde_json::json!("no_additional_work")
        || t.cache_completion_work != serde_json::json!("no_additional_work")
    {
        return Err(invalid("ineligible original terminal receipt"));
    }
    Ok(())
}
pub(super) fn convert(
    h: &Header,
    input: StructuredInputV2,
    n: &Numeric,
    phase: StructuredProfilePhaseV10,
    member: u64,
    offer: u64,
    call: u64,
    wall: u64,
    observed: u64,
) -> Result<StructuredNumericObservationV2, CostProfileError> {
    if n.call_id != call
        || n.wall_ns != wall
        || n.observed_at_ns != observed
        || n.domain != *input.domain_signature()
        || n.basis != input.regression_axes()
        || n.support != input.joint_support_coordinates()
        || n.basis.len() > h.settings.max_axes
        || n.support.len() > h.settings.max_axes
    {
        return Err(invalid(
            "V2 numeric payload differs from original complete recipe projection",
        ));
    }
    Ok(StructuredNumericObservationV2 {
        source: h.capture_identity,
        protocol: h.protocol,
        ordinal: n.fifo,
        membership: StructuredMemberBindingV2 {
            rule_signature: h.rule_signature,
            offered_ordinal: offer,
            member_ordinal: member,
            phase: phase.native(),
        },
        call_id: call,
        fingerprint: h.fingerprint.clone().into(),
        input,
        boundary: CostBoundary::PreparationToHostSettledV1,
        outcome: WaveObservationOutcome::Completed,
        observed_at_ns: observed,
        wall_ns: wall,
    })
}
pub(super) fn stage_binding(
    s: &Stages,
    independent: Option<&IndependentAttentionWaveEvidenceWireV2>,
) -> Result<[u8; 32], CostProfileError> {
    #[derive(Serialize)]
    struct ShapeBinding<'a> {
        actual_shape: &'a Option<Shape>,
        statistics: &'a Option<StatisticalWaveEvidenceWireV1>,
        independent_attention: Option<&'a IndependentAttentionWaveEvidenceWireV2>,
    }
    let shape = ShapeBinding {
        actual_shape: &s.actual_shape,
        statistics: &s.statistical_evidence,
        independent_attention: independent,
    };
    let fingerprint = s.fingerprint.as_ref().map(|v| {
        (
            v.model_weights,
            v.numerical_policy,
            v.device_runtime,
            v.execution_config,
        )
    });
    let bytes = serde_json::to_vec(&(
        s.call_id,
        fingerprint,
        &s.rows,
        shape,
        s.prepare_started_at_ns,
        s.executor_returned_at_ns,
        s.finalized_at_ns,
        s.full_wall_ns,
        &s.completeness,
    ))?;
    let mut hash = Sha256::new();
    hash.update(b"ferrum.structured-host-settlement.v1");
    hash.update(bytes);
    Ok(hash.finalize().into())
}
