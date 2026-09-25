//! Validate a replay diagnostic and project numbers. Never construct a live
//! QualifiedStructuredWaveEvidenceV1 or attach a deserialized recipe to one.
use super::*;
use ferrum_interfaces::execution_cost::*;

pub(super) fn convert(
    h: &Header,
    offered: &[OfferedRow],
    stages: &Stages,
    recipe: &Recipe,
    independent: Option<&IndependentAttentionWaveEvidenceWireV2>,
    n: &Numeric,
    member: u64,
    offer: u64,
) -> Result<StructuredNumericObservationV1, CostProfileError> {
    let fail = || invalid("structured source observation/receipt mismatch");
    let settled = stages
        .structured_evidence
        .as_ref()
        .and_then(|s| s.as_ref().ok())
        .ok_or_else(fail)?;
    let shape = stages.actual_shape.as_ref().ok_or_else(fail)?;
    let stat = stages.statistical_evidence.as_ref().ok_or_else(fail)?;
    if stages.schema_version != 1
        || stages.call_id == 0
        || stages.fingerprint.as_ref() != Some(&h.fingerprint)
        || stages.completeness != "complete_single_wave"
        || settled.protocol != "ferrum.structured-host-settled-capture.v1"
        || settled.call_id != stages.call_id
        || &settled.recipe != recipe
        || n.call_id != stages.call_id
        || stages.rows.len() != h.scope.rows
        || offered.len() != stages.rows.len()
        || recipe.physical_host_rows.len() != stages.rows.len()
        || n.domain != h.scope.domain
        || n.basis.len() > h.settings.max_axes
        || n.support.len() > h.settings.max_axes
    {
        return Err(fail());
    }
    if stage_binding(stages, independent)? != settled.stage_binding {
        return Err(invalid(
            "original structured stage binding cannot be reproduced",
        ));
    }
    let exact = &shape.exact;
    if exact.kind != ProfileWaveKind::Decode
        || exact.path != ProfileExecutionPath::PlanRuntime
        || exact.graph_state != ProfileGraphState::Disabled
        || exact.order != ProfileBatchOrder::Ordered
        || exact.restore_bytes != 0
        || exact.maintenance_bytes != 0
        || exact.maintenance_units != 0
        || !exact.prefill_chunks.is_empty()
        || exact.decode_kv_tokens.len() != stages.rows.len()
    {
        return Err(fail());
    }
    let features = shape.numeric_features.as_ref().ok_or_else(fail)?;
    features.validate(stages.rows.len()).map_err(|_| fail())?;
    shape
        .row_multiset_features
        .as_ref()
        .ok_or_else(fail)?
        .validate(stages.rows.len())
        .map_err(|_| fail())?;
    shape
        .host_content_features
        .as_ref()
        .ok_or_else(fail)?
        .validate()
        .map_err(|_| fail())?;
    let prepare = stages.prepare_started_at_ns.ok_or_else(fail)?;
    let returned = stages.executor_returned_at_ns.ok_or_else(fail)?;
    let finalized = stages.finalized_at_ns.ok_or_else(fail)?;
    if prepare < h.opening.monotonic_ns
        || prepare < h.opened_at_ns
        || returned < prepare
        || finalized < returned
        || finalized != n.observed_at_ns
    {
        return Err(fail());
    }
    let mut end = returned;
    let mut seen = std::collections::BTreeSet::new();
    let mut ordinals = std::collections::BTreeSet::new();
    for (i, ((row, declared), feature)) in stages
        .rows
        .iter()
        .zip(&recipe.physical_host_rows)
        .zip(&features.rows)
        .enumerate()
    {
        let source = offered
            .iter()
            .find(|o| o.request_id == row.request_id)
            .ok_or_else(fail)?;
        if !seen.insert(&row.request_id)
            || row.owner_incarnation != source.owner
            || row.work_generation != source.generation
            || source.generated != feature.generated_tokens_before
            || !source.decode
            || row.completeness != "complete_single_wave"
            || !matches!(row.actual_work,RowWork::Decode{kv_tokens} if kv_tokens==exact.decode_kv_tokens[i])
            || declared.physical_position as usize != i
            || declared.role != HostRowRoleV2::Decode
            || shape.row_multiset_features.as_ref().unwrap().rows[i].role != HostRowRoleV2::Decode
            || declared.no_generated_history != (feature.generated_tokens_before == 0)
            || feature.decoded_prefix_tokens
                != feature
                    .generated_tokens_before
                    .checked_add(1)
                    .ok_or_else(fail)?
        {
            return Err(fail());
        }
        let expected = if feature.decoded_prefix_tokens == feature.maximum_output_tokens {
            Expectation::LengthBoundary
        } else {
            Expectation::TokenMayTerminate
        };
        if declared.terminal_expectation != expected {
            return Err(fail());
        }
        let ordinal = row.host_processing_ordinal.ok_or_else(fail)?;
        if ordinal as usize >= stages.rows.len() || !ordinals.insert(ordinal) {
            return Err(fail());
        }
        let start = row.host_started_at_ns.ok_or_else(fail)?;
        let committed = row.token_committed_at_ns.ok_or_else(fail)?;
        let finish = row.settled_at_ns.ok_or_else(fail)?;
        if start < returned
            || committed < start
            || finish < committed
            || finish > finalized
            || row
                .output_published_at_ns
                .is_some_and(|v| v < committed || v > finish)
            || row
                .completion_started_at_ns
                .is_some_and(|v| v < committed || v > finish)
        {
            return Err(fail());
        }
        if let Some(prior) = stages
            .rows
            .iter()
            .find(|v| v.host_processing_ordinal.and_then(|v| v.checked_add(1)) == Some(ordinal))
        {
            if prior.settled_at_ns.is_none_or(|v| v > start) {
                return Err(fail());
            }
        }
        match (expected, &row.terminal) {
            (Expectation::TokenMayTerminate, None) => {}
            (Expectation::LengthBoundary, Some(t)) => {
                if t.finish_reason != ferrum_types::FinishReason::Length
                    || t.generated_tokens != feature.decoded_prefix_tokens
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
                    return Err(fail());
                }
            }
            _ => return Err(fail()),
        }
        end = end.max(finish);
    }
    let wall = end
        .checked_sub(prepare)
        .filter(|v| *v > 0)
        .ok_or_else(fail)?;
    if n.wall_ns != wall
        || stages.full_wall_ns != Some(wall)
        || settled.full_wall_ns != wall
        || settled.executor_envelope_ns != returned - prepare
        || settled.host_settled_after_executor_ns != end - returned
    {
        return Err(fail());
    }
    let canonical = CanonicalWaveCostShape {
        kind: ActualWaveKind::Decode,
        path: ActualWavePath::PlanRuntime,
        graph: ActualWaveGraphState::Disabled,
        row_order: ActualWaveRowOrder::Ordered,
        provider_signature: exact.provider_signature,
        output_policy_signature: exact.output_policy_signature,
        numeric_features: shape.numeric_features.clone(),
        host_content_features: shape.host_content_features,
        row_multiset_features: shape.row_multiset_features.clone(),
        rows: stages.rows.iter().map(|r| r.actual_work.actual()).collect(),
        recurrent_state_bytes: exact.recurrent_state_bytes,
    };
    let selected =
        StatisticalWaveEvidenceV1::from_wire_v1(stat.clone(), &canonical).map_err(|_| fail())?;
    if let Some(v) = independent {
        IndependentAttentionWaveEvidenceV2::from_wire_v2(v.clone(), &canonical)
            .map_err(|_| fail())?;
    }
    validate_recipe(recipe, stat, &selected)?;
    let algorithms = recipe.device.algorithm_work.as_ref().map_err(|_| fail())?;
    let host: Vec<_> = recipe
        .physical_host_rows
        .iter()
        .map(HostRow::native)
        .collect();
    let input = StructuredInputV1::from_replay_parts(
        &canonical,
        &selected,
        recipe.device.ordered_template,
        recipe.device.provider_grouped_template,
        &host,
        algorithms
            .entries
            .iter()
            .map(|v| (v.algorithm, v.kind.native(), v.commands, v.work.native())),
    )
    .map_err(numeric)?;
    if input.domain_signature() != &n.domain
        || input.regression_axes() != n.basis
        || input.joint_support_coordinates() != n.support
    {
        return Err(invalid(
            "structured numeric payload differs from recipe projection",
        ));
    }
    Ok(StructuredNumericObservationV1 {
        source: h.capture_identity,
        protocol: h.protocol,
        ordinal: n.fifo,
        membership: Some(
            StructuredMemberBindingV1::new(h.rule_signature, offer, member).map_err(numeric)?,
        ),
        call_id: n.call_id,
        fingerprint: h.fingerprint.clone().into(),
        input,
        boundary: CostBoundary::PreparationToHostSettledV1,
        outcome: WaveObservationOutcome::Completed,
        observed_at_ns: n.observed_at_ns,
        wall_ns: wall,
    })
}
fn validate_recipe(
    r: &Recipe,
    stat: &StatisticalWaveEvidenceWireV1,
    selected: &StatisticalWaveEvidenceV1,
) -> Result<(), CostProfileError> {
    let fail = || invalid("invalid structured algorithm recipe");
    let a = r.device.algorithm_work.as_ref().map_err(|_| fail())?;
    let wire = serde_json::to_value(stat)?;
    let exact: [u8; 32] = serde_json::from_value(wire["exact_binding"].clone())?;
    if r.protocol != "ferrum.structured-cost-input.v1"
        || r.exact_binding != exact
        || a.exact_binding != exact
        || a.protocol != "ferrum.device-algorithm-work.v1"
        || r.device.ordered_template == [0; 32]
        || r.device.provider_grouped_template == Some([0; 32])
        || a.ordered_command_binding == [0; 32]
        || a.physical_commands == 0
        || a.physical_commands as usize > MAX_COST_COMMANDS
        || r.device.physical_commands != a.physical_commands
        || wire["physical_commands"].as_u64() != Some(u64::from(a.physical_commands))
        || a.selected_commands == 0
        || a.selected_commands > MAX_COST_COMMANDS as u64
        || a.entries.is_empty()
        || a.entries.len() > MAX_COST_COMMANDS
        || a.aggregate_work != r.device.aggregate_work
        || a.aggregate_work.native() != selected.work()
        || !matches!(r.device.product.as_str(), "full_logits" | "greedy_token")
        || r.device.readback == CoreReadbackRoute::Unknown
    {
        return Err(fail());
    }
    let mut total = Work::default();
    let mut commands = 0u64;
    let mut prior = None;
    for entry in &a.entries {
        let key = (entry.algorithm, entry.kind);
        if entry.algorithm == [0; 32] || prior.is_some_and(|p| p >= key) || entry.commands == 0 {
            return Err(fail());
        }
        prior = Some(key);
        commands = commands.checked_add(entry.commands).ok_or_else(fail)?;
        let w = entry.work;
        if w.padded_units < w.logical_units
            || (w.logical_units > 0 && (w.inner_work_units == 0 || w.grid_blocks == 0))
        {
            return Err(fail());
        }
        macro_rules! add {($($f:ident),*)=>{$(total.$f=total.$f.checked_add(w.$f).ok_or_else(fail)?;)*};}
        add!(
            logical_units,
            padded_units,
            inner_work_units,
            grid_blocks,
            staged_weight_bytes,
            host_to_device_bytes,
            device_to_host_bytes,
            device_to_device_bytes,
            fill_bytes
        );
        total.peak_scratch_bytes = total.peak_scratch_bytes.max(w.peak_scratch_bytes);
    }
    if commands != a.selected_commands || total != a.aggregate_work {
        return Err(fail());
    }
    Ok(())
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
