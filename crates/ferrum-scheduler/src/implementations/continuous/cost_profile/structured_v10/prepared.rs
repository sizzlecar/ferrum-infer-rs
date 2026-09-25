//! Rebuild numerical input from original pre-execution wire. No live receipt is created.
use super::*;
use ferrum_interfaces::execution_cost::*;

pub(super) fn actual_work(work: PreparedWorkV2) -> ActualRowWork {
    match work {
        PreparedWorkV2::Decode { kv_tokens } => ActualRowWork::Decode { kv_tokens },
        PreparedWorkV2::Prefill {
            offset,
            count,
            total_prompt_tokens,
        } => ActualRowWork::Prefill {
            offset,
            count,
            total_prompt_tokens,
        },
    }
}
pub(super) fn project(
    p: &Prepared,
    offered: &[OfferedRow],
) -> Result<StructuredInputV2, CostProfileError> {
    let fail = || invalid("invalid original Prepared facts or membership projection");
    let n = p.rows.len();
    if n == 0 || n > 128 || offered.len() != n || p.recipe.physical_host_rows.len() != n {
        return Err(fail());
    }
    let e = &p.exact.exact;
    if e.path != ProfileExecutionPath::PlanRuntime
        || e.graph_state != ProfileGraphState::Disabled
        || e.order != ProfileBatchOrder::Ordered
        || e.restore_bytes != 0
        || e.maintenance_bytes != 0
        || e.maintenance_units != 0
    {
        return Err(fail());
    }
    let features = p.exact.numeric_features.as_ref().ok_or_else(fail)?;
    features.validate(n).map_err(|_| fail())?;
    p.exact
        .host_content_features
        .as_ref()
        .ok_or_else(fail)?
        .validate()
        .map_err(|_| fail())?;
    let multiset = p.exact.row_multiset_features.as_ref().ok_or_else(fail)?;
    multiset.validate(n).map_err(|_| fail())?;
    let mut decode = Vec::new();
    let mut prefill = Vec::new();
    let mut identities = HashSet::new();
    for (i, ((row, host), numeric)) in p
        .rows
        .iter()
        .zip(&p.recipe.physical_host_rows)
        .zip(&features.rows)
        .enumerate()
    {
        row.frontier.validate().map_err(numeric_error)?;
        let offered = offered
            .iter()
            .find(|o| o.request_id == row.request_id)
            .ok_or_else(fail)?;
        let f = &row.frontier;
        if !identities.insert(&row.request_id)
            || row.owner_incarnation == 0
            || row.work_generation == 0
            || row.owner_incarnation != offered.owner
            || row.work_generation != offered.generation
            || f.generated_before != offered.generated
            || f.work != offered.work.native()
            || f.physical_position as usize != i
            || host.physical_position as usize != i
            || f.generated_before != numeric.generated_tokens_before
            || f.maximum_output != numeric.maximum_output_tokens
            || host.no_generated_history != (f.generated_before == 0)
            || host.role != multiset.rows[i].role
            || !host.role.matches_work(actual_work(f.work))
        {
            return Err(fail());
        }
        let emits = f.work.emits_token().map_err(numeric_error)?;
        if numeric.decoded_prefix_tokens
            != f.generated_before
                .checked_add(u64::from(emits))
                .ok_or_else(fail)?
            || numeric.sampling_history_tokens != f.generated_before
            || numeric.decoded_text_bytes_bound
                != numeric
                    .decoded_prefix_tokens
                    .checked_mul(host.installed_policy.decoder_text_bytes_per_token)
                    .ok_or_else(fail)?
            || numeric.decode_scratch_bytes_bound
                != numeric
                    .decoded_prefix_tokens
                    .checked_mul(host.installed_policy.decoder_scratch_bytes_per_token)
                    .ok_or_else(fail)?
        {
            return Err(fail());
        }
        let terminal = if !emits {
            Expectation::NoTokenProduced
        } else if numeric.decoded_prefix_tokens == f.maximum_output {
            Expectation::LengthBoundary
        } else {
            Expectation::TokenMayTerminate
        };
        if host.terminal_expectation != terminal {
            return Err(fail());
        }
        match f.work {
            PreparedWorkV2::Decode { kv_tokens } => {
                if host.initial_prefill
                    || host.final_prefill
                    || host.decode_requires_full_logits.is_none()
                    || host.repetition_penalty_bits.is_none()
                {
                    return Err(fail());
                }
                decode.push(kv_tokens);
            }
            PreparedWorkV2::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => {
                if host.initial_prefill != (offset == 0)
                    || host.final_prefill != emits
                    || host.decode_requires_full_logits.is_some()
                    || host.repetition_penalty_bits.is_some()
                    || numeric.repetition_tokens != 0
                {
                    return Err(fail());
                }
                prefill.push(ProfilePrefillShape {
                    offset,
                    count: std::num::NonZeroU32::new(count).ok_or_else(fail)?,
                    total_prompt_tokens: std::num::NonZeroU32::new(total_prompt_tokens)
                        .ok_or_else(fail)?,
                });
            }
        }
    }
    let (kind, profile_kind) = if decode.is_empty() {
        (ActualWaveKind::Prefill, ProfileWaveKind::Prefill)
    } else if prefill.is_empty() {
        (ActualWaveKind::Decode, ProfileWaveKind::Decode)
    } else {
        (ActualWaveKind::Mixed, ProfileWaveKind::Mixed)
    };
    if e.kind != profile_kind || e.decode_kv_tokens != decode || e.prefill_chunks != prefill {
        return Err(fail());
    }
    let canonical = CanonicalWaveCostShape {
        kind,
        path: ActualWavePath::PlanRuntime,
        graph: ActualWaveGraphState::Disabled,
        row_order: ActualWaveRowOrder::Ordered,
        provider_signature: e.provider_signature,
        output_policy_signature: e.output_policy_signature,
        numeric_features: p.exact.numeric_features.clone(),
        host_content_features: p.exact.host_content_features,
        row_multiset_features: p.exact.row_multiset_features.clone(),
        rows: p
            .rows
            .iter()
            .map(|r| actual_work(r.frontier.work))
            .collect(),
        recurrent_state_bytes: e.recurrent_state_bytes,
    };
    let selected = StatisticalWaveEvidenceV1::from_wire_v1(p.selected.clone(), &canonical)
        .map_err(|_| fail())?;
    if let Some(v) = &p.selected_independent_attention_v2 {
        IndependentAttentionWaveEvidenceV2::from_wire_v2(v.clone(), &canonical)
            .map_err(|_| fail())?;
    }
    validate_recipe(&p.recipe, &p.selected, &selected)?;
    let host = p
        .recipe
        .physical_host_rows
        .iter()
        .map(HostRow::native)
        .collect::<Vec<_>>();
    let algorithms = p
        .recipe
        .device
        .algorithm_work
        .as_ref()
        .map_err(|_| fail())?
        .entries
        .iter()
        .map(|a| (a.algorithm, a.kind.native(), a.commands, a.work.native()))
        .collect::<Vec<_>>();
    let product = match p.recipe.device.product.as_str() {
        "greedy_token" => StructuredProductV2::GreedyToken,
        "full_logits" => StructuredProductV2::FullLogits,
        _ => return Err(fail()),
    };
    let (input, facts) = StructuredInputV2::from_replay_parts(
        &canonical,
        &selected,
        p.recipe.device.ordered_template,
        p.recipe.device.provider_grouped_template,
        product,
        p.recipe.device.readback,
        &host,
        &algorithms,
    )
    .map_err(numeric_error)?;
    if serde_json::to_value(facts)? != p.owner_facts {
        return Err(invalid("Prepared owner facts differ from original recipe"));
    }
    Ok(input)
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
        entry
            .kind
            .native()
            .validate_work(w.native())
            .map_err(|_| fail())?;
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
