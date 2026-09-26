//! Preparation is real physical work but never a structured numeric sample.
use super::*;

pub(super) fn validate(
    common: &CommonDeclarationV4,
    offered: &[Offered],
    s: &Stages,
    earliest: u64,
) -> Result<u64, CostProfileError> {
    let fail = || invalid("incomplete original source5 preparation settlement");
    if s.schema_version != 1
        || s.call_id == 0
        || s.fingerprint.as_ref() != Some(&common.fingerprint)
        || s.completeness != "complete_single_wave"
        || s.rows.len() != offered.len()
        || s.rows.is_empty()
        || s.statistical_evidence.is_some()
        || s.structured_evidence.is_some()
    {
        return Err(fail());
    }
    let shape = s.actual_shape.as_ref().ok_or_else(fail)?;
    let exact = &shape.exact;
    if exact.path != ProfileExecutionPath::PlanRuntime
        || exact.order != ProfileBatchOrder::Ordered
        || exact.restore_bytes != 0
        || exact.maintenance_bytes != 0
        || exact.maintenance_units != 0
    {
        return Err(fail());
    }
    let prepare = s.prepare_started_at_ns.ok_or_else(fail)?;
    let returned = s.executor_returned_at_ns.ok_or_else(fail)?;
    let finalized = s.finalized_at_ns.ok_or_else(fail)?;
    if prepare < earliest
        || prepare < common.opening.monotonic_ns
        || returned < prepare
        || finalized < returned
    {
        return Err(CostProfileError::Clock("source5 preparation clock order"));
    }
    let mut input_positions = HashSet::new();
    let mut host_positions = HashSet::new();
    let mut identities = HashSet::new();
    let mut end = returned;
    for row in &s.rows {
        let before = offered
            .iter()
            .find(|o| o.before.request_id == row.request_id)
            .ok_or_else(fail)?;
        let emits = before.work.emits_token().map_err(numeric_error)?;
        if row.owner_incarnation != before.before.owner_incarnation
            || row.work_generation != before.before.work_generation
            || row.actual_work.actual() != prepared::actual_work(before.work)
            || row.completeness != "complete_single_wave"
            || row.terminal.is_some()
            || row.completion_started_at_ns.is_some()
            || row.input_index as usize >= s.rows.len()
            || !input_positions.insert(row.input_index)
            || !identities.insert(&row.request_id)
        {
            return Err(fail());
        }
        let ordinal = row.host_processing_ordinal.ok_or_else(fail)?;
        let started = row.host_started_at_ns.ok_or_else(fail)?;
        let committed = row.token_committed_at_ns.ok_or_else(fail)?;
        let settled = row.settled_at_ns.ok_or_else(fail)?;
        if ordinal as usize >= s.rows.len()
            || !host_positions.insert(ordinal)
            || started < returned
            || committed < started
            || settled < committed
            || settled > finalized
            || row.output_published_at_ns.is_some() != emits
            || row
                .output_published_at_ns
                .is_some_and(|p| p < committed || p > settled)
        {
            return Err(fail());
        }
        if let Some(previous) = s
            .rows
            .iter()
            .find(|r| r.host_processing_ordinal.and_then(|v| v.checked_add(1)) == Some(ordinal))
        {
            if previous.settled_at_ns.is_none_or(|p| p > started) {
                return Err(fail());
            }
        }
        end = end.max(settled);
    }
    // The physical row index comes from the actual stage, not wire vector order.
    let mut physical = s.rows.iter().collect::<Vec<_>>();
    physical.sort_by_key(|r| r.input_index);
    let mut decode = Vec::new();
    let mut prefill = Vec::new();
    for row in physical {
        match &row.actual_work {
            RowWork::Decode { kv_tokens } => decode.push(*kv_tokens),
            RowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            } => prefill.push(ProfilePrefillShape {
                offset: *offset,
                count: std::num::NonZeroU32::new(*count).ok_or_else(fail)?,
                total_prompt_tokens: std::num::NonZeroU32::new(*total_prompt_tokens)
                    .ok_or_else(fail)?,
            }),
            _ => return Err(fail()),
        }
    }
    let kind = if decode.is_empty() {
        ProfileWaveKind::Prefill
    } else if prefill.is_empty() {
        ProfileWaveKind::Decode
    } else {
        ProfileWaveKind::Mixed
    };
    if exact.kind != kind
        || exact.decode_kv_tokens != decode
        || exact.prefill_chunks != prefill
        || end.checked_sub(prepare).filter(|v| *v > 0) != s.full_wall_ns
    {
        return Err(fail());
    }
    Ok(finalized)
}
