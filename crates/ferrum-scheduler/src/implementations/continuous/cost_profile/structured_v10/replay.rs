//! Full original source3 replay. A footer or a filtered list is never a population.
use super::*;
pub(super) struct Replayed {
    pub header: Header,
    pub closing: PairedClock,
    pub phases: [StructuredPhaseProvenanceV10; 3],
    pub offered_attempts: u64,
    pub reserved_members: u64,
    pub total_shape_rows: u64,
    pub model: QualifiedStructuredModelV2,
    pub oldest_observed: u64,
    pub newest_observed: u64,
}
struct Attempt {
    offered: u64,
    cohort: usize,
    rows: Vec<OfferedRow>,
    prepared: Option<Prepared>,
    input: Option<StructuredInputV2>,
    member: Option<u64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line {
    source_record_ordinal: u64,
    record: serde_json::Value,
}
pub(super) const MAX_SOURCE_RECORD_BYTES: usize = 8 * 1024 * 1024;

fn source_line(bytes: &[u8]) -> Result<Line, CostProfileError> {
    if bytes.len() > MAX_SOURCE_RECORD_BYTES {
        return Err(CostProfileError::Limit("structured source record bytes"));
    }
    Ok(serde_json::from_slice(bytes)?)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Producer {
    executable_path: String,
    executable_sha256: String,
    executable_bytes: u64,
    package_version: String,
    source_revision: Option<String>,
}

/// Borrow common and child declarations without expanding shared payloads.
#[derive(Clone, Copy)]
pub(super) struct HeaderRef<'a> {
    pub artifact_type: &'a str,
    pub schema_version: u32,
    pub model_revision: &'a str,
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub declared_protocol: [u8; 32],
    pub rule_signature: [u8; 32],
    pub fingerprint: &'a ProfileFingerprint,
    pub producer: &'a serde_json::Value,
    pub opening: PairedClock,
    pub opened_at_ns: u64,
    pub initial_fifo_cutoff: u64,
    pub scope: &'a StructuredScopeV2,
    pub membership_rule: &'a MembershipRuleV2,
    pub cohort_plan: &'a CohortPlanV2,
    pub cohort_manifest_payload: &'a serde_json::Value,
    pub cohort_manifest_sha256: [u8; 32],
    pub phase_members: [usize; 3],
    pub maximum_offered_waves: usize,
    pub maximum_file_bytes: u64,
    pub settings: &'a Settings,
}
impl<'a> From<&'a Header> for HeaderRef<'a> {
    fn from(h: &'a Header) -> Self {
        Self {
            artifact_type: &h.artifact_type,
            schema_version: h.schema_version,
            model_revision: &h.model_revision,
            capture_identity: h.capture_identity,
            protocol: h.protocol,
            declared_protocol: h.declared_protocol,
            rule_signature: h.rule_signature,
            fingerprint: &h.fingerprint,
            producer: &h.producer,
            opening: h.opening,
            opened_at_ns: h.opened_at_ns,
            initial_fifo_cutoff: h.initial_fifo_cutoff,
            scope: &h.scope,
            membership_rule: &h.membership_rule,
            cohort_plan: &h.cohort_plan,
            cohort_manifest_payload: &h.cohort_manifest_payload,
            cohort_manifest_sha256: h.cohort_manifest_sha256,
            phase_members: h.phase_members,
            maximum_offered_waves: h.maximum_offered_waves,
            maximum_file_bytes: h.maximum_file_bytes,
            settings: &h.settings,
        }
    }
}

pub(super) fn header_valid(
    h: &Header,
    bytes: usize,
    limits: &CostProfileLoadLimits,
) -> Result<(), CostProfileError> {
    header_valid_ref(h.into(), bytes, limits, true)
}
/// `validate_common` is false only after this same immutable common declaration
/// has already passed the complete check in a shared-source import.
pub(super) fn header_valid_ref(
    h: HeaderRef<'_>,
    bytes: usize,
    limits: &CostProfileLoadLimits,
    validate_common: bool,
) -> Result<(), CostProfileError> {
    if h.model_revision != MODEL_REVISION_V2 {
        return Err(invalid("unsupported structured source revision"));
    }
    let fail = || invalid("invalid source3 header, scope or protocol");
    h.settings.native().validate().map_err(numeric_error)?;
    h.scope.validate().map_err(numeric_error)?;
    h.membership_rule.validate().map_err(numeric_error)?;
    if validate_common {
        h.cohort_plan.validate().map_err(numeric_error)?;
    }
    if h.artifact_type != "ferrum.structured-live-source"
        || h.schema_version != 3
        || h.capture_identity == [0; 32]
        || h.declared_protocol == [0; 32]
        || h.scope.owner != h.membership_rule.owner
        || h.maximum_offered_waves == 0
        || h.maximum_offered_waves > 65_536
        || h.maximum_file_bytes < bytes as u64
        || h.opening.wall_unix_ns == 0
        || h.opening.monotonic_ns < h.opened_at_ns
        || h.phase_members
            .iter()
            .any(|n| *n < h.settings.min_samples || *n > h.settings.max_phase_samples)
        || h.membership_rule.signature().map_err(numeric_error)? != h.rule_signature
        || validate_common
            && h.cohort_plan
                .signature(h.cohort_manifest_payload)
                .map_err(numeric_error)?
                != h.cohort_manifest_sha256
    {
        return Err(fail());
    }
    let count = h
        .phase_members
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b))
        .ok_or_else(fail)?;
    if count > limits.max_samples.get() || count > h.maximum_offered_waves {
        return Err(CostProfileError::Limit("source3 population bound"));
    }
    let memory = h
        .settings
        .max_phase_samples
        .checked_mul(h.settings.max_axes)
        .and_then(|n| n.checked_mul(12 * std::mem::size_of::<f64>()))
        .and_then(|n| {
            h.settings
                .max_phase_samples
                .checked_mul(128)
                .and_then(|v| {
                    v.checked_mul(
                        4 * std::mem::size_of::<
                            ferrum_interfaces::execution_cost::StructuredHostRowV1,
                        >(),
                    )
                })
                .and_then(|v| n.checked_add(v))
        });
    if memory.is_none_or(|n| n > 128 * 1024 * 1024) {
        return Err(CostProfileError::Limit("source3 numerical memory bound"));
    }
    let mut protocol = Sha256::new();
    protocol.update(b"ferrum.structured-live-source.v2\0");
    protocol.update(MODEL_REVISION_V2.as_bytes());
    protocol.update(h.declared_protocol);
    protocol.update(h.rule_signature);
    protocol.update(h.cohort_manifest_sha256);
    protocol.update(serde_json::to_vec(&h.scope)?);
    for n in h.phase_members.into_iter().map(|v| v as u64).chain([
        h.settings.min_samples as u64,
        h.settings.redundancy as u64,
        h.settings.max_phase_samples as u64,
        h.settings.max_axes as u64,
        h.settings.max_rank as u64,
        h.settings.max_wave_ns,
        h.settings.max_age_ns,
        h.settings.margin_ns,
        h.maximum_offered_waves as u64,
        h.maximum_file_bytes,
    ]) {
        protocol.update(n.to_le_bytes());
    }
    if <[u8; 32]>::from(protocol.finalize()) != h.protocol {
        return Err(fail());
    }
    if !validate_common {
        return Ok(());
    }
    let producer: Producer = serde_json::from_value(h.producer.clone())?;
    if producer.executable_bytes == 0
        || producer.executable_bytes > 1 << 30
        || producer.executable_sha256.len() != 64
        || !producer
            .executable_sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit())
    {
        return Err(fail());
    }
    for field in [
        Some(producer.executable_path.as_str()),
        Some(producer.package_version.as_str()),
        producer.source_revision.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if field.is_empty()
            || field.len() > limits.max_source_field_bytes.get()
            || field.chars().any(char::is_control)
        {
            return Err(fail());
        }
    }
    Ok(())
}
pub(super) fn replay_source(
    bytes: &[u8],
    limits: &CostProfileLoadLimits,
) -> Result<Replayed, CostProfileError> {
    let fail = || invalid("incomplete or inconsistent original source3 population");
    if bytes.is_empty() || bytes.len() > limits.max_file_bytes.get() || bytes.last() != Some(&b'\n')
    {
        return Err(fail());
    }
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let first = lines.next().ok_or_else(fail)?;
    let wrapper = source_line(first)?;
    if wrapper.source_record_ordinal != 1 {
        return Err(fail());
    }
    let h: Header = serde_json::from_value(wrapper.record)?;
    header_valid(&h, bytes.len(), limits)?;
    let mut lifecycle = lifecycle::Lifecycle::new(h.cohort_plan.clone());
    let requests = h
        .cohort_plan
        .phases
        .iter()
        .flatten()
        .map(|c| c.requests.len() as u64)
        .sum::<u64>();
    let cohorts = h
        .cohort_plan
        .phases
        .iter()
        .map(|c| c.len() as u64)
        .sum::<u64>();
    let record_limit = 4 * h.maximum_offered_waves as u64 + 2 * requests + 2 * cohorts + 8;
    let mut prefix = Sha256::new();
    prefix.update(first);
    let mut offset = first.len() as u64;
    let mut record_ordinal = 1u64;
    let mut phase = 0usize;
    let mut offered = 0u64;
    let mut members = 0u64;
    let mut phase_members = 0usize;
    let mut last_fifo = h.initial_fifo_cutoff;
    let mut pending: Option<Attempt> = None;
    let mut samples = Vec::new();
    let mut fitted: Option<FittedStructuredModelV2> = None;
    let mut calibrated: Option<CalibratedStructuredModelV2> = None;
    let mut qualified = None;
    let mut receipts = Vec::new();
    let mut closing = None;
    let mut calls = HashSet::new();
    let mut total_rows = 0usize;
    let mut oldest = u64::MAX;
    let mut newest = 0u64;
    let mut coverage_recorded = false;
    let mut last_freeze = h.opening.monotonic_ns;
    let mut last_finalized = h.opening.monotonic_ns;
    let contract = StructuredSourceContractV2 {
        capture_identity: h.capture_identity,
        protocol: h.protocol,
        membership_rule: h.rule_signature,
        cohort_manifest: h.cohort_manifest_sha256,
        phase_members: h.phase_members,
    };
    for line in lines {
        if closing.is_some() {
            return Err(fail());
        }
        record_ordinal = record_ordinal.checked_add(1).ok_or_else(fail)?;
        if record_ordinal > record_limit {
            return Err(CostProfileError::Limit("source3 record bound"));
        }
        let wrapper = source_line(line)?;
        if wrapper.source_record_ordinal != record_ordinal {
            return Err(fail());
        }
        // Missing original sidecar is different from explicitly recorded None.
        if wrapper.record["kind"] == "reserved"
            && wrapper.record["prepared"]
                .get("selected_independent_attention_v2")
                .is_none()
            || wrapper.record["kind"] == "completed"
                && wrapper
                    .record
                    .get("selected_independent_attention_v2")
                    .is_none()
        {
            return Err(invalid("source3 lacks original sidecar field"));
        }
        let record: Record = serde_json::from_value(wrapper.record)?;
        if lifecycle.expects_completion() && !matches!(&record, Record::RequestCompleted { .. }) {
            return Err(fail());
        }
        if coverage_recorded && !matches!(&record, Record::PhaseFreeze { .. }) {
            return Err(fail());
        }
        match record {
            Record::CohortBegin {
                phase: p,
                cohort,
                manifest_case,
                repetition,
            } => {
                if p.index() != phase || pending.is_some() {
                    return Err(fail());
                }
                lifecycle.begin(phase, cohort, manifest_case, repetition)?;
            }
            Record::RequestAdmitted {
                phase: p,
                cohort,
                slot,
                request_id,
                maximum_output,
            } => {
                if p.index() != phase || pending.is_some() {
                    return Err(fail());
                }
                lifecycle.admit(
                    phase,
                    cohort,
                    slot,
                    request_id,
                    maximum_output,
                    limits.max_source_field_bytes.get(),
                )?;
            }
            Record::CohortEnd {
                phase: p,
                cohort,
                admitted_count,
                completed_count,
            } => {
                if p.index() != phase || pending.is_some() {
                    return Err(fail());
                }
                lifecycle.end(phase, cohort, admitted_count, completed_count)?;
            }
            Record::RequestCompleted { request } => {
                if pending.is_some() || request.phase.index() != phase {
                    return Err(fail());
                }
                lifecycle.request_completed(request)?;
            }
            Record::Offered {
                offered: n,
                phase: p,
                cohort,
                rows,
            } => {
                lifecycle.active(phase, cohort)?;
                if phase >= 3
                    || p.index() != phase
                    || pending.is_some()
                    || n != offered.checked_add(1).ok_or_else(fail)?
                    || n > h.maximum_offered_waves as u64
                    || rows.is_empty()
                    || rows.len() > 128
                {
                    return Err(fail());
                }
                let mut ids = HashSet::new();
                for row in &rows {
                    if row.request_id.is_empty()
                        || row.request_id.len() > limits.max_source_field_bytes.get()
                        || row.owner == 0
                        || row.generation == 0
                        || !ids.insert(&row.request_id)
                    {
                        return Err(fail());
                    }
                    row.work.native().emits_token().map_err(numeric_error)?;
                }
                offered = n;
                pending = Some(Attempt {
                    offered: n,
                    cohort,
                    rows,
                    prepared: None,
                    input: None,
                    member: None,
                });
            }
            Record::Reserved {
                offered: n,
                member,
                window,
                phase: p,
                cohort,
                boundary,
                prepared,
            } => {
                let a = pending.as_mut().ok_or_else(fail)?;
                if p.index() != phase
                    || a.offered != n
                    || a.cohort != cohort
                    || a.prepared.is_some()
                    || boundary != "prepared_before_execute"
                {
                    return Err(fail());
                }
                let input = prepared::project(&prepared, &a.rows)?;
                let rows = prepared.rows.iter().map(|r| r.frontier).collect::<Vec<_>>();
                let expected_window = h
                    .membership_rule
                    .classify(input.owner(), &rows)
                    .map_err(numeric_error)?;
                if window != expected_window {
                    return Err(invalid(
                        "source3 reserved membership differs from original Prepared window",
                    ));
                }
                if window.is_some() {
                    if phase_members >= h.phase_members[phase] || member != members.checked_add(1) {
                        return Err(fail());
                    }
                    members += 1;
                    phase_members += 1;
                } else if member.is_some() {
                    return Err(fail());
                }
                lifecycle.prepared(phase, cohort, &prepared)?;
                a.prepared = Some(prepared);
                a.input = Some(input);
                a.member = member;
            }
            Record::PreparationUnavailable {
                offered: n,
                phase: p,
                cohort,
                reason,
            } => {
                let a = pending.take().ok_or_else(fail)?;
                if p.index() != phase
                    || a.offered != n
                    || a.cohort != cohort
                    || a.prepared.is_some()
                    || reason.is_empty()
                {
                    return Err(fail());
                }
            }
            Record::Unsubmitted {
                offered: n,
                member,
                phase: p,
                cohort,
                reason,
            } => {
                let a = pending.take().ok_or_else(fail)?;
                if p.index() != phase
                    || a.offered != n
                    || a.cohort != cohort
                    || a.prepared.is_none()
                    || member != a.member
                    || member.is_some()
                    || reason.is_empty()
                {
                    return Err(fail());
                }
            }
            Record::Completed {
                offered: n,
                member,
                phase: p,
                cohort,
                queue,
                reconciled,
                host_stages,
                outside_settlement,
                selected_structured_capture,
                selected_independent_attention_v2,
                numeric,
                conversion_error,
            } => {
                let a = pending.take().ok_or_else(fail)?;
                if p.index() != phase
                    || a.offered != n
                    || a.cohort != cohort
                    || a.member != member
                    || !reconciled
                    || conversion_error.is_some()
                {
                    return Err(fail());
                }
                let prepared = a.prepared.ok_or_else(fail)?;
                let input = a.input.ok_or_else(fail)?;
                let q = queue.ok_or_else(fail)?;
                if q.disposition != "published" || q.accepted_ordinal != last_fifo.checked_add(1) {
                    return Err(fail());
                }
                last_fifo = q.accepted_ordinal.unwrap();
                let (stages, binding, independent) = if member.is_some() {
                    if outside_settlement.is_some() {
                        return Err(fail());
                    }
                    let stages = host_stages.ok_or_else(fail)?;
                    let settled = stages
                        .structured_evidence
                        .as_ref()
                        .and_then(|v| v.as_ref().ok())
                        .ok_or_else(fail)?;
                    if selected_structured_capture
                        .as_ref()
                        .and_then(|v| v.as_ref().ok())
                        != Some(&prepared.recipe)
                    {
                        return Err(fail());
                    }
                    let binding = settled.stage_binding;
                    (stages, binding, selected_independent_attention_v2)
                } else {
                    if host_stages.is_some()
                        || numeric.is_some()
                        || selected_structured_capture.is_some()
                        || selected_independent_attention_v2.is_some()
                    {
                        return Err(fail());
                    }
                    let outside = outside_settlement.ok_or_else(fail)?;
                    let binding = outside.stage_binding;
                    let independent = prepared.selected_independent_attention_v2.clone();
                    (
                        observation::outside(&prepared, outside),
                        binding,
                        independent,
                    )
                };
                total_rows = total_rows.checked_add(stages.rows.len()).ok_or_else(fail)?;
                if total_rows > limits.max_total_shape_rows.get() || !calls.insert(stages.call_id) {
                    return Err(fail());
                }
                let (wall, observed) =
                    observation::validate(&h, &prepared, &stages, independent.as_ref(), binding)?;
                if observed < last_finalized
                    || stages.prepare_started_at_ns.is_none_or(|v| v < last_freeze)
                {
                    return Err(CostProfileError::Clock(
                        "source3 phase or FIFO clock backfill",
                    ));
                }
                last_finalized = observed;
                lifecycle.completed(p, cohort, &prepared, &stages, last_fifo)?;
                if let Some(member) = member {
                    let numeric = numeric.ok_or_else(fail)?;
                    if numeric.fifo != last_fifo || samples.len() >= h.phase_members[phase] {
                        return Err(fail());
                    }
                    let sample = observation::convert(
                        &h,
                        input,
                        &numeric,
                        p,
                        member,
                        n,
                        stages.call_id,
                        wall,
                        observed,
                    )?;
                    oldest = oldest.min(observed);
                    newest = newest.max(observed);
                    samples.push(sample);
                }
            }
            Record::Coverage { phase: p, report } => {
                if p.index() != phase || pending.is_some() {
                    return Err(fail());
                }
                lifecycle.freeze(phase)?;
                if report
                    != serde_json::to_value(
                        h.scope.coverage_report(&samples).map_err(numeric_error)?,
                    )?
                {
                    return Err(invalid(
                        "source3 coverage report differs from original population",
                    ));
                }
                coverage_recorded = true;
            }
            Record::PhaseFreeze { receipt: r } => {
                if phase >= 3
                    || !coverage_recorded
                    || pending.is_some()
                    || r.phase.index() != phase
                    || phase_members != h.phase_members[phase]
                    || samples.len() != phase_members
                    || r.capture_identity != h.capture_identity
                    || r.protocol != h.protocol
                    || r.rule_signature != h.rule_signature
                    || r.member_cutoff != members
                    || r.accepted_fifo_cutoff != last_fifo
                    || r.source_prefix_bytes != offset
                    || r.source_prefix_sha256 != <[u8; 32]>::from(prefix.clone().finalize())
                    || r.frozen_at_ns < last_freeze
                    || r.frozen_at_ns < last_finalized
                {
                    return Err(fail());
                }
                lifecycle.freeze(phase)?;
                let signature = match phase {
                    0 => {
                        let m = FittedStructuredModelV2::fit(
                            h.fingerprint.clone().into(),
                            h.settings.native(),
                            h.scope.clone(),
                            contract.clone(),
                            &samples,
                            r.frozen_at_ns,
                        )
                        .map_err(numeric_error)?;
                        let s = m.parameters_signature();
                        fitted = Some(m);
                        s
                    }
                    1 => {
                        let m = fitted
                            .take()
                            .ok_or_else(fail)?
                            .calibrate(&samples, r.frozen_at_ns)
                            .map_err(numeric_error)?;
                        let s = m.parameters_signature();
                        calibrated = Some(m);
                        s
                    }
                    2 => {
                        let m = calibrated
                            .take()
                            .ok_or_else(fail)?
                            .qualify(&samples, r.frozen_at_ns)
                            .map_err(numeric_error)?;
                        let s = m.parameters_signature();
                        qualified = Some(m);
                        s
                    }
                    _ => return Err(fail()),
                };
                if signature != r.parameters_sha256 {
                    return Err(invalid(
                        "source3 original frozen numerical parameters differ",
                    ));
                }
                receipts.push(StructuredPhaseProvenanceV10 {
                    phase: r.phase,
                    members: phase_members,
                    member_cutoff: r.member_cutoff,
                    accepted_fifo_cutoff: r.accepted_fifo_cutoff,
                    frozen_at_ns: r.frozen_at_ns,
                    source_prefix_bytes: r.source_prefix_bytes,
                    source_prefix_sha256: r.source_prefix_sha256,
                    parameters_sha256: r.parameters_sha256,
                });
                last_freeze = r.frozen_at_ns;
                samples.clear();
                phase_members = 0;
                phase += 1;
                coverage_recorded = false;
            }
            Record::Footer {
                phase: p,
                failure,
                offered: n,
                members: m,
                failed_members,
                accepted_fifo_cutoff,
                last_captured_fifo,
                fifo_audit_complete,
                closing: c,
            } => {
                if phase != 3
                    || p != "qualified"
                    || failure.is_some()
                    || pending.is_some()
                    || lifecycle.expects_completion()
                    || n != offered
                    || m != members
                    || failed_members != 0
                    || accepted_fifo_cutoff != last_fifo
                    || last_captured_fifo != last_fifo
                    || !fifo_audit_complete
                {
                    return Err(fail());
                }
                let c = c.ok_or_else(fail)?;
                if c.monotonic_ns < last_freeze || c.wall_unix_ns < h.opening.wall_unix_ns {
                    return Err(fail());
                }
                closing = Some(c);
            }
        }
        prefix.update(line);
        offset = offset.checked_add(line.len() as u64).ok_or_else(fail)?;
    }
    Ok(Replayed {
        header: h,
        closing: closing.ok_or_else(fail)?,
        phases: receipts.try_into().map_err(|_| fail())?,
        offered_attempts: offered,
        reserved_members: members,
        total_shape_rows: total_rows as u64,
        model: qualified.ok_or_else(fail)?,
        oldest_observed: oldest,
        newest_observed: newest,
    })
}
