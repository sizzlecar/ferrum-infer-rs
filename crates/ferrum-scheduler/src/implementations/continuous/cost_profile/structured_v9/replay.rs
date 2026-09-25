//! Replay the original ordered ledger. Missing/failed slots cannot be removed
//! and successful footer text cannot establish a qualified population.
use super::*;
pub(super) struct Replayed {
    pub header: Header,
    pub closing: PairedClock,
    pub phases: [StructuredPhaseProvenanceV9; 3],
    pub offered_attempts: u64,
    pub reserved_members: u64,
    pub model: QualifiedStructuredModelV1,
    pub oldest_observed: u64,
    pub newest_observed: u64,
}
struct Attempt {
    offered: u64,
    candidate: bool,
    rows: Vec<OfferedRow>,
    reserved: bool,
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

pub(super) fn header_valid(
    h: &Header,
    bytes: usize,
    limits: &CostProfileLoadLimits,
) -> Result<(), CostProfileError> {
    let fail = || invalid("invalid structured source header/protocol");
    h.settings.native().validate().map_err(numeric)?;
    if h.artifact_type != "ferrum.structured-live-source"
        || !matches!(h.schema_version, 1 | 2)
        || h.model_revision != MODEL_REVISION
        || h.population_revision != POPULATION_REVISION
        || h.capture_identity == [0; 32]
        || h.declared_protocol == [0; 32]
        || h.scope.domain == [0; 32]
        || h.scope.rows == 0
        || h.scope.rows > 128
        || h.maximum_offered_waves == 0
        || h.maximum_offered_waves > 65_536
        || h.maximum_file_bytes < bytes as u64
        || h.opening.wall_unix_ns == 0
        || h.opening.monotonic_ns < h.opened_at_ns
        || h.phase_members
            .iter()
            .any(|n| *n < h.settings.min_samples || *n > h.settings.max_phase_samples)
        || h.phase_members[2] < h.scope.rows + 1
    {
        return Err(fail());
    }
    let count = h
        .phase_members
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b))
        .ok_or_else(fail)?;
    if count > limits.max_samples.get() || count > h.maximum_offered_waves {
        return Err(CostProfileError::Limit("structured population bound"));
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
        return Err(CostProfileError::Limit("structured numerical memory bound"));
    }
    let mut rule = Sha256::new();
    rule.update(POPULATION_REVISION.as_bytes());
    rule.update(b"prepared-wave; all-decode-with-generated-history; before-execute; outcome-independent; one-scope\0");
    rule.update((h.scope.rows as u64).to_le_bytes());
    rule.update(h.scope.domain);
    if <[u8; 32]>::from(rule.finalize()) != h.rule_signature {
        return Err(fail());
    }
    let mut protocol = Sha256::new();
    protocol.update(b"ferrum.structured-live-source.v1\0");
    protocol.update(MODEL_REVISION.as_bytes());
    protocol.update(h.declared_protocol);
    protocol.update(h.rule_signature);
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
    let producer: Producer = serde_json::from_value(h.producer.clone())?;
    if producer.executable_bytes == 0
        || producer.executable_bytes > (1 << 30)
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
    let fail = || invalid("incomplete or inconsistent structured source population");
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
    let mut fitted: Option<FittedStructuredModelV1> = None;
    let mut calibrated: Option<CalibratedStructuredModelV1> = None;
    let mut qualified = None;
    let mut receipts = Vec::new();
    let mut closing = None;
    let mut calls = HashSet::new();
    let mut total_rows = 0usize;
    let mut oldest = u64::MAX;
    let mut newest = 0u64;
    let partition = StructuredPartitionV1 {
        source: h.capture_identity,
        protocol: h.protocol,
        population: StructuredPopulationV1::ReservedMembers {
            rule_signature: h.rule_signature,
        },
        fit_through: h.phase_members[0] as u64,
        residual_through: (h.phase_members[0] + h.phase_members[1]) as u64,
        qualification_through: h.phase_members.iter().sum::<usize>() as u64,
    };
    for line in lines {
        if closing.is_some() {
            return Err(fail());
        }
        record_ordinal = record_ordinal.checked_add(1).ok_or_else(fail)?;
        if record_ordinal > 4 * h.maximum_offered_waves as u64 + 5 {
            return Err(CostProfileError::Limit("structured source record bound"));
        }
        let wrapper = source_line(line)?;
        if wrapper.source_record_ordinal != record_ordinal {
            return Err(fail());
        }
        if h.schema_version == 2
            && wrapper.record["kind"] == "completed"
            && wrapper
                .record
                .get("selected_independent_attention_v2")
                .is_none()
        {
            return Err(invalid(
                "structured source v2 lacks original independent sidecar",
            ));
        }
        let record: Record = serde_json::from_value(wrapper.record)?;
        match record {
            Record::Offered {
                offered: n,
                member_candidate,
                phase: p,
                rows,
            } => {
                if phase >= 3
                    || p.index() != phase
                    || pending.is_some()
                    || n != offered.checked_add(1).ok_or_else(fail)?
                    || n > h.maximum_offered_waves as u64
                    || rows.is_empty()
                    || rows.len() > 1024
                {
                    return Err(fail());
                }
                let mut identities = HashSet::new();
                if rows.iter().any(|r| {
                    r.request_id.is_empty()
                        || r.request_id.len() > limits.max_source_field_bytes.get()
                        || r.owner == 0
                        || r.generation == 0
                        || !identities.insert(&r.request_id)
                }) {
                    return Err(fail());
                }
                let actual_candidate =
                    rows.len() == h.scope.rows && rows.iter().all(|r| r.decode && r.generated > 0);
                if actual_candidate != member_candidate {
                    return Err(fail());
                }
                offered = n;
                pending = Some(Attempt {
                    offered: n,
                    candidate: member_candidate,
                    rows,
                    reserved: false,
                    member: None,
                });
            }
            Record::Reserved {
                offered: n,
                member,
                phase: p,
                boundary,
            } => {
                let a = pending.as_mut().ok_or_else(fail)?;
                if p.index() != phase
                    || a.offered != n
                    || a.reserved
                    || boundary != "prepared_before_execute"
                {
                    return Err(fail());
                }
                if a.candidate {
                    if phase_members >= h.phase_members[phase] || member != members.checked_add(1) {
                        return Err(fail());
                    }
                    members += 1;
                    phase_members += 1;
                } else if member.is_some() {
                    return Err(fail());
                }
                a.reserved = true;
                a.member = member;
            }
            Record::PreparationUnavailable {
                offered: n,
                member_candidate,
                phase: p,
                reason,
            } => {
                let a = pending.take().ok_or_else(fail)?;
                if p.index() != phase
                    || a.offered != n
                    || a.reserved
                    || member_candidate != a.candidate
                    || reason.is_empty()
                {
                    return Err(fail());
                }
            }
            Record::Unsubmitted {
                offered: n,
                member,
                phase: p,
                reason,
            } => {
                let a = pending.take().ok_or_else(fail)?;
                if p.index() != phase
                    || a.offered != n
                    || !a.reserved
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
                queue,
                reconciled,
                host_stages,
                selected_structured_capture,
                selected_independent_attention_v2,
                numeric: observation,
                conversion_error,
            } => {
                let a = pending.take().ok_or_else(fail)?;
                if p.index() != phase || a.offered != n || !a.reserved || a.member != member {
                    return Err(fail());
                }
                let q = queue.ok_or_else(fail)?;
                if q.disposition != "published" || q.accepted_ordinal != last_fifo.checked_add(1) {
                    return Err(fail());
                }
                last_fifo = q.accepted_ordinal.unwrap();
                if let Some(s) = &host_stages {
                    total_rows = total_rows.checked_add(s.rows.len()).ok_or_else(fail)?;
                    if total_rows > limits.max_total_shape_rows.get()
                        || !calls.insert(s.call_id)
                        || s.call_id == 0
                    {
                        return Err(fail());
                    }
                }
                if let Some(member) = member {
                    if !reconciled || conversion_error.is_some() {
                        return Err(fail());
                    }
                    let o = observation.ok_or_else(fail)?;
                    if o.fifo != last_fifo {
                        return Err(fail());
                    }
                    let stages = host_stages.as_ref().ok_or_else(fail)?;
                    let recipe = selected_structured_capture
                        .as_ref()
                        .and_then(|v| v.as_ref().ok())
                        .ok_or_else(fail)?;
                    let sample = observation::convert(
                        &h,
                        &a.rows,
                        stages,
                        recipe,
                        selected_independent_attention_v2.as_ref(),
                        &o,
                        member,
                        n,
                    )?;
                    oldest = oldest.min(sample.observed_at_ns);
                    newest = newest.max(sample.observed_at_ns);
                    samples.push(sample);
                }
            }
            Record::PhaseFreeze { receipt: r } => {
                if phase >= 3
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
                    || r.frozen_at_ns < h.opening.monotonic_ns
                {
                    return Err(fail());
                }
                let signature = match phase {
                    0 => {
                        let m = FittedStructuredModelV1::fit(
                            h.fingerprint.clone().into(),
                            h.settings.native(),
                            partition,
                            &samples,
                            r.frozen_at_ns,
                        )
                        .map_err(numeric)?;
                        let s = m.parameters_signature();
                        fitted = Some(m);
                        s
                    }
                    1 => {
                        let m = fitted
                            .take()
                            .ok_or_else(fail)?
                            .calibrate(&samples, r.frozen_at_ns)
                            .map_err(numeric)?;
                        let s = m.parameters_signature();
                        calibrated = Some(m);
                        s
                    }
                    2 => {
                        let m = calibrated
                            .take()
                            .ok_or_else(fail)?
                            .qualify(&samples, r.frozen_at_ns)
                            .map_err(numeric)?;
                        let s = m.parameters_signature();
                        qualified = Some(m);
                        s
                    }
                    _ => return Err(fail()),
                };
                if signature != r.parameters_sha256 {
                    return Err(invalid(
                        "structured phase parameters differ from original freeze",
                    ));
                }
                receipts.push(StructuredPhaseProvenanceV9 {
                    phase: r.phase,
                    members: phase_members,
                    member_cutoff: r.member_cutoff,
                    accepted_fifo_cutoff: r.accepted_fifo_cutoff,
                    frozen_at_ns: r.frozen_at_ns,
                    source_prefix_bytes: r.source_prefix_bytes,
                    source_prefix_sha256: r.source_prefix_sha256,
                    parameters_sha256: r.parameters_sha256,
                });
                samples.clear();
                phase_members = 0;
                phase += 1;
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
                if c.monotonic_ns < receipts[2].frozen_at_ns
                    || c.wall_unix_ns < h.opening.wall_unix_ns
                {
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
        model: qualified.ok_or_else(fail)?,
        oldest_observed: oldest,
        newest_observed: newest,
    })
}
