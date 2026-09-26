//! Replay one original physical transcript. Numerical populations remain private
//! to each declared child; neither this module nor its DTOs grant live authority.
use super::*;

pub(super) struct ReplayedV4 {
    pub capture_protocol: [u8; 32],
    pub children: Vec<ReplayedChildV4>,
}

pub(super) struct ReplayedChildV4 {
    pub header: ChildHeaderV4,
    pub closing: PairedClock,
    pub phases: [StructuredPhaseProvenanceV10; 3],
    pub offered_attempts: u64,
    pub reserved_members: u64,
    pub total_shape_rows: u64,
    pub model: QualifiedStructuredModelV2,
    pub oldest_observed: u64,
    pub newest_observed: u64,
}

struct Child {
    header: ChildHeaderV4,
    contract: StructuredSourceContractV2,
    members: u64,
    phase_members: usize,
    samples: Vec<StructuredNumericObservationV2>,
    fitted: Option<FittedStructuredModelV2>,
    calibrated: Option<CalibratedStructuredModelV2>,
    qualified: Option<QualifiedStructuredModelV2>,
    receipts: Vec<StructuredPhaseProvenanceV10>,
    last_freeze: u64,
    oldest: u64,
    newest: u64,
}
impl Child {
    fn new(header: ChildHeaderV4) -> Self {
        Self {
            contract: StructuredSourceContractV2 {
                capture_identity: header.capture_identity,
                protocol: header.protocol,
                membership_rule: header.rule_signature,
                cohort_manifest: header.common.cohort_manifest_sha256,
                phase_members: header.phase_members,
            },
            last_freeze: header.common.opening.monotonic_ns,
            header,
            members: 0,
            phase_members: 0,
            samples: Vec::new(),
            fitted: None,
            calibrated: None,
            qualified: None,
            receipts: Vec::new(),
            oldest: u64::MAX,
            newest: 0,
        }
    }

    fn freeze(&mut self, phase: usize, r: Freeze) -> Result<(), CostProfileError> {
        let fail = failure;
        let signature = match phase {
            0 => {
                let model = FittedStructuredModelV2::fit(
                    self.header.common.fingerprint.clone().into(),
                    self.header.settings.native(),
                    self.header.scope.clone(),
                    self.contract.clone(),
                    &self.samples,
                    r.frozen_at_ns,
                )
                .map_err(numeric_error)?;
                let signature = model.parameters_signature();
                self.fitted = Some(model);
                signature
            }
            1 => {
                let model = self
                    .fitted
                    .take()
                    .ok_or_else(fail)?
                    .calibrate(&self.samples, r.frozen_at_ns)
                    .map_err(numeric_error)?;
                let signature = model.parameters_signature();
                self.calibrated = Some(model);
                signature
            }
            2 => {
                let model = self
                    .calibrated
                    .take()
                    .ok_or_else(fail)?
                    .qualify(&self.samples, r.frozen_at_ns)
                    .map_err(numeric_error)?;
                let signature = model.parameters_signature();
                self.qualified = Some(model);
                signature
            }
            _ => return Err(fail()),
        };
        if signature != r.parameters_sha256 {
            return Err(invalid(
                "source4 original child numerical parameters differ",
            ));
        }
        self.receipts.push(StructuredPhaseProvenanceV10 {
            phase: r.phase,
            members: self.phase_members,
            member_cutoff: r.member_cutoff,
            accepted_fifo_cutoff: r.accepted_fifo_cutoff,
            frozen_at_ns: r.frozen_at_ns,
            source_prefix_bytes: r.source_prefix_bytes,
            source_prefix_sha256: r.source_prefix_sha256,
            parameters_sha256: r.parameters_sha256,
        });
        self.last_freeze = r.frozen_at_ns;
        self.samples.clear();
        self.phase_members = 0;
        Ok(())
    }
}

struct Attempt {
    offered: u64,
    cohort: usize,
    rows: Vec<OfferedRow>,
    prepared: Option<Prepared>,
    input: Option<StructuredInputV2>,
    members: Vec<Option<u64>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Line {
    source_record_ordinal: u64,
    record: serde_json::Value,
}
fn failure() -> CostProfileError {
    invalid("incomplete or inconsistent original source4 population")
}
fn source_line(bytes: &[u8]) -> Result<Line, CostProfileError> {
    if bytes.len() > super::super::replay::MAX_SOURCE_RECORD_BYTES {
        return Err(CostProfileError::Limit("structured source record bytes"));
    }
    Ok(serde_json::from_slice(bytes)?)
}

fn validate_header(
    h: &HeaderV4,
    bytes: usize,
    limits: &CostProfileLoadLimits,
) -> Result<(), CostProfileError> {
    validate_shared_header(h, bytes, limits, false)
}
fn validate_shared_header(
    h: &HeaderV4,
    bytes: usize,
    limits: &CostProfileLoadLimits,
    prefix_source: bool,
) -> Result<(), CostProfileError> {
    if !prefix_source
        && (h.artifact_type != "ferrum.structured-shared-live-source"
            || h.schema_version != 4
            || h.model_revision != MODEL_REVISION_V2)
        || h.children.is_empty()
        || h.maximum_children == 0
        || h.children.len() > h.maximum_children
        || h.maximum_children > 128
        || h.maximum_retained_numeric_bytes == 0
        || h.maximum_retained_numeric_bytes > 512 * 1024 * 1024
        || h.maximum_retained_coordinates == 0
        || h.maximum_retained_coordinates > 16_777_216
        || h.maximum_file_bytes < bytes as u64
        || (!prefix_source && h.capture_protocol != h.signature()?)
    {
        return Err(failure());
    }
    let mut samples = 0usize;
    let mut numeric = 0usize;
    let mut coordinates = 0usize;
    for (index, child) in h.children.iter().enumerate() {
        super::super::replay::header_valid_ref(
            child.view(&h.common, h.maximum_file_bytes),
            bytes,
            limits,
            index == 0,
        )?;
        if h.children[..index].iter().any(|old| {
            old.scope.owner == child.scope.owner || old.capture_identity == child.capture_identity
        }) {
            return Err(invalid("source4 children lack unique identity"));
        }
        for n in child.phase_members {
            samples = samples.checked_add(n).ok_or_else(failure)?;
            coordinates = n
                .checked_mul(child.settings.max_axes)
                .and_then(|n| n.checked_mul(2))
                .and_then(|n| coordinates.checked_add(n))
                .ok_or_else(failure)?;
        }
        let child_numeric = child
            .settings
            .max_phase_samples
            .checked_mul(child.settings.max_axes)
            .and_then(|n| n.checked_mul(12 * std::mem::size_of::<f64>()))
            .and_then(|n| {
                child
                    .settings
                    .max_phase_samples
                    .checked_mul(128)
                    .and_then(|rows| {
                        rows.checked_mul(
                            4 * std::mem::size_of::<
                                ferrum_interfaces::execution_cost::StructuredHostRowV1,
                            >(),
                        )
                    })
                    .and_then(|host| n.checked_add(host))
            })
            .ok_or_else(failure)?;
        numeric = numeric.checked_add(child_numeric).ok_or_else(failure)?;
    }
    if samples > limits.max_samples.get()
        || numeric > h.maximum_retained_numeric_bytes
        || coordinates > h.maximum_retained_coordinates
    {
        return Err(CostProfileError::Limit(
            "source4 aggregate numerical population",
        ));
    }
    Ok(())
}

pub(super) fn replay_source(
    bytes: &[u8],
    limits: &CostProfileLoadLimits,
) -> Result<ReplayedV4, CostProfileError> {
    replay_with_observer(bytes, limits, || {})
}

// The observer is a monomorphized no-op in production. Tests count successful
// physical validations at this exact boundary, not estimates from child totals.
fn replay_with_observer(
    bytes: &[u8],
    limits: &CostProfileLoadLimits,
    physical_validated: impl FnMut(),
) -> Result<ReplayedV4, CostProfileError> {
    replay_driver(bytes, limits, physical_validated, false)
}
pub(super) fn replay_prefix_source(
    bytes: &[u8],
    limits: &CostProfileLoadLimits,
) -> Result<ReplayedV4, CostProfileError> {
    replay_driver(bytes, limits, || {}, true)
}
pub(super) fn replay_driver(
    bytes: &[u8],
    limits: &CostProfileLoadLimits,
    mut physical_validated: impl FnMut(),
    prefix_source: bool,
) -> Result<ReplayedV4, CostProfileError> {
    limits.validate()?;
    if bytes.is_empty() || bytes.len() > limits.max_file_bytes.get() || bytes.last() != Some(&b'\n')
    {
        return Err(failure());
    }
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let first = lines.next().ok_or_else(failure)?;
    let wrapper = source_line(first)?;
    if wrapper.source_record_ordinal != 1 {
        return Err(failure());
    }
    let (header, prefix_plan) = if prefix_source {
        let (header, plan) = prefix::header(wrapper.record)?;
        (header, Some(plan))
    } else {
        (serde_json::from_value::<HeaderV4>(wrapper.record)?, None)
    };
    if prefix_source {
        validate_shared_header(&header, bytes.len(), limits, true)?;
    } else {
        validate_header(&header, bytes.len(), limits)?;
    }
    let mut preparation = prefix_plan.map(prefix::Preparation::new);
    let capture_protocol = header.capture_protocol;
    let h = &header.common;
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
    // Source5 adds exactly one release record per declared request, while the
    // unchanged record-byte/file/offer/physical-row budgets remain in force.
    let record_limit = 4 * h.maximum_offered_waves as u64
        + (2 + u64::from(prefix_source)) * requests
        + 2 * cohorts
        + 8;
    let maximum_offered = h.maximum_offered_waves as u64;
    let mut last_fifo = h.initial_fifo_cutoff;
    let mut last_finalized = h.opening.monotonic_ns;
    let common = Arc::new(header.common);
    let mut children = header
        .children
        .into_iter()
        .map(|declaration| {
            Child::new(ChildHeaderV4 {
                declaration,
                common: Arc::clone(&common),
                maximum_file_bytes: header.maximum_file_bytes,
            })
        })
        .collect::<Vec<_>>();
    let mut prefix = Sha256::new();
    prefix.update(first);
    let mut offset = first.len() as u64;
    let mut ordinal = 1u64;
    let mut phase = 0usize;
    let mut offered = 0u64;
    let mut pending: Option<Attempt> = None;
    let mut closing = None;
    let mut calls = HashSet::new();
    let mut total_rows = 0usize;
    let mut coverage_recorded = false;
    for line in lines {
        if closing.is_some() {
            return Err(failure());
        }
        ordinal = ordinal.checked_add(1).ok_or_else(failure)?;
        if ordinal > record_limit {
            return Err(CostProfileError::Limit("source4 record bound"));
        }
        let wrapper = source_line(line)?;
        if wrapper.source_record_ordinal != ordinal {
            return Err(failure());
        }
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
            return Err(invalid("source4 lacks original sidecar field"));
        }
        if prefix::is_preparation(&wrapper.record) {
            if pending.is_some()
                || coverage_recorded
                || lifecycle.expects_completion()
                || phase >= 3
            {
                return Err(failure());
            }
            let prep = preparation
                .as_mut()
                .ok_or_else(|| invalid("source4 cannot contain source5 preparation"))?;
            let earliest = children
                .iter()
                .map(|c| c.last_freeze.max(c.header.opened_at_ns))
                .max()
                .ok_or_else(failure)?;
            let validated = prep.handle(
                wrapper.record,
                &mut prefix::Progress {
                    phase,
                    offered: &mut offered,
                    maximum_offered,
                    last_fifo: &mut last_fifo,
                    last_finalized: &mut last_finalized,
                    earliest,
                    calls: &mut calls,
                    total_rows: &mut total_rows,
                    common: &common,
                    limits,
                    lifecycle: &mut lifecycle,
                },
            )?;
            if validated {
                physical_validated();
            }
            prefix.update(line);
            offset = offset.checked_add(line.len() as u64).ok_or_else(failure)?;
            continue;
        }
        if preparation.as_ref().is_some_and(|p| !p.idle()) {
            return Err(failure());
        }
        let record: RecordV4 = serde_json::from_value(wrapper.record)?;
        if lifecycle.expects_completion()
            && !matches!(
                &record,
                RecordV4::Common {
                    record: Record::RequestCompleted { .. }
                }
            )
        {
            return Err(failure());
        }
        if coverage_recorded && !matches!(&record, RecordV4::PhaseFreeze { .. }) {
            return Err(failure());
        }
        match record {
            RecordV4::Common { record } => match record {
                Record::CohortBegin {
                    phase: p,
                    cohort,
                    manifest_case,
                    repetition,
                } => {
                    if p.index() != phase || pending.is_some() {
                        return Err(failure());
                    }
                    lifecycle.begin(phase, cohort, manifest_case, repetition)?;
                    if let Some(p) = &mut preparation {
                        p.begin(phase, cohort, &common.cohort_plan)?;
                    }
                }
                Record::RequestAdmitted {
                    phase: p,
                    cohort,
                    slot,
                    request_id,
                    maximum_output,
                } => {
                    if p.index() != phase || pending.is_some() {
                        return Err(failure());
                    }
                    if let Some(p) = &mut preparation {
                        p.admit(slot, &request_id)?;
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
                        return Err(failure());
                    }
                    lifecycle.end(phase, cohort, admitted_count, completed_count)?;
                    if let Some(p) = &mut preparation {
                        p.end()?;
                    }
                }
                Record::RequestCompleted { request } => {
                    if pending.is_some() || request.phase.index() != phase {
                        return Err(failure());
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
                    if let Some(p) = &preparation {
                        p.ready()?;
                    }
                    if phase >= 3
                        || p.index() != phase
                        || pending.is_some()
                        || n != offered.checked_add(1).ok_or_else(failure)?
                        || n > maximum_offered
                        || rows.is_empty()
                        || rows.len() > 128
                    {
                        return Err(failure());
                    }
                    let mut ids = HashSet::new();
                    for row in &rows {
                        if row.request_id.is_empty()
                            || row.request_id.len() > limits.max_source_field_bytes.get()
                            || row.owner == 0
                            || row.generation == 0
                            || !ids.insert(&row.request_id)
                        {
                            return Err(failure());
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
                        members: Vec::new(),
                    });
                }
                Record::PreparationUnavailable {
                    offered: n,
                    phase: p,
                    cohort,
                    reason,
                } => {
                    let a = pending.take().ok_or_else(failure)?;
                    if p.index() != phase
                        || a.offered != n
                        || a.cohort != cohort
                        || a.prepared.is_some()
                        || reason.is_empty()
                    {
                        return Err(failure());
                    }
                }
                _ => {
                    return Err(invalid(
                        "source4 common record cannot contain child-dependent state",
                    ));
                }
            },
            RecordV4::Reserved {
                offered: n,
                phase: p,
                cohort,
                boundary,
                prepared,
                memberships,
            } => {
                let a = pending.as_mut().ok_or_else(failure)?;
                if phase >= 3
                    || p.index() != phase
                    || a.offered != n
                    || a.cohort != cohort
                    || a.prepared.is_some()
                    || boundary != "prepared_before_execute"
                    || memberships.len() != children.len()
                {
                    return Err(failure());
                }
                let input = prepared::project(&prepared, &a.rows)?;
                if let Some(p) = &preparation {
                    p.prepared(&prepared)?;
                }
                let rows = prepared.rows.iter().map(|r| r.frontier).collect::<Vec<_>>();
                for (child, membership) in children.iter_mut().zip(&memberships) {
                    let window = child
                        .header
                        .membership_rule
                        .classify(input.owner(), &rows)
                        .map_err(numeric_error)?;
                    if membership.window != window {
                        return Err(invalid(
                            "source4 child membership differs from original Prepared window",
                        ));
                    }
                    if window.is_some() {
                        if child.phase_members >= child.header.phase_members[phase]
                            || membership.member != child.members.checked_add(1)
                        {
                            return Err(failure());
                        }
                        child.members += 1;
                        child.phase_members += 1;
                    } else if membership.member.is_some() {
                        return Err(failure());
                    }
                }
                lifecycle.prepared(phase, cohort, &prepared)?;
                a.prepared = Some(prepared);
                a.input = Some(input);
                a.members = memberships.into_iter().map(|m| m.member).collect();
            }
            RecordV4::Unsubmitted {
                offered: n,
                phase: p,
                cohort,
                members,
                reason,
            } => {
                let a = pending.take().ok_or_else(failure)?;
                if p.index() != phase
                    || a.offered != n
                    || a.cohort != cohort
                    || a.prepared.is_none()
                    || members != a.members
                    || members.iter().any(Option::is_some)
                    || reason.is_empty()
                {
                    return Err(failure());
                }
            }
            RecordV4::Completed {
                offered: n,
                phase: p,
                cohort,
                members,
                queue,
                reconciled,
                host_stages,
                outside_settlement,
                selected_structured_capture,
                selected_independent_attention_v2,
                numeric,
                conversion_error,
            } => {
                let a = pending.take().ok_or_else(failure)?;
                if phase >= 3
                    || p.index() != phase
                    || a.offered != n
                    || a.cohort != cohort
                    || a.members != members
                    || members.len() != children.len()
                    || !reconciled
                    || conversion_error.is_some()
                {
                    return Err(failure());
                }
                let prepared = a.prepared.ok_or_else(failure)?;
                let input = a.input.ok_or_else(failure)?;
                let q = queue.ok_or_else(failure)?;
                if q.disposition != "published" || q.accepted_ordinal != last_fifo.checked_add(1) {
                    return Err(failure());
                }
                last_fifo = q.accepted_ordinal.ok_or_else(failure)?;
                let (stages, binding, independent) = if members.iter().any(Option::is_some) {
                    if outside_settlement.is_some() {
                        return Err(failure());
                    }
                    let stages = host_stages.ok_or_else(failure)?;
                    let settled = stages
                        .structured_evidence
                        .as_ref()
                        .and_then(|v| v.as_ref().ok())
                        .ok_or_else(failure)?;
                    if selected_structured_capture
                        .as_ref()
                        .and_then(|v| v.as_ref().ok())
                        != Some(&prepared.recipe)
                    {
                        return Err(failure());
                    }
                    let binding = settled.stage_binding;
                    (stages, binding, selected_independent_attention_v2)
                } else {
                    if host_stages.is_some()
                        || numeric.is_some()
                        || selected_structured_capture.is_some()
                        || selected_independent_attention_v2.is_some()
                    {
                        return Err(failure());
                    }
                    let outside = outside_settlement.ok_or_else(failure)?;
                    let binding = outside.stage_binding;
                    let independent = prepared.selected_independent_attention_v2.clone();
                    (
                        observation::outside(&prepared, outside),
                        binding,
                        independent,
                    )
                };
                total_rows = total_rows
                    .checked_add(stages.rows.len())
                    .ok_or_else(failure)?;
                if total_rows > limits.max_total_shape_rows.get() || !calls.insert(stages.call_id) {
                    return Err(failure());
                }
                // One full physical verification regardless of the child count.
                let (wall, observed) = observation::validate(
                    children[0].header.view(),
                    &prepared,
                    &stages,
                    independent.as_ref(),
                    binding,
                )?;
                if let Some(p) = &mut preparation {
                    p.observed_call(stages.call_id)?;
                }
                physical_validated();
                if observed < last_finalized
                    || children.iter().any(|child| {
                        stages.prepare_started_at_ns.is_none_or(|t| {
                            t < child.last_freeze
                                || t < child.header.common.opening.monotonic_ns
                                || t < child.header.opened_at_ns
                        })
                    })
                {
                    return Err(CostProfileError::Clock(
                        "source4 child phase or FIFO clock backfill",
                    ));
                }
                last_finalized = observed;
                lifecycle.completed(p, cohort, &prepared, &stages, last_fifo)?;
                if let Some(p) = &mut preparation {
                    p.ordinary_completed(&prepared)?;
                }
                // Unique declared owners make at most one child a member. Move
                // the private projection into that population without copying
                // its recipe-derived axes or allocating a second projection.
                let mut input = Some(input);
                for (child, member) in children.iter_mut().zip(members) {
                    if let Some(member) = member {
                        let numeric = numeric.as_ref().ok_or_else(failure)?;
                        if numeric.fifo != last_fifo
                            || child.samples.len() >= child.header.phase_members[phase]
                        {
                            return Err(failure());
                        }
                        let sample = observation::convert(
                            child.header.view(),
                            input.take().ok_or_else(failure)?,
                            numeric,
                            p,
                            member,
                            n,
                            stages.call_id,
                            wall,
                            observed,
                        )?;
                        child.oldest = child.oldest.min(observed);
                        child.newest = child.newest.max(observed);
                        child.samples.push(sample);
                    }
                }
            }
            RecordV4::Coverage { phase: p, reports } => {
                if p.index() != phase || pending.is_some() || reports.len() != children.len() {
                    return Err(failure());
                }
                lifecycle.freeze(phase)?;
                for (child, report) in children.iter().zip(reports) {
                    if report
                        != serde_json::to_value(
                            child
                                .header
                                .scope
                                .coverage_report(&child.samples)
                                .map_err(numeric_error)?,
                        )?
                    {
                        return Err(invalid(
                            "source4 child coverage differs from original population",
                        ));
                    }
                }
                coverage_recorded = true;
            }
            RecordV4::PhaseFreeze { receipts } => {
                if phase >= 3
                    || !coverage_recorded
                    || pending.is_some()
                    || receipts.len() != children.len()
                {
                    return Err(failure());
                }
                lifecycle.freeze(phase)?;
                let expected_prefix: [u8; 32] = prefix.clone().finalize().into();
                // Validate every child's boundary before any numerical phase advances.
                for (child, r) in children.iter().zip(&receipts) {
                    let h = &child.header;
                    if r.phase.index() != phase
                        || child.phase_members != h.phase_members[phase]
                        || child.samples.len() != child.phase_members
                        || r.capture_identity != h.capture_identity
                        || r.protocol != h.protocol
                        || r.rule_signature != h.rule_signature
                        || r.member_cutoff != child.members
                        || r.accepted_fifo_cutoff != last_fifo
                        || r.source_prefix_bytes != offset
                        || r.source_prefix_sha256 != expected_prefix
                        || r.frozen_at_ns < child.last_freeze
                        || r.frozen_at_ns < last_finalized
                        || r.frozen_at_ns != receipts[0].frozen_at_ns
                    {
                        return Err(failure());
                    }
                }
                for (child, receipt) in children.iter_mut().zip(receipts) {
                    child.freeze(phase, receipt)?;
                }
                phase += 1;
                coverage_recorded = false;
            }
            RecordV4::PhaseFailed { .. } => {
                return Err(invalid("source4 original group phase failed"));
            }
            RecordV4::Footer {
                phase: p,
                failure: error,
                offered: n,
                members,
                failed_members,
                accepted_fifo_cutoff,
                last_captured_fifo,
                fifo_audit_complete,
                closing: c,
            } => {
                if phase != 3
                    || p != "qualified"
                    || error.is_some()
                    || pending.is_some()
                    || lifecycle.expects_completion()
                    || n != offered
                    || members.len() != children.len()
                    || failed_members.len() != children.len()
                    || children
                        .iter()
                        .zip(members)
                        .any(|(child, n)| child.members != n)
                    || failed_members.iter().any(|n| *n != 0)
                    || accepted_fifo_cutoff != last_fifo
                    || last_captured_fifo != last_fifo
                    || !fifo_audit_complete
                {
                    return Err(failure());
                }
                let c = c.ok_or_else(failure)?;
                if children.iter().any(|child| {
                    c.monotonic_ns < child.last_freeze
                        || c.wall_unix_ns < child.header.common.opening.wall_unix_ns
                }) {
                    return Err(failure());
                }
                closing = Some(c);
            }
        }
        prefix.update(line);
        offset = offset.checked_add(line.len() as u64).ok_or_else(failure)?;
    }
    let closing = closing.ok_or_else(failure)?;
    let children = children
        .into_iter()
        .map(|child| {
            Ok(ReplayedChildV4 {
                header: child.header,
                closing,
                phases: child.receipts.try_into().map_err(|_| failure())?,
                offered_attempts: offered,
                reserved_members: child.members,
                total_shape_rows: total_rows as u64,
                model: child.qualified.ok_or_else(failure)?,
                oldest_observed: child.oldest,
                newest_observed: child.newest,
            })
        })
        .collect::<Result<_, CostProfileError>>()?;
    Ok(ReplayedV4 {
        capture_protocol,
        children,
    })
}

#[cfg(test)]
#[path = "replay/tests.rs"]
pub(super) mod tests;
