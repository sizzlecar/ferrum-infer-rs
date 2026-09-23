use std::collections::{BTreeMap, BTreeSet};
use std::io::BufRead;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Default, Serialize)]
pub(super) struct Counts {
    input_records: u64,
    native_records: u64,
    physical_submission_records: u64,
    ignored_records: u64,
    duplicate_native_records: u64,
    duplicate_submission_records: u64,
    records_missing_identity: u64,
}

#[derive(Debug, Default, Clone, Serialize)]
pub(super) struct Timing {
    commands: u64,
    timing_statuses: BTreeMap<String, u64>,
    unavailable_reasons: BTreeMap<String, u64>,
    measured_commands: u64,
    measured_ns: u128,
    invalid_timing_commands: u64,
    missing_elapsed_commands: u64,
    missing_intervals_commands: u64,
    missing_shape_commands: u64,
    unlabeled_interval_count: u64,
    unlabeled_interval_ns: u128,
}

#[derive(Debug, Default, Clone, Serialize)]
pub(super) struct Subwork {
    interval_count: u64,
    measured_ns: u128,
}

#[derive(Debug, Default, Serialize)]
pub(super) struct Breakdown {
    timing: Timing,
    operations: BTreeMap<String, Timing>,
    /// Decomposes command intervals; never added to the command total again.
    subwork: BTreeMap<String, Subwork>,
    command_phases: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
pub(super) struct Submission {
    fingerprint: String,
    /// Exact native command shapes, including output heads and housekeeping.
    observed_command_token_counts: Vec<u64>,
    maximum_command_token_count: Option<u64>,
    /// Compute-only participant evidence; housekeeping can name the whole wave.
    compute_command_shapes: Vec<ComputeCommandShape>,
    observed_compute_participant_counts: Vec<u64>,
    compute_commands_missing_participant_evidence: u64,
    compute_commands_invalid_participant_evidence: u64,
    /// Present only when every observed compute command has complete, agreeing evidence.
    uniform_verified_compute_participant_count: Option<u64>,
    /// No prefill/decode inference is made from token counts.
    explicit_wave_phases: Vec<String>,
    phase: Option<String>,
    physical_submission_record_present: bool,
    physical_submission_timing_status: Option<String>,
    physical_submission_unavailable_reason: Option<String>,
    declared_command_count: Option<u64>,
    missing_declared_commands: Option<u64>,
    command_index_gaps_below_maximum: u64,
    native_work: Breakdown,
}

#[derive(Debug, Serialize)]
pub(super) struct Report {
    schema_version: u32,
    counts: Counts,
    native_work: Breakdown,
    submissions: Vec<Submission>,
    limitations: Vec<&'static str>,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
struct Shape {
    command_index: Option<u64>,
    command_count: Option<u64>,
    token_count: Option<u64>,
    participant_count: Option<u64>,
    participant_start: Option<u64>,
    participant_end: Option<u64>,
    device_elapsed_ns: Option<u64>,
    device_interval_count: Option<u64>,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
struct Attributes {
    physical_submission_fingerprint: Option<String>,
    native_op_id: Option<String>,
    command_phase: Option<String>,
    wave_phase: Option<String>,
    participant_request_ids: Option<Vec<String>>,
    device_timing_status: Option<String>,
    device_timing_unavailable_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Interval {
    kind: String,
    start_offset_ns: u64,
    end_offset_ns: u64,
    subwork_id: Option<String>,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
struct Detail {
    device_intervals: Option<Vec<Interval>>,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
struct Event {
    phase: String,
    #[serde(default)]
    shape: Shape,
    #[serde(default)]
    attributes: Attributes,
    backend_detail: Option<Detail>,
}

struct Record {
    event: Event,
    canonical: Value,
    line: usize,
}

#[derive(Default)]
struct Pending {
    commands: BTreeMap<u64, Record>,
    physical: Option<Record>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum ParticipantEvidence {
    Complete,
    Missing,
    Invalid,
}

#[derive(Debug, Serialize)]
struct ComputeCommandShape {
    command_index: u64,
    native_op_id: Option<String>,
    wave_phase: Option<String>,
    token_count: Option<u64>,
    participant_count: Option<u64>,
    participant_start: Option<u64>,
    participant_end: Option<u64>,
    participant_request_id_count: Option<u64>,
    unique_participant_request_id_count: Option<u64>,
    participant_evidence: ParticipantEvidence,
    participant_evidence_issues: Vec<&'static str>,
}

impl ComputeCommandShape {
    fn from_event(event: &Event) -> Self {
        let shape = &event.shape;
        let ids = event.attributes.participant_request_ids.as_ref();
        let id_count = ids.map(|ids| ids.len() as u64);
        let unique_count = ids.map(|ids| ids.iter().collect::<BTreeSet<_>>().len() as u64);
        let mut issues = Vec::new();
        let mut invalid = false;
        if shape.participant_count.is_none() {
            issues.push("missing_participant_count");
        }
        let range_count = match (shape.participant_start, shape.participant_end) {
            (Some(start), Some(end)) => match end.checked_sub(start) {
                Some(count) => {
                    if shape
                        .participant_count
                        .is_some_and(|declared| declared != count)
                    {
                        issues.push("participant_range_count_mismatch");
                        invalid = true;
                    }
                    Some(count)
                }
                None => {
                    issues.push("reversed_participant_range");
                    invalid = true;
                    None
                }
            },
            _ => {
                issues.push("missing_participant_range");
                None
            }
        };
        if let Some(count) = id_count {
            if shape
                .participant_count
                .is_some_and(|declared| declared != count)
            {
                issues.push("participant_request_id_count_mismatch");
                invalid = true;
            }
            if range_count.is_some_and(|range| range != count) {
                issues.push("participant_range_request_id_count_mismatch");
                invalid = true;
            }
            if unique_count != id_count {
                issues.push("duplicate_participant_request_ids");
                invalid = true;
            }
            if ids.is_some_and(|ids| ids.iter().any(String::is_empty)) {
                issues.push("empty_participant_request_id");
                invalid = true;
            }
        } else {
            issues.push("missing_participant_request_ids");
        }
        Self {
            command_index: shape.command_index.expect("native identity was checked"),
            native_op_id: event.attributes.native_op_id.clone(),
            wave_phase: event.attributes.wave_phase.clone(),
            token_count: shape.token_count,
            participant_count: shape.participant_count,
            participant_start: shape.participant_start,
            participant_end: shape.participant_end,
            participant_request_id_count: id_count,
            unique_participant_request_id_count: unique_count,
            participant_evidence: if invalid {
                ParticipantEvidence::Invalid
            } else if issues.is_empty() {
                ParticipantEvidence::Complete
            } else {
                ParticipantEvidence::Missing
            },
            participant_evidence_issues: issues,
        }
    }
}

/// Ignore only per-observer envelopes. All physical shape, identity, provider,
/// status and timing fields remain in conflict detection, including unknown ones.
fn canonical(mut value: Value) -> Value {
    let object = value.as_object_mut().expect("event is an object");
    for field in [
        "event_id",
        "request_id",
        "correlation_id",
        "timestamp",
        "ts_unix_nanos",
    ] {
        object.remove(field);
    }
    if let Some(ids) = value.pointer_mut("/attributes/participant_request_ids") {
        if let Some(array) = ids.as_array_mut() {
            array.sort_by_key(Value::to_string);
        }
    }
    value
}

fn same_record(previous: &Record, next: &Record, fingerprint: &str) -> Result<()> {
    if previous.canonical != next.canonical {
        bail!(
            "conflicting physical evidence for {fingerprint} on lines {} and {}",
            previous.line,
            next.line
        );
    }
    Ok(())
}

pub(super) fn summarize(reader: impl BufRead) -> Result<Report> {
    let mut counts = Counts::default();
    let mut pending = BTreeMap::<String, Pending>::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        counts.input_records += 1;
        let value: Value = serde_json::from_str(&line)
            .with_context(|| format!("invalid JSON on line {}", index + 1))?;
        let phase = value.get("phase").and_then(Value::as_str);
        let physical = match phase {
            Some("vnext.device_native_work") => {
                counts.native_records += 1;
                false
            }
            Some("vnext.device_physical_submission") => {
                counts.physical_submission_records += 1;
                true
            }
            _ => {
                counts.ignored_records += 1;
                continue;
            }
        };
        let event: Event = serde_json::from_value(value.clone())
            .with_context(|| format!("invalid native schema on line {}", index + 1))?;
        let Some(fingerprint) = event
            .attributes
            .physical_submission_fingerprint
            .as_ref()
            .filter(|value| !value.is_empty())
            .cloned()
        else {
            counts.records_missing_identity += 1;
            continue;
        };
        if !physical && event.shape.command_index.is_none() {
            counts.records_missing_identity += 1;
            continue;
        }
        let command_index = event.shape.command_index;
        let record = Record {
            event,
            canonical: canonical(value),
            line: index + 1,
        };
        let submission = pending.entry(fingerprint.clone()).or_default();
        if physical {
            if let Some(previous) = &submission.physical {
                same_record(previous, &record, &fingerprint)?;
                counts.duplicate_submission_records += 1;
            } else {
                submission.physical = Some(record);
            }
        } else {
            match submission.commands.entry(command_index.unwrap()) {
                std::collections::btree_map::Entry::Occupied(previous) => {
                    same_record(previous.get(), &record, &fingerprint)?;
                    counts.duplicate_native_records += 1;
                }
                std::collections::btree_map::Entry::Vacant(slot) => {
                    slot.insert(record);
                }
            }
        }
    }
    let mut native_work = Breakdown::default();
    let mut submissions = Vec::new();
    for (fingerprint, pending) in pending {
        let mut work = Breakdown::default();
        let mut tokens = BTreeSet::new();
        let mut phases = BTreeSet::new();
        let mut compute_command_shapes = Vec::new();
        let mut participant_counts = BTreeSet::new();
        for record in pending.commands.values() {
            if let Some(count) = record.event.shape.token_count {
                tokens.insert(count);
            }
            if let Some(phase) = &record.event.attributes.wave_phase {
                phases.insert(phase.clone());
            }
            if record.event.attributes.command_phase.as_deref() == Some("compute") {
                let shape = ComputeCommandShape::from_event(&record.event);
                if let Some(count) = shape.participant_count {
                    participant_counts.insert(count);
                }
                compute_command_shapes.push(shape);
            }
            work.record(&record.event);
            native_work.record(&record.event);
        }
        if let Some(phase) = pending
            .physical
            .as_ref()
            .and_then(|record| record.event.attributes.wave_phase.as_ref())
        {
            phases.insert(phase.clone());
        }
        let declared = pending.physical.as_ref().and_then(|record| {
            (record.event.attributes.device_timing_status.as_deref() == Some("measured"))
                .then_some(record.event.shape.command_count)
                .flatten()
        });
        if declared.is_some_and(|count| pending.commands.keys().any(|index| *index >= count)) {
            bail!("native command index exceeds declared command count for {fingerprint}");
        }
        let observed = pending.commands.len() as u64;
        let gaps = pending
            .commands
            .last_key_value()
            .map(|(index, _)| u128::from(*index) + 1 - u128::from(observed))
            .unwrap_or(0);
        let missing_participants = compute_command_shapes
            .iter()
            .filter(|shape| shape.participant_evidence == ParticipantEvidence::Missing)
            .count() as u64;
        let invalid_participants = compute_command_shapes
            .iter()
            .filter(|shape| shape.participant_evidence == ParticipantEvidence::Invalid)
            .count() as u64;
        let uniform_participants = participant_counts.first().copied().filter(|_| {
            participant_counts.len() == 1 && missing_participants == 0 && invalid_participants == 0
        });
        submissions.push(Submission {
            fingerprint,
            maximum_command_token_count: tokens.last().copied(),
            observed_command_token_counts: tokens.into_iter().collect(),
            compute_command_shapes,
            observed_compute_participant_counts: participant_counts.into_iter().collect(),
            compute_commands_missing_participant_evidence: missing_participants,
            compute_commands_invalid_participant_evidence: invalid_participants,
            uniform_verified_compute_participant_count: uniform_participants,
            phase: phases
                .first()
                .filter(|phase| {
                    phases.len() == 1 && matches!(phase.as_str(), "prefill" | "decode" | "mixed")
                })
                .cloned(),
            explicit_wave_phases: phases.into_iter().collect(),
            physical_submission_record_present: pending.physical.is_some(),
            physical_submission_timing_status: pending
                .physical
                .as_ref()
                .and_then(|record| record.event.attributes.device_timing_status.clone()),
            physical_submission_unavailable_reason: pending.physical.as_ref().and_then(|record| {
                record
                    .event
                    .attributes
                    .device_timing_unavailable_reason
                    .clone()
            }),
            declared_command_count: declared,
            missing_declared_commands: declared.map(|count| count - observed),
            command_index_gaps_below_maximum: u64::try_from(gaps)
                .context("command index range overflow")?,
            native_work: work,
        });
    }
    Ok(Report {
        schema_version: 1,
        counts,
        native_work,
        submissions,
        limitations: vec![
            "Native command times are counted once per physical submission and command index, never once per participant.",
            "Subwork and unlabeled intervals decompose command time; do not add them to operation totals. Encoder gaps are excluded from the sum of intervals.",
            "Token counts are observed command shapes, not request counts. A maximum command token count is not proof of a prefill or decode phase; single-token prefill, output heads and mixed waves can coexist.",
            "Compute participant evidence validates declared counts against ranges and distinct request IDs. Uniform verified counts cover observed compute commands only, never missing commands or whole submissions; they do not infer a phase or token count.",
            "Phase remains unclassified without explicit wave_phase evidence. Conflicting explicit phases are preserved and leave the phase unclassified.",
            "Unavailable, invalid and missing measurements are excluded from measured time, not treated as zero-duration commands. Missing interval details leave subwork coverage incomplete.",
            "Missing declared commands is unknown without a measured physical-submission command count. Index gaps cannot detect absent final commands or absent whole submissions.",
            "Physical span/submission timing is enclosing evidence only and is never added to native command totals. Reusable-span-only native records do not provide per-operation elapsed time.",
            "Kernel instrumentation changes execution boundaries. These diagnostic sums do not establish wall latency, device utilization, output quality or performance improvement.",
        ],
    })
}

impl Breakdown {
    fn record(&mut self, event: &Event) {
        let operation = event
            .attributes
            .native_op_id
            .as_deref()
            .unwrap_or("<missing>");
        let measured = measure(event);
        self.timing.record(event, &measured);
        self.operations
            .entry(operation.to_owned())
            .or_default()
            .record(event, &measured);
        *self
            .command_phases
            .entry(
                event
                    .attributes
                    .command_phase
                    .clone()
                    .unwrap_or_else(|| "<missing>".into()),
            )
            .or_default() += 1;
        if measured.valid {
            for interval in measured.intervals {
                if let Some(id) = &interval.subwork_id {
                    let entry = self.subwork.entry(id.clone()).or_default();
                    entry.interval_count += 1;
                    entry.measured_ns +=
                        u128::from(interval.end_offset_ns - interval.start_offset_ns);
                }
            }
        }
    }
}

struct Measurement<'a> {
    valid: bool,
    invalid: bool,
    intervals: &'a [Interval],
}

fn measure(event: &Event) -> Measurement<'_> {
    let intervals = event
        .backend_detail
        .as_ref()
        .and_then(|detail| detail.device_intervals.as_deref());
    let mut result = Measurement {
        valid: false,
        invalid: false,
        intervals: intervals.unwrap_or_default(),
    };
    if event.attributes.device_timing_status.as_deref() != Some("measured") {
        result.invalid = event.shape.device_elapsed_ns.is_some()
            || intervals.is_some_and(|items| !items.is_empty());
        return result;
    }
    let Some(elapsed) = event.shape.device_elapsed_ns else {
        return result;
    };
    result.invalid = elapsed == 0;
    if let Some(intervals) = intervals {
        let mut end = 0;
        let mut total = 0_u128;
        for interval in intervals {
            if interval.end_offset_ns <= interval.start_offset_ns || interval.start_offset_ns < end
            {
                result.invalid = true;
            } else {
                total += u128::from(interval.end_offset_ns - interval.start_offset_ns);
            }
            end = interval.end_offset_ns;
        }
        result.invalid |= intervals.is_empty()
            || total != u128::from(elapsed)
            || event
                .shape
                .device_interval_count
                .is_some_and(|count| count != intervals.len() as u64);
    }
    result.valid = !result.invalid;
    result
}

impl Timing {
    fn record(&mut self, event: &Event, measured: &Measurement<'_>) {
        self.commands += 1;
        let status = event
            .attributes
            .device_timing_status
            .as_deref()
            .unwrap_or("<missing>");
        *self.timing_statuses.entry(status.to_owned()).or_default() += 1;
        if let Some(reason) = &event.attributes.device_timing_unavailable_reason {
            *self.unavailable_reasons.entry(reason.clone()).or_default() += 1;
        }
        self.missing_shape_commands += u64::from(event.shape.token_count.is_none());
        self.invalid_timing_commands += u64::from(measured.invalid);
        if status == "measured" {
            self.missing_elapsed_commands += u64::from(event.shape.device_elapsed_ns.is_none());
            self.missing_intervals_commands += u64::from(
                event
                    .backend_detail
                    .as_ref()
                    .and_then(|detail| detail.device_intervals.as_ref())
                    .is_none(),
            );
        }
        if measured.valid {
            self.measured_commands += 1;
            self.measured_ns += u128::from(event.shape.device_elapsed_ns.unwrap());
            for interval in measured
                .intervals
                .iter()
                .filter(|interval| interval.subwork_id.is_none())
            {
                self.unlabeled_interval_count += 1;
                self.unlabeled_interval_ns +=
                    u128::from(interval.end_offset_ns - interval.start_offset_ns);
            }
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
