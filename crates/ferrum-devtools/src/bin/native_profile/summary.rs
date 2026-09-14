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
    device_elapsed_ns: Option<u64>,
    device_interval_count: Option<u64>,
}

#[derive(Debug, Default, Clone, Deserialize, Serialize)]
struct Attributes {
    physical_submission_fingerprint: Option<String>,
    native_op_id: Option<String>,
    command_phase: Option<String>,
    wave_phase: Option<String>,
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
            array.dedup();
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
        for record in pending.commands.values() {
            if let Some(count) = record.event.shape.token_count {
                tokens.insert(count);
            }
            if let Some(phase) = &record.event.attributes.wave_phase {
                phases.insert(phase.clone());
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
        submissions.push(Submission {
            fingerprint,
            maximum_command_token_count: tokens.last().copied(),
            observed_command_token_counts: tokens.into_iter().collect(),
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
