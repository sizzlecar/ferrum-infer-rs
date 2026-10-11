//! Read-only analysis of the coarse host-cycle capture embedded in a profile.
//! Host graph launch is not GPU execution; terminal readback includes its fence,
//! copies and validation. These intervals do not establish removable CPU time.

use crate::percentile;
use ferrum_types::FerrumProfileEvent;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodeCycleCapture {
    pub clock: CycleClock,
    pub capacity: usize,
    pub active_producers_at_close: u64,
    pub stats: CycleStats,
    pub events: Vec<CycleEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleClock {
    pub id: u64,
    pub wall_anchor_unix_nanos: i64,
    pub wall_anchor_max_error_nanos: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleStats {
    pub attempted: u64,
    pub accepted: u64,
    pub busy: u64,
    pub full: u64,
    pub closed: u64,
    pub poisoned: u64,
    pub invalid_metadata: u64,
    pub outstanding_producers: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CycleTimestamp {
    pub clock_id: u64,
    pub nanos: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CycleCallContext {
    pub clock_id: u64,
    pub iteration: u64,
    pub call: u32,
    pub caller_ordinal: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CycleAuthority {
    pub sparse_id: u32,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CycleParticipantOwner {
    pub sequence_authority: CycleAuthority,
    pub request_authority: CycleAuthority,
    pub frame_id: u64,
    pub request_id: String,
    pub request_identity: String,
    pub canonical_ordinal: u32,
    pub caller_ordinal: u32,
    pub context: Option<CycleCallContext>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleWaveWitness {
    pub clock_id: u64,
    pub kind: String,
    pub batch_step_id: u64,
    pub batch_invocation_id: u64,
    pub lane_id: u64,
    pub device_id: String,
    pub runtime_implementation_fingerprint: String,
    pub prepared_wave_fingerprint: String,
    pub owners: Vec<CycleParticipantOwner>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum CycleScope {
    Iteration {
        iteration: u64,
    },
    Call {
        context: CycleCallContext,
    },
    Wave {
        clock_id: u64,
        batch_step_id: u64,
        batch_invocation_id: u64,
        prepared_wave_fingerprint: String,
    },
    Submission {
        clock_id: u64,
        batch_step_id: u64,
        batch_invocation_id: u64,
        prepared_wave_fingerprint: String,
        slot_id: u64,
        submission_fingerprint: String,
    },
    Participant {
        clock_id: u64,
        slot_id: u64,
        submission_fingerprint: String,
        owner: CycleParticipantOwner,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleEvent {
    pub at: CycleTimestamp,
    pub stage: String,
    pub scope: CycleScope,
    #[serde(default)]
    pub outcome: Option<String>,
    #[serde(default)]
    pub wave_witness: Option<CycleWaveWitness>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct CycleWaveKey {
    pub clock_id: u64,
    pub batch_step_id: u64,
    pub batch_invocation_id: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct IntervalMetric {
    pub count: usize,
    pub median_ns: Option<f64>,
    pub p99_ns: Option<f64>,
    /// Union of observed intervals on this one clock, not a sum of prefixes.
    pub union_ns: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeCycleCohort {
    pub wave: CycleWaveKey,
    pub width: usize,
    pub iteration: Option<u64>,
    pub call: Option<u32>,
    pub next_wave: Option<CycleWaveKey>,
    pub next_wave_kind: Option<String>,
    pub next_width: Option<usize>,
    pub classification: String,
    pub boundaries: BTreeMap<String, Option<u64>>,
    pub missing_edges: Vec<String>,
    pub owner_coverage: CycleOwnerCoverage,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CycleOwnerCoverage {
    pub expected: usize,
    pub complete: usize,
    pub required_missing: BTreeMap<String, usize>,
    /// Owners with an observed optional branch. Zero means absent, not a
    /// zero-duration observation; measured intervals are in phase_metrics.
    pub optional_branches: BTreeMap<String, usize>,
    pub invalid_order: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DecodeCycleAnalysis {
    pub clock_id: u64,
    pub events: usize,
    pub waves: usize,
    pub physical_width_filter: Option<usize>,
    pub collector_complete: bool,
    pub identity_valid: bool,
    /// Collector and identity qualification only. Inspect each cohort's edges.
    pub usable: bool,
    pub issues: BTreeMap<String, usize>,
    pub stage_counts: BTreeMap<String, usize>,
    pub outcome_counts: BTreeMap<String, usize>,
    pub phase_metrics: BTreeMap<String, IntervalMetric>,
    pub unmatched_boundaries: BTreeMap<String, usize>,
    pub observed_interval_union_ns: u64,
    pub chain_metrics: BTreeMap<String, IntervalMetric>,
    pub cohorts: Vec<DecodeCycleCohort>,
    pub semantics: Vec<&'static str>,
}

type Interval = (u64, u64);
// A completion slot is local to its reaper; the receipt fingerprint disambiguates.
type SubmissionKey = (u64, u64, String);
type CallKey = (u64, u64, u32);

#[derive(Debug)]
struct Wave<'a> {
    witness: &'a CycleWaveWitness,
    stages: BTreeMap<&'a str, Vec<u64>>,
    submission: Option<(SubmissionKey, &'a str)>,
}

mod analysis;
mod cohorts;
mod coverage;
pub use analysis::analyze_capture;

fn issue(issues: &mut BTreeMap<String, usize>, name: &str) {
    *issues.entry(name.to_owned()).or_default() += 1;
}

impl CycleScope {
    fn clock_id(&self) -> Option<u64> {
        match self {
            Self::Iteration { .. } => None,
            Self::Call { context } => Some(context.clock_id),
            Self::Wave { clock_id, .. }
            | Self::Submission { clock_id, .. }
            | Self::Participant { clock_id, .. } => Some(*clock_id),
        }
    }

    fn wave_key(&self) -> Option<CycleWaveKey> {
        match self {
            Self::Wave {
                clock_id,
                batch_step_id,
                batch_invocation_id,
                ..
            }
            | Self::Submission {
                clock_id,
                batch_step_id,
                batch_invocation_id,
                ..
            } => Some(CycleWaveKey {
                clock_id: *clock_id,
                batch_step_id: *batch_step_id,
                batch_invocation_id: *batch_invocation_id,
            }),
            _ => None,
        }
    }

    fn prepared_fingerprint(&self) -> Option<&str> {
        match self {
            Self::Wave {
                prepared_wave_fingerprint,
                ..
            }
            | Self::Submission {
                prepared_wave_fingerprint,
                ..
            } => Some(prepared_wave_fingerprint),
            _ => None,
        }
    }
}

fn witness_key(witness: &CycleWaveWitness) -> CycleWaveKey {
    CycleWaveKey {
        clock_id: witness.clock_id,
        batch_step_id: witness.batch_step_id,
        batch_invocation_id: witness.batch_invocation_id,
    }
}

/// Parse the existing FerrumProfileEvent envelope. Empty captures are not a
/// successful observation; malformed JSON, including a truncated export, errors.
pub fn analyze_profile_jsonl(
    input: &str,
    physical_width: Option<usize>,
) -> Result<Vec<DecodeCycleAnalysis>, String> {
    let mut result = Vec::new();
    for (index, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event: FerrumProfileEvent =
            serde_json::from_str(line).map_err(|error| format!("line {}: {error}", index + 1))?;
        event
            .validate()
            .map_err(|error| format!("line {}: {error}", index + 1))?;
        if event.phase != "engine.decode_cycle_capture" {
            continue;
        }
        let capture = event
            .attributes
            .get("capture")
            .ok_or_else(|| format!("line {}: capture is missing", index + 1))?;
        let capture: DecodeCycleCapture = serde_json::from_value(capture.clone())
            .map_err(|error| format!("line {}: invalid capture: {error}", index + 1))?;
        let mut analysis = analyze_capture(&capture, physical_width);
        if event
            .attributes
            .get("collector_complete")
            .and_then(serde_json::Value::as_bool)
            != Some(analysis.collector_complete)
        {
            issue(&mut analysis.issues, "declared_collector_complete_mismatch");
            analysis.usable = false;
        }
        if event
            .attributes
            .get("clock_source")
            .and_then(serde_json::Value::as_str)
            != Some("rust_std_instant")
        {
            issue(&mut analysis.issues, "profile_clock_source_mismatch");
            analysis.usable = false;
        }
        if event
            .shape
            .get("collected_events")
            .and_then(serde_json::Value::as_u64)
            != Some(capture.events.len() as u64)
            || event
                .shape
                .get("event_capacity")
                .and_then(serde_json::Value::as_u64)
                != Some(capture.capacity as u64)
        {
            issue(&mut analysis.issues, "profile_shape_counter_mismatch");
            analysis.usable = false;
        }
        if !analysis.usable {
            analysis.disqualify_chains();
        }
        result.push(analysis);
    }
    if result.is_empty() {
        return Err("profile contains no decode-cycle capture".to_owned());
    }
    Ok(result)
}

impl DecodeCycleAnalysis {
    fn disqualify_chains(&mut self) {
        for cohort in &mut self.cohorts {
            if cohort.classification == "complete_decode_to_decode" {
                cohort.classification = "unqualified_capture".to_owned();
            }
        }
        for metric in self.chain_metrics.values_mut() {
            *metric = IntervalMetric::default();
        }
    }
}

#[cfg(test)]
mod tests;
