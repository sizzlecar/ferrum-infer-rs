//! Bounded analysis of schema-1/2 image step JSONL artifacts. Schema 1 retains
//! structural evidence only; schema 2 adds same-session host-clock intervals.
//! This observes execution records; it grants no image-quality, G3, latency,
//! throughput or weight-authentication qualification. Limits bound accepted
//! input/state extents, rather than claiming a process allocator/RSS ceiling.
use serde::{Deserialize, Serialize};
use serde_json::{value::RawValue, Value};
use std::{
    fmt,
    io::{self, BufRead, Cursor},
};
mod expansion;
mod join;
mod producer_extensions;
mod state;
mod timing;
pub use expansion::*;
pub use join::*;
pub use producer_extensions::{
    ImageTraceExpansionTokenWaveSummary, ImageTraceReferenceLatentDigest, ImageTraceSharedOwner,
};
use state::Analyzer;
pub use timing::*;

#[derive(Clone, Copy, Debug)]
pub struct ImageContinuousTraceReadLimits {
    pub maximum_bytes: usize,
    pub maximum_record_bytes: usize,
    pub maximum_records: u64,
    pub maximum_requests: usize,
    pub maximum_members: usize,
    pub maximum_row_runs_per_member: usize,
}
impl Default for ImageContinuousTraceReadLimits {
    fn default() -> Self {
        Self {
            maximum_bytes: 64 << 20,
            maximum_record_bytes: 4 << 20,
            maximum_records: 1_000_000,
            maximum_requests: 100_000,
            maximum_members: 256,
            maximum_row_runs_per_member: 100_000,
        }
    }
}
impl ImageContinuousTraceReadLimits {
    fn validate(self) -> Result<(), ImageContinuousTraceError> {
        if self.maximum_bytes == 0
            || self.maximum_record_bytes == 0
            || self.maximum_record_bytes > self.maximum_bytes
            || self.maximum_bytes > isize::MAX as usize
            || self.maximum_records == 0
            || self.maximum_requests == 0
            || self.maximum_members == 0
            || self.maximum_row_runs_per_member == 0
        {
            return Err(error(
                0,
                ImageContinuousTraceErrorKind::Capacity,
                "invalid trace reader limits",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageContinuousTraceErrorKind {
    Io,
    Capacity,
    Invalid,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ImageContinuousTraceError {
    pub line: u64,
    pub kind: ImageContinuousTraceErrorKind,
    pub reason: String,
}
impl fmt::Display for ImageContinuousTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "image trace line {}: {}", self.line, self.reason)
    }
}
impl std::error::Error for ImageContinuousTraceError {}
fn error(
    line: u64,
    kind: ImageContinuousTraceErrorKind,
    reason: impl Into<String>,
) -> ImageContinuousTraceError {
    ImageContinuousTraceError {
        line,
        kind,
        reason: reason.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageStepEvidenceStatus {
    Observed,
    NotEvaluated,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTraceIncompleteReason {
    MissingHeader,
    MissingFooter,
    UnterminatedRecord,
    TruncatedRecord,
    Unsealed,
    ProducerTruncated,
    ProducerDroppedRecords,
    RequestsWithoutExit,
}

/// Exact header event digest plus its actual parsed identity/configuration.
/// Assets may explicitly require an external weight lock; these hashes do
/// not authenticate mapped weights or supersede that original requirement.
#[derive(Clone, Debug, Serialize)]
pub struct ImageTraceIdentity {
    pub header_event_sha256: String,
    pub device: Value,
    pub device_opaque_allowance_bytes: u64,
    pub assets: Value,
    pub prompt_expansion: Option<Value>,
    pub noise_profile: Value,
    /// Absent in legacy traces; absence does not identify a noise policy.
    pub initial_noise_policy: Option<String>,
    pub edit_noise_policy: Option<String>,
    pub vae_decoder_tile_batch: usize,
    pub pipeline: Value,
    pub host: Value,
    pub context: Value,
    pub serve_context_present: bool,
    pub metadata_digests_authenticate_weights: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ImageContinuousTraceFacts {
    pub initialized_requests: u64,
    pub published_cohorts: u64,
    pub committed_waves: u64,
    pub multi_request_committed_waves: u64,
    pub distinct_step_index_waves: u64,
    pub distinct_scheduler_timestep_waves: u64,
    /// The basis can differ because of audio/reference times. This count
    /// alone is not evidence of asynchronous request progress.
    pub nonuniform_time_basis_waves: u64,
    pub refill_continuation_waves: u64,
    pub completed_exits: u64,
    pub cancelled_exits: u64,
    pub failed_exits: u64,
    pub requests_without_exit: usize,
    pub first_joint_commit: Option<ImageTraceCommitWitness>,
    pub first_mixed_progress_commit: Option<ImageTraceCommitWitness>,
    pub first_refill_commit: Option<ImageTraceCommitWitness>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ImageTraceCommitWitness {
    pub sequence: u64,
    pub cohort_job: u64,
    pub time_basis: Vec<f32>,
    pub members: Vec<ImageTraceMember>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ImageContinuousTraceAnalysis {
    pub status: ImageStepEvidenceStatus,
    pub schema_version: u32,
    pub session_uuid: Option<String>,
    pub identity: Option<ImageTraceIdentity>,
    pub records: u64,
    pub event_bytes: u64,
    pub input_bytes: u64,
    pub footer: Option<ImageTraceFooter>,
    pub facts: ImageContinuousTraceFacts,
    pub incomplete_reasons: Vec<ImageTraceIncompleteReason>,
    pub timing: ImageTraceTimingAnalysis,
    /// Opt-in expansion metadata, independent of queue/step timing evidence.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub prompt_expansion_observations: Vec<ImageTracePromptExpansionObservation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_expansion_token_waves: Option<ImageTraceExpansionTokenWaveSummary>,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub reference_latents: std::collections::BTreeMap<String, ImageTraceReferenceLatentDigest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTraceFooter {
    pub sealed: bool,
    pub truncated: bool,
    pub recorded: u64,
    pub dropped: u64,
    pub used_bytes: u64,
    pub maximum_bytes: u64,
    pub backing_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTraceRowRun {
    pub length: u64,
    pub time_index: u64,
    pub modality: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTraceMember {
    pub request: String,
    pub packed_row_start: u64,
    pub packed_row_end: u64,
    pub step_index: u64,
    pub sigma: f32,
    pub next_sigma: f32,
    pub scheduler_timestep: f32,
    pub row_runs: Vec<ImageTraceRowRun>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    device: Value,
    device_opaque_allowance_bytes: u64,
    assets: Value,
    prompt_expansion: Value,
    noise_profile: Value,
    #[serde(default, deserialize_with = "declared_noise_policy")]
    initial_noise_policy: Option<String>,
    #[serde(default, deserialize_with = "declared_noise_policy")]
    edit_noise_policy: Option<String>,
    #[serde(default = "default_decoder_tile_batch")]
    vae_decoder_tile_batch: usize,
    pipeline: Value,
    host: Value,
    context: Value,
}
fn declared_noise_policy<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    // Missing fields default to None; a supplied value must be a string.
    String::deserialize(deserializer).map(Some)
}
fn default_decoder_tile_batch() -> usize {
    1
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Event {
    Header(Header),
    PromptExpansionObserved(ImageTracePromptExpansionObservation),
    PromptExpansionTokenWave(producer_extensions::TokenWave),
    RequestInitialized {
        request: String,
        job: u64,
        seed: u64,
        width: u32,
        height: u32,
        editing: bool,
        reference_latent: Option<ImageTraceReferenceLatentDigest>,
    },
    CohortPublished {
        cohort_job: u64,
        ordered_members: Vec<String>,
        plan_hash: String,
        program_fingerprint: String,
    },
    PackedStepCommitted {
        #[serde(default)]
        start_sequence: Option<u64>,
        cohort_job: u64,
        plan_hash: String,
        program_fingerprint: String,
        packed_request_count: usize,
        time_basis: Vec<f32>,
        members: Vec<ImageTraceMember>,
    },
    RequestExitedAtFence {
        request: String,
        completed_steps: u64,
        reason: ExitReason,
    },
    RequestSubmissionRejected {
        correlation: ferrum_types::ImageRequestCorrelation,
        outcome: String,
    },
    RequestQueued {
        correlation: ferrum_types::ImageRequestCorrelation,
    },
    RequestDequeued {
        request_id: String,
    },
    RequestRequeued {
        request_id: String,
    },
    InitializationStarted {
        request_id: String,
    },
    RequestLinked {
        request_id: String,
        request: String,
    },
    RequestServiceFinished {
        request_id: String,
        outcome: String,
    },
    PhaseStarted {
        request_id: Option<String>,
        request: Option<String>,
        phase: String,
    },
    PhaseFinished {
        start_sequence: u64,
        outcome: String,
        #[serde(default)]
        main_vision_binding: Option<ImageTraceMainVisionBinding>,
    },
    PackedStepStarted {
        cohort_job: u64,
        plan_hash: String,
        program_fingerprint: String,
        members: Vec<ImageTraceStepStartMember>,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTraceStepStartMember {
    pub request: String,
    pub step_index: u64,
}
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExitReason {
    Completed,
    Cancelled,
    Failed,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema_version: u32,
    session_uuid: String,
    sequence: u64,
    #[serde(default)]
    monotonic_ns: Option<u64>,
    #[serde(default)]
    event_ns: Option<u64>,
    event: Box<RawValue>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FooterEnvelope {
    schema_version: u32,
    session_uuid: String,
    footer: ImageTraceFooter,
}
#[derive(Deserialize)]
struct Probe<'a> {
    #[serde(default, borrow)]
    footer: Option<&'a RawValue>,
    #[serde(default)]
    sequence: Option<u64>,
}

pub fn analyze_image_continuous_trace_bytes(
    bytes: &[u8],
    limits: &ImageContinuousTraceReadLimits,
) -> Result<ImageContinuousTraceAnalysis, ImageContinuousTraceError> {
    analyze_image_continuous_trace(Cursor::new(bytes), limits)
}

/// Consume one JSONL export. Complete but semantically inconsistent records
/// are errors. A structurally valid prefix remains NotEvaluated when the
/// export is unfinished; observations in it cannot establish complete history.
pub fn analyze_image_continuous_trace(
    mut reader: impl BufRead,
    limits: &ImageContinuousTraceReadLimits,
) -> Result<ImageContinuousTraceAnalysis, ImageContinuousTraceError> {
    limits.validate()?;
    let mut analyzer = Analyzer::new(*limits);
    let mut timing = timing::TimingAnalyzer::new(*limits);
    let mut buffer = Vec::new();
    let mut line = 0u64;
    let mut input_bytes = 0usize;
    let mut event_bytes = 0u64;
    let mut records = 0u64;
    let mut session = None;
    let mut schema = None;
    let mut last_monotonic_ns = None;
    let mut footer = None;
    let mut incomplete = Vec::new();
    loop {
        let terminated = read_line(&mut reader, &mut buffer, &mut input_bytes, limits, line + 1)?;
        if buffer.is_empty() {
            break;
        }
        line += 1;
        if footer.is_some() {
            return Err(error(
                line,
                ImageContinuousTraceErrorKind::Invalid,
                "records follow the export footer",
            ));
        }
        let probe: Probe<'_> = match serde_json::from_slice(&buffer) {
            Ok(probe) => probe,
            Err(e) if !terminated && truncated_json_prefix(&buffer, &e) => {
                incomplete.push(ImageTraceIncompleteReason::TruncatedRecord);
                break;
            }
            Err(_) => {
                return Err(error(
                    line,
                    ImageContinuousTraceErrorKind::Invalid,
                    "invalid JSONL record",
                ))
            }
        };
        if !terminated {
            incomplete.push(ImageTraceIncompleteReason::UnterminatedRecord);
        }
        if probe.footer.is_some() {
            if probe.sequence.is_some() {
                return Err(error(
                    line,
                    ImageContinuousTraceErrorKind::Invalid,
                    "record mixes footer and event fields",
                ));
            }
            let record: FooterEnvelope = parse(&buffer, line)?;
            identity(
                record.schema_version,
                &record.session_uuid,
                &mut session,
                &mut schema,
                line,
            )?;
            if record.footer.recorded != records
                || record.footer.used_bytes != event_bytes
                || record.footer.maximum_bytes == 0
                || record.footer.used_bytes > record.footer.maximum_bytes
                || record.footer.backing_bytes < record.footer.maximum_bytes
            {
                return Err(error(
                    line,
                    ImageContinuousTraceErrorKind::Invalid,
                    "footer counters/capacities differ from the actual prefix",
                ));
            }
            footer = Some(record.footer);
        } else {
            if records >= limits.maximum_records {
                return Err(error(
                    line,
                    ImageContinuousTraceErrorKind::Capacity,
                    "event count exceeds reader capacity",
                ));
            }
            let record: Envelope = parse(&buffer, line)?;
            identity(
                record.schema_version,
                &record.session_uuid,
                &mut session,
                &mut schema,
                line,
            )?;
            records += 1;
            if record.sequence != records {
                return Err(error(
                    line,
                    ImageContinuousTraceErrorKind::Invalid,
                    "event sequence is missing, repeated or out of order",
                ));
            }
            let event: Event = parse(record.event.get().as_bytes(), line)?;
            let event_ns = match record.schema_version {
                1 if record.monotonic_ns.is_none() && record.event_ns.is_none() => None,
                2 => {
                    let (Some(observed), Some(boundary)) = (record.monotonic_ns, record.event_ns)
                    else {
                        return Err(error(
                            line,
                            ImageContinuousTraceErrorKind::Invalid,
                            "schema-2 event lacks its clock endpoints",
                        ));
                    };
                    if boundary > observed
                        || last_monotonic_ns.is_some_and(|previous| observed < previous)
                    {
                        return Err(error(
                            line,
                            ImageContinuousTraceErrorKind::Invalid,
                            "trace observation clock regresses or boundary is in its future",
                        ));
                    }
                    last_monotonic_ns = Some(observed);
                    Some(boundary)
                }
                _ => {
                    return Err(error(
                        line,
                        ImageContinuousTraceErrorKind::Invalid,
                        "schema-1 event contains schema-2 timing fields",
                    ))
                }
            };
            timing.event(&event, record.sequence, event_ns, line)?;
            analyzer.event(event, record.event.get().as_bytes(), record.sequence, line)?;
            event_bytes = event_bytes
                .checked_add(buffer.len() as u64)
                .ok_or_else(|| {
                    error(
                        line,
                        ImageContinuousTraceErrorKind::Capacity,
                        "event byte count overflow",
                    )
                })?;
        }
    }
    let mut analysis = analyzer.finish(
        session,
        records,
        event_bytes,
        input_bytes as u64,
        footer,
        incomplete,
        line,
    )?;
    analysis.schema_version = schema.unwrap_or(1);
    analysis.timing = timing.finish(
        analysis.schema_version,
        analysis.incomplete_reasons.is_empty(),
    );
    Ok(analysis)
}
fn truncated_json_prefix(bytes: &[u8], probe_error: &serde_json::Error) -> bool {
    if probe_error.is_eof() {
        return true;
    }
    // Serde's IgnoredAny/RawValue scanner reports Syntax/InvalidNumber for
    // EOF immediately after a decimal point or exponent (e.g. {"x":0.).
    // Its ordinary number parser distinguishes these genuine prefixes from
    // malformed syntax. Retry only this unfinished final record, still under
    // the same byte bound; a syntax/data error does not become incomplete.
    probe_error.is_syntax()
        && serde_json::from_slice::<Value>(bytes).is_err_and(|error| error.is_eof())
}
fn parse<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    line: u64,
) -> Result<T, ImageContinuousTraceError> {
    serde_json::from_slice(bytes).map_err(|_| {
        error(
            line,
            ImageContinuousTraceErrorKind::Invalid,
            "record differs from the declared trace schema",
        )
    })
}
fn identity(
    version: u32,
    uuid: &str,
    session: &mut Option<String>,
    schema: &mut Option<u32>,
    line: u64,
) -> Result<(), ImageContinuousTraceError> {
    let valid = uuid.len() == 36
        && uuid.bytes().enumerate().all(|(i, c)| {
            if [8, 13, 18, 23].contains(&i) {
                c == b'-'
            } else {
                c.is_ascii_hexdigit()
            }
        });
    if ![1, 2].contains(&version)
        || !valid
        || session.as_ref().is_some_and(|old| old != uuid)
        || schema.is_some_and(|old| old != version)
    {
        return Err(error(
            line,
            ImageContinuousTraceErrorKind::Invalid,
            "unsupported schema or inconsistent session UUID",
        ));
    }
    if session.is_none() {
        *session = Some(uuid.to_owned());
    }
    *schema = Some(version);
    Ok(())
}
fn read_line(
    reader: &mut impl BufRead,
    out: &mut Vec<u8>,
    total: &mut usize,
    limits: &ImageContinuousTraceReadLimits,
    line: u64,
) -> Result<bool, ImageContinuousTraceError> {
    out.clear();
    loop {
        let bytes = reader.fill_buf().map_err(|_: io::Error| {
            error(
                line,
                ImageContinuousTraceErrorKind::Io,
                "trace input read failed",
            )
        })?;
        if bytes.is_empty() {
            return Ok(false);
        }
        let take = bytes
            .iter()
            .position(|&b| b == b'\n')
            .map_or(bytes.len(), |p| p + 1);
        let next = out
            .len()
            .checked_add(take)
            .filter(|&n| n <= limits.maximum_record_bytes);
        let next_total = total
            .checked_add(take)
            .filter(|&n| n <= limits.maximum_bytes);
        if next.is_none() || next_total.is_none() {
            return Err(error(
                line,
                ImageContinuousTraceErrorKind::Capacity,
                "trace input exceeds record or total byte capacity",
            ));
        }
        out.try_reserve(take).map_err(|_| {
            error(
                line,
                ImageContinuousTraceErrorKind::Capacity,
                "trace record allocation failed",
            )
        })?;
        out.extend_from_slice(&bytes[..take]);
        *total = next_total.unwrap();
        let terminated = bytes[take - 1] == b'\n';
        reader.consume(take);
        if terminated {
            return Ok(true);
        }
    }
}

#[cfg(test)]
mod tests;
