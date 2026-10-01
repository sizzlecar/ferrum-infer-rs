//! Independent original reference measurements. No cost trainer, model import,
//! or declared wall-clock accuracy is required for a fixed scoring unit.
use super::*;
use profile::v2::ProfileSampleV2;
use std::io::{Read, Write};

const ARTIFACT: &str = "ferrum.reference-source";
const CLOCK: &str = "original_monotonic_projected_from_checkpoint_wall";

/// A source sealed after a real FIFO checkpoint. Fields are private so callers
/// cannot supply a different cut, clock, or hash to the reference assembler.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationReferenceSourceV1 {
    path: PathBuf,
    #[serde(flatten)]
    seal: SourceSeal,
}

#[derive(Debug, Clone, Serialize)]
struct SourceSeal {
    sha256: [u8; 32],
    bytes: u64,
    accepted_ordinal: u64,
    checkpoint_monotonic_ns: u64,
    generated_unix_ns: NonZeroU64,
    samples: usize,
}
impl CalibrationReferenceSourceV1 {
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn sha256(&self) -> [u8; 32] {
        self.seal.sha256
    }
    pub fn bytes(&self) -> u64 {
        self.seal.bytes
    }
    pub fn accepted_ordinal(&self) -> u64 {
        self.seal.accepted_ordinal
    }
}

/// The same sealed source bytes as the file adapter, retained only in memory.
/// No path or wall-clock accuracy claim is manufactured for this storage.
pub(in crate::continuous_engine::inner::calibration) struct MemoryReferenceSourceV1 {
    seal: SourceSeal,
    bytes: Vec<u8>,
}
impl MemoryReferenceSourceV1 {
    pub(in crate::continuous_engine::inner::calibration) fn into_bytes(self) -> Arc<[u8]> {
        self.bytes.into()
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    artifact_type: String,
    schema_version: u32,
    fingerprint: profile::ProfileFingerprint,
    plan_sha256: [u8; 32],
    frozen_accepted_ordinal: u64,
    frozen_at_monotonic_ns: u64,
    accepted_ordinal: u64,
    checkpoint_monotonic_ns: u64,
    generated_unix_ns: NonZeroU64,
    clock_projection: String,
    samples: usize,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Commit {
    request_id: RequestId,
    owner_incarnation: u64,
    work_generation: u64,
    input_index: u32,
    committed_at_ns: u64,
    work: Work,
}
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Work {
    Prefill {
        start: u32,
        end: u32,
        total_prompt_tokens: u32,
        generated_before: u64,
        generated_after: u64,
    },
    Decode {
        kv_before: u32,
        kv_after: u32,
        generated_before: u64,
        generated_after: u64,
    },
}
impl From<&CalibrationCommittedRow> for Commit {
    fn from(row: &CalibrationCommittedRow) -> Self {
        let work = match row.work {
            CalibrationCommittedWork::Prefill {
                start,
                end,
                total_prompt_tokens,
                generated_before,
                generated_after,
            } => Work::Prefill {
                start,
                end,
                total_prompt_tokens,
                generated_before,
                generated_after,
            },
            CalibrationCommittedWork::Decode {
                kv_before,
                kv_after,
                generated_before,
                generated_after,
            } => Work::Decode {
                kv_before,
                kv_after,
                generated_before,
                generated_after,
            },
        };
        Self {
            request_id: row.request_id.clone(),
            owner_incarnation: row.owner_incarnation,
            work_generation: row.work_generation,
            input_index: row.input_index,
            committed_at_ns: row.committed_at_ns,
            work,
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    accepted_ordinal: u64,
    observed_at_monotonic_ns: u64,
    sample: ProfileSampleV2,
    host: HostCostFeaturesV1,
    commit: Commit,
}
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Record {
    Header {
        header: Header,
    },
    Observation {
        observation: Observation,
    },
    Summary {
        accepted_ordinal: u64,
        samples: usize,
    },
}

impl CalibrationSession {
    /// Seal original successful singleton receipts behind the actual worker
    /// barrier. This does not fit, import, or publish a cost model. The retained
    /// discovery and trials came through Witness::capture before this call.
    pub async fn capture_reference_source_v1(
        &mut self,
        collector: &CalibrationReferenceCollector,
        destination: &Path,
    ) -> Result<CalibrationReferenceSourceV1> {
        let source = self.capture_reference_source_memory_v1(collector).await?;
        let path = assemble::publish(destination, &source.bytes)?;
        Ok(CalibrationReferenceSourceV1 {
            path,
            seal: source.seal,
        })
    }

    pub(in crate::continuous_engine::inner::calibration) async fn capture_reference_source_memory_v1(
        &mut self,
        collector: &CalibrationReferenceCollector,
    ) -> Result<MemoryReferenceSourceV1> {
        if self.pending.is_some()
            || self.indeterminate
            || !Arc::ptr_eq(&self.identity, &collector.session)
        {
            return Err(invalid(
                "reference source requires its quiescent original session",
            ));
        }
        let runtime = self
            .engine
            .inner
            .cost_runtime
            .as_ref()
            .ok_or_else(|| invalid("reference source requires original observation runtime"))?;
        let checkpoint = runtime
            .request_checkpoint()
            .map_err(|error| invalid(format!("reference source checkpoint: {error}")))?
            .wait()
            .await
            .map_err(|error| invalid(format!("reference source checkpoint: {error}")))?;
        let monotonic = runtime
            .clock
            .now_ns()
            .ok_or_else(|| invalid("reference source monotonic clock unavailable"))?;
        let unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()
            .and_then(|duration| u64::try_from(duration.as_nanos()).ok())
            .and_then(NonZeroU64::new)
            .ok_or_else(|| invalid("reference source wall anchor unavailable"))?;
        write(collector, checkpoint.accepted_ordinal, monotonic, unix)
    }
}

impl CalibrationReferenceCollector {
    /// Validate the sealed independent source, then use the identical product
    /// reference builder and loader used by the legacy training-cut adapter.
    pub fn finish_from_source_v1(
        self,
        source: &CalibrationReferenceSourceV1,
        destination: &Path,
    ) -> Result<CalibrationReferenceArtifact> {
        let joined = read(&self, source)?;
        self.assemble_from_joined(
            joined,
            source.seal.accepted_ordinal,
            source.seal.sha256,
            destination,
        )
    }

    pub(in crate::continuous_engine::inner::calibration) fn finish_from_memory_source_v1(
        self,
        source: &MemoryReferenceSourceV1,
    ) -> Result<assemble::MemoryReferenceArtifact> {
        let joined = read_bytes(&self, &source.seal, &source.bytes)?;
        self.assemble_memory(joined, source.seal.accepted_ordinal, source.seal.sha256)
    }
}

fn header(collector: &CalibrationReferenceCollector, source: &SourceSeal) -> Header {
    Header {
        artifact_type: ARTIFACT.into(),
        schema_version: 1,
        fingerprint: collector.fingerprint.clone(),
        plan_sha256: collector.plan_sha256,
        frozen_accepted_ordinal: collector.frozen_accepted_ordinal,
        frozen_at_monotonic_ns: collector.frozen_at_ns,
        accepted_ordinal: source.accepted_ordinal,
        checkpoint_monotonic_ns: source.checkpoint_monotonic_ns,
        generated_unix_ns: source.generated_unix_ns,
        clock_projection: CLOCK.into(),
        samples: source.samples,
    }
}

fn observation(witness: &Witness, record: u64, source: &SourceSeal) -> Result<Observation> {
    // Only this wire field is projected into Unix time. Durations and useful
    // work remain the original measured timing. There is no cross-process TTL
    // or asserted absolute wall-clock error in a fixed reference source.
    let measured_unix_ns = source
        .checkpoint_monotonic_ns
        .checked_sub(witness.sample.observed_at_ns)
        .and_then(|age| source.generated_unix_ns.get().checked_sub(age))
        .filter(|at| *at > 0)
        .ok_or_else(|| invalid("reference observation is outside checkpoint clock"))?;
    if witness.sample.boundary != CostBoundary::PreparationToCommit
        || witness.sample.outcome != WaveObservationOutcome::Completed
        || witness.commit.committed_at_ns > witness.sample.observed_at_ns
    {
        return Err(invalid(
            "reference source lost original successful singleton receipt",
        ));
    }
    Ok(Observation {
        accepted_ordinal: witness.accepted,
        observed_at_monotonic_ns: witness.sample.observed_at_ns,
        sample: ProfileSampleV2 {
            source_record: record,
            measured_unix_ns,
            shape: ProfileWaveShapeV2::from(&witness.sample.actual_shape),
            boundary: profile::ProfileCostBoundary::PreparationToCommit,
            outcome: profile::ProfileObservationOutcome::Completed {},
            timing: timing(&witness.sample.timing),
        },
        host: witness.host,
        commit: Commit::from(&witness.commit),
    })
}

fn timing(
    value: &ferrum_scheduler::implementations::continuous::cost_model::WaveTiming,
) -> profile::ProfileWaveTiming {
    let span =
        |value: Option<ferrum_scheduler::implementations::continuous::cost_model::MeasuredSpan>| {
            value.map(|value| profile::ProfileMeasuredSpan {
                start_ns: value.start_ns,
                end_ns: value.end_ns,
            })
        };
    profile::ProfileWaveTiming {
        wall_total_ns: value.wall_total_ns,
        device_elapsed_ns: value.device_elapsed_ns,
        stages: profile::ProfileStageTimings {
            prepare: span(value.stages.prepare),
            device_wait: span(value.stages.device_wait),
            commit: span(value.stages.commit),
            restore: span(value.stages.restore),
            maintenance: span(value.stages.maintenance),
        },
    }
}

struct BoundedBytes {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self
            .bytes
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > self.limit)
        {
            return Err(std::io::Error::other(
                "reference source byte limit exceeded",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl BoundedBytes {
    fn record(&mut self, record: &Record) -> Result<()> {
        serde_json::to_writer(&mut *self, record).map_err(json_error)?;
        self.write_all(b"\n").map_err(source::io_error)
    }
}

fn write(
    collector: &CalibrationReferenceCollector,
    accepted_ordinal: u64,
    checkpoint_monotonic_ns: u64,
    generated_unix_ns: NonZeroU64,
) -> Result<MemoryReferenceSourceV1> {
    let requested = collector.requested_witnesses(accepted_ordinal)?;
    if checkpoint_monotonic_ns < collector.frozen_at_ns {
        return Err(invalid("reference checkpoint precedes protocol freeze"));
    }
    let mut receipt = SourceSeal {
        sha256: [0; 32],
        bytes: 0,
        accepted_ordinal,
        checkpoint_monotonic_ns,
        generated_unix_ns,
        samples: requested.len(),
    };
    let mut output = BoundedBytes {
        bytes: Vec::new(),
        limit: collector.plan.limits.max_file_bytes.get(),
    };
    output.record(&Record::Header {
        header: header(collector, &receipt),
    })?;
    for (index, witness) in requested.values().enumerate() {
        if profile::ProfileFingerprint::from(&witness.sample.fingerprint) != collector.fingerprint {
            return Err(invalid("reference source fingerprint changed"));
        }
        output.record(&Record::Observation {
            observation: observation(witness, index as u64 + 1, &receipt)?,
        })?;
    }
    output.record(&Record::Summary {
        accepted_ordinal,
        samples: requested.len(),
    })?;
    receipt.sha256 = Sha256::digest(&output.bytes).into();
    receipt.bytes = output.bytes.len() as u64;
    Ok(MemoryReferenceSourceV1 {
        seal: receipt,
        bytes: output.bytes,
    })
}

fn read(
    collector: &CalibrationReferenceCollector,
    receipt: &CalibrationReferenceSourceV1,
) -> Result<source::Joined> {
    let file = std::fs::File::open(&receipt.path).map_err(source::io_error)?;
    let metadata = file.metadata().map_err(source::io_error)?;
    if !metadata.is_file()
        || metadata.len() != receipt.seal.bytes
        || receipt.seal.bytes > collector.plan.limits.max_file_bytes.get() as u64
    {
        return Err(invalid("reference source length changed"));
    }
    let mut bytes = Vec::new();
    file.take(receipt.seal.bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(source::io_error)?;
    read_bytes(collector, &receipt.seal, &bytes)
}

fn read_bytes(
    collector: &CalibrationReferenceCollector,
    receipt: &SourceSeal,
    bytes: &[u8],
) -> Result<source::Joined> {
    let requested = collector.requested_witnesses(receipt.accepted_ordinal)?;
    if receipt.bytes == 0
        || receipt.bytes > collector.plan.limits.max_file_bytes.get() as u64
        || receipt.samples != requested.len()
        || receipt.checkpoint_monotonic_ns < collector.frozen_at_ns
    {
        return Err(invalid("invalid independent reference source receipt"));
    }
    if bytes.len() as u64 != receipt.bytes
        || <[u8; 32]>::from(Sha256::digest(&bytes)) != receipt.sha256
    {
        return Err(invalid("reference source bytes changed"));
    }
    let mut lines = bytes
        .strip_suffix(b"\n")
        .ok_or_else(|| invalid("reference source has an unterminated record"))?
        .split(|byte| *byte == b'\n');
    let parse = |line: Option<&[u8]>| -> Result<Record> {
        serde_json::from_slice(line.ok_or_else(|| invalid("incomplete reference source"))?)
            .map_err(json_error)
    };
    match parse(lines.next())? {
        Record::Header { header: actual } if actual == header(collector, receipt) => {}
        _ => {
            return Err(invalid(
                "reference source header differs from frozen protocol/checkpoint",
            ))
        }
    }
    let mut samples = BTreeMap::new();
    for (index, (&ordinal, witness)) in requested.iter().enumerate() {
        let expected = observation(witness, index as u64 + 1, receipt)?;
        match parse(lines.next())? {
            Record::Observation {
                observation: actual,
            } if actual == expected => {
                samples.insert(ordinal, actual.sample);
            }
            _ => {
                return Err(invalid(
                    "reference source differs from original observation/commit",
                ))
            }
        }
    }
    match parse(lines.next())? {
        Record::Summary {
            accepted_ordinal,
            samples,
        } if accepted_ordinal == receipt.accepted_ordinal && samples == requested.len() => {}
        _ => return Err(invalid("reference source summary differs from checkpoint")),
    }
    if lines.next().is_some() {
        return Err(invalid("reference source has data after summary"));
    }
    Ok(source::Joined {
        samples,
        generated_unix_ns: receipt.generated_unix_ns,
    })
}
