//! Read the existing cut schema, retaining only requested singleton records.
use super::*;
use profile::v2::ProfileSampleV2;
use serde::de::IgnoredAny;
use std::{
    fs::File,
    io::{BufRead, BufReader, Read},
};

// Matches the product export/import hard ceiling, not an evidence-quality gate.
const MAX_CUT_BYTES: u64 = ferrum_types::SloCostProfileImportConfig::MAX_FILE_BYTES as u64;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Clock {
    wall_unix_ns: u64,
    monotonic_ns: u64,
}
#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
enum Training {
    Recorded {},
    Skipped {
        #[serde(rename = "reason")]
        _reason: IgnoredAny,
    },
    Error {
        #[serde(rename = "reason")]
        _reason: IgnoredAny,
    },
    Unavailable {},
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Record {
    Header {
        artifact_type: String,
        schema_version: u32,
        accepted_ordinal: u64,
        fingerprint: profile::ProfileFingerprint,
        #[serde(rename = "settings")]
        _settings: IgnoredAny,
        #[serde(rename = "producer")]
        _producer: IgnoredAny,
        opening: Clock,
        declared_clock_max_error_ns: u64,
        #[serde(rename = "coverage")]
        _coverage: IgnoredAny,
    },
    Observation {
        accepted_ordinal: u64,
        observed_at_monotonic_ns: u64,
        fingerprint: profile::ProfileFingerprint,
        training: Training,
        #[serde(rename = "pre_update_prediction")]
        _prediction: IgnoredAny,
        sample: ProfileSampleV2,
    },
    HostStagesV1 {
        accepted_ordinal: u64,
        source_record: Option<u64>,
        #[serde(rename = "legacy_rejection")]
        _legacy_rejection: Option<IgnoredAny>,
        #[serde(rename = "evidence")]
        _evidence: IgnoredAny,
    },
    HostContentTrainingV1 {
        accepted_ordinal: u64,
        host_source_record: Option<u64>,
        #[serde(rename = "evaluation")]
        _evaluation: IgnoredAny,
        #[serde(rename = "sample")]
        _sample: IgnoredAny,
    },
    HostRowMultisetTrainingV2 {
        accepted_ordinal: u64,
        host_source_record: Option<u64>,
        #[serde(rename = "evaluation")]
        _evaluation: IgnoredAny,
        #[serde(rename = "sample")]
        _sample: IgnoredAny,
    },
    Summary {
        accepted_ordinal: u64,
        closing: Clock,
        #[serde(rename = "counts")]
        _counts: IgnoredAny,
        #[serde(rename = "sink")]
        _sink: IgnoredAny,
        #[serde(rename = "training")]
        _training: IgnoredAny,
        #[serde(rename = "coverage")]
        _coverage: IgnoredAny,
    },
}
pub(super) struct Joined {
    pub samples: BTreeMap<u64, ProfileSampleV2>,
    pub generated_unix_ns: NonZeroU64,
}
pub(super) fn read(
    artifact: &CalibrationProfileArtifact,
    fingerprint: &profile::ProfileFingerprint,
    requested: &BTreeMap<u64, &Witness>,
) -> Result<Joined> {
    if artifact.source_bytes == 0
        || artifact.source_bytes > MAX_CUT_BYTES
        || artifact.source_digest == [0; 32]
        || artifact.source_sha256 != hex(&artifact.source_digest)
    {
        return Err(invalid("invalid reference cut source receipt"));
    }
    let file = File::open(&artifact.source).map_err(io_error)?;
    let metadata = file.metadata().map_err(io_error)?;
    if !metadata.is_file() {
        return Err(invalid("reference cut must be a regular file"));
    }
    if metadata.len() != artifact.source_bytes {
        return Err(invalid("reference cut length differs from its receipt"));
    }
    let mut reader = BufReader::new(file.take(artifact.source_bytes + 1));
    let mut line = Vec::new();
    let mut bytes = 0_u64;
    let mut hash = Sha256::new();
    let mut opening = None;
    let mut closing = None;
    let mut clock_error = 0;
    let mut last_accepted = 0;
    let mut last_record = None;
    let mut schema = 0;
    let mut last_was_observation = false;
    let mut last_was_stages = false;
    let mut last_host_record = None;
    let mut observations = 0_u64;
    let mut samples = BTreeMap::new();
    loop {
        line.clear();
        let count = reader.read_until(b'\n', &mut line).map_err(io_error)?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(count as u64)
            .filter(|n| *n <= artifact.source_bytes)
            .ok_or_else(|| invalid("reference cut byte limit exceeded"))?;
        hash.update(&line);
        if closing.is_some() {
            return Err(invalid("reference cut has data after its summary"));
        }
        let record: Record = serde_json::from_slice(&line).map_err(json_error)?;
        let host_schema = match &record {
            Record::HostContentTrainingV1 { .. } => 3,
            Record::HostRowMultisetTrainingV2 { .. } => 4,
            _ => 0,
        };
        match record {
            Record::Header {
                artifact_type,
                schema_version,
                accepted_ordinal,
                fingerprint: actual,
                opening: at,
                declared_clock_max_error_ns,
                ..
            } => {
                if opening.is_some()
                    || observations != 0
                    || artifact_type != "ferrum.cost-training-cut"
                    || !matches!(schema_version, 1 | 2 | 3 | 4)
                    || accepted_ordinal != artifact.accepted_ordinal
                    || actual != *fingerprint
                    || at.wall_unix_ns == 0
                {
                    return Err(invalid("incompatible or repeated reference cut header"));
                }
                opening = Some(at);
                schema = schema_version;
                clock_error = declared_clock_max_error_ns;
            }
            Record::Observation {
                accepted_ordinal,
                observed_at_monotonic_ns,
                fingerprint: source_fingerprint,
                training,
                sample,
                ..
            } => {
                let at = opening.ok_or_else(|| invalid("reference cut has no header"))?;
                let projected = observed_at_monotonic_ns
                    .checked_sub(at.monotonic_ns)
                    .and_then(|delta| at.wall_unix_ns.checked_add(delta));
                if accepted_ordinal <= last_accepted
                    || accepted_ordinal > artifact.accepted_ordinal
                    || last_record.is_some_and(|old| sample.source_record <= old)
                    || projected != Some(sample.measured_unix_ns)
                {
                    return Err(invalid(
                        "cut source order/fingerprint/original clock projection mismatch",
                    ));
                }
                last_accepted = accepted_ordinal;
                last_record = Some(sample.source_record);
                last_was_observation = true;
                last_was_stages = false;
                observations += 1;
                if let Some(witness) = requested.get(&accepted_ordinal) {
                    let actual = &witness.sample;
                    if !matches!(training,Training::Recorded{}) || source_fingerprint!=*fingerprint
                        || actual.observed_at_ns!=observed_at_monotonic_ns
                        || sample.shape!=ProfileWaveShapeV2::from(&actual.actual_shape)
                        || sample.boundary!=profile::ProfileCostBoundary::PreparationToCommit
                        || sample.outcome!=(profile::ProfileObservationOutcome::Completed{})
                        || ferrum_scheduler::implementations::continuous::cost_model::WaveTiming::from(sample.timing.clone())!=actual.timing
                    {return Err(invalid("reference source is not the exact Recorded actual measurement"));}
                    samples.insert(accepted_ordinal, sample);
                }
            }
            Record::HostStagesV1 {
                accepted_ordinal,
                source_record,
                ..
            } => {
                // An auxiliary row never satisfies a requested measurement.
                // A same-entry suffix must name the immediately preceding cost
                // record; StagesOnly has its own FIFO position and no record ID.
                let joined = match source_record {
                    Some(record) => {
                        last_was_observation
                            && accepted_ordinal == last_accepted
                            && Some(record) == last_record
                    }
                    None => accepted_ordinal > last_accepted,
                };
                if opening.is_none()
                    || !matches!(schema, 2 | 3 | 4)
                    || !joined
                    || accepted_ordinal > artifact.accepted_ordinal
                {
                    return Err(invalid(
                        "invalid auxiliary host-stage cut ordering or schema",
                    ));
                }
                last_accepted = accepted_ordinal;
                last_was_observation = false;
                last_was_stages = true;
            }
            Record::HostContentTrainingV1 {
                accepted_ordinal,
                host_source_record,
                ..
            }
            | Record::HostRowMultisetTrainingV2 {
                accepted_ordinal,
                host_source_record,
                ..
            } => {
                if schema != host_schema
                    || accepted_ordinal != last_accepted
                    || host_source_record.is_some() != last_was_stages
                    || !(last_was_stages || last_was_observation)
                    || host_source_record
                        .zip(last_host_record)
                        .is_some_and(|(new, old)| new <= old)
                {
                    return Err(invalid("invalid separate host-content evaluation ordering"));
                }
                if host_source_record.is_some() {
                    last_host_record = host_source_record;
                }
                // A host-settled profile sample is never a token-commit witness.
                last_was_observation = false;
                last_was_stages = false;
            }
            Record::Summary {
                accepted_ordinal,
                closing: at,
                ..
            } => {
                let start = opening.ok_or_else(|| invalid("reference cut has no opening clock"))?;
                let mono = at.monotonic_ns.checked_sub(start.monotonic_ns);
                let wall = at.wall_unix_ns.checked_sub(start.wall_unix_ns);
                if accepted_ordinal != artifact.accepted_ordinal
                    || at.wall_unix_ns == 0
                    || mono.zip(wall).is_none_or(|(m, w)| {
                        u128::from(m) > u128::from(w) + 2 * u128::from(clock_error)
                    })
                    || requested
                        .values()
                        .any(|row| row.sample.observed_at_ns > at.monotonic_ns)
                    || samples
                        .values()
                        .any(|sample| sample.measured_unix_ns > at.wall_unix_ns)
                {
                    return Err(invalid(
                        "reference cut summary/clock contradicts original measurements",
                    ));
                }
                closing = Some(at);
            }
        }
    }
    let digest: [u8; 32] = hash.finalize().into();
    if bytes != artifact.source_bytes
        || digest != artifact.source_digest
        || observations != artifact.raw_retained_observations
        || samples.len() != requested.len()
    {
        return Err(invalid(
            "reference cut changed or required original observations were not retained",
        ));
    }
    let generated = closing
        .and_then(|clock| NonZeroU64::new(clock.wall_unix_ns))
        .ok_or_else(|| invalid("reference cut summary missing"))?;
    Ok(Joined {
        samples,
        generated_unix_ns: generated,
    })
}
pub(super) fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
pub(super) fn io_error(error: std::io::Error) -> FerrumError {
    FerrumError::backend(format!("reference artifact IO: {error}"))
}
