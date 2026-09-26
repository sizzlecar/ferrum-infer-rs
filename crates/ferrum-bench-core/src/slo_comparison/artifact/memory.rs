use super::*;
use files::{jsonl_rows, parse, Reader};
use serde::{Deserialize, Serialize};

// Wire-compatible with ferrum_interfaces::vnext::DeviceMemoryTelemetrySnapshot.
// Kept private to avoid introducing an execution-layer dependency into bench-core.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MetalRecord {
    pub schema_version: u32,
    pub record_type: String,
    pub source: String,
    pub scope: String,
    pub phase: String,
    pub pid: u32,
    pub device_registry_id: String,
    pub device_name: String,
    pub started_unix_ns: String,
    pub elapsed_ns: u64,
    pub current_allocated_bytes: u64,
    pub peak_allocated_bytes: u64,
    pub sample_count: u64,
    pub interval_ms: u64,
    pub max_sample_gap_ns: u64,
    pub error_count: u64,
    pub last_error: Option<String>,
    pub complete: bool,
    pub end_reason: Option<String>,
}

fn wall(anchor: u64, elapsed: u64) -> Result<u64, ArtifactError> {
    anchor
        .checked_add(elapsed)
        .ok_or_else(|| err("memory observation clock overflow"))
}

fn missing(kind: MemoryMeasurement) -> MemoryPeakEvidence {
    MemoryPeakEvidence {
        measurement: kind,
        peak_bytes: None,
        source_sha256: String::new(),
        complete: false,
        error_count: 0,
        started_unix_ns: 0,
        ended_unix_ns: 0,
        window: String::new(),
        sample_count: None,
        interval_ms: None,
        max_sample_gap_ns: None,
    }
}

fn metal(
    bytes: &[u8],
    hash: &str,
    execution: &ExecutionArtifact,
    limits: &ArtifactLoadLimits,
) -> Result<MemoryPeakEvidence, ArtifactError> {
    let rows = jsonl_rows(bytes, limits.max_memory_samples + 1)?;
    let first: MetalRecord = parse(rows[0])?;
    let anchor = first
        .started_unix_ns
        .parse::<u64>()
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| err("invalid Metal memory Unix anchor"))?;
    if first.schema_version != 1
        || first.source != "MTLDevice.currentAllocatedSize"
        || first.scope != "process_metal_device"
        || first.phase != "pre_weight_load_to_shutdown"
        || first.pid != execution.identity.server_pid
        || execution.device_registry_id.as_ref() != Some(&first.device_registry_id)
        || first.device_registry_id.is_empty()
        || first.device_name.is_empty()
        || first.interval_ms == 0
    {
        return Err(err(
            "Metal memory source/version/process/device identity mismatch",
        ));
    }
    let mut previous_elapsed = None;
    let mut first_elapsed = None;
    let mut peak = 0;
    let mut current = 0;
    let mut successful = 0;
    let mut samples = 0;
    let mut max_gap = 0;
    let mut errors = 0;
    let mut summary_seen = false;
    let mut finished = false;
    for row in rows {
        let row: MetalRecord = parse(row)?;
        if summary_seen
            || row.schema_version != first.schema_version
            || row.source != first.source
            || row.scope != first.scope
            || row.phase != first.phase
            || row.pid != first.pid
            || row.device_registry_id != first.device_registry_id
            || row.device_name != first.device_name
            || row.started_unix_ns != first.started_unix_ns
            || row.interval_ms != first.interval_ms
        {
            return Err(err("mixed or trailing Metal memory capture records"));
        }
        match row.record_type.as_str() {
            "sample" => {
                samples += 1;
                if samples > limits.max_memory_samples || row.complete || row.end_reason.is_some() {
                    return Err(err("invalid Metal sample or sample limit exceeded"));
                }
                if let Some(previous) = previous_elapsed {
                    max_gap = max_gap.max(
                        row.elapsed_ns
                            .checked_sub(previous)
                            .ok_or_else(|| err("Metal sample clock regressed"))?,
                    );
                }
                first_elapsed.get_or_insert(row.elapsed_ns);
                previous_elapsed = Some(row.elapsed_ns);
                if row.error_count == errors && row.sample_count == successful + 1 {
                    successful += 1;
                    current = row.current_allocated_bytes;
                    peak = peak.max(current);
                } else if row.error_count > errors
                    && row.sample_count == successful
                    && row.last_error.is_some()
                {
                    errors = row.error_count;
                } else {
                    return Err(err(
                        "Metal sample cumulative count/error transition is inconsistent",
                    ));
                }
            }
            "summary" => {
                if samples == 0 || previous_elapsed != Some(row.elapsed_ns) {
                    return Err(err("Metal summary has no matching final sample"));
                }
                summary_seen = true;
                finished = row.end_reason.as_deref() == Some("shutdown") && errors == 0;
                if row.complete != finished {
                    return Err(err(
                        "Metal summary completion conflicts with shutdown/error evidence",
                    ));
                }
            }
            _ => return Err(err("unsupported Metal memory record type")),
        }
        if row.current_allocated_bytes != current
            || row.peak_allocated_bytes != peak
            || row.sample_count != successful
            || row.max_sample_gap_ns != max_gap
            || row.error_count != errors
            || (errors == 0 && row.last_error.is_some())
        {
            return Err(err("Metal cumulative summary does not match raw samples"));
        }
    }
    let start = first_elapsed.ok_or_else(|| err("Metal capture contains no samples"))?;
    let end = previous_elapsed.expect("sample checked");
    Ok(MemoryPeakEvidence {
        measurement: MemoryMeasurement::SampledDeviceAllocation,
        peak_bytes: (successful > 0).then_some(peak),
        source_sha256: hash.into(),
        complete: summary_seen && finished,
        error_count: errors,
        started_unix_ns: wall(anchor, start)?,
        ended_unix_ns: wall(anchor, end)?,
        window: first.phase,
        sample_count: Some(successful),
        interval_ms: Some(first.interval_ms),
        max_sample_gap_ns: Some(max_gap),
    })
}

fn sampled(
    bytes: &[u8],
    hash: &str,
    execution: &ExecutionArtifact,
    kind: MemoryMeasurement,
    limits: &ArtifactLoadLimits,
) -> Result<MemoryPeakEvidence, ArtifactError> {
    let source: SampledMemoryArtifact = parse(bytes)?;
    if source.schema_version != 1
        || source.identity != execution.identity
        || source.measurement != kind
        || source.collector.trim().is_empty()
        || source.window.trim().is_empty()
        || source.started_unix_ns == 0
        || source.interval_ms == 0
        || source.observations.is_empty()
        || source.observations.len() > limits.max_memory_samples
    {
        return Err(err("sampled memory schema/identity/count is invalid"));
    }
    let mut previous = None;
    let mut max_gap = 0;
    let mut peak = None::<u64>;
    let mut successful = 0;
    let mut errors = 0;
    for observation in &source.observations {
        if let Some(previous) = previous {
            max_gap = max_gap.max(
                observation
                    .elapsed_ns
                    .checked_sub(previous)
                    .ok_or_else(|| err("sampled memory clock regressed"))?,
            );
        }
        previous = Some(observation.elapsed_ns);
        match (observation.bytes, observation.error.as_deref()) {
            (Some(bytes), None) => {
                successful += 1;
                peak = Some(peak.unwrap_or(0).max(bytes));
            }
            (None, Some(message)) if !message.is_empty() => errors += 1,
            _ => return Err(err("memory observation needs exactly one value or error")),
        }
    }
    let end = previous.expect("nonempty observations");
    if source
        .finished_elapsed_ns
        .is_some_and(|finished| finished < end)
    {
        return Err(err("memory finish precedes final sample"));
    }
    Ok(MemoryPeakEvidence {
        measurement: kind,
        peak_bytes: peak,
        source_sha256: hash.into(),
        complete: source.finished_elapsed_ns.is_some() && errors == 0,
        error_count: errors,
        started_unix_ns: wall(source.started_unix_ns, source.observations[0].elapsed_ns)?,
        ended_unix_ns: wall(source.started_unix_ns, end)?,
        window: source.window,
        sample_count: Some(successful),
        interval_ms: Some(source.interval_ms),
        max_sample_gap_ns: Some(max_gap),
    })
}

fn rss(
    bytes: &[u8],
    hash: &str,
    execution: &ExecutionArtifact,
) -> Result<MemoryPeakEvidence, ArtifactError> {
    let source: MaximumRssArtifact = parse(bytes)?;
    if source.schema_version != 1
        || source.identity != execution.identity
        || source.collector.trim().is_empty()
        || source.window.trim().is_empty()
        || source.process_started_unix_ns == 0
        || source
            .process_ended_unix_ns
            .is_some_and(|end| end <= source.process_started_unix_ns)
    {
        return Err(err("maximum RSS schema/identity/window is invalid"));
    }
    Ok(MemoryPeakEvidence {
        measurement: MemoryMeasurement::ProcessMaximumRss,
        peak_bytes: source.maximum_rss_bytes,
        source_sha256: hash.into(),
        complete: source.process_ended_unix_ns.is_some()
            && source.maximum_rss_bytes.is_some()
            && source.error.is_none(),
        error_count: u64::from(source.error.is_some()),
        started_unix_ns: source.process_started_unix_ns,
        ended_unix_ns: source
            .process_ended_unix_ns
            .unwrap_or(source.process_started_unix_ns),
        window: source.window,
        sample_count: None,
        interval_ms: None,
        max_sample_gap_ns: None,
    })
}

pub(super) fn load(
    reader: &mut Reader<'_>,
    refs: &MemoryArtifactRefs,
    execution: &ExecutionArtifact,
) -> Result<Option<PeakMemoryEvidence>, ArtifactError> {
    if refs.device.is_none()
        && refs.footprint.is_none()
        && refs.rss.is_none()
        && refs.process_lifetime.is_none()
    {
        return Ok(None);
    }
    if refs.process_lifetime.is_some() && (refs.footprint.is_some() || refs.rss.is_some()) {
        return Err(err(
            "native process peaks cannot be mixed with footprint/RSS overrides",
        ));
    }
    let device = match &refs.device {
        Some(DeviceMemoryRef::FerrumMetalV1 { file }) => {
            metal(&reader.read(file)?, &file.sha256, execution, reader.limits)?
        }
        Some(DeviceMemoryRef::SampledMemoryV1 { file }) => sampled(
            &reader.read(file)?,
            &file.sha256,
            execution,
            MemoryMeasurement::SampledDeviceAllocation,
            reader.limits,
        )?,
        None => missing(MemoryMeasurement::SampledDeviceAllocation),
    };
    if let Some(refs) = &refs.process_lifetime {
        let (os_footprint, maximum_rss) = macos_time::load(reader, refs, execution)?;
        return Ok(Some(PeakMemoryEvidence {
            device_allocation: device,
            os_footprint,
            maximum_rss,
        }));
    }
    let footprint = match &refs.footprint {
        Some(file) => sampled(
            &reader.read(file)?,
            &file.sha256,
            execution,
            MemoryMeasurement::SampledOsPhysicalFootprint,
            reader.limits,
        )?,
        None => missing(MemoryMeasurement::SampledOsPhysicalFootprint),
    };
    let rss = match &refs.rss {
        Some(file) => rss(&reader.read(file)?, &file.sha256, execution)?,
        None => missing(MemoryMeasurement::ProcessMaximumRss),
    };
    Ok(Some(PeakMemoryEvidence {
        device_allocation: device,
        os_footprint: footprint,
        maximum_rss: rss,
    }))
}
