//! Read-only import of native macOS time -l. No sampling, process inspection,
//! or caller-supplied memory values. Parentage remains a source declaration.
use super::*;

// A native report is a few dozen lines. This is an input resource bound, not
// a workload/measurement threshold; the shared Reader also enforces its budget.
const MAX_TIME_REPORT_BYTES: usize = 64 * 1024;

#[derive(Debug, Default, PartialEq, Eq)]
struct NativePeaks {
    footprint: Option<u64>,
    rss: Option<u64>,
}

fn parse_peaks(bytes: &[u8]) -> Result<NativePeaks, ArtifactError> {
    if bytes.len() > MAX_TIME_REPORT_BYTES {
        return Err(err("native time report exceeds byte limit"));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| err("native time report is not UTF-8"))?;
    let mut peaks = NativePeaks::default();
    for line in text.lines() {
        for (label, value) in [
            ("maximum resident set size", &mut peaks.rss),
            ("peak memory footprint", &mut peaks.footprint),
        ] {
            if !line.contains(label) {
                continue;
            }
            if value.is_some() {
                return Err(err(format!("duplicate native time field: {label}")));
            }
            let number = line
                .trim()
                .strip_suffix(label)
                .filter(|prefix| {
                    prefix
                        .as_bytes()
                        .last()
                        .is_some_and(u8::is_ascii_whitespace)
                })
                .map(str::trim)
                .filter(|number| {
                    !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit())
                })
                .ok_or_else(|| err(format!("invalid native time bytes or units: {label}")))?;
            *value = Some(
                number
                    .parse()
                    .map_err(|_| err("native time byte count overflow"))?,
            );
        }
    }
    // Missing fields are incomplete evidence, not zero memory. Malformed or
    // duplicate fields above are corrupt/ambiguous sources and fail loading.
    Ok(peaks)
}

fn parse_exit(bytes: &[u8]) -> Result<u8, ArtifactError> {
    if bytes.len() > 64 {
        return Err(err("native time exit status exceeds byte limit"));
    }
    let value = std::str::from_utf8(bytes)
        .ok()
        .map(str::trim)
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| err("native time exit status must be one decimal shell status"))?;
    value
        .parse()
        .map_err(|_| err("native time exit status is outside 0..255"))
}

pub(super) fn load(
    reader: &mut Reader<'_>,
    reference: &ProcessLifetimeMemoryRef,
    execution: &ExecutionArtifact,
) -> Result<(MemoryPeakEvidence, MemoryPeakEvidence), ArtifactError> {
    let ProcessLifetimeMemoryRef::MacosTimeLV1 { capture } = reference;
    let capture: MacosTimeCaptureArtifact = parse(&reader.read(capture)?)?;
    if capture.schema_version != 1
        || capture.identity != execution.identity
        || capture.identity.server_pid == 0
        || capture.process_started_unix_ns == 0
        || capture
            .process_ended_unix_ns
            .is_some_and(|end| end <= capture.process_started_unix_ns)
    {
        return Err(err(
            "native time capture schema/identity/lifetime is invalid",
        ));
    }
    if capture.time_output.bytes > MAX_TIME_REPORT_BYTES as u64 {
        return Err(err("native time report exceeds byte limit"));
    }
    let peaks = parse_peaks(&reader.read(&capture.time_output)?)?;
    let exit = capture
        .exit_status
        .as_ref()
        .map(|file| {
            if file.bytes > 64 {
                return Err(err("native time exit status exceeds byte limit"));
            }
            parse_exit(&reader.read(file)?)
        })
        .transpose()?;
    if let Some(source) = &capture.launch_evidence {
        let _ = reader.read(source)?;
    }
    let complete = exit == Some(0)
        && capture.process_ended_unix_ns.is_some()
        && peaks.footprint.is_some_and(|bytes| bytes > 0)
        && peaks.rss.is_some_and(|bytes| bytes > 0);
    let evidence = |measurement, peak_bytes| MemoryPeakEvidence {
        measurement,
        peak_bytes,
        source_sha256: capture.time_output.sha256.clone(),
        complete,
        error_count: u64::from(exit.is_some_and(|status| status != 0)),
        started_unix_ns: capture.process_started_unix_ns,
        ended_unix_ns: capture
            .process_ended_unix_ns
            .unwrap_or(capture.process_started_unix_ns),
        window: "process_lifetime".into(),
        sample_count: None,
        interval_ms: None,
        max_sample_gap_ns: None,
    };
    Ok((
        evidence(
            MemoryMeasurement::ProcessPeakPhysicalFootprint,
            peaks.footprint,
        ),
        evidence(MemoryMeasurement::ProcessMaximumRss, peaks.rss),
    ))
}

#[cfg(test)]
mod tests;
