use super::*;
use crate::slo::{
    AdmissionEvidence, RawOutputCount, RawOutputCountSource, RequestOutcome, VisibleTextEvidence,
};
use crate::{BenchmarkPhase, BenchmarkRequestRecord, ItlEvidenceSource, RequestItlEvidence};
use std::collections::BTreeSet;

fn aligned<'a, T>(
    values: &'a Option<Vec<Vec<T>>>,
    repeat: usize,
    repeats: usize,
    rows: usize,
    name: &str,
) -> Result<Option<&'a [T]>, OfflineSloError> {
    values
        .as_ref()
        .map(|values| {
            if values.len() != repeats || values[repeat].len() != rows {
                return Err(error(format!("{name} is not aligned with request_records")));
            }
            Ok(values[repeat].as_slice())
        })
        .transpose()
}

pub(super) fn cell(
    index: usize,
    report: &BenchReport,
    config: &SloEvaluationConfig,
) -> Result<OfflineSloCell, OfflineSloError> {
    let repeat_count = report.n_repeats as usize;
    if report
        .request_records
        .as_ref()
        .is_some_and(|rows| rows.len() != repeat_count)
    {
        return Err(error("request_records repeat count mismatch"));
    }
    if !report.repeat_metrics.is_empty() && report.repeat_metrics.len() != repeat_count {
        return Err(error("partial repeat_metrics cannot be paired safely"));
    }
    let mut repeats = Vec::with_capacity(repeat_count);
    for repeat in 0..repeat_count {
        let records = report
            .request_records
            .as_ref()
            .map(|rows| rows[repeat].as_slice());
        let rows = records.map_or(0, <[_]>::len);
        if rows > report.n_requests_per_run as usize {
            return Err(error("too many request records"));
        }
        let summary = report.repeat_metrics.get(repeat);
        if let Some(summary) = summary {
            if summary.repeat != repeat as u32 + 1
                || (summary.expected_requests != 0
                    && summary.expected_requests != report.n_requests_per_run)
            {
                return Err(error("repeat identity/expected request count mismatch"));
            }
            if !summary.duration_s.is_finite() || summary.duration_s <= 0.0 {
                return Err(error("invalid recorded measurement duration"));
            }
        }
        // These arrays are in completion order. Never sort correlation IDs and
        // then zip them to un-reordered token/event evidence.
        let events = if records.is_some() {
            aligned(
                &report.itl_evidence_per_request,
                repeat,
                repeat_count,
                rows,
                "ITL evidence",
            )?
        } else {
            None
        };
        let outputs = if records.is_some() {
            aligned(
                &report.output_tokens_per_request,
                repeat,
                repeat_count,
                rows,
                "output tokens",
            )?
        } else {
            None
        };
        let source = summary
            .map(|s| s.output_token_count_source.as_str())
            .or(report.output_token_count_source.as_deref());
        let mut seen = BTreeSet::new();
        let mut evidence = Vec::with_capacity(report.n_requests_per_run as usize);
        let mut reconstruction = Vec::with_capacity(report.n_requests_per_run as usize);
        for (position, record) in records.unwrap_or(&[]).iter().enumerate() {
            let identity = &record.correlation;
            if identity.phase != BenchmarkPhase::Measured
                || identity.repeat_index != repeat as u32
                || identity.request_index >= report.n_requests_per_run
                || !seen.insert(identity.request_index)
                || report
                    .cell_id
                    .as_ref()
                    .is_some_and(|id| *id != identity.cell_id)
                || report
                    .benchmark_run_id
                    .as_ref()
                    .is_some_and(|id| *id != identity.benchmark_run_id)
            {
                return Err(error("invalid/duplicate measured request correlation"));
            }
            let (row, issues) = request(
                record,
                events.map(|v| &v[position]),
                outputs.map(|v| v[position]),
                source,
            )?;
            evidence.push(row);
            reconstruction.push(RequestReconstruction {
                correlation: Some(identity.clone()),
                server_request_id: record.server_request_id.clone(),
                issues,
            });
        }
        for _ in rows..report.n_requests_per_run as usize {
            evidence.push(unknown());
            reconstruction.push(RequestReconstruction {
                correlation: None,
                server_request_id: None,
                issues: vec![ReconstructionIssue::MissingRequestRecord],
            });
        }
        let (evaluation, unevaluated) = if let Some(summary) = summary {
            (
                Some(evaluate_slo(config, &evidence, summary.duration_s).map_err(|e| error(e.0))?),
                None,
            )
        } else {
            (None, Some(evidence))
        };
        repeats.push(OfflineSloRepeat {
            repeat_index: repeat as u32,
            legacy_summary: summary.cloned(), reconstruction, evaluation,
            unevaluated_request_evidence: unevaluated,
            evaluation_unavailable: summary.is_none().then_some("missing original per-repeat measurement duration; no throughput or SLO evaluation inferred from aggregate statistics"),
        });
    }
    Ok(OfflineSloCell {
        input_cell_index: index,
        benchmark_run_id: report.benchmark_run_id.clone(),
        cell_id: report.cell_id.clone(),
        model: report.model.clone(),
        backend: report.backend.clone(),
        scenario: report.scenario,
        concurrency: report.concurrency,
        request_rate: report.request_rate,
        repeats,
    })
}

fn unknown() -> RequestSloEvidence {
    RequestSloEvidence {
        outcome: RequestOutcome::Pending,
        admission: AdmissionEvidence::Unknown,
        first_visible_ms: None,
        last_visible_ms: None,
        terminal_ms: None,
        usage_output_tokens: None,
        raw_output_count: RawOutputCount {
            count: 0,
            source: RawOutputCountSource::Unknown,
        },
        visible_text: None,
        strict_token_evidence: RequestItlEvidence::default(),
    }
}

fn request(
    record: &BenchmarkRequestRecord,
    events: Option<&RequestItlEvidence>,
    output_count: Option<u32>,
    source: Option<&str>,
) -> Result<(RequestSloEvidence, Vec<ReconstructionIssue>), OfflineSloError> {
    let mut row = unknown();
    let mut issues = Vec::new();
    let usage = events.and_then(|e| e.usage_output_tokens);
    if usage
        .zip(output_count)
        .is_some_and(|(usage, count)| usage != count)
    {
        return Err(error("conflicting aligned usage and output token counts"));
    }
    if usage.is_some() && matches!(source, Some("stream_chunks" | "none")) {
        return Err(error(
            "usage evidence contradicts repeat token-count source",
        ));
    }
    row.usage_output_tokens =
        usage.or_else(|| (source == Some("usage")).then_some(output_count).flatten());
    row.raw_output_count = match (row.usage_output_tokens, output_count, source) {
        (Some(count), _, _) => RawOutputCount {
            count,
            source: RawOutputCountSource::Usage,
        },
        (_, Some(count), Some("stream_chunks")) => RawOutputCount {
            count,
            source: RawOutputCountSource::TextEvents,
        },
        (_, count, _) => RawOutputCount {
            count: count.unwrap_or(0),
            source: RawOutputCountSource::Unknown,
        },
    };
    if row.usage_output_tokens.is_none() {
        issues.push(ReconstructionIssue::MissingUsage);
    }
    if let Some(events) = events {
        row.strict_token_evidence = events.clone();
    }
    let Some(timing) = &record.timing else {
        issues.push(ReconstructionIssue::MissingTiming);
        return Ok((row, issues));
    };
    row.outcome = if timing.success {
        RequestOutcome::Completed
    } else {
        RequestOutcome::Failed
    };
    if !timing.success {
        issues.push(ReconstructionIssue::FailedAdmissionUnknown);
    }
    // `observed_first_output=None` includes task-join placeholders. Their zero
    // terminal value is not a measured terminal timestamp.
    if timing.observed_first_output.is_some() {
        row.terminal_ms = Some(timing.reported_e2e_ms);
    }
    if let Some(events) = events {
        if timing.event_source != events.source {
            return Err(error("timing/event evidence sources disagree"));
        }
    }
    if timing.event_source != ItlEvidenceSource::SseDeltaEvents {
        issues.push(ReconstructionIssue::NonSseTiming);
        return Ok((row, issues));
    }
    if timing.observed_first_output == Some(true) {
        row.first_visible_ms = Some(timing.reported_ttft_ms);
    } else {
        issues.push(ReconstructionIssue::MissingObservedFirst);
    }
    let Some(events) = events else {
        issues.push(ReconstructionIssue::MissingVisibleCounts);
        return Ok((row, issues));
    };
    if timing
        .observed_first_output
        .is_some_and(|first| first != (events.output_events > 0))
    {
        return Err(error("observed-first flag contradicts visible event count"));
    }
    if timing
        .raw_event_gaps_ms
        .iter()
        .any(|gap| !gap.is_finite() || *gap < 0.0)
    {
        return Err(error("nonfinite/negative visible gap"));
    }
    // Preserve observed long stalls even when some other intervals are absent,
    // usage and events differ, or the request ultimately failed.
    row.visible_text = Some(VisibleTextEvidence {
        output_events: events.output_events,
        gaps_ms: timing.raw_event_gaps_ms.clone(),
        transport_coalesced_output_chunks: events.transport_coalesced_output_chunks,
    });
    let complete = timing.raw_event_gaps_ms.len()
        == events.output_events.saturating_sub(1) as usize
        && timing.raw_event_gaps_ms.len() == events.observed_intervals as usize;
    if complete {
        if let Some(first) = row.first_visible_ms {
            let last = first + timing.raw_event_gaps_ms.iter().sum::<f64>();
            if !last.is_finite() {
                return Err(error("last-visible timestamp overflow"));
            }
            row.last_visible_ms = Some(last);
        }
    } else {
        issues.push(ReconstructionIssue::IncompleteVisibleGaps);
    }
    if timing.success && row.first_visible_ms.is_some() && events.output_events > 0 {
        row.admission = AdmissionEvidence::Accepted;
    }
    Ok((row, issues))
}
