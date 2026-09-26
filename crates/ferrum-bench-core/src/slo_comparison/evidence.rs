use super::*;
use crate::{
    dataset::ShareGptDatasetEvidence,
    slo::{evaluate_slo, SloStatus},
    BenchmarkPhase, ItlEvidenceSource, QualityIssueCounts, Scenario,
};

fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0)
}
fn same_numbers(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&a, &b)| close(a, b))
}
fn no_quality_failures(quality: &QualityIssueCounts) -> bool {
    quality.bad_output == 0
        && quality.malformed_stream == 0
        && quality.missing_done == 0
        && quality.duplicate_done == 0
        && quality.zero_output_tokens == 0
        && quality.http_500 == 0
        && quality.panic == 0
}
fn same_dataset_policy(
    actual: &ShareGptDatasetEvidence,
    expected: &ShareGptDatasetEvidence,
) -> bool {
    actual.dataset == expected.dataset
        && same_digest(&actual.source_sha256, &expected.source_sha256)
        && actual.source_format == expected.source_format
        && same_digest(&actual.tokenizer_sha256, &expected.tokenizer_sha256)
        && actual.filter == expected.filter
        && actual.counts == expected.counts
        && actual.prompt_seed == expected.prompt_seed
        && actual.sampling == expected.sampling
        && actual.ignore_eos == expected.ignore_eos
        && actual.enable_thinking == expected.enable_thinking
}
pub(super) struct ArmValues {
    pub(super) metrics: BTreeMap<ComparisonMetric, f64>,
    pub(super) server_inputs: Vec<u32>,
    pub(super) outputs: Vec<u32>,
    pub(super) status: ComparisonStatus,
    pub(super) absolute_slo_status: SloStatus,
    pub(super) issues: Vec<String>,
}

pub(super) fn memory_peak_issues(
    actual: &MemoryPeakEvidence,
    expected: MemoryMeasurement,
    execution: &ArmExecutionEvidence,
    policy: &FrozenMemoryPolicy,
) -> Vec<String> {
    let mut issues = Vec::new();
    let sampled = matches!(
        expected,
        MemoryMeasurement::SampledDeviceAllocation | MemoryMeasurement::SampledOsPhysicalFootprint
    );
    if actual.measurement != expected
        || actual.peak_bytes.is_none_or(|n| n == 0)
        || !actual.complete
        || actual.error_count != 0
        || !valid_digest(&actual.source_sha256)
        || actual.window.trim().is_empty()
        || actual.started_unix_ns > execution.measurement_started_unix_ns
        || actual.ended_unix_ns < execution.measurement_ended_unix_ns
        || actual.started_unix_ns >= actual.ended_unix_ns
        || (sampled
            && (actual.sample_count.is_none_or(|n| n == 0)
                || actual.interval_ms.is_none_or(|n| n == 0)
                || actual.max_sample_gap_ns.is_none()
                || actual
                    .sample_count
                    .zip(actual.max_sample_gap_ns)
                    .is_none_or(|(count, gap)| {
                        u128::from(count.saturating_sub(1)) * u128::from(gap)
                            < u128::from(
                                actual.ended_unix_ns.saturating_sub(actual.started_unix_ns),
                            )
                    })))
    {
        issues.push(format!("incomplete or incompatible {expected:?} evidence"));
    }
    if expected == MemoryMeasurement::ProcessPeakPhysicalFootprint
        && (actual.sample_count.is_some()
            || actual.interval_ms.is_some()
            || actual.max_sample_gap_ns.is_some())
    {
        issues.push("process lifetime footprint cannot claim sampling coverage".into());
    }
    let sampled_policy = match expected {
        MemoryMeasurement::SampledDeviceAllocation => Some(&policy.device_allocation),
        MemoryMeasurement::SampledOsPhysicalFootprint => policy.os_footprint.sampled(),
        MemoryMeasurement::ProcessPeakPhysicalFootprint | MemoryMeasurement::ProcessMaximumRss => {
            None
        }
    };
    if let Some(sampled) = sampled_policy {
        if actual.window != sampled.window
            || actual.interval_ms != Some(sampled.interval_ms)
            || actual
                .max_sample_gap_ns
                .is_none_or(|gap| gap > sampled.max_sample_gap_ns)
        {
            issues.push(format!(
                "{:?} does not follow the frozen sampling policy",
                actual.measurement
            ));
        }
    } else {
        let window = if expected == MemoryMeasurement::ProcessMaximumRss {
            &policy.maximum_rss_window
        } else {
            policy.os_footprint.window()
        };
        if actual.window != window {
            issues.push(format!("{expected:?} boundary differs from frozen policy"));
        }
    }
    issues
}

fn memory_issues(
    memory: Option<&PeakMemoryEvidence>,
    execution: &ArmExecutionEvidence,
    policy: &FrozenMemoryPolicy,
) -> Vec<String> {
    let Some(memory) = memory else {
        return vec!["missing device allocation / footprint / maximum RSS evidence".into()];
    };
    [
        (
            &memory.device_allocation,
            MemoryMeasurement::SampledDeviceAllocation,
        ),
        (&memory.os_footprint, policy.os_footprint.measurement()),
        (&memory.maximum_rss, MemoryMeasurement::ProcessMaximumRss),
    ]
    .into_iter()
    .flat_map(|(actual, expected)| memory_peak_issues(actual, expected, execution, policy))
    .collect()
}

pub(super) fn validate_arm(
    contract: &FrozenComparisonContract,
    concurrency: u32,
    pair: &FrozenPair,
    input: &ArmRepeatInput<'_>,
    candidate: bool,
    seen: &mut BTreeSet<(bool, String, String, u32)>,
) -> ArmValues {
    let report = input.legacy_benchmark;
    let expected_server = if candidate {
        &contract.candidate
    } else {
        &contract.baseline
    };
    let execution = input.execution;
    let mut values = ArmValues {
        metrics: BTreeMap::new(),
        server_inputs: Vec::new(),
        outputs: Vec::new(),
        status: ComparisonStatus::ObservedPass,
        absolute_slo_status: SloStatus::Unknown,
        issues: memory_issues(input.memory, execution, &contract.memory),
    };
    if execution.shared != contract.shared
        || execution.server != *expected_server
        || execution.capacity != contract.capacity
        || !valid_digest(&execution.source_manifest_sha256)
        || execution.measurement_started_unix_ns < contract.frozen_unix_ns
        || execution.measurement_ended_unix_ns <= execution.measurement_started_unix_ns
    {
        values.issues.push(
            "execution identity, fixed capacity or frozen measurement window mismatch".into(),
        );
    }
    if report.scenario != Scenario::ClosedLoop
        || report.concurrency != Some(concurrency)
        || report.request_rate.is_some()
        || report.model != expected_server.request_model_alias
        || report.backend != expected_server.backend
        || report.env_hash != report.env.hash()
        || report.env.http_request_sampling != Some(contract.sampling)
        || report.env.http_connection_mode.as_deref()
            != Some(contract.http_connection_mode.as_str())
    {
        values
            .issues
            .push("legacy report cell, client policy or environment identity mismatch".into());
    }
    let config_hash = report
        .env
        .runtime_config
        .entries
        .iter()
        .find(|entry| entry.key == "bench_slo_client_config_sha256");
    if config_hash.is_none_or(|entry| {
        !same_digest(
            &entry.effective_value,
            &contract.shared.client_slo_config_sha256,
        )
    }) {
        values
            .issues
            .push("legacy report does not bind the frozen client SLO config".into());
    }
    match (&report.benchmark_run_id, &report.cell_id) {
        (Some(run), Some(cell)) if !run.is_empty() && !cell.is_empty() => {
            if !seen.insert((
                candidate,
                run.clone(),
                cell.clone(),
                input.report_repeat_index,
            )) {
                values
                    .issues
                    .push("one observed run/cell/repeat was reused as independent evidence".into());
            }
        }
        _ => values
            .issues
            .push("missing benchmark run/cell correlation identity".into()),
    }
    let index = input.report_repeat_index as usize;
    let selection = report.dataset_evidence.as_ref().and_then(|dataset| {
        if !same_dataset_policy(dataset, &contract.dataset) {
            values
                .issues
                .push("ShareGPT source/filter/output policy mismatch".into());
        }
        dataset
            .repeats
            .iter()
            .find(|repeat| repeat.repeat_index == input.report_repeat_index)
    });
    match selection {
        Some(selection)
            if selection.samples == pair.selection.samples
                && selection.rng_seed == pair.selection.rng_seed
                && same_digest(
                    &selection.selection_sha256,
                    &pair.selection.selection_sha256,
                )
                && json_digest(&selection.samples)
                    .is_ok_and(|hash| same_digest(&hash, &selection.selection_sha256)) => {}
        _ => values
            .issues
            .push("missing or different ordered ShareGPT selection".into()),
    }
    let expected: Vec<_> = pair
        .selection
        .samples
        .iter()
        .filter(|sample| sample.phase == BenchmarkPhase::Measured)
        .collect();
    let warmup = pair.selection.samples.len() - expected.len();
    let Some(legacy) = report.repeat_metrics.get(index) else {
        values.issues.push("missing legacy repeat row".into());
        values.status = ComparisonStatus::Unknown;
        return values;
    };
    if report.n_repeats as usize != report.repeat_metrics.len()
        // The historical summary's display repeat is one-based; sidecar,
        // dataset and request correlation indices remain zero-based.
        || input.report_repeat_index.checked_add(1) != Some(legacy.repeat)
        || report.n_requests_per_run as usize != expected.len()
        || legacy.expected_requests as usize != expected.len()
        || report.warmup_requests as usize != warmup
        || legacy.warmup_expected as usize != warmup
    {
        values.issues.push(format!(
            "repeat/sample/warmup counts differ from frozen selection: local index={}, legacy one-based repeat={}, declared/observed repeats={}/{}, measured configured/legacy/frozen={}/{}/{}, warmup configured/legacy/frozen={}/{}/{}",
            input.report_repeat_index, legacy.repeat, report.n_repeats, report.repeat_metrics.len(),
            report.n_requests_per_run, legacy.expected_requests, expected.len(), report.warmup_requests,
            legacy.warmup_expected, warmup
        ));
    }
    if legacy.output_token_count_source != "usage"
        || report.output_token_count_source.as_deref() != Some("usage")
    {
        values
            .issues
            .push("output throughput source is not complete usage-token evidence".into());
    }
    if legacy.completed_requests as usize != expected.len()
        || legacy.errored_requests != 0
        || legacy.warmup_completed as usize != warmup
        || legacy.warmup_errored != 0
        || !no_quality_failures(&legacy.quality_issues)
        || !no_quality_failures(&legacy.warmup_quality_issues)
    {
        values.status = ComparisonStatus::Failed;
        values
            .issues
            .push("request or warmup output/protocol failure".into());
    }
    if input.evaluation.schema_version != crate::slo::SCHEMA_VERSION
        || input.evaluation.config != contract.slo
        || !close(input.evaluation.duration_s, legacy.duration_s)
        || input.evaluation.request_evidence.len() != expected.len()
    {
        values
            .issues
            .push("sidecar contract, duration or request count differs from legacy repeat".into());
    }
    if (execution
        .measurement_ended_unix_ns
        .saturating_sub(execution.measurement_started_unix_ns) as f64)
        / 1e9
        + 1e-6
        < legacy.duration_s
    {
        values
            .issues
            .push("execution evidence does not cover the measured duration".into());
    }
    let evaluation = match evaluate_slo(
        &contract.slo,
        &input.evaluation.request_evidence,
        input.evaluation.duration_s,
    ) {
        Ok(evaluation) => evaluation,
        Err(error) => {
            values
                .issues
                .push(format!("invalid raw SLO evidence: {error}"));
            values.status = ComparisonStatus::Unknown;
            return values;
        }
    };
    values.absolute_slo_status = evaluation.latency_and_outcome_status;
    if evaluation.outcomes.failed != 0
        || evaluation.outcomes.rejected != 0
        || evaluation.task_success.fail != 0
    {
        values.status = ComparisonStatus::Failed;
        values
            .issues
            .push("failed/rejected/zero-output request remains in the denominator".into());
    }
    if evaluation.outcomes.pending != 0
        || evaluation.admissions.unknown != 0
        || evaluation.task_success.unknown != 0
        || evaluation.admissions.accepted != expected.len() as u64
        || [
            evaluation.ttft.status,
            evaluation.tpot.status,
            evaluation.pooled_visible_itl.status,
        ]
        .contains(&SloStatus::Unknown)
    {
        values
            .issues
            .push("unknown admission, outcome or latency coverage".into());
    }
    if candidate && evaluation.latency_and_outcome_status == SloStatus::Fail {
        values.status = ComparisonStatus::Failed;
        values
            .issues
            .push("candidate violates the frozen absolute client-visible SLO".into());
    }
    if candidate
        && evaluation.latency_and_outcome_status != SloStatus::Pass
        && evaluation.latency_and_outcome_status != SloStatus::Fail
    {
        values
            .issues
            .push("candidate absolute SLO is not established".into());
    }
    let records = report
        .request_records
        .as_ref()
        .and_then(|rows| rows.get(index));
    let inputs = report
        .actual_input_tokens_per_request
        .as_ref()
        .and_then(|rows| rows.get(index));
    let outputs = report
        .output_tokens_per_request
        .as_ref()
        .and_then(|rows| rows.get(index));
    let server_inputs = report
        .server_input_tokens_per_request
        .as_ref()
        .and_then(|rows| rows.get(index));
    let itl = report
        .itl_evidence_per_request
        .as_ref()
        .and_then(|rows| rows.get(index));
    let aligned = records.is_some_and(|v| v.len() == expected.len())
        && inputs.is_some_and(|v| v.len() == expected.len())
        && outputs.is_some_and(|v| v.len() == expected.len())
        && server_inputs.is_some_and(|v| v.len() == expected.len())
        && itl.is_some_and(|v| v.len() == expected.len())
        && evaluation.request_evidence.len() == expected.len();
    if aligned {
        let (records, inputs, outputs, server_inputs, itl) = (
            records.unwrap(),
            inputs.unwrap(),
            outputs.unwrap(),
            server_inputs.unwrap(),
            itl.unwrap(),
        );
        values.outputs = outputs.clone();
        for (i, sample) in expected.iter().enumerate() {
            let evidence = &evaluation.request_evidence[i];
            let record = &records[i];
            let correlation = &record.correlation;
            if Some(&correlation.benchmark_run_id) != report.benchmark_run_id.as_ref()
                || Some(&correlation.cell_id) != report.cell_id.as_ref()
                || correlation.repeat_index != input.report_repeat_index
                || correlation.phase != BenchmarkPhase::Measured
                || correlation.request_index as usize != i
                || inputs[i] != sample.input_tokens
                || evidence.usage_output_tokens != Some(outputs[i])
                || evidence.strict_token_evidence != itl[i]
            {
                values.issues.push(format!(
                    "request {i} correlation/input/usage/ITL evidence mismatch"
                ));
            }
            if contract.dataset.ignore_eos && outputs[i] != sample.requested_output_tokens {
                values.status = ComparisonStatus::Failed;
                values.issues.push(format!(
                    "request {i} did not execute the frozen output budget"
                ));
            }
            match server_inputs[i] {
                Some(tokens) if tokens > 0 => values.server_inputs.push(tokens),
                _ => values
                    .issues
                    .push(format!("request {i} has missing server prompt usage")),
            }
            let text = evidence.visible_text.as_ref();
            let timing = record.timing.as_ref();
            let timing_matches = timing.is_some_and(|timing| {
                timing.success == (evidence.outcome == crate::slo::RequestOutcome::Completed)
                    && timing.event_source == ItlEvidenceSource::SseDeltaEvents
                    && timing.observed_first_output == Some(true)
                    && evidence
                        .first_visible_ms
                        .is_some_and(|first| close(first, timing.reported_ttft_ms))
                    && evidence
                        .terminal_ms
                        .is_some_and(|terminal| close(terminal, timing.reported_e2e_ms))
                    && text
                        .is_some_and(|text| same_numbers(&text.gaps_ms, &timing.raw_event_gaps_ms))
            });
            let last_matches = evidence
                .first_visible_ms
                .zip(evidence.last_visible_ms)
                .zip(text)
                .is_some_and(|((first, last), text)| {
                    close(first + text.gaps_ms.iter().sum::<f64>(), last)
                });
            if !timing_matches || !last_matches {
                values.issues.push(format!(
                    "request {i} raw timing or last-visible boundary mismatch"
                ));
            }
        }
        if outputs.iter().map(|&n| u64::from(n)).sum::<u64>() != legacy.output_tokens
            || inputs.iter().map(|&n| u64::from(n)).sum::<u64>() != legacy.actual_input_tokens
            || legacy.server_input_tokens
                != Some(values.server_inputs.iter().map(|&n| u64::from(n)).sum())
        {
            values
                .issues
                .push("legacy usage aggregates differ from aligned request evidence".into());
        }
    } else {
        values
            .issues
            .push("missing aligned legacy per-request evidence".into());
    }
    for (metric, value) in [
        (
            ComparisonMetric::TtftP50,
            evaluation.ttft.observed_percentiles_ms.map(|p| p.p50),
        ),
        (
            ComparisonMetric::TtftP99,
            evaluation.ttft.observed_percentiles_ms.map(|p| p.p99),
        ),
        (
            ComparisonMetric::TpotP50,
            evaluation.tpot.observed_percentiles_ms.map(|p| p.p50),
        ),
        (
            ComparisonMetric::TpotP99,
            evaluation.tpot.observed_percentiles_ms.map(|p| p.p99),
        ),
        (
            ComparisonMetric::VisibleItlP50,
            evaluation
                .pooled_visible_itl
                .observed_percentiles_ms
                .map(|p| p.p50),
        ),
        (
            ComparisonMetric::VisibleItlP99,
            evaluation
                .pooled_visible_itl
                .observed_percentiles_ms
                .map(|p| p.p99),
        ),
        (
            ComparisonMetric::SuccessfulUsageOutputTps,
            evaluation.successful_output.tokens_per_second,
        ),
    ] {
        match value {
            Some(value) if value.is_finite() && value > 0.0 => {
                values.metrics.insert(metric, value);
            }
            _ => values
                .issues
                .push(format!("{metric:?} has no positive eligible measurement")),
        }
    }
    if values.status != ComparisonStatus::Failed && !values.issues.is_empty() {
        values.status = ComparisonStatus::Unknown;
    }
    values
}

pub(super) fn observed_source(input: &ArmRepeatInput<'_>, values: &ArmValues) -> ObservedArmSource {
    let records = &input.evaluation.request_evidence;
    ObservedArmSource {
        benchmark_run_id: input.legacy_benchmark.benchmark_run_id.clone(),
        cell_id: input.legacy_benchmark.cell_id.clone(),
        report_repeat_index: input.report_repeat_index,
        execution: input.execution.clone(),
        absolute_slo_status: values.absolute_slo_status,
        offered_requests: records.len(),
        failed_requests: records
            .iter()
            .filter(|r| r.outcome == crate::slo::RequestOutcome::Failed)
            .count() as u64,
        rejected_requests: records
            .iter()
            .filter(|r| r.outcome == crate::slo::RequestOutcome::Rejected)
            .count() as u64,
        pending_requests: records
            .iter()
            .filter(|r| r.outcome == crate::slo::RequestOutcome::Pending)
            .count() as u64,
        observed_visible_gaps: records
            .iter()
            .filter_map(|r| r.visible_text.as_ref())
            .map(|text| text.gaps_ms.len())
            .sum(),
        visible_gap_requests: records
            .iter()
            .filter(|r| {
                r.visible_text
                    .as_ref()
                    .is_some_and(|text| !text.gaps_ms.is_empty())
            })
            .count(),
        usage_output_tokens: records.iter().map(|r| r.usage_output_tokens).collect(),
        server_input_tokens: values.server_inputs.clone(),
    }
}
