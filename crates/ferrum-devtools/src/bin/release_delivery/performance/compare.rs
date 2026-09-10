//! Compare controlled same-host standard HTTP measurements. The controller
//! binds binary/model identity, seed and effective capacity; BenchReport alone
//! does not observe those settings or prove which machine executed a request.
use ferrum_bench_core::{BenchReport, MetricSet, RepeatPercentiles, ScalarStats, Scenario};
use serde::{Deserialize, Serialize};

pub(super) use ferrum_bench_core::release_regression::performance::{Limits, Workload};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ComparisonStatus {
    Passed,
    Regressed,
    Inconclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct TimingChanges {
    pub ttft: ScalarStats,
    pub tpot: ScalarStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Comparison {
    pub status: ComparisonStatus,
    pub aa_relative: TimingChanges,
    pub candidate_relative: TimingChanges,
}

impl Comparison {
    pub(super) fn matches_recomputed(&self, expected: &Self) -> bool {
        // JSON and cross-platform floating-point replay can round a derived
        // statistic by a few ULPs. Use the same numerical precision check as
        // the raw report aggregates; policy decisions still must match exactly.
        self.status == expected.status
            && same_stats(self.aa_relative.ttft, expected.aa_relative.ttft)
            && same_stats(self.aa_relative.tpot, expected.aa_relative.tpot)
            && same_stats(
                self.candidate_relative.ttft,
                expected.candidate_relative.ttft,
            )
            && same_stats(
                self.candidate_relative.tpot,
                expected.candidate_relative.tpot,
            )
    }
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}
fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0)
}

fn same_stats(actual: ScalarStats, expected: ScalarStats) -> bool {
    actual.stddev >= 0.0
        && actual.ci95_hw >= 0.0
        && close(actual.mean, expected.mean)
        && close(actual.stddev, expected.stddev)
        && close(actual.ci95_hw, expected.ci95_hw)
}

fn validate_metric(
    metric: &MetricSet,
    rows: impl Iterator<Item = RepeatPercentiles>,
    required_positive: bool,
) -> Result<(), String> {
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for row in rows {
        let values = [row.p50, row.p75, row.p95, row.p99];
        if !values.iter().all(|v| {
            v.is_finite()
                && if required_positive {
                    *v > 0.0
                } else {
                    *v >= 0.0
                }
        }) || values.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err("invalid or unordered repeat latency percentiles".into());
        }
        for (samples, value) in samples.iter_mut().zip(values) {
            samples.push(value);
        }
    }
    for (actual, samples) in [metric.p50, metric.p75, metric.p95, metric.p99]
        .into_iter()
        .zip(samples)
    {
        if !same_stats(actual, ScalarStats::from_samples(&samples)) {
            return Err("aggregate latency does not match repeat measurements".into());
        }
    }
    Ok(())
}

fn validate_report(report: &BenchReport, workload: &Workload) -> Result<(), String> {
    if report.model != "release-perf"
        || report.backend != "metal"
        || report.scenario != Scenario::ClosedLoop
        || report.concurrency != Some(workload.concurrency)
        || report.request_rate.is_some()
        || report.n_prompt != workload.input_tokens
        || report.n_gen != workload.output_tokens
        || report.n_repeats != workload.repeats
        || report.n_requests_per_run != workload.measured_requests
        || report.warmup_requests != workload.warmup_requests
        || report.output_token_count_source.as_deref() != Some("usage")
    {
        return Err("benchmark identity, workload or usage-token source mismatch".into());
    }
    if report.env.hw_id.trim().is_empty() || report.env_hash != report.env.hash() {
        return Err("benchmark environment is missing or its digest is inconsistent".into());
    }
    let repeats = workload.repeats as usize;
    let requests = workload.measured_requests as usize;
    if report.repeat_metrics.len() != repeats
        || report.completed_per_run.len() != repeats
        || report.quality_issues_per_run.len() != repeats
    {
        return Err("missing repeat outcomes".into());
    }
    for values in [
        &report.errored_per_run,
        &report.bad_output_per_run,
        &report.malformed_stream_per_run,
        &report.missing_done_per_run,
        &report.duplicate_done_per_run,
        &report.zero_output_tokens_per_run,
        &report.http_500_per_run,
        &report.panic_per_run,
    ] {
        if values.len() != repeats || values.iter().any(|value| *value != 0) {
            return Err("missing or failed request/quality outcomes".into());
        }
    }
    if report.stream_bulk_flush_per_run.len() != repeats {
        return Err("missing stream diagnostic outcomes".into());
    }
    let inputs = report
        .actual_input_tokens_per_request
        .as_ref()
        .ok_or("missing per-request input lengths")?;
    let server_inputs = report
        .server_input_tokens_per_request
        .as_ref()
        .ok_or("missing per-request server prompt usage")?;
    let outputs = report
        .output_tokens_per_request
        .as_ref()
        .ok_or("missing per-request output lengths")?;
    let evidence = report
        .itl_evidence_per_request
        .as_ref()
        .ok_or("missing per-request usage observations")?;
    if inputs.len() != repeats
        || server_inputs.len() != repeats
        || outputs.len() != repeats
        || evidence.len() != repeats
    {
        return Err("token observation repeat count mismatch".into());
    }
    let expected_output = u64::from(workload.output_tokens) * u64::from(workload.measured_requests);
    for (index, repeat) in report.repeat_metrics.iter().enumerate() {
        if repeat.repeat != index as u32 + 1
            || repeat.expected_requests != workload.measured_requests
            || repeat.completed_requests != workload.measured_requests
            || repeat.errored_requests != 0
            || report.completed_per_run[index] != workload.measured_requests
            || repeat.warmup_expected != workload.warmup_requests
            || repeat.warmup_completed != workload.warmup_requests
            || repeat.warmup_errored != 0
            || repeat.quality_issues.request_error_count() != 0
            || repeat.warmup_quality_issues.request_error_count() != 0
            || report.quality_issues_per_run[index] != repeat.quality_issues
            || report.stream_bulk_flush_per_run[index] != repeat.quality_issues.stream_bulk_flush
            || repeat.output_token_count_source != "usage"
            || repeat.output_tokens != expected_output
            || repeat.actual_input_tokens == 0
            || repeat.actual_input_tokens
                > u64::from(workload.measured_requests)
                    * u64::from(workload.max_model_len - workload.output_tokens)
            || !positive(repeat.duration_s)
        {
            return Err(format!(
                "invalid measured/warmup outcomes for repeat {}",
                index + 1
            ));
        }
        if inputs[index].len() != requests
            || server_inputs[index].len() != requests
            || outputs[index].len() != requests
            || evidence[index].len() != requests
            || inputs[index].iter().any(|n| {
                *n == 0
                    || n.checked_add(workload.output_tokens)
                        .is_none_or(|total| total > workload.max_model_len)
            })
            || outputs[index].iter().any(|n| *n != workload.output_tokens)
            || server_inputs[index].iter().any(|n| {
                n.is_none_or(|tokens| {
                    tokens == 0
                        || tokens
                            .checked_add(workload.output_tokens)
                            .is_none_or(|total| total > workload.max_model_len)
                })
            })
            || evidence[index].iter().any(|e| {
                e.source != ferrum_bench_core::ItlEvidenceSource::SseDeltaEvents
                    || e.usage_output_tokens != Some(workload.output_tokens)
                    || e.output_events == 0
            })
        {
            return Err(
                "missing, invalid or budget-mismatched per-request token observations".into(),
            );
        }
        // Keep supplied content and the server's rendered prompt distinct.
        if repeat.actual_input_tokens != inputs[index].iter().map(|n| u64::from(*n)).sum::<u64>()
            || repeat.server_input_tokens
                != Some(
                    server_inputs[index]
                        .iter()
                        .flatten()
                        .map(|n| u64::from(*n))
                        .sum::<u64>(),
                )
        {
            return Err("input totals differ from the provided or server-observed lengths".into());
        }
        for (actual, expected) in [
            (
                repeat.output_throughput_tps,
                repeat.output_tokens as f64 / repeat.duration_s,
            ),
            (
                repeat.total_throughput_tps,
                (repeat.actual_input_tokens + repeat.output_tokens) as f64 / repeat.duration_s,
            ),
            (
                repeat.request_throughput_rps,
                f64::from(workload.measured_requests) / repeat.duration_s,
            ),
        ] {
            if !positive(actual) || !close(actual, expected) {
                return Err("repeat throughput is invalid or inconsistent".into());
            }
        }
        if !repeat.goodput_rps.is_finite()
            || repeat.goodput_rps < 0.0
            || repeat.goodput_rps > repeat.request_throughput_rps
        {
            return Err("invalid repeat goodput".into());
        }
    }
    let input_stats = report
        .actual_input_tokens
        .as_ref()
        .ok_or("missing input-length summary")?;
    let input_values: Vec<u32> = inputs.iter().flatten().copied().collect();
    let input_mean =
        input_values.iter().map(|n| f64::from(*n)).sum::<f64>() / input_values.len() as f64;
    if input_stats.requested != workload.input_tokens
        || input_stats.min != *input_values.iter().min().unwrap()
        || input_stats.max != *input_values.iter().max().unwrap()
        || !close(input_stats.mean, input_mean)
    {
        return Err("input-length summary does not match measured prompts".into());
    }
    validate_metric(
        &report.ttft_ms,
        report.repeat_metrics.iter().map(|r| r.ttft_ms),
        true,
    )?;
    validate_metric(
        &report.tpot_ms,
        report.repeat_metrics.iter().map(|r| r.tpot_ms),
        true,
    )?;
    validate_metric(
        &report.e2e_ms,
        report.repeat_metrics.iter().map(|r| r.e2e_ms),
        true,
    )?;
    // Coalesced SSE makes ITL ineligible. Zero ITL is valid here; this gate
    // compares only TTFT and usage-normalized TPOT, never token-event timing.
    // compute_metrics suppresses all aggregate ITL if any repeat is ineligible;
    // eligible individual repeats may still contain nonzero ITL. Do not compare
    // their aggregate or infer ITL eligibility from successful timing outcomes.
    for stats in [
        report.itl_ms.p50,
        report.itl_ms.p75,
        report.itl_ms.p95,
        report.itl_ms.p99,
    ] {
        if [stats.mean, stats.stddev, stats.ci95_hw]
            .into_iter()
            .any(|v| !v.is_finite() || v < 0.0)
        {
            return Err("invalid ITL diagnostic value".into());
        }
    }
    for (actual, values) in [
        (
            report.output_throughput_tps,
            report
                .repeat_metrics
                .iter()
                .map(|r| r.output_throughput_tps)
                .collect::<Vec<_>>(),
        ),
        (
            report.total_throughput_tps,
            report
                .repeat_metrics
                .iter()
                .map(|r| r.total_throughput_tps)
                .collect(),
        ),
        (
            report.request_throughput_rps,
            report
                .repeat_metrics
                .iter()
                .map(|r| r.request_throughput_rps)
                .collect(),
        ),
        (
            report.goodput_rps,
            report
                .repeat_metrics
                .iter()
                .map(|r| r.goodput_rps)
                .collect(),
        ),
    ] {
        if !same_stats(actual, ScalarStats::from_samples(&values)) {
            return Err("aggregate throughput does not match repeats".into());
        }
    }
    Ok(())
}

fn relative(before: &BenchReport, after: &BenchReport) -> Result<TimingChanges, String> {
    let metric =
        |pick: fn(&ferrum_bench_core::BenchRepeatMetrics) -> f64| -> Result<ScalarStats, String> {
            let values: Vec<f64> = before
                .repeat_metrics
                .iter()
                .zip(&after.repeat_metrics)
                .map(|(a, b)| pick(b) / pick(a) - 1.0)
                .collect();
            let stats = ScalarStats::from_samples(&values);
            if values.iter().any(|v| !v.is_finite())
                || !stats.mean.is_finite()
                || !stats.stddev.is_finite()
                || !stats.ci95_hw.is_finite()
                || !(stats.mean - stats.ci95_hw).is_finite()
                || !(stats.mean + stats.ci95_hw).is_finite()
            {
                return Err("relative timing interval is non-finite".into());
            }
            Ok(stats)
        };
    Ok(TimingChanges {
        ttft: metric(|r| r.ttft_ms.p50)?,
        tpot: metric(|r| r.tpot_ms.p50)?,
    })
}

fn validate_settings(workload: &Workload, limits: &Limits) -> Result<(), String> {
    workload.validate()?;
    if workload.repeats < 3
        || workload.input_tokens == 0
        || workload.output_tokens < 2
        || workload.measured_requests == 0
        || workload.warmup_requests == 0
        || workload
            .input_tokens
            .checked_add(workload.output_tokens)
            .is_none_or(|n| n >= workload.max_model_len)
        || !positive(limits.ttft_max_relative_increase)
        || !positive(limits.tpot_max_relative_increase)
    {
        return Err("invalid workload or predeclared performance limits".into());
    }
    Ok(())
}

/// Bench arrays share completion order, which changes under concurrency. Bind
/// content and rendered lengths to the original request index within each repeat.
fn ordered_inputs(report: &BenchReport) -> Result<Vec<Vec<(u32, Option<u32>)>>, String> {
    let content = report
        .actual_input_tokens_per_request
        .as_ref()
        .ok_or("missing content lengths")?;
    let rendered = report
        .server_input_tokens_per_request
        .as_ref()
        .ok_or("missing rendered lengths")?;
    let values: Vec<Vec<_>> = content
        .iter()
        .zip(rendered)
        .map(|(a, b)| a.iter().copied().zip(b.iter().copied()).collect())
        .collect();
    let Some(records) = &report.request_records else {
        if report.concurrency != Some(1) {
            return Err(
                "concurrent performance measurements require per-request correlation".into(),
            );
        }
        return Ok(values);
    };
    if records.len() != values.len() {
        return Err("request correlation repeat count differs from measured observations".into());
    }
    records
        .iter()
        .zip(values)
        .enumerate()
        .map(|(repeat, (records, values))| {
            if records.len() != values.len() {
                return Err("incomplete request correlation observations".into());
            }
            let mut ordered = vec![None; values.len()];
            for (record, value) in records.iter().zip(values) {
                let id = &record.correlation;
                if Some(&id.benchmark_run_id) != report.benchmark_run_id.as_ref()
                    || Some(&id.cell_id) != report.cell_id.as_ref()
                    || id.repeat_index as usize != repeat
                    || id.phase != ferrum_bench_core::BenchmarkPhase::Measured
                    || ferrum_bench_core::BenchmarkRequestCorrelation::new(
                        id.benchmark_run_id.clone(),
                        id.cell_id.clone(),
                        id.repeat_index,
                        id.phase,
                        id.request_index,
                    )
                    .is_err()
                {
                    return Err("request correlation differs from measured run/cell/repeat".into());
                }
                let slot = ordered
                    .get_mut(id.request_index as usize)
                    .ok_or("request correlation index exceeds measured workload")?;
                if slot.replace(value).is_some() {
                    return Err("duplicate measured request correlation index".into());
                }
            }
            ordered
                .into_iter()
                .map(|v| v.ok_or("missing measured request correlation index".into()))
                .collect()
        })
        .collect()
}

fn validate_pair(first: &BenchReport, second: &BenchReport) -> Result<(), String> {
    if second.env.hw_id != first.env.hw_id
        || second.env.driver != first.env.driver
        || second.env.cuda != first.env.cuda
        || second.env.http_connection_mode != first.env.http_connection_mode
        || second.env.gpu_clock_lock_mhz != first.env.gpu_clock_lock_mhz
        || second.env.gpu_power_limit_w != first.env.gpu_power_limit_w
        || second.env.gpu_persistence_mode != first.env.gpu_persistence_mode
        || second.env.gpu_auto_boost != first.env.gpu_auto_boost
        || second.env.ferrum_env != first.env.ferrum_env
        || serde_json::to_value(&second.env.runtime_config).map_err(|e| e.to_string())?
            != serde_json::to_value(&first.env.runtime_config).map_err(|e| e.to_string())?
        || ordered_inputs(second)? != ordered_inputs(first)?
        || second
            .repeat_metrics
            .iter()
            .map(|r| r.actual_input_tokens)
            .ne(first.repeat_metrics.iter().map(|r| r.actual_input_tokens))
    {
        return Err("hardware/connection configuration or actual input workload differs".into());
    }
    Ok(())
}

fn calibration_is_stable(changes: &TimingChanges, limits: &Limits) -> bool {
    [changes.ttft, changes.tpot]
        .into_iter()
        .zip([
            limits.ttft_max_relative_increase,
            limits.tpot_max_relative_increase,
        ])
        .all(|(s, limit)| s.mean - s.ci95_hw >= -limit && s.mean + s.ci95_hw <= limit)
}

pub(super) fn calibration(
    first: &BenchReport,
    second: &BenchReport,
    workload: &Workload,
    limits: &Limits,
) -> Result<bool, String> {
    validate_settings(workload, limits)?;
    validate_report(first, workload)?;
    validate_report(second, workload)?;
    validate_pair(first, second)?;
    Ok(calibration_is_stable(&relative(first, second)?, limits))
}

pub(super) fn compare(
    aa_first: &BenchReport,
    aa_second: &BenchReport,
    candidate: &BenchReport,
    workload: &Workload,
    limits: &Limits,
) -> Result<Comparison, String> {
    validate_settings(workload, limits)?;
    for report in [aa_first, aa_second, candidate] {
        validate_report(report, workload)?;
    }
    validate_pair(aa_first, aa_second)?;
    validate_pair(aa_first, candidate)?;
    let aa_relative = relative(aa_first, aa_second)?;
    let candidate_relative = relative(aa_second, candidate)?;
    let limits_array = [
        limits.ttft_max_relative_increase,
        limits.tpot_max_relative_increase,
    ];
    let noise_ok = calibration_is_stable(&aa_relative, limits);
    let status = if !noise_ok {
        ComparisonStatus::Inconclusive
    } else if [candidate_relative.ttft, candidate_relative.tpot]
        .into_iter()
        .zip(limits_array)
        .any(|(s, limit)| s.mean - s.ci95_hw > limit)
    {
        ComparisonStatus::Regressed
    } else if [candidate_relative.ttft, candidate_relative.tpot]
        .into_iter()
        .zip(limits_array)
        .all(|(s, limit)| s.mean + s.ci95_hw <= limit)
    {
        ComparisonStatus::Passed
    } else {
        ComparisonStatus::Inconclusive
    };
    Ok(Comparison {
        status,
        aa_relative,
        candidate_relative,
    })
}

#[cfg(test)]
pub(super) fn fixture_report(factors: [f64; 3]) -> BenchReport {
    tests::report(factors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_bench_core::{
        compute_metrics, Env, OutputTokenCountSource, QualityIssueCounts, RequestItlEvidence,
        RequestRecord, RunRecord, Slo, TokenLengthStats, WarmupSummary,
    };

    fn workload() -> Workload {
        Workload {
            input_tokens: 32,
            output_tokens: 8,
            measured_requests: 2,
            warmup_requests: 1,
            repeats: 3,
            seed: 7,
            max_model_len: 128,
            concurrency: 1,
            max_num_batched_tokens: None,
        }
    }
    fn limits() -> Limits {
        Limits {
            ttft_max_relative_increase: 0.1,
            tpot_max_relative_increase: 0.1,
        }
    }
    pub(super) fn report(factors: [f64; 3]) -> BenchReport {
        let w = workload();
        let runs = factors
            .into_iter()
            .map(|factor| RunRecord {
                records: (0..w.measured_requests)
                    .map(|_| RequestRecord {
                        benchmark_correlation: None,
                        server_request_id: None,
                        success: true,
                        ttft_ms: 10.0 * factor,
                        e2e_ms: 24.0 * factor,
                        input_tokens: w.input_tokens,
                        server_input_tokens: Some(w.input_tokens + 4),
                        output_tokens: w.output_tokens,
                        output_token_count_source: OutputTokenCountSource::Usage,
                        itl_evidence: RequestItlEvidence::sse(true, 4, Some(8), 3, 1),
                        quality_issues: QualityIssueCounts::default(),
                        itl_ms: vec![2.0 * factor; 3],
                    })
                    .collect(),
                expected_requests: w.measured_requests,
                duration_s: factor,
                warmup: WarmupSummary {
                    expected: w.warmup_requests,
                    completed: w.warmup_requests,
                    errored: 0,
                    quality_issues: QualityIssueCounts::default(),
                },
            })
            .collect();
        let mut env = Env::default();
        env.hw_id = "fixture-metal".into();
        env.http_connection_mode = Some("pooled".into());
        let mut report = compute_metrics(
            "release-perf".into(),
            "metal".into(),
            Scenario::ClosedLoop,
            Some(1),
            None,
            w.input_tokens,
            w.output_tokens,
            w.warmup_requests,
            Slo::unbounded(),
            runs,
            env,
        );
        report.actual_input_tokens = Some(TokenLengthStats {
            requested: w.input_tokens,
            min: w.input_tokens,
            max: w.input_tokens,
            mean: f64::from(w.input_tokens),
        });
        report.actual_input_tokens_per_request =
            Some(vec![
                vec![w.input_tokens; w.measured_requests as usize];
                w.repeats as usize
            ]);
        report.output_token_count_source = Some("usage".into());
        report
    }

    #[test]
    fn stable_measurements_pass_and_material_regression_fails() {
        let base = report([1.0; 3]);
        let passed = compare(&base, &base, &report([1.05; 3]), &workload(), &limits()).unwrap();
        assert_eq!(passed.status, ComparisonStatus::Passed);
        assert_eq!(
            compare(&base, &base, &report([1.2; 3]), &workload(), &limits())
                .unwrap()
                .status,
            ComparisonStatus::Regressed
        );
    }

    #[test]
    fn aa_noise_and_candidate_boundary_overlap_remain_inconclusive() {
        let base = report([1.0; 3]);
        let noisy = report([0.9, 1.0, 1.1]);
        assert_eq!(
            compare(&base, &noisy, &noisy, &workload(), &limits())
                .unwrap()
                .status,
            ComparisonStatus::Inconclusive
        );
        assert_eq!(
            compare(
                &base,
                &base,
                &report([1.0, 1.1, 1.2]),
                &workload(),
                &limits()
            )
            .unwrap()
            .status,
            ComparisonStatus::Inconclusive
        );
    }

    #[test]
    fn incomplete_failed_and_forged_token_evidence_cannot_pass() {
        let base = report([1.0; 3]);
        let mutations: Vec<Box<dyn Fn(&mut BenchReport)>> = vec![
            Box::new(|r| {
                r.repeat_metrics.pop();
            }),
            Box::new(|r| r.errored_per_run[0] = 1),
            Box::new(|r| r.repeat_metrics[0].warmup_completed = 0),
            Box::new(|r| r.bad_output_per_run.clear()),
            Box::new(|r| r.output_tokens_per_request.as_mut().unwrap()[0][0] = 7),
            Box::new(|r| {
                r.itl_evidence_per_request.as_mut().unwrap()[0][0].usage_output_tokens = None
            }),
            Box::new(|r| r.repeat_metrics[0].output_token_count_source = "stream_chunks".into()),
            Box::new(|r| r.repeat_metrics[0].ttft_ms.p50 = f64::NAN),
            Box::new(|r| r.repeat_metrics[0].actual_input_tokens = u64::MAX),
            Box::new(|r| r.server_input_tokens_per_request = None),
            Box::new(|r| r.server_input_tokens_per_request.as_mut().unwrap()[0][0] = None),
            Box::new(|r| {
                r.server_input_tokens_per_request.as_mut().unwrap()[0][0] =
                    Some(workload().max_model_len)
            }),
            Box::new(|r| r.repeat_metrics[0].server_input_tokens = Some(1)),
            Box::new(|r| {
                *r.server_input_tokens_per_request.as_mut().unwrap()[0][0]
                    .as_mut()
                    .unwrap() += 1;
                *r.repeat_metrics[0].server_input_tokens.as_mut().unwrap() += 1;
            }),
            Box::new(|r| r.ttft_ms.p50.mean = 0.1),
            Box::new(|r| r.actual_input_tokens_per_request.as_mut().unwrap()[0][0] = 31),
            Box::new(|r| r.backend = "cpu".into()),
            Box::new(|r| {
                r.env.hw_id = "different-hardware".into();
                r.env_hash = r.env.hash();
            }),
        ];
        for change in mutations {
            let mut candidate = base.clone();
            change(&mut candidate);
            assert!(compare(&base, &base, &candidate, &workload(), &limits()).is_err());
        }
    }

    #[test]
    fn calibration_can_reject_noise_before_candidate_execution() {
        let first = report([1.0; 3]);
        assert!(calibration(&first, &first, &workload(), &limits()).unwrap());
        assert!(!calibration(&first, &report([0.9, 1.0, 1.1]), &workload(), &limits()).unwrap());
        let mut failed = first.clone();
        failed.repeat_metrics[0].warmup_completed = 0;
        assert!(calibration(&first, &failed, &workload(), &limits()).is_err());
        let mut w = workload();
        w.warmup_requests = 0;
        assert!(calibration(&first, &first, &w, &limits()).is_err());
        let mut w = workload();
        w.max_model_len = w.input_tokens + w.output_tokens;
        assert!(calibration(&first, &first, &w, &limits()).is_err());
    }

    #[test]
    fn invalid_limits_and_insufficient_repeats_are_rejected() {
        let base = report([1.0; 3]);
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut policy = limits();
            policy.ttft_max_relative_increase = value;
            assert!(compare(&base, &base, &base, &workload(), &policy).is_err());
        }
        let mut w = workload();
        w.repeats = 2;
        assert!(compare(&base, &base, &base, &w, &limits()).is_err());
    }
}

#[cfg(test)]
#[path = "concurrency_tests.rs"]
mod concurrency_tests;
