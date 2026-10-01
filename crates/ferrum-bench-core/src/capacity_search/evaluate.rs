use super::types::*;
use crate::slo::{evaluate_slo, RequestOutcome, SloStatus};
use crate::{BenchmarkPhase, ItlEvidenceSource};
use sha2::{Digest, Sha256};

pub fn evaluate_run(
    contract: &CapacityContract,
    planned: &PlannedCapacityRun,
    evidence: &CapacityRunEvidence,
) -> Result<CapacityRunAssessment, CapacityError> {
    if !evidence.warmup.acquisition.is_executed() {
        return Err(error(
            "warmup reuse requires validation against original chronological capacity history",
        ));
    }
    evaluate_run_verified(contract, planned, evidence, false)
}

pub(super) fn evaluate_run_verified(
    contract: &CapacityContract,
    planned: &PlannedCapacityRun,
    evidence: &CapacityRunEvidence,
    warmup_reuse_verified: bool,
) -> Result<CapacityRunAssessment, CapacityError> {
    contract.validate()?;
    let digest = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(contract).map_err(|e| error(e.to_string()))?)
    );
    if planned != &super::plan::generate_run(contract, &digest, &planned.key)? {
        return Err(error(
            "planned arrivals or output budgets differ from the frozen contract",
        ));
    }
    if planned.contract_sha256 != evidence.contract_sha256
        || planned.key != evidence.key
        || contract.identity != evidence.identity
        || contract.queue_observation_source != evidence.queue_observation_source
    {
        return Err(error("run identity, frozen contract or phase differs"));
    }
    if evidence.run_started_unix_ns < contract.frozen_unix_ns
        || evidence.run_ended_unix_ns <= evidence.run_started_unix_ns
    {
        return Err(error(
            "run must follow contract freeze with an ordered lifecycle",
        ));
    }
    let count = planned.scheduled_arrival_ms.len();
    let mut result = CapacityRunAssessment {
        acquisition_disposition: CapacityAcquisitionDisposition::IncompleteEvidence,
        key: planned.key.clone(),
        status: SloStatus::Unknown,
        evidence_complete: true,
        workload_completed: true,
        arrival_schedule_delivered: true,
        issues: Vec::new(),
        evaluation: None,
        requested_rate_rps: planned.rate_rps,
        planned_requests: count,
        delivered_requests: 0,
        realized_scheduled_rate_rps: count as f64 / planned.send_seconds,
        realized_delivered_rate_rps: 0.0,
        maximum_start_lag_ms: None,
        maximum_client_backlog: None,
        unfinished_requests_slope_per_second: None,
        oldest_age_slope_ms_per_second: None,
        waiting_requests_slope_per_second: None,
        oldest_waiting_ingress_age_slope_ms_per_second: None,
        drain_seconds: None,
    };
    let mut failed = false;
    let expected_warmup = contract
        .workload
        .samples
        .iter()
        .filter(|s| s.phase == BenchmarkPhase::Warmup)
        .count();
    if !warmup_reuse_verified
        && (evidence.warmup.expected as usize != expected_warmup
            || evidence.warmup.completed != evidence.warmup.expected
            || evidence.warmup.errored != 0
            || evidence.warmup.quality.request_error_count() > 0)
    {
        failed = true;
        result.workload_completed = false;
        result
            .issues
            .push("warmup differs from frozen workload or did not complete successfully".into());
    }
    if !evidence.measured_duration_seconds.is_finite()
        || evidence.measured_duration_seconds < planned.send_seconds
        || evidence.send_window_seconds != planned.send_seconds
        || evidence.measured_duration_seconds
            > (evidence.run_ended_unix_ns - evidence.run_started_unix_ns) as f64 / 1_000_000_000.0
    {
        missing(&mut result, "missing or inconsistent send/drain window");
    } else {
        let drain = evidence.measured_duration_seconds - planned.send_seconds;
        result.drain_seconds = Some(drain);
        if drain > contract.window.maximum_drain_seconds {
            failed = true;
            result.issues.push("drain exceeds declared limit".into());
        }
    }
    if evidence.requests.len() != count
        || evidence.arrivals.len() != count
        || evidence.quality.len() != count
        || evidence.request_records.len() != count
    {
        missing(
            &mut result,
            "missing/extra offered request, arrival or quality rows",
        );
        result.workload_completed = false;
        result.arrival_schedule_delivered = false;
    }
    let mut lags = Vec::new();
    let mut backlogs = Vec::new();
    for (index, arrival) in evidence.arrivals.iter().enumerate() {
        let Some((&target, (scheduled, (dispatched, (started, backlog))))) =
            planned.scheduled_arrival_ms.get(index).zip(
                arrival.scheduled_arrival_ms.zip(
                    arrival.dispatched_ms.zip(
                        arrival
                            .request_started_ms
                            .zip(arrival.client_dispatch_backlog),
                    ),
                ),
            )
        else {
            result.arrival_schedule_delivered = false;
            missing(
                &mut result,
                "incomplete planned/dispatch/start/backlog evidence",
            );
            continue;
        };
        if [scheduled, dispatched, started]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            || scheduled != target
            || dispatched < scheduled
            || started < dispatched
        {
            result.arrival_schedule_delivered = false;
            missing(
                &mut result,
                "arrival differs from offered schedule or has invalid timing",
            );
            continue;
        }
        // Includes this due request, matching bench-serve's existing collector.
        let due = planned
            .scheduled_arrival_ms
            .partition_point(|t| *t <= dispatched)
            .saturating_sub(index) as u64;
        if backlog != due {
            result.arrival_schedule_delivered = false;
            missing(
                &mut result,
                "client backlog disagrees with planned schedule",
            );
        }
        result.delivered_requests += 1;
        lags.push(started - target);
        backlogs.push(backlog);
        if started > planned.send_seconds * 1000.0 + contract.window.maximum_request_start_lag_ms {
            failed = true;
            result.arrival_schedule_delivered = false;
            result
                .issues
                .push("request start falls outside allowed send window".into());
        }
        if let Some(terminal) = evidence.requests.get(index).and_then(|r| r.terminal_ms) {
            if started + terminal > evidence.measured_duration_seconds * 1000.0 {
                missing(&mut result, "terminal falls outside measured drain");
            }
        }
    }
    result.realized_delivered_rate_rps = result.delivered_requests as f64 / planned.send_seconds;
    result.maximum_start_lag_ms = lags.into_iter().reduce(f64::max);
    result.maximum_client_backlog = backlogs.into_iter().max();
    if result
        .maximum_start_lag_ms
        .is_some_and(|lag| lag > contract.window.maximum_request_start_lag_ms)
        || result
            .maximum_client_backlog
            .is_some_and(|b| b > contract.window.maximum_client_dispatch_backlog)
    {
        failed = true;
        result.arrival_schedule_delivered = false;
        result.issues.push(
            "client did not deliver the prescribed arrival schedule within its explicit tolerances"
                .into(),
        );
    }
    for (index, request) in evidence.requests.iter().enumerate() {
        let aligned = evidence
            .request_records
            .get(index)
            .zip(planned.workload_sample_indices.get(index))
            .is_some_and(|(observed, &sample_index)| {
                let sample = &contract.workload.samples[sample_index];
                let record = &observed.record;
                let correlation = &record.correlation;
                let input_matches = observed.workload_sample_index == sample_index
                    && observed.dispatched_prompt_sha256 == sample.prompt_sha256
                    && observed.input_tokens == sample.input_tokens
                    && observed.server_input_tokens.is_some_and(|n| n > 0);
                let correlation_matches = correlation.benchmark_run_id == evidence.benchmark_run_id
                    && !evidence.benchmark_run_id.is_empty()
                    && correlation.cell_id == planned.cell_id
                    && correlation.repeat_index == planned.key.repetition
                    && correlation.phase == BenchmarkPhase::Measured
                    && correlation.request_index as usize == index;
                let timing_matches = record
                    .timing
                    .as_ref()
                    .zip(request.visible_text.as_ref())
                    .is_some_and(|(timing, text)| {
                        timing.success == (request.outcome == RequestOutcome::Completed)
                            && timing.event_source == ItlEvidenceSource::SseDeltaEvents
                            && timing.observed_first_output == Some(true)
                            && request
                                .first_visible_ms
                                .is_some_and(|v| close(v, timing.reported_ttft_ms))
                            && request
                                .terminal_ms
                                .is_some_and(|v| close(v, timing.reported_e2e_ms))
                            && timing.raw_event_gaps_ms.len() == text.gaps_ms.len()
                            && timing
                                .raw_event_gaps_ms
                                .iter()
                                .zip(&text.gaps_ms)
                                .all(|(&a, &b)| close(a, b))
                    });
                input_matches && correlation_matches && timing_matches
            });
        let last_matches = request
            .first_visible_ms
            .zip(request.last_visible_ms)
            .zip(request.visible_text.as_ref())
            .is_some_and(|((first, last), text)| {
                close(first + text.gaps_ms.iter().sum::<f64>(), last)
            });
        if !aligned || !last_matches {
            result.workload_completed = false;
            missing(&mut result,format!("request {index} workload/correlation/raw timing or last-visible boundary mismatch"));
        }
        if request.outcome != RequestOutcome::Completed {
            failed = true;
            result.workload_completed = false;
            result.issues.push(format!(
                "request {index} failed, was rejected or remains pending"
            ));
        }
        match request
            .usage_output_tokens
            .zip(planned.output_token_budgets.get(index).copied())
        {
            Some((actual, expected)) if actual == expected => {}
            Some(_) => {
                failed = true;
                result.workload_completed = false;
                result.issues.push(format!(
                    "request {index} output does not complete its original budget"
                ));
            }
            None => {
                result.workload_completed = false;
                missing(&mut result, "missing original output budget or usage");
            }
        }
        if request.terminal_ms.is_none() {
            result.workload_completed = false;
            missing(&mut result, "missing terminal completion evidence");
        }
    }
    if evidence.quality.iter().any(|q| q.request_error_count() > 0) {
        failed = true;
        result.workload_completed = false;
        result
            .issues
            .push("output/protocol quality failures".into());
    }
    match evaluate_slo(
        &contract.slo,
        &evidence.requests,
        evidence.measured_duration_seconds,
    ) {
        Ok(report) => {
            if report.latency_and_outcome_status == SloStatus::Fail {
                failed = true;
            } else if report.latency_and_outcome_status != SloStatus::Pass {
                missing(&mut result, "SLO evidence does not establish attainment");
            }
            result.evaluation = Some(report);
        }
        Err(e) => missing(&mut result, format!("invalid raw SLO evidence: {e}")),
    }
    evaluate_queue(contract, planned, evidence, &mut result, &mut failed);
    result.status = if failed {
        SloStatus::Fail
    } else if result.evidence_complete {
        SloStatus::Pass
    } else {
        SloStatus::Unknown
    };
    result.acquisition_disposition =
        super::acquisition::classify(contract, planned, evidence, &result);
    Ok(result)
}

pub(super) fn missing(result: &mut CapacityRunAssessment, message: impl Into<String>) {
    result.evidence_complete = false;
    result.issues.push(message.into());
}

fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0)
}

pub(super) fn slope(
    samples: &[&ServiceQueueSample],
    value: impl Fn(&ServiceQueueSample) -> f64,
) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let n = samples.len() as f64;
    let x = samples.iter().map(|s| s.at_seconds).sum::<f64>() / n;
    let y = samples.iter().map(|s| value(s)).sum::<f64>() / n;
    let denom = samples
        .iter()
        .map(|s| (s.at_seconds - x).powi(2))
        .sum::<f64>();
    (denom > 0.0)
        .then(|| {
            samples
                .iter()
                .map(|s| (s.at_seconds - x) * (value(s) - y))
                .sum::<f64>()
                / denom
        })
        .filter(|v| v.is_finite())
}

fn evaluate_queue(
    contract: &CapacityContract,
    planned: &PlannedCapacityRun,
    evidence: &CapacityRunEvidence,
    result: &mut CapacityRunAssessment,
    failed: &mut bool,
) {
    if evidence.queue_observation_source == QueueObservationSource::ServerAdmissionV1 {
        super::server_queue::evaluate(contract, evidence, result, failed);
        return;
    }
    let q = &evidence.queue;
    let w = &contract.window;
    if !evidence.queue_capture_complete || q.len() > contract.maximum_queue_samples_per_run {
        missing(
            result,
            "queue capture was incomplete or exceeded its declared resource bound",
        );
    }
    if q.len() < 2
        || q.iter().any(|s| {
            !s.at_seconds.is_finite()
                || s.at_seconds < 0.0
                || !s.oldest_request_age_ms.is_finite()
                || s.oldest_request_age_ms < 0.0
                || (s.waiting_requests == 0
                    && s.active_requests == 0
                    && s.oldest_request_age_ms != 0.0)
        })
        || q.windows(2).any(|p| p[1].at_seconds <= p[0].at_seconds)
    {
        missing(result, "missing or malformed server queue time series");
        return;
    }
    if q[0].at_seconds > w.observe_from_seconds
        || q.last().unwrap().at_seconds < evidence.measured_duration_seconds
        || q.last().unwrap().at_seconds
            > evidence.measured_duration_seconds + w.maximum_queue_sample_gap_seconds
        || q.windows(2)
            .any(|p| p[1].at_seconds - p[0].at_seconds > w.maximum_queue_sample_gap_seconds)
    {
        missing(
            result,
            "server queue samples do not cover the observation and drain window",
        );
    }
    if evidence.queue_observation_source == QueueObservationSource::ClientScheduledLifecycle {
        for sample in q {
            let now_ms = sample.at_seconds * 1000.0;
            let mut waiting = 0_u64;
            let mut active = 0_u64;
            let mut oldest = 0.0_f64;
            for (index, &scheduled) in planned
                .scheduled_arrival_ms
                .iter()
                .enumerate()
                .take_while(|(_, s)| **s <= now_ms)
            {
                let arrival = evidence.arrivals.get(index);
                let ended = arrival
                    .and_then(|a| a.request_started_ms)
                    .zip(evidence.requests.get(index).and_then(|r| r.terminal_ms))
                    .is_some_and(|(started, elapsed)| started + elapsed <= now_ms);
                if ended {
                    continue;
                }
                oldest = oldest.max(now_ms - scheduled);
                if arrival
                    .and_then(|a| a.dispatched_ms)
                    .is_some_and(|t| t <= now_ms)
                {
                    active += 1;
                } else {
                    waiting += 1;
                }
            }
            if sample.waiting_requests != waiting
                || sample.active_requests != active
                || !close(sample.oldest_request_age_ms, oldest)
            {
                missing(result,"client lifecycle queue sample disagrees with actual arrival/completion evidence");
                break;
            }
        }
    }
    let samples: Vec<_> = q
        .iter()
        .filter(|s| s.at_seconds >= w.observe_from_seconds && s.at_seconds <= w.send_seconds)
        .collect();
    result.unfinished_requests_slope_per_second = slope(&samples, |s| {
        s.waiting_requests as f64 + s.active_requests as f64
    });
    result.oldest_age_slope_ms_per_second = slope(&samples, |s| s.oldest_request_age_ms);
    if result.unfinished_requests_slope_per_second.is_none()
        || result.oldest_age_slope_ms_per_second.is_none()
    {
        missing(
            result,
            "insufficient queue samples inside the observation window",
        );
    }
    if result
        .unfinished_requests_slope_per_second
        .is_some_and(|v| v > w.maximum_unfinished_requests_slope_per_second)
        || result
            .oldest_age_slope_ms_per_second
            .is_some_and(|v| v > w.maximum_oldest_age_slope_ms_per_second)
    {
        *failed = true;
        result.issues.push(
            "observed service queue or oldest age grows beyond its declared tolerance".into(),
        );
    }
    let last = q.last().unwrap();
    if last.waiting_requests != 0 || last.active_requests != 0 {
        *failed = true;
        result
            .issues
            .push("unfinished server work remains after drain".into());
    }
}
