//! Client-visible latency limits, independent of serving policy and legacy goodput.
//! TPOT ends at the last visible text event. ITL pools all observed SSE gaps,
//! including coalesced updates and failed streams. No request-joint gate is used.

use crate::{
    repeat_percentiles, sse_text_event_gap, ItlEvidenceSource, OutputTokenCountSource,
    RepeatPercentiles, RequestRecord, RunRecord,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientSloConfig {
    pub ttft_p99_ms: f64,
    pub tpot_p99_ms: f64,
    pub visible_itl_p99_ms: f64,
}

impl ClientSloConfig {
    pub fn validate(self) -> Result<(), String> {
        if [self.ttft_p99_ms, self.tpot_p99_ms, self.visible_itl_p99_ms]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            Ok(())
        } else {
            Err("latency SLO limits must be positive finite milliseconds".into())
        }
    }
}

impl std::str::FromStr for ClientSloConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut limits = [None; 3];
        for field in value
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
        {
            let (key, value) = field
                .split_once(':')
                .ok_or("expected ttft:MS,tpot:MS,itl:MS")?;
            let index = match key {
                "ttft" => 0,
                "tpot" => 1,
                "itl" => 2,
                _ => return Err(format!("unknown latency SLO metric {key}")),
            };
            if limits[index]
                .replace(value.parse::<f64>().map_err(|e| e.to_string())?)
                .is_some()
            {
                return Err(format!("duplicate latency SLO metric {key}"));
            }
        }
        let config = Self {
            ttft_p99_ms: limits[0].ok_or("missing ttft")?,
            tpot_p99_ms: limits[1].ok_or("missing tpot")?,
            visible_itl_p99_ms: limits[2].ok_or("missing itl")?,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Missing observations are optional, never zero-filled.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestTiming {
    pub success: bool,
    pub first_visible_ms: Option<f64>,
    pub last_visible_ms: Option<f64>,
    pub terminal_ms: Option<f64>,
    pub usage_output_tokens: Option<u32>,
    pub event_source: ItlEvidenceSource,
    pub visible_text_events: u32,
    pub raw_event_gaps_ms: Vec<f64>,
    pub transport_coalesced_output_chunks: u32,
}

impl RequestTiming {
    pub fn from_record(record: &RequestRecord) -> Self {
        let evidence = &record.itl_evidence;
        let visible = evidence.source == ItlEvidenceSource::SseDeltaEvents
            && evidence.output_events > 0
            && record.ttft_ms.is_finite()
            && record.ttft_ms >= 0.0;
        let first = visible.then_some(record.ttft_ms);
        let complete = evidence.observed_intervals == evidence.output_events.saturating_sub(1)
            && evidence.observed_intervals as usize == record.itl_ms.len()
            && record
                .itl_ms
                .iter()
                .all(|gap| gap.is_finite() && *gap >= 0.0);
        // Consecutive gaps telescope to the last observed timestamp. This
        // preserves existing collectors and never includes a delayed terminal.
        let last = first
            .filter(|_| complete)
            .map(|first| first + record.itl_ms.iter().sum::<f64>())
            .filter(|last| last.is_finite());
        Self {
            success: record.success,
            first_visible_ms: first,
            last_visible_ms: last,
            terminal_ms: (record.quality_issues.panic == 0 && record.e2e_ms.is_finite())
                .then_some(record.e2e_ms),
            usage_output_tokens: (record.output_token_count_source
                == OutputTokenCountSource::Usage)
                .then_some(evidence.usage_output_tokens)
                .flatten(),
            event_source: evidence.source,
            visible_text_events: evidence.output_events,
            raw_event_gaps_ms: record.itl_ms.clone(),
            transport_coalesced_output_chunks: evidence.transport_coalesced_output_chunks,
        }
    }

    pub fn tpot_ms(&self) -> Option<f64> {
        let tokens = self.usage_output_tokens.filter(|tokens| *tokens > 1)?;
        Some((self.last_visible_ms? - self.first_visible_ms?) / f64::from(tokens - 1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientSloStatus {
    Pass,
    Fail,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientSloRepeat {
    pub repeat: u32,
    pub status: ClientSloStatus,
    pub offered_requests: u32,
    pub completed_requests: u32,
    /// Includes admission rejections and stream/protocol failures.
    pub errors: u32,
    /// HTTP 429 and 503 responses. Other HTTP errors remain in `errors`.
    pub rejected_requests: u32,
    pub duration_s: f64,
    pub ttft_ms: Option<RepeatPercentiles>,
    pub last_visible_tpot_ms: Option<RepeatPercentiles>,
    pub visible_sse_itl_ms: Option<RepeatPercentiles>,
    pub ttft_samples: usize,
    pub tpot_samples: usize,
    /// Successful responses with exactly one usage token have no TPOT.
    pub tpot_not_applicable_requests: u32,
    /// Missing/invalid usage or incomplete visible timing, never zero-filled.
    pub tpot_unknown_requests: u32,
    pub successful_usage_output_tokens: Option<u64>,
    pub successful_output_throughput_tps: Option<f64>,
    pub text_timing: Option<sse_text_event_gap::SseTextEventGapEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientSloReport {
    pub schema_version: u32,
    pub config: ClientSloConfig,
    pub status: ClientSloStatus,
    pub repeats: Vec<ClientSloRepeat>,
}

fn combine(statuses: impl IntoIterator<Item = ClientSloStatus>) -> ClientSloStatus {
    let statuses: Vec<_> = statuses.into_iter().collect();
    if statuses.contains(&ClientSloStatus::Fail) {
        ClientSloStatus::Fail
    } else if statuses.is_empty() || statuses.contains(&ClientSloStatus::Unknown) {
        ClientSloStatus::Unknown
    } else {
        ClientSloStatus::Pass
    }
}

impl ClientSloReport {
    pub fn evaluate(config: ClientSloConfig, runs: &[RunRecord]) -> Result<Self, String> {
        config.validate()?;
        let repeats = runs
            .iter()
            .enumerate()
            .map(|(index, run)| evaluate_repeat(config, index as u32, run))
            .collect::<Vec<_>>();
        Ok(Self {
            schema_version: 1,
            config,
            status: combine(repeats.iter().map(|r| r.status)),
            repeats,
        })
    }
}

fn evaluate_repeat(config: ClientSloConfig, repeat: u32, run: &RunRecord) -> ClientSloRepeat {
    let mut ttft = Vec::new();
    let mut tpot = Vec::new();
    let mut tpot_not_applicable_requests = 0;
    let mut tpot_unknown_requests = 0;
    let mut complete = run.records.len() == run.expected_requests as usize;
    let mut successful_tokens = Some(0_u64);
    for record in run.records.iter().filter(|r| r.success) {
        let timing = RequestTiming::from_record(record);
        if let Some(first) = timing.first_visible_ms {
            ttft.push(first);
        } else {
            complete = false;
        }
        match timing.usage_output_tokens {
            Some(tokens) if tokens > 0 && tokens == record.output_tokens => {
                successful_tokens =
                    successful_tokens.and_then(|total| total.checked_add(u64::from(tokens)));
                if tokens > 1 {
                    if let Some(value) = timing.tpot_ms() {
                        tpot.push(value);
                    } else {
                        complete = false;
                        tpot_unknown_requests += 1;
                    }
                } else {
                    tpot_not_applicable_requests += 1;
                }
            }
            _ => {
                successful_tokens = None;
                complete = false;
                tpot_unknown_requests += 1;
            }
        }
    }
    let percentile = |values: &[f64]| (!values.is_empty()).then(|| repeat_percentiles(values));
    let ttft_ms = percentile(&ttft);
    let last_visible_tpot_ms = percentile(&tpot);
    let (visible_sse_itl_ms, text_timing) = sse_text_event_gap::summarize_requests(&run.records);
    let throughput = successful_tokens
        .filter(|_| run.duration_s.is_finite() && run.duration_s > 0.0)
        .map(|tokens| tokens as f64 / run.duration_s);
    let errors = run.n_errored();
    let rejected_requests = run
        .records
        .iter()
        .filter(|r| r.quality_issues.http_rejected > 0)
        .count() as u32;
    let latency_statuses = [
        (ttft_ms, config.ttft_p99_ms),
        (last_visible_tpot_ms, config.tpot_p99_ms),
        (visible_sse_itl_ms, config.visible_itl_p99_ms),
    ]
    .map(|(metric, limit)| match metric {
        Some(metric) if metric.p99 > limit => ClientSloStatus::Fail,
        Some(_) => ClientSloStatus::Pass,
        None => ClientSloStatus::Unknown,
    });
    let status =
        combine(
            latency_statuses
                .into_iter()
                .chain([if errors > 0 || rejected_requests > 0 {
                    ClientSloStatus::Fail
                } else if !complete || throughput.is_none() {
                    ClientSloStatus::Unknown
                } else {
                    ClientSloStatus::Pass
                }]),
        );
    ClientSloRepeat {
        repeat,
        status,
        offered_requests: run.expected_requests,
        completed_requests: run.n_completed(),
        errors,
        rejected_requests,
        duration_s: run.duration_s,
        ttft_ms,
        last_visible_tpot_ms,
        visible_sse_itl_ms,
        ttft_samples: ttft.len(),
        tpot_samples: tpot.len(),
        tpot_not_applicable_requests,
        tpot_unknown_requests,
        successful_usage_output_tokens: successful_tokens,
        successful_output_throughput_tps: throughput,
        text_timing,
    }
}

#[cfg(test)]
mod tests;
