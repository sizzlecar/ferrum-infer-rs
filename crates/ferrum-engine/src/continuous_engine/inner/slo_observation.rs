//! Terminal internal timing observation, separate from client-visible SLOs.

use super::*;

#[derive(Debug, Serialize)]
pub(super) struct SloTerminalObservation {
    service_class: String,
    finish_reason: FinishReason,
    committed_tokens: u64,
    timing_trusted: bool,
    internal_timing_pass: Option<bool>,
    first_token_commit_ms: Option<f64>,
    last_token_commit_ms: Option<f64>,
    token_tpot_ms: Option<f64>,
    max_token_itl_ms: Option<f64>,
    first_engine_text_prepared_ms: Option<f64>,
    last_engine_text_prepared_ms: Option<f64>,
    engine_terminal_ms: Option<f64>,
    ttft_missed: bool,
    tpot_missed: bool,
    itl_missed: bool,
}

impl SloTerminalObservation {
    pub(super) fn capture(
        sequence: &SequenceState,
        reason: FinishReason,
        at: Instant,
    ) -> Option<Self> {
        let timing = sequence.slo.as_ref()?;
        let offset_ms = |instant: Instant| {
            instant
                .checked_duration_since(timing.ingress())
                .map(|duration| duration.as_secs_f64() * 1000.0)
        };
        let valid = timing.is_trusted()
            && usize::try_from(timing.committed_tokens()).ok()
                == Some(sequence.generated_tokens.len())
            && offset_ms(at).is_some();
        let completed = matches!(
            reason,
            FinishReason::EOS | FinishReason::Stop | FinishReason::Length
        );
        let violations = timing.violations();
        let token_tpot_ms = (timing.committed_tokens() >= 2)
            .then(|| {
                timing
                    .last_commit()?
                    .checked_duration_since(timing.first_commit()?)
                    .map(|duration| {
                        duration.as_secs_f64() * 1000.0 / (timing.committed_tokens() - 1) as f64
                    })
            })
            .flatten();
        Some(Self {
            service_class: timing.service_class().to_owned(),
            finish_reason: reason,
            committed_tokens: timing.committed_tokens(),
            timing_trusted: valid,
            internal_timing_pass: valid
                .then_some(completed && timing.committed_tokens() > 0 && !violations.any()),
            first_token_commit_ms: timing.first_commit().and_then(offset_ms),
            last_token_commit_ms: timing.last_commit().and_then(offset_ms),
            token_tpot_ms,
            max_token_itl_ms: timing
                .max_token_gap()
                .map(|duration| duration.as_secs_f64() * 1000.0),
            first_engine_text_prepared_ms: sequence.first_emit_at.and_then(offset_ms),
            last_engine_text_prepared_ms: sequence.last_emit_at.and_then(offset_ms),
            engine_terminal_ms: offset_ms(at),
            ttft_missed: violations.ttft,
            tpot_missed: violations.tpot,
            itl_missed: violations.itl,
        })
    }
}

impl EngineInner {
    /// Runs outside the global sequence map lock. No JSON or diagnostic history
    /// is constructed on each token; the default Off path has no observation.
    pub(super) fn record_slo_terminal(
        &self,
        request_id: &RequestId,
        observation: Option<SloTerminalObservation>,
    ) {
        let Some(observation) = observation else {
            return;
        };
        let status = match observation.internal_timing_pass {
            Some(true) => "pass",
            Some(false) => "fail",
            None => "unknown",
        };
        counter!("ferrum.engine.slo_terminal_requests_total",
            "service_class" => observation.service_class.clone(), "internal_timing" => status)
        .increment(1);
        if observation.timing_trusted {
            for (metric, value) in [
                (
                    "ferrum.engine.slo_token_ttft_seconds",
                    observation.first_token_commit_ms,
                ),
                (
                    "ferrum.engine.slo_token_tpot_seconds",
                    observation.token_tpot_ms,
                ),
                (
                    "ferrum.engine.slo_request_max_token_itl_seconds",
                    observation.max_token_itl_ms,
                ),
            ] {
                if let Some(ms) = value {
                    histogram!(metric, "service_class" => observation.service_class.clone())
                        .record(ms / 1000.0);
                }
            }
        }
        if self.scheduler_trace_jsonl.is_some() {
            self.write_executor_scheduler_profile_event(request_id, "engine_slo_terminal",
                ProfileEventKind::Instant, ProfileStatus::DiagnosticOnly, None,
                BTreeMap::from([("committed_tokens".to_owned(), serde_json::json!(observation.committed_tokens))]),
                BTreeMap::from([
                    ("timing_boundary".to_owned(), serde_json::json!("trusted_ingress_to_engine_token_commit")),
                    ("terminal_boundary".to_owned(), serde_json::json!("engine_sequence_removed_before_resource_cleanup_and_transport_terminal")),
                    ("text_boundary".to_owned(), serde_json::json!("engine_text_prepared_before_channel_send_not_http_or_client_visible")),
                    ("observation".to_owned(), serde_json::json!(observation)),
                ]), None);
        }
    }
}
