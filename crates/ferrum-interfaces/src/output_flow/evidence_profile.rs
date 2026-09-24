//! Borrowed diagnostic serialization. No token/stage arrays or text histories
//! are cloned into JSON Values. An owning journal must keep this record alive
//! through writing; its Arc retains the original projection lease.
use super::{OutputCompletion, OutputFlowError};
use crate::output_credit::LeasedOutput;
use ferrum_types::{
    FerrumProfileEvent, ObservabilityProfileDetail, ProfileEntrypoint, ProfileError,
    ProfileEventKind, ProfileStatus, OBSERVABILITY_PROFILE_SCHEMA_VERSION,
};
use serde::{
    ser::{SerializeMap, SerializeSeq, SerializeStruct},
    Serialize, Serializer,
};
use std::{collections::BTreeMap, sync::Arc};

pub struct CreditedExecutionProfile {
    completion: Arc<LeasedOutput<OutputCompletion>>,
    metadata: FerrumProfileEvent,
}
impl CreditedExecutionProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        completion: Arc<LeasedOutput<OutputCompletion>>,
        entrypoint: ProfileEntrypoint,
        model: String,
        endpoint: &'static str,
        detail: ObservabilityProfileDetail,
        duration_us: u64,
        runtime_preset_hash: String,
        mut attributes: BTreeMap<String, serde_json::Value>,
    ) -> Result<Self, OutputFlowError> {
        // Caller metadata may add correlation, never override measured fields.
        if attributes.keys().any(|key| {
            key.starts_with("engine_")
                || key.starts_with("clock_")
                || key.starts_with("itl_")
                || key.starts_with("decode_")
                || matches!(
                    key.as_str(),
                    "ttft_us"
                        | "prompt_token_count"
                        | "completion_token_count"
                        | "output_token_count"
                        | "total_token_count"
                        | "token_count_source"
                        | "finish_reason"
                        | "endpoint"
                        | "profile_detail"
                        | "diagnostic_only"
                        | "output_transport"
                        | "actual_model_smoke"
                        | "execution_request_id"
                        | "http_stream_flush_unavailable_reason"
                )
        }) {
            return Err(OutputFlowError::BoundExceeded);
        }
        let request_id = completion.request_id().to_string();
        let (status, error) = match completion.payload() {
            OutputCompletion::Succeeded {
                execution_evidence,
                usage,
                ..
            } => {
                let timing = execution_evidence
                    .as_ref()
                    .and_then(|e| e.engine_token_timing.as_ref());
                if detail.captures_engine_token_timing() && timing.is_none() {
                    return Err(OutputFlowError::BoundExceeded);
                }
                if let Some(timing) = timing {
                    timing
                        .validate(usage.completion_tokens)
                        .map_err(|_| OutputFlowError::BoundExceeded)?;
                }
                (ProfileStatus::Ok, None)
            }
            OutputCompletion::Failed(error) => (
                ProfileStatus::Failure,
                Some(ProfileError {
                    kind: "credited_execution".into(),
                    message: error.message().into(),
                    blocking: false,
                }),
            ),
        };
        attributes.extend([
            ("endpoint".into(), serde_json::json!(endpoint)),
            ("profile_detail".into(), serde_json::json!(detail.as_str())),
            (
                "diagnostic_only".into(),
                serde_json::json!(detail.diagnostic_only()),
            ),
            ("output_transport".into(), serde_json::json!("credited")),
            ("actual_model_smoke".into(), serde_json::json!(true)),
            (
                "execution_request_id".into(),
                serde_json::json!(format!("request.product.{request_id}")),
            ),
            (
                "http_stream_flush_unavailable_reason".into(),
                serde_json::json!("terminal engine evidence does not observe socket flush"),
            ),
        ]);
        let timestamp = chrono::Utc::now();
        let metadata = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp
                .timestamp_nanos_opt()
                .unwrap_or_else(|| timestamp.timestamp_micros() * 1000),
            event_id: format!("evt-credited-{request_id}"),
            correlation_id: Some(request_id.clone()),
            request_id,
            entrypoint,
            backend: "actual".into(),
            runtime_preset_hash,
            phase: "credited_generation".into(),
            event_kind: ProfileEventKind::TimedSpan,
            timestamp,
            status,
            model: Some(model),
            duration_us: Some(duration_us),
            memory: None,
            resource: None,
            error,
            replay: None,
            shape: BTreeMap::from([("batch_size".into(), serde_json::json!(1))]),
            backend_detail: None,
            attributes,
        };
        metadata
            .validate()
            .map_err(|_| OutputFlowError::BoundExceeded)?;
        Ok(Self {
            completion,
            metadata,
        })
    }
}
impl Serialize for CreditedExecutionProfile {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let m = &self.metadata;
        let mut s = serializer.serialize_struct("FerrumProfileEvent", 21)?;
        macro_rules! field { ($($f:ident),*) => { $(s.serialize_field(stringify!($f),&m.$f)?;)* }; }
        field!(
            schema_version,
            ts_unix_nanos,
            event_id,
            request_id,
            correlation_id,
            entrypoint,
            backend,
            runtime_preset_hash,
            phase,
            event_kind,
            timestamp,
            status,
            model,
            duration_us,
            memory,
            resource,
            error,
            replay,
            shape,
            backend_detail
        );
        s.serialize_field(
            "attributes",
            &Attributes {
                metadata: &m.attributes,
                outcome: self.completion.payload(),
            },
        )?;
        s.end()
    }
}
struct Attributes<'a> {
    metadata: &'a BTreeMap<String, serde_json::Value>,
    outcome: &'a OutputCompletion,
}
struct Gaps<'a>(&'a [u64]);
impl Serialize for Gaps<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len().saturating_sub(1)))?;
        for pair in self.0.windows(2) {
            seq.serialize_element(&pair[1].saturating_sub(pair[0]))?;
        }
        seq.end()
    }
}
impl Serialize for Attributes<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(None)?;
        for (key, value) in self.metadata {
            m.serialize_entry(key, value)?;
        }
        if let OutputCompletion::Succeeded {
            usage,
            reason,
            execution_evidence,
            ..
        } = self.outcome
        {
            m.serialize_entry("prompt_token_count", &usage.prompt_tokens)?;
            m.serialize_entry("completion_token_count", &usage.completion_tokens)?;
            m.serialize_entry("output_token_count", &usage.completion_tokens)?;
            m.serialize_entry("total_token_count", &usage.total_tokens)?;
            m.serialize_entry("token_count_source", "usage")?;
            let reason = match reason {
                ferrum_types::FinishReason::Length => "length",
                ferrum_types::FinishReason::Stop => "stop",
                ferrum_types::FinishReason::EOS => "eos",
                ferrum_types::FinishReason::Cancelled => "cancelled",
                ferrum_types::FinishReason::Error => "error",
                ferrum_types::FinishReason::ContentFilter => "content_filter",
            };
            m.serialize_entry("finish_reason", reason)?;
            if let Some(t) = execution_evidence
                .as_ref()
                .and_then(|e| e.engine_token_timing.as_ref())
            {
                m.serialize_entry("engine_token_clock_source", &t.clock_source)?;
                m.serialize_entry(
                    "engine_token_wall_anchor_unix_nanos",
                    &t.wall_anchor_unix_nanos,
                )?;
                m.serialize_entry(
                    "clock_conversion_max_error_nanos",
                    &t.wall_anchor_max_error_nanos,
                )?;
                m.serialize_entry(
                    "engine_token_commit_nanos_since_request_start",
                    &t.token_commit_nanos_since_request_start,
                )?;
                m.serialize_entry(
                    "engine_token_commit_count",
                    &t.token_commit_nanos_since_request_start.len(),
                )?;
                m.serialize_entry("engine_decode_stage_intervals", &t.decode_stage_intervals)?;
                m.serialize_entry(
                    "engine_decode_stage_interval_count",
                    &t.decode_stage_intervals.len(),
                )?;
                m.serialize_entry(
                    "engine_decode_stage_intervals_omitted",
                    &t.decode_stage_intervals_omitted,
                )?;
                m.serialize_entry(
                    "engine_decode_stages_complete",
                    &(t.decode_stage_intervals_omitted == 0),
                )?;
                m.serialize_entry("itl_source", "engine_token_commit")?;
                m.serialize_entry(
                    "itl_interval_count",
                    &t.token_commit_nanos_since_request_start
                        .len()
                        .saturating_sub(1),
                )?;
                m.serialize_entry(
                    "itl_nanos",
                    &Gaps(&t.token_commit_nanos_since_request_start),
                )?;
                let n = t
                    .token_commit_nanos_since_request_start
                    .len()
                    .saturating_sub(1);
                let sum = t
                    .token_commit_nanos_since_request_start
                    .windows(2)
                    .fold(0u128, |v, p| v + u128::from(p[1].saturating_sub(p[0])));
                let avg = if n == 0 {
                    0
                } else {
                    u64::try_from(sum / n as u128).unwrap_or(u64::MAX)
                };
                m.serialize_entry("itl_us_avg", &(avg / 1000))?;
                if let Some(v) = t.ttft_nanos() {
                    m.serialize_entry("ttft_us", &(v / 1000))?;
                }
                if let Some(v) = t.decode_ready_nanos_since_request_start {
                    m.serialize_entry("engine_decode_ready_nanos_since_request_start", &v)?;
                }
                if let Some(v) = t.decode_wall_nanos().filter(|v| *v > 0) {
                    m.serialize_entry("engine_decode_wall_nanos", &v)?;
                    let ppm = u64::try_from(
                        u128::from(t.wall_anchor_max_error_nanos) * 1_000_000 / u128::from(v),
                    )
                    .unwrap_or(u64::MAX);
                    m.serialize_entry("clock_conversion_error_ppm", &ppm)?;
                    m.serialize_entry("decode_wall_timing_eligible", &true)?;
                } else {
                    m.serialize_entry("decode_wall_timing_eligible", &false)?;
                    let reason = if t.token_commit_nanos_since_request_start.is_empty() {
                        "no_token_commits"
                    } else if t.decode_ready_nanos_since_request_start.is_none() {
                        "decode_not_entered"
                    } else {
                        "no_positive_decode_commit_interval"
                    };
                    m.serialize_entry("decode_wall_timing_unavailable_reason", reason)?;
                }
            }
        }
        m.end()
    }
}
/// A separately written replay token artifact sharing exactly the same lease.
pub struct CreditedPromptEvidence {
    pub completion: Arc<LeasedOutput<OutputCompletion>>,
    pub model: String,
}
impl Serialize for CreditedPromptEvidence {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let evidence = match self.completion.payload() {
            OutputCompletion::Succeeded {
                execution_evidence: Some(e),
                ..
            } => e,
            _ => {
                return Err(serde::ser::Error::custom(
                    "successful engine evidence unavailable",
                ))
            }
        };
        let mut m = s.serialize_map(Some(8))?;
        m.serialize_entry("schema_version", &OBSERVABILITY_PROFILE_SCHEMA_VERSION)?;
        m.serialize_entry("request_id", &self.completion.request_id().to_string())?;
        m.serialize_entry("model", &self.model)?;
        m.serialize_entry("tokenizer_or_model", &self.model)?;
        m.serialize_entry("token_ids", &evidence.prompt_token_ids)?;
        m.serialize_entry("token_count", &evidence.prompt_token_ids.len())?;
        m.serialize_entry("unavailable_reason", &Option::<&str>::None)?;
        m.serialize_entry("sanitized", &true)?;
        m.end()
    }
}
