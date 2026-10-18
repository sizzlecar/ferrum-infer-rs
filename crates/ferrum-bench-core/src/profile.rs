//! Historical structured profile schema and artifact readers.
//!
//! Runtime producers use the instance-owned product journal. This module keeps
//! old profile artifacts readable without installing a writer or reading process
//! configuration.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Required top-level fields for one JSONL profile event.
pub const REQUIRED_PROFILE_EVENT_FIELDS: &[&str] = &[
    "event",
    "commit_sha",
    "env_hash",
    "model",
    "concurrency",
    "shape",
    "stage_us",
    "graph_enabled",
    "runtime_flags",
];

/// Stable profile event envelope.
///
/// Field order is schema-significant for JSON output. Do not reorder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileEvent {
    /// Event family, for example `unified_prof`, `bucket_prof`, or
    /// `vllm_moe_config`.
    pub event: String,
    /// Git commit for the binary under test. `None` is allowed only when the
    /// artifact producer cannot access VCS metadata, but the key must exist.
    pub commit_sha: Option<String>,
    /// Canonical runtime environment hash (`sha256:...`).
    pub env_hash: String,
    /// HuggingFace model id or local model label.
    pub model: String,
    /// Closed-loop concurrency for the profiled cell.
    pub concurrency: u32,
    /// Shape attributes such as batch size, top-k pairs, blocks, or sequence
    /// lengths. Values are JSON scalars so producers can extend keys without a
    /// schema bump.
    pub shape: BTreeMap<String, Value>,
    /// Timings in microseconds for the relevant stage or substage.
    pub stage_us: BTreeMap<String, Value>,
    /// Whether the profiled path ran under CUDA graph replay.
    pub graph_enabled: bool,
    /// Runtime flags/config snapshot that affected this event.
    pub runtime_flags: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_line: Option<String>,
}

impl ProfileEvent {
    pub fn validate(&self) -> Result<(), ProfileValidationError> {
        if self.event.trim().is_empty() {
            return Err(ProfileValidationError::new("event must be non-empty"));
        }
        if let Some(commit_sha) = &self.commit_sha {
            if commit_sha.trim().is_empty() {
                return Err(ProfileValidationError::new(
                    "commit_sha must be non-empty when present",
                ));
            }
        }
        if !self.env_hash.starts_with("sha256:") {
            return Err(ProfileValidationError::new(
                "env_hash must start with sha256:",
            ));
        }
        if self.model.trim().is_empty() {
            return Err(ProfileValidationError::new("model must be non-empty"));
        }
        if self.concurrency == 0 {
            return Err(ProfileValidationError::new("concurrency must be > 0"));
        }
        if !self.runtime_flags.is_object() {
            return Err(ProfileValidationError::new(
                "runtime_flags must be an object",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileValidationError {
    message: String,
}

impl ProfileValidationError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn line(line_no: usize, message: impl Into<String>) -> Self {
        Self {
            message: format!("line {line_no}: {}", message.into()),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ProfileValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProfileValidationError {}

/// Parse and validate one JSON value as a [`ProfileEvent`].
///
/// This checks field presence before deserialization so optional values like
/// `commit_sha: null` are distinct from a missing key.
pub fn parse_profile_event_value(value: Value) -> Result<ProfileEvent, ProfileValidationError> {
    let object = value
        .as_object()
        .ok_or_else(|| ProfileValidationError::new("profile event must be a JSON object"))?;
    for key in REQUIRED_PROFILE_EVENT_FIELDS {
        if !object.contains_key(*key) {
            return Err(ProfileValidationError::new(format!(
                "missing required field: {key}"
            )));
        }
    }
    let event: ProfileEvent = serde_json::from_value(value)
        .map_err(|err| ProfileValidationError::new(format!("invalid profile event: {err}")))?;
    event.validate()?;
    Ok(event)
}

/// Parse a profile JSONL payload and validate every non-blank line.
pub fn parse_profile_jsonl_str(input: &str) -> Result<Vec<ProfileEvent>, ProfileValidationError> {
    let mut events = Vec::new();
    for (line_idx, line) in input.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|err| ProfileValidationError::line(line_no, format!("invalid JSON: {err}")))?;
        let event = parse_profile_event_value(value)
            .map_err(|err| ProfileValidationError::line(line_no, err.to_string()))?;
        events.push(event);
    }
    Ok(events)
}

/// Ensure that a profile contains at least one event for every required group.
pub fn require_profile_event_groups(
    events: &[ProfileEvent],
    required: &[&str],
) -> Result<(), ProfileValidationError> {
    let mut missing = Vec::new();
    for required_event in required {
        if !events.iter().any(|event| event.event == *required_event) {
            missing.push(*required_event);
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(ProfileValidationError::new(format!(
            "missing profile event groups: {}",
            missing.join(", ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_event_value() -> Value {
        serde_json::json!({
            "event": "unified_prof",
            "commit_sha": "abc123",
            "env_hash": "sha256:env",
            "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
            "concurrency": 32,
            "shape": {"batch": 32, "active_blocks": 65},
            "stage_us": {"model": 12500.0, "decode_post": 350.0},
            "graph_enabled": true,
            "runtime_flags": {
                "schema_version": 1,
                "entries": []
            },
            "source": "server_log",
            "source_line": "[unified-prof] model=12500 decode_post=350"
        })
    }

    #[test]
    fn profile_event_round_trips_and_validates() {
        let event = parse_profile_event_value(valid_event_value()).expect("valid event");
        event.validate().expect("valid schema");
        assert_eq!(event.event, "unified_prof");
        assert_eq!(event.concurrency, 32);
        assert_eq!(
            event.shape.get("active_blocks").and_then(Value::as_i64),
            Some(65)
        );

        let encoded = serde_json::to_string(&event).expect("serialize");
        let decoded = parse_profile_event_value(serde_json::from_str(&encoded).unwrap())
            .expect("decode serialized event");
        assert_eq!(decoded, event);
    }

    #[test]
    fn profile_event_rejects_every_missing_required_field() {
        for key in REQUIRED_PROFILE_EVENT_FIELDS {
            let mut event = valid_event_value();
            event.as_object_mut().unwrap().remove(*key);
            let err = parse_profile_event_value(event).unwrap_err();
            assert!(
                err.message()
                    .contains(&format!("missing required field: {key}")),
                "unexpected error for missing {key}: {}",
                err.message()
            );
        }
    }

    #[test]
    fn profile_event_accepts_null_commit_sha_when_key_is_present() {
        let mut event = valid_event_value();
        event["commit_sha"] = Value::Null;
        let parsed = parse_profile_event_value(event).expect("null commit_sha is allowed");
        assert_eq!(parsed.commit_sha, None);
    }

    #[test]
    fn profile_event_rejects_bad_env_hash_and_zero_concurrency() {
        let mut bad_hash = valid_event_value();
        bad_hash["env_hash"] = Value::String("env".to_string());
        assert!(parse_profile_event_value(bad_hash)
            .unwrap_err()
            .message()
            .contains("env_hash"));

        let mut zero_concurrency = valid_event_value();
        zero_concurrency["concurrency"] = Value::Number(0.into());
        assert!(parse_profile_event_value(zero_concurrency)
            .unwrap_err()
            .message()
            .contains("concurrency"));
    }

    #[test]
    fn profile_jsonl_parses_multiple_events() {
        let one = serde_json::to_string(&valid_event_value()).unwrap();
        let mut two_value = valid_event_value();
        two_value["event"] = Value::String("bucket_prof".to_string());
        let two = serde_json::to_string(&two_value).unwrap();

        let events = parse_profile_jsonl_str(&format!("{one}\n\n{two}\n")).unwrap();
        assert_eq!(events.len(), 2);
        require_profile_event_groups(&events, &["unified_prof", "bucket_prof"]).unwrap();
    }

    #[test]
    fn required_event_groups_reject_missing_groups() {
        let event = parse_profile_event_value(valid_event_value()).unwrap();
        let err = require_profile_event_groups(&[event], &["unified_prof", "bucket_prof"])
            .expect_err("missing bucket profile");
        assert!(err.message().contains("bucket_prof"));
    }

    #[test]
    fn profile_parser_covers_three_fixture_artifact_shapes() {
        let default_graph_on = [
            serde_json::json!({
                "event": "graph_prof",
                "commit_sha": "abc123",
                "env_hash": "sha256:graph",
                "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
                "concurrency": 32,
                "shape": {"call": 64},
                "stage_us": {"upload": 110, "launch": 240, "sync": 0, "total": 350},
                "graph_enabled": true,
                "runtime_flags": {"preset": "m3_qwen3_30b_a3b_int4"}
            }),
            serde_json::json!({
                "event": "unified_prof",
                "commit_sha": "abc123",
                "env_hash": "sha256:graph",
                "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
                "concurrency": 32,
                "shape": {"items": 32, "decode": 32},
                "stage_us": {"total": 14900, "model": 14000, "decode_post": 600},
                "graph_enabled": true,
                "runtime_flags": {"preset": "m3_qwen3_30b_a3b_int4"}
            }),
        ];
        let graph_off_route_dump = [
            serde_json::json!({
                "event": "moe_dump",
                "commit_sha": "abc123",
                "env_hash": "sha256:route",
                "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
                "concurrency": 32,
                "shape": {"batch_x_topk": 256, "active_blocks": 65, "unique_experts": 61},
                "stage_us": {},
                "graph_enabled": false,
                "runtime_flags": {"FERRUM_MOE_GRAPH": "0"}
            }),
            serde_json::json!({
                "event": "vllm_moe_config",
                "commit_sha": "abc123",
                "env_hash": "sha256:route",
                "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
                "concurrency": 32,
                "shape": {"batch_x_topk": 256, "prob_m": 32, "thread_k": 64, "thread_n": 128},
                "stage_us": {},
                "graph_enabled": false,
                "runtime_flags": {"FERRUM_MOE_GRAPH": "0"}
            }),
            serde_json::json!({
                "event": "bucket_prof",
                "commit_sha": "abc123",
                "env_hash": "sha256:route",
                "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
                "concurrency": 32,
                "shape": {"layers": 48},
                "stage_us": {"gemm1": 5860, "gemm3": 2890, "combine": 250},
                "graph_enabled": false,
                "runtime_flags": {"FERRUM_MOE_PROFILE": "1"}
            }),
        ];
        let fa_layout_attention_ab = [
            serde_json::json!({
                "event": "unified_layer_prof",
                "commit_sha": "abc123",
                "env_hash": "sha256:fa",
                "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
                "concurrency": 32,
                "shape": {"m": 128, "seqs": 4, "sampled": 4},
                "stage_us": {"attn": 46000, "moe": 36000, "layer_sum": 91000},
                "graph_enabled": false,
                "runtime_flags": {"FERRUM_FA_LAYOUT_VARLEN": "1"}
            }),
            serde_json::json!({
                "event": "unified_prof",
                "commit_sha": "abc123",
                "env_hash": "sha256:fa",
                "model": "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
                "concurrency": 32,
                "shape": {"items": 32, "prefill": 4, "decode": 28},
                "stage_us": {"total": 61000, "model": 59000, "decode_post": 350},
                "graph_enabled": false,
                "runtime_flags": {"FERRUM_FA_LAYOUT_VARLEN": "1"}
            }),
        ];

        for (fixture, required) in [
            (&default_graph_on[..], &["graph_prof", "unified_prof"][..]),
            (
                &graph_off_route_dump[..],
                &["moe_dump", "vllm_moe_config", "bucket_prof"][..],
            ),
            (
                &fa_layout_attention_ab[..],
                &["unified_layer_prof", "unified_prof"][..],
            ),
        ] {
            let jsonl = fixture
                .iter()
                .map(|value| serde_json::to_string(value).unwrap())
                .collect::<Vec<_>>()
                .join("\n");
            let events = parse_profile_jsonl_str(&jsonl).unwrap();
            require_profile_event_groups(&events, required).unwrap();
        }
    }
}
