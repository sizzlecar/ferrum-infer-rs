//! Structured profile event schema shared by benchmark runners and consumers.
//!
//! This intentionally accepts log-derived transitional events as long as they
//! carry the locked envelope fields. Native runtime emitters should use the
//! same shape and fill `shape` / `stage_us` with typed values.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const PROFILE_JSONL_ENV: &str = "FERRUM_PROFILE_JSONL";
pub const PROFILE_COMMIT_SHA_ENV: &str = "FERRUM_PROFILE_COMMIT_SHA";
pub const PROFILE_ENV_HASH_ENV: &str = "FERRUM_PROFILE_ENV_HASH";
pub const PROFILE_MODEL_ENV: &str = "FERRUM_PROFILE_MODEL";
pub const PROFILE_CONCURRENCY_ENV: &str = "FERRUM_PROFILE_CONCURRENCY";
pub const PROFILE_RUNTIME_FLAGS_JSON_ENV: &str = "FERRUM_PROFILE_RUNTIME_FLAGS_JSON";

static GLOBAL_PROFILE: OnceLock<ProfileJsonlWriter> = OnceLock::new();

/// Global structured profile writer, lazily configured from
/// `FERRUM_PROFILE_JSONL`.
pub fn global_profile() -> &'static ProfileJsonlWriter {
    GLOBAL_PROFILE.get_or_init(ProfileJsonlWriter::from_env)
}

/// Configure the global profile writer from typed startup config.
///
/// Returns `Ok(false)` if another caller already initialized the global writer.
pub fn configure_global_profile(config: ProfileSinkConfig) -> io::Result<bool> {
    let writer = ProfileJsonlWriter::from_config(config)?;
    Ok(GLOBAL_PROFILE.set(writer).is_ok())
}

pub fn flush_global_profile() {
    if let Some(writer) = GLOBAL_PROFILE.get() {
        let _ = writer.flush();
    }
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileMetadata {
    pub commit_sha: Option<String>,
    pub env_hash: String,
    pub model: String,
    pub concurrency: u32,
    pub runtime_flags: Value,
}

impl Default for ProfileMetadata {
    fn default() -> Self {
        Self {
            commit_sha: None,
            env_hash: "sha256:unknown".to_string(),
            model: "unknown".to_string(),
            concurrency: 1,
            runtime_flags: Value::Object(serde_json::Map::new()),
        }
    }
}

impl ProfileMetadata {
    pub fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    pub fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let vars = env_vars_map(vars);
        Self::from_env_map(&vars)
    }

    fn from_env_map(vars: &BTreeMap<String, String>) -> Self {
        let commit_sha = vars
            .get(PROFILE_COMMIT_SHA_ENV)
            .filter(|value| !value.trim().is_empty());
        let env_hash = vars
            .get(PROFILE_ENV_HASH_ENV)
            .filter(|value| value.starts_with("sha256:"))
            .cloned()
            .unwrap_or_else(|| "sha256:unknown".to_string());
        let model = vars
            .get(PROFILE_MODEL_ENV)
            .filter(|value| !value.trim().is_empty())
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let concurrency = vars
            .get(PROFILE_CONCURRENCY_ENV)
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1);
        let runtime_flags = vars
            .get(PROFILE_RUNTIME_FLAGS_JSON_ENV)
            .and_then(|value| serde_json::from_str::<Value>(&value).ok())
            .filter(Value::is_object)
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

        Self {
            commit_sha: commit_sha.cloned(),
            env_hash,
            model,
            concurrency,
            runtime_flags,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfileSinkConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jsonl_path: Option<PathBuf>,
    #[serde(default)]
    pub metadata: ProfileMetadata,
}

impl ProfileSinkConfig {
    pub fn disabled() -> Self {
        Self {
            jsonl_path: None,
            metadata: ProfileMetadata::default(),
        }
    }

    pub fn enabled(jsonl_path: PathBuf, metadata: ProfileMetadata) -> Self {
        Self {
            jsonl_path: Some(jsonl_path),
            metadata,
        }
    }

    pub fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    pub fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let vars = env_vars_map(vars);
        match vars.get(PROFILE_JSONL_ENV) {
            Some(path) if !path.trim().is_empty() => {
                Self::enabled(PathBuf::from(path), ProfileMetadata::from_env_map(&vars))
            }
            _ => Self::disabled(),
        }
    }
}

fn env_vars_map<I, K, V>(vars: I) -> BTreeMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: Into<String>,
    V: Into<String>,
{
    vars.into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .collect()
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

pub struct ProfileJsonlWriter {
    inner: Mutex<ProfileJsonlWriterInner>,
    metadata: ProfileMetadata,
}

enum ProfileJsonlWriterInner {
    Disabled,
    File { path: PathBuf, file: File },
}

impl ProfileJsonlWriter {
    pub fn from_env() -> Self {
        let config = ProfileSinkConfig::from_env();
        match Self::from_config(config) {
            Ok(writer) => writer,
            Err(err) => {
                eprintln!("[profile-jsonl] failed to open configured sink: {err}");
                Self::disabled()
            }
        }
    }

    pub fn from_config(config: ProfileSinkConfig) -> io::Result<Self> {
        match config.jsonl_path {
            Some(path) => Self::enabled(path, config.metadata),
            None => Ok(Self::disabled()),
        }
    }

    pub fn enabled(path: PathBuf, metadata: ProfileMetadata) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            inner: Mutex::new(ProfileJsonlWriterInner::File { path, file }),
            metadata,
        })
    }

    pub fn disabled() -> Self {
        Self {
            inner: Mutex::new(ProfileJsonlWriterInner::Disabled),
            metadata: ProfileMetadata::default(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        matches!(
            *self.inner.lock().unwrap(),
            ProfileJsonlWriterInner::File { .. }
        )
    }

    pub fn push_event(
        &self,
        event: impl Into<String>,
        shape: BTreeMap<String, Value>,
        stage_us: BTreeMap<String, Value>,
        graph_enabled: bool,
    ) -> io::Result<()> {
        if !self.is_enabled() {
            return Ok(());
        }
        let event = ProfileEvent {
            event: event.into(),
            commit_sha: self.metadata.commit_sha.clone(),
            env_hash: self.metadata.env_hash.clone(),
            model: self.metadata.model.clone(),
            concurrency: self.metadata.concurrency,
            shape,
            stage_us,
            graph_enabled,
            runtime_flags: self.metadata.runtime_flags.clone(),
            source: Some("native".to_string()),
            source_line: None,
        };
        event
            .validate()
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

        let mut inner = self.inner.lock().unwrap();
        if let ProfileJsonlWriterInner::File { file, .. } = &mut *inner {
            serde_json::to_writer(&mut *file, &event)?;
            file.write_all(b"\n")?;
            file.flush()?;
        }
        Ok(())
    }

    pub fn flush(&self) -> io::Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if let ProfileJsonlWriterInner::File { file, .. } = &mut *inner {
            file.flush()?;
        }
        Ok(())
    }

    pub fn path(&self) -> Option<PathBuf> {
        let inner = self.inner.lock().unwrap();
        match &*inner {
            ProfileJsonlWriterInner::Disabled => None,
            ProfileJsonlWriterInner::File { path, .. } => Some(path.clone()),
        }
    }
}

pub fn profile_fields_from_json(value: Value) -> BTreeMap<String, Value> {
    match value {
        Value::Object(map) => map.into_iter().collect(),
        _ => BTreeMap::new(),
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

    #[test]
    fn disabled_profile_writer_is_noop() {
        let writer = ProfileJsonlWriter::disabled();
        assert!(!writer.is_enabled());
        writer
            .push_event(
                "iter_prof",
                profile_fields_from_json(serde_json::json!({"batch_size": 1})),
                profile_fields_from_json(serde_json::json!({"total": 10})),
                false,
            )
            .unwrap();
        assert_eq!(writer.path(), None);
    }

    #[test]
    fn enabled_profile_writer_writes_valid_jsonl() {
        let dir = tempdir();
        let path = dir.join("profile.jsonl");
        let writer = ProfileJsonlWriter::enabled(
            path.clone(),
            ProfileMetadata {
                commit_sha: Some("abc123".to_string()),
                env_hash: "sha256:env".to_string(),
                model: "model".to_string(),
                concurrency: 32,
                runtime_flags: serde_json::json!({"schema_version": 1}),
            },
        )
        .unwrap();

        writer
            .push_event(
                "bucket_prof",
                profile_fields_from_json(serde_json::json!({"layers": 48})),
                profile_fields_from_json(serde_json::json!({"gemm1": 1200, "gemm3": 800})),
                true,
            )
            .unwrap();
        writer.flush().unwrap();

        let jsonl = std::fs::read_to_string(&path).unwrap();
        let events = parse_profile_jsonl_str(&jsonl).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event, "bucket_prof");
        assert_eq!(events[0].concurrency, 32);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn typed_profile_sink_config_builds_writer() {
        let dir = tempdir();
        let path = dir.join("typed-profile.jsonl");
        let config = ProfileSinkConfig::enabled(
            path.clone(),
            ProfileMetadata {
                commit_sha: Some("def456".to_string()),
                env_hash: "sha256:typed".to_string(),
                model: "typed-model".to_string(),
                concurrency: 16,
                runtime_flags: serde_json::json!({"source": "typed"}),
            },
        );
        let writer = ProfileJsonlWriter::from_config(config).unwrap();
        assert!(writer.is_enabled());
        assert_eq!(writer.path().as_deref(), Some(path.as_path()));

        writer
            .push_event(
                "unified_prof",
                profile_fields_from_json(serde_json::json!({"items": 16})),
                profile_fields_from_json(serde_json::json!({"total": 1000})),
                false,
            )
            .unwrap();
        writer.flush().unwrap();

        let jsonl = std::fs::read_to_string(&path).unwrap();
        let events = parse_profile_jsonl_str(&jsonl).unwrap();
        assert_eq!(events[0].commit_sha.as_deref(), Some("def456"));
        assert_eq!(events[0].env_hash, "sha256:typed");
        assert_eq!(events[0].model, "typed-model");
        assert_eq!(events[0].concurrency, 16);
        assert_eq!(events[0].runtime_flags["source"], "typed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn profile_metadata_parses_env_snapshot() {
        let metadata = ProfileMetadata::from_env_vars([
            (PROFILE_COMMIT_SHA_ENV, "abc123"),
            (PROFILE_ENV_HASH_ENV, "sha256:env"),
            (PROFILE_MODEL_ENV, "model"),
            (PROFILE_CONCURRENCY_ENV, "32"),
            (PROFILE_RUNTIME_FLAGS_JSON_ENV, r#"{"fa_layout":true}"#),
        ]);

        assert_eq!(metadata.commit_sha.as_deref(), Some("abc123"));
        assert_eq!(metadata.env_hash, "sha256:env");
        assert_eq!(metadata.model, "model");
        assert_eq!(metadata.concurrency, 32);
        assert_eq!(metadata.runtime_flags["fa_layout"], true);
    }

    #[test]
    fn profile_sink_config_parses_env_snapshot() {
        let config = ProfileSinkConfig::from_env_vars([
            (PROFILE_JSONL_ENV, "/tmp/profile.jsonl"),
            (PROFILE_ENV_HASH_ENV, "sha256:env"),
            (PROFILE_MODEL_ENV, "model"),
        ]);

        assert_eq!(
            config.jsonl_path.as_deref(),
            Some(std::path::Path::new("/tmp/profile.jsonl"))
        );
        assert_eq!(config.metadata.env_hash, "sha256:env");
        assert_eq!(config.metadata.model, "model");

        let disabled = ProfileSinkConfig::from_env_vars([(PROFILE_JSONL_ENV, "")]);
        assert_eq!(disabled.jsonl_path, None);
    }

    fn tempdir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "ferrum-profile-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}
