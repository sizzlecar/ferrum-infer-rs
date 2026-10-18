//! Limits on output sessions and application-owned output buffers.

use crate::{FerrumError, Result};
use serde::{Deserialize, Serialize};

/// Shared by CLI and library engines. Semantic limits count retained logical
/// payload; queue/frame limits count serialized content. Neither counts allocator
/// overhead, tokenizer scratch space or network buffers.
/// These pools are independent of model slots and device capacity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    /// Includes completed streams whose receiver, body or output is still held.
    pub max_sessions: usize,
    pub engine_queue_items: usize,
    /// Generated histories, completed results and retained downstream copies.
    pub semantic_bytes_total: usize,
    /// All retained copies originating from one admitted output session.
    pub semantic_bytes_per_session: usize,
    /// Content owned by queued chunks and chunks retained by their consumers.
    pub queued_bytes_total: usize,
    /// Encoded HTTP frames, including clones retained after the body is dropped.
    pub encoded_bytes_total: usize,
    /// Reject an oversized event rather than waiting for impossible capacity.
    pub max_event_bytes: usize,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            max_sessions: 2048,
            engine_queue_items: 100,
            semantic_bytes_total: 256 * 1024 * 1024,
            semantic_bytes_per_session: 32 * 1024 * 1024,
            queued_bytes_total: 64 * 1024 * 1024,
            encoded_bytes_total: 64 * 1024 * 1024,
            max_event_bytes: 32 * 1024 * 1024,
        }
    }
}

impl OutputConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("max_sessions", self.max_sessions),
            ("engine_queue_items", self.engine_queue_items),
            ("semantic_bytes_total", self.semantic_bytes_total),
            (
                "semantic_bytes_per_session",
                self.semantic_bytes_per_session,
            ),
            ("queued_bytes_total", self.queued_bytes_total),
            ("encoded_bytes_total", self.encoded_bytes_total),
            ("max_event_bytes", self.max_event_bytes),
        ] {
            if value == 0 || u32::try_from(value).is_err() {
                return Err(FerrumError::config(format!(
                    "output.{name} must be within 1..={}",
                    u32::MAX
                )));
            }
        }
        if self.semantic_bytes_per_session > self.semantic_bytes_total {
            return Err(FerrumError::config(
                "output.semantic_bytes_per_session must not exceed output.semantic_bytes_total",
            ));
        }
        if self.max_event_bytes > self.encoded_bytes_total {
            return Err(FerrumError::config(
                "output.max_event_bytes must not exceed output.encoded_bytes_total",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_explicit_limits_round_trip() {
        let default: OutputConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(default, OutputConfig::default());
        default.validate().unwrap();
        let config: OutputConfig = serde_json::from_str(
            r#"{"max_sessions":2,"engine_queue_items":1,"queued_bytes_total":1024,"encoded_bytes_total":4096,"max_event_bytes":2048}"#,
        ).unwrap();
        config.validate().unwrap();
        assert_eq!(
            serde_json::from_value::<OutputConfig>(serde_json::to_value(&config).unwrap()).unwrap(),
            config
        );
        assert!(serde_json::from_str::<OutputConfig>(r#"{"unlimited":true}"#).is_err());
    }

    #[test]
    fn impossible_or_unbounded_limits_are_rejected() {
        let mut config = OutputConfig::default();
        config.max_sessions = 0;
        assert!(config.validate().is_err());
        config = OutputConfig::default();
        config.max_event_bytes = config.encoded_bytes_total + 1;
        assert!(config.validate().is_err());
        config = OutputConfig::default();
        config.queued_bytes_total = usize::MAX;
        assert!(config.validate().is_err());
        config = OutputConfig::default();
        config.semantic_bytes_per_session = config.semantic_bytes_total + 1;
        assert!(config.validate().is_err());
        config = OutputConfig::default();
        config.semantic_bytes_per_session = 0;
        assert!(config.validate().is_err());
    }
}
