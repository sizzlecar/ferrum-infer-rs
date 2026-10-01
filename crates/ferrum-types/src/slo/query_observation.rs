//! Passive, non-authorizing planner diagnostics. Never a profile or witness.
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloRequiredQueryObservationConfig {
    #[default]
    Disabled,
    /// Observe-only discovery of actually visited root queries without a
    /// calibrated model. Unknown stops each edge; this is not full future coverage.
    StructuredUncalibratedV1 {
        path: PathBuf,
        #[serde(default)]
        limits: SloRequiredQueryObservationLimits,
    },
    StructuredRequiredV1 {
        path: PathBuf,
        #[serde(default)]
        limits: SloRequiredQueryObservationLimits,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloRequiredQueryObservationLimits {
    pub queue_events: usize,
    pub queue_bytes: usize,
    pub max_event_bytes: usize,
    pub max_transaction_events: u64,
    pub max_events: u64,
    pub max_file_bytes: u64,
}
impl Default for SloRequiredQueryObservationLimits {
    fn default() -> Self {
        Self {
            queue_events: 256,
            queue_bytes: 8 * 1024 * 1024,
            max_event_bytes: 1024 * 1024,
            max_transaction_events: 4096,
            max_events: 1_000_000,
            max_file_bytes: 1024 * 1024 * 1024,
        }
    }
}
impl SloRequiredQueryObservationConfig {
    pub fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
    pub fn enabled(&self) -> bool {
        !matches!(self, Self::Disabled)
    }
    pub fn is_uncalibrated(&self) -> bool {
        matches!(self, Self::StructuredUncalibratedV1 { .. })
    }
    pub fn validate(&self) -> Result<(), String> {
        let (Self::StructuredRequiredV1 { path, limits }
        | Self::StructuredUncalibratedV1 { path, limits }) = self
        else {
            return Ok(());
        };
        if path.as_os_str().is_empty() {
            return Err("required-query observation path is empty".into());
        }
        // Independently cap a producer allocation and all queued allocations.
        // These are diagnostic storage ceilings, never planner search limits.
        if limits.queue_events == 0
            || limits.queue_events > 65_536
            || limits.max_event_bytes == 0
            || limits.max_event_bytes > 1024 * 1024
            || limits.queue_bytes < limits.max_event_bytes
            || limits.queue_bytes > 256 * 1024 * 1024
            || limits.max_transaction_events == 0
            || limits.max_events == 0
            || limits.max_transaction_events > limits.max_events
            || limits.max_file_bytes == 0
        {
            return Err("invalid required-query observation storage limits".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_is_default_and_storage_limits_are_checked() {
        assert_eq!(
            SloRequiredQueryObservationConfig::default(),
            SloRequiredQueryObservationConfig::Disabled
        );
        let mut limits = SloRequiredQueryObservationLimits::default();
        limits.queue_bytes = limits.max_event_bytes - 1;
        assert!(SloRequiredQueryObservationConfig::StructuredRequiredV1 {
            path: "queries.jsonl".into(),
            limits
        }
        .validate()
        .is_err());
    }
}
