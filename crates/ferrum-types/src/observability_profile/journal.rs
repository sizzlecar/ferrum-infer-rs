//! Declared per-writer limits for product observability delivery.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ObservabilityJournalConfig {
    pub queue_capacity: usize,
    pub payload_capacity_bytes: usize,
    pub max_record_bytes: usize,
    pub max_batch_bytes: usize,
    pub max_batch_records: usize,
}

impl Default for ObservabilityJournalConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 256,
            payload_capacity_bytes: 16 * 1024 * 1024,
            max_record_bytes: 1024 * 1024,
            max_batch_bytes: 4 * 1024 * 1024,
            max_batch_records: 1024,
        }
    }
}

impl ObservabilityJournalConfig {
    pub fn validate(&self) -> crate::Result<()> {
        for (name, value) in [
            ("queue_capacity", self.queue_capacity),
            ("payload_capacity_bytes", self.payload_capacity_bytes),
            ("max_record_bytes", self.max_record_bytes),
            ("max_batch_bytes", self.max_batch_bytes),
            ("max_batch_records", self.max_batch_records),
        ] {
            if value == 0 || value > isize::MAX as usize {
                return Err(crate::FerrumError::config(format!(
                    "profile_journal.{name} must be nonzero and addressable"
                )));
            }
        }
        if self.max_record_bytes > self.max_batch_bytes
            || self.max_batch_bytes > self.payload_capacity_bytes
        {
            return Err(crate::FerrumError::config(
                "profile_journal requires max_record_bytes <= max_batch_bytes <= payload_capacity_bytes",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_config_preserves_finite_defaults_and_rejects_invalid_limits() {
        let defaults = ObservabilityJournalConfig::default();
        defaults.validate().unwrap();
        assert_eq!(
            serde_json::from_str::<ObservabilityJournalConfig>("{}").unwrap(),
            defaults
        );
        for invalid in [
            ObservabilityJournalConfig {
                queue_capacity: 0,
                ..defaults
            },
            ObservabilityJournalConfig {
                max_batch_records: 0,
                ..defaults
            },
            ObservabilityJournalConfig {
                max_record_bytes: defaults.max_batch_bytes + 1,
                ..defaults
            },
            ObservabilityJournalConfig {
                max_batch_bytes: defaults.payload_capacity_bytes + 1,
                ..defaults
            },
            ObservabilityJournalConfig {
                payload_capacity_bytes: usize::MAX,
                ..defaults
            },
        ] {
            assert!(invalid.validate().is_err());
        }
        assert!(
            serde_json::from_str::<ObservabilityJournalConfig>(r#"{"unlimited":true}"#).is_err()
        );
    }
}
