//! Fixed reference units are explicit calibration inputs, not online costs.
use serde::{Deserialize, Serialize};
use std::{num::NonZeroUsize, path::PathBuf};

pub const PREFILL_REFERENCE_MAX_FILE_BYTES: usize = 16 * 1024 * 1024;
pub const PREFILL_REFERENCE_MAX_CURVES: usize = 256;
pub const PREFILL_REFERENCE_MAX_POINTS: usize = 8192;
pub const PREFILL_REFERENCE_MAX_POINTS_PER_CURVE: usize = 1024;
pub const PREFILL_REFERENCE_MAX_SAMPLES: usize = 65_536;
pub const PREFILL_REFERENCE_MAX_REPETITIONS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloPrefillReferenceLimits {
    pub max_file_bytes: NonZeroUsize,
    pub max_curves: NonZeroUsize,
    pub max_points: NonZeroUsize,
    pub max_points_per_curve: NonZeroUsize,
    pub max_samples: NonZeroUsize,
}
impl Default for SloPrefillReferenceLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: NonZeroUsize::new(PREFILL_REFERENCE_MAX_FILE_BYTES).unwrap(),
            max_curves: NonZeroUsize::new(PREFILL_REFERENCE_MAX_CURVES).unwrap(),
            max_points: NonZeroUsize::new(PREFILL_REFERENCE_MAX_POINTS).unwrap(),
            max_points_per_curve: NonZeroUsize::new(PREFILL_REFERENCE_MAX_POINTS_PER_CURVE)
                .unwrap(),
            max_samples: NonZeroUsize::new(PREFILL_REFERENCE_MAX_SAMPLES).unwrap(),
        }
    }
}
impl SloPrefillReferenceLimits {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value, ceiling) in [
            (
                "max_file_bytes",
                self.max_file_bytes,
                PREFILL_REFERENCE_MAX_FILE_BYTES,
            ),
            ("max_curves", self.max_curves, PREFILL_REFERENCE_MAX_CURVES),
            ("max_points", self.max_points, PREFILL_REFERENCE_MAX_POINTS),
            (
                "max_points_per_curve",
                self.max_points_per_curve,
                PREFILL_REFERENCE_MAX_POINTS_PER_CURVE,
            ),
            (
                "max_samples",
                self.max_samples,
                PREFILL_REFERENCE_MAX_SAMPLES,
            ),
        ] {
            if value.get() > ceiling {
                return Err(format!(
                    "prefill reference {name} exceeds hard bound {ceiling}"
                ));
            }
        }
        if self.max_points_per_curve.get() < 2 || self.max_points_per_curve > self.max_points {
            return Err("prefill reference point budgets cannot hold a complete curve".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SloPrefillReferenceConfig {
    pub artifact_path: PathBuf,
    /// SHA-256 of the explicitly chosen reference protocol, as 32 bytes.
    /// Loading a different protocol cannot silently change the scoring unit.
    pub expected_protocol_sha256: [u8; 32],
    #[serde(default)]
    pub limits: SloPrefillReferenceLimits,
}
impl SloPrefillReferenceConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.artifact_path.as_os_str().is_empty() {
            return Err("prefill reference artifact_path must not be empty".into());
        }
        if self.expected_protocol_sha256 == [0; 32] {
            return Err("prefill reference requires an explicit protocol digest".into());
        }
        self.limits.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn limits_and_config_are_explicit_and_bounded() {
        let mut config = SloPrefillReferenceConfig {
            artifact_path: "reference.json".into(),
            expected_protocol_sha256: [7; 32],
            limits: Default::default(),
        };
        assert!(config.validate().is_ok());
        config.limits.max_points_per_curve = NonZeroUsize::new(1).unwrap();
        assert!(config.validate().is_err());
        config.limits = Default::default();
        config.limits.max_samples = NonZeroUsize::new(usize::MAX).unwrap();
        assert!(config.validate().is_err());
    }
    #[test]
    fn unknown_reference_fields_are_rejected() {
        let config = SloPrefillReferenceConfig {
            artifact_path: "reference.json".into(),
            expected_protocol_sha256: [7; 32],
            limits: Default::default(),
        };
        let mut wire = serde_json::to_value(config).unwrap();
        wire["expected_protocol_sha"] = serde_json::json!([7]);
        assert!(serde_json::from_value::<SloPrefillReferenceConfig>(wire).is_err());
    }
}
