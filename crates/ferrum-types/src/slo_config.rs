//! Typed bounds and fallback behavior for optional execution-cost feedback.

use serde::{Deserialize, Serialize};

/// Bounds for one instance's reference estimator, not a device or client SLO.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReferenceEstimatorConfig {
    pub queue_capacity: usize,
    pub max_queue_bytes: usize,
    pub max_participants: usize,
    pub max_summary_bytes: usize,
    pub max_domains: usize,
    pub window_samples: usize,
    pub min_samples: usize,
    pub snapshot_ttl_ms: u64,
    /// Fraction added to the observed window maximum. This is an engineering
    /// margin, not a statistical confidence or latency percentile guarantee.
    pub safety_margin: f64,
}

impl Default for ReferenceEstimatorConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 256,
            max_queue_bytes: 4 * 1024 * 1024,
            max_participants: 256,
            max_summary_bytes: 128 * 1024,
            max_domains: 256,
            window_samples: 32,
            min_samples: 2,
            snapshot_ttl_ms: 30_000,
            safety_margin: 0.25,
        }
    }
}

impl ReferenceEstimatorConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("queue_capacity", self.queue_capacity),
            ("max_queue_bytes", self.max_queue_bytes),
            ("max_participants", self.max_participants),
            ("max_summary_bytes", self.max_summary_bytes),
            ("max_domains", self.max_domains),
            ("window_samples", self.window_samples),
            ("min_samples", self.min_samples),
        ] {
            if value == 0 {
                return Err(format!("SLO reference {name} must be positive"));
            }
        }
        if self.max_queue_bytes < self.max_summary_bytes {
            return Err("SLO reference queue must fit one permitted summary".into());
        }
        if self.min_samples > self.window_samples {
            return Err("SLO reference min_samples exceeds the bounded sample window".into());
        }
        if self.snapshot_ttl_ms == 0 {
            return Err("SLO reference snapshot_ttl_ms must be positive".into());
        }
        if !self.safety_margin.is_finite() || self.safety_margin < 0.0 {
            return Err("SLO reference safety_margin must be finite and nonnegative".into());
        }
        // Reject overflowing declared storage bounds before constructing any
        // queue or history. Actual summary-size validation belongs to the
        // estimator, which knows the execution contract's representation.
        self.max_domains
            .checked_mul(self.window_samples)
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<std::time::Duration>()))
            .ok_or("SLO reference sample storage bound overflows")?;
        Ok(())
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostFallback {
    #[default]
    StaticFallback,
    Reject,
}

/// Unknown/expired predictions cannot silently become reliable zero costs.
/// A bounded wait ends at the original request's ingress plus `max_wait_ms`;
/// replanning must not renew the deadline.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum UnknownCostPolicy {
    #[default]
    StaticFallback,
    Reject,
    Wait {
        max_wait_ms: u64,
        #[serde(default)]
        on_timeout: CostFallback,
    },
}

impl UnknownCostPolicy {
    pub fn validate(&self) -> Result<(), String> {
        if matches!(self, Self::Wait { max_wait_ms: 0, .. }) {
            return Err("SLO unknown-cost wait must have a positive finite deadline".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_bounds_reject_unusable_and_overflowing_storage() {
        let defaults = ReferenceEstimatorConfig::default();
        defaults.validate().unwrap();
        for invalid in [
            ReferenceEstimatorConfig {
                queue_capacity: 0,
                ..defaults
            },
            ReferenceEstimatorConfig {
                max_queue_bytes: defaults.max_summary_bytes - 1,
                ..defaults
            },
            ReferenceEstimatorConfig {
                min_samples: defaults.window_samples + 1,
                ..defaults
            },
            ReferenceEstimatorConfig {
                max_domains: usize::MAX,
                ..defaults
            },
            ReferenceEstimatorConfig {
                snapshot_ttl_ms: 0,
                ..defaults
            },
            ReferenceEstimatorConfig {
                safety_margin: f64::NAN,
                ..defaults
            },
            ReferenceEstimatorConfig {
                safety_margin: -0.1,
                ..defaults
            },
        ] {
            assert!(invalid.validate().is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn unknown_wait_has_an_explicit_nonrenewable_timeout_policy() {
        let wait: UnknownCostPolicy =
            serde_json::from_str(r#"{"mode":"wait","max_wait_ms":25,"on_timeout":"reject"}"#)
                .unwrap();
        assert_eq!(
            wait,
            UnknownCostPolicy::Wait {
                max_wait_ms: 25,
                on_timeout: CostFallback::Reject
            }
        );
        wait.validate().unwrap();
        assert!(UnknownCostPolicy::Wait {
            max_wait_ms: 0,
            on_timeout: CostFallback::StaticFallback
        }
        .validate()
        .is_err());
        assert!(serde_json::from_str::<UnknownCostPolicy>(r#"{"mode":"wait"}"#).is_err());
    }
}
