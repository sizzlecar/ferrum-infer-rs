//! Explicit runtime policy, separate from immutable profile6 fit/residual data.
use super::*;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloSelectedFeedbackPolicy {
    #[default]
    #[serde(deserialize_with = "deserialize_disabled")]
    Disabled,
    RetrospectiveFamilyMarginV1 {
        policy: SloSelectedFeedbackSettingsV1,
        storage: SloSelectedFeedbackStorageV1,
    },
}

// Serde's internally tagged unit-variant visitor discards remaining entries.
// Use an empty struct visitor so Disabled has the same strict wire boundary
// as the configured variant, without changing its public unit variant or wire.
fn deserialize_disabled<'de, D>(deserializer: D) -> Result<(), D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Empty {}
    Empty::deserialize(deserializer).map(|_| ())
}

/// No empirical defaults: every limit is declared by the operator/protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SloSelectedFeedbackSettingsV1 {
    pub window_samples: NonZeroUsize,
    pub minimum_underestimates: NonZeroUsize,
    pub minimum_consecutive_underestimates: NonZeroUsize,
    pub trigger_excess_ns: NonZeroU64,
    pub correction_padding_ns: u64,
    pub maximum_family_margin_ns: NonZeroU64,
    pub maximum_consumption_lag_ns: NonZeroU64,
    pub maximum_uncomparable_observations: u64,
    pub maximum_failed_or_partial: u64,
    pub maximum_queue_drops: u64,
    pub maximum_state_bytes: NonZeroUsize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloSelectedFeedbackStorageV1 {
    CreateNew { path: PathBuf },
    Resume { path: PathBuf },
}
impl SloSelectedFeedbackStorageV1 {
    pub fn path(&self) -> &std::path::Path {
        match self {
            Self::CreateNew { path } | Self::Resume { path } => path,
        }
    }
}
impl SloSelectedFeedbackPolicy {
    pub fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
    pub(super) fn validate(&self, config: &SloCostObservationConfig) -> Result<(), String> {
        let Self::RetrospectiveFamilyMarginV1 { policy: p, storage } = self else {
            return Ok(());
        };
        if !config.predictor.is_selected() {
            return Err(
                "selected feedback requires the explicit selected whole-wave predictor".into(),
            );
        }
        if storage.path().as_os_str().is_empty()
            || !storage.path().is_absolute()
            || storage.path().file_name().is_none()
        {
            return Err("selected feedback requires an absolute versioned receipt path".into());
        }
        if p.minimum_consecutive_underestimates > p.minimum_underestimates
            || p.minimum_underestimates > p.window_samples
            || p.window_samples
                .get()
                .checked_mul(config.model.max_buckets.get())
                .is_none_or(|n| n > config.model.max_retained_samples.get())
            || p.maximum_family_margin_ns > config.model.max_wave_ns
            || p.correction_padding_ns > p.maximum_family_margin_ns.get()
            || p.trigger_excess_ns > p.maximum_family_margin_ns
            || p.maximum_consumption_lag_ns > config.model.max_sample_age_ns
            || p.maximum_state_bytes.get() > config.profile_import.max_file_bytes.get()
        {
            return Err("selected feedback exceeds declared model/storage bounds or has inconsistent thresholds".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_feedback_is_disabled_on_old_wire_and_requires_all_new_policy_limits() {
        let old: SloCostObservationConfig = serde_json::from_str("{}").unwrap();
        assert!(old.selected_feedback.is_disabled());
        assert!(serde_json::to_value(old)
            .unwrap()
            .get("selected_feedback")
            .is_none());
        assert!(serde_json::from_str::<SloSelectedFeedbackPolicy>(r#"{"kind":"retrospective_family_margin_v1","policy":{},"storage":{"mode":"resume","path":"/receipt.json"}}"#).is_err());
        let disabled = SloSelectedFeedbackPolicy::Disabled;
        assert_eq!(
            serde_json::to_string(&disabled).unwrap(),
            r#"{"kind":"disabled"}"#
        );
        assert_eq!(
            serde_json::from_str::<SloSelectedFeedbackPolicy>(r#"{"kind":"disabled"}"#).unwrap(),
            disabled
        );
        for invalid in [
            r#"{"kind":"disabled","threshold":1}"#,
            r#"{"threshold":null,"kind":"disabled"}"#,
            r#"{"kind":"disabled","policy":{}}"#,
            r#"{"kind":"disabled","storage":{"mode":"resume","path":"/receipt.json"}}"#,
        ] {
            assert!(serde_json::from_str::<SloSelectedFeedbackPolicy>(invalid).is_err());
        }
    }
    #[test]
    fn selected_feedback_policy_cannot_apply_to_legacy_or_exceed_bounded_observation_storage() {
        let policy = SloSelectedFeedbackPolicy::RetrospectiveFamilyMarginV1 {
            policy: SloSelectedFeedbackSettingsV1 {
                window_samples: NonZeroUsize::new(2).unwrap(),
                minimum_underestimates: NonZeroUsize::new(2).unwrap(),
                minimum_consecutive_underestimates: NonZeroUsize::new(2).unwrap(),
                trigger_excess_ns: NonZeroU64::new(2).unwrap(),
                correction_padding_ns: 1,
                maximum_family_margin_ns: NonZeroU64::new(100).unwrap(),
                maximum_consumption_lag_ns: NonZeroU64::new(50).unwrap(),
                maximum_uncomparable_observations: 0,
                maximum_failed_or_partial: 0,
                maximum_queue_drops: 0,
                maximum_state_bytes: NonZeroUsize::new(64 * 1024).unwrap(),
            },
            storage: SloSelectedFeedbackStorageV1::Resume {
                path: PathBuf::from("/receipt.json"),
            },
        };
        let wire = serde_json::to_vec(&policy).unwrap();
        assert_eq!(
            serde_json::from_slice::<SloSelectedFeedbackPolicy>(&wire).unwrap(),
            policy
        );
        let mut config = SloCostObservationConfig {
            selected_feedback: policy,
            ..Default::default()
        };
        assert!(config.validate().is_err());
        config.predictor = SloCostPredictor::SelectedWholeWaveV1;
        config.validate().unwrap();
        let SloSelectedFeedbackPolicy::RetrospectiveFamilyMarginV1 { policy, .. } =
            &mut config.selected_feedback
        else {
            unreachable!()
        };
        policy.window_samples = NonZeroUsize::new(config.model.max_retained_samples.get()).unwrap();
        assert!(config.validate().is_err());
    }
}
