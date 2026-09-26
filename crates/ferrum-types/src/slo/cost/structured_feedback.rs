use super::*;

/// Runtime corrections for an already qualified source3/profile10 snapshot.
/// The immutable fit, support, qualification and original TTL do not change.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloStructuredFeedbackPolicy {
    #[default]
    #[serde(deserialize_with = "super::feedback::deserialize_disabled")]
    Disabled,
    RetrospectiveOwnerMarginV1 {
        /// Reuses the bounded feedback policy. Here a family is one complete
        /// structured owner/domain, never a row, token or partial template.
        policy: SloSelectedFeedbackSettingsV1,
        storage: SloSelectedFeedbackStorageV1,
    },
}
impl SloStructuredFeedbackPolicy {
    pub fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
    pub(super) fn validate(&self, config: &SloCostObservationConfig) -> Result<(), String> {
        let Self::RetrospectiveOwnerMarginV1 { policy, storage } = self else {
            return Ok(());
        };
        if config.predictor != SloCostPredictor::StructuredWholeWaveV2
            || !config.selected_feedback.is_disabled()
        {
            return Err(
                "structured owner feedback requires only the structured whole-wave V2 predictor"
                    .into(),
            );
        }
        // Source-frozen per-child limits are checked again after real import;
        // legacy model defaults are not a replacement for those limits.
        super::feedback::validate_feedback_bounds(
            policy,
            storage,
            128,
            128 * 4096,
            1 << 53,
            u64::MAX,
            config.profile_import.max_file_bytes.get(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SloStructuredFeedbackPolicy {
        SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 {
            policy: SloSelectedFeedbackSettingsV1 {
                window_samples: NonZeroUsize::new(2).unwrap(),
                minimum_underestimates: NonZeroUsize::new(2).unwrap(),
                minimum_consecutive_underestimates: NonZeroUsize::MIN,
                trigger_excess_ns: NonZeroU64::new(2).unwrap(),
                correction_padding_ns: 1,
                maximum_family_margin_ns: NonZeroU64::new(100).unwrap(),
                maximum_consumption_lag_ns: NonZeroU64::new(50).unwrap(),
                maximum_uncomparable_observations: 0,
                maximum_failed_or_partial: 0,
                maximum_queue_drops: 0,
                maximum_state_bytes: NonZeroUsize::new(65536).unwrap(),
            },
            storage: SloSelectedFeedbackStorageV1::CreateNew {
                path: "/structured-feedback.json".into(),
            },
        }
    }
    #[test]
    fn structured_feedback_is_explicit_default_off_and_strict() {
        let old: SloCostObservationConfig = serde_json::from_str("{}").unwrap();
        assert!(old.structured_feedback.is_disabled());
        assert!(serde_json::to_value(old)
            .unwrap()
            .get("structured_feedback")
            .is_none());
        let policy = policy();
        assert_eq!(
            serde_json::from_slice::<SloStructuredFeedbackPolicy>(
                &serde_json::to_vec(&policy).unwrap()
            )
            .unwrap(),
            policy
        );
        for wire in [
            r#"{"kind":"disabled","policy":{}}"#,
            r#"{"kind":"retrospective_owner_margin_v1","policy":{},"storage":{"mode":"create_new","path":"/receipt"}}"#,
            r#"{"kind":"assume_qualified"}"#,
        ] {
            assert!(serde_json::from_str::<SloStructuredFeedbackPolicy>(wire).is_err());
        }
    }
    #[test]
    fn structured_feedback_requires_v2_and_keeps_bounded_declared_policy() {
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        config.structured_feedback = policy();
        config.validate().unwrap();
        for predictor in [
            SloCostPredictor::LegacyFeatureModel,
            SloCostPredictor::SelectedWholeWaveV1,
            SloCostPredictor::StructuredWholeWaveV1,
        ] {
            config.predictor = predictor;
            assert!(config.validate().is_err());
        }
        config.predictor = SloCostPredictor::StructuredWholeWaveV2;
        let SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 { policy, .. } =
            &mut config.structured_feedback
        else {
            unreachable!()
        };
        policy.window_samples = NonZeroUsize::new(4097).unwrap();
        assert!(config.validate().is_err());
        config.structured_feedback = self::policy();
        let SloStructuredFeedbackPolicy::RetrospectiveOwnerMarginV1 { storage, .. } =
            &mut config.structured_feedback
        else {
            unreachable!()
        };
        *storage = SloSelectedFeedbackStorageV1::CreateNew {
            path: "relative.json".into(),
        };
        assert!(config.validate().is_err());
    }
}
