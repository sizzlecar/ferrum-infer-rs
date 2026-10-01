//! Explicit owner-block prediction age policy, independent of collection work.
use super::*;

/// Historical manual/native declarations retain their absent-policy behavior.
/// This runtime default is explicitly bound into new automatic source7/source8
/// schedules; old profiles cannot acquire it through import.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloAutomaticCalibrationPredictionValidityV1 {
    /// Preserve the original model expiry clamp to the collection deadline.
    CollectionWindowV1,
    /// Finish all phases before the collection deadline, then retain only the
    /// validity derived from original member ages. This never refreshes TTL.
    #[default]
    OriginalSampleAgeV1,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automatic_prediction_validity_is_explicit_and_legacy_selection_roundtrips() {
        let defaults: SloAutomaticCalibrationSettingsV1 = serde_json::from_str("{}").unwrap();
        assert_eq!(
            defaults.prediction_validity,
            SloAutomaticCalibrationPredictionValidityV1::OriginalSampleAgeV1
        );
        let legacy: SloAutomaticCalibrationSettingsV1 =
            serde_json::from_str(r#"{"prediction_validity":"collection_window_v1"}"#).unwrap();
        legacy.validate().unwrap();
        assert_eq!(
            legacy.prediction_validity,
            SloAutomaticCalibrationPredictionValidityV1::CollectionWindowV1
        );
        assert_eq!(
            serde_json::from_slice::<SloAutomaticCalibrationSettingsV1>(
                &serde_json::to_vec(&legacy).unwrap()
            )
            .unwrap(),
            legacy
        );
        assert!(serde_json::from_str::<SloAutomaticCalibrationSettingsV1>(
            r#"{"prediction_validity":"renew_on_publication"}"#
        )
        .is_err());
        assert_eq!(defaults.maximum_window_ns, legacy.maximum_window_ns);
    }
}
