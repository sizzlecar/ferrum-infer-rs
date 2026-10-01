use super::*;

/// Automatic numerical policy, distinct from historical native/source defaults.
/// Neither variant changes phase membership, TTL or the qualification rule that
/// every original heldout duration must fit the already frozen planning value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloAutomaticCalibrationPredictionMarginV1 {
    /// Preserve the existing fit-floor, empirical residual p99 and static margin.
    Disabled,
    /// Add the observed signed-residual span, frozen using only the independent
    /// Residual phase. Its cap is the native numerical maximum_wave_ns; exceeding
    /// that cap fails calibration instead of clipping the measured span. This
    /// empirical buffer supplies no future timing or statistical guarantee.
    #[default]
    ObservedResidualSpanV1,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn automatic_prediction_margin_is_typed_default_and_explicitly_disableable() {
        for partial in [json!({}), json!({"diagnostics":{"kind":"memory_only"}})] {
            let settings: SloAutomaticCalibrationSettingsV1 =
                serde_json::from_value(partial).unwrap();
            assert_eq!(
                settings.prediction_margin,
                SloAutomaticCalibrationPredictionMarginV1::ObservedResidualSpanV1
            );
            settings.validate().unwrap();
            assert_eq!(
                serde_json::from_value::<SloAutomaticCalibrationSettingsV1>(
                    serde_json::to_value(&settings).unwrap()
                )
                .unwrap(),
                settings
            );
        }
        let disabled: SloAutomaticCalibrationSettingsV1 =
            serde_json::from_value(json!({"prediction_margin":"disabled"})).unwrap();
        assert_eq!(
            disabled.prediction_margin,
            SloAutomaticCalibrationPredictionMarginV1::Disabled
        );
        disabled.validate().unwrap();
        for invalid in [
            json!("qualification_error"),
            json!({"kind":"observed_residual_span_v1","maximum_span_margin_ns":500000}),
        ] {
            assert!(
                serde_json::from_value::<SloAutomaticCalibrationPredictionMarginV1>(invalid)
                    .is_err()
            );
        }
    }
}
