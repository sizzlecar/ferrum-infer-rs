//! Shared startup/runtime numerical policy. Historical native declarations
//! continue to deserialize their original defaults independently of this map.
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    StructuredLearnedDriftV2, StructuredSettingsV2,
};
use ferrum_types::{SloAutomaticCalibrationPredictionMarginV1, SloAutomaticCalibrationSettingsV1};
use std::num::NonZeroU64;

pub(in crate::continuous_engine::inner) fn automatic_numerical_settings(
    automatic: &SloAutomaticCalibrationSettingsV1,
) -> StructuredSettingsV2 {
    let mut numerical = StructuredSettingsV2::default();
    if automatic.population_schedule
        == ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1
    {
        return numerical;
    }
    numerical.learned_drift = match automatic.prediction_margin {
        SloAutomaticCalibrationPredictionMarginV1::Disabled => StructuredLearnedDriftV2::Disabled,
        SloAutomaticCalibrationPredictionMarginV1::ObservedResidualSpanV1 => {
            StructuredLearnedDriftV2::ObservedResidualSpanV1 {
                maximum_span_margin_ns: NonZeroU64::new(numerical.max_wave_ns)
                    .expect("native numerical default has a positive wave bound"),
            }
        }
    };
    numerical
}

/// Runtime policy mapping only. Historical fixed-window/native sources retain
/// their original expiry contract, and no already imported model is rewritten.
pub(in crate::continuous_engine::inner) fn automatic_prediction_validity(
    automatic: &SloAutomaticCalibrationSettingsV1,
) -> Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::OwnerPredictionValidityPolicyV1>{
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::OwnerPredictionValidityPolicyV1;
    use ferrum_types::{
        SloAutomaticCalibrationPopulationScheduleV1, SloAutomaticCalibrationPredictionValidityV1,
    };
    if automatic.population_schedule == SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1
    {
        return None;
    }
    match automatic.prediction_validity {
        SloAutomaticCalibrationPredictionValidityV1::CollectionWindowV1 => None,
        SloAutomaticCalibrationPredictionValidityV1::OriginalSampleAgeV1 => {
            Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_prediction_validity_maps_only_explicit_new_owner_sources() {
        use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::OwnerPredictionValidityPolicyV1;
        let mut settings = SloAutomaticCalibrationSettingsV1::default();
        assert_eq!(
            automatic_prediction_validity(&settings),
            Some(OwnerPredictionValidityPolicyV1::OriginalSampleAgeV1)
        );
        settings.prediction_validity =
            ferrum_types::SloAutomaticCalibrationPredictionValidityV1::CollectionWindowV1;
        assert_eq!(automatic_prediction_validity(&settings), None);
        settings.prediction_validity =
            ferrum_types::SloAutomaticCalibrationPredictionValidityV1::OriginalSampleAgeV1;
        settings.population_schedule =
            ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1;
        assert_eq!(automatic_prediction_validity(&settings), None);
    }

    #[test]
    fn automatic_prediction_margin_uses_existing_wave_cap_without_changing_native_defaults() {
        let automatic = SloAutomaticCalibrationSettingsV1::default();
        let settings = automatic_numerical_settings(&automatic);
        settings.validate().unwrap();
        assert_eq!(
            settings.learned_drift,
            StructuredLearnedDriftV2::ObservedResidualSpanV1 {
                maximum_span_margin_ns: NonZeroU64::new(settings.max_wave_ns).unwrap(),
            }
        );
        let original = StructuredSettingsV2::default();
        assert!(original.learned_drift.is_disabled());
        assert_eq!(settings.max_wave_ns, original.max_wave_ns);
        assert_eq!(settings.max_sample_age_ns, original.max_sample_age_ns);
        assert_eq!(settings.static_margin_ns, original.static_margin_ns);
        let disabled = automatic_numerical_settings(&SloAutomaticCalibrationSettingsV1 {
            prediction_margin: SloAutomaticCalibrationPredictionMarginV1::Disabled,
            ..automatic
        });
        assert_eq!(
            serde_json::to_vec(&disabled).unwrap(),
            serde_json::to_vec(&original).unwrap()
        );
        let historical = automatic_numerical_settings(&SloAutomaticCalibrationSettingsV1 {
            population_schedule:
                ferrum_types::SloAutomaticCalibrationPopulationScheduleV1::FixedWindowsV1,
            ..Default::default()
        });
        assert_eq!(
            serde_json::to_vec(&historical).unwrap(),
            serde_json::to_vec(&original).unwrap()
        );
    }
}
