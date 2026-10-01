use super::{SloCostObservationConfig, SloStructuredCostCapture};
use serde::{Deserialize, Serialize};

/// Controls dynamic actual samples, independently of future-route evidence and
/// templates retained during the first real executable capture. Legacy callers
/// and omitted configuration retain their existing every-wave behavior.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloStructuredActualCapturePolicy {
    #[default]
    LegacyEveryWave,
    ConsumerDrivenV1,
}

impl SloStructuredActualCapturePolicy {
    pub fn is_legacy(&self) -> bool {
        *self == Self::LegacyEveryWave
    }
}

impl SloCostObservationConfig {
    /// Fixed consumers require every-wave evidence. Live service windows add
    /// their actual reserved tickets at runtime; installed automatic feedback
    /// adds its coherent source-generation demand there as well. Manual
    /// calibration still needs discovery receipts before a collector is opened.
    pub fn needs_structured_actual_sample(&self, manual_calibration: bool) -> bool {
        self.structured_capture == SloStructuredCostCapture::HostSettledV1
            && (self.structured_actual_capture.is_legacy()
                || manual_calibration
                || self.profile_export.is_some()
                || self.predictor.is_selected()
                || !self.selected_feedback.is_disabled()
                || !self.structured_feedback.is_disabled()
                || !self.prospective_structured_capture.is_disabled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_capture_defaults_preserve_wire_and_route_capability() {
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        assert!(config.needs_structured_actual_sample(false));
        let old = serde_json::to_value(&config).unwrap();
        assert!(old.get("structured_actual_capture").is_none());
        assert_eq!(
            serde_json::from_value::<SloCostObservationConfig>(old).unwrap(),
            config
        );
        config.structured_actual_capture = SloStructuredActualCapturePolicy::ConsumerDrivenV1;
        config.validate().unwrap();
        assert!(!config.needs_structured_actual_sample(false));
        assert!(config.needs_structured_actual_sample(true));
        assert_eq!(
            config.structured_capture,
            SloStructuredCostCapture::HostSettledV1
        );
        assert_eq!(
            serde_json::from_slice::<SloCostObservationConfig>(
                &serde_json::to_vec(&config).unwrap()
            )
            .unwrap(),
            config
        );
        config.structured_capture = SloStructuredCostCapture::Disabled;
        assert!(config.validate().is_err());
        assert!(!config.needs_structured_actual_sample(true));
    }

    #[test]
    fn prospective_consumer_requires_actual_samples_in_consumer_driven_mode() {
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        config.structured_actual_capture = SloStructuredActualCapturePolicy::ConsumerDrivenV1;
        config.prospective_structured_capture =
            super::super::SloProspectiveStructuredCapture::ReplayedFirstWaveV1;
        config.validate().unwrap();
        assert!(config.needs_structured_actual_sample(false));
    }

    #[test]
    fn consumer_driven_live_calibration_requires_runtime_ticket_or_feedback_demand() {
        use std::num::{NonZeroU64, NonZeroUsize};

        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        config.structured_actual_capture = SloStructuredActualCapturePolicy::ConsumerDrivenV1;
        assert!(!config.needs_structured_actual_sample(false));
        config.live_structured_calibration =
            super::super::SloLiveStructuredCalibration::ServiceWindowsV1 {
                declaration: "declaration.json".into(),
                evidence_directory: "evidence".into(),
                maximum_generations: NonZeroUsize::new(4).unwrap(),
                maximum_source_bytes: NonZeroU64::new(1 << 28).unwrap(),
                maximum_retained_numeric_bytes: NonZeroUsize::new(1 << 26).unwrap(),
            };
        config.validate().unwrap();
        assert!(!config.needs_structured_actual_sample(false));
        assert!(config.needs_structured_actual_sample(true));
        config.live_structured_calibration =
            super::super::SloLiveStructuredCalibration::AutomaticV1 {
                settings: Default::default(),
            };
        config.validate().unwrap();
        assert!(!config.needs_structured_actual_sample(false));
        config.structured_actual_capture = SloStructuredActualCapturePolicy::LegacyEveryWave;
        assert!(config.needs_structured_actual_sample(false));

        config.structured_capture = SloStructuredCostCapture::Disabled;
        assert!(!config.needs_structured_actual_sample(false));
        assert!(config.validate().is_err());
    }

    #[test]
    fn reference_export_remains_an_actual_consumer_without_structured_predictor() {
        let config = SloCostObservationConfig {
            structured_capture: SloStructuredCostCapture::HostSettledV1,
            structured_actual_capture: SloStructuredActualCapturePolicy::ConsumerDrivenV1,
            profile_export: Some(super::super::SloCostProfileExportConfig {
                path: "/profile.json".into(),
                observations_path: "/source.jsonl".into(),
                declared_clock_max_error_ns: Some(1),
                ..Default::default()
            }),
            ..Default::default()
        };
        config.validate().unwrap();
        assert!(config.needs_structured_actual_sample(false));
    }
}
