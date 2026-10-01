use super::{SloCostObservationConfig, SloCostPredictor, SloStructuredCostCapture};
use serde::{Deserialize, Serialize};
use std::{num::NonZeroU64, num::NonZeroUsize, path::PathBuf};
mod route_population;
pub use route_population::*;
mod automatic;
pub use automatic::*;

/// Ordinary service-wave evidence, independent of the manual complete-request
/// calibration protocols. Enabling collection does not qualify any cost model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SloLiveStructuredCalibration {
    #[default]
    #[serde(deserialize_with = "super::feedback::deserialize_disabled")]
    Disabled,
    /// Automatic discovery and independent service windows. Configuration alone
    /// grants no profile, reference, forecast, or SLO readiness.
    AutomaticV1 {
        #[serde(default)]
        settings: SloAutomaticCalibrationSettingsV1,
    },
    ServiceWindowsV1 {
        /// Immutable source6 owner/scope and offered-window declaration.
        declaration: PathBuf,
        /// New generations use create-new files, never replace old evidence.
        evidence_directory: PathBuf,
        maximum_generations: NonZeroUsize,
        maximum_source_bytes: NonZeroU64,
        maximum_retained_numeric_bytes: NonZeroUsize,
    },
}
impl SloLiveStructuredCalibration {
    pub fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }

    pub(super) fn validate(&self, config: &SloCostObservationConfig) -> Result<(), String> {
        if self.is_disabled() {
            return Ok(());
        }
        if config.predictor != SloCostPredictor::StructuredWholeWaveV2
            || config.structured_capture != SloStructuredCostCapture::HostSettledV1
        {
            return Err("live structured calibration requires V2 host-settled evidence".into());
        }
        match self {
            Self::Disabled => Ok(()),
            Self::AutomaticV1 { settings } => settings.validate(),
            Self::ServiceWindowsV1 {
                declaration,
                evidence_directory,
                maximum_generations,
                maximum_source_bytes,
                maximum_retained_numeric_bytes,
            } => {
                if declaration.as_os_str().is_empty()
                    || evidence_directory.as_os_str().is_empty()
                    || declaration == evidence_directory
                    || maximum_generations.get() > 65_536
                    || maximum_source_bytes.get() > 8 * 1024 * 1024 * 1024
                    || maximum_retained_numeric_bytes.get() > 512 * 1024 * 1024
                {
                    return Err("live structured calibration requires distinct declared paths and bounded generation/source/numeric capacities".into());
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled() -> SloLiveStructuredCalibration {
        SloLiveStructuredCalibration::ServiceWindowsV1 {
            declaration: "declared.json".into(),
            evidence_directory: "new-evidence".into(),
            maximum_generations: NonZeroUsize::new(4).unwrap(),
            maximum_source_bytes: NonZeroU64::new(1 << 28).unwrap(),
            maximum_retained_numeric_bytes: NonZeroUsize::new(1 << 26).unwrap(),
        }
    }

    #[test]
    fn live_calibration_default_wire_and_typed_capacity() {
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        let old = serde_json::to_value(&config).unwrap();
        assert!(old.get("live_structured_calibration").is_none());
        config.live_structured_calibration = enabled();
        config.validate().unwrap();
        assert_eq!(
            serde_json::from_value::<SloCostObservationConfig>(
                serde_json::to_value(&config).unwrap()
            )
            .unwrap(),
            config
        );
        config.predictor = SloCostPredictor::LegacyFeatureModel;
        assert!(config.validate().is_err());
        config.predictor = SloCostPredictor::StructuredWholeWaveV2;
        if let SloLiveStructuredCalibration::ServiceWindowsV1 {
            maximum_retained_numeric_bytes,
            ..
        } = &mut config.live_structured_calibration
        {
            *maximum_retained_numeric_bytes = NonZeroUsize::new(512 * 1024 * 1024 + 1).unwrap();
        }
        assert!(config.validate().is_err());
    }
}
