use super::{SloCostObservationConfig, SloCostPredictor};
use serde::{Deserialize, Serialize};

/// Per-wave provenance only. This does not declare source3 cohort membership,
/// fit a model, extend its TTL, or authorize catalog replacement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloProspectiveStructuredCapture {
    #[default]
    Disabled,
    /// Freeze every published structured V2 cost witness before preparation.
    /// Completion-only and legacy dispatches remain outside this population.
    ReplayedFirstWaveV1,
}
impl SloProspectiveStructuredCapture {
    pub fn is_disabled(&self) -> bool {
        *self == Self::Disabled
    }
    pub(super) fn validate(&self, config: &SloCostObservationConfig) -> Result<(), String> {
        if !self.is_disabled()
            && (config.predictor != SloCostPredictor::StructuredWholeWaveV2
                || config.structured_capture.is_disabled())
        {
            return Err("prospective structured capture requires structured_whole_wave_v2 and host-settled capture".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prospective_capture_is_explicit_typed_and_requires_existing_settlement() {
        let default = SloCostObservationConfig::default();
        assert!(default.prospective_structured_capture.is_disabled());
        assert!(serde_json::to_value(&default)
            .unwrap()
            .get("prospective_structured_capture")
            .is_none());
        let mut config = SloCostObservationConfig::structured_whole_wave_v2();
        config.prospective_structured_capture =
            SloProspectiveStructuredCapture::ReplayedFirstWaveV1;
        config.validate().unwrap();
        let wire = serde_json::to_vec(&config).unwrap();
        assert_eq!(
            serde_json::from_slice::<SloCostObservationConfig>(&wire).unwrap(),
            config
        );
        config.predictor = SloCostPredictor::LegacyFeatureModel;
        assert!(config.validate().is_err());
        assert!(serde_json::from_str::<SloProspectiveStructuredCapture>("\"source3\"").is_err());
    }
}
