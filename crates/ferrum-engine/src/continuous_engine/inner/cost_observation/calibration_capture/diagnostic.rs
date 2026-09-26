//! Bounded diagnostic projection of the original recorder. It neither supplies
//! a missing shape nor changes the sample/source eligibility decision.
use super::*;
use serde::{Serialize, Serializer};

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationActualWaveUnknown {
    pub physical_wave_ordinal: u32,
    #[serde(serialize_with = "serialize_reason")]
    pub reason: ActualWaveEvidenceUnknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationActualEvidenceDiagnostic {
    pub call_id: u64,
    pub physical_waves: usize,
    pub retained_waves: usize,
    pub lost_observations: u64,
    #[serde(serialize_with = "serialize_optional_reason")]
    pub dispatch_unknown: Option<ActualWaveEvidenceUnknown>,
    /// Only retained unknown-shape observations, in original ordinal order.
    /// No observation is fabricated to fill a lost recorder slot.
    pub waves: Vec<CalibrationActualWaveUnknown>,
    pub retained_wave_details_complete: bool,
}

fn serialize_reason<S: Serializer>(
    reason: &ActualWaveEvidenceUnknown,
    s: S,
) -> Result<S::Ok, S::Error> {
    s.serialize_str(&format!("{reason:?}"))
}
fn serialize_optional_reason<S: Serializer>(
    reason: &Option<ActualWaveEvidenceUnknown>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match reason {
        Some(reason) => s.serialize_some(&format!("{reason:?}")),
        None => s.serialize_none(),
    }
}

impl EngineCostCall {
    pub(in crate::continuous_engine::inner::cost_observation) fn capture_actual_unknown_diagnostic(
        &self,
    ) {
        let Some(capture) = &self.calibration_capture else {
            return;
        };
        let observations = self.recorder.observations();
        let lost_observations = match self.recorder.coverage() {
            CostObservationCoverage::Unknown {
                lost_observations, ..
            } => lost_observations,
            CostObservationCoverage::Complete => 0,
        };
        let unknown_count = observations
            .iter()
            .filter(|wave| wave.shape_unknown.is_some())
            .count();
        if self.dispatch.unknown.is_none() && unknown_count == 0 && lost_observations == 0 {
            return;
        }
        // The recorder already enforces max_waves <= 4096 before dispatch.
        // Do not copy rows/commands or enlarge that limit for diagnostics.
        let mut waves = Vec::new();
        let retained_wave_details_complete = waves.try_reserve_exact(unknown_count).is_ok();
        if retained_wave_details_complete {
            waves.extend(observations.iter().filter_map(|wave| {
                wave.shape_unknown
                    .map(|reason| CalibrationActualWaveUnknown {
                        physical_wave_ordinal: wave.physical_wave_ordinal,
                        reason,
                    })
            }));
        }
        let diagnostic = CalibrationActualEvidenceDiagnostic {
            call_id: self.call_id.get(),
            physical_waves: self.dispatch.waves,
            retained_waves: observations.len(),
            lost_observations,
            dispatch_unknown: self.dispatch.unknown,
            waves,
            retained_wave_details_complete,
        };
        if capture
            .actual_evidence_diagnostic
            .set(Arc::new(diagnostic))
            .is_err()
        {
            capture.mark_conflict();
        }
    }
}
