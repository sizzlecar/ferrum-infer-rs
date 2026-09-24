//! Slow-path product-loader import of an immutable accepted-cut artifact.
use super::*;
use crate::continuous_engine::inner::cost_observation::CostProfileCutReceipt;

pub(in crate::continuous_engine) struct LoadedCalibrationProfile {
    pub snapshot: Arc<EngineCostSnapshot>,
    pub receipt: ferrum_types::SloCostProfileReceipt,
}

impl EngineCostRuntime {
    /// The input is the worker's actual publication receipt, never a path or
    /// predicted model supplied by the external calibration caller.
    pub fn load_calibration_profile(
        &self,
        config: &SloCostObservationConfig,
        cut: &CostProfileCutReceipt,
    ) -> Result<LoadedCalibrationProfile, FerrumError> {
        let clock = profile::read_load_clock(self.clock.as_ref(), &config.profile_import)?;
        let seed = profile::load_seed(&self.identity, config, Some(&cut.profile), Some(clock))?;
        let receipt = seed.receipt.ok_or_else(|| {
            FerrumError::internal("calibration artifact import returned no provenance receipt")
        })?;
        if receipt.file_sha256.strip_prefix("sha256:") != Some(cut.profile_sha256.as_str())
            || u64::try_from(receipt.file_bytes).ok() != Some(cut.profile_bytes)
            || receipt.source_observation_artifact_sha256 != cut.source_digest
        {
            return Err(FerrumError::config(
                "calibration imported artifact differs from the worker's published cut",
            ));
        }
        let snapshot = seed.snapshot.ok_or_else(|| {
            FerrumError::internal("calibration artifact import returned no model")
        })?;
        // The imported trainer is deliberately dropped. Heldout observations
        // may update the live worker, but can never update this independent Arc.
        Ok(LoadedCalibrationProfile { snapshot, receipt })
    }
}
