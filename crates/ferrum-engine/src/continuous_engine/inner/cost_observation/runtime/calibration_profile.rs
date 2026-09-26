//! Slow-path product-loader import of an immutable accepted-cut artifact.
use super::*;
use crate::continuous_engine::inner::cost_observation::CostProfileCutReceipt;

pub(in crate::continuous_engine) struct LoadedCalibrationProfile {
    pub snapshot: Arc<EngineCostSnapshot>,
    pub receipt: ferrum_types::SloCostProfileReceipt,
}

#[cfg(test)]
mod tests;

impl EngineCostRuntime {
    /// Read-only verification through the exact product loader, original live
    /// identity and clock. No trainer, snapshot installation or TTL renewal.
    pub fn inspect_structured_cost_profile_v2(
        &self,
        config: &SloCostObservationConfig,
        path: &Path,
    ) -> Result<ferrum_types::SloCostProfileReceipt, FerrumError> {
        if config.predictor != ferrum_types::SloCostPredictor::StructuredWholeWaveV2 {
            return Err(FerrumError::config(
                "structured profile inspection requires the V2 predictor",
            ));
        }
        config.validate().map_err(FerrumError::config)?;
        let clock = profile::read_load_clock(self.clock.as_ref(), &config.profile_import)?;
        let seed = profile::load_seed(&self.identity, config, Some(path), Some(clock))?;
        seed.receipt
            .filter(|r| r.structured_whole_wave_v2.is_some())
            .ok_or_else(|| {
                FerrumError::config("structured profile inspection has no V2 import receipt")
            })
    }

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
