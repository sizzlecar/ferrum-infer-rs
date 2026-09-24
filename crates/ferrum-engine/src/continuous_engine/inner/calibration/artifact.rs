//! Training evidence exported at an accepted cut and loaded as a new model.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    CostProfileCutPaths, CostProfileCutReceipt, LoadedCalibrationProfile,
};
use ferrum_interfaces::execution_cost::CostObservationClock;
use ferrum_scheduler::implementations::continuous::cost_model::{
    CostPrediction, WaveExecutionShape,
};
use std::path::PathBuf;

/// New destinations; publication never overwrites existing artifacts.
#[derive(Debug, Clone)]
pub struct CalibrationProfilePaths {
    pub profile: PathBuf,
    pub source: PathBuf,
}

/// Actual published bytes. The cut is a retained training subset and does not
/// claim complete instrumentation, numerical validity or heldout coverage.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CalibrationProfileArtifact {
    pub accepted_ordinal: u64,
    pub profile: PathBuf,
    pub profile_sha256: String,
    pub profile_bytes: u64,
    pub source: PathBuf,
    pub source_sha256: String,
    pub source_digest: [u8; 32],
    pub source_bytes: u64,
    pub retained_samples: u64,
    pub raw_retained_observations: u64,
}

/// The product loader's interpretation of the exported profile. This is not
/// the online model frozen by freeze_cost_model: its calibration, floor and
/// retained support come from importing these exact recorded observations.
pub struct ImportedCalibrationModel {
    pub(super) selected_session: Option<Arc<()>>,
    pub(super) artifact: CalibrationProfileArtifact,
    pub(super) imported: LoadedCalibrationProfile,
    pub(super) clock: Arc<dyn CostObservationClock>,
    pub(super) audit: serde_json::Value,
}

impl ImportedCalibrationModel {
    pub fn planning_boundary(
        &self,
    ) -> ferrum_scheduler::implementations::continuous::cost_model::CostBoundary {
        self.imported.snapshot.planning_boundary()
    }
    pub fn artifact(&self) -> &CalibrationProfileArtifact {
        &self.artifact
    }
    pub fn import_receipt(&self) -> &ferrum_types::SloCostProfileReceipt {
        &self.imported.receipt
    }
    pub fn accepted_ordinal(&self) -> u64 {
        self.artifact.accepted_ordinal
    }
    pub fn model_version(&self) -> u64 {
        self.imported.snapshot.model_version()
    }
    pub fn audit(&self) -> &serde_json::Value {
        &self.audit
    }
    /// The original runtime clock keeps advancing imported sample age. This
    /// actual-shape diagnostic conveys no permission to execute a candidate.
    pub fn predict(&self, shape: &WaveExecutionShape) -> Result<CostPrediction> {
        let now = self.clock.now_ns().ok_or_else(|| {
            FerrumError::internal("imported calibration prediction clock unavailable")
        })?;
        let snapshot = self.imported.snapshot.as_ref();
        Ok(snapshot.predict(
            snapshot.fingerprint(),
            shape,
            snapshot.planning_boundary(),
            now,
        ))
    }
}

impl From<CostProfileCutReceipt> for CalibrationProfileArtifact {
    fn from(receipt: CostProfileCutReceipt) -> Self {
        Self {
            accepted_ordinal: receipt.accepted_ordinal,
            profile: receipt.profile,
            profile_sha256: receipt.profile_sha256,
            profile_bytes: receipt.profile_bytes,
            source: receipt.source,
            source_sha256: receipt.source_sha256,
            source_digest: receipt.source_digest,
            source_bytes: receipt.source_bytes,
            retained_samples: receipt.retained_samples,
            raw_retained_observations: receipt.raw_retained_observations,
        }
    }
}

impl CalibrationSession {
    /// Export the precise accepted training cut, then import the newly
    /// published profile through the same loader as ordinary product startup.
    /// The live worker keeps its own model and exporter. Failures preserve any
    /// already published source/profile bytes; no artifact is overwritten.
    pub async fn export_and_load_cost_profile(
        &mut self,
        paths: CalibrationProfilePaths,
    ) -> Result<ImportedCalibrationModel> {
        if self.pending.is_some() || self.indeterminate {
            return Err(FerrumError::invalid_request(
                "reap the calibration wave before exporting its training cut",
            ));
        }
        let runtime =
            Arc::clone(
                self.engine.inner.cost_runtime.as_ref().ok_or_else(|| {
                    FerrumError::internal("calibration cost runtime is unavailable")
                })?,
            );
        let checkpoint = runtime
            .request_profile_cut(CostProfileCutPaths {
                profile: paths.profile,
                source: paths.source,
            })
            .map_err(|error| {
                FerrumError::resource_exhausted(format!("calibration profile cut: {error}"))
            })?
            .wait()
            .await
            .map_err(|error| FerrumError::internal(format!("calibration profile cut: {error}")))?;
        let cut = checkpoint
            .profile_cut
            .ok_or_else(|| {
                FerrumError::internal("calibration checkpoint returned no requested profile export")
            })?
            .map_err(FerrumError::backend)?;
        if cut.accepted_ordinal != checkpoint.accepted_ordinal {
            return Err(FerrumError::internal(
                "calibration export cutoff differs from checkpoint",
            ));
        }
        let audit = serde_json::json!({
            "scope": "accepted training cut loaded through the product profile importer; not the live online snapshot; heldout coverage remains to be measured",
            "accepted_ordinal": checkpoint.accepted_ordinal,
            "training": checkpoint.training,
            "export": checkpoint.export,
        });
        let config = self
            .engine
            .inner
            .config
            .scheduler
            .slo
            .cost_observation
            .clone();
        let clock = Arc::clone(&runtime.clock);
        let (cut, imported) = tokio::task::spawn_blocking(move || {
            runtime
                .load_calibration_profile(&config, &cut)
                .map(|imported| (cut, imported))
        })
        .await
        .map_err(|error| {
            FerrumError::internal(format!("calibration product import task: {error}"))
        })??;
        Ok(ImportedCalibrationModel {
            selected_session: None,
            artifact: cut.into(),
            imported,
            clock,
            audit,
        })
    }
}
