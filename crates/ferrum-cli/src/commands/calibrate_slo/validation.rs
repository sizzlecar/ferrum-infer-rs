//! Exactly one immutable predictor for held-out requests. Model construction,
//! import, clock mapping and publication remain owned by the real engine.
use super::*;
use ferrum_engine::continuous_engine::{
    CalibrationProfilePaths, CalibrationSession, FrozenCalibrationModel, ImportedCalibrationModel,
};
use ferrum_scheduler::implementations::continuous::cost_model::{
    CostPrediction, WaveExecutionShape,
};

pub(super) enum ValidationModel {
    LiveFrozen(FrozenCalibrationModel),
    ExportedProfile(ImportedCalibrationModel),
    SelectedWholeWave(ImportedCalibrationModel, &'static str),
}

impl ValidationModel {
    pub(super) fn planning_boundary(
        &self,
    ) -> Option<ferrum_scheduler::implementations::continuous::cost_model::CostBoundary> {
        match self {
            Self::LiveFrozen(model) => model.planning_boundary(),
            Self::ExportedProfile(model) | Self::SelectedWholeWave(model, _) => {
                Some(model.planning_boundary())
            }
        }
    }
    pub(super) fn artifact(
        &self,
    ) -> Option<&ferrum_engine::continuous_engine::CalibrationProfileArtifact> {
        match self {
            Self::LiveFrozen(_) | Self::SelectedWholeWave(_, _) => None,
            Self::ExportedProfile(model) => Some(model.artifact()),
        }
    }
    pub(super) async fn prepare(
        session: &mut CalibrationSession,
        source: &manifest::ValidationSource,
    ) -> Result<Self> {
        match source {
            manifest::ValidationSource::StructuredWholeWaveGroupV2 { .. } | manifest::ValidationSource::RequiredFutureAuditV2 { .. } | manifest::ValidationSource::StructuredDiscoveryV2 { .. } | manifest::ValidationSource::StructuredWholeWaveV1 { .. } | manifest::ValidationSource::StructuredWholeWaveV2 {..} => Err(FerrumError::internal(
                "structured calibration must use its real three-phase session driver",
            )),
            manifest::ValidationSource::LiveFrozen => {
                session.freeze_cost_model().await.map(Self::LiveFrozen)
            }
            manifest::ValidationSource::ExportedProfile { profile, source } => session
                .export_and_load_cost_profile(CalibrationProfilePaths {
                    profile: profile.clone(),
                    source: source.clone(),
                })
                .await
                .map(Self::ExportedProfile),
            manifest::ValidationSource::SelectedWholeWaveV1 { .. } | manifest::ValidationSource::SelectedIndependentAttentionV2 { .. } | manifest::ValidationSource::SelectedWorkSupportV1 { .. } => Err(FerrumError::internal(
                "selected calibration must freeze fit and complete independent residual collection before import",
            )),
        }
    }

    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::LiveFrozen(_) => "live_frozen",
            Self::ExportedProfile(_) => "exported_profile",
            Self::SelectedWholeWave(_, kind) => kind,
        }
    }

    pub(super) fn predict(&self, shape: &WaveExecutionShape) -> Result<Option<CostPrediction>> {
        match self {
            Self::LiveFrozen(model) => model.predict(shape),
            Self::ExportedProfile(model) => model.predict(shape).map(Some),
            Self::SelectedWholeWave(_, _) => Err(FerrumError::internal(
                "selected whole-wave prediction requires its privately bound terminal wave evidence",
            )),
        }
    }

    pub(super) fn selected(&self) -> Option<&ImportedCalibrationModel> {
        match self {
            Self::SelectedWholeWave(model, _) => Some(model),
            _ => None,
        }
    }

    pub(super) fn receipt(&self) -> Result<serde_json::Value> {
        match self {
            Self::LiveFrozen(model) => Ok(serde_json::json!({
                "kind":self.kind(), "audit":model.audit()?,
                "profile_artifact":null,
                "scope":"immutable live online snapshot, diagnostic only; not a deployed artifact"
            })),
            Self::ExportedProfile(model) => Ok(serde_json::json!({
                "kind":self.kind(), "audit":model.audit(), "artifact":model.artifact(),
                "import_receipt":model.import_receipt(), "accepted_ordinal":model.accepted_ordinal(),
                "model_version":model.model_version(),
                "scope":"accepted training cut exported then loaded through the real product importer; validation uses only this immutable imported model"
            })),
            Self::SelectedWholeWave(model, _) => Ok(serde_json::json!({
                "kind":self.kind(), "audit":model.audit(), "artifact":model.artifact(),
                "import_receipt":model.import_receipt(), "accepted_ordinal":model.accepted_ordinal(),
                "model_version":model.model_version(),
                "scope":"fit frozen before independent residual collection; sealed explicitly versioned profile reloaded by the product importer; heldout never trains or republishes the model"
            })),
        }
    }
}
