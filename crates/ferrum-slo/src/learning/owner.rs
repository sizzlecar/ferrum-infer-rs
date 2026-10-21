use super::{
    snapshot, state, LearningConfig, LearningStateError, StateLoadReport, StateSaveReport,
};
use crate::{
    model::ModelFamily, EstimatorError, ReferenceEstimatorOwner, ReferenceEstimatorStats,
    WorkerExit,
};
use ferrum_execution::execution_attempt::ExecutionAttemptObserver;
use ferrum_scheduler::cost_prediction::CostSnapshotSource;
use std::{path::Path, sync::Arc};

/// Numerical state qualification is not comparative performance qualification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "resolution")]
pub enum FamilyResolution {
    Explicit { family: ModelFamily },
    AutoRestoredBoundState { family: ModelFamily },
    AutoPhysicalFallback { reason: AutoFallbackReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum AutoFallbackReason {
    NoBoundStateProvided,
    MissingState,
    StaleState,
}

/// Sole learning worker lifecycle owner plus immutable state persistence.
/// Saving is an explicit caller operation and never runs on an inference path.
pub struct LearningEstimatorOwner {
    pub(super) common: ReferenceEstimatorOwner,
    pub(super) source: Arc<snapshot::Source>,
    pub(super) config: LearningConfig,
    pub(super) family: ModelFamily,
    pub(super) resolution: FamilyResolution,
    pub(super) telemetry: Arc<super::telemetry::Telemetry>,
    pub(super) load_report: Option<StateLoadReport>,
}

impl LearningEstimatorOwner {
    pub fn common_owner(&self) -> &ReferenceEstimatorOwner {
        &self.common
    }
    pub fn observer(&self) -> Arc<dyn ExecutionAttemptObserver> {
        Arc::new(super::telemetry::Observer {
            target: self.common.observer(),
            source: self.source.clone(),
            telemetry: self.telemetry.clone(),
        })
    }
    pub fn source(&self) -> Arc<dyn CostSnapshotSource> {
        self.common.source()
    }
    pub fn stats(&self) -> ReferenceEstimatorStats {
        self.common.stats()
    }
    pub fn prediction_stats(&self) -> super::LearningPredictionStats {
        self.telemetry.stats()
    }
    pub fn family(&self) -> ModelFamily {
        self.family
    }
    pub fn family_resolution(&self) -> FamilyResolution {
        self.resolution
    }
    pub fn load_report(&self) -> Option<StateLoadReport> {
        self.load_report
    }
    pub async fn shutdown(&self) -> Result<WorkerExit, EstimatorError> {
        self.common.shutdown().await
    }

    pub fn save_state(&self, path: &Path) -> Result<StateSaveReport, LearningStateError> {
        state::save(path, &self.source.load_typed(), &self.config, self.family)
    }

    pub async fn shutdown_and_save(
        &self,
        path: &Path,
    ) -> Result<(WorkerExit, StateSaveReport), LearningStateError> {
        let exit = self.shutdown().await?;
        Ok((exit, self.save_state(path)?))
    }
}
