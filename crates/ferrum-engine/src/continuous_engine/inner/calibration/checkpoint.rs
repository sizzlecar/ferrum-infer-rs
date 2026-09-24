//! An immutable training cut for independent calibration validation.
use super::*;
use crate::continuous_engine::inner::cost_observation::FrozenCostCheckpoint;
use ferrum_interfaces::execution_cost::CostObservationClock;
use ferrum_scheduler::implementations::continuous::cost_model::{
    CostPrediction, WaveExecutionShape,
};

/// This handle retains the original model and observation clock. Later
/// training cannot improve its predictions; freezing cannot renew sample age.
/// Its queries are diagnostic and confer no permission to execute work.
pub struct FrozenCalibrationModel {
    checkpoint: FrozenCostCheckpoint,
    clock: Arc<dyn CostObservationClock>,
}

impl FrozenCalibrationModel {
    pub fn planning_boundary(
        &self,
    ) -> Option<ferrum_scheduler::implementations::continuous::cost_model::CostBoundary> {
        self.checkpoint
            .snapshot
            .as_ref()
            .map(|model| model.planning_boundary())
    }
    pub(in crate::continuous_engine) fn new(
        checkpoint: FrozenCostCheckpoint,
        clock: Arc<dyn CostObservationClock>,
    ) -> Self {
        Self { checkpoint, clock }
    }

    /// Successfully queued observation positions, including training rejects.
    /// Queue losses and uninstrumented calls are outside this denominator.
    pub fn accepted_ordinal(&self) -> u64 {
        self.checkpoint.accepted_ordinal
    }

    pub fn model_version(&self) -> Option<u64> {
        self.checkpoint
            .snapshot
            .as_ref()
            .map(|model| model.model_version())
    }

    /// Query at the current original runtime clock, never a caller-supplied
    /// historical timestamp. None means no model had been published at the cut.
    /// A retrospectively supplied actual shape is not a pre-execution forecast.
    pub fn predict(&self, shape: &WaveExecutionShape) -> Result<Option<CostPrediction>> {
        let Some(model) = self.checkpoint.snapshot.as_ref() else {
            return Ok(None);
        };
        let now = self
            .clock
            .now_ns()
            .ok_or_else(|| FerrumError::internal("calibration prediction clock unavailable"))?;
        Ok(Some(model.predict(
            model.fingerprint(),
            shape,
            model.planning_boundary(),
            now,
        )))
    }

    /// Frozen worker counters at this accepted-observation cut. Export status
    /// can still be Active; this does not finalize or rotate the live exporter.
    pub fn audit(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "accepted_ordinal": self.checkpoint.accepted_ordinal,
            "model_version": self.model_version(),
            "training": self.checkpoint.training,
            "export": self.checkpoint.export,
        }))
    }
}

impl CalibrationSession {
    /// Drain all cost observations accepted before the barrier and freeze
    /// their resulting model. A dropped waiter does not withdraw the barrier.
    /// No wave may be in flight when an independent validation phase starts.
    pub async fn freeze_cost_model(&mut self) -> Result<FrozenCalibrationModel> {
        if self.pending.is_some() || self.indeterminate {
            return Err(FerrumError::invalid_request(
                "reap the calibration wave before freezing its cost model",
            ));
        }
        let runtime = self
            .engine
            .inner
            .cost_runtime
            .as_ref()
            .ok_or_else(|| FerrumError::internal("calibration cost runtime is unavailable"))?;
        let waiter = runtime.request_checkpoint().map_err(|error| {
            FerrumError::resource_exhausted(format!("calibration checkpoint: {error}"))
        })?;
        let checkpoint = waiter
            .wait()
            .await
            .map_err(|error| FerrumError::internal(format!("calibration checkpoint: {error}")))?;
        Ok(FrozenCalibrationModel::new(
            checkpoint,
            Arc::clone(&runtime.clock),
        ))
    }
}
