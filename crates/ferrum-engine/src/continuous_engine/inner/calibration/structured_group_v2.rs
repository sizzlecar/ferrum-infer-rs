//! Bounded group observers on the existing real calibration execution path.
use super::super::cost_observation::StructuredCalibrationGroupV2;
pub use super::super::cost_observation::{
    StructuredCalibrationGroupArtifactV2, StructuredCalibrationGroupLimitsV2,
    StructuredCalibrationGroupOptionsV2,
};
use super::structured::capture_error;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredCoverageReportV2;

impl CalibrationSession {
    pub async fn begin_structured_cost_group_v2(
        &mut self,
        options: StructuredCalibrationGroupOptionsV2,
    ) -> Result<()> {
        self.selected_phase_boundary()?;
        if self.structured_group_v2.is_some()
            || self.structured_capture_v2.is_some()
            || self.structured_capture.is_some()
            || self.selected_capture_identity.is_some()
        {
            return Err(FerrumError::invalid_request(
                "a calibration collector is already active",
            ));
        }
        if self
            .configuration()
            .scheduler
            .slo
            .cost_observation
            .structured_capture
            != ferrum_types::SloStructuredCostCapture::HostSettledV1
        {
            return Err(FerrumError::config(
                "group calibration requires capture configured before engine creation",
            ));
        }
        let checkpoint = self.freeze_cost_model().await?;
        let runtime = self
            .engine
            .inner
            .cost_runtime
            .as_ref()
            .ok_or_else(|| FerrumError::internal("cost runtime unavailable"))?;
        let ferrum_interfaces::execution_cost::ExecutorCostIdentityAvailability::Known(identity) =
            &runtime.identity
        else {
            return Err(FerrumError::unsupported(
                "group calibration requires real executor identity",
            ));
        };
        if identity.schema_version
            != ferrum_interfaces::execution_cost::EXECUTOR_COST_IDENTITY_SCHEMA
        {
            return Err(FerrumError::unsupported(
                "group executor identity schema mismatch",
            ));
        }
        let fingerprint =
            ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            };
        self.structured_group_v2 = Some(
            StructuredCalibrationGroupV2::new(
                options,
                fingerprint,
                Arc::clone(&runtime.clock),
                checkpoint.accepted_ordinal(),
            )
            .map_err(capture_error)?,
        );
        Ok(())
    }
    pub fn structured_cost_group_progress_v2(&self) -> Option<Vec<StructuredCalibrationProgress>> {
        self.structured_group_v2
            .as_ref()
            .map(|group| group.progress())
    }
    pub fn structured_cost_group_coverage_v2(&self) -> Result<Vec<StructuredCoverageReportV2>> {
        self.structured_group_v2
            .as_ref()
            .ok_or_else(|| FerrumError::invalid_request("group not started"))?
            .coverage()
            .map_err(capture_error)
    }
    pub fn begin_structured_cost_group_cohort_v2(&mut self, ordinal: usize) -> Result<()> {
        self.selected_phase_boundary()?;
        self.structured_group_v2
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("group not started"))?
            .begin_cohort(ordinal)
            .map_err(capture_error)
    }
    pub fn end_structured_cost_group_cohort_v2(&mut self) -> Result<()> {
        self.selected_phase_boundary()?;
        self.structured_group_v2
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("group not started"))?
            .end_cohort()
            .map_err(capture_error)
    }
    pub async fn freeze_structured_cost_group_phase_v2(
        &mut self,
    ) -> Result<Vec<StructuredPhaseFreezeReceipt>> {
        self.selected_phase_boundary()?;
        let checkpoint = self.freeze_cost_model().await?;
        self.structured_group_v2
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("group not started"))?
            .freeze(checkpoint.accepted_ordinal())
            .map_err(capture_error)
    }
    pub async fn finish_structured_cost_group_v2(
        &mut self,
    ) -> Result<StructuredCalibrationGroupArtifactV2> {
        self.selected_phase_boundary()?;
        let checkpoint = self.freeze_cost_model().await?;
        self.structured_group_v2
            .take()
            .ok_or_else(|| FerrumError::invalid_request("group not started"))?
            .finish(checkpoint.accepted_ordinal())
            .map_err(capture_error)
    }
}
