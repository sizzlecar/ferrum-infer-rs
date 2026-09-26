//! V2 capture is an observer on the original manual executor path. It cannot
//! submit waves, construct receipts or terminate requests at a sample count.
use super::super::cost_observation::StructuredCalibrationCollectorV2;
pub use super::super::cost_observation::{
    StructuredCalibrationArtifactV2, StructuredCalibrationOptionsV2,
};
use super::structured::capture_error;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredCoverageReportV2;

impl CalibrationSession {
    pub async fn begin_structured_cost_calibration_v2(
        &mut self,
        options: StructuredCalibrationOptionsV2,
    ) -> Result<()> {
        self.selected_phase_boundary()?;
        if self.structured_capture_v2.is_some()
            || self.structured_group_v2.is_some()
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
                "V2 calibration requires capture configured before engine creation",
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
                "V2 calibration requires real executor identity",
            ));
        };
        if identity.schema_version
            != ferrum_interfaces::execution_cost::EXECUTOR_COST_IDENTITY_SCHEMA
        {
            return Err(FerrumError::unsupported(
                "V2 executor identity schema mismatch",
            ));
        }
        let fingerprint =
            ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            };
        self.structured_capture_v2 = Some(
            StructuredCalibrationCollectorV2::new(
                options,
                fingerprint,
                Arc::clone(&runtime.clock),
                checkpoint.accepted_ordinal(),
            )
            .map_err(capture_error)?,
        );
        Ok(())
    }
    pub fn structured_cost_progress_v2(&self) -> Option<StructuredCalibrationProgress> {
        self.structured_capture_v2.as_ref().map(|c| c.progress())
    }
    pub fn structured_cost_coverage_v2(&self) -> Result<StructuredCoverageReportV2> {
        self.structured_capture_v2
            .as_ref()
            .ok_or_else(|| FerrumError::invalid_request("V2 collector not started"))?
            .coverage()
            .map_err(capture_error)
    }
    pub fn begin_structured_cost_cohort_v2(&mut self, ordinal: usize) -> Result<()> {
        self.selected_phase_boundary()?;
        let collector = self
            .structured_capture_v2
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("V2 collector not started"))?;
        let result = collector.begin_cohort(ordinal);
        if let Err(error) = &result {
            collector.invalidate(error.to_string());
        }
        result.map_err(capture_error)
    }
    pub fn end_structured_cost_cohort_v2(&mut self) -> Result<()> {
        // The CLI drains credited terminal frames first. Independently require
        // actual owners/scheduler work gone and every slot's real Length receipt.
        self.selected_phase_boundary()?;
        let collector = self
            .structured_capture_v2
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("V2 collector not started"))?;
        let result = collector.end_cohort();
        if let Err(error) = &result {
            collector.invalidate(error.to_string());
        }
        result.map_err(capture_error)
    }
    pub(super) fn record_structured_v2_capture(
        &mut self,
        receipt: &CalibrationWaveReceipt,
        report: &CalibrationWaveReport,
    ) {
        if let Some(group) = &mut self.structured_group_v2 {
            if group.collecting() && !group.prefix_pending() {
                if let Err(error) = group.complete(
                    receipt.capture(),
                    report.submission == CalibrationSubmissionState::HostReconciled
                        && report.error.is_none(),
                ) {
                    group.invalidate(error.to_string());
                }
            }
        }
        if let Some(collector) = &mut self.structured_capture_v2 {
            if collector.collecting() {
                if let Err(error) = collector.complete(
                    receipt.capture(),
                    report.submission == CalibrationSubmissionState::HostReconciled
                        && report.error.is_none(),
                ) {
                    collector.invalidate(error.to_string());
                }
            }
        }
    }
    pub async fn freeze_structured_cost_phase_v2(
        &mut self,
    ) -> Result<StructuredPhaseFreezeReceipt> {
        self.selected_phase_boundary()?;
        let checkpoint = self.freeze_cost_model().await?;
        self.structured_capture_v2
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("V2 collector not started"))?
            .freeze(checkpoint.accepted_ordinal())
            .map_err(capture_error)
    }
    pub async fn finish_structured_cost_calibration_v2(
        &mut self,
    ) -> Result<StructuredCalibrationArtifactV2> {
        self.selected_phase_boundary()?;
        let checkpoint = self.freeze_cost_model().await?;
        self.structured_capture_v2
            .take()
            .ok_or_else(|| FerrumError::invalid_request("V2 collector not started"))?
            .finish(checkpoint.accepted_ordinal())
            .map_err(capture_error)
    }
}

impl CalibrationWaveReport {
    /// Original typed settlement input for a later independently declared V2
    /// owner/window population. This never creates source membership or permits.
    pub fn structured_cost_input_v2(&self)->std::result::Result<
        ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredInputV2,
        ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2,
    >{
        use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2 as U;
        if self.submission != CalibrationSubmissionState::HostReconciled || self.error.is_some() {
            return Err(U::InvalidSample);
        }
        super::super::cost_observation::structured_discovery_input_v2(
            self.host_stages.as_ref().ok_or(U::MissingEvidence)?,
        )
    }
}
