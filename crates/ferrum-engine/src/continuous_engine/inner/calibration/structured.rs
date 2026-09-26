//! Manual live collection with immutable pre-execution membership. Collection
//! failures remove model eligibility, never permission to execute a request.
use super::super::cost_observation::StructuredCalibrationCollector;
pub use super::super::cost_observation::{
    StructuredCalibrationArtifact, StructuredCalibrationOptions, StructuredCalibrationProgress,
    StructuredCalibrationScopeV1, StructuredCapturePhase, StructuredPhaseFreezeReceipt,
};
use super::*;

mod discovery;

pub(super) fn capture_error(error: impl std::fmt::Display) -> FerrumError {
    FerrumError::config(format!("structured calibration: {error}"))
}
impl CalibrationSession {
    pub fn structured_cost_progress(&self) -> Option<StructuredCalibrationProgress> {
        self.structured_capture
            .as_ref()
            .map(|collector| collector.progress())
    }
    fn structured_phase_boundary(&self) -> Result<()> {
        if self.prefix_preparation.is_some() || self.prefix_source5 {
            return Err(FerrumError::invalid_request(
                "prefix preparation needs its independent source5 protocol; old collectors cannot omit its request frontier",
            ));
        }
        if self.pending.is_some() || self.indeterminate {
            return Err(FerrumError::invalid_request(
                "reap the real wave before a structured phase boundary",
            ));
        }
        Ok(())
    }

    /// Start an explicit diagnostic collector at a drained original FIFO cut.
    /// The discovery scope, protocol, limits and phase sizes are immutable.
    pub async fn begin_structured_cost_calibration(
        &mut self,
        options: StructuredCalibrationOptions,
    ) -> Result<()> {
        self.structured_phase_boundary()?;
        if self.structured_capture.is_some()
            || self.selected_capture_identity.is_some()
            || self.structured_capture_v2.is_some()
            || self.structured_group_v2.is_some()
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
            return Err(FerrumError::config("structured calibration requires HostSettledV1 capture configured before engine creation"));
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
                "structured calibration requires real executor identity",
            ));
        };
        if identity.schema_version
            != ferrum_interfaces::execution_cost::EXECUTOR_COST_IDENTITY_SCHEMA
        {
            return Err(FerrumError::unsupported(
                "structured executor identity schema mismatch",
            ));
        }
        let fingerprint =
            ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            };
        self.structured_capture = Some(
            StructuredCalibrationCollector::new(
                options,
                fingerprint,
                Arc::clone(&runtime.clock),
                checkpoint.accepted_ordinal(),
            )
            .map_err(capture_error)?,
        );
        Ok(())
    }

    pub(super) fn record_structured_unsubmitted(&mut self, reason: &str) {
        if let Some(group) = &mut self.structured_group_v2 {
            if group.collecting() {
                if let Err(error) = group.complete_unsubmitted(reason) {
                    group.invalidate(error.to_string());
                }
            }
        }

        if let Some(collector) = &mut self.structured_capture_v2 {
            if collector.collecting() {
                if let Err(error) = collector.complete_unsubmitted(reason) {
                    collector.invalidate(error.to_string());
                }
            }
        }
        if let Some(collector) = &mut self.structured_capture {
            if collector.collecting() {
                if let Err(error) = collector.complete_unsubmitted(reason) {
                    collector.invalidate(error.to_string());
                }
            }
        }
    }
    pub(super) fn record_structured_capture(
        &mut self,
        receipt: &CalibrationWaveReceipt,
        report: &CalibrationWaveReport,
    ) {
        if let Some(collector) = &mut self.structured_capture {
            if !collector.collecting() {
                return;
            }
            // Even a NotSubmitted receipt can contain an accepted stages-only
            // queue record. Preserve that original FIFO position and raw call;
            // a reserved member still fails because it was not reconciled.
            let result = collector.complete(
                receipt.capture(),
                report.submission == CalibrationSubmissionState::HostReconciled
                    && report.error.is_none(),
            );
            if let Err(error) = result {
                collector.invalidate(error.to_string());
            }
        }
    }

    /// Barrier receipts bind the original source prefix and numerical state.
    /// Real output owners may continue after the cut; none is cancelled here.
    pub async fn freeze_structured_cost_phase(&mut self) -> Result<StructuredPhaseFreezeReceipt> {
        self.structured_phase_boundary()?;
        let checkpoint = self.freeze_cost_model().await?;
        self.structured_capture
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("structured calibration not started"))?
            .freeze(checkpoint.accepted_ordinal())
            .map_err(capture_error)
    }
    pub async fn finish_structured_cost_calibration(
        &mut self,
    ) -> Result<StructuredCalibrationArtifact> {
        self.structured_phase_boundary()?;
        let checkpoint = self.freeze_cost_model().await?;
        self.structured_capture
            .take()
            .ok_or_else(|| FerrumError::invalid_request("structured calibration not started"))?
            .finish(checkpoint.accepted_ordinal())
            .map_err(capture_error)
    }
}
