//! Product-owned fit/residual phase boundaries. Every cohort finishes its full
//! request/output policy; phase cuts never cancel an owner to create a sample.
use super::super::cost_observation::SelectedCalibrationCapture;
pub use super::super::cost_observation::SelectedFitFreezeReceipt;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::statistical::model::{
    HeldoutEvaluationV1, ModelUnknown, WholeWaveObservationV1,
};

#[derive(Debug, Clone)]
pub struct SelectedCalibrationOptions {
    pub export: ferrum_types::SloCostProfileExportConfig,
    pub protocol_sha256: [u8; 32],
}

impl CalibrationSession {
    fn selected_phase_boundary(&self) -> Result<()> {
        if self.pending.is_some()
            || self.indeterminate
            || !self.frontiers()?.is_empty()
            || self.engine.inner.scheduler.active_count() != 0
            || self.engine.inner.scheduler.waiting_count() != 0
        {
            return Err(FerrumError::invalid_request(
                "selected calibration phase requires all real owners/output work completed",
            ));
        }
        Ok(())
    }
    pub fn begin_selected_cost_calibration(
        &mut self,
        options: SelectedCalibrationOptions,
    ) -> Result<()> {
        self.selected_phase_boundary()?;
        if self.selected_capture_identity.is_some() {
            return Err(FerrumError::invalid_request(
                "selected calibration already started",
            ));
        }
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
                "selected calibration needs real executor identity",
            ));
        };
        if identity.schema_version
            != ferrum_interfaces::execution_cost::EXECUTOR_COST_IDENTITY_SCHEMA
        {
            return Err(FerrumError::unsupported(
                "selected calibration executor identity schema mismatch",
            ));
        }
        let config = &self.engine.inner.config.scheduler.slo.cost_observation;
        let fingerprint =
            ferrum_scheduler::implementations::continuous::cost_model::ExecutionFingerprint {
                model_weights: identity.model_weights,
                numerical_policy: identity.numerical_policy,
                device_runtime: identity.device_runtime,
                execution_config: identity.execution_config,
            };
        self.selected_capture = Some(
            SelectedCalibrationCapture::new(
                options.export,
                config,
                fingerprint,
                options.protocol_sha256,
                Arc::clone(&runtime.clock),
            )
            .map_err(|e| FerrumError::config(e.to_string()))?,
        );
        self.selected_capture_identity = self
            .selected_capture
            .as_ref()
            .map(|capture| capture.identity());
        Ok(())
    }
    pub(super) fn record_selected_capture(
        &mut self,
        receipt: &CalibrationWaveReceipt,
        report: &mut CalibrationWaveReport,
    ) -> Result<()> {
        let reconciled = report.submission == CalibrationSubmissionState::HostReconciled
            && report.error.is_none();
        if let Some(identity) = self.selected_capture_identity {
            report.selected = Some(SelectedCalibrationEvidence {
                session: Arc::clone(&self.identity),
                observation: SelectedCalibrationCapture::observation(
                    &receipt.capture,
                    reconciled,
                    identity,
                ),
            });
        }
        if let Some(capture) = &mut self.selected_capture {
            capture
                .record(
                    &receipt.capture,
                    report.submission == CalibrationSubmissionState::HostReconciled
                        && report.error.is_none(),
                )
                .map_err(|e| {
                    FerrumError::resource_exhausted(format!("selected calibration capture: {e}"))
                })?;
        }
        Ok(())
    }
    pub async fn freeze_selected_cost_fit(&mut self) -> Result<SelectedFitFreezeReceipt> {
        self.selected_phase_boundary()?;
        let checkpoint = self.freeze_cost_model().await?;
        self.selected_capture
            .as_mut()
            .ok_or_else(|| FerrumError::invalid_request("selected calibration not started"))?
            .freeze_fit(checkpoint.accepted_ordinal())
            .map_err(|e| FerrumError::config(e.to_string()))
    }
    pub async fn finish_selected_cost_residual(&mut self) -> Result<ImportedCalibrationModel> {
        self.selected_phase_boundary()?;
        let checkpoint = self.freeze_cost_model().await?;
        let capture = self
            .selected_capture
            .take()
            .ok_or_else(|| FerrumError::invalid_request("selected calibration not started"))?;
        let runtime = Arc::clone(
            self.engine
                .inner
                .cost_runtime
                .as_ref()
                .ok_or_else(|| FerrumError::internal("cost runtime unavailable"))?,
        );
        let clock = Arc::clone(&runtime.clock);
        let config = self
            .engine
            .inner
            .config
            .scheduler
            .slo
            .cost_observation
            .clone();
        let cut_ordinal = checkpoint.accepted_ordinal();
        let (cut, imported, support) = tokio::task::spawn_blocking(move || {
            let (cut, support) = capture
                .finish(cut_ordinal)
                .map_err(|e| FerrumError::config(e.to_string()))?;
            let imported = runtime.load_calibration_profile(&config, &cut)?;
            Ok::<_, FerrumError>((cut, imported, support))
        })
        .await
        .map_err(|e| FerrumError::internal(format!("selected calibration export: {e}")))??;
        let audit = serde_json::json!({"scope":"independent completed fit/residual capture, reloaded through product importer; heldout excluded",
            "accepted_ordinal":cut_ordinal,"worker_cut":checkpoint.audit()?,"capture_support":support});
        Ok(ImportedCalibrationModel {
            selected_session: Some(Arc::clone(&self.identity)),
            artifact: cut.into(),
            imported,
            clock,
            audit,
        })
    }
}

// Private provenance is minted only from this manual session's durable actual
// receipt. Public report metadata cannot substitute a fabricated sample.
#[derive(Debug)]
pub(super) struct SelectedCalibrationEvidence {
    session: Arc<()>,
    observation: std::result::Result<WholeWaveObservationV1, ModelUnknown>,
}
impl ImportedCalibrationModel {
    /// Independent completed heldout lookup. No training mutation, model
    /// publication or new execution permission is produced by this method.
    pub fn evaluate_selected_wave(
        &self,
        report: &CalibrationWaveReport,
    ) -> std::result::Result<HeldoutEvaluationV1, ModelUnknown> {
        use ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown;
        let evidence = report.selected.as_ref().ok_or(ModelUnknown::Evidence(
            StatisticalEvidenceUnknown::MissingProducer,
        ))?;
        self.evaluate_selected_evidence(evidence)
    }
    fn evaluate_selected_evidence(
        &self,
        evidence: &SelectedCalibrationEvidence,
    ) -> std::result::Result<HeldoutEvaluationV1, ModelUnknown> {
        if !self
            .selected_session
            .as_ref()
            .is_some_and(|session| Arc::ptr_eq(session, &evidence.session))
        {
            return Err(ModelUnknown::WrongSource);
        }
        let observation = evidence.observation.as_ref().map_err(|reason| *reason)?;
        let receipt = self
            .imported
            .receipt
            .selected_whole_wave
            .as_ref()
            .ok_or(ModelUnknown::WrongSource)?;
        if observation.source_sha256 != receipt.capture_identity_sha256 {
            return Err(ModelUnknown::WrongSource);
        }
        if observation.accepted_ordinal <= self.artifact.accepted_ordinal {
            return Err(ModelUnknown::PhaseLeakage);
        }
        if &observation.fingerprint != self.imported.snapshot.fingerprint() {
            return Err(ModelUnknown::WrongFingerprint);
        }
        let now = self
            .clock
            .now_ns()
            .filter(|now| *now >= observation.observed_at_ns)
            .ok_or(ModelUnknown::Clock)?;
        let prediction = self.imported.snapshot.predict_selected_wave(
            &observation.exact,
            &observation.selected,
            now,
        );
        let underestimate_ns = prediction
            .as_ref()
            .ok()
            .map(|value| observation.wall_ns.saturating_sub(value.planning_ns));
        Ok(HeldoutEvaluationV1 {
            prediction,
            actual_ns: observation.wall_ns,
            underestimate_ns,
        })
    }
}

#[cfg(test)]
mod tests;
