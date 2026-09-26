//! Exclusive manual driver for measured calibration. It uses product-created
//! owners, real output credits and the guarded dispatcher, never a synthetic
//! cost observation or a modified request clock.
use super::*;
use ferrum_interfaces::engine::{InferenceEngine, LlmInferenceEngine};
use ferrum_interfaces::output_flow::{CreditedOutputSession, OutputProjectionContract};
use ferrum_interfaces::InferenceRequestContext;

mod artifact;
mod selected;
pub use selected::{SelectedCalibrationOptions, SelectedFitFreezeReceipt};
mod structured;
mod structured_group_v2;
mod structured_v2;
pub use structured::{
    StructuredCalibrationArtifact, StructuredCalibrationOptions, StructuredCalibrationProgress,
    StructuredCalibrationScopeV1, StructuredCapturePhase, StructuredPhaseFreezeReceipt,
};
pub use structured_group_v2::{
    StructuredCalibrationGroupArtifactV2, StructuredCalibrationGroupLimitsV2,
    StructuredCalibrationGroupOptionsV2,
};
pub use structured_v2::{StructuredCalibrationArtifactV2, StructuredCalibrationOptionsV2};
mod audit_readiness;
mod checkpoint;
mod evidence;
mod planning_audit;
mod prepared_projection;
pub use audit_readiness::{CalibrationAuditReadinessV2, CalibrationAuditRowReadinessV2};
pub use planning_audit::{
    RequiredFutureAuditActionV2, RequiredFutureAuditCostV2, RequiredFutureAuditFailureV2,
    RequiredFutureAuditFrontierV2, RequiredFutureAuditLimitsV2, RequiredFutureAuditPathReportV2,
    RequiredFutureAuditPathV2, RequiredFutureAuditPlanV2, RequiredFutureAuditQueryV2,
    RequiredFutureAuditReportV2, RequiredFutureAuditRowV2,
};
pub use prepared_projection::{
    StructuredPreparedProjectionBudgetV2, StructuredPreparedProjectionReportV2,
};
mod observation;
mod reference;
mod token_policy_residency;
pub use artifact::{CalibrationProfileArtifact, CalibrationProfilePaths, ImportedCalibrationModel};
pub use checkpoint::FrozenCalibrationModel;
pub use evidence::CalibrationRequestEvidence;
pub use reference::{
    CalibrationReferenceArtifact, CalibrationReferenceCollector, CalibrationReferenceCurve,
    CalibrationReferenceDiscoverySample, CalibrationReferencePlan, CalibrationReferenceTrial,
};
mod types;
pub use observation::{
    CalibrationCommittedRow, CalibrationCommittedWork, CalibrationObservation,
    CalibrationQueueDisposition,
};
#[cfg(test)]
mod tests;
pub(super) use types::CalibrationWaveReceipt;
pub use types::{
    CalibrationAction, CalibrationBlockReason, CalibrationDecodeRoute, CalibrationFrontier,
    CalibrationLimits, CalibrationSubmissionState, CalibrationTurn, CalibrationWaveReport,
    CalibrationWork,
};

pub struct CalibrationSession {
    selected_capture: Option<super::cost_observation::SelectedCalibrationCapture>,
    selected_capture_identity: Option<[u8; 32]>,
    structured_capture: Option<super::cost_observation::StructuredCalibrationCollector>,
    structured_capture_v2: Option<super::cost_observation::StructuredCalibrationCollectorV2>,
    structured_group_v2: Option<super::cost_observation::StructuredCalibrationGroupV2>,
    engine: ContinuousBatchEngine,
    identity: Arc<()>,
    limits: CalibrationLimits,
    pending: Option<Arc<CalibrationWaveReceipt>>,
    indeterminate: bool,
    /// Original owner/frontier returned by this session's actual request
    /// initialization. Retained only while the corresponding owner is live.
    reference_origins: std::collections::HashMap<RequestId, super::cost_observation::CostFrontier>,
}

impl CalibrationSession {
    pub(crate) fn from_fresh_engine(
        mut engine: ContinuousBatchEngine,
        limits: CalibrationLimits,
    ) -> Result<Self> {
        limits.validate()?;
        let inner = Arc::get_mut(&mut engine.inner).ok_or_else(|| {
            FerrumError::invalid_request("calibration requires a fresh exclusive engine")
        })?;
        if inner.config.scheduler.slo.mode != ferrum_types::SloMode::Observe
            || inner.config.scheduler.slo.admission.time_policy
                != ferrum_types::SloTimeAdmissionPolicy::CompleteRequests
            || inner.model_executor.execution_resource_authority()
                != ExecutionResourceAuthority::PlanRuntime
            || inner.spec_config.is_some()
            || inner.bg_loop_spawned.load(Ordering::Acquire)
            || inner.is_running.load(Ordering::Acquire)
            || inner.shutdown_started.load(Ordering::Acquire)
            || !inner.sequences.read().is_empty()
            || inner.scheduler.active_count() != 0
            || inner.scheduler.waiting_count() != 0
            || inner.cost_runtime.is_none()
        {
            return Err(FerrumError::config("calibration requires fresh PlanRuntime Observe/CompleteRequests with cost observation and no background driver"));
        }
        inner.manual_calibration_driver = true;
        Ok(Self {
            selected_capture: None,
            selected_capture_identity: None,
            structured_capture: None,
            structured_capture_v2: None,
            structured_group_v2: None,
            engine,
            identity: Arc::new(()),
            limits,
            pending: None,
            indeterminate: false,
            reference_origins: Default::default(),
        })
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) fn test_engine_inner(&self) -> Arc<EngineInner> {
        Arc::clone(&self.engine.inner)
    }

    pub fn configuration(&self) -> &EngineConfig {
        &self.engine.inner.config
    }

    pub fn context_capacity(&self) -> usize {
        self.engine
            .inner
            .model_executor
            .capabilities()
            .max_sequence_length
            .min(
                self.engine
                    .inner
                    .model_executor
                    .kv_capacity()
                    .unwrap_or(usize::MAX),
            )
    }

    pub async fn add_request(
        &mut self,
        request: ferrum_types::InferenceRequest,
        context: InferenceRequestContext,
        contract: Arc<OutputProjectionContract>,
    ) -> Result<CreditedOutputSession> {
        if self.pending.is_some() || self.indeterminate {
            return Err(FerrumError::invalid_request(
                "reap the calibration wave before adding work",
            ));
        }
        {
            let sequences = self.engine.inner.sequences.read();
            self.reference_origins.retain(|id, origin| {
                sequences.get(id).is_some_and(|sequence| {
                    sequence.cost_frontier.is_some_and(|current| {
                        current.owner_incarnation == origin.owner_incarnation
                    })
                })
            });
            if sequences.len() >= self.limits.maximum_requests().get() {
                return Err(FerrumError::resource_exhausted(
                    "calibration request bound reached",
                ));
            }
        }
        let id = request.id.clone();
        let declared_maximum = request.sampling_params.max_tokens as u64;
        let output = self
            .engine
            .infer_credited_stream(request, context, contract)
            .await?;
        if let Some(origin) = self
            .engine
            .inner
            .sequences
            .read()
            .get(&id)
            .and_then(|sequence| sequence.cost_frontier)
        {
            self.reference_origins.insert(id.clone(), origin);
        }
        if let Some(collector) = &mut self.structured_capture_v2 {
            if collector.collecting() {
                if let Err(error) = collector.admitted(id.clone(), declared_maximum) {
                    collector.invalidate(error.to_string());
                }
            }
        }
        if let Some(group) = &mut self.structured_group_v2 {
            if group.collecting() {
                if let Err(error) = group.admitted(id, declared_maximum) {
                    group.invalidate(error.to_string());
                }
            }
        }
        Ok(output)
    }

    pub fn frontiers(&self) -> Result<Vec<CalibrationFrontier>> {
        if self.pending.is_some() {
            return Err(FerrumError::invalid_request(
                "reap the calibration wave before observing a new frontier",
            ));
        }
        let sequences = self
            .engine
            .inner
            .sequences
            .try_read()
            .ok_or_else(|| FerrumError::resource_exhausted("calibration frontier is busy"))?;
        if sequences.len() > self.limits.maximum_requests().get() {
            return Err(FerrumError::resource_exhausted(
                "calibration request bound exceeded",
            ));
        }
        sequences
            .values()
            .map(|sequence| {
                let frontier = sequence.cost_frontier.ok_or_else(|| {
                    FerrumError::invalid_request("calibration owner identity is unavailable")
                })?;
                Ok(CalibrationFrontier {
                    session: Arc::clone(&self.identity),
                    request_id: sequence.request_id.clone(),
                    owner: frontier.owner_incarnation,
                    generation: frontier.work_generation,
                    // This explicit cold query hashes existing token storage;
                    // it neither tokenizes again nor copies the token history.
                    request_evidence: CalibrationRequestEvidence::capture(sequence),
                    generated: sequence.generated_tokens.len(),
                    prefill: (!sequence.prefill_complete).then(|| {
                        (
                            sequence.prefill_tokens_processed,
                            sequence.prefill_context_len(),
                        )
                    }),
                    kv_tokens: sequence
                        .model_kv
                        .as_ref()
                        .map_or(0, |kv| kv.handle().num_tokens()),
                })
            })
            .collect()
    }

    /// One turn. A cancelled waiter leaves the same durable wave pending;
    /// the next call only reaps that wave and does not execute its new action.
    pub async fn step(&mut self, action: CalibrationAction) -> Result<CalibrationTurn> {
        if let Some(receipt) = self.pending.as_ref().cloned() {
            let result = self.engine.inner.drain_slo_execution().await;
            self.pending.take();
            let mut report = receipt.report(result.err());
            self.indeterminate |= report.submission == CalibrationSubmissionState::InFlightUnknown;
            self.record_selected_capture(&receipt, &mut report)?;
            self.record_structured_capture(&receipt, &report);
            self.record_structured_v2_capture(&receipt, &report);
            return Ok(CalibrationTurn::Reaped(report));
        }
        if self.indeterminate {
            return Ok(CalibrationTurn::Blocked(
                CalibrationBlockReason::PreviousSubmissionIndeterminate,
            ));
        }
        // A preparation whose budget expired can abandon its publication
        // before executor entry. Reap that exact owner before accepting work.
        if self.engine.inner.drain_slo_execution().await?.is_some() {
            return Ok(CalibrationTurn::Blocked(
                CalibrationBlockReason::PublicationUnavailable,
            ));
        }
        let inner = Arc::clone(&self.engine.inner);
        let iteration = inner.iteration_lock.lock().await;
        inner.cancel_abandoned_requests().await?;
        inner.complete_credited_output_failures().await?;
        inner.refresh_credited_output_readiness();
        inner.complete_execution_readiness_failures().await?;
        match action {
            CalibrationAction::Reap => Ok(CalibrationTurn::Blocked(
                CalibrationBlockReason::NoPendingWork,
            )),
            CalibrationAction::AdmitOne => {
                if inner.prepare_slo_admission_turn(1).await? {
                    Ok(CalibrationTurn::AdmittedOrMaintained)
                } else {
                    Ok(CalibrationTurn::Blocked(
                        CalibrationBlockReason::AdmissionUnavailable,
                    ))
                }
            }
            CalibrationAction::Maintenance => match inner.prepare_slo_maintenance_turn().await? {
                Some(EngineIterationOutcome::Progressed) => {
                    Ok(CalibrationTurn::MaintenanceReconciled)
                }
                _ => Ok(CalibrationTurn::Blocked(
                    CalibrationBlockReason::MaintenanceUnavailable,
                )),
            },
            CalibrationAction::Wave(rows) => {
                if rows.is_empty()
                    || rows.len() > self.limits.maximum_requests().get()
                    || rows.len() > 256
                    || rows
                        .iter()
                        .any(|row| !Arc::ptr_eq(&row.frontier.session, &self.identity))
                {
                    return Err(FerrumError::invalid_request(
                        "calibration wave has invalid width or another session's frontier",
                    ));
                }
                if let Some(collector) = &mut self.structured_capture {
                    if let Err(error) = collector.offer(&rows) {
                        collector.invalidate(error.to_string());
                    }
                }
                if let Some(collector) = &mut self.structured_capture_v2 {
                    if collector.collecting() {
                        if let Err(error) = collector.offer(&rows) {
                            collector.invalidate(error.to_string());
                        }
                    }
                }
                if let Some(group) = &mut self.structured_group_v2 {
                    if group.collecting() {
                        if let Err(error) = group.offer(&rows) {
                            group.invalidate(error.to_string());
                        }
                    }
                }
                let preparation =
                    inner.prepare_calibration_wave(&rows, self.limits.maximum_requests());
                let prepared = match preparation {
                    Err(error) => {
                        self.record_structured_unsubmitted(&format!("preparation error: {error}"));
                        return Err(error);
                    }
                    Ok(preparation) => match preparation {
                        slo_controller::calibration::CalibrationPreparation::Blocked(reason) => {
                            self.record_structured_unsubmitted(&format!("{reason:?}"));
                            return Ok(CalibrationTurn::Blocked(reason));
                        }
                        slo_controller::calibration::CalibrationPreparation::Selected(prepared) => {
                            prepared
                        }
                    },
                };
                let receipt = prepared.calibration_receipt().ok_or_else(|| {
                    FerrumError::internal("manual calibration wave lost its submission receipt")
                })?;
                if let Some(collector) = &mut self.structured_capture {
                    if collector.collecting() {
                        if let Err(error) = collector.reserve_prepared() {
                            collector.invalidate(error.to_string());
                        }
                    }
                    if collector.collecting() {
                        match collector.pending_capture() {
                            Ok(capture) => {
                                if let Err(error) = receipt.bind_structured_capture(capture) {
                                    collector.invalidate(error.to_string());
                                }
                            }
                            Err(error) => collector.invalidate(error.to_string()),
                        }
                    }
                }
                if let Some(collector) = &mut self.structured_capture_v2 {
                    if collector.collecting() {
                        match prepared.structured_prepared_facts(
                            &inner,
                            self.limits.structured_prepared_projection_budget(),
                        ) {
                            Ok(facts) => {
                                if let Err(error) = collector.reserve_prepared(facts) {
                                    collector.invalidate(error.to_string());
                                }
                            }
                            Err(error) => collector.invalidate(error.to_string()),
                        }
                    }
                    if collector.collecting() {
                        match collector.pending_capture() {
                            Ok(capture) => {
                                if let Err(error) = receipt.bind_structured_capture(capture) {
                                    collector.invalidate(error.to_string());
                                }
                            }
                            Err(error) => collector.invalidate(error.to_string()),
                        }
                    }
                }
                if let Some(collector) = &mut self.structured_group_v2 {
                    if collector.collecting() {
                        match prepared.structured_prepared_facts(
                            &inner,
                            self.limits.structured_prepared_projection_budget(),
                        ) {
                            Ok(facts) => {
                                if let Err(error) = collector.reserve_prepared(facts) {
                                    collector.invalidate(error.to_string());
                                }
                            }
                            Err(error) => collector.invalidate(error.to_string()),
                        }
                    }
                    if collector.collecting() {
                        match collector.pending_capture() {
                            Ok(capture) => {
                                if let Err(error) = receipt.bind_structured_capture(capture) {
                                    collector.invalidate(error.to_string());
                                }
                            }
                            Err(error) => collector.invalidate(error.to_string()),
                        }
                    }
                }
                self.pending = Some(Arc::clone(&receipt));
                drop(iteration);
                let result = inner.execute_slo_controller_wave(prepared).await;
                self.pending.take();
                let mut report = receipt.report(result.err());
                self.indeterminate |=
                    report.submission == CalibrationSubmissionState::InFlightUnknown;
                self.record_selected_capture(&receipt, &mut report)?;
                self.record_structured_capture(&receipt, &report);
                self.record_structured_v2_capture(&receipt, &report);
                Ok(CalibrationTurn::Wave(report))
            }
        }
    }

    pub async fn shutdown(self) -> Result<()> {
        self.engine.shutdown().await
    }
}
