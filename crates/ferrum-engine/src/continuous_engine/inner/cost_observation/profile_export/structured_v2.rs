//! V2 live population capture. One immutable owner/window rule and complete
//! request cohorts; execution still belongs to the existing calibration driver.
use super::selected::raw::RawSource;
use super::structured::{
    StructuredCalibrationProgress, StructuredCapturePhase, StructuredPhaseFreezeReceipt,
};
use super::*;
use crate::continuous_engine::inner::calibration::CalibrationWork;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::windows::{
    CohortPlanV2, MembershipRuleV2,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::*;
use std::num::{NonZeroU64, NonZeroUsize};

#[path = "structured_v2/prepared.rs"]
mod prepared;
pub(in crate::continuous_engine::inner) use prepared::{
    PreparedRowBindingV2, PreparedStructuredFactsV2,
};
#[path = "structured_v2/options.rs"]
mod options;
pub use options::StructuredCalibrationOptionsV2;
#[path = "structured_v2/population.rs"]
mod population;
use population::{numeric_phase, PopulationLedgerV2};
#[path = "structured_v2/cohorts.rs"]
mod cohorts;
use cohorts::CohortLedgerV2;
#[path = "structured_v2/source.rs"]
mod source;
use source::StructuredSource;
#[path = "structured_v2/group.rs"]
mod group;
#[path = "structured_v2/wire.rs"]
mod wire;
pub(in crate::continuous_engine::inner) use group::StructuredCalibrationGroupV2;
pub use group::{
    StructuredCalibrationGroupArtifactV2, StructuredCalibrationGroupLimitsV2,
    StructuredCalibrationGroupOptionsV2,
};

pub struct StructuredCalibrationArtifactV2 {
    pub source_path: PathBuf,
    pub source_sha256: [u8; 32],
    pub source_bytes: u64,
    pub phase: StructuredCapturePhase,
    pub offered_waves: u64,
    pub scope_members: u64,
    pub scope_failures: u64,
    pub failure: Option<String>,
    pub model: Option<QualifiedStructuredModelV2>,
}
pub(in crate::continuous_engine::inner) struct StructuredCalibrationCollectorV2 {
    options: StructuredCalibrationOptionsV2,
    binding: Arc<StructuredCaptureSessionBinding>,
    contract: StructuredSourceContractV2,
    clock: Arc<dyn CostObservationClock>,
    source: StructuredSource,
    ledger: PopulationLedgerV2,
    cohorts: CohortLedgerV2,
    phase: StructuredCapturePhase,
    samples: Vec<StructuredNumericObservationV2>,
    fitted: Option<FittedStructuredModelV2>,
    calibrated: Option<CalibratedStructuredModelV2>,
    qualified: Option<QualifiedStructuredModelV2>,
    failure: Option<String>,
}
fn numeric_error(reason: StructuredUnknownV2) -> ExportError {
    ExportError::Config(format!("structured V2 capture: {reason:?}"))
}
impl StructuredCalibrationCollectorV2 {
    pub fn new(
        options: StructuredCalibrationOptionsV2,
        fingerprint: model::ExecutionFingerprint,
        clock: Arc<dyn CostObservationClock>,
        initial_fifo_cutoff: u64,
    ) -> Result<Self, ExportError> {
        options.validate()?;
        let rule = options.membership_rule.signature().map_err(numeric_error)?;
        let manifest = options
            .cohort_plan
            .signature(&options.cohort_manifest_payload)
            .map_err(numeric_error)?;
        let binding = Arc::new(
            StructuredCaptureSessionBinding::new(
                options.protocol_signature(rule, manifest)?,
                fingerprint.clone(),
                clock.as_ref(),
            )
            .map_err(numeric_error)?,
        );
        let contract = StructuredSourceContractV2 {
            capture_identity: binding.identity(),
            protocol: binding.protocol(),
            membership_rule: rule,
            cohort_manifest: manifest,
            phase_members: options.phase_members,
        };
        let mut source =
            StructuredSource::create(&options.observations_path, options.maximum_file_bytes.get())?;
        source.record(&serde_json::json!({
            "artifact_type":"ferrum.structured-live-source","schema_version":3,"model_revision":MODEL_REVISION_V2,
            "capture_identity":binding.identity(),"protocol":binding.protocol(),"declared_protocol":options.protocol_sha256,
            "rule_signature":rule,"fingerprint":profile::ProfileFingerprint::from(&fingerprint),
            "producer":ProducerIdentity::current()?,"opening":ExportClockReading::opening(clock.as_ref())?,
            "opened_at_ns":binding.opened_at_ns(),"initial_fifo_cutoff":initial_fifo_cutoff,
            "scope":options.scope,"membership_rule":options.membership_rule,
            "cohort_plan":options.cohort_plan,"cohort_manifest_payload":options.cohort_manifest_payload,
            "cohort_manifest_sha256":manifest,"phase_members":options.phase_members,
            "maximum_offered_waves":options.maximum_offered_waves.get(),"maximum_file_bytes":options.maximum_file_bytes.get(),
            "settings":{"min_samples":options.settings.min_phase_samples,"redundancy":options.settings.min_fit_redundancy,
                "max_phase_samples":options.settings.max_phase_samples,"max_axes":options.settings.max_axes,
                "max_rank":options.settings.max_rank,"max_wave_ns":options.settings.max_wave_ns,
                "max_age_ns":options.settings.max_sample_age_ns,"margin_ns":options.settings.static_margin_ns}
        }))?;
        let cohorts = CohortLedgerV2::new(options.cohort_plan.clone())?;
        Ok(Self {
            options,
            binding,
            contract,
            clock,
            source,
            cohorts,
            ledger: PopulationLedgerV2::new(initial_fifo_cutoff),
            phase: StructuredCapturePhase::Fit,
            samples: Vec::new(),
            fitted: None,
            calibrated: None,
            qualified: None,
            failure: None,
        })
    }
    pub fn progress(&self) -> StructuredCalibrationProgress {
        StructuredCalibrationProgress {
            phase: self.phase,
            offered_attempts: self.ledger.offered,
            required_members: self.options.phase_members,
            reserved_members: self.ledger.reserved,
            completed_members: self.ledger.completed,
            failed_members: self.ledger.failed,
            failure: self.failure.clone(),
        }
    }
    pub fn coverage(&self) -> Result<StructuredCoverageReportV2, ExportError> {
        self.options
            .scope
            .coverage_report(&self.samples)
            .map_err(numeric_error)
    }
    pub fn collecting(&self) -> bool {
        matches!(
            self.phase,
            StructuredCapturePhase::Fit
                | StructuredCapturePhase::Residual
                | StructuredCapturePhase::Qualification
        )
    }
    pub fn invalidate(&mut self, reason: String) {
        if self.phase != StructuredCapturePhase::Failed {
            let _ = self.source.record(
                &serde_json::json!({"kind":"phase_failed","phase":self.phase,"reason":reason}),
            );
        }
        self.phase = StructuredCapturePhase::Failed;
        self.fitted = None;
        self.calibrated = None;
        self.qualified = None;
        self.samples.clear();
        self.ledger.abandon();
        self.failure.get_or_insert(reason);
    }
    pub fn begin_cohort(&mut self, ordinal: usize) -> Result<(), ExportError> {
        if !self.collecting() {
            return Err(ExportError::Source("V2 collector is closed"));
        }
        self.cohorts.begin(self.phase, ordinal)?;
        let declared =
            &self.options.cohort_plan.phases[population::phase_index(self.phase)?][ordinal];
        self.source.record(
            &serde_json::json!({"kind":"cohort_begin","phase":self.phase,"cohort":ordinal,
            "manifest_case":declared.manifest_case,"repetition":declared.repetition}),
        )
    }
    pub fn admitted(&mut self, id: RequestId, maximum: u64) -> Result<(), ExportError> {
        let slot = self.cohorts.admit(id.clone(), maximum)?;
        self.source.record(&serde_json::json!({"kind":"request_admitted","phase":self.phase,
            "cohort":self.cohorts.active_ordinal()?,"slot":slot,"request_id":id,"maximum_output":maximum}))
    }
    pub fn end_cohort(&mut self) -> Result<(), ExportError> {
        let (cohort, count) = self.cohorts.end(self.phase)?;
        self.source.record(
            &serde_json::json!({"kind":"cohort_end","phase":self.phase,"cohort":cohort,
            "admitted_count":count,"completed_count":count}),
        )
    }
    pub fn offer(&mut self, work: &[CalibrationWork]) -> Result<(), ExportError> {
        let cohort = self.cohorts.active_ordinal()?;
        let attempt =
            self.ledger
                .offer(self.phase, cohort, self.options.maximum_offered_waves.get())?;
        self.source.record(&serde_json::json!({"kind":"offered","offered":attempt.offered,"phase":self.phase,
            "cohort":cohort,"rows":work.iter().map(|r|serde_json::json!({"request_id":r.frontier.request_id(),
                "owner":r.frontier.owner_incarnation().get(),"generation":r.frontier.work_generation().get(),
                "generated":r.frontier.generated_tokens(),"work":match r.work {
                    ActualRowWork::Decode{kv_tokens}=>serde_json::json!({"decode":{"kv_tokens":kv_tokens}}),
                    ActualRowWork::Prefill{offset,count,total_prompt_tokens}=>serde_json::json!({"prefill":{"offset":offset,"count":count,"total_prompt_tokens":total_prompt_tokens}}),
                    _=>serde_json::Value::Null,
                }})).collect::<Vec<_>>()}))
    }
    pub fn reserve_prepared(
        &mut self,
        prepared: PreparedStructuredFactsV2,
    ) -> Result<(), ExportError> {
        self.cohorts.prepared(&prepared)?;
        let reserved = self.ledger.reserve(
            prepared,
            &self.options.membership_rule,
            self.options.phase_members,
            Arc::clone(&self.binding),
        )?;
        self.source.record(&serde_json::json!({"kind":"reserved","offered":reserved.attempt.offered,
            "member":reserved.member,"window":reserved.window,"phase":reserved.attempt.phase,"cohort":reserved.attempt.cohort,
            "boundary":"prepared_before_execute","prepared":wire::PreparedWire::new(&reserved.prepared)?}))
    }
    pub fn pending_capture(&self) -> Result<Arc<CostCalibrationCapture>, ExportError> {
        self.ledger
            .pending
            .as_ref()
            .map(|r| Arc::clone(&r.capture))
            .ok_or(ExportError::Source("V2 capture not reserved"))
    }
    pub fn complete_unsubmitted(&mut self, reason: &str) -> Result<(), ExportError> {
        if let Some(attempt) = self.ledger.attempt.take() {
            return self.source.record(
                &serde_json::json!({"kind":"preparation_unavailable","offered":attempt.offered,
                "phase":attempt.phase,"cohort":attempt.cohort,"reason":reason}),
            );
        }
        let reserved = self.ledger.take_pending(None)?;
        self.ledger.settle_member(&reserved, false);
        self.source.record(&serde_json::json!({"kind":"unsubmitted","offered":reserved.attempt.offered,
            "member":reserved.member,"phase":reserved.attempt.phase,"cohort":reserved.attempt.cohort,"reason":reason}))
    }
    pub fn complete(
        &mut self,
        capture: &Arc<CostCalibrationCapture>,
        reconciled: bool,
    ) -> Result<(), ExportError> {
        let reserved = self.ledger.take_pending(Some(capture))?;
        let queue = capture.host_stage_queue();
        let validated = self.ledger.accept_fifo(queue).and_then(|()| {
            super::super::trainer::structured_v2::validate_capture_v2(
                capture,
                reconciled,
                &reserved.prepared,
            )
            .map_err(numeric_error)
        });
        let actual = match validated {
            Ok(actual) => actual,
            Err(error) => return self.completion_failed(&reserved, capture, reconciled, error),
        };
        self.complete_validated(reserved, capture, reconciled, &actual)
    }
    fn complete_validated(
        &mut self,
        reserved: population::ReservedWaveV2,
        capture: &Arc<CostCalibrationCapture>,
        reconciled: bool,
        actual: &super::super::trainer::structured_v2::ValidatedStructuredWaveV2,
    ) -> Result<(), ExportError> {
        let queue = capture.host_stage_queue();
        let stages = Arc::clone(&actual.stages);
        let converted = (|| {
            let completed = self
                .cohorts
                .completed(self.phase, &reserved.prepared, &actual)?;
            let numeric = if let Some(member) = reserved.member {
                let value = actual
                    .member_for(
                        &self.binding,
                        StructuredMemberBindingV2 {
                            rule_signature: self.contract.membership_rule,
                            offered_ordinal: reserved.attempt.offered,
                            member_ordinal: member,
                            phase: numeric_phase(reserved.attempt.phase)?,
                        },
                    )
                    .map_err(numeric_error)?;
                if value.input.owner() != &self.options.scope.owner
                    || value.input.regression_axes().len() > self.options.settings.max_axes
                    || value.input.joint_support_coordinates().len()
                        > self.options.settings.max_axes
                {
                    return Err(ExportError::Source(
                        "member input exceeded declared scope or storage",
                    ));
                }
                Some(value)
            } else {
                None
            };
            let binding = stages
                .structured_evidence
                .as_ref()
                .and_then(|r| r.as_ref().ok())
                .ok_or(ExportError::Source(
                    "original qualified settlement disappeared",
                ))?
                .stage_binding();
            Ok((completed, numeric, binding))
        })();
        let (completed, numeric, stage_binding) = match converted {
            Ok(value) => value,
            Err(error) => return self.completion_failed(&reserved, capture, reconciled, error),
        };
        let written = (|| {
            self.source.record_borrowed(&wire::CompletedWire {
                reserved: &reserved,
                stages: &stages,
                queue,
                reconciled,
                numeric: numeric.as_ref(),
                stage_binding,
            })?;
            for request in completed {
                self.source
                    .record(&serde_json::json!({"kind":"request_completed","request":request}))?;
            }
            Ok::<(), ExportError>(())
        })();
        if let Err(error) = written {
            self.ledger.settle_member(&reserved, false);
            return Err(error);
        }
        self.ledger.settle_member(&reserved, true);
        if let Some(numeric) = numeric {
            self.samples.push(numeric);
        }
        Ok(())
    }
    fn completion_failed<T>(
        &mut self,
        reserved: &population::ReservedWaveV2,
        capture: &Arc<CostCalibrationCapture>,
        reconciled: bool,
        error: ExportError,
    ) -> Result<T, ExportError> {
        self.ledger.settle_member(reserved, false);
        // Retain the original failed call even when a later lifecycle/numeric
        // check rejects it. This diagnostic record cannot qualify a source.
        let stages = capture.host_stages();
        self.source.record(&serde_json::json!({"kind":"completed","offered":reserved.attempt.offered,
            "member":reserved.member,"phase":reserved.attempt.phase,"cohort":reserved.attempt.cohort,
            "queue":capture.host_stage_queue(),"reconciled":reconciled,"conversion_error":error.to_string(),
            "host_stages":stages.as_ref().map(|s|s.structured_diagnostic_view()),
            "selected_independent_attention_v2":stages.as_ref().and_then(|s|s.statistical_evidence.as_ref()).and_then(|v|v.independent_attention_v2()),
            "selected_structured_capture":stages.as_ref().and_then(|s|s.statistical_evidence.as_ref()).and_then(|v|v.structured_capture()).map(|v|v.map(AsRef::as_ref))}))?;
        Err(error)
    }
    pub fn freeze(&mut self, cutoff: u64) -> Result<StructuredPhaseFreezeReceipt, ExportError> {
        let result = self.freeze_inner(cutoff);
        if let Err(error) = &result {
            self.invalidate(error.to_string());
        }
        result
    }
    fn freeze_inner(&mut self, cutoff: u64) -> Result<StructuredPhaseFreezeReceipt, ExportError> {
        let now = self
            .clock
            .now_ns()
            .ok_or(ExportError::Clock("V2 original freeze clock unavailable"))?;
        self.freeze_at(cutoff, now)
    }
    fn check_freeze(&self, cutoff: u64) -> Result<(), ExportError> {
        self.cohorts.freeze(self.phase)?;
        self.ledger.freeze(
            self.phase,
            cutoff,
            self.samples.len(),
            self.options.phase_members,
        )
    }
    fn freeze_at(
        &mut self,
        cutoff: u64,
        now: u64,
    ) -> Result<StructuredPhaseFreezeReceipt, ExportError> {
        self.check_freeze(cutoff)?;
        let coverage = self.coverage()?;
        self.source
            .record(&serde_json::json!({"kind":"coverage","phase":self.phase,"report":coverage}))?;
        let signature = match self.phase {
            StructuredCapturePhase::Fit => {
                let model = FittedStructuredModelV2::fit(
                    self.binding.fingerprint().clone(),
                    self.options.settings.clone(),
                    self.options.scope.clone(),
                    self.contract.clone(),
                    &self.samples,
                    now,
                )
                .map_err(numeric_error)?;
                let signature = model.parameters_signature();
                self.fitted = Some(model);
                signature
            }
            StructuredCapturePhase::Residual => {
                let model = self
                    .fitted
                    .take()
                    .ok_or(ExportError::Source("V2 fit state missing"))?
                    .calibrate(&self.samples, now)
                    .map_err(numeric_error)?;
                let signature = model.parameters_signature();
                self.calibrated = Some(model);
                signature
            }
            StructuredCapturePhase::Qualification => {
                let model = self
                    .calibrated
                    .take()
                    .ok_or(ExportError::Source("V2 residual state missing"))?
                    .qualify(&self.samples, now)
                    .map_err(numeric_error)?;
                let signature = model.parameters_signature();
                self.qualified = Some(model);
                signature
            }
            _ => return Err(ExportError::Source("V2 phase is closed")),
        };
        self.source.flush()?;
        let receipt = StructuredPhaseFreezeReceipt {
            capture_identity: self.binding.identity(),
            protocol: self.binding.protocol(),
            rule_signature: self.contract.membership_rule,
            phase: self.phase,
            accepted_fifo_cutoff: cutoff,
            member_cutoff: self.ledger.members,
            source_prefix_bytes: self.source.bytes(),
            source_prefix_sha256: self.source.prefix_digest(),
            frozen_at_ns: now,
            parameters_sha256: signature,
        };
        self.source
            .record(&serde_json::json!({"kind":"phase_freeze","receipt":receipt}))?;
        self.samples.clear();
        self.phase = match self.phase {
            StructuredCapturePhase::Fit => StructuredCapturePhase::Residual,
            StructuredCapturePhase::Residual => StructuredCapturePhase::Qualification,
            _ => StructuredCapturePhase::Qualified,
        };
        Ok(receipt)
    }
    pub fn finish(mut self, cutoff: u64) -> Result<StructuredCalibrationArtifactV2, ExportError> {
        let closing = match ExportClockReading::closing(self.clock.as_ref()) {
            Ok(value) => Some(value),
            Err(error) => {
                self.invalidate(error.to_string());
                None
            }
        };
        self.finish_with_closing(cutoff, closing)
    }
    fn finish_with_closing(
        mut self,
        cutoff: u64,
        closing: Option<ExportClockReading>,
    ) -> Result<StructuredCalibrationArtifactV2, ExportError> {
        if !self.ledger.audit_complete(cutoff) || self.phase != StructuredCapturePhase::Qualified {
            self.invalidate(
                "V2 source ended without complete original FIFO and qualification".into(),
            );
        }
        if let Some(error) = self.source.incomplete_error() {
            return Err(error);
        }
        self.source.record(&serde_json::json!({"kind":"footer","phase":self.phase,"failure":self.failure,
            "offered":self.ledger.offered,"members":self.ledger.members,"failed_members":self.ledger.failed.iter().sum::<usize>(),
            "accepted_fifo_cutoff":cutoff,"last_captured_fifo":self.ledger.last_fifo,"fifo_audit_complete":self.ledger.audit_complete(cutoff),"closing":closing}))?;
        let file = self.source.finish()?;
        Ok(StructuredCalibrationArtifactV2 {
            source_path: file.path,
            source_sha256: file.digest,
            source_bytes: file.bytes,
            phase: self.phase,
            offered_waves: self.ledger.offered,
            scope_members: self.ledger.members,
            scope_failures: self.ledger.failed.iter().sum::<usize>() as u64,
            failure: self.failure,
            model: self.qualified,
        })
    }
}
