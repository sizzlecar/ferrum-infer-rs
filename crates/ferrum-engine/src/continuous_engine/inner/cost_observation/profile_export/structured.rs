//! Explicit single-scope live collector. Membership is reserved after preparation and before execute;
//! failed members remain in the source and invalidate the entire phase.
use super::selected::raw::RawSource;
use super::*;
use crate::continuous_engine::inner::calibration::CalibrationWork;
use ferrum_scheduler::implementations::continuous::cost_model::structured::{
    CalibratedStructuredModelV1, FittedStructuredModelV1, QualifiedStructuredModelV1,
    StructuredMemberBindingV1, StructuredNumericObservationV1, StructuredPartitionV1,
    StructuredPopulationV1, StructuredSettingsV1, StructuredUnknown, MODEL_REVISION,
    POPULATION_REVISION,
};
use std::num::{NonZeroU64, NonZeroUsize};

mod population;
use population::PopulationLedger;
mod source;
use source::StructuredSource;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub struct StructuredCalibrationScopeV1 {
    pub rows: NonZeroUsize,
    /// Frozen by an independent discovery, never chosen from qualification.
    pub domain_signature: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct StructuredCalibrationOptions {
    pub observations_path: PathBuf,
    pub protocol_sha256: [u8; 32],
    pub scope: StructuredCalibrationScopeV1,
    pub settings: StructuredSettingsV1,
    pub fit_members: NonZeroUsize,
    pub residual_members: NonZeroUsize,
    pub qualification_members: NonZeroUsize,
    pub maximum_offered_waves: NonZeroUsize,
    pub maximum_file_bytes: NonZeroU64,
}
impl StructuredCalibrationOptions {
    fn validate(&self) -> Result<(), ExportError> {
        self.settings.validate().map_err(numeric_error)?;
        if self.protocol_sha256 == [0; 32]
            || self.scope.domain_signature == [0; 32]
            || self.scope.rows.get() > 128
            || self.observations_path.as_os_str().is_empty()
            || self.maximum_offered_waves.get() > 65_536
        {
            return Err(ExportError::Config(
                "invalid structured capture identity or bound".into(),
            ));
        }
        // Includes live sample vectors, temporary fitting rows/residuals, two
        // retained support populations and bounded physical host rows. This
        // conservative admission bound is separate from raw source byte size.
        let numeric_bytes = self
            .settings
            .max_phase_samples
            .checked_mul(self.settings.max_axes)
            .and_then(|n| n.checked_mul(12 * std::mem::size_of::<f64>()))
            .and_then(|n| {
                self.settings
                    .max_phase_samples
                    .checked_mul(128)
                    .and_then(|rows| {
                        rows.checked_mul(4 * std::mem::size_of::<StructuredHostRowV1>())
                    })
                    .and_then(|host| n.checked_add(host))
            });
        if numeric_bytes.is_none_or(|bytes| bytes > 128 * 1024 * 1024) {
            return Err(ExportError::Config(
                "structured numeric storage exceeds 128 MiB bound".into(),
            ));
        }
        let counts = self.counts();
        if counts
            .iter()
            .any(|n| *n < self.settings.min_phase_samples || *n > self.settings.max_phase_samples)
            || counts[2] < self.scope.rows.get() + 1
            || counts.iter().sum::<usize>() > self.maximum_offered_waves.get()
        {
            return Err(ExportError::Config(
                "structured phase population exceeds declared bounds".into(),
            ));
        }
        Ok(())
    }
    fn counts(&self) -> [usize; 3] {
        [
            self.fit_members.get(),
            self.residual_members.get(),
            self.qualification_members.get(),
        ]
    }
    fn rule_signature(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(POPULATION_REVISION.as_bytes());
        digest.update(b"prepared-wave; all-decode-with-generated-history; before-execute; outcome-independent; one-scope\0");
        digest.update((self.scope.rows.get() as u64).to_le_bytes());
        digest.update(self.scope.domain_signature);
        digest.finalize().into()
    }
    fn protocol_signature(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-live-source.v1\0");
        digest.update(MODEL_REVISION.as_bytes());
        digest.update(self.protocol_sha256);
        digest.update(self.rule_signature());
        for value in self.counts().into_iter().map(|n| n as u64).chain([
            self.settings.min_phase_samples as u64,
            self.settings.min_fit_redundancy as u64,
            self.settings.max_phase_samples as u64,
            self.settings.max_axes as u64,
            self.settings.max_rank as u64,
            self.settings.max_wave_ns,
            self.settings.max_sample_age_ns,
            self.settings.static_margin_ns,
            self.maximum_offered_waves.get() as u64,
            self.maximum_file_bytes.get(),
        ]) {
            digest.update(value.to_le_bytes());
        }
        digest.finalize().into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredCapturePhase {
    Fit,
    Residual,
    Qualification,
    Qualified,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct StructuredPhaseFreezeReceipt {
    pub capture_identity: [u8; 32],
    pub protocol: [u8; 32],
    pub rule_signature: [u8; 32],
    pub phase: StructuredCapturePhase,
    pub accepted_fifo_cutoff: u64,
    pub member_cutoff: u64,
    pub source_prefix_bytes: u64,
    pub source_prefix_sha256: [u8; 32],
    pub frozen_at_ns: u64,
    pub parameters_sha256: [u8; 32],
}

#[derive(Debug, Clone, Serialize)]
pub struct StructuredCalibrationProgress {
    pub phase: StructuredCapturePhase,
    pub offered_attempts: u64,
    pub required_members: [usize; 3],
    pub reserved_members: [usize; 3],
    pub completed_members: [usize; 3],
    pub failed_members: [usize; 3],
    pub failure: Option<String>,
}

pub struct StructuredCalibrationArtifact {
    pub source_path: PathBuf,
    pub source_sha256: [u8; 32],
    pub source_bytes: u64,
    pub phase: StructuredCapturePhase,
    pub offered_waves: u64,
    pub scope_members: u64,
    pub scope_failures: u64,
    pub failure: Option<String>,
    /// Numerical diagnostic state only; no serving model/epoch is installed.
    pub model: Option<QualifiedStructuredModelV1>,
}

pub(in crate::continuous_engine::inner) struct StructuredCalibrationCollector {
    options: StructuredCalibrationOptions,
    binding: Arc<StructuredCaptureSessionBinding>,
    clock: Arc<dyn CostObservationClock>,
    source: StructuredSource,
    ledger: PopulationLedger,
    phase: StructuredCapturePhase,
    samples: Vec<StructuredNumericObservationV1>,
    fitted: Option<FittedStructuredModelV1>,
    calibrated: Option<CalibratedStructuredModelV1>,
    qualified: Option<QualifiedStructuredModelV1>,
    failure: Option<String>,
}

fn numeric_error(reason: StructuredUnknown) -> ExportError {
    ExportError::Config(format!("structured capture: {reason:?}"))
}

impl StructuredCalibrationCollector {
    pub fn new(
        options: StructuredCalibrationOptions,
        fingerprint: model::ExecutionFingerprint,
        clock: Arc<dyn CostObservationClock>,
        initial_fifo_cutoff: u64,
    ) -> Result<Self, ExportError> {
        options.validate()?;
        let binding = Arc::new(
            StructuredCaptureSessionBinding::new(
                options.protocol_signature(),
                fingerprint.clone(),
                clock.as_ref(),
            )
            .map_err(numeric_error)?,
        );
        let mut source =
            StructuredSource::create(&options.observations_path, options.maximum_file_bytes.get())?;
        let header=source.record(&serde_json::json!({
            "artifact_type":"ferrum.structured-live-source", "schema_version":2,
            "model_revision":MODEL_REVISION, "population_revision":POPULATION_REVISION,
            "capture_identity":binding.identity(),"protocol":binding.protocol(),
            "declared_protocol":options.protocol_sha256,"rule_signature":options.rule_signature(),
            "fingerprint":profile::ProfileFingerprint::from(&fingerprint),
            "producer":ProducerIdentity::current()?,"opening":ExportClockReading::opening(clock.as_ref())?,
            "opened_at_ns":binding.opened_at_ns(),"initial_fifo_cutoff":initial_fifo_cutoff,
            "scope":{"rows":options.scope.rows.get(),"domain":options.scope.domain_signature},
            "phase_members":options.counts(),"maximum_offered_waves":options.maximum_offered_waves.get(),
            "maximum_file_bytes":options.maximum_file_bytes.get(),
            "settings":{"min_samples":options.settings.min_phase_samples,"redundancy":options.settings.min_fit_redundancy,
                "max_phase_samples":options.settings.max_phase_samples,"max_axes":options.settings.max_axes,
                "max_rank":options.settings.max_rank,"max_wave_ns":options.settings.max_wave_ns,
                "max_age_ns":options.settings.max_sample_age_ns,"margin_ns":options.settings.static_margin_ns}
        }));
        if let Err(error) = header {
            return Err(source.incomplete_error().unwrap_or(error));
        }
        let ledger = PopulationLedger::new(&options, initial_fifo_cutoff);
        Ok(Self {
            options,
            binding,
            clock,
            source,
            ledger,
            phase: StructuredCapturePhase::Fit,
            samples: Vec::new(),
            fitted: None,
            calibrated: None,
            qualified: None,
            failure: None,
        })
    }

    pub fn progress(&self) -> StructuredCalibrationProgress {
        let (reserved_members, completed_members, failed_members) = self.ledger.phase_counts();
        StructuredCalibrationProgress {
            phase: self.phase,
            offered_attempts: self.ledger.offered(),
            required_members: self.options.counts(),
            reserved_members,
            completed_members,
            failed_members,
            failure: self.failure.clone(),
        }
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
        self.phase = StructuredCapturePhase::Failed;
        self.qualified = None;
        self.fitted = None;
        self.calibrated = None;
        self.samples.clear();
        self.ledger.abandon_pending();
        self.failure.get_or_insert(reason);
    }

    pub fn offer(&mut self, work: &[CalibrationWork]) -> Result<(), ExportError> {
        if self.phase == StructuredCapturePhase::Qualified
            || self.phase == StructuredCapturePhase::Failed
        {
            return Err(ExportError::Source("structured collector phase is closed"));
        }
        let attempt = self.ledger.offer(work, self.phase)?;
        self.source.record(&serde_json::json!({"kind":"offered","offered":attempt.offered,
            "member_candidate":attempt.member_candidate,"phase":self.phase,"rows":work.iter().map(|row|
                serde_json::json!({"request_id":row.frontier.request_id().to_string(),
                    "owner":row.frontier.owner_incarnation().get(),"generation":row.frontier.work_generation().get(),
                    "generated":row.frontier.generated_tokens(),"decode":matches!(row.work,ActualRowWork::Decode{..})})
                ).collect::<Vec<_>>() }))?;
        Ok(())
    }

    pub fn reserve_prepared(&mut self) -> Result<(), ExportError> {
        let reservation = self.ledger.reserve_prepared(Arc::clone(&self.binding))?;
        self.source.record(&serde_json::json!({"kind":"reserved","offered":reservation.offered,
            "member":reservation.member,"phase":reservation.phase,"boundary":"prepared_before_execute"}))
    }

    pub fn pending_capture(&self) -> Result<Arc<CostCalibrationCapture>, ExportError> {
        self.ledger.pending_capture()
    }

    pub fn complete_unsubmitted(&mut self, reason: &str) -> Result<(), ExportError> {
        if let Some(attempt) = self.ledger.take_unprepared_attempt() {
            return self.source.record(
                &serde_json::json!({"kind":"preparation_unavailable","offered":attempt.offered,
                "member_candidate":attempt.member_candidate,"phase":attempt.phase,"reason":reason}),
            );
        }
        let reserved = self.ledger.take_pending(None)?;
        self.ledger.fail_member(&reserved);
        self.source.record(
            &serde_json::json!({"kind":"unsubmitted","offered":reserved.offered,
            "member":reserved.member,"phase":reserved.phase,"reason":reason}),
        )
    }

    pub fn complete(
        &mut self,
        capture: &Arc<CostCalibrationCapture>,
        reconciled: bool,
    ) -> Result<(), ExportError> {
        let reserved = self.ledger.take_pending(Some(capture))?;
        let queue = capture.host_stage_queue();
        let fifo = self.ledger.accept_fifo(queue);
        let mut converted = fifo.and_then(|_| {
            super::super::trainer::structured::capture_numeric_observation(capture, reconciled)
        });
        if reserved.member.is_some() {
            if let Ok(observation) = &mut converted {
                if observation.input.domain_signature() != &self.options.scope.domain_signature {
                    converted = Err(StructuredUnknown::WrongDomain);
                } else if observation.input.regression_axes().len() > self.options.settings.max_axes
                    || observation.input.joint_support_coordinates().len()
                        > self.options.settings.max_axes
                {
                    converted = Err(StructuredUnknown::Capacity);
                } else {
                    observation.membership = Some(
                        StructuredMemberBindingV1::new(
                            self.options.rule_signature(),
                            reserved.offered,
                            reserved.member.unwrap(),
                        )
                        .map_err(numeric_error)?,
                    );
                }
            }
        }
        let stages = capture.host_stages();
        let recorded = self.source.record(
            &serde_json::json!({"kind":"completed","offered":reserved.offered,
            "member":reserved.member,"phase":reserved.phase,"queue":queue,"reconciled":reconciled,
            "host_stages":stages.as_ref().map(|value|value.structured_diagnostic_view()),
            "selected_independent_attention_v2":stages.as_ref().and_then(|value|value.statistical_evidence.as_ref())
                .and_then(|value|value.independent_attention_v2()),
            "selected_structured_capture":stages.as_ref().and_then(|value|value.statistical_evidence.as_ref())
                .and_then(|value|value.structured_capture()).map(|value|value.map(AsRef::as_ref)),
            "numeric":converted.as_ref().ok().map(|s|serde_json::json!({"fifo":s.ordinal,
                "call_id":s.call_id,"observed_at_ns":s.observed_at_ns,"wall_ns":s.wall_ns,
                "domain":s.input.domain_signature(),"basis":s.input.regression_axes(),
                "support":s.input.joint_support_coordinates()})),
            "conversion_error":converted.as_ref().err().map(|e|format!("{e:?}"))}),
        );
        if let Err(error) = recorded {
            self.ledger.fail_member(&reserved);
            return Err(error);
        }
        match (reserved.member, converted) {
            (Some(_), Ok(observation)) => {
                self.ledger.complete_member(&reserved);
                self.samples.push(observation);
            }
            (Some(_), Err(_)) => self.ledger.fail_member(&reserved),
            (None, _) => {} // Explicitly excluded before the result was known.
        }
        Ok(())
    }

    fn partition(&self) -> StructuredPartitionV1 {
        let [a, b, c] = self.options.counts().map(|n| n as u64);
        StructuredPartitionV1 {
            source: self.binding.identity(),
            protocol: self.binding.protocol(),
            population: StructuredPopulationV1::ReservedMembers {
                rule_signature: self.options.rule_signature(),
            },
            fit_through: a,
            residual_through: a + b,
            qualification_through: a + b + c,
        }
    }

    pub fn freeze(
        &mut self,
        fifo_cutoff: u64,
    ) -> Result<StructuredPhaseFreezeReceipt, ExportError> {
        let result = self.freeze_inner(fifo_cutoff);
        if let Err(error) = &result {
            self.invalidate(error.to_string());
        }
        result
    }
    fn freeze_inner(
        &mut self,
        fifo_cutoff: u64,
    ) -> Result<StructuredPhaseFreezeReceipt, ExportError> {
        let now = self
            .clock
            .now_ns()
            .ok_or(ExportError::Clock("structured freeze clock unavailable"))?;
        let current = self.phase;
        if let Err(reason) = self.ledger.freeze(current, fifo_cutoff, self.samples.len()) {
            self.invalidate(format!("{reason:?}"));
            self.source.record(&serde_json::json!({"kind":"phase_failed","phase":current,"reason":format!("{reason:?}"),"accepted_fifo_cutoff":fifo_cutoff}))?;
            return Err(numeric_error(reason));
        }
        let frozen = match current {
            StructuredCapturePhase::Fit => FittedStructuredModelV1::fit(
                self.binding.fingerprint().clone(),
                self.options.settings.clone(),
                self.partition(),
                &self.samples,
                now,
            )
            .map(|model| {
                let signature = model.parameters_signature();
                self.fitted = Some(model);
                signature
            }),
            StructuredCapturePhase::Residual => self
                .fitted
                .take()
                .ok_or(StructuredUnknown::PhaseLeakage)
                .and_then(|model| model.calibrate(&self.samples, now))
                .map(|model| {
                    let signature = model.parameters_signature();
                    self.calibrated = Some(model);
                    signature
                }),
            StructuredCapturePhase::Qualification => self
                .calibrated
                .take()
                .ok_or(StructuredUnknown::PhaseLeakage)
                .and_then(|model| model.qualify(&self.samples, now))
                .map(|model| {
                    let signature = model.parameters_signature();
                    self.qualified = Some(model);
                    signature
                }),
            _ => Err(StructuredUnknown::PhaseLeakage),
        };
        let parameters = match frozen {
            Ok(signature) => signature,
            Err(reason) => {
                self.phase = StructuredCapturePhase::Failed;
                self.failure = Some(format!("{reason:?}"));
                self.source.record(&serde_json::json!({"kind":"phase_failed","phase":current,"reason":format!("{reason:?}")}))?;
                return Err(numeric_error(reason));
            }
        };
        self.source.flush()?;
        let receipt = StructuredPhaseFreezeReceipt {
            capture_identity: self.binding.identity(),
            protocol: self.binding.protocol(),
            rule_signature: self.options.rule_signature(),
            phase: current,
            accepted_fifo_cutoff: fifo_cutoff,
            member_cutoff: self.ledger.members(),
            source_prefix_bytes: self.source.bytes(),
            source_prefix_sha256: self.source.prefix_digest(),
            frozen_at_ns: now,
            parameters_sha256: parameters,
        };
        self.source
            .record(&serde_json::json!({"kind":"phase_freeze","receipt":receipt}))?;
        self.samples.clear();
        self.phase = match current {
            StructuredCapturePhase::Fit => StructuredCapturePhase::Residual,
            StructuredCapturePhase::Residual => StructuredCapturePhase::Qualification,
            _ => StructuredCapturePhase::Qualified,
        };
        Ok(receipt)
    }

    pub fn finish(
        mut self,
        fifo_cutoff: u64,
    ) -> Result<StructuredCalibrationArtifact, ExportError> {
        if !self.ledger.audit_complete(fifo_cutoff) {
            self.invalidate("capture contains pending, dropped or unaudited FIFO work".into());
        }
        if let Some(error) = self.source.incomplete_error() {
            return Err(error);
        }
        if self.phase != StructuredCapturePhase::Qualified {
            self.invalidate("capture finished without complete qualification".into());
        }
        let closing = match ExportClockReading::closing(self.clock.as_ref()) {
            Ok(reading) => Some(reading),
            Err(error) => {
                self.invalidate(error.to_string());
                None
            }
        };
        let footer=self.source.record(&serde_json::json!({"kind":"footer","phase":self.phase,"failure":self.failure,
            "offered":self.ledger.offered(),"members":self.ledger.members(),"failed_members":self.ledger.failures(),
            "accepted_fifo_cutoff":fifo_cutoff,"last_captured_fifo":self.ledger.last_fifo(),
            "fifo_audit_complete":self.ledger.audit_complete(fifo_cutoff),
            "closing":closing}));
        if let Err(error) = footer {
            return Err(self.source.incomplete_error().unwrap_or(error));
        }
        let file = self.source.finish()?;
        Ok(StructuredCalibrationArtifact {
            source_path: file.path,
            source_sha256: file.digest,
            source_bytes: file.bytes,
            phase: self.phase,
            offered_waves: self.ledger.offered(),
            scope_members: self.ledger.members(),
            scope_failures: self.ledger.failures(),
            failure: self.failure,
            model: self.qualified,
        })
    }
}
