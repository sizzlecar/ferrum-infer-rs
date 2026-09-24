//! Worker-only capture state; the final result survives shutdown cancellation
//! and repeated waits. This state owns no engine or worker handle.
use super::*;

pub(in crate::continuous_engine::inner::cost_observation) struct ExportSession {
    enabled: bool,
    pending: Option<ExportPlan>,
    active: Option<ProfileExporter>,
    result: Option<Result<ExportReceipt, String>>,
    audit: ExportAuditSnapshot,
}

impl ExportSession {
    pub fn new(plan: Option<ExportPlan>) -> Self {
        let status = if plan.is_some() {
            ExportStatus::Pending
        } else {
            ExportStatus::Disabled
        };
        Self {
            enabled: plan.is_some(),
            pending: plan,
            active: None,
            result: None,
            audit: ExportAuditSnapshot {
                status,
                counts: ExportCounts::default(),
                failure: None,
                unavailable_observations: 0,
                unavailable_host_stages: 0,
                counter_exhausted: false,
            },
        }
    }

    pub fn needs_observation(&self) -> bool {
        self.enabled && self.result.is_none()
    }

    fn start(&mut self) {
        if let Some(plan) = self.pending.take() {
            match ProfileExporter::start(plan) {
                Ok(exporter) => {
                    self.audit.status = ExportStatus::Active;
                    self.active = Some(exporter);
                }
                Err(error) => self.fail(error),
            }
        }
    }

    fn fail(&mut self, error: ExportError) {
        self.audit.status = ExportStatus::Failed;
        self.audit.failure = Some(match &error {
            ExportError::Io(_) => ExportFailureReason::Io,
            ExportError::Json(_) => ExportFailureReason::Json,
            ExportError::Source(_) => ExportFailureReason::Source,
            ExportError::Clock(_) => ExportFailureReason::Clock,
            ExportError::Config(_) => ExportFailureReason::Config,
            ExportError::SourceOnly { .. } => ExportFailureReason::SourceOnly,
        });
        if let Some(exporter) = &self.active {
            self.audit.counts = exporter.counts.clone();
        }
        tracing::error!(error = %error, "cost profile export failed");
        self.active = None;
        self.result = Some(Err(error.to_string()));
    }

    pub fn record_with_host(
        &mut self,
        accepted_ordinal: u64,
        observation: &model::WaveCostObservation,
        disposition: TrainingDisposition,
        prediction: PreUpdatePrediction,
        stages: Option<Arc<HostStageEvidenceV1>>,
        host: Option<(HostContentEvaluation, Option<&model::WaveCostObservation>)>,
    ) {
        self.start();
        if let Some(exporter) = &mut self.active {
            if let Err(error) = exporter.record_entry_with_host(
                accepted_ordinal,
                Some((observation, disposition, prediction)),
                stages,
                None,
                host,
            ) {
                self.fail(error);
            }
        } else {
            self.note_unavailable_observation();
            if stages.is_some() {
                self.note_unavailable_host_stages();
            }
        }
    }

    pub fn record_stages_with_host(
        &mut self,
        accepted_ordinal: u64,
        stages: Arc<HostStageEvidenceV1>,
        legacy_rejection: CostCallRejection,
        host: Option<(HostContentEvaluation, Option<&model::WaveCostObservation>)>,
    ) {
        self.start();
        if let Some(exporter) = &mut self.active {
            if let Err(error) = exporter.record_entry_with_host(
                accepted_ordinal,
                None,
                Some(stages),
                Some(legacy_rejection),
                host,
            ) {
                self.fail(error);
            }
        } else {
            self.note_unavailable_host_stages();
        }
    }

    pub fn note_unavailable_host_stages(&mut self) {
        if self.enabled {
            audit::add_one(
                &mut self.audit.unavailable_host_stages,
                &mut self.audit.counter_exhausted,
            );
        }
    }

    pub fn note_unavailable_observation(&mut self) {
        if self.enabled {
            audit::add_one(
                &mut self.audit.unavailable_observations,
                &mut self.audit.counter_exhausted,
            );
        }
    }

    pub fn export_cut(
        &mut self,
        paths: CostProfileCutPaths,
        cutoff: u64,
        clock: &dyn CostObservationClock,
        stats: CostSampleStats,
        training: &TrainingAuditSnapshot,
    ) -> Result<CostProfileCutReceipt, ExportError> {
        self.start();
        let exporter = self.active.as_ref().ok_or(ExportError::Source(
            "training cut requires an active original-observation exporter",
        ))?;
        exporter.write_cut(
            paths,
            cutoff,
            ExportClockReading::closing(clock)?,
            stats,
            training,
        )
    }

    pub fn audit_snapshot(&self) -> ExportAuditSnapshot {
        let mut snapshot = self.audit.clone();
        if let Some(exporter) = &self.active {
            snapshot.counts = exporter.counts.clone();
        }
        snapshot
    }

    pub fn finish(
        &mut self,
        clock: &dyn CostObservationClock,
        stats: CostSampleStats,
        training: TrainingAuditSnapshot,
    ) {
        self.start();
        let Some(exporter) = self.active.take() else {
            return;
        };
        self.audit.counts = exporter.counts.clone();
        let result = ExportClockReading::closing(clock)
            .and_then(|closing| exporter.finish(closing, stats, training));
        match result {
            Ok(receipt) => {
                self.audit.status = ExportStatus::Published;
                tracing::info!(profile = %receipt.profile.path.display(),
                    profile_sha256 = %receipt.profile.sha256,
                    source = %receipt.source.path.display(), source_sha256 = %receipt.source.sha256,
                    retained_samples = receipt.counts.retained_samples, coverage = receipt.coverage,
                    retained_host_content_samples = receipt.counts.retained_host_content_samples,
                    "cost profile export published");
                self.result = Some(Ok(receipt));
            }
            Err(error) => self.fail(error),
        }
    }

    pub fn check_finished(&self) -> Result<(), ferrum_types::FerrumError> {
        if !self.enabled {
            return Ok(());
        }
        match &self.result {
            Some(Ok(_)) => Ok(()),
            Some(Err(error)) => Err(ferrum_types::FerrumError::backend(error.clone())),
            None => Err(ferrum_types::FerrumError::internal(
                "cost export worker did not finalize",
            )),
        }
    }
}
