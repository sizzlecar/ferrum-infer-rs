//! One physical observation, immutable pre-execution child populations.
//! Sources stay schema3. A group is not a catalog or execution permission.
use super::*;
#[path = "group/options.rs"]
mod options;
pub use options::{StructuredCalibrationGroupLimitsV2, StructuredCalibrationGroupOptionsV2};

pub struct StructuredCalibrationGroupArtifactV2 {
    pub fingerprint: model::ExecutionFingerprint,
    pub children: Vec<StructuredCalibrationArtifactV2>,
    pub failure: Option<String>,
}
pub(in crate::continuous_engine::inner) struct StructuredCalibrationGroupV2 {
    children: Vec<StructuredCalibrationCollectorV2>,
    clock: Arc<dyn CostObservationClock>,
    pending_prepared: Option<Arc<PreparedStructuredFactsV2>>,
    failure: Option<String>,
}
impl StructuredCalibrationGroupV2 {
    pub fn new(
        options: StructuredCalibrationGroupOptionsV2,
        fingerprint: model::ExecutionFingerprint,
        clock: Arc<dyn CostObservationClock>,
        cutoff: u64,
    ) -> Result<Self, ExportError> {
        options.validate()?;
        let children = options
            .children
            .into_iter()
            .map(|options| {
                StructuredCalibrationCollectorV2::new(
                    options,
                    fingerprint.clone(),
                    Arc::clone(&clock),
                    cutoff,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            children,
            clock,
            pending_prepared: None,
            failure: None,
        })
    }
    pub fn progress(&self) -> Vec<StructuredCalibrationProgress> {
        self.children.iter().map(|c| c.progress()).collect()
    }
    pub fn coverage(&self) -> Result<Vec<StructuredCoverageReportV2>, ExportError> {
        self.children.iter().map(|c| c.coverage()).collect()
    }
    pub fn collecting(&self) -> bool {
        self.failure.is_none() && self.children.iter().all(|c| c.collecting())
    }
    pub fn invalidate(&mut self, reason: String) {
        // Retain every reservation before closing even if no physical work was
        // submitted. I/O poison remains explicit, never a skipped child.
        let _ = self.flush_prepared();
        self.failure.get_or_insert_with(|| reason.clone());
        for child in &mut self.children {
            child.invalidate(reason.clone());
        }
    }
    fn each(
        &mut self,
        mut f: impl FnMut(&mut StructuredCalibrationCollectorV2) -> Result<(), ExportError>,
    ) -> Result<(), ExportError> {
        if !self.collecting() {
            return Err(ExportError::Source("structured group is closed"));
        }
        let mut failure = None;
        for child in &mut self.children {
            if let Err(error) = f(child) {
                failure.get_or_insert(error);
            }
        }
        match failure {
            Some(error) => {
                self.invalidate(error.to_string());
                Err(error)
            }
            None => Ok(()),
        }
    }
    pub fn begin_cohort(&mut self, ordinal: usize) -> Result<(), ExportError> {
        self.each(|c| c.begin_cohort(ordinal))
    }
    pub fn admitted(&mut self, id: RequestId, maximum: u64) -> Result<(), ExportError> {
        self.each(|c| c.admitted(id.clone(), maximum))
    }
    pub fn end_cohort(&mut self) -> Result<(), ExportError> {
        self.each(|c| c.end_cohort())
    }
    pub fn offer(&mut self, work: &[CalibrationWork]) -> Result<(), ExportError> {
        self.each(|c| c.offer(work))
    }
    pub fn reserve_prepared(
        &mut self,
        prepared: PreparedStructuredFactsV2,
    ) -> Result<(), ExportError> {
        if !self.collecting() || self.pending_prepared.is_some() {
            return Err(ExportError::Source(
                "structured group has an outstanding reservation",
            ));
        }
        let owner = prepared.validate().map_err(numeric_error)?;
        let prepared = Arc::new(prepared);
        let capture = Arc::new(
            CostCalibrationCapture::for_structured_sessions(
                self.children
                    .iter()
                    .map(|c| Arc::clone(&c.binding))
                    .collect(),
            )
            .map_err(numeric_error)?,
        );
        self.pending_prepared = Some(Arc::clone(&prepared));
        self.each(|child| {
            child.cohorts.prepared(&prepared)?;
            child.ledger.reserve_shared(
                Arc::clone(&prepared),
                &owner,
                &child.options.membership_rule,
                child.options.phase_members,
                Arc::clone(&capture),
            )?;
            // Reserve every child before execution. The original measured wall
            // starts later in dispatch_controller_wave; diagnostic projection
            // and membership assignment do not shift that boundary. Defer the
            // shared JSON output until the original receipt is finalized.
            Ok(())
        })
    }
    pub fn pending_capture(&self) -> Result<Arc<CostCalibrationCapture>, ExportError> {
        self.children[0].pending_capture()
    }
    fn flush_prepared(&mut self) -> Result<(), ExportError> {
        let Some(prepared) = self.pending_prepared.take() else {
            return Ok(());
        };
        let wire = wire::PreparedWire::new(&prepared)?;
        #[derive(Serialize)]
        struct Reservation<'a, 'b> {
            kind: &'static str,
            offered: u64,
            member: Option<u64>,
            window: Option<u32>,
            phase: StructuredCapturePhase,
            cohort: usize,
            boundary: &'static str,
            prepared: &'a wire::PreparedWire<'b>,
        }
        let mut failure = None;
        for child in &mut self.children {
            if let Some(r) = &child.ledger.pending {
                let record = Reservation {
                    kind: "reserved",
                    offered: r.attempt.offered,
                    member: r.member,
                    window: r.window,
                    phase: r.attempt.phase,
                    cohort: r.attempt.cohort,
                    boundary: "prepared_before_execute",
                    prepared: &wire,
                };
                if let Err(error) = child.source.record_borrowed(&record) {
                    failure.get_or_insert(error);
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }
    pub fn complete_unsubmitted(&mut self, reason: &str) -> Result<(), ExportError> {
        let flush = self.flush_prepared();
        let result = self.each(|c| c.complete_unsubmitted(reason));
        match flush {
            Err(error) => {
                self.invalidate(error.to_string());
                Err(error)
            }
            Ok(()) => result,
        }
    }
    pub fn complete(
        &mut self,
        capture: &Arc<CostCalibrationCapture>,
        reconciled: bool,
    ) -> Result<(), ExportError> {
        if !self.collecting() {
            return Err(ExportError::Source("structured group is closed"));
        }
        let prepared = Arc::clone(self.pending_prepared.as_ref().ok_or(ExportError::Source(
            "structured group has no original Prepared",
        ))?);
        // Validate the one real private receipt/recipe/wall once. The bridge
        // retains only the bindings attached before execute, never child JSON.
        let actual = super::super::super::trainer::structured_v2::validate_capture_v2(
            capture, reconciled, &prepared,
        );
        // The original finalized wall is immutable before any multi-file output.
        let mut failure = self.flush_prepared().err();
        for child in &mut self.children {
            let settled = (|| {
                let reserved = child.ledger.take_pending(Some(capture))?;
                let valid = child
                    .ledger
                    .accept_fifo(capture.host_stage_queue())
                    .and_then(|()| actual.as_ref().map_err(|reason| numeric_error(*reason)));
                match valid {
                    Ok(actual) => child.complete_validated(reserved, capture, reconciled, actual),
                    Err(error) => child.completion_failed(&reserved, capture, reconciled, error),
                }
            })();
            if let Err(error) = settled {
                failure.get_or_insert(error);
            }
        }
        match failure {
            Some(error) => {
                self.invalidate(error.to_string());
                Err(error)
            }
            None => Ok(()),
        }
    }
    pub fn freeze(
        &mut self,
        cutoff: u64,
    ) -> Result<Vec<StructuredPhaseFreezeReceipt>, ExportError> {
        let result = (|| {
            if !self.collecting() || self.pending_prepared.is_some() {
                return Err(ExportError::Source(
                    "group phase is not at a complete boundary",
                ));
            }
            let phase = self.children[0].phase;
            for child in &self.children {
                if child.phase != phase {
                    return Err(ExportError::Source("group phase mismatch"));
                }
                child.check_freeze(cutoff)?;
            }
            let now = self
                .clock
                .now_ns()
                .ok_or(ExportError::Clock("original group clock unavailable"))?;
            // No result is published until every child has frozen. If numeric
            // fitting fails partway, all artifacts are marked Failed below.
            self.children
                .iter_mut()
                .map(|c| c.freeze_at(cutoff, now))
                .collect()
        })();
        if let Err(error) = &result {
            self.invalidate(error.to_string());
        }
        result
    }
    pub fn finish(
        mut self,
        cutoff: u64,
    ) -> Result<StructuredCalibrationGroupArtifactV2, ExportError> {
        if self.children.iter().any(|c| {
            c.phase != StructuredCapturePhase::Qualified || !c.ledger.audit_complete(cutoff)
        }) {
            self.invalidate("group ended without every child's complete FIFO/qualification".into());
        }
        // Read the actual common closing clock once before publishing any
        // child footer. A later child cannot silently return Failed while the
        // group reports success because its second clock read disappeared.
        let closing = match ExportClockReading::closing(self.clock.as_ref()) {
            Ok(value) => Some(value),
            Err(error) => {
                self.invalidate(error.to_string());
                None
            }
        };
        if self
            .children
            .iter()
            .any(|c| c.source.incomplete_error().is_some())
        {
            self.invalidate("a child source is incomplete at group close".into());
        }
        let mut failure = self.failure;
        let fingerprint = self.children[0].binding.fingerprint().clone();
        // Finish every file even if one writer is poisoned; never publish a
        // partial child collection as a successful catalog.
        let mut children = Vec::with_capacity(self.children.len());
        let mut error = None;
        for child in self.children {
            match child.finish_with_closing(cutoff, closing) {
                Ok(value) => {
                    if value.phase != StructuredCapturePhase::Qualified
                        || value.model.is_none()
                        || value.failure.is_some()
                    {
                        failure.get_or_insert_with(|| "a child did not finish qualified".into());
                    }
                    children.push(value);
                }
                Err(e) => {
                    error.get_or_insert(e);
                }
            }
        }
        match error {
            Some(error) => Err(error),
            None => Ok(StructuredCalibrationGroupArtifactV2 {
                fingerprint,
                children,
                failure,
            }),
        }
    }
}

#[cfg(test)]
#[path = "group/tests.rs"]
mod tests;
