//! One physical observation, immutable pre-execution child populations.
//! Explicit shared source4 deduplicates physical evidence; default remains source3.
use super::*;
#[path = "group/options.rs"]
mod options;
#[path = "group/shared.rs"]
mod shared;
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
    shared: Option<StructuredSource>,
}
impl StructuredCalibrationGroupV2 {
    pub fn new(
        options: StructuredCalibrationGroupOptionsV2,
        fingerprint: model::ExecutionFingerprint,
        clock: Arc<dyn CostObservationClock>,
        cutoff: u64,
    ) -> Result<Self, ExportError> {
        options.validate()?;
        let mut shared = options
            .shared_source
            .as_ref()
            .map(|path| {
                StructuredSource::create(path, options.children[0].maximum_file_bytes.get())
            })
            .transpose()?;
        let mut children = options
            .children
            .into_iter()
            .map(|child| {
                if shared.is_some() {
                    let source = StructuredSource::shared_child(&child.observations_path);
                    StructuredCalibrationCollectorV2::new_with_source(
                        child,
                        fingerprint.clone(),
                        Arc::clone(&clock),
                        cutoff,
                        source,
                    )
                } else {
                    StructuredCalibrationCollectorV2::new(
                        child,
                        fingerprint.clone(),
                        Arc::clone(&clock),
                        cutoff,
                    )
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(source) = &mut shared {
            // Pair one actual opening after all original child bindings exist.
            let opening = serde_json::to_value(ExportClockReading::opening(clock.as_ref())?)?;
            let headers = children
                .iter_mut()
                .map(|child| {
                    let mut records = child.source.take_staged();
                    if records.len() != 1 {
                        return Err(ExportError::Source("shared child declaration missing"));
                    }
                    let mut header = records.remove(0);
                    header["opening"] = opening.clone();
                    Ok(header)
                })
                .collect::<Result<Vec<_>, ExportError>>()?;
            let header = profile::structured_shared_source_header_v4(
                headers,
                source_limit(&children),
                options.limits.maximum_children.get(),
                options.limits.maximum_retained_numeric_bytes.get(),
                options.limits.maximum_retained_coordinates.get(),
            )
            .map_err(|e| ExportError::Config(e.to_string()))?;
            source.record(&header)?;
        }
        Ok(Self {
            children,
            clock,
            pending_prepared: None,
            failure: None,
            shared,
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
        if let Some(source) = &mut self.shared {
            let _ = source.record(&serde_json::json!({"kind":"phase_failed","reason":reason,"child_failures":[],"completed_freezes":[]}));
            for child in &mut self.children {
                child.source.take_staged();
            }
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
        if failure.is_none() {
            if let Err(error) = self.flush_common_metadata() {
                failure = Some(error);
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
        if let Some(source) = &mut self.shared {
            let first = self.children[0]
                .ledger
                .pending
                .as_ref()
                .ok_or(ExportError::Source("shared reservation missing"))?;
            let memberships = self
                .children
                .iter()
                .map(|c| {
                    let r = c
                        .ledger
                        .pending
                        .as_ref()
                        .ok_or(ExportError::Source("shared child reservation missing"))?;
                    Ok(serde_json::json!({"member":r.member,"window":r.window}))
                })
                .collect::<Result<Vec<_>, ExportError>>()?;
            return source.record(&serde_json::json!({"kind":"reserved","offered":first.attempt.offered,
                "phase":first.attempt.phase,"cohort":first.attempt.cohort,"boundary":"prepared_before_execute",
                "prepared":wire,"memberships":memberships}));
        }
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
        // The original finalized wall is immutable before any source output.
        let mut failure = self.flush_prepared().err();
        let members = self
            .children
            .iter()
            .map(|c| c.ledger.pending.as_ref().and_then(|r| r.member))
            .collect::<Vec<_>>();
        if self.shared.is_some() {
            let selected = members.iter().position(Option::is_some).unwrap_or(0);
            for (i, child) in self.children.iter_mut().enumerate() {
                child.source.collect_completed(i == selected);
            }
        }
        let mut child_failures = Vec::new();
        for (index, child) in self.children.iter_mut().enumerate() {
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
                child_failures.push(serde_json::json!({"child":index,"capture_identity":child.binding.identity(),"reason":error.to_string()}));
                failure.get_or_insert(error);
            }
        }
        if self.shared.is_some() {
            if let Some(error) = &failure {
                // Keep the one original failed physical receipt before closing.
                let _ = self.flush_shared_failed_completion(
                    &members,
                    &child_failures,
                    &error.to_string(),
                );
            } else if let Err(error) = self.flush_shared_completion(&members) {
                failure = Some(error);
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
            if self.shared.is_some() {
                return self.freeze_shared(cutoff, now);
            }
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
        if self.shared.is_some() {
            return self.finish_shared(cutoff, closing);
        }
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

fn source_limit(children: &[StructuredCalibrationCollectorV2]) -> u64 {
    children[0].options.maximum_file_bytes.get()
}
