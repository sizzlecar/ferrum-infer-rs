//! A single call's actual observation and host receipts, with no execution authority.
//! Calibration owns one slot per durable wave; normal inference attaches none.
pub(in crate::continuous_engine::inner) use super::live_calibration::OriginalNoSubmissionReceipt;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::WaveCostObservation;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    OnceLock,
};

mod diagnostic;
mod structured_session;
pub use diagnostic::{CalibrationActualEvidenceDiagnostic, CalibrationActualWaveUnknown};
pub(in crate::continuous_engine::inner) use structured_session::StructuredCaptureSessionBinding;

#[derive(Debug)]
pub(in crate::continuous_engine) enum CostCalibrationResult {
    Observed {
        sample: Box<WaveCostObservation>,
        /// Physical recorder order, retaining the actual request/owner/row map.
        actual_rows: Vec<ActualWaveRow>,
        /// Joined by full identity to actual_rows, in the same physical order.
        /// These passed make_sample's physical/host identity and work checks.
        commits: Vec<HostCommitEvidence>,
        /// Original call participants, joined into the same physical order.
        /// Missing features remain unknown; the report never reconstructs them.
        host_features: Vec<Option<HostCostFeaturesV1>>,
        /// The actual sink acceptance position, not a call or source-record id.
        /// Dropped observations have no accepted queue position.
        accepted_ordinal: Option<u64>,
        /// A valid measurement is not proof it entered training or export.
        disposition: CostCallDisposition,
        observation_memory: Option<Arc<super::memory::ObservationBytePermit>>,
    },
    Rejected(CostCallRejection),
    UnresolvedDropped(CostSampleDrop),
}

#[derive(Debug, Default)]
pub(in crate::continuous_engine) struct CostCalibrationCapture {
    value: OnceLock<Arc<CostCalibrationResult>>,
    host_stages: OnceLock<Arc<HostStageEvidenceV1>>,
    no_submission: OnceLock<Arc<OriginalNoSubmissionReceipt>>,
    private_settlement: OnceLock<Arc<CompletePrivateCalibrationSettlement>>,
    host_stage_queue: OnceLock<HostStageQueueReceipt>,
    actual_evidence_diagnostic: OnceLock<Arc<CalibrationActualEvidenceDiagnostic>>,
    original_route_capture: bool,
    startup_readiness: bool,
    claimed: AtomicBool,
    conflict: AtomicBool,
    resolved: tokio::sync::Notify,
    structured_projection: OnceLock<SharedStructuredProjection>,
    actual_projection: OnceLock<SharedActualProjection>,
    // Set only by construction, before attach/context/execute. Existing capture
    // paths keep None and cannot be retroactively relabeled as this protocol.
    structured_sessions: Box<[Arc<StructuredCaptureSessionBinding>]>,
}

struct SharedActualProjection(
    Result<
        Arc<trainer::host_content::statistical::CompleteSelectedObservation>,
        ferrum_scheduler::implementations::continuous::cost_model::statistical::model::ModelUnknown,
    >,
);
impl std::fmt::Debug for SharedActualProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedActualProjection")
            .field("available", &self.0.is_ok())
            .finish()
    }
}

struct SharedStructuredProjection(
    Result<Arc<super::resolved::StructuredActual>, super::resolved::ProjectionError>,
);
impl std::fmt::Debug for SharedStructuredProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedStructuredProjection")
            .field("available", &self.0.is_ok())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub(in crate::continuous_engine) enum CostCalibrationStatus {
    Pending,
    Complete(Arc<CostCalibrationResult>),
    ConflictingCalls,
}

impl CostCalibrationCapture {
    /// Only the drained, exclusive startup-inventory session attaches this
    /// fresh identity before submission. It grants no feedback exclusion by
    /// itself: the original recorder and host settlement must prove completion.
    pub(in crate::continuous_engine::inner) fn for_startup_readiness() -> Self {
        Self {
            original_route_capture: true,
            startup_readiness: true,
            ..Self::default()
        }
    }

    pub(super) fn requests_startup_readiness(&self) -> bool {
        self.startup_readiness && self.requests_original_route()
    }

    /// Source8 calls this only while fixing its private capture, before attach.
    /// Existing manual captures retain their original route population behavior.
    pub(in crate::continuous_engine::inner) fn with_original_route_capture(
        mut self,
    ) -> Result<Self, ferrum_types::FerrumError> {
        if self.claimed.load(Ordering::Acquire)
            || self.conflict.load(Ordering::Acquire)
            || self.value.get().is_some()
        {
            return Err(ferrum_types::FerrumError::invalid_request(
                "original route capture must be fixed before call attachment",
            ));
        }
        self.original_route_capture = true;
        Ok(self)
    }
    pub(super) fn requests_original_route(&self) -> bool {
        self.original_route_capture && !self.conflict.load(Ordering::Acquire)
    }
    pub(in crate::continuous_engine::inner) fn no_submission_proof(
        &self,
    ) -> Option<Arc<OriginalNoSubmissionReceipt>> {
        if self.conflict.load(Ordering::Acquire) {
            None
        } else {
            self.no_submission.get().map(Arc::clone)
        }
    }
    pub(super) fn complete_no_submission(&self, proof: Arc<OriginalNoSubmissionReceipt>) {
        if self.no_submission.set(proof).is_err() {
            self.mark_conflict();
        }
    }
    pub(super) fn retained_diagnostic_bytes(&self) -> Option<usize> {
        self.actual_evidence_diagnostic
            .get()
            .map_or(Some(0), |v| v.retained_payload_bytes())
    }
    pub(super) fn complete_actual_projection(
        &self,
        actual: Result<Arc<trainer::host_content::statistical::CompleteSelectedObservation>,
        ferrum_scheduler::implementations::continuous::cost_model::statistical::model::ModelUnknown>,
    ) {
        if self
            .actual_projection
            .set(SharedActualProjection(actual))
            .is_err()
        {
            self.mark_conflict();
        }
    }
    /// Only the original immutable stages can reuse the FIFO validation.
    pub(super) fn actual_projection(&self, stages: &Arc<HostStageEvidenceV1>) -> Option<Result<Arc<trainer::host_content::statistical::CompleteSelectedObservation>,
    ferrum_scheduler::implementations::continuous::cost_model::statistical::model::ModelUnknown>>{
        if self.conflict.load(Ordering::Acquire)
            || !self
                .host_stages
                .get()
                .is_some_and(|original| Arc::ptr_eq(original, stages))
        {
            return None;
        }
        self.actual_projection.get().map(|value| value.0.clone())
    }
    pub(super) fn complete_structured_projection(
        &self,
        projection: Result<
            Arc<super::resolved::StructuredActual>,
            super::resolved::ProjectionError,
        >,
    ) {
        if self
            .structured_projection
            .set(SharedStructuredProjection(projection))
            .is_err()
        {
            self.mark_conflict();
        }
    }
    /// A getter for a private frozen result, never a lazy projector. A caller
    /// replacing or mutating the diagnostic stages must use full validation.
    pub(super) fn structured_projection(
        &self,
        stages: &Arc<HostStageEvidenceV1>,
    ) -> Option<Result<Arc<super::resolved::StructuredActual>, super::resolved::ProjectionError>>
    {
        if self.conflict.load(Ordering::Acquire)
            || !self
                .host_stages
                .get()
                .is_some_and(|original| Arc::ptr_eq(original, stages))
        {
            return None;
        }
        self.structured_projection.get().map(|v| v.0.clone())
    }
    pub(super) fn retained_bytes(&self) -> Option<usize> {
        std::mem::size_of::<Self>().checked_add(self.structured_sessions.len().checked_mul(
            std::mem::size_of::<StructuredCaptureSessionBinding>()
                + std::mem::size_of::<Arc<StructuredCaptureSessionBinding>>()
                + 2 * std::mem::size_of::<usize>(),
        )?)
    }
    pub fn actual_evidence_diagnostic(&self) -> Option<Arc<CalibrationActualEvidenceDiagnostic>> {
        if self.conflict.load(Ordering::Acquire) {
            None
        } else {
            self.actual_evidence_diagnostic.get().map(Arc::clone)
        }
    }
    pub(in crate::continuous_engine::inner) fn for_structured_session(
        session: Arc<StructuredCaptureSessionBinding>,
    ) -> Self {
        Self {
            structured_sessions: Box::new([session]),
            ..Self::default()
        }
    }

    pub(super) fn structured_session(&self) -> Option<&StructuredCaptureSessionBinding> {
        match self.structured_sessions.as_ref() {
            [session] => Some(session),
            _ => None,
        }
    }

    /// Immutable population attached before the one real call. This never
    /// accepts a receipt or mutates an already attached capture.
    pub(in crate::continuous_engine::inner) fn for_structured_sessions(
        sessions: Vec<Arc<StructuredCaptureSessionBinding>>,
    ) -> Result<
        Self,
        ferrum_scheduler::implementations::continuous::cost_model::structured::StructuredUnknown,
    > {
        use ferrum_scheduler::implementations::continuous::cost_model::structured::StructuredUnknown as U;
        if sessions.is_empty() || sessions.len() > 128 {
            return Err(U::Capacity);
        }
        for (index, session) in sessions.iter().enumerate() {
            if session.fingerprint() != sessions[0].fingerprint()
                || sessions[..index]
                    .iter()
                    .any(|s| s.identity() == session.identity())
            {
                return Err(U::WrongSource);
            }
        }
        Ok(Self {
            structured_sessions: sessions.into_boxed_slice(),
            ..Self::default()
        })
    }

    pub(super) fn structured_sessions(&self) -> &[Arc<StructuredCaptureSessionBinding>] {
        &self.structured_sessions
    }

    pub fn host_stage_queue(&self) -> Option<HostStageQueueReceipt> {
        if self.conflict.load(Ordering::Acquire) {
            None
        } else {
            self.host_stage_queue.get().copied()
        }
    }

    pub(super) fn complete_host_stage_queue(&self, receipt: HostStageQueueReceipt) {
        if self.host_stage_queue.set(receipt).is_err() {
            self.mark_conflict();
        }
    }
    pub fn host_stages(&self) -> Option<Arc<HostStageEvidenceV1>> {
        if self.conflict.load(Ordering::Acquire) {
            None
        } else {
            self.host_stages.get().map(Arc::clone)
        }
    }

    pub(in crate::continuous_engine::inner) fn private_prefix_settlement(
        &self,
        stages: &Arc<HostStageEvidenceV1>,
    ) -> Option<Arc<CompletePrivateCalibrationSettlement>> {
        if self.conflict.load(Ordering::Acquire)
            || !self
                .host_stages
                .get()
                .is_some_and(|original| Arc::ptr_eq(original, stages))
        {
            return None;
        }
        let proof = self.private_settlement.get()?;
        proof.prefix_observed_at(stages)?;
        Some(Arc::clone(proof))
    }
    pub(super) fn complete_private_settlement(&self, proof: CompletePrivateCalibrationSettlement) {
        if self
            .host_stages
            .get()
            .is_some_and(|stages| proof.prefix_observed_at(stages).is_some())
            && self.private_settlement.set(Arc::new(proof)).is_err()
        {
            self.mark_conflict();
        }
    }
    pub(super) fn complete_host_stages(&self, stages: Arc<HostStageEvidenceV1>) {
        if self.host_stages.set(stages).is_err() {
            self.mark_conflict();
        }
    }
    pub fn status(&self) -> CostCalibrationStatus {
        if self.conflict.load(Ordering::Acquire) {
            CostCalibrationStatus::ConflictingCalls
        } else if let Some(value) = self.value.get() {
            CostCalibrationStatus::Complete(Arc::clone(value))
        } else {
            CostCalibrationStatus::Pending
        }
    }

    pub(super) fn complete(&self, value: CostCalibrationResult) {
        if self.value.set(Arc::new(value)).is_err() {
            self.mark_conflict();
        }
        self.resolved.notify_waiters();
    }
    pub(super) fn complete_if_pending(&self, value: CostCalibrationResult) {
        let _ = self.value.set(Arc::new(value));
        self.resolved.notify_waiters();
    }
    /// Waits for the same FIFO consumer; cancellation does not withdraw facts.
    pub(in crate::continuous_engine) async fn wait_resolved(&self) {
        if !self.claimed.load(Ordering::Acquire) {
            return;
        }
        loop {
            let notified = self.resolved.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !matches!(self.status(), CostCalibrationStatus::Pending) {
                return;
            }
            notified.await;
        }
    }

    fn mark_conflict(&self) {
        self.conflict.store(true, Ordering::Release);
        self.resolved.notify_waiters();
    }

    fn claim(&self) -> bool {
        if self.claimed.swap(true, Ordering::AcqRel) {
            self.mark_conflict();
            false
        } else {
            true
        }
    }
}

impl EngineCostCall {
    /// Attach before constructing the actual executor observation context.
    /// Reusing a slot never overwrites earlier evidence or affects inference.
    pub fn attach_calibration_capture(&mut self, capture: Arc<CostCalibrationCapture>) {
        if self.context_created || self.calibration_capture.is_some() {
            capture.mark_conflict();
            if let Some(previous) = &self.calibration_capture {
                previous.mark_conflict();
            }
        } else if capture.claim() {
            if capture.requests_original_route()
                && tracing::enabled!(
                    target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                    tracing::Level::DEBUG
                )
            {
                // Fixed-size passive facts only. A private capture does not
                // acquire a live ticket or change route/settlement authority.
                self.recorder.enable_route_diagnostics();
            }
            self.calibration_capture = Some(capture);
        }
    }
}
