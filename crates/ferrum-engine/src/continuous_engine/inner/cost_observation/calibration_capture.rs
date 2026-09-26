//! A single call's actual observation and host receipts, with no execution authority.
//! Calibration owns one slot per durable wave; normal inference attaches none.
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
    },
    Rejected(CostCallRejection),
}

#[derive(Debug, Default)]
pub(in crate::continuous_engine) struct CostCalibrationCapture {
    value: OnceLock<Arc<CostCalibrationResult>>,
    host_stages: OnceLock<Arc<HostStageEvidenceV1>>,
    host_stage_queue: OnceLock<HostStageQueueReceipt>,
    actual_evidence_diagnostic: OnceLock<Arc<CalibrationActualEvidenceDiagnostic>>,
    claimed: AtomicBool,
    conflict: AtomicBool,
    // Set only by construction, before attach/context/execute. Existing capture
    // paths keep None and cannot be retroactively relabeled as this protocol.
    structured_sessions: Box<[Arc<StructuredCaptureSessionBinding>]>,
}

#[derive(Debug, Clone)]
pub(in crate::continuous_engine) enum CostCalibrationStatus {
    Pending,
    Complete(Arc<CostCalibrationResult>),
    ConflictingCalls,
}

impl CostCalibrationCapture {
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
    }

    fn mark_conflict(&self) {
        self.conflict.store(true, Ordering::Release);
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
            self.calibration_capture = Some(capture);
        }
    }
}
