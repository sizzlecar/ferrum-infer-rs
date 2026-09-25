use super::*;
use ferrum_interfaces::execution_cost::{ActualRowWork, ExpectedWaveWork};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};

#[derive(Debug, Clone, Copy)]
pub struct CalibrationLimits {
    maximum_requests: NonZeroUsize,
}
impl CalibrationLimits {
    pub fn new(maximum_requests: NonZeroUsize) -> Result<Self> {
        let limits = Self { maximum_requests };
        limits.validate()?;
        Ok(limits)
    }
    pub const fn maximum_requests(self) -> NonZeroUsize {
        self.maximum_requests
    }
    pub(crate) fn validate(self) -> Result<()> {
        if self.maximum_requests.get() > 4096 {
            return Err(FerrumError::config(
                "calibration request bound exceeds 4096",
            ));
        }
        Ok(())
    }
}

/// A session-bound observation, not permission to submit. Work constructors
/// retain this owner/frontier; a later step rechecks it against the live owner.
#[derive(Debug, Clone)]
pub struct CalibrationFrontier {
    pub(in crate::continuous_engine::inner) session: Arc<()>,
    pub(in crate::continuous_engine::inner) request_id: RequestId,
    pub(in crate::continuous_engine::inner) owner: NonZeroU64,
    pub(in crate::continuous_engine::inner) generation: NonZeroU64,
    pub(in crate::continuous_engine::inner) request_evidence: CalibrationRequestEvidence,
    pub(in crate::continuous_engine::inner) generated: usize,
    pub(in crate::continuous_engine::inner) prefill: Option<(usize, usize)>,
    pub(in crate::continuous_engine::inner) kv_tokens: usize,
}
impl CalibrationFrontier {
    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }
    pub const fn owner_incarnation(&self) -> NonZeroU64 {
        self.owner
    }
    pub const fn work_generation(&self) -> NonZeroU64 {
        self.generation
    }
    pub fn request_evidence(&self) -> &CalibrationRequestEvidence {
        &self.request_evidence
    }
    pub const fn generated_tokens(&self) -> usize {
        self.generated
    }
    pub const fn prefill_progress(&self) -> Option<(usize, usize)> {
        self.prefill
    }
    pub const fn kv_tokens(&self) -> usize {
        self.kv_tokens
    }

    pub fn prefill_work(&self, count: NonZeroU32) -> Result<CalibrationWork> {
        let (offset, total) = self
            .prefill
            .ok_or_else(|| FerrumError::invalid_request("frontier is not prefill"))?;
        let offset = u32::try_from(offset)
            .map_err(|_| FerrumError::invalid_request("prefill offset overflow"))?;
        let total = u32::try_from(total)
            .map_err(|_| FerrumError::invalid_request("prefill total overflow"))?;
        if offset
            .checked_add(count.get())
            .is_none_or(|end| end > total)
        {
            return Err(FerrumError::invalid_request(
                "calibration prefill exceeds observed total",
            ));
        }
        Ok(CalibrationWork {
            frontier: self.clone(),
            decode_route: CalibrationDecodeRoute::Actual,
            work: ActualRowWork::Prefill {
                offset,
                count: count.get(),
                total_prompt_tokens: total,
            },
        })
    }
    pub fn decode_work(&self) -> Result<CalibrationWork> {
        self.decode_work_with_route(CalibrationDecodeRoute::Actual)
    }

    /// Select a real readback/sampling route for diagnostic calibration. The
    /// sequence's sampler, token history, UTF-8 state and output budget stay
    /// unchanged; FullLogits is an actual executor request, not a relabeling.
    pub fn decode_work_with_route(
        &self,
        decode_route: CalibrationDecodeRoute,
    ) -> Result<CalibrationWork> {
        if self.prefill.is_some() {
            return Err(FerrumError::invalid_request(
                "frontier has not completed prefill",
            ));
        }
        let kv_tokens = u32::try_from(self.kv_tokens)
            .map_err(|_| FerrumError::invalid_request("decode context overflow"))?;
        Ok(CalibrationWork {
            frontier: self.clone(),
            decode_route,
            work: ActualRowWork::Decode { kv_tokens },
        })
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationDecodeRoute {
    #[default]
    Actual,
    FullLogits,
}

#[derive(Debug, Clone)]
pub struct CalibrationWork {
    pub(in crate::continuous_engine::inner) frontier: CalibrationFrontier,
    pub(in crate::continuous_engine::inner) work: ActualRowWork,
    pub(in crate::continuous_engine::inner) decode_route: CalibrationDecodeRoute,
}
impl CalibrationWork {
    pub fn frontier(&self) -> &CalibrationFrontier {
        &self.frontier
    }
    pub const fn work(&self) -> ActualRowWork {
        self.work
    }
}

#[derive(Debug)]
pub enum CalibrationAction {
    /// One physical admission probe, with at most its one maintenance action.
    AdmitOne,
    /// Consume at most one retained capacity continuation. Never submits a model wave.
    Maintenance,
    /// Execute exactly these rows, reordered only by actual resource authority.
    Wave(Vec<CalibrationWork>),
    /// Reconcile an earlier call whose waiter was dropped, without another wave.
    Reap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationBlockReason {
    NoPendingWork,
    AdmissionUnavailable,
    MaintenanceUnavailable,
    /// The actual bounded resource read failed. No row was published and no
    /// executor was entered. Keep capacity/identity/limit failures distinct
    /// from a nonblocking read miss; callers must not treat every Unknown as
    /// transient lock contention.
    ResourceUnavailable(ferrum_interfaces::vnext::ResourcePlanningUnknown),
    SelectionUnavailable(&'static str),
    PublicationUnavailable,
    PreviousSubmissionIndeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CalibrationSubmissionState {
    NotSubmitted = 0,
    /// Executor entry occurred; no conclusive outcome has been reconciled.
    InFlightUnknown = 1,
    Submitted = 2,
    /// Host reconciliation returned normally. Individual requests can still
    /// have failed; this is not an all-row success or quality certificate.
    HostReconciled = 3,
}

#[derive(Debug)]
pub struct CalibrationWaveReport {
    pub(super) selected: Option<super::selected::SelectedCalibrationEvidence>,
    /// Mandatory work in the authority order checked at real submission.
    /// For NotSubmitted/Unknown this is selected intent, not executed evidence.
    pub ordered_work: ExpectedWaveWork,
    pub submission: CalibrationSubmissionState,
    pub error: Option<FerrumError>,
    /// Actual recorder/sample/host evidence. A completed host flow alone does
    /// not make this observation eligible for calibration or training.
    pub observation: CalibrationObservation,
    /// Independent host-settled evidence; never promotes a rejected legacy
    /// observation into a training/reference sample.
    pub host_stages: Option<Arc<super::super::cost_observation::HostStageEvidenceV1>>,
    pub host_stage_queue: Option<super::super::cost_observation::HostStageQueueReceipt>,
}

#[derive(Debug)]
pub enum CalibrationTurn {
    AdmittedOrMaintained,
    MaintenanceReconciled,
    Blocked(CalibrationBlockReason),
    Wave(CalibrationWaveReport),
    Reaped(CalibrationWaveReport),
}

/// Only the actual controller dispatch writes this shared one-wave receipt.
pub(in crate::continuous_engine::inner) struct CalibrationWaveReceipt {
    ordered_work: ExpectedWaveWork,
    state: std::sync::atomic::AtomicU8,
    capture: std::sync::OnceLock<Arc<super::super::cost_observation::CostCalibrationCapture>>,
}
impl CalibrationWaveReceipt {
    pub(in crate::continuous_engine::inner) fn new(ordered_work: ExpectedWaveWork) -> Self {
        Self {
            ordered_work,
            state: std::sync::atomic::AtomicU8::new(0),
            capture: std::sync::OnceLock::new(),
        }
    }
    pub(in crate::continuous_engine::inner) fn capture(
        &self,
    ) -> &Arc<super::super::cost_observation::CostCalibrationCapture> {
        self.capture.get_or_init(|| Arc::new(Default::default()))
    }
    pub(in crate::continuous_engine::inner) fn bind_structured_capture(
        &self,
        capture: Arc<super::super::cost_observation::CostCalibrationCapture>,
    ) -> Result<()> {
        if self.state() != CalibrationSubmissionState::NotSubmitted {
            return Err(FerrumError::invalid_request(
                "structured capture must bind before execution",
            ));
        }
        self.capture.set(capture).map_err(|_| {
            FerrumError::invalid_request(
                "calibration capture was already fixed before structured binding",
            )
        })
    }
    pub(in crate::continuous_engine::inner) fn record(&self, state: CalibrationSubmissionState) {
        self.state.store(state as u8, Ordering::Release);
    }
    fn state(&self) -> CalibrationSubmissionState {
        match self.state.load(Ordering::Acquire) {
            0 => CalibrationSubmissionState::NotSubmitted,
            1 => CalibrationSubmissionState::InFlightUnknown,
            2 => CalibrationSubmissionState::Submitted,
            3 => CalibrationSubmissionState::HostReconciled,
            _ => unreachable!("private calibration receipt state"),
        }
    }
    pub(super) fn report(&self, error: Option<FerrumError>) -> CalibrationWaveReport {
        CalibrationWaveReport {
            selected: None,
            ordered_work: self.ordered_work.clone(),
            submission: self.state(),
            error,
            observation: super::observation::project_capture(self.capture()),
            host_stages: self.capture().host_stages(),
            host_stage_queue: self.capture().host_stage_queue(),
        }
    }
}
