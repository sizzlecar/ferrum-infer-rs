//! A backend gate after native encoding and before irreversible submission.
use super::{
    DefinitelyNotSubmitted, DeviceCostGraphConfiguration, DeviceCostGraphStreamState,
    DeviceExecutionPath, DeviceReusableExecutionCapture, DeviceSubmissionAttribution,
};
use crate::execution_cost::GuardedNotSubmittedReason;

/// A per-call authorization policy, distinct from a backend capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardedSubmissionMode {
    ExactRoute,
    /// Preserve adaptive selection but allow no preparation before the exact
    /// final route guard. A capturable or missing-warm route must reject.
    ExactAdaptiveRoute,
    /// No latency/cost witness is asserted. The original owned wave may enter
    /// adaptive preparation only after its dedicated authorization callback.
    CompleteRequestsAdaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceGuardedAdaptiveCapability {
    Unsupported,
    CompleteRequestsOnDemand,
}

/// Borrowed intent BEFORE adaptive preparation. This is neither an actual
/// replay attribution nor an execution permit and must not populate a cost
/// model. No Clone/serialization or retained resource authority is provided.
pub struct DeviceAdaptiveSubmissionIntent<'a> {
    encoded_work: &'a DeviceSubmissionAttribution,
    graph_before: DeviceCostGraphStreamState,
    capture: Option<&'a DeviceReusableExecutionCapture>,
}
impl<'a> DeviceAdaptiveSubmissionIntent<'a> {
    pub fn new(
        encoded_work: &'a DeviceSubmissionAttribution,
        graph_before: DeviceCostGraphStreamState,
        capture: Option<&'a DeviceReusableExecutionCapture>,
    ) -> Option<Self> {
        if graph_before.configuration() != DeviceCostGraphConfiguration::OnDemand
            || !graph_before.is_ready()
            || !encoded_work.replayed_segments().is_empty()
            || encoded_work
                .commands()
                .iter()
                .any(|command| command.execution_path() != DeviceExecutionPath::Eager)
        {
            return None;
        }
        Some(Self {
            encoded_work,
            graph_before,
            capture,
        })
    }
    pub fn encoded_work(&self) -> &DeviceSubmissionAttribution {
        self.encoded_work
    }
    pub fn graph_before(&self) -> DeviceCostGraphStreamState {
        self.graph_before
    }
    pub fn capture(&self) -> Option<&DeviceReusableExecutionCapture> {
        self.capture
    }
}

/// Called after every potentially blocking preparation step. The implementation
/// must only perform bounded nonblocking checks; success is valid for this call.
pub trait DeviceSubmissionGuard: Send + Sync {
    /// True only for a dispatch committed to a cost witness. This does not
    /// require legacy commands to provide selected cost evidence. Backends use
    /// it only for commands that explicitly declare a library cost contract.
    fn relies_on_cost_witness(&self) -> bool {
        false
    }

    fn submission_mode(&self) -> GuardedSubmissionMode {
        GuardedSubmissionMode::ExactRoute
    }
    /// This commits one CompleteRequests attempt before potentially blocking
    /// capture/upload work. Once successful, the backend cannot subsequently
    /// reject as NotSubmitted; ordinary fence/indeterminate ownership applies.
    fn check_adaptive_preparation(
        &self,
        _intent: &DeviceAdaptiveSubmissionIntent<'_>,
    ) -> Result<(), GuardedNotSubmittedReason> {
        Err(GuardedNotSubmittedReason::AttributionUnavailable)
    }
    fn check(
        &self,
        attribution: Option<&DeviceSubmissionAttribution>,
    ) -> Result<(), GuardedNotSubmittedReason>;
}

#[derive(Debug)]
pub enum GuardedDeviceSubmissionError<E> {
    Device(DefinitelyNotSubmitted<E>),
    Rejected(GuardedNotSubmittedReason),
}

impl<E> From<DefinitelyNotSubmitted<E>> for GuardedDeviceSubmissionError<E> {
    fn from(error: DefinitelyNotSubmitted<E>) -> Self {
        Self::Device(error)
    }
}
