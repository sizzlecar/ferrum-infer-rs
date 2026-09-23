//! A backend gate after native encoding and before irreversible submission.
use super::{DefinitelyNotSubmitted, DeviceSubmissionAttribution};
use crate::execution_cost::GuardedNotSubmittedReason;

/// Called after every potentially blocking preparation step. The implementation
/// must only perform bounded nonblocking checks; success is valid for this call.
pub trait DeviceSubmissionGuard: Send + Sync {
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
