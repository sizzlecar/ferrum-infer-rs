//! One scheduler-owned lane for successive blocking tensor executions.
use super::*;

/// Peak execution-stream ownership for a live serial actor. Initialization
/// streams may overlap the retained lane even when no computation overlaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct TensorSerialActorStreamRequirements {
    retained_streams: usize,
    execution_additional_streams: usize,
}

impl TensorSerialActorStreamRequirements {
    pub(super) const fn new() -> Self {
        Self {
            retained_streams: 1,
            execution_additional_streams: 0,
        }
    }

    pub const fn retained_streams(self) -> usize {
        self.retained_streams
    }

    pub const fn execution_additional_streams(self) -> usize {
        self.execution_additional_streams
    }
}

/// A serial scheduler retains this owner across components on one runtime.
/// The original stream and its native capacity grants remain alive together.
/// Every call needs exclusive access. Errors and unwinds stop this actor except
/// a proved pre-submission cancellation whose transient owners have retired.
pub struct TensorSerialActor<R: DeviceRuntime> {
    lane: Arc<ExecutionLane<R>>,
    failed: bool,
    admission_control: Option<TensorAdmissionControl>,
}

impl<R: DeviceRuntime> TensorSerialActor<R> {
    pub fn new(runtime: Arc<R>) -> Result<Self, ExecutionLaneCreationError<R::Error>> {
        Ok(Self {
            lane: ExecutionLane::create(runtime)?,
            failed: false,
            admission_control: None,
        })
    }

    pub const fn is_failed(&self) -> bool {
        self.failed
    }

    /// Opt in to capacity waiting. Plain actors retain fail-fast admission.
    pub fn set_admission_control(&mut self, control: TensorAdmissionControl) {
        self.admission_control = Some(control);
    }

    pub fn admission_control(&self) -> Option<&TensorAdmissionControl> {
        self.admission_control.as_ref()
    }

    pub(super) fn execute<T>(
        &mut self,
        execute: impl FnOnce(&Arc<ExecutionLane<R>>) -> Result<T, TensorExecutionError<R>>,
    ) -> Result<T, TensorExecutionError<R>> {
        if self.failed {
            return Err(invalid("failed tensor serial actor cannot execute again").into());
        }
        // Set before calling any fallible work. Unwinding needs no cleanup
        // callback to preserve the permanent failure mark.
        self.failed = true;
        let result = execute(&self.lane);
        if !result
            .as_ref()
            .is_err_and(|error| !error.is_cancelled_before_submission())
        {
            self.failed = false;
        }
        result
    }
}

impl<R: DeviceRuntime> fmt::Debug for TensorSerialActor<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorSerialActor")
            .field("lane", &self.lane)
            .field("failed", &self.failed)
            .finish()
    }
}

#[cfg(test)]
mod tests;
