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
/// Every call needs exclusive access; any error or unwind permanently stops
/// this actor. Error/completion owners retain their ordinary recovery authority.
pub struct TensorSerialActor<R: DeviceRuntime> {
    lane: Arc<ExecutionLane<R>>,
    failed: bool,
}

impl<R: DeviceRuntime> TensorSerialActor<R> {
    pub fn new(runtime: Arc<R>) -> Result<Self, ExecutionLaneCreationError<R::Error>> {
        Ok(Self {
            lane: ExecutionLane::create(runtime)?,
            failed: false,
        })
    }

    pub const fn is_failed(&self) -> bool {
        self.failed
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
        if result.is_ok() {
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
