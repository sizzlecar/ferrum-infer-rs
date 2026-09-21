use super::{CopyRegion, DeviceBufferRetention, HostTransferLayout, VNextError};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct StagingState {
    limit: u64,
    used: u64,
}

/// A lane-local payload byte budget. Zero disables staged readbacks.
/// Backend allocation alignment and bookkeeping are not payload bytes; every
/// nonempty allocation still requires its own bounded payload reservation.
#[derive(Clone, Default)]
pub(crate) struct DeviceReadbackStagingBudget(Arc<Mutex<StagingState>>);

impl DeviceReadbackStagingBudget {
    pub(crate) fn configure(&self, limit: u64) -> Result<(), VNextError> {
        let mut state = self
            .0
            .lock()
            .map_err(|_| invalid("staging budget is poisoned"))?;
        if state.used != 0 {
            return Err(invalid(
                "cannot resize a staging budget with outstanding leases",
            ));
        }
        state.limit = limit;
        Ok(())
    }

    pub(crate) fn reserve(&self, bytes: u64) -> Option<DeviceReadbackStagingLease> {
        let mut state = self.0.lock().ok()?;
        let next = state.used.checked_add(bytes)?;
        if bytes == 0 || next > state.limit {
            return None;
        }
        state.used = next;
        Some(DeviceReadbackStagingLease(Arc::new(StagingClaim {
            budget: self.clone(),
            bytes,
        })))
    }
}

fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.to_owned(),
    }
}

struct StagingClaim {
    budget: DeviceReadbackStagingBudget,
    bytes: u64,
}

impl Drop for StagingClaim {
    fn drop(&mut self) {
        let mut state = self
            .budget
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.used -= self.bytes;
    }
}

/// Opaque host staging payload capacity shared by a snapshot and its DMA command.
/// The reservation is released only after every owner has dropped it.
#[derive(Clone)]
pub struct DeviceReadbackStagingLease(Arc<StagingClaim>);

impl DeviceReadbackStagingLease {
    pub fn bytes(&self) -> u64 {
        self.0.bytes
    }
}

/// One core-validated typed physical range and its source/staging ownership.
pub struct DeviceSubmissionReadbackRequest<'a, B> {
    pub(crate) source: &'a B,
    pub(crate) region: CopyRegion,
    pub(crate) layout: HostTransferLayout,
    pub(crate) retention: DeviceBufferRetention,
    pub(crate) staging: DeviceReadbackStagingLease,
}

impl<'a, B> DeviceSubmissionReadbackRequest<'a, B> {
    pub fn into_parts(
        self,
    ) -> (
        &'a B,
        CopyRegion,
        HostTransferLayout,
        DeviceBufferRetention,
        DeviceReadbackStagingLease,
    ) {
        (
            self.source,
            self.region,
            self.layout,
            self.retention,
            self.staging,
        )
    }
}

pub(crate) struct DeviceReadbackSnapshot<E> {
    reader: Box<dyn Fn() -> Result<Vec<u8>, E> + Send + Sync>,
}

impl<E> DeviceReadbackSnapshot<E> {
    // Accessible only inside core's successful terminal action; neither a
    // backend consumer nor a product caller can read a pending snapshot.
    pub(crate) fn read_at_terminal(&self) -> Result<Vec<u8>, E> {
        (self.reader)()
    }
}

/// Backend implementation of an ordered copy and its terminal-only reader.
pub struct PreparedDeviceSubmissionReadback<C, E> {
    command: Option<C>,
    snapshot: DeviceReadbackSnapshot<E>,
}

impl<C, E> PreparedDeviceSubmissionReadback<C, E> {
    pub fn new(
        command: Option<C>,
        reader: impl Fn() -> Result<Vec<u8>, E> + Send + Sync + 'static,
    ) -> Self {
        Self {
            command,
            snapshot: DeviceReadbackSnapshot {
                reader: Box::new(reader),
            },
        }
    }

    pub(crate) fn into_parts(self) -> (Option<C>, DeviceReadbackSnapshot<E>) {
        (self.command, self.snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceReadbackStagingBudget;

    #[test]
    fn staging_remains_reserved_until_command_and_snapshot_release_it() {
        let budget = DeviceReadbackStagingBudget::default();
        assert!(budget.reserve(4).is_none());
        budget.configure(8).unwrap();
        let snapshot = budget.reserve(4).unwrap();
        let command = snapshot.clone();
        let sibling = budget.reserve(4).unwrap();
        assert!(budget.reserve(1).is_none());
        assert!(budget.configure(16).is_err());
        drop(snapshot);
        assert!(budget.reserve(1).is_none());
        drop(command);
        let replacement = budget.reserve(4).unwrap();
        drop((sibling, replacement));
        budget.configure(4).unwrap();
        assert!(budget.reserve(5).is_none());
        assert!(budget.reserve(u64::MAX).is_none());
        assert!(budget.reserve(0).is_none());
        assert!(budget.reserve(4).is_some());
    }
}
