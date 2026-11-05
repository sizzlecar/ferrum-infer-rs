use super::*;
use ferrum_runtime::{ManagedHostBuffer, ManagedHostBytes};
use std::{num::NonZeroUsize, sync::Mutex, sync::MutexGuard};

/// One explicitly owned installation with one continuously charged allocation.
/// Payloads expose exact component slices while retaining the whole slot. Drop
/// releases this installation's owner; outstanding payload/native owners keep
/// the original allocation and Host charge alive independently.
///
/// A live payload or concurrent conversion makes another checkout fail. There
/// is no second-slot fallback, hidden wait, cache eviction or scope exchange.
pub struct ManagedDenseArchiveInstallation<'archive> {
    source: ManagedDenseArchiveSource<'archive>,
    backing: ReusableBacking,
}

impl<'archive> ManagedDenseArchiveInstallation<'archive> {
    pub(super) fn new(
        archive: &'archive dyn TensorArchive,
        host: &'archive HostMemoryAccount,
        maximum_component_bytes: NonZeroUsize,
    ) -> std::result::Result<Self, VNextError> {
        let buffer = host
            .allocate_zeroed(
                HostMemoryClass::WeightBacking,
                maximum_component_bytes.get(),
            )
            .map_err(|error| invalid(error.to_string()))?;
        let full = buffer.freeze();
        Ok(Self {
            source: ManagedDenseArchiveSource::new(archive, host),
            backing: ReusableBacking {
                logical_bytes: maximum_component_bytes.get(),
                physical_bytes: full.backing_bytes(),
                state: Mutex::new(SlotState::Ready(full)),
            },
        })
    }

    /// Full physical charge, including padding, even while the slot is idle or
    /// the last published logical component is smaller than this allocation.
    pub fn backing_bytes(&self) -> usize {
        self.backing.physical_bytes
    }

    /// Same arithmetic and counters as the fresh source. The existing
    /// reserve_allocate_zeroed phase records slot checkout here; initial
    /// allocation belongs to the installation constructor and must be timed by
    /// its caller when comparing total preparation cost.
    pub fn component_with_diagnostics<'source>(
        &'source self,
        component: &WeightComponentSpec,
        source_touch: DenseArchiveSourceTouch,
        report: &mut DenseArchiveMaterializationReport,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        self.source
            .component_diagnostic(component, source_touch, report, Some(&self.backing))
    }
}

impl WeightComponentSource for ManagedDenseArchiveInstallation<'_> {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        self.source.component_inner::<false>(
            component,
            DenseArchiveSourceTouch::AsMapped,
            &mut DenseArchiveMaterializationReport::default(),
            Some(&self.backing),
        )
    }
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: format!("managed archive backing: {}", reason.into()),
    }
}

enum SlotState {
    Ready(ManagedHostBytes),
    CheckedOut,
}

pub(super) struct ReusableBacking {
    logical_bytes: usize,
    physical_bytes: usize,
    state: Mutex<SlotState>,
}

impl ReusableBacking {
    fn checkout(&self, bytes: usize) -> std::result::Result<ReuseLease<'_>, VNextError> {
        if bytes == 0 || bytes > self.logical_bytes {
            return Err(invalid(
                "component exceeds the declared nonempty slot extent",
            ));
        }
        let mut state = self
            .state
            .try_lock()
            .map_err(|_| invalid("slot is busy or poisoned"))?;
        match std::mem::replace(&mut *state, SlotState::CheckedOut) {
            SlotState::Ready(full) => match full.try_thaw() {
                Ok(buffer) => Ok(ReuseLease {
                    state,
                    buffer: Some(buffer),
                    bytes,
                }),
                Err(full) => {
                    *state = SlotState::Ready(full);
                    Err(invalid(
                        "slot is busy: an immutable payload or native region is retained",
                    ))
                }
            },
            SlotState::CheckedOut => Err(invalid("slot is already checked out")),
        }
    }
}

pub(super) struct ReuseLease<'slot> {
    // Holding the guard excludes concurrent checkout through conversion and
    // publication; CheckedOut is never interpreted as an unallocated slot.
    state: MutexGuard<'slot, SlotState>,
    buffer: Option<ManagedHostBuffer>,
    bytes: usize,
}

impl ReuseLease<'_> {
    fn freeze(mut self) -> std::result::Result<ManagedHostBytes, VNextError> {
        let full = self.buffer.take().expect("lease owns its buffer").freeze();
        let payload = full.slice(0..self.bytes);
        *self.state = SlotState::Ready(full);
        payload.map_err(|error| invalid(error.to_string()))
    }
}

impl Drop for ReuseLease<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // A failed conversion publishes nothing. Preserve its initialized
            // allocation and charge; the next conversion overwrites its entire
            // logical payload before publication. Panics poison the mutex.
            *self.state = SlotState::Ready(buffer.freeze());
        }
    }
}

pub(super) enum BackingLease<'slot> {
    Fresh(ManagedHostBuffer),
    Reuse(ReuseLease<'slot>),
}

impl<'slot> BackingLease<'slot> {
    pub(super) fn allocate(
        host: &HostMemoryAccount,
        bytes: usize,
        reuse: Option<&'slot ReusableBacking>,
    ) -> std::result::Result<Self, String> {
        match reuse {
            Some(slot) => slot
                .checkout(bytes)
                .map(Self::Reuse)
                .map_err(|error| error.to_string()),
            None => host
                .allocate_zeroed(HostMemoryClass::WeightBacking, bytes)
                .map(Self::Fresh)
                .map_err(|error| error.to_string()),
        }
    }

    pub(super) fn freeze(self) -> std::result::Result<ManagedHostBytes, VNextError> {
        match self {
            Self::Fresh(buffer) => Ok(buffer.freeze()),
            Self::Reuse(lease) => lease.freeze(),
        }
    }
}

impl AsMut<[u8]> for BackingLease<'_> {
    fn as_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Fresh(buffer) => buffer.as_mut(),
            Self::Reuse(lease) => &mut lease
                .buffer
                .as_mut()
                .expect("lease owns its buffer")
                .as_mut()[..lease.bytes],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

    #[test]
    fn concurrent_checkout_fails_without_allocating_a_second_slot() {
        let host = HostMemoryAccount::new(
            HostMemoryLimits::new(
                8,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 0,
                    transfer_staging_bytes: 0,
                    output_bytes: 0,
                    weight_backing_bytes: 8,
                },
            )
            .unwrap(),
        );
        let full = host
            .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
            .unwrap()
            .freeze();
        let slot = ReusableBacking {
            logical_bytes: 8,
            physical_bytes: 8,
            state: Mutex::new(SlotState::Ready(full)),
        };
        let usage = host.usage().unwrap();
        let loan = slot.checkout(8).unwrap();
        std::thread::scope(|threads| {
            threads
                .spawn(|| assert!(slot.checkout(1).is_err()))
                .join()
                .unwrap();
        });
        assert_eq!(host.usage().unwrap(), usage);
        drop(loan);
        let payload = slot.checkout(3).unwrap().freeze().unwrap();
        assert_eq!(payload.len(), 3);
        assert_eq!(payload.backing_bytes(), 8);
        assert_eq!(host.usage().unwrap(), usage);
        drop(slot);
        assert_eq!(host.usage().unwrap(), usage);
        drop(payload);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
