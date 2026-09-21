use super::{CopyRegion, DeviceBufferRetention, HostTransferLayout, VNextError};
use std::any::{type_name, Any};
use std::fmt;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct StagingState {
    limit: u64,
    used: u64,
    resident: u64,
    slots: Vec<StagingSlotEntry>,
}

struct StagingSlotEntry {
    active: bool,
    slot: Arc<StagingSlot>,
}

struct StagingSlot {
    bytes: u64,
    storage: Mutex<Option<CachedStagingStorage>>,
}

struct CachedStagingStorage {
    value: Arc<dyn Any + Send + Sync>,
    type_name: &'static str,
}

/// A lane-local payload byte budget. Zero disables staged readbacks.
/// Backend allocation alignment and bookkeeping are not payload bytes; every
/// nonempty allocation still requires its own bounded payload reservation.
/// Idle cached slots count against the same resident payload limit.
#[derive(Clone, Default)]
pub(crate) struct DeviceReadbackStagingBudget(Arc<Mutex<StagingState>>);

impl DeviceReadbackStagingBudget {
    pub(crate) fn configure(&self, limit: u64) -> Result<(), VNextError> {
        let evicted = {
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
            state.resident = 0;
            std::mem::take(&mut state.slots)
        };
        // Backend storage destructors may wait for their own device events.
        // Never run them while holding the budget lock.
        drop(evicted);
        Ok(())
    }

    pub(crate) fn reserve(&self, bytes: u64) -> Option<DeviceReadbackStagingLease> {
        // Declare this before the lock scope so even exceptional exits destroy
        // evicted backend allocations only after releasing the budget lock.
        let mut evicted = Vec::new();
        let slot = {
            let mut state = self.0.lock().ok()?;
            let next = state.used.checked_add(bytes)?;
            if bytes == 0 || next > state.limit {
                return None;
            }
            if let Some(entry) = state
                .slots
                .iter_mut()
                .find(|entry| !entry.active && entry.slot.bytes == bytes)
            {
                entry.active = true;
                let slot = Arc::clone(&entry.slot);
                state.used = next;
                slot
            } else {
                // Active capacity already fits. Any resident excess therefore
                // belongs to idle slots; active slots are never removed.
                while state.resident > state.limit - bytes {
                    let index = state.slots.iter().position(|entry| !entry.active)?;
                    let entry = state.slots.swap_remove(index);
                    state.resident -= entry.slot.bytes;
                    evicted.push(entry);
                }
                let slot = Arc::new(StagingSlot {
                    bytes,
                    storage: Mutex::new(None),
                });
                state.slots.push(StagingSlotEntry {
                    active: true,
                    slot: Arc::clone(&slot),
                });
                state.used = next;
                state.resident += bytes;
                slot
            }
        };
        let lease = DeviceReadbackStagingLease(Arc::new(StagingClaim {
            budget: self.clone(),
            slot,
        }));
        drop(evicted);
        Some(lease)
    }
}

fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.to_owned(),
    }
}

struct StagingClaim {
    budget: DeviceReadbackStagingBudget,
    slot: Arc<StagingSlot>,
}

impl Drop for StagingClaim {
    fn drop(&mut self) {
        let mut state = self
            .budget
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = state
            .slots
            .iter_mut()
            .find(|entry| Arc::ptr_eq(&entry.slot, &self.slot))
            .expect("an active staging slot remains resident until its last lease drops");
        debug_assert!(entry.active);
        entry.active = false;
        state.used -= self.slot.bytes;
    }
}

/// Opaque host staging payload capacity shared by a snapshot and its DMA command.
/// The reservation is released only after every owner has dropped it.
#[derive(Clone)]
pub struct DeviceReadbackStagingLease(Arc<StagingClaim>);

impl DeviceReadbackStagingLease {
    pub fn bytes(&self) -> u64 {
        self.0.slot.bytes
    }

    /// Initialize or borrow this slot's host-only storage. The returned handle
    /// retains the reservation, so it cannot outlive the slot's active lease.
    ///
    /// Storage must not retain a staging lease, source buffer, request, or Step
    /// resource. Those belong to each submission, not this reusable host cache.
    /// Failed initialization leaves the slot empty and may be retried.
    pub fn get_or_try_init_storage<T, E>(
        &self,
        initialize: impl FnOnce() -> Result<T, E>,
    ) -> Result<DeviceReadbackStagingStorage<T>, DeviceReadbackStagingStorageError<E>>
    where
        T: Any + Send + Sync,
    {
        let mut cached = self
            .0
            .slot
            .storage
            .lock()
            .map_err(|_| DeviceReadbackStagingStorageError::Poisoned)?;
        if cached.is_none() {
            let value = initialize().map_err(DeviceReadbackStagingStorageError::Initialization)?;
            *cached = Some(CachedStagingStorage {
                value: Arc::new(value),
                type_name: type_name::<T>(),
            });
        }
        let cached = cached.as_ref().expect("initialized staging storage");
        let storage = Arc::clone(&cached.value).downcast::<T>().map_err(|_| {
            DeviceReadbackStagingStorageError::TypeMismatch {
                expected: type_name::<T>(),
                actual: cached.type_name,
            }
        })?;
        Ok(DeviceReadbackStagingStorage {
            storage,
            _lease: self.clone(),
        })
    }
}

/// A typed borrow of cached host storage that keeps its slot exclusively leased.
/// There is deliberately no method exposing an independently owned `Arc<T>`.
pub struct DeviceReadbackStagingStorage<T> {
    storage: Arc<T>,
    _lease: DeviceReadbackStagingLease,
}

impl<T> Clone for DeviceReadbackStagingStorage<T> {
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            _lease: self._lease.clone(),
        }
    }
}

impl<T> Deref for DeviceReadbackStagingStorage<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.storage
    }
}

#[derive(Debug)]
pub enum DeviceReadbackStagingStorageError<E> {
    Initialization(E),
    TypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    Poisoned,
}

impl<E: fmt::Display> fmt::Display for DeviceReadbackStagingStorageError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initialization(error) => {
                write!(formatter, "staging storage initialization: {error}")
            }
            Self::TypeMismatch { expected, actual } => write!(
                formatter,
                "staging storage type differs: expected {expected}, cached {actual}"
            ),
            Self::Poisoned => formatter.write_str("staging storage is poisoned"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for DeviceReadbackStagingStorageError<E> {}

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
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    #[test]
    fn cached_host_storage_is_reused_only_after_all_lease_and_handle_owners_drop() {
        let budget = DeviceReadbackStagingBudget::default();
        budget.configure(4).unwrap();
        let lease = budget.reserve(4).unwrap();
        let storage = lease
            .get_or_try_init_storage(|| Ok::<_, ()>(Mutex::new([7_u8; 4])))
            .unwrap();
        let storage_clone = storage.clone();
        *storage.lock().unwrap() = [1, 2, 3, 4];
        drop((lease, storage));
        assert!(budget.reserve(4).is_none());
        assert!(budget.configure(8).is_err());
        drop(storage_clone);
        let replacement = budget.reserve(4).unwrap();
        let restored = replacement
            .get_or_try_init_storage::<Mutex<[u8; 4]>, ()>(|| {
                panic!("idle same-size slot must retain its initialized storage")
            })
            .unwrap();
        assert_eq!(*restored.lock().unwrap(), [1, 2, 3, 4]);
        let state = budget.0.lock().unwrap();
        assert_eq!((state.used, state.resident, state.limit), (4, 4, 4));
    }

    struct DropProbe {
        budget: std::sync::Weak<Mutex<StagingState>>,
        drops: Arc<AtomicUsize>,
        unlocked_drops: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            if let Some(budget) = self.budget.upgrade() {
                if budget.try_lock().is_ok() {
                    self.unlocked_drops.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
    }

    #[test]
    fn resident_limit_evicts_only_idle_slots_and_drops_storage_outside_the_budget_lock() {
        let budget = DeviceReadbackStagingBudget::default();
        budget.configure(12).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let unlocked = Arc::new(AtomicUsize::new(0));
        let initialize = || {
            Ok::<_, ()>(DropProbe {
                budget: Arc::downgrade(&budget.0),
                drops: Arc::clone(&drops),
                unlocked_drops: Arc::clone(&unlocked),
            })
        };
        let small = budget.reserve(4).unwrap();
        let small_storage = small.get_or_try_init_storage(initialize).unwrap();
        let large = budget.reserve(8).unwrap();
        let large_storage = large.get_or_try_init_storage(initialize).unwrap();
        drop((small, small_storage));
        assert!(budget.reserve(6).is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(budget.0.lock().unwrap().resident, 12);
        let reused = budget.reserve(4).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop((reused, large, large_storage));
        // Neither idle object can remain resident beside a new 12-byte slot.
        let whole = budget.reserve(12).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(unlocked.load(Ordering::SeqCst), 2);
        assert_eq!(budget.0.lock().unwrap().resident, 12);
        let whole_storage = whole.get_or_try_init_storage(initialize).unwrap();
        drop(whole);
        assert!(budget.configure(0).is_err());
        drop(whole_storage);
        budget.configure(0).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        assert_eq!(unlocked.load(Ordering::SeqCst), 3);
        assert_eq!(budget.0.lock().unwrap().resident, 0);
        assert!(budget.reserve(1).is_none());
    }

    #[test]
    fn storage_initialization_failure_is_retryable_and_type_mismatch_never_replaces_cache() {
        let budget = DeviceReadbackStagingBudget::default();
        budget.configure(4).unwrap();
        let lease = budget.reserve(4).unwrap();
        assert!(matches!(
            lease.get_or_try_init_storage::<u32, _>(|| Err("allocation failed")),
            Err(DeviceReadbackStagingStorageError::Initialization(
                "allocation failed"
            ))
        ));
        let word = lease
            .get_or_try_init_storage(|| Ok::<_, ()>(19_u32))
            .unwrap();
        assert_eq!(*word, 19);
        assert!(matches!(
            lease.get_or_try_init_storage::<u64, ()>(|| panic!("cached type must not be replaced")),
            Err(DeviceReadbackStagingStorageError::TypeMismatch { .. })
        ));
        drop((lease, word));
        let next = budget.reserve(4).unwrap();
        let preserved = next
            .get_or_try_init_storage::<u32, ()>(|| {
                panic!("failed type request must preserve cache")
            })
            .unwrap();
        assert_eq!(*preserved, 19);
    }
}
