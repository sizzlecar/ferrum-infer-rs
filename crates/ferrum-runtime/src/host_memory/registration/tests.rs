use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

struct RegistrationProbe {
    released: Arc<AtomicUsize>,
    dropped: Arc<AtomicUsize>,
    succeeds: bool,
}

// SAFETY: this probe creates no native accesses. A false retirement result is
// used only to verify that the original payload and accounting stay retained.
unsafe impl NativeHostMemoryRegistration for RegistrationProbe {
    fn release(&mut self) -> bool {
        self.released.fetch_add(1, Ordering::SeqCst);
        self.succeeds
    }
}

impl Drop for RegistrationProbe {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

fn account(bytes: usize, alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}

#[test]
fn registration_preserves_physical_owner_and_survives_freeze_thaw() {
    let host = account(256, 16);
    let bytes = host
        .allocate_zeroed(HostMemoryClass::WeightBacking, 18)
        .unwrap()
        .freeze();
    let released = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let slice = bytes.slice(4..6).unwrap();
    let physical = bytes.backing_bytes();
    slice
        .try_register_native(7, |owner, original| {
            assert_eq!(owner.len(), physical);
            assert_eq!(original.len(), 18);
            assert_eq!((owner.as_ptr() as usize + original.start) % 16, 0);
            Ok::<_, ()>(RegistrationProbe {
                released: released.clone(),
                dropped: dropped.clone(),
                succeeds: true,
            })
        })
        .unwrap();
    drop(slice);
    let mut writable = bytes.try_thaw().unwrap();
    writable.as_mut().fill(42);
    let bytes = writable.freeze();
    bytes
        .try_register_native::<RegistrationProbe, ()>(7, |_, _| {
            panic!("the same allocation must keep its original registration")
        })
        .unwrap();
    assert_eq!(released.load(Ordering::SeqCst), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
    let retained = bytes.retained_region().unwrap();
    drop(bytes);
    assert_eq!(released.load(Ordering::SeqCst), 0);
    assert_eq!(retained.bytes(), &[42; 18]);
    drop(retained);
    assert_eq!(released.load(Ordering::SeqCst), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn failed_registration_leaves_owner_reusable_and_does_not_grant_capacity() {
    let host = account(32, 1);
    let bytes = host
        .allocate_zeroed(HostMemoryClass::WeightBacking, 32)
        .unwrap()
        .freeze();
    assert_eq!(
        bytes.try_register_native::<RegistrationProbe, _>(1, |_, _| Err("register failed")),
        Err("register failed")
    );
    assert!(host.reserve(HostMemoryClass::WeightBacking, 1).is_err());
    let writable = bytes.try_thaw().unwrap();
    drop(writable);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn failed_native_retirement_retains_payload_registration_and_quota() {
    struct AdmissionWaiter {
        host: HostMemoryAccount,
        wakes: AtomicUsize,
    }
    impl std::task::Wake for AdmissionWaiter {
        fn wake(self: Arc<Self>) {
            assert_eq!(self.host.usage(), Err(HostMemoryError::Poisoned));
            self.wakes.fetch_add(1, Ordering::SeqCst);
        }
    }
    let host = account(16, 1);
    let waiter = Arc::new(AdmissionWaiter {
        host: host.clone(),
        wakes: AtomicUsize::new(0),
    });
    let _subscription = host.subscribe_capacity(Waker::from(waiter.clone()));
    let released = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let bytes = host
        .allocate_zeroed(HostMemoryClass::WeightBacking, 16)
        .unwrap()
        .freeze();
    let pointer = bytes.bytes().as_ptr();
    bytes
        .try_register_native(1, |_, _| {
            Ok::<_, ()>(RegistrationProbe {
                released: released.clone(),
                dropped: dropped.clone(),
                succeeds: false,
            })
        })
        .unwrap();
    drop(bytes);
    assert_eq!(waiter.wakes.load(Ordering::SeqCst), 1);
    assert_eq!(released.load(Ordering::SeqCst), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 0);
    assert_eq!(host.usage(), Err(HostMemoryError::Poisoned));
    assert!(matches!(
        host.reserve(HostMemoryClass::WeightBacking, 1),
        Err(HostMemoryError::Poisoned)
    ));
    assert_eq!(host.0.state.lock().unwrap().allocated.total, 16);
    // SAFETY: retirement failure deliberately retains the exact allocation.
    assert_eq!(unsafe { std::slice::from_raw_parts(pointer, 16) }, &[0; 16]);
}

#[test]
fn concurrent_slices_register_the_same_owner_once() {
    let host = account(32, 1);
    let bytes = host
        .allocate_zeroed(HostMemoryClass::WeightBacking, 32)
        .unwrap()
        .freeze();
    let registrations = AtomicUsize::new(0);
    let released = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                bytes
                    .try_register_native(1, |_, _| {
                        registrations.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, ()>(RegistrationProbe {
                            released: released.clone(),
                            dropped: dropped.clone(),
                            succeeds: true,
                        })
                    })
                    .unwrap();
            });
        }
    });
    assert_eq!(registrations.load(Ordering::SeqCst), 1);
    drop(bytes);
    assert_eq!(released.load(Ordering::SeqCst), 1);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}
