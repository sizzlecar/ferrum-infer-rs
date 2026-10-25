use super::*;

#[test]
fn aligned_backing_charges_padding_before_allocation_and_retains_sliced_owner() {
    let alignment: usize = 4096;
    for length in [1, alignment, alignment + 1] {
        let required = length.div_ceil(alignment) * alignment + alignment - 1;
        let aligned =
            HostMemoryAccount::with_allocation_alignment(account(required).limits(), alignment)
                .unwrap();
        let short =
            HostMemoryAccount::with_allocation_alignment(account(required - 1).limits(), alignment)
                .unwrap();
        assert!(matches!(
            short.allocate_zeroed(HostMemoryClass::InputPreprocess, length),
            Err(HostMemoryError::Capacity { .. })
        ));
        assert_eq!(short.usage().unwrap().committed_bytes(), 0);
        let reservation = aligned
            .reserve(HostMemoryClass::InputPreprocess, length)
            .unwrap();
        assert_eq!(reservation.bytes(), length);
        assert_eq!(reservation.backing_bytes(), required);
        assert_eq!(aligned.usage().unwrap().reserved_bytes(), required);
        let mut buffer = reservation.allocate_zeroed().unwrap();
        assert_eq!(buffer.len(), length);
        assert_eq!(buffer.as_ref().as_ptr() as usize % alignment, 0);
        assert!(buffer.as_ref().iter().all(|&byte| byte == 0));
        buffer.as_mut().fill(17);
        let bytes = buffer.freeze();
        let slice = bytes.slice(length - 1..length).unwrap();
        let region = slice.retained_region().unwrap();
        assert_eq!(region.bytes(), &[17]);
        assert_eq!(region.owner_bytes().len(), required);
        let owner_start = region.owner_bytes().as_ptr() as usize;
        let page_start = region.bytes().as_ptr() as usize & !(alignment - 1);
        assert!(page_start >= owner_start);
        assert!(page_start + alignment <= owner_start + required);
        assert!(region.owner_bytes()[..bytes.range.start]
            .iter()
            .all(|&byte| byte == 0));
        assert!(region.owner_bytes()[bytes.range.end..]
            .iter()
            .all(|&byte| byte == 0));
        drop((bytes, slice));
        assert_eq!(aligned.usage().unwrap().allocated_bytes(), required);
        assert_eq!(region.bytes(), &[17]);
        drop(region);
        assert_eq!(aligned.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn aligned_copy_and_multi_reservation_preserve_logical_extents_and_atomicity() {
    let alignment = 16;
    let single_backing = 2 * alignment - 1;
    let aligned = HostMemoryAccount::with_allocation_alignment(
        account(2 * single_backing).limits(),
        alignment,
    )
    .unwrap();
    let mut reservations = aligned
        .reserve_many(&[
            (HostMemoryClass::InputPreprocess, 3),
            (HostMemoryClass::Output, 5),
        ])
        .unwrap();
    let output = reservations
        .pop()
        .unwrap()
        .allocate_copy(&[1, 2, 3, 4, 5])
        .unwrap()
        .freeze();
    assert_eq!(output.bytes(), &[1, 2, 3, 4, 5]);
    assert_eq!(output.bytes().as_ptr() as usize % alignment, 0);
    assert!(aligned.owns(&output));
    assert!(aligned.reserve(HostMemoryClass::Output, 1).is_err());
    assert_eq!(
        aligned.usage().unwrap().committed_bytes(),
        2 * single_backing
    );
    drop(reservations);
    assert_eq!(aligned.usage().unwrap().committed_bytes(), single_backing);
    assert!(aligned
        .reserve_many(&[(HostMemoryClass::Output, 1), (HostMemoryClass::Output, 1)])
        .is_err());
    assert_eq!(aligned.usage().unwrap().committed_bytes(), single_backing);
    drop(output);
    assert_eq!(aligned.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn aligned_allocation_rejects_invalid_layouts_without_changing_capacity() {
    let limits = account(usize::MAX).limits();
    for alignment in [0, 3, usize::MAX] {
        assert!(HostMemoryAccount::with_allocation_alignment(limits, alignment).is_err());
    }
    let aligned = HostMemoryAccount::with_allocation_alignment(limits, 4096).unwrap();
    for length in [0, isize::MAX as usize, usize::MAX] {
        assert!(aligned.reserve(HostMemoryClass::Output, length).is_err());
        assert_eq!(aligned.usage().unwrap().committed_bytes(), 0);
    }
}

fn account(total: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            total,
            HostMemoryClassLimits {
                input_preprocess_bytes: total,
                transfer_staging_bytes: total,
                output_bytes: total,
                weight_backing_bytes: total,
            },
        )
        .unwrap(),
    )
}

#[test]
fn reservations_precede_allocation_and_rollback_when_dropped() {
    let account = account(64);
    let reservation = account.reserve(HostMemoryClass::Output, 32).unwrap();
    let usage = account.usage().unwrap();
    assert_eq!(usage.committed_bytes(), 32);
    assert_eq!(usage.reserved_bytes(), 32);
    assert_eq!(usage.allocated_bytes(), 0);
    let buffer = reservation.allocate_zeroed().unwrap();
    assert_eq!(buffer.as_ref(), &[0; 32]);
    let usage = account.usage().unwrap();
    assert_eq!(usage.committed_bytes(), 32);
    assert_eq!(usage.reserved_bytes(), 0);
    assert_eq!(usage.allocated_bytes(), 32);
    drop(buffer);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    drop(
        account
            .reserve(HostMemoryClass::InputPreprocess, 64)
            .unwrap(),
    );
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn admitted_copy_is_independent_and_aliases_keep_the_full_initialized_charge() {
    let account = account(8);
    let reservation = account
        .reserve(HostMemoryClass::TransferStaging, 8)
        .unwrap();
    assert_eq!(account.usage().unwrap().committed_bytes(), 8);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    let expected = [0xff, 0, 0x80, 0x7f, 0x35, 0xa1, 0, 0xff];
    let mut source = expected;
    let buffer = reservation.allocate_copy(&source).unwrap();
    source.fill(0x11);
    assert_eq!(buffer.as_ref(), &expected);
    assert_ne!(buffer.as_ref(), &source);
    assert_eq!(buffer.class(), HostMemoryClass::TransferStaging);
    let usage = account.usage().unwrap();
    assert_eq!(usage.committed_bytes(), 8);
    assert_eq!(usage.allocated_bytes(), 8);
    assert_eq!(usage.reserved_bytes(), 0);
    let bytes = buffer.freeze();
    assert!(account.owns(&bytes));
    let clone = bytes.clone();
    let slice = clone.slice(2..6).unwrap();
    let empty = slice.slice(4..4).unwrap();
    drop((bytes, clone));
    assert_eq!(slice.bytes(), &expected[2..6]);
    assert_eq!(
        account
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::TransferStaging),
        8
    );
    assert!(account
        .reserve(HostMemoryClass::TransferStaging, 1)
        .is_err());
    drop(slice);
    assert!(empty.is_empty());
    assert_eq!(account.usage().unwrap().allocated_bytes(), 8);
    drop(empty);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
}

#[test]
fn admitted_copy_rejects_empty_or_mismatched_source_and_rolls_back_reservation() {
    let account = account(4);
    let source = [1, 2, 3, 4, 5];
    for length in [0, 3, 5] {
        let reservation = account
            .reserve(HostMemoryClass::TransferStaging, 4)
            .unwrap();
        assert!(matches!(
            reservation.allocate_copy(&source[..length]),
            Err(HostMemoryError::InvalidReservation)
        ));
        let usage = account.usage().unwrap();
        assert_eq!(usage.committed_bytes(), 0);
        assert_eq!(usage.allocated_bytes(), 0);
        assert_eq!(usage.reserved_bytes(), 0);
    }
    assert!(matches!(
        account.reserve(HostMemoryClass::TransferStaging, 0),
        Err(HostMemoryError::InvalidReservation)
    ));
}

#[test]
fn batch_admission_is_atomic_across_parent_and_repeated_classes() {
    let account = HostMemoryAccount::new(
        HostMemoryLimits::new(
            64,
            HostMemoryClassLimits {
                input_preprocess_bytes: 64,
                transfer_staging_bytes: 32,
                output_bytes: 16,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let baseline = account
        .reserve(HostMemoryClass::InputPreprocess, 8)
        .unwrap();
    let before = account.usage().unwrap();
    assert!(matches!(
        account.reserve_many(&[
            (HostMemoryClass::TransferStaging, 8),
            (HostMemoryClass::Output, 8),
            (HostMemoryClass::Output, 9)
        ]),
        Err(HostMemoryError::Capacity {
            scope: HostMemoryPressureScope::Class(HostMemoryClass::Output),
            ..
        })
    ));
    assert_eq!(account.usage().unwrap(), before);
    assert!(matches!(
        account.reserve_many(&[
            (HostMemoryClass::InputPreprocess, 40),
            (HostMemoryClass::TransferStaging, 32)
        ]),
        Err(HostMemoryError::Capacity {
            scope: HostMemoryPressureScope::Total,
            ..
        })
    ));
    assert_eq!(account.usage().unwrap(), before);
    assert!(matches!(
        account.reserve(HostMemoryClass::WeightBacking, 1),
        Err(HostMemoryError::Capacity {
            scope: HostMemoryPressureScope::Class(HostMemoryClass::WeightBacking),
            ..
        })
    ));
    let claims = account
        .reserve_many(&[
            (HostMemoryClass::Output, 8),
            (HostMemoryClass::Output, 8),
            (HostMemoryClass::TransferStaging, 32),
        ])
        .unwrap();
    assert_eq!(account.usage().unwrap().committed_bytes(), 56);
    assert_eq!(
        account
            .usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output),
        16
    );
    drop(claims);
    assert_eq!(account.usage().unwrap(), before);
    drop(baseline);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn separate_engine_handles_compete_for_one_parent_capacity() {
    let parent = account(64);
    let first_engine = parent.clone();
    let second_engine = parent.clone();
    let first = first_engine
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 40)
        .unwrap();
    let second = second_engine.reserve(HostMemoryClass::Output, 24).unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 64);
    assert!(matches!(
        second_engine.reserve(HostMemoryClass::Output, 1),
        Err(HostMemoryError::Capacity {
            scope: HostMemoryPressureScope::Total,
            ..
        })
    ));
    drop(first);
    let replacement = second_engine
        .allocate_zeroed(HostMemoryClass::WeightBacking, 40)
        .unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 64);
    drop((replacement, second));
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn frozen_clones_and_nested_slices_retain_one_full_backing_charge() {
    let account = account(8);
    let mut buffer = account.allocate_zeroed(HostMemoryClass::Output, 8).unwrap();
    buffer.as_mut().copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);
    let bytes = buffer.freeze();
    let clone = bytes.clone();
    let middle = bytes.slice(2..7).unwrap();
    let nested = middle.slice(1..3).unwrap();
    let empty = nested.slice(2..2).unwrap();
    assert_eq!(nested.bytes(), &[3, 4]);
    assert!(empty.is_empty());
    assert_eq!(nested.bytes().as_ptr(), &bytes.bytes()[3] as *const u8);
    assert!(matches!(
        middle.slice(0..6),
        Err(HostMemoryError::InvalidSlice)
    ));
    let inverted = Range { start: 3, end: 2 };
    assert!(matches!(
        middle.slice(inverted),
        Err(HostMemoryError::InvalidSlice)
    ));
    drop((bytes, clone, middle, nested));
    // Even an empty view retains its owner; a caller may still hold its pointer.
    assert_eq!(account.usage().unwrap().allocated_bytes(), 8);
    assert!(account.reserve(HostMemoryClass::Output, 1).is_err());
    drop(empty);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn stable_host_region_keeps_real_allocation_and_claim_alive() {
    let account = account(16);
    let mut buffer = account
        .allocate_zeroed(HostMemoryClass::WeightBacking, 16)
        .unwrap();
    buffer.as_mut()[4..8].copy_from_slice(&[1, 2, 3, 4]);
    let bytes = buffer.freeze();
    let owner = Arc::new(bytes.clone());
    let region = crate::RetainedHostMemoryRegion::new(owner.clone(), 4, 4).unwrap();
    let region_clone = region.clone();
    drop((owner, bytes, region));
    assert_eq!(region_clone.bytes(), &[1, 2, 3, 4]);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 16);
    drop(region_clone);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn backing_provenance_uses_the_parent_identity_not_equal_limits() {
    let parent = account(16);
    let unrelated = account(16);
    let bytes = parent
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
        .unwrap()
        .freeze();
    assert!(parent.clone().owns(&bytes));
    assert!(parent.owns(&bytes.slice(2..4).unwrap()));
    assert!(!unrelated.owns(&bytes));
}

#[test]
fn region_identity_requires_the_same_backing_and_exact_normalized_range() {
    let parent = account(16);
    let bytes = parent
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
        .unwrap()
        .freeze();
    let independent = parent
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 8)
        .unwrap()
        .freeze();
    assert_eq!(bytes.bytes(), independent.bytes());
    assert!(!bytes.same_region(&independent));
    assert!(bytes.same_region(&bytes.clone()));
    assert!(bytes.same_region(&bytes.slice(0..8).unwrap()));

    let middle = bytes.slice(2..7).unwrap();
    let nested = middle.slice(1..3).unwrap();
    assert!(nested.same_region(&bytes.slice(3..5).unwrap()));
    assert!(!nested.same_region(&bytes.slice(2..4).unwrap()));
    assert!(!nested.same_region(&bytes.slice(3..6).unwrap()));
    assert!(!nested.same_region(&independent.slice(3..5).unwrap()));
    // Empty views still identify their original allocation and exact offset.
    let empty = nested.slice(2..2).unwrap();
    assert!(empty.same_region(&bytes.slice(5..5).unwrap()));
    assert!(!empty.same_region(&bytes.slice(4..4).unwrap()));
    assert!(!empty.same_region(&independent.slice(5..5).unwrap()));
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 16);
    drop((bytes, independent, middle, nested));
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 8);
    drop(empty);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn real_copies_require_overlapping_capacity_and_release_independently() {
    let account = account(16);
    let mut source = account
        .allocate_zeroed(HostMemoryClass::TransferStaging, 8)
        .unwrap();
    source.as_mut().copy_from_slice(b"transfer");
    let source = source.freeze();
    let mut destination = account
        .reserve(HostMemoryClass::Output, 8)
        .unwrap()
        .allocate_zeroed()
        .unwrap();
    assert_eq!(account.usage().unwrap().allocated_bytes(), 16);
    assert!(account.allocate_zeroed(HostMemoryClass::Output, 1).is_err());
    destination.as_mut().copy_from_slice(source.bytes());
    let destination = destination.freeze();
    assert_ne!(source.bytes().as_ptr(), destination.bytes().as_ptr());
    drop(source);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 8);
    assert_eq!(destination.bytes(), b"transfer");
    drop(destination);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn invalid_limits_empty_extents_and_arithmetic_overflow_never_charge() {
    let classes = HostMemoryClassLimits {
        input_preprocess_bytes: 1,
        transfer_staging_bytes: 0,
        output_bytes: 0,
        weight_backing_bytes: 0,
    };
    assert_eq!(
        HostMemoryLimits::new(0, classes),
        Err(HostMemoryError::InvalidLimits)
    );
    assert_eq!(
        HostMemoryLimits::new(
            1,
            HostMemoryClassLimits {
                output_bytes: 2,
                ..classes
            }
        ),
        Err(HostMemoryError::InvalidLimits)
    );
    let account = account(usize::MAX);
    assert!(matches!(
        account.reserve_many(&[]),
        Err(HostMemoryError::InvalidReservation)
    ));
    for bytes in [0, isize::MAX as usize + 1] {
        assert!(matches!(
            account.reserve(HostMemoryClass::Output, bytes),
            Err(HostMemoryError::InvalidReservation)
        ));
    }
    assert!(matches!(
        account.reserve_many(&[
            (HostMemoryClass::Output, isize::MAX as usize),
            (HostMemoryClass::InputPreprocess, isize::MAX as usize),
            (HostMemoryClass::TransferStaging, 2),
        ]),
        Err(HostMemoryError::ArithmeticOverflow)
    ));
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn concurrent_admission_never_exceeds_the_shared_parent() {
    let account = account(64);
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let successes = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let account = account.clone();
                let barrier = barrier.clone();
                scope.spawn(move || {
                    let reservation = account.reserve(HostMemoryClass::Output, 16);
                    barrier.wait(); // Every successful claim remains live until all have tried.
                    assert!(account.usage().unwrap().committed_bytes() <= 64);
                    reservation.is_ok()
                })
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .filter(|success| *success)
            .count()
    });
    assert_eq!(successes, 4);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn poisoned_account_fails_closed_including_after_existing_owner_drops() {
    let account = account(8);
    let bytes = account
        .allocate_zeroed(HostMemoryClass::Output, 8)
        .unwrap()
        .freeze();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = account.0.state.lock().unwrap();
        panic!("simulate an interrupted accounting mutation");
    }));
    assert!(result.is_err());
    assert_eq!(account.usage(), Err(HostMemoryError::Poisoned));
    drop(bytes);
    assert!(matches!(
        account.reserve(HostMemoryClass::Output, 1),
        Err(HostMemoryError::Poisoned)
    ));
}
