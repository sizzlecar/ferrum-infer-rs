use super::*;

fn account(total: usize, alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
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
        alignment,
    )
    .unwrap()
}

#[test]
fn unique_thaw_preserves_aligned_allocation_contents_charge_and_scope_identity() {
    let alignment = 64;
    let physical_bytes = 2 * alignment - 1;
    let parent = account(physical_bytes, alignment);
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical_bytes)])
        .unwrap();
    let scoped = scope.account();
    let mut original = scoped
        .allocate_zeroed(HostMemoryClass::WeightBacking, 5)
        .unwrap();
    original.as_mut().copy_from_slice(&[7, 0, 255, 1, 42]);
    let address = original.as_ref().as_ptr();
    let frozen = original.freeze();
    let usage = parent.usage().unwrap();
    assert_eq!(frozen.backing_bytes(), physical_bytes);
    assert_eq!(usage.committed_bytes(), physical_bytes);
    assert_eq!(usage.allocated_bytes(), physical_bytes);
    assert_eq!(usage.reserved_bytes(), 0);
    assert!(scoped.owns_in_scope(&frozen));
    assert!(!parent.owns_in_scope(&frozen));

    let mut writable = frozen.try_thaw().unwrap();
    assert_eq!(writable.as_ref().as_ptr(), address);
    assert_eq!(writable.as_ref(), &[7, 0, 255, 1, 42]);
    assert_eq!(writable.class(), HostMemoryClass::WeightBacking);
    assert_eq!(parent.usage().unwrap(), usage);
    assert!(scoped.reserve(HostMemoryClass::WeightBacking, 1).is_err());
    writable.as_mut()[2] = 11;
    let frozen = writable.freeze();
    assert_eq!(frozen.bytes().as_ptr(), address);
    assert_eq!(frozen.bytes(), &[7, 0, 11, 1, 42]);
    assert_eq!(frozen.backing_bytes(), physical_bytes);
    assert!(scoped.owns_in_scope(&frozen));
    assert!(!parent.owns_in_scope(&frozen));
    assert_eq!(parent.usage().unwrap(), usage);
    drop(frozen);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    // Freed allocation credit returns to the original live scope first.
    assert_eq!(parent.usage().unwrap().committed_bytes(), physical_bytes);
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn clones_partial_and_empty_slices_prevent_thaw_until_last_alias_drops() {
    let host = account(8, 1);
    let mut original = host
        .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
        .unwrap();
    original.as_mut().fill(19);
    let frozen = original.freeze();
    let address = frozen.bytes().as_ptr();
    let usage = host.usage().unwrap();
    let clone = frozen.clone();
    let frozen = frozen.try_thaw().unwrap_err();
    assert!(frozen.same_region(&clone));
    assert_eq!(host.usage().unwrap(), usage);
    drop(clone);

    let slice = frozen.slice(2..5).unwrap();
    let frozen = frozen.try_thaw().unwrap_err();
    assert_eq!(frozen.bytes().as_ptr(), address);
    assert_eq!(frozen.bytes(), &[19; 8]);
    assert_eq!(slice.bytes(), &[19; 3]);
    assert_eq!(host.usage().unwrap(), usage);
    drop(slice);

    let empty = frozen.slice(8..8).unwrap();
    let frozen = frozen.try_thaw().unwrap_err();
    assert_eq!(frozen.bytes().as_ptr(), address);
    assert_eq!(frozen.bytes(), &[19; 8]);
    assert_eq!(host.usage().unwrap(), usage);
    drop(empty);
    let writable = frozen.try_thaw().unwrap();
    assert_eq!(writable.as_ref().as_ptr(), address);
    assert_eq!(writable.as_ref(), &[19; 8]);
    assert_eq!(host.usage().unwrap(), usage);
    drop(writable);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn sole_partial_or_empty_view_cannot_recover_bytes_outside_its_logical_range() {
    for range in [0..3, 1..4, 2..2] {
        let host = account(4, 1);
        let mut original = host
            .allocate_zeroed(HostMemoryClass::WeightBacking, 4)
            .unwrap();
        original.as_mut().copy_from_slice(&[1, 2, 3, 4]);
        let frozen = original.freeze();
        let slice = frozen.slice(range.clone()).unwrap();
        let address = slice.bytes().as_ptr();
        drop(frozen);
        let usage = host.usage().unwrap();
        let slice = slice.try_thaw().unwrap_err();
        assert_eq!(slice.bytes().as_ptr(), address);
        assert_eq!(slice.bytes(), &[1, 2, 3, 4][range]);
        assert_eq!(slice.backing_bytes(), 4);
        assert_eq!(host.usage().unwrap(), usage);
        drop(slice);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn full_range_slice_recovers_original_buffer_after_other_owner_drops() {
    let host = account(4, 1);
    let frozen = host
        .allocate_zeroed(HostMemoryClass::WeightBacking, 4)
        .unwrap()
        .freeze();
    let full = frozen.slice(0..frozen.len()).unwrap();
    let address = frozen.bytes().as_ptr();
    drop(frozen);
    let recovered = full.try_thaw().unwrap();
    assert_eq!(recovered.as_ref().as_ptr(), address);
    assert_eq!(recovered.as_ref(), &[0; 4]);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 4);
    drop(recovered);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn native_regions_block_thaw_and_closed_scope_keeps_charge_until_final_release() {
    let parent = account(16, 1);
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 16)])
        .unwrap();
    let scoped = scope.account();
    let mut original = scoped
        .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
        .unwrap();
    original.as_mut().fill(29);
    let frozen = original.freeze();
    let address = frozen.bytes().as_ptr();
    let region = frozen.retained_region().unwrap();
    let native_owner = region.clone();
    let frozen = frozen.try_thaw().unwrap_err();
    assert_eq!(frozen.bytes().as_ptr(), address);
    assert_eq!(region.bytes(), &[29; 8]);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 16);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 8);

    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 8);
    assert!(matches!(
        scoped.reserve(HostMemoryClass::WeightBacking, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    drop(region);
    let usage = parent.usage().unwrap();
    let frozen = frozen.try_thaw().unwrap_err();
    assert_eq!(native_owner.bytes(), &[29; 8]);
    assert_eq!(parent.usage().unwrap(), usage);
    drop(native_owner);

    let mut recovered = frozen.try_thaw().unwrap();
    assert_eq!(recovered.as_ref().as_ptr(), address);
    assert_eq!(recovered.as_ref(), &[29; 8]);
    assert_eq!(parent.usage().unwrap(), usage);
    recovered.as_mut()[0] = 37;
    let frozen = recovered.freeze();
    assert!(scoped.owns_in_scope(&frozen));
    assert!(!parent.owns_in_scope(&frozen));
    assert_eq!(frozen.bytes(), &[37, 29, 29, 29, 29, 29, 29, 29]);
    assert!(matches!(
        scoped.reserve(HostMemoryClass::WeightBacking, 1),
        Err(HostMemoryError::ScopeClosed)
    ));
    assert_eq!(parent.usage().unwrap(), usage);
    drop(frozen);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(parent.usage().unwrap().reserved_bytes(), 0);
}
