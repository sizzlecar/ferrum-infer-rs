use super::*;
use std::num::NonZeroUsize;

#[test]
fn one_allocation_keeps_full_charge_and_exact_smaller_payloads_match_fresh_conversion() {
    let (_directory, archive) = archive(&[
        bf("first", 1, 8, &[1., -0., 2., 3., 4., 5., 6., 7.]),
        bf("second", 1, 8, &[8., 9., 10., 11., 12., 13., 14., 15.]),
    ]);
    let specs = [
        component(&["second", "first"], &[2, 8], ElementType::F16),
        component(&["first"], &[1, 8], ElementType::F16),
        component(&["second"], &[1, 8], ElementType::F32),
    ];
    let host = host();
    let fresh = ManagedDenseArchiveSource::new(&archive, &host);
    let expected = specs
        .iter()
        .map(|spec| fresh.component(spec).unwrap().bytes().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let installation = fresh
        .reusable_installation(NonZeroUsize::new(32).unwrap())
        .unwrap();
    let physical = host.backing_bytes(32).unwrap();
    assert_eq!(installation.backing_bytes(), physical);
    let mut first_address = None;
    for (spec, expected) in specs.iter().zip(expected) {
        let mut report = DenseArchiveMaterializationReport::default();
        let payload = installation
            .component_with_diagnostics(spec, DenseArchiveSourceTouch::AsMapped, &mut report)
            .unwrap();
        assert!(report.completed);
        assert_eq!(payload.bytes(), expected);
        assert_eq!(payload.bytes().len() as u64, spec.physical_bytes().unwrap());
        assert_eq!(
            payload.bytes().as_ptr(),
            *first_address.get_or_insert(payload.bytes().as_ptr())
        );
        assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
        assert!(installation.component(spec).is_err());
        drop(payload);
        assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
        assert_eq!(host.usage().unwrap().committed_bytes(), physical);
    }
    // Opting in did not change the original source: it still grants fresh
    // independently charged payloads rather than entering the installation.
    let fresh_payload = fresh.component(&specs[0]).unwrap();
    assert_eq!(host.usage().unwrap().allocated_bytes(), physical * 2);
    drop(fresh_payload);
    drop(installation);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn retained_native_owner_blocks_reuse_and_outlives_installation_and_scope_close() {
    let (_directory, archive) = archive(&[bf("value", 1, 8, &[0.5; 8])]);
    let spec = component(&["value"], &[1, 8], ElementType::F16);
    let host = host();
    let physical = host.backing_bytes(16).unwrap();
    let scope = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, physical)])
        .unwrap();
    let scoped = scope.account();
    let source = ManagedDenseArchiveSource::new(&archive, &scoped);
    let installation = source
        .reusable_installation(NonZeroUsize::new(16).unwrap())
        .unwrap();
    let payload = installation.component(&spec).unwrap();
    let region = payload.retained_host_memory().unwrap().clone();
    let expected = payload.bytes().to_vec();
    drop(payload);
    assert!(installation.component(&spec).is_err());
    assert_eq!(region.bytes(), expected);
    assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
    drop(region);
    let payload = installation.component(&spec).unwrap();
    assert_eq!(payload.bytes(), expected);
    let native_owner = payload.retained_host_memory().unwrap().clone();
    drop(payload);
    scope.close().unwrap();
    drop(installation);
    assert_eq!(host.usage().unwrap().committed_bytes(), physical);
    assert_eq!(native_owner.bytes(), expected);
    drop(native_owner);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn exact_scope_capacity_is_required_and_oversized_components_never_allocate_fallback() {
    let (_directory, archive) = archive(&[bf("value", 1, 9, &[0.5; 9])]);
    let spec = component(&["value"], &[1, 9], ElementType::F16);
    let host = host();
    let physical = host.backing_bytes(16).unwrap();
    for bytes in [physical - 1, physical] {
        let scope = host
            .reserve_scope(&[(HostMemoryClass::WeightBacking, bytes)])
            .unwrap();
        let scoped = scope.account();
        let source = ManagedDenseArchiveSource::new(&archive, &scoped);
        let installation = source.reusable_installation(NonZeroUsize::new(16).unwrap());
        assert_eq!(installation.is_ok(), bytes == physical);
        if let Ok(installation) = installation {
            assert!(installation.component(&spec).is_err());
            assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
            drop(installation);
        }
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        scope.close().unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn late_conversion_error_publishes_nothing_and_next_conversion_overwrites_the_same_slot() {
    let mut bad = [1.0; 16];
    bad[15] = 65536.0;
    let (_directory, archive) = archive(&[bf("bad", 2, 8, &bad), bf("good", 2, 8, &[0.25; 16])]);
    let host = host();
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    let installation = source
        .reusable_installation(NonZeroUsize::new(32).unwrap())
        .unwrap();
    let physical = installation.backing_bytes();
    let bad = component(&["bad"], &[2, 8], ElementType::F16);
    let good = component(&["good"], &[2, 8], ElementType::F16);
    let initial = installation.component(&good).unwrap();
    let address = initial.bytes().as_ptr();
    drop(initial);
    let mut report = DenseArchiveMaterializationReport::default();
    assert!(installation
        .component_with_diagnostics(&bad, DenseArchiveSourceTouch::AsMapped, &mut report)
        .is_err());
    assert!(!report.completed);
    assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
    let payload = installation.component(&good).unwrap();
    assert_eq!(payload.bytes().as_ptr(), address);
    assert_eq!(payload.bytes(), f16_bytes(&[0.25; 16]));
    drop(payload);
    drop(installation);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn affine_and_borrowed_components_preserve_default_math_and_storage_contracts() {
    let (_directory, archive) = archive(&[bf("value", 1, 8, &[0.125; 8])]);
    let host = host();
    let source = ManagedDenseArchiveSource::new(&archive, &host);
    let mut affine = component(&["value"], &[1, 8], ElementType::F16);
    affine.encoding = WeightEncoding::DenseAffine {
        element_type: ElementType::F16,
        scale: CanonicalRational::new(3, 2).unwrap(),
        bias: CanonicalRational::new(-1, 1).unwrap(),
    };
    let expected = source.component(&affine).unwrap().bytes().to_vec();
    let installation = source
        .reusable_installation(NonZeroUsize::new(16).unwrap())
        .unwrap();
    let payload = installation.component(&affine).unwrap();
    assert_eq!(payload.bytes(), expected);
    drop(payload);
    let borrowed = component(&["value"], &[1, 8], ElementType::Bf16);
    let payload = installation.component(&borrowed).unwrap();
    assert_eq!(
        payload.bytes().as_ptr(),
        archive.tensor("value").unwrap().bytes().as_ptr()
    );
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        installation.backing_bytes()
    );
    drop(payload);
    drop(installation);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
