use super::*;

fn spec() -> WeightComponentStreamSpec {
    WeightComponentStreamSpec::new(
        &super::super::tests::packed_component(),
        vec!["one.safetensors".into(), "two.safetensors".into()],
        8,
    )
    .unwrap()
}

#[test]
fn stream_ranges_reject_zero_unaligned_overflow_and_outside_complete_identity() {
    let spec = spec();
    assert_eq!(spec.component().physical_bytes().unwrap(), 16);
    for (offset, length) in [
        (0, 0),
        (1, 2),
        (0, 3),
        (0, 10),
        (16, 2),
        (usize::MAX - 1, 4),
    ] {
        assert!(WeightComponentRangeRequest::new(&spec, offset, length).is_err());
    }
    let first = WeightComponentRangeRequest::new(&spec, 0, 8).unwrap();
    let tail = WeightComponentRangeRequest::new(&spec, 8, 8).unwrap();
    assert_eq!(first.component(), spec.component());
    assert_eq!(tail.offset_bytes() + tail.length_bytes(), 16);
    assert!(WeightComponentStreamSpec::new(spec.component(), vec!["../bad".into(); 2], 8).is_err());
    assert!(WeightComponentStreamSpec::new(spec.component(), vec!["one".into()], 8).is_err());
}

#[test]
fn stream_range_rejects_wrong_request_class_length_and_original_host_scope() {
    let host = super::super::tests::managed_account(128);
    let spec = spec();
    let first = WeightComponentRangeRequest::new(&spec, 0, 8).unwrap();
    let tail = WeightComponentRangeRequest::new(&spec, 8, 8).unwrap();
    for (class, length) in [
        (HostMemoryClass::Output, 8),
        (HostMemoryClass::WeightBacking, 4),
    ] {
        let bytes = host.allocate_zeroed(class, length).unwrap().freeze();
        assert!(WeightComponentRange::new(&first, bytes).is_err());
    }
    let make = || {
        host.allocate_zeroed(HostMemoryClass::WeightBacking, 8)
            .unwrap()
            .freeze()
    };
    assert!(WeightComponentRange::new(&first, make())
        .unwrap()
        .validate_and_take(&tail, &host)
        .is_err());
    let foreign = super::super::tests::managed_account(128);
    assert!(WeightComponentRange::new(&first, make())
        .unwrap()
        .validate_and_take(&first, &foreign)
        .is_err());
    let target = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 32)])
        .unwrap();
    assert!(WeightComponentRange::new(&first, make())
        .unwrap()
        .validate_and_take(&first, &target.account())
        .is_err());
    let bytes = target
        .account()
        .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
        .unwrap()
        .freeze();
    let original = bytes.bytes().as_ptr();
    let retained = WeightComponentRange::new(&first, bytes)
        .unwrap()
        .validate_and_take(&first, &target.account().clone())
        .unwrap();
    assert_eq!(retained.bytes().as_ptr(), original);
    drop(retained);
    drop(target);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
