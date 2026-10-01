use super::*;

fn resource(name: &str) -> ResourceId {
    ResourceId::new(name).unwrap()
}
fn shape(copy: &[(&str, u64)], zero: &[(&str, u64)]) -> NativeCheckpointTransferGeometry {
    let mut builder = NativeCheckpointTransferGeometryBuilder::new();
    for &(id, bytes) in copy {
        builder.push_copy(&resource(id), bytes).unwrap();
    }
    for &(id, bytes) in zero {
        builder.push_initialization(&resource(id), bytes).unwrap();
    }
    builder.finish().unwrap()
}

#[test]
fn checkpoint_geometry_preserves_order_fragmentation_and_initialization_separately() {
    let original = shape(&[("kv", 3), ("kv", 5)], &[("state", 16)]);
    assert_eq!(original.copy_bytes(), 8);
    assert_eq!(original.copy_commands(), 2);
    assert_eq!(original.initialization_bytes(), 16);
    assert_eq!(original.initialization_commands(), 1);
    for changed in [
        shape(&[("kv", 5), ("kv", 3)], &[("state", 16)]),
        shape(&[("kv", 4), ("kv", 4)], &[("state", 16)]),
        shape(&[("other", 3), ("kv", 5)], &[("state", 16)]),
        shape(&[("kv", 3), ("kv", 5)], &[("state", 8), ("state", 8)]),
    ] {
        assert_ne!(original, changed);
        assert_ne!(
            original.ordered_fragments_fingerprint(),
            changed.ordered_fragments_fingerprint()
        );
    }
    let mut forecast = NativeCheckpointTransferGeometryBuilder::new();
    forecast
        .push_initialization(&resource("state"), 16)
        .unwrap();
    forecast.push_copy(&resource("kv"), 3).unwrap();
    forecast.push_copy(&resource("kv"), 5).unwrap();
    assert_eq!(
        original,
        forecast.finish().unwrap(),
        "preparation order cannot change native zero-then-copy order"
    );
}

#[test]
fn checkpoint_geometry_rejects_overflow_and_invalid_partial_result() {
    let mut zero = NativeCheckpointTransferGeometryBuilder::new();
    assert!(zero.push_copy(&resource("kv"), 0).is_err());
    assert!(zero.push_copy(&resource("kv"), 1).is_err());
    assert!(zero.finish().is_err());
    let mut overflow = NativeCheckpointTransferGeometryBuilder::new();
    overflow.push_copy(&resource("kv"), u64::MAX).unwrap();
    assert!(overflow.push_copy(&resource("kv"), 1).is_err());
    assert!(overflow.finish().is_err());
    let mut empty = NativeCheckpointTransferGeometryBuilder::new();
    empty.push_initialization(&resource("state"), 4).unwrap();
    assert!(empty.finish().is_err());
}

#[test]
fn checkpoint_host_work_preserves_both_token_extents_and_rejects_invalid_lengths() {
    let short = NativeCheckpointTransferHostWork::from_lengths(2, 4).unwrap();
    assert_eq!(short.prefix_tokens(), 2);
    assert_eq!(short.full_input_tokens(), 4);
    assert_ne!(
        short,
        NativeCheckpointTransferHostWork::from_lengths(2, 5).unwrap()
    );
    assert_ne!(
        short,
        NativeCheckpointTransferHostWork::from_lengths(3, 4).unwrap()
    );
    assert!(NativeCheckpointTransferHostWork::from_lengths(0, 4).is_err());
    assert!(NativeCheckpointTransferHostWork::from_lengths(5, 4).is_err());
    let start = CheckpointTransferObservationStart::now();
    assert_eq!(start.host_work(), None);
    assert_eq!(start.with_host_work(short).host_work(), Some(short));
}

#[test]
fn strided_checkpoint_geometry_binds_width_height_and_both_pitches() {
    let build = |region: StridedCopyRegion| {
        let mut builder = NativeCheckpointTransferGeometryBuilder::new();
        builder.push_strided_copy(&resource("kv"), region).unwrap();
        builder.finish().unwrap()
    };
    let original = StridedCopyRegion::new(9, 13, 6, 8, 32, 6).unwrap();
    let shape = build(original);
    assert_eq!(shape.copy_bytes(), 48);
    assert_eq!(shape.copy_commands(), 1);
    assert_eq!(
        shape,
        build(StridedCopyRegion::new(19, 23, 6, 8, 32, 6).unwrap()),
        "physical allocation offsets are ownership facts outside the cost family"
    );
    assert_ne!(shape, build(original.reversed()));
    assert_ne!(
        shape,
        build(StridedCopyRegion::new(9, 13, 6, 8, 64, 6).unwrap())
    );
    assert_ne!(
        shape,
        build(StridedCopyRegion::new(9, 13, 12, 4, 32, 12).unwrap())
    );
    assert_ne!(shape, super::tests::shape(&[("kv", 48)], &[]));
}
