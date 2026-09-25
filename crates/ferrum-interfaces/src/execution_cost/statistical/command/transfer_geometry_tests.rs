use super::*;

fn evidence(row_bytes: u64, rows: u64, stride: u64) -> SelectedCommandCostEvidenceV1 {
    let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(4);
    builder
        .transfer_with_replay_geometry(
            SelectedAlgorithmClassV1::new("copy.2d", 1, [1; 32], [2; 32]).unwrap(),
            StatisticalTransferKindV1::HostToDevice,
            row_bytes * rows,
            TransferReplayGeometryV1 {
                row_bytes,
                rows,
                destination_stride_bytes: stride,
            },
        )
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn transfer_geometry_keeps_numeric_work_out_of_family_but_not_sealed_binding() {
    let captured = evidence(8, 4, 64);
    let template = SelectedReplayAlgorithmTemplateV1::from_selected(&captured, 4, 0, 1).unwrap();
    template.validate_binding(&evidence(8, 4, 64)).unwrap();
    for changed in [
        evidence(12, 4, 64),
        evidence(8, 4, 128),
        evidence(16, 2, 64),
    ] {
        assert_eq!(captured.family_signature(), changed.family_signature());
        assert!(template.validate_binding(&changed).is_err());
    }
    // Equal bytes/work do not establish fixed pitch or dimensions.
    assert_eq!(captured.work(), evidence(8, 4, 128).work());
    assert_eq!(captured.work(), evidence(16, 2, 64).work());
    assert_eq!(evidence(12, 4, 64).work().host_to_device_bytes, 48);
}

#[test]
fn transfer_geometry_rejects_invalid_or_overflowing_spans_and_keeps_failure_sticky() {
    for (row_bytes, rows, stride, total) in [
        (0, 4, 64, 1),
        (8, 0, 64, 1),
        (8, 4, 7, 32),
        (8, 4, 64, 31),
        (u64::MAX, 2, u64::MAX, 1),
        (1, 3, u64::MAX, 3),
    ] {
        let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(4);
        let class = SelectedAlgorithmClassV1::new("copy.2d", 1, [1; 32], [2; 32]).unwrap();
        let result = builder.transfer_with_replay_geometry(
            class,
            StatisticalTransferKindV1::HostToDevice,
            total,
            TransferReplayGeometryV1 {
                row_bytes,
                rows,
                destination_stride_bytes: stride,
            },
        );
        assert!(result.is_err(), "{row_bytes}/{rows}/{stride}/{total}");
        assert_eq!(
            builder.transfer(class, StatisticalTransferKindV1::HostToDevice, 8),
            result
        );
        assert_eq!(builder.finish().unwrap_err(), result.unwrap_err());
    }
}

#[test]
fn transfer_geometry_cannot_be_omitted_to_reuse_a_sealed_template() {
    let actual = evidence(8, 4, 64);
    let template = SelectedReplayAlgorithmTemplateV1::from_selected(&actual, 4, 0, 1).unwrap();
    let mut legacy = SelectedCommandCostBuilderV1::new_with_algorithm_work(4);
    legacy
        .transfer(
            SelectedAlgorithmClassV1::new("copy.2d", 1, [1; 32], [2; 32]).unwrap(),
            StatisticalTransferKindV1::HostToDevice,
            32,
        )
        .unwrap();
    let legacy = legacy.finish().unwrap();
    assert_eq!(actual.family_signature(), legacy.family_signature());
    assert_eq!(actual.work(), legacy.work());
    assert!(template.validate_binding(&legacy).is_err());
}
