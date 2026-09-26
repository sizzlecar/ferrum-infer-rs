use super::*;

fn selected(capture: bool, context: u64, block: u32) -> SelectedCommandCostEvidenceV1 {
    let mut builder = if capture {
        SelectedCommandCostBuilderV1::new_with_algorithm_work(4)
    } else {
        SelectedCommandCostBuilderV1::new(4)
    };
    let algorithm = SelectedAlgorithmClassV1::new("binding.reuse", 1, [3; 32], [7; 32]).unwrap();
    builder
        .kernel_with_replay_geometry(
            algorithm,
            KernelNumericWorkV1 {
                logical_units: 4,
                padded_units: 8,
                inner_units_per_logical_unit: context,
                grid: [2, 2, 1],
                scratch_bytes: 256,
                staged_weight_bytes: 32,
            },
            KernelReplayGeometryV1 {
                block: [block, 1, 1],
                dynamic_shared_bytes: 128,
                fixed_parameters: &[17, 31],
            },
        )
        .unwrap();
    for kind in [
        StatisticalTransferKindV1::HostToDevice,
        StatisticalTransferKindV1::DeviceToHost,
        StatisticalTransferKindV1::DeviceToDevice,
        StatisticalTransferKindV1::Fill,
    ] {
        builder
            .transfer_with_replay_geometry(
                algorithm,
                kind,
                64,
                TransferReplayGeometryV1 {
                    row_bytes: 16,
                    rows: 4,
                    destination_stride_bytes: 32,
                },
            )
            .unwrap();
    }
    builder.finish().unwrap()
}

#[test]
fn selected_binding_reuse_preserves_original_digest_wire_and_shared_storage() {
    for context in [1, 257, 65_537] {
        let captured = selected(true, context, 128);
        let plain = selected(false, context, 128);
        let serialized = serde_json::to_vec(&captured).unwrap();
        assert_eq!(serialized, serde_json::to_vec(&plain).unwrap());
        let mut constructing = captured.clone();
        // Reproduce the state at the original private construction boundary:
        // same final immutable fields, before attaching the validated table.
        let table = constructing.algorithm_work.take().unwrap().unwrap();
        let original_digest = constructing.algorithm_work_binding().unwrap();
        assert_eq!(captured.algorithm_work_binding().unwrap(), original_digest);
        assert_eq!(serialized, serde_json::to_vec(&constructing).unwrap());
        table.validate_command(&captured).unwrap();
        table.validate_command(&constructing).unwrap();
        let cloned = captured.clone();
        let shared = cloned.algorithm_work.as_ref().unwrap().as_ref().unwrap();
        assert!(std::sync::Arc::ptr_eq(&table, shared));
        assert_eq!(cloned.algorithm_work_binding().unwrap(), original_digest);
        assert_eq!(table.retained_bytes(), shared.retained_bytes());
    }
}

#[test]
fn selected_binding_reuse_keeps_fresh_context_and_fixed_replay_boundaries() {
    let captured = selected(true, 32, 128);
    let current = selected(true, 257, 128);
    assert_eq!(captured.family_signature(), current.family_signature());
    assert_ne!(captured.work(), current.work());
    assert_ne!(
        captured.algorithm_work_binding(),
        current.algorithm_work_binding()
    );
    assert_eq!(
        captured
            .algorithm_work()
            .unwrap()
            .unwrap()
            .validate_command(&current),
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
    let template = SelectedReplayAlgorithmTemplateV1::from_selected(&captured, 4, 1, 4).unwrap();
    // Replay authorizes the same fixed launch with freshly computed work; it
    // must not accidentally require or borrow capture-time numeric work.
    template.validate_binding(&current).unwrap();
    assert!(template.validate_binding(&selected(true, 257, 64)).is_err());
    let missing = selected(false, 257, 128);
    assert_eq!(
        missing.algorithm_work_binding(),
        Err(StatisticalEvidenceUnknown::MissingProducer)
    );
    assert!(template.validate_binding(&missing).is_err());
}
