use super::*;

fn class(index: u32) -> SelectedAlgorithmClassV1 {
    let mut layout = [1; 32];
    layout[..4].copy_from_slice(&index.to_le_bytes());
    SelectedAlgorithmClassV1::new("selected_kernel", 1, [2; 32], layout).unwrap()
}
fn work(units: u64, scratch: u64) -> KernelNumericWorkV1 {
    KernelNumericWorkV1 {
        logical_units: units,
        padded_units: units,
        inner_units_per_logical_unit: 4,
        grid: [1, 1, 1],
        scratch_bytes: scratch,
        staged_weight_bytes: 0,
    }
}
fn command(capture: bool, tokens: u64, left: u64, right: u64) -> SelectedCommandCostEvidenceV1 {
    let mut builder = if capture {
        SelectedCommandCostBuilderV1::new_with_algorithm_work(tokens)
    } else {
        SelectedCommandCostBuilderV1::new(tokens)
    };
    builder.kernel(class(1), work(left, 16)).unwrap();
    builder.kernel(class(2), work(right, 32)).unwrap();
    builder.finish().unwrap()
}

#[test]
fn algorithm_work_opt_in_preserves_legacy_equality_wire_and_ordered_family() {
    let plain = command(false, 8, 4, 12);
    let captured = command(true, 8, 4, 12);
    assert!(plain.algorithm_work().is_none());
    assert_eq!(plain, captured);
    assert_eq!(
        serde_json::to_vec(&plain).unwrap(),
        serde_json::to_vec(&captured).unwrap()
    );
    assert_eq!(
        plain.independent_attention_family_v2(),
        captured.independent_attention_family_v2()
    );
    captured
        .algorithm_work()
        .unwrap()
        .unwrap()
        .validate_command(&captured)
        .unwrap();
}

#[test]
fn algorithm_work_distinguishes_equal_totals_assigned_to_different_algorithms() {
    let a = command(true, 8, 4, 12);
    let b = command(true, 8, 12, 4);
    assert_eq!(a.work(), b.work());
    assert_eq!(a.family_signature(), b.family_signature());
    assert_eq!(
        a.algorithm_work().unwrap().unwrap().validate_command(&b),
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
    let a = a.algorithm_work().unwrap().unwrap();
    let b = b.algorithm_work().unwrap().unwrap();
    assert_ne!(a.entries(), b.entries());
    let first = a
        .entries()
        .iter()
        .find(|e| e.algorithm() == class(1))
        .unwrap();
    assert_eq!(first.work().inner_work_units, 16);
}

#[test]
fn algorithm_work_keeps_copy_kinds_and_scratch_peaks_separate() {
    let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(8);
    builder.kernel(class(1), work(3, 64)).unwrap();
    builder.kernel(class(1), work(5, 16)).unwrap();
    builder.kernel(class(2), work(2, 128)).unwrap();
    builder
        .transfer(class(1), StatisticalTransferKindV1::HostToDevice, 100)
        .unwrap();
    builder
        .transfer(class(1), StatisticalTransferKindV1::HostToDevice, 200)
        .unwrap();
    builder
        .transfer(class(1), StatisticalTransferKindV1::DeviceToHost, 25)
        .unwrap();
    builder
        .transfer(class(1), StatisticalTransferKindV1::DeviceToDevice, 50)
        .unwrap();
    builder
        .transfer(class(1), StatisticalTransferKindV1::Fill, 75)
        .unwrap();
    let command = builder.finish().unwrap();
    let capture = command.algorithm_work().unwrap().unwrap();
    capture.validate_command(&command).unwrap();
    let find = |kind| {
        capture
            .entries()
            .iter()
            .find(|e| e.algorithm() == class(1) && e.kind() == kind)
            .unwrap()
    };
    assert_eq!(find(AlgorithmWorkKindV1::Kernel).commands(), 2);
    assert_eq!(find(AlgorithmWorkKindV1::Kernel).work().logical_units, 8);
    assert_eq!(
        find(AlgorithmWorkKindV1::Kernel).work().peak_scratch_bytes,
        64
    );
    assert_eq!(find(AlgorithmWorkKindV1::HostToDevice).commands(), 2);
    assert_eq!(
        find(AlgorithmWorkKindV1::HostToDevice)
            .work()
            .host_to_device_bytes,
        300
    );
    assert_eq!(
        find(AlgorithmWorkKindV1::DeviceToHost)
            .work()
            .device_to_host_bytes,
        25
    );
    assert_eq!(
        find(AlgorithmWorkKindV1::DeviceToDevice)
            .work()
            .device_to_device_bytes,
        50
    );
    assert_eq!(find(AlgorithmWorkKindV1::Fill).work().fill_bytes, 75);
    assert_eq!(command.work().peak_scratch_bytes, 128);
    assert!(
        capture.retained_bytes().unwrap()
            >= std::mem::size_of_val(capture)
                + std::mem::size_of_val(capture.entries())
                + 2 * std::mem::size_of::<usize>()
    );
}

#[test]
fn algorithm_work_binding_retains_numeric_order_within_one_algorithm() {
    let build = |first, second| {
        let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(8);
        builder.kernel(class(1), work(first, 0)).unwrap();
        builder.kernel(class(1), work(second, 0)).unwrap();
        builder.finish().unwrap()
    };
    let a = build(4, 12);
    let b = build(12, 4);
    assert_eq!(a, b); // Legacy semantics deliberately remain unchanged.
    let left = a.algorithm_work().unwrap().unwrap();
    let right = b.algorithm_work().unwrap().unwrap();
    assert_eq!(left.entries(), right.entries());
    assert_eq!(
        left.validate_command(&b),
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
}

#[test]
fn algorithm_work_rejects_binding_to_a_different_selected_command() {
    let a = command(true, 8, 4, 12);
    let changed_tokens = command(true, 4, 4, 12);
    let changed_work = command(true, 8, 5, 12);
    let capture = a.algorithm_work().unwrap().unwrap();
    assert_eq!(
        capture.validate_command(&changed_tokens),
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
    assert_eq!(
        capture.validate_command(&changed_work),
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
    let mut reverse = SelectedCommandCostBuilderV1::new_with_algorithm_work(8);
    reverse.kernel(class(2), work(12, 32)).unwrap();
    reverse.kernel(class(1), work(4, 16)).unwrap();
    let reverse = reverse.finish().unwrap();
    assert_eq!(a.work(), reverse.work());
    assert_eq!(
        capture.entries(),
        reverse.algorithm_work().unwrap().unwrap().entries()
    );
    assert_eq!(
        capture.validate_command(&reverse),
        Err(StatisticalEvidenceUnknown::CommandMismatch)
    );
}

#[test]
fn algorithm_work_preserves_failed_and_overflowing_command_ineligibility() {
    for capture in [false, true] {
        let mut builder = if capture {
            SelectedCommandCostBuilderV1::new_with_algorithm_work(1)
        } else {
            SelectedCommandCostBuilderV1::new(1)
        };
        builder.kernel(class(1), work(1, 0)).unwrap();
        assert_eq!(
            builder.kernel(class(2), work(u64::MAX, 0)),
            Err(StatisticalEvidenceUnknown::Overflow)
        );
        assert_eq!(builder.finish(), Err(StatisticalEvidenceUnknown::Overflow));
    }
}

#[test]
fn algorithm_work_covers_the_real_selected_command_capacity_boundary() {
    let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    for index in 0..MAX_COST_COMMANDS {
        builder.kernel(class(index as u32), work(1, 0)).unwrap();
    }
    let command = builder.finish().unwrap();
    let capture = command.algorithm_work().unwrap().unwrap();
    assert_eq!(capture.entries().len(), MAX_COST_COMMANDS);
    capture.validate_command(&command).unwrap();
    let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    for _ in 0..MAX_COST_COMMANDS {
        builder.kernel(class(1), work(1, 0)).unwrap();
    }
    assert_eq!(
        builder.kernel(class(1), work(1, 0)),
        Err(StatisticalEvidenceUnknown::Capacity)
    );
    assert_eq!(builder.finish(), Err(StatisticalEvidenceUnknown::Capacity));
}
