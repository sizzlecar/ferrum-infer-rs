use super::*;

fn part(format: Option<GgufBlockFormat>, offset: u32) -> MatrixPart {
    MatrixPart {
        format,
        outputs: 3,
        columns: 256,
        output_offset: offset,
        transformed: false,
    }
}

fn mixed() -> [MatrixPart; 4] {
    [
        part(Some(GgufBlockFormat::Q4K), 0),
        part(Some(GgufBlockFormat::Q5K), 3),
        part(Some(GgufBlockFormat::Q6K), 6),
        part(Some(GgufBlockFormat::Q8_0), 9),
    ]
}

#[test]
fn q8_matrix_plan_leaf_partition_preserves_whole_arithmetic_and_shared_pack() {
    let plan = MatrixPlan::new(8, 256, 12, Q8SumPolicy::Input, mixed()).unwrap();
    let packed = plan.leaf(8).unwrap();
    let leaf = plan.leaf(1).unwrap();
    assert_eq!(packed.policy, Q8SumPolicy::Input);
    assert_eq!(leaf.policy, packed.policy);
    assert_eq!(packed.quantized_kernel, Some(QuantizedKernel::Mma));
    assert_eq!(leaf.quantized_kernel, Some(QuantizedKernel::Scalar));
    assert_eq!(
        plan.leaf(7).unwrap().quantized_kernel,
        Some(QuantizedKernel::Tiled)
    );
    // All three integer formats share one pack; the Q8_0 partition stays strict.
    assert_eq!(packed.dispatches, 5);
    assert_eq!(leaf.dispatches, 5);
    assert_eq!(
        packed.pack.unwrap().total_bytes,
        8 * leaf.pack.unwrap().total_bytes
    );
    assert!(plan.leaf(0).is_err());
    assert!(plan.leaf(9).is_err());
}

#[test]
fn q8_matrix_plan_chunk_topology_and_live_scratch_cover_the_same_rows() {
    let rows = MAX_LEAF_ROWS + 1;
    let plan = MatrixPlan::new(rows, 256, 12, Q8SumPolicy::Quantized, mixed()).unwrap();
    assert_eq!(
        plan.dispatches(rows).unwrap(),
        plan.leaf(MAX_LEAF_ROWS).unwrap().dispatches + plan.leaf(1).unwrap().dispatches
    );
    let workspace = plan.workspace_bytes(MAX_LEAF_ROWS).unwrap();
    assert_eq!(
        workspace,
        plan.leaf(MAX_LEAF_ROWS).unwrap().pack.unwrap().total_bytes
    );
    assert!(plan.leaf(1).unwrap().pack.unwrap().total_bytes <= workspace);
    assert!(plan.workspace_bytes(rows).unwrap() > workspace);
    assert!(plan.leaf(rows).is_err());
    assert!(plan.dispatches(rows + 1).is_err());
}

#[test]
fn q8_matrix_plan_keeps_sum_policy_in_scratch_and_leaf_identity() {
    let quantized = MatrixPlan::new(33, 256, 12, Q8SumPolicy::Quantized, mixed()).unwrap();
    let input = MatrixPlan::new(33, 256, 12, Q8SumPolicy::Input, mixed()).unwrap();
    assert_eq!(
        quantized.dispatches(33).unwrap(),
        input.dispatches(33).unwrap()
    );
    assert_ne!(quantized.policy(), input.policy());
    let a = quantized.leaf(33).unwrap().pack.unwrap();
    let b = input.leaf(33).unwrap().pack.unwrap();
    assert_eq!(a.sums_bytes, 0);
    assert_eq!(b.sums_bytes, b.scales_bytes);
    assert_eq!(b.total_bytes - a.total_bytes, b.sums_bytes);
    assert_eq!(a.words_offset, a.scales_bytes);
    assert_eq!(b.words_offset, b.scales_bytes + b.sums_bytes);
}

#[test]
fn q8_matrix_plan_strict_only_parts_need_no_pack_and_keep_valid_dense_columns() {
    let dense = MatrixPart {
        columns: 33,
        ..part(None, 1)
    };
    let plan = MatrixPlan::new(3, 33, 4, Q8SumPolicy::Input, [dense]).unwrap();
    assert_eq!(plan.columns(), 33);
    assert_eq!(plan.output_stride(), 4);
    assert_eq!(plan.pack_bytes_per_row(), 0);
    assert_eq!(plan.workspace_bytes(3).unwrap(), 0);
    assert_eq!(plan.leaf(3).unwrap().pack, None);
    assert_eq!(plan.leaf(3).unwrap().quantized_kernel, None);
    assert_eq!(plan.dispatches(3).unwrap(), 1);
    let q8 = MatrixPlan::new(
        3,
        256,
        3,
        Q8SumPolicy::Input,
        [part(Some(GgufBlockFormat::Q8_0), 0)],
    )
    .unwrap();
    assert_eq!(q8.workspace_bytes(3).unwrap(), 0);
}

#[test]
fn q8_matrix_plan_rejects_invalid_matrix_support_and_overflow_before_launch() {
    let good = part(Some(GgufBlockFormat::Q5K), 0);
    for invalid in [
        MatrixPart {
            transformed: true,
            ..good
        },
        MatrixPart {
            columns: 512,
            ..good
        },
        MatrixPart { outputs: 0, ..good },
        MatrixPart {
            output_offset: u32::MAX,
            ..good
        },
        MatrixPart {
            output_offset: 1,
            ..good
        },
    ] {
        assert!(MatrixPlan::new(8, 256, 3, Q8SumPolicy::Input, [invalid]).is_err());
    }
    assert!(MatrixPlan::new(
        8,
        255,
        3,
        Q8SumPolicy::Input,
        [MatrixPart {
            columns: 255,
            ..good
        }]
    )
    .is_err());
    assert!(MatrixPlan::new(
        8,
        33,
        3,
        Q8SumPolicy::Input,
        [MatrixPart {
            columns: 33,
            format: Some(GgufBlockFormat::Q8_0),
            ..good
        }]
    )
    .is_err());
    assert!(MatrixPlan::new(1, 256, 3, Q8SumPolicy::Input, []).is_err());
    assert!(MatrixPlan::new(0, 256, 3, Q8SumPolicy::Input, [good]).is_err());
    assert!(MatrixPlan::new(u64::MAX, 256, 3, Q8SumPolicy::Input, [good]).is_err());
}
