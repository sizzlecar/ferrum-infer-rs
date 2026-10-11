//! Shape and retained-resource invariants for native eight-row partitions.

use super::*;

const FORMATS: [GgufBlockFormat; 3] = [
    GgufBlockFormat::Q3K,
    GgufBlockFormat::Iq3S,
    GgufBlockFormat::Iq4Nl,
];

fn launch(
    format: GgufBlockFormat,
    rows: u64,
    width: u64,
    outputs: u32,
    activation_type: ElementType,
) -> LinearLaunch {
    linear_launch_typed(
        PreparedLinearPart {
            region: 7,
            transform: None,
            format: LinearPhysicalFormat::Native(format),
            output_offset: 3,
            out_features: outputs,
        },
        2,
        5,
        rows,
        width,
        u64::from(outputs) + 12,
        32,
        48,
        activation_type,
    )
    .unwrap()
}

fn assert_single(launch: LinearLaunch) {
    let plan = PlainLinearPlan::for_launch(launch);
    assert!(matches!(plan, PlainLinearPlan::Single));
    assert_eq!(plan.dispatch_count(), 1);
    assert!(plan.parts(launch).is_none());
}

#[test]
fn native_m8_plan_covers_rows_once_and_preserves_retained_resources() {
    for format in FORMATS {
        for width in [1024, 1280, 5120, 17408] {
            for outputs in [
                SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES,
                SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES + 1,
                5120,
                17408,
            ] {
                let original = launch(format, 8, width, outputs, ElementType::F16);
                let [head, tail] = original.plain_plan.parts(original).expect("two B4 spans");
                assert_eq!(original.dispatch_count(), 2);
                assert_eq!((head.params.rows, tail.params.rows), (4, 4));
                assert_eq!(head.params.rows + tail.params.rows, original.params.rows);
                let scalar_bytes = original.activation_type.size_bytes();
                for part in [head, tail] {
                    assert_eq!(part.input_region, original.input_region);
                    assert_eq!(part.weight_region, original.weight_region);
                    assert_eq!(part.output_region, original.output_region);
                    assert_eq!(part.activation_type, original.activation_type);
                    assert_eq!(part.format, original.format);
                    assert_eq!(part.params.in_features, original.params.in_features);
                    assert_eq!(part.params.out_features, original.params.out_features);
                    assert_eq!(part.params.output_stride, original.params.output_stride);
                    assert_eq!(
                        part.params.output_column_offset,
                        original.params.output_column_offset
                    );
                    assert!(part.transform.is_none());
                    assert!(part.transform_workspace.is_none());
                    // Prepared children must not recursively split or invent
                    // extra workspaces/resources during command encoding.
                    assert_eq!(part.dispatch_count(), 1);
                    assert_single(part);
                }
                assert_eq!(head.input_offset_bytes, original.input_offset_bytes);
                assert_eq!(head.output_offset_bytes, original.output_offset_bytes);
                let head_input_bytes = u64::from(head.params.rows) * width * scalar_bytes;
                let head_output_bytes = u64::from(head.params.rows)
                    * u64::from(original.params.output_stride)
                    * scalar_bytes;
                assert_eq!(
                    tail.input_offset_bytes,
                    head.input_offset_bytes + head_input_bytes
                );
                assert_eq!(
                    tail.output_offset_bytes,
                    head.output_offset_bytes + head_output_bytes
                );
                assert_eq!(
                    tail.input_offset_bytes + u64::from(tail.params.rows) * width * scalar_bytes,
                    original.input_offset_bytes
                        + u64::from(original.params.rows) * width * scalar_bytes,
                );
                assert_eq!(
                    tail.output_offset_bytes
                        + u64::from(tail.params.rows)
                            * u64::from(original.params.output_stride)
                            * scalar_bytes,
                    original.output_offset_bytes
                        + u64::from(original.params.rows)
                            * u64::from(original.params.output_stride)
                            * scalar_bytes,
                );
            }
        }
    }
}

#[test]
fn native_m8_plan_retains_short_narrow_and_partial_group_routes() {
    let wide = SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES;
    for format in FORMATS {
        // Complete B4 groups alone are not an authorization to change M4,
        // M12 or M16; partial groups must also keep their previous route.
        for rows in [1, 2, 3, 4, 5, 7, 9, 12, 16] {
            assert_single(launch(format, rows, 1024, wide, ElementType::F16));
        }
        for width in [
            format.block_values() as u64,
            1024 - format.block_values() as u64,
        ] {
            assert_single(launch(format, 8, width, wide, ElementType::F16));
        }
        for outputs in [7, wide - 1] {
            assert_single(launch(format, 8, 1024, outputs, ElementType::F16));
        }
        assert_single(launch(format, 8, 1024, wide, ElementType::F32));
        let mut transformed = launch(format, 8, 1024, wide, ElementType::F16);
        transformed.transform = Some(HadamardTransform {
            block_size: 128,
            signs_region: None,
            inverse: false,
            permutation: None,
        });
        assert_single(transformed);
    }
    // Neighboring native formats use different reconstruction/dispatch paths.
    for format in [GgufBlockFormat::Iq4Xs, GgufBlockFormat::Pq2_0] {
        assert_single(launch(format, 8, 1024, wide, ElementType::F16));
    }
}

#[test]
fn native_m8_plan_rejects_partition_offset_overflow() {
    for format in FORMATS {
        let original = launch(
            format,
            8,
            1024,
            SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES,
            ElementType::F16,
        );
        let input_delta = 4 * u64::from(original.params.in_features) * 2;
        let output_delta = 4 * u64::from(original.params.output_stride) * 2;
        assert_single(LinearLaunch {
            input_offset_bytes: u64::MAX - input_delta + 1,
            ..original
        });
        assert_single(LinearLaunch {
            output_offset_bytes: u64::MAX - output_delta + 1,
            ..original
        });
    }
}
