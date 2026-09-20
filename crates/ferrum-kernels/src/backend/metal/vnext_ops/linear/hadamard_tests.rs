use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::{
    BufferRequest, BufferUsage, DeviceId, HadamardApplication, HadamardSigns,
    HadamardTransformSpec, PhysicalWeightComponentBinding, ResourceId, WeightId,
};
use half::f16;
use metal::MTLCommandBufferStatus;
use std::num::NonZeroU32;

fn region<T: Copy>(
    runtime: &MetalDeviceRuntime,
    name: &str,
    values: &[T],
    element_type: ElementType,
) -> MetalBufferRegion {
    let region = runtime
        .allocate_test_region(
            &BufferRequest::new(
                ResourceId::new(name).unwrap(),
                std::mem::size_of_val(values) as u64,
                64,
                BufferUsage::Transfer,
                element_type,
            )
            .unwrap(),
        )
        .unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            std::mem::size_of_val(values),
        );
    }
    region
}

fn read<T: Copy>(region: &MetalBufferRegion) -> Vec<T> {
    unsafe {
        std::slice::from_raw_parts(
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize)
                .cast::<T>(),
            region.length_bytes() as usize / std::mem::size_of::<T>(),
        )
        .to_vec()
    }
}

#[test]
fn hadamard_pq2_projection_keeps_f32_transform_through_dot_and_plain_siblings_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.hadamard.linear").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    const WIDTH: usize = 1024;
    // Cover mixed GEMV, mixed M32 with token/output tails, and F32 master GEMV.
    for (dtype, rows, columns) in [
        (ElementType::F16, 1, 7),
        (ElementType::F16, 3, 17),
        (ElementType::F16, 3, 16),
        (ElementType::F16, 33, 1025),
        (ElementType::F32, 2, 7),
        (ElementType::F32, 1, 17),
        (ElementType::F32, 1, 16),
    ] {
        let signs = (0..WIDTH)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let mut input = vec![0.0_f32; rows * WIDTH];
        for row in 0..rows {
            for col in 0..WIDTH {
                input[row * WIDTH + col] = if col % 2 == 0 { 2.0 * signs[col] } else { 0.0 };
            }
            input[row * WIDTH + 1] = (row + 1) as f32
                * if dtype == ElementType::F16 {
                    1.0 / 1024.0
                } else {
                    1.0 / 4096.0
                };
        }
        let spec = HadamardTransformSpec {
            block_size: NonZeroU32::new(WIDTH as u32).unwrap(),
            signs: HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
                WeightId::new("linear.signs").unwrap(),
            )),
            application: HadamardApplication::BeforeMatmul {
                input_permutation: None,
            },
        };
        let transformed =
            crate::hadamard::apply_reference(&input, rows, WIDTH, &spec, Some(&signs)).unwrap();
        let format = GgufBlockFormat::Pq2_0;
        let mut weights = Vec::new();
        for column in 0..columns {
            for block in 0..WIDTH / 128 {
                weights.extend_from_slice(
                    &f16::from_f32(if column == 0 { 1.0 } else { 0.125 }).to_le_bytes(),
                );
                for byte in 0..32 {
                    // First output cancels the two large H coefficients. If H
                    // is rounded to half before the dot, its nonzero result is
                    // lost. Remaining outputs also exercise every packed code.
                    weights.push(if column == 0 {
                        if block == 0 && byte == 0 {
                            0x58
                        } else {
                            0x55
                        }
                    } else {
                        [0xe4, 0x1b, 0x55, 0xa0][(column + block + byte) % 4]
                    });
                }
            }
        }
        let mut decoded = vec![0.0; columns * WIDTH];
        format.decode(&weights, &mut decoded).unwrap();
        let stride = columns + 4;
        let prefix = 16 / dtype.size_bytes() as usize;
        let guard = -123.0_f32;
        let output_values = vec![guard; prefix + rows * stride + 8];
        let input_region = if dtype == ElementType::F16 {
            region(
                runtime,
                "linear.input",
                &input.iter().copied().map(f16::from_f32).collect::<Vec<_>>(),
                dtype,
            )
        } else {
            region(runtime, "linear.input", &input, dtype)
        };
        let output_region = if dtype == ElementType::F16 {
            region(
                runtime,
                "linear.output",
                &output_values
                    .iter()
                    .copied()
                    .map(f16::from_f32)
                    .collect::<Vec<_>>(),
                dtype,
            )
        } else {
            region(runtime, "linear.output", &output_values, dtype)
        };
        let scratch_bytes = vec![0xa5_u8; 128 + rows * WIDTH * 4];
        let regions = vec![
            input_region,
            region(runtime, "linear.weight", &weights, ElementType::U8),
            region(runtime, "linear.signs", &signs, ElementType::F32),
            output_region,
            region(runtime, "linear.scratch", &scratch_bytes, ElementType::U8),
        ];
        let part = PreparedLinearPart {
            region: 1,
            format: LinearPhysicalFormat::Native(format),
            output_offset: 0,
            out_features: columns as u32,
            transform: Some(HadamardTransform {
                block_size: WIDTH as u32,
                signs_region: Some(2),
                inverse: false,
                permutation: None,
            }),
        };
        let mut transformed_launch = linear_launch_typed(
            part,
            0,
            3,
            rows as u64,
            WIDTH as u64,
            stride as u64,
            0,
            16,
            dtype,
        )
        .unwrap();
        assert!(validate_launch_regions(&regions, &[transformed_launch]).is_err());
        assert!(transformed_launch
            .bind_hadamard_workspace(&pipelines, &regions, 4, 65)
            .is_err());
        transformed_launch
            .bind_hadamard_workspace(&pipelines, &regions, 4, 64)
            .unwrap();
        // The adjacent unrotated projection must continue to read the original
        // activation, as B/A do beside transformed GDN QKV/Z projections.
        let plain_launch = linear_launch_typed(
            PreparedLinearPart {
                transform: None,
                output_offset: columns as u32,
                out_features: 1,
                ..part
            },
            0,
            3,
            rows as u64,
            WIDTH as u64,
            stride as u64,
            0,
            16,
            dtype,
        )
        .unwrap();
        validate_launch_regions(&regions, &[transformed_launch, plain_launch]).unwrap();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        dispatch_linear(&pipelines, encoder, &regions, transformed_launch);
        dispatch_linear(&pipelines, encoder, &regions, plain_launch);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let output = if dtype == ElementType::F16 {
            read::<f16>(&regions[3])
                .into_iter()
                .map(f16::to_f32)
                .collect()
        } else {
            read::<f32>(&regions[3])
        };
        let round = |value: f32| {
            if dtype == ElementType::F16 {
                f16::from_f32(value).to_f32()
            } else {
                value
            }
        };
        for row in 0..rows {
            for column in 0..columns {
                let terms = transformed[row * WIDTH..(row + 1) * WIDTH]
                    .iter()
                    .zip(&decoded[column * WIDTH..(column + 1) * WIDTH])
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .collect::<Vec<_>>();
                let expected = round(terms.iter().sum::<f64>() as f32);
                let actual = output[prefix + row * stride + column];
                if column == 0 {
                    assert_ne!(expected, 0.0);
                    assert_eq!(
                        actual, expected,
                        "H must not narrow before dot: {dtype:?} row {row}"
                    );
                } else {
                    let accumulation_bound = 2.0
                        * WIDTH as f64
                        * f64::from(f32::EPSILON)
                        * terms.iter().map(|x| x.abs()).sum::<f64>();
                    let final_rounding = if dtype == ElementType::F16 {
                        f64::from(expected.abs()) / 1024.0 + 1.0 / 16777216.0
                    } else {
                        0.0
                    };
                    assert!(
                        (f64::from(actual) - f64::from(expected)).abs()
                            <= accumulation_bound + final_rounding,
                        "{dtype:?} row {row} col {column}: {actual} != {expected}"
                    );
                }
            }
            assert_eq!(
                output[prefix + row * stride + columns],
                round(-input[row * WIDTH] + input[row * WIDTH + 1])
            );
            assert!(
                output[prefix + row * stride + columns + 1..prefix + (row + 1) * stride]
                    .iter()
                    .all(|&x| x == guard)
            );
        }
        assert!(output[..prefix]
            .iter()
            .chain(&output[prefix + rows * stride..])
            .all(|&x| x == guard));
        let scratch = read::<u8>(&regions[4]);
        assert_eq!(&scratch[..64], &scratch_bytes[..64]);
        assert_eq!(
            &scratch[scratch.len() - 64..],
            &scratch_bytes[scratch_bytes.len() - 64..]
        );
        assert_eq!(read::<u8>(&regions[1]), weights);
        assert_eq!(read::<f32>(&regions[2]), signs);
        let actual_input = if dtype == ElementType::F16 {
            read::<f16>(&regions[0])
                .into_iter()
                .map(f16::to_f32)
                .collect()
        } else {
            read::<f32>(&regions[0])
        };
        assert_eq!(actual_input, input);
    }
}
