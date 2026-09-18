use super::*;
use std::num::NonZeroU32;

use ferrum_interfaces::vnext::{
    GroupedFeatureTranspose as ContractTranspose, HadamardApplication, HadamardSigns,
    HadamardTransformSpec, PhysicalWeightComponentBinding, WeightId,
};
use half::f16;
use metal::{Buffer, MTLCommandBufferStatus, MTLResourceOptions};

fn buffer<T>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn specification(transform: HadamardTransform) -> HadamardTransformSpec {
    HadamardTransformSpec {
        block_size: NonZeroU32::new(transform.block_size).unwrap(),
        signs: if transform.signs_region.is_some() {
            HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
                WeightId::new("weight.hadamard.signs").unwrap(),
            ))
        } else {
            HadamardSigns::Identity
        },
        application: if transform.inverse {
            HadamardApplication::AfterEmbeddingLookup
        } else {
            HadamardApplication::BeforeMatmul {
                input_permutation: transform.permutation.map(|p| ContractTranspose {
                    inner_extent: u64::from(p.inner_extent),
                    first_outer_extent: u64::from(p.first_outer_extent),
                    second_outer_extent: u64::from(p.second_outer_extent),
                }),
            }
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_transform(
    device: &Device,
    pipelines: &MetalHadamardPipelines,
    transform: HadamardTransform,
    input: &[f32],
    rows: usize,
    width: usize,
    signs: Option<&[f32]>,
    input_type: ElementType,
    output_type: ElementType,
) {
    pipelines
        .validate_dispatch(transform, width as u32, input_type, output_type)
        .unwrap();
    assert_eq!(input.len(), rows * width);
    let input = input
        .iter()
        .map(|&value| {
            if input_type == ElementType::F16 {
                f16::from_f32(value).to_f32()
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let reference =
        crate::hadamard::apply_reference(&input, rows, width, &specification(transform), signs)
            .unwrap();
    let input_prefix = 16 / input_type.size_bytes() as usize;
    let output_prefix = 16 / output_type.size_bytes() as usize;
    let stride = width + 8;
    let guard = -123.0_f32;
    let mut padded = vec![guard; input_prefix + rows * stride + 8];
    for row in 0..rows {
        padded[input_prefix + row * stride..input_prefix + row * stride + width]
            .copy_from_slice(&input[row * width..(row + 1) * width]);
    }
    let input_buffer = if input_type == ElementType::F16 {
        buffer(
            device,
            &padded
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        )
    } else {
        buffer(device, &padded)
    };
    let output_elements = output_prefix + rows * stride + 8;
    let output = if output_type == ElementType::F16 {
        buffer(device, &vec![f16::from_f32(guard); output_elements])
    } else {
        buffer(device, &vec![guard; output_elements])
    };
    let signs_buffer = signs.map(|signs| {
        let mut padded = vec![f32::NAN; 4];
        padded.extend_from_slice(signs);
        padded.extend([f32::NAN; 4]);
        buffer(device, &padded)
    });
    let queue = device.new_command_queue();
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    pipelines.dispatch_raw(
        encoder,
        transform,
        &input_buffer,
        16,
        input_type,
        &output,
        16,
        output_type,
        signs_buffer.as_ref().map(|buffer| (buffer.as_ref(), 16)),
        rows as u32,
        width as u32,
        stride as u32,
        stride as u32,
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let actual = if output_type == ElementType::F16 {
        // SAFETY: The shared buffer is F16 aligned and its command completed.
        unsafe { std::slice::from_raw_parts(output.contents().cast::<f16>(), output_elements) }
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>()
    } else {
        // SAFETY: The shared buffer is F32 aligned and its command completed.
        unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), output_elements) }
            .to_vec()
    };
    for (index, &actual) in actual.iter().enumerate() {
        let position = index
            .checked_sub(output_prefix)
            .filter(|&index| index < rows * stride && index % stride < width);
        let expected = position
            .map(|index| reference[index / stride * width + index % stride])
            .unwrap_or(guard);
        let expected = if output_type == ElementType::F16 {
            f16::from_f32(expected).to_f32()
        } else {
            expected
        };
        // These dyadic fixtures keep every F32 butterfly exact. The independent
        // F64/Walsh oracle must therefore match with no empirical tolerance.
        assert_eq!(
            actual, expected,
            "{transform:?} {input_type:?}->{output_type:?} [{index}]"
        );
    }
    // SAFETY: The input remains alive and no command is writing to it.
    if input_type == ElementType::F16 {
        let actual = unsafe {
            std::slice::from_raw_parts(input_buffer.contents().cast::<f16>(), padded.len())
        };
        assert!(actual.iter().zip(&padded).all(|(a, b)| a.to_f32() == *b));
    } else {
        let actual = unsafe {
            std::slice::from_raw_parts(input_buffer.contents().cast::<f32>(), padded.len())
        };
        assert_eq!(actual, padded);
    }
}

#[test]
fn hadamard_forward_preserves_full_width_signs_blocks_and_input_precision_on_metal() {
    let device = Device::system_default().expect("Hadamard conformance requires Metal");
    let pipelines = MetalHadamardPipelines::new(&device).unwrap();
    for (width, explicit) in [(1024, false), (5120, true)] {
        let input = (0..2 * width)
            .map(|i| 1.0 + ((i * 17 % 257) as f32 - 128.0) / 4096.0)
            .collect::<Vec<_>>();
        let signs = (0..width)
            .map(|i| {
                if (i / 1024 + i % 7) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect::<Vec<_>>();
        let transform = HadamardTransform {
            block_size: 1024,
            signs_region: explicit.then_some(0),
            inverse: false,
            permutation: None,
        };
        for input_type in [ElementType::F16, ElementType::F32] {
            assert_transform(
                &device,
                &pipelines,
                transform,
                &input,
                2,
                width,
                explicit.then_some(signs.as_slice()),
                input_type,
                ElementType::F32,
            );
        }
    }
}

#[test]
fn hadamard_grouped_permutation_precedes_signs_across_blocks_on_metal() {
    let device = Device::system_default().expect("Hadamard permutation requires Metal");
    let pipelines = MetalHadamardPipelines::new(&device).unwrap();
    let width = 128 * 16 * 3;
    let input = (0..2 * width)
        .map(|i| ((i * 19 % 4093) as f32 - 2046.0) / 1024.0)
        .collect::<Vec<_>>();
    let signs = (0..width)
        .map(|i| {
            if (i / 1024 + i % 11) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect::<Vec<_>>();
    assert_transform(
        &device,
        &pipelines,
        HadamardTransform {
            block_size: 1024,
            signs_region: Some(0),
            inverse: false,
            permutation: Some(GroupedFeatureTranspose {
                inner_extent: 128,
                first_outer_extent: 16,
                second_outer_extent: 3,
            }),
        },
        &input,
        2,
        width,
        Some(&signs),
        ElementType::F32,
        ElementType::F32,
    );
}

#[test]
fn hadamard_inverse_keeps_wide_embedding_values_f32_until_final_store_on_metal() {
    let device = Device::system_default().expect("Hadamard inverse requires Metal");
    let pipelines = MetalHadamardPipelines::new(&device).unwrap();
    let width = 2048;
    let mut input = vec![0.0; 2 * width];
    for row in 0..2 {
        for block in 0..2 {
            input[row * width + block * 1024] = 131008.0;
            input[row * width + block * 1024 + 7] = if row == block { 32.0 } else { -32.0 };
        }
    }
    let signs = (0..width)
        .map(|i| {
            if (i / 1024 + i % 3) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect::<Vec<_>>();
    let transform = HadamardTransform {
        block_size: 1024,
        signs_region: Some(0),
        inverse: true,
        permutation: None,
    };
    for output_type in [ElementType::F16, ElementType::F32] {
        assert_transform(
            &device,
            &pipelines,
            transform,
            &input,
            2,
            width,
            Some(&signs),
            ElementType::F32,
            output_type,
        );
    }
}
