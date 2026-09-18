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

fn generic_pipelines(pipelines: &MetalHadamardPipelines) -> MetalHadamardPipelines {
    MetalHadamardPipelines {
        f16_f32: pipelines.f16_f32.clone(),
        f32_f32: pipelines.f32_f32.clone(),
        f32_f16: pipelines.f32_f16.clone(),
        specialized_1024: None,
        maximum_threadgroup_bytes: pipelines.maximum_threadgroup_bytes,
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
    let generic = generic_pipelines(pipelines);
    for path in [pipelines, &generic] {
        assert_transform_path(
            device,
            path,
            transform,
            input,
            rows,
            width,
            signs,
            input_type,
            output_type,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_transform_path(
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
fn hadamard_1024_selection_requires_exact_simd_threads_and_shared_capacity() {
    assert!(supports_hadamard_1024(32, 256, 0, 256, 4096));
    assert!(supports_hadamard_1024(32, 512, 128, 1024, 4224));
    for (simd, threads, static_bytes, device_threads, device_bytes) in [
        (16, 256, 0, 256, 4096),
        (64, 256, 0, 256, 4096),
        (32, 255, 0, 256, 4096),
        (32, 256, 0, 255, 4096),
        (32, 256, 0, 256, 4095),
        (32, 256, 128, 256, 4223),
        (32, 256, u64::MAX, 256, u64::MAX),
    ] {
        assert!(!supports_hadamard_1024(
            simd,
            threads,
            static_bytes,
            device_threads,
            device_bytes,
        ));
    }
}

#[test]
fn hadamard_1024_production_selection_and_generic_fallback_preserve_transforms() {
    let device = Device::system_default().expect("Hadamard selection requires Metal");
    let pipelines = MetalHadamardPipelines::new(&device).unwrap();
    let specialized = pipelines
        .specialized_1024
        .as_ref()
        .expect("Hadamard 1024 conformance requires supported specialized pipelines");
    let generic = generic_pipelines(&pipelines);
    for (input_type, output_type, selected) in [
        (ElementType::F16, ElementType::F32, &specialized.f16_f32),
        (ElementType::F32, ElementType::F32, &specialized.f32_f32),
        (ElementType::F32, ElementType::F16, &specialized.f32_f16),
    ] {
        for block_size in [64, 256, 1024] {
            let transform = HadamardTransform {
                block_size,
                signs_region: None,
                inverse: output_type == ElementType::F16,
                permutation: None,
            };
            let original = pipelines.pipeline(input_type, output_type).unwrap();
            assert!(std::ptr::eq(
                pipelines
                    .selected_pipeline(transform, input_type, output_type)
                    .unwrap(),
                if block_size == 1024 {
                    selected
                } else {
                    original
                },
            ));
            assert!(std::ptr::eq(
                generic
                    .selected_pipeline(transform, input_type, output_type)
                    .unwrap(),
                generic.pipeline(input_type, output_type).unwrap(),
            ));
            // The same production dispatch_raw must actually execute generic
            // when the optional PSOs are absent, and for other block sizes.
            let width = block_size as usize * 2;
            let input = (0..2 * width)
                .map(|i| ((i * 7 % 31) as f32 - 15.0) / 32.0)
                .collect::<Vec<_>>();
            assert_transform_path(
                &device,
                &generic,
                transform,
                &input,
                2,
                width,
                None,
                input_type,
                output_type,
            );
            assert_transform_path(
                &device,
                &pipelines,
                transform,
                &input,
                2,
                width,
                None,
                input_type,
                output_type,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn raw_transform_bits(
    device: &Device,
    pipelines: &MetalHadamardPipelines,
    transform: HadamardTransform,
    input: &[f32],
    rows: usize,
    width: usize,
    signs: Option<&[f32]>,
    input_type: ElementType,
    output_type: ElementType,
) -> Vec<u32> {
    pipelines
        .validate_dispatch(transform, width as u32, input_type, output_type)
        .unwrap();
    assert_eq!(input.len(), rows * width);
    let stride = width + 8;
    let input_prefix = 16 / input_type.size_bytes() as usize;
    let output_prefix = 16 / output_type.size_bytes() as usize;
    let mut padded_input = vec![-123.0; input_prefix + rows * stride + 8];
    for row in 0..rows {
        padded_input[input_prefix + row * stride..input_prefix + row * stride + width]
            .copy_from_slice(&input[row * width..(row + 1) * width]);
    }
    let input_half = padded_input
        .iter()
        .copied()
        .map(f16::from_f32)
        .collect::<Vec<_>>();
    let input_buffer = if input_type == ElementType::F16 {
        buffer(device, &input_half)
    } else {
        buffer(device, &padded_input)
    };
    let mut padded_signs = vec![f32::NAN; 4];
    if let Some(signs) = signs {
        assert_eq!(signs.len(), width);
        padded_signs.extend_from_slice(signs);
    }
    padded_signs.extend([f32::NAN; 4]);
    let signs_buffer = buffer(device, &padded_signs);
    let output_len = output_prefix + rows * stride + 8;
    let output = if output_type == ElementType::F16 {
        buffer(device, &vec![f16::from_f32(-123.0); output_len])
    } else {
        buffer(device, &vec![-123.0_f32; output_len])
    };
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
        signs.map(|_| (signs_buffer.as_ref(), 16)),
        rows as u32,
        width as u32,
        stride as u32,
        stride as u32,
    );
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: every shared allocation has the recorded scalar layout/length,
    // and the owning command has completed before these readbacks.
    unsafe {
        if input_type == ElementType::F16 {
            let actual =
                std::slice::from_raw_parts(input_buffer.contents().cast::<f16>(), input_half.len());
            assert!(actual
                .iter()
                .zip(&input_half)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        } else {
            let actual = std::slice::from_raw_parts(
                input_buffer.contents().cast::<f32>(),
                padded_input.len(),
            );
            assert!(actual
                .iter()
                .zip(&padded_input)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        let actual_signs =
            std::slice::from_raw_parts(signs_buffer.contents().cast::<f32>(), padded_signs.len());
        assert!(actual_signs
            .iter()
            .zip(&padded_signs)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    let bits = if output_type == ElementType::F16 {
        // SAFETY: the completed output allocation holds output_len F16 values.
        unsafe { std::slice::from_raw_parts(output.contents().cast::<f16>(), output_len) }
            .iter()
            .map(|value| u32::from(value.to_bits()))
            .collect::<Vec<_>>()
    } else {
        // SAFETY: the completed output allocation holds output_len F32 values.
        unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), output_len) }
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let guard = if output_type == ElementType::F16 {
        u32::from(f16::from_f32(-123.0).to_bits())
    } else {
        (-123.0_f32).to_bits()
    };
    for (index, &actual) in bits.iter().enumerate() {
        let is_output = index
            .checked_sub(output_prefix)
            .is_some_and(|index| index < rows * stride && index % stride < width);
        if !is_output {
            assert_eq!(actual, guard, "output guard at {index}");
        }
    }
    bits
}

#[test]
fn hadamard_1024_shuffle_preserves_generic_f32_order_special_values_and_signed_zero() {
    let device = Device::system_default().expect("Hadamard bitwise conformance requires Metal");
    let pipelines = MetalHadamardPipelines::new(&device).unwrap();
    assert!(
        pipelines.specialized_1024.is_some(),
        "specialized path must really be exercised"
    );
    let generic = generic_pipelines(&pipelines);
    let width = 6144;
    let rows = 2;
    let signs = (0..width)
        .map(|i| {
            if (i / 1024 + i % 7) % 2 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect::<Vec<_>>();
    let arbitrary = (0..rows * width)
        .map(|i| {
            let seed = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
            f32::from_bits((seed & 0x807f_ffff) | ((112 + i as u32 % 27) << 23))
        })
        .collect::<Vec<_>>();
    let mut extremes = vec![0.0; rows * width];
    for block in extremes.chunks_mut(1024) {
        block[0] = f32::MAX;
        block[1] = f32::MAX;
        block[2] = -f32::MAX;
        block[31] = f32::from_bits(0x0080_0001);
        block[32] = f32::from_bits(1);
        block[255] = -f32::from_bits(1);
        block[256] = 131008.0;
        block[512] = -65504.0;
    }
    let mut nonfinite = arbitrary.clone();
    for (index, block) in nonfinite.chunks_mut(1024).enumerate() {
        block[0] = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(0x7fc0_0123),
        ][index % 3];
        block[511] = [
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::from_bits(0xffc0_0456),
        ][index % 3];
    }
    let subnormals = (0..rows * width)
        .map(|i| {
            f32::from_bits(
                [
                    1,
                    0x8000_0001,
                    0x007f_ffff,
                    0x807f_ffff,
                    0x0080_0000,
                    0x8080_0001,
                ][i % 6],
            )
        })
        .collect::<Vec<_>>();
    for (name, input) in [
        ("arbitrary-f32", arbitrary),
        ("intermediate-overflow-and-subnormals", extremes),
        ("subnormal-normal-boundary", subnormals),
        ("nan-and-infinity", nonfinite),
        ("positive-zero", vec![0.0; rows * width]),
        ("negative-zero", vec![-0.0; rows * width]),
    ] {
        for (input_type, output_type, inverse, permutation) in [
            (ElementType::F16, ElementType::F32, false, None),
            (ElementType::F32, ElementType::F32, false, None),
            (
                ElementType::F32,
                ElementType::F32,
                false,
                Some(GroupedFeatureTranspose {
                    inner_extent: 128,
                    first_outer_extent: 16,
                    second_outer_extent: 3,
                }),
            ),
            (ElementType::F32, ElementType::F32, true, None),
            (ElementType::F32, ElementType::F16, true, None),
        ] {
            for explicit in [false, true] {
                let transform = HadamardTransform {
                    block_size: 1024,
                    signs_region: explicit.then_some(0),
                    inverse,
                    permutation,
                };
                let signs = explicit.then_some(signs.as_slice());
                let reference = raw_transform_bits(
                    &device,
                    &generic,
                    transform,
                    &input,
                    rows,
                    width,
                    signs,
                    input_type,
                    output_type,
                );
                let actual = raw_transform_bits(
                    &device,
                    &pipelines,
                    transform,
                    &input,
                    rows,
                    width,
                    signs,
                    input_type,
                    output_type,
                );
                for (index, (&actual, &reference)) in actual.iter().zip(&reference).enumerate() {
                    let value = |bits| {
                        if output_type == ElementType::F16 {
                            f16::from_bits(bits as u16).to_f32()
                        } else {
                            f32::from_bits(bits)
                        }
                    };
                    // NaN payload propagation is not a portable Metal promise.
                    // Finite values, infinities and signed zeros remain bitwise.
                    if value(reference).is_nan() {
                        assert!(
                            value(actual).is_nan(),
                            "{name} {transform:?} NaN at {index}"
                        );
                    } else {
                        assert_eq!(
                            actual, reference,
                            "{name} {transform:?} {input_type:?}->{output_type:?} [{index}]"
                        );
                    }
                }
            }
        }
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

struct ForwardTimingFixture {
    width: u32,
    transform: HadamardTransform,
    input: Buffer,
    signs: Buffer,
    output: Buffer,
    input_values: Vec<f16>,
    sign_values: Vec<f32>,
    initial_output: Vec<f32>,
    reference: Vec<f32>,
}

impl ForwardTimingFixture {
    fn new(device: &Device, width: u32, permutation: Option<GroupedFeatureTranspose>) -> Self {
        const GUARD: f32 = -123.0;
        let transform = HadamardTransform {
            block_size: 1024,
            signs_region: Some(0),
            inverse: false,
            permutation,
        };
        let values = (0..width)
            .map(|i| f16::from_f32(((i * 19 % 4093) as f32 - 2046.0) / 1024.0))
            .collect::<Vec<_>>();
        // Whole-width signs vary across block boundaries; the GDN case uses
        // the same non-symmetric 128 x 16 x 3 transpose as the product layout.
        // Sign values are deterministic fixtures, not model-weight captures.
        let signs = (0..width)
            .map(|i| {
                if (i / 1024 + i % 11) % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect::<Vec<_>>();
        let reference = crate::hadamard::apply_reference(
            &values
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>(),
            1,
            width as usize,
            &specification(transform),
            Some(&signs),
        )
        .unwrap();
        let mut input_values = vec![f16::from_f32(GUARD); 8];
        input_values.extend(values);
        input_values.extend([f16::from_f32(GUARD); 8]);
        let mut sign_values = vec![f32::NAN; 4];
        sign_values.extend(signs);
        sign_values.extend([f32::NAN; 4]);
        let mut initial_output = vec![GUARD; width as usize + 8];
        initial_output[4..4 + width as usize].fill(f32::NAN);
        Self {
            width,
            transform,
            input: buffer(device, &input_values),
            signs: buffer(device, &sign_values),
            output: buffer(device, &initial_output),
            input_values,
            sign_values,
            initial_output,
            reference,
        }
    }

    fn run(
        &self,
        pipelines: &MetalHadamardPipelines,
        queue: &metal::CommandQueueRef,
        dispatches: usize,
    ) -> (f64, f64) {
        assert!(dispatches > 0);
        // SAFETY: this shared F32 allocation has the recorded length and the
        // preceding command, if any, completed before this reset.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.initial_output.as_ptr(),
                self.output.contents().cast::<f32>(),
                self.initial_output.len(),
            );
        }
        let started = std::time::Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        // Use the production dispatch implementation, with all transforms in
        // one encoder as in ordinary inference. Every dispatch rereads the
        // same F16 input; the F32 output is scratch, not the next input.
        for _ in 0..dispatches {
            pipelines.dispatch_raw(
                encoder,
                self.transform,
                &self.input,
                16,
                ElementType::F16,
                &self.output,
                16,
                ElementType::F32,
                Some((&self.signs, 16)),
                1,
                self.width,
                self.width,
                self.width,
            );
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let wall_ns = started.elapsed().as_secs_f64() * 1e9;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let gpu_ns = forward_gpu_elapsed_ns(command)
            .expect("completed Hadamard timing command must expose valid GPU timestamps");
        (gpu_ns, wall_ns)
    }

    fn validate(&self) {
        // SAFETY: all three shared allocations retain these lengths/types,
        // and run waits for completion before any readback.
        let (input, signs, output) = unsafe {
            (
                std::slice::from_raw_parts(
                    self.input.contents().cast::<f16>(),
                    self.input_values.len(),
                ),
                std::slice::from_raw_parts(
                    self.signs.contents().cast::<f32>(),
                    self.sign_values.len(),
                ),
                std::slice::from_raw_parts(
                    self.output.contents().cast::<f32>(),
                    self.initial_output.len(),
                ),
            )
        };
        assert!(input
            .iter()
            .zip(&self.input_values)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(signs
            .iter()
            .zip(&self.sign_values)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        for (index, &actual) in output.iter().enumerate() {
            let expected = index
                .checked_sub(4)
                .and_then(|index| self.reference.get(index))
                .copied()
                .unwrap_or(-123.0);
            // These bounded dyadic F16 fixtures have exact F32 butterflies and
            // power-of-two normalization. Preserve the independent F64/Walsh
            // equality used by the existing transform conformance fixtures.
            assert!(actual.is_finite(), "non-finite Hadamard output at {index}");
            assert_eq!(actual, expected, "Hadamard output/guard at {index}");
        }
    }
}

#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 macros expand their legacy cargo-clippy feature cfg in the calling crate"
)]
fn forward_gpu_elapsed_ns(command: &metal::CommandBufferRef) -> Option<f64> {
    use metal::objc::runtime::{BOOL, YES};
    use metal::objc::{msg_send, sel, sel_impl};
    let start_selector = sel!(GPUStartTime);
    let end_selector = sel!(GPUEndTime);
    // SAFETY: query availability before calling documented double-valued
    // Metal timestamp properties on a completed command buffer.
    let (has_start, has_end): (BOOL, BOOL) = unsafe {
        (
            msg_send![command, respondsToSelector: start_selector],
            msg_send![command, respondsToSelector: end_selector],
        )
    };
    if has_start != YES || has_end != YES {
        return None;
    }
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command, GPUStartTime],
            msg_send![command, GPUEndTime],
        )
    };
    (start.is_finite() && end.is_finite() && start > 0.0 && end > start)
        .then_some((end - start) * 1e9)
}

#[test]
#[ignore = "GPU timing experiment: coordinate exclusive device access"]
fn hadamard_forward_f16_f32_decode_widths_gpu_microbench() {
    let device = Device::system_default().expect("Hadamard timing requires Metal");
    let pipelines = MetalHadamardPipelines::new(&device).unwrap();
    assert!(
        pipelines.specialized_1024.is_some(),
        "timing requires both actual paths"
    );
    let generic = generic_pipelines(&pipelines);
    let paths = [&generic, &pipelines];
    let queue = device.new_command_queue();
    const WARMUP_ROUNDS: usize = 4;
    const MEASURED_ROUNDS: usize = 12;
    for (name, width, permutation) in [
        ("hidden", 5120, None),
        (
            "gdn_grouped_output",
            6144,
            Some(GroupedFeatureTranspose {
                inner_extent: 128,
                first_outer_extent: 16,
                second_outer_extent: 3,
            }),
        ),
        ("ffn_intermediate", 17408, None),
    ] {
        let fixture = ForwardTimingFixture::new(&device, width, permutation);
        pipelines
            .validate_dispatch(fixture.transform, width, ElementType::F16, ElementType::F32)
            .unwrap();
        // Correctness is mandatory before collecting timing samples. Readback,
        // the CPU oracle and guard reset remain outside the measured interval.
        for path in paths {
            for dispatches in [1, 64] {
                let _ = fixture.run(path, &queue, dispatches);
                fixture.validate();
            }
        }
        let mut samples = Vec::new();
        for round in 0..WARMUP_ROUNDS + MEASURED_ROUNDS {
            let order = if round % 2 == 0 { [1, 64] } else { [64, 1] };
            for dispatches in order {
                let path_order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
                let mut times = [(0.0_f64, 0.0_f64); 2];
                for path in path_order {
                    times[path] = fixture.run(paths[path], &queue, dispatches);
                    fixture.validate();
                }
                if round >= WARMUP_ROUNDS {
                    samples.push(serde_json::json!({
                        "round": round - WARMUP_ROUNDS,
                        "single_dispatch_first": order[0] == 1,
                        "specialized_first": path_order[0] == 1,
                        "dispatches_per_command": dispatches,
                        "generic_gpu_ns_per_command": times[0].0,
                        "specialized_gpu_ns_per_command": times[1].0,
                        "generic_gpu_ns_per_dispatch": times[0].0 / dispatches as f64,
                        "specialized_gpu_ns_per_dispatch": times[1].0 / dispatches as f64,
                        "specialized_over_generic": times[1].0 / times[0].0,
                        "generic_host_encode_submit_wait_ns": times[0].1,
                        "specialized_host_encode_submit_wait_ns": times[1].1,
                    }));
                }
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "schema_version": 2,
                "kind": "hadamard_forward_f16_f32_decode_widths_gpu_microbench",
                "device": device.name(), "shape": name,
                "rows": 1, "width": width, "block_size": 1024,
                "input_dtype": "f16", "intermediate_dtype": "f32", "output_dtype": "f32",
                "signs": "explicit-full-width-synthetic",
                "permutation": permutation.map(|p| serde_json::json!({
                    "inner_extent": p.inner_extent,
                    "first_outer_extent": p.first_outer_extent,
                    "second_outer_extent": p.second_outer_extent,
                })),
                "warmup_rounds": WARMUP_ROUNDS,
                "variant_names": ["generic-shared-butterflies", "production-shuffle-shared-register-1024"],
                "independent_oracle": "f64-walsh-exact-dyadic-fixture",
                "correctness": "all-outputs-finite-and-exact-input-signs-guards-unchanged",
                "working_set": "resident-input-signs-and-output-scratch",
                "scope": "production-transform-isolated-command-timing-not-whole-model-speedup",
                "samples": samples,
            })
        );
    }
}
