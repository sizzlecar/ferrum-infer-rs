use super::*;
use cudarc::driver::{DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::{GroupedFeatureTranspose, PhysicalWeightComponentBinding, WeightId};
use half::f16;
use std::num::NonZeroU32;

fn spec(block: u32, inverse: bool, permute: bool, width: usize) -> HadamardTransformSpec {
    HadamardTransformSpec {
        block_size: NonZeroU32::new(block).unwrap(),
        signs: HadamardSigns::Explicit(PhysicalWeightComponentBinding::exact_contiguous(
            WeightId::new("component.shared-signs").unwrap(),
        )),
        application: if inverse {
            HadamardApplication::AfterEmbeddingLookup
        } else {
            HadamardApplication::BeforeMatmul {
                input_permutation: permute.then_some(GroupedFeatureTranspose {
                    inner_extent: 8,
                    first_outer_extent: 2,
                    second_outer_extent: (width / 16) as u64,
                }),
            }
        },
    }
}

#[test]
fn hadamard_native_capacity_is_declared_before_launch() {
    for block in [1, 8, 256, 1024, 8192] {
        assert!(validate(&spec(block, false, false, block as usize), u64::from(block)).is_ok());
    }
    assert!(validate(&spec(16384, false, false, 16384), 16384).is_err());
    assert!(validate(&spec(1024, false, false, 1024), 1025).is_err());
    assert!(validate(&spec(1024, false, false, 1024), 0).is_err());
    assert!(validate(&spec(1024, false, false, 1024), u64::from(u32::MAX) + 1).is_err());
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn hadamard_forward_and_inverse_match_independent_walsh_on_cuda() {
    let context = CudaContext::new(0).expect("Hadamard conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    for (block, width, inverse, permute) in [
        (8, 96, false, true),
        (256, 768, false, false),
        (1024, 3072, false, true),
        (1024, 3072, true, false),
    ] {
        let rows = 3;
        let spec = spec(block, inverse, permute, width);
        let input = (0..rows * width)
            .map(|i| ((i * 7 % 113) as f32 - 56.0) / 32.0)
            .collect::<Vec<_>>();
        let signs = (0..width)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let expected =
            crate::hadamard::apply_reference(&input, rows, width, &spec, Some(&signs)).unwrap();
        let x = stream.clone_htod(&input).unwrap();
        let s = stream.clone_htod(&signs).unwrap();
        let mut y = stream
            .clone_htod(&vec![-12345.0_f32; rows * width + 8])
            .unwrap();
        let (xp, _xg) = x.device_ptr(&stream);
        let (sp, _sg) = s.device_ptr(&stream);
        let (yp, yg) = y.device_ptr_mut(&stream);
        kernels
            .hadamard
            .launch(
                &stream,
                xp,
                yp + 16,
                sp,
                rows as u32,
                width as u32,
                ElementType::F32,
                ElementType::F32,
                &spec,
            )
            .unwrap();
        drop(yg);
        let actual = stream.clone_dtoh(&y).unwrap();
        assert!(actual[..4]
            .iter()
            .chain(&actual[actual.len() - 4..])
            .all(|&value| value == -12345.0));
        for (actual, expected) in actual[4..4 + expected.len()].iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2e-5 * expected.abs().max(1.0),
                "{actual} != {expected}"
            );
        }
        assert_eq!(stream.clone_dtoh(&x).unwrap(), input);
        assert_eq!(stream.clone_dtoh(&s).unwrap(), signs);
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn hadamard_pq2_embedding_preserves_f32_until_inverse_on_cuda() {
    use super::super::weights::{MatrixFormat, MatrixPart};
    use crate::gguf_blocks::GgufBlockFormat;
    let context = CudaContext::new(0).expect("Hadamard embedding requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    // The legal fourth PQ2 code with maximum finite F16 scale decodes to
    // 131008. Narrowing the lookup to half before inverse H would produce inf.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&f16::MAX.to_bits().to_le_bytes());
    bytes.extend(std::iter::repeat_n(0xff_u8, 32));
    let mut transform = spec(128, true, false, 128);
    transform.signs = HadamardSigns::Identity;
    let part = MatrixPart {
        component_id: WeightId::new("component.embedding").unwrap(),
        format: MatrixFormat::Block(GgufBlockFormat::Pq2_0),
        rows: 1,
        columns: 128,
        output_offset: 0,
        transform: Some(transform.clone()),
        signs_region: None,
    };
    let tokens = stream.clone_htod(&[0_u32]).unwrap();
    let weight = stream.clone_htod(&bytes).unwrap();
    let mut scratch = stream.alloc_zeros::<f32>(128).unwrap();
    let mut output = stream.clone_htod(&vec![-12345.0_f32; 136]).unwrap();
    let (tp, _tg) = tokens.device_ptr(&stream);
    let (wp, _wg) = weight.device_ptr(&stream);
    let (sp, sg) = scratch.device_ptr_mut(&stream);
    let (yp, yg) = output.device_ptr_mut(&stream);
    kernels
        .transformed_embedding(&stream, tp, wp, yp + 16, &part, 1, ElementType::F32, 0, sp)
        .unwrap();
    drop((sg, yg));
    let expected =
        crate::hadamard::apply_reference(&vec![131008.0; 128], 1, 128, &transform, None).unwrap();
    let actual = stream.clone_dtoh(&output).unwrap();
    assert!(actual[..4]
        .iter()
        .chain(&actual[132..])
        .all(|&value| value == -12345.0));
    for (actual, expected) in actual[4..132].iter().zip(expected) {
        assert!(actual.is_finite() && (actual - expected).abs() <= expected.abs().max(1.0) * 1e-6);
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn hadamard_pq2_projection_mixes_transformed_and_plain_inputs_on_cuda() {
    use super::super::weights::{MatrixFormat, MatrixPart};
    use crate::gguf_blocks::GgufBlockFormat;
    let context = CudaContext::new(0).expect("Hadamard projection requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    let (rows, width, outputs) = (9_usize, 1024_usize, 3_usize);
    let signs = (0..width)
        .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
        .collect::<Vec<_>>();
    let mut input = (0..rows * width)
        .map(|i| {
            f16::from_f32(if i % 2 == 0 {
                2.0 * signs[i % width]
            } else {
                0.0
            })
        })
        .collect::<Vec<_>>();
    for row in 0..rows {
        input[row * width + 1] = f16::from_f32((row + 1) as f32 / 1024.0);
    }
    let source = input.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
    let transform = spec(1024, false, false, width);
    let rotated =
        crate::hadamard::apply_reference(&source, rows, width, &transform, Some(&signs)).unwrap();
    let mut bytes = Vec::new();
    for column in 0..outputs {
        for block in 0..width / 128 {
            bytes.extend_from_slice(&0x3800_u16.to_le_bytes());
            for byte in 0..32 {
                bytes.push(if column == 0 {
                    if block == 0 && byte == 0 {
                        0x58
                    } else {
                        0x55
                    }
                } else {
                    0xe4
                });
            }
        }
    }
    let x = stream.clone_htod(&input).unwrap();
    let w = stream.clone_htod(&bytes).unwrap();
    let s = stream.clone_htod(&signs).unwrap();
    let mut scratch = stream.alloc_zeros::<f32>(rows * width).unwrap();
    let stride = outputs * 2 + 2;
    let canary = f16::from_f32(-123.0);
    let mut y = stream.clone_htod(&vec![canary; rows * stride]).unwrap();
    let (xp, _xg) = x.device_ptr(&stream);
    let (wp, _wg) = w.device_ptr(&stream);
    let (sp, _sg) = s.device_ptr(&stream);
    let (tp, tg) = scratch.device_ptr_mut(&stream);
    let (yp, yg) = y.device_ptr_mut(&stream);
    for (offset, transform) in [(1, Some(transform)), (1 + outputs, None)] {
        let part = MatrixPart {
            component_id: WeightId::new("component.projection").unwrap(),
            format: MatrixFormat::Block(GgufBlockFormat::Pq2_0),
            rows: outputs as u32,
            columns: width as u32,
            output_offset: offset as u32,
            transform,
            signs_region: None,
        };
        kernels
            .transformed_linear(
                &stream,
                xp,
                wp,
                yp,
                &part,
                rows as u32,
                stride as u32,
                ElementType::F16,
                sp,
                tp,
            )
            .unwrap();
    }
    drop((tg, yg));
    let actual = stream.clone_dtoh(&y).unwrap();
    for row in 0..rows {
        assert_eq!(actual[row * stride], canary);
        assert_eq!(actual[row * stride + stride - 1], canary);
        for col in 0..outputs * 2 {
            let input = if col < outputs { &rotated } else { &source };
            let expected = (0..width)
                .map(|i| {
                    let code = if col % outputs == 0 {
                        match i {
                            0 => -1.0,
                            1 => 1.0,
                            _ => 0.0,
                        }
                    } else {
                        (i % 4) as f32 - 1.0
                    };
                    f64::from(input[row * width + i]) * f64::from(code) * 0.5
                })
                .sum::<f64>();
            let actual = actual[row * stride + col + 1].to_f64();
            if col == 0 {
                assert_ne!(expected, 0.0, "fixture must detect premature F16 narrowing");
                assert_eq!(actual, f16::from_f64(expected).to_f64());
            }
            assert!(
                actual.is_finite() && (actual - expected).abs() <= expected.abs().max(1.0) * 0.001
            );
        }
    }
    assert_eq!(stream.clone_dtoh(&x).unwrap(), input);
}
