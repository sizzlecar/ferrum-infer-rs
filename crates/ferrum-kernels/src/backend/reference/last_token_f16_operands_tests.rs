use super::tests::{buffer, config};
use super::*;
use crate::backend::reference::ReferenceVNextComposition;
use ferrum_interfaces::vnext::{DeviceId, LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_CAPABILITY_ID};

fn q6_uniform_rows(rows: usize, blocks_per_row: usize) -> Vec<u8> {
    let mut weights = vec![0xff; rows * blocks_per_row * 210];
    for (index, block) in weights.chunks_exact_mut(210).enumerate() {
        // Every code is 63: dequantized value = d * signed_scale * 31.
        // The coefficient requires a second rounding when consumed as F16.
        block[192..208].fill(127);
        let scale = f16::from_f32(
            0.001_001
                + (index / blocks_per_row) as f32 * 0.000_013
                + (index % blocks_per_row) as f32 * 0.000_007,
        );
        block[208..210].copy_from_slice(&scale.to_le_bytes());
    }
    weights
}

#[test]
fn half_operands_head_reference_registers_a_distinct_capability() {
    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.reference.half-head-test").unwrap(),
    )
    .unwrap();
    let capabilities = &composition.runtime().descriptor().capabilities;
    assert!(capabilities
        .iter()
        .any(|id| id.as_str() == LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_CAPABILITY_ID));
    assert!(!capabilities.iter().any(
        |id| id.as_str() == ferrum_interfaces::vnext::LAST_TOKEN_DENSE_LINEAR_F32_CAPABILITY_ID
    ));
}

#[test]
fn half_operands_head_reference_rounds_both_operands_and_retains_f32_logits() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    for (participants, hidden, vocabulary) in
        [(1, 256, 1), (3, 512, 7), (8, 768, 17), (33, 256, 65)]
    {
        let weights = q6_uniform_rows(vocabulary, hidden / 256);
        let weight = buffer(
            &runtime,
            "resource.reference.half-head.weight",
            &weights,
            16,
            ElementType::U8,
        );
        let mut launches = Vec::new();
        let mut outputs = Vec::new();
        for participant in 0..participants {
            let a = 1.000_488_3_f32; // exact half-way tie rounds to even (1.0).
            let b = 0.000_7_f32 + participant as f32 * 0.000_01;
            let c = -0.000_031_7_f32;
            let mut input = vec![0_u8; hidden * 4 + 8];
            input[..4].fill(0xa5);
            input[4..8].copy_from_slice(&a.to_le_bytes());
            input[8..12].copy_from_slice(&b.to_le_bytes());
            input[hidden * 4..hidden * 4 + 4].copy_from_slice(&c.to_le_bytes());
            input[hidden * 4 + 4..].fill(0xa5);
            let input = buffer(
                &runtime,
                "resource.reference.half-head.input",
                &input,
                16,
                ElementType::F32,
            );
            let output = buffer(
                &runtime,
                "resource.reference.half-head.output",
                &vec![0xa5; vocabulary * 4 + 8],
                16,
                ElementType::F32,
            );
            launches.push(
                ReferenceHalfHeadLaunch::new(
                    input.region(4..(hidden * 4 + 4) as u64).unwrap(),
                    weight.region(0..weights.len() as u64).unwrap(),
                    output.region(4..(vocabulary * 4 + 4) as u64).unwrap(),
                    hidden,
                    vocabulary,
                )
                .unwrap(),
            );
            outputs.push((output, a, b, c));
            // The encoded command retains the input allocation after its handle is gone.
            drop(input);
        }
        let command = ReferenceDeviceCommand::half_operands_head(
            launches,
            if participants == 1 {
                DeviceBatchingForm::Scalar
            } else {
                DeviceBatchingForm::ParticipantLoop
            },
            participants as u32,
            participants as u64,
        )
        .unwrap();
        drop(weight);
        assert!(command.validate_runtime(runtime.runtime_instance).is_ok());
        command.execute(&runtime.counters);
        for (output, a, b, c) in outputs {
            let bytes = output
                .region(0..(vocabulary * 4 + 8) as u64)
                .unwrap()
                .read();
            assert_eq!(&bytes[..4], &[0xa5; 4]);
            assert_eq!(&bytes[vocabulary * 4 + 4..], &[0xa5; 4]);
            for row in 0..vocabulary {
                let scale_offset = row * (hidden / 256) * 210 + 208;
                let scale =
                    f16::from_le_bytes(weights[scale_offset..scale_offset + 2].try_into().unwrap())
                        .to_f32();
                let decoded = (scale * 127.0) * 31.0;
                let rounded = f16::from_f32(decoded).to_f32();
                let final_scale_offset = ((row + 1) * (hidden / 256) - 1) * 210 + 208;
                let final_scale = f16::from_le_bytes(
                    weights[final_scale_offset..final_scale_offset + 2]
                        .try_into()
                        .unwrap(),
                )
                .to_f32();
                let final_decoded = (final_scale * 127.0) * 31.0;
                assert_ne!(
                    decoded.to_bits(),
                    rounded.to_bits(),
                    "fixture must exercise weight rounding"
                );
                let expected = rounded * f16::from_f32(a).to_f32()
                    + rounded * f16::from_f32(b).to_f32()
                    + f16::from_f32(final_decoded).to_f32() * f16::from_f32(c).to_f32();
                let actual =
                    f32::from_le_bytes(bytes[4 + row * 4..8 + row * 4].try_into().unwrap());
                assert_eq!(actual.to_bits(), expected.to_bits());
                assert_ne!(
                    actual.to_bits(),
                    (decoded * a + decoded * b + final_decoded * c).to_bits(),
                    "strict F32 arithmetic must not be substituted"
                );
                assert_ne!(
                    actual.to_bits(),
                    f16::from_f32(actual).to_f32().to_bits(),
                    "logits must retain F32 precision"
                );
            }
        }
        drop(command);
        assert_eq!(runtime.snapshot().live_allocations, 0);
    }
}

#[test]
fn half_operands_head_reference_rejects_wrong_ranges_types_and_runtime() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    let foreign = ReferenceDeviceRuntime::new(config()).unwrap();
    let input = buffer(
        &runtime,
        "resource.reference.half-head.input",
        &[0; 1028],
        16,
        ElementType::F32,
    );
    let weight = buffer(
        &runtime,
        "resource.reference.half-head.weight",
        &[0; 420],
        16,
        ElementType::U8,
    );
    let output = buffer(
        &runtime,
        "resource.reference.half-head.output",
        &[0; 8],
        16,
        ElementType::F32,
    );
    let launch = |input_bytes, weight_bytes, output_bytes, hidden, vocabulary| {
        ReferenceHalfHeadLaunch::new(
            input.region(0..input_bytes).unwrap(),
            weight.region(0..weight_bytes).unwrap(),
            output.region(0..output_bytes).unwrap(),
            hidden,
            vocabulary,
        )
    };
    assert!(launch(1024, 420, 8, 256, 2).is_ok());
    assert!(launch(1028, 420, 8, 257, 2).is_err());
    assert!(launch(1024, 419, 8, 256, 2).is_err());
    assert!(launch(1024, 420, 4, 256, 2).is_err());
    assert!(launch(1024, 420, 8, usize::MAX, 2).is_err());
    let wrong_type = buffer(
        &runtime,
        "resource.reference.half-head.wrong-type",
        &[0; 1024],
        16,
        ElementType::F16,
    );
    assert!(ReferenceHalfHeadLaunch::new(
        wrong_type.region(0..1024).unwrap(),
        weight.region(0..420).unwrap(),
        output.region(0..8).unwrap(),
        256,
        2
    )
    .is_err());
    let foreign_input = buffer(
        &foreign,
        "resource.reference.half-head.foreign",
        &[0; 1024],
        16,
        ElementType::F32,
    );
    let command = ReferenceDeviceCommand::half_operands_head(
        vec![ReferenceHalfHeadLaunch::new(
            foreign_input.region(0..1024).unwrap(),
            weight.region(0..420).unwrap(),
            output.region(0..8).unwrap(),
            256,
            2,
        )
        .unwrap()],
        DeviceBatchingForm::Scalar,
        1,
        1,
    )
    .unwrap();
    assert!(command.validate_runtime(runtime.runtime_instance).is_err());
    assert_eq!(output.region(0..8).unwrap().read(), [0; 8]);
}

#[test]
fn half_operands_head_reference_defines_half_rounding_edges() {
    let runtime = ReferenceDeviceRuntime::new(config()).unwrap();
    // Q6 codes are all 32 (zero), except the first value is 33 (+1).
    let mut coefficients = [0_u8; 210];
    coefficients[0] = 1;
    coefficients[128..192].fill(0xaa);
    coefficients[192..208].fill(1);
    coefficients[208..210].copy_from_slice(&f16::ONE.to_le_bytes());
    let weight = buffer(
        &runtime,
        "resource.reference.half-head.edge-weight",
        &coefficients,
        16,
        ElementType::U8,
    );
    let minimum_half = 2_f32.powi(-24);
    let underflow_tie = 2_f32.powi(-25);
    for (value, expected) in [
        (minimum_half, minimum_half),
        (-minimum_half, -minimum_half),
        (underflow_tie, 0.0),
        (f32::from_bits(underflow_tie.to_bits() + 1), minimum_half),
        (1.000_488_3, 1.0),
        (65_504.0, 65_504.0),
        (65_520.0, f32::INFINITY),
        (-65_520.0, f32::NEG_INFINITY),
        (f32::INFINITY, f32::INFINITY),
        (f32::NAN, f32::NAN),
    ] {
        let mut inputs = [0_u8; 1024];
        inputs[..4].copy_from_slice(&value.to_le_bytes());
        let input = buffer(
            &runtime,
            "resource.reference.half-head.edge-input",
            &inputs,
            16,
            ElementType::F32,
        );
        let output = buffer(
            &runtime,
            "resource.reference.half-head.edge-output",
            &[0; 4],
            16,
            ElementType::F32,
        );
        let launch = ReferenceHalfHeadLaunch::new(
            input.region(0..1024).unwrap(),
            weight.region(0..210).unwrap(),
            output.region(0..4).unwrap(),
            256,
            1,
        )
        .unwrap();
        launch.execute();
        let actual = f32::from_le_bytes(output.region(0..4).unwrap().read().try_into().unwrap());
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual, expected, "F32 input {value:?}");
        }
    }
}
