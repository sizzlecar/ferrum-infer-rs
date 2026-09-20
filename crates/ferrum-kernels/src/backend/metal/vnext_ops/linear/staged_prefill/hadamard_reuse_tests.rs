use super::tests::{bytes, overwrite, region};
use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::DeviceId;
use half::f16;
use metal::MTLCommandBufferStatus;

fn pq2_weights(width: usize, columns: usize, phase: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    for column in 0..columns {
        for block in 0..width / 128 {
            bytes.extend_from_slice(&f16::from_f32(1.0 / 64.0).to_le_bytes());
            for byte in 0..32 {
                bytes.push([0xe4, 0x1b, 0x55, 0xa0][(column + block + byte + phase) % 4]);
            }
        }
    }
    bytes
}

#[test]
fn adjacent_hadamard_reuse_preserves_complete_swiglu_and_fallbacks_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.swiglu.hadamard.reuse").unwrap())
            .unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    const WIDTH: usize = 256;
    const OUTPUTS: usize = 7;
    for (rows, intermediate) in [(3_usize, 17_usize), (33, 1025), (129, 4096)] {
        let guard = f16::from_f32(-123.0);
        let mut input = vec![guard; 8];
        input.extend(
            (0..(rows + 1) * WIDTH).map(|i| f16::from_f32((i as f32 * 0.031).sin() / 32.0)),
        );
        input.extend([guard; 8]);
        let other_input = input.iter().map(|value| -*value).collect::<Vec<_>>();
        let signs = (0..WIDTH)
            .map(|i| if i % 3 == 0 { -1.0_f32 } else { 1.0 })
            .collect::<Vec<_>>();
        let other_signs = signs.iter().map(|value| -*value).collect::<Vec<_>>();
        let gate_bytes = rows * intermediate * 4;
        let activation_offset = align_up_bytes((64 + gate_bytes + 64) as u64, 64).unwrap();
        let transform_offset = align_up_bytes(
            activation_offset + (rows * intermediate * 2) as u64 + 64,
            64,
        )
        .unwrap();
        let transform_bytes = rows * WIDTH.max(intermediate) * 4;
        let scratch = vec![0xa5_u8; transform_offset as usize + transform_bytes + 128];
        let mut output = vec![guard; 8 + rows * OUTPUTS + 8];
        output[8..8 + rows * OUTPUTS].fill(f16::NAN);
        let down_weights = (0..OUTPUTS * intermediate)
            .map(|i| f16::from_f32(if i % 3 == 0 { -0.03125 } else { 0.015625 }))
            .collect::<Vec<_>>();
        let mut regions = vec![
            region(runtime, "reuse.input", &input, ElementType::F16),
            region(
                runtime,
                "reuse.gate",
                &pq2_weights(WIDTH, intermediate, 0),
                ElementType::U8,
            ),
            region(
                runtime,
                "reuse.up",
                &pq2_weights(WIDTH, intermediate, 1),
                ElementType::U8,
            ),
            region(runtime, "reuse.down", &down_weights, ElementType::F16),
            region(runtime, "reuse.output", &output, ElementType::F16),
            region(runtime, "reuse.scratch", &scratch, ElementType::U8),
            region(runtime, "reuse.signs", &signs, ElementType::F32),
            region(runtime, "reuse.other_signs", &other_signs, ElementType::F32),
            region(runtime, "reuse.other_input", &other_input, ElementType::F16),
        ];
        regions.push(regions[6].clone());
        regions.push(regions[0].clone());
        let source_indices = [0, 1, 2, 3, 6, 7, 8];
        let source_before = source_indices.map(|index| bytes(&regions[index]));
        let transform = HadamardTransform {
            block_size: 128,
            signs_region: Some(6),
            inverse: false,
            permutation: None,
        };
        let mut gate = linear_launch(
            PreparedLinearPart {
                region: 1,
                format: LinearPhysicalFormat::Native(GgufBlockFormat::Pq2_0),
                output_offset: 0,
                out_features: intermediate as u32,
                transform: Some(transform),
            },
            0,
            5,
            rows as u64,
            WIDTH as u64,
            (2 * intermediate) as u64,
            16,
            64,
        )
        .unwrap();
        gate.bind_hadamard_workspace(&pipelines, &regions, 5, transform_offset)
            .unwrap();
        assert_eq!(gate.dispatch_count(), if rows == 129 { 3 } else { 2 });
        let mut up = gate;
        up.weight_region = 2;
        up.params.output_column_offset = intermediate as u32;
        let mut down = linear_launch(
            PreparedLinearPart {
                region: 3,
                format: LinearPhysicalFormat::DenseF16,
                output_offset: 0,
                out_features: OUTPUTS as u32,
                transform: Some(HadamardTransform {
                    block_size: 1,
                    signs_region: None,
                    inverse: false,
                    permutation: None,
                }),
            },
            5,
            4,
            rows as u64,
            intermediate as u64,
            OUTPUTS as u64,
            activation_offset,
            16,
        )
        .unwrap();
        down.bind_hadamard_workspace(&pipelines, &regions, 5, transform_offset)
            .unwrap();
        for case in [
            "same",
            "physical_alias",
            "signs",
            "no_signs",
            "input",
            "offset",
            "rows",
            "width",
            "block",
            "permutation",
            "workspace",
            "plain",
        ] {
            let mut second = up;
            match case {
                "physical_alias" => {
                    second.input_region = 10;
                    second.transform.as_mut().unwrap().signs_region = Some(9);
                }
                "signs" => second.transform.as_mut().unwrap().signs_region = Some(7),
                "no_signs" => second.transform.as_mut().unwrap().signs_region = None,
                "input" => second.input_region = 8,
                "offset" => second.input_offset_bytes += 16,
                "rows" => second.params.rows -= 1,
                "width" => second.params.in_features /= 2,
                "block" => second.transform.as_mut().unwrap().block_size = 64,
                "permutation" => {
                    second.transform.as_mut().unwrap().permutation =
                        Some(hadamard::GroupedFeatureTranspose {
                            inner_extent: 16,
                            first_outer_extent: 4,
                            second_outer_extent: 4,
                        })
                }
                "workspace" => second.transform_workspace = Some((5, transform_offset + 64)),
                "plain" => second.transform = None,
                _ => {}
            }
            if let Some((region, offset)) = second.transform_workspace {
                second
                    .bind_hadamard_workspace(&pipelines, &regions, region, offset)
                    .unwrap();
            }
            let reuse = matches!(case, "same" | "physical_alias");
            let sequence = Sequence {
                gate_up: vec![gate, second],
                down,
                activation: swiglu_launch(
                    64,
                    activation_offset,
                    rows as u64,
                    intermediate as u64,
                    (2 * intermediate) as u64,
                )
                .unwrap(),
                scratch_region: 5,
                workspace: None,
            };
            validate_launch_regions_with_raw_workspace(&regions, &sequence.gate_up, &[5]).unwrap();
            validate_launch_regions_with_raw_workspace(&regions, &[down], &[5]).unwrap();
            assert_eq!(
                sequence
                    .gate_up_steps(&regions)
                    .map(|(_, reuse)| reuse)
                    .collect::<Vec<_>>(),
                [false, reuse],
                "{case}"
            );
            let unreused_count =
                gate.dispatch_count() + second.dispatch_count() + down.dispatch_count() + 1;
            assert_eq!(
                sequence.dispatch_count(&regions),
                unreused_count - u64::from(reuse)
            );
            let mut reference = None;
            for optimized in [false, true] {
                overwrite(&regions[4], &output);
                overwrite(&regions[5], &scratch);
                let command = queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                if optimized {
                    sequence.encode(&pipelines, &regions, |_, encode| encode(encoder));
                } else {
                    for launch in &sequence.gate_up {
                        dispatch_linear(&pipelines, encoder, &regions, *launch);
                    }
                    dispatch_swiglu(&pipelines, encoder, &regions[5], sequence.activation);
                    dispatch_linear(&pipelines, encoder, &regions, down);
                }
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
                let actual = (bytes(&regions[4]), bytes(&regions[5]));
                for (index, pair) in actual.0.chunks_exact(2).enumerate() {
                    let value = f16::from_le_bytes(pair.try_into().unwrap());
                    if (8..8 + rows * OUTPUTS).contains(&index) {
                        assert!(value.is_finite(), "{case}: output {index}");
                    } else {
                        assert_eq!(value, guard, "{case}: output guard {index}");
                    }
                }
                assert_eq!(&actual.1[..64], &scratch[..64]);
                assert_eq!(
                    &actual.1[actual.1.len() - 64..],
                    &scratch[scratch.len() - 64..]
                );
                if let Some(reference) = &reference {
                    assert_eq!(&actual, reference, "{case}: complete sequence parity");
                } else {
                    reference = Some(actual);
                }
            }
            assert_eq!(
                source_indices.map(|index| bytes(&regions[index])),
                source_before,
                "{case}: immutable sources"
            );
        }
        // Reject destructive aliasing even for otherwise identical transform
        // metadata. These invalid overlap examples are never submitted.
        for protected in [0, 5, 6] {
            let mut overlapping = gate;
            overlapping.output_region = protected;
            overlapping.output_offset_bytes = match protected {
                0 => gate.input_offset_bytes,
                5 => transform_offset,
                _ => 0,
            };
            overlapping.params.output_stride = 1;
            assert!(!can_reuse_hadamard(overlapping, up, &regions));
        }
        let mut different = up;
        different.activation_type = ElementType::F32;
        assert!(!can_reuse_hadamard(gate, different, &regions));
        different = up;
        different.transform.as_mut().unwrap().inverse = true;
        assert!(!can_reuse_hadamard(gate, different, &regions));
        different = gate;
        different.transform = None;
        assert!(!can_reuse_hadamard(different, up, &regions));
        let mut aliased_gate = gate;
        let mut aliased_up = up;
        for launch in [&mut aliased_gate, &mut aliased_up] {
            launch.input_region = 5;
            launch.input_offset_bytes = transform_offset;
        }
        assert!(!can_reuse_hadamard(aliased_gate, aliased_up, &regions));
        aliased_gate = gate;
        aliased_up = up;
        for launch in [&mut aliased_gate, &mut aliased_up] {
            launch.transform.as_mut().unwrap().signs_region = Some(5);
        }
        assert!(!can_reuse_hadamard(aliased_gate, aliased_up, &regions));
    }
}
