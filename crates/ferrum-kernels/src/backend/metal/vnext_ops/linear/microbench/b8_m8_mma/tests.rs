use super::*;

#[test]
fn b8_m8_mma_shared_tiles_and_output_tails_have_unique_coverage() {
    let mut weights = vec![0; 64 * 32];
    let mut inputs = vec![0; 8 * 32];
    for tid in 0..128 {
        for i in 0..16 {
            let tile_x = 2 * (tid % 2) + i / 8;
            let tile_y = (tid / 2) / 8;
            let local_x = (tid / 2) % 8;
            weights[64 * (8 * tile_x + tile_y) + 8 * (i % 8) + local_x] += 1;
        }
        if tid < 32 {
            for i in 0..8 {
                inputs[64 * (tid % 4) + 8 * (tid / 4) + i] += 1;
            }
        }
    }
    assert!(weights.iter().all(|&count| count == 1));
    assert!(inputs.iter().all(|&count| count == 1));
    // Four K8 reads of the exact eight-token fragment, two N8 fragments per SG.
    let mut outputs = vec![0; 8 * 64];
    for sg in 0..4 {
        for fragment in 0..2 {
            for row in 0..8 {
                for col in 0..8 {
                    outputs[row * 64 + 16 * sg + 8 * fragment + col] += 1;
                }
            }
        }
    }
    assert!(outputs.iter().all(|&count| count == 1));
    assert_eq!(4096 + inputs.len() * 2, 4608);
    assert!(outputs.len() * 4 <= 8192);
    for (rows, columns) in [(1_usize, 1_usize), (7, 63), (8, 64), (9, 65)] {
        let stride = columns + 9;
        let mut stores = vec![0; rows * stride];
        for by in 0..rows.div_ceil(8) {
            for bx in 0..columns.div_ceil(64) {
                for tid in 0..128 {
                    for index in (tid..8 * 64).step_by(128) {
                        let row = by * 8 + index / 64;
                        let col = bx * 64 + index % 64;
                        if row < rows && col < columns {
                            stores[row * stride + 4 + col] += 1;
                        }
                    }
                }
            }
        }
        for (index, count) in stores.into_iter().enumerate() {
            assert_eq!(
                count,
                usize::from((4..4 + columns).contains(&(index % stride)))
            );
        }
    }
}

#[test]
fn b8_m8_mma_matches_original_mma_at_offsets_tails_and_k_blocks() {
    let device = Device::system_default().expect("M8 mapping requires Metal");
    let queue = device.new_command_queue();
    let pipelines = pipelines(&device);
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
    ] {
        for (rows, input_width, width) in [
            (1, 256, 1),
            (7, 512, 65),
            (8, 256, 1),
            (8, 256, 63),
            (8, 512, 64),
            (8, 512, 65),
            (9, 512, 129),
        ] {
            let shape = Shape {
                name: "m8_mma_tail",
                input: input_width as u32,
                output: width as u32,
                format,
            };
            let projection = Projection::new(&device, shape, 1);
            let values: Vec<_> = (0..rows * input_width)
                .map(|index| {
                    let phase = (index + 1) as f32 * 0.017 + (index / input_width) as f32 * 0.113;
                    f16::from_f32(phase.sin() * 0.03125 + (phase * 0.37).cos() * 0.0078125)
                })
                .collect();
            let input = Halves::new(&device, &values);
            let stride = width + 9;
            let output: [Halves; 2] =
                std::array::from_fn(|_| Halves::new(&device, &vec![HALF_GUARD; rows * stride]));
            let params = LinearParams {
                rows: rows as u32,
                in_features: input_width as u32,
                out_features: width as u32,
                output_stride: stride as u32,
                output_column_offset: 4,
            };
            let original = match format {
                GgufBlockFormat::Q4K => &pipelines.production.k_quant_gemm.q4_k,
                GgufBlockFormat::Q5K => &pipelines.production.k_quant_gemm.q5_k,
                GgufBlockFormat::Q6K => &pipelines.production.k_quant_gemm.q6_k,
                _ => unreachable!(),
            };
            for (index, arm) in [Arm::ProductionMma, Arm::M8Mma].into_iter().enumerate() {
                run(&queue, arm, 1, 1, |encoder| {
                    encoder.set_compute_pipeline_state(if arm == Arm::M8Mma {
                        pipelines.candidate(format).unwrap()
                    } else {
                        original
                    });
                    encoder.set_buffer(0, Some(&input.buffer), (PREFIX * 2) as u64);
                    encoder.set_buffer(1, Some(&projection.weight), WEIGHT_PREFIX as u64);
                    encoder.set_buffer(2, Some(&output[index].buffer), (PREFIX * 2) as u64);
                    bind_linear_params(encoder, params, physical(format), ElementType::F16);
                    if arm == Arm::M8Mma {
                        encoder.set_threadgroup_memory_length(0, 8192);
                        encoder.dispatch_thread_groups(
                            MTLSize::new((rows as u64).div_ceil(8), (width as u64).div_ceil(64), 1),
                            MTLSize::new(128, 1, 1),
                        );
                    } else {
                        dispatch_linear_grid(encoder, params, LinearDispatchKind::TiledGemm);
                    }
                });
            }
            assert_finite_bits(
                output[1].values(),
                output[0].values(),
                "strided original MMA parity",
            );
            if rows == ROWS {
                let selected = Halves::new(&device, &vec![HALF_GUARD; rows * stride]);
                run(&queue, Arm::M8Mma, 1, 1, |encoder| {
                    let (pipeline, kind) = pipelines.production.plain_linear_dispatch(
                        physical(format),
                        ElementType::F16,
                        params,
                    );
                    assert_eq!(kind, LinearDispatchKind::TiledGemmM8);
                    encoder.set_compute_pipeline_state(pipeline);
                    encoder.set_buffer(0, Some(&input.buffer), (PREFIX * 2) as u64);
                    encoder.set_buffer(1, Some(&projection.weight), WEIGHT_PREFIX as u64);
                    encoder.set_buffer(2, Some(&selected.buffer), (PREFIX * 2) as u64);
                    bind_linear_params(encoder, params, physical(format), ElementType::F16);
                    dispatch_linear_grid(encoder, params, kind);
                });
                assert_finite_bits(
                    selected.values(),
                    output[0].values(),
                    "product selector and grid preserve original MMA and guards",
                );
            }
            for all in &output {
                for row in all.values().chunks_exact(stride) {
                    assert!(row[..4]
                        .iter()
                        .chain(&row[4 + width..])
                        .all(|v| v.to_bits() == HALF_GUARD.to_bits()));
                }
            }
            if rows == ROWS {
                let expected =
                    super::super::b8_two_b4::oracle::project(&projection, &values, false);
                let actual: Vec<_> = output[1]
                    .values()
                    .chunks_exact(stride)
                    .flat_map(|row| row[4..4 + width].iter().copied())
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({"kind":"b8_m8_mma_mapping","format":format.format_id(),"shape":[rows,input_width,width],
                    "bitwise_equal_to_original_mma":true,"raw_coefficient_f64":metrics(&actual,&expected,width),"absolute_numerical_quality_claimed":false})
                );
            }
            exact_bits(input.values(), &values);
            projection.immutable();
        }
    }
}

#[test]
fn b8_m8_mma_product_route_retains_neighbor_rows_q8_and_f32_paths() {
    let device = Device::system_default().expect("M8 product routes require Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    for format in [
        LinearPhysicalFormat::Q4K,
        LinearPhysicalFormat::Q5K,
        LinearPhysicalFormat::Q6K,
        LinearPhysicalFormat::Q8_0,
    ] {
        for rows in [1, 4, 7, 8, 9, 16, 32] {
            let params = LinearParams {
                rows,
                in_features: 256,
                out_features: 65,
                output_stride: 73,
                output_column_offset: 3,
            };
            let kind = pipelines
                .plain_linear_dispatch(format, ElementType::F16, params)
                .1;
            assert_eq!(
                kind == LinearDispatchKind::TiledGemmM8,
                rows == 8 && format != LinearPhysicalFormat::Q8_0,
                "{format:?} rows={rows}",
            );
            if rows != 8 || format == LinearPhysicalFormat::Q8_0 {
                assert_eq!(
                    kind,
                    if rows < 8 {
                        LinearDispatchKind::CooperativeGemv
                    } else {
                        LinearDispatchKind::TiledGemm
                    }
                );
            }
            assert_ne!(
                pipelines
                    .plain_linear_dispatch(format, ElementType::F32, params)
                    .1,
                LinearDispatchKind::TiledGemmM8,
                "F32 input must not be bound to an F16 M8 kernel",
            );
        }
    }
}
