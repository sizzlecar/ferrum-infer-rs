use super::*;
use crate::backend::metal::vnext_ops::native_blocks::supports_m64_threadgroup;
use crate::gguf_blocks::fixtures::{oracle_blocks, FORMATS};
use half::f16;
use metal::{Buffer, CommandQueueRef, MTLCommandBufferStatus, MTLResourceOptions};

fn buffer<T>(device: &Device, data: &[T]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[test]
fn native_block_linears_preserve_rows_offsets_strides_and_precision_on_real_metal() {
    let device = Device::system_default().expect("native block linear conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    assert_native_linears(&device, &pipelines, &queue, false, 7);
}

#[test]
fn native_shared_linears_preserve_independent_rows_offsets_and_precision() {
    let device = Device::system_default().expect("native shared linears require Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for output_width in [7, 1025] {
        assert_native_linears(&device, &pipelines, &queue, true, output_width);
    }
    for format in [
        GgufBlockFormat::Q3K,
        GgufBlockFormat::Iq3S,
        GgufBlockFormat::Iq4Nl,
        GgufBlockFormat::Iq4Xs,
        GgufBlockFormat::Q5K,
    ] {
        for dtype in [ElementType::F16, ElementType::F32] {
            if format == GgufBlockFormat::Q5K && dtype == ElementType::F16 {
                continue;
            }
            let physical = if format == GgufBlockFormat::Q5K {
                LinearPhysicalFormat::Q5K
            } else {
                LinearPhysicalFormat::Native(format)
            };
            for rows in [0, 1, 2, 3, 4, 5, 7, 8, 31, 32] {
                assert_eq!(
                    pipelines
                        .native
                        .shared_linear(format, rows, dtype)
                        .is_some(),
                    (2..=4).contains(&rows),
                );
                for output_width in [1023, 1024] {
                    let (_, kind) = if dtype == ElementType::F16 {
                        pipelines.linear_pipeline(physical, rows, output_width)
                    } else {
                        pipelines
                            .f32_linear_dispatch(physical, rows, output_width)
                            .unwrap()
                    };
                    let expected = if (2..=4).contains(&rows) && output_width >= 1024 {
                        LinearDispatchKind::SharedWeightGemv
                    } else if dtype == ElementType::F16 && rows >= 32 && output_width >= 1024 {
                        LinearDispatchKind::NativeTiledGemm
                    } else {
                        LinearDispatchKind::CooperativeGemv
                    };
                    assert_eq!(
                        kind, expected,
                        "{format:?} {dtype:?} B{rows} N{output_width}"
                    );
                }
            }
        }
    }
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q5K,
        GgufBlockFormat::Q6K,
        GgufBlockFormat::Q8_0,
    ] {
        assert!(pipelines
            .native
            .shared_linear(format, 3, ElementType::F16)
            .is_none());
    }
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q6K,
        GgufBlockFormat::Q8_0,
    ] {
        assert!(pipelines
            .native
            .shared_linear(format, 3, ElementType::F32)
            .is_none());
    }
}

fn assert_native_linears(
    device: &Device,
    pipelines: &MetalLinearPipelines,
    queue: &CommandQueueRef,
    shared: bool,
    output_width: usize,
) {
    for format in FORMATS {
        let blocks = oracle_blocks(format);
        // Five blocks exercise both 32- and 256-value formats without assuming
        // a multiple-of-four block count in the shared kernel.
        let blocks = if shared {
            blocks
                .chunks_exact(format.block_bytes())
                .cycle()
                .take(5)
                .flat_map(|block| block.iter().copied())
                .collect::<Vec<_>>()
        } else {
            blocks
        };
        let input_width = blocks.len() / format.block_bytes() * format.block_values();
        let output_stride = output_width + 4;
        let mut weight_bytes = vec![0x55_u8; 16];
        let block_rows: Vec<Vec<u8>> = (0..output_width)
            .map(|row| {
                let mut bytes = blocks.clone();
                bytes.rotate_left(
                    (row % (blocks.len() / format.block_bytes())) * format.block_bytes(),
                );
                bytes
            })
            .collect();
        let weights: Vec<Vec<f32>> = block_rows
            .iter()
            .map(|bytes| {
                let mut values = vec![0.0; input_width];
                format.decode(bytes, &mut values).unwrap();
                weight_bytes.extend_from_slice(bytes);
                values
            })
            .collect();
        let weight = buffer(device, &weight_bytes);
        for rows in if shared { [2, 3, 4] } else { [1, 3, 9] } {
            for activation_type in [ElementType::F16, ElementType::F32] {
                if shared
                    && pipelines
                        .native
                        .shared_linear(format, rows as u32, activation_type)
                        .is_none()
                {
                    continue;
                }
                // F32 inputs deliberately contain values not representable
                // in F16, so an accidental narrowing changes this oracle.
                let input: Vec<f32> = (0..rows * input_width)
                    .map(|i| {
                        let value = (i as f32 * 0.071).sin() * 0.03125
                            + (i / input_width) as f32 * 0.0012345;
                        if activation_type == ElementType::F16 {
                            f16::from_f32(value).to_f32()
                        } else {
                            value
                        }
                    })
                    .collect();
                let element_bytes = activation_type.size_bytes() as usize;
                let prefix = 16 / element_bytes;
                let mut padded_input = vec![123.0; prefix];
                padded_input.extend_from_slice(&input);
                let input_buffer = if activation_type == ElementType::F16 {
                    buffer(
                        device,
                        &padded_input
                            .iter()
                            .copied()
                            .map(f16::from_f32)
                            .collect::<Vec<_>>(),
                    )
                } else {
                    buffer(device, &padded_input)
                };
                let total = prefix + rows * output_stride + prefix;
                let output = if activation_type == ElementType::F16 {
                    buffer(device, &vec![f16::from_f32(-123.0); total])
                } else {
                    buffer(device, &vec![-123.0_f32; total])
                };
                // Q5 F32 exercises the production selector added for its
                // formerly missing master-precision projection kernel.
                let physical =
                    if format == GgufBlockFormat::Q5K && activation_type == ElementType::F32 {
                        LinearPhysicalFormat::Q5K
                    } else {
                        LinearPhysicalFormat::Native(format)
                    };
                let params = LinearParams {
                    rows: rows as u32,
                    in_features: input_width as u32,
                    out_features: output_width as u32,
                    output_stride: output_stride as u32,
                    output_column_offset: 2,
                };
                // Keep the comparison explicitly on the original format-specialized
                // per-row PSO, even when the production selector chooses shared.
                let pipeline = if activation_type == ElementType::F16 {
                    pipelines.native.linear_f16(format)
                } else {
                    pipelines.native.linear_f32(format)
                };
                let kind = LinearDispatchKind::CooperativeGemv;
                let execute = |pipeline: &metal::ComputePipelineState, kind| {
                    let command = queue.new_command_buffer();
                    let encoder = command.new_compute_command_encoder();
                    encoder.set_compute_pipeline_state(pipeline);
                    encoder.set_buffer(0, Some(&input_buffer), 16);
                    encoder.set_buffer(1, Some(&weight), 16);
                    encoder.set_buffer(2, Some(&output), 16);
                    bind_linear_params(encoder, params, physical, activation_type);
                    dispatch_linear_grid(encoder, params, kind);
                    encoder.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                    assert_eq!(
                        command.status(),
                        MTLCommandBufferStatus::Completed,
                        "{format:?} {activation_type:?}"
                    );
                };
                execute(pipeline, kind);
                if shared {
                    // SAFETY: the previous command completed; all buffers are
                    // shared, fully sized, and retained until both runs finish.
                    let bytes = |buffer: &Buffer| unsafe {
                        std::slice::from_raw_parts(
                            buffer.contents().cast::<u8>(),
                            buffer.length() as usize,
                        )
                        .to_vec()
                    };
                    let baseline = bytes(&output);
                    let input_before = bytes(&input_buffer);
                    let weight_before = bytes(&weight);
                    let selected = if activation_type == ElementType::F16 {
                        pipelines.linear_pipeline(physical, params.rows, params.out_features)
                    } else {
                        pipelines
                            .f32_linear_dispatch(physical, params.rows, params.out_features)
                            .unwrap()
                    };
                    assert_eq!(
                        selected.1 == LinearDispatchKind::SharedWeightGemv,
                        output_width >= SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES as usize
                    );
                    let (shared_pipeline, dispatch) =
                        if selected.1 == LinearDispatchKind::SharedWeightGemv {
                            selected
                        } else {
                            // Exercise the output tail directly below the selector's
                            // profitability floor without changing that floor.
                            (
                                pipelines
                                    .native
                                    .shared_linear(format, params.rows, activation_type)
                                    .unwrap(),
                                LinearDispatchKind::SharedWeightGemv,
                            )
                        };
                    // Poison only writable output cells; a missing shared
                    // write cannot pass by inheriting the baseline's result.
                    unsafe {
                        for row in 0..rows {
                            for column in 0..output_width {
                                let index = prefix + row * output_stride + 2 + column;
                                if activation_type == ElementType::F16 {
                                    *output.contents().cast::<f16>().add(index) = f16::NAN;
                                } else {
                                    *output.contents().cast::<f32>().add(index) = f32::NAN;
                                }
                            }
                        }
                    }
                    execute(shared_pipeline, dispatch);
                    assert_eq!(bytes(&output), baseline, "shared kernel changed per-row result/guards: {format:?} {activation_type:?} B{rows}");
                    assert_eq!(bytes(&input_buffer), input_before);
                    assert_eq!(bytes(&weight), weight_before);
                }
                // SAFETY: Shared, correctly sized aligned buffers; the GPU
                // completed all writes and the buffers remain owned here.
                let actual: Vec<f32> = if activation_type == ElementType::F16 {
                    unsafe { std::slice::from_raw_parts(output.contents().cast::<f16>(), total) }
                        .iter()
                        .map(|x| x.to_f32())
                        .collect()
                } else {
                    unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), total) }
                        .to_vec()
                };
                for (index, actual) in actual.into_iter().enumerate() {
                    let logical = index
                        .checked_sub(prefix)
                        .filter(|i| *i < rows * output_stride);
                    let location = logical.and_then(|i| {
                        (2..2 + output_width)
                            .contains(&(i % output_stride))
                            .then(|| (i / output_stride, i % output_stride - 2))
                    });
                    let Some((row, column)) = location else {
                        assert_eq!(actual, -123.0, "modified padding {format:?} at {index}");
                        continue;
                    };
                    let products = input[row * input_width..(row + 1) * input_width]
                        .iter()
                        .zip(&weights[column])
                        .map(|(&x, &w)| f64::from(x) * f64::from(w));
                    let expected = products.clone().sum::<f64>();
                    let absolute_sum = products.map(f64::abs).sum::<f64>();
                    // Each SIMD lane accumulates ceil(K/32) products, then
                    // a 32-lane reduction; include product rounding and the
                    // final F16 boundary where applicable.
                    let operations = input_width.div_ceil(32) + 8;
                    let accumulation = operations as f64 * f64::from(f32::EPSILON) * absolute_sum;
                    let rounding = if activation_type == ElementType::F16 {
                        expected.abs() / 1024.0 + 2_f64.powi(-24)
                    } else {
                        0.0
                    };
                    assert!(actual.is_finite() && (f64::from(actual) - expected).abs() <= accumulation + rounding, "{format:?} {activation_type:?} rows={rows} ({row},{column}): {actual} != {expected}; bound={}", accumulation + rounding);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum PrefillTile {
    Production,
    M32,
    M64,
}

fn assert_native_prefill(
    device: &Device,
    pipelines: &MetalLinearPipelines,
    queue: &CommandQueueRef,
    format: GgufBlockFormat,
    blocks: &[u8],
    rows: usize,
    columns: usize,
    input_value: fn(usize, usize) -> f16,
    tile: PrefillTile,
) -> Vec<u16> {
    let width = blocks.len() / format.block_bytes() * format.block_values();
    let stride = columns + 5;
    let prefix = 8;
    let guard = f16::from_f32(-123.0);
    let mut input = vec![guard; prefix];
    for row in 0..rows {
        input.extend((0..width).map(|column| input_value(row, column)));
    }
    input.extend([guard; 8]);
    let mut encoded = vec![0x55_u8; 16];
    let mut decoded = Vec::with_capacity(columns);
    for column in 0..columns {
        let mut bytes = blocks.to_vec();
        bytes.rotate_left((column % (blocks.len() / format.block_bytes())) * format.block_bytes());
        let mut values = vec![0.0_f32; width];
        format.decode(&bytes, &mut values).unwrap();
        encoded.extend_from_slice(&bytes);
        decoded.push(values);
    }
    encoded.extend([0x55; 16]);
    let mut initial_output = vec![guard; prefix + rows * stride + 8];
    for row in 0..rows {
        initial_output[prefix + row * stride + 2..prefix + row * stride + 2 + columns]
            .fill(f16::NAN);
    }
    let input_buffer = buffer(device, &input);
    let weight_buffer = buffer(device, &encoded);
    let output = buffer(device, &initial_output);
    let params = LinearParams {
        rows: rows as u32,
        in_features: width as u32,
        out_features: columns as u32,
        output_stride: stride as u32,
        output_column_offset: 2,
    };
    let physical = LinearPhysicalFormat::Native(format);
    let (pipeline, kind) = pipelines.linear_pipeline(physical, params.rows, params.out_features);
    let expected = if format == GgufBlockFormat::Iq4Xs
        && rows >= 1024
        && columns >= 1024
        && pipelines.native.gemm_f16_f32_m64.is_some()
    {
        LinearDispatchKind::NativeTiledGemmM64
    } else if rows >= 32 && columns >= 1024 {
        LinearDispatchKind::NativeTiledGemm
    } else {
        LinearDispatchKind::CooperativeGemv
    };
    assert_eq!(kind, expected);
    let (_, f32_kind) = pipelines
        .f32_linear_dispatch(physical, params.rows, params.out_features)
        .unwrap();
    assert!(matches!(f32_kind, LinearDispatchKind::CooperativeGemv));
    // Exercise partial output tiles independently of the performance guard.
    // Wide fixtures below still execute through the actual product selector.
    let (pipeline, kind) = if rows >= 32 && columns < 1024 {
        (
            &pipelines.native.gemm_f16_f32,
            LinearDispatchKind::NativeTiledGemm,
        )
    } else {
        (pipeline, kind)
    };
    let pipeline = match tile {
        PrefillTile::Production => pipeline,
        PrefillTile::M32 => &pipelines.native.gemm_f16_f32,
        PrefillTile::M64 => pipelines.native.gemm_f16_f32_m64.as_ref().unwrap(),
    };
    let command = queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&input_buffer), 16);
    encoder.set_buffer(1, Some(&weight_buffer), 16);
    encoder.set_buffer(2, Some(&output), 16);
    bind_linear_params(encoder, params, physical, ElementType::F16);
    match tile {
        PrefillTile::Production => dispatch_linear_grid(encoder, params, kind),
        PrefillTile::M32 => {
            dispatch_linear_grid(encoder, params, LinearDispatchKind::NativeTiledGemm)
        }
        PrefillTile::M64 => {
            dispatch_linear_grid(encoder, params, LinearDispatchKind::NativeTiledGemmM64)
        }
    }
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

    // SAFETY: These shared buffers have the declared lengths and scalar types;
    // the completed command can no longer mutate them.
    let actual = unsafe {
        std::slice::from_raw_parts(output.contents().cast::<f16>(), initial_output.len())
    };
    let input_after =
        unsafe { std::slice::from_raw_parts(input_buffer.contents().cast::<f16>(), input.len()) };
    let weight_after =
        unsafe { std::slice::from_raw_parts(weight_buffer.contents().cast::<u8>(), encoded.len()) };
    assert_eq!(input_after, input, "input mutated: {format:?}");
    assert_eq!(weight_after, encoded, "weight mutated: {format:?}");
    for (index, value) in actual.iter().enumerate() {
        let location = index
            .checked_sub(prefix)
            .filter(|index| *index < rows * stride)
            .and_then(|index| {
                (2..columns + 2)
                    .contains(&(index % stride))
                    .then(|| (index / stride, index % stride - 2))
            });
        let Some((row, column)) = location else {
            assert_eq!(
                *value, guard,
                "output padding overwritten: {format:?} at {index}"
            );
            continue;
        };
        let products = input[prefix + row * width..prefix + (row + 1) * width]
            .iter()
            .zip(&decoded[column])
            .map(|(x, w)| f64::from(x.to_f32()) * f64::from(*w));
        let expected = products.clone().sum::<f64>();
        let absolute_sum = products.map(f64::abs).sum::<f64>();
        // Bound a full sequential F32 product/sum chain, which also bounds
        // regrouped SIMD MMA. Decoded weights remain F32: no F16 weight-cast
        // error is permitted. Account separately for the final F16 store.
        let n_u = (2 * width + 8) as f64 * f64::from(f32::EPSILON) / 2.0;
        let accumulation = n_u / (1.0 - n_u) * absolute_sum;
        let rounding = (expected.abs() + accumulation) * 2_f64.powi(-11) + 2_f64.powi(-25);
        assert!(
            value.is_finite()
                && (f64::from(value.to_f32()) - expected).abs() <= accumulation + rounding,
            "{format:?} rows={rows} ({row},{column}): {} != {expected}; bound={}",
            value.to_f32(),
            accumulation + rounding
        );
    }
    actual.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn native_prefill_preserves_tile_tails_block_layouts_and_output_boundaries() {
    let device = Device::system_default().expect("native prefill conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    for format in FORMATS {
        for (rows, columns) in [(31, 7), (32, 63), (33, 65), (65, 67), (33, 1025)] {
            let _ = assert_native_prefill(
                &device,
                &pipelines,
                &queue,
                format,
                &oracle_blocks(format),
                rows,
                columns,
                |row, column| f16::from_f32(((row * 17 + column) as f32 * 0.073).sin() * 0.03125),
                PrefillTile::Production,
            );
        }
    }
}

#[test]
fn native_prefill_m64_matches_m32_with_tile_tails_and_guards() {
    let device = Device::system_default().expect("native M64 conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    pipelines
        .native
        .gemm_f16_f32_m64
        .as_ref()
        .expect("M64 numerical conformance requires a valid M64 PSO");
    let queue = device.new_command_queue();
    for format in [
        GgufBlockFormat::Q3K,
        GgufBlockFormat::Iq3S,
        GgufBlockFormat::Iq4Nl,
        GgufBlockFormat::Iq4Xs,
    ] {
        for (rows, columns) in [(32, 1), (33, 65), (63, 63), (64, 64), (65, 67), (129, 1025)] {
            let run = |tile| {
                assert_native_prefill(
                    &device,
                    &pipelines,
                    &queue,
                    format,
                    &oracle_blocks(format),
                    rows,
                    columns,
                    |row, column| {
                        f16::from_f32(((row * 17 + column) as f32 * 0.073).sin() * 0.03125)
                    },
                    tile,
                )
            };
            let baseline = run(PrefillTile::M32);
            let candidate = run(PrefillTile::M64);
            for (index, (old, new)) in baseline.iter().zip(&candidate).enumerate() {
                assert_eq!(
                    old, new,
                    "{format:?} {rows}x{columns} M32/M64 output[{index}]"
                );
            }
        }
    }
}

#[test]
fn native_prefill_m64_selection_preserves_shape_format_and_capability_fallback() {
    assert!(supports_m64_threadgroup(32, 256, 0, 16384));
    assert!(supports_m64_threadgroup(32, 512, 1024, 17408));
    for (width, threads, static_bytes, maximum_bytes) in [
        (16, 256, 0, 16384),
        (32, 255, 0, 16384),
        (32, 256, 0, 16383),
        (32, 256, 1, 16384),
        (32, 256, u64::MAX, u64::MAX),
    ] {
        assert!(!supports_m64_threadgroup(
            width,
            threads,
            static_bytes,
            maximum_bytes
        ));
    }
    let device = Device::system_default().expect("native prefill selector requires Metal");
    let mut pipelines = MetalLinearPipelines::new(&device).unwrap();
    for format in [
        GgufBlockFormat::Q3K,
        GgufBlockFormat::Iq3S,
        GgufBlockFormat::Iq4Nl,
        GgufBlockFormat::Iq4Xs,
    ] {
        let physical = LinearPhysicalFormat::Native(format);
        for rows in [31, 32, 1023, 1024, 1025] {
            for columns in [1023, 1024, 1025] {
                let expected = if rows < 32 || columns < 1024 {
                    LinearDispatchKind::CooperativeGemv
                } else if format == GgufBlockFormat::Iq4Xs
                    && rows >= 1024
                    && pipelines.native.gemm_f16_f32_m64.is_some()
                {
                    LinearDispatchKind::NativeTiledGemmM64
                } else {
                    LinearDispatchKind::NativeTiledGemm
                };
                assert_eq!(
                    pipelines.linear_pipeline(physical, rows, columns).1,
                    expected
                );
                assert_eq!(
                    pipelines
                        .f32_linear_dispatch(physical, rows, columns)
                        .unwrap()
                        .1,
                    LinearDispatchKind::CooperativeGemv
                );
            }
        }
    }
    // The same production selector must retain M32 if optional registration
    // could not retain a compatible M64 pipeline.
    pipelines.native.gemm_f16_f32_m64 = None;
    assert_eq!(
        pipelines
            .linear_pipeline(
                LinearPhysicalFormat::Native(GgufBlockFormat::Iq4Xs),
                1024,
                1024
            )
            .1,
        LinearDispatchKind::NativeTiledGemm
    );
}

#[test]
fn native_prefill_m64_production_matches_explicit_m32_with_row_and_output_tails() {
    let device =
        Device::system_default().expect("native M64 production conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    pipelines
        .native
        .gemm_f16_f32_m64
        .as_ref()
        .expect("M64 production conformance requires a valid M64 PSO");
    let queue = device.new_command_queue();
    let format = GgufBlockFormat::Iq4Xs;
    let blocks = oracle_blocks(format);
    assert_eq!(
        pipelines
            .linear_pipeline(LinearPhysicalFormat::Native(format), 1025, 1025)
            .1,
        LinearDispatchKind::NativeTiledGemmM64
    );
    let run = |tile| {
        assert_native_prefill(
            &device,
            &pipelines,
            &queue,
            format,
            &blocks[..format.block_bytes()],
            1025,
            1025,
            |row, column| f16::from_f32(((row * 17 + column) as f32 * 0.073).sin() * 0.03125),
            tile,
        )
    };
    let baseline = run(PrefillTile::M32);
    let production = run(PrefillTile::Production);
    for (index, (old, new)) in baseline.iter().zip(&production).enumerate() {
        assert_eq!(old, new, "production M32/M64 output[{index}]");
    }
}

#[test]
fn native_prefill_retains_decoded_weights_beyond_the_f16_range() {
    let device = Device::system_default().expect("native prefill conformance requires Metal");
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let queue = device.new_command_queue();
    let format = GgufBlockFormat::Iq4Nl;
    // Positive/negative coefficients exceed half's finite range, while their
    // products and final output remain finite. A half weight staging route
    // would create infinities and destroy cancellation in this valid input.
    let mut block = vec![0x0f; format.block_bytes()];
    block[..2].copy_from_slice(&f16::from_f32(1024.0).to_le_bytes());
    let mut decoded = vec![0.0; format.block_values()];
    format.decode(&block, &mut decoded).unwrap();
    assert!(decoded.iter().any(|value| value.abs() > f16::MAX.to_f32()));
    let run = |tile| {
        assert_native_prefill(
            &device,
            &pipelines,
            &queue,
            format,
            &block,
            33,
            65,
            |row, _| {
                f16::from_f32(if row % 2 == 0 {
                    1.0 / 1024.0
                } else {
                    -1.0 / 1024.0
                })
            },
            tile,
        )
    };
    let baseline = run(PrefillTile::Production);
    pipelines
        .native
        .gemm_f16_f32_m64
        .as_ref()
        .expect("wide weight conformance requires a valid M64 PSO");
    let candidate = run(PrefillTile::M64);
    for (index, (old, new)) in baseline.iter().zip(&candidate).enumerate() {
        assert_eq!(old, new, "wide decoded weight M32/M64 output[{index}]");
    }
}
