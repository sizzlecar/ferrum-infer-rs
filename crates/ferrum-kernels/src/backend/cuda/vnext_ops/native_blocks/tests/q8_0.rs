//! Q8_0 format specialization and the independent row-tile timing comparison.

use super::{q4k::Fixture, *};

mod warp_grid;

fn boundary_weights(inputs: usize, outputs: usize) -> Vec<u8> {
    assert_eq!(inputs % 32, 0);
    // Both zero signs, subnormal endpoints, normal endpoints and ordinary
    // positive/negative scales. Literal signed bytes include both int8 limits.
    const SCALES: [u16; 14] = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3000, 0xb300, 0x3c00,
        0xbc00, 0x7bff, 0xfbff,
    ];
    const CODES: [i8; 8] = [i8::MIN, i8::MAX, -127, -1, 0, 1, 63, -64];
    let mut bytes = Vec::with_capacity(outputs * (inputs / 32) * 34);
    for block in 0..outputs * (inputs / 32) {
        let scale = SCALES[block % SCALES.len()];
        bytes.extend_from_slice(&scale.to_le_bytes());
        for lane in 0..32 {
            // Keep the largest finite scales' outputs representable in F16;
            // the remaining scales independently exercise every signed limit.
            let code = if scale & 0x7fff == 0x7bff {
                if lane % 2 == 0 {
                    -1_i8
                } else {
                    1_i8
                }
            } else {
                CODES[(block + lane) % CODES.len()]
            };
            bytes.push(code as u8);
        }
    }
    bytes
}

#[test]
fn q8_0_row_tile_respects_input_and_sm_grid_boundaries() {
    // SM counts are ordinary policy inputs here, not device identities.
    for multiprocessors in [1_u32, 3, 17, 257] {
        assert_eq!(q8_0_f16_row_tile(1, 32, 49, multiprocessors), 1);
        let threshold = 4 * u64::from(multiprocessors);
        for rows in [4_u32, 8, 9, 16, 64] {
            let row_tiles = u64::from(rows).div_ceil(u64::from(LINEAR_ROW_TILE));
            let groups = threshold.div_ceil(row_tiles);
            // First column of the threshold group also checks ceil(N/4).
            let at_threshold = u32::try_from((groups - 1) * 4 + 1).unwrap();
            for outputs in [at_threshold, at_threshold + 4] {
                assert_eq!(
                    q8_0_f16_row_tile(rows, 1024, outputs, multiprocessors),
                    LINEAR_ROW_TILE
                );
            }
            if groups > 1 {
                let below = u32::try_from((groups - 1) * 4).unwrap();
                assert_eq!(q8_0_f16_row_tile(rows, 1024, below, multiprocessors), 1);
                assert_eq!(
                    q8_0_f16_row_tile(rows, 992, below, multiprocessors),
                    LINEAR_ROW_TILE
                );
            }
        }
    }
    assert_eq!(q8_0_f16_row_tile(8, 1024, 4, 0), LINEAR_ROW_TILE);
    assert_eq!(
        q8_0_f16_row_tile(u32::MAX, 1024, u32::MAX, u32::MAX),
        LINEAR_ROW_TILE
    );
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn q8_0_specialization_preserves_generic_bits_and_f64_oracle_on_cuda() {
    use std::collections::BTreeSet;

    let context = CudaContext::new(0).expect("Q8_0 specialization requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    // K32/K96 detect an accidental 256-value block assumption. Partial column
    // groups and row tiles cover the launch boundaries; K5120 is the observed
    // recurrent projection width. The shared fixture supplies odd byte views,
    // nonzero output offsets, padded strides, input/weight guards and F64 dots.
    let mut shapes = BTreeSet::from([
        (1, 32, 17),
        (4, 96, 7),
        (8, 32, 49),
        (9, 96, 47),
        (65, 96, 7),
        (8, 5120, 48),
        (8, 992, 49),
        (8, 1024, 49),
    ]);
    // M9 supplies a partial row tile. At both sides of the K boundary, probe
    // below/at/above the real device's grid threshold, including column tails.
    let boundary_rows = 9_usize;
    let row_tiles = (boundary_rows as u64).div_ceil(u64::from(LINEAR_ROW_TILE));
    let threshold_groups = (4 * u64::from(kernels.multiprocessors)).div_ceil(row_tiles);
    for inputs in [992, 1024] {
        for outputs in [
            (threshold_groups - 1) * 4,
            (threshold_groups - 1) * 4 + 1,
            threshold_groups * 4 + 1,
        ] {
            shapes.insert((
                boundary_rows,
                inputs,
                u32::try_from(outputs).expect("SM-derived conformance width fits launch ABI")
                    as usize,
            ));
        }
    }
    for (rows, inputs, outputs) in shapes {
        let mut fixture = if inputs <= 96 {
            Fixture::<f16>::for_encoded_weights(
                &stream,
                GgufBlockFormat::Q8_0,
                rows,
                inputs,
                outputs,
                true,
                boundary_weights(inputs, outputs),
            )
        } else {
            Fixture::<f16>::for_format(&stream, GgufBlockFormat::Q8_0, rows, inputs, outputs, true)
        };
        fixture.run(&stream, &kernels.linear_f16, 1, 1);
        let reference = fixture.validate(&stream);
        for (kernel, row_tile) in [
            (&kernels.linear_tiled_f16, LINEAR_ROW_TILE),
            (&kernels.linear_q8_0_f16, 1),
            (&kernels.linear_q8_0_tiled_f16, LINEAR_ROW_TILE),
        ] {
            for iterations in [1, 2] {
                fixture.run(&stream, kernel, row_tile, iterations);
                assert_eq!(
                    fixture.validate(&stream),
                    reference,
                    "Q8_0 {rows}x{inputs}x{outputs}, row_tile={row_tile}, iterations={iterations} changed generic bits"
                );
            }
        }
        for _ in 0..2 {
            fixture.run_dispatch(&stream, &kernels);
            assert_eq!(
                fixture.validate(&stream),
                reference,
                "Q8_0 {rows}x{inputs}x{outputs} production dispatch changed bits"
            );
        }
    }
}

#[test]
#[ignore = "paired GPU timing; coordinate exclusive CUDA access"]
fn q8_0_specialization_dispatch_microbench() {
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
    use std::collections::BTreeSet;

    let context = CudaContext::new(0).expect("Q8_0 microbench requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    let multiprocessors = u32::try_from(
        context
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .expect("Q8_0 microbench requires the actual device SM count"),
    )
    .expect("CUDA SM count must be nonnegative");
    let generic_t8_blocks_per_sm = kernels
        .linear_tiled_f16
        .occupancy_max_active_blocks_per_multiprocessor(128, 0, None)
        .expect("Q8_0 microbench requires generic T8 occupancy");
    assert!(multiprocessors > 0 && generic_t8_blocks_per_sm > 0);
    let resident_grid = u64::from(multiprocessors) * u64::from(generic_t8_blocks_per_sm);
    let two_block_grid = u64::from(multiprocessors) * 2;
    let four_block_grid = u64::from(multiprocessors) * 4;
    let mut shapes = BTreeSet::new();
    for rows in [4_usize, 8, 16, 32, 64] {
        shapes.extend([(5120_usize, 48_usize, rows), (5120, 1024, rows)]);
        let row_tiles = (rows as u64).div_ceil(u64::from(LINEAR_ROW_TILE));
        // Probe below, first reaching, and above each conservative grid
        // threshold. Full-resident-grid boundaries are preserved in the prior
        // sweep; these candidates are diagnostic, not production dispatch.
        for threshold in [two_block_grid, four_block_grid] {
            let column_groups_at_threshold = threshold.div_ceil(row_tiles);
            for column_groups in [
                column_groups_at_threshold.saturating_sub(1).max(1),
                column_groups_at_threshold,
                column_groups_at_threshold + 1,
            ] {
                let outputs = u32::try_from(column_groups * 4)
                    .expect("SM-derived output width must fit the launch ABI");
                shapes.insert((5120, outputs as usize, rows));
            }
        }
    }
    shapes.extend([(5120, 48, 1024), (5120, 1024, 1024)]);
    for inputs in [1024, 8192] {
        for outputs in [48, 1024] {
            for rows in [8, 64] {
                shapes.insert((inputs, outputs, rows));
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "benchmark":"native_q8_0_specialization_plan",
            "multiprocessors":multiprocessors,
            "generic_t8_blocks_per_sm":generic_t8_blocks_per_sm,
            "resident_grid_blocks":resident_grid,
            "two_blocks_per_sm_grid":two_block_grid,
            "four_blocks_per_sm_grid":four_block_grid,
            "shapes":shapes.iter().map(|&(input, output, rows)| serde_json::json!({
                "rows":rows, "input":input, "output":output,
                "projection_iterations":if rows == 1024 { 1 } else { 8 }
            })).collect::<Vec<_>>(),
            "largest_encoded_weight_bytes":shapes.iter().map(|&(k, n, _)| n * (k / 32) * 34).max(),
            "largest_cpu_decoded_weight_bytes":shapes.iter().map(|&(k, n, _)| n * k * std::mem::size_of::<f32>()).max()
        })
    );
    let routes = [
        ("generic_t8", &kernels.linear_tiled_f16, LINEAR_ROW_TILE),
        (
            "specialized_t8",
            &kernels.linear_q8_0_tiled_f16,
            LINEAR_ROW_TILE,
        ),
        ("specialized_t1", &kernels.linear_q8_0_f16, 1),
    ];
    // T8 specialization isolates format overhead; T1 changes cross-row reuse
    // and grid size. The prediction is recorded only, not used by production
    // dispatch or asserted against timings. Large-M probes bound GPU work to
    // one projection per command while retaining all paired formal rounds.
    for (inputs, outputs, rows) in shapes {
        let iterations = if rows == 1024 { 1 } else { 8 };
        let generic_t8_grid =
            (outputs as u64).div_ceil(4) * (rows as u64).div_ceil(u64::from(LINEAR_ROW_TILE));
        let predicted_t1_for_two_blocks = generic_t8_grid < two_block_grid;
        let predicted_t1_for_four_blocks = generic_t8_grid < four_block_grid;
        let mut fixture = Fixture::<f16>::for_format(
            &stream,
            GgufBlockFormat::Q8_0,
            rows,
            inputs,
            outputs,
            false,
        );
        fixture.run(&stream, &kernels.linear_f16, 1, 1);
        let reference = fixture.validate(&stream);
        for round in 0..6 {
            // Pair all routes on one fixture; rotate their order to avoid
            // assigning one route the same position in every round.
            for position in 0..routes.len() {
                let (route, kernel, row_tile) = routes[(round + position) % routes.len()];
                let (wall_ns, gpu_ns) = fixture.run(&stream, kernel, row_tile, iterations);
                assert_eq!(
                    fixture.validate_output(&stream),
                    reference,
                    "Q8_0 timed route {route} changed bits"
                );
                if round >= 2 {
                    println!(
                        "{}",
                        serde_json::json!({
                            "benchmark":"native_q8_0_specialization", "format":"Q8_0",
                            "input_dtype":"f16", "output_dtype":"f16", "synthetic_weights":true,
                            "rows":rows, "input":inputs, "output":outputs, "row_tile":row_tile,
                            "route":route, "round":round-2, "position":position,
                            "projection_iterations":iterations,
                            "multiprocessors":multiprocessors,
                            "generic_t8_blocks_per_sm":generic_t8_blocks_per_sm,
                            "resident_grid_blocks":resident_grid,
                            "generic_t8_grid_blocks":generic_t8_grid,
                            "predicted_t1_for_two_blocks":predicted_t1_for_two_blocks,
                            "predicted_t1_for_four_blocks":predicted_t1_for_four_blocks,
                            "command_wall_ns":wall_ns, "command_gpu_ns":gpu_ns,
                            "wall_ns":wall_ns/f64::from(iterations), "gpu_ns":gpu_ns/f64::from(iterations)
                        })
                    );
                }
            }
        }
        assert_eq!(fixture.validate(&stream), reference);
    }
}
