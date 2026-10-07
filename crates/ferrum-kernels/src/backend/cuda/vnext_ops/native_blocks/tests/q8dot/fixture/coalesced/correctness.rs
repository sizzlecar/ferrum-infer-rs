use super::*;

#[test]
fn qword_layout_permutation_preserves_extent_and_inverse() {
    for rows in [1, 3, 8, 9] {
        for groups in [1, 3, 8, 24, 33, 40, 160, 544] {
            for layout in [QwordLayout::GroupMajor, QwordLayout::WordMajor] {
                let mut seen = vec![false; rows * groups * 8];
                for row in 0..rows {
                    for group in 0..groups {
                        for word in 0..8 {
                            let physical = layout.physical(row, group, word, groups);
                            assert!(physical < seen.len());
                            assert!(!seen[physical], "qword alias");
                            seen[physical] = true;
                            assert_eq!(
                                layout.logical(physical, groups),
                                (row * groups + group) * 8 + word
                            );
                        }
                    }
                }
                assert!(seen.into_iter().all(|written| written));
            }
        }
    }
}

#[test]
#[ignore = "requires exclusive CUDA access; IQ4 layout, oracle and batch conformance"]
fn iq4_group32_word_major_layout_preserves_contract_on_cuda() {
    let context = CudaContext::new(0).expect("IQ4 layout conformance requires CUDA");
    // SAFETY: one stream/test thread with explicit synchronization owns buffers.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let experiment = LayoutExperiment::load(&context, GgufBlockFormat::Iq4Xs);
    pack_edges(&stream, &experiment);
    format_conformance(&stream, &kernels, &experiment, &[1, 8, 9], "iq4");
}

#[test]
#[ignore = "requires exclusive CUDA access; Q4/Q5 layout, affine oracle and batch conformance"]
fn q4_q5_group32_word_major_layout_preserves_contract_on_cuda() {
    let context = CudaContext::new(0).expect("Q4/Q5 layout conformance requires CUDA");
    // SAFETY: one stream/test thread owns buffers with explicit synchronization.
    unsafe { context.disable_event_tracking() };
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q5K] {
        let experiment = LayoutExperiment::load(&context, format);
        format_conformance(&stream, &kernels, &experiment, &[1, 3, 8, 9], "q4q5");
        affine_edges(&stream, &kernels, &experiment);
    }
}

fn format_conformance(
    stream: &Arc<CudaStream>,
    kernels: &CudaNativeBlockKernels,
    experiment: &LayoutExperiment,
    row_counts: &[usize],
    label: &str,
) {
    for inputs in [256, 768, 1280] {
        for outputs in [7, 17] {
            for &rows in row_counts {
                let mut case = Fixture::new(stream, experiment.format, rows, inputs, outputs);
                check_case(
                    &mut case,
                    stream,
                    kernels,
                    experiment,
                    label,
                    "shape_tail_guards",
                );
            }
        }
    }
    for (inputs, outputs) in [(768, 17), (1280, 17), (5120, 49)] {
        batch_rows(stream, kernels, experiment, inputs, outputs, label);
    }
    nonfinite_output(stream, kernels, experiment);
}

fn check_case(
    case: &mut Fixture,
    stream: &Arc<CudaStream>,
    kernels: &CudaNativeBlockKernels,
    experiment: &LayoutExperiment,
    label: &str,
    case_name: &str,
) {
    let mut reference = None;
    for tile in [1, 8] {
        for route in LayoutRoute::APPROXIMATE {
            for repeat in 0..2 {
                experiment.prepare(case, stream, route);
                experiment.enqueue(case, stream, kernels, route, tile, 1);
                let (bits, errors) = experiment.validate(case, stream, route);
                if let Some(reference) = &reference {
                    assert_eq!(&bits, reference, "layout/tile/repeat altered G32 bits");
                } else {
                    reference = Some(bits);
                }
                println!(
                    "{}",
                    serde_json::json!({
                        "experiment":format!("{label}_qword_layout_conformance"),
                        "format":format!("{:?}",case.format),"case":case_name,
                        "input":case.inputs,"output":case.outputs,"rows":case.rows,
                        "row_tile":tile,"route":route.name(),"repeat":repeat,"errors":errors
                    })
                );
            }
        }
    }
}

fn affine_edges(
    stream: &Arc<CudaStream>,
    kernels: &CudaNativeBlockKernels,
    experiment: &LayoutExperiment,
) {
    let maximum = match experiment.format {
        GgufBlockFormat::Q4K => 15,
        GgufBlockFormat::Q5K => 31,
        _ => panic!("affine edge fixture requires Q4K/Q5K"),
    };
    // With x=±127 each group reaches the integer dot/qsum bounds. Cover
    // min-only and exact coefficient cancellation, plus signed subnormal scales.
    for (q, minimum) in [(maximum, 0), (0, 1), (maximum, maximum)] {
        for (d, dmin) in [
            (f16::from_f32(1.0 / 1024.0), f16::from_f32(1.0 / 1024.0)),
            (f16::from_bits(1), f16::from_bits(0x8001)),
            (f16::ZERO, f16::ZERO),
        ] {
            let mut case = Fixture::new(stream, experiment.format, 3, 256, 7);
            let low = q4k_reference::Q4Block {
                d,
                dmin,
                scales: [1; 8],
                minima: [minimum; 8],
                quants: [q & 15; 256],
            };
            let encoded = match experiment.format {
                GgufBlockFormat::Q4K => low.encode().to_vec(),
                GgufBlockFormat::Q5K => q5k_reference::Q5Block {
                    low,
                    high: [q >= 16; 256],
                }
                .encode()
                .to_vec(),
                _ => unreachable!(),
            };
            for column in 0..case.outputs {
                let start = WEIGHT_PREFIX + column * encoded.len();
                case.weights[start..start + encoded.len()].copy_from_slice(&encoded);
            }
            for row in 0..case.rows {
                case.input
                    [INPUT_PREFIX + row * case.inputs..INPUT_PREFIX + (row + 1) * case.inputs]
                    .fill(f16::from_f32(if row % 2 == 0 { 127.0 } else { -127.0 }));
            }
            stream
                .memcpy_htod(&case.weights, &mut case.weights_gpu)
                .unwrap();
            stream
                .memcpy_htod(&case.input, &mut case.input_gpu)
                .unwrap();
            case.prepare_references();
            check_case(
                &mut case,
                stream,
                kernels,
                experiment,
                "q4q5",
                "affine_integer_bounds_min_scales",
            );
        }
    }
}

fn pack_edges(stream: &Arc<CudaStream>, experiment: &LayoutExperiment) {
    let values = [
        [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5],
        [0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0],
        [65504.0, -65504.0, 1.0, -1.0, 32.0, -32.0, 0.0, -0.0],
        [
            f16::from_bits(1).to_f32(),
            -f16::from_bits(1).to_f32(),
            f16::from_bits(0x03ff).to_f32(),
            -f16::from_bits(0x03ff).to_f32(),
            f16::from_bits(0x0400).to_f32(),
            -f16::from_bits(0x0400).to_f32(),
            0.0,
            -0.0,
        ],
        [f32::NAN, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ];
    for inputs in [32, 96, 1056] {
        for rows in [1, 3, 8, 9] {
            let mut case = Fixture::new(stream, GgufBlockFormat::Iq4Xs, rows, inputs, 1);
            // Each pattern also runs on the single-group/partial-CTA cases.
            for pattern in 0..values.len() {
                for index in 0..rows * inputs {
                    case.input[INPUT_PREFIX + index] =
                        f16::from_f32(values[(index / 32 + pattern) % values.len()][index % 8]);
                }
                stream
                    .memcpy_htod(&case.input, &mut case.input_gpu)
                    .unwrap();
                for layout in [QwordLayout::GroupMajor, QwordLayout::WordMajor] {
                    for _ in 0..2 {
                        case.reset(stream);
                        let before = output_bits(&case, stream);
                        experiment.pack(&mut case, stream, layout);
                        case.validate_pack_layout(stream, layout);
                        assert_eq!(output_bits(&case, stream), before, "pack changed output");
                    }
                }
            }
        }
    }
}

fn batch_rows(
    stream: &Arc<CudaStream>,
    kernels: &CudaNativeBlockKernels,
    experiment: &LayoutExperiment,
    inputs: usize,
    outputs: usize,
    label: &str,
) {
    let mut one = Fixture::new(stream, experiment.format, 1, inputs, outputs);
    let mut eight = Fixture::new(stream, experiment.format, 8, inputs, outputs);
    assert_eq!(one.weights, eight.weights);
    for row in [0, 4, 7] {
        eight.input[INPUT_PREFIX + row * inputs..INPUT_PREFIX + (row + 1) * inputs]
            .copy_from_slice(&one.input[INPUT_PREFIX..INPUT_PREFIX + inputs]);
    }
    assert!(
        eight.input[INPUT_PREFIX + inputs..INPUT_PREFIX + 2 * inputs]
            .iter()
            .zip(&one.input[INPUT_PREFIX..INPUT_PREFIX + inputs])
            .any(|(a, b)| a.to_bits() != b.to_bits())
    );
    stream
        .memcpy_htod(&eight.input, &mut eight.input_gpu)
        .unwrap();
    eight.prepare_references();
    let mut reference = None;
    for tile in [1, 8] {
        for route in LayoutRoute::APPROXIMATE {
            for _ in 0..2 {
                for case in [&mut one, &mut eight] {
                    experiment.prepare(case, stream, route);
                    experiment.enqueue(case, stream, kernels, route, tile, 1);
                    experiment.validate(case, stream, route);
                    let selected: &[usize] = if case.rows == 1 { &[0] } else { &[0, 4, 7] };
                    for &row in selected {
                        let actual = row_bits(case, stream, route.layout(), row);
                        if let Some(reference) = &reference {
                            assert_eq!(&actual, reference, "M1/M8 row{row} {route:?} T{tile}");
                        } else {
                            reference = Some(actual);
                        }
                    }
                }
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"experiment":format!("{label}_qword_layout_batch_bits"),
        "format":format!("{:?}",experiment.format),
        "input":inputs,"output":outputs,"rows":[1,8],"positions":[0,4,7],
        "comparison":"canonical qwords, scale and output bits across layouts/T1/T8/inclusive/dot/repeats"})
    );
}

fn nonfinite_output(
    stream: &Arc<CudaStream>,
    kernels: &CudaNativeBlockKernels,
    experiment: &LayoutExperiment,
) {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut case = Fixture::new(stream, experiment.format, 9, 1280, 7);
        for row in 0..case.rows {
            // Last K32 group exercises a partial final lane group as well.
            case.input[INPUT_PREFIX + (row + 1) * case.inputs - 1] = f16::from_f32(invalid);
        }
        stream
            .memcpy_htod(&case.input, &mut case.input_gpu)
            .unwrap();
        for tile in [1, 8] {
            for route in LayoutRoute::APPROXIMATE {
                experiment.prepare(&mut case, stream, route);
                let mut poison = stream.clone_dtoh(&case.output_gpu).unwrap();
                for row in 0..case.rows {
                    for column in 0..case.outputs {
                        poison[OUTPUT_PREFIX + row * case.stride + COLUMN_OFFSET + column] =
                            f16::ZERO;
                    }
                }
                stream.memcpy_htod(&poison, &mut case.output_gpu).unwrap();
                experiment.enqueue(&mut case, stream, kernels, route, tile, 1);
                case.validate_pack_layout(stream, route.layout());
                for (index, value) in stream
                    .clone_dtoh(&case.output_gpu)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    let logical = index.checked_sub(OUTPUT_PREFIX).is_some_and(|j| {
                        j < case.rows * case.stride
                            && (COLUMN_OFFSET..COLUMN_OFFSET + case.outputs)
                                .contains(&(j % case.stride))
                    });
                    if logical {
                        assert!(value.is_nan(), "nonfinite marker did not reach output");
                    } else {
                        assert_eq!(value.to_bits(), poison[index].to_bits(), "output guard");
                    }
                }
            }
        }
    }
}
