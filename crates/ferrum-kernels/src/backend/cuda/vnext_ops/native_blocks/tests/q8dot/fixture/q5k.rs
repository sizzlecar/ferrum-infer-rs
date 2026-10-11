//! Q5-only conformance for complete-group32 activation-Q8 integer dots.
//! No Q5 dot4 route or production operation is implicitly selected.

use super::*;
use q5k_reference::Q5Block;

fn check(
    case: &mut Fixture,
    stream: &Arc<CudaStream>,
    kernels: &CudaNativeBlockKernels,
    experiment: &Experiment,
    label: &str,
) {
    for tile in [1, 8] {
        for route in Route::Q5 {
            let mut stable = None;
            for _ in 0..2 {
                case.run(stream, kernels, experiment, route, tile, 1);
                let (bits, errors) = case.validate(stream, route);
                if let Some(previous) = &stable {
                    assert_eq!(&bits, previous, "Q5 route repeat");
                }
                stable = Some(bits);
                println!(
                    "{}",
                    serde_json::json!({
                        "experiment":"q5_activation_q8_group32_conformance","case":label,
                        "format":"Q5K","input":case.inputs,"output":case.outputs,"rows":case.rows,
                        "row_tile":tile,"route":route.name(),"integer_partial_values":route.integer_partial_values(),
                        "errors":errors
                    })
                );
            }
        }
    }
}

fn install(
    case: &mut Fixture,
    stream: &Arc<CudaStream>,
    mut block: impl FnMut(usize, usize) -> Q5Block,
    input: impl Fn(usize, usize) -> f16,
) {
    for col in 0..case.outputs {
        for bi in 0..case.inputs / 256 {
            let start = WEIGHT_PREFIX + (col * (case.inputs / 256) + bi) * 176;
            case.weights[start..start + 176].copy_from_slice(&block(col, bi).encode());
        }
    }
    for row in 0..case.rows {
        for i in 0..case.inputs {
            case.input[INPUT_PREFIX + row * case.inputs + i] = input(row, i);
        }
    }
    stream
        .memcpy_htod(&case.input, &mut case.input_gpu)
        .unwrap();
    stream
        .memcpy_htod(&case.weights, &mut case.weights_gpu)
        .unwrap();
    case.prepare_references();
}

fn constant_block(q: u8, minimum: u8, scale: f16) -> Q5Block {
    Q5Block {
        low: q4k_reference::Q4Block {
            d: scale,
            dmin: scale,
            scales: [1; 8],
            minima: [minimum; 8],
            quants: [q & 15; 256],
        },
        high: [q >= 16; 256],
    }
}

#[test]
#[ignore = "requires actual CUDA; Q5 group32 activation-Q8 conformance"]
fn q5_q8_group32_contract_and_batch_consistency_on_cuda() {
    let context = CudaContext::new(0).expect("Q5 activation-Q8 conformance requires CUDA");
    // Explicit single-stream ordering, as in the graph diagnostic/production runtime.
    unsafe {
        context.disable_event_tracking();
    }
    let stream = context.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let experiment = Experiment::load(&context);
    for inputs in [256, 768, 1280] {
        for rows in [1, 8, 9] {
            for outputs in [7, 17] {
                let mut case = Fixture::new(&stream, GgufBlockFormat::Q5K, rows, inputs, outputs);
                check(
                    &mut case,
                    &stream,
                    &kernels,
                    &experiment,
                    "shape_tail_guards",
                );
            }
        }
    }
    // K1280 has40 groups: lanes0..7 execute a second group while the others
    // retain one. The shared batch-row test also includes a realistic K5120.
    for (inputs, outputs) in [(768, 17), (1280, 17), (5120, 49)] {
        Fixture::batch_row_equivalence(
            &stream,
            &kernels,
            &experiment,
            GgufBlockFormat::Q5K,
            inputs,
            outputs,
        );
    }
    // Each row isolates one different K32 group. Sweep all32 positions so
    // every high-bit plane/address is exercised on the GPU, including0/15/16/31.
    for selected in 0..32 {
        let mut case = Fixture::new(&stream, GgufBlockFormat::Q5K, 8, 256, 7);
        install(
            &mut case,
            &stream,
            |col, _| {
                let mut b = q5k_reference::fixture_q5(col, 0);
                b.low.d = f16::from_f32(0.125);
                b.low.dmin = f16::from_f32(-0.25);
                for i in 0..256 {
                    let q = [0, 15, 16, 31][(col + i + i / 32) % 4];
                    b.low.quants[i] = q & 15;
                    b.high[i] = q >= 16;
                }
                b
            },
            |row, i| {
                if i == row * 32 + selected {
                    f16::ONE
                } else {
                    f16::ZERO
                }
            },
        );
        check(
            &mut case,
            &stream,
            &kernels,
            &experiment,
            "every_group_high_bit_position",
        );
    }
    // dot32 reaches±125984 and qsum reaches±4064 with dA=1. The weight scale
    // bounds final F16 output without weakening the integer-extrema coverage.
    for sign in [-1.0, 1.0] {
        for (q, minimum) in [(31, 0), (0, 1), (31, 31)] {
            let mut case = Fixture::new(&stream, GgufBlockFormat::Q5K, 3, 256, 7);
            install(
                &mut case,
                &stream,
                |_, _| constant_block(q, minimum, f16::from_f32(1.0 / 1024.0)),
                |_, _| f16::from_f32(sign * 127.0),
            );
            check(
                &mut case,
                &stream,
                &kernels,
                &experiment,
                "integer_extrema_min_and_cancellation",
            );
        }
    }
    for (d, dmin) in [
        (f16::ZERO, f16::ZERO),
        (f16::from_f32(-0.25), f16::from_f32(0.125)),
        (f16::from_f32(0.25), f16::from_f32(-0.125)),
        (f16::from_bits(1), f16::from_bits(0x8001)),
        (f16::from_bits(0x8001), f16::from_bits(1)),
    ] {
        let mut case = Fixture::new(&stream, GgufBlockFormat::Q5K, 3, 256, 7);
        install(
            &mut case,
            &stream,
            |col, _| {
                let mut b = q5k_reference::fixture_q5(col, 0);
                b.low.d = d;
                b.low.dmin = dmin;
                b
            },
            |row, i| f16::from_f32(((i + row) % 17) as f32 / 127.0 - 0.0625),
        );
        check(
            &mut case,
            &stream,
            &kernels,
            &experiment,
            "signed_zero_subnormal_scales",
        );
    }
    // The retained nonfinite policy is a NaN scale/q0 group and a NaN output.
    // Do not pass this case to the finite-only F64 formula/bound.
    for invalid in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY] {
        let mut case = Fixture::new(&stream, GgufBlockFormat::Q5K, 1, 256, 7);
        case.input[INPUT_PREFIX + 17] = invalid;
        stream
            .memcpy_htod(&case.input, &mut case.input_gpu)
            .unwrap();
        for tile in [1, 8] {
            for route in [Route::Group32QuantizeAndDot, Route::Group32DotOnly] {
                case.reset(&stream);
                let mut poison =
                    vec![f16::from_f32(-12345.0); OUTPUT_PREFIX + case.rows * case.stride + 5];
                poison[OUTPUT_PREFIX + COLUMN_OFFSET..OUTPUT_PREFIX + COLUMN_OFFSET + case.outputs]
                    .fill(f16::from_f32(123.0));
                stream.memcpy_htod(&poison, &mut case.output_gpu).unwrap();
                case.pack(&stream, &experiment);
                case.multiply(&stream, &kernels, &experiment, route, tile);
                stream.synchronize().unwrap();
                case.validate_pack(&stream);
                let output = stream.clone_dtoh(&case.output_gpu).unwrap();
                for (i, &value) in output.iter().enumerate() {
                    if (OUTPUT_PREFIX + COLUMN_OFFSET..OUTPUT_PREFIX + COLUMN_OFFSET + case.outputs)
                        .contains(&i)
                    {
                        assert!(value.is_nan(), "Q5 nonfinite group must propagate NaN");
                    } else {
                        assert_eq!(
                            value.to_bits(),
                            f16::from_f32(-12345.0).to_bits(),
                            "Q5 nonfinite guard"
                        );
                    }
                }
            }
        }
    }
}
