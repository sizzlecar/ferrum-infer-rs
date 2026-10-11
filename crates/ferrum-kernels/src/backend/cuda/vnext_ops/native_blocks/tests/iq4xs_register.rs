//! Register codebook versus the previous fixed-format constant lookup.

use super::{q4k::Fixture, *};

const SCALES: [u16; 12] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x2400, 0xa400, 0x3000, 0xb000,
];

fn boundary_weights(blocks: usize) -> Vec<u8> {
    let mut encoded = vec![0; blocks * 136];
    for (index, block) in encoded.chunks_exact_mut(136).enumerate() {
        block[..2].copy_from_slice(&SCALES[(index / 8) % SCALES.len()].to_le_bytes());
        let mut scales_h = 0_u16;
        for group in 0..8 {
            // Eight blocks cover every signed six-bit scale, -32 through 31,
            // for each half-scale sign/zero/subnormal boundary above.
            let scale = (index % 8) * 8 + group;
            block[4 + group / 2] |= ((scale & 15) as u8) << (4 * (group % 2));
            scales_h |= ((scale >> 4) as u16) << (2 * group);
            for lane in 0..16 {
                // Both nibbles independently visit all sixteen codebook codes.
                let low = (lane + group + index) % 16;
                let high = (15 - lane + group + index) % 16;
                block[8 + group * 16 + lane] = (low | (high << 4)) as u8;
            }
        }
        block[2..4].copy_from_slice(&scales_h.to_le_bytes());
    }
    encoded
}

fn check_decoded_coefficients(stream: &Arc<CudaStream>, kernels: &CudaNativeBlockKernels) {
    let bytes = boundary_weights(SCALES.len() * 8);
    let mut decoded = vec![0.0_f32; bytes.len() / 136 * 256];
    GgufBlockFormat::Iq4Xs.decode(&bytes, &mut decoded).unwrap();
    // A partial final block checks count guards independently of full matrices.
    let count = decoded.len() - 7;
    let guard = -12345.0_f32;
    let initial = vec![guard; count + 16];
    let mut expected = initial.clone();
    expected[8..8 + count].copy_from_slice(&decoded[..count]);
    let mut padded = vec![0xcc_u8; 5];
    padded.extend_from_slice(&bytes);
    padded.extend([0xcc; 8]);
    let weights = stream.clone_htod(&padded).unwrap();
    let mut output = stream.clone_htod(&initial).unwrap();
    for register_lookup in [false, true] {
        for _ in 0..2 {
            stream.memcpy_htod(&initial, &mut output).unwrap();
            {
                let input = weights.slice(5..5 + bytes.len());
                let mut region = output.slice_mut(8..);
                let kernel = if register_lookup {
                    &kernels.iq4xs_register_decode
                } else {
                    &kernels.decode
                };
                let count_u32 = count as u32;
                let mut launch = stream.launch_builder(kernel);
                launch.arg(&input).arg(&mut region).arg(&count_u32);
                let parameters = [23_u32, 256, 136];
                if !register_lookup {
                    for parameter in &parameters {
                        launch.arg(parameter);
                    }
                }
                // SAFETY: Both decoders guard count. The input retains the
                // complete final block at an odd byte address, and the output
                // retains count plus suffix guards. Excess threads must not write.
                unsafe { launch.launch(LaunchConfig::for_num_elems(count_u32 + 17)) }.unwrap();
            }
            let actual = stream.clone_dtoh(&output).unwrap();
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "register={register_lookup} decoded coefficient {index}"
                );
            }
        }
    }
    assert_eq!(stream.clone_dtoh(&weights).unwrap(), padded);
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn iq4xs_register_lookup_preserves_coefficients_and_linear_bits_on_cuda() {
    let context = CudaContext::new(0).expect("IQ4_XS register conformance requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    check_decoded_coefficients(&stream, &kernels);
    for (rows, inputs, outputs) in [
        (1, 256, 7),
        (2, 768, 17),
        (4, 512, 7),
        (7, 768, 17),
        (8, 2560, 17),
        (9, 768, 7),
        (65, 512, 17),
    ] {
        let mut fixture = Fixture::<f16>::for_encoded_weights(
            &stream,
            GgufBlockFormat::Iq4Xs,
            rows,
            inputs,
            outputs,
            true,
            boundary_weights(outputs * (inputs / 256)),
        );
        // Generic remains the old indexed lookup. The fixture independently
        // decodes on CPU and checks a F64 dot-product oracle plus all guards.
        fixture.run(&stream, &kernels.linear_f16, 1, 1);
        let reference = fixture.validate(&stream);
        for (kernel, tile) in [
            (&kernels.linear_tiled_f16, LINEAR_ROW_TILE),
            (&kernels.linear_iq4xs_constant_f16, 1),
            (&kernels.linear_iq4xs_constant_tiled_f16, LINEAR_ROW_TILE),
            (&kernels.linear_iq4xs_f16, 1),
            (&kernels.linear_iq4xs_tiled_f16, LINEAR_ROW_TILE),
        ] {
            for iterations in [1, 2] {
                fixture.run(&stream, kernel, tile, iterations);
                assert_eq!(
                    fixture.validate(&stream),
                    reference,
                    "{rows}x{inputs}x{outputs}"
                );
            }
        }
        for _ in 0..2 {
            fixture.run_dispatch(&stream, &kernels);
            assert_eq!(fixture.validate(&stream), reference, "production dispatch");
        }
    }
}

#[test]
#[ignore = "paired GPU timing; coordinate exclusive CUDA access"]
fn iq4xs_register_lookup_paired_microbench() {
    let context = CudaContext::new(0).expect("IQ4_XS register microbench requires CUDA");
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let stream = context.default_stream();
    // Actual FFN (K,N) in both directions; synthetic GGUF blocks and sparse
    // activations keep the full independent CPU oracle bounded even for M1025.
    // Both routes keep tile8, launch geometry, reconstruction and accumulation.
    // These timings isolate lookup, not a new large-M tiling strategy.
    for (inputs, outputs) in [(5120, 17408), (17408, 5120)] {
        for rows in [4, 8, 1024, 1025] {
            let iterations = if rows >= 1024 { 1 } else { 8 };
            let mut fixture = Fixture::<f16>::for_format(
                &stream,
                GgufBlockFormat::Iq4Xs,
                rows,
                inputs,
                outputs,
                false,
            );
            fixture.run(
                &stream,
                &kernels.linear_iq4xs_constant_tiled_f16,
                LINEAR_ROW_TILE,
                1,
            );
            let reference = fixture.validate(&stream);
            for round in 0..6 {
                for (position, register_lookup) in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                }
                .into_iter()
                .enumerate()
                {
                    let (route, kernel) = if register_lookup {
                        ("register_t8", &kernels.linear_iq4xs_tiled_f16)
                    } else {
                        ("constant_t8", &kernels.linear_iq4xs_constant_tiled_f16)
                    };
                    let (wall_ns, gpu_ns) =
                        fixture.run(&stream, kernel, LINEAR_ROW_TILE, iterations);
                    assert_eq!(
                        fixture.validate_output(&stream),
                        reference,
                        "timed {route} changed bits"
                    );
                    if round >= 2 {
                        println!(
                            "{}",
                            serde_json::json!({
                                "benchmark":"native_iq4xs_register_lookup", "format":"Iq4Xs",
                                "input_dtype":"f16", "output_dtype":"f16", "synthetic_weights":true,
                                "activation_pattern":"sparse16", "rows":rows, "input":inputs,
                                "output":outputs, "row_tile":LINEAR_ROW_TILE,
                                "route":route, "round":round-2, "position":position,
                                "projection_iterations":iterations,
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
}
