//! The dense F16 launchers are distinct from the native block matrix path.
use super::test_support::Guarded;
use super::*;
use cudarc::driver::CudaContext;
use half::f16;

fn values(count: usize, salt: usize) -> Vec<f16> {
    (0..count)
        .map(|i| f16::from_f32(((i * 13 + salt * 7) % 41) as f32 / 32.0 - 0.625))
        .collect()
}

fn projection(input: &[f16], weight: &[f16], actual: &[f16], hidden: usize, outputs: usize) {
    assert_eq!(actual.len(), input.len() / hidden * outputs);
    for (row, x) in input.chunks_exact(hidden).enumerate() {
        for (column, w) in weight.chunks_exact(hidden).enumerate() {
            let products = x.iter().zip(w).map(|(a, b)| a.to_f64() * b.to_f64());
            let expected = products.clone().sum::<f64>();
            // Same bound as native_swiglu: F32 reduction error plus one F16
            // relative spacing and the smallest F16 subnormal for output rounding.
            let bound = (hidden as f64 * f64::from(f32::EPSILON) + 0.0009765625)
                * products.map(f64::abs).sum::<f64>()
                + f16::from_bits(1).to_f64();
            let value = actual[row * outputs + column].to_f64();
            assert!(
                value.is_finite() && (value - expected).abs() <= bound,
                "projection[{row},{column}]: {value}, F64={expected}, bound={bound}"
            );
        }
    }
}

fn activation(actual: f16, expected: f64) {
    // Preserve the existing F16 activation bound used by the native providers.
    let bound = expected.abs().max(1.0) * 0.001 + 1.0e-5;
    assert!(
        actual.is_finite() && (actual.to_f64() - expected).abs() <= bound,
        "activation={}, F64={expected}, bound={bound}",
        actual.to_f32()
    );
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn cublas_swiglu_preserves_stage_oracles_offsets_and_tail_rows_on_cuda() {
    let context = CudaContext::new(0).expect("F16 projection conformance requires CUDA");
    let stream = context.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let silu = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    for (hidden, intermediate) in [(17, 19), (256, 128)] {
        for rows in [1, 3, 33] {
            let input = values(rows * hidden, 0);
            let weight = values(2 * intermediate * hidden, 1);
            let down_weight = values(hidden * intermediate, 2);
            let guard = f16::from_f32(-117.0);
            let x = Guarded::new(&stream, &input, guard);
            let w = Guarded::new(&stream, &weight, guard);
            let d = Guarded::new(&stream, &down_weight, guard);
            let gates = Guarded::new(&stream, &vec![f16::NAN; rows * 2 * intermediate], guard);
            let acts = Guarded::new(&stream, &vec![f16::NAN; rows * intermediate], guard);
            let output = Guarded::new(&stream, &vec![f16::NAN; rows * hidden], guard);
            launch_gemm_f16(
                &blas,
                x.pointer(&stream),
                w.pointer(&stream),
                gates.pointer(&stream),
                rows as i32,
                (2 * intermediate) as i32,
                hidden as i32,
                "test gate-up",
            )
            .unwrap();
            launch_silu_mul(
                &stream,
                &silu,
                gates.pointer(&stream),
                acts.pointer(&stream),
                intermediate as i32,
                (rows * intermediate) as u64,
            )
            .unwrap();
            launch_gemm_f16(
                &blas,
                acts.pointer(&stream),
                d.pointer(&stream),
                output.pointer(&stream),
                rows as i32,
                hidden as i32,
                intermediate as i32,
                "test down",
            )
            .unwrap();
            let gate_actual = gates.read(&stream);
            let act_actual = acts.read(&stream);
            projection(&input, &weight, &gate_actual, hidden, 2 * intermediate);
            for (row, pair) in gate_actual.chunks_exact(2 * intermediate).enumerate() {
                for column in 0..intermediate {
                    let g = pair[column].to_f64();
                    let u = pair[intermediate + column].to_f64();
                    activation(
                        act_actual[row * intermediate + column],
                        g / (1.0 + (-g).exp()) * u,
                    );
                }
            }
            projection(
                &act_actual,
                &down_weight,
                &output.read(&stream),
                intermediate,
                hidden,
            );
            x.assert_unchanged(&stream);
            w.assert_unchanged(&stream);
            d.assert_unchanged(&stream);
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn f16_residual_preserves_rounding_alias_and_tail_guards_on_cuda() {
    let context = CudaContext::new(0).expect("F16 residual conformance requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::RESIDUAL_ADD))
        .unwrap();
    let function = module
        .load_function(ResidualPrecision::F16.kernel())
        .unwrap();
    for count in [1, 33, 513] {
        // Include halfway rounding, cancellation, subnormal sums and large
        // finite operands rather than only exactly representable additions.
        let pairs = [
            (1.0, 0.00048828125),
            (1.0009765625, 0.00048828125),
            (-1.0, 1.0),
            (f16::from_bits(1).to_f32(), f16::from_bits(1).to_f32()),
            (60000.0, 16.0),
            (-60000.0, -16.0),
        ];
        let left = (0..count)
            .map(|i| f16::from_f32(pairs[i % pairs.len()].0))
            .collect::<Vec<_>>();
        let right = (0..count)
            .map(|i| f16::from_f32(pairs[i % pairs.len()].1))
            .collect::<Vec<_>>();
        let expected = left
            .iter()
            .zip(&right)
            .map(|(a, b)| f16::from_f32(a.to_f32() + b.to_f32()))
            .collect::<Vec<_>>();
        for alias in [false, true] {
            let guard = f16::from_f32(-117.0);
            let x = Guarded::new(&stream, &left, guard);
            let update = Guarded::new(&stream, &right, guard);
            let output = Guarded::new(&stream, &vec![f16::NAN; count], guard);
            let destination = if alias { &x } else { &output };
            let (a, b, c, n) = (
                x.pointer(&stream),
                update.pointer(&stream),
                destination.pointer(&stream),
                count as i32,
            );
            let mut launch = stream.launch_builder(&function);
            launch.arg(&a).arg(&b).arg(&c).arg(&n);
            // SAFETY: Live offset allocations have exactly n elements; the
            // only alias is the provider's declared residual/output alias.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count as u32)) }.unwrap();
            assert_eq!(
                destination
                    .read(&stream)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            update.assert_unchanged(&stream);
            if !alias {
                x.assert_unchanged(&stream);
            }
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn f16_embedding_preserves_ids_offsets_and_output_guards_on_cuda() {
    let context = CudaContext::new(0).expect("F16 embedding conformance requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::EMBEDDING_LOOKUP))
        .unwrap();
    let function = module
        .load_function(super::super::EMBEDDING_FUNCTION_NAME)
        .unwrap();
    let vocabulary = 7_u32;
    for hidden in [1, 33, 257] {
        let weights = values(vocabulary as usize * hidden, 3);
        for ids in [vec![6], vec![6, 0, 6, 7, u32::MAX, 2]] {
            let guard = f16::from_f32(-117.0);
            let table = Guarded::new(&stream, &weights, guard);
            let tokens = Guarded::new(&stream, &ids, 0x5a5a1234_u32);
            let output = Guarded::new(&stream, &vec![f16::NAN; ids.len() * hidden], guard);
            let (w, t, y, batch, dim) = (
                table.pointer(&stream),
                tokens.pointer(&stream),
                output.pointer(&stream),
                ids.len() as i32,
                hidden as i32,
            );
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&w)
                .arg(&t)
                .arg(&y)
                .arg(&batch)
                .arg(&dim)
                .arg(&vocabulary);
            // SAFETY: Launch the same 2D geometry as the F16 provider; its
            // invalid-id branch must zero-fill without reading outside table.
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (
                        (hidden as u32).div_ceil(THREADS_PER_BLOCK),
                        ids.len() as u32,
                        1,
                    ),
                    block_dim: (THREADS_PER_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
            let actual = output.read(&stream);
            for (row, &id) in ids.iter().enumerate() {
                let expected = if id < vocabulary {
                    weights[id as usize * hidden..][..hidden].to_vec()
                } else {
                    vec![f16::ZERO; hidden]
                };
                assert_eq!(&actual[row * hidden..][..hidden], expected);
            }
            table.assert_unchanged(&stream);
            tokens.assert_unchanged(&stream);
        }
    }
}

#[cfg(feature = "vllm-marlin")]
#[test]
#[ignore = "requires an actual CUDA device"]
fn planar_gated_activations_match_f64_and_preserve_guards_on_cuda() {
    let context = CudaContext::new(0).expect("planar activation conformance requires CUDA");
    let stream = context.default_stream();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let silu = module.load_function(PLANAR_SILU_MUL_FUNCTION_NAME).unwrap();
    let gelu = module
        .load_function(PLANAR_GELU_TANH_MUL_FUNCTION_NAME)
        .unwrap();
    for count in [1, 33, 513] {
        let gate = (0..count)
            .map(|i| f16::from_f32(((i * 7) % 97) as f32 / 4.0 - 12.0))
            .collect::<Vec<_>>();
        let up = values(count, 4);
        let guard = f16::from_f32(-117.0);
        let g = Guarded::new(&stream, &gate, guard);
        let u = Guarded::new(&stream, &up, guard);
        for is_gelu in [false, true] {
            let output = Guarded::new(&stream, &vec![f16::NAN; count], guard);
            if is_gelu {
                launch_planar_gelu_tanh_mul(
                    &stream,
                    &gelu,
                    g.pointer(&stream),
                    u.pointer(&stream),
                    output.pointer(&stream),
                    count as u64,
                )
                .unwrap();
            } else {
                launch_planar_silu_mul(
                    &stream,
                    &silu,
                    g.pointer(&stream),
                    u.pointer(&stream),
                    output.pointer(&stream),
                    count as u64,
                )
                .unwrap();
            }
            for ((actual, g), u) in output.read(&stream).iter().zip(&gate).zip(&up) {
                let g = g.to_f64();
                let expected = if is_gelu {
                    0.5 * g
                        * (1.0
                            + (std::f64::consts::FRAC_2_PI.sqrt() * (g + 0.044715 * g.powi(3)))
                                .tanh())
                } else {
                    g / (1.0 + (-g).exp())
                };
                activation(*actual, expected * u.to_f64());
            }
            g.assert_unchanged(&stream);
            u.assert_unchanged(&stream);
        }
    }
}
