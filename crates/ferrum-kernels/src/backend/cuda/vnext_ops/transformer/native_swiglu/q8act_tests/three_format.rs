//! Fused three-format provider gate; reuses the independent G32 byte oracle.
//! Each projection consumes the actual preceding F16 boundary. Activation
//! quantization error is reported separately from the implementation bound.

use super::*;

#[path = "../../../native_blocks/tests/q8dot/oracle.rs"]
mod oracle;
#[path = "../../../native_blocks/tests/q8dot/q4k_reference.rs"]
mod q4k_reference;
#[path = "../../../native_blocks/tests/q8dot/q5k_reference.rs"]
mod q5k_reference;

pub(super) fn affine_matrix(
    format: MatrixFormat,
    n: usize,
    k: usize,
    salt: usize,
) -> (Vec<u8>, Vec<f32>) {
    let MatrixFormat::Block(f @ (GgufBlockFormat::Q4K | GgufBlockFormat::Q5K)) = format else {
        return matrix(format, n, k, salt);
    };
    assert_eq!(k % 256, 0);
    let mut bytes = Vec::new();
    for column in 0..n {
        for block in 0..k / 256 {
            let mut value = q5k_reference::fixture_q5(column + salt, block);
            match column % 6 {
                0 => {
                    // Nonzero min-only reconstruction: omitting qsum fails.
                    value.low.scales.fill(0);
                    value.low.minima.fill(31);
                }
                1 => {
                    // Dot and min cancel, but the rounding bound must retain
                    // both magnitudes instead of using abs(decoded weight).
                    value.low.d = f16::from_f32(1.0 / 256.0);
                    value.low.dmin = value.low.d;
                    value.low.scales.fill(1);
                    value
                        .low
                        .minima
                        .fill(if f == GgufBlockFormat::Q5K { 31 } else { 15 });
                    value.low.quants.fill(15);
                    value.high.fill(true);
                }
                2 => value.low.d = -value.low.d,
                3 => value.low.dmin = -value.low.dmin,
                4 => {
                    value.low.d = f16::from_bits(1);
                    value.low.dmin = f16::from_bits(0x8001);
                }
                _ => {}
            }
            if f == GgufBlockFormat::Q5K {
                bytes.extend(value.encode());
            } else {
                bytes.extend(value.low.encode());
            }
        }
    }
    let mut decoded = vec![0.0; n * k];
    f.decode(&bytes, &mut decoded).unwrap();
    (bytes, decoded)
}

#[derive(Default)]
pub(super) struct Errors {
    outputs: usize,
    implementation: f64,
    activation_quantization: f64,
    factoring: f64,
    versus_original: f64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assert_staged_projection(
    input: &[f16],
    bytes: &[u8],
    decoded: &[f32],
    actual: &[f16],
    format: GgufBlockFormat,
    k: usize,
    n: usize,
    stride: usize,
    offset: usize,
) -> Errors {
    let rows = input.len() / k;
    let (q, scales) = oracle::quantize(input, rows, k, k);
    let mut errors = Errors::default();
    for row in 0..rows {
        for column in 0..n {
            let mut policy = 0.0;
            let mut expanded = 0.0;
            for group in 0..k / 32 {
                let b = (column * (k / 256) + group / 8) * format.block_bytes();
                let begin = row * k + group * 32;
                let (value, magnitude) = oracle::block_formula(
                    format,
                    &bytes[b..b + format.block_bytes()],
                    group % 8,
                    &q[begin..begin + 32],
                    scales[row * (k / 32) + group],
                );
                policy += value;
                expanded += magnitude;
            }
            let mut original = 0.0;
            let mut quantized = 0.0;
            let mut original_abs = 0.0;
            let mut activation_bound = 0.0;
            for i in 0..k {
                let x = input[row * k + i].to_f64();
                let xhat = f64::from(q[row * k + i]) * f64::from(scales[row * (k / 32) + i / 32]);
                let w = f64::from(decoded[column * k + i]);
                original += w * x;
                quantized += w * xhat;
                original_abs += (w * x).abs();
                activation_bound += w.abs() * (xhat - x).abs();
            }
            // Exact I32 dot/qsum; <=4 FP32 rescale/min steps, one per-lane
            // serial sum and five warp-tree additions. This is the retained
            // G32 bound, including expanded affine terms before cancellation.
            let nu = ((k / 32).div_ceil(32) + 9) as f64 * f64::from(f32::EPSILON);
            let accumulation = nu / (1.0 - nu) * expanded;
            let bound = accumulation
                + 2.0_f64.powi(-11) * (policy.abs() + accumulation)
                + f16::from_bits(1).to_f64() / 2.0;
            let got = actual[row * stride + offset + column].to_f64();
            let error = (got - policy).abs();
            assert!(
                got.is_finite() && error <= bound,
                "{format:?} row={row} column={column}: {got} vs policy={policy}, bound={bound}"
            );
            let f64_slack = k as f64 * f64::EPSILON * (original_abs + expanded);
            assert!((quantized - original).abs() <= activation_bound + f64_slack);
            errors.outputs += 1;
            errors.implementation = errors.implementation.max(error);
            errors.activation_quantization = errors
                .activation_quantization
                .max((quantized - original).abs());
            errors.factoring = errors.factoring.max((policy - quantized).abs());
            errors.versus_original = errors.versus_original.max((got - original).abs());
        }
    }
    errors
}

#[test]
#[ignore = "requires actual CUDA and the explicit IQ4_XS/Q4_K/Q5_K G32 provider"]
fn three_format_q8act_swiglu_matches_affine_stage_oracles_and_batch_rows_on_cuda() {
    use GgufBlockFormat::{Iq4Xs, Q4K, Q5K, Q6K};
    use MatrixFormat::{Block as Q, DenseF16 as D};
    let profile = Q8ActSwiGluProfile::Q4KQ5KIq4Xs;
    let context = CudaContext::new(0).expect("three-format SwiGLU conformance requires CUDA");
    let stream = context.default_stream();
    let native = CudaNativeBlockKernels::load(&context).unwrap();
    let q8 = Q8ActKernels::load_for_profile(&context, profile).unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let silu = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    let hidden = 256;
    for (case_index, (intermediate, formats)) in [
        (256, [Q(Iq4Xs), Q(Q4K), Q(Q5K)]),
        (256, [Q(Q4K), Q(Q5K), Q(Iq4Xs)]),
        (256, [Q(Q5K), Q(Iq4Xs), Q(Q4K)]),
        (256, [Q(Q5K), D, Q(Q6K)]),
        (17, [Q(Q4K), Q(Q5K), D]),
    ]
    .into_iter()
    .enumerate()
    {
        let part = |i: usize, n: usize, k: usize, offset: usize| MatrixPart {
            component_id: id(&format!("component.{i}")),
            format: formats[i],
            rows: n as u32,
            columns: k as u32,
            output_offset: offset as u32,
            transform: None,
            signs_region: None,
        };
        let gate_up = [
            part(0, intermediate, hidden, 0),
            part(1, intermediate, hidden, intermediate),
        ];
        let down = [part(2, hidden, intermediate, 0)];
        let values = vec![
            binding(1, &gate_up, hidden, intermediate),
            binding(2, &down, intermediate, hidden),
        ];
        let plan = PreparedProjectionNumerics::prepare(&profile.arithmetic(), &values).unwrap();
        for (role, parts) in [
            (ProjectionRole::SwiGluGateUp, gate_up.as_slice()),
            (ProjectionRole::SwiGluDown, down.as_slice()),
        ] {
            Q8ActKernels::validate_parts_for_profile(
                profile,
                plan.projection(role).unwrap(),
                parts,
            )
            .unwrap();
        }
        let weights = [
            affine_matrix(formats[0], intermediate, hidden, 0),
            affine_matrix(formats[1], intermediate, hidden, 1),
            affine_matrix(formats[2], hidden, intermediate, 2),
        ];
        // Five extra bytes make each block pointer deliberately odd. All input
        // weights and the inner/outer canaries are checked after execution.
        let gpu_weights = weights
            .iter()
            .map(|(bytes, _)| {
                let mut padded = vec![0xBA_u8; 5];
                padded.extend(bytes);
                padded.extend([0xBA_u8; 3]);
                Guarded::new(&stream, &padded, 0xAB_u8)
            })
            .collect::<Vec<_>>();
        let pointers = gpu_weights
            .iter()
            .map(|w| w.pointer(&stream) + 5)
            .collect::<Vec<_>>();
        // Dense weights require natural F16 alignment; retain the odd pointer
        // exercise only for the byte-safe quantized formats.
        let mut pointers = pointers;
        let dense_weights = weights
            .iter()
            .zip(formats)
            .map(|((bytes, _), f)| matches!(f, D).then(|| Guarded::new(&stream, bytes, 0xAB_u8)))
            .collect::<Vec<_>>();
        for (i, dense) in dense_weights.iter().enumerate() {
            if let Some(dense) = dense {
                pointers[i] = dense.pointer(&stream);
            }
        }
        let common = (0..hidden)
            .map(|i| f16::from_f32(((i * 13 % 17) as f32 - 8.0) / 32768.0))
            .collect::<Vec<_>>();
        let mut scalar_reference: Option<[Vec<u16>; 3]> = None;
        for rows in [1, 3, 8, 9, 1025] {
            if rows == 1025 && case_index != 0 {
                continue;
            }
            let mut input = (0..rows * hidden)
                .map(|i| {
                    f16::from_f32(((i * 7 + i / hidden * 3) % 23) as f32 / 32768.0 - 11.0 / 32768.0)
                })
                .collect::<Vec<_>>();
            let mut copied_rows = vec![0, rows / 2, rows - 1];
            copied_rows.sort_unstable();
            copied_rows.dedup();
            for &row in &copied_rows {
                input[row * hidden..(row + 1) * hidden].copy_from_slice(&common);
            }
            let x = Guarded::new(&stream, &input, f16::from_f32(79.0));
            let bytes = q8act::workspace_per_token(&plan).unwrap() * rows as u64;
            let pack = Guarded::new(&stream, &vec![0xCB_u8; bytes as usize], 0xAB_u8);
            let mut previous = None;
            for repetition in 0..2 {
                let guard = f16::from_f32(-12345.0);
                let gates = Guarded::new(&stream, &vec![f16::NAN; rows * 2 * intermediate], guard);
                let act = Guarded::new(&stream, &vec![f16::NAN; rows * intermediate], guard);
                let y = Guarded::new(&stream, &vec![f16::NAN; rows * hidden], guard);
                launch(
                    &native,
                    &silu,
                    &stream,
                    &gate_up,
                    &down,
                    &pointers,
                    x.pointer(&stream),
                    y.pointer(&stream),
                    gates.pointer(&stream),
                    act.pointer(&stream),
                    rows as u32,
                    hidden as u32,
                    intermediate as u32,
                    0,
                    Some(&q8),
                    Some(&plan),
                    pack.pointer(&stream),
                    bytes,
                )
                .unwrap();
                let gate = gates.read(&stream);
                let activation = act.read(&stream);
                let output = y.read(&stream);
                for (index, source, actual, k, n, stride, offset, leaf) in [
                    (
                        0,
                        input.as_slice(),
                        gate.as_slice(),
                        hidden,
                        intermediate,
                        2 * intermediate,
                        0,
                        &plan.projections()[0].leaves()[0],
                    ),
                    (
                        1,
                        input.as_slice(),
                        gate.as_slice(),
                        hidden,
                        intermediate,
                        2 * intermediate,
                        intermediate,
                        &plan.projections()[0].leaves()[1],
                    ),
                    (
                        2,
                        activation.as_slice(),
                        output.as_slice(),
                        intermediate,
                        hidden,
                        hidden,
                        0,
                        &plan.projections()[1].leaves()[0],
                    ),
                ] {
                    if leaf.is_staged() {
                        let Q(format) = formats[index] else {
                            panic!("dense leaf cannot be staged")
                        };
                        let e = assert_staged_projection(
                            source,
                            &weights[index].0,
                            &weights[index].1,
                            actual,
                            format,
                            k,
                            n,
                            stride,
                            offset,
                        );
                        println!(
                            "{}",
                            serde_json::json!({"record":"three_format_swiglu_oracle", "case":case_index,"rows":rows,"projection":index,"format":format.format_id(),"repetition":repetition,"outputs":e.outputs,"implementation_max_abs":e.implementation,"activation_quantization_max_abs":e.activation_quantization,"factoring_max_abs":e.factoring,"versus_original_max_abs":e.versus_original,"oracle_scope":"all projection outputs; implementation bound excludes activation error"})
                        );
                    } else {
                        projection_oracle(
                            source,
                            &weights[index].1,
                            actual,
                            k,
                            n,
                            stride,
                            offset,
                            false,
                        );
                    }
                }
                for row in 0..rows {
                    for col in 0..intermediate {
                        let g = gate[row * 2 * intermediate + col].to_f64();
                        let u = gate[row * 2 * intermediate + intermediate + col].to_f64();
                        let reference = g / (1.0 + (-g).exp()) * u;
                        let got = activation[row * intermediate + col].to_f64();
                        assert!(
                            got.is_finite()
                                && (got - reference).abs()
                                    <= 0.001 * reference.abs() + f16::from_bits(1).to_f64()
                        );
                    }
                }
                let stages = [gate, activation, output]
                    .map(|x| x.into_iter().map(f16::to_bits).collect::<Vec<_>>());
                if let Some(prior) = &previous {
                    assert_eq!(
                        &stages, prior,
                        "own-route repeat case={case_index} rows={rows}"
                    );
                }
                let widths = [2 * intermediate, intermediate, hidden];
                if scalar_reference.is_none() {
                    assert_eq!(rows, 1);
                    scalar_reference =
                        Some(std::array::from_fn(|i| stages[i][..widths[i]].to_vec()));
                }
                for &row in &copied_rows {
                    for i in 0..3 {
                        assert_eq!(
                            &stages[i][row * widths[i]..(row + 1) * widths[i]],
                            scalar_reference.as_ref().unwrap()[i].as_slice(),
                            "same-row case={case_index} rows={rows} row={row} stage={i}"
                        );
                    }
                }
                previous = Some(stages);
            }
            x.assert_unchanged(&stream);
            let _ = pack.read(&stream);
        }
        for weights in gpu_weights {
            weights.assert_unchanged(&stream);
        }
        for weights in dense_weights.into_iter().flatten() {
            weights.assert_unchanged(&stream);
        }
    }
}
