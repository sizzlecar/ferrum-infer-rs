//! Composed, test-only projection policy. No new provider or numerical profile.
use super::*;
use crate::backend::cuda::vnext_ops::native_blocks::{
    q8_f32scale::{PackLayout, Q8F32ScaleKernels},
    weights::{MatrixFormat, MatrixPart},
};
use crate::gguf_blocks::{
    q4k_q8_reference::{dot_reference, fixture_block, pack_rows, Q4Block},
    q56k_q8_reference::{dot_q5, fixture_q5, Q5Block},
    GgufBlockFormat,
};
use ferrum_interfaces::vnext::WeightId;

#[path = "q8_projection/q5k_fixed_abi.rs"]
mod q5k_fixed_abi;

const HIDDEN: usize = 256;
const BATCHES: [[usize; 3]; 4] = [[0, 1, 6], [0, 1, 7], [0, 1, 32], [0, 1, 63]];

fn q4_block(column: usize, shift: u32) -> Q4Block {
    let mut block = fixture_block(column, 0);
    let scale = 2.0_f32.powi(-(shift as i32));
    for value in [&mut block.d, &mut block.dmin] {
        let expected = value.to_f32() * scale;
        let scaled = f16::from_f32(expected);
        assert!(expected != 0.0 && scaled != f16::ZERO);
        assert_eq!(
            scaled.to_f32(),
            expected,
            "fixture scale must be exact in F16"
        );
        *value = scaled;
    }
    block
}

fn q5_block(column: usize, shift: u32) -> Q5Block {
    let mut block = fixture_q5(column, 0);
    block.low = q4_block(column, shift);
    block
}

struct Matrix {
    part: MatrixPart,
    salt: usize,
    scale_shift: u32,
    dense: Vec<f32>,
    bytes: Guarded<u8>,
}

impl Matrix {
    fn new(
        stream: &Arc<CudaStream>,
        format: GgufBlockFormat,
        rows: usize,
        offset: usize,
        salt: usize,
        scale_shift: u32,
    ) -> Self {
        let mut bytes = Vec::new();
        for col in 0..rows {
            match format {
                GgufBlockFormat::Q4K => bytes.extend(q4_block(col + salt, scale_shift).encode()),
                GgufBlockFormat::Q5K => bytes.extend(q5_block(col + salt, scale_shift).encode()),
                GgufBlockFormat::Q8_0 => {
                    assert_eq!(scale_shift, 0);
                    for block in 0..HIDDEN / 32 {
                        bytes.extend(f16::from_f32(1.0 / 1024.0).to_le_bytes());
                        bytes.extend(
                            (0..32).map(|i| (((i * 7 + block + col + salt) % 23) as i8 - 11) as u8),
                        );
                    }
                }
                _ => unreachable!("bounded GDN projection formats"),
            }
        }
        let mut dense = vec![0.0; rows * HIDDEN];
        format.decode(&bytes, &mut dense).unwrap();
        // Only the output fixture is scaled: its original independent result
        // exceeded F16_MAX in attempt2. Preserve high-amplitude QKV/z coverage.
        // Every reconstructed coefficient must scale exactly and remain nonzero
        // whenever its original coefficient was nonzero; no underflow shortcut.
        if scale_shift != 0 {
            let scale = 2.0_f32.powi(-(scale_shift as i32));
            for col in 0..rows {
                let original = fixture_q5(col + salt, 0);
                assert_eq!(format, GgufBlockFormat::Q5K);
                for i in 0..HIDDEN {
                    let before = original.strict_weight(i);
                    let after = dense[col * HIDDEN + i];
                    assert_eq!(after.to_bits(), (before * scale).to_bits());
                    assert_eq!(before == 0.0, after == 0.0);
                }
            }
        }
        Self {
            part: MatrixPart {
                component_id: WeightId::new(format!("component.gdn-q8.{salt}")).unwrap(),
                format: MatrixFormat::Block(format),
                rows: rows as u32,
                columns: HIDDEN as u32,
                output_offset: offset as u32,
                transform: None,
                signs_region: None,
            },
            salt,
            scale_shift,
            dense,
            bytes: Guarded::new(stream, &bytes, 0xab),
        }
    }

    fn reference(&self, input: &[f16], col: usize, quantized: bool) -> (f64, f64) {
        let dot = match (quantized, self.part.format) {
            (true, MatrixFormat::Block(GgufBlockFormat::Q4K)) => Some(dot_reference(
                input,
                &[q4_block(col + self.salt, self.scale_shift)],
            )),
            (true, MatrixFormat::Block(GgufBlockFormat::Q5K)) => Some(dot_q5(
                input,
                &[q5_block(col + self.salt, self.scale_shift)],
            )),
            _ => None,
        };
        if let Some(dot) = dot {
            // Same conservative Lane/MMA bound as the existing Q8 SwiGLU oracle.
            let nu = ((HIDDEN / 32).div_ceil(4) + 9) as f64 * f64::from(f32::EPSILON);
            (dot.policy, nu / (1.0 - nu) * dot.expanded_abs_terms)
        } else {
            let products = input
                .iter()
                .zip(&self.dense[col * HIDDEN..][..HIDDEN])
                .map(|(x, &w)| x.to_f64() * f64::from(w));
            (
                products.clone().sum(),
                HIDDEN as f64 * f64::from(f32::EPSILON) * products.map(f64::abs).sum::<f64>(),
            )
        }
    }
}

fn assert_half(actual: f16, policy: f64, accumulation: f64, label: &str) {
    let expected = f16::from_f64(policy);
    // Existing native_q8_swiglu stage oracle, unchanged.
    let bound =
        accumulation + 0.0009765625 * (policy.abs() + accumulation) + f16::from_bits(1).to_f64();
    assert!(
        actual.is_finite() && (actual.to_f64() - expected.to_f64()).abs() <= bound,
        "{label}: actual={actual}, policy={policy}, reference={expected}, bound={bound}"
    );
}

fn gamma(operations: f64) -> f64 {
    let nu = operations * f64::from(f32::EPSILON) * 0.5;
    nu / (1.0 - nu)
}

fn gated_error_budget(z: f64, reference: f64) -> f64 {
    // linear_attention.cu uses --use_fast_math. CUDA 12.4.1 Programming Guide,
    // mathematical-functions Tables 13/16 and intrinsic-functions: rsqrt 2 ULP,
    // fast division 2 ULP, __expf (2 + floor(1.173*abs(x))) ULP. Include rounding.
    // https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#intrinsic-functions
    // dim128: one square and seven tree additions; divide/add epsilon and
    // reciprocal sqrt produce eta_r. sigmoid is 1/(1+exp(-z)); the negative-z
    // branch below bounds that actual positive exponential, not exp(z).
    // This is a documented budget on the checked fixture domain, not a proof
    // for all real inputs. FTZ/exp saturation at tiny sigmoid values contributes
    // far below the retained 2e-5 absolute floor for these weights/epsilon.
    let u = f64::from(f32::EPSILON) * 0.5;
    let eta_r = (1.0 + 5.0 * u) / (1.0 - gamma(14.0)).sqrt() - 1.0;
    let t = (-z.abs()).exp();
    let k = 2.0 + (1.173 * z.abs()).floor();
    let eta_e = (2.0 * k + 1.0) * u;
    assert!(eta_e < 1.0);
    let eta_se = if z >= 0.0 {
        t * eta_e / (1.0 + t * (1.0 - eta_e))
    } else {
        eta_e / (1.0 + t - eta_e)
    };
    let eta_s = (1.0 + eta_se) * (1.0 + 5.0 * u) / (1.0 - u) - 1.0;
    let eta_y = (1.0 + eta_r) * (1.0 + eta_s) * (1.0 + gamma(4.0)) - 1.0;
    2.0e-5 + reference.abs() * eta_y
}

#[allow(clippy::too_many_arguments)]
fn project(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    q8: &Q8F32ScaleKernels,
    matrices: &[Matrix],
    input: &Guarded<f16>,
    workspace: &Guarded<u32>,
    rows: usize,
    width: usize,
    quantized: bool,
) -> Guarded<f16> {
    let values = input.read(stream);
    assert_eq!(values.len(), rows * HIDDEN);
    let result = Guarded::new(stream, &vec![f16::NAN; rows * width], f16::from_f32(43.0));
    let parts = matrices.iter().map(|m| m.part.clone()).collect::<Vec<_>>();
    let pointers = matrices
        .iter()
        .map(|m| m.bytes.pointer(stream))
        .collect::<Vec<_>>();
    if quantized {
        q8.launch(
            &functions.native,
            stream,
            &parts,
            &pointers,
            input.pointer(stream),
            result.pointer(stream),
            rows as u32,
            HIDDEN as u32,
            width as u32,
            workspace.pointer(stream),
        )
        .unwrap();
        let packed = pack_rows(&values, rows, HIDDEN);
        let expected = packed
            .scales
            .iter()
            .flat_map(|s| s.to_bits().to_le_bytes())
            .chain(packed.quants.iter().map(|q| *q as u8))
            .collect::<Vec<_>>();
        let actual = workspace
            .read(stream)
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "entire packed activation must match independent CPU policy"
        );
    } else {
        for (part, pointer) in parts.iter().zip(pointers) {
            functions
                .native
                .transformed_linear(
                    stream,
                    input.pointer(stream),
                    pointer,
                    result.pointer(stream),
                    part,
                    rows as u32,
                    width as u32,
                    ElementType::F16,
                    0,
                    0,
                )
                .unwrap();
        }
    }
    let actual = result.read(stream);
    for row in 0..rows {
        for matrix in matrices {
            for col in 0..matrix.part.rows as usize {
                let (policy, bound) =
                    matrix.reference(&values[row * HIDDEN..][..HIDDEN], col, quantized);
                assert_half(
                    actual[row * width + matrix.part.output_offset as usize + col],
                    policy,
                    bound,
                    if quantized {
                        "Q8/strict-leaf projection"
                    } else {
                        "strict projection"
                    },
                );
            }
            matrix.bytes.assert_unchanged(stream);
        }
    }
    assert_eq!(input.read(stream), values, "projection modified its input");
    result
}

struct Frame {
    hidden: Guarded<f32>,
    qkv: Guarded<f16>,
    pack: Guarded<u32>,
}

fn frames(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    q8: &Q8F32ScaleKernels,
    matrices: &[Matrix],
    shape: AttentionShape,
    quantized: bool,
) -> Vec<Frame> {
    frames_with_batches(stream, functions, q8, matrices, shape, quantized, &BATCHES)
}

fn frames_with_batches(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    q8: &Q8F32ScaleKernels,
    matrices: &[Matrix],
    shape: AttentionShape,
    quantized: bool,
    batches: &[[usize; 3]],
) -> Vec<Frame> {
    let weights = (0..HIDDEN)
        .map(|i| f16::from_f32(1.0 + sample(i, 5, 0.015625)))
        .collect::<Vec<_>>();
    let norm = Guarded::new(stream, &weights, f16::from_f32(27.0));
    let mut positions = [0; 3];
    batches
        .iter()
        .map(|counts| {
            let rows: usize = counts.iter().sum();
            let mut hidden = Vec::new();
            for (sequence, &count) in counts.iter().enumerate() {
                for local in 0..count {
                    hidden.extend((0..HIDDEN).map(|i| {
                        1.0001
                            + sample(
                                i + (positions[sequence] + local) * HIDDEN,
                                sequence + 7,
                                0.03125,
                            )
                    }));
                }
                positions[sequence] += count;
            }
            let hidden_gpu = Guarded::new(stream, &hidden, 19.0_f32);
            let normalized =
                Guarded::new(stream, &vec![f16::NAN; rows * HIDDEN], f16::from_f32(23.0));
            launch_rms_norm(
                stream,
                &functions.rms_norm,
                hidden_gpu.pointer(stream),
                norm.pointer(stream),
                normalized.pointer(stream),
                rows as u64,
                HIDDEN as i32,
                shape.epsilon,
            )
            .unwrap();
            let actual = normalized.read(stream);
            for (row, x) in hidden.chunks_exact(HIDDEN).enumerate() {
                let inv = (x.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / HIDDEN as f64
                    + f64::from(shape.epsilon))
                .sqrt()
                .recip();
                for col in 0..HIDDEN {
                    let expected = f64::from(x[col]) * inv * weights[col].to_f64();
                    assert!(
                        (actual[row * HIDDEN + col].to_f64() - expected).abs()
                            <= 0.0005 * expected.abs() + 1.0e-6
                    );
                }
            }
            let pack = Guarded::new(
                stream,
                &vec![
                    0xabab_abab_u32;
                    PackLayout::new(rows as u64, HIDDEN as u64)
                        .unwrap()
                        .total_bytes as usize
                        / 4
                ],
                0xdead_beef,
            );
            let qkv = project(
                stream,
                functions,
                q8,
                matrices,
                &normalized,
                &pack,
                rows,
                shape.qkvzba_features as usize,
                quantized,
            );
            hidden_gpu.assert_unchanged(stream);
            norm.assert_unchanged(stream);
            Frame {
                hidden: hidden_gpu,
                qkv,
                pack,
            }
        })
        .collect()
}

fn shape() -> AttentionShape {
    AttentionShape {
        hidden_size: HIDDEN as u64,
        key_heads: 1,
        value_heads: 2,
        key_head_dim: 128,
        value_head_dim: 128,
        qkv_features: 512,
        value_features: HIDDEN as u64,
        qkvz_features: 768,
        ba_features: 4,
        qkvzba_features: 772,
        conv_kernel: 4,
        conv_state_width: 3,
        epsilon: 1.0e-6,
        layer_index: 0,
        decay_parameterization: GatedDeltaDecayParameterization::NegativeRate,
        value_head_mapping: GatedDeltaValueHeadMapping::InterleavedByKeyHead,
    }
}

fn compose(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    q8: &Q8F32ScaleKernels,
    inputs: &[Frame],
    output_matrix: &[Matrix],
    quantized: bool,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    compose_with_batches(
        stream,
        functions,
        q8,
        inputs,
        output_matrix,
        quantized,
        &BATCHES,
    )
}

fn compose_with_batches(
    stream: &Arc<CudaStream>,
    functions: &AttentionFunctions,
    q8: &Q8F32ScaleKernels,
    inputs: &[Frame],
    output_matrix: &[Matrix],
    quantized: bool,
    batches: &[[usize; 3]],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let shape = shape();
    let norm_values = (0..128)
        .map(|i| 1.0 + sample(i, 13, 0.015625))
        .collect::<Vec<_>>();
    let norm = Guarded::new(stream, &norm_values, 37.0_f32);
    let raw = inputs.iter().map(|frame| &frame.qkv).collect::<Vec<_>>();
    let mut hidden_outputs = Vec::new();
    let (core_outputs, states) = exercise_with_raw(
        stream,
        functions,
        shape,
        batches,
        Some(&raw),
        |index, counts, core, z| {
            let rows: usize = counts.iter().sum();
            let core_values = core.read(stream);
            let z_values = z.read(stream);
            let gated = Guarded::new(stream, &vec![f32::NAN; rows * HIDDEN], 31.0_f32);
            launch_gated_norm(
                stream,
                &functions.gated_norm,
                core.pointer(stream),
                z.pointer(stream),
                norm.pointer(stream),
                gated.pointer(stream),
                rows as u64,
                shape.cuda_shape().unwrap(),
            )
            .unwrap();
            let mut expected = Vec::new();
            for (head, x) in core_values.chunks_exact(128).enumerate() {
                assert!(x.iter().all(|x| x.is_finite()));
                let sum_squares = x.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
                assert!(sum_squares * (1.0 + gamma(8.0)) <= f64::from(f32::MAX));
                let inv = (sum_squares / 128.0 + f64::from(shape.epsilon))
                    .sqrt()
                    .recip();
                expected.extend((0..128).map(|col| {
                    let z = z_values[head * 128 + col].to_f64();
                    assert!(z.is_finite() && norm_values[col].is_finite());
                    f64::from(x[col]) * inv * f64::from(norm_values[col]) * z / (1.0 + (-z).exp())
                }));
            }
            let gated_values = gated.read(stream);
            let mut maximum_error = 0.0_f64;
            let mut maximum_budget = 0.0_f64;
            for (index, (&actual, &reference)) in gated_values.iter().zip(&expected).enumerate() {
                let budget = gated_error_budget(z_values[index].to_f64(), reference);
                let error = (f64::from(actual) - reference).abs();
                assert!(
                    actual.is_finite() && error <= budget,
                    "gated normalization: {actual} vs {reference}, budget={budget}"
                );
                maximum_error = maximum_error.max(error);
                maximum_budget = maximum_budget.max(budget);
            }
            println!(
                "{}",
                serde_json::json!({"kind":"gdn_gated_norm_budget", "q8":quantized,
                "rows":rows,"max_abs_error":maximum_error,"max_budget":maximum_budget})
            );
            // Production reuses z as the F16 output-projection activation.
            launch_cast(
                stream,
                &functions.f32_to_f16,
                gated.pointer(stream),
                z.pointer(stream),
                (rows * HIDDEN) as u64,
            )
            .unwrap();
            assert_eq!(
                z.read(stream),
                gated_values
                    .iter()
                    .map(|x| f16::from_f32(*x))
                    .collect::<Vec<_>>()
            );
            let projected = project(
                stream,
                functions,
                q8,
                output_matrix,
                z,
                &inputs[index].pack,
                rows,
                HIDDEN,
                quantized,
            );
            let result = Guarded::new(stream, &vec![f32::NAN; rows * HIDDEN], 47.0_f32);
            launch_residual(
                stream,
                &functions.residual_add,
                inputs[index].hidden.pointer(stream),
                projected.pointer(stream),
                result.pointer(stream),
                (rows * HIDDEN) as u64,
            )
            .unwrap();
            let expected = inputs[index]
                .hidden
                .read(stream)
                .iter()
                .zip(projected.read(stream))
                .map(|(x, y)| *x + y.to_f32())
                .collect::<Vec<_>>();
            let actual = result.read(stream);
            assert_eq!(actual, expected, "F32 residual boundary");
            hidden_outputs.extend(actual);
            inputs[index].hidden.assert_unchanged(stream);
            norm.assert_unchanged(stream);
        },
    );
    (core_outputs, states, hidden_outputs)
}

fn fixed_raw(
    stream: &Arc<CudaStream>,
    frames: &[Frame],
    batches: &[[usize; 3]],
) -> Vec<Guarded<f16>> {
    let width = shape().qkvzba_features as usize;
    let mut by_sequence: [Vec<f16>; 3] = std::array::from_fn(|_| Vec::new());
    for (frame, counts) in frames.iter().zip(BATCHES) {
        let values = frame.qkv.read(stream);
        let mut start = 0;
        for (sequence, count) in counts.into_iter().enumerate() {
            by_sequence[sequence].extend_from_slice(&values[start..start + count * width]);
            start += count * width;
        }
    }
    let mut positions = [0; 3];
    let result = batches
        .iter()
        .map(|counts| {
            let mut values = Vec::new();
            for (sequence, &count) in counts.iter().enumerate() {
                let start = positions[sequence] * width;
                values.extend_from_slice(&by_sequence[sequence][start..start + count * width]);
                positions[sequence] += count;
            }
            Guarded::new(stream, &values, f16::from_f32(41.0))
        })
        .collect();
    for sequence in 0..3 {
        assert_eq!(positions[sequence] * width, by_sequence[sequence].len());
    }
    result
}

fn error(actual: &[f32], reference: &[f32]) -> serde_json::Value {
    assert_eq!(actual.len(), reference.len());
    let max_abs = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max);
    let squared = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>();
    serde_json::json!({"elements":actual.len(),"max_abs":max_abs,"rms":(squared / actual.len().max(1) as f64).sqrt()})
}

#[test]
#[ignore = "requires SM80+ CUDA; test-only GDN Q8 composition and fixed-QKV state carry"]
fn recurrent_q8_projections_match_policy_and_preserve_fixed_qkv_state_carry_on_cuda() {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.gdn-q8-composition").unwrap(),
            AttentionExecutionPolicy::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let provider = CudaGatedDeltaRecurrentAttentionProvider::new_f32_master(&runtime).unwrap();
    let stream = runtime.context().default_stream();
    let q8 = Q8F32ScaleKernels::load(runtime.context()).unwrap();
    let matrices = [
        Matrix::new(&stream, GgufBlockFormat::Q5K, 512, 0, 0, 0),
        Matrix::new(&stream, GgufBlockFormat::Q4K, 256, 512, 1, 0),
        Matrix::new(&stream, GgufBlockFormat::Q8_0, 2, 768, 2, 0),
        Matrix::new(&stream, GgufBlockFormat::Q8_0, 2, 770, 3, 0),
    ];
    let out = [Matrix::new(&stream, GgufBlockFormat::Q5K, HIDDEN, 0, 4, 6)];
    let strict = frames(&stream, &provider.functions, &q8, &matrices, shape(), false);
    let quantized = frames(&stream, &provider.functions, &q8, &matrices, shape(), true);
    for (strict, quantized) in strict.iter().zip(&quantized) {
        for (strict, quantized) in strict
            .qkv
            .read(&stream)
            .chunks_exact(772)
            .zip(quantized.qkv.read(&stream).chunks_exact(772))
        {
            assert_eq!(
                &strict[768..],
                &quantized[768..],
                "Q8_0 a/b leaves retain strict arithmetic"
            );
        }
    }
    let old = compose(&stream, &provider.functions, &q8, &strict, &out, false);
    let new = compose(&stream, &provider.functions, &q8, &quantized, &out, true);
    // Quantize once, then regroup exactly those F16 QKV values. Re-running the
    // projection with a different Lane/MMA geometry is a different comparison.
    let whole = [[0, 4, 108]];
    let raw = fixed_raw(&stream, &quantized, &whole);
    let refs = raw.iter().collect::<Vec<_>>();
    let whole_result = exercise_with_raw(
        &stream,
        &provider.functions,
        shape(),
        &whole,
        Some(&refs),
        |_, _, _, _| {},
    );
    assert_eq!(new.0, whole_result.0, "fixed QKV split/whole core outputs");
    assert_eq!(
        new.1, whole_result.1,
        "fixed QKV split/whole F32 states including idle slot"
    );
    let flatten = |rows: &[Vec<f32>]| rows.iter().flatten().copied().collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::json!({
            "kind":"gdn_q8_composed_oracle", "rows":[7,8,33,64], "hidden":HIDDEN,
            "qkv_width":772, "key_dim":128, "value_dim":128,
            "output_weight_scale":1.0/64.0,
            "strict_vs_q8_core":error(&flatten(&new.0), &flatten(&old.0)),
            "strict_vs_q8_delta_state":error(&flatten(&new.1), &flatten(&old.1)),
            "strict_vs_q8_hidden":error(&new.2, &old.2),
            "fixed_quantized_qkv_split_whole_exact":true,
            "scope":"independent stage-policy oracles on actual F16 boundaries; not strict equivalence, model quality, provider admission or service performance"
        })
    );
}
