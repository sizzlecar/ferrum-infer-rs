//! Independent arithmetic and isolation checks for the vision primitives.
use super::*;
use ferrum_execution::component::TensorComponentPolicy;
use transforms::{fixture::Fixture, run_case, spec};

fn uint(attrs: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> usize {
    match attrs[&id::<AttributeId>(name)] {
        SemanticValue::Unsigned(n) => n as usize,
        _ => panic!("{name}"),
    }
}
fn ratio(attrs: &BTreeMap<AttributeId, SemanticValue>, name: &str) -> CanonicalRational {
    match attrs[&id::<AttributeId>(name)] {
        SemanticValue::Rational(n) => n,
        _ => panic!("{name}"),
    }
}
fn same(got: f32, want: f32) {
    if want.is_nan() {
        assert!(got.is_nan());
    } else {
        assert_eq!(got.to_bits(), want.to_bits());
    }
}
fn gelu(x: f32, got: f32) {
    if !x.is_finite() {
        same(got, if x == f32::INFINITY { x } else { f32::NAN });
        return;
    }
    // This independent FP64 erf checks the declared 16-ULP FP32 envelope of
    // the piecewise implementation over *every* BF16 input, including its
    // negative-tail cancellation. It is not an empirical BF16 tolerance.
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let e = libm::erf(z as f64) as f32;
    let (mut lo, mut hi) = (e, e);
    for _ in 0..16 {
        lo = lo.next_down();
        hi = hi.next_up();
    }
    let a = rounds((0.5 * x) * (1.0 + lo.max(-1.0)));
    let b = rounds((0.5 * x) * (1.0 + hi.min(1.0)));
    // MSL allows FP32 subnormals to flush. Permit only the subnormal result
    // class here, rather than widening the normal-range error envelope.
    if x.abs() < f32::MIN_POSITIVE * 4.0 && got == 0.0 {
        return;
    }
    assert!(
        got.is_finite() && got >= a.min(b) && got <= a.max(b),
        "GELU erf x={x:?}: {got:?} outside [{a:?}, {b:?}]"
    );
}

pub(super) fn check(
    kind: MetalTensorOperation,
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    inputs: &[(&ProgramTensorSpec, &[f32])],
    output: &ProgramTensorSpec,
    actual: &[f32],
) {
    use MetalTensorOperation::*;
    let x = inputs[0].1;
    assert_eq!(
        actual.len(),
        output.dimensions.iter().product::<u64>() as usize
    );
    match kind {
        AffineLinear | PatchProjection => {
            let k = inputs[0].0.dimensions[1] as usize;
            let n = inputs[1].0.dimensions[0] as usize;
            let weight = inputs[1].1;
            for (i, &got) in actual.iter().enumerate() {
                let (r, c) = (i / n, i % n);
                let (mut sum, mut absolute) = (0.0_f64, 0.0_f64);
                for j in 0..k {
                    let p = x[r * k + j] as f64 * weight[c * k + j] as f64;
                    sum += p;
                    absolute += p.abs();
                }
                let nu = 2.0 * k as f64 * 2_f64.powi(-24);
                let reference = 2.0 * k as f64 * f64::EPSILON;
                let error = absolute * (nu / (1.0 - nu) + reference / (1.0 - reference));
                let bias = inputs[2].1[c];
                let evaluate = |mut dot| {
                    if kind == PatchProjection {
                        dot = rounds(dot);
                    }
                    rounds(dot + bias)
                };
                let lo = evaluate(((sum - error) as f32).next_down());
                let hi = evaluate(((sum + error) as f32).next_up());
                assert!(
                    got.is_finite() && got >= lo && got <= hi,
                    "{kind:?} [{r},{c}]: {got} outside [{lo},{hi}]"
                );
            }
        }
        AffineLayerNorm => {
            let eps = ratio(attrs, "epsilon");
            transforms::oracle::affine_norm(
                x,
                inputs[1].1,
                inputs[2].1,
                uint(attrs, "hidden_size"),
                (eps.numerator() as f64 / eps.denominator() as f64) as f32,
                actual,
            );
        }
        GeluErf => x.iter().zip(actual).for_each(|(&x, &got)| gelu(x, got)),
        HalfRotary => {
            let columns = inputs[0].0.dimensions[1] as usize;
            let width = uint(attrs, "head_dim");
            for (i, &got) in actual.iter().enumerate() {
                let column = i % width;
                let partner = if column < width / 2 {
                    i + width / 2
                } else {
                    i - width / 2
                };
                let sign = if column < width / 2 { -1.0 } else { 1.0 };
                let table = i / columns * width + column;
                same(
                    got,
                    rounds(x[i] * inputs[1].1[table] + (sign * x[partner]) * inputs[2].1[table]),
                );
            }
        }
        ReshapeCopy => x
            .iter()
            .zip(actual)
            .for_each(|(&x, &got)| assert_eq!(got.to_bits(), rounds(x).to_bits())),
        SegmentedAttention => {
            let SemanticValue::Integers(lengths) = &attrs[&id::<AttributeId>("segment_lengths")]
            else {
                panic!("segments")
            };
            let width = uint(attrs, "head_dim");
            let columns = inputs[0].0.dimensions[1] as usize;
            let mut start = 0;
            for &length in lengths {
                let length = length as usize;
                let end = start + length * columns;
                let case = attention::Case {
                    sequence: length,
                    heads: columns / width,
                    kv_heads: columns / width,
                    width,
                    segments: vec![(length, true)],
                    keys: vec![vec![(length, true)]],
                    scale: ratio(attrs, "scale"),
                };
                // A separate reference problem per image: no shared causal prefix.
                case.check(
                    &inputs
                        .iter()
                        .map(|v| v.1[start..end].to_vec())
                        .collect::<Vec<_>>(),
                    &actual[start..end],
                );
                start = end;
            }
            assert_eq!(start, actual.len());
        }
        _ => panic!("unexpected primitive {kind:?}"),
    }
}

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
) {
    use MetalTensorOperation::*;
    let policy = TensorComponentPolicy::new(
        CAPACITY,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    let matrix = |rows, columns| spec(vec![rows, columns], ElementType::Bf16);
    let vector = |width| spec(vec![width], ElementType::Bf16);
    let attrs = |entries: Vec<(&str, SemanticValue)>| {
        entries
            .into_iter()
            .map(|(k, v)| (id(k), v))
            .collect::<BTreeMap<_, _>>()
    };
    let unsigned = |name, value| (name, SemanticValue::Unsigned(value));
    let data = |spec: &ProgramTensorSpec, port: usize| {
        (0..spec.dimensions.iter().product::<u64>() as usize)
            .map(|i| rounds(((i * 7 + port * 13) % 67) as f32 / 32.0 - 1.0))
            .collect::<Vec<_>>()
    };
    let run = |kind,
               inputs: Vec<ProgramTensorSpec>,
               output: ProgramTensorSpec,
               attributes,
               values: Vec<Vec<f32>>| {
        let mut fixture = Fixture::new(kind, inputs, attributes);
        fixture.output = Some(output.clone());
        run_case(
            runtime,
            registry,
            catalog,
            &policy,
            host,
            &fixture,
            &values,
            |actual| {
                check(
                    kind,
                    &fixture.attributes,
                    &fixture
                        .inputs
                        .iter()
                        .zip(&values)
                        .map(|(s, v)| (s, v.as_slice()))
                        .collect::<Vec<_>>(),
                    &output,
                    actual,
                )
            },
        );
    };
    for kind in [AffineLinear, PatchProjection] {
        let inputs = vec![
            matrix(33, 36),
            if kind == AffineLinear {
                matrix(65, 36)
            } else {
                spec(vec![65, 3, 2, 2, 3], ElementType::Bf16)
            },
            vector(65),
        ];
        let attributes = if kind == AffineLinear {
            attrs(vec![
                unsigned("in_features", 36),
                unsigned("out_features", 65),
            ])
        } else {
            BTreeMap::new()
        };
        let values = inputs.iter().enumerate().map(|(p, s)| data(s, p)).collect();
        run(kind, inputs, matrix(33, 65), attributes, values);
        // The two projection contracts intentionally differ at the dot boundary.
        run(
            kind,
            vec![
                matrix(1, 1),
                if kind == AffineLinear {
                    matrix(1, 1)
                } else {
                    spec(vec![1, 1, 1, 1, 1], ElementType::Bf16)
                },
                vector(1),
            ],
            matrix(1, 1),
            if kind == AffineLinear {
                attrs(vec![
                    unsigned("in_features", 1),
                    unsigned("out_features", 1),
                ])
            } else {
                BTreeMap::new()
            },
            vec![vec![1.0078125], vec![1.5], vec![-1.5]],
        );
    }
    for width in [1, 7, 1152, 4608] {
        let inputs = vec![matrix(3, width), vector(width), vector(width)];
        let mut values: Vec<_> = inputs.iter().enumerate().map(|(p, s)| data(s, p)).collect();
        values[0][..width as usize].fill(2.0);
        run(
            AffineLayerNorm,
            inputs,
            matrix(3, width),
            attrs(vec![
                unsigned("hidden_size", width),
                (
                    "epsilon",
                    SemanticValue::Rational(CanonicalRational::new(1, 1_000_000).unwrap()),
                ),
            ]),
            values,
        );
    }
    let all_bf16: Vec<_> = (0..=u16::MAX)
        .map(|bits| bf16::from_bits(bits).to_f32())
        .collect();
    run(
        GeluErf,
        vec![matrix(256, 256)],
        matrix(256, 256),
        BTreeMap::new(),
        vec![all_bf16.clone()],
    );
    run(
        ReshapeCopy,
        vec![matrix(256, 256)],
        matrix(128, 512),
        attrs(vec![unsigned("rows", 128), unsigned("columns", 512)]),
        vec![all_bf16],
    );
    for width in [2, 6, 72, 256] {
        let inputs = vec![
            matrix(3, width * 2),
            spec(vec![3, width], ElementType::F32),
            spec(vec![3, width], ElementType::F32),
        ];
        let values = inputs
            .iter()
            .enumerate()
            .map(|(p, s)| {
                if p == 0 {
                    data(s, p)
                } else {
                    (0..3 * width)
                        .map(|i| (i as f32 * 0.12345 + p as f32 * 0.4321).sin())
                        .collect()
                }
            })
            .collect();
        run(
            HalfRotary,
            inputs,
            matrix(3, width * 2),
            attrs(vec![unsigned("head_dim", width)]),
            values,
        );
        for poison in 0..4 {
            let inputs = vec![matrix(65, width * 2); 3];
            let mut values: Vec<_> = inputs.iter().enumerate().map(|(p, s)| data(s, p)).collect();
            if poison > 0 {
                values[poison - 1][1] = if poison == 3 { f32::INFINITY } else { f32::NAN };
            }
            run(
                SegmentedAttention,
                inputs,
                matrix(65, width * 2),
                attrs(vec![
                    unsigned("head_dim", width),
                    (
                        "scale",
                        SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
                    ),
                    (
                        "segment_lengths",
                        SemanticValue::Integers(vec![3, 30, 1, 31]),
                    ),
                ]),
                values,
            );
        }
    }
}
