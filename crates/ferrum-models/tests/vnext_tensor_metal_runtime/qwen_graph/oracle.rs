//! Check each real family node against independent semantic arithmetic using
//! its observed preceding BF16 boundary. Does not claim whole-graph bitwise
//! agreement with a differently ordered CPU/CUDA implementation.
use super::*;
use transforms::oracle as transforms_oracle;

fn unsigned(node: &ProgramNode, name: &str) -> usize {
    match &node.attributes[&id::<AttributeId>(name)] {
        SemanticValue::Unsigned(v) => *v as usize,
        other => panic!("unsigned {name}: {other:?}"),
    }
}
fn integers(node: &ProgramNode, name: &str) -> Vec<usize> {
    match &node.attributes[&id::<AttributeId>(name)] {
        SemanticValue::Integers(v) => v.iter().map(|&n| usize::try_from(n).unwrap()).collect(),
        other => panic!("integers {name}: {other:?}"),
    }
}
fn rational(node: &ProgramNode, name: &str) -> CanonicalRational {
    match &node.attributes[&id::<AttributeId>(name)] {
        SemanticValue::Rational(v) => *v,
        other => panic!("rational {name}: {other:?}"),
    }
}
fn dense(x: &Data, weight: &Data, actual: &[f32]) {
    let k = x.spec.dimensions[1] as usize;
    let n = weight.spec.dimensions[0] as usize;
    assert_eq!(weight.spec.dimensions[1] as usize, k);
    assert_eq!(actual.len(), x.values.len() / k * n);
    for (i, &got) in actual.iter().enumerate() {
        let (row, col) = (i / n, i % n);
        let (mut sum, mut absolute) = (0.0_f64, 0.0_f64);
        for j in 0..k {
            let p = x.values[row * k + j] as f64 * weight.values[col * k + j] as f64;
            sum += p;
            absolute += p.abs();
        }
        let nu = (2 * k) as f64 * 2_f64.powi(-24);
        let ref_nu = (2 * k) as f64 * f64::EPSILON;
        let error = (nu / (1.0 - nu) + ref_nu / (1.0 - ref_nu)) * absolute;
        let bound = error + (sum.abs() + error) / 256.0 + 2_f64.powi(-134);
        assert!(
            got.is_finite() && (got as f64 - sum).abs() <= bound,
            "matrix [{row},{col}]: {got} != {sum} +/- {bound}"
        );
    }
}
fn pointwise(kind: MetalTensorOperation, x: f32, p: f32, u: f32, got: f32) {
    use MetalTensorOperation::*;
    let (lo, hi) = if kind == GeluTanh {
        gelu_interval(x)
    } else if matches!(kind, Silu | SiluMul | Tanh) {
        let center = if kind == Tanh {
            libm::tanh(x as f64) as f32
        } else {
            libm::exp(-(x as f64)) as f32
        };
        let (mut a, mut b) = (center, center);
        // MSL tanh <=5 / exp <=4 FP32 ULP, plus reference rounding.
        for _ in 0..if kind == Tanh { 6 } else { 5 } {
            a = a.next_down();
            b = b.next_up();
        }
        let evaluate = |v| {
            if kind == Tanh {
                rounds(v)
            } else {
                let silu = rounds(x / (1.0 + v));
                if kind == SiluMul {
                    rounds(silu * p)
                } else {
                    silu
                }
            }
        };
        let (a, b) = (evaluate(a), evaluate(b));
        (a.min(b), a.max(b))
    } else {
        let y = bf16::from_bits(expected(kind, x, p, u)).to_f32();
        (y, y)
    };
    assert!(
        got.is_finite() && got >= lo && got <= hi,
        "{kind:?}({x},{p},{u}): {got} outside [{lo},{hi}]"
    );
}

pub(super) fn graph(graph: &ModelProgram, data: &BTreeMap<ProgramValueId, Data>) {
    use MetalTensorOperation::*;
    for node in graph.blocks().iter().flat_map(|b| &b.nodes) {
        let kind = MetalTensorOperation::ALL
            .into_iter()
            .find(|op| op.contract().unwrap().descriptor().id == node.operation_id)
            .unwrap();
        let inputs: Vec<_> = node.inputs.iter().map(|id| &data[id]).collect();
        let output = &data[&node.outputs[0]];
        let x = &inputs[0].values;
        let y = &output.values;
        match kind {
            DenseLinear | BlockLinear => dense(inputs[0], inputs[1], y),
            LayerNorm | HeadRmsNorm | ZeroCenteredRmsNorm => {
                let width = if kind == HeadRmsNorm {
                    unsigned(node, "head_dim")
                } else {
                    inputs[0].spec.dimensions[1] as usize
                };
                let eps = rational(node, "epsilon");
                transforms_oracle::norm(
                    kind,
                    x,
                    inputs.get(1).map_or(&[], |v| v.values.as_slice()),
                    width,
                    (eps.numerator() as f64 / eps.denominator() as f64) as f32,
                    y,
                );
            }
            AdjacentRope => transforms_oracle::rope(
                kind,
                x,
                &inputs[1].values,
                &[],
                inputs[0].spec.dimensions[1] as usize,
                unsigned(node, "head_dim"),
                inputs[1].spec.dimensions[0] as usize,
                y,
            ),
            HalfSplitRope => transforms_oracle::rope(
                kind,
                x,
                &inputs[1].values,
                &inputs[2].values,
                inputs[0].spec.dimensions[1] as usize,
                unsigned(node, "head_dim"),
                inputs[0].spec.dimensions[0] as usize,
                y,
            ),
            StructuredAttention | StructuredGqaAttention => {
                let width = unsigned(node, "head_dim");
                let lengths = integers(node, "key_run_lengths");
                let flags = integers(node, "key_run_valid");
                let mut offset = 0;
                let keys = integers(node, "key_run_counts")
                    .into_iter()
                    .map(|n| {
                        let runs = lengths[offset..offset + n]
                            .iter()
                            .zip(&flags[offset..offset + n])
                            .map(|(&n, &v)| (n, v != 0))
                            .collect();
                        offset += n;
                        runs
                    })
                    .collect();
                let case = attention::Case {
                    sequence: unsigned(node, "sequence_length"),
                    width,
                    heads: inputs[0].spec.dimensions[1] as usize / width,
                    kv_heads: inputs[1].spec.dimensions[1] as usize / width,
                    segments: integers(node, "segment_lengths")
                        .into_iter()
                        .zip(integers(node, "segment_bidirectional"))
                        .map(|(n, b)| (n, b != 0))
                        .collect(),
                    keys,
                    scale: rational(node, "scale"),
                };
                case.check(
                    &inputs.iter().map(|v| v.values.clone()).collect::<Vec<_>>(),
                    y,
                );
            }
            RowGather => {
                let batch = unsigned(node, "batch");
                let columns = unsigned(node, "column_count");
                let start = unsigned(node, "column_start");
                let lengths = integers(node, "run_lengths");
                let starts = integers(node, "source_starts");
                let batch_strides = integers(node, "source_batch_strides");
                let row_strides = integers(node, "source_row_strides");
                let mut reference = Vec::new();
                for b in 0..batch {
                    for run in 0..lengths.len() {
                        for row in 0..lengths[run] {
                            let source =
                                (starts[run] + b * batch_strides[run] + row * row_strides[run])
                                    * inputs[0].spec.dimensions[1] as usize
                                    + start;
                            reference.extend_from_slice(&x[source..source + columns]);
                        }
                    }
                }
                assert_eq!(
                    y.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "gather {}",
                    node.id
                );
            }
            BatchedRowConcat => {
                let batch = unsigned(node, "batch");
                let mut reference = Vec::new();
                for b in 0..batch {
                    for input in &inputs {
                        let size = input.values.len() / batch;
                        reference.extend_from_slice(&input.values[b * size..(b + 1) * size]);
                    }
                }
                assert_eq!(y, &reference, "concat {}", node.id);
            }
            RowScatterAdd => {
                let width = inputs[0].spec.dimensions[1] as usize;
                let selected_rows: Vec<_> = integers(node, "destination_starts")
                    .into_iter()
                    .zip(integers(node, "run_lengths"))
                    .flat_map(|(start, length)| start..start + length)
                    .collect();
                let mut reference = x.clone();
                for (destination, update) in selected_rows
                    .iter()
                    .flat_map(|row| row * width..(row + 1) * width)
                    .zip(&inputs[1].values)
                {
                    reference[destination] = rounds(reference[destination] + update);
                }
                assert_eq!(y, &reference, "scatter {}", node.id);
            }
            _ => {
                for (i, (&x, &got)) in x.iter().zip(y).enumerate() {
                    let p = inputs.get(1).map_or(0.0, |v| v.values[i % v.values.len()]);
                    let u = inputs.get(2).map_or(0.0, |v| v.values[i]);
                    pointwise(kind, x, p, u, got);
                }
                assert_eq!(x.len(), y.len());
            }
        }
    }
}
