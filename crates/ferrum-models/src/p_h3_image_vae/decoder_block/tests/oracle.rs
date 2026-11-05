//! Independent scalar vertical-block oracle. Raw projection arithmetic comes
//! from canonical INT8/A8 CPU references. F64 norm and two-pass full attention
//! deliberately do not mirror CUDA's strided reductions/online softmax.
use super::*;
use ferrum_quantization::comfy_convrot::ComfyQuantizedMatrix;
fn rms(x: &[f32], gamma: Option<&[f32]>, width: usize, epsilon: f64) -> Vec<f32> {
    x.chunks_exact(width)
        .flat_map(|head| {
            let inverse = (head.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / width as f64
                + epsilon)
                .sqrt()
                .recip();
            head.iter()
                .enumerate()
                .map(move |(i, v)| ((f64::from(*v) * inverse) as f32) * gamma.map_or(1.0, |g| g[i]))
        })
        .collect()
}
fn grouped_slice(
    x: &[f32],
    source_columns: usize,
    group_width: usize,
    start: usize,
    count: usize,
) -> Vec<f32> {
    x.chunks_exact(source_columns)
        .flat_map(|row| {
            row.chunks_exact(group_width)
                .flat_map(|head| head[start..start + count].iter().copied())
        })
        .collect()
}
fn partial_rope(
    x: &[f32],
    cosine: &[f32],
    sine: &[f32],
    columns: usize,
    d: usize,
    rot: usize,
) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            let row = i / columns;
            let local = i % d;
            if local >= rot {
                return *v;
            }
            let base = i - local;
            let other = if local < rot / 2 {
                -x[base + local + rot / 2]
            } else {
                x[base + local - rot / 2]
            };
            (*v * cosine[row * rot + local]) + (other * sine[row * rot + local])
        })
        .collect()
}
fn full_attention(c: &PH3ImageVaeDecoderBlockConfig, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let d = c.architecture().head_dim() as usize;
    let h = c.architecture().hidden_size().unwrap() as usize;
    let sequence = c.sequence_length() as usize;
    let scale = c.scale().unwrap();
    let scale = (scale.numerator() as f64 / scale.denominator() as f64) as f32;
    let mut output = vec![0.0; q.len()];
    for batch in 0..c.batch() as usize {
        for query in 0..sequence {
            for head in 0..h / d {
                let qstart = (batch * sequence + query) * h + head * d;
                let scores: Vec<_> = (0..sequence)
                    .map(|key| {
                        let kstart = (batch * sequence + key) * h + head * d;
                        (0..d)
                            .map(|i| f64::from(q[qstart + i]) * f64::from(k[kstart + i]))
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|v| (v - maximum).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for i in 0..d {
                    output[qstart + i] = (0..sequence)
                        .map(|key| {
                            let start = (batch * sequence + key) * h + head * d;
                            f64::from(v[start + i]) * weights[key] / sum
                        })
                        .sum::<f64>() as f32;
                }
            }
        }
    }
    output
}
pub(in crate::p_h3_image_vae) fn execute(
    c: &PH3ImageVaeDecoderBlockConfig,
    source: &ComfyCanonicalWeightSource,
    values: &BTreeMap<ProgramValueId, Vec<f32>>,
) -> Vec<f32> {
    capture(c, source, values)
        .remove(&id::<ProgramValueId>(Family::OUTPUT_ID))
        .unwrap()
}
pub(super) fn capture(
    c: &PH3ImageVaeDecoderBlockConfig,
    source: &ComfyCanonicalWeightSource,
    values: &BTreeMap<ProgramValueId, Vec<f32>>,
) -> BTreeMap<ProgramValueId, Vec<f32>> {
    let mut outputs = BTreeMap::new();
    let mut record = |name: &str, data: &[f32]| {
        outputs.insert(program::value(name).unwrap(), data.to_vec());
    };
    let a = c.architecture();
    let h = a.hidden_size().unwrap() as usize;
    let m = a.intermediate_size().unwrap() as usize;
    let d = a.head_dim() as usize;
    let rot = a.rotary_dim() as usize;
    let rows = c.rows().unwrap() as usize;
    let epsilon = a.norm_epsilon();
    let epsilon = epsilon.numerator() as f64 / epsilon.denominator() as f64;
    let parameter = |name: &str| -> Vec<f32> {
        source
            .archive()
            .tensor(&weights::external(c, name))
            .unwrap()
            .bytes()
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>()
    };
    let project = |name: &str, x: &[f32]| {
        let raw = ComfyQuantizedMatrix::from_archive(source.archive(), &weights::external(c, name))
            .unwrap()
            .linear_f32_a8_reference(x, rows)
            .unwrap()
            .output_f32;
        let bias_name = format!("{name}.bias");
        let bias = parameter(&bias_name);
        let n = bias.len();
        let biased = raw
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| v + bias[i % n])
            .collect::<Vec<_>>();
        (raw, biased)
    };
    let input = &values[&id::<ProgramValueId>(Family::INPUT_ID)];
    let cosine = &values[&id::<ProgramValueId>(Family::COSINE_ID)];
    let sine = &values[&id::<ProgramValueId>(Family::SINE_ID)];
    let normalized = rms(input, Some(&parameter("norm1.weight")), h, epsilon);
    record("norm1", &normalized);
    let (raw_qkv, qkv) = project("attn.to_qkv", &normalized);
    record("qkv_linear", &raw_qkv);
    record("qkv_biased", &qkv);
    let q = grouped_slice(&qkv, 3 * h, 3 * d, 0, d);
    let k = grouped_slice(&qkv, 3 * h, 3 * d, d, d);
    let v = grouped_slice(&qkv, 3 * h, 3 * d, 2 * d, d);
    record("query", &q);
    record("key", &k);
    record("value", &v);
    let q = partial_rope(&rms(&q, None, d, epsilon), cosine, sine, h, d, rot);
    let k = partial_rope(&rms(&k, None, d, epsilon), cosine, sine, h, d, rot);
    record("query_normalized_rotated", &q);
    record("key_normalized_rotated", &k);
    let raw_attention = full_attention(c, &q, &k, &v);
    record("attention", &raw_attention);
    let attention = raw_attention
        .into_iter()
        .map(|v| {
            if v.is_nan() {
                0.0
            } else {
                v.clamp(-f32::MAX, f32::MAX)
            }
        })
        .collect::<Vec<_>>();
    record("attention_finite", &attention);
    let (raw_branch, branch) = project("attn.to_out", &attention);
    record("attention_linear", &raw_branch);
    record("attention_biased", &branch);
    let scale = parameter("scale1");
    let residual: Vec<_> = input
        .iter()
        .zip(branch)
        .enumerate()
        .map(|(i, (base, branch))| branch.mul_add(scale[i % h], *base))
        .collect();
    record("attention_residual", &residual);
    let normalized = rms(&residual, Some(&parameter("norm2.weight")), h, epsilon);
    record("norm2", &normalized);
    let (raw_w1, w1) = project("ff.w1", &normalized);
    record("w1_linear", &raw_w1);
    record("w1_biased", &w1);
    let gate = grouped_slice(&w1, 2 * m, 2 * m, 0, m);
    let up = grouped_slice(&w1, 2 * m, 2 * m, m, m);
    record("gate", &gate);
    record("up", &up);
    let activated = gate
        .iter()
        .zip(up)
        .map(|(g, u)| (*g / (1.0 + libm::expf(-*g))) * u)
        .collect::<Vec<_>>();
    record("activation", &activated);
    let (raw_branch, branch) = project("ff.w2", &activated);
    record("w2_linear", &raw_branch);
    record("w2_biased", &branch);
    let scale = parameter("scale2");
    let output: Vec<_> = residual
        .iter()
        .zip(branch)
        .enumerate()
        .map(|(i, (base, branch))| branch.mul_add(scale[i % h], *base))
        .collect();
    record("output", &output);
    outputs
}

#[test]
fn per_head_qkv_oracle_uses_real_interleaving_and_normalized_rotary_tail() {
    let packed: Vec<_> = (0..2 * 3 * 4).map(|i| i as f32).collect();
    assert_eq!(
        grouped_slice(&packed, 24, 12, 0, 4),
        [0., 1., 2., 3., 12., 13., 14., 15.]
    );
    assert_eq!(
        grouped_slice(&packed, 24, 12, 4, 4),
        [4., 5., 6., 7., 16., 17., 18., 19.]
    );
    assert_eq!(
        grouped_slice(&packed, 24, 12, 8, 4),
        [8., 9., 10., 11., 20., 21., 22., 23.]
    );
    let x: Vec<_> = (0..64).map(|i| i as f32 / 32.0 - 1.0).collect();
    let n = rms(&x, None, 64, 1.0e-5);
    let out = partial_rope(&n, &[0.0; 48], &[1.0; 48], 64, 64, 48);
    for i in 0..24 {
        assert_eq!(out[i], -n[i + 24]);
        assert_eq!(out[i + 24], n[i]);
    }
    assert_eq!(&out[48..], &n[48..]);
    assert_ne!(&out[48..], &x[48..]);
}
#[test]
fn full_scalar_decoder_has_nonzero_branches_and_isolates_both_requests() {
    let c = config(2);
    let (_directory, source) = archive(&original(&c));
    let values = input_values(&c);
    let expected = execute(&c, &source, &values);
    let input = &values[&id::<ProgramValueId>(Family::INPUT_ID)];
    assert!(expected.iter().all(|v| v.is_finite()));
    assert_ne!(&expected, input);
    assert!(expected.iter().any(|v| bf16::from_f32(*v).to_f32() != *v));
    for sample in 0..2 {
        let one = PH3ImageVaeDecoderBlockConfig::new(
            c.architecture().clone(),
            c.block_index(),
            1,
            c.sequence_length(),
            c.execution_preset(),
        )
        .unwrap();
        let inputs: BTreeMap<_, _> = values
            .iter()
            .map(|(id, data)| {
                let width = if id.as_str() == Family::INPUT_ID {
                    256
                } else {
                    48
                };
                let elements = width * c.sequence_length() as usize;
                (
                    id.clone(),
                    data[sample * elements..(sample + 1) * elements].to_vec(),
                )
            })
            .collect();
        assert_eq!(
            execute(&one, &source, &inputs),
            expected[sample * 7 * 256..(sample + 1) * 7 * 256]
        );
    }
}
