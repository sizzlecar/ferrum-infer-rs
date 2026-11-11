//! Independent scalar graph oracle. Quantized projections use the existing
//! canonical A8 CPU oracle; attention uses F64 two-pass softmax rather than the
//! candidate kernel's F32 online recurrence. This establishes a declared
//! mathematical tolerance, never Torch/libdevice binary parity.
use super::*;
use ferrum_quantization::comfy_convrot::ComfyQuantizedMatrix;

fn norm(x: &[f32], gamma: &[f32], width: usize, epsilon: f64) -> Vec<f32> {
    x.chunks_exact(width)
        .flat_map(|head| {
            let inverse = (head.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / width as f64
                + epsilon)
                .sqrt()
                .recip();
            head.iter()
                .zip(gamma)
                .map(move |(&v, &g)| ((f64::from(v) * inverse) as f32) * g)
        })
        .collect()
}
fn rotate(x: &[f32], cosine: &[f32], sine: &[f32], columns: usize, d: usize) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let row = i / columns;
            let local = i % d;
            let base = i - local;
            let other = if local < d / 2 {
                -x[base + local + d / 2]
            } else {
                x[base + local - d / 2]
            };
            // Ordinary separate products/addition; no hidden half boundary/FMA.
            (v * cosine[row * d + local]) + (other * sine[row * d + local])
        })
        .collect()
}
fn attention(c: &Qwen3VLTextComfyConfig, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let text = c.text();
    let a = text.architecture();
    let layout = text.layout();
    let s = layout.sequence_length() as usize;
    let d = a.head_dim() as usize;
    let qh = a.query_heads() as usize;
    let kh = a.key_value_heads() as usize;
    let scale = text.scale().unwrap();
    let scale = (scale.numerator() as f64 / scale.denominator() as f64) as f32;
    let mut output = vec![0.0; q.len()];
    for batch in 0..layout.batch() {
        let valid: Vec<bool> = layout.key_runs()[batch]
            .iter()
            .flat_map(|r| std::iter::repeat_n(r.is_valid(), r.length() as usize))
            .collect();
        for query in 0..s {
            for head in 0..qh {
                let start = (batch * s + query) * qh * d + head * d;
                let keys: Vec<_> = (0..=query).filter(|&key| valid[key]).collect();
                if keys.is_empty() {
                    continue;
                }
                let scores: Vec<_> = keys
                    .iter()
                    .map(|&key| {
                        let base = (batch * s + key) * kh * d + (head / (qh / kh)) * d;
                        (0..d)
                            .map(|i| f64::from(q[start + i]) * f64::from(k[base + i]))
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<_> = scores.iter().map(|v| (v - max).exp()).collect();
                let sum: f64 = exp.iter().sum();
                for i in 0..d {
                    output[start + i] = keys
                        .iter()
                        .zip(&exp)
                        .map(|(&key, &weight)| {
                            let base = (batch * s + key) * kh * d + (head / (qh / kh)) * d;
                            f64::from(v[base + i]) * weight / sum
                        })
                        .sum::<f64>() as f32;
                }
            }
        }
    }
    output
}
pub(super) fn execute(
    c: &Qwen3VLTextComfyConfig,
    source: &ComfyCanonicalWeightSource,
    inputs: &BTreeMap<ProgramValueId, Vec<f32>>,
) -> BTreeMap<ProgramValueId, Vec<f32>> {
    let text = c.text();
    let a = text.architecture();
    let rows = text.layout().rows() as usize;
    let epsilon = a.epsilon();
    let epsilon = epsilon.numerator() as f64 / epsilon.denominator() as f64;
    let gamma = |name: &str| {
        source
            .archive()
            .tensor(name)
            .unwrap()
            .bytes()
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect::<Vec<_>>()
    };
    let mut x = inputs[&id::<ProgramValueId>(Family::EMBEDDINGS_ID)].clone();
    if let Some(selection) = c.visual_inputs() {
        let h = a.hidden_size() as usize;
        let visual = &inputs[&id::<ProgramValueId>(Family::VISUAL_EMBEDDINGS_ID)];
        let mut source = 0;
        for run in selection.runs() {
            let destination = run.start() as usize * h;
            let length = run.length() as usize * h;
            x[destination..destination + length].copy_from_slice(&visual[source..source + length]);
            source += length;
        }
    }
    let cosine = &inputs[&id::<ProgramValueId>(Family::COSINE_ID)];
    let sine = &inputs[&id::<ProgramValueId>(Family::SINE_ID)];
    let mut outputs = BTreeMap::new();
    for layer in 0..a.layers() {
        let name = |suffix| super::super::super::weights::external_name(layer, suffix);
        let projection = |suffix, activation: &[f32]| {
            ComfyQuantizedMatrix::from_archive(source.archive(), &name(suffix))
                .unwrap()
                .linear_f32_a8_reference(activation, rows)
                .unwrap()
                .output_f32
        };
        let normalized = norm(
            &x,
            &gamma(&name("input_layernorm.weight")),
            a.hidden_size() as usize,
            epsilon,
        );
        let q = projection("self_attn.q_proj", &normalized);
        let k = projection("self_attn.k_proj", &normalized);
        let v = projection("self_attn.v_proj", &normalized);
        let q = norm(
            &q,
            &gamma(&name("self_attn.q_norm.weight")),
            a.head_dim() as usize,
            epsilon,
        );
        let k = norm(
            &k,
            &gamma(&name("self_attn.k_norm.weight")),
            a.head_dim() as usize,
            epsilon,
        );
        let q = rotate(
            &q,
            cosine,
            sine,
            a.query_columns().unwrap() as usize,
            a.head_dim() as usize,
        );
        let k = rotate(
            &k,
            cosine,
            sine,
            a.key_value_columns().unwrap() as usize,
            a.head_dim() as usize,
        );
        let projected = projection("self_attn.o_proj", &attention(c, &q, &k, &v));
        let residual: Vec<_> = x.iter().zip(projected).map(|(x, y)| *x + y).collect();
        let normalized = norm(
            &residual,
            &gamma(&name("post_attention_layernorm.weight")),
            a.hidden_size() as usize,
            epsilon,
        );
        let gate = projection("mlp.gate_proj", &normalized);
        let up = projection("mlp.up_proj", &normalized);
        let activation: Vec<_> = gate
            .iter()
            .zip(up)
            .map(|(g, u)| (*g / (1.0 + libm::expf(-*g))) * u)
            .collect();
        let down = projection("mlp.down_proj", &activation);
        x = residual.iter().zip(down).map(|(x, y)| *x + y).collect();
        if let Some(selection) = c
            .visual_inputs()
            .filter(|_| layer < Family::DEEPSTACK_IDS.len() as u64)
        {
            let h = a.hidden_size() as usize;
            let delta = &inputs[&id::<ProgramValueId>(Family::DEEPSTACK_IDS[layer as usize])];
            let mut source = 0;
            for run in selection.runs() {
                let destination = run.start() as usize * h;
                let length = run.length() as usize * h;
                for (x, &d) in x[destination..destination + length]
                    .iter_mut()
                    .zip(&delta[source..source + length])
                {
                    *x += d;
                }
                source += length;
            }
        }
        if layer + 1 < a.layers() && text.hidden_state_indices().contains(&(layer + 1)) {
            outputs.insert(
                Family::hidden_state_output_id(layer + 1).unwrap(),
                x.clone(),
            );
        }
    }
    outputs.insert(id(Family::OUTPUT_ID), x.clone());
    if text.hidden_state_indices().last() == Some(&a.layers()) {
        outputs.insert(
            Family::hidden_state_output_id(a.layers()).unwrap(),
            norm(
                &x,
                &gamma(DenseFamily::FINAL_NORM_EXTERNAL_NAME),
                a.hidden_size() as usize,
                epsilon,
            ),
        );
    }
    outputs
}

#[test]
fn scalar_oracle_has_nonzero_residuals_and_distinct_post_final_norm_tap() {
    let c = config(text(2, 2), false);
    let (_directory, source) = archive(&original(&c));
    let raw = inputs(&c);
    let expected = execute(&c, &source, &raw);
    let output = &expected[&id::<ProgramValueId>(Family::OUTPUT_ID)];
    assert!(output.iter().all(|v| v.is_finite()));
    assert_ne!(output, &raw[&id::<ProgramValueId>(Family::EMBEDDINGS_ID)]);
    assert_ne!(
        output,
        &expected[&Family::hidden_state_output_id(1).unwrap()]
    );
    assert_ne!(
        output,
        &expected[&Family::hidden_state_output_id(2).unwrap()]
    );
    assert!(output.iter().any(|v| bf16::from_f32(*v).to_f32() != *v));
}
