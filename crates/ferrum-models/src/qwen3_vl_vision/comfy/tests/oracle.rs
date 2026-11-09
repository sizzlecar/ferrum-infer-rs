//! Independent sequential matrix oracle. F64 reductions/transcendentals are
//! rounded at explicit F32 operator boundaries; this checks selected bounded
//! fixtures, rather than duplicating the provider's reduction tree/softmax.
use super::*;
use fixture::Original;
fn affine(x: &[f32], rows: usize, k: usize, name: &str, w: &Original) -> Vec<f32> {
    let weight = &w.values[&format!("{name}.weight")];
    let bias = &w.values[&format!("{name}.bias")];
    let n = bias.len();
    (0..rows * n)
        .map(|i| {
            let r = i / n;
            let o = i % n;
            let dot = (0..k)
                .map(|j| f64::from(x[r * k + j]) * f64::from(weight[o * k + j]))
                .sum::<f64>() as f32;
            dot + bias[o]
        })
        .collect()
}
fn norm(x: &[f32], h: usize, name: &str, w: &Original) -> Vec<f32> {
    let gamma = &w.values[&format!("{name}.weight")];
    let beta = &w.values[&format!("{name}.bias")];
    x.chunks_exact(h)
        .flat_map(|row| {
            let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / h as f64;
            let var = row
                .iter()
                .map(|&x| (f64::from(x) - mean).powi(2))
                .sum::<f64>()
                / h as f64;
            let inverse = 1. / (var + 1e-6).sqrt();
            row.iter()
                .enumerate()
                .map(move |(i, &v)| (((f64::from(v) - mean) * inverse) as f32 * gamma[i]) + beta[i])
        })
        .collect()
}
fn gelu(x: Vec<f32>, tanh: bool) -> Vec<f32> {
    x.into_iter()
        .map(|v| {
            let d = f64::from(v);
            let a = if tanh {
                libm::tanh(0.7978845608028654 * (d + 0.044715 * d * d * d))
            } else {
                libm::erf(d * std::f64::consts::FRAC_1_SQRT_2)
            };
            (0.5 * d * (1. + a)) as f32
        })
        .collect()
}
fn add(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    a.into_iter().zip(b).map(|(a, b)| a + b).collect()
}
fn merger(x: &[f32], r: usize, h: usize, name: &str, deep: bool, w: &Original) -> Vec<f32> {
    let merged = h * 4;
    let x = if deep {
        norm(x, merged, &format!("{name}.norm"), w)
    } else {
        norm(x, h, &format!("{name}.norm"), w)
    };
    let x = affine(&x, r / 4, merged, &format!("{name}.linear_fc1"), w);
    affine(
        &gelu(x, false),
        r / 4,
        merged,
        &format!("{name}.linear_fc2"),
        w,
    )
}
pub(in crate::qwen3_vl_vision) fn forward(
    c: &Qwen3VLVisionComfyConfig,
    patches: &[f32],
    positions: &[f32],
    cos: &[f32],
    sin: &[f32],
    w: &Original,
) -> BTreeMap<String, Vec<f32>> {
    let a = c.architecture();
    let r = c.rows().unwrap() as usize;
    let h = a.hidden_size() as usize;
    let d = a.head_dim() as usize;
    let m = a.intermediate_size() as usize;
    let mut x = add(
        affine(
            patches,
            r,
            a.patch_columns().unwrap() as usize,
            "model.visual.patch_embed.proj",
            w,
        ),
        positions.to_vec(),
    );
    let mut outputs = BTreeMap::new();
    for layer in 0..a.depth() {
        let name = format!("model.visual.blocks.{layer}");
        let n = norm(&x, h, &format!("{name}.norm1"), w);
        let qkv = affine(&n, r, h, &format!("{name}.attn.qkv"), w);
        let mut parts = [
            Vec::with_capacity(r * h),
            Vec::with_capacity(r * h),
            Vec::with_capacity(r * h),
        ];
        for row in qkv.chunks_exact(3 * h) {
            for (i, p) in parts.iter_mut().enumerate() {
                p.extend_from_slice(&row[i * h..(i + 1) * h]);
            }
        }
        for p in &mut parts[..2] {
            let before = p.clone();
            for row in 0..r {
                for head in 0..h / d {
                    for lane in 0..d {
                        let index = row * h + head * d + lane;
                        let partner = row * h + head * d + (lane + d / 2) % d;
                        let rotated = if lane < d / 2 {
                            -before[partner]
                        } else {
                            before[partner]
                        };
                        p[index] =
                            before[index] * cos[row * d + lane] + rotated * sin[row * d + lane];
                    }
                }
            }
        }
        let mut attended = vec![0.; r * h];
        for query in 0..r {
            for head in 0..h / d {
                let mut scores = (0..r)
                    .map(|key| {
                        (0..d)
                            .map(|j| {
                                f64::from(parts[0][query * h + head * d + j])
                                    * f64::from(parts[1][key * h + head * d + j])
                            })
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores {
                    *s = libm::exp(*s - max);
                }
                let sum = scores.iter().sum::<f64>();
                for j in 0..d {
                    attended[query * h + head * d + j] = (scores
                        .iter()
                        .enumerate()
                        .map(|(key, &s)| s * f64::from(parts[2][key * h + head * d + j]))
                        .sum::<f64>()
                        / sum) as f32;
                }
            }
        }
        x = add(x, affine(&attended, r, h, &format!("{name}.attn.proj"), w));
        let n = norm(&x, h, &format!("{name}.norm2"), w);
        let n = gelu(affine(&n, r, h, &format!("{name}.mlp.linear_fc1"), w), true);
        x = add(x, affine(&n, r, m, &format!("{name}.mlp.linear_fc2"), w));
        if let Some(i) = a.deepstack_layers().iter().position(|&i| i == layer) {
            outputs.insert(
                Qwen3VLVisionComfyFamily::DEEPSTACK_IDS[i].into(),
                merger(
                    &x,
                    r,
                    h,
                    &format!("model.visual.deepstack_merger_list.{i}"),
                    true,
                    w,
                ),
            );
        }
    }
    outputs.insert(
        Qwen3VLVisionComfyFamily::OUTPUT_ID.into(),
        merger(&x, r, h, "model.visual.merger", false, w),
    );
    outputs
}
