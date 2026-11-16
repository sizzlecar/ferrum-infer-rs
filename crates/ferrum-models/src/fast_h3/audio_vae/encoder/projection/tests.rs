use super::*;
use crate::fast_h3::audio_vae::tests::{account, limits};
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

type Tensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;
fn architecture() -> FastH3AudioVaeConfig {
    let mut a = crate::fast_h3::audio_vae::tests::architecture();
    a.latent_dim = 8;
    a.num_attention_heads = 2;
    a.latents_mean = vec![0.125, -0.25];
    a.latents_std = vec![2., 0.5];
    a
}
fn fixture(a: &FastH3AudioVaeConfig) -> (tempfile::TempDir, Arc<SafetensorsArchive>, Tensors) {
    let mut tensors = Tensors::new();
    for entry in entries(a).unwrap() {
        let sources = match entry.kind {
            EntryKind::Original { source, shape } => vec![(source, shape)],
            EntryKind::ConcatVectors { sources, width } => sources
                .into_iter()
                .map(|name| (name, vec![width]))
                .collect(),
            _ => unreachable!("projection has original parameters or ordered concatenation"),
        };
        for (name, shape) in sources {
            let shape: Vec<usize> = shape.into_iter().map(|d| d as usize).collect();
            let mut values = vec![0f32; shape.iter().product()];
            if name == "pre_block.attn.qkv.weight" {
                for part in 0..3 {
                    for c in 0..8 {
                        values[(part * 8 + c) * 8 + c] = [0.125, 0.25, 0.5][part];
                    }
                }
            } else if shape.len() >= 2 {
                for (i, value) in values.iter_mut().enumerate() {
                    *value = ((i * 3 + 1) % 7) as f32 / 16. - 0.125;
                }
            } else if name.ends_with(".weight") {
                for (i, value) in values.iter_mut().enumerate() {
                    *value = 0.75 + i as f32 / 32.;
                }
            } else if name != "pre_block.attn.zero_k_bias" {
                for (i, value) in values.iter_mut().enumerate() {
                    *value = (i % 3) as f32 / 32. - 0.03125;
                }
            }
            tensors.insert(name, (shape, values));
        }
    }
    let bytes: BTreeMap<_, Vec<_>> = tensors
        .iter()
        .map(|(name, (_, values))| (name, values.iter().flat_map(|v| v.to_le_bytes()).collect()))
        .collect();
    let views = tensors
        .iter()
        .map(|(name, (shape, _))| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), &bytes[name]).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let dir = tempfile::tempdir().unwrap();
    serialize_to_file(views, &None, &dir.path().join("projection.safetensors")).unwrap();
    let archive = Arc::new(
        SafetensorsArchive::open_single_file(dir.path(), "projection.safetensors").unwrap(),
    );
    (dir, archive, tensors)
}
fn norm(x: &[f64], name: &str, w: &Tensors) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            (v - mean) / (variance + 1e-5).sqrt() * f64::from(w[&format!("{name}.weight")].1[i])
                + f64::from(w[&format!("{name}.bias")].1[i])
        })
        .collect()
}
fn affine(x: &[f64], name: &str, w: &Tensors) -> Vec<f64> {
    let weight = &w[&format!("{name}.weight")].1;
    let bias = &w[&format!("{name}.bias")].1;
    weight
        .chunks_exact(x.len())
        .zip(bias)
        .map(|(row, &b)| {
            x.iter()
                .zip(row)
                .map(|(&x, &w)| x * f64::from(w))
                .sum::<f64>()
                + f64::from(b)
        })
        .collect()
}
fn oracle(input: &[f32], w: &Tensors) -> Vec<f64> {
    let rows: Vec<_> = input
        .chunks_exact(8)
        .map(|x| x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>())
        .collect();
    let normed: Vec<_> = rows.iter().map(|x| norm(x, "pre_block.norm1", w)).collect();
    let qkv: Vec<_> = normed
        .iter()
        .map(|x| {
            w["pre_block.attn.qkv.weight"]
                .1
                .chunks_exact(8)
                .enumerate()
                .map(|(i, weights)| {
                    x.iter()
                        .zip(weights)
                        .map(|(&x, &w)| x * f64::from(w))
                        .sum::<f64>()
                        + f64::from(
                            w[[
                                "pre_block.attn.q_bias",
                                "pre_block.attn.zero_k_bias",
                                "pre_block.attn.v_bias",
                            ][i / 8]]
                                .1[i % 8],
                        )
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let mut output = Vec::new();
    for row in 0..6 {
        let mut attention = vec![0f64; 8];
        for head in 0..2 {
            let keys = row / 3 * 3..=row;
            let mut scores: Vec<_> = keys
                .clone()
                .map(|k| {
                    (0..4)
                        .map(|j| qkv[row][head * 4 + j] * qkv[k][8 + head * 4 + j])
                        .sum::<f64>()
                        / 2.
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for s in &mut scores {
                *s = (*s - max).exp();
            }
            let total = scores.iter().sum::<f64>();
            for (k, score) in keys.zip(scores) {
                for j in 0..4 {
                    attention[head * 4 + j] += score / total * qkv[k][16 + head * 4 + j];
                }
            }
        }
        let mean: Vec<_> = (0..4)
            .map(|j| (attention[j] + attention[4 + j]) / 2.)
            .collect();
        let pool = [(mean[0] + mean[1]) / 2., (mean[2] + mean[3]) / 2.];
        let attn = affine(&pool, "pre_block.attn.proj", w);
        let direct = affine(&norm(&rows[row], "pre_block.norm3", w), "pre_block.proj", w);
        let residual: Vec<_> = direct.iter().zip(attn).map(|(a, b)| a + b).collect();
        let normalized = norm(
            &norm(&residual, "pre_block.norm2", w),
            "pre_block.mlp.norm",
            w,
        );
        let gate = affine(&normalized, "pre_block.mlp.w0", w);
        let up = affine(&normalized, "pre_block.mlp.w1", w);
        let hidden: Vec<_> = gate
            .into_iter()
            .zip(up)
            .map(|(x, u)| {
                0.5 * x
                    * (1.
                        + ((2. / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
                    * u
            })
            .collect();
        let mlp = affine(&hidden, "pre_block.mlp.w2", w);
        let x: Vec<_> = residual.into_iter().zip(mlp).map(|(a, b)| a + b).collect();
        output.extend(affine(&x, "mean_proj", w));
    }
    output
}
fn input(
    prepared: &PreparedFastH3AudioEncoderProjection,
    host: &HostMemoryAccount,
    values: &[f32],
) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
        .unwrap();
    for (dst, v) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(PreparedFastH3AudioEncoderProjection::INPUT_ID).unwrap(),
        prepared.input_spec().clone(),
        bytes.freeze(),
    )
    .unwrap()
}

#[test]
fn posterior_mode_matches_independent_f64_graph_and_keeps_causal_stereo_boundaries() {
    let a = architecture();
    let (_dir, archive, weights) = fixture(&a);
    let host = account(16 << 20);
    let expected_backing =
        PreparedFastH3AudioEncoderProjection::host_weight_backing_bytes(&a, &host).unwrap();
    let prepared =
        PreparedFastH3AudioEncoderProjection::prepare(&a, archive, 2, 3, &host, limits()).unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), expected_backing);
    let composition =
        ReferenceVNextComposition::create(id("device.audio.projection").unwrap()).unwrap();
    let d = composition.runtime().descriptor();
    let compiled = CompiledTensorComponent::compile(
        prepared.family().clone(),
        composition.registry(),
        composition.catalog(),
        &TensorComponentPolicy::new(
            d.total_memory_bytes,
            d.dynamic_storage_profiles.iter().copied().collect(),
        )
        .unwrap(),
        composition.weight_materializers(),
        prepared.compile_options(),
    )
    .unwrap();
    let mut session = compiled
        .initialize(
            composition.runtime().clone(),
            host.clone(),
            ComponentWeightSource::Stored(&prepared),
            ComponentInitializationConfig {
                run_id: id("run.projection.init").unwrap(),
                request_id: id("request.projection.init").unwrap(),
                transaction_id: id("transaction.projection.init").unwrap(),
                static_policy: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let mut values: Vec<_> = (0..48)
        .map(|i| ((i * 11 + 3) % 31) as f32 / 16. - 0.75)
        .collect();
    let outputs = session
        .execute(
            id("run.projection.first").unwrap(),
            id("request.projection.first").unwrap(),
            vec![input(&prepared, &host, &values)],
        )
        .unwrap();
    let got: Vec<_> = outputs[0]
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let expected = oracle(&values, &weights);
    for (i, (&got, expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(got) - expected).abs() < 2e-5,
            "row/channel {i}: {got} != {expected}"
        );
    }
    let normalized = prepared.normalized_mode(&outputs[0], &host).unwrap();
    assert_eq!(normalized.tensor().dimensions(), [2, 2, 3]);
    for b in 0..2 {
        for c in 0..2 {
            for t in 0..3 {
                let dst = ((b * 2 + c) * 3 + t) * 4;
                let value = f32::from_le_bytes(
                    normalized.bytes().bytes()[dst..dst + 4].try_into().unwrap(),
                );
                assert_eq!(
                    value,
                    (got[(b * 3 + t) * 2 + c] - a.latent_mean()[c]) / a.latent_std()[c]
                );
            }
        }
    }
    values[2 * 8] += 11.;
    values[5 * 8 + 3] -= 7.;
    let changed = session
        .execute(
            id("run.projection.changed").unwrap(),
            id("request.projection.changed").unwrap(),
            vec![input(&prepared, &host, &values)],
        )
        .unwrap();
    for row in [0, 1, 3, 4] {
        assert_eq!(
            &outputs[0].bytes().bytes()[row * 8..row * 8 + 8],
            &changed[0].bytes().bytes()[row * 8..row * 8 + 8]
        );
    }
    assert_ne!(outputs[0].bytes().bytes(), changed[0].bytes().bytes());
    assert!(session.close().is_ok());
    drop(outputs);
    drop(changed);
    drop(normalized);
    drop(prepared);
    drop(compiled);
    drop(composition);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
