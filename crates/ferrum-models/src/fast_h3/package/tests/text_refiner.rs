use super::input_projection::{close, execute, host, managed, prepare, session, Counting};
use super::*;
use crate::fast_h3::*;
use ferrum_execution::{component::ComponentWeightSource, TensorExecutionOutput};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_runtime::*;
use half::bf16;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn refiner_limits() -> FastH3TextRefinerLimits {
    FastH3TextRefinerLimits {
        maximum_rows: 64,
        maximum_requests: 4,
        maximum_layers: 2,
        maximum_metadata_bytes: 65536,
        maximum_stored_bytes: 1 << 20,
    }
}
fn two_layer_config() -> FastH3TransformerConfig {
    let mut t: serde_json::Value =
        serde_json::from_slice(include_bytes!("../../fixtures/transformer.json")).unwrap();
    for (k, v) in [
        ("hidden_size", 32),
        ("num_attention_heads", 3),
        ("attention_head_dim", 16),
        ("ffn_dim", 48),
        ("freq_dim", 8),
        ("in_channels", 2),
        ("audio_in_channels", 4),
        ("num_layers", 2),
        ("num_refiner_layers", 2),
        ("text_dim", 16),
        ("time_embed_dim", 16),
        ("time_embed_hidden_dim", 32),
        ("rope_freq_dim", 2),
    ] {
        t[k] = v.into();
    }
    FastH3Checkpoint::parse(
        include_bytes!("../../fixtures/inference.json"),
        &serde_json::to_vec(&t).unwrap(),
        include_bytes!("../../fixtures/video_scheduler.json"),
        include_bytes!("../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap()
    .transformer()
    .clone()
}
fn fixture() -> Fixture {
    let mut f = Fixture::new();
    let second = f
        .dense
        .iter()
        .filter(|(n, _)| n.starts_with("token_refiner.refiner_blocks.0."))
        .map(|(n, t)| {
            (
                n.replace("refiner_blocks.0.", "refiner_blocks.1."),
                t.clone(),
            )
        })
        .collect::<Vec<_>>();
    f.dense.extend(second);
    for (name, (_, shape, bytes)) in &mut f.dense {
        if !name.starts_with("token_refiner.") {
            continue;
        }
        let seed = name.bytes().fold(0usize, |n, b| n + usize::from(b));
        for (i, chunk) in bytes.chunks_exact_mut(2).enumerate() {
            let x = if shape.len() == 1 {
                0.75 + (i % 7) as f32 / 16.
            } else {
                (((i * 13 + seed * 7) % 29) as f32 - 14.) / 64.
            };
            chunk.copy_from_slice(&bf16::from_f32(x).to_bits().to_le_bytes());
        }
    }
    f
}

// Test-only observability adds Host outputs to the original program. Every
// operation, weight binding and input remains the real model component.
struct CaptureFamily(FastH3TextRefinerFamily);
impl ModelFamilyProvider for CaptureFamily {
    type Config = FastH3TextRefinerConfig;
    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.0.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        self.0.validate_config_identity(r, c)
    }
    fn validated_external_metadata_id(
        &self,
        r: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.0.validated_external_metadata_id(r, c)
    }
    fn parse_config(&self, r: &serde_json::Value) -> Result<Self::Config, VNextError> {
        self.0.parse_config(r)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.0.weights(c)
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        self.0.numerical_profiles(c)
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let b = self.0.semantic_program(c, p)?;
        let outputs = b
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| n.outputs.clone())
            .collect();
        ModelProgram::new(
            b.family_id().clone(),
            b.inputs().to_vec(),
            b.blocks().to_vec(),
            b.states().to_vec(),
            b.weights().to_vec(),
            outputs,
        )?
        .with_input_tensor_specs(b.input_tensor_specs().clone())
    }
    fn semantic_metadata(&self, c: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        self.0.semantic_metadata(c)
    }
}
fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
        .collect()
}
fn round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn bytes(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}
fn compare(label: &str, actual: &[f32], expected: &[f32], budget: u16) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let ordered = |x: f32| {
            let b = bf16::from_f32(x).to_bits();
            if b & 0x8000 != 0 {
                !b
            } else {
                b | 0x8000
            }
        };
        assert!(
            a.is_finite() && ordered(a).abs_diff(ordered(e)) <= budget,
            "{label}[{i}]: {a} != {e}, BF16 ULP budget {budget}"
        );
    }
}
fn norm(x: &[f32], g: &[f32], eps: f64) -> Vec<f32> {
    x.chunks_exact(g.len())
        .flat_map(|row| {
            let mean = row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / g.len() as f64;
            let inv = 1.0f32 / ((mean as f32) + (eps as f32)).sqrt();
            row.iter()
                .zip(g)
                .map(move |(&x, &g)| round(round(x * inv) * g))
        })
        .collect()
}
fn linear(x: &[f32], w: &[f32], k: usize) -> Vec<f32> {
    x.chunks_exact(k)
        .flat_map(|row| {
            w.chunks_exact(k).map(move |w| {
                round(
                    row.iter()
                        .zip(w)
                        .map(|(&x, &w)| f64::from(x) * f64::from(w))
                        .sum::<f64>() as f32,
                )
            })
        })
        .collect()
}
fn attention(q: &[f32], k: &[f32], v: &[f32], lengths: &[u64], width: usize, d: usize) -> Vec<f32> {
    let mut result = vec![0.; q.len()];
    let mut start = 0;
    let scale = (1.0 / (d as f64).sqrt()) as f32;
    for &length in lengths {
        let end = start + length as usize;
        for row in start..end {
            for head in 0..width / d {
                let base = row * width + head * d;
                let logits = (start..end)
                    .map(|j| {
                        (0..d)
                            .map(|c| {
                                f64::from(q[base + c]) * f64::from(k[j * width + head * d + c])
                            })
                            .sum::<f64>()
                            * f64::from(scale)
                    })
                    .collect::<Vec<_>>();
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probs = logits.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                let sum = probs.iter().sum::<f64>();
                for c in 0..d {
                    result[base + c] = round(
                        (probs
                            .iter()
                            .enumerate()
                            .map(|(j, p)| p * f64::from(v[(start + j) * width + head * d + c]))
                            .sum::<f64>()
                            / sum) as f32,
                    );
                }
            }
        }
        start = end;
    }
    result
}
fn verify(
    f: &Fixture,
    t: &FastH3TransformerConfig,
    c: &FastH3TextRefinerConfig,
    input: &[f32],
    outputs: &[TensorExecutionOutput],
) {
    let capture = outputs
        .iter()
        .map(|o| (o.value_id().as_str().to_owned(), decode(o.bytes().bytes())))
        .collect::<BTreeMap<_, _>>();
    let h = t.hidden_size();
    let inner = t.attention_heads() * t.attention_head_dim();
    let ffn = t.ffn_dim();
    let mut previous = input;
    let mut staging_witnesses = 0;
    for layer in 0..t.refiner_layers() {
        let value =
            |s: &str| capture[&format!("value.fast_h3.text_refiner.layer{layer}.{s}")].as_slice();
        let weight = |s: &str| {
            let s = s
                .replace("attn.to_out", "attn.to_out.0")
                .replace("ff.fc_in", "ff.net.0.proj")
                .replace("ff.fc_out", "ff.net.2");
            decode(&f.dense[&format!("token_refiner.refiner_blocks.{layer}.{s}")].2)
        };
        compare(
            "norm1",
            value("norm1"),
            &norm(previous, &weight("norm1.weight"), t.norm_eps()),
            1,
        );
        for s in ["q", "k", "v"] {
            compare(
                s,
                value(s),
                &linear(value("norm1"), &weight(&format!("attn.to_{s}.weight")), h),
                1,
            );
        }
        for s in ["q", "k"] {
            compare(
                "qk norm",
                value(&format!("{s}_norm")),
                &norm(
                    value(s),
                    &weight(&format!("attn.norm_{s}.weight")),
                    t.qk_norm_eps(),
                ),
                1,
            );
        }
        compare(
            "attention",
            value("attention"),
            &attention(
                value("q_norm"),
                value("k_norm"),
                value("v"),
                c.request_lengths(),
                inner,
                t.attention_head_dim(),
            ),
            1,
        );
        compare(
            "attention_out",
            value("attention_out"),
            &linear(value("attention"), &weight("attn.to_out.weight"), inner),
            1,
        );
        compare(
            "residual1",
            value("residual1"),
            &previous
                .iter()
                .zip(value("attention_out"))
                .map(|(&x, &a)| round(x + a))
                .collect::<Vec<_>>(),
            0,
        );
        compare(
            "norm2",
            value("norm2"),
            &norm(value("residual1"), &weight("norm2.weight"), t.norm_eps()),
            1,
        );
        compare(
            "fc_in",
            value("fc_in"),
            &linear(value("norm2"), &weight("ff.fc_in.weight"), h),
            1,
        );
        let mut swiglu = Vec::new();
        for row in value("fc_in").chunks_exact(2 * ffn) {
            for (&v, &g) in row[..ffn].iter().zip(&row[ffn..]) {
                let silu = g / (1.0 + (-g).exp());
                let staged = round(round(silu) * v);
                staging_witnesses += usize::from(staged.to_bits() != round(silu * v).to_bits());
                swiglu.push(staged);
            }
        }
        compare("eager SwiGLU", value("swiglu"), &swiglu, 0);
        compare(
            "fc_out",
            value("fc_out"),
            &linear(value("swiglu"), &weight("ff.fc_out.weight"), ffn),
            1,
        );
        compare(
            "residual2",
            value("residual2"),
            &value("residual1")
                .iter()
                .zip(value("fc_out"))
                .map(|(&x, &f)| round(x + f))
                .collect::<Vec<_>>(),
            0,
        );
        previous = value("residual2");
    }
    assert!(
        staging_witnesses > 0,
        "fixture must distinguish eager BF16 SiLU from main-block fused F32 SwiGLU"
    );
    compare(
        "final norm",
        &capture[FastH3TextRefinerFamily::OUTPUT_ID],
        &norm(
            previous,
            &decode(&f.dense["token_refiner.final_norm.weight"].2),
            t.final_norm_eps(),
        ),
        1,
    );
}

#[test]
fn two_dense_refiner_layers_execute_original_sources_with_staged_oracles_and_request_isolation() {
    let f = fixture();
    let t = two_layer_config();
    let (_dir, p) = f.open_with_config(&t, limits());
    let p = p.unwrap();
    let c = FastH3TextRefinerConfig::from_package(&p, vec![2, 3], refiner_limits()).unwrap();
    let account = host(4 << 20);
    let composition = ReferenceVNextComposition::create_with_capacity(
        id("device.reference.text_refiner"),
        4 << 20,
    )
    .unwrap();
    let source = FastH3TextRefinerSource::new(&p, &c, refiner_limits()).unwrap();
    let counted = Counting {
        source: &source,
        reads: AtomicUsize::new(0),
    };
    let prepared = prepare(
        CaptureFamily(FastH3TextRefinerFamily::new(refiner_limits()).unwrap()),
        &c,
        FastH3TextRefinerFamily::PROFILE_ID,
    );
    assert!(prepared
        .program()
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .all(|n| !n.operation_id.as_str().contains("rope")
            && !n.operation_id.as_str().contains("nvfp4")
            && !n.operation_id.as_str().contains("sparse")));
    let mut s = session(
        &composition,
        prepared,
        c.compile_options().unwrap(),
        ComponentWeightSource::Stored(&counted),
        &account,
    );
    assert_eq!(counted.reads.load(Ordering::SeqCst), 21);
    let input = (0..5 * t.hidden_size())
        .map(|i| round(((i * 7 % 43) as f32 - 21.) / 16.))
        .collect::<Vec<_>>();
    let mut first = None;
    for iteration in 0..2 {
        let mut x = input.clone();
        if iteration == 1 {
            for v in &mut x[2 * t.hidden_size()..] {
                *v = round(*v * 3. + 1.);
            }
        }
        let outputs = execute(
            &mut s,
            vec![managed(
                &account,
                FastH3TextRefinerFamily::INPUT_ID,
                5,
                t.hidden_size() as u64,
                ElementType::Bf16,
                &bytes(&x),
            )],
        );
        verify(&f, &t, &c, &x, &outputs);
        let final_bytes = outputs
            .iter()
            .find(|o| o.value_id().as_str() == FastH3TextRefinerFamily::OUTPUT_ID)
            .unwrap()
            .bytes()
            .bytes();
        if let Some(ref expected) = first {
            assert_eq!(&final_bytes[..2 * t.hidden_size() * 2], expected);
        } else {
            first = Some(final_bytes[..2 * t.hidden_size() * 2].to_vec());
        }
        assert!(outputs.iter().all(|o| account.owns(o.bytes())));
    }
    assert_eq!(counted.reads.load(Ordering::SeqCst), 21);
    close(s);
    let single = FastH3TextRefinerConfig::from_package(&p, vec![2], refiner_limits()).unwrap();
    let prepared = prepare(
        FastH3TextRefinerFamily::new(refiner_limits()).unwrap(),
        &single,
        FastH3TextRefinerFamily::PROFILE_ID,
    );
    let mut s = session(
        &composition,
        prepared,
        single.compile_options().unwrap(),
        ComponentWeightSource::Stored(&source),
        &account,
    );
    let out = execute(
        &mut s,
        vec![managed(
            &account,
            FastH3TextRefinerFamily::INPUT_ID,
            2,
            t.hidden_size() as u64,
            ElementType::Bf16,
            &bytes(&input[..2 * t.hidden_size()]),
        )],
    );
    close(s);
    assert_eq!(out[0].bytes().bytes(), first.unwrap());
    assert!(account.owns(out[0].bytes()));
    drop(out);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
}

#[test]
fn refiner_configuration_rejects_source_substitution_partition_and_capacity_violations() {
    let f = fixture();
    let t = two_layer_config();
    let (_dir, p) = f.open_with_config(&t, limits());
    let p = p.unwrap();
    let c = FastH3TextRefinerConfig::from_package(&p, vec![2, 3], refiner_limits()).unwrap();
    let family = FastH3TextRefinerFamily::new(refiner_limits()).unwrap();
    for field in ["layers", "hidden", "head_dim"] {
        let mut raw = serde_json::to_value(&c).unwrap();
        raw[field] = 0.into();
        assert!(family.parse_config(&raw).is_err());
    }
    let mut raw = serde_json::to_value(&c).unwrap();
    raw["requests"] = serde_json::json!([2, 0]);
    assert!(family.parse_config(&raw).is_err());
    raw = serde_json::to_value(&c).unwrap();
    raw["requests"] = serde_json::json!([u64::MAX, 1]);
    let bad: FastH3TextRefinerConfig = serde_json::from_value(raw).unwrap();
    assert!(bad.compile_options().is_err());
    raw = serde_json::to_value(&c).unwrap();
    raw["components"].as_array_mut().unwrap().swap(0, 1);
    let bad: FastH3TextRefinerConfig = serde_json::from_value(raw).unwrap();
    assert!(FastH3TextRefinerSource::new(&p, &bad, refiner_limits()).is_err());
    let mut bound = refiner_limits();
    bound.maximum_stored_bytes = 1;
    assert!(FastH3TextRefinerConfig::from_package(&p, vec![2], bound).is_err());
    bound = refiner_limits();
    bound.maximum_layers = 1;
    assert!(FastH3TextRefinerConfig::from_package(&p, vec![2], bound).is_err());
    bound = refiner_limits();
    bound.maximum_requests = 1;
    assert!(FastH3TextRefinerConfig::from_package(&p, vec![2, 3], bound).is_err());
}
