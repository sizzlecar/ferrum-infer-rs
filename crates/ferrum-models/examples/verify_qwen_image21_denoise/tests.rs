use super::*;
use anyhow::Result;
use ferrum_diffusion::{Bf16EulerProfile, FlowMatchEulerSchedule, ManagedBf16Euler};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::qwen_image21::{
    ExpectedWeights, QwenImage21Config, QwenImage21T2iLayout, QwenImage21TransformerConfig,
    QwenImage21TransformerFamily as Family, StructureLimits,
};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use half::bf16;
use serde_json::json;
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};

struct CountingSource {
    archive: SafetensorsArchive,
    reads: AtomicUsize,
}
impl WeightComponentSource for CountingSource {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.archive.component(c)
    }
    fn component_segments<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentSegments<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.archive.component_segments(c)
    }
}
fn config() -> QwenImage21TransformerConfig {
    QwenImage21TransformerConfig::new(QwenImage21Config::from_hf_json(&serde_json::to_vec(&json!({
        "_class_name":"QwenImage21Transformer2DModel","num_layers":2,"num_attention_heads":1,"attention_head_dim":8,
        "context_in_dim":8,"in_channels":8,"out_channels":8,"patch_size":1,"mlp_ratio":3,"axes_dims_rope":[2,2,4],"causal_condition":true,"eps":0.75
    })).unwrap(),StructureLimits::default()).unwrap(),QwenImage21T2iLayout::new(2,2,2,vec![2]).unwrap()).unwrap()
}
fn archive(directory: &std::path::Path, config: &QwenImage21TransformerConfig) -> CountingSource {
    let weights =
        ExpectedWeights::from_config(config.transformer(), StructureLimits::default()).unwrap();
    let tensors = weights
        .tensors()
        .values()
        .map(|w| {
            let shape = w
                .dimensions()
                .iter()
                .map(|n| *n as usize)
                .collect::<Vec<_>>();
            let bytes = (0..shape.iter().product())
                .flat_map(|i| {
                    // Distinct active attention/MLP layers and nonzero gates. The test
                    // checks loop ownership/advancement, not an invented full-DiT oracle.
                    let value = if shape.len() == 1 {
                        if w.external_name() == "txt_in.text_norm.weight" {
                            0.0
                        } else {
                            1.0
                        }
                    } else {
                        let row = i / shape[1];
                        let col = i % shape[1];
                        let name = w.external_name();
                        match name {
                            "img_in.weight" | "proj_out.weight" => {
                                if row == col {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            "time_text_embed.timestep_embedder.linear_1.weight"
                            | "time_text_embed.timestep_embedder.linear_2.weight" => {
                                if row == 0 && col == 0 {
                                    1.0
                                } else {
                                    0.0
                                }
                            }
                            "modulation.1.weight" | "norm_out.linear.weight" => {
                                if col == 0 {
                                    0.125
                                } else {
                                    0.0
                                }
                            }
                            _ => {
                                if row % shape[1] == col {
                                    if name.starts_with("transformer_blocks.1.") {
                                        0.25
                                    } else {
                                        0.125
                                    }
                                } else {
                                    0.0
                                }
                            }
                        }
                    };
                    bf16::from_f32(value).to_bits().to_le_bytes()
                })
                .collect::<Vec<_>>();
            (w.external_name().to_owned(), (shape, bytes))
        })
        .collect::<BTreeMap<_, _>>();
    let views = tensors
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(views, &None, &directory.join("model.safetensors"))
        .unwrap();
    CountingSource {
        archive: SafetensorsArchive::open(directory).unwrap(),
        reads: AtomicUsize::new(0),
    }
}
fn inputs(
    config: &QwenImage21TransformerConfig,
    host: &HostMemoryAccount,
    budget: &inputs::HostBudget,
    directory: &std::path::Path,
) -> inputs::Inputs {
    let latent = (0..budget.inputs[0] / 2)
        .flat_map(|i| {
            bf16::from_f32(if i % 2 == 0 { 1.0 } else { -1.0 })
                .to_bits()
                .to_le_bytes()
        })
        .collect::<Vec<_>>();
    let text = (0..budget.inputs[1] / 2)
        .flat_map(|i| {
            bf16::from_f32(if i % 3 == 0 { 0.5 } else { -0.25 })
                .to_bits()
                .to_le_bytes()
        })
        .collect::<Vec<_>>();
    std::fs::write(directory.join("latent.bf16"), &latent).unwrap();
    std::fs::write(directory.join("text.bf16"), &text).unwrap();
    let metadata = inputs::TextMetadata {
        sha256: artifacts::sha256(&text),
        bytes: text.len(),
        shape: vec![2, 8],
        dtype: "BF16".into(),
        text_tokens: 2,
        valid_text_lengths: vec![2],
        nonfinite_count: 0,
    };
    inputs::Inputs::read(
        config,
        host,
        budget,
        &directory.join("latent.bf16"),
        &directory.join("text.bf16"),
        &metadata,
    )
    .unwrap()
}

#[test]
fn two_nonzero_managed_dit_steps_share_initialized_weights_and_retire() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let cfg = config();
    let source = archive(dir.path(), &cfg);
    let budget = inputs::HostBudget::new(&cfg, 128, 1 << 20)?;
    let host = budget.account()?;
    let input = inputs(&cfg, &host, &budget, dir.path());
    let original = artifacts::sha256(input.sample.bytes().bytes());
    let family = TypedFamilyRegistration::new(Family::new()?).prepare_with_profile(
        &serde_json::to_value(&cfg)?,
        &NumericalProfileId::new(Family::REFERENCE_PROFILE_ID).map_err(anyhow::Error::msg)?,
    )?;
    let c = ReferenceVNextComposition::create_with_capacity(
        shared::id("device.test.denoise")?,
        16 << 20,
    )?;
    let mut reads = Vec::new();
    let result = execution::execute(
        &cfg,
        &family,
        &source,
        shared::StoredComponentRuntime {
            runtime: c.runtime().clone(),
            registry: c.registry(),
            catalog: c.catalog(),
            materializers: c.weight_materializers(),
            capacity_bytes: 16 << 20,
            staging_bytes: 128,
            provenance: json!({"backend":"Reference"}),
        },
        &host,
        &budget,
        input,
        FlowMatchEulerSchedule::new(&[1.0, 0.5], 0.0, 1000.0)?,
        |report, sample| {
            reads.push(source.reads.load(Ordering::SeqCst));
            assert!(sample
                .bytes()
                .bytes()
                .chunks_exact(2)
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7fff != 0));
            assert_ne!(report["input_latent_sha256"], report["latent_sha256"]);
            Ok(())
        },
    )?;
    assert_eq!(reads.len(), 2);
    assert!(reads[0] > 0);
    assert_eq!(reads[0], reads[1]);
    assert_eq!(source.reads.load(Ordering::SeqCst), reads[0]);
    assert_eq!(result.steps[0]["input_latent_sha256"], original);
    assert_eq!(
        result.steps[1]["input_latent_sha256"],
        result.steps[0]["latent_sha256"]
    );
    assert_eq!(result.steps[1]["completed_steps"], 2);
    assert_eq!(result.provenance["node_count"], 64);
    assert_eq!(host.usage()?.allocated_bytes(), budget.inputs[0]);
    drop(result);
    assert_eq!(host.usage()?.allocated_bytes(), 0);
    assert_eq!(host.usage()?.committed_bytes(), 0);
    Ok(())
}

#[test]
fn failed_update_does_not_advance_or_replace_latent_and_retry_is_valid() -> Result<()> {
    let host = HostMemoryAccount::new(HostMemoryLimits::new(
        12,
        HostMemoryClassLimits {
            input_preprocess_bytes: 8,
            output_bytes: 4,
            transfer_staging_bytes: 0,
            weight_backing_bytes: 0,
        },
    )?);
    let tensor = ResolvedTensorSpec::new(
        vec![1, 2],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )?;
    let mut sample = host.allocate_zeroed(HostMemoryClass::InputPreprocess, 4)?;
    sample.as_mut().copy_from_slice(&[0x00, 0x40, 0x80, 0x40]); // [2,4]
    let sample = ManagedHostTensor::new(
        shared::id(Family::LATENTS_ID)?,
        tensor.clone(),
        sample.freeze(),
    )?;
    let mut solver = ManagedBf16Euler::new(
        FlowMatchEulerSchedule::new(&[1.0, 0.5], 0.0, 1000.0)?,
        host.clone(),
        tensor.clone(),
        Bf16EulerProfile::CudaScalarBf16StagedRneV1,
    )?;
    let mut bad = host.allocate_zeroed(HostMemoryClass::Output, 4)?;
    bad.as_mut().copy_from_slice(&[0x80, 0x3f, 0xc1, 0x7f]);
    let bad = bad.freeze();
    assert!(solver.advance(&sample, &tensor, &bad).is_err());
    assert_eq!(solver.step_index(), 0);
    assert_eq!(sample.bytes().bytes(), [0, 0x40, 0x80, 0x40]);
    assert_eq!(host.usage()?.allocated_bytes(), 8);
    drop(bad);
    let mut velocity = host.allocate_zeroed(HostMemoryClass::Output, 4)?;
    velocity.as_mut().copy_from_slice(&[0x80, 0x3f, 0, 0x40]);
    let velocity = velocity.freeze();
    let (_, next) = solver.advance(&sample, &tensor, &velocity)?;
    assert_eq!(next.bytes().bytes(), [0xc0, 0x3f, 0x40, 0x40]); // [1.5,3]
    assert_eq!(solver.step_index(), 1);
    drop(sample);
    drop(velocity);
    drop(next);
    drop(solver);
    assert_eq!(host.usage()?.allocated_bytes(), 0);
    Ok(())
}

#[test]
fn text_hash_failure_releases_atomic_input_backings() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let cfg = config();
    let budget = inputs::HostBudget::new(&cfg, 128, 1 << 20)?;
    let host = budget.account()?;
    let initial = inputs(&cfg, &host, &budget, dir.path());
    drop(initial);
    let metadata = inputs::TextMetadata {
        sha256: "0".repeat(64),
        bytes: budget.inputs[1],
        shape: vec![2, 8],
        dtype: "BF16".into(),
        text_tokens: 2,
        valid_text_lengths: vec![2],
        nonfinite_count: 0,
    };
    assert!(inputs::Inputs::read(
        &cfg,
        &host,
        &budget,
        &dir.path().join("latent.bf16"),
        &dir.path().join("text.bf16"),
        &metadata
    )
    .is_err());
    assert_eq!(host.usage()?.allocated_bytes(), 0);
    assert_eq!(host.usage()?.committed_bytes(), 0);
    assert!(inputs::HostBudget::new(&cfg, 128, budget.total - 1).is_err());
    Ok(())
}
