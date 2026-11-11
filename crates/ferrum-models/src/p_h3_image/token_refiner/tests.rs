use super::*;
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use half::bf16;
fn transformer() -> PH3ImageTransformerConfig {
    PH3ImageTransformerConfig::from_hf_json(&serde_json::to_vec(&serde_json::json!({"_class_name":"MiniMaxH3Transformer3DModel","attention_head_dim":128,"audio_in_channels":32,"ffn_dim":512,"final_norm_eps":1e-5,"freq_dim":256,"hidden_size":256,"in_channels":24,"norm_eps":1e-5,"num_attention_heads":4,"num_layers":50,"num_refiner_layers":2,"patch_size":[1,2,2],"qk_norm_eps":1e-5,"rope_freq_dim":16,"rope_theta":10000.,"text_dim":32,"time_embed_dim":2688,"time_embed_hidden_dim":5376})).unwrap()).unwrap()
}
fn limits() -> PH3ImageTokenRefinerLimits {
    PH3ImageTokenRefinerLimits {
        max_metadata_bytes: 1 << 16,
        max_stored_bytes: 8 << 20,
        max_plan_nodes: 1024,
    }
}
type Payloads = BTreeMap<String, (safetensors::Dtype, Vec<usize>, Vec<u8>)>;
fn payloads(c: &PH3ImageTokenRefinerConfig) -> Payloads {
    let mut result = Payloads::new();
    for (name, shape, dtype) in weights::entries(c) {
        let count = shape.iter().product::<u64>() as usize;
        let (dtype, bytes) = match dtype {
            ElementType::Bf16 => {
                let values = (0..count).map(|i| {
                    if name == "condition_proj.weight" {
                        ((i * 7) % 13) as f32 / 256. - 6. / 256.
                    } else if name == "condition_proj.bias" {
                        (i % 7) as f32 / 128. - 3. / 128.
                    } else if name.contains("q_norm") || name.contains("k_norm") {
                        0.0625 + (i % 3) as f32 / 256.
                    } else {
                        0.75 + (i % 5) as f32 / 32.
                    }
                });
                (
                    safetensors::Dtype::BF16,
                    values
                        .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
                        .collect(),
                )
            }
            ElementType::F32 => (
                safetensors::Dtype::F32,
                (0..count)
                    .flat_map(|i| (0.015625 * (1. + (i % 3) as f32 / 8.)).to_le_bytes())
                    .collect(),
            ),
            ElementType::I8 => {
                let (n, k) = (shape[0] as usize, shape[1] as usize);
                let phase = name.bytes().map(usize::from).sum::<usize>();
                let mut values = vec![0u8; count];
                for row in 0..n {
                    for entry in 0..16 {
                        values[row * k + (row * 37 + entry * 53 + phase) % k] =
                            (((row + entry + phase) % 7) as i8 - 3) as u8;
                    }
                }
                (safetensors::Dtype::I8, values)
            }
            _ => unreachable!(),
        };
        result.insert(
            name,
            (
                dtype,
                shape.into_iter().map(|n| n as usize).collect(),
                bytes,
            ),
        );
    }
    for layer in 0..c.layers {
        for module in ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"] {
            let metadata =
                br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}"#.to_vec();
            result.insert(
                format!("token_refiner.blocks.{layer}.{module}.comfy_quant"),
                (safetensors::Dtype::U8, vec![metadata.len()], metadata),
            );
        }
    }
    result
}
struct ArchiveFixture {
    _directory: tempfile::TempDir,
    source: ComfyCanonicalWeightSource,
}
fn archive(payloads: &Payloads) -> ArchiveFixture {
    let directory = tempfile::tempdir().unwrap();
    let views = payloads
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(
        views,
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source =
        ComfyCanonicalWeightSource::new(SafetensorsArchive::open(directory.path()).unwrap())
            .unwrap();
    ArchiveFixture {
        _directory: directory,
        source,
    }
}
#[test]
fn real_two_layer_graph_has_no_rope_or_modulation_and_declares_every_output_shape() {
    let c = PH3ImageTokenRefinerConfig::new(&transformer(), vec![3, 5]).unwrap();
    assert_ne!(c.hidden, c.inner());
    let family = PH3ImageTokenRefinerFamily::new(limits()).unwrap();
    let prepared = TypedFamilyRegistration::new(family).prepare_with_profile(
        &serde_json::to_value(&c).unwrap(),
        &NumericalProfileId::new(PH3ImageTokenRefinerFamily::CUDA_PROFILE_ID).unwrap(),
    );
    assert!(prepared.is_ok(), "{:?}", prepared.err());
    let family = PH3ImageTokenRefinerFamily::new(limits()).unwrap();
    let profiles = family.numerical_profiles(&c).unwrap();
    let profile = profiles
        .resolve(&NumericalProfileId::new(PH3ImageTokenRefinerFamily::CUDA_PROFILE_ID).unwrap())
        .unwrap();
    let body = family.semantic_program(&c, profile).unwrap();
    profile.validate_program(&body).unwrap();
    let nodes = &body.blocks()[0].nodes;
    assert_eq!(nodes.len(), 38);
    assert_eq!(nodes.len(), c.plan_nodes().unwrap());
    let options = c.compile_options().unwrap();
    let expected = body
        .inputs()
        .iter()
        .chain(nodes.iter().flat_map(|n| &n.outputs))
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(expected, options.tensor_specs().keys().cloned().collect());
    for layer in 0..2 {
        for name in ["q_heads", "k_heads", "q_norm_heads", "k_norm_heads"] {
            assert_eq!(
                options.tensor_specs()[&ProgramValueId::new(program::stage(layer, name)).unwrap()]
                    .dimensions,
                vec![32, 128]
            );
        }
        for name in ["q", "k", "v", "q_norm", "k_norm", "attention"] {
            assert_eq!(
                options.tensor_specs()[&ProgramValueId::new(program::stage(layer, name)).unwrap()]
                    .dimensions,
                vec![8, 512]
            );
        }
        let attention = nodes
            .iter()
            .find(|n| {
                n.id.as_str() == format!("node.p_h3_image.token_refiner.layer{layer}.attention")
            })
            .unwrap();
        assert_eq!(
            attention.attributes[&AttributeId::new("segment_lengths").unwrap()],
            SemanticValue::Integers(vec![3, 5])
        );
    }
    let operations = nodes
        .iter()
        .map(|n| n.operation_id.as_str())
        .collect::<BTreeSet<_>>();
    assert!(
        operations.contains(AFFINE_LINEAR_BF16_OPERATION_ID)
            && operations.contains(COMFY_CONVROT_W8A8_BF16_OPERATION_ID)
            && operations.contains(COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID)
    );
    assert!(
        !operations.contains(FUSED_HEAD_RMS_PARTIAL_ROPE_BF16_OPERATION_ID)
            && !operations.contains(ADDCMUL_BF16_OPERATION_ID)
            && !operations.contains(MODULATION_SCALE_BF16_OPERATION_ID)
    );
    assert_eq!(
        body.outputs(),
        &[ProgramValueId::new(PH3ImageTokenRefinerFamily::OUTPUT_ID).unwrap()]
    );
}
#[test]
fn stored_source_retains_original_dense_and_int8_parameters_with_exact_shapes() {
    let c = PH3ImageTokenRefinerConfig::new(&transformer(), vec![7, 11]).unwrap();
    let tensors = payloads(&c);
    let files = archive(&tensors);
    let checkpoint = PH3ImageTokenRefinerCheckpoint::new(&c, &files.source, limits()).unwrap();
    assert_eq!(checkpoint.config(), &c);
    let schema = weights::schema(&c, u64::MAX).unwrap();
    assert_eq!(schema.tensors.len(), 27);
    for component in &schema.components {
        let view = checkpoint.source().component(component).unwrap();
        let original = files
            .source
            .archive()
            .tensor(&component.external_names[0])
            .unwrap();
        assert_eq!(view.bytes().as_ptr(), original.bytes().as_ptr());
        assert_eq!(view.bytes().len(), original.bytes().len());
    }
    let mut wrong = tensors.clone();
    let entry = wrong.get_mut("condition_proj.weight").unwrap();
    entry.0 = safetensors::Dtype::F32;
    entry.2 = vec![0; 4 * (c.hidden * c.text_dim) as usize];
    let files = archive(&wrong);
    assert!(PH3ImageTokenRefinerCheckpoint::new(&c, &files.source, limits()).is_err());
}
#[test]
fn invalid_configuration_and_declared_resource_capacities_fail_admission() {
    let c = PH3ImageTokenRefinerConfig::new(&transformer(), vec![3, 5]).unwrap();
    for field in ["hidden", "text_dim", "heads", "head_dim", "ffn", "layers"] {
        let mut value = serde_json::to_value(&c).unwrap();
        value[field] = serde_json::json!(0);
        assert!(serde_json::from_value::<PH3ImageTokenRefinerConfig>(value).is_err());
    }
    for requests in [vec![], vec![0], vec![i32::MAX as u64, 1]] {
        assert!(PH3ImageTokenRefinerConfig::new(&transformer(), requests).is_err());
    }
    let schema = weights::schema(&c, u64::MAX).unwrap();
    let bytes = schema
        .components
        .iter()
        .map(|v| v.physical_bytes().unwrap())
        .sum::<u64>();
    for limits in [
        PH3ImageTokenRefinerLimits {
            max_metadata_bytes: 1,
            ..limits()
        },
        PH3ImageTokenRefinerLimits {
            max_plan_nodes: c.plan_nodes().unwrap() - 1,
            ..limits()
        },
        PH3ImageTokenRefinerLimits {
            max_stored_bytes: bytes - 1,
            ..limits()
        },
    ] {
        assert!(PH3ImageTokenRefinerFamily::new(limits)
            .unwrap()
            .weights(&c)
            .is_err());
    }
}

#[cfg(feature = "cuda")]
mod cuda;
