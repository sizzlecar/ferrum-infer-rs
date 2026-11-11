use super::super::{
    PH3ImageCurveLerpArithmetic, PH3ImagePackedBatchLayout, PH3ImageReferenceImage,
    PH3ImageRotaryEmbedding,
};
use super::*;
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use half::bf16;
type Payloads = BTreeMap<String, (safetensors::Dtype, Vec<usize>, Vec<u8>)>;
fn limits() -> PH3ImageCurveBlockLimits {
    PH3ImageCurveBlockLimits {
        max_metadata_bytes: 1 << 16,
        max_stored_bytes: 8 << 20,
        max_plan_nodes: 1024,
    }
}
fn transformer() -> PH3ImageTransformerConfig {
    PH3ImageTransformerConfig::from_hf_json(&serde_json::to_vec(&serde_json::json!({"_class_name":"MiniMaxH3Transformer3DModel","attention_head_dim":128,"audio_in_channels":32,"ffn_dim":512,"final_norm_eps":1e-5,"freq_dim":256,"hidden_size":256,"in_channels":24,"norm_eps":1e-5,"num_attention_heads":4,"num_layers":1,"num_refiner_layers":2,"patch_size":[1,2,2],"qk_norm_eps":1e-5,"rope_freq_dim":16,"rope_theta":10000.,"text_dim":5120,"time_embed_dim":2688,"time_embed_hidden_dim":5376})).unwrap()).unwrap()
}
struct Scene {
    config: PH3ImageCurveBlockConfig,
    times: Vec<f32>,
    x: Vec<u16>,
    cos: Vec<u16>,
    sin: Vec<u16>,
}
fn scene(alternating_prefix: bool) -> Scene {
    let t = transformer();
    let refs = [PH3ImageReferenceImage::new(4, 4).unwrap()];
    let target = PH3ImageReferenceImage::new(20, 100).unwrap();
    let first_tags = if alternating_prefix {
        (0..70)
            .map(|i| {
                if i % 2 == 0 {
                    PH3ImageModality::Text
                } else {
                    PH3ImageModality::Image
                }
            })
            .collect()
    } else {
        vec![
            PH3ImageModality::Text,
            PH3ImageModality::Image,
            PH3ImageModality::Text,
            PH3ImageModality::Text,
        ]
    };
    let second_tags = vec![PH3ImageModality::Text; 8];
    let requests = [
        PH3ImageRequestLayout::new(&t, &first_tags, target, &refs).unwrap(),
        PH3ImageRequestLayout::new(&t, &second_tags, target, &refs).unwrap(),
    ];
    let times = vec![0.125, 0.625, 0.9999];
    let plans = requests
        .iter()
        .enumerate()
        .map(|(index, layout)| {
            let mut indices = vec![index as u64; layout.rows()];
            indices[layout.reference_rows()].fill(2);
            PH3ImageCurveBlockRequest::from_layout(*layout, &indices, limits().max_metadata_bytes)
                .unwrap()
        })
        .collect();
    let curve = PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap();
    let config = PH3ImageCurveBlockConfig::new(&t, curve, 0, 3, plans).unwrap();
    let packed = PH3ImagePackedBatchLayout::new(&requests).unwrap();
    let mut positions = vec![[0.; 3]; packed.rows()];
    packed.write_positions(&mut positions).unwrap();
    let rotary = PH3ImageRotaryEmbedding::new(&t).unwrap();
    let mut cos = vec![0.; rotary.frequency_elements(packed.rows()).unwrap()];
    let mut sin = vec![0.; cos.len()];
    rotary
        .write_frequencies(&positions, &mut cos, &mut sin)
        .unwrap();
    let x = (0..packed.rows())
        .flat_map(|row| {
            let (request, local) = packed.request_at_row(row).unwrap();
            let tag = requests[request].modality(local).unwrap() as u32;
            (0..256).map(move |column| {
                bf16::from_f32(
                    (((row % 7) * 13 + column * 5) % 31) as f32 / 32. - 0.46875
                        + tag as f32 / 16.
                        + request as f32 / 8.,
                )
                .to_bits()
            })
        })
        .collect();
    assert!(requests.iter().all(|r| r.rows() >= 512));
    Scene {
        config,
        times,
        x,
        cos: cos
            .into_iter()
            .map(|x| bf16::from_f32(x).to_bits())
            .collect(),
        sin: sin
            .into_iter()
            .map(|x| bf16::from_f32(x).to_bits())
            .collect(),
    }
}
fn payloads(c: &PH3ImageCurveBlockConfig) -> Payloads {
    let mut tensors = Payloads::new();
    for (name, shape, dtype) in weights::entries(c) {
        let external = weights::external(c, &name);
        let count = shape.iter().product::<u64>() as usize;
        let dimensions = shape.iter().map(|n| *n as usize).collect();
        let (dtype, bytes) = match dtype {
            ElementType::Bf16 => {
                let qk = name.contains("q_norm") || name.contains("k_norm");
                (
                    safetensors::Dtype::BF16,
                    (0..count)
                        .flat_map(|i| {
                            bf16::from_f32(if qk {
                                0.0625 + (i % 3) as f32 / 256.
                            } else {
                                0.75 + (i % 5) as f32 / 32.
                            })
                            .to_bits()
                            .to_le_bytes()
                        })
                        .collect(),
                )
            }
            ElementType::F32 => {
                let values: Vec<f32> = if name.ends_with("weight_s_channel") {
                    (0..count)
                        .map(|i| 0.03125 * (1. + (i % 3) as f32 / 8.))
                        .collect()
                } else if name.ends_with("weight_codebook") {
                    (0..16).map(|i| i as f32 - 8.).collect()
                } else if name.ends_with("adaln_proj.linear.weight") {
                    (0..count)
                        .map(|i| {
                            (((i / c.basis as usize) * 3 + (i % c.basis as usize) * 5) % 11) as f32
                                / 1024.
                                - 5. / 1024.
                        })
                        .collect()
                } else {
                    (0..count)
                        .map(|i| {
                            let group = (i / c.hidden as usize) % 6;
                            let modality = i / (6 * c.hidden) as usize;
                            let channel = i % c.hidden as usize;
                            match group {
                                2 | 5 => {
                                    0.15625 + modality as f32 / 32. + (channel % 7) as f32 / 512.
                                }
                                1 | 4 => 0.03125 + (channel % 5) as f32 / 1024.,
                                _ => -0.015625 + (channel % 9) as f32 / 512.,
                            }
                        })
                        .collect()
                };
                (
                    safetensors::Dtype::F32,
                    values.into_iter().flat_map(f32::to_le_bytes).collect(),
                )
            }
            ElementType::U8 => (safetensors::Dtype::F8_E4M3, vec![0x38; count]),
            ElementType::I8 => {
                let (n, k) = (shape[0] as usize, shape[1] as usize * 2);
                let phase = if name.starts_with("attn.qkv") {
                    1
                } else if name.starts_with("attn.out") {
                    2
                } else if name.starts_with("mlp.fc1") {
                    3
                } else {
                    4
                };
                let mut packed = vec![0x88; count];
                for row in 0..n {
                    for slot in 0..16 {
                        let col = (row * 37 + slot * 53 + phase * 11) % k;
                        let mut code = ((row + slot * 3 + phase) % 15) as u8;
                        if code == 8 {
                            code = 15;
                        }
                        let byte = &mut packed[row * (k / 2) + col / 2];
                        if col % 2 == 0 {
                            *byte = (*byte & 0xf0) | code;
                        } else {
                            *byte = (*byte & 0x0f) | (code << 4);
                        }
                    }
                }
                (safetensors::Dtype::I8, packed)
            }
            _ => unreachable!(),
        };
        tensors.insert(external, (dtype, dimensions, bytes));
    }
    for module in ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"] {
        let bytes=br#"{"format":"asym_w4a8_int8","group_size":16,"convrot":true,"convrot_groupsize":256}"#.to_vec();
        tensors.insert(
            format!("{}.comfy_quant", weights::external(c, module)),
            (safetensors::Dtype::U8, vec![bytes.len()], bytes),
        );
    }
    let table = (0..c.grid as usize)
        .flat_map(|row| {
            (0..c.basis as usize).map(move |col| {
                let t = row as f32 / (c.grid - 1) as f32;
                (t - 0.5) * (1. + col as f32 / 16.) + col as f32 / 128.
            })
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    tensors.insert(
        "adaln_t_table".into(),
        (
            safetensors::Dtype::F32,
            vec![c.grid as usize, c.basis as usize],
            table,
        ),
    );
    tensors
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
fn complete_program_prepares_for_single_and_multiple_routing_chunks() {
    for alternating in [false, true] {
        let s = scene(alternating);
        assert_ne!(s.config.hidden_size(), s.config.attention_inner_dim());
        let family = PH3ImageCurveBlockFamily::new(limits()).unwrap();
        let prepared = TypedFamilyRegistration::new(family).prepare_with_profile(
            &serde_json::to_value(&s.config).unwrap(),
            &NumericalProfileId::new(PH3ImageCurveBlockFamily::CUDA_PROFILE_ID).unwrap(),
        );
        assert!(prepared.is_ok(), "{:?}", prepared.err());
        let family = PH3ImageCurveBlockFamily::new(limits()).unwrap();
        let p = family.numerical_profiles(&s.config).unwrap();
        let program = family
            .semantic_program(
                &s.config,
                p.resolve(
                    &NumericalProfileId::new(PH3ImageCurveBlockFamily::CUDA_PROFILE_ID).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        p.resolve(&NumericalProfileId::new(PH3ImageCurveBlockFamily::CUDA_PROFILE_ID).unwrap())
            .unwrap()
            .validate_program(&program)
            .unwrap();
        let options = s.config.compile_options().unwrap();
        let expected: BTreeSet<_> = program
            .inputs()
            .iter()
            .chain(
                program
                    .blocks()
                    .iter()
                    .flat_map(|block| &block.nodes)
                    .flat_map(|node| &node.outputs),
            )
            .cloned()
            .collect();
        assert_eq!(
            options
                .tensor_specs()
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>(),
            expected
        );
        for name in [program::Q, program::K, program::V] {
            assert_eq!(
                options.tensor_specs()[&ProgramValueId::new(name).unwrap()].dimensions,
                vec![s.config.rows().unwrap(), s.config.attention_inner_dim()]
            );
        }
        for name in program::PARAMETERS {
            let spec = options
                .tensor_specs()
                .get(&ProgramValueId::new(name).unwrap());
            if program::compact_parameters(&s.config) {
                assert!(spec.is_none(), "compact table must not expand {name}");
            } else {
                assert_eq!(
                    spec.unwrap().dimensions,
                    vec![s.config.rows().unwrap(), s.config.hidden]
                );
            }
        }
        assert_eq!(
            options.tensor_specs()
                [&ProgramValueId::new("value.p_h3_image.curve_block.adaln_rows").unwrap()]
                .dimensions,
            vec![3 * s.config.basis_rows, 6 * s.config.hidden]
        );
        let operations: BTreeSet<_> = program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .map(|n| n.operation_id.as_str())
            .collect();
        assert!(
            operations.contains(AFFINE_LINEAR_F32_OPERATION_ID)
                && operations.contains(CAST_F32_TO_BF16_OPERATION_ID)
                && operations.contains(COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID)
                && operations.contains(if program::compact_parameters(&s.config) {
                    ROW_SELECTED_ADDCMUL_BF16_OPERATION_ID
                } else {
                    ADDCMUL_BF16_OPERATION_ID
                })
        );
        assert_eq!(
            operations.contains(BATCHED_ROW_CONCAT_BF16_OPERATION_ID),
            alternating
        );
    }
}
#[test]
fn actual_checkpoint_types_and_declared_capacity_are_enforced() {
    let s = scene(false);
    let tensors = payloads(&s.config);
    let source = archive(&tensors);
    let checked = PH3ImageCurveBlockCheckpoint::new(&s.config, &source.source, limits()).unwrap();
    assert_eq!(checked.config(), &s.config);
    let schema = weights::schema(&s.config, u64::MAX).unwrap();
    let bytes = schema
        .components
        .iter()
        .map(|c| c.dimensions.iter().product::<u64>() * c.physical_element_type().size_bytes())
        .sum::<u64>();
    assert!(PH3ImageCurveBlockFamily::new(PH3ImageCurveBlockLimits {
        max_stored_bytes: bytes - 1,
        ..limits()
    })
    .unwrap()
    .weights(&s.config)
    .is_err());
    let bad_family = PH3ImageCurveBlockFamily::new(PH3ImageCurveBlockLimits {
        max_metadata_bytes: 1,
        ..limits()
    })
    .unwrap();
    assert!(bad_family.weights(&s.config).is_err());
    let mut bad = serde_json::to_value(&s.config).unwrap();
    bad["requests"][0]["runs"][0]["time_index"] = serde_json::json!(s.config.basis_rows);
    assert!(serde_json::from_value::<PH3ImageCurveBlockConfig>(bad).is_err());
    let mut bad = tensors.clone();
    bad.get_mut("blocks.0.norm1.weight").unwrap().0 = safetensors::Dtype::F32;
    bad.get_mut("blocks.0.norm1.weight").unwrap().2 = vec![0; 4 * s.config.hidden as usize];
    let source = archive(&bad);
    assert!(PH3ImageCurveBlockCheckpoint::new(&s.config, &source.source, limits()).is_err());
}
#[test]
fn individual_projection_storage_drives_real_ports_and_staged_operation_contracts() {
    let s = scene(false);
    let storage = PH3ImageBlockProjectionStorage::new(
        PH3ImageProjectionStorage::Int8W8A8,
        PH3ImageProjectionStorage::CodebookW4A8,
        PH3ImageProjectionStorage::CodebookW4A8,
        PH3ImageProjectionStorage::Int8W8A8,
    );
    let c = s.config.with_projection_storage(storage).unwrap();
    let family = PH3ImageCurveBlockFamily::new(limits()).unwrap();
    let profiles = family.numerical_profiles(&c).unwrap();
    let profile = profiles
        .resolve(&NumericalProfileId::new(PH3ImageCurveBlockFamily::cuda_profile_id(&c)).unwrap())
        .unwrap();
    let graph = family.semantic_program(&c, profile).unwrap();
    profile.validate_program(&graph).unwrap();
    let ModelWeights::Stored(schema) = family.weights(&c).unwrap() else {
        panic!("Stored required")
    };
    assert_eq!(schema.components.len(), 18);
    let ops = profile
        .operations
        .iter()
        .map(|o| o.operation_id.as_str())
        .collect::<BTreeSet<_>>();
    assert!(
        ops.contains(COMFY_CONVROT_W4A8_BF16_OPERATION_ID)
            && ops.contains(COMFY_CONVROT_W8A8_BF16_OPERATION_ID)
            && ops.contains(COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID)
    );
    assert!(!ops.contains(COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID));
    let qkv = graph.blocks()[0]
        .nodes
        .iter()
        .find(|n| n.outputs[0].as_str() == program::QKV)
        .unwrap();
    assert_eq!(qkv.inputs.len(), 3);
    assert_eq!(
        qkv.operation_id.as_str(),
        COMFY_CONVROT_W8A8_BF16_OPERATION_ID
    );
}

#[cfg(feature = "cuda")]
mod cuda;
