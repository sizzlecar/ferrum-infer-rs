use super::*;
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use half::bf16;
pub(in crate::p_h3_image) fn transformer() -> PH3ImageTransformerConfig {
    transformer_with_image_channels(2).unwrap()
}
pub(in crate::p_h3_image) fn transformer_metadata(image_channels: u64) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({"_class_name":"MiniMaxH3Transformer3DModel","attention_head_dim":128,"audio_in_channels":2,"ffn_dim":512,"final_norm_eps":1e-5,"freq_dim":256,"hidden_size":256,"in_channels":image_channels,"norm_eps":1e-5,"num_attention_heads":4,"num_layers":2,"num_refiner_layers":2,"patch_size":[1,2,2],"qk_norm_eps":1e-5,"rope_freq_dim":16,"rope_theta":10000.,"text_dim":4,"time_embed_dim":2688,"time_embed_hidden_dim":5376})).unwrap()
}
pub(in crate::p_h3_image) fn transformer_with_image_channels(
    image_channels: u64,
) -> anyhow::Result<PH3ImageTransformerConfig> {
    PH3ImageTransformerConfig::from_hf_json(&transformer_metadata(image_channels))
}
pub(in crate::p_h3_image) fn limits() -> PH3ImageDiTLimits {
    PH3ImageDiTLimits {
        max_metadata_bytes: 32 << 20,
        max_plan_nodes: 4096,
        max_stored_bytes: 1 << 30,
    }
}
type Payloads = BTreeMap<String, (safetensors::Dtype, Vec<usize>, Vec<u8>)>;
pub(in crate::p_h3_image) struct Files {
    pub _directory: tempfile::TempDir,
    pub source: ComfyCanonicalWeightSource,
}
pub(in crate::p_h3_image) fn config() -> PH3ImageDiTConfig {
    config_with_transformer(&transformer())
}
pub(in crate::p_h3_image) fn config_with_transformer(
    t: &PH3ImageTransformerConfig,
) -> PH3ImageDiTConfig {
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let reference = PH3ImageReferenceImage::new(2, 4).unwrap();
    let requests = (0..2)
        .map(|request| {
            let mut tags = vec![PH3ImageModality::Text; 512];
            tags[1] = PH3ImageModality::Image;
            let refs = Some(reference);
            let layout = PH3ImageRequestLayout::new(t, &tags, target, refs.as_slice()).unwrap();
            let mut times = vec![request as u64; layout.rows()];
            times[layout.reference_rows()].fill(2);
            PH3ImageDiTRequest::new(t, &tags, target, refs, &times, 1 << 20).unwrap()
        })
        .collect();
    PH3ImageDiTConfig::new(
        t,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        3,
        requests,
        PH3ImageDiTPhase::WithTextRefiner,
        PH3ImageStackProjectionStorage::PerLayer(vec![
            PH3ImageBlockProjectionStorage::default(),
            PH3ImageBlockProjectionStorage::uniform(
                crate::p_h3_image::PH3ImageProjectionStorage::Int8W8A8,
            ),
        ]),
    )
    .unwrap()
}
pub(in crate::p_h3_image) fn files(c: &PH3ImageDiTConfig) -> Files {
    let schema = program::compose(c, limits()).unwrap().schema;
    let mut result = Payloads::new();
    for component in schema.components {
        let name = component.external_names[0].clone();
        let shape = component.dimensions;
        let count = shape.iter().product::<u64>() as usize;
        let phase = name.bytes().map(usize::from).sum::<usize>();
        let (dtype, bytes) = match component.encoding {
            WeightEncoding::Dense {
                element_type: ElementType::Bf16,
            } => (
                safetensors::Dtype::BF16,
                (0..count)
                    .flat_map(|i| {
                        bf16::from_f32(if name == "condition_proj.weight" {
                            ((i * 7) % 13) as f32 / 256. - 6. / 256.
                        } else if name == "condition_proj.bias" {
                            (i % 7) as f32 / 128. - 3. / 128.
                        } else if name.contains("q_norm") || name.contains("k_norm") {
                            0.0625 + (i % 3) as f32 / 256.
                        } else {
                            0.75 + (i % 5) as f32 / 32.
                        })
                        .to_bits()
                        .to_le_bytes()
                    })
                    .collect(),
            ),
            WeightEncoding::Dense {
                element_type: ElementType::F32,
            } => {
                let values = (0..count).map(|i| {
                    if name.ends_with("weight_s_channel") || name.ends_with("weight_scale") {
                        0.03125 * (1. + (i % 3) as f32 / 8.)
                    } else if name.ends_with("weight_codebook") {
                        i as f32 - 8.
                    } else if name.contains("adaln_proj.linear.weight") {
                        ((i * 5 + phase) % 11) as f32 / 1024. - 5. / 1024.
                    } else if name.starts_with("blocks.")
                        && name.ends_with("adaln_proj.linear.bias")
                    {
                        let parameter = (i / c.hidden as usize) % 6;
                        match parameter {
                            2 | 5 => 0.1875 + (i % 7) as f32 / 512.,
                            1 | 4 => 0.03125 + (i % 5) as f32 / 1024.,
                            _ => -0.015625 + (i % 9) as f32 / 512.,
                        }
                    } else if name.ends_with(".weight") {
                        ((i * 7 + phase) % 17) as f32 / 1024. - 8. / 1024.
                    } else {
                        0.03125 + (i % 7) as f32 / 1024.
                    }
                });
                (
                    safetensors::Dtype::F32,
                    values.flat_map(f32::to_le_bytes).collect(),
                )
            }
            WeightEncoding::Dense {
                element_type: ElementType::U8,
            } => (safetensors::Dtype::F8_E4M3, vec![0x38; count]),
            WeightEncoding::Dense {
                element_type: ElementType::I8,
            } => {
                let w4 = if name.starts_with("blocks.") {
                    let layer = name.split('.').nth(1).unwrap().parse::<u64>().unwrap();
                    let module = name
                        .strip_prefix(&format!("blocks.{layer}."))
                        .unwrap()
                        .strip_suffix(".weight")
                        .unwrap();
                    c.stack
                        .storage_for_layer(layer)
                        .unwrap()
                        .modules()
                        .into_iter()
                        .find(|(n, _)| *n == module)
                        .unwrap()
                        .1
                        == crate::p_h3_image::PH3ImageProjectionStorage::CodebookW4A8
                } else {
                    false
                };
                let n = shape[0] as usize;
                let k = shape[1] as usize * if w4 { 2 } else { 1 };
                let mut bytes = vec![if w4 { 0x88 } else { 0 }; count];
                for row in 0..n {
                    for j in 0..16 {
                        let col = (row * 37 + j * 53 + phase) % k;
                        if w4 {
                            let mut code = ((row + j * 3 + phase) % 15) as u8;
                            if code == 8 {
                                code = 15;
                            }
                            let b = &mut bytes[row * (k / 2) + col / 2];
                            if col % 2 == 0 {
                                *b = (*b & 0xf0) | code;
                            } else {
                                *b = (*b & 0xf) | (code << 4);
                            }
                        } else {
                            bytes[row * k + col] = (((row + j * 3 + phase) % 7) as i8 - 3) as u8;
                        }
                    }
                }
                (safetensors::Dtype::I8, bytes)
            }
            _ => panic!("unexpected physical encoding"),
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
    for layer in 0..c.stack.layers() {
        for (module, storage) in c.stack.storage_for_layer(layer).unwrap().modules() {
            let bytes=match storage{crate::p_h3_image::PH3ImageProjectionStorage::CodebookW4A8=>br#"{"format":"asym_w4a8_int8","group_size":16,"convrot":true,"convrot_groupsize":256}"#.to_vec(),crate::p_h3_image::PH3ImageProjectionStorage::Int8W8A8=>br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}"#.to_vec()};
            result.insert(
                format!("blocks.{layer}.{module}.comfy_quant"),
                (safetensors::Dtype::U8, vec![bytes.len()], bytes),
            );
        }
    }
    for layer in 0..2 {
        for module in ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"] {
            let bytes =
                br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}"#.to_vec();
            result.insert(
                format!("token_refiner.blocks.{layer}.{module}.comfy_quant"),
                (safetensors::Dtype::U8, vec![bytes.len()], bytes),
            );
        }
    }
    result.insert(
        "adaln_t_table".into(),
        (
            safetensors::Dtype::F32,
            vec![4097, 16],
            (0..4097)
                .flat_map(|row| {
                    (0..16).map(move |column| {
                        (row as f32 / 4096. - 0.5) * (1. + column as f32 / 16.)
                            + column as f32 / 128.
                    })
                })
                .flat_map(f32::to_le_bytes)
                .collect(),
        ),
    );
    result.insert(
        "rope.inv_freq".into(),
        (
            safetensors::Dtype::F32,
            vec![16],
            (0..16)
                .flat_map(|i| (1.0_f32 / (i + 1) as f32).to_le_bytes())
                .collect(),
        ),
    );
    let directory = tempfile::tempdir().unwrap();
    let views = result
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
    Files {
        _directory: directory,
        source,
    }
}
