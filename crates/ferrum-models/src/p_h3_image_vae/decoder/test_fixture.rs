//! Original INT8 decoder weights, shared with complete assets fixtures.
use super::super::PH3ImageVaeWeightEncoding;
use super::*;
use ferrum_quantization::{
    comfy_convrot::{ComfyCanonicalWeightSource, ComfyQuantizationMetadata},
    SafetensorsArchive,
};
use half::bf16;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
type Tensors = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;
#[cfg(test)]
pub(in crate::p_h3_image_vae) fn limits() -> PH3ImageVaeDecoderLimits {
    PH3ImageVaeDecoderLimits {
        max_metadata_bytes: 64 << 20,
        max_stored_bytes: 256 << 20,
        max_plan_nodes: 8192,
    }
}
#[cfg(test)]
pub(in crate::p_h3_image_vae) fn architecture(layers: u64) -> PH3ImageVaeDecoderConfig {
    PH3ImageVaeDecoderConfig::from_hf_json(&serde_json::to_vec(&serde_json::json!({
        "_class_name":"AutoencoderKLPH3Image","latent_channels":24,"out_channels":3,"spatial_downsample_factors":[2],
        "decoder_num_layers":layers,"decoder_num_attention_heads":4,"decoder_attention_head_dim":64,"decoder_num_register_tokens":4,
        "decoder_ffn_mult":4,"decoder_rope_dim_ratio":0.75,"decoder_rope_theta":100,"decoder_norm_eps":0.00001,
    })).unwrap()).unwrap()
}
pub(in crate::p_h3_image_vae) fn f32_bytes(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(f32::to_le_bytes).collect()
}
pub(in crate::p_h3_image_vae) fn originals(c: &PH3ImageVaeDecoderExecutionConfig) -> Tensors {
    let mut tensors = Tensors::new();
    for w in c.architecture().expected_comfy_weights().unwrap() {
        let name = w.external_name();
        let shape = w
            .dimensions()
            .iter()
            .map(|&v| v as usize)
            .collect::<Vec<_>>();
        let count = shape.iter().product();
        let salt = name.bytes().fold(0usize, |n, b| n + usize::from(b));
        match w.encoding() {
            PH3ImageVaeWeightEncoding::Bf16 => {
                let values = (0..count).map(|i| {
                    if name.ends_with("norm1.weight")
                        || name.ends_with("norm2.weight")
                        || name == "decoder.norm_out.weight"
                    {
                        0.75 + (i % 9) as f32 / 16.
                    } else if name.ends_with("scale1") || name.ends_with("scale2") {
                        ((i % 7) as f32 - 3.) / 8.
                    } else {
                        ((i * 7 + salt) % 23) as f32 / 128. - 0.0625
                    }
                });
                tensors.insert(
                    name.into(),
                    (
                        Dtype::BF16,
                        shape,
                        values
                            .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
                            .collect(),
                    ),
                );
            }
            PH3ImageVaeWeightEncoding::ConvRotInt8 => {
                let rows = shape[0];
                let columns = shape[1];
                let bytes = (0..count)
                    .map(|i| {
                        if i % columns == 0 && (i / columns) % 17 == 0 {
                            (-128i8) as u8
                        } else {
                            (((i / columns * 7 + i % columns * 3 + salt) % 19) as i8 - 9) as u8
                        }
                    })
                    .collect();
                tensors.insert(name.into(), (Dtype::I8, shape, bytes));
                tensors.insert(
                    w.scale_external_name().unwrap(),
                    (
                        Dtype::F32,
                        vec![rows, 1],
                        f32_bytes((0..rows).map(|r| (1 + r % 3) as f32 / 32768.)),
                    ),
                );
                let data = serde_json::to_vec(&ComfyQuantizationMetadata::Int8Tensorwise {
                    convrot: true,
                    convrot_groupsize: 256,
                })
                .unwrap();
                tensors.insert(
                    w.quantization_external_name().unwrap(),
                    (Dtype::U8, vec![data.len()], data),
                );
            }
        }
    }
    tensors
}
pub(in crate::p_h3_image_vae) fn archive(
    t: &Tensors,
) -> (tempfile::TempDir, ComfyCanonicalWeightSource) {
    let d = tempfile::tempdir().unwrap();
    let views = t
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &d.path().join("weights.safetensors")).unwrap();
    let source = SafetensorsArchive::open_single_file(d.path(), "weights.safetensors").unwrap();
    (d, ComfyCanonicalWeightSource::new(source).unwrap())
}
