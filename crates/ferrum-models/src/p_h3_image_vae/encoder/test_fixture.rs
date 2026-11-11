//! Original CNN temporal weights, shared with complete assets fixtures.
use super::*;
#[cfg(test)]
use ferrum_quantization::SafetensorsArchive;
#[cfg(test)]
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
type Tensors = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;
#[cfg(test)]
pub(in crate::p_h3_image_vae) fn architecture(tiny: bool) -> PH3ImageVaeEncoderArchitecture {
    PH3ImageVaeEncoderArchitecture::from_hf_json(&serde_json::to_vec(&serde_json::json!({
    "_class_name":"AutoencoderKLPH3Image","in_channels":3,"latent_channels":24,
    "block_out_channels":if tiny{vec![4,8,8,16,16,32]}else{vec![128,256,256,512,512,1024]},"layers_per_block":2,
    "spatial_downsample_factors":[2,2,2,2,1,1],"temporal_downsample_factors":[1,2,2,1,1,1],
    "norm_num_groups":if tiny{2}else{32},"norm_eps":0.000001,"spatial_padding_mode":"reflect",
})).unwrap()).unwrap()
}
#[cfg(test)]
pub(in crate::p_h3_image_vae) fn limits() -> PH3ImageVaeEncoderLimits {
    PH3ImageVaeEncoderLimits {
        max_metadata_bytes: 1 << 20,
        max_stored_bytes: 1 << 30,
        max_plan_nodes: 4096,
    }
}
pub(in crate::p_h3_image_vae) fn bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
pub(in crate::p_h3_image_vae) fn originals(c: &PH3ImageVaeEncoderConfig) -> Tensors {
    weights::entries(c)
        .unwrap()
        .into_iter()
        .map(|entry| {
            let count = entry.original.iter().product::<u64>() as usize;
            let shape = entry
                .original
                .iter()
                .map(|&n| n as usize)
                .collect::<Vec<_>>();
            let salt = entry.name.bytes().map(usize::from).sum::<usize>();
            let values = (0..count).map(|i| {
                if shape.len() == 5 {
                    let plane = shape[3] * shape[4];
                    let temporal = i / plane % shape[2];
                    let value = ((i * 7 + i / plane + salt) % 19) as f32 / 2048. - 0.004 + 0.000001;
                    if temporal + 1 == shape[2] {
                        value
                    } else if temporal == 0 {
                        value + 1.
                    } else {
                        value - 2.
                    }
                } else if entry.name.contains("norm") && entry.name.ends_with("weight") {
                    0.875 + (i % 7) as f32 / 32. + 0.000123
                } else {
                    ((i * 3 + salt) % 13) as f32 / 512. - 0.0078125 + 0.000123
                }
            });
            let payload = bytes(values);
            (entry.name, (Dtype::F32, shape, payload))
        })
        .collect()
}
#[cfg(test)]
pub(in crate::p_h3_image_vae) fn archive(t: &Tensors) -> (tempfile::TempDir, SafetensorsArchive) {
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
    (d, source)
}
