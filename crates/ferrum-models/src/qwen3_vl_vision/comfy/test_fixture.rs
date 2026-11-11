//! Unique original BF16 visual fixture; shared by Comfy/HF and integration tests.
use super::*;
use half::bf16;
#[cfg(test)]
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
pub(crate) fn limits() -> Qwen3VLVisionComfyLimits {
    Qwen3VLVisionComfyLimits {
        max_metadata_bytes: 2 << 20,
        max_plan_nodes: 256,
        max_stored_bytes: 4 << 20,
    }
}
pub(crate) fn metadata() -> serde_json::Value {
    serde_json::json!({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl","vision_config":{"model_type":"qwen3_vl_vision","dtype":"bfloat16","hidden_act":"gelu_pytorch_tanh","depth":3,"hidden_size":8,"intermediate_size":12,"num_heads":2,"num_position_embeddings":16,"out_hidden_size":256,"deepstack_visual_indexes":[0,1,2],"in_channels":3,"patch_size":2,"temporal_patch_size":2,"spatial_merge_size":2,"rope_parameters":{"rope_type":"axial","rope_theta":10000.0}}})
}
pub(crate) fn config(height: u64, width: u64) -> Qwen3VLVisionComfyConfig {
    let raw = metadata();
    Qwen3VLVisionComfyConfig::new(
        Qwen3VLVisionArchitecture::from_hf_value(&raw).unwrap(),
        Qwen3VLVisionGrid::new(1, height, width).unwrap(),
    )
    .unwrap()
}
#[cfg(test)]
pub(crate) fn qwen35_config(height: u64, width: u64) -> Qwen3VLVisionComfyConfig {
    // Same real block/merger implementation with the Qwen3.5 declaration;
    // absent DeepStack weights must not be invented or initialized.
    let raw = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
        "vision_config": {
            "model_type": "qwen3_5_vision", "dtype": "float32",
            "hidden_act": "gelu_pytorch_tanh", "depth": 3,
            "hidden_size": 8, "intermediate_size": 12, "num_heads": 2,
            "num_position_embeddings": 16, "out_hidden_size": 256,
            "deepstack_visual_indexes": [], "in_channels": 3, "patch_size": 2,
            "temporal_patch_size": 2, "spatial_merge_size": 2,
            "rope_parameters": {"rope_type": "axial", "rope_theta": 10000.0}
        }
    });
    Qwen3VLVisionComfyConfig::new(
        Qwen3VLVisionArchitecture::from_qwen35_hf_value(&raw).unwrap(),
        Qwen3VLVisionGrid::new(1, height, width).unwrap(),
    )
    .unwrap()
}
pub(crate) struct Original {
    pub values: BTreeMap<String, Vec<f32>>,
    pub shapes: BTreeMap<String, Vec<usize>>,
}
impl Original {
    pub fn new(c: &Qwen3VLVisionComfyConfig) -> Self {
        let schema = program::schema(c).unwrap();
        let mut values = BTreeMap::new();
        let mut shapes = BTreeMap::new();
        for w in &schema.components {
            let name = &w.external_names[0];
            let shape = w.dimensions.iter().map(|&n| n as usize).collect::<Vec<_>>();
            let len = shape.iter().product::<usize>();
            let salt = name
                .bytes()
                .fold(0usize, |a, b| a.wrapping_mul(31).wrapping_add(b as usize));
            let v = (0..len)
                .map(|i| {
                    let pattern = ((i * 7 + salt) % 23) as f32 - 11.;
                    let value = if name.contains("norm") && name.ends_with("weight") {
                        1. + pattern / 64.
                    } else if name.ends_with("bias") {
                        pattern / 128. + 1. / 256.
                    } else {
                        pattern / 128. + 1. / 512.
                    };
                    bf16::from_f32(value).to_f32()
                })
                .collect();
            values.insert(name.clone(), v);
            shapes.insert(name.clone(), shape);
        }
        let a = c.architecture();
        let name = Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME.to_owned();
        values.insert(
            name.clone(),
            (0..a.position_embeddings() * a.hidden_size())
                .map(|i| bf16::from_f32(((i * 11 % 29) as f32 - 14.) / 32.).to_f32())
                .collect(),
        );
        shapes.insert(
            name,
            vec![a.position_embeddings() as usize, a.hidden_size() as usize],
        );
        Self { values, shapes }
    }
    #[cfg(test)]
    pub fn archive(&self) -> (tempfile::TempDir, SafetensorsArchive) {
        let dir = tempfile::tempdir().unwrap();
        let bytes = self
            .values
            .iter()
            .map(|(name, v)| {
                (
                    name.clone(),
                    v.iter()
                        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let views = bytes
            .iter()
            .map(|(name, b)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::BF16, self.shapes[name].clone(), b).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, &None, &dir.path().join("vision.safetensors")).unwrap();
        let archive =
            SafetensorsArchive::open_single_file(dir.path(), "vision.safetensors").unwrap();
        (dir, archive)
    }
}
