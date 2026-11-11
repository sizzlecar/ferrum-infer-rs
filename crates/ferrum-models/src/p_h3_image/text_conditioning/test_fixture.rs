//! One original-format numerical fixture shared by model and downstream tests.
use super::*;
use crate::qwen3_vl_text::Qwen3VLTextComfyProjectionQuantization as Quant;
use ferrum_quantization::{comfy_convrot::ComfyQuantizationMetadata, SafetensorsArchive};
use half::bf16;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
pub(in crate::p_h3_image) fn metadata() -> serde_json::Value {
    serde_json::json!({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl",
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","attention_bias":false,"attention_dropout":0.0,"hidden_act":"silu",
        "num_hidden_layers":5,"hidden_size":256,"intermediate_size":256,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":128,
        "vocab_size":16,"max_position_embeddings":128,"rms_norm_eps":0.25,"rope_theta":5000000,
        "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[64,0,0]}}})
}
pub(in crate::p_h3_image) fn limits() -> PH3ImageTextConditioningLimits {
    PH3ImageTextConditioningLimits {
        max_metadata_bytes: 1 << 20,
        max_stored_bytes: 16 << 20,
        max_plan_nodes: 128,
    }
}
pub(in crate::p_h3_image) fn config(layout: Qwen3VLTextLayout) -> PH3ImageTextConditioningConfig {
    let raw = metadata();
    let text = Qwen3VLTextConfig::new(
        Qwen3VLTextArchitecture::from_hf_value(&raw).unwrap(),
        layout,
    )
    .unwrap()
    .with_hidden_state_indices(vec![1, 2, 3, 4, 5])
    .unwrap();
    let projections = (0..5)
        .flat_map(|layer| {
            [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]
            .into_iter()
            .enumerate()
            .map(move |(i, suffix)| {
                (
                    format!("model.language_model.layers.{layer}.{suffix}"),
                    if i % 3 == 0 { Quant::W8A8 } else { Quant::W4A8 },
                )
            })
        })
        .collect();
    PH3ImageTextConditioningConfig::new(Qwen3VLTextComfyConfig::new(text, projections).unwrap(), 4)
        .unwrap()
}
pub(in crate::p_h3_image) fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
pub(in crate::p_h3_image) type Tensors = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;
pub(in crate::p_h3_image) fn original(c: &PH3ImageTextConditioningConfig) -> Tensors {
    let family = PH3ImageTextConditioningFamily::new(limits()).unwrap();
    let weights = family.weights(c).unwrap();
    let mut tensors = BTreeMap::new();
    for component in &weights.stored_schema().unwrap().components {
        let name = &component.external_names[0];
        let shape = component
            .dimensions
            .iter()
            .map(|&n| n as usize)
            .collect::<Vec<_>>();
        let n = shape.iter().product::<usize>();
        let WeightEncoding::Dense { element_type } = &component.encoding else {
            panic!("dense physical ports")
        };
        let (dtype, bytes) = if name.ends_with("norm.weight") {
            (
                Dtype::BF16,
                std::iter::repeat_n(bf16::ONE.to_bits().to_le_bytes(), n)
                    .flatten()
                    .collect(),
            )
        } else if name.ends_with("weight_s_rel") {
            (Dtype::F8_E4M3, vec![0x38; n])
        } else if *element_type == ElementType::I8 {
            let mut bytes = vec![0; n];
            if name == Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME {
                bytes[0] = 16;
                bytes[256] = (-32_i8) as u8;
            }
            (Dtype::I8, bytes)
        } else {
            assert_eq!(*element_type, ElementType::F32);
            let x = if name.ends_with("weight_codebook") {
                0.0
            } else if name == "model.language_model.embed_tokens.weight_scale" {
                0.5
            } else {
                1.0
            };
            (Dtype::F32, floats(std::iter::repeat_n(x, n)))
        };
        tensors.insert(name.clone(), (dtype, shape, bytes));
    }
    for (module, quant) in c.encoder().projections() {
        let metadata = serde_json::to_vec(&quant.metadata()).unwrap();
        tensors.insert(
            format!("{module}.comfy_quant"),
            (Dtype::U8, vec![metadata.len()], metadata),
        );
    }
    let metadata = serde_json::to_vec(&ComfyQuantizationMetadata::Int8Tensorwise {
        convrot: true,
        convrot_groupsize: 256,
    })
    .unwrap();
    tensors.insert(
        "model.language_model.embed_tokens.comfy_quant".into(),
        (Dtype::U8, vec![metadata.len()], metadata),
    );
    let input = c.connector_config().unwrap().input_dim() as usize;
    let mut projection = vec![0.0; 4 * input];
    projection[0] = 1.0;
    projection[input + 4 * 256] = 1.0;
    projection[2 * input + 256 + 1] = 1.0;
    projection[3 * input + 2 * 256] = -0.5;
    for (name, shape, values) in [
        ("input_mean", vec![input], vec![0.0; input]),
        ("input_std", vec![input], vec![1.0; input]),
        ("proj.weight", vec![4, input], projection),
        ("output_mean", vec![4], vec![1.0, -1.0, 0.5, 0.25]),
        ("output_std", vec![4], vec![0.5, 0.25, 1.0, 2.0]),
        ("sink_embedding", vec![4], vec![-3.0, 2.0, 7.0, 1.0]),
    ] {
        tensors.insert(
            format!("text_connector.{name}"),
            (Dtype::F32, shape, floats(values)),
        );
    }
    tensors
}
pub(in crate::p_h3_image) fn archive(
    tensors: &Tensors,
) -> (tempfile::TempDir, ComfyCanonicalWeightSource) {
    let directory = tempfile::tempdir().unwrap();
    let views = tensors
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &directory.path().join("text.safetensors")).unwrap();
    let source = ComfyCanonicalWeightSource::new(
        SafetensorsArchive::open_single_file(directory.path(), "text.safetensors").unwrap(),
    )
    .unwrap();
    (directory, source)
}
