use super::*;
use ferrum_kernels::backend::reference::{ReferenceDeviceRuntime, ReferenceVNextComposition};
use ferrum_models::qwen_image21::{
    QwenImage21ImageProjectionFamily, IMAGE_PROJECTION_INPUT_ID, IMAGE_PROJECTION_PROFILE_ID,
};
use serde_json::json;

pub(super) const DEVICE_BYTES: u64 = 64 * 1024 * 1024;

pub(super) fn transformer() -> QwenImage21Config {
    QwenImage21Config::from_hf_json(
        &serde_json::to_vec(&transformer_metadata()).unwrap(),
        StructureLimits::default(),
    )
    .unwrap()
}
pub(super) fn transformer_metadata() -> serde_json::Value {
    json!({
        "_class_name":"QwenImage21Transformer2DModel",
        "num_layers":1,"num_attention_heads":1,"attention_head_dim":8,
        "context_in_dim":8,"in_channels":64,"out_channels":64,
        "patch_size":1,"mlp_ratio":2,"axes_dims_rope":[2,2,4],
        "causal_condition":true,"eps":0.000001
    })
}

pub(super) fn text() -> Qwen3VLTextArchitecture {
    Qwen3VLTextArchitecture::from_hf_value(&text_metadata()).unwrap()
}
pub(super) fn text_metadata() -> serde_json::Value {
    json!({
        "model_type":"qwen3_vl","architectures":["Qwen3VLForConditionalGeneration"],
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16",
            "hidden_act":"silu","attention_bias":false,"attention_dropout":0.0,
            "num_hidden_layers":1,"hidden_size":8,"intermediate_size":16,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,
            "vocab_size":32,"max_position_embeddings":1024,"rms_norm_eps":0.000001,
            "rope_theta":100,"rope_scaling":{"rope_type":"default",
                "mrope_interleaved":true,"mrope_section":[2,1,1]}}
    })
}

pub(super) fn decoder() -> QwenImage21VaeDecoderConfig {
    QwenImage21VaeDecoderConfig::new(
        64,
        1,
        vec![1, 2, 4, 8, 8],
        2,
        vec![true, true, true, false],
        1,
        2,
        2,
    )
    .unwrap()
}

pub(super) fn composition(name: &str) -> ReferenceVNextComposition {
    ReferenceVNextComposition::create_with_capacity(id(name).unwrap(), DEVICE_BYTES).unwrap()
}

pub(super) fn compile(
    composition: &ReferenceVNextComposition,
    family: PreparedModelFamily,
    options: &ProgramPlanCompileOptions,
) -> Result<CompiledTensorComponent<ReferenceDeviceRuntime>, VNextError> {
    let policy = TensorComponentPolicy::new(
        DEVICE_BYTES,
        composition
            .runtime()
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    CompiledTensorComponent::compile(
        family,
        composition.registry(),
        composition.catalog(),
        &policy,
        composition.weight_materializers(),
        options,
    )
}

pub(super) fn projection() -> PreparedModelFamily {
    TypedFamilyRegistration::new(QwenImage21ImageProjectionFamily::new().unwrap())
        .prepare_with_profile(
            &serde_json::to_value(transformer()).unwrap(),
            &NumericalProfileId::new(IMAGE_PROJECTION_PROFILE_ID).unwrap(),
        )
        .unwrap()
}

pub(super) fn projection_options() -> ProgramPlanCompileOptions {
    ProgramPlanCompileOptions::new(BTreeMap::from([(
        id(IMAGE_PROJECTION_INPUT_ID).unwrap(),
        ProgramTensorSpec {
            dimensions: vec![1, 64],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap()
}

/// Small real safetensors headers/payloads, exclusively a metadata fixture.
/// No fake operation/provider is registered and no normal samples are supplied.
pub(super) fn archive(
    directory: &std::path::Path,
    shape: Vec<usize>,
    dtype: safetensors::tensor::Dtype,
) -> SafetensorsArchive {
    use safetensors::tensor::{serialize_to_file, TensorView};
    let bytes = vec![0_u8; shape.iter().product::<usize>() * dtype.size()];
    serialize_to_file(
        vec![(
            "img_in.weight",
            TensorView::new(dtype, shape, &bytes).unwrap(),
        )],
        &None::<std::collections::HashMap<String, String>>,
        &directory.join("model.safetensors"),
    )
    .unwrap();
    SafetensorsArchive::open(directory).unwrap()
}
