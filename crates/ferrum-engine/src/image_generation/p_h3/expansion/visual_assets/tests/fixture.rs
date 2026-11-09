//! One original-format archive containing the existing tiny causal fixture and
//! real main-only visual weights. These tests qualify producer/source custody;
//! the zero causal branches are not proof that visual features affect logits.
use super::*;
use crate::image_generation::p_h3::expansion::tests::fixture::{protocol_limits, Fixture};
use crate::image_generation::p_h3::{PH3ImageAssets, PH3ImagePromptExpansionOptions};
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use ferrum_models::vnext::ProductionModelSourceBundle;
use ferrum_runtime::{ElementType, ModelFamilyProvider, WeightEncoding};
use half::bf16;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::collections::BTreeMap;

pub(in crate::image_generation::p_h3) struct MergedExpansionFixture {
    pub directory: std::path::PathBuf,
    pub expansion: Arc<super::super::super::NativeTextExpansion>,
    _original: Fixture,
}
pub(in crate::image_generation::p_h3::expansion) fn limits() -> Qwen3VLVisionHfLimits {
    Qwen3VLVisionHfLimits {
        max_metadata_bytes: 2 << 20,
        max_plan_nodes: 256,
        max_stored_bytes: 4 << 20,
    }
}
pub(in crate::image_generation::p_h3::expansion) fn processor() -> Qwen3VLVisionHfProcessor {
    let raw = serde_json::json!({"processor_class":"Qwen3VLProcessor","image_processor":{"image_processor_type":"Qwen2VLImageProcessor","do_convert_rgb":true,"do_normalize":true,"do_rescale":true,"do_resize":true,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"patch_size":2,"merge_size":2,"temporal_patch_size":2,"rescale_factor":1./255.,"resample":3,"size":{"shortest_edge":16,"longest_edge":64}}});
    Qwen3VLVisionHfProcessor::from_source_bytes(
        &serde_json::to_vec(&raw).unwrap(),
        Qwen3VLVisionHfProcessorLimits {
            max_processor_bytes: 4096,
            image: ImageInputLimits::new(1 << 20, 2_000_000, 2048, 2048, 1 << 20).unwrap(),
            max_patch_rows: 500_000,
            max_patch_bytes: 48 << 20,
        },
    )
    .unwrap()
}
impl MergedExpansionFixture {
    pub(in crate::image_generation::p_h3) fn new() -> Self {
        let mut original = Fixture::new();
        // No live mmap is mutated while constructing this new fixture source.
        drop(original.take_expansion());
        let directory = original.directory.clone();
        let config_path = directory.join("config.json");
        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        raw["vision_start_token_id"] = 4.into();
        raw["image_token_id"] = 5.into();
        raw["vision_end_token_id"] = 6.into();
        raw["text_config"]["rope_parameters"] = serde_json::json!({"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":1.,"mrope_interleaved":true,"mrope_section":[1,0,0]});
        raw["vision_config"] = serde_json::json!({"model_type":"qwen3_5_vision","dtype":"float32","hidden_act":"gelu_pytorch_tanh","depth":3,"hidden_size":8,"intermediate_size":12,"num_heads":2,"num_position_embeddings":16,"out_hidden_size":2,"deepstack_visual_indexes":[],"in_channels":3,"patch_size":2,"temporal_patch_size":2,"spatial_merge_size":2,"rope_parameters":{"rope_type":"axial","rope_theta":10000.}});
        let p = processor();
        let a = Qwen3VLVisionArchitecture::from_qwen35_hf_value(&raw).unwrap();
        let c =
            Qwen3VLVisionHfConfig::new(a, Qwen3VLVisionGrid::new(1, 2, 2).unwrap(), &p).unwrap();
        let family = Qwen3VLVisionHfFamily::new(limits()).unwrap();
        let model_weights = family.weights(&c).unwrap();
        let schema = model_weights.stored_schema().unwrap();
        let weights_path = directory.join("model.safetensors");
        let old = std::fs::read(&weights_path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&old).unwrap();
        let mut owned = tensors
            .tensors()
            .into_iter()
            .map(|(name, t)| {
                let name = if name == "model.lm_head.weight" {
                    "lm_head.weight".to_owned()
                } else {
                    name.replacen("model.", "model.language_model.", 1)
                };
                (name, (t.dtype(), t.shape().to_vec(), t.data().to_vec()))
            })
            .collect::<BTreeMap<_, _>>();
        for component in &schema.components {
            assert_eq!(
                component.encoding,
                WeightEncoding::Dense {
                    element_type: ElementType::F32
                }
            );
            let name = &component.external_names[0];
            let length = component.dimensions.iter().product::<u64>() as usize;
            let norm = name.contains("norm") && name.ends_with("weight");
            let salt = name
                .bytes()
                .fold(0usize, |n, v| n.wrapping_mul(31).wrapping_add(v as usize));
            let bytes = (0..length)
                .flat_map(|i| {
                    let v = if norm {
                        1. + ((i + salt) % 7) as f32 / 64.
                    } else {
                        (((i * 7 + salt) % 23) as f32 - 11.) / 128. + 1. / 512.
                    };
                    bf16::from_f32(v).to_le_bytes()
                })
                .collect();
            owned.insert(
                name.clone(),
                (
                    Dtype::BF16,
                    component.dimensions.iter().map(|&d| d as usize).collect(),
                    bytes,
                ),
            );
        }
        owned.insert(
            "model.visual.pos_embed.weight".into(),
            (
                Dtype::BF16,
                vec![16, 8],
                (0..128)
                    .flat_map(|i| bf16::from_f32(((i * 11 % 29) as f32 - 14.) / 32.).to_le_bytes())
                    .collect(),
            ),
        );
        let views = owned
            .iter()
            .map(|(name, (dtype, shape, bytes))| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, &None, &weights_path).unwrap();
        std::fs::write(config_path, serde_json::to_vec(&raw).unwrap()).unwrap();
        let tokenizer_path = directory.join("tokenizer.json");
        let mut tokenizer: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&tokenizer_path).unwrap()).unwrap();
        let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
        let answer = vocab
            .iter()
            .find(|(_, v)| v.as_u64() == Some(7))
            .unwrap()
            .0
            .clone();
        vocab.remove(&answer);
        vocab.insert(
            "</think> Edit <Picture 1>. Keep the subject and change the sky to blue.".into(),
            7.into(),
        );
        std::fs::write(tokenizer_path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let sources =
            Arc::new(ProductionModelSourceBundle::open_colocated_safetensors(&directory).unwrap());
        let expansion =
            super::super::super::NativeTextExpansion::from_sources(sources, protocol_limits())
                .unwrap()
                .with_visual_assets(
                    p,
                    limits(),
                    NumericalProfileId::new(Qwen3VLVisionHfFamily::REFERENCE_PROFILE_ID).unwrap(),
                )
                .unwrap();
        Self {
            directory,
            expansion: Arc::new(expansion),
            _original: original,
        }
    }
    pub(in crate::image_generation::p_h3::expansion) fn assets(
        &self,
    ) -> Arc<ExpansionVisualAssets> {
        self.expansion.visual_assets().unwrap().clone()
    }

    /// Use the same frozen source bundle and actual processor through the
    /// production attach API. The existing CPU fixture keeps its Reference
    /// profile; this creates a separately declared CUDA visual preparation.
    /// It allocates no runtime, returns no fake completion and does not reopen
    /// the directory to discover a different set of artifacts.
    pub(in crate::image_generation::p_h3) fn attach_cuda(
        &self,
        assets: PH3ImageAssets,
        options: PH3ImagePromptExpansionOptions,
    ) -> anyhow::Result<PH3ImageAssets> {
        let visual = self.assets();
        assets.with_prompt_expansion_visual_sources(
            Arc::clone(self.expansion.model.prepared().sources()),
            protocol_limits(),
            options,
            visual.processor().clone(),
            visual.limits(),
        )
    }

    /// Two real generated tokens are sufficient for this fixture's answer
    /// followed by its distinct tokenizer-declared EOS. Backend/device policy
    /// is supplied by the actual caller's EngineConfig.
    pub(in crate::image_generation::p_h3) fn native_options(
        engine: ferrum_types::EngineConfig,
    ) -> PH3ImagePromptExpansionOptions {
        PH3ImagePromptExpansionOptions {
            engine,
            max_new_tokens: 2,
            prefill_chunk_tokens: 2,
        }
    }
}
