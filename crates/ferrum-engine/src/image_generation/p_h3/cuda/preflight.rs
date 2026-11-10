use super::*;
use ferrum_models::{
    normal_noise::{NormalNoiseConfig, NormalNoiseFamily, NormalNoisePrecision},
    p_h3_image::{
        PH3ImageDenoiserPhase, PH3ImageModality, PH3ImageReferenceImage, PH3ImageRequestLayout,
        PH3ImageTextConditioningConfig, PH3ImageTextConditioningFamily,
        PH3ImageTextConnectorProjectionFamily,
    },
    p_h3_image_vae::PH3ImageVaeSpatialPolicy,
    qwen3_vl_text::{
        Qwen3VLTextDeepStack, Qwen3VLTextKeyRun, Qwen3VLTextLayout, Qwen3VLTextVisualRun,
    },
    qwen3_vl_vision::{
        comfy::{Qwen3VLVisionComfyConfig, Qwen3VLVisionComfyFamily},
        Qwen3VLVisionGrid,
    },
};

/// Actual fixed-topology families with their smallest legal payload-free
/// geometry. This neither compiles a backend plan nor installs any weights.
pub(super) fn families(
    assets: &PH3ImageAssets,
    options: &PH3ImagePipelineOptions,
) -> Result<Vec<PreparedModelFamily>> {
    let config = PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
        assets.text.clone(),
        Qwen3VLTextLayout::new(1, vec![vec![Qwen3VLTextKeyRun::new(1, true)]])?,
        &assets.sources.text_encoder,
        &assets.connector,
    )?;
    let text =
        TypedFamilyRegistration::new(PH3ImageTextConditioningFamily::new(assets.limits.text)?)
            .prepare_with_profile(
                &serde_json::to_value(&config)?,
                &profile(PH3ImageTextConditioningFamily::CUDA_PROFILE_ID)?,
            )?;
    let projection = TypedFamilyRegistration::new(PH3ImageTextConnectorProjectionFamily::new(
        assets.limits.connector,
    )?)
    .prepare_with_profile(
        &serde_json::to_value(config.projection_config()?)?,
        &profile(PH3ImageTextConnectorProjectionFamily::CUDA_PROFILE_ID)?,
    )?;
    let noise_config = NormalNoiseConfig::new_with_precision(1, 1, 1, NormalNoisePrecision::F32)?;
    let noise = TypedFamilyRegistration::new(NormalNoiseFamily::new_with_precision(
        NormalNoisePrecision::F32,
        options.noise_profile.as_str(),
    )?)
    .prepare_with_profile(&serde_json::to_value(noise_config)?, &options.noise_profile)?;
    let mut families = vec![text, projection, noise];
    if let Some(expansion) = assets.prompt_expansion_family() {
        families.push(expansion.clone());
    }
    match (
        assets.prompt_expansion_visual_causal_family(),
        assets.prompt_expansion_vision_family(),
    ) {
        (Some(causal), Some(vision)) => {
            ensure!(
                assets.has_prompt_expansion_visual(),
                "visual expansion families lack their validated capability"
            );
            ensure!(
                matches!(vision.numerical_profile().id.as_str(),
                    ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfFamily::CUDA_PROFILE_ID
                    | ferrum_models::qwen3_vl_vision::hf::Qwen3VLVisionHfFamily::CUDA_FUSED_F32_PROFILE_ID),
                "CUDA composition requires the declared CUDA HF vision profile"
            );
            families.extend([causal.clone(), vision.clone()]);
        }
        (None, None) => ensure!(
            !assets.has_prompt_expansion_visual(),
            "visual expansion capability lacks actual preflight families"
        ),
        _ => anyhow::bail!("visual expansion requires both causal and HF vision families"),
    }
    // One full merge cell produces one main feature and all three DeepStack
    // rows. These are real visual/visual-TE programs over the original schemas,
    // so a preset cannot pass by advertising only its text-only operations.
    let merge = assets.vision.merge_size();
    let visual_config = Qwen3VLVisionComfyConfig::new(
        assets.vision.clone(),
        Qwen3VLVisionGrid::new(1, merge, merge)?,
    )?;
    families.push(
        TypedFamilyRegistration::new(Qwen3VLVisionComfyFamily::new(assets.limits.vision)?)
            .prepare_with_profile(
                &serde_json::to_value(visual_config)?,
                &profile(assets.main_vision_profile)?,
            )?,
    );
    // The feature occupies the middle row. Boundary-token embeddings and the
    // first real/sink token are outside the replacement/addition selection.
    let visual_text = PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
        assets.text.clone(),
        Qwen3VLTextLayout::new(3, vec![vec![Qwen3VLTextKeyRun::new(3, true)]])?,
        &assets.sources.text_encoder,
        &assets.connector,
    )?
    .with_visual_inputs(Qwen3VLTextDeepStack::new(vec![Qwen3VLTextVisualRun::new(
        1, 1,
    )?])?)?;
    families.push(
        TypedFamilyRegistration::new(PH3ImageTextConditioningFamily::new(assets.limits.text)?)
            .prepare_with_profile(
                &serde_json::to_value(&visual_text)?,
                &profile(PH3ImageTextConditioningFamily::cuda_profile_id(
                    &visual_text,
                ))?,
            )?,
    );
    let transformer = assets.denoiser.transformer();
    let [_, patch_h, patch_w] = transformer.patch_size();
    let target = PH3ImageReferenceImage::new(usize::try_from(patch_h)?, usize::try_from(patch_w)?)?;
    let tags = [PH3ImageModality::Text];
    let layout = PH3ImageRequestLayout::new(transformer, &tags, target, &[])?;
    let times = vec![0; layout.rows()];
    for phase in [
        PH3ImageDenoiserPhase::InstallFullWeights,
        PH3ImageDenoiserPhase::RefineFreshText,
        PH3ImageDenoiserPhase::DenoiseRefinedText,
    ] {
        let request = assets.denoiser.request(
            RequestIdentity::new("request.ph3.cuda.preflight")?,
            &tags,
            target,
            None,
            &times,
        )?;
        families.push(
            assets
                .denoiser
                .prepare(phase, 1, vec![request])?
                .family()
                .clone(),
        );
    }
    let minimum_pixels = usize::try_from(assets.spatial_scale())?
        .checked_mul(2)
        .context("H3 codec preflight dimensions overflow")?;
    families.push(
        assets
            .vae
            .prepare_encoder(
                1,
                minimum_pixels,
                minimum_pixels,
                PH3ImageVaeSpatialPolicy::WholeImage,
                options.vae_encoder,
            )?
            .family()
            .clone(),
    );
    // Cover every legal physical tile count, including a real final tail. These
    // small payload-free geometries establish the selected provider topology;
    // request preparation still compiles/admit-checks every actual full shape.
    let maximum_count = match options.vae_spatial_policy {
        PH3ImageVaeSpatialPolicy::WholeImage => 1,
        PH3ImageVaeSpatialPolicy::Tiled { max_tiles, .. } => {
            max_tiles.min(options.vae_decoder_tile_batch.maximum_tiles())
        }
    };
    ensure!(
        maximum_count > 0,
        "H3 decoder tile capacity must be positive"
    );
    for count in 1..=maximum_count {
        let scale = usize::try_from(assets.spatial_scale())?;
        let width = count
            .checked_add(1)
            .and_then(|n| n.checked_mul(scale))
            .context("H3 decoder preflight group width overflows")?;
        let policy = if count == 1 {
            PH3ImageVaeSpatialPolicy::WholeImage
        } else {
            PH3ImageVaeSpatialPolicy::Tiled {
                tile_pixels: minimum_pixels,
                minimum_overlap_pixels: scale,
                max_tiles: count,
            }
        };
        let recipes = assets.vae.prepare_decoder_with_tile_batch(
            1,
            width,
            minimum_pixels,
            policy,
            options.vae_decoder,
            options.vae_decoder_tile_batch,
        )?;
        for shape in recipes.shapes() {
            families.push(shape.family().clone());
        }
    }
    Ok(families)
}

pub(super) fn validate(
    families: &[PreparedModelFamily],
    contracts: &[Box<dyn OperationContract>],
    maximum_commands: usize,
) -> Result<u64> {
    let supported = contracts
        .iter()
        .map(|c| c.descriptor().id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        supported.len() == contracts.len(),
        "H3 CUDA contract inventory contains duplicate operations"
    );
    crate::image_generation::composition::validate_family_command_limits(
        families,
        maximum_commands,
    )?;
    let mut largest = 0u64;
    for family in families {
        for node in family.program().blocks().iter().flat_map(|b| &b.nodes) {
            ensure!(
                supported.contains(&node.operation_id),
                "H3 CUDA preset lacks operation {} required by {}",
                node.operation_id,
                family.family_id()
            );
        }
        if let ModelWeights::Stored(schema) = family.weights() {
            let bytes = schema.components.iter().try_fold(0u64, |sum, c| {
                sum.checked_add(c.physical_bytes()?)
                    .context("H3 execution weight bytes overflow")
            })?;
            largest = largest.max(bytes);
        }
    }
    Ok(largest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_models::{
        p_h3_image::PH3ImageTextConditioningLimits,
        qwen3_vl_text::{
            Qwen3VLTextArchitecture, Qwen3VLTextComfyConfig,
            Qwen3VLTextComfyProjectionQuantization, Qwen3VLTextConfig,
        },
        qwen3_vl_vision::{comfy::Qwen3VLVisionComfyLimits, Qwen3VLVisionArchitecture},
    };

    fn graphs() -> Vec<PreparedModelFamily> {
        let raw = serde_json::json!({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl", "vision_config":{
            "model_type":"qwen3_vl_vision","dtype":"bfloat16","hidden_act":"gelu_pytorch_tanh",
            "depth":3,"hidden_size":8,"intermediate_size":12,"num_heads":2,"num_position_embeddings":16,
            "out_hidden_size":256,"deepstack_visual_indexes":[0,1,2],"in_channels":3,"patch_size":2,
            "temporal_patch_size":2,"spatial_merge_size":2,"rope_parameters":{"rope_type":"axial","rope_theta":10000.0}},
            "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","attention_bias":false,"attention_dropout":0.0,
            "hidden_act":"silu","num_hidden_layers":5,"hidden_size":256,"intermediate_size":256,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":128,"vocab_size":16,
            "max_position_embeddings":128,"rms_norm_eps":0.25,"rope_theta":5000000,
            "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[64,0,0]}}});
        let vision = Qwen3VLVisionComfyConfig::new(
            Qwen3VLVisionArchitecture::from_hf_value(&raw).unwrap(),
            Qwen3VLVisionGrid::new(1, 2, 2).unwrap(),
        )
        .unwrap();
        let vision = TypedFamilyRegistration::new(
            Qwen3VLVisionComfyFamily::new(Qwen3VLVisionComfyLimits {
                max_metadata_bytes: 2 << 20,
                max_stored_bytes: 4 << 20,
                max_plan_nodes: 256,
            })
            .unwrap(),
        )
        .prepare_with_profile(
            &serde_json::to_value(vision).unwrap(),
            &profile(Qwen3VLVisionComfyFamily::CUDA_PROFILE_ID).unwrap(),
        )
        .unwrap();
        let text = Qwen3VLTextConfig::new(
            Qwen3VLTextArchitecture::from_hf_value(&raw).unwrap(),
            Qwen3VLTextLayout::new(3, vec![vec![Qwen3VLTextKeyRun::new(3, true)]]).unwrap(),
        )
        .unwrap()
        .with_hidden_state_indices(vec![1, 2, 3, 4, 5])
        .unwrap();
        let projections = text
            .expected_weights()
            .unwrap()
            .into_iter()
            .filter(|w| w.dimensions().len() == 2)
            .enumerate()
            .map(|(i, w)| {
                (
                    w.external_name()
                        .strip_suffix(".weight")
                        .unwrap()
                        .to_owned(),
                    if i % 2 == 0 {
                        Qwen3VLTextComfyProjectionQuantization::W4A8
                    } else {
                        Qwen3VLTextComfyProjectionQuantization::W8A8
                    },
                )
            })
            .collect();
        let text = PH3ImageTextConditioningConfig::new(
            Qwen3VLTextComfyConfig::new(text, projections).unwrap(),
            4,
        )
        .unwrap()
        .with_visual_inputs(
            Qwen3VLTextDeepStack::new(vec![Qwen3VLTextVisualRun::new(1, 1).unwrap()]).unwrap(),
        )
        .unwrap();
        let text = TypedFamilyRegistration::new(
            PH3ImageTextConditioningFamily::new(PH3ImageTextConditioningLimits {
                max_metadata_bytes: 1 << 20,
                max_stored_bytes: 16 << 20,
                max_plan_nodes: 256,
            })
            .unwrap(),
        )
        .prepare_with_profile(
            &serde_json::to_value(&text).unwrap(),
            &profile(PH3ImageTextConditioningFamily::cuda_profile_id(&text)).unwrap(),
        )
        .unwrap();
        vec![vision, text]
    }
    fn contracts() -> Vec<Box<dyn OperationContract>> {
        let mut contracts = super::super::operations::selected()
            .unwrap()
            .into_iter()
            .map(|(_, c)| c)
            .collect::<Vec<_>>();
        contracts.extend(super::super::operations::independent_contracts().unwrap());
        contracts
    }
    #[test]
    fn real_visual_and_visual_text_graphs_require_every_new_native_boundary() {
        let graphs = graphs();
        let commands = graphs
            .iter()
            .map(|f| crate::image_generation::composition::component_command_count(f).unwrap())
            .max()
            .unwrap();
        assert!(validate(&graphs, &contracts(), commands).unwrap() > 0);
        for missing in [
            PATCH_PROJECTION_F32_OPERATION_ID,
            RESHAPE_COPY_F32_OPERATION_ID,
            LAYER_NORM_F32_OPERATION_ID,
            GELU_TANH_F32_OPERATION_ID,
            GELU_ERF_F32_OPERATION_ID,
            ROW_SCATTER_REPLACE_F32_OPERATION_ID,
            ROW_SCATTER_ADD_F32_OPERATION_ID,
        ] {
            let mut contracts = contracts();
            contracts.retain(|c| c.descriptor().id.as_str() != missing);
            assert!(
                format!("{:#}", validate(&graphs, &contracts, commands).unwrap_err())
                    .contains(missing)
            );
        }
        assert!(validate(&graphs, &contracts(), commands - 1).is_err());
    }
}
