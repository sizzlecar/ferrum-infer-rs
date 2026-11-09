use super::*;
use crate::image_generation::{ImageGenerationRequest, ImagePipelineConfig};
use ferrum_models::{
    p_h3_image::{PH3ImageDenoiserPhase, PH3ImageModality, PH3ImageReferenceImage},
    p_h3_image_vae::{
        PH3ImageVaeDecoderPreset, PH3ImageVaeEncoderPreset, PH3ImageVaeSpatialPolicy,
    },
    qwen3_vl_vision::comfy::Qwen3VLVisionComfyPixelPolicy,
};
use ferrum_runtime::{RequestIdentity, StaticInitializationPolicy};
use sha2::{Digest, Sha256};
use std::sync::Arc;

mod conditioning_family;
#[cfg(feature = "cuda")]
mod cuda_expansion;
#[cfg(feature = "cuda")]
pub(in crate::image_generation::p_h3) mod cuda_pipeline;
pub(in crate::image_generation::p_h3) mod fixture;

fn config() -> ImagePipelineConfig {
    ImagePipelineConfig {
        device_capacity_bytes: 1 << 30,
        maximum_user_tokens: 512,
        maximum_width: 64,
        maximum_height: 64,
        maximum_batch_size: 2,
        weight_arena_target_bytes: None,
        decoder_batch_policy: Default::default(),
        edit_prefix_policy: Default::default(),
        plan_cache: Default::default(),
        residency: Default::default(),
        dynamic_residency: Default::default(),
        static_initialization: StaticInitializationPolicy::new(256, 8).unwrap(),
        executor: Default::default(),
    }
}

#[test]
fn decoder_tile_batch_keeps_one_image_and_queries_every_real_group_without_host_grants(
) -> anyhow::Result<()> {
    use ferrum_models::p_h3_image_vae::PH3ImageVaeTileBatch;
    use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};

    let fixture = fixture::CompleteFixture::new()?;
    let assets = fixture.assets()?;
    let mut options = PH3ImagePipelineOptions::comfy_f32(
        ferrum_runtime::NumericalProfileId::new("fixture.h3.noise.f32")
            .map_err(anyhow::Error::msg)?,
        32,
    )?;
    assert_eq!(options.vae_decoder_tile_batch.maximum_tiles(), 1);
    options.vae_spatial_policy = PH3ImageVaeSpatialPolicy::Tiled {
        tile_pixels: 32,
        minimum_overlap_pixels: 16,
        max_tiles: 32,
    };
    let serial = assets.vae.prepare_decoder_with_tile_batch(
        1,
        96,
        64,
        options.vae_spatial_policy,
        options.vae_decoder,
        options.vae_decoder_tile_batch,
    )?;
    options.vae_decoder_tile_batch = PH3ImageVaeTileBatch::new(4)?;
    let grouped = assets.vae.prepare_decoder_with_tile_batch(
        1,
        96,
        64,
        options.vae_spatial_policy,
        options.vae_decoder,
        options.vae_decoder_tile_batch,
    )?;
    assert_eq!(grouped.tile_count(), 15);
    assert_eq!(
        grouped
            .shapes()
            .iter()
            .map(|shape| (
                shape.request_batch(),
                shape.tile_count(),
                shape.physical_batch()
            ))
            .collect::<Vec<_>>(),
        [(1, 4, 4), (1, 3, 3)]
    );
    assert_eq!(
        grouped.shapes()[0].family().stored_weight_schema()?,
        grouped.shapes()[1].family().stored_weight_schema()?
    );
    assert_eq!(
        grouped.normalized_latent_spec()?,
        serial.normalized_latent_spec()?
    );
    assert_eq!(grouped.pixel_output_spec()?, serial.pixel_output_spec()?);
    assert_eq!(
        grouped.normalized_latent_spec()?.dimensions(),
        &[1, 24, 4, 6]
    );
    assert_eq!(grouped.pixel_output_spec()?.dimensions(), &[1, 3, 64, 96]);
    // A query uses the actual alignment, without demanding a grant or payload.
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )?,
        64,
    )?;
    let serial = serial.host_requirements(&host)?;
    let grouped = grouped.host_requirements(&host)?;
    assert_eq!(grouped.canvas_output_bytes, serial.canvas_output_bytes);
    assert!(grouped.input_peak_bytes > serial.input_peak_bytes);
    assert!(grouped.output_peak_bytes >= serial.output_peak_bytes);
    assert_eq!(host.usage()?.committed_bytes(), 0);
    Ok(())
}

#[test]
fn complete_original_assets_prepare_generation_and_edit_recipes() -> anyhow::Result<()> {
    let fixture = fixture::CompleteFixture::new()?;
    let assets = fixture.assets()?;
    assert!(Arc::ptr_eq(
        &assets.sources.text_encoder,
        &fixture.conditioning.source()
    ));
    assert!(Arc::ptr_eq(
        &assets.sources.transformer,
        &fixture.denoiser.source()
    ));
    assert!(Arc::ptr_eq(&assets.sources.vae, &fixture.vae.source()));
    assert_eq!((assets.latent_channels(), assets.spatial_scale()), (24, 16));
    assert_eq!(assets.schedule.evaluation_count(), 2);
    assert_eq!(
        assets.receipt().metadata_sha256["text_connector"],
        format!("{:x}", Sha256::digest(&fixture.connector))
    );
    let request = ImageGenerationRequest {
        prompt: "cat".into(),
        width: 32,
        height: 32,
        seed: 42,
        n: 1,
    };
    let generate = assets.prepare_text_conditioning(
        &request,
        RequestIdentity::new("request.h3.complete.generate")?,
        &config(),
    )?;
    let edit = assets.prepare_image_conditioning(
        &request,
        RequestIdentity::new("request.h3.complete.edit")?,
        &config(),
        32,
        32,
        PH3ImageReferenceImage::new(2, 2)?,
        Qwen3VLVisionComfyPixelPolicy::default(),
    )?;
    assert!(generate.prompts.visual().is_none());
    assert!(edit.prompts.visual().is_some());
    assert!(edit.geometry.tags.contains(&PH3ImageModality::Image));
    assert_eq!(
        generate.config.encoder().text().hidden_state_indices(),
        &[1, 2, 3, 4, 5]
    );
    assert_eq!(
        generate.config.connector_output_dim(),
        assets.denoiser.transformer().text_dim()
    );
    assert_eq!(
        generate.encoder.stored_weight_schema()?,
        edit.encoder.stored_weight_schema()?
    );
    assert_eq!(
        generate.connector.stored_weight_schema()?,
        edit.connector.stored_weight_schema()?
    );
    let batch = request::BatchGeometry::new(
        assets.denoiser.transformer(),
        vec![&generate.geometry, &edit.geometry],
        2,
        assets.limits.denoiser.max_metadata_bytes,
    )?;
    let install = batch.prepare(&assets.denoiser, PH3ImageDenoiserPhase::InstallFullWeights)?;
    for phase in [
        PH3ImageDenoiserPhase::RefineFreshText,
        PH3ImageDenoiserPhase::DenoiseRefinedText,
    ] {
        let recipe = batch.prepare(&assets.denoiser, phase)?;
        assert_eq!(
            install.family().stored_weight_schema()?,
            recipe.family().stored_weight_schema()?
        );
    }
    let encoder = assets.vae.prepare_encoder(
        1,
        32,
        32,
        PH3ImageVaeSpatialPolicy::WholeImage,
        PH3ImageVaeEncoderPreset::CudaF32Fma,
    )?;
    let decoder = assets.vae.prepare_decoder(
        1,
        32,
        32,
        PH3ImageVaeSpatialPolicy::WholeImage,
        PH3ImageVaeDecoderPreset::F32A8FullAttention,
    )?;
    let latent = encoder.normalized_latent_spec()?;
    assert_eq!(latent, decoder.normalized_latent_spec()?);
    assert_eq!(latent.dimensions(), &[1, 24, 2, 2]);
    assert_eq!(
        generate.geometry.image_sample.dimensions(),
        &[1, 24, 1, 2, 2]
    );
    assert_eq!(
        latent.minimum_storage_bytes()?,
        generate.geometry.image_sample.minimum_storage_bytes()?
    );
    assert_eq!(decoder.pixel_output_spec()?.dimensions(), &[1, 3, 32, 32]);
    Ok(())
}

#[test]
fn complete_assets_preserve_metadata_provenance_and_reject_incompatible_components(
) -> anyhow::Result<()> {
    let mut fixture = fixture::CompleteFixture::new()?;
    let original = fixture.assets()?;
    let mut connector: serde_json::Value = serde_json::from_slice(&fixture.connector)?;
    connector["_diffusers_version"] = serde_json::json!("fixture-next-version");
    fixture.connector = serde_json::to_vec(&connector)?;
    let changed_version = fixture.assets()?;
    assert_ne!(
        original.receipt().metadata_sha256["text_connector"],
        changed_version.receipt().metadata_sha256["text_connector"]
    );
    assert_eq!(
        original.receipt().required_weight_bytes,
        changed_version.receipt().required_weight_bytes
    );
    connector["out_channels"] = serde_json::json!(8);
    fixture.connector = serde_json::to_vec(&connector)?;
    assert!(format!(
        "{:#}",
        fixture.assets().err().expect("incompatible DiT width")
    )
    .contains("output width differs from transformer"));
    connector["out_channels"] = serde_json::json!(4);
    connector["hidden_state_indices"] = serde_json::json!([1, 2, 3, 4]);
    fixture.connector = serde_json::to_vec(&connector)?;
    assert!(format!(
        "{:#}",
        fixture
            .assets()
            .err()
            .expect("missing normalized final tap")
    )
    .contains("five taps ending at the final normalized"));
    Ok(())
}
