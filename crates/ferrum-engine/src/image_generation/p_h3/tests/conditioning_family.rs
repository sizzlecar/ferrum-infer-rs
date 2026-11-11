//! Actual original-format tiny assets exercise metadata preparation, not a
//! complete H3 CPU runtime or a fabricated successful component forward.
use super::*;
use ferrum_runtime::{
    ModelFamilyProvider, ModelFamilyRegistration, NumericalProfileId, PreparedModelFamily,
    TypedFamilyRegistration,
};

fn build<P: ModelFamilyProvider>(
    provider: P,
    config: &P::Config,
    profile: &NumericalProfileId,
) -> anyhow::Result<PreparedModelFamily> {
    Ok(TypedFamilyRegistration::new(provider)
        .prepare_with_profile(&serde_json::to_value(config)?, profile)?)
}

fn request() -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: "cat".into(),
        width: 32,
        height: 32,
        seed: 42,
        n: 1,
    }
}

#[test]
fn conditioning_family_preparer_preserves_real_families_and_rebinds_only_request_identity(
) -> anyhow::Result<()> {
    let fixture = fixture::CompleteFixture::new()?;
    let assets = fixture.assets()?;
    let request = request();
    let cold = assets.prepare_text_conditioning(
        &request,
        RequestIdentity::new("request.conditioning.cold")?,
        &config(),
    )?;
    let rebound = assets.prepare_text_conditioning_with_family(
        &request,
        RequestIdentity::new("request.conditioning.new")?,
        &config(),
        |provider, config, profile| {
            assert_eq!(build(provider, config, profile)?, cold.encoder);
            Ok(cold.encoder.clone())
        },
        |provider, config, profile| {
            assert_eq!(build(provider, config, profile)?, cold.connector);
            Ok(cold.connector.clone())
        },
    )?;
    assert_eq!(rebound.encoder, cold.encoder);
    assert_eq!(rebound.connector, cold.connector);
    assert_eq!(rebound.encoder_options, cold.encoder_options);
    assert_eq!(rebound.connector_options, cold.connector_options);
    assert_eq!(
        rebound.geometry.identity.as_str(),
        "request.conditioning.new"
    );
    assert_eq!(
        rebound.prompts.text().token_ids(),
        cold.prompts.text().token_ids()
    );
    Ok(())
}

#[test]
fn conditioning_family_preparer_keeps_text_visual_domains_and_fresh_connector_row_options(
) -> anyhow::Result<()> {
    let fixture = fixture::CompleteFixture::new()?;
    let assets = fixture.assets()?;
    let mut request = request();
    let cold = assets.prepare_text_conditioning(
        &request,
        RequestIdentity::new("request.conditioning.cold")?,
        &config(),
    )?;
    request.prompt = "cat cat".into();
    assert!(assets
        .prepare_text_conditioning_with_family(
            &request,
            RequestIdentity::new("request.conditioning.wrong_text")?,
            &config(),
            |_, _, _| Ok(cold.encoder.clone()),
            |_, _, _| Ok(cold.connector.clone())
        )
        .is_err());
    let longer = assets.prepare_text_conditioning_with_family(
        &request,
        RequestIdentity::new("request.conditioning.longer")?,
        &config(),
        build,
        |_, _, _| Ok(cold.connector.clone()),
    )?;
    assert_ne!(longer.encoder, cold.encoder);
    assert_eq!(longer.connector, cold.connector);
    assert_ne!(longer.connector_options, cold.connector_options);
    let image = assets.prepare_image_conditioning_with_family(
        &request,
        RequestIdentity::new("request.conditioning.image")?,
        &config(),
        32,
        32,
        PH3ImageReferenceImage::new(2, 2)?,
        Qwen3VLVisionComfyPixelPolicy::default(),
        |_, _, _| Ok(cold.encoder.clone()),
        |_, _, _| Ok(cold.connector.clone()),
    );
    assert!(
        image.is_err(),
        "a cached text-only family cannot bind visual/deepstack inputs"
    );
    Ok(())
}
