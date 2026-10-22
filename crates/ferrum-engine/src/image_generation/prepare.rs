use super::*;

pub(super) struct PreparedRequest<R: DeviceRuntime> {
    pub prompts: QwenImage21T2iPromptBatch,
    pub transformer: QwenImage21TransformerConfig,
    pub decoder: QwenImage21VaeDecoderConfig,
    pub latent_shape: QwenImage21LatentShape,
    pub text_stage: PreparedStage<R>,
    pub noise_stage: PreparedStage<R>,
    pub denoise_stage: PreparedStage<R>,
    pub decode_stage: PreparedStage<R>,
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    /// No payload allocation or device submission occurs during preparation.
    /// Missing noise/decoder capabilities are discovered before TE uploads.
    pub(super) fn prepare_batch(
        &self,
        requests: &[ImageGenerationRequest],
    ) -> anyhow::Result<PreparedRequest<R>> {
        let (height, width) =
            batch_dimensions(requests, &self.config, self.assets.spatial_scale()?)?;
        let request = &requests[0];
        let batch = requests.len();
        let prompt_text = requests
            .iter()
            .map(|r| r.prompt.as_str())
            .collect::<Vec<_>>();
        let prompts = self
            .assets
            .tokenizer
            .prepare(&prompt_text, self.config.maximum_user_tokens)?;
        let text = Qwen3VLTextConfig::new(self.assets.text.clone(), prompts.layout().clone())?;
        let layout = QwenImage21T2iLayout::new(
            u32::try_from(prompts.feature_sequence_length())?,
            height,
            width,
            prompts
                .valid_feature_lengths()
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?,
        )?;
        let transformer =
            QwenImage21TransformerConfig::new(self.assets.transformer.clone(), layout)?;
        let decoder = self.assets.decoder_batch(batch, height, width)?;
        ensure!(
            decoder.output_tensor()?.dimensions()
                == [
                    u64::try_from(batch)?,
                    4,
                    u64::from(request.height),
                    u64::from(request.width)
                ],
            "decoder output does not reproduce exact requested size"
        );
        let noise = NormalNoiseConfig::new(64, u64::from(height), u64::from(width))?;
        let latent_shape =
            QwenImage21LatentShape::new(batch, usize::try_from(height)?, usize::try_from(width)?)?;
        let text_family = TypedFamilyRegistration::new(Qwen3VLTextEncoderFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(&text)?, &self.profiles.text_encoder)?;
        let noise_family =
            TypedFamilyRegistration::new(NormalNoiseFamily::new(self.profiles.noise.as_str())?)
                .prepare_with_profile(&serde_json::to_value(&noise)?, &self.profiles.noise)?;
        let dit_family = TypedFamilyRegistration::new(QwenImage21TransformerFamily::new()?)
            .prepare_with_profile(
                &serde_json::to_value(&transformer)?,
                &self.profiles.denoiser,
            )?;
        let vae_family = TypedFamilyRegistration::new(QwenImage21VaeDecoderFamily::new()?)
            .prepare_with_profile(&serde_json::to_value(&decoder)?, &self.profiles.decoder)?;
        let text_stage = PreparedStage::compile(
            ImageStage::TextEncoder,
            text_family,
            &text.compile_options()?,
            self,
        )?;
        let noise_stage = PreparedStage::compile(
            ImageStage::Noise,
            noise_family,
            &noise.compile_options()?,
            self,
        )?;
        let denoise_stage = PreparedStage::compile(
            ImageStage::Denoiser,
            dit_family,
            &transformer.compile_options()?,
            self,
        )?;
        let decode_stage = PreparedStage::compile(
            ImageStage::Decoder,
            vae_family,
            &decoder.compile_options()?,
            self,
        )?;
        Ok(PreparedRequest {
            prompts,
            transformer,
            decoder,
            latent_shape,
            text_stage,
            noise_stage,
            denoise_stage,
            decode_stage,
        })
    }
}

pub(super) fn batch_dimensions(
    requests: &[ImageGenerationRequest],
    config: &ImagePipelineConfig,
    scale: u32,
) -> anyhow::Result<(u32, u32)> {
    ensure!(
        (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&requests.len()),
        "image request batch exceeds the declared attention layout capacity"
    );
    let first = requests[0].latent_dimensions(config, scale)?;
    for request in &requests[1..] {
        ensure!(
            request.latent_dimensions(config, scale)? == first,
            "request batch requires equal image dimensions"
        );
    }
    Ok(first)
}
