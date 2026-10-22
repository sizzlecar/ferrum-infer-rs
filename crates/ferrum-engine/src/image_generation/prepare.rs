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
    pub(super) fn prepare(
        &self,
        request: &ImageGenerationRequest,
    ) -> anyhow::Result<PreparedRequest<R>> {
        let (height, width) =
            request.latent_dimensions(&self.config, self.assets.spatial_scale()?)?;
        let prompts = self
            .assets
            .tokenizer
            .prepare(&[request.prompt.as_str()], self.config.maximum_user_tokens)?;
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
        let decoder = self.assets.decoder(height, width)?;
        ensure!(
            decoder.output_tensor()?.dimensions()
                == [1, 4, u64::from(request.height), u64::from(request.width)],
            "decoder output does not reproduce exact requested size"
        );
        let noise = NormalNoiseConfig::new(64, u64::from(height), u64::from(width))?;
        let latent_shape =
            QwenImage21LatentShape::new(1, usize::try_from(height)?, usize::try_from(width)?)?;
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
