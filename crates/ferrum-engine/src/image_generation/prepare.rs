use super::*;
use ferrum_models::qwen_image21::QwenImage21TransformerPhase;

pub(super) struct PreparedPrefixDecode<R: DeviceRuntime> {
    pub config: QwenImage21TransformerConfig,
    pub stage: PreparedStage<R>,
}

pub(super) struct PreparedRequest<R: DeviceRuntime> {
    pub host_scope: Option<ferrum_runtime::HostMemoryScope>,
    pub host_backing: Vec<ferrum_runtime::HostBackingPreparation>,
    pub prompts: QwenImage21T2iPromptBatch,
    pub request_order: BatchOrder,
    pub transformer: QwenImage21TransformerConfig,
    pub decoder: QwenImage21VaeDecoderConfig,
    pub latent_shape: QwenImage21LatentShape,
    pub text_stage: PreparedStage<R>,
    pub noise_stage: PreparedStage<R>,
    pub denoise_stage: PreparedStage<R>,
    pub prefix_decode: Option<PreparedPrefixDecode<R>>,
    pub decode_stage: PreparedStage<R>,
    pub edit: Option<super::editing::PreparedEdit<R>>,
}

/// Execution-to-request mapping. This only moves bounded owners/metadata, never
/// copies tensor payloads or changes the exact mask used to compile a plan.
pub(super) struct BatchOrder {
    request_indices: Vec<usize>,
}
impl BatchOrder {
    pub(super) fn new(request_indices: Vec<usize>, batch: usize) -> anyhow::Result<Self> {
        ensure!(
            (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&batch)
                && request_indices.len() == batch,
            "canonical batch permutation has an invalid length"
        );
        let mut seen = [false; STRUCTURED_ATTENTION_BF16_MAX_BATCH];
        for &index in &request_indices {
            ensure!(
                index < batch && !seen[index],
                "canonical batch permutation must contain each request exactly once"
            );
            seen[index] = true;
        }
        Ok(Self { request_indices })
    }

    pub(super) fn requests<'a>(
        &'a self,
        requests: &'a [ImageGenerationRequest],
    ) -> anyhow::Result<impl Iterator<Item = &'a ImageGenerationRequest>> {
        ensure!(
            requests.len() == self.request_indices.len(),
            "request count differs from its canonical batch permutation"
        );
        Ok(self.request_indices.iter().map(|&index| &requests[index]))
    }

    pub(super) fn restore_outputs(
        &self,
        outputs: Vec<TensorExecutionOutput>,
    ) -> anyhow::Result<Vec<TensorExecutionOutput>> {
        ensure!(
            outputs.len() == self.request_indices.len(),
            "decoder output count differs from its canonical batch permutation"
        );
        let mut indexed: Vec<_> = self.request_indices.iter().copied().zip(outputs).collect();
        indexed.sort_unstable_by_key(|(index, _)| *index);
        Ok(indexed.into_iter().map(|(_, output)| output).collect())
    }
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    /// No payload allocation or device submission occurs during preparation.
    /// Missing noise/decoder capabilities are discovered before TE uploads.
    pub(super) fn prepare_batch(
        &self,
        requests: &[ImageGenerationRequest],
    ) -> anyhow::Result<PreparedRequest<R>> {
        self.prepare_batch_with_edit(requests, None)
    }

    pub(super) fn prepare_batch_with_edit(
        &self,
        requests: &[ImageGenerationRequest],
        edit: Option<super::editing::EditInputs>,
    ) -> anyhow::Result<PreparedRequest<R>> {
        let candidate_cache = self.plan_cache.begin_candidate()?;
        let prepare_started = std::time::Instant::now();
        let (height, width) =
            batch_dimensions(requests, &self.config, self.assets.spatial_scale()?)?;
        let request = &requests[0];
        let batch = requests.len();
        let prompt_text = requests
            .iter()
            .map(|r| r.prompt.as_str())
            .collect::<Vec<_>>();
        let (prompts, order) = if let Some(edit) = &edit {
            edit.dimensions(batch)?;
            // Reference inputs and vision outputs use caller order. Do not
            // apply generation's prompt-length canonicalization to edits.
            (edit.prompt.prompts().clone(), (0..batch).collect())
        } else {
            self.assets
                .tokenizer
                .prepare_canonical(&prompt_text, self.config.maximum_user_tokens)?
        };
        let request_order = BatchOrder::new(order, batch)?;
        let text = if let Some(edit) = &edit {
            edit.prompt.text_config(&self.assets.text)?
        } else {
            Qwen3VLTextConfig::new(self.assets.text.clone(), prompts.layout().clone())?
        };
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
        let mut transformer =
            QwenImage21TransformerConfig::new(self.assets.transformer.clone(), layout)?
                .with_weight_encodings(self.assets.transformer_encodings.clone())?;
        if let Some(edit) = &edit {
            let (reference_height, reference_width) = edit.dimensions(batch)?;
            transformer = transformer.with_reference_image(
                edit.prompt.image_slot_start(),
                edit.prompt.image_slots(),
                reference_height / 16,
                reference_width / 16,
            )?;
        }
        let prefix_decode_config = if edit.is_some()
            && self.config.edit_prefix_policy == ImageEditPrefixPolicy::RequestCache
        {
            let decode = transformer
                .clone()
                .with_execution_phase(QwenImage21TransformerPhase::PrefixDecode)?;
            transformer =
                transformer.with_execution_phase(QwenImage21TransformerPhase::PrefixPrefill)?;
            Some(decode)
        } else {
            None
        };
        let decoder_batch = self.config.decoder_batch_policy.batch_size(batch);
        let decoder = self.assets.decoder_batch(decoder_batch, height, width)?;
        ensure!(
            decoder.output_tensor()?.dimensions()
                == [
                    u64::try_from(decoder_batch)?,
                    4,
                    u64::from(request.height),
                    u64::from(request.width)
                ],
            "decoder output does not reproduce exact requested size"
        );
        let noise = NormalNoiseConfig::new(64, u64::from(height), u64::from(width))?;
        let latent_shape =
            QwenImage21LatentShape::new(batch, usize::try_from(height)?, usize::try_from(width)?)?;
        let text_family = self.prepare_family(
            ImageStage::TextEncoder,
            ImageStagePlanRole::WholeStage,
            batch,
            Qwen3VLTextEncoderFamily::new()?,
            &text,
            &self.profiles.text_encoder,
        )?;
        let noise_family = self.prepare_family(
            ImageStage::Noise,
            ImageStagePlanRole::WholeStage,
            1,
            NormalNoiseFamily::new(self.profiles.noise.as_str())?,
            &noise,
            &self.profiles.noise,
        )?;
        let vae_family = self.prepare_family(
            ImageStage::Decoder,
            ImageStagePlanRole::WholeStage,
            decoder_batch,
            QwenImage21VaeDecoderFamily::new()?,
            &decoder,
            &self.profiles.decoder,
        )?;
        let started = std::time::Instant::now();
        let mut text_options = text.compile_options()?;
        let text_compile_options_us = trace_elapsed_us(started);
        let started = std::time::Instant::now();
        if Qwen3VLTextEncoderFamily::is_fp8_profile(self.profiles.text_encoder.as_str()) {
            let conversion = precision::materializer(
                precision::TEXT_MATERIALIZER,
                &self.host,
                Qwen3VLTextEncoderFamily::fp8_weight_groups(&text)?,
            )?;
            precision::select(&text_family, &mut text_options, &conversion, &self.host)?;
        }
        let text_fp8_select_us = trace_elapsed_us(started);
        let text_stage = PreparedStage::compile(
            ImageStage::TextEncoder,
            batch,
            text_family,
            &text_options,
            self,
        )?;
        let noise_stage = PreparedStage::compile(
            ImageStage::Noise,
            1,
            noise_family,
            &noise.compile_options()?,
            self,
        )?;
        let mut denoise_stage = self.prepare_denoiser(&transformer, batch)?;
        let prefix_decode = prefix_decode_config
            .map(|config| {
                let stage = self.prepare_denoiser(&config, batch)?;
                Ok::<_, anyhow::Error>(PreparedPrefixDecode { config, stage })
            })
            .transpose()?;
        if let Some(prefix) = &prefix_decode {
            denoise_stage.bind_prefix_admission(&prefix.stage)?;
        }
        let decode_stage = PreparedStage::compile(
            ImageStage::Decoder,
            decoder_batch,
            vae_family,
            &decoder.compile_options()?,
            self,
        )?;
        let edit = edit.map(|edit| edit.compile(self)).transpose()?;
        let mut all_stages = vec![&text_stage, &noise_stage, &denoise_stage, &decode_stage];
        if let Some(edit) = &edit {
            all_stages.extend([&edit.vision_stage, &edit.encoder_stage]);
        }
        if let Some(prefix) = &prefix_decode {
            all_stages.push(&prefix.stage);
        }
        let started = std::time::Instant::now();
        self.validate_capacity_with_prefix(
            &all_stages,
            prefix_decode
                .as_ref()
                .map(|prefix| (&denoise_stage, &prefix.stage)),
        )?;
        let capacity_us = trace_elapsed_us(started);
        // This total includes family/plan preparation and the listed subspans.
        // It must not be added to those subspans or stage compile receipts.
        tracing::debug!(
            target: "ferrum_engine::image_generation::timing",
            job = self.next_job,
            batch_size = batch,
            prepare_batch_us = trace_elapsed_us(prepare_started),
            text_compile_options_us,
            text_fp8_select_us,
            capacity_us,
            "image batch plans prepared"
        );
        let mut prepared = PreparedRequest {
            host_scope: None,
            host_backing: Vec::new(),
            prompts,
            request_order,
            transformer,
            decoder,
            latent_shape,
            text_stage,
            noise_stage,
            denoise_stage,
            prefix_decode,
            decode_stage,
            edit,
        };
        prepared.host_scope = Some(self.reserve_execution_host(&mut prepared)?);
        candidate_cache.commit();
        Ok(prepared)
    }

    pub(super) fn prepare_denoiser(
        &self,
        transformer: &QwenImage21TransformerConfig,
        batch: usize,
    ) -> anyhow::Result<PreparedStage<R>> {
        let role = match transformer.execution_phase() {
            QwenImage21TransformerPhase::Full => ImageStagePlanRole::WholeStage,
            QwenImage21TransformerPhase::PrefixPrefill => ImageStagePlanRole::PrefixPrefill,
            QwenImage21TransformerPhase::PrefixDecode => ImageStagePlanRole::PrefixDecode,
        };
        let family = self.prepare_family(
            ImageStage::Denoiser,
            role,
            batch,
            QwenImage21TransformerFamily::new()?,
            transformer,
            &self.profiles.denoiser,
        )?;
        let started = std::time::Instant::now();
        let mut options = transformer.compile_options()?;
        let compile_options_us = trace_elapsed_us(started);
        let started = std::time::Instant::now();
        if QwenImage21TransformerFamily::is_fp8_profile(self.profiles.denoiser.as_str()) {
            let conversion = precision::materializer(
                precision::DIT_MATERIALIZER,
                &self.host,
                QwenImage21TransformerFamily::fp8_weight_groups(transformer)?,
            )?;
            precision::select(&family, &mut options, &conversion, &self.host)?;
        }
        tracing::debug!(
            target: "ferrum_engine::image_generation::timing",
            job = self.next_job,
            ?role,
            batch_size = batch,
            compile_options_us,
            fp8_select_us = trace_elapsed_us(started),
            "image denoiser options prepared"
        );
        PreparedStage::compile_with_role(ImageStage::Denoiser, role, batch, family, &options, self)
    }
}

pub(super) fn batch_dimensions(
    requests: &[ImageGenerationRequest],
    config: &ImagePipelineConfig,
    scale: u32,
) -> anyhow::Result<(u32, u32)> {
    config.validate()?;
    ensure!(
        (1..=config.maximum_batch_size).contains(&requests.len()),
        "image request batch exceeds configured maximum_batch_size"
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
