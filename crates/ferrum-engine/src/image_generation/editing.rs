//! Additional managed conditioning stages for one reference image. All six
//! plans are compiled and admitted before any model upload or device execution.
use super::*;
use ferrum_models::qwen3_vl_vision::{Qwen3VLVisionConfig, Qwen3VLVisionFamily, Qwen3VLVisionGrid};
use ferrum_models::qwen_image21::image_input::{
    ImageInputLimits, ManagedRgba8Image, QwenImage21ConditionInputs,
};
use ferrum_models::qwen_image21::t2i_prompt::QwenImage21EditPrompt;
use ferrum_models::qwen_image21::{QwenImage21VaeEncoderConfig, QwenImage21VaeEncoderFamily};

/// Reuse only the causally isolated, timestep-zero prefix within one edit.
/// Cache storage is admitted as fixed Request state and retired before VAE decode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageEditPrefixPolicy {
    #[default]
    Recompute,
    RequestCache,
}

#[derive(Debug, Clone)]
pub struct QwenImage21EditProfiles {
    pub vision: NumericalProfileId,
    pub encoder: NumericalProfileId,
}

pub(super) struct EditInputs {
    pub prompt: QwenImage21EditPrompt,
    pub inputs: QwenImage21ConditionInputs,
    vision: Qwen3VLVisionConfig,
    encoder: QwenImage21VaeEncoderConfig,
}
pub(super) struct PreparedEdit<R: DeviceRuntime> {
    pub prompt: QwenImage21EditPrompt,
    pub inputs: QwenImage21ConditionInputs,
    pub vision: Qwen3VLVisionConfig,
    pub encoder: QwenImage21VaeEncoderConfig,
    pub vision_stage: PreparedStage<R>,
    pub encoder_stage: PreparedStage<R>,
}
impl EditInputs {
    pub fn compile<R: DeviceRuntime>(
        self,
        pipeline: &QwenImage21Pipeline<R>,
    ) -> anyhow::Result<PreparedEdit<R>> {
        let profiles = pipeline
            .editing_profiles
            .as_ref()
            .context("editing numerical profiles are not configured")?;
        let vision_family = pipeline.prepare_family(
            ImageStage::VisionEncoder,
            ImageStagePlanRole::WholeStage,
            1,
            Qwen3VLVisionFamily::new()?,
            &self.vision,
            &profiles.vision,
        )?;
        let encoder_family = pipeline.prepare_family(
            ImageStage::VaeEncoder,
            ImageStagePlanRole::WholeStage,
            1,
            QwenImage21VaeEncoderFamily::new()?,
            &self.encoder,
            &profiles.encoder,
        )?;
        let vision_stage = PreparedStage::compile(
            ImageStage::VisionEncoder,
            1,
            vision_family,
            &self.vision.compile_options()?,
            pipeline,
        )?;
        let encoder_stage = PreparedStage::compile(
            ImageStage::VaeEncoder,
            1,
            encoder_family,
            &self.encoder.compile_options()?,
            pipeline,
        )?;
        Ok(PreparedEdit {
            prompt: self.prompt,
            inputs: self.inputs,
            vision: self.vision,
            encoder: self.encoder,
            vision_stage,
            encoder_stage,
        })
    }
}
impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    /// Numerical identities come from the composition, independently of model
    /// names or runtime IDs. Missing weights cannot be enabled by this switch.
    pub fn with_editing_profiles(
        mut self,
        profiles: QwenImage21EditProfiles,
    ) -> anyhow::Result<Self> {
        ensure!(
            self.assets.editing.is_some(),
            "editing profiles require validated vision/encoder assets"
        );
        self.editing_profiles = Some(profiles);
        Ok(self)
    }
    pub fn supports_edits(&self) -> bool {
        self.assets.editing.is_some() && self.editing_profiles.is_some()
    }

    pub fn edit(
        &mut self,
        request: ImageEditRequest,
    ) -> Result<ImageGenerationOutput, ImageGenerationError<R>> {
        let preflight = |source| ImageGenerationError {
            failure: ImageGenerationFailure::Preflight(source),
            completed_stages: Vec::new(),
            completed_steps: Vec::new(),
        };
        if self.stopped {
            return Err(ImageGenerationError {
                failure: ImageGenerationFailure::Unavailable,
                completed_stages: Vec::new(),
                completed_steps: Vec::new(),
            });
        }
        let prepared = (|| -> anyhow::Result<_> {
            ensure!(self.supports_edits(), "image editing is not configured");
            request
                .generation
                .latent_dimensions(&self.config, self.assets.spatial_scale()?)?;
            let assets = self
                .assets
                .editing
                .as_ref()
                .context("editing assets are missing")?;
            let limits = ImageInputLimits::default();
            let started = std::time::Instant::now();
            let decoded = ManagedRgba8Image::decode_png(&self.host, &request.image, limits)?;
            let decode_png_us = trace_elapsed_us(started);
            let started = std::time::Instant::now();
            let inputs = assets.processor.prepare(&self.host, &decoded, limits)?;
            let reference_preprocess_us = trace_elapsed_us(started);
            drop(decoded);
            let started = std::time::Instant::now();
            let [temporal, height, width] = inputs.grid_thw();
            let vision = Qwen3VLVisionConfig::new(
                assets.vision.clone(),
                vec![Qwen3VLVisionGrid::new(temporal, height, width)?],
            )?;
            let encoder = assets.encoder_config(inputs.height(), inputs.width())?;
            let prompt = self.assets.tokenizer.prepare_edit(
                &request.generation.prompt,
                self.config.maximum_user_tokens,
                height,
                width,
                assets.vision.merge_size(),
            )?;
            let edit_layout_and_prompt_us = trace_elapsed_us(started);
            let prepared = self.prepare_batch_with_edit(
                std::slice::from_ref(&request.generation),
                Some(EditInputs {
                    prompt,
                    inputs,
                    vision,
                    encoder,
                }),
            )?;
            tracing::debug!(
                target: "ferrum_engine::image_generation::timing",
                job = self.next_job,
                decode_png_us,
                reference_preprocess_us,
                edit_layout_and_prompt_us,
                "image edit inputs prepared"
            );
            Ok(prepared)
        })()
        .map_err(preflight)?;
        // Encoded bytes are no longer needed; decoded model inputs own their
        // separate claims. Queue cancellation drops this same admitted owner.
        drop(request.image);
        let mut outputs =
            self.execute_prepared(prepared, std::slice::from_ref(&request.generation))?;
        Ok(outputs.remove(0))
    }

    pub(super) fn run_edit_conditioning(
        &mut self,
        edit: PreparedEdit<R>,
        text_stage: PreparedStage<R>,
        job: u64,
        stages: &mut Vec<ImageStageReceipt>,
    ) -> Result<(ManagedHostTensor, ManagedHostTensor), ImageGenerationFailure<R>> {
        let process = |source| ImageGenerationFailure::Processing {
            source,
            active: None,
        };
        let PreparedEdit {
            prompt,
            inputs,
            vision,
            encoder,
            vision_stage,
            encoder_stage,
        } = edit;
        let started = std::time::Instant::now();
        let positions = self
            .assets
            .sources
            .text_encoder
            .tensor(Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)
            .map_err(|e| process(e.into()))?;
        let position_inputs = vision
            .managed_position_inputs(&self.host, &positions)
            .map_err(process)?;
        let vision_position_us = trace_elapsed_us(started);
        let (patches, rgba) = inputs.into_tensors();
        let mut vision_inputs = Vec::from(position_inputs);
        vision_inputs.insert(0, patches);
        let active = self.start_stage(vision_stage, job, stages)?;
        let (active, outputs) = active.execute(job, vision_inputs)?;
        self.finish_stage(active, stages)?;
        let started = std::time::Instant::now();
        let embedding = self
            .assets
            .sources
            .text_encoder
            .tensor(Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME)
            .map_err(|e| process(e.into()))?;
        let text_inputs = prompt
            .managed_text_inputs(&self.host, &self.assets.text, &embedding, outputs)
            .map_err(process)?;
        let text_bridge_us = trace_elapsed_us(started);
        let active = self.start_stage(text_stage, job, stages)?;
        let (active, outputs) = active.execute(job, text_inputs)?;
        self.finish_stage(active, stages)?;
        let started = std::time::Instant::now();
        let hidden = only_output(outputs, Qwen3VLTextEncoderFamily::OUTPUT_ID).map_err(process)?;
        let text = prompt
            .prompts()
            .crop_hidden(&self.host, &self.assets.text, &hidden)
            .map_err(process)?
            .into_tensor();
        drop(hidden);
        let hidden_crop_us = trace_elapsed_us(started);

        let active = self.start_stage(encoder_stage, job, stages)?;
        let (active, outputs) = active.execute(job, vec![rgba])?;
        self.finish_stage(active, stages)?;
        let started = std::time::Instant::now();
        let encoded =
            only_output(outputs, QwenImage21VaeEncoderFamily::OUTPUT_ID).map_err(process)?;
        let scale = encoder.spatial_scale();
        let shape = QwenImage21LatentShape::new(
            1,
            usize::try_from(encoder.height() / scale).map_err(|e| process(e.into()))?,
            usize::try_from(encoder.width() / scale).map_err(|e| process(e.into()))?,
        )
        .map_err(process)?;
        let posterior = QwenImage21ManagedLatents::from_output(
            &self.host,
            &encoded,
            shape,
            QwenImage21LatentLayout::Nchw,
            QwenImage21LatentDomain::Denormalized,
        )
        .map_err(process)?;
        let reference = posterior
            .normalize_from_vae(
                &self.host,
                &self.assets.statistics,
                id(QwenImage21TransformerFamily::REFERENCE_LATENTS_ID).map_err(process)?,
            )
            .map_err(process)?
            .into_input_tensor()
            .map_err(process)?;
        tracing::debug!(
            target: "ferrum_engine::image_generation::timing",
            job,
            vision_position_us,
            text_bridge_us,
            hidden_crop_us,
            reference_latent_bridge_us = trace_elapsed_us(started),
            "image edit conditioning bridges completed"
        );
        Ok((text, reference))
    }
}
