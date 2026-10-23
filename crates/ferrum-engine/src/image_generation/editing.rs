//! One independent reference image per request. Vision and VAE encoding reuse
//! B1 plans while text and denoising execute the real request batch. Every plan
//! compiles and passes capacity preflight before any upload or device execution.
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
    pub inputs: Vec<QwenImage21ConditionInputs>,
    vision: Qwen3VLVisionConfig,
    encoder: QwenImage21VaeEncoderConfig,
}
pub(super) struct PreparedEdit<R: DeviceRuntime> {
    pub prompt: QwenImage21EditPrompt,
    pub inputs: Vec<QwenImage21ConditionInputs>,
    pub vision: Qwen3VLVisionConfig,
    pub encoder: QwenImage21VaeEncoderConfig,
    pub vision_stage: PreparedStage<R>,
    pub encoder_stage: PreparedStage<R>,
}
impl EditInputs {
    pub(super) fn dimensions(&self, batch: usize) -> anyhow::Result<(u32, u32)> {
        ensure!(
            self.prompt.prompts().batch() == batch,
            "edit prompt batch differs from request count"
        );
        let geometry = condition_geometry(&self.inputs, batch)?;
        Ok((geometry.height, geometry.width))
    }

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
        let mut outputs = self.edit_batch(vec![request])?;
        Ok(outputs.remove(0))
    }

    /// One reference per request, in caller order. Processed reference geometry
    /// and output dimensions must be shared; reference pixels and prompts need
    /// not match. All plans and capacity checks precede the first device work.
    pub fn edit_batch(
        &mut self,
        requests: Vec<ImageEditRequest>,
    ) -> Result<Vec<ImageGenerationOutput>, ImageGenerationError<R>> {
        let preflight = |source| ImageGenerationError {
            failure: ImageGenerationFailure::Preflight(source),
            completed_stages: Vec::new(),
            completed_steps: Vec::new(),
        };
        if self.is_stopped() {
            return Err(ImageGenerationError {
                failure: ImageGenerationFailure::Unavailable,
                completed_stages: Vec::new(),
                completed_steps: Vec::new(),
            });
        }
        let (prepared, generation) = (|| -> anyhow::Result<_> {
            ensure!(self.supports_edits(), "image editing is not configured");
            self.config.validate()?;
            ensure!(
                (1..=self.config.maximum_batch_size).contains(&requests.len()),
                "image request batch exceeds configured maximum_batch_size"
            );
            let (generation, images): (Vec<_>, Vec<_>) = requests
                .into_iter()
                .map(|request| (request.generation, request.image))
                .unzip();
            super::prepare::batch_dimensions(
                &generation,
                &self.config,
                self.assets.spatial_scale()?,
            )?;
            let assets = self
                .assets
                .editing
                .as_ref()
                .context("editing assets are missing")?;
            let limits = ImageInputLimits::default();
            let mut decode_png_us = 0_u64;
            let mut reference_preprocess_us = 0_u64;
            let mut inputs = Vec::with_capacity(generation.len());
            for image in images {
                let started = std::time::Instant::now();
                let decoded = ManagedRgba8Image::decode_png(&self.host, &image, limits)?;
                decode_png_us = decode_png_us.saturating_add(trace_elapsed_us(started));
                let started = std::time::Instant::now();
                inputs.push(assets.processor.prepare(&self.host, &decoded, limits)?);
                reference_preprocess_us =
                    reference_preprocess_us.saturating_add(trace_elapsed_us(started));
                // Each prepared image owns its distinct admitted payload. Release
                // encoded/decoded input owners without retaining extra clones.
                drop(decoded);
                drop(image);
            }
            let started = std::time::Instant::now();
            let geometry = condition_geometry(&inputs, generation.len())?;
            let [temporal, height, width] = geometry.grid;
            let vision = Qwen3VLVisionConfig::new(
                assets.vision.clone(),
                vec![Qwen3VLVisionGrid::new(temporal, height, width)?],
            )?;
            let encoder = assets.encoder_config(geometry.height, geometry.width)?;
            let prompt_text: Vec<_> = generation.iter().map(|r| r.prompt.as_str()).collect();
            let prompt = self.assets.tokenizer.prepare_edit_batch(
                &prompt_text,
                self.config.maximum_user_tokens,
                height,
                width,
                assets.vision.merge_size(),
            )?;
            let edit_layout_and_prompt_us = trace_elapsed_us(started);
            let prepared = self.prepare_batch_with_edit(
                &generation,
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
                batch_size = generation.len(),
                decode_png_us,
                reference_preprocess_us,
                edit_layout_and_prompt_us,
                "image edit inputs prepared"
            );
            Ok((prepared, generation))
        })()
        .map_err(preflight)?;
        self.execute_prepared(prepared, &generation)
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
        let batch = inputs.len();
        let mut rgba_inputs = Vec::with_capacity(batch);
        let mut vision_outputs = Vec::with_capacity(batch);
        let mut active = self.start_stage(vision_stage, job, stages)?;
        for input in inputs {
            let (patches, rgba) = input.into_tensors();
            // Position owners are immutable and common to the validated grid;
            // the patch owner is fresh for this particular request.
            let mut vision_inputs = Vec::from(position_inputs.clone());
            vision_inputs.insert(0, patches);
            let (next, outputs) = self.execute_stage(active, job, vision_inputs)?;
            active = next;
            vision_outputs.push(outputs);
            rgba_inputs.push(rgba);
        }
        self.finish_stage(active, stages)?;
        drop(position_inputs);
        let started = std::time::Instant::now();
        let embedding = self
            .assets
            .sources
            .text_encoder
            .tensor(Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME)
            .map_err(|e| process(e.into()))?;
        let text_inputs = prompt
            .managed_text_batch_inputs(&self.host, &self.assets.text, &embedding, vision_outputs)
            .map_err(process)?;
        let text_bridge_us = trace_elapsed_us(started);
        let active = self.start_stage(text_stage, job, stages)?;
        let (active, outputs) = self.execute_stage(active, job, text_inputs)?;
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

        let started = std::time::Instant::now();
        let scale = encoder.spatial_scale();
        let shape = QwenImage21LatentShape::new(
            1,
            usize::try_from(encoder.height() / scale).map_err(|e| process(e.into()))?,
            usize::try_from(encoder.width() / scale).map_err(|e| process(e.into()))?,
        )
        .map_err(process)?;
        let mut references =
            ReferenceLatentBatch::new(&self.host, batch, shape).map_err(process)?;
        let mut reference_latent_bridge_us = trace_elapsed_us(started);
        let mut active = self.start_stage(encoder_stage, job, stages)?;
        for rgba in rgba_inputs {
            let (next, outputs) = self.execute_stage(active, job, vec![rgba])?;
            active = next;
            let started = std::time::Instant::now();
            let normalize = (|| -> anyhow::Result<()> {
                let encoded = only_output(outputs, QwenImage21VaeEncoderFamily::OUTPUT_ID)?;
                let posterior = QwenImage21ManagedLatents::from_output(
                    &self.host,
                    &encoded,
                    shape,
                    QwenImage21LatentLayout::Nchw,
                    QwenImage21LatentDomain::Denormalized,
                )?;
                // Keep the original staged BF16 subtraction/division unchanged.
                let reference = posterior
                    .normalize_from_vae(
                        &self.host,
                        &self.assets.statistics,
                        id(QwenImage21TransformerFamily::REFERENCE_LATENTS_ID)?,
                    )?
                    .into_input_tensor()?;
                references.push(reference)
            })();
            if let Err(source) = normalize {
                return Err(active.processing(source));
            }
            reference_latent_bridge_us =
                reference_latent_bridge_us.saturating_add(trace_elapsed_us(started));
        }
        let reference = match references.finish() {
            Ok(reference) => reference,
            Err(source) => return Err(active.processing(source)),
        };
        self.finish_stage(active, stages)?;
        tracing::debug!(
            target: "ferrum_engine::image_generation::timing",
            job,
            vision_position_us,
            text_bridge_us,
            hidden_crop_us,
            reference_latent_bridge_us,
            "image edit conditioning bridges completed"
        );
        Ok((text, reference))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ConditionGeometry {
    height: u32,
    width: u32,
    grid: [u64; 3],
}

fn condition_geometry(
    inputs: &[QwenImage21ConditionInputs],
    batch: usize,
) -> anyhow::Result<ConditionGeometry> {
    shared_condition_geometry(
        inputs.iter().map(|input| ConditionGeometry {
            height: input.height(),
            width: input.width(),
            grid: input.grid_thw(),
        }),
        batch,
    )
}

fn shared_condition_geometry(
    geometries: impl IntoIterator<Item = ConditionGeometry>,
    batch: usize,
) -> anyhow::Result<ConditionGeometry> {
    ensure!(
        (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&batch),
        "invalid reference batch size"
    );
    let mut geometries = geometries.into_iter();
    let first = geometries
        .next()
        .context("edit batch has no processed references")?;
    ensure!(
        first.height > 0 && first.width > 0 && first.grid.into_iter().all(|n| n > 0),
        "reference geometry must be positive"
    );
    let mut count = 1;
    for geometry in geometries {
        ensure!(
            geometry == first,
            "edit batch requires equal processed reference height, width and vision grid"
        );
        count += 1;
        ensure!(count <= batch, "reference count exceeds request batch");
    }
    ensure!(count == batch, "reference count differs from request batch");
    Ok(first)
}

/// Only bounded owners/metadata live outside the Host ledger. B1 keeps its
/// original normalized input owner; B>1 copies samples in order into one exact
/// admitted packed backing and drops each temporary normalized owner promptly.
struct ReferenceLatentBatch {
    host: HostMemoryAccount,
    shape: QwenImage21LatentShape,
    item_shape: QwenImage21LatentShape,
    appended: usize,
    backing: Option<ManagedHostBuffer>,
    single: Option<ManagedHostTensor>,
}

impl ReferenceLatentBatch {
    fn new(
        host: &HostMemoryAccount,
        batch: usize,
        item_shape: QwenImage21LatentShape,
    ) -> anyhow::Result<Self> {
        ensure!(
            (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&batch),
            "invalid packed reference batch size"
        );
        let [item_batch, _, height, width] = item_shape.nchw_dimensions();
        ensure!(item_batch == 1, "reference encoder must use B1 geometry");
        let shape = QwenImage21LatentShape::new(batch, height, width)?;
        let backing = if batch == 1 {
            None
        } else {
            Some(host.allocate_zeroed(HostMemoryClass::InputPreprocess, shape.byte_len())?)
        };
        Ok(Self {
            host: host.clone(),
            shape,
            item_shape,
            appended: 0,
            backing,
            single: None,
        })
    }

    fn push(&mut self, reference: ManagedHostTensor) -> anyhow::Result<()> {
        ensure!(
            self.appended < self.shape.nchw_dimensions()[0],
            "too many reference samples"
        );
        // ManagedHostTensor seals immutable InputPreprocess bytes with exactly
        // its contiguous tensor's storage length. Matching the complete packed
        // spec therefore proves the copy stride, also preserving B1's class.
        ensure!(
            self.host.owns(reference.bytes())
                && reference.value_id().as_str()
                    == QwenImage21TransformerFamily::REFERENCE_LATENTS_ID
                && reference.tensor()
                    == &self.item_shape.tensor(QwenImage21LatentLayout::Packed)?,
            "reference sample has a foreign owner, semantic ID or packed geometry"
        );
        if let Some(backing) = &mut self.backing {
            let stride = self.item_shape.byte_len();
            let start = self
                .appended
                .checked_mul(stride)
                .context("reference offset overflows")?;
            let end = start
                .checked_add(stride)
                .context("reference end overflows")?;
            backing
                .as_mut()
                .get_mut(start..end)
                .context("reference exceeds admitted batch")?
                .copy_from_slice(reference.bytes().bytes());
        } else {
            self.single = Some(reference);
        }
        self.appended += 1;
        Ok(())
    }

    fn finish(self) -> anyhow::Result<ManagedHostTensor> {
        ensure!(
            self.appended == self.shape.nchw_dimensions()[0],
            "reference batch is incomplete"
        );
        if let Some(single) = self.single {
            return Ok(single);
        }
        Ok(ManagedHostTensor::new(
            id(QwenImage21TransformerFamily::REFERENCE_LATENTS_ID)?,
            self.shape.tensor(QwenImage21LatentLayout::Packed)?,
            self.backing
                .context("missing packed reference backing")?
                .freeze(),
        )?)
    }
}

#[cfg(test)]
#[path = "tests/edit_batch.rs"]
mod tests;
