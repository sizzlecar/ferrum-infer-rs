//! Ordered reference images per request. Vision and VAE encoding reuse B1
//! plans per reference geometry while text and denoising execute the request batch. Every plan
//! compiles and passes capacity preflight before any upload or device execution.
use super::*;
use ferrum_models::qwen3_vl_vision::{Qwen3VLVisionConfig, Qwen3VLVisionFamily, Qwen3VLVisionGrid};
use ferrum_models::qwen_image21::image_input::{
    ImageInputLimits, ManagedRgba8Image, QwenImage21ConditionInputs,
};
use ferrum_models::qwen_image21::t2i_prompt::{QwenImage21EditImageGrid, QwenImage21EditPrompt};
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
    pub inputs: Vec<Vec<QwenImage21ConditionInputs>>,
    references: Vec<ReferenceConfig>,
}
struct ReferenceConfig {
    vision: Qwen3VLVisionConfig,
    encoder: QwenImage21VaeEncoderConfig,
}
pub(super) struct PreparedEdit<R: DeviceRuntime> {
    pub prompt: QwenImage21EditPrompt,
    pub inputs: Vec<Vec<QwenImage21ConditionInputs>>,
    pub references: Vec<PreparedReference<R>>,
}
pub(super) struct PreparedReference<R: DeviceRuntime> {
    pub vision: Qwen3VLVisionConfig,
    pub encoder: QwenImage21VaeEncoderConfig,
    pub vision_stage: PreparedStage<R>,
    pub encoder_stage: PreparedStage<R>,
}
impl<R: DeviceRuntime> PreparedEdit<R> {
    pub(super) fn stages(&self) -> impl Iterator<Item = &PreparedStage<R>> {
        self.references
            .iter()
            .flat_map(|reference| [&reference.vision_stage, &reference.encoder_stage])
    }
}
impl EditInputs {
    pub(super) fn geometries(&self, batch: usize) -> anyhow::Result<Vec<(u32, u32)>> {
        ensure!(
            self.prompt.prompts().batch() == batch,
            "edit prompt batch differs from request count"
        );
        Ok(condition_geometries(&self.inputs, batch)?
            .into_iter()
            .map(|geometry| (geometry.height, geometry.width))
            .collect())
    }

    pub fn compile<R: DeviceRuntime>(
        self,
        pipeline: &QwenImage21Pipeline<R>,
    ) -> anyhow::Result<PreparedEdit<R>> {
        let profiles = pipeline
            .editing_profiles
            .as_ref()
            .context("editing numerical profiles are not configured")?;
        let mut references = Vec::with_capacity(self.references.len());
        for reference in self.references {
            let vision_family = pipeline.prepare_family(
                ImageStage::VisionEncoder,
                ImageStagePlanRole::WholeStage,
                1,
                Qwen3VLVisionFamily::new()?,
                &reference.vision,
                &profiles.vision,
            )?;
            let encoder_family = pipeline.prepare_family(
                ImageStage::VaeEncoder,
                ImageStagePlanRole::WholeStage,
                1,
                QwenImage21VaeEncoderFamily::new()?,
                &reference.encoder,
                &profiles.encoder,
            )?;
            let vision_stage = PreparedStage::compile(
                ImageStage::VisionEncoder,
                1,
                vision_family,
                &reference.vision.compile_options()?,
                pipeline,
            )?;
            let encoder_stage = PreparedStage::compile(
                ImageStage::VaeEncoder,
                1,
                encoder_family,
                &reference.encoder.compile_options()?,
                pipeline,
            )?;
            references.push(PreparedReference {
                vision: reference.vision,
                encoder: reference.encoder,
                vision_stage,
                encoder_stage,
            });
        }
        Ok(PreparedEdit {
            prompt: self.prompt,
            inputs: self.inputs,
            references,
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
            self.execution.sources.assets.editing.is_some(),
            "editing profiles require validated vision/encoder assets"
        );
        self.editing_profiles = Some(profiles);
        Ok(self)
    }
    pub fn supports_edits(&self) -> bool {
        self.execution.sources.assets.editing.is_some() && self.editing_profiles.is_some()
    }
    pub fn maximum_reference_images(&self) -> usize {
        if self.supports_edits() {
            10
        } else {
            0
        }
    }

    /// Decode each admitted reference once. The queue can retain this owner
    /// across candidate reductions without retaining a second decoded payload.
    pub(super) fn preprocess_edit(
        &self,
        request: &ImageEditRequest,
    ) -> anyhow::Result<Vec<QwenImage21ConditionInputs>> {
        ensure!(self.supports_edits(), "editing assets/profiles are missing");
        ensure!(
            request.image_count() <= self.maximum_reference_images(),
            "reference image count exceeds the model capability"
        );
        request.generation.latent_dimensions(
            &self.execution.config,
            self.execution.sources.assets.spatial_scale()?,
        )?;
        let assets = self
            .execution
            .sources
            .assets
            .editing
            .as_ref()
            .context("editing assets are missing")?;
        let limits = ImageInputLimits::default();
        let mut inputs = Vec::with_capacity(request.image_count());
        for image in request.images() {
            ensure!(
                self.execution.host.owns(image),
                "edit reference belongs to another Host account"
            );
            let started = std::time::Instant::now();
            let decoded = ManagedRgba8Image::decode_png(&self.execution.host, image, limits)?;
            let decode_png_us = trace_elapsed_us(started);
            let started = std::time::Instant::now();
            let input = assets
                .processor
                .prepare(&self.execution.host, &decoded, limits)?;
            tracing::debug!(
                target: "ferrum_engine::image_generation::timing",
                job = self.execution.next_job,
                decode_png_us,
                reference_preprocess_us = trace_elapsed_us(started),
                "image edit reference prepared"
            );
            inputs.push(input);
        }
        Ok(inputs)
    }

    /// Compile one compatible prefix of already preprocessed request owners.
    /// Failed candidates release their clones; the original queue owners remain
    /// admitted and can be used by the next smaller candidate without decoding.
    pub(super) fn prepare_preprocessed_edits(
        &self,
        requests: &[ImageGenerationRequest],
        inputs: &[Vec<QwenImage21ConditionInputs>],
    ) -> anyhow::Result<PreparedRequest<R>> {
        self.prepare_preprocessed_edits_with_host_admission(requests, inputs, true)
    }

    pub(super) fn prepare_preprocessed_edits_with_host_admission(
        &self,
        requests: &[ImageGenerationRequest],
        inputs: &[Vec<QwenImage21ConditionInputs>],
        reserve_host: bool,
    ) -> anyhow::Result<PreparedRequest<R>> {
        ensure!(self.supports_edits(), "editing assets/profiles are missing");
        super::prepare::batch_dimensions(
            requests,
            &self.execution.config,
            self.execution.sources.assets.spatial_scale()?,
        )?;
        for input in inputs.iter().flatten() {
            ensure!(
                self.execution.host.owns(input.vision_patches().bytes())
                    && self.execution.host.owns(input.vae_rgba().bytes()),
                "preprocessed edit input belongs to another Host account"
            );
        }
        let started = std::time::Instant::now();
        let assets = self
            .execution
            .sources
            .assets
            .editing
            .as_ref()
            .context("editing assets are missing")?;
        let geometries = condition_geometries(inputs, requests.len())?;
        let mut references = Vec::with_capacity(geometries.len());
        let mut grids = Vec::with_capacity(geometries.len());
        for geometry in geometries {
            let [temporal, height, width] = geometry.grid;
            grids.push(QwenImage21EditImageGrid::new(
                height,
                width,
                assets.vision.merge_size(),
            )?);
            references.push(ReferenceConfig {
                vision: Qwen3VLVisionConfig::new(
                    assets.vision.clone(),
                    vec![Qwen3VLVisionGrid::new(temporal, height, width)?],
                )?,
                encoder: assets.encoder_config(geometry.height, geometry.width)?,
            });
        }
        let prompt_text: Vec<_> = requests
            .iter()
            .map(|request| request.prompt.as_str())
            .collect();
        let prompt = self
            .execution
            .sources
            .assets
            .tokenizer
            .prepare_edit_batch_multi(
                &prompt_text,
                self.execution.config.maximum_user_tokens,
                &grids,
            )?;
        tracing::debug!(
            target: "ferrum_engine::image_generation::timing",
            job = self.execution.next_job,
            batch_size = requests.len(),
            edit_layout_and_prompt_us = trace_elapsed_us(started),
            "image edit batch metadata prepared"
        );
        self.prepare_batch_with_host_admission(
            requests,
            Some(EditInputs {
                prompt,
                inputs: inputs.to_vec(),
                references,
            }),
            reserve_host,
        )
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
            self.execution.config.validate()?;
            ensure!(
                (1..=self.execution.config.maximum_batch_size).contains(&requests.len()),
                "image request batch exceeds configured maximum_batch_size"
            );
            let generation: Vec<_> = requests
                .iter()
                .map(|request| request.generation.clone())
                .collect();
            super::prepare::batch_dimensions(
                &generation,
                &self.execution.config,
                self.execution.sources.assets.spatial_scale()?,
            )?;
            let mut inputs = Vec::with_capacity(generation.len());
            for request in requests {
                inputs.push(self.preprocess_edit(&request)?);
            }
            let prepared = self.prepare_preprocessed_edits(&generation, &inputs)?;
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
            references,
        } = edit;
        let batch = inputs.len();
        let mut columns: Vec<Vec<QwenImage21ConditionInputs>> = (0..references.len())
            .map(|_| Vec::with_capacity(batch))
            .collect();
        for row in inputs {
            for (column, input) in columns.iter_mut().zip(row) {
                column.push(input);
            }
        }
        let mut vision_outputs: Vec<Vec<Vec<TensorExecutionOutput>>> = (0..batch)
            .map(|_| Vec::with_capacity(references.len()))
            .collect();
        let mut encoder_jobs = Vec::with_capacity(references.len());
        let mut vision_position_us = 0u64;
        for (reference, inputs) in references.into_iter().zip(columns) {
            let PreparedReference {
                vision,
                encoder,
                vision_stage,
                encoder_stage,
            } = reference;
            let started = std::time::Instant::now();
            let positions = self
                .execution
                .sources
                .assets
                .sources
                .text_encoder
                .tensor(Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME)
                .map_err(|e| process(e.into()))?;
            let position_inputs = vision
                .managed_position_inputs(self.execution_host(), &positions)
                .map_err(process)?;
            vision_position_us = vision_position_us.saturating_add(trace_elapsed_us(started));
            let mut rgba_inputs = Vec::with_capacity(batch);
            let mut active = self.start_stage(vision_stage, job, stages)?;
            for (row, input) in inputs.into_iter().enumerate() {
                let (patches, rgba) = input.into_tensors();
                // Position owners are immutable and common to the validated grid;
                // the patch owner is fresh for this particular request.
                let mut vision_inputs = Vec::from(position_inputs.clone());
                vision_inputs.insert(0, patches);
                let (next, outputs) = self.execute_stage(active, job, vision_inputs)?;
                active = next;
                vision_outputs[row].push(outputs);
                rgba_inputs.push(rgba);
            }
            self.finish_stage(active, stages)?;
            drop(position_inputs);
            encoder_jobs.push((encoder, encoder_stage, rgba_inputs));
        }
        let started = std::time::Instant::now();
        let embedding = self
            .execution
            .sources
            .assets
            .sources
            .text_encoder
            .tensor(Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME)
            .map_err(|e| process(e.into()))?;
        let text_inputs = prompt
            .managed_text_batch_inputs_multi(
                self.execution_host(),
                &self.execution.sources.assets.text,
                &embedding,
                vision_outputs,
            )
            .map_err(process)?;
        let text_bridge_us = trace_elapsed_us(started);
        let active = self.start_stage(text_stage, job, stages)?;
        let (active, outputs) = self.execute_stage(active, job, text_inputs)?;
        self.finish_stage(active, stages)?;
        let started = std::time::Instant::now();
        let hidden = only_output(outputs, Qwen3VLTextEncoderFamily::OUTPUT_ID).map_err(process)?;
        let text = prompt
            .prompts()
            .crop_hidden(
                self.execution_host(),
                &self.execution.sources.assets.text,
                &hidden,
            )
            .map_err(process)?
            .into_tensor();
        drop(hidden);
        let hidden_crop_us = trace_elapsed_us(started);

        let started = std::time::Instant::now();
        let shapes = encoder_jobs
            .iter()
            .map(|(encoder, _, _)| {
                let scale = encoder.spatial_scale();
                QwenImage21LatentShape::new(
                    1,
                    (encoder.height() / scale) as usize,
                    (encoder.width() / scale) as usize,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(process)?;
        let mut references = if shapes.len() == 1 {
            ReferenceLatentBatch::new(self.execution_host(), batch, shapes[0])
        } else {
            ReferenceLatentBatch::multiple(self.execution_host(), batch, &shapes)
        }
        .map_err(process)?;
        let mut reference_latent_bridge_us = trace_elapsed_us(started);
        for ((_, encoder_stage, rgba_inputs), shape) in encoder_jobs.into_iter().zip(shapes) {
            let mut active = self.start_stage(encoder_stage, job, stages)?;
            for rgba in rgba_inputs {
                let (next, outputs) = self.execute_stage(active, job, vec![rgba])?;
                active = next;
                let started = std::time::Instant::now();
                let normalize = (|| -> anyhow::Result<()> {
                    let encoded = only_output(outputs, QwenImage21VaeEncoderFamily::OUTPUT_ID)?;
                    let posterior = QwenImage21ManagedLatents::from_output(
                        self.execution_host(),
                        &encoded,
                        shape,
                        QwenImage21LatentLayout::Nchw,
                        QwenImage21LatentDomain::Denormalized,
                    )?;
                    // Keep the original staged BF16 subtraction/division unchanged.
                    let reference = posterior
                        .normalize_from_vae(
                            self.execution_host(),
                            &self.execution.sources.assets.statistics,
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
            self.finish_stage(active, stages)?;
        }
        let reference = references.finish().map_err(process)?;
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

fn condition_geometries(
    inputs: &[Vec<QwenImage21ConditionInputs>],
    batch: usize,
) -> anyhow::Result<Vec<ConditionGeometry>> {
    ensure!(
        inputs.len() == batch && batch > 0,
        "reference request count differs from batch"
    );
    let count = inputs[0].len();
    ensure!(
        (1..=10).contains(&count),
        "unsupported reference image count"
    );
    ensure!(
        inputs.iter().all(|row| row.len() == count),
        "reference counts differ across request batch"
    );
    (0..count)
        .map(|index| {
            shared_condition_geometry(
                inputs.iter().map(|row| ConditionGeometry {
                    height: row[index].height(),
                    width: row[index].width(),
                    grid: row[index].grid_thw(),
                }),
                batch,
            )
        })
        .collect()
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
    batch: usize,
    item_shapes: Vec<QwenImage21LatentShape>,
    offsets: Vec<usize>,
    sample_bytes: usize,
    tensor: ResolvedTensorSpec,
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
        Self::multiple(host, batch, &[item_shape])
    }

    fn multiple(
        host: &HostMemoryAccount,
        batch: usize,
        item_shapes: &[QwenImage21LatentShape],
    ) -> anyhow::Result<Self> {
        ensure!(
            (1..=STRUCTURED_ATTENTION_BF16_MAX_BATCH).contains(&batch),
            "invalid packed reference batch size"
        );
        ensure!(
            (1..=10).contains(&item_shapes.len()),
            "unsupported reference image count"
        );
        let mut offsets = Vec::with_capacity(item_shapes.len());
        let mut sample_bytes = 0usize;
        for shape in item_shapes {
            ensure!(
                shape.nchw_dimensions()[0] == 1,
                "reference encoder must use B1 geometry"
            );
            offsets.push(sample_bytes);
            sample_bytes = sample_bytes
                .checked_add(shape.byte_len())
                .context("reference sample bytes overflow")?;
        }
        let total_bytes = sample_bytes
            .checked_mul(batch)
            .context("reference batch bytes overflow")?;
        let channels = ferrum_models::qwen_image21::pipeline_adapters::QWEN_IMAGE21_LATENT_CHANNELS;
        let tensor = ResolvedTensorSpec::new(
            vec![
                u64::try_from(total_bytes / (channels * 2))?,
                channels as u64,
            ],
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?;
        let backing = if batch == 1 && item_shapes.len() == 1 {
            None
        } else {
            Some(host.allocate_zeroed(HostMemoryClass::InputPreprocess, total_bytes)?)
        };
        Ok(Self {
            host: host.clone(),
            batch,
            item_shapes: item_shapes.to_vec(),
            offsets,
            sample_bytes,
            tensor,
            appended: 0,
            backing,
            single: None,
        })
    }

    fn push(&mut self, reference: ManagedHostTensor) -> anyhow::Result<()> {
        ensure!(
            self.appended < self.batch * self.item_shapes.len(),
            "too many reference samples"
        );
        let reference_index = self.appended / self.batch;
        let sample_index = self.appended % self.batch;
        let item_shape = self.item_shapes[reference_index];
        // ManagedHostTensor seals immutable InputPreprocess bytes with exactly
        // its contiguous tensor's storage length. Matching the complete packed
        // spec therefore proves the copy stride, also preserving B1's class.
        ensure!(
            self.host.owns(reference.bytes())
                && reference.value_id().as_str()
                    == QwenImage21TransformerFamily::REFERENCE_LATENTS_ID
                && reference.tensor() == &item_shape.tensor(QwenImage21LatentLayout::Packed)?,
            "reference sample has a foreign owner, semantic ID or packed geometry"
        );
        if let Some(backing) = &mut self.backing {
            let stride = item_shape.byte_len();
            let start = sample_index
                .checked_mul(self.sample_bytes)
                .and_then(|start| start.checked_add(self.offsets[reference_index]))
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
            self.appended == self.batch * self.item_shapes.len(),
            "reference batch is incomplete"
        );
        if let Some(single) = self.single {
            return Ok(single);
        }
        Ok(ManagedHostTensor::new(
            id(QwenImage21TransformerFamily::REFERENCE_LATENTS_ID)?,
            self.tensor,
            self.backing
                .context("missing packed reference backing")?
                .freeze(),
        )?)
    }
}

#[cfg(test)]
#[path = "tests/edit_batch.rs"]
mod tests;
