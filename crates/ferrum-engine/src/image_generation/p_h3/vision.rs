//! Real F32 visual execution under the existing image stage/actor owner. The
//! recipe owns metadata only; the RGBA input is borrowed after admission.
use super::{PH3ImageAssets, PH3ImagePipeline};
use crate::image_generation::{
    ImageGenerationFailure, ImageStage, ImageStageReceipt, PreparedImageStage,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::{
    p_h3_image::{PH3ImageEditPrompt, PH3ImageTextConditioningConfig},
    qwen3_vl_vision::{
        comfy::{
            Qwen3VLVisionComfyFamily, Qwen3VLVisionComfyImagePlan, Qwen3VLVisionComfyPixelPolicy,
        },
        Qwen3VLVisionFamily,
    },
    qwen_image21::image_input::ManagedRgba8Image,
};
use ferrum_runtime::*;
pub(super) mod reuse;
use reuse::{MainVisionBindingReceipt, MainVisionReuseBinding};

pub(super) struct PreparedVision {
    pub additional: Vec<PreparedVision>,
    pub prompt: PH3ImageEditPrompt,
    pub source_width: u32,
    pub source_height: u32,
    pub pixels: Qwen3VLVisionComfyImagePlan,
    pub family: PreparedModelFamily,
    pub options: ProgramPlanCompileOptions,
}
impl PreparedVision {
    pub fn new(
        assets: &PH3ImageAssets,
        text: &str,
        source_width: u32,
        source_height: u32,
        policy: Qwen3VLVisionComfyPixelPolicy,
    ) -> Result<Self> {
        let pixels = Qwen3VLVisionComfyImagePlan::new(
            assets.vision.clone(),
            source_width,
            source_height,
            policy,
        )?;
        let prompt = assets.tokenizer.prepare_single_image_instruction(
            text,
            pixels.config().grid(),
            assets.vision.merge_size(),
        )?;
        let family =
            TypedFamilyRegistration::new(Qwen3VLVisionComfyFamily::new(assets.limits.vision)?)
                .prepare_with_profile(
                    &serde_json::to_value(pixels.config())?,
                    &NumericalProfileId::new(assets.main_vision_profile)
                        .map_err(anyhow::Error::msg)?,
                )?;
        let options = pixels.config().compile_options()?;
        Ok(Self {
            additional: Vec::new(),
            prompt,
            source_width,
            source_height,
            pixels,
            family,
            options,
        })
    }
    pub fn images(&self) -> impl Iterator<Item = &PreparedVision> {
        std::iter::once(self).chain(self.additional.iter())
    }
    pub fn new_images(
        assets: &PH3ImageAssets,
        text: &str,
        dimensions: &[(u32, u32)],
        policy: Qwen3VLVisionComfyPixelPolicy,
    ) -> Result<Self> {
        ensure!(
            (1..=8).contains(&dimensions.len()),
            "H3 requires 1..=8 ordered Pictures"
        );
        let mut images = dimensions
            .iter()
            .map(|&(width, height)| Self::new(assets, text, width, height, policy))
            .collect::<Result<Vec<_>>>()?;
        let grids = images
            .iter()
            .map(|image| image.pixels.config().grid().clone())
            .collect::<Vec<_>>();
        let mut first = images.remove(0);
        first.prompt = assets.tokenizer.prepare_images_instruction(
            text,
            &grids,
            assets.vision.merge_size(),
        )?;
        first.additional = images;
        Ok(first)
    }
}
impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn execute_vision_image_inputs(
        &mut self,
        recipe: &PreparedVision,
        stages: &[&PreparedImageStage<R>],
        images: &[ManagedRgba8Image],
        text: &PH3ImageTextConditioningConfig,
        identity: &RequestIdentity,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
        reuse: Option<&MainVisionReuseBinding>,
    ) -> std::result::Result<
        (Vec<ManagedHostTensor>, Option<MainVisionBindingReceipt>),
        ImageGenerationFailure<R>,
    > {
        let count = recipe.images().count();
        if count != images.len() || count != stages.len() {
            return Err(ImageGenerationFailure::Preflight(anyhow::anyhow!(
                "ordered visual images and stages differ"
            )));
        }
        if count == 1 {
            return self.execute_vision_inputs(
                recipe, stages[0], &images[0], text, identity, job, receipts, reuse,
            );
        }
        let host = self.execution.execution_host().clone();
        let mut readbacks = Vec::with_capacity(count);
        for ((recipe, prepared), image) in recipe.images().zip(stages).zip(images) {
            let inputs = (|| -> Result<_> {
                ensure!(
                    prepared.declaration().stage == ImageStage::VisionEncoder
                        && prepared.family().fingerprint()? == recipe.family.fingerprint()?,
                    "visual stage differs from its original component"
                );
                image.validate_parent(&host)?;
                ensure!(
                    (image.width(), image.height()) == (recipe.source_width, recipe.source_height),
                    "visual Picture dimensions changed after preparation"
                );
                let source = &self.execution.sources.assets.sources.text_encoder;
                let table = ferrum_quantization::TensorArchive::tensor(
                    source.archive(),
                    Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME,
                )?;
                let positions = recipe
                    .pixels
                    .config()
                    .managed_position_inputs(&host, &table)?;
                let patches = recipe.pixels.managed_patches(&host, image)?;
                Ok(std::iter::once(patches).chain(positions).collect())
            })()
            .map_err(ImageGenerationFailure::Preflight)?;
            let mut active = self
                .execution
                .start_stage((*prepared).clone(), job, receipts)?;
            if let Err(error) = active.session.set_output_host_account(host.clone()) {
                return Err(active.processing(error.into()));
            }
            let Some(actor) = self.execution.serial_actor.as_mut() else {
                return Err(active.processing(anyhow::anyhow!("image serial actor is closed")));
            };
            let (active, outputs) = active.execute_on_actor(actor, job, inputs)?;
            let validate = (|| -> Result<()> {
                let ports = active
                    .session
                    .executable()
                    .execution_plan()
                    .payload()
                    .tensor_ports()
                    .context("visual stage lacks actual ports")?;
                ensure!(
                    outputs.len() == 4 && ports.outputs().len() == 4,
                    "each Picture requires all four visual readbacks"
                );
                for output in &outputs {
                    let port = ports
                        .outputs()
                        .iter()
                        .find(|p| p.value_id() == output.value_id())
                        .context("visual readback is not declared by this session")?;
                    ensure!(
                        port.tensor() == output.tensor()
                            && host.owns(output.bytes())
                            && output.bytes().class() == HostMemoryClass::Output
                            && output.tensor().element_type() == ElementType::F32
                            && output
                                .bytes()
                                .bytes()
                                .chunks_exact(4)
                                .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite()),
                        "visual readback has a foreign owner, invalid shape, precision or value"
                    );
                }
                Ok(())
            })();
            if let Err(error) = validate {
                return Err(active.processing(error));
            }
            self.execution.finish_stage(active, receipts)?;
            readbacks.push(outputs);
        }
        let ordered = readbacks.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let inputs = recipe
            .prompt
            .managed_image_inputs(text, &host, &ordered)
            .map_err(ImageGenerationFailure::Preflight)?;
        Ok((inputs, None))
    }

    /// Four actual Output readbacks are copied by the model's checked adapter
    /// into request-local Input owners. An enabled cohort may retain only the
    /// immutable readbacks after the original visual stage becomes quiescent.
    pub(super) fn execute_vision_inputs(
        &mut self,
        recipe: &PreparedVision,
        prepared: &PreparedImageStage<R>,
        image: &ManagedRgba8Image,
        text: &PH3ImageTextConditioningConfig,
        identity: &RequestIdentity,
        job: u64,
        receipts: &mut Vec<ImageStageReceipt>,
        reuse: Option<&MainVisionReuseBinding>,
    ) -> std::result::Result<
        (Vec<ManagedHostTensor>, Option<MainVisionBindingReceipt>),
        ImageGenerationFailure<R>,
    > {
        let host = self.execution.execution_host().clone();
        (|| -> Result<()> {
            ensure!(
                prepared.declaration().stage == ImageStage::VisionEncoder
                    && prepared.family().fingerprint()? == recipe.family.fingerprint()?,
                "visual stage differs from its prepared original component"
            );
            image.validate_parent(&host)?;
            ensure!(
                image.width() == recipe.source_width && image.height() == recipe.source_height,
                "visual image differs from the metadata-only source dimensions"
            );
            ensure!(
                recipe.prompt.grid() == recipe.pixels.config().grid(),
                "visual presentation and patch grid differ"
            );
            Ok(())
        })()
        .map_err(ImageGenerationFailure::Preflight)?;
        if let Some(binding) = reuse {
            if let Some((outputs, receipt)) = binding
                .lookup(&self.execution.service_preparation_identity, recipe)
                .map_err(ImageGenerationFailure::Preflight)?
            {
                let inputs = recipe
                    .prompt
                    .managed_inputs(text, &host, &outputs)
                    .map_err(ImageGenerationFailure::Preflight)?;
                super::conditioning_diagnostics::observe_validated_main_vision(
                    &mut self.main_vision_diagnostics,
                    identity,
                    job,
                    recipe.pixels.config().grid(),
                    image,
                    &outputs,
                )
                .map_err(ImageGenerationFailure::Preflight)?;
                return Ok((inputs, Some(receipt)));
            }
        }
        let inputs = (|| -> Result<_> {
            let source = &self.execution.sources.assets.sources.text_encoder;
            let table = ferrum_quantization::TensorArchive::tensor(
                source.archive(),
                Qwen3VLVisionFamily::POSITION_EXTERNAL_NAME,
            )?;
            let positions = recipe
                .pixels
                .config()
                .managed_position_inputs(&host, &table)?;
            let patches = recipe.pixels.managed_patches(&host, image)?;
            Ok(std::iter::once(patches).chain(positions).collect())
        })()
        .map_err(ImageGenerationFailure::Preflight)?;
        let mut active = self
            .execution
            .start_stage(prepared.clone(), job, receipts)?;
        let output_host = if let Some(binding) = reuse {
            match binding.cohort.output_account() {
                Ok(host) => host,
                Err(error) => return Err(active.processing(error)),
            }
        } else {
            host.clone()
        };
        if let Err(error) = active.session.set_output_host_account(output_host) {
            return Err(active.processing(error.into()));
        }
        let Some(actor) = self.execution.serial_actor.as_mut() else {
            return Err(active.processing(anyhow::anyhow!("image serial actor is closed")));
        };
        let (active, outputs) = active.execute_on_actor(actor, job, inputs)?;
        // Validate the session's submitted program/ports before accepting a
        // readback with a familiar value ID. The adapter then checks all four
        // exact IDs, F32 shapes, finite values and the actual Host provenance.
        let inputs = (|| -> Result<_> {
            ensure!(
                active
                    .session
                    .executable()
                    .execution_plan()
                    .payload()
                    .program_fingerprint()
                    == recipe.family.program().fingerprint()?,
                "visual session program differs from the submitted recipe"
            );
            let ports = active
                .session
                .executable()
                .execution_plan()
                .payload()
                .tensor_ports()
                .context("visual stage lacks actual tensor ports")?;
            ensure!(
                ports.outputs().len() == outputs.len() && outputs.len() == 4,
                "visual stage must read back exactly the four declared tensors"
            );
            for output in &outputs {
                let declared = ports
                    .outputs()
                    .iter()
                    .find(|p| p.value_id() == output.value_id())
                    .context("visual output is not declared by this actual session")?;
                ensure!(
                    declared.tensor() == output.tensor(),
                    "visual readback differs from its actual declared port"
                );
            }
            recipe.prompt.managed_inputs(text, &host, &outputs)
        })();
        let inputs = match inputs {
            Ok(inputs) => inputs,
            Err(source) => return Err(active.processing(source)),
        };
        // The adapter above already verified finite features and original Host
        // provenance. Borrow these exact readbacks and the admitted RGBA owner
        // once, before retirement; observer failure retains the original active.
        if let Err(source) = super::conditioning_diagnostics::observe_validated_main_vision(
            &mut self.main_vision_diagnostics,
            identity,
            job,
            recipe.pixels.config().grid(),
            image,
            &outputs,
        ) {
            return Err(active.processing(source));
        }
        let receipt = if let Some(binding) = reuse {
            self.execution.finish_stage(active, receipts)?;
            // The original producer has fenced and returned successfully to
            // the pipeline (possibly Resident). Only Host readbacks survive.
            Some(
                binding
                    .publish(recipe, outputs, job, identity)
                    .map_err(ImageGenerationFailure::Preflight)?,
            )
        } else {
            drop(outputs);
            self.execution.finish_stage(active, receipts)?;
            None
        };
        Ok((inputs, receipt))
    }
}
