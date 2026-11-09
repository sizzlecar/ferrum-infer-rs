//! Ordered image editing metadata and real per-Picture reference initialization.
//! Each PNG keeps its original ingress owner until admission and strict decode.
use super::{
    edit_noise::CompiledEditNoise, reference::CompiledReferenceEncoder, stages::PreparedGeneration,
    PH3ImagePipeline,
};
use crate::image_generation::{
    ImageEditRequest, ImageGenerationFailure, ImageGenerationRequest, ImageStage,
    ImageStagePlanRole, ImageStageReceipt,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::{
    p_h3_image::{
        PH3ImageDenoiserReference, PH3ImageEditPixelPlan, PH3ImageEditPixels,
        PH3ImageTransformerConfig,
    },
    qwen3_vl_vision::comfy::Qwen3VLVisionComfyPixelPolicy,
    qwen_image21::image_input::{png_dimensions_hint, ImageInputLimits, ManagedRgba8Image},
};
use ferrum_runtime::*;
mod budget;

pub(super) struct PreparedReferenceEdit<R: DeviceRuntime> {
    pub additional: Vec<PreparedReferenceEdit<R>>,
    pub pixels: PH3ImageEditPixelPlan,
    pub source_dimensions: (u32, u32),
    pub ingress_backing_bytes: usize,
    pub encoded: ManagedHostBytes,
    pub encoder: CompiledReferenceEncoder<R>,
    pub noise: CompiledEditNoise<R>,
    pub latent_spec: ResolvedTensorSpec,
    pub patch_noise_spec: ResolvedTensorSpec,
}
impl<R: DeviceRuntime> PreparedReferenceEdit<R> {
    pub fn images(&self) -> impl Iterator<Item = &Self> {
        std::iter::once(self).chain(&self.additional)
    }
    pub fn ingress_bytes(&self) -> Result<usize> {
        self.images().try_fold(0usize, |sum, image| {
            sum.checked_add(image.ingress_backing_bytes)
                .context("reference ingress backing overflows")
        })
    }
}

/// These immutable tensors belong to the request's persistent scope. They
/// follow its identity through packing/reordering and are never solver outputs.
pub(super) struct ReferenceState {
    pub latent: ManagedHostTensor,
    pub patch_noise: ManagedHostTensor,
    pub augmentation: f64,
}
impl ReferenceState {
    pub fn view(&self) -> PH3ImageDenoiserReference<'_> {
        PH3ImageDenoiserReference {
            latent: &self.latent,
            patch_noise: Some(&self.patch_noise),
            augmentation: self.augmentation,
        }
    }
    pub fn validate(
        &self,
        extent: ferrum_models::p_h3_image::PH3ImageReferenceImage,
        transformer: &PH3ImageTransformerConfig,
        host: &HostMemoryAccount,
        augmentation: f64,
    ) -> Result<()> {
        let (latent, patch) =
            reference_specs(transformer, extent.latent_height(), extent.latent_width())?;
        ensure!(
            self.augmentation == augmentation
                && augmentation.is_finite()
                && (0.0..=1.0).contains(&augmentation),
            "reference time and noise augmentation differ from the declared policy"
        );
        for (value, expected) in [(&self.latent, latent), (&self.patch_noise, patch)] {
            ensure!(
                value.tensor() == &expected
                    && host.owns(value.bytes())
                    && value.bytes().class() == HostMemoryClass::InputPreprocess
                    && value.bytes().len() as u64 == expected.minimum_storage_bytes()?,
                "reference geometry, precision, extent or persistent Host owner differs"
            );
        }
        Ok(())
    }
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn preprocess_reference_edit(
        &self,
        prepared: &PreparedReferenceEdit<R>,
        request: &ImageEditRequest,
        work: &HostMemoryAccount,
    ) -> Result<Vec<PH3ImageEditPixels>> {
        ensure!(
            prepared.images().count() == request.image_count(),
            "prepared image count changed"
        );
        prepared
            .images()
            .zip(request.images())
            .map(|(prepared, image)| {
                let decoded =
                    ManagedRgba8Image::decode_png(work, image, ImageInputLimits::default())?;
                ensure!(
                    (decoded.width(), decoded.height()) == prepared.source_dimensions,
                    "decoded reference differs from its pre-admission dimensions"
                );
                prepared.pixels.prepare(work, &decoded)
            })
            .collect()
    }

    pub(super) fn prepare_reference_edit(
        &self,
        request: &ImageEditRequest,
        identity: RequestIdentity,
    ) -> Result<(PreparedGeneration<R>, PreparedReferenceEdit<R>)> {
        ensure!(
            (1..=8).contains(&request.image_count()),
            "H3 editing supports 1..=8 ordered images"
        );
        let mut images = request
            .images()
            .map(|image| self.prepare_reference_image(image, &request.generation))
            .collect::<Result<Vec<_>>>()?;
        let assets = &self.execution.sources.assets;
        let dimensions = images
            .iter()
            .map(|image| image.pixels.vision_dimensions())
            .collect::<Vec<_>>();
        let references = images
            .iter()
            .map(|image| image.pixels.reference_image(assets.spatial_scale()))
            .collect::<Result<Vec<_>>>()?;
        let conditioning = assets.prepare_images_conditioning_with_family(
            &request.generation,
            identity,
            &self.execution.config,
            &dimensions,
            references,
            Qwen3VLVisionComfyPixelPolicy::default(),
            |provider, config, profile| {
                self.execution.prepare_family(
                    ImageStage::TextEncoder,
                    ImageStagePlanRole::WholeStage,
                    1,
                    provider,
                    config,
                    profile,
                )
            },
            |provider, config, profile| {
                self.execution.prepare_family(
                    ImageStage::TextConnector,
                    ImageStagePlanRole::WholeStage,
                    1,
                    provider,
                    config,
                    profile,
                )
            },
        )?;
        let generation =
            self.prepare_generation_with_conditioning(&request.generation, conditioning)?;
        let mut first = images.remove(0);
        first.additional = images;
        Ok((generation, first))
    }

    fn prepare_reference_image(
        &self,
        image: &ManagedHostBytes,
        request: &ImageGenerationRequest,
    ) -> Result<PreparedReferenceEdit<R>> {
        ensure!(
            self.execution.host.owns(image) && image.class() == HostMemoryClass::InputPreprocess,
            "H3 reference PNG requires the original Input Host owner"
        );
        let limits = ImageInputLimits::default();
        let dimensions = png_dimensions_hint(image.bytes(), limits)
            .context("H3 reference requires a bounded static RGB/RGBA8 PNG header")?;
        // IHDR is only a scheduling hint. CRC/chunks/pixels are checked by the
        // strict managed decoder after admission, before any device submission.
        let pixels = PH3ImageEditPixelPlan::new(
            dimensions.0,
            dimensions.1,
            request.width,
            request.height,
            self.options.edit_pixels,
            limits,
        )?;
        let assets = &self.execution.sources.assets;
        let extent = pixels.reference_image(assets.spatial_scale())?;
        let (latent_spec, patch_noise_spec) = reference_specs(
            assets.denoiser.transformer(),
            extent.latent_height(),
            extent.latent_width(),
        )?;
        let (rw, rh) = pixels.vae_dimensions();
        let encoder = self.compile_reference_encoder(rw, rh)?;
        ensure!(
            encoder
                .recipe
                .normalized_latent_spec()?
                .minimum_storage_bytes()?
                == latent_spec.minimum_storage_bytes()?,
            "reference VAE and DiT latent storage differ"
        );
        let noise = self.compile_edit_noise(latent_spec.minimum_storage_bytes()? / 4)?;
        Ok(PreparedReferenceEdit {
            additional: Vec::new(),
            pixels,
            source_dimensions: dimensions,
            ingress_backing_bytes: image.backing_bytes(),
            encoded: image.clone(),
            encoder,
            noise,
            latent_spec,
            patch_noise_spec,
        })
    }

    /// Encode the actual supplied image and retain its normalized latent and
    /// augmentation noise. Vision pixels outlive this call only until TE input
    /// preparation. Posterior draws cover the whole image exactly once.
    pub(super) fn initialize_reference_edit(
        &mut self,
        prepared: &PreparedReferenceEdit<R>,
        request: &ImageEditRequest,
        pixels: Vec<PH3ImageEditPixels>,
        persistent: &HostMemoryAccount,
        job: u64,
        identity: &RequestIdentity,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<(Vec<ReferenceState>, Vec<ManagedRgba8Image>), ImageGenerationFailure<R>>
    {
        let mut references = Vec::with_capacity(pixels.len());
        let mut visions = Vec::with_capacity(pixels.len());
        for (image, pixels) in prepared.images().zip(pixels) {
            let (reference, vision) = self.initialize_reference_image(
                image, request, pixels, persistent, job, identity, receipts,
            )?;
            references.push(reference);
            visions.push(vision);
        }
        Ok((references, visions))
    }

    fn initialize_reference_image(
        &mut self,
        prepared: &PreparedReferenceEdit<R>,
        request: &ImageEditRequest,
        pixels: PH3ImageEditPixels,
        persistent: &HostMemoryAccount,
        job: u64,
        identity: &RequestIdentity,
        receipts: &mut Vec<ImageStageReceipt>,
    ) -> std::result::Result<(ReferenceState, ManagedRgba8Image), ImageGenerationFailure<R>> {
        let work = self.execution.execution_host().clone();
        let posterior_noise = self.execute_edit_noise(
            &prepared.noise,
            self.options.edit_noise.posterior_seed(),
            &work,
            prepared
                .encoder
                .recipe
                .posterior_noise_spec()
                .map_err(ImageGenerationFailure::Preflight)?,
            ProgramValueId::new("value.h3.reference.posterior_noise")
                .map_err(|e| ImageGenerationFailure::Preflight(e.into()))?,
            job,
            receipts,
        )?;
        let normalized = self.execute_reference_encoder(
            &prepared.encoder,
            &pixels.vae,
            &posterior_noise,
            job,
            identity,
            receipts,
        )?;
        drop(posterior_noise);
        drop(pixels.vae);
        let latent = copy_reference_latent(persistent, &normalized, &prepared.latent_spec)
            .map_err(ImageGenerationFailure::Preflight)?;
        drop(normalized);
        let patch_noise = self.execute_edit_noise(
            &prepared.noise,
            request.generation.seed,
            persistent,
            prepared.patch_noise_spec.clone(),
            ProgramValueId::new("value.h3.reference.patch_noise")
                .map_err(|e| ImageGenerationFailure::Preflight(e.into()))?,
            job,
            receipts,
        )?;
        Ok((
            ReferenceState {
                latent,
                patch_noise,
                augmentation: self.options.edit_noise.augmentation(),
            },
            pixels.vision,
        ))
    }
}

fn reference_specs(
    transformer: &PH3ImageTransformerConfig,
    height: usize,
    width: usize,
) -> Result<(ResolvedTensorSpec, ResolvedTensorSpec)> {
    let [pt, ph, pw] = transformer.patch_size();
    let h = u64::try_from(height)?;
    let w = u64::try_from(width)?;
    ensure!(
        pt == 1 && h > 0 && w > 0 && h % ph == 0 && w % pw == 0,
        "single-image reference requires an exact T1 DiT patch layout"
    );
    let rows = (h / ph)
        .checked_mul(w / pw)
        .context("reference patch rows overflow")?;
    let columns = transformer
        .in_channels()
        .checked_mul(ph)
        .and_then(|v| v.checked_mul(pw))
        .context("reference patch width overflows")?;
    let spec =
        |dims| ResolvedTensorSpec::new(dims, ElementType::F32, ResolvedTensorLayout::Contiguous);
    Ok((
        spec(vec![1, transformer.in_channels(), 1, h, w])?,
        spec(vec![rows, columns])?,
    ))
}
fn copy_reference_latent(
    host: &HostMemoryAccount,
    source: &ManagedHostTensor,
    target: &ResolvedTensorSpec,
) -> Result<ManagedHostTensor> {
    ensure!(
        matches!(source.tensor().dimensions(), [1,c,h,w]
        if target.dimensions() == [1,*c,1,*h,*w])
            && source.tensor().element_type() == ElementType::F32
            && source.tensor().layout() == &ResolvedTensorLayout::Contiguous
            && source.bytes().class() == HostMemoryClass::InputPreprocess
            && host.owns(source.bytes())
            && source.bytes().len() as u64 == target.minimum_storage_bytes()?,
        "reference posterior requires the exact original F32 normalized T1 latent"
    );
    let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, source.bytes().len())?;
    bytes.as_mut().copy_from_slice(source.bytes().bytes());
    Ok(ManagedHostTensor::new(
        ProgramValueId::new("value.h3.reference.latent")?,
        target.clone(),
        bytes.freeze(),
    )?)
}

#[cfg(test)]
mod tests;
