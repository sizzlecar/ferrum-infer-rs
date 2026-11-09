//! Real edit lifetimes: ingress is already admitted, preprocessing precedes
//! reference encoding, and only vision pixels survive into text conditioning.
use super::*;
use crate::image_generation::p_h3::host_budget::GenerationHostRequirements;
use ferrum_models::normal_noise::NormalNoiseConfig;

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(in crate::image_generation::p_h3) fn continuous_host_requirements(
        &self,
        generation: &PreparedGeneration<R>,
        edit: Option<&PreparedReferenceEdit<R>>,
    ) -> Result<GenerationHostRequirements> {
        let mut requirements = self.generation_host_requirements(generation)?;
        let Some(edit) = edit else {
            return Ok(requirements);
        };
        let host = self.execution.host_account();
        let tensor_bytes = |spec: &ResolvedTensorSpec| -> Result<usize> {
            Ok(host.backing_bytes(usize::try_from(spec.minimum_storage_bytes()?)?)?)
        };
        let rgba = |dims: (u32, u32)| -> Result<usize> {
            Ok(host.backing_bytes(usize::try_from(u64::from(dims.0) * u64::from(dims.1) * 4)?)?)
        };
        let sum = |values: &[usize]| -> Result<usize> {
            values.iter().try_fold(0usize, |total, &n| {
                total
                    .checked_add(n)
                    .context("H3 edit Host backing overflows")
            })
        };
        let retained_pixels = edit
            .images()
            .map(|image| {
                sum(&[
                    rgba(image.pixels.vision_dimensions())?,
                    tensor_bytes(&image.encoder.recipe.pixel_input_spec()?)?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        let all_pixels = sum(&retained_pixels)?;
        let all_vision = sum(&edit
            .images()
            .map(|image| rgba(image.pixels.vision_dimensions()))
            .collect::<Result<Vec<_>>>()?)?;
        let mut stages = Vec::new();
        for (edit, retained_pixels) in edit.images().zip(retained_pixels) {
            let other_pixels = all_pixels - retained_pixels;
            let reference_bytes = tensor_bytes(&edit.latent_spec)?;
            let patch_bytes = tensor_bytes(&edit.patch_noise_spec)?;
            requirements.persistent_input_bytes = sum(&[
                requirements.persistent_input_bytes,
                reference_bytes,
                patch_bytes,
            ])?;
            let decoded = rgba(edit.source_dimensions)?;
            let preprocessing = sum(&[decoded, edit.pixels.new_input_peak_bytes(host)?])?;
            // Strict PNG decode allocates exactly its RGBA extent. If vision keeps
            // that same backing, this is that owner, not an additional grant.
            let vision = rgba(edit.pixels.vision_dimensions())?;
            let vae_pixels = tensor_bytes(&edit.encoder.recipe.pixel_input_spec()?)?;
            let posterior_noise = tensor_bytes(&edit.encoder.recipe.posterior_noise_spec()?)?;
            let noise_state = tensor_bytes(&NormalNoiseConfig::state_tensor()?)?;
            let noise_readback = tensor_bytes(&edit.noise.config.output_tensor()?)?;
            let vae = edit.encoder.recipe.host_requirements(host)?;
            let noise_input = sum(&[vision, vae_pixels, posterior_noise, noise_state])?;
            let encoder_input = sum(&[vision, vae_pixels, posterior_noise, vae.input_peak_bytes])?;
            let conditioning_input = sum(&[
                all_vision,
                requirements
                    .conditioning
                    .input_peak_bytes
                    .max(requirements.visual.map_or(0, |v| v.input_peak_bytes)),
            ])?;
            requirements.initialization.input = requirements
                .initialization
                .input
                .max(preprocessing)
                .max(noise_input)
                .max(encoder_input)
                .max(conditioning_input)
                .max(sum(&[
                    other_pixels,
                    preprocessing.max(noise_input).max(encoder_input),
                ])?);
            requirements.initialization.output = requirements
                .initialization
                .output
                .max(noise_readback)
                .max(vae.output_peak_bytes);
            stages.extend([&edit.encoder.stage, &edit.noise.stage]);
        }
        if let Some(visual) = &generation.conditioning.visual {
            stages.push(visual);
        }
        stages.extend(&generation.conditioning.additional_visuals);
        stages.extend([
            &generation.conditioning.encoder,
            &generation.conditioning.connector,
        ]);
        stages.extend(generation.noise.stage());
        stages.push(&generation.denoising.install.stage);
        let (weights, backing) = self.source_backing_budget(&stages)?;
        requirements.initialization.weights = weights;
        requirements.initialization_backing = backing;
        requirements.initialization.staging = self.static_staging_budget(&stages)?;
        Ok(requirements)
    }
}
