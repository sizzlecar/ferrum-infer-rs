//! Request geometry is independent of a request's live solver cursor. Packing
//! membership changes rebuild this recipe; advancing time changes input data.
use super::PH3ImageAssets;
use crate::image_generation::{ImageGenerationRequest, ImagePipelineConfig};
use anyhow::{ensure, Context, Result};
use ferrum_models::p_h3_image::{
    PH3ImageDenoiserAdapter, PH3ImageDenoiserFamilyConfig, PH3ImageDenoiserFamilyProvider,
    PH3ImageDenoiserPhase, PH3ImageModality, PH3ImageReferenceImage, PH3ImageRequestLayout,
    PH3ImageSamplingStep, PH3ImageTransformerConfig, PreparedPH3ImageDenoiser,
};
use ferrum_runtime::{
    ElementType, NumericalProfileId, PreparedModelFamily, RequestIdentity, ResolvedTensorLayout,
    ResolvedTensorSpec,
};
use std::collections::BTreeSet;

pub(super) struct RequestGeometry {
    pub identity: RequestIdentity,
    pub target: PH3ImageReferenceImage,
    pub reference: Option<PH3ImageReferenceImage>,
    pub tags: Vec<PH3ImageModality>,
    pub image_sample: ResolvedTensorSpec,
    pub audio_sample: ResolvedTensorSpec,
}
impl RequestGeometry {
    /// The caller already selected and tokenized the conditioning path. Image
    /// tags must come from real vision conditioning, never guessed from text.
    pub fn new(
        request: &ImageGenerationRequest,
        identity: RequestIdentity,
        config: &ImagePipelineConfig,
        spatial_scale: u64,
        transformer: &PH3ImageTransformerConfig,
        tags: Vec<PH3ImageModality>,
        reference: Option<PH3ImageReferenceImage>,
    ) -> Result<Self> {
        validate_request_metadata(request, config, spatial_scale)?;
        ensure!(
            !tags.is_empty()
                && tags
                    .iter()
                    .filter(|tag| **tag == PH3ImageModality::Text)
                    .count()
                    <= config.maximum_user_tokens,
            "H3 text conditioning exceeds its nonempty declared capacity"
        );
        let target = PH3ImageReferenceImage::new(
            usize::try_from(u64::from(request.height) / spatial_scale)?,
            usize::try_from(u64::from(request.width) / spatial_scale)?,
        )?;
        PH3ImageRequestLayout::new(transformer, &tags, target, reference.as_slice())?;
        let tensor = |dimensions| -> Result<ResolvedTensorSpec> {
            let tensor = ResolvedTensorSpec::new(
                dimensions,
                ElementType::F32,
                ResolvedTensorLayout::Contiguous,
            )?;
            ensure!(
                tensor.minimum_storage_bytes()? <= isize::MAX as u64,
                "H3 latent storage exceeds the Host address range"
            );
            Ok(tensor)
        };
        let image_sample = tensor(vec![
            1,
            transformer.in_channels(),
            1,
            target.latent_height() as u64,
            target.latent_width() as u64,
        ])?;
        let audio_sample = tensor(vec![1, transformer.audio_in_channels(), 2, 2])?;
        Ok(Self {
            identity,
            target,
            reference,
            tags,
            image_sample,
            audio_sample,
        })
    }
    fn layout<'a>(
        &'a self,
        transformer: &PH3ImageTransformerConfig,
    ) -> Result<PH3ImageRequestLayout<'a>> {
        PH3ImageRequestLayout::new(
            transformer,
            &self.tags,
            self.target,
            self.reference.as_slice(),
        )
    }
}

/// Validate ingress without tokenizing or freezing downstream activation shapes.
pub(super) fn validate_request_metadata(
    request: &ImageGenerationRequest,
    config: &ImagePipelineConfig,
    spatial_scale: u64,
) -> Result<()> {
    ensure!(request.n == 1, "H3 requests require n=1");
    ensure!(
        request.prompt.len() <= 1 << 20,
        "prompt exceeds tokenizer byte capacity"
    );
    ensure!(
        request.width > 0
            && request.height > 0
            && request.width <= config.maximum_width
            && request.height <= config.maximum_height
            && spatial_scale > 0
            && u64::from(request.width) % spatial_scale == 0
            && u64::from(request.height) % spatial_scale == 0,
        "H3 dimensions exceed declared bounds or do not divide by the VAE scale"
    );
    Ok(())
}

/// A stable small basis per request: image time, audio time, and (for an edit)
/// clamped reference time. Their values are execution inputs, not cache keys.
pub(super) struct BatchGeometry<'a> {
    members: Vec<&'a RequestGeometry>,
    indices: Vec<Vec<u64>>,
    pub positions: Vec<[f64; 3]>,
    pub basis_rows: usize,
}
impl<'a> BatchGeometry<'a> {
    pub(super) fn request_count(&self) -> usize {
        self.members.len()
    }

    pub fn new(
        transformer: &PH3ImageTransformerConfig,
        members: Vec<&'a RequestGeometry>,
        maximum_batch: usize,
        maximum_metadata_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            !members.is_empty() && members.len() <= maximum_batch,
            "H3 physical membership exceeds batch capacity"
        );
        ensure!(
            members
                .iter()
                .map(|m| &m.identity)
                .collect::<BTreeSet<_>>()
                .len()
                == members.len(),
            "H3 physical membership contains duplicate request identities"
        );
        let rows = members.iter().try_fold(0usize, |sum, m| {
            sum.checked_add(m.layout(transformer)?.rows())
                .context("H3 packed rows overflow")
        })?;
        let bytes = rows
            .checked_mul(std::mem::size_of::<[f64; 3]>() + std::mem::size_of::<u64>())
            .context("H3 position/time metadata extent overflows")?;
        ensure!(
            bytes <= maximum_metadata_bytes && bytes <= isize::MAX as usize,
            "H3 packed position/time metadata exceeds capacity"
        );
        let mut positions = Vec::with_capacity(rows);
        let mut indices = Vec::with_capacity(members.len());
        let mut basis_rows = 0usize;
        for member in &members {
            let layout = member.layout(transformer)?;
            let start = positions.len();
            positions.resize(start + layout.rows(), [0.0; 3]);
            layout.write_positions(&mut positions[start..])?;
            let mut mapping = vec![basis_rows as u64; layout.rows()];
            mapping[layout.audio_rows()].fill((basis_rows + 1) as u64);
            if member.reference.is_some() {
                mapping[layout.reference_rows()].fill((basis_rows + 2) as u64);
            }
            basis_rows = basis_rows
                .checked_add(if member.reference.is_some() { 3 } else { 2 })
                .context("H3 time basis extent overflows")?;
            indices.push(mapping);
        }
        Ok(Self {
            members,
            indices,
            positions,
            basis_rows,
        })
    }
    pub fn prepare(
        &self,
        adapter: &PH3ImageDenoiserAdapter,
        phase: PH3ImageDenoiserPhase,
    ) -> Result<PreparedPH3ImageDenoiser> {
        self.prepare_with_family(adapter, phase, |provider, config, profile| {
            use ferrum_runtime::{ModelFamilyRegistration, TypedFamilyRegistration};
            Ok(TypedFamilyRegistration::new(provider)
                .prepare_with_profile(&serde_json::to_value(config)?, profile)?)
        })
    }
    pub(super) fn prepare_with_family(
        &self,
        adapter: &PH3ImageDenoiserAdapter,
        phase: PH3ImageDenoiserPhase,
        prepare_family: impl FnOnce(
            PH3ImageDenoiserFamilyProvider,
            &PH3ImageDenoiserFamilyConfig,
            &NumericalProfileId,
        ) -> Result<PreparedModelFamily>,
    ) -> Result<PreparedPH3ImageDenoiser> {
        let requests = self
            .members
            .iter()
            .zip(&self.indices)
            .map(|(member, indices)| {
                adapter.request(
                    member.identity.clone(),
                    &member.tags,
                    member.target,
                    member.reference,
                    indices,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        adapter.prepare_with_family(
            phase,
            u64::try_from(self.basis_rows)?,
            requests,
            prepare_family,
        )
    }
    /// Check cursor identities/order before model input allocation. The live
    /// request owner remains responsible for revision and atomic step commit;
    /// image/audio share the declared deterministic grid.
    pub fn times(
        &self,
        steps: &[(&RequestIdentity, PH3ImageSamplingStep)],
        augmentation: f64,
    ) -> Result<Vec<f32>> {
        ensure!(
            steps.len() == self.members.len()
                && steps
                    .iter()
                    .zip(&self.members)
                    .all(|((id, _), member)| *id == &member.identity),
            "H3 live cursors differ from compiled physical membership"
        );
        ensure!(
            augmentation.is_finite() && (0.0..=1.0).contains(&augmentation),
            "H3 reference augmentation must lie in [0,1]"
        );
        let mut times = Vec::with_capacity(self.basis_rows);
        for ((_, step), member) in steps.iter().zip(&self.members) {
            times.extend([step.timestep(), step.timestep()]);
            if member.reference.is_some() {
                // Comfy compares Python scalars, then constructs the F32
                // curve-input tensor. Solver/model timesteps remain F32.
                times.push(f64::from(step.timestep()).max(augmentation) as f32);
            }
        }
        Ok(times)
    }
}

impl PH3ImageAssets {
    pub(super) fn text_request_geometry(
        &self,
        request: &ImageGenerationRequest,
        identity: RequestIdentity,
        config: &ImagePipelineConfig,
        valid_tokens: u32,
    ) -> Result<RequestGeometry> {
        RequestGeometry::new(
            request,
            identity,
            config,
            self.spatial_scale(),
            self.denoiser.transformer(),
            vec![PH3ImageModality::Text; valid_tokens as usize],
            None,
        )
    }
}

#[cfg(test)]
mod tests;
