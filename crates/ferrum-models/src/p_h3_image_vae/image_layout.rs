//! Managed single-frame layout boundaries for the explicit F32 decoder route.
//! ComfyUI 56f179: PH3ImageDecoder._token_ids, RotaryEmbeddingND, unpatchify.
//! This adapter neither samples a posterior nor changes the latent domain.
use super::latent_statistics::PH3ImageVaeLatentStatistics;
use super::PH3ImageVaeDecoderConfig;
use anyhow::{ensure, Context, Result};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostBytes, ManagedHostTensor,
    ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
};

#[derive(Clone, Debug)]
pub(crate) struct PH3ImageVaeImageLayout {
    config: PH3ImageVaeDecoderConfig,
    batch: usize,
    height: usize,
    width: usize,
    patches: usize,
    sequence: usize,
}
impl PH3ImageVaeImageLayout {
    /// A declared host trig profile, not a claim of Torch/CUDA bit equivalence.
    pub const ROTARY_PROFILE_ID: &'static str = "p_h3_image.vae.rotary.libm-0.2.15.f32_angle.v1";

    pub fn new(
        config: PH3ImageVaeDecoderConfig,
        batch: usize,
        latent_height: usize,
        latent_width: usize,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            batch > 0 && latent_height > 0 && latent_width > 0,
            "VAE image dimensions must be positive"
        );
        let patches = latent_height
            .checked_mul(latent_width)
            .context("VAE patch count overflows")?;
        let sequence =
            usize::try_from(config.sequence_length(latent_height as u64, latent_width as u64)?)?;
        let layout = Self {
            config,
            batch,
            height: latent_height,
            width: latent_width,
            patches,
            sequence,
        };
        // Validate all physical extents before any loop or backing allocation.
        layout.latent_spec()?;
        layout.row_spec()?;
        layout.frequency_spec()?;
        layout.patch_output_spec()?;
        layout.pixel_spec()?;
        Ok(layout)
    }

    pub fn sequence_length(&self) -> usize {
        self.sequence
    }
    pub fn patch_count(&self) -> usize {
        self.patches
    }
    pub fn latent_spec(&self) -> Result<ResolvedTensorSpec> {
        spec(&[
            self.batch,
            usize::try_from(self.config.latent_channels())?,
            self.height,
            self.width,
        ])
    }
    pub fn row_spec(&self) -> Result<ResolvedTensorSpec> {
        spec(&[
            product(self.batch, self.patches)?,
            usize::try_from(self.config.latent_channels())?,
        ])
    }
    pub fn frequency_spec(&self) -> Result<ResolvedTensorSpec> {
        spec(&[
            product(self.batch, self.sequence)?,
            usize::try_from(self.config.rotary_dim())?,
        ])
    }
    /// Full decoder output, including register and CLS rows; suffix rows never
    /// enter the returned pixel image.
    pub fn patch_output_spec(&self) -> Result<ResolvedTensorSpec> {
        spec(&[
            product(self.batch, self.sequence)?,
            usize::try_from(self.config.patch_output_channels()?)?,
        ])
    }
    pub fn pixel_spec(&self) -> Result<ResolvedTensorSpec> {
        let patch = usize::try_from(self.config.patch_size())?;
        spec(&[
            self.batch,
            usize::try_from(self.config.out_channels())?,
            product(self.height, patch)?,
            product(self.width, patch)?,
        ])
    }

    /// NCHW -> [B*H*W,C], preserving every F32 bit. The caller supplies already
    /// denormalized latents; no normalization or post-quant convolution occurs.
    pub fn pack_denormalized_latents(
        &self,
        host: &HostMemoryAccount,
        input: &ManagedHostTensor,
        output_id: ProgramValueId,
    ) -> Result<ManagedHostTensor> {
        self.pack_latents(host, input, output_id, None)
    }

    /// Apply original per-channel F32 statistics while transposing, using one
    /// admitted successor allocation. The normalized sampler state is retained
    /// unchanged, including on late invalid input or capacity failure.
    pub fn denormalize_and_pack(
        &self,
        host: &HostMemoryAccount,
        input: &ManagedHostTensor,
        output_id: ProgramValueId,
        statistics: &PH3ImageVaeLatentStatistics<'_>,
    ) -> Result<ManagedHostTensor> {
        ensure!(
            statistics.channels() as u64 == self.config.latent_channels(),
            "VAE statistics differ from latent channels"
        );
        self.pack_latents(host, input, output_id, Some(statistics))
    }

    fn pack_latents(
        &self,
        host: &HostMemoryAccount,
        input: &ManagedHostTensor,
        output_id: ProgramValueId,
        statistics: Option<&PH3ImageVaeLatentStatistics<'_>>,
    ) -> Result<ManagedHostTensor> {
        ensure!(
            input.tensor() == &self.latent_spec()?
                && host.owns(input.bytes())
                && input.bytes().class() == HostMemoryClass::InputPreprocess,
            "VAE latent input differs from its layout or Host owner"
        );
        let output_spec = self.row_spec()?;
        let mut output =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, extent(&output_spec)?)?;
        let channels = usize::try_from(self.config.latent_channels())?;
        for b in 0..self.batch {
            for p in 0..self.patches {
                for c in 0..channels {
                    let source = ((b * channels + c) * self.patches + p) * 4;
                    let destination = ((b * self.patches + p) * channels + c) * 4;
                    let raw = &input.bytes().bytes()[source..source + 4];
                    if let Some(statistics) = statistics {
                        let value = statistics
                            .denormalize(c, f32::from_le_bytes(raw.try_into().unwrap()))?;
                        output.as_mut()[destination..destination + 4]
                            .copy_from_slice(&value.to_le_bytes());
                    } else {
                        output.as_mut()[destination..destination + 4].copy_from_slice(raw);
                    }
                }
            }
        }
        Ok(ManagedHostTensor::new(
            output_id,
            output_spec,
            output.freeze(),
        )?)
    }

    /// Image time is the fixed -6/7 coordinate. Spatial coordinates follow the
    /// half-cell grid; register tokens and the final zero CLS use [0,0,0].
    fn position(&self, row: usize) -> [f32; 3] {
        if row >= self.patches {
            return [0.0; 3];
        }
        let grid = |index: usize, length: usize| {
            let centered = index as f32 + 0.5;
            let fraction = centered / length as f32;
            2.0 * fraction - 1.0
        };
        [
            (2.0_f64 * (0.5 / 7.0) - 1.0) as f32,
            grid(row / self.width, self.height),
            grid(row % self.width, self.width),
        ]
    }

    /// Both outputs are charged to the existing account and published together.
    /// The source VAE frequency buffer is nonpersistent: this named profile
    /// generates F32 inverse frequencies from the declared architecture theta.
    pub fn managed_rotary(
        &self,
        host: &HostMemoryAccount,
        cosine_id: ProgramValueId,
        sine_id: ProgramValueId,
    ) -> Result<(ManagedHostTensor, ManagedHostTensor)> {
        ensure!(cosine_id != sine_id, "VAE frequency identities must differ");
        let tensor = self.frequency_spec()?;
        let theta = (self.config.rope_theta().numerator() as f64
            / self.config.rope_theta().denominator() as f64) as f32;
        ensure!(
            theta.is_finite() && theta > 0.0,
            "VAE rotary theta must fit positive F32"
        );
        let width = usize::try_from(self.config.rotary_dim())?;
        let count = width / 6;
        // A scalar validation pass avoids an uncharged copied frequency table.
        let inverse = |f: usize| 1.0_f32 / libm::powf(theta, f as f32 / count as f32);
        ensure!(
            (0..count).all(|f| inverse(f).is_finite()),
            "VAE inverse frequency must be finite"
        );
        let mut cosine =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, extent(&tensor)?)?;
        let mut sine = host.allocate_zeroed(HostMemoryClass::InputPreprocess, extent(&tensor)?)?;
        for f in 0..count {
            let inverse = inverse(f);
            for row in 0..self.sequence {
                for (axis, coordinate) in self.position(row).into_iter().enumerate() {
                    let scaled = std::f32::consts::TAU * coordinate;
                    let angle = scaled * inverse;
                    ensure!(angle.is_finite(), "VAE rotary angle must be finite");
                    let c = libm::cosf(angle).to_le_bytes();
                    let s = libm::sinf(angle).to_le_bytes();
                    for b in 0..self.batch {
                        for half in [0, width / 2] {
                            let offset =
                                ((b * self.sequence + row) * width + axis * count + f + half) * 4;
                            cosine.as_mut()[offset..offset + 4].copy_from_slice(&c);
                            sine.as_mut()[offset..offset + 4].copy_from_slice(&s);
                        }
                    }
                }
            }
        }
        Ok((
            ManagedHostTensor::new(cosine_id, tensor.clone(), cosine.freeze())?,
            ManagedHostTensor::new(sine_id, tensor, sine.freeze())?,
        ))
    }

    /// Raw patch values -> raw NCHW pixels. ImageNet rescaling/clamping belongs
    /// to the final output adapter, never to patch placement. Only selected
    /// image rows must be finite; discarded register/CLS rows are not pixels.
    pub fn unpatchify(
        &self,
        host: &HostMemoryAccount,
        source_spec: &ResolvedTensorSpec,
        source: &ManagedHostBytes,
    ) -> Result<(ResolvedTensorSpec, ManagedHostBytes)> {
        self.unpatchify_with_profile(host, source_spec, source, false)
    }

    /// Pinned VAE pixel finalization: separate F32 `raw * std + mean`, then
    /// clamp to [0,1]. This performs placement and finalization in the same
    /// admitted Output backing, without an extra full-resolution image copy.
    pub fn unpatchify_rgb(
        &self,
        host: &HostMemoryAccount,
        source_spec: &ResolvedTensorSpec,
        source: &ManagedHostBytes,
    ) -> Result<(ResolvedTensorSpec, ManagedHostBytes)> {
        ensure!(
            self.config.out_channels() == 3,
            "ImageNet pixel finalization requires exactly RGB channels"
        );
        self.unpatchify_with_profile(host, source_spec, source, true)
    }

    fn unpatchify_with_profile(
        &self,
        host: &HostMemoryAccount,
        source_spec: &ResolvedTensorSpec,
        source: &ManagedHostBytes,
        finalize: bool,
    ) -> Result<(ResolvedTensorSpec, ManagedHostBytes)> {
        let expected = self.patch_output_spec()?;
        ensure!(
            source_spec == &expected
                && source.len() == extent(&expected)?
                && host.owns(source)
                && source.class() == HostMemoryClass::Output,
            "VAE patch output differs from its layout or Output owner"
        );
        let patch = usize::try_from(self.config.patch_size())?;
        let channels = usize::try_from(self.config.out_channels())?;
        let patch_width = channels * patch * patch;
        for b in 0..self.batch {
            let start = b * self.sequence * patch_width * 4;
            ensure!(
                source.bytes()[start..start + self.patches * patch_width * 4]
                    .chunks_exact(4)
                    .all(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()).is_finite()),
                "VAE image patch contains nonfinite pixels"
            );
        }
        let tensor = self.pixel_spec()?;
        let mut output = host.allocate_zeroed(HostMemoryClass::Output, extent(&tensor)?)?;
        let height = self.height * patch;
        let width = self.width * patch;
        for b in 0..self.batch {
            for y in 0..height {
                for x in 0..width {
                    let p = (y / patch) * self.width + x / patch;
                    for c in 0..channels {
                        let input = ((b * self.sequence + p) * patch_width
                            + (c * patch + y % patch) * patch
                            + x % patch)
                            * 4;
                        let out = (((b * channels + c) * height + y) * width + x) * 4;
                        let raw = &source.bytes()[input..input + 4];
                        if finalize {
                            let value = f32::from_le_bytes(raw.try_into().unwrap());
                            let scaled = value * [0.229_f32, 0.224, 0.225][c];
                            let shifted = scaled + [0.485_f32, 0.456, 0.406][c];
                            output.as_mut()[out..out + 4]
                                .copy_from_slice(&shifted.clamp(0.0, 1.0).to_le_bytes());
                        } else {
                            output.as_mut()[out..out + 4].copy_from_slice(raw);
                        }
                    }
                }
            }
        }
        Ok((tensor, output.freeze()))
    }
}
fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).context("VAE image extent overflows")
}
fn extent(tensor: &ResolvedTensorSpec) -> Result<usize> {
    let bytes = usize::try_from(tensor.minimum_storage_bytes()?)?;
    ensure!(
        bytes <= isize::MAX as usize,
        "VAE image exceeds Host address space"
    );
    Ok(bytes)
}
fn spec(dimensions: &[usize]) -> Result<ResolvedTensorSpec> {
    let tensor = ResolvedTensorSpec::new(
        dimensions.iter().map(|&d| d as u64).collect(),
        ElementType::F32,
        ResolvedTensorLayout::Contiguous,
    )?;
    extent(&tensor)?;
    Ok(tensor)
}

#[cfg(test)]
mod tests;
