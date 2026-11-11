//! Zero-copy views of one completed DiT output in its submitted request order.
//! The physical tensor remains the original patch/row matrix. Index mapping,
//! not a fabricated reshape, selects each request's NCHW velocity elements.
use super::*;
use crate::p_h3_image::{
    PH3ImageFinalLayerFamily, PH3ImageReferenceImage, PH3ImageTransformerConfig,
};
use ferrum_runtime::RequestIdentity;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct PH3ImageVelocityRequest {
    request: RequestIdentity,
    target: PH3ImageReferenceImage,
}
impl PH3ImageVelocityRequest {
    pub fn new(request: RequestIdentity, target: PH3ImageReferenceImage) -> Self {
        Self { request, target }
    }
    pub fn request(&self) -> &RequestIdentity {
        &self.request
    }
}

/// Borrowed submitted membership. The execution consumer must pair this layout
/// with that submission's compiled ports/output; equal matrix shapes alone do
/// not authenticate a different run's membership. No tensor or ledger is made.
pub struct PH3ImageVelocityBatchLayout<'a> {
    requests: &'a [PH3ImageVelocityRequest],
    channels: usize,
    audio_channels: usize,
    patch_h: usize,
    patch_w: usize,
    image_rows: usize,
    patch_width: usize,
}
impl<'a> PH3ImageVelocityBatchLayout<'a> {
    pub fn new(
        t: &PH3ImageTransformerConfig,
        requests: &'a [PH3ImageVelocityRequest],
        metadata_capacity: usize,
    ) -> Result<Self> {
        ensure!(
            !requests.is_empty(),
            "velocity layout requires submitted request membership"
        );
        let metadata = requests
            .len()
            .checked_mul(
                std::mem::size_of::<PH3ImageVelocityRequest>()
                    + std::mem::size_of::<&RequestIdentity>(),
            )
            .context("velocity metadata overflow")?;
        ensure!(
            metadata <= metadata_capacity,
            "velocity membership exceeds declared metadata capacity"
        );
        ensure!(
            requests
                .iter()
                .map(|r| &r.request)
                .collect::<BTreeSet<_>>()
                .len()
                == requests.len(),
            "velocity membership contains duplicate request identities"
        );
        let [pt, ph, pw] = t.patch_size();
        ensure!(pt == 1, "velocity layout requires one latent frame");
        let (channels, audio_channels, patch_h, patch_w) = (
            usize::try_from(t.in_channels())?,
            usize::try_from(t.audio_in_channels())?,
            usize::try_from(ph)?,
            usize::try_from(pw)?,
        );
        let patch_width = channels
            .checked_mul(patch_h)
            .and_then(|n| n.checked_mul(patch_w))
            .context("velocity patch width overflow")?;
        let image_rows = requests.iter().try_fold(0usize, |sum, r| {
            let (h, w) = (r.target.latent_height(), r.target.latent_width());
            ensure!(
                h % patch_h == 0 && w % patch_w == 0,
                "velocity extent must contain complete patches"
            );
            let rows = (h / patch_h)
                .checked_mul(w / patch_w)
                .context("velocity patch rows overflow")?;
            let elements = rows
                .checked_mul(patch_width)
                .context("velocity image extent overflow")?;
            ensure!(
                elements > 0 && elements <= isize::MAX as usize / 4,
                "velocity request exceeds Host address space"
            );
            sum.checked_add(rows)
                .context("velocity image rows overflow")
        })?;
        let total = image_rows
            .checked_mul(patch_width)
            .context("velocity batch extent overflow")?;
        let audio = requests
            .len()
            .checked_mul(4)
            .and_then(|n| n.checked_mul(audio_channels))
            .context("velocity audio extent overflow")?;
        ensure!(
            total <= isize::MAX as usize / 4 && audio <= isize::MAX as usize / 4,
            "velocity batch exceeds Host address space"
        );
        Ok(Self {
            requests,
            channels,
            audio_channels,
            patch_h,
            patch_w,
            image_rows,
            patch_width,
        })
    }
    pub fn image_output<'output>(
        &'output self,
        request: &RequestIdentity,
        declared: &CompiledTensorPort,
        output: &'output TensorExecutionOutput,
    ) -> Result<PH3ImageSamplingVelocity<'output>> {
        ensure!(
            declared.value_id().as_str() == PH3ImageFinalLayerFamily::IMAGE_VELOCITY_ID
                && declared.value_id() == output.value_id()
                && declared.tensor() == output.tensor(),
            "image velocity differs from submitted compiled output port"
        );
        self.image_storage(request, output.tensor(), output.bytes())
    }
    pub fn audio_output<'output>(
        &'output self,
        request: &RequestIdentity,
        declared: &CompiledTensorPort,
        output: &'output TensorExecutionOutput,
    ) -> Result<PH3ImageSamplingVelocity<'output>> {
        ensure!(
            declared.value_id().as_str() == PH3ImageFinalLayerFamily::AUDIO_VELOCITY_ID
                && declared.value_id() == output.value_id()
                && declared.tensor() == output.tensor(),
            "audio velocity differs from submitted compiled output port"
        );
        self.audio_storage(request, output.tensor(), output.bytes())
    }
    fn image_storage<'output>(
        &'output self,
        request: &RequestIdentity,
        tensor: &'output ResolvedTensorSpec,
        bytes: &'output ManagedHostBytes,
    ) -> Result<PH3ImageSamplingVelocity<'output>> {
        validate_storage(
            tensor,
            bytes,
            &[self.image_rows as u64, self.patch_width as u64],
        )?;
        let mut base = 0usize;
        for member in self.requests {
            let h = member.target.latent_height();
            let w = member.target.latent_width();
            let rows = (h / self.patch_h) * (w / self.patch_w);
            if &member.request == request {
                return Ok(PH3ImageSamplingVelocity {
                    tensor,
                    bytes,
                    mapping: VelocityMapping::Image {
                        request: &member.request,
                        base,
                        channels: self.channels,
                        height: h,
                        width: w,
                        patch_h: self.patch_h,
                        patch_w: self.patch_w,
                        patch_width: self.patch_width,
                    },
                });
            }
            base = base
                .checked_add(rows)
                .context("velocity request offset overflow")?;
        }
        anyhow::bail!("request is absent from submitted image velocity membership")
    }
    fn audio_storage<'output>(
        &'output self,
        request: &RequestIdentity,
        tensor: &'output ResolvedTensorSpec,
        bytes: &'output ManagedHostBytes,
    ) -> Result<PH3ImageSamplingVelocity<'output>> {
        validate_storage(
            tensor,
            bytes,
            &[(self.requests.len() * 4) as u64, self.audio_channels as u64],
        )?;
        let (index, member) = self
            .requests
            .iter()
            .enumerate()
            .find(|(_, r)| &r.request == request)
            .context("request is absent from submitted audio velocity membership")?;
        Ok(PH3ImageSamplingVelocity {
            tensor,
            bytes,
            mapping: VelocityMapping::Audio {
                request: &member.request,
                base: index * 4,
                channels: self.audio_channels,
            },
        })
    }
}
fn validate_storage(
    tensor: &ResolvedTensorSpec,
    bytes: &ManagedHostBytes,
    shape: &[u64],
) -> Result<()> {
    ensure!(
        tensor.dimensions() == shape
            && tensor.layout() == &ResolvedTensorLayout::Contiguous
            && matches!(tensor.element_type(), ElementType::F32 | ElementType::Bf16)
            && bytes.class() == HostMemoryClass::Output
            && tensor.minimum_storage_bytes()? == u64::try_from(bytes.len())?,
        "velocity physical output shape, dtype, layout, class or byte extent differs"
    );
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) enum VelocityMapping<'a> {
    Contiguous,
    Image {
        request: &'a RequestIdentity,
        base: usize,
        channels: usize,
        height: usize,
        width: usize,
        patch_h: usize,
        patch_w: usize,
        patch_width: usize,
    },
    Audio {
        request: &'a RequestIdentity,
        base: usize,
        channels: usize,
    },
}
impl VelocityMapping<'_> {
    pub(super) fn validate(
        &self,
        name: &str,
        request: Option<&RequestIdentity>,
        sample: &ResolvedTensorSpec,
    ) -> Result<()> {
        match *self {
            Self::Contiguous => ensure!(
                request.is_none(),
                "request-bound sampling requires an identity-bound velocity view"
            ),
            Self::Image {
                request: member,
                channels,
                height,
                width,
                ..
            } => {
                ensure!(
                    name == "image" && request == Some(member),
                    "image velocity request identity or stream differs"
                );
                let shape = sample.dimensions();
                ensure!(
                    shape == [1, channels as u64, height as u64, width as u64]
                        || shape == [1, channels as u64, 1, height as u64, width as u64],
                    "image velocity map differs from request's NCHW or T=1 sample shape"
                );
            }
            Self::Audio {
                request: member,
                channels,
                ..
            } => ensure!(
                name == "audio"
                    && request == Some(member)
                    && sample.dimensions() == [1, channels as u64, 2, 2],
                "audio velocity map differs from request identity, stream or stereo NCHW shape"
            ),
        }
        Ok(())
    }
    pub(super) fn source_element(self, index: usize) -> usize {
        match self {
            Self::Contiguous => index,
            Self::Image {
                base,
                height,
                width,
                patch_h,
                patch_w,
                patch_width,
                ..
            } => {
                let channel = index / (height * width);
                let position = index % (height * width);
                let y = position / width;
                let x = position % width;
                let row = (y / patch_h) * (width / patch_w) + x / patch_w;
                let column = (channel * patch_h + y % patch_h) * patch_w + x % patch_w;
                (base + row) * patch_width + column
            }
            Self::Audio { base, channels, .. } => (base + index % 4) * channels + index / 4,
        }
    }
}

#[cfg(test)]
mod tests;
