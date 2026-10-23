//! Managed single-frame adapters for the Qwen-Image 2.1 component boundary.
//!
//! The latent layout and staged BF16 decode arithmetic follow vLLM-Omni
//! c0551d22674d0db81305c800b66ad83035fdc980, `pipeline_qwen_image_21.py`
//! (`_pack_latents` / `_decode_latents`). Pixel conversion preserves the two
//! BF16 boundaries in the v0.40.0 diffusers image processor's denormalization.
//! These are local arithmetic contracts, not a claim of full Torch parity.
//!
//! Every owned payload comes from the caller's Host account. Conversions retain
//! their source while admitting and completing a new InputPreprocess backing.
//! The account measures backing bytes, not metadata, caller writer buffers, PNG
//! codec storage, or process RSS. This module does not create files or runtimes.

use super::QwenImage21VaeDecoderFamily;
use anyhow::{ensure, Context, Result};
use ferrum_diffusion::{pack_spatial_into, unpack_spatial_into, SingleFrameLatentShape};
use ferrum_execution::TensorExecutionOutput;
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, ManagedHostBytes, ManagedHostTensor,
    ProgramValueId, ResolvedTensorLayout, ResolvedTensorSpec,
};
use half::bf16;
use serde_json::Value;
use std::io::Write;

pub const QWEN_IMAGE21_LATENT_CHANNELS: usize = 64;
pub const QWEN_IMAGE21_LATENT_DECODE_PROFILE: &str = "qwen_image21.latent.bf16.mul_b.add_b.v1";
/// Fixed Omni source casts both statistics to BF16, subtracts, then divides;
/// it does not materialize a BF16 reciprocal or substitute reciprocal multiply.
pub const QWEN_IMAGE21_LATENT_ENCODE_PROFILE: &str = "qwen_image21.latent.bf16.sub_b.div_b.v1";
pub const QWEN_IMAGE21_RGBA8_PROFILE: &str =
    "qwen_image21.rgba8.clamp.bf16_half.bf16_shift.f32_scale.rne.v1";

/// Statistics are consumed from the supplied VAE metadata, without model-name
/// dispatch. Values are converted F64 -> F32 -> BF16 before being retained.
#[derive(Clone, Debug)]
pub struct QwenImage21LatentStatistics {
    mean: [bf16; QWEN_IMAGE21_LATENT_CHANNELS],
    std: [bf16; QWEN_IMAGE21_LATENT_CHANNELS],
}

impl QwenImage21LatentStatistics {
    pub fn from_vae_config(raw: &Value) -> Result<Self> {
        fn parse(raw: &Value, field: &str, positive: bool) -> Result<[bf16; 64]> {
            let values = raw[field]
                .as_array()
                .with_context(|| format!("missing numeric array {field}"))?;
            ensure!(
                values.len() == 64,
                "{field} must contain exactly 64 channels"
            );
            let mut result = [bf16::ZERO; 64];
            for (index, value) in values.iter().enumerate() {
                let f = value.as_f64().context("latent statistic must be numeric")? as f32;
                let b = bf16::from_f32(f);
                ensure!(
                    f.is_finite() && b.is_finite() && (!positive || b.to_f32() > 0.0),
                    "invalid finite/positive BF16 {field} at channel {index}"
                );
                result[index] = b;
            }
            Ok(result)
        }
        Ok(Self {
            mean: parse(raw, "latents_mean", false)?,
            std: parse(raw, "latents_std", true)?,
        })
    }

    pub fn mean(&self) -> &[bf16; 64] {
        &self.mean
    }
    pub fn std(&self) -> &[bf16; 64] {
        &self.std
    }
}

/// Packed means the actual DiT port `[B * H * W, 64]`, not an arbitrary tensor
/// with equal element count. NCHW has exactly four axes and one implicit frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenImage21LatentLayout {
    Packed,
    Nchw,
}

/// A caller's semantic declaration, not a property inferable from finite bytes.
/// Layout changes preserve it; VAE denormalization accepts only Normalized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QwenImage21LatentDomain {
    Normalized,
    Denormalized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QwenImage21LatentShape(SingleFrameLatentShape);

impl QwenImage21LatentShape {
    pub fn new(batch: usize, height: usize, width: usize) -> Result<Self> {
        let shape = SingleFrameLatentShape::new(batch, 64, height, width)?;
        shape.byte_len::<[u8; 2]>()?;
        Ok(Self(shape))
    }

    pub fn nchw_dimensions(self) -> [usize; 4] {
        self.0.nchw_dims()
    }

    pub fn packed_dimensions(self) -> [usize; 2] {
        [self.0.element_count() / 64, 64]
    }

    /// Bytes of each source/destination backing. A conversion needs this much
    /// additional InputPreprocess capacity while the full source remains live.
    pub fn byte_len(self) -> usize {
        // Constructor has checked the element and BF16 byte address ranges.
        self.0.element_count() * 2
    }

    pub fn tensor(self, layout: QwenImage21LatentLayout) -> Result<ResolvedTensorSpec> {
        let dimensions = match layout {
            QwenImage21LatentLayout::Packed => self.packed_dimensions().to_vec(),
            QwenImage21LatentLayout::Nchw => self.nchw_dimensions().to_vec(),
        };
        Ok(ResolvedTensorSpec::new(
            dimensions
                .into_iter()
                .map(u64::try_from)
                .collect::<Result<_, _>>()?,
            ElementType::Bf16,
            ResolvedTensorLayout::Contiguous,
        )?)
    }
}

/// Validated owner of a finite latent. Cloning its managed bytes retains the
/// same physical charge. It is not proof of how a caller obtained its values.
#[derive(Clone)]
pub struct QwenImage21ManagedLatents {
    value_id: ProgramValueId,
    tensor: ResolvedTensorSpec,
    bytes: ManagedHostBytes,
    shape: QwenImage21LatentShape,
    layout: QwenImage21LatentLayout,
    domain: QwenImage21LatentDomain,
}

impl QwenImage21ManagedLatents {
    pub fn new(
        host: &HostMemoryAccount,
        input: ManagedHostTensor,
        shape: QwenImage21LatentShape,
        layout: QwenImage21LatentLayout,
        domain: QwenImage21LatentDomain,
    ) -> Result<Self> {
        Self::from_parts(
            host,
            input.value_id().clone(),
            input.tensor().clone(),
            input.bytes().clone(),
            shape,
            layout,
            domain,
            HostMemoryClass::InputPreprocess,
        )
    }

    pub fn from_output(
        host: &HostMemoryAccount,
        output: &TensorExecutionOutput,
        shape: QwenImage21LatentShape,
        layout: QwenImage21LatentLayout,
        domain: QwenImage21LatentDomain,
    ) -> Result<Self> {
        Self::from_parts(
            host,
            output.value_id().clone(),
            output.tensor().clone(),
            output.bytes().clone(),
            shape,
            layout,
            domain,
            HostMemoryClass::Output,
        )
    }

    fn from_parts(
        host: &HostMemoryAccount,
        value_id: ProgramValueId,
        tensor: ResolvedTensorSpec,
        bytes: ManagedHostBytes,
        shape: QwenImage21LatentShape,
        layout: QwenImage21LatentLayout,
        domain: QwenImage21LatentDomain,
        class: HostMemoryClass,
    ) -> Result<Self> {
        ensure!(
            host.owns(&bytes),
            "latent belongs to a different Host account"
        );
        ensure!(
            bytes.class() == class,
            "latent has the wrong Host memory class"
        );
        ensure!(
            tensor == shape.tensor(layout)?,
            "latent tensor differs from exact shape/layout"
        );
        ensure!(
            bytes.len() == shape.byte_len(),
            "latent byte length differs from shape"
        );
        validate_finite(bytes.bytes())?;
        Ok(Self {
            value_id,
            tensor,
            bytes,
            shape,
            layout,
            domain,
        })
    }

    pub fn value_id(&self) -> &ProgramValueId {
        &self.value_id
    }
    pub fn tensor(&self) -> &ResolvedTensorSpec {
        &self.tensor
    }
    pub fn bytes(&self) -> &ManagedHostBytes {
        &self.bytes
    }
    pub fn shape(&self) -> QwenImage21LatentShape {
        self.shape
    }
    pub fn layout(&self) -> QwenImage21LatentLayout {
        self.layout
    }
    pub fn domain(&self) -> QwenImage21LatentDomain {
        self.domain
    }

    /// Reuse an existing InputPreprocess backing without changing its value ID.
    /// An Output backing must first be copied through `repack` (even when the
    /// desired layout is unchanged), to account for the next input allocation.
    pub fn into_input_tensor(self) -> Result<ManagedHostTensor> {
        Ok(ManagedHostTensor::new(
            self.value_id,
            self.tensor,
            self.bytes,
        )?)
    }

    /// Always creates a new backing with an explicitly supplied destination ID.
    /// No arithmetic or domain conversion occurs. The source survives both
    /// admission and completion; even an unchanged layout consumes new capacity.
    pub fn repack(
        &self,
        host: &HostMemoryAccount,
        target_layout: QwenImage21LatentLayout,
        target_value_id: ProgramValueId,
    ) -> Result<Self> {
        ensure!(
            host.owns(&self.bytes),
            "latent belongs to a different Host account"
        );
        let tensor = self.shape.tensor(target_layout)?;
        let mut output =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, self.shape.byte_len())?;
        // [u8; 2] has alignment one: Host backing need not be u16-aligned.
        let source = bytemuck::try_cast_slice::<u8, [u8; 2]>(self.bytes.bytes())
            .map_err(|_| anyhow::anyhow!("invalid BF16 source byte extent"))?;
        let destination = bytemuck::try_cast_slice_mut::<u8, [u8; 2]>(output.as_mut())
            .map_err(|_| anyhow::anyhow!("invalid BF16 destination byte extent"))?;
        match (self.layout, target_layout) {
            (QwenImage21LatentLayout::Packed, QwenImage21LatentLayout::Nchw) => {
                unpack_spatial_into(self.shape.0, source, destination)?
            }
            (QwenImage21LatentLayout::Nchw, QwenImage21LatentLayout::Packed) => {
                pack_spatial_into(self.shape.0, source, destination)?
            }
            _ => destination.copy_from_slice(source),
        }
        Ok(Self {
            value_id: target_value_id,
            tensor,
            bytes: output.freeze(),
            shape: self.shape,
            layout: target_layout,
            domain: self.domain,
        })
    }

    /// Produce the decoder's NCHW input with two explicit BF16 boundaries:
    /// `B(B(x * std[channel]) + mean[channel])`. This is not fused arithmetic.
    /// The source must be Normalized; a second denormalization is rejected.
    /// A late nonfinite result drops the private destination and releases its
    /// charge. No partially transformed output is published.
    pub fn denormalize_for_vae(
        &self,
        host: &HostMemoryAccount,
        statistics: &QwenImage21LatentStatistics,
    ) -> Result<Self> {
        ensure!(
            host.owns(&self.bytes),
            "latent belongs to a different Host account"
        );
        ensure!(
            self.domain == QwenImage21LatentDomain::Normalized,
            "VAE denormalization requires Normalized latents"
        );
        let tensor = self.shape.tensor(QwenImage21LatentLayout::Nchw)?;
        let value_id = ProgramValueId::new(QwenImage21VaeDecoderFamily::INPUT_ID)?;
        let mut output =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, self.shape.byte_len())?;
        let [_, positions, _] = self.shape.0.sequence_dims();
        for (index, bytes) in self.bytes.bytes().chunks_exact(2).enumerate() {
            let (channel, destination) = match self.layout {
                QwenImage21LatentLayout::Nchw => ((index / positions) % 64, index),
                QwenImage21LatentLayout::Packed => {
                    let channel = index % 64;
                    let position = (index / 64) % positions;
                    let batch = index / (64 * positions);
                    (channel, (batch * 64 + channel) * positions + position)
                }
            };
            let input = decode(bytes);
            let product = bf16::from_f32(input.to_f32() * statistics.std[channel].to_f32());
            ensure!(
                product.is_finite(),
                "latent BF16 product overflow at source element {index}"
            );
            let result = bf16::from_f32(product.to_f32() + statistics.mean[channel].to_f32());
            ensure!(
                result.is_finite(),
                "latent BF16 sum overflow at source element {index}"
            );
            output.as_mut()[destination * 2..destination * 2 + 2]
                .copy_from_slice(&result.to_bits().to_le_bytes());
        }
        Ok(Self {
            value_id,
            tensor,
            bytes: output.freeze(),
            shape: self.shape,
            layout: QwenImage21LatentLayout::Nchw,
            domain: QwenImage21LatentDomain::Denormalized,
        })
    }

    /// Convert a finite VAE posterior-mode NCHW tensor to the normalized packed
    /// reference-image port. The exact staged arithmetic is
    /// `B(RN32(B(z - mean[channel]) / std[channel]))`, matching the BF16 tensor
    /// subtraction and division in pinned `_encode_vae_image` (lines 817–830).
    /// No reciprocal tensor or intermediate payload is created. The old backing
    /// remains live while the complete destination is admitted and populated.
    pub fn normalize_from_vae(
        &self,
        host: &HostMemoryAccount,
        statistics: &QwenImage21LatentStatistics,
        output_id: ProgramValueId,
    ) -> Result<Self> {
        ensure!(
            host.owns(&self.bytes),
            "latent belongs to a different Host account"
        );
        ensure!(
            self.domain == QwenImage21LatentDomain::Denormalized
                && self.layout == QwenImage21LatentLayout::Nchw,
            "VAE normalization requires Denormalized NCHW latents"
        );
        let tensor = self.shape.tensor(QwenImage21LatentLayout::Packed)?;
        let mut output =
            host.allocate_zeroed(HostMemoryClass::InputPreprocess, self.shape.byte_len())?;
        let [_, positions, _] = self.shape.0.sequence_dims();
        for (index, bytes) in self.bytes.bytes().chunks_exact(2).enumerate() {
            let channel = (index / positions) % 64;
            let position = index % positions;
            let batch = index / (64 * positions);
            let difference =
                bf16::from_f32(decode(bytes).to_f32() - statistics.mean[channel].to_f32());
            ensure!(
                difference.is_finite(),
                "latent BF16 difference overflow at source element {index}"
            );
            let normalized = bf16::from_f32(difference.to_f32() / statistics.std[channel].to_f32());
            ensure!(
                normalized.is_finite(),
                "latent BF16 quotient overflow at source element {index}"
            );
            let destination = ((batch * positions + position) * 64 + channel) * 2;
            output.as_mut()[destination..destination + 2]
                .copy_from_slice(&normalized.to_bits().to_le_bytes());
        }
        Ok(Self {
            value_id: output_id,
            tensor,
            bytes: output.freeze(),
            shape: self.shape,
            layout: QwenImage21LatentLayout::Packed,
            domain: QwenImage21LatentDomain::Normalized,
        })
    }
}

/// A borrowed, validated decoder output. RGBA stays RGBA: the alpha channel
/// receives the same conversion as RGB, without premultiplication or blending.
pub struct QwenImage21RgbaOutput<'a> {
    bytes: &'a ManagedHostBytes,
    batch: usize,
    height: usize,
    width: usize,
    positions: usize,
    frame_bytes: usize,
}

impl<'a> QwenImage21RgbaOutput<'a> {
    /// Checks the complete output for finite values before any writer is used.
    /// Only exact contiguous BF16 `[N,4,H,W]` Output backing is accepted.
    pub fn new(
        host: &HostMemoryAccount,
        tensor: &ResolvedTensorSpec,
        bytes: &'a ManagedHostBytes,
    ) -> Result<Self> {
        ensure!(
            host.owns(bytes),
            "RGBA output belongs to a different Host account"
        );
        ensure!(
            bytes.class() == HostMemoryClass::Output,
            "RGBA requires Output backing"
        );
        ensure!(
            tensor.element_type() == ElementType::Bf16
                && tensor.layout() == &ResolvedTensorLayout::Contiguous,
            "RGBA requires contiguous BF16"
        );
        let [batch, channels, height, width] = tensor.dimensions() else {
            anyhow::bail!("RGBA requires exactly NCHW dimensions");
        };
        ensure!(*channels == 4, "RGBA requires exactly four channels");
        let shape = SingleFrameLatentShape::new(
            usize::try_from(*batch)?,
            4,
            usize::try_from(*height)?,
            usize::try_from(*width)?,
        )?;
        let length = shape.byte_len::<[u8; 2]>()?;
        ensure!(
            bytes.len() == length,
            "RGBA byte length differs from its shape"
        );
        validate_finite(bytes.bytes())?;
        let [batch, _, height, width] = shape.nchw_dims();
        let positions = shape.sequence_dims()[1];
        Ok(Self {
            bytes,
            batch,
            height,
            width,
            positions,
            frame_bytes: positions * 4,
        })
    }

    pub fn dimensions(&self) -> [usize; 4] {
        [self.batch, 4, self.height, self.width]
    }
    pub fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    /// Write a single interleaved RGBA8 frame using caller-owned scratch.
    /// Invalid frame/scratch is rejected before writing. An I/O error may leave
    /// bytes in the writer: sink rollback is the caller's responsibility.
    /// Scratch is neither allocated nor claimed by this adapter's Host account.
    pub fn write_frame_rgba8<W: Write + ?Sized>(
        &self,
        frame: usize,
        scratch: &mut [u8],
        writer: &mut W,
    ) -> Result<()> {
        ensure!(frame < self.batch, "RGBA frame index exceeds batch");
        ensure!(!scratch.is_empty(), "RGBA writer scratch must be nonempty");
        let source = self.bytes.bytes();
        let frame_start = frame * self.positions * 4;
        let mut used = 0;
        for position in 0..self.positions {
            for channel in 0..4 {
                let index = frame_start + channel * self.positions + position;
                scratch[used] = rgba8_finite(decode(&source[index * 2..index * 2 + 2]));
                used += 1;
                if used == scratch.len() {
                    writer.write_all(&scratch[..used])?;
                    used = 0;
                }
            }
        }
        if used != 0 {
            writer.write_all(&scratch[..used])?;
        }
        Ok(())
    }
}

fn decode(bytes: &[u8]) -> bf16 {
    bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn validate_finite(bytes: &[u8]) -> Result<()> {
    for (index, value) in bytes.chunks_exact(2).enumerate() {
        ensure!(
            decode(value).is_finite(),
            "nonfinite BF16 payload at element {index}"
        );
    }
    Ok(())
}

fn rgba8_finite(value: bf16) -> u8 {
    let clamped = value.to_f32().clamp(-1.0, 1.0);
    let scaled = bf16::from_f32(clamped * 0.5);
    let shifted = bf16::from_f32(scaled.to_f32() + 0.5)
        .to_f32()
        .clamp(0.0, 1.0);
    (shifted * 255.0).round_ties_even() as u8
}

#[cfg(test)]
mod tests;
